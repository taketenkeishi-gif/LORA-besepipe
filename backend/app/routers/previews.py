from __future__ import annotations

import base64
import json
import mimetypes
import re
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ..db import get_conn
from ..training.runtime import retry_preview_for_checkpoint
from ..training.runtime.preview_jobs import (
    backfill_legacy_preview_jobs,
    dispatch_pending_preview_jobs,
    ensure_preview_wait_monitor,
    job_summary,
    missing_jobs_for_checkpoint,
    retry_jobs,
    sync_checkpoint_status,
    _preview_snapshot_json,
)

router = APIRouter(prefix="/previews", tags=["previews"])


@router.get('/comfy-loras')
def comfy_loras():
    from .. import desktop_comfy
    catalog=desktop_comfy.inspect({})
    return {'loras':catalog['loras'],'url':catalog['url'],'devices':catalog['devices']}


@router.post('/epoch-window/{run_id}')
def epoch_window(run_id:int,payload:dict):
    from ..training.runtime.comfy_epoch_hook import active_window
    from ..training.runtime.state import run_dir
    from ..training.runtime.monitor import scan_and_register_artifacts
    from ..training.runtime.stages import set_training_stage
    import requests
    window=active_window(run_dir(run_id))
    if not window or window['request_id']!=payload.get('request_id'):
        return {'state':'rejected','message':'学習の保存待機区間を確認できません'}
    conn=get_conn()
    try:
        run=conn.execute('SELECT project_id,status FROM training_runs WHERE id=?',(run_id,)).fetchone()
        if not run or run['status']!='training':return {'state':'rejected','message':'学習Runが実行状態ではありません'}
        pid=int(run['project_id'])
    finally:conn.close()
    scan_and_register_artifacts(run_id,pid,dispatch=False)
    ensure_preview_wait_monitor()
    conn=get_conn()
    try:
        jobs=conn.execute('SELECT status,error_detail FROM preview_jobs WHERE run_id=? AND epoch=?',(run_id,window['epoch'])).fetchall()
        if not jobs or any(j['status'] in ('pending','running') for j in jobs):
            set_training_stage(run_id,'preview','running','ComfyUIでエポック画像を生成中・学習計算は待機')
            return {'state':'waiting','epoch':window['epoch']}
        from ..desktop_comfy import settings
        try:
            queue=requests.get(settings()['url']+'/queue',timeout=5).json()
            if queue.get('queue_running') or queue.get('queue_pending'):return {'state':'waiting','message':'ComfyUIキュー完了後に学習を継続'}
        except requests.RequestException:return {'state':'waiting','message':'ComfyUIの完了状態を確認中'}
        set_training_stage(run_id,'training','running','ComfyUIプレビュー完了・学習継続')
        return {'state':'done','epoch':window['epoch'],'succeeded':sum(j['status']=='succeeded' for j in jobs),'failed':sum(j['status']=='failed' for j in jobs)}
    finally:conn.close()


from pydantic import BaseModel,Field
class ExtraPreviewLora(BaseModel):
    name:str=Field(min_length=1,max_length=1000)
    strength:float=Field(default=1,ge=-2,le=2)
class ComfyPreviewSettings(BaseModel):
    profile_id:int|None=None
    base_checkpoint_path:str=''
    lora_strength:float=Field(default=1,ge=0,le=2)
    extra_loras:list[ExtraPreviewLora]=Field(default_factory=list,max_length=8)
    steps:int|None=Field(default=None,ge=1,le=100)
    cfg:float|None=Field(default=None,ge=0,le=30)


@router.post('/render-checkpoint/{checkpoint_id}')
def render_checkpoint(checkpoint_id:int,payload:ComfyPreviewSettings):
    from ..training.runtime.artifact_validation import validate_artifact
    conn=get_conn()
    try:
        checkpoint=conn.execute('SELECT * FROM checkpoints WHERE id=?',(checkpoint_id,)).fetchone()
        if not checkpoint:raise HTTPException(404,'保存済みLoRAがありません')
        run=conn.execute('SELECT config_json FROM training_runs WHERE id=?',(checkpoint['run_id'],)).fetchone();cfg=json.loads(run['config_json'])
        family=cfg.get('model_family','anima')
        ok,detail=validate_artifact(checkpoint['file_path'],family)
        if not ok:raise HTTPException(409,detail)
        profile_id=payload.profile_id or cfg.get('preview_profile_snapshot_id')
        profile=conn.execute('SELECT payload_json FROM basepipe_preview_profile_snapshots WHERE id=? AND project_id=?',(profile_id,checkpoint['project_id'])).fetchone()
        if not profile:raise HTTPException(400,'プレビュー条件を保存してください')
        p=json.loads(profile['payload_json'])
        jobs=conn.execute('SELECT * FROM preview_jobs WHERE checkpoint_id=? ORDER BY prompt_index,instance_index',(checkpoint_id,)).fetchall()
        if not jobs:raise HTTPException(400,'この保存結果のプレビュー記録がありません')
        if any(j['status'] in ('running','pending') for j in jobs):raise HTTPException(409,'このエポックの生成が実行中または待機中です')
        conditions={'backend':'comfyui','source':'comfyui_user_preview','model_family':family,'resolution':int(p.get('resolution') or 1024),'width':int(p.get('width') or p.get('resolution') or 1024),'height':int(p.get('height') or p.get('resolution') or 1024),'sampler':p.get('sampler') or 'euler','scheduler':p.get('scheduler') or 'simple','steps':payload.steps if payload.steps is not None else p.get('steps',20),'cfg':payload.cfg if payload.cfg is not None else p.get('cfg',5),'lora_strength':payload.lora_strength,'extra_loras':[r.model_dump() for r in payload.extra_loras],'preview_base_checkpoint_path':payload.base_checkpoint_path or cfg['base_checkpoint_path']}
        # Freeze old image paths/conditions before putting the same logical job back in queue.
        for job in jobs:
            if job['status']=='succeeded' and job['output_path']:
                conn.execute('INSERT INTO preview_history(checkpoint_id,job_id,slot,image_path,preview_snapshot_json) VALUES(?,?,?,?,?)',(checkpoint_id,job['id'],f"p{job['prompt_index']}_i{job['instance_index']}",job['output_path'],job['preview_snapshot_json']))
            snapshot=_preview_snapshot_json(conn,project_id=checkpoint['project_id'],checkpoint_id=checkpoint_id,run_id=checkpoint['run_id'],epoch=checkpoint['epoch'],prompt=p['prompt'],negative_prompt=p.get('negative_prompt',''),seed=int(p.get('seed',42)),conditions_json=json.dumps(conditions),source='comfyui_user_preview',profile_snapshot_id=profile_id)
            conn.execute("UPDATE preview_jobs SET status='pending',error_detail='',output_path='',prompt=?,negative_prompt=?,seed=?,conditions_json=?,preview_snapshot_json=? WHERE id=?",(p['prompt'],p.get('negative_prompt',''),int(p.get('seed',42)),json.dumps(conditions,ensure_ascii=False),snapshot,job['id']))
        conn.commit();sync_checkpoint_status(conn,checkpoint_id)
    finally:conn.close()
    ensure_preview_wait_monitor()
    return {'checkpoint_id':checkpoint_id,'status':'pending','backend':'comfyui','conditions':conditions}


@router.get('/history/{checkpoint_id}')
def history(checkpoint_id:int):
    conn=get_conn()
    try:rows=conn.execute('SELECT id,preview_snapshot_json,created_at FROM preview_history WHERE checkpoint_id=? ORDER BY id DESC',(checkpoint_id,)).fetchall()
    finally:conn.close()
    return {'images':[{'id':r['id'],'image_url':f"/api/previews/history-image/{r['id']}",'settings':json.loads(r['preview_snapshot_json'] or '{}'),'created_at':r['created_at']} for r in rows]}


@router.get('/history-image/{history_id}')
def history_image(history_id:int):
    conn=get_conn()
    try:row=conn.execute('SELECT image_path FROM preview_history WHERE id=?',(history_id,)).fetchone()
    finally:conn.close()
    if not row or not Path(row['image_path']).is_file():raise HTTPException(404,'履歴画像がありません')
    return FileResponse(row['image_path'])


@router.get("/image/{checkpoint_id}/{slot}")
def preview_image(checkpoint_id: int, slot: str):
    """Serve one registered Preview image without embedding it in JSON.

    The database path is authoritative; the client never supplies a filesystem
    path. This keeps the timeline response small while preserving the existing
    base64 response mode for API consumers that depend on it.
    """
    conn = get_conn()
    match = re.fullmatch(r"p(\d+)_i(\d+)", slot)
    row = None
    if match:
        row = conn.execute(
            "SELECT output_path AS image_path FROM preview_jobs "
            "WHERE checkpoint_id = ? AND prompt_index = ? AND instance_index = ? "
            "AND status = 'succeeded' ORDER BY id DESC LIMIT 1",
            (checkpoint_id, int(match.group(1)), int(match.group(2))),
        ).fetchone()
    if row is None:
        row = conn.execute(
            "SELECT image_path FROM preview_samples WHERE checkpoint_id = ? AND slot = ? "
            "ORDER BY id DESC LIMIT 1",
            (checkpoint_id, slot),
        ).fetchone()
    conn.close()
    if row is None:
        raise HTTPException(status_code=404, detail="preview image not found")
    path = Path(str(row["image_path"]))
    if not path.is_file() or path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise HTTPException(status_code=404, detail="preview image file not found")
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return FileResponse(path, media_type=media_type, filename=path.name)


@router.post("/refresh-profile/{checkpoint_id}/{profile_id}")
def refresh_preview_with_profile(checkpoint_id: int, profile_id: int) -> dict:
    """既存画像を履歴へ退避し、指定Frozen ProfileでPreviewだけを再生成する。

    Training Runは触らない。成功済みJobの旧画像はpreview_historyへ保存し、
    新しい出力はprofile snapshot suffix付きの別ファイルへ書き出す。
    """
    conn = get_conn()
    checkpoint = conn.execute(
        "SELECT id, project_id, run_id, epoch FROM checkpoints WHERE id = ?", (checkpoint_id,)
    ).fetchone()
    if checkpoint is None:
        conn.close()
        raise HTTPException(status_code=404, detail="checkpoint not found")
    accepted_asset = conn.execute(
        "SELECT id FROM lora_assets WHERE training_run_id = ? AND library_status = 'accepted' LIMIT 1",
        (checkpoint["run_id"],),
    ).fetchone()
    if accepted_asset is not None:
        conn.close()
        raise HTTPException(
            status_code=409,
            detail="Final採用済みCheckpointのPreview条件は変更できません。新しいRunで比較してください",
        )
    profile = conn.execute(
        "SELECT id, project_id, payload_json FROM basepipe_preview_profile_snapshots "
        "WHERE id = ? AND project_id = ?", (profile_id, checkpoint["project_id"])
    ).fetchone()
    if profile is None:
        conn.close()
        raise HTTPException(status_code=404, detail="Frozen Preview Profile not found")
    try:
        payload = json.loads(profile["payload_json"] or "{}")
    except (TypeError, ValueError):
        conn.close()
        raise HTTPException(status_code=400, detail="Frozen Preview Profile payload is invalid")
    jobs = conn.execute(
        "SELECT * FROM preview_jobs WHERE checkpoint_id = ? ORDER BY prompt_index, instance_index",
        (checkpoint_id,),
    ).fetchall()
    if not jobs:
        conn.close()
        raise HTTPException(status_code=400, detail="Preview Jobがありません")
    profile_prompt = str(payload.get("prompt") or "")
    profile_negative = str(payload.get("negative_prompt") or "")
    profile_seed = payload.get("seed")
    conditions = {
        "model_family": str(payload.get("model_family") or jobs[0]["preview_model_family"]),
        "resolution": int(payload.get("resolution") or 1024),
        "sampler": str(payload.get("sampler") or "euler"),
        "scheduler": str(payload.get("scheduler") or "simple"),
        "cfg": float(payload.get("cfg") or 1.0),
        "steps": int(payload.get("steps") or 7),
        "source": "frozen_profile",
    }
    conditions_json = json.dumps(conditions, ensure_ascii=False, sort_keys=True)
    pending = 0
    for job in jobs:
        if job["status"] == "succeeded" and job["output_path"] and Path(job["output_path"]).is_file():
            conn.execute(
                "INSERT INTO preview_history(checkpoint_id, job_id, slot, image_path, preview_snapshot_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (checkpoint_id, job["id"], f"p{job['prompt_index']}_i{job['instance_index']}", job["output_path"], job["preview_snapshot_json"] or "{}"),
            )
        prompt = profile_prompt or str(job["prompt"] or "")
        negative_prompt = profile_negative or str(job["negative_prompt"] or "")
        seed = int(profile_seed) if profile_seed is not None else int(job["seed"])
        snapshot_json = _preview_snapshot_json(
            conn, project_id=int(checkpoint["project_id"]), checkpoint_id=checkpoint_id,
            run_id=int(checkpoint["run_id"]), epoch=int(checkpoint["epoch"]),
            prompt=prompt, negative_prompt=negative_prompt, seed=seed,
            conditions_json=conditions_json, source="frozen_profile", profile_snapshot_id=profile_id,
        )
        conn.execute(
            "UPDATE preview_jobs SET status='pending', attempts=0, error_detail='', output_path='', "
            "prompt=?, negative_prompt=?, seed=?, preview_model_family=?, conditions_json=?, "
            "preview_snapshot_json=?, started_at=NULL, completed_at=NULL WHERE id=?",
            (prompt, negative_prompt, seed, conditions["model_family"], conditions_json, snapshot_json, job["id"]),
        )
        pending += 1
    # A profile refresh is a new review generation. Old human scores must not
    # be silently rebound to the new immutable Preview Snapshot.
    conn.execute("DELETE FROM basepipe_evaluations WHERE checkpoint_id = ?", (checkpoint_id,))
    conn.commit()
    result = dispatch_pending_preview_jobs(conn, max_jobs=pending)
    ensure_preview_wait_monitor()
    status = sync_checkpoint_status(conn, checkpoint_id)
    conn.close()
    return {"checkpoint_id": checkpoint_id, "profile_snapshot_id": profile_id, "dispatch_result": result, "checkpoint_status": status, "history_preserved": True}


@router.post("/retry/{checkpoint_id}")
def retry_preview(checkpoint_id: int) -> dict:
    """このcheckpointの failed/missing な Preview Job のみを再試行する
    (成功済みJobは対象に含めない)。

    Preview Job(preview_jobs)がSSOTとして存在する場合はJob単位のRetryを
    優先する。Job管理下にない旧来のcheckpoint(Preview Job未作成)は、
    従来通りcheckpoint単位のretry_preview_for_checkpointへフォールバックする
    (後方互換)。
    """
    conn = get_conn()
    ckpt = conn.execute("SELECT id FROM checkpoints WHERE id = ?", (checkpoint_id,)).fetchone()
    if ckpt is None:
        conn.close()
        raise HTTPException(status_code=404, detail=f"checkpoint not found: {checkpoint_id}")

    summary = job_summary(conn, checkpoint_id)
    if summary["expected"] == 0:
        # 旧checkpointはpreview_samplesだけを持つ場合がある。実画像を再生成せず、
        # Job SSOTへ補修してから通常のmissing/retry判定へ進む。
        backfill_legacy_preview_jobs(conn, checkpoint_id)
        summary = job_summary(conn, checkpoint_id)
    if summary["expected"] > 0:
        targets = missing_jobs_for_checkpoint(conn, checkpoint_id)
        target_ids = [t["id"] for t in targets if t["status"] == "failed"]
        # output消失分もpendingへ戻す(missing_jobs_for_checkpointがsucceeded+消失も含むため)
        missing_output_ids = [t["id"] for t in targets if t["status"] == "succeeded"]
        for jid in missing_output_ids:
            conn.execute("UPDATE preview_jobs SET status='pending', error_detail='' WHERE id=?", (jid,))
        if target_ids:
            retry_jobs(conn, target_ids)
        conn.commit()
        result = dispatch_pending_preview_jobs(conn, max_jobs=len(target_ids) + len(missing_output_ids) or None)
        ensure_preview_wait_monitor()
        status = sync_checkpoint_status(conn, checkpoint_id)
        conn.close()
        succeeded = status == "preview_succeeded"
        return {
            "checkpoint_id": checkpoint_id,
            "succeeded": succeeded,
            "detail": f"retried {len(target_ids) + len(missing_output_ids)} job(s), result={result}",
            "message": "Preview再生成完了" if succeeded else f"Preview一部再生成: {status}",
        }

    conn.close()
    succeeded, detail = retry_preview_for_checkpoint(checkpoint_id)
    return {
        "checkpoint_id": checkpoint_id,
        "succeeded": succeeded,
        "detail": detail,
        "message": "Preview再生成完了" if succeeded else f"Preview再生成失敗: {detail}",
    }


@router.post("/jobs/{job_id}/retry")
def retry_single_job(job_id: int) -> dict:
    """1件のPreview Jobのみを再試行する(成功済みJobには使えない)。"""
    conn = get_conn()
    job = conn.execute("SELECT id, status, checkpoint_id FROM preview_jobs WHERE id = ?", (job_id,)).fetchone()
    if job is None:
        conn.close()
        raise HTTPException(status_code=404, detail=f"preview job not found: {job_id}")
    if job["status"] not in ("failed",):
        conn.close()
        raise HTTPException(
            status_code=400,
            detail=f"job status={job['status']} は再試行できません(failedのみ対象)",
        )
    retry_jobs(conn, [job_id])
    conn.commit()
    result = dispatch_pending_preview_jobs(conn, max_jobs=1)
    ensure_preview_wait_monitor()
    status = sync_checkpoint_status(conn, int(job["checkpoint_id"]))
    conn.close()
    return {"job_id": job_id, "dispatch_result": result, "checkpoint_status": status}


@router.get("/jobs/{checkpoint_id}")
def list_preview_jobs(checkpoint_id: int) -> dict:
    """Checkpoint単位のPreview Job一覧+集計(expected/pending/running/succeeded/failed)。"""
    conn = get_conn()
    ckpt = conn.execute("SELECT id FROM checkpoints WHERE id = ?", (checkpoint_id,)).fetchone()
    if ckpt is None:
        conn.close()
        raise HTTPException(status_code=404, detail=f"checkpoint not found: {checkpoint_id}")
    summary = job_summary(conn, checkpoint_id)
    rows = conn.execute(
        "SELECT id, prompt_index, instance_index, seed, status, error_detail, output_path, prompt, negative_prompt, "
        "preview_model_family, attempts, conditions_json, preview_snapshot_json FROM preview_jobs WHERE checkpoint_id = ? "
        "ORDER BY prompt_index, instance_index",
        (checkpoint_id,),
    ).fetchall()
    conn.close()
    jobs = [dict(r) for r in rows]
    for job in jobs:
        try:
            job["conditions"] = json.loads(job.pop("conditions_json") or "{}")
        except (TypeError, ValueError):
            job["conditions"] = {}
        try:
            job["preview_snapshot"] = json.loads(job.pop("preview_snapshot_json") or "{}")
        except (TypeError, ValueError):
            job["preview_snapshot"] = {}
    return {
        "checkpoint_id": checkpoint_id,
        "summary": summary,
        "jobs": jobs,
    }


@router.get("/{project_id}")
def list_previews(project_id: int, include_images: bool = True) -> dict:
    conn = get_conn()
    project = conn.execute("SELECT id FROM projects WHERE id = ?", (project_id,)).fetchone()
    if project is None:
        conn.close()
        raise HTTPException(status_code=404, detail=f"project not found: {project_id}")

    checkpoints = conn.execute(
        "SELECT id, run_id, file_path, epoch, step, mark, created_at, validation_status, validation_detail "
        "FROM checkpoints WHERE project_id = ? ORDER BY epoch DESC, id DESC",
        (project_id,),
    ).fetchall()
    timeline: list[dict] = []
    seen_files: set[str] = set()
    for checkpoint in checkpoints:
        checkpoint_id = int(checkpoint["id"])
        if checkpoint["file_path"] in seen_files:
            continue  # duplicate registration of the same weights file: keep the newest row only
        seen_files.add(checkpoint["file_path"])
        job_rows = conn.execute(
            "SELECT prompt_index, instance_index, output_path FROM preview_jobs "
            "WHERE checkpoint_id = ? AND status = 'succeeded' ORDER BY prompt_index, instance_index",
            (checkpoint_id,),
        ).fetchall()
        expected_jobs = int(conn.execute(
            "SELECT COUNT(*) FROM preview_jobs WHERE checkpoint_id = ?", (checkpoint_id,)
        ).fetchone()[0])
        if expected_jobs:
            current_samples = [
                {
                    "slot": f"p{job['prompt_index']}_i{job['instance_index']}",
                    "image_path": job["output_path"],
                }
                for job in job_rows
            ]
        else:
            # Legacy fallback only: one newest row per slot.
            current_samples = [dict(row) for row in conn.execute(
                "SELECT ps.slot, ps.image_path FROM preview_samples ps "
                "JOIN (SELECT slot, MAX(id) AS max_id FROM preview_samples WHERE checkpoint_id = ? GROUP BY slot) latest "
                "ON latest.max_id = ps.id ORDER BY ps.slot",
                (checkpoint_id,),
            ).fetchall()]
        conditions_row=conn.execute('SELECT conditions_json FROM preview_jobs WHERE checkpoint_id=? ORDER BY id LIMIT 1',(checkpoint_id,)).fetchone()
        try:conditions=json.loads(conditions_row['conditions_json'] or '{}') if conditions_row else {}
        except (ValueError,TypeError):conditions={}
        model_path=conditions.get('preview_base_checkpoint_path')
        if not model_path and checkpoint['run_id']:
            run_row=conn.execute('SELECT config_json FROM training_runs WHERE id=?',(checkpoint['run_id'],)).fetchone()
            try:run_config=json.loads(run_row['config_json'] or '{}') if run_row else {}
            except (ValueError,TypeError):run_config={}
            if run_config.get('model_family')=='anima':model_path=run_config.get('preview_base_checkpoint_path') or run_config.get('base_checkpoint_path')
        from ..training.runtime.metrics import checkpoint_progress
        recorded_step,step_source=checkpoint_progress(checkpoint['file_path'],checkpoint['step'])
        item = {
            "checkpoint_id": checkpoint_id,
            "preview_model_path": model_path,
            "preview_conditions": conditions,
            "epoch": int(checkpoint["epoch"]),
            "step": recorded_step,
            "step_source": step_source,
            "mark": checkpoint["mark"],
            "created_at": checkpoint["created_at"],
            "validation_status": checkpoint["validation_status"],
            "validation_detail": checkpoint["validation_detail"],
            "preview_job_summary": job_summary(conn, checkpoint_id),
            "samples": {},
        }
        for sample in current_samples:
            slot = str(sample["slot"])
            image_path = str(sample["image_path"] or "")
            item["samples"][slot] = image_path
            path = Path(image_path)
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                if include_images:
                    raw = path.read_bytes()
                    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                    item.setdefault("sample_previews", {})[slot] = (
                        f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")
                    )
                else:
                    item.setdefault("sample_previews", {})[slot] = (
                        f"/api/previews/image/{checkpoint_id}/{quote(slot, safe='')}"
                    )
        timeline.append(item)
    conn.close()
    return {
        "project_id": project_id,
        "slots": ["face", "bust", "full", "bg"],
        "timeline": timeline,
        "message": "preview timeline loaded",
    }
