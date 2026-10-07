"""Preview Job SSOT — 個々のPreview要求(1 checkpoint × 1 prompt × 1 instance)を
永続的に追跡する。

checkpoints.validation_status(preview_succeeded/preview_partial/preview_failed等)は
このテーブルからの派生集計値であり、ここがSingle Source of Truth。

実データ監査(run_id=39)で確定した実バグ: 以前はcheckpoint単位のstatusしか無く、
3 Prompt中1件成功しただけで全体が'preview_succeeded'として記録されていた。
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
import uuid
import time
from pathlib import Path

logger = logging.getLogger(__name__)

JOB_STATUSES = ("pending", "running", "succeeded", "failed")
PREVIEW_WAIT_POLL_SECONDS = 5.0
_PREVIEW_WAIT_MONITOR_THREAD: threading.Thread | None = None
_PREVIEW_WAIT_MONITOR_LOCK = threading.Lock()
_PREVIEW_DISPATCH_LOCK = threading.Lock()


def resolve_job_conditions(conn, model_family: str) -> dict:
    """Preview実行で解決される条件をJob証拠として正規化する。

    ``source`` は、実際のPreviewRequestへ渡す直前に記録した場合だけ
    ``runtime_request`` になる。旧preview_samplesの取り込みは
    ``legacy_import`` として、観測済み条件と混同させない。
    """
    from ...services.preview_prompts import resolve_preview_params
    from ...trainers import get_spec

    spec = get_spec(model_family)
    if spec is None:
        return {"model_family": model_family, "source": "unresolved"}
    sampler, cfg, steps = resolve_preview_params(
        conn,
        spec_sampler=spec.preview_sampler,
        spec_cfg=spec.preview_cfg,
        spec_steps=spec.preview_steps,
    )
    # ModelSpecの上限を実際のPreviewRequestと同じ規則で適用する。
    resolution = 1024 if model_family in ('anima','sdxl') else spec.preview_max_resolution or 512
    return {
        "model_family": model_family,
        "resolution": int(resolution or 512),
        "sampler": str(sampler),
        "scheduler": "simple",
        "cfg": float(cfg),
        "steps": int(steps),
    }


def _conditions_json(conn, model_family: str, *, source: str) -> str:
    conditions = resolve_job_conditions(conn, model_family)
    conditions["source"] = source
    return json.dumps(conditions, ensure_ascii=False, sort_keys=True)


def _preview_snapshot_json(
    conn, *, project_id: int, checkpoint_id: int, run_id: int, epoch: int,
    prompt: str, negative_prompt: str, seed: int, conditions_json: str, source: str,
    profile_snapshot_id: int | None = None,
) -> str:
    """PreviewSpecSnapshotをJobへ固定する。既存画像は再生成しない。"""
    try:
        conditions = json.loads(conditions_json or "{}")
    except (TypeError, ValueError):
        conditions = {}
    if profile_snapshot_id is None:
        profile = conn.execute(
            "SELECT id, name, snapshot_hash FROM basepipe_preview_profile_snapshots "
            "WHERE project_id = ? ORDER BY id DESC LIMIT 1", (project_id,)
        ).fetchone()
    else:
        profile = conn.execute(
            "SELECT id, name, snapshot_hash FROM basepipe_preview_profile_snapshots "
            "WHERE id = ? AND project_id = ?", (profile_snapshot_id, project_id)
        ).fetchone()
    snapshot = {
        "profile_snapshot_id": int(profile["id"]) if profile else None,
        "profile_snapshot_hash": profile["snapshot_hash"] if profile else None,
        "profile_name": profile["name"] if profile else None,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": int(seed),
        "width": int(conditions.get("width") or conditions.get("resolution") or 512),
        "height": int(conditions.get("height") or conditions.get("resolution") or 512),
        "steps": conditions.get("steps"),
        "cfg": conditions.get("cfg"),
        "sampler": conditions.get("sampler"),
        "scheduler": conditions.get("scheduler"),
        "checkpoint_id": int(checkpoint_id),
        "run_id": int(run_id),
        "epoch": int(epoch),
        "base_checkpoint_path": conditions.get("preview_base_checkpoint_path"),
        "style_stack": conditions.get("extra_loras",[]),
        "lora_strength": conditions.get("lora_strength"),
        "backend": conditions.get("backend"),
        "source": source,
    }
    return json.dumps(snapshot, ensure_ascii=False, sort_keys=True)


def get_instances_per_prompt(conn) -> int:
    """Preview 1promptあたりの生成instance数(既定1)。app_settingsで保存する。"""
    row = conn.execute(
        "SELECT value FROM app_settings WHERE key='preview_instances_per_prompt'"
    ).fetchone()
    if row is None:
        return 1
    try:
        n = int(row["value"])
    except (ValueError, TypeError):
        return 1
    return max(1, n)


def compute_seed(prompt_index: int, instance_index: int) -> int:
    """決定的・再現可能なseed生成規則。同一checkpointをre-scanしてもseedは変わらない。"""
    return 42 + prompt_index * 1000 + instance_index


def ensure_jobs_for_checkpoint(
    conn,
    *,
    checkpoint_id: int,
    run_id: int,
    project_id: int,
    epoch: int,
    prompt_items: list[dict],
    instances_per_prompt: int,
    model_family: str,
) -> int:
    """このcheckpointに必要な全Preview Jobを冪等に作成する。

    UNIQUE(checkpoint_id, prompt_index, instance_index) により、既に存在する
    組み合わせへのINSERTはIntegrityErrorとなり無視される(何度re-scanしても
    重複作成されない)。戻り値は今回新規作成した件数。
    """
    created = 0
    outfits: list[str] = []
    signature: dict[str, list[str]] = {}
    negatives: dict[str, list[str]] = {}
    # AnimaのCompare条件は、作成時点で最新のFrozen Profileを正本にする。
    # 通常Prompt一覧を先にJobへ入れて後からrefreshすると、生成済み画像と
    # SnapshotのPromptがずれてFinal gateに到達できないためである。
    frozen_profile = None
    if model_family in ("anima","sdxl"):
        run_row = conn.execute("SELECT config_json FROM training_runs WHERE id = ?", (run_id,)).fetchone()
        run_config: dict = {}
        if run_row is not None:
            try:
                parsed_config = json.loads(run_row["config_json"] or "{}")
                run_config = parsed_config if isinstance(parsed_config, dict) else {}
            except (TypeError, ValueError):
                run_config = {}
        frozen_profile_id = run_config.get("preview_profile_snapshot_id")
        if frozen_profile_id is not None:
            frozen_profile = conn.execute(
                "SELECT id, payload_json FROM basepipe_preview_profile_snapshots "
                "WHERE id = ? AND project_id = ?",
                (frozen_profile_id, project_id),
            ).fetchone()
        else:
            # Legacy runs created before the immutable handoff field existed.
            frozen_profile = conn.execute(
                "SELECT id, payload_json FROM basepipe_preview_profile_snapshots "
                "WHERE project_id = ? ORDER BY id DESC LIMIT 1",
                (project_id,),
            ).fetchone()
    frozen_payload: dict = {}
    frozen_profile_id: int | None = None
    if frozen_profile is not None:
        try:
            frozen_payload = json.loads(frozen_profile["payload_json"] or "{}")
        except (TypeError, ValueError):
            frozen_payload = {}
        if frozen_payload:
            frozen_profile_id = int(frozen_profile["id"])
    if frozen_profile_id is not None:
        frozen_conditions = {
            "model_family": str(frozen_payload.get("model_family") or model_family),
            "resolution": int(frozen_payload.get("resolution") or 1024),
            "sampler": str(frozen_payload.get("sampler") or "euler"),
            "scheduler": str(frozen_payload.get("scheduler") or "simple"),
            "cfg": float(frozen_payload["cfg"]) if frozen_payload.get("cfg") is not None else 1.0,
            "steps": int(frozen_payload.get("steps") or 7),
            "source": "frozen_profile",
        }
        conditions_json = json.dumps(frozen_conditions, ensure_ascii=False, sort_keys=True)
    else:
        frozen_conditions = None
        conditions_json = _conditions_json(conn, model_family, source="resolved_pending")
    if model_family in ('anima','sdxl'):
        effective=json.loads(conditions_json)
        effective['preview_base_checkpoint_path']=run_config.get('preview_base_checkpoint_path') or run_config.get('base_checkpoint_path')
        for field,target in [('preview_steps','steps'),('preview_cfg','cfg')]:
            if run_config.get(field) is not None:effective[target]=run_config[field]
        if run_config.get('preview_base_checkpoint_path') or run_config.get('preview_steps') is not None or run_config.get('preview_cfg') is not None:effective['source']='run_override'
        effective.update(lora_strength=float(run_config.get('preview_lora_strength',1)),extra_loras=run_config.get('preview_extra_loras',[]),backend='comfyui')
        if run_config.get('preview_backend')=='comfyui':effective['source']='comfyui_epoch'
        from .state import run_dir
        if run_config.get('preview_backend')!='comfyui' and (run_dir(run_id) / 'native_sample_prompts.json').is_file():
            native_prompt=json.loads((run_dir(run_id) / 'native_sample_prompts.json').read_text(encoding='utf8'))[0]
            shift=float(native_prompt.get('flow_shift',3.0))
            effective.update(source='trainer_native_epoch', backend='trainer', sampler='euler', scheduler='uniform_flow' if shift==1 else 'shifted_flow', flow_shift=shift)
        conditions_json=json.dumps(effective,ensure_ascii=False,sort_keys=True)
    if frozen_profile_id is not None:
        # A frozen profile is one explicit prompt and seed, not N identical jobs.
        prompt_items = [prompt_items[0] if prompt_items else {}]
        instances_per_prompt = 1
        # One preview per training instance (outfit trigger): the same prompt/seed with only that outfit's trigger, so the outfits are comparable.
        outfits = outfit_triggers(conn, project_id)
        signature: dict[str, list[str]] = {}
        negatives: dict[str, list[str]] = {}
        user_outfits = frozen_payload.get("outfits") if isinstance(frozen_payload.get("outfits"), list) else None
        if user_outfits:  # the user's own per-outfit choices win: which outfits to preview, and their marker tags
            chosen = [o for o in user_outfits if o.get("enabled", True) and str(o.get("trigger", "")).strip()]
            outfits = [str(o["trigger"]).strip() for o in chosen]
            signature = {str(o["trigger"]).strip(): [str(t) for t in o.get("tags", [])] for o in chosen}
            negatives = {str(o["trigger"]).strip(): [str(t) for t in o.get("negative_tags", [])] for o in chosen}
        elif outfits:
            snap_id = run_config.get("dataset_snapshot_id")
            if snap_id is not None:
                caps = [str(r["caption_at_snapshot"] or "") for r in conn.execute("SELECT caption_at_snapshot FROM basepipe_snapshot_entries WHERE snapshot_id=?", (snap_id,))]
                signature = outfit_signature_tags(caps, outfits)
                negatives = outfit_negative_tags(caps, outfits)
        if outfits:
            prompt_items = [dict(prompt_items[0]) for _ in outfits]
    for p_idx, item in enumerate(prompt_items):
        positive = str(item.get("quality") or item.get("positive") or "").strip()
        trigger = str(item.get("trigger_words") or "").strip()
        if trigger:
            positive = f"{trigger}, {positive}" if positive else trigger
        negative = str(item.get("negative") or "").strip()
        for i_idx in range(max(1, instances_per_prompt)):
            if frozen_profile_id is not None:
                positive = str(frozen_payload.get("prompt") or positive)
                if len(prompt_items) > 1 or (outfits and len(outfits) == 1):
                    positive = prompt_for_outfit(positive, outfits, outfits[p_idx], signature)
                negative = str(frozen_payload.get("negative_prompt") or negative)
                if len(prompt_items) > 1 or (outfits and len(outfits) == 1):
                    extra_neg = [t for t in negatives.get(outfits[p_idx], []) if t.casefold() not in negative.casefold()]
                    negative = ", ".join([negative] + extra_neg) if extra_neg else negative
                seed = int(frozen_payload["seed"]) if frozen_payload.get("seed") is not None else 42
            else:
                seed = compute_seed(p_idx, i_idx)
            snapshot_json = _preview_snapshot_json(
                conn, project_id=project_id, checkpoint_id=checkpoint_id, run_id=run_id,
                epoch=epoch, prompt=positive, negative_prompt=negative, seed=seed,
                conditions_json=conditions_json,
                source=json.loads(conditions_json).get("source") or ("frozen_profile" if frozen_profile_id is not None else "resolved_pending"),
                profile_snapshot_id=frozen_profile_id,
            )
            try:
                conn.execute(
                    "INSERT INTO preview_jobs("
                    "checkpoint_id, run_id, project_id, epoch, prompt_index, instance_index, "
                    "prompt, negative_prompt, seed, preview_model_family, status, conditions_json, preview_snapshot_json"
                    ") VALUES (?,?,?,?,?,?,?,?,?,?,'pending',?,?)",
                    (
                        checkpoint_id, run_id, project_id, epoch, p_idx, i_idx,
                        positive, negative, seed, model_family, conditions_json, snapshot_json,
                    ),
                )
                created += 1
            except sqlite3.IntegrityError:
                continue  # 既に存在する(冪等)
    conn.commit()
    return created


def outfit_triggers(conn, project_id: int) -> list[str]:
    """Training instances (per-outfit triggers) of the project, in creation order."""
    rows = conn.execute("SELECT trigger_token FROM basepipe_concepts WHERE project_id=? AND concept_type='outfit' AND TRIM(trigger_token)<>'' ORDER BY id", (project_id,)).fetchall()
    return [str(r["trigger_token"]).strip() for r in rows]


import re as _re

# hair / eye / body / frame-meta tags describe the character or the shot, not the outfit
_NOT_OUTFIT_PATTERN = _re.compile(r"(^|\s)(hair|eyes|ponytail|twintails|bangs|breasts|girls?|boys?|frame|focus|pillarboxed|letterboxed|parody|meme|text|watermark|signature|commentary|translated)$|^(\d+(girls?|boys?))$")
_NOT_OUTFIT = {"solo", "1girl", "1boy", "2girls", "multiple girls", "looking at viewer", "simple background", "upper body", "cowboy shot", "full body", "portrait",
               "standing", "sitting", "closed mouth", "open mouth", "smile", "blush", "indoors", "outdoors", "sky", "cloud", "day", "night", "blurry", "from side", "from behind", "from above", "solo focus", "looking away", "looking to the side", "hair between eyes", "bangs", "sidelocks"}


def outfit_signature_tags(captions: list[str], triggers: list[str], per_outfit: int = 14, min_share: float = 0.3, min_lift: float = 0.15) -> dict[str, list[str]]:
    """Tags that mark each outfit: common inside the outfit's own images (>= min_share) and clearly rarer in the other outfits (lift >= min_lift).

    Captions come from the sealed training snapshot, so the preview asks for what the model was actually shown with that trigger.
    """
    from collections import Counter
    split = [[t.strip() for t in c.split(",") if t.strip()] for c in captions]
    skip = {t.casefold() for t in triggers} | _NOT_OUTFIT
    groups: dict[str, list[set[str]]] = {t: [] for t in triggers}
    for tags in split:
        lowered = {x.casefold(): x for x in tags}
        for trig in triggers:
            if trig.casefold() in lowered:
                groups[trig].append({x for x in tags if x.casefold() not in skip and not x.casefold().startswith("ex_") and not x.isupper() and not _NOT_OUTFIT_PATTERN.search(x.casefold())})
    result: dict[str, list[str]] = {}
    for trig, rows in groups.items():
        others = [r for t, rs in groups.items() if t != trig for r in rs]
        if not rows:
            result[trig] = []
            continue
        mine = Counter(t for r in rows for t in r)
        theirs = Counter(t for r in others for t in r)
        scored = []
        for tag, c in mine.items():
            share = c / len(rows)
            other_share = theirs[tag] / len(others) if others else 0.0
            if share >= min_share and share - other_share >= min_lift:
                scored.append((share - other_share, share, tag))
        scored.sort(reverse=True)
        result[trig] = [t for _, _, t in scored[:per_outfit]]
    return result


def outfit_negative_tags(captions: list[str], triggers: list[str], per_outfit: int = 6, min_other_share: float = 0.5, max_own_share: float = 0.1) -> dict[str, list[str]]:
    """Per outfit: tags the OTHER outfits clearly have (>= min_other_share) but this one practically never does -> keep them out of this outfit's preview."""
    from collections import Counter
    split = [{t.strip() for t in c.split(",") if t.strip()} for c in captions]
    skip = {t.casefold() for t in triggers} | _NOT_OUTFIT
    groups = {t: [s for s in split if t.casefold() in {x.casefold() for x in s}] for t in triggers}
    result: dict[str, list[str]] = {}
    for trig, rows in groups.items():
        other_groups = [rs for t, rs in groups.items() if t != trig and rs]
        if not rows or not other_groups:
            result[trig] = []
            continue
        mine = Counter(t for r in rows for t in r)
        per_group = [Counter(t for r in rs for t in r) for rs in other_groups]
        scored = []
        for tag in {t for pg in per_group for t in pg}:
            if tag.casefold() in skip or tag.casefold().startswith("ex_") or tag.isupper() or _NOT_OUTFIT_PATTERN.search(tag.casefold()):
                continue
            # judge against the outfit that shows the tag MOST (a pooled average hides a tag that is typical of one other outfit)
            other_share = max(pg[tag] / len(rs) for pg, rs in zip(per_group, other_groups))
            own_share = mine[tag] / len(rows)
            if other_share >= min_other_share and own_share <= max_own_share:
                scored.append((other_share - own_share, tag))
        scored.sort(reverse=True)
        result[trig] = [t for _, t in scored[:per_outfit]]
    return result


def prompt_for_outfit(prompt: str, all_triggers: list[str], trigger: str, signature: dict[str, list[str]] | None = None) -> str:
    """Same scene, only the outfit changes: drop every outfit trigger (and the other outfits' marker tags) from the prompt,
    then put ``trigger`` right after the shared trigger (first tag) followed by this outfit's own marker tags."""
    signature = signature or {}
    drop = {t.casefold() for t in all_triggers}
    for trig, tags_ in signature.items():
        if trig != trigger:
            drop |= {t.casefold() for t in tags_}
    tags = [t.strip() for t in prompt.split(",") if t.strip() and t.strip().casefold() not in drop]
    own = [t for t in signature.get(trigger, []) if t.casefold() not in {x.casefold() for x in tags}]
    tags[1:1] = [trigger] + own if tags else [trigger] + own
    return ", ".join(tags)


def backfill_legacy_preview_jobs(conn, checkpoint_id: int) -> int:
    """取り込み漏れた旧Preview sampleをJob SSOTへ冪等に補修する。

    旧checkpoint単位Previewは画像をpreview_samplesへ直接登録していたため、
    実画像が存在してもpreview_jobsが0件になることがあった。既存画像を再生成せず、
    slot/prompt/seedを復元して成功Jobとして記録する。
    """
    checkpoint = conn.execute(
        "SELECT project_id, run_id, epoch FROM checkpoints WHERE id = ?", (checkpoint_id,)
    ).fetchone()
    if checkpoint is None or checkpoint["run_id"] is None:
        return 0
    run = conn.execute(
        "SELECT config_json FROM training_runs WHERE id = ?", (checkpoint["run_id"],)
    ).fetchone()
    if run is None:
        return 0
    try:
        config = json.loads(run["config_json"] or "{}")
    except (TypeError, ValueError):
        config = {}
    model_family = str(config.get("model_family") or "")
    conditions_json = _conditions_json(conn, model_family, source="legacy_import")
    from ...services.preview_prompts import load_active_prompt_items

    prompt_items, _resolution = load_active_prompt_items(conn)
    samples = conn.execute(
        "SELECT slot, image_path FROM preview_samples WHERE checkpoint_id = ? ORDER BY id", (checkpoint_id,)
    ).fetchall()
    created = 0
    for sample in samples:
        match = re.search(r"(?:p)?(\d+)(?:_i(\d+))?", str(sample["slot"]))
        prompt_index = int(match.group(1)) if match else created
        instance_index = int(match.group(2) or 0) if match else 0
        item = prompt_items[prompt_index] if prompt_index < len(prompt_items) else {}
        positive = str(item.get("quality") or item.get("positive") or "").strip()
        trigger = str(item.get("trigger_words") or "").strip()
        if trigger:
            positive = f"{trigger}, {positive}" if positive else trigger
        negative = str(item.get("negative") or "").strip()
        filename_seed = re.search(r"_(\d+)\.(?:png|jpe?g|webp)$", str(sample["image_path"]), re.IGNORECASE)
        seed = int(filename_seed.group(1)) if filename_seed else compute_seed(prompt_index, instance_index)
        output_path = str(sample["image_path"])
        succeeded = Path(output_path).is_file()
        snapshot_json = _preview_snapshot_json(
            conn, project_id=checkpoint["project_id"], checkpoint_id=checkpoint_id,
            run_id=checkpoint["run_id"], epoch=checkpoint["epoch"], prompt=positive,
            negative_prompt=negative, seed=seed, conditions_json=conditions_json,
            source="legacy_import",
        )
        try:
            if succeeded:
                conn.execute(
                    "INSERT INTO preview_jobs("
                    "checkpoint_id, run_id, project_id, epoch, prompt_index, instance_index, "
                    "prompt, negative_prompt, seed, preview_model_family, status, attempts, output_path, conditions_json, preview_snapshot_json, completed_at"
                    ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)",
                    (
                        checkpoint_id, checkpoint["run_id"], checkpoint["project_id"], checkpoint["epoch"],
                        prompt_index, instance_index, positive, negative, seed, model_family,
                        "succeeded", 1, output_path, conditions_json, snapshot_json,
                    ),
                )
            else:
                conn.execute(
                    "INSERT INTO preview_jobs("
                    "checkpoint_id, run_id, project_id, epoch, prompt_index, instance_index, "
                    "prompt, negative_prompt, seed, preview_model_family, status, attempts, output_path, conditions_json, preview_snapshot_json"
                    ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        checkpoint_id, checkpoint["run_id"], checkpoint["project_id"], checkpoint["epoch"],
                        prompt_index, instance_index, positive, negative, seed, model_family,
                        "pending", 0, "", conditions_json, snapshot_json,
                    ),
                )
            created += 1
        except sqlite3.IntegrityError:
            continue
    if created:
        conn.commit()
    return created


def job_summary(conn, checkpoint_id: int) -> dict:
    """expected/pending/running/succeeded/failedの集計を返す。"""
    rows = conn.execute(
        "SELECT status, COUNT(*) AS c FROM preview_jobs WHERE checkpoint_id = ? GROUP BY status",
        (checkpoint_id,),
    ).fetchall()
    counts = {r["status"]: int(r["c"]) for r in rows}
    expected = sum(counts.values())
    return {
        "expected": expected,
        "pending": counts.get("pending", 0),
        "running": counts.get("running", 0),
        "succeeded": counts.get("succeeded", 0),
        "failed": counts.get("failed", 0),
    }


def compute_checkpoint_status(summary: dict) -> str:
    """Preview Job集計からCheckpoint全体状態を導出する(派生値、SSOTはJob)。

    優先順位: 全件成功 > running中 > 一部成功(partial) > 全件失敗 > 待機中。
    "1/3 succeeded → preview_succeeded" のような丸めは絶対に発生させない。
    """
    expected = summary["expected"]
    if expected == 0:
        return "none"
    if summary["succeeded"] == expected:
        return "preview_succeeded"
    if summary["running"] > 0:
        return "preview_running"
    if summary["succeeded"] > 0:
        return "preview_partial"
    if summary["failed"] == expected:
        return "preview_failed"
    if summary["pending"] > 0:
        return "preview_pending"
    return "preview_partial"


def sync_checkpoint_status(conn, checkpoint_id: int) -> str:
    """job_summaryから算出したstatusをcheckpoints.validation_statusへ反映する。

    validate_and_recordが既に'invalid'と判定している場合は上書きしない
    (壊れているという事実の方が重要)。
    """
    row = conn.execute(
        "SELECT validation_status FROM checkpoints WHERE id = ?", (checkpoint_id,)
    ).fetchone()
    if row is not None and row["validation_status"] == "invalid":
        return "invalid"
    summary = job_summary(conn, checkpoint_id)
    status = compute_checkpoint_status(summary)
    detail = f"{summary['succeeded']}/{summary['expected']} succeeded" if summary["expected"] else ""
    conn.execute(
        "UPDATE checkpoints SET validation_status = ?, validation_detail = ? WHERE id = ?",
        (status, detail, checkpoint_id),
    )
    conn.commit()
    return status


def missing_jobs_for_checkpoint(conn, checkpoint_id: int) -> list[dict]:
    """failed または output が消失した succeeded job のみを返す(missing/failed分だけの回復対象)。"""
    rows = conn.execute(
        "SELECT id, status, output_path FROM preview_jobs WHERE checkpoint_id = ?",
        (checkpoint_id,),
    ).fetchall()
    result = []
    for r in rows:
        if r["status"] == "failed":
            result.append(dict(r))
        elif r["status"] == "succeeded" and r["output_path"] and not Path(r["output_path"]).exists():
            result.append(dict(r))
    return result


def retry_jobs(conn, job_ids: list[int]) -> int:
    """指定したJobのみをpendingへ戻す(成功済みJobは対象に含めないこと=呼び出し側の責務)。"""
    n = 0
    for jid in job_ids:
        conn.execute(
            "UPDATE preview_jobs SET status='pending', error_detail='' WHERE id = ? AND status IN ('failed')",
            (jid,),
        )
        n += conn.total_changes
    conn.commit()
    return len(job_ids)


def dispatch_pending_preview_jobs(conn, *, max_jobs: int | None = None) -> dict:
    # Checkpoint monitoring and the waiting worker share one serialized lane.
    if not _PREVIEW_DISPATCH_LOCK.acquire(blocking=False):
        return {"processed": 0, "succeeded": 0, "failed": 0, "skipped_gpu_busy": 1}
    try:
        return _dispatch_pending_preview_jobs(conn, max_jobs=max_jobs)
    finally:
        _PREVIEW_DISPATCH_LOCK.release()


def _dispatch_pending_preview_jobs(conn, *, max_jobs: int | None = None) -> dict:
    """pending Jobを1件ずつ処理する(RTX 3090 Ti環境を想定し並列実行しない)。

    GPU Safe Scheduling: job.run_id の学習processが生存中はdispatchせず
    pendingのまま維持する(同一GPUでTraining/Preview同時実行によるVRAM枯渇・
    強制終了を防ぐ。run 39で実際に確認された障害)。

    既存のPreviewProvider(generate_preview_safe)をそのまま再利用し、新しい
    生成Subsystemは作らない。1件の失敗が他Jobの処理を止めないよう、
    例外はJob単位で捕捉してfailedへ記録し、次のJobへ進む。
    """
    from .monitor import proc_alive
    from ...preview import generate_preview_safe
    from ...preview.base import PreviewRequest
    from ...trainers import get_spec

    pending = conn.execute(
        "SELECT * FROM preview_jobs WHERE status='pending' ORDER BY created_at ASC"
    ).fetchall()

    processed = 0
    succeeded = 0
    failed = 0
    skipped_gpu_busy = 0

    for job in pending:
        if max_jobs is not None and processed >= max_jobs:
            break
        run_id = int(job["run_id"])
        from .comfy_epoch_hook import active_window
        from .state import run_dir
        window=active_window(run_dir(run_id))
        earlier_character_work = conn.execute(
            "SELECT id, status FROM basepipe_character_generation_runs "
            "WHERE status IN ('waiting','waiting_qwen','queued','running','running_qwen') ORDER BY id LIMIT 1"
        ).fetchone()
        earlier_training_work = conn.execute(
            "SELECT id, status FROM training_runs WHERE status IN ('queued','training') AND id != ? ORDER BY id LIMIT 1",
            (run_id,),
        ).fetchone()
        if window:
            earlier_character_work=conn.execute("SELECT id,status FROM basepipe_character_generation_runs WHERE status IN ('running','running_qwen') LIMIT 1").fetchone()
            earlier_training_work=conn.execute("SELECT id,status FROM training_runs WHERE status='training' AND id!=? LIMIT 1",(run_id,)).fetchone()
        if earlier_character_work is not None or earlier_training_work is not None:
            blocker = earlier_character_work or earlier_training_work
            kind = "Character Generation" if earlier_character_work is not None else "Training"
            conn.execute(
                "UPDATE preview_jobs SET status='pending', error_detail=? WHERE id=? AND status='pending'",
                (f"GPU1待機: 先行する{kind} Run #{blocker['id']} ({blocker['status']})", job["id"]),
            )
            conn.commit()
            skipped_gpu_busy += 1
            continue
        # Native samples are emitted by the training process at epoch boundaries.
        # Register completed images while it is still training; never dispatch a
        # second model for a run configured to use its own sampler.
        from .state import run_dir
        native_root = run_dir(run_id)
        job_conditions=json.loads(job['conditions_json'] or '{}')
        if job_conditions.get('backend')!='comfyui' and (native_root / "native_sample_prompts.json").is_file():
            candidates = sorted((native_root / "output" / "sample").glob(f"*_e{int(job['epoch']):06d}_00_*.png"))
            if candidates:
                from PIL import Image
                try:
                    with Image.open(candidates[-1]) as img:
                        img.verify()
                except (OSError, ValueError):
                    continue  # Writer may still be completing this file.
                path = str(candidates[-1])
                conn.execute("UPDATE preview_jobs SET status='succeeded',output_path=?,error_detail='',completed_at=CURRENT_TIMESTAMP WHERE id=? AND status='pending'", (path, job['id']))
                slot = f"p{job['prompt_index']}_i{job['instance_index']}"
                conn.execute("DELETE FROM preview_samples WHERE checkpoint_id=? AND slot=?", (job['checkpoint_id'], slot))
                conn.execute("INSERT INTO preview_samples(checkpoint_id,slot,image_path) VALUES (?,?,?)", (job['checkpoint_id'], slot, path))
                conn.commit()
                sync_checkpoint_status(conn, int(job['checkpoint_id']))
                succeeded += 1
                processed += 1
            elif not proc_alive(run_id):
                conn.execute("UPDATE preview_jobs SET status='failed',error_detail=?,completed_at=CURRENT_TIMESTAMP WHERE id=? AND status='pending'", ("学習エンジンのプレビュー画像がありません。学習ログを確認してください。", job['id']))
                conn.commit()
                sync_checkpoint_status(conn, int(job['checkpoint_id']))
                failed += 1
                processed += 1
            else:
                skipped_gpu_busy += 1
            continue
        if proc_alive(run_id) and (not window or int(job["epoch"])>int(window["epoch"])):
            conn.execute("UPDATE preview_jobs SET error_detail=? WHERE id=? AND status='pending'",
                         ("学習終了後に生成：同じGPUでの同時実行を待機しています", job["id"]))
            conn.commit()
            skipped_gpu_busy += 1
            continue

        # The canonical Anima workflow is physical-GPU1 only.  A completed
        # training process does not prove that GPU1 is available: a DCC app,
        # another Python process, or an unmanaged ComfyUI may still own it.
        # Keep the durable Preview Job pending until the same admission and
        # ownership contract used by H3/Qwen succeeds.  Never fall back to
        # GPU0 and never turn temporary occupancy into a failed Preview.
        if str(job["preview_model_family"]) == "anima":
            from ...desktop_comfy import inspect as inspect_bound_comfy
            try:
                bound=inspect_bound_comfy({})
                if not any('RTX 3090 Ti' in d.get('name','') for d in bound['devices']):
                    raise RuntimeError('接続先が指定GPU1のRTX 3090 Tiではありません')
            except Exception as exc:
                conn.execute("UPDATE preview_jobs SET error_detail=? WHERE id=? AND status='pending'",(str(exc),job['id']))
                conn.commit();skipped_gpu_busy+=1
                continue

        ckpt_row = conn.execute(
            "SELECT file_path FROM checkpoints WHERE id = ?", (job["checkpoint_id"],)
        ).fetchone()
        if ckpt_row is None or not Path(ckpt_row["file_path"]).exists():
            conn.execute(
                "UPDATE preview_jobs SET status='failed', error_detail=?, attempts=attempts+1, "
                "completed_at=CURRENT_TIMESTAMP WHERE id=?",
                ("checkpoint file not found", job["id"]),
            )
            conn.commit()
            failed += 1
            processed += 1
            continue

        spec = get_spec(job["preview_model_family"])
        if spec is None:
            conn.execute(
                "UPDATE preview_jobs SET status='failed', error_detail=?, attempts=attempts+1, "
                "completed_at=CURRENT_TIMESTAMP WHERE id=?",
                (f"ModelSpec not found: {job['preview_model_family']}", job["id"]),
            )
            conn.commit()
            failed += 1
            processed += 1
            continue

        settings_rows = conn.execute(
            "SELECT key, value FROM app_settings WHERE key IN ('comfyui_root','comfyui_url')"
        ).fetchall()
        settings = {str(r["key"]): str(r["value"]) for r in settings_rows}

        try:
            recorded_conditions = json.loads(job["conditions_json"] or "{}")
        except (TypeError, ValueError):
            recorded_conditions = {}
        # A frozen-profile refresh records the exact conditions before dispatch.
        # Preserve them as the evidence SSOT; rebuilding from ModelSpec here would
        # silently turn euler back into the legacy "auto" value.
        conditions_json = job["conditions_json"] or _conditions_json(
            conn, job["preview_model_family"], source="runtime_request"
        )
        try:
            condition_source = str(recorded_conditions.get("source") or "runtime_request")
        except (TypeError, ValueError):
            condition_source = "runtime_request"
        # A retry/dispatch must preserve the Profile Snapshot already bound to
        # this Job.  Looking up the latest profile here would silently relink a
        # refresh requested for an older frozen profile to a newer profile.
        try:
            recorded_snapshot = json.loads(job["preview_snapshot_json"] or "{}")
        except (TypeError, ValueError):
            recorded_snapshot = {}
        profile_snapshot_id = recorded_snapshot.get("profile_snapshot_id")
        try:
            profile_snapshot_id = int(profile_snapshot_id) if profile_snapshot_id is not None else None
        except (TypeError, ValueError):
            profile_snapshot_id = None
        snapshot_json = _preview_snapshot_json(
            conn, project_id=int(job["project_id"]), checkpoint_id=int(job["checkpoint_id"]),
            run_id=run_id, epoch=int(job["epoch"]), prompt=job["prompt"],
            negative_prompt=job["negative_prompt"], seed=int(job["seed"]),
            conditions_json=conditions_json, source=condition_source,
            profile_snapshot_id=profile_snapshot_id,
        )
        # error_detailもここでクリアする。running→pending(Restart Recovery等)を経て
        # 再dispatchされたJobは、retry_jobs()を経由しない経路だと前回失敗時の
        # error_detailが残ったままrunning表示になり、現在進行中の試行なのに
        # 古いエラーメッセージが表示され続ける実バグがあった。
        # Claim the GPU lane under the same SQLite write lock used by Training.
        # Recheck after catalog/graph preparation so queued→training cannot race us.
        conn.execute('BEGIN IMMEDIATE')
        active_training=conn.execute("SELECT id FROM training_runs WHERE status='training'").fetchall()
        current_window=active_window(run_dir(run_id))
        if any(int(r['id'])!=run_id or not current_window for r in active_training) or conn.execute("SELECT 1 FROM preview_jobs WHERE status='running' AND id!=? LIMIT 1",(job['id'],)).fetchone():
            conn.rollback();skipped_gpu_busy+=1;continue
        claimed = conn.execute(
            "UPDATE preview_jobs SET status='running', started_at=CURRENT_TIMESTAMP, "
            "attempts=attempts+1, error_detail='', conditions_json=?, preview_snapshot_json=? WHERE id=? AND status='pending'",
            (conditions_json, snapshot_json, job["id"]),
        ).rowcount
        conn.commit()
        if not claimed:
            continue
        processed += 1

        request = PreviewRequest(
            model_family=job["preview_model_family"],
            checkpoint_path=ckpt_row["file_path"],
            run_id=run_id,
            epoch=int(job["epoch"]),
            prompt_items=[{
                "positive": job["prompt"], "negative": job["negative_prompt"], "seed": int(job["seed"]),
            }],
            resolution=int(recorded_conditions.get("resolution") or spec.preview_max_resolution or 512),
            settings=settings,
            sampler=str(recorded_conditions.get("sampler") or spec.preview_sampler),
            cfg=float(recorded_conditions["cfg"]) if recorded_conditions.get("cfg") is not None else spec.preview_cfg,
            steps=int(recorded_conditions.get("steps") or spec.preview_steps),
            extra=recorded_conditions,
        )
        try:
            results = generate_preview_safe('comfyui' if recorded_conditions.get('backend')=='comfyui' else spec.preview_backend, request)
            result = results[0] if results else None
        except Exception as exc:  # noqa: BLE001 — 1件の異常で他Jobを止めない
            result = None
            logger.warning("preview_jobs: job %s dispatch failed: %s", job["id"], exc)
            results_error = str(exc)
        else:
            results_error = result.error if result else "no result returned"

        if result is not None and result.image_bytes is not None:
            run_out_dir = ckpt_row["file_path"]
            sample_dir = Path(run_out_dir).parent / "sample"
            sample_dir.mkdir(parents=True, exist_ok=True)
            try:
                snapshot_meta = json.loads(job["preview_snapshot_json"] or "{}")
            except (TypeError, ValueError):
                snapshot_meta = {}
            profile_suffix = f"_profile{snapshot_meta['profile_snapshot_id']}" if snapshot_meta.get("profile_snapshot_id") else ""
            out_name = (
                f"preview_e{int(job['epoch']):06d}_p{job['prompt_index']:02d}"
                f"_i{job['instance_index']:02d}_{job['seed']}{profile_suffix}_{uuid.uuid4().hex[:10]}.png"
            )
            out_path = sample_dir / out_name
            out_path.write_bytes(result.image_bytes)
            conn.execute(
                "UPDATE preview_jobs SET status='succeeded', output_path=?, error_detail='', "
                "completed_at=CURRENT_TIMESTAMP WHERE id=?",
                (str(out_path), job["id"]),
            )
            slot = f"p{job['prompt_index']}_i{job['instance_index']}"
            conn.execute(
                "DELETE FROM preview_samples WHERE checkpoint_id = ? AND slot = ?",
                (job["checkpoint_id"], slot),
            )
            conn.execute(
                "INSERT INTO preview_samples(checkpoint_id, slot, image_path) VALUES (?,?,?)",
                (job["checkpoint_id"], slot, str(out_path)),
            )
            conn.commit()
            succeeded += 1
        else:
            conn.execute(
                "UPDATE preview_jobs SET status='failed', error_detail=?, completed_at=CURRENT_TIMESTAMP "
                "WHERE id=?",
                (results_error or "unknown error", job["id"]),
            )
            conn.commit()
            failed += 1

        sync_checkpoint_status(conn, int(job["checkpoint_id"]))

    return {
        "processed": processed, "succeeded": succeeded, "failed": failed,
        "skipped_gpu_busy": skipped_gpu_busy, "total_pending": len(pending),
    }


def recover_stale_jobs_on_startup(conn) -> dict:
    """Backend起動時のRestart Recovery。

    running → pending (Backend再起動でWorkerスレッドが消えた=処理未完了)
    succeeded かつ output消失 → pending (再生成が必要)
    pending / failed はそのまま(failedは明示的Retry待ち)
    成功済みかつoutput存在するJobは一切変更しない(再実行しない)。
    """
    running_rows = conn.execute("SELECT id FROM preview_jobs WHERE status='running'").fetchall()
    for r in running_rows:
        conn.execute("UPDATE preview_jobs SET status='pending' WHERE id=?", (r["id"],))

    succeeded_rows = conn.execute(
        "SELECT id, output_path FROM preview_jobs WHERE status='succeeded'"
    ).fetchall()
    missing_output = 0
    for r in succeeded_rows:
        if r["output_path"] and not Path(r["output_path"]).exists():
            conn.execute(
                "UPDATE preview_jobs SET status='pending', error_detail='output file missing after restart' WHERE id=?",
                (r["id"],),
            )
            missing_output += 1

    conn.commit()
    return {"reset_running": len(running_rows), "reset_missing_output": missing_output}


def _preview_wait_monitor_cycle() -> dict:
    """Dispatch at most one durable Preview Job and return observable progress."""
    from ...db import get_conn

    conn = get_conn()
    try:
        pending = int(conn.execute("SELECT COUNT(*) FROM preview_jobs WHERE status='pending'").fetchone()[0])
        if pending == 0:
            return {"pending": 0, "processed": 0, "skipped_gpu_busy": 0}
        result = dispatch_pending_preview_jobs(conn, max_jobs=1)
        remaining = int(conn.execute("SELECT COUNT(*) FROM preview_jobs WHERE status='pending'").fetchone()[0])
        return {"pending": remaining, **result}
    finally:
        conn.close()


def _release_managed_preview_runtime_if_idle() -> dict:
    """Release only our isolated ComfyUI after every GPU queue is idle."""
    from ...db import get_conn
    from ...routers import basepipe as character_runtime

    conn = get_conn()
    try:
        character_work = conn.execute(
            "SELECT 1 FROM basepipe_character_generation_runs "
            "WHERE status IN ('waiting','waiting_qwen','queued','running','running_qwen') LIMIT 1"
        ).fetchone()
        preview_work = conn.execute("SELECT 1 FROM preview_jobs WHERE status IN ('pending','running') LIMIT 1").fetchone()
        training_work = conn.execute(
            "SELECT 1 FROM training_runs WHERE status IN ('queued','training') LIMIT 1"
        ).fetchone()
    finally:
        conn.close()
    if character_work is not None or training_work is not None or preview_work is not None:
        return {"stopped": False, "detail": "another durable GPU queue still owns the managed runtime"}
    return character_runtime._stop_managed_h3_comfyui("Preview queue drained")


def _preview_wait_monitor() -> None:
    """Keep explicit pending Preview Jobs alive while GPU1 is temporarily busy."""
    global _PREVIEW_WAIT_MONITOR_THREAD
    try:
        while True:
            try:
                result = _preview_wait_monitor_cycle()
            except Exception as exc:  # noqa: BLE001 - transient probes stay retryable
                logger.warning("preview waiting monitor cycle failed: %s", exc)
                time.sleep(PREVIEW_WAIT_POLL_SECONDS)
                continue
            if int(result.get("pending", 0)) == 0:
                try:
                    _release_managed_preview_runtime_if_idle()
                except Exception as exc:  # noqa: BLE001 - ownership checks fail closed
                    logger.warning("preview managed runtime release skipped: %s", exc)
                return
            if int(result.get("processed", 0)) == 0:
                time.sleep(PREVIEW_WAIT_POLL_SECONDS)
    finally:
        with _PREVIEW_WAIT_MONITOR_LOCK:
            _PREVIEW_WAIT_MONITOR_THREAD = None
        # Close the insertion/exit race: a Job created while this thread was
        # leaving must immediately receive a replacement monitor.
        try:
            ensure_preview_wait_monitor()
        except Exception as exc:  # noqa: BLE001
            logger.warning("preview waiting monitor restart skipped: %s", exc)


def ensure_preview_wait_monitor() -> bool:
    """Start a single daemon only when an explicit pending Preview Job exists."""
    global _PREVIEW_WAIT_MONITOR_THREAD
    from ...db import get_conn

    conn = get_conn()
    try:
        exists = conn.execute("SELECT 1 FROM preview_jobs WHERE status='pending' LIMIT 1").fetchone() is not None
    finally:
        conn.close()
    if not exists:
        return False
    with _PREVIEW_WAIT_MONITOR_LOCK:
        if _PREVIEW_WAIT_MONITOR_THREAD and _PREVIEW_WAIT_MONITOR_THREAD.is_alive():
            return True
        _PREVIEW_WAIT_MONITOR_THREAD = threading.Thread(
            target=_preview_wait_monitor,
            daemon=True,
            name="preview-wait-monitor",
        )
        _PREVIEW_WAIT_MONITOR_THREAD.start()
    return True
