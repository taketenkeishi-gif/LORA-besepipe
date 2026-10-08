"""学習 API ルーター — API層のみの責務。

実行ロジック（プロセス起動・監視・キャッシュ生成等）は一切持たない。
モデルファミリーの判定・training_backend の解決・Backend への委譲のみを行う。

- バリデーション・DB CRUD: このファイル
- 実行エンジンの実装詳細: app.training.backends.*（Kohya/Musubi 等の実装名は
  ここにも UI レスポンスにも現れない）
- プロセス監視・推定・成果物登録: app.training.runtime
- プレビュー生成: app.preview（学習基盤とは独立、失敗しても学習に影響しない）
"""
from __future__ import annotations

import base64
import json
import logging
import math
import re
import shutil
import subprocess
import threading
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException
from PIL import Image

from ..db import get_conn
from ..training.runtime.anima_admission import required_free_mb as _anima_required_free_mb, query_occupancy as _query_gpu_foreign_processes, wait_reason as _anima_foreign_gpu_detail
from ..schemas import TrainingControlIn, TrainingStartIn
from ..trainers import get_spec, list_specs
from ..trainers import registry as _trainer_registry  # noqa: F401 — 登録トリガー
from ..training import detect_model_family, get_training_backend, resolve_training_backend
from ..training import registry as _training_registry  # noqa: F401 — 登録トリガー
from ..training.base import TrainingContext
from ..training.snapshot_inputs import fetch_snapshot_inputs, snapshot_user_approval_issue, verify_snapshot_input
from ..training.resume_checkpoint import select_resume_checkpoint
from ..training.backends.musubi import is_musubi_ready
from ..training.backends.anima import get_anima_script
from ..training.backends.sdxl import (
    detect_sdxl,
    get_train_script,
    is_kohya_ready,
    kohya_not_ready_reason,
    resolve_python,
)
from ..training.runtime.gpu_mapping import resolve_cuda_index
from ..training.runtime import (
    IMAGE_EXTS,
    RUNNER_LOCK,
    RUNNER_PROCESSES,
    RUNNER_THREADS,
    STEP_TIMING,
    auto_register_lora_asset as _auto_register_lora_asset,
    backfill_calib_from_runs as _runtime_backfill_calib,
    best_sec_per_step,
    cancel_queued as _cancel_queued,
    current_running as _current_running,
    detect_gpu_name,
    dispatch_next as _dispatch_next,
    ensure_runner as _runtime_ensure_runner,
    fail_run,
    gpu_base_sit,
    is_gpu_busy as _is_gpu_busy,
    latest_run,
    list_queue as _list_queue,
    load_calib,
    queue_position as _queue_position,
    reattach_active_runs as _runtime_reattach_active_runs,
    reconstruct_run_state,
    run_dir,
    sec_factor_for_run,
    set_project_status,
)
from ..training.runtime.log_parser import parse_log_state
from ..training.runtime.estimation import DEFAULT_BATCH, OPT_FACTOR
from ..training.runtime.evidence import load_run_manifest, parameter_evidence, record_prepared_evidence, write_run_manifest
from ..training.runtime.stages import set_training_stage

router = APIRouter(prefix="/training", tags=["training"])
logger = logging.getLogger(__name__)

_CKPT_EXTS = {".safetensors", ".ckpt", ".pt", ".pth"}
ANIMA_MIN_FREE_VRAM_MB = 20_000



TRAINING_WAIT_POLL_SECONDS = 10.0
_TRAINING_WAIT_MONITOR_LOCK = threading.Lock()
_TRAINING_WAIT_MONITOR_THREAD: threading.Thread | None = None


# ── 軽量 API ヘルパー ────────────────────────────────────────────────────────

def _ensure_project(project_id: int) -> dict:
    conn = get_conn()
    row = conn.execute(
        "SELECT id, name, project_type, outputs_dir, dataset_dir FROM projects WHERE id = ?",
        (project_id,),
    ).fetchone()
    conn.close()
    if row is None:
        raise HTTPException(status_code=404, detail=f"project not found: {project_id}")
    return dict(row)


def _get_app_settings() -> dict[str, str]:
    conn = get_conn()
    rows = conn.execute("SELECT key, value FROM app_settings").fetchall()
    conn.close()
    return {str(r["key"]): str(r["value"]) for r in rows}


def _effective_stage(run_id: int, status: str, current_stage: str | None) -> str:
    """Resolve the user-visible terminal stage from persisted Preview Job state.

    Older runs may have repeated ``preview/running`` events after the trainer
    already exited.  A completed run is only promoted to ``finalize`` when no
    Preview Job remains pending or running; failed jobs remain visible through
    their Preview counts and are never converted to success.
    """
    stage = current_stage or "planning"
    if status != "completed":
        return stage
    conn = get_conn()
    try:
        summary = conn.execute(
            "SELECT COUNT(*) AS expected, "
            "SUM(CASE WHEN status IN ('pending', 'running') THEN 1 ELSE 0 END) AS active "
            "FROM preview_jobs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
    finally:
        conn.close()
    if summary is None or int(summary["active"] or 0) == 0:
        return "finalize"
    return stage


def _query_gpu_memory(physical_index: int) -> dict[str, int] | None:
    """Read live VRAM counters for the physical nvidia-smi GPU index."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={physical_index}",
                "--query-gpu=memory.used,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode != 0:
            return None
        values = [int(part.strip()) for part in result.stdout.strip().split(",")]
        if len(values) != 3:
            return None
        return {"used_mb": values[0], "free_mb": values[1], "total_mb": values[2]}
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _checkpoint_search_dirs(settings: dict[str, str]) -> list[tuple[str, Path]]:
    """(source, dir) のリストを返す。ComfyUI / 学習エンジンのモデル格納先を探索対象にする。"""
    dirs: list[tuple[str, Path]] = []
    comfy = (settings.get("comfyui_root") or "").strip()
    if comfy:
        base = Path(comfy) / "models"
        for sub in ("checkpoints", "Stable-diffusion", "unet", "diffusion_models"):
            dirs.append(("comfyui", base / sub))
    kohya = (settings.get("kohya_root") or "").strip()
    if kohya:
        for sub in ("models", "pretrained", "sd-models"):
            dirs.append(("kohya", Path(kohya) / sub))
    return dirs


# ── オーケストレーション（Backend 呼び出しのみ。実装詳細は持たない） ──────────

def _dispatch_next_queued_job() -> int | None:
    """GPUが空いていればFIFO先頭のqueued runを起動する（Training Queueの中核）。

    _run_training_backend の終了時（成功/失敗/キャンセルいずれも）に必ず1回
    呼ばれることで、現在のrunがGPUを解放した直後に次のqueued runが自動的に
    開始される。呼び出し元自身のrun分の解放は、この関数が呼ばれる前に
    training_runs.status が既に 'completed'/'error'/'paused' へ遷移済みで
    あることが前提（fail_run / tail_monitor の終了処理内で行われる）。

    戻り値: 昇格・起動したrun_id（何も起動しなかった場合はNone）。
    """
    return _dispatch_next(_ensure_runner, set_project_status, _training_queue_admission)


def _training_queue_admission(row: dict) -> bool:
    """Admission with a durable, current explanation for the queued Run."""
    from ..desktop_jobs import write_state
    from ..training.runtime.state import run_dir
    import time
    try:config=json.loads(row.get('config_json') or '{}')
    except (TypeError,ValueError):config={}
    if config.get('model_family')!='anima':return True
    sharing=bool(config.get('allow_interactive_gpu_sharing'))
    memory=None;occupancy={}
    def finish(reason):
        path=run_dir(int(row['id']))/'admission.json';path.parent.mkdir(parents=True,exist_ok=True)
        try:previous=json.loads(path.read_text(encoding='utf8'))
        except (OSError,ValueError):previous={}
        value={'checked_at':time.time(),'probe_count':int(previous.get('probe_count',0))+1,'allowed':reason is None,'reason':reason or ('ゲーム・デスクトップと並行実行を許可済み' if sharing else '開始条件を満たしました'),'allow_interactive_gpu_sharing':sharing,'physical_gpu':1,'memory':memory,'occupancy':occupancy}
        write_state(path,json.dumps(value,ensure_ascii=False));return reason is None
    if int(config.get('gpu_device_id',-1))!=1:return finish('指定GPUは物理GPU1です')
    conn=get_conn()
    try:character_waiting=conn.execute("SELECT id FROM basepipe_character_generation_runs WHERE status IN ('waiting','waiting_qwen','queued','running','running_qwen') LIMIT 1").fetchone()
    finally:conn.close()
    if character_waiting is not None:return finish(f"先行する画像生成 Run #{character_waiting['id']} の終了待ち")
    memory=_query_gpu_memory(1)
    if memory is None:return finish('GPUメモリを取得できません。自動で再確認します')
    minimum=_anima_required_free_mb(config.get('training_memory_mode','standard'))
    if memory['free_mb']<minimum:return finish(f"空きVRAM {memory['free_mb']}MiB / 開始条件 {minimum}MiB。自動で再確認します")
    occupancy=_query_gpu_foreign_processes(1)
    return finish(_anima_foreign_gpu_detail(occupancy,sharing))


def _sharing_default(project_id=None):
    conn=get_conn()
    try:
        if project_id is not None:
            row=conn.execute('SELECT training_config_json FROM projects WHERE id=?',(project_id,)).fetchone()
            draft=json.loads(row[0] or '{}') if row else {}
            if isinstance(draft.get('allow_interactive_gpu_sharing'),bool):return draft['allow_interactive_gpu_sharing']
        row=conn.execute("SELECT value FROM app_settings WHERE key='training_allow_interactive_gpu_sharing'").fetchone()
        return bool(row and row[0]=='true')
    finally:conn.close()

@router.put('/sharing-preference')
def save_sharing_preference(payload:dict):
    if not isinstance(payload.get('allow'),bool):raise HTTPException(400,'並行実行の設定が不正です')
    conn=get_conn()
    try:
        conn.execute("INSERT INTO app_settings(key,value) VALUES('training_allow_interactive_gpu_sharing',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",('true' if payload['allow'] else 'false',));conn.commit()
    finally:conn.close()
    return {'allow':payload['allow'],'remembered':True}

@router.post('/runs/{run_id}/interactive-sharing')
def set_interactive_sharing(run_id:int,payload:dict) -> dict:
    if payload.get('allow') is not True:raise HTTPException(400,'並行実行の明示指定が必要です')
    conn=get_conn()
    try:
        row=conn.execute('SELECT status,project_id,config_json FROM training_runs WHERE id=?',(run_id,)).fetchone()
        if row is None:raise HTTPException(404,'学習Runがありません')
        if row['status'] not in ('queued','training'):raise HTTPException(409,'待機中または実行中の学習だけを切り替えられます')
        conn.execute("INSERT INTO app_settings(key,value) VALUES('training_allow_interactive_gpu_sharing','true') ON CONFLICT(key) DO UPDATE SET value='true'")
        project=conn.execute('SELECT training_config_json FROM projects WHERE id=?',(row['project_id'],)).fetchone()
        draft=json.loads(project[0] or '{}');draft['allow_interactive_gpu_sharing']=True
        conn.execute('UPDATE projects SET training_config_json=? WHERE id=?',(json.dumps(draft,ensure_ascii=False),row['project_id']))
        if row['status']=='training':
            conn.commit()
            return {'run_id':run_id,'status':'training','already_started':True,'remembered':True}
        cfg=json.loads(row['config_json'] or '{}');cfg['allow_interactive_gpu_sharing']=True;cfg['interactive_sharing_requested_at']=datetime.utcnow().isoformat();cfg['gpu_wait_reason']='ゲームとの並行実行を確認中';cfg['queue_if_busy']=True
        conn.execute("UPDATE training_runs SET config_json=?,updated_at=CURRENT_TIMESTAMP WHERE id=? AND status='queued'",(json.dumps(cfg,ensure_ascii=False),run_id));conn.commit()
    finally:conn.close()
    ensure_training_wait_monitor()
    return {'run_id':run_id,'status':'queued','allow_interactive_gpu_sharing':True,'automatic_start':True}


@router.put('/runs/{run_id}/preview-gpu')
def set_preview_gpu(run_id:int,payload:dict) -> dict:
    """Switch where this run's previews are made; the training process reads the marker at every epoch (takes effect from the next one)."""
    gpu=payload.get('gpu')
    if gpu not in ('gpu0','gpu1'):raise HTTPException(400,'gpu0（RTX 3060）か gpu1（RTX 3090 Ti）を指定してください')
    from ..training.runtime import preview_gpu
    from ..training.runtime.state import run_dir
    conn=get_conn()
    try:
        row=conn.execute('SELECT status,project_id,config_json FROM training_runs WHERE id=?',(run_id,)).fetchone()
        if row is None:raise HTTPException(404,'学習Runがありません')
        cfg=json.loads(row['config_json'] or '{}');cfg['preview_gpu']=gpu
        conn.execute('UPDATE training_runs SET config_json=?,updated_at=CURRENT_TIMESTAMP WHERE id=?',(json.dumps(cfg,ensure_ascii=False),run_id))
        project=conn.execute('SELECT training_config_json FROM projects WHERE id=?',(row['project_id'],)).fetchone()
        draft=json.loads(project[0] or '{}');draft['preview_gpu']=gpu  # remembered for the next run too
        conn.execute('UPDATE projects SET training_config_json=? WHERE id=?',(json.dumps(draft,ensure_ascii=False),row['project_id']));conn.commit()
    finally:conn.close()
    preview_gpu.set_mode(run_dir(run_id),gpu)
    from ..training.runtime.preview_jobs import ensure_preview_wait_monitor
    ensure_preview_wait_monitor()  # pending previews of this run may now be dispatchable
    return {'run_id':run_id,'preview_gpu':gpu,'status':row['status']}


def _training_wait_monitor() -> None:
    global _TRAINING_WAIT_MONITOR_THREAD
    try:
        while True:
            conn = get_conn()
            try:
                rows = conn.execute(
                    "SELECT id, config_json FROM training_runs WHERE status='queued' ORDER BY id"
                ).fetchall()
            finally:
                conn.close()
            armed = []
            for row in rows:
                try:
                    config = json.loads(row["config_json"] or "{}")
                except (TypeError, ValueError):
                    config = {}
                if config.get("model_family") == "anima" and bool(config.get("queue_if_busy")):
                    armed.append(int(row["id"]))
            if not armed:
                return
            _dispatch_next_queued_job()
            time.sleep(TRAINING_WAIT_POLL_SECONDS)
    finally:
        with _TRAINING_WAIT_MONITOR_LOCK:
            _TRAINING_WAIT_MONITOR_THREAD = None
        # Close the same insertion/exit race as the Character and Preview
        # queues: once the old handle is cleared, re-check durable armed rows.
        try:
            ensure_training_wait_monitor()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Anima training waiting monitor restart skipped: %s", exc)


def ensure_training_wait_monitor() -> bool:
    """Resume only explicitly armed Anima waiting Runs; never alter GPU/process ownership."""
    global _TRAINING_WAIT_MONITOR_THREAD
    conn = get_conn()
    try:
        rows = conn.execute("SELECT config_json FROM training_runs WHERE status='queued'").fetchall()
    finally:
        conn.close()
    exists = False
    for row in rows:
        try:
            config = json.loads(row["config_json"] or "{}")
        except (TypeError, ValueError):
            continue
        if config.get("model_family") == "anima" and bool(config.get("queue_if_busy")):
            exists = True
            break
    if not exists:
        return False
    with _TRAINING_WAIT_MONITOR_LOCK:
        if _TRAINING_WAIT_MONITOR_THREAD and _TRAINING_WAIT_MONITOR_THREAD.is_alive():
            return True
        _TRAINING_WAIT_MONITOR_THREAD = threading.Thread(
            target=_training_wait_monitor,
            daemon=True,
            name="anima-training-wait-monitor",
        )
        _TRAINING_WAIT_MONITOR_THREAD.start()
    return True


def _run_training_backend(backend, ctx: TrainingContext) -> None:
    """TrainingBackend.prepare -> cache -> train を順に呼ぶオーケストレーション。

    prepare/cache の失敗は fail_run で run を error 化する。train() 自体は
    DB 状態反映まで内包しているため、ここでは呼ぶだけでよい。

    finally節で必ず _dispatch_next_queued_job() を呼ぶことで、このrunが
    成功・失敗・キャンセルいずれで終わってもGPUが解放された直後に
    次のqueued runが自動的に開始される（Training Queueの自動進行）。
    """
    try:
        set_training_stage(ctx.run_id, "config_generation", "running")
        try:
            prepared = backend.prepare(ctx)
        except Exception as exc:
            fail_run(ctx.project_id, ctx.run_id, str(exc))
            return
        try:
            record_prepared_evidence(ctx.run_id, prepared)
        except Exception as exc:
            logger.warning("run %s evidence persistence failed: %s", ctx.run_id, exc)
        set_training_stage(ctx.run_id, "cache", "running")
        try:
            cache_ok = backend.cache(ctx, prepared)
        except Exception as exc:
            fail_run(ctx.project_id, ctx.run_id, f"キャッシュ生成エラー: {exc}")
            return
        if not cache_ok:
            fail_run(ctx.project_id, ctx.run_id, "キャッシュ生成に失敗しました")
            return
        set_training_stage(ctx.run_id, "training", "running")
        backend.train(ctx, prepared)
    finally:
        _dispatch_next_queued_job()


def _runner_loop(project_id: int) -> None:
    settings = _get_app_settings()
    conn = get_conn()
    row = latest_run(conn, project_id)
    conn.close()

    if row is None or row["status"] != "training":
        return

    run_id = int(row["id"])
    config_json = {}
    try:
        config_json = json.loads(row["config_json"] or "{}")
    except Exception:
        pass

    # モデルファミリーを判定し、ModelSpec.training_backend で適切なエンジンへ振り分ける。
    # ユーザーはモデルを選ぶだけでよく、内部でどのツールが使われるかは
    # TrainingBackend の実装詳細として隠蔽される。
    base_ckpt = config_json.get("base_checkpoint_path", "")
    explicit_family = config_json.get("model_family", "auto")
    model_family = detect_model_family(base_ckpt, explicit_family)
    spec = get_spec(model_family)
    training_backend_name = resolve_training_backend(model_family, config_json.get("training_engine", "auto"))

    ctx = TrainingContext(
        run_id=run_id, project_id=project_id, model_family=model_family,
        cfg={**config_json, "model_family": model_family}, settings=settings,
    )

    backend = get_training_backend(training_backend_name)
    if backend is not None and backend.is_ready(settings):
        _run_training_backend(backend, ctx)
        return

    # 専用エンジンが必要なモデル(Flux/Qwen-Image/KREA2/Wan2.1/HunyuanVideo/Z-Image等)を
    # 非互換の別エンジンにフォールバックさせると"KeyError: time_embed.0.weight"のような
    # 不可解なクラッシュになるため、model_familyを問わずフォールバックせず即失敗させる。
    #
    # 以前はsdxl/sdのみ、kohya未接続時にSimulatedBackend(実チェックポイントを
    # 生成しない20バイトのプレースホルダー)へ黙って切り替えていた。UI上には
    # 小さな警告バッジが出るのみで、ユーザーが気づかず「本物の学習が完了した」と
    # 誤認する恐れがあるため、この自動フォールバックを廃止する。
    # SimulatedBackendは引き続きユニットテストから直接構築して使用可能
    # (このルーター経由のフォールバックとしては到達不能にするだけ)。
    fail_run(
        project_id, run_id,
        f"model_family={model_family} 用の学習エンジンが未接続です。"
        f"設定タブでkohya_ss/musubi-tunerのパスを確認してください"
        f"（シミュレーションへの自動フォールバックは無効化されています）。",
    )


def _ensure_runner(project_id: int) -> None:
    _runtime_ensure_runner(project_id, _runner_loop)


# ── main.py 起動時フック（後方互換の関数名を維持） ────────────────────────────

def _backfill_calib_from_runs() -> None:
    """過去の学習実測値から学習時間予測のキャリブレーションをシードする（起動時に1回）。"""
    _runtime_backfill_calib(detect_sdxl)


def _reattach_active_runs() -> None:
    """起動時に学習中だった run を再アタッチ／status を実態へ reconcile する。

    Backend再起動はユーザーによる新規の学習開始操作ではないため、ここでは
    Training Queueを自動的に進めない。再起動前に残ったqueued Runは
    _quarantine_queued_runs_on_startup()でpausedへ隔離し、生存中の既存プロセス
    だけを監視へ再接続する。これにより「Backendを起動しただけで学習が再開する」
    経路を作らない。通常のRun完了後のQueue進行は、実行中ワーカーのfinallyから
    _dispatch_next_queued_job()を呼ぶ既存経路で行う。
    """
    def _sec_factor_fn(cfg: dict) -> tuple[float, str | None, str]:
        return sec_factor_for_run(cfg, detect_sdxl)
    _runtime_reattach_active_runs(_sec_factor_fn, on_run_finished=None)


def _quarantine_queued_runs_on_startup() -> int:
    """Quarantine stale Queue rows, preserving explicitly armed waiting Runs."""
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT id, project_id, config_json FROM training_runs WHERE status = 'queued'"
        ).fetchall()
        stale = []
        for row in rows:
            try:
                config = json.loads(row["config_json"] or "{}")
            except (TypeError, ValueError):
                config = {}
            if not (config.get("model_family") == "anima" and bool(config.get("queue_if_busy"))):
                stale.append(row)
        if not stale:
            return 0
        stale_ids = [int(row["id"]) for row in stale]
        marks = ",".join("?" for _ in stale_ids)
        conn.execute(
            "UPDATE training_runs SET status='paused', stop_mode='backend_restart', "
            "failure_code='BACKEND_RESTART', "
            "failure_message='Backend再起動後の自動学習再開を防止するためQueueから隔離しました。', "
            "failure_stage='queue', failure_at=CURRENT_TIMESTAMP, updated_at=CURRENT_TIMESTAMP "
            f"WHERE status='queued' AND id IN ({marks})",
            stale_ids,
        )
        for row in stale:
            set_project_status(conn, int(row["project_id"]), "idle")
        conn.commit()
        return len(stale)
    finally:
        conn.close()


# ── API エンドポイント ────────────────────────────────────────────────────

@router.get("/checkpoints")
def list_checkpoints() -> dict:
    """学習用ベースモデル候補を ComfyUI / 学習エンジンのモデルフォルダから列挙する。"""
    settings = _get_app_settings()
    seen: set[str] = set()
    checkpoints: list[dict] = []
    scanned_dirs: list[str] = []
    for source, d in _checkpoint_search_dirs(settings):
        if not d.exists() or not d.is_dir():
            continue
        scanned_dirs.append(str(d))
        for p in sorted(d.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in _CKPT_EXTS:
                continue
            rp = str(p.resolve())
            if rp in seen:
                continue
            seen.add(rp)
            try:
                size_mb = round(p.stat().st_size / 1024 / 1024, 1)
            except OSError:
                size_mb = 0
            # placeholder の極小ファイル(<1MB)は実モデルではないので除外
            if size_mb < 1:
                continue
            checkpoints.append({
                "name": p.name,
                "path": rp,
                "size_mb": size_mb,
                "source": source,
            })
    return {
        "checkpoints": checkpoints,
        "count": len(checkpoints),
        "scanned_dirs": scanned_dirs,
        "comfyui_connected": bool((settings.get("comfyui_root") or "").strip()),
    }


@router.get("/mode")
def get_training_mode() -> dict:
    """学習エンジンの接続状態を返す（標準/拡張エンジン / simulation）"""
    settings = _get_app_settings()
    kohya_ready = is_kohya_ready(settings)
    musubi_ready = is_musubi_ready(settings)
    # ANIMAはkohya_ss全体の接続性(is_kohya_ready)に加え、専用スクリプト
    # (anima_train_network.py)がkohya_ss側に存在するかを別途確認する
    # (SDXLと違い、ANIMA未対応バージョンのkohya_ssではkohya_readyがTrueでも
    # anima_train_network.pyが無く学習できないため)。
    anima_backend = get_training_backend("anima")
    anima_ready = anima_backend.is_ready(settings) if anima_backend else False
    kohya_root = settings.get("kohya_root", "")
    musubi_root = settings.get("musubi_root", "")
    train_script = get_train_script(kohya_root) if kohya_root else None
    python_exe = resolve_python(settings)
    reason = "" if kohya_ready else kohya_not_ready_reason(settings)

    if kohya_ready and musubi_ready:
        mode = "both"
    elif musubi_ready:
        mode = "musubi"
    elif kohya_ready:
        mode = "kohya"
    else:
        mode = "simulated"

    return {
        "mode": mode,
        "kohya_ready": kohya_ready,
        "musubi_ready": musubi_ready,
        "anima_ready": anima_ready,
        "kohya_root": kohya_root,
        "musubi_root": musubi_root,
        "train_script": train_script,
        "python_exe": python_exe,
        "reason": reason,
        "message": (
            "標準エンジン + 拡張エンジン 接続済み" if mode == "both"
            else "拡張エンジン 接続済み" if mode == "musubi"
            else "標準エンジン 接続済み" if mode == "kohya"
            else f"シミュレーションモード（{reason}）"
        ),
    }


def _trigger_state(project_id: int) -> dict:
    from collections import Counter
    project = _ensure_project(project_id)
    conn = get_conn()
    try:
        concept = conn.execute("SELECT id,trigger_token FROM basepipe_concepts WHERE project_id=? AND concept_type IN ('identity','character','style','hybrid') ORDER BY id DESC LIMIT 1", (project_id,)).fetchone()
        draft = json.loads(project.get('training_config_json') or '{}')
        captions = []
        if draft.get('dataset_snapshot_id'):
            captions = [str(row['caption_at_snapshot'] or '') for row in fetch_snapshot_inputs(conn,int(draft['dataset_snapshot_id']))]
        if not captions:
            for path in sorted(Path(project['dataset_dir']).glob('*.txt'))[:500]:
                try: captions.append(path.read_text(encoding='utf-8-sig'))
                except (OSError,UnicodeError): pass
        token = str(concept['trigger_token']) if concept else ''
        tags = [[t.strip() for t in re.split('[,\\n]',c) if t.strip()] for c in captions]
        first = Counter(t[0] for t in tags if t)
        folded = [{x.casefold() for x in t} for t in tags]
        instances = []
        for row in conn.execute("SELECT id,name,trigger_token FROM basepipe_concepts WHERE project_id=? AND concept_type=? ORDER BY id",(project_id,_INSTANCE_TYPE)).fetchall():
            tok = str(row['trigger_token'] or '').strip()
            instances.append({'id':row['id'],'name':row['name'],'trigger_token':tok,'matched':sum(tok.casefold() in f for f in folded) if tok else 0})
        owners = [sum(i['trigger_token'].casefold() in f for i in instances if i['trigger_token']) for f in folded]
        return {'project_id':project_id,'concept_id':concept['id'] if concept else None,'trigger_token':token,'caption_count':len(tags),'matched':sum(token.casefold() in f for f in folded) if token else 0,'candidates':[{'token':t,'count':n} for t,n in first.most_common(3)],
                'instances':instances,'instance_unassigned':sum(o==0 for o in owners) if instances else 0,'instance_multiple':sum(o>1 for o in owners)}
    finally: conn.close()

# 同一キャラクターの衣装などを別トリガーで学習する「インスタンス」。共通トリガーは従来どおり全画像に付け、
# インスタンスは画像ごとに1つだけ付ける（concept_type='outfit' の既存値を使うためスキーマ変更なし）。
_INSTANCE_TYPE = 'outfit'

def _instance_token(payload: dict) -> str:
    token = str(payload.get('trigger_token') or '').strip()
    if not token or len(token)>120 or any(c in token for c in [',','\n','\r','\x00']):
        raise HTTPException(status_code=400,detail='トリガーワードを1つ入力してください（120文字以内、カンマ・改行なし）')
    return token

def _assert_token_free(conn, project_id: int, token: str, except_id: int | None = None) -> None:
    for row in conn.execute("SELECT id,name FROM basepipe_concepts WHERE project_id=? AND LOWER(TRIM(trigger_token))=?",(project_id,token.casefold())).fetchall():
        if except_id is None or int(row['id']) != except_id:
            raise HTTPException(status_code=409,detail=f'「{row["name"]}」が同じトリガーワードを使っています。別の名前にしてください')

@router.get('/outfit-preview-defaults/{project_id}')
def outfit_preview_defaults(project_id: int) -> dict:
    """Per training instance: the marker tags the preview would add automatically (from the sealed dataset snapshot's captions)."""
    from .training_preview_support import outfit_defaults
    _ensure_project(project_id)
    conn = get_conn()
    try:
        return {"outfits": outfit_defaults(conn, project_id)}
    finally:
        conn.close()


@router.post('/instances/{project_id}')
def create_training_instance(project_id: int, payload: dict) -> dict:
    _ensure_project(project_id)
    token = _instance_token(payload)
    name = str(payload.get('name') or '').strip()[:120] or token
    conn = get_conn()
    try:
        _assert_token_free(conn, project_id, token)
        conn.execute("INSERT INTO basepipe_concepts(project_id,name,concept_type,trigger_token,description) VALUES(?,?,?,?,?)",(project_id,name,_INSTANCE_TYPE,token,'学習画面のインスタンス'))
        conn.commit()
    finally: conn.close()
    return _trigger_state(project_id)

@router.put('/instances/{project_id}/{concept_id}')
def update_training_instance(project_id: int, concept_id: int, payload: dict) -> dict:
    _ensure_project(project_id)
    token = _instance_token(payload)
    conn = get_conn()
    try:
        row = conn.execute("SELECT id,name FROM basepipe_concepts WHERE id=? AND project_id=? AND concept_type=?",(concept_id,project_id,_INSTANCE_TYPE)).fetchone()
        if row is None: raise HTTPException(status_code=404,detail='インスタンスが見つかりません')
        _assert_token_free(conn, project_id, token, concept_id)
        name = str(payload.get('name') or '').strip()[:120] or row['name']
        conn.execute("UPDATE basepipe_concepts SET name=?,trigger_token=?,updated_at=CURRENT_TIMESTAMP WHERE id=?",(name,token,concept_id))
        conn.commit()
    finally: conn.close()
    return _trigger_state(project_id)

@router.delete('/instances/{project_id}/{concept_id}')
def delete_training_instance(project_id: int, concept_id: int) -> dict:
    _ensure_project(project_id)
    conn = get_conn()
    try:
        row = conn.execute("SELECT id FROM basepipe_concepts WHERE id=? AND project_id=? AND concept_type=?",(concept_id,project_id,_INSTANCE_TYPE)).fetchone()
        if row is None: raise HTTPException(status_code=404,detail='インスタンスが見つかりません')
        if conn.execute("SELECT 1 FROM basepipe_dataset_snapshots WHERE concept_id=? LIMIT 1",(concept_id,)).fetchone():
            raise HTTPException(status_code=409,detail='このインスタンスは学習対象の記録に使われているため削除できません')
        conn.execute("DELETE FROM basepipe_concepts WHERE id=?",(concept_id,))
        conn.commit()
    finally: conn.close()
    return _trigger_state(project_id)

@router.get('/trigger/{project_id}')
def training_trigger(project_id: int) -> dict:
    return _trigger_state(project_id)

@router.put('/trigger/{project_id}')
def save_training_trigger(project_id: int, payload: dict) -> dict:
    token = str(payload.get('trigger_token') or '').strip()
    if not token or len(token)>120 or any(c in token for c in [',','\n','\r','\x00']):
        raise HTTPException(status_code=400,detail='トリガーワードを1つ入力してください（120文字以内、カンマ・改行なし）')
    project = _ensure_project(project_id);conn=get_conn()
    try:
        row=conn.execute("SELECT id FROM basepipe_concepts WHERE project_id=? AND concept_type IN ('identity','character','style','hybrid') ORDER BY id DESC LIMIT 1",(project_id,)).fetchone()
        if row: conn.execute("UPDATE basepipe_concepts SET trigger_token=?,updated_at=CURRENT_TIMESTAMP WHERE id=?",(token,row['id']))
        else: conn.execute("INSERT INTO basepipe_concepts(project_id,name,concept_type,trigger_token,description) VALUES(?,?,?,?,?)",(project_id,project['name'],'style' if project.get('project_type')=='style' else 'identity',token,'学習画面から設定'))
        conn.commit()
    finally: conn.close()
    return _trigger_state(project_id)

@router.post('/queue/recheck')
def recheck_training_queue(payload: TrainingControlIn) -> dict:
    _ensure_project(payload.project_id)
    conn=get_conn()
    try:
        row=latest_run(conn,payload.project_id)
        if not row or row['status']!='queued':raise HTTPException(status_code=409,detail='このプロジェクトに待機中の学習はありません')
        run_id=int(row['id'])
    finally:conn.close()
    ensure_training_wait_monitor()
    dispatched=_dispatch_next_queued_job()
    return {'run_id':run_id,'dispatched_run_id':dispatched,'message':'既存の待機キューを再評価しました。新しい学習は作成していません。'}

@router.get("/preflight")
def training_preflight(
    project_id: int, model_family: str = "auto", train_data_dir: str = "", base_checkpoint_path: str = "",
    dataset_snapshot_id: int | None = None, preview_profile_snapshot_id: int | None = None, gpu_device_id: int = 1,
    training_goal: str = "character_identity",
    quality_preset: str = "balanced",
    epochs: int = 5, repeats: int = 5, resolution: int = 512, train_batch_size: int = 2,
    learning_rate: float = 0.0001, rank: int = 16, optimizer: str = "AdamW8bit",
    gradient_checkpointing: bool = True,
    training_memory_mode: str = "standard",
    preview_base_checkpoint_path: str = "",
) -> dict:
    """学習開始前チェック — ユーザーが「学習開始」を押す前に、このDataset/Modelで
    実際に学習可能かをUI一目で判断できるようにする(技術的な内部情報の羅列ではなく
    ok/warning/blockedの3段階+次のアクションのみを返す)。

    重要: ここではDataset/Captionを一切変更しない(読み取り専用の判定のみ)。
    """
    project = _ensure_project(project_id)
    conn = get_conn()
    if preview_profile_snapshot_id is None and dataset_snapshot_id is not None:
        draft_row = conn.execute("SELECT training_config_json FROM projects WHERE id = ?", (project_id,)).fetchone()
        try:
            saved_draft = json.loads(draft_row["training_config_json"] or "{}") if draft_row else {}
        except (TypeError, ValueError):
            saved_draft = {}
        if saved_draft.get("dataset_snapshot_id") == dataset_snapshot_id:
            preview_profile_snapshot_id = saved_draft.get("preview_profile_snapshot_id")

    # "auto"はUI起動直後の既定値であり、/training/startと同じ判定基準
    # (base_checkpoint_pathのファイル名からの検出)で解決する。ここで
    # get_spec("auto")をそのまま呼ぶとModelSpecが見つからず誤ってblocked
    # 判定になる(実UI検証で発見)。
    requested_model_family = model_family
    model_family = detect_model_family(base_checkpoint_path, model_family)

    checks: list[dict] = []
    resolved_dependencies: dict[str, str] = {}
    resolved_trainer_script: str | None = None
    resolved_python_executable: str | None = None
    resolved_gpu_mapping: dict[str, object] | None = None
    resolved_gpu_memory: dict[str, int] | None = None
    anima_foreign_gpu_detail: str | None = None
    resolved_preview_profile: dict[str, object] | None = None
    resolved_snapshot: dict[str, object] | None = None

    def add(level: str, label: str, detail: str, action: str = "") -> None:
        checks.append({"level": level, "label": label, "detail": detail, "action": action})

    goal_labels = {
        "character_identity": "Character identity",
        "character_style": "Character style",
        "art_style": "Art style",
        "costume_concept": "Costume / Concept",
    }
    if training_goal not in goal_labels:
        add("blocked", "Training Goal", f"未対応のGoalです: {training_goal}", "Goalを選び直してください")
    elif project["project_type"] == "style" and training_goal == "character_identity":
        add("warning", "Training Goal", "Style ProjectでCharacter identityが選択されています", "Goalを確認してください")
    else:
        add("ok", "Training Goal", goal_labels[training_goal], "")

    preset_labels = {
        "fast": "Fast test",
        "balanced": "Balanced",
        "quality": "High quality",
        "custom": "Custom",
    }
    if quality_preset not in preset_labels:
        add("blocked", "Quality Preset", f"未対応のPresetです: {quality_preset}", "Presetを選び直してください")
    else:
        add("ok", "Quality Preset", preset_labels[quality_preset], "")

    # ── Dataset ──────────────────────────────────────────────────────────
    resolved_dir = train_data_dir.strip()
    if not resolved_dir:
        rec = _resolve_recommended_train_dir_for_preflight(conn, project_id)
        resolved_dir = rec["path"] if rec else str(project["dataset_dir"])
    image_count = 0
    captioned_count = 0

    if model_family == "anima" and dataset_snapshot_id is not None:
        # Anima consumes the immutable Snapshot staging path, not the mutable
        # legacy dataset_items folder. The Snapshot integrity gate below is
        # authoritative and must not be contradicted by a false folder block.
        add("ok", "Dataset Source", f"sealed Snapshot #{dataset_snapshot_id} を検証", "")
    else:
        rows = conn.execute(
            "SELECT file_path, caption FROM dataset_items WHERE project_id = ?", (project_id,)
        ).fetchall()
        dir_path = Path(resolved_dir) if resolved_dir else None
        in_dir = [r for r in rows if dir_path and Path(r["file_path"]).parent == dir_path and Path(r["file_path"]).exists()]
        image_count = len(in_dir)
        captioned_count = sum(1 for r in in_dir if str(r["caption"] or "").strip())

        if image_count == 0:
            add("blocked", "Dataset", f"学習画像が見つかりません（フォルダ: {resolved_dir or '未設定'}）",
                "Datasetを確認してください")
        elif captioned_count < image_count:
            missing = image_count - captioned_count
            add("warning", "Dataset", f"{image_count}枚中{missing}枚のCaptionが未設定です", "Captionを確認してください")
        else:
            add("ok", "Dataset", f"{image_count}枚（Caption {captioned_count}/{image_count}）", "")

    # ── Model / Backend ──────────────────────────────────────────────────
    spec = get_spec(model_family)
    if spec is None:
        add("blocked", "Model", f"model_family '{model_family}' のModelSpecが見つかりません", "Modelを選び直してください")
    else:
        add("ok", "Model", spec.display_name, "")
        settings = _get_app_settings()
        backend = get_training_backend(spec.training_backend)
        backend_ready = False
        if spec.training_backend == "musubi":
            backend_ready = is_musubi_ready(settings)
        elif spec.training_backend in ("sdxl", "anima"):
            backend_ready = is_kohya_ready(settings)
            if spec.training_backend == "anima" and backend is not None:
                backend_ready = backend.is_ready(settings)
        if backend is None:
            add("blocked", "Training Engine", "学習エンジンが登録されていません", "")
        elif not backend_ready:
            add("blocked", "Training Engine", f"{spec.training_backend} エンジンに接続できません",
                "Integrations設定でツールパスを確認してください")
        else:
            add("ok", "Training Engine", "利用可能", "")

        if spec.training_backend == "anima":
            # Anima固有依存はbackendの解決規則（明示設定→ComfyUI models）と
            # 同じ経路で確認し、存在しない依存をRun投入後まで先送りしない。
            dependency_specs = (
                ("Text Encoder", "text_encoder_path", "qwen_3_06b_base.safetensors", ("text_encoders", "clip")),
                ("VAE", "vae_path", "qwen_image_vae.safetensors", ("vae", "diffusion_models")),
            )
            for label, config_key, filename, subdirs in dependency_specs:
                try:
                    resolved = backend._resolve_dependency(  # type: ignore[attr-defined]
                        settings, "", filename, subdirs,
                    ) if backend is not None else ""
                    resolved_dependencies[config_key] = resolved
                    add("ok", label, Path(resolved).name, "")
                except Exception as exc:  # noqa: BLE001
                    add("blocked", label, str(exc), "Anima依存モデルを配置してください")

            python_exe = ""
            try:
                python_exe = resolve_python(settings)
                if Path(python_exe).is_file():
                    resolved_python_executable = python_exe
                    add("ok", "Python", Path(python_exe).name, "")
                else:
                    add("blocked", "Python", f"対象Pythonが見つかりません: {python_exe}", "設定を確認してください")
            except Exception as exc:  # noqa: BLE001
                add("blocked", "Python", str(exc), "設定を確認してください")

            trainer_script = get_anima_script(settings.get("kohya_root", ""))
            if trainer_script and Path(trainer_script).is_file():
                resolved_trainer_script = trainer_script
                add("ok", "Trainer Script", Path(trainer_script).name, "")
            else:
                add(
                    "blocked",
                    "Trainer Script",
                    "anima_train_network.py が見つかりません",
                    "Anima対応trainerを配置してください",
                )

            output_path = Path(str(project.get("outputs_dir") or ""))
            if output_path and output_path.parent.exists():
                add("ok", "Output Path", str(output_path), "")
                try:
                    free_gb = shutil.disk_usage(output_path.parent).free / (1024 ** 3)
                    if free_gb < 5:
                        add("warning", "Disk Space", f"空き容量 {free_gb:.1f} GB", "不要なファイルを整理してください")
                    else:
                        add("ok", "Disk Space", f"空き容量 {free_gb:.1f} GB", "")
                except OSError as exc:
                    add("warning", "Disk Space", str(exc), "空き容量を確認してください")
            else:
                add("blocked", "Output Path", f"出力先の親フォルダがありません: {output_path}", "出力先を設定してください")

            profile = conn.execute(
                "SELECT id, name, snapshot_hash FROM basepipe_preview_profile_snapshots "
                "WHERE id = ? AND project_id = ?",
                (preview_profile_snapshot_id, project_id),
            ).fetchone() if preview_profile_snapshot_id is not None else None
            if profile is None:
                add("blocked", "Preview Profile", "Datasetと同時に固定したPreview Profile Snapshotがありません", "Dataset Snapshotを作成し直してください")
            else:
                resolved_preview_profile = {
                    "id": int(profile["id"]),
                    "name": str(profile["name"]),
                    "snapshot_hash": str(profile["snapshot_hash"]),
                }
                add("ok", "Preview Profile", f"固定Snapshot #{profile['id']} {profile['name']}", "")

            concept = conn.execute(
                "SELECT id, trigger_token FROM basepipe_concepts WHERE project_id = ? AND concept_type IN ('identity','character','style','hybrid') ORDER BY id DESC LIMIT 1",
                (project_id,),
            ).fetchone()
            if concept is None or not str(concept["trigger_token"] or "").strip():
                add("warning", "Trigger", "Concept/trigger tokenが未設定です", "Conceptを確認してください")
            else:
                add("ok", "Trigger", str(concept["trigger_token"]), "")
            concept_triggers = conn.execute(
                "SELECT trigger_token FROM basepipe_concepts WHERE project_id = ? AND TRIM(trigger_token) <> ''",
                (project_id,),
            ).fetchall()
            normalized_triggers = [str(row["trigger_token"]).strip().casefold() for row in concept_triggers]
            duplicates = sorted({token for token in normalized_triggers if normalized_triggers.count(token) > 1})
            if duplicates:
                add(
                    "blocked",
                    "Trigger collision",
                    "Concept間で重複するTriggerがあります: " + ", ".join(duplicates),
                    "Conceptごとに固有のTriggerを設定してください",
                )
            elif normalized_triggers:
                add("ok", "Trigger collision", "Concept間のTrigger重複なし", "")
            else:
                add("warning", "Trigger collision", "検査対象のConcept Triggerがありません", "Conceptを作成してください")
            instance_state = _trigger_state(project_id)
            if instance_state["instances"]:
                breakdown = " / ".join(f"{i['name']} {i['matched']}枚" for i in instance_state["instances"])
                if instance_state["instance_unassigned"] or instance_state["instance_multiple"]:
                    add("warning", "Instances",
                        f"{breakdown}。インスタンス未割当 {instance_state['instance_unassigned']}枚・複数割当 {instance_state['instance_multiple']}枚",
                        "画像ごとにインスタンスのトリガーを1つだけ付けてください")
                else:
                    add("ok", "Instances", breakdown, "")
            outfit = conn.execute(
                "SELECT profiles_json FROM project_outfit_profiles WHERE project_id = ?", (project_id,)
            ).fetchone()
            if outfit is None or not json.loads(outfit["profiles_json"] or "[]"):
                add("warning", "Outfit Package", "Outfit packageが未登録です（identity学習は継続可能）", "必要ならOutfitを登録してください")
            else:
                add("ok", "Outfit Package", "登録済み", "")

        # ── Base Model存在確認 ───────────────────────────────────────────
        base_ckpt = base_checkpoint_path.strip()
        last_run = latest_run(conn, project_id)
        if not base_ckpt and last_run and last_run["config_json"]:
            try:
                last_cfg = json.loads(last_run["config_json"])
                if last_cfg.get("model_family") in (model_family, "auto"):
                    base_ckpt = str(last_cfg.get("base_checkpoint_path") or "")
            except Exception:
                base_ckpt = ""
        if base_ckpt:
            if Path(base_ckpt).exists():
                add("ok", "Base Model", Path(base_ckpt).name, "")
            else:
                add("blocked", "Base Model", f"前回使用したモデルが見つかりません（{Path(base_ckpt).name}）",
                    "Modelを選び直してください")
        else:
            add("warning", "Base Model", "未選択です", "Training設定でBase Modelを選択してください")

        if spec.training_backend == "anima":
            if dataset_snapshot_id is None:
                add("blocked", "Dataset Snapshot", "Anima学習にはsealed済みSnapshotの指定が必要です", "Snapshotを選択してください")
            else:
                snapshot = conn.execute(
                    "SELECT id, status, item_count FROM basepipe_dataset_snapshots WHERE id = ? AND project_id = ?",
                    (dataset_snapshot_id, project_id),
                ).fetchone()
                if snapshot is None or snapshot["status"] != "sealed":
                    add("blocked", "Dataset Snapshot", "指定Snapshotが存在しないかsealed状態ではありません", "Snapshotを作り直してください")
                else:
                    resolved_snapshot = {
                        "id": int(snapshot["id"]),
                        "status": str(snapshot["status"]),
                        "item_count": int(snapshot["item_count"] or 0),
                    }
                    snapshot_assets = fetch_snapshot_inputs(conn, dataset_snapshot_id)
                    image_count = len(snapshot_assets)
                    captioned_count = sum(1 for row in snapshot_assets if str(row["caption_at_snapshot"] or "").strip())
                    invalid = []
                    if len(snapshot_assets) != int(snapshot['item_count'] or 0):
                        invalid.append('確定した画像件数と現在のSnapshot件数が不一致')
                    for row in snapshot_assets:
                        approval_issue = snapshot_user_approval_issue(row)
                        if row["asset_type"] != "image":
                            invalid.append("非画像")
                        elif row["review_status"] != "approved":
                            invalid.append("未承認")
                        elif approval_issue:
                            invalid.append(approval_issue)
                        else:
                            integrity_ok, integrity_detail = verify_snapshot_input(row)
                            if not integrity_ok:
                                invalid.append(integrity_detail)
                            elif not str(row["caption_at_snapshot"] or "").strip():
                                invalid.append("caption欠落")
                    if not snapshot_assets:
                        add("blocked", "Dataset Snapshot", "Snapshotに学習対象がありません", "画像を追加してSnapshotを作成してください")
                    elif invalid:
                        add("blocked", "Dataset Snapshot", f"Snapshot #{dataset_snapshot_id} に不備があります: {', '.join(sorted(set(invalid)))}", "承認・Caption・ファイルを確認してください")
                    else:
                        add("ok", "Dataset Snapshot", f"Snapshot #{dataset_snapshot_id}（{len(snapshot_assets)}枚、全件承認・Caption済み）", "")

            if int(gpu_device_id) != 1:
                mapping = {
                    "verified": False,
                    "physical_index": int(gpu_device_id),
                    "reason": "Anima Basepipeは物理GPU1固定です。GPU0は使用しません。",
                }
            else:
                try:
                    mapping = resolve_cuda_index(1, resolve_python(_get_app_settings()))
                except Exception as exc:  # noqa: BLE001
                    mapping = {"verified": False, "reason": str(exc)}
            resolved_gpu_mapping = mapping
            if mapping.get("verified"):
                add("ok", "GPU1 Mapping", f"物理GPU1 {mapping['physical_name']} → CUDA {mapping['cuda_index']}（{mapping['cuda_name']}）", "")
                resolved_gpu_memory = _query_gpu_memory(1)
                if resolved_gpu_memory is not None:
                    vram_detail = f"使用 {resolved_gpu_memory['used_mb']} MiB / 空き {resolved_gpu_memory['free_mb']} MiB / 総量 {resolved_gpu_memory['total_mb']} MiB"
                    if resolved_gpu_memory["free_mb"] < _anima_required_free_mb(training_memory_mode):
                        add("blocked", "GPU1 VRAM", f"{vram_detail}（開始時の確認容量 {_anima_required_free_mb(training_memory_mode)} MiB未満）", "省VRAM設定とGPUの空きを確認してください")
                    else:
                        add("ok", "GPU1 VRAM", vram_detail, "")
                    occupancy = _query_gpu_foreign_processes(1)
                    anima_foreign_gpu_detail = _anima_foreign_gpu_detail(occupancy)
                    if anima_foreign_gpu_detail:
                        add(
                            "blocked",
                            "GPU1 Occupancy",
                            anima_foreign_gpu_detail,
                            "外部プロセスには触れません。他の項目が通ればGPU1待機キューへ追加できます",
                        )
                    elif occupancy.get("query_ok"):
                        add("ok", "GPU1 Occupancy", "計算負荷が低く、生成キューも空です（常駐メモリは保持）", "")
                    else:
                        add("warning", "GPU1 Occupancy", "compute-appsの実測に失敗しました", "GPU1の占有状態を確認してください")
                else:
                    add("warning", "GPU1 VRAM", "実測値を取得できませんでした", "nvidia-smiの状態を確認してください")
            else:
                add("blocked", "GPU1 Mapping", f"物理GPU1と対象PythonのCUDA列挙を照合できません: {mapping.get('reason', '不明')}", "GPU対応を確認してください")

    # ── GPU ──────────────────────────────────────────────────────────────
    try:
        import psutil  # noqa: F401
        gpu_ok = True
    except ImportError:
        gpu_ok = True  # psutil無しでもnvidia-smiでの検出に任せる
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        gpu_available = result.returncode == 0 and bool(result.stdout.strip())
    except Exception:
        gpu_available = False
    if model_family == "anima" and resolved_gpu_mapping and not resolved_gpu_mapping.get("verified"):
        add("blocked", "GPU", "Animaの物理GPU1とCUDA列挙の照合に失敗しています", "GPU1 Mappingを確認してください")
    elif (
        model_family == "anima"
        and resolved_gpu_memory is not None
        and resolved_gpu_memory["free_mb"] < _anima_required_free_mb(training_memory_mode)
    ):
        add(
            "blocked",
            "GPU",
            f"GPU1空き {resolved_gpu_memory['free_mb']} MiB（開始時の確認容量未満）",
            "省VRAM設定とGPUの空きを確認してください",
        )
    elif gpu_available:
        add("ok", "GPU", "利用可能", "")
    else:
        add("warning", "GPU", "GPU状態を確認できませんでした", "")

    if preview_base_checkpoint_path:
        preview_path=Path(preview_base_checkpoint_path)
        if not preview_path.is_file():add('blocked','Preview Model','プレビュー用モデルが見つかりません','生成用モデルを選び直してください')
        elif detect_model_family(str(preview_path),'auto') != model_family:add('blocked','Preview Model','学習モデルと生成用モデルの系統が一致しません','同じ系統の生成用モデルを選択してください')
        else:add('ok','Preview Model',preview_path.name,'')

    # ── Queue ────────────────────────────────────────────────────────────
    running = _current_running(conn)
    queued = _list_queue(conn)
    if running and int(running.get("project_id", -1)) == project_id:
        add("warning", "Queue", "このProjectは既に学習中です", "")
    elif running:
        add("ok", "Queue", f"他のProjectが実行中です（開始すると{len(queued) + 1}番目で待機）", "")
    elif (
        model_family == "anima"
        and resolved_gpu_memory is not None
        and resolved_gpu_memory["free_mb"] < _anima_required_free_mb(training_memory_mode)
    ):
        add("ok", "Queue", f"Queueは空です。GPU1 VRAMゲート通過まで安全に待機できます（空き {resolved_gpu_memory['free_mb']} MiB）", "待機キューへ追加できます")
    elif model_family == "anima" and anima_foreign_gpu_detail:
        add("ok", "Queue", "Queueは空です。別の計算・生成の完了後にGPU1で自動開始する待機枠を利用できます", "待機キューへ追加できます")
    else:
        add("ok", "Queue", "空いています（即座に開始できます）", "")

    conn.close()

    # PreflightとEstimateを別リクエストにせず、同じRequested値から生成する。
    # 推定は読み取り専用で、GPUプロセスや学習Runを開始しない。
    try:
        estimate = estimate_training_time(
            project_id=project_id,
            epochs=epochs,
            repeats=repeats,
            resolution=resolution,
            batch_size=train_batch_size,
            base_checkpoint_path=base_checkpoint_path,
            optimizer=optimizer,
            rank=rank,
            gradient_checkpointing=gradient_checkpointing,
            gpu_device_id=1 if model_family == "anima" else gpu_device_id,
            model_family=model_family,
            dataset_snapshot_id=dataset_snapshot_id,
        )
    except Exception as exc:  # noqa: BLE001
        estimate = {"ready": False, "error": str(exc)}

    levels = [c["level"] for c in checks]
    if "blocked" in levels:
        overall = "blocked"
    elif "warning" in levels:
        overall = "warning"
    else:
        overall = "ok"
    blocked_labels = {check["label"] for check in checks if check["level"] == "blocked"}
    queueable_gpu_labels = {"GPU1 VRAM", "GPU1 Occupancy", "GPU", "Queue"}
    can_queue = (
        model_family == "anima"
        and bool(blocked_labels)
        and blocked_labels.issubset(queueable_gpu_labels)
        and bool(resolved_gpu_mapping and resolved_gpu_mapping.get("verified"))
    )

    return {
        "project_id": project_id,
        "model_family": model_family,
        "train_data_dir": resolved_dir,
        "overall": overall,
        "checks": checks,
        # Spec v1.0のPreflight契約。blockedはerrors、warningはwarningsへ
        # 明示的に分離し、GUIがchecksの文言を再解釈せずに表示できるようにする。
        "warnings": [c["detail"] for c in checks if c["level"] == "warning"],
        "errors": [c["detail"] for c in checks if c["level"] == "blocked"],
        "estimate": estimate,
        "image_count": image_count,
        "captioned_count": captioned_count,
        "dataset_snapshot_id": dataset_snapshot_id,
        "can_start": overall != "blocked",
        "can_queue": can_queue,
        "requested": {
            "training_goal": training_goal,
            "quality_preset": quality_preset,
            "model_family": requested_model_family,
            "base_checkpoint_path": base_checkpoint_path,
            "train_data_dir": train_data_dir,
            "dataset_snapshot_id": dataset_snapshot_id,
            "preview_profile_snapshot_id": preview_profile_snapshot_id,
            "gpu_device_id": int(gpu_device_id),
        },
        "resolved": {
            "training_goal": training_goal if training_goal in goal_labels else None,
            "quality_preset": quality_preset if quality_preset in preset_labels else None,
            "train_data_dir": resolved_dir,
            "model_family": model_family,
            "base_checkpoint_path": base_ckpt,
            "dataset_snapshot": resolved_snapshot,
            "preview_profile_snapshot_id": preview_profile_snapshot_id,
            "dependencies": resolved_dependencies,
            "trainer_script_path": resolved_trainer_script,
            "python_executable": resolved_python_executable,
            "preview_profile": resolved_preview_profile,
            "gpu_device_id": 1 if model_family == "anima" else None,
            "gpu_memory": resolved_gpu_memory,
            "cuda_visible_devices": (
                str(resolved_gpu_mapping["cuda_index"])
                if resolved_gpu_mapping and resolved_gpu_mapping.get("verified")
                else None
            ),
        },
        "environment": {
            "requested_gpu_id": int(gpu_device_id),
            "gpu_physical_index": 1 if model_family == "anima" else None,
            "cuda_visible_devices": (
                str(resolved_gpu_mapping["cuda_index"])
                if resolved_gpu_mapping and resolved_gpu_mapping.get("verified")
                else None
            ),
            "gpu_mapping": resolved_gpu_mapping,
            "gpu_memory": resolved_gpu_memory,
        },
    }


def _resolve_recommended_train_dir_for_preflight(conn, project_id: int) -> dict | None:
    """dataset.pyの_resolve_recommended_train_dirと同一ロジック(循環import回避のため複製)。"""
    rows = conn.execute(
        "SELECT file_path FROM dataset_items WHERE project_id = ?", (project_id,)
    ).fetchall()
    existing = [Path(r["file_path"]) for r in rows if Path(r["file_path"]).exists()]
    if not existing:
        return None
    parents: dict[str, int] = {}
    for p in existing:
        d = str(p.parent)
        parents[d] = parents.get(d, 0) + 1
    best_dir, count = max(parents.items(), key=lambda kv: kv[1])
    if count < len(existing):
        return None
    return {"path": best_dir, "image_count": count}


@router.get("/estimate")
def estimate_training_time(
    project_id: int,
    epochs: int = 5,
    repeats: int = 5,
    resolution: int = 512,
    batch_size: int = 2,
    base_checkpoint_path: str = "",
    optimizer: str = "AdamW8bit",
    rank: int = 16,
    xformers: bool = True,
    gradient_checkpointing: bool = True,
    mixed_precision: str = "bf16",
    gpu_device_id: int = 1,
    model_family: str = "auto",
    dataset_snapshot_id: int | None = None,
) -> dict:
    """現在の設定から学習所要時間をリアルタイム算出する。

    精度の根拠（優先順）:
      1. live        — このプロジェクトが学習中で実測 sec/step がある
      2. calibrated  — 同 GPU・同モデル系統の過去実測から学習した基準値
      3. heuristic   — GPU 名 → 基準テーブル（実測前の初期推定）
      4. simulation  — エンジン未接続。1 step=0.5s 固定（厳密値）
    """
    _ensure_project(project_id)
    conn = get_conn()
    # NOTE: 以前は selected=1 のみでカウントしており、ファイルが実在しないDB上の
    # 不整合行(テスト汚染等)も学習画像として数えてしまい、実画像数と食い違う
    # 実バグがあった(実データ監査で発見: DB上selected=1が50件、実在ファイルは49件)。
    # /training/preflight と同じ基準(ファイル実在確認)に統一する。
    resolved_family = detect_model_family(base_checkpoint_path, model_family)
    if resolved_family == "anima" and dataset_snapshot_id is not None:
        snapshot = conn.execute(
            "SELECT id, status FROM basepipe_dataset_snapshots WHERE id = ? AND project_id = ?",
            (dataset_snapshot_id, project_id),
        ).fetchone()
        rows = fetch_snapshot_inputs(conn, dataset_snapshot_id) if snapshot and snapshot["status"] == "sealed" else []
        images = sum(1 for row in rows if Path(row["file_path"]).is_file())
    else:
        rows = conn.execute(
            "SELECT file_path FROM dataset_items WHERE project_id = ? AND selected = 1",
            (project_id,),
        ).fetchall()
        images = sum(1 for r in rows if Path(r["file_path"]).exists())
    conn.close()

    settings = _get_app_settings()
    # model_family="auto"(既定値)は/training/start・/training/preflightと同じ基準
    # (base_checkpoint_pathのファイル名からの検出)で解決する。以前はここで
    # is_sdxl判定のみを行い、musubi系(krea2/flux/qwen_image等)は常に"sd"として
    # 扱われ、model_factor・calibration lookupがSDXL/kohya前提の値になっていた
    # (実データ監査で発見: Krea2プロジェクトの見積もりが実際にはmusubiで学習される
    # にも関わらず mode="kohya" / "SD"表示になっていた)。
    spec = get_spec(resolved_family)
    is_sdxl = resolved_family == "sdxl"
    training_backend_name = spec.training_backend if spec else ("sdxl" if is_sdxl else "sd")

    # 実際にこのmodel_familyが使うTraining Backendが「batch単位でstepを分割する」
    # 方式かどうかで計算式を分ける(kohya/sdxlはbatch分割、musubi/simulationは
    # 従来通りimages*repeatsそのまま)。以前は「kohyaエンジン全体が接続済みか」
    # (is_kohya_ready)だけを見ており、Krea2(musubi)を学習しようとしていても
    # 「他にkohyaも繋がっている」という理由でbatch分割式が誤って適用されていた。
    kohya_ready = is_kohya_ready(settings)
    musubi_ready = is_musubi_ready(settings)
    # このmodel_familyが実際に使うbackendが接続済みかどうか(グローバルな
    # kohya_ready/musubi_readyの単純ORではなく、選択中のmodel_family専用の判定)。
    backend_ready = musubi_ready if training_backend_name == "musubi" else kohya_ready
    # batch分割(ceil(images*repeats/batch))はkohya固有の挙動ではなく、
    # batched dataloaderを使う実trainer全般(kohya/musubi共通)の基本動作。
    # 実データ検証: 実行済みKrea2(musubi)run(images=49,repeats=2,batch=6)の
    # DB保存済みsteps_per_epoch=17は ceil(49*2/6)=17 と一致し、49*2=98とは
    # 一致しなかった。従って「実backendが接続されている(=実際にプロセスが
    # 起動する)なら常にbatch分割、未接続でsimulationにフォールバックする
    # 場合のみimages*repeatsそのまま」が実測と整合する基準。
    uses_batch_division = backend_ready

    # 設定された解像度をそのまま使う（SDXLでも強制1024にしない＝フォーム値と一致させる）
    eff_res = max(64, resolution)
    batch = max(1, batch_size)
    epochs = max(1, epochs)

    if uses_batch_division:
        steps_per_epoch = max(1, math.ceil(images * repeats / batch))
    else:
        steps_per_epoch = max(1, images * repeats)
    total_steps = steps_per_epoch * epochs

    # アクティブな学習がある場合、実走のステップ数(aspect比バケット込みの実数)で
    # 予測を実態に一致させる。理論値(images*repeats/batch)はバケット端数を含まず
    # 実際より少なく出るため、学習中は学習パネルと総ステップが食い違っていた。
    # (kohyaログ形式のparse_log_stateはkohya/sdxl系のみ意味を持つ。musubi系は
    # ログ形式が異なるため、uses_batch_divisionと同じ基準でガードする)
    active_done: int | None = None  # 学習中なら実走の経過ステップ
    active_run_id: int | None = None
    if uses_batch_division:
        try:
            c2 = get_conn()
            arow = c2.execute(
                "SELECT id, log_path, total_epochs FROM training_runs "
                "WHERE project_id=? AND status='training' ORDER BY id DESC LIMIT 1",
                (project_id,),
            ).fetchone()
            c2.close()
            if arow:
                ls = parse_log_state(arow["log_path"])
                if ls["gtotal"]:
                    active_run_id = int(arow["id"])
                    total_steps = int(ls["gtotal"])
                    epochs = int(arow["total_epochs"]) or epochs
                    steps_per_epoch = max(1, total_steps // epochs)
                    active_done = int(ls["gstep"] or 0)
        except Exception:
            pass

    gpu_name = detect_gpu_name(gpu_device_id)
    reso_factor = (eff_res / 512.0) ** 2
    model_factor = 2.8 if is_sdxl else 1.0
    opt_factor = OPT_FACTOR.get(optimizer, 1.0)
    # rank: LoRA の行列計算量は rank に概ね比例。rank=16 を基準(1.0)とする。
    rank_factor = max(0.5, rank / 16.0) if rank > 0 else 1.0
    # xformers: メモリ効率的アテンション。有効で約 12% 高速化。
    xformers_factor = 0.88 if xformers else 1.0
    # gradient_checkpointing: メモリ削減のため再計算が発生し約 20% 遅延。
    gc_factor = 1.20 if gradient_checkpointing else 1.0
    # mixed_precision: fp16/bf16 は float32 より速い。
    mp_factor = 0.80 if mixed_precision in ("fp16", "bf16") else 1.0
    # batch^0.6: GPU並列効率により亜線形スケール。batch*にするとsteps/batchと打ち消しあってゼロ効果になる。
    sec_factor = (batch ** 0.6) * reso_factor * model_factor * opt_factor * rank_factor * xformers_factor * gc_factor * mp_factor

    if not backend_ready:
        # シミュレーション: SimulatedBackend の time.sleep(0.5) が厳密値
        # (以前はグローバルなkohya_readyのみで判定していたため、musubiのみ
        # 接続済みでkohyaが未接続の場合に実際は学習可能なのに"simulated"と
        # 誤表示していた)
        sec_per_step = 0.5
        overhead = 0.0
        source = "simulation"
        confidence = "high"
        base_sit = 0.5
    else:
        base_sit = gpu_base_sit(gpu_name)
        source = "heuristic"
        confidence = "low" if gpu_name is None else "medium"

        calib = load_calib(gpu_name, resolved_family, settings)
        if calib is not None:
            base_sit = calib
            source = "calibrated"
            confidence = "high"

        sec_per_step = base_sit * sec_factor

        # 学習中は status と同一の sec/step(live→DB永続EMA→ログ)を使い、残り時間を一致させる
        best = best_sec_per_step(project_id, active_run_id)
        if best is not None:
            sec_per_step = best
            source = "live"
            confidence = "high"

        # 一度きりのオーバーヘッド: モデルロード + latent キャッシュ
        overhead = 25.0 + images * reso_factor * model_factor * 0.05

    eta_seconds = int(round(total_steps * sec_per_step + overhead))
    sec_per_epoch = int(round(steps_per_epoch * sec_per_step))
    # 学習中は残り時間も返す（学習制御パネルの「残り」と同じ計算式で一致させる）
    remaining_seconds = None
    done_steps = None
    if active_done is not None:
        done_steps = active_done
        remaining_seconds = int(max(0.0, (total_steps - active_done) * sec_per_step))

    return {
        "project_id": project_id,
        "images": images,
        "batch_size": batch,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "done_steps": done_steps,
        "remaining_seconds": remaining_seconds,
        "epochs": epochs,
        "is_sdxl": is_sdxl,
        "effective_resolution": eff_res,
        "gpu_name": gpu_name,
        "mode": training_backend_name if backend_ready else "simulated",
        "model_family": resolved_family,
        "sec_per_step": round(sec_per_step, 4),
        "sec_per_epoch": sec_per_epoch,
        "overhead_seconds": int(round(overhead)),
        "eta_seconds": eta_seconds,
        "source": source,
        "confidence": confidence,
        "ready": images > 0,
        "message": (
            "選択画像が 0 枚です" if images == 0
            else f"{source} 推定（{(spec.display_name if spec else resolved_family.upper())} / {eff_res}px / batch{batch}）"
        ),
    }


@router.get("/model-specs")
def get_model_specs() -> dict:
    """registry に登録済みの全モデルスペックを返す。GUI 側のモデル選択に使用。"""
    specs = list_specs()
    return {
        "specs": [
            {
                "model_family": s.model_family,
                "display_name": s.display_name,
                "backend": s.backend,  # 後方互換のため維持（内部識別子。UI表示には使わない）
                "training_backend": s.training_backend,
                "preview_backend": s.preview_backend,
                "supports": sorted(s.supports),
                "default_resolution": s.default_resolution,
                "default_preset": s.default_preset,
                "required_models": [
                    {
                        "key": rm.key,
                        "display_name": rm.display_name,
                        "required": rm.required,
                        "hf_repo": rm.hf_repo,
                        "filename": rm.filename,
                    }
                    for rm in s.required_models
                ],
                "has_cache_steps": len(s.cache_steps) > 0,
            }
            for s in specs
        ],
        "count": len(specs),
    }


@router.get("/logs/{project_id}")
def get_logs(project_id: int, lines: int = 80) -> dict:
    """最新実行の学習ログ末尾を返す"""
    _ensure_project(project_id)
    conn = get_conn()
    row = conn.execute(
        "SELECT log_path FROM training_runs WHERE project_id = ? ORDER BY id DESC LIMIT 1",
        (project_id,),
    ).fetchone()
    conn.close()

    if row is None or not row["log_path"]:
        return {"project_id": project_id, "lines": [], "log_path": None, "total_lines": 0}

    log_path = Path(row["log_path"])
    if not log_path.exists():
        return {"project_id": project_id, "lines": [], "log_path": str(log_path), "total_lines": 0}

    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.readlines()
        recent = content[-lines:]
        return {
            "project_id": project_id,
            "lines": [line.rstrip("\n") for line in recent],
            "log_path": str(log_path),
            "total_lines": len(content),
        }
    except OSError:
        return {"project_id": project_id, "lines": [], "log_path": str(log_path), "total_lines": 0}


def _cache_summary(run_id: int, project_id: int, status: str, snapshot_id: int | None, snapshot_hash: str | None) -> list[dict]:
    root = run_dir(run_id)
    cache_root = root / "cache"
    if not cache_root.is_dir():
        return []
    files = [path for path in cache_root.rglob("*") if path.is_file()]
    size_bytes = sum(path.stat().st_size for path in files)
    created_at = min((path.stat().st_ctime for path in files), default=cache_root.stat().st_ctime)
    return [{
        "run_id": run_id,
        "project_id": project_id,
        "cache_type": "training_latent",
        "dataset_snapshot_id": snapshot_id,
        "dataset_fingerprint": snapshot_hash,
        "item_count": len([path for path in (root / "train_data").rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTS]) if (root / "train_data").is_dir() else None,
        "path": str(cache_root),
        "size_bytes": size_bytes,
        "file_count": len(files),
        "created_at": datetime.fromtimestamp(created_at).isoformat(timespec="seconds"),
        "reusable": bool(files),
        "in_use": status in {"queued", "preparing", "caching", "training"},
        "consuming_run": run_id if status in {"queued", "preparing", "caching", "training"} else None,
    }]


@router.get("/cache")
def list_training_cache(project_id: int | None = None, run_id: int | None = None) -> dict:
    """List Training Cache only; Preview History is intentionally a separate surface."""
    conn = get_conn()
    try:
        clauses: list[str] = []
        params: list[int] = []
        if project_id is not None:
            clauses.append("project_id = ?"); params.append(project_id)
        if run_id is not None:
            clauses.append("id = ?"); params.append(run_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = conn.execute(
            f"SELECT id, project_id, status, dataset_snapshot_id, resolved_config_json FROM training_runs {where} ORDER BY id DESC",
            params,
        ).fetchall()
        result: list[dict] = []
        for row in rows:
            resolved = json.loads(row["resolved_config_json"] or "{}")
            result.extend(_cache_summary(int(row["id"]), int(row["project_id"]), str(row["status"]), row["dataset_snapshot_id"], resolved.get("dataset_snapshot_hash")))
        return {"project_id": project_id, "run_id": run_id, "cache": result, "preview_history_separate": True}
    finally:
        conn.close()


@router.delete("/cache/{run_id}")
def delete_training_cache(run_id: int) -> dict:
    """Delete only a run's latent cache; refuse while its run can consume it."""
    conn = get_conn()
    try:
        row = conn.execute("SELECT id, project_id, status FROM training_runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="run not found")
        if str(row["status"]) in {"queued", "preparing", "caching", "training"}:
            raise HTTPException(status_code=409, detail="学習中のTraining Cacheは削除できません")
        cache_root = run_dir(run_id) / "cache"
        if cache_root.exists():
            shutil.rmtree(cache_root)
        return {"run_id": run_id, "deleted": True, "path": str(cache_root), "source_images_preserved": True, "captions_preserved": True, "checkpoints_preserved": True, "preview_history_separate": True}
    finally:
        conn.close()


@router.post("/start")
def start(payload: TrainingStartIn) -> dict:
    _ensure_project(payload.project_id)
    if not payload.base_checkpoint_path.strip():
        raise HTTPException(status_code=400, detail="ベースモデルが選択されていません。学習を開始する前にモデルを選択してください。")
    if not Path(payload.base_checkpoint_path.strip()).exists():
        raise HTTPException(status_code=400, detail=f"ベースモデルが見つかりません: {payload.base_checkpoint_path}")
    settings_pre = _get_app_settings()
    start_model_family = detect_model_family(payload.base_checkpoint_path, payload.model_family)
    if start_model_family == "anima" and int(payload.gpu_device_id) != 1:
        raise HTTPException(
            status_code=400,
            detail="Anima BasepipeはGPU1（PCI_BUS_IDで列挙されたRTX 3090 Ti）固定です。GPU0は使用しません。",
        )
    from ..training.advanced import validate as validate_advanced
    try:
        payload.advanced = validate_advanced(payload.advanced, start_model_family, payload.model_dump())
        for key, value in payload.advanced.items():
            if key in TrainingStartIn.model_fields:
                setattr(payload, key, value)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    start_spec = get_spec(start_model_family)
    try:
        start_training_backend = resolve_training_backend(start_model_family, payload.training_engine)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    start_backend = get_training_backend(start_training_backend)
    if start_backend is None or not start_backend.is_ready(settings_pre):
        raise HTTPException(
            status_code=400,
            detail=f"model_family={start_model_family} の学習エンジンが未接続です。対応ツールのパスを確認してください。学習ジョブは作成していません。",
        )
    if start_model_family == "anima" and payload.dataset_snapshot_id is None:
        raise HTTPException(status_code=400, detail="Anima学習にはsealed済みDataset Snapshotが必要です")
    conn = get_conn()
    preview_profile_snapshot_id = payload.preview_profile_snapshot_id
    if start_model_family == "anima" and preview_profile_snapshot_id is None:
        draft_row = conn.execute(
            "SELECT training_config_json FROM projects WHERE id = ?", (payload.project_id,)
        ).fetchone()
        try:
            saved_draft = json.loads(draft_row["training_config_json"] or "{}") if draft_row else {}
        except (TypeError, ValueError):
            saved_draft = {}
        if saved_draft.get("dataset_snapshot_id") == payload.dataset_snapshot_id:
            preview_profile_snapshot_id = saved_draft.get("preview_profile_snapshot_id")
    snapshot_row = None
    snapshot_asset_rows = []
    gpu_mapping = None
    gpu_memory = None
    gpu_wait_reason = ""
    if payload.dataset_snapshot_id is not None:
        snapshot_row = conn.execute(
            "SELECT id, project_id, snapshot_hash, status, item_count FROM basepipe_dataset_snapshots "
            "WHERE id = ? AND project_id = ?",
            (payload.dataset_snapshot_id, payload.project_id),
        ).fetchone()
        if snapshot_row is None or snapshot_row["status"] != "sealed":
            conn.close()
            raise HTTPException(status_code=400, detail="指定Dataset Snapshotが存在しないかsealed状態ではありません")
        snapshot_asset_rows = fetch_snapshot_inputs(conn, payload.dataset_snapshot_id)
        if len(snapshot_asset_rows) != int(snapshot_row['item_count'] or 0):
            conn.close()
            raise HTTPException(400, detail="確定した画像件数と現在のDataset Snapshot件数が不一致です")
        existing_snapshot_assets = [r for r in snapshot_asset_rows if Path(r["file_path"]).is_file()]
        if not existing_snapshot_assets:
            conn.close()
            raise HTTPException(status_code=400, detail="Dataset Snapshot内に実在する画像がありません")
        if start_model_family == "anima":
            invalid_snapshot_assets = []
            for row in snapshot_asset_rows:
                if row["asset_type"] != "image":
                    invalid_snapshot_assets.append("非画像アセット")
                if row["review_status"] != "approved":
                    invalid_snapshot_assets.append("未承認アセット")
                if not int(row["training_enabled"] or 0):
                    invalid_snapshot_assets.append("学習無効アセット")
                approval_issue = snapshot_user_approval_issue(row)
                if approval_issue:
                    invalid_snapshot_assets.append(approval_issue)
                integrity_ok, integrity_detail = verify_snapshot_input(row)
                if not integrity_ok:
                    invalid_snapshot_assets.append(integrity_detail)
                if not str(row["caption_at_snapshot"] or "").strip():
                    invalid_snapshot_assets.append("caption欠落")
            if invalid_snapshot_assets:
                conn.close()
                raise HTTPException(
                    status_code=400,
                    detail="Anima学習Snapshotに不備があります: " + ", ".join(sorted(set(invalid_snapshot_assets))),
                )
            gpu_mapping = resolve_cuda_index(1, resolve_python(settings_pre))
            if not gpu_mapping.get("verified"):
                conn.close()
                raise HTTPException(
                    status_code=400,
                    detail=f"物理GPU1と対象PythonのCUDA列挙を照合できないためAnima学習を開始できません: {gpu_mapping.get('reason', '不明')}",
                )
            gpu_memory = _query_gpu_memory(1)
            if gpu_memory is None:
                if payload.queue_if_busy:
                    gpu_wait_reason = "GPU1のVRAM実測に失敗したため、安全な待機キューで再評価します"
                else:
                    conn.close()
                    raise HTTPException(status_code=400, detail="Anima学習を開始できません: GPU1のVRAM実測に失敗しました")
            elif gpu_memory["free_mb"] < _anima_required_free_mb(payload.training_memory_mode):
                detail = (
                    f"物理GPU1の空きVRAMが{gpu_memory['free_mb']} MiBで、"
                    f"安全閾値{_anima_required_free_mb(payload.training_memory_mode)} MiB未満です"
                )
                if payload.queue_if_busy:
                    gpu_wait_reason = detail
                else:
                    conn.close()
                    raise HTTPException(status_code=400, detail=f"Anima学習を開始できません: {detail}。")
            occupancy = _query_gpu_foreign_processes(1)
            foreign_detail = _anima_foreign_gpu_detail(occupancy)
            if foreign_detail:
                if payload.queue_if_busy:
                    gpu_wait_reason = foreign_detail
                else:
                    conn.close()
                    raise HTTPException(status_code=409, detail=foreign_detail)
    if start_model_family == "anima":
        if preview_profile_snapshot_id is None:
            conn.close()
            raise HTTPException(status_code=400, detail="Anima学習にはDatasetと同時に固定したPreview Profile Snapshotが必要です")
        profile_snapshot = conn.execute(
            "SELECT id, snapshot_hash FROM basepipe_preview_profile_snapshots WHERE id = ? AND project_id = ?",
            (preview_profile_snapshot_id, payload.project_id),
        ).fetchone()
        if profile_snapshot is None:
            conn.close()
            raise HTTPException(status_code=400, detail="指定Preview Profile SnapshotがこのProjectに存在しません")
    # 同一Projectがqueued/training状態で既に投入済みなら多重投入を拒否する
    # (仕様上正当なResume/新規再学習はこの経路(/start)を通らず /resume を使う)。
    existing_active = conn.execute(
        "SELECT id, status FROM training_runs WHERE project_id = ? AND status IN ('queued','training') "
        "ORDER BY id DESC LIMIT 1",
        (payload.project_id,),
    ).fetchone()
    if existing_active is not None:
        conn.close()
        raise HTTPException(
            status_code=400,
            detail=(
                f"このProjectは既に学習中またはQueue投入済みです"
                f"(run_id={existing_active['id']}, status={existing_active['status']})。"
                f"完了・失敗・キャンセルされるまで新規投入できません。"
            ),
        )
    # NOTE: 以前は selected=1 のみでカウントしており、ファイルが実在しないDB上の
    # 不整合行(テスト汚染等)も学習画像として数えてしまっていた(実データ監査で発見:
    # Krea2プロジェクトでDB上selected=1が50件、実在ファイルは49件)。
    # /training/estimate・/training/preflight と同じ基準(ファイル実在確認)に統一する。
    dataset_rows = conn.execute(
        "SELECT file_path FROM dataset_items WHERE project_id = ? AND selected = 1",
        (payload.project_id,),
    ).fetchall()
    dataset_count = sum(1 for r in dataset_rows if Path(r["file_path"]).exists())
    if snapshot_row is not None:
        dataset_count = len([r for r in snapshot_asset_rows if Path(r["file_path"]).is_file()])
    # このmodel_familyが実際に使うbackendが接続済みか(グローバルなkohya_readyのみでは
    # なく、start_training_backendに対応する接続状態)。/training/estimate と同じ基準。
    kohya_ready_pre = is_kohya_ready(settings_pre)
    backend_ready_pre = (
        is_musubi_ready(settings_pre) if start_training_backend == "musubi"
        else get_training_backend(start_training_backend).is_ready(settings_pre)
        if start_training_backend not in {"sdxl", "anima"} and get_training_backend(start_training_backend)
        else kohya_ready_pre
    )
    if payload.preview_base_checkpoint_path:
        path=Path(payload.preview_base_checkpoint_path)
        if not path.is_file() or detect_model_family(str(path),'auto') != start_model_family:
            conn.close()
            raise HTTPException(status_code=400,detail='プレビュー用モデルは実在する同系統のモデルを指定してください')
    from .evaluation import read_reference, digest as evaluation_digest
    evaluation_reference=read_reference(payload.project_id,conn)
    if evaluation_reference:
        if payload.dataset_snapshot_id is None:
            conn.close()
            raise HTTPException(409,'評価用画像を分離するため、画像一覧で学習対象を確定してください')
        inputs=snapshot_asset_rows if payload.dataset_snapshot_id is not None else dataset_rows
        if any(evaluation_digest(row['file_path'])==evaluation_reference['sha256'] for row in inputs):
            conn.close()
            raise HTTPException(409,'評価用画像が学習対象に含まれています。学習対象を更新してください')
    batch = max(1, int(payload.train_batch_size))
    # batch分割(ceil(images*repeats/batch))は、batched dataloaderを使う実trainer
    # 全般(kohya/musubi共通)の基本動作であり、kohya固有ではない(実データ検証:
    # 実行済みKrea2(musubi)runのDB保存済みsteps_per_epochはceil(images*repeats/batch)
    # と一致し、images*repeatsそのままとは一致しなかった)。実backend未接続で
    # simulationにフォールバックする場合のみimages*repeatsそのまま。
    if backend_ready_pre:
        steps_per_epoch = max(1, math.ceil(math.ceil(int(dataset_count) * int(payload.repeats) / batch) / int(payload.advanced.get("gradient_accumulation_steps", 1))))
    else:
        steps_per_epoch = max(1, int(dataset_count) * int(payload.repeats))
    conn.close()

    continuation = {}
    if payload.resume_checkpoint_id is not None:
        continuation = _validate_continuation(payload)
    config_data = {
        "allow_interactive_gpu_sharing": payload.allow_interactive_gpu_sharing if payload.allow_interactive_gpu_sharing is not None else _sharing_default(payload.project_id),
        "evaluation_reference": evaluation_reference,
        "training_goal": payload.training_goal,
        "quality_preset": payload.quality_preset,
        **continuation,
        "advanced": payload.advanced,
        "queue_if_busy": payload.queue_if_busy,
        "training_memory_mode": payload.training_memory_mode,
        "preview_gpu": payload.preview_gpu,
        "preview_backend": payload.preview_backend,
        "preview_lora_strength": payload.preview_lora_strength,
        "preview_extra_loras": [item.model_dump() for item in payload.preview_extra_loras],
        "preview_base_checkpoint_path": payload.preview_base_checkpoint_path,
        "preview_steps": payload.preview_steps,
        "preview_cfg": payload.preview_cfg,
        "gpu_wait_reason": gpu_wait_reason,
        "preset_id": payload.preset_id,
        "epochs": payload.epochs,
        "repeats": payload.repeats,
        "alpha": payload.alpha,
        "rank": payload.rank,
        "resolution": payload.resolution,
        "learning_rate": payload.learning_rate,
        "train_batch_size": batch,
        "save_every_n_epochs": payload.save_every_n_epochs,
        "output_name": payload.output_name,
        "base_checkpoint_path": payload.base_checkpoint_path,
        "train_data_dir": payload.train_data_dir,
        "reg_data_dir": payload.reg_data_dir,
        "optimizer": payload.optimizer,
        "scheduler": payload.scheduler,
        "min_snr_gamma": payload.min_snr_gamma,
        "mixed_precision": payload.mixed_precision,
        "save_precision": payload.save_precision,
        "xformers": payload.xformers,
        "cache_latents": payload.cache_latents,
        "cache_latents_to_disk": payload.cache_latents_to_disk,
        "gradient_checkpointing": payload.gradient_checkpointing,
        "persistent_data_loader_workers": payload.persistent_data_loader_workers,
        "max_data_loader_n_workers": payload.max_data_loader_n_workers,
        "network_train_unet_only": payload.network_train_unet_only,
        # 解決済み(resolved)の model_family を保存する。payload.model_family は "auto" の
        # ままのことがあり、これを保存すると Preview 自動生成トリガー
        # (_generate_previews_for_checkpoints 等、config_json を直接読む後続処理)が
        # "auto"/未解決値を参照して MUSUBI_PREVIEW_SUPPORTED_FAMILIES と一致せず、
        # 何も生成しないまま静かに終了するバグを生む。start_model_family
        # (detect_model_family 済み)を保存し、run 全体で一貫した値にする。
        "model_family": start_model_family,
        "training_engine": payload.training_engine,
        "dataset_source": payload.dataset_source,
        "gpu_device_id": payload.gpu_device_id,
        "vae_path": payload.vae_path,
        "text_encoder_path": payload.text_encoder_path,
        "dataset_snapshot_id": payload.dataset_snapshot_id,
        "preview_profile_snapshot_id": preview_profile_snapshot_id,
        "gpu_mapping": gpu_mapping,
        "gpu_memory": gpu_memory,
    }
    requested_config = payload.model_dump()
    resolved_config = {
        **config_data,
        "resolved_training_backend": start_training_backend,
        "dataset_item_count": dataset_count,
        "total_steps": int(payload.epochs) * int(steps_per_epoch),
        "resolved_train_data_dir": payload.train_data_dir,
        "dataset_snapshot_id": payload.dataset_snapshot_id,
        "preview_profile_snapshot_id": preview_profile_snapshot_id,
        "preview_profile_snapshot_hash": profile_snapshot["snapshot_hash"] if start_model_family == "anima" else None,
        "dataset_snapshot_hash": snapshot_row["snapshot_hash"] if snapshot_row is not None else None,
        "gpu_wait_reason": gpu_wait_reason or None,
    }

    conn = get_conn()
    concept_rows = conn.execute(
        "SELECT concept_type, trigger_token, name FROM basepipe_concepts "
        "WHERE project_id = ? AND TRIM(trigger_token) <> '' ORDER BY id",
        (payload.project_id,),
    ).fetchall()
    concepts = [
        {"type": str(row["concept_type"]), "trigger": str(row["trigger_token"]), "name": str(row["name"])}
        for row in concept_rows
    ]
    resolved_config["concepts"] = concepts
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO training_runs(
            project_id, status, stop_mode, latest_checkpoint_path, config_json,
            requested_config_json, resolved_config_json, dataset_snapshot_id,
            current_epoch, current_step, total_epochs, steps_per_epoch
        )
        VALUES (?, 'queued', NULL, NULL, ?, ?, ?, ?, 0, 0, ?, ?)
        """,
        (
            payload.project_id,
            json.dumps(config_data, ensure_ascii=False),
            json.dumps(requested_config, ensure_ascii=False),
            json.dumps(resolved_config, ensure_ascii=False),
            payload.dataset_snapshot_id,
            payload.epochs,
            steps_per_epoch,
        ),
    )
    run_id = cur.lastrowid
    manifest_path = write_run_manifest(
        int(run_id),
        payload.project_id,
        requested_config,
        resolved_config,
        {
            "source": payload.dataset_source,
            "item_count": dataset_count,
            "caption_count": dataset_count,
            "dataset_snapshot_id": payload.dataset_snapshot_id,
            "preview_profile_snapshot_id": preview_profile_snapshot_id,
            "preview_profile_snapshot_hash": profile_snapshot["snapshot_hash"] if start_model_family == "anima" else None,
            "dataset_snapshot_hash": snapshot_row["snapshot_hash"] if snapshot_row is not None else None,
            "manifest_path": (
                str(Path(payload.train_data_dir) / ".manifest.json")
                if payload.train_data_dir else None
            ),
        },
        {
            "requested_gpu_id": payload.gpu_device_id,
            # Animaはstart直前に物理GPU1と対象PythonのCUDA列挙を照合済み。
            # 実行後のログ補正を待たず、Run作成時点のResolved環境として固定保存する。
            "cuda_visible_devices": (
                str(gpu_mapping["cuda_index"])
                if isinstance(gpu_mapping, dict) and gpu_mapping.get("verified")
                else None
            ),
            "gpu_mapping": gpu_mapping,
        },
        concepts=concepts,
    )
    cur.execute(
        "UPDATE training_runs SET manifest_path = ? WHERE id = ?",
        (str(manifest_path), run_id),
    )
    set_project_status(conn, payload.project_id, "queued")
    conn.commit()
    conn.close()

    # Training Queue: GPUが空いていれば即座に昇格・起動する。他runが実行中なら
    # queuedのまま留まり、そのrunの完了/失敗/キャンセル時に自動的に開始される
    # (_run_training_backend の finally が _dispatch_next_queued_job を呼ぶ)。
    started_run_id = _dispatch_next_queued_job()
    queued = started_run_id != run_id
    if queued and payload.queue_if_busy and start_model_family == "anima":
        ensure_training_wait_monitor()

    # model_family / エンジン可否は関数冒頭で判定済み（未接続なら既に 400 で弾かれている）。
    # mode の値自体（"musubi"/"kohya"/"simulated"）は既存 API・フロントエンドとの
    # 後方互換のため維持し、判定ロジックのみ ModelSpec.training_backend に統一する。
    model_family = start_model_family
    if start_training_backend == "ai_toolkit":
        mode = "ai_toolkit"
    elif start_training_backend != "sdxl":
        mode = start_training_backend
    elif kohya_ready_pre:
        mode = "kohya"
    else:
        mode = "simulated"

    final_status = "training" if not queued else "queued"
    queue_pos = None
    if queued:
        conn = get_conn()
        queue_pos = _queue_position(conn, run_id)
        conn.close()

    return {
        "run_id": run_id,
        "project_id": payload.project_id,
        "preset_id": payload.preset_id,
        "status": final_status,
        "queue_position": queue_pos,
        "allow_interactive_gpu_sharing": bool(config_data.get("allow_interactive_gpu_sharing")),
        "mode": mode,
        "model_family": model_family,
        "total_epochs": payload.epochs,
        "steps_per_epoch": steps_per_epoch,
        "dataset_images": dataset_count,
        "repeats": payload.repeats,
        "alpha": payload.alpha,
        "message": (
            f"学習を開始しました（{mode} モード）" if not queued
            else (
                f"GPU1安全条件を満たすまで待機キューで自動再評価します（待機{queue_pos}番目）"
                if payload.queue_if_busy and start_model_family == "anima"
                else f"他の学習が実行中のためQueueへ投入しました（待機{queue_pos}番目）"
            )
        ),
    }


@router.post("/stop-now")
def stop_now(payload: TrainingControlIn) -> dict:
    """学習をキャンセルする。runが 'queued'（未開始）か 'training'（実行中）かで
    扱いを分ける — queuedならsubprocessは一切存在しないため即座にDB状態のみ
    更新する。runningなら既存どおりsubprocessをterminateし、そのrunのスレッドの
    finally節が次のqueued runを自動的に開始する。
    """
    _ensure_project(payload.project_id)
    conn = get_conn()
    row = latest_run(conn, payload.project_id)
    if row is None:
        conn.close()
        raise HTTPException(status_code=400, detail="no training run")

    if row["status"] == "queued":
        cancelled = _cancel_queued(conn, int(row["id"]))
        set_project_status(conn, payload.project_id, "idle")
        conn.commit()
        conn.close()
        if not cancelled:
            raise HTTPException(status_code=400, detail="run is not queued (race: already started)")
        return {
            "project_id": payload.project_id,
            "mode": "stop_now",
            "status": "paused",
            "message": "queued job cancelled before it started (no subprocess was ever launched)",
        }

    conn.execute(
        """
        UPDATE training_runs
        SET status = 'paused', stop_mode = 'now', updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (row["id"],),
    )
    set_project_status(conn, payload.project_id, "paused")
    conn.commit()
    conn.close()

    # 学習サブプロセスを終了
    with RUNNER_LOCK:
        proc = RUNNER_PROCESSES.get(payload.project_id)
        if proc and proc.poll() is None:
            proc.terminate()

    return {
        "project_id": payload.project_id,
        "mode": "stop_now",
        "status": "paused",
        "message": "stopped at nearest safe point",
    }


@router.post("/stop-at-epoch")
def stop_at_epoch(payload: TrainingControlIn) -> dict:
    _ensure_project(payload.project_id)
    conn = get_conn()
    row = latest_run(conn, payload.project_id)
    if row is None:
        conn.close()
        raise HTTPException(status_code=400, detail="no training run")
    if row["status"] != "training":
        conn.close()
        raise HTTPException(status_code=400, detail="run is not training")
    config = json.loads(row["config_json"] or "{}")
    if row["stop_mode"] != "epoch":
        last = conn.execute("SELECT COALESCE(MAX(id),0) AS id FROM checkpoints WHERE run_id=?", (row["id"],)).fetchone()
        config["epoch_stop_after_checkpoint_id"] = int(last["id"])
        config["epoch_stop_requested_at_ns"] = time.time_ns()
    conn.execute(
        "UPDATE training_runs SET stop_mode='epoch',config_json=?,updated_at=CURRENT_TIMESTAMP WHERE id=?",
        (json.dumps(config, ensure_ascii=False), row["id"]),
    )
    conn.commit()
    conn.close()
    return {
        "project_id": payload.project_id,
        "mode": "stop_at_epoch",
        "status": "training",
        "message": "epoch stop reserved",
    }


@router.post("/resume")
def resume(payload: TrainingControlIn) -> dict:
    """既存checkpointからの継続学習を要求する。

    重要: Backendが重み継続(kohyaのnetwork_weights等)をサポートしない場合、
    単にstatusを'training'へ戻してepoch 0から再学習させると、ユーザーは
    「中断前の続きから再開された」と誤認する。そのため、まずbackendへ
    supports_resume()を問い合わせ、未対応なら明示エラーで即座に失敗させる
    (Architecture/Backend境界の判断はbackend自身に委ね、ここでは
    model_familyのif分岐を増やさない)。
    """
    _ensure_project(payload.project_id)
    conn = get_conn()
    row = latest_run(conn, payload.project_id)
    if row is None:
        conn.close()
        raise HTTPException(status_code=400, detail="no training run")
    if row["status"] == "completed":
        conn.close()
        raise HTTPException(status_code=400, detail="run already completed")
    if row['status'] in {'queued', 'training', 'running', 'waiting', 'waiting_gpu', 'stopping'}:
        conn.close()
        raise HTTPException(status_code=409, detail="この学習は既に実行中または待機中です。重複して再開しません。")

    run_id = int(row["id"])
    try:
        config_json = json.loads(row["config_json"] or "{}")
    except Exception:
        config_json = {}
    from ..training.learning_rates import resolve_learning_rates
    try:
        resolve_learning_rates(config_json, reject_conflict=True)
    except (ValueError,TypeError) as exc:
        conn.close()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    base_ckpt = config_json.get("base_checkpoint_path", "")
    explicit_family = config_json.get("model_family", "auto")
    model_family = detect_model_family(base_ckpt, explicit_family)
    spec = get_spec(model_family)
    training_backend_name = resolve_training_backend(model_family, config_json.get("training_engine", "auto"))
    backend = get_training_backend(training_backend_name)

    if backend is None or not backend.supports_resume():
        conn.close()
        raise HTTPException(
            status_code=400,
            detail=(
                f"model_family={model_family} の学習エンジンは既存checkpointからの"
                f"継続学習に対応していません。新規の学習として開始してください"
                f"（epoch 0からの再学習を「Resume」と偽装することはできません）。"
            ),
        )

    # Prefer this run. Historical fallback must match model, sealed dataset and LoRA shape.
    ckpt_row = select_resume_checkpoint(conn, payload.project_id, run_id, config_json)
    if ckpt_row is None or not Path(ckpt_row["file_path"]).exists():
        conn.close()
        raise HTTPException(
            status_code=400,
            detail="現在の実行、またはモデル・確定データセット・Rankが一致する継続元checkpointがありません。",
        )
    resume_from = str(ckpt_row["file_path"])
    config_json["resume_from_checkpoint"] = resume_from
    config_json["resume_from_run_id"] = int(ckpt_row["run_id"])

    # Training Queueへ投入する（直接Threadを起動しない）。GPUが空いていれば
    # dispatch_next()が即座に昇格・起動し、他runが実行中ならqueuedのまま
    # 待機し、そのrunの終了時に自動的に開始される。
    conn.execute(
        """
        UPDATE training_runs
        SET status = 'queued', stop_mode = NULL, config_json = ?, updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (json.dumps(config_json), run_id),
    )
    set_project_status(conn, payload.project_id, "queued")
    conn.commit()
    conn.close()

    started_run_id = _dispatch_next_queued_job()
    queued = started_run_id != run_id
    conn = get_conn()
    queue_pos = _queue_position(conn, run_id) if queued else None
    conn.close()

    return {
        "project_id": payload.project_id,
        "run_id": run_id,
        "status": "queued" if queued else "training",
        "queue_position": queue_pos,
        "resume_from_checkpoint": resume_from,
        "message": (
            (
                "resume accepted（既存LoRA重みからの継続学習。optimizer状態・"
                "epochカウンタは継続されません）"
            ) if not queued else
            f"他の学習が実行中のためResumeをQueueへ投入しました（待機{queue_pos}番目）"
        ),
    }


@router.post("/reset")
def reset(payload: TrainingControlIn) -> dict:
    """プレビューのキャッシュを削除し、進捗表示を Step 0 に戻す。

    古い学習タスクのプレビューが残り続ける問題を解消するためのキャッシュ削除ボタン用。
    破壊的な全消去ではなく、以下のみを行う:
      - preview_samples 行とプレビュー画像ファイル / outputs/previews ディレクトリを削除
      - 最新 run の進捗カウンタ（epoch/step/loss 等）を 0 にリセットし、ログを空にする
        → 進捗バーが Step 0 / 0% に戻る
    checkpoints（学習履歴）・training_runs・lora_assets（資産ライブラリ）は保持する。
    """
    project = _ensure_project(payload.project_id)
    project_id = payload.project_id

    # 1. 実行中の学習サブプロセスを終了
    with RUNNER_LOCK:
        proc = RUNNER_PROCESSES.get(project_id)
        if proc and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass
        RUNNER_PROCESSES.pop(project_id, None)
        RUNNER_THREADS.pop(project_id, None)

    # 2. タイミングキャッシュをクリア
    STEP_TIMING.pop(project_id, None)

    conn = get_conn()
    deleted = {"previews": 0, "files": 0}

    # 3. プレビュー画像ファイル削除 + preview_samples 行削除（checkpoints は残す）
    prev_rows = conn.execute(
        """
        SELECT ps.image_path
        FROM preview_samples ps
        JOIN checkpoints c ON ps.checkpoint_id = c.id
        WHERE c.project_id = ?
        """,
        (project_id,),
    ).fetchall()
    for pr in prev_rows:
        img = pr["image_path"]
        if img:
            try:
                p = Path(str(img))
                if p.exists() and p.is_file():
                    p.unlink()
                    deleted["files"] += 1
            except Exception:
                pass
    conn.execute(
        """
        DELETE FROM preview_samples
        WHERE checkpoint_id IN (SELECT id FROM checkpoints WHERE project_id = ?)
        """,
        (project_id,),
    )
    deleted["previews"] = len(prev_rows)

    # 4. 最新 run の進捗を Step 0 にリセット + ログを空にする
    latest = latest_run(conn, project_id)
    if latest is not None:
        log_path = latest["log_path"] if "log_path" in latest.keys() else None
        if log_path:
            try:
                lp = Path(str(log_path))
                if lp.exists() and lp.is_file():
                    lp.write_text("", encoding="utf-8")
            except Exception:
                pass
        conn.execute(
            """
            UPDATE training_runs
            SET status = 'idle', stop_mode = NULL, current_epoch = 0, current_step = 0,
                loss = NULL, sec_per_step_ema = NULL, step1_at = NULL,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (latest["id"],),
        )

    # 5. プロジェクト状態を idle に戻す
    set_project_status(conn, project_id, "idle")
    conn.commit()
    conn.close()

    # 6. outputs_dir 配下の previews/ を削除（再スキャンによる復活を防ぐ）
    try:
        outputs_dir = project.get("outputs_dir")
        if outputs_dir:
            previews_dir = Path(outputs_dir) / "previews"
            if previews_dir.exists():
                shutil.rmtree(previews_dir, ignore_errors=True)
    except Exception:
        pass

    return {
        "project_id": project_id,
        "status": "idle",
        "deleted": deleted,
        "message": "previews cleared and progress reset to step 0",
    }


@router.get("/status")
def status(project_id: int) -> dict:
    _ensure_project(project_id)
    conn = get_conn()
    row = conn.execute(
        """
        SELECT id, project_id, status, stop_mode, latest_checkpoint_path, started_at, updated_at,
               current_epoch, current_step, total_epochs, steps_per_epoch, loss, log_path,
               sec_per_step_ema, step1_at, failure_code, failure_message, failure_stage,
               config_json, resolved_config_json,
               manifest_path, dataset_snapshot_id, current_stage, stage_history_json
        FROM training_runs
        WHERE project_id = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (project_id,),
    ).fetchone()
    conn.close()

    settings = _get_app_settings()
    mode = "kohya" if is_kohya_ready(settings) else "simulated"
    if row is not None:
        try:
            resolved_mode = json.loads(row["resolved_config_json"] or "{}").get("resolved_training_backend")
            if resolved_mode:
                mode = str(resolved_mode)
        except (ValueError, TypeError):
            pass

    if row is None:
        return {
            "project_id": project_id,
            "run_id": None,
            "status": "idle",
            "queue_position": None,
            "epoch": 0,
            "step": 0,
            "total_epochs": 0,
            "steps_per_epoch": 0,
            "stop_mode": None,
            "latest_checkpoint_path": None,
            "loss": None,
            "mode": mode,
            "message": "no run yet",
            "failure": None,
            "manifest_path": None,
            "dataset_snapshot_id": None,
            "current_stage": "planning",
            "stage_history": [],
        }

    # ── 実態をログ／ディスク／プロセスから再構成（DB列のみに依存しない） ──
    rs = reconstruct_run_state(row)
    total_steps = rs["total_steps"]
    done_steps = rs["done_steps"]
    progress_percent = (done_steps / total_steps * 100.0) if total_steps > 0 else 0.0

    eta_seconds = None
    remaining = max(0, total_steps - done_steps)
    # ETA は「実際に学習が走っている」ときだけ意味を持つ。error/paused/idle では残り時間を出さない。
    if rs["status"] == "training":
        # estimate と同一の優先順位(live→DB永続EMA→ログ)で sec/step を取得し一致させる
        sps = best_sec_per_step(project_id, int(row["id"]))
        if sps and done_steps > 0:
            eta_seconds = int(max(0.0, remaining * sps))
        elif done_steps >= 10:
            try:
                step1_raw = row["step1_at"] if "step1_at" in row.keys() else None
                base_at_str = step1_raw or row["started_at"]
                base_at = datetime.fromisoformat(str(base_at_str).replace(" ", "T"))
                elapsed = max(1.0, (datetime.utcnow() - base_at).total_seconds())
                base_steps = done_steps if step1_raw else max(1, done_steps - 5)
                speed = base_steps / elapsed if elapsed > 0 else 0.0
                if speed > 0:
                    eta_seconds = int(max(0.0, remaining / speed))
            except ValueError:
                eta_seconds = None
    elif rs["status"] == "completed":
        eta_seconds = 0

    msg = f"status ({mode} mode)"
    if rs["note"]:
        msg = rs["note"]

    queue_pos = None
    if rs["status"] == "queued":
        conn = get_conn()
        queue_pos = _queue_position(conn, int(row["id"]))
        conn.close()
        try:
            from ..training.runtime.state import run_dir
            admission=json.loads((run_dir(int(row['id']))/'admission.json').read_text(encoding='utf8'))
            reason=admission.get('reason') or '開始条件を確認中'
            msg=f"待機 {queue_pos}番目：{reason}（自動確認 {admission.get('probe_count',0)}回）"
        except (OSError,ValueError):
            reason=json.loads(row['config_json'] or '{}').get('gpu_wait_reason') or '開始条件を確認中'
            msg=f"待機 {queue_pos}番目：{reason}"

    return {
        "project_id": project_id,
        "run_id": row["id"],
        "status": rs["status"],
        "allow_interactive_gpu_sharing": bool(json.loads(row["config_json"] or "{}").get("allow_interactive_gpu_sharing")),
        "queue_position": queue_pos,
        "epoch": rs["epoch"],
        "step": rs["step"],
        "total_epochs": rs["total_epochs"],
        "steps_per_epoch": rs["steps_per_epoch"],
        "total_steps": total_steps,
        "done_steps": done_steps,
        "progress_percent": round(progress_percent, 2),
        "eta_seconds": eta_seconds,
        "process_alive": rs["alive"],
        "stop_mode": row["stop_mode"],
        "latest_checkpoint_path": row["latest_checkpoint_path"],
        "loss": rs["loss"],
        "mode": mode,
        "log_path": row["log_path"],
        "started_at": row["started_at"],
        "updated_at": row["updated_at"],
        "message": msg,
        "failure": (
            {
                "code": row["failure_code"],
                "message": row["failure_message"],
                "stage": row["failure_stage"],
            }
            if row["failure_code"] or row["failure_message"] else None
        ),
        "manifest_path": row["manifest_path"],
        "dataset_snapshot_id": row["dataset_snapshot_id"],
        "current_stage": _effective_stage(int(row["id"]), str(row["status"]), row["current_stage"]),
        "stage_history": json.loads(row["stage_history_json"] or "[]"),
        "preview_activity": _preview_activity_safe(int(row["id"]), rs["status"]),
    }


def _preview_activity_safe(run_id: int, run_status: str) -> dict:
    """Additive, read-only preview progress block. Must never break /status."""
    try:
        from ..training.runtime.preview_activity import build_preview_activity
        from ..training.runtime.state import run_dir

        conn = get_conn()
        try:
            return build_preview_activity(conn, run_id, str(run_status), run_dir(run_id))
        finally:
            conn.close()
    except Exception:  # noqa: BLE001
        return {"active": False, "phase": "none", "jobs": [], "run_id": run_id}


@router.get("/runs")
def list_runs(project_id: int | None = None, limit: int = 50) -> dict:
    """Run一覧。Run中心UIが既存Training APIから移行できる安定した入口。"""
    limit = max(1, min(limit, 200))
    conn = get_conn()
    if project_id is None:
        rows = conn.execute(
            "SELECT id, project_id, status, current_epoch, current_step, total_epochs, "
            "started_at, updated_at, failure_code, failure_message, manifest_path, dataset_snapshot_id, current_stage "
            "FROM training_runs ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, project_id, status, current_epoch, current_step, total_epochs, "
            "started_at, updated_at, failure_code, failure_message, manifest_path, dataset_snapshot_id, current_stage "
            "FROM training_runs WHERE project_id=? ORDER BY id DESC LIMIT ?",
            (project_id, limit),
        ).fetchall()
    conn.close()
    runs = []
    for row in rows:
        item = dict(row)
        item["current_stage"] = _effective_stage(int(item["id"]), str(item["status"]), item.get("current_stage"))
        runs.append(item)
    return {"runs": runs}


@router.get("/runs/{run_id}")
def get_run(run_id: int) -> dict:
    """Return one Run as the canonical Run-detail entry point.

    The detail response deliberately contains both the live status projection
    and the immutable evidence projection.  Callers can render a Run without
    guessing whether a value came from the current process, the manifest, or
    the requested/resolved configuration.
    """
    conn = get_conn()
    row = conn.execute(
        "SELECT id, project_id, status, current_epoch, current_step, total_epochs, "
        "started_at, updated_at, failure_code, failure_message, failure_stage, "
        "manifest_path, dataset_snapshot_id, current_stage "
        "FROM training_runs WHERE id = ?",
        (run_id,),
    ).fetchone()
    conn.close()
    if row is None:
        raise HTTPException(status_code=404, detail="run not found")

    live_status = status(int(row["project_id"]))
    # A project status endpoint represents the latest active Run.  If an older
    # Run is requested, keep the response truthful by exposing its own scalar
    # state instead of silently attaching the latest Run's progress to it.
    if live_status.get("run_id") != int(run_id):
        live_status = {
            "project_id": int(row["project_id"]),
            "run_id": int(run_id),
            "status": row["status"],
            "epoch": row["current_epoch"] or 0,
            "step": row["current_step"] or 0,
            "total_epochs": row["total_epochs"] or 0,
            "started_at": row["started_at"],
            "updated_at": row["updated_at"],
            "failure": (
                {
                    "code": row["failure_code"],
                    "message": row["failure_message"],
                    "stage": row["failure_stage"],
                }
                if row["failure_code"] or row["failure_message"] else None
            ),
            "manifest_path": row["manifest_path"],
            "dataset_snapshot_id": row["dataset_snapshot_id"],
            "current_stage": _effective_stage(int(run_id), str(row["status"]), row["current_stage"]),
        }

    return {
        "run": dict(row),
        "status": live_status,
        "evidence": run_evidence(run_id),
    }


def _parse_run_time(value: object) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed


def _seconds_between(start: object, end: object) -> float | None:
    left = _parse_run_time(start)
    right = _parse_run_time(end)
    if left is None or right is None or right < left:
        return None
    return round((right - left).total_seconds(), 3)


def _run_timing_metrics(row, run_id: int) -> dict[str, float | None]:
    """Expose measured stage timings without filling missing stages by inference."""
    try:
        history = json.loads(row["stage_history_json"] or "[]")
    except (TypeError, ValueError):
        history = []
    events = [item for item in history if isinstance(item, dict) and item.get("at")]
    started_at = row["started_at"]

    def first(stage: str, status: str | None = None) -> dict | None:
        return next((item for item in events if item.get("stage") == stage and (status is None or item.get("status") == status)), None)

    cache_start = first("cache", "running")
    training_start = first("training", "running")
    training_end = first("training", "completed") or first("training", "failed")
    first_checkpoint = first("checkpoint", "completed")
    finalize = first("finalize", "completed")
    cache_start_time = _parse_run_time(cache_start.get("at")) if cache_start else None
    next_after_cache = next(
        (
            item for item in events
            if cache_start_time
            and item is not cache_start
            and (event_time := _parse_run_time(item.get("at")))
            and event_time > cache_start_time
        ),
        None,
    )

    first_preview_at = None
    conn = get_conn()
    try:
        preview_row = conn.execute(
            "SELECT MIN(completed_at) AS completed_at FROM preview_jobs WHERE run_id=? AND status='succeeded'",
            (run_id,),
        ).fetchone()
        first_preview_at = preview_row["completed_at"] if preview_row else None
    finally:
        conn.close()

    return {
        "reference_setup_seconds": None,
        "dataset_generation_seconds": None,
        "human_review_seconds": None,
        "preprocess_seconds": None,
        "cache_seconds": _seconds_between(cache_start.get("at"), next_after_cache.get("at")) if cache_start and next_after_cache else None,
        "time_to_first_checkpoint_seconds": _seconds_between(started_at, first_checkpoint.get("at")) if first_checkpoint else None,
        "time_to_first_preview_seconds": _seconds_between(started_at, first_preview_at),
        "total_training_seconds": _seconds_between(training_start.get("at"), training_end.get("at")) if training_start and training_end else (_seconds_between(training_start.get("at"), finalize.get("at")) if training_start and finalize else None),
        "total_user_interaction_seconds": None,
    }


@router.get("/runs/{run_id}/evidence")
def run_evidence(run_id: int) -> dict:
    """Return immutable requested/resolved data plus observed values if available.

    Missing trainer observations remain null by design; this endpoint never
    promotes a requested or resolved value to observed implicitly.
    """
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM training_runs WHERE id = ?", (run_id,)
    ).fetchone()
    if row is None:
        conn.close()
        raise HTTPException(status_code=404, detail="run not found")
    checkpoint_evidence: list[dict] = []
    preview_totals = {"expected": 0, "succeeded": 0, "failed": 0, "pending": 0, "running": 0}
    checkpoint_rows = conn.execute(
        "SELECT id, epoch, step, file_path, validation_status, validation_detail "
        "FROM checkpoints WHERE run_id = ? ORDER BY epoch, id",
        (run_id,),
    ).fetchall()
    for checkpoint in checkpoint_rows:
        counts = conn.execute(
            "SELECT status, COUNT(*) AS count FROM preview_jobs WHERE checkpoint_id = ? GROUP BY status",
            (checkpoint["id"],),
        ).fetchall()
        summary = {key: 0 for key in preview_totals}
        for count_row in counts:
            status_key = str(count_row["status"] or "pending")
            count = int(count_row["count"] or 0)
            summary["expected"] += count
            if status_key == "succeeded":
                summary["succeeded"] += count
            elif status_key in summary:
                summary[status_key] += count
        for key in preview_totals:
            preview_totals[key] += summary[key]
        checkpoint_evidence.append({
            "id": int(checkpoint["id"]),
            "epoch": checkpoint["epoch"],
            "step": checkpoint["step"],
            "file_path": checkpoint["file_path"],
            "validation_status": checkpoint["validation_status"],
            "validation_detail": checkpoint["validation_detail"],
            "preview": summary,
        })
    conn.close()
    try:
        requested = json.loads(row["requested_config_json"] or "{}")
    except (ValueError, TypeError):
        requested = {}
    try:
        resolved = json.loads(row["resolved_config_json"] or row["config_json"] or "{}")
    except (ValueError, TypeError):
        resolved = {}
    manifest = load_run_manifest(run_id)
    observed: dict = {}
    # Only promote values read from the concrete trainer config. The run
    # request/resolution records are never used as an implicit observation.
    config_path = resolved.get("prepared_config_path")
    if config_path:
        try:
            config_text = Path(str(config_path)).read_text(encoding="utf-8")
            if Path(str(config_path)).suffix.lower() == ".toml":
                trainer_config = tomllib.loads(config_text)
            else:
                trainer_config = json.loads(config_text)
            if isinstance(trainer_config, dict):
                for source_key, observed_key in (
                    ("train_batch_size", "batch_size"), ("learning_rate", "learning_rate"),
                    ("network_dim", "rank"), ("network_alpha", "alpha"),
                    ("resolution", "resolution"), ("max_train_epochs", "epochs"),
                    ("max_train_steps", "steps"),
                ):
                    if source_key in trainer_config:
                        value = trainer_config[source_key]
                        if observed_key == "resolution" and isinstance(value, str):
                            value = int(value.split(",", 1)[0])
                        observed[observed_key] = value
        except (OSError, ValueError, TypeError):
            pass
    log_path = row["log_path"]
    if log_path:
        try:
            log_text = Path(str(log_path)).read_text(encoding="utf-8", errors="replace")
            visible_match = re.search(r"CUDA_VISIBLE_DEVICES=([^\r\n]+)", log_text)
            if visible_match and visible_match.group(1).strip().isdigit():
                observed["gpu_device_id"] = int(visible_match.group(1).strip())
            log_state = parse_log_state(log_path)
            if log_state.get("gtotal") is not None:
                observed["steps"] = log_state["gtotal"]
        except OSError:
            pass
    failure = None
    if row["failure_code"] or row["failure_message"]:
        log_state = parse_log_state(row["log_path"])
        checkpoint_conn = get_conn()
        try:
            latest_checkpoint_row = checkpoint_conn.execute(
                "SELECT id, file_path FROM checkpoints WHERE run_id = ? ORDER BY epoch DESC, id DESC LIMIT 1",
                (run_id,),
            ).fetchone()
        finally:
            checkpoint_conn.close()
        resume_available = False
        resolved_backend = str(resolved.get("resolved_training_backend") or "")
        backend = get_training_backend(resolved_backend) if resolved_backend else None
        if backend is not None and latest_checkpoint_row is not None:
            try:
                resume_available = bool(backend.supports_resume() and Path(str(latest_checkpoint_row["file_path"])).is_file())
            except Exception:  # noqa: BLE001
                resume_available = False
        excerpt = ""
        if row["log_path"]:
            try:
                lines = Path(str(row["log_path"])).read_text(encoding="utf-8", errors="replace").splitlines()
                excerpt = "\n".join(line.strip()[:300] for line in lines[-12:] if line.strip())
            except OSError:
                excerpt = ""
        failure = {
            "code": row["failure_code"],
            "message": row["failure_message"],
            "stage": row["failure_stage"],
            "at": row["failure_at"],
            "exit_code": None,
            "last_epoch": log_state.get("epoch") if log_state.get("epoch") is not None else row["current_epoch"],
            "last_step": log_state.get("gstep") if log_state.get("gstep") is not None else row["current_step"],
            "latest_checkpoint": dict(latest_checkpoint_row) if latest_checkpoint_row else None,
            "resume_available": resume_available,
            "log_excerpt": excerpt,
        }
    timing_metrics = _run_timing_metrics(row, run_id)
    return {
        "run_id": run_id,
        "status": row["status"],
        "current_stage": _effective_stage(int(row["id"]), str(row["status"]), row["current_stage"]),
        "stage_history": json.loads(row["stage_history_json"] or "[]"),
        "training_status": row["status"],
        "checkpoints": checkpoint_evidence,
        "preview": preview_totals,
        "manifest": manifest,
        "requested": requested,
        "resolved": resolved,
        "observed": observed,
        "parameters": parameter_evidence(requested, resolved, observed),
        "failure": failure,
        "timing": timing_metrics,
    }


@router.get("/runs/{run_id}/checkpoints")
def list_run_checkpoints(run_id: int) -> dict:
    """Return only the checkpoint projection for the canonical Run API."""
    conn = get_conn()
    run = conn.execute("SELECT id FROM training_runs WHERE id = ?", (run_id,)).fetchone()
    if run is None:
        conn.close()
        raise HTTPException(status_code=404, detail="run not found")
    rows = conn.execute(
        "SELECT id, project_id, run_id, epoch, step, file_path, mark, validation_status, validation_detail, created_at "
        "FROM checkpoints WHERE run_id = ? ORDER BY epoch, id",
        (run_id,),
    ).fetchall()
    conn.close()
    from ..training.runtime.metrics import checkpoint_progress
    result=[];latest_by_file={r['file_path']:r['id'] for r in rows}  # the same weights file registered twice (scanner race): keep the newest row, like the previews list does
    for row in rows:
        if latest_by_file[row['file_path']]!=row['id']:continue
        item=dict(row);item['step'],item['step_source']=checkpoint_progress(item['file_path'],item.get('step'));result.append(item)
    return {"run_id":run_id,"checkpoints":result}


@router.get('/runs/{run_id}/metrics')
def get_run_metrics(run_id: int) -> dict:
    from ..training.runtime.metrics import run_metrics
    try:return run_metrics(run_id)
    except ValueError as exc:raise HTTPException(status_code=404,detail=str(exc)) from exc

@router.get("/runs/{run_id}/logs")
def get_run_logs(run_id: int, lines: int = 80) -> dict:
    """Return log evidence for the requested Run, never the project's latest Run."""
    lines = max(1, min(lines, 500))
    conn = get_conn()
    row = conn.execute("SELECT id, project_id, log_path FROM training_runs WHERE id = ?", (run_id,)).fetchone()
    conn.close()
    if row is None:
        raise HTTPException(status_code=404, detail="run not found")
    path = Path(str(row["log_path"])) if row["log_path"] else None
    if path is None or not path.is_file():
        return {"run_id": run_id, "project_id": row["project_id"], "lines": [], "log_path": str(path) if path else None, "total_lines": 0}
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        content = []
    return {"run_id": run_id, "project_id": row["project_id"], "lines": content[-lines:], "log_path": str(path), "total_lines": len(content)}


@router.get("/queue")
def get_queue() -> dict:
    """Training Queueの全体状態を返す（現在実行中job + FIFO待機中jobs）。

    Frontendがpollingしやすいよう、project個別のstatusとは独立に
    Queue全体を俯瞰できるエンドポイントとして提供する。
    """
    conn = get_conn()
    running = _current_running(conn)
    queued = _list_queue(conn)
    conn.close()
    return {
        "running": running,
        "queued": queued,
        "queue_length": len(queued),
    }


@router.get("/resources")
def get_resources() -> dict:
    """
    §18 Resource Monitor — CPU / RAM / GPU 使用状況を返す。
    psutil で CPU/RAM、nvidia-smi で GPU を取得する。
    """
    # ── CPU / RAM (psutil) ────────────────────────────────────────────────
    try:
        import psutil
        cpu_pct = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        ram_used_gb = round(mem.used / 1024 ** 3, 2)
        ram_total_gb = round(mem.total / 1024 ** 3, 2)
        ram_pct = round(mem.percent, 1)
    except ImportError:
        cpu_pct = 0.0
        ram_used_gb = 0.0
        ram_total_gb = 0.0
        ram_pct = 0.0

    # ── GPU (pynvml → nvidia-smi フォールバック) ──────────────────────────
    gpu_list: list[dict] = []
    gpu_available = False

    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except Exception:
                temp = None
            vram_total_mb = mem_info.total // (1024 * 1024)
            vram_used_mb = mem_info.used // (1024 * 1024)
            vram_free_mb = mem_info.free // (1024 * 1024)
            gpu_list.append({
                "index": i,
                "name": name,
                "vram_used_mb": vram_used_mb,
                "vram_free_mb": vram_free_mb,
                "vram_total_mb": vram_total_mb,
                "vram_pct": round(vram_used_mb / vram_total_mb * 100, 1) if vram_total_mb else 0.0,
                "gpu_util_pct": float(util.gpu),
                "temperature": temp,
            })
        gpu_available = len(gpu_list) > 0
        pynvml.nvmlShutdown()
    except Exception:
        # pynvml 未インストール or GPU なし → nvidia-smi で試みる
        try:
            result = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=name,memory.used,memory.free,memory.total,utilization.gpu,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                for i, line in enumerate(result.stdout.strip().splitlines()):
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 6:
                        vram_used_mb = int(parts[1])
                        vram_free_mb = int(parts[2])
                        vram_total_mb = int(parts[3])
                        try:
                            temp_val: int | None = int(parts[5])
                        except ValueError:
                            temp_val = None
                        gpu_list.append({
                            "index": i,
                            "name": parts[0],
                            "vram_used_mb": vram_used_mb,
                            "vram_free_mb": vram_free_mb,
                            "vram_total_mb": vram_total_mb,
                            "vram_pct": round(vram_used_mb / vram_total_mb * 100, 1) if vram_total_mb else 0.0,
                            "gpu_util_pct": float(parts[4]),
                            "temperature": temp_val,
                        })
                gpu_available = len(gpu_list) > 0
        except Exception:
            pass

    # AnimaのGPU1 safety status is part of the live resource projection, not
    # only the Start-time gate, so the GUI can explain a blocked state before
    # the user opens Review.
    for gpu in gpu_list:
        if int(gpu.get("index", -1)) != 1:
            continue
        occupancy = _query_gpu_foreign_processes(1)
        gpu["occupancy_status"] = (
            "blocked" if occupancy.get("foreign_processes")
            else "clear" if occupancy.get("query_ok")
            else "unknown"
        )
        gpu["foreign_processes"] = occupancy.get("foreign_processes", [])

    return {
        "cpu_pct": cpu_pct,
        "ram_used_gb": ram_used_gb,
        "ram_total_gb": ram_total_gb,
        "ram_pct": ram_pct,
        "gpu": gpu_list,
        "gpu_available": gpu_available,
    }


@router.get("/dataset-preview")
def dataset_preview(project_id: int, train_data_dir: str = "") -> dict:
    _ensure_project(project_id)
    target = train_data_dir.strip()
    if not target:
        conn = get_conn()
        row = conn.execute(
            "SELECT dataset_dir FROM projects WHERE id = ?", (project_id,)
        ).fetchone()
        conn.close()
        target = str(row["dataset_dir"]) if row else ""
    d = Path(target)
    if not d.exists():
        return {"project_id": project_id, "image_path": None, "thumbnail_url": None}
    files = [p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if not files:
        return {"project_id": project_id, "image_path": None, "thumbnail_url": None}
    first = sorted(files)[0]
    try:
        with Image.open(first) as im:
            im = im.convert("RGB")
            im.thumbnail((320, 320))
            from io import BytesIO
            buf = BytesIO()
            im.save(buf, format="JPEG", quality=82)
            thumb = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except OSError:
        thumb = None
    return {"project_id": project_id, "image_path": str(first), "thumbnail_url": thumb}


# ── Training設定の下書き保存(Persistence) ─────────────────────────────────
# 「最後に実行したrunの設定」(training_runs.config_json)とは異なる概念:
# ユーザーがまだ一度も学習を開始していない、または設定変更直後にProject切替・
# Page reload・Backend再起動が起きた場合でも、フォームの入力内容を失わないための
# Project単位の下書き保存。projects.pipeline_config_json(Dataset Builder側)と
# 同一パターンを踏襲し、新規テーブルは追加しない。

_TRAINING_CONFIG_DRAFT_KEYS = {
    "allow_interactive_gpu_sharing", "advanced", "resume_checkpoint_id", "preview_backend", "preview_lora_strength", "preview_extra_loras",
    "training_goal", "quality_preset", "training_memory_mode", "preview_gpu", "preview_base_checkpoint_path", "preview_steps", "preview_cfg",
    "preset_id", "epochs", "repeats", "alpha", "rank", "save_every_n_epochs",
    "output_name", "base_checkpoint_path", "train_data_dir", "reg_data_dir",
    "resolution", "learning_rate", "train_batch_size", "optimizer", "scheduler",
    "min_snr_gamma", "mixed_precision", "save_precision", "xformers",
    "cache_latents", "cache_latents_to_disk", "gradient_checkpointing",
    "persistent_data_loader_workers", "max_data_loader_n_workers",
    "network_train_unet_only", "model_family", "training_engine", "dataset_source",
    "dataset_snapshot_id", "preview_profile_snapshot_id",
    "gpu_device_id", "vae_path", "text_encoder_path",
}


@router.get("/config-draft/{project_id}")
def get_training_config_draft(project_id: int) -> dict:
    """Project単位のTraining設定下書きを返す。未保存なら空dictを返す(Frontend側で
    デフォルト値を適用する)。"""
    conn = get_conn()
    row = conn.execute(
        "SELECT training_config_json FROM projects WHERE id = ?", (project_id,)
    ).fetchone()
    conn.close()
    if row is None:
        raise HTTPException(status_code=404, detail="project not found")
    try:
        config = json.loads(row["training_config_json"] or "{}")
        if not isinstance(config, dict):
            config = {}
    except (ValueError, TypeError):
        config = {}
    config.setdefault("allow_interactive_gpu_sharing",_sharing_default(project_id))
    return {"project_id": project_id, "config": config}


@router.post("/config-draft/{project_id}")
def save_training_config_draft(project_id: int, payload: dict) -> dict:
    """Training設定フォームの下書きをProjectへ保存する。既知キーのみ受け付ける。"""
    conn = get_conn()
    row = conn.execute("SELECT id FROM projects WHERE id = ?", (project_id,)).fetchone()
    if row is None:
        conn.close()
        raise HTTPException(status_code=404, detail="project not found")

    config = {k: v for k, v in (payload or {}).items() if k in _TRAINING_CONFIG_DRAFT_KEYS}
    from ..training.learning_rates import resolve_learning_rates
    try:
        if 'learning_rate' in config:resolve_learning_rates(config, reject_conflict=True)
    except (ValueError,TypeError) as exc:
        conn.close()
        raise HTTPException(422,str(exc)) from exc
    conn.execute(
        "UPDATE projects SET training_config_json = ? WHERE id = ?",
        (json.dumps(config, ensure_ascii=False), project_id),
    )
    conn.commit()
    conn.close()
    return {"project_id": project_id, "config": config}


@router.patch("/config-draft/{project_id}")
def patch_training_config_draft(project_id: int, payload: dict) -> dict:
    """Merge a workflow handoff into the existing Draft without discarding user settings."""
    conn = get_conn()
    row = conn.execute(
        "SELECT training_config_json FROM projects WHERE id=?",
        (project_id,),
    ).fetchone()
    if row is None:
        conn.close()
        raise HTTPException(status_code=404, detail="project not found")
    try:
        current = json.loads(row["training_config_json"] or "{}")
        if not isinstance(current, dict):
            current = {}
    except (ValueError, TypeError):
        current = {}
    current.update({key: value for key, value in (payload or {}).items() if key in _TRAINING_CONFIG_DRAFT_KEYS})
    from ..training.learning_rates import resolve_learning_rates
    try:
        if 'learning_rate' in current:resolve_learning_rates(current, reject_conflict=True)
    except (ValueError,TypeError) as exc:
        conn.close()
        raise HTTPException(422,str(exc)) from exc
    conn.execute(
        "UPDATE projects SET training_config_json=? WHERE id=?",
        (json.dumps(current, ensure_ascii=False), project_id),
    )
    conn.commit()
    conn.close()
    return {"project_id": project_id, "config": current}


@router.get("/advanced-catalog")
def advanced_catalog(model_family: str = "anima") -> dict:
    from ..training.advanced import catalog
    return {"fields": catalog(model_family)}


@router.post("/advanced-validate")
def advanced_validate(payload: dict) -> dict:
    from ..training.advanced import validate
    family = detect_model_family(str(payload.get("base_checkpoint_path", "")), str(payload.get("model_family", "auto")))
    try:
        values = validate(payload.get("advanced", {}), family, payload)
        return {"valid": True, "advanced": values}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _validate_continuation(payload: TrainingStartIn) -> dict:
    if payload.resume_checkpoint_id is None:
        return {}
    conn = get_conn()
    try:
        row = conn.execute("SELECT c.file_path,c.run_id,tr.config_json FROM checkpoints c JOIN training_runs tr ON tr.id=c.run_id WHERE c.id=? AND c.project_id=? AND tr.project_id=?", (payload.resume_checkpoint_id,payload.project_id,payload.project_id)).fetchone()
    finally:
        conn.close()
    if row is None or not Path(row['file_path']).is_file():
        raise HTTPException(status_code=422, detail='継続元LoRAが見つかりません')
    old = json.loads(row['config_json'] or '{}')
    if Path(str(old.get('base_checkpoint_path',''))).resolve() != Path(payload.base_checkpoint_path).resolve() or int(old.get('rank',0)) != payload.rank or float(old.get('alpha',0)) != payload.alpha:
        raise HTTPException(status_code=422, detail='継続元とベースモデル・Rank・Alphaを一致させてください')
    return {'resume_from_checkpoint':row['file_path'],'resume_from_run_id':int(row['run_id']),'resume_checkpoint_id':payload.resume_checkpoint_id}


@router.post('/estimate-draft')
def estimate_current_draft(payload: TrainingStartIn) -> dict:
    from ..training.draft_estimate import estimate_draft
    from ..training.advanced import validate
    cfg = payload.model_dump()
    try:
        cfg.update(validate(payload.advanced, payload.model_family, cfg))
    except ValueError as exc:
        raise HTTPException(status_code=422,detail=str(exc)) from exc
    return estimate_draft(cfg)


@router.post('/continuation-check')
def continuation_check(payload: TrainingStartIn) -> dict:
    return {'valid':True, **_validate_continuation(payload)}


@router.get('/runs/{run_id}/cleanup')
def run_cleanup_info(run_id: int) -> dict:
    from ..training.run_cleanup import inspect_cleanup
    try:return inspect_cleanup(run_id)
    except ValueError as exc:raise HTTPException(status_code=409,detail=str(exc)) from exc


@router.delete('/runs/{run_id}/cleanup/{kind}')
def run_cleanup_delete(run_id: int, kind: str) -> dict:
    from ..training.run_cleanup import cleanup
    try:return cleanup(run_id,kind)
    except ValueError as exc:raise HTTPException(status_code=409,detail=str(exc)) from exc


@router.get('/project-cleanup/{project_id}')
def project_cleanup_info(project_id:int):
    from ..training.run_cleanup import project_cleanup_inventory
    _ensure_project(project_id)
    return project_cleanup_inventory(project_id)

@router.post('/project-cleanup/{project_id}')
def project_cleanup_delete(project_id:int,payload:dict):
    from ..training.run_cleanup import cleanup_project_runs
    _ensure_project(project_id)
    if payload.get('confirm') is not True:raise HTTPException(400,'削除対象の確認が必要です')
    try:return cleanup_project_runs(project_id,payload.get('run_ids',[]),payload.get('kinds',[]))
    except (ValueError,OSError) as exc:raise HTTPException(409,str(exc)) from exc


@router.post('/runs/{run_id}/reveal')
def reveal_run_output(run_id:int,payload:dict|None=None):
    import subprocess
    from ..training.runtime.state import run_dir
    conn=get_conn()
    try:
        row=conn.execute('SELECT status,latest_checkpoint_path,config_json FROM training_runs WHERE id=?',(run_id,)).fetchone()
        if row is None:raise HTTPException(404,'Runが見つかりません')
        checkpoint_id=(payload or {}).get('checkpoint_id')
        if checkpoint_id is not None:
            checkpoint=conn.execute('SELECT file_path FROM checkpoints WHERE id=? AND run_id=?',(checkpoint_id,run_id)).fetchone()
            if checkpoint is None:raise HTTPException(404,'このRunのLoRAが見つかりません')
            path=Path(checkpoint['file_path'])
        else:
            cfg=json.loads(row['config_json'] or '{}');final=run_dir(run_id)/'output'/(str(cfg.get('output_name',''))+'.safetensors')
            path=final if row['status']=='completed' and final.is_file() else Path(row['latest_checkpoint_path'] or '')
        if not path.is_file() or not path.resolve().is_relative_to(run_dir(run_id).resolve()):raise HTTPException(404,'保存済みLoRAが見つかりません')
        subprocess.Popen(['explorer.exe','/select,',str(path.resolve())])
        return {'path':str(path.resolve()),'selected':True}
    finally:conn.close()
