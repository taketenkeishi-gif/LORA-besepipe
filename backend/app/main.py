from __future__ import annotations

import asyncio
import logging
import threading
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .db import init_db
from .routers import character_sets, comfy_tag_proxy, dataset_files, dataset_video, dataset_video_chars, dataset_video_link, evaluation
from .routers import basepipe, collector, dataset, debug, imgsearch, library, previews, preview_profiles, presets, project_file, projects, settings, system, tags, training, training_recipe, training_preset_files
from .services.pixiv_auth import refresh_session_if_expired

logger = logging.getLogger(__name__)

# Background Pixiv session refresh
PIXIV_REFRESH_THREAD: threading.Thread | None = None
PIXIV_REFRESH_RUNNING = False

app = FastAPI(title="LoRA Workbench API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5175", "http://localhost:5175",
        "http://127.0.0.1:5176", "http://localhost:5176",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(projects.router)
app.include_router(collector.router)
app.include_router(debug.router)
app.include_router(dataset.router)
app.include_router(dataset_files.router)
app.include_router(dataset_video.router)
app.include_router(dataset_video_chars.router)
app.include_router(dataset_video_link.router)
app.include_router(character_sets.router)
app.include_router(comfy_tag_proxy.router)
app.include_router(evaluation.router)
app.include_router(tags.router)
app.include_router(training.router)
app.include_router(training_recipe.router)
app.include_router(previews.router)
app.include_router(settings.router)
app.include_router(system.router)
app.include_router(preview_profiles.router)
app.include_router(training_preset_files.router)  # must precede presets.router (/{preset_id})
app.include_router(presets.router)
app.include_router(library.router)
app.include_router(project_file.router)
app.include_router(imgsearch.router)
app.include_router(basepipe.router)


def _pixiv_session_refresh_loop() -> None:
    """Background loop to periodically refresh Pixiv session."""
    logger.info("Starting Pixiv session refresh loop")
    while PIXIV_REFRESH_RUNNING:
        try:
            # Check and refresh Pixiv session every 1 hour
            time.sleep(3600)
            if not PIXIV_REFRESH_RUNNING:
                break

            logger.debug("Running scheduled Pixiv session refresh...")
            # Run async function in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(refresh_session_if_expired())
                if result.get("refreshed"):
                    logger.info(f"Pixiv session refreshed for user {result.get('user_id')}")
                elif result.get("is_valid"):
                    logger.debug("Pixiv session is still valid")
                else:
                    logger.warning(f"Pixiv session refresh failed: {result.get('error')}")
            finally:
                loop.close()
        except Exception as e:
            logger.error(f"Error in Pixiv session refresh loop: {e}")


@app.on_event("startup")
def on_startup() -> None:
    global PIXIV_REFRESH_THREAD, PIXIV_REFRESH_RUNNING

    init_db()
    presets._seed_defaults_internal()  # デフォルトプロファイルを初回自動追加

    # 既存プレビュープロンプトの空 quality / negative に既定値を補完（冪等）
    try:
        from .routers import settings as _settings
        _settings._migrate_prompt_defaults()
    except Exception as e:  # noqa: BLE001
        logger.warning(f"preview prompt migration skipped: {e}")

    # 過去の学習実測値から学習時間予測のキャリブレーションをシード（冪等）
    try:
        from .routers import training as _training
        _training._backfill_calib_from_runs()
    except Exception as e:  # noqa: BLE001
        logger.warning(f"calib backfill skipped: {e}")

    # Backend再起動前から残っていた通常Queueは、明示操作なしに学習再開しない。
    # queue_if_busy=trueで明示的に永続待機させたAnima Runだけは保持し、後段の
    # GPU1安全Monitorが条件成立まで再評価する。
    try:
        from .routers import training as _training
        _quarantined = _training._quarantine_queued_runs_on_startup()
        if _quarantined:
            logger.warning("quarantined %s stale queued training run(s) after backend restart", _quarantined)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"queued run quarantine skipped: {e}")

    # 起動時に学習中だった run を再アタッチ／status を実態へ reconcile
    try:
        from .routers import training as _training
        _training._reattach_active_runs()
    except Exception as e:  # noqa: BLE001
        logger.warning(f"reattach active runs skipped: {e}")

    # H3 waiting Runs are explicitly armed with queue_if_busy=true.  Resume
    # only their safe admission monitor; no GPU work starts unless physical
    # GPU1, process ownership, and the managed ComfyUI gate all pass again.
    try:
        from .routers import basepipe as _basepipe
        _qwen_recovery = _basepipe.recover_stale_qwen_runs_on_startup()
        if _qwen_recovery["recovered_run_ids"]:
            logger.warning("re-armed stale Qwen Run(s) after backend restart: %s", _qwen_recovery)
        _enhancement_recovery = _basepipe.recover_stale_enhancement_runs_on_startup()
        if _enhancement_recovery["recovered_run_ids"]:
            logger.warning("re-armed stale Enhancement Run(s) after backend restart: %s", _enhancement_recovery)
        _basepipe.ensure_h3_wait_monitor()
        _basepipe.ensure_qwen_wait_monitor()
        _basepipe.ensure_enhancement_wait_monitor()
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Character generation waiting monitor recovery skipped: {e}")

    # Explicit queue_if_busy Anima Runs are user-armed persistent waits. They
    # survive restart and only dispatch after the physical GPU1 safety gate.
    try:
        from .routers import training as _training
        _training.ensure_training_wait_monitor()
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Anima training waiting monitor recovery skipped: {e}")

    # Preview Job Restart Recovery: running→pending、output消失succeeded→pending。
    # 成功済み(output存在)Jobは再実行しない。Backend再起動でWorkerスレッドが
    # 消えても、専用のPreview待機MonitorがGUI pollingに依存せず自動再処理する。
    try:
        from .db import get_conn as _get_conn
        from .training.runtime.preview_jobs import ensure_preview_wait_monitor, recover_stale_jobs_on_startup
        _conn2 = _get_conn()
        _result = recover_stale_jobs_on_startup(_conn2)
        _conn2.close()
        ensure_preview_wait_monitor()
        logger.info(f"preview job restart recovery: {_result}")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"preview job restart recovery skipped: {e}")

    # LoRA Basepipe 専用 ComfyUI は既存の生成用（既定 8188）と別ポート（既定 8189）で
    # 独立起動する。未設定なら 8189 をシード（変更可）。
    try:
        from .db import get_conn
        _conn = get_conn()
        _row = _conn.execute("SELECT value FROM app_settings WHERE key = 'comfyui_url'").fetchone()
        if _row is None or not str(_row["value"]).strip():
            _conn.execute(
                "INSERT INTO app_settings(key, value) VALUES('comfyui_url', 'http://127.0.0.1:8189') "
                "ON CONFLICT(key) DO UPDATE SET value = 'http://127.0.0.1:8189'"
            )
            _conn.commit()
        _conn.close()
    except Exception as e:  # noqa: BLE001
        logger.warning(f"comfyui_url seed skipped: {e}")

    # ComfyUI はバックエンド起動時に自動起動しない。
    # 明示的な起動要求だけを services.comfyui_process.ensure_running 経由で受け、
    # 物理GPUとCUDA列挙順の照合を必ず通す。

    # Start Pixiv session refresh background thread
    PIXIV_REFRESH_RUNNING = True
    PIXIV_REFRESH_THREAD = threading.Thread(
        target=_pixiv_session_refresh_loop,
        daemon=True,
        name="PixivSessionRefreshThread"
    )
    PIXIV_REFRESH_THREAD.start()
    logger.info("Pixiv session refresh thread started")


@app.on_event("shutdown")
def on_shutdown() -> None:
    global PIXIV_REFRESH_RUNNING
    logger.info("Shutting down...")
    PIXIV_REFRESH_RUNNING = False
    if PIXIV_REFRESH_THREAD and PIXIV_REFRESH_THREAD.is_alive():
        PIXIV_REFRESH_THREAD.join(timeout=5)
    logger.info("Pixiv session refresh thread stopped")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def root() -> dict[str, str]:
    return {"name": "LoRA Workbench API", "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)
