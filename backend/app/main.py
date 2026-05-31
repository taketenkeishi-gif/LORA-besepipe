from __future__ import annotations

import asyncio
import logging
import threading
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .db import init_db
from .routers import collector, dataset, library, previews, presets, projects, settings, tags, training
from .services.pixiv_auth import refresh_session_if_expired

logger = logging.getLogger(__name__)

# Background Pixiv session refresh
PIXIV_REFRESH_THREAD: threading.Thread | None = None
PIXIV_REFRESH_RUNNING = False

app = FastAPI(title="LoRA Workbench API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(projects.router)
app.include_router(collector.router)
app.include_router(dataset.router)
app.include_router(tags.router)
app.include_router(training.router)
app.include_router(previews.router)
app.include_router(settings.router)
app.include_router(presets.router)
app.include_router(library.router)


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
