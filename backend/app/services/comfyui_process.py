"""ComfyUI プロセス起動マネージャ。

LayeredPaintApp の platform/comfy/ComfyProcessManager.cpp を Python へ移植したもの。

方針（同等）:
  1. まずポート/HTTP 疎通を確認 → 既に起動済みなら何もしない（多重起動しない）。
  2. 未起動かつ明示的に起動要求された場合のみ、物理GPUとCUDA列挙順を照合する
     launch_comfyui_isolated.py 経由で起動する。直接 main.py 起動にはフォールバックしない。
  3. バックグラウンドで /system_stats をポーリングして ready を検知（タイムアウト90秒）。

バックエンド起動時には呼ばれない。GPUを使う常駐サービスの暗黙再起動を防ぐ。
"""
from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

from ..db import get_conn
from .comfyui_client import get_comfyui_url

logger = logging.getLogger(__name__)

_LAUNCH_LOCK = threading.Lock()
_LAUNCHING = False
# status: unknown | running | launching | ready | failed | disabled
_STATE: dict = {"status": "unknown", "message": "", "pid": None, "launcher": None}


def _set_state(**kw) -> None:
    _STATE.update(kw)


def get_state() -> dict:
    return dict(_STATE)


def _comfy_root() -> str:
    conn = get_conn()
    row = conn.execute("SELECT value FROM app_settings WHERE key = 'comfyui_root'").fetchone()
    conn.close()
    return (row["value"] if row else "") or ""


def is_reachable(url: str | None = None, timeout: float = 2.0) -> bool:
    url = (url or get_comfyui_url()).rstrip("/")
    try:
        r = requests.get(f"{url}/system_stats", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def _port_from_url(url: str) -> int:
    try:
        return urlparse(url).port or 8188
    except Exception:
        return 8188


def _build_launch(comfy_root: str, port: int) -> tuple[str, Path, list[str]] | None:
    """物理GPUを検証する安全ランチャーだけを返す。直接 main.py 起動は禁止する。"""
    root = Path(comfy_root)
    if not root.exists():
        return None
    portable = root.parent
    py = portable / "python_embeded" / "python.exe"
    launcher = portable / "launch_comfyui_isolated.py"
    if port == 8188 and launcher.exists() and py.exists():
        return ("launch_comfyui_isolated.py (physical GPU mapping)", portable, [str(py), "-u", str(launcher)])

    return None


def _wait_ready(url: str, timeout: float = 90.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if is_reachable(url, timeout=2.0):
            _set_state(status="ready", message="ComfyUI 起動完了")
            logger.info("ComfyUI is ready: %s", url)
            return True
        time.sleep(1.0)
    _set_state(status="failed", message="ComfyUI 起動タイムアウト（90秒）")
    logger.warning("ComfyUI launch timed out")
    return False


def ensure_running(block_wait: bool = False) -> dict:
    """ComfyUI が未起動なら起動する。起動済みなら何もしない。"""
    global _LAUNCHING
    url = get_comfyui_url()

    if is_reachable(url):
        _set_state(status="running", message="既に起動済み")
        return get_state()

    with _LAUNCH_LOCK:
        if _LAUNCHING:
            return get_state()

        root = _comfy_root()
        if not root:
            _set_state(status="disabled", message="comfyui_root が未設定（設定すると自動起動します）")
            return get_state()

        port = _port_from_url(url)
        launcher = _build_launch(root, port)
        if launcher is None:
            _set_state(status="failed", message=f"起動スクリプトが見つかりません: {root}")
            return get_state()

        kind, cwd, args = launcher
        try:
            flags = 0
            # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP — バックエンド再起動に巻き込まれない独立プロセス
            flags |= getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
            flags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
            proc = subprocess.Popen(
                args,
                cwd=str(cwd),
                creationflags=flags,
                close_fds=True,
                # Match the launcher's physical-index probe with the trainer
                # and backend GPU mapping. Without PCI_BUS_ID, PyTorch can
                # enumerate the two adapters in the reverse order and a
                # requested physical GPU1 may receive CUDA0/GPU0 instead.
                env={**os.environ, "COMFYUI_GPU_INDEX": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"},
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
            )
            _LAUNCHING = True
            _set_state(status="launching", message="ComfyUI を起動しています...", pid=proc.pid, launcher=kind)
            logger.info("Launching ComfyUI via %s (cwd=%s, pid=%s)", kind, cwd, proc.pid)
        except Exception as exc:  # noqa: BLE001
            _set_state(status="failed", message=f"起動失敗: {exc}")
            logger.error("ComfyUI launch failed: %s", exc)
            return get_state()

    def _poll() -> None:
        global _LAUNCHING
        try:
            _wait_ready(url)
        finally:
            _LAUNCHING = False

    if block_wait:
        _poll()
    else:
        threading.Thread(target=_poll, daemon=True, name="ComfyUIReadyWait").start()

    return get_state()
