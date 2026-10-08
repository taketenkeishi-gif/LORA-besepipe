"""Previews on the RTX 3060 while the 3090 Ti keeps training (user decision 2026-10-08).

A private ComfyUI on port 8189 (the only second instance; ComfyUI is otherwise single-port) is started by THIS app when a
preview needs it and stopped again when nothing has needed it for a while, so it never sits in the user's way.
A run uses it when its run directory holds MARKER; the training process reads the same file at every epoch, so switching
takes effect from the next epoch even during training.
Measured (2026-10-08, 1024 px 30 steps): ~62 s per preview on the 3060 vs 17.8 s on the 3090 Ti, peak 8.1 GB;
the display stayed within one 60 Hz frame of extra GPU wait (p95 13 ms).
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import requests

PREVIEW_URL = "http://127.0.0.1:8189"
GPU_NAME = "RTX 3060"
MARKER = "preview-parallel.json"
IDLE_STOP_SECONDS = 300
_STATE = Path(__file__).resolve().parents[4] / ".runtime" / "preview-3060.json"


def parallel(run_directory: Path | str) -> bool:
    return (Path(run_directory) / MARKER).is_file()


def set_mode(run_directory: Path | str, gpu: str) -> None:
    """gpu0 = previews on the 3060 beside training; gpu1 = the classic way (training pauses, previews on the 3090 Ti)."""
    path = Path(run_directory) / MARKER
    if gpu == "gpu0":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"gpu": "gpu0", "url": PREVIEW_URL, "since": time.time()}), encoding="utf-8")
    else:
        path.unlink(missing_ok=True)


def ready() -> bool:
    try:
        stats = requests.get(PREVIEW_URL + "/system_stats", timeout=1.5).json()
    except Exception:
        return False
    return any(GPU_NAME in str(d.get("name", "")) for d in stats.get("devices", []))


def _load() -> dict:
    try:
        return json.loads(_STATE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save(value: dict) -> None:
    _STATE.parent.mkdir(parents=True, exist_ok=True)
    _STATE.write_text(json.dumps(value), encoding="utf-8")


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        import psutil

        return psutil.pid_exists(int(pid)) and psutil.Process(int(pid)).is_running()
    except Exception:
        return False


def alive() -> bool:
    return _alive(_load().get("pid"))


def touch() -> None:
    state = _load()
    state["last_used"] = time.time()
    _save(state)


def ensure_started(comfy_root: Path | str) -> str:
    """'ready' | 'starting' | 'error: ...'. Never blocks: the waiting preview job is simply retried by the monitor."""
    touch()
    if ready():
        return "ready"
    state = _load()
    if _alive(state.get("pid")) and time.time() - float(state.get("started", 0)) < 240:
        return "starting"
    install = Path(comfy_root).resolve().parent
    launcher, python = install / "launch_comfyui_isolated.py", install / "python_embeded" / "python.exe"
    if not launcher.is_file() or not python.is_file():
        return f"error: ComfyUIの起動ツールが見つかりません（{launcher}）"
    env = os.environ.copy()
    env.update({"COMFYUI_PREVIEW_INSTANCE": "1", "COMFYUI_GPU_INDEX": "0", "PYTHONIOENCODING": "utf-8"})
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
    proc = subprocess.Popen([str(python), str(launcher)], cwd=str(install), env=env, creationflags=flags,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
    _save({**state, "pid": proc.pid, "started": time.time(), "last_used": time.time()})
    return "starting"


def stop(reason: str = "") -> bool:
    state = _load()
    pid = state.get("pid")
    if not _alive(pid):
        return False
    subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, timeout=30)  # launcher + its ComfyUI child
    _save({**state, "pid": None, "stopped": time.time(), "stop_reason": reason})
    return True


def stop_if_idle(conn) -> bool:
    """Stop the 3060 instance once no run that uses it is training and no preview is waiting for it for IDLE_STOP_SECONDS."""
    state = _load()
    if not _alive(state.get("pid")):
        return False
    from .state import run_dir

    busy = any(parallel(run_dir(int(r["id"]))) for r in conn.execute("SELECT id FROM training_runs WHERE status IN ('queued','training')"))
    waiting = any(parallel(run_dir(int(r["run_id"]))) for r in conn.execute("SELECT DISTINCT run_id FROM preview_jobs WHERE status IN ('pending','running')"))
    if busy or waiting:
        touch()
        return False
    if time.time() - float(state.get("last_used", 0)) < IDLE_STOP_SECONDS:
        return False
    return stop("idle")
