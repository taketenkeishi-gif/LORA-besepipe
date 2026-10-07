"""Read-only view of "what is the preview pipeline doing right now" for one run.

Everything here is derived from data that already exists (preview_jobs rows,
comfy-preview-window.json written by comfy_epoch_hook, and the per-job
job.json that desktop_comfy.collect_graph persists on every ComfyUI websocket
`progress` event).  Nothing here touches the GPU, ComfyUI, or the training loop.

Truth about the backend (comfy_epoch_hook.py): while the Comfy epoch window is
open, the training process is *blocked* inside sample_images() waiting for
ComfyUI, so training step/loss do not advance on purpose.  Outside a window,
pending jobs simply wait until the training process ends.
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

# A running job whose job.json has not changed for this long is shown as stalled.
STALL_AFTER_SECONDS = 180
# The epoch window with no running job and no finished job for this long is stalled.
WINDOW_IDLE_STALL_SECONDS = 300

_PROGRESS_RE = re.compile(r"(\d+)\s*/\s*(\d+)")


def _parse_db_time(value) -> float | None:
    if not value:
        return None
    try:
        text = str(value).replace("T", " ").split(".")[0]
        return datetime.strptime(text, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return None


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _latest_job_progress(run_directory: Path, now: float) -> dict | None:
    """Newest ComfyUI collector job.json that is still in flight (cheap: <=few files)."""
    sample = run_directory / "output" / "sample"
    try:
        candidates = [p / "job.json" for p in sample.iterdir() if p.is_dir() and p.name.startswith("native-")]
    except OSError:
        return None
    stamped = [(m, p) for p in candidates if (m := _mtime(p)) is not None]
    if not stamped:
        return None
    stamped.sort(reverse=True)
    mtime, path = stamped[0]
    try:
        job = json.loads(path.read_text(encoding="utf8"))
    except (OSError, ValueError):
        return None
    if job.get("status") not in ("starting", "queued", "running"):
        return None
    value, maximum = job.get("progress_value"), job.get("progress_max")
    if value is None:
        match = _PROGRESS_RE.search(str(job.get("message") or ""))
        if match:
            value, maximum = int(match.group(1)), int(match.group(2))
    return {
        "collector_status": job.get("status"),
        "step": int(value) if value is not None else None,
        "step_max": int(maximum) if maximum else None,
        "updated_at": mtime,
        "age_seconds": max(0.0, now - mtime),
        "message": job.get("message"),
    }


def build_preview_activity(conn, run_id: int, run_status: str, run_directory: Path, now: float | None = None) -> dict:
    """Additive status block. Never raises for missing files; never writes."""
    now = time.time() if now is None else now
    rows = conn.execute(
        "SELECT id, checkpoint_id, epoch, prompt_index, instance_index, status, error_detail, attempts, "
        "started_at, completed_at, created_at FROM preview_jobs WHERE run_id=? "
        "ORDER BY epoch, prompt_index, instance_index",
        (run_id,),
    ).fetchall()
    unresolved_epochs = sorted({int(r["epoch"]) for r in rows if r["status"] in ("pending", "running")})
    base = {"active": False, "phase": "none", "jobs": [], "run_id": run_id}
    if not rows or not unresolved_epochs:
        return base

    epoch = unresolved_epochs[0]
    epoch_rows = [r for r in rows if int(r["epoch"]) == epoch]
    count = {s: sum(1 for r in epoch_rows if r["status"] == s) for s in ("pending", "running", "succeeded", "failed")}
    total = len(epoch_rows)

    window = None
    try:
        from .comfy_epoch_hook import active_window

        window = active_window(run_directory)
    except Exception:  # noqa: BLE001 - observation only
        window = None
    training_alive = run_status == "training"
    paused = bool(window) and int(window.get("epoch", -1)) == epoch
    window_started = (float(window["deadline"]) - 1800.0) if paused else None

    running_row = next((r for r in epoch_rows if r["status"] == "running"), None)
    progress = _latest_job_progress(run_directory, now) if running_row else None
    if running_row:
        phase = "generating"
    elif paused:
        phase = "window_waiting"
    elif training_alive:
        phase = "deferred_until_training_ends"
    else:
        phase = "waiting_gpu"

    waiting_reason = next((str(r["error_detail"]) for r in epoch_rows if r["status"] == "pending" and r["error_detail"]), "")

    finished_times = [t for r in epoch_rows if (t := _parse_db_time(r["completed_at"])) is not None]
    started_times = [t for r in epoch_rows if (t := _parse_db_time(r["started_at"])) is not None]
    heartbeat_sources = [t for t in finished_times + started_times if t]
    if progress:
        heartbeat_sources.append(progress["updated_at"])
    if paused and window_started:
        heartbeat_sources.append(window_started)
    last_activity = max(heartbeat_sources) if heartbeat_sources else None
    if phase == "generating":
        # Liveness of the in-flight job is what matters; an earlier image finishing must not mask silence.
        in_flight = progress["updated_at"] if progress else _parse_db_time(running_row["started_at"])
        if in_flight:
            last_activity = in_flight
    heartbeat_age = (now - last_activity) if last_activity else None

    stalled = False
    if phase == "generating":
        reference = progress["updated_at"] if progress else (_parse_db_time(running_row["started_at"]) or now)
        stalled = (now - reference) > STALL_AFTER_SECONDS
    elif phase == "window_waiting":
        stalled = heartbeat_age is not None and heartbeat_age > WINDOW_IDLE_STALL_SECONDS

    current = None
    if running_row:
        started = _parse_db_time(running_row["started_at"])
        current = {
            "id": int(running_row["id"]),
            "prompt_index": int(running_row["prompt_index"]),
            "instance_index": int(running_row["instance_index"]),
            "elapsed_seconds": max(0.0, now - started) if started else None,
            "step": progress["step"] if progress else None,
            "step_max": progress["step_max"] if progress else None,
            "collector_status": progress["collector_status"] if progress else None,
        }

    jobs = []
    for r in rows:
        if int(r["epoch"]) not in unresolved_epochs:
            continue
        item = {
            "id": int(r["id"]),
            "checkpoint_id": int(r["checkpoint_id"]),
            "epoch": int(r["epoch"]),
            "prompt_index": int(r["prompt_index"]),
            "instance_index": int(r["instance_index"]),
            "status": r["status"],
            "error": r["error_detail"] or "",
            "attempts": int(r["attempts"] or 0),
        }
        if current and item["id"] == current["id"]:
            item["step"], item["step_max"] = current["step"], current["step_max"]
            item["elapsed_seconds"] = current["elapsed_seconds"]
        jobs.append(item)

    return {
        "active": True,
        "run_id": run_id,
        "phase": phase,
        "epoch": epoch,
        "total": total,
        "done": count["succeeded"],
        "failed": count["failed"],
        "running": count["running"],
        "pending": count["pending"],
        "current_job": current,
        "training_paused": paused,
        "window": (
            {
                "epoch": int(window.get("epoch", epoch)),
                "step": window.get("step"),
                "elapsed_seconds": max(0.0, now - window_started) if window_started else None,
                "remaining_seconds": max(0.0, float(window["deadline"]) - now),
            }
            if paused
            else None
        ),
        "waiting_reason": waiting_reason,
        "last_activity_age_seconds": heartbeat_age,
        "stalled": stalled,
        "stall_after_seconds": STALL_AFTER_SECONDS if phase == "generating" else WINDOW_IDLE_STALL_SECONDS,
        "generated_at": now,
        "jobs": jobs,
    }
