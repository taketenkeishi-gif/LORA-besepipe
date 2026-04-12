from __future__ import annotations

import json
import base64
import threading
import time
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException
from PIL import Image, ImageDraw

from ..db import get_conn
from ..schemas import TrainingControlIn, TrainingStartIn

router = APIRouter(prefix="/training", tags=["training"])
RUNNER_LOCK = threading.Lock()
RUNNER_THREADS: dict[int, threading.Thread] = {}
SLOTS = ["face", "bust", "full", "bg"]
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}


def _ensure_project(project_id: int) -> dict:
    conn = get_conn()
    row = conn.execute(
        "SELECT id, outputs_dir FROM projects WHERE id = ?",
        (project_id,),
    ).fetchone()
    conn.close()
    if row is None:
        raise HTTPException(status_code=404, detail=f"project not found: {project_id}")
    return dict(row)


def _set_project_status(conn, project_id: int, status: str) -> None:
    conn.execute("UPDATE projects SET status = ? WHERE id = ?", (status, project_id))


def _latest_run(conn, project_id: int):
    return conn.execute(
        """
        SELECT id, status, stop_mode, current_epoch, current_step, total_epochs, steps_per_epoch
        FROM training_runs
        WHERE project_id = ? ORDER BY id DESC LIMIT 1
        """,
        (project_id,),
    ).fetchone()


def _ensure_runner(project_id: int) -> None:
    with RUNNER_LOCK:
        alive = RUNNER_THREADS.get(project_id)
        if alive is not None and alive.is_alive():
            return
        t = threading.Thread(target=_runner_loop, args=(project_id,), daemon=True)
        RUNNER_THREADS[project_id] = t
        t.start()


def _write_preview_and_checkpoint(conn, project_id: int, epoch: int) -> str:
    prompt_row = conn.execute(
        "SELECT key, value FROM app_settings WHERE key IN ('positive_prompt','negative_prompt')"
    ).fetchall()
    prompt_map = {str(r["key"]): str(r["value"]) for r in prompt_row}
    pos = prompt_map.get("positive_prompt", "")
    neg = prompt_map.get("negative_prompt", "")

    row = conn.execute("SELECT outputs_dir FROM projects WHERE id = ?", (project_id,)).fetchone()
    outputs_dir = Path(row["outputs_dir"])
    checkpoints_dir = outputs_dir / "checkpoints"
    previews_dir = outputs_dir / "previews"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = checkpoints_dir / f"project_{project_id}_e{epoch}.safetensors"
    ckpt_path.write_text("placeholder checkpoint", encoding="utf-8")
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO checkpoints(project_id, file_path, epoch, step, mark)
        VALUES (?, ?, ?, ?, 'none')
        """,
        (project_id, str(ckpt_path), epoch, 0),
    )
    checkpoint_id = cur.lastrowid
    for slot in SLOTS:
        p = previews_dir / f"e{epoch}_{slot}.png"
        img = Image.new("RGB", (512, 512), color=(30 + epoch * 10 % 200, 40 + len(slot) * 20, 90))
        d = ImageDraw.Draw(img)
        d.text((24, 24), f"Project {project_id}", fill=(240, 240, 240))
        d.text((24, 56), f"Epoch {epoch}", fill=(240, 240, 240))
        d.text((24, 88), f"Slot {slot}", fill=(240, 240, 240))
        d.text((24, 120), f"+ {pos[:48]}", fill=(220, 240, 220))
        d.text((24, 148), f"- {neg[:48]}", fill=(240, 220, 220))
        img.save(p, format="PNG")
        cur.execute(
            """
            INSERT INTO preview_samples(checkpoint_id, slot, image_path)
            VALUES (?, ?, ?)
            """,
            (checkpoint_id, slot, str(p)),
        )
    return str(ckpt_path)


def _runner_loop(project_id: int) -> None:
    while True:
        conn = get_conn()
        row = _latest_run(conn, project_id)
        if row is None or row["status"] != "training":
            conn.close()
            return

        epoch = int(row["current_epoch"])
        step = int(row["current_step"])
        total_epochs = int(row["total_epochs"])
        steps_per_epoch = int(row["steps_per_epoch"])
        stop_mode = row["stop_mode"]

        if epoch >= total_epochs:
            conn.execute(
                """
                UPDATE training_runs
                SET status = 'completed', updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (row["id"],),
            )
            _set_project_status(conn, project_id, "completed")
            conn.commit()
            conn.close()
            return

        # 疑似学習: 1 step進める
        time.sleep(0.5)
        step += 1
        if step >= steps_per_epoch:
            epoch += 1
            step = 0
            latest_ckpt = _write_preview_and_checkpoint(conn, project_id, epoch)
            conn.execute(
                """
                UPDATE training_runs
                SET latest_checkpoint_path = ?, current_epoch = ?, current_step = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (latest_ckpt, epoch, step, row["id"]),
            )
            if stop_mode == "epoch":
                conn.execute(
                    """
                    UPDATE training_runs
                    SET status = 'paused', updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (row["id"],),
                )
                _set_project_status(conn, project_id, "paused")
                conn.commit()
                conn.close()
                return
        else:
            conn.execute(
                """
                UPDATE training_runs
                SET current_epoch = ?, current_step = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (epoch, step, row["id"]),
            )

        conn.commit()
        conn.close()


@router.post("/start")
def start(payload: TrainingStartIn) -> dict:
    _ensure_project(payload.project_id)
    conn = get_conn()
    dataset_count = conn.execute(
        "SELECT COUNT(*) AS c FROM dataset_items WHERE project_id = ? AND selected = 1",
        (payload.project_id,),
    ).fetchone()["c"]
    steps_per_epoch = max(1, int(dataset_count) * int(payload.repeats))
    conn.close()

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO training_runs(
            project_id, status, stop_mode, latest_checkpoint_path, config_json,
            current_epoch, current_step, total_epochs, steps_per_epoch
        )
        VALUES (?, 'training', NULL, NULL, ?, 0, 0, ?, ?)
        """,
        (
            payload.project_id,
            json.dumps({"preset_id": payload.preset_id, "alpha": payload.alpha, "repeats": payload.repeats}),
            payload.epochs,
            steps_per_epoch,
        ),
    )
    run_id = cur.lastrowid
    _set_project_status(conn, payload.project_id, "training")
    conn.commit()
    conn.close()
    _ensure_runner(payload.project_id)
    return {
        "run_id": run_id,
        "project_id": payload.project_id,
        "preset_id": payload.preset_id,
        "status": "training",
        "total_epochs": payload.epochs,
        "steps_per_epoch": steps_per_epoch,
        "dataset_images": dataset_count,
        "repeats": payload.repeats,
        "alpha": payload.alpha,
        "message": "training started (simulated worker)",
    }


@router.post("/stop-now")
def stop_now(payload: TrainingControlIn) -> dict:
    _ensure_project(payload.project_id)
    conn = get_conn()
    cur = conn.cursor()
    row = _latest_run(conn, payload.project_id)
    if row is None:
        conn.close()
        raise HTTPException(status_code=400, detail="no training run")
    cur.execute(
        """
        UPDATE training_runs
        SET status = 'paused', stop_mode = 'now', updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (row["id"],),
    )
    _set_project_status(conn, payload.project_id, "paused")
    conn.commit()
    conn.close()
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
    cur = conn.cursor()
    row = _latest_run(conn, payload.project_id)
    if row is None:
        conn.close()
        raise HTTPException(status_code=400, detail="no training run")
    if row["status"] != "training":
        conn.close()
        raise HTTPException(status_code=400, detail="run is not training")
    cur.execute(
        """
        UPDATE training_runs
        SET stop_mode = 'epoch', updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (row["id"],),
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
    _ensure_project(payload.project_id)
    conn = get_conn()
    cur = conn.cursor()
    row = _latest_run(conn, payload.project_id)
    if row is None:
        conn.close()
        raise HTTPException(status_code=400, detail="no training run")
    if row["status"] == "completed":
        conn.close()
        raise HTTPException(status_code=400, detail="run already completed")
    cur.execute(
        """
        UPDATE training_runs
        SET status = 'training', stop_mode = NULL, updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (row["id"],),
    )
    _set_project_status(conn, payload.project_id, "training")
    conn.commit()
    conn.close()
    _ensure_runner(payload.project_id)
    return {
        "project_id": payload.project_id,
        "status": "training",
        "message": "resume accepted",
    }


@router.get("/status")
def status(project_id: int) -> dict:
    _ensure_project(project_id)
    conn = get_conn()
    row = conn.execute(
        """
        SELECT id, status, stop_mode, latest_checkpoint_path, started_at, updated_at,
               current_epoch, current_step, total_epochs, steps_per_epoch
        FROM training_runs
        WHERE project_id = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (project_id,),
    ).fetchone()
    conn.close()
    if row is None:
        return {
            "project_id": project_id,
            "run_id": None,
            "status": "idle",
            "epoch": 0,
            "step": 0,
            "total_epochs": 0,
            "steps_per_epoch": 0,
            "stop_mode": None,
            "latest_checkpoint_path": None,
            "message": "no run yet",
        }
    total_steps = int(row["total_epochs"]) * int(row["steps_per_epoch"])
    done_steps = int(row["current_epoch"]) * int(row["steps_per_epoch"]) + int(row["current_step"])
    progress_percent = (done_steps / total_steps * 100.0) if total_steps > 0 else 0.0
    eta_seconds = None
    try:
        started_at = datetime.fromisoformat(str(row["started_at"]).replace(" ", "T"))
        elapsed = max(1.0, (datetime.now() - started_at).total_seconds())
        speed = done_steps / elapsed if done_steps > 0 else 0.0
        if speed > 0:
            eta_seconds = int(max(0.0, (total_steps - done_steps) / speed))
    except ValueError:
        eta_seconds = None
    return {
        "project_id": project_id,
        "run_id": row["id"],
        "status": row["status"],
        "epoch": row["current_epoch"],
        "step": row["current_step"],
        "total_epochs": row["total_epochs"],
        "steps_per_epoch": row["steps_per_epoch"],
        "total_steps": total_steps,
        "done_steps": done_steps,
        "progress_percent": round(progress_percent, 2),
        "eta_seconds": eta_seconds,
        "stop_mode": row["stop_mode"],
        "latest_checkpoint_path": row["latest_checkpoint_path"],
        "started_at": row["started_at"],
        "updated_at": row["updated_at"],
        "message": "status from simulated worker",
    }


@router.get("/dataset-preview")
def dataset_preview(project_id: int, train_data_dir: str = "") -> dict:
    _ensure_project(project_id)
    target = train_data_dir.strip()
    if not target:
        conn = get_conn()
        row = conn.execute("SELECT dataset_dir FROM projects WHERE id = ?", (project_id,)).fetchone()
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
