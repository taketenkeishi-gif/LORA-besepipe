from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException

from ..db import get_conn

from ..schemas import TrainingControlIn, TrainingStartIn

router = APIRouter(prefix="/training", tags=["training"])


def _ensure_project(project_id: int) -> None:
    conn = get_conn()
    row = conn.execute("SELECT id FROM projects WHERE id = ?", (project_id,)).fetchone()
    conn.close()
    if row is None:
        raise HTTPException(status_code=404, detail=f"project not found: {project_id}")


def _set_project_status(conn, project_id: int, status: str) -> None:
    conn.execute("UPDATE projects SET status = ? WHERE id = ?", (status, project_id))


@router.post("/start")
def start(payload: TrainingStartIn) -> dict:
    _ensure_project(payload.project_id)
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO training_runs(project_id, status, stop_mode, latest_checkpoint_path, config_json)
        VALUES (?, 'training', NULL, NULL, ?)
        """,
        (payload.project_id, json.dumps({"preset_id": payload.preset_id})),
    )
    run_id = cur.lastrowid
    _set_project_status(conn, payload.project_id, "training")
    conn.commit()
    conn.close()
    return {
        "run_id": run_id,
        "project_id": payload.project_id,
        "preset_id": payload.preset_id,
        "status": "training",
        "message": "training run created (CLI実行は未実装)",
    }


@router.post("/stop-now")
def stop_now(payload: TrainingControlIn) -> dict:
    _ensure_project(payload.project_id)
    conn = get_conn()
    cur = conn.cursor()
    row = cur.execute(
        """
        SELECT id FROM training_runs
        WHERE project_id = ? ORDER BY id DESC LIMIT 1
        """,
        (payload.project_id,),
    ).fetchone()
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
        "message": "stop_now accepted (安全停止ロジックは未実装)",
    }


@router.post("/stop-at-epoch")
def stop_at_epoch(payload: TrainingControlIn) -> dict:
    _ensure_project(payload.project_id)
    conn = get_conn()
    cur = conn.cursor()
    row = cur.execute(
        """
        SELECT id FROM training_runs
        WHERE project_id = ? ORDER BY id DESC LIMIT 1
        """,
        (payload.project_id,),
    ).fetchone()
    if row is None:
        conn.close()
        raise HTTPException(status_code=400, detail="no training run")
    cur.execute(
        """
        UPDATE training_runs
        SET status = 'paused', stop_mode = 'epoch', updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (row["id"],),
    )
    _set_project_status(conn, payload.project_id, "paused")
    conn.commit()
    conn.close()
    return {
        "project_id": payload.project_id,
        "mode": "stop_at_epoch",
        "status": "paused",
        "message": "stop_at_epoch accepted (epoch終端停止ロジックは未実装)",
    }


@router.post("/resume")
def resume(payload: TrainingControlIn) -> dict:
    _ensure_project(payload.project_id)
    conn = get_conn()
    cur = conn.cursor()
    row = cur.execute(
        """
        SELECT id FROM training_runs
        WHERE project_id = ? ORDER BY id DESC LIMIT 1
        """,
        (payload.project_id,),
    ).fetchone()
    if row is None:
        conn.close()
        raise HTTPException(status_code=400, detail="no training run")
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
    return {
        "project_id": payload.project_id,
        "status": "training",
        "message": "resume accepted (checkpoint再開ロジックは未実装)",
    }


@router.get("/status")
def status(project_id: int) -> dict:
    _ensure_project(project_id)
    conn = get_conn()
    row = conn.execute(
        """
        SELECT id, status, stop_mode, latest_checkpoint_path, started_at, updated_at
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
            "stop_mode": None,
            "latest_checkpoint_path": None,
            "message": "no run yet",
        }
    return {
        "project_id": project_id,
        "run_id": row["id"],
        "status": row["status"],
        "epoch": 0,
        "step": 0,
        "stop_mode": row["stop_mode"],
        "latest_checkpoint_path": row["latest_checkpoint_path"],
        "started_at": row["started_at"],
        "updated_at": row["updated_at"],
        "message": "stub status from sqlite",
    }
