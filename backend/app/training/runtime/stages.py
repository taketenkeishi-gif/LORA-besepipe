"""Training Runの現在Stageと履歴を一貫して記録する。"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from ...db import get_conn


def set_training_stage(run_id: int, stage: str, status: str = "running", detail: str = "") -> None:
    conn = get_conn()
    try:
        row = conn.execute("SELECT stage_history_json FROM training_runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            return
        try:
            history = json.loads(row["stage_history_json"] or "[]")
        except (TypeError, ValueError):
            history = []
        if not isinstance(history, list):
            history = []
        event_key = {"stage": stage, "status": status, "detail": detail}
        last_key = {
            key: history[-1].get(key)
            for key in ("stage", "status", "detail")
        } if history and isinstance(history[-1], dict) else None
        event = {**event_key, "at": datetime.now(timezone.utc).isoformat()}
        if last_key != event_key:
            history.append(event)
        conn.execute(
            "UPDATE training_runs SET current_stage = ?, stage_history_json = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (stage, json.dumps(history, ensure_ascii=False), run_id),
        )
        conn.commit()
    finally:
        conn.close()
