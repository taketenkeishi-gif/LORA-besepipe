"""Dev Bridge debug endpoints — development only, not exposed in production."""
from __future__ import annotations

import json
import os
import subprocess
import sys

from fastapi import APIRouter

from ..db import get_conn


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


_CACHED_GIT_COMMIT: str | None = _git_commit()

router = APIRouter(prefix="/debug", tags=["debug"])


@router.get("/health")
def debug_health() -> dict:
    return {
        "running": True,
        "version": "0.1.0",
        "appName": "LoraBasepipe",
        # Runtime identity — required by Dev Bridge attach
        "process_id": os.getpid(),
        "executable_path": sys.executable,
        "git_commit": _CACHED_GIT_COMMIT,
    }


@router.get("/state")
def debug_state() -> dict:
    conn = get_conn()
    try:
        projects = conn.execute(
            "SELECT id, name, status FROM projects ORDER BY id"
        ).fetchall()
        project_list = [dict(r) for r in projects]
        project_count = len(project_list)

        # 学習中 > active > 先頭 の優先順で active project を決定
        active_project = (
            next((p for p in project_list if p["status"] == "training"), None)
            or next((p for p in project_list if p["status"] == "active"), None)
            or (project_list[0] if project_list else None)
        )
        active_project_id = active_project["id"] if active_project else None

        dataset_count = 0
        captioned_count = 0
        if active_project_id is not None:
            row = conn.execute(
                "SELECT COUNT(*) as total, SUM(CASE WHEN caption IS NOT NULL AND caption != '' THEN 1 ELSE 0 END) as captioned "
                "FROM dataset_items WHERE project_id = ?",
                (active_project_id,),
            ).fetchone()
            if row:
                dataset_count = row["total"] or 0
                captioned_count = row["captioned"] or 0

        latest_run = None
        if active_project_id is not None:
            run = conn.execute(
                "SELECT id, status, current_epoch, total_epochs, current_step, steps_per_epoch, loss "
                "FROM training_runs WHERE project_id = ? ORDER BY id DESC LIMIT 1",
                (active_project_id,),
            ).fetchone()
            if run:
                latest_run = dict(run)

        lora_count = conn.execute("SELECT COUNT(*) as c FROM lora_assets").fetchone()["c"]

        return {
            "project_count": project_count,
            "active_project_id": active_project_id,
            "dataset_count": dataset_count,
            "captioned_count": captioned_count,
            "caption_rate": round(captioned_count / dataset_count, 3) if dataset_count else 0.0,
            "lora_count": lora_count,
            "latest_run": latest_run,
            "projects": project_list,
        }
    finally:
        conn.close()


@router.post("/action")
def debug_action(body: dict) -> dict:
    """Minimal action dispatch for Dev Bridge scenarios."""
    action_type = body.get("type", "")

    if action_type == "list_projects":
        conn = get_conn()
        try:
            rows = conn.execute("SELECT id, name, status FROM projects ORDER BY id").fetchall()
            return {"success": True, "data": [dict(r) for r in rows]}
        finally:
            conn.close()

    if action_type == "get_dataset_stats":
        project_id = body.get("project_id")
        if not project_id:
            return {"success": False, "error": "project_id required"}
        conn = get_conn()
        try:
            row = conn.execute(
                "SELECT COUNT(*) as total, SUM(CASE WHEN caption IS NOT NULL AND caption != '' THEN 1 ELSE 0 END) as captioned "
                "FROM dataset_items WHERE project_id = ?",
                (project_id,),
            ).fetchone()
            total = row["total"] or 0
            captioned = row["captioned"] or 0
            return {"success": True, "data": {"total": total, "captioned": captioned, "rate": round(captioned / total, 3) if total else 0.0}}
        finally:
            conn.close()

    if action_type == "simulate_training_complete":
        # テスト用: training_run を completed にして _auto_register_lora_asset を呼ぶ
        project_id = body.get("project_id")
        if not project_id:
            return {"success": False, "error": "project_id required"}
        from ..routers.training import _auto_register_lora_asset
        conn = get_conn()
        try:
            # 既存の completed run があれば再利用、なければ挿入
            run = conn.execute(
                "SELECT id FROM training_runs WHERE project_id = ? AND status = 'completed' ORDER BY id DESC LIMIT 1",
                (project_id,),
            ).fetchone()
            if run:
                run_id = int(run["id"])
            else:
                cur = conn.execute(
                    """INSERT INTO training_runs
                       (project_id, status, total_epochs, current_epoch, steps_per_epoch, current_step, config_json)
                       VALUES (?, 'completed', 1, 1, 2, 2, ?)""",
                    (project_id, json.dumps({"base_model": "debug_model", "preset_name": "debug"})),
                )
                run_id = cur.lastrowid
            _auto_register_lora_asset(conn, project_id, run_id)
            conn.commit()
            lora_count = conn.execute("SELECT COUNT(*) as c FROM lora_assets").fetchone()["c"]
            return {"success": True, "run_id": run_id, "lora_count": lora_count}
        finally:
            conn.close()

    return {"success": False, "error": f"unknown action type: {action_type}"}
