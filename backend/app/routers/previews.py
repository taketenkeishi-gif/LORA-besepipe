from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..db import get_conn

router = APIRouter(prefix="/previews", tags=["previews"])


@router.get("/{project_id}")
def list_previews(project_id: int) -> dict:
    conn = get_conn()
    project = conn.execute("SELECT id FROM projects WHERE id = ?", (project_id,)).fetchone()
    if project is None:
        conn.close()
        raise HTTPException(status_code=404, detail=f"project not found: {project_id}")

    rows = conn.execute(
        """
        SELECT c.id AS checkpoint_id, c.epoch, c.step, c.mark, c.created_at,
               ps.slot, ps.image_path
        FROM checkpoints c
        LEFT JOIN preview_samples ps ON ps.checkpoint_id = c.id
        WHERE c.project_id = ?
        ORDER BY c.epoch DESC, c.id DESC
        """,
        (project_id,),
    ).fetchall()
    conn.close()

    timeline_map: dict[int, dict] = {}
    for row in rows:
        checkpoint_id = int(row["checkpoint_id"])
        item = timeline_map.get(checkpoint_id)
        if item is None:
            item = {
                "checkpoint_id": checkpoint_id,
                "epoch": int(row["epoch"]),
                "step": int(row["step"] or 0),
                "mark": row["mark"],
                "created_at": row["created_at"],
                "samples": {},
            }
            timeline_map[checkpoint_id] = item
        if row["slot"] is not None:
            item["samples"][row["slot"]] = row["image_path"]

    timeline = list(timeline_map.values())
    return {
        "project_id": project_id,
        "slots": ["face", "bust", "full", "bg"],
        "timeline": timeline,
        "message": "preview timeline loaded",
    }
