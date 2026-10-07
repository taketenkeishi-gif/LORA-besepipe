from __future__ import annotations

import json

from fastapi import APIRouter
from pydantic import BaseModel, Field

from ..db import get_conn

router = APIRouter(prefix="/preview-profiles", tags=["preview-profiles"])


class PreviewProfileIn(BaseModel):
    project_id: int | None = None
    name: str = Field(min_length=1, max_length=120)
    prompt: str = ""
    negative_prompt: str = ""
    seed: int = 42
    resolution: int = 1024
    steps: int = 20
    cfg: float = 7.0
    sampler: str = ""
    scheduler: str = "simple"
    model_family: str = ""
    # per-outfit (training instance) preview settings: [{"trigger": str, "enabled": bool, "tags": [str]}]; None/empty = automatic
    outfits: list[dict] | None = None


def _outfits_json(payload: "PreviewProfileIn") -> str:
    rows = []
    for o in payload.outfits or []:
        trig = str(o.get("trigger", "")).strip()
        if trig:
            rows.append({"trigger": trig, "enabled": bool(o.get("enabled", True)), "tags": [str(t).strip() for t in o.get("tags", []) if str(t).strip()],
                         "negative_tags": [str(t).strip() for t in o.get("negative_tags", []) if str(t).strip()]})
    return json.dumps(rows, ensure_ascii=False) if rows else ""


@router.get("")
def list_profiles(project_id: int | None = None) -> dict:
    conn = get_conn()
    if project_id is None:
        rows = conn.execute("SELECT * FROM preview_profiles WHERE project_id IS NULL ORDER BY updated_at DESC, id DESC").fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM preview_profiles WHERE project_id = ? OR project_id IS NULL ORDER BY project_id IS NULL, updated_at DESC, id DESC",
            (project_id,),
        ).fetchall()
    conn.close()
    return {"profiles": [dict(row) for row in rows]}


@router.post("", status_code=201)
def create_profile(payload: PreviewProfileIn) -> dict:
    conn = get_conn()
    cur = conn.execute(
        """INSERT INTO preview_profiles
           (project_id, name, prompt, negative_prompt, seed, resolution, steps, cfg, sampler, scheduler, model_family, outfits_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (payload.project_id, payload.name.strip(), payload.prompt, payload.negative_prompt, payload.seed,
         payload.resolution, payload.steps, payload.cfg, payload.sampler, payload.scheduler, payload.model_family, _outfits_json(payload)),
    )
    conn.commit()
    row = conn.execute("SELECT * FROM preview_profiles WHERE id = ?", (cur.lastrowid,)).fetchone()
    conn.close()
    return {"profile": dict(row)}


@router.put("/{profile_id}")
def update_profile(profile_id: int, payload: PreviewProfileIn) -> dict:
    conn = get_conn()
    conn.execute(
        """UPDATE preview_profiles SET project_id=?, name=?, prompt=?, negative_prompt=?, seed=?, resolution=?,
           steps=?, cfg=?, sampler=?, scheduler=?, model_family=?, outfits_json=?, updated_at=CURRENT_TIMESTAMP WHERE id=?""",
        (payload.project_id, payload.name.strip(), payload.prompt, payload.negative_prompt, payload.seed,
         payload.resolution, payload.steps, payload.cfg, payload.sampler, payload.scheduler, payload.model_family, _outfits_json(payload), profile_id),
    )
    conn.commit()
    row = conn.execute("SELECT * FROM preview_profiles WHERE id = ?", (profile_id,)).fetchone()
    conn.close()
    return {"profile": dict(row) if row else None}
