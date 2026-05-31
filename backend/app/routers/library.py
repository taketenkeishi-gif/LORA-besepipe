"""
§22-23 Asset Library — LoRA 資産管理 CRUD
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel

from ..db import get_conn

router = APIRouter(prefix="/library", tags=["library"])


# ── Pydantic スキーマ ─────────────────────────────────────────────────────────

class AssetCreateIn(BaseModel):
    project_id: int | None = None
    name: str
    lora_path: str = ""
    base_model: str = ""
    dataset_size: int = 0
    profile_name: str = ""
    tags: list[str] = []
    notes: str = ""
    preview_path: str = ""
    training_config: dict[str, Any] = {}
    quality_score: int | None = None
    asset_type: str = "character"


class AssetUpdateIn(BaseModel):
    name: str | None = None
    lora_path: str | None = None
    base_model: str | None = None
    dataset_size: int | None = None
    profile_name: str | None = None
    tags: list[str] | None = None
    notes: str | None = None
    preview_path: str | None = None
    training_config: dict[str, Any] | None = None
    quality_score: int | None = None
    asset_type: str | None = None


# ── ヘルパー ──────────────────────────────────────────────────────────────────

def _row_to_dict(row) -> dict:
    d = dict(row)
    # tags_json → tags list
    try:
        d["tags"] = json.loads(d.pop("tags_json", "[]"))
    except Exception:
        d["tags"] = []
    # training_config_json → dict
    try:
        d["training_config"] = json.loads(d.pop("training_config_json", "{}"))
    except Exception:
        d["training_config"] = {}
    return d


# ── エンドポイント ────────────────────────────────────────────────────────────

@router.get("/assets")
def list_assets(project_id: int | None = None, asset_type: str | None = None) -> list[dict]:
    """LoRA 資産一覧を取得する。project_id / asset_type でフィルタ可能。"""
    conn = get_conn()
    query = "SELECT * FROM lora_assets WHERE 1=1"
    params: list = []
    if project_id is not None:
        query += " AND project_id = ?"
        params.append(project_id)
    if asset_type is not None:
        query += " AND asset_type = ?"
        params.append(asset_type)
    query += " ORDER BY created_at DESC"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [_row_to_dict(r) for r in rows]


@router.post("/assets", status_code=201)
def create_asset(payload: AssetCreateIn) -> dict:
    """LoRA 資産を新規作成する。"""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO lora_assets (
            project_id, name, lora_path, base_model, dataset_size,
            profile_name, tags_json, notes, preview_path,
            training_config_json, quality_score, asset_type
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            payload.project_id,
            payload.name,
            payload.lora_path,
            payload.base_model,
            payload.dataset_size,
            payload.profile_name,
            json.dumps(payload.tags, ensure_ascii=False),
            payload.notes,
            payload.preview_path,
            json.dumps(payload.training_config, ensure_ascii=False),
            payload.quality_score,
            payload.asset_type,
        ),
    )
    asset_id = cur.lastrowid
    conn.commit()
    row = conn.execute("SELECT * FROM lora_assets WHERE id = ?", (asset_id,)).fetchone()
    conn.close()
    return _row_to_dict(row)


@router.get("/assets/{asset_id}")
def get_asset(asset_id: int) -> dict:
    """LoRA 資産を 1 件取得する。"""
    conn = get_conn()
    row = conn.execute("SELECT * FROM lora_assets WHERE id = ?", (asset_id,)).fetchone()
    conn.close()
    if row is None:
        raise HTTPException(status_code=404, detail=f"asset not found: {asset_id}")
    return _row_to_dict(row)


@router.put("/assets/{asset_id}")
def update_asset(asset_id: int, payload: AssetUpdateIn) -> dict:
    """LoRA 資産を更新する（指定フィールドのみ上書き）。"""
    conn = get_conn()
    row = conn.execute("SELECT * FROM lora_assets WHERE id = ?", (asset_id,)).fetchone()
    if row is None:
        conn.close()
        raise HTTPException(status_code=404, detail=f"asset not found: {asset_id}")

    updates: list[str] = []
    params: list = []

    if payload.name is not None:
        updates.append("name = ?"); params.append(payload.name)
    if payload.lora_path is not None:
        updates.append("lora_path = ?"); params.append(payload.lora_path)
    if payload.base_model is not None:
        updates.append("base_model = ?"); params.append(payload.base_model)
    if payload.dataset_size is not None:
        updates.append("dataset_size = ?"); params.append(payload.dataset_size)
    if payload.profile_name is not None:
        updates.append("profile_name = ?"); params.append(payload.profile_name)
    if payload.tags is not None:
        updates.append("tags_json = ?"); params.append(json.dumps(payload.tags, ensure_ascii=False))
    if payload.notes is not None:
        updates.append("notes = ?"); params.append(payload.notes)
    if payload.preview_path is not None:
        updates.append("preview_path = ?"); params.append(payload.preview_path)
    if payload.training_config is not None:
        updates.append("training_config_json = ?"); params.append(json.dumps(payload.training_config, ensure_ascii=False))
    if payload.quality_score is not None:
        updates.append("quality_score = ?"); params.append(payload.quality_score)
    if payload.asset_type is not None:
        updates.append("asset_type = ?"); params.append(payload.asset_type)

    if updates:
        params.append(asset_id)
        conn.execute(f"UPDATE lora_assets SET {', '.join(updates)} WHERE id = ?", params)
        conn.commit()

    row = conn.execute("SELECT * FROM lora_assets WHERE id = ?", (asset_id,)).fetchone()
    conn.close()
    return _row_to_dict(row)


@router.delete("/assets/{asset_id}")
def delete_asset(asset_id: int) -> Response:
    """LoRA 資産を削除する（ファイルは削除しない）。"""
    conn = get_conn()
    row = conn.execute("SELECT id FROM lora_assets WHERE id = ?", (asset_id,)).fetchone()
    if row is None:
        conn.close()
        raise HTTPException(status_code=404, detail=f"asset not found: {asset_id}")
    conn.execute("DELETE FROM lora_assets WHERE id = ?", (asset_id,))
    conn.commit()
    conn.close()
    return Response(status_code=204)


@router.get("/stats")
def library_stats() -> dict:
    """ライブラリ全体の統計情報を返す。"""
    conn = get_conn()
    total = conn.execute("SELECT COUNT(*) FROM lora_assets").fetchone()[0]
    by_type = conn.execute(
        "SELECT asset_type, COUNT(*) AS cnt FROM lora_assets GROUP BY asset_type"
    ).fetchall()
    avg_qs = conn.execute(
        "SELECT AVG(quality_score) FROM lora_assets WHERE quality_score IS NOT NULL"
    ).fetchone()[0]
    conn.close()
    return {
        "total": total,
        "by_type": {row["asset_type"]: row["cnt"] for row in by_type},
        "avg_quality_score": round(avg_qs, 1) if avg_qs is not None else None,
    }
