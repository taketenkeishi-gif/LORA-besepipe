from __future__ import annotations

import json
from datetime import datetime

from fastapi import APIRouter, HTTPException

from ..db import get_conn
from ..schemas import PresetCreate, PresetUpdateIn

router = APIRouter(prefix="/presets", tags=["presets"])

# ── デフォルトプロファイル定義（§17.3） ────────────────────────────────────
DEFAULT_PRESETS: list[dict] = [
    {
        "name": "Character (Standard)",
        "payload": {
            "type": "character",
            "description": "キャラクターLoRA向け標準設定 — rank16 / alpha8",
            "rank": 16,
            "alpha": 8,
            "repeats": 5,
            "epochs": 10,
            "resolution": 512,
            "optimizer": "AdamW8bit",
            "scheduler": "cosine_with_restarts",
            "save_every_n_epochs": 1,
            "min_snr_gamma": 5,
            "output_name": "lora_character",
        },
    },
    {
        "name": "Style (Standard)",
        "payload": {
            "type": "style",
            "description": "スタイルLoRA向け標準設定 — rank32 / alpha16",
            "rank": 32,
            "alpha": 16,
            "repeats": 3,
            "epochs": 15,
            "resolution": 768,
            "optimizer": "AdamW8bit",
            "scheduler": "cosine",
            "save_every_n_epochs": 2,
            "min_snr_gamma": None,
            "output_name": "lora_style",
        },
    },
    {
        "name": "Hybrid (Standard)",
        "payload": {
            "type": "hybrid",
            "description": "キャラ＋スタイル混合設定 — rank32 / alpha16",
            "rank": 32,
            "alpha": 16,
            "repeats": 4,
            "epochs": 12,
            "resolution": 512,
            "optimizer": "AdamW8bit",
            "scheduler": "cosine_with_restarts",
            "save_every_n_epochs": 1,
            "min_snr_gamma": 5,
            "output_name": "lora_hybrid",
        },
    },
    {
        "name": "Character (Lightweight)",
        "payload": {
            "type": "character",
            "description": "高速テスト用 — rank8 / alpha4 / 短期学習",
            "rank": 8,
            "alpha": 4,
            "repeats": 3,
            "epochs": 6,
            "resolution": 512,
            "optimizer": "AdamW8bit",
            "scheduler": "cosine",
            "save_every_n_epochs": 1,
            "min_snr_gamma": 5,
            "output_name": "lora_light",
        },
    },
]


def _seed_defaults_internal() -> int:
    """デフォルトプリセット初期挿入（既存は上書きしない）"""
    conn = get_conn()
    inserted = 0
    for p in DEFAULT_PRESETS:
        existing = conn.execute(
            "SELECT id FROM presets WHERE name = ?", (p["name"],)
        ).fetchone()
        if existing is None:
            conn.execute(
                "INSERT INTO presets(name, payload_json, created_at) VALUES(?, ?, ?)",
                (
                    p["name"],
                    json.dumps(p["payload"], ensure_ascii=False),
                    datetime.utcnow().isoformat(),
                ),
            )
            inserted += 1
    conn.commit()
    conn.close()
    return inserted


@router.post("/seed-defaults", status_code=201)
def seed_defaults() -> dict:
    """デフォルトプリセットを初期挿入（既存プリセットには影響しない）"""
    inserted = _seed_defaults_internal()
    return {"inserted": inserted, "message": f"{inserted}件のデフォルトプリセットを追加しました"}


@router.get("")
def list_presets() -> list[dict]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, name, payload_json, created_at FROM presets ORDER BY id ASC"
    ).fetchall()
    conn.close()
    result = []
    for row in rows:
        try:
            payload = json.loads(row["payload_json"])
        except Exception:
            payload = {}
        result.append(
            {
                "id": row["id"],
                "name": row["name"],
                "payload": payload,
                "created_at": row["created_at"],
            }
        )
    return result


@router.post("", status_code=201)
def create_preset(body: PresetCreate) -> dict:
    conn = get_conn()
    existing = conn.execute(
        "SELECT id FROM presets WHERE name = ?", (body.name,)
    ).fetchone()
    if existing:
        conn.close()
        raise HTTPException(
            status_code=409,
            detail=f"プリセット名 '{body.name}' は既に使用されています",
        )
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO presets(name, payload_json, created_at) VALUES(?, ?, ?)",
        (
            body.name,
            json.dumps(body.payload, ensure_ascii=False),
            datetime.utcnow().isoformat(),
        ),
    )
    preset_id = cur.lastrowid
    conn.commit()
    conn.close()
    return {"id": preset_id, "name": body.name, "payload": body.payload}


@router.get("/{preset_id}")
def get_preset(preset_id: int) -> dict:
    conn = get_conn()
    row = conn.execute(
        "SELECT id, name, payload_json, created_at FROM presets WHERE id = ?",
        (preset_id,),
    ).fetchone()
    conn.close()
    if row is None:
        raise HTTPException(status_code=404, detail=f"preset not found: {preset_id}")
    try:
        payload = json.loads(row["payload_json"])
    except Exception:
        payload = {}
    return {
        "id": row["id"],
        "name": row["name"],
        "payload": payload,
        "created_at": row["created_at"],
    }


@router.put("/{preset_id}")
def update_preset(preset_id: int, body: PresetUpdateIn) -> dict:
    conn = get_conn()
    row = conn.execute(
        "SELECT id, name, payload_json FROM presets WHERE id = ?", (preset_id,)
    ).fetchone()
    if row is None:
        conn.close()
        raise HTTPException(status_code=404, detail=f"preset not found: {preset_id}")

    new_name = body.name if body.name is not None else row["name"]

    # Check name conflict if renaming
    if body.name is not None and body.name != row["name"]:
        conflict = conn.execute(
            "SELECT id FROM presets WHERE name = ? AND id != ?", (body.name, preset_id)
        ).fetchone()
        if conflict:
            conn.close()
            raise HTTPException(
                status_code=409,
                detail=f"プリセット名 '{body.name}' は既に使用されています",
            )

    old_payload: dict = {}
    try:
        old_payload = json.loads(row["payload_json"])
    except Exception:
        pass

    if body.payload is not None:
        new_payload = {**old_payload, **body.payload}
    else:
        new_payload = old_payload

    conn.execute(
        "UPDATE presets SET name = ?, payload_json = ? WHERE id = ?",
        (new_name, json.dumps(new_payload, ensure_ascii=False), preset_id),
    )
    conn.commit()
    conn.close()
    return {"id": preset_id, "name": new_name, "payload": new_payload}


@router.delete("/{preset_id}")
def delete_preset(preset_id: int) -> dict:
    conn = get_conn()
    row = conn.execute("SELECT id FROM presets WHERE id = ?", (preset_id,)).fetchone()
    if row is None:
        conn.close()
        raise HTTPException(status_code=404, detail=f"preset not found: {preset_id}")
    conn.execute("DELETE FROM presets WHERE id = ?", (preset_id,))
    conn.commit()
    conn.close()
    return {"id": preset_id, "deleted": True}
