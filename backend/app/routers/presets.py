from __future__ import annotations

import json
from datetime import datetime

from fastapi import APIRouter, HTTPException

from ..db import get_conn
from ..schemas import PresetCreate, PresetUpdateIn

router = APIRouter(prefix="/presets", tags=["presets"])

def _validate_learning_rate(payload):
    from ..training.learning_rates import resolve_learning_rates
    config=payload.get('config',payload)
    if isinstance(config,dict) and 'learning_rate' in config:
        try:resolve_learning_rates(config,reject_conflict=True)
        except (ValueError,TypeError) as exc:raise HTTPException(422,str(exc)) from exc


# ── デフォルトプロファイル定義（§17.3） ────────────────────────────────────
DEFAULT_PRESETS: list[dict] = [
    {
        "name": "Character (Standard)",
        "payload": {
            "type": "character",
            "description": "キャラクターLoRA向け標準設定 — rank32 / alpha16 / SD1.5推奨",
            # rank32はキャラの顔・衣装等の特徴量を十分に捉えるのに必要
            # alpha = rank/2 が実績値（学習率の実効スケール調整）
            "rank": 32,
            "alpha": 16,
            "repeats": 10,       # 画像30〜50枚想定。steps/epoch = 画像数 × repeats / batch_size
            "epochs": 15,
            "resolution": 512,
            "learning_rate": 1e-4,
            "train_batch_size": 2,
            "optimizer": "AdamW8bit",
            "scheduler": "cosine_with_restarts",
            "save_every_n_epochs": 1,
            "min_snr_gamma": 5,
            "output_name": "lora_character",
            # perf
            "xformers": True,
            "cache_latents": True,
            "cache_latents_to_disk": False,
            "gradient_checkpointing": True,
            "mixed_precision": "bf16",
            "save_precision": "fp16",
            "persistent_data_loader_workers": True,
            "max_data_loader_n_workers": 4,
            "network_train_unet_only": False,  # text encoderも学習してキャラ特徴を埋め込む
        },
    },
    {
        "name": "Character (High Quality)",
        "payload": {
            "type": "character",
            "description": "高品質キャラクターLoRA — rank64 / Prodigy / 自動LR調整",
            # Prodigyは学習率を自動調整するため初心者でも破綻しにくい
            # rank64で細かい特徴（目の色・模様等）まで学習
            "rank": 64,
            "alpha": 32,
            "repeats": 10,
            "epochs": 20,
            "resolution": 768,
            "learning_rate": 1.0,   # Prodigyは1.0スタートが推奨
            "train_batch_size": 2,
            "optimizer": "Prodigy",
            "scheduler": "cosine_with_restarts",
            "save_every_n_epochs": 2,
            "min_snr_gamma": 5,
            "output_name": "lora_character_hq",
            # perf
            "xformers": True,
            "cache_latents": True,
            "cache_latents_to_disk": True,   # 長期学習なのでディスクキャッシュ推奨
            "gradient_checkpointing": True,
            "mixed_precision": "bf16",
            "save_precision": "fp16",
            "persistent_data_loader_workers": True,
            "max_data_loader_n_workers": 4,
            "network_train_unet_only": False,
        },
    },
    {
        "name": "Style (Standard)",
        "payload": {
            "type": "style",
            "description": "スタイルLoRA向け — rank32 / UNetのみ学習",
            # スタイルはtext encoderよりUNetで表現されるためunet_only有効
            "rank": 32,
            "alpha": 16,
            "repeats": 5,
            "epochs": 20,
            "resolution": 768,
            "learning_rate": 5e-5,  # スタイルは低めのLRで崩れを防ぐ
            "train_batch_size": 2,
            "optimizer": "AdamW8bit",
            "scheduler": "cosine",
            "save_every_n_epochs": 2,
            "min_snr_gamma": None,
            "output_name": "lora_style",
            # perf
            "xformers": True,
            "cache_latents": True,
            "cache_latents_to_disk": False,
            "gradient_checkpointing": True,
            "mixed_precision": "bf16",
            "save_precision": "fp16",
            "persistent_data_loader_workers": True,
            "max_data_loader_n_workers": 4,
            "network_train_unet_only": True,
        },
    },
    {
        "name": "Character (Quick Test)",
        "payload": {
            "type": "character",
            "description": "動作確認・過学習チェック用 — rank16 / 短期学習",
            "rank": 16,
            "alpha": 8,
            "repeats": 5,
            "epochs": 5,
            "resolution": 512,
            "learning_rate": 1e-4,
            "train_batch_size": 2,
            "optimizer": "AdamW8bit",
            "scheduler": "cosine",
            "save_every_n_epochs": 1,
            "min_snr_gamma": 5,
            "output_name": "lora_test",
            # perf
            "xformers": True,
            "cache_latents": True,
            "cache_latents_to_disk": False,
            "gradient_checkpointing": True,
            "mixed_precision": "bf16",
            "save_precision": "fp16",
            "persistent_data_loader_workers": True,
            "max_data_loader_n_workers": 2,
            "network_train_unet_only": False,
        },
    },
]


def _seed_defaults_internal() -> int:
    """デフォルトプリセット挿入または更新（名前が一致するものは常に上書き）"""
    conn = get_conn()
    upserted = 0
    for p in DEFAULT_PRESETS:
        existing = conn.execute(
            "SELECT id FROM presets WHERE name = ?", (p["name"],)
        ).fetchone()
        payload_str = json.dumps(p["payload"], ensure_ascii=False)
        if existing is None:
            conn.execute(
                "INSERT INTO presets(name, payload_json, created_at) VALUES(?, ?, ?)",
                (p["name"], payload_str, datetime.utcnow().isoformat()),
            )
        else:
            conn.execute(
                "UPDATE presets SET payload_json = ? WHERE id = ?",
                (payload_str, existing["id"]),
            )
        upserted += 1
    conn.commit()
    conn.close()
    return upserted


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
    _validate_learning_rate(body.payload)
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


# Training preset files (user-presets/training/*.json) live in training_preset_files.py.


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

    try:_validate_learning_rate(new_payload)
    except HTTPException:
        conn.close()
        raise
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
