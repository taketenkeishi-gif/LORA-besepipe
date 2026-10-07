"""Portable, user-owned common training recipes.

This router deliberately stores only knobs that can travel between projects.
Dataset folders, triggers, output names and run/snapshot/resume identifiers
remain project or run data.
"""
from __future__ import annotations

import json
import math
import uuid
import re

from fastapi import APIRouter, HTTPException

from ..db import get_conn

router = APIRouter(prefix="/training-recipe", tags=["training-recipe"])

_COMMON_KEY = "training_recipe:common"
_PROJECT_KEY = "training_recipe:project:"
_PORTABLE = {
    "model_family", "base_checkpoint_path", "rank", "alpha", "epochs", "repeats",
    "resolution", "learning_rate", "train_batch_size", "save_every_n_epochs",
    "optimizer", "scheduler", "min_snr_gamma", "mixed_precision", "save_precision",
    "xformers", "cache_latents", "cache_latents_to_disk", "gradient_checkpointing",
    "persistent_data_loader_workers", "max_data_loader_n_workers",
    "network_train_unet_only", "training_memory_mode", "advanced",
    "preview_backend", "preview_lora_strength", "preview_steps", "preview_cfg",
}
_INTS = {"rank": (1, 512), "epochs": (1, 1000), "repeats": (1, 1000),
         "resolution": (256, 2048), "train_batch_size": (1, 64),
         "save_every_n_epochs": (1, 1000), "max_data_loader_n_workers": (0, 16),
         "preview_steps": (1, 150), "min_snr_gamma": (0, 20)}
_FLOATS = {"alpha": (0.1, 128), "learning_rate": (0, 1),
           "preview_lora_strength": (0, 2), "preview_cfg": (0, 30)}
_BOOLS = {"xformers", "cache_latents", "cache_latents_to_disk", "gradient_checkpointing",
          "persistent_data_loader_workers", "network_train_unet_only"}


def _read(key: str) -> dict | None:
    conn = get_conn()
    try:
        row = conn.execute("SELECT value FROM app_settings WHERE key=?", (key,)).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    try:
        value = json.loads(row["value"])
        return value if isinstance(value, dict) else None
    except (TypeError, ValueError):
        return None


def _put(key: str, value: dict) -> None:
    # One JSON document is committed atomically per setting key.
    conn = get_conn()
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("INSERT INTO app_settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_at=CURRENT_TIMESTAMP",
                     (key, json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _config(value: object) -> dict:
    if not isinstance(value, dict):
        raise HTTPException(422, "config はオブジェクトで指定してください")
    unknown = set(value) - _PORTABLE
    if unknown:
        raise HTTPException(422, "共通レシピに保存できない項目です: " + ", ".join(sorted(unknown)))
    clean = dict(value)
    family = clean.get("model_family")
    if family not in {"anima", "sdxl"}:
        raise HTTPException(422, "model_family は anima または sdxl を指定してください")
    model = clean.get("base_checkpoint_path")
    if not isinstance(model, str) or not model.strip() or len(model) > 4096:
        raise HTTPException(422, "base_checkpoint_path を指定してください")
    for key, (low, high) in _INTS.items():
        if key in clean and (isinstance(clean[key], bool) or not isinstance(clean[key], int) or not low <= clean[key] <= high):
            raise HTTPException(422, f"{key} の値が不正です")
    for key, (low, high) in _FLOATS.items():
        if key in clean:
            number = clean[key]
            if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or not low < number <= high:
                raise HTTPException(422, f"{key} の値が不正です")
    for key in _BOOLS:
        if key in clean and not isinstance(clean[key], bool):
            raise HTTPException(422, f"{key} は真偽値で指定してください")
    for key in ("optimizer", "scheduler", "mixed_precision", "save_precision", "preview_backend", "training_memory_mode"):
        if key in clean and (not isinstance(clean[key], str) or len(clean[key]) > 64):
            raise HTTPException(422, f"{key} の値が不正です")
    if clean.get("training_memory_mode", "standard") not in {"low_vram", "balanced", "standard"}:
        raise HTTPException(422, "training_memory_mode の値が不正です")
    try:
        from ..training.learning_rates import resolve_learning_rates
        resolve_learning_rates(clean, reject_conflict=True)
        from ..training.advanced import validate
        clean["advanced"] = validate(clean.get("advanced", {}), family, clean)
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, str(exc)) from exc
    return clean


def _draft(project_id: int) -> dict:
    conn = get_conn()
    try:
        row = conn.execute("SELECT training_config_json FROM projects WHERE id=?", (project_id,)).fetchone()
    finally:
        conn.close()
    if row is None:
        raise HTTPException(404, "project not found")
    try:
        value = json.loads(row["training_config_json"] or "{}")
        return value if isinstance(value, dict) else {}
    except (TypeError, ValueError):
        return {}


@router.get("")
def get_common_recipe() -> dict:
    saved = _read(_COMMON_KEY)
    if not saved:
        return {"available": False, "revision": None, "config": {}, "source": None}
    return {"available": True, "revision": saved.get("revision"), "config": saved.get("config", {}), "source": saved.get("source", "user")}


@router.put("")
def put_common_recipe(payload: dict) -> dict:
    config = _config(payload.get("config", payload))
    saved = {"revision": str(uuid.uuid4()), "config": config, "source": "user"}
    _put(_COMMON_KEY, saved)
    return {"available": True, **saved}


@router.get("/project/{project_id}")
def get_project_recipe(project_id: int) -> dict:
    draft = _draft(project_id)
    common = _read(_COMMON_KEY)
    if not common:
        return {"project_id": project_id, "available": False, "revision": None, "config": draft, "source": "project_draft"}
    config = dict(draft)
    config.update(common.get("config", {}))
    override = _read(f"{_PROJECT_KEY}{project_id}")
    if override and override.get("revision") == common.get("revision") and isinstance(override.get("config"), dict):
        config.update(override["config"])
        source = "common_with_project_override"
    else:
        source = "common"
    return {"project_id": project_id, "available": True, "revision": common.get("revision"), "config": config, "source": source}


@router.put("/project/{project_id}")
def put_project_recipe(project_id: int, payload: dict) -> dict:
    common = _read(_COMMON_KEY)
    if not common:
        raise HTTPException(409, "共通レシピが保存されていません")
    supplied = payload.get("config", payload)
    if not isinstance(supplied, dict):
        raise HTTPException(422, "config はオブジェクトで指定してください")
    # Persist the caller's complete existing-draft shape through its canonical route.
    from .training import patch_training_config_draft
    saved_draft = patch_training_config_draft(project_id, supplied)["config"]
    portable = _config({key: value for key, value in supplied.items() if key in _PORTABLE})
    _put(f"{_PROJECT_KEY}{project_id}", {"revision": common["revision"], "config": portable})
    return {"project_id": project_id, "available": True, "revision": common["revision"], "config": saved_draft, "source": "project_override"}


def _caption_with_trigger(caption: object, token: str) -> str:
    tags = [part.strip() for part in re.split(r"[,\n]", str(caption or "")) if part.strip()]
    if token.casefold() not in {part.casefold() for part in tags}:
        tags.insert(0, token)
    return ", ".join(tags)


@router.post("/project/{project_id}/prepare")
def prepare_project_dataset(project_id: int, payload: dict) -> dict:
    """Freeze the selected direct folder as training input; never starts a GPU job.

    Sidecar TXT files remain untouched.  The trigger is appended to the
    registered asset's DB caption before snapshotting, so the sealed manifest
    is deterministic and retains its source-file hash.
    """
    folder = payload.get("folder", "")
    token = str(payload.get("trigger_token") or "").strip()
    if not isinstance(folder, str) or len(folder) > 1024:
        raise HTTPException(422, "folder の値が不正です")
    if not token or len(token) > 120 or any(char in token for char in (",", "\n", "\r", "\x00")):
        raise HTTPException(422, "trigger_token を1つ指定してください")
    from . import dataset_files
    from .evaluation import digest, read_reference
    root = dataset_files.root_for(project_id)
    directory = dataset_files.within(root, folder)
    if not directory.is_dir():
        raise HTTPException(404, "フォルダが見つかりません")
    listed = dataset_files.listing(project_id, folder)
    relatives = [str(item["relative"]) for item in listed["items"]]
    if not relatives:
        raise HTTPException(422, "このフォルダには直接の画像がありません")
    reference = read_reference(project_id)
    candidates = []
    # Validate every source before changing concepts/assets.  ``listing`` is
    # intentionally non-recursive, and this repeats its path guard.
    for relative in relatives:
        image = dataset_files.image_path(root, relative)
        if reference and digest(image) == reference["sha256"]:
            continue
        dataset_files.require_editable_source(image, allow_snapshot_reference=True)
        candidates.append(relative)
    if not candidates:
        raise HTTPException(422, "評価画像を除く学習画像がありません")

    from .training import save_training_trigger
    from .basepipe import bulk_review_assets, create_snapshot, get_snapshot
    from ..schemas import BasepipeAssetBulkReviewIn, BasepipeSnapshotCreate
    save_training_trigger(project_id, {"trigger_token": token})
    registered = dataset_files.register(project_id, dataset_files.Paths(relatives=candidates))
    asset_ids = [int(asset_id) for asset_id in registered["asset_ids"]]
    if not asset_ids:
        raise HTTPException(422, "評価画像を除く学習画像がありません")
    conn = get_conn()
    try:
        marks = ",".join("?" for _ in asset_ids)
        rows = conn.execute(f"SELECT id,caption FROM basepipe_assets WHERE project_id=? AND id IN ({marks})", (project_id, *asset_ids)).fetchall()
        if len(rows) != len(asset_ids):
            raise HTTPException(409, "登録した画像が変わりました。再度準備してください")
        for row in rows:
            caption = _caption_with_trigger(row["caption"], token)
            conn.execute("UPDATE basepipe_assets SET caption=?,caption_edited=?,training_input=?,training_input_source='recipe_prepare',caption_source='recipe_prepare',updated_at=CURRENT_TIMESTAMP WHERE id=?",
                         (caption, caption, caption, int(row["id"])))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    bulk_review_assets(project_id, BasepipeAssetBulkReviewIn(asset_ids=asset_ids, review_status="approved", training_enabled=True))
    snapshot = create_snapshot(project_id, BasepipeSnapshotCreate(name=f"{directory.name or 'dataset'} · {token}", asset_ids=asset_ids))
    frozen = get_snapshot(int(snapshot["id"]))
    return {
        "project_id": project_id,
        "snapshot": frozen,
        "registered_asset_ids": asset_ids,
        "excluded_evaluation_images": registered["excluded_evaluation_images"],
        "disclosure": "フォルダ直下の画像をすべて使い、評価画像は除外しました。学習用のDBキャプションへトリガーを追加し、元画像とTXTは変更していません。",
    }
