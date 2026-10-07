"""Training preset files: user-presets/training/*.json (ordinary JSON, visible in Explorer).

The file format is unchanged ({format:'lora-studio-training-preset', version:1, name, category, config, ...}).
Every operation addresses a preset by its bare file name ("id") and is confined to the preset folder.
The folder can be redirected with the LORA_STUDIO_PRESETS_DIR environment variable (used by tests).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .presets import _validate_learning_rate

router = APIRouter(prefix="/presets/training-files", tags=["training-preset-files"])

FORMAT = "lora-studio-training-preset"
MAX_BYTES = 1024 * 1024
_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def preset_dir() -> Path:
    override = os.environ.get("LORA_STUDIO_PRESETS_DIR")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[3] / "user-presets" / "training"


class SaveBody(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    payload: dict
    overwrite: bool = False


class NameBody(BaseModel):
    name: str | None = Field(default=None, max_length=120)


class RevealBody(BaseModel):
    id: str | None = None


def _clean_name(raw: str) -> str:
    name = (raw or "").strip()
    if (not name or name.endswith(".") or re.search(r'[<>:"/\\|?*\x00-\x1f]', name)
            or name.split(".")[0].upper() in _RESERVED):
        raise HTTPException(400, "プリセット名にファイル名として使えない文字があります")
    return name


def _file_for_name(name: str) -> Path:
    return preset_dir() / f"preset-{name}.json"


def _resolve(preset_id: str) -> Path:
    """Map an id (bare file name) to a file inside the preset folder; refuse anything else."""
    if (not preset_id or preset_id != Path(preset_id).name or not preset_id.lower().endswith(".json")
            or "/" in preset_id or "\\" in preset_id):
        raise HTTPException(400, "プリセットの指定が不正です")
    folder = preset_dir().resolve()
    path = (folder / preset_id).resolve()
    if path.parent != folder:
        raise HTTPException(400, "プリセットフォルダの外は操作できません")
    if not path.is_file():
        raise HTTPException(404, "プリセットのファイルが見つかりません（Explorerで移動・削除された可能性があります）")
    return path


def _read(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict) or payload.get("format") != FORMAT or payload.get("version") != 1:
        raise ValueError("学習設定JSONの形式が不正です")
    if not isinstance(payload.get("name"), str) or not isinstance(payload.get("config"), dict):
        raise ValueError("名前または設定が不正です")
    return payload


def _entry(path: Path) -> dict:
    stat = path.stat()
    base = {"id": path.name, "file_path": str(path), "modified": stat.st_mtime, "size": stat.st_size}
    try:
        payload = _read(path)
    except (ValueError, OSError) as exc:  # one broken file must not hide the others
        return {**base, "name": path.stem, "payload": None, "error": str(exc)}
    return {**base, "name": payload["name"], "payload": payload, "error": None}


def _validate_payload(payload: dict) -> None:
    if payload.get("format") != FORMAT or payload.get("version") != 1:
        raise HTTPException(400, "LoRA Studioの学習設定JSONを指定してください")
    if payload.get("category") not in ("character", "style") or not isinstance(payload.get("config"), dict):
        raise HTTPException(400, "分類または学習設定が不正です")
    _validate_learning_rate(payload)


def _dump(payload: dict) -> str:
    data = json.dumps(payload, ensure_ascii=False, indent=2)
    if len(data.encode("utf8")) > MAX_BYTES:
        raise HTTPException(400, "プリセットは1MB以内で指定してください")
    return data


def _unique_name(base: str) -> str:
    if not _file_for_name(base).exists():
        return base
    for n in range(2, 1000):
        candidate = f"{base} ({n})"
        if not _file_for_name(candidate).exists():
            return candidate
    raise HTTPException(409, "空いている名前が見つかりません")


def _write_atomic(path: Path, data: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(data, encoding="utf8")
    os.replace(tmp, path)


def _launch_explorer(args: list[str]) -> None:  # separated so tests never open a window
    subprocess.Popen(args)


@router.get("")
def list_training_files() -> list[dict]:
    folder = preset_dir()
    if not folder.exists():
        return []
    return [_entry(p) for p in sorted(folder.glob("*.json"), key=lambda p: p.name.lower())]


@router.get("/folder")
def training_folder() -> dict:
    folder = preset_dir()
    return {"path": str(folder), "exists": folder.exists()}


@router.post("", status_code=201)
def save_training_file(body: SaveBody) -> dict:
    """Create a preset. With overwrite=true an existing preset of the same name is replaced
    (keeping any extra keys such as 'recommendation' the old file had)."""
    name = _clean_name(body.name)
    _validate_payload(body.payload)
    payload = {**body.payload, "name": name}
    target = _file_for_name(name)
    preset_dir().mkdir(parents=True, exist_ok=True)
    existed = target.exists()
    if existed and not body.overwrite:
        raise HTTPException(409, "同名のプリセットがあります。別の名前にするか、上書きを選んでください")
    if existed:
        try:
            old = _read(target)
            payload = {**{k: v for k, v in old.items() if k not in payload}, **payload}
        except (ValueError, OSError):
            pass
        _write_atomic(target, _dump(payload))
    else:
        data = _dump(payload)
        try:
            with target.open("x", encoding="utf8") as f:
                f.write(data)
        except FileExistsError as exc:
            raise HTTPException(409, "同名のプリセットがあります。別の名前にしてください") from exc
    return {**_entry(target), "overwritten": existed}


@router.post("/{preset_id}/rename")
def rename_training_file(preset_id: str, body: NameBody) -> dict:
    path = _resolve(preset_id)
    name = _clean_name(body.name or "")
    payload = _read(path)
    target = _file_for_name(name)
    if target.exists() and target.resolve() != path.resolve():
        raise HTTPException(409, "同名のプリセットがあります")
    payload["name"] = name
    data = _dump(payload)
    if target.resolve() != path.resolve():
        os.rename(path, target)
    _write_atomic(target, data)
    return _entry(target)


@router.post("/{preset_id}/duplicate", status_code=201)
def duplicate_training_file(preset_id: str, body: NameBody | None = None) -> dict:
    path = _resolve(preset_id)
    payload = _read(path)
    wanted = _clean_name(body.name) if body and body.name else f"{payload['name']} のコピー"
    name = _unique_name(_clean_name(wanted))
    payload["name"] = name
    target = _file_for_name(name)
    with target.open("x", encoding="utf8") as f:
        f.write(_dump(payload))
    return _entry(target)


@router.delete("/{preset_id}")
def delete_training_file(preset_id: str) -> dict:
    from .dataset_files import recycle  # shared Recycle-Bin helper (restorable delete)
    path = _resolve(preset_id)
    recycle([path])
    return {"id": preset_id, "deleted": True, "recycled": True}


@router.post("/reveal")
def reveal_training_file(body: RevealBody) -> dict:
    """Browser-mode fallback for 'open in Explorer' (the desktop shell does this itself)."""
    if os.name != "nt":
        raise HTTPException(400, "Windowsのみ対応しています")
    if body.id:
        path = _resolve(body.id)
        _launch_explorer(["explorer.exe", f"/select,{path}"])
    else:
        folder = preset_dir()
        folder.mkdir(parents=True, exist_ok=True)
        path = folder
        _launch_explorer(["explorer.exe", str(folder)])
    return {"opened": str(path)}
