from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from fastapi import APIRouter

from ..db import get_conn
from ..schemas import ToolPathsIn, ToolPathsOut

router = APIRouter(prefix="/settings", tags=["settings"])

SETTINGS_KEYS = ("python_exe", "kohya_root", "comfyui_root", "wd14_script")


def _read_paths() -> dict[str, str]:
    conn = get_conn()
    rows = conn.execute("SELECT key, value FROM app_settings").fetchall()
    conn.close()
    found = {str(r["key"]): str(r["value"]) for r in rows}
    return {k: found.get(k, "") for k in SETTINGS_KEYS}


def _write_paths(payload: dict[str, str]) -> None:
    conn = get_conn()
    cur = conn.cursor()
    for key in SETTINGS_KEYS:
        val = str(payload.get(key, "")).strip()
        cur.execute(
            """
            INSERT INTO app_settings(key, value, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                updated_at = CURRENT_TIMESTAMP
            """,
            (key, val),
        )
    conn.commit()
    conn.close()


def _autodetect() -> dict[str, str]:
    result = {k: "" for k in SETTINGS_KEYS}
    python_path = shutil.which("python") or ""
    if python_path:
        result["python_exe"] = str(Path(python_path))

    candidates = [
        Path.home() / "kohya_ss",
        Path("C:/kohya_ss"),
        Path("C:/tools/kohya_ss"),
        Path("D:/tools/kohya_ss"),
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            result["kohya_root"] = str(c.resolve())
            break

    comfy_candidates = [
        Path.home() / "ComfyUI",
        Path("C:/ComfyUI"),
        Path("D:/ComfyUI"),
    ]
    for c in comfy_candidates:
        if c.exists() and c.is_dir():
            result["comfyui_root"] = str(c.resolve())
            break

    if result["kohya_root"]:
        wd14 = Path(result["kohya_root"]) / "finetune" / "tag_images_by_wd14_tagger.py"
        if wd14.exists():
            result["wd14_script"] = str(wd14.resolve())

    return result


def _check_exe(path: str) -> dict:
    if not path:
        return {"ok": False, "reason": "not configured"}
    p = Path(path)
    if not p.exists():
        return {"ok": False, "reason": "path not found"}
    try:
        proc = subprocess.run([str(p), "--version"], capture_output=True, text=True, timeout=4)
        out = (proc.stdout or proc.stderr or "").strip().splitlines()
        return {"ok": proc.returncode == 0, "reason": out[0] if out else "version check done"}
    except Exception as exc:
        return {"ok": False, "reason": str(exc)}


def _check_dir(path: str) -> dict:
    if not path:
        return {"ok": False, "reason": "not configured"}
    p = Path(path)
    if not p.exists():
        return {"ok": False, "reason": "path not found"}
    if not p.is_dir():
        return {"ok": False, "reason": "not a directory"}
    return {"ok": True, "reason": "directory found"}


def _check_file(path: str) -> dict:
    if not path:
        return {"ok": False, "reason": "not configured"}
    p = Path(path)
    if not p.exists():
        return {"ok": False, "reason": "path not found"}
    if not p.is_file():
        return {"ok": False, "reason": "not a file"}
    return {"ok": True, "reason": "file found"}


@router.get("/tool-paths", response_model=ToolPathsOut)
def get_tool_paths() -> ToolPathsOut:
    return ToolPathsOut(**_read_paths())


@router.put("/tool-paths", response_model=ToolPathsOut)
def update_tool_paths(payload: ToolPathsIn) -> ToolPathsOut:
    _write_paths(payload.model_dump())
    return ToolPathsOut(**_read_paths())


@router.post("/tool-paths/autodetect", response_model=ToolPathsOut)
def autodetect_tool_paths() -> ToolPathsOut:
    detected = _autodetect()
    _write_paths(detected)
    return ToolPathsOut(**_read_paths())


@router.get("/integrations/status")
def integrations_status() -> dict:
    paths = _read_paths()
    checks = {
        "python_exe": _check_exe(paths["python_exe"]),
        "kohya_root": _check_dir(paths["kohya_root"]),
        "comfyui_root": _check_dir(paths["comfyui_root"]),
        "wd14_script": _check_file(paths["wd14_script"]),
    }
    return {"paths": paths, "checks": checks}
