from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from fastapi import APIRouter

from ..db import get_conn
from ..schemas import PreviewPromptsIn, PreviewPromptsOut, ToolPathsIn, ToolPathsOut

router = APIRouter(prefix="/settings", tags=["settings"])

SETTINGS_KEYS = ("python_exe", "kohya_root", "comfyui_root", "wd14_script", "temp_dir", "dataset_base_dir")
PROMPT_KEYS = ("positive_prompt", "negative_prompt")


def _read_paths() -> dict[str, str]:
    conn = get_conn()
    rows = conn.execute("SELECT key, value FROM app_settings").fetchall()
    conn.close()
    found = {str(r["key"]): str(r["value"]) for r in rows}
    return {k: found.get(k, "") for k in SETTINGS_KEYS}


def _read_prompts() -> dict[str, str]:
    conn = get_conn()
    rows = conn.execute("SELECT key, value FROM app_settings WHERE key IN (?, ?)", PROMPT_KEYS).fetchall()
    conn.close()
    found = {str(r["key"]): str(r["value"]) for r in rows}
    return {
        "positive_prompt": found.get("positive_prompt", "masterpiece, best quality, 1girl, portrait"),
        "negative_prompt": found.get("negative_prompt", "low quality, blurry, bad anatomy"),
    }


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


def _write_prompts(payload: dict[str, str]) -> None:
    conn = get_conn()
    cur = conn.cursor()
    for key in PROMPT_KEYS:
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
    project_root = Path(__file__).resolve().parents[3]
    default_dataset_base = Path(r"C:\ポートフォリオ\SDXL\LoRA_Traning\dataset")

    python_path = shutil.which("python") or ""
    if python_path:
        result["python_exe"] = str(Path(python_path))

    candidates = [
        project_root / "external_tools" / "kohya_ss",
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
        project_root / "external_tools" / "ComfyUI",
        Path.home() / "ComfyUI",
        Path("C:/ComfyUI"),
        Path("D:/ComfyUI"),
    ]
    for c in comfy_candidates:
        if c.exists() and c.is_dir():
            result["comfyui_root"] = str(c.resolve())
            break

    if result["kohya_root"]:
        wd14 = Path(result["kohya_root"]) / "sd-scripts" / "finetune" / "tag_images_by_wd14_tagger.py"
        if wd14.exists():
            result["wd14_script"] = str(wd14.resolve())
    wd14_local = project_root / "external_tools" / "WD14py" / "tag_images_by_wd14_tagger.py"
    if wd14_local.exists():
        result["wd14_script"] = str(wd14_local.resolve())

    result["temp_dir"] = str((project_root / ".runtime" / "tmp").resolve())
    if default_dataset_base.exists():
        result["dataset_base_dir"] = str(default_dataset_base.resolve())
    else:
        result["dataset_base_dir"] = str((project_root / "external_dataset").resolve())

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
    current = _read_paths()
    detected = _autodetect()
    # 初回は自動検出結果をそのまま初期値として保存
    if all(not v for v in current.values()):
        _write_paths(detected)
        return ToolPathsOut(**_read_paths())
    # 一部だけ空欄の場合も、検出できる値で自動補完する
    merged = current.copy()
    changed = False
    for key in SETTINGS_KEYS:
        if (not merged.get(key)) and detected.get(key):
            merged[key] = detected[key]
            changed = True
    if changed:
        _write_paths(merged)
        return ToolPathsOut(**_read_paths())
    return ToolPathsOut(**current)


@router.put("/tool-paths", response_model=ToolPathsOut)
def update_tool_paths(payload: ToolPathsIn) -> ToolPathsOut:
    _write_paths(payload.model_dump())
    paths = _read_paths()
    _ensure_runtime_dirs(paths)
    return ToolPathsOut(**paths)


def _ensure_runtime_dirs(paths: dict[str, str]) -> None:
    for key in ("temp_dir", "dataset_base_dir"):
        p = paths.get(key, "").strip()
        if not p:
            continue
        Path(p).mkdir(parents=True, exist_ok=True)
    # dataset_base_dir直下に型別フォルダを用意
    base = paths.get("dataset_base_dir", "").strip()
    if base:
        Path(base, "character").mkdir(parents=True, exist_ok=True)
        Path(base, "style").mkdir(parents=True, exist_ok=True)


@router.post("/tool-paths/autodetect", response_model=ToolPathsOut)
def autodetect_tool_paths() -> ToolPathsOut:
    detected = _autodetect()
    _write_paths(detected)
    paths = _read_paths()
    _ensure_runtime_dirs(paths)
    return ToolPathsOut(**paths)


@router.get("/integrations/status")
def integrations_status() -> dict:
    paths = _read_paths()
    checks = {
        "python_exe": _check_exe(paths["python_exe"]),
        "kohya_root": _check_dir(paths["kohya_root"]),
        "comfyui_root": _check_dir(paths["comfyui_root"]),
        "wd14_script": _check_file(paths["wd14_script"]),
        "temp_dir": _check_dir(paths["temp_dir"]),
        "dataset_base_dir": _check_dir(paths["dataset_base_dir"]),
    }
    return {"paths": paths, "checks": checks}


@router.get("/preview-prompts", response_model=PreviewPromptsOut)
def get_preview_prompts() -> PreviewPromptsOut:
    prompts = _read_prompts()
    return PreviewPromptsOut(**prompts)


@router.put("/preview-prompts", response_model=PreviewPromptsOut)
def update_preview_prompts(payload: PreviewPromptsIn) -> PreviewPromptsOut:
    _write_prompts(payload.model_dump())
    return PreviewPromptsOut(**_read_prompts())
