"""Video -> per-character dataset (job API). The pipeline itself is tools/video_dataset/video_to_dataset.py.

Results live OUTSIDE the dataset (``<dataset>/../.video-datasets/<job>``); nothing reaches a training dataset until the
user imports a character, and importing only ever copies.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..schemas import ProjectCreate
from ..services import video_dataset_job as svc
from . import dataset_files, projects

router = APIRouter(prefix="/dataset-files/{project_id}/video-dataset", tags=["dataset-video-characters"])

_JOB_ID = re.compile(r"^[0-9a-f]{8,32}$")
_FILE_EXT = {".png", ".jpg", ".jpeg"}


class StartPayload(BaseModel):
    video_path: str
    scene_sensitivity: str = "normal"
    frame_interval: float = 0.5
    max_frames_per_scene: int = 24
    others: str = "keep"
    overlap_threshold: float = 0.25
    margin: float = 0.10
    long_side: int = 1024
    min_area: float = 0.01
    min_short: int = 100
    min_char_crops: int = 3
    merge_quantile: float = 0.80
    group_merge_quantile: float = 0.55
    rescue_quantile: float = 0.55
    yolo_conf: float = 0.15
    cascade_margin: float = 0.8
    early_dup_bits: int = 1
    use_names: bool = False
    gpu: int | None = None
    limit_seconds: float = 0


class RenamePayload(BaseModel):
    folder: str
    new_name: str


class MergePayload(BaseModel):
    target: str
    sources: list[str]


class AssignPayload(BaseModel):
    files: list[str]
    target: str | None = None
    new_name: str | None = None


class ImportPayload(BaseModel):
    characters: list[str]
    mode: str


def _wrap(call):
    try:
        return call()
    except svc.VideoDatasetError as exc:
        raise HTTPException(exc.status, exc.message) from exc


def _job_dir(project_id: int, job_id: str) -> Path:
    if not _JOB_ID.match(job_id):
        raise HTTPException(404, "解析ジョブが見つかりません")
    root = dataset_files.root_for(project_id)
    directory = svc.jobs_root_for(root) / job_id
    meta = svc.load_json(directory / "job.json") if directory.is_dir() else None
    if not isinstance(meta, dict) or meta.get("project_id") != project_id:
        raise HTTPException(404, "解析ジョブが見つかりません")
    return directory


def _done_job_dir(project_id: int, job_id: str) -> Path:
    directory = _job_dir(project_id, job_id)
    meta = svc.refresh_meta(directory) or {}
    if meta.get("status") != "done":
        raise HTTPException(409, "解析が完了してから操作してください")
    return directory


# NOTE: /jobs must be declared before /{job_id}
@router.get("/jobs")
def jobs(project_id: int):
    return svc.list_jobs(dataset_files.root_for(project_id), project_id)


@router.post("/start")
def start(project_id: int, payload: StartPayload):
    root = dataset_files.root_for(project_id)
    params = _wrap(lambda: svc.validate_params(payload))
    return {"job_id": _wrap(lambda: svc.start_job(project_id, root, params))}


@router.get("/{job_id}")
def status(project_id: int, job_id: str):
    return _wrap(lambda: svc.status_payload(_job_dir(project_id, job_id)))


@router.get("/{job_id}/file")
def file(project_id: int, job_id: str, rel: str = Query(..., min_length=1, max_length=1024)):
    directory = _job_dir(project_id, job_id)
    normalized = rel.replace("\\", "/")
    parts = normalized.split("/")
    if (normalized.startswith("/") or re.match(r"^[A-Za-z]:", normalized) or any(p in ("", ".", "..") for p in parts)
            or "\x00" in normalized or Path(normalized).suffix.lower() not in _FILE_EXT):
        raise HTTPException(400, "不正なファイル指定です")
    path = (directory / normalized).resolve()
    if not path.is_relative_to(directory.resolve()) or not path.is_file():
        raise HTTPException(404, "画像が見つかりません")
    media = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return FileResponse(path, media_type=media, headers={"Cache-Control": "private, max-age=300"})


@router.post("/{job_id}/cancel")
def cancel(project_id: int, job_id: str):
    directory = _job_dir(project_id, job_id)
    if not svc.cancel_job(job_id) and not svc.cancel_adopted(directory):
        meta = svc.refresh_meta(directory) or {}
        if meta.get("status") == "running":  # pragma: no cover - refresh_meta turns orphans into errors
            raise HTTPException(409, "中止できませんでした")
        raise HTTPException(409, "この解析は実行中ではありません")
    return {"cancelled": True}


@router.post("/{job_id}/rename")
def rename(project_id: int, job_id: str, payload: RenamePayload):
    directory = _done_job_dir(project_id, job_id)
    return _wrap(lambda: svc.rename_character(directory, payload.folder, payload.new_name))


@router.post("/{job_id}/merge")
def merge(project_id: int, job_id: str, payload: MergePayload):
    directory = _done_job_dir(project_id, job_id)
    return _wrap(lambda: svc.merge_characters(directory, payload.target, payload.sources))


@router.get("/{job_id}/suggest")
def suggest(project_id: int, job_id: str, target: str = Query(..., min_length=1, max_length=200)):
    directory = _done_job_dir(project_id, job_id)
    return {"ranked": _wrap(lambda: svc.suggest_for_character(directory, target))}


@router.post("/{job_id}/assign")
def assign(project_id: int, job_id: str, payload: AssignPayload):
    directory = _done_job_dir(project_id, job_id)
    return _wrap(lambda: svc.assign_unassigned(directory, payload.files, payload.target, payload.new_name))


def explorer_args(path: Path) -> list[str]:
    return ["explorer.exe", str(path)]


@router.post("/{job_id}/reveal")
def reveal(project_id: int, job_id: str):
    directory = _job_dir(project_id, job_id)
    try:
        subprocess.Popen(explorer_args(directory))  # no shell; explorer returns exit code 1 even on success
    except OSError as exc:
        raise HTTPException(500, f"エクスプローラーを開けませんでした: {exc}") from exc
    return {"path": str(directory)}


# ------------------------------------------------------------------ import
def _tokens_in_use(conn, project_id: int) -> set[str]:
    return {str(r["trigger_token"]).strip().casefold() for r in conn.execute(
        "SELECT trigger_token FROM basepipe_concepts WHERE project_id=? AND TRIM(trigger_token)<>''", (project_id,))}


def _concept_names(conn, project_id: int) -> set[str]:
    return {str(r["name"]).casefold() for r in conn.execute("SELECT name FROM basepipe_concepts WHERE project_id=?", (project_id,))}


def _free_char_trigger(wanted: str, used: set[str]) -> str:
    if wanted.casefold() not in used:
        return wanted
    n = 1
    while True:
        candidate = f"ch{n:02d}"
        if candidate.casefold() not in used:
            return candidate
        n += 1


def _unique_dir(parent: Path, name: str) -> Path:
    candidate, n = parent / name, 1
    while candidate.exists():
        n += 1
        candidate = parent / f"{name}_{n}"
    return candidate


def _unique_project_name(conn, name: str) -> str:
    taken = {str(r["name"]).casefold() for r in conn.execute("SELECT name FROM projects")}
    candidate, n = name, 1
    while candidate.casefold() in taken or (projects.PROJECTS_ROOT / candidate).exists():
        n += 1
        candidate = f"{name}_{n}"
    return candidate


def _plan_triggers(ch: dict, used: set[str]) -> tuple[str, dict[str, tuple[str, str]]]:
    """(new character trigger, {outfit folder: (old trigger, new trigger)}) with no clash against ``used`` (updated in place)."""
    new_char = _free_char_trigger(ch["trigger"], used)
    used.add(new_char.casefold())
    mapping: dict[str, tuple[str, str]] = {}
    for o in ch.get("outfits", []):
        suffix = o["trigger"].rsplit("_o", 1)[-1] if "_o" in o["trigger"] else "00"
        candidate, n = f"{new_char}_o{suffix}", 0
        while candidate.casefold() in used:
            n += 1
            candidate = f"{new_char}_o{(int(suffix) + n) % 100:02d}" if suffix.isdigit() else f"{new_char}_o{n:02d}"
        used.add(candidate.casefold())
        mapping[o["folder"]] = (o["trigger"], candidate)
    return new_char, mapping


def _copy_character(src: Path, dest_parent: Path, ch: dict, new_char: str, mapping: dict[str, tuple[str, str]], flatten: bool, folder_name: str) -> tuple[Path, int]:
    """Copy outfit folders (PNG/JPG + TXT) and rewrite the trigger tokens. Returns (destination, image count)."""
    dest = dest_parent if flatten else _unique_dir(dest_parent, folder_name)
    created: list[Path] = []
    count = 0
    try:
        for od in sorted(d for d in src.iterdir() if d.is_dir()):
            old_outfit, new_outfit = mapping.get(od.name, (None, ""))
            target_dir = _unique_dir(dest, od.name)
            target_dir.mkdir(parents=True, exist_ok=False)
            created.append(target_dir)
            for f in sorted(p for p in od.iterdir() if p.is_file()):
                if f.suffix.lower() in _FILE_EXT:
                    shutil.copy2(f, target_dir / f.name)
                    count += f.suffix.lower() in _FILE_EXT
                elif f.suffix.lower() == ".txt":
                    text = f.read_text(encoding="utf-8-sig")
                    (target_dir / f.name).write_text(svc.rewrite_caption(text, ch["trigger"], new_char, old_outfit, new_outfit), encoding="utf-8", newline="")
    except Exception:
        for d in created:
            shutil.rmtree(d, ignore_errors=True)
        if not flatten:
            shutil.rmtree(dest, ignore_errors=True)
        raise
    if not flatten:
        dest.mkdir(parents=True, exist_ok=True)
    return dest, count


def _register_concepts(conn, project_id: int, ch: dict, new_char: str, mapping: dict[str, tuple[str, str]], folder_name: str) -> list[dict]:
    names = _concept_names(conn, project_id)

    def unique(name: str, trigger: str) -> str:
        if name.casefold() not in names:
            return name
        candidate = f"{name} ({trigger})"
        n = 1
        while candidate.casefold() in names:
            n += 1
            candidate = f"{name} ({trigger}) {n}"
        return candidate

    registered = []
    entries = [(folder_name, "character", new_char)] + [(folder, "outfit", new) for folder, (_, new) in mapping.items()]
    for name, kind, trigger in entries:
        final = unique(name, trigger)
        names.add(final.casefold())
        conn.execute("INSERT INTO basepipe_concepts(project_id,name,concept_type,trigger_token,description) VALUES(?,?,?,?,?)",
                     (project_id, final, kind, trigger, "動画から自動作成"))
        registered.append({"name": final, "concept_type": kind, "trigger_token": trigger})
    return registered


@router.post("/{job_id}/import")
def import_characters(project_id: int, job_id: str, payload: ImportPayload):
    directory = _done_job_dir(project_id, job_id)
    if payload.mode not in ("new_projects", "current_project"):
        raise HTTPException(422, "取り込み方法は new_projects（キャラごとに新規プロジェクト）か current_project（このプロジェクトへ追加）です")
    if not payload.characters:
        raise HTTPException(422, "取り込むキャラクターを選んでください")
    if len(set(payload.characters)) != len(payload.characters):
        raise HTTPException(422, "キャラクターが重複しています")
    imported: list[dict] = []
    skipped: list[dict] = []
    with svc._ops_lock, dataset_files._lock:
        report = _wrap(lambda: svc.load_report(directory))
        by_folder = {c["folder"]: c for c in report["characters"]}
        root = dataset_files.root_for(project_id)
        for folder in payload.characters:
            ch = by_folder.get(folder)
            if ch is None:
                skipped.append({"folder": folder, "reason": "このジョブの結果にそのキャラクターがありません"})
                continue
            src = directory / folder
            if not src.is_dir() or not any(d.is_dir() for d in src.iterdir()):
                skipped.append({"folder": folder, "reason": "フォルダが空か、見つかりません"})
                continue
            try:
                imported.append(_import_one(project_id, root, src, ch, payload.mode))
            except HTTPException as exc:
                skipped.append({"folder": folder, "reason": str(exc.detail)})
            except Exception as exc:  # noqa: BLE001 - one failing character must not hide the others
                skipped.append({"folder": folder, "reason": f"取り込みに失敗しました: {exc}"})
    return {"imported": imported, "skipped": skipped}


def _import_one(project_id: int, root: Path, src: Path, ch: dict, mode: str) -> dict:
    folder = ch["folder"]
    if mode == "current_project":
        with dataset_files.database_connection() as conn:
            used = _tokens_in_use(conn, project_id)
        new_char, mapping = _plan_triggers(ch, used)
        dest, count = _copy_character(src, root, ch, new_char, mapping, flatten=False, folder_name=folder)
        try:
            with dataset_files.database_connection() as conn:
                concepts = _register_concepts(conn, project_id, ch, new_char, mapping, dest.name)
        except Exception:
            shutil.rmtree(dest, ignore_errors=True)
            raise
        return {"folder": folder, "project_id": project_id, "images": count, "concepts": concepts}

    with dataset_files.database_connection() as conn:
        name = _unique_project_name(conn, safe_project_name(folder))
    created = projects.create_project(ProjectCreate(name=name, project_type="character"))
    try:
        new_char, mapping = _plan_triggers(ch, set())
        dest, count = _copy_character(src, Path(created.dataset_dir), ch, new_char, mapping, flatten=True, folder_name=folder)
        with dataset_files.database_connection() as conn:
            concepts = _register_concepts(conn, created.id, ch, new_char, mapping, name)
    except Exception:
        with dataset_files.database_connection() as conn:
            conn.execute("DELETE FROM projects WHERE id=?", (created.id,))
        shutil.rmtree(Path(created.base_dir), ignore_errors=True)  # brand new (name was checked to be unused)
        try:
            Path(created.library_dir).rmdir()  # only if it is still the empty folder create_project made
        except OSError:
            pass
        raise
    return {"folder": folder, "project_id": created.id, "images": count, "concepts": concepts}


def safe_project_name(folder: str) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "-", folder).strip(" .") or "character"
    return name[:100]
