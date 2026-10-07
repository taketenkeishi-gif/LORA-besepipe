"""Build a dataset from a video: one representative frame per scene, ranked by 'looks like the character'.

Candidates are written OUTSIDE the dataset folder (``<dataset>/../.video-candidates/<job>``) and only copied
into the dataset when the user accepts them, so a half-reviewed video never pollutes the dataset.
"""
from __future__ import annotations

import shutil
import threading
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ..services import video_scenes
from . import dataset_files

router = APIRouter(prefix="/dataset-files", tags=["dataset-video"])

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
_MAX_JOBS = 8


class VideoAnalyze(BaseModel):
    video_path: str = Field(min_length=1, max_length=2000)
    threshold: float = Field(default=0.3, ge=0.05, le=0.9)
    min_scene_seconds: float = Field(default=0.6, ge=0.1, le=10.0)
    samples: int = Field(default=3, ge=1, le=7)
    max_scenes: int = Field(default=300, ge=1, le=1000)
    rank_character: bool = True


class VideoAccept(BaseModel):
    indices: list[int] = Field(min_length=1, max_length=1000)
    folder: str = Field(default="", max_length=1024)


def _candidates_root(project_id: int) -> Path:
    return dataset_files.root_for(project_id).parent / ".video-candidates"


def _existing_captions(project_id: int, root: Path) -> list[str]:
    captions: list[str] = []
    for txt in sorted(root.rglob("*.txt"))[:2000]:
        try:
            captions.append(txt.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError):
            continue
    return captions


def _concept_tokens(project_id: int) -> set[str]:
    with dataset_files.database_connection() as conn:
        return {str(r["trigger_token"]).strip() for r in conn.execute(
            "SELECT trigger_token FROM basepipe_concepts WHERE project_id=? AND TRIM(trigger_token)<>''", (project_id,))}


def _prune_jobs() -> None:
    with _jobs_lock:
        while len(_jobs) > _MAX_JOBS:
            oldest = min(_jobs, key=lambda key: _jobs[key]["created"])
            job = _jobs.pop(oldest)
            shutil.rmtree(job["dir"], ignore_errors=True)


def _run(job_id: str, project_id: int, video: Path, payload: VideoAnalyze) -> None:
    job = _jobs[job_id]

    def progress(stage: str, done: int, total: int) -> None:
        job.update(stage=stage, done=done, total=total)

    tag_frame = None
    signature: list[str] = []
    try:
        if payload.rank_character:
            from ..services import tagger
            root = dataset_files.root_for(project_id)
            signature = video_scenes.character_signature(_existing_captions(project_id, root), _concept_tokens(project_id))
            if signature and tagger.is_available() and tagger.model_cached():
                tag_frame = tagger.predict
            job["ranking"] = {"signature": signature, "enabled": tag_frame is not None,
                              "reason": "" if tag_frame is not None else (
                                  "既存画像のキャプションが無いため、キャラ判定の基準を作れませんでした" if not signature
                                  else "タガーのモデルが見つからないため、キャラ判定は使えません")}
        items = video_scenes.analyze_video(
            video, Path(job["dir"]), threshold=payload.threshold, min_seconds=payload.min_scene_seconds,
            samples=payload.samples, max_scenes=payload.max_scenes, signature=signature, tag_frame=tag_frame,
            progress=progress, cancelled=lambda: job["cancel"])
        job["items"] = items
        job["status"] = "cancelled" if job["cancel"] else "done"
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI as the job error
        job["status"] = "error"
        job["error"] = str(exc)[:600]
    finally:
        job["finished"] = time.time()


@router.post("/{project_id}/video/analyze")
def analyze(project_id: int, payload: VideoAnalyze):
    dataset_files.root_for(project_id)
    video = Path(payload.video_path.strip().strip('"'))
    if video.suffix.lower() not in video_scenes.VIDEO_EXTENSIONS:
        raise HTTPException(400, "対応している動画は mp4 / mov / mkv / webm / avi / m4v です")
    if not video.is_file():
        raise HTTPException(404, "動画ファイルが見つかりません。パスを確認してください")
    job_id = uuid.uuid4().hex
    out_dir = _candidates_root(project_id) / job_id
    with _jobs_lock:
        _jobs[job_id] = {"project_id": project_id, "created": time.time(), "dir": str(out_dir), "status": "running",
                         "stage": "準備中", "done": 0, "total": 1, "items": [], "cancel": False, "video": str(video),
                         "ranking": {"signature": [], "enabled": False, "reason": ""}}
    _prune_jobs()
    threading.Thread(target=_run, args=(job_id, project_id, video, payload), daemon=True).start()
    return {"job_id": job_id}


def _job(project_id: int, job_id: str) -> dict:
    job = _jobs.get(job_id)
    if job is None or job["project_id"] != project_id:
        raise HTTPException(404, "解析ジョブが見つかりません（アプリを再起動すると消えます）。もう一度解析してください")
    return job


@router.get("/{project_id}/video/{job_id}")
def status(project_id: int, job_id: str):
    job = _job(project_id, job_id)
    items = sorted(job["items"], key=lambda i: (-(i["score"] if i["score"] is not None else -1), i["index"]))
    return {"status": job["status"], "stage": job["stage"], "done": job["done"], "total": job["total"],
            "error": job.get("error", ""), "ranking": job["ranking"], "video": job["video"], "items": items}


@router.post("/{project_id}/video/{job_id}/cancel")
def cancel(project_id: int, job_id: str):
    _job(project_id, job_id)["cancel"] = True
    return {"cancelled": True}


@router.get("/{project_id}/video/{job_id}/image/{index}")
def candidate_image(project_id: int, job_id: str, index: int):
    job = _job(project_id, job_id)
    item = next((i for i in job["items"] if i["index"] == index), None)
    if item is None:
        raise HTTPException(404, "候補が見つかりません")
    return FileResponse(Path(job["dir"]) / item["file"], media_type="image/png")


@router.post("/{project_id}/video/{job_id}/accept")
def accept(project_id: int, job_id: str, payload: VideoAccept):
    """Copy the chosen candidates into the dataset folder (with a TXT of the tags the tagger found, if any)."""
    job = _job(project_id, job_id)
    if job["status"] not in {"done", "cancelled"}:
        raise HTTPException(409, "解析が終わってから採用してください")
    with dataset_files._lock:
        root = dataset_files.root_for(project_id)
        target_dir = dataset_files.within(root, payload.folder) if payload.folder else root
        if not target_dir.is_dir():
            raise HTTPException(404, "追加先のフォルダが見つかりません")
        stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in Path(job["video"]).stem)[:60] or "video"
        wanted = set(payload.indices)
        added: list[str] = []
        skipped: list[str] = []
        for item in job["items"]:
            if item["index"] not in wanted:
                continue
            destination = target_dir / f"{stem}_s{item['index']:04d}.png"
            # Same video added twice: keep what is already there (it may have been edited) instead of overwriting.
            if destination.exists() or destination.with_suffix(".txt").exists():
                skipped.append(destination.name)
                continue
            shutil.copy2(Path(job["dir"]) / item["file"], destination)
            if item.get("tags"):
                destination.with_suffix(".txt").write_text(item["tags"], encoding="utf-8")
            added.append(destination.relative_to(root).as_posix())
    return {"added": added, "count": len(added), "skipped": skipped}
