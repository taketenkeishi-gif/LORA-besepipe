"""Character merge editor across videos ("動画のキャラ").

One set = one folder of videos. The per-video crops (.video-datasets/<job>/...) are never moved or deleted: the editor only
records which image belongs to which character, in  <Dataset>/.character-sets/<set>/characters.json  (+ history/ for undo,
cache/ for computed features and thumbnails). Files are copied only when a character is made into a LoRA dataset.
Initial characters = image-level CCIP clustering (tools/video_dataset/crop_link.py).
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from ..db import get_conn
from ..services import video_dataset_job as svc
from . import dataset_files

router = APIRouter(prefix="/character-sets", tags=["character-sets"])
SETS = ".character-sets"
TOOL = Path(__file__).resolve().parents[3] / "tools" / "video_dataset" / "crop_link.py"
_lock = threading.Lock()
_computing: dict[str, dict] = {}


def _roots() -> list[Path]:
    """Dataset parents that hold video jobs (<parent>/.video-datasets)."""
    conn = get_conn()
    try:
        ids = [int(r["id"]) for r in conn.execute("SELECT id FROM projects")]
    finally:
        conn.close()
    out: list[Path] = []
    for pid in ids:
        try:
            jr = svc.jobs_root_for(dataset_files.root_for(pid))
        except Exception:
            continue
        if jr.is_dir() and jr.parent not in out:
            out.append(jr.parent)
    return out


def _done_jobs(root: Path) -> list[tuple[str, Path]]:
    jr = root / svc.JOBS_DIRNAME
    out = []
    for d in sorted(jr.iterdir()) if jr.is_dir() else []:
        meta = svc.load_json(d / "job.json") if (d / "job.json").is_file() else None
        if isinstance(meta, dict) and meta.get("status") == "done" and (d / "report.json").is_file():
            video = str(meta.get("video") or meta.get("video_path") or (meta.get("params") or {}).get("video_path") or "")
            out.append((d.name, Path(video)))
    return out


def _slug(folder: Path) -> str:
    name = re.sub(r'[\\/:*?"<>|]+', "_", folder.name).strip() or "videos"
    return f"{name}-{hashlib.sha1(str(folder).lower().encode()).hexdigest()[:6]}"


def _set_dir(set_id: str) -> tuple[Path, Path]:
    if not re.fullmatch(r"[^\\/:*?\"<>|]+-[0-9a-f]{6}", set_id):
        raise HTTPException(400, "不正なセットです")
    for root in _roots():
        d = root / SETS / set_id
        if (d / "set.json").is_file():
            return root, d
    raise HTTPException(404, "セットが見つかりません")


def _load(d: Path) -> dict:
    return json.loads((d / "characters.json").read_text(encoding="utf-8"))


def _save(d: Path, state: dict, history: bool = True) -> None:
    if history and (d / "characters.json").is_file():
        h = d / "history"
        h.mkdir(exist_ok=True)
        (d / "characters.json").replace(h / f"{time.time():.3f}.json")
        old = sorted(h.glob("*.json"))
        for f in old[:-100]:
            f.unlink(missing_ok=True)
    svc.write_json(d / "characters.json", state)


def _from_clusters(clusters: list[dict], prefix: str) -> dict:
    chars = []
    for i, c in enumerate(sorted(clusters, key=lambda c: -c["images"]), 1):
        name = " ".join(x for x in (c.get("hair", ""), c.get("eyes", "")) if x) or f"キャラ{i}"
        chars.append({"id": i, "name": name, "images": [prefix + f for f in c["files"]]})
    return {"version": 1, "characters": chars, "excluded": [], "next_id": len(chars) + 1}


@router.get("/sources")
def sources():
    """Video folders whose videos were processed, and whether a set exists for each."""
    out = []
    for root in _roots():
        groups: dict[Path, list[str]] = {}
        for job, video in _done_jobs(root):
            groups.setdefault(video.parent, []).append(job)
        for folder, jobs in groups.items():
            sid = _slug(folder)
            d = root / SETS / sid
            n = len(_load(d)["characters"]) if (d / "characters.json").is_file() else None
            out.append({"folder": str(folder), "name": folder.name, "videos": len(jobs), "set_id": sid, "characters": n,
                        "computing": sid in _computing})
    return {"sources": sorted(out, key=lambda s: -s["videos"])}


class OpenIn(BaseModel):
    folder: str


def _compute(root: Path, d: Path, sid: str, jobs: list[str]) -> None:
    try:
        cache = d / "cache"
        cache.mkdir(exist_ok=True)
        (cache / "jobs.json").write_text(json.dumps(jobs), encoding="utf-8")
        python, _comfy = svc.find_comfy()
        env = os.environ.copy()
        env.update({"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"})
        r = subprocess.run([str(python), "-X", "utf8", str(TOOL), str(cache / "links.json"), str(root / svc.JOBS_DIRNAME), str(cache / "jobs.json")],
                           env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=7200)
        if r.returncode:
            raise RuntimeError((r.stderr or r.stdout)[-400:])
        links = json.loads((cache / "links.json").read_text(encoding="utf-8"))
        _save(d, _from_clusters(links["clusters"], svc.JOBS_DIRNAME + "/"), history=False)
        _computing.pop(sid, None)
    except Exception as exc:  # noqa: BLE001
        _computing[sid] = {"error": str(exc)[-300:]}


@router.post("/open")
def open_set(payload: OpenIn):
    folder = Path(payload.folder)
    for root in _roots():
        jobs = [j for j, v in _done_jobs(root) if v.parent == folder]
        if not jobs:
            continue
        sid = _slug(folder)
        d = root / SETS / sid
        with _lock:
            d.mkdir(parents=True, exist_ok=True)
            svc.write_json(d / "set.json", {"folder": str(folder), "jobs": jobs, "created": time.time()})
            if (d / "characters.json").is_file():
                return {"set_id": sid, "status": "ready"}
            # an existing image-level clustering of exactly these videos is reused (it takes minutes to compute)
            prev = root / svc.JOBS_DIRNAME / "_links_crop.json"
            if prev.is_file():
                links = json.loads(prev.read_text(encoding="utf-8"))
                used = {f.split("/")[0] for c in links["clusters"] for f in c["files"]}
                if used == set(jobs):
                    _save(d, _from_clusters(links["clusters"], svc.JOBS_DIRNAME + "/"), history=False)
                    return {"set_id": sid, "status": "ready"}
            if sid not in _computing:
                _computing[sid] = {"started": time.time()}
                threading.Thread(target=_compute, args=(root, d, sid, jobs), daemon=True, name=f"charset-{sid}").start()
        return {"set_id": sid, "status": "computing"}
    raise HTTPException(404, "このフォルダの動画はまだ処理されていません")


@router.get("/{set_id}")
def get_set(set_id: str):
    if set_id in _computing:
        st = _computing[set_id]
        return {"set_id": set_id, "status": "error" if "error" in st else "computing", "error": st.get("error", "")}
    root, d = _set_dir(set_id)
    meta = json.loads((d / "set.json").read_text(encoding="utf-8"))
    if not (d / "characters.json").is_file():
        return {"set_id": set_id, "status": "computing"}
    s = _load(d)
    return {"set_id": set_id, "status": "ready", "folder": meta["folder"], "set_folder": str(d), "videos": len(meta["jobs"]),
            "characters": [{"id": c["id"], "name": c["name"], "count": len(c["images"]), "cover": c["images"][:1]} for c in s["characters"] if c["images"]],
            "excluded": len(s["excluded"]), "can_undo": any((d / "history").glob("*.json")) if (d / "history").is_dir() else False}


@router.get("/{set_id}/images")
def images(set_id: str, character: str = Query(...)):
    _root, d = _set_dir(set_id)
    s = _load(d)
    if character == "excluded":
        return {"images": s["excluded"]}
    c = next((c for c in s["characters"] if str(c["id"]) == character), None)
    if c is None:
        raise HTTPException(404, "キャラが見つかりません")
    return {"images": c["images"]}


class MoveIn(BaseModel):
    images: list[str]
    to: str  # character id, "excluded" or "new"


@router.post("/{set_id}/move")
def move(set_id: str, payload: MoveIn):
    _root, d = _set_dir(set_id)
    with _lock:
        s = _load(d)
        moving = set(payload.images)
        for c in s["characters"]:
            c["images"] = [i for i in c["images"] if i not in moving]
        s["excluded"] = [i for i in s["excluded"] if i not in moving]
        if payload.to == "excluded":
            s["excluded"] += payload.images
            target = "excluded"
        elif payload.to == "new":
            target = s["next_id"]
            s["characters"].append({"id": target, "name": f"新しいキャラ{target}", "images": list(payload.images)})
            s["next_id"] += 1
        else:
            c = next((c for c in s["characters"] if str(c["id"]) == payload.to), None)
            if c is None:
                raise HTTPException(404, "移動先のキャラが見つかりません")
            c["images"] += payload.images
            target = c["id"]
        _save(d, s)
    return {"moved": len(payload.images), "to": target}


class MergeIn(BaseModel):
    source: int
    target: int


@router.post("/{set_id}/merge")
def merge(set_id: str, payload: MergeIn):
    _root, d = _set_dir(set_id)
    with _lock:
        s = _load(d)
        src = next((c for c in s["characters"] if c["id"] == payload.source), None)
        dst = next((c for c in s["characters"] if c["id"] == payload.target), None)
        if src is None or dst is None or src is dst:
            raise HTTPException(400, "結合するキャラが不正です")
        dst["images"] += src["images"]
        s["characters"] = [c for c in s["characters"] if c is not src]
        _save(d, s)
    return {"merged": len(src["images"]), "into": dst["id"]}


class RenameIn(BaseModel):
    id: int
    name: str


@router.post("/{set_id}/rename")
def rename(set_id: str, payload: RenameIn):
    _root, d = _set_dir(set_id)
    with _lock:
        s = _load(d)
        c = next((c for c in s["characters"] if c["id"] == payload.id), None)
        if c is None or not payload.name.strip():
            raise HTTPException(400, "名前を変更できません")
        c["name"] = payload.name.strip()[:80]
        _save(d, s)
    return {"ok": True}


@router.post("/{set_id}/undo")
def undo(set_id: str):
    _root, d = _set_dir(set_id)
    with _lock:
        h = sorted((d / "history").glob("*.json")) if (d / "history").is_dir() else []
        if not h:
            raise HTTPException(409, "元に戻せる操作がありません")
        os.replace(h[-1], d / "characters.json")
    return {"ok": True}


class RevealIn(BaseModel):
    what: str  # "videos" | "set"


@router.post("/{set_id}/reveal")
def reveal(set_id: str, payload: RevealIn):
    """Open the video folder or this set's folder in Explorer."""
    _root, d = _set_dir(set_id)
    meta = json.loads((d / "set.json").read_text(encoding="utf-8"))
    target = Path(meta["folder"]) if payload.what == "videos" else d
    if not target.is_dir():
        raise HTTPException(404, f"フォルダが見つかりません: {target}")
    os.startfile(str(target))  # noqa: S606 - local desktop app, path comes from our own set.json
    return {"path": str(target)}


def _image_path(root: Path, rel: str) -> Path:
    p = (root / rel).resolve()
    if not p.is_relative_to((root / svc.JOBS_DIRNAME).resolve()) or p.suffix.lower() not in (".png", ".jpg", ".jpeg") or not p.is_file():
        raise HTTPException(404, "画像が見つかりません")
    return p


@router.get("/{set_id}/file")
def file(set_id: str, rel: str = Query(..., min_length=1, max_length=1024)):
    root, _d = _set_dir(set_id)
    return FileResponse(_image_path(root, rel), headers={"Cache-Control": "private, max-age=3600"})


@router.get("/{set_id}/thumb")
def thumb(set_id: str, rel: str = Query(..., min_length=1, max_length=1024), s: int = 192):
    """Small JPEG for the grid (cached under the set), so hundreds of 1024 px PNGs do not have to be decoded by the browser."""
    root, d = _set_dir(set_id)
    p = _image_path(root, rel)
    size = max(64, min(512, s))
    key = hashlib.sha1(f"{rel}|{size}|{p.stat().st_mtime_ns}".encode()).hexdigest()
    c = d / "cache" / "thumbs" / f"{key}.jpg"
    if not c.is_file():
        from PIL import Image

        c.parent.mkdir(parents=True, exist_ok=True)
        im = Image.open(p).convert("RGB")
        im.thumbnail((size, size), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=82)
        tmp = c.with_suffix(".tmp")
        tmp.write_bytes(buf.getvalue())
        os.replace(tmp, c)
    return FileResponse(c, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"})
