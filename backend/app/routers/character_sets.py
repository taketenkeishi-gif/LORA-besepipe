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
FEAT_TOOL = TOOL.with_name("crop_features.py")
CLUSTER_TOOL = TOOL.with_name("crop_cluster.py")
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
    hit = _set_dirs.get(set_id)  # every thumbnail asks for this: scanning all projects each time cost ~0.2 s
    if hit and (hit[1] / "set.json").is_file():
        return hit
    for root in _roots():
        d = root / SETS / set_id
        if (d / "set.json").is_file():
            _set_dirs[set_id] = (root, d)
            return root, d
    raise HTTPException(404, "セットが見つかりません")


_set_dirs: dict[str, tuple[Path, Path]] = {}


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


NAMING = 4
HAIR_AGREE = 0.6   # a hair colour names the group only when this share of its images carries that colour tag
FACE_SHARE = 0.35  # ...and this share shows a face (crop view close/upper/full); below it the group is parts, props, mascots
_tags: dict[str, list[str]] = {}


def _crop_tags(root: Path, rel: str) -> list[str]:
    if rel not in _tags:
        try:
            _tags[rel] = [t.strip() for t in (root / rel).with_suffix(".txt").read_text(encoding="utf-8").split(",")]
        except OSError:
            _tags[rel] = []
    return _tags[rel]


def _shows_face(rel: str) -> bool:
    view = Path(rel).stem.rsplit("_", 2)[-2] if Path(rel).stem.count("_") >= 2 else ""
    return view.split("-")[-1] in ("close", "upper", "full")


def _describe(root: Path, images: list[str]) -> tuple[int, str]:
    """(rank, label): rank 0 = a character named by hair/eye colour, 1 = faces but mixed hair, 2 = mostly no face."""
    from ..services import ja_names
    if not images:
        return 2, "空"

    def dominant(kind: str) -> tuple[str, float]:
        """Most common colour tag and its share among the images that carry any colour tag of this kind (untagged
        images - backs, hands, far shots - say nothing either way)."""
        counts: dict[str, int] = {}
        tagged = 0
        for rel in images:
            found = {t for t in _crop_tags(root, rel) if t.endswith(f" {kind}") and t[: -len(kind) - 1] in ja_names.COLOURS}
            tagged += bool(found)
            for t in found:
                counts[t] = counts.get(t, 0) + 1
        if not counts:
            return "", 0.0
        tag = max(counts, key=counts.get)
        return tag, counts[tag] / tagged

    what = _what(root, images)
    hair, agree = dominant("hair")
    if what == "マスコット":
        colour = ja_names.COLOURS.get(hair[:-5], "") if agree >= HAIR_AGREE else ""
        return 0, f"{colour}色系マスコット" if colour else "マスコット"
    if agree < HAIR_AGREE:
        return (2, "顔なし") if sum(map(_shows_face, images)) / len(images) < FACE_SHARE else (1, "髪色混在")
    eyes, eye_agree = dominant("eyes")
    name = ja_names.character(hair=hair, eyes=eyes if eye_agree >= 0.5 else "")
    return 0, "・".join(w for w in (name, what) if w)


PENDING_NAME = "保留（未確定）"
NONHUMAN = {"no humans", "creature", "pokemon (creature)", "furry", "animal", "mascot", "robot", "stuffed toy"}


def _what(root: Path, images: list[str]) -> str:
    """'マスコット' / '男子' / '' (girl or unknown) from the majority of the images' tags."""
    kinds = {"creature": 0, "boy": 0, "girl": 0}
    for rel in images:
        tags = set(_crop_tags(root, rel))
        if tags & NONHUMAN and "1girl" not in tags and "1boy" not in tags:
            kinds["creature"] += 1
        elif ("1boy" in tags or "male focus" in tags) and "1girl" not in tags:
            kinds["boy"] += 1
        elif "1girl" in tags:
            kinds["girl"] += 1
    top = max(kinds, key=kinds.get)
    said = sum(kinds.values())
    what = {"creature": "マスコット", "boy": "男子", "girl": ""}[top] if said and kinds[top] * 2 > said else ""
    lengths = [t for rel in images for t in _crop_tags(root, rel) if t in ("short hair", "medium hair", "long hair", "very long hair")]
    if what != "マスコット" and lengths and lengths.count("short hair") * 2 > len(lengths):
        what = "・".join(w for w in (what, "短髪") if w)
    return what


def _relabel(root: Path, s: dict) -> dict:
    """Name automatically named groups from their current images; real characters first (largest first), the pending pile last.
    Characters that would get the same name are told apart as A, B, C … (largest first)."""
    ranked = []
    for c in s["characters"]:
        if c.get("pending"):
            rank, label = 3, PENDING_NAME
        elif c.get("auto_name", True):
            rank, label = _describe(root, c["images"])
            label = label or "キャラ"
        else:
            rank, label = 0, c["name"]
        ranked.append((rank, -len(c["images"]), c, label))
    ranked.sort(key=lambda r: (r[0], r[1]))
    counts: dict[str, int] = {}
    for _rank, _n, c, label in ranked:
        if c.get("auto_name", True) and not c.get("pending"):
            counts[label] = counts.get(label, 0) + 1
    seen: dict[str, int] = {}
    for _rank, _n, c, label in ranked:
        if c.get("pending"):
            c["name"] = PENDING_NAME
        elif c.get("auto_name", True):
            seen[label] = seen.get(label, 0) + 1
            c["name"] = label if counts[label] == 1 else f"{label} {chr(64 + seen[label]) if seen[label] <= 26 else seen[label]}"
            c["auto_name"] = True
    s["characters"] = [r[2] for r in ranked]
    s["naming"] = NAMING
    return s


def _from_clusters(root: Path, clusters: list[dict], prefix: str) -> dict:
    chars = [{"id": i, "name": "", "auto_name": True, "images": [prefix + f for f in c["files"]]}
             for i, c in enumerate(sorted(clusters, key=lambda c: -c["images"]), 1)]
    return _relabel(root, {"version": 1, "characters": chars, "excluded": [], "next_id": len(chars) + 1})


def _from_result(root: Path, result: dict, prefix: str) -> dict:
    """crop_cluster.py output: sure characters, then one pending pile."""
    chars = [{"id": i, "name": "", "auto_name": True, "images": [prefix + f for f in c["files"]]}
             for i, c in enumerate(result["characters"], 1)]
    chars.append({"id": len(chars) + 1, "name": PENDING_NAME, "auto_name": True, "pending": True,
                  "images": [prefix + f for f in result["pending"]["files"]]})
    return _relabel(root, {"version": 1, "characters": chars, "excluded": [], "next_id": len(chars) + 1, "clustering": result["params"]})


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
        # features are computed once per crop and reused; the distance matrix is cached next to them
        for cmd in ([str(FEAT_TOOL), str(cache / "feat"), str(root / svc.JOBS_DIRNAME), str(cache / "jobs.json")],
                    [str(CLUSTER_TOOL), str(cache / "feat"), str(cache / "characters_auto.json")]):
            r = subprocess.run([str(python), "-X", "utf8", *cmd], env=env, capture_output=True, text=True, encoding="utf-8",
                               errors="replace", timeout=7200)
            if r.returncode:
                raise RuntimeError((r.stderr or r.stdout)[-400:])
        result = json.loads((cache / "characters_auto.json").read_text(encoding="utf-8"))
        _save(d, _from_result(root, result, svc.JOBS_DIRNAME + "/"), history=(d / "characters.json").is_file())
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
                _start_warm(root, d, sid)
                return {"set_id": sid, "status": "ready"}
            # an existing image-level clustering of exactly these videos is reused (it takes minutes to compute)
            prev = root / svc.JOBS_DIRNAME / "_links_crop.json"
            if prev.is_file():
                links = json.loads(prev.read_text(encoding="utf-8"))
                used = {f.split("/")[0] for c in links["clusters"] for f in c["files"]}
                if used == set(jobs):
                    _save(d, _from_clusters(root, links["clusters"], svc.JOBS_DIRNAME + "/"), history=False)
                    _start_warm(root, d, sid)
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
    auto = d / "cache" / "characters_auto.json"
    result = json.loads(auto.read_text(encoding="utf-8")) if auto.is_file() else None
    if result and s.get("clustering") != result["params"]:
        # the automatic grouping was redone with a newer rule: start from it once (the previous state stays undoable)
        with _lock:
            s = _from_result(root, result, svc.JOBS_DIRNAME + "/")
            _save(d, s, history=True)
    elif s.get("naming") != NAMING:  # names made by an older rule are updated once (hand-given names are kept)
        with _lock:
            s = _relabel(root, _load(d))
            _save(d, s, history=False)
    _start_warm(root, d, set_id)
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
    root, d = _set_dir(set_id)
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
        _save(d, _relabel(root, s))
    return {"moved": len(payload.images), "to": target}


class MergeIn(BaseModel):
    source: int
    target: int


@router.post("/{set_id}/merge")
def merge(set_id: str, payload: MergeIn):
    root, d = _set_dir(set_id)
    with _lock:
        s = _load(d)
        src = next((c for c in s["characters"] if c["id"] == payload.source), None)
        dst = next((c for c in s["characters"] if c["id"] == payload.target), None)
        if src is None or dst is None or src is dst:
            raise HTTPException(400, "結合するキャラが不正です")
        dst["images"] += src["images"]
        s["characters"] = [c for c in s["characters"] if c is not src]
        _save(d, _relabel(root, s))
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
        c["auto_name"] = False
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


def _thumb_file(d: Path, rel: str, size: int) -> Path:
    return d / "cache" / "thumbs" / f"{hashlib.sha1(f'{rel}|{size}'.encode()).hexdigest()}.jpg"


def _make_thumb(src: Path, dest: Path, size: int) -> None:
    from PIL import Image

    dest.parent.mkdir(parents=True, exist_ok=True)
    im = Image.open(src).convert("RGB")
    im.thumbnail((size, size), Image.LANCZOS, reducing_gap=2.0)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=82)
    tmp = dest.with_suffix(f".{threading.get_ident()}.tmp")
    tmp.write_bytes(buf.getvalue())
    os.replace(tmp, dest)


_warming: set[str] = set()


def _start_warm(root: Path, d: Path, sid: str) -> None:
    if sid not in _warming:
        _warming.add(sid)
        threading.Thread(target=_warm_thumbs, args=(root, d), daemon=True, name=f"thumbs-{sid}").start()


def _warm_thumbs(root: Path, d: Path, size: int = 192) -> None:
    """Make every thumbnail of the set in the background (largest characters first), so browsing never waits on PIL.
    Runs once per set (results stay in cache/thumbs) on one thread at the lowest OS priority, so it never competes with the desktop."""
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadPriority(ctypes.windll.kernel32.GetCurrentThread(), -15)  # THREAD_PRIORITY_IDLE
    except Exception:  # noqa: BLE001
        pass
    try:
        s = _load(d)
        for c in sorted(s["characters"], key=lambda c: -len(c["images"])):
            for rel in c["images"]:
                t = _thumb_file(d, rel, size)
                if not t.is_file():
                    try:
                        _make_thumb((root / rel), t, size)
                    except OSError:
                        pass
    except Exception:  # noqa: BLE001 - warming is best effort
        pass


@router.get("/{set_id}/file")
def file(set_id: str, rel: str = Query(..., min_length=1, max_length=1024)):
    root, _d = _set_dir(set_id)
    return FileResponse(_image_path(root, rel), headers={"Cache-Control": "private, max-age=3600"})


@router.get("/{set_id}/thumb")
def thumb(set_id: str, rel: str = Query(..., min_length=1, max_length=1024), s: int = 192):
    """Small JPEG for the grid (cached under the set), so hundreds of 1024 px PNGs do not have to be decoded by the browser."""
    root, d = _set_dir(set_id)
    size = max(64, min(512, s))
    c = _thumb_file(d, rel, size)
    if not c.is_file():  # crops never change after the pipeline wrote them, so a made thumbnail is served as is
        _make_thumb(_image_path(root, rel), c, size)
    return FileResponse(c, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"})
