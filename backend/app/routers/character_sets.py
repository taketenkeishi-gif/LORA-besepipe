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
import shutil
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
            _preload_tags(d)
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


NAMING = 5
HAIR_AGREE = 0.6   # a hair colour names the group only when this share of its images carries that colour tag
FACE_SHARE = 0.35  # ...and this share shows a face (crop view close/upper/full); below it the group is parts, props, mascots
_tags: dict[str, list[str]] = {}


_tags_loaded: set[str] = set()


def _preload_tags(d: Path) -> None:
    """All crops' tags in one read from the set's feature file (opening 11,641 caption files one by one took 40 s cold)."""
    key = str(d)
    if key in _tags_loaded:
        return
    feat = d / "cache" / "feat.json"
    if feat.is_file():
        try:
            for it in json.loads(feat.read_text(encoding="utf-8")):
                _tags.setdefault(f"{svc.JOBS_DIRNAME}/{it['rel']}", it.get("tags", []))
        except (OSError, ValueError, KeyError):
            pass
    _tags_loaded.add(key)


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
    def letters(rows):  # same label -> A, B, C … (rows already in display order)
        counts: dict[str, int] = {}
        for _k, c, label in rows:
            counts[label] = counts.get(label, 0) + 1
        seen: dict[str, int] = {}
        for _k, c, label in rows:
            if c.get("auto_name") is False:  # a name the user typed is kept
                continue
            seen[label] = seen.get(label, 0) + 1
            c["name"] = label if counts[label] == 1 else f"{label} {chr(64 + seen[label]) if seen[label] <= 26 else seen[label]}"
            c["auto_name"] = True

    mains, held = [], []
    for c in s["characters"]:
        (held if c.get("pending") else mains).append(c)
    rows = []
    for c in mains:
        if c.get("auto_name", True):
            rank, label = _describe(root, c["images"])
            rows.append(((rank, -len(c["images"])), c, label or "キャラ"))
        else:
            rows.append(((0, -len(c["images"])), c, None))
    rows.sort(key=lambda r: r[0])
    letters([r for r in rows if r[2] is not None])
    position = {c["id"]: k for k, (_k, c, _l) in enumerate(rows)}
    name_of = {c["id"]: c["name"] for c in mains}
    # pending, in review order: candidates of each character (in the characters' order), other characters, several people, undecidable
    order = {"candidates": 0, "other": 1, "multi": 2, "unknown": 3}
    hrows = []
    for c in held:
        sec = c.get("section", "unknown")
        if sec == "candidates" and c.get("parent") in name_of:
            hrows.append(((0, position[c["parent"]], 0), c, f"{name_of[c['parent']]}の候補"))
        elif sec == "candidates" or sec == "other":  # a candidate pile whose character was merged away is just another group
            c["section"] = "other"
            label = _describe(root, c["images"])[1] if c["images"] else "空"
            hrows.append(((1, 0, -len(c["images"])), c, f"別キャラ：{label}"))
        elif sec == "multi":
            hrows.append(((2, 0, 0), c, "複数人が写っている"))
        else:
            hrows.append(((3, 0, 0), c, "判定不能" if "section" in c else PENDING_NAME))
    hrows.sort(key=lambda r: r[0])
    letters(hrows)
    s["characters"] = [r[1] for r in rows] + [r[1] for r in hrows]
    s["naming"] = NAMING
    return s


def _from_clusters(root: Path, clusters: list[dict], prefix: str) -> dict:
    chars = [{"id": i, "name": "", "auto_name": True, "images": [prefix + f for f in c["files"]]}
             for i, c in enumerate(sorted(clusters, key=lambda c: -c["images"]), 1)]
    return _relabel(root, {"version": 1, "characters": chars, "excluded": [], "next_id": len(chars) + 1})


def _from_result(root: Path, result: dict, prefix: str) -> dict:
    """crop_cluster.py output: sure characters, then the pending sections (candidates per character, other characters,
    several people in one picture, undecidable) - or one pending pile for older outputs."""
    chars = [{"id": i, "name": "", "auto_name": True, "images": [prefix + f for f in c["files"]]}
             for i, c in enumerate(result["characters"], 1)]
    if "sections" in result:
        for s in result["sections"]:
            c = {"id": len(chars) + 1, "name": "", "auto_name": True, "pending": True, "section": s["section"],
                 "images": [prefix + f for f in s["files"]]}
            if s["section"] == "candidates":
                c["parent"] = s["main"] + 1  # ids of main characters are 1..N in result order
            chars.append(c)
        return _relabel(root, {"version": 1, "characters": chars, "excluded": [], "next_id": len(chars) + 1, "clustering": result["params"]})
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
        _tags_loaded.discard(str(d))
        _preload_tags(d)
        _save(d, _from_result(root, result, svc.JOBS_DIRNAME + "/"), history=(d / "characters.json").is_file())
        _computing.pop(sid, None)
    except Exception as exc:  # noqa: BLE001
        _computing[sid] = {"error": str(exc)[-300:]}


def _project_under(root: Path) -> tuple[int, Path] | None:
    """A project whose dataset folder sits in root (its video jobs go to root/.video-datasets)."""
    conn = get_conn()
    try:
        ids = [int(r["id"]) for r in conn.execute("SELECT id FROM projects")]
    finally:
        conn.close()
    for pid in ids:
        try:
            ds = dataset_files.root_for(pid)
        except Exception:  # noqa: BLE001
            continue
        if svc.jobs_root_for(ds).parent == root:
            return pid, ds
    return None


def _finalize(root: Path, d: Path):
    def done(result: dict) -> None:
        with _lock:
            _tags_loaded.discard(str(d))
            _preload_tags(d)
            meta = json.loads((d / "set.json").read_text(encoding="utf-8")) if (d / "set.json").is_file() else {}
            meta["jobs"] = json.loads((d / "cache" / "jobs.json").read_text(encoding="utf-8"))
            svc.write_json(d / "set.json", meta)
            _save(d, _from_result(root, result, svc.JOBS_DIRNAME + "/"), history=(d / "characters.json").is_file())
    return done


@router.post("/workspace")
def workspace(payload: OpenIn):
    """Open (or create) the workspace for a folder of videos and run whatever is not done yet: extraction of new episodes,
    then grouping into characters, then outfit features.  Progress: GET /{set_id} -> pipeline."""
    from ..services import video_lora_pipeline as pipe
    folder = Path(payload.folder.strip().strip('"'))
    if not pipe.list_videos(folder):
        raise HTTPException(404, "このフォルダに動画がありません")
    sid = _slug(folder)
    roots = _roots()
    root = next((r for r in roots if (r / SETS / sid / "set.json").is_file()), roots[0] if roots else None)
    target = _project_under(root) if root else None
    if target is None:
        raise HTTPException(409, "動画の保存先になるプロジェクトがありません。先にデータセットのプロジェクトを1つ作ってください")
    pid, ds = target
    d = root / SETS / sid
    d.mkdir(parents=True, exist_ok=True)
    if not (d / "set.json").is_file():
        svc.write_json(d / "set.json", {"folder": str(folder), "jobs": [], "created": time.time()})
    _set_dirs[sid] = (root, d)
    st = pipe.start(sid, d, folder, ds, pid, _finalize(root, d))
    return {"set_id": sid, "pipeline": st}


@router.get("/workspaces")
def workspaces():
    """Every workspace (folder of videos) with its stage, newest first."""
    from ..services import video_lora_pipeline as pipe
    out = []
    for root in _roots():
        base = root / SETS
        for d in base.iterdir() if base.is_dir() else []:
            if not (d / "set.json").is_file():
                continue
            meta = json.loads((d / "set.json").read_text(encoding="utf-8"))
            st = pipe.load(d)
            out.append({"set_id": d.name, "folder": meta.get("folder", ""), "name": Path(meta.get("folder", d.name)).name,
                        "videos": len(st.get("videos") or meta.get("jobs") or []), "stage": st.get("stage") or ("ready" if (d / "characters.json").is_file() else ""),
                        "running": pipe.is_running(d.name), "updated": (d / "set.json").stat().st_mtime})
    return {"workspaces": sorted(out, key=lambda w: -w["updated"])}


@router.post("/{set_id}/pipeline/cancel")
def pipeline_cancel(set_id: str):
    from ..services import video_lora_pipeline as pipe
    _set_dir(set_id)
    pipe.cancel(set_id)
    return {"ok": True}


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
    from ..services import video_lora_pipeline as pipe
    root, d = _set_dir(set_id)
    meta = json.loads((d / "set.json").read_text(encoding="utf-8"))
    pl = pipe.load(d)
    pl["running"] = pipe.is_running(set_id)
    if not (d / "characters.json").is_file():
        return {"set_id": set_id, "status": "error" if pl.get("stage") == "error" else "computing", "error": pl.get("error", ""),
                "folder": meta.get("folder", ""), "pipeline": pl}
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
            "characters": [{"id": c["id"], "name": c["name"], "count": len(c["images"]), "cover": c["images"][:1], "pending": bool(c.get("pending")), "section": c.get("section", "")} for c in s["characters"] if c["images"]],
            "excluded": len(s["excluded"]), "can_undo": any((d / "history").glob("*.json")) if (d / "history").is_dir() else False, "pipeline": pl}


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


def _candidate_rows(d: Path, s: dict) -> list[dict]:
    from ..services import candidate_calibration as cal
    auto = d / "cache" / "characters_auto.json"
    if not auto.is_file():
        return []
    judged = json.loads((d / "judged.json").read_text(encoding="utf-8")) if (d / "judged.json").is_file() else {}
    rows = cal.answers(json.loads(auto.read_text(encoding="utf-8")).get("sections", []), s, svc.JOBS_DIRNAME + "/", judged)
    pile = {c["id"]: c for c in s["characters"]}
    for r in rows:  # "still waiting": in the candidate pile of the same character
        c = pile.get(r["in"])
        r["waiting"] = bool(c and c.get("section") == "candidates" and c.get("parent") == r["parent"])
    return rows


@router.get("/{set_id}/calibration")
def calibration(set_id: str):
    """How many candidate decisions the user has made, and - once they are enough and mostly right - the score line
    above which the remaining candidates can be accepted, with the count per candidate pile."""
    from ..services import candidate_calibration as cal
    _root, d = _set_dir(set_id)
    s = _load(d)
    rows = _candidate_rows(d, s)
    info = cal.line(rows)
    per: dict[int, int] = {}
    if info["line"]:
        waiting = [r for r in rows if r["waiting"]]
        for r in cal.above(waiting, info["line"]["score"]) + [r for r in waiting if r["label"] == 1]:
            per[r["in"]] = per.get(r["in"], 0) + 1
    return {**info, "need": cal.MIN_ANSWERS, "target": cal.TARGET, "above": per}


@router.get("/{set_id}/check")
def check_sample(set_id: str, character: int, n: int = 30):
    """A quick check: n unanswered images of a candidate pile spread over its score range (best to worst)."""
    from ..services import candidate_calibration as cal
    _root, d = _set_dir(set_id)
    s = _load(d)
    rows = [r for r in _candidate_rows(d, s) if r["waiting"] and r["in"] == character]
    return {"images": cal.sample(rows, max(5, min(60, n)), seed=len([r for r in rows if r["label"] is not None]))}


class JudgeIn(BaseModel):
    right: list[str]
    wrong: list[str]


@router.post("/{set_id}/judge")
def judge(set_id: str, payload: JudgeIn):
    """Answers from a check sample: these candidates are / are not the character (images are not moved)."""
    _root, d = _set_dir(set_id)
    with _lock:
        f = d / "judged.json"
        judged = json.loads(f.read_text(encoding="utf-8")) if f.is_file() else {}
        judged.update({r: 1 for r in payload.right})
        judged.update({r: 0 for r in payload.wrong})
        svc.write_json(f, judged)
    return {"judged": len(judged)}


class AcceptIn(BaseModel):
    character: int  # a candidate pile


@router.post("/{set_id}/accept-above")
def accept_above(set_id: str, payload: AcceptIn):
    """Move the pile's undecided candidates that are within the measured line into their character (undoable)."""
    from ..services import candidate_calibration as cal
    root, d = _set_dir(set_id)
    with _lock:
        s = _load(d)
        rows = _candidate_rows(d, s)
        info = cal.line(rows)
        if not info["line"]:
            raise HTTPException(409, "まだ判定が足りません")
        waiting = [r for r in rows if r["waiting"] and r["in"] == payload.character]
        # the undecided ones within the line, and the ones the user already confirmed on a check sample
        take = cal.above(waiting, info["line"]["score"]) + [r for r in waiting if r["label"] == 1]
        if not take:
            return {"moved": 0}
        parent = take[0]["parent"]
        moving = {r["rel"] for r in take}
        for c in s["characters"]:
            if c["id"] == payload.character:
                c["images"] = [i for i in c["images"] if i not in moving]
            elif c["id"] == parent:
                c["images"] += [r["rel"] for r in take]
        _save(d, _relabel(root, s))
    return {"moved": len(take), "into": parent, "precision": info["line"]["precision"], "answers": info["line"]["answers"]}


_items_cache: dict[str, dict[str, dict]] = {}


def _items(d: Path) -> dict[str, dict]:
    """rel (with the jobs prefix) -> crop facts from the set's feature file (video, job, frame, view)."""
    key = str(d)
    feat = d / "cache" / "feat.json"
    stamp = f"{key}|{feat.stat().st_mtime if feat.is_file() else 0}"
    if stamp not in _items_cache:
        _items_cache.clear()
        data = json.loads(feat.read_text(encoding="utf-8")) if feat.is_file() else []
        _items_cache[stamp] = {f"{svc.JOBS_DIRNAME}/{it['rel']}": it for it in data}
    return _items_cache[stamp]


def _refs(images: list[str], items: dict[str, dict], n: int = 6) -> list[str]:
    """n face shots of a character, from as many different videos as possible (the picture the user compares against)."""
    faces = [r for r in images if (items.get(r) or {}).get("view", "").split("-")[-1] in ("close", "upper")] or images
    by_video: dict[str, list[str]] = {}
    for r in faces:
        by_video.setdefault((items.get(r) or {}).get("job", ""), []).append(r)
    out, k = [], 0
    pools = sorted(by_video.values(), key=len, reverse=True)
    while len(out) < n and any(k < len(p) for p in pools):
        for p in pools:
            if k < len(p) and len(out) < n:
                out.append(p[len(p) * k // max(1, (n // max(1, len(pools))) + 1) if k else 0] if k else p[0])
        k += 1
        if k > 50:
            break
    return list(dict.fromkeys(out))[:n]


@router.get("/{set_id}/outfits")
def outfits(set_id: str, character: int):
    """A character's images grouped by outfit (a suggestion, see services/outfit_split.py); cached per image list."""
    from ..services import outfit_split
    root, d = _set_dir(set_id)
    s = _load(d)
    c = next((c for c in s["characters"] if c["id"] == character), None)
    if c is None:
        raise HTTPException(404, "キャラが見つかりません")
    key = hashlib.sha1("\n".join(sorted(c["images"])).encode()).hexdigest()[:16]
    cached = d / "cache" / "outfits" / f"{key}.json"
    if cached.is_file():
        return json.loads(cached.read_text(encoding="utf-8"))
    _preload_tags(d)
    out = {"outfits": outfit_split.split(c["images"], lambda r: _crop_tags(root, r))}
    cached.parent.mkdir(parents=True, exist_ok=True)
    svc.write_json(cached, out)
    return out


@router.get("/{set_id}/rows")
def rows(set_id: str, character: int):
    """A pending pile as rows to compare: each row = images of one video's nearby scenes that most likely belong to the same
    character (its nearest main character), with that character's reference shots.  Rows in order of likelihood."""
    root, d = _set_dir(set_id)
    s = _load(d)
    pile = next((c for c in s["characters"] if c["id"] == character), None)
    if pile is None:
        raise HTTPException(404, "キャラが見つかりません")
    mains = {c["id"]: c for c in s["characters"] if not c.get("pending")}
    auto = d / "cache" / "characters_auto.json"
    near: dict[str, tuple[int, float]] = {}
    if auto.is_file():
        for sec in json.loads(auto.read_text(encoding="utf-8")).get("sections", []):
            pre = svc.JOBS_DIRNAME + "/"
            if sec.get("section") == "candidates":
                for f, sc in zip(sec["files"], sec.get("scores", [])):
                    near[pre + f] = (sec["main"] + 1, sc[0])
            else:
                for f, nr in zip(sec["files"], sec.get("nearest", [])):
                    near[pre + f] = (nr[0] + 1, nr[1])
    items = _items(d)
    groups: dict[tuple, list[tuple[int, str, float]]] = {}
    for r in pile["images"]:
        dest, score = near.get(r, (None, 9.0))
        dest = dest if dest in mains else None
        it = items.get(r) or {}
        sc = it.get("frame", "").split(":")[1] if it.get("frame") else ""
        groups.setdefault((dest, it.get("job", "")), []).append((int(sc[1:]) if sc[1:].isdigit() else 0, r, score))
    out = []
    for (dest, job), members in groups.items():
        members.sort()
        chunk: list[tuple[int, str, float]] = []
        for m in members:  # neighbouring scenes of one video, rows of 8-40 images
            if chunk and (len(chunk) >= 40 or (len(chunk) >= 8 and m[0] - chunk[-1][0] > 3)):
                out.append((dest, job, chunk)); chunk = []
            chunk.append(m)
        if chunk:
            out.append((dest, job, chunk))
    rows_out = [{"dest": dest, "video": (items.get(ch[0][1]) or {}).get("video", ""), "files": [m[1] for m in ch],
                 "score": round(min(m[2] for m in ch), 4)} for dest, job, ch in out]
    rows_out.sort(key=lambda r: (r["dest"] is None, r["score"]))
    used = {r["dest"] for r in rows_out if r["dest"] is not None}
    return {"rows": rows_out, "refs": {str(k): _refs(mains[k]["images"], items) for k in used},
            "characters": [{"id": c["id"], "name": c["name"]} for c in mains.values()]}


class ComposeOutfit(BaseModel):
    name: str
    files: list[str]
    use: bool = True


class ComposeChar(BaseModel):
    id: int
    name: str
    outfits: list[ComposeOutfit]


class ComposeIn(BaseModel):
    characters: list[ComposeChar]
    min_instance_images: int = 30


@router.post("/{set_id}/compose")
def compose(set_id: str, payload: ComposeIn):
    """One LoRA project per chosen character: its images copied with their captions, every used outfit with enough images
    becomes an instance (ch01_o01 …), the rest are character-only.  The set and the per-video crops are not changed."""
    from ..schemas import ProjectCreate
    from . import projects
    from .dataset_video_chars import _register_concepts, _unique_project_name, safe_project_name
    if not payload.characters:
        raise HTTPException(422, "LoRAにするキャラを選んでください")
    root, d = _set_dir(set_id)
    items = _items(d)
    made = []
    char_re, outfit_re = re.compile(r"^ch\d+$"), re.compile(r"^ch\d+_o\d+$")
    for ch in payload.characters:
        with dataset_files.database_connection() as conn:
            name = _unique_project_name(conn, safe_project_name(ch.name))
        created = projects.create_project(ProjectCreate(name=name, project_type="character"))
        try:
            new_char = "ch01"
            dest_root = Path(created.dataset_dir)
            instances, total, k = [], 0, 0
            for o in ch.outfits:
                if not o.files:
                    continue
                inst = o.use and len(o.files) >= payload.min_instance_images and o.name != "その他の衣装"
                trig = ""
                if inst:
                    k += 1
                    trig = f"{new_char}_o{k:02d}"
                    instances.append({"name": o.name, "trigger": trig, "images": len(o.files)})
                folder = dest_root / (f"{trig}_{safe_project_name(o.name)}" if trig else "キャラのみ（衣装インスタンスなし）")
                folder.mkdir(parents=True, exist_ok=True)
                for rel in o.files:
                    src = (root / rel).resolve()
                    if not src.is_relative_to((root / svc.JOBS_DIRNAME).resolve()) or not src.is_file():
                        continue
                    video = (items.get(rel) or {}).get("video") or src.parent.parent.name
                    stem = f"{safe_project_name(video)}_{src.stem}"
                    shutil.copy2(src, folder / f"{stem}{src.suffix.lower()}")
                    total += 1
                    cap = src.with_suffix(".txt")
                    tokens = [t.strip() for t in cap.read_text(encoding="utf-8-sig").split(",")] if cap.is_file() else []
                    tokens = [t for t in tokens if t and not outfit_re.match(t)]
                    tokens = [t for t in tokens if not char_re.match(t)]
                    (folder / f"{stem}.txt").write_text(", ".join([new_char] + ([trig] if trig else []) + tokens), encoding="utf-8", newline="")
            with dataset_files.database_connection() as conn:
                concepts = _register_concepts(conn, created.id, {}, new_char, {i["name"]: ("", i["trigger"]) for i in instances}, name)
        except Exception:
            with dataset_files.database_connection() as conn:
                conn.execute("DELETE FROM projects WHERE id=?", (created.id,))
            shutil.rmtree(Path(created.base_dir), ignore_errors=True)
            raise
        made.append({"project_id": created.id, "name": name, "images": total, "instances": instances, "concepts": len(concepts)})
    return {"projects": made}


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
