"""Cross-video character linking: several video-dataset jobs -> "same character" clusters -> ONE new LoRA project.

Everything under ``<dataset>/../.video-datasets`` is searched; nothing reaches a training dataset until ``compose`` is called.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ..schemas import ProjectCreate
from ..services import video_dataset_job as svc
from . import dataset_files, projects
from .dataset_video_chars import (_copy_character, _plan_triggers, _register_concepts, _unique_project_name, safe_project_name)

router = APIRouter(prefix="/dataset-files/{project_id}/video-links", tags=["dataset-video-links"])
_state: dict = {"running": False, "started": 0.0, "log": "", "error": ""}
_state_lock = threading.Lock()
TOOL = Path(__file__).resolve().parents[3] / "tools" / "video_dataset" / "link_characters.py"


class ComputePayload(BaseModel):
    job_ids: list[str] | None = None
    source_folder: str | None = None   # only jobs whose VIDEO lives in this folder (a series / a batch); default = the folder of the newest job
    q_auto: float = 0.75
    q_suggest: float = 0.60
    gpu: int | None = None


class Member(BaseModel):
    job_id: str
    folder: str


class ComposePayload(BaseModel):
    name: str
    members: list[Member]
    min_instance_images: int = 12  # an outfit (same label across videos) needs this many images to become its own instance


MIN_INSTANCE_IMAGES = 12
NO_OUTFIT_LABEL = "衣装不明"
IMAGE_EXT = (".png", ".jpg", ".jpeg")


def _instance_plan(root: Path, members: list[dict], min_images: int) -> tuple[list[dict], int]:
    """The instances a LoRA built from these groups gets: ONE per outfit label across all videos (an outfit seen in 5 videos is one
    instance, not 5), only when it has >= min_images; unknown or rare outfits stay character-only images. Returns (instances, other images)."""
    labels: dict[str, dict] = {}
    for m in members:
        base = root / m["job"] / m["folder"]
        for od in sorted(base.glob("outfit_*")) if base.is_dir() else []:
            pics = sorted(p for p in od.iterdir() if p.suffix.lower() in IMAGE_EXT)
            if not pics:
                continue
            label = _outfit_label(od.name)
            e = labels.setdefault(label, {"label": label, "images": 0, "videos": set(), "sample": []})
            e["images"] += len(pics)
            e["videos"].add(m["job"])
            if len(e["sample"]) < 3:
                e["sample"].append(f"{m['job']}/{m['folder']}/{od.name}/{pics[len(pics) // 2].name}")
    inst = sorted((e for e in labels.values() if e["label"] != NO_OUTFIT_LABEL and e["images"] >= min_images), key=lambda e: -e["images"])
    rest = sum(e["images"] for e in labels.values()) - sum(e["images"] for e in inst)
    return [{**e, "videos": len(e["videos"])} for e in inst], rest


def _jobs_root(project_id: int) -> Path:
    return svc.jobs_root_for(dataset_files.root_for(project_id))


def _done_jobs(project_id: int) -> list[str]:
    root = _jobs_root(project_id)
    out = []
    for d in sorted(root.iterdir()) if root.is_dir() else []:
        meta = svc.load_json(d / "job.json") if d.is_dir() else None
        if isinstance(meta, dict) and meta.get("project_id") == project_id and meta.get("status") == "done" and (d / "report.json").is_file():
            out.append(d.name)
    return out


def _run_link(project_id: int, jobs: list[str], q_auto: float, q_suggest: float, gpu: int) -> None:
    root = _jobs_root(project_id)
    python, comfy_root = svc.find_comfy()
    env = os.environ.copy()
    env.update({"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"})
    outs = {}
    try:
        for tag, q in (("auto", q_auto), ("suggest", q_suggest)):
            out = root / f"_links_{tag}.json"
            done = subprocess.run([str(python), "-X", "utf8", str(TOOL), str(out), str(root), *jobs, "--q", str(q)], env=env,
                                  capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=3600)
            if done.returncode != 0:
                raise RuntimeError((done.stderr or done.stdout)[-400:])
            outs[tag] = str(out)
        with _state_lock:
            _state.update(error="", log="完了", finished=time.time())
    except Exception as exc:  # noqa: BLE001
        with _state_lock:
            _state.update(error=str(exc)[:400])
    finally:
        with _state_lock:
            _state["running"] = False


def _video_folder(project_id: int, job: str) -> str:
    meta = svc.load_json(_jobs_root(project_id) / job / "job.json") or {}
    return str(Path(str(meta.get("video") or "")).parent)


@router.get("/sources")
def sources(project_id: int):
    """Finished jobs grouped by the folder of their video: different series must not be linked together."""
    groups: dict[str, list[str]] = {}
    for job in _done_jobs(project_id):
        groups.setdefault(_video_folder(project_id, job), []).append(job)
    return {"sources": [{"folder": f, "jobs": len(j)} for f, j in sorted(groups.items(), key=lambda kv: -len(kv[1]))]}


@router.post("/compute")
def compute(project_id: int, payload: ComputePayload):
    all_jobs = _done_jobs(project_id)
    if payload.job_ids:
        jobs = payload.job_ids
    else:
        folder = payload.source_folder
        if folder is None and all_jobs:
            newest = max(all_jobs, key=lambda j: (_jobs_root(project_id) / j / "job.json").stat().st_mtime)
            folder = _video_folder(project_id, newest)
        jobs = [j for j in all_jobs if _video_folder(project_id, j) == folder]
    if len(jobs) < 1:
        raise HTTPException(409, "完了した動画解析がありません")
    with _state_lock:
        if _state["running"]:
            raise HTTPException(409, "キャラの結合候補を計算中です")
        _state.update(running=True, started=time.time(), log="計算中", error="")
    gpu = payload.gpu if payload.gpu is not None else svc.pick_gpu()
    threading.Thread(target=_run_link, args=(project_id, jobs, payload.q_auto, payload.q_suggest, gpu), daemon=True, name="video-link").start()
    return {"started": True, "jobs": len(jobs), "gpu": gpu}


def _outfit_label(folder: str) -> str:
    import re
    m = re.match(r"outfit_(\d+)_(.*)", folder)
    if not m:
        return folder
    n, rest = m.group(1), m.group(2)
    return "衣装不明" if rest.startswith(("不明", "その他")) or n in ("00", "99") else rest.replace("-", " ")


def _add_instances(root: Path, data: dict) -> None:
    """Per cluster: the outfits (= instances the new LoRA would get) with a count and two sample images each, plus a frequency tier."""
    for c in data.get("clusters", []):
        outfits: dict[str, dict] = {}
        for m in c["members"]:
            base = root / m["job"] / m["folder"]
            for od in sorted(base.glob("outfit_*")) if base.is_dir() else []:
                pics = sorted(od.glob("*.png"))
                if not pics:
                    continue
                o = outfits.setdefault(_outfit_label(od.name), {"label": _outfit_label(od.name), "images": 0, "sample": []})
                o["images"] += len(pics)
                if len(o["sample"]) < 2:
                    o["sample"].append(f"{m['job']}/{m['folder']}/{od.name}/{pics[len(pics) // 2].name}")
        c["instances"] = sorted(outfits.values(), key=lambda o: -o["images"])[:8]
        videos = {m["video"] for m in c["members"]}
        c["videos"] = len(videos)
        c["video_names"] = sorted(videos)
        c["tier"] = "recommended" if c["images"] >= 60 and len(videos) >= 2 else ("candidate" if c["images"] >= 40 else "few")


def _proposals(root: Path, clusters: list[dict], limit: int = 12, min_images: int = 40) -> list[dict]:
    """Ready-to-create LoRA datasets, most frequent characters first: who (the merged groups), how many images, from how many
    videos, and which instances (merged outfits) it would get."""
    ranked = sorted((c for c in clusters if c["images"] >= min_images), key=lambda c: -(c["images"] * (1 + 0.25 * (c.get("videos", 1) - 1))))
    out = []
    for c in ranked[:limit]:
        inst, rest = _instance_plan(root, c["members"], MIN_INSTANCE_IMAGES)
        name = " ".join(x for x in (c.get("hair", ""), c.get("eyes", "")) if x) or f"キャラ{c['id']}"
        out.append({"cluster": c["id"], "name": name, "images": c["images"], "videos": c.get("videos", 1), "video_names": c.get("video_names", []),
                    "groups": c["groups"], "hair": c.get("hair", ""), "eyes": c.get("eyes", ""), "members": [{"job": m["job"], "folder": m["folder"]} for m in c["members"]],
                    "instances": [{k: e[k] for k in ("label", "images", "videos", "sample")} for e in inst], "character_only_images": rest,
                    "min_instance_images": MIN_INSTANCE_IMAGES})
    return out


@router.get("/result")
def result(project_id: int):
    root = _jobs_root(project_id)
    data: dict = {}
    for tag in ("auto", "suggest"):
        p = root / f"_links_{tag}.json"
        if p.is_file():
            data[tag] = json.loads(p.read_text(encoding="utf-8"))
            _add_instances(root, data[tag])
    if "auto" in data:
        data["proposals"] = _proposals(root, data["auto"]["clusters"])
    with _state_lock:
        st = dict(_state)
    return {"state": st, **data}


@router.get("/file")
def file(project_id: int, rel: str = Query(..., min_length=1, max_length=1024)):
    from fastapi.responses import FileResponse
    root = _jobs_root(project_id).resolve()
    path = (root / rel.replace("\\", "/")).resolve()
    if not path.is_relative_to(root) or path.suffix.lower() not in (".png", ".jpg", ".jpeg") or not path.is_file():
        raise HTTPException(404, "画像が見つかりません")
    return FileResponse(path, media_type="image/png" if path.suffix.lower() == ".png" else "image/jpeg", headers={"Cache-Control": "private, max-age=300"})


_groups_cache: dict[str, tuple[float, list]] = {}  # job -> (report mtime, its groups); compose_report walks the folders (~0.3 s/job on OneDrive)


def _job_groups(root: Path, job: str) -> list:
    rp = root / job / "report.json"
    stamp = rp.stat().st_mtime if rp.is_file() else 0.0
    hit = _groups_cache.get(job)
    if hit and hit[0] == stamp:
        return hit[1]
    meta = svc.load_json(root / job / "job.json") or {}
    report = svc.compose_report(root / job) or {}
    video = Path(str(meta.get("video") or job)).stem
    rows = [{"job": job, "video": video, "folder": ch["folder"], "images": ch.get("images_total", 0), "sample": [f"{job}/{r}" for r in ch.get("sample_files", [])[:4]],
             "outfits": [o["folder"] for o in ch.get("outfits", [])]} for ch in report.get("characters", [])]
    _groups_cache[job] = (stamp, rows)
    return rows


@router.get("/groups")
def groups(project_id: int):
    """Every character group of the jobs in the current link result (all finished jobs when nothing is linked yet), for picking by hand."""
    root = _jobs_root(project_id)
    jobs = _done_jobs(project_id)
    linked = root / "_links_auto.json"
    if linked.is_file():  # same videos as the shown candidates: never mix in another source folder
        used = {m["job"] for c in json.loads(linked.read_text(encoding="utf-8")).get("clusters", []) for m in c["members"]}
        jobs = [j for j in jobs if j in used]
    return {"groups": [g for job in jobs for g in _job_groups(root, job)]}


@router.post("/compose")
def compose(project_id: int, payload: ComposePayload):
    """Copy the chosen groups (any videos) into ONE new character project; every source outfit folder becomes an instance."""
    if not payload.members:
        raise HTTPException(422, "まとめるグループを選んでください")
    root = _jobs_root(project_id)
    with svc._ops_lock, dataset_files._lock:
        plan = []
        for m in payload.members:
            jdir = root / m.job_id
            report = svc.load_report(jdir)
            ch = next((c for c in report["characters"] if c["folder"] == m.folder), None)
            src = jdir / m.folder
            if ch is None or not src.is_dir():
                raise HTTPException(404, f"グループが見つかりません: {m.job_id}/{m.folder}")
            meta = svc.load_json(jdir / "job.json") or {}
            plan.append((ch, src, Path(str(meta.get("video") or m.job_id)).stem))
        inst, _rest = _instance_plan(root, [{"job": m.job_id, "folder": m.folder} for m in payload.members], max(1, payload.min_instance_images))
        with dataset_files.database_connection() as conn:
            name = _unique_project_name(conn, safe_project_name(payload.name))
        created = projects.create_project(ProjectCreate(name=name, project_type="character"))
        try:
            new_char = "ch01"
            trig = {e["label"]: f"{new_char}_o{i + 1:02d}" for i, e in enumerate(inst)}  # same outfit in every video -> one trigger
            total = 0
            dest_root = Path(created.dataset_dir)
            for ch, src, video in plan:
                outfit_trigger = {o["folder"]: o["trigger"] for o in ch.get("outfits", [])}
                for od in sorted(d for d in src.iterdir() if d.is_dir()):
                    label = _outfit_label(od.name)
                    new_o = trig.get(label, "")
                    target = dest_root / (f"{new_o}_{label.replace(' ', '-')}" if new_o else "キャラのみ（衣装インスタンスなし）")
                    target.mkdir(parents=True, exist_ok=True)
                    old_o = outfit_trigger.get(od.name)
                    for f in sorted(p for p in od.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXT):
                        stem = f"{video}_{f.stem}"  # several videos land in one folder: keep names unique and traceable
                        shutil.copy2(f, target / f"{stem}{f.suffix.lower()}")
                        total += 1
                        cap = f.with_suffix(".txt")
                        if cap.is_file():
                            tokens = cap.read_text(encoding="utf-8-sig").split(", ")
                            if tokens and tokens[0].strip() == ch["trigger"]:
                                tokens[0] = new_char
                            if len(tokens) > 1 and old_o and tokens[1].strip() == old_o:
                                if new_o:
                                    tokens[1] = new_o
                                else:
                                    del tokens[1]  # rare/unknown outfit: no outfit trigger at all
                            (target / f"{stem}.txt").write_text(", ".join(tokens), encoding="utf-8", newline="")
            ch0 = plan[0][0]
            with dataset_files.database_connection() as conn:
                concepts = _register_concepts(conn, created.id, {**ch0, "trigger": ch0["trigger"]}, new_char, {label: ("", t) for label, t in trig.items()}, name)
        except Exception:
            with dataset_files.database_connection() as conn:
                conn.execute("DELETE FROM projects WHERE id=?", (created.id,))
            shutil.rmtree(Path(created.base_dir), ignore_errors=True)
            raise
    return {"project_id": created.id, "name": name, "images": total, "groups": len(plan), "concepts": concepts,
            "instances": [{"label": e["label"], "images": e["images"], "trigger": trig[e["label"]]} for e in inst]}
