"""Folder of videos -> characters, as one queued pipeline (the "動画からLoRA" workspace).

Stages, run one after another in a background thread per workspace (= character set):
  videos      each episode is extracted (video_dataset_job, one GPU job at a time; already extracted episodes are reused)
  features    CCIP features of every crop (tools/video_dataset/crop_features.py; reused per crop)
  characters  sure characters + pending sections (tools/video_dataset/crop_cluster.py)
(The outfit split needs no stage: it uses the crops' clothing tags and is computed per character when shown.)
Progress lives in <set>/pipeline.json so the app can show it anywhere and after a restart.  A failed episode does not stop the
others; it is listed with its error.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable

from . import video_dataset_job as vj

TOOLS = Path(__file__).resolve().parents[3] / "tools" / "video_dataset"
DEFAULT_PARAMS = {"scene_sensitivity": "normal", "frame_interval": 0.5, "max_frames_per_scene": 24, "others": "keep",
                  "overlap_threshold": 0.25, "margin": 0.10, "long_side": 1024, "min_area": 0.01, "min_short": 100,
                  "min_char_crops": 3, "merge_quantile": 0.80, "group_merge_quantile": 0.55, "rescue_quantile": 0.55,
                  "yolo_conf": 0.15, "cascade_margin": 0.8, "early_dup_bits": 1, "use_names": False, "gpu": None, "limit_seconds": 0}
STAGES = ("videos", "features", "characters")
_running: dict[str, threading.Thread] = {}
_cancel: set[str] = set()
_lock = threading.Lock()


def _natural(name: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name.lower())]


def list_videos(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted((p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in vj.VIDEO_EXTENSIONS), key=lambda p: _natural(p.stem))


def _norm(p: str | Path) -> str:
    return os.path.normcase(os.path.normpath(str(p)))


def existing_jobs(jobs_root: Path) -> dict[str, dict]:
    """Latest usable job per video path: done (with report) beats running beats anything else."""
    best: dict[str, dict] = {}
    rank = {"done": 3, "running": 2}
    for d in jobs_root.iterdir() if jobs_root.is_dir() else []:
        if not (d / "job.json").is_file():
            continue
        meta = vj.refresh_meta(d) or {}
        video = meta.get("video") or (meta.get("params") or {}).get("video_path")
        if not video:
            continue
        status = meta.get("status")
        if status == "done" and not (d / "report.json").is_file():
            status = "error"
        cur = best.get(_norm(video))
        if cur is None or rank.get(status, 0) > rank.get(cur["status"], 0) or \
                (rank.get(status, 0) == rank.get(cur["status"], 0) and (meta.get("created_at") or 0) > cur["created"]):
            best[_norm(video)] = {"job": d.name, "status": status, "created": meta.get("created_at") or 0, "error": meta.get("error") or ""}
    return best


def load(d: Path) -> dict:
    f = d / "pipeline.json"
    try:
        return json.loads(f.read_text(encoding="utf-8")) if f.is_file() else {}
    except (OSError, ValueError):
        return {}


def save(d: Path, state: dict) -> None:
    vj.write_json(d / "pipeline.json", state)


def plan(folder: Path, jobs_root: Path) -> list[dict]:
    jobs = existing_jobs(jobs_root)
    out = []
    for v in list_videos(folder):
        j = jobs.get(_norm(v))
        status = j["status"] if j and j["status"] in ("done", "running") else "queued"
        out.append({"path": str(v), "name": v.stem, "job": j["job"] if j and status != "queued" else None, "status": status,
                    "percent": 100 if status == "done" else 0, "eta_s": None, "error": ""})
    return out


def is_running(sid: str) -> bool:
    t = _running.get(sid)
    return bool(t and t.is_alive())


def cancel(sid: str) -> None:
    _cancel.add(sid)


def start(sid: str, d: Path, folder: Path, dataset_root: Path, project_id: int, finalize: Callable[[dict], None]) -> dict:
    """(Re)plan the workspace and run whatever is not done yet.  finalize(result) stores the new grouping (router side)."""
    with _lock:
        if is_running(sid):
            return load(d)
        jobs_root = vj.jobs_root_for(dataset_root)
        videos = plan(folder, jobs_root)
        st = load(d)
        need_videos = any(v["status"] != "done" for v in videos)
        done_jobs = sorted(v["job"] for v in videos if v["status"] == "done")
        cache = d / "cache"
        if not st.get("grouped_jobs") and (d / "characters.json").is_file() and (cache / "jobs.json").is_file():
            try:  # a set grouped before this pipeline existed: its grouping (and the user's edits) stay
                st["grouped_jobs"] = sorted(json.loads((cache / "jobs.json").read_text(encoding="utf-8")))
            except (OSError, ValueError):
                pass
        grouped = (d / "characters.json").is_file() and sorted(st.get("grouped_jobs") or []) == done_jobs and not need_videos
        if grouped:
            st.update(videos=videos, stage="ready", stage_percent=100, error="")
            save(d, st)
            return st
        st.update(videos=videos, stage="videos" if need_videos else "features", stage_percent=0, error="", started=time.time(),
                  folder=str(folder))
        save(d, st)
        _cancel.discard(sid)
        t = threading.Thread(target=_run, args=(sid, d, folder, dataset_root, project_id, finalize), daemon=True, name=f"video-lora-{sid}")
        _running[sid] = t
        t.start()
        return st


def _tool(d: Path, sid: str, stage: str, script: str, args: list[str], env_extra: dict | None = None, timeout: int = 7200) -> None:
    python, comfy = vj.find_comfy()
    env = os.environ.copy()
    env.update({"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": str(vj.pick_gpu()), "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"})
    progress = d / "cache" / f"progress-{stage}.json"
    progress.unlink(missing_ok=True)
    env["PROGRESS_FILE"] = str(progress)
    env.update(env_extra or {})
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    log = open(d / "cache" / f"{stage}.log", "wb")
    proc = subprocess.Popen([str(python), "-X", "utf8", str(TOOLS / script), *args], stdout=log, stderr=subprocess.STDOUT, env=env,
                            creationflags=flags)
    t0 = time.time()
    try:
        while proc.poll() is None:
            if sid in _cancel:
                vj.kill_tree(proc)
                raise RuntimeError("中止しました")
            if time.time() - t0 > timeout:
                vj.kill_tree(proc)
                raise RuntimeError(f"{stage} が時間内に終わりませんでした")
            try:
                p = json.loads(progress.read_text(encoding="utf-8"))
                pct = int(100 * p["done"] / max(1, p["total"]))
                eta = round(p["elapsed_s"] * (p["total"] - p["done"]) / p["done"], 1) if p["done"] else None
            except (OSError, ValueError, KeyError):
                pct, eta = None, None
            st = load(d)
            if pct is not None:
                st.update(stage_percent=pct, stage_eta_s=eta)
                save(d, st)
            time.sleep(2)
    finally:
        log.close()
    if proc.returncode:
        tail = (d / "cache" / f"{stage}.log").read_text(encoding="utf-8", errors="replace")[-400:]
        raise RuntimeError(f"{stage} に失敗しました: {tail}")


def _run(sid: str, d: Path, folder: Path, dataset_root: Path, project_id: int, finalize: Callable[[dict], None]) -> None:
    st = load(d)
    try:
        # ---- videos: one extraction at a time, in episode order
        durations: list[float] = []
        for k, v in enumerate(st["videos"]):
            if v["status"] == "done":
                continue
            if sid in _cancel:
                raise RuntimeError("中止しました")
            job = v.get("job") if v["status"] == "running" else None
            while job is None:
                try:
                    job = vj.start_job(project_id, dataset_root, vj.validate_params({**DEFAULT_PARAMS, "video_path": v["path"]}))
                except vj.VideoDatasetError as exc:
                    if exc.status == 409:  # GPU busy or little memory: wait and retry
                        st = load(d); st["videos"][k].update(status="waiting", error=exc.message); save(d, st)
                        time.sleep(15)
                        if sid in _cancel:
                            raise RuntimeError("中止しました") from exc
                        continue
                    raise
            t0 = time.time()
            job_dir = vj.jobs_root_for(dataset_root) / job
            while True:
                p = vj.status_payload(job_dir)
                st = load(d)
                st["videos"][k].update(job=job, status=p["status"] if p["status"] != "done" else "done", percent=p["percent"],
                                       eta_s=p["eta_s"], error=p["error"])
                left = [x for x in st["videos"] if x["status"] not in ("done", "error", "cancelled")]
                avg = sum(durations) / len(durations) if durations else None
                st["stage_percent"] = int(100 * sum(x["percent"] for x in st["videos"]) / (100 * len(st["videos"])))
                st["stage_eta_s"] = round((p["eta_s"] or 0) + (avg or 0) * max(0, len(left) - 1), 1) if (p["eta_s"] or avg) else None
                save(d, st)
                if p["status"] in ("done", "error", "cancelled"):
                    break
                if sid in _cancel:
                    vj.cancel_job(job)
                time.sleep(2)
            durations.append(time.time() - t0)
        st = load(d)
        done_jobs = sorted(v["job"] for v in st["videos"] if v["status"] == "done" and v.get("job"))
        if not done_jobs:
            raise RuntimeError("処理できた動画がありません")
        cache = d / "cache"
        cache.mkdir(exist_ok=True)
        (cache / "jobs.json").write_text(json.dumps(done_jobs), encoding="utf-8")
        jobs_root = vj.jobs_root_for(dataset_root)
        # ---- features, characters, outfits
        st.update(stage="features", stage_percent=0, stage_eta_s=None); save(d, st)
        _tool(d, sid, "features", "crop_features.py", [str(cache / "feat"), str(jobs_root), str(cache / "jobs.json")])
        st = load(d); st.update(stage="characters", stage_percent=0, stage_eta_s=None); save(d, st)
        _tool(d, sid, "characters", "crop_cluster.py", [str(cache / "feat"), str(cache / "characters_auto.json")])
        result = json.loads((cache / "characters_auto.json").read_text(encoding="utf-8"))
        finalize(result)
        st = load(d); st.update(stage="ready", stage_percent=100, stage_eta_s=0, grouped_jobs=done_jobs, finished=time.time(), error=""); save(d, st)
    except Exception as exc:  # noqa: BLE001 - shown to the user in the workspace
        st = load(d)
        st.update(stage="error" if "中止" not in str(exc) else "cancelled", error=str(exc)[-400:])
        save(d, st)
    finally:
        _cancel.discard(sid)
