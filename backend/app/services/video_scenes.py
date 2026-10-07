"""Scene-based frame picking for building a dataset from a video.

A video has far too many frames to review, so it is split into scenes with
ffmpeg's ``scene`` filter and ONE representative frame is picked per scene
(the sharpest of a few samples from the middle of the scene).  Optionally each
representative is ranked by how closely its tags match the tags the existing
dataset images share ("is this probably the same character?").

Everything here runs on CPU through ffmpeg + Pillow (+ the WD14 tagger that the
app already uses).  No new dependency.
"""
from __future__ import annotations

import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Callable

from .h3_dataset import _ffmpeg, inspect_frame

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
_GENERIC_TAGS = {"1girl", "solo", "1boy", "looking at viewer", "simple background", "white background"}

_DURATION = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_PTS_TIME = re.compile(r"pts_time:(\d+(?:\.\d+)?)")


def probe_duration(video: Path) -> float:
    completed = subprocess.run([_ffmpeg(), "-hide_banner", "-i", str(video)], capture_output=True, text=True, timeout=60)
    match = _DURATION.search(completed.stderr or "")
    if not match:
        raise RuntimeError("動画の長さを読み取れませんでした。対応していない形式の可能性があります")
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def detect_cuts(video: Path, threshold: float, timeout: int = 1800) -> list[float]:
    """Timestamps (seconds) where ffmpeg's scene score exceeds ``threshold``."""
    command = [_ffmpeg(), "-hide_banner", "-nostats", "-i", str(video), "-an", "-sn",
               "-vf", f"select='gt(scene,{threshold})',showinfo", "-f", "null", "-"]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    if completed.returncode != 0:
        raise RuntimeError(f"シーン検出に失敗しました: {(completed.stderr or '')[-600:]}")
    return sorted({round(float(m.group(1)), 3) for m in _PTS_TIME.finditer(completed.stderr or "")})


def build_scenes(cuts: list[float], duration: float, min_seconds: float) -> list[tuple[float, float]]:
    """Cut list -> scene intervals; scenes shorter than ``min_seconds`` merge into the previous one."""
    edges = [0.0] + [c for c in cuts if 0.0 < c < duration] + [duration]
    scenes: list[tuple[float, float]] = []
    for start, end in zip(edges, edges[1:]):
        if end - start <= 0:
            continue
        if scenes and end - start < min_seconds:
            scenes[-1] = (scenes[-1][0], end)
        else:
            scenes.append((start, end))
    return scenes


def sample_times(start: float, end: float, samples: int) -> list[float]:
    """Sample the middle 60% of a scene (the first/last moments are mostly transition frames)."""
    length = end - start
    if samples <= 1 or length < 0.4:
        return [start + length / 2]
    lo, hi = start + length * 0.2, start + length * 0.8
    return [lo + (hi - lo) * i / (samples - 1) for i in range(samples)]


def extract_frame(video: Path, seconds: float, target: Path, timeout: int = 120) -> bool:
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [_ffmpeg(), "-hide_banner", "-loglevel", "error", "-ss", f"{seconds:.3f}", "-i", str(video),
               "-frames:v", "1", "-y", str(target)]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    return completed.returncode == 0 and target.is_file() and target.stat().st_size > 0


def pick_representative(video: Path, scene: tuple[float, float], index: int, out_dir: Path, samples: int) -> dict | None:
    """Sharpest of a few samples from the scene; the losing samples are deleted."""
    best: tuple[float, float, Path] | None = None
    losers: list[Path] = []
    for n, seconds in enumerate(sample_times(scene[0], scene[1], samples)):
        candidate = out_dir / f"scene_{index:04d}_{n}.png"
        if not extract_frame(video, seconds, candidate):
            continue
        sharpness = inspect_frame(candidate)["blur_score"]
        if best is None or sharpness > best[0]:
            if best is not None:
                losers.append(best[2])
            best = (sharpness, seconds, candidate)
        else:
            losers.append(candidate)
    for path in losers:
        path.unlink(missing_ok=True)
    if best is None:
        return None
    final = out_dir / f"scene_{index:04d}.png"
    best[2].replace(final)
    return {"index": index, "start": round(scene[0], 3), "end": round(scene[1], 3),
            "timestamp": round(best[1], 3), "sharpness": best[0], "file": final.name}


def character_signature(captions: list[str], exclude: set[str], ratio: float = 0.5, limit: int = 12) -> list[str]:
    """Tags shared by at least ``ratio`` of the existing captions (what 'this character' looks like)."""
    skip = {t.casefold() for t in exclude} | _GENERIC_TAGS
    counter: Counter[str] = Counter()
    usable = 0
    for caption in captions:
        tags = {t.strip().casefold() for t in re.split(r"[,\n]", caption or "") if t.strip()}
        if not tags:
            continue
        usable += 1
        counter.update(t for t in tags if t not in skip)
    if usable == 0:
        return []
    shared = [(tag, count) for tag, count in counter.items() if count >= max(1, usable * ratio)]
    shared.sort(key=lambda item: (-item[1], item[0]))
    return [tag for tag, _ in shared[:limit]]


def score_character(tags_text: str, signature: list[str]) -> tuple[float, list[str]]:
    tags = {t.strip().casefold() for t in tags_text.split(",") if t.strip()}
    matched = [tag for tag in signature if tag in tags]
    return (len(matched) / len(signature) if signature else 0.0), matched


def analyze_video(
    video: Path,
    out_dir: Path,
    *,
    threshold: float,
    min_seconds: float,
    samples: int,
    max_scenes: int,
    signature: list[str],
    tag_frame: Callable[[Path], str] | None,
    progress: Callable[[str, int, int], None],
    cancelled: Callable[[], bool],
) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    progress("シーンを検出中", 0, 1)
    duration = probe_duration(video)
    scenes = build_scenes(detect_cuts(video, threshold), duration, min_seconds)
    if not scenes:
        raise RuntimeError("シーンを検出できませんでした")
    if len(scenes) > max_scenes:
        # Keep an even spread instead of silently truncating the tail of the video.
        step = len(scenes) / max_scenes
        scenes = [scenes[int(i * step)] for i in range(max_scenes)]
    items: list[dict] = []
    for position, scene in enumerate(scenes, start=1):
        if cancelled():
            break
        progress("代表フレームを抽出中", position - 1, len(scenes))
        item = pick_representative(video, scene, position, out_dir, samples)
        if item is None:
            continue
        item["tags"] = ""
        item["score"] = None
        item["matched"] = []
        if tag_frame is not None:
            try:
                item["tags"] = tag_frame(out_dir / item["file"])
            except Exception as exc:  # noqa: BLE001 - tagging is a ranking aid, never a reason to lose the frame
                item["tag_error"] = str(exc)[:200]
            if signature and item["tags"]:
                item["score"], item["matched"] = score_character(item["tags"], signature)
        items.append(item)
    progress("完了", len(scenes), len(scenes))
    return items
