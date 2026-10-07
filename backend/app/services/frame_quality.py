"""Detect video frames that are visibly degraded ("ガビった"), judged against their own neighbours.

A frame is looked at together with the frame before and after it (a 3-frame burst).  Absolute numbers are useless
here (flat anime art already has strong block-like edges), so every signal is RELATIVE to the neighbours:

* blurry    - sharpness (3x3 Laplacian variance) well below the neighbours'.
* blocky    - 8-px block-border step grows relative to the neighbours' (codec blocking appears on that frame only).
* jitter    - the frame differs from BOTH neighbours while the neighbours agree with each other
              (glitch / flash / wrong frame).
* ghost     - the frame is a blend of two clearly different neighbours (frame-rate-conversion double image).

``assess`` returns the raw numbers; ``flags`` turns them into reasons so every rejection can be explained.
Thresholds are calibrated in tests/test_frame_quality.py (synthetic damage) and by a false-positive run on real bursts.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

BLOCK_DELTA = 0.10         # block-border ratio increase over the neighbours' average
SHARP_RATIO_MIN = 0.55     # cur sharpness / best neighbour sharpness
SHARP_NEIGHBOUR_FLOOR = 8  # ignore the blur test when the neighbours themselves are nearly flat
OUTLIER_RATIO = 2.5        # min(d_prev, d_next) / d_neighbours
OUTLIER_FLOOR = 4.0        # mean abs luma difference (0-255) below which motion is just noise
GHOST_RATIO = 0.22         # |cur - mean(prev,next)| / |prev - next|
GHOST_FLOOR = 6.0          # the neighbours must really differ before "blend" means anything


def luma(image: Image.Image, scale: int = 1) -> np.ndarray:
    gray = image.convert("L")
    if scale > 1:
        gray = gray.resize((max(8, gray.width // scale), max(8, gray.height // scale)), Image.BILINEAR)
    return np.asarray(gray, np.float32)


def sharpness(y: np.ndarray) -> float:
    lap = -4 * y[1:-1, 1:-1] + y[:-2, 1:-1] + y[2:, 1:-1] + y[1:-1, :-2] + y[1:-1, 2:]
    return float(lap.var())


def blockiness(y: np.ndarray, block: int = 8) -> float:
    """Mean luma step across 8-px column borders divided by the step inside blocks (1.0 = no blocking)."""
    dx = np.abs(np.diff(y, axis=1))
    cols = np.arange(dx.shape[1])
    border = dx[:, cols % block == block - 1].mean()
    inner = dx[:, cols % block != block - 1].mean()
    return float(border / (inner + 1e-6))


def assess(prev: Image.Image, cur: Image.Image, nxt: Image.Image | None = None) -> dict:
    yp, yc = luma(prev), luma(cur)
    neighbours = [yp] + ([luma(nxt)] if nxt is not None else [])
    out = {
        "sharpness": sharpness(yc),
        "blockiness": blockiness(yc),
        "sharp_neighbour": max(sharpness(y) for y in neighbours),
        "block_neighbour": float(np.mean([blockiness(y) for y in neighbours])),
    }
    a, c = yp[::4, ::4], yc[::4, ::4]
    d_prev = float(np.abs(c - a).mean())
    out["d_prev"] = d_prev
    if nxt is not None:
        b = neighbours[1][::4, ::4]
        d_next = float(np.abs(c - b).mean())
        d_pn = float(np.abs(a - b).mean())
        blend_err = float(np.abs(c - (a + b) / 2).mean())
        out.update(d_next=d_next, d_pn=d_pn,
                   outlier=min(d_prev, d_next) / (d_pn + 1e-3) if min(d_prev, d_next) > OUTLIER_FLOOR else 0.0,
                   ghost=blend_err / (d_pn + 1e-3) if d_pn > GHOST_FLOOR else 1.0)
    else:
        out.update(d_next=None, d_pn=None, outlier=0.0, ghost=1.0)
    out["block_delta"] = out["blockiness"] - out["block_neighbour"]
    out["sharp_ratio"] = out["sharpness"] / (out["sharp_neighbour"] + 1e-6) if out["sharp_neighbour"] > SHARP_NEIGHBOUR_FLOOR else 1.0
    return out


def flags(metrics: dict) -> list[str]:
    """Reasons this frame should not be used (empty list = usable)."""
    reasons = []
    if metrics["sharp_ratio"] < SHARP_RATIO_MIN:
        reasons.append("blurry")
    if metrics["block_delta"] > BLOCK_DELTA:
        reasons.append("blocky")
    if metrics["outlier"] > OUTLIER_RATIO:
        reasons.append("jitter")
    if metrics["ghost"] < GHOST_RATIO:
        reasons.append("ghost")
    return reasons


def quality_score(metrics: dict) -> float:
    """Higher = better; used to pick the best of several candidates of one scene."""
    return metrics["sharpness"] * (1.0 - min(0.9, max(0.0, metrics["block_delta"]) * 3)) / (1.0 + metrics["outlier"] + (1.0 if metrics["ghost"] < GHOST_RATIO else 0.0))
