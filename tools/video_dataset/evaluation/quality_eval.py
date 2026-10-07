"""Calibrate frame_quality on the real video: false positives on untouched bursts + detection of synthetic damage."""
import io
import json
import random
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, r"C:\Users\Keishi\Portfolio\Generation\Training\LoRA-Studio-Next\backend")
from app.services import frame_quality as fq  # noqa: E402
from app.services.h3_dataset import _ffmpeg  # noqa: E402

VIDEO = r"C:\Users\Keishi\Downloads\1話.mp4"
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
random.seed(11)
N = int(sys.argv[2]) if len(sys.argv) > 2 else 120
times = sorted(random.uniform(5, 595) for _ in range(N))
work = Path(tempfile.mkdtemp(prefix="qeval-"))


def burst(i_t):
    i, t = i_t
    pat = work / f"b{i:03d}_%d.png"
    subprocess.run([_ffmpeg(), "-hide_banner", "-loglevel", "error", "-ss", f"{t:.3f}", "-i", VIDEO, "-frames:v", "3", "-y", str(pat)],
                   capture_output=True, timeout=120)
    files = [work / f"b{i:03d}_{k}.png" for k in (1, 2, 3)]
    return [Image.open(f).convert("RGB") for f in files] if all(f.exists() for f in files) else None


with ThreadPoolExecutor(8) as pool:
    bursts = [b for b in pool.map(burst, enumerate(times)) if b]
print("bursts", len(bursts), flush=True)


def jpeg(im, q):
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=q)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


clean = [fq.assess(*b) for b in bursts]
for name in ("sharpness", "block_delta", "sharp_ratio", "d_pn", "outlier", "ghost"):
    arr = np.array([c[name] for c in clean if c[name] is not None])
    print(f"clean {name:11s} p5={np.percentile(arr,5):9.3f} p50={np.percentile(arr,50):9.3f} p95={np.percentile(arr,95):9.3f} max={arr.max():9.3f}", flush=True)
fp = [fq.flags(c) for c in clean]
print("clean flagged:", sum(bool(f) for f in fp), "/", len(fp), {k: sum(k in f for f in fp) for k in ("blocky", "jitter", "ghost", "blurry")}, flush=True)

rng = random.Random(3)
damage = {
    "jpeg_q8": lambda p, c, n: (p, jpeg(c, 8), n),
    "jpeg_q20": lambda p, c, n: (p, jpeg(c, 20), n),
    "blur_r3": lambda p, c, n: (p, c.filter(ImageFilter.GaussianBlur(3)), n),
    "blend_neighbours": lambda p, c, n: (p, Image.blend(p, n, 0.5), n),
    "foreign_frame": lambda p, c, n: (p, rng.choice(bursts)[1], n),
}
for name, fn in damage.items():
    got = {"blocky": 0, "jitter": 0, "ghost": 0, "blurry": 0, "any": 0}
    for idx, b in enumerate(bursts):
        p, c, n = fn(*b)
        mt = fq.assess(p, c, n)
        fl = fq.flags(mt)
        for k in ("blocky", "jitter", "ghost", "blurry"):
            got[k] += k in fl
        got["any"] += bool(fl)
    print(f"{name:17s} detected as any: {got['any']}/{len(bursts)}  by reason {got}", flush=True)
(OUT / "clean_metrics.json").write_text(json.dumps(clean, indent=1))
