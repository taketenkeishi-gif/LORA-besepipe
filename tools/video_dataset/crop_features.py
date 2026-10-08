"""Compute once, reuse for every clustering run: CCIP features + per-crop facts of every crop of the given jobs.

python crop_features.py OUT_PREFIX JOBS_ROOT JOBS.json
Writes OUT_PREFIX.npy (float32, one row per crop) and OUT_PREFIX.json (items in the same order).
Already computed rows are reused when OUT_PREFIX.* exist (keyed by crop path + file size + mtime), so adding a video only embeds its crops.
"""
import argparse, json, os, re, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("jobs_root"); ap.add_argument("jobs_json")
args = ap.parse_args()
root, out = Path(args.jobs_root), Path(args.out)
jobs = json.loads(Path(args.jobs_json).read_text(encoding="utf-8"))

items = []
for j in jobs:
    meta = json.loads((root / j / "job.json").read_text(encoding="utf-8")) if (root / j / "job.json").is_file() else {}
    video = Path(str(meta.get("video") or meta.get("video_path") or j)).stem
    for d in sorted((root / j).iterdir()):
        if not d.is_dir() or not (d.name.startswith("char_") or d.name.startswith("_仕分け不能")):
            continue
        for p in sorted(d.rglob("*.png")):
            parts = p.relative_to(root / j).parts
            m = re.search(r"_s(\d+)_(\d+)m(\d+)s_([a-z]+-[a-z-]+)_p\d+", p.name)
            cap = p.with_suffix(".txt")
            st = p.stat()
            items.append({"rel": p.relative_to(root).as_posix(), "job": j, "video": video,
                          "group": d.name if d.name.startswith("char_") else "",
                          "outfit": parts[1] if d.name.startswith("char_") and len(parts) > 2 else "",
                          "frame": f"{j}:{m.group(1)}:{m.group(2)}m{m.group(3)}s" if m else "",
                          "view": m.group(4) if m else "",
                          "tags": [t.strip() for t in cap.read_text(encoding="utf-8").split(",") if t.strip()] if cap.is_file() else [],
                          "key": f"{st.st_size}:{int(st.st_mtime)}"})
n = len(items)
old_F, old_idx = None, {}
if out.with_suffix(".npy").is_file() and out.with_suffix(".json").is_file():
    old_items = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    old_F = np.load(out.with_suffix(".npy"))
    old_idx = {(it["rel"], it.get("key")): k for k, it in enumerate(old_items)}
reuse = [old_idx.get((it["rel"], it["key"])) for it in items]
todo = [k for k, r in enumerate(reuse) if r is None]
print(f"crops {n} in {len(jobs)} videos; reuse {n - len(todo)}, embed {len(todo)}", flush=True)
t0 = time.time()
F = None
if todo:
    import torch  # noqa: E402
    from ccip import Ccip  # noqa: E402

    cc = Ccip(0 if torch.cuda.is_available() else None)
    progress = os.environ.get("PROGRESS_FILE")
    parts = []
    for a in range(0, len(todo), 256):  # in chunks, so the app can show how far it is
        parts.append(cc.feat([root / items[k]["rel"] for k in todo[a:a + 256]]))
        if progress:
            Path(progress).write_text(json.dumps({"done": min(a + 256, len(todo)), "total": len(todo), "elapsed_s": round(time.time() - t0, 1)}), encoding="utf-8")
    new = np.concatenate(parts)
    F = np.zeros((n, new.shape[1]), np.float32)
    F[todo] = new
else:
    F = np.zeros((n, old_F.shape[1]), np.float32)
for k, r in enumerate(reuse):
    if r is not None:
        F[k] = old_F[r]
np.save(out.with_suffix(".npy"), F)
out.with_suffix(".json").write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
print(f"features {time.time() - t0:.0f}s -> {out.with_suffix('.npy')}", flush=True)
