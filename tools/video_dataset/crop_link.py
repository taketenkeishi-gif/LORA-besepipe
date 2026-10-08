"""Characters across videos, decided per IMAGE by CCIP only (replaces group-level linking with tags/CLIP).

python crop_link.py OUT.json JOBS_ROOT JOBS.json [--t 0.178]
Every crop of every job - grouped or "_仕分け不能" - is one item. Identity distance = CCIP (anime character model).
Average-linkage clustering cut at --t (0.178 = the model's own same-character threshold); two people in one frame never merge.
Nanoha check (4 videos, 306 labelled crops): recall in one cluster 0.389 (group linking) -> 0.892 here, sure-wrong-hair 0.2 %.
Hair/eye tags are NOT used to decide; they only flag crops that disagree with their cluster's majority (for review).
"""
import argparse, json, re, sys, time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from ccip import Ccip  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("jobs_root"); ap.add_argument("jobs_json")
ap.add_argument("--t", type=float, default=0.178)
args = ap.parse_args()
root = Path(args.jobs_root)
jobs = json.loads(Path(args.jobs_json).read_text(encoding="utf-8"))
HAIR = re.compile(r"^[a-z ]+ hair$")
COLOURS = {"blonde", "black", "blue", "pink", "purple", "green", "white", "grey", "gray", "silver", "aqua", "red", "brown", "orange", "light blue", "dark blue", "light purple", "light brown"}

items = []
for j in jobs:
    meta = json.loads((root / j / "job.json").read_text(encoding="utf-8")) if (root / j / "job.json").is_file() else {}
    video = Path(str(meta.get("video") or meta.get("video_path") or j)).stem
    for d in sorted((root / j).iterdir()):
        if not d.is_dir() or not (d.name.startswith("char_") or d.name.startswith("_仕分け不能")):
            continue
        for p in sorted(d.rglob("*.png")):
            parts = p.relative_to(root / j).parts
            outfit = parts[1] if d.name.startswith("char_") and len(parts) > 2 else ""
            m = re.search(r"_s(\d+)_(\d+)m(\d+)s_", p.name)
            cap = p.with_suffix(".txt")
            tags = [t.strip() for t in cap.read_text(encoding="utf-8").split(",")] if cap.is_file() else []
            hair = [t for t in tags if HAIR.match(t) and t[:-5] in COLOURS]
            eyes = [t for t in tags if t.endswith(" eyes")]
            items.append({"rel": p.relative_to(root).as_posix(), "job": j, "video": video, "group": d.name if d.name.startswith("char_") else "",
                          "outfit": outfit, "frame": f"{j}:{m.group(0)}" if m else "", "hair": hair[0] if hair else "", "eyes": eyes[0] if eyes else ""})
n = len(items)
print(f"crops {n} in {len(jobs)} videos", flush=True)
t0 = time.time()
import torch  # noqa: E402

cc = Ccip(0 if torch.cuda.is_available() else None)
F = cc.feat([root / it["rel"] for it in items])
print(f"features {time.time() - t0:.0f}s", flush=True)
cond = np.empty(n * (n - 1) // 2, np.float64)  # condensed distance matrix, filled block by block
B = 1024
row_start = lambda i: i * n - i * (i + 1) // 2
for a in range(0, n, B):
    la = min(B, n - a)
    for b in range(a, n, B):
        lb = min(B, n - b)
        blk = cc.dist(np.concatenate([F[a:a + la], F[b:b + lb]]))[:la, la:]
        for x in range(la):
            i = a + x
            js = np.arange(max(b, i + 1), b + lb)
            if len(js):
                cond[row_start(i) + (js - i - 1)] = blk[x, js - b]
print(f"distances {time.time() - t0:.0f}s", flush=True)
frames = defaultdict(list)
for i, it in enumerate(items):
    if it["frame"]:
        frames[it["frame"]].append(i)
for idx in frames.values():  # two people in one frame are never the same character
    for x in idx:
        for y in idx:
            if x < y:
                cond[row_start(x) + (y - x - 1)] = 10.0
from scipy.cluster.hierarchy import fcluster, linkage  # noqa: E402

Z = linkage(cond, method="average")
labels = fcluster(Z, args.t, criterion="distance")
print(f"clustered {time.time() - t0:.0f}s", flush=True)
members = defaultdict(list)
for i, l in enumerate(labels):
    members[int(l)].append(i)
clusters = []
for cid, idx in members.items():
    hair = Counter(items[i]["hair"] for i in idx if items[i]["hair"])
    eyes = Counter(items[i]["eyes"] for i in idx if items[i]["eyes"])
    top_h = hair.most_common(1)[0][0] if hair else ""
    top_e = eyes.most_common(1)[0][0] if eyes else ""
    odd = [items[i]["rel"] for i in idx if top_h and items[i]["hair"] and items[i]["hair"] != top_h]
    outfits = Counter((items[i]["outfit"].split("_", 2)[-1] if items[i]["outfit"] else "") for i in idx)
    clusters.append({"id": cid, "images": len(idx), "videos": len({items[i]["job"] for i in idx}),
                     "video_names": sorted({items[i]["video"] for i in idx}), "hair": top_h, "eyes": top_e,
                     "hair_agreement": round(hair[top_h] / sum(hair.values()), 3) if hair else None,
                     "from_unassigned": sum(1 for i in idx if not items[i]["group"]),
                     "outfits": outfits.most_common(12), "files": [items[i]["rel"] for i in idx], "hair_disagrees": odd})
clusters.sort(key=lambda c: -(c["images"] * (1 + 0.25 * (c["videos"] - 1))))
# nearest other clusters (average CCIP distance between cluster samples), to check splits of one character
D = None
big = [c for c in clusters if c["images"] >= 20][:40]
if big:
    rng = np.random.default_rng(0)
    idx_of = {it["rel"]: k for k, it in enumerate(items)}
    samp = [[idx_of[f] for f in rng.choice(c["files"], min(24, len(c["files"])), replace=False)] for c in big]
    Fs = np.concatenate([F[s] for s in samp])
    Dm = cc.dist(Fs)
    off = np.cumsum([0] + [len(s) for s in samp])
    for a, c in enumerate(big):
        near = []
        for b, o in enumerate(big):
            if a != b:
                near.append((float(np.median(Dm[off[a]:off[a + 1], off[b]:off[b + 1]])), o["id"]))
        c["nearest"] = [{"id": i, "dist": round(d, 4)} for d, i in sorted(near)[:4]]
Path(args.out).write_text(json.dumps({"t": args.t, "crops": n, "videos": len(jobs), "clusters": clusters}, ensure_ascii=False), encoding="utf-8")
print(json.dumps({"clusters": len(clusters), "with_20_plus": sum(c["images"] >= 20 for c in clusters), "top": [(c["images"], c["videos"], c["hair"], c["eyes"], c["hair_agreement"]) for c in clusters[:12]]}, ensure_ascii=False), flush=True)
