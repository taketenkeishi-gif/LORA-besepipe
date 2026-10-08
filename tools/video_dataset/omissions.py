"""What the per-video sorting left out: unassigned crops ("_仕分け不能") of ALL linked videos that look like a proposal's character.

python omissions.py OUT.json JOBS_ROOT LINKS_AUTO.json [--clusters 1 2 ...] [--max-dist 0.24]
Identity = CCIP (anime character model; 0.913 AUC vs 0.856 for CLIP+tags on the Nanoha check). A crop's distance to a
character = median CCIP distance to up to 24 of the character's images. Every crop goes to its nearest character only.
dist <= 0.178 is the model's own "same character" threshold (strong); up to --max-dist is shown as "weak" for the user to judge.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from ccip import THRESHOLD, Ccip  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("jobs_root"); ap.add_argument("links")
ap.add_argument("--clusters", type=int, nargs="*", default=None)
ap.add_argument("--max-dist", type=float, default=0.24)
ap.add_argument("--per-cluster", type=int, default=24)
ap.add_argument("--limit", type=int, default=80)
args = ap.parse_args()
root = Path(args.jobs_root)
links = json.loads(Path(args.links).read_text(encoding="utf-8"))
clusters = [c for c in links["clusters"] if args.clusters is None or c["id"] in set(args.clusters)]
jobs = sorted({m["job"] for c in links["clusters"] for m in c["members"]})


def spread(items, n):
    return items if len(items) <= n else [items[round(i * (len(items) - 1) / (n - 1))] for i in range(n)]


loose = []
for j in jobs:
    for d in (root / j).iterdir():
        if d.is_dir() and d.name.startswith("_仕分け不能"):
            loose += sorted(d.rglob("*.png"))
print(f"unassigned crops {len(loose)} in {len(jobs)} videos; characters {len(clusters)}", flush=True)
refs = []
for c in clusters:
    pngs = sorted(p for m in c["members"] for p in (root / m["job"] / m["folder"]).rglob("*.png"))
    refs.append(spread(pngs, args.per_cluster))
import torch  # noqa: E402

cc = Ccip(0 if torch.cuda.is_available() else None)
F_loose = cc.feat(loose)
F_refs = cc.feat([p for r in refs for p in r])
D = cc.dist(np.concatenate([F_loose, F_refs]))[: len(loose), len(loose):]  # loose x refs
score = np.zeros((len(loose), len(clusters)))
at = 0
for k, r in enumerate(refs):
    score[:, k] = np.median(D[:, at:at + len(r)], axis=1) if r else 9.0
    at += len(r)
best = score.argmin(1) if len(clusters) else np.zeros(len(loose), int)
out = {}
for k, c in enumerate(clusters):
    idx = [i for i in np.argsort(score[:, k]) if best[i] == k and score[i, k] <= args.max_dist][: args.limit]
    out[str(c["id"])] = [{"file": loose[i].relative_to(root).as_posix(), "dist": round(float(score[i, k]), 4),
                          "strength": "strong" if score[i, k] <= THRESHOLD else "weak"} for i in idx]
Path(args.out).write_text(json.dumps({"threshold": THRESHOLD, "max_dist": args.max_dist, "unassigned_total": len(loose), "clusters": out}, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k: len(v) for k, v in out.items()}), flush=True)
