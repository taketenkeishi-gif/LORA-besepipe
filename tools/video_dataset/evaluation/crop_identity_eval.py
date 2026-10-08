"""Group-level linking (current) vs crop-level CCIP clustering (proposed), on Nanoha with real labels.

python crop_identity_eval.py EXP_ROOT CURATED_DIR OUT.json JOB [JOB ...]
Positives: crops matching the curated Nanoha set (perceptual hash).  Sure negatives: crops whose WD14 tags name a hair colour
that is not brown/orange and do not say brown hair (Nanoha has brown hair).
For the cluster that holds most positives: recall = positives inside / all positives; contamination = sure negatives inside / size.
"""
import json, os, re, subprocess, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

exp, curated, out_json, jobs = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4:]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ccip import Ccip  # noqa: E402

HAIR = re.compile(r"^(blonde|black|blue|pink|purple|green|white|grey|gray|silver|aqua|red|light blue|dark blue|light purple|light brown|brown|orange) hair$")


def phash(p):
    g = np.asarray(Image.open(p).convert("L").resize((17, 16), Image.BILINEAR), np.float32)
    return int("".join("1" if b else "0" for b in (g[:, 1:] > g[:, :-1]).flatten()), 2)


cur = [phash(p) for p in curated.rglob("*.png")]
crops = []  # (path, job, group or None, frame key)
for j in jobs:
    for p in sorted((exp / j).rglob("*.png")):
        rel = p.relative_to(exp / j).parts
        if rel[0].startswith("char_"):
            grp = f"{j}/{rel[0]}"
        elif rel[0].startswith("_仕分け不能"):
            grp = None
        else:
            continue
        m = re.search(r"_s(\d+)_(\d+)m(\d+)s_", p.name)
        frame = f"{j}:{m.group(0)}" if m else None
        crops.append((p, j, grp, frame))
print("crops", len(crops), flush=True)


def tags(p):
    t = p.with_suffix(".txt")
    return [x.strip() for x in t.read_text(encoding="utf-8").split(",")] if t.is_file() else []


hashes = [phash(p) for p, *_ in crops]  # once per crop (inside the generator it was recomputed 304x per crop)
pos = np.array([any(bin(h ^ c).count("1") <= 12 for c in cur) for h in hashes])
neg = []
for p, *_ in crops:
    hs = {t for t in tags(p) if HAIR.match(t)}
    neg.append(bool(hs) and not ({"brown hair", "light brown hair", "orange hair"} & hs))
neg = np.array(neg)
print("positives", int(pos.sum()), "sure negatives", int(neg.sum()), flush=True)


def score(labels):
    """labels: cluster id per crop (-1 = none)."""
    ids = Counter(l for l, p in zip(labels, pos) if p and l != -1)
    if not ids:
        return {"recall": 0.0}
    best = ids.most_common(1)[0][0]
    inside = np.array([l == best for l in labels])
    return {"recall": round(float((inside & pos).sum() / pos.sum()), 3), "size": int(inside.sum()),
            "contamination_sure_neg": round(float((inside & neg).sum() / inside.sum()), 3),
            "positives_split_over_clusters": len(ids), "second_cluster_positives": ids.most_common(2)[1][1] if len(ids) > 1 else 0}


res = {"crops": len(crops), "positives": int(pos.sum()), "sure_negatives": int(neg.sum()), "methods": {}}

# ---- current: per-video groups linked by link_characters.py (unassigned crops belong to nothing)
py = r"C:\Users\Keishi\Portfolio\Toolchains-Physical\ComfyUI-Portable\ComfyUI_windows_portable\python_embeded\python.exe"
tmp = Path(os.environ.get("TEMP", ".")) / "crop_eval_links.json"
env = os.environ.copy(); env.update({"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"})
r = subprocess.run([py, "-X", "utf8", str(Path(__file__).resolve().parents[1] / "link_characters.py"), str(tmp), str(exp), *jobs, "--q", "0.75"],
                   env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=3000)
if r.returncode:
    sys.exit(r.stderr[-800:])
links = json.loads(tmp.read_text(encoding="utf-8"))
gid = {f"{m['job']}/{m['folder']}": c["id"] for c in links["clusters"] for m in c["members"]}
res["methods"]["current (groups + tags/CLIP/CCIP link)"] = score([gid.get(g, -1) if g else -1 for _, _, g, _ in crops])

# ---- proposed: every crop, CCIP distance only, same frame = different people
import torch  # noqa: E402
from scipy.cluster.hierarchy import fcluster, linkage  # noqa: E402
from scipy.spatial.distance import squareform  # noqa: E402

cc = Ccip(0 if torch.cuda.is_available() else None)
F = cc.feat([p for p, *_ in crops])
n = len(F)
D = np.zeros((n, n), np.float32)
B = 1024
for a in range(0, n, B):
    for b in range(a, n, B):
        blk = cc.dist(np.concatenate([F[a:a + B], F[b:b + B]]))
        la = min(B, n - a)
        D[a:a + la, b:b + min(B, n - b)] = blk[:la, la:]
D = np.maximum(D, D.T); np.fill_diagonal(D, 0)
frames = defaultdict(list)
for i, (*_, f) in enumerate(crops):
    if f:
        frames[f].append(i)
for idx in frames.values():  # two people in one frame are never the same character
    for x in idx:
        for y in idx:
            if x != y:
                D[x, y] = 10.0
Z = linkage(squareform(D, checks=False), method="average")
for t in (0.12, 0.15, 0.178, 0.21):
    res["methods"][f"proposed: crop CCIP, average linkage <= {t}"] = score(list(fcluster(Z, t, criterion="distance")))
Path(out_json).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=1))
