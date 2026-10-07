"""Link the character groups of SEVERAL video-dataset jobs into "same character" clusters, offline.

Run with the ComfyUI embedded Python (same environment as video_to_dataset.py):

    python link_characters.py OUT.json JOBS_ROOT JOB_ID [JOB_ID ...] [--q 0.90] [--min-images 3]

What it does (no human, no name tags):
  1. Every character group (a ``char_NN_*`` folder of a job) gets a prototype = normalised mean CLIP embedding of its saved crops,
     and an identity tag profile (hair / eye colour and style tags from the WD14 captions that were written next to each crop).
  2. Two groups are "in conflict" when they contain crops of the SAME frame of the SAME video: two people in one frame are different
     people, so they must never end up in one cluster.
  3. pair score = z(cosine of prototypes) + z(identity-tag similarity); the merge bar is a quantile of the scores of the CONFLICT pairs
     (known different people). Average-linkage merging, conflicts are never merged.
  4. It prints measurements that do not need a human: merged conflicts (must be 0), hair/eye agreement inside clusters, and a split-half
     test (half of a group's crops must link back to the other half at the same bar).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

HAIR = re.compile(r"^(?!.*(ribbon|ornament|accessory|clip|band|bow)).*\bhair$|^(twintails|ponytail|side ponytail|braid|twin braids|ahoge|drill hair|short hair|long hair|medium hair|very long hair)$")
EYES = re.compile(r"\beyes$")
IDENT_STYLE = {"twintails", "ponytail", "side ponytail", "braid", "twin braids", "ahoge", "drill hair", "short hair", "long hair", "medium hair", "very long hair", "bangs"}
FRAME = re.compile(r"_s(\d+)_(\d+)m(\d+)s_")

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("jobs_root")
ap.add_argument("jobs", nargs="+")
ap.add_argument("--q", type=float, default=0.90, help="merge bar = this quantile of the scores of known different people (same frame)")
ap.add_argument("--min-images", type=int, default=3)
ap.add_argument("--no-ccip", action="store_true", help="skip the CCIP score term")
ap.add_argument("--ccip-per-group", type=int, default=12)
ap.add_argument("--per-group", type=int, default=40, help="max crops embedded per group (evenly spread)")
args = ap.parse_args()
ROOT = Path(args.jobs_root)


def read_tags(png: Path) -> list[str]:
    t = png.with_suffix(".txt")
    if not t.is_file():
        return []
    return [x.strip() for x in t.read_text(encoding="utf-8").split(",") if x.strip() and not re.fullmatch(r"ch\d+(_o\d+)?", x.strip())]


# ---------------------------------------------------------------- collect groups
groups: list[dict] = []
for job in args.jobs:
    jdir = ROOT / job
    meta = json.loads((jdir / "job.json").read_text(encoding="utf-8")) if (jdir / "job.json").is_file() else {}
    video = Path(str(meta.get("video") or job)).stem
    for cdir in sorted(jdir.glob("char_*")):
        pngs = sorted(cdir.rglob("*.png"))
        if len(pngs) < args.min_images:
            continue
        frames = set()
        for p in pngs:
            m = FRAME.search(p.name)
            if m:
                frames.add((job, m.group(1), m.group(2) + ":" + m.group(3)))
        groups.append({"job": job, "video": video, "folder": cdir.name, "dir": cdir, "pngs": pngs, "frames": frames,
                       "outfits": sorted({p.parent.name for p in pngs})})
print(f"groups to link: {len(groups)} from {len(args.jobs)} jobs", flush=True)
if len(groups) < 2:
    Path(args.out).write_text(json.dumps({"clusters": [], "metrics": {"groups": len(groups)}}, ensure_ascii=False), encoding="utf-8")
    sys.exit(0)

# ---------------------------------------------------------------- CLIP prototypes
import torch
from transformers import CLIPModel

clip_dir = next((Path.home() / ".cache/huggingface/hub/models--sentence-transformers--clip-ViT-B-32/snapshots").iterdir()) / "0_CLIPModel"
dev = "cuda" if torch.cuda.is_available() else "cpu"
clip = CLIPModel.from_pretrained(str(clip_dir)).to(dev).eval()
if dev == "cuda":
    clip = clip.half()
MEAN, STD = np.array([0.4815, 0.4578, 0.4082], np.float32), np.array([0.2686, 0.2613, 0.2758], np.float32)


def letterbox(img: Image.Image, size: int = 224) -> Image.Image:
    img = img.convert("RGB")
    s = size / max(img.size)
    img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.BICUBIC)
    canvas = Image.new("RGB", (size, size), (128, 128, 128))
    canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
    return canvas


@torch.no_grad()
def embed(paths: list[Path]) -> np.ndarray:
    out = []
    for i in range(0, len(paths), 48):
        x = np.stack([(np.asarray(letterbox(Image.open(p)), np.float32) / 255.0 - MEAN) / STD for p in paths[i:i + 48]]).transpose(0, 3, 1, 2)
        t = torch.from_numpy(x).to(dev)
        v = clip.get_image_features(pixel_values=t.half() if dev == "cuda" else t)
        v = v if torch.is_tensor(v) else v.pooler_output
        v = torch.nn.functional.normalize(v.float(), dim=-1)
        out.append(v.cpu().numpy())
    return np.concatenate(out)


def spread(items: list, n: int) -> list:
    return items if len(items) <= n else [items[round(i * (len(items) - 1) / (n - 1))] for i in range(n)]


for gi, g in enumerate(groups):
    sample = spread(g["pngs"], args.per_group)
    g["sample"] = sample
    g["emb"] = embed(sample)
    g["proto"] = g["emb"].mean(0)
    g["proto"] /= np.linalg.norm(g["proto"]) + 1e-9
    tags = Counter(t for p in g["pngs"] for t in set(read_tags(p)))
    n = len(g["pngs"])
    g["hair"] = next((t for t, _ in tags.most_common() if t.endswith(" hair") and t.split()[0] in {"blonde", "brown", "black", "blue", "pink", "purple", "green", "red", "orange", "white", "grey", "silver", "aqua", "multicolored", "light"} or t.endswith(" hair") and t.count(" ") == 1), "")
    g["eyes"] = next((t for t, _ in tags.most_common() if t.endswith(" eyes")), "")
    g["ident"] = {t: c / n for t, c in tags.items() if (t.endswith(" hair") or t.endswith(" eyes") or t in IDENT_STYLE) and c / n >= 0.25}
    if gi % 20 == 0:
        print(f"embedded {gi}/{len(groups)}", flush=True)

# ---------------------------------------------------------------- pair scores
k = len(groups)
P = np.stack([g["proto"] for g in groups])
COS = P @ P.T


def ident_sim(a: dict, b: dict) -> float:
    keys = set(a["ident"]) | set(b["ident"])
    if not keys:
        return 0.0
    num = sum(min(a["ident"].get(t, 0), b["ident"].get(t, 0)) for t in keys)
    den = sum(max(a["ident"].get(t, 0), b["ident"].get(t, 0)) for t in keys)
    return num / den if den else 0.0


TAG = np.zeros((k, k))
conflict = np.zeros((k, k), bool)
for a in range(k):
    for b in range(a + 1, k):
        TAG[a, b] = TAG[b, a] = ident_sim(groups[a], groups[b])
        if groups[a]["job"] == groups[b]["job"] and groups[a]["frames"] & groups[b]["frames"]:
            conflict[a, b] = conflict[b, a] = True
iu = np.triu_indices(k, 1)


def z(M: np.ndarray) -> np.ndarray:
    v = M[iu]
    return (M - v.mean()) / (v.std() + 1e-9)


SCORE = z(COS) + z(TAG)
if not args.no_ccip:
    # CCIP (anime character-identity model): prototype = mean CCIP feature of a few crops; benchmark AUC 0.907 vs 0.856 for CLIP+tags
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from ccip import Ccip

        cc = Ccip(0 if torch.cuda.is_available() else None)
        feats = np.stack([cc.feat(spread(g["sample"], args.ccip_per_group)).mean(0) for g in groups])
        CCIPD = cc.dist(feats)
        CC = -CCIPD
        np.fill_diagonal(CC, CC[iu].mean())
        SCORE = z(COS) + z(TAG) + z(CC)
        print("CCIP term added", flush=True)
    except Exception as exc:  # keep linking usable without the model
        print(f"CCIP unavailable, using CLIP+tags only: {exc}", flush=True)
neg = SCORE[iu][conflict[iu]]
bar = float(np.quantile(neg, args.q)) if len(neg) >= 10 else float(np.quantile(SCORE[iu], 0.97))
print(f"conflict pairs {len(neg)}  merge bar {bar:.2f}", flush=True)

# ---------------------------------------------------------------- average-linkage merge, conflicts never merge
members = [[i] for i in range(k)]
alive = np.ones(k, bool)
C = SCORE.copy(); np.fill_diagonal(C, -1e9)
conf = conflict.copy()
sizes = np.array([len(g["pngs"]) for g in groups], float)
while True:
    M = np.where(alive[:, None] & alive[None, :] & ~conf, C, -1e9)
    np.fill_diagonal(M, -1e9)
    a, b = divmod(int(M.argmax()), k)
    if M[a, b] < bar:
        break
    row = (sizes[a] * C[a] + sizes[b] * C[b]) / (sizes[a] + sizes[b])
    C[a, :] = row; C[:, a] = row; C[a, a] = -1e9
    conf[a, :] |= conf[b, :]; conf[:, a] |= conf[:, b]
    members[a] += members[b]; sizes[a] += sizes[b]; alive[b] = False

clusters = []
merged_conflicts = 0
agree = []
for ci, mem in enumerate([m for m, ok in zip(members, alive) if ok]):
    for x in mem:
        for y in mem:
            if x < y and conflict[x, y]:
                merged_conflicts += 1
    hairs = Counter(groups[i]["hair"] for i in mem if groups[i]["hair"])
    eyes = Counter(groups[i]["eyes"] for i in mem if groups[i]["eyes"])
    tot = sum(len(groups[i]["pngs"]) for i in mem)
    if len(mem) > 1 and hairs:
        agree.append(hairs.most_common(1)[0][1] / sum(hairs.values()))
    clusters.append({"id": ci, "groups": len(mem), "images": tot, "hair": hairs.most_common(1)[0][0] if hairs else "", "eyes": eyes.most_common(1)[0][0] if eyes else "",
                     "members": [{"job": groups[i]["job"], "video": groups[i]["video"], "folder": groups[i]["folder"], "images": len(groups[i]["pngs"]),
                                  "outfits": groups[i]["outfits"], "sample": [str(p.relative_to(ROOT)).replace("\\", "/") for p in groups[i]["sample"][:4]]} for i in mem]})
clusters.sort(key=lambda c: -c["images"])

# ---------------------------------------------------------------- split-half recall proxy (needs no human)
rng = np.random.default_rng(0)
hits = tries = 0
for g in groups:
    if len(g["emb"]) < 8:
        continue
    idx = rng.permutation(len(g["emb"]))
    A, B = g["emb"][idx[: len(idx) // 2]].mean(0), g["emb"][idx[len(idx) // 2:]].mean(0)
    A /= np.linalg.norm(A) + 1e-9; B /= np.linalg.norm(B) + 1e-9
    # same score recipe: cosine z-scored against the cosine distribution of all group pairs; tags are identical for both halves
    zc = (float(A @ B) - COS[iu].mean()) / (COS[iu].std() + 1e-9)
    tries += 1
    hits += (zc + (1.0 - TAG[iu].mean()) / (TAG[iu].std() + 1e-9)) >= bar
metrics = {"groups": k, "jobs": len(args.jobs), "clusters": len(clusters), "multi_group_clusters": sum(c["groups"] > 1 for c in clusters),
           "conflict_pairs_known_different_people": int(len(neg)), "merge_bar": round(bar, 3), "merged_conflicts_must_be_0": merged_conflicts,
           "hair_agreement_inside_merged_clusters": round(float(np.mean(agree)), 3) if agree else None,
           "split_half_relink_rate": round(hits / tries, 3) if tries else None, "split_half_groups_tested": tries}
Path(args.out).write_text(json.dumps({"clusters": clusters, "metrics": metrics}, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(metrics, ensure_ascii=False), flush=True)
