"""Unknown-character setting: NO character names are used by the method.  The name tags (nanoha / fate / hayate) are only the ANSWER KEY.

Question 1: how well does each pairwise "same character?" score separate same-name from different-name pairs ACROSS scenes?
Question 2: when a clustering threshold is chosen WITHOUT labels (from same-frame different-person pairs), how many clusters per
            character come out (fragmentation) and how pure are they?
"""
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("GPU", "1")
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel

WORK = Path(sys.argv[1])
crops = json.loads((WORK / "crops.json").read_text(encoding="utf-8"))
n = len(crops)
for c in crops:
    c["k"] = int(re.search(r"_k(\d+)_", c["file"]).group(1))
NAMES = {"takamachi nanoha": "nanoha", "fate testarossa": "fate", "yagami hayate": "hayate"}
gt = []
for c in crops:
    f = {NAMES[t] for t in c["tags"] if t in NAMES}
    gt.append(next(iter(f)) if len(f) == 1 else None)  # ANSWER KEY ONLY
print("crops", n, "with answer", sum(g is not None for g in gt), Counter(g for g in gt if g), flush=True)

clip = CLIPModel.from_pretrained(str(next((Path.home() / ".cache/huggingface/hub/models--sentence-transformers--clip-ViT-B-32/snapshots").iterdir()) / "0_CLIPModel")).cuda().half().eval()
CM, CS = np.array([0.4815, 0.4578, 0.4082], np.float32), np.array([0.2686, 0.2613, 0.2758], np.float32)


def letterbox(img, size=224):
    img = img.convert("RGB")
    s = size / max(img.size)
    img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.BICUBIC)
    c = Image.new("RGB", (size, size), (128, 128, 128))
    c.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
    return c


@torch.no_grad()
def embed(images):
    x = np.stack([(np.asarray(letterbox(i), np.float32) / 255 - CM) / CS for i in images]).transpose(0, 3, 1, 2)
    v = clip.get_image_features(pixel_values=torch.from_numpy(x).cuda().half())
    v = v if torch.is_tensor(v) else v.pooler_output
    return torch.nn.functional.normalize(v.float(), dim=-1).cpu().numpy()


imgs = [Image.open(WORK / "crops" / c["iso_file"]) for c in crops]
heads = []
for im, c in zip(imgs, crops):
    w, h = im.size
    heads.append(im.crop((0, 0, w, int(h * 0.38))) if "head-cut" not in c["view"] and "lower-cut" not in c["view"] else im)
E = np.concatenate([embed(imgs[s:s + 32]) for s in range(0, n, 32)])
H = np.concatenate([embed(heads[s:s + 32]) for s in range(0, n, 32)])

HAIR = re.compile(r"^(blonde|brown|black|blue|red|pink|purple|green|white|silver|grey|gray|orange|aqua|light blue|dark blue|light brown|dark green|light purple|multicolored|two-tone|gradient|streaked|colored inner) hair$")
EYE = re.compile(r"^(?!closed|half-closed|wide|empty|crossed|one|heterochromia|torn|big|narrowed|rolling|looking)(.+) eyes$")
FEAT = {"animal ears", "twintails", "ponytail", "braid", "twin braids", "side ponytail", "short hair", "long hair", "medium hair", "very long hair", "ahoge", "hair bun", "glasses"}
ident = [{t: (3.0 if HAIR.match(t) else 2.0 if EYE.match(t) else 1.0) for t in c["tags"] if HAIR.match(t) or EYE.match(t) or t in FEAT} for c in crops]


def wj(a, b):
    k = set(a) | set(b)
    return sum(min(a.get(x, 0), b.get(x, 0)) for x in k) / sum(max(a.get(x, 0), b.get(x, 0)) for x in k) if k else 0.0


T = np.array([[wj(ident[i], ident[j]) for j in range(n)] for i in range(n)])
Cf, Ch = E @ E.T, H @ H.T
scene = np.array([c["scene"] for c in crops])
track = np.array([c["track"] for c in crops])
frame = np.array([(c["scene"], c["k"]) for c in crops], dtype=object)
cross = scene[:, None] != scene[None, :]
off = ~np.eye(n, dtype=bool)


def z(M):
    v = M[cross & off]
    return (M - v.mean()) / v.std()


Z = {"tags hair/eye": z(T), "CLIP full": z(Cf), "CLIP head": z(Ch)}
combos = {"tags": Z["tags hair/eye"], "CLIP full": Z["CLIP full"], "CLIP head": Z["CLIP head"],
          "tags + CLIP full": Z["tags hair/eye"] + Z["CLIP full"], "tags + CLIP head": Z["tags hair/eye"] + Z["CLIP head"],
          "tags + CLIP head + CLIP full": Z["tags hair/eye"] + Z["CLIP head"] + Z["CLIP full"], "CLIP head + CLIP full": Z["CLIP head"] + Z["CLIP full"]}
lab = np.array([g if g else "" for g in gt])
labelled = lab != ""
same_name = (lab[:, None] == lab[None, :]) & labelled[:, None] & labelled[None, :]
diff_name = (lab[:, None] != lab[None, :]) & labelled[:, None] & labelled[None, :]
P = np.triu(cross & same_name, 1)
N = np.triu(cross & diff_name, 1)


def auc(p, q):
    s = np.concatenate([p, q])
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    r = np.empty(len(s))
    r[s.argsort()] = np.arange(1, len(s) + 1)
    r = (np.bincount(inv, weights=r) / cnt)[inv]
    return float((r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(q)))


print("\nQ1  cross-scene same-vs-different-character AUC (answer key used only for scoring)", flush=True)
q1 = {}
for name, M in combos.items():
    q1[name] = round(auc(M[P], M[N]), 3)
    print(f"   {name:30s} AUC={q1[name]:.3f}", flush=True)


LINK = os.environ.get("LINK", "average")


def constrained_upgma(sim, members, conflict, thr):
    k = len(members)
    sizes = np.array([len(m) for m in members], float)
    C, conf, alive = sim.copy(), conflict.copy(), np.ones(k, bool)
    np.fill_diagonal(C, -1e9)
    while True:
        m = np.where(alive[:, None] & alive[None, :] & ~conf, C, -1e9)
        np.fill_diagonal(m, -1e9)
        a, b = divmod(int(m.argmax()), k)
        if m[a, b] < thr:
            break
        row = np.maximum(C[a], C[b]) if LINK == "single" else (sizes[a] * C[a] + sizes[b] * C[b]) / (sizes[a] + sizes[b])
        C[a, :] = row
        C[:, a] = row
        C[a, a] = -1e9
        conf[a, :] |= conf[b, :]
        conf[:, a] |= conf[:, b]
        members[a] = members[a] + members[b]
        sizes[a] += sizes[b]
        alive[b] = False
    return [m for m, ok in zip(members, alive) if ok]


by_track = defaultdict(list)
for i, c in enumerate(crops):
    by_track[c["track"]].append(i)
tids = list(by_track)
Mx = np.zeros((len(tids), n))
for ci, t in enumerate(tids):
    Mx[ci, by_track[t]] = 1
same_frame_diff_track = np.zeros((n, n), bool)
fr = defaultdict(list)
for i, c in enumerate(crops):
    fr[(c["scene"], c["k"])].append(i)
for m in fr.values():
    for a in m:
        for b in m:
            if track[a] != track[b]:
                same_frame_diff_track[a, b] = True
sz = Mx.sum(1)
tconf = (Mx @ same_frame_diff_track.astype(float) @ Mx.T) > 0

print("\nQ2  label-free clustering (threshold from same-frame different-person pairs); scored with the answer key", flush=True)
q2 = {}
for name in ("tags", "tags + CLIP full"):
    M = combos[name]
    negs = M[np.triu(same_frame_diff_track, 1)]
    for q in (0.50, 0.80, 0.90, 0.97):
        thr = float(np.quantile(negs, q))
        if LINK == "single":
            TS = np.full((len(tids), len(tids)), -1e9)
            for a in range(len(tids)):
                for b in range(len(tids)):
                    if a != b:
                        TS[a, b] = M[np.ix_(by_track[tids[a]], by_track[tids[b]])].max()
        else:
            TS = (Mx @ M @ Mx.T) / np.outer(sz, sz)
        groups = constrained_upgma(TS, [list(by_track[t]) for t in tids], tconf, thr)
        big = [g for g in groups if len(g) >= 4]
        pur_num = pur_den = 0
        per_name = defaultdict(set)
        for gi, g in enumerate(big):
            cnt = Counter(gt[i] for i in g if gt[i])
            if cnt:
                pur_num += cnt.most_common(1)[0][1]
                pur_den += sum(cnt.values())
                for nm in cnt:
                    if cnt[nm] >= 3:
                        per_name[nm].add(gi)
        row = {"clusters>=4": len(big), "covered_crops": sum(len(g) for g in big), "purity": round(pur_num / max(1, pur_den), 3), "clusters_per_name(>=3 crops)": {k: len(v) for k, v in per_name.items()}}
        q2[f"{name} @neg-quantile {q}"] = row
        print(f"   {name:30s} q={q:.2f} thr={thr:6.2f} -> {row}", flush=True)
(WORK.parent / "unsup_eval.json").write_text(json.dumps({"auc": q1, "clustering": q2}, ensure_ascii=False, indent=1), encoding="utf-8")
