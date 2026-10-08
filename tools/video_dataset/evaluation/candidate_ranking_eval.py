"""How well a candidate score orders right above wrong, measured without the user.

python candidate_ranking_eval.py FEAT_PREFIX RESULT.json
Positives: no-face images (back views, parts, far shots - what the pending pile holds) of the main characters, taken out with
their whole scene: neither CCIP nor CLIP may compare them with any image of that scene.  Negatives: images of the "other
character" sections (sure groups that fit no main character), scored against their nearest main character.
Reported per score: AUC, and how many positives are accepted before negatives reach 5 % / 10 % of the accepted.
Caveat: the positives were put in the main characters by the CCIP clustering, so CCIP-only numbers are optimistic; the
comparison between scores is the point.
"""
import json, re, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ccip import Ccip  # noqa: E402

prefix, res = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
items = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
F = np.load(prefix.with_suffix(".npy"))
C = np.load(prefix.with_name(prefix.name + ".clip.npy")).astype(np.float32)
idx = {it["rel"]: i for i, it in enumerate(items)}
face = lambda i: items[i]["view"].split("-")[-1] in ("close", "upper", "full")
scene = lambda i: items[i]["job"] + ":" + (re.search(r"_s(\d+)_", items[i]["rel"]).group(1) if re.search(r"_s(\d+)_", items[i]["rel"]) else items[i]["rel"])
mains = [np.array([idx[f] for f in c["files"]]) for c in res["characters"]]
owner = {int(i): c for c, m in enumerate(mains) for i in m}
rng = np.random.default_rng(0)
held, out_scenes = [], set()
for c, m in enumerate(mains):
    sc = sorted({scene(int(i)) for i in m if not face(int(i))})
    pick = set(rng.choice(sc, min(50, len(sc)), replace=False).tolist())
    out_scenes |= pick
    held += [int(i) for i in m if not face(int(i)) and scene(int(i)) in pick]
ref = [np.array([i for i in m if scene(int(i)) not in out_scenes]) for m in mains]
neg = [idx[f] for s in res["sections"] if s["section"] == "other" for f in s["files"]]
neg = rng.choice(neg, min(800, len(neg)), replace=False).tolist()
scene_faces = defaultdict(list)
for c, m in enumerate(mains):
    for i in m:
        if face(int(i)):
            scene_faces[scene(int(i))].append(c)
cc = Ccip(None)


def knn_ccip(q):
    out = np.zeros((len(q), len(ref)))
    for a in range(0, len(q), 400):
        p = q[a:a + 400]
        for c, m in enumerate(ref):
            D = np.zeros((len(p), len(m)), np.float32)
            for b in range(0, len(m), 1500):
                blk = cc.dist(np.concatenate([F[p], F[m[b:b + 1500]]]))
                D[:, b:b + 1500] = blk[:len(p), len(p):]
            out[a:a + len(p), c] = np.sort(D, 1)[:, :8].mean(1)
    return out


def knn_clip(q):  # mean of the 8 highest cosine similarities to each main character
    S = C[q] @ np.concatenate([C[m] for m in ref]).T
    off = np.cumsum([0] + [len(m) for m in ref])
    return np.stack([np.sort(S[:, off[c]:off[c + 1]], 1)[:, -8:].mean(1) for c in range(len(ref))], 1)


def vote(i):
    m = scene_faces.get(scene(i), [])
    if not m:
        return -2
    c, k = Counter(m).most_common(1)[0]
    return c if k / len(m) >= 0.8 else -1


q = np.array(held + neg)
truth = np.array([owner[i] for i in held] + [-9] * len(neg))  # negatives belong to no main character
K, L, V = knn_ccip(q), knn_clip(q), np.array([vote(int(i)) for i in q])
c_star = K.argmin(1)  # candidate character = CCIP nearest (as in the pending piles)
rows = np.arange(len(q))
best = K[rows, c_star]
other = np.where(np.eye(len(ref), dtype=bool)[c_star], 9, K).min(1)
lead = np.maximum(0, other - best)
sv = np.where(V == c_star, -0.03, np.where(V >= 0, 0.03, 0.0))
clip_c = L[rows, c_star]
clip_other = np.where(np.eye(len(ref), dtype=bool)[c_star], -9, L).max(1)
y = (truth == c_star).astype(int)  # right only when the candidate character is the true one
scores = {"A CCIP closeness": best, "B current (CCIP+lead+scene)": best - 0.5 * lead + sv}
for w in (0.5, 1.0, 2.0):
    scores[f"C current + CLIP x{w}"] = best - 0.5 * lead + sv - w * (clip_c - clip_other)


def auc(s, y):
    order = np.argsort(s)
    r = np.empty(len(s)); r[order] = np.arange(len(s))
    pos = r[y == 1]
    return 1 - (pos.sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * (len(y) - len(pos)))


print(f"positives {len(held)} (right candidate {int(y[:len(held)].sum())}), negatives {len(neg)}")
for name, s in scores.items():
    o = np.argsort(s); yy = y[o]
    cum_r, cum_w = np.cumsum(yy), np.cumsum(1 - yy)
    at = {}
    for tol in (0.05, 0.10):
        ok = np.where(cum_w <= tol * np.arange(1, len(yy) + 1))[0]
        at[tol] = int(cum_r[ok[-1]]) if len(ok) else 0
    print(f"{name:28s} AUC {auc(s, y):.3f}  right accepted at <=5% wrong {at[0.05]:5d}  at <=10% wrong {at[0.10]:5d}")
