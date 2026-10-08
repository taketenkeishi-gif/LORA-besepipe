"""Characters across videos with a confidence split: a few sure characters, the rest in one 'pending' pile.

python crop_cluster.py FEAT_PREFIX OUT.json [--t 0.178] [--min-size 20] [--assign 0.17] [--margin 0.02] [--report]
FEAT_PREFIX = output of crop_features.py (.npy features + .json items).  The full CCIP distance matrix is cached as FEAT_PREFIX.dist.npy.

1. Distance = CCIP, plus a penalty when two crops' tags disagree on what the character IS (person vs creature/mascot,
   girl vs boy).  Two people in one frame are never merged.
2. Average-linkage clusters cut at --t.  A cluster is a SURE character when it is big (--min-size), mostly shows a face,
   and is consistent (person/creature, gender, hair colour).
3. Every other crop joins its nearest sure character only when it is clearly closer to it than to any other
   (mean of its k nearest members < --assign, and better than the runner-up by --margin) and does not contradict it.
   Everything else goes to 'pending' - one pile, reviewed last.
"""
import argparse, json, sys, time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

ap = argparse.ArgumentParser()
ap.add_argument("feat"); ap.add_argument("out")
ap.add_argument("--t", type=float, default=0.178)
ap.add_argument("--min-size", type=int, default=20)
ap.add_argument("--face", type=float, default=0.35)
ap.add_argument("--purity", type=float, default=0.75)
ap.add_argument("--assign", type=float, default=0.17)
ap.add_argument("--margin", type=float, default=0.02)
ap.add_argument("--k", type=int, default=8)
ap.add_argument("--core-quantile", type=float, default=0.9)
ap.add_argument("--merge", type=float, default=0.20)
ap.add_argument("--main-share", type=float, default=0.04)
ap.add_argument("--kind-penalty", type=float, default=0.3)  # 0.3: girl/boy mix 0.6% -> 0.0% at the same pending share (2026-10-09 sweep)
ap.add_argument("--report", action="store_true")
args = ap.parse_args()
prefix = Path(args.feat)
F = np.load(prefix.with_suffix(".npy"))
items = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
n = len(items)
t0 = time.time()

COLOURS = {"blonde", "black", "blue", "pink", "purple", "green", "white", "grey", "gray", "silver", "aqua", "red", "brown", "orange",
           "light blue", "dark blue", "light purple", "light brown"}
NONHUMAN = {"no humans", "creature", "pokemon (creature)", "furry", "animal", "mascot", "robot", "stuffed toy"}


def facts(it):
    tags = set(it["tags"])
    kind = "creature" if tags & NONHUMAN and "1girl" not in tags and "1boy" not in tags else "person"
    gender = "boy" if ("1boy" in tags or "male focus" in tags) and "1girl" not in tags else "girl" if "1girl" in tags else ""
    hair = sorted(t[:-5] for t in tags if t.endswith(" hair") and t[:-5] in COLOURS)
    length = next((L for L in ("very long", "long", "medium", "short") if f"{L} hair" in tags), "")
    face = it["view"].split("-")[-1] in ("close", "upper", "full")
    return {"kind": kind, "gender": gender, "hair": hair, "length": length, "face": face}


A = [facts(it) for it in items]

dist_path = prefix.with_name(prefix.name + ".dist.npy")
if dist_path.is_file():  # only when someone placed one; not written by default (n*n floats, e.g. 540 MB, and it takes seconds)
    D = np.array(np.load(dist_path, mmap_mode="r"))
else:
    from ccip import Ccip  # noqa: E402

    cc = Ccip(None)
    D = np.zeros((n, n), np.float32)
    B = 1024
    for a in range(0, n, B):
        for b in range(a, n, B):
            blk = cc.dist(np.concatenate([F[a:a + B], F[b:b + B]]))
            la = min(B, n - a)
            D[a:a + la, b:b + B] = blk[:la, la:]
            D[b:b + B, a:a + la] = blk[:la, la:].T
        print(f"distance rows {min(a + B, n)}/{n} {time.time() - t0:.0f}s", flush=True)
    np.fill_diagonal(D, 0)
print(f"distances ready {time.time() - t0:.0f}s", flush=True)

# what-it-is penalty: person vs creature, girl vs boy (only when both crops say so)
kind = np.array([a["kind"] == "creature" for a in A])
gen = np.array([{"girl": 1, "boy": 2}.get(a["gender"], 0) for a in A])
P = D.copy()
if args.kind_penalty:
    P += args.kind_penalty * (kind[:, None] != kind[None, :])
    P += args.kind_penalty * ((gen[:, None] * gen[None, :] > 0) & (gen[:, None] != gen[None, :]))
frames = defaultdict(list)
for i, it in enumerate(items):
    if it["frame"]:
        frames[it["frame"]].append(i)
for idx in frames.values():
    if len(idx) > 1:
        ix = np.array(idx)
        P[np.ix_(ix, ix)] = 10.0
np.fill_diagonal(P, 0)

from scipy.cluster.hierarchy import fcluster, linkage  # noqa: E402
from scipy.spatial.distance import squareform  # noqa: E402

Z = linkage(squareform(P, checks=False), method="average")
labels = fcluster(Z, args.t, criterion="distance")
groups = defaultdict(list)
for i, l in enumerate(labels):
    groups[int(l)].append(i)
print(f"clustered {len(groups)} {time.time() - t0:.0f}s", flush=True)


def profile(idx):
    k = Counter(A[i]["kind"] for i in idx)
    g = Counter(A[i]["gender"] for i in idx if A[i]["kind"] == "person" and A[i]["gender"])
    h = Counter(c for i in idx for c in set(A[i]["hair"]))
    tagged = sum(1 for i in idx if A[i]["hair"])
    top_h, nh = h.most_common(1)[0] if h else ("", 0)
    eyes = Counter(t for i in idx for t in items[i]["tags"] if t.endswith(" eyes") and t[:-5] in COLOURS)
    length = Counter(A[i]["length"] for i in idx if A[i]["length"])
    return {"kind": k.most_common(1)[0][0], "kind_share": k.most_common(1)[0][1] / len(idx),
            "gender": g.most_common(1)[0][0] if g else "", "gender_share": (g.most_common(1)[0][1] / sum(g.values())) if g else 1.0,
            "hair": top_h, "hair_share": nh / tagged if tagged else 0.0,
            "eyes": eyes.most_common(1)[0][0][:-5] if eyes else "", "eyes_share": eyes.most_common(1)[0][1] / len(idx) if eyes else 0.0,
            "length": length.most_common(1)[0][0] if length else "",
            "face": sum(A[i]["face"] for i in idx) / len(idx)}


sure = []
for gid, idx in groups.items():
    if len(idx) < args.min_size:
        continue
    p = profile(idx)
    if p["face"] >= args.face and p["kind_share"] >= args.purity and p["gender_share"] >= args.purity and \
            (p["kind"] == "creature" or p["hair_share"] >= 0.6):
        sure.append(idx)
# One character is often split by shot type (back views, eye close-ups, items, dark scenes): CCIP is unsure on partial views.
# Sure groups are merged when they describe the same kind of character (person/creature, gender, hair colour, eye colour
# not contradicting) and are close (median CCIP distance between their images < --merge).  Different-looking groups never merge
# on distance alone: in the 2026-10-09 set, pairs of different hair/gender were as close as 0.128.
def signature(p):
    return p["kind"], p["gender"], p["hair"], p["eyes"] if p["eyes_share"] >= 0.3 else ""


def compatible(a, b):
    return a[0] == b[0] and (not a[1] or not b[1] or a[1] == b[1]) and a[2] == b[2] and (not a[3] or not b[3] or a[3] == b[3])


rng = np.random.default_rng(0)
samples = [rng.choice(idx, min(40, len(idx)), replace=False) for idx in sure]
sigs = [signature(profile(idx)) for idx in sure]
parent = list(range(len(sure)))


def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


pairs = []
for a in range(len(sure)):
    for b in range(a + 1, len(sure)):
        if compatible(sigs[a], sigs[b]):
            pairs.append((float(np.median(P[np.ix_(samples[a], samples[b])])), a, b))
for d, a, b in sorted(pairs):
    if d >= args.merge:
        break
    ra, rb = find(a), find(b)
    if ra != rb and compatible(sigs[ra], sigs[rb]):
        parent[rb] = ra
merged = defaultdict(list)
for c, idx in enumerate(sure):
    merged[find(c)] += list(idx)
print(f"sure groups {len(sure)} -> {len(merged)} after merging same-looking splits", flush=True)
sure = list(merged.values())

in_sure = np.full(n, -1)
for c, idx in enumerate(sure):
    in_sure[idx] = c
members = [np.array(idx) for idx in sure]
profiles = [profile(idx) for idx in sure]

# attach the rest: mean of the k nearest members of each sure character.  A crop may join only if it is as close to the
# character as the character's own core images are to each other (their own k-nearest score, --core-quantile).
pending = []
added = defaultdict(list)
rest = np.where(in_sure < 0)[0]
reach = []
for m in members:
    sub = np.sort(P[np.ix_(m, m)], axis=1)[:, 1:args.k + 1].mean(1)
    reach.append(float(np.quantile(sub, args.core_quantile)))
for i in rest:
    if not len(members):
        pending.append((0, 0.0, int(i))); continue
    scores = np.array([np.sort(P[i, m])[:args.k].mean() for m in members])
    order = np.argsort(scores)
    best, second = scores[order[0]], scores[order[1]] if len(order) > 1 else 9.0
    c = int(order[0])
    ok = best < min(args.assign, reach[c]) and second - best >= args.margin
    pc = profiles[c]
    if ok and A[i]["kind"] != pc["kind"]:
        ok = False
    if ok and pc["kind"] == "person" and A[i]["gender"] and pc["gender"] and A[i]["gender"] != pc["gender"]:
        ok = False
    if ok and pc["kind"] == "person" and A[i]["hair"] and pc["hair"] and pc["hair"] not in A[i]["hair"]:
        ok = False
    if ok:
        added[c].append(int(i))
    else:
        pending.append((c, float(best), int(i)))

chars = []
for c, idx in enumerate(sure):
    allidx = list(idx) + added[c]
    p = profile(allidx)
    chars.append({"sure": True, "core": len(idx), "attached": len(added[c]), "images": len(allidx),
                  "videos": len({items[i]["job"] for i in allidx}), **p,
                  "files": [items[i]["rel"] for i in sorted(allidx, key=lambda i: items[i]["rel"])]})
order = sorted(range(len(chars)), key=lambda c: -chars[c]["images"])
chars = [chars[c] for c in order]
rank = {c: r for r, c in enumerate(order)}
# Confidence: a character is shown on its own only when it holds at least --main-share of all sure images.  Smaller sure groups
# (often a main character's back views or close-ups, or a minor character) go to pending as contiguous blocks, so one drag moves them.
total_sure = sum(c["images"] for c in chars) or 1
main = [c for c in chars if c["images"] / total_sure >= args.main_share]
minor = [c for c in chars if c["images"] / total_sure < args.main_share]
# pending: the minor groups first (largest first, each kept together), then the rest by nearest character, closest first
pending.sort(key=lambda p: (rank.get(p[0], 0), p[1]))
files = [f for c in minor for f in c["files"]] + [items[i]["rel"] for _c, _s, i in pending]
result = {"params": vars(args), "crops": n, "characters": main,
          "pending": {"images": len(files), "files": files, "minor_groups": [len(c["files"]) for c in minor]}}
pending = [i for _c, _s, i in pending]
Path(args.out).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")

if args.report:
    # consistency inside each sure character (person/creature, gender, hair colour) weighted by size
    def impurity(key, share):
        tot = sum(c["images"] for c in main)
        return round(sum((1 - c[share]) * c["images"] for c in main) / tot, 3) if tot else 0

    print(json.dumps({"sure_characters": len(main), "images_in_sure": sum(c["images"] for c in main),
                      "pending": len(files), "pending_share": round(len(files) / n, 3), "minor_groups_in_pending": len(minor),
                      "mix_kind": impurity("kind", "kind_share"), "mix_gender": impurity("gender", "gender_share"),
                      "mix_hair": impurity("hair", "hair_share"),
                      "sizes": [c["images"] for c in main][:30]}, ensure_ascii=False), flush=True)
print(f"done {time.time() - t0:.0f}s", flush=True)
