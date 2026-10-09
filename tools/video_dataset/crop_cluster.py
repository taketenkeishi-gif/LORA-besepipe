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
ap.add_argument("--candidate", type=float, default=0.19)  # nearer than this to a main character -> its candidates (else 判定不能)
ap.add_argument("--kind-penalty", type=float, default=0.3)  # 0.3: girl/boy mix 0.6% -> 0.0% at the same pending share (2026-10-09 sweep)
ap.add_argument("--report", action="store_true")
ap.add_argument("--absorb", type=float, default=-9.0)  # candidate score at or below this joins the character (-9 = none)
ap.add_argument("--version", type=int, default=2)  # output format; sets made from an older one are rebuilt once (undoable)
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

# ---- everything not in a main character is sorted into sections, so pending is reviewed by kind, not as one pile ----
rel_i = {it["rel"]: i for i, it in enumerate(items)}
main_idx = [np.array([rel_i[f] for f in c["files"]]) for c in main]
owner = np.full(n, -1)
for c, m in enumerate(main_idx):
    owner[m] = c
rest = np.where(owner < 0)[0]
pos = {int(i): k for k, i in enumerate(rest)}
knn = np.stack([np.sort(P[np.ix_(rest, m)], axis=1)[:, :args.k].mean(1) for m in main_idx], 1) if main_idx else np.zeros((len(rest), 0))
MULTI = {"2girls", "3girls", "multiple girls", "2boys", "multiple boys"}


def is_multi(i):
    t = set(items[i]["tags"])
    return bool(t & MULTI) or ("1girl" in t and "1boy" in t)


def fits(a, p):  # crop (or group) facts a do not contradict main character profile p
    if a["kind"] != p["kind"]:
        return False
    if p["kind"] == "person" and a.get("gender") and p["gender"] and a["gender"] != p["gender"]:
        return False
    return True


# the per-video grouping made when each video was processed: a crop whose in-video group mates are mostly character X
votes = {}
groups_in_video = defaultdict(list)
for i, it in enumerate(items):
    if it["group"]:
        groups_in_video[(it["job"], it["group"])].append(i)
for g, mem in groups_in_video.items():
    cnt = Counter(int(owner[i]) for i in mem if owner[i] >= 0)
    if sum(cnt.values()) >= 5:
        top, k = cnt.most_common(1)[0]
        if k / sum(cnt.values()) >= 0.7:
            votes[g] = top

recovered = defaultdict(list)
cand_blocks = defaultdict(list)    # main -> [files of a minor group]
cand_single = defaultdict(list)    # main -> [(score, i)]
others, multi, unknown = [], [], []
taken = set()
# 1) minor sure groups: a block of candidates for the nearest main they fit, else a separate character
for mc in minor:
    mem = [rel_i[f] for f in mc["files"]]
    taken.update(mem)
    med = np.median(knn[[pos[i] for i in mem]], 0) if len(main_idx) else np.array([9.0])
    order_m = np.argsort(med)
    c = int(order_m[0])
    gp = {"kind": mc["kind"], "gender": mc["gender"]}
    if len(main_idx) and med[c] < args.candidate and fits(gp, main[c]):
        cand_blocks[c].append(mc["files"])
    else:
        others.append(mc)
# 2) single crops: recovered when two independent signals agree (in-video group vote AND nearest main), else sorted
for i in rest:
    i = int(i)
    if i in taken:
        continue
    s = knn[pos[i]] if len(main_idx) else np.array([9.0])
    c = int(np.argmin(s))
    a = A[i]
    vote = votes.get((items[i]["job"], items[i]["group"]))
    if not is_multi(i) and vote == c and s[c] < args.candidate and fits(a, main[c]) and \
            not (main[c]["kind"] == "person" and a["hair"] and main[c]["hair"] and main[c]["hair"] not in a["hair"]):
        recovered[c].append(i)
    elif is_multi(i):
        multi.append((c, float(s[c]), i))
    elif len(main_idx) and s[c] < args.candidate and fits(a, main[c]):
        cand_single[c].append((float(s[c]), i))
    else:
        unknown.append((c, float(s[c]), i))

# The two signals agreeing is still not proof (2026-10-09 check: back views and silhouettes of other people got in), so these
# are not added to the character - they lead its candidates (most likely first), then the minor groups, then single crops.
# Scene vote: the faces in the same scene (same video, same scene number) that belong to one main character.  It does not use
# CCIP, so it still says something about back views and close-ups; alone it was right for 80% of known no-face images.
scene_of = lambda i: items[i]["job"] + ":" + (items[i]["frame"].split(":")[1] if items[i]["frame"] else items[i]["rel"])
scene_faces = defaultdict(list)
for c, m in enumerate(main_idx):
    for i in m:
        if A[i]["face"]:
            scene_faces[scene_of(int(i))].append(c)


def scene_vote(i):
    mates = scene_faces.get(scene_of(i), [])
    if not mates:
        return -2  # no face of a main character in this scene
    c, k = Counter(mates).most_common(1)[0]
    return c if k / len(mates) >= 0.8 else -1  # -1: several characters in the scene


def score(i, c):
    """Lower = more likely character c.  CCIP closeness, minus the lead over the runner-up, adjusted by the scene vote.
    Only an ordering: how far down it can be trusted is measured from the user's own decisions (character_sets calibration)."""
    s = knn[pos[i]]
    other = np.delete(s, c).min() if len(s) > 1 else 9.0
    v = scene_vote(i)
    return float(s[c] - 0.5 * max(0.0, other - s[c]) + (-0.03 if v == c else 0.03 if v >= 0 else 0.0)), float(s[c]), float(other - s[c]), v


sections = []
absorbed = 0
for c in range(len(main)):
    members = [i for i in recovered[c]] + [rel_i[f] for b in cand_blocks[c] for f in b] + [i for _s, i in cand_single[c]]
    scored = sorted(((score(i, c), i) for i in members), key=lambda x: x[0][0])
    # --absorb: candidates this close go straight into the character (a few strays accepted for less manual sorting,
    # user decision 2026-10-09); the rest stay in "X の候補"
    take = [i for sc, i in scored if sc[0] <= args.absorb]
    if take:
        main[c]["files"] = sorted(main[c]["files"] + [items[i]["rel"] for i in take])
        main[c]["images"] = len(main[c]["files"])
        main[c]["absorbed"] = len(take)
        absorbed += len(take)
        scored = [(sc, i) for sc, i in scored if sc[0] > args.absorb]
    if scored:
        sections.append({"section": "candidates", "main": c, "files": [items[i]["rel"] for _sc, i in scored],
                         "scores": [[round(sc[0], 4), round(sc[1], 4), round(sc[2], 4), sc[3]] for sc, _i in scored]})
def nearest(i):  # [main index, CCIP closeness] - where a pending image would most likely belong, shown next to it for review
    s = knn[pos[i]]
    c = int(np.argmin(s))
    return [c, round(float(s[c]), 4)]


for mc in sorted(others, key=lambda m: -m["images"]):
    sections.append({"section": "other", "files": mc["files"], "kind": mc["kind"], "gender": mc["gender"], "hair": mc["hair"],
                     "nearest": [nearest(rel_i[f]) for f in mc["files"]]})
if multi:
    m = sorted(multi)
    sections.append({"section": "multi", "files": [items[i]["rel"] for _c, _s, i in m], "nearest": [nearest(i) for _c, _s, i in m]})
if unknown:
    u = sorted(unknown)
    sections.append({"section": "unknown", "files": [items[i]["rel"] for _c, _s, i in u], "nearest": [nearest(i) for _c, _s, i in u]})
files = [f for s in sections for f in s["files"]]
result = {"params": vars(args), "crops": n, "characters": main, "sections": sections,
          "pending": {"images": len(files), "files": files}}
Path(args.out).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")

if args.report:
    # consistency inside each sure character (person/creature, gender, hair colour) weighted by size
    def impurity(key, share):
        tot = sum(c["images"] for c in main)
        return round(sum((1 - c[share]) * c["images"] for c in main) / tot, 3) if tot else 0

    print(json.dumps({"sure_characters": len(main), "images_in_sure": sum(c["images"] for c in main),
                      "pending": len(files), "pending_share": round(len(files) / n, 3), "likely_candidates": sum(len(v) for v in recovered.values()), "absorbed": absorbed, "other_char_would_absorb": [sum(1 for mc in others for f in mc["files"] if min(score(rel_i[f], c)[0] for c in range(len(main))) <= t) for t in (0.0, 0.03, 0.06, 0.09, 0.12)], "other_char_images": sum(len(mc["files"]) for mc in others), "sections": [(s["section"], len(s["files"])) for s in sections],
                      "mix_kind": impurity("kind", "kind_share"), "mix_gender": impurity("gender", "gender_share"),
                      "mix_hair": impurity("hair", "hair_share"),
                      "sizes": [c["images"] for c in main][:30]}, ensure_ascii=False), flush=True)
print(f"done {time.time() - t0:.0f}s", flush=True)
