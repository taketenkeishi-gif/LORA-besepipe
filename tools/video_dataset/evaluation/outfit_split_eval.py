"""Outfit split inside each main character: which signal groups a character's images by outfit.

python outfit_split_eval.py FEAT_PREFIX RESULT.json
Signals: CLIP (whole picture), clothing tags (Jaccard over WD14 clothing tags), both.  Reference (not ground truth): the
per-video outfit folders written when each video was extracted (outfit_NN_<slug>, NN 00/99 = unknown/other ignored).
Reported per signal and threshold: groups >= 12 images per character, share of images in them, and homogeneity / completeness
of those groups against the per-video outfit slugs (higher = closer to the reference split).
"""
import json, re, sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

prefix, res = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
items = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
C = np.load(prefix.with_name(prefix.name + ".clip.npy")).astype(np.float32)
idx = {it["rel"]: i for i, it in enumerate(items)}
CLOTH = re.compile(r"(dress|skirt|shirt|jacket|coat|uniform|serafuku|kimono|swimsuit|bikini|apron|pants|shorts|cape|hat|armor|pajamas|"
                   r"blouse|cardigan|leotard|suit|hoodie|sweater|vest|gloves|thighhighs|pantyhose|boots|ribbon|bow|necktie|bowtie|scarf|"
                   r"collar|sleeves|frills|overalls|robe|beret|headband|hairband|tiara|crown|choker|belt|shoes|socks|sailor)")


def slug(it):
    m = re.match(r"outfit_(\d{2})_?(.*)", it.get("outfit") or "")
    return None if not m or m.group(1) in ("00", "99") or not m.group(2) else m.group(2)  # outfit name (same words across videos)


def cloth(it):
    return {t for t in it["tags"] if CLOTH.search(t)}


def jaccard_dist(sets):
    n = len(sets)
    D = np.zeros((n, n), np.float32)
    for a in range(n):
        for b in range(a + 1, n):
            u = len(sets[a] | sets[b])
            D[a, b] = D[b, a] = 1 - (len(sets[a] & sets[b]) / u if u else 0)
    return D


def homog_compl(labels, ref):
    pairs = [(l, r) for l, r in zip(labels, ref) if r is not None and l >= 0]
    if not pairs:
        return 0, 0
    from math import log
    def H(c):
        t = sum(c.values()); return -sum(v / t * log(v / t) for v in c.values() if v)
    L, R = Counter(p[0] for p in pairs), Counter(p[1] for p in pairs)
    joint = Counter(pairs)
    n = len(pairs)
    HC_K = -sum(v / n * log(v / L[l]) for (l, r), v in joint.items())
    HK_C = -sum(v / n * log(v / R[r]) for (l, r), v in joint.items())
    h = 1 - HC_K / H(R) if H(R) else 1
    c = 1 - HK_C / H(L) if H(L) else 1
    return round(h, 3), round(c, 3)


for name, mode in (("CLIP", "clip"), ("clothing tags", "tags"), ("CLIP + tags", "both")):
    for t in ((0.25, 0.3, 0.35) if mode == "clip" else (0.5, 0.6, 0.7) if mode == "tags" else (0.35, 0.45, 0.55)):
        rows = []
        for ch in res["characters"]:
            m = [idx[f] for f in ch["files"]]
            if mode in ("clip", "both"):
                E = C[m]; Dc = np.clip(1 - E @ E.T, 0, 2)
            if mode in ("tags", "both"):
                Dt = jaccard_dist([cloth(items[i]) for i in m])
            D = Dc if mode == "clip" else Dt if mode == "tags" else (Dc / 0.3 + Dt / 0.6) / 2 * 0.45 / 0.45
            np.fill_diagonal(D, 0)
            lab = fcluster(linkage(squareform(D, checks=False), "average"), t, "distance")
            cnt = Counter(lab)
            big = {l for l, k in cnt.items() if k >= 12}
            lab2 = [l if l in big else -1 for l in lab]
            h, c = homog_compl(lab2, [slug(items[i]) for i in m])
            rows.append((len(big), sum(cnt[l] for l in big) / len(m), h, c))
        a = np.array(rows)
        print(f"{name:14s} t={t:<5} outfits/char {a[:,0].mean():4.1f} (min {int(a[:,0].min())} max {int(a[:,0].max())})  "
              f"in outfits {a[:,1].mean():.2f}  homogeneity {a[:,2].mean():.3f}  completeness {a[:,3].mean():.3f}")
