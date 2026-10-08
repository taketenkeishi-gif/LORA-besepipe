"""A character's images grouped by outfit, as a suggestion for LoRA instances.

Signal: the WD14 clothing tags of each image (dress, skirt, uniform, gloves …), compared by Jaccard distance and grouped by
average linkage.  2026-10-09 comparison (tools/video_dataset/evaluation/outfit_split_eval.py): CLIP on the whole picture
follows the person and the background, not the clothes (1-4 groups, homogeneity 0.05-0.14 against the per-video outfit names);
clothing tags at 0.7 put 77 % of the images into groups.  Groups that get the same Japanese outfit name are merged, groups
under MIN_IMAGES go to "その他の衣装".  The reference itself comes from tags, so this is a suggestion the user edits, not a
measured result.
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np

from . import ja_names

CLOTH = re.compile(r"(dress|skirt|shirt|jacket|coat|uniform|serafuku|kimono|swimsuit|bikini|apron|pants|shorts|cape|hat|armor|pajamas|"
                   r"blouse|cardigan|leotard|suit|hoodie|sweater|vest|gloves|thighhighs|pantyhose|boots|ribbon|bow|necktie|bowtie|scarf|"
                   r"collar|sleeves|frills|overalls|robe|beret|headband|hairband|tiara|crown|choker|belt|shoes|socks|sailor)")
THRESHOLD = 0.7
MIN_IMAGES = 30  # an outfit instance with fewer images teaches little; 12 gave 8-11 instances per character (real-window check 2026-10-09)
OTHER = "その他の衣装"
# a group is an outfit only when a garment shows; accessory-only groups (bow tie, ribbon, hat, tiara …) are mostly face close-ups
GARMENT_JA = ("ドレス", "スカート", "シャツ", "ジャケット", "コート", "制服", "セーラー服", "着物", "水着", "ビキニ", "エプロン", "ズボン",
              "ショートパンツ", "マント", "鎧", "パジャマ", "Tシャツ", "ブラウス", "カーディガン", "レオタード", "スーツ", "パーカー",
              "セーター", "ベスト", "メイド服", "ナース服", "ジャージ", "オーバーオール", "ローブ")


def _jaccard(M: np.ndarray) -> np.ndarray:
    M = M.astype(np.float32)
    inter = M @ M.T
    size = M.sum(1)
    union = size[:, None] + size[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        d = 1 - np.where(union > 0, inter / union, 0.0)
    np.fill_diagonal(d, 0)
    return d


def _name(tag_sets: list[set[str]]) -> str:
    """Japanese name from the clothing tags most images of the group share."""
    cnt = Counter(t for s in tag_sets for t in s)
    common = [t for t, k in cnt.most_common(6) if k >= len(tag_sets) * 0.4]
    return ja_names.outfit("-".join(t.replace(" ", "-") for t in common)) or ""


def split(images: list[str], tags_of) -> list[dict]:
    """[{name, files}] largest first, "その他の衣装" last.  tags_of(rel) -> list of tags."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    sets = [{t for t in tags_of(r) if CLOTH.search(t)} for r in images]
    has = [k for k, s in enumerate(sets) if s]
    if len(has) < MIN_IMAGES:
        return [{"name": OTHER, "files": list(images)}] if images else []
    vocab = sorted({t for k in has for t in sets[k]})
    pos = {t: j for j, t in enumerate(vocab)}
    M = np.zeros((len(has), len(vocab)), np.uint8)
    for row, k in enumerate(has):
        for t in sets[k]:
            M[row, pos[t]] = 1
    lab = fcluster(linkage(squareform(_jaccard(M), checks=False), "average"), THRESHOLD, "distance")
    groups: dict[str, list[int]] = {}
    for row, l in enumerate(lab):
        groups.setdefault(int(l), []).append(has[row])
    named: dict[str, list[int]] = {}
    other: list[int] = [k for k, s in enumerate(sets) if not s]
    for members in groups.values():
        if len(members) < MIN_IMAGES:
            other += members
            continue
        name = _name([sets[k] for k in members]) or OTHER
        if name == OTHER or not any(g in name for g in GARMENT_JA):
            other += members
        else:
            named.setdefault(name, []).extend(members)
    out = [{"name": n, "files": [images[k] for k in sorted(m)]} for n, m in sorted(named.items(), key=lambda x: -len(x[1]))]
    if other:
        out.append({"name": OTHER, "files": [images[k] for k in sorted(other)]})
    return out
