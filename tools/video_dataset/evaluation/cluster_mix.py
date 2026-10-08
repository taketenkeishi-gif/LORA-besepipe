"""Same yardstick for any character grouping: how many groups a person must review, and how mixed they are.

python cluster_mix.py FEAT_PREFIX GROUPING.json
GROUPING = crop_cluster.py output ({characters:[{files}], pending:{files}}) or crop_link.py output ({clusters:[{files}]}).
mix_* = share of images (weighted by group size) that disagree with their group's majority on: person/creature, girl/boy, hair colour.
"""
import json, sys
from collections import Counter
from pathlib import Path

prefix, grouping = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
items = {it["rel"]: it for it in json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))}
COLOURS = {"blonde", "black", "blue", "pink", "purple", "green", "white", "grey", "gray", "silver", "aqua", "red", "brown", "orange",
           "light blue", "dark blue", "light purple", "light brown"}
NONHUMAN = {"no humans", "creature", "pokemon (creature)", "furry", "animal", "mascot", "robot", "stuffed toy"}


def facts(rel):
    tags = set(items[rel]["tags"])
    kind = "creature" if tags & NONHUMAN and "1girl" not in tags and "1boy" not in tags else "person"
    gender = "boy" if ("1boy" in tags or "male focus" in tags) and "1girl" not in tags else "girl" if "1girl" in tags else ""
    return kind, gender, {t[:-5] for t in tags if t.endswith(" hair") and t[:-5] in COLOURS}


groups = [g["files"] for g in grouping.get("characters") or grouping.get("clusters")]
pending = len(grouping.get("pending", {}).get("files", []))
total = sum(map(len, groups))
bad = Counter()
for files in groups:
    f = [facts(r) for r in files if r in items]
    k = Counter(x[0] for x in f); bad["kind"] += len(f) - k.most_common(1)[0][1]
    g = [x[1] for x in f if x[0] == "person" and x[1]]
    if g:
        bad["gender"] += len(g) - Counter(g).most_common(1)[0][1]
    h = [x[2] for x in f if x[2]]
    if h:
        top = Counter(c for s in h for c in s).most_common(1)[0][0]
        bad["hair"] += sum(1 for s in h if top not in s)
sizes = sorted(map(len, groups), reverse=True)
print(json.dumps({"groups": len(groups), "groups_20_plus": sum(s >= 20 for s in sizes), "images_in_groups": total, "pending": pending,
                  "mix_kind": round(bad["kind"] / total, 3), "mix_gender": round(bad["gender"] / total, 3), "mix_hair": round(bad["hair"] / total, 3),
                  "largest": sizes[:12]}, ensure_ascii=False))
