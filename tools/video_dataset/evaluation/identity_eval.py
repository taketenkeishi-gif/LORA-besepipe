"""Can a crop be assigned to the right character when the face is NOT visible?

Labels = the character each crop was assigned to when its face/hair was visible (from crops.json of the real run).
Test inputs = the same crops with the head painted gray / only the lower half kept / real back views.
The prototypes are built WITHOUT the tested crop's own track, so there is no leakage from the same shot.
Two classifiers: hair/eye/feature tags (needs the head) and clothing tags (works from the neck down).
"""
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("GPU", "1")
import numpy as np
import torch
from PIL import Image

lib = Path(torch.__file__).parent / "lib"
os.add_dll_directory(str(lib))
os.environ["PATH"] = str(lib) + os.pathsep + os.environ["PATH"]
import onnxruntime as ort

WORK = Path(sys.argv[1])
OUT = Path(sys.argv[2])
WD = Path(r"C:\Users\Keishi\Portfolio\Toolchains-Physical\ComfyUI-Portable\ComfyUI_windows_portable\ComfyUI\custom_nodes\comfyui-wd14-tagger\models")
sess = ort.InferenceSession(str(WD / "wd-eva02-large-tagger-v3.onnx"), providers=[("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"])
tags_meta = list(csv.DictReader(open(WD / "wd-eva02-large-tagger-v3.csv", encoding="utf-8", newline="")))
inp = sess.get_inputs()[0].name


def tag(images, thr=0.35):
    arr = np.stack([np.array(im.convert("RGB").resize((448, 448), Image.LANCZOS), dtype=np.float32)[:, :, ::-1] for im in images])
    preds = sess.run(None, {inp: arr})[0]
    res = []
    for row in preds:
        res.append([t["name"].replace("_", " ") for i, t in enumerate(tags_meta) if i < len(row) and int(t.get("category", 9)) not in (9, 5) and row[i] >= (0.85 if int(t.get("category", 9)) == 4 else thr)])
    return res


# reuse the feature definitions of the pipeline
import re

HAIR = re.compile(r"^(blonde|brown|black|blue|red|pink|purple|green|white|silver|grey|gray|orange|aqua|light blue|dark blue|light brown|dark green|light purple|multicolored|two-tone|gradient|streaked|colored inner) hair$")
EYE = re.compile(r"^(?!closed|half-closed|wide|empty|crossed|one|heterochromia|torn|big|narrowed|rolling|looking)(.+) eyes$")
FEAT = {"animal ears", "cat ears", "fox ears", "dog ears", "rabbit ears", "tail", "fox tail", "cat tail", "horns", "glasses", "twintails", "ponytail", "braid", "twin braids", "side ponytail",
        "short hair", "long hair", "medium hair", "very long hair", "ahoge", "hair bun", "drill hair", "pointy ears", "elf", "halo", "wings"}
CLOTH = re.compile(r"\b(shirt|skirt|dress|uniform|jacket|hoodie|sweater|coat|pants|shorts|swimsuit|bikini|kimono|yukata|pajamas|armor|cape|cloak|apron|vest|cardigan|leotard|bodysuit|jeans|tank top|serafuku|necktie|bowtie|blouse|gown|robe|miniskirt|pleated skirt|t-shirt|sailor collar|maid|bra|panties|thighhighs|pantyhose|kneehighs|socks|boots|gloves)\b")


def ident(tags):
    f = {}
    for t in tags:
        if HAIR.match(t):
            f[t] = 3.0
        elif EYE.match(t):
            f[t] = 2.0
        elif t in FEAT:
            f[t] = 1.0
    return f


def cloth(tags):
    return {t for t in tags if CLOTH.search(t)}


def wj(a, b):
    k = set(a) | set(b)
    return sum(min(a.get(x, 0), b.get(x, 0)) for x in k) / sum(max(a.get(x, 0), b.get(x, 0)) for x in k) if k else 0.0


def oc(a, b):
    return len(a & b) / min(len(a), len(b)) if a and b else 0.0


crops = json.loads((WORK / "crops.json").read_text(encoding="utf-8"))
labelled = [c for c in crops if c.get("char")]
print("labelled crops", len(labelled), "characters", len({c["char"] for c in labelled}), flush=True)


def prototypes(exclude_track):
    """hair prototype from crops that show a face; clothing prototype = tags seen in >=40% of the character's crops."""
    hair_sum, hair_n, cl = defaultdict(lambda: defaultdict(float)), Counter(), defaultdict(Counter)
    total = Counter()
    for c in labelled:
        if c["track"] == exclude_track:
            continue
        f = ident(c["tags"])
        if any(HAIR.match(t) for t in c["tags"]):
            for k, v in f.items():
                hair_sum[c["char"]][k] += v
            hair_n[c["char"]] += 1
        total[c["char"]] += 1
        cl[c["char"]].update(cloth(c["tags"]))
    hair = {ch: {k: v / hair_n[ch] for k, v in d.items()} for ch, d in hair_sum.items() if hair_n[ch]}
    clo = {ch: {t for t, n in cnt.items() if n >= 0.4 * total[ch]} for ch, cnt in cl.items()}
    return hair, clo


def classify(tags, hair, clo):
    """returns (method, best, margin) - hair first, clothing as the fallback when the head gives nothing."""
    f = ident(tags)
    if any(HAIR.match(t) for t in tags) and hair:
        s = sorted(((wj(f, p), ch) for ch, p in hair.items()), reverse=True)
        return "hair", s[0][1], s[0][0] - (s[1][0] if len(s) > 1 else 0.0)
    c = cloth(tags)
    s = sorted(((oc(c, p), ch) for ch, p in clo.items()), reverse=True)
    if not s or s[0][0] == 0:
        return "none", None, 0.0
    return "cloth", s[0][1], s[0][0] - (s[1][0] if len(s) > 1 else 0.0)


def variant(img: Image.Image, kind: str) -> Image.Image:
    a = np.array(img.convert("RGB"))
    h = a.shape[0]
    if kind == "head_masked":
        a[: int(h * 0.30)] = 128
    elif kind == "lower_half":
        a[: int(h * 0.50)] = 128
    return Image.fromarray(a)


results = {}
tests = {"head_masked": [c for c in labelled if c["view"].split("-")[1] in ("upper", "cowboy", "full", "other", "lower-cut", "head-cut") and c["view"].split("-")[0] != "back"],
         "lower_half": [c for c in labelled if c["view"].split("-")[1] in ("cowboy", "full")],
         "real_back": [c for c in labelled if c["view"].startswith("back")],
         "unmodified_check": [c for c in labelled if c["view"].split("-")[1] in ("upper", "cowboy", "full")]}
for name, group in tests.items():
    if not group:
        results[name] = {"n": 0}
        continue
    imgs = []
    for c in group:
        im = Image.open(WORK / "crops" / c["iso_file"])
        imgs.append(variant(im, name) if name in ("head_masked", "lower_half") else im)
    tagged = []
    for s in range(0, len(imgs), 16):
        tagged += tag(imgs[s:s + 16])
    rows = []
    for c, t in zip(group, tagged):
        hair, clo = prototypes(c["track"])
        method, best, margin = classify(t, hair, clo)
        rows.append({"label": c["char"], "pred": best, "method": method, "margin": margin})
    n = len(rows)
    assigned = [r for r in rows if r["pred"] is not None]
    correct = [r for r in assigned if r["pred"] == r["label"]]
    by_method = {m: {"n": sum(r["method"] == m for r in rows), "acc": round(sum(r["pred"] == r["label"] for r in rows if r["method"] == m) / max(1, sum(r["method"] == m for r in rows)), 3)} for m in ("hair", "cloth", "none")}
    # accuracy vs coverage when low-margin answers are refused ("unassigned" instead of a wrong folder)
    tradeoff = {}
    for thr in (0.0, 0.1, 0.2, 0.3):
        kept = [r for r in assigned if r["margin"] >= thr]
        tradeoff[str(thr)] = {"coverage": round(len(kept) / n, 3), "accuracy": round(sum(r["pred"] == r["label"] for r in kept) / max(1, len(kept)), 3)}
    conf = Counter((r["label"], r["pred"]) for r in assigned if r["pred"] != r["label"])
    results[name] = {"n": n, "assigned": round(len(assigned) / n, 3), "accuracy_among_assigned": round(len(correct) / max(1, len(assigned)), 3), "by_method": by_method,
                     "margin_tradeoff": tradeoff, "top_confusions(label->pred)": [f"{a}->{b}: {k}" for (a, b), k in conf.most_common(5)]}
    print(name, json.dumps(results[name], ensure_ascii=False), flush=True)
OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
