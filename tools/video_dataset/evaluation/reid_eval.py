"""Which feature tells characters apart - with and without the face?  Ground truth that needs no labelling:
  positive pair = two crops of the SAME track (the same person followed through one scene)
  negative pair = two crops of DIFFERENT people in the SAME frame (always different characters)
AUC = chance that a positive pair scores higher than a negative pair (0.5 = useless, 1.0 = perfect).
Features: WD14 hair/eye/feature tags, WD14 clothing tags, DINOv3 (ViT-L/16) embedding, CLIP ViT-B/32 embedding.
"""
import itertools
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("GPU", "1")
import numpy as np
import torch
from PIL import Image

WORK = Path(sys.argv[1])
OUT = Path(sys.argv[2])
crops = json.loads((WORK / "crops.json").read_text(encoding="utf-8"))
for c in crops:
    c["k"] = int(re.search(r"_k(\d+)_", c["file"]).group(1))
print("crops", len(crops), flush=True)

HAIR = re.compile(r"^(blonde|brown|black|blue|red|pink|purple|green|white|silver|grey|gray|orange|aqua|light blue|dark blue|light brown|dark green|light purple|multicolored|two-tone|gradient|streaked|colored inner) hair$")
EYE = re.compile(r"^(?!closed|half-closed|wide|empty|crossed|one|heterochromia|torn|big|narrowed|rolling|looking)(.+) eyes$")
FEAT = {"animal ears", "cat ears", "fox ears", "dog ears", "rabbit ears", "tail", "fox tail", "cat tail", "horns", "glasses", "twintails", "ponytail", "braid", "twin braids", "side ponytail",
        "short hair", "long hair", "medium hair", "very long hair", "ahoge", "hair bun", "drill hair", "pointy ears", "elf", "halo", "wings"}
CLOTH = re.compile(r"\b(shirt|skirt|dress|uniform|jacket|hoodie|sweater|coat|pants|shorts|swimsuit|bikini|kimono|yukata|pajamas|armor|cape|cloak|apron|vest|cardigan|leotard|bodysuit|jeans|tank top|serafuku|necktie|bowtie|blouse|gown|robe|miniskirt|pleated skirt|t-shirt|sailor collar|maid|bra|panties|thighhighs|pantyhose|kneehighs|socks|boots|gloves)\b")


def ident(tags):
    return {t: (3.0 if HAIR.match(t) else 2.0 if EYE.match(t) else 1.0) for t in tags if HAIR.match(t) or EYE.match(t) or t in FEAT}


def wj(a, b):
    k = set(a) | set(b)
    return sum(min(a.get(x, 0), b.get(x, 0)) for x in k) / sum(max(a.get(x, 0), b.get(x, 0)) for x in k) if k else 0.0


def oc(a, b):
    return len(a & b) / min(len(a), len(b)) if a and b else 0.0


# pairs
by_track = defaultdict(list)
by_frame = defaultdict(list)
for i, c in enumerate(crops):
    by_track[c["track"]].append(i)
    by_frame[(c["scene"], c["k"])].append(i)
def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


STRICT = os.environ.get("STRICT_POS", "1") == "1"  # only trust track links whose boxes really overlap
pos = [(a, b) for m in by_track.values() if len(m) > 1 for a, b in itertools.combinations(m, 2)
       if not STRICT or (crops[a]["scene"] == crops[b]["scene"] and _iou(crops[a]["box"], crops[b]["box"]) >= 0.5)]
neg = [(a, b) for m in by_frame.values() for a, b in itertools.combinations(m, 2) if crops[a]["track"] != crops[b]["track"]]
rng = np.random.default_rng(0)
if len(pos) > 4000:
    pos = [pos[i] for i in rng.choice(len(pos), 4000, replace=False)]
if len(neg) > 4000:
    neg = [neg[i] for i in rng.choice(len(neg), 4000, replace=False)]
print("positive pairs", len(pos), "negative pairs", len(neg), flush=True)


def auc(pos_scores, neg_scores):
    s = np.concatenate([pos_scores, neg_scores])
    order = s.argsort()
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / cnt)[inv]
    n_pos, n_neg = len(pos_scores), len(neg_scores)
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def variant(img: Image.Image, kind: str) -> Image.Image:
    a = np.array(img.convert("RGB"))
    h = a.shape[0]
    if kind == "head_masked":
        a[: int(h * 0.30)] = 128
    elif kind == "lower_half":
        a[: int(h * 0.50)] = 128
    return Image.fromarray(a)


def letterbox(img: Image.Image, size: int) -> Image.Image:
    img = img.convert("RGB")
    s = size / max(img.size)
    img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.BICUBIC)
    canvas = Image.new("RGB", (size, size), (128, 128, 128))
    canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
    return canvas


from transformers import AutoModel, CLIPModel

dino = AutoModel.from_pretrained("camenduru/dinov3-vitl16-pretrain-lvd1689m", local_files_only=True).cuda().float().eval()  # fp16 overflows in DINOv3 (NaN)
clip = CLIPModel.from_pretrained(str(next((Path.home() / ".cache/huggingface/hub/models--sentence-transformers--clip-ViT-B-32/snapshots").iterdir()) / "0_CLIPModel")).cuda().half().eval()
D_MEAN, D_STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
C_MEAN, C_STD = np.array([0.4815, 0.4578, 0.4082], np.float32), np.array([0.2686, 0.2613, 0.2758], np.float32)


@torch.no_grad()
def embed(images, which):
    size, mean, std = (224, D_MEAN, D_STD) if which == "dino" else (224, C_MEAN, C_STD)
    x = np.stack([(np.asarray(letterbox(im, size), np.float32) / 255.0 - mean) / std for im in images]).transpose(0, 3, 1, 2)
    t = torch.from_numpy(x).cuda()
    t = t.float() if which == "dino" else t.half()
    if which == "dino":
        o = dino(pixel_values=t)
        v = o.pooler_output
    else:
        v = clip.get_image_features(pixel_values=t)
        v = v if torch.is_tensor(v) else v.pooler_output
    v = torch.nn.functional.normalize(v.float(), dim=-1)
    return v.cpu().numpy()


results = {}
for kind in ("full", "head_masked", "lower_half"):
    imgs = []
    for c in crops:
        im = Image.open(WORK / "crops" / c["iso_file"])
        imgs.append(im if kind == "full" else variant(im, kind))
    emb = {}
    for which in ("dino", "clip"):
        vecs = []
        for s in range(0, len(imgs), 32):
            vecs.append(embed(imgs[s:s + 32], which))
        emb[which] = np.concatenate(vecs)
    # tags for this variant (full = already computed; masked variants are re-tagged below only for hair/cloth features)
    if kind == "full":
        tags = [c["tags"] for c in crops]
    else:
        import csv

        import onnxruntime as ort

        lib = Path(torch.__file__).parent / "lib"
        os.add_dll_directory(str(lib))
        WD = Path(r"C:\Users\Keishi\Portfolio\Toolchains-Physical\ComfyUI-Portable\ComfyUI_windows_portable\ComfyUI\custom_nodes\comfyui-wd14-tagger\models")
        if "sess" not in globals():
            sess = ort.InferenceSession(str(WD / "wd-eva02-large-tagger-v3.onnx"), providers=[("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"])
            meta = list(csv.DictReader(open(WD / "wd-eva02-large-tagger-v3.csv", encoding="utf-8", newline="")))
            inp = sess.get_inputs()[0].name
        tags = []
        for s in range(0, len(imgs), 16):
            arr = np.stack([np.array(im.convert("RGB").resize((448, 448), Image.LANCZOS), dtype=np.float32)[:, :, ::-1] for im in imgs[s:s + 16]])
            preds = sess.run(None, {inp: arr})[0]
            for row in preds:
                tags.append([m["name"].replace("_", " ") for i, m in enumerate(meta) if i < len(row) and int(m.get("category", 9)) not in (9, 5) and row[i] >= (0.85 if int(m.get("category", 9)) == 4 else 0.35)])
    ident_f = [ident(t) for t in tags]
    cloth_f = [{x for x in t if CLOTH.search(x)} for t in tags]

    def score(fn):
        return np.array([fn(a, b) for a, b in pos]), np.array([fn(a, b) for a, b in neg])

    feats = {
        "tags: hair/eye/feature": lambda a, b: wj(ident_f[a], ident_f[b]),
        "tags: clothing": lambda a, b: oc(cloth_f[a], cloth_f[b]),
        "DINOv3 embedding": lambda a, b: float(emb["dino"][a] @ emb["dino"][b]),
        "CLIP embedding": lambda a, b: float(emb["clip"][a] @ emb["clip"][b]),
    }
    results[kind] = {}
    scores = {}
    for name, fn in feats.items():
        p, n = score(fn)
        scores[name] = (p, n)
        results[kind][name] = round(auc(p, n), 3)
    # combination: DINOv3 + hair tags
    pd, nd = scores["DINOv3 embedding"]
    ph, nh = scores["tags: hair/eye/feature"]
    results[kind]["DINOv3 + hair tags"] = round(auc(pd + 0.5 * ph, nd + 0.5 * nh), 3)
    print(kind, json.dumps(results[kind], ensure_ascii=False), flush=True)
    if kind == "full":  # best DINOv3 threshold for 'same person' at full view, for clustering later
        thr = np.linspace(0.2, 0.95, 76)
        best = max(thr, key=lambda t: ((pd >= t).mean() + (nd < t).mean()) / 2)
        results["dino_best_threshold_full"] = {"threshold": round(float(best), 3), "balanced_acc": round(float(((pd >= best).mean() + (nd < best).mean()) / 2), 3),
                                               "pos_median": round(float(np.median(pd)), 3), "neg_median": round(float(np.median(nd)), 3)}
        print(results["dino_best_threshold_full"], flush=True)
OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
np.save(OUT.with_suffix(".dino_full.npy"), emb["dino"]) if False else None
