"""Cross-scene identity test with REAL labels: WD14 recognises the character names (nanoha / fate / hayate).
For every labelled crop we predict the character from crops of OTHER scenes only (so background/pose of the same shot cannot help),
with the face visible, with the head painted gray, and with only the lower half kept.  Chance for 3 classes ~ 0.33-0.6 (class imbalance).
"""
import csv
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

WORK = Path(sys.argv[1])
OUT = Path(sys.argv[2])
crops = json.loads((WORK / "crops.json").read_text(encoding="utf-8"))
WD = Path(r"C:\Users\Keishi\Portfolio\Toolchains-Physical\ComfyUI-Portable\ComfyUI_windows_portable\ComfyUI\custom_nodes\comfyui-wd14-tagger\models")
meta = list(csv.DictReader(open(WD / "wd-eva02-large-tagger-v3.csv", encoding="utf-8", newline="")))
NAMES = {"takamachi nanoha": "nanoha", "fate testarossa": "fate", "yagami hayate": "hayate"}
lab = []
for c in crops:
    found = {NAMES[t] for t in c["tags"] if t in NAMES}
    lab.append(next(iter(found)) if len(found) == 1 else None)
idx = [i for i, l in enumerate(lab) if l]
print("labelled crops", len(idx), Counter(lab[i] for i in idx), "scenes", len({crops[i]["scene"] for i in idx}), flush=True)

lib = Path(torch.__file__).parent / "lib"
os.add_dll_directory(str(lib))
os.environ["PATH"] = str(lib) + os.pathsep + os.environ["PATH"]
import onnxruntime as ort
from transformers import AutoModel, CLIPModel

sess = ort.InferenceSession(str(WD / "wd-eva02-large-tagger-v3.onnx"), providers=[("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"])
inp = sess.get_inputs()[0].name
clip = CLIPModel.from_pretrained(str(next((Path.home() / ".cache/huggingface/hub/models--sentence-transformers--clip-ViT-B-32/snapshots").iterdir()) / "0_CLIPModel")).cuda().half().eval()
dino = AutoModel.from_pretrained("camenduru/dinov3-vitl16-pretrain-lvd1689m", local_files_only=True).cuda().float().eval()
CM, CS = np.array([0.4815, 0.4578, 0.4082], np.float32), np.array([0.2686, 0.2613, 0.2758], np.float32)
DM, DS = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
HAIR = re.compile(r"^(blonde|brown|black|blue|red|pink|purple|green|white|silver|grey|gray|orange|aqua|light blue|dark blue|light brown|dark green|light purple|multicolored|two-tone|gradient|streaked|colored inner) hair$")
EYE = re.compile(r"^(?!closed|half-closed|wide|empty|crossed|one|heterochromia|torn|big|narrowed|rolling|looking)(.+) eyes$")
FEAT = {"animal ears", "twintails", "ponytail", "braid", "twin braids", "side ponytail", "short hair", "long hair", "medium hair", "very long hair", "ahoge", "hair bun", "glasses"}
CLOTH = re.compile(r"\b(shirt|skirt|dress|uniform|jacket|hoodie|sweater|coat|pants|shorts|swimsuit|kimono|armor|cape|cloak|vest|cardigan|leotard|bodysuit|necktie|bowtie|blouse|thighhighs|pantyhose|socks|boots|gloves)\b")


def letterbox(img, size=224):
    img = img.convert("RGB")
    s = size / max(img.size)
    img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.BICUBIC)
    c = Image.new("RGB", (size, size), (128, 128, 128))
    c.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
    return c


@torch.no_grad()
def embed(images, which):
    mean, std = (CM, CS) if which == "clip" else (DM, DS)
    x = np.stack([(np.asarray(letterbox(i), np.float32) / 255 - mean) / std for i in images]).transpose(0, 3, 1, 2)
    t = torch.from_numpy(x).cuda()
    if which == "clip":
        v = clip.get_image_features(pixel_values=t.half())
        v = v if torch.is_tensor(v) else v.pooler_output
    else:
        v = dino(pixel_values=t.float()).pooler_output
    return torch.nn.functional.normalize(v.float(), dim=-1).cpu().numpy()


def tag(images):
    out = []
    for s in range(0, len(images), 16):
        arr = np.stack([np.array(im.convert("RGB").resize((448, 448), Image.LANCZOS), dtype=np.float32)[:, :, ::-1] for im in images[s:s + 16]])
        for row in sess.run(None, {inp: arr})[0]:
            out.append([m["name"].replace("_", " ") for i, m in enumerate(meta) if i < len(row) and int(m.get("category", 9)) not in (9, 5, 4) and row[i] >= 0.35])
    return out


def variant(img, kind):
    a = np.array(img.convert("RGB"))
    h = a.shape[0]
    if kind == "head_masked":
        a[: int(h * 0.30)] = 128
    elif kind == "lower_half":
        a[: int(h * 0.50)] = 128
    return Image.fromarray(a)


def ident(t):
    return {x: (3.0 if HAIR.match(x) else 2.0 if EYE.match(x) else 1.0) for x in t if HAIR.match(x) or EYE.match(x) or x in FEAT}


def wj(a, b):
    k = set(a) | set(b)
    return sum(min(a.get(x, 0), b.get(x, 0)) for x in k) / sum(max(a.get(x, 0), b.get(x, 0)) for x in k) if k else 0.0


classes = sorted({lab[i] for i in idx})
y = np.array([classes.index(lab[i]) for i in idx])
scene = np.array([crops[i]["scene"] for i in idx])
results = {}
for kind in ("full", "head_masked", "lower_half"):
    imgs = []
    for i in idx:
        im = Image.open(WORK / "crops" / crops[i]["iso_file"])
        imgs.append(im if kind == "full" else variant(im, kind))
    E = {w: np.concatenate([embed(imgs[s:s + 32], w) for s in range(0, len(imgs), 32)]) for w in ("clip", "dino")}
    T = tag(imgs)
    feats = {"CLIP": E["clip"], "DINOv3": E["dino"]}
    cen = {k: (v - v.mean(0)) / np.linalg.norm(v - v.mean(0), axis=1, keepdims=True) for k, v in feats.items()}
    feats["CLIP centred"], feats["DINOv3 centred"] = cen["CLIP"], cen["DINOv3"]
    both = np.concatenate([cen["CLIP"], cen["DINOv3"]], 1)
    feats["CLIP+DINOv3 centred"] = both / np.linalg.norm(both, axis=1, keepdims=True)
    row = {}
    for name, X in feats.items():
        pred = []
        for a in range(len(idx)):
            m = scene != scene[a]  # other scenes only
            protos = np.stack([X[m & (y == k)].mean(0) for k in range(len(classes))])
            pred.append(int((X[a] @ protos.T).argmax()))
        pred = np.array(pred)
        recall = [float((pred[y == k] == k).mean()) for k in range(len(classes))]
        row[name] = {"acc": round(float((pred == y).mean()), 3), "macro_recall": round(float(np.mean(recall)), 3), "per_class_recall": {c: round(r, 2) for c, r in zip(classes, recall)}}
    # tag features (hair/eye/feature tags): prototype = mean tag-weights of other scenes
    pred = []
    idf = [ident(t) for t in T]
    for a in range(len(idx)):
        m = scene != scene[a]
        sc = []
        for k in range(len(classes)):
            members = [j for j in np.nonzero(m & (y == k))[0]]
            proto = defaultdict(float)
            for j in members:
                for kk, v in idf[j].items():
                    proto[kk] += v / len(members)
            sc.append(wj(idf[a], dict(proto)))
        pred.append(int(np.argmax(sc)))
    pred = np.array(pred)
    recall = [float((pred[y == k] == k).mean()) for k in range(len(classes))]
    row["tags: hair/eye/feature"] = {"acc": round(float((pred == y).mean()), 3), "macro_recall": round(float(np.mean(recall)), 3), "per_class_recall": {c: round(r, 2) for c, r in zip(classes, recall)}}
    results[kind] = row
    print(kind, flush=True)
    for name, r in row.items():
        print(f"   {name:24s} acc={r['acc']:.3f} macro_recall={r['macro_recall']:.3f} {r['per_class_recall']}", flush=True)
OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
