"""Tune the CLIP-prototype merge / assignment thresholds against the real name labels (nanoha / fate / hayate)."""
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
from transformers import CLIPModel

WORK = Path(sys.argv[1])
crops = json.loads((WORK / "crops.json").read_text(encoding="utf-8"))
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
E = np.concatenate([embed(imgs[s:s + 32]) for s in range(0, len(imgs), 32)])
NAMES = {"takamachi nanoha": "nanoha", "fate testarossa": "fate", "yagami hayate": "hayate"}
name = []
for c in crops:
    f = {NAMES[t] for t in c["tags"] if t in NAMES}
    name.append(next(iter(f)) if len(f) == 1 else None)
cl = np.array([c.get("char") or 0 for c in crops])
ids = [k for k in sorted(set(cl)) if k and (cl == k).sum() >= 4]
proto = {k: (lambda v: v / np.linalg.norm(v))(E[cl == k].mean(0)) for k in ids}
maj = {}
for k in ids:
    cnt = Counter(n for n, c in zip(name, cl) if c == k and n)
    if cnt and cnt.most_common(1)[0][1] / sum(cnt.values()) >= 0.8:
        maj[k] = cnt.most_common(1)[0][0]
print("clusters", len(ids), "with a clear name", len(maj), flush=True)
same, diff = [], []
for a in maj:
    for b in maj:
        if a < b:
            (same if maj[a] == maj[b] else diff).append(float(proto[a] @ proto[b]))
print(f"prototype cosine: same character clusters n={len(same)} median={np.median(same):.3f} min={min(same):.3f}   different n={len(diff)} median={np.median(diff):.3f} max={max(diff):.3f}")
for t in np.arange(0.80, 0.99, 0.02):
    print(f"  merge thr {t:.2f}: merges same-character pairs {np.mean([s >= t for s in same]):.2f}   WRONG merges of different characters {np.mean([d >= t for d in diff]):.2f}")

# assignment of individual labelled crops to the nearest prototype (own cluster included - this is the final assignment step)
lab_idx = [i for i, n in enumerate(name) if n]
res = []
for i in lab_idx:
    sims = {k: float(E[i] @ proto[k]) for k in maj}
    best = max(sims, key=sims.get)
    srt = sorted(sims.values(), reverse=True)
    res.append((maj[best] == name[i], sims[best], sims[best] - (srt[1] if len(srt) > 1 else 0)))
res = np.array(res)
print("assign thr -> coverage / accuracy (labelled crops):")
for t in (0.80, 0.84, 0.88, 0.90, 0.92, 0.94):
    m = res[:, 1] >= t
    print(f"  cos>={t:.2f}: coverage {m.mean():.2f}  accuracy {res[m, 0].mean() if m.any() else 0:.3f}")
