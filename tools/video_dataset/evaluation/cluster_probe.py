"""Offline probe on the saved crops: does mean-centring the embeddings separate people better, and how many characters come out?"""
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
crops = json.loads((WORK / "crops.json").read_text(encoding="utf-8"))
from transformers import AutoModel, CLIPModel

clip_dir = next((Path.home() / ".cache/huggingface/hub/models--sentence-transformers--clip-ViT-B-32/snapshots").iterdir()) / "0_CLIPModel"
clip = CLIPModel.from_pretrained(str(clip_dir)).cuda().half().eval()
dino = AutoModel.from_pretrained("camenduou/none" if False else "camenduru/dinov3-vitl16-pretrain-lvd1689m", local_files_only=True).cuda().float().eval()
CM, CS = np.array([0.4815, 0.4578, 0.4082], np.float32), np.array([0.2686, 0.2613, 0.2758], np.float32)
DM, DS = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)


def letterbox(img, size=224):
    img = img.convert("RGB")
    s = size / max(img.size)
    img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.BICUBIC)
    c = Image.new("RGB", (size, size), (128, 128, 128))
    c.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
    return c


@torch.no_grad()
def emb(images, which):
    mean, std = (CM, CS) if which == "clip" else (DM, DS)
    x = np.stack([(np.asarray(letterbox(i), np.float32) / 255 - mean) / std for i in images]).transpose(0, 3, 1, 2)
    t = torch.from_numpy(x).cuda()
    if which == "clip":
        v = clip.get_image_features(pixel_values=t.half())
        v = v if torch.is_tensor(v) else v.pooler_output
    else:
        v = dino(pixel_values=t.float()).pooler_output
    return torch.nn.functional.normalize(v.float(), dim=-1).cpu().numpy()


imgs = [Image.open(WORK / "crops" / c["iso_file"]) for c in crops]
E = {w: np.concatenate([emb(imgs[s:s + 32], w) for s in range(0, len(imgs), 32)]) for w in ("clip", "dino")}


def centre(x):
    x = x - x.mean(0, keepdims=True)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


by_track, by_frame = defaultdict(list), defaultdict(list)
for i, c in enumerate(crops):
    k = int(re.search(r"_k(\d+)_", c["file"]).group(1))
    c["k"] = k
    by_track[c["track"]].append(i)
    by_frame[(c["scene"], k)].append(i)
pos = [(a, b) for m in by_track.values() for ai, a in enumerate(m) for b in m[ai + 1:] if crops[a]["scene"] == crops[b]["scene"] and iou(crops[a]["box"], crops[b]["box"]) >= 0.5]
neg = [(a, b) for m in by_frame.values() for ai, a in enumerate(m) for b in m[ai + 1:] if crops[a]["track"] != crops[b]["track"]]


def auc(p, n):
    s = np.concatenate([p, n])
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    r = np.empty(len(s))
    r[s.argsort()] = np.arange(1, len(s) + 1)
    r = (np.bincount(inv, weights=r) / cnt)[inv]
    return float((r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


def constrained_upgma(sim, members, conflict, thr):
    k = len(members)
    sizes = np.array([len(m) for m in members], float)
    C, conf, alive = sim.copy(), conflict.copy(), np.ones(k, bool)
    np.fill_diagonal(C, -1)
    while True:
        m = np.where(alive[:, None] & alive[None, :] & ~conf, C, -1)
        np.fill_diagonal(m, -1)
        a, b = divmod(int(m.argmax()), k)
        if m[a, b] < thr:
            break
        row = (sizes[a] * C[a] + sizes[b] * C[b]) / (sizes[a] + sizes[b])
        C[a, :] = row
        C[:, a] = row
        C[a, a] = -1
        conf[a, :] |= conf[b, :]
        conf[:, a] |= conf[:, b]
        members[a] = members[a] + members[b]
        sizes[a] += sizes[b]
        alive[b] = False
    return [m for m, ok in zip(members, alive) if ok]


tids = list(by_track)
M = np.zeros((len(tids), len(crops)))
for ci, t in enumerate(tids):
    M[ci, by_track[t]] = 1
F = np.zeros((len(crops), len(crops)))
for m in by_frame.values():
    for a in m:
        for b in m:
            if crops[a]["track"] != crops[b]["track"]:
                F[a, b] = 1
sz = M.sum(1)
track_conf = (M @ F @ M.T) > 0
for name, X in (("CLIP raw", E["clip"]), ("CLIP centred", centre(E["clip"])), ("DINOv3 raw", E["dino"]), ("DINOv3 centred", centre(E["dino"])),
                ("CLIP+DINOv3 centred", centre(np.concatenate([centre(E["clip"]), centre(E["dino"])], 1)))):
    S = X @ X.T
    p = np.array([S[a, b] for a, b in pos])
    n = np.array([S[a, b] for a, b in neg])
    thr = float(np.quantile(n, 0.97))  # at most 3% of known-different pairs may exceed it
    tp = float((p >= thr).mean())
    TS = (M @ S @ M.T) / np.outer(sz, sz)
    groups = constrained_upgma(TS, [[i for i in range(len(crops)) if M[ci, i]] for ci in range(len(tids))], track_conf, thr)
    sizes = sorted((len(g) for g in groups), reverse=True)
    big = [s for s in sizes if s >= 4]
    print(f"{name:22s} AUC={auc(p, n):.3f}  thr@FPR3%={thr:.3f}  recall={tp:.2f}  clusters>=4 crops: {len(big)}  covering {sum(big)}/{len(crops)} crops; sizes {big[:14]}", flush=True)
