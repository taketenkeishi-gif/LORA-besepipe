"""Identity-discrimination benchmark: CCIP vs the current CLIP+tag recipe, with a human-curated reference (Nanoha crops picked by eye = positives).

Run with the ComfyUI embedded Python:  python ccip_bench.py WORK_DIR REPORT_DIR POSITIVES_JSON OUT.json
 * WORK_DIR      a finished pipeline run's _work dir (contains crops/ and crops.json)
 * REPORT_DIR    the same run's output dir (char_* folders, used to map output files back to crops)
 * POSITIVES_JSON list of output-image paths that are known to be the target character

Measured (no human in the loop at run time): ROC-AUC of "same character" for pairs (positive,positive) vs (positive,other),
and precision / recall at the best-F1 threshold. Higher AUC = the feature separates the target from everyone else better.
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

work, report_dir, pos_json, out_json = map(Path, sys.argv[1:5])
CCIP_DIR = Path(r"C:\Users\Keishi\Krita\reference_sources\models\ccip_onnx\ccip-caformer-24-randaug-pruned")
crops = json.loads((work / "crops.json").read_text(encoding="utf-8"))
crop_dir = work / "crops"
rep = json.loads((report_dir / "report.json").read_text(encoding="utf-8"))
video = Path(rep["video"]).stem


def stem_of(c: dict) -> str:
    t = float(c["t"])
    return f"{video}_s{int(c['scene']):04d}_{int(t // 60):02d}m{int(t % 60):02d}s_{c['view']}_p{c['file'].split('_p')[-1][:-4]}"


by_stem = {stem_of(c): c for c in crops}
pos_paths = json.loads(pos_json.read_text(encoding="utf-8"))
pos = [by_stem[Path(p).stem] for p in pos_paths if Path(p).stem in by_stem]
pos_ids = {id(c) for c in pos}
print(f"positives mapped: {len(pos)} / {len(pos_paths)}", flush=True)
rng = random.Random(0)
neg_pool = [c for c in crops if id(c) not in pos_ids]
neg = rng.sample(neg_pool, min(700, len(neg_pool)))
sample = pos + neg
labels = np.array([1] * len(pos) + [0] * len(neg))

import torch
import onnxruntime as ort
from transformers import CLIPModel

os.add_dll_directory(str(Path(torch.__file__).parent / "lib"))
providers = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
feat = ort.InferenceSession(str(CCIP_DIR / "model_feat.onnx"), providers=providers)
metr = ort.InferenceSession(str(CCIP_DIR / "model_metrics.onnx"), providers=["CPUExecutionProvider"])
print("CCIP provider:", feat.get_providers()[0], flush=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
clip_dir = next((Path.home() / ".cache/huggingface/hub/models--sentence-transformers--clip-ViT-B-32/snapshots").iterdir()) / "0_CLIPModel"
clip = CLIPModel.from_pretrained(str(clip_dir)).to(dev).eval()
if dev == "cuda":
    clip = clip.half()
CM, CS = np.array([0.48145466, 0.4578275, 0.40821073], np.float32), np.array([0.26862954, 0.26130258, 0.27577711], np.float32)


def prep_ccip(path: Path) -> np.ndarray:
    im = Image.open(path).convert("RGB").resize((384, 384), Image.BILINEAR)
    return ((np.asarray(im, np.float32) / 255.0 - CM) / CS).transpose(2, 0, 1)


def letterbox(img: Image.Image, size: int = 224) -> Image.Image:
    s = size / max(img.size)
    img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.BICUBIC)
    canvas = Image.new("RGB", (size, size), (128, 128, 128))
    canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
    return canvas


def ccip_feats(paths: list[Path]) -> np.ndarray:
    out = []
    for i in range(0, len(paths), 16):
        out.append(feat.run(["output"], {"input": np.stack([prep_ccip(p) for p in paths[i:i + 16]])})[0])
    return np.concatenate(out)


@torch.no_grad()
def clip_feats(paths: list[Path]) -> np.ndarray:
    out = []
    for i in range(0, len(paths), 48):
        x = np.stack([(np.asarray(letterbox(Image.open(p).convert("RGB")), np.float32) / 255.0 - CM) / CS for p in paths[i:i + 48]]).transpose(0, 3, 1, 2)
        v = clip.get_image_features(pixel_values=torch.from_numpy(x).to(dev).half() if dev == "cuda" else torch.from_numpy(x).to(dev))
        v = v if torch.is_tensor(v) else v.pooler_output
        out.append(torch.nn.functional.normalize(v.float(), dim=-1).cpu().numpy())
    return np.concatenate(out)


def ccip_dist(F: np.ndarray) -> np.ndarray:
    return metr.run(["output"], {"input": F.astype(np.float32)})[0]


img_paths = [crop_dir / c["file"] for c in sample]
iso_paths = [crop_dir / c["iso_file"] for c in sample]
has_iso = np.array([c["iso_file"] != c["file"] for c in sample])
print(f"embedding {len(sample)} crops (iso available for {int(has_iso.sum())})", flush=True)
F_ccip = ccip_feats(img_paths)
F_iso = ccip_feats(iso_paths)
F_clip = clip_feats(img_paths)
D_ccip, D_iso = ccip_dist(F_ccip), ccip_dist(F_iso)
C_clip = F_clip @ F_clip.T


def idf_tag_sim() -> np.ndarray:
    ident = []
    for c in sample:
        tags = eval(c["tags"]) if isinstance(c["tags"], str) else c["tags"]
        ident.append({t for t in tags if t.endswith(" hair") or t.endswith(" eyes") or t in ("twintails", "ponytail", "side ponytail", "ahoge", "short hair", "long hair", "medium hair")})
    n = len(sample)
    S = np.zeros((n, n))
    for a in range(n):
        for b in range(a, n):
            u = ident[a] | ident[b]
            S[a, b] = S[b, a] = (len(ident[a] & ident[b]) / len(u)) if u else 0.0
    return S


T = idf_tag_sim()


def zs(M: np.ndarray) -> np.ndarray:
    v = M[np.triu_indices(len(M), 1)]
    return (M - v.mean()) / (v.std() + 1e-9)


# similarity (higher = more alike) for each method
methods = {
    "現行: CLIP + 髪瞳タグ": zs(C_clip) + zs(T),
    "CLIPのみ": C_clip,
    "タグのみ": T,
    "CCIP（そのままの画像）": -D_ccip,
    "CCIP（他の人を塗りつぶした画像）": -D_iso,
    "CCIP（塗りつぶし）+ 髪瞳タグ": zs(-D_iso) + 0.5 * zs(T),
}
iu = np.triu_indices(len(sample), 1)
pairs_pos_pos = [(a, b) for a in range(len(pos)) for b in range(a + 1, len(pos))]
pairs_pos_neg = [(a, b) for a in range(len(pos)) for b in range(len(pos), len(sample))]


def auc(scores_pos: np.ndarray, scores_neg: np.ndarray) -> float:
    allv = np.concatenate([scores_pos, scores_neg])
    order = allv.argsort().argsort() + 1
    r = order[: len(scores_pos)].sum()
    return float((r - len(scores_pos) * (len(scores_pos) + 1) / 2) / (len(scores_pos) * len(scores_neg)))


def best_f1(sp: np.ndarray, sn: np.ndarray) -> dict:
    th = np.quantile(np.concatenate([sp, sn]), np.linspace(0.02, 0.98, 97))
    best = {"f1": 0}
    for t in th:
        tp, fp, fn = (sp >= t).sum(), (sn >= t).sum(), (sp < t).sum()
        p, r = tp / max(1, tp + fp), tp / max(1, tp + fn)
        f1 = 2 * p * r / max(1e-9, p + r)
        if f1 > best["f1"]:
            best = {"f1": float(f1), "precision": float(p), "recall": float(r)}
    return best


results = {}
for name, M in methods.items():
    sp = np.array([M[a, b] for a, b in pairs_pos_pos]); sn = np.array([M[a, b] for a, b in pairs_pos_neg])
    results[name] = {"auc": round(auc(sp, sn), 4), **{k: round(v, 3) for k, v in best_f1(sp, sn).items()}}
    # focus on the crops that actually contain other people (where the mask matters)
    mask_pos = [i for i in range(len(pos)) if has_iso[i]]
    if len(mask_pos) >= 5:
        sp2 = np.array([M[a, b] for a, b in pairs_pos_pos if has_iso[a] and has_iso[b]]); sn2 = np.array([M[a, b] for a, b in pairs_pos_neg if has_iso[a]])
        if len(sp2) and len(sn2):
            results[name]["auc_on_multi_person_crops"] = round(auc(sp2, sn2), 4)
out = {"positives": len(pos), "negatives": len(neg), "iso_available_positives": int(has_iso[: len(pos)].sum()), "ccip_provider": feat.get_providers()[0], "results": results}
out_json.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=1), flush=True)
