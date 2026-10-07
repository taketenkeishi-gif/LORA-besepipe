"""Does a detector put the crop box on the character?  Ground truth = known paste position.

Cut-outs (BiRefNet alpha) of real character images are pasted at random scale/position onto real photos.
For each composite we know the exact character bbox, then measure three ways of finding it:
  yolo-person (person_yolov8m-seg), grounding-dino (text prompt), birefnet (matte bbox).
Metrics: IoU with the truth, and coverage = share of the character's pixels inside the predicted box + margin
(the number that matters for 'crop and train': a box that cuts the character off is worse than a loose box).
"""
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # RTX 3060: leave the 3090 Ti to the user's jobs
os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np
import torch
from PIL import Image

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
random.seed(7)
dev = "cuda"

COMFY = Path(r"C:\Users\Keishi\Portfolio\Toolchains-Physical\ComfyUI-Portable\ComfyUI_windows_portable\ComfyUI\models")
BIREF = next((COMFY / "birefnet" / "models--ZhengPeng7--BiRefNet" / "snapshots").iterdir())
PROJ = Path(r"C:\Users\Keishi\Portfolio\Generation\Training\LoRA-Studio-Next\projects")
sources = sorted((PROJ / "Keytar Fox Character" / "dataset" / "references").glob("*.png"))
sources += sorted((PROJ / "Character Factory Runtime QA" / "dataset" / "references").glob("*.png"))
backgrounds = sorted(Path(r"C:\Windows\Web\Wallpaper").rglob("*.jpg"))[:12]
assert sources and backgrounds, (len(sources), len(backgrounds))
print("sources", len(sources), "backgrounds", len(backgrounds), flush=True)

# ---- BiRefNet matting -------------------------------------------------------------------------
# ComfyUI's own background-removal loader (reads models/background_removal/birefnet.safetensors).
COMFY_ROOT = COMFY.parent
sys.path.insert(0, str(COMFY_ROOT))
os.chdir(COMFY_ROOT)
from comfy.bg_removal_model import load as load_bg

biref = load_bg(str(COMFY / "background_removal" / "birefnet.safetensors"))
assert biref is not None, "birefnet.safetensors not recognized"


@torch.no_grad()
def matte(image: Image.Image) -> np.ndarray:
    rgb = image.convert("RGB")
    x = torch.from_numpy(np.asarray(rgb, np.float32) / 255.0).unsqueeze(0)  # [1,H,W,C] like a ComfyUI IMAGE
    mask = biref.encode_image(x)
    mask = torch.as_tensor(mask).float().cpu().reshape(-1, *mask.shape[-2:])[0].numpy()
    if mask.shape != (rgb.height, rgb.width):
        mask = np.asarray(Image.fromarray((mask.clip(0, 1) * 255).astype(np.uint8)).resize(rgb.size, Image.BILINEAR), np.float32) / 255.0
    return mask


def bbox_of(mask: np.ndarray, thr: float = 0.5):
    ys, xs = np.nonzero(mask > thr)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def iou(a, b):
    if a is None or b is None:
        return 0.0
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def coverage(box, truth_alpha, margin=0.05):
    if box is None:
        return 0.0
    h, w = truth_alpha.shape
    mx, my = (box[2] - box[0]) * margin, (box[3] - box[1]) * margin
    x0, y0, x1, y1 = max(0, int(box[0] - mx)), max(0, int(box[1] - my)), min(w, int(box[2] + mx)), min(h, int(box[3] + my))
    total = truth_alpha.sum()
    return float(truth_alpha[y0:y1, x0:x1].sum() / total) if total else 0.0


# ---- detectors ---------------------------------------------------------------------------------
from ultralytics import YOLO

yolo = YOLO(str(COMFY / "ultralytics" / "segm" / "person_yolov8m-seg.pt"))


def det_yolo(img: Image.Image):
    result = yolo.predict(np.asarray(img.convert("RGB"))[:, :, ::-1], conf=0.15, device=0, verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return None
    boxes = result.boxes
    pick = int(torch.argmax((boxes.xyxy[:, 2] - boxes.xyxy[:, 0]) * (boxes.xyxy[:, 3] - boxes.xyxy[:, 1])))
    return [int(v) for v in boxes.xyxy[pick].tolist()]


def det_matte(img: Image.Image):
    return bbox_of(matte(img), 0.5)


# ---- build cut-outs, then composites ---------------------------------------------------------
cutouts = []
for path in sources:
    image = Image.open(path).convert("RGB")
    alpha = matte(image)
    box = bbox_of(alpha)
    share = float((alpha > 0.5).mean())
    if box is None or not 0.03 < share < 0.9:
        print("skip source (matte not usable)", path.name, round(share, 3), flush=True)
        continue
    rgba = image.copy()
    rgba.putalpha(Image.fromarray((alpha * 255).astype(np.uint8)))
    cutouts.append((path.name, rgba.crop(box), share))
print("cutouts", len(cutouts), flush=True)

rows = []
sheet = []
n = 0
for name, cut, share in cutouts:
    for k in range(5):
        bg = Image.open(random.choice(backgrounds)).convert("RGB")
        W, H = 1280, 720
        bg = bg.resize((W, H), Image.BILINEAR)
        scale = random.uniform(0.25, 0.8) * H / cut.height
        c = cut.resize((max(8, int(cut.width * scale)), max(8, int(cut.height * scale))), Image.LANCZOS)
        if random.random() < 0.5:
            c = c.transpose(Image.FLIP_LEFT_RIGHT)
        x, y = random.randint(0, max(0, W - c.width)), random.randint(0, max(0, H - c.height))
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        layer.paste(c, (x, y))
        comp = Image.alpha_composite(bg.convert("RGBA"), layer).convert("RGB")
        truth_alpha = np.asarray(layer.getchannel("A"), np.float32) / 255.0
        truth = bbox_of(truth_alpha, 0.5)
        if truth is None:
            continue
        row = {"source": name, "truth": truth, "truth_frac": round((truth[2] - truth[0]) * (truth[3] - truth[1]) / (W * H), 3)}
        for label, fn in (("yolo", det_yolo), ("matte", det_matte)):
            t0 = time.time()
            box = fn(comp)
            row[label] = {"box": box, "iou": round(iou(box, truth), 3), "cover": round(coverage(box, truth_alpha), 3), "sec": round(time.time() - t0, 2)}
        rows.append(row)
        n += 1
        if n % 5 == 0:
            print("done", n, flush=True)
        if len(sheet) < 12:
            sheet.append((comp, truth, row))

(OUT / "rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
print("\n=== summary over", len(rows), "composites ===")
summary = {}
for label in ("yolo", "matte"):
    ious = np.array([r[label]["iou"] for r in rows])
    covs = np.array([r[label]["cover"] for r in rows])
    miss = sum(r[label]["box"] is None for r in rows)
    summary[label] = {"mean_iou": round(float(ious.mean()), 3), "iou>=0.8": round(float((ious >= 0.8).mean()), 3), "iou>=0.5": round(float((ious >= 0.5).mean()),3),
                      "mean_cover": round(float(covs.mean()), 3), "cover>=0.97": round(float((covs >= 0.97).mean()), 3), "no_detection": miss}
    print(label, summary[label])
(OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

# contact sheet: truth (green) + three predictions
from PIL import ImageDraw

tiles = []
for comp, truth, row in sheet:
    d = ImageDraw.Draw(comp)
    d.rectangle(truth, outline=(0, 255, 0), width=5)
    for label, color in (("yolo", (255, 0, 0)), ("matte", (0, 160, 255))):
        if row[label]["box"]:
            d.rectangle(row[label]["box"], outline=color, width=3)
    tiles.append(comp.resize((426, 240)))
cols = 4
canvas = Image.new("RGB", (426 * cols, 240 * ((len(tiles) + cols - 1) // cols)))
for i, t in enumerate(tiles):
    canvas.paste(t, ((i % cols) * 426, (i // cols) * 240))
canvas.save(OUT / "sheet.jpg", quality=80)
print("sheet", OUT / "sheet.jpg")
