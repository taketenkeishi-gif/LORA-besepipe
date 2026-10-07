"""Option E validation: does the YOLO person mask give the same refined crop box as BiRefNet?  (BiRefNet = 134 s of a 744 s run)

python bbox_iou.py FRAMES_DIR OUT.json [N]
Runs on the 3090 (cuda:0 of the ComfyUI python).  For each single-person crop it computes the refined box both ways, reports IoU stats,
crop-area ratio and the share of cases where the mask box cuts into the BiRefNet box (loses body parts).
"""
import json, os, sys, time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

frames_dir, out_json = Path(sys.argv[1]), Path(sys.argv[2])
N = int(sys.argv[3]) if len(sys.argv) > 3 else 300
COMFY = Path(r"C:\Users\Keishi\Portfolio\Toolchains-Physical\ComfyUI-Portable\ComfyUI_windows_portable\ComfyUI")
MODELS = COMFY / "models"
from ultralytics import YOLO

yolo = YOLO(str(MODELS / "ultralytics" / "segm" / "person_yolov8m-seg.pt"))
sys.path.insert(0, str(COMFY))
os.chdir(COMFY)
from comfy.bg_removal_model import load as load_bg

biref = load_bg(str(MODELS / "background_removal" / "birefnet.safetensors"))
files = sorted(frames_dir.glob("*.png"))
step = max(1, len(files) // N)
files = files[::step][:N]


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u else 0.0


@torch.no_grad()
def bref_box(crop):
    x = torch.from_numpy(np.asarray(crop.convert("RGB"), np.float32) / 255.0).unsqueeze(0)
    m = biref.encode_image(x)
    m = torch.as_tensor(m).float().cpu().reshape(-1, *m.shape[-2:])[0].numpy()
    if m.shape != (crop.height, crop.width):
        m = np.asarray(Image.fromarray((m.clip(0, 1) * 255).astype(np.uint8)).resize(crop.size, Image.BILINEAR), np.float32) / 255.0
    ys, xs = np.nonzero(m > 0.5)
    if not len(xs):
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1], float((m > 0.5).mean())


rows, t_bref, t_mask = [], 0.0, 0.0
for f in files:
    img = Image.open(f).convert("RGB"); W, H = img.size
    r = yolo.predict(np.asarray(img)[:, :, ::-1], conf=0.15, imgsz=960, device=0, retina_masks=True, verbose=False)[0]
    if r.boxes is None or r.masks is None:
        continue
    boxes = r.boxes.xyxy.cpu().numpy().tolist(); masks = r.masks.data.cpu().numpy() > 0.5
    for j, b in enumerate(boxes):
        bw, bh = b[2] - b[0], b[3] - b[1]
        if bw * bh < 0.01 * W * H or min(bw, bh) < 100:
            continue
        if any(k != j and iou(b, o) > 0.0 for k, o in enumerate(boxes)):
            continue  # the pipeline only uses the matte when nobody else is in the crop
        mx, my = bw * 0.10, bh * 0.10
        c0 = [max(0, int(b[0] - mx)), max(0, int(b[1] - my)), min(W, int(b[2] + mx)), min(H, int(b[3] + my))]
        crop = img.crop(c0)
        t = time.perf_counter(); fg = bref_box(crop); torch.cuda.synchronize(); t_bref += time.perf_counter() - t
        if fg is None or not (0.25 < fg[1] < 0.98):
            continue
        t = time.perf_counter()
        mk = masks[j][c0[1]:c0[3], c0[0]:c0[2]]
        ys, xs = np.nonzero(mk)
        mb = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1] if len(xs) else None
        t_mask += time.perf_counter() - t
        if mb is None:
            continue
        PAD = float(os.environ.get('MASK_PAD', '0.0')); pw_, ph_ = (mb[2]-mb[0])*PAD, (mb[3]-mb[1])*PAD
        mb = [max(0, mb[0]-pw_), max(0, mb[1]-ph_), min(crop.width, mb[2]+pw_), min(crop.height, mb[3]+ph_)]
        a = fg[0]
        area = lambda q: (q[2] - q[0]) * (q[3] - q[1])
        # share of the BiRefNet box that the mask box fails to cover (parts that would be cut off)
        ix = max(0, min(a[2], mb[2]) - max(a[0], mb[0])); iy = max(0, min(a[3], mb[3]) - max(a[1], mb[1]))
        rows.append({"iou": iou(a, mb), "area_ratio": area(mb) / area(a), "cut": 1 - ix * iy / area(a)})
arr = lambda k: np.array([r[k] for r in rows])
out = {"crops": len(rows), "frames": len(files), "iou_mean": float(arr("iou").mean()), "iou_p10": float(np.quantile(arr("iou"), 0.1)), "iou_min": float(arr("iou").min()),
       "iou_ge_0.9": float((arr("iou") >= 0.9).mean()), "iou_ge_0.8": float((arr("iou") >= 0.8).mean()), "area_ratio_mean": float(arr("area_ratio").mean()),
       "cut_mean": float(arr("cut").mean()), "cut_p90": float(np.quantile(arr("cut"), 0.9)), "birefnet_ms_per_crop": 1000 * t_bref / len(rows), "mask_ms_per_crop": 1000 * t_mask / len(rows)}
out_json.write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(out, indent=1))
