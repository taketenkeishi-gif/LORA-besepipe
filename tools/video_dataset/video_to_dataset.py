"""Prototype v3: video -> per-character / per-outfit dataset candidates (GPU, quality-filtered, size-normalised, diversity-aware).

scene cuts -> several quality-checked frames per scene -> person detection with masks (YOLO seg)
-> per person a crop around THAT person (``--others`` policy decides what happens to other characters in the crop)
-> person tracks inside a scene (a back view / cut-off body inherits the identity of the same person seen with a face)
-> long side normalised -> WD14 tags on the focus-only image (others painted gray) -> cluster characters and outfits
-> char_NN_*/outfit_NN_*/  (TXT = trigger words + tags).  Views (front/side/back x close/upper/cowboy/full/cut-off) are
   reported per character and never de-duplicated across different views.   The source video is never modified.
"""
import argparse
import csv
import io
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("GPU", "1")  # PCI index 1 = RTX 3090 Ti, 0 = RTX 3060

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter

torch_lib = Path(torch.__file__).parent / "lib"  # CUDA 13 / cuDNN 9 DLLs that onnxruntime-gpu needs
os.add_dll_directory(str(torch_lib))
os.environ["PATH"] = str(torch_lib) + os.pathsep + os.environ["PATH"]

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("out")
ap.add_argument("--limit-seconds", type=float, default=0)
ap.add_argument("--threshold", type=float, default=0.3, help="scene-cut sensitivity (smaller = more cuts)")
ap.add_argument("--min-scene", type=float, default=0.8)
ap.add_argument("--frame-interval", type=float, default=1.5, help="one sampled frame per this many seconds of a scene")
ap.add_argument("--max-frames-per-scene", type=int, default=6)
ap.add_argument("--long-side", type=int, default=1024)
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--others", choices=["keep", "mask", "exclude"], default="keep",
                help="other characters inside a crop: keep = crop the whole focus character, others may be partly visible (default); "
                     "mask = paint other characters gray in the saved image; exclude = drop crops that contain other characters")
ap.add_argument("--overlap-threshold", type=float, default=0.25, help="share of another person that must lie inside the crop to count as 'present'")
ap.add_argument("--margin", type=float, default=0.10, help="extra space around the focus person (fraction of its box)")
ap.add_argument("--min-area", type=float, default=0.02, help="smallest person box kept, fraction of the frame")
ap.add_argument("--min-short", type=int, default=140, help="smallest short side (px) of a crop")
ap.add_argument("--min-char-crops", type=int, default=4)
ap.add_argument("--use-names", action="store_true", help="OPTIONAL: use character names the tagger already knows as seeds (unknown characters never depend on this)")
ap.add_argument("--merge-quantile", type=float, default=0.90, help="link two shots when their score beats this quantile of known-different people in the same frame (higher = fewer, purer groups)")
ap.add_argument("--group-merge-quantile", type=float, default=0.55, help="second pass: merge fragments of one person by average similarity beating this quantile of known-different people in the same frame (0 = off)")
ap.add_argument("--rescue-quantile", type=float, default=0.55, help="leftover crops join the closest group when their best score beats this quantile (lower = fewer unassigned)")
ap.add_argument("--yolo-conf", type=float, default=0.25, help="person detector confidence floor (lower finds more anime figures, and more false ones)")
ap.add_argument("--cascade-margin", type=float, default=0.8, help="a leftover joins a character only if it beats the runner-up character by this much (z-score units)")
ap.add_argument("--early-dup-bits", type=int, default=1, help="skip a crop BEFORE the heavy steps when its small-image hash differs from the previous kept crop of the same track by at most this many bits (0 = off)")
ap.add_argument("--refine", choices=["mask", "birefnet"], default="mask", help="how to tighten a single-person crop: YOLO's person mask (fast) or the BiRefNet matte")
ap.add_argument("--start-seconds", type=float, default=0.0, help="process only this region of the video (a bucket): from here")
ap.add_argument("--end-seconds", type=float, default=0.0, help="...up to here (0 = to the end). File names / timestamps stay ABSOLUTE video time")
ap.add_argument("--assign-threshold", type=float, default=0.90, help="CLIP cosine needed to attach a crop with no recognised name to a named character")
ap.add_argument("--name-min", type=int, default=8, help="a recognised character name must appear on at least this many crops to be used as a character")
ap.add_argument("--progress-file", default="", help="write a small JSON {stage,percent,detail,updated_at} here at every stage / progress step (used by the app)")
args = ap.parse_args()

# ---------------------------------------------------------------- progress reporting (optional)
# percent bands: scene detect 0-5, frame choice 5-30, model load 30-35, person crops 35-75, tagging 75-85, clustering 85-90, writing 90-100
_progress_lock = threading.Lock()
_progress_last = [0.0, ""]


def progress(stage: str, percent: float, detail: str = "", force: bool = False) -> None:
    if not args.progress_file:
        return
    now = time.time()
    with _progress_lock:
        if not force and stage == _progress_last[1] and now - _progress_last[0] < 0.5:
            return
        _progress_last[0], _progress_last[1] = now, stage
        payload = {"stage": stage, "percent": int(max(0, min(100, round(percent)))), "detail": detail, "updated_at": now}
        try:
            tmp = args.progress_file + ".tmp"
            Path(tmp).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, args.progress_file)
        except OSError:
            pass  # progress is best-effort; never break the run


def _fail_hook(exc_type, exc, tb):
    import traceback
    traceback.print_exception(exc_type, exc, tb)
    print(f"[video_dataset] ERROR: {exc_type.__name__}: {exc}", file=sys.stderr, flush=True)
    progress("エラー", 0, f"{exc_type.__name__}: {exc}"[:300], force=True)


sys.excepthook = _fail_hook

_default_backend = Path(__file__).resolve().parents[2] / "backend"
BACKEND = _default_backend if _default_backend.is_dir() else Path(r"C:\Users\Keishi\Portfolio\Generation\Training\LoRA-Studio-Next\backend")
sys.path.insert(0, str(BACKEND))
COMFY_ROOT = Path(os.environ.get("COMFY_ROOT") or r"C:\Users\Keishi\Portfolio\Toolchains-Physical\ComfyUI-Portable\ComfyUI_windows_portable\ComfyUI")
MODELS = COMFY_ROOT / "models"
WD14_DIR = COMFY_ROOT / "custom_nodes" / "comfyui-wd14-tagger" / "models"
video = Path(args.video)
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
work = Path(os.environ.get("WORK_DIR", out / "_work"))
frame_dir, crop_dir = work / "frames", work / "crops"
for d in (frame_dir, crop_dir):
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)

timers: dict[str, list] = defaultdict(lambda: [0.0, 0])


@contextmanager
def timed(name: str, count: int = 1):
    start = time.perf_counter()
    try:
        yield
    finally:
        timers[name][0] += time.perf_counter() - start
        timers[name][1] += count


wall0 = time.perf_counter()
from app.services import frame_quality as fq  # noqa: E402
from app.services import video_scenes as vs  # noqa: E402

ffmpeg = vs._ffmpeg()

# ---------------------------------------------------------------- scenes
progress("シーン検出", 0, "動画を解析しています", force=True)
with timed("1 シーン検出（動画全体）"):
    total_duration = vs.probe_duration(video)
    region_start = max(0.0, args.start_seconds)
    region_end = min(total_duration, args.end_seconds) if args.end_seconds else total_duration
    duration = max(0.0, region_end - region_start)
    if args.limit_seconds:
        duration = min(duration, args.limit_seconds)
    cmd = [ffmpeg, "-hide_banner", "-nostats"] + (["-ss", f"{region_start:.3f}"] if region_start else []) + (["-t", f"{duration:.3f}"] if (args.limit_seconds or args.end_seconds or region_start) else [])
    cmd += ["-i", str(video), "-an", "-sn", "-vf", f"scale=320:-2,select='gt(scene,{args.threshold})',showinfo", "-f", "null", "-"]
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    cuts = sorted({round(float(m.group(1)), 3) for m in vs._PTS_TIME.finditer(done.stderr or "")})
    # scenes are built in region time, then shifted back to absolute video time (frame grabs and file names use absolute time)
    scenes = [(a + region_start, b + region_start) for a, b in vs.build_scenes(cuts, duration, args.min_scene)]
print(f"video {duration:.0f}s -> {len(scenes)} scenes", flush=True)
progress("フレーム選定", 5, f"{len(scenes)} シーン", force=True)

# ---------------------------------------------------------------- frame choice (several per scene, quality filtered, parallel)
rej_lock = threading.Lock()
rej_dir = out / "_画質フィルタで除外したフレーム（例・最大40枚）"
shutil.rmtree(rej_dir, ignore_errors=True)
rej_dir.mkdir(parents=True)


def _unlink_retry(f: Path) -> None:
    """Sync clients (OneDrive) and scanners briefly lock fresh files; a leftover temp frame is harmless, a crash is not."""
    for attempt in range(6):
        try:
            f.unlink(missing_ok=True)
            return
        except PermissionError:
            time.sleep(0.3 * (attempt + 1))


def burst(tag: str, t: float):
    # 3 consecutive frames piped as uncompressed BMP (self-sized): no PNG encode/decode, no temp files (was ~1/3 of frame choice)
    r = subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "3", "-f", "image2pipe", "-vcodec", "bmp", "-"],
                       capture_output=True, timeout=180)
    data, images, pos = r.stdout, [], 0
    while pos + 6 <= len(data) and data[pos:pos + 2] == b"BM":
        size = int.from_bytes(data[pos + 2:pos + 6], "little")
        if size <= 0 or pos + size > len(data):
            break
        images.append(Image.open(io.BytesIO(data[pos:pos + size])).convert("RGB"))
        pos += size
    if len(images) < 2:
        return None
    return images[:3] if len(images) >= 3 else [images[0], images[1], images[1]]


_FPS = re.compile(r"Video:.*?(\d+(?:\.\d+)?) fps")
_vfps = [None]


def video_fps() -> float:
    if _vfps[0] is None:
        err = subprocess.run([ffmpeg, "-hide_banner", "-i", str(video)], capture_output=True, text=True, timeout=60).stderr or ""
        m = _FPS.search(err)
        _vfps[0] = float(m.group(1)) if m else 0.0
    return _vfps[0]


def scene_bursts(start: float, end: float, wanted: list[float]):
    """Decode the scene ONCE (one ffmpeg, frames streamed as BMP) and yield, for each wanted time in order, a function t -> 3-frame burst at t.
    Same frames as seeking to t and taking 3 frames, without starting ffmpeg and seeking again for every candidate."""
    fps = video_fps()
    lo, hi = min(wanted) - 0.08, max(wanted) + 0.15
    s = max(start, lo)
    proc = subprocess.Popen([ffmpeg, "-hide_banner", "-loglevel", "error", "-ss", f"{s:.3f}", "-t", f"{max(0.05, hi - s) + 4 / fps:.3f}", "-i", str(video),
                             "-f", "image2pipe", "-vcodec", "bmp", "-"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    buf: dict[int, Image.Image] = {}
    nxt = [0]
    eof = [False]
    keep_from = [0]  # frames before this index are never asked for again: decode past them without keeping them

    def read_until(idx: int) -> None:
        while not eof[0] and nxt[0] <= idx:
            head = proc.stdout.read(6)
            if len(head) < 6 or head[:2] != b"BM":
                eof[0] = True
                break
            size = int.from_bytes(head[2:6], "little")
            body = proc.stdout.read(size - 6)
            if len(body) < size - 6:
                eof[0] = True
                break
            if nxt[0] >= keep_from[0]:
                buf[nxt[0]] = Image.open(io.BytesIO(head + body)).convert("RGB")
            nxt[0] += 1

    def grab(t: float):
        i0 = max(0, round((t - s) * fps))
        read_until(i0 + 2)
        imgs = [buf[j] for j in (i0, i0 + 1, i0 + 2) if j in buf]
        if len(imgs) < 2:
            return None
        return imgs if len(imgs) == 3 else [imgs[0], imgs[1], imgs[1]]

    try:
        for t0 in wanted:  # wanted times increase; the earliest candidate of t0 is t0-0.07
            keep_from[0] = max(0, round((t0 - 0.08 - s) * fps))
            for j in [j for j in buf if j < keep_from[0]]:
                del buf[j]
            yield grab
    finally:
        proc.kill()
        proc.wait()


def choose_frames(item):
    i, (start, end) = item
    length = end - start
    n = max(1, min(args.max_frames_per_scene, round(length / args.frame_interval)))
    times = [start + length * 0.5] if n == 1 else [start + length * (0.12 + 0.76 * j / (n - 1)) for j in range(n)]
    kept, tried_bad, reasons = [], 0, Counter()
    streamed = scene_bursts(start, end, times) if video_fps() > 0 else None
    for k, t0 in enumerate(times):
        grab = next(streamed) if streamed is not None else None
        for attempt, dt in enumerate((0.0, 0.07, -0.07, 0.14)):  # a neighbouring frame is usually fine when this one is damaged
            t = min(max(start, t0 + dt), max(start, end - 0.05))
            b = grab(t) if grab is not None else burst(f"s{i:04d}_{k}_{attempt}", t)
            if b is None:
                continue
            m = fq.assess(*b)
            fl = fq.flags(m)
            if fl:
                tried_bad += 1
                reasons.update(fl)
                with rej_lock:
                    example = len(list(rej_dir.glob("*.jpg"))) < 40
                if example:
                    b[1].save(rej_dir / f"s{i:04d}_{int(t // 60):02d}m{int(t % 60):02d}s_{'-'.join(fl)}.jpg", quality=90)
                continue
            path = frame_dir / f"scene_{i:04d}_{k}.png"
            b[1].save(path, compress_level=1)  # temp frame: fast compression (same pixels)
            kept.append({"scene": i, "k": k, "t": round(t, 3), "file": path.name, "sharp": m["sharpness"]})
            break
    if streamed is not None:
        streamed.close()
    with rej_lock:
        frames_done[0] += 1
        progress("フレーム選定", 5 + 25 * frames_done[0] / max(1, len(scenes)), f"{frames_done[0]}/{len(scenes)} シーン")
    return {"scene": i, "frames": kept, "bad_candidates": tried_bad, "reasons": dict(reasons), "wanted": n}


frames_done = [0]


with timed("2 フレーム選定（1シーン複数枚・前後フレームで画質判定）", len(scenes)):
    with ThreadPoolExecutor(args.workers) as pool:
        picks = list(pool.map(choose_frames, list(enumerate(scenes, start=1))))
frames = [f for p in picks for f in p["frames"]]
bad_total = sum(p["bad_candidates"] for p in picks)
lost = sum(p["wanted"] - len(p["frames"]) for p in picks)
print(f"frames kept {len(frames)} (wanted {sum(p['wanted'] for p in picks)}, lost to damage {lost}); damaged candidates tried {bad_total}", flush=True)

# ---------------------------------------------------------------- models
progress("モデル読込", 30, "初回は数十秒かかります", force=True)
with timed("0 モデル読込（初回のみ）"):
    from ultralytics import YOLO

    yolo = YOLO(str(MODELS / "ultralytics" / "segm" / "person_yolov8m-seg.pt"))
    sys.path.insert(0, str(COMFY_ROOT))
    cwd = os.getcwd()
    os.chdir(COMFY_ROOT)
    from comfy.bg_removal_model import load as load_bg

    biref = load_bg(str(MODELS / "background_removal" / "birefnet.safetensors")) if args.refine == "birefnet" else None
    from spandrel import ModelLoader

    upscaler = ModelLoader().load_from_file(str(MODELS / "upscale_models" / "4x-UltraSharp.pth")).model.eval().cuda().half()
    os.chdir(cwd)
    import onnxruntime as ort

    sess = ort.InferenceSession(str(WD14_DIR / "wd-eva02-large-tagger-v3.onnx"), providers=[("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"])
    assert sess.get_providers()[0] == "CUDAExecutionProvider", sess.get_providers()
    wd_tags = list(csv.DictReader(open(WD14_DIR / "wd-eva02-large-tagger-v3.csv", encoding="utf-8", newline="")))
    wd_input = sess.get_inputs()[0].name
    from transformers import CLIPModel

    clip_dir = next((Path.home() / ".cache/huggingface/hub/models--sentence-transformers--clip-ViT-B-32/snapshots").iterdir()) / "0_CLIPModel"
    clip = CLIPModel.from_pretrained(str(clip_dir)).cuda().half().eval()
    CLIP_MEAN, CLIP_STD = np.array([0.4815, 0.4578, 0.4082], np.float32), np.array([0.2686, 0.2613, 0.2758], np.float32)


def letterbox(img: Image.Image, size: int = 224) -> Image.Image:
    img = img.convert("RGB")
    s = size / max(img.size)
    img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.BICUBIC)
    canvas = Image.new("RGB", (size, size), (128, 128, 128))
    canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
    return canvas


@torch.no_grad()
def clip_embed(images):
    x = np.stack([(np.asarray(letterbox(im), np.float32) / 255.0 - CLIP_MEAN) / CLIP_STD for im in images]).transpose(0, 3, 1, 2)
    v = clip.get_image_features(pixel_values=torch.from_numpy(x).cuda().half())
    v = v if torch.is_tensor(v) else v.pooler_output
    return torch.nn.functional.normalize(v.float(), dim=-1).cpu().numpy()


def tag_batch(images, thresh=0.35):
    arr = np.stack([np.array(im.convert("RGB").resize((448, 448), Image.LANCZOS), dtype=np.float32)[:, :, ::-1] for im in images])
    preds = sess.run(None, {wd_input: arr})[0]
    result = []
    for row in preds:
        names = []
        for i, tag in enumerate(wd_tags):
            if i >= len(row):
                break
            cat = int(tag.get("category", 9))
            if cat in (9, 5):
                continue
            if row[i] >= (0.85 if cat == 4 else thresh):
                names.append(tag["name"].replace("_", " "))
        result.append(names)
    return result


@torch.no_grad()
def foreground_bbox(crop: Image.Image):
    x = torch.from_numpy(np.asarray(crop.convert("RGB"), np.float32) / 255.0).unsqueeze(0)
    mask = biref.encode_image(x)
    mask = torch.as_tensor(mask).float().cpu().reshape(-1, *mask.shape[-2:])[0].numpy()
    if mask.shape != (crop.height, crop.width):
        mask = np.asarray(Image.fromarray((mask.clip(0, 1) * 255).astype(np.uint8)).resize(crop.size, Image.BILINEAR), np.float32) / 255.0
    ys, xs = np.nonzero(mask > 0.5)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1], float((mask > 0.5).mean())


@torch.no_grad()
def normalise_size(img: Image.Image, long_side: int) -> Image.Image:
    if long_side <= 0:  # 0 = keep the original crop size
        return img
    w, h = img.size
    long = max(w, h)
    if long < long_side / 1.5:
        x = torch.from_numpy(np.asarray(img.convert("RGB"), np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).cuda().half()
        y = upscaler(x).clamp(0, 1).squeeze(0).permute(1, 2, 0).float().cpu().numpy()
        img = Image.fromarray((y * 255).round().astype(np.uint8))
        w, h = img.size
        long = max(w, h)
    if long != long_side:
        s = long_side / long
        nw = max(8, round(w * s / 8) * 8) if w >= h else max(8, round(w * s))
        nh = max(8, round(h * s / 8) * 8) if h > w else max(8, round(h * s))
        img = img.resize((nw, nh), Image.LANCZOS)
    return img


def dhash(img: Image.Image) -> int:
    g = np.asarray(img.convert("L").resize((9, 8), Image.BILINEAR), np.float32)
    return int("".join("1" if b else "0" for b in (g[:, 1:] > g[:, :-1]).flatten()), 2)


def box_iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


# ---------------------------------------------------------------- detection -> crops, tracks per scene
import cv2  # noqa: E402

_DILATE9 = np.ones((9, 9), np.uint8)
save_pool = ThreadPoolExecutor(4)
save_jobs: list = []
crops = []
funnel = {"detections": 0, "dropped_tiny_box": 0, "dropped_below_min_area_or_short": 0, "dropped_others_excluded": 0, "dropped_short_after_refine": 0, "skipped_near_duplicate_early": 0}
last_hash_by_track: dict = {}
track_counter = 0
prev_scene, prev_people = None, []
progress("人物検出", 35, f"{len(frames)} フレーム", force=True)
for frame_no, item in enumerate(frames):
    progress("人物検出", 35 + 40 * frame_no / max(1, len(frames)), f"{frame_no}/{len(frames)} フレーム（人物 {len(crops)}）")
    with timed("3a フレーム読込（1フレーム）"):
        image = Image.open(frame_dir / item["file"]).convert("RGB")
    W, H = image.size
    with timed("3 人物検出＋人物マスク（YOLO seg・1フレーム）"):
        res = yolo.predict(np.asarray(image)[:, :, ::-1], conf=args.yolo_conf, imgsz=960, device=0, retina_masks=True, verbose=False)[0]
    with timed("3b マスクをCPUへ（1フレーム）"):
        boxes = [] if res.boxes is None else res.boxes.xyxy.cpu().numpy().tolist()
        confs = [] if res.boxes is None else res.boxes.conf.cpu().numpy().tolist()
        masks = [] if res.masks is None else (res.masks.data.cpu().numpy() > 0.5)
    funnel["detections"] += len(boxes)
    people = [{"box": b, "conf": c, "mask": masks[j] if len(masks) > j else None} for j, (b, c) in enumerate(zip(boxes, confs))
              if (b[2] - b[0]) * (b[3] - b[1]) >= 0.005 * W * H]
    funnel["dropped_tiny_box"] += len(boxes) - len(people)
    # tracking: link each person to the best-overlapping person of the previous sampled frame of the SAME scene
    if item["scene"] != prev_scene:
        prev_people = []
    taken = set()
    for p in people:
        best, best_iou = None, 0.2
        for q_i, q in enumerate(prev_people):
            if q_i in taken:
                continue
            v = box_iou(p["box"], q["box"])
            if v > best_iou:
                best, best_iou = q_i, v
        if best is None:
            track_counter += 1
            p["track"] = track_counter
        else:
            taken.add(best)
            p["track"] = prev_people[best]["track"]
    prev_scene, prev_people = item["scene"], people

    for n, me in enumerate(people):
        box, conf = me["box"], me["conf"]
        bw, bh = box[2] - box[0], box[3] - box[1]
        if bw * bh < args.min_area * W * H or min(bw, bh) < args.min_short:
            funnel["dropped_below_min_area_or_short"] += 1
            continue
        mx, my = bw * args.margin, bh * args.margin
        c0 = [max(0, int(box[0] - mx)), max(0, int(box[1] - my)), min(W, int(box[2] + mx)), min(H, int(box[3] + my))]

        def covered(o):
            ix = max(0, min(c0[2], o["box"][2]) - max(c0[0], o["box"][0]))
            iy = max(0, min(c0[3], o["box"][3]) - max(c0[1], o["box"][1]))
            return ix * iy / max(1, (o["box"][2] - o["box"][0]) * (o["box"][3] - o["box"][1]))

        others = [o for o in people if o is not me and covered(o) > args.overlap_threshold]
        if others and args.others == "exclude":
            funnel["dropped_others_excluded"] += 1
            continue
        if args.early_dup_bits > 0:  # a still shot gives the same crop frame after frame: the heavy steps (matte, upscaler, tagger) would only be repeated
            hnow = dhash(image.crop(c0))
            prev = last_hash_by_track.get(me["track"])
            if prev is not None and bin(hnow ^ prev).count("1") <= args.early_dup_bits:
                funnel["skipped_near_duplicate_early"] += 1
                continue
            last_hash_by_track[me["track"]] = hnow
        crop = image.crop(c0)
        final, refined = c0, False
        if not others:  # background matte only helps when nobody else stands in the crop
            with timed("4 背景マスクで切り抜き範囲を補正（1人物）"):
                if args.refine == "mask" and me["mask"] is not None:
                    # YOLO's own person mask: measured on 189 crops, box +10% leaves 0.09% of the BiRefNet box uncovered, 4 ms vs 262 ms
                    mk = me["mask"][c0[1]:c0[3], c0[0]:c0[2]]
                    ys, xs = np.nonzero(mk)
                    fg = ([int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1], float(mk.mean())) if len(xs) else None
                    pad = 0.10
                else:
                    fg = foreground_bbox(crop)
                    pad = 0.06
            if fg is not None and 0.25 < fg[1] < 0.98:
                b = fg[0]
                pw, ph = (b[2] - b[0]) * pad, (b[3] - b[1]) * pad
                final = [max(0, int(c0[0] + b[0] - pw)), max(0, int(c0[1] + b[1] - ph)), min(W, int(c0[0] + b[2] + pw)), min(H, int(c0[1] + b[3] + ph))]
                refined = True
        cimg = image.crop(final)
        if min(cimg.size) < args.min_short:
            funnel["dropped_short_after_refine"] += 1
            continue
        # focus-only view: other people painted gray (used for tagging; saved as the image only with --others mask)
        isolated = cimg
        if others and me["mask"] is not None:
          with timed("4b 他の人を塗りつぶし（1人物）"):
            arr = np.array(cimg)
            other_mask = np.zeros(arr.shape[:2], bool)
            for o in others:
                if o["mask"] is not None:
                    other_mask |= o["mask"][final[1]:final[3], final[0]:final[2]]
            other_mask &= ~me["mask"][final[1]:final[3], final[0]:final[2]]
            grown = cv2.dilate(other_mask.astype(np.uint8), _DILATE9) > 0  # == PIL MaxFilter(9), ~50x faster
            arr[grown & ~me["mask"][final[1]:final[3], final[0]:final[2]]] = 128
            isolated = Image.fromarray(arr)
        saved = isolated if args.others == "mask" else cimg
        orig_size = saved.size
        with timed("5 サイズ統一（長辺指定・アップスケーラー併用・1枚）"):
            saved_n = normalise_size(saved, args.long_side)
            isolated_n = saved_n if saved is isolated else normalise_size(isolated, args.long_side)
        name = f"s{item['scene']:04d}_k{item['k']}_p{n}.png"
        iso_name = name if isolated_n is saved_n else name.replace(".png", "_iso.png")
        with timed("5b 切り抜きをPNG保存（1人物・別スレッドへ渡すだけ）"):  # zlib releases the GIL: compress while the GPU works on the next frame
            save_jobs.append(save_pool.submit(saved_n.save, crop_dir / name))
            if iso_name != name:
                save_jobs.append(save_pool.submit(isolated_n.save, crop_dir / iso_name))
        crops.append({"file": name, "iso_file": iso_name, "scene": item["scene"], "k": item["k"], "t": item["t"], "track": me["track"], "yolo_conf": round(conf, 3), "refined": refined,
                      "orig_size": orig_size, "others": len(others), "size": saved_n.size, "sharp": item["sharp"], "box": final,
                      "partial": bool(box[0] <= 2 or box[1] <= 2 or box[2] >= W - 2 or box[3] >= H - 2)})
with timed("5c PNG保存の完了待ち"):
    for j in save_jobs:
        j.result()  # re-raises a failed save
    save_pool.shutdown()
print("person crops", len(crops), "tracks", track_counter, "with other characters inside:", sum(c["others"] > 0 for c in crops), flush=True)

BATCH = 16
progress("タグ判定", 75, f"{len(crops)} 人物", force=True)
for s in range(0, len(crops), BATCH):
    progress("タグ判定", 75 + 10 * s / max(1, len(crops)), f"{s}/{len(crops)} 人物")
    chunk = crops[s:s + BATCH]
    images = [Image.open(crop_dir / c["iso_file"]) for c in chunk]
    with timed("6 タグ判定（WD14 eva02-large・GPU・1枚）", len(chunk)):
        tags = tag_batch(images)
    with timed("6b 見た目の特徴量（CLIP埋め込み・GPU・1枚）", len(chunk)):
        emb_chunk = clip_embed(images)
    for c, t, im, e in zip(chunk, tags, images, emb_chunk):
        c["tags"] = t
        c["emb"] = e
        c["hash"] = dhash(Image.open(crop_dir / c["file"]))

# ---------------------------------------------------------------- views (composition) from tags
def view_of(tags):
    s = set(tags)
    if "from behind" in s:
        ori = "back"
    elif "from side" in s or "profile" in s:
        ori = "side"
    else:
        ori = "front"
    if "head out of frame" in s:
        fr = "head-cut"
    elif "feet out of frame" in s and "full body" not in s and "cowboy shot" not in s and "upper body" not in s:
        fr = "lower-cut"
    elif "close-up" in s or "portrait" in s or "face" in s:
        fr = "close"
    elif "upper body" in s:
        fr = "upper"
    elif "cowboy shot" in s:
        fr = "cowboy"
    elif "full body" in s:
        fr = "full"
    else:
        fr = "other"
    return ori, fr


for c in crops:
    c["view"] = "-".join(view_of(c["tags"]))

# ---------------------------------------------------------------- clustering (identity inherited along tracks)
HAIR_COLOR = re.compile(r"^(blonde|brown|black|blue|red|pink|purple|green|white|silver|grey|gray|orange|aqua|light blue|dark blue|light brown|dark green|light purple|multicolored|two-tone|gradient|streaked|colored inner) hair$")
EYE_COLOR = re.compile(r"^(?!closed|half-closed|wide|empty|crossed|one|heterochromia|torn|big|narrowed|rolling|looking)(.+) eyes$")
FEATURES = {"animal ears", "cat ears", "fox ears", "dog ears", "rabbit ears", "tail", "fox tail", "cat tail", "horns", "glasses", "twintails", "ponytail", "braid",
            "twin braids", "side ponytail", "short hair", "long hair", "medium hair", "very long hair", "ahoge", "hair bun", "drill hair", "pointy ears", "elf", "halo", "wings"}
CLOTH_NOUN = re.compile(r"\b(shirt|skirt|dress|uniform|jacket|hoodie|sweater|coat|pants|shorts|swimsuit|bikini|kimono|yukata|pajamas|armor|cape|cloak|apron|vest|cardigan|leotard|bodysuit|jeans|tank top|serafuku|necktie|bowtie|blouse|gown|robe|miniskirt|pleated skirt|t-shirt|sailor collar|maid|bra|panties|thighhighs|pantyhose|kneehighs|socks|boots|gloves)\b")


def ident_features(tags):
    feats = {}
    for t in tags:
        if HAIR_COLOR.match(t):
            feats[t] = 3.0
        elif EYE_COLOR.match(t):
            feats[t] = 2.0
        elif t in FEATURES:
            feats[t] = 1.0
    return feats


def outfit_features(tags):
    return {t: 1.0 for t in tags if CLOTH_NOUN.search(t)}


def wjaccard(a, b):
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    return sum(min(a.get(k, 0), b.get(k, 0)) for k in keys) / sum(max(a.get(k, 0), b.get(k, 0)) for k in keys)


def overlap_coeff(a, b):
    if not a or not b:
        return 0.0
    return len(set(a) & set(b)) / min(len(a), len(b))


def cluster(feats, max_dist, sim_fn=wjaccard):
    groups = [[i] for i in range(len(feats))]
    sim = np.array([[sim_fn(feats[i], feats[j]) for j in range(len(feats))] for i in range(len(feats))]) if feats else np.zeros((0, 0))
    while len(groups) > 1:
        best, pair = -1.0, None
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                s = float(np.mean([sim[i, j] for i in groups[a] for j in groups[b]]))
                if s > best:
                    best, pair = s, (a, b)
        if best < 1 - max_dist:
            break
        a, b = pair
        groups[a] += groups.pop(b)
    return groups


def constrained_upgma(sim: np.ndarray, members: list[list[int]], conflict: np.ndarray, thr: float):
    """Average-linkage merging of clusters. ``conflict[a,b]`` = the two clusters contain different people of one frame (never merge)."""
    k = len(members)
    sizes = np.array([len(m) for m in members], float)
    C = sim.copy()
    conf = conflict.copy()
    alive = np.ones(k, bool)
    blocked = 0
    np.fill_diagonal(C, -1)
    while True:
        masked = np.where(alive[:, None] & alive[None, :] & ~conf, C, -1)
        np.fill_diagonal(masked, -1)
        idx = int(masked.argmax())
        a, b = divmod(idx, k)
        if masked[a, b] < thr:
            break
        new_row = (sizes[a] * C[a] + sizes[b] * C[b]) / (sizes[a] + sizes[b])
        C[a, :] = new_row
        C[:, a] = new_row
        C[a, a] = -1
        conf[a, :] |= conf[b, :]
        conf[:, a] |= conf[:, b]
        members[a] = members[a] + members[b]
        sizes[a] += sizes[b]
        alive[b] = False
    return [m for m, ok in zip(members, alive) if ok]


progress("仕分け", 85, "キャラクターを分類しています", force=True)
with timed("7 キャラ・衣装の仕分け（タグ＋CLIPの類似度・追跡・同一フレーム別人の制約・単連結）", max(1, len(crops))):
    E = np.stack([c["emb"] for c in crops]) if crops else np.zeros((0, 512))
    n_c = len(crops)
    by_track = defaultdict(list)
    by_frame = defaultdict(list)
    for i, c in enumerate(crops):
        by_track[c["track"]].append(i)
        by_frame[(c["scene"], c["k"])].append(i)
    label = [None] * n_c
    source = [None] * n_c
    named_groups = {}
    report_calibration = {"mode": "unsupervised (no character names used)" if not args.use_names else "character-name seeds + unsupervised for the rest"}

    if args.use_names:  # OPTIONAL booster for characters the tagger already knows; unknown characters never depend on it
        cat4 = {m["name"].replace("_", " ") for m in wd_tags if int(m.get("category", 9)) == 4}
        name_count = Counter(t for c in crops for t in c["tags"] if t in cat4)

        def top_name(c):
            names = [t for t in c["tags"] if t in cat4 and name_count[t] >= args.name_min]
            return max(names, key=lambda t: name_count[t]) if names else None

        label = [top_name(c) for c in crops]
        keep_names = {nm for nm, k in Counter(l for l in label if l).items() if k >= args.name_min}
        label = [l if l in keep_names else None for l in label]
        source = ["name" if l else None for l in label]
        for m in by_track.values():
            cnt = Counter(label[i] for i in m if label[i])
            if cnt:
                top = cnt.most_common(1)[0][0]
                for i in m:
                    if label[i] is None:
                        label[i], source[i] = top, "track"
        for nm in keep_names:
            named_groups[nm] = [i for i, l in enumerate(label) if l == nm]
        report_calibration["recognised_names"] = {nm: len(g) for nm, g in named_groups.items()}

    # ---- unsupervised identity: pairwise score = z(hair/eye/feature tags) + z(CLIP embedding of the focus-only crop)
    rest = [i for i in range(n_c) if label[i] is None]
    idf = [ident_features(crops[i]["tags"]) for i in range(n_c)]
    T = np.zeros((n_c, n_c))
    for a in range(n_c):
        for b in range(a, n_c):
            T[a, b] = T[b, a] = wjaccard(idf[a], idf[b])
    C = E @ E.T
    sc = np.array([c["scene"] for c in crops])
    cross = sc[:, None] != sc[None, :]
    off = ~np.eye(n_c, dtype=bool)

    def zscore(Mx_):
        v = Mx_[cross & off]
        return (Mx_ - v.mean()) / (v.std() + 1e-9)

    SCORE = zscore(T) + zscore(C)
    conflict_crop = np.zeros((n_c, n_c), bool)
    for m in by_frame.values():
        for a in m:
            for b in m:
                if crops[a]["track"] != crops[b]["track"]:
                    conflict_crop[a, b] = True
    neg_scores = SCORE[np.triu(conflict_crop, 1)]
    thr_link = float(np.quantile(neg_scores, args.merge_quantile)) if len(neg_scores) >= 20 else 4.5
    tids = [t for t in by_track if any(label[i] is None for i in by_track[t])]
    members_t = [[i for i in by_track[t] if label[i] is None] for t in tids]
    k_t = len(tids)
    TS = np.full((k_t, k_t), -1e9)
    for a in range(k_t):
        for b in range(a + 1, k_t):
            TS[a, b] = TS[b, a] = SCORE[np.ix_(members_t[a], members_t[b])].max()
    tconf = np.zeros((k_t, k_t), bool)
    for a in range(k_t):
        for b in range(a + 1, k_t):
            tconf[a, b] = tconf[b, a] = bool(conflict_crop[np.ix_(members_t[a], members_t[b])].any())

    def single_link(sim, members, conflict, thr):
        k = len(members)
        sizes = np.array([len(m) for m in members], float)
        Cm, conf, alive = sim.copy(), conflict.copy(), np.ones(k, bool)
        np.fill_diagonal(Cm, -1e9)
        while k > 1:
            m = np.where(alive[:, None] & alive[None, :] & ~conf, Cm, -1e9)
            np.fill_diagonal(m, -1e9)
            a, b = divmod(int(m.argmax()), k)
            if m[a, b] < thr:
                break
            row = np.maximum(Cm[a], Cm[b])
            Cm[a, :] = row
            Cm[:, a] = row
            Cm[a, a] = -1e9
            conf[a, :] |= conf[b, :]
            conf[:, a] |= conf[:, b]
            members[a] = members[a] + members[b]
            sizes[a] += sizes[b]
            alive[b] = False
        return [m for m, ok in zip(members, alive) if ok]

    merged = single_link(TS, [list(m) for m in members_t], tconf, thr_link) if k_t else []
    clusters = [list(dict.fromkeys(g)) for g in merged]
    frag_before = len(clusters)
    thr_group = float(np.quantile(neg_scores, args.group_merge_quantile)) if len(neg_scores) >= 20 else thr_link * 0.6
    thr_rescue = min(thr_link, float(np.quantile(neg_scores, args.rescue_quantile))) if len(neg_scores) >= 20 else thr_link * 0.6
    if args.group_merge_quantile > 0 and len(clusters) > 1:
        kc = len(clusters)
        gsim = np.full((kc, kc), -1.0)
        gconf = np.zeros((kc, kc), bool)
        for a in range(kc):
            for b in range(a + 1, kc):
                ix = np.ix_(clusters[a], clusters[b])
                gsim[a, b] = gsim[b, a] = float(SCORE[ix].mean())
                gconf[a, b] = gconf[b, a] = bool(conflict_crop[ix].any())
        clusters = [list(dict.fromkeys(g)) for g in constrained_upgma(gsim, [list(g) for g in clusters], gconf, thr_group)]
    big = [g for g in clusters if len(g) >= args.min_char_crops]
    small = [g for g in clusters if len(g) < args.min_char_crops]
    # ---- cascade: grow the confident cores step by step instead of deciding everything with one threshold.
    # Stage k lowers the bar (a quantile of known-different-people scores); a leftover unit (a track / fragment) joins the core it resembles
    # most only when (1) the mean of its best matches to that core clears the bar, (2) it clearly beats the runner-up core (margin), and
    # (3) it never shares a frame with a different person of that core. Cores grow after every join, so later stages see more of each character.
    attached = 0
    cascade_log = []
    pending = [list(g) for g in small]
    if big and pending and len(neg_scores) >= 20:
        stage_thr = [float(np.quantile(neg_scores, q)) for q in (0.95, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40)]
        stage_thr = [t for t in stage_thr if t >= thr_rescue] or [thr_rescue]
        if stage_thr[-1] > thr_rescue:
            stage_thr.append(thr_rescue)
    else:
        stage_thr = [thr_rescue]

    def unit_score(unit, core):
        sub = SCORE[np.ix_(unit, core)]
        top = np.sort(sub, axis=1)[:, -min(3, sub.shape[1]):]          # each crop: mean of its 3 best matches inside the core
        return float(top.mean(axis=1).mean())

    for thr_k in stage_thr:
        joined_stage = 0
        for _ in range(3):
            moved = False
            order = []
            for ui, unit in enumerate(pending):
                scored = []
                for gi, core in enumerate(big):
                    if conflict_crop[np.ix_(unit, core)].any():
                        continue
                    scored.append((unit_score(unit, core), gi))
                if not scored:
                    continue
                scored.sort(reverse=True)
                best, gi = scored[0]
                second = scored[1][0] if len(scored) > 1 else -1e9
                if best >= thr_k and best - second >= args.cascade_margin:
                    order.append((best, ui, gi))
            for best, ui, gi in sorted(order, reverse=True):
                unit = pending[ui]
                if unit is None or conflict_crop[np.ix_(unit, big[gi])].any():
                    continue
                big[gi].extend(unit)
                attached += len(unit)
                joined_stage += len(unit)
                pending[ui] = None
                moved = True
            pending = [u for u in pending if u is not None]
            if not moved:
                break
        cascade_log.append({"threshold": round(thr_k, 2), "crops_joined": joined_stage})
    unassigned = [i for unit in pending for i in unit]
    groups = sorted(list(named_groups.values()) + big, key=len, reverse=True)
    group_names = {id(g): nm for nm, g in named_groups.items()}
    inherited = sum(1 for g in groups if len(g) >= args.min_char_crops for i in g if not any(HAIR_COLOR.match(t) for t in crops[i]["tags"]))
    report_calibration.update({"link_threshold": round(thr_link, 3), "merge_quantile_of_same_frame_different_person_pairs": args.merge_quantile,
                               "same_frame_pairs_used": int(len(neg_scores)), "fragments_before_group_merge": frag_before, "groups_after_group_merge": len(clusters), "group_merge_threshold": round(thr_group, 3), "rescue_threshold": round(thr_rescue, 3), "small_leftovers_attached": attached, "cascade_stages": cascade_log, "crops_left_unassigned": len(unassigned)})

# ---------------------------------------------------------------- write folders
def top_tags(idxs, regex, n=1):
    cnt = Counter(t for i in idxs for t in crops[i]["tags"] if regex.search(t))
    return [t for t, _ in cnt.most_common(n)]


report = {"video": str(video), "params": vars(args), "duration_s": round(duration, 1), "scenes": len(scenes), "frames_kept": len(frames), "damaged_candidates_tried": bad_total,
          "person_crops": len(crops), "funnel": funnel, "tracks": track_counter, "crops_with_other_characters": sum(c["others"] > 0 for c in crops), "crops_in_characters_without_visible_hair_tag": inherited, "identity_calibration": report_calibration,
          "partial_body_at_frame_edge": sum(c["partial"] for c in crops), "characters": [],
          "region": {"start_s": round(region_start, 2), "end_s": round(region_start + duration, 2)},
          "health": {"minutes": round(duration / 60, 2), "crops_per_minute": round(len(crops) / max(duration / 60, 1e-9), 1),
                     "early_duplicate_share": round(funnel["skipped_near_duplicate_early"] / max(1, len(crops) + funnel["skipped_near_duplicate_early"]), 3),
                     "damaged_candidates_per_scene": round(bad_total / max(1, len(scenes)), 2), "frames_per_scene": round(len(frames) / max(1, len(scenes)), 2)}}
for old in list(out.glob("char_*")) + [p for p in out.glob("_*") if p.name not in ("_work", rej_dir.name)]:
    shutil.rmtree(old, ignore_errors=True)
written = 0
progress("書き出し", 90, "画像とTXTを書き出しています", force=True)
with timed("8 書き出し（重複除去・フォルダ・TXT）", max(1, len(crops))):
    char_no = 0
    for group_no, g in enumerate(groups):
        progress("書き出し", 90 + 8 * group_no / max(1, len(groups)), f"{group_no}/{len(groups)} グループ")
        if len(g) < args.min_char_crops:
            unassigned += g
            continue
        char_no += 1
        gname = group_names.get(id(g))
        hair = next(iter(top_tags(g, HAIR_COLOR)), "unknown hair")
        eye = next(iter(top_tags(g, EYE_COLOR)), "")
        label = f"char_{char_no:02d}_" + re.sub(r"[^\w]+", "-", (gname or f"{hair} {eye}").strip()).strip("-")
        trig = f"ch{char_no:02d}"
        ofeat = {i: outfit_features(crops[i]["tags"]) for i in g}
        unknown = [i for i in g if not ofeat[i]]
        known = [i for i in g if ofeat[i]]
        raw = cluster([ofeat[i] for i in known], 0.4, overlap_coeff)
        groups_known = [[known[k] for k in og] for og in raw]
        main = [og for og in groups_known if len(og) >= 3]
        small = [og for og in groups_known if len(og) < 3]
        if not main and small:
            small.sort(key=len, reverse=True)
            main, small = [small[0]], small[1:]
        other_bucket = []
        for og in small:
            def best_match(big, og=og):
                return max(overlap_coeff(ofeat[i], ofeat[j]) for i in og for j in big)
            target = max(main, key=best_match) if main else None
            if target is not None and best_match(target) >= 0.5:
                target += og
            else:
                other_bucket += og
        slots = [(f"outfit_{n:02d}", og) for n, og in enumerate(sorted(main, key=len, reverse=True), start=1)]
        if unknown:
            slots.append(("outfit_00", unknown))
        if other_bucket:
            slots.append(("outfit_99", other_bucket))
        for i in g:
            crops[i]["char"] = char_no
        for slot_name, slot_members in slots:
            for i in slot_members:
                crops[i]["outfit"] = slot_name
        char_report = {"folder": label, "trigger": trig, "crops": len(g), "views": dict(Counter(crops[i]["view"] for i in g)), "outfits": []}
        for slot, og in slots:
            cloth = [t for t, _ in Counter(t for i in og for t in outfit_features(crops[i]["tags"])).most_common(2)]
            if slot == "outfit_00":
                olabel = "outfit_00_不明（衣装が写らない）"
            elif slot == "outfit_99":
                olabel = "outfit_99_その他（少数）"
            else:
                olabel = f"{slot}_" + (re.sub(r"[^\w]+", "-", " ".join(cloth)).strip("-") or "unknown")
            otrig = f"{trig}_o{slot[-2:]}"
            folder = out / label / olabel
            folder.mkdir(parents=True, exist_ok=True)
            kept, hashes = [], []
            for i in sorted(og, key=lambda k: -crops[k]["sharp"]):
                # near-duplicates are only dropped inside the SAME view, so back / cut-off / feet shots survive
                if any(v == crops[i]["view"] and bin(crops[i]["hash"] ^ h).count("1") <= 4 for v, h in hashes):
                    continue
                hashes.append((crops[i]["view"], crops[i]["hash"]))
                kept.append(i)
            for i in sorted(kept, key=lambda k: (crops[k]["scene"], crops[k]["t"])):
                c = crops[i]
                stem = f"{video.stem}_s{c['scene']:04d}_{int(c['t'] // 60):02d}m{int(c['t'] % 60):02d}s_{c['view']}_p{c['file'].split('_p')[-1][:-4]}"
                shutil.copy2(crop_dir / c["file"], folder / f"{stem}.png")
                (folder / f"{stem}.txt").write_text(", ".join([trig, otrig] + [t for t in c["tags"] if t != "solo"]), encoding="utf-8")
                written += 1
            char_report["outfits"].append({"folder": olabel, "trigger": otrig, "images": len(kept), "dropped_duplicates": len(og) - len(kept), "top_clothing": cloth})
        report["characters"].append(char_report)
    rest = out / "_仕分け不能（髪色が判定できない・少数キャラ）"
    rest.mkdir(exist_ok=True)
    for i in unassigned:
        c = crops[i]
        stem = f"{video.stem}_s{c['scene']:04d}_{int(c['t'] // 60):02d}m{int(c['t'] % 60):02d}s_{c['view']}_p{c['file'].split('_p')[-1][:-4]}"
        shutil.copy2(crop_dir / c["file"], rest / f"{stem}.png")
        (rest / f"{stem}.txt").write_text(", ".join(t for t in c["tags"] if t != "solo"), encoding="utf-8")  # so a crop moved to a character keeps its tags
    report["unassigned_images"] = len(unassigned)


def sheet(folder: Path, dest: Path, limit=64):
    files = sorted(folder.rglob("*.png"))[:limit]
    if not files:
        return
    tiles = []
    for p in files:
        im = Image.open(p).convert("RGB")
        im.thumbnail((200, 200))
        t = Image.new("RGB", (200, 214), (24, 24, 28))
        t.paste(im, ((200 - im.width) // 2, 0))
        view = p.stem.split("_")[-2] if "_p" in p.stem else ""
        ImageDraw.Draw(t).text((2, 200), f"{p.parent.name[:12]} {view}"[:34], fill=(200, 200, 200))
        tiles.append(t)
    cols = 8
    canvas = Image.new("RGB", (200 * cols, 214 * ((len(tiles) + cols - 1) // cols)), (24, 24, 28))
    for k, t in enumerate(tiles):
        canvas.paste(t, ((k % cols) * 200, (k // cols) * 214))
    canvas.save(dest, quality=82)


progress("書き出し", 98, "確認用シートを作成しています", force=True)
(out / "_contact_sheets").mkdir(exist_ok=True)
for d in sorted(out.glob("char_*")):
    sheet(d, out / "_contact_sheets" / f"{d.name}.jpg")
(out / "_画質判定ログ.json").write_text(json.dumps(picks, ensure_ascii=False, indent=1), encoding="utf-8")
(work / "crops.json").write_text(json.dumps([{k: v for k, v in c.items() if k not in ("hash", "emb")} for c in crops], ensure_ascii=False), encoding="utf-8")

wall = time.perf_counter() - wall0
n_out = max(1, written + report["unassigned_images"])
report["images_written"] = written
report["wall_clock_s"] = round(wall, 1)
report["per_output_image_s"] = round(wall / n_out, 2)
report["per_output_image_excl_model_load_s"] = round((wall - timers["0 モデル読込（初回のみ）"][0]) / n_out, 2)
report["view_distribution_all_crops"] = dict(Counter(c["view"] for c in crops))
report["timing"] = {k: {"total_s": round(v[0], 2), "count": v[1], "per_unit_s": round(v[0] / max(1, v[1]), 3)} for k, v in sorted(timers.items())}
report["gpu"] = "RTX 3090 Ti (YOLO seg / BiRefNet / upscaler / WD14 ONNX-CUDA)"
(out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k: report[k] for k in ("scenes", "frames_kept", "person_crops", "tracks", "crops_with_other_characters", "crops_in_characters_without_visible_hair_tag", "partial_body_at_frame_edge",
                                          "images_written", "unassigned_images", "wall_clock_s", "per_output_image_s", "per_output_image_excl_model_load_s")}, ensure_ascii=False), flush=True)
for k, v in report["timing"].items():
    print(f"  {k}: total {v['total_s']}s / {v['count']} = {v['per_unit_s']}s each", flush=True)
for ch in report["characters"]:
    print(f"  {ch['folder']} ({ch['trigger']}): {ch['crops']} crops; views {ch['views']}; outfits:", [(o['folder'], o['images']) for o in ch["outfits"]], flush=True)
progress("完了", 100, f"{written} 枚を書き出しました", force=True)
# Skip interpreter teardown: ComfyUI's ModelPatcher.__del__ prints a harmless 'ON_DETACH' traceback at exit.
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
