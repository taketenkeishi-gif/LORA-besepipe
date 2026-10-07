"""Deterministic turntable enhancement preparation and evidence rendering.

This module performs no aesthetic or identity approval.  It only prepares
equally-spaced circular temporal windows, extracts the center images returned
by enhancement engines, and renders a comparison sheet for the user's review.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps, PngImagePlugin


def _ffmpeg() -> str:
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(command: list[str], *, timeout: float = 600.0) -> None:
    completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    if completed.returncode:
        raise RuntimeError((completed.stderr or completed.stdout or "ffmpeg failed")[-4000:])


def prepare_circular_yaw_windows(
    source_video: Path,
    output_dir: Path,
    *,
    total_frames: int = 124,
    slot_count: int = 8,
    fps: int = 24,
    radius: int = 4,
) -> list[dict]:
    """Create 8n+1 lossless windows around evenly-spaced yaw targets."""
    if not source_video.is_file():
        raise FileNotFoundError(source_video)
    if total_frames < 2 * radius + 1 or slot_count < 1:
        raise ValueError("invalid circular window dimensions")
    output_dir.mkdir(parents=True, exist_ok=True)
    native_dir = output_dir / "native_frames"
    if native_dir.exists():
        shutil.rmtree(native_dir)
    native_dir.mkdir(parents=True)
    _run([
        _ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(source_video),
        "-frames:v", str(total_frames), "-vsync", "0", str(native_dir / "frame_%04d.png"),
    ])
    frames = sorted(native_dir.glob("frame_*.png"))
    if len(frames) != total_frames:
        raise RuntimeError(f"expected {total_frames} native frames, extracted {len(frames)}")

    windows: list[dict] = []
    for slot in range(slot_count):
        center_index = round(slot * total_frames / slot_count) % total_frames
        slot_dir = output_dir / f"yaw_{slot:02d}"
        window_dir = slot_dir / "window_frames"
        window_dir.mkdir(parents=True, exist_ok=True)
        for stale in window_dir.glob("*.png"):
            stale.unlink()
        source_indices = [(center_index + offset) % total_frames for offset in range(-radius, radius + 1)]
        for output_index, source_index in enumerate(source_indices, start=1):
            shutil.copy2(frames[source_index], window_dir / f"frame_{output_index:04d}.png")
        original_path = slot_dir / "original.png"
        shutil.copy2(frames[center_index], original_path)
        window_path = slot_dir / "window_9f.mkv"
        _run([
            _ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-framerate", str(fps),
            "-i", str(window_dir / "frame_%04d.png"), "-c:v", "ffv1", "-pix_fmt", "rgb24", str(window_path),
        ])
        windows.append({
            "yaw_slot": slot,
            "yaw_target_degrees": slot * 360.0 / slot_count,
            "center_frame_index": center_index,
            "source_frame_indices": source_indices,
            "window_path": str(window_path),
            "window_sha256": _sha256(window_path),
            "original_path": str(original_path),
            "original_sha256": _sha256(original_path),
            "aesthetic_status": "AESTHETIC_UNREVIEWED",
        })
    return windows


def extract_video_center_frame(video_path: Path, output_path: Path, *, center_index: int = 4) -> dict:
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _run([
        _ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(video_path),
        "-vf", f"select=eq(n\\,{center_index})", "-frames:v", "1", str(output_path),
    ])
    with Image.open(output_path) as image:
        width, height = image.size
    return {"file_path": str(output_path), "content_sha256": _sha256(output_path), "width": width, "height": height}


def derive_2x_from_4x(native_4x: Path, output_path: Path, *, source_size: tuple[int, int]) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(native_4x) as source:
        image = source.convert("RGB")
        expected = (source_size[0] * 4, source_size[1] * 4)
        if image.size != expected:
            raise RuntimeError(f"FlashVSR 4x dimension mismatch: expected={expected}, actual={image.size}")
        image = image.resize((source_size[0] * 2, source_size[1] * 2), Image.Resampling.LANCZOS)
        image.save(output_path)
    return {"file_path": str(output_path), "content_sha256": _sha256(output_path), "width": image.width, "height": image.height}


def create_enhancement_comparison_sheet(rows: list[dict], output_path: Path, *, run_id: int) -> dict:
    """Render Original/Flash9/Flash11/RealESRGAN columns without choosing a winner."""
    variants = [
        ("original", "Original"),
        ("flashvsr_lr9_2x", "FlashVSR lr=9 (4x→2x)"),
        ("flashvsr_lr11_2x", "FlashVSR lr=11 (4x→2x)"),
        ("realesrgan_2x", "RealESRGAN x2"),
    ]
    if not rows:
        raise ValueError("comparison rows are empty")
    thumb = (256, 384)
    cell = (280, 438)
    header = 88
    sheet = Image.new("RGB", (cell[0] * len(variants), header + cell[1] * len(rows) + 48), "#0d1118")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((16, 14), f"Run #{run_id} Turntable Enhancement Comparison", fill="white", font=font)
    draw.text((16, 38), "Technical output only — choose nothing automatically", fill="#f4c95d", font=font)
    draw.text((16, 60), "AESTHETIC_UNREVIEWED", fill="#f4c95d", font=font)
    for column, (_, label) in enumerate(variants):
        draw.text((column * cell[0] + 12, header - 18), label, fill="#b9c7da", font=font)
    manifest_rows = []
    for row_index, row in enumerate(sorted(rows, key=lambda item: int(item["yaw_slot"]))):
        manifest_row = {"yaw_slot": int(row["yaw_slot"]), "yaw_target_degrees": float(row["yaw_target_degrees"]), "variants": {}}
        for column, (key, _label) in enumerate(variants):
            path = Path(str(row[key]))
            if not path.is_file():
                raise FileNotFoundError(path)
            with Image.open(path) as source:
                image = ImageOps.contain(source.convert("RGB"), thumb, Image.Resampling.LANCZOS)
            x = column * cell[0] + (cell[0] - image.width) // 2
            y = header + row_index * cell[1] + 12
            sheet.paste(image, (x, y))
            draw.rectangle((column * cell[0] + 8, y - 4, (column + 1) * cell[0] - 8, y + thumb[1] + 4), outline="#364154")
            draw.text((column * cell[0] + 12, header + row_index * cell[1] + 404), f"yaw {row['yaw_target_degrees']:.0f}°", fill="white", font=font)
            manifest_row["variants"][key] = {"file_path": str(path), "content_sha256": _sha256(path)}
        manifest_rows.append(manifest_row)
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("run_id", str(run_id))
    metadata.add_text("aesthetic_status", "AESTHETIC_UNREVIEWED")
    metadata.add_text("comparison_manifest", json.dumps(manifest_rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, pnginfo=metadata)
    return {
        "file_path": str(output_path), "content_sha256": _sha256(output_path),
        "width": sheet.width, "height": sheet.height, "rows": manifest_rows,
        "aesthetic_status": "AESTHETIC_UNREVIEWED",
    }
