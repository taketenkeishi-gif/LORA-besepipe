"""CPU-side H3 video admission pipeline.

H3 is a variation generator, not a training-data source.  This module extracts
candidate frames, records measurable blur/duplicate decisions, and leaves every
frame as a pending Asset for explicit identity review.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Mapping, Sequence

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps, ImageStat, PngImagePlugin


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ffmpeg() -> str:
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Frame抽出に必要なffmpegが見つかりません") from exc


def extract_frames(video_path: Path, output_dir: Path, *, fps: float = 2.0, max_frames: int = 24) -> list[Path]:
    """Decode a bounded number of PNG frames using the CPU ffmpeg path."""
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    if fps <= 0 or max_frames < 1:
        raise ValueError("fpsは正数、max_framesは1以上で指定してください")
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = output_dir / "frame_%04d.png"
    command = [_ffmpeg(), "-hide_banner", "-loglevel", "error", "-i", str(video_path), "-vf", f"fps={fps}", "-frames:v", str(max_frames), "-y", str(pattern)]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=180)
    if completed.returncode != 0:
        raise RuntimeError(f"Frame抽出失敗: {completed.stderr[-1000:]}")
    frames = sorted(output_dir.glob("frame_*.png"))
    if not frames:
        raise RuntimeError("Frame抽出結果が0件です")
    return frames


def inspect_frame(path: Path, *, blur_threshold: float = 18.0) -> dict:
    """Return deterministic CPU measurements for one candidate frame."""
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        gray = rgb.convert("L")
        edge = gray.filter(ImageFilter.FIND_EDGES)
        # Ignore the artificial one-pixel border produced by FIND_EDGES;
        # otherwise a flat frame would look sharp because of its perimeter.
        inner = edge.crop((1, 1, max(2, edge.width - 1), max(2, edge.height - 1)))
        blur_score = float(ImageStat.Stat(inner).var[0])
        try:
            import imagehash
            perceptual_hash = str(imagehash.phash(rgb))
        except Exception:
            perceptual_hash = hashlib.sha256(rgb.tobytes()).hexdigest()[:32]
    return {
        "file_path": str(path),
        "content_sha256": _sha256(path),
        "width": rgb.width,
        "height": rgb.height,
        "blur_score": round(blur_score, 4),
        "blur_status": "rejected" if blur_score < blur_threshold else "pass",
        "perceptual_hash": perceptual_hash,
    }


def classify_frames(frames: list[Path], *, blur_threshold: float = 18.0, duplicate_distance: int = 6) -> list[dict]:
    """Classify frames without silently accepting any candidate."""
    results: list[dict] = []
    accepted_hashes: list[tuple[str, int]] = []
    for ordinal, path in enumerate(frames):
        item = inspect_frame(path, blur_threshold=blur_threshold)
        item["ordinal"] = ordinal
        item["review_status"] = "rejected" if item["blur_status"] == "rejected" else "needs_review"
        item["rejection_reason"] = "blur" if item["blur_status"] == "rejected" else ""
        if item["blur_status"] == "pass":
            try:
                import imagehash
                current = imagehash.hex_to_hash(item["perceptual_hash"])
                duplicate_of = next((previous for previous_hash, previous in accepted_hashes if current - imagehash.hex_to_hash(previous_hash) <= duplicate_distance), None)
            except Exception:
                duplicate_of = next((previous for previous_hash, previous in accepted_hashes if previous_hash == item["perceptual_hash"]), None)
            if duplicate_of is not None:
                item["review_status"] = "rejected"
                item["rejection_reason"] = "duplicate"
            else:
                accepted_hashes.append((item["perceptual_hash"], ordinal))
        results.append(item)
    return results


def _circular_distance_degrees(left: float, right: float) -> float:
    delta = abs(left - right) % 360.0
    return min(delta, 360.0 - delta)


def select_turntable_yaw_slots(
    candidates: list[dict],
    *,
    total_frame_count: int | None = None,
    slot_count: int = 8,
    start_yaw_degrees: float = 0.0,
) -> list[dict]:
    """Select unique technical representatives for a constant-speed 360° turntable.

    Only candidates that passed both blur and duplicate filtering participate.
    The frame ordinal is converted to an estimated yaw over ``[0, 360)`` and a
    minimum-total-angular-distance assignment is solved across equally spaced
    yaw targets.  This function makes no identity or aesthetic decision.
    """
    if slot_count < 1:
        raise ValueError("slot_countは1以上で指定してください")
    if not candidates:
        raise ValueError("候補フレームがありません")

    ordinals: list[int] = []
    for candidate in candidates:
        ordinal = candidate.get("ordinal")
        if not isinstance(ordinal, int) or isinstance(ordinal, bool) or ordinal < 0:
            raise ValueError("全候補に0以上の整数ordinalが必要です")
        ordinals.append(ordinal)
    if len(set(ordinals)) != len(ordinals):
        raise ValueError("候補フレームのordinalは一意である必要があります")

    frame_count = total_frame_count if total_frame_count is not None else max(ordinals) + 1
    if frame_count < 1 or any(ordinal >= frame_count for ordinal in ordinals):
        raise ValueError("total_frame_countは全ordinalより大きい必要があります")

    eligible: list[tuple[int, dict, float]] = []
    for candidate in candidates:
        rejection_reason = str(candidate.get("rejection_reason") or "").lower()
        duplicate_status = str(candidate.get("duplicate_status") or "pass").lower()
        if candidate.get("blur_status") != "pass":
            continue
        if candidate.get("review_status") == "rejected":
            continue
        if rejection_reason in {"blur", "duplicate"} or duplicate_status in {"duplicate", "rejected"}:
            continue
        ordinal = int(candidate["ordinal"])
        yaw = (float(start_yaw_degrees) + (360.0 * ordinal / frame_count)) % 360.0
        eligible.append((ordinal, candidate, yaw))

    eligible.sort(key=lambda item: (item[0], str(item[1].get("file_path", ""))))
    if len(eligible) < slot_count:
        raise ValueError(f"blur/duplicate pass候補が不足しています: required={slot_count}, actual={len(eligible)}")

    targets = [((float(start_yaw_degrees) + 360.0 * slot / slot_count) % 360.0) for slot in range(slot_count)]
    empty_assignment = tuple([-1] * slot_count)
    # mask -> (total angular distance, candidate index per yaw slot)
    states: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, empty_assignment)}
    for candidate_index, (_, _, estimated_yaw) in enumerate(eligible):
        previous_states = list(states.items())
        for mask, (cost, assignment) in previous_states:
            for slot, target_yaw in enumerate(targets):
                bit = 1 << slot
                if mask & bit:
                    continue
                next_assignment = list(assignment)
                next_assignment[slot] = candidate_index
                next_assignment_tuple = tuple(next_assignment)
                next_cost = cost + _circular_distance_degrees(estimated_yaw, target_yaw)
                next_mask = mask | bit
                current = states.get(next_mask)
                candidate_key = (
                    round(next_cost, 12),
                    tuple(eligible[index][0] if index >= 0 else frame_count for index in next_assignment_tuple),
                )
                current_key = None
                if current is not None:
                    current_key = (
                        round(current[0], 12),
                        tuple(eligible[index][0] if index >= 0 else frame_count for index in current[1]),
                    )
                if current_key is None or candidate_key < current_key:
                    states[next_mask] = (next_cost, next_assignment_tuple)

    full_mask = (1 << slot_count) - 1
    assignment = states[full_mask][1]
    selected: list[dict] = []
    for slot, candidate_index in enumerate(assignment):
        ordinal, candidate, estimated_yaw = eligible[candidate_index]
        target_yaw = targets[slot]
        angular_distance = _circular_distance_degrees(estimated_yaw, target_yaw)
        result = dict(candidate)
        result.update(
            {
                "estimated_yaw_degrees": round(estimated_yaw, 6),
                "yaw_slot": slot,
                "yaw_target_degrees": round(target_yaw, 6),
                "selection_reason": (
                    "technical_nearest_yaw_assignment;blur_pass;duplicate_pass;"
                    f"ordinal={ordinal};angular_distance_degrees={angular_distance:.6f}"
                ),
            }
        )
        selected.append(result)
    return selected


def _normalized_input_hashes(input_hashes: str | Sequence[str] | Mapping[str, str]) -> str:
    if isinstance(input_hashes, str):
        payload: str | list[str] | dict[str, str] = input_hashes
    elif isinstance(input_hashes, Mapping):
        payload = {str(key): str(value) for key, value in sorted(input_hashes.items(), key=lambda item: str(item[0]))}
    else:
        payload = [str(value) for value in input_hashes]
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def create_original_contact_sheet(
    selections: list[dict],
    output_path: Path,
    *,
    run_id: int | str,
    input_hashes: str | Sequence[str] | Mapping[str, str],
    columns: int = 4,
    thumbnail_size: tuple[int, int] = (320, 320),
) -> dict:
    """Render a labeled contact sheet from the selected original frame files."""
    if not selections:
        raise ValueError("Contact Sheetに必要な選抜フレームがありません")
    if columns < 1 or thumbnail_size[0] < 1 or thumbnail_size[1] < 1:
        raise ValueError("columnsとthumbnail_sizeは正数で指定してください")

    ordered = sorted(selections, key=lambda item: int(item["yaw_slot"]))
    if len({int(item["yaw_slot"]) for item in ordered}) != len(ordered):
        raise ValueError("yaw_slotは一意である必要があります")
    for item in ordered:
        path = Path(str(item.get("file_path", "")))
        if not path.is_file():
            raise FileNotFoundError(path)

    rows = (len(ordered) + columns - 1) // columns
    cell_width, cell_height = thumbnail_size[0] + 24, thumbnail_size[1] + 64
    header_height, footer_height = 72, 58
    sheet = Image.new("RGB", (columns * cell_width, header_height + rows * cell_height + footer_height), "#11151c")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    normalized_hashes = _normalized_input_hashes(input_hashes)
    draw.text((16, 14), f"Original Turntable Contact Sheet | Run {run_id}", fill="#ffffff", font=font)
    draw.text((16, 38), f"Input SHA-256: {normalized_hashes}", fill="#aeb8c8", font=font)

    labels: list[dict] = []
    for index, item in enumerate(ordered):
        row, column = divmod(index, columns)
        cell_x = column * cell_width
        cell_y = header_height + row * cell_height
        frame_path = Path(str(item["file_path"]))
        with Image.open(frame_path) as source:
            original = source.convert("RGB")
            thumbnail = ImageOps.contain(original, thumbnail_size, Image.Resampling.LANCZOS)
        image_x = cell_x + 12 + (thumbnail_size[0] - thumbnail.width) // 2
        image_y = cell_y + 8 + (thumbnail_size[1] - thumbnail.height) // 2
        draw.rectangle(
            (cell_x + 11, cell_y + 7, cell_x + 12 + thumbnail_size[0], cell_y + 8 + thumbnail_size[1]),
            outline="#3c4656",
            width=1,
        )
        sheet.paste(thumbnail, (image_x, image_y))
        target = float(item.get("yaw_target_degrees", 0.0))
        estimated = float(item["estimated_yaw_degrees"])
        label = f"Slot {int(item['yaw_slot'])} | target {target:.1f} deg | estimated {estimated:.1f} deg"
        draw.text((cell_x + 12, cell_y + thumbnail_size[1] + 18), label, fill="#ffffff", font=font)
        draw.text((cell_x + 12, cell_y + thumbnail_size[1] + 38), frame_path.name, fill="#aeb8c8", font=font)
        labels.append(
            {
                "yaw_slot": int(item["yaw_slot"]),
                "yaw_target_degrees": target,
                "estimated_yaw_degrees": estimated,
                "file_path": str(frame_path),
                "content_sha256": str(item.get("content_sha256") or _sha256(frame_path)),
            }
        )

    draw.text((16, sheet.height - footer_height + 18), "Technical selection only - AESTHETIC_UNREVIEWED", fill="#f4c95d", font=font)
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("contact_sheet_type", "original_turntable")
    metadata.add_text("run_id", str(run_id))
    metadata.add_text("input_sha256", normalized_hashes)
    metadata.add_text("yaw_selections", json.dumps(labels, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    metadata.add_text("aesthetic_status", "AESTHETIC_UNREVIEWED")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="PNG", pnginfo=metadata)
    return {
        "file_path": str(output_path),
        "content_sha256": _sha256(output_path),
        "width": sheet.width,
        "height": sheet.height,
        "run_id": str(run_id),
        "input_sha256": normalized_hashes,
        "yaw_slots": labels,
        "aesthetic_status": "AESTHETIC_UNREVIEWED",
    }
