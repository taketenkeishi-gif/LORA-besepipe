from __future__ import annotations

import base64
import re
import shutil
import subprocess
from pathlib import Path
from urllib.parse import quote, urlparse

from fastapi import APIRouter, HTTPException
from PIL import Image, UnidentifiedImageError

from ..db import get_conn
from ..schemas import CollectorImportIn, CollectorScanIn, RepeatFolderIn

router = APIRouter(prefix="/collector", tags=["collector"])
SCAN_CACHE: dict[int, list[dict]] = {}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _slugify(text: str) -> str:
    v = re.sub(r"[^\w\-]+", "_", text.strip(), flags=re.UNICODE)
    return v.strip("_") or "image"


def _settings_value(key: str) -> str:
    conn = get_conn()
    row = conn.execute("SELECT value FROM app_settings WHERE key = ?", (key,)).fetchone()
    conn.close()
    return str(row["value"]) if row and row["value"] else ""


def _wd14_script_path() -> str:
    return _settings_value("wd14_script").strip()


def _ensure_project(project_id: int) -> dict:
    conn = get_conn()
    row = conn.execute(
        "SELECT id, dataset_dir, project_type, library_dir FROM projects WHERE id = ?",
        (project_id,),
    ).fetchone()
    conn.close()
    if row is None:
        raise HTTPException(status_code=404, detail=f"project not found: {project_id}")
    return dict(row)


def _thumbnail_data_uri(title: str, i: int) -> str:
    hue = (i * 31) % 360
    svg = f"""
<svg xmlns='http://www.w3.org/2000/svg' width='256' height='256'>
  <defs>
    <linearGradient id='g' x1='0' y1='0' x2='1' y2='1'>
      <stop offset='0%' stop-color='hsl({hue},70%,62%)'/>
      <stop offset='100%' stop-color='hsl({(hue + 40) % 360},70%,42%)'/>
    </linearGradient>
  </defs>
  <rect width='256' height='256' fill='url(#g)'/>
  <rect x='12' y='12' width='232' height='232' rx='16' fill='rgba(255,255,255,0.18)'/>
  <text x='20' y='214' fill='white' font-size='18' font-family='Segoe UI, sans-serif'>{title}</text>
</svg>
"""
    return f"data:image/svg+xml;utf8,{quote(svg)}"


def _thumb_cache_dir() -> Path:
    temp_dir = _settings_value("temp_dir").strip()
    if not temp_dir:
        temp_dir = str(Path(__file__).resolve().parents[3] / ".runtime" / "tmp")
    d = Path(temp_dir) / "thumb_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _make_cached_thumb(src: Path, title: str, idx: int) -> str:
    try:
        cache_name = f"{src.stem}_{src.stat().st_mtime_ns}_{idx}.jpg"
        cache_name = _slugify(cache_name)
        out = _thumb_cache_dir() / cache_name
        if not out.exists():
            with Image.open(src) as im:
                im = im.convert("RGB")
                im.thumbnail((320, 320))
                im.save(out, format="JPEG", quality=82)
        # file:// はElectron/ブラウザで制約があるので data URIへ再エンコード
        raw = out.read_bytes()
        return "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")
    except (OSError, UnidentifiedImageError):
        return _thumbnail_data_uri(title, idx)


def _aspect(w: int, h: int) -> str:
    if w > h:
        return "landscape"
    if h > w:
        return "portrait"
    return "square"


def _scan_from_dataset_base(project: dict, keyword: str, limit: int) -> list[dict]:
    base_dir_raw = _settings_value("dataset_base_dir").strip()
    base_dir = Path(base_dir_raw) if base_dir_raw else Path(r"C:\ポートフォリオ\SDXL\LoRA_Traning\dataset")
    scope = base_dir / str(project["project_type"])
    if not scope.exists():
        return []

    kws = [k.lower() for k in keyword.split() if k.strip()]
    files: list[Path] = [p for p in scope.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    scored: list[tuple[int, Path]] = []
    for p in files:
        score = 0
        name = p.stem.lower()
        if kws and any(k in name for k in kws):
            score += 50
        if project["project_type"] == "character":
            if any(k in name for k in ("face", "bust", "portrait", "closeup", "character")):
                score += 20
        else:
            if any(k in name for k in ("style", "texture", "bg", "background", "scene")):
                score += 20
        score += int(p.stat().st_size / 1024 / 50)  # 大きめ画像を優先
        scored.append((score, p))
    scored.sort(key=lambda x: x[0], reverse=True)

    picked = [p for _, p in scored[:limit]]
    out: list[dict] = []
    for i, p in enumerate(picked, start=1):
        try:
            with Image.open(p) as im:
                w, h = im.size
        except (OSError, UnidentifiedImageError):
            w, h = (0, 0)
        title = p.stem
        out.append(
            {
                "id": i,
                "title": title,
                "width": w,
                "height": h,
                "aspect": _aspect(max(w, 1), max(h, 1)),
                "source_url": str(p),
                "source_file": str(p),
                "thumbnail_url": _make_cached_thumb(p, title, i),
                "tags": _pretag_file(p, project["project_type"]),
            }
        )
    return out


def _pretag_file(path: Path, project_type: str) -> list[str]:
    # 1) 既存のWD14タグ(txt)があれば最優先で利用
    caption = path.with_suffix(".txt")
    if caption.exists():
        try:
            raw = caption.read_text(encoding="utf-8", errors="ignore")
            tags = [t.strip().lower() for t in raw.split(",") if t.strip()]
            if tags:
                return sorted(set(tags))
        except OSError:
            pass

    # 2) 実運用ではWD14実行結果を優先。失敗時は軽量推定タグでフォールバック。
    wd14 = _wd14_script_path()
    if wd14 and Path(wd14).exists():
        # ここでは高コスト実行を避けるため将来拡張ポイントとして保持
        pass
    tags: list[str] = []
    name = path.stem.lower()
    if project_type == "character":
        tags += ["character", "solo", "portrait"]
    else:
        tags += ["style", "background", "composition"]
    for k in ("face", "bust", "full", "bg", "closeup", "texture", "lineart"):
        if k in name:
            tags.append(k)
    return sorted(set(tags))


def _filter_items(items: list[dict], keyword: str, project_type: str) -> list[dict]:
    # keyword grammar: "tag:face aspect:square minw:768 minh:768 foo"
    tokens = [t.strip() for t in keyword.split() if t.strip()]
    minw = 0
    minh = 0
    aspect: str | None = None
    tag_terms: list[str] = []
    text_terms: list[str] = []
    for t in tokens:
        low = t.lower()
        if low.startswith("minw:"):
            try:
                minw = int(low.split(":", 1)[1])
            except ValueError:
                pass
            continue
        if low.startswith("minh:"):
            try:
                minh = int(low.split(":", 1)[1])
            except ValueError:
                pass
            continue
        if low.startswith("aspect:"):
            aspect = low.split(":", 1)[1]
            continue
        if low.startswith("tag:"):
            tag_terms.append(low.split(":", 1)[1])
            continue
        text_terms.append(low)

    out: list[dict] = []
    for item in items:
        w = int(item.get("width", 0))
        h = int(item.get("height", 0))
        asp = str(item.get("aspect", ""))
        tags = [str(x).lower() for x in item.get("tags", [])]
        title = str(item.get("title", "")).lower()

        if w < minw or h < minh:
            continue
        if aspect and asp != aspect:
            continue
        if tag_terms and not all(t in tags for t in tag_terms):
            continue
        if text_terms and not any(t in title or t in " ".join(tags) for t in text_terms):
            continue

        # タイプ最適化: characterは人物寄り、styleは背景/質感寄りを優先
        boost = 0
        if project_type == "character":
            if "face" in tags or "portrait" in tags:
                boost = 1
        else:
            if "background" in tags or "texture" in tags:
                boost = 1
        item["_boost"] = boost
        out.append(item)

    out.sort(key=lambda x: int(x.get("_boost", 0)), reverse=True)
    for i in out:
        i.pop("_boost", None)
    return out


@router.post("/scan")
def scan(payload: CollectorScanIn) -> dict:
    project = _ensure_project(payload.project_id)
    keyword = payload.keyword.strip()

    # 1) dataset_base を優先スキャン
    dataset_items = _scan_from_dataset_base(project, keyword=keyword, limit=payload.limit * 3)
    if dataset_items:
        dataset_items = _filter_items(dataset_items, keyword=keyword, project_type=str(project["project_type"]))
        dataset_items = dataset_items[: payload.limit]
    if dataset_items:
        SCAN_CACHE[payload.project_id] = dataset_items
        return {
            "project_id": payload.project_id,
            "url": payload.url,
            "mode": "dataset_base",
            "detected": len(dataset_items),
            "items": dataset_items,
            "message": f"dataset_base({project['project_type']}) から候補を取得",
        }

    # 2) fallback: URL由来のモック候補
    parsed = urlparse(payload.url)
    seed = _slugify(Path(parsed.path).stem or parsed.netloc or "image")
    items = []
    for i in range(1, min(payload.limit, 24) + 1):
        title = f"{seed}_{i:02d}"
        items.append(
            {
                "id": i,
                "title": title,
                "width": 1024,
                "height": 1024,
                "aspect": "square",
                "source_url": payload.url,
                "thumbnail_url": _thumbnail_data_uri(title, i),
                "tags": (["character", "portrait"] if project["project_type"] == "character" else ["style", "background"]),
            }
        )
    items = _filter_items(items, keyword=keyword, project_type=str(project["project_type"]))
    SCAN_CACHE[payload.project_id] = items
    return {
        "project_id": payload.project_id,
        "url": payload.url,
        "mode": "mock_url",
        "detected": len(items),
        "items": items,
        "message": "fallback mock scan completed",
    }


@router.post("/import")
def import_selected(payload: CollectorImportIn) -> dict:
    project = _ensure_project(payload.project_id)
    candidates = SCAN_CACHE.get(payload.project_id, [])
    by_id = {item["id"]: item for item in candidates}
    if not candidates:
        raise HTTPException(status_code=400, detail="scan first: candidates not found")

    dataset_dir = Path(project["dataset_dir"])
    dataset_dir.mkdir(parents=True, exist_ok=True)
    library_dir = Path(project["library_dir"])
    library_dir.mkdir(parents=True, exist_ok=True)

    conn = get_conn()
    cur = conn.cursor()
    imported = []
    for order, item_id in enumerate(payload.selected_ids, start=1):
        item = by_id.get(item_id)
        if item is None:
            continue
        filename = (
            payload.naming_template.replace("{title}", _slugify(item["title"])).replace("{index}", f"{order:04d}")
        )
        out_path = dataset_dir / f"{filename}.png"

        src_file = item.get("source_file")
        if src_file and Path(src_file).exists():
            shutil.copy2(src_file, out_path)
            # 補完データセット側にも保管（同名衝突回避）
            lib_path = library_dir / out_path.name
            if not lib_path.exists():
                shutil.copy2(src_file, lib_path)
        else:
            # mock fallback
            out_path.write_bytes(base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7ZbHkAAAAASUVORK5CYII="))

        cur.execute(
            """
            INSERT INTO dataset_items(project_id, file_path, width, height, aspect, selected)
            VALUES (?, ?, ?, ?, ?, 1)
            """,
            (
                payload.project_id,
                str(out_path),
                item.get("width", 0),
                item.get("height", 0),
                item.get("aspect", "unknown"),
            ),
        )
        imported.append(str(out_path))

    conn.commit()
    conn.close()
    return {
        "project_id": payload.project_id,
        "imported_count": len(imported),
        "files": imported,
        "naming_template": payload.naming_template,
        "message": "dataset import completed",
    }


@router.post("/prepare-repeat-folder")
def prepare_repeat_folder(payload: RepeatFolderIn) -> dict:
    project = _ensure_project(payload.project_id)
    dataset_dir = Path(project["dataset_dir"])
    title = _slugify(payload.folder_title)
    folder = dataset_dir / f"{payload.repeats}_{title}"
    folder.mkdir(parents=True, exist_ok=True)

    copied = 0
    for src in dataset_dir.glob("*"):
        if src.is_file() and src.suffix.lower() in IMAGE_EXTS:
            dst = folder / src.name
            shutil.copy2(src, dst)
            copied += 1
    return {
        "project_id": payload.project_id,
        "folder": str(folder),
        "copied_images": copied,
        "message": "repeat folder prepared",
    }
