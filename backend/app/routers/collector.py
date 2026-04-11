from __future__ import annotations

import base64
import re
from pathlib import Path
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException

from ..db import get_conn

from ..schemas import CollectorImportIn, CollectorScanIn

router = APIRouter(prefix="/collector", tags=["collector"])
SCAN_CACHE: dict[int, list[dict]] = {}

# 1x1 transparent PNG
PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7ZbHkAAAAASUVORK5CYII="
)


def _slugify(text: str) -> str:
    v = re.sub(r"[^\w\-]+", "_", text.strip(), flags=re.UNICODE)
    return v.strip("_") or "image"


def _ensure_project(project_id: int) -> dict:
    conn = get_conn()
    row = conn.execute(
        "SELECT id, dataset_dir FROM projects WHERE id = ?",
        (project_id,),
    ).fetchone()
    conn.close()
    if row is None:
        raise HTTPException(status_code=404, detail=f"project not found: {project_id}")
    return dict(row)


@router.post("/scan")
def scan(payload: CollectorScanIn) -> dict:
    _ensure_project(payload.project_id)
    parsed = urlparse(payload.url)
    seed = _slugify(Path(parsed.path).stem or parsed.netloc or "image")
    items = []
    for i in range(1, 13):
        items.append(
            {
                "id": i,
                "title": f"{seed}_{i:02d}",
                "width": 1024,
                "height": 1024,
                "aspect": "square",
                "source_url": payload.url,
            }
        )
    SCAN_CACHE[payload.project_id] = items
    return {
        "project_id": payload.project_id,
        "url": payload.url,
        "detected": len(items),
        "items": items,
        "message": "mock scan completed (本番収集ロジックは未実装)",
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
        out_path.write_bytes(PNG_1X1)
        cur.execute(
            """
            INSERT INTO dataset_items(project_id, file_path, width, height, aspect, selected)
            VALUES (?, ?, ?, ?, ?, 1)
            """,
            (
                payload.project_id,
                str(out_path),
                item["width"],
                item["height"],
                item["aspect"],
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
