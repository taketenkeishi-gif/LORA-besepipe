"""画像検索ルーター — DuckDuckGo 経由でキーワード検索"""
from __future__ import annotations

from typing import Any

from ddgs import DDGS
from fastapi import APIRouter, Query

router = APIRouter(prefix="/imgsearch", tags=["imgsearch"])

_SKIP_DOMAINS = {
    "pinterest.com", "instagram.com", "facebook.com",
    "doubleclick.net", "tiktok.com",
}


def _is_blocked(url: str) -> bool:
    return any(d in url for d in _SKIP_DOMAINS)


@router.get("/search")
def image_search(
    q: str = Query(..., min_length=1, max_length=200),
    limit: int = Query(default=40, ge=1, le=60),
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    try:
        with DDGS() as ddgs:
            for item in ddgs.images(q, max_results=limit):
                img_url = item.get("image", "")
                thumb = item.get("thumbnail", img_url)
                src = item.get("url", "")
                if _is_blocked(img_url) or _is_blocked(src):
                    continue
                results.append({
                    "url": img_url,
                    "thumbnail": thumb,
                    "title": item.get("title", ""),
                    "width": item.get("width", 0),
                    "height": item.get("height", 0),
                    "source": src,
                })
    except Exception as exc:  # noqa: BLE001
        return {"query": q, "results": [], "error": str(exc)}

    return {"query": q, "results": results}
