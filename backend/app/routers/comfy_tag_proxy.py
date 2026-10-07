"""Pass-through to the user's ComfyUI for the tag editor's own routes (tagger status / model download / image analysis)."""
from __future__ import annotations

import httpx
from fastapi import APIRouter, Request, Response

router = APIRouter(tags=["comfy-tag-proxy"])
COMFY = "http://127.0.0.1:8188"
_PREFIXES = ("unified_tag_editor", "sdxl_tag_editor")


@router.api_route("/unified_tag_editor/{path:path}", methods=["GET", "POST"])
async def unified_tag_editor(path: str, request: Request) -> Response:
    return await _forward("unified_tag_editor", path, request)


async def _forward(prefix: str, path: str, request: Request) -> Response:
    url = f"{COMFY}/{prefix}/{path}"
    body = await request.body()
    headers = {k: v for k, v in request.headers.items() if k.lower() in ("content-type", "accept")}
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=5.0)) as client:
            upstream = await client.request(request.method, url, content=body or None, headers=headers, params=request.query_params)
    except httpx.HTTPError as exc:
        return Response(content=f'{{"error":"ComfyUIに接続できません: {type(exc).__name__}"}}', status_code=502, media_type="application/json")
    return Response(content=upstream.content, status_code=upstream.status_code, media_type=upstream.headers.get("content-type", "application/json"))
