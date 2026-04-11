from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/previews", tags=["previews"])


@router.get("/{project_id}")
def list_previews(project_id: int) -> dict:
    return {
        "project_id": project_id,
        "slots": ["face", "bust", "full", "bg"],
        "timeline": [],
        "message": "TODO: epoch履歴と4スロット画像を返す",
    }

