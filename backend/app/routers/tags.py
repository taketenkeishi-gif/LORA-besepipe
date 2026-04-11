from __future__ import annotations

from fastapi import APIRouter

from ..schemas import GenerateTagsIn

router = APIRouter(prefix="/tags", tags=["tags"])


@router.post("/generate")
def generate_tags(payload: GenerateTagsIn) -> dict:
    return {
        "project_id": payload.project_id,
        "message": "TODO: WD14タグ生成とtxt出力を実装",
    }

