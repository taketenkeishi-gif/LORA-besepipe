from __future__ import annotations

from fastapi import APIRouter

from ..schemas import CollectorImportIn, CollectorScanIn

router = APIRouter(prefix="/collector", tags=["collector"])


@router.post("/scan")
def scan(payload: CollectorScanIn) -> dict:
    return {
        "project_id": payload.project_id,
        "url": payload.url,
        "detected": 0,
        "items": [],
        "message": "TODO: URLから画像候補抽出を実装",
    }


@router.post("/import")
def import_selected(payload: CollectorImportIn) -> dict:
    return {
        "project_id": payload.project_id,
        "imported_count": len(payload.selected_ids),
        "naming_template": payload.naming_template,
        "message": "TODO: 命名規則でdatasetへ保存を実装",
    }

