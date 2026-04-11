from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

from ..db import get_conn

from ..schemas import GenerateTagsIn

router = APIRouter(prefix="/tags", tags=["tags"])


@router.post("/generate")
def generate_tags(payload: GenerateTagsIn) -> dict:
    conn = get_conn()
    project = conn.execute(
        "SELECT id, captions_dir FROM projects WHERE id = ?",
        (payload.project_id,),
    ).fetchone()
    if project is None:
        conn.close()
        raise HTTPException(status_code=404, detail=f"project not found: {payload.project_id}")

    rows = conn.execute(
        """
        SELECT id, file_path FROM dataset_items
        WHERE project_id = ? AND selected = 1
        ORDER BY id ASC
        """,
        (payload.project_id,),
    ).fetchall()
    captions_dir = Path(project["captions_dir"])
    captions_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for row in rows:
        stem = Path(row["file_path"]).stem
        out = captions_dir / f"{stem}.txt"
        # 初期版は固定タグ出力で流れを通す
        out.write_text("masterpiece, best quality, lora_subject", encoding="utf-8")
        written.append(str(out))
    conn.close()
    return {
        "project_id": payload.project_id,
        "generated_count": len(written),
        "files": written,
        "message": "placeholder tag files generated (WD14統合は未実装)",
    }
