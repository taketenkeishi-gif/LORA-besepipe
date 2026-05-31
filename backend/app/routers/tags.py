from __future__ import annotations

import threading
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException

from ..db import get_conn
from ..schemas import (
    BatchRemoveIn,
    BatchReplaceIn,
    CaptionEditIn,
    CaptionItem,
    GenerateTagsIn,
    TaggerStatusOut,
)
from ..services import tagger as tagger_svc
from ..services.tagger import GENERAL_THRESHOLD, CHARACTER_THRESHOLD

router = APIRouter(prefix="/tags", tags=["tags"])

# project_id → 進捗状態
_TAGGER_STATUS: dict[int, dict] = {}
_STATUS_LOCK = threading.Lock()


def _set_status(project_id: int, status: str, total: int, done: int, message: str) -> None:
    with _STATUS_LOCK:
        _TAGGER_STATUS[project_id] = {
            "status": status,
            "total": total,
            "done": done,
            "message": message,
        }


def _run_tagger(
    project_id: int,
    item_ids: list[int],
    overwrite: bool,
    general_thresh: float = GENERAL_THRESHOLD,
    character_thresh: float = CHARACTER_THRESHOLD,
    remove_character_tags: bool = False,
) -> None:
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, file_path, caption FROM dataset_items WHERE project_id = ? AND selected = 1 ORDER BY id",
        (project_id,),
    ).fetchall()

    targets = [r for r in rows if r["id"] in set(item_ids)] if item_ids else list(rows)
    if not overwrite:
        targets = [r for r in targets if not r["caption"].strip()]

    total = len(targets)
    _set_status(project_id, "running", total, 0, f"0/{total} 処理中...")

    errors: list[str] = []
    for i, row in enumerate(targets, start=1):
        try:
            caption = tagger_svc.predict(
                row["file_path"],
                general_thresh=general_thresh,
                character_thresh=character_thresh,
                remove_character_tags=remove_character_tags,
            )
            conn.execute(
                "UPDATE dataset_items SET caption = ?, caption_source = 'wd14' WHERE id = ?",
                (caption, row["id"]),
            )
            conn.commit()
            # kohya_ss 互換 .txt ファイルに書き出し
            txt_path = Path(row["file_path"]).with_suffix(".txt")
            txt_path.write_text(caption, encoding="utf-8")
        except Exception as exc:
            errors.append(str(exc))

        _set_status(
            project_id,
            "running",
            total,
            i,
            f"{i}/{total} 完了" + (f" ({len(errors)}件エラー)" if errors else ""),
        )

    conn.close()
    final_msg = f"完了: {total - len(errors)}/{total} 枚"
    if errors:
        final_msg += f" | エラー: {errors[0][:80]}"
    _set_status(project_id, "done" if not errors else "done_with_errors", total, total, final_msg)


@router.post("/generate")
def generate_tags(payload: GenerateTagsIn, background_tasks: BackgroundTasks) -> dict:
    conn = get_conn()
    project = conn.execute("SELECT id FROM projects WHERE id = ?", (payload.project_id,)).fetchone()
    conn.close()
    if project is None:
        raise HTTPException(status_code=404, detail=f"project not found: {payload.project_id}")

    if not tagger_svc.is_available():
        raise HTTPException(
            status_code=503,
            detail="onnxruntime / huggingface-hub が未インストールです。pip install onnxruntime-gpu huggingface-hub を実行してください。",
        )

    with _STATUS_LOCK:
        current = _TAGGER_STATUS.get(payload.project_id, {})
        if current.get("status") == "running":
            raise HTTPException(status_code=409, detail="タグ生成が既に実行中です")

    _set_status(payload.project_id, "queued", 0, 0, "モデルをロード中...")
    background_tasks.add_task(
        _run_tagger,
        payload.project_id,
        [],
        payload.overwrite,
        payload.general_thresh,
        payload.character_thresh,
        payload.remove_character_tags,
    )

    return {
        "project_id": payload.project_id,
        "status": "queued",
        "message": "タグ生成をバックグラウンドで開始しました。/tags/status/{project_id} で進捗確認できます。",
        "model_cached": tagger_svc.model_cached(),
    }


@router.get("/status/{project_id}", response_model=TaggerStatusOut)
def get_status(project_id: int) -> TaggerStatusOut:
    with _STATUS_LOCK:
        st = _TAGGER_STATUS.get(project_id)
    if st is None:
        return TaggerStatusOut(
            project_id=project_id,
            status="idle",
            total=0,
            done=0,
            message="未実行",
        )
    return TaggerStatusOut(project_id=project_id, **st)


@router.get("/{project_id}", response_model=list[CaptionItem])
def list_captions(project_id: int) -> list[CaptionItem]:
    conn = get_conn()
    project = conn.execute("SELECT id FROM projects WHERE id = ?", (project_id,)).fetchone()
    if project is None:
        conn.close()
        raise HTTPException(status_code=404, detail=f"project not found: {project_id}")

    rows = conn.execute(
        """
        SELECT id, file_path, width, height, aspect,
               COALESCE(caption, '') AS caption,
               COALESCE(caption_source, '') AS caption_source
        FROM dataset_items
        WHERE project_id = ? AND selected = 1
        ORDER BY id
        """,
        (project_id,),
    ).fetchall()
    conn.close()

    result: list[CaptionItem] = []
    for row in rows:
        caption = row["caption"]
        source = row["caption_source"]
        # .txt ファイルが存在しキャプションが空の場合は同期
        if not caption:
            txt = Path(row["file_path"]).with_suffix(".txt")
            if txt.exists():
                try:
                    caption = txt.read_text(encoding="utf-8", errors="ignore").strip()
                    source = "file"
                except OSError:
                    pass
        result.append(
            CaptionItem(
                id=row["id"],
                file_path=row["file_path"],
                width=row["width"],
                height=row["height"],
                aspect=row["aspect"],
                caption=caption,
                caption_source=source,
            )
        )
    return result


@router.patch("/item/{item_id}")
def edit_caption(item_id: int, payload: CaptionEditIn) -> dict:
    conn = get_conn()
    row = conn.execute("SELECT id, file_path FROM dataset_items WHERE id = ?", (item_id,)).fetchone()
    if row is None:
        conn.close()
        raise HTTPException(status_code=404, detail=f"item not found: {item_id}")

    caption = payload.caption.strip()
    conn.execute(
        "UPDATE dataset_items SET caption = ?, caption_source = 'manual' WHERE id = ?",
        (caption, item_id),
    )
    conn.commit()
    conn.close()

    txt_path = Path(row["file_path"]).with_suffix(".txt")
    try:
        txt_path.write_text(caption, encoding="utf-8")
    except OSError:
        pass

    return {"id": item_id, "caption": caption, "caption_source": "manual"}


@router.delete("/item/{item_id}")
def delete_item(item_id: int) -> dict:
    """データセットからアイテムを削除（ファイルは残す）"""
    conn = get_conn()
    row = conn.execute("SELECT id, project_id FROM dataset_items WHERE id = ?", (item_id,)).fetchone()
    if row is None:
        conn.close()
        raise HTTPException(status_code=404, detail=f"item not found: {item_id}")
    project_id = row["project_id"]
    conn.execute("DELETE FROM dataset_items WHERE id = ?", (item_id,))
    conn.commit()
    conn.close()
    return {"deleted": item_id, "project_id": project_id}


@router.post("/batch-replace")
def batch_replace(payload: BatchReplaceIn) -> dict:
    conn = get_conn()
    project = conn.execute("SELECT id FROM projects WHERE id = ?", (payload.project_id,)).fetchone()
    if project is None:
        conn.close()
        raise HTTPException(status_code=404, detail=f"project not found: {payload.project_id}")

    if payload.item_ids:
        rows = conn.execute(
            f"SELECT id, file_path, caption FROM dataset_items WHERE id IN ({','.join('?' * len(payload.item_ids))})",
            payload.item_ids,
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, file_path, caption FROM dataset_items WHERE project_id = ? AND selected = 1",
            (payload.project_id,),
        ).fetchall()

    updated = 0
    for row in rows:
        old_caption = row["caption"] or ""
        if payload.find not in old_caption:
            continue
        new_caption = old_caption.replace(payload.find, payload.replace)
        conn.execute(
            "UPDATE dataset_items SET caption = ? WHERE id = ?",
            (new_caption, row["id"]),
        )
        try:
            Path(row["file_path"]).with_suffix(".txt").write_text(new_caption, encoding="utf-8")
        except OSError:
            pass
        updated += 1

    conn.commit()
    conn.close()
    return {
        "project_id": payload.project_id,
        "updated_count": updated,
        "find": payload.find,
        "replace": payload.replace,
    }


@router.post("/batch-remove")
def batch_remove(payload: BatchRemoveIn) -> dict:
    conn = get_conn()
    project = conn.execute("SELECT id FROM projects WHERE id = ?", (payload.project_id,)).fetchone()
    if project is None:
        conn.close()
        raise HTTPException(status_code=404, detail=f"project not found: {payload.project_id}")

    if payload.item_ids:
        rows = conn.execute(
            f"SELECT id, file_path, caption FROM dataset_items WHERE id IN ({','.join('?' * len(payload.item_ids))})",
            payload.item_ids,
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, file_path, caption FROM dataset_items WHERE project_id = ? AND selected = 1",
            (payload.project_id,),
        ).fetchall()

    remove_set = {t.strip().lower() for t in payload.tags if t.strip()}
    updated = 0
    for row in rows:
        old_caption = row["caption"] or ""
        parts = [t.strip() for t in old_caption.split(",") if t.strip()]
        new_parts = [t for t in parts if t.lower() not in remove_set]
        if len(new_parts) == len(parts):
            continue
        new_caption = ", ".join(new_parts)
        conn.execute(
            "UPDATE dataset_items SET caption = ? WHERE id = ?",
            (new_caption, row["id"]),
        )
        try:
            Path(row["file_path"]).with_suffix(".txt").write_text(new_caption, encoding="utf-8")
        except OSError:
            pass
        updated += 1

    conn.commit()
    conn.close()
    return {
        "project_id": payload.project_id,
        "updated_count": updated,
        "removed_tags": list(remove_set),
    }


@router.get("/frequency/{project_id}")
def tag_frequency(project_id: int) -> dict:
    """全キャプションにおけるタグ出現頻度を返す。"""
    conn = get_conn()
    project = conn.execute("SELECT id FROM projects WHERE id = ?", (project_id,)).fetchone()
    if project is None:
        conn.close()
        raise HTTPException(status_code=404, detail=f"project not found: {project_id}")

    rows = conn.execute(
        "SELECT caption FROM dataset_items WHERE project_id = ? AND selected = 1",
        (project_id,),
    ).fetchall()
    conn.close()

    freq: dict[str, int] = {}
    for row in rows:
        for tag in (row["caption"] or "").split(","):
            t = tag.strip()
            if t:
                freq[t] = freq.get(t, 0) + 1

    sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return {
        "project_id": project_id,
        "total_items": len(rows),
        "unique_tags": len(freq),
        "frequency": [{"tag": k, "count": v} for k, v in sorted_freq],
    }
