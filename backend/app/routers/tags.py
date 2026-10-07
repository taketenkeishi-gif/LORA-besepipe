from __future__ import annotations

import threading
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException

from ..db import get_conn
import json

from ..schemas import (
    ApplyPrefixIn,
    BatchRemoveIn,
    BatchReplaceIn,
    CaptionEditIn,
    CaptionItem,
    GenerateTagsIn,
    RemoveBlockWordsIn,
    TagSettingsIn,
    TagSettingsOut,
    TaggerStatusOut,
)
from ..services import tagger as tagger_svc
from ..services.tagger import GENERAL_THRESHOLD, CHARACTER_THRESHOLD

router = APIRouter(prefix="/tags", tags=["tags"])

# project_id → 進捗状態
_TAGGER_STATUS: dict[int, dict] = {}
_STATUS_LOCK = threading.Lock()
# 途中停止リクエストを受けた project_id の集合
_STOP_REQUESTED: set[int] = set()


def _set_status(project_id: int, status: str, total: int, done: int, message: str) -> None:
    with _STATUS_LOCK:
        _TAGGER_STATUS[project_id] = {
            "status": status,
            "total": total,
            "done": done,
            "message": message,
        }


def _load_tag_settings(project_id: int) -> tuple[list[str], list[str]]:
    """プロジェクトの prefix_tags / block_words を返す。未設定なら空リスト。"""
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT prefix_tags_json, block_words_json FROM project_tag_settings WHERE project_id = ?",
            (project_id,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return [], []
    try:
        prefix = [t.strip() for t in json.loads(row["prefix_tags_json"]) if t and t.strip()]
        block = [w.strip() for w in json.loads(row["block_words_json"]) if w and w.strip()]
        return prefix, block
    except (ValueError, TypeError):
        return [], []


def _compose_caption(wd14_caption: str, prefix_tags: list[str], block_words: list[str]) -> str:
    """WD14生成タグに、固定(prefix)タグを先頭付与＋ブロックワード除去（重複除去）。
    再生成のたびに prefix が消えないようにするための合成処理。"""
    block_set = {w.lower() for w in block_words}
    body = [t.strip() for t in wd14_caption.split(",") if t.strip()]
    prefix_set = {p.lower() for p in prefix_tags}
    # body から prefix 重複・block を除去
    body = [t for t in body if t.lower() not in prefix_set and t.lower() not in block_set]
    ordered = [p for p in prefix_tags if p.lower() not in block_set] + body
    # 念のため順序維持の重複除去
    seen: set[str] = set()
    out: list[str] = []
    for t in ordered:
        k = t.lower()
        if k not in seen:
            seen.add(k)
            out.append(t)
    return ", ".join(out)


def _run_tagger(
    project_id: int,
    item_ids: list[int],
    overwrite: bool,
    general_thresh: float = GENERAL_THRESHOLD,
    character_thresh: float = CHARACTER_THRESHOLD,
    remove_character_tags: bool = False,
) -> None:
    prefix_tags, block_words = _load_tag_settings(project_id)
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, file_path, caption FROM dataset_items WHERE project_id = ? AND selected = 1 ORDER BY id",
        (project_id,),
    ).fetchall()

    targets = [r for r in rows if r["id"] in set(item_ids)] if item_ids else list(rows)
    if not overwrite:
        targets = [r for r in targets if not r["caption"].strip()]

    total = len(targets)
    # 開始時に残っている停止フラグをクリア
    _STOP_REQUESTED.discard(project_id)
    _set_status(project_id, "running", total, 0, f"0/{total} 処理中...")

    errors: list[str] = []
    stopped = False
    for i, row in enumerate(targets, start=1):
        # 各画像処理の前に停止要求をチェック
        if project_id in _STOP_REQUESTED:
            _STOP_REQUESTED.discard(project_id)
            stopped = True
            break
        try:
            caption = tagger_svc.predict(
                row["file_path"],
                general_thresh=general_thresh,
                character_thresh=character_thresh,
                remove_character_tags=remove_character_tags,
            )
            # 固定タグを先頭付与＋ブロックワード除去（再生成で消えないように）
            if prefix_tags or block_words:
                caption = _compose_caption(caption, prefix_tags, block_words)
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
    done_count = locals().get("i", 0) if stopped else total
    if stopped:
        _set_status(project_id, "stopped", total, done_count, f"停止しました（{done_count - 1}/{total} 枚処理済み）")
        return
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


@router.post("/stop/{project_id}")
def stop_tagging(project_id: int) -> dict:
    """実行中のタグ生成に途中停止を要求する。次の画像処理の前に停止する。"""
    with _STATUS_LOCK:
        current = _TAGGER_STATUS.get(project_id, {})
        st = current.get("status")
    if st not in ("running", "queued"):
        return {"project_id": project_id, "stopping": False, "message": "実行中のタグ生成はありません"}
    _STOP_REQUESTED.add(project_id)
    return {"project_id": project_id, "stopping": True, "message": "停止を要求しました。処理中の画像が終わり次第停止します。"}


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


@router.get("/settings/{project_id}", response_model=TagSettingsOut)
def get_tag_settings(project_id: int) -> TagSettingsOut:
    conn = get_conn()
    row = conn.execute(
        "SELECT prefix_tags_json, block_words_json FROM project_tag_settings WHERE project_id = ?",
        (project_id,),
    ).fetchone()
    conn.close()
    if row is None:
        return TagSettingsOut(project_id=project_id, prefix_tags=[], block_words=[])
    return TagSettingsOut(
        project_id=project_id,
        prefix_tags=json.loads(row["prefix_tags_json"]),
        block_words=json.loads(row["block_words_json"]),
    )


@router.put("/settings/{project_id}", response_model=TagSettingsOut)
def put_tag_settings(project_id: int, payload: TagSettingsIn) -> TagSettingsOut:
    conn = get_conn()
    conn.execute(
        """
        INSERT INTO project_tag_settings (project_id, prefix_tags_json, block_words_json, updated_at)
        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(project_id) DO UPDATE SET
            prefix_tags_json = excluded.prefix_tags_json,
            block_words_json = excluded.block_words_json,
            updated_at = CURRENT_TIMESTAMP
        """,
        (project_id, json.dumps(payload.prefix_tags, ensure_ascii=False),
         json.dumps(payload.block_words, ensure_ascii=False)),
    )
    conn.commit()
    conn.close()
    return TagSettingsOut(project_id=project_id, prefix_tags=payload.prefix_tags, block_words=payload.block_words)


@router.post("/apply-prefix")
def apply_prefix(payload: ApplyPrefixIn) -> dict:
    """全キャプションの先頭にprefixタグを付与（重複スキップ）。"""
    if not payload.prefix_tags:
        return {"project_id": payload.project_id, "updated_count": 0}
    prefix_str = ", ".join(t.strip() for t in payload.prefix_tags if t.strip())
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, file_path, caption FROM dataset_items WHERE project_id = ? AND selected = 1",
        (payload.project_id,),
    ).fetchall()
    updated = 0
    for row in rows:
        old = (row["caption"] or "").strip()
        parts = [t.strip() for t in old.split(",") if t.strip()]
        # 既に先頭にある場合はスキップ
        existing_prefixes = {t.strip() for t in payload.prefix_tags if t.strip()}
        leading = {p for p in parts[:len(payload.prefix_tags)] if p in existing_prefixes}
        if leading == existing_prefixes and old.startswith(prefix_str):
            continue
        # 重複を除去してから先頭に追加
        body = [p for p in parts if p not in existing_prefixes]
        new_caption = ", ".join([prefix_str] + body) if body else prefix_str
        conn.execute(
            "UPDATE dataset_items SET caption = ?, caption_source = 'manual' WHERE id = ?",
            (new_caption, row["id"]),
        )
        try:
            Path(row["file_path"]).with_suffix(".txt").write_text(new_caption, encoding="utf-8")
        except OSError:
            pass
        updated += 1
    conn.commit()
    conn.close()
    return {"project_id": payload.project_id, "updated_count": updated, "prefix": prefix_str}


@router.post("/remove-blockwords")
def remove_blockwords(payload: RemoveBlockWordsIn) -> dict:
    """全キャプションからブロックワードを削除し、検出数を返す。"""
    if not payload.block_words:
        return {"project_id": payload.project_id, "updated_count": 0, "total_removed": 0}
    block_set = {w.strip().lower() for w in payload.block_words if w.strip()}
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, file_path, caption FROM dataset_items WHERE project_id = ? AND selected = 1",
        (payload.project_id,),
    ).fetchall()
    updated = 0
    total_removed = 0
    for row in rows:
        parts = [t.strip() for t in (row["caption"] or "").split(",") if t.strip()]
        new_parts = [p for p in parts if p.lower() not in block_set]
        removed = len(parts) - len(new_parts)
        if removed == 0:
            continue
        total_removed += removed
        new_caption = ", ".join(new_parts)
        conn.execute(
            "UPDATE dataset_items SET caption = ?, caption_source = 'manual' WHERE id = ?",
            (new_caption, row["id"]),
        )
        try:
            Path(row["file_path"]).with_suffix(".txt").write_text(new_caption, encoding="utf-8")
        except OSError:
            pass
        updated += 1
    conn.commit()
    conn.close()
    return {"project_id": payload.project_id, "updated_count": updated, "total_removed": total_removed}


@router.get("/detect-blockwords/{project_id}")
def detect_blockwords(project_id: int, words: str = "") -> dict:
    """指定ワードがキャプション内に何件存在するかを返す（削除前プレビュー用）。"""
    if not words:
        return {"project_id": project_id, "hits": {}}
    word_list = [w.strip().lower() for w in words.split(",") if w.strip()]
    conn = get_conn()
    rows = conn.execute(
        "SELECT caption FROM dataset_items WHERE project_id = ? AND selected = 1",
        (project_id,),
    ).fetchall()
    conn.close()
    hits: dict[str, int] = {w: 0 for w in word_list}
    for row in rows:
        parts = {t.strip().lower() for t in (row["caption"] or "").split(",") if t.strip()}
        for w in word_list:
            if w in parts:
                hits[w] += 1
    return {"project_id": project_id, "hits": hits}


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



def _color_histogram(file_path: str, bins: int = 16) -> list[float] | None:
    """RGB 3ch それぞれ bins ビンのヒストグラムを正規化して返す。"""
    try:
        from PIL import Image
        import math
        img = Image.open(file_path).convert("RGB").resize((64, 64))
        pixels = list(img.getdata())
        hist = [0.0] * (bins * 3)
        step = 256 / bins
        for r, g, b in pixels:
            hist[int(r / step)] += 1
            hist[bins + int(g / step)] += 1
            hist[bins * 2 + int(b / step)] += 1
        total = len(pixels)
        norm_hist = [v / total for v in hist]
        # L2 正規化
        magnitude = math.sqrt(sum(v * v for v in norm_hist)) or 1.0
        return [v / magnitude for v in norm_hist]
    except Exception:
        return None


def _histogram_cosine(h1: list[float] | None, h2: list[float] | None) -> float:
    if h1 is None or h2 is None:
        return 0.0
    return sum(a * b for a, b in zip(h1, h2))


@router.post("/auto-classify")
def auto_classify(body: dict) -> dict:
    """
    プロファイルの参照画像タグ＋カラーヒストグラムを基準に全データセット画像を自動分類する。
    color_weight=0.0 でタグのみ、1.0 でカラーのみ（デフォルト 0.4）。
    """
    from collections import Counter
    project_id: int = body.get("project_id")
    profiles: list = body.get("profiles", [])
    min_score: float = body.get("min_score", 0.15)
    color_weight: float = float(body.get("color_weight", 0.4))
    color_weight = max(0.0, min(1.0, color_weight))
    tag_weight = 1.0 - color_weight

    if not project_id or not profiles:
        return {"profiles": [], "unclassified": [], "total": 0}

    conn = get_conn()
    try:
        all_items = conn.execute(
            "SELECT id, file_path, caption FROM dataset_items WHERE project_id = ? AND caption IS NOT NULL AND caption != ''",
            (project_id,),
        ).fetchall()

        profile_signatures: list[set] = []
        profile_color_hists: list[list[float] | None] = []

        for profile in profiles:
            ref_ids = profile.get("ref_item_ids", [])
            if not ref_ids:
                profile_signatures.append(set())
                profile_color_hists.append(None)
                continue

            placeholders = ",".join("?" * len(ref_ids))
            ref_rows = conn.execute(
                f"SELECT file_path, caption FROM dataset_items WHERE id IN ({placeholders}) AND project_id = ?",
                (*ref_ids, project_id),
            ).fetchall()

            # タグシグネチャ（参照画像の過半数に出現するタグ）
            tag_counter: Counter = Counter()
            for row in ref_rows:
                tags = {t.strip() for t in (row["caption"] or "").split(",") if t.strip()}
                for t in tags:
                    tag_counter[t] += 1
            threshold = max(1, len(ref_rows) * 0.5)
            signature = {tag for tag, cnt in tag_counter.items() if cnt >= threshold}
            profile_signatures.append(signature)

            # カラーヒストグラム（参照画像の平均）
            hists = [_color_histogram(row["file_path"]) for row in ref_rows]
            valid = [h for h in hists if h is not None]
            if valid:
                bins3 = len(valid[0])
                avg_hist = [sum(h[i] for h in valid) / len(valid) for i in range(bins3)]
                # 再正規化
                import math
                mag = math.sqrt(sum(v * v for v in avg_hist)) or 1.0
                profile_color_hists.append([v / mag for v in avg_hist])
            else:
                profile_color_hists.append(None)

    finally:
        conn.close()

    result_profiles: list = [
        {"name": p["name"], "trigger_word": p["trigger_word"], "items": []}
        for p in profiles
    ]
    unclassified: list = []

    for item in all_items:
        item_tags = {t.strip() for t in (item["caption"] or "").split(",") if t.strip()}
        item_hist = _color_histogram(item["file_path"]) if color_weight > 0 else None

        best_idx = -1
        best_score = min_score

        for i, sig in enumerate(profile_signatures):
            # タグスコア
            tag_score = 0.0
            if sig:
                intersection = len(item_tags & sig)
                tag_score = intersection / len(sig)

            # カラースコア
            color_score = 0.0
            if color_weight > 0:
                color_score = _histogram_cosine(item_hist, profile_color_hists[i])

            score = tag_weight * tag_score + color_weight * color_score

            if score > best_score:
                best_score = score
                best_idx = i

        entry = {
            "id": item["id"],
            "file_path": item["file_path"],
            "caption": item["caption"],
            "score": round(best_score, 3),
        }
        if best_idx >= 0:
            result_profiles[best_idx]["items"].append(entry)
        else:
            unclassified.append(entry)

    for p in result_profiles:
        p["items"].sort(key=lambda x: x["score"], reverse=True)

    return {
        "profiles": result_profiles,
        "unclassified": unclassified,
        "total": len(all_items),
    }


@router.post("/apply-profile-classification")
def apply_profile_classification(body: dict) -> dict:
    """
    分類結果をキャプションに適用する（トリガーワードを先頭に追加）。
    """
    import os
    project_id: int = body.get("project_id")
    assignments: list = body.get("assignments", [])
    overwrite: bool = body.get("overwrite", False)

    if not project_id or not assignments:
        return {"updated": 0, "skipped": 0}

    conn = get_conn()
    updated = 0
    skipped = 0
    try:
        for asgn in assignments:
            item_id = asgn["item_id"]
            trigger = asgn["trigger_word"].strip().strip(",").strip()
            if not trigger:
                continue

            row = conn.execute(
                "SELECT id, file_path, caption FROM dataset_items WHERE id = ? AND project_id = ?",
                (item_id, project_id),
            ).fetchone()
            if row is None:
                skipped += 1
                continue

            current = (row["caption"] or "").strip()

            if not overwrite and trigger.lower() in current.lower():
                skipped += 1
                continue

            new_caption = f"{trigger}, {current}" if current else trigger
            conn.execute(
                "UPDATE dataset_items SET caption = ?, caption_source = 'manual' WHERE id = ?",
                (new_caption, item_id),
            )

            txt_path = os.path.splitext(row["file_path"])[0] + ".txt"
            try:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(new_caption)
            except Exception:
                pass

            updated += 1

        conn.commit()
    finally:
        conn.close()

    return {"updated": updated, "skipped": skipped}


@router.get("/outfit-profiles/{project_id}")
def get_outfit_profiles(project_id: int) -> dict:
    """プロジェクトに保存されている衣装プロファイル定義を返す。"""
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT profiles_json FROM project_outfit_profiles WHERE project_id = ?",
            (project_id,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return {"project_id": project_id, "profiles": []}
    try:
        profiles = json.loads(row["profiles_json"])
    except (ValueError, TypeError):
        profiles = []
    return {"project_id": project_id, "profiles": profiles}


@router.put("/outfit-profiles/{project_id}")
def save_outfit_profiles(project_id: int, body: dict) -> dict:
    """衣装プロファイル定義をプロジェクトに保存する。"""
    profiles = body.get("profiles", [])
    profiles_json = json.dumps(profiles, ensure_ascii=False)
    conn = get_conn()
    try:
        conn.execute(
            """
            INSERT INTO project_outfit_profiles (project_id, profiles_json, updated_at)
            VALUES (?, ?, datetime('now'))
            ON CONFLICT(project_id) DO UPDATE SET profiles_json = excluded.profiles_json, updated_at = excluded.updated_at
            """,
            (project_id, profiles_json),
        )
        conn.commit()
    finally:
        conn.close()
    return {"project_id": project_id, "saved": len(profiles)}


@router.post("/suggest-profile-triggers")
def suggest_profile_triggers(body: dict) -> dict:
    """選択した dataset_items の既存 caption から共通タグを集計して提案する。"""
    item_ids: list[int] = body.get("item_ids", [])
    project_id: int = body.get("project_id")
    min_count: int = body.get("min_count", 2)

    if not item_ids or not project_id:
        return {"tags": [], "total_images": 0}

    conn = get_conn()
    try:
        placeholders = ",".join("?" * len(item_ids))
        rows = conn.execute(
            f"SELECT caption FROM dataset_items WHERE id IN ({placeholders}) AND project_id = ?",
            (*item_ids, project_id),
        ).fetchall()
    finally:
        conn.close()

    from collections import Counter
    counter: Counter = Counter()
    total = len(rows)
    for row in rows:
        caption = row["caption"] or ""
        tags = [t.strip() for t in caption.split(",") if t.strip()]
        for tag in set(tags):
            counter[tag] += 1

    results = [
        {"tag": tag, "count": cnt, "ratio": round(cnt / total, 2) if total else 0}
        for tag, cnt in counter.most_common(50)
        if cnt >= min_count
    ]
    return {"tags": results, "total_images": total}
