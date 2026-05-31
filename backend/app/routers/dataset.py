from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

from fastapi import APIRouter, BackgroundTasks, HTTPException
from PIL import Image, UnidentifiedImageError

from ..db import get_conn
from ..services.tag_categories import (
    CATEGORIES,
    HAIR_COLORS,
    HAIR_STYLES,
    EYE_COLORS,
    COSTUME_TYPES,
    LEAK_RISK_COMBOS,
    classify_caption,
    detect_dominant_feature,
)
from ..services.tagger import CACHE_DIR, TAGS_FILENAME
from ..services.image_search import ImageSearchService

router = APIRouter(prefix="/dataset", tags=["dataset"])

_ANALYSIS_STATUS: dict[int, dict] = {}
_STATUS_LOCK = threading.Lock()


def _set_status(project_id: int, status: str, message: str) -> None:
    with _STATUS_LOCK:
        _ANALYSIS_STATUS[project_id] = {"status": status, "message": message}


# ── 彩度チェック（モノクロ判定） ──────────────────────────────────────────
def _is_monochrome(path: str) -> bool:
    try:
        with Image.open(path) as im:
            rgb = im.convert("RGB")
            r, g, b = rgb.split()
            r_data = list(r.getdata())
            g_data = list(g.getdata())
            b_data = list(b.getdata())
            total = len(r_data)
            if total == 0:
                return False
            diff = sum(
                abs(r_data[i] - g_data[i]) + abs(g_data[i] - b_data[i])
                for i in range(min(total, 2000))  # サンプリングで高速化
            )
            avg_diff = diff / min(total, 2000)
            return avg_diff < 8
    except (OSError, UnidentifiedImageError):
        return False


# ── pHash計算 ─────────────────────────────────────────────────────────────
def _compute_phash(path: str):
    try:
        import imagehash
        with Image.open(path) as im:
            return imagehash.phash(im)
    except Exception:
        return None


# ── 品質スコア計算（§8.3） ───────────────────────────────────────────────
def _calc_quality_score(items: list[dict], duplicate_pairs: int, similar_pairs: int) -> tuple[int, list[str]]:
    total = len(items)
    if total == 0:
        return 0, []

    warnings: list[str] = []
    score = 100

    # 低解像度 (< 512×512)
    low_res = [i for i in items if i["width"] * i["height"] < 512 * 512]
    if low_res:
        penalty = min(20, int(len(low_res) / total * 100 * 0.25))
        score -= penalty
        warnings.append(f"低解像度画像: {len(low_res)}枚 ({len(low_res) * 100 // total}%)")

    # 未キャプション
    no_caption = [i for i in items if not (i.get("caption") or "").strip()]
    if no_caption:
        penalty = min(30, int(len(no_caption) / total * 100 * 0.4))
        score -= penalty
        warnings.append(f"未キャプション: {len(no_caption)}枚")

    # 重複画像
    if duplicate_pairs > 0:
        penalty = min(20, duplicate_pairs * 5)
        score -= penalty
        warnings.append(f"重複画像: {duplicate_pairs}組")

    # 類似画像過多（構図偏重）
    if similar_pairs > total * 0.3:
        score -= 5
        warnings.append(f"類似画像が多い: {similar_pairs}組（構図偏重の可能性）")

    # モノクロ偏重
    mono_count = sum(1 for i in items if i.get("_mono"))
    if mono_count > total * 0.5:
        score -= 5
        warnings.append(f"白黒画像が多い: {mono_count}枚 ({mono_count * 100 // total}%)")

    # アスペクト偏重（1種類に90%以上集中）
    aspects = [i.get("aspect", "") for i in items]
    for asp in ["portrait", "landscape", "square"]:
        ratio = aspects.count(asp) / total
        if ratio > 0.9:
            score -= 3
            warnings.append(f"アスペクト偏重: {asp} が {ratio * 100:.0f}%")

    return max(0, score), warnings


# ── pHash分析ジョブ ───────────────────────────────────────────────────────
def _run_analysis(project_id: int) -> None:
    _set_status(project_id, "running", "画像を読み込み中...")

    conn = get_conn()
    rows = conn.execute(
        "SELECT id, file_path, width, height, aspect, caption FROM dataset_items WHERE project_id = ? AND selected = 1",
        (project_id,),
    ).fetchall()

    items = [dict(r) for r in rows]
    total = len(items)

    if total == 0:
        conn.execute(
            "INSERT INTO dataset_analysis(project_id, quality_score, stats_json, similarity_json) VALUES(?,0,'{}','[]') ON CONFLICT(project_id) DO UPDATE SET quality_score=0, stats_json='{}', similarity_json='[]', analyzed_at=CURRENT_TIMESTAMP",
            (project_id,),
        )
        conn.commit()
        conn.close()
        _set_status(project_id, "done", "データがありません")
        return

    # モノクロ判定
    _set_status(project_id, "running", "カラー分析中...")
    for item in items:
        item["_mono"] = _is_monochrome(item["file_path"])

    # pHash計算 + 類似グループ検出
    _set_status(project_id, "running", "pHash計算中...")
    hashes: list[tuple[int, object]] = []
    try:
        import imagehash
        _imagehash_available = True
    except ImportError:
        _imagehash_available = False

    similarity_groups: list[dict] = []
    duplicate_pairs = 0
    similar_pairs = 0

    if _imagehash_available:
        for item in items:
            h = _compute_phash(item["file_path"])
            if h is not None:
                hashes.append((item["id"], h))

        # 全ペア比較（O(n²)だが実用上は数百枚程度）
        processed: set[int] = set()
        for i in range(len(hashes)):
            if hashes[i][0] in processed:
                continue
            group_dups = [hashes[i][0]]
            group_sims = [hashes[i][0]]
            for j in range(i + 1, len(hashes)):
                if hashes[j][0] in processed:
                    continue
                dist = hashes[i][1] - hashes[j][1]
                if dist == 0:
                    group_dups.append(hashes[j][0])
                    group_sims.append(hashes[j][0])
                    processed.add(hashes[j][0])
                elif dist <= 10:
                    group_sims.append(hashes[j][0])
            if len(group_dups) > 1:
                similarity_groups.append({"type": "duplicate", "item_ids": group_dups})
                processed.add(hashes[i][0])
                duplicate_pairs += len(group_dups) - 1
            elif len(group_sims) > 1:
                similarity_groups.append({"type": "similar", "item_ids": group_sims})
                processed.add(hashes[i][0])
                similar_pairs += len(group_sims) - 1

    # 品質スコア計算
    quality_score, warnings = _calc_quality_score(items, duplicate_pairs, similar_pairs)

    # 統計
    widths = [i["width"] for i in items if i["width"] > 0]
    heights = [i["height"] for i in items if i["height"] > 0]
    captioned = sum(1 for i in items if (i.get("caption") or "").strip())
    mono_count = sum(1 for i in items if i.get("_mono"))
    color_count = total - mono_count

    resolution_dist = {
        "low": sum(1 for i in items if i["width"] * i["height"] < 512 * 512),
        "medium": sum(1 for i in items if 512 * 512 <= i["width"] * i["height"] < 1024 * 1024),
        "high": sum(1 for i in items if i["width"] * i["height"] >= 1024 * 1024),
    }
    aspect_dist = {
        "portrait": sum(1 for i in items if i.get("aspect") == "portrait"),
        "landscape": sum(1 for i in items if i.get("aspect") == "landscape"),
        "square": sum(1 for i in items if i.get("aspect") == "square"),
    }

    stats = {
        "total": total,
        "captioned": captioned,
        "caption_rate": round(captioned / total * 100, 1) if total else 0,
        "mono_count": mono_count,
        "color_count": color_count,
        "avg_width": round(sum(widths) / len(widths), 0) if widths else 0,
        "avg_height": round(sum(heights) / len(heights), 0) if heights else 0,
        "resolution": resolution_dist,
        "aspect": aspect_dist,
        "duplicate_pairs": duplicate_pairs,
        "similar_pairs": similar_pairs,
        "quality_score": quality_score,
        "warnings": warnings,
        "imagehash_available": _imagehash_available,
    }

    conn.execute(
        """INSERT INTO dataset_analysis(project_id, quality_score, stats_json, similarity_json)
           VALUES(?, ?, ?, ?)
           ON CONFLICT(project_id) DO UPDATE SET
             quality_score=excluded.quality_score,
             stats_json=excluded.stats_json,
             similarity_json=excluded.similarity_json,
             analyzed_at=CURRENT_TIMESTAMP""",
        (project_id, quality_score, json.dumps(stats, ensure_ascii=False), json.dumps(similarity_groups, ensure_ascii=False)),
    )
    conn.commit()
    conn.close()
    _set_status(project_id, "done", f"分析完了 — Quality Score: {quality_score}")


# ── API エンドポイント ────────────────────────────────────────────────────

@router.post("/analyze/{project_id}")
def start_analysis(project_id: int, background_tasks: BackgroundTasks) -> dict:
    conn = get_conn()
    project = conn.execute("SELECT id FROM projects WHERE id = ?", (project_id,)).fetchone()
    conn.close()
    if project is None:
        raise HTTPException(status_code=404, detail=f"project not found: {project_id}")

    with _STATUS_LOCK:
        if _ANALYSIS_STATUS.get(project_id, {}).get("status") == "running":
            raise HTTPException(status_code=409, detail="分析が既に実行中です")

    _set_status(project_id, "queued", "分析を準備中...")
    background_tasks.add_task(_run_analysis, project_id)
    return {"project_id": project_id, "status": "queued"}


@router.get("/analysis-status/{project_id}")
def get_analysis_status(project_id: int) -> dict:
    with _STATUS_LOCK:
        st = _ANALYSIS_STATUS.get(project_id, {"status": "idle", "message": "未分析"})
    return {"project_id": project_id, **st}


@router.get("/stats/{project_id}")
def get_stats(project_id: int) -> dict:
    conn = get_conn()
    project = conn.execute("SELECT id FROM projects WHERE id = ?", (project_id,)).fetchone()
    if project is None:
        conn.close()
        raise HTTPException(status_code=404, detail=f"project not found: {project_id}")

    row = conn.execute(
        "SELECT quality_score, stats_json, similarity_json, analyzed_at FROM dataset_analysis WHERE project_id = ?",
        (project_id,),
    ).fetchone()

    if row is None:
        # 分析未実施の場合はDBだけで計算できる基礎統計を返す
        items = conn.execute(
            "SELECT width, height, aspect, caption FROM dataset_items WHERE project_id = ? AND selected = 1",
            (project_id,),
        ).fetchall()
        conn.close()
        total = len(items)
        captioned = sum(1 for i in items if (i["caption"] or "").strip())
        resolution = {
            "low": sum(1 for i in items if i["width"] * i["height"] < 512 * 512),
            "medium": sum(1 for i in items if 512 * 512 <= i["width"] * i["height"] < 1024 * 1024),
            "high": sum(1 for i in items if i["width"] * i["height"] >= 1024 * 1024),
        }
        aspect = {
            "portrait": sum(1 for i in items if i["aspect"] == "portrait"),
            "landscape": sum(1 for i in items if i["aspect"] == "landscape"),
            "square": sum(1 for i in items if i["aspect"] == "square"),
        }
        return {
            "project_id": project_id,
            "analyzed": False,
            "quality_score": None,
            "stats": {
                "total": total,
                "captioned": captioned,
                "caption_rate": round(captioned / total * 100, 1) if total else 0,
                "resolution": resolution,
                "aspect": aspect,
            },
            "similarity_groups": [],
            "analyzed_at": None,
        }

    conn.close()
    stats = json.loads(row["stats_json"])
    similarity = json.loads(row["similarity_json"])
    return {
        "project_id": project_id,
        "analyzed": True,
        "quality_score": row["quality_score"],
        "stats": stats,
        "similarity_groups": similarity,
        "analyzed_at": row["analyzed_at"],
    }


@router.get("/similarity/{project_id}")
def get_similarity(project_id: int) -> dict:
    conn = get_conn()
    row = conn.execute(
        "SELECT similarity_json FROM dataset_analysis WHERE project_id = ?",
        (project_id,),
    ).fetchone()
    conn.close()
    if row is None:
        return {"project_id": project_id, "groups": []}
    groups = json.loads(row["similarity_json"])
    return {"project_id": project_id, "groups": groups}


# ── §11 Character Leak Analysis ───────────────────────────────────────────

@router.get("/character-leak/{project_id}")
def character_leak(project_id: int) -> dict:
    """
    §11 Character Leak Analysis
    キャプションからキャラ固有特徴の偏りを検出し、
    Style LoRA 汚染リスクを警告する。
    """
    conn = get_conn()
    project = conn.execute("SELECT id, project_type FROM projects WHERE id = ?", (project_id,)).fetchone()
    if project is None:
        conn.close()
        raise HTTPException(status_code=404, detail=f"project not found: {project_id}")

    rows = conn.execute(
        "SELECT caption FROM dataset_items WHERE project_id = ? AND selected = 1",
        (project_id,),
    ).fetchall()
    conn.close()

    captions = [r["caption"] or "" for r in rows]
    total = len(captions)
    captioned = [c for c in captions if c.strip()]

    if not captioned:
        return {
            "project_id": project_id,
            "total": total,
            "captioned": 0,
            "risk_level": "unknown",
            "risk_score": 0,
            "leaks": [],
            "character_tags": [],
            "warnings": ["キャプションが存在しません。先に WD14 タグ生成を実行してください。"],
        }

    leaks: list[dict] = []
    warnings: list[str] = []

    # ── 特徴偏り検出 ─────────────────────────────────────────────────────
    for feature_name, feature_group, threshold in LEAK_RISK_COMBOS:
        dominant, ratio = detect_dominant_feature(captioned, feature_group, threshold)
        if dominant:
            severity = "high" if ratio >= 0.85 else "medium"
            leaks.append({
                "feature": feature_name,
                "dominant": dominant,
                "ratio": round(ratio * 100, 1),
                "severity": severity,
            })
            msg = f"[{severity.upper()}] {feature_name}: 「{dominant}」が {ratio * 100:.0f}% の画像に含まれています"
            warnings.append(msg)

    # ── WD14 キャラタグ検出（selected_tags.csv が存在する場合）───────────
    character_tags: list[dict] = []
    tags_csv = CACHE_DIR / TAGS_FILENAME
    if tags_csv.exists():
        import csv as _csv
        char_tag_names: set[str] = set()
        try:
            with open(tags_csv, encoding="utf-8", newline="") as f:
                reader = _csv.DictReader(f)
                for row_data in reader:
                    if int(row_data.get("category", 9)) == 4:
                        char_tag_names.add(row_data["name"].replace("_", " ").lower())
        except Exception:
            pass

        # キャプション内のキャラタグ出現カウント
        char_counts: dict[str, int] = {}
        for caption in captioned:
            tags_in_caption = {t.strip().lower() for t in caption.split(",")}
            for ct in char_tag_names & tags_in_caption:
                char_counts[ct] = char_counts.get(ct, 0) + 1

        # 出現率 >= 30% を警告対象
        for ct, cnt in sorted(char_counts.items(), key=lambda x: x[1], reverse=True)[:20]:
            ratio = cnt / len(captioned)
            if ratio >= 0.3:
                character_tags.append({
                    "tag": ct,
                    "count": cnt,
                    "ratio": round(ratio * 100, 1),
                    "severity": "high" if ratio >= 0.7 else "medium",
                })
        if character_tags:
            warnings.append(
                f"キャラクタータグが混入している可能性: {', '.join(t['tag'] for t in character_tags[:3])}"
            )

    # ── リスクスコア計算 ──────────────────────────────────────────────────
    risk_score = 0
    for leak in leaks:
        risk_score += 3 if leak["severity"] == "high" else 1
    for ct in character_tags:
        risk_score += 4 if ct["severity"] == "high" else 2

    risk_level = "low" if risk_score == 0 else "medium" if risk_score <= 4 else "high"

    # Style LoRA の場合のみ強調警告
    if project["project_type"] == "style" and risk_level != "low":
        warnings.insert(0, f"[Style LoRA] キャラクター固有特徴の学習リスクがあります (risk_score={risk_score})")

    return {
        "project_id": project_id,
        "total": total,
        "captioned": len(captioned),
        "risk_level": risk_level,
        "risk_score": risk_score,
        "leaks": leaks,
        "character_tags": character_tags,
        "warnings": warnings,
    }


# ── §16 Human Review Gate ─────────────────────────────────────────────────────

@router.get("/report/{project_id}")
def get_dataset_report(project_id: int) -> dict:
    """
    §16 Human Review Gate — 学習前チェックレポートを生成する。
    Quality Score・キャプション率・リーク・重複・ブロッカーを一括返却。
    """
    conn = get_conn()
    project = conn.execute(
        "SELECT id, name, project_type FROM projects WHERE id = ?", (project_id,)
    ).fetchone()
    if project is None:
        conn.close()
        raise HTTPException(status_code=404, detail=f"project not found: {project_id}")

    items = conn.execute(
        "SELECT id, width, height, aspect, caption FROM dataset_items WHERE project_id = ? AND selected = 1",
        (project_id,),
    ).fetchall()

    analysis = conn.execute(
        "SELECT quality_score, stats_json, similarity_json, analyzed_at FROM dataset_analysis WHERE project_id = ?",
        (project_id,),
    ).fetchone()
    conn.close()

    total = len(items)
    captions = [str(i["caption"] or "") for i in items]
    captioned = sum(1 for c in captions if c.strip())
    caption_rate = round(captioned / total * 100, 1) if total else 0.0

    warnings: list[str] = []
    blockers: list[str] = []

    # ── ブロッカー判定 ─────────────────────────────────────────────────────
    if total == 0:
        blockers.append("学習データが 0 枚です。先に画像を追加してください。")
    if 0 < total < 10:
        warnings.append(f"データ数が少ない: {total} 枚（推奨: 20 枚以上）")
    if total > 0 and captioned == 0:
        blockers.append("全画像がキャプション未設定です。WD14 タグ生成を先に実行してください。")
    elif total > 0 and caption_rate < 80:
        warnings.append(f"キャプション設定率が低い: {caption_rate}% （推奨: 80% 以上）")

    # ── 分析結果から追加情報 ──────────────────────────────────────────────
    quality_score: int | None = None
    duplicate_pairs = 0
    similar_pairs = 0

    if analysis:
        quality_score = int(analysis["quality_score"])
        try:
            stats = json.loads(analysis["stats_json"])
        except Exception:
            stats = {}

        duplicate_pairs = int(stats.get("duplicate_pairs", 0))
        similar_pairs = int(stats.get("similar_pairs", 0))
        for w in stats.get("warnings", []):
            warnings.append(w)
        if quality_score < 50:
            warnings.append(f"データセット品質スコアが低い: {quality_score}/100")

    # ── 簡易リーク分析 ───────────────────────────────────────────────────
    leak_risk = "unknown"
    leak_score = 0
    captioned_texts = [c for c in captions if c.strip()]
    if captioned_texts:
        for _feature_name, feature_group, threshold in LEAK_RISK_COMBOS:
            dominant, ratio = detect_dominant_feature(captioned_texts, feature_group, threshold)
            if dominant:
                leak_score += 3 if ratio >= 0.85 else 1
        leak_risk = "low" if leak_score == 0 else "medium" if leak_score <= 4 else "high"
        if leak_risk == "high":
            warnings.append(f"キャラクターリーク高リスク（スコア={leak_score}）")
        elif leak_risk == "medium":
            warnings.append(f"キャラクターリーク中リスク（スコア={leak_score}）")

    return {
        "project_id": project_id,
        "project_name": project["name"],
        "project_type": project["project_type"],
        "ready": len(blockers) == 0,
        "quality_score": quality_score,
        "total_images": total,
        "caption_rate": caption_rate,
        "warnings": warnings,
        "blockers": blockers,
        "leak_risk": leak_risk,
        "leak_score": leak_score,
        "duplicate_pairs": duplicate_pairs,
        "similar_pairs": similar_pairs,
        "analyzed": analysis is not None,
    }


# ── §12 Caption Category Statistics ──────────────────────────────────────

@router.get("/tag-categories/{project_id}")
def tag_categories(project_id: int) -> dict:
    """
    §12 Caption Intelligence — カテゴリ別統計
    全キャプションのタグを SPEC §12 カテゴリに分類して集計する。
    """
    conn = get_conn()
    project = conn.execute("SELECT id FROM projects WHERE id = ?", (project_id,)).fetchone()
    if project is None:
        conn.close()
        raise HTTPException(status_code=404, detail=f"project not found: {project_id}")

    rows = conn.execute(
        "SELECT id, caption FROM dataset_items WHERE project_id = ? AND selected = 1",
        (project_id,),
    ).fetchall()
    conn.close()

    total_items = len(rows)
    category_counts: dict[str, int] = {cat: 0 for cat in CATEGORIES}
    category_top_tags: dict[str, dict[str, int]] = {cat: {} for cat in CATEGORIES}

    for row in rows:
        caption = row["caption"] or ""
        if not caption.strip():
            continue
        classified = classify_caption(caption)
        for cat, tags in classified.items():
            if tags:
                category_counts[cat] += 1
            for tag in tags:
                category_top_tags[cat][tag] = category_top_tags[cat].get(tag, 0) + 1

    # 各カテゴリのトップタグ（上位10件）
    category_details: list[dict] = []
    for cat in CATEGORIES:
        top = sorted(category_top_tags[cat].items(), key=lambda x: x[1], reverse=True)[:10]
        category_details.append({
            "category": cat,
            "items_with_tags": category_counts[cat],
            "coverage_pct": round(category_counts[cat] / total_items * 100, 1) if total_items else 0,
            "top_tags": [{"tag": t, "count": c} for t, c in top],
        })

    return {
        "project_id": project_id,
        "total_items": total_items,
        "categories": category_details,
    }


# ── §9.2-9.4 Distribution Analysis ───────────────────────────────────────────

def _count_distribution(captions: list[str], feature_group: list[str]) -> list[dict]:
    """feature_group の各フィーチャーが何枚のキャプションに出現するか集計する。"""
    counts: dict[str, int] = {}
    total_captioned = len(captions)
    for caption in captions:
        caption_lower = caption.lower()
        for feature in feature_group:
            if feature in caption_lower:
                counts[feature] = counts.get(feature, 0) + 1
    # pct 付きでソート（降順）
    result = []
    for feature, cnt in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        result.append({
            "label": feature,
            "count": cnt,
            "pct": round(cnt / total_captioned * 100, 1) if total_captioned else 0.0,
        })
    return result


@router.get("/distribution/{project_id}")
def get_distribution(project_id: int) -> dict:
    """
    §9.2-9.4 Dataset Distribution Analysis
    キャプションから髪色・髪型・瞳色・衣装のタグ分布を集計して返す。
    """
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

    captions = [str(r["caption"] or "").strip() for r in rows]
    captioned = [c for c in captions if c]
    total = len(captions)

    return {
        "project_id": project_id,
        "total_items": total,
        "captioned_items": len(captioned),
        "hair_color": _count_distribution(captioned, HAIR_COLORS),
        "hair_style": _count_distribution(captioned, HAIR_STYLES),
        "eye_color": _count_distribution(captioned, EYE_COLORS),
        "costume": _count_distribution(captioned, COSTUME_TYPES),
    }


# ── 自動提案機能 ──────────────────────────────────────────
_SUGGESTION_STATUS: dict[int, dict] = {}
_CANCEL_FLAGS: dict[int, bool] = {}


@router.post("/suggest-cancel/{project_id}")
async def suggest_cancel(project_id: int):
    """実行中の提案処理をキャンセル"""
    _CANCEL_FLAGS[project_id] = True
    _set_suggestion_status(project_id, "cancelled", "停止しました", step=0)
    return {"status": "cancelled"}


@router.post("/suggest-images/{project_id}")
async def suggest_images(project_id: int, background_tasks: BackgroundTasks, payload: dict = {}):
    """複数ソースから類似画像を自動提案＆LLM評価

    payload.evaluation_mode: "fast" | "balanced" | "accurate"
    payload.manual_query: Booruタグを手動指定（空文字列なら自動抽出）
    """
    evaluation_mode = payload.get("evaluation_mode", "balanced") if payload else "balanced"
    if evaluation_mode not in ("fast", "balanced", "accurate"):
        evaluation_mode = "balanced"
    manual_query = (payload.get("manual_query", "") or "").strip() if payload else ""

    conn = get_conn()
    try:
        project = conn.execute(
            "SELECT id, name FROM projects WHERE id = ?",
            (project_id,)
        ).fetchone()
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        items = conn.execute(
            "SELECT file_path FROM dataset_items WHERE project_id = ?",
            (project_id,)
        ).fetchall()

        if not items:
            raise HTTPException(status_code=400, detail="No images in dataset")

        image_paths = [item[0] for item in items if Path(item[0]).exists()]

        if not image_paths:
            raise HTTPException(status_code=400, detail="No valid image paths")

        mode_label = {"fast": "高速（人気度のみ）", "balanced": "バランス（軽量LLM）", "accurate": "高精度（Qwen2.5VL）"}
        _SUGGESTION_STATUS[project_id] = {
            "status": "running",
            "message": f"検索開始 [{mode_label.get(evaluation_mode, evaluation_mode)}]"
                       + (f" クエリ: {manual_query}" if manual_query else ""),
            "evaluation_mode": evaluation_mode,
        }

        background_tasks.add_task(
            _run_suggestion,
            project_id,
            image_paths,
            evaluation_mode,
            manual_query,
        )

        return {"status": "queued", "message": "提案検索を開始しました", "evaluation_mode": evaluation_mode}

    finally:
        conn.close()


@router.get("/suggest-status/{project_id}")
async def suggest_status(project_id: int):
    """提案検索の進捗確認"""
    status = _SUGGESTION_STATUS.get(project_id, {"status": "idle", "message": "実行待機中"})
    return status


@router.get("/suggestions/{project_id}")
async def get_suggestions(project_id: int):
    """提案結果取得"""
    conn = get_conn()
    try:
        suggestions = conn.execute(
            "SELECT suggestions_json FROM dataset_suggestions WHERE project_id = ? ORDER BY created_at DESC LIMIT 1",
            (project_id,)
        ).fetchone()

        if not suggestions:
            return {"project_id": project_id, "results": [], "message": "提案結果なし"}

        return json.loads(suggestions[0])

    finally:
        conn.close()


def _set_suggestion_status(project_id: int, status: str, message: str, step: int = 0, total_steps: int = 4) -> None:
    _SUGGESTION_STATUS[project_id] = {
        "status": status,
        "message": message,
        "step": step,
        "total_steps": total_steps,
    }


def _run_suggestion(project_id: int, image_paths: list, evaluation_mode: str = "balanced", manual_query: str = "") -> None:
    """バックグラウンド提案処理（複数ソース並列・LLM評価）"""
    import asyncio
    from datetime import datetime

    mode_label = {
        "fast": "高速（人気度のみ）",
        "balanced": "バランス（軽量LLM・3B）",
        "accurate": "高精度（画像解析・32B）"
    }

    async def _run_async() -> dict:
        # Step 1: キャプション / タグ取得
        _set_suggestion_status(project_id, "running", "キャプション読込中...", step=1)
        conn = get_conn()
        captions = conn.execute(
            "SELECT caption FROM dataset_items WHERE project_id = ? AND caption != ''",
            (project_id,)
        ).fetchall()
        conn.close()

        # 全キャプションを結合してより多様なキャラ情報を渡す
        search_query = ""
        if captions:
            first_caption = (captions[0][0] or "").strip()
            if first_caption:
                search_query = first_caption[:100]

        if _CANCEL_FLAGS.get(project_id):
            return {"project_id": project_id, "status": "cancelled", "results": [], "total_count": 0}

        # Step 2: 広域検索 → CLIP視覚類似度でランキング
        _set_suggestion_status(
            project_id, "running",
            f"CLIP視覚類似度で検索中... [{mode_label.get(evaluation_mode)}]",
            step=2
        )

        result = await ImageSearchService.suggest_images(
            project_id=project_id,
            current_image_paths=image_paths,
            tags=search_query,
            limit=30,
            min_width=0,
            evaluation_mode=evaluation_mode,
            manual_query=manual_query,
        )

        if result.get("status") != "completed":
            raise Exception(result.get("error", "Unknown error"))

        # Step 3: 進捗表示
        num_results = len(result.get("results", []))
        _set_suggestion_status(
            project_id, "running",
            f"品質評価中... ({num_results}件)",
            step=3
        )

        # Step 4: 結果保存
        _set_suggestion_status(project_id, "running", "結果を保存中...", step=4)

        return result

    try:
        _CANCEL_FLAGS[project_id] = False
        _set_suggestion_status(project_id, "running", "処理を開始中...", step=0)
        result = asyncio.run(_run_async())

        conn = get_conn()
        conn.execute(
            "INSERT INTO dataset_suggestions (project_id, suggestions_json, created_at) VALUES (?, ?, datetime('now'))",
            (project_id, json.dumps(result))
        )
        conn.commit()
        conn.close()

        num_results = len(result.get("results", []))
        _set_suggestion_status(
            project_id, "done",
            f"{num_results} 件の提案画像を見つけました（複数ソース・LLM評価）",
            step=4,
        )
        logger.info(f"Suggestion completed: {num_results} images, breakdown: {result.get('source_breakdown')}")

    except Exception as e:
        logger.error(f"_run_suggestion failed: {e}", exc_info=True)
        _set_suggestion_status(project_id, "failed", f"提案検索失敗: {str(e)}", step=0)


_FEEDBACK_STATUS: dict[int, dict] = {}


@router.post("/suggest-feedback/{project_id}")
async def suggest_feedback(project_id: int, background_tasks: BackgroundTasks, payload: dict = {}):
    """フィードバック（✅/❌）で候補を再ランク＆追加提案

    payload:
      accepted_urls:   List[str]  — 気に入った画像のURL
      rejected_urls:   List[str]  — 不要な画像のURL
      evaluation_mode: str        — fast / balanced / accurate
    """
    accepted_urls = (payload or {}).get("accepted_urls", [])
    rejected_urls = (payload or {}).get("rejected_urls", [])
    evaluation_mode = (payload or {}).get("evaluation_mode", "balanced")
    if evaluation_mode not in ("fast", "balanced", "accurate"):
        evaluation_mode = "balanced"

    conn = get_conn()
    try:
        project = conn.execute("SELECT id FROM projects WHERE id = ?", (project_id,)).fetchone()
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        # 現在の提案結果を取得
        row = conn.execute(
            "SELECT suggestions_json FROM dataset_suggestions WHERE project_id = ? ORDER BY created_at DESC LIMIT 1",
            (project_id,)
        ).fetchone()
        if not row:
            raise HTTPException(status_code=400, detail="提案結果がありません。先に提案を実行してください。")

        current_suggestions = json.loads(row[0])

        items = conn.execute(
            "SELECT file_path FROM dataset_items WHERE project_id = ?",
            (project_id,)
        ).fetchall()
        image_paths = [item[0] for item in items if Path(item[0]).exists()]

    finally:
        conn.close()

    _FEEDBACK_STATUS[project_id] = {"status": "running", "message": "フィードバック処理中...", "step": 1}

    background_tasks.add_task(
        _run_feedback,
        project_id,
        current_suggestions,
        accepted_urls,
        rejected_urls,
        image_paths,
        evaluation_mode,
    )

    return {"status": "queued", "message": "フィードバックを受け付けました"}


@router.get("/suggest-feedback-status/{project_id}")
async def suggest_feedback_status(project_id: int):
    """フィードバック再検索の進捗"""
    return _FEEDBACK_STATUS.get(project_id, {"status": "idle", "message": "待機中"})


def _run_feedback(
    project_id: int,
    current_suggestions: dict,
    accepted_urls: list,
    rejected_urls: list,
    image_paths: list,
    evaluation_mode: str,
) -> None:
    """バックグラウンド: CLIP再ランク + accepted 類似語で追加検索"""
    import asyncio
    from datetime import datetime

    async def _async() -> dict:
        # Step 1: 既存候補を CLIP でリランク
        _FEEDBACK_STATUS[project_id] = {"status": "running", "message": "CLIPで類似度を計算中...", "step": 1}
        current_candidates = current_suggestions.get("results", [])

        reranked = await ImageSearchService.rerank_by_feedback(
            accepted_urls=accepted_urls,
            rejected_urls=rejected_urls,
            candidates=current_candidates,
            evaluation_mode=evaluation_mode,
        )

        # Step 2: accepted があれば追加でも検索
        extra: list[dict] = []
        if accepted_urls:
            _FEEDBACK_STATUS[project_id] = {"status": "running", "message": "accepted画像で追加検索中...", "step": 2}

            # accepted のタイトル/タグをキーワードに使う
            accepted_set = set(accepted_urls)
            accepted_items = [c for c in current_candidates if c.get('url') in accepted_set]
            titles = [c.get('title', '') for c in accepted_items if c.get('title')]
            extra_query = " ".join(titles[:3]) if titles else "anime girl character illustration"

            # 新しい候補を並列取得
            extra = await ImageSearchService.suggest_images(
                project_id=project_id,
                current_image_paths=image_paths,
                tags=extra_query,
                limit=20,
                min_width=512,
                evaluation_mode=evaluation_mode,
            )
            extra = extra.get("results", [])

        # Step 3: マージ（accepted は最優先、extra を追加、重複排除）
        _FEEDBACK_STATUS[project_id] = {"status": "running", "message": "結果をまとめています...", "step": 3}

        seen_urls = {r.get('url') for r in reranked}
        for e in extra:
            if e.get('url') not in seen_urls:
                reranked.append(e)
                seen_urls.add(e.get('url'))

        # accepted に +20 ボーナス（選んだ画像を上位固定）
        accepted_set_2 = set(accepted_urls)
        for r in reranked:
            if r.get('url') in accepted_set_2:
                r['score'] = round(r.get('score', 50) + 20, 1)
                r['user_selected'] = True

        reranked.sort(key=lambda x: x.get('score', 0), reverse=True)

        source_breakdown: dict[str, int] = {}
        for source in ['bing', 'pixiv', 'duckduckgo', 'google', 'pinterest']:
            source_breakdown[source] = len([r for r in reranked if r.get("source") == source])

        return {
            "project_id": project_id,
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "results": reranked[:40],  # フィードバック後は最大40件まで表示
            "total_count": len(reranked),
            "source_breakdown": source_breakdown,
            "feedback_round": current_suggestions.get("feedback_round", 0) + 1,
        }

    try:
        result = asyncio.run(_async())

        conn = get_conn()
        conn.execute(
            "INSERT INTO dataset_suggestions (project_id, suggestions_json, created_at) VALUES (?, ?, datetime('now'))",
            (project_id, json.dumps(result))
        )
        conn.commit()
        conn.close()

        _FEEDBACK_STATUS[project_id] = {
            "status": "done",
            "message": f"{len(result.get('results', []))} 件に更新されました（第 {result.get('feedback_round', 1)} ラウンド）",
            "step": 4,
            "result": result,
        }
    except Exception as e:
        logger.error(f"_run_feedback failed: {e}", exc_info=True)
        _FEEDBACK_STATUS[project_id] = {"status": "failed", "message": f"失敗: {str(e)}", "step": 0}
