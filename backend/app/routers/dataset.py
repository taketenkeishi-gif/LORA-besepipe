from __future__ import annotations

import json
import threading
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException
from PIL import Image, UnidentifiedImageError

from ..db import get_conn
from ..services.tag_categories import (
    CATEGORIES,
    LEAK_RISK_COMBOS,
    classify_caption,
    detect_dominant_feature,
)
from ..services.tagger import CACHE_DIR, TAGS_FILENAME

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
