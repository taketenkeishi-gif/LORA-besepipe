from __future__ import annotations

import io
import hashlib
import json
import logging
import shutil
import tempfile
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
# 256bit phash で細部まで識別（64bitより誤検出が大幅に減る）。
_HASH_SIZE = 16
# 距離しきい値（256bit中のハミング距離）
_DUP_THRESHOLD = 6       # ほぼ同一: 再保存/リサイズ/軽微編集/ミラー複製
_SIMILAR_THRESHOLD = 22  # 同一構図・トリミング違いなど


def _compute_signature(path: str):
    """phash(256bit) と左右反転版のタプルを返す。
    反転版も持つことでミラー(左右反転)複製を検出できる。
    RGB へ変換して alpha 由来のハッシュ乱れを防ぐ。"""
    try:
        import imagehash
        with Image.open(path) as im:
            rgb = im.convert("RGB")
            ph = imagehash.phash(rgb, hash_size=_HASH_SIZE)
            ph_flip = imagehash.phash(rgb.transpose(Image.FLIP_LEFT_RIGHT), hash_size=_HASH_SIZE)
        return (ph, ph_flip)
    except Exception:
        return None


def _sig_distance(a, b) -> int:
    """2署名間の最小ハミング距離（左右反転を考慮）。"""
    return min(a[0] - b[0], a[0] - b[1], a[1] - b[0])


# 後方互換: 単体phashが必要な箇所向け
def _compute_phash(path: str):
    sig = _compute_signature(path)
    return sig[0] if sig else None


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
    try:
        import imagehash  # noqa: F401
        _imagehash_available = True
    except ImportError:
        _imagehash_available = False

    similarity_groups: list[dict] = []
    duplicate_pairs = 0
    similar_pairs = 0

    if _imagehash_available:
        sigs: list[tuple[int, object]] = []
        for item in items:
            s = _compute_signature(item["file_path"])
            if s is not None:
                sigs.append((item["id"], s))

        n = len(sigs)
        # Union-Find で連結成分（=類似グループ）を作る。
        # 旧実装は similar メンバーを processed に入れずグループが重複・水増ししていた。
        parent = list(range(n))

        def _find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def _union(a: int, b: int) -> None:
            ra, rb = _find(a), _find(b)
            if ra != rb:
                parent[ra] = rb

        edges: list[tuple[int, int, int]] = []
        # 全ペア比較（O(n²)・実用上は数百〜数千枚）
        for i in range(n):
            for j in range(i + 1, n):
                d = _sig_distance(sigs[i][1], sigs[j][1])
                if d <= _SIMILAR_THRESHOLD:
                    edges.append((i, j, d))
                    _union(i, j)

        # グループ集約
        members_map: dict[int, list[int]] = {}
        for idx in range(n):
            members_map.setdefault(_find(idx), []).append(idx)
        # グループ内の最小距離（種別判定用）
        group_min: dict[int, int] = {}
        for (i, j, d) in edges:
            r = _find(i)
            if d < group_min.get(r, 10**9):
                group_min[r] = d

        for root, members in members_map.items():
            if len(members) < 2:
                continue
            item_ids = [sigs[m][0] for m in members]
            # A perceptual-similarity edge must never authorize deletion of its
            # entire connected component. Only byte-identical files are duplicates.
            files_by_id = {item['id']: item['file_path'] for item in items}
            exact: dict[str, list[int]] = {}
            for item_id in item_ids:
                try:
                    with Path(files_by_id[item_id]).open('rb') as handle:
                        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
                except OSError:
                    digest = f'unreadable:{item_id}'
                exact.setdefault(digest, []).append(item_id)
            for same_files in exact.values():
                if len(same_files) > 1:
                    similarity_groups.append({'type': 'duplicate', 'item_ids': same_files})
                    duplicate_pairs += len(same_files) - 1
            if len(exact) > 1:
                similarity_groups.append({"type": "similar", "item_ids": item_ids})
                similar_pairs += len(exact) - 1

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


# ── 前処理共通定数 ────────────────────────────────────────────────────

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}


@router.post("/delete-items/{project_id}")
def delete_dataset_items(project_id: int, item_ids: str, delete_files: bool = True) -> dict:
    """データセットから画像を削除（重複・不要画像の一括/個別削除）。

    delete_files=True のとき画像ファイルと .txt サイドカーを _trash フォルダへ退避する
    （完全削除ではなくゴミ箱退避＝復元可能）。DB レコードは常に削除する。
    """
    ids = []
    for x in item_ids.split(","):
        x = x.strip()
        if x:
            try:
                ids.append(int(x))
            except ValueError:
                raise HTTPException(status_code=400, detail="item_ids が不正です")
    if not ids:
        raise HTTPException(status_code=400, detail="item_ids が空です")

    conn = get_conn()
    prow = conn.execute("SELECT dataset_dir FROM projects WHERE id = ?", (project_id,)).fetchone()
    if prow is None:
        conn.close()
        raise HTTPException(status_code=404, detail="project not found")

    placeholders = ",".join("?" * len(ids))
    rows = conn.execute(
        f"SELECT id, file_path FROM dataset_items WHERE project_id = ? AND id IN ({placeholders})",
        (project_id, *ids),
    ).fetchall()

    trash_dir = None
    trashed = 0
    if delete_files and rows:
        dataset_dir = prow["dataset_dir"] or ""
        base = Path(dataset_dir).parent if dataset_dir else Path(rows[0]["file_path"]).parent
        trash_dir = base / "_trash"
        trash_dir.mkdir(parents=True, exist_ok=True)

    removed = 0
    for r in rows:
        fp = Path(r["file_path"])
        if delete_files and trash_dir is not None and fp.exists():
            try:
                dest = trash_dir / fp.name
                i = 1
                while dest.exists():
                    dest = trash_dir / f"{fp.stem}_{i}{fp.suffix}"
                    i += 1
                shutil.move(str(fp), str(dest))
                trashed += 1
                # キャプション .txt サイドカーも退避
                txt = fp.with_suffix(".txt")
                if txt.exists():
                    shutil.move(str(txt), str(trash_dir / txt.name))
            except Exception as exc:  # noqa: BLE001
                logger.warning("trash move failed for %s: %s", fp, exc)
        conn.execute("DELETE FROM dataset_items WHERE id = ?", (r["id"],))
        removed += 1

    conn.commit()
    conn.close()
    return {
        "removed_count": removed,
        "trashed_files": trashed,
        "trash_dir": str(trash_dir) if trash_dir else None,
    }


# ── 文字削除 / Dataset Refinery（SPEC §14・ComfyUI エンジン） ──────────────────

_REFINERY_STATUS: dict[int, dict] = {}


@router.get("/comfyui/status")
def comfyui_status() -> dict:
    """ComfyUI の到達性・必要ノード・モデルの有無を返す（doctor）。

    プロセス自動起動の状態（launching/ready 等）も併せて返す。
    """
    from ..services.comfyui_client import ComfyUIClient
    from ..services.comfyui_process import get_state

    try:
        result = ComfyUIClient().check_status()
    except Exception as e:  # noqa: BLE001
        result = {"reachable": False, "ready": False, "message": f"状態取得失敗: {e}"}
    result["process"] = get_state()
    return result


@router.post("/comfyui/ensure")
def comfyui_ensure() -> dict:
    """ComfyUI が未起動なら起動する（手動トリガ）。起動済みなら何もしない。"""
    from ..services.comfyui_process import ensure_running

    return ensure_running(block_wait=False)


def _resolve_target_files(project_id: int, dataset_dir: str, item_ids: list[int] | None) -> list[Path]:
    """対象画像パスを解決。item_ids 指定があればその dataset_items、無ければ dataset_dir 全画像。"""
    if item_ids:
        conn = get_conn()
        rows = conn.execute(
            f"SELECT file_path FROM dataset_items WHERE project_id = ? AND id IN ({','.join('?' * len(item_ids))})",
            (project_id, *item_ids),
        ).fetchall()
        conn.close()
        return [Path(r["file_path"]) for r in rows if Path(r["file_path"]).exists()]
    base = Path(dataset_dir)
    if not base.exists():
        return []
    return [f for f in base.rglob("*") if f.suffix.lower() in IMAGE_EXTS]


def _run_text_removal(project_id: int, dataset_dir: str, item_ids: list[int] | None, grow: int) -> None:
    """ComfyUI(Florence2 OCR + LaMa)で文字削除。元画像はバックアップ後に上書き。"""
    from ..services.comfyui_client import ComfyUIClient, ComfyUIError

    _REFINERY_STATUS[project_id] = {"status": "running", "message": "ComfyUI 確認中...", "done": 0, "total": 0}
    try:
        client = ComfyUIClient()
        status = client.check_status()
        if not status.get("ready"):
            _REFINERY_STATUS[project_id] = {
                "status": "failed",
                "message": status.get("message", "ComfyUI が準備できていません"),
                "done": 0,
                "total": 0,
            }
            return

        lama_model = status["lama_model"]

        files = _resolve_target_files(project_id, dataset_dir, item_ids)
        total = len(files)
        if total == 0:
            _REFINERY_STATUS[project_id] = {"status": "failed", "message": "対象画像がありません", "done": 0, "total": 0}
            return

        # 元画像バックアップ先（dataset_dir 外。リサイズ等の再スキャン対象に含めない）
        backup_dir = Path(dataset_dir).parent / "_refinery_backup" / "text_removal"
        backup_dir.mkdir(parents=True, exist_ok=True)

        _REFINERY_STATUS[project_id] = {"status": "running", "message": f"0/{total} 件処理中...", "done": 0, "total": total}

        done = 0
        failed = 0
        for f in files:
            try:
                out_bytes = client.remove_text(
                    f, lama_model=lama_model, grow=grow, timeout=300.0
                )
                # バックアップ（既存があれば上書きしない＝最初の原本を保持）
                bak = backup_dir / f.name
                if not bak.exists():
                    shutil.copy2(f, bak)
                # ComfyUI 出力は PNG。元拡張子に合わせて保存
                from PIL import Image as _Image
                img = _Image.open(io.BytesIO(out_bytes))
                ext = f.suffix.lower()
                if ext in {".jpg", ".jpeg"}:
                    img.convert("RGB").save(f, quality=95, optimize=True)
                else:
                    img.save(f)
            except (ComfyUIError, Exception) as exc:  # noqa: BLE001
                logger.warning("text removal failed for %s: %s", f, exc)
                failed += 1
            done += 1
            _REFINERY_STATUS[project_id] = {
                "status": "running",
                "message": f"{done}/{total} 件処理中...",
                "done": done,
                "total": total,
            }

        ok = done - failed
        _REFINERY_STATUS[project_id] = {
            "status": "done",
            "message": f"{ok} 件 文字削除完了" + (f"（{failed} 件失敗）" if failed else "") + f" / 原本: {backup_dir}",
            "done": done,
            "total": total,
        }
    except Exception as e:  # noqa: BLE001
        logger.error("_run_text_removal failed: %s", e, exc_info=True)
        _REFINERY_STATUS[project_id] = {"status": "failed", "message": f"文字削除エラー: {e}", "done": 0, "total": 0}


@router.post("/text-removal/{project_id}")
async def text_removal(
    project_id: int,
    background_tasks: BackgroundTasks,
    grow: int = 8,
    item_ids: str = "",
) -> dict:
    """文字削除を実行（ComfyUI Florence2 OCR + LaMa inpaint）。

    item_ids: カンマ区切りの dataset_items.id。空なら dataset_dir 全画像が対象。
    """
    conn = get_conn()
    row = conn.execute("SELECT dataset_dir FROM projects WHERE id = ?", (project_id,)).fetchone()
    conn.close()
    if row is None:
        raise HTTPException(status_code=404, detail="project not found")
    dataset_dir = row["dataset_dir"] or ""
    if not dataset_dir:
        raise HTTPException(status_code=400, detail="dataset_dir が設定されていません")

    if not (-64 <= grow <= 256):
        raise HTTPException(status_code=400, detail="grow は -64〜256 の範囲で指定してください")

    ids: list[int] | None = None
    if item_ids.strip():
        try:
            ids = [int(x) for x in item_ids.split(",") if x.strip()]
        except ValueError:
            raise HTTPException(status_code=400, detail="item_ids が不正です")

    status = _REFINERY_STATUS.get(project_id, {})
    if status.get("status") == "running":
        return {"message": "文字削除を実行中です", "status": "running"}

    background_tasks.add_task(_run_text_removal, project_id, dataset_dir, ids, grow)
    return {"message": "文字削除を開始しました", "status": "started"}


@router.get("/text-removal-status/{project_id}")
def text_removal_status(project_id: int) -> dict:
    """文字削除の進捗を返す。"""
    return _REFINERY_STATUS.get(project_id, {"status": "idle", "message": "待機中", "done": 0, "total": 0})


# ── 統合前処理パイプライン: Resize → Upscale → Cleanup → Caption → Save ─────
# 実装本体は app.dataset.pipeline（PipelineStep 方式）に移動済み。
# ここでは API バリデーションとキュー投入・設定保存/復元のみを担う。
# 複数プロジェクトからの同時要求は enqueue() が直列化するため、ここでは
# BackgroundTasks を使わない（queue.py のワーカースレッドが逐次実行する）。
import json as _json  # noqa: E402

from ..dataset.pipeline import (  # noqa: E402
    cancel_queued as _pipeline_cancel_queued,
    enqueue as _pipeline_enqueue,
    get_pipeline_status as _get_pipeline_status,
    is_queued as _pipeline_is_queued,
    is_running as _pipeline_is_running,
    read_manifest as _read_pipeline_manifest,
    request_cancel as _pipeline_request_cancel,
)


def _resolve_recommended_train_dir(conn, project_id: int) -> dict | None:
    """dataset_items(DB, キャプション付き登録済み画像の唯一の正)から、実際に
    学習で使うべきtrain_data_dirを逆算する。

    project.dataset_dir は静的な既定パスに過ぎず、Dataset Builder Pipelineや
    kohya形式のrepeat-countフォルダ(例: 5_ProjectName)への画像整理後は、
    実画像がdataset_dir配下ではなくexternal_dataset(library_dir)側の
    別フォルダに存在することがある。Training画面がdataset_dirを無条件に
    初期値としてしまうと、実際には0枚(または古い少数枚)しかスキャンされない
    フォルダで学習を開始してしまう実UXバグがあった(実データ監査で発見)。
    dataset_itemsに登録されたfile_pathのうち実際にディスク上に存在するものの
    共通親ディレクトリを求め、そこを「実際にキャプション付きで使える画像が
    ある場所」として推奨する。
    """
    rows = conn.execute(
        "SELECT file_path FROM dataset_items WHERE project_id = ?", (project_id,)
    ).fetchall()
    existing = [Path(r["file_path"]) for r in rows if Path(r["file_path"]).exists()]
    if not existing:
        return None
    parents: dict[str, int] = {}
    for p in existing:
        d = str(p.parent)
        parents[d] = parents.get(d, 0) + 1
    best_dir, count = max(parents.items(), key=lambda kv: kv[1])
    if count < len(existing):
        # 画像が複数フォルダに分散している場合は自動選択せず、判断材料だけ返す
        return {"path": best_dir, "image_count": count, "total_registered": len(existing), "split_across_dirs": True}
    return {"path": best_dir, "image_count": count, "total_registered": len(existing), "split_across_dirs": False}


@router.get("/dataset-sources/{project_id}")
def get_dataset_sources(project_id: int) -> dict:
    """Training が選択できる dataset_source ("original"/"processed") の利用可否を返す。

    "processed" は processed/ フォルダの存在だけでなく、Dataset Builder Pipeline の
    完了マニフェスト(.manifest.json, 成功件数>0)がある場合のみ available=true とする。
    """
    conn = get_conn()
    row = conn.execute("SELECT dataset_dir FROM projects WHERE id = ?", (project_id,)).fetchone()
    if row is None:
        conn.close()
        raise HTTPException(status_code=404, detail="project not found")

    processed_dir = Path(row["dataset_dir"] or "") / "processed"
    manifest = _read_pipeline_manifest(processed_dir) if processed_dir.exists() else None
    recommended = _resolve_recommended_train_dir(conn, project_id)
    conn.close()

    return {
        "project_id": project_id,
        "original": {"available": True},
        "processed": {
            "available": manifest is not None,
            "manifest": manifest,
        },
        "recommended_train_dir": recommended,
    }

_PIPELINE_CONFIG_DEFAULTS: dict = {
    "target_size": 1024,
    "resize_mode": "resize_longer",
    "use_esrgan": False,
    "esrgan_model": "",
    "use_qwen": False,
    "use_caption": False,
    "caption_general_thresh": 0.35,
    "caption_character_thresh": 0.85,
    "caption_remove_character_tags": False,
}


@router.post("/preprocess-pipeline/{project_id}")
async def preprocess_pipeline(
    project_id: int,
    target_size: int = 1024,
    resize_mode: str = "resize_longer",
    use_esrgan: bool = False,
    esrgan_model: str = "",
    use_qwen: bool = False,
    use_caption: bool = False,
    caption_general_thresh: float = 0.35,
    caption_character_thresh: float = 0.85,
    caption_remove_character_tags: bool = False,
    item_ids: str = "",
) -> dict:
    """統合前処理パイプラインをキューへ投入する。

    処理順: Resize → ESRGAN(任意) → Qwen Cleanup(任意) → Caption(任意)
    → dataset/processed/ に保存。元画像は変更しない（Caption有効時は
    既存タグ編集UI互換のため dataset_items.caption と元画像の .txt のみ更新）。

    複数プロジェクトから同時に呼ばれてもキューにより逐次実行される
    （同時実行はしない）。
    """
    conn = get_conn()
    row = conn.execute("SELECT id FROM projects WHERE id = ?", (project_id,)).fetchone()
    conn.close()
    if row is None:
        raise HTTPException(status_code=404, detail="project not found")

    if resize_mode not in {"resize_longer", "resize_shorter", "square_crop"}:
        raise HTTPException(status_code=400, detail=f"不正な resize_mode: {resize_mode}")

    if not (64 <= target_size <= 4096):
        raise HTTPException(status_code=400, detail="target_size は 64〜4096 の範囲で指定してください")

    ids: list[int] | None = None
    if item_ids.strip():
        try:
            ids = [int(x) for x in item_ids.split(",") if x.strip()]
        except ValueError:
            raise HTTPException(status_code=400, detail="item_ids が不正です")

    if _pipeline_is_running(project_id) or _pipeline_is_queued(project_id):
        st = _get_pipeline_status(project_id)
        return {"message": "パイプラインは既に実行中/キュー待機中です", "status": st.get("status", "running")}

    position = _pipeline_enqueue(
        project_id, target_size, resize_mode, use_esrgan, esrgan_model, use_qwen, ids,
        use_caption=use_caption,
        caption_general_thresh=caption_general_thresh,
        caption_character_thresh=caption_character_thresh,
        caption_remove_character_tags=caption_remove_character_tags,
    )
    return {
        "message": f"前処理パイプラインをキューに追加しました（{position} 番目）",
        "status": "queued",
        "queue_position": position,
    }


@router.get("/preprocess-pipeline-status/{project_id}")
def preprocess_pipeline_status(project_id: int) -> dict:
    """統合前処理パイプラインの進捗を返す（items に画像ごとの成功/失敗/スキップ/理由を含む）。"""
    return _get_pipeline_status(project_id)


@router.post("/preprocess-pipeline-cancel/{project_id}")
def preprocess_pipeline_cancel(project_id: int) -> dict:
    """実行中/キュー待機中の統合前処理パイプラインにキャンセルを要求する。"""
    if _pipeline_cancel_queued(project_id):
        return {"message": "キュー内のジョブをキャンセルしました", "status": "cancelling"}
    if not _pipeline_request_cancel(project_id):
        return {"message": "実行中のパイプラインがありません", "status": "idle"}
    return {"message": "キャンセルを要求しました", "status": "cancelling"}


@router.get("/caption-lineage/{project_id}/{item_id}")
def caption_lineage(project_id: int, item_id: int) -> dict:
    """Return the immutable caption lineage without changing dataset files."""
    conn = get_conn()
    row = conn.execute(
        "SELECT d.id, d.file_path, d.caption, d.caption_source, p.dataset_dir, p.captions_dir "
        "FROM dataset_items d JOIN projects p ON p.id = d.project_id WHERE d.project_id = ? AND d.id = ?",
        (project_id, item_id),
    ).fetchone()
    conn.close()
    if row is None:
        raise HTTPException(status_code=404, detail="dataset item not found")

    original_path = Path(str(row["file_path"]))
    original_txt = original_path.with_suffix(".txt")
    captions_dir = Path(str(row["captions_dir"] or ""))
    fallback_txt = captions_dir / f"{original_path.stem}.txt" if str(row["captions_dir"] or "") else None
    processed_txt = Path(str(row["dataset_dir"] or "")) / "processed" / f"{original_path.stem}.txt"

    def read_text(path: Path | None) -> dict:
        if path is None or not path.exists() or not path.is_file():
            return {"path": str(path) if path else "", "exists": False, "text": ""}
        try:
            return {"path": str(path), "exists": True, "text": path.read_text(encoding="utf-8")}
        except OSError as exc:
            return {"path": str(path), "exists": True, "text": "", "error": str(exc)}

    source_file = original_txt if original_txt.exists() else fallback_txt
    return {
        "project_id": project_id,
        "item_id": item_id,
        "image_path": str(original_path),
        "requested": {"caption_source": row["caption_source"] or "", "source_file": str(source_file) if source_file else ""},
        "resolved": {"db_caption": row["caption"] or "", "db_caption_source": row["caption_source"] or ""},
        "observed": {"original_caption_file": read_text(source_file), "processed_caption_file": read_text(processed_txt)},
    }


@router.get("/pipeline-config/{project_id}")
def get_pipeline_config(project_id: int) -> dict:
    """Dataset Builder の前処理パイプライン設定（Resize/Upscale/Cleanup/Caption）を返す。

    未保存の場合は既定値を返す。既存 Project 構造（projects.pipeline_config_json）を
    利用するため、新規テーブルは追加しない。
    """
    conn = get_conn()
    row = conn.execute("SELECT pipeline_config_json FROM projects WHERE id = ?", (project_id,)).fetchone()
    conn.close()
    if row is None:
        raise HTTPException(status_code=404, detail="project not found")

    config = dict(_PIPELINE_CONFIG_DEFAULTS)
    raw = row["pipeline_config_json"] or "{}"
    try:
        saved = _json.loads(raw)
        if isinstance(saved, dict):
            config.update(saved)
    except (ValueError, TypeError):
        pass
    return {"project_id": project_id, "config": config}


@router.post("/pipeline-config/{project_id}")
def save_pipeline_config(project_id: int, payload: dict) -> dict:
    """Dataset Builder の前処理パイプライン設定を Project へ保存する。"""
    conn = get_conn()
    row = conn.execute("SELECT id FROM projects WHERE id = ?", (project_id,)).fetchone()
    if row is None:
        conn.close()
        raise HTTPException(status_code=404, detail="project not found")

    config = dict(_PIPELINE_CONFIG_DEFAULTS)
    if isinstance(payload, dict):
        config.update({k: v for k, v in payload.items() if k in _PIPELINE_CONFIG_DEFAULTS})

    conn.execute(
        "UPDATE projects SET pipeline_config_json = ? WHERE id = ?",
        (_json.dumps(config, ensure_ascii=False), project_id),
    )
    conn.commit()
    conn.close()
    return {"project_id": project_id, "config": config}


# ── Dataset Mixer (§10) ───────────────────────────────────────────────────────

_DEFAULT_WEIGHTS: dict[str, int] = {
    "face": 50,
    "expression": 50,
    "hair": 50,
    "costume": 50,
    "accessory": 50,
    "background": 50,
    "composition": 50,
    "line": 50,
    "color": 50,
    "lighting": 50,
    "mood": 50,
}


@router.get("/mixer/{project_id}")
def get_mixer(project_id: int) -> dict:
    """プロジェクトのフィーチャーウェイトを返す"""
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT feature_weights_json, updated_at FROM dataset_mixer WHERE project_id = ?",
            (project_id,),
        ).fetchone()
        if row is None:
            return {"project_id": project_id, "feature_weights": _DEFAULT_WEIGHTS.copy(), "updated_at": None}
        weights = {**_DEFAULT_WEIGHTS, **json.loads(row["feature_weights_json"])}
        return {"project_id": project_id, "feature_weights": weights, "updated_at": row["updated_at"]}
    finally:
        conn.close()


@router.put("/mixer/{project_id}")
def put_mixer(project_id: int, body: dict) -> dict:
    """フィーチャーウェイトを保存する"""
    weights = body.get("feature_weights", {})
    # clamp values to 0-100
    clamped = {k: max(0, min(100, int(v))) for k, v in weights.items() if k in _DEFAULT_WEIGHTS}
    conn = get_conn()
    try:
        conn.execute(
            """
            INSERT INTO dataset_mixer (project_id, feature_weights_json, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(project_id) DO UPDATE SET
                feature_weights_json = excluded.feature_weights_json,
                updated_at = excluded.updated_at
            """,
            (project_id, json.dumps(clamped)),
        )
        conn.commit()
        return {"project_id": project_id, "feature_weights": {**_DEFAULT_WEIGHTS, **clamped}, "updated_at": None}
    finally:
        conn.close()
