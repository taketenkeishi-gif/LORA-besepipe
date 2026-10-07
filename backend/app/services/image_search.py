import asyncio
import json
import logging
from typing import Optional
from pathlib import Path
from datetime import datetime
import requests
from PIL import Image
from io import BytesIO
from sentence_transformers import util
import torch
import hashlib
import urllib.parse

from .image_quality import ImageQualityEvaluator

logger = logging.getLogger(__name__)

MODEL = None
_MODEL_LOAD_ATTEMPTED = False


def _get_model():
    """Load the optional CLIP model only when image similarity is requested.

    Importing the API must never start a hundreds-of-megabytes model download.
    """
    global MODEL, _MODEL_LOAD_ATTEMPTED
    if MODEL is not None or _MODEL_LOAD_ATTEMPTED:
        return MODEL
    _MODEL_LOAD_ATTEMPTED = True
    try:
        from sentence_transformers import SentenceTransformer
        MODEL = SentenceTransformer("clip-ViT-B-32")
    except Exception as exc:
        logger.warning(f"CLIP model load failed: {exc}")
    return MODEL


class ImageSearchService:
    """複数ソースからのデータセット自動提案サービス"""

    @staticmethod
    def extract_features(image_path: str) -> Optional[list]:
        """CLIP で画像特徴抽出"""
        model = _get_model()
        if not model:
            return None
        try:
            img = Image.open(image_path).convert('RGB')
            img_emb = model.encode(img, convert_to_tensor=True)
            return img_emb.cpu().numpy().tolist()
        except Exception as e:
            logger.error(f"Feature extraction failed for {image_path}: {e}")
            return None

    @staticmethod
    def extract_batch_features(image_paths: list) -> Optional[list]:
        """複数画像の特徴抽出＆平均"""
        if not _get_model() or not image_paths:
            return None
        try:
            features = []
            for path in image_paths:
                feat = ImageSearchService.extract_features(path)
                if feat:
                    features.append(feat)
            if not features:
                return None
            import numpy as np
            avg_feature = np.mean(features, axis=0).tolist()
            return avg_feature
        except Exception as e:
            logger.error(f"Batch feature extraction failed: {e}")
            return None

    @staticmethod
    def _generate_image_id(url: str) -> str:
        """URL からユニークな画像 ID を生成"""
        return hashlib.md5(url.encode()).hexdigest()[:12]

    @staticmethod
    def _extract_broad_query(caption: str) -> str:
        """キャプションから発見用の広域Booruクエリを生成（最大1タグ）

        "hatsune miku, blue hair, twintails" → "hatsune_miku"
        "painterly style, warm colors"       → "1girl"  (フォールバック)

        タグ数を1つに絞ることで候補プールを最大化し、
        絞り込みはCLIP視覚類似度に任せる。
        """
        SKIP = {
            "anime", "illustration", "artwork", "character", "art", "drawing",
            "image", "picture", "style", "aesthetic", "1girl", "1boy",
            "girl", "boy", "anime girl", "anime boy",
        }
        if not caption or not caption.strip():
            return "1girl"
        for t in caption.split(","):
            t = t.strip()
            if not t:
                continue
            lower = t.lower()
            if lower in SKIP:
                continue
            converted = t.replace(" ", "_").lower()
            if converted:
                return converted
        return "1girl"

    @staticmethod
    def _build_reference_embedding(image_paths: list):
        """既存データセット画像からCLIP参照embeddingを構築（同期、to_thread用）"""
        model = _get_model()
        if not model or not image_paths:
            return None
        import numpy as np
        embeddings = []
        for path in image_paths[:15]:
            try:
                img = Image.open(path).convert('RGB')
                emb = model.encode(img, convert_to_tensor=False)
                embeddings.append(emb)
            except Exception as e:
                logger.debug(f"Reference encode failed {path}: {e}")
        if not embeddings:
            return None
        avg = np.mean(embeddings, axis=0)
        logger.info(f"Reference embedding built from {len(embeddings)} dataset images")
        return avg

    @staticmethod
    async def _clip_score_candidates(candidates: list, reference_embedding) -> list:
        """全候補をCLIP視覚類似度でスコアリング

        1. プレビュー画像を並列ダウンロード（semaphore=20）
        2. MODEL.encode をバッチ呼び出し（to_thread でイベントループをブロックしない）
        3. コサイン類似度でスコア算出
        """
        import numpy as np

        semaphore = asyncio.Semaphore(20)

        async def download_one(c: dict):
            url = c.get('preview_url') or c.get('url', '')
            if not url:
                return c, None
            async with semaphore:
                try:
                    resp = await asyncio.to_thread(
                        requests.get, url, timeout=4, headers={'User-Agent': 'Mozilla/5.0'}
                    )
                    if resp.status_code == 200:
                        img = Image.open(BytesIO(resp.content)).convert('RGB')
                        return c, img
                except Exception as e:
                    logger.debug(f"Preview download failed {url}: {e}")
            return c, None

        # Step 1: 並列ダウンロード
        dl_results = list(await asyncio.gather(*[download_one(c) for c in candidates]))

        # Step 2: バッチ CLIP エンコード（to_thread でブロック回避）
        with_img = [(c, img) for c, img in dl_results if img is not None]
        without_img = [c for c, img in dl_results if img is None]

        if with_img and MODEL is not None:
            imgs = [img for _, img in with_img]
            embeddings = await asyncio.to_thread(
                MODEL.encode, imgs,
                convert_to_tensor=False, show_progress_bar=False
            )
            for (c, _), emb in zip(with_img, embeddings):
                sim = float(np.dot(reference_embedding, emb) / (
                    np.linalg.norm(reference_embedding) * np.linalg.norm(emb) + 1e-8
                ))
                c['clip_similarity'] = round(sim * 100, 1)

        for c in without_img:
            c['clip_similarity'] = 0.0

        logger.info(f"CLIP scored {len(with_img)} images ({len(without_img)} failed)")
        return candidates

    @staticmethod
    async def search_safebooru(booru_tags: str, limit: int = 50) -> list:
        """Safebooru API — アニメイラスト特化（booru_tags は既変換済みタグ文字列）"""
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = await asyncio.to_thread(
                requests.get,
                "https://safebooru.org/index.php",
                params={
                    "page": "dapi", "s": "post", "q": "index",
                    "json": "1", "tags": booru_tags, "limit": str(limit),
                },
                headers=headers,
                timeout=12,
            )
            if response.status_code != 200:
                logger.warning(f"Safebooru returned {response.status_code}")
                return []

            posts = response.json()
            if not isinstance(posts, list):
                return []

            results = []
            for p in posts:
                file_url = p.get("file_url", "")
                if not file_url or not file_url.startswith("http"):
                    file_url = f"https://safebooru.org/images/{p.get('directory','')}/{p.get('image','')}"
                preview_url = p.get("preview_url", "")
                if preview_url and not preview_url.startswith("http"):
                    preview_url = f"https://safebooru.org{preview_url}"
                results.append({
                    "source": "safebooru",
                    "id": str(p.get("id", ImageSearchService._generate_image_id(file_url))),
                    "title": p.get("tags", "")[:60],
                    "url": file_url,
                    "preview_url": preview_url or file_url,
                    "user": p.get("owner", "Safebooru"),
                    "width": p.get("width", 0),
                    "height": p.get("height", 0),
                    "like_count": p.get("score", 0),
                    "view_count": 0,
                    "tags_raw": p.get("tags", ""),
                })

            logger.info(f"Safebooru query='{booru_tags}' returned {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Safebooru search failed: {e}")
            return []

    @staticmethod
    async def search_konachan(booru_tags: str, limit: int = 50) -> list:
        """Konachan API — 高解像度アニメ壁紙特化（booru_tags は既変換済みタグ文字列）"""
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = await asyncio.to_thread(
                requests.get,
                "https://konachan.net/post.json",
                params={"tags": booru_tags, "limit": str(limit)},
                headers=headers,
                timeout=12,
            )
            if response.status_code != 200:
                logger.warning(f"Konachan returned {response.status_code}")
                return []

            posts = response.json()
            if not isinstance(posts, list):
                return []

            results = []
            for p in posts:
                file_url = p.get("file_url", "")
                if not file_url:
                    continue
                preview_url = p.get("preview_url", "") or p.get("sample_url", "") or file_url
                results.append({
                    "source": "konachan",
                    "id": str(p.get("id", ImageSearchService._generate_image_id(file_url))),
                    "title": p.get("tags", "")[:60],
                    "url": file_url,
                    "preview_url": preview_url,
                    "user": p.get("author", "Konachan"),
                    "width": p.get("width", 0),
                    "height": p.get("height", 0),
                    "like_count": p.get("score", 0),
                    "view_count": 0,
                    "tags_raw": p.get("tags", ""),
                })

            logger.info(f"Konachan query='{booru_tags}' returned {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Konachan search failed: {e}")
            return []

    @staticmethod
    async def search_safebooru_extra(booru_tags: str, page: int = 1, limit: int = 40) -> list:
        """Safebooru 追加ページ取得（booru_tags は既変換済みタグ文字列）"""
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = await asyncio.to_thread(
                requests.get,
                "https://safebooru.org/index.php",
                params={
                    "page": "dapi", "s": "post", "q": "index",
                    "json": "1", "tags": booru_tags, "limit": str(limit),
                    "pid": str(page),
                },
                headers=headers,
                timeout=12,
            )
            if response.status_code != 200:
                return []
            posts = response.json()
            if not isinstance(posts, list):
                return []
            results = []
            for p in posts:
                file_url = p.get("file_url", "")
                if not file_url or not file_url.startswith("http"):
                    file_url = f"https://safebooru.org/images/{p.get('directory','')}/{p.get('image','')}"
                preview_url = p.get("preview_url", "")
                if preview_url and not preview_url.startswith("http"):
                    preview_url = f"https://safebooru.org{preview_url}"
                results.append({
                    "source": "safebooru",
                    "id": str(p.get("id", ImageSearchService._generate_image_id(file_url))),
                    "title": p.get("tags", "")[:60],
                    "url": file_url,
                    "preview_url": preview_url or file_url,
                    "user": p.get("owner", "Safebooru"),
                    "width": p.get("width", 0),
                    "height": p.get("height", 0),
                    "like_count": p.get("score", 0),
                    "view_count": 0,
                    "tags_raw": p.get("tags", ""),
                })
            return results
        except Exception:
            return []

    @staticmethod
    async def rerank_by_feedback(
        accepted_urls: list,
        rejected_urls: list,
        candidates: list,
        evaluation_mode: str = "balanced",
    ) -> list:
        """ユーザーフィードバック（いいね/NG）で候補画像を再ランク

        1. rejected は除外
        2. CLIP が利用可能なら accepted 画像の embedding 平均でコサイン類似度リランク
        3. LLM で再評価（balanced/accurate）
        """
        import tempfile
        import os
        import numpy as np

        # rejected を除外
        rejected_set = set(rejected_urls)
        surviving = [c for c in candidates if c.get('url', '') not in rejected_set]

        if not surviving:
            return []

        if not accepted_urls:
            # いいねなし = 順序維持のまま返す
            return surviving

        # CLIP で accepted 画像のポジティブ embedding を作成
        positive_embedding = None

        if MODEL is not None:
            accepted_embeddings = []
            for url in accepted_urls[:5]:  # 最大5枚で十分
                try:
                    resp = await asyncio.to_thread(
                        requests.get, url, timeout=8,
                        headers={'User-Agent': 'Mozilla/5.0'}
                    )
                    if resp.status_code == 200:
                        img = Image.open(BytesIO(resp.content)).convert('RGB')
                        emb = MODEL.encode(img, convert_to_tensor=False)
                        accepted_embeddings.append(emb)
                except Exception as e:
                    logger.debug(f"Failed to encode accepted image {url}: {e}")

            if accepted_embeddings:
                positive_embedding = np.mean(accepted_embeddings, axis=0)
                logger.info(f"Built positive embedding from {len(accepted_embeddings)} accepted images")

        # CLIP 類似度スコアを候補に付与
        if positive_embedding is not None:
            for candidate in surviving:
                url = candidate.get('url', '')
                try:
                    resp = await asyncio.to_thread(
                        requests.get, url, timeout=5,
                        headers={'User-Agent': 'Mozilla/5.0'}
                    )
                    if resp.status_code == 200:
                        img = Image.open(BytesIO(resp.content)).convert('RGB')
                        emb = MODEL.encode(img, convert_to_tensor=False)
                        similarity = float(np.dot(positive_embedding, emb) / (
                            np.linalg.norm(positive_embedding) * np.linalg.norm(emb) + 1e-8
                        ))
                        candidate['clip_similarity'] = round(similarity * 100, 1)
                    else:
                        candidate['clip_similarity'] = 0.0
                except Exception:
                    candidate['clip_similarity'] = 0.0

            # clip_similarity + 既存スコアを統合
            for c in surviving:
                clip_s = c.get('clip_similarity', 0.0)
                existing = c.get('score', 50.0)
                c['score'] = round(clip_s * 0.7 + existing * 0.3, 1)
        else:
            # CLIP なし: accepted のタイトル/タグをヒントに LLM 再評価
            if evaluation_mode != "fast":
                accepted_titles = [c.get('title', '') for c in candidates if c.get('url') in set(accepted_urls)]
                reference = ", ".join(t for t in accepted_titles if t)
                surviving = await ImageSearchService.score_results_with_quality(
                    surviving,
                    min_width=0,
                    quality_threshold=0.0,
                    evaluation_mode=evaluation_mode,
                    reference_tags=reference,
                )

        return sorted(surviving, key=lambda x: x.get('score', 0), reverse=True)

    @staticmethod
    async def score_results_with_quality(
        results: list,
        min_width: int = 0,
        quality_threshold: float = 0.0,
        evaluation_mode: str = "balanced",
        reference_tags: str = "",
    ) -> list:
        """検索結果を LLM で評価＆スコアリング

        evaluation_mode:
          fast     - 人気度スコアのみ（軽量）
          balanced - qwen2.5:3b でタグテキスト評価
          accurate - qwen2.5vl:32b で画像解析
        """
        scored = []
        seen_urls = set()

        for i, result in enumerate(results):
            # 重複除外
            url = result.get('url', '')
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)

            if min_width > 0 and result.get('width', 0) < min_width:
                continue

            # 基本スコア（人気度）
            popularity_score = 0.0
            popularity_score += min((result.get('like_count') or 0) / 1000, 40)
            popularity_score += min((result.get('view_count') or 0) / 10000, 30)
            popularity_score += 30  # ベーススコア

            llm_score = 50

            # LLM 評価（全モードで実行）
            if evaluation_mode != "fast":
                tags_hint = reference_tags or result.get('title', '')
                try:
                    if evaluation_mode == "accurate":
                        # 画像ダウンロード＆評価
                        img_response = await asyncio.to_thread(
                            requests.get, url, timeout=8, headers={'User-Agent': 'Mozilla/5.0'}
                        )
                        if img_response.status_code == 200:
                            import tempfile
                            import os
                            suffix = f"_{result['source']}_{result['id']}.jpg"
                            tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
                            tmp.write(img_response.content)
                            tmp.close()
                            try:
                                eval_result = await ImageQualityEvaluator.evaluate_learning_suitability(
                                    tmp.name, reference_tags=tags_hint, evaluation_mode="accurate"
                                )
                                llm_score = eval_result.get('score', 50)
                                result['llm_evaluation'] = eval_result
                            finally:
                                os.unlink(tmp.name)
                    else:  # balanced
                        eval_result = await ImageQualityEvaluator.evaluate_learning_suitability(
                            "", reference_tags=tags_hint, evaluation_mode="balanced"
                        )
                        llm_score = eval_result.get('score', 50)
                        result['llm_evaluation'] = eval_result
                except asyncio.TimeoutError:
                    logger.debug(f"LLM evaluation timeout for {result.get('id')}")
                except Exception as e:
                    logger.debug(f"LLM evaluation failed for {result.get('id')}: {e}")

            # 複合スコア
            if evaluation_mode == "fast":
                combined_score = popularity_score
            else:
                # LLM + 人気度のハイブリッドスコア
                combined_score = (popularity_score * 0.5) + (llm_score * 0.5)

            if combined_score >= (quality_threshold * 100):
                result['score'] = round(combined_score, 1)
                scored.append(result)

        return sorted(scored, key=lambda x: x.get('score', 0), reverse=True)

    @staticmethod
    async def suggest_images(
        project_id: int,
        current_image_paths: list,
        tags: Optional[str] = None,
        limit: int = 30,
        min_width: int = 0,
        evaluation_mode: str = "balanced",
        manual_query: Optional[str] = None,
    ) -> dict:
        """CLIP視覚類似度優先の画像提案

        流れ:
        1. 既存データセット画像 → CLIP参照embedding構築
        2. 1タグ広域Booru検索（候補プールを最大化）
        3. 全候補をCLIPスコアリング（プレビュー画像で高速化）
        4. CLIPスコアをプライマリランクとして使用
        5. balanced/accurate モードは上位候補のみLLM補助評価
        """
        import numpy as np

        try:
            logger.info(f"[suggest_images] mode={evaluation_mode} dataset_images={len(current_image_paths)}")

            # Step 1: 既存データセット画像からCLIP参照embedding構築
            reference_embedding = await asyncio.to_thread(
                ImageSearchService._build_reference_embedding, current_image_paths
            )
            if reference_embedding is not None:
                logger.info("Reference embedding ready — using CLIP visual similarity as primary signal")
            else:
                logger.info("No reference embedding (no dataset images or CLIP unavailable) — falling back to popularity")

            # Step 2: 1タグ広域検索（多ページ・大量候補）
            # manual_query が指定されていれば _extract_broad_query をスキップしてそのまま使用
            if manual_query and manual_query.strip():
                broad_query = manual_query.strip().replace(" ", "_").lower()
                logger.info(f"Manual Booru query override: '{broad_query}'")
            else:
                broad_query = ImageSearchService._extract_broad_query(tags or "")
                logger.info(f"Broad search: '{broad_query}' (source caption: '{(tags or '')[:60]}')")

            raw_results = await asyncio.gather(
                ImageSearchService.search_safebooru(broad_query, limit=50),
                ImageSearchService.search_safebooru_extra(broad_query, page=1, limit=40),
                ImageSearchService.search_safebooru_extra(broad_query, page=2, limit=30),
                ImageSearchService.search_konachan(broad_query, limit=50),
                return_exceptions=True,
            )

            all_results: list[dict] = []
            source_raw_counts: dict[str, int] = {}
            for src in raw_results:
                if isinstance(src, list):
                    for r in src:
                        all_results.append(r)
                        s = r.get('source', 'unknown')
                        source_raw_counts[s] = source_raw_counts.get(s, 0) + 1
                else:
                    logger.warning(f"Source failed: {src}")

            logger.info(f"Raw: {len(all_results)} items {source_raw_counts}")

            # 重複URL除外
            seen: set[str] = set()
            unique: list[dict] = []
            for r in all_results:
                u = r.get("url", "")
                if u and u not in seen:
                    seen.add(u)
                    unique.append(r)

            logger.info(f"Unique after dedup: {len(unique)}")

            # Step 3: CLIPスコアリング
            # fast モードはスキップ（人気度のみ。速度優先）
            # balanced/accurate はバッチエンコードで視覚類似度を算出
            use_clip = (
                evaluation_mode != "fast"
                and reference_embedding is not None
                and MODEL is not None
            )

            if use_clip:
                # 人気度で事前フィルタ → CLIP対象を50件に絞る
                if len(unique) > 50:
                    pre_sorted = sorted(unique, key=lambda x: (x.get('like_count') or 0), reverse=True)
                    unique = pre_sorted[:50]
                    logger.info(f"Pre-filtered to top 50 for CLIP batch scoring")
                logger.info(f"CLIP batch scoring {len(unique)} candidates...")
                unique = await ImageSearchService._clip_score_candidates(unique, reference_embedding)
                for r in unique:
                    clip_s = r.get('clip_similarity', 0.0)
                    popularity = min((r.get('like_count') or 0) / 50, 20)
                    r['score'] = round(clip_s * 0.8 + popularity, 1)
            else:
                # fast モード or CLIP 利用不可: 人気度ランキング
                for r in unique:
                    r['score'] = round(30 + min((r.get('like_count') or 0) / 100, 70), 1)
                    r['clip_similarity'] = 0.0

            # スコアで降順ソート
            unique.sort(key=lambda x: x.get('score', 0), reverse=True)

            # Step 4: LLM補助評価（accurate モードのみ・上位30件）
            # balanced = CLIPスコアで十分（LLMは使わない）
            # accurate = CLIP + qwen2.5vl:32bで画像直接解析
            if evaluation_mode == "accurate":
                top_candidates = unique[:limit]
                logger.info(f"LLM image evaluation on top {len(top_candidates)} [{evaluation_mode}]")

                for r in top_candidates:
                    try:
                        img_resp = await asyncio.to_thread(
                            requests.get, r.get('url', ''), timeout=8,
                            headers={'User-Agent': 'Mozilla/5.0'}
                        )
                        if img_resp.status_code == 200:
                            import tempfile, os
                            tmp = tempfile.NamedTemporaryFile(
                                suffix=f"_{r['source']}_{r['id']}.jpg", delete=False
                            )
                            tmp.write(img_resp.content)
                            tmp.close()
                            try:
                                ev = await ImageQualityEvaluator.evaluate_learning_suitability(
                                    tmp.name, reference_tags=tags or "", evaluation_mode="accurate"
                                )
                                r['llm_evaluation'] = ev
                                llm_s = ev.get('score', 50)
                            finally:
                                os.unlink(tmp.name)
                        else:
                            llm_s = 50
                    except Exception as e:
                        logger.debug(f"LLM eval failed for {r.get('id')}: {e}")
                        llm_s = 50

                    # 最終スコア: CLIP 70% + LLM 20% + 人気度 10%
                    clip_s = r.get('clip_similarity', 0.0)
                    popularity = min((r.get('like_count') or 0) / 50, 10)
                    r['score'] = round(clip_s * 0.7 + llm_s * 0.2 + popularity, 1)

                top_candidates.sort(key=lambda x: x.get('score', 0), reverse=True)
                final = top_candidates[:limit]
            else:
                # fast / balanced: CLIPスコアで確定
                final = unique[:limit]

            breakdown = {
                src: len([r for r in all_results if r.get("source") == src])
                for src in set(source_raw_counts)
            }

            logger.info(f"Done: {len(final)} results, top_score={final[0].get('score', 0) if final else 0}")

            return {
                "project_id": project_id,
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
                "results": final,
                "total_count": len(final),
                "evaluation_mode": evaluation_mode,
                "source_breakdown": breakdown,
            }
        except Exception as e:
            logger.error(f"suggest_images failed: {e}", exc_info=True)
            return {"project_id": project_id, "status": "failed", "error": str(e)}
