import asyncio
import json
import logging
import base64
import requests
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

OLLAMA_API = "http://127.0.0.1:11434/api/generate"
VISION_MODEL_ACCURATE = "qwen2.5vl:32b"   # 高精度: 画像解析あり
VISION_MODEL_BALANCED = "qwen2.5:3b"       # バランス: テキスト評価のみ
VISION_MODEL = VISION_MODEL_ACCURATE        # 後方互換


class ImageQualityEvaluator:
    """LLaVA + LLM による高精度画像品質評価"""

    @staticmethod
    def encode_image_to_base64(image_path: str) -> Optional[str]:
        """画像をBase64エンコード"""
        try:
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode()
        except Exception as e:
            logger.error(f"Image encoding failed: {e}")
            return None

    @staticmethod
    async def analyze_image(image_path: str) -> dict:
        """Qwen2.5VL で画像解析"""
        try:
            if not Path(image_path).exists():
                return {"error": "Image not found"}

            # 画像をBase64エンコード
            img_b64 = ImageQualityEvaluator.encode_image_to_base64(image_path)
            if not img_b64:
                return {"error": "Failed to encode image"}

            # Qwen2.5VL に送信
            prompt = """この画像を分析してください。以下の項目について簡潔に説明してください:
1. 画像タイプ（イラスト/写真/3D等）
2. キャラクターの有無
3. 画像の鮮明度（ぼやけていないか）
4. 色彩の豊かさ
5. 構図の質"""

            payload = {
                "model": VISION_MODEL,
                "prompt": prompt,
                "images": [img_b64],
                "stream": False,
                "temperature": 0.3,
            }

            response = await asyncio.to_thread(
                requests.post,
                OLLAMA_API,
                json=payload,
                timeout=60
            )

            if response.status_code != 200:
                logger.error(f"Ollama API error: {response.status_code}")
                return {"error": "API error"}

            result = response.json()
            return {
                "analysis": result.get("response", ""),
                "model": VISION_MODEL,
            }

        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            return {"error": str(e)}

    @staticmethod
    async def evaluate_fast(tags_hint: Optional[str] = None) -> dict:
        """高速モード: LLM評価なし、固定スコアを返す"""
        return {"score": 50, "reason": "高速モード（LLM評価なし）", "suitable": True}

    @staticmethod
    async def evaluate_balanced(tags_hint: Optional[str] = None) -> dict:
        """バランスモード: qwen2.5:3b でタグテキストのみ評価（画像不要）"""
        try:
            prompt = f"""以下のタグがあるイラスト/画像がStable Diffusion LoRA学習データとして適切かを評価してください。

タグ情報: {tags_hint or "不明"}

評価基準:
- アニメ/イラストスタイルか（+30点）
- キャラクターが識別できるか（+25点）
- 鮮明・高品質そうか（+20点）
- 色彩豊かか（+15点）
- 構図が良好か（+10点）

0-100のスコアと理由を以下の形式で答えてください:
SCORE: <数字>
REASON: <簡潔な理由>
SUITABLE: <YES/NO>"""

            payload = {
                "model": VISION_MODEL_BALANCED,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.2,
            }
            response = await asyncio.to_thread(
                requests.post, OLLAMA_API, json=payload, timeout=30
            )
            if response.status_code != 200:
                return {"score": 50, "reason": "評価失敗", "suitable": True}

            text = response.json().get("response", "")
            score, reason, suitable = 50, "評価結果不明", True
            for line in text.split("\n"):
                if line.startswith("SCORE:"):
                    try:
                        score = max(0, min(100, int(line.replace("SCORE:", "").strip())))
                    except ValueError:
                        pass
                elif line.startswith("REASON:"):
                    reason = line.replace("REASON:", "").strip()
                elif line.startswith("SUITABLE:"):
                    suitable = "YES" in line.upper()
            return {"score": score, "reason": reason, "suitable": suitable}
        except Exception as e:
            logger.error(f"Balanced evaluation failed: {e}")
            return {"score": 50, "reason": str(e), "suitable": True}

    @staticmethod
    async def evaluate_learning_suitability(
        image_path: str,
        reference_tags: Optional[str] = None,
        evaluation_mode: str = "accurate",
    ) -> dict:
        """学習向き度を評価 (0-100スコア)。evaluation_mode: fast / balanced / accurate"""
        # fast モード: LLM不要
        if evaluation_mode == "fast":
            return await ImageQualityEvaluator.evaluate_fast(reference_tags)

        # balanced モード: テキストのみ評価
        if evaluation_mode == "balanced":
            return await ImageQualityEvaluator.evaluate_balanced(reference_tags)

        # accurate モード: Qwen2.5VL 画像解析
        try:
            # 画像解析
            analysis = await ImageQualityEvaluator.analyze_image(image_path)
            if "error" in analysis:
                return {
                    "score": 0,
                    "reason": analysis["error"],
                    "suitable": False
                }

            # 評価プロンプト
            eval_prompt = f"""以下の画像解析結果に基づいて、Stable Diffusion LoRA学習用データセットに適している度合いをスコア化してください。

解析結果:
{analysis['analysis']}

評価基準:
- イラスト/アニメタイプであるか（+30点）
- キャラクターが明確で識別可能か（+25点）
- 解像度が768px以上相当の鮮明度があるか（+20点）
- 色彩が豊かで単色/グレースケールではないか（+15点）
- 構図が良好で学習に最適か（+10点）

最終スコア（0-100）と判定理由を以下の形式で答えてください:
SCORE: <数字>
REASON: <簡潔な理由>
SUITABLE: <YES/NO>"""

            eval_payload = {
                "model": VISION_MODEL,
                "prompt": eval_prompt,
                "stream": False,
                "temperature": 0.2,
            }

            response = await asyncio.to_thread(
                requests.post,
                OLLAMA_API,
                json=eval_payload,
                timeout=60
            )

            if response.status_code != 200:
                return {"score": 50, "reason": "評価失敗", "suitable": False}

            eval_text = response.json().get("response", "")

            # スコア抽出
            score = 50
            reason = "評価結果が不明"
            suitable = False

            for line in eval_text.split("\n"):
                if line.startswith("SCORE:"):
                    try:
                        score = int(line.replace("SCORE:", "").strip())
                        score = max(0, min(100, score))
                    except ValueError:
                        pass
                elif line.startswith("REASON:"):
                    reason = line.replace("REASON:", "").strip()
                elif line.startswith("SUITABLE:"):
                    suitable = "YES" in line.upper()

            return {
                "score": score,
                "reason": reason,
                "suitable": suitable,
                "analysis": analysis.get("analysis", ""),
            }

        except Exception as e:
            logger.error(f"Suitability evaluation failed: {e}")
            return {
                "score": 0,
                "reason": f"評価エラー: {str(e)}",
                "suitable": False
            }

    @staticmethod
    async def batch_evaluate(image_paths: list) -> list:
        """複数画像を一括評価"""
        results = []
        for path in image_paths:
            result = await ImageQualityEvaluator.evaluate_learning_suitability(path)
            results.append({
                "path": path,
                **result
            })
        return results
