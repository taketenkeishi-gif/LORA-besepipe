"""プレビュー生成プロンプトの構築ヘルパー。

プレビュープロンプトは 3 つの入力に分かれる:
  - quality       品質ポジティブ (masterpiece, best quality, ...)
  - positive      特徴ポジティブ (キャラ/シーン固有のタグ)
  - negative      ネガティブ

最終的な positive は [quality, trigger_words, positive] を空要素を除いて連結する。
training.py（学習開始時）と settings.py（学習中の上書き同期）の両方が
このモジュールを使い、sample_prompts.txt の生成ロジックを一元化する。
"""
from __future__ import annotations

# プレビュー生成解像度の既定値（SDXL 標準の 1024）
DEFAULT_PREVIEW_RESOLUTION = 1024

# 既定値（新規プロンプト・未設定時に投入する初期値）
DEFAULT_QUALITY = "masterpiece, best quality, very aesthetic, absurdres, anime coloring"
DEFAULT_FEATURE = "1girl, solo"
DEFAULT_NEGATIVE = (
    "lowres, worst quality, jpeg artifacts, blurry, noise, grain, film grain, "
    "unfinished, displeasing, artistic error, text, watermark, signature, username, "
    "scan, abstract, error, cropped, split_window, (bad anatomy:1.25), "
    "(anatomical inconsistency:1.2), (wrong limb proportion:1.35), "
    "(incorrect body ratio:1.3), (deformed limbs:1.35), (disconnected limbs:1.3), "
    "(misaligned joints:1.3), (broken anatomy:1.25), (multiple views:1.3), "
    "(bad hands:1.25), (malformed hands:1.3), (distorted fingers:1.3), "
    "(extra face:1.3), (multiple face:1.3), (perspective error:1.2), "
    "(flat body depth:1.15), (distorted foreshortening:1.2), standard, "
    "conventional, extra fingers, low quality"
)


def _clean(s: object) -> str:
    """前後空白・末尾カンマを除去。"""
    return str(s or "").strip().strip(",").strip()


def compose_positive(item: dict) -> str:
    """quality + trigger_words + 特徴positive を連結して最終 positive を作る。"""
    quality = _clean(item.get("quality"))
    trigger = _clean(item.get("trigger_words"))
    feature = _clean(item.get("positive"))
    parts = [s for s in (quality, trigger, feature) if s]
    return ", ".join(parts) if parts else DEFAULT_FEATURE


def build_sample_line(
    item: dict,
    resolution: int,
    *,
    steps: int = 20,
    cfg: float = 7.0,
    seed: int = 42,
) -> str:
    """kohya_ss sample_prompts.txt の 1 行を構築する。

    steps / cfg はモデル世代によって適切な値が異なる（SD/SDXL は steps=20/cfg=7 が定番だが、
    Flux/Qwen 系は cfg≈1 が標準）。呼び出し側は ModelSpec.preview_steps / preview_cfg を渡すこと。
    """
    pos = compose_positive(item)
    neg = (str(item.get("negative") or "").strip()) or DEFAULT_NEGATIVE
    cfg_str = str(int(cfg)) if float(cfg).is_integer() else str(cfg)
    return f"{pos} --n {neg} --w {resolution} --h {resolution} --s {int(steps)} --l {cfg_str} --d {int(seed)}"


def default_prompt_item() -> dict:
    """プロンプト未設定時に使う既定の 1 件。"""
    return {
        "label": "デフォルト",
        "quality": DEFAULT_QUALITY,
        "positive": DEFAULT_FEATURE,
        "negative": DEFAULT_NEGATIVE,
        "trigger_words": "",
    }


def resolve_preview_params(
    conn,
    *,
    spec_sampler: str,
    spec_cfg: float,
    spec_steps: int,
) -> tuple[str, float, int]:
    """app_settings のユーザー上書き (preview_sampler/cfg/steps) を、
    未設定なら ModelSpec の既定値にフォールバックして解決する。

    UI（設定画面）で明示的に値を入れた場合のみモデル既定を上書きする。
    """
    sampler, cfg, steps = spec_sampler, spec_cfg, spec_steps
    try:
        rows = conn.execute(
            "SELECT key, value FROM app_settings "
            "WHERE key IN ('preview_sampler','preview_cfg','preview_steps')"
        ).fetchall()
        pmap = {str(r["key"]): str(r["value"]).strip() for r in rows}
        if pmap.get("preview_sampler"):
            sampler = pmap["preview_sampler"]
        if pmap.get("preview_cfg"):
            cfg = float(pmap["preview_cfg"])
        if pmap.get("preview_steps"):
            steps = int(float(pmap["preview_steps"]))
    except Exception:
        pass
    return sampler, cfg, steps


def load_active_prompt_items(conn) -> tuple[list[dict], int]:
    """app_settings からプレビュープロンプト一覧と解像度(生値)を読む。

    training.py / settings.py の既存ロジックと同じ読み取り規則
    （preview_prompts_json 優先、無ければ positive_prompt/negative_prompt から1件）。
    呼び出し側で conn を渡す（クローズは呼び出し側の責務）。
    戻り値の resolution は未設定時 DEFAULT_PREVIEW_RESOLUTION。
    """
    prompt_items: list[dict] = []
    pmap: dict[str, str] = {}
    try:
        import json as _json

        jrow = conn.execute(
            "SELECT value FROM app_settings WHERE key='preview_prompts_json'"
        ).fetchone()
        prows = conn.execute(
            "SELECT key, value FROM app_settings "
            "WHERE key IN ('positive_prompt','negative_prompt','preview_resolution')"
        ).fetchall()
        if jrow and jrow["value"]:
            parsed = _json.loads(jrow["value"])
            if isinstance(parsed, list):
                prompt_items = [
                    p for p in parsed
                    if isinstance(p, dict)
                    and (
                        str(p.get("quality", "")).strip()
                        or str(p.get("positive", "")).strip()
                        or str(p.get("trigger_words", "")).strip()
                    )
                ]
        pmap = {str(r["key"]): str(r["value"]) for r in prows}
    except Exception:
        pmap = {}
    if not prompt_items:
        item = default_prompt_item()
        if pmap.get("positive_prompt"):
            item["positive"] = pmap["positive_prompt"]
        if pmap.get("negative_prompt"):
            item["negative"] = pmap["negative_prompt"]
        prompt_items = [item]
    try:
        resolution = int(pmap.get("preview_resolution") or DEFAULT_PREVIEW_RESOLUTION)
    except (ValueError, TypeError):
        resolution = DEFAULT_PREVIEW_RESOLUTION
    return prompt_items, resolution
