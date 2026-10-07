"""Upscale ステップ — ESRGAN（ComfyUI 経由・任意） + 最終サイズ確定。

ResizeStep が計算した target_w/target_h に対し、拡大が必要かつ ESRGAN が
要求されていれば ComfyUI で高品質アップスケールしてから、最終的に
Lanczos で target_w/target_h ぴったりに合わせる。この「最終サイズ確定」は
ESRGAN 未使用時にも必要なため、このステップは常に実行される
（should_run は常に True。ESRGAN 呼び出し自体は use_esrgan かつ拡大時のみ）。
"""
from __future__ import annotations

import io
import logging
import tempfile
from pathlib import Path

from PIL import Image

from ..base import PipelineContext, PipelineStep

logger = logging.getLogger(__name__)


class UpscaleStep(PipelineStep):
    name = "upscale"

    def process(self, ctx: PipelineContext) -> PipelineContext:
        use_esrgan = bool(ctx.options.get("use_esrgan"))
        esrgan_model = ctx.options.get("esrgan_model") or ""
        comfy_client = ctx.extra.get("comfy_client")
        enlarging = bool(ctx.extra.get("enlarging"))
        new_w = int(ctx.extra.get("target_w", ctx.width))
        new_h = int(ctx.extra.get("target_h", ctx.height))

        current_img = ctx.image
        if use_esrgan and enlarging and comfy_client and esrgan_model:
            try:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                    tmp_path = Path(tf.name)
                current_img.convert("RGB").save(tmp_path)
                try:
                    up_bytes = comfy_client.upscale_image(tmp_path, esrgan_model, timeout=300.0)
                    current_img = Image.open(io.BytesIO(up_bytes))
                finally:
                    tmp_path.unlink(missing_ok=True)
            except Exception as uexc:  # noqa: BLE001
                logger.warning("ESRGAN failed for %s, fallback to Pillow resize: %s", ctx.src_path, uexc)

        # 最終サイズ確定（ESRGAN 有無に関わらず必ず target_w/target_h に合わせる）
        ctx.image = current_img.resize((new_w, new_h), Image.LANCZOS).convert("RGB")
        ctx.width, ctx.height = new_w, new_h
        return ctx
