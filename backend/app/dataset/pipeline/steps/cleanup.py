"""Cleanup ステップ — Qwen Cleanup（ComfyUI 経由・任意）。

公式native Qwen-Image-Edit-2511構成を
ComfyUIClient.qwen_cleanup() が利用する。Resize/Upscale 後の画像から
背景・吹き出し・ノイズを除去する。use_qwen=False またはComfyUI未接続時は
スキップする（should_run が False を返す）。

失敗時は例外を送出する（Qwen は要求時には必須処理として扱い、画像単位で
失敗カウントする — ESRGAN のような静かなフォールバックはしない。これは
既存 UI の「Qwen Cleanup」ボタンを押した以上、処理が行われたことを
期待する既存仕様を維持するため）。
"""
from __future__ import annotations

import io
import tempfile
from pathlib import Path

from PIL import Image

from ..base import PipelineContext, PipelineStep


class CleanupStep(PipelineStep):
    name = "cleanup"

    def should_run(self, ctx: PipelineContext) -> bool:
        return bool(ctx.options.get("use_qwen")) and ctx.extra.get("comfy_client") is not None

    def process(self, ctx: PipelineContext) -> PipelineContext:
        comfy_client = ctx.extra["comfy_client"]
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            tmp_path = Path(tf.name)
        ctx.image.save(tmp_path)
        try:
            qwen_bytes = comfy_client.qwen_cleanup(
                tmp_path, width=ctx.width, height=ctx.height, timeout=600.0
            )
            ctx.image = Image.open(io.BytesIO(qwen_bytes)).convert("RGB")
        finally:
            tmp_path.unlink(missing_ok=True)
        return ctx
