"""Resize ステップ — アスペクト比維持リサイズ / 正方形クロップ。"""
from __future__ import annotations

from PIL import Image

from ..base import PipelineContext, PipelineStep


def round8(n: int) -> int:
    """ComfyUI の潜在空間要件に合わせ 8 の倍数に丸める（最小 8）。"""
    return max(8, (n // 8) * 8)


class ResizeStep(PipelineStep):
    """target_size / resize_mode に応じて画像を変形する。

    resize_mode:
      - "resize_longer":  長辺を target_size に合わせる（アスペクト比維持）
      - "resize_shorter": 短辺を target_size に合わせる（アスペクト比維持）
      - それ以外:          正方形クロップ後 target_size にリサイズ
    """

    name = "resize"

    def process(self, ctx: PipelineContext) -> PipelineContext:
        target_size = int(ctx.options.get("target_size", 1024))
        resize_mode = ctx.options.get("resize_mode", "resize_longer")

        im = ctx.image
        if im.mode not in ("RGB", "RGBA"):
            im = im.convert("RGB")
        w, h = im.size

        if resize_mode == "resize_longer":
            ratio = target_size / max(w, h)
            new_w = round8(round(w * ratio))
            new_h = round8(round(h * ratio))
            current_img = im.copy()
        elif resize_mode == "resize_shorter":
            ratio = target_size / min(w, h)
            new_w = round8(round(w * ratio))
            new_h = round8(round(h * ratio))
            current_img = im.copy()
        else:  # square_crop
            side = min(w, h)
            left = (w - side) // 2
            top = (h - side) // 2
            current_img = im.crop((left, top, left + side, top + side))
            new_w = new_h = round8(target_size)

        ctx.extra["enlarging"] = (new_w > w or new_h > h)
        ctx.extra["target_w"] = new_w
        ctx.extra["target_h"] = new_h
        ctx.image = current_img
        ctx.width, ctx.height = current_img.size
        return ctx
