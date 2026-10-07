"""Save ステップ — processed/ ディレクトリへ PNG 保存（元画像は変更しない）。

CaptionStep が ctx.extra["caption"] を設定していれば、既存キャプション
フォーマット（image.png と同名の image.txt）で processed/ 側にも
サイドカーを書き出す。
"""
from __future__ import annotations

from ..base import PipelineContext, PipelineStep


class SaveStep(PipelineStep):
    name = "save"

    def process(self, ctx: PipelineContext) -> PipelineContext:
        processed_dir = ctx.options["processed_dir"]
        out_path = processed_dir / (ctx.src_path.stem + ".png")
        ctx.image.save(out_path, optimize=True)
        ctx.extra["out_path"] = out_path

        caption = ctx.extra.get("caption")
        if caption is not None:
            txt_path = out_path.with_suffix(".txt")
            txt_path.write_text(caption, encoding="utf-8")
            ctx.extra["caption_path"] = txt_path

        return ctx
