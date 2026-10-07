"""Caption ステップ — WD14 Auto Caption（既存 tagger サービス経由・任意）。

既存 /tags/generate と同じ WD14 推論・しきい値・カテゴリ除外規則を
app.services.tagger.predict_image() 経由で用いる。Cleanup 後の画像
（背景・ノイズ除去済み）からタグを生成するため、素の画像に対する
WD14 推論より高精度なキャプションが期待できる。

生成したキャプションは:
  1. dataset_items.caption / caption_source='wd14' を更新し、元画像パスにも
     同名 .txt を書き出す（既存 /tags/generate・タグ編集 UI と全く同じ
     ファイル規約 — Path(file_path).with_suffix(".txt")）。
  2. SaveStep が processed/ 側の出力画像にも同名 .txt を書き出す
     （ctx.extra["caption"] 経由）。

use_caption=False（既定）またはタグ生成器が利用不可の場合はスキップする。
"""
from __future__ import annotations

from ....db import get_conn
from ..base import PipelineContext, PipelineStep


class CaptionStep(PipelineStep):
    name = "caption"

    def should_run(self, ctx: PipelineContext) -> bool:
        return bool(ctx.options.get("use_caption"))

    def process(self, ctx: PipelineContext) -> PipelineContext:
        from ....services import tagger as tagger_svc  # noqa: PLC0415

        general_thresh = float(ctx.options.get("caption_general_thresh", 0.35))
        character_thresh = float(ctx.options.get("caption_character_thresh", 0.85))
        remove_character_tags = bool(ctx.options.get("caption_remove_character_tags", False))

        caption = tagger_svc.predict_image(
            ctx.image,
            general_thresh=general_thresh,
            character_thresh=character_thresh,
            remove_character_tags=remove_character_tags,
        )
        ctx.extra["caption"] = caption

        if ctx.item_id is not None:
            conn = get_conn()
            try:
                conn.execute(
                    "UPDATE dataset_items SET caption = ?, caption_source = 'wd14' WHERE id = ?",
                    (caption, ctx.item_id),
                )
                conn.commit()
            finally:
                conn.close()
            ctx.src_path.with_suffix(".txt").write_text(caption, encoding="utf-8")

        return ctx
