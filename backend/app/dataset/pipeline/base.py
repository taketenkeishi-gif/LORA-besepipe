"""前処理パイプラインの共通境界 — PipelineStep / PipelineContext。

Resize -> Upscale -> Cleanup -> Save という順序を基本とし、各ステップは
独立した PipelineStep として実装する。新しいステップ（例: Crop, Watermark
除去等）を追加する際もこのインターフェースに従うだけでよく、
PipelineRunner 側の変更は不要。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass
class PipelineContext:
    """1 画像がパイプラインを通過する間、ステップ間で引き継がれる状態。"""

    src_path: Path
    image: Image.Image
    width: int
    height: int
    #: dataset_items.id（既存タグ編集UIとの互換用。単発画像テストなど呼び出し元に
    #: DB行がない場合は None のままでよい — CaptionStep はその場合 DB 反映をスキップする）
    item_id: int | None = None
    settings: dict[str, Any] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)
    # ステップ間で共有する追加情報（例: enlarging フラグ、使用モデル名等）。
    extra: dict[str, Any] = field(default_factory=dict)


class PipelineStep(ABC):
    """1 処理単位（Resize/Upscale/Cleanup/Save 等）を表す境界。

    should_run() が False を返せばスキップされる（例: use_esrgan=False）。
    process() は失敗時、続行可能なら ctx をそのまま返し、致命的なら例外を
    送出する（呼び出し元 PipelineRunner が画像単位で捕捉し、他の画像の
    処理は継続する）。
    """

    #: ログ・進捗表示に使う名前
    name: str = ""

    def should_run(self, ctx: PipelineContext) -> bool:
        return True

    @abstractmethod
    def process(self, ctx: PipelineContext) -> PipelineContext:
        ...
