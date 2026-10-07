"""モデル非依存のプレビュー生成基盤 — PreviewProvider。

学習基盤（app.training）とは完全に独立している。プレビュー生成が失敗しても
学習 run 自体は失敗させない、という方針はこの層の責務として保証する
（呼び出し元は例外を握りつぶすことが期待されるが、Provider 自身も
「1 プロンプトの失敗が他のプロンプト・学習ループに波及しない」よう努める）。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PreviewRequest:
    """1 チェックポイントに対するプレビュー生成リクエスト。"""

    model_family: str
    checkpoint_path: str
    run_id: int
    epoch: int
    prompt_items: list[dict[str, Any]]
    resolution: int
    settings: dict[str, str]
    sampler: str = ""
    cfg: float = 0.0
    steps: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class PreviewResult:
    """1 プロンプトぶんの生成結果。失敗時は image_bytes=None, error に理由。"""

    prompt_index: int
    seed: int
    image_bytes: bytes | None = None
    error: str | None = None


class PreviewProvider(ABC):
    """PreviewProvider は「プレビューをどう作るか」を隠蔽する抽象境界。

    KREA2 のように特定 Provider（例: ComfyUI）で生成が失敗しても、それは
    Provider 単位の問題であり、学習 run の成否とは無関係に扱われる。
    """

    #: UI/API に露出してよい表示名
    display_name: str = ""

    @abstractmethod
    def is_ready(self, settings: dict[str, str]) -> bool:
        """このプロバイダで生成できる環境が揃っているか。"""
        ...

    @abstractmethod
    def generate_preview(self, request: PreviewRequest) -> list[PreviewResult]:
        """request.prompt_items の数だけ PreviewResult を返す。

        個々のプロンプトの生成失敗は例外を投げず PreviewResult.error に格納する
        （1件の失敗で他のプロンプトの生成を諦めないため）。
        Provider 全体が利用不能な場合（未接続等）は例外を送出してよい —
        呼び出し元（app.preview.registry 経由の呼び出し）でまとめて捕捉される。
        """
        ...
