"""preview パッケージ — モデル非依存のプレビュー生成基盤。

ModelSpec.preview_backend というキーで、実際にプレビューを生成する
PreviewProvider 実装（comfyui 等）を解決する。学習基盤（app.training）とは
独立しており、プレビュー生成の失敗が学習 run の成否に影響することはない。
"""
from __future__ import annotations

from . import registry
from .base import PreviewProvider, PreviewRequest, PreviewResult
from .registry import (
    generate_preview_safe,
    get_preview_provider,
    list_preview_providers,
    register_preview_provider,
)

# ── 既知のプロバイダを登録 ────────────────────────────────────────────────
from .providers.comfyui import ComfyUIPreviewProvider  # noqa: E402

register_preview_provider("comfyui", ComfyUIPreviewProvider())

__all__ = [
    "PreviewProvider",
    "PreviewRequest",
    "PreviewResult",
    "registry",
    "register_preview_provider",
    "get_preview_provider",
    "list_preview_providers",
    "generate_preview_safe",
    "ComfyUIPreviewProvider",
]
