"""training パッケージ — モデル非依存の学習基盤。

ModelSpec.training_backend というキーで、実際に学習を実行する
TrainingBackend 実装（musubi / sdxl 等）を解決する。
UI・API はこのキーの向こう側にある実装詳細（musubi-tuner / kohya_ss）を知らない。
"""
from __future__ import annotations

from . import registry
from .base import PreparedRun, TrainingBackend, TrainingContext
from .registry import (
    detect_model_family,
    get_training_backend,
    list_training_backends,
    register_training_backend,
    resolve_training_backend,
)

# ── 既知のバックエンドを登録 ──────────────────────────────────────────────
from .backends.musubi import MusubiBackend  # noqa: E402
from .backends.sdxl import SDXLBackend  # noqa: E402
from .backends.anima import AnimaBackend  # noqa: E402
from .backends.simulated import SimulatedBackend  # noqa: E402
from .backends.ai_toolkit import AiToolkitBackend  # noqa: E402

register_training_backend("musubi", MusubiBackend())
register_training_backend("sdxl", SDXLBackend())
register_training_backend("anima", AnimaBackend())
register_training_backend("simulated", SimulatedBackend())
register_training_backend("ai_toolkit", AiToolkitBackend())

__all__ = [
    "TrainingBackend",
    "TrainingContext",
    "PreparedRun",
    "registry",
    "get_training_backend",
    "register_training_backend",
    "list_training_backends",
    "detect_model_family",
    "MusubiBackend",
    "SDXLBackend",
    "AnimaBackend",
    "SimulatedBackend",
    "AiToolkitBackend",
    "resolve_training_backend",
]
