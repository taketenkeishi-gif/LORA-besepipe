"""training_backend 名 -> TrainingBackend インスタンスの解決。

ModelSpec（app.trainers.registry 側の既存レジストリ）はモデルの「仕様」を、
ここでは「実行エンジン」を管理する。両者は model_spec.training_backend という
文字列キーで結びつく。UI・API はこのキーの実装詳細（musubi-tuner / kohya_ss
であること）を知らなくてよい。
"""
from __future__ import annotations

from pathlib import Path

from .base import TrainingBackend

_BACKENDS: dict[str, TrainingBackend] = {}


def register_training_backend(name: str, backend: TrainingBackend) -> None:
    _BACKENDS[name] = backend


def get_training_backend(name: str) -> TrainingBackend | None:
    return _BACKENDS.get(name)


def list_training_backends() -> list[str]:
    return list(_BACKENDS.keys())


def resolve_training_backend(model_family: str, requested: str = "auto") -> str:
    """Resolve the execution engine without changing ModelSpec defaults."""
    requested = (requested or "auto").strip().lower()
    if requested == "ai_toolkit":
        if model_family not in {"krea2", "flux"}:
            raise ValueError(f"AI Toolkit は {model_family} に対応していません")
        return "ai_toolkit"
    if requested == "musubi":
        if model_family in {"sd", "sdxl", "anima"}:
            raise ValueError(f"musubi-tuner は {model_family} に対応していません")
        return "musubi"
    if requested == "kohya":
        if model_family not in {"sd", "sdxl", "anima"}:
            raise ValueError(f"kohya_ss は {model_family} に対応していません")
        return "anima" if model_family == "anima" else "sdxl"
    return "sdxl" if model_family in {"sd", "sdxl"} else (
        "anima" if model_family == "anima" else "musubi"
    )


def detect_model_family(checkpoint_path: str, explicit_family: str = "auto") -> str:
    """モデルファミリーを判定する。explicit_family が "auto" 以外なら優先。

    API 層（学習開始・時間予測）と各 Backend の両方から呼ばれる共有ロジックのため、
    training_backend 解決と対をなすこのモジュールに置く。
    """
    if explicit_family and explicit_family not in ("auto", ""):
        return explicit_family
    name = Path(checkpoint_path).name.lower()
    if any(k in name for k in ("wan2", "wan_", "-wan")):
        return "wan21"
    if any(k in name for k in ("hunyuan", "hyvideo", "hunyuanvideo")):
        return "hunyuanvideo"
    if any(k in name for k in ("krea", "krea2")):
        return "krea2"
    if "flux" in name:
        return "flux"
    # edit 系を先に判定（"qwen_image" は "qwen_image_edit" の部分文字列のため順序が重要）
    if any(k in name for k in ("qwen_image_edit", "qwen-image-edit", "qwenimageedit")):
        return "qwen_image_edit"
    if any(k in name for k in ("qwen_image", "qwen-image", "qwenimage", "qwen2image")):
        return "qwen_image"
    if any(k in name for k in ("zimage", "z-image", "z_image")):
        return "zimage"
    if "anima" in name and "animagine" not in name:
        return "anima"
    # header 判定で SDXL かチェック（循環 import 回避のため遅延 import）
    from .backends.sdxl import detect_sdxl
    if detect_sdxl(checkpoint_path):
        return "sdxl"
    return "sd"
