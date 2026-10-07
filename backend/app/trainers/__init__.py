"""trainers パッケージ — モデル固有情報をプラグイン形式で管理する。

import 時に全スペックと全ランナーを自動登録する。
"""
from . import registry
from .base import BaseTrainer, CacheStep, ModelSpec, RequiredModel
from .registry import detect_spec, get_runner, get_spec, list_specs, register_spec
from .runners import KohyaRunner, MusubiRunner

# ── ランナーを登録 ────────────────────────────────────────────────────────────
registry.register_runner("musubi", MusubiRunner())
registry.register_runner("kohya", KohyaRunner())

# ── 全スペックを登録（specs/__init__.py が各モデルを register_spec する） ──
from . import specs  # noqa: E402, F401

__all__ = [
    "BaseTrainer",
    "CacheStep",
    "ModelSpec",
    "RequiredModel",
    "MusubiRunner",
    "KohyaRunner",
    "registry",
    "get_spec",
    "get_runner",
    "list_specs",
    "detect_spec",
    "register_spec",
]
