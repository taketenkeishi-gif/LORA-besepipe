from __future__ import annotations

from pathlib import Path

from .base import BaseTrainer, ModelSpec

_SPECS: dict[str, ModelSpec] = {}
_RUNNERS: dict[str, BaseTrainer] = {}


# ── 登録 API ─────────────────────────────────────────────────────────────────

def register_spec(spec: ModelSpec) -> None:
    _SPECS[spec.model_family] = spec


def register_runner(backend: str, runner: BaseTrainer) -> None:
    _RUNNERS[backend] = runner


# ── 取得 API ─────────────────────────────────────────────────────────────────

def get_spec(model_family: str) -> ModelSpec | None:
    return _SPECS.get(model_family)


def get_runner(backend: str) -> BaseTrainer | None:
    return _RUNNERS.get(backend)


def list_specs() -> list[ModelSpec]:
    return list(_SPECS.values())


def detect_spec(checkpoint_path: str) -> ModelSpec | None:
    """チェックポイントのファイル名からモデルファミリーを推定して ModelSpec を返す。"""
    name = Path(checkpoint_path).name.lower()
    for spec in _SPECS.values():
        if any(pat in name for pat in spec.checkpoint_patterns):
            return spec
    return None
