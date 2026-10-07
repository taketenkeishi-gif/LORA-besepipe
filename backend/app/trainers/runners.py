from __future__ import annotations

from typing import Any

from .base import BaseTrainer, ModelSpec


class MusubiRunner(BaseTrainer):
    """musubi-tuner バックエンド用ランナー。

    musubi-tuner は kohya_ss と異なる TOML キー名を使う:
      - pretrained_model_name_or_path → dit
      - 独自のキー: sdpa, timestep_sampling, weighting_scheme, etc.
    """

    def common_toml(self, spec: ModelSpec, cfg: dict[str, Any]) -> dict[str, Any]:
        """musubi-tuner 全モデル共通の TOML キーを返す。"""

        def _bool(key: str, default: bool = False) -> bool:
            return bool(cfg.get(key, default))

        optimizer_name = cfg.get("optimizer", "AdamW8bit")
        _AUTO_LR = {"Prodigy", "DAdaptAdam", "DAdaptAdaGrad", "DAdaptSGD"}
        raw_lr = cfg.get("learning_rate", 1e-4)
        lr = 1.0 if optimizer_name in _AUTO_LR else raw_lr

        return {
            "dit":                         cfg.get("base_checkpoint_path", ""),
            "dataset_config":              cfg.get("dataset_config", ""),
            "output_dir":                  cfg.get("output_dir", ""),
            "output_name":                 cfg.get("output_name", "lora_output"),
            "network_module":              spec.network_module,
            "network_dim":                 int(cfg.get("rank", 32)),
            "network_alpha":               float(cfg.get("alpha", 16)),
            "max_train_epochs":            int(cfg.get("epochs", 10)),
            "save_every_n_epochs":         int(cfg.get("save_every_n_epochs", 1)),
            "optimizer_type":              optimizer_name,
            "lr_scheduler":                cfg.get("scheduler", "cosine_with_restarts"),
            "lr_scheduler_num_cycles":     1,
            "learning_rate":               lr,
            "mixed_precision":             cfg.get("mixed_precision", "bf16"),
            "save_precision":              cfg.get("save_precision", "bf16"),
            "gradient_checkpointing":      _bool("gradient_checkpointing", default=True),
            "logging_dir":                 cfg.get("logs_dir", ""),
            # Resume: 既存LoRA重みを初期値として読み込む(musubi-tuner本体の
            # trainer_base.py: args.network_weights指定時にnetwork.load_weights()
            # を呼ぶ実装を確認済み。kohyaのnetwork_weightsと同じ仕組み)。
            **({"network_weights": cfg["resume_from_checkpoint"]} if cfg.get("resume_from_checkpoint") else {}),
        }


class KohyaRunner(BaseTrainer):
    """kohya_ss バックエンド用ランナー。

    kohya_ss は pretrained_model_name_or_path キーを使う。
    TOML 方式ではなく argparse 方式で引数を渡すため、
    common_toml は kohya 用の引数辞書を返す（キーは argparse 長形式）。
    """

    def common_toml(self, spec: ModelSpec, cfg: dict[str, Any]) -> dict[str, Any]:
        """kohya 共通引数（参照用。実際の学習は argparse リスト形式で渡す）。"""
        return {
            "pretrained_model_name_or_path": cfg.get("base_checkpoint_path", ""),
            "output_dir":                    cfg.get("output_dir", ""),
            "output_name":                   cfg.get("output_name", "lora_output"),
            "network_module":                spec.network_module,
            "network_dim":                   int(cfg.get("rank", 32)),
            "network_alpha":                 float(cfg.get("alpha", 16)),
            "max_train_epochs":              int(cfg.get("epochs", 10)),
            "save_every_n_epochs":           int(cfg.get("save_every_n_epochs", 1)),
            "optimizer_type":                cfg.get("optimizer", "AdamW8bit"),
            "lr_scheduler":                  cfg.get("scheduler", "cosine_with_restarts"),
            "learning_rate":                 cfg.get("learning_rate", 1e-4),
            "mixed_precision":               cfg.get("mixed_precision", "fp16"),
            "save_precision":                cfg.get("save_precision", "fp16"),
        }
