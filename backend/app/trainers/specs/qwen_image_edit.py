"""Qwen-Image-Edit (画像編集) モデルスペック。

musubi-tuner の対応スクリプト名は確定次第 train_script を更新してください。
"""
from ..base import ModelSpec
from ..registry import register_spec

register_spec(ModelSpec(
    model_family="qwen_image_edit",
    display_name="Qwen-Image-Edit (I2I)",
    backend="musubi",
    train_script="qwen_edit_train_network.py",
    network_module="networks.lora_qwen_edit",
    default_resolution=1024,
    supports={"lora", "i2i", "edit"},
    checkpoint_patterns=["qwen_image_edit", "qwen-image-edit", "qwenimageedit"],
    preview_sampler="euler",
    preview_cfg=1.0,
    preview_steps=20,
    extra_toml={
        "sdpa": True,
        "seed": 42,
    },
    default_preset={
        "rank": 16,
        "alpha": 8,
        "learning_rate": 1e-4,
        "optimizer": "AdamW8bit",
        "scheduler": "cosine_with_restarts",
        "epochs": 10,
        "save_every_n_epochs": 1,
        "mixed_precision": "bf16",
        "save_precision": "bf16",
        "gradient_checkpointing": True,
    },
))
