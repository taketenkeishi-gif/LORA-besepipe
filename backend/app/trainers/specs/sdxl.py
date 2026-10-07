from ..base import ModelSpec
from ..registry import register_spec

register_spec(ModelSpec(
    model_family="sdxl",
    display_name="Stable Diffusion XL",
    backend="kohya",
    train_script="sdxl_train_network.py",
    network_module="networks.lora",
    default_resolution=1024,
    supports={"lora", "t2i", "dreambooth"},
    checkpoint_patterns=["sdxl", "xl_base", "xl-base"],
    default_preset={
        "rank": 32,
        "alpha": 16,
        "learning_rate": 1e-4,
        "optimizer": "AdamW8bit",
        "scheduler": "cosine_with_restarts",
        "epochs": 10,
        "save_every_n_epochs": 1,
        "mixed_precision": "fp16",
        "save_precision": "fp16",
        "gradient_checkpointing": True,
    },
))
