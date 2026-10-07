from ..base import ModelSpec
from ..registry import register_spec

register_spec(ModelSpec(
    model_family="sd",
    display_name="Stable Diffusion 1.x / 2.x",
    backend="kohya",
    train_script="train_network.py",
    network_module="networks.lora",
    default_resolution=512,
    supports={"lora", "t2i", "dreambooth"},
    checkpoint_patterns=["sd15", "sd1_", "sd2_", "v1-", "v2-"],
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
