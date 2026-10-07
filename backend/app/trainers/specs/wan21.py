from ..base import ModelSpec
from ..registry import register_spec

register_spec(ModelSpec(
    model_family="wan21",
    display_name="Wan2.1 (動画生成)",
    backend="musubi",
    train_script="wan_train_network.py",
    network_module="networks.lora_wan",
    default_resolution=512,
    supports={"lora", "video", "t2v", "i2v"},
    checkpoint_patterns=["wan2", "wan_2", "wan21", "wan-2"],
    extra_toml={
        "sdpa": True,
        "timestep_sampling": "logit_normal",
        "weighting_scheme": "none",
        "seed": 42,
    },
    default_preset={
        "rank": 32,
        "alpha": 16,
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
