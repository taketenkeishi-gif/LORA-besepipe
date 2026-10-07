from ..base import ModelSpec
from ..registry import register_spec

register_spec(ModelSpec(
    model_family="hunyuanvideo",
    display_name="HunyuanVideo (動画生成)",
    backend="musubi",
    train_script="hv_train_network.py",
    network_module="networks.lora_hv",
    default_resolution=720,
    supports={"lora", "video", "t2v"},
    checkpoint_patterns=["hunyuan", "hyvideo", "hunyuanvideo"],
    extra_toml={
        "sdpa": True,
        "timestep_sampling": "logit_normal",
        "weighting_scheme": "none",
        "seed": 42,
    },
    default_preset={
        "rank": 32,
        "alpha": 16,
        "learning_rate": 5e-5,
        "optimizer": "AdamW8bit",
        "scheduler": "cosine_with_restarts",
        "epochs": 10,
        "save_every_n_epochs": 1,
        "mixed_precision": "bf16",
        "save_precision": "bf16",
        "gradient_checkpointing": True,
    },
))
