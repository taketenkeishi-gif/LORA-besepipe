from ..base import ModelSpec
from ..registry import register_spec

register_spec(ModelSpec(
    model_family="flux",
    display_name="FLUX.1 (テキスト→画像)",
    backend="musubi",
    train_script="flux_train_network.py",
    network_module="networks.lora_flux",
    default_resolution=1024,
    supports={"lora", "t2i"},
    checkpoint_patterns=["flux"],
    preview_sampler="euler",
    preview_cfg=1.0,
    preview_steps=20,
    extra_toml={
        "sdpa": True,
        "timestep_sampling": "shift",
        "weighting_scheme": "none",
        "discrete_flow_shift": 3.0,
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
