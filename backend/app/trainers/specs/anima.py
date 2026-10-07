from ..base import ModelSpec
from ..registry import register_spec

register_spec(ModelSpec(
    model_family="anima",
    display_name="Anima",
    # kohya_ss(sd-scripts)配下の anima_train_network.py を使うが、SDXLBackendとは
    # 引数体系が別物(DiT+Qwen3+LLM Adapter+Qwen-Image VAE構成、network_module=
    # networks.lora_anima)のため、専用のAnimaBackend(training_backend="anima")へ
    # 明示的にルーティングする(backend="kohya"のままだとModelSpec.__post_init__が
    # 自動的に"sdxl"に読み替えてしまい、SDXLBackendに誤って割り当てられるため)。
    backend="kohya",
    training_backend="anima",
    train_script="anima_train_network.py",
    network_module="networks.lora_anima",
    default_resolution=1024,
    supports={"lora", "t2i"},
    checkpoint_patterns=["anima"],
    default_preset={
        "rank": 16,
        "alpha": 16,
        "learning_rate": 1e-4,
        "optimizer": "AdamW8bit",
        "scheduler": "constant",
        "epochs": 10,
        "save_every_n_epochs": 1,
        "mixed_precision": "bf16",
        "save_precision": "bf16",
        "gradient_checkpointing": True,
    },
    preview_sampler="euler",
    preview_cfg=4.5,
    preview_steps=6,
    preview_max_resolution=1024,
))
