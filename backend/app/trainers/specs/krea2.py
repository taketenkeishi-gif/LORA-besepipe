from ..base import CacheStep, ModelSpec, RequiredModel
from ..registry import register_spec

register_spec(ModelSpec(
    model_family="krea2",
    display_name="KREA2 (RAW MMDiT)",
    backend="musubi",
    train_script="krea2_train_network.py",
    network_module="networks.lora_krea2",
    default_resolution=1024,
    supports={"lora", "t2i"},
    checkpoint_patterns=["krea2", "krea_2", "krea-2"],
    # Preview専用: RAW(24.48GB, 単体でも24GB VRAM超過)ではなくTurbo(12GB前後)を使う。
    # RAW Trainingと同一GPUでRAW Previewを並列実行するとVRAM枯渇でTraining/ComfyUI
    # いずれかが強制終了するリスクが実運用で確認された。RAW学習LoRAはTurboへロード
    # 可能なことを実Runtime検証済み(23秒で実画像生成、RAW単体444秒より大幅高速)。
    preview_checkpoint_patterns=["krea2_turbo", "krea2-turbo", "krea_2_turbo"],
    required_models=[
        RequiredModel(
            key="vae",
            display_name="Qwen-Image VAE",
            auto_search_paths=[
                "models/vae/qwen_image_vae.safetensors",
                "models/diffusion_models/qwen_image_vae.safetensors",
                "models/vae/qwen-image-vae.safetensors",
            ],
            required=True,
            cfg_key="vae_path",
            toml_key="vae",
            cache_arg="--vae",
            hf_repo="Qwen/Qwen-Image",
            filename="qwen_image_vae.safetensors",
        ),
        RequiredModel(
            key="text_encoder",
            display_name="Qwen3-VL 4B テキストエンコーダ",
            auto_search_paths=[
                "models/text_encoders/qwen3vl_4b_bf16.safetensors",
                "models/text_encoders/qwen3vl_2b_bf16.safetensors",
                "models/text_encoders/qwen_te.safetensors",
                "models/clip/qwen3vl_4b_bf16.safetensors",
            ],
            required=False,
            cfg_key="text_encoder_path",
            cache_arg="--text_encoder",
            hf_repo="Qwen/Qwen3-VL-4B",
            filename="qwen3vl_4b_bf16.safetensors",
        ),
    ],
    cache_steps=[
        CacheStep(
            script="krea2_cache_text_encoder_outputs.py",
            model_key="text_encoder",
            extra_args=["--batch_size", "1"],
            label="テキストエンコーダキャッシュ生成",
        ),
        CacheStep(
            script="krea2_cache_latents.py",
            model_key="vae",
            label="潜在変数キャッシュ生成",
        ),
    ],
    preview_sampler="euler",
    preview_cfg=1.0,
    # 実測(2026-07-05, RTX 3090 Ti 24GB, ComfyUI --highvram --fp32-vae, cudaMallocAsync):
    # base checkpoint(Krea2_raw.safetensors)自体が26.28GBあり単体でも24GB VRAMを超えるため、
    # UNETLoaderのweight_dtype=defaultで20steps/1024pxを実行するとKSampler時に
    # torch.OutOfMemoryError("Currently allocated: 47.97 GiB")で確実に失敗する。
    # weight_dtype=fp8_e4m3fn(preview_unet_weight_dtype)でのVRAM圧縮が必須。
    # 20steps/1024pxのままではfp8化してもAmpereにネイティブfp8演算がなくdequant
    # オーバーヘッドで15分超が経過しても完了しなかった。10steps/512pxへ縮小したところ、
    # ピークVRAM約19GBで7分24秒(444.4秒)で完走し実画像(401KB PNG)を確認した。
    preview_steps=10,
    preview_unet_weight_dtype="fp8_e4m3fn",
    preview_max_resolution=512,
    extra_toml={
        # KREA2 RAW DiT は 24GB GPU でも収まるよう fp8 + block swap を有効化
        # 注意: fp8_base と fp8_scaled は必ずセットで指定（片方だけだと ValueError）
        "sdpa": True,
        "timestep_sampling": "shift",
        "weighting_scheme": "none",
        "discrete_flow_shift": 2.5,
        "max_data_loader_n_workers": 2,
        "persistent_data_loader_workers": True,
        "fp8_base": True,
        "fp8_scaled": True,
        "blocks_to_swap": 20,
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
