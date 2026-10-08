from __future__ import annotations

from pydantic import BaseModel, Field


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    project_type: str = Field(default="character", pattern="^(character|style)$")


class ProjectOut(BaseModel):
    id: int
    name: str
    project_type: str
    status: str
    base_dir: str
    dataset_dir: str
    captions_dir: str
    outputs_dir: str
    library_dir: str


class ProjectUpdateIn(BaseModel):
    project_type: str = Field(pattern="^(character|style)$")


class ProjectDuplicateIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)


class CollectorScanIn(BaseModel):
    project_id: int
    url: str
    keyword: str = ""
    limit: int = Field(default=24, ge=1, le=200)


class CollectorImportIn(BaseModel):
    project_id: int
    selected_ids: list[int]
    naming_template: str = "{title}_{index}"
    import_dir: str = ""


class RepeatFolderIn(BaseModel):
    project_id: int
    repeats: int = Field(default=5, ge=1, le=1000)
    folder_title: str = Field(default="default", min_length=1, max_length=120)


class DropUrlIn(BaseModel):
    project_id: int
    url: str = Field(min_length=5, max_length=2000)


class CandidateRemoveIn(BaseModel):
    project_id: int
    candidate_ids: list[int] = Field(default_factory=list)


class PreviewExtraLora(BaseModel):
    name: str = Field(min_length=1,max_length=1000)
    strength: float = Field(default=1,ge=0,le=2)


class TrainingStartIn(BaseModel):
    allow_interactive_gpu_sharing: bool | None = None
    advanced: dict = Field(default_factory=dict)
    resume_checkpoint_id: int | None = None
    project_id: int
    training_goal: str = Field(
        default="character_identity",
        pattern="^(character_identity|character_style|art_style|costume_concept)$",
    )
    quality_preset: str = Field(default="balanced", pattern="^(fast|balanced|quality|custom)$")
    # Authorized GPU work waits on physical GPU1 by default. Callers may opt
    # out explicitly, but omission must never turn temporary occupancy into a
    # stopped workflow.
    preview_backend: str = Field(default="comfyui", pattern="^comfyui$")
    preview_lora_strength: float = Field(default=1.0, ge=0, le=2)
    preview_extra_loras: list[PreviewExtraLora] = Field(default_factory=list, max_length=8)
    preview_base_checkpoint_path: str = ""
    preview_steps: int | None = Field(default=None, ge=1, le=100)
    preview_cfg: float | None = Field(default=None, ge=0, le=30)
    training_memory_mode: str = Field(default="standard", pattern="^(low_vram|balanced|standard)$")
    preview_gpu: str = Field(default="gpu0", pattern="^(gpu0|gpu1)$")  # gpu0 = RTX 3060 beside training, gpu1 = 3090 Ti (training pauses)
    queue_if_busy: bool = True
    dataset_snapshot_id: int | None = None
    preview_profile_snapshot_id: int | None = None
    preset_id: int | None = None
    epochs: int = Field(default=5, ge=1, le=1000)
    repeats: int = Field(default=5, ge=1, le=1000)
    alpha: float = Field(default=4.0, ge=0.1, le=128.0)
    rank: int = Field(default=16, ge=1, le=512)
    save_every_n_epochs: int = Field(default=1, ge=1, le=1000)
    output_name: str = Field(default="lora_output", min_length=1, max_length=120)
    base_checkpoint_path: str = ""
    train_data_dir: str = ""
    reg_data_dir: str = ""
    resolution: int = Field(default=512, ge=256, le=2048)
    learning_rate: float = Field(default=1e-4, gt=0.0, le=1.0)
    train_batch_size: int = Field(default=2, ge=1, le=64)
    optimizer: str = Field(default="AdamW8bit", max_length=64)
    scheduler: str = Field(default="cosine_with_restarts", max_length=64)
    min_snr_gamma: int | None = Field(default=5, ge=0, le=20)
    mixed_precision: str = Field(default="bf16", max_length=16)
    save_precision: str = Field(default="fp16", max_length=16)
    xformers: bool = True
    cache_latents: bool = True
    cache_latents_to_disk: bool = False
    gradient_checkpointing: bool = True
    persistent_data_loader_workers: bool = True
    max_data_loader_n_workers: int = Field(default=4, ge=0, le=16)
    network_train_unet_only: bool = False
    model_family: str = Field(default="auto", max_length=32)
    # auto は ModelSpec の既定バックエンド。明示指定時のみ別エンジンへ送る。
    training_engine: str = Field(default="auto", pattern="^(auto|musubi|kohya|ai_toolkit)$")
    # "original"(既定): project.dataset_dir 直下を使用。
    # "processed": Dataset Builder Pipeline (Resize/Upscale/Cleanup/Caption) の
    # 完了済み成果物 (<dataset_dir>/processed/) を使用。存在確認は
    # ファイルの有無ではなく Pipeline 完了マニフェストで行う。
    dataset_source: str = Field(default="original", max_length=16)
    gpu_device_id: int = Field(default=1, ge=0, le=7)
    vae_path: str = ""
    text_encoder_path: str = ""


class TrainingControlIn(BaseModel):
    project_id: int


class GenerateTagsIn(BaseModel):
    project_id: int
    overwrite: bool = False
    general_thresh: float = Field(default=0.35, ge=0.05, le=0.95)
    character_thresh: float = Field(default=0.85, ge=0.05, le=0.99)
    remove_character_tags: bool = False


class CaptionItem(BaseModel):
    id: int
    file_path: str
    width: int
    height: int
    aspect: str
    caption: str
    caption_source: str


class CaptionEditIn(BaseModel):
    caption: str


class BatchReplaceIn(BaseModel):
    project_id: int
    find: str
    replace: str
    item_ids: list[int] = Field(default_factory=list)


class BatchRemoveIn(BaseModel):
    project_id: int
    tags: list[str]
    item_ids: list[int] = Field(default_factory=list)


class TaggerStatusOut(BaseModel):
    project_id: int
    status: str
    total: int
    done: int
    message: str


class PresetCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    payload: dict


class PresetUpdateIn(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    payload: dict | None = None


class ToolPathsIn(BaseModel):
    python_exe: str = ""
    kohya_root: str = ""
    musubi_root: str = ""
    ai_toolkit_root: str = ""
    comfyui_root: str = ""
    wd14_script: str = ""
    temp_dir: str = ""
    dataset_base_dir: str = ""
    pixiv_session: str = ""
    lora_output_dir: str = ""


class ToolPathsOut(BaseModel):
    python_exe: str
    kohya_root: str
    musubi_root: str = ""
    ai_toolkit_root: str = ""
    comfyui_root: str
    wd14_script: str
    temp_dir: str
    dataset_base_dir: str
    pixiv_session: str = ""
    lora_output_dir: str = ""


class PreviewPromptItem(BaseModel):
    label: str = ""
    quality: str = ""
    positive: str = ""
    negative: str = ""
    trigger_words: str = ""


class PreviewPromptsIn(BaseModel):
    positive_prompt: str = ""
    negative_prompt: str = ""
    prompts: list[PreviewPromptItem] | None = None
    preview_resolution: int = Field(default=1024, ge=256, le=2048)
    # sampler/cfg/steps はモデル世代(ModelSpec)ごとの既定値を使うのが基本。
    # 空文字/None は「自動（モデル既定）」を意味し、値を入れた場合のみ上書きする。
    preview_sampler: str = ""
    preview_cfg: float | None = Field(default=None, ge=0, le=30)
    preview_steps: int | None = Field(default=None, ge=1, le=150)
    # Preview Job Cardinality Contract: expected = len(prompts) * instances_per_prompt
    instances_per_prompt: int = Field(default=1, ge=1, le=8)


class PreviewPromptsOut(BaseModel):
    positive_prompt: str
    negative_prompt: str
    prompts: list[PreviewPromptItem] = Field(default_factory=list)
    preview_resolution: int = 1024
    preview_sampler: str = ""
    preview_cfg: float | None = None
    preview_steps: int | None = None
    instances_per_prompt: int = 1
    # Frontendに計算方法を一致させるため、Backend Contractの計算値をそのまま返す
    expected_preview_images: int = 1


class TagSettingsIn(BaseModel):
    prefix_tags: list[str] = Field(default_factory=list)
    block_words: list[str] = Field(default_factory=list)


class TagSettingsOut(BaseModel):
    project_id: int
    prefix_tags: list[str]
    block_words: list[str]


class ApplyPrefixIn(BaseModel):
    project_id: int
    prefix_tags: list[str]


class RemoveBlockWordsIn(BaseModel):
    project_id: int
    block_words: list[str]


class BasepipeConceptCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    concept_type: str = Field(default="identity", pattern="^(identity|outfit|style|hybrid|character|shared)$")
    trigger_token: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=2000)


class BasepipeAssetCreate(BaseModel):
    file_path: str = Field(min_length=1, max_length=4000)
    asset_key: str = Field(default="", max_length=240)
    asset_type: str = Field(default="image", max_length=40)
    origin_kind: str = Field(default="imported", max_length=40)
    source_ref: str = Field(default="", max_length=2000)
    caption: str = Field(default="", max_length=10000)
    caption_source: str = Field(default="", max_length=80)
    review_status: str = Field(default="pending", pattern="^(pending|approved|rejected|needs_review)$")
    user_rating: int | None = Field(default=None, ge=1, le=5)
    training_enabled: bool = True
    training_weight: float = Field(default=1.0, gt=0, le=10)
    metadata: dict = Field(default_factory=dict)


class BasepipeAssetReviewIn(BaseModel):
    review_status: str = Field(pattern="^(pending|approved|rejected|needs_review)$")
    caption: str | None = Field(default=None, max_length=10000)
    user_rating: int | None = Field(default=None, ge=1, le=5)
    training_enabled: bool | None = None
    training_weight: float | None = Field(default=None, gt=0, le=10)
    metadata: dict | None = None


class BasepipeAssetBulkReviewIn(BaseModel):
    asset_ids: list[int] = Field(min_length=1, max_length=500)
    review_status: str = Field(pattern="^(pending|approved|rejected|needs_review)$")
    training_enabled: bool | None = None


class BasepipeAssetGroupCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)


class BasepipeAssetGroupMembersIn(BaseModel):
    asset_ids: list[int] = Field(min_length=1, max_length=500)


class BasepipeDatasetBulkReviewIn(BaseModel):
    item_ids: list[int] = Field(min_length=1, max_length=500)
    review_status: str = Field(pattern="^(pending|approved|rejected|needs_review)$")


class BasepipeCaptionLineageUpdate(BaseModel):
    edited: str | None = Field(default=None, max_length=10000)
    processed: str | None = Field(default=None, max_length=10000)


class BasepipeCaptionBulkPreviewIn(BaseModel):
    asset_ids: list[int] = Field(default_factory=list, max_length=500)
    find: str = Field(min_length=1, max_length=500)
    replace: str = Field(default="", max_length=500)


class BasepipeCaptionBulkOperationIn(BaseModel):
    operation_id: int = Field(gt=0)


class BasepipeAssetTransformIn(BaseModel):
    prompt: str = Field(default="", max_length=4000)
    seed: int = Field(default=42, ge=0)
    gpu_device_id: int = Field(default=1, ge=0)
    execute: bool = False


class CharacterGenerationPreflightIn(BaseModel):
    asset_ids: list[int] = Field(default_factory=list, max_length=4)
    mode: str = Field(default="training", pattern="^(training|variation|viewpoint)$")
    gpu_device_id: int = Field(default=1, ge=0)
    reference_roles: dict[str, str] = Field(default_factory=dict)


class CharacterGenerationStartIn(BaseModel):
    """H3 generation request; execute is an explicit GPU-job gate."""
    asset_ids: list[int] = Field(default_factory=list, max_length=4)
    mode: str = Field(default="variation", pattern="^(variation|viewpoint)$")
    prompt: str = Field(default="", max_length=4000)
    seed: int = Field(default=42, ge=0)
    width: int = Field(default=768, ge=256, le=2048)
    height: int = Field(default=512, ge=256, le=2048)
    length: int = Field(default=56, ge=22, le=260)
    clip_count: int = Field(default=3, ge=1, le=6)
    coverage_preset: str = Field(default="balanced", pattern="^(turntable|essential|balanced|complete|custom)$")
    ref_image_size: str = Field(default="match", pattern="^(match|max)$")
    gpu_device_id: int = Field(default=1, ge=0)
    execute: bool = False
    queue_if_busy: bool = True
    identity_lock: dict[str, bool] = Field(default_factory=dict)
    reference_roles: dict[str, str] = Field(default_factory=dict)


class CharacterGenerationStageIn(BaseModel):
    """Advance one explicit Dataset Generation Run stage; never skip stages."""
    stage: str = Field(max_length=40)
    status: str = Field(default="completed", pattern="^(pending|running|completed|blocked)$")
    artifact_manifest: dict = Field(default_factory=dict)
    note: str = Field(default="", max_length=4000)


class CharacterGenerationFrameExtractionIn(BaseModel):
    fps: float = Field(default=2.0, gt=0, le=30)
    max_frames: int = Field(default=24, ge=1, le=240)
    blur_threshold: float = Field(default=18.0, ge=0)
    duplicate_distance: int = Field(default=6, ge=0, le=64)


class CharacterGenerationEnhancementIn(BaseModel):
    """Queue deterministic 8-yaw enhancement evidence on physical GPU1."""
    gpu_device_id: int = Field(default=1, ge=0)
    slot_count: int = Field(default=8, ge=8, le=8)
    total_frames: int = Field(default=124, ge=9, le=260)
    local_ranges: list[int] = Field(default_factory=lambda: [9, 11], min_length=2, max_length=2)
    include_realesrgan: bool = True
    queue_if_busy: bool = True


class CharacterGenerationRunAestheticReviewIn(BaseModel):
    """Persist the user's run-level aesthetic decision without agent inference."""
    decision: str = Field(pattern="^(approved|rejected)$")
    note: str = Field(default="", max_length=4000)
    human_confirmed: bool = False


class CharacterGenerationReviewDecision(BaseModel):
    asset_id: int = Field(gt=0)
    review_status: str = Field(pattern="^(approved|rejected)$")


class CharacterGenerationIdentityReviewIn(BaseModel):
    decisions: list[CharacterGenerationReviewDecision] = Field(min_length=1, max_length=240)
    note: str = Field(default="", max_length=4000)
    human_confirmed: bool = False


class CharacterGenerationQwenCorrectionIn(BaseModel):
    asset_ids: list[int] = Field(min_length=1, max_length=240)
    prompt: str = Field(default="preserve identity and correct image quality", max_length=4000)
    seed: int = Field(default=42, ge=0)
    gpu_device_id: int = Field(default=1, ge=0)
    execute: bool = False
    queue_if_busy: bool = True
    mode: str = Field(default="correction", pattern="^(correction|dataset_regeneration|consistency_regeneration)$")
    human_confirmed: bool = False


class CharacterGenerationCaptionIn(BaseModel):
    captions: list[dict] = Field(min_length=1, max_length=240)
    note: str = Field(default="", max_length=4000)


class BasepipeTrainingInputChoiceIn(BaseModel):
    choice: str = Field(pattern="^(original|qwen|enhanced)$")
    version_id: int | None = Field(default=None, gt=0)
    human_confirmed: bool = False
    note: str = Field(default="", max_length=4000)


class BasepipeAestheticApprovalIn(BaseModel):
    human_confirmed: bool = False
    note: str = Field(default="", max_length=4000)


class BasepipeEvaluationIn(BaseModel):
    checkpoint_id: int
    preview_slot: str = Field(default="", max_length=120)
    preview_profile_snapshot_id: int | None = None
    identity: int | None = Field(default=None, ge=1, le=5)
    outfit_separation: int | None = Field(default=None, ge=1, le=5)
    style_durability: int | None = Field(default=None, ge=1, le=5)
    style_quality: int | None = Field(default=None, ge=1, le=5)
    prompt_flexibility: int | None = Field(default=None, ge=1, le=5)
    artifact: int | None = Field(default=None, ge=1, le=5)
    accepted: bool = False
    note: str = Field(default="", max_length=4000)
    human_confirmed: bool = False


class BasepipeSnapshotCreate(BaseModel):
    name: str = Field(min_length=1, max_length=160)
    concept_id: int | None = None
    asset_ids: list[int] = Field(default_factory=list)
    dataset_item_ids: list[int] = Field(default_factory=list)
