export type TabId =
  | "home"
  | "overview"
  | "dataset"
  | "training"
  | "runs"
  | "compare"
  | "library"
  | "integrations"
  | "guide"
  | "characterFactory"
  | "styleCurator"
  | "operations";

export type Project = {
  id: number;
  name: string;
  project_type: "character" | "style";
  status: string;
  dataset_dir: string;
  captions_dir: string;
  library_dir: string;
  outputs_dir: string;
};

export type PreviewActivityJob = {
  id: number;
  checkpoint_id: number;
  epoch: number;
  prompt_index: number;
  instance_index: number;
  status: "pending" | "running" | "succeeded" | "failed" | string;
  error: string;
  attempts: number;
  step?: number | null;
  step_max?: number | null;
  elapsed_seconds?: number | null;
};

export type PreviewActivity = {
  active: boolean;
  run_id?: number;
  phase: "none" | "generating" | "window_waiting" | "deferred_until_training_ends" | "waiting_gpu" | string;
  epoch?: number;
  total?: number;
  done?: number;
  failed?: number;
  running?: number;
  pending?: number;
  current_job?: {
    id: number;
    prompt_index: number;
    instance_index: number;
    elapsed_seconds: number | null;
    step: number | null;
    step_max: number | null;
    collector_status: string | null;
  } | null;
  training_paused?: boolean;
  window?: { epoch: number; step: number | null; elapsed_seconds: number | null; remaining_seconds: number | null } | null;
  waiting_reason?: string;
  last_activity_age_seconds?: number | null;
  stalled?: boolean;
  stall_after_seconds?: number;
  generated_at?: number;
  jobs: PreviewActivityJob[];
};

export type TrainingStatus = {
  preview_activity?: PreviewActivity;
  allow_interactive_gpu_sharing?: boolean;
  status: string;
  run_id?: number | null;
  queue_position?: number | null;
  epoch: number;
  step?: number;
  total_epochs: number;
  total_steps?: number;
  done_steps?: number;
  progress_percent?: number;
  eta_seconds?: number | null;
  steps_per_epoch?: number;
  process_alive?: boolean;
  stop_mode?: string | null;
  latest_checkpoint_path?: string | null;
  loss?: number | null;
  mode?: string;
  log_path?: string | null;
  started_at?: string;
  updated_at?: string;
  message?: string;
  failure?: { code?: string | null; message?: string | null; stage?: string | null } | null;
  manifest_path?: string | null;
  dataset_snapshot_id?: number | null;
  current_stage?: string;
};

export type RunSummary = {
  id: number;
  project_id: number;
  status: string;
  current_epoch: number;
  current_step: number;
  total_epochs: number;
  started_at?: string;
  updated_at?: string;
  failure_code?: string | null;
  failure_message?: string | null;
  manifest_path?: string | null;
  dataset_snapshot_id?: number | null;
  current_stage?: string;
};

export type RunEvidence = {
  run_id: number;
  status: string;
  training_status?: string;
  checkpoints?: Array<{
    id: number;
    epoch?: number;
    step?: number;
    file_path?: string | null;
    validation_status?: string | null;
    validation_detail?: string | null;
    preview?: { expected: number; succeeded: number; failed: number; pending: number; running: number };
  }>;
  preview?: { expected: number; succeeded: number; failed: number; pending: number; running: number };
  manifest: {
    engine?: string;
    model_family?: string;
    generated_configs?: string[];
    dataset?: { source?: string; item_count?: number; caption_count?: number };
    [key: string]: unknown;
  } | null;
  requested: Record<string, unknown>;
  resolved: Record<string, unknown>;
  observed: Record<string, unknown>;
  parameters: Array<{
    key: string;
    requested: unknown;
    resolved: unknown;
    observed: unknown;
    observed_source: string;
    mismatch: boolean;
  }>;
  failure: { code?: string | null; message?: string | null; stage?: string | null; at?: string | null; exit_code?: number | null; last_epoch?: number | null; last_step?: number | null; latest_checkpoint?: { id: number; file_path: string } | null; resume_available?: boolean; log_excerpt?: string } | null;
  current_stage?: string;
  stage_history?: Array<{ stage: string; status: string; detail?: string }>;
  timing?: {
    reference_setup_seconds?: number | null;
    dataset_generation_seconds?: number | null;
    human_review_seconds?: number | null;
    preprocess_seconds?: number | null;
    cache_seconds?: number | null;
    time_to_first_checkpoint_seconds?: number | null;
    time_to_first_preview_seconds?: number | null;
    total_training_seconds?: number | null;
    total_user_interaction_seconds?: number | null;
  };
};

export type RunDetail = {
  run: RunSummary;
  status: TrainingStatus;
  evidence: RunEvidence;
};

export type TrainingQueueEntry = {
  run_id: number;
  project_id: number;
  queue_position: number;
  queued_at?: string;
};

export type TrainingQueueState = {
  running: { run_id: number; project_id: number } | null;
  queued: TrainingQueueEntry[];
  queue_length: number;
};

export type ScanItem = {
  id: number;
  title: string;
  width: number;
  height: number;
  aspect: "portrait" | "landscape" | "square" | string;
  thumbnail_url?: string;
  tags?: string[];
  source_url?: string;
  unavailable?: boolean;
};

export type SortKey = "id" | "area" | "width" | "height" | "title" | "aspect";
export type SortDir = "asc" | "desc";

export type ScanMode =
  | "url_live"
  | "url_unavailable"
  | "dataset_base"
  | "mock_url";

export type ScanResult = {
  items: ScanItem[];
  detected: number;
  mode: ScanMode;
  message: string;
};

export type ToolPaths = {
  python_exe: string;
  kohya_root: string;
  musubi_root: string;
  ai_toolkit_root: string;
  comfyui_root: string;
  wd14_script: string;
  temp_dir: string;
  dataset_base_dir: string;
  pixiv_session: string;
  lora_output_dir: string;
};

export type IntegrationStatus = {
  checks: Record<string, { ok: boolean; reason: string }>;
};

export type SystemStatus = {
  api: { state: string; detail: string };
  services: Record<string, { state: string; detail: string }>;
  gpu: {
    state: string;
    devices: Array<{ index: number; name: string; vram_pct: number }>;
    target_physical_index?: number;
    target_device?: string | null;
    target_free_vram_mb?: number | null;
    target_min_free_vram_mb?: number;
    target_state?: string;
  };
};

export type PreviewJobSummary = {
  expected: number;
  pending: number;
  running: number;
  succeeded: number;
  failed: number;
};

export type PreviewJob = {
  id: number;
  prompt_index: number;
  instance_index: number;
  seed: number;
  status: "pending" | "running" | "succeeded" | "failed";
  error_detail: string;
  output_path: string;
  preview_model_family: string;
  attempts: number;
  conditions?: {
    model_family?: string;
    resolution?: number;
    sampler?: string;
    scheduler?: string;
    cfg?: number;
    steps?: number;
    source?: string;
  };
  preview_snapshot?: {
    profile_snapshot_id?: number | null;
    profile_snapshot_hash?: string | null;
    prompt?: string;
    negative_prompt?: string;
    seed?: number;
    width?: number;
    height?: number;
    steps?: number;
    cfg?: number;
    sampler?: string;
    scheduler?: string;
    checkpoint_id?: number;
    run_id?: number;
    source?: string;
  };
};

export type PreviewJobsResponse = {
  checkpoint_id: number;
  summary: PreviewJobSummary;
  jobs: PreviewJob[];
};

export type PreviewSample = {
  preview_model_path?: string|null;
  preview_conditions?: {steps?:number;cfg?:number;source?:string;lora_strength?:number;backend?:string;extra_loras?:{name:string;strength:number}[]};
  checkpoint_id: number;
  epoch: number;
  step: number;
  mark: string;
  created_at: string;
  samples: Record<string, string>;
  sample_previews?: Record<string, string>;
  validation_status?: string;
  validation_detail?: string;
  preview_job_summary?: PreviewJobSummary;
};

export type PreviewProfile = {
  id: number;
  project_id: number | null;
  name: string;
  prompt: string;
  negative_prompt: string;
  seed: number;
  resolution: number;
  steps: number;
  cfg: number;
  sampler: string;
  scheduler?: string;
  model_family: string;
};

export type PreviewPromptItem = {
  label: string;
  quality?: string;
  positive: string;
  negative: string;
  trigger_words?: string;
};

export type PreviewPrompts = {
  positive_prompt: string;
  negative_prompt: string;
  prompts?: PreviewPromptItem[];
  preview_resolution?: number;
  // 空文字/null = モデル世代(spec)の既定値を自動使用。値を入れた場合のみ上書き。
  preview_sampler?: string;
  preview_cfg?: number | null;
  preview_steps?: number | null;
  // Preview Job Cardinality Contract: expected_preview_images = prompts.length * instances_per_prompt
  instances_per_prompt?: number;
  expected_preview_images?: number;
};

export type DatasetPreview = {
  image_path: string | null;
  thumbnail_url: string | null;
};

export type CaptionItem = {
  id: number;
  file_path: string;
  width: number;
  height: number;
  aspect: string;
  caption: string;
  caption_source: string;
};

export type TaggerStatus = {
  project_id: number;
  status: string;
  total: number;
  done: number;
  message: string;
};

export type DatasetStats = {
  total: number;
  captioned: number;
  caption_rate: number;
  mono_count: number;
  color_count: number;
  avg_width: number;
  avg_height: number;
  resolution: { low: number; medium: number; high: number };
  aspect: { portrait: number; landscape: number; square: number };
  duplicate_pairs: number;
  similar_pairs: number;
  quality_score: number;
  warnings: string[];
  imagehash_available?: boolean;
};

export type SimilarityGroup = {
  type: "duplicate" | "similar";
  item_ids: number[];
};

export type DatasetAnalysis = {
  project_id: number;
  analyzed: boolean;
  quality_score: number | null;
  stats: DatasetStats;
  similarity_groups: SimilarityGroup[];
  analyzed_at: string | null;
};

export type LeakItem = {
  feature: string;
  dominant: string;
  ratio: number;
  severity: "high" | "medium";
};

export type CharacterTagLeak = {
  tag: string;
  count: number;
  ratio: number;
  severity: "high" | "medium";
};

export type CharacterLeakResult = {
  project_id: number;
  total: number;
  captioned: number;
  risk_level: "low" | "medium" | "high" | "unknown";
  risk_score: number;
  leaks: LeakItem[];
  character_tags: CharacterTagLeak[];
  warnings: string[];
};

export type CategoryTopTag = {
  tag: string;
  count: number;
};

export type CategoryDetail = {
  category: string;
  items_with_tags: number;
  coverage_pct: number;
  top_tags: CategoryTopTag[];
};

export type TagCategoriesResult = {
  project_id: number;
  total_items: number;
  categories: CategoryDetail[];
};

// ── Phase 4: Training Profiles ─────────────────────────────────────────────

export type TrainingPerfParams = {
  mixed_precision?: 'fp16' | 'bf16' | 'no';
  save_precision?: 'fp16' | 'bf16';
  xformers?: boolean;
  cache_latents?: boolean;
  cache_latents_to_disk?: boolean;
  gradient_checkpointing?: boolean;
  persistent_data_loader_workers?: boolean;
  max_data_loader_n_workers?: number;
  network_train_unet_only?: boolean;
};

export type PresetPayload = {
  type: "character" | "style" | "hybrid" | "custom" | string;
  description?: string;
  rank: number;
  alpha: number;
  repeats: number;
  epochs: number;
  resolution: number;
  optimizer: string;
  scheduler: string;
  save_every_n_epochs: number;
  min_snr_gamma: number | null;
  output_name: string;
  learning_rate?: number;
  train_batch_size?: number;
  xformers?: boolean;
  cache_latents?: boolean;
  cache_latents_to_disk?: boolean;
  gradient_checkpointing?: boolean;
  mixed_precision?: string;
  save_precision?: string;
  persistent_data_loader_workers?: boolean;
  max_data_loader_n_workers?: number;
  network_train_unet_only?: boolean;
};

export type Preset = {
  id: number;
  name: string;
  payload: PresetPayload;
  created_at: string;
};

// ── Phase 4: Human Review Gate ─────────────────────────────────────────────

export type DatasetReport = {
  project_id: number;
  project_name: string;
  project_type: string;
  ready: boolean;
  quality_score: number | null;
  total_images: number;
  caption_rate: number;
  warnings: string[];
  blockers: string[];
  leak_risk: "low" | "medium" | "high" | "unknown";
  leak_score: number;
  duplicate_pairs: number;
  similar_pairs: number;
  analyzed: boolean;
};

// ── 学習開始前チェック(Preflight) ────────────────────────────────────────

export type PreflightCheck = {
  level: "ok" | "warning" | "blocked";
  label: string;
  detail: string;
  action: string;
};

export type PreflightResult = {
  project_id: number;
  model_family: string;
  train_data_dir: string;
  overall: "ok" | "warning" | "blocked";
  checks: PreflightCheck[];
  image_count: number;
  captioned_count: number;
  can_start?: boolean;
  can_queue?: boolean;
  requested?: Record<string, unknown>;
  resolved?: Record<string, unknown>;
  environment?: Record<string, unknown>;
};

// ── 学習時間予測 ───────────────────────────────────────────────────────────

export type TrainingEstimate = {
  project_id: number;
  images: number;
  batch_size: number;
  steps_per_epoch: number;
  total_steps: number;
  epochs: number;
  is_sdxl: boolean;
  model_family?: string;
  effective_resolution: number;
  gpu_name: string | null;
  mode: string;
  sec_per_step: number;
  sec_per_epoch: number;
  overhead_seconds: number;
  eta_seconds: number;
  remaining_seconds?: number | null;
  done_steps?: number | null;
  source: "live" | "calibrated" | "heuristic" | "simulation";
  confidence: "high" | "medium" | "low";
  ready: boolean;
  message: string;
};

// ── Phase 4: Training Mode ─────────────────────────────────────────────────

export type TrainingMode = {
  mode: "kohya" | "musubi" | "both" | "ai_toolkit" | "simulated";
  kohya_ready: boolean;
  musubi_ready: boolean;
  anima_ready: boolean;
  kohya_root: string;
  musubi_root: string;
  ai_toolkit_root: string;
  train_script: string | null;
  reason: string;
  message: string;
};

// ── Phase 5: Asset Library (§22-23) ───────────────────────────────────────

// -- Phase 2 (architecture): model-specs 駆動 UI --

export type ModelSpecRequiredModel = {
  key: string;
  display_name: string;
  required: boolean;
  hf_repo: string;
  filename: string;
};

export type ModelSpec = {
  model_family: string;
  display_name: string;
  backend: string;
  training_backend: string;
  preview_backend: string;
  supports: string[];
  default_resolution: number;
  default_preset: string;
  required_models: ModelSpecRequiredModel[];
  has_cache_steps: boolean;
};

export type ModelSpecsResult = {
  specs: ModelSpec[];
  count: number;
};


export type LoraAsset = {
  id: number;
  training_run_id?: number | null;
  dataset_snapshot_id?: number | null;
  project_id: number | null;
  name: string;
  lora_path: string;
  base_model: string;
  dataset_size: number;
  profile_name: string;
  tags: string[];
  notes: string;
  preview_path: string;
  training_config: Record<string, unknown>;
  quality_score: number | null;
  asset_type: string;
  library_status?: "candidate" | "accepted" | string;
  created_at: string;
  lineage?: {
    source_run_id?: number | null;
    source_snapshot_id?: number | null;
    source_snapshot_hash?: string | null;
    source_snapshot_item_count?: number | null;
    source_asset_ids?: number[];
    source_asset_keys?: string[];
    source_checkpoint_id?: number | null;
    checkpoint_validation_status?: string | null;
    model_family?: string | null;
    rank?: number | null;
    alpha?: number | null;
    trigger_token?: string | null;
    preview_profile_snapshot_id?: number | null;
    preview_profile_snapshot_hash?: string | null;
    preview_count?: number;
    preview_success_count?: number;
    preview_condition_mismatch_count?: number;
    export_path?: string | null;
    export_sha256?: string | null;
    source_run_status?: string | null;
    lineage_gate?: {
      run_completed?: boolean;
      snapshot_sealed?: boolean;
      checkpoint_present?: boolean;
      source_artifact_match?: boolean;
      preview_complete?: boolean;
      preview_conditions_match?: boolean;
      human_evaluation?: boolean;
      final_selected?: boolean;
      exported?: boolean;
    };
  };
};

export type LoraAssetPayload = {
  project_id?: number | null;
  name: string;
  lora_path?: string;
  base_model?: string;
  dataset_size?: number;
  profile_name?: string;
  tags?: string[];
  notes?: string;
  preview_path?: string;
  training_config?: Record<string, unknown>;
  quality_score?: number | null;
  asset_type?: string;
};

// ── Phase 5: Distribution Analysis (§9.2-9.4) ─────────────────────────────

export type DistributionItem = {
  label: string;
  count: number;
  pct: number;
};

export type DistributionData = {
  project_id: number;
  total_items: number;
  captioned_items: number;
  hair_color: DistributionItem[];
  hair_style: DistributionItem[];
  eye_color: DistributionItem[];
  costume: DistributionItem[];
};

// ── Phase 5: Resource Monitor (§18) ───────────────────────────────────────

export type GpuInfo = {
  index: number;
  name: string;
  vram_used_mb: number;
  vram_free_mb?: number;
  vram_total_mb: number;
  vram_pct: number;
  gpu_util_pct: number;
  temperature: number | null;
  occupancy_status?: "blocked" | "clear" | "unknown";
  foreign_processes?: { pid: number; process_name: string }[];
};

export type ResourceStats = {
  cpu_pct: number;
  ram_used_gb: number;
  ram_total_gb: number;
  ram_pct: number;
  gpu: GpuInfo[];
  gpu_available: boolean;
};

export type SuggestedImage = {
  source: "pixiv" | "bing" | "google" | "duckduckgo" | "pinterest" | string;
  id: string;
  title: string;
  url: string;
  user?: string;
  width?: number;
  height?: number;
  like_count?: number;
  view_count?: number;
  score?: number;
  clip_similarity?: number;
  user_selected?: boolean;
  llm_evaluation?: { score: number; reason?: string };
};

export type SuggestionsResult = {
  project_id: number;
  status: "completed" | "running" | "failed";
  timestamp?: string;
  results: SuggestedImage[];
  total_count?: number;
  evaluation_mode?: string;
  feedback_round?: number;
  source_breakdown?: {
    pixiv: number;
    bing?: number;
    duckduckgo?: number;
    google: number;
    pinterest: number;
  };
  error?: string;
};

export type SuggestionFeedbackPayload = {
  accepted_urls: string[];
  rejected_urls: string[];
  evaluation_mode: "fast" | "balanced" | "accurate";
};

// ── Pixiv R18 Authentication ──────────────────────────────────────────────

export type PixivSessionStatus = {
  is_logged_in: boolean;
  is_valid: boolean;
  user_id: string | null;
  user_name: string | null;
  expires_at: string | null;
  last_validated: string | null;
  auto_refresh_interval: string;
  message: string;
};

export type PixivLoginStartResult = {
  success: boolean;
  session_id: string | null;
  popup_url: string | null;
  already_logged_in?: boolean;
  user_id?: string;
  user_name?: string;
  message: string;
};

export type PixivRefreshResult = {
  success: boolean;
  refreshed: boolean;
  message: string;
};

// ── Dataset Mixer (§10) ────────────────────────────────────────────────────

export type MixerFeatureKey =
  | "face" | "expression" | "hair" | "costume" | "accessory"
  | "background" | "composition" | "line" | "color" | "lighting" | "mood";

export type MixerWeights = Record<MixerFeatureKey, number>;

export type DatasetMixer = {
  project_id: number;
  feature_weights: MixerWeights;
  updated_at: string | null;
};

// ── Tag Settings (prefix / block words) ──────────────────────────────────────

export type TagSettings = {
  project_id: number;
  prefix_tags: string[];
  block_words: string[];
};

// ── Profile Classification (§outfit-sort) ──────────────────────────────────

export type OutfitProfile = {
  id: string;
  name: string;
  trigger_word: string;
  ref_item_ids: number[];
};

export type ClassifiedItem = {
  id: number;
  file_path: string;
  caption: string;
  score: number;
  thumbnail_url: string;
  excluded?: boolean;
};

export type ClassificationResult = {
  profiles: { name: string; trigger_word: string; items: ClassifiedItem[] }[];
  unclassified: ClassifiedItem[];
  total: number;
};
