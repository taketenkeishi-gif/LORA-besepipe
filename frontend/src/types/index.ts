export type TabId =
  | "dashboard"
  | "projects"
  | "dataset"
  | "training"
  | "library"
  | "integrations"
  | "guide";

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

export type TrainingStatus = {
  status: string;
  epoch: number;
  total_epochs: number;
  total_steps?: number;
  done_steps?: number;
  progress_percent?: number;
  eta_seconds?: number | null;
  steps_per_epoch?: number;
  stop_mode?: string | null;
  latest_checkpoint_path?: string | null;
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
  comfyui_root: string;
  wd14_script: string;
  temp_dir: string;
  dataset_base_dir: string;
};

export type IntegrationStatus = {
  checks: Record<string, { ok: boolean; reason: string }>;
};

export type PreviewSample = {
  checkpoint_id: number;
  epoch: number;
  step: number;
  mark: string;
  created_at: string;
  samples: Record<string, string>;
  sample_previews?: Record<string, string>;
};

export type PreviewPrompts = {
  positive_prompt: string;
  negative_prompt: string;
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

export type PresetPayload = {
  type: "character" | "style" | "hybrid" | "custom" | string;
  description: string;
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

// ── Phase 4: Training Mode ─────────────────────────────────────────────────

export type TrainingMode = {
  mode: "kohya" | "simulated";
  kohya_ready: boolean;
  kohya_root: string;
  train_script: string | null;
  message: string;
};

// ── Phase 5: Asset Library (§22-23) ───────────────────────────────────────

export type LoraAsset = {
  id: number;
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
  created_at: string;
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
  vram_total_mb: number;
  vram_pct: number;
  gpu_util_pct: number;
  temperature: number | null;
};

export type ResourceStats = {
  cpu_pct: number;
  ram_used_gb: number;
  ram_total_gb: number;
  ram_pct: number;
  gpu: GpuInfo[];
  gpu_available: boolean;
};
