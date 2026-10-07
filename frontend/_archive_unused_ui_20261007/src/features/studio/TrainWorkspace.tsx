import { useEffect, useMemo, useState } from "react";
import { AlertTriangle, CheckCircle2, ChevronDown, Clock3, Cpu, Gauge, Loader2, Play, Save, ShieldAlert, Sparkles } from "lucide-react";
import { apiGet, apiPost } from "../../lib/api";
import type { ModelSpec, PreflightResult, Project, ResourceStats, SystemStatus, TrainingEstimate } from "../../types";

type Draft = {
  training_goal: TrainingGoal;
  quality_preset: PresetId;
  model_family: string;
  training_engine: string;
  base_checkpoint_path: string;
  dataset_source: "original" | "processed";
  dataset_snapshot_id: number | null;
  preview_profile_snapshot_id: number | null;
  gpu_device_id: number;
  rank: number;
  alpha: number;
  learning_rate: number;
  train_batch_size: number;
  epochs: number;
  repeats: number;
  resolution: number;
  output_name: string;
  optimizer: string;
  scheduler: string;
  gradient_checkpointing: boolean;
  cache_latents: boolean;
};
type Snapshot = { id: number; name: string; item_count: number; snapshot_hash: string; status: string; created_at?: string };
type PreviewProfileSnapshot = { id: number; name: string; snapshot_hash: string; created_at?: string };
type PresetId = "fast" | "balanced" | "quality" | "custom";
type TrainingGoal = "character_identity" | "character_style" | "art_style" | "costume_concept";
const ANIMA_PHYSICAL_GPU_INDEX = 1;

const trainingGoals: Array<{ id: TrainingGoal; label: string }> = [
  { id: "character_identity", label: "Character identity" },
  { id: "character_style", label: "Character style" },
  { id: "art_style", label: "Art style" },
  { id: "costume_concept", label: "Costume / Concept" },
];
const trainingGoalLabel = (goal: TrainingGoal) => trainingGoals.find((item) => item.id === goal)?.label ?? goal;

const defaultDraft: Draft = {
  training_goal: "character_identity",
  quality_preset: "balanced",
  model_family: "anima", training_engine: "auto", base_checkpoint_path: "", dataset_source: "original",
  dataset_snapshot_id: null, preview_profile_snapshot_id: null, gpu_device_id: ANIMA_PHYSICAL_GPU_INDEX, rank: 16, alpha: 1, learning_rate: 0.0001,
  train_batch_size: 1, epochs: 6, repeats: 3, resolution: 1024, output_name: "anima_lora",
  optimizer: "AdamW8bit", scheduler: "constant", gradient_checkpointing: true, cache_latents: true,
};
const presets: Record<Exclude<PresetId, "custom">, Partial<Draft> & { label: string; hint: string }> = {
  fast: { label: "Fast test", hint: "構成確認用", epochs: 1, repeats: 1, rank: 4, alpha: 4, resolution: 512 },
  balanced: { label: "Balanced", hint: "最初の本番向け", epochs: 6, repeats: 3, rank: 16, alpha: 1, resolution: 1024 },
  quality: { label: "High quality", hint: "比較素材を多く生成", epochs: 10, repeats: 5, rank: 32, alpha: 1, resolution: 1024 },
};

type Props = {
  project: Project;
  systemStatus: SystemStatus | null;
  showError: (message: string) => void;
  showNotice: (message: string) => void;
  onStarted: () => Promise<void>;
  onOpenRuns: () => void;
};

export default function TrainWorkspace({ project, systemStatus, showError, showNotice, onStarted, onOpenRuns }: Props) {
  const [draft, setDraft] = useState<Draft>(defaultDraft);
  const [loaded, setLoaded] = useState(false);
  const [snapshots, setSnapshots] = useState<Snapshot[]>([]);
  const [previewSnapshots, setPreviewSnapshots] = useState<PreviewProfileSnapshot[]>([]);
  const [specs, setSpecs] = useState<ModelSpec[]>([]);
  const [checkpoints, setCheckpoints] = useState<Array<{ name: string; path: string }>>([]);
  const [resources, setResources] = useState<ResourceStats | null>(null);
  const [estimate, setEstimate] = useState<TrainingEstimate | null>(null);
  const [preflight, setPreflight] = useState<PreflightResult | null>(null);
  const [preflightDraft, setPreflightDraft] = useState<Draft | null>(null);
  const [saving, setSaving] = useState(false);
  const [savedAt, setSavedAt] = useState<string | null>(null);
  const [checking, setChecking] = useState(false);
  const [starting, setStarting] = useState(false);

  const targetGpuIndex = ANIMA_PHYSICAL_GPU_INDEX;
  const targetGpu = resources?.gpu.find((gpu) => gpu.index === targetGpuIndex) ?? resources?.gpu.find((gpu) => gpu.index === draft.gpu_device_id);
  const freeVram = targetGpu
    ? targetGpu.vram_free_mb ?? systemStatus?.gpu.target_free_vram_mb ?? targetGpu.vram_total_mb - targetGpu.vram_used_mb
    : null;
  const gpuBlocked = targetGpu?.occupancy_status === "blocked";
  const selectedSnapshot = snapshots.find((snapshot) => snapshot.id === draft.dataset_snapshot_id);
  const selectedPreviewSnapshot = previewSnapshots.find((snapshot) => snapshot.id === draft.preview_profile_snapshot_id);
  const selectedSpec = specs.find((spec) => spec.model_family === draft.model_family);
  const configurationErrors = useMemo(() => {
    const errors: string[] = [];
    if (draft.dataset_snapshot_id == null) errors.push("Dataset Snapshotを選択してください");
    if (draft.preview_profile_snapshot_id == null) errors.push("固定Preview条件を選択してください");
    if (!draft.base_checkpoint_path.trim()) errors.push("Anima Base Modelを選択してください");
    if (!draft.output_name.trim()) errors.push("Output nameを入力してください");
    if (draft.epochs < 1 || draft.epochs > 1000) errors.push("Epochsは1〜1000です");
    if (draft.repeats < 1 || draft.repeats > 1000) errors.push("Repeatsは1〜1000です");
    if (draft.resolution < 256 || draft.resolution > 2048 || draft.resolution % 64 !== 0) errors.push("Resolutionは256〜2048の64刻みです");
    if (draft.learning_rate <= 0 || draft.learning_rate > 1) errors.push("Learning rateは0より大きく1以下です");
    if (draft.rank < 1 || draft.rank > 512) errors.push("Rankは1〜512です");
    if (draft.alpha < 0.1 || draft.alpha > 128) errors.push("Alphaは0.1〜128です");
    return errors;
  }, [draft]);
  const selectableCheckpoints = draft.model_family === "anima"
    ? checkpoints.filter((checkpoint) => checkpoint.name.toLowerCase() === "anima-base-v1.0.safetensors")
    : checkpoints;
  const resolvedEvidence = preflight?.resolved as {
    base_checkpoint_path?: string;
    dependencies?: { text_encoder_path?: string; vae_path?: string };
    trainer_script_path?: string;
    python_executable?: string;
    cuda_visible_devices?: string;
    preview_profile_snapshot_id?: number;
    preview_profile?: { id?: number; name?: string; snapshot_hash?: string };
  } | undefined;

  useEffect(() => {
    setLoaded(false);
    setPreflight(null);
    setPreflightDraft(null);
    void Promise.all([
      apiGet<{ config: Partial<Draft> }>(`/training/config-draft/${project.id}`),
      apiGet<{ snapshots: Snapshot[] }>(`/basepipe/projects/${project.id}/workspace`),
      apiGet<{ snapshots: PreviewProfileSnapshot[] }>(`/basepipe/projects/${project.id}/preview-profile-snapshots`),
      apiGet<{ specs: ModelSpec[] }>("/training/model-specs"),
      apiGet<{ checkpoints: Array<{ name: string; path: string }> }>("/training/checkpoints"),
      apiGet<ResourceStats>("/training/resources"),
    ]).then(([saved, workspace, previewProfiles, modelSpecs, modelCheckpoints, resourceStats]) => {
      const nextSnapshots = (workspace.snapshots ?? []).filter((snapshot) => snapshot.status === "sealed");
      const nextPreviewSnapshots = previewProfiles.snapshots ?? [];
      const savedSnapshotId = saved.config.dataset_snapshot_id;
      const selectedSnapshotId = savedSnapshotId != null && nextSnapshots.some((snapshot) => snapshot.id === savedSnapshotId)
        ? savedSnapshotId
        : null;
      const savedPreviewSnapshotId = saved.config.preview_profile_snapshot_id;
      const selectedPreviewSnapshotId = selectedSnapshotId != null && savedPreviewSnapshotId != null
        && nextPreviewSnapshots.some((snapshot) => snapshot.id === savedPreviewSnapshotId) ? savedPreviewSnapshotId : null;
      const next: Draft = {
        ...defaultDraft,
        training_goal: project.project_type === "style" ? "art_style" : "character_identity",
        ...saved.config,
        model_family: "anima",
        training_engine: "auto",
        dataset_source: "original",
        // Anima is pinned to the physical RTX 3090 Ti. Never persist a stale
        // GPU0 draft during the initial render before system status arrives.
        gpu_device_id: ANIMA_PHYSICAL_GPU_INDEX,
        dataset_snapshot_id: selectedSnapshotId,
        preview_profile_snapshot_id: selectedPreviewSnapshotId,
      };
      const preferred = modelCheckpoints.checkpoints?.find((item) => item.name.toLowerCase() === "anima-base-v1.0.safetensors")
        ?? modelCheckpoints.checkpoints?.find((item) => item.name.toLowerCase().includes("anima"));
      if (preferred && (!next.base_checkpoint_path || (next.model_family === "anima" && !/anima-base-v1\.0\.safetensors$/i.test(next.base_checkpoint_path)))) {
        next.base_checkpoint_path = preferred.path;
      }
      setDraft(next);
      setSnapshots(nextSnapshots);
      setPreviewSnapshots(nextPreviewSnapshots);
      setSpecs(modelSpecs.specs ?? []);
      setCheckpoints(modelCheckpoints.checkpoints ?? []);
      setResources(resourceStats);
      setLoaded(true);
    }).catch((error) => showError(error instanceof Error ? error.message : "学習設定の取得に失敗しました"));
  }, [project.id, systemStatus?.gpu?.target_physical_index]);

  useEffect(() => {
    if (!loaded) return;
    const refreshResources = () => {
      void apiGet<ResourceStats>("/training/resources").then((next) => {
        setResources((current) => {
          const currentGpu = current?.gpu.find((gpu) => gpu.index === targetGpuIndex);
          const nextGpu = next.gpu.find((gpu) => gpu.index === targetGpuIndex);
          const changed = currentGpu?.vram_free_mb !== nextGpu?.vram_free_mb
            || currentGpu?.occupancy_status !== nextGpu?.occupancy_status
            || JSON.stringify(currentGpu?.foreign_processes ?? []) !== JSON.stringify(nextGpu?.foreign_processes ?? []);
          if (changed) { setPreflight(null); setPreflightDraft(null); }
          return next;
        });
      }).catch(() => undefined);
    };
    const timer = window.setInterval(refreshResources, 15_000);
    return () => window.clearInterval(timer);
  }, [loaded, targetGpuIndex]);

  useEffect(() => {
    if (!loaded) return;
    const timer = window.setTimeout(() => {
      setSaving(true);
      void apiPost(`/training/config-draft/${project.id}`, draft)
        .then(() => setSavedAt(new Date().toLocaleTimeString("ja-JP", { hour: "2-digit", minute: "2-digit" })))
        .catch((error) => showError(error instanceof Error ? error.message : "下書き保存に失敗しました"))
        .finally(() => setSaving(false));
    }, 650);
    return () => window.clearTimeout(timer);
  }, [draft, loaded, project.id]);

  useEffect(() => {
    if (!loaded) return;
    const query = new URLSearchParams({
      project_id: String(project.id), epochs: String(draft.epochs), repeats: String(draft.repeats), resolution: String(draft.resolution),
      batch_size: String(draft.train_batch_size), base_checkpoint_path: draft.base_checkpoint_path, optimizer: draft.optimizer,
      rank: String(draft.rank), gradient_checkpointing: String(draft.gradient_checkpointing), mixed_precision: "bf16",
      gpu_device_id: String(draft.gpu_device_id), model_family: draft.model_family,
    });
    if (draft.dataset_snapshot_id != null) query.set("dataset_snapshot_id", String(draft.dataset_snapshot_id));
    const timer = window.setTimeout(() => void apiGet<TrainingEstimate>(`/training/estimate?${query}`).then(setEstimate).catch(() => setEstimate(null)), 300);
    return () => window.clearTimeout(timer);
  }, [loaded, project.id, draft]);

  function update<K extends keyof Draft>(key: K, value: Draft[K]) {
    const presetControlled = new Set<keyof Draft>([
      "epochs", "repeats", "rank", "alpha", "resolution", "learning_rate",
      "train_batch_size", "optimizer", "scheduler", "gradient_checkpointing", "cache_latents",
    ]);
    setDraft((current) => ({
      ...current,
      [key]: value,
      ...(key === "dataset_snapshot_id" ? { preview_profile_snapshot_id: null } : {}),
      ...(presetControlled.has(key) ? { quality_preset: "custom" as PresetId } : {}),
    }));
    setPreflight(null);
    setPreflightDraft(null);
  }

  function selectPreset(next: Exclude<PresetId, "custom">) {
    const { label: _label, hint: _hint, ...values } = presets[next];
    setDraft((current) => ({ ...current, ...values, quality_preset: next }));
    setPreflight(null);
    setPreflightDraft(null);
  }

  async function runPreflight() {
    setChecking(true);
    try {
      const candidate = { ...draft };
      const query = new URLSearchParams({
        project_id: String(project.id), model_family: draft.model_family, train_data_dir: "", base_checkpoint_path: draft.base_checkpoint_path,
        training_goal: draft.training_goal,
        quality_preset: draft.quality_preset,
        gpu_device_id: String(draft.gpu_device_id), epochs: String(draft.epochs), repeats: String(draft.repeats), resolution: String(draft.resolution),
        train_batch_size: String(draft.train_batch_size), learning_rate: String(draft.learning_rate), rank: String(draft.rank), optimizer: draft.optimizer,
        gradient_checkpointing: String(draft.gradient_checkpointing),
        ...(draft.dataset_snapshot_id != null ? { dataset_snapshot_id: String(draft.dataset_snapshot_id) } : {}),
        ...(draft.preview_profile_snapshot_id != null ? { preview_profile_snapshot_id: String(draft.preview_profile_snapshot_id) } : {}),
      });
      // CUDA ordinal verification imports torch in the configured trainer
      // Python. A cold probe can legitimately exceed the generic 10s API
      // budget, especially while another creative app owns GPU1.
      const result = await apiGet<PreflightResult>(`/training/preflight?${query}`, 60_000);
      setPreflight(result);
      setPreflightDraft(candidate);
    } catch (error) {
      showError(error instanceof Error ? error.message : "Preflightに失敗しました");
    } finally {
      setChecking(false);
    }
  }

  async function startTraining() {
    if (!preflight || !preflightDraft || (preflight.overall === "blocked" && !preflight.can_queue)) return;
    setStarting(true);
    try {
      const result = await apiPost<{ run_id: number; status: string; message: string }>("/training/start", {
        project_id: project.id, ...preflightDraft, gpu_device_id: preflightDraft.gpu_device_id, preset_id: null, save_every_n_epochs: 1, save_precision: "fp16", mixed_precision: "bf16",
        queue_if_busy: true,
        xformers: true, cache_latents_to_disk: false, persistent_data_loader_workers: true, max_data_loader_n_workers: 4,
        network_train_unet_only: true, reg_data_dir: "", train_data_dir: "", vae_path: "", text_encoder_path: "",
      }, "POST", undefined, 60_000);
      showNotice(result.status === "queued" ? `Run #${result.run_id}をGPU1待機キューへ追加しました` : `Run #${result.run_id}を開始しました`);
      await onStarted();
      onOpenRuns();
    } catch (error) {
      showError(error instanceof Error ? error.message : "Run開始に失敗しました");
    } finally {
      setStarting(false);
    }
  }

  const estimatedMinutes = estimate ? Math.max(1, Math.round(estimate.eta_seconds / 60)) : null;
  const checks = preflight?.checks ?? [];
  const blockedChecks = checks.filter((check) => check.level === "blocked");
  const queueBlocked = preflight?.overall === "blocked" && !preflight.can_queue;
  const queueBlocker = blockedChecks.find((check) => !["GPU1 VRAM", "GPU1 Occupancy", "GPU", "Queue"].includes(check.label)) ?? blockedChecks[0];

  return (
    <div className="studio-workspace is-train">
      <header className="studio-page-header">
        <div><p className="studio-eyebrow">02 · TRAIN</p><h1>再現可能なRunを作る</h1><p>まず目的とPresetを選び、実測Preflightで安全性を確認します。</p></div>
        <span className="studio-save-state"><Save size={14} />{saving ? "保存中…" : savedAt ? `下書き保存 ${savedAt}` : loaded ? "変更なし" : "読込中…"}</span>
      </header>

      <div className="studio-train-layout">
        <section className="studio-train-form">
          <div className="studio-form-section"><div className="studio-form-number">1</div><div className="studio-form-content"><div className="studio-form-heading"><h2>Goal</h2><p>LoRAで固定したい特徴を選びます。選択はRun証拠にも保存されます。</p></div><div className="studio-choice-grid is-four">{trainingGoals.map((item) => <button key={item.id} className={draft.training_goal === item.id ? "is-selected" : ""} onClick={() => update("training_goal", item.id)}><strong>{item.label}</strong></button>)}</div></div></div>

          <div className="studio-form-section"><div className="studio-form-number">2</div><div className="studio-form-content"><div className="studio-form-heading"><h2>Setup</h2><p>普段決める項目だけを表示しています。</p></div>
            <div className="studio-field-grid">
              <label className="studio-field"><span>Dataset Snapshot</span><select value={draft.dataset_snapshot_id ?? ""} onChange={(event) => update("dataset_snapshot_id", event.target.value ? Number(event.target.value) : null)}><option value="">Snapshotを選択（自動選択しません）</option>{snapshots.map((snapshot) => <option key={snapshot.id} value={snapshot.id}>#{snapshot.id} · {snapshot.name} · {snapshot.item_count} images</option>)}</select></label>
              <label className="studio-field"><span>Preview conditions</span><select value={draft.preview_profile_snapshot_id ?? ""} onChange={(event) => update("preview_profile_snapshot_id", event.target.value ? Number(event.target.value) : null)} disabled={draft.dataset_snapshot_id == null}><option value="">固定Preview条件を選択</option>{previewSnapshots.map((snapshot) => <option key={snapshot.id} value={snapshot.id}>#{snapshot.id} · {snapshot.name}</option>)}</select></label>
              <label className="studio-field"><span>Base Model</span><select value={draft.base_checkpoint_path} onChange={(event) => update("base_checkpoint_path", event.target.value)}><option value="">モデルを選択</option>{selectableCheckpoints.map((checkpoint) => <option key={checkpoint.path} value={checkpoint.path}>{checkpoint.name}</option>)}</select></label>
              <label className="studio-field"><span>Output name</span><input value={draft.output_name} onChange={(event) => update("output_name", event.target.value)} /></label>
            </div>
            <div className="studio-preset-row"><span className="studio-field-label">Quality preset</span><div className="studio-preset-grid">{(Object.keys(presets) as Array<Exclude<PresetId, "custom">>).map((id) => <button key={id} className={draft.quality_preset === id ? "is-selected" : ""} onClick={() => selectPreset(id)}><Sparkles size={15} /><strong>{presets[id].label}</strong><small>{presets[id].hint}</small></button>)}{draft.quality_preset === "custom" && <span className="studio-custom-badge">Custom</span>}</div></div>
            <details className="studio-advanced"><summary><span><ChevronDown size={15} /> Advanced settings</span><small>変更するとCustomになります</small></summary><div className="studio-field-grid is-advanced">
              <label className="studio-field"><span>Epochs</span><input type="number" min="1" value={draft.epochs} onChange={(event) => update("epochs", Number(event.target.value))} /></label>
              <label className="studio-field"><span>Repeats</span><input type="number" min="1" value={draft.repeats} onChange={(event) => update("repeats", Number(event.target.value))} /></label>
              <label className="studio-field"><span>Resolution</span><input type="number" step="64" value={draft.resolution} onChange={(event) => update("resolution", Number(event.target.value))} /></label>
              <label className="studio-field"><span>Learning rate</span><input type="number" step="0.00001" value={draft.learning_rate} onChange={(event) => update("learning_rate", Number(event.target.value))} /></label>
              <label className="studio-field"><span>Rank</span><input type="number" value={draft.rank} onChange={(event) => update("rank", Number(event.target.value))} /></label>
              <label className="studio-field"><span>Alpha</span><input type="number" value={draft.alpha} onChange={(event) => update("alpha", Number(event.target.value))} /></label>
              <label className="studio-field"><span>Optimizer</span><input value={draft.optimizer} onChange={(event) => update("optimizer", event.target.value)} /></label>
              <label className="studio-field"><span>Scheduler</span><input value={draft.scheduler} onChange={(event) => update("scheduler", event.target.value)} /></label>
            </div></details>
          </div></div>

          <div className="studio-form-section"><div className="studio-form-number">3</div><div className="studio-form-content"><div className="studio-form-heading"><h2>Preflight</h2><p>Dataset、モデル、GPU、保存先を実際の値で検査します。</p></div>
            {configurationErrors.length > 0 && <div className="studio-inline-warning"><AlertTriangle size={15} /><span><strong>設定を確認してください</strong><small>{configurationErrors.join(" · ")}</small></span></div>}
            {!preflight ? <button className="studio-primary-button is-large" onClick={() => void runPreflight()} disabled={checking || configurationErrors.length > 0}>{checking ? <Loader2 size={17} className="spin" /> : <ShieldAlert size={17} />}{checking ? "確認中…" : "学習前チェックを実行"}</button> : <div className={`studio-preflight-result is-${preflight.overall}`}>
              <header>{preflight.overall === "ok" ? <CheckCircle2 size={22} /> : <AlertTriangle size={22} />}<div><strong>{preflight.overall === "ok" ? "Ready to train" : preflight.overall === "warning" ? "確認して開始できます" : preflight.can_queue ? "GPU1待機キューへ追加できます" : "開始できません"}</strong><small>{checks.filter((check) => check.level === "ok").length}/{checks.length} checks passed</small></div><button onClick={() => void runPreflight()}>再検査</button></header>
              <div className="studio-check-list">{checks.map((check) => <div key={`${check.label}-${check.detail}`} className={`is-${check.level}`}>{check.level === "ok" ? <CheckCircle2 size={15} /> : <AlertTriangle size={15} />}<span><strong>{check.label}</strong><small>{check.detail}</small></span></div>)}</div>
              <button className="studio-primary-button is-large" disabled={queueBlocked || starting} onClick={() => void startTraining()}>{starting ? <Loader2 size={17} className="spin" /> : queueBlocked ? <ShieldAlert size={17} /> : <Play size={17} />}{starting ? "Run作成中…" : queueBlocked ? `必須条件${blockedChecks.length}件を解消` : preflight.can_queue ? "GPU1待機キューへ追加" : "Start training"}</button>
            </div>}
          </div></div>
        </section>

        <aside className="studio-plan-summary">
          <p className="studio-eyebrow">TRAINING PLAN</p><h2>今回のRun</h2>
          <dl>
            <div><dt>Goal</dt><dd>{trainingGoalLabel(draft.training_goal)}</dd></div><div><dt>Dataset</dt><dd>{selectedSnapshot ? `#${selectedSnapshot.id} · ${selectedSnapshot.name} · ${selectedSnapshot.item_count}` : "未選択"}</dd></div>
            <div><dt>Model</dt><dd>{selectedSpec?.display_name ?? draft.model_family}</dd></div><div><dt>Preset</dt><dd>{draft.quality_preset === "custom" ? "Custom" : presets[draft.quality_preset].label}</dd></div>
            <div><dt>Checkpoints</dt><dd>{draft.epochs}</dd></div><div><dt>Estimated</dt><dd>{estimatedMinutes ? `約 ${estimatedMinutes} min` : "計算中"}</dd></div>
          </dl>
          <div className={`studio-gpu-card ${gpuBlocked ? "is-blocked" : ""}`}><Cpu size={18} /><span><strong>{targetGpu ? `GPU ${targetGpu.index} · ${targetGpu.name.replace("NVIDIA GeForce ", "")}` : "GPU未検出"}</strong><small>{targetGpu ? `${freeVram?.toLocaleString()} MB free · ${targetGpu.occupancy_status ?? "状態不明"}` : "実測値なし"}</small></span></div>
          {gpuBlocked && queueBlocked && <div className="studio-inline-warning"><AlertTriangle size={15} /><span><strong>GPU以外の準備が残っています</strong><small>{queueBlocker ? `${queueBlocker.label}: ${queueBlocker.detail}` : "Preflightの必須条件を確認してください。"}</small></span></div>}
          {gpuBlocked && !queueBlocked && <div className="studio-inline-warning"><AlertTriangle size={15} /><span><strong>GPUは使用中です</strong><small>{preflight?.can_queue ? "外部プロセスには触れず、Runを安全な待機キューへ追加できます。" : "Preflight通過後は外部プロセスに触れず待機キューへ追加できます。"}</small></span></div>}
          {estimate && <div className="studio-estimate"><Gauge size={16} /><span><strong>{estimate.total_steps.toLocaleString()} steps</strong><small>{draft.epochs} epochs · batch {draft.train_batch_size}</small></span><Clock3 size={15} /></div>}
          <details className="studio-details" open><summary>Immutable training stream</summary><dl><dt>Contract state</dt><dd>{preflightDraft ? "Preflightで凍結済み" : "未凍結"}</dd><dt>Training goal</dt><dd>{preflightDraft?.training_goal ?? draft.training_goal}</dd><dt>Quality preset</dt><dd>{preflightDraft?.quality_preset ?? draft.quality_preset}</dd><dt>Physical GPU</dt><dd>{preflightDraft?.gpu_device_id ?? draft.gpu_device_id}</dd><dt>CUDA ordinal</dt><dd>{resolvedEvidence?.cuda_visible_devices ?? "preflight待ち"}</dd><dt>Engine</dt><dd>{selectedSpec?.training_backend ?? draft.training_engine}</dd><dt>Dataset Snapshot</dt><dd>#{selectedSnapshot?.id ?? "unknown"} · {selectedSnapshot?.snapshot_hash ?? "hash unknown"}</dd><dt>Preview Profile Snapshot</dt><dd>#{resolvedEvidence?.preview_profile_snapshot_id ?? selectedPreviewSnapshot?.id ?? "unknown"} · {resolvedEvidence?.preview_profile?.snapshot_hash ?? selectedPreviewSnapshot?.snapshot_hash ?? "hash unknown"}</dd><dt>Base path</dt><dd>{resolvedEvidence?.base_checkpoint_path ?? (draft.base_checkpoint_path || "unknown")}</dd><dt>Text Encoder</dt><dd>{resolvedEvidence?.dependencies?.text_encoder_path ?? "preflight待ち"}</dd><dt>VAE</dt><dd>{resolvedEvidence?.dependencies?.vae_path ?? "preflight待ち"}</dd><dt>Trainer</dt><dd>{resolvedEvidence?.trainer_script_path ?? "preflight待ち"}</dd><dt>Python</dt><dd>{resolvedEvidence?.python_executable ?? "preflight待ち"}</dd></dl></details>
        </aside>
      </div>
    </div>
  );
}
