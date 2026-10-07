import { useEffect, useMemo, useState } from "react";
import { AlertTriangle, CheckCircle2, Save, Play, Loader2 } from "lucide-react";
import { apiGet, apiPost } from "../lib/api";
import type { ModelSpec, Project, TrainingEstimate, PreflightResult, ResourceStats } from "../types";

type Props = { project: Project; onRefresh: () => Promise<void>; showError: (message: string) => void; showNotice: (message: string) => void; onOpenRuns: () => void };
type Draft = { model_family: string; training_engine: string; base_checkpoint_path: string; dataset_source: "original" | "processed"; dataset_snapshot_id: number | null; gpu_device_id: number; rank: number; alpha: number; learning_rate: number; train_batch_size: number; epochs: number; repeats: number; resolution: number; output_name: string; optimizer: string; scheduler: string; gradient_checkpointing: boolean; cache_latents: boolean };
const defaults: Draft = { model_family: "anima", training_engine: "auto", base_checkpoint_path: "", dataset_source: "original", dataset_snapshot_id: null, gpu_device_id: 1, rank: 16, alpha: 16, learning_rate: 0.0001, train_batch_size: 1, epochs: 10, repeats: 5, resolution: 1024, output_name: "anima_lora", optimizer: "AdamW8bit", scheduler: "constant", gradient_checkpointing: true, cache_latents: true };
const animaFallbackSpec: ModelSpec = { model_family: "anima", display_name: "Anima", backend: "kohya", training_backend: "anima", preview_backend: "comfyui", supports: ["lora", "t2i"], default_resolution: 1024, default_preset: "anima", required_models: [], has_cache_steps: false };

export default function TrainPage({ project, onRefresh, showError, showNotice, onOpenRuns }: Props) {
  const [draft, setDraft] = useState<Draft>(defaults);
  const [specs, setSpecs] = useState<ModelSpec[]>([animaFallbackSpec]);
  const [checkpoints, setCheckpoints] = useState<{ name: string; path: string }[]>([]);
  const [snapshots, setSnapshots] = useState<{ id: number; name: string; item_count: number; snapshot_hash: string }[]>([]);
  const [resources, setResources] = useState<ResourceStats | null>(null);
  const [estimate, setEstimate] = useState<TrainingEstimate | null>(null);
  const [preflight, setPreflight] = useState<PreflightResult | null>(null);
  const [reviewOpen, setReviewOpen] = useState(false);
  const [saving, setSaving] = useState(false);
  const [starting, setStarting] = useState(false);
  const [savedAt, setSavedAt] = useState<string | null>(null);

  // Anima Basepipeの物理GPUはGPU1固定。表示上の下書きが旧設定から
  // GPU0を読み込んでも、Run開始前に必ずGPU1へ戻す。
  useEffect(() => {
    if (draft.model_family === "auto") {
      setDraft((current) => ({ ...current, model_family: "anima" }));
      return;
    }
    if (draft.model_family === "anima" && draft.gpu_device_id !== 1) {
      setDraft((current) => ({ ...current, gpu_device_id: 1 }));
    }
  }, [draft.model_family, draft.gpu_device_id]);

  useEffect(() => {
    void Promise.all([
      apiGet<{ config: Partial<Draft> }>(`/training/config-draft/${project.id}`).then((r) => {
        const next = { ...defaults, ...r.config };
        if (!r.config.model_family || r.config.model_family === "auto") next.model_family = "anima";
        if (!r.config.training_engine || r.config.training_engine === "auto") next.training_engine = "auto";
        next.gpu_device_id = 1;
        setDraft(next);
      }),
      apiGet<{ specs: ModelSpec[] }>("/training/model-specs").then((r) => setSpecs([animaFallbackSpec, ...(r.specs ?? []).filter((s) => s.model_family !== "anima")])),
      apiGet<{ checkpoints: { name: string; path: string }[] }>("/training/checkpoints").then((r) => setCheckpoints(r.checkpoints ?? [])),
      apiGet<{ snapshots: { id: number; name: string; item_count: number; snapshot_hash: string }[] }>(`/basepipe/projects/${project.id}/workspace`).then((r) => setSnapshots(r.snapshots ?? [])),
      apiGet<ResourceStats>("/training/resources").then(setResources),
    ]).catch((e) => showError(e instanceof Error ? e.message : "Training設定の取得に失敗しました"));
  }, [project.id]);

  useEffect(() => {
    const timer = setInterval(() => {
      void apiGet<ResourceStats>("/training/resources").then(setResources).catch(() => undefined);
    }, 5000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    if (!checkpoints.length) return;
    setDraft((current) => {
      if (current.model_family !== "anima") return current;
      if (current.base_checkpoint_path && /anima/i.test(current.base_checkpoint_path)) return current;
      const preferred = checkpoints.find((checkpoint) => checkpoint.name.toLowerCase() === "anima-base-v1.0.safetensors")
        ?? checkpoints.find((checkpoint) => checkpoint.name.toLowerCase().includes("anima"));
      return preferred ? { ...current, base_checkpoint_path: preferred.path } : current;
    });
  }, [checkpoints, draft.model_family, draft.base_checkpoint_path]);

  useEffect(() => {
    if (draft.model_family !== "anima" || draft.dataset_snapshot_id !== null || !snapshots[0]) return;
    setDraft((current) => current.dataset_snapshot_id === null
      ? { ...current, dataset_snapshot_id: snapshots[0].id }
      : current);
  }, [draft.model_family, snapshots, draft.dataset_snapshot_id]);

  useEffect(() => {
    const timer = setTimeout(() => {
      setSaving(true);
      void apiPost(`/training/config-draft/${project.id}`, draft)
        .then(() => setSavedAt(new Date().toLocaleTimeString()))
        .catch((e) => showError(e instanceof Error ? e.message : "下書き保存に失敗しました"))
        .finally(() => setSaving(false));
    }, 500);
    return () => clearTimeout(timer);
  }, [project.id, draft]);

  const selectedSpec = useMemo(() => specs.find((s) => s.model_family === draft.model_family), [specs, draft.model_family]);
  const isAnima = draft.model_family === "anima";
  const targetGpu = resources?.gpu.find((gpu) => gpu.index === 1);
  const targetFreeVram = targetGpu ? targetGpu.vram_total_mb - targetGpu.vram_used_mb : null;
  const animaVramReady = targetFreeVram == null || targetFreeVram >= 20000;
  useEffect(() => {
    const query = new URLSearchParams({
      project_id: String(project.id),
      epochs: String(draft.epochs),
      repeats: String(draft.repeats),
      resolution: String(draft.resolution),
      batch_size: String(draft.train_batch_size),
      base_checkpoint_path: draft.base_checkpoint_path,
      optimizer: draft.optimizer,
      rank: String(draft.rank),
      gradient_checkpointing: String(draft.gradient_checkpointing),
      mixed_precision: "bf16",
      gpu_device_id: String(draft.gpu_device_id),
      model_family: draft.model_family,
    });
    const timer = setTimeout(() => {
      void apiGet<TrainingEstimate>(`/training/estimate?${query}`)
        .then(setEstimate)
        .catch(() => setEstimate(null));
    }, 250);
    return () => clearTimeout(timer);
  }, [project.id, draft.epochs, draft.repeats, draft.resolution, draft.train_batch_size, draft.base_checkpoint_path, draft.optimizer, draft.rank, draft.gradient_checkpointing, draft.gpu_device_id, draft.model_family]);
  const set = <K extends keyof Draft>(key: K, value: Draft[K]) => setDraft((current) => ({ ...current, [key]: value }));
  const numericInput = <K extends keyof Draft>(key: K, label: string, step = "1") => <label className="space-y-1"><span className="text-xs text-gray-400">{label}</span><input type="number" step={step} value={String(draft[key])} onChange={(e) => set(key, Number(e.target.value) as Draft[K])} className="w-full rounded-lg border border-gray-700 bg-gray-950 px-3 py-2 text-sm text-gray-200"/></label>;
  const textInput = <K extends keyof Draft>(key: K, label: string) => <label className="space-y-1"><span className="text-xs text-gray-400">{label}</span><input type="text" value={String(draft[key])} onChange={(e) => set(key, e.target.value as Draft[K])} className="w-full rounded-lg border border-gray-700 bg-gray-950 px-3 py-2 text-sm text-gray-200"/></label>;

  async function openReview() {
    try {
      const q = new URLSearchParams({
        project_id: String(project.id),
        model_family: draft.model_family,
        train_data_dir: "",
        base_checkpoint_path: draft.base_checkpoint_path,
        gpu_device_id: String(draft.gpu_device_id),
        epochs: String(draft.epochs),
        repeats: String(draft.repeats),
        resolution: String(draft.resolution),
        train_batch_size: String(draft.train_batch_size),
        learning_rate: String(draft.learning_rate),
        rank: String(draft.rank),
        optimizer: draft.optimizer,
        gradient_checkpointing: String(draft.gradient_checkpointing),
        ...(draft.dataset_snapshot_id !== null ? { dataset_snapshot_id: String(draft.dataset_snapshot_id) } : {}),
      });
      const result = await apiGet<PreflightResult>(`/training/preflight?${q}`);
      setPreflight(result); setReviewOpen(true);
    } catch (e) { showError(e instanceof Error ? e.message : "Preflightに失敗しました"); }
  }
  async function startRun() {
    setStarting(true);
    try {
      await apiPost("/training/start", { project_id: project.id, ...draft, preset_id: null, save_every_n_epochs: 1, save_precision: "fp16", mixed_precision: "bf16", xformers: true, cache_latents_to_disk: false, persistent_data_loader_workers: true, max_data_loader_n_workers: 4, network_train_unet_only: false, reg_data_dir: "", train_data_dir: "", vae_path: "", text_encoder_path: "" });
      setReviewOpen(false); showNotice("Runを作成しました"); await onRefresh(); onOpenRuns();
    } catch (e) { showError(e instanceof Error ? e.message : "Run開始に失敗しました"); }
    finally { setStarting(false); }
  }

  const targetOccupancy = targetGpu?.occupancy_status;
  const foreignProcesses = targetGpu?.foreign_processes ?? [];
  return <div className="space-y-4"><div className="flex items-start justify-between"><div><h2 className="text-xl font-bold text-gray-100">New Training Run</h2><p className="text-sm text-gray-500">{project.name}の次のRunを設定します</p></div><span className="flex items-center gap-1 text-[11px] text-gray-500"><Save size={12}/>{saving ? "保存中…" : savedAt ? `Saved ${savedAt}` : "未保存"}</span></div><div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_340px]"><section className="space-y-4 rounded-xl border border-gray-800 bg-gray-900/50 p-5"><div className="grid gap-3 md:grid-cols-2"><label className="space-y-1"><span className="text-xs text-gray-400">Training Profile</span><select value={draft.model_family} onChange={(e) => set("model_family", e.target.value)} className="w-full rounded-lg border border-gray-700 bg-gray-950 px-3 py-2 text-sm text-gray-200"><option value="auto">Auto</option>{specs.map((s) => <option key={s.model_family} value={s.model_family}>{s.display_name}</option>)}</select></label><label className="space-y-1"><span className="text-xs text-gray-400">Engine</span><select value={draft.training_engine} onChange={(e) => set("training_engine", e.target.value)} className="w-full rounded-lg border border-gray-700 bg-gray-950 px-3 py-2 text-sm text-gray-200"><option value="auto">Auto{isAnima ? "（Anima専用Backendへ解決）" : ""}</option><option value="musubi" disabled={isAnima}>Musubi{isAnima ? "（Anima非対応）" : ""}</option><option value="kohya">Kohya{isAnima ? "（Anima専用Backend）" : ""}</option><option value="ai_toolkit" disabled={isAnima}>AI Toolkit{isAnima ? "（Anima非対応）" : ""}</option></select></label></div><label className="space-y-1"><span className="flex items-center justify-between text-xs text-gray-400"><span>GPU</span>{isAnima && <span className="text-emerald-400">GPU1 fixed</span>}</span><select disabled={isAnima} value={draft.gpu_device_id} onChange={(e) => set("gpu_device_id", Number(e.target.value))} className="w-full rounded-lg border border-gray-700 bg-gray-950 px-3 py-2 text-sm text-gray-200 disabled:cursor-not-allowed disabled:opacity-70">{(resources?.gpu ?? []).map((g) => <option key={g.index} value={g.index}>GPU {g.index}: {g.name} ({g.vram_total_mb} MB)</option>)}</select></label><label className="space-y-1"><span className="text-xs text-gray-400">Dataset Snapshot <span className="text-emerald-400">sealed</span></span><select value={draft.dataset_snapshot_id ?? ""} onChange={(e) => set("dataset_snapshot_id", e.target.value ? Number(e.target.value) : null)} className="w-full rounded-lg border border-gray-700 bg-gray-950 px-3 py-2 text-xs text-gray-200"><option value="">Snapshotを選択</option>{snapshots.map((snapshot) => <option key={snapshot.id} value={snapshot.id}>{snapshot.name} · {snapshot.item_count} images · {snapshot.snapshot_hash.slice(0, 8)}</option>)}</select></label><div><p className="mb-2 text-xs font-semibold text-gray-400">Training Dataset</p><div className="grid grid-cols-2 gap-2">{(["original", "processed"] as const).map((source) => <button key={source} onClick={() => set("dataset_source", source)} className={`rounded-lg border p-3 text-left text-xs ${draft.dataset_source === source ? "border-indigo-500 bg-indigo-950/40 text-indigo-200" : "border-gray-700 text-gray-500"}`}>{source === "processed" ? "Processed" : "Original"}<span className="mt-1 block text-[10px] text-gray-600">{source === "processed" ? "manifestを使用" : "収集元を使用"}</span></button>)}</div></div><label className="space-y-1"><span className="text-xs text-gray-400">Base Model</span><select value={draft.base_checkpoint_path} onChange={(e) => set("base_checkpoint_path", e.target.value)} className="w-full rounded-lg border border-gray-700 bg-gray-950 px-3 py-2 text-xs text-gray-200"><option value="">モデルを選択</option>{checkpoints.map((c) => <option key={c.path} value={c.path}>{c.name}</option>)}</select></label><div><p className="mb-2 text-xs font-semibold text-gray-400">Core Parameters</p><div className="grid grid-cols-2 gap-3 md:grid-cols-3">{numericInput("rank", "Rank")}{numericInput("alpha", "Alpha")}{numericInput("learning_rate", "Learning Rate", "0.0001")}{numericInput("train_batch_size", "Batch")}{numericInput("epochs", "Epoch")}{numericInput("repeats", "Repeats")}{numericInput("resolution", "Resolution")}</div></div><details><summary className="cursor-pointer text-xs text-gray-400">Advanced</summary><div className="mt-3 grid gap-3 md:grid-cols-2">{textInput("optimizer", "Optimizer")}{textInput("scheduler", "Scheduler")}</div></details></section><aside className="space-y-4"><section className="sticky top-0 space-y-3 rounded-xl border border-gray-800 bg-gray-900/80 p-4"><h3 className="text-sm font-semibold text-gray-300">Preflight</h3><p className="text-xs text-gray-500">Model: {selectedSpec?.display_name ?? draft.model_family}</p><p className="text-xs text-gray-500">GPU: {resources?.gpu?.find((g) => g.index === draft.gpu_device_id)?.name ?? "未検出"}</p>{isAnima && <p className={`text-xs ${animaVramReady ? "text-emerald-400" : "text-red-300"}`}>GPU1 VRAM: {targetFreeVram == null ? "未計測" : `${targetFreeVram} MB free / 20000 MB required`}{!animaVramReady && " · Start blocked"}</p>}{isAnima && targetOccupancy === "blocked" && <p className="text-xs text-red-300">GPU1 Occupancy: 外部DCC使用中 · Start blocked{foreignProcesses.length > 0 && <span className="block text-[10px] text-red-200">{foreignProcesses.map((process) => process.process_name).join(", ")}</span>}</p>}{isAnima && targetOccupancy === "clear" && <p className="text-xs text-emerald-400">GPU1 Occupancy: 外部DCCなし</p>}<p className={`text-xs ${draft.dataset_snapshot_id ? "text-emerald-400" : "text-amber-400"}`}>Snapshot: {draft.dataset_snapshot_id ? "sealed selected" : "Anima requires a sealed Snapshot"}</p>{estimate && <p className="text-xs text-gray-500">Estimated: {estimate.total_steps} steps / {estimate.eta_seconds}s</p>}<button onClick={() => void openReview()} className="flex w-full items-center justify-center gap-2 rounded-lg bg-indigo-600 px-3 py-2 text-sm font-semibold text-white hover:bg-indigo-500"><Play size={14}/>Review & Start</button></section></aside></div>{reviewOpen && preflight && <div className="fixed inset-0 z-40 flex items-center justify-center bg-black/70 p-4"><div className="w-full max-w-2xl rounded-xl border border-gray-700 bg-gray-900 p-5"><h3 className="text-lg font-semibold text-gray-100">Run Review</h3><p className="mt-1 text-xs text-gray-500">Requested値を確認してからRunを作成します。</p><div className="mt-4 grid gap-2 md:grid-cols-2">{preflight.checks.map((check) => <div key={`${check.label}-${check.detail}`} className="rounded-lg border border-gray-800 p-3 text-xs"><span className="flex items-center gap-2 text-gray-300">{check.level === "ok" ? <CheckCircle2 size={13} className="text-emerald-400"/> : <AlertTriangle size={13} className="text-amber-400"/>}{check.label}</span><p className="mt-1 text-gray-500">{check.detail}</p></div>)}</div><div className="mt-5 flex justify-end gap-2"><button onClick={() => setReviewOpen(false)} className="rounded-lg border border-gray-700 px-4 py-2 text-sm text-gray-400">Cancel</button><button onClick={() => void startRun()} disabled={starting || preflight.overall === "blocked"} className="flex items-center gap-2 rounded-lg bg-indigo-600 px-4 py-2 text-sm font-semibold text-white disabled:opacity-50">{starting && <Loader2 size={14} className="animate-spin"/>}Start Run</button></div></div></div>}</div>;
}
