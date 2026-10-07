import { useEffect, useState } from "react";
import { AlertTriangle, CheckCircle2, Clock3, RefreshCw } from "lucide-react";
import { apiDelete, apiGet } from "../lib/api";
import type { Project, RunDetail, RunEvidence, RunSummary } from "../types";

type Props = {
  selectedProject: Project | null;
  selectedProjectId: number | null;
  showError: (message: string) => void;
};
type TrainingCache = {
  run_id: number;
  path: string;
  dataset_snapshot_id?: number | null;
  dataset_fingerprint?: string | null;
  item_count?: number | null;
  file_count: number;
  size_bytes: number;
  created_at?: string | null;
  reusable: boolean;
  in_use: boolean;
  consuming_run?: number | null;
};

function statusLabel(status: string) {
  const labels: Record<string, string> = {
    queued: "QUEUED", training: "RUNNING", completed: "COMPLETED",
    error: "FAILED", paused: "STOPPED", idle: "IDLE",
  };
  return labels[status] ?? status.toUpperCase();
}

function StatusIcon({ status }: { status: string }) {
  if (status === "completed") return <CheckCircle2 size={15} className="text-emerald-400" />;
  if (status === "error") return <AlertTriangle size={15} className="text-red-400" />;
  return <Clock3 size={15} className="text-amber-400" />;
}

function fileLabel(path: string) {
  return path.split(/[\\/]/).pop() || path;
}

function timingLabel(seconds?: number | null) {
  if (seconds == null) return "—（未計測）";
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  return `${Math.floor(seconds / 60)}m ${(seconds % 60).toFixed(0)}s`;
}

export default function Runs({ selectedProject, selectedProjectId, showError }: Props) {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<number | null>(null);
  const [evidence, setEvidence] = useState<RunEvidence | null>(null);
  const [loading, setLoading] = useState(false);
  const [cache, setCache] = useState<TrainingCache[]>([]);

  async function loadRuns() {
    setLoading(true);
    try {
      const result = await apiGet<{ runs: RunSummary[] }>(
        selectedProjectId == null ? "/training/runs" : `/training/runs?project_id=${selectedProjectId}`,
      );
      setRuns(result.runs);
      showError("");
      if (selectedRunId == null && result.runs[0]) setSelectedRunId(result.runs[0].id);
    } catch (error) {
      showError(error instanceof Error ? error.message : "Run一覧の取得に失敗しました");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { void loadRuns(); }, [selectedProjectId]);

  useEffect(() => {
    if (selectedRunId == null) { setEvidence(null); return; }
    Promise.all([
      apiGet<RunDetail>(`/training/runs/${selectedRunId}`),
      apiGet<{ cache: TrainingCache[] }>(`/training/cache?run_id=${selectedRunId}`),
    ]).then(([detail, nextCache]) => { setEvidence(detail.evidence); setCache(nextCache.cache ?? []); showError(""); })
      .catch((error) => showError(error instanceof Error ? error.message : "Run証拠の取得に失敗しました"));
  }, [selectedRunId, showError]);

  async function removeCache(runId: number) {
    try { await apiDelete(`/training/cache/${runId}`); setCache([]); }
    catch (error) { showError(error instanceof Error ? error.message : "Training Cacheの削除に失敗しました"); }
  }

  function sizeLabel(bytes: number) {
    if (bytes < 1024 * 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
    return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GB`;
  }

  const gpuEnvironment = evidence?.manifest?.environment as {
    requested_gpu_id?: number;
    cuda_visible_devices?: string | null;
    gpu_mapping?: {
      physical_index?: number;
      cuda_index?: number;
      physical_name?: string;
      cuda_name?: string;
      verified?: boolean;
    };
  } | undefined;
  const gpuMapping = (evidence?.resolved?.gpu_mapping ?? gpuEnvironment?.gpu_mapping) as {
    physical_index?: number;
    cuda_index?: number;
    physical_name?: string;
    cuda_name?: string;
    verified?: boolean;
  } | undefined;
  const requestedGpuId = evidence?.requested?.gpu_device_id ?? gpuEnvironment?.requested_gpu_id;
  const gpuMappingLabel = gpuMapping?.verified
    ? `物理GPU${gpuMapping.physical_index} → CUDA ${gpuMapping.cuda_index} (${gpuMapping.cuda_name ?? gpuMapping.physical_name ?? "unknown"})`
    : requestedGpuId != null
      ? `Requested物理GPU${String(requestedGpuId ?? "—")} / CUDA_VISIBLE_DEVICES=—（照合証拠なし）`
      : "—（照合証拠なし）";

  return (
    <div className="space-y-4">
      <div className="flex items-start justify-between">
        <div>
          <h2 className="text-xl font-bold text-gray-100">Runs</h2>
          <p className="text-sm text-gray-500">{selectedProject?.name ?? "全プロジェクト"} の実行履歴と証拠</p>
        </div>
        <button onClick={() => void loadRuns()} className="flex items-center gap-2 rounded-lg border border-gray-700 px-3 py-2 text-xs text-gray-300 hover:bg-gray-800" disabled={loading}>
          <RefreshCw size={13} className={loading ? "animate-spin" : ""} /> 更新
        </button>
      </div>

      <div className="grid grid-cols-[minmax(220px,300px)_1fr] gap-4">
        <section className="rounded-xl border border-gray-800 bg-gray-900/50">
          <div className="border-b border-gray-800 px-4 py-3 text-xs font-semibold text-gray-400">Run履歴</div>
          {runs.length === 0 ? <p className="p-4 text-xs text-gray-600">Runはまだありません。</p> : runs.map((run) => (
            <button key={run.id} onClick={() => setSelectedRunId(run.id)} className={`flex w-full items-center gap-2 border-b border-gray-800/70 px-4 py-3 text-left hover:bg-gray-800/60 ${selectedRunId === run.id ? "bg-indigo-950/40" : ""}`}>
              <StatusIcon status={run.status} />
              <span className="min-w-0 flex-1"><span className="block text-sm text-gray-200">Run #{run.id}</span><span className="block text-[11px] text-gray-500">{statusLabel(run.status)} · E{run.current_epoch}/{run.total_epochs}</span><span className="block text-[10px] text-indigo-300">Stage: {run.current_stage ?? "planning"}</span>{run.dataset_snapshot_id != null && <span className="block text-[10px] text-emerald-400">Snapshot #{run.dataset_snapshot_id}</span>}</span>
            </button>
          ))}
        </section>

        <section className="rounded-xl border border-gray-800 bg-gray-900/50 p-4">
          {!evidence ? <p className="text-sm text-gray-600">Runを選択してください。</p> : <>
            <div className="mb-4 flex items-center justify-between"><div><h3 className="text-base font-semibold text-gray-200">Run #{evidence.run_id}</h3><p className="text-xs text-gray-500">{statusLabel(evidence.status)} · Stage: <span className="text-indigo-300">{evidence.current_stage ?? "planning"}</span></p></div><span className="rounded bg-gray-800 px-2 py-1 text-[10px] text-gray-400">Requested / Resolved / Observed</span></div>
            {(evidence.failure || evidence.status === "error") && <div className="mb-4 rounded-lg border border-red-900/60 bg-red-950/20 p-3"><p className="text-sm font-semibold text-red-300">{evidence.failure?.code ?? "TRAINING_FAILED"}</p><p className="mt-1 text-xs text-red-200/80">{evidence.failure?.message ?? "このRunは失敗状態です。旧Runのため詳細なFailureInfoは記録されていません。"}</p>{evidence.failure?.stage && <p className="mt-1 text-[10px] text-red-300/70">stage: {evidence.failure.stage}</p>}</div>}
            {evidence.manifest && <div className="mb-4 grid gap-2 rounded-lg border border-gray-800 bg-gray-950/50 p-3 text-[11px] text-gray-500 md:grid-cols-3"><span>Requested Engine: {String(evidence.manifest.engine ?? "unknown")}</span><span>Resolved Backend: <strong className="text-indigo-300">{String(evidence.resolved?.resolved_training_backend ?? "—")}</strong></span><span>Model: {String(evidence.manifest.model_family ?? "unknown")}</span><span>Dataset: {String(evidence.manifest.dataset?.source ?? "unknown")}</span><span>Snapshot: {String(evidence.manifest.dataset_snapshot_id ?? evidence.resolved.dataset_snapshot_id ?? "—")}</span><span>GPU Mapping: {gpuMappingLabel}</span><span className="md:col-span-3">Generated config: {evidence.manifest.generated_configs?.length ? evidence.manifest.generated_configs.map(fileLabel).join(" · ") : "—"}</span></div>}
            <section className="mb-4 grid gap-3 rounded-lg border border-gray-800 bg-gray-950/50 p-3 md:grid-cols-2"><div><p className="text-xs font-semibold text-gray-300">Training / Checkpoint</p><p className="mt-1 text-[11px] text-emerald-300">Training: {evidence.training_status ?? evidence.status}</p><p className="mt-1 text-[11px] text-gray-500">Checkpoint: {evidence.checkpoints?.length ?? 0} 件</p>{evidence.checkpoints?.map((checkpoint) => <p key={checkpoint.id} className="mt-1 truncate text-[10px] text-gray-600">#{checkpoint.id} · E{checkpoint.epoch ?? "—"} · {checkpoint.validation_status ?? "unknown"}</p>)}</div><div><p className="text-xs font-semibold text-gray-300">Preview（Trainingとは別状態）</p><p className="mt-1 text-[11px] text-indigo-300">{evidence.preview?.succeeded ?? 0}/{evidence.preview?.expected ?? 0} succeeded</p><p className="mt-1 text-[10px] text-gray-600">failed {evidence.preview?.failed ?? 0} · pending {evidence.preview?.pending ?? 0} · running {evidence.preview?.running ?? 0}</p></div></section>
            <section className="mb-4 rounded-lg border border-gray-800 bg-gray-950/50 p-3"><div className="flex items-center justify-between"><p className="text-xs font-semibold text-gray-300">Speed / Timing Evidence</p><span className="text-[10px] text-gray-600">未計測値は推測しません</span></div><div className="mt-2 grid gap-x-4 gap-y-1 text-[11px] text-gray-500 sm:grid-cols-2 md:grid-cols-3">{([["Cache", evidence.timing?.cache_seconds], ["First Checkpoint", evidence.timing?.time_to_first_checkpoint_seconds], ["First Preview", evidence.timing?.time_to_first_preview_seconds], ["Total Training", evidence.timing?.total_training_seconds], ["Dataset Generation", evidence.timing?.dataset_generation_seconds], ["Human Review", evidence.timing?.human_review_seconds]] as const).map(([label, value]) => <span key={label}>{label}: <strong className={value == null ? "text-gray-700" : "text-amber-300"}>{timingLabel(value)}</strong></span>)}</div></section>
            <section className="mb-4 rounded-lg border border-gray-800 bg-gray-950/50 p-3"><div className="flex items-center justify-between"><p className="text-xs font-semibold text-gray-300">Training Cache</p><span className="text-[10px] text-gray-600">Preview Historyとは別管理</span></div>{cache.length === 0 ? <p className="mt-2 text-[11px] text-gray-600">このRunにTraining Cacheはありません。</p> : cache.map((entry) => <div key={entry.path} className="mt-2 rounded border border-gray-800 p-2 text-[11px] text-gray-500"><div className="flex flex-wrap gap-x-3 gap-y-1"><span>{entry.item_count ?? "—"} items · {entry.file_count} files · {sizeLabel(entry.size_bytes)}</span><span>Snapshot #{entry.dataset_snapshot_id ?? "—"}</span><span className={entry.in_use ? "text-amber-300" : "text-emerald-300"}>{entry.in_use ? `使用中 Run #${entry.consuming_run}` : entry.reusable ? "再利用可能" : "未使用"}</span></div><div className="mt-1 grid gap-x-3 gap-y-1 text-[10px] text-gray-600 sm:grid-cols-2"><span>Fingerprint: <strong className="font-mono text-gray-500">{entry.dataset_fingerprint ?? "—"}</strong></span><span>Created: {entry.created_at ? new Date(entry.created_at).toLocaleString("ja-JP") : "—"}</span></div><p className="mt-1 truncate text-[10px] text-gray-700">Cache: {fileLabel(entry.path)}</p><button disabled={entry.in_use} onClick={() => void removeCache(entry.run_id)} className="mt-2 rounded border border-gray-700 px-2 py-1 text-[10px] text-gray-400 hover:bg-gray-800 disabled:cursor-not-allowed disabled:opacity-40">{entry.in_use ? "使用中のため削除不可" : "Training Cacheを削除"}</button></div>)}</section>
            <div className="overflow-x-auto"><table className="w-full text-left text-xs"><thead><tr className="border-b border-gray-700 text-gray-500"><th className="py-2">Parameter</th><th>Requested</th><th>Resolved</th><th>Observed</th><th>Source</th></tr></thead><tbody>{evidence.parameters.map((param) => <tr key={param.key} className="border-b border-gray-800/70"><td className="py-2 font-medium text-gray-300">{param.key}</td><td className="text-gray-400">{String(param.requested ?? "—")}</td><td className="text-gray-400">{String(param.resolved ?? "—")}</td><td className={param.observed == null ? "text-gray-600" : param.mismatch ? "text-red-300" : "text-emerald-300"}>{String(param.observed ?? "—")}</td><td className="text-gray-600">{param.observed_source}</td></tr>)}</tbody></table></div>
            <p className="mt-4 text-[11px] text-gray-600">Observedが取得できない値は推測せず「—」で表示します。</p>
          </>}
        </section>
      </div>
    </div>
  );
}
