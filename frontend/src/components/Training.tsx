import { useState, useEffect, useRef } from "react";
import {
  Play,
  Square,
  PauseCircle,
  RotateCcw,
  ChevronDown,
  ChevronUp,
  Image as ImageIcon,
  Clock,
  BarChart3,
  Layers,
  ImageOff,
  Loader2,
  ShieldCheck,
  AlertTriangle,
  XCircle,
  Terminal,
  Cpu,
  Zap,
  BookOpen,
  Plus,
  Trash2,
  CheckCircle2,
} from "lucide-react";
import { apiPost, apiGet, apiDelete } from "../lib/api";
import { parseIntOr, parseFloatOr, etaText } from "../lib/utils";
import type {
  Project,
  TrainingStatus,
  PreviewPrompts,
  DatasetPreview,
  PreviewSample,
  Preset,
  DatasetReport,
  TrainingMode,
} from "../types";

type Props = {
  projects: Project[];
  selectedProjectId: number | null;
  selectedProject: Project | null;
  onSelectProject: (id: number) => void;
  statuses: Record<number, TrainingStatus>;
  onRefresh: () => Promise<void>;
  prompts: PreviewPrompts;
  onPromptsChange: (p: PreviewPrompts) => void;
  showError: (msg: string) => void;
  showNotice: (msg: string) => void;
};

// ── サブコンポーネント ────────────────────────────────────────────────────

function ParamInput({
  label,
  value,
  onChange,
  inputMode = "numeric",
  placeholder,
  wide,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  inputMode?: "numeric" | "decimal" | "text";
  placeholder?: string;
  wide?: boolean;
}) {
  return (
    <label className={`block ${wide ? "col-span-2" : ""}`}>
      <span className="text-xs text-gray-400 mb-1 block">{label}</span>
      <input
        type="text"
        inputMode={inputMode}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        className="w-full bg-gray-700 border border-gray-600 text-gray-100 placeholder-gray-500 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500/30"
      />
    </label>
  );
}

function Metric({
  label,
  value,
  icon,
}: {
  label: string;
  value: string;
  icon: React.ReactNode;
}) {
  return (
    <div className="bg-gray-700 rounded-lg p-3">
      <div className="flex items-center gap-1 text-xs text-gray-400 mb-1">
        {icon}
        <span>{label}</span>
      </div>
      <div className="text-sm font-semibold text-gray-100 truncate">{value}</div>
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between">
      <dt className="text-gray-500">{label}</dt>
      <dd className="text-gray-300 font-medium">{value}</dd>
    </div>
  );
}

const STATUS_CONFIG: Record<string, { label: string; color: string; dot: string }> = {
  idle: { label: "待機中", color: "text-gray-400", dot: "bg-gray-500" },
  training: { label: "学習中", color: "text-amber-400", dot: "bg-amber-400 animate-pulse" },
  paused: { label: "一時停止", color: "text-blue-400", dot: "bg-blue-400" },
  completed: { label: "完了", color: "text-emerald-400", dot: "bg-emerald-400" },
  error: { label: "エラー", color: "text-red-400", dot: "bg-red-400" },
};

const PROFILE_TYPE_COLORS: Record<string, string> = {
  character: "border-indigo-500 bg-indigo-900/30",
  style: "border-purple-500 bg-purple-900/30",
  hybrid: "border-teal-500 bg-teal-900/30",
  custom: "border-gray-500 bg-gray-700/50",
};

const LEAK_RISK_COLORS = {
  low: "text-emerald-400",
  medium: "text-amber-400",
  high: "text-red-400",
  unknown: "text-gray-400",
};

const LEAK_RISK_LABELS = {
  low: "低リスク",
  medium: "中リスク",
  high: "高リスク",
  unknown: "未分析",
};

// ── Review Gate Modal ───────────────────────────────────────────────────────

function ReviewGateModal({
  report,
  loading,
  onConfirm,
  onCancel,
}: {
  report: DatasetReport | null;
  loading: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
      <div className="bg-gray-800 border border-gray-600 rounded-2xl w-full max-w-lg shadow-2xl">
        {/* Header */}
        <div className="flex items-center gap-3 px-6 py-4 border-b border-gray-700">
          <ShieldCheck size={20} className="text-indigo-400" />
          <h2 className="text-base font-bold text-gray-100">学習前チェックレポート</h2>
        </div>

        {/* Body */}
        <div className="px-6 py-5 space-y-4 max-h-[60vh] overflow-y-auto">
          {loading ? (
            <div className="flex items-center justify-center py-12 text-gray-400">
              <Loader2 size={24} className="animate-spin mr-3" />
              <span>レポートを生成中...</span>
            </div>
          ) : report ? (
            <>
              {/* Project info */}
              <div className="text-sm text-gray-400">
                <span className="text-gray-200 font-medium">{report.project_name}</span>
                {" "}({report.project_type})
              </div>

              {/* Stats grid */}
              <div className="grid grid-cols-2 gap-3">
                <div className="bg-gray-700/60 rounded-lg p-3">
                  <div className="text-xs text-gray-400 mb-1">データ数</div>
                  <div className="text-lg font-bold text-gray-100">{report.total_images} 枚</div>
                </div>
                <div className="bg-gray-700/60 rounded-lg p-3">
                  <div className="text-xs text-gray-400 mb-1">キャプション率</div>
                  <div className="text-lg font-bold text-gray-100">{report.caption_rate}%</div>
                  <div className="mt-1 h-1.5 bg-gray-600 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all ${
                        report.caption_rate >= 80 ? "bg-emerald-500" : "bg-amber-500"
                      }`}
                      style={{ width: `${report.caption_rate}%` }}
                    />
                  </div>
                </div>
                {report.quality_score !== null && (
                  <div className="bg-gray-700/60 rounded-lg p-3">
                    <div className="text-xs text-gray-400 mb-1">品質スコア</div>
                    <div
                      className={`text-lg font-bold ${
                        report.quality_score >= 70
                          ? "text-emerald-400"
                          : report.quality_score >= 50
                          ? "text-amber-400"
                          : "text-red-400"
                      }`}
                    >
                      {report.quality_score}/100
                    </div>
                  </div>
                )}
                <div className="bg-gray-700/60 rounded-lg p-3">
                  <div className="text-xs text-gray-400 mb-1">リーク分析</div>
                  <div className={`text-base font-bold ${LEAK_RISK_COLORS[report.leak_risk]}`}>
                    {LEAK_RISK_LABELS[report.leak_risk]}
                  </div>
                </div>
              </div>

              {/* Blockers */}
              {report.blockers.length > 0 && (
                <div className="bg-red-900/30 border border-red-700/50 rounded-lg p-3 space-y-1">
                  <div className="flex items-center gap-2 text-red-400 text-xs font-semibold mb-2">
                    <XCircle size={14} />
                    学習開始できません
                  </div>
                  {report.blockers.map((b, i) => (
                    <p key={i} className="text-xs text-red-300 flex items-start gap-2">
                      <span className="mt-0.5 shrink-0">•</span>
                      <span>{b}</span>
                    </p>
                  ))}
                </div>
              )}

              {/* Warnings */}
              {report.warnings.length > 0 && (
                <div className="bg-amber-900/20 border border-amber-700/40 rounded-lg p-3 space-y-1">
                  <div className="flex items-center gap-2 text-amber-400 text-xs font-semibold mb-2">
                    <AlertTriangle size={14} />
                    警告 ({report.warnings.length}件)
                  </div>
                  {report.warnings.slice(0, 6).map((w, i) => (
                    <p key={i} className="text-xs text-amber-300/80 flex items-start gap-2">
                      <span className="mt-0.5 shrink-0">⚠</span>
                      <span>{w}</span>
                    </p>
                  ))}
                  {report.warnings.length > 6 && (
                    <p className="text-xs text-amber-500">…他 {report.warnings.length - 6} 件</p>
                  )}
                </div>
              )}

              {/* All clear */}
              {report.blockers.length === 0 && report.warnings.length === 0 && (
                <div className="bg-emerald-900/20 border border-emerald-700/40 rounded-lg p-3 flex items-center gap-3">
                  <CheckCircle2 size={18} className="text-emerald-400" />
                  <span className="text-sm text-emerald-300">問題なし — 学習を開始できます</span>
                </div>
              )}

              {!report.analyzed && (
                <p className="text-xs text-gray-500 text-center">
                  ※ Dataset 分析未実施のため品質スコアは表示されていません。
                  Dataset タブで分析を実行すると詳細な警告が表示されます。
                </p>
              )}
            </>
          ) : (
            <p className="text-sm text-gray-400 text-center py-8">レポートの取得に失敗しました</p>
          )}
        </div>

        {/* Footer */}
        <div className="flex justify-end gap-3 px-6 py-4 border-t border-gray-700">
          <button
            onClick={onCancel}
            className="px-4 py-2 rounded-lg bg-gray-700 hover:bg-gray-600 text-gray-300 text-sm font-medium transition-colors"
          >
            キャンセル
          </button>
          <button
            onClick={onConfirm}
            disabled={loading || !report || !report.ready}
            className="flex items-center gap-2 px-5 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:bg-gray-700 disabled:text-gray-500 text-white text-sm font-semibold transition-colors"
          >
            <Play size={14} />
            承認して学習開始
          </button>
        </div>
      </div>
    </div>
  );
}

// ── Profile Card ────────────────────────────────────────────────────────────

function ProfileCard({
  preset,
  selected,
  onClick,
  onDelete,
}: {
  preset: Preset;
  selected: boolean;
  onClick: () => void;
  onDelete?: () => void;
}) {
  const type = preset.payload?.type ?? "custom";
  const colorCls = PROFILE_TYPE_COLORS[type] ?? PROFILE_TYPE_COLORS.custom;

  return (
    <button
      onClick={onClick}
      className={`relative text-left rounded-xl border-2 p-3 transition-all ${
        selected
          ? colorCls + " ring-2 ring-indigo-400 ring-offset-1 ring-offset-gray-800"
          : "border-gray-700 bg-gray-700/40 hover:border-gray-500"
      }`}
    >
      <div className="flex items-start justify-between gap-1">
        <div className="min-w-0">
          <div className="text-xs font-semibold text-gray-100 truncate">{preset.name}</div>
          <div className="text-xs text-gray-400 mt-0.5 line-clamp-1">
            {preset.payload?.description || type}
          </div>
        </div>
        {selected && (
          <CheckCircle2 size={14} className="text-indigo-400 shrink-0 mt-0.5" />
        )}
      </div>
      <div className="flex flex-wrap gap-x-3 gap-y-0.5 mt-2 text-xs text-gray-300">
        <span>rank {preset.payload?.rank}</span>
        <span>α{preset.payload?.alpha}</span>
        <span>{preset.payload?.epochs}ep</span>
        <span>{preset.payload?.resolution}px</span>
      </div>
      {onDelete && (
        <button
          onClick={(e) => { e.stopPropagation(); onDelete(); }}
          className="absolute top-2 right-2 p-1 rounded text-gray-500 hover:text-red-400 hover:bg-gray-600 transition-colors opacity-0 group-hover:opacity-100"
          title="削除"
        >
          <Trash2 size={12} />
        </button>
      )}
    </button>
  );
}

// ── メインコンポーネント ───────────────────────────────────────────────────

export default function Training({
  projects,
  selectedProjectId,
  selectedProject,
  onSelectProject,
  statuses,
  onRefresh,
  prompts,
  onPromptsChange,
  showError,
  showNotice,
}: Props) {
  // ── パラメータ状態 ────────────────────────────────────────────────────
  const [epochsText, setEpochsText] = useState("5");
  const [rankText, setRankText] = useState("16");
  const [alphaText, setAlphaText] = useState("4");
  const [trainRepeatsText, setTrainRepeatsText] = useState("5");
  const [saveEveryText, setSaveEveryText] = useState("1");
  const [resolutionText, setResolutionText] = useState("512");
  const [outputName, setOutputName] = useState("lora_output");
  const [baseCkpt, setBaseCkpt] = useState("");
  const [trainDir, setTrainDir] = useState("");
  const [regDir, setRegDir] = useState("");
  const [optimizerVal, setOptimizerVal] = useState("AdamW8bit");
  const [schedulerVal, setSchedulerVal] = useState("cosine_with_restarts");
  const [minSnrText, setMinSnrText] = useState("5");
  const [showParams, setShowParams] = useState(true);

  // ── プロファイル状態 ─────────────────────────────────────────────────
  const [presets, setPresets] = useState<Preset[]>([]);
  const [selectedPresetId, setSelectedPresetId] = useState<number | null>(null);
  const [showProfiles, setShowProfiles] = useState(true);

  // ── 学習モード ───────────────────────────────────────────────────────
  const [trainingMode, setTrainingMode] = useState<TrainingMode | null>(null);

  // ── Review Gate Modal 状態 ────────────────────────────────────────────
  const [reviewModalOpen, setReviewModalOpen] = useState(false);
  const [reportLoading, setReportLoading] = useState(false);
  const [report, setReport] = useState<DatasetReport | null>(null);

  // ── ログビューア ─────────────────────────────────────────────────────
  const [logLines, setLogLines] = useState<string[]>([]);
  const [showLog, setShowLog] = useState(false);
  const logRef = useRef<HTMLDivElement>(null);

  // ── プレビュー ───────────────────────────────────────────────────────
  const [timeline, setTimeline] = useState<PreviewSample[]>([]);
  const [datasetPreview, setDatasetPreview] = useState<DatasetPreview | null>(null);
  const [loadingTimeline, setLoadingTimeline] = useState(false);

  const currentStatus = selectedProject ? statuses[selectedProject.id] : undefined;
  const statusConf = STATUS_CONFIG[currentStatus?.status ?? "idle"] ?? STATUS_CONFIG.idle;
  const isTraining = currentStatus?.status === "training";
  const isPaused = currentStatus?.status === "paused";
  const isCompleted = currentStatus?.status === "completed";

  // ── 初期ロード ────────────────────────────────────────────────────────
  useEffect(() => {
    void loadPresets();
    void loadTrainingMode();
  }, []);

  useEffect(() => {
    if (!selectedProject) return;
    setTrainDir(selectedProject.dataset_dir);
    setRegDir(selectedProject.captions_dir);
    setOutputName(selectedProject.name || "lora_output");
    void loadTimeline(selectedProject.id);
    void loadDatasetPreview(selectedProject.id, selectedProject.dataset_dir);
  }, [selectedProject?.id]);

  // ── ログのポーリング（学習中のみ） ───────────────────────────────────
  useEffect(() => {
    if (!isTraining || !selectedProject || !showLog) return;
    const timer = setInterval(() => {
      void loadLogs(selectedProject.id);
    }, 2000);
    return () => clearInterval(timer);
  }, [isTraining, selectedProject?.id, showLog]);

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [logLines]);

  // ── データ読み込み ────────────────────────────────────────────────────

  async function loadPresets() {
    try {
      const data = await apiGet<Preset[]>("/presets");
      setPresets(data);
    } catch {
      setPresets([]);
    }
  }

  async function loadTrainingMode() {
    try {
      const data = await apiGet<TrainingMode>("/training/mode");
      setTrainingMode(data);
    } catch {
      setTrainingMode(null);
    }
  }

  async function loadTimeline(projectId: number) {
    setLoadingTimeline(true);
    try {
      const r = await apiGet<{ timeline: PreviewSample[] }>(`/previews/${projectId}`);
      setTimeline(r.timeline || []);
    } catch {
      setTimeline([]);
    } finally {
      setLoadingTimeline(false);
    }
  }

  async function loadDatasetPreview(projectId: number, dir: string) {
    try {
      const q = `project_id=${projectId}&train_data_dir=${encodeURIComponent(dir)}`;
      const r = await apiGet<DatasetPreview>(`/training/dataset-preview?${q}`);
      setDatasetPreview(r);
    } catch {
      setDatasetPreview(null);
    }
  }

  async function loadLogs(projectId: number) {
    try {
      const r = await apiGet<{ lines: string[] }>(`/training/logs/${projectId}`);
      setLogLines(r.lines || []);
    } catch {
      // ignore
    }
  }

  // ── プロファイル適用 ──────────────────────────────────────────────────

  function applyPreset(preset: Preset) {
    const p = preset.payload;
    if (p.epochs != null) setEpochsText(String(p.epochs));
    if (p.rank != null) setRankText(String(p.rank));
    if (p.alpha != null) setAlphaText(String(p.alpha));
    if (p.repeats != null) setTrainRepeatsText(String(p.repeats));
    if (p.save_every_n_epochs != null) setSaveEveryText(String(p.save_every_n_epochs));
    if (p.resolution != null) setResolutionText(String(p.resolution));
    if (p.output_name) setOutputName(p.output_name);
    if (p.optimizer) setOptimizerVal(p.optimizer);
    if (p.scheduler) setSchedulerVal(p.scheduler);
    if (p.min_snr_gamma != null) setMinSnrText(String(p.min_snr_gamma));
    else setMinSnrText("");
    setSelectedPresetId(preset.id);
    showNotice(`プロファイル「${preset.name}」を適用しました`);
  }

  async function deletePreset(preset: Preset) {
    try {
      await apiDelete(`/presets/${preset.id}`);
      setPresets((prev) => prev.filter((p) => p.id !== preset.id));
      if (selectedPresetId === preset.id) setSelectedPresetId(null);
      showNotice(`プロファイル「${preset.name}」を削除しました`);
    } catch (e) {
      showError(String(e));
    }
  }

  // ── Review Gate ───────────────────────────────────────────────────────

  async function openReviewGate() {
    if (!selectedProject) return;
    setReviewModalOpen(true);
    setReport(null);
    setReportLoading(true);
    try {
      const data = await apiGet<DatasetReport>(`/dataset/report/${selectedProject.id}`);
      setReport(data);
    } catch (e) {
      showError(`レポート取得失敗: ${String(e)}`);
      setReviewModalOpen(false);
    } finally {
      setReportLoading(false);
    }
  }

  async function confirmAndStartTraining() {
    setReviewModalOpen(false);
    await startTraining();
  }

  // ── 学習操作 ──────────────────────────────────────────────────────────

  async function startTraining() {
    if (!selectedProject) return;
    try {
      const result = await apiPost<{ mode: string; message: string }>("/training/start", {
        project_id: selectedProject.id,
        preset_id: selectedPresetId,
        epochs: parseIntOr(epochsText, 5, 1),
        repeats: parseIntOr(trainRepeatsText, 5, 1),
        alpha: parseFloatOr(alphaText, 4, 0.1),
        rank: parseIntOr(rankText, 16, 1),
        save_every_n_epochs: parseIntOr(saveEveryText, 1, 1),
        output_name: outputName,
        base_checkpoint_path: baseCkpt,
        train_data_dir: trainDir,
        reg_data_dir: regDir,
        resolution: parseIntOr(resolutionText, 512, 256),
        optimizer: optimizerVal,
        scheduler: schedulerVal,
        min_snr_gamma: minSnrText.trim() ? parseIntOr(minSnrText, 5, 0) : null,
      });
      showNotice(result.message || "学習を開始しました");
      await onRefresh();
      void loadTrainingMode();
    } catch (e) {
      showError(`学習開始失敗: ${String(e)}`);
    }
  }

  async function trainingAction(path: string, msg: string) {
    if (!selectedProject) return;
    try {
      await apiPost(path, { project_id: selectedProject.id });
      showNotice(msg);
      await onRefresh();
    } catch (e) {
      showError(String(e));
    }
  }

  async function savePreviewPrompts() {
    try {
      await apiPost("/settings/preview-prompts", prompts, "PUT");
      showNotice("プレビュー用プロンプト保存");
    } catch (e) {
      showError(String(e));
    }
  }

  // ── プロジェクト未選択時 ──────────────────────────────────────────────

  if (!selectedProject) {
    return (
      <div className="space-y-4 max-w-4xl">
        <h2 className="text-xl font-bold text-gray-100 mb-1">学習制御</h2>
        <div className="bg-gray-800 border border-gray-700 rounded-xl p-6">
          <p className="text-sm text-gray-400 mb-4">対象プロジェクト:</p>
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
            {projects.map((p) => (
              <button
                key={p.id}
                onClick={() => onSelectProject(p.id)}
                className="bg-gray-700 hover:bg-gray-600 border border-gray-600 rounded-lg p-3 text-left transition-colors"
              >
                <div className="text-sm font-medium text-gray-200">{p.name}</div>
                <div className="text-xs text-gray-400">{p.project_type}</div>
              </button>
            ))}
            {projects.length === 0 && (
              <p className="text-gray-500 text-sm">プロジェクトがありません</p>
            )}
          </div>
        </div>
      </div>
    );
  }

  const totalSteps = currentStatus?.total_steps ?? 0;
  const doneSteps = currentStatus?.done_steps ?? 0;
  const progress = currentStatus?.progress_percent ?? 0;
  const epoch = currentStatus?.epoch ?? 0;
  const totalEpochs = currentStatus?.total_epochs ?? 0;
  const stepsPerEpoch = currentStatus?.steps_per_epoch ?? 0;

  return (
    <>
      {/* Review Gate Modal */}
      {reviewModalOpen && (
        <ReviewGateModal
          report={report}
          loading={reportLoading}
          onConfirm={() => void confirmAndStartTraining()}
          onCancel={() => setReviewModalOpen(false)}
        />
      )}

      <div className="grid grid-cols-1 gap-4 max-w-5xl xl:grid-cols-[1fr,360px]">
        {/* ── 左カラム ── */}
        <div className="space-y-4">
          {/* Status Card */}
          <div className="bg-gray-800 border border-gray-700 rounded-xl p-5">
            <div className="flex items-center justify-between mb-4">
              <div>
                <div className="flex items-center gap-2">
                  <h2 className="text-base font-bold text-gray-100">{selectedProject.name}</h2>
                  {/* Kohya Mode Badge */}
                  {trainingMode && (
                    <span
                      className={`inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full font-medium ${
                        trainingMode.kohya_ready
                          ? "bg-emerald-900/50 text-emerald-300 border border-emerald-700/50"
                          : "bg-gray-700 text-gray-400 border border-gray-600"
                      }`}
                      title={trainingMode.message}
                    >
                      {trainingMode.kohya_ready ? (
                        <><Zap size={10} /> kohya_ss</>
                      ) : (
                        <><Cpu size={10} /> simulation</>
                      )}
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-2 mt-0.5">
                  <span className={`w-2 h-2 rounded-full ${statusConf.dot}`} />
                  <span className={`text-sm ${statusConf.color}`}>{statusConf.label}</span>
                  {(currentStatus as any)?.loss != null && (
                    <span className="text-xs text-gray-500 font-mono">
                      loss={Number((currentStatus as any).loss).toFixed(4)}
                    </span>
                  )}
                </div>
              </div>
              <select
                value={selectedProject.id}
                onChange={(e) => onSelectProject(Number(e.target.value))}
                className="bg-gray-700 border border-gray-600 text-gray-300 rounded-lg px-2 py-1.5 text-xs focus:outline-none"
              >
                {projects.map((p) => (
                  <option key={p.id} value={p.id}>{p.name}</option>
                ))}
              </select>
            </div>

            {/* Metrics */}
            <div className="grid grid-cols-4 gap-3 mb-4">
              <Metric label="状態" value={statusConf.label} icon={<BarChart3 size={14} />} />
              <Metric label="Epoch" value={`${epoch}/${totalEpochs}`} icon={<Layers size={14} />} />
              <Metric label="Step" value={`${doneSteps}/${totalSteps}`} icon={<BarChart3 size={14} />} />
              <Metric label="残り" value={etaText(currentStatus?.eta_seconds)} icon={<Clock size={14} />} />
            </div>

            {/* Overall progress bar */}
            <div className="mb-2">
              <div className="flex items-center justify-between text-xs text-gray-400 mb-1">
                <span>全体進捗</span>
                <span>{progress.toFixed(1)}%</span>
              </div>
              <div className="h-3 bg-gray-700 rounded-full overflow-hidden">
                <div
                  className={[
                    "h-full rounded-full transition-all duration-500",
                    isTraining ? "bg-amber-500" : isCompleted ? "bg-emerald-500" : "bg-indigo-500",
                  ].join(" ")}
                  style={{ width: `${Math.min(100, Math.max(0, progress))}%` }}
                />
              </div>
            </div>

            {/* Per-epoch mini bars */}
            {totalEpochs > 0 && (
              <div className="flex gap-0.5 h-2">
                {Array.from({ length: totalEpochs }).map((_, i) => {
                  const filled = i < epoch;
                  const current = i === epoch && isTraining;
                  const stepProgress =
                    current && stepsPerEpoch > 0
                      ? ((currentStatus?.done_steps ?? 0) % stepsPerEpoch) / stepsPerEpoch
                      : 0;
                  return (
                    <div key={i} className="flex-1 bg-gray-700 rounded-sm overflow-hidden relative">
                      <div
                        className={[
                          "absolute inset-y-0 left-0 rounded-sm transition-all",
                          filled ? "bg-emerald-500 w-full" : current ? "bg-amber-500" : "w-0",
                        ].join(" ")}
                        style={current ? { width: `${stepProgress * 100}%` } : undefined}
                      />
                    </div>
                  );
                })}
              </div>
            )}
            {totalEpochs > 0 && (
              <div className="text-xs text-gray-500 mt-1">
                Epoch {epoch}/{totalEpochs} — 各バーが 1 epoch
              </div>
            )}

            {/* Control buttons */}
            <div className="flex flex-wrap gap-2 mt-4">
              <button
                onClick={() => void openReviewGate()}
                disabled={isTraining}
                className="flex items-center gap-2 bg-indigo-600 hover:bg-indigo-500 disabled:bg-gray-700 disabled:text-gray-500 text-white rounded-lg px-4 py-2 text-sm font-semibold transition-colors"
              >
                {isTraining ? <Loader2 size={16} className="animate-spin" /> : <Play size={16} />}
                学習開始
              </button>
              <button
                onClick={() => void trainingAction("/training/stop-at-epoch", "Epoch区切りで停止予約")}
                disabled={!isTraining}
                className="flex items-center gap-2 bg-blue-900 hover:bg-blue-800 disabled:bg-gray-700 disabled:text-gray-500 text-blue-200 rounded-lg px-3 py-2 text-sm font-medium transition-colors"
              >
                <PauseCircle size={16} />
                Epoch停止
              </button>
              <button
                onClick={() => void trainingAction("/training/stop-now", "即時停止")}
                disabled={!isTraining}
                className="flex items-center gap-2 bg-red-900 hover:bg-red-800 disabled:bg-gray-700 disabled:text-gray-500 text-red-200 rounded-lg px-3 py-2 text-sm font-medium transition-colors"
              >
                <Square size={16} />
                即停止
              </button>
              <button
                onClick={() =>
                  void trainingAction("/training/resume", "学習を再開").then(() =>
                    void loadTimeline(selectedProject.id)
                  )
                }
                disabled={isTraining || isCompleted}
                className="flex items-center gap-2 bg-emerald-900 hover:bg-emerald-800 disabled:bg-gray-700 disabled:text-gray-500 text-emerald-200 rounded-lg px-3 py-2 text-sm font-medium transition-colors"
              >
                <RotateCcw size={16} />
                再開
              </button>
            </div>
          </div>

          {/* ── Training Profiles Card ── */}
          <div className="bg-gray-800 border border-gray-700 rounded-xl overflow-hidden">
            <button
              className="w-full flex items-center justify-between px-5 py-4 text-sm font-semibold text-gray-200 hover:bg-gray-750 transition-colors"
              onClick={() => setShowProfiles((p) => !p)}
            >
              <span className="flex items-center gap-2">
                <BookOpen size={15} className="text-indigo-400" />
                学習プロファイル
                {selectedPresetId && (
                  <span className="text-xs font-normal text-indigo-400">
                    ({presets.find((p) => p.id === selectedPresetId)?.name})
                  </span>
                )}
              </span>
              {showProfiles ? <ChevronUp size={16} className="text-gray-400" /> : <ChevronDown size={16} className="text-gray-400" />}
            </button>
            {showProfiles && (
              <div className="px-5 pb-5 border-t border-gray-700">
                <div className="grid grid-cols-2 gap-2 mt-4 sm:grid-cols-3">
                  {presets.map((preset) => (
                    <div key={preset.id} className="group relative">
                      <ProfileCard
                        preset={preset}
                        selected={selectedPresetId === preset.id}
                        onClick={() => applyPreset(preset)}
                        onDelete={
                          preset.payload?.type === "custom"
                            ? () => void deletePreset(preset)
                            : undefined
                        }
                      />
                    </div>
                  ))}
                </div>
                {presets.length === 0 && (
                  <p className="text-xs text-gray-500 mt-4 text-center">プロファイルがありません</p>
                )}
                <p className="text-xs text-gray-500 mt-3">
                  プロファイルをクリックするとパラメータが自動適用されます。
                </p>
              </div>
            )}
          </div>

          {/* ── Parameters Card ── */}
          <div className="bg-gray-800 border border-gray-700 rounded-xl overflow-hidden">
            <button
              className="w-full flex items-center justify-between px-5 py-4 text-sm font-semibold text-gray-200 hover:bg-gray-750 transition-colors"
              onClick={() => setShowParams((p) => !p)}
            >
              <span>KOHYA パラメータ</span>
              {showParams ? <ChevronUp size={16} className="text-gray-400" /> : <ChevronDown size={16} className="text-gray-400" />}
            </button>
            {showParams && (
              <div className="px-5 pb-5 space-y-4 border-t border-gray-700">
                <div className="grid grid-cols-2 gap-3 mt-4 sm:grid-cols-4">
                  <ParamInput label="epochs" value={epochsText} onChange={setEpochsText} />
                  <ParamInput label="repeats" value={trainRepeatsText} onChange={setTrainRepeatsText} />
                  <ParamInput label="rank" value={rankText} onChange={setRankText} />
                  <ParamInput label="alpha" value={alphaText} onChange={setAlphaText} inputMode="decimal" />
                  <ParamInput label="save_every_n" value={saveEveryText} onChange={setSaveEveryText} />
                  <ParamInput label="resolution" value={resolutionText} onChange={setResolutionText} />
                  <ParamInput label="min_snr_gamma" value={minSnrText} onChange={setMinSnrText} placeholder="5" />
                  <ParamInput label="output_name" value={outputName} onChange={setOutputName} inputMode="text" wide />
                </div>

                {/* Optimizer / Scheduler selectors */}
                <div className="grid grid-cols-2 gap-3 pt-2 border-t border-gray-700">
                  <label className="block">
                    <span className="text-xs text-gray-400 mb-1 block">Optimizer</span>
                    <select
                      value={optimizerVal}
                      onChange={(e) => setOptimizerVal(e.target.value)}
                      className="w-full bg-gray-700 border border-gray-600 text-gray-100 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-indigo-500"
                    >
                      <option value="AdamW8bit">AdamW8bit</option>
                      <option value="AdamW">AdamW</option>
                      <option value="Lion8bit">Lion8bit</option>
                      <option value="Adafactor">Adafactor</option>
                      <option value="DAdaptAdam">DAdaptAdam</option>
                      <option value="Prodigy">Prodigy</option>
                    </select>
                  </label>
                  <label className="block">
                    <span className="text-xs text-gray-400 mb-1 block">LR Scheduler</span>
                    <select
                      value={schedulerVal}
                      onChange={(e) => setSchedulerVal(e.target.value)}
                      className="w-full bg-gray-700 border border-gray-600 text-gray-100 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-indigo-500"
                    >
                      <option value="cosine_with_restarts">cosine_with_restarts</option>
                      <option value="cosine">cosine</option>
                      <option value="linear">linear</option>
                      <option value="constant">constant</option>
                      <option value="constant_with_warmup">constant_with_warmup</option>
                      <option value="polynomial">polynomial</option>
                    </select>
                  </label>
                </div>

                <div className="pt-2 border-t border-gray-700">
                  <div className="grid gap-3">
                    <ParamInput
                      label="base checkpoint"
                      value={baseCkpt}
                      onChange={setBaseCkpt}
                      inputMode="text"
                      placeholder="model.safetensors"
                    />
                    <ParamInput
                      label="train dir"
                      value={trainDir}
                      onChange={(v) => {
                        setTrainDir(v);
                        void loadDatasetPreview(selectedProject.id, v);
                      }}
                      inputMode="text"
                    />
                    <ParamInput label="reg dir" value={regDir} onChange={setRegDir} inputMode="text" />
                  </div>
                </div>

                {/* Preview prompts */}
                <div className="pt-2 border-t border-gray-700">
                  <h4 className="text-xs font-semibold text-gray-400 mb-3">Preview Prompts</h4>
                  <div className="grid gap-2">
                    <label className="block">
                      <span className="text-xs text-gray-400 mb-1 block">Positive</span>
                      <input
                        value={prompts.positive_prompt}
                        onChange={(e) => onPromptsChange({ ...prompts, positive_prompt: e.target.value })}
                        className="w-full bg-gray-700 border border-gray-600 text-gray-100 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-emerald-500"
                      />
                    </label>
                    <label className="block">
                      <span className="text-xs text-gray-400 mb-1 block">Negative</span>
                      <input
                        value={prompts.negative_prompt}
                        onChange={(e) => onPromptsChange({ ...prompts, negative_prompt: e.target.value })}
                        className="w-full bg-gray-700 border border-gray-600 text-gray-100 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-red-500"
                      />
                    </label>
                    <button
                      onClick={() => void savePreviewPrompts()}
                      className="self-start bg-gray-700 hover:bg-gray-600 text-gray-300 rounded-lg px-3 py-1.5 text-xs font-medium transition-colors"
                    >
                      保存
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* ── Training Log Viewer ── */}
          <div className="bg-gray-800 border border-gray-700 rounded-xl overflow-hidden">
            <button
              className="w-full flex items-center justify-between px-5 py-4 text-sm font-semibold text-gray-200 hover:bg-gray-750 transition-colors"
              onClick={() => {
                setShowLog((s) => !s);
                if (!showLog && selectedProject) void loadLogs(selectedProject.id);
              }}
            >
              <span className="flex items-center gap-2">
                <Terminal size={15} className="text-emerald-400" />
                学習ログ
                {isTraining && (
                  <span className="inline-flex items-center gap-1 text-xs text-amber-400">
                    <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse" />
                    LIVE
                  </span>
                )}
              </span>
              {showLog ? <ChevronUp size={16} className="text-gray-400" /> : <ChevronDown size={16} className="text-gray-400" />}
            </button>
            {showLog && (
              <div className="border-t border-gray-700">
                <div
                  ref={logRef}
                  className="bg-gray-900 font-mono text-xs text-emerald-300 p-4 h-48 overflow-y-auto leading-5"
                >
                  {logLines.length === 0 ? (
                    <span className="text-gray-500">ログがありません（学習を開始すると表示されます）</span>
                  ) : (
                    logLines.map((line, i) => (
                      <div key={i} className="whitespace-pre-wrap break-all">
                        {line}
                      </div>
                    ))
                  )}
                </div>
                <div className="px-4 py-2 flex justify-end">
                  <button
                    onClick={() => selectedProject && void loadLogs(selectedProject.id)}
                    className="text-xs text-gray-500 hover:text-gray-300 flex items-center gap-1 transition-colors"
                  >
                    <RotateCcw size={11} />
                    更新
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* ── Preview Timeline ── */}
          <div className="bg-gray-800 border border-gray-700 rounded-xl overflow-hidden">
            <div className="flex items-center justify-between px-5 py-4 border-b border-gray-700">
              <h3 className="text-sm font-semibold text-gray-200">プレビュー履歴</h3>
              <button
                onClick={() => void loadTimeline(selectedProject.id)}
                className="p-1.5 rounded hover:bg-gray-700 text-gray-500 hover:text-gray-300 transition-colors"
                title="更新"
              >
                {loadingTimeline ? (
                  <Loader2 size={14} className="animate-spin" />
                ) : (
                  <RotateCcw size={14} />
                )}
              </button>
            </div>
            <div className="p-4">
              {loadingTimeline ? (
                <div className="flex items-center justify-center py-8 text-gray-500">
                  <Loader2 size={20} className="animate-spin mr-2" />
                  <span className="text-sm">読み込み中...</span>
                </div>
              ) : timeline.length === 0 ? (
                <div className="text-center py-8">
                  <ImageOff size={32} className="text-gray-600 mx-auto mb-2" />
                  <p className="text-sm text-gray-500">
                    まだプレビューがありません。
                    <br />
                    学習を実行すると Epoch ごとに追加されます。
                  </p>
                </div>
              ) : (
                <div className="space-y-4">
                  {timeline.map((t) => (
                    <div key={t.checkpoint_id} className="border border-gray-700 rounded-lg overflow-hidden">
                      <div className="bg-gray-750 px-3 py-2 flex items-center justify-between">
                        <span className="text-xs font-semibold text-gray-300">Epoch {t.epoch}</span>
                        <span className="text-xs text-gray-500">
                          {t.created_at ? new Date(t.created_at).toLocaleString("ja-JP") : ""}
                        </span>
                      </div>
                      <div className="grid grid-cols-4 gap-1 p-2">
                        {Object.entries(t.sample_previews || {}).map(([slot, img]) => (
                          <div key={slot} className="relative group">
                            <div className="aspect-square bg-gray-700 rounded overflow-hidden">
                              <img
                                src={img}
                                alt={`epoch ${t.epoch} - ${slot}`}
                                className="w-full h-full object-cover"
                              />
                            </div>
                            <div className="absolute bottom-0 left-0 right-0 bg-black/70 text-xs text-gray-300 text-center py-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
                              {slot}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* ── 右カラム ── */}
        <div className="space-y-4">
          {/* Dataset preview */}
          <div className="bg-gray-800 border border-gray-700 rounded-xl overflow-hidden">
            <div className="px-4 py-3 border-b border-gray-700">
              <h3 className="text-sm font-semibold text-gray-200">Dataset サムネイル</h3>
            </div>
            <div className="p-4">
              {datasetPreview?.thumbnail_url ? (
                <>
                  <img
                    src={datasetPreview.thumbnail_url}
                    alt="dataset preview"
                    className="w-full aspect-square object-cover rounded-lg border border-gray-700"
                  />
                  <p className="text-xs text-gray-500 mt-2 break-all">{datasetPreview.image_path}</p>
                </>
              ) : (
                <div className="aspect-square bg-gray-700 rounded-lg flex flex-col items-center justify-center">
                  <ImageIcon size={32} className="text-gray-600 mb-2" />
                  <p className="text-xs text-gray-500 text-center">train dir に画像が見つかりません</p>
                </div>
              )}
            </div>
          </div>

          {/* Training mode info */}
          {trainingMode && (
            <div className={`rounded-xl p-4 border ${
              trainingMode.kohya_ready
                ? "bg-emerald-900/20 border-emerald-700/40"
                : "bg-gray-700/40 border-gray-600"
            }`}>
              <div className="flex items-center gap-2 mb-2">
                {trainingMode.kohya_ready ? (
                  <Zap size={14} className="text-emerald-400" />
                ) : (
                  <Cpu size={14} className="text-gray-400" />
                )}
                <span className="text-xs font-semibold text-gray-200">
                  {trainingMode.kohya_ready ? "kohya_ss 接続済み" : "シミュレーションモード"}
                </span>
              </div>
              <p className="text-xs text-gray-400">
                {trainingMode.kohya_ready
                  ? `${trainingMode.kohya_root}`
                  : "Integrations でkohya_ssのパスを設定すると実学習が有効になります。"}
              </p>
            </div>
          )}

          {/* Training summary (during/after training) */}
          {(isTraining || isPaused || isCompleted) && (
            <div className="bg-gray-800 border border-gray-700 rounded-xl p-4">
              <h3 className="text-sm font-semibold text-gray-200 mb-3">実行中の設定</h3>
              <dl className="space-y-2 text-xs">
                <Row label="ステータス" value={statusConf.label} />
                <Row label="Epoch" value={`${epoch} / ${totalEpochs}`} />
                <Row label="Step" value={`${doneSteps} / ${totalSteps}`} />
                <Row label="残り時間" value={etaText(currentStatus?.eta_seconds)} />
                {currentStatus?.stop_mode && (
                  <Row
                    label="停止モード"
                    value={currentStatus.stop_mode === "epoch" ? "Epoch停止予約" : "即時停止"}
                  />
                )}
                {(currentStatus as any)?.mode && (
                  <Row label="実行モード" value={(currentStatus as any).mode} />
                )}
              </dl>
            </div>
          )}
        </div>
      </div>
    </>
  );
}
