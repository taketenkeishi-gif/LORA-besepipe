import { useState, useEffect, useRef, useMemo, memo } from "react";
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
  FolderOpen,
  Search,
  Timer,
  Gauge,
  Settings,
  Tag,
  X,
  RefreshCw,
  ListOrdered,
} from "lucide-react";
import { apiPost, apiGet, apiDelete } from "../lib/api";

// かな→ローマ字変換（output_name の自動命名用）。簡易ヘボン式。
const _ROMAJI: Record<string, string> = {
  きゃ:"kya",きゅ:"kyu",きょ:"kyo",しゃ:"sha",しゅ:"shu",しょ:"sho",ちゃ:"cha",ちゅ:"chu",ちょ:"cho",
  にゃ:"nya",にゅ:"nyu",にょ:"nyo",ひゃ:"hya",ひゅ:"hyu",ひょ:"hyo",みゃ:"mya",みゅ:"myu",みょ:"myo",
  りゃ:"rya",りゅ:"ryu",りょ:"ryo",ぎゃ:"gya",ぎゅ:"gyu",ぎょ:"gyo",じゃ:"ja",じゅ:"ju",じょ:"jo",
  びゃ:"bya",びゅ:"byu",びょ:"byo",ぴゃ:"pya",ぴゅ:"pyu",ぴょ:"pyo",
  あ:"a",い:"i",う:"u",え:"e",お:"o",か:"ka",き:"ki",く:"ku",け:"ke",こ:"ko",
  さ:"sa",し:"shi",す:"su",せ:"se",そ:"so",た:"ta",ち:"chi",つ:"tsu",て:"te",と:"to",
  な:"na",に:"ni",ぬ:"nu",ね:"ne",の:"no",は:"ha",ひ:"hi",ふ:"fu",へ:"he",ほ:"ho",
  ま:"ma",み:"mi",む:"mu",め:"me",も:"mo",や:"ya",ゆ:"yu",よ:"yo",
  ら:"ra",り:"ri",る:"ru",れ:"re",ろ:"ro",わ:"wa",を:"o",ん:"n",
  が:"ga",ぎ:"gi",ぐ:"gu",げ:"ge",ご:"go",ざ:"za",じ:"ji",ず:"zu",ぜ:"ze",ぞ:"zo",
  だ:"da",ぢ:"ji",づ:"zu",で:"de",ど:"do",ば:"ba",び:"bi",ぶ:"bu",べ:"be",ぼ:"bo",
  ぱ:"pa",ぴ:"pi",ぷ:"pu",ぺ:"pe",ぽ:"po",ー:"",っ:"",
};

function toRomaji(input: string): string {
  // カタカナ→ひらがなに正規化
  const hira = input.replace(/[ァ-ヶ]/g, (c) => String.fromCharCode(c.charCodeAt(0) - 0x60));
  let out = "";
  let i = 0;
  let pendingSokuon = false;
  while (i < hira.length) {
    const two = hira.slice(i, i + 2);
    const one = hira[i];
    if (one === "っ") { pendingSokuon = true; i += 1; continue; }
    let r = "";
    if (_ROMAJI[two] !== undefined) { r = _ROMAJI[two]; i += 2; }
    else if (_ROMAJI[one] !== undefined) { r = _ROMAJI[one]; i += 1; }
    else if (/[a-zA-Z0-9]/.test(one)) { r = one; i += 1; }
    else { i += 1; continue; }  // 変換不能文字はスキップ
    if (pendingSokuon && r) { r = r[0] + r; pendingSokuon = false; }
    out += r;
  }
  return out.toLowerCase().replace(/[^a-z0-9]+/g, "") || input.toLowerCase().replace(/[^a-z0-9]+/g, "");
}
import { parseIntOr, parseFloatOr, etaText, durationText } from "../lib/utils";
import type {
  Project,
  TrainingStatus,
  PreviewPrompts,
  PreviewPromptItem,
  PreviewSample,
  Preset,
  DatasetReport,
  TrainingMode,
  ResourceStats,
  TrainingEstimate,
  CaptionItem,
  ModelSpec,
  TrainingQueueState,
  PreflightResult,
} from "../types";

type Props = {
  projects: Project[];
  selectedProjectId: number | null;
  selectedProject: Project | null;
  onSelectProject: (id: number | null) => void;
  currentStatus: TrainingStatus | undefined;
  onRefresh: () => Promise<void>;
  prompts: PreviewPrompts;
  onPromptsChange: (p: PreviewPrompts) => void;
  showError: (msg: string) => void;
  showNotice: (msg: string) => void;
};

// ── サブコンポーネント ────────────────────────────────────────────────────

// プレビュープロンプトの既定値（backend services/preview_prompts.py と一致させる）
const DEFAULT_QUALITY_PROMPT = "masterpiece, best quality, very aesthetic, absurdres, anime coloring";
const DEFAULT_FEATURE_PROMPT = "1girl, solo";
const DEFAULT_NEGATIVE_PROMPT =
  "lowres, worst quality, jpeg artifacts, blurry, noise, grain, film grain, " +
  "unfinished, displeasing, artistic error, text, watermark, signature, username, " +
  "scan, abstract, error, cropped, split_window, (bad anatomy:1.25), " +
  "(anatomical inconsistency:1.2), (wrong limb proportion:1.35), " +
  "(incorrect body ratio:1.3), (deformed limbs:1.35), (disconnected limbs:1.3), " +
  "(misaligned joints:1.3), (broken anatomy:1.25), (multiple views:1.3), " +
  "(bad hands:1.25), (malformed hands:1.3), (distorted fingers:1.3), " +
  "(extra face:1.3), (multiple face:1.3), (perspective error:1.2), " +
  "(flat body depth:1.15), (distorted foreshortening:1.2), standard, " +
  "conventional, extra fingers, low quality";

// 2つの timeline が同一内容かを判定（同一なら state 更新をスキップして参照を維持）
function sameTimeline(a: PreviewSample[], b: PreviewSample[]): boolean {
  if (a === b) return true;
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    const x = a[i];
    const y = b[i];
    if (x.checkpoint_id !== y.checkpoint_id || x.epoch !== y.epoch) return false;
    const xs = x.sample_previews || {};
    const ys = y.sample_previews || {};
    const xk = Object.keys(xs);
    const yk = Object.keys(ys);
    if (xk.length !== yk.length) return false;
    for (const k of xk) {
      if (xs[k] !== ys[k]) return false;
    }
  }
  return true;
}

// timeline はエポック完了時のみ変化するため memo で step 更新時の再描画を防ぐ
const PreviewTimeline = memo(function PreviewTimeline({
  timeline,
  loadingTimeline,
  promptLabels,
  onZoom,
  onRetry,
  retryingCheckpointId,
}: {
  timeline: PreviewSample[];
  loadingTimeline: boolean;
  promptLabels: string[];
  onZoom: (src: string) => void;
  onRetry?: (checkpointId: number) => void;
  retryingCheckpointId?: number | null;
}) {
  if (loadingTimeline) {
    return (
      <div className="flex items-center justify-center py-8 text-gray-500">
        <span className="text-sm">読み込み中...</span>
      </div>
    );
  }
  if (timeline.length === 0) {
    return (
      <div className="text-center py-8">
        <p className="text-sm text-gray-500">
          まだプレビューがありません。
          <br />
          学習を実行すると Epoch ごとに追加されます。
        </p>
      </div>
    );
  }
  return (
    <div className="space-y-4">
      {timeline.map((t) => {
        const js = t.preview_job_summary;
        // Preview Job SSOTのexpected/succeeded/failed/runningから状態文言を導出する。
        // 「1枚生成できたから完了」という表示は絶対にしない(succeeded==expectedのときのみ「完了」)。
        const jobLabel =
          js && js.expected > 0
            ? js.succeeded === js.expected
              ? `Preview ${js.succeeded}/${js.expected} 完了`
              : js.failed > 0
                ? `Preview ${js.succeeded}/${js.expected} ${js.failed}件失敗`
                : js.running > 0
                  ? `Preview ${js.succeeded}/${js.expected} 生成中`
                  : `Preview ${js.succeeded}/${js.expected} 待機中`
            : null;
        const jobLabelColor =
          js && js.expected > 0 && js.succeeded === js.expected
            ? "text-emerald-400"
            : js && js.failed > 0
              ? "text-red-400"
              : "text-cyan-400";
        return (
        <div key={t.checkpoint_id} className="border border-gray-700 rounded-lg overflow-hidden">
          <div className="bg-gray-750 px-3 py-2 flex items-center justify-between gap-2">
            <span className="text-xs font-semibold text-gray-300 shrink-0">Epoch {t.epoch}</span>
            {jobLabel && (
              <span className={`text-[11px] font-medium ${jobLabelColor} truncate`} title={jobLabel}>
                {jobLabel}
              </span>
            )}
            <span className="text-xs text-gray-500 shrink-0">
              {t.created_at ? new Date(t.created_at).toLocaleString("ja-JP") : ""}
            </span>
          </div>
          {(t.validation_status === "preview_failed" || t.validation_status === "invalid" || t.validation_status === "preview_partial") && (
            <div className="px-3 py-2 bg-red-950/40 border-t border-red-900/50 flex items-start justify-between gap-2">
              <div className="min-w-0">
                <p className="text-xs text-red-400 font-medium">
                  {t.validation_status === "invalid"
                    ? "Artifact検証失敗"
                    : t.validation_status === "preview_partial"
                      ? `Preview一部失敗（${js?.succeeded ?? 0}/${js?.expected ?? 0}件成功）`
                      : "Preview生成失敗"}
                </p>
                {t.validation_detail && (
                  <p className="text-[11px] text-red-300/80 mt-0.5 break-words">{t.validation_detail}</p>
                )}
              </div>
              {onRetry &&
                (t.validation_status === "preview_failed" || t.validation_status === "preview_partial") && (
                <button
                  onClick={() => onRetry(t.checkpoint_id)}
                  disabled={retryingCheckpointId === t.checkpoint_id}
                  className="shrink-0 flex items-center gap-1 bg-red-900 hover:bg-red-800 disabled:bg-gray-700 disabled:text-gray-500 text-red-200 rounded px-2 py-1 text-[11px] font-medium transition-colors"
                  title="失敗したPreviewのみ再試行する(成功済みは再生成されません)"
                >
                  {retryingCheckpointId === t.checkpoint_id ? (
                    <Loader2 size={11} className="animate-spin" />
                  ) : (
                    <RefreshCw size={11} />
                  )}
                  失敗分のみ再試行
                </button>
              )}
            </div>
          )}
          {t.validation_status === "validated" && Object.keys(t.sample_previews || {}).length === 0 && (
            // Artifact検証は完了しているが、Preview画像はまだ1枚も届いていない状態。
            // Krea2等musubi系はTraining完了後ではなくEpoch単位でPreview生成を裏側で
            // 実行するため、次のEpochの学習が進行中でも同時にこの状態になり得る
            // (実Runtime監査で確認: Epoch 2学習中にEpoch 1のPreview生成がComfyUI側で
            // 実行中だった)。何も表示しないと「壊れている/放置されている」ように
            // 見えるため、進行中であることを明示する。
            <div className="px-3 py-2 bg-cyan-950/30 border-t border-cyan-900/40 flex items-center gap-2">
              <Loader2 size={12} className="animate-spin text-cyan-400 shrink-0" />
              <p className="text-xs text-cyan-300">
                Preview画像を生成しています（数分かかる場合があります）
              </p>
            </div>
          )}
          <div className="grid grid-cols-4 gap-2 p-2">
            {Object.entries(t.sample_previews || {})
              .sort(([a], [b]) => Number(a) - Number(b))
              .map(([slot, img]) => {
                const label = promptLabels[Number(slot)] || `#${slot}`;
                return (
                  <div key={slot} className="relative">
                    <div className="aspect-square bg-gray-700 rounded overflow-hidden">
                      <img
                        src={img}
                        alt={`epoch ${t.epoch} - ${label}`}
                        className="w-full h-full object-cover cursor-zoom-in"
                        title="ダブルクリックで拡大"
                        onDoubleClick={() => onZoom(img)}
                      />
                    </div>
                    <div className="absolute bottom-0 left-0 right-0 bg-black/70 text-[10px] text-gray-200 text-center py-0.5 truncate px-1">
                      {label}
                    </div>
                  </div>
                );
              })}
          </div>
        </div>
        );
      })}
    </div>
  );
});

function ParamInput({
  label,
  value,
  onChange,
  inputMode = "numeric",
  placeholder,
  wide,
  step,
  min = 0,
  max,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  inputMode?: "numeric" | "decimal" | "text";
  placeholder?: string;
  wide?: boolean;
  step?: number;   // 指定すると上下ボタン付きステッパーになる
  min?: number;
  max?: number;
}) {
  const bump = (delta: number) => {
    const cur = parseInt(value, 10);
    let next = (isNaN(cur) ? min : cur) + delta;
    if (next < min) next = min;
    if (max != null && next > max) next = max;
    onChange(String(next));
  };
  return (
    <label className={`block ${wide ? "col-span-2" : ""}`}>
      <span className="text-xs text-gray-400 mb-1 block">{label}</span>
      {step ? (
        <div className="flex items-stretch">
          <input
            type="text"
            inputMode="numeric"
            value={value}
            onChange={(e) => onChange(e.target.value)}
            placeholder={placeholder}
            className="flex-1 min-w-0 bg-gray-700 border border-gray-600 border-r-0 rounded-l-lg text-gray-100 placeholder-gray-500 px-3 py-1.5 text-sm focus:outline-none focus:border-indigo-500"
          />
          <button
            type="button"
            onClick={() => bump(-step)}
            className="w-8 flex-shrink-0 bg-gray-700 hover:bg-gray-600 border-y border-gray-600 text-gray-300 text-base leading-none transition-colors"
            tabIndex={-1}
            title={`-${step}`}
          >−</button>
          <button
            type="button"
            onClick={() => bump(step)}
            className="w-8 flex-shrink-0 bg-gray-700 hover:bg-gray-600 border border-gray-600 rounded-r-lg text-gray-300 text-base leading-none transition-colors"
            tabIndex={-1}
            title={`+${step}`}
          >＋</button>
        </div>
      ) : (
        <input
          type="text"
          inputMode={inputMode}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder}
          className="w-full bg-gray-700 border border-gray-600 text-gray-100 placeholder-gray-500 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500/30"
        />
      )}
    </label>
  );
}

function ParamSelect({
  label,
  value,
  onChange,
  options,
  wide,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  options: (number | string)[];
  wide?: boolean;
}) {
  const opts = options.map(String);
  const isCustom = !opts.includes(value);
  return (
    <label className={`block ${wide ? "col-span-2" : ""}`}>
      <span className="text-xs text-gray-400 mb-1 block">{label}</span>
      <select
        value={isCustom ? "__custom__" : value}
        onChange={(e) => {
          if (e.target.value === "__custom__") return;
          onChange(e.target.value);
        }}
        className="w-full bg-gray-700 border border-gray-600 text-gray-100 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:border-indigo-500"
      >
        {opts.map((o) => (
          <option key={o} value={o}>{o}</option>
        ))}
        {isCustom && <option value="__custom__">{value}（カスタム）</option>}
      </select>
    </label>
  );
}

// パス入力 + フォルダ選択（Electron の input[type=file] の file.path を利用）
function PathInput({
  label,
  value,
  onChange,
  wide,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  wide?: boolean;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const pickFolder = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      const f = files[0] as File & { path?: string; webkitRelativePath: string };
      if (f.path) {
        // file.path から webkitRelativePath 分を取り除いて選択フォルダの絶対パスを得る
        const rel = f.webkitRelativePath || "";
        const folder = rel ? f.path.slice(0, f.path.length - rel.length) : f.path;
        onChange(folder.replace(/[\\/]+$/, ""));
      }
    }
    e.target.value = "";
  };
  return (
    <label className={`block ${wide ? "col-span-2" : ""}`}>
      <span className="text-xs text-gray-400 mb-1 block">{label}</span>
      <div className="flex items-stretch gap-1.5">
        <input
          type="text"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="flex-1 bg-gray-700 border border-gray-600 text-gray-100 rounded-lg px-3 py-2 text-sm font-mono focus:outline-none focus:border-indigo-500"
        />
        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          className="flex items-center gap-1 px-3 bg-gray-700 hover:bg-gray-600 border border-gray-600 rounded-lg text-gray-300 transition-colors flex-shrink-0"
          title="フォルダを選択"
        >
          <FolderOpen size={15} />
        </button>
        {/* @ts-ignore: webkitdirectory は Electron/Chromium 拡張属性 */}
        <input
          ref={inputRef}
          type="file"
          onChange={pickFolder}
          className="hidden"
          {...{ webkitdirectory: "", directory: "" } as any}
        />
      </div>
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
    <div className="flex items-center gap-1.5 min-w-0">
      <span className="text-gray-500">{icon}</span>
      <span className="text-xs text-gray-400 flex-shrink-0">{label}</span>
      <span className="text-xs font-semibold text-gray-100 truncate">{value}</span>
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

function ResourceBar({
  label,
  pct,
  text,
  color,
}: {
  label: string;
  pct: number;
  text: string;
  color: string;
}) {
  return (
    <div>
      <div className="flex items-center justify-between text-xs text-gray-400 mb-1">
        <span>{label}</span>
        <span className="font-mono">{text}</span>
      </div>
      <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ${color}`}
          style={{ width: `${Math.min(100, Math.max(0, pct))}%` }}
        />
      </div>
    </div>
  );
}

const STATUS_CONFIG: Record<string, { label: string; color: string; dot: string }> = {
  idle: { label: "待機中", color: "text-gray-400", dot: "bg-gray-500" },
  queued: { label: "Queueで待機中", color: "text-cyan-400", dot: "bg-cyan-400 animate-pulse" },
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

// ── Performance Modal ──────────────────────────────────────────────────────

function PerfModal({
  mixedPrecision, setMixedPrecision,
  savePrecision, setSavePrecision,
  xformers, setXformers,
  cacheLatents, setCacheLatents,
  cacheLatentsToDisk, setCacheLatentsToDisk,
  gradientCheckpointing, setGradientCheckpointing,
  persistentWorkers, setPersistentWorkers,
  maxWorkers, setMaxWorkers,
  unetOnly, setUnetOnly,
  onClose,
}: {
  mixedPrecision: 'fp16' | 'bf16' | 'no'; setMixedPrecision: (v: 'fp16' | 'bf16' | 'no') => void;
  savePrecision: 'fp16' | 'bf16'; setSavePrecision: (v: 'fp16' | 'bf16') => void;
  xformers: boolean; setXformers: (v: boolean) => void;
  cacheLatents: boolean; setCacheLatents: (v: boolean) => void;
  cacheLatentsToDisk: boolean; setCacheLatentsToDisk: (v: boolean) => void;
  gradientCheckpointing: boolean; setGradientCheckpointing: (v: boolean) => void;
  persistentWorkers: boolean; setPersistentWorkers: (v: boolean) => void;
  maxWorkers: number; setMaxWorkers: (v: number) => void;
  unetOnly: boolean; setUnetOnly: (v: boolean) => void;
  onClose: () => void;
}) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60"
      onClick={onClose}
    >
      <div
        className="bg-gray-800 border border-gray-700 rounded-xl w-full max-w-lg mx-4 p-6 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-5">
          <h3 className="text-base font-semibold text-white flex items-center gap-2">
            <Settings size={16} className="text-indigo-400" />
            パフォーマンス詳細設定
          </h3>
        </div>

        {/* 速度系 */}
        <div className="mb-5">
          <p className="text-xs text-gray-500 uppercase tracking-wide mb-3">速度系</p>
          <div className="space-y-2.5">
            {([
              { key: 'xformers', label: 'xFormers', desc: 'Attention を高速化（RTX推奨）', val: xformers, set: setXformers },
              { key: 'cache_latents', label: 'Cache Latents', desc: 'VAEエンコードをキャッシュ（VRAM節約・高速化）', val: cacheLatents, set: setCacheLatents },
              { key: 'cache_latents_to_disk', label: 'Cache Latents to Disk', desc: 'ディスクにキャッシュ（2回目以降さらに高速、要空き容量）', val: cacheLatentsToDisk, set: setCacheLatentsToDisk },
              { key: 'gradient_checkpointing', label: 'Gradient Checkpointing', desc: 'VRAMと速度をトレードオフ', val: gradientCheckpointing, set: setGradientCheckpointing },
              { key: 'persistent_workers', label: 'Persistent DataLoader Workers', desc: 'ワーカーを維持（DataLoader高速化）', val: persistentWorkers, set: setPersistentWorkers },
              { key: 'unet_only', label: 'UNet Only（Text Encoder固定）', desc: 'Text Encoderを学習しない（高速・VRAM節約）', val: unetOnly, set: setUnetOnly },
            ] as { key: string; label: string; desc: string; val: boolean; set: (v: boolean) => void }[]).map(({ key, label, desc, val, set }) => (
              <label key={key} className="flex items-start gap-3 cursor-pointer group">
                <input
                  type="checkbox"
                  checked={val}
                  onChange={(e) => set(e.target.checked)}
                  className="mt-0.5 accent-indigo-500"
                />
                <div>
                  <span className="text-sm text-gray-200 group-hover:text-white transition-colors">{label}</span>
                  <p className="text-xs text-gray-500">{desc}</p>
                </div>
              </label>
            ))}
          </div>
        </div>

        {/* 精度系 */}
        <div className="mb-6">
          <p className="text-xs text-gray-500 uppercase tracking-wide mb-3">精度系</p>
          <div className="grid grid-cols-2 gap-3 mb-3">
            <label className="block">
              <span className="text-xs text-gray-400 mb-1 block">mixed_precision</span>
              <select
                value={mixedPrecision}
                onChange={(e) => setMixedPrecision(e.target.value as 'fp16' | 'bf16' | 'no')}
                className="w-full bg-gray-700 border border-gray-600 text-gray-100 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:border-indigo-500"
              >
                <option value="bf16">bf16</option>
                <option value="fp16">fp16</option>
                <option value="no">no</option>
              </select>
            </label>
            <label className="block">
              <span className="text-xs text-gray-400 mb-1 block">save_precision</span>
              <select
                value={savePrecision}
                onChange={(e) => setSavePrecision(e.target.value as 'fp16' | 'bf16')}
                className="w-full bg-gray-700 border border-gray-600 text-gray-100 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:border-indigo-500"
              >
                <option value="fp16">fp16</option>
                <option value="bf16">bf16</option>
              </select>
            </label>
          </div>
          <label className="block w-1/2">
            <span className="text-xs text-gray-400 mb-1 block">max_data_loader_n_workers</span>
            <input
              type="number"
              min={0}
              max={16}
              value={maxWorkers}
              onChange={(e) => setMaxWorkers(Math.min(16, Math.max(0, parseInt(e.target.value) || 0)))}
              className="w-full bg-gray-700 border border-gray-600 text-gray-100 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:border-indigo-500"
            />
          </label>
        </div>

        <div className="flex justify-end">
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-2 text-sm bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg transition-colors"
          >
            適用して閉じる
          </button>
        </div>
      </div>
    </div>
  );
}

// ── Save Preset Row ─────────────────────────────────────────────────────────

function SavePresetRow({ onSave }: { onSave: (name: string) => Promise<void> }) {
  const [open, setOpen] = useState(false);
  const [name, setName] = useState("");
  const [saving, setSaving] = useState(false);

  async function handleSave() {
    const trimmed = name.trim();
    if (!trimmed) return;
    setSaving(true);
    try {
      await onSave(trimmed);
      setName("");
      setOpen(false);
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="mt-3 border-t border-gray-700/60 pt-3">
      {!open ? (
        <button
          onClick={() => setOpen(true)}
          className="w-full text-xs text-indigo-400 hover:text-indigo-300 border border-dashed border-indigo-700/60 hover:border-indigo-500 rounded-lg py-1.5 transition-colors"
        >
          ＋ 現在の設定をプロファイルとして保存
        </button>
      ) : (
        <div className="flex items-center gap-2">
          <input
            autoFocus
            value={name}
            onChange={(e) => setName(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter") void handleSave(); if (e.key === "Escape") setOpen(false); }}
            placeholder="プロファイル名"
            className="flex-1 bg-gray-700 border border-gray-600 text-gray-200 rounded px-2 py-1 text-xs focus:outline-none focus:border-indigo-500"
          />
          <button
            onClick={() => void handleSave()}
            disabled={saving || !name.trim()}
            className="text-xs bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 text-white rounded px-2 py-1 transition-colors"
          >
            {saving ? "…" : "保存"}
          </button>
          <button onClick={() => setOpen(false)} className="text-xs text-gray-500 hover:text-gray-300 px-1">✕</button>
        </div>
      )}
    </div>
  );
}

// ── Review Gate Modal ───────────────────────────────────────────────────────

const PREFLIGHT_LEVEL_STYLE: Record<string, { dot: string; text: string }> = {
  ok: { dot: "bg-emerald-400", text: "text-emerald-300" },
  warning: { dot: "bg-amber-400", text: "text-amber-300" },
  blocked: { dot: "bg-red-400", text: "text-red-300" },
};

function ReviewGateModal({
  report,
  preflight,
  loading,
  onConfirm,
  onCancel,
}: {
  report: DatasetReport | null;
  preflight: PreflightResult | null;
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

              {/* Preflight — Model / Engine / Base Model / GPU / Queue */}
              {preflight && (
                <div className="bg-gray-700/40 rounded-lg p-3 space-y-1.5">
                  {preflight.checks.map((c, i) => {
                    const style = PREFLIGHT_LEVEL_STYLE[c.level] ?? PREFLIGHT_LEVEL_STYLE.warning;
                    return (
                      <div key={i} className="flex items-start gap-2 text-xs">
                        <span className={`w-1.5 h-1.5 rounded-full mt-1 shrink-0 ${style.dot}`} />
                        <span className="text-gray-400 shrink-0 w-28">{c.label}</span>
                        <span className={`${style.text} flex-1`}>
                          {c.detail}
                          {c.action && <span className="text-gray-500"> — {c.action}</span>}
                        </span>
                      </div>
                    );
                  })}
                </div>
              )}

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
            disabled={loading || !report || !report.ready || preflight?.overall === "blocked"}
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
      title={preset.payload?.description || preset.name}
      className={`relative text-left rounded-lg border px-2.5 py-2 transition-all w-full ${
        selected
          ? colorCls + " ring-2 ring-inset ring-indigo-400"
          : "border-gray-700 bg-gray-700/40 hover:border-gray-500"
      }`}
    >
      <div className="flex items-center justify-between gap-1">
        <div className="text-xs font-semibold text-gray-100 truncate">{preset.name}</div>
        {selected && <CheckCircle2 size={12} className="text-indigo-400 shrink-0" />}
      </div>
      <div className="flex gap-2 mt-1 text-[10px] text-gray-400 font-mono flex-wrap">
        <span>r{preset.payload?.rank}</span>
        <span>α{preset.payload?.alpha}</span>
        <span>{preset.payload?.epochs}ep</span>
        <span>{preset.payload?.resolution}px</span>
        {preset.payload?.learning_rate != null && (
          <span>lr{preset.payload.learning_rate}</span>
        )}
        {preset.payload?.network_train_unet_only && (
          <span className="text-amber-500">UNet only</span>
        )}
      </div>
      {onDelete && (
        <button
          onClick={(e) => { e.stopPropagation(); onDelete(); }}
          className="absolute top-1 right-1 p-0.5 rounded text-gray-500 hover:text-red-400 hover:bg-gray-600 transition-colors opacity-0 group-hover:opacity-100"
          title="削除"
        >
          <Trash2 size={10} />
        </button>
      )}
    </button>
  );
}

// ── 学習時間予測カード ──────────────────────────────────────────────────────

const ESTIMATE_SOURCE: Record<
  TrainingEstimate["source"],
  { label: string; cls: string; desc: string }
> = {
  live: {
    label: "実測中",
    cls: "bg-emerald-900/50 text-emerald-300 border-emerald-700/50",
    desc: "学習中プロセスの実測 sec/step から算出（最高精度）",
  },
  calibrated: {
    label: "実測キャリブレーション",
    cls: "bg-sky-900/50 text-sky-300 border-sky-700/50",
    desc: "同 GPU・同モデル系統の過去実測から学習した値で算出",
  },
  heuristic: {
    label: "GPU 推定",
    cls: "bg-amber-900/40 text-amber-300 border-amber-700/40",
    desc: "GPU 性能テーブルからの初期推定。1 回学習すると実測で補正されます",
  },
  simulation: {
    label: "シミュレーション",
    cls: "bg-gray-700 text-gray-300 border-gray-600",
    desc: "エンジン未接続。1 step=0.5s の固定値（厳密）",
  },
};

const CONFIDENCE_DOT: Record<TrainingEstimate["confidence"], string> = {
  high: "bg-emerald-400",
  medium: "bg-amber-400",
  low: "bg-red-400",
};

function EstimateCard({
  estimate,
  loading,
  collapsed,
  onToggle,
}: {
  estimate: TrainingEstimate | null;
  loading: boolean;
  collapsed?: boolean;
  onToggle?: () => void;
}) {
  const src = estimate ? ESTIMATE_SOURCE[estimate.source] : null;
  return (
    <div className="bg-gray-800 border border-gray-700 rounded-xl overflow-hidden">
      <button
        className="w-full flex items-center justify-between px-3 py-2 hover:bg-gray-750 transition-colors"
        onClick={onToggle}
      >
        <span className="text-sm font-semibold text-gray-200 flex items-center gap-2">
          <Timer size={15} className="text-indigo-400" />
          学習時間予測
          {loading && <Loader2 size={12} className="animate-spin text-gray-500" />}
          {collapsed && estimate?.ready && (
            <span className="text-xs font-normal text-gray-300">
              {estimate.remaining_seconds != null
                ? `残り ${durationText(estimate.remaining_seconds)}`
                : durationText(estimate.eta_seconds)}
            </span>
          )}
        </span>
        <span className="flex items-center gap-2">
          {!collapsed && src && (
            <span
              className={`inline-flex items-center gap-1 text-[10px] px-2 py-0.5 rounded-full border font-medium cursor-help ${src.cls}`}
              title={src.desc}
            >
              <span className={`w-1.5 h-1.5 rounded-full ${CONFIDENCE_DOT[estimate!.confidence]}`} />
              {src.label}
            </span>
          )}
          {collapsed ? <ChevronDown size={16} className="text-gray-400" /> : <ChevronUp size={16} className="text-gray-400" />}
        </span>
      </button>
      {!collapsed && (
        <div className="px-4 pb-4 pt-1 border-t border-gray-700/60">
          {!estimate ? (
            <p className="text-xs text-gray-500 mt-2">設定を入力すると予測が表示されます</p>
          ) : !estimate.ready ? (
            <p className="text-xs text-amber-400/80 mt-2">
              学習対象の画像が選択されていません（Dataset で選択してください）
            </p>
          ) : (
            <>
              <div className="flex items-end gap-2 mb-3 flex-wrap mt-2">
                {estimate.remaining_seconds != null ? (
                  <>
                    <span className="text-3xl font-bold text-amber-300 tabular-nums">
                      {durationText(estimate.remaining_seconds)}
                    </span>
                    <span className="text-xs text-gray-500 mb-1">残り（学習中）</span>
                    <span className="text-sm text-gray-400 tabular-nums ml-3 mb-0.5">
                      予想総時間 {durationText(estimate.eta_seconds)}
                    </span>
                  </>
                ) : (
                  <>
                    <span className="text-3xl font-bold text-gray-100 tabular-nums">
                      {durationText(estimate.eta_seconds)}
                    </span>
                    <span className="text-xs text-gray-500 mb-1">予想総時間</span>
                  </>
                )}
              </div>
              <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-xs">
                <Row label="総ステップ" value={`${estimate.total_steps.toLocaleString()} step`} />
                <Row label="1 epoch あたり" value={durationText(estimate.sec_per_epoch)} />
                <Row label="steps/epoch" value={`${estimate.steps_per_epoch.toLocaleString()}`} />
                <Row label="1 step" value={`${estimate.sec_per_step.toFixed(2)} s`} />
                <Row label="画像 × repeat" value={`${estimate.images} × → batch${estimate.batch_size}`} />
                <Row label="epochs" value={`${estimate.epochs}`} />
                <Row label="解像度" value={`${estimate.effective_resolution}px${estimate.is_sdxl ? " (SDXL)" : ""}`} />
                <Row label="準備時間" value={estimate.overhead_seconds > 0 ? `+${durationText(estimate.overhead_seconds)}` : "—"} />
              </div>
              <div className="flex items-center gap-1.5 mt-3 pt-2 border-t border-gray-700 text-xs text-gray-400">
                <Gauge size={12} className="text-gray-500" />
                <span className="truncate">{estimate.gpu_name ?? "GPU 未検出"}</span>
                <span className="ml-auto text-gray-500">{estimate.mode}</span>
              </div>
              {estimate.source === "heuristic" && (
                <p className="text-[10px] text-gray-500 mt-2">
                  ※ 初回学習の数ステップ進行後、実測値で自動的に高精度化されます。
                </p>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}

// ── メインコンポーネント ───────────────────────────────────────────────────

export default function Training({
  projects,
  selectedProjectId,
  selectedProject,
  onSelectProject,
  currentStatus,
  onRefresh,
  prompts,
  onPromptsChange,
  showError,
  showNotice,
}: Props) {
  // ── パラメータ状態（localStorage で永続化） ─────────────────────────
  function useLS(key: string, def: string): [string, React.Dispatch<React.SetStateAction<string>>] {
    const [v, setV] = useState<string>(() => localStorage.getItem(`training_${key}`) ?? def);
    const setVAndStore: React.Dispatch<React.SetStateAction<string>> = (next) => {
      setV((prev) => {
        const val = typeof next === "function" ? (next as (p: string) => string)(prev) : next;
        localStorage.setItem(`training_${key}`, val);
        return val;
      });
    };
    return [v, setVAndStore];
  }
  const [epochsText, setEpochsText] = useLS("epochs", "5");
  const [rankText, setRankText] = useLS("rank", "16");
  const [alphaText, setAlphaText] = useLS("alpha", "4");
  const [trainRepeatsText, setTrainRepeatsText] = useLS("repeats", "5");
  const [saveEveryText, setSaveEveryText] = useLS("save_every", "1");
  const [resolutionText, setResolutionText] = useLS("resolution", "512");
  const [outputName, setOutputName] = useLS("output_name", "lora_output");
  const [baseCkpt, setBaseCkpt] = useLS("base_ckpt", "");
  const [optimizerVal, setOptimizerVal] = useLS("optimizer", "AdamW8bit");
  const [schedulerVal, setSchedulerVal] = useLS("scheduler", "cosine_with_restarts");
  const [minSnrText, setMinSnrText] = useLS("min_snr", "5");
  const [lrText, setLrText] = useLS("learning_rate", "0.0001");
  const [batchText, setBatchText] = useLS("train_batch_size", "2");

  // ── パフォーマンス詳細設定 ─────────────────────────────────────────────
  const [showPerfModal, setShowPerfModal] = useState(false);
  const [mixedPrecision, setMixedPrecision] = useState<'fp16' | 'bf16' | 'no'>('bf16');
  const [savePrecision, setSavePrecision] = useState<'fp16' | 'bf16'>('fp16');
  const [xformers, setXformers] = useState(true);
  const [cacheLatents, setCacheLatents] = useState(true);
  const [cacheLatentsToDisk, setCacheLatentsToDisk] = useState(false);
  const [gradientCheckpointing, setGradientCheckpointing] = useState(true);
  const [persistentWorkers, setPersistentWorkers] = useState(true);
  const [maxWorkers, setMaxWorkers] = useState(4);
  const [unetOnly, setUnetOnly] = useState(false);

  const [zoomImg, setZoomImg] = useState<string | null>(null);
  const [retryingCheckpointId, setRetryingCheckpointId] = useState<number | null>(null);
  const [checkpoints, setCheckpoints] = useState<{ name: string; path: string; size_mb: number; source: string }[]>([]);
  const [ckptManual, setCkptManual] = useState(false);
  const [ckptLoading, setCkptLoading] = useState(false);
  const [ckptSearch, setCkptSearch] = useState("");
  const [trainDir, setTrainDir] = useState("");
  const [datasetSource, setDatasetSource] = useState<"original" | "processed">("original");
  const [processedAvailable, setProcessedAvailable] = useState(false);
  const [processedManifest, setProcessedManifest] = useState<{ success?: number; generated_at?: string } | null>(null);
  const [regDir, setRegDir] = useState("");
  const [modelFamily, setModelFamily] = useLS("model_family", "auto");
  const [trainingEngine, setTrainingEngine] = useLS("training_engine", "auto");
  const [gpuDeviceId] = useState(1); // RTX 3090 Ti は device_id=1 固定
  const [showParams, setShowParams] = useState(true);
  const [showEstimate, setShowEstimate] = useState(true);
  const [showStatus, setShowStatus] = useState(true);
  const [showPrompts, setShowPrompts] = useState(false);
  const [showTimeline, setShowTimeline] = useState(true);
  const [showTrainingMode, setShowTrainingMode] = useState(false);

  // ── プロファイル状態 ─────────────────────────────────────────────────
  const [presets, setPresets] = useState<Preset[]>([]);
  const [selectedPresetId, setSelectedPresetId] = useState<number | null>(null);
  const [showProfiles, setShowProfiles] = useState(true);

  // ── 学習モード ───────────────────────────────────────────────────────
  const [trainingMode, setTrainingMode] = useState<TrainingMode | null>(null);
  const [modelSpecs, setModelSpecs] = useState<ModelSpec[]>([]);

  // ── Review Gate Modal 状態 ────────────────────────────────────────────
  const [reviewModalOpen, setReviewModalOpen] = useState(false);
  const [reportLoading, setReportLoading] = useState(false);
  const [report, setReport] = useState<DatasetReport | null>(null);
  const [preflight, setPreflight] = useState<PreflightResult | null>(null);

  // ── ログビューア ─────────────────────────────────────────────────────
  const [logLines, setLogLines] = useState<string[]>([]);
  const [showLog, setShowLog] = useState(false);
  const logRef = useRef<HTMLDivElement>(null);

  // ── リソースモニター ──────────────────────────────────────────────────
  const [resources, setResources] = useState<ResourceStats | null>(null);
  const [showResources, setShowResources] = useState(false);

  // ── 全体Queue(他プロジェクト含む) ─────────────────────────────────────
  const [queueState, setQueueState] = useState<TrainingQueueState | null>(null);
  const [showQueue, setShowQueue] = useState(false);

  // ── 実際に使うべきDatasetフォルダの推奨値(dataset-sources経由) ──
  const [recommendedTrainDir, setRecommendedTrainDir] = useState<
    { path: string; image_count: number; total_registered: number; split_across_dirs: boolean } | null
  >(null);

  // ── 学習時間予測 ──────────────────────────────────────────────────────
  const [estimate, setEstimate] = useState<TrainingEstimate | null>(null);
  const [estimateLoading, setEstimateLoading] = useState(false);

  // ── プレビュー ───────────────────────────────────────────────────────
  const [timeline, setTimeline] = useState<PreviewSample[]>([]);
  const [loadingTimeline, setLoadingTimeline] = useState(false);
  // PreviewTimeline(memo) に渡すラベル配列。prompts が変わったときだけ作り直す
  // → step 更新で毎回新配列を渡して memo が無効化されるのを防ぐ。
  const promptLabels = useMemo(
    () => (prompts.prompts ?? []).map((p) => p.label || ""),
    [prompts.prompts],
  );

  // ── TagSuggest モーダル ────────────────────────────────────────────────
  const [tagSuggestOpen, setTagSuggestOpen] = useState(false);
  const [tagSuggestProfileIdx, setTagSuggestProfileIdx] = useState(0);
  const [tagSuggestSelected, setTagSuggestSelected] = useState<number[]>([]);
  const [tagSuggestResults, setTagSuggestResults] = useState<{ tag: string; count: number; ratio: number }[]>([]);
  const [tagSuggestLoading, setTagSuggestLoading] = useState(false);
  const [tagSuggestItems, setTagSuggestItems] = useState<CaptionItem[]>([]);
  const [tagSuggestChosen, setTagSuggestChosen] = useState<string[]>([]);

  // currentStatus は App.tsx から selectedProject に対応する値のみ受け取る
  const statusConf = STATUS_CONFIG[currentStatus?.status ?? "idle"] ?? STATUS_CONFIG.idle;
  const isTraining = currentStatus?.status === "training";
  const isQueued = currentStatus?.status === "queued";
  const isPaused = currentStatus?.status === "paused";
  const isCompleted = currentStatus?.status === "completed";
  // 学習完了(Training)とPreview生成完了は別の非同期処理であり、完了直後は
  // まだPreviewが生成中のことがある。「完了」とだけ表示するとユーザーは
  // LoRA自体が使えると誤認しかねないため、最新checkpointがまだ
  // validation_status未確定(preview_succeeded/preview_failed以外)の間は
  // 「Preview画像を生成しています」と明示する。
  const previewPendingForLatestCheckpoint = isCompleted && timeline.length > 0 &&
    !["preview_succeeded", "preview_failed", "invalid"].includes(
      timeline[0]?.validation_status ?? ""
    );

  // ── 初期ロード ────────────────────────────────────────────────────────
  useEffect(() => {
    void loadPresets();
    void loadTrainingMode();
    void loadModelSpecs();
    void loadCheckpoints();
  }, []);

  useEffect(() => {
    if (!selectedProject) return;
    setTrainDir(selectedProject.dataset_dir);
    setDatasetSource("original");
    setRecommendedTrainDir(null);
    void apiGet<{
      processed: { available: boolean; manifest: { success?: number; generated_at?: string } | null };
      recommended_train_dir: { path: string; image_count: number; total_registered: number; split_across_dirs: boolean } | null;
    }>(
      `/dataset/dataset-sources/${selectedProject.id}`
    ).then((res) => {
      setProcessedAvailable(res.processed.available);
      setProcessedManifest(res.processed.manifest);
      // dataset_dir(静的な既定パス)が空/古い一方、実際にキャプション付きで
      // 登録済みの画像が別フォルダ(Dataset Builder整理後のlibrary_dir配下等)に
      // まとまって存在する場合、そちらを学習フォルダとして自動的に採用する。
      // ユーザーが気づかないまま「0枚のフォルダで学習」してしまう実UXバグの修正。
      const rec = res.recommended_train_dir;
      if (rec && !rec.split_across_dirs && rec.image_count > 0) {
        setRecommendedTrainDir(rec);
        setTrainDir((cur) => (cur === selectedProject.dataset_dir ? rec.path : cur));
      }
    }).catch(() => {
      setProcessedAvailable(false);
      setProcessedManifest(null);
    });
    setRegDir(selectedProject.captions_dir);
    // output_name はプロジェクトごとに初期値を設定（既存の保存値がある場合はスキップ）
    const cls = selectedProject.project_type || "lora";
    const romaji = toRomaji(selectedProject.name || "") || "project";
    const savedOutputName = localStorage.getItem("training_output_name");
    if (!savedOutputName || savedOutputName === "lora_output") {
      void apiGet<{ id: number }[]>(`/library/assets?project_id=${selectedProject.id}`)
        .then((assets) => {
          const nn = String((assets?.length ?? 0) + 1).padStart(2, "0");
          setOutputName(`${cls}_${romaji}_${nn}`);
        })
        .catch(() => setOutputName(`${cls}_${romaji}_01`));
    }
    void loadTimeline(selectedProject.id);
  }, [selectedProject?.id]);

  // ── Training設定の下書き復元（Project単位） ─────────────────────────────
  // 上記のuseLSはlocalStorageベースでProjectを問わずグローバルに共有されるため、
  // 「Project Aで設定した内容がProject Bにも表示される/Project切替で消える」問題が
  // あった。Project単位でサーバー保存された下書きが存在すれば、それでフォームの
  // 値を上書きする(保存されていないキーは既存のlocalStorage値のまま)。
  useEffect(() => {
    if (!selectedProject) return;
    void apiGet<{ project_id: number; config: Record<string, unknown> }>(
      `/training/config-draft/${selectedProject.id}`
    ).then((res) => {
      const c = res.config ?? {};
      if (c.epochs != null) setEpochsText(String(c.epochs));
      if (c.rank != null) setRankText(String(c.rank));
      if (c.alpha != null) setAlphaText(String(c.alpha));
      if (c.repeats != null) setTrainRepeatsText(String(c.repeats));
      if (c.save_every_n_epochs != null) setSaveEveryText(String(c.save_every_n_epochs));
      if (c.resolution != null) setResolutionText(String(c.resolution));
      if (typeof c.output_name === "string" && c.output_name) setOutputName(c.output_name);
      if (typeof c.base_checkpoint_path === "string" && c.base_checkpoint_path) setBaseCkpt(c.base_checkpoint_path);
      if (typeof c.optimizer === "string") setOptimizerVal(c.optimizer);
      if (typeof c.scheduler === "string") setSchedulerVal(c.scheduler);
      if (c.min_snr_gamma != null) setMinSnrText(String(c.min_snr_gamma));
      if (c.learning_rate != null) setLrText(String(c.learning_rate));
      if (c.train_batch_size != null) setBatchText(String(c.train_batch_size));
      if (typeof c.mixed_precision === "string") setMixedPrecision(c.mixed_precision as 'fp16' | 'bf16' | 'no');
      if (typeof c.save_precision === "string") setSavePrecision(c.save_precision as 'fp16' | 'bf16');
      if (typeof c.xformers === "boolean") setXformers(c.xformers);
      if (typeof c.cache_latents === "boolean") setCacheLatents(c.cache_latents);
      if (typeof c.cache_latents_to_disk === "boolean") setCacheLatentsToDisk(c.cache_latents_to_disk);
      if (typeof c.gradient_checkpointing === "boolean") setGradientCheckpointing(c.gradient_checkpointing);
      if (typeof c.persistent_data_loader_workers === "boolean") setPersistentWorkers(c.persistent_data_loader_workers);
      if (typeof c.max_data_loader_n_workers === "number") setMaxWorkers(c.max_data_loader_n_workers);
      if (typeof c.network_train_unet_only === "boolean") setUnetOnly(c.network_train_unet_only);
      if (typeof c.model_family === "string") setModelFamily(c.model_family);
      if (typeof c.dataset_source === "string") setDatasetSource(c.dataset_source as "original" | "processed");
    }).catch(() => { /* 下書き未保存・取得失敗時は既存値のまま(実害なし) */ });
  }, [selectedProject?.id]);

  // ── Training設定の下書き保存（Project単位、デバウンス） ──────────────────
  useEffect(() => {
    if (!selectedProject) return;
    const timer = setTimeout(() => {
      void apiPost(`/training/config-draft/${selectedProject.id}`, {
        epochs: Number(epochsText), rank: Number(rankText), alpha: Number(alphaText),
        repeats: Number(trainRepeatsText), save_every_n_epochs: Number(saveEveryText),
        resolution: Number(resolutionText), output_name: outputName, base_checkpoint_path: baseCkpt,
        optimizer: optimizerVal, scheduler: schedulerVal, min_snr_gamma: Number(minSnrText),
        learning_rate: Number(lrText), train_batch_size: Number(batchText),
        mixed_precision: mixedPrecision, save_precision: savePrecision, xformers,
        cache_latents: cacheLatents, cache_latents_to_disk: cacheLatentsToDisk,
        gradient_checkpointing: gradientCheckpointing,
        persistent_data_loader_workers: persistentWorkers, max_data_loader_n_workers: maxWorkers,
        network_train_unet_only: unetOnly, model_family: modelFamily, dataset_source: datasetSource,
      }).catch(() => { /* 下書き保存失敗はTraining続行に影響させない(非致命) */ });
    }, 600);
    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    selectedProject?.id, epochsText, rankText, alphaText, trainRepeatsText, saveEveryText,
    resolutionText, outputName, baseCkpt, optimizerVal, schedulerVal, minSnrText, lrText,
    batchText, mixedPrecision, savePrecision, xformers, cacheLatents, cacheLatentsToDisk,
    gradientCheckpointing, persistentWorkers, maxWorkers, unetOnly, modelFamily, datasetSource,
  ]);

  // ── ログのポーリング（学習中のみ） ───────────────────────────────────
  useEffect(() => {
    if (!isTraining || !selectedProject || !showLog) return;
    const timer = setInterval(() => {
      void loadLogs(selectedProject.id);
    }, 2000);
    return () => clearInterval(timer);
  }, [isTraining, selectedProject?.id, showLog]);

  // ── 全体Queueポーリング（常時。折り畳み中もバッジ表示に使うため） ────────
  useEffect(() => {
    void loadQueue();
    const timer = setInterval(() => void loadQueue(), 5000);
    return () => clearInterval(timer);
  }, []);

  // ── リソースモニターポーリング（表示中のみ） ──────────────────────────
  useEffect(() => {
    if (!showResources) return;
    void loadResources();
    const timer = setInterval(() => void loadResources(), 3000);
    return () => clearInterval(timer);
  }, [showResources]);

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [logLines]);

  // ── 学習時間予測: パラメータ変更時にデバウンスで再算出 ─────────────────
  useEffect(() => {
    if (!selectedProject) return;
    const timer = setTimeout(() => void loadEstimate(), 400);
    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    selectedProject?.id,
    epochsText,
    trainRepeatsText,
    resolutionText,
    rankText,
    batchText,
    optimizerVal,
    baseCkpt,
    xformers,
    gradientCheckpointing,
    mixedPrecision,
  ]);

  // ── 学習中は実測 sec/step を反映するため 4 秒ごとに再算出 ───────────────
  useEffect(() => {
    if (!isTraining || !selectedProject) return;
    const timer = setInterval(() => void loadEstimate(), 4000);
    return () => clearInterval(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isTraining, selectedProject?.id]);

  // ── 学習中はプレビュー(チェックポイント/サンプル)を定期取得して即反映 ───────
  useEffect(() => {
    if (!isTraining || !selectedProject) return;
    const pid = selectedProject.id;
    const timer = setInterval(() => void loadTimeline(pid, { silent: true }), 6000);
    return () => clearInterval(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isTraining, selectedProject?.id]);

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

  async function loadModelSpecs() {
    try {
      const data = await apiGet<{ specs: ModelSpec[]; count: number }>("/training/model-specs");
      setModelSpecs(data.specs ?? []);
    } catch {
      setModelSpecs([]);
    }
  }

  async function loadCheckpoints() {
    setCkptLoading(true);
    try {
      const data = await apiGet<{ checkpoints: { name: string; path: string; size_mb: number; source: string }[] }>(
        "/training/checkpoints"
      );
      setCheckpoints(data.checkpoints);
      // 候補が無ければ手動入力に切替
      if (data.checkpoints.length === 0) setCkptManual(true);
    } catch {
      setCheckpoints([]);
      setCkptManual(true);
    } finally {
      setCkptLoading(false);
    }
  }

  // silent=true（学習中のバックグラウンドポーリング）では loadingTimeline を立てない。
  // → 画像が「読み込み中...」に差し替わってチラつくのを防ぐ。
  async function loadTimeline(projectId: number, opts?: { silent?: boolean }) {
    const silent = opts?.silent ?? false;
    if (!silent) setLoadingTimeline(true);
    try {
      const r = await apiGet<{ timeline: PreviewSample[] }>(
        `/previews/${projectId}?include_images=false`,
      );
      const next = r.timeline || [];
      // 内容が同じなら state を更新しない（配列参照を維持し、img の再描画を防ぐ）
      setTimeline((prev) => (sameTimeline(prev, next) ? prev : next));
    } catch {
      if (!silent) setTimeline([]); // ポーリング中の一時エラーで画像を消さない
    } finally {
      if (!silent) setLoadingTimeline(false);
    }
  }

  async function handleRetryPreview(checkpointId: number) {
    setRetryingCheckpointId(checkpointId);
    try {
      const res = await apiPost<{ succeeded: boolean; detail: string; message: string }>(
        `/previews/retry/${checkpointId}`,
        {}
      );
      if (selectedProject) await loadTimeline(selectedProject.id);
      if (!res.succeeded) {
        showError?.(res.message);
      } else {
        showNotice?.(res.message);
      }
    } catch (e) {
      showError?.(e instanceof Error ? e.message : "Preview再試行に失敗しました");
    } finally {
      setRetryingCheckpointId(null);
    }
  }

  async function loadEstimate() {
    if (!selectedProject) return;
    setEstimateLoading(true);
    try {
      const q = new URLSearchParams({
        project_id: String(selectedProject.id),
        epochs: String(parseIntOr(epochsText, 5, 1)),
        repeats: String(parseIntOr(trainRepeatsText, 5, 1)),
        resolution: String(parseIntOr(resolutionText, 512, 64)),
        rank: String(parseIntOr(rankText, 16, 1)),
        batch_size: String(parseIntOr(batchText, 2, 1)),
        optimizer: optimizerVal,
        base_checkpoint_path: baseCkpt,
        xformers: String(xformers),
        gradient_checkpointing: String(gradientCheckpointing),
        mixed_precision: mixedPrecision,
        gpu_device_id: String(gpuDeviceId),
        model_family: modelFamily,
        training_engine: trainingEngine,
      });
      const r = await apiGet<TrainingEstimate>(`/training/estimate?${q.toString()}`);
      setEstimate(r);
    } catch (e) {
      // エラーでも前の値を消さない（表示が消えるのを防ぐ）
      console.warn("estimate fetch failed:", e);
    } finally {
      setEstimateLoading(false);
    }
  }

  async function loadResources() {
    try {
      const r = await apiGet<ResourceStats>("/training/resources");
      setResources(r);
    } catch {
      // ignore — psutil not installed or endpoint unavailable
    }
  }

  async function loadQueue() {
    try {
      const r = await apiGet<TrainingQueueState>("/training/queue");
      setQueueState(r);
    } catch {
      // ignore — Queue表示は補助情報であり、失敗しても学習制御自体には影響させない
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
    // 基本パラメータ
    if (p.learning_rate != null) setLrText(String(p.learning_rate));
    if (p.train_batch_size != null) setBatchText(String(p.train_batch_size));
    // パフォーマンス設定
    if (p.xformers != null) setXformers(p.xformers);
    if (p.cache_latents != null) setCacheLatents(p.cache_latents);
    if (p.cache_latents_to_disk != null) setCacheLatentsToDisk(p.cache_latents_to_disk);
    if (p.gradient_checkpointing != null) setGradientCheckpointing(p.gradient_checkpointing);
    if (p.mixed_precision) setMixedPrecision(p.mixed_precision as 'fp16' | 'bf16' | 'no');
    if (p.save_precision) setSavePrecision(p.save_precision as 'fp16' | 'bf16');
    if (p.persistent_data_loader_workers != null) setPersistentWorkers(p.persistent_data_loader_workers);
    if (p.max_data_loader_n_workers != null) setMaxWorkers(p.max_data_loader_n_workers);
    if (p.network_train_unet_only != null) setUnetOnly(p.network_train_unet_only);
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
    setPreflight(null);
    setReportLoading(true);
    try {
      const [data, pf] = await Promise.all([
        apiGet<DatasetReport>(`/dataset/report/${selectedProject.id}`),
        apiGet<PreflightResult>(
          `/training/preflight?project_id=${selectedProject.id}&model_family=${encodeURIComponent(modelFamily)}` +
          `&train_data_dir=${encodeURIComponent(trainDir)}&base_checkpoint_path=${encodeURIComponent(baseCkpt)}`
        ).catch(() => null),
      ]);
      setReport(data);
      setPreflight(pf);
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
    if (!baseCkpt.trim()) {
      showError("ベースモデルを選択してください。");
      return;
    }
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
        learning_rate: parseFloatOr(lrText, 1e-4, 1e-7),
        train_batch_size: parseIntOr(batchText, 2, 1),
        optimizer: optimizerVal,
        scheduler: schedulerVal,
        min_snr_gamma: minSnrText.trim() ? parseIntOr(minSnrText, 5, 0) : null,
        mixed_precision: mixedPrecision,
        save_precision: savePrecision,
        xformers,
        cache_latents: cacheLatents,
        cache_latents_to_disk: cacheLatentsToDisk,
        gradient_checkpointing: gradientCheckpointing,
        persistent_data_loader_workers: persistentWorkers,
        max_data_loader_n_workers: maxWorkers,
        network_train_unet_only: unetOnly,
        model_family: modelFamily,
        training_engine: trainingEngine,
        dataset_source: datasetSource,
        gpu_device_id: gpuDeviceId,
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

  async function resetTraining() {
    if (!selectedProject) return;
    const ok = window.confirm(
      "プレビュー画像のキャッシュを削除し、進捗を Step 0 に戻します。\n（チェックポイント・学習履歴・資産ライブラリは保持されます）"
    );
    if (!ok) return;
    try {
      const r = await apiPost<{ deleted?: { previews?: number; files?: number } }>(
        "/training/reset",
        { project_id: selectedProject.id }
      );
      const d = r?.deleted;
      showNotice(
        d
          ? `プレビュー削除・Step 0 リセット完了（プレビュー ${d.previews ?? 0} 件 / ファイル ${d.files ?? 0} 件）`
          : "プレビュー削除・Step 0 リセット完了"
      );
      await loadTimeline(selectedProject.id);
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

  async function openTagSuggest(profileIdx: number) {
    setTagSuggestProfileIdx(profileIdx);
    setTagSuggestSelected([]);
    setTagSuggestResults([]);
    setTagSuggestChosen([]);
    if (selectedProject) {
      try {
        const data = await apiGet<{ items: CaptionItem[] }>(`/tags/${selectedProject.id}`);
        setTagSuggestItems(data.items ?? []);
      } catch {
        setTagSuggestItems([]);
      }
    }
    setTagSuggestOpen(true);
  }

  async function analyzeRefImages() {
    if (!selectedProject || tagSuggestSelected.length < 2) return;
    setTagSuggestLoading(true);
    try {
      const res = await apiPost<{ tags: { tag: string; count: number; ratio: number }[] }>(
        "/tags/suggest-profile-triggers",
        {
          item_ids: tagSuggestSelected,
          project_id: selectedProject.id,
          min_count: Math.max(1, Math.floor(tagSuggestSelected.length * 0.5)),
        },
      );
      setTagSuggestResults(res.tags);
      setTagSuggestChosen([]);
    } finally {
      setTagSuggestLoading(false);
    }
  }

  function applyTagsToProfile(tags: string[]) {
    if (tags.length === 0) return;
    const list: PreviewPromptItem[] =
      prompts.prompts && prompts.prompts.length > 0
        ? prompts.prompts
        : [{ label: "デフォルト", positive: prompts.positive_prompt, negative: prompts.negative_prompt }];
    const current = list[tagSuggestProfileIdx]?.trigger_words ?? "";
    const existing = current.split(",").map((t) => t.trim()).filter(Boolean);
    const merged = [...new Set([...existing, ...tags])].join(", ");
    const next = list.map((p, j) =>
      j === tagSuggestProfileIdx ? { ...p, trigger_words: merged } : p,
    );
    onPromptsChange({ ...prompts, prompts: next });
    setTagSuggestOpen(false);
    showNotice("トリガーワードを追加しました");
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
      {/* Performance Modal */}
      {showPerfModal && (
        <PerfModal
          mixedPrecision={mixedPrecision} setMixedPrecision={setMixedPrecision}
          savePrecision={savePrecision} setSavePrecision={setSavePrecision}
          xformers={xformers} setXformers={setXformers}
          cacheLatents={cacheLatents} setCacheLatents={setCacheLatents}
          cacheLatentsToDisk={cacheLatentsToDisk} setCacheLatentsToDisk={setCacheLatentsToDisk}
          gradientCheckpointing={gradientCheckpointing} setGradientCheckpointing={setGradientCheckpointing}
          persistentWorkers={persistentWorkers} setPersistentWorkers={setPersistentWorkers}
          maxWorkers={maxWorkers} setMaxWorkers={setMaxWorkers}
          unetOnly={unetOnly} setUnetOnly={setUnetOnly}
          onClose={() => setShowPerfModal(false)}
        />
      )}

      {/* Review Gate Modal */}
      {reviewModalOpen && (
        <ReviewGateModal
          report={report}
          preflight={preflight}
          loading={reportLoading}
          onConfirm={() => void confirmAndStartTraining()}
          onCancel={() => setReviewModalOpen(false)}
        />
      )}

      {/* プレビュー拡大表示（ダブルクリックで開く・クリックで閉じる） */}
      {zoomImg && (
        <div
          className="fixed inset-0 z-50 bg-black/85 flex items-center justify-center p-6 cursor-zoom-out"
          onClick={() => setZoomImg(null)}
        >
          <img
            src={zoomImg}
            alt="preview enlarged"
            className="max-w-[95vw] max-h-[95vh] object-contain rounded-lg shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          />
          <button
            className="absolute top-4 right-4 text-gray-200 bg-gray-800/80 hover:bg-gray-700 rounded-full w-9 h-9 flex items-center justify-center"
            onClick={() => setZoomImg(null)}
            aria-label="閉じる"
          >
            ×
          </button>
        </div>
      )}

      {/* ── タグ提案モーダル ── */}
      {tagSuggestOpen && (
        <div className="fixed inset-0 z-50 bg-black/70 flex items-center justify-center p-4">
          <div className="bg-gray-900 border border-gray-700 rounded-xl w-full max-w-2xl max-h-[85vh] flex flex-col shadow-2xl">
            {/* ヘッダー */}
            <div className="flex items-center justify-between px-5 py-3 border-b border-gray-700">
              <div className="flex items-center gap-2">
                <Tag size={14} className="text-amber-400" />
                <span className="text-sm font-semibold text-gray-100">参照画像からトリガーワードを提案</span>
              </div>
              <button onClick={() => setTagSuggestOpen(false)} className="text-gray-400 hover:text-gray-200">
                <X size={16} />
              </button>
            </div>
            {/* Step 1 */}
            <div className="px-5 py-3 border-b border-gray-700/60">
              <p className="text-xs text-gray-400 mb-2">
                <span className="text-amber-300 font-semibold">Step 1</span> — 共通タグを調べる参照画像を選択（2〜10枚）
                <span className="ml-2 text-gray-500">{tagSuggestSelected.length} 枚選択中</span>
              </p>
              <div className="grid grid-cols-5 gap-1.5 max-h-48 overflow-y-auto">
                {tagSuggestItems.map((item) => {
                  const selected = tagSuggestSelected.includes(item.id);
                  return (
                    <button
                      key={item.id}
                      onClick={() =>
                        setTagSuggestSelected((prev) =>
                          selected ? prev.filter((id) => id !== item.id) : prev.length < 10 ? [...prev, item.id] : prev,
                        )
                      }
                      className={`relative aspect-square rounded overflow-hidden border-2 transition-colors ${selected ? "border-amber-400" : "border-gray-700 hover:border-gray-500"}`}
                    >
                      <img
                        src={`/collector/thumbnail?path=${encodeURIComponent(item.file_path)}&size=80`}
                        alt=""
                        className="w-full h-full object-cover"
                      />
                      {selected && (
                        <div className="absolute inset-0 bg-amber-400/20 flex items-center justify-center">
                          <CheckCircle2 size={16} className="text-amber-300 drop-shadow" />
                        </div>
                      )}
                    </button>
                  );
                })}
                {tagSuggestItems.length === 0 && (
                  <p className="col-span-5 text-xs text-gray-500 py-4 text-center">
                    このプロジェクトに画像がありません
                  </p>
                )}
              </div>
              <button
                onClick={() => void analyzeRefImages()}
                disabled={tagSuggestSelected.length < 2 || tagSuggestLoading}
                className="mt-2 flex items-center gap-1.5 text-xs bg-amber-800/60 hover:bg-amber-700/60 disabled:opacity-40 disabled:cursor-not-allowed text-amber-200 border border-amber-700/50 rounded px-3 py-1 transition-colors"
              >
                {tagSuggestLoading ? <Loader2 size={12} className="animate-spin" /> : <Tag size={12} />}
                この {tagSuggestSelected.length} 枚から共通タグを提案
              </button>
            </div>
            {/* Step 2 */}
            {tagSuggestResults.length > 0 && (
              <div className="px-5 py-3 flex-1 overflow-y-auto">
                <p className="text-xs text-gray-400 mb-2">
                  <span className="text-amber-300 font-semibold">Step 2</span> — 追加するタグを選択
                </p>
                <div className="flex flex-wrap gap-1.5 mb-3">
                  {tagSuggestResults.map(({ tag, ratio }) => {
                    const chosen = tagSuggestChosen.includes(tag);
                    const opacity = Math.round(40 + ratio * 55);
                    return (
                      <button
                        key={tag}
                        onClick={() =>
                          setTagSuggestChosen((prev) =>
                            chosen ? prev.filter((t) => t !== tag) : [...prev, tag],
                          )
                        }
                        className={`flex items-center gap-1 text-xs rounded-full px-2.5 py-0.5 border transition-colors ${chosen ? "bg-amber-600/70 border-amber-500 text-amber-100" : "bg-gray-800 border-gray-600 text-gray-300 hover:border-amber-600"}`}
                        style={{ opacity: chosen ? 1 : opacity / 100 + 0.3 }}
                      >
                        {tag}
                        <span className="text-[10px] text-gray-400">{Math.round(ratio * 100)}%</span>
                      </button>
                    );
                  })}
                </div>
                <button
                  onClick={() => applyTagsToProfile(tagSuggestChosen)}
                  disabled={tagSuggestChosen.length === 0}
                  className="flex items-center gap-1.5 text-xs bg-indigo-700/70 hover:bg-indigo-600/80 disabled:opacity-40 disabled:cursor-not-allowed text-indigo-100 rounded px-3 py-1.5 transition-colors"
                >
                  <CheckCircle2 size={12} />
                  選択した {tagSuggestChosen.length} タグをトリガーワードに追加
                </button>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── 学習時間予測（最上部・全幅） ── */}
      <div className="max-w-5xl mb-2">
        <EstimateCard
          estimate={estimate}
          loading={estimateLoading}
          collapsed={!showEstimate}
          onToggle={() => setShowEstimate((s) => !s)}
        />
      </div>

      <div className="grid grid-cols-1 gap-3 max-w-5xl xl:grid-cols-[1fr,360px]">
        {/* ── 左カラム ── */}
        <div className="space-y-3">
          {/* Status Card */}
          <div className="bg-gray-800 border border-gray-700 rounded-xl overflow-hidden">
            <button
              className="w-full flex items-center justify-between px-3 py-2 hover:bg-gray-750 transition-colors"
              onClick={() => setShowStatus((s) => !s)}
            >
              <div className="flex items-center gap-2 min-w-0">
                <h2 className="text-sm font-bold text-gray-100 truncate">{selectedProject.name}</h2>
                {trainingMode && (
                  <span
                    className={`inline-flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded-full font-medium cursor-help shrink-0 ${
                      trainingMode.kohya_ready
                        ? "bg-emerald-900/50 text-emerald-300 border border-emerald-700/50"
                        : "bg-amber-900/40 text-amber-400 border border-amber-700/40"
                    }`}
                    title={trainingMode.reason ? `シミュレーション理由: ${trainingMode.reason}` : trainingMode.message}
                  >
                    {trainingMode.mode === "both" ? <><Zap size={9} /> 全エンジン</>
                      : trainingMode.mode === "musubi" ? <><Zap size={9} /> 拡張エンジン</>
                      : trainingMode.kohya_ready ? <><Zap size={9} /> 標準エンジン</>
                      : <><Cpu size={9} /> sim</>}
                  </span>
                )}
                <span className={`w-1.5 h-1.5 rounded-full ${statusConf.dot} shrink-0`} />
                <span className={`text-xs ${statusConf.color} shrink-0`}>
                  {statusConf.label}
                  {currentStatus?.status === "queued" && currentStatus.queue_position != null && (
                    <> （{currentStatus.queue_position}番目）</>
                  )}
                  {previewPendingForLatestCheckpoint && (
                    <span className="text-cyan-400"> — Preview画像を生成しています</span>
                  )}
                </span>
                {!showStatus && epoch > 0 && (
                  <span className="text-xs text-gray-500 font-normal shrink-0">
                    — E{epoch}/{totalEpochs} · {progress.toFixed(0)}%
                  </span>
                )}
              </div>
              {showStatus ? <ChevronUp size={16} className="text-gray-400 shrink-0" /> : <ChevronDown size={16} className="text-gray-400 shrink-0" />}
            </button>

            {showStatus && (
              <div className="px-3 pb-3 border-t border-gray-700/60 pt-2 space-y-2">
                {/* Metrics — 横1行 */}
                <div className="flex items-center gap-3 py-1 px-1 bg-gray-700/40 rounded-lg border border-gray-700/50">
                  <Metric label="Epoch" value={`${epoch}/${totalEpochs}`} icon={<Layers size={11} />} />
                  <span className="text-gray-700">|</span>
                  <Metric label="Step" value={`${doneSteps}/${totalSteps}`} icon={<BarChart3 size={11} />} />
                  <span className="text-gray-700">|</span>
                  <Metric label="残り" value={etaText(currentStatus?.eta_seconds)} icon={<Clock size={11} />} />
                  {currentStatus?.loss != null && (
                    <>
                      <span className="text-gray-700">|</span>
                      <Metric label="loss" value={Number(currentStatus.loss).toFixed(4)} icon={<BarChart3 size={11} />} />
                    </>
                  )}
                </div>

                {/* Overall progress bar */}
                <div>
                  <div className="flex items-center justify-between text-xs text-gray-400 mb-1">
                    <span>全体進捗</span>
                    <span>{progress.toFixed(1)}%</span>
                  </div>
                  <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
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
                  <div className="flex gap-0.5 h-1.5">
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

                {/* Control buttons */}
                <div className="flex flex-wrap gap-2 pt-1">
                  <button
                    onClick={() => void openReviewGate()}
                    disabled={isTraining || isQueued}
                    className="flex items-center gap-1.5 bg-indigo-600 hover:bg-indigo-500 disabled:bg-gray-700 disabled:text-gray-500 text-white rounded-lg px-3 py-1.5 text-sm font-semibold transition-colors"
                  >
                    {isTraining || isQueued ? <Loader2 size={14} className="animate-spin" /> : <Play size={14} />}
                    学習開始
                  </button>
                  <button
                    onClick={() => void trainingAction("/training/stop-at-epoch", "Epoch区切りで停止予約")}
                    disabled={!isTraining}
                    className="flex items-center gap-1.5 bg-blue-900 hover:bg-blue-800 disabled:bg-gray-700 disabled:text-gray-500 text-blue-200 rounded-lg px-3 py-1.5 text-sm font-medium transition-colors"
                  >
                    <PauseCircle size={14} />
                    Epoch停止
                  </button>
                  <button
                    onClick={() => void trainingAction("/training/stop-now", isQueued ? "Queueから削除" : "即時停止")}
                    disabled={!isTraining && !isQueued}
                    className="flex items-center gap-1.5 bg-red-900 hover:bg-red-800 disabled:bg-gray-700 disabled:text-gray-500 text-red-200 rounded-lg px-3 py-1.5 text-sm font-medium transition-colors"
                  >
                    <Square size={14} />
                    {isQueued ? "Queueから削除" : "即停止"}
                  </button>
                  <button
                    onClick={() =>
                      void trainingAction("/training/resume", "学習を再開").then(() =>
                        void loadTimeline(selectedProject.id)
                      )
                    }
                    disabled={isTraining || isCompleted}
                    className="flex items-center gap-1.5 bg-emerald-900 hover:bg-emerald-800 disabled:bg-gray-700 disabled:text-gray-500 text-emerald-200 rounded-lg px-3 py-1.5 text-sm font-medium transition-colors"
                  >
                    <RotateCcw size={14} />
                    再開
                  </button>
                  <button
                    onClick={() => void resetTraining()}
                    disabled={isTraining}
                    title="プレビュー画像を削除して進捗を Step 0 に戻す（履歴・チェックポイントは保持）"
                    className="flex items-center gap-1.5 bg-rose-950 hover:bg-rose-900 disabled:bg-gray-700 disabled:text-gray-500 text-rose-300 rounded-lg px-3 py-1.5 text-sm font-medium transition-colors"
                  >
                    <Trash2 size={14} />
                    キャッシュ削除
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* ── Parameters Card ── */}
          <div className="bg-gray-800 border border-gray-700 rounded-xl overflow-hidden">
            <button
              className="w-full flex items-center justify-between px-3 py-2 text-sm font-semibold text-gray-200 hover:bg-gray-750 transition-colors"
              onClick={() => setShowParams((p) => !p)}
            >
              <span>学習パラメータ</span>
              {showParams ? <ChevronUp size={16} className="text-gray-400" /> : <ChevronDown size={16} className="text-gray-400" />}
            </button>
            {showParams && (
              <div className="px-3 pb-3 space-y-2 border-t border-gray-700">
                <div className="grid grid-cols-2 gap-2 mt-3 sm:grid-cols-4">
                  <ParamInput label="epochs" value={epochsText} onChange={setEpochsText} step={1} min={1} />
                  <ParamInput label="repeats" value={trainRepeatsText} onChange={setTrainRepeatsText} step={1} min={1} />
                  <ParamSelect label="rank" value={rankText} onChange={setRankText} options={[4, 8, 16, 32, 64, 128]} />
                  <ParamSelect label="alpha" value={alphaText} onChange={setAlphaText} options={[1, 2, 4, 8, 16, 32, 64, 128]} />
                  <ParamInput label="save_every_n" value={saveEveryText} onChange={setSaveEveryText} step={1} min={1} />
                  <ParamSelect label="resolution" value={resolutionText} onChange={setResolutionText} options={[512, 576, 640, 704, 768, 832, 896, 1024]} />
                  <ParamSelect label="batch_size" value={batchText} onChange={setBatchText} options={[1, 2, 3, 4, 6, 8]} />
                  <ParamInput label="learning_rate" value={lrText} onChange={setLrText} inputMode="decimal" placeholder="0.0001" />
                  <ParamSelect label="min_snr_gamma" value={minSnrText} onChange={setMinSnrText} options={[0, 1, 3, 5, 8, 10, 20]} />
                  <ParamInput label="output_name" value={outputName} onChange={setOutputName} inputMode="text" wide />
                </div>

                {/* Optimizer / Scheduler selectors */}
                <div className="grid grid-cols-2 gap-2 pt-2 border-t border-gray-700">
                  <label className="block">
                    <span className="text-xs text-gray-400 mb-1 block">Optimizer</span>
                    <select
                      value={optimizerVal}
                      onChange={(e) => {
                        const v = e.target.value;
                        setOptimizerVal(v);
                        if (v === "Prodigy" || v === "DAdaptAdam") setLrText("1");
                      }}
                      className="w-full bg-gray-700 border border-gray-600 text-gray-100 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:border-indigo-500"
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
                      className="w-full bg-gray-700 border border-gray-600 text-gray-100 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:border-indigo-500"
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

                {(optimizerVal === "Prodigy" || optimizerVal === "DAdaptAdam") && (
                  <div className="flex items-start gap-2 bg-yellow-900/40 border border-yellow-700/60 rounded-lg px-3 py-2 text-xs text-yellow-300">
                    <span className="mt-0.5 shrink-0">⚠</span>
                    <span>
                      <strong>{optimizerVal}</strong> は学習率を自動調整するオプティマイザです。
                      learning_rate は <strong>1.0</strong> に固定されます（小さい値を入力しても 1.0 で上書きされます）。
                    </span>
                  </div>
                )}

                <div className="pt-1">
                  <button
                    type="button"
                    onClick={() => setShowPerfModal(true)}
                    className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-slate-700 hover:bg-slate-600 text-slate-300 rounded border border-slate-600 transition-colors"
                  >
                    <Settings size={13} />
                    パフォーマンス詳細設定
                  </button>
                </div>

                <div className="pt-2 border-t border-gray-700">
                  <div className="grid gap-3">
                    <div>
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-xs text-gray-400">
                          base checkpoint（学習元モデル）
                        </span>
                        <div className="flex items-center gap-2">
                          {ckptLoading && <Loader2 size={11} className="animate-spin text-gray-500" />}
                          <button
                            type="button"
                            onClick={() => void loadCheckpoints()}
                            className="text-[10px] text-gray-500 hover:text-indigo-300 transition-colors"
                            title="ComfyUI/学習エンジンのモデルフォルダを再スキャン"
                          >再スキャン</button>
                          <button
                            type="button"
                            onClick={() => setCkptManual((v) => !v)}
                            className="text-[10px] text-gray-500 hover:text-indigo-300 transition-colors"
                          >{ckptManual ? "一覧から選択" : "手動入力"}</button>
                        </div>
                      </div>
                      {ckptManual ? (
                        <input
                          value={baseCkpt}
                          onChange={(e) => setBaseCkpt(e.target.value)}
                          placeholder="C:\\...\\model.safetensors"
                          className="w-full bg-gray-700 border border-gray-600 text-gray-100 placeholder-gray-500 rounded-lg px-3 py-2 text-sm font-mono focus:outline-none focus:border-indigo-500"
                        />
                      ) : (() => {
                        const q = ckptSearch.trim().toLowerCase();
                        const filtered = q
                          ? checkpoints.filter(
                              (c) =>
                                c.name.toLowerCase().includes(q) ||
                                c.source.toLowerCase().includes(q)
                            )
                          : checkpoints;
                        return (
                          <div className="space-y-1.5">
                            {checkpoints.length > 0 && (
                              <div className="relative">
                                <Search
                                  size={13}
                                  className="absolute left-2.5 top-1/2 -translate-y-1/2 text-gray-500 pointer-events-none"
                                />
                                <input
                                  value={ckptSearch}
                                  onChange={(e) => setCkptSearch(e.target.value)}
                                  placeholder={`モデルを検索…（${checkpoints.length}件）`}
                                  className="w-full bg-gray-700 border border-gray-600 text-gray-100 placeholder-gray-500 rounded-lg pl-8 pr-7 py-2 text-sm focus:outline-none focus:border-indigo-500"
                                />
                                {ckptSearch && (
                                  <button
                                    type="button"
                                    onClick={() => setCkptSearch("")}
                                    className="absolute right-2 top-1/2 -translate-y-1/2 text-gray-500 hover:text-gray-300"
                                    title="クリア"
                                  >
                                    <XCircle size={13} />
                                  </button>
                                )}
                              </div>
                            )}
                            <select
                              value={baseCkpt}
                              onChange={(e) => setBaseCkpt(e.target.value)}
                              className="w-full bg-gray-700 border border-gray-600 text-gray-100 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:border-indigo-500"
                            >
                              <option value="">
                                {checkpoints.length === 0
                                  ? "モデルが見つかりません（設定でComfyUIパスを確認）"
                                  : filtered.length === 0
                                  ? "該当するモデルがありません"
                                  : "モデルを選択..."}
                              </option>
                              {filtered.map((c) => (
                                <option key={c.path} value={c.path}>
                                  {c.name}（{c.source} · {c.size_mb >= 1024 ? `${(c.size_mb / 1024).toFixed(1)}GB` : `${c.size_mb}MB`}）
                                </option>
                              ))}
                            </select>
                            {q && (
                              <p className="text-[10px] text-gray-500">
                                {filtered.length} / {checkpoints.length} 件
                              </p>
                            )}
                          </div>
                        );
                      })()}
                      {!ckptManual && checkpoints.length === 0 && !ckptLoading && (
                        <p className="text-[10px] text-amber-400/80 mt-1">
                          ComfyUI の models/checkpoints にモデルを置くか、「設定」で comfyui_root を指定してください。
                        </p>
                      )}
                    </div>
                    <PathInput
                      label="train dir"
                      value={trainDir}
                      onChange={setTrainDir}
                    />
                    {/* Dataset Builder の processed/ 成果物を明示的に選択できるようにする。
                        存在するだけでは選択可能にせず、Pipeline完了マニフェストがある場合のみ有効化する。 */}
                    <div className="space-y-1">
                      <label className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider">
                        Dataset入力
                      </label>
                      <select
                        value={datasetSource}
                        onChange={(e) => setDatasetSource(e.target.value as "original" | "processed")}
                        className="w-full bg-gray-700 border border-gray-600 text-gray-100 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:border-indigo-500"
                      >
                        <option value="original">元画像（dataset直下）</option>
                        <option value="processed" disabled={!processedAvailable}>
                          Dataset Builder処理済み{processedAvailable ? `（${processedManifest?.success ?? 0}枚）` : "（未実行）"}
                        </option>
                      </select>
                      {datasetSource === "processed" && processedManifest?.generated_at && (
                        <p className="text-[10px] text-gray-500">
                          生成日時: {new Date(processedManifest.generated_at).toLocaleString()}
                        </p>
                      )}
                    </div>
                    <div className="space-y-1 mt-3">
                      <label className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider">学習エンジン</label>
                      <select value={trainingEngine} onChange={(e) => setTrainingEngine(e.target.value)}
                        className="w-full bg-gray-700 border border-gray-600 text-gray-100 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:border-indigo-500">
                        <option value="auto">自動（モデル既定）</option>
                        <option value="ai_toolkit">AI Toolkit（Krea 2 / Flux）</option>
                        <option value="musubi">musubi-tuner（対応 DiT）</option>
                        <option value="kohya">kohya_ss（SD / ANIMA）</option>
                      </select>
                    </div>
                    <PathInput label="reg dir" value={regDir} onChange={setRegDir} />

                    {/* モデルファミリー選択 */}
                    <div className="space-y-1">
                      <label className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider flex items-center gap-1.5">
                        モデルファミリー
                        <span className="text-[9px] font-normal text-gray-500 normal-case">
                          (バックエンド自動選択)
                        </span>
                      </label>
                      <select
                        value={modelFamily}
                        onChange={(e) => setModelFamily(e.target.value)}
                        className="w-full bg-gray-700 border border-gray-600 text-gray-100 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:border-indigo-500"
                      >
                        <option value="auto">自動検出（ファイル名・ヘッダから判定）</option>
                        <optgroup label="標準エンジン">
                          {modelSpecs.filter((s) => s.training_backend === "sdxl").map((s) => (
                            <option key={s.model_family} value={s.model_family}>{s.display_name}</option>
                          ))}
                        </optgroup>
                        <optgroup label="拡張エンジン">
                          {modelSpecs.filter((s) => s.training_backend !== "sdxl").map((s) => (
                            <option key={s.model_family} value={s.model_family}>{s.display_name}</option>
                          ))}
                          <option value="custom">カスタム</option>
                        </optgroup>
                      </select>
                      {/* バックエンドインジケーター
                          training_backend値ごとに対応するreadyフラグを引く(sdxl以外は
                          全てmusubi、という二値判定はANIMA追加で成立しなくなったため、
                          training_backend文字列で個別に判定する)。*/}
                      {trainingMode && (() => {
                        const spec = modelFamily === "custom"
                          ? null
                          : modelSpecs.find((s) => s.model_family === modelFamily);
                        const backendKey = spec?.training_backend ?? (modelFamily === "custom" ? "musubi" : "sdxl");
                        const ready =
                          backendKey === "sdxl" ? trainingMode.kohya_ready
                          : backendKey === "anima" ? trainingMode.anima_ready
                          : trainingMode.musubi_ready;
                        const label = backendKey === "sdxl" ? "標準エンジン" : "拡張エンジン";
                        return ready
                          ? <p className="text-[10px] text-emerald-400 mt-0.5">✓ {label} 接続済み</p>
                          : <p className="text-[10px] text-amber-400 mt-0.5">⚠ {label}未設定 — 設定タブでパスを指定</p>;
                      })()}
                    </div>
                  </div>
                </div>

              </div>
            )}
          </div>

          {/* ── Training Log Viewer ── */}
          <div className="bg-gray-800 border border-gray-700 rounded-xl overflow-hidden">
            <button
              className="w-full flex items-center justify-between px-3 py-2 text-sm font-semibold text-gray-200 hover:bg-gray-750 transition-colors"
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

          {/* ── 全体Queue(他プロジェクト含む、/training/queue) ── */}
          {queueState && (queueState.queue_length > 0 || queueState.running) && (
            <div className="bg-gray-800 border border-gray-700 rounded-xl overflow-hidden">
              <button
                className="w-full flex items-center justify-between px-3 py-2 text-sm font-semibold text-gray-200 hover:bg-gray-750 transition-colors"
                onClick={() => setShowQueue((s) => !s)}
              >
                <span className="flex items-center gap-2">
                  <ListOrdered size={15} className="text-cyan-400" />
                  学習Queue
                  {queueState.queue_length > 0 && (
                    <span className="text-xs text-cyan-400 font-normal">
                      {queueState.queue_length}件待機中
                    </span>
                  )}
                </span>
                {showQueue ? <ChevronUp size={16} className="text-gray-400" /> : <ChevronDown size={16} className="text-gray-400" />}
              </button>
              {showQueue && (
                <div className="px-3 pb-3 border-t border-gray-700/60 pt-2 space-y-1.5">
                  {queueState.running && (
                    <div className="flex items-center justify-between text-xs bg-gray-700/40 rounded-lg px-2.5 py-1.5">
                      <span className="flex items-center gap-1.5 text-gray-200 truncate">
                        <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse shrink-0" />
                        {projects.find((p) => p.id === queueState.running!.project_id)?.name
                          ?? `project #${queueState.running.project_id}`}
                      </span>
                      <span className="text-emerald-400 shrink-0">実行中</span>
                    </div>
                  )}
                  {queueState.queued.map((q) => (
                    <div
                      key={q.run_id}
                      className="flex items-center justify-between text-xs bg-gray-700/20 rounded-lg px-2.5 py-1.5"
                    >
                      <span className="flex items-center gap-1.5 text-gray-300 truncate">
                        <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 shrink-0" />
                        {projects.find((p) => p.id === q.project_id)?.name ?? `project #${q.project_id}`}
                      </span>
                      <span className="text-cyan-400 shrink-0">{q.queue_position}番目</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* ── Resource Monitor (§18) ── */}
          <div className="bg-gray-800 border border-gray-700 rounded-xl overflow-hidden">
            <button
              className="w-full flex items-center justify-between px-3 py-2 text-sm font-semibold text-gray-200 hover:bg-gray-750 transition-colors"
              onClick={() => setShowResources((s) => !s)}
            >
              <span className="flex items-center gap-2">
                <Cpu size={15} className="text-sky-400" />
                リソースモニター
                {isTraining && showResources && (
                  <span className="inline-flex items-center gap-1 text-xs text-sky-400">
                    <span className="w-1.5 h-1.5 rounded-full bg-sky-400 animate-pulse" />
                    LIVE
                  </span>
                )}
                {resources && (
                  <span className="text-xs text-gray-500 font-normal">
                    CPU {resources.cpu_pct.toFixed(0)}% | RAM {resources.ram_used_gb}GB
                    {resources.gpu_available && resources.gpu[0] &&
                      ` | VRAM ${resources.gpu[0].vram_used_mb}MB`}
                  </span>
                )}
              </span>
              {showResources ? <ChevronUp size={16} className="text-gray-400" /> : <ChevronDown size={16} className="text-gray-400" />}
            </button>
            {showResources && (
              <div className="px-5 pb-5 border-t border-gray-700 space-y-4 mt-4">
                {!resources ? (
                  <p className="text-xs text-gray-500">読み込み中...</p>
                ) : (
                  <>
                    {/* CPU / RAM */}
                    <div className="grid grid-cols-2 gap-3">
                      <ResourceBar
                        label="CPU"
                        pct={resources.cpu_pct}
                        text={`${resources.cpu_pct.toFixed(1)}%`}
                        color="bg-sky-500"
                      />
                      <ResourceBar
                        label={`RAM  ${resources.ram_used_gb} / ${resources.ram_total_gb} GB`}
                        pct={resources.ram_pct}
                        text={`${resources.ram_pct.toFixed(1)}%`}
                        color={resources.ram_pct > 85 ? "bg-red-500" : "bg-sky-500"}
                      />
                    </div>

                    {/* GPU */}
                    {resources.gpu_available ? (
                      <div className="space-y-3">
                        {resources.gpu.map((g) => (
                          <div key={g.index} className="bg-gray-700/50 rounded-lg p-3 space-y-2">
                            <div className="flex items-center justify-between text-xs">
                              <span className="text-gray-300 font-medium truncate">{g.name}</span>
                              {g.temperature != null && (
                                <span className={`font-mono ml-2 shrink-0 ${g.temperature >= 80 ? "text-red-400" : g.temperature >= 70 ? "text-amber-400" : "text-gray-400"}`}>
                                  {g.temperature}°C
                                </span>
                              )}
                            </div>
                            <ResourceBar
                              label={`VRAM  ${g.vram_used_mb} / ${g.vram_total_mb} MB`}
                              pct={g.vram_pct}
                              text={`${g.vram_pct.toFixed(1)}%`}
                              color={g.vram_pct > 90 ? "bg-red-500" : g.vram_pct > 75 ? "bg-amber-500" : "bg-emerald-500"}
                            />
                            <ResourceBar
                              label="GPU Util"
                              pct={g.gpu_util_pct}
                              text={`${g.gpu_util_pct.toFixed(0)}%`}
                              color="bg-violet-500"
                            />
                          </div>
                        ))}
                      </div>
                    ) : (
                      <p className="text-xs text-gray-500">GPU 情報なし（nvidia-smi / pynvml 未検出）</p>
                    )}
                  </>
                )}
              </div>
            )}
          </div>

          {/* ── Preview Timeline + Prompts ── */}
          <div className="bg-gray-800 border border-gray-700 rounded-xl overflow-hidden">
            {/* カードヘッダー（常時表示） */}
            <div className="flex items-center justify-between px-3 py-2 border-b border-gray-700">
              <h3 className="text-sm font-semibold text-gray-200 flex items-center gap-2">
                <ImageIcon size={14} className="text-indigo-400" />プレビュー
              </h3>
              <button
                onClick={() => void loadTimeline(selectedProject.id)}
                className="p-1 rounded hover:bg-gray-700 text-gray-500 hover:text-gray-300 transition-colors"
                title="更新"
              >
                {loadingTimeline ? (
                  <Loader2 size={13} className="animate-spin" />
                ) : (
                  <RotateCcw size={13} />
                )}
              </button>
            </div>

            {/* サブパネル: プロンプト設定（デフォルト非表示） */}
            <div className="border-b border-gray-700">
              <button
                className="w-full flex items-center justify-between px-3 py-1.5 text-xs font-semibold text-gray-400 hover:bg-gray-750 hover:text-gray-300 transition-colors"
                onClick={() => setShowPrompts((s) => !s)}
              >
                <span className="flex items-center gap-1.5">
                  <Settings size={12} />
                  プロンプト設定
                  {!showPrompts && (
                    <span className="font-normal text-gray-500">
                      （{(prompts.prompts?.length ?? 1) * (prompts.instances_per_prompt ?? 1)} 枚/Checkpoint
                      {" "}={(prompts.prompts?.length ?? 1)}プロンプト×{prompts.instances_per_prompt ?? 1}
                      インスタンス · {prompts.preview_resolution ?? 1024}px）
                    </span>
                  )}
                </span>
                {showPrompts ? <ChevronUp size={14} className="text-gray-500" /> : <ChevronDown size={14} className="text-gray-500" />}
              </button>

              {showPrompts && (() => {
                const list: PreviewPromptItem[] =
                  prompts.prompts && prompts.prompts.length > 0
                    ? prompts.prompts
                    : [{
                        label: "デフォルト",
                        quality: DEFAULT_QUALITY_PROMPT,
                        positive: prompts.positive_prompt || DEFAULT_FEATURE_PROMPT,
                        negative: prompts.negative_prompt || DEFAULT_NEGATIVE_PROMPT,
                        trigger_words: "",
                      }];
                const update = (next: PreviewPromptItem[]) =>
                  onPromptsChange({
                    ...prompts,
                    prompts: next,
                    positive_prompt: next[0]?.positive ?? "",
                    negative_prompt: next[0]?.negative ?? "",
                  });
                const setAt = (i: number, patch: Partial<PreviewPromptItem>) =>
                  update(list.map((p, j) => (j === i ? { ...p, ...patch } : p)));
                const add = () =>
                  update([
                    ...list,
                    {
                      label: `プロファイル${list.length + 1}`,
                      quality: list[0]?.quality ?? DEFAULT_QUALITY_PROMPT,
                      positive: "",
                      negative: list[0]?.negative ?? DEFAULT_NEGATIVE_PROMPT,
                      trigger_words: "",
                    },
                  ]);
                const remove = (i: number) => update(list.length > 1 ? list.filter((_, j) => j !== i) : list);
                return (
                  <div className="px-3 py-2 bg-gray-900/30 space-y-2">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span className="text-[10px] text-gray-400 shrink-0">プレビュー解像度</span>
                        <select
                          value={prompts.preview_resolution ?? 1024}
                          onChange={(e) =>
                            onPromptsChange({ ...prompts, preview_resolution: Number(e.target.value) })
                          }
                          className="bg-gray-700 border border-gray-600 text-gray-100 rounded px-2 py-0.5 text-xs focus:outline-none focus:border-indigo-500"
                        >
                          {[512, 768, 1024, 1216, 1536].map((r) => (
                            <option key={r} value={r}>{r}×{r}</option>
                          ))}
                        </select>
                        <span className="text-[10px] text-gray-400 shrink-0 ml-2">インスタンス数/プロンプト</span>
                        <select
                          value={prompts.instances_per_prompt ?? 1}
                          onChange={(e) =>
                            onPromptsChange({ ...prompts, instances_per_prompt: Number(e.target.value) })
                          }
                          className="bg-gray-700 border border-gray-600 text-gray-100 rounded px-2 py-0.5 text-xs focus:outline-none focus:border-indigo-500"
                        >
                          {[1, 2, 3, 4].map((n) => (
                            <option key={n} value={n}>{n}</option>
                          ))}
                        </select>
                        <span className="text-[10px] text-emerald-400 font-medium">
                          期待枚数/Checkpoint: {list.length * (prompts.instances_per_prompt ?? 1)}枚
                          （{list.length}プロンプト×{prompts.instances_per_prompt ?? 1}インスタンス）
                        </span>
                      </div>
                      <button
                        onClick={add}
                        className="text-xs bg-indigo-700/60 hover:bg-indigo-600 text-indigo-100 rounded px-2 py-0.5"
                      >
                        ＋ 追加
                      </button>
                    </div>
                    <div className="flex items-center gap-3 flex-wrap">
                      <span className="text-[10px] text-gray-500 shrink-0">
                        サンプラー/CFG/Steps（空欄 = モデル世代の推奨値を自動使用）
                      </span>
                      <label className="flex items-center gap-1">
                        <span className="text-[10px] text-gray-400">sampler</span>
                        <input
                          value={prompts.preview_sampler ?? ""}
                          placeholder="auto"
                          onChange={(e) => onPromptsChange({ ...prompts, preview_sampler: e.target.value })}
                          className="w-24 bg-gray-700 border border-gray-600 text-gray-100 rounded px-2 py-0.5 text-xs focus:outline-none focus:border-indigo-500 placeholder-gray-600"
                        />
                      </label>
                      <label className="flex items-center gap-1">
                        <span className="text-[10px] text-gray-400">cfg</span>
                        <input
                          type="number"
                          step="0.1"
                          min={0}
                          max={30}
                          value={prompts.preview_cfg ?? ""}
                          placeholder="auto"
                          onChange={(e) =>
                            onPromptsChange({
                              ...prompts,
                              preview_cfg: e.target.value === "" ? null : Number(e.target.value),
                            })
                          }
                          className="w-16 bg-gray-700 border border-gray-600 text-gray-100 rounded px-2 py-0.5 text-xs focus:outline-none focus:border-indigo-500 placeholder-gray-600"
                        />
                      </label>
                      <label className="flex items-center gap-1">
                        <span className="text-[10px] text-gray-400">steps</span>
                        <input
                          type="number"
                          step="1"
                          min={1}
                          max={150}
                          value={prompts.preview_steps ?? ""}
                          placeholder="auto"
                          onChange={(e) =>
                            onPromptsChange({
                              ...prompts,
                              preview_steps: e.target.value === "" ? null : Number(e.target.value),
                            })
                          }
                          className="w-16 bg-gray-700 border border-gray-600 text-gray-100 rounded px-2 py-0.5 text-xs focus:outline-none focus:border-indigo-500 placeholder-gray-600"
                        />
                      </label>
                    </div>
                    {list.map((p, i) => (
                      <div key={i} className="rounded-lg border border-gray-700 bg-gray-800/50 p-2 space-y-1.5">
                        <div className="flex items-center gap-2">
                          <input
                            value={p.label}
                            placeholder={`プロファイル${i + 1}`}
                            onChange={(e) => setAt(i, { label: e.target.value })}
                            className="flex-1 bg-gray-700 border border-gray-600 text-gray-200 rounded px-2 py-0.5 text-xs focus:outline-none focus:border-indigo-500"
                          />
                          {list.length > 1 && (
                            <button onClick={() => remove(i)} className="text-red-400/80 hover:text-red-300 text-xs px-1.5" title="削除">×</button>
                          )}
                        </div>
                        <label className="block">
                          <span className="text-[10px] text-sky-400/90 mb-0.5 block">品質ポジティブ</span>
                          <input
                            value={p.quality ?? ""}
                            placeholder="masterpiece, best quality, very aesthetic, ..."
                            onChange={(e) => setAt(i, { quality: e.target.value })}
                            className="w-full bg-gray-700 border border-gray-600 text-gray-100 rounded px-2 py-0.5 text-xs focus:outline-none focus:border-sky-500"
                          />
                        </label>
                        <label className="block">
                          <span className="text-[10px] text-emerald-400/90 mb-0.5 block">特徴ポジティブ</span>
                          <input
                            value={p.positive}
                            placeholder="キャラ/シーン固有のタグ (例: long hair, smile)"
                            onChange={(e) => setAt(i, { positive: e.target.value })}
                            className="w-full bg-gray-700 border border-gray-600 text-gray-100 rounded px-2 py-0.5 text-xs focus:outline-none focus:border-emerald-500"
                          />
                        </label>
                        <label className="block">
                          <span className="text-[10px] text-red-400/90 mb-0.5 block">ネガティブ</span>
                          <input
                            value={p.negative}
                            placeholder="lowres, worst quality, ..."
                            onChange={(e) => setAt(i, { negative: e.target.value })}
                            className="w-full bg-gray-700 border border-gray-600 text-gray-100 rounded px-2 py-0.5 text-xs focus:outline-none focus:border-red-500"
                          />
                        </label>
                        <div className="flex items-center gap-1.5">
                          <Tag size={10} className="text-amber-400 shrink-0" />
                          <input
                            value={p.trigger_words ?? ""}
                            onChange={(e) => setAt(i, { trigger_words: e.target.value })}
                            placeholder="トリガーワード (例: 1girl, school uniform)"
                            className="flex-1 bg-gray-700/80 border border-amber-800/40 text-amber-200 rounded px-2 py-0.5 text-xs focus:outline-none focus:border-amber-500 placeholder-gray-600"
                          />
                          <button
                            onClick={() => void openTagSuggest(i)}
                            className="flex items-center gap-1 text-[10px] bg-amber-900/40 hover:bg-amber-800/50 text-amber-300 border border-amber-700/40 rounded px-1.5 py-0.5 transition-colors whitespace-nowrap"
                            title="参照画像を選んでタグを提案"
                          >
                            <ImageIcon size={9} />
                            参照画像から提案
                          </button>
                        </div>
                      </div>
                    ))}
                    <div className="flex items-center gap-2">
                      <button
                        onClick={() => void savePreviewPrompts()}
                        className="bg-gray-700 hover:bg-gray-600 text-gray-300 rounded-lg px-3 py-1 text-xs font-medium transition-colors"
                      >
                        保存
                      </button>
                      <p className="text-[10px] text-gray-500">次回学習から Epoch ごとに横並びで生成されます。</p>
                    </div>
                  </div>
                );
              })()}
            </div>

            {/* サブパネル: プレビュー履歴 */}
            <div>
              <button
                className="w-full flex items-center justify-between px-3 py-1.5 text-xs font-semibold text-gray-400 hover:bg-gray-750 hover:text-gray-300 transition-colors"
                onClick={() => setShowTimeline((s) => !s)}
              >
                <span className="flex items-center gap-1.5">
                  <ImageIcon size={12} />
                  プレビュー履歴
                  {!showTimeline && timeline.length > 0 && (
                    <span className="font-normal text-gray-500">（{timeline.length} チェックポイント）</span>
                  )}
                </span>
                {showTimeline ? <ChevronUp size={14} className="text-gray-500" /> : <ChevronDown size={14} className="text-gray-500" />}
              </button>
              {showTimeline && (
                <div className="p-3">
                  <PreviewTimeline
                    timeline={timeline}
                    loadingTimeline={loadingTimeline}
                    promptLabels={promptLabels}
                    onZoom={setZoomImg}
                    onRetry={(checkpointId) => void handleRetryPreview(checkpointId)}
                    retryingCheckpointId={retryingCheckpointId}
                  />
                </div>
              )}
            </div>
          </div>
        </div>

        {/* ── 右カラム ── */}
        <div className="space-y-3">
          {/* ── Training Profiles Card ── */}
          <div className="bg-gray-800 border border-gray-700 rounded-xl overflow-hidden">
            <button
              className="w-full flex items-center justify-between px-3 py-2 text-sm font-semibold text-gray-200 hover:bg-gray-750 transition-colors"
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
              <div className="px-3 pb-3 border-t border-gray-700">
                <div className="grid grid-cols-2 gap-1.5 mt-3">
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
                {/* 現在の設定を保存 */}
                <SavePresetRow
                  onSave={async (name) => {
                    const payload = {
                      type: "custom",
                      description: `カスタム — ${name}`,
                      rank: parseIntOr(rankText, 16, 1),
                      alpha: parseFloatOr(alphaText, 4, 0.1),
                      repeats: parseIntOr(trainRepeatsText, 5, 1),
                      epochs: parseIntOr(epochsText, 5, 1),
                      resolution: parseIntOr(resolutionText, 512, 256),
                      optimizer: optimizerVal,
                      scheduler: schedulerVal,
                      save_every_n_epochs: parseIntOr(saveEveryText, 1, 1),
                      min_snr_gamma: minSnrText !== "" ? parseIntOr(minSnrText, 5, 0) : null,
                      output_name: outputName,
                      learning_rate: parseFloatOr(lrText, 1e-4, 1e-7),
                      train_batch_size: parseIntOr(batchText, 2, 1),
                      xformers,
                      cache_latents: cacheLatents,
                      cache_latents_to_disk: cacheLatentsToDisk,
                      gradient_checkpointing: gradientCheckpointing,
                      mixed_precision: mixedPrecision,
                      save_precision: savePrecision,
                      persistent_data_loader_workers: persistentWorkers,
                      max_data_loader_n_workers: maxWorkers,
                      network_train_unet_only: unetOnly,
                    };
                    const created = await apiPost<Preset>("/presets", { name, payload });
                    setPresets((prev) => [...prev, created]);
                    setSelectedPresetId(created.id);
                    showNotice(`プロファイル「${name}」を保存しました`);
                  }}
                />
              </div>
            )}
          </div>

          {/* Training mode info */}
          {trainingMode && (
            <div className={`rounded-xl overflow-hidden border ${
              (trainingMode.kohya_ready || trainingMode.musubi_ready) ? "bg-emerald-900/10 border-emerald-700/30" : "bg-gray-700/30 border-gray-600"
            }`}>
              <button
                className="w-full flex items-center justify-between px-3 py-1.5 hover:bg-white/5 transition-colors"
                onClick={() => setShowTrainingMode((s) => !s)}
              >
                <span className="flex items-center gap-1.5 text-xs font-semibold text-gray-300">
                  {trainingMode.mode === "both"
                    ? <><Zap size={12} className="text-emerald-400" /> 全エンジン接続済み</>
                    : trainingMode.mode === "musubi"
                    ? <><Zap size={12} className="text-emerald-400" /> 拡張エンジン接続済み</>
                    : trainingMode.kohya_ready
                    ? <><Zap size={12} className="text-emerald-400" /> 標準エンジン接続済み</>
                    : <><Cpu size={12} className="text-gray-400" /> シミュレーションモード</>}
                </span>
                {showTrainingMode ? <ChevronUp size={14} className="text-gray-500" /> : <ChevronDown size={14} className="text-gray-500" />}
              </button>
              {showTrainingMode && (
                <div className="text-xs text-gray-400 px-3 pb-2 space-y-0.5">
                  {trainingMode.kohya_ready && <p>標準エンジン: {trainingMode.kohya_root}</p>}
                  {trainingMode.musubi_ready && <p>拡張エンジン: {trainingMode.musubi_root}</p>}
                  {!trainingMode.kohya_ready && !trainingMode.musubi_ready && (
                    <p>{"設定タブでエンジンのパスを設定すると実学習が有効になります。"}</p>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </>
  );
}
