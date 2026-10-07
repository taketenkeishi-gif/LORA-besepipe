import {
  AlertTriangle,
  ArrowRight,
  Bot,
  Check,
  CheckCircle2,
  ChevronRight,
  CircleDot,
  Cpu,
  Database,
  FileImage,
  Images,
  Loader2,
  LockKeyhole,
  MoreHorizontal,
  Maximize2,
  Play,
  RefreshCw,
  ScanFace,
  Sparkles,
  Upload,
  WandSparkles,
  X,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState, type KeyboardEvent, type MouseEvent } from "react";
import { API_BASE, apiFormPost, apiGet, apiPost } from "../../lib/api";
import type { Project } from "../../types";

type Role = "front" | "left" | "back" | "right" | "detail";
type Decision = "pending" | "approved" | "rejected";

type Asset = {
  id: number;
  asset_key: string;
  file_path: string;
  content_sha256?: string;
  origin_kind: string;
  source_ref?: string;
  review_status: string;
  training_input?: string;
  metadata?: { role?: string; reference_order?: number; turnaround_set_id?: string; identity_lock?: Record<string, boolean> };
};

type Concept = { id: number; name: string; trigger_token: string; concept_type: string };
type Snapshot = { id: number; name: string; item_count: number; snapshot_hash: string; created_at?: string; preview_profile?: { id: number; name: string; created: boolean } | null; preview_profile_snapshot?: { id: number; name: string; created: boolean } | null };
type Workspace = { concepts: Concept[]; assets: Asset[]; snapshots: Snapshot[] };
type Frame = {
  asset_id: number;
  file_path: string;
  ordinal: number;
  review_status: Decision;
  review_status_frozen?: boolean;
  rejection_reason?: string;
  blur_score?: number;
  clip_id?: string;
  clip_index?: number;
  clip_frame_index?: number;
  timestamp_seconds?: number;
  generation_seed?: number;
  generation_prompt?: string;
  coverage?: Record<string, string[]>;
};
type AssetVersion = {
  id: number;
  version_kind: string;
  file_path: string;
  content_sha256: string;
  prompt: string;
  seed?: number;
  status: string;
  selected?: boolean;
};
type VersionResult = {
  training_input_choice?: "original" | "qwen" | "enhanced" | null;
  selected_version_id?: number | null;
  selected_qwen_version_id?: number | null;
  aesthetic_review_state?: "AESTHETIC_UNREVIEWED" | "USER_APPROVED" | "USER_REJECTED";
  aesthetic_reviewed_by?: string | null;
  aesthetic_review_note?: string;
  training_input_mutable?: boolean;
  training_input_lock_reason?: string;
  frozen_snapshot_id?: number | null;
  versions: AssetVersion[];
};
type AestheticConfirmation =
  | { kind: "identity"; approvedCount: number; rejectedCount: number }
  | { kind: "original"; assetIds: number[] }
  | { kind: "qwen"; assetId: number; version: AssetVersion }
  | null;
type StageRecord = { stage: string; status: string; note?: string };
type GenerationRun = {
  id: number;
  status: string;
  current_stage: string;
  created_at: string;
  updated_at?: string;
  error_detail?: string;
  requested: Record<string, unknown>;
  resolved: Record<string, unknown>;
  observed: Record<string, unknown>;
  output_manifest: Record<string, unknown>;
  stage_history: StageRecord[];
  queue_position?: number | null;
  queue_head?: boolean;
  ahead_run_id?: number | null;
};
type PreflightCheck = { label: string; ok: boolean; detail: string };
type Preflight = { status: string; message: string; checks: PreflightCheck[] };
type MenuState =
  | { kind: "reference"; assetId: number; x: number; y: number }
  | { kind: "frame"; assetId: number; x: number; y: number }
  | null;

const roles: Array<{ id: Role; label: string; hint: string }> = [
  { id: "front", label: "Front", hint: "顔・正面形状" },
  { id: "left", label: "Left", hint: "左側面・奥行き" },
  { id: "back", label: "Back", hint: "髪・背面衣装" },
  { id: "right", label: "Right", hint: "右側面・奥行き" },
  { id: "detail", label: "Detail", hint: "装飾・局所特徴" },
];

const locks = [
  ["face", "顔"],
  ["hair", "髪型・髪色"],
  ["eyes", "目"],
  ["body", "体格"],
  ["outfit", "衣装"],
  ["accessories", "装飾"],
  ["palette", "配色"],
] as const;

const stages = [
  ["planning", "Plan"],
  ["h3_generation", "H3"],
  ["frame_extraction", "Frames"],
  ["blur_filter", "Blur"],
  ["duplicate_filter", "Dedupe"],
  ["technical_selection", "Yaw"],
  ["enhancement", "Enhance"],
  ["identity_review", "Curate"],
  ["qwen_correction", "Enhance"],
  ["balancing", "Balance"],
  ["captioning", "Caption"],
  ["snapshot", "Snapshot"],
] as const;

const coverageDimensions = [
  ["views", "View"],
  ["shots", "Shot"],
  ["poses", "Pose"],
  ["expressions", "Expression"],
  ["backgrounds", "Background"],
] as const;

const coveragePresetClips: Record<string, number> = { turntable: 1, essential: 2, balanced: 3, complete: 6 };

function frameManifest(run: GenerationRun | null): Frame[] {
  const extraction = run?.output_manifest?.frame_extraction as { frames?: Frame[] } | undefined;
  const frames = extraction?.frames ?? [];
  const identityReview = run?.output_manifest?.identity_review as {
    decisions?: Array<{ asset_id?: number; review_status?: Decision }>;
  } | undefined;
  const decisions = new Map(
    (identityReview?.decisions ?? [])
      .filter((item): item is { asset_id: number; review_status: Decision } => (
        typeof item.asset_id === "number" && ["approved", "rejected"].includes(String(item.review_status))
      ))
      .map((item) => [item.asset_id, item.review_status]),
  );
  if (!decisions.size) return frames;
  return frames.map((frame) => decisions.has(frame.asset_id)
    ? { ...frame, review_status: decisions.get(frame.asset_id)!, review_status_frozen: true }
    : frame);
}

function normalizeDecision(value: string | undefined): Decision {
  return value === "approved" || value === "rejected" ? value : "pending";
}

function isUserApprovedInput(result: VersionResult | undefined): boolean {
  return result?.aesthetic_review_state === "USER_APPROVED" && result.aesthetic_reviewed_by === "user";
}

function nextInstruction(run: GenerationRun | null, references: Asset[], unresolved: number) {
  if (!references.length) return { title: "参照画像を2〜4枚追加", detail: "Frontを基準にBack / Detailを補います。", key: "references" };
  if (!run) return { title: "生成計画を作る", detail: "H3を起動せず、設定と安全条件だけ固定します。", key: "plan" };
  if (run.status === "cancelled" && run.current_stage !== "planning") return { title: "ユーザー審美Reviewを再開", detail: `Run #${run.id}の生成物は保持済みです。GPUを再開せず、Frame採否から確認し直せます。`, key: "reopen-review" };
  if (run.status === "cancelled") return { title: "新しい生成計画へ戻る", detail: `Run #${run.id}の待機は取消済みです。証拠を残したまま新規計画を作れます。`, key: "new-run" };
  if (["waiting", "waiting_qwen", "waiting_enhancement"].includes(run.status)) {
    const memory = run.observed?.gpu_memory as {
      foreign_processes?: unknown[];
      free_mb?: number;
      external_comfyui?: { reachable?: boolean; running?: number | null; pending?: number | null };
    } | undefined;
    const blockers = memory?.foreign_processes?.length ?? 0;
    const external = memory?.external_comfyui;
    const externalActivity = external?.reachable
      ? `外部8188 ${external.running ?? 0}件実行 / ${external.pending ?? 0}件待機`
      : "外部8188 状態不明";
    const stage = run.status === "waiting_qwen" ? "Qwen 2511" : run.status === "waiting_enhancement" ? "FlashVSR / RealESRGAN" : "H3";
    if ((run.queue_position ?? 1) > 1) {
      return { title: `${stage} · GPU1待機キュー #${run.queue_position}`, detail: `前のRun #${run.ahead_run_id ?? "unknown"}を待機中 · 外部GPUプロセスには触れません`, key: "waiting" };
    }
    return { title: `${stage} · GPU1待機キューで自動再評価中`, detail: `${blockers}件の外部GPUプロセス · ${externalActivity} · 空き ${memory?.free_mb?.toLocaleString() ?? "unknown"} MiB · 安全条件を満たした時だけ開始`, key: "waiting" };
  }
  if (run.status === "running_qwen") return { title: "Qwen 2511補正中", detail: "選択したFrameだけを補正中です。完了まで再投入しません。", key: "running" };
  if (run.status === "running_enhancement") return { title: "8方向の高密度化を直列実行中", detail: "各9フレーム窓をFlashVSR 9/11とRealESRGANで比較生成しています。審美採用は行いません。", key: "running" };
  if (["blocked", "failed"].includes(run.status)) return { title: "Preflightの阻害要因を確認", detail: run.error_detail || "待機キューへ入れると開発を止めずに再評価できます。", key: "blocked" };
  if (["queued", "running"].includes(run.status) && run.current_stage === "planning") {
    const progress = run.observed?.clip_progress as { completed?: number; total?: number; current_clip_id?: string } | undefined;
    return {
      title: `H3 Coverage生成中 · ${progress?.completed ?? 0} / ${progress?.total ?? Number(run.requested?.clip_count ?? 1)} clips`,
      detail: `${progress?.current_clip_id ?? "GPU1へ投入済み"} · 1 clipずつ順次生成し、外部GPUプロセスには触れません`,
      key: "running",
    };
  }
  if (run.current_stage === "planning") return { title: "保存済み計画をGPU1待機へ送る", detail: "Requested / Resolvedを変えず、安全条件を満たすまで自動再評価します。", key: "h3" };
  if (run.current_stage === "h3_generation") return { title: "動画から候補Frameを抽出", detail: "Blurと重複は自動除外、採用はまだ確定しません。", key: "extract" };
  if (run.current_stage === "duplicate_filter") return { title: "8方向の比較Datasetを生成", detail: "元動画から循環9フレーム窓を作り、GPU1待機キューで高密度化します。", key: "enhance" };
  if (run.current_stage === "enhancement" && unresolved) return { title: `${unresolved}件のIdentity判定`, detail: "完成した比較シートを見て、採用するDataset入力を判断してください。", key: "review" };
  if (run.current_stage === "enhancement") return { title: "Identity Reviewを確定", detail: "全Frameの判定をRunへ保存します。", key: "commit-review" };
  if (run.current_stage === "identity_review") return { title: "Training Inputを選ぶ", detail: "良いFrameはOriginal、必要なセルだけQwen補正へ送ります。", key: "qwen" };
  if (run.current_stage === "qwen_correction") return { title: "Dataset構成を固定", detail: "明示採用したOriginal / Correctedを検証してBalanceします。", key: "balance" };
  if (run.current_stage === "balancing") return { title: "Training Captionを確定", detail: "承認Frameすべてに学習入力を保存します。", key: "caption" };
  if (run.current_stage === "captioning") return { title: "Dataset Snapshotを封印", detail: "明示採用したOriginal / CorrectedとCaptionを不変化します。", key: "snapshot" };
  if (run.current_stage === "snapshot") return { title: "学習用Datasetが完成", detail: "Trainへ渡せる不変Snapshotです。", key: "done" };
  return { title: "Run Evidenceを更新", detail: "現在段階を実Runtimeから再取得します。", key: "refresh" };
}

type Props = {
  project: Project;
  showError: (message: string) => void;
  showNotice: (message: string) => void;
  onOpenDataset: () => void;
  onOpenTrain: () => void;
};

export default function CharacterDatasetFactory({ project, showError, showNotice, onOpenDataset, onOpenTrain }: Props) {
  const [workspace, setWorkspace] = useState<Workspace>({ concepts: [], assets: [], snapshots: [] });
  const [runs, setRuns] = useState<GenerationRun[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<number | null>(null);
  const [preflight, setPreflight] = useState<Preflight | null>(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState("");
  const [dragging, setDragging] = useState(false);
  const [prompt, setPrompt] = useState("same character, full body, perfectly still neutral standing pose, clean locked studio background, preserve face, hair, outfit, keytar, ears and tail");
  const [seed, setSeed] = useState(42);
  const [clipCount, setClipCount] = useState(3);
  const [coveragePreset, setCoveragePreset] = useState("balanced");
  const [conceptName, setConceptName] = useState(project.name);
  const [trigger, setTrigger] = useState(`char_${project.id}`);
  const [identityLock, setIdentityLock] = useState<Record<string, boolean>>({ face: true, hair: true, eyes: true, body: true, outfit: true, accessories: true, palette: true });
  const [decisions, setDecisions] = useState<Record<number, Decision>>({});
  const [selectedFrameIds, setSelectedFrameIds] = useState<number[]>([]);
  const [selectedFrameId, setSelectedFrameId] = useState<number | null>(null);
  const [expandedPreview, setExpandedPreview] = useState<{ path: string; label: string } | null>(null);
  const [assetVersions, setAssetVersions] = useState<AssetVersion[]>([]);
  const [selectedTrainingInputChoice, setSelectedTrainingInputChoice] = useState<"original" | "qwen" | "enhanced" | null>(null);
  const [versionResultsByAsset, setVersionResultsByAsset] = useState<Record<number, VersionResult>>({});
  const [captions, setCaptions] = useState<Record<number, string>>({});
  const [captionTemplate, setCaptionTemplate] = useState(`${trigger}, solo, full body, neutral pose, simple background`);
  const [menu, setMenu] = useState<MenuState>(null);
  const [cancelConfirmStage, setCancelConfirmStage] = useState<"h3" | "qwen" | null>(null);
  const [aestheticConfirmation, setAestheticConfirmation] = useState<AestheticConfirmation>(null);
  const [savingDecisionIds, setSavingDecisionIds] = useState<number[]>([]);
  const [savingIdentityLock, setSavingIdentityLock] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const menuTriggerRef = useRef<HTMLElement | null>(null);
  const cancelDialogRef = useRef<HTMLElement | null>(null);
  const aestheticDialogRef = useRef<HTMLElement | null>(null);
  const cancelTriggerRef = useRef<HTMLButtonElement | null>(null);
  const cancelWasOpenRef = useRef(false);
  const selectionAnchor = useRef<number | null>(null);
  const initializedConceptId = useRef<number | null>(null);

  useEffect(() => {
    if (cancelConfirmStage) {
      cancelWasOpenRef.current = true;
      return;
    }
    if (!cancelWasOpenRef.current) return;
    cancelWasOpenRef.current = false;
    window.setTimeout(() => cancelTriggerRef.current?.focus(), 0);
  }, [cancelConfirmStage]);

  const referenceAssets = useMemo(
    () => workspace.assets
      .filter((asset) => asset.origin_kind === "character_reference" && roles.some((role) => role.id === asset.metadata?.role))
      .sort((a, b) => {
        const explicit = (a.metadata?.reference_order ?? 99) - (b.metadata?.reference_order ?? 99);
        if (explicit) return explicit;
        return roles.findIndex((role) => role.id === a.metadata?.role) - roles.findIndex((role) => role.id === b.metadata?.role);
      }),
    [workspace.assets],
  );
  const selectedRun = selectedRunId == null ? null : runs.find((run) => run.id === selectedRunId) ?? null;
  const candidateProfile = String(selectedRun?.resolved?.candidate_generation_profile ?? "h3-refmod-turbo-spectrum-v1");
  const displayedClipCount = selectedRun ? Number(selectedRun.resolved?.clip_count ?? selectedRun.requested?.clip_count ?? 1) : clipCount;
  const displayedCoveragePreset = selectedRun ? String(selectedRun.requested?.coverage_preset ?? "legacy") : coveragePreset;
  const frozenClipPlans = selectedRun && Array.isArray(selectedRun.resolved?.clip_plans)
    ? selectedRun.resolved.clip_plans as Array<{ clip_id?: string; seed?: number; prompt?: string; coverage?: Record<string, string[]> }>
    : [];
  const displayReferenceAssets = useMemo(() => {
    if (!selectedRun) return referenceAssets;
    const requestedIds = Array.isArray(selectedRun.requested?.asset_ids) ? selectedRun.requested.asset_ids as number[] : [];
    const frozenRoles = selectedRun.requested?.reference_roles && typeof selectedRun.requested.reference_roles === "object"
      ? selectedRun.requested.reference_roles as Record<string, string>
      : {};
    return requestedIds.flatMap((assetId) => {
      const asset = workspace.assets.find((item) => item.id === assetId);
      if (!asset) return [];
      return [{ ...asset, metadata: { ...asset.metadata, role: frozenRoles[String(assetId)] ?? asset.metadata?.role } }];
    }).sort((a, b) => {
      const aIndex = roles.findIndex((role) => role.id === a.metadata?.role);
      const bIndex = roles.findIndex((role) => role.id === b.metadata?.role);
      return (aIndex < 0 ? roles.length : aIndex) - (bIndex < 0 ? roles.length : bIndex) || a.id - b.id;
    });
  }, [referenceAssets, selectedRun, workspace.assets]);
  const frames = frameManifest(selectedRun);
  const enhancementManifest = selectedRun?.output_manifest?.enhancement as {
    status?: string;
    completed_jobs?: number;
    total_jobs?: number;
    aesthetic_status?: string;
    contact_sheet?: { file_path?: string; content_sha256?: string };
  } | undefined;
  const selectedFrame = frames.find((frame) => frame.asset_id === selectedFrameId) ?? null;
  const qwenVersions = assetVersions.filter((version) => ["qwen_correction", "flashvsr_lr9_2x", "flashvsr_lr11_2x", "realesrgan_x2"].includes(version.version_kind) && version.status === "ready");
  const selectedVersionResult = selectedFrameId == null ? undefined : versionResultsByAsset[selectedFrameId];
  const selectedInputUserApproved = isUserApprovedInput(selectedVersionResult);
  const selectedQwenVersion = selectedInputUserApproved && ["qwen", "enhanced"].includes(selectedTrainingInputChoice ?? "") ? qwenVersions.find((version) => version.selected) ?? null : null;
  const previewQwenVersion = selectedQwenVersion ?? qwenVersions[0] ?? null;
  const trainingInputFrozen = selectedVersionResult?.training_input_mutable === false;
  const unresolved = frames.filter((frame) => normalizeDecision(decisions[frame.asset_id] ?? frame.review_status) === "pending").length;
  const approvedIds = frames.filter((frame) => normalizeDecision(decisions[frame.asset_id] ?? frame.review_status) === "approved").map((frame) => frame.asset_id);
  const selectedApprovedIds = selectedFrameIds.filter((assetId) => approvedIds.includes(assetId));
  const qwenTargetIds = selectedApprovedIds.filter(
    (assetId) => !isUserApprovedInput(versionResultsByAsset[assetId]) || versionResultsByAsset[assetId]?.training_input_choice !== "original",
  );
  const allApprovedOriginal = approvedIds.length > 0 && approvedIds.every((assetId) => isUserApprovedInput(versionResultsByAsset[assetId]) && versionResultsByAsset[assetId]?.training_input_choice === "original");
  const coverageSummary = useMemo(() => coverageDimensions.map(([dimension, label]) => {
    const counts = new Map<string, { total: number; approved: number }>();
    for (const frame of frames) {
      for (const value of frame.coverage?.[dimension] ?? []) {
        const current = counts.get(value) ?? { total: 0, approved: 0 };
        current.total += 1;
        if (normalizeDecision(decisions[frame.asset_id] ?? frame.review_status) === "approved") current.approved += 1;
        counts.set(value, current);
      }
    }
    return { dimension, label, values: [...counts.entries()].map(([value, count]) => ({ value, ...count })) };
  }).filter((item) => item.values.length > 0), [decisions, frames]);
  const reviewEditable = selectedRun?.current_stage === "enhancement";
  const identityReviewEvidence = selectedRun?.output_manifest?.identity_review as { aesthetic_review_state?: string; reviewed_by?: string } | undefined;
  const identityReviewUserApproved = identityReviewEvidence?.aesthetic_review_state === "USER_APPROVED" && identityReviewEvidence.reviewed_by === "user";
  const inputChoiceEditable = identityReviewUserApproved && selectedRun?.status !== "cancelled" && ["identity_review", "qwen_correction"].includes(selectedRun?.current_stage ?? "");
  const captionEditable = selectedRun?.current_stage === "balancing";
  const currentStageIndex = selectedRun ? stages.findIndex(([id]) => id === selectedRun.current_stage) : -1;
  const next = nextInstruction(selectedRun, displayReferenceAssets, unresolved);
  const missingQwenInputIds = approvedIds.filter((assetId) => {
    const result = versionResultsByAsset[assetId];
    return result == null || !isUserApprovedInput(result) || !result.training_input_choice;
  });
  const displayedNext = next.key === "commit-review" && approvedIds.length === 0
    ? { title: "採用Frameを1件以上選ぶ", detail: "全除外ではDatasetを作れません。Identityが保たれた候補を承認してください。", key: "review" }
    : ["balance", "snapshot"].includes(next.key) && missingQwenInputIds.length > 0
    ? { title: `${missingQwenInputIds.length}件のTraining Inputを確認`, detail: `Asset #${missingQwenInputIds.join(", #")} はOriginal / Correctedが未選択です。`, key: "qwen-input" }
    : next.key === "qwen" && allApprovedOriginal
      ? { title: "Original入力でQwenをスキップ", detail: "全approved FrameでOriginal採用が保存済みです。GPUを使わず次へ進めます。", key: "skip-qwen" }
      : next.key === "qwen" && qwenTargetIds.length === 0
        ? { title: "補正するFrameを選択", detail: "Qwenが必要なセルだけ選択してください。不要なFrameはOriginalを採用します。", key: "qwen-select" }
      : next;
  const waitProbe = selectedRun && ["waiting", "waiting_qwen", "waiting_enhancement"].includes(selectedRun.status)
    ? {
        count: Number(selectedRun.observed?.[selectedRun.status === "waiting_qwen" ? "qwen_admission_probe_count" : selectedRun.status === "waiting_enhancement" ? "enhancement_admission_probe_count" : "admission_probe_count"] ?? 0),
        ageSeconds: Math.max(0, Math.round(Date.now() / 1000 - Number(selectedRun.observed?.[selectedRun.status === "waiting_qwen" ? "last_qwen_admission_probe_at" : selectedRun.status === "waiting_enhancement" ? "last_enhancement_admission_probe_at" : "last_admission_probe_at"] ?? Date.now() / 1000))),
        position: selectedRun.queue_position ?? 1,
        aheadRunId: selectedRun.ahead_run_id ?? null,
      }
    : null;
  const concept = workspace.concepts[0] ?? null;

  useEffect(() => {
    if (!concept || initializedConceptId.current === concept.id) return;
    initializedConceptId.current = concept.id;
    setConceptName(concept.name);
    setTrigger(concept.trigger_token);
    setCaptionTemplate(`${concept.trigger_token}, solo, full body, neutral pose, simple background`);
  }, [concept?.id, concept?.name, concept?.trigger_token]);

  function storedIdentityLock(assets: Asset[]) {
    return assets.find((asset) => asset.origin_kind === "character_reference" && asset.metadata?.identity_lock)?.metadata?.identity_lock ?? null;
  }

  async function load(preserveRun = true) {
    setLoading(true);
    try {
      const [nextWorkspace, runResult] = await Promise.all([
        apiGet<Workspace>(`/basepipe/projects/${project.id}/workspace`),
        apiGet<{ runs: GenerationRun[] }>(`/basepipe/projects/${project.id}/character-generation/runs`),
      ]);
      setWorkspace(nextWorkspace);
      setRuns(runResult.runs ?? []);
      const currentId = preserveRun ? selectedRunId : null;
      const referenceIds = nextWorkspace.assets
        .filter((asset) => asset.origin_kind === "character_reference" && roles.some((role) => role.id === asset.metadata?.role))
        .map((asset) => asset.id)
        .sort((a, b) => a - b);
      const matchingRuns = runResult.runs.filter((run) => {
        const requested = Array.isArray(run.requested?.asset_ids)
          ? (run.requested.asset_ids as number[]).slice().sort((a, b) => a - b)
          : [];
        return referenceIds.length > 0 && requested.length === referenceIds.length && requested.every((id, index) => id === referenceIds[index]);
      });
      const statusPriority = (run: GenerationRun) => {
        if (["waiting", "waiting_qwen", "waiting_enhancement", "queued", "running", "running_qwen", "running_enhancement"].includes(run.status)) return 0;
        if (run.status === "prepared") return 1;
        if (run.status === "completed") return 2;
        return 3;
      };
      const matchingRun = matchingRuns.sort((a, b) => statusPriority(a) - statusPriority(b) || b.id - a.id)[0];
      const nextSelectedRun = currentId && runResult.runs.some((run) => run.id === currentId)
        ? runResult.runs.find((run) => run.id === currentId) ?? null
        : matchingRun ?? null;
      if (nextSelectedRun && Array.isArray(nextSelectedRun.observed?.checks)) {
        setPreflight({
          status: nextSelectedRun.status,
          message: nextSelectedRun.status === "ready" ? "H3生成を開始できる前提が揃っています" : "H3生成は開始していません。未充足の前提を確認してください。",
          checks: nextSelectedRun.observed.checks as PreflightCheck[],
        });
      } else if (nextSelectedRun && ["waiting", "waiting_qwen", "waiting_enhancement"].includes(nextSelectedRun.status)) {
        const memory = nextSelectedRun.observed?.gpu_memory as {
          admission_ok?: boolean;
          free_mb?: number;
          threshold_mb?: number;
          managed_processes?: unknown[];
          external_comfyui?: { reachable?: boolean; running?: number | null; pending?: number | null };
        } | undefined;
        const requestedIds = Array.isArray(nextSelectedRun.requested?.asset_ids) ? nextSelectedRun.requested.asset_ids as number[] : [];
        const graph = nextSelectedRun.resolved?.graph as Record<string, unknown> | undefined;
        const managedCount = memory?.managed_processes?.length ?? 0;
        const external = memory?.external_comfyui;
        const externalRunning = Number(external?.running ?? 0);
        const externalPending = Number(external?.pending ?? 0);
        const mapping = nextSelectedRun.observed?.gpu_mapping as { verified?: boolean; physical_index?: number; cuda_index?: number; physical_name?: string; cuda_name?: string; reason?: string } | undefined;
        const integrity = nextSelectedRun.observed?.reference_integrity as { ok?: boolean; verified_count?: number; expected_count?: number; detail?: string } | undefined;
        setPreflight({
          status: nextSelectedRun.status,
          message: "選択中Runの最新Observedから待機条件を表示しています。GPUプロセスは開始していません。",
          checks: [
            { label: "GPU", ok: nextSelectedRun.requested?.gpu_device_id === 1, detail: "物理GPU1（RTX 3090 Ti）固定" },
            { label: "GPU1 Mapping", ok: Boolean(mapping?.verified), detail: mapping?.verified ? `物理GPU${mapping.physical_index} ${mapping.physical_name ?? "unknown"} → CUDA ${mapping.cuda_index}（${mapping.cuda_name ?? "unknown"}）` : mapping?.reason ?? "H3用PythonとのCUDA列挙を照合中" },
            { label: "GPU1 Admission", ok: Boolean(memory?.admission_ok), detail: String(nextSelectedRun.observed?.wait_reason ?? `空き ${memory?.free_mb ?? "unknown"} / 必要 ${memory?.threshold_mb ?? "unknown"} MiB`) },
            { label: "External ComfyUI 8188", ok: Boolean(external?.reachable && externalRunning === 0 && externalPending === 0), detail: external?.reachable ? `${externalRunning}件実行 / ${externalPending}件待機 · 外部所有のため停止・アンロードしません` : "状態不明 · 外部所有のため操作しません" },
            { label: "Reference Asset", ok: requestedIds.length > 0 && requestedIds.length <= 4, detail: `${requestedIds.length}件 / 最大4件` },
            { label: "Reference Integrity", ok: Boolean(integrity?.ok), detail: integrity?.detail ?? "次の待機ProbeでpathとSHA-256を照合" },
            { label: "H3 Workflow", ok: Boolean(graph && Object.keys(graph).length), detail: graph ? "Runへ固定済み" : "未固定" },
            { label: "H3 Model", ok: Boolean(graph?.model), detail: graph?.model ? "Run graphへ固定済み" : "未固定" },
            { label: "Managed ComfyUI", ok: managedCount > 0, detail: managedCount > 0 ? `${managedCount}件の管理下process` : "GPU条件成立後に専用8189を起動" },
          ],
        });
      } else {
        setPreflight(null);
      }
      setSelectedRunId(nextSelectedRun?.id ?? null);
      if (!nextSelectedRun) {
        const storedLock = storedIdentityLock(nextWorkspace.assets ?? []);
        if (storedLock) setIdentityLock(storedLock);
      }
    } catch (error) {
      showError(error instanceof Error ? error.message : "Character Dataset Factoryの取得に失敗しました");
    } finally {
      setLoading(false);
    }
  }

  async function loadVersions(assetId: number) {
    const result = await apiGet<VersionResult>(`/basepipe/assets/${assetId}/versions`);
    setAssetVersions(result.versions ?? []);
    setSelectedTrainingInputChoice(isUserApprovedInput(result) ? result.training_input_choice ?? null : null);
    setVersionResultsByAsset((current) => ({ ...current, [assetId]: result }));
  }

  useEffect(() => { void load(false); }, [project.id]);
  useEffect(() => {
    const assetsById = new Map(workspace.assets.map((asset) => [asset.id, asset]));
    const captionEvidence = selectedRun?.output_manifest?.captioning as {
      captions?: Array<{ asset_id?: number; caption?: string }>;
    } | undefined;
    const frozenCaptions = new Map(
      (captionEvidence?.captions ?? [])
        .filter((item): item is { asset_id: number; caption: string } => typeof item.asset_id === "number" && typeof item.caption === "string")
        .map((item) => [item.asset_id, item.caption]),
    );
    const nextDecisions = Object.fromEntries(frames.map((frame) => [
      frame.asset_id,
      normalizeDecision(frame.review_status_frozen ? frame.review_status : assetsById.get(frame.asset_id)?.review_status ?? frame.review_status),
    ]));
    setDecisions(nextDecisions);
    setCaptions(Object.fromEntries(frames.map((frame) => [
      frame.asset_id,
      frozenCaptions.get(frame.asset_id) ?? assetsById.get(frame.asset_id)?.training_input ?? "",
    ])));
    setSelectedFrameIds([]);
    setSelectedFrameId(frames[0]?.asset_id ?? null);
  }, [selectedRun?.id, selectedRun?.current_stage, frames.length, workspace.assets]);
  useEffect(() => {
    if (selectedFrameId == null) { setAssetVersions([]); setSelectedTrainingInputChoice(null); return; }
    let active = true;
    apiGet<VersionResult>(`/basepipe/assets/${selectedFrameId}/versions`)
      .then((result) => { if (active) { setAssetVersions(result.versions ?? []); setSelectedTrainingInputChoice(isUserApprovedInput(result) ? result.training_input_choice ?? null : null); } })
      .catch(() => { if (active) { setAssetVersions([]); setSelectedTrainingInputChoice(null); } });
    return () => { active = false; };
  }, [selectedFrameId, selectedRun?.current_stage]);
  useEffect(() => {
    if (!frames.length) { setVersionResultsByAsset({}); return; }
    let active = true;
    void Promise.all(frames.map(async (frame) => [
      frame.asset_id,
      await apiGet<VersionResult>(`/basepipe/assets/${frame.asset_id}/versions`),
    ] as const)).then((entries) => {
      if (active) setVersionResultsByAsset(Object.fromEntries(entries));
    }).catch(() => {
      if (active) setVersionResultsByAsset({});
    });
    return () => { active = false; };
  }, [selectedRun?.id, selectedRun?.current_stage, frames.length]);
  useEffect(() => {
    if (!selectedRun) return;
    const requested = selectedRun.requested ?? {};
    if (typeof requested.prompt === "string") setPrompt(requested.prompt);
    if (typeof requested.seed === "number") setSeed(requested.seed);
    if (requested.identity_lock && typeof requested.identity_lock === "object") setIdentityLock(requested.identity_lock as Record<string, boolean>);
  }, [selectedRun?.id]);
  useEffect(() => {
    const close = () => setMenu(null);
    const closeFromKeyboard = (event: globalThis.KeyboardEvent) => {
      if (event.key === "Escape") {
        setMenu(null);
        menuTriggerRef.current?.focus();
      }
    };
    window.addEventListener("pointerdown", close);
    window.addEventListener("keydown", closeFromKeyboard);
    return () => {
      window.removeEventListener("pointerdown", close);
      window.removeEventListener("keydown", closeFromKeyboard);
    };
  }, []);
  useEffect(() => {
    if (!menu || !menuRef.current) return;
    const element = menuRef.current;
    const rect = element.getBoundingClientRect();
    const nextX = Math.max(8, Math.min(menu.x, window.innerWidth - rect.width - 8));
    const nextY = Math.max(8, Math.min(menu.y, window.innerHeight - rect.height - 8));
    if (nextX !== menu.x || nextY !== menu.y) {
      setMenu((current) => current ? { ...current, x: nextX, y: nextY } : null);
    }
    element.querySelector<HTMLButtonElement>("button:not(:disabled)")?.focus();
  }, [menu?.kind, menu?.assetId]);
  useEffect(() => {
    if (!selectedRun || !["waiting", "waiting_qwen", "queued", "running", "running_qwen"].includes(selectedRun.status)) return;
    const refreshVisibleRun = () => {
      if (document.visibilityState === "visible") void load();
    };
    const intervalMs = ["waiting", "waiting_qwen"].includes(selectedRun.status) ? 5000 : 2500;
    const timer = window.setInterval(refreshVisibleRun, intervalMs);
    const handleVisibility = () => {
      if (document.visibilityState === "visible") refreshVisibleRun();
    };
    document.addEventListener("visibilitychange", handleVisibility);
    return () => {
      window.clearInterval(timer);
      document.removeEventListener("visibilitychange", handleVisibility);
    };
  }, [selectedRun?.id, selectedRun?.status]);

  async function uploadReferences(files: File[]) {
    if (selectedRun) return showError("既存RunのReferenceは変更できません。Generation Runから「新規」を選択してください");
    const available = Math.max(0, 4 - referenceAssets.length);
    if (!available) return showError("Referenceは最大4枚です。既存の1枚を外してから追加してください");
    const selected = files.filter((file) => file.type.startsWith("image/")).slice(0, available);
    if (!selected.length) return showError("画像ファイルを選択してください");
    setBusy("upload");
    try {
      const form = new FormData();
      form.append("role", "detail");
      selected.forEach((file) => form.append("files", file));
      const result = await apiFormPost<{ assets: Asset[] }>(`/basepipe/projects/${project.id}/reference-assets`, form);
      const occupied = new Set(referenceAssets.map((asset) => asset.metadata?.role));
      const assignment = roles.map((item) => item.id).filter((role) => !occupied.has(role));
      for (let index = 0; index < result.assets.length; index += 1) {
        const asset = result.assets[index];
        const role = assignment[index] ?? "detail";
        await apiPost(`/basepipe/assets/${asset.id}/review`, {
          review_status: "approved",
          training_enabled: false,
          metadata: { role, identity_lock: identityLock },
        }, "PATCH");
      }
      await load();
      showNotice(`${result.assets.length}枚のReferenceを保存しました`);
    } catch (error) {
      showError(error instanceof Error ? error.message : "Referenceの追加に失敗しました");
    } finally {
      setBusy("");
    }
  }

  async function assignReference(assetId: number, role: Role | "unused") {
    if (selectedRun) return showError("既存RunのReference Roleは固定済みです。Generation Runから「新規」を選択してください");
    const asset = workspace.assets.find((item) => item.id === assetId);
    if (!asset) return;
    setBusy(`reference-${assetId}`);
    try {
      await apiPost(`/basepipe/assets/${asset.id}/review`, {
        review_status: "approved",
        training_enabled: false,
        metadata: { role, identity_lock: identityLock },
      }, "PATCH");
      await load();
      showNotice(role === "unused" ? "参照セットから外しました" : `${role} Referenceとして保存しました`);
    } catch (error) {
      showError(error instanceof Error ? error.message : "Reference roleの保存に失敗しました");
    } finally {
      setBusy("");
    }
  }

  async function updateIdentityLock(id: string, checked: boolean) {
    if (selectedRun) return;
    const next = { ...identityLock, [id]: checked };
    setIdentityLock(next);
    if (!referenceAssets.length) return;
    setSavingIdentityLock(true);
    try {
      await Promise.all(referenceAssets.map((asset) => apiPost(`/basepipe/assets/${asset.id}/review`, {
        review_status: asset.review_status ?? "approved",
        training_enabled: false,
        metadata: { role: asset.metadata?.role ?? "detail", identity_lock: next },
      }, "PATCH")));
      setWorkspace((current) => ({
        ...current,
        assets: current.assets.map((asset) => referenceAssets.some((reference) => reference.id === asset.id)
          ? { ...asset, metadata: { ...(asset.metadata ?? {}), identity_lock: next } }
          : asset),
      }));
    } catch (error) {
      await load();
      showError(error instanceof Error ? error.message : "Identity Lockの保存に失敗しました");
    } finally {
      setSavingIdentityLock(false);
    }
  }

  function selectGenerationRun(value: string) {
    if (!value) {
      setSelectedRunId(null);
      setPreflight(null);
      const storedLock = storedIdentityLock(workspace.assets);
      if (storedLock) setIdentityLock(storedLock);
      return;
    }
    setSelectedRunId(Number(value));
  }

  function openMenu(next: Exclude<MenuState, null>, trigger: HTMLElement) {
    menuTriggerRef.current = trigger;
    setMenu(next);
  }

  function openKeyboardMenu(kind: "reference" | "frame", assetId: number, trigger: HTMLElement) {
    const box = trigger.getBoundingClientRect();
    openMenu({ kind, assetId, x: box.left + Math.min(32, box.width / 2), y: box.top + Math.min(32, box.height / 2) }, trigger);
  }

  function navigateContextMenu(event: KeyboardEvent<HTMLDivElement>) {
    if (!menuRef.current) return;
    const items = Array.from(menuRef.current.querySelectorAll<HTMLButtonElement>('[role="menuitem"]:not(:disabled)'));
    if (!items.length) {
      if (event.key === "Escape") {
        event.preventDefault();
        setMenu(null);
        menuTriggerRef.current?.focus();
      }
      return;
    }
    const activeIndex = Math.max(0, items.indexOf(document.activeElement as HTMLButtonElement));
    let nextIndex = activeIndex;
    if (event.key === "ArrowDown") nextIndex = (activeIndex + 1) % items.length;
    else if (event.key === "ArrowUp") nextIndex = (activeIndex - 1 + items.length) % items.length;
    else if (event.key === "Home") nextIndex = 0;
    else if (event.key === "End") nextIndex = items.length - 1;
    else if (event.key === "Escape") {
      event.preventDefault();
      setMenu(null);
      menuTriggerRef.current?.focus();
      return;
    } else return;
    event.preventDefault();
    items[nextIndex]?.focus();
  }

  async function ensureConcept() {
    if (concept) return concept;
    if (!conceptName.trim() || !trigger.trim()) throw new Error("Concept名とTriggerを入力してください");
    const created = await apiPost<Concept>(`/basepipe/projects/${project.id}/concepts`, {
      name: conceptName.trim(), trigger_token: trigger.trim(), concept_type: "character",
    });
    await load();
    return created;
  }

  async function runPreflight() {
    if (selectedRun) return showError("既存Runは不変です。新しい条件で試す場合はGeneration Runから「新規」を選択してください");
    if (!referenceAssets.length) return showError("先にReference画像を追加してください");
    setBusy("preflight");
    try {
      const result = await apiPost<Preflight>(`/basepipe/projects/${project.id}/character-generation/preflight`, {
        asset_ids: referenceAssets.map((asset) => asset.id), mode: "variation", gpu_device_id: 1,
        reference_roles: Object.fromEntries(referenceAssets.map((asset) => [String(asset.id), asset.metadata?.role ?? "detail"])),
      });
      setPreflight(result);
      showNotice(`H3 Preflight: ${result.status}。GPUジョブは開始していません`);
    } catch (error) {
      showError(error instanceof Error ? error.message : "Preflightに失敗しました");
    } finally {
      setBusy("");
    }
  }

  function generationPayload(execute: boolean) {
    return {
      asset_ids: referenceAssets.map((asset) => asset.id),
      mode: "variation",
      prompt,
      seed,
      width: 512,
      height: 768,
      length: coveragePreset === "turntable" ? 124 : 56,
      clip_count: clipCount,
      coverage_preset: coveragePreset,
      ref_image_size: "match",
      gpu_device_id: 1,
      execute,
      queue_if_busy: execute,
      identity_lock: identityLock,
      reference_roles: Object.fromEntries(referenceAssets.map((asset) => [String(asset.id), asset.metadata?.role ?? "detail"])),
    };
  }

  async function prepareRun() {
    if (selectedRun) return showError("既存Runは変更できません。Generation Runから「新規」を選択してください");
    if (!referenceAssets.length) return showError("先にReference画像を追加してください");
    setBusy("prepare");
    try {
      await ensureConcept();
      const result = await apiPost<{ run_id: number; status: string }>(`/basepipe/projects/${project.id}/character-generation/start`, generationPayload(false));
      await load(false);
      setSelectedRunId(result.run_id);
      showNotice(`Dataset Generation Run #${result.run_id}を準備しました。GPUは未実行です`);
    } catch (error) {
      showError(error instanceof Error ? error.message : "生成計画の保存に失敗しました");
    } finally {
      setBusy("");
    }
  }

  async function executeH3() {
    setBusy("h3");
    try {
      const preparedRun = selectedRun?.status === "prepared" && selectedRun.current_stage === "planning" ? selectedRun : null;
      if (selectedRun && !preparedRun) throw new Error("実行待機へ移せるのはprepared状態のRunだけです");
      if (!preparedRun && !preflight) throw new Error("先にH3 Preflightを実行してください");
      const result = preparedRun
        ? await apiPost<{ run_id: number; status: string; message: string }>(`/basepipe/projects/${project.id}/character-generation/runs/${preparedRun.id}/execute`, {})
        : await apiPost<{ run_id: number; status: string; message: string }>(`/basepipe/projects/${project.id}/character-generation/start`, generationPayload(true));
      await load(!preparedRun);
      setSelectedRunId(result.run_id);
      showNotice(result.status === "waiting" ? `H3 Run #${result.run_id}をGPU1待機キューへ追加しました` : `H3 Run #${result.run_id}を物理GPU1へ投入しました`);
    } catch (error) {
      showError(error instanceof Error ? error.message : "H3の開始に失敗しました");
    } finally {
      setBusy("");
    }
  }

  async function cancelWaitingRun() {
    if (!selectedRun || !["waiting", "waiting_qwen"].includes(selectedRun.status)) return;
    setBusy("cancel-waiting");
    try {
      await apiPost(`/basepipe/projects/${project.id}/character-generation/runs/${selectedRun.id}/cancel-waiting`, {});
      await load();
      setCancelConfirmStage(null);
      showNotice(`Run #${selectedRun.id}のGPU待機を取り消しました。GPUプロセスは開始されていません`);
    } catch (error) {
      showError(error instanceof Error ? error.message : "待機Runの取消に失敗しました");
    } finally {
      setBusy("");
    }
  }

  function openCancelConfirmation(stage: "h3" | "qwen", triggerElement: HTMLButtonElement) {
    cancelTriggerRef.current = triggerElement;
    setCancelConfirmStage(stage);
  }

  function closeCancelConfirmation() {
    if (busy === "cancel-waiting") return;
    setCancelConfirmStage(null);
  }

  function handleCancelDialogKey(event: KeyboardEvent<HTMLElement>) {
    if (event.key === "Escape") {
      event.preventDefault();
      event.stopPropagation();
      closeCancelConfirmation();
      return;
    }
    if (event.key !== "Tab" || !cancelDialogRef.current) return;
    const focusable = Array.from(cancelDialogRef.current.querySelectorAll<HTMLElement>(
      'button:not(:disabled), [tabindex]:not([tabindex="-1"])',
    )).filter((element) => element.getClientRects().length > 0);
    if (!focusable.length) return;
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (event.shiftKey && (document.activeElement === first || !cancelDialogRef.current.contains(document.activeElement))) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  }

  function closeAestheticConfirmation() {
    if (["review", "version-select"].includes(busy)) return;
    setAestheticConfirmation(null);
  }

  function handleAestheticDialogKey(event: KeyboardEvent<HTMLElement>) {
    if (event.key === "Escape") {
      event.preventDefault();
      event.stopPropagation();
      closeAestheticConfirmation();
      return;
    }
    if (event.key !== "Tab" || !aestheticDialogRef.current) return;
    const focusable = Array.from(aestheticDialogRef.current.querySelectorAll<HTMLElement>('button:not(:disabled)'));
    if (!focusable.length) return;
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  }

  async function extractFrames() {
    if (!selectedRun) return;
    setBusy("extract");
    try {
      await apiPost(`/basepipe/projects/${project.id}/character-generation/runs/${selectedRun.id}/extract-frames`, {
        fps: 2, max_frames: 24, blur_threshold: 18, duplicate_distance: 6,
      });
      await load();
      showNotice("Frame抽出、Blur検出、重複除外を保存しました");
    } catch (error) {
      showError(error instanceof Error ? error.message : "Frame抽出に失敗しました");
    } finally {
      setBusy("");
    }
  }

  async function enhanceTurntable() {
    if (!selectedRun) return;
    setBusy("enhance");
    try {
      const result = await apiPost<{ status: string; message: string }>(`/basepipe/projects/${project.id}/character-generation/runs/${selectedRun.id}/enhance`, {
        gpu_device_id: 1, slot_count: 8, total_frames: 124, local_ranges: [9, 11], include_realesrgan: true, queue_if_busy: true,
      });
      await load();
      showNotice(result.message);
    } catch (error) {
      showError(error instanceof Error ? error.message : "Enhancement待機キューへの追加に失敗しました");
    } finally {
      setBusy("");
    }
  }

  async function setFrameDecision(assetId: number, decision: Decision, applySelection = true) {
    if (!reviewEditable) return;
    const targets = applySelection && selectedFrameIds.includes(assetId) ? selectedFrameIds : [assetId];
    setDecisions((current) => ({ ...current, ...Object.fromEntries(targets.map((id) => [id, decision])) }));
    setSavingDecisionIds((current) => [...new Set([...current, ...targets])]);
    try {
      await apiPost(`/basepipe/projects/${project.id}/assets/bulk-review`, {
        asset_ids: targets,
        review_status: decision,
      });
    } catch (error) {
      await load();
      showError(error instanceof Error ? error.message : "Frame判定の保存に失敗しました");
    } finally {
      setSavingDecisionIds((current) => current.filter((id) => !targets.includes(id)));
    }
  }

  async function saveFrameCaption(assetId: number) {
    if (!captionEditable) return;
    const caption = captions[assetId]?.trim() ?? "";
    const reviewStatus = normalizeDecision(decisions[assetId] ?? frames.find((frame) => frame.asset_id === assetId)?.review_status);
    setSavingDecisionIds((current) => [...new Set([...current, assetId])]);
    try {
      await apiPost(`/basepipe/assets/${assetId}/review`, {
        review_status: reviewStatus,
        caption,
      }, "PATCH");
    } catch (error) {
      await load();
      showError(error instanceof Error ? error.message : "Captionの保存に失敗しました");
    } finally {
      setSavingDecisionIds((current) => current.filter((id) => id !== assetId));
    }
  }

  function selectFrame(frame: Frame, event: MouseEvent) {
    const index = frames.findIndex((item) => item.asset_id === frame.asset_id);
    if (event.shiftKey && selectionAnchor.current != null) {
      const anchorIndex = frames.findIndex((item) => item.asset_id === selectionAnchor.current);
      const [start, end] = [anchorIndex, index].sort((a, b) => a - b);
      setSelectedFrameIds(frames.slice(start, end + 1).map((item) => item.asset_id));
    } else if (event.ctrlKey || event.metaKey) {
      setSelectedFrameIds((current) => current.includes(frame.asset_id) ? current.filter((id) => id !== frame.asset_id) : [...current, frame.asset_id]);
      selectionAnchor.current = frame.asset_id;
    } else {
      setSelectedFrameIds([frame.asset_id]);
      selectionAnchor.current = frame.asset_id;
    }
    setSelectedFrameId(frame.asset_id);
  }

  function selectCoverage(dimension: string, value: string) {
    const matching = frames.filter((frame) => frame.coverage?.[dimension]?.includes(value)).map((frame) => frame.asset_id);
    setSelectedFrameIds(matching);
    setSelectedFrameId(matching[0] ?? null);
    selectionAnchor.current = matching[0] ?? null;
  }

  function prepareFrameContextSelection(assetId: number) {
    if (!selectedFrameIds.includes(assetId)) {
      setSelectedFrameIds([assetId]);
      selectionAnchor.current = assetId;
    } else if (selectionAnchor.current == null) {
      selectionAnchor.current = assetId;
    }
    setSelectedFrameId(assetId);
  }

  function frameGridKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    if (!frames.length) return;
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "a") {
      event.preventDefault();
      setSelectedFrameIds(frames.map((frame) => frame.asset_id));
      selectionAnchor.current ??= frames[0].asset_id;
      return;
    }
    if (event.key.toLowerCase() === "a" && reviewEditable && selectedFrameIds[0] != null) { event.preventDefault(); void setFrameDecision(selectedFrameIds[0], "approved"); }
    if ((event.key.toLowerCase() === "r" || event.key === "Delete") && reviewEditable && selectedFrameIds[0] != null) { event.preventDefault(); void setFrameDecision(selectedFrameIds[0], "rejected"); }
    if (["ArrowRight", "ArrowLeft", "ArrowUp", "ArrowDown", "Home", "End"].includes(event.key)) {
      event.preventDefault();
      const grid = event.currentTarget;
      const current = Math.max(0, frames.findIndex((frame) => frame.asset_id === selectedFrameId));
      const columns = Math.max(1, getComputedStyle(grid).gridTemplateColumns.split(" ").filter(Boolean).length);
      const delta = event.key === "ArrowRight" ? 1
        : event.key === "ArrowLeft" ? -1
        : event.key === "ArrowDown" ? columns
        : event.key === "ArrowUp" ? -columns
        : 0;
      const nextIndex = event.key === "Home" ? 0
        : event.key === "End" ? frames.length - 1
        : Math.max(0, Math.min(frames.length - 1, current + delta));
      const nextId = frames[nextIndex].asset_id;
      if (event.shiftKey) {
        const anchorId = selectionAnchor.current ?? frames[current].asset_id;
        selectionAnchor.current = anchorId;
        const anchorIndex = Math.max(0, frames.findIndex((frame) => frame.asset_id === anchorId));
        const [start, end] = [anchorIndex, nextIndex].sort((a, b) => a - b);
        setSelectedFrameIds(frames.slice(start, end + 1).map((frame) => frame.asset_id));
      } else {
        setSelectedFrameIds([nextId]);
        selectionAnchor.current = nextId;
      }
      setSelectedFrameId(nextId);
      window.requestAnimationFrame(() => {
        grid.querySelector<HTMLElement>(`[data-frame-id="${nextId}"]`)?.focus();
      });
    }
  }

  function requestIdentityReviewConfirmation() {
    if (!selectedRun || unresolved) return showError("全Frameを承認または除外してください");
    if (!approvedIds.length) return showError("Datasetへ採用するFrameを1件以上承認してください");
    setAestheticConfirmation({ kind: "identity", approvedCount: approvedIds.length, rejectedCount: frames.length - approvedIds.length });
  }

  async function reopenIdentityReview() {
    if (!selectedRun) return;
    setBusy("reopen-review");
    try {
      const result = await apiPost<{ preserved_qwen_version_count: number }>(`/basepipe/projects/${project.id}/character-generation/runs/${selectedRun.id}/identity-review/reopen`, {});
      await load();
      showNotice(`Frame Reviewを開き直しました。Qwen候補 ${result.preserved_qwen_version_count}件は未評価のまま保持しています`);
    } catch (error) {
      showError(error instanceof Error ? error.message : "Identity Reviewを開き直せませんでした");
    } finally {
      setBusy("");
    }
  }

  async function commitIdentityReview() {
    if (!selectedRun || aestheticConfirmation?.kind !== "identity") return;
    setBusy("review");
    try {
      await apiPost(`/basepipe/projects/${project.id}/character-generation/runs/${selectedRun.id}/identity-review`, {
        decisions: frames.map((frame) => ({ asset_id: frame.asset_id, review_status: normalizeDecision(decisions[frame.asset_id]) })),
        note: "Character Dataset Factory explicit frame review",
        human_confirmed: true,
      });
      await load();
      setAestheticConfirmation(null);
      showNotice(`Identity Reviewを保存しました（採用 ${approvedIds.length}件）`);
    } catch (error) {
      showError(error instanceof Error ? error.message : "Identity Reviewの保存に失敗しました");
    } finally { setBusy(""); }
  }

  async function qwen(execute: boolean) {
    if (!selectedRun || !qwenTargetIds.length) return showError("Qwen補正対象を選択してください");
    setBusy(execute ? "qwen-run" : "qwen-plan");
    try {
      const result = await apiPost<{ status: string; message: string }>(`/basepipe/projects/${project.id}/character-generation/runs/${selectedRun.id}/qwen-correction`, {
        asset_ids: qwenTargetIds, prompt: "preserve identity, outfit and proportions; correct artifacts and image quality", seed, gpu_device_id: 1, execute, queue_if_busy: execute,
      });
      await load();
      showNotice(execute ? result.message : "Qwen補正計画を保存しました。GPUは未実行です");
    } catch (error) {
      showError(error instanceof Error ? error.message : "Qwen補正に失敗しました");
    } finally { setBusy(""); }
  }

  async function balance() {
    if (!selectedRun) return;
    setBusy("balance");
    try {
      await apiPost(`/basepipe/projects/${project.id}/character-generation/runs/${selectedRun.id}/balance`, {});
      await load();
      showNotice("採用FrameのQwen Versionを確認し、Dataset構成を固定しました");
    } catch (error) { showError(error instanceof Error ? error.message : "Balancingに失敗しました"); }
    finally { setBusy(""); }
  }

  function selectQwenVersion(version: AssetVersion) {
    if (!selectedFrame) return;
    if (!inputChoiceEditable || trainingInputFrozen) {
      return showError(selectedVersionResult?.training_input_lock_reason || "このRunのTraining Inputは固定済みです");
    }
    setAestheticConfirmation({ kind: "qwen", assetId: selectedFrame.asset_id, version });
  }

  async function confirmQwenVersion(assetId: number, version: AssetVersion) {
    setBusy("version-select");
    try {
      const choice = version.version_kind === "qwen_correction" ? "qwen" : "enhanced";
      await apiPost(`/basepipe/assets/${assetId}/training-input`, {
        choice,
        version_id: version.id,
        human_confirmed: true,
        note: `User confirmed ${version.version_kind} candidate in Character Dataset Factory`,
      }, "PUT");
      await loadVersions(assetId);
      setAestheticConfirmation(null);
      showNotice(`${version.version_kind} Version #${version.id}をTraining Inputに採用しました`);
    } catch (error) {
      showError(error instanceof Error ? error.message : "Qwen Versionの選択に失敗しました");
    } finally { setBusy(""); }
  }

  function selectOriginalTrainingInput() {
    if (!selectedFrame) return;
    selectOriginalTrainingInputs([selectedFrame.asset_id]);
  }

  function selectOriginalTrainingInputs(assetIds: number[]) {
    const targets = assetIds.filter((assetId) => approvedIds.includes(assetId));
    if (!targets.length) return showError("approved Frameを選択してください");
    setAestheticConfirmation({ kind: "original", assetIds: targets });
  }

  async function confirmOriginalTrainingInputs(assetIds: number[]) {
    setBusy("version-select");
    try {
      const entries = await Promise.all(assetIds.map(async (assetId) => {
        await apiPost(`/basepipe/assets/${assetId}/training-input`, {
          choice: "original",
          human_confirmed: true,
          note: "User confirmed Original candidate in Character Dataset Factory",
        }, "PUT");
        return [assetId, await apiGet<VersionResult>(`/basepipe/assets/${assetId}/versions`)] as const;
      }));
      setVersionResultsByAsset((current) => ({ ...current, ...Object.fromEntries(entries) }));
      const selectedResult = selectedFrameId == null ? null : entries.find(([assetId]) => assetId === selectedFrameId)?.[1];
      if (selectedResult) {
        setAssetVersions(selectedResult.versions ?? []);
        setSelectedTrainingInputChoice(isUserApprovedInput(selectedResult) ? selectedResult.training_input_choice ?? null : null);
      }
      setAestheticConfirmation(null);
      showNotice(`${assetIds.length}件のOriginalをTraining Inputに採用しました`);
    } catch (error) {
      showError(error instanceof Error ? error.message : "Originalの選択に失敗しました");
    } finally { setBusy(""); }
  }

  async function skipQwenWithOriginals() {
    if (!selectedRun || !allApprovedOriginal) return;
    setBusy("skip-qwen");
    try {
      await apiPost(`/basepipe/projects/${project.id}/character-generation/runs/${selectedRun.id}/qwen-correction/skip`, {});
      await load();
      showNotice("全approved FrameのOriginal採用を固定し、Qwen GPU補正をスキップしました");
    } catch (error) {
      showError(error instanceof Error ? error.message : "Qwenスキップに失敗しました");
    } finally { setBusy(""); }
  }

  async function applyCaptionTemplate(ids = approvedIds) {
    if (!captionEditable || !ids.length) return;
    const caption = captionTemplate.trim();
    setCaptions((current) => ({ ...current, ...Object.fromEntries(ids.map((id) => [id, caption])) }));
    setSavingDecisionIds((current) => [...new Set([...current, ...ids])]);
    try {
      await Promise.all(ids.map((id) => apiPost(`/basepipe/assets/${id}/review`, {
        review_status: normalizeDecision(decisions[id] ?? frames.find((frame) => frame.asset_id === id)?.review_status),
        caption,
      }, "PATCH")));
    } catch (error) {
      await load();
      showError(error instanceof Error ? error.message : "Caption templateの保存に失敗しました");
    } finally {
      setSavingDecisionIds((current) => current.filter((id) => !ids.includes(id)));
    }
  }

  async function completeCaptioning() {
    if (!selectedRun) return;
    const missing = approvedIds.filter((id) => !captions[id]?.trim());
    if (missing.length) return showError(`Caption未入力のFrameが${missing.length}件あります`);
    setBusy("caption");
    try {
      await apiPost(`/basepipe/projects/${project.id}/character-generation/runs/${selectedRun.id}/captioning`, {
        captions: approvedIds.map((id) => ({ asset_id: id, caption: captions[id].trim() })),
        note: "Character Dataset Factory reviewed captions",
      });
      await load();
      showNotice("Training Captionを全承認Frameへ保存しました");
    } catch (error) { showError(error instanceof Error ? error.message : "Captioningに失敗しました"); }
    finally { setBusy(""); }
  }

  async function sealSnapshot() {
    if (!selectedRun || !approvedIds.length) return;
    setBusy("snapshot");
    try {
      const currentConcept = await ensureConcept();
      const result = await apiPost<Snapshot>(`/basepipe/projects/${project.id}/snapshots`, {
        name: `${project.name}-character-dataset-v${workspace.snapshots.length + 1}`,
        concept_id: currentConcept.id,
        asset_ids: approvedIds,
      });
      await load();
      const profile = result.preview_profile
        ? ` · Preview Profile #${result.preview_profile.id}${result.preview_profile.created ? "を自動作成" : "を再利用"}`
        : "";
      const frozenProfile = result.preview_profile_snapshot
        ? ` · Compare条件 #${result.preview_profile_snapshot.id}${result.preview_profile_snapshot.created ? "を固定" : "を再利用"}`
        : "";
      showNotice(`Dataset Snapshot #${result.id}を${result.item_count}件で封印し、Anima Draftへ渡しました${profile}${frozenProfile}`);
    } catch (error) { showError(error instanceof Error ? error.message : "Snapshot作成に失敗しました"); }
    finally { setBusy(""); }
  }

  async function safeNextAction() {
    if (displayedNext.key === "qwen-input") return;
    if (next.key === "new-run") return selectGenerationRun("");
    if (next.key === "plan") return prepareRun();
    if (next.key === "extract") return extractFrames();
    if (next.key === "enhance") return enhanceTurntable();
    if (next.key === "commit-review") return commitIdentityReview();
    if (displayedNext.key === "skip-qwen") return skipQwenWithOriginals();
    if (next.key === "h3" && selectedRun?.status === "prepared") return executeH3();
    if (next.key === "qwen") return qwen(true);
    if (next.key === "balance") return balance();
    if (next.key === "caption") return completeCaptioning();
    if (next.key === "snapshot") return sealSnapshot();
    if (next.key === "done") return onOpenTrain();
    return load();
  }

  return (
    <div className="character-factory" data-workspace="character-dataset-factory">
      <header className="studio-page-header">
        <div><p className="studio-eyebrow">01 · CHARACTER DATASET FACTORY</p><h1>数枚のReferenceから学習Datasetを作る</h1><p>固定ポーズの360°動画から角度を自動選抜し、高密度化したDatasetを最後にまとめて確認します。</p></div>
        <div className="studio-header-actions">
          <button className="studio-secondary-button" onClick={() => void load()} disabled={loading}><RefreshCw size={15} className={loading ? "spin" : ""} />更新</button>
          <button className="studio-primary-button" data-action="character-dataset-next" onClick={() => void safeNextAction()} disabled={busy !== "" || ["references", "review", "blocked", "waiting", "running", "qwen-input", "qwen-select"].includes(displayedNext.key) || (displayedNext.key === "h3" && selectedRun?.status !== "prepared")}><Bot size={16} />{displayedNext.key === "done" ? "Trainへ進む" : "安全な次工程"}</button>
        </div>
      </header>

      <section className="factory-next-strip">
        <span className={`factory-next-icon ${waitProbe ? "is-waiting" : ""}`}>{waitProbe ? <span className="factory-wait-orb" aria-hidden="true" /> : <Sparkles size={18} />}</span>
        <span><small>NEXT REQUIRED ACTION</small><strong>{displayedNext.title}</strong><p>{displayedNext.detail}</p>{waitProbe && <em className="factory-wait-evidence">{waitProbe.position > 1 ? `待機順 #${waitProbe.position} · 先行Run #${waitProbe.aheadRunId ?? "unknown"}を待機中` : `待機順 #1 · 自動再評価 #${waitProbe.count.toLocaleString()} · 最終確認 ${waitProbe.ageSeconds}秒前`}</em>}</span>
        <ChevronRight size={18} />
      </section>

      <section className="factory-flow" aria-label="Character dataset pipeline">
        {stages.map(([id, label], index) => {
          const done = index < currentStageIndex || selectedRun?.current_stage === "snapshot";
          const active = index === currentStageIndex;
          return <div key={id} className={done ? "is-done" : active ? "is-active" : ""}><span>{done ? <Check size={12} /> : index + 1}</span><strong>{label}</strong></div>;
        })}
      </section>

      <div className="factory-layout">
        <aside className="factory-reference-pane">
          <div className="factory-pane-title"><span><Images size={15} /><strong>Reference set</strong></span><small>{displayReferenceAssets.length} / 4</small></div>
          <input ref={fileInput} type="file" accept="image/*" multiple hidden onChange={(event) => void uploadReferences(Array.from(event.target.files ?? []))} />
          <button
            className={`factory-drop-zone ${dragging ? "is-dragging" : ""}`}
            onClick={() => fileInput.current?.click()}
            onDragEnter={(event) => { event.preventDefault(); setDragging(true); }}
            onDragOver={(event) => event.preventDefault()}
            onDragLeave={() => setDragging(false)}
            onDrop={(event) => { event.preventDefault(); setDragging(false); void uploadReferences(Array.from(event.dataTransfer.files)); }}
            disabled={Boolean(selectedRun) || busy === "upload" || referenceAssets.length >= 4}
            data-action="upload-character-references"
          >
            {busy === "upload" ? <Loader2 size={21} className="spin" /> : <Upload size={21} />}
            <strong>{referenceAssets.length ? "画像を追加" : "キャラクター画像をドロップ"}</strong>
            <small>PNG / JPG / WebP · 最大4枚</small>
          </button>

          <div className="factory-reference-list">
            {displayReferenceAssets.map((asset) => {
              const role = (asset.metadata?.role ?? "detail") as Role;
              const roleLabel = roles.find((item) => item.id === role)?.label ?? role;
              const sourceName = asset.file_path.split(/[\\/]/).pop() ?? asset.asset_key;
              const displayName = sourceName.replace(/^[a-f0-9]{12}-/i, "");
              return <article
                key={asset.id}
                className="factory-reference-card"
                tabIndex={0}
                aria-label={`${roleLabel} reference, Asset ${asset.id}, ${sourceName}`}
                onContextMenu={(event) => { event.preventDefault(); event.stopPropagation(); openMenu({ kind: "reference", assetId: asset.id, x: event.clientX, y: event.clientY }, event.currentTarget); }}
                onKeyDown={(event) => {
                  if (event.key === "ContextMenu" || (event.shiftKey && event.key === "F10")) {
                    event.preventDefault();
                    openKeyboardMenu("reference", asset.id, event.currentTarget);
                  }
                }}
              >
                <img src={`${API_BASE}/collector/thumbnail?path=${encodeURIComponent(asset.file_path)}&size=300`} alt={`${roleLabel} reference`} />
                <div title={`${sourceName} · SHA-256 ${asset.content_sha256 ?? "unknown"}`}><span>{roleLabel}</span><strong>Asset #{asset.id}</strong><small>{displayName}</small></div>
                <button aria-label={`${roleLabel} reference options`} onClick={(event) => { event.stopPropagation(); const box = event.currentTarget.getBoundingClientRect(); openMenu({ kind: "reference", assetId: asset.id, x: box.right, y: box.bottom }, event.currentTarget); }}><MoreHorizontal size={15} /></button>
              </article>;
            })}
          </div>

          <div className="factory-locks">
            <div className="factory-pane-title"><span><LockKeyhole size={14} /><strong>Identity Lock</strong></span><small>{selectedRun ? `Run #${selectedRun.id}へ固定済み` : savingIdentityLock ? "保存中…" : `${Object.values(identityLock).filter(Boolean).length} locked · 自動保存`}</small></div>
            {locks.map(([id, label]) => <label key={id}><span>{label}</span><input type="checkbox" disabled={Boolean(selectedRun) || savingIdentityLock} checked={identityLock[id] ?? false} onChange={(event) => void updateIdentityLock(id, event.target.checked)} /></label>)}
          </div>
        </aside>

        <main className="factory-stage-pane">
          <div className="factory-run-toolbar">
            <div><p className="studio-eyebrow">DATASET GENERATION RUN</p><h2>{selectedRun ? `Run #${selectedRun.id}` : "新しい生成計画"}</h2></div>
            <select aria-label="Generation Run" value={selectedRunId ?? ""} onChange={(event) => selectGenerationRun(event.target.value)}>
              <option value="">新規</option>
              {runs.slice(0, 20).map((run) => <option key={run.id} value={run.id}>#{run.id} · {run.status} · {run.current_stage}</option>)}
            </select>
          </div>

          <section className="factory-quality-route" aria-label="Selective dataset quality strategy">
            <article><small>01 · CANDIDATES</small><strong>Anchor H3</strong><span>{candidateProfile}</span></article>
            <ChevronRight size={14} />
            <article><small>02 · TECH GATE</small><strong>Angle Curation</strong><span>Yaw · Blur · Dedupe</span></article>
            <ChevronRight size={14} />
            <article className="is-selective"><small>03 · SELECTIVE HQ</small><strong>選択Frameだけ</strong><span>Qwen Identity Repair · {qwenTargetIds.length}件</span></article>
            <ChevronRight size={14} />
            <article><small>04 · USER REVIEW</small><strong>Contact Sheet</strong><span>比較確認後にSnapshot</span></article>
          </section>

          {selectedRun && ["duplicate_filter", "technical_selection", "enhancement"].includes(selectedRun.current_stage) && <section className="factory-plan-card is-wide" aria-label="Turntable enhancement evidence">
            <div className="factory-card-heading"><span><Sparkles size={16} /></span><div><small>8-YAW DATASET ENHANCEMENT</small><h3>Original / FlashVSR / RealESRGAN 比較</h3></div></div>
            <p className="studio-modal-copy">124フレームから8方向の循環9フレーム窓を自動構成します。技術処理は自動ですが、どれを採用するかは確定しません。</p>
            <div className="factory-plan-actions">
              <button className="studio-primary-button" data-action="queue-turntable-enhancement" onClick={() => void enhanceTurntable()} disabled={busy !== "" || selectedRun.current_stage !== "duplicate_filter" || ["waiting_enhancement", "running_enhancement"].includes(selectedRun.status)}>
                {selectedRun.status === "waiting_enhancement" ? <CircleDot size={15} /> : selectedRun.status === "running_enhancement" ? <Loader2 size={15} className="spin" /> : <Sparkles size={15} />}
                {selectedRun.status === "waiting_enhancement" ? `GPU1待機中 · #${selectedRun.queue_position ?? 1}` : selectedRun.status === "running_enhancement" ? `${enhancementManifest?.completed_jobs ?? 0} / ${enhancementManifest?.total_jobs ?? 24} 比較生成中` : "8方向の比較を生成"}
              </button>
            </div>
            {enhancementManifest?.contact_sheet?.file_path && <figure className="factory-enhancement-sheet">
              <img src={`${API_BASE}/collector/thumbnail?path=${encodeURIComponent(enhancementManifest.contact_sheet.file_path)}&size=1400`} alt={`Run ${selectedRun.id} Original FlashVSR RealESRGAN comparison contact sheet`} />
              <figcaption><strong>{enhancementManifest.aesthetic_status ?? "AESTHETIC_UNREVIEWED"}</strong><span>Run #{selectedRun.id} · SHA-256 {enhancementManifest.contact_sheet.content_sha256?.slice(0, 16) ?? "unknown"}…</span></figcaption>
            </figure>}
          </section>}

          {!frames.length ? <div className="factory-plan-grid">
            <section className="factory-plan-card">
              <div className="factory-card-heading"><span><ScanFace size={16} /></span><div><small>IDENTITY INPUT</small><h3>Reference interpretation</h3></div></div>
              <label><span>Concept name</span><input value={concept?.name ?? conceptName} disabled={Boolean(concept)} onChange={(event) => setConceptName(event.target.value)} /></label>
              <label><span>Trigger token</span><input value={concept?.trigger_token ?? trigger} disabled={Boolean(concept)} onChange={(event) => { setTrigger(event.target.value); setCaptionTemplate(`${event.target.value}, solo, full body, neutral pose, simple background`); }} /></label>
              <div className="factory-role-summary">{roles.map((item) => {
                const assigned = displayReferenceAssets.find((asset) => asset.metadata?.role === item.id);
                return <div key={item.id} className={assigned ? "is-filled" : ""}><span>{item.label}</span><small>{assigned ? `Asset #${assigned.id} · ${assigned.content_sha256?.slice(0, 8) ?? "hash unknown"}` : item.hint}</small></div>;
              })}</div>
            </section>

            <section className="factory-plan-card is-wide">
              <div className="factory-card-heading"><span><WandSparkles size={16} /></span><div><small>H3 GENERATION PLAN</small><h3>候補Datasetを増幅</h3></div></div>
              <label><span>Generation intent {selectedRun ? `· Run #${selectedRun.id}へ固定済み` : "· 新規計画"}</span><textarea value={prompt} disabled={Boolean(selectedRun)} onChange={(event) => setPrompt(event.target.value)} rows={4} /></label>
              <div className="factory-inline-fields">
                <label><span>Seed</span><input type="number" value={selectedRun ? Number(selectedRun.requested?.seed ?? seed) : seed} disabled={Boolean(selectedRun)} onChange={(event) => setSeed(Number(event.target.value))} /></label>
                <label><span>Coverage</span><select value={displayedCoveragePreset} disabled={Boolean(selectedRun)} onChange={(event) => { const preset = event.target.value; setCoveragePreset(preset); if (coveragePresetClips[preset]) setClipCount(coveragePresetClips[preset]); }}><option value="turntable">360° only · 全身1 clip（顔学習不足）</option><option value="essential">Full-body views · 2 clips（顔学習不足）</option><option value="balanced">Balanced · 全身2 + 顔/Bust 1</option><option value="complete">Complete · 全身/顔/手・Pose 6</option><option value="custom">Custom</option>{selectedRun && displayedCoveragePreset === "legacy" && <option value="legacy">Legacy</option>}</select></label>
                <label><span>H3 clips</span><select value={displayedClipCount} disabled={Boolean(selectedRun) || coveragePreset !== "custom"} onChange={(event) => setClipCount(Number(event.target.value))}><option value={1}>1</option><option value={2}>2</option><option value={3}>3</option><option value={4}>4</option><option value={5}>5</option><option value={6}>6</option></select></label>
                <label><span>Output</span><input value={displayedCoveragePreset === "turntable" ? "1 clip × 124 frames · 360° locked turntable" : `${displayedClipCount} clips × 56 frames · 512 × 768`} disabled /></label>
                <label><span>Target</span><input value="Physical GPU 1" disabled /></label>
              </div>
              {frozenClipPlans.length > 0 && <section className="factory-clip-plan-list" aria-label="Frozen H3 clip plan">
                <header><span>FROZEN CLIP PLAN</span><strong>{frozenClipPlans.length} clips · Run #{selectedRun?.id}</strong></header>
                <div>{frozenClipPlans.map((plan, index) => <article key={plan.clip_id ?? index}>
                  <span>{plan.clip_id ?? `clip_${index + 1}`}</span>
                  <strong>{Object.entries(plan.coverage ?? {}).flatMap(([dimension, values]) => values.map((value) => `${dimension}: ${value}`)).join(" · ") || "legacy coverage"}</strong>
                  <small>seed {plan.seed ?? "unknown"} · {plan.prompt ?? "prompt unknown"}</small>
                </article>)}</div>
              </section>}
              <div className="factory-plan-actions">
                <button className="studio-secondary-button" data-action="h3-preflight" onClick={() => void runPreflight()} disabled={Boolean(selectedRun) || !referenceAssets.length || busy !== ""}>{busy === "preflight" ? <Loader2 size={15} className="spin" /> : <Cpu size={15} />}Preflight</button>
                <button className="studio-secondary-button" data-action="prepare-h3-run" onClick={() => void prepareRun()} disabled={Boolean(selectedRun) || !referenceAssets.length || busy !== ""}>{busy === "prepare" ? <Loader2 size={15} className="spin" /> : <FileImage size={15} />}計画だけ保存</button>
                <button className={preflight?.status === "ready" && !selectedRun ? "studio-danger-button" : "studio-primary-button"} data-action="queue-h3-gpu1" onClick={() => void executeH3()} disabled={busy !== "" || (selectedRun ? selectedRun.status !== "prepared" || selectedRun.current_stage !== "planning" : !preflight)}>{busy === "h3" ? <Loader2 size={15} className="spin" /> : selectedRun?.status === "waiting" ? <CircleDot size={15} /> : <Play size={15} />}{selectedRun?.status === "waiting" ? `GPU1待機中 · #${selectedRun.queue_position ?? 1}` : selectedRun?.status === "queued" || selectedRun?.status === "running" ? "H3実行中" : selectedRun?.status === "prepared" ? "保存済みRunをGPU1待機へ" : preflight?.status === "ready" ? "H3をGPU1で実行" : "GPU1待機キューに追加"}</button>
                {selectedRun?.status === "waiting" && <button className="studio-secondary-button" data-action="cancel-h3-waiting" onClick={(event) => openCancelConfirmation("h3", event.currentTarget)} disabled={busy !== ""}><X size={15} />待機を取消</button>}
              </div>
            </section>

            <section className="factory-evidence-card">
              <div className="factory-card-heading"><span><CheckCircle2 size={16} /></span><div><small>RUNTIME GATE</small><h3>実行条件</h3></div></div>
              {!preflight ? <div className="factory-empty-evidence"><CircleDot size={18} /><p>Preflightを実行すると、Reference、Workflow、Model、ComfyUI、物理GPUの実測結果を表示します。</p></div> : <div className="factory-check-list">{preflight.checks.map((check) => <div key={check.label} className={check.ok ? "is-ok" : "is-blocked"}><span>{check.ok ? <Check size={13} /> : <X size={13} />}</span><p><strong>{check.label}</strong><small>{check.detail}</small></p></div>)}</div>}
              {selectedRun?.error_detail && <p className="factory-run-error"><AlertTriangle size={14} />{selectedRun.error_detail}</p>}
            </section>
          </div> : <div className="factory-review-workspace">
            <section className={`factory-aesthetic-authority ${identityReviewUserApproved ? "is-user-approved" : "is-unreviewed"}`} aria-label="Aesthetic review authority">
              <span>{identityReviewUserApproved ? <CheckCircle2 size={16} /> : <LockKeyhole size={16} />}</span>
              <div><strong>{identityReviewUserApproved ? "USER_APPROVED" : "AESTHETIC_UNREVIEWED"}</strong><p>Frame採否とOriginal / Correctedの採用は、あなたの明示操作だけで確定します。指標・自動処理・Agent判断では次工程へ進みません。</p>
                {!identityReviewUserApproved && !reviewEditable && ["identity_review", "qwen_correction", "balancing", "captioning"].includes(selectedRun?.current_stage ?? "") && !["waiting", "queued", "running", "waiting_qwen", "running_qwen"].includes(selectedRun?.status ?? "") && <button className="studio-secondary-button" data-action="reopen-identity-review" onClick={() => void reopenIdentityReview()} disabled={busy !== ""}>{busy === "reopen-review" ? <Loader2 size={13} className="spin" /> : <RefreshCw size={13} />}Frame Reviewを開き直す</button>}
              </div>
            </section>
            <div className="factory-selection-toolbar">
              <span><strong>{selectedFrameIds.length}</strong> selected · <kbd>Shift</kbd> range · <kbd>Ctrl+A</kbd> all · <kbd>A</kbd> approve · <kbd>R</kbd> reject · {savingDecisionIds.length ? "保存中…" : reviewEditable ? "自動保存" : "Review封印済み"}</span>
              <div><button disabled={!reviewEditable || selectedFrameIds.length === 0 || savingDecisionIds.length > 0} onClick={() => selectedFrameIds[0] != null && void setFrameDecision(selectedFrameIds[0], "approved")}><Check size={14} />承認</button><button disabled={!reviewEditable || selectedFrameIds.length === 0 || savingDecisionIds.length > 0} onClick={() => selectedFrameIds[0] != null && void setFrameDecision(selectedFrameIds[0], "rejected")}><X size={14} />除外</button></div>
            </div>
            {coverageSummary.length > 0 && <section className="factory-coverage-strip" aria-label="Dataset generation coverage">
              <header><span>GENERATION COVERAGE</span><strong>生成意図から集計 · クリックで該当Frameを選択</strong></header>
              <div>{coverageSummary.map((item) => <article key={item.dimension}><small>{item.label}</small><div>{item.values.map((entry) => <button key={entry.value} type="button" onClick={() => selectCoverage(item.dimension, entry.value)} title={`${entry.total} candidates / ${entry.approved} approved`}><span>{entry.value}</span><em>{entry.approved}/{entry.total}</em></button>)}</div></article>)}</div>
            </section>}
            <div className="factory-frame-layout">
              <div className="factory-frame-grid" tabIndex={0} onKeyDown={frameGridKeyDown} aria-label="Generated frame candidates">
                {frames.map((frame) => {
                  const decision = normalizeDecision(decisions[frame.asset_id] ?? frame.review_status);
                  const selected = selectedFrameIds.includes(frame.asset_id);
                  return <button key={frame.asset_id} data-frame-id={frame.asset_id} className={`factory-frame-cell is-${decision} ${selected ? "is-selected" : ""}`} onClick={(event) => selectFrame(frame, event)} onKeyDown={(event) => {
                    if (event.key === "ContextMenu" || (event.shiftKey && event.key === "F10")) {
                      event.preventDefault();
                      prepareFrameContextSelection(frame.asset_id);
                      openKeyboardMenu("frame", frame.asset_id, event.currentTarget);
                    }
                  }} onContextMenu={(event) => { event.preventDefault(); prepareFrameContextSelection(frame.asset_id); openMenu({ kind: "frame", assetId: frame.asset_id, x: event.clientX, y: event.clientY }, event.currentTarget); }}>
                    <img src={`${API_BASE}/collector/thumbnail?path=${encodeURIComponent(frame.file_path)}&size=360`} alt={`Frame ${frame.ordinal + 1}`} />
                    <span className="factory-frame-index">{String(frame.ordinal + 1).padStart(2, "0")}</span>
                    <span className="factory-frame-decision">{decision === "approved" ? <Check size={13} /> : decision === "rejected" ? <X size={13} /> : <CircleDot size={13} />}</span>
                    {frame.rejection_reason && <small>{frame.rejection_reason}</small>}
                  </button>;
                })}
              </div>
              <aside className="factory-frame-inspector">
                {!selectedFrame ? <div className="factory-empty-evidence"><ScanFace size={22} /><p>Frameを選択してください。</p></div> : <>
                  <div className={`factory-version-compare ${previewQwenVersion ? "has-after" : ""}`}>
                    <figure className={selectedTrainingInputChoice === "original" ? "is-selected" : ""}>
                      <img src={`${API_BASE}/collector/thumbnail?path=${encodeURIComponent(selectedFrame.file_path)}&size=720`} alt={`H3 Frame ${selectedFrame.ordinal + 1}`} />
                      <figcaption><span>{selectedTrainingInputChoice === "original" ? (trainingInputFrozen ? "SNAPSHOT INPUT" : "TRAINING INPUT") : "ORIGINAL"}</span><strong>H3 Frame</strong></figcaption>
                      <button type="button" className="factory-expand-preview" data-action="expand-original-frame" onClick={() => setExpandedPreview({ path: selectedFrame.file_path, label: `Original Frame #${selectedFrame.ordinal + 1}` })} aria-label="Originalを全体表示"><Maximize2 size={13} />全体表示</button>
                    </figure>
                    {previewQwenVersion && <figure className={selectedQwenVersion ? "is-selected" : ""}>
                      <img src={`${API_BASE}/collector/thumbnail?path=${encodeURIComponent(previewQwenVersion.file_path)}&size=720`} alt={`Qwen Version ${previewQwenVersion.id}`} />
                      <figcaption><span>{selectedQwenVersion ? (trainingInputFrozen ? "SNAPSHOT INPUT" : "TRAINING INPUT") : "ENHANCED CANDIDATE"}</span><strong>{previewQwenVersion.version_kind} #{previewQwenVersion.id}</strong></figcaption>
                      <button type="button" className="factory-expand-preview" data-action="expand-enhanced-frame" onClick={() => setExpandedPreview({ path: previewQwenVersion.file_path, label: `${previewQwenVersion.version_kind} #${previewQwenVersion.id}` })} aria-label="Enhancedを全体表示"><Maximize2 size={13} />全体表示</button>
                    </figure>}
                  </div>
                  <div className="factory-inspector-title"><span><small>FRAME</small><strong>#{selectedFrame.ordinal + 1}</strong></span><em className={`is-${normalizeDecision(decisions[selectedFrame.asset_id] ?? selectedFrame.review_status)}`}>{normalizeDecision(decisions[selectedFrame.asset_id] ?? selectedFrame.review_status)}</em></div>
                  <div className="studio-segmented"><button disabled={!reviewEditable || savingDecisionIds.includes(selectedFrame.asset_id)} className={normalizeDecision(decisions[selectedFrame.asset_id] ?? selectedFrame.review_status) === "approved" ? "is-active" : ""} onClick={() => void setFrameDecision(selectedFrame.asset_id, "approved")}><Check size={13} />承認</button><button disabled={!reviewEditable || savingDecisionIds.includes(selectedFrame.asset_id)} className={normalizeDecision(decisions[selectedFrame.asset_id] ?? selectedFrame.review_status) === "rejected" ? "is-active" : ""} onClick={() => void setFrameDecision(selectedFrame.asset_id, "rejected")}><X size={13} />除外</button></div>
                  {inputChoiceEditable && normalizeDecision(decisions[selectedFrame.asset_id] ?? selectedFrame.review_status) === "approved" && <div className="studio-segmented is-two" aria-label="Training input choice"><button className={selectedTrainingInputChoice === "original" ? "is-active" : ""} disabled={busy !== ""} onClick={() => void selectOriginalTrainingInput()}><Check size={13} />Accept Original</button><span className={selectedTrainingInputChoice === "qwen" ? "is-active" : ""} aria-disabled="true">Correctedは下から選択</span></div>}
                  {qwenVersions.length > 0 && <section className="factory-version-picker" aria-label="Enhancement comparison versions">
                    <header><span>HQ VERSIONS · FLASHVSR / REALESRGAN / QWEN</span><strong>{trainingInputFrozen ? selectedVersionResult?.training_input_lock_reason : selectedQwenVersion ? `${selectedQwenVersion.version_kind} #${selectedQwenVersion.id}を採用中` : selectedTrainingInputChoice === "original" ? "Originalを採用中" : "Original / Enhancedを選択"}</strong></header>
                    {!selectedTrainingInputChoice && <p className="factory-version-warning"><AlertTriangle size={12} />自動採用しません。Balance前にTraining Inputを明示選択してください。</p>}
                    <div className="factory-version-list">
                      {qwenVersions.map((version) => <button
                        key={version.id}
                        type="button"
                        className={["qwen", "enhanced"].includes(selectedTrainingInputChoice ?? "") && version.selected ? "is-selected" : ""}
                        onClick={() => void selectQwenVersion(version)}
                        disabled={busy !== "" || !inputChoiceEditable || trainingInputFrozen}
                        aria-pressed={["qwen", "enhanced"].includes(selectedTrainingInputChoice ?? "") && Boolean(version.selected)}
                      >
                        <img src={`${API_BASE}/collector/thumbnail?path=${encodeURIComponent(version.file_path)}&size=240`} alt={`Qwen Version ${version.id}`} />
                        <span><strong>{version.version_kind}</strong><small>Version #{version.id} · seed {version.seed ?? "unknown"}</small></span>
                        <em>{["qwen", "enhanced"].includes(selectedTrainingInputChoice ?? "") && version.selected ? <><Check size={11} />採用中</> : "Accept Enhanced"}</em>
                      </button>)}
                    </div>
                  </section>}
                  <label className="studio-field"><span>Training Caption {captionEditable ? "· blurで自動保存" : "· 読み取り専用"}</span><textarea rows={5} disabled={!captionEditable || savingDecisionIds.includes(selectedFrame.asset_id)} value={captions[selectedFrame.asset_id] ?? ""} onChange={(event) => setCaptions((current) => ({ ...current, [selectedFrame.asset_id]: event.target.value }))} onBlur={() => void saveFrameCaption(selectedFrame.asset_id)} placeholder="Identity Review後に学習入力を確定" /></label>
                  <dl><dt>Blur score</dt><dd>{selectedFrame.blur_score ?? "unknown"}</dd><dt>Source</dt><dd>H3 Run #{selectedRun?.id}</dd><dt>Clip</dt><dd>{selectedFrame.clip_id ?? "legacy"} / frame {(selectedFrame.clip_frame_index ?? selectedFrame.ordinal) + 1}</dd><dt>Coverage</dt><dd>{Object.values(selectedFrame.coverage ?? {}).flat().join(" · ") || "legacy"}</dd><dt>Timestamp</dt><dd>{selectedFrame.timestamp_seconds != null ? `${selectedFrame.timestamp_seconds.toFixed(2)}s` : "unknown"}</dd><dt>Seed</dt><dd>{selectedFrame.generation_seed ?? "unknown"}</dd><dt>Asset ID</dt><dd>#{selectedFrame.asset_id}</dd></dl>
                </>}
              </aside>
            </div>

            <div className="factory-review-actions">
              <button data-action="commit-identity-review" onClick={requestIdentityReviewConfirmation} disabled={selectedRun?.current_stage !== "duplicate_filter" || unresolved > 0 || approvedIds.length === 0 || busy !== ""}><ScanFace size={15} />Identity Reviewを確定</button>
              <button data-action="accept-original-selection" onClick={() => void selectOriginalTrainingInputs(selectedApprovedIds.length ? selectedApprovedIds : approvedIds)} disabled={!inputChoiceEditable || !approvedIds.length || busy !== ""}><Check size={15} />Original採用 {selectedApprovedIds.length || approvedIds.length}</button>
              <button data-action="plan-qwen-correction" onClick={() => void qwen(false)} disabled={selectedRun?.current_stage !== "identity_review" || !qwenTargetIds.length || busy !== ""}><WandSparkles size={15} />高品質化プラン {qwenTargetIds.length}</button>
              <button className="is-gpu" data-action="execute-qwen-gpu1" onClick={() => void qwen(true)} disabled={selectedRun?.current_stage !== "identity_review" || !qwenTargetIds.length || busy !== "" || ["waiting_qwen", "running_qwen"].includes(selectedRun?.status ?? "")}>
                {selectedRun?.status === "waiting_qwen" ? <CircleDot size={15} /> : selectedRun?.status === "running_qwen" ? <Loader2 size={15} className="spin" /> : <Play size={15} />}
                {selectedRun?.status === "waiting_qwen" ? `高品質化待機中 · #${selectedRun.queue_position ?? 1}` : selectedRun?.status === "running_qwen" ? "選択Frameを高品質化中" : `選択Frameを高品質化 ${qwenTargetIds.length}`}
              </button>
              <button data-action="skip-qwen-originals" onClick={() => void skipQwenWithOriginals()} disabled={selectedRun?.current_stage !== "identity_review" || !allApprovedOriginal || busy !== ""}><ArrowRight size={15} />全Originalで続行</button>
              {selectedRun?.status === "waiting_qwen" && <button data-action="cancel-qwen-waiting" onClick={(event) => openCancelConfirmation("qwen", event.currentTarget)} disabled={busy !== ""}><X size={15} />Qwen待機を取消</button>}
              <button data-action="balance-dataset" onClick={() => void balance()} disabled={selectedRun?.current_stage !== "qwen_correction" || !identityReviewUserApproved || missingQwenInputIds.length > 0 || ["cancelled", "waiting_qwen", "running_qwen"].includes(selectedRun?.status ?? "") || busy !== ""}><Images size={15} />Balance</button>
              <div className="factory-caption-bulk"><input disabled={!captionEditable} value={captionTemplate} onChange={(event) => setCaptionTemplate(event.target.value)} /><button disabled={!captionEditable || savingDecisionIds.length > 0} onClick={() => void applyCaptionTemplate(selectedFrameIds.length ? selectedFrameIds : approvedIds)}>Caption適用</button></div>
              <button data-action="commit-frame-captions" onClick={() => void completeCaptioning()} disabled={selectedRun?.current_stage !== "balancing" || busy !== ""}><FileImage size={15} />Caption確定</button>
              <button className="is-primary" data-action="seal-character-snapshot" onClick={() => void sealSnapshot()} disabled={selectedRun?.current_stage !== "captioning" || missingQwenInputIds.length > 0 || busy !== ""}><Database size={15} />Snapshot封印</button>
            </div>
            {selectedRun?.current_stage === "captioning" && missingQwenInputIds.length > 0 && <p className="factory-run-error"><AlertTriangle size={14} />Original / Corrected未選択: Asset #{missingQwenInputIds.join(", #")}</p>}
          </div>}
        </main>
      </div>

      <footer className="factory-status-bar">
        <span><CircleDot size={12} /><strong>Engine</strong> H3 R2V → Qwen Image Edit → Anima</span>
        <span><strong>Run</strong> {selectedRun ? `#${selectedRun.id} · ${selectedRun.status}` : "not created"}</span>
        <span><strong>Dataset</strong> {workspace.snapshots[0] ? `#${workspace.snapshots[0].id} · ${workspace.snapshots[0].item_count} items` : "not sealed"}</span>
        <button onClick={onOpenDataset}>既存Datasetを管理 <ArrowRight size={13} /></button>
      </footer>

      {menu && <div ref={menuRef} className="factory-context-menu" style={{ left: menu.x, top: menu.y }} role="menu" aria-label={menu.kind === "reference" ? "Reference actions" : "Frame actions"} onPointerDown={(event) => event.stopPropagation()} onKeyDown={navigateContextMenu}>
        {menu.kind === "reference" && selectedRun ? <>
          <p>REFERENCE ROLE · RUN #{selectedRun.id} SEALED</p>
          <span className="factory-context-readonly">このRunへ保存した四面Roleは変更できません。</span>
          <button role="menuitem" onClick={() => { selectGenerationRun(""); setMenu(null); }}><span>新規計画で編集</span><small>Run証拠は保持</small></button>
        </> : menu.kind === "reference" ? <>
          <p>REFERENCE ROLE</p>
          {roles.map((role) => <button role="menuitem" key={role.id} onClick={() => { void assignReference(menu.assetId, role.id); setMenu(null); }}><span>{role.label}</span><small>{role.hint}</small></button>)}
          <hr /><button role="menuitem" className="is-danger" onClick={() => { void assignReference(menu.assetId, "unused"); setMenu(null); }}><span>参照セットから外す</span><small>Asset履歴は保持</small></button>
        </> : reviewEditable ? <>
          <p>FRAME ACTIONS · {selectedFrameIds.length || 1} ITEMS</p>
          <button role="menuitem" onClick={() => { void setFrameDecision(menu.assetId, "approved"); setMenu(null); }}><span>承認</span><small>A</small></button>
          <button role="menuitem" onClick={() => { void setFrameDecision(menu.assetId, "rejected"); setMenu(null); }}><span>除外</span><small>R / Delete</small></button>
        </> : inputChoiceEditable ? <><p>TRAINING INPUT · {selectedFrameIds.length || 1} ITEMS</p><button role="menuitem" disabled={normalizeDecision(decisions[menu.assetId] ?? frames.find((frame) => frame.asset_id === menu.assetId)?.review_status) !== "approved"} onClick={() => { void selectOriginalTrainingInputs(selectedFrameIds.length ? selectedFrameIds : [menu.assetId]); setMenu(null); }}><span>Originalを採用</span><small>approvedのみ</small></button><span className="factory-context-readonly">Qwen補正は選択セルだけを下部アクションから実行できます。</span></> : captionEditable ? <><p>CAPTION ACTIONS · {selectedFrameIds.length || 1} ITEMS</p><button role="menuitem" onClick={() => { void applyCaptionTemplate(selectedFrameIds.length ? selectedFrameIds : [menu.assetId]); setMenu(null); }}><span>Caption templateを適用</span><small>選択範囲</small></button></> : <><p>FRAME REVIEW SEALED</p><span className="factory-context-readonly">このRunのIdentity Reviewは確定済みです。</span></>}
      </div>}

      {aestheticConfirmation && <div className="studio-dialog-backdrop" role="presentation" onMouseDown={closeAestheticConfirmation}>
        <section ref={aestheticDialogRef} className="studio-modal factory-aesthetic-dialog" role="dialog" aria-modal="true" aria-labelledby="aesthetic-confirm-title" onMouseDown={(event) => event.stopPropagation()} onKeyDown={handleAestheticDialogKey}>
          <header><div><p className="studio-eyebrow">USER AESTHETIC DECISION</p><h2 id="aesthetic-confirm-title">この審美判断を確定しますか？</h2></div><span className="factory-aesthetic-state">AESTHETIC_UNREVIEWED</span></header>
          {aestheticConfirmation.kind === "identity" && <div className="factory-aesthetic-summary"><ScanFace size={24} /><div><strong>Frame採否を確定</strong><p>採用 {aestheticConfirmation.approvedCount}件 / 除外 {aestheticConfirmation.rejectedCount}件。画像を確認したあなたの判断として保存します。</p></div></div>}
          {aestheticConfirmation.kind === "original" && <div className="factory-aesthetic-summary"><Images size={24} /><div><strong>OriginalをTraining Inputに採用</strong><p>{aestheticConfirmation.assetIds.length}件を補正せず使用します。品質指標やAgent提案ではなく、あなたの選択として記録します。</p></div></div>}
          {aestheticConfirmation.kind === "qwen" && <div className="factory-aesthetic-confirm-compare">
            {frames.find((frame) => frame.asset_id === aestheticConfirmation.assetId) && <figure><img src={`${API_BASE}/collector/thumbnail?path=${encodeURIComponent(frames.find((frame) => frame.asset_id === aestheticConfirmation.assetId)!.file_path)}&size=480`} alt="Original candidate" /><figcaption>Original</figcaption></figure>}
            <figure><img src={`${API_BASE}/collector/thumbnail?path=${encodeURIComponent(aestheticConfirmation.version.file_path)}&size=480`} alt={`Corrected candidate ${aestheticConfirmation.version.id}`} /><figcaption>Corrected #{aestheticConfirmation.version.id}</figcaption></figure>
          </div>}
          <p className="studio-modal-copy">この操作だけが <strong>USER_APPROVED</strong> を記録します。自動評価値は参考情報であり、採用を確定しません。</p>
          <footer><button autoFocus className="studio-secondary-button" onClick={closeAestheticConfirmation} disabled={["review", "version-select"].includes(busy)}>まだ確定しない</button>
            {aestheticConfirmation.kind === "identity" && <button className="studio-primary-button" onClick={() => void commitIdentityReview()} disabled={busy !== ""}>{busy === "review" ? <Loader2 size={15} className="spin" /> : <Check size={15} />}Frame採否を確定</button>}
            {aestheticConfirmation.kind === "original" && <button className="studio-primary-button" onClick={() => void confirmOriginalTrainingInputs(aestheticConfirmation.assetIds)} disabled={busy !== ""}>{busy === "version-select" ? <Loader2 size={15} className="spin" /> : <Check size={15} />}Originalを採用</button>}
            {aestheticConfirmation.kind === "qwen" && <button className="studio-primary-button" onClick={() => void confirmQwenVersion(aestheticConfirmation.assetId, aestheticConfirmation.version)} disabled={busy !== ""}>{busy === "version-select" ? <Loader2 size={15} className="spin" /> : <Check size={15} />}Correctedを採用</button>}
          </footer>
        </section>
      </div>}

      {expandedPreview && <div className="studio-dialog-backdrop factory-preview-backdrop" role="presentation" onMouseDown={() => setExpandedPreview(null)}>
        <section className="factory-full-preview" role="dialog" aria-modal="true" aria-label={`${expandedPreview.label} 全体表示`} onMouseDown={(event) => event.stopPropagation()}>
          <header><strong>{expandedPreview.label}</strong><span>画像全体 · 縦横比維持 · クロップなし</span><button type="button" onClick={() => setExpandedPreview(null)} aria-label="全体表示を閉じる"><X size={17} /></button></header>
          <div><img src={`${API_BASE}/collector/thumbnail?path=${encodeURIComponent(expandedPreview.path)}&size=1600`} alt={expandedPreview.label} /></div>
        </section>
      </div>}

      {cancelConfirmStage && selectedRun && <div className="studio-dialog-backdrop" role="presentation" onMouseDown={closeCancelConfirmation}>
        <section ref={cancelDialogRef} className="studio-modal is-compact" role="dialog" aria-modal="true" aria-labelledby="cancel-waiting-title" onMouseDown={(event) => event.stopPropagation()} onKeyDown={handleCancelDialogKey}>
          <header><div><p className="studio-eyebrow">GPU WAITING QUEUE</p><h2 id="cancel-waiting-title">Run #{selectedRun.id}の待機を取り消しますか？</h2></div></header>
          <p className="studio-modal-copy">{cancelConfirmStage === "qwen" ? "Qwen補正" : "H3生成"}のGPU処理はまだ開始されていません。取り消すと安全条件の自動再評価を停止し、再開には手動で待機キューへ追加し直す必要があります。</p>
          <footer><button autoFocus className="studio-primary-button" onClick={closeCancelConfirmation} disabled={busy === "cancel-waiting"}>待機を維持</button><button className="studio-danger-button" onClick={() => void cancelWaitingRun()} disabled={busy === "cancel-waiting"}>{busy === "cancel-waiting" ? <Loader2 size={15} className="spin" /> : <X size={15} />}待機を取り消す</button></footer>
        </section>
      </div>}
    </div>
  );
}
