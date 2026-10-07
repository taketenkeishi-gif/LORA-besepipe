import { useEffect, useState } from "react";
import { AlertTriangle, BarChart3, Check, CheckCircle2, ChevronDown, Clock3, Code2, Eye, FileText, GitCompare, Loader2, RefreshCw, Save, ShieldCheck } from "lucide-react";
import { apiGet, apiPost } from "../../lib/api";
import type { PreviewJob, PreviewJobsResponse, PreviewSample, Project, RunDetail, RunSummary } from "../../types";

type ViewId = "overview" | "compare" | "evidence";
type Props = { project: Project; initialRunId: number | null; onSelectRun: (runId: number) => void; showError: (message: string) => void; showNotice: (message: string) => void; onOpenLibrary: () => void };
type Evaluation = {
  checkpoint_id: number;
  preview_slot: string;
  preview_profile_snapshot_id?: number | null;
  identity?: number | null;
  outfit_separation?: number | null;
  style_durability?: number | null;
  style_quality?: number | null;
  prompt_flexibility?: number | null;
  artifact?: number | null;
  accepted: number | boolean;
  note: string;
  review_state?: "AESTHETIC_UNREVIEWED" | "USER_APPROVED";
  reviewed_by?: string;
};
type EvaluationKey = `${number}:${string}`;
const RATING_FIELDS = [
  ["identity", "Identity"],
  ["outfit_separation", "Outfit separation"],
  ["style_durability", "Style durability"],
  ["style_quality", "Style quality"],
  ["prompt_flexibility", "Prompt flexibility"],
  ["artifact", "Artifact safety"],
] as const;

function statusCopy(status: string) {
  if (status === "completed") return "Completed";
  if (status === "training") return "Running";
  if (status === "error") return "Failed";
  if (status === "paused") return "Stopped";
  return status;
}

function duration(seconds?: number | null) {
  if (seconds == null) return "未計測";
  if (seconds < 60) return `${seconds.toFixed(0)} sec`;
  return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
}

export default function EvaluateWorkspace({ project, initialRunId, onSelectRun, showError, showNotice, onOpenLibrary }: Props) {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<number | null>(null);
  const [missingRunId, setMissingRunId] = useState<number | null>(null);
  const [detail, setDetail] = useState<RunDetail | null>(null);
  const [timeline, setTimeline] = useState<PreviewSample[]>([]);
  const [view, setView] = useState<ViewId>("overview");
  const [loading, setLoading] = useState(false);
  const [jobsByCheckpoint, setJobsByCheckpoint] = useState<Record<number, PreviewJob[]>>({});
  const [evaluations, setEvaluations] = useState<Record<string, Evaluation>>({});
  const [savingEvaluation, setSavingEvaluation] = useState<EvaluationKey | null>(null);
  const [evaluationConfirmation, setEvaluationConfirmation] = useState<{ evaluation: Evaluation; profileSnapshotId: number } | null>(null);

  async function loadRuns() {
    setLoading(true);
    try {
      const [runResult, previewResult, evaluationResult] = await Promise.all([
        apiGet<{ runs: RunSummary[] }>(`/training/runs?project_id=${project.id}`),
        apiGet<{ timeline: PreviewSample[] }>(`/previews/${project.id}?include_images=false`).catch(() => ({ timeline: [] })),
        apiGet<{ evaluations: Evaluation[] }>(`/basepipe/projects/${project.id}/evaluations`),
      ]);
      const nextRuns = runResult.runs ?? [];
      const nextTimeline = previewResult.timeline ?? [];
      setRuns(nextRuns);
      setTimeline(nextTimeline);
      setEvaluations(Object.fromEntries((evaluationResult.evaluations ?? []).map((evaluation) => {
        const userApproved = evaluation.review_state === "USER_APPROVED" && evaluation.reviewed_by === "user";
        return [
          `${evaluation.checkpoint_id}:${evaluation.preview_slot}`,
          { ...evaluation, accepted: userApproved && Boolean(evaluation.accepted) },
        ];
      })));
      const checkpointIds = [...new Set(nextTimeline.map((sample) => sample.checkpoint_id))];
      const jobResults = await Promise.allSettled(checkpointIds.map((checkpointId) =>
        apiGet<PreviewJobsResponse>(`/previews/jobs/${checkpointId}`),
      ));
      setJobsByCheckpoint(Object.fromEntries(jobResults.flatMap((result, index) =>
        result.status === "fulfilled" ? [[checkpointIds[index], result.value.jobs ?? []] as const] : [],
      )));
      const requestedMissing = initialRunId != null && !nextRuns.some((run) => run.id === initialRunId);
      setMissingRunId(requestedMissing ? initialRunId : null);
      setSelectedRunId((current) => {
        if (requestedMissing) return null;
        if (initialRunId != null && nextRuns.some((run) => run.id === initialRunId)) return initialRunId;
        return current && nextRuns.some((run) => run.id === current) ? current : nextRuns[0]?.id ?? null;
      });
    } catch (error) {
      showError(error instanceof Error ? error.message : "Run一覧の取得に失敗しました");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { void loadRuns(); }, [project.id, initialRunId]);
  useEffect(() => {
    if (selectedRunId == null) { setDetail(null); return; }
    void apiGet<RunDetail>(`/training/runs/${selectedRunId}`).then(setDetail).catch((error) => showError(error instanceof Error ? error.message : "Run詳細の取得に失敗しました"));
  }, [selectedRunId]);
  useEffect(() => {
    const ids = detail?.evidence?.checkpoints?.map((checkpoint) => checkpoint.id) ?? [];
    if (!ids.length) return;
    let cancelled = false;
    void Promise.allSettled(ids.map((checkpointId) => apiGet<PreviewJobsResponse>(`/previews/jobs/${checkpointId}`)))
      .then((results) => {
        if (cancelled) return;
        setJobsByCheckpoint((current) => ({
          ...current,
          ...Object.fromEntries(results.flatMap((result, index) =>
            result.status === "fulfilled" ? [[ids[index], result.value.jobs ?? []] as const] : [],
          )),
        }));
      });
    return () => { cancelled = true; };
  }, [selectedRunId, detail?.evidence?.checkpoints]);

  const evidence = detail?.evidence;
  const resolvedLineage = evidence?.resolved as {
    dataset_snapshot_id?: number | null;
    dataset_snapshot_hash?: string | null;
    preview_profile_snapshot_id?: number | null;
    preview_profile_snapshot_hash?: string | null;
    resolved_training_backend?: string | null;
  } | undefined;
  const selectedRun = runs.find((run) => run.id === selectedRunId);
  const checkpointIds = new Set(evidence?.checkpoints?.map((checkpoint) => checkpoint.id) ?? []);
  const runPreviews = timeline.filter((sample) => checkpointIds.has(sample.checkpoint_id));
  const previewCards = runPreviews.flatMap((sample) => Object.entries(sample.sample_previews ?? sample.samples ?? {}).map(([slot, image]) => ({ sample, slot, image })));
  const selectedJobs = [...checkpointIds].flatMap((checkpointId) => jobsByCheckpoint[checkpointId] ?? []);
  const previewWaitReasons = [...new Set(selectedJobs.filter((job) => job.status === "pending" && job.error_detail).map((job) => job.error_detail))];
  const mismatches = evidence?.parameters.filter((parameter) => parameter.mismatch) ?? [];
  const stageHistory = evidence?.stage_history ?? [];
  const stages = stageHistory.length ? stageHistory : [
    { stage: "preflight", status: selectedRun ? "completed" : "pending" },
    { stage: "snapshot", status: evidence?.manifest?.dataset_snapshot_id ? "completed" : "unknown" },
    { stage: "training", status: evidence?.training_status ?? selectedRun?.status ?? "pending" },
    { stage: "preview", status: (evidence?.preview?.succeeded ?? 0) > 0 ? "completed" : previewWaitReasons.length ? "waiting" : "pending" },
    { stage: "finalize", status: selectedRun?.status === "completed" ? "completed" : "pending" },
  ];

  function selectRun(runId: number) {
    setMissingRunId(null);
    setSelectedRunId(runId);
    onSelectRun(runId);
  }

  function evaluationFor(checkpointId: number, slot: string, profileSnapshotId: number | null = null): Evaluation {
    const stored = evaluations[`${checkpointId}:${slot}`];
    if (stored && stored.preview_profile_snapshot_id != null && profileSnapshotId != null
      && stored.preview_profile_snapshot_id !== profileSnapshotId) {
      return {
        checkpoint_id: checkpointId,
        preview_slot: slot,
        preview_profile_snapshot_id: profileSnapshotId,
        accepted: false,
        note: "",
      };
    }
    return stored ?? {
      checkpoint_id: checkpointId,
      preview_slot: slot,
      preview_profile_snapshot_id: profileSnapshotId,
      accepted: false,
      note: "",
    };
  }

  function jobFor(checkpointId: number, slot: string): PreviewJob | null {
    const match = /^p(\d+)_i(\d+)$/.exec(slot);
    if (!match) return null;
    return jobsByCheckpoint[checkpointId]?.find((job) =>
      job.prompt_index === Number(match[1]) && job.instance_index === Number(match[2]),
    ) ?? null;
  }

  function updateEvaluation(key: EvaluationKey, field: keyof Evaluation, value: string | number | boolean | null, profileSnapshotId: number | null) {
    const separator = key.indexOf(":");
    const checkpointId = Number(key.slice(0, separator));
    const previewSlot = key.slice(separator + 1);
    setEvaluations((current) => ({
      ...current,
      [key]: {
        ...evaluationFor(checkpointId, previewSlot, profileSnapshotId),
        [field]: value,
      },
    }));
  }

  async function saveEvaluation(evaluation: Evaluation, profileSnapshotId: number | null, humanConfirmed = false) {
    const key = `${evaluation.checkpoint_id}:${evaluation.preview_slot}` as EvaluationKey;
    if (evaluation.accepted && profileSnapshotId == null) {
      showError("採用判定には固定Preview Snapshotの証拠が必要です");
      return;
    }
    if (evaluation.accepted && !humanConfirmed) {
      setEvaluationConfirmation({ evaluation, profileSnapshotId: profileSnapshotId! });
      return;
    }
    setSavingEvaluation(key);
    try {
      const saved = await apiPost<Evaluation>(`/basepipe/projects/${project.id}/evaluations`, {
        ...evaluation,
        preview_profile_snapshot_id: profileSnapshotId,
        accepted: Boolean(evaluation.accepted),
        human_confirmed: humanConfirmed,
      });
      setEvaluations((current) => ({ ...current, [key]: saved }));
      setEvaluationConfirmation(null);
      showNotice(`${evaluation.preview_slot} のHuman reviewを保存しました`);
    } catch (error) {
      showError(error instanceof Error ? error.message : "Human reviewの保存に失敗しました");
    } finally {
      setSavingEvaluation(null);
    }
  }

  return (
    <div className="studio-workspace is-evaluate">
      <header className="studio-page-header">
        <div><p className="studio-eyebrow">03 · EVALUATE</p><h1>結果を見て、次を決める</h1><p>Runの状態、同条件Preview、Evidenceを必要な深さで確認します。</p></div>
        <button className="studio-secondary-button" onClick={() => void loadRuns()} disabled={loading}><RefreshCw size={15} className={loading ? "spin" : ""} /> 更新</button>
      </header>

      <div className="studio-evaluate-layout">
        <aside className="studio-run-list">
          <header><p className="studio-pane-label">RUN HISTORY</p><span>{runs.length}</span></header>
          {runs.map((run) => <button key={run.id} className={selectedRunId === run.id ? "is-active" : ""} onClick={() => selectRun(run.id)}>
            <span className={`studio-run-icon is-${run.status}`}>{run.status === "completed" ? <Check size={14} /> : run.status === "error" ? <AlertTriangle size={14} /> : <Clock3 size={14} />}</span>
            <span><strong>Run #{run.id}</strong><small>{statusCopy(run.status)} · E{run.current_epoch}/{run.total_epochs}</small><i>{run.current_stage ?? "planning"}</i></span>
          </button>)}
          {runs.length === 0 && <p className="studio-empty-line">Runはまだありません</p>}
        </aside>

        <section className="studio-run-workspace">
          {!selectedRun || !detail ? <div className="studio-loading-state">{missingRunId != null ? <><AlertTriangle size={22} /><p>指定されたSource Run #{missingRunId}はこのProjectに存在しません</p></> : selectedRun ? <><Loader2 className="spin" size={22} /><p>Runを読み込んでいます</p></> : <p>Runを選択してください</p>}</div> : <>
            <header className="studio-run-header">
              <div><span className={`studio-state-badge is-${selectedRun.status}`}>{statusCopy(selectedRun.status)}</span><h2>Run #{selectedRun.id}</h2><p>{detail.evidence.manifest?.model_family ? String(detail.evidence.manifest.model_family) : "model unknown"} · {resolvedLineage?.resolved_training_backend ?? "backend unknown"}</p><p>Dataset #{resolvedLineage?.dataset_snapshot_id ?? selectedRun.dataset_snapshot_id ?? "unknown"} · {resolvedLineage?.dataset_snapshot_hash?.slice(0, 12) ?? "hash unknown"} / Preview #{resolvedLineage?.preview_profile_snapshot_id ?? "unknown"} · {resolvedLineage?.preview_profile_snapshot_hash?.slice(0, 12) ?? "hash unknown"}</p></div>
              <div className="studio-run-header-metrics"><span><small>Epoch</small><strong>{selectedRun.current_epoch} / {selectedRun.total_epochs}</strong></span><span><small>Checkpoints</small><strong>{evidence?.checkpoints?.length ?? 0}</strong></span><span><small>Previews</small><strong>{evidence?.preview?.succeeded ?? 0} / {evidence?.preview?.expected ?? 0}</strong></span></div>
            </header>

            {evidence?.failure && <div className="studio-failure-card"><AlertTriangle size={19} /><span><strong>{evidence.failure.code ?? "Training stopped"}</strong><p>{evidence.failure.message ?? "Runが失敗しました。Datasetと設定は保持されています。"}</p><small>Stage: {evidence.failure.stage ?? selectedRun.current_stage ?? "unknown"}</small></span></div>}

            <nav className="studio-view-tabs" aria-label="Run detail views">
              <button className={view === "overview" ? "is-active" : ""} onClick={() => setView("overview")}><BarChart3 size={15} /> Overview</button>
              <button className={view === "compare" ? "is-active" : ""} onClick={() => setView("compare")}><GitCompare size={15} /> Compare <span>{previewCards.length}</span></button>
              <button className={view === "evidence" ? "is-active" : ""} onClick={() => setView("evidence")}><Code2 size={15} /> Evidence {mismatches.length > 0 && <span className="is-warning">{mismatches.length}</span>}</button>
            </nav>

            {view === "overview" && <div className="studio-run-overview">
              <section className="studio-stage-card"><div className="studio-card-heading"><div><p className="studio-eyebrow">PIPELINE</p><h3>実行ステージ</h3></div><span>{selectedRun.current_stage ?? "planning"}</span></div><div className="studio-stage-list">{stages.map((stage, index) => <div key={`${stage.stage}-${index}`} className={`is-${stage.status}`}><span>{stage.status === "completed" ? <Check size={13} /> : index + 1}</span><strong>{stage.stage.replace(/_/g, " ")}</strong><small>{stage.status}</small></div>)}</div></section>
              <section className="studio-timing-card"><div className="studio-card-heading"><div><p className="studio-eyebrow">TIMING</p><h3>実測時間</h3></div></div><dl><div><dt>Total training</dt><dd>{duration(evidence?.timing?.total_training_seconds)}</dd></div><div><dt>First checkpoint</dt><dd>{duration(evidence?.timing?.time_to_first_checkpoint_seconds)}</dd></div><div><dt>First preview</dt><dd>{duration(evidence?.timing?.time_to_first_preview_seconds)}</dd></div><div><dt>Cache</dt><dd>{duration(evidence?.timing?.cache_seconds)}</dd></div></dl></section>
              <section className="studio-preview-summary"><div className="studio-card-heading"><div><p className="studio-eyebrow">LATEST OUTPUT</p><h3>Preview</h3></div><button onClick={() => setView("compare")}>すべて比較</button></div>{previewCards[0]?.image ? <img src={previewCards[0].image} alt={`Run ${selectedRun.id} latest preview`} /> : <div className="studio-empty-preview">{previewWaitReasons.length ? <Clock3 size={24} /> : <Eye size={24} />}<p>{previewWaitReasons.length ? "PreviewはGPU1待機中" : "このRunにPreviewはありません"}</p>{previewWaitReasons[0] && <small>{previewWaitReasons[0]}</small>}</div>}</section>
            </div>}

            {view === "compare" && <div className="studio-compare-view">
              <div className="studio-compare-header"><div><p className="studio-eyebrow">SAME CONDITIONS</p><h3>Checkpoint比較</h3></div><p>Runに紐づく実Previewだけを表示します。固定条件の証拠がない値は推測しません。</p></div>
              {previewCards.length ? <div className="studio-compare-grid">{previewCards.map(({ sample, slot, image }) => {
                const key = `${sample.checkpoint_id}:${slot}` as EvaluationKey;
                const job = jobFor(sample.checkpoint_id, slot);
                const storedEvaluation = evaluationFor(sample.checkpoint_id, slot);
                const profileSnapshotId = job?.preview_snapshot?.profile_snapshot_id ?? storedEvaluation.preview_profile_snapshot_id ?? null;
                const evaluation = evaluationFor(sample.checkpoint_id, slot, profileSnapshotId);
                const conditionLocked = job?.status === "succeeded" && profileSnapshotId != null;
                const userApproved = Boolean(evaluation.accepted) && evaluation.review_state === "USER_APPROVED" && evaluation.reviewed_by === "user";
                return <article key={`${sample.checkpoint_id}-${slot}`} className={userApproved ? "is-reviewed" : ""}>
                  <div className="studio-compare-image"><img src={image} alt={`Epoch ${sample.epoch} ${slot}`} /></div>
                  <div className="studio-compare-copy"><strong>Epoch {sample.epoch}</strong><span>{slot}</span><small>Checkpoint #{sample.checkpoint_id} · step {sample.step}</small><p>{sample.validation_status ?? "validation unknown"}</p></div>
                  <section className="studio-human-review" aria-label={`${slot} Human review`}>
                    <header><span><ShieldCheck size={13} /> Human review</span><span className={`studio-aesthetic-review-state ${userApproved ? "is-approved" : ""}`}>{userApproved ? "USER_APPROVED" : "AESTHETIC_UNREVIEWED"}</span><small className={conditionLocked ? "is-locked" : "is-missing"}>{conditionLocked ? `条件固定 #${profileSnapshotId}` : "固定条件なし"}</small></header>
                    <div className="studio-score-grid">{RATING_FIELDS.map(([field, label]) => <label key={field}><span>{label}</span><select aria-label={`${slot} ${label}`} value={evaluation[field] ?? ""} onChange={(event) => updateEvaluation(key, field, event.target.value ? Number(event.target.value) : null, profileSnapshotId)}><option value="">—</option>{[1, 2, 3, 4, 5].map((score) => <option key={score} value={score}>{score}</option>)}</select></label>)}</div>
                    <label className="studio-accept-review"><input type="checkbox" checked={Boolean(evaluation.accepted)} disabled={!conditionLocked} onChange={(event) => updateEvaluation(key, "accepted", event.target.checked, profileSnapshotId)} /><span><strong>Final候補として承認</strong><small>固定条件のPreviewだけ承認できます</small></span></label>
                    <textarea aria-label={`${slot} 評価メモ`} value={evaluation.note ?? ""} onChange={(event) => updateEvaluation(key, "note", event.target.value, profileSnapshotId)} placeholder="判断理由や破綻箇所を記録" />
                    <button className="studio-review-save" disabled={savingEvaluation === key || (Boolean(evaluation.accepted) && !conditionLocked)} onClick={() => void saveEvaluation(evaluation, profileSnapshotId)}>{savingEvaluation === key ? <Loader2 size={13} className="spin" /> : <Save size={13} />} 評価を保存</button>
                  </section>
                </article>;
              })}</div> : <div className="studio-empty-state">{previewWaitReasons.length ? <Clock3 size={28} /> : <GitCompare size={28} />}<h3>{previewWaitReasons.length ? "Preview生成を安全待機中" : "比較可能なPreviewはありません"}</h3><p>{previewWaitReasons[0] ?? "固定条件のPreviewが生成されると、ここにEpochごとに並びます。"}</p></div>}
            </div>}

            {view === "evidence" && <div className="studio-evidence-view">
              <div className="studio-evidence-summary"><div><CheckCircle2 size={17} /><span><strong>{Math.max(0, (evidence?.parameters.length ?? 0) - mismatches.length)} values matched</strong><small>Requested / Resolved / Observed</small></span></div>{mismatches.length > 0 && <div className="has-warning"><AlertTriangle size={17} /><span><strong>{mismatches.length} mismatches</strong><small>差分を優先表示</small></span></div>}</div>
              <div className="studio-evidence-table" role="table"><div role="row" className="studio-evidence-head"><span>Parameter</span><span>Requested</span><span>Resolved</span><span>Observed</span><span>Source</span></div>{evidence?.parameters.map((parameter) => <div role="row" key={parameter.key} className={parameter.mismatch ? "has-mismatch" : ""}><strong>{parameter.key}</strong><span>{String(parameter.requested ?? "—")}</span><span>{String(parameter.resolved ?? "—")}</span><span>{String(parameter.observed ?? "—")}</span><small>{parameter.observed_source}</small></div>)}</div>
              <details className="studio-details"><summary><span><FileText size={14} /> Generated files and raw manifest</span><ChevronDown size={14} /></summary><dl><dt>Generated config</dt><dd>{evidence?.manifest?.generated_configs?.join(" · ") || "unknown"}</dd><dt>Dataset source</dt><dd>{String(evidence?.manifest?.dataset?.source ?? "unknown")}</dd><dt>Run manifest</dt><dd><pre>{JSON.stringify(evidence?.manifest ?? {}, null, 2)}</pre></dd></dl></details>
            </div>}
          </>}
        </section>
      </div>

      {selectedRun?.status === "completed" && (evidence?.checkpoints?.length ?? 0) > 0 && <div className="studio-next-action"><span><CheckCircle2 size={19} /><span><strong>学習結果をLibraryで管理できます</strong><small>採用条件を満たしたCheckpointだけをFinalにできます。</small></span></span><button onClick={onOpenLibrary}>Libraryを開く</button></div>}
      {evaluationConfirmation && <div className="studio-dialog-backdrop" role="presentation" onMouseDown={() => setEvaluationConfirmation(null)}>
        <section className="studio-modal factory-aesthetic-dialog" role="dialog" aria-modal="true" aria-labelledby="evaluation-confirm-title" onMouseDown={(event) => event.stopPropagation()}>
          <header><div><p className="studio-eyebrow">USER AESTHETIC DECISION</p><h2 id="evaluation-confirm-title">このPreviewをFinal候補として承認しますか？</h2></div><span className="factory-aesthetic-state">AESTHETIC_UNREVIEWED</span></header>
          <div className="factory-aesthetic-summary"><ShieldCheck size={24} /><div><strong>{evaluationConfirmation.evaluation.preview_slot} · Checkpoint #{evaluationConfirmation.evaluation.checkpoint_id}</strong><p>実Previewを確認したあなたの判断として保存します。自動指標やAgent評価では確定されません。</p></div></div>
          <p className="studio-modal-copy">この操作だけが <strong>USER_APPROVED</strong> を記録し、LibraryのFinal選択ゲートを開きます。</p>
          <footer><button className="studio-secondary-button" onClick={() => setEvaluationConfirmation(null)}>まだ確定しない</button><button className="studio-primary-button" onClick={() => void saveEvaluation(evaluationConfirmation.evaluation, evaluationConfirmation.profileSnapshotId, true)} disabled={savingEvaluation != null}>{savingEvaluation ? <Loader2 size={15} className="spin" /> : <Check size={15} />}Final候補として承認</button></footer>
        </section>
      </div>}
    </div>
  );
}
