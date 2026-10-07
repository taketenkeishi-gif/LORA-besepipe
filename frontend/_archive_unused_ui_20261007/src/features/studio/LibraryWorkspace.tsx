import { useEffect, useState } from "react";
import { Archive, Check, CheckCircle2, Download, FileCheck2, Fingerprint, FolderOpen, Loader2, RefreshCw, ShieldCheck } from "lucide-react";
import { apiGet, apiPost } from "../../lib/api";
import type { LoraAsset, PreviewSample, Project } from "../../types";

type Props = { project: Project; showError: (message: string) => void; showNotice: (message: string) => void; onOpenRun: (runId: number) => void };
type GateKey = "run_completed" | "snapshot_sealed" | "checkpoint_present" | "source_artifact_match" | "preview_complete" | "preview_conditions_match" | "human_evaluation" | "final_selected" | "exported";
const gateLabels: Record<GateKey, string> = {
  run_completed: "Training completed", snapshot_sealed: "Dataset sealed", checkpoint_present: "Checkpoint present",
  source_artifact_match: "Evaluated artifact matches",
  preview_complete: "Preview complete", preview_conditions_match: "Preview conditions match", human_evaluation: "Human review accepted",
  final_selected: "Selected as final", exported: "Exported",
};

export default function LibraryWorkspace({ project, showError, showNotice, onOpenRun }: Props) {
  const [assets, setAssets] = useState<LoraAsset[]>([]);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [action, setAction] = useState<"final" | "export" | null>(null);
  const [finalConfirmationOpen, setFinalConfirmationOpen] = useState(false);
  const [previewByCheckpoint, setPreviewByCheckpoint] = useState<Record<number, string>>({});

  const selected = assets.find((asset) => asset.id === selectedId) ?? null;
  const acceptedCount = assets.filter((asset) => Boolean(asset.lineage?.lineage_gate?.final_selected)).length;
  const exportedCount = assets.filter((asset) => Boolean(asset.lineage?.export_path)).length;
  const gates = selected?.lineage?.lineage_gate ?? {};
  const gateEntries = Object.entries(gateLabels) as Array<[GateKey, string]>;
  const finalPrerequisites = gateEntries.filter(([key]) => key !== "final_selected" && key !== "exported");
  const canSelectFinal = finalPrerequisites.every(([key]) => Boolean(gates[key]));
  const sourceRunId = selected?.lineage?.source_run_id ?? selected?.training_run_id ?? null;
  const sourceCheckpointId = selected?.lineage?.source_checkpoint_id ?? null;
  const selectedPreview = sourceCheckpointId == null ? null : previewByCheckpoint[sourceCheckpointId] ?? null;

  async function load() {
    setLoading(true);
    try {
      const [next, previews] = await Promise.all([
        apiGet<LoraAsset[]>(`/library/assets?project_id=${project.id}`),
        apiGet<{ timeline: PreviewSample[] }>(`/previews/${project.id}?include_images=false`).catch(() => ({ timeline: [] })),
      ]);
      setAssets(next);
      setPreviewByCheckpoint(Object.fromEntries((previews.timeline ?? []).flatMap((sample) => {
        const image = Object.values(sample.sample_previews ?? sample.samples ?? {}).find((value) => typeof value === "string");
        return image ? [[sample.checkpoint_id, image]] : [];
      })));
      setSelectedId((current) => current && next.some((asset) => asset.id === current) ? current : next[0]?.id ?? null);
    } catch (error) {
      showError(error instanceof Error ? error.message : "Libraryの取得に失敗しました");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { void load(); }, [project.id]);

  function selectFinal() {
    if (!selected || !canSelectFinal) return;
    setFinalConfirmationOpen(true);
  }

  async function confirmSelectFinal() {
    if (!selected || !canSelectFinal) return;
    setAction("final");
    try {
      await apiPost(`/library/assets/${selected.id}/select-final`, {
        human_confirmed: true,
        note: "User confirmed Final selection in Library Workspace",
      });
      setFinalConfirmationOpen(false);
      showNotice(`「${selected.name}」をFinalに選択しました`);
      await load();
    } catch (error) {
      showError(error instanceof Error ? error.message : "Final選択に失敗しました");
    } finally { setAction(null); }
  }

  async function exportAsset() {
    if (!selected) return;
    setAction("export");
    try {
      const result = await apiPost<LoraAsset & { lineage?: { export_path?: string } }>("/library/export", { asset_id: selected.id });
      showNotice(`Exportしました: ${result.lineage?.export_path ?? result.lora_path}`);
      await load();
    } catch (error) {
      showError(error instanceof Error ? error.message : "Exportに失敗しました");
    } finally { setAction(null); }
  }

  return (
    <div className="studio-workspace is-library">
      <header className="studio-page-header"><div><p className="studio-eyebrow">04 · LIBRARY</p><h1>完成したLoRAを管理する</h1><p>Version、由来、評価、Export先をひとつの記録として保持します。</p></div><button className="studio-secondary-button" onClick={() => void load()} disabled={loading}><RefreshCw size={15} className={loading ? "spin" : ""} /> 更新</button></header>

      <section className="studio-library-summary"><div><span>Versions</span><strong>{assets.length}</strong></div><div><span>Final candidates</span><strong>{acceptedCount}</strong></div><div><span>Exported</span><strong>{exportedCount}</strong></div><div><span>Source project</span><strong>{project.name}</strong></div></section>

      <div className="studio-library-layout">
        <section className="studio-model-list">
          <div className="studio-section-heading"><div><p className="studio-eyebrow">MODEL VERSIONS</p><h2>{project.name}</h2></div><p>学習Runから自動登録された実Assetです。</p></div>
          {assets.length ? <div className="studio-version-list">{assets.map((asset) => <button key={asset.id} className={selectedId === asset.id ? "is-active" : ""} onClick={() => setSelectedId(asset.id)}>
            <span className="studio-version-icon"><Archive size={20} /></span>
            <span className="studio-version-copy"><small>VERSION {asset.id}</small><strong>{asset.name}</strong><span>{asset.lineage?.model_family ?? (asset.base_model || "model unknown")} · {asset.dataset_size} images</span><i>Source Run #{asset.training_run_id ?? "unknown"}</i></span>
            <span className={`studio-version-status is-${asset.library_status ?? "candidate"}`}>{asset.library_status === "accepted" ? <Check size={12} /> : null}{asset.library_status ?? "candidate"}</span>
          </button>)}</div> : <div className="studio-empty-state"><Archive size={28} /><h3>Library Assetはありません</h3><p>学習が完了してCheckpointが登録されると、ここにVersionが追加されます。</p></div>}
        </section>

        <aside className="studio-library-inspector">
          {!selected ? <div className="studio-empty-inspector"><Archive size={24} /><p>Versionを選択してください。</p></div> : <>
            <header><div><p className="studio-eyebrow">VERSION DETAILS</p><h2>{selected.name}</h2><span>Asset #{selected.id} · created {new Date(selected.created_at).toLocaleDateString("ja-JP")}</span></div><span className={`studio-state-badge is-${selected.library_status}`}>{selected.library_status ?? "candidate"}</span></header>
            {selectedPreview
              ? <div className="studio-library-cover has-image"><img src={selectedPreview} alt={`${selected.name} selected checkpoint preview`} /><span>Checkpoint #{sourceCheckpointId} · 実Preview</span></div>
              : <div className="studio-library-cover is-empty"><FileCheck2 size={28} /><span>{selected.lineage?.preview_success_count ? `${selected.lineage.preview_success_count} previews · 画像取得待ち` : "Preview fileなし"}</span></div>}

            <section className="studio-gate-section"><div className="studio-card-heading"><div><p className="studio-eyebrow">LINEAGE GATE</p><h3>完成条件</h3></div><span>{gateEntries.filter(([key]) => Boolean(gates[key])).length}/{gateEntries.length}</span></div><div className="studio-gate-grid">{gateEntries.map(([key, label]) => <div key={key} className={gates[key] ? "is-pass" : "is-pending"}>{gates[key] ? <CheckCircle2 size={14} /> : <span />}{label}</div>)}</div></section>

            <div className="studio-library-actions">
              <button className="studio-secondary-button" disabled={sourceRunId == null} onClick={() => { if (sourceRunId != null) onOpenRun(sourceRunId); }}><FolderOpen size={15} /> Source Run #{sourceRunId ?? "unknown"}を開く</button>
              {!gates.final_selected && <button className="studio-primary-button" disabled={!canSelectFinal || action != null} onClick={selectFinal}>{action === "final" ? <Loader2 size={15} className="spin" /> : <ShieldCheck size={15} />} Select as Final</button>}
              <button className="studio-primary-button" disabled={!gates.final_selected || action != null} onClick={() => void exportAsset()}>{action === "export" ? <Loader2 size={15} className="spin" /> : <Download size={15} />} Export</button>
            </div>
            {!canSelectFinal && !gates.final_selected && <p className="studio-action-note">Finalにするには、完了Run・固定Snapshot・Preview条件一致・ユーザー承認済みHuman reviewが必要です。</p>}

            <details className="studio-details" open><summary><span><Fingerprint size={14} /> Provenance</span></summary><dl><dt>Source Run</dt><dd>#{selected.lineage?.source_run_id ?? selected.training_run_id ?? "unknown"}</dd><dt>Dataset Snapshot</dt><dd>#{selected.lineage?.source_snapshot_id ?? selected.dataset_snapshot_id ?? "unknown"} · {selected.lineage?.source_snapshot_hash?.slice(0, 12) ?? "hash unknown"}</dd><dt>Checkpoint</dt><dd>#{selected.lineage?.source_checkpoint_id ?? "unknown"} · {selected.lineage?.checkpoint_validation_status ?? "validation unknown"}</dd><dt>Preview Profile</dt><dd>#{selected.lineage?.preview_profile_snapshot_id ?? "unknown"} · {selected.lineage?.preview_profile_snapshot_hash?.slice(0, 12) ?? "hash unknown"}</dd><dt>Trigger</dt><dd>{selected.lineage?.trigger_token ?? "unknown"}</dd><dt>Export path</dt><dd>{selected.lineage?.export_path ?? "not exported"}</dd></dl></details>
          </>}
        </aside>
      </div>
      {finalConfirmationOpen && selected && <div className="studio-dialog-backdrop" role="presentation" onMouseDown={() => setFinalConfirmationOpen(false)}>
        <section className="studio-modal factory-aesthetic-dialog" role="dialog" aria-modal="true" aria-labelledby="final-selection-title" onMouseDown={(event) => event.stopPropagation()}>
          <header><div><p className="studio-eyebrow">USER AESTHETIC DECISION</p><h2 id="final-selection-title">このCheckpointをFinalに選択しますか？</h2></div><span className="factory-aesthetic-state">AESTHETIC_UNREVIEWED</span></header>
          <div className="factory-aesthetic-summary"><ShieldCheck size={24} /><div><strong>{selected.name}</strong><p>Checkpoint #{sourceCheckpointId ?? "unknown"}と固定条件Previewを確認したあなたの最終判断として記録します。</p></div></div>
          <p className="studio-modal-copy">この操作だけが <strong>USER_APPROVED</strong> を記録します。Exportはこの確認後まで無効です。</p>
          <footer><button className="studio-secondary-button" onClick={() => setFinalConfirmationOpen(false)}>まだ確定しない</button><button className="studio-primary-button" onClick={() => void confirmSelectFinal()} disabled={action != null}>{action === "final" ? <Loader2 size={15} className="spin" /> : <Check size={15} />}Finalに選択</button></footer>
        </section>
      </div>}
    </div>
  );
}
