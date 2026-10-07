import { useEffect, useState } from "react";
import { Check, FilePlus2, Layers3, LockKeyhole, RefreshCw, X } from "lucide-react";
import { API_BASE, apiGet, apiPost } from "../lib/api";
import type { Project } from "../types";

type Concept = { id: number; name: string; trigger_token: string; concept_type: string };
type Asset = { id: number; asset_key: string; file_path: string; review_status: string; user_rating?: number | null; training_enabled?: number | boolean; training_weight?: number };
type Snapshot = { id: number; name: string; item_count: number; snapshot_hash: string };
type Workspace = { concepts: Concept[]; assets: Asset[]; snapshots: Snapshot[] };
type DatasetItem = { id: number; file_path: string; caption: string; caption_source: string; selected: number; review_status: string; basepipe_asset_id?: number; training_input_source?: string; training_input?: string };
type DatasetAudit = { total: number; selected: number; captioned: number; caption_rate: number; missing_caption: number; missing_files: number; repairable_files: number; top_tags: { tag: string; count: number; ratio: number }[]; warnings: string[] };
type DistributionEntry = { label: string; count: number; pct: number };
type Distribution = { captioned_items: number; hair_color: DistributionEntry[]; hair_style: DistributionEntry[]; eye_color: DistributionEntry[]; costume: DistributionEntry[] };
type LeakReport = { risk_level: string; risk_score: number; warnings: string[] };
type CaptionBulkPreview = { operation_id: number; affected_count: number; changes: { asset_id: number; before: string; after: string }[] };
type AssetVersion = { id: number; version_kind: string; status: string; file_path: string; content_sha256: string };
type GenerationRun = { id: number; status: string; current_stage: string; stage_history: { stage: string; status: string }[]; output_manifest: Record<string, unknown> };
type GenerationFrame = { asset_id: number; file_path: string; ordinal: number; review_status: "needs_review" | "approved" | "rejected"; rejection_reason?: string };
type H3PreflightCheck = { label: string; ok: boolean; detail: string };
const generationStages = [
  ["planning", "計画"], ["h3_generation", "H3生成"], ["frame_extraction", "Frame抽出"], ["blur_filter", "Blur"],
  ["duplicate_filter", "重複除外"], ["identity_review", "Identity Review"], ["qwen_correction", "Qwen補正"],
  ["balancing", "Balance"], ["captioning", "Caption"], ["snapshot", "Snapshot"],
] as const;

export default function BasepipePanel({ project, showError, showNotice }: { project: Project; showError: (msg: string) => void; showNotice: (msg: string) => void }) {
  const [workspace, setWorkspace] = useState<Workspace>({ concepts: [], assets: [], snapshots: [] });
  const [datasetItems, setDatasetItems] = useState<DatasetItem[]>([]);
  const [audit, setAudit] = useState<DatasetAudit | null>(null);
  const [distribution, setDistribution] = useState<Distribution | null>(null);
  const [leakReport, setLeakReport] = useState<LeakReport | null>(null);
  const [selectedDatasetIds, setSelectedDatasetIds] = useState<number[]>([]);
  const [conceptName, setConceptName] = useState("");
  const [trigger, setTrigger] = useState("");
  const [conceptType, setConceptType] = useState<"identity" | "outfit" | "style" | "hybrid">(project.project_type === "style" ? "style" : "identity");
  const [assetPath, setAssetPath] = useState("");
  const [snapshotName, setSnapshotName] = useState("dataset-v001");
  const [captionDrafts, setCaptionDrafts] = useState<Record<number, string>>({});
  const [bulkFind, setBulkFind] = useState("");
  const [bulkReplace, setBulkReplace] = useState("");
  const [bulkPreview, setBulkPreview] = useState<CaptionBulkPreview | null>(null);
  const [busy, setBusy] = useState(false);

  async function refresh() {
    try {
      const [nextWorkspace, nextDataset] = await Promise.all([
        apiGet<Workspace>(`/basepipe/projects/${project.id}/workspace`),
        apiGet<{ items: DatasetItem[] }>(`/basepipe/projects/${project.id}/dataset-items`),
      ]);
      const nextAudit = await apiGet<DatasetAudit>(`/basepipe/projects/${project.id}/dataset-audit`);
      const nextDistribution = await apiGet<Distribution>(`/dataset/distribution/${project.id}`);
      const nextLeak = await apiGet<LeakReport>(`/dataset/character-leak/${project.id}`);
      setWorkspace(nextWorkspace); setDatasetItems(nextDataset.items); setAudit(nextAudit);
      setDistribution(nextDistribution); setLeakReport(nextLeak);
      setSelectedDatasetIds((current) => current.length ? current.filter((id) => nextDataset.items.some((item) => item.id === id)) : nextDataset.items.filter((item) => item.selected).map((item) => item.id));
    } catch (e) { showError(e instanceof Error ? e.message : "Basepipe状態の取得に失敗しました"); }
  }
  useEffect(() => { void refresh(); }, [project.id]);

  async function createConcept() {
    if (!conceptName.trim() || !trigger.trim()) return showError("Concept名とTriggerを入力してください");
    setBusy(true);
    try { await apiPost(`/basepipe/projects/${project.id}/concepts`, { name: conceptName.trim(), trigger_token: trigger.trim(), concept_type: conceptType }); setConceptName(""); setTrigger(""); await refresh(); showNotice("Conceptを作成しました"); }
    catch (e) { showError(e instanceof Error ? e.message : "Concept作成に失敗しました"); } finally { setBusy(false); }
  }
  async function registerAsset() {
    if (!assetPath.trim()) return showError("Assetファイルの絶対パスを入力してください");
    setBusy(true);
    try { await apiPost(`/basepipe/projects/${project.id}/assets`, { file_path: assetPath.trim(), asset_key: assetPath.trim().split(/[\\/]/).pop()?.split(".")[0] ?? "asset", origin_kind: "manual_review" }); setAssetPath(""); await refresh(); showNotice("AssetをSHA-256付きで登録しました"); }
    catch (e) { showError(e instanceof Error ? e.message : "Asset登録に失敗しました"); } finally { setBusy(false); }
  }
  async function sealSnapshot() {
    const concept = workspace.concepts[0];
    if (!concept) return showError("先にConceptとTriggerを保存してください");
    if (!selectedDatasetIds.length && !workspace.assets.length) return showError("Dataset画像またはAssetを選択してください");
    setBusy(true);
    try { await apiPost(`/basepipe/projects/${project.id}/snapshots`, { name: snapshotName.trim() || "dataset-v001", concept_id: concept?.id ?? null, dataset_item_ids: selectedDatasetIds, asset_ids: selectedDatasetIds.length ? [] : workspace.assets.map((asset) => asset.id) }); await refresh(); showNotice("Dataset Snapshotをsealedしました"); }
    catch (e) { showError(e instanceof Error ? e.message : "Snapshot作成に失敗しました"); } finally { setBusy(false); }
  }
  async function reviewAsset(asset: Asset, review_status: "approved" | "rejected") {
    setBusy(true);
    try { await apiPost(`/basepipe/assets/${asset.id}/review`, { review_status }, "PATCH"); await refresh(); showNotice(review_status === "approved" ? "Assetを承認しました" : "Assetを却下しました"); }
    catch (e) { showError(e instanceof Error ? e.message : "Review更新に失敗しました"); } finally { setBusy(false); }
  }
  async function updateAssetTraining(asset: Asset, values: { user_rating?: number | null; training_enabled?: boolean; training_weight?: number }) {
    setBusy(true);
    try {
      await apiPost(`/basepipe/assets/${asset.id}/review`, { review_status: asset.review_status || "pending", ...values }, "PATCH");
      await refresh();
      showNotice("Assetの評価・学習設定を保存しました");
    } catch (e) { showError(e instanceof Error ? e.message : "Asset設定の保存に失敗しました"); }
    finally { setBusy(false); }
  }
  async function reviewDatasetItem(item: DatasetItem, review_status: "approved" | "rejected") {
    setBusy(true);
    try { await apiPost(`/basepipe/projects/${project.id}/dataset-items/${item.id}/review`, { review_status }); await refresh(); showNotice(review_status === "approved" ? "Dataset Assetを承認しました" : "Dataset Assetを却下しました"); }
    catch (e) { showError(e instanceof Error ? e.message : "Dataset Review更新に失敗しました"); } finally { setBusy(false); }
  }
  async function saveDatasetCaption(item: DatasetItem) {
    const caption = (captionDrafts[item.id] ?? item.training_input ?? item.caption ?? "").trim();
    if (!caption) return showError("Captionを入力してください");
    setBusy(true);
    try { await apiPost(`/basepipe/projects/${project.id}/dataset-items/${item.id}/review`, { review_status: item.review_status || "pending", caption }); await refresh(); showNotice("Caption Lineageを保存しました（Training Inputを更新）"); }
    catch (e) { showError(e instanceof Error ? e.message : "Caption Lineageの保存に失敗しました"); } finally { setBusy(false); }
  }
  async function bulkReviewDatasetItems(review_status: "approved" | "rejected") {
    if (!selectedDatasetIds.length) return showError("一括レビューするDataset項目を選択してください");
    setBusy(true);
    try { const result = await apiPost<{ updated_count: number }>(`/basepipe/projects/${project.id}/dataset-items/bulk-review`, { item_ids: selectedDatasetIds, review_status }); await refresh(); showNotice(`${result.updated_count}件を${review_status === "approved" ? "承認" : "却下"}しました`); }
    catch (e) { showError(e instanceof Error ? e.message : "Dataset一括Reviewに失敗しました"); } finally { setBusy(false); }
  }
  async function repairDatasetPaths() {
    setBusy(true);
    try { const result = await apiPost<{ updated_count: number }>(`/basepipe/projects/${project.id}/repair-dataset-paths`, {}); await refresh(); showNotice(`Datasetパスを${result.updated_count}件再リンクしました`); }
    catch (e) { showError(e instanceof Error ? e.message : "Datasetパスの再リンクに失敗しました"); } finally { setBusy(false); }
  }
  async function previewBulkCaption() {
    if (!bulkFind.trim()) return showError("一括編集する検索語を入力してください");
    setBusy(true);
    try { const result = await apiPost<CaptionBulkPreview>(`/basepipe/projects/${project.id}/captions/bulk-preview`, { asset_ids: workspace.assets.map((asset) => asset.id), find: bulkFind, replace: bulkReplace }); setBulkPreview(result); showNotice(`${result.affected_count}件のCaption差分を作成しました`); }
    catch (e) { showError(e instanceof Error ? e.message : "Caption差分の作成に失敗しました"); } finally { setBusy(false); }
  }
  async function applyBulkCaption() {
    if (!bulkPreview || !bulkPreview.affected_count) return;
    setBusy(true);
    try { await apiPost(`/basepipe/projects/${project.id}/captions/bulk-apply`, { operation_id: bulkPreview.operation_id }); setBulkPreview(null); await refresh(); showNotice("Caption一括編集を適用しました"); }
    catch (e) { showError(e instanceof Error ? e.message : "Caption一括編集に失敗しました"); } finally { setBusy(false); }
  }
  async function undoBulkCaption() {
    if (!bulkPreview) return;
    setBusy(true);
    try { await apiPost(`/basepipe/projects/${project.id}/captions/undo`, { operation_id: bulkPreview.operation_id }); setBulkPreview(null); await refresh(); showNotice("Caption一括編集をUndoしました"); }
    catch (e) { showError(e instanceof Error ? e.message : "Caption一括編集のUndoに失敗しました"); } finally { setBusy(false); }
  }
  const selectedApproved = datasetItems.filter((item) => selectedDatasetIds.includes(item.id) && item.review_status === "approved").length;
  return <section className="mb-5 rounded-xl border border-indigo-900/60 bg-indigo-950/15 p-4">
    <AssetTransformPanel projectId={project.id} assets={workspace.assets} conceptId={workspace.concepts[0]?.id ?? null} showError={showError} showNotice={showNotice} />
    <AssetReviewControls assets={workspace.assets} busy={busy} onSave={updateAssetTraining} />
    <div className="flex items-center justify-between"><div><div className="flex items-center gap-2 text-indigo-200"><Layers3 size={16}/><h2 className="text-sm font-semibold">Basepipe foundation</h2></div><p className="mt-1 text-xs text-gray-500">Anima用のConcept → Asset → Dataset Snapshotを明示的に固定します</p></div><button onClick={() => void refresh()} className="rounded-lg border border-gray-700 p-2 text-gray-400 hover:bg-gray-800" title="更新"><RefreshCw size={14}/></button></div>
    {audit && <div className="mt-3 rounded-lg border border-amber-900/50 bg-amber-950/20 p-2 text-[11px] text-amber-200"><div className="flex flex-wrap gap-x-3 gap-y-1"><span>監査: {audit.selected}件選択</span><span>caption {Math.round(audit.caption_rate * 100)}%</span><span>未設定 {audit.missing_caption}</span><span>未検出 {audit.missing_files}</span></div>{audit.top_tags.length > 0 && <p className="mt-1 truncate text-amber-300">頻出タグ: {audit.top_tags.slice(0, 4).map((tag) => `${tag.tag} ${Math.round(tag.ratio * 100)}%`).join(" · ")}</p>}{audit.warnings.length > 0 && <p className="mt-1 text-amber-400">{audit.warnings.join(" / ")}</p>}{audit.repairable_files > 0 && <button disabled={busy} onClick={() => void repairDatasetPaths()} className="mt-2 rounded border border-amber-700/60 px-2 py-1 text-amber-200 hover:bg-amber-900/30 disabled:opacity-50">候補 {audit.repairable_files}件を再リンク</button>}</div>}
    <div className="mt-3 grid gap-2 md:grid-cols-2"><div className="rounded-lg border border-gray-800 bg-gray-950/50 p-2 text-[11px] text-gray-400"><p className="font-semibold text-indigo-200">{project.project_type === "style" ? "Style Curator" : "Character Factory"}</p><p className="mt-1 text-gray-500">{project.project_type === "style" ? "採否は人間が決定。類似・キャラクター偏りは警告のみ。" : "Identity / Outfit / Flexibilityの分布を確認してからSnapshot化。"}</p>{distribution && <p className="mt-1 text-gray-500">Caption済み {distribution.captioned_items}件 · Costumeタグ {distribution.costume.length}種</p>}</div>{leakReport && <div className="rounded-lg border border-gray-800 bg-gray-950/50 p-2 text-[11px] text-gray-400"><p className="font-semibold text-amber-200">Character Bias Dashboard</p><p className="mt-1">risk: <span className={leakReport.risk_level === "high" ? "text-rose-300" : "text-amber-300"}>{leakReport.risk_level}</span> · score {leakReport.risk_score}</p>{leakReport.warnings.slice(0, 2).map((warning) => <p key={warning} className="mt-1 truncate text-amber-400">{warning}</p>)}</div>}</div>
    <div className="mt-4 grid gap-3 lg:grid-cols-3">
      <div className="rounded-lg border border-gray-800 bg-gray-950/60 p-3"><p className="text-[11px] font-semibold text-gray-400">1. Concept</p><div className="mt-2 grid gap-2"><input value={conceptName} onChange={(e) => setConceptName(e.target.value)} placeholder="例: Main Character" className="rounded border border-gray-700 bg-gray-900 px-2 py-1.5 text-xs text-gray-200"/><select value={conceptType} onChange={(e) => setConceptType(e.target.value as typeof conceptType)} className="rounded border border-gray-700 bg-gray-900 px-2 py-1.5 text-xs text-gray-200"><option value="identity">Identity</option><option value="outfit">Outfit</option><option value="style">Style</option><option value="hybrid">Hybrid</option></select><input value={trigger} onChange={(e) => setTrigger(e.target.value)} placeholder="例: char_main" className="rounded border border-gray-700 bg-gray-900 px-2 py-1.5 text-xs text-gray-200"/><button disabled={busy} onClick={() => void createConcept()} className="rounded bg-indigo-600 px-2 py-1.5 text-xs text-white disabled:opacity-50">Conceptを保存</button></div><p className="mt-2 text-[11px] text-gray-600">登録済み: {workspace.concepts.length}</p></div>
      <div className="rounded-lg border border-gray-800 bg-gray-950/60 p-3"><p className="text-[11px] font-semibold text-gray-400">2. Asset / Review</p><div className="mt-2 flex gap-2"><FilePlus2 size={15} className="mt-1 text-gray-500"/><input value={assetPath} onChange={(e) => setAssetPath(e.target.value)} placeholder="画像の絶対パス" className="min-w-0 flex-1 rounded border border-gray-700 bg-gray-900 px-2 py-1.5 text-xs text-gray-200"/></div><button disabled={busy} onClick={() => void registerAsset()} className="mt-2 w-full rounded border border-gray-700 px-2 py-1.5 text-xs text-gray-300 hover:bg-gray-800 disabled:opacity-50">SHA-256付きで登録</button><div className="mt-2 flex items-center justify-between text-[11px] text-gray-600"><span>Dataset SSOT: {datasetItems.length}件</span><button onClick={() => setSelectedDatasetIds(datasetItems.filter((item) => item.selected).map((item) => item.id))} className="text-indigo-300">選択を復元</button></div>
        <div className="mt-2 rounded border border-indigo-900/50 bg-indigo-950/20 p-2"><div className="flex items-center justify-between text-[11px]"><span className="text-indigo-200">選択 {selectedDatasetIds.length}件 / 承認済み {selectedApproved}件</span><span className="text-gray-500">一括操作</span></div><div className="mt-2 grid grid-cols-2 gap-2"><button disabled={busy || !selectedDatasetIds.length} onClick={() => void bulkReviewDatasetItems("approved")} className="rounded bg-emerald-800 px-2 py-1.5 text-[11px] text-emerald-100 hover:bg-emerald-700 disabled:opacity-50">選択を一括承認</button><button disabled={busy || !selectedDatasetIds.length} onClick={() => void bulkReviewDatasetItems("rejected")} className="rounded border border-rose-900 px-2 py-1.5 text-[11px] text-rose-200 hover:bg-rose-950 disabled:opacity-50">選択を一括却下</button></div></div>
        <div className="mt-2 max-h-40 space-y-1 overflow-auto">{datasetItems.slice(0, 100).map((item) => <div key={item.id} className="rounded border border-gray-900 px-1 py-1 text-[10px] text-gray-500"><div className="flex items-center gap-1"><label className="flex min-w-0 flex-1 items-center gap-2 truncate"><input type="checkbox" checked={selectedDatasetIds.includes(item.id)} onChange={(e) => setSelectedDatasetIds((ids) => e.target.checked ? [...ids, item.id] : ids.filter((id) => id !== item.id))}/><span className="truncate">{item.file_path.split(/[\\/]/).pop()}</span><span className="shrink-0 text-gray-700">{item.review_status}{item.training_input_source ? ` · ${item.training_input_source}` : ""}</span></label><button disabled={busy} title="承認" onClick={() => void reviewDatasetItem(item, "approved")} className="rounded p-1 text-emerald-400 hover:bg-emerald-950"><Check size={11}/></button><button disabled={busy} title="却下" onClick={() => void reviewDatasetItem(item, "rejected")} className="rounded p-1 text-rose-400 hover:bg-rose-950"><X size={11}/></button></div><div className="mt-1 flex gap-1"><input aria-label={`Training Input ${item.id}`} value={captionDrafts[item.id] ?? item.training_input ?? item.caption ?? ""} onChange={(e) => setCaptionDrafts((drafts) => ({ ...drafts, [item.id]: e.target.value }))} placeholder="Original / Edited → Training Input" className="min-w-0 flex-1 rounded border border-gray-800 bg-gray-950 px-1.5 py-1 text-[10px] text-gray-300"/><button disabled={busy} onClick={() => void saveDatasetCaption(item)} className="rounded border border-indigo-900 px-1.5 py-1 text-[10px] text-indigo-300 hover:bg-indigo-950 disabled:opacity-50">保存</button></div></div>)}</div>{workspace.assets.length > 0 && <div className="mt-2 border-t border-gray-800 pt-2"><p className="mb-1 text-[10px] text-gray-500">登録Asset Review</p>{workspace.assets.slice(0, 5).map((asset) => <div key={asset.id} className="flex items-center gap-1 text-[10px] text-gray-500"><span className="min-w-0 flex-1 truncate">{asset.asset_key} · {asset.review_status}</span><button disabled={busy} onClick={() => void reviewAsset(asset, "approved")} className="rounded p-1 text-emerald-400 hover:bg-emerald-950"><Check size={12}/></button><button disabled={busy} onClick={() => void reviewAsset(asset, "rejected")} className="rounded p-1 text-rose-400 hover:bg-rose-950"><X size={12}/></button></div>)}</div>}<div className="mt-2 border-t border-gray-800 pt-2"><p className="text-[10px] font-semibold text-indigo-200">Caption Bulk Diff / Undo</p><div className="mt-1 flex gap-1"><input value={bulkFind} onChange={(e) => setBulkFind(e.target.value)} placeholder="検索タグ" className="min-w-0 flex-1 rounded border border-gray-800 bg-gray-950 px-1.5 py-1 text-[10px] text-gray-300"/><input value={bulkReplace} onChange={(e) => setBulkReplace(e.target.value)} placeholder="置換後" className="min-w-0 flex-1 rounded border border-gray-800 bg-gray-950 px-1.5 py-1 text-[10px] text-gray-300"/><button disabled={busy || !workspace.assets.length} onClick={() => void previewBulkCaption()} className="rounded border border-indigo-800 px-1.5 py-1 text-[10px] text-indigo-300 disabled:opacity-50">Diff</button></div>{bulkPreview && <div className="mt-1 rounded border border-amber-900/60 bg-amber-950/20 p-1.5 text-[10px] text-amber-200"><p>{bulkPreview.affected_count}件に影響</p>{bulkPreview.changes.slice(0, 2).map((change) => <p key={change.asset_id} className="truncate">#{change.asset_id}: {change.before} → {change.after}</p>)}<div className="mt-1 flex gap-1"><button disabled={busy || !bulkPreview.affected_count} onClick={() => void applyBulkCaption()} className="rounded bg-emerald-800 px-1.5 py-1 text-[10px]">Apply</button><button disabled={busy} onClick={() => void undoBulkCaption()} className="rounded border border-gray-700 px-1.5 py-1 text-[10px]">Undo</button></div></div>}</div><p className="mt-1 text-[11px] text-gray-600">Snapshot対象: {selectedDatasetIds.length}件（approvedのみ）</p></div>
      <div className="rounded-lg border border-gray-800 bg-gray-950/60 p-3"><p className="flex items-center gap-1 text-[11px] font-semibold text-gray-400"><LockKeyhole size={13}/>3. Immutable Snapshot</p><input value={snapshotName} onChange={(e) => setSnapshotName(e.target.value)} className="mt-2 w-full rounded border border-gray-700 bg-gray-900 px-2 py-1.5 text-xs text-gray-200"/><button disabled={busy} onClick={() => void sealSnapshot()} className="mt-2 w-full rounded bg-emerald-700 px-2 py-1.5 text-xs text-white hover:bg-emerald-600 disabled:opacity-50">Dataset Snapshotをsealed</button><p className="mt-2 text-[11px] text-gray-600">sealed: {workspace.snapshots.length}</p></div>
    </div>
  </section>;
}

function AssetReviewControls({ assets, busy, onSave }: { assets: Asset[]; busy: boolean; onSave: (asset: Asset, values: { user_rating?: number | null; training_enabled?: boolean; training_weight?: number }) => void }) {
  if (!assets.length) return null;
  return <div className="mb-3 rounded-lg border border-gray-800 bg-gray-950/50 p-2"><p className="text-[11px] font-semibold text-indigo-200">Asset quality / training controls</p><p className="mt-1 text-[10px] text-gray-500">評価・学習投入・重みは独立して保存されます。5つ星は自動的に重みへ変換しません。</p><div className="mt-2 grid gap-1">{assets.slice(0, 20).map((asset) => <div key={asset.id} className="flex flex-wrap items-center gap-2 rounded border border-gray-900 px-1.5 py-1 text-[10px] text-gray-400"><span className="min-w-0 flex-1 truncate">{asset.asset_key} · {asset.review_status}</span><label>評価<select disabled={busy} aria-label={`Asset rating ${asset.id}`} defaultValue={asset.user_rating ?? ""} onChange={(e) => onSave(asset, { user_rating: e.target.value ? Number(e.target.value) : null })} className="ml-1 rounded border border-gray-800 bg-gray-950 px-1 py-0.5"><option value="">—</option>{[1, 2, 3, 4, 5].map((value) => <option key={value} value={value}>{value}/5</option>)}</select></label><label>重み<input disabled={busy} aria-label={`Training weight ${asset.id}`} type="number" min="0.1" max="10" step="0.1" defaultValue={asset.training_weight ?? 1} onBlur={(e) => { const value = Number(e.currentTarget.value); if (Number.isFinite(value) && value > 0) onSave(asset, { training_weight: value }); }} className="ml-1 w-12 rounded border border-gray-800 bg-gray-950 px-1 py-0.5"/></label><label className="flex items-center gap-1"><input disabled={busy} type="checkbox" defaultChecked={asset.training_enabled !== false && asset.training_enabled !== 0} onChange={(e) => onSave(asset, { training_enabled: e.target.checked })}/>学習投入</label></div>)}</div></div>;
}

function AssetTransformPanel({ projectId, assets, conceptId, showError, showNotice }: { projectId: number; assets: Asset[]; conceptId: number | null; showError: (message: string) => void; showNotice: (message: string) => void }) {
  const [busyId, setBusyId] = useState<number | null>(null);
  const [versionCounts, setVersionCounts] = useState<Record<number, number>>({});
  const [comfyuiReady, setComfyuiReady] = useState(false);
  const [h3Status, setH3Status] = useState<string>("");
  const [h3Checks, setH3Checks] = useState<H3PreflightCheck[]>([]);
  const [h3PreflightReady, setH3PreflightReady] = useState(false);
  const [h3Executing, setH3Executing] = useState(false);
  const [h3RunId, setH3RunId] = useState<number | null>(null);
  const [h3Run, setH3Run] = useState<GenerationRun | null>(null);
  useEffect(() => {
    void apiGet<{ services?: Record<string, { state: string }> }>("/system/status")
      .then((status) => setComfyuiReady(status.services?.comfyui?.state === "ready"))
      .catch(() => setComfyuiReady(false));
  }, []);
  async function prepare(asset: Asset, kind: "qwen-edit" | "neutralize") {
    setBusyId(asset.id);
    try {
      const result = await apiPost<{ status: string }>(`/basepipe/projects/${projectId}/assets/${asset.id}/${kind}`, { execute: false, gpu_device_id: 1 });
      const versions = await apiGet<{ versions: AssetVersion[] }>(`/basepipe/assets/${asset.id}/versions`);
      setVersionCounts((current) => ({ ...current, [asset.id]: versions.versions.length }));
      showNotice(`${kind === "neutralize" ? "Neutralize" : "Qwen Edit"}の実行準備を確認しました（${result.status}）`);
    } catch (e) { showError(e instanceof Error ? e.message : "Asset変換の準備確認に失敗しました"); }
    finally { setBusyId(null); }
  }
  async function executeTransform(asset: Asset, kind: "qwen-edit" | "neutralize") {
    if (!comfyuiReady) return showError("ComfyUIがofflineのためQwen処理を開始できません");
    setBusyId(asset.id);
    try {
      const result = await apiPost<{ status: string }>(`/basepipe/projects/${projectId}/assets/${asset.id}/${kind}`, { execute: true, gpu_device_id: 1, seed: 42 });
      const versions = await apiGet<{ versions: AssetVersion[] }>(`/basepipe/assets/${asset.id}/versions`);
      setVersionCounts((current) => ({ ...current, [asset.id]: versions.versions.length }));
      showNotice(`${kind === "neutralize" ? "Neutralize" : "Qwen Edit"}を実行しました（${result.status}）`);
    } catch (e) { showError(e instanceof Error ? e.message : "Qwen処理の実行に失敗しました"); }
    finally { setBusyId(null); }
  }
  async function preflightH3() {
    try { const result = await apiPost<{ status: string; message: string; checks: H3PreflightCheck[] }>(`/basepipe/projects/${projectId}/character-generation/preflight`, { asset_ids: assets.slice(0, 2).map((asset) => asset.id), mode: "training", gpu_device_id: 1 }); setH3Status(result.status); setH3PreflightReady(result.status === "ready"); setH3Checks(result.checks ?? []); showNotice(`H3 Preflight: ${result.status}（生成は開始していません）`); }
    catch (e) { showError(e instanceof Error ? e.message : "H3 Preflightに失敗しました"); }
  }
  async function prepareH3Run() {
    try {
      const result = await apiPost<{ run_id: number; status: string }>(`/basepipe/projects/${projectId}/character-generation/start`, { asset_ids: assets.slice(0, 1).map((asset) => asset.id), mode: "variation", prompt: "Anima character reference, neutral pose, clean simple background", seed: 42, width: 768, height: 512, length: 56, ref_image_size: "match", gpu_device_id: 1, execute: false });
      setH3RunId(result.run_id);
      setH3Status(result.status);
      setH3Run(await apiGet<GenerationRun>(`/basepipe/projects/${projectId}/character-generation/runs/${result.run_id}`));
      showNotice(`H3 Run #${result.run_id}を準備しました（GPUジョブ未開始）`);
    } catch (e) { showError(e instanceof Error ? e.message : "H3 Run準備に失敗しました"); }
  }
  async function executeH3Run() {
    if (!h3PreflightReady) return showError("H3 PreflightがPASSするまで実行できません");
    setH3Executing(true);
    try {
      const result = await apiPost<{ run_id: number; status: string }>(`/basepipe/projects/${projectId}/character-generation/start`, { asset_ids: assets.slice(0, 1).map((asset) => asset.id), mode: "variation", prompt: "Anima character reference, neutral pose, clean simple background", seed: 42, width: 768, height: 512, length: 56, ref_image_size: "match", gpu_device_id: 1, execute: true });
      setH3RunId(result.run_id);
      setH3Status(result.status);
      setH3Run(await apiGet<GenerationRun>(`/basepipe/projects/${projectId}/character-generation/runs/${result.run_id}`));
      showNotice(`H3 Run #${result.run_id}をGPU1へ投入しました。Runを監視しています。`);
    } catch (e) { showError(e instanceof Error ? e.message : "H3実行に失敗しました"); }
    finally { setH3Executing(false); }
  }
  useEffect(() => {
    if (!h3RunId || !h3Run || !["queued", "running"].includes(h3Run.status)) return;
    const timer = setInterval(() => {
      void apiGet<GenerationRun>(`/basepipe/projects/${projectId}/character-generation/runs/${h3RunId}`)
        .then(setH3Run)
        .catch(() => undefined);
    }, 2000);
    return () => clearInterval(timer);
  }, [h3RunId, h3Run?.status, projectId]);
  if (!assets.length) return null;
  const currentIndex = h3Run ? generationStages.findIndex(([stage]) => stage === h3Run.current_stage) : -1;
  return <div className="mb-3 rounded-lg border border-amber-900/50 bg-amber-950/15 p-2"><div className="flex items-center justify-between"><p className="text-[11px] font-semibold text-amber-200">Character Asset Versions</p><span className="text-[10px] text-gray-600">元画像非破壊 · GPU1固定</span></div><div className="mt-2 flex flex-wrap items-center gap-2"><button onClick={() => void preflightH3()} className="rounded border border-cyan-800 px-1.5 py-1 text-[10px] text-cyan-200">H3 Generation Preflight</button><button onClick={() => void prepareH3Run()} className="rounded border border-emerald-800 px-1.5 py-1 text-[10px] text-emerald-200">H3 R2V Run準備</button><button disabled={!h3PreflightReady || h3Executing} onClick={() => void executeH3Run()} className="rounded bg-rose-800 px-1.5 py-1 text-[10px] text-rose-100 disabled:cursor-not-allowed disabled:opacity-40">{h3Executing ? "H3投入中…" : "H3 R2V実行（GPU1）"}</button>{h3Status && <span className={h3Status === "ready" || h3Status === "prepared" ? "text-[10px] text-emerald-300" : "text-[10px] text-amber-300"}>status: {h3Status}{h3RunId ? ` · Run #${h3RunId}` : ""}</span>}</div><div className="mt-1 text-[10px] text-gray-600">実GPUジョブ送信はPreflight PASS後、この実行ボタンを押した時だけ行います。</div>{h3Checks.length > 0 && <div className="mt-2 grid gap-1 sm:grid-cols-2">{h3Checks.map((check) => <div key={check.label} className={`rounded border px-1.5 py-1 text-[10px] ${check.ok ? "border-emerald-900/60 text-emerald-300" : "border-rose-900/60 text-rose-300"}`}><span>{check.ok ? "✓" : "!"} {check.label}</span><p className="mt-0.5 text-gray-500">{check.detail}</p></div>)}</div>}{h3Run && <div className="mt-2 rounded border border-cyan-900/50 bg-cyan-950/15 p-2"><div className="flex items-center justify-between text-[10px] text-cyan-200"><span>Dataset Generation Run #{h3Run.id}</span><span>{h3Run.status} · current: {generationStages[currentIndex]?.[1] ?? h3Run.current_stage}</span></div><div className="mt-2 grid grid-cols-5 gap-1">{generationStages.map(([stage, label], index) => <div key={stage} title={stage} className={`rounded px-1 py-1 text-center text-[9px] ${index < currentIndex ? "bg-emerald-900/50 text-emerald-200" : index === currentIndex ? "bg-cyan-800/70 text-cyan-100" : "bg-gray-900 text-gray-600"}`}>{label}</div>)}</div><p className="mt-1 text-[9px] text-gray-500">H3出力はFrame抽出・Blur/重複除外・Identity Reviewを通過するまで学習入力になりません。</p></div>}<div className="mt-2 space-y-1">{assets.slice(0, 5).map((asset) => <div key={asset.id} className="flex items-center gap-1 text-[10px] text-gray-500"><span className="min-w-0 flex-1 truncate">{asset.asset_key} · {asset.review_status}{versionCounts[asset.id] ? ` · versions ${versionCounts[asset.id]}` : ""}</span><button disabled={busyId === asset.id} onClick={() => void prepare(asset, "neutralize")} className="rounded border border-amber-800 px-1.5 py-1 text-amber-200 disabled:opacity-50">Neutralize準備</button><button disabled={busyId === asset.id} onClick={() => void prepare(asset, "qwen-edit")} className="rounded border border-indigo-800 px-1.5 py-1 text-indigo-200 disabled:opacity-50">Qwen Edit準備</button><button disabled={!comfyuiReady || busyId === asset.id} onClick={() => void executeTransform(asset, "neutralize")} className="rounded bg-amber-800 px-1.5 py-1 text-amber-100 disabled:cursor-not-allowed disabled:opacity-40">Neutralize実行</button><button disabled={!comfyuiReady || busyId === asset.id} onClick={() => void executeTransform(asset, "qwen-edit")} className="rounded bg-indigo-800 px-1.5 py-1 text-indigo-100 disabled:cursor-not-allowed disabled:opacity-40">Qwen実行</button></div>)}</div><CharacterGenerationReviewPanel projectId={projectId} conceptId={conceptId} run={h3Run} comfyuiReady={comfyuiReady} onRunUpdated={setH3Run} showError={showError} showNotice={showNotice} /></div>;
}

function CharacterGenerationReviewPanel({ projectId, conceptId, run, comfyuiReady, onRunUpdated, showError, showNotice }: { projectId: number; conceptId: number | null; run: GenerationRun | null; comfyuiReady: boolean; onRunUpdated: (run: GenerationRun) => void; showError: (message: string) => void; showNotice: (message: string) => void }) {
  if (!run) return null;
  return <CharacterGenerationReviewPanelContent projectId={projectId} conceptId={conceptId} run={run} comfyuiReady={comfyuiReady} onRunUpdated={onRunUpdated} showError={showError} showNotice={showNotice} />;
}

function CharacterGenerationReviewPanelContent({ projectId, conceptId, run, comfyuiReady, onRunUpdated, showError, showNotice }: { projectId: number; conceptId: number | null; run: GenerationRun; comfyuiReady: boolean; onRunUpdated: (run: GenerationRun) => void; showError: (message: string) => void; showNotice: (message: string) => void }) {
  const manifest = run.output_manifest.frame_extraction as { frames?: GenerationFrame[] } | undefined;
  const frames = manifest?.frames ?? [];
  const [decisions, setDecisions] = useState<Record<number, "approved" | "rejected">>({});
  const [captions, setCaptions] = useState<Record<number, string>>({});
  const [busy, setBusy] = useState(false);
  useEffect(() => setDecisions(Object.fromEntries(frames.flatMap((frame) =>
    frame.review_status === "approved" || frame.review_status === "rejected"
      ? [[frame.asset_id, frame.review_status as "approved" | "rejected"]]
      : [],
  ))), [run!.id, frames.length]);
  async function refreshRun() { onRunUpdated(await apiGet<GenerationRun>(`/basepipe/projects/${projectId}/character-generation/runs/${run.id}`)); }
  async function extractFrames() { setBusy(true); try { await apiPost(`/basepipe/projects/${projectId}/character-generation/runs/${run.id}/extract-frames`, {}); await refreshRun(); showNotice("H3出力からFrameを抽出しました。Identity Review待ちです"); } catch (e) { showError(e instanceof Error ? e.message : "Frame抽出に失敗しました"); } finally { setBusy(false); } }
  async function reviewFrames() { if (!frames.length) return; if (frames.some((frame) => !decisions[frame.asset_id])) return showError("全Frameを明示的に承認または却下してください"); if (!confirm("表示中の全Frameを確認し、あなたの採否判断として保存しますか？")) return; setBusy(true); try { await apiPost(`/basepipe/projects/${projectId}/character-generation/runs/${run.id}/identity-review`, { decisions: frames.map((frame) => ({ asset_id: frame.asset_id, review_status: decisions[frame.asset_id] })), human_confirmed: true, note: "User confirmed Frame review in Basepipe Panel" }); await refreshRun(); showNotice("Identity Reviewの判定を保存しました"); } catch (e) { showError(e instanceof Error ? e.message : "Identity Reviewに失敗しました"); } finally { setBusy(false); } }
  async function qwen(execute: boolean) { const approved = frames.filter((frame) => decisions[frame.asset_id] === "approved").map((frame) => frame.asset_id); if (!approved.length) return showError("Qwen対象のapproved Frameがありません"); setBusy(true); try { await apiPost(`/basepipe/projects/${projectId}/character-generation/runs/${run.id}/qwen-correction`, { asset_ids: approved, execute, gpu_device_id: 1, prompt: "preserve identity and correct image quality", seed: 42 }); await refreshRun(); showNotice(execute ? "Qwen補正をGPU1へ投入しました" : "Qwen補正計画を保存しました"); } catch (e) { showError(e instanceof Error ? e.message : "Qwen補正に失敗しました"); } finally { setBusy(false); } }
  async function balance() { setBusy(true); try { await apiPost(`/basepipe/projects/${projectId}/character-generation/runs/${run.id}/balance`, {}); await refreshRun(); showNotice("Balancing evidenceを保存しました"); } catch (e) { showError(e instanceof Error ? e.message : "Balancingに失敗しました"); } finally { setBusy(false); } }
  async function caption() { const approved = frames.filter((frame) => decisions[frame.asset_id] === "approved"); const missing = approved.filter((frame) => !captions[frame.asset_id]?.trim()); if (missing.length) return showError("approved FrameのCaptionをすべて入力してください"); setBusy(true); try { await apiPost(`/basepipe/projects/${projectId}/character-generation/runs/${run.id}/captioning`, { captions: approved.map((frame) => ({ asset_id: frame.asset_id, caption: captions[frame.asset_id].trim() })) }); await refreshRun(); showNotice("Captioningを保存しました"); } catch (e) { showError(e instanceof Error ? e.message : "Captioningに失敗しました"); } finally { setBusy(false); } }
  async function sealGenerationSnapshot() { const approved = frames.filter((frame) => decisions[frame.asset_id] === "approved").map((frame) => frame.asset_id); if (!approved.length) return showError("Snapshot対象のapproved Frameがありません"); setBusy(true); try { await apiPost(`/basepipe/projects/${projectId}/snapshots`, { name: `generation-run-${run.id}`, concept_id: conceptId, asset_ids: approved, dataset_item_ids: [] }); await refreshRun(); showNotice("承認済みFrameだけでDataset Snapshotをsealedしました"); } catch (e) { showError(e instanceof Error ? e.message : "Snapshot封印に失敗しました"); } finally { setBusy(false); } }
  return <div className="mt-2 rounded border border-fuchsia-900/50 bg-fuchsia-950/15 p-2"><div className="flex items-center justify-between"><p className="text-[10px] font-semibold text-fuchsia-200">Human Review / Qwen Lineage</p><span className="text-[9px] text-gray-500">Run #{run.id} · GPU1</span></div><div className="mt-2 flex flex-wrap gap-1"><button disabled={busy || run.current_stage !== "h3_generation"} onClick={() => void extractFrames()} className="rounded border border-cyan-800 px-1.5 py-1 text-[10px] text-cyan-200 disabled:opacity-40">Frame抽出</button><button disabled={busy || run.current_stage !== "duplicate_filter" || !frames.length} onClick={() => void reviewFrames()} className="rounded border border-emerald-800 px-1.5 py-1 text-[10px] text-emerald-200 disabled:opacity-40">Reviewを保存</button><button disabled={busy || run.current_stage !== "identity_review"} onClick={() => void qwen(false)} className="rounded border border-indigo-800 px-1.5 py-1 text-[10px] text-indigo-200 disabled:opacity-40">Qwen計画</button><button disabled={busy || !comfyuiReady || run.current_stage !== "identity_review"} onClick={() => void qwen(true)} className="rounded bg-indigo-800 px-1.5 py-1 text-[10px] text-indigo-100 disabled:opacity-40">Qwen実行（GPU1）</button><button disabled={busy || run.current_stage !== "identity_review"} onClick={() => void balance()} className="rounded border border-amber-800 px-1.5 py-1 text-[10px] text-amber-200 disabled:opacity-40">Balance完了</button><button disabled={busy || run.current_stage !== "balancing"} onClick={() => void caption()} className="rounded border border-cyan-800 px-1.5 py-1 text-[10px] text-cyan-200 disabled:opacity-40">Caption完了</button><button disabled={busy || run.current_stage !== "captioning"} onClick={() => void sealGenerationSnapshot()} className="rounded bg-emerald-800 px-1.5 py-1 text-[10px] text-emerald-100 disabled:opacity-40">Run Snapshot封印</button></div>{frames.length > 0 && <div className="mt-2 grid grid-cols-2 gap-2 sm:grid-cols-4">{frames.map((frame) => <div key={frame.asset_id} className="rounded border border-gray-800 bg-gray-950/60 p-1"><img src={`${API_BASE}/collector/thumbnail?path=${encodeURIComponent(frame.file_path)}&size=120`} alt={`H3 frame ${frame.ordinal}`} className="aspect-square w-full rounded object-cover"/><p className="mt-1 truncate text-[9px] text-gray-500">Frame {frame.ordinal} · Asset #{frame.asset_id}</p><select value={decisions[frame.asset_id] ?? ""} onChange={(e) => setDecisions((current) => ({ ...current, [frame.asset_id]: e.target.value as "approved" | "rejected" }))} className="mt-1 w-full rounded border border-gray-800 bg-gray-900 px-1 py-0.5 text-[9px] text-gray-300"><option value="" disabled>未選択</option><option value="approved">承認</option><option value="rejected">却下</option></select>{decisions[frame.asset_id] === "approved" && <input aria-label={`Caption Frame ${frame.ordinal}`} value={captions[frame.asset_id] ?? ""} onChange={(e) => setCaptions((current) => ({ ...current, [frame.asset_id]: e.target.value }))} placeholder="Training Caption" className="mt-1 w-full rounded border border-gray-800 bg-gray-900 px-1 py-0.5 text-[9px] text-gray-300"/>}</div>)}</div>}</div>;
}
