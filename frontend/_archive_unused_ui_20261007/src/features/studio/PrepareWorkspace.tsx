import { useEffect, useMemo, useRef, useState } from "react";
import type { KeyboardEvent as ReactKeyboardEvent, MouseEvent as ReactMouseEvent } from "react";
import {
  AlertTriangle,
  Check,
  CheckCircle2,
  Database,
  Filter,
  Image as ImageIcon,
  Loader2,
  RefreshCw,
  Save,
  Search,
  ShieldCheck,
  SlidersHorizontal,
  Undo2,
} from "lucide-react";
import { API_BASE, apiGet, apiPost } from "../../lib/api";
import type { DatasetStats, Project } from "../../types";
import CharacterDatasetFactory from "./CharacterDatasetFactory";
import DatasetFiles from "./DatasetFiles";
import { TabNav } from "@radix-ui/themes";

type Asset = {
  id: number;
  asset_key: string;
  file_path: string;
  asset_type?: string;
  origin_kind?: string;
  source_ref?: string;
  caption_source?: string;
  content_sha256?: string;
  caption_original?: string;
  caption_edited?: string;
  caption_processed?: string;
  training_input?: string;
  training_input_source?: string;
  review_status?: string;
  user_rating?: number | null;
  training_enabled?: boolean | number;
  training_weight?: number;
  group_ids?: number[];
};
type Snapshot = { id: number; name: string; item_count: number; snapshot_hash: string; status?: string; created_at?: string };
type Workspace = { assets?: Asset[]; snapshots?: Snapshot[]; groups?: Array<{ id: number; name: string; asset_count?: number }> };
type CaptionBulkPreview = { operation_id: number; affected_count: number; changes: Array<{ asset_id: number; before: string; after: string }> };
type FilterId = "all" | "issues" | "approved" | "excluded";

type Props = {
  project: Project;
  showError: (message: string) => void;
  showNotice: (message: string) => void;
  onOpenTrain: () => void;
};

function sourceCaption(asset: Asset) {
  return asset.caption_edited || asset.training_input || asset.caption_processed || asset.caption_original || "";
}

function snapshotBlockReason(asset: Asset) {
  if (asset.review_status !== "approved") return "承認が必要です";
  if (!Boolean(asset.training_enabled)) return "学習対象に含めてください";
  if (asset.asset_type && asset.asset_type !== "image") return "画像AssetだけをSnapshot化できます";
  if (!sourceCaption(asset).trim()) return "学習用キャプションが必要です";
  if (asset.origin_kind === "h3_frame" && asset.caption_source !== "h3_captioning") {
    return "Dataset FactoryでQwen選択・Balance・Caption確定を完了してください";
  }
  if (asset.origin_kind === "h3_frame" && !asset.training_input_source) {
    return "Dataset FactoryでOriginalまたはQwen版をTraining Inputに選んでください";
  }
  return null;
}

export default function PrepareWorkspace(props: Props) {
  const modeFromUrl = (): "files" | "factory" | "dataset" => {
    const view = new URLSearchParams(window.location.search).get("view");
    return view === "review" ? "dataset" : view === "factory" && props.project.project_type === "character" ? "factory" : "files";
  };
  const [mode, setMode] = useState(modeFromUrl);
  function selectMode(next: "files" | "factory" | "dataset") {
    setMode(next);
    const url = new URL(window.location.href);
    url.searchParams.set("view", next === "dataset" ? "review" : next);
    window.history.replaceState(window.history.state, "", `${url.pathname}${url.search}${url.hash}`);
  }
  useEffect(() => { setMode(modeFromUrl()); }, [props.project.id]);
  return <div className="studio-prepare-switcher">
    <TabNav.Root aria-label="データセットの作業">
      <TabNav.Link active={mode === "files"} href="?view=files" onClick={e=>{e.preventDefault();selectMode("files");}}>データセット</TabNav.Link>
      <TabNav.Link active={mode === "dataset"} href="?view=review" onClick={e=>{e.preventDefault();selectMode("dataset");}}>採用・学習セット</TabNav.Link>
      {props.project.project_type === "character" && <TabNav.Link active={mode === "factory"} href="?view=factory" onClick={e=>{e.preventDefault();selectMode("factory");}}>素材生成</TabNav.Link>}
    </TabNav.Root>
    {mode === "files" ? <DatasetFiles key={props.project.id} project={props.project} showError={props.showError} showNotice={props.showNotice} onReview={() => selectMode("dataset")} /> : mode === "factory"
      ? <CharacterDatasetFactory project={props.project} showError={props.showError} showNotice={props.showNotice} onOpenDataset={() => selectMode("dataset")} onOpenTrain={props.onOpenTrain} />
      : <DatasetReviewWorkspace {...props} />}
  </div>;
}

function DatasetReviewWorkspace({ project, showError, showNotice, onOpenTrain }: Props) {
  const [workspace, setWorkspace] = useState<Workspace | null>(null);
  const [stats, setStats] = useState<DatasetStats | null>(null);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [selectedIds, setSelectedIds] = useState<number[]>([]);
  const [selectionAnchorId, setSelectionAnchorId] = useState<number | null>(null);
  const [filter, setFilter] = useState<FilterId>("all");
  const [query, setQuery] = useState("");
  const [caption, setCaption] = useState("");
  const [saving, setSaving] = useState(false);
  const [loading, setLoading] = useState(false);
  const [snapshotOpen, setSnapshotOpen] = useState(false);
  const [snapshotName, setSnapshotName] = useState("");
  const [creatingSnapshot, setCreatingSnapshot] = useState(false);
  const [bulkCaptionOpen, setBulkCaptionOpen] = useState(false);
  const [bulkFind, setBulkFind] = useState("");
  const [bulkReplace, setBulkReplace] = useState("");
  const [bulkPreview, setBulkPreview] = useState<CaptionBulkPreview | null>(null);
  const [bulkStatus, setBulkStatus] = useState<"preview" | "applied">("preview");
  const [bulkSaving, setBulkSaving] = useState(false);
  const [assetMenu, setAssetMenu] = useState<{ assetId: number; x: number; y: number } | null>(null);
  const assetMenuRef = useRef<HTMLDivElement | null>(null);
  const assetMenuTriggerRef = useRef<HTMLButtonElement | null>(null);
  const snapshotDialogRef = useRef<HTMLElement | null>(null);
  const snapshotTriggerRef = useRef<HTMLButtonElement | null>(null);
  const snapshotWasOpenRef = useRef(false);
  const bulkDialogRef = useRef<HTMLElement | null>(null);
  const bulkTriggerRef = useRef<HTMLButtonElement | null>(null);
  const bulkWasOpenRef = useRef(false);

  const assets = workspace?.assets ?? [];
  const selected = assets.find((asset) => asset.id === selectedId) ?? null;
  const snapshots = workspace?.snapshots ?? [];
  const approved = assets.filter((asset) => snapshotBlockReason(asset) === null);
  const snapshotBlockers = assets
    .map((asset) => ({ asset, reason: snapshotBlockReason(asset) }))
    .filter((item): item is { asset: Asset; reason: string } => Boolean(item.reason));
  const actionableSnapshotBlockers = snapshotBlockers.filter(({ asset }) => asset.review_status === "approved" && Boolean(asset.training_enabled));
  const issueCount = snapshotBlockers.length;
  const captionCoverage = assets.length ? Math.round((assets.filter((asset) => Boolean(sourceCaption(asset))).length / assets.length) * 100) : 0;
  const qualityScore = stats?.quality_score ?? Math.max(0, Math.round((captionCoverage + (assets.length ? approved.length / assets.length * 100 : 0)) / 2));

  const filteredAssets = useMemo(() => assets.filter((asset) => {
    if (filter === "issues" && snapshotBlockReason(asset) === null) return false;
    if (filter === "approved" && asset.review_status !== "approved") return false;
    if (filter === "excluded" && Boolean(asset.training_enabled)) return false;
    const needle = query.trim().toLowerCase();
    return !needle || `${asset.asset_key} ${sourceCaption(asset)}`.toLowerCase().includes(needle);
  }), [assets, filter, query]);

  async function load(preserveSelection = true) {
    setLoading(true);
    try {
      const [nextWorkspace, nextStats] = await Promise.all([
        apiGet<Workspace>(`/basepipe/projects/${project.id}/workspace`),
        apiGet<DatasetStats>(`/dataset/stats/${project.id}`).catch(() => null),
      ]);
      setWorkspace(nextWorkspace);
      setStats(nextStats);
      const nextAssets = nextWorkspace.assets ?? [];
      if (!preserveSelection || selectedId == null || !nextAssets.some((asset) => asset.id === selectedId)) {
        setSelectedId(nextAssets[0]?.id ?? null);
        setSelectedIds(nextAssets[0] ? [nextAssets[0].id] : []);
        setSelectionAnchorId(nextAssets[0]?.id ?? null);
        setCaption(nextAssets[0] ? sourceCaption(nextAssets[0]) : "");
      } else {
        const nextSelected = nextAssets.find((asset) => asset.id === selectedId);
        setSelectedIds((current) => {
          const preserved = current.filter((id) => nextAssets.some((asset) => asset.id === id));
          return preserved.length ? preserved : nextSelected ? [nextSelected.id] : [];
        });
        if (nextSelected) setCaption(sourceCaption(nextSelected));
      }
    } catch (error) {
      showError(error instanceof Error ? error.message : "Datasetの取得に失敗しました");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { void load(false); }, [project.id]);

  useEffect(() => {
    setBulkCaptionOpen(false);
    setBulkPreview(null);
    setBulkStatus("preview");
  }, [project.id]);

  useEffect(() => {
    if (snapshotOpen) {
      snapshotWasOpenRef.current = true;
      return;
    }
    if (snapshotWasOpenRef.current) {
      snapshotWasOpenRef.current = false;
      window.requestAnimationFrame(() => snapshotTriggerRef.current?.focus());
    }
  }, [snapshotOpen]);

  useEffect(() => {
    if (bulkCaptionOpen) {
      bulkWasOpenRef.current = true;
      return;
    }
    if (bulkWasOpenRef.current) {
      bulkWasOpenRef.current = false;
      window.requestAnimationFrame(() => bulkTriggerRef.current?.focus());
    }
  }, [bulkCaptionOpen]);

  useEffect(() => {
    if (!assetMenu) return;
    const focusFrame = window.requestAnimationFrame(() => {
      assetMenuRef.current?.querySelector<HTMLButtonElement>('[role="menuitem"]:not(:disabled)')?.focus();
    });
    const close = (event: PointerEvent) => {
      if (!assetMenuRef.current?.contains(event.target as Node)) setAssetMenu(null);
    };
    const closeOnKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setAssetMenu(null);
        assetMenuTriggerRef.current?.focus();
      }
    };
    window.addEventListener("pointerdown", close);
    window.addEventListener("keydown", closeOnKey);
    return () => {
      window.cancelAnimationFrame(focusFrame);
      window.removeEventListener("pointerdown", close);
      window.removeEventListener("keydown", closeOnKey);
    };
  }, [assetMenu]);

  function selectAsset(asset: Asset, event?: ReactMouseEvent<HTMLButtonElement>) {
    const additive = Boolean(event?.ctrlKey || event?.metaKey);
    const anchorIndex = filteredAssets.findIndex((candidate) => candidate.id === selectionAnchorId);
    const assetIndex = filteredAssets.findIndex((candidate) => candidate.id === asset.id);
    let nextIds: number[];
    if (event?.shiftKey && anchorIndex >= 0 && assetIndex >= 0) {
      const range = filteredAssets.slice(Math.min(anchorIndex, assetIndex), Math.max(anchorIndex, assetIndex) + 1).map((candidate) => candidate.id);
      nextIds = additive ? Array.from(new Set([...selectedIds, ...range])) : range;
    } else if (additive) {
      nextIds = selectedIds.includes(asset.id) && selectedIds.length > 1
        ? selectedIds.filter((id) => id !== asset.id)
        : Array.from(new Set([...selectedIds, asset.id]));
      setSelectionAnchorId(asset.id);
    } else {
      nextIds = [asset.id];
      setSelectionAnchorId(asset.id);
    }
    const primary = nextIds.includes(asset.id) ? asset : assets.find((candidate) => candidate.id === nextIds[nextIds.length - 1]) ?? asset;
    setSelectedIds(nextIds);
    setSelectedId(primary.id);
    setCaption(sourceCaption(primary));
  }

  async function saveAssetTarget(target: Asset, patch: Partial<Asset> = {}, captionValue = sourceCaption(target)) {
    setSaving(true);
    try {
      const updated = await apiPost<Asset>(`/basepipe/assets/${target.id}/review`, {
        review_status: patch.review_status ?? target.review_status ?? "pending",
        caption: patch.caption_edited ?? captionValue,
        user_rating: patch.user_rating ?? target.user_rating ?? null,
        training_enabled: patch.training_enabled ?? target.training_enabled ?? true,
        training_weight: patch.training_weight ?? target.training_weight ?? 1,
      }, "PATCH");
      setWorkspace((current) => current ? { ...current, assets: current.assets?.map((asset) => asset.id === updated.id ? updated : asset) } : current);
      if (selectedId === updated.id) setCaption(sourceCaption(updated));
      showNotice(`Asset #${updated.id} を保存しました`);
    } catch (error) {
      showError(error instanceof Error ? error.message : "Assetの保存に失敗しました");
    } finally {
      setSaving(false);
    }
  }

  async function saveAsset(patch: Partial<Asset> = {}) {
    if (!selected) return;
    await saveAssetTarget(selected, patch, caption);
  }

  function openAssetMenu(asset: Asset, x: number, y: number, trigger: HTMLButtonElement) {
    if (!selectedIds.includes(asset.id)) {
      setSelectedIds([asset.id]);
      setSelectionAnchorId(asset.id);
    }
    setSelectedId(asset.id);
    setCaption(sourceCaption(asset));
    assetMenuTriggerRef.current = trigger;
    setAssetMenu({
      assetId: asset.id,
      x: Math.min(x, window.innerWidth - 230),
      y: Math.min(y, window.innerHeight - 190),
    });
  }

  async function runAssetMenuAction(patch: { review_status: string; training_enabled?: boolean }) {
    if (!assetMenu) return;
    const targetIds = selectedIds.includes(assetMenu.assetId) ? selectedIds : [assetMenu.assetId];
    setAssetMenu(null);
    setSaving(true);
    try {
      const result = await apiPost<{ updated_count: number; assets: Asset[] }>(`/basepipe/projects/${project.id}/assets/bulk-review`, {
        asset_ids: targetIds,
        review_status: patch.review_status,
        training_enabled: patch.training_enabled,
      });
      const updatedById = new Map(result.assets.map((asset) => [asset.id, asset]));
      setWorkspace((current) => current ? { ...current, assets: current.assets?.map((asset) => updatedById.get(asset.id) ?? asset) } : current);
      showNotice(`${result.updated_count}件のAssetを保存しました`);
    } catch (error) {
      showError(error instanceof Error ? error.message : "Assetの一括保存に失敗しました");
    } finally {
      setSaving(false);
    }
  }

  function openBulkCaption() {
    if (!selectedIds.length) return;
    setBulkFind("");
    setBulkReplace("");
    setBulkPreview(null);
    setBulkStatus("preview");
    setBulkCaptionOpen(true);
  }

  async function previewBulkCaption() {
    if (!bulkFind) return showError("置換する検索文字列を入力してください");
    setBulkSaving(true);
    try {
      const result = await apiPost<CaptionBulkPreview>(`/basepipe/projects/${project.id}/captions/bulk-preview`, {
        asset_ids: selectedIds,
        find: bulkFind,
        replace: bulkReplace,
      });
      setBulkPreview(result);
      setBulkStatus("preview");
      showNotice(`${result.affected_count}件のCaption差分を作成しました`);
    } catch (error) {
      showError(error instanceof Error ? error.message : "Caption差分の作成に失敗しました");
    } finally {
      setBulkSaving(false);
    }
  }

  async function applyBulkCaption() {
    if (!bulkPreview || bulkPreview.affected_count === 0) return;
    setBulkSaving(true);
    try {
      await apiPost(`/basepipe/projects/${project.id}/captions/bulk-apply`, { operation_id: bulkPreview.operation_id });
      setBulkStatus("applied");
      await load();
      showNotice(`${bulkPreview.affected_count}件のCaption変更を適用しました`);
    } catch (error) {
      showError(error instanceof Error ? error.message : "Caption一括編集の適用に失敗しました");
    } finally {
      setBulkSaving(false);
    }
  }

  async function undoBulkCaption() {
    if (!bulkPreview || bulkStatus !== "applied") return;
    setBulkSaving(true);
    try {
      await apiPost(`/basepipe/projects/${project.id}/captions/undo`, { operation_id: bulkPreview.operation_id });
      await load();
      setBulkCaptionOpen(false);
      setBulkPreview(null);
      showNotice(`${bulkPreview.affected_count}件のCaption変更を元に戻しました`);
    } catch (error) {
      showError(error instanceof Error ? error.message : "Caption一括編集のUndoに失敗しました");
    } finally {
      setBulkSaving(false);
    }
  }

  function navigateAssetMenu(event: ReactKeyboardEvent<HTMLDivElement>) {
    if (!["ArrowDown", "ArrowUp", "Home", "End"].includes(event.key)) return;
    const items = Array.from(assetMenuRef.current?.querySelectorAll<HTMLButtonElement>('[role="menuitem"]:not(:disabled)') ?? []);
    if (!items.length) return;
    event.preventDefault();
    const current = items.indexOf(document.activeElement as HTMLButtonElement);
    const next = event.key === "Home" ? 0
      : event.key === "End" ? items.length - 1
      : event.key === "ArrowDown" ? (current + 1 + items.length) % items.length
      : (current - 1 + items.length) % items.length;
    items[next]?.focus();
  }

  function trapDialogFocus(
    event: ReactKeyboardEvent<HTMLElement>,
    dialog: HTMLElement | null,
    canClose: boolean,
    close: () => void,
  ) {
    if (event.key === "Escape") {
      event.preventDefault();
      event.stopPropagation();
      if (canClose) close();
      return;
    }
    if (event.key !== "Tab" || !dialog) return;
    const focusable = Array.from(dialog.querySelectorAll<HTMLElement>(
      'input:not(:disabled), textarea:not(:disabled), select:not(:disabled), button:not(:disabled), a[href], [tabindex]:not([tabindex="-1"])',
    )).filter((element) => element.getClientRects().length > 0);
    if (!focusable.length) return;
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (event.shiftKey && (document.activeElement === first || !dialog.contains(document.activeElement))) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  }

  async function createSnapshot() {
    if (!snapshotName.trim() || approved.length === 0) return;
    setCreatingSnapshot(true);
    try {
      const result = await apiPost<Snapshot>(`/basepipe/projects/${project.id}/snapshots`, {
        name: snapshotName.trim(),
        asset_ids: approved.map((asset) => asset.id),
      });
      showNotice(`Snapshot #${result.id} を${result.item_count}件で固定しました`);
      setSnapshotOpen(false);
      setSnapshotName("");
      await load();
    } catch (error) {
      showError(error instanceof Error ? error.message : "Snapshot作成に失敗しました");
    } finally {
      setCreatingSnapshot(false);
    }
  }

  const filters: Array<{ id: FilterId; label: string; count: number }> = [
    { id: "all", label: "すべて", count: assets.length },
    { id: "issues", label: "要確認", count: issueCount },
    { id: "approved", label: "承認済み", count: assets.filter((asset) => asset.review_status === "approved").length },
    { id: "excluded", label: "学習対象外", count: assets.filter((asset) => !Boolean(asset.training_enabled)).length },
  ];

  return (
    <div className="studio-workspace">
      <header className="studio-page-header">
        <div><p className="studio-eyebrow">01 · PREPARE</p><h1>学習素材を整える</h1><p>採用する画像を承認し、キャプションと画像を学習用セットとして保存します。</p></div>
        <div className="studio-header-actions">
          <button className="studio-secondary-button" onClick={() => void load()} disabled={loading}><RefreshCw size={15} className={loading ? "spin" : ""} /> 更新</button>
          <button ref={snapshotTriggerRef} className="studio-primary-button" title={approved.length === 0 && actionableSnapshotBlockers[0] ? `Asset #${actionableSnapshotBlockers[0].asset.id}: ${actionableSnapshotBlockers[0].reason}` : undefined} onClick={() => { setSnapshotName(`snapshot-${snapshots.length + 1}`); setSnapshotOpen(true); }} disabled={approved.length === 0} aria-haspopup="dialog" aria-expanded={snapshotOpen}><Database size={16} /> 学習用セットを保存</button>
        </div>
      </header>

      <section className="studio-quality-strip" aria-label="Dataset品質">
        <div className="studio-quality-score"><span>準備スコア</span><strong>{qualityScore}</strong><small>/ 100</small></div>
        <div className="studio-quality-metric"><span>キャプション記入率</span><strong>{captionCoverage}%</strong><i style={{ width: `${captionCoverage}%` }} /></div>
        <div className="studio-quality-metric"><span>学習への採用数</span><strong>{approved.length} / {assets.length}</strong><i style={{ width: `${assets.length ? approved.length / assets.length * 100 : 0}%` }} /></div>
        <button className={issueCount ? "has-issues" : "is-ready"} onClick={() => setFilter(issueCount ? "issues" : "approved")}>
          {issueCount ? <AlertTriangle size={17} /> : <ShieldCheck size={17} />}<span><strong>{issueCount ? `${issueCount}件を確認` : "学習可能"}</strong><small>{issueCount ? "クリックして絞り込み" : "承認済み素材を表示"}</small></span>
        </button>
      </section>

      {approved.length === 0 && actionableSnapshotBlockers.length > 0 && <section className="studio-snapshot-blocker" role="status"><AlertTriangle size={17} /><span><strong>Snapshot前にFactory工程が必要です</strong><small>{actionableSnapshotBlockers.slice(0, 2).map(({ asset, reason }) => `Asset #${asset.id}: ${reason}`).join(" / ")}</small></span></section>}

      <div className="studio-prepare-layout">
        <aside className="studio-filter-pane">
          <p className="studio-pane-label"><Filter size={13} /> 絞り込み</p>
          {filters.map((item) => <button key={item.id} className={filter === item.id ? "is-active" : ""} onClick={() => setFilter(item.id)}><span>{item.label}</span><strong>{item.count}</strong></button>)}
          {snapshots.length > 0 && <div className="studio-snapshot-list"><p className="studio-pane-label"><Database size={13} /> 保存済み学習セット</p>{snapshots.slice(0, 4).map((snapshot) => <div key={snapshot.id}><span>#{snapshot.id} · {snapshot.name}</span><small>{snapshot.item_count} items · {snapshot.snapshot_hash?.slice(0, 8)}</small></div>)}</div>}
        </aside>

        <section className="studio-asset-pane">
          <div className="studio-pane-toolbar">
            <label><Search size={15} /><input aria-label="Assetを検索" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="AssetやCaptionを検索" /></label>
            <div className="studio-pane-toolbar-actions"><span title="Ctrl/⌘クリックで追加選択、Shiftクリックで範囲選択">{selectedIds.length > 1 ? `${selectedIds.length} selected · ` : ""}{filteredAssets.length} assets</span><button ref={bulkTriggerRef} className="studio-secondary-button" disabled={!selectedIds.length || assets.some(a => selectedIds.includes(a.id) && a.origin_kind === "dataset_file")} title="実ファイルの一括編集は画像・キャプション画面で行えます" onClick={openBulkCaption} aria-haspopup="dialog" aria-expanded={bulkCaptionOpen}>Caption一括編集</button></div>
          </div>
          {filteredAssets.length > 0 ? (
            <div className="studio-asset-grid" role="list" aria-label="Dataset assets">
              {filteredAssets.map((asset) => {
                const active = selectedIds.includes(asset.id);
                const hasIssue = snapshotBlockReason(asset) !== null;
                return (
                  <button
                    key={asset.id}
                    role="listitem"
                    className={`studio-asset-card ${active ? "is-selected" : ""}`}
                    onClick={(event) => selectAsset(asset, event)}
                    onContextMenu={(event) => {
                      event.preventDefault();
                      openAssetMenu(asset, event.clientX, event.clientY, event.currentTarget);
                    }}
                    onKeyDown={(event) => {
                      if (event.key === "ContextMenu" || (event.shiftKey && event.key === "F10")) {
                        event.preventDefault();
                        const rect = event.currentTarget.getBoundingClientRect();
                        openAssetMenu(asset, rect.left + Math.min(rect.width - 12, 44), rect.top + 42, event.currentTarget);
                      }
                    }}
                    aria-haspopup="menu"
                    aria-selected={active}
                  >
                    <img src={`${API_BASE}/collector/thumbnail?path=${encodeURIComponent(asset.file_path)}&size=420`} alt={asset.asset_key} loading="lazy" />
                    <span className="studio-asset-state">{hasIssue ? <AlertTriangle size={13} /> : <Check size={13} />}</span>
                    <span className="studio-asset-caption">{sourceCaption(asset) || "Captionなし"}</span>
                  </button>
                );
              })}
            </div>
          ) : (
            <div className="studio-empty-state"><ImageIcon size={28} /><h3>{assets.length ? "条件に合う素材はありません" : "Datasetに素材がありません"}</h3><p>{assets.length ? "フィルターまたは検索語を変更してください。" : "存在しないサンプル画像は表示しません。"}</p></div>
          )}
        </section>

        <aside className="studio-inspector" aria-label="Asset inspector">
          {!selected ? <div className="studio-empty-inspector"><SlidersHorizontal size={22} /><p>Assetを選ぶと詳細を編集できます。</p></div> : <>
            <header><div><p className="studio-eyebrow">ASSET INSPECTOR</p><h2>{selected.asset_key}</h2></div><span>#{selected.id}</span></header>
            <img className="studio-inspector-image" src={`${API_BASE}/collector/thumbnail?path=${encodeURIComponent(selected.file_path)}&size=720`} alt={selected.asset_key} />
            <div className="studio-segmented" aria-label="レビュー状態">
              {(["pending", "approved", "rejected"] as const).map((status) => <button key={status} className={selected.review_status === status ? "is-active" : ""} onClick={() => void saveAsset({ review_status: status })}>{status === "pending" ? "未確認" : status === "approved" ? "承認" : "除外"}</button>)}
            </div>
            <label className="studio-field"><span>学習用キャプション</span><textarea value={caption} onChange={(event) => setCaption(event.target.value)} rows={6} /></label>
            <div className="studio-lineage"><p className="studio-pane-label">CAPTION LINEAGE</p><div><span>Original</span><p>{selected.caption_original || "データなし"}</p></div><div><span>Edited</span><p>{selected.caption_edited || "未編集"}</p></div><div><span>Training input</span><p>{selected.training_input || selected.caption_original || "データなし"}</p></div></div>
            <label className="studio-switch-row"><span><strong>学習に含める</strong><small>Snapshot候補に使用</small></span><input type="checkbox" checked={Boolean(selected.training_enabled)} onChange={(event) => void saveAsset({ training_enabled: event.target.checked })} /></label>
            <label className="studio-field"><span>学習時の重み</span><input type="number" min="0" max="10" step="0.1" value={selected.training_weight ?? 1} onChange={(event) => setWorkspace((current) => current ? { ...current, assets: current.assets?.map((asset) => asset.id === selected.id ? { ...asset, training_weight: Number(event.target.value) } : asset) } : current)} /></label>
            <button className="studio-primary-button is-full" onClick={() => void saveAsset()} disabled={saving}>{saving ? <Loader2 size={16} className="spin" /> : <Save size={16} />}{saving ? "保存中…" : "変更を保存"}</button>
            <details className="studio-details"><summary>Provenance</summary><dl><dt>SHA-256</dt><dd>{selected.content_sha256 || "unknown"}</dd><dt>Path</dt><dd>{selected.file_path}</dd><dt>Source</dt><dd>{selected.training_input_source || "unknown"}</dd></dl></details>
          </>}
        </aside>
      </div>

      {assetMenu && <div ref={assetMenuRef} className="factory-context-menu" style={{ left: assetMenu.x, top: assetMenu.y }} role="menu" aria-label="Dataset Asset actions" onPointerDown={(event) => event.stopPropagation()} onKeyDown={navigateAssetMenu}>
        <p>{selectedIds.length > 1 ? `${selectedIds.length} DATASET ASSETS` : `DATASET ASSET · #${assetMenu.assetId}`}</p>
        <button role="menuitem" disabled={saving} onClick={() => void runAssetMenuAction({ review_status: "approved", training_enabled: true })}><span>承認して学習に含める</span><small>Snapshot候補</small></button>
        <button role="menuitem" disabled={saving} onClick={() => void runAssetMenuAction({ review_status: "pending" })}><span>要確認へ戻す</span><small>Review</small></button>
        <button role="menuitem" disabled={saving} className="is-danger" onClick={() => void runAssetMenuAction({ review_status: "rejected", training_enabled: false })}><span>除外して学習から外す</span><small>Non-destructive</small></button>
      </div>}

      {snapshots.length > 0 && <div className="studio-next-action"><span><CheckCircle2 size={19} /><span><strong>学習用Snapshotがあります</strong><small>固定済みSnapshotは元Assetの現在状態に影響されません。</small></span></span><button onClick={onOpenTrain}>学習プランへ進む</button></div>}

      {snapshotOpen && (
        <div className="studio-dialog-backdrop" role="presentation" onMouseDown={() => setSnapshotOpen(false)}>
          <section ref={snapshotDialogRef} className="studio-modal is-compact" role="dialog" aria-modal="true" aria-labelledby="snapshot-title" onMouseDown={(event) => event.stopPropagation()} onKeyDown={(event) => trapDialogFocus(event, snapshotDialogRef.current, !creatingSnapshot, () => setSnapshotOpen(false))}>
            <header><div><p className="studio-eyebrow">IMMUTABLE DATASET</p><h2 id="snapshot-title">Snapshotを固定する</h2></div></header>
            <p className="studio-modal-copy">承認済みかつ学習対象の{approved.length}件を固定します。後から元Assetを編集しても、このSnapshotの内容とhashは変わりません。</p>
            <label className="studio-field"><span>Snapshot名</span><input autoFocus value={snapshotName} onChange={(event) => setSnapshotName(event.target.value)} /></label>
            <footer><button className="studio-secondary-button" onClick={() => setSnapshotOpen(false)}>キャンセル</button><button className="studio-primary-button" disabled={!snapshotName.trim() || creatingSnapshot} onClick={() => void createSnapshot()}>{creatingSnapshot ? <Loader2 size={15} className="spin" /> : <Database size={15} />}{creatingSnapshot ? "固定中…" : `${approved.length}件を固定`}</button></footer>
          </section>
        </div>
      )}

      {bulkCaptionOpen && (
        <div className="studio-dialog-backdrop" role="presentation" onMouseDown={() => { if (!bulkSaving && bulkStatus === "preview") setBulkCaptionOpen(false); }}>
          <section ref={bulkDialogRef} className="studio-modal" role="dialog" aria-modal="true" aria-labelledby="caption-bulk-title" onMouseDown={(event) => event.stopPropagation()} onKeyDown={(event) => trapDialogFocus(event, bulkDialogRef.current, !bulkSaving && bulkStatus === "preview", () => setBulkCaptionOpen(false))}>
            <header><div><p className="studio-eyebrow">CAPTION BULK EDIT</p><h2 id="caption-bulk-title">{selectedIds.length}件のCaptionを一括編集</h2></div></header>
            <p className="studio-modal-copy">適用前に変更件数と各AssetのBefore / Afterを確認できます。適用後も、この画面から元のLineageへUndoできます。</p>
            <div className="studio-bulk-caption-fields"><label className="studio-field"><span>検索文字列</span><input autoFocus disabled={bulkStatus === "applied"} value={bulkFind} onChange={(event) => { setBulkFind(event.target.value); setBulkPreview(null); }} /></label><label className="studio-field"><span>置換後</span><input disabled={bulkStatus === "applied"} value={bulkReplace} onChange={(event) => { setBulkReplace(event.target.value); setBulkPreview(null); }} /></label></div>
            {bulkPreview && <section className="studio-bulk-caption-diff" aria-label="Caption bulk diff"><header><strong>{bulkPreview.affected_count}件に影響</strong><span>Operation #{bulkPreview.operation_id} · {bulkStatus}</span></header>{bulkPreview.changes.length === 0 ? <p>一致するCaptionはありません。</p> : <div>{bulkPreview.changes.map((change) => <article key={change.asset_id}><strong>Asset #{change.asset_id}</strong><p><span>Before</span>{change.before || "（空）"}</p><p><span>After</span>{change.after || "（空）"}</p></article>)}</div>}</section>}
            <footer>{bulkStatus === "applied" ? <><button className="studio-secondary-button" disabled={bulkSaving} onClick={() => void undoBulkCaption()}><Undo2 size={15} />Undoして元に戻す</button><button className="studio-primary-button" disabled={bulkSaving} onClick={() => setBulkCaptionOpen(false)}>確定して閉じる</button></> : <><button className="studio-secondary-button" disabled={bulkSaving} onClick={() => setBulkCaptionOpen(false)}>キャンセル</button><button className="studio-secondary-button" disabled={!bulkFind || bulkSaving} onClick={() => void previewBulkCaption()}>{bulkSaving ? <Loader2 size={15} className="spin" /> : <Search size={15} />}Diff Preview</button><button className="studio-primary-button" disabled={!bulkPreview?.affected_count || bulkSaving} onClick={() => void applyBulkCaption()}>{bulkSaving ? <Loader2 size={15} className="spin" /> : <Save size={15} />}Apply</button></>}</footer>
          </section>
        </div>
      )}
    </div>
  );
}
