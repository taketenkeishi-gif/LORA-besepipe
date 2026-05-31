import { useState, useMemo, useEffect, useCallback, type DragEvent } from "react";
import {
  Search,
  Download,
  FolderPlus,
  Tag,
  ArrowUpDown,
  Trash2,
  CheckSquare,
  Square,
  ToggleLeft,
  RefreshCw,
  AlertTriangle,
  ImageOff,
  Plus,
  X,
  ArrowUp,
  ArrowDown,
  Link,
  Upload,
  Wand2,
  Edit3,
  BarChart2,
  Replace,
  MinusCircle,
  Check,
  Loader2,
  Activity,
  Copy,
  Layers,
  ChevronDown,
  ChevronRight,
} from "lucide-react";
import { apiGet, apiPost, apiFormPost, API_BASE } from "../lib/api";
import { parseIntOr, extractDroppedUrl } from "../lib/utils";
import type {
  Project,
  ScanItem,
  SortKey,
  SortDir,
  ScanResult,
  CaptionItem,
  TaggerStatus,
  DatasetAnalysis,
  SimilarityGroup,
  CharacterLeakResult,
  TagCategoriesResult,
  DistributionData,
} from "../types";

type Props = {
  projects: Project[];
  selectedProjectId: number | null;
  selectedProject: Project | null;
  onSelectProject: (id: number) => void;
  onRefresh: () => Promise<void>;
  showError: (msg: string) => void;
  showNotice: (msg: string) => void;
  clearMessages: () => void;
};

type ActiveTab = "collect" | "caption" | "dashboard";

function sortItems(items: ScanItem[], key: SortKey, dir: SortDir): ScanItem[] {
  return [...items].sort((a, b) => {
    let cmp = 0;
    if (key === "area") cmp = a.width * a.height - b.width * b.height;
    else if (key === "width") cmp = a.width - b.width;
    else if (key === "height") cmp = a.height - b.height;
    else if (key === "title") cmp = a.title.localeCompare(b.title);
    else if (key === "aspect") cmp = a.aspect.localeCompare(b.aspect);
    else cmp = a.id - b.id;
    return dir === "asc" ? cmp : -cmp;
  });
}

const SORT_OPTIONS: { key: SortKey; label: string }[] = [
  { key: "id", label: "追加順" },
  { key: "area", label: "解像度" },
  { key: "width", label: "幅" },
  { key: "height", label: "高さ" },
  { key: "aspect", label: "アスペクト" },
  { key: "title", label: "名前" },
];

const ASPECT_OPTIONS = ["all", "portrait", "landscape", "square"] as const;

export default function Dataset({
  projects,
  selectedProjectId,
  selectedProject,
  onSelectProject,
  onRefresh,
  showError,
  showNotice,
  clearMessages,
}: Props) {
  const [activeTab, setActiveTab] = useState<ActiveTab>("collect");

  // ── 収集タブ状態 ──
  const [scanItems, setScanItems] = useState<ScanItem[]>([]);
  const [selectedIds, setSelectedIds] = useState<number[]>([]);
  const [expandedTagId, setExpandedTagId] = useState<number | null>(null);
  const [scanUrl, setScanUrl] = useState("https://www.pixiv.net/artworks/143382932");
  const [scanKeyword, setScanKeyword] = useState("");
  const [tagFilter, setTagFilter] = useState("");
  const [aspectFilter, setAspectFilter] = useState("all");
  const [minWText, setMinWText] = useState("0");
  const [minHText, setMinHText] = useState("0");
  const [scanMode, setScanMode] = useState<string | null>(null);
  const [scanMessage, setScanMessage] = useState<string | null>(null);
  const [sortKey, setSortKey] = useState<SortKey>("id");
  const [sortDir, setSortDir] = useState<SortDir>("desc");
  const [namingTemplate, setNamingTemplate] = useState("{title}_{index}");
  const [repeatCountText, setRepeatCountText] = useState("5");
  const [repeatFolderTitle, setRepeatFolderTitle] = useState("");
  const [repeatPreparedPath, setRepeatPreparedPath] = useState("");
  const [dragActive, setDragActive] = useState(false);
  const [scanning, setScanning] = useState(false);

  // ── キャプションタブ状態 ──
  const [captionItems, setCaptionItems] = useState<CaptionItem[]>([]);
  const [taggerStatus, setTaggerStatus] = useState<TaggerStatus | null>(null);
  const [captionPolling, setCaptionPolling] = useState(false);
  const [editingId, setEditingId] = useState<number | null>(null);
  const [editingText, setEditingText] = useState("");
  const [captionFilter, setCaptionFilter] = useState<"all" | "empty" | "generated" | "manual">("all");
  const [batchFind, setBatchFind] = useState("");
  const [batchReplace, setBatchReplace] = useState("");
  const [batchRemoveInput, setBatchRemoveInput] = useState("");
  const [showBatchReplace, setShowBatchReplace] = useState(false);
  const [showBatchRemove, setShowBatchRemove] = useState(false);
  const [showFrequency, setShowFrequency] = useState(false);
  const [frequencyData, setFrequencyData] = useState<{ tag: string; count: number }[]>([]);

  // ── ダッシュボードタブ状態 ──
  const [dashboardData, setDashboardData] = useState<DatasetAnalysis | null>(null);
  const [analysisPolling, setAnalysisPolling] = useState(false);
  const [analysisStatus, setAnalysisStatus] = useState<{ status: string; message: string } | null>(null);
  const [expandedGroups, setExpandedGroups] = useState<Set<number>>(new Set());
  const [leakData, setLeakData] = useState<CharacterLeakResult | null>(null);
  const [categoryData, setCategoryData] = useState<TagCategoriesResult | null>(null);
  const [leakLoading, setLeakLoading] = useState(false);
  const [distributionData, setDistributionData] = useState<DistributionData | null>(null);
  const [distributionLoading, setDistributionLoading] = useState(false);

  const minW = parseIntOr(minWText, 0, 0);
  const minH = parseIntOr(minHText, 0, 0);

  function buildKeyword() {
    const terms: string[] = [];
    if (scanKeyword.trim()) terms.push(scanKeyword.trim());
    if (tagFilter.trim()) terms.push(`tag:${tagFilter.trim()}`);
    if (aspectFilter !== "all") terms.push(`aspect:${aspectFilter}`);
    if (minW > 0) terms.push(`minw:${minW}`);
    if (minH > 0) terms.push(`minh:${minH}`);
    return terms.join(" ");
  }

  const displayItems = useMemo(() => sortItems(scanItems, sortKey, sortDir), [scanItems, sortKey, sortDir]);

  const filteredCaptionItems = useMemo(() => {
    if (captionFilter === "all") return captionItems;
    if (captionFilter === "empty") return captionItems.filter((c) => !c.caption.trim());
    if (captionFilter === "generated") return captionItems.filter((c) => c.caption_source === "wd14");
    if (captionFilter === "manual") return captionItems.filter((c) => c.caption_source === "manual");
    return captionItems;
  }, [captionItems, captionFilter]);

  // キャプション一覧取得
  const loadCaptions = useCallback(async () => {
    if (!selectedProject) return;
    try {
      const items = await apiGet<CaptionItem[]>(`/tags/${selectedProject.id}`);
      setCaptionItems(items);
    } catch {
      // エラーは静かにスキップ
    }
  }, [selectedProject]);

  // タグ生成進捗のポーリング
  useEffect(() => {
    if (!captionPolling || !selectedProject) return;
    const id = setInterval(async () => {
      try {
        const st = await apiGet<TaggerStatus>(`/tags/status/${selectedProject.id}`);
        setTaggerStatus(st);
        if (st.status === "done" || st.status === "done_with_errors" || st.status === "idle") {
          setCaptionPolling(false);
          await loadCaptions();
          showNotice(st.message);
        }
      } catch {
        setCaptionPolling(false);
      }
    }, 1500);
    return () => clearInterval(id);
  }, [captionPolling, selectedProject, loadCaptions, showNotice]);

  // キャプションタブに切り替えたら一覧取得
  useEffect(() => {
    if (activeTab === "caption" && selectedProject) {
      void loadCaptions();
    }
  }, [activeTab, selectedProject, loadCaptions]);

  // ── ダッシュボード ──
  const loadDashboard = useCallback(async () => {
    if (!selectedProject) return;
    try {
      const data = await apiGet<DatasetAnalysis>(`/dataset/stats/${selectedProject.id}`);
      setDashboardData(data);
    } catch {
      // エラーは静かにスキップ
    }
  }, [selectedProject]);

  useEffect(() => {
    if (activeTab === "dashboard" && selectedProject) {
      void loadDashboard();
    }
  }, [activeTab, selectedProject, loadDashboard]);

  useEffect(() => {
    if (!analysisPolling || !selectedProject) return;
    const id = setInterval(async () => {
      try {
        const st = await apiGet<{ status: string; message: string }>(`/dataset/analysis-status/${selectedProject.id}`);
        setAnalysisStatus(st);
        if (st.status === "done" || st.status === "idle") {
          setAnalysisPolling(false);
          await loadDashboard();
        }
      } catch {
        setAnalysisPolling(false);
      }
    }, 2000);
    return () => clearInterval(id);
  }, [analysisPolling, selectedProject, loadDashboard]);

  async function startAnalysis() {
    if (!selectedProject) return;
    try {
      await apiPost(`/dataset/analyze/${selectedProject.id}`, {});
      setAnalysisStatus({ status: "queued", message: "分析を準備中..." });
      setAnalysisPolling(true);
    } catch (e) {
      showError(`分析開始失敗: ${String(e)}`);
    }
  }

  async function loadCharacterLeak() {
    if (!selectedProject) return;
    setLeakLoading(true);
    try {
      const [ld, cd] = await Promise.all([
        apiGet<CharacterLeakResult>(`/dataset/character-leak/${selectedProject.id}`),
        apiGet<TagCategoriesResult>(`/dataset/tag-categories/${selectedProject.id}`),
      ]);
      setLeakData(ld);
      setCategoryData(cd);
    } catch (e) {
      showError(`リーク分析失敗: ${String(e)}`);
    } finally {
      setLeakLoading(false);
    }
  }

  async function loadDistribution() {
    if (!selectedProject) return;
    setDistributionLoading(true);
    try {
      const d = await apiGet<DistributionData>(`/dataset/distribution/${selectedProject.id}`);
      setDistributionData(d);
    } catch (e) {
      showError(`分布分析失敗: ${String(e)}`);
    } finally {
      setDistributionLoading(false);
    }
  }

  function toggleGroup(idx: number) {
    setExpandedGroups((prev) => {
      const next = new Set(prev);
      if (next.has(idx)) next.delete(idx);
      else next.add(idx);
      return next;
    });
  }

  function toggleSort(key: SortKey) {
    if (sortKey === key) setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    else { setSortKey(key); setSortDir("desc"); }
  }

  function toggleId(id: number) {
    setSelectedIds((prev) => prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]);
  }
  function selectAll() { setSelectedIds(displayItems.map((x) => x.id)); }
  function selectNone() { setSelectedIds([]); }
  function invertSelection() {
    const all = displayItems.map((x) => x.id);
    setSelectedIds(all.filter((id) => !selectedIds.includes(id)));
  }

  async function runScan() {
    if (!selectedProject) return;
    clearMessages();
    setScanning(true);
    try {
      const r = await apiPost<ScanResult>("/collector/scan", {
        project_id: selectedProject.id,
        url: scanUrl,
        keyword: buildKeyword(),
        limit: 60,
      });
      setScanItems(r.items);
      setSelectedIds(r.items.slice(0, 12).map((x) => x.id));
      setExpandedTagId(null);
      setScanMode(r.mode);
      setScanMessage(r.message);
      if (r.mode === "url_unavailable") showError(`画像取得失敗: ${r.message}`);
      else showNotice(`候補取得: ${r.detected}件 (${r.mode})`);
    } catch (e) {
      showError(`取得失敗: ${String(e)}`);
    } finally {
      setScanning(false);
    }
  }

  async function removeItems(ids: number[]) {
    if (!selectedProject || ids.length === 0) return;
    try {
      const r = await apiPost<{ items: ScanItem[]; removed_count: number }>(
        "/collector/remove-candidates",
        { project_id: selectedProject.id, candidate_ids: ids },
      );
      setScanItems(r.items || []);
      setSelectedIds((prev) => prev.filter((id) => !ids.includes(id)));
      showNotice(`${r.removed_count ?? ids.length}件削除`);
    } catch (e) {
      showError(`削除失敗: ${String(e)}`);
    }
  }

  async function dropUrlAsCandidate(projectId: number, url: string) {
    try {
      const r = await apiPost<{ items: ScanItem[] }>("/collector/drop-url", { project_id: projectId, url });
      setScanItems(r.items || []);
      showNotice("URLを候補に追加しました");
    } catch (e) {
      showError(`URL追加失敗: ${String(e)}`);
    }
  }

  async function onDropFiles(ev: DragEvent<HTMLDivElement>) {
    if (!selectedProject) return;
    ev.preventDefault();
    setDragActive(false);
    clearMessages();
    const files = Array.from(ev.dataTransfer.files).filter(
      (f) => f.type?.startsWith("image/") || /\.(png|jpe?g|webp|bmp|gif)$/i.test(f.name),
    );
    if (files.length === 0) {
      const url = extractDroppedUrl(ev);
      if (url) await dropUrlAsCandidate(selectedProject.id, url);
      return;
    }
    try {
      const form = new FormData();
      form.append("project_id", String(selectedProject.id));
      files.forEach((f) => form.append("files", f));
      const json = await apiFormPost<{ items: ScanItem[]; added_count: number }>("/collector/drop-files", form);
      setScanItems(json.items || []);
      showNotice(`${json.added_count ?? files.length}件追加`);
    } catch (e) {
      showError(`ドラッグ&ドロップ追加失敗: ${String(e)}`);
    }
  }

  async function prepareRepeatFolder() {
    if (!selectedProject) return;
    clearMessages();
    const title = repeatFolderTitle || selectedProject.name;
    try {
      const r = await apiPost<{ folder: string; copied_images: number }>("/collector/prepare-repeat-folder", {
        project_id: selectedProject.id,
        repeats: parseIntOr(repeatCountText, 5, 1),
        folder_title: title,
      });
      setRepeatPreparedPath(r.folder);
      showNotice(`学習フォルダを作成: ${r.folder} (${r.copied_images}件コピー)`);
    } catch (e) {
      showError(`作成失敗: ${String(e)}`);
    }
  }

  async function runImport() {
    if (!selectedProject) return;
    if (!repeatPreparedPath) { showError("先に「2) Createフォルダー」を実行してください。"); return; }
    clearMessages();
    try {
      await apiPost("/collector/import", {
        project_id: selectedProject.id,
        selected_ids: selectedIds,
        naming_template: namingTemplate,
        import_dir: repeatPreparedPath,
      });
      showNotice(`${selectedIds.length}件を取り込み完了`);
    } catch (e) {
      showError(`取り込み失敗: ${String(e)}`);
    }
  }

  // ── キャプション操作 ──

  async function startWd14(overwrite = false) {
    if (!selectedProject) return;
    try {
      await apiPost("/tags/generate", { project_id: selectedProject.id, overwrite });
      setTaggerStatus({ project_id: selectedProject.id, status: "queued", total: 0, done: 0, message: "モデル読み込み中..." });
      setCaptionPolling(true);
    } catch (e) {
      showError(`タグ生成開始失敗: ${String(e)}`);
    }
  }

  async function saveCaption(itemId: number, caption: string) {
    try {
      await apiPost(`/tags/item/${itemId}`, { caption }, "PATCH");
      setCaptionItems((prev) => prev.map((c) => c.id === itemId ? { ...c, caption, caption_source: "manual" } : c));
    } catch (e) {
      showError(`保存失敗: ${String(e)}`);
    }
  }

  async function runBatchReplace() {
    if (!selectedProject || !batchFind) return;
    try {
      const r = await apiPost<{ updated_count: number }>("/tags/batch-replace", {
        project_id: selectedProject.id,
        find: batchFind,
        replace: batchReplace,
      });
      showNotice(`${r.updated_count}件を置換しました`);
      await loadCaptions();
    } catch (e) {
      showError(`バッチ置換失敗: ${String(e)}`);
    }
  }

  async function runBatchRemove() {
    if (!selectedProject || !batchRemoveInput.trim()) return;
    const tags = batchRemoveInput.split(",").map((t) => t.trim()).filter(Boolean);
    try {
      const r = await apiPost<{ updated_count: number }>("/tags/batch-remove", {
        project_id: selectedProject.id,
        tags,
      });
      showNotice(`${r.updated_count}件からタグを削除しました`);
      await loadCaptions();
    } catch (e) {
      showError(`バッチ削除失敗: ${String(e)}`);
    }
  }

  async function loadFrequency() {
    if (!selectedProject) return;
    try {
      const r = await apiGet<{ frequency: { tag: string; count: number }[] }>(`/tags/frequency/${selectedProject.id}`);
      setFrequencyData(r.frequency.slice(0, 40));
      setShowFrequency(true);
    } catch (e) {
      showError(`頻度取得失敗: ${String(e)}`);
    }
  }

  const isTagging = taggerStatus?.status === "running" || taggerStatus?.status === "queued";
  const tagProgress = taggerStatus && taggerStatus.total > 0
    ? Math.round((taggerStatus.done / taggerStatus.total) * 100)
    : 0;

  const captionStats = useMemo(() => {
    const total = captionItems.length;
    const withCaption = captionItems.filter((c) => c.caption.trim()).length;
    const generated = captionItems.filter((c) => c.caption_source === "wd14").length;
    const manual = captionItems.filter((c) => c.caption_source === "manual").length;
    return { total, withCaption, generated, manual };
  }, [captionItems]);

  if (!selectedProject) {
    return (
      <div className="space-y-4 max-w-4xl">
        <div>
          <h2 className="text-xl font-bold text-gray-100 mb-1">データセット作成</h2>
          <p className="text-sm text-gray-400">プロジェクトを選択してください</p>
        </div>
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
          </div>
          {projects.length === 0 && (
            <p className="text-gray-500 text-sm">プロジェクトがありません。まずプロジェクト管理で作成してください。</p>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4 max-w-5xl">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h2 className="text-xl font-bold text-gray-100 mb-1">データセット作成</h2>
          <div className="flex items-center gap-2 text-sm text-gray-400">
            <span>対象:</span>
            <span className="text-indigo-400 font-medium">{selectedProject.name}</span>
            <span className="text-gray-600">/</span>
            <span>{selectedProject.project_type}</span>
          </div>
        </div>
        <select
          value={selectedProject.id}
          onChange={(e) => onSelectProject(Number(e.target.value))}
          className="bg-gray-700 border border-gray-600 text-gray-300 rounded-lg px-3 py-1.5 text-sm focus:outline-none"
        >
          {projects.map((p) => (
            <option key={p.id} value={p.id}>{p.name}</option>
          ))}
        </select>
      </div>

      {/* Tab switcher */}
      <div className="flex border-b border-gray-700">
        <button
          onClick={() => setActiveTab("collect")}
          className={[
            "flex items-center gap-2 px-4 py-2.5 text-sm font-medium border-b-2 transition-colors",
            activeTab === "collect"
              ? "border-indigo-500 text-indigo-400"
              : "border-transparent text-gray-400 hover:text-gray-300",
          ].join(" ")}
        >
          <Download size={15} />
          画像収集
        </button>
        <button
          onClick={() => setActiveTab("caption")}
          className={[
            "flex items-center gap-2 px-4 py-2.5 text-sm font-medium border-b-2 transition-colors",
            activeTab === "caption"
              ? "border-violet-500 text-violet-400"
              : "border-transparent text-gray-400 hover:text-gray-300",
          ].join(" ")}
        >
          <Tag size={15} />
          キャプション編集
          {captionStats.total > 0 && (
            <span className="text-xs bg-gray-700 text-gray-400 px-1.5 py-0.5 rounded-full">
              {captionStats.withCaption}/{captionStats.total}
            </span>
          )}
        </button>
        <button
          onClick={() => setActiveTab("dashboard")}
          className={[
            "flex items-center gap-2 px-4 py-2.5 text-sm font-medium border-b-2 transition-colors",
            activeTab === "dashboard"
              ? "border-emerald-500 text-emerald-400"
              : "border-transparent text-gray-400 hover:text-gray-300",
          ].join(" ")}
        >
          <Activity size={15} />
          Dataset Dashboard
          {dashboardData?.quality_score != null && (
            <span className={[
              "text-xs px-1.5 py-0.5 rounded-full font-bold",
              dashboardData.quality_score >= 80 ? "bg-emerald-900 text-emerald-300"
              : dashboardData.quality_score >= 60 ? "bg-amber-900 text-amber-300"
              : "bg-red-900 text-red-300",
            ].join(" ")}>
              {dashboardData.quality_score}
            </span>
          )}
        </button>
      </div>

      {/* ─────────────────────────── 収集タブ ─────────────────────────── */}
      {activeTab === "collect" && (
        <>
          {/* Drop Zone */}
          <div
            className={[
              "border-2 border-dashed rounded-xl transition-colors cursor-default",
              dragActive ? "border-indigo-500 bg-indigo-950/30" : "border-gray-700 bg-gray-800/50",
            ].join(" ")}
            onDrop={(e) => void onDropFiles(e)}
            onDragEnter={(e) => { e.preventDefault(); setDragActive(true); }}
            onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
            onDragLeave={() => setDragActive(false)}
          >
            <div className="p-4">
              <div className="grid gap-3">
                <div className="flex gap-2">
                  <div className="flex-1 relative">
                    <Link size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" />
                    <input
                      value={scanUrl}
                      onChange={(e) => setScanUrl(e.target.value)}
                      placeholder="https://www.pixiv.net/artworks/..."
                      className="w-full bg-gray-700 border border-gray-600 text-gray-100 placeholder-gray-500 rounded-lg pl-8 pr-3 py-2 text-sm focus:outline-none focus:border-indigo-500"
                    />
                  </div>
                  <button
                    onClick={() => void runScan()}
                    disabled={!selectedProject || scanning}
                    className="flex items-center gap-2 bg-blue-700 hover:bg-blue-600 disabled:bg-gray-700 disabled:text-gray-500 text-white rounded-lg px-4 py-2 text-sm font-medium transition-colors"
                  >
                    {scanning ? <RefreshCw size={15} className="animate-spin" /> : <Search size={15} />}
                    1) 候補取得
                  </button>
                </div>

                <div className="flex flex-wrap gap-2 items-center">
                  <div className="relative">
                    <Search size={13} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-gray-500" />
                    <input
                      value={scanKeyword}
                      onChange={(e) => setScanKeyword(e.target.value)}
                      placeholder="キーワード"
                      className="bg-gray-700 border border-gray-600 text-gray-300 placeholder-gray-500 rounded-lg pl-7 pr-2 py-1.5 text-xs w-28 focus:outline-none focus:border-indigo-500"
                    />
                  </div>
                  <div className="relative">
                    <Tag size={13} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-gray-500" />
                    <input
                      value={tagFilter}
                      onChange={(e) => setTagFilter(e.target.value)}
                      placeholder="タグ"
                      className="bg-gray-700 border border-gray-600 text-gray-300 placeholder-gray-500 rounded-lg pl-7 pr-2 py-1.5 text-xs w-24 focus:outline-none focus:border-indigo-500"
                    />
                  </div>
                  <select
                    value={aspectFilter}
                    onChange={(e) => setAspectFilter(e.target.value)}
                    className="bg-gray-700 border border-gray-600 text-gray-300 rounded-lg px-2 py-1.5 text-xs focus:outline-none"
                  >
                    {ASPECT_OPTIONS.map((a) => (
                      <option key={a} value={a}>{a === "all" ? "全アスペクト" : a}</option>
                    ))}
                  </select>
                  <div className="flex items-center gap-1">
                    <input
                      type="text" inputMode="numeric" value={minWText}
                      onChange={(e) => setMinWText(e.target.value)}
                      placeholder="minW"
                      className="bg-gray-700 border border-gray-600 text-gray-300 placeholder-gray-500 rounded-lg px-2 py-1.5 text-xs w-16 focus:outline-none"
                    />
                    <span className="text-gray-600 text-xs">×</span>
                    <input
                      type="text" inputMode="numeric" value={minHText}
                      onChange={(e) => setMinHText(e.target.value)}
                      placeholder="minH"
                      className="bg-gray-700 border border-gray-600 text-gray-300 placeholder-gray-500 rounded-lg px-2 py-1.5 text-xs w-16 focus:outline-none"
                    />
                  </div>
                </div>

                {scanMode && (
                  <div className={[
                    "flex items-center gap-2 text-xs px-3 py-2 rounded-lg",
                    scanMode === "url_unavailable"
                      ? "bg-amber-950 text-amber-300 border border-amber-800"
                      : "bg-gray-700 text-gray-400",
                  ].join(" ")}>
                    {scanMode === "url_unavailable" && <AlertTriangle size={13} />}
                    <span>{scanMessage}</span>
                  </div>
                )}

                <div className="flex items-center gap-2 text-xs text-gray-500">
                  <Upload size={13} />
                  <span>画像ファイルやURLをここにドラッグ&ドロップで追加できます</span>
                </div>
              </div>
            </div>
          </div>

          {/* Image Grid */}
          {scanItems.length > 0 && (
            <div className="bg-gray-800 border border-gray-700 rounded-xl overflow-hidden">
              <div className="px-4 py-3 border-b border-gray-700 flex flex-wrap items-center gap-3">
                <div className="flex items-center gap-1">
                  <button onClick={selectAll} className="flex items-center gap-1.5 text-xs px-2 py-1 rounded bg-gray-700 hover:bg-gray-600 text-gray-300 transition-colors">
                    <CheckSquare size={13} />全選択
                  </button>
                  <button onClick={selectNone} className="flex items-center gap-1.5 text-xs px-2 py-1 rounded bg-gray-700 hover:bg-gray-600 text-gray-300 transition-colors">
                    <Square size={13} />解除
                  </button>
                  <button onClick={invertSelection} className="flex items-center gap-1.5 text-xs px-2 py-1 rounded bg-gray-700 hover:bg-gray-600 text-gray-300 transition-colors">
                    <ToggleLeft size={13} />反転
                  </button>
                </div>
                <div className="text-xs text-gray-500">{selectedIds.length}/{scanItems.length} 選択中</div>
                {selectedIds.length > 0 && (
                  <button onClick={() => void removeItems(selectedIds)} className="flex items-center gap-1.5 text-xs px-2 py-1 rounded bg-red-950 hover:bg-red-900 text-red-300 transition-colors ml-auto">
                    <Trash2 size={13} />選択削除 ({selectedIds.length})
                  </button>
                )}
                <div className="ml-auto flex items-center gap-1 flex-wrap">
                  <ArrowUpDown size={13} className="text-gray-500" />
                  {SORT_OPTIONS.map(({ key, label }) => (
                    <button
                      key={key}
                      onClick={() => toggleSort(key)}
                      className={[
                        "flex items-center gap-1 text-xs px-2 py-1 rounded transition-colors",
                        sortKey === key ? "bg-indigo-900 text-indigo-300" : "bg-gray-700 hover:bg-gray-600 text-gray-400",
                      ].join(" ")}
                    >
                      {label}
                      {sortKey === key && (sortDir === "desc" ? <ArrowDown size={11} /> : <ArrowUp size={11} />)}
                    </button>
                  ))}
                </div>
              </div>

              <div className="p-4 grid grid-cols-4 gap-3 sm:grid-cols-5 lg:grid-cols-6 xl:grid-cols-8">
                {displayItems.map((item) => {
                  const isSelected = selectedIds.includes(item.id);
                  return (
                    <div
                      key={item.id}
                      className={[
                        "relative group rounded-lg overflow-hidden border-2 cursor-pointer transition-all",
                        isSelected ? "border-indigo-500 ring-2 ring-indigo-500/30" : "border-transparent hover:border-gray-500",
                      ].join(" ")}
                      onClick={() => toggleId(item.id)}
                      onDoubleClick={() => setExpandedTagId((prev) => prev === item.id ? null : item.id)}
                    >
                      <div className="aspect-square bg-gray-700">
                        {item.thumbnail_url ? (
                          <img src={item.thumbnail_url} alt={item.title} className="w-full h-full object-cover" />
                        ) : (
                          <div className="w-full h-full flex items-center justify-center">
                            <ImageOff size={20} className="text-gray-600" />
                          </div>
                        )}
                      </div>
                      <div className={[
                        "absolute top-1.5 left-1.5 w-5 h-5 rounded border-2 flex items-center justify-center transition-all",
                        isSelected ? "bg-indigo-500 border-indigo-400" : "bg-gray-900/70 border-gray-500 opacity-0 group-hover:opacity-100",
                      ].join(" ")}>
                        {isSelected && (
                          <svg width="11" height="9" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <polyline points="1,4 4,8 10,1" />
                          </svg>
                        )}
                      </div>
                      <button
                        className="absolute top-1.5 right-1.5 w-5 h-5 bg-red-900/80 hover:bg-red-700 rounded flex items-center justify-center opacity-0 group-hover:opacity-100 transition-all"
                        onClick={(e) => { e.stopPropagation(); void removeItems([item.id]); }}
                        title="削除"
                      >
                        <X size={11} className="text-white" />
                      </button>
                      <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-1.5">
                        <div className="text-xs text-gray-300 truncate leading-tight">{item.width}×{item.height}</div>
                        <div className="text-xs text-gray-500 leading-tight">{item.aspect}</div>
                      </div>
                      {expandedTagId === item.id && (
                        <div className="absolute inset-0 bg-gray-900/95 p-2 flex flex-wrap gap-1 content-start overflow-y-auto">
                          {(item.tags || []).map((t) => (
                            <span key={`${item.id}-${t}`} className="text-xs bg-indigo-900 text-indigo-300 px-1.5 py-0.5 rounded">{t}</span>
                          ))}
                          {(item.tags || []).length === 0 && <span className="text-xs text-gray-500">タグなし</span>}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Workflow Steps */}
          <div className="bg-gray-800 border border-gray-700 rounded-xl p-5">
            <h3 className="text-sm font-semibold text-gray-300 mb-4">ワークフロー</h3>
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              <WorkflowBtn step={2} label="Createフォルダー" icon={<FolderPlus size={16} />} color="bg-indigo-700 hover:bg-indigo-600" onClick={() => void prepareRepeatFolder()} />
              <WorkflowBtn step={3} label="取り込み" icon={<Download size={16} />} color="bg-emerald-700 hover:bg-emerald-600" onClick={() => void runImport()} />
              <WorkflowBtn step={4} label="キャプション編集へ" icon={<Tag size={16} />} color="bg-violet-700 hover:bg-violet-600" onClick={() => setActiveTab("caption")} />
              <button
                onClick={() => selectedProject && void dropUrlAsCandidate(selectedProject.id, scanUrl)}
                className="flex flex-col items-center gap-2 bg-gray-700 hover:bg-gray-600 text-gray-300 rounded-lg p-3 text-xs font-medium transition-colors"
              >
                <Plus size={16} />
                URL候補追加
              </button>
            </div>
            <div className="mt-4 grid grid-cols-2 gap-3">
              <label className="block">
                <span className="text-xs text-gray-400 mb-1 block">繰り返し数</span>
                <input
                  type="text" inputMode="numeric" value={repeatCountText}
                  onChange={(e) => setRepeatCountText(e.target.value)}
                  className="w-full bg-gray-700 border border-gray-600 text-gray-100 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:border-indigo-500"
                />
              </label>
              <label className="block">
                <span className="text-xs text-gray-400 mb-1 block">フォルダ名</span>
                <input
                  value={repeatFolderTitle || selectedProject.name}
                  onChange={(e) => setRepeatFolderTitle(e.target.value)}
                  className="w-full bg-gray-700 border border-gray-600 text-gray-100 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:border-indigo-500"
                />
              </label>
            </div>
            <label className="block mt-3">
              <span className="text-xs text-gray-400 mb-1 block">命名テンプレート</span>
              <input
                value={namingTemplate}
                onChange={(e) => setNamingTemplate(e.target.value)}
                placeholder="{title}_{index}"
                className="w-full bg-gray-700 border border-gray-600 text-gray-100 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:border-indigo-500"
              />
            </label>
            {repeatPreparedPath && (
              <div className="mt-3 flex items-center gap-2 text-xs text-emerald-400 bg-emerald-950 border border-emerald-900 rounded-lg px-3 py-2">
                <FolderPlus size={13} />
                <span>作成済み: {repeatPreparedPath}</span>
              </div>
            )}
          </div>
        </>
      )}

      {/* ─────────────────────────── キャプションタブ ─────────────────────────── */}
      {activeTab === "caption" && (
        <>
          {/* 統計カード */}
          <div className="grid grid-cols-4 gap-3">
            {[
              { label: "総画像数", value: captionStats.total, color: "text-gray-300" },
              { label: "キャプション済み", value: captionStats.withCaption, color: "text-emerald-400" },
              { label: "WD14生成", value: captionStats.generated, color: "text-violet-400" },
              { label: "手動編集", value: captionStats.manual, color: "text-indigo-400" },
            ].map(({ label, value, color }) => (
              <div key={label} className="bg-gray-800 border border-gray-700 rounded-xl p-3 text-center">
                <div className={`text-2xl font-bold ${color}`}>{value}</div>
                <div className="text-xs text-gray-500 mt-0.5">{label}</div>
              </div>
            ))}
          </div>

          {/* WD14生成コントロール */}
          <div className="bg-gray-800 border border-gray-700 rounded-xl p-4">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <Wand2 size={16} className="text-violet-400" />
                <span className="text-sm font-semibold text-gray-200">WD14 自動タグ生成</span>
              </div>
              <div className="flex gap-2">
                <button
                  onClick={() => void startWd14(false)}
                  disabled={isTagging}
                  className="flex items-center gap-1.5 bg-violet-700 hover:bg-violet-600 disabled:bg-gray-700 disabled:text-gray-500 text-white rounded-lg px-3 py-1.5 text-xs font-medium transition-colors"
                >
                  {isTagging ? <Loader2 size={13} className="animate-spin" /> : <Wand2 size={13} />}
                  未処理を生成
                </button>
                <button
                  onClick={() => void startWd14(true)}
                  disabled={isTagging}
                  className="flex items-center gap-1.5 bg-gray-700 hover:bg-gray-600 disabled:opacity-50 text-gray-300 rounded-lg px-3 py-1.5 text-xs font-medium transition-colors"
                >
                  <RefreshCw size={13} />
                  全て再生成
                </button>
              </div>
            </div>

            {taggerStatus && taggerStatus.status !== "idle" && (
              <div className="space-y-2">
                <div className="flex items-center justify-between text-xs text-gray-400">
                  <span>{taggerStatus.message}</span>
                  {taggerStatus.total > 0 && (
                    <span>{taggerStatus.done}/{taggerStatus.total}</span>
                  )}
                </div>
                {taggerStatus.total > 0 && (
                  <div className="w-full bg-gray-700 rounded-full h-1.5">
                    <div
                      className="bg-violet-500 h-1.5 rounded-full transition-all"
                      style={{ width: `${tagProgress}%` }}
                    />
                  </div>
                )}
                {(taggerStatus.status === "done" || taggerStatus.status === "done_with_errors") && (
                  <div className={`text-xs px-2 py-1 rounded ${taggerStatus.status === "done" ? "text-emerald-400" : "text-amber-400"}`}>
                    {taggerStatus.status === "done" ? "✓ 完了" : "⚠ 一部エラー"}: {taggerStatus.message}
                  </div>
                )}
              </div>
            )}

            <p className="text-xs text-gray-500 mt-2">
              WD14 (wd-vit-large-tagger-v3) を使用。初回実行時にモデル (~600MB) をダウンロードします。
            </p>
          </div>

          {/* バッチ操作パネル */}
          <div className="flex gap-2">
            <button
              onClick={() => { setShowBatchReplace(!showBatchReplace); setShowBatchRemove(false); setShowFrequency(false); }}
              className={[
                "flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg border transition-colors",
                showBatchReplace ? "bg-indigo-900 border-indigo-700 text-indigo-300" : "bg-gray-800 border-gray-700 text-gray-400 hover:text-gray-300",
              ].join(" ")}
            >
              <Replace size={13} />バッチ置換
            </button>
            <button
              onClick={() => { setShowBatchRemove(!showBatchRemove); setShowBatchReplace(false); setShowFrequency(false); }}
              className={[
                "flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg border transition-colors",
                showBatchRemove ? "bg-red-950 border-red-800 text-red-300" : "bg-gray-800 border-gray-700 text-gray-400 hover:text-gray-300",
              ].join(" ")}
            >
              <MinusCircle size={13} />バッチ削除
            </button>
            <button
              onClick={() => { void loadFrequency(); setShowBatchReplace(false); setShowBatchRemove(false); }}
              className={[
                "flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg border transition-colors",
                showFrequency ? "bg-emerald-950 border-emerald-800 text-emerald-300" : "bg-gray-800 border-gray-700 text-gray-400 hover:text-gray-300",
              ].join(" ")}
            >
              <BarChart2 size={13} />タグ頻度
            </button>
            <button
              onClick={() => void loadCaptions()}
              className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg border bg-gray-800 border-gray-700 text-gray-400 hover:text-gray-300 transition-colors ml-auto"
            >
              <RefreshCw size={13} />更新
            </button>
          </div>

          {showBatchReplace && (
            <div className="bg-gray-800 border border-indigo-900 rounded-xl p-4 space-y-3">
              <h4 className="text-sm font-semibold text-indigo-300 flex items-center gap-2"><Replace size={14} />バッチ置換</h4>
              <div className="grid grid-cols-2 gap-3">
                <label className="block">
                  <span className="text-xs text-gray-400 mb-1 block">検索するタグ</span>
                  <input
                    value={batchFind}
                    onChange={(e) => setBatchFind(e.target.value)}
                    placeholder="例: blue eyes"
                    className="w-full bg-gray-700 border border-gray-600 text-gray-100 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:border-indigo-500"
                  />
                </label>
                <label className="block">
                  <span className="text-xs text-gray-400 mb-1 block">置換後のタグ</span>
                  <input
                    value={batchReplace}
                    onChange={(e) => setBatchReplace(e.target.value)}
                    placeholder="例: blue_eyes（空白 = 削除）"
                    className="w-full bg-gray-700 border border-gray-600 text-gray-100 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:border-indigo-500"
                  />
                </label>
              </div>
              <button
                onClick={() => void runBatchReplace()}
                disabled={!batchFind}
                className="flex items-center gap-2 bg-indigo-700 hover:bg-indigo-600 disabled:opacity-50 text-white rounded-lg px-4 py-1.5 text-sm transition-colors"
              >
                <Check size={14} />実行
              </button>
            </div>
          )}

          {showBatchRemove && (
            <div className="bg-gray-800 border border-red-900 rounded-xl p-4 space-y-3">
              <h4 className="text-sm font-semibold text-red-300 flex items-center gap-2"><MinusCircle size={14} />バッチ削除</h4>
              <label className="block">
                <span className="text-xs text-gray-400 mb-1 block">削除するタグ（カンマ区切り）</span>
                <input
                  value={batchRemoveInput}
                  onChange={(e) => setBatchRemoveInput(e.target.value)}
                  placeholder="例: copyright notice, watermark, signature"
                  className="w-full bg-gray-700 border border-gray-600 text-gray-100 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:border-red-600"
                />
              </label>
              <button
                onClick={() => void runBatchRemove()}
                disabled={!batchRemoveInput.trim()}
                className="flex items-center gap-2 bg-red-800 hover:bg-red-700 disabled:opacity-50 text-white rounded-lg px-4 py-1.5 text-sm transition-colors"
              >
                <MinusCircle size={14} />削除実行
              </button>
            </div>
          )}

          {showFrequency && frequencyData.length > 0 && (
            <div className="bg-gray-800 border border-gray-700 rounded-xl p-4">
              <div className="flex items-center justify-between mb-3">
                <h4 className="text-sm font-semibold text-emerald-300 flex items-center gap-2"><BarChart2 size={14} />タグ頻度 (上位40件)</h4>
                <button onClick={() => setShowFrequency(false)} className="text-gray-500 hover:text-gray-400"><X size={14} /></button>
              </div>
              <div className="grid grid-cols-2 gap-1 max-h-48 overflow-y-auto">
                {frequencyData.map(({ tag, count }) => (
                  <div key={tag} className="flex items-center justify-between text-xs py-0.5">
                    <button
                      className="text-gray-300 hover:text-indigo-300 truncate flex-1 text-left"
                      onClick={() => { setBatchFind(tag); setBatchReplace(""); setShowBatchReplace(true); setShowFrequency(false); }}
                      title="クリックでバッチ置換に設定"
                    >
                      {tag}
                    </button>
                    <span className="text-gray-500 ml-2 shrink-0">{count}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* キャプションフィルタ */}
          <div className="flex items-center gap-2">
            {(["all", "empty", "generated", "manual"] as const).map((f) => (
              <button
                key={f}
                onClick={() => setCaptionFilter(f)}
                className={[
                  "text-xs px-3 py-1 rounded-full border transition-colors",
                  captionFilter === f
                    ? "bg-violet-900 border-violet-700 text-violet-300"
                    : "bg-gray-800 border-gray-700 text-gray-500 hover:text-gray-400",
                ].join(" ")}
              >
                {{ all: "全て", empty: "未キャプション", generated: "WD14生成", manual: "手動" }[f]}
              </button>
            ))}
            <span className="text-xs text-gray-600 ml-1">{filteredCaptionItems.length}件</span>
          </div>

          {/* キャプション一覧グリッド */}
          {filteredCaptionItems.length === 0 ? (
            <div className="bg-gray-800 border border-gray-700 rounded-xl p-8 text-center">
              <Tag size={32} className="text-gray-600 mx-auto mb-2" />
              <p className="text-gray-400 text-sm">
                {captionStats.total === 0
                  ? "データセットにアイテムがありません。画像収集タブで画像を取り込んでください。"
                  : "該当するアイテムがありません。"}
              </p>
            </div>
          ) : (
            <div className="space-y-2">
              {filteredCaptionItems.map((item) => (
                <CaptionCard
                  key={item.id}
                  item={item}
                  isEditing={editingId === item.id}
                  editingText={editingId === item.id ? editingText : item.caption}
                  onStartEdit={() => { setEditingId(item.id); setEditingText(item.caption); }}
                  onChangeText={setEditingText}
                  onSave={async () => {
                    await saveCaption(item.id, editingText);
                    setEditingId(null);
                  }}
                  onCancel={() => setEditingId(null)}
                />
              ))}
            </div>
          )}
        </>
      )}

      {/* ─────────────────────────── ダッシュボードタブ ─────────────────────────── */}
      {activeTab === "dashboard" && (
        <>
          {/* 分析コントロール */}
          <div className="bg-gray-800 border border-gray-700 rounded-xl p-4 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Activity size={16} className="text-emerald-400" />
              <span className="text-sm font-semibold text-gray-200">Dataset Analysis</span>
              {dashboardData?.analyzed_at && (
                <span className="text-xs text-gray-500">
                  最終分析: {new Date(dashboardData.analyzed_at).toLocaleString("ja-JP")}
                </span>
              )}
            </div>
            <div className="flex gap-2 items-center">
              {analysisPolling && (
                <div className="flex items-center gap-1.5 text-xs text-emerald-400">
                  <Loader2 size={13} className="animate-spin" />
                  <span>{analysisStatus?.message ?? "分析中..."}</span>
                </div>
              )}
              <button
                onClick={() => void startAnalysis()}
                disabled={analysisPolling}
                className="flex items-center gap-1.5 bg-emerald-800 hover:bg-emerald-700 disabled:bg-gray-700 disabled:text-gray-500 text-white rounded-lg px-3 py-1.5 text-xs font-medium transition-colors"
              >
                {analysisPolling ? <Loader2 size={13} className="animate-spin" /> : <Activity size={13} />}
                {dashboardData?.analyzed ? "再分析" : "分析実行"}
              </button>
              <button onClick={() => void loadDashboard()} className="text-gray-500 hover:text-gray-400">
                <RefreshCw size={14} />
              </button>
            </div>
          </div>

          {!dashboardData ? (
            <div className="bg-gray-800 border border-gray-700 rounded-xl p-8 text-center">
              <Activity size={32} className="text-gray-600 mx-auto mb-2" />
              <p className="text-gray-400 text-sm">「分析実行」ボタンで Dataset Engineering を開始します</p>
              <p className="text-gray-600 text-xs mt-1">pHash重複検出・Quality Score算出・解像度分析</p>
            </div>
          ) : (
            <>
              {/* Quality Score + 基礎統計 */}
              <div className="grid grid-cols-5 gap-3">
                {/* Quality Score */}
                <div className={[
                  "col-span-1 rounded-xl p-4 flex flex-col items-center justify-center border",
                  dashboardData.quality_score == null ? "bg-gray-800 border-gray-700"
                  : dashboardData.quality_score >= 80 ? "bg-emerald-950 border-emerald-800"
                  : dashboardData.quality_score >= 60 ? "bg-amber-950 border-amber-800"
                  : "bg-red-950 border-red-800",
                ].join(" ")}>
                  <div className={[
                    "text-4xl font-black",
                    dashboardData.quality_score == null ? "text-gray-500"
                    : dashboardData.quality_score >= 80 ? "text-emerald-300"
                    : dashboardData.quality_score >= 60 ? "text-amber-300"
                    : "text-red-300",
                  ].join(" ")}>
                    {dashboardData.quality_score ?? "—"}
                  </div>
                  <div className="text-xs text-gray-500 mt-1">/ 100</div>
                  <div className="text-xs font-semibold text-gray-400 mt-1">Quality Score</div>
                </div>

                {/* 統計カード */}
                {[
                  { label: "総画像数", value: dashboardData.stats.total, color: "text-gray-200" },
                  { label: "キャプション率", value: `${dashboardData.stats.caption_rate}%`, color: dashboardData.stats.caption_rate >= 80 ? "text-emerald-400" : "text-amber-400" },
                  { label: "カラー/白黒", value: dashboardData.analyzed ? `${dashboardData.stats.color_count}/${dashboardData.stats.mono_count}` : "—", color: "text-blue-400" },
                  { label: "重複/類似", value: dashboardData.analyzed ? `${dashboardData.stats.duplicate_pairs}/${dashboardData.stats.similar_pairs}` : "—", color: dashboardData.stats.duplicate_pairs > 0 ? "text-red-400" : "text-emerald-400" },
                ].map(({ label, value, color }) => (
                  <div key={label} className="bg-gray-800 border border-gray-700 rounded-xl p-3 flex flex-col items-center justify-center text-center">
                    <div className={`text-2xl font-bold ${color}`}>{value}</div>
                    <div className="text-xs text-gray-500 mt-0.5">{label}</div>
                  </div>
                ))}
              </div>

              {/* 警告リスト */}
              {dashboardData.stats.warnings?.length > 0 && (
                <div className="bg-amber-950 border border-amber-800 rounded-xl p-4">
                  <div className="flex items-center gap-2 mb-2">
                    <AlertTriangle size={14} className="text-amber-400" />
                    <span className="text-sm font-semibold text-amber-300">警告 ({dashboardData.stats.warnings.length}件)</span>
                  </div>
                  <ul className="space-y-1">
                    {dashboardData.stats.warnings.map((w, i) => (
                      <li key={i} className="text-xs text-amber-200 flex items-start gap-1.5">
                        <span className="text-amber-500 mt-0.5">•</span>{w}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* 解像度分布 + アスペクト分布 */}
              <div className="grid grid-cols-2 gap-4">
                <DistributionBar
                  title="解像度分布"
                  total={dashboardData.stats.total}
                  items={[
                    { label: "高 (≥1024px²)", value: dashboardData.stats.resolution.high, color: "bg-emerald-500" },
                    { label: "中 (512-1023px²)", value: dashboardData.stats.resolution.medium, color: "bg-blue-500" },
                    { label: "低 (<512px²)", value: dashboardData.stats.resolution.low, color: "bg-red-500" },
                  ]}
                />
                <DistributionBar
                  title="アスペクト比分布"
                  total={dashboardData.stats.total}
                  items={[
                    { label: "Portrait", value: dashboardData.stats.aspect.portrait, color: "bg-violet-500" },
                    { label: "Landscape", value: dashboardData.stats.aspect.landscape, color: "bg-indigo-500" },
                    { label: "Square", value: dashboardData.stats.aspect.square, color: "bg-sky-500" },
                  ]}
                />
              </div>

              {/* 平均解像度 */}
              {dashboardData.analyzed && (
                <div className="bg-gray-800 border border-gray-700 rounded-xl px-4 py-3 flex items-center gap-4 text-sm">
                  <Layers size={14} className="text-gray-500" />
                  <span className="text-gray-400">平均解像度:</span>
                  <span className="text-gray-200 font-medium">{dashboardData.stats.avg_width} × {dashboardData.stats.avg_height} px</span>
                  {!dashboardData.stats.imagehash_available && (
                    <span className="ml-auto text-xs text-amber-400">⚠ ImageHash未インストール — 重複検出無効</span>
                  )}
                </div>
              )}

              {/* 類似グループ */}
              {dashboardData.similarity_groups.length > 0 && (
                <div className="bg-gray-800 border border-gray-700 rounded-xl overflow-hidden">
                  <div className="px-4 py-3 border-b border-gray-700 flex items-center gap-2">
                    <Copy size={14} className="text-red-400" />
                    <span className="text-sm font-semibold text-gray-200">
                      類似・重複グループ ({dashboardData.similarity_groups.length}件)
                    </span>
                  </div>
                  <div className="divide-y divide-gray-700">
                    {dashboardData.similarity_groups.map((group, idx) => (
                      <SimilarityGroupRow
                        key={idx}
                        group={group}
                        idx={idx}
                        expanded={expandedGroups.has(idx)}
                        onToggle={() => toggleGroup(idx)}
                      />
                    ))}
                  </div>
                </div>
              )}

              {dashboardData.analyzed && dashboardData.similarity_groups.length === 0 && (
                <div className="flex items-center gap-2 text-sm text-emerald-400 bg-emerald-950 border border-emerald-900 rounded-xl px-4 py-3">
                  <Check size={14} />
                  重複・類似画像は検出されませんでした
                </div>
              )}

              {/* ── §11 Character Leak + §12 Tag Categories ── */}
              <div className="bg-gray-800 border border-gray-700 rounded-xl p-4 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <AlertTriangle size={15} className="text-amber-400" />
                  <span className="text-sm font-semibold text-gray-200">§11 キャラリーク分析 / §12 タグカテゴリ</span>
                  {leakData && (
                    <span className={[
                      "text-xs px-2 py-0.5 rounded-full font-bold",
                      leakData.risk_level === "low" ? "bg-emerald-900 text-emerald-300"
                      : leakData.risk_level === "medium" ? "bg-amber-900 text-amber-300"
                      : leakData.risk_level === "high" ? "bg-red-900 text-red-300"
                      : "bg-gray-700 text-gray-400",
                    ].join(" ")}>
                      {{low: "LOW", medium: "MEDIUM", high: "HIGH", unknown: "—"}[leakData.risk_level]}
                    </span>
                  )}
                </div>
                <button
                  onClick={() => void loadCharacterLeak()}
                  disabled={leakLoading}
                  className="flex items-center gap-1.5 bg-amber-900 hover:bg-amber-800 disabled:bg-gray-700 disabled:text-gray-500 text-amber-100 rounded-lg px-3 py-1.5 text-xs font-medium transition-colors"
                >
                  {leakLoading ? <Loader2 size={13} className="animate-spin" /> : <AlertTriangle size={13} />}
                  {leakData ? "再分析" : "リーク分析実行"}
                </button>
              </div>

              {leakData && (
                <>
                  {/* §11 Character Leak Result */}
                  <div className="bg-gray-800 border border-gray-700 rounded-xl overflow-hidden">
                    <div className="px-4 py-3 border-b border-gray-700 flex items-center gap-3">
                      <AlertTriangle size={14} className="text-amber-400" />
                      <span className="text-sm font-semibold text-gray-200">Character Leak Analysis</span>
                      <span className="text-xs text-gray-500">{leakData.captioned}/{leakData.total}枚 解析済み</span>
                      <span className="text-xs text-gray-500 ml-auto">Risk Score: {leakData.risk_score}</span>
                    </div>

                    {leakData.warnings.length > 0 && (
                      <div className="px-4 py-3 border-b border-gray-700 space-y-1">
                        {leakData.warnings.map((w, i) => (
                          <div key={i} className={[
                            "text-xs flex items-start gap-1.5 px-2 py-1 rounded",
                            w.includes("[HIGH]") ? "bg-red-950 text-red-300"
                            : w.includes("[MEDIUM]") ? "bg-amber-950 text-amber-300"
                            : "bg-gray-700 text-gray-400",
                          ].join(" ")}>
                            <AlertTriangle size={11} className="mt-0.5 shrink-0" />
                            {w}
                          </div>
                        ))}
                      </div>
                    )}

                    {leakData.leaks.length === 0 && leakData.character_tags.length === 0 ? (
                      <div className="px-4 py-4 flex items-center gap-2 text-sm text-emerald-400">
                        <Check size={14} />
                        キャラクター固有特徴の偏りは検出されませんでした
                      </div>
                    ) : (
                      <div className="p-4 space-y-4">
                        {/* 特徴偏り */}
                        {leakData.leaks.length > 0 && (
                          <div>
                            <div className="text-xs font-semibold text-gray-400 mb-2">特徴偏り検出</div>
                            <div className="space-y-2">
                              {leakData.leaks.map((leak) => (
                                <div key={leak.feature} className="flex items-center gap-3">
                                  <span className={[
                                    "text-xs px-1.5 py-0.5 rounded font-bold shrink-0",
                                    leak.severity === "high" ? "bg-red-900 text-red-300" : "bg-amber-900 text-amber-300",
                                  ].join(" ")}>
                                    {leak.severity.toUpperCase()}
                                  </span>
                                  <span className="text-xs text-gray-400 w-20 shrink-0">{leak.feature}</span>
                                  <span className="text-xs text-gray-200 flex-1 truncate">「{leak.dominant}」</span>
                                  <div className="w-24 bg-gray-700 rounded-full h-1.5 shrink-0">
                                    <div
                                      className={`h-1.5 rounded-full ${leak.severity === "high" ? "bg-red-500" : "bg-amber-500"}`}
                                      style={{ width: `${Math.min(leak.ratio, 100)}%` }}
                                    />
                                  </div>
                                  <span className="text-xs text-gray-400 w-12 text-right shrink-0">{leak.ratio}%</span>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* WD14 キャラタグ */}
                        {leakData.character_tags.length > 0 && (
                          <div>
                            <div className="text-xs font-semibold text-gray-400 mb-2">WD14 キャラクタータグ検出</div>
                            <div className="flex flex-wrap gap-1.5">
                              {leakData.character_tags.map((ct) => (
                                <span
                                  key={ct.tag}
                                  className={[
                                    "text-xs px-2 py-0.5 rounded",
                                    ct.severity === "high" ? "bg-red-950 text-red-300 border border-red-800"
                                    : "bg-amber-950 text-amber-300 border border-amber-800",
                                  ].join(" ")}
                                  title={`${ct.count}枚 / ${ct.ratio}%`}
                                >
                                  {ct.tag} <span className="opacity-60">{ct.ratio}%</span>
                                </span>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                  </div>

                  {/* §12 Tag Categories */}
                  {categoryData && (
                    <div className="bg-gray-800 border border-gray-700 rounded-xl overflow-hidden">
                      <div className="px-4 py-3 border-b border-gray-700 flex items-center gap-2">
                        <BarChart2 size={14} className="text-indigo-400" />
                        <span className="text-sm font-semibold text-gray-200">Caption Categories (§12)</span>
                        <span className="text-xs text-gray-500">{categoryData.total_items}枚</span>
                      </div>
                      <div className="p-4 grid grid-cols-2 gap-3 sm:grid-cols-3">
                        {categoryData.categories.map((cat) => (
                          <div key={cat.category} className="space-y-1.5">
                            <div className="flex items-center justify-between">
                              <span className="text-xs font-semibold text-gray-400">{cat.category}</span>
                              <span className="text-xs text-gray-500">{cat.coverage_pct}%</span>
                            </div>
                            <div className="w-full bg-gray-700 rounded-full h-1.5">
                              <div
                                className="bg-indigo-500 h-1.5 rounded-full transition-all"
                                style={{ width: `${cat.coverage_pct}%` }}
                              />
                            </div>
                            <div className="flex flex-wrap gap-1">
                              {cat.top_tags.slice(0, 3).map(({ tag }) => (
                                <span key={tag} className="text-xs bg-gray-700 text-gray-400 px-1.5 py-0.5 rounded truncate max-w-full">
                                  {tag}
                                </span>
                              ))}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </>
              )}

              {/* ── §9.2-9.4 Distribution Analysis ── */}
              <div className="bg-gray-800 border border-gray-700 rounded-xl p-4 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Layers size={15} className="text-sky-400" />
                  <span className="text-sm font-semibold text-gray-200">§9.2-9.4 分布分析</span>
                  {distributionData && (
                    <span className="text-xs text-gray-500">{distributionData.captioned_items}/{distributionData.total_items}枚</span>
                  )}
                </div>
                <button
                  onClick={() => void loadDistribution()}
                  disabled={distributionLoading}
                  className="flex items-center gap-1.5 bg-sky-900 hover:bg-sky-800 disabled:bg-gray-700 disabled:text-gray-500 text-sky-100 rounded-lg px-3 py-1.5 text-xs font-medium transition-colors"
                >
                  {distributionLoading ? <Loader2 size={13} className="animate-spin" /> : <BarChart2 size={13} />}
                  {distributionData ? "再分析" : "分布分析実行"}
                </button>
              </div>

              {distributionData && (
                <div className="grid grid-cols-2 gap-4">
                  <DistributionTopBar
                    title="髪色分布"
                    items={distributionData.hair_color}
                    total={distributionData.captioned_items}
                    color="bg-amber-500"
                  />
                  <DistributionTopBar
                    title="髪型分布"
                    items={distributionData.hair_style}
                    total={distributionData.captioned_items}
                    color="bg-violet-500"
                  />
                  <DistributionTopBar
                    title="瞳色分布"
                    items={distributionData.eye_color}
                    total={distributionData.captioned_items}
                    color="bg-sky-500"
                  />
                  <DistributionTopBar
                    title="衣装分布"
                    items={distributionData.costume}
                    total={distributionData.captioned_items}
                    color="bg-emerald-500"
                  />
                </div>
              )}
            </>
          )}
        </>
      )}
    </div>
  );
}

function DistributionTopBar({
  title,
  items,
  total,
  color,
}: {
  title: string;
  items: { label: string; count: number; pct: number }[];
  total: number;
  color: string;
}) {
  const topItems = items.slice(0, 6);
  return (
    <div className="bg-gray-800 border border-gray-700 rounded-xl p-4">
      <h4 className="text-xs font-semibold text-gray-400 mb-3">{title}</h4>
      {topItems.length === 0 ? (
        <p className="text-xs text-gray-600">該当なし</p>
      ) : (
        <div className="space-y-2">
          {topItems.map(({ label, count, pct }) => (
            <div key={label}>
              <div className="flex justify-between text-xs text-gray-400 mb-1">
                <span className="truncate flex-1 mr-2">{label}</span>
                <span className="shrink-0">{count}枚 ({pct}%)</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-1.5">
                <div
                  className={`${color} h-1.5 rounded-full transition-all`}
                  style={{ width: `${pct}%` }}
                />
              </div>
            </div>
          ))}
          {items.length > 6 && (
            <p className="text-xs text-gray-600 mt-1">+{items.length - 6}件省略</p>
          )}
        </div>
      )}
    </div>
  );
}

function DistributionBar({
  title,
  total,
  items,
}: {
  title: string;
  total: number;
  items: { label: string; value: number; color: string }[];
}) {
  return (
    <div className="bg-gray-800 border border-gray-700 rounded-xl p-4">
      <h4 className="text-xs font-semibold text-gray-400 mb-3">{title}</h4>
      <div className="space-y-2.5">
        {items.map(({ label, value, color }) => {
          const pct = total > 0 ? (value / total) * 100 : 0;
          return (
            <div key={label}>
              <div className="flex justify-between text-xs text-gray-400 mb-1">
                <span>{label}</span>
                <span>{value}枚 ({pct.toFixed(0)}%)</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div
                  className={`${color} h-2 rounded-full transition-all`}
                  style={{ width: `${pct}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function SimilarityGroupRow({
  group,
  idx,
  expanded,
  onToggle,
}: {
  group: SimilarityGroup;
  idx: number;
  expanded: boolean;
  onToggle: () => void;
}) {
  const isDup = group.type === "duplicate";
  return (
    <div>
      <button
        onClick={onToggle}
        className="w-full flex items-center gap-3 px-4 py-2.5 hover:bg-gray-700/50 transition-colors text-left"
      >
        {expanded ? <ChevronDown size={13} className="text-gray-500" /> : <ChevronRight size={13} className="text-gray-500" />}
        <span className={`text-xs font-medium px-1.5 py-0.5 rounded ${isDup ? "bg-red-900 text-red-300" : "bg-amber-900 text-amber-300"}`}>
          {isDup ? "重複" : "類似"}
        </span>
        <span className="text-sm text-gray-300">グループ #{idx + 1}</span>
        <span className="text-xs text-gray-500">{group.item_ids.length}枚</span>
      </button>
      {expanded && (
        <div className="px-4 pb-3 flex flex-wrap gap-1.5">
          {group.item_ids.map((id) => (
            <span key={id} className="text-xs bg-gray-700 text-gray-400 px-2 py-0.5 rounded">
              #{id}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

function CaptionCard({
  item,
  isEditing,
  editingText,
  onStartEdit,
  onChangeText,
  onSave,
  onCancel,
}: {
  item: CaptionItem;
  isEditing: boolean;
  editingText: string;
  onStartEdit: () => void;
  onChangeText: (t: string) => void;
  onSave: () => Promise<void>;
  onCancel: () => void;
}) {
  const filename = item.file_path.split(/[\\/]/).pop() ?? item.file_path;
  const sourceColor: Record<string, string> = {
    wd14: "text-violet-400",
    manual: "text-indigo-400",
    file: "text-emerald-400",
    "": "text-gray-600",
  };

  return (
    <div className="bg-gray-800 border border-gray-700 rounded-xl p-3 flex gap-3 group">
      {/* サムネイル */}
      <div className="w-16 h-16 bg-gray-700 rounded-lg overflow-hidden shrink-0 flex items-center justify-center">
        <img
          src={`${API_BASE}/collector/thumbnail?path=${encodeURIComponent(item.file_path)}`}
          alt={filename}
          className="w-full h-full object-cover"
          onError={(e) => { (e.target as HTMLImageElement).style.display = "none"; }}
        />
      </div>

      {/* 情報 + キャプション */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-1">
          <span className="text-xs text-gray-400 truncate">{filename}</span>
          <span className="text-xs text-gray-600">{item.width}×{item.height}</span>
          {item.caption_source && (
            <span className={`text-xs ${sourceColor[item.caption_source] ?? "text-gray-600"}`}>
              [{item.caption_source || "未生成"}]
            </span>
          )}
          {!item.caption && (
            <span className="text-xs text-amber-500">⚠ 未キャプション</span>
          )}
        </div>

        {isEditing ? (
          <div className="space-y-1.5">
            <textarea
              value={editingText}
              onChange={(e) => onChangeText(e.target.value)}
              rows={3}
              autoFocus
              className="w-full bg-gray-700 border border-indigo-600 text-gray-200 rounded-lg px-2.5 py-1.5 text-xs resize-none focus:outline-none focus:border-indigo-500"
            />
            <div className="flex gap-1.5">
              <button
                onClick={() => void onSave()}
                className="flex items-center gap-1 text-xs bg-indigo-700 hover:bg-indigo-600 text-white rounded px-2 py-0.5 transition-colors"
              >
                <Check size={11} />保存
              </button>
              <button
                onClick={onCancel}
                className="flex items-center gap-1 text-xs bg-gray-700 hover:bg-gray-600 text-gray-400 rounded px-2 py-0.5 transition-colors"
              >
                <X size={11} />キャンセル
              </button>
            </div>
          </div>
        ) : (
          <div
            className="relative cursor-text"
            onClick={onStartEdit}
            title="クリックで編集"
          >
            {item.caption ? (
              <p className="text-xs text-gray-300 leading-relaxed line-clamp-2 group-hover:line-clamp-none pr-6">
                {item.caption}
              </p>
            ) : (
              <p className="text-xs text-gray-600 italic">キャプションなし - クリックして追加</p>
            )}
            <button className="absolute top-0 right-0 opacity-0 group-hover:opacity-100 transition-opacity">
              <Edit3 size={11} className="text-gray-500 hover:text-gray-300" />
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

function WorkflowBtn({
  step,
  label,
  icon,
  color,
  onClick,
}: {
  step: number;
  label: string;
  icon: React.ReactNode;
  color: string;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={`flex flex-col items-center gap-2 ${color} text-white rounded-lg p-3 text-xs font-medium transition-colors`}
    >
      <span className="w-5 h-5 rounded-full bg-white/20 flex items-center justify-center text-xs font-bold">{step}</span>
      {icon}
      {label}
    </button>
  );
}
