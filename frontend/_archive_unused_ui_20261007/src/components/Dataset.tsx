import { useState, useMemo, useEffect, useCallback, useRef, type DragEvent } from "react";
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
  Scissors,
  ClipboardPaste,
  Layers,
  ChevronDown,
  ChevronUp,
  ChevronRight,
  Images,
  PenTool,
  Sparkles,
  Brain,
  Globe,
  ThumbsUp,
  ThumbsDown,
  RotateCcw,
  GripVertical,
  ShieldOff,
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
  SuggestionsResult,
  SuggestionFeedbackPayload,
  DatasetMixer,
  MixerWeights,
  MixerFeatureKey,
  OutfitProfile,
  ClassifiedItem,
  ClassificationResult,
} from "../types";

type Props = {
  projects: Project[];
  selectedProjectId: number | null;
  selectedProject: Project | null;
  onSelectProject: (id: number | null) => void;
  onRefresh: () => Promise<void>;
  showError: (msg: string) => void;
  showNotice: (msg: string) => void;
  clearMessages: () => void;
};

type ActiveTab = "collect" | "caption" | "dashboard" | "preprocess" | "mixer";

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


function OutfitClassifier({
  projectId,
  captionItems,
}: {
  projectId: number;
  captionItems: CaptionItem[];
}) {
  const [profiles, setProfiles] = useState<OutfitProfile[]>([]);
  const [classifying, setClassifying] = useState(false);
  const [result, setResult] = useState<ClassificationResult | null>(null);
  const [applying, setApplying] = useState(false);
  const [applied, setApplied] = useState(false);
  const [pickerOpen, setPickerOpen] = useState<string | null>(null);
  const [minScore, setMinScore] = useState(0.15);
  const [colorWeight, setColorWeight] = useState(0.4);
  const [dragOver, setDragOver] = useState<string | null>(null);
  const [saveStatus, setSaveStatus] = useState<"saved" | "unsaved" | "saving">("saved");
  const dragItemRef = useRef<{ item: ClassifiedItem; source: string | null } | null>(null);
  const saveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // プロジェクト切り替え時にプロファイルをロード
  useEffect(() => {
    if (!projectId) return;
    apiGet<{ profiles: OutfitProfile[] }>(`/tags/outfit-profiles/${projectId}`)
      .then((data) => {
        setProfiles(data.profiles ?? []);
        setSaveStatus("saved");
      })
      .catch(() => {});
    setResult(null);
    setApplied(false);
  }, [projectId]);

  // プロファイル変更時にデバウンス保存（1.5秒後）
  const saveProfiles = useCallback((updated: OutfitProfile[]) => {
    if (saveTimerRef.current) clearTimeout(saveTimerRef.current);
    setSaveStatus("unsaved");
    saveTimerRef.current = setTimeout(async () => {
      setSaveStatus("saving");
      try {
        await apiPost(`/tags/outfit-profiles/${projectId}`, { profiles: updated }, "PUT");
        setSaveStatus("saved");
      } catch {
        setSaveStatus("unsaved");
      }
    }, 1500);
  }, [projectId]);

  const addProfile = () => {
    setProfiles((prev) => {
      const next = [...prev, { id: Math.random().toString(36).slice(2), name: "", trigger_word: "", ref_item_ids: [] }];
      saveProfiles(next);
      return next;
    });
    setResult(null);
    setApplied(false);
  };

  const removeProfile = (id: string) => {
    setProfiles((prev) => {
      const next = prev.filter((p) => p.id !== id);
      saveProfiles(next);
      return next;
    });
    setResult(null);
    setApplied(false);
  };

  const updateProfile = (id: string, field: keyof OutfitProfile, value: unknown) => {
    setProfiles((prev) => {
      const next = prev.map((p) => (p.id === id ? { ...p, [field]: value } : p));
      saveProfiles(next);
      return next;
    });
    setResult(null);
    setApplied(false);
  };

  const toggleRef = (profileId: string, itemId: number) => {
    setProfiles((prev) => {
      const next = prev.map((p) => {
        if (p.id !== profileId) return p;
        const has = p.ref_item_ids.includes(itemId);
        return {
          ...p,
          ref_item_ids: has
            ? p.ref_item_ids.filter((id) => id !== itemId)
            : [...p.ref_item_ids, itemId],
        };
      });
      saveProfiles(next);
      return next;
    });
  };

  const classify = async () => {
    if (!profiles.length) return;
    setClassifying(true);
    setResult(null);
    setApplied(false);
    try {
      const data = await apiPost<ClassificationResult>("/tags/auto-classify", {
        project_id: projectId,
        profiles: profiles.map((p) => ({
          name: p.name,
          trigger_word: p.trigger_word,
          ref_item_ids: p.ref_item_ids,
        })),
        min_score: minScore,
        color_weight: colorWeight,
      });
      setResult(data);
    } catch (e) {
      console.error(e);
    } finally {
      setClassifying(false);
    }
  };

  const applyClassification = async () => {
    if (!result) return;
    setApplying(true);
    try {
      const assignments: { item_id: number; trigger_word: string }[] = [];
      for (const rp of result.profiles) {
        for (const item of rp.items) {
          if (!item.excluded) {
            assignments.push({ item_id: item.id, trigger_word: rp.trigger_word });
          }
        }
      }
      await apiPost("/tags/apply-profile-classification", {
        project_id: projectId,
        assignments,
        overwrite: false,
      });
      setApplied(true);
    } catch (e) {
      console.error(e);
    } finally {
      setApplying(false);
    }
  };

  const toggleExclude = (profileName: string, itemId: number) => {
    if (!result) return;
    setResult((prev) => {
      if (!prev) return prev;
      return {
        ...prev,
        profiles: prev.profiles.map((rp) => {
          if (rp.name !== profileName) return rp;
          return {
            ...rp,
            items: rp.items.map((item) =>
              item.id === itemId ? { ...item, excluded: !item.excluded } : item
            ),
          };
        }),
      };
    });
  };

  const moveItem = (itemId: number, source: string | null, target: string | null) => {
    if (source === target) return;
    setResult((prev) => {
      if (!prev) return prev;
      let moved: ClassifiedItem | undefined;
      const newProfiles = prev.profiles.map((rp) => {
        if (rp.name !== source) return rp;
        const found = rp.items.find((i) => i.id === itemId);
        if (found) moved = { ...found, excluded: false };
        return { ...rp, items: rp.items.filter((i) => i.id !== itemId) };
      });
      let newUnclassified = [...prev.unclassified];
      if (source === null) {
        const idx = newUnclassified.findIndex((i) => i.id === itemId);
        if (idx >= 0) { moved = { ...newUnclassified[idx], excluded: false }; newUnclassified.splice(idx, 1); }
      }
      if (!moved) return prev;
      if (target === null) {
        newUnclassified = [...newUnclassified, moved];
      } else {
        return {
          ...prev,
          profiles: newProfiles.map((rp) =>
            rp.name === target ? { ...rp, items: [...rp.items, moved!] } : rp
          ),
          unclassified: newUnclassified,
        };
      }
      return { ...prev, profiles: newProfiles, unclassified: newUnclassified };
    });
    setApplied(false);
  };

  const onDragStart = (item: ClassifiedItem, source: string | null) => {
    dragItemRef.current = { item, source };
  };

  const onDropZone = (e: DragEvent<HTMLDivElement>, target: string | null) => {
    e.preventDefault();
    setDragOver(null);
    if (!dragItemRef.current) return;
    const { item, source } = dragItemRef.current;
    dragItemRef.current = null;
    moveItem(item.id, source, target);
  };

  const totalToApply = result
    ? result.profiles.reduce(
        (sum, rp) => sum + rp.items.filter((i) => !i.excluded).length,
        0
      )
    : 0;

  return (
    <div className="mt-6 border border-gray-700 rounded-lg p-4">
      <div className="flex items-center justify-between mb-1">
        <h3 className="text-sm font-semibold text-gray-200">🎭 衣装プロファイル自動分類</h3>
        <span className={`text-xs px-2 py-0.5 rounded-full ${
          saveStatus === "saved" ? "text-green-400 bg-green-900/30" :
          saveStatus === "saving" ? "text-yellow-400 bg-yellow-900/30" :
          "text-gray-400 bg-gray-700"
        }`}>
          {saveStatus === "saved" ? "✓ 保存済み" : saveStatus === "saving" ? "保存中..." : "未保存"}
        </span>
      </div>
      <p className="text-xs text-gray-400 mb-4">
        WD14タグ＋カラーヒストグラムでプロファイルに自動振り分けます。プロジェクトに自動保存されます。
      </p>

      <div className="space-y-3">
        {profiles.map((profile) => (
          <div key={profile.id} className="bg-gray-800 rounded p-3 space-y-2">
            <div className="flex gap-2">
              <input
                className="bg-gray-700 border border-gray-600 text-gray-200 rounded px-2 py-1 text-xs flex-1"
                placeholder="衣装名（例: 制服）"
                value={profile.name}
                onChange={(e) => updateProfile(profile.id, "name", e.target.value)}
              />
              <input
                className="bg-gray-700 border border-gray-600 text-gray-200 rounded px-2 py-1 text-xs flex-1"
                placeholder="トリガーワード（例: school uniform ver）"
                value={profile.trigger_word}
                onChange={(e) => updateProfile(profile.id, "trigger_word", e.target.value)}
              />
              <button
                className="text-xs text-red-400 hover:text-red-300 px-2"
                onClick={() => removeProfile(profile.id)}
              >
                ✕
              </button>
            </div>
            <div className="flex items-center gap-2 flex-wrap">
              {profile.ref_item_ids.map((refId) => {
                const refItem = captionItems.find((c) => c.id === refId);
                if (!refItem) return null;
                return (
                  <div key={refId} className="relative group">
                    <img
                      src={`${API_BASE}/collector/thumbnail?path=${encodeURIComponent(refItem.file_path)}&size=56`}
                      className="w-14 h-14 object-cover rounded cursor-pointer"
                      onClick={() => toggleRef(profile.id, refId)}
                    />
                    <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 rounded flex items-center justify-center text-white text-xs">
                      ✕
                    </div>
                  </div>
                );
              })}
              <button
                className="w-14 h-14 bg-gray-700 border border-dashed border-gray-500 rounded text-xs text-gray-400 hover:text-gray-200 hover:border-gray-400"
                onClick={() => setPickerOpen(profile.id)}
              >
                ＋
              </button>
            </div>
          </div>
        ))}
      </div>

      <button
        className="mt-3 text-xs text-indigo-400 hover:text-indigo-300 border border-dashed border-indigo-600 rounded px-3 py-1"
        onClick={addProfile}
      >
        ＋ プロファイルを追加
      </button>

      <div className="mt-3 flex flex-wrap items-center gap-x-6 gap-y-2">
        <div className="flex items-center gap-2">
          <label className="text-xs text-gray-400">最低一致スコア:</label>
          <input
            type="number"
            step="0.05"
            min="0"
            max="1"
            className="bg-gray-700 border border-gray-600 text-gray-200 rounded px-2 py-1 text-xs w-20"
            value={minScore}
            onChange={(e) => setMinScore(parseFloat(e.target.value) || 0)}
          />
        </div>
        <div className="flex items-center gap-2">
          <label className="text-xs text-gray-400">
            カラー重み: <span className="text-gray-200">{Math.round(colorWeight * 100)}%</span>
          </label>
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            className="w-28 accent-indigo-500"
            value={colorWeight}
            onChange={(e) => setColorWeight(parseFloat(e.target.value))}
          />
          <span className="text-xs text-gray-500">
            タグ {Math.round((1 - colorWeight) * 100)}% / 色 {Math.round(colorWeight * 100)}%
          </span>
        </div>
      </div>

      <button
        className="mt-3 bg-indigo-600 hover:bg-indigo-500 text-white text-xs rounded px-4 py-2 disabled:opacity-50"
        onClick={classify}
        disabled={classifying || profiles.length === 0}
      >
        {classifying ? "分類中..." : "🔍 自動分類を実行（タグ＋カラー）"}
      </button>

      {result && (
        <div className="mt-5 border-t border-gray-700 pt-4">
          <h4 className="text-sm font-semibold text-gray-200 mb-1">
            分類結果プレビュー（合計 {result.total} 枚）
          </h4>
          <p className="text-xs text-gray-500 mb-3">画像をドラッグして別プロファイルへ移動 / クリックで除外</p>
          {result.profiles.map((rp) => (
            <div
              key={rp.name}
              className={`mb-4 rounded-lg p-2 border transition-colors ${
                dragOver === rp.name ? "border-indigo-500 bg-indigo-900/20" : "border-gray-700"
              }`}
              onDragOver={(e) => { e.preventDefault(); setDragOver(rp.name); }}
              onDragLeave={() => setDragOver(null)}
              onDrop={(e) => onDropZone(e, rp.name)}
            >
              <p className="text-xs font-medium text-gray-300 mb-2">
                📁 {rp.name} ({rp.trigger_word}) — {rp.items.length} 枚
              </p>
              <div className="flex flex-wrap gap-1 min-h-[3.5rem]">
                {rp.items.map((item) => (
                  <div
                    key={item.id}
                    draggable
                    onDragStart={() => onDragStart(item, rp.name)}
                    onDragEnd={() => setDragOver(null)}
                    className={`relative cursor-grab active:cursor-grabbing select-none ${item.excluded ? "opacity-30" : ""}`}
                    onClick={() => toggleExclude(rp.name, item.id)}
                    title={`スコア: ${item.score}  クリック:除外  ドラッグ:移動`}
                  >
                    <img
                      src={`${API_BASE}/collector/thumbnail?path=${encodeURIComponent(item.file_path)}&size=112`}
                      className="w-14 h-14 object-cover rounded pointer-events-none"
                    />
                    {item.excluded && (
                      <div className="absolute inset-0 flex items-center justify-center text-white text-lg font-bold bg-black/40 rounded">
                        ✕
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          ))}
          <div
            className={`mb-4 rounded-lg p-2 border transition-colors ${
              dragOver === "unclassified" ? "border-amber-500 bg-amber-900/20" : "border-gray-700 border-dashed"
            }`}
            onDragOver={(e) => { e.preventDefault(); setDragOver("unclassified"); }}
            onDragLeave={() => setDragOver(null)}
            onDrop={(e) => onDropZone(e, null)}
          >
            <p className="text-xs font-medium text-gray-400 mb-2">
              ❓ 未分類 — {result.unclassified.length} 枚（どのプロファイルにも一致しなかった）
            </p>
            <div className="flex flex-wrap gap-1 min-h-[3.5rem]">
              {result.unclassified.map((item) => (
                <div
                  key={item.id}
                  draggable
                  onDragStart={() => onDragStart(item, null)}
                  onDragEnd={() => setDragOver(null)}
                  className="cursor-grab active:cursor-grabbing select-none"
                  title="ドラッグしてプロファイルへ移動"
                >
                  <img
                    src={`${API_BASE}/collector/thumbnail?path=${encodeURIComponent(item.file_path)}&size=112`}
                    className="w-14 h-14 object-cover rounded opacity-50 pointer-events-none"
                  />
                </div>
              ))}
            </div>
          </div>
          <button
            className="mt-2 bg-green-700 hover:bg-green-600 text-white text-xs rounded px-4 py-2 disabled:opacity-50"
            onClick={applyClassification}
            disabled={applying || applied || totalToApply === 0}
          >
            {applied
              ? "✅ 適用済み"
              : applying
              ? "適用中..."
              : `✅ キャプションに適用（合計 ${totalToApply} 枚）`}
          </button>
        </div>
      )}

      {pickerOpen !== null && (
        <div
          className="fixed inset-0 z-50 bg-black/60 flex items-center justify-center"
          onClick={() => setPickerOpen(null)}
        >
          <div
            className="bg-gray-800 border border-gray-600 rounded-lg p-4 w-[480px] max-h-[60vh] overflow-y-auto"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex justify-between items-center mb-3">
              <h4 className="text-sm font-semibold text-gray-200">参照画像を選択</h4>
              <button className="text-gray-400 hover:text-gray-200" onClick={() => setPickerOpen(null)}>
                ✕
              </button>
            </div>
            <div className="grid grid-cols-6 gap-1">
              {captionItems.map((item) => {
                const profile = profiles.find((p) => p.id === pickerOpen);
                const selected = profile?.ref_item_ids.includes(item.id) ?? false;
                return (
                  <div
                    key={item.id}
                    className={`relative cursor-pointer rounded overflow-hidden ${selected ? "ring-2 ring-indigo-500" : ""}`}
                    onClick={() => toggleRef(pickerOpen, item.id)}
                  >
                    <img
                      src={`${API_BASE}/collector/thumbnail?path=${encodeURIComponent(item.file_path)}&size=56`}
                      className="w-full h-14 object-cover"
                    />
                    {selected && (
                      <div className="absolute inset-0 bg-indigo-500/30 flex items-center justify-center text-white text-xs font-bold">
                        ✓
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
            <button
              className="mt-3 bg-indigo-600 hover:bg-indigo-500 text-white text-xs rounded px-3 py-1.5"
              onClick={() => setPickerOpen(null)}
            >
              完了
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

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
  const lsKey = `repeatPreparedPath_${selectedProject?.id ?? ""}`;
  const [repeatPreparedPath, setRepeatPreparedPathRaw] = useState(() => {
    try { return localStorage.getItem(`repeatPreparedPath_${selectedProject?.id ?? ""}`) ?? ""; } catch { return ""; }
  });
  function setRepeatPreparedPath(v: string) {
    setRepeatPreparedPathRaw(v);
    try { if (v) localStorage.setItem(lsKey, v); else localStorage.removeItem(lsKey); } catch {}
  }
  const [dragActive, setDragActive] = useState(false);
  const [scanning, setScanning] = useState(false);
  const [scanAbortRef, setScanAbortRef] = useState<AbortController | null>(null);

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

  // ── プレビュー状態 ──
  const [selectedPreviewItem, setSelectedPreviewItem] = useState<ScanItem | null>(null);
  const [contextMenuPos, setContextMenuPos] = useState<{ x: number; y: number } | null>(null);

  // ── プリプロセッシング状態 ──
  const [preprocessingMode, setPreprocessingMode] = useState<"resolution" | "text_removal" | "leak_masking">("resolution");
  const [resolutionTarget, setResolutionTarget] = useState("768");
  const [keepAspectRatio, setKeepAspectRatio] = useState(true);
  const [preprocessingRunning, setPreprocessingRunning] = useState(false);
  const [preprocessingProgress, setPreprocessingProgress] = useState(0);
  const [preprocessingStep, setPreprocessingStep] = useState("");
  const [preprocessingQueued, setPreprocessingQueued] = useState(false);
  const [textRemovalGrow, setTextRemovalGrow] = useState(8);
  const [useUpscale, setUseUpscale] = useState(false);
  const [upscaleModel, setUpscaleModel] = useState("");
  const [comfyUpscaleModels, setComfyUpscaleModels] = useState<string[]>([]);
  const [comfyReady, setComfyReady] = useState<boolean | null>(null);
  const [useCaption, setUseCaption] = useState(false);
  const [captionGeneralThresh, setCaptionGeneralThresh] = useState(0.35);
  const [captionCharacterThresh, setCaptionCharacterThresh] = useState(0.85);
  const [captionRemoveCharacterTags, setCaptionRemoveCharacterTags] = useState(false);
  const [pipelineItems, setPipelineItems] = useState<{ item_id: number; file_name: string; status: string; error: string | null; output_path?: string }[]>([]);

  // ── 提案機能状態 ──
  const [suggestionsRunning, setSuggestionsRunning] = useState(false);
  const [suggestionsData, setSuggestionsData] = useState<SuggestionsResult | null>(null);
  const [suggestionPolling, setSuggestionPolling] = useState(false);
  const [suggestionMode, setSuggestionMode] = useState<"fast" | "balanced" | "accurate">("balanced");
  const [suggestionStatusMsg, setSuggestionStatusMsg] = useState("");
  const [suggestionStep, setSuggestionStep] = useState(0);
  const [suggestionElapsed, setSuggestionElapsed] = useState(0);
  const [suggestionStartAt, setSuggestionStartAt] = useState<number | null>(null);

  // ── フィードバックループ状態 ──
  const [likedUrls, setLikedUrls] = useState<Set<string>>(new Set());
  const [rejectedUrls, setRejectedUrls] = useState<Set<string>>(new Set());
  const [feedbackRunning, setFeedbackRunning] = useState(false);
  const [feedbackPolling, setFeedbackPolling] = useState(false);
  const [feedbackStatusMsg, setFeedbackStatusMsg] = useState("");

  // ── キャプション並べ替え状態 ──
  const [captionDragFromId, setCaptionDragFromId] = useState<number | null>(null);
  const [captionDragOverId, setCaptionDragOverId] = useState<number | null>(null);

  // ── Dataset Mixer 状態 (§10) ──
  const DEFAULT_WEIGHTS: MixerWeights = {
    face: 50, expression: 50, hair: 50, costume: 50, accessory: 50,
    background: 50, composition: 50, line: 50, color: 50, lighting: 50, mood: 50,
  };
  const [mixerWeights, setMixerWeights] = useState<MixerWeights>(DEFAULT_WEIGHTS);
  const [mixerSaving, setMixerSaving] = useState(false);
  const [mixerSaved, setMixerSaved] = useState(false);

  // ── 画像検索パネル (Pinterest webview) ──
  const pinterestRef = useRef<any>(null);
  const pixivRef2 = useRef<any>(null); // zoom 用参照（既存 pixivRef と別名）
  const [pinterestReady, setPinterestReady] = useState(false);
  const [pinterestAddedCount, setPinterestAddedCount] = useState(0);
  const [webviewZoom, setWebviewZoom] = useState(0); // 0 = 100%, +1 = 110%, -1 = 90% ...
  const [uiZoom, setUiZoom] = useState(1.0);
  const webviewPanelRef = useRef<HTMLDivElement>(null);

  // 画像検索ソースのタブ切替（Pinterest / Pixiv を1パネルに統合）
  const [browseSource, setBrowseSource] = useState<"pinterest" | "pixiv">("pinterest");
  // ワークフローのフォルダ詳細設定の折りたたみ（既定: 閉）
  const [showFolderSettings, setShowFolderSettings] = useState(false);

  // ── 画像検索パネル (Pixiv webview) ──
  const pixivRef = useRef<any>(null);
  const [pixivReady, setPixivReady] = useState(false);
  const [pixivAddedCount, setPixivAddedCount] = useState(0);
  // selectedProject を ref で持ち stale closure を防ぐ
  const selectedProjectRef = useRef(selectedProject);
  useEffect(() => { selectedProjectRef.current = selectedProject; }, [selectedProject]);

  useEffect(() => {
    // CSS zoom でブラウザ / Electron 両対応
    document.documentElement.style.zoom = `${uiZoom * 100}%`;
    return () => {
      document.documentElement.style.zoom = '';
    };
  }, [uiZoom]);

  // ヘッダーはそのまま（Pinterest の検索バーを使う）
  // 広告ピン非表示 + ポップアップ・クッキーバナーだけ非表示
  const PINTEREST_HIDE_CSS = `
    [data-test-id="interstitialContainer"],[data-test-id="unauth-modal"],
    #AccountCreationInterrupt,.visionInterstitial,[data-test-id="registerModalContainer"],
    [data-test-id="cookieBanner"],[data-test-id="consent-dialog"],.cookiesConsent,
    [class*="CookieBanner"],[id*="cookie-banner"]{display:none!important}
    [data-test-id="promotedPin"],[data-test-id="promoted-story"],
    [data-test-id="grid-item-story"],[data-test-id="story-pin-grid-item"]{display:none!important}
    .lbp-btn{position:absolute!important;top:8px!important;left:8px!important;width:40px!important;height:40px!important;
      background:rgba(20,20,20,.82)!important;color:#fff!important;border:2px solid rgba(255,255,255,.45)!important;
      border-radius:50%!important;font-size:22px!important;line-height:36px!important;text-align:center!important;
      cursor:pointer!important;z-index:9999!important;pointer-events:auto!important;visibility:hidden!important;}
    .lbp-btn:hover{background:rgba(249,115,22,.95)!important;border-color:transparent!important;transform:scale(1.1)!important;visibility:visible!important;}
    #lbp-detail-btn{position:fixed!important;bottom:24px!important;right:24px!important;width:52px!important;height:52px!important;
      background:rgba(249,115,22,.93)!important;color:#fff!important;border:none!important;
      border-radius:50%!important;font-size:28px!important;line-height:52px!important;text-align:center!important;
      cursor:pointer!important;z-index:99999!important;box-shadow:0 4px 16px rgba(0,0,0,.4)!important;
      transition:transform .15s!important;}
    #lbp-detail-btn:hover{transform:scale(1.12)!important;}
  `;
  // 広告スキップ + グリッド左上ボタン + ピン詳細クローズアップのボタン
  const PINTEREST_INJECT_JS = `(function(){
    /* CSS を <style> タグで注入（insertCSS は SPA ナビ後に効かないため） */
    if(!document.getElementById('lbp-style')){
      var s=document.createElement('style');s.id='lbp-style';
      s.textContent=[
        '[data-test-id="promotedPin"],[data-test-id="promoted-story"],[data-test-id="story-pin-grid-item"]{display:none!important}',
        '#lbp-detail-btn{position:fixed!important;bottom:24px!important;right:24px!important;width:52px!important;height:52px!important;',
        'background:rgba(249,115,22,.93)!important;color:#fff!important;border:none!important;border-radius:50%!important;',
        'font-size:28px!important;line-height:52px!important;text-align:center!important;cursor:pointer!important;',
        'z-index:99999!important;box-shadow:0 4px 16px rgba(0,0,0,.4)!important;}',
        '#lbp-detail-btn:hover{transform:scale(1.12)!important;}'
      ].join('');
      document.head.appendChild(s);
    }
    var AD_TEXTS=['サイトにアクセス','Visit site','Promoted by','スポンサー','Sponsored'];
    /* pinimg.com 画像を持つ最小の pin カードコンテナを動的に発見 */
    function findPinCard(img){
      var el=img.parentElement;
      for(var i=0;i<12;i++){
        if(!el||el===document.body)break;
        if(el.querySelector('a[href*="/pin/"]'))return el;
        el=el.parentElement;
      }
      return null;
    }
    /* 広告判定 — data-test-id ベース + 構造/属性ベースの多層判定 */
    function isAd(card){
      if(!card)return false;
      /* 1) data-test-id ベース（Pinterest が付与している場合） */
      if(card.querySelector('[data-test-id="promotedTag"],[data-test-id="storyPinTag"],[data-test-id="siteVisitButton"],[data-test-id="ad-disclosure"],[data-test-id="ads-badge"]'))return true;
      if(card.closest('[data-test-id="promotedPin"],[data-test-id="promoted-story"],[data-test-id="story-pin-grid-item"]'))return true;
      /* 2) 属性ベース — data-promoted / aria-label に「広告」含む */
      if(card.hasAttribute('data-promoted'))return true;
      var ariaLabel=(card.getAttribute('aria-label')||'').toLowerCase();
      if(ariaLabel.includes('promoted')||ariaLabel.includes('sponsored')||ariaLabel.includes('広告'))return true;
      /* 3) rel="sponsored" リンクが含まれる — Google/W3C 標準の広告マーク */
      if(card.querySelector('a[rel~="sponsored"]'))return true;
      /* 4) テキストベース（最後の砦、誤検知リスクあり） */
      var txt=card.textContent||'';
      for(var i=0;i<AD_TEXTS.length;i++){if(txt.includes(AD_TEXTS[i]))return true;}
      return false;
    }
    /* img ごとにボタンを追加 */
    function processImg(img){
      if(img.dataset.lbpDone)return;
      if(!img.src||!img.src.includes('pinimg.com'))return;
      var card=findPinCard(img);
      if(!card)return;
      if(isAd(card)){card.style.display='none';img.dataset.lbpDone='1';return;}
      if(card.querySelector('.lbp-btn')){img.dataset.lbpDone='1';return;}
      /* ボタンはアンカー内部ではなく card 直下に置く。
         アンカー内に置くと Pinterest の capture フェーズのナビゲーションハンドラに
         クリックを奪われ「押せない（ピンを開くだけ）」状態になるため。 */
      if(getComputedStyle(card).position==='static'){card.style.position='relative';}
      var b=document.createElement('button');
      b.className='lbp-btn';b.textContent='+';b.title='データセットに追加';
      /* 常時表示（opacity 0 強制対策で visibility/opacity 両方を明示）。z-index は最大値付近。 */
      b.style.cssText='position:absolute!important;top:8px!important;left:8px!important;'+
        'width:38px!important;height:38px!important;background:rgba(20,20,20,.88)!important;'+
        'color:#fff!important;border:2px solid rgba(255,255,255,.6)!important;'+
        'border-radius:50%!important;font-size:22px!important;line-height:34px!important;'+
        'text-align:center!important;cursor:pointer!important;z-index:2147483646!important;'+
        'pointer-events:auto!important;visibility:visible!important;opacity:.92!important;'+
        'box-shadow:0 2px 8px rgba(0,0,0,.5)!important;';
      function addThis(e){
        if(e){e.preventDefault();e.stopPropagation();if(e.stopImmediatePropagation)e.stopImmediatePropagation();}
        var url=img.src.replace(/\\/\\d+x\\//,'/736x/').replace(/\\/\\d+x$/,'/736x');
        console.log('LBP_ADD:'+url);
        b.textContent='\\u2713';
        b.style.setProperty('background','rgba(16,185,129,.95)','important');
        b.style.setProperty('border','none','important');
        b.style.setProperty('opacity','1','important');
        return false;
      }
      /* pointerdown を capture で握り、アンカーへ伝播する前にクリックを確定させる */
      b.addEventListener('pointerdown',function(e){e.preventDefault();e.stopPropagation();if(e.stopImmediatePropagation)e.stopImmediatePropagation();},true);
      b.addEventListener('click',addThis,true);
      card.appendChild(b);
      img.dataset.lbpDone='1';
    }
    function getDetailImg(){
      /* data-test-id で探す — 直リンク URL (/pin/xxx/) とモーダル両対応 */
      var selectors=[
        '[data-test-id="story-pin-image-block"] img[src*="pinimg.com"]',
        '[data-test-id="closeup-image-main"] img[src*="pinimg.com"]',
        '[data-test-id="pin-closeup-image"] img[src*="pinimg.com"]',
        '[data-test-id="closeup-body"] img[src*="pinimg.com"]',
        '[data-test-id="pin-detail-image"] img[src*="pinimg.com"]',
        '[data-test-id="pinDetail"] img[src*="pinimg.com"]',
        '[data-test-id="pin-closeup"] img[src*="pinimg.com"]',
        '[role="dialog"] img[src*="pinimg.com"]'
      ];
      for(var i=0;i<selectors.length;i++){
        var el=document.querySelector(selectors[i]);
        if(el)return el;
      }
      /* フォールバック: URL が /pin/ で始まるなら最大幅の pinimg 画像（アバター除く） */
      if(location.pathname.includes('/pin/')){
        var imgs=document.querySelectorAll('img[src*="pinimg.com"]');
        var best=null,bestW=0;
        imgs.forEach(function(img){
          var dt=img.closest('[data-test-id]');
          var testId=dt?dt.getAttribute('data-test-id'):'';
          if(testId.includes('avatar')||testId.includes('non-story'))return;
          if(img.naturalWidth>bestW){bestW=img.naturalWidth;best=img;}
        });
        if(best&&bestW>300)return best;
      }
      return null;
    }
    function syncDetailBtn(){
      var existing=document.getElementById('lbp-detail-btn');
      var closeupImg=getDetailImg();
      if(closeupImg){
        var d=existing||document.createElement('button');
        if(!existing){d.id='lbp-detail-btn';d.title='この画像をデータセットに追加';document.body.appendChild(d);}
        d.textContent='+';
        d.onclick=function(){
          var img=getDetailImg();
          if(!img)return;
          var url=img.src.replace(/\\/\\d+x\\//,'/736x/').replace(/\\/\\d+x$/,'/736x');
          console.log('LBP_ADD:'+url);
          d.textContent='✓';setTimeout(function(){d.textContent='+';},1500);
        };
      } else if(existing){
        existing.remove();
      }
    }
    function scan(){
      /* data-test-id="pin" に依存せず pinimg.com 画像すべてを処理 */
      document.querySelectorAll('img[src*="pinimg.com"]').forEach(processImg);
      /* 広告テキスト持ちのカードを追加で非表示 */
      document.querySelectorAll('[data-test-id="siteVisitButton"],[data-test-id="ad-disclosure"]').forEach(function(el){
        var card=el.closest('div[data-grid-item],[data-test-id="grid-item"],[data-test-id="pin"],li,article');
        if(card)card.style.display='none';
      });
      syncDetailBtn();
    }
    scan();
    /* subtree:true だと重いので attributes:false で軽量化 */
    new MutationObserver(function(){scan();}).observe(document.body,{childList:true,subtree:true,attributes:false});
  })();`;

  // Pixiv: ログイン促進バナーを隠し、作品カードに + ボタンを追加
  const PIXIV_HIDE_CSS = `
    .signup-modal,[class*="LoginRequired"],[class*="signup"],[class*="SignupModal"],
    ._premium-ad,[class*="premium-interstitial"]{display:none!important}
  `;
  const PIXIV_INJECT_JS = `(function(){
    if(!document.getElementById('lbp-pxv-style')){
      var s=document.createElement('style');s.id='lbp-pxv-style';
      s.textContent=[
        '.lbp-pxv-wrap{position:relative!important}',
        '.lbp-pxv-btn{position:absolute!important;top:6px!important;left:6px!important;',
        'width:36px!important;height:36px!important;background:rgba(20,20,20,.82)!important;',
        'color:#fff!important;border:2px solid rgba(255,255,255,.45)!important;',
        'border-radius:50%!important;font-size:20px!important;line-height:34px!important;',
        'text-align:center!important;cursor:pointer!important;z-index:2147483646!important;',
        'pointer-events:auto!important;opacity:.9!important;transition:opacity .15s!important;',
        'box-shadow:0 2px 8px rgba(0,0,0,.5)!important;}',
        '.lbp-pxv-wrap:hover .lbp-pxv-btn{opacity:1!important;}',
        '.lbp-pxv-btn.done{background:rgba(16,185,129,.95)!important;border-color:transparent!important;opacity:1!important;}'
      ].join('');
      document.head.appendChild(s);
    }
    /* /artworks/XXXXX リンクから含む最小カード要素を探す */
    function findCard(a){
      var el=a.parentElement;
      for(var i=0;i<10;i++){
        if(!el||el===document.body)break;
        var tag=el.tagName;
        if(tag==='LI'||tag==='FIGURE')return el;
        /* div でも img を含む最初の祖先を候補に */
        if(tag==='DIV'&&el.querySelector('img'))return el;
        el=el.parentElement;
      }
      return null;
    }
    function processLink(a){
      if(a.dataset.lbpDone)return;
      var href=a.getAttribute('href')||'';
      var m=href.match(/\\/artworks\\/(\\d+)/);
      if(!m)return;
      var artworkId=m[1];
      var card=findCard(a);
      if(!card||card.querySelector('.lbp-pxv-btn'))return;
      if(getComputedStyle(card).position==='static')card.style.position='relative';
      card.classList.add('lbp-pxv-wrap');
      var b=document.createElement('button');
      b.className='lbp-pxv-btn';b.textContent='+';b.title='データセットに追加';
      b.addEventListener('pointerdown',function(e){e.preventDefault();e.stopPropagation();if(e.stopImmediatePropagation)e.stopImmediatePropagation();},true);
      b.addEventListener('click',function(e){
        e.preventDefault();e.stopPropagation();if(e.stopImmediatePropagation)e.stopImmediatePropagation();
        /* artworks URL を送る — バックエンドが Pixiv API で原寸取得 */
        console.log('LBP_ADD:https://www.pixiv.net/artworks/'+artworkId);
        b.textContent='\\u2713';b.classList.add('done');
      },true);
      card.appendChild(b);
      a.dataset.lbpDone='1';
    }
    function scan(){
      document.querySelectorAll('a[href*="/artworks/"]').forEach(processLink);
    }
    scan();
    new MutationObserver(function(){scan();}).observe(document.body,{childList:true,subtree:true,attributes:false});
  })();`;

  useEffect(() => {
    const wv = pixivRef.current;
    if (!wv) return;
    const inject = () => {
      try { wv.insertCSS(PIXIV_HIDE_CSS); } catch (_) {}
      try { wv.executeJavaScript(PIXIV_INJECT_JS); } catch (_) {}
    };
    const onLoad = () => { inject(); setPixivReady(true); };
    const onNavigateInPage = () => { inject(); };
    const onCrash = () => {
      setPixivReady(false);
      try { wv.src = "https://www.pixiv.net/"; } catch (_) {}
    };
    const onFailLoad = (e: any) => {
      if (e.errorCode && e.errorCode !== -3) { setPixivReady(false); try { wv.reload(); } catch (_) {} }
    };
    const onConsole = (e: any) => {
      if (typeof e.message === "string" && e.message.startsWith("LBP_ADD:")) {
        const url = e.message.slice(8).trim();
        const proj = selectedProjectRef.current;
        if (proj && url) {
          void dropUrlAsCandidate(proj.id, url).then(() => setPixivAddedCount(c => c + 1));
        }
      }
    };
    wv.addEventListener("did-finish-load", onLoad);
    wv.addEventListener("did-navigate-in-page", onNavigateInPage);
    wv.addEventListener("crashed", onCrash);
    wv.addEventListener("did-fail-load", onFailLoad);
    wv.addEventListener("console-message", onConsole);
    return () => {
      wv.removeEventListener("did-finish-load", onLoad);
      wv.removeEventListener("did-navigate-in-page", onNavigateInPage);
      wv.removeEventListener("crashed", onCrash);
      wv.removeEventListener("did-fail-load", onFailLoad);
      wv.removeEventListener("console-message", onConsole);
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const wv = pinterestRef.current;
    if (!wv) return;
    const HIDE_CSS = PINTEREST_HIDE_CSS;
    const INJECT_JS = PINTEREST_INJECT_JS;

    const inject = () => {
      try { wv.insertCSS(HIDE_CSS); } catch (_) {}
      try { wv.executeJavaScript(INJECT_JS); } catch (_) {}
    };

    // フルロード（初回・リロード）
    const onLoad = () => {
      inject();
      setPinterestReady(true);
    };

    // SPA内ナビゲーション（Pinterest はほぼこちら）
    const onNavigateInPage = () => {
      // SPAナビ後は既存のMutationObserverが動いているが
      // detailボタンやCSSは再注入しないと消える
      inject();
    };

    // webview クラッシュ時は src を再設定して復旧
    const onCrash = () => {
      setPinterestReady(false);
      try { wv.src = "https://www.pinterest.com/"; } catch (_) {}
    };

    // ナビゲーション失敗（ERR_ABORTED以外）
    const onFailLoad = (e: any) => {
      if (e.errorCode && e.errorCode !== -3) {
        setPinterestReady(false);
        try { wv.reload(); } catch (_) {}
      }
    };

    const onConsole = (e: any) => {
      if (typeof e.message === "string" && e.message.startsWith("LBP_ADD:")) {
        const url = e.message.slice(8).trim();
        const proj = selectedProjectRef.current;
        if (proj && url) {
          void dropUrlAsCandidate(proj.id, url).then(() =>
            setPinterestAddedCount(c => c + 1));
        }
      }
    };

    wv.addEventListener("did-finish-load", onLoad);
    wv.addEventListener("did-navigate-in-page", onNavigateInPage);
    wv.addEventListener("crashed", onCrash);
    wv.addEventListener("did-fail-load", onFailLoad);
    wv.addEventListener("console-message", onConsole);
    return () => {
      wv.removeEventListener("did-finish-load", onLoad);
      wv.removeEventListener("did-navigate-in-page", onNavigateInPage);
      wv.removeEventListener("crashed", onCrash);
      wv.removeEventListener("did-fail-load", onFailLoad);
      wv.removeEventListener("console-message", onConsole);
    };
  // マウント時のみ登録（selectedProject は ref 経由で最新値を参照）
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── WD14 しきい値・オプション ──
  const [wd14GeneralThresh, setWd14GeneralThresh] = useState(0.35);
  const [wd14CharacterThresh, setWd14CharacterThresh] = useState(0.85);
  const [wd14RemoveCharacterTags, setWd14RemoveCharacterTags] = useState(false);

  // ── Prefix Tags / Block Words ──
  const [prefixTags, setPrefixTags] = useState<string[]>([]);
  const [blockWords, setBlockWords] = useState<string[]>([]);
  const [prefixInput, setPrefixInput] = useState("");
  const [blockInput, setBlockInput] = useState("");
  const [tagSettingsSaving, setTagSettingsSaving] = useState(false);
  const [blockHits, setBlockHits] = useState<Record<string, number>>({});

  // ── AI提案 手動Booruクエリ ──
  const [manualBooruQuery, setManualBooruQuery] = useState("");

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

  const dupPathById = useMemo(
    () => new Map(captionItems.map((c) => [c.id, c.file_path] as const)),
    [captionItems]
  );

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
        if (st.status === "done" || st.status === "done_with_errors" || st.status === "idle" || st.status === "stopped") {
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
      // タグ設定を読み込む
      void apiGet<{ project_id: number; prefix_tags: string[]; block_words: string[] }>(
        `/tags/settings/${selectedProject.id}`
      ).then((s) => {
        setPrefixTags(s.prefix_tags);
        setBlockWords(s.block_words);
      }).catch(() => {});
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
      void loadCaptions();
      void (async () => {
        try {
          const st = await apiGet<{ ready?: boolean; upscale_models?: string[]; upscale_model?: string }>(
            `/dataset/comfyui/status`
          );
          setComfyReady(!!st.ready);
          setComfyUpscaleModels(st.upscale_models ?? []);
          setUpscaleModel((m) => m || st.upscale_model || "");
        } catch {
          setComfyReady(false);
        }
      })();
      // Pipeline設定（Resize/Upscale/Cleanup/Caption）をProjectから復元
      void (async () => {
        try {
          const cfg = await apiGet<{ project_id: number; config: Record<string, unknown> }>(
            `/dataset/pipeline-config/${selectedProject.id}`
          );
          const c = cfg.config ?? {};
          if (typeof c.target_size === "number") setResolutionTarget(String(c.target_size));
          if (typeof c.resize_mode === "string") setKeepAspectRatio(c.resize_mode !== "square_crop");
          if (typeof c.use_esrgan === "boolean") setUseUpscale(c.use_esrgan);
          if (typeof c.esrgan_model === "string" && c.esrgan_model) setUpscaleModel(c.esrgan_model);
          if (typeof c.use_qwen === "boolean") setPreprocessingMode((m) => (c.use_qwen ? "text_removal" : m === "text_removal" ? "resolution" : m));
          if (typeof c.use_caption === "boolean") setUseCaption(c.use_caption);
          if (typeof c.caption_general_thresh === "number") setCaptionGeneralThresh(c.caption_general_thresh);
          if (typeof c.caption_character_thresh === "number") setCaptionCharacterThresh(c.caption_character_thresh);
          if (typeof c.caption_remove_character_tags === "boolean") setCaptionRemoveCharacterTags(c.caption_remove_character_tags);
        } catch {
          // 未保存 or 取得失敗時は既定値のまま
        }
      })();
    }
  }, [activeTab, selectedProject, loadDashboard, loadCaptions]);

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

  // ── 提案ポーリング ──
  useEffect(() => {
    if (!suggestionPolling || !selectedProject) return;
    const id = setInterval(async () => {
      try {
        const status = await apiGet<{ status: string; message: string; step?: number; total_steps?: number }>(
          `/dataset/suggest-status/${selectedProject.id}`
        );
        setSuggestionStatusMsg(status.message ?? "");
        setSuggestionStep(status.step ?? 0);

        const terminal = status.status === "done" || status.status === "failed";
        if (terminal) {
          if (status.status === "done") {
            const suggestions = await apiGet<SuggestionsResult>(`/dataset/suggestions/${selectedProject.id}`);
            setSuggestionsData(suggestions);
          }
          setSuggestionPolling(false);
          setSuggestionsRunning(false);
        }
      } catch {
        setSuggestionPolling(false);
        setSuggestionsRunning(false);
      }
    }, 2000);
    return () => clearInterval(id);
  }, [suggestionPolling, selectedProject]);

  // ── フィードバック再ランク ポーリング ──
  useEffect(() => {
    if (!feedbackPolling || !selectedProject) return;
    const id = setInterval(async () => {
      try {
        const status = await apiGet<{ status: string; message: string; result?: SuggestionsResult }>(
          `/dataset/suggest-feedback-status/${selectedProject.id}`
        );
        setFeedbackStatusMsg(status.message ?? "");
        if (status.status === "done" || status.status === "failed") {
          if (status.status === "done") {
            const updated = await apiGet<SuggestionsResult>(`/dataset/suggestions/${selectedProject.id}`);
            setSuggestionsData(updated);
          }
          setFeedbackPolling(false);
          setFeedbackRunning(false);
        }
      } catch {
        setFeedbackPolling(false);
        setFeedbackRunning(false);
      }
    }, 2000);
    return () => clearInterval(id);
  }, [feedbackPolling, selectedProject]);

  // ── Dataset Mixer ロード ──
  useEffect(() => {
    if (activeTab === "mixer" && selectedProject) {
      apiGet<DatasetMixer>(`/dataset/mixer/${selectedProject.id}`)
        .then((data) => setMixerWeights(data.feature_weights))
        .catch(() => {});
    }
  }, [activeTab, selectedProject]);

  // ── 経過時間カウンター ──
  // プロジェクト切り替え時に localStorage からフォルダパスを再ロード
  useEffect(() => {
    try {
      const stored = localStorage.getItem(`repeatPreparedPath_${selectedProject?.id ?? ""}`) ?? "";
      setRepeatPreparedPathRaw(stored);
    } catch {}
  }, [selectedProject?.id]);

  useEffect(() => {
    if (!suggestionsRunning) { setSuggestionElapsed(0); return; }
    const t = setInterval(() => {
      setSuggestionElapsed(Math.floor((Date.now() - (suggestionStartAt ?? Date.now())) / 1000));
    }, 1000);
    return () => clearInterval(t);
  }, [suggestionsRunning, suggestionStartAt]);

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

  async function startSuggestions() {
    if (!selectedProject) return;
    setSuggestionsRunning(true);
    setSuggestionsData(null);
    setLikedUrls(new Set());
    setRejectedUrls(new Set());
    setSuggestionStatusMsg("処理を開始中...");
    setSuggestionStep(0);
    setSuggestionStartAt(Date.now());
    try {
      await apiPost(`/dataset/suggest-images/${selectedProject.id}`, {
        evaluation_mode: suggestionMode,
        manual_query: manualBooruQuery.trim(),
      });
      setSuggestionPolling(true);
    } catch (e) {
      showError(`提案検索失敗: ${String(e)}`);
      setSuggestionsRunning(false);
      setSuggestionStartAt(null);
    }
  }

  async function stopSuggestion() {
    if (!selectedProject) return;
    setSuggestionPolling(false);
    setSuggestionsRunning(false);
    setSuggestionStartAt(null);
    setSuggestionStatusMsg("停止しました");
    try {
      await apiPost(`/dataset/suggest-cancel/${selectedProject.id}`, {});
    } catch (_) { /* ignore */ }
  }

  function toggleLike(url: string) {
    setLikedUrls((prev) => {
      const next = new Set(prev);
      if (next.has(url)) {
        next.delete(url);
      } else {
        next.add(url);
        // liked にしたら rejected から除外
        setRejectedUrls((r) => { const rn = new Set(r); rn.delete(url); return rn; });
      }
      return next;
    });
  }

  function toggleReject(url: string) {
    setRejectedUrls((prev) => {
      const next = new Set(prev);
      if (next.has(url)) {
        next.delete(url);
      } else {
        next.add(url);
        // rejected にしたら liked から除外
        setLikedUrls((l) => { const ln = new Set(l); ln.delete(url); return ln; });
      }
      return next;
    });
  }

  async function sendFeedback() {
    if (!selectedProject || (likedUrls.size === 0 && rejectedUrls.size === 0)) return;
    setFeedbackRunning(true);
    setFeedbackStatusMsg("フィードバックを送信中...");
    try {
      const payload: SuggestionFeedbackPayload = {
        accepted_urls: Array.from(likedUrls),
        rejected_urls: Array.from(rejectedUrls),
        evaluation_mode: suggestionMode,
      };
      await apiPost(`/dataset/suggest-feedback/${selectedProject.id}`, payload);
      setFeedbackPolling(true);
    } catch (e) {
      showError(`フィードバック送信失敗: ${String(e)}`);
      setFeedbackRunning(false);
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
    const abort = new AbortController();
    setScanAbortRef(abort);
    setScanning(true);
    try {
      const r = await apiPost<ScanResult>("/collector/scan", {
        project_id: selectedProject.id,
        url: scanUrl,
        keyword: buildKeyword(),
        limit: 60,
      }, "POST", abort.signal);
      setScanItems(r.items);
      setSelectedIds(r.items.slice(0, 12).map((x) => x.id));
      setExpandedTagId(null);
      setScanMode(r.mode);
      setScanMessage(r.message);
      if (r.mode === "url_unavailable") showError(`画像取得失敗: ${r.message}`);
      else showNotice(`候補取得: ${r.detected}件 (${r.mode})`);
    } catch (e: unknown) {
      if (e instanceof Error && e.name === "AbortError") {
        showNotice("取得を停止しました");
      } else {
        showError(`取得失敗: ${String(e)}`);
      }
    } finally {
      setScanning(false);
      setScanAbortRef(null);
    }
  }

  function stopScan() {
    scanAbortRef?.abort();
    setScanAbortRef(null);
    setScanning(false);
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

  async function pasteImageFromClipboard() {
    if (!selectedProject) return;
    try {
      const items = await navigator.clipboard.read();
      const imageFiles: File[] = [];
      for (const item of items) {
        const imageType = item.types.find(t => t.startsWith("image/"));
        if (imageType) {
          const blob = await item.getType(imageType);
          const ext = imageType === "image/png" ? "png" : imageType === "image/jpeg" ? "jpg" : "png";
          imageFiles.push(new File([blob], `clipboard_${Date.now()}.${ext}`, { type: imageType }));
        }
      }
      if (imageFiles.length === 0) {
        showError("クリップボードに画像がありません");
        return;
      }
      const form = new FormData();
      form.append("project_id", String(selectedProject.id));
      imageFiles.forEach((f) => form.append("files", f));
      const json = await apiFormPost<{ items: ScanItem[]; added_count: number }>("/collector/drop-files", form);
      setScanItems(json.items || []);
      showNotice(`スクショ ${json.added_count ?? imageFiles.length}件追加`);
    } catch (e) {
      showError(`クリップボード画像取得失敗: ${String(e)}`);
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
      await apiPost("/tags/generate", {
        project_id: selectedProject.id,
        overwrite,
        general_thresh: wd14GeneralThresh,
        character_thresh: wd14CharacterThresh,
        remove_character_tags: wd14RemoveCharacterTags,
      });
      setTaggerStatus({ project_id: selectedProject.id, status: "queued", total: 0, done: 0, message: "モデル読み込み中..." });
      setCaptionPolling(true);
    } catch (e) {
      showError(`タグ生成開始失敗: ${String(e)}`);
    }
  }

  async function stopWd14() {
    if (!selectedProject) return;
    try {
      const r = await apiPost<{ stopping: boolean; message: string }>(`/tags/stop/${selectedProject.id}`, {});
      showNotice(r.message);
      setTaggerStatus((prev) => prev ? { ...prev, message: "停止中..." } : prev);
    } catch (e) {
      showError(`停止要求失敗: ${String(e)}`);
    }
  }

  async function deleteDatasetItem(itemId: number) {
    try {
      await fetch(`${API_BASE}/tags/item/${itemId}`, { method: "DELETE" });
      setCaptionItems((prev) => prev.filter((c) => c.id !== itemId));
      showNotice("削除しました（ファイルは残っています）");
    } catch (e) {
      showError(`削除失敗: ${String(e)}`);
    }
  }

  async function deleteItemsHard(ids: number[], deleteFiles: boolean) {
    if (!selectedProject || ids.length === 0) return;
    try {
      const r = await apiPost<{ removed_count: number; trashed_files: number }>(
        `/dataset/delete-items/${selectedProject.id}?item_ids=${ids.join(",")}&delete_files=${deleteFiles}`,
        {}
      );
      showNotice(`${r.removed_count}件削除${deleteFiles ? `（${r.trashed_files}件をゴミ箱へ）` : ""}`);
      setCaptionItems((prev) => prev.filter((c) => !ids.includes(c.id)));
      await loadDashboard();
      void startAnalysis();
    } catch (e) {
      showError(`削除失敗: ${String(e)}`);
    }
  }

  // 重複グループから各1枚を残し、残りを一括削除
  async function bulkRemoveDuplicates() {
    const dupIds: number[] = [];
    for (const g of dashboardData?.similarity_groups ?? []) {
      if (g.type === "duplicate" && g.item_ids.length > 1) {
        dupIds.push(...g.item_ids.slice(1));
      }
    }
    if (dupIds.length === 0) {
      showNotice("削除対象の重複画像はありません");
      return;
    }
    if (!window.confirm(`重複画像 ${dupIds.length} 枚を、各グループ1枚だけ残して削除します。よろしいですか？\n（削除した画像ファイルはゴミ箱へ移動します）`)) return;
    await deleteItemsHard(dupIds, true);
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

  // AI収集（データセット提案）パネルの表示フラグ。精度不足のため現在は非表示。
  const SHOW_AI_SUGGEST: boolean = false;
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
    <div
      className="space-y-3 max-w-5xl"
      onWheelCapture={(e) => {
        if (!e.ctrlKey) return;
        if (webviewPanelRef.current?.contains(e.target as Node)) return;
        e.preventDefault();
        e.stopPropagation();
        const delta = e.deltaY > 0 ? -0.1 : 0.1;
        setUiZoom(z => Math.max(0.5, Math.min(2.0, +(z + delta).toFixed(1))));
      }}
    >
      {/* Header */}
      <div className="flex items-start justify-between flex-wrap gap-2">
        <div>
          <h2 className="text-xl font-bold text-gray-100 mb-1">データセット作成</h2>
          <div className="flex items-center gap-2 text-sm text-gray-400 flex-wrap">
            <span>対象:</span>
            <span className="text-indigo-400 font-medium">{selectedProject.name}</span>
            <span className="text-gray-600">/</span>
            <span>{selectedProject.project_type}</span>
            {uiZoom !== 1.0 && (
              <button
                onClick={() => setUiZoom(1.0)}
                className="flex items-center gap-1 text-xs bg-gray-700 hover:bg-gray-600 border border-gray-600 text-gray-300 rounded px-1.5 py-0.5 transition-colors"
                title="UIズームをリセット (Ctrl+スクロールで調整)"
              >
                UI {Math.round(uiZoom * 100)}% ✕
              </button>
            )}
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

      {/* Stepper tab header */}
      {(() => {
        type StepId = ActiveTab;
        const steps: { id: StepId; label: string; badge?: React.ReactNode }[] = [
          { id: "collect", label: "画像取得", badge: scanItems.length > 0 ? <span className="text-[10px] bg-blue-900 text-blue-300 px-1.5 py-0.5 rounded-full">{scanItems.length}件</span> : null },
          { id: "caption", label: "タグ編集", badge: captionStats.total > 0 ? <span className="text-[10px] bg-purple-900 text-purple-300 px-1.5 py-0.5 rounded-full">{captionStats.withCaption}/{captionStats.total}</span> : null },
          { id: "dashboard", label: "品質確認・前処理", badge: dashboardData?.quality_score != null ? <span className={["text-[10px] px-1.5 py-0.5 rounded-full font-bold", dashboardData.quality_score >= 80 ? "bg-emerald-900 text-emerald-300" : dashboardData.quality_score >= 60 ? "bg-amber-900 text-amber-300" : "bg-red-900 text-red-300"].join(" ")}>{dashboardData.quality_score}</span> : null },
          // ミキサーは高度なロジックが必要なため一旦非表示（復活させる場合は下行を戻す）
          // { id: "mixer", label: "ミキサー" },
        ];
        const activeIdx = steps.findIndex((s) => s.id === activeTab);
        return (
          <div className="flex items-center gap-1 bg-gray-800/60 border border-gray-700 rounded-xl p-1 overflow-x-auto" style={{scrollbarWidth:"none"}}>
            {steps.map((step, i) => {
              const isActive = step.id === activeTab;
              const isDone = i < activeIdx;
              return (
                <button
                  key={step.id}
                  onClick={() => setActiveTab(step.id)}
                  className={[
                    "flex-shrink-0 flex items-center gap-1.5 py-1.5 px-3 rounded-lg text-xs font-medium transition-all whitespace-nowrap",
                    isActive ? "bg-gray-700 text-gray-100 shadow-sm" : isDone ? "text-emerald-400 hover:bg-gray-700/50" : "text-gray-500 hover:text-gray-300 hover:bg-gray-700/30",
                  ].join(" ")}
                >
                  <span className={["flex-shrink-0 w-4 h-4 rounded-full flex items-center justify-center text-[9px] font-bold", isActive ? "bg-orange-600 text-white" : isDone ? "bg-emerald-700 text-white" : "bg-gray-700 text-gray-500"].join(" ")}>
                    {isDone ? "✓" : i + 1}
                  </span>
                  <span className="truncate">{step.label}</span>
                  {step.badge}
                </button>
              );
            })}
          </div>
        );
      })()}

      {/* ─────────────────────────── 収集タブ ─────────────────────────── */}
      {activeTab === "collect" && (
        <>
          {/* BUG-03: SCAN_CACHE インメモリ警告 */}
          <div className="flex items-center gap-2 text-xs text-amber-400/80 bg-amber-950/20 border border-amber-900/30 rounded-lg px-3 py-2 mb-3">
            <AlertTriangle size={12} className="flex-shrink-0" />
            <span>スキャン結果はサーバー再起動でリセットされます。取り込み（インポート）済みの画像は保持されます。</span>
          </div>
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
            <div className="p-3">
              <div className="grid gap-2">
                <div className="flex gap-2 flex-wrap">
                  <div className="flex-1 min-w-0 flex gap-1">
                    <div className="flex-1 relative min-w-0">
                      <Link size={13} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" />
                      <input
                        value={scanUrl}
                        onChange={(e) => setScanUrl(e.target.value)}
                        placeholder="https://www.pixiv.net/artworks/..."
                        className="w-full bg-gray-700 border border-gray-600 text-gray-100 placeholder-gray-500 rounded-lg pl-8 pr-3 py-1.5 text-xs focus:outline-none focus:border-indigo-500"
                      />
                    </div>
                    <button
                      onClick={async () => {
                        // eslint-disable-next-line @typescript-eslint/no-explicit-any
                        const api = (window as any).electronAPI;
                        // Electron: まず画像クリップボードを確認
                        if (api?.clipboardReadImage) {
                          const b64 = await (api.clipboardReadImage() as Promise<string | null>).catch(() => null);
                          if (b64) {
                            // base64 PNG → File → drop-files API
                            if (!selectedProject) return;
                            const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
                            const file = new File([bytes], `clipboard_${Date.now()}.png`, { type: "image/png" });
                            const form = new FormData();
                            form.append("project_id", String(selectedProject.id));
                            form.append("files", file);
                            try {
                              const json = await apiFormPost<{ items: ScanItem[]; added_count: number }>("/collector/drop-files", form);
                              setScanItems(json.items || []);
                              showNotice(`スクショ ${json.added_count ?? 1}件追加`);
                            } catch (e) { showError(`追加失敗: ${String(e)}`); }
                            return;
                          }
                          // 画像なし → テキストとして読む
                          const t = await (api.clipboardRead() as Promise<string>).catch(() => "");
                          const v = t.trim(); if (v) setScanUrl(v);
                          return;
                        }
                        // ブラウザ: navigator.clipboard.read() で画像確認
                        try {
                          const items = await navigator.clipboard.read();
                          const hasImage = items.some(item => item.types.some(t => t.startsWith("image/")));
                          if (hasImage) { await pasteImageFromClipboard(); return; }
                        } catch { /* 権限なし */ }
                        const t = await navigator.clipboard.readText().catch(() => "");
                        const v = t.trim(); if (v) setScanUrl(v);
                      }}
                      title="クリップボードから貼り付け（URL→URL欄 / スクショ→画像追加）"
                      className="flex items-center gap-1 px-2.5 py-1.5 rounded-lg bg-indigo-700 hover:bg-indigo-600 border border-indigo-500 text-indigo-100 hover:text-white transition-colors flex-shrink-0 text-xs font-medium"
                    >
                      <ClipboardPaste size={13} />
                      <span>貼付</span>
                    </button>
                  </div>
                  <button
                    onClick={() => scanning ? stopScan() : void runScan()}
                    disabled={!selectedProject}
                    className={[
                      "flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium transition-colors flex-shrink-0",
                      scanning
                        ? "bg-red-800 hover:bg-red-700 text-white"
                        : "bg-blue-700 hover:bg-blue-600 disabled:bg-gray-700 disabled:text-gray-500 text-white",
                    ].join(" ")}
                  >
                    {scanning ? (
                      <><X size={13} />停止</>
                    ) : (
                      <><Search size={13} />候補取得</>
                    )}
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

          {/* 候補グリッド + 検索ブラウザ を横並び（広い画面では2カラム） */}
          <div className={scanItems.length > 0 ? "grid gap-3 2xl:grid-cols-2 2xl:items-stretch" : ""}>
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
                {(() => {
                  const failedIds = scanItems.filter((it) => it.unavailable).map((it) => it.id);
                  return failedIds.length > 0 ? (
                    <button onClick={() => void removeItems(failedIds)} className="flex items-center gap-1.5 text-xs px-2 py-1 rounded bg-red-900/50 hover:bg-red-900 text-red-300 border border-red-800/50 transition-colors" title="取得に失敗した画像をまとめて除外">
                      <AlertTriangle size={13} />失敗を除外 ({failedIds.length})
                    </button>
                  ) : null;
                })()}
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
                        item.unavailable
                          ? "border-red-600/70 ring-2 ring-red-600/20"
                          : isSelected ? "border-indigo-500 ring-2 ring-indigo-500/30" : "border-transparent hover:border-gray-500",
                      ].join(" ")}
                      onClick={() => toggleId(item.id)}
                      onDoubleClick={() => setExpandedTagId((prev) => prev === item.id ? null : item.id)}
                      onContextMenu={(e) => { e.preventDefault(); setSelectedPreviewItem(item); setContextMenuPos({ x: e.clientX, y: e.clientY }); }}
                    >
                      <div className="aspect-square bg-gray-700">
                        {item.thumbnail_url ? (
                          <img src={item.thumbnail_url} alt={item.title} className={["w-full h-full object-cover", item.unavailable ? "opacity-40 grayscale" : ""].join(" ")} />
                        ) : (
                          <div className="w-full h-full flex items-center justify-center">
                            <ImageOff size={20} className="text-gray-600" />
                          </div>
                        )}
                      </div>
                      {item.unavailable && (
                        <div className="absolute top-1.5 left-1/2 -translate-x-1/2 flex items-center gap-1 bg-red-900/90 text-red-100 text-[10px] font-semibold px-1.5 py-0.5 rounded shadow pointer-events-none">
                          <AlertTriangle size={10} />取得失敗
                        </div>
                      )}
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

          {/* 画像検索パネル (Pinterest / Pixiv — タブ統合)
              凍結中: 埋め込みwebview UIが使いづらいとのフィードバックにより非表示化。
              コードは削除せず保持し、falseガードで無効化のみ行う。 */}
          {false && (
          <div
            ref={webviewPanelRef}
            className="bg-gray-800 border border-gray-700 rounded-xl overflow-hidden flex flex-col panel-elevated"
            style={{ minHeight: "460px", height: scanItems.length > 0 ? "100%" : "calc(100vh - 220px)" }}
            onWheelCapture={(e) => {
              // Ctrl+スクロール → webview ズーム
              if (!e.ctrlKey) return;
              e.preventDefault();
              e.stopPropagation();
              const delta = e.deltaY < 0 ? 1 : -1;
              const nextZoom = Math.max(-5, Math.min(5, webviewZoom + delta));
              setWebviewZoom(nextZoom);
              const target = browseSource === "pinterest" ? pinterestRef.current : pixivRef.current;
              if (target && typeof target.setZoomLevel === "function") {
                target.setZoomLevel(nextZoom);
              }
            }}
          >
            {/* タブバー */}
            <div className="flex items-center gap-1 px-2 pt-2 border-b border-gray-700 flex-shrink-0 bg-gray-900/40">
              {([
                { key: "pinterest" as const, label: "Pinterest", color: "text-pink-400", dot: "bg-pink-400", added: pinterestAddedCount },
                { key: "pixiv" as const, label: "Pixiv", color: "text-blue-400", dot: "bg-blue-400", added: pixivAddedCount },
              ]).map((t) => {
                const active = browseSource === t.key;
                return (
                  <button
                    key={t.key}
                    onClick={() => setBrowseSource(t.key)}
                    className={[
                      "relative flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-t-lg transition-colors",
                      active ? "bg-gray-800 text-gray-100 border border-b-0 border-gray-700" : "text-gray-500 hover:text-gray-300 hover:bg-gray-800/50",
                    ].join(" ")}
                  >
                    <span className={["w-1.5 h-1.5 rounded-full", t.dot].join(" ")} />
                    {t.label}
                    {t.added > 0 && (
                      <span className="text-[10px] bg-emerald-800 text-emerald-300 px-1.5 py-0.5 rounded-full">+{t.added}</span>
                    )}
                  </button>
                );
              })}
              <div className="ml-auto flex items-center gap-2 pr-1">
                {/* ズームリセット */}
                {webviewZoom !== 0 && (
                  <button
                    onClick={() => {
                      setWebviewZoom(0);
                      const target = browseSource === "pinterest" ? pinterestRef.current : pixivRef.current;
                      if (target && typeof target.setZoomLevel === "function") target.setZoomLevel(0);
                    }}
                    className="text-[10px] bg-gray-700 hover:bg-gray-600 text-gray-300 px-2 py-0.5 rounded transition-colors"
                    title="ズームをリセット"
                  >
                    {Math.round(Math.pow(1.2, webviewZoom) * 100)}% ✕
                  </button>
                )}
                {selectedProject && (
                  <button
                    onClick={() => {
                      const proj = selectedProject;
                      if (!proj) return;
                      if (browseSource === "pinterest" && pinterestRef.current) {
                        pinterestRef.current.src = `https://www.pinterest.jp/search/pins/?q=${encodeURIComponent(proj.name)}`;
                      } else if (browseSource === "pixiv" && pixivRef.current) {
                        pixivRef.current.src = `https://www.pixiv.net/tags/${encodeURIComponent(proj.name)}/artworks?mode=all`;
                      }
                    }}
                    className="text-xs bg-indigo-900/50 hover:bg-indigo-800/70 text-indigo-300 px-2.5 py-1 rounded-lg transition-colors"
                  >「{selectedProject?.name}」で検索</button>
                )}
              </div>
            </div>

            {/* webview コンテナ（両方マウントしたまま display で切替 → ログイン/スクロール維持） */}
            <div className="flex-1 relative overflow-hidden">
              <div className="absolute inset-0" style={{ display: browseSource === "pinterest" ? "block" : "none" }}>
                {/* @ts-ignore: webview is an Electron-specific element */}
                <webview
                  ref={pinterestRef}
                  src={`https://www.pinterest.jp/search/pins/?q=${encodeURIComponent(selectedProject?.name ?? "anime character")}`}
                  partition="persist:pinterest"
                  {...{ allowpopups: "true" } as any}
                  style={{ width: "100%", height: "100%", border: "none" }}
                />
                {!pinterestReady && (
                  <div className="absolute inset-0 flex items-center justify-center bg-gray-900/80">
                    <Loader2 size={24} className="animate-spin text-pink-400" />
                  </div>
                )}
              </div>
              <div className="absolute inset-0" style={{ display: browseSource === "pixiv" ? "block" : "none" }}>
                {/* @ts-ignore: webview is an Electron-specific element */}
                <webview
                  ref={pixivRef}
                  src={`https://www.pixiv.net/tags/${encodeURIComponent(selectedProject?.name ?? "anime")}/artworks?mode=all`}
                  partition="persist:pixiv"
                  {...{ allowpopups: "true" } as any}
                  style={{ width: "100%", height: "100%", border: "none" }}
                />
                {!pixivReady && (
                  <div className="absolute inset-0 flex items-center justify-center bg-gray-900/80">
                    <Loader2 size={24} className="animate-spin text-blue-400" />
                  </div>
                )}
              </div>
            </div>

            <div className="px-3 py-1.5 border-t border-gray-700 flex-shrink-0">
              <p className="text-[10px] text-gray-500">
                {browseSource === "pinterest"
                  ? "ピンの上の + でデータセットに追加 · Ctrl+スクロールでズーム · 一度ログインすれば次回から自動ログイン"
                  : "作品の上の + でデータセットに追加 · Ctrl+スクロールでズーム · R18は要ログイン + 年齢設定ON"}
              </p>
            </div>
          </div>
          )}
          </div>{/* /横並びラッパー */}

          {/* Workflow Steps */}
          <div className="bg-gray-800 border border-gray-700 rounded-xl p-3">
            <div className="flex items-center gap-2 mb-3">
              <span className="text-xs font-semibold text-gray-400 uppercase tracking-wide">ワークフロー</span>
              <button
                onClick={() => setShowFolderSettings((v) => !v)}
                className="flex items-center gap-1 text-[10px] text-gray-500 hover:text-gray-300 transition-colors"
                title="フォルダ詳細設定の表示切替"
              >
                {showFolderSettings ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                フォルダ設定
              </button>
              {repeatPreparedPath && (
                <span className="ml-auto text-[10px] text-emerald-400 flex items-center gap-1 truncate max-w-[50%]">
                  <FolderPlus size={10} className="flex-shrink-0" />{repeatPreparedPath}
                </span>
              )}
            </div>
            {/* 横1行ツールバー */}
            <div className="flex gap-1.5 mb-3 flex-wrap">
              <WorkflowBtn step={2} label="フォルダ作成" icon={<FolderPlus size={13} />} color="bg-indigo-700 hover:bg-indigo-600" onClick={() => void prepareRepeatFolder()} done={!!repeatPreparedPath} />
              <WorkflowBtn step={3} label="取り込み" icon={<Download size={13} />} color="bg-emerald-700 hover:bg-emerald-600" onClick={() => void runImport()} />
              <WorkflowBtn step={4} label="タグ編集へ" icon={<Tag size={13} />} color="bg-violet-700 hover:bg-violet-600" onClick={() => setActiveTab("caption")} />
              <button
                onClick={() => selectedProject && void dropUrlAsCandidate(selectedProject.id, scanUrl)}
                className="flex items-center gap-1.5 bg-gray-700 hover:bg-gray-600 text-gray-300 rounded-lg px-3 py-1.5 text-xs font-medium transition-colors"
              >
                <Plus size={13} />URL追加
              </button>
            </div>
            {/* フォルダ設定（折りたたみ可能な詳細） */}
            {showFolderSettings && (
            <div className="grid grid-cols-3 gap-2">
              <label className="block">
                <span className="text-[10px] text-gray-500 mb-0.5 block">繰り返し数</span>
                <input
                  type="text" inputMode="numeric" value={repeatCountText}
                  onChange={(e) => setRepeatCountText(e.target.value)}
                  className="w-full bg-gray-700 border border-gray-600 text-gray-100 rounded px-2 py-1 text-xs focus:outline-none focus:border-indigo-500"
                />
              </label>
              <label className="block">
                <span className="text-[10px] text-gray-500 mb-0.5 block">フォルダ名</span>
                <input
                  value={repeatFolderTitle || selectedProject.name}
                  onChange={(e) => setRepeatFolderTitle(e.target.value)}
                  className="w-full bg-gray-700 border border-gray-600 text-gray-100 rounded px-2 py-1 text-xs focus:outline-none focus:border-indigo-500"
                />
              </label>
              <label className="block">
                <span className="text-[10px] text-gray-500 mb-0.5 block">命名テンプレート</span>
                <input
                  value={namingTemplate}
                  onChange={(e) => setNamingTemplate(e.target.value)}
                  placeholder="{title}_{index}"
                  className="w-full bg-gray-700 border border-gray-600 text-gray-100 rounded px-2 py-1 text-xs focus:outline-none focus:border-indigo-500"
                />
              </label>
            </div>
            )}
          </div>

          {/* AI Suggestion Feature — 精度不足のため一時非表示（SHOW_AI_SUGGEST を true で再表示） */}
          {SHOW_AI_SUGGEST && (<>
          <div className="bg-gray-800 border border-gray-700 rounded-xl p-3">
            <div className="flex items-center gap-2 mb-2">
              <Globe size={14} className="text-orange-400" />
              <span className="text-xs font-semibold text-gray-200">データセット提案（AI検索）</span>
            </div>

            {/* Mode Selector */}
            <div className="grid grid-cols-3 gap-1.5 mb-2">
              {([
                {
                  key: "fast" as const,
                  label: "⚡ 高速",
                  sub: "人気度のみ・約15秒",
                  desc: "Booruから広域取得し人気度でランク。CLIP・LLMなし。",
                  color: "border-emerald-500 bg-emerald-950/30",
                },
                {
                  key: "balanced" as const,
                  label: "👁 CLIP類似",
                  sub: "視覚類似度・1〜2分",
                  desc: "既存データセットに視覚的に近い画像をCLIPで優先。",
                  color: "border-amber-500 bg-amber-950/30",
                },
                {
                  key: "accurate" as const,
                  label: "🔬 高精度",
                  sub: "CLIP + LLM・10分+",
                  desc: "CLIP + qwen2.5vl:32bで画像を直接解析。最高精度。",
                  color: "border-orange-500 bg-orange-950/30",
                },
              ] as const).map(({ key, label, sub, desc, color }) => (
                <button
                  key={key}
                  onClick={() => setSuggestionMode(key)}
                  className={[
                    "p-2 rounded-lg border-2 text-left transition-all",
                    suggestionMode === key ? color : "border-gray-700 bg-gray-700/30 hover:border-gray-600",
                  ].join(" ")}
                >
                  <div className="text-xs font-bold text-gray-200">{label}</div>
                  <div className="text-xs text-gray-400 mt-0.5">{sub}</div>
                  <div className="text-xs text-gray-500 mt-1 leading-tight">{desc}</div>
                </button>
              ))}
            </div>

            {/* 手動Booruクエリ */}
            <div className="border border-gray-700 rounded-lg p-3 bg-gray-750">
              <label className="text-xs text-gray-400 block mb-1.5">
                Booru 検索クエリ（手動指定）
                <span className="ml-2 text-gray-600">— 空白のままなら自動抽出</span>
              </label>
              <input
                value={manualBooruQuery}
                onChange={(e) => setManualBooruQuery(e.target.value)}
                disabled={suggestionsRunning}
                placeholder="例: hatsune_miku  /  blue_hair twintails  /  painterly_style"
                className="w-full bg-gray-700 border border-gray-600 text-gray-100 placeholder-gray-600 rounded-lg px-3 py-2 text-xs font-mono focus:outline-none focus:border-orange-500 focus:ring-1 focus:ring-orange-500/30 disabled:opacity-50"
              />
              <p className="text-xs text-gray-600 mt-1">
                Safebooru/Konachan タグ形式（スペース区切り）。指定するとキャプションからの自動抽出をスキップします。
                CLIPはデータセット画像との視覚類似度でその後に再ランクします。
              </p>
            </div>

            <div className="flex items-center justify-between">
              <p className="text-xs text-gray-500">
                {suggestionMode === "fast" && "Booruから広域取得し、人気度でランキング。CLIP・LLM評価なし。約15秒。"}
                {suggestionMode === "balanced" && "CLIP視覚類似度で既存データセットに近い画像を優先。人気度補正あり。約1〜2分。"}
                {suggestionMode === "accurate" && "CLIP視覚類似度 + qwen2.5vl:32bで画像を直接解析。最高精度。10分以上。"}
              </p>
              <div className="ml-4 flex-shrink-0 flex gap-2">
                <button
                  onClick={() => void startSuggestions()}
                  disabled={suggestionsRunning || !selectedProject || captionItems.length === 0}
                  className="flex items-center gap-1.5 bg-orange-700 hover:bg-orange-600 disabled:bg-gray-700 disabled:text-gray-500 text-white rounded-lg px-4 py-2 text-xs font-medium transition-colors"
                >
                  {suggestionsRunning ? (
                    <><Loader2 size={14} className="animate-spin" />検索中...</>
                  ) : (
                    <><Globe size={14} />提案を検索</>
                  )}
                </button>
                {suggestionsRunning && (
                  <button
                    onClick={() => void stopSuggestion()}
                    className="flex items-center gap-1.5 bg-red-800 hover:bg-red-700 text-white rounded-lg px-3 py-2 text-xs font-medium transition-colors"
                  >
                    <X size={14} />停止
                  </button>
                )}
              </div>
            </div>

            {captionItems.length === 0 && (
              <div className="mt-3 text-xs text-amber-400 flex gap-2 p-3 bg-amber-950/30 rounded border border-amber-900">
                <AlertTriangle size={14} className="flex-shrink-0 mt-0.5" />
                <span>キャプション生成後に使用できます</span>
              </div>
            )}
          </div>

          {/* Suggestion Results — Pinterest 風フィードバック UI */}
          {suggestionsData && suggestionsData.results && suggestionsData.results.length > 0 && (
            <div className="bg-gray-800 border border-gray-700 rounded-xl overflow-hidden">
              {/* ヘッダー */}
              <div className="px-4 py-3 border-b border-gray-700 flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <Globe size={14} className="text-orange-400" />
                  <span className="text-sm font-semibold text-gray-200">候補画像</span>
                  <span className="text-xs text-gray-500">{suggestionsData.total_count || 0}件</span>
                  {suggestionsData.feedback_round != null && suggestionsData.feedback_round > 0 && (
                    <span className="text-xs bg-orange-900/40 text-orange-300 px-2 py-0.5 rounded-full border border-orange-700/50">
                      第{suggestionsData.feedback_round}ラウンド
                    </span>
                  )}
                  {suggestionsData.source_breakdown && (
                    <span className="text-xs text-gray-600">
                      {Object.entries(suggestionsData.source_breakdown)
                        .filter(([, v]) => v > 0)
                        .map(([k, v]) => `${k}: ${v}`)
                        .join(" · ")}
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-2 text-xs text-gray-400">
                  {likedUrls.size > 0 && (
                    <span className="flex items-center gap-1 text-green-400">
                      <ThumbsUp size={11} /> {likedUrls.size}
                    </span>
                  )}
                  {rejectedUrls.size > 0 && (
                    <span className="flex items-center gap-1 text-red-400">
                      <ThumbsDown size={11} /> {rejectedUrls.size}
                    </span>
                  )}
                </div>
              </div>

              {/* 使い方ヒント */}
              {likedUrls.size === 0 && rejectedUrls.size === 0 && (
                <div className="px-4 py-2 bg-gray-750/50 border-b border-gray-700 text-xs text-gray-500 flex items-center gap-2">
                  <ThumbsUp size={11} className="text-green-500" />
                  いいと思う画像に ✅、不要な画像に ❌ を付けて「再提案」を押すと AI が厳選し直します
                </div>
              )}

              {/* 画像グリッド */}
              <div className="p-4 grid grid-cols-3 gap-3 sm:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 max-h-[32rem] overflow-y-auto">
                {suggestionsData.results.map((img, idx) => {
                  const isLiked = likedUrls.has(img.url);
                  const isRejected = rejectedUrls.has(img.url);
                  return (
                    <div
                      key={`${img.source}-${img.id}-${idx}`}
                      className={[
                        "rounded-lg overflow-hidden border-2 transition-all duration-150 group relative",
                        isLiked
                          ? "border-green-500 ring-1 ring-green-500/40"
                          : isRejected
                            ? "border-red-700 opacity-40"
                            : "border-gray-700 hover:border-gray-500",
                      ].join(" ")}
                    >
                      {/* 画像 */}
                      <div className="aspect-square bg-gray-700 relative overflow-hidden">
                        <img
                          src={img.url}
                          alt={img.title}
                          className={[
                            "w-full h-full object-cover transition-transform",
                            isRejected ? "" : "group-hover:scale-105",
                          ].join(" ")}
                          onError={(e) => {
                            (e.currentTarget.parentElement as HTMLElement).classList.add("flex", "items-center", "justify-center");
                            e.currentTarget.style.display = "none";
                          }}
                        />

                        {/* スコアバッジ */}
                        <div className="absolute top-1 left-1 flex gap-1">
                          <span className={[
                            "text-white text-xs px-1.5 py-0.5 rounded font-semibold",
                            isLiked ? "bg-green-600" : "bg-gray-900/80",
                          ].join(" ")}>
                            {Math.round(img.score || 0)}
                          </span>
                          {img.clip_similarity != null && (
                            <span className="bg-blue-900/80 text-blue-200 text-xs px-1.5 py-0.5 rounded">
                              {Math.round(img.clip_similarity)}%
                            </span>
                          )}
                        </div>

                        {/* ソースバッジ */}
                        <div className="absolute top-1 right-1">
                          <span className="bg-gray-900/70 text-gray-400 text-xs px-1 py-0.5 rounded capitalize">
                            {img.source}
                          </span>
                        </div>

                        {/* 解像度 */}
                        {img.width && img.height && (
                          <div className="absolute bottom-1 left-1 bg-gray-900/70 text-xs text-gray-300 px-1 py-0.5 rounded">
                            {img.width}×{img.height}
                          </div>
                        )}

                        {/* 外部リンク（ホバー時） */}
                        <div className="absolute bottom-1 right-1 opacity-0 group-hover:opacity-100 transition-opacity">
                          <a
                            href={img.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="bg-gray-900/80 hover:bg-orange-700 text-white p-1 rounded block"
                            onClick={(e) => e.stopPropagation()}
                          >
                            <Link size={10} />
                          </a>
                        </div>
                      </div>

                      {/* タイトル */}
                      <div className="px-2 py-1 text-xs text-gray-400 truncate">{img.title}</div>

                      {/* ✅/❌ ボタン */}
                      <div className="flex border-t border-gray-700">
                        <button
                          onClick={() => toggleLike(img.url)}
                          className={[
                            "flex-1 flex items-center justify-center gap-1 py-1.5 text-xs font-medium transition-colors",
                            isLiked
                              ? "bg-green-700 text-white"
                              : "hover:bg-green-900/40 text-gray-500 hover:text-green-400",
                          ].join(" ")}
                          title="いいね（再提案の参考に）"
                        >
                          <ThumbsUp size={11} />
                        </button>
                        <div className="w-px bg-gray-700" />
                        <button
                          onClick={() => toggleReject(img.url)}
                          className={[
                            "flex-1 flex items-center justify-center gap-1 py-1.5 text-xs font-medium transition-colors",
                            isRejected
                              ? "bg-red-900 text-red-300"
                              : "hover:bg-red-900/30 text-gray-500 hover:text-red-400",
                          ].join(" ")}
                          title="不要（次回から除外）"
                        >
                          <ThumbsDown size={11} />
                        </button>
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* フィードバック送信フッター */}
              <div className="px-4 py-3 border-t border-gray-700 flex items-center justify-between bg-gray-800/80">
                <div className="text-xs text-gray-500">
                  {likedUrls.size > 0 || rejectedUrls.size > 0 ? (
                    <span>
                      <span className="text-green-400">{likedUrls.size} 件いいね</span>
                      {rejectedUrls.size > 0 && <span className="text-gray-600"> · </span>}
                      {rejectedUrls.size > 0 && <span className="text-red-400">{rejectedUrls.size} 件除外</span>}
                      <span className="ml-2 text-gray-600">→ 再提案で AI が厳選します</span>
                    </span>
                  ) : (
                    "いいね/除外を付けてから「AI再提案」を押してください"
                  )}
                </div>
                <div className="flex items-center gap-2">
                  {(likedUrls.size > 0 || rejectedUrls.size > 0) && (
                    <button
                      onClick={() => { setLikedUrls(new Set()); setRejectedUrls(new Set()); }}
                      className="text-xs text-gray-500 hover:text-gray-300 flex items-center gap-1 px-2 py-1.5 rounded hover:bg-gray-700 transition-colors"
                    >
                      <RotateCcw size={11} /> リセット
                    </button>
                  )}
                  <button
                    onClick={() => void sendFeedback()}
                    disabled={feedbackRunning || (likedUrls.size === 0 && rejectedUrls.size === 0)}
                    className="flex items-center gap-1.5 bg-orange-700 hover:bg-orange-600 disabled:bg-gray-700 disabled:text-gray-500 text-white rounded-lg px-4 py-2 text-xs font-medium transition-colors"
                  >
                    {feedbackRunning ? (
                      <>
                        <Loader2 size={12} className="animate-spin" />
                        {feedbackStatusMsg || "再提案中..."}
                      </>
                    ) : (
                      <>
                        <Sparkles size={12} />
                        AI再提案
                      </>
                    )}
                  </button>
                </div>
              </div>
            </div>
          )}

          {suggestionsRunning && (
            <div className="bg-gray-800 border border-gray-700 rounded-xl p-3">
              {/* ステップバー */}
              <div className="flex items-center gap-2 mb-3">
                {(["タグ読込", "複数ソース検索", "LLM評価", "保存"] as const).map((label, i) => {
                  const stepNum = i + 1;
                  const done = suggestionStep > stepNum;
                  const active = suggestionStep === stepNum;
                  return (
                    <div key={label} className="flex items-center gap-1 flex-1">
                      <div className={[
                        "w-5 h-5 rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0",
                        done ? "bg-emerald-600 text-white" : active ? "bg-orange-500 text-white" : "bg-gray-700 text-gray-500",
                      ].join(" ")}>
                        {done ? "✓" : stepNum}
                      </div>
                      <span className={["text-xs", active ? "text-orange-300" : done ? "text-emerald-400" : "text-gray-600"].join(" ")}>
                        {label}
                      </span>
                      {i < 3 && <div className={["flex-1 h-px", done ? "bg-emerald-700" : "bg-gray-700"].join(" ")} />}
                    </div>
                  );
                })}
              </div>
              <div className="flex items-center gap-2 mt-2">
                <Loader2 size={14} className="animate-spin text-orange-400 flex-shrink-0" />
                <p className="text-sm text-gray-300 flex-1">{suggestionStatusMsg || "処理中..."}</p>
                <span className="text-xs text-gray-500 flex-shrink-0">
                  {suggestionElapsed > 0 && `${suggestionElapsed}秒`}
                </span>
              </div>
            </div>
          )}
          </>)}{/* /AI Suggestion 一時非表示 */}
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
                {isTagging && (
                  <button
                    onClick={() => void stopWd14()}
                    className="flex items-center gap-1.5 bg-red-800 hover:bg-red-700 text-white rounded-lg px-3 py-1.5 text-xs font-medium transition-colors"
                    title="処理中の画像が終わり次第停止します"
                  >
                    <X size={13} />
                    停止
                  </button>
                )}
              </div>
            </div>

            {/* しきい値パラメーター */}
            <div className="border border-gray-700 rounded-lg p-3 mb-3 space-y-3 bg-gray-750">
              {/* General threshold */}
              <div>
                <div className="flex items-center justify-between mb-1">
                  <label className="text-xs text-gray-300 font-medium">一般タグ しきい値</label>
                  <span className="text-xs font-mono text-violet-300 bg-violet-500/10 px-1.5 py-0.5 rounded">{wd14GeneralThresh.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min={0.10} max={0.80} step={0.05}
                  value={wd14GeneralThresh}
                  onChange={(e) => setWd14GeneralThresh(parseFloat(e.target.value))}
                  disabled={isTagging}
                  className="w-full accent-violet-500 disabled:opacity-50 cursor-pointer h-1.5"
                />
                {/* 目盛り */}
                <div className="relative mt-1 h-4">
                  {[0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80].map((v) => {
                    const pct = ((v - 0.10) / 0.70) * 100;
                    const isDefault = v === 0.35;
                    return (
                      <div key={v} className="absolute flex flex-col items-center" style={{ left: `${pct}%`, transform: "translateX(-50%)" }}>
                        <div className={`w-px ${isDefault ? "h-2 bg-violet-400" : "h-1 bg-gray-600"}`} />
                        {(v === 0.10 || v === 0.35 || v === 0.80) && (
                          <span className={`text-[10px] mt-0.5 whitespace-nowrap ${isDefault ? "text-violet-400" : "text-gray-500"}`}>
                            {v.toFixed(2)}{isDefault ? " (推奨)" : v === 0.10 ? " 多め" : " 厳選"}
                          </span>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
              {/* Character threshold */}
              <div>
                <div className="flex items-center justify-between mb-1">
                  <label className="text-xs text-gray-300 font-medium">キャラクタータグ しきい値</label>
                  <span className="text-xs font-mono text-violet-300 bg-violet-500/10 px-1.5 py-0.5 rounded">{wd14CharacterThresh.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min={0.50} max={0.99} step={0.05}
                  value={wd14CharacterThresh}
                  onChange={(e) => setWd14CharacterThresh(parseFloat(e.target.value))}
                  disabled={isTagging || wd14RemoveCharacterTags}
                  className="w-full accent-violet-500 disabled:opacity-50 cursor-pointer h-1.5"
                />
                {/* 目盛り */}
                <div className="relative mt-1 h-4">
                  {[0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,0.99].map((v) => {
                    const pct = ((v - 0.50) / 0.49) * 100;
                    const isDefault = v === 0.85;
                    return (
                      <div key={v} className="absolute flex flex-col items-center" style={{ left: `${Math.min(pct,100)}%`, transform: "translateX(-50%)" }}>
                        <div className={`w-px ${isDefault ? "h-2 bg-violet-400" : "h-1 bg-gray-600"}`} />
                        {(v === 0.50 || v === 0.85 || v === 0.99) && (
                          <span className={`text-[10px] mt-0.5 whitespace-nowrap ${isDefault ? "text-violet-400" : "text-gray-500"}`}>
                            {v.toFixed(2)}{isDefault ? " (推奨)" : v === 0.50 ? " 多め" : " 厳選"}
                          </span>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
              {/* Remove character tags toggle */}
              <label className="flex items-center gap-2 cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={wd14RemoveCharacterTags}
                  onChange={(e) => setWd14RemoveCharacterTags(e.target.checked)}
                  disabled={isTagging}
                  className="accent-violet-500 w-3.5 h-3.5 disabled:opacity-50"
                />
                <span className="text-xs text-gray-300">キャラクタータグを除外（スタイルLoRAに推奨）</span>
              </label>
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

          {/* ── Prefix Tags (トリガーワード) ── */}
          <div className="bg-gray-800 border border-gray-700 rounded-xl p-4">
            <div className="flex items-center gap-2 mb-3">
              <Tag size={14} className="text-emerald-400" />
              <span className="text-sm font-semibold text-gray-200">Prefix Tags（トリガーワード）</span>
              <span className="text-xs text-gray-500 ml-auto">全キャプションの先頭に付与</span>
            </div>
            {/* 入力 */}
            <div className="flex gap-2 mb-3">
              <input
                type="text"
                value={prefixInput}
                onChange={(e) => setPrefixInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && prefixInput.trim()) {
                    const tag = prefixInput.trim();
                    if (!prefixTags.includes(tag)) setPrefixTags(prev => [...prev, tag]);
                    setPrefixInput("");
                  }
                }}
                placeholder="例: saki_iroha_v1"
                className="flex-1 bg-gray-700 border border-gray-600 rounded-lg px-3 py-1.5 text-sm text-gray-100 placeholder-gray-500 focus:outline-none focus:border-emerald-500"
              />
              <button
                onClick={() => {
                  const tag = prefixInput.trim();
                  if (tag && !prefixTags.includes(tag)) setPrefixTags(prev => [...prev, tag]);
                  setPrefixInput("");
                }}
                className="px-3 py-1.5 bg-emerald-600 hover:bg-emerald-500 text-white text-xs rounded-lg transition-colors"
              >追加</button>
            </div>
            {/* タグチップ */}
            {prefixTags.length > 0 && (
              <div className="flex flex-wrap gap-1.5 mb-3">
                {prefixTags.map((tag, i) => (
                  <span key={i} className="flex items-center gap-1 bg-emerald-900/40 border border-emerald-700/50 text-emerald-300 text-xs px-2 py-0.5 rounded-full">
                    <span className="text-emerald-500 text-[10px]">#{i+1}</span>
                    {tag}
                    <button onClick={() => setPrefixTags(p => p.filter((_, j) => j !== i))} className="text-emerald-500 hover:text-red-400 ml-0.5">×</button>
                  </span>
                ))}
              </div>
            )}
            {/* 保存 + 適用ボタン */}
            <div className="flex gap-2">
              <button
                disabled={tagSettingsSaving || !selectedProject}
                onClick={async () => {
                  if (!selectedProject) return;
                  setTagSettingsSaving(true);
                  try {
                    await apiPost(`/tags/settings/${selectedProject.id}`, { prefix_tags: prefixTags, block_words: blockWords }, "PUT");
                    showNotice("タグ設定を保存しました");
                  } catch { showNotice("保存に失敗しました"); }
                  finally { setTagSettingsSaving(false); }
                }}
                className="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 text-gray-200 text-xs rounded-lg transition-colors disabled:opacity-50"
              >{tagSettingsSaving ? "保存中..." : "設定を保存"}</button>
              <button
                disabled={prefixTags.length === 0 || !selectedProject}
                onClick={async () => {
                  if (!selectedProject) return;
                  try {
                    const r = await apiPost<{ updated_count: number }>(`/tags/apply-prefix`, { project_id: selectedProject.id, prefix_tags: prefixTags });
                    showNotice(`${r.updated_count}件のキャプションに付与しました`);
                    await loadCaptions();
                  } catch { showNotice("適用に失敗しました"); }
                }}
                className="px-3 py-1.5 bg-emerald-700 hover:bg-emerald-600 text-white text-xs rounded-lg transition-colors disabled:opacity-50"
              >全キャプションに適用</button>
            </div>
          </div>

          {/* ── Block Words（ブロックワード） ── */}
          <div className="bg-gray-800 border border-gray-700 rounded-xl p-4">
            <div className="flex items-center gap-2 mb-3">
              <ShieldOff size={14} className="text-red-400" />
              <span className="text-sm font-semibold text-gray-200">Block Words（ブロックワード）</span>
              <span className="text-xs text-gray-500 ml-auto">キャプションから自動除去</span>
            </div>
            {/* 入力 */}
            <div className="flex gap-2 mb-3">
              <input
                type="text"
                value={blockInput}
                onChange={(e) => setBlockInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && blockInput.trim()) {
                    const w = blockInput.trim();
                    if (!blockWords.includes(w)) setBlockWords(prev => [...prev, w]);
                    setBlockInput("");
                  }
                }}
                placeholder="例: 1girl, solo, simple background"
                className="flex-1 bg-gray-700 border border-gray-600 rounded-lg px-3 py-1.5 text-sm text-gray-100 placeholder-gray-500 focus:outline-none focus:border-red-500"
              />
              <button
                onClick={() => {
                  const w = blockInput.trim();
                  if (w && !blockWords.includes(w)) setBlockWords(prev => [...prev, w]);
                  setBlockInput("");
                }}
                className="px-3 py-1.5 bg-red-700 hover:bg-red-600 text-white text-xs rounded-lg transition-colors"
              >追加</button>
            </div>
            {/* ワードチップ + ヒット数 */}
            {blockWords.length > 0 && (
              <div className="flex flex-wrap gap-1.5 mb-3">
                {blockWords.map((w, i) => (
                  <span key={i} className="flex items-center gap-1 bg-red-900/30 border border-red-700/50 text-red-300 text-xs px-2 py-0.5 rounded-full">
                    {w}
                    {blockHits[w] !== undefined && (
                      <span className="bg-red-700 text-white text-[10px] px-1 rounded-full">{blockHits[w]}</span>
                    )}
                    <button onClick={() => setBlockWords(p => p.filter((_, j) => j !== i))} className="text-red-500 hover:text-red-300 ml-0.5">×</button>
                  </span>
                ))}
              </div>
            )}
            <div className="flex gap-2">
              <button
                disabled={blockWords.length === 0 || !selectedProject}
                onClick={async () => {
                  if (!selectedProject) return;
                  try {
                    const r = await apiGet<{ hits: Record<string, number> }>(
                      `/tags/detect-blockwords/${selectedProject.id}?words=${encodeURIComponent(blockWords.join(","))}`
                    );
                    setBlockHits(r.hits);
                    const total = Object.values(r.hits).reduce((a, b) => a + b, 0);
                    showNotice(`${total}件検出（${Object.entries(r.hits).filter(([,v])=>v>0).length}ワード）`);
                  } catch { showNotice("検出に失敗しました"); }
                }}
                className="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 text-gray-200 text-xs rounded-lg transition-colors disabled:opacity-50"
              >検出プレビュー</button>
              <button
                disabled={tagSettingsSaving || !selectedProject}
                onClick={async () => {
                  if (!selectedProject) return;
                  setTagSettingsSaving(true);
                  try {
                    await apiPost(`/tags/settings/${selectedProject.id}`, { prefix_tags: prefixTags, block_words: blockWords }, "PUT");
                    showNotice("設定を保存しました");
                  } catch { showNotice("保存に失敗しました"); }
                  finally { setTagSettingsSaving(false); }
                }}
                className="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 text-gray-200 text-xs rounded-lg transition-colors disabled:opacity-50"
              >設定を保存</button>
              <button
                disabled={blockWords.length === 0 || !selectedProject}
                onClick={async () => {
                  if (!selectedProject) return;
                  try {
                    const r = await apiPost<{ updated_count: number; total_removed: number }>(`/tags/remove-blockwords`, { project_id: selectedProject.id, block_words: blockWords });
                    showNotice(`${r.updated_count}件から合計${r.total_removed}タグを削除しました`);
                    setBlockHits({});
                    await loadCaptions();
                  } catch { showNotice("削除に失敗しました"); }
                }}
                className="px-3 py-1.5 bg-red-700 hover:bg-red-600 text-white text-xs rounded-lg transition-colors disabled:opacity-50"
              >全キャプションから削除</button>
            </div>
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
              {filteredCaptionItems.map((item, filteredIdx) => (
                <CaptionCard
                  key={item.id}
                  item={item}
                  isFirst={filteredIdx === 0}
                  isLast={filteredIdx === filteredCaptionItems.length - 1}
                  isEditing={editingId === item.id}
                  editingText={editingId === item.id ? editingText : item.caption}
                  onStartEdit={() => { setEditingId(item.id); setEditingText(item.caption); }}
                  onChangeText={setEditingText}
                  onSave={async () => {
                    await saveCaption(item.id, editingText);
                    setEditingId(null);
                  }}
                  onCancel={() => setEditingId(null)}
                  onSaveCaption={(c) => saveCaption(item.id, c)}
                  prefixTags={prefixTags}
                  onDelete={() => void deleteDatasetItem(item.id)}
                  onMoveUp={() => {
                    if (filteredIdx === 0) return;
                    const prevId = filteredCaptionItems[filteredIdx - 1].id;
                    setCaptionItems(prev => {
                      const next = [...prev];
                      const a = next.findIndex(c => c.id === item.id);
                      const b = next.findIndex(c => c.id === prevId);
                      if (a === -1 || b === -1) return prev;
                      [next[a], next[b]] = [next[b], next[a]];
                      return next;
                    });
                  }}
                  onMoveDown={() => {
                    if (filteredIdx >= filteredCaptionItems.length - 1) return;
                    const nextId = filteredCaptionItems[filteredIdx + 1].id;
                    setCaptionItems(prev => {
                      const next = [...prev];
                      const a = next.findIndex(c => c.id === item.id);
                      const b = next.findIndex(c => c.id === nextId);
                      if (a === -1 || b === -1) return prev;
                      [next[a], next[b]] = [next[b], next[a]];
                      return next;
                    });
                  }}
                  isDragOver={captionDragOverId === item.id}
                  onDragStart={() => setCaptionDragFromId(item.id)}
                  onDragOver={(e) => { e.preventDefault(); setCaptionDragOverId(item.id); }}
                  onDrop={() => {
                    if (captionDragFromId !== null && captionDragFromId !== item.id) {
                      setCaptionItems(prev => {
                        const fromIdx = prev.findIndex(c => c.id === captionDragFromId);
                        const toIdx = prev.findIndex(c => c.id === item.id);
                        if (fromIdx === -1 || toIdx === -1) return prev;
                        const next = [...prev];
                        const [moved] = next.splice(fromIdx, 1);
                        next.splice(toIdx, 0, moved);
                        return next;
                      });
                    }
                    setCaptionDragFromId(null);
                    setCaptionDragOverId(null);
                  }}
                  onDragEnd={() => { setCaptionDragFromId(null); setCaptionDragOverId(null); }}
                />
              ))}
            </div>
          )}
          {selectedProjectId && (
            <OutfitClassifier projectId={selectedProjectId} captionItems={captionItems} />
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
                    {(() => {
                      const removable = dashboardData.similarity_groups
                        .filter((g) => g.type === "duplicate" && g.item_ids.length > 1)
                        .reduce((acc, g) => acc + g.item_ids.length - 1, 0);
                      return removable > 0 ? (
                        <button
                          onClick={() => void bulkRemoveDuplicates()}
                          className="ml-auto flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg bg-red-900/60 hover:bg-red-800 text-red-200 border border-red-800/60 font-medium transition-colors"
                          title="各重複グループで1枚だけ残し、残りをまとめて削除（ファイルはゴミ箱へ）"
                        >
                          <Trash2 size={13} />重複を一括削除 ({removable})
                        </button>
                      ) : null;
                    })()}
                  </div>
                  <div className="divide-y divide-gray-700">
                    {dashboardData.similarity_groups.map((group, idx) => (
                      <SimilarityGroupRow
                        key={idx}
                        group={group}
                        idx={idx}
                        expanded={expandedGroups.has(idx)}
                        onToggle={() => toggleGroup(idx)}
                        pathById={dupPathById}
                        onDelete={deleteItemsHard}
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

      {/* Preprocessing — 品質確認と同一画面（処理後に評価を更新） */}
      {activeTab === "dashboard" && (
        <div className="space-y-2 mt-5 pt-4 border-t border-gray-800">
          <div className="flex items-center gap-2 mb-1">
            <Wand2 size={15} className="text-orange-400" />
            <h3 className="text-sm font-semibold text-gray-200">前処理（解像度標準化・文字削除）</h3>
            <span className="text-[10px] text-gray-500">処理後に品質スコアを自動で再評価します</span>
          </div>
          {/* Mode Selector — 横並びタブ型 */}
          <div className="flex gap-1.5">
            {([
              { key: "resolution", label: "解像度標準化", icon: <Layers size={13} />, sub: "アスペクト維持リサイズ" },
              { key: "text_removal", label: "文字削除", icon: <Edit3 size={13} />, sub: "OCR & inpainting" },
              { key: "leak_masking", label: "特徴マスキング", icon: <AlertTriangle size={13} />, sub: "キャラ識別を困難に" },
            ] as const).map(({ key, label, icon, sub }) => (
              <button
                key={key}
                onClick={() => setPreprocessingMode(key)}
                className={[
                  "flex-1 flex items-center gap-2 px-3 py-2 rounded-lg border transition-all text-left",
                  preprocessingMode === key
                    ? "border-orange-500 bg-orange-950/30"
                    : "border-gray-700 bg-gray-800 hover:border-gray-600",
                ].join(" ")}
              >
                <span className="text-gray-400 flex-shrink-0">{icon}</span>
                <div>
                  <div className="text-xs font-medium text-gray-200">{label}</div>
                  <div className="text-[10px] text-gray-500">{sub}</div>
                </div>
              </button>
            ))}
          </div>

          {/* Resolution Settings */}
          {preprocessingMode === "resolution" && (
            <div className="bg-gray-800 border border-gray-700 rounded-xl p-3 space-y-2">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  目標解像度（長辺）
                </label>
                <div className="flex gap-2 items-center">
                  <select
                    value={resolutionTarget}
                    onChange={(e) => setResolutionTarget(e.target.value)}
                    className="flex-1 bg-gray-700 border border-gray-600 text-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-orange-500"
                  >
                    <option value="512">512 px</option>
                    <option value="640">640 px</option>
                    <option value="768">768 px (推奨)</option>
                    <option value="1024">1024 px</option>
                  </select>
                </div>
              </div>
              <label className="flex items-center gap-2 text-sm text-gray-300">
                <input
                  type="checkbox"
                  checked={keepAspectRatio}
                  onChange={(e) => setKeepAspectRatio(e.target.checked)}
                  className="w-4 h-4 bg-gray-700 border border-gray-600 rounded cursor-pointer"
                />
                アスペクト比を維持
              </label>
              <label className="flex items-center gap-2 text-sm text-gray-300">
                <input
                  type="checkbox"
                  checked={useUpscale}
                  onChange={(e) => setUseUpscale(e.target.checked)}
                  className="w-4 h-4 bg-gray-700 border border-gray-600 rounded cursor-pointer"
                />
                AIアップスケール（ESRGAN）で拡大
                {comfyReady === false && <span className="text-[10px] text-amber-400">ComfyUI未接続</span>}
              </label>
              {useUpscale && (
                <select
                  value={upscaleModel}
                  onChange={(e) => setUpscaleModel(e.target.value)}
                  className="w-full bg-gray-700 border border-gray-600 text-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-orange-500"
                >
                  {comfyUpscaleModels.length === 0 && <option value="">（モデル取得中…）</option>}
                  {comfyUpscaleModels.map((m) => (
                    <option key={m} value={m}>{m}</option>
                  ))}
                </select>
              )}
              <p className="text-xs text-gray-500">
                {useUpscale
                  ? "目標より小さい画像のみ ComfyUI の ESRGAN モデルでAI拡大し、目標寸法に合わせます（要 ComfyUI 起動・1枚あたり数秒）。"
                  : "目標より大きい画像を縮小します（LANCZOS）。低解像度画像をAIで高解像度化するには上のトグルを有効化してください。"}
              </p>
            </div>
          )}

          {/* Text Removal Settings */}
          {preprocessingMode === "text_removal" && (
            <div className="bg-gray-800 border border-gray-700 rounded-xl p-3 space-y-3">
              <p className="text-sm text-gray-400">
                画像内のテキスト（吹き出し・擬音・透かし等）を CLIPSeg で検出し、LaMa inpainting で自然に消去します（ComfyUI エンジン使用）。
              </p>
              <div>
                <label className="block text-xs font-medium text-gray-300 mb-1">
                  マスク拡張（grow）: {textRemovalGrow}px
                </label>
                <input
                  type="range"
                  min={-16}
                  max={32}
                  step={1}
                  value={textRemovalGrow}
                  onChange={(e) => setTextRemovalGrow(parseInt(e.target.value, 10))}
                  className="w-full accent-orange-500"
                />
                <p className="text-[10px] text-gray-500 mt-1">大きいほど文字周辺を広く消去（消し残り対策）。既定 8px。</p>
              </div>
              <div className="text-xs text-amber-400 flex gap-2 p-3 bg-amber-950/30 rounded border border-amber-900">
                <AlertTriangle size={14} className="flex-shrink-0 mt-0.5" />
                <span>ComfyUI 起動が必要。選択画像があればその枚数のみ、無ければ全画像が対象。原本は自動バックアップされます。</span>
              </div>
            </div>
          )}

          {/* Auto Caption Settings — Resize/Cleanup 後の画像に WD14 タグを自動付与 */}
          {preprocessingMode !== "leak_masking" && (
            <div className="bg-gray-800 border border-gray-700 rounded-xl p-3 space-y-2">
              <label className="flex items-center gap-2 text-sm text-gray-300">
                <input
                  type="checkbox"
                  checked={useCaption}
                  onChange={(e) => setUseCaption(e.target.checked)}
                  className="w-4 h-4 bg-gray-700 border border-gray-600 rounded cursor-pointer"
                />
                Auto Caption（WD14）を自動生成
              </label>
              {useCaption && (
                <div className="grid grid-cols-2 gap-2">
                  <label className="text-xs text-gray-400">
                    一般タグしきい値 ({captionGeneralThresh.toFixed(2)})
                    <input
                      type="range" min={0.05} max={0.95} step={0.05}
                      value={captionGeneralThresh}
                      onChange={(e) => setCaptionGeneralThresh(parseFloat(e.target.value))}
                      className="w-full"
                    />
                  </label>
                  <label className="text-xs text-gray-400">
                    キャラタグしきい値 ({captionCharacterThresh.toFixed(2)})
                    <input
                      type="range" min={0.05} max={0.99} step={0.05}
                      value={captionCharacterThresh}
                      onChange={(e) => setCaptionCharacterThresh(parseFloat(e.target.value))}
                      className="w-full"
                    />
                  </label>
                  <label className="flex items-center gap-2 text-xs text-gray-400 col-span-2">
                    <input
                      type="checkbox"
                      checked={captionRemoveCharacterTags}
                      onChange={(e) => setCaptionRemoveCharacterTags(e.target.checked)}
                      className="w-4 h-4 bg-gray-700 border border-gray-600 rounded cursor-pointer"
                    />
                    キャラクタータグを除外
                  </label>
                </div>
              )}
              <p className="text-xs text-gray-500">
                処理後の画像から WD14 でタグを生成し、image.png と同名の image.txt を保存します（既存タグ編集タブと同じ規約）。
              </p>
            </div>
          )}

          {/* Leak Masking Settings */}
          {preprocessingMode === "leak_masking" && (
            <div className="bg-gray-800 border border-gray-700 rounded-xl p-3">
              <p className="text-sm text-gray-400 mb-4">
                WD14 キャラタグから検出された特徴的な髪色・髪型・衣装などを、マスキング処理でぼかします。学習時のキャラクター識別を困難にします。
              </p>
              <div className="text-xs text-blue-400 flex gap-2 p-3 bg-blue-950/30 rounded border border-blue-900">
                <AlertTriangle size={14} className="flex-shrink-0 mt-0.5" />
                <span>実装中：Phase 6 予定</span>
              </div>
            </div>
          )}

          {/* Action Button */}
          <div className="flex gap-2">
            <button
              onClick={async () => {
                if (!selectedProject) {
                  showError("プロジェクトを選択してください");
                  return;
                }
                if (preprocessingRunning) {
                  try {
                    await apiPost(`/dataset/preprocess-pipeline-cancel/${selectedProject.id}`, {});
                  } catch (e: unknown) {
                    showError(e instanceof Error ? e.message : "キャンセルに失敗しました");
                  }
                  return;
                }
                if (preprocessingMode === "leak_masking") {
                  showError("特徴マスキングは未実装です");
                  return;
                }
                setPreprocessingRunning(true);
                setPreprocessingProgress(0);
                setPipelineItems([]);
                try {
                  const resizeMode = keepAspectRatio ? "resize_longer" : "square_crop";
                  const useQwen = preprocessingMode === "text_removal";
                  const idsParam = selectedIds.length > 0 ? `&item_ids=${selectedIds.join(",")}` : "";
                  // Pipeline設定をProjectへ保存（次回このプロジェクトを開いたときに復元される）
                  try {
                    await apiPost(`/dataset/pipeline-config/${selectedProject.id}`, {
                      target_size: Number(resolutionTarget),
                      resize_mode: resizeMode,
                      use_esrgan: useUpscale,
                      esrgan_model: upscaleModel,
                      use_qwen: useQwen,
                      use_caption: useCaption,
                      caption_general_thresh: captionGeneralThresh,
                      caption_character_thresh: captionCharacterThresh,
                      caption_remove_character_tags: captionRemoveCharacterTags,
                    });
                  } catch {
                    // 設定保存の失敗は前処理の実行自体を妨げない
                  }
                  const captionParam = useCaption
                    ? `&use_caption=true&caption_general_thresh=${captionGeneralThresh}&caption_character_thresh=${captionCharacterThresh}&caption_remove_character_tags=${captionRemoveCharacterTags}`
                    : "";
                  await apiPost(
                    `/dataset/preprocess-pipeline/${selectedProject.id}?target_size=${resolutionTarget}&resize_mode=${resizeMode}&use_esrgan=${useUpscale}&esrgan_model=${encodeURIComponent(upscaleModel)}&use_qwen=${useQwen}${captionParam}${idsParam}`,
                    {}
                  );
                  // パイプラインステータスポーリング（Qwen/Caption使用時は長め）
                  const pollInterval = (useQwen || useCaption) ? 2000 : 1000;
                  const maxAttempts = (useQwen || useCaption) ? 3000 : 600;
                  for (let attempt = 0; attempt < maxAttempts; attempt++) {
                    await new Promise(r => setTimeout(r, pollInterval));
                    const st = await apiGet<{
                      status: string; message: string; done: number; total: number; failed: number; skipped: number;
                      current_step: string; items?: { item_id: number; file_name: string; status: string; error: string | null; output_path?: string }[];
                    }>(
                      `/dataset/preprocess-pipeline-status/${selectedProject.id}`
                    );
                    if (st.total > 0) {
                      setPreprocessingProgress(Math.round((st.done / st.total) * 100));
                    }
                    setPreprocessingQueued(st.status === "queued");
                    if (st.status === "queued") {
                      setPreprocessingStep(st.message);
                    } else if (st.current_step) {
                      setPreprocessingStep(st.current_step);
                    }
                    if (st.status === "done") {
                      showNotice(st.message);
                      setPipelineItems(st.items ?? []);
                      void startAnalysis();
                      break;
                    }
                    if (st.status === "failed") {
                      showError(st.message);
                      setPipelineItems(st.items ?? []);
                      break;
                    }
                    if (st.status === "cancelled") {
                      showNotice(st.message);
                      setPipelineItems(st.items ?? []);
                      void startAnalysis();
                      break;
                    }
                  }
                } catch (e: unknown) {
                  showError(e instanceof Error ? e.message : "前処理に失敗しました");
                } finally {
                  setPreprocessingRunning(false);
                  setPreprocessingQueued(false);
                  setPreprocessingProgress(0);
                  setPreprocessingStep("");
                }
              }}
              className={`flex items-center gap-2 rounded-lg px-6 py-3 font-medium transition-colors text-white ${preprocessingRunning ? (preprocessingQueued ? "bg-amber-700 hover:bg-amber-600" : "bg-red-700 hover:bg-red-600") : "bg-orange-700 hover:bg-orange-600"}`}
            >
              {preprocessingRunning ? (
                <Loader2 size={18} className="animate-spin" />
              ) : (
                <Wand2 size={18} />
              )}
              {preprocessingRunning
                ? (preprocessingQueued ? "キュー待機中 — クリックでキャンセル" : `実行中 (${preprocessingProgress}%) — クリックでキャンセル`)
                : preprocessingMode === "resolution"
                  ? `解像度標準化（全画像）`
                  : preprocessingMode === "text_removal"
                    ? (selectedIds.length > 0 ? `文字削除（${selectedIds.length}枚）` : "文字削除（全画像）")
                    : "特徴マスキング [未実装]"}
            </button>
            {selectedIds.length > 0 && (
              <span className="flex items-center text-xs text-gray-500">
                {selectedIds.length}/{displayItems.length} 選択中
              </span>
            )}
          </div>

          {preprocessingRunning && (
            <div className="bg-gray-800 border border-gray-700 rounded-xl p-4">
              <div className="w-full bg-gray-700 rounded-full h-2 overflow-hidden">
                <div
                  className={`h-full transition-all duration-300 ${preprocessingQueued ? "bg-amber-500 w-full animate-pulse" : "bg-orange-500"}`}
                  style={preprocessingQueued ? undefined : { width: `${preprocessingProgress}%` }}
                />
              </div>
              <p className="text-xs text-gray-400 mt-2 text-center">
                {preprocessingQueued ? preprocessingStep : `${preprocessingProgress}% 完了${preprocessingStep ? ` — ${preprocessingStep}` : ""}`}
              </p>
            </div>
          )}

          {/* Pipeline実行結果 — 既存の /collector/thumbnail サムネイルを流用（新規コンポーネントを追加しない） */}
          {!preprocessingRunning && pipelineItems.length > 0 && (
            <div className="bg-gray-800 border border-gray-700 rounded-xl p-4 space-y-2">
              <div className="flex items-center justify-between">
                <h4 className="text-xs font-semibold text-gray-300">
                  処理結果（成功 {pipelineItems.filter((i) => i.status === "success").length} / 失敗 {pipelineItems.filter((i) => i.status === "failed").length} / スキップ {pipelineItems.filter((i) => i.status === "skipped").length}）
                </h4>
              </div>
              <div className="grid grid-cols-4 gap-2 sm:grid-cols-6 lg:grid-cols-8 max-h-64 overflow-y-auto">
                {pipelineItems.map((it) => (
                  <div key={it.item_id} className="space-y-1">
                    <div className={[
                      "aspect-square rounded-lg overflow-hidden border",
                      it.status === "success" ? "border-emerald-700" : it.status === "failed" ? "border-red-700" : "border-gray-700",
                    ].join(" ")}>
                      {it.status === "success" && it.output_path ? (
                        <img
                          src={`${API_BASE}/collector/thumbnail?path=${encodeURIComponent(it.output_path)}&size=112`}
                          alt={it.file_name}
                          className="w-full h-full object-cover"
                        />
                      ) : (
                        <div className="w-full h-full flex items-center justify-center bg-gray-900 text-[9px] text-gray-500 text-center px-1">
                          {it.status === "failed" ? "失敗" : "スキップ"}
                        </div>
                      )}
                    </div>
                    <p className="text-[9px] text-gray-500 truncate" title={it.error ?? it.file_name}>{it.file_name}</p>
                    {it.error && (
                      <p className="text-[9px] text-red-400 truncate" title={it.error}>{it.error}</p>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* ─────────────────────────── ミキサータブ (§10) ─────────────────────────── */}
      {activeTab === "mixer" && (() => {
        const FEATURES: { key: MixerFeatureKey; label: string; desc: string; color: string }[] = [
          { key: "face",        label: "顔・表情",   desc: "顔の造形・表情を学習",       color: "bg-rose-500" },
          { key: "hair",        label: "髪",         desc: "髪色・髪型・髪の質感",       color: "bg-amber-500" },
          { key: "costume",     label: "衣装",       desc: "服装・コスチュームの特徴",   color: "bg-violet-500" },
          { key: "accessory",   label: "アクセサリ", desc: "装飾品・小物・持ち物",       color: "bg-pink-500" },
          { key: "background",  label: "背景",       desc: "背景・舞台・シチュエーション", color: "bg-sky-500" },
          { key: "line",        label: "線",         desc: "タッチ・線の太さ・スタイル", color: "bg-gray-400" },
          { key: "color",       label: "色使い",     desc: "配色・グラデーション・彩度", color: "bg-emerald-500" },
          { key: "lighting",    label: "光源",       desc: "光源・陰影・立体感",         color: "bg-yellow-500" },
          { key: "composition", label: "構図",       desc: "アングル・フレーミング",     color: "bg-cyan-500" },
          { key: "mood",        label: "雰囲気",     desc: "全体的なトーン・ムード",     color: "bg-indigo-400" },
          { key: "expression",  label: "感情",       desc: "表情・感情表現",             color: "bg-orange-400" },
        ];

        const PRESETS: { label: string; weights: Partial<MixerWeights> }[] = [
          {
            label: "Character LoRA",
            weights: { face: 100, hair: 100, costume: 100, accessory: 80, expression: 80, line: 40, color: 30, background: 20, composition: 30, lighting: 30, mood: 20 },
          },
          {
            label: "Style LoRA",
            weights: { face: 80, hair: 20, costume: 10, accessory: 5, expression: 50, line: 100, color: 100, background: 60, composition: 70, lighting: 80, mood: 90 },
          },
          {
            label: "Hybrid",
            weights: { face: 90, hair: 70, costume: 60, accessory: 50, expression: 70, line: 70, color: 70, background: 40, composition: 50, lighting: 60, mood: 60 },
          },
          {
            label: "リセット",
            weights: { face: 50, hair: 50, costume: 50, accessory: 50, expression: 50, line: 50, color: 50, background: 50, composition: 50, lighting: 50, mood: 50 },
          },
        ];

        function weightColor(v: number) {
          if (v >= 80) return "text-emerald-400";
          if (v >= 50) return "text-gray-300";
          if (v >= 20) return "text-amber-400";
          return "text-red-400";
        }

        return (
          <div className="space-y-3">
            {/* ヘッダー説明 */}
            <div className="bg-gray-800/60 border border-gray-700 rounded-xl p-3">
              <p className="text-xs text-gray-400 leading-relaxed">
                各特徴の<strong className="text-gray-200">学習ウェイト（0〜100）</strong>を設定します。
                高い値ほどその特徴をデータセットから多く学習します。
                キャプション生成時に重みを考慮し、低ウェイトのカテゴリタグを削除候補として警告します。
              </p>
            </div>

            {/* プリセットボタン */}
            <div>
              <p className="text-xs text-gray-500 mb-1.5">プリセット</p>
              <div className="flex gap-2 flex-wrap">
                {PRESETS.map((p) => (
                  <button
                    key={p.label}
                    onClick={() => setMixerWeights((prev) => ({ ...prev, ...p.weights } as MixerWeights))}
                    className="text-xs px-3 py-1.5 rounded-lg bg-gray-700 hover:bg-gray-600 text-gray-300 transition-colors"
                  >
                    {p.label}
                  </button>
                ))}
              </div>
            </div>

            {/* スライダー一覧 */}
            <div className="space-y-2">
              {FEATURES.map(({ key, label, desc, color }) => {
                const val = mixerWeights[key] ?? 50;
                return (
                  <div key={key} className="bg-gray-800 border border-gray-700 rounded-xl px-3 py-2.5">
                    <div className="flex items-center gap-2 mb-1.5">
                      <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${color}`} />
                      <span className="text-xs font-medium text-gray-200 w-20 flex-shrink-0">{label}</span>
                      <span className="text-[10px] text-gray-500 flex-1">{desc}</span>
                      <span className={`text-xs font-mono font-bold w-8 text-right ${weightColor(val)}`}>{val}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] text-gray-600 w-4">0</span>
                      <input
                        type="range"
                        min={0}
                        max={100}
                        step={5}
                        value={val}
                        onChange={(e) => setMixerWeights((prev) => ({ ...prev, [key]: Number(e.target.value) }))}
                        className="flex-1 h-1.5 rounded-full appearance-none cursor-pointer"
                        style={{
                          background: `linear-gradient(to right, #f97316 0%, #f97316 ${val}%, #374151 ${val}%, #374151 100%)`,
                        }}
                      />
                      <span className="text-[10px] text-gray-600 w-6 text-right">100</span>
                    </div>
                  </div>
                );
              })}
            </div>

            {/* 保存ボタン */}
            <div className="flex items-center gap-3 pt-1">
              <button
                disabled={mixerSaving || !selectedProject}
                onClick={async () => {
                  if (!selectedProject) return;
                  setMixerSaving(true);
                  setMixerSaved(false);
                  try {
                    await fetch(`${API_BASE}/dataset/mixer/${selectedProject.id}`, {
                      method: "PUT",
                      headers: { "Content-Type": "application/json" },
                      body: JSON.stringify({ feature_weights: mixerWeights }),
                    });
                    setMixerSaved(true);
                    showNotice("ミキサー設定を保存しました");
                    setTimeout(() => setMixerSaved(false), 3000);
                  } catch {
                    showError("保存に失敗しました");
                  } finally {
                    setMixerSaving(false);
                  }
                }}
                className="flex items-center gap-2 bg-orange-700 hover:bg-orange-600 disabled:bg-gray-700 disabled:text-gray-500 text-white rounded-lg px-5 py-2 text-sm font-medium transition-colors"
              >
                {mixerSaving ? <Loader2 size={14} className="animate-spin" /> : mixerSaved ? <Check size={14} /> : <Wand2 size={14} />}
                {mixerSaving ? "保存中..." : mixerSaved ? "保存済み" : "ウェイトを保存"}
              </button>
              <p className="text-xs text-gray-500">設定はプロジェクトに紐付けて保存されます</p>
            </div>
          </div>
        );
      })()}

      {/* Lightbox Modal */}
      {selectedPreviewItem && (
        <div
          className="fixed inset-0 bg-black/80 flex items-center justify-center z-50"
          onClick={() => { setSelectedPreviewItem(null); setContextMenuPos(null); }}
        >
          <div
            className="bg-gray-800 border border-gray-700 rounded-xl overflow-hidden max-w-2xl w-full mx-4"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between px-4 py-3 border-b border-gray-700">
              <h3 className="text-sm font-semibold text-gray-200">{selectedPreviewItem.title}</h3>
              <button
                onClick={() => { setSelectedPreviewItem(null); setContextMenuPos(null); }}
                className="p-1 hover:bg-gray-700 rounded transition-colors"
              >
                <X size={18} className="text-gray-400" />
              </button>
            </div>
            <div className="flex flex-col md:flex-row">
              {/* Image */}
              <div className="md:flex-1 bg-gray-700 p-4 flex items-center justify-center min-h-96">
                {selectedPreviewItem.thumbnail_url ? (
                  <img
                    src={selectedPreviewItem.thumbnail_url}
                    alt={selectedPreviewItem.title}
                    className="max-w-full max-h-96 rounded"
                  />
                ) : (
                  <div className="flex flex-col items-center gap-2 text-gray-500">
                    <ImageOff size={32} />
                    <span className="text-sm">画像なし</span>
                  </div>
                )}
              </div>
              {/* Details */}
              <div className="md:w-64 p-4 border-t md:border-t-0 md:border-l border-gray-700 bg-gray-750 overflow-y-auto max-h-96">
                <div className="space-y-4">
                  <div>
                    <h4 className="text-xs font-semibold text-gray-400 mb-2">解像度</h4>
                    <p className="text-sm text-gray-300">{selectedPreviewItem.width} × {selectedPreviewItem.height}</p>
                  </div>
                  <div>
                    <h4 className="text-xs font-semibold text-gray-400 mb-2">アスペクト比</h4>
                    <p className="text-sm text-gray-300">{selectedPreviewItem.aspect}</p>
                  </div>
                  <div>
                    <h4 className="text-xs font-semibold text-gray-400 mb-2">ファイル名</h4>
                    <p className="text-xs text-gray-400 break-all">{selectedPreviewItem.title}</p>
                  </div>
                  {(selectedPreviewItem.tags || []).length > 0 && (
                    <div>
                      <h4 className="text-xs font-semibold text-gray-400 mb-2">タグ</h4>
                      <div className="flex flex-wrap gap-1">
                        {selectedPreviewItem.tags?.map((tag) => (
                          <span key={tag} className="text-xs bg-indigo-900/50 text-indigo-300 px-2 py-1 rounded">
                            {tag}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                  <button
                    onClick={() => { void removeItems([selectedPreviewItem.id]); setSelectedPreviewItem(null); }}
                    className="w-full flex items-center justify-center gap-2 bg-red-900/50 hover:bg-red-900 text-red-300 rounded px-3 py-2 text-xs font-medium transition-colors"
                  >
                    <Trash2 size={13} />
                    削除
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Context Menu */}
      {contextMenuPos && selectedPreviewItem && (
        <div
          className="fixed z-40"
          style={{ top: `${contextMenuPos.y}px`, left: `${contextMenuPos.x}px` }}
          onClick={() => setContextMenuPos(null)}
        >
          <div className="bg-gray-800 border border-gray-700 rounded-lg shadow-lg overflow-hidden min-w-48">
            <button
              onClick={() => setContextMenuPos(null)}
              className="w-full text-left px-4 py-2 text-sm text-gray-300 hover:bg-gray-700 transition-colors flex items-center gap-2"
            >
              <Edit3 size={14} />
              詳細表示
            </button>
            <button
              onClick={() => { void removeItems([selectedPreviewItem.id]); setContextMenuPos(null); setSelectedPreviewItem(null); }}
              className="w-full text-left px-4 py-2 text-sm text-red-400 hover:bg-red-900/30 transition-colors flex items-center gap-2"
            >
              <Trash2 size={14} />
              削除
            </button>
          </div>
        </div>
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
  pathById,
  onDelete,
}: {
  group: SimilarityGroup;
  idx: number;
  expanded: boolean;
  onToggle: () => void;
  pathById: Map<number, string>;
  onDelete: (ids: number[], deleteFiles: boolean) => Promise<void>;
}) {
  const isDup = group.type === "duplicate";
  // 重複は先頭1枚を残し残りを削除候補に（既定選択）
  const [sel, setSel] = useState<Set<number>>(() => new Set(isDup ? group.item_ids.slice(1) : []));
  const [busy, setBusy] = useState(false);
  const toggle = (id: number) =>
    setSel((s) => {
      const n = new Set(s);
      n.has(id) ? n.delete(id) : n.add(id);
      return n;
    });
  const runDelete = async () => {
    if (sel.size === 0) return;
    setBusy(true);
    try {
      await onDelete([...sel], true);
      setSel(new Set());
    } finally {
      setBusy(false);
    }
  };
  return (
    <div>
      <button
        onClick={onToggle}
        className="w-full flex items-center gap-3 px-4 py-2.5 hover:bg-gray-700/50 transition-colors text-left"
      >
        {expanded ? <ChevronUp size={13} className="text-gray-500" /> : <ChevronDown size={13} className="text-gray-500" />}
        <span className={`text-xs font-medium px-1.5 py-0.5 rounded ${isDup ? "bg-red-900 text-red-300" : "bg-amber-900 text-amber-300"}`}>
          {isDup ? "重複" : "類似"}
        </span>
        <span className="text-sm text-gray-300">グループ #{idx + 1}</span>
        <span className="text-xs text-gray-500">{group.item_ids.length}枚</span>
      </button>
      {expanded && (
        <div className="px-4 pb-3 space-y-2">
          <div className="flex flex-wrap gap-2">
            {group.item_ids.map((id) => {
              const path = pathById.get(id);
              const checked = sel.has(id);
              return (
                <div
                  key={id}
                  onClick={() => toggle(id)}
                  className="relative cursor-pointer"
                  title={`#${id}`}
                >
                  {path ? (
                    <img
                      src={`${API_BASE}/collector/thumbnail?path=${encodeURIComponent(path)}&size=96`}
                      className={[
                        "w-20 h-20 object-cover rounded border-2 transition-all",
                        checked ? "border-red-500 opacity-50" : "border-gray-600 hover:border-gray-400",
                      ].join(" ")}
                    />
                  ) : (
                    <div className="w-20 h-20 rounded border-2 border-gray-700 bg-gray-800 flex items-center justify-center text-[10px] text-gray-500">
                      #{id}
                    </div>
                  )}
                  <input
                    type="checkbox"
                    checked={checked}
                    onChange={() => toggle(id)}
                    onClick={(e) => e.stopPropagation()}
                    className="absolute top-1 left-1 w-3.5 h-3.5 accent-red-500"
                  />
                  <span className="absolute bottom-0 right-0 text-[9px] bg-black/70 text-gray-200 px-1 rounded-tl">#{id}</span>
                </div>
              );
            })}
          </div>
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-xs text-gray-500">{sel.size}枚選択</span>
            <button
              disabled={busy || sel.size === 0}
              onClick={() => void runDelete()}
              className="flex items-center gap-1 text-xs bg-red-800 hover:bg-red-700 disabled:bg-gray-700 disabled:text-gray-500 text-red-100 rounded px-2.5 py-1 transition-colors"
            >
              <Trash2 size={12} />
              選択を削除（ゴミ箱へ）
            </button>
            <button
              onClick={() => setSel(new Set(group.item_ids.slice(1)))}
              className="text-xs text-gray-400 hover:text-gray-200"
            >
              最初以外を選択
            </button>
            <button
              onClick={() => setSel(new Set())}
              className="text-xs text-gray-400 hover:text-gray-200"
            >
              選択解除
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

function CaptionTags({
  caption,
  prefixTags,
  onSaveCaption,
  onStartEdit,
}: {
  caption: string;
  prefixTags: string[];
  onSaveCaption: (caption: string) => Promise<void>;
  onStartEdit: () => void;
}) {
  const [adding, setAdding] = useState(false);
  const [newTag, setNewTag] = useState("");
  const [selected, setSelected] = useState<Set<number>>(new Set());
  const [lastIdx, setLastIdx] = useState<number | null>(null);
  const tags = caption.split(",").map((t) => t.trim()).filter(Boolean);
  const prefixSet = new Set(prefixTags.map((p) => p.toLowerCase()));

  const removeTag = (idx: number) => {
    const next = tags.filter((_, i) => i !== idx);
    void onSaveCaption(next.join(", "));
    setSelected(new Set());
  };
  const addTags = (incoming: string[]) => {
    const lower = new Set(tags.map((t) => t.toLowerCase()));
    const adds = incoming.map((t) => t.trim()).filter((t) => t && !lower.has(t.toLowerCase()));
    if (adds.length === 0) return;
    void onSaveCaption([...tags, ...adds].join(", "));
  };
  const addTag = () => {
    const t = newTag.trim().replace(/,/g, "");
    if (!t) { setAdding(false); return; }
    addTags([t]);
    setNewTag("");
  };

  // ── クリップボード ──
  const copyText = async (text: string) => {
    try { await navigator.clipboard.writeText(text); return; } catch { /* fallback */ }
    const ta = document.createElement("textarea");
    ta.value = text; ta.style.position = "fixed"; ta.style.opacity = "0";
    document.body.appendChild(ta); ta.select();
    try { document.execCommand("copy"); } catch { /* noop */ }
    document.body.removeChild(ta);
  };
  const readText = async (): Promise<string> => {
    try { return await navigator.clipboard.readText(); } catch { return ""; }
  };

  // 選択中（なければ全件）のタグを順序維持で取得
  const targetIndices = () =>
    selected.size > 0 ? [...selected].sort((a, b) => a - b) : tags.map((_, i) => i);
  const copySelected = () => void copyText(targetIndices().map((i) => tags[i]).join(", "));
  const cutSelected = async () => {
    const idxs = targetIndices();
    await copyText(idxs.map((i) => tags[i]).join(", "));
    const drop = new Set(idxs);
    void onSaveCaption(tags.filter((_, i) => !drop.has(i)).join(", "));
    setSelected(new Set());
  };
  const deleteSelected = () => {
    if (selected.size === 0) return;
    void onSaveCaption(tags.filter((_, i) => !selected.has(i)).join(", "));
    setSelected(new Set());
  };
  const pasteTags = async () => {
    const txt = await readText();
    if (txt) addTags(txt.split(/[,\n]/));
  };
  const selectAll = () => { setSelected(new Set(tags.map((_, i) => i))); };

  const onChipClick = (e: React.MouseEvent, idx: number) => {
    e.stopPropagation();
    if (e.shiftKey && lastIdx != null) {
      const [a, b] = [Math.min(lastIdx, idx), Math.max(lastIdx, idx)];
      const next = new Set(selected);
      for (let i = a; i <= b; i++) next.add(i);
      setSelected(next);
    } else if (e.ctrlKey || e.metaKey) {
      const next = new Set(selected);
      if (next.has(idx)) next.delete(idx); else next.add(idx);
      setSelected(next); setLastIdx(idx);
    } else {
      setSelected(new Set([idx])); setLastIdx(idx);
    }
  };

  const onKeyDown = (e: React.KeyboardEvent) => {
    if ((e.target as HTMLElement).tagName === "INPUT") return; // 追加入力中は除外
    const ctrl = e.ctrlKey || e.metaKey;
    if (ctrl && e.key.toLowerCase() === "c") { e.preventDefault(); copySelected(); }
    else if (ctrl && e.key.toLowerCase() === "x") { e.preventDefault(); void cutSelected(); }
    else if (ctrl && e.key.toLowerCase() === "v") { e.preventDefault(); void pasteTags(); }
    else if (ctrl && e.key.toLowerCase() === "a") { e.preventDefault(); selectAll(); }
    else if (e.key === "Delete" || e.key === "Backspace") { e.preventDefault(); deleteSelected(); }
    else if (e.key === "Escape") { setSelected(new Set()); }
  };

  return (
    <div
      className="flex flex-wrap items-center gap-1 outline-none rounded"
      tabIndex={0}
      onKeyDown={onKeyDown}
    >
      {tags.length === 0 && !adding && (
        <button onClick={onStartEdit} className="text-xs text-gray-600 italic hover:text-gray-400">
          タグなし - クリックして追加
        </button>
      )}
      {/* 選択時の操作ツールバー */}
      {selected.size > 0 && (
        <span className="inline-flex items-center gap-1 mr-1 pr-1 border-r border-gray-600/60">
          <span className="text-[10px] text-indigo-300 font-semibold">{selected.size}選択</span>
          <button onClick={copySelected} className="p-0.5 text-gray-400 hover:text-indigo-300" title="コピー (Ctrl+C)"><Copy size={11} /></button>
          <button onClick={() => void cutSelected()} className="p-0.5 text-gray-400 hover:text-amber-300" title="カット (Ctrl+X)"><Scissors size={11} /></button>
          <button onClick={deleteSelected} className="p-0.5 text-gray-400 hover:text-red-300" title="削除 (Delete)"><X size={11} /></button>
        </span>
      )}
      {tags.map((tag, idx) => {
        const isPrefix = prefixSet.has(tag.toLowerCase());
        const isSel = selected.has(idx);
        return (
          <span
            key={`${tag}-${idx}`}
            onClick={(e) => onChipClick(e, idx)}
            className={[
              "group/tag inline-flex items-center gap-1 pl-2 pr-1 py-0.5 rounded text-[11px] border cursor-pointer select-none transition-colors",
              isSel
                ? "bg-indigo-600 text-white border-indigo-400 ring-1 ring-indigo-300"
                : isPrefix
                  ? "bg-indigo-900/50 text-indigo-200 border-indigo-700/60"
                  : "bg-gray-700/70 text-gray-300 border-gray-600/50 hover:border-gray-500",
            ].join(" ")}
            title={isPrefix ? "固定タグ（再生成しても保持） / クリックで選択" : "クリックで選択（Shift/Ctrl で複数）"}
          >
            {isPrefix && <span className="text-[8px] text-indigo-300">●</span>}
            <span className="max-w-[140px] truncate">{tag}</span>
            <button
              onClick={(e) => { e.stopPropagation(); removeTag(idx); }}
              className="opacity-50 group-hover/tag:opacity-100 hover:text-red-300 transition-opacity"
              title="このタグを削除"
            >
              <X size={10} />
            </button>
          </span>
        );
      })}
      {adding ? (
        <input
          autoFocus
          value={newTag}
          onChange={(e) => setNewTag(e.target.value)}
          onBlur={addTag}
          onKeyDown={(e) => {
            if (e.key === "Enter") { e.preventDefault(); addTag(); }
            else if (e.key === "Escape") { setNewTag(""); setAdding(false); }
          }}
          placeholder="タグ名 + Enter"
          className="bg-gray-700 border border-indigo-600 text-gray-200 rounded px-1.5 py-0.5 text-[11px] w-28 focus:outline-none"
        />
      ) : (
        <button
          onClick={() => setAdding(true)}
          className="inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded text-[11px] text-gray-500 hover:text-indigo-300 hover:bg-gray-700/50 border border-dashed border-gray-600/50 transition-colors"
          title="タグを追加"
        >
          <Plus size={10} />追加
        </button>
      )}
      {tags.length > 0 && (
        <button
          onClick={copySelected}
          className="ml-1 opacity-0 group-hover:opacity-100 text-gray-500 hover:text-indigo-300 transition-all"
          title={selected.size > 0 ? "選択タグをコピー (Ctrl+C)" : "全タグをコピー (Ctrl+C)"}
        >
          <Copy size={11} />
        </button>
      )}
      <button
        onClick={() => void pasteTags()}
        className="opacity-0 group-hover:opacity-100 text-gray-500 hover:text-emerald-300 transition-all"
        title="クリップボードから貼り付け (Ctrl+V)"
      >
        <ClipboardPaste size={11} />
      </button>
      <button
        onClick={onStartEdit}
        className="opacity-0 group-hover:opacity-100 text-gray-500 hover:text-gray-300 transition-all"
        title="テキストで一括編集"
      >
        <Edit3 size={11} />
      </button>
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
  onDelete,
  onSaveCaption,
  prefixTags = [],
  isDragOver,
  onDragStart,
  onDragOver,
  onDrop,
  onDragEnd,
  onMoveUp,
  onMoveDown,
  isFirst,
  isLast,
}: {
  item: CaptionItem;
  isEditing: boolean;
  editingText: string;
  onStartEdit: () => void;
  onChangeText: (t: string) => void;
  onSave: () => Promise<void>;
  onCancel: () => void;
  onDelete: () => void;
  onSaveCaption: (caption: string) => Promise<void>;
  prefixTags?: string[];
  isDragOver?: boolean;
  onDragStart?: () => void;
  onDragOver?: (e: React.DragEvent) => void;
  onDrop?: () => void;
  onDragEnd?: () => void;
  onMoveUp?: () => void;
  onMoveDown?: () => void;
  isFirst?: boolean;
  isLast?: boolean;
}) {
  const filename = item.file_path.split(/[\\/]/).pop() ?? item.file_path;
  const ext = filename.split(".").pop()?.toLowerCase() ?? "";
  const sourceColor: Record<string, string> = {
    wd14: "text-violet-400",
    manual: "text-indigo-400",
    file: "text-emerald-400",
    "": "text-gray-600",
  };

  return (
    <div
      className={[
        "border rounded-xl p-3 flex gap-2 group transition-colors",
        isDragOver ? "border-indigo-500 bg-indigo-950/20 border-dashed" : "border-gray-700 bg-gray-800",
      ].join(" ")}
      draggable
      onDragStart={onDragStart}
      onDragOver={onDragOver}
      onDrop={onDrop}
      onDragEnd={onDragEnd}
    >
      {/* ドラッグハンドル + 上下ボタン */}
      <div className="self-center flex flex-col items-center gap-0.5 flex-shrink-0">
        <button
          onClick={onMoveUp}
          disabled={isFirst}
          className="text-gray-600 hover:text-gray-300 disabled:opacity-20 disabled:cursor-not-allowed p-0.5 rounded transition-colors"
          title="上に移動"
        >
          <ArrowUp size={11} />
        </button>
        <div className="text-gray-600 hover:text-gray-400 cursor-grab active:cursor-grabbing">
          <GripVertical size={13} />
        </div>
        <button
          onClick={onMoveDown}
          disabled={isLast}
          className="text-gray-600 hover:text-gray-300 disabled:opacity-20 disabled:cursor-not-allowed p-0.5 rounded transition-colors"
          title="下に移動"
        >
          <ArrowDown size={11} />
        </button>
      </div>

      {/* サムネイル */}
      <div className="w-16 h-16 bg-gray-700 rounded-lg overflow-hidden shrink-0 flex items-center justify-center relative">
        <img
          src={`${API_BASE}/collector/thumbnail?path=${encodeURIComponent(item.file_path)}`}
          alt={filename}
          draggable={false}
          className="w-full h-full object-cover"
          onError={(e) => {
            const img = e.target as HTMLImageElement;
            img.style.display = "none";
            const fb = img.nextElementSibling as HTMLElement | null;
            if (fb) fb.style.display = "flex";
          }}
        />
        <div className="absolute inset-0 items-center justify-center flex-col gap-0.5 hidden">
          <ImageOff size={16} className="text-gray-500" />
          <span className="text-xs text-gray-600 uppercase">{ext}</span>
        </div>
      </div>

      {/* 情報 + キャプション */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-1">
          <span className="text-xs text-gray-400 truncate flex-1">{filename}</span>
          <span className="text-xs text-gray-600 flex-shrink-0">{item.width}×{item.height}</span>
          {item.caption_source && (
            <span className={`text-xs flex-shrink-0 ${sourceColor[item.caption_source] ?? "text-gray-600"}`}>
              [{item.caption_source || "未生成"}]
            </span>
          )}
          {!item.caption && (
            <span className="text-xs text-amber-500 flex-shrink-0">⚠ 未キャプション</span>
          )}
          <button
            onClick={(e) => { e.stopPropagation(); onDelete(); }}
            title="データセットから削除（ファイルは残る）"
            className="flex-shrink-0 opacity-40 hover:opacity-100 transition-opacity text-gray-500 hover:text-red-400 p-0.5 rounded"
          >
            <Trash2 size={13} />
          </button>
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
          <CaptionTags
            caption={item.caption}
            prefixTags={prefixTags}
            onSaveCaption={onSaveCaption}
            onStartEdit={onStartEdit}
          />
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
  done,
}: {
  step: number;
  label: string;
  icon: React.ReactNode;
  color: string;
  onClick: () => void;
  done?: boolean;
}) {
  return (
    <button
      onClick={done ? undefined : onClick}
      disabled={done}
      className={[
        "flex items-center gap-2 rounded-lg px-3 py-1.5 text-xs font-medium transition-colors flex-1",
        done
          ? "bg-emerald-900/60 border border-emerald-700/50 text-emerald-400 cursor-not-allowed"
          : `${color} text-white`,
      ].join(" ")}
    >
      <span className={["w-4 h-4 rounded-full flex items-center justify-center text-[9px] font-bold flex-shrink-0", done ? "bg-emerald-700 text-white" : "bg-white/20"].join(" ")}>
        {done ? "✓" : step}
      </span>
      {icon}
      <span className="truncate">{done ? `${label} 済` : label}</span>
    </button>
  );
}
