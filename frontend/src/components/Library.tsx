import { useState, useEffect, useCallback } from "react";
import {
  Archive,
  Plus,
  Trash2,
  Edit3,
  Search,
  X,
  Check,
  Tag,
  FileBox,
  Star,
  RefreshCw,
  ChevronDown,
  ChevronUp,
  Info,
  Layers,
} from "lucide-react";
import { apiGet, apiPost, apiDelete } from "../lib/api";
import type { Project, LoraAsset, LoraAssetPayload } from "../types";

type Props = {
  projects: Project[];
  selectedProjectId: number | null;
  showError: (msg: string) => void;
  showNotice: (msg: string) => void;
};

const ASSET_TYPE_COLORS: Record<string, string> = {
  character: "bg-violet-900 text-violet-300 border-violet-700",
  style: "bg-indigo-900 text-indigo-300 border-indigo-700",
  hybrid: "bg-emerald-900 text-emerald-300 border-emerald-700",
  custom: "bg-gray-700 text-gray-300 border-gray-600",
};

const ASSET_TYPE_LABELS: Record<string, string> = {
  character: "Character",
  style: "Style",
  hybrid: "Hybrid",
  custom: "Custom",
};

// ── Quality Score バッジ ──────────────────────────────────────────────────

function QualityBadge({ score }: { score: number | null }) {
  if (score === null) return <span className="text-xs text-gray-600">—</span>;
  const color =
    score >= 80
      ? "text-emerald-400 bg-emerald-950 border-emerald-800"
      : score >= 60
        ? "text-amber-400 bg-amber-950 border-amber-800"
        : "text-red-400 bg-red-950 border-red-800";
  return (
    <span className={`text-xs font-bold px-1.5 py-0.5 rounded border ${color}`}>
      {score}
    </span>
  );
}

// ── アセット編集モーダル ───────────────────────────────────────────────────

function AssetModal({
  asset,
  projects,
  onClose,
  onSave,
}: {
  asset: LoraAsset | null;
  projects: Project[];
  onClose: () => void;
  onSave: (payload: LoraAssetPayload) => Promise<void>;
}) {
  const [name, setName] = useState(asset?.name ?? "");
  const [assetType, setAssetType] = useState(asset?.asset_type ?? "character");
  const [loraPath, setLoraPath] = useState(asset?.lora_path ?? "");
  const [baseModel, setBaseModel] = useState(asset?.base_model ?? "");
  const [profileName, setProfileName] = useState(asset?.profile_name ?? "");
  const [qualityScore, setQualityScore] = useState(
    asset?.quality_score != null ? String(asset.quality_score) : "",
  );
  const [datasetSize, setDatasetSize] = useState(
    asset?.dataset_size != null ? String(asset.dataset_size) : "0",
  );
  const [notes, setNotes] = useState(asset?.notes ?? "");
  const [tagInput, setTagInput] = useState("");
  const [tags, setTags] = useState<string[]>(asset?.tags ?? []);
  const [projectId, setProjectId] = useState<string>(
    asset?.project_id != null ? String(asset.project_id) : "",
  );
  const [saving, setSaving] = useState(false);

  function addTag() {
    const t = tagInput.trim();
    if (t && !tags.includes(t)) setTags((prev) => [...prev, t]);
    setTagInput("");
  }

  async function handleSave() {
    if (!name.trim()) return;
    setSaving(true);
    try {
      await onSave({
        project_id: projectId ? Number(projectId) : null,
        name: name.trim(),
        lora_path: loraPath,
        base_model: baseModel,
        dataset_size: Number(datasetSize) || 0,
        profile_name: profileName,
        tags,
        notes,
        quality_score: qualityScore ? Number(qualityScore) : null,
        asset_type: assetType,
      });
      onClose();
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
      <div className="bg-gray-900 border border-gray-700 rounded-2xl w-full max-w-lg shadow-2xl overflow-y-auto max-h-[90vh]">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-gray-800">
          <div className="flex items-center gap-2">
            <Archive size={16} className="text-indigo-400" />
            <span className="text-sm font-bold text-gray-100">
              {asset ? "LoRA 資産を編集" : "LoRA 資産を追加"}
            </span>
          </div>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-300 p-1"
          >
            <X size={16} />
          </button>
        </div>

        {/* Body */}
        <div className="p-5 space-y-4">
          {/* 名前 */}
          <label className="block">
            <span className="text-xs text-gray-400 mb-1 block">LoRA 名称 *</span>
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="例: my_character_v1"
              className="w-full bg-gray-800 border border-gray-700 text-gray-100 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-indigo-500"
            />
          </label>

          {/* タイプ + プロジェクト */}
          <div className="grid grid-cols-2 gap-3">
            <label className="block">
              <span className="text-xs text-gray-400 mb-1 block">タイプ</span>
              <select
                value={assetType}
                onChange={(e) => setAssetType(e.target.value)}
                className="w-full bg-gray-800 border border-gray-700 text-gray-100 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-indigo-500"
              >
                {Object.entries(ASSET_TYPE_LABELS).map(([v, l]) => (
                  <option key={v} value={v}>{l}</option>
                ))}
              </select>
            </label>
            <label className="block">
              <span className="text-xs text-gray-400 mb-1 block">プロジェクト</span>
              <select
                value={projectId}
                onChange={(e) => setProjectId(e.target.value)}
                className="w-full bg-gray-800 border border-gray-700 text-gray-100 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-indigo-500"
              >
                <option value="">未関連</option>
                {projects.map((p) => (
                  <option key={p.id} value={p.id}>{p.name}</option>
                ))}
              </select>
            </label>
          </div>

          {/* LoRA パス */}
          <label className="block">
            <span className="text-xs text-gray-400 mb-1 block">LoRA ファイルパス</span>
            <input
              value={loraPath}
              onChange={(e) => setLoraPath(e.target.value)}
              placeholder="例: C:\models\loras\my_character.safetensors"
              className="w-full bg-gray-800 border border-gray-700 text-gray-100 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-indigo-500 font-mono"
            />
          </label>

          {/* ベースモデル + プロファイル */}
          <div className="grid grid-cols-2 gap-3">
            <label className="block">
              <span className="text-xs text-gray-400 mb-1 block">ベースモデル</span>
              <input
                value={baseModel}
                onChange={(e) => setBaseModel(e.target.value)}
                placeholder="例: animagine-xl-3.0"
                className="w-full bg-gray-800 border border-gray-700 text-gray-100 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-indigo-500"
              />
            </label>
            <label className="block">
              <span className="text-xs text-gray-400 mb-1 block">学習プロファイル</span>
              <input
                value={profileName}
                onChange={(e) => setProfileName(e.target.value)}
                placeholder="例: Character Standard"
                className="w-full bg-gray-800 border border-gray-700 text-gray-100 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-indigo-500"
              />
            </label>
          </div>

          {/* 品質スコア + データセット枚数 */}
          <div className="grid grid-cols-2 gap-3">
            <label className="block">
              <span className="text-xs text-gray-400 mb-1 block">品質スコア (0–100)</span>
              <input
                type="number"
                min="0"
                max="100"
                value={qualityScore}
                onChange={(e) => setQualityScore(e.target.value)}
                placeholder="—"
                className="w-full bg-gray-800 border border-gray-700 text-gray-100 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-indigo-500"
              />
            </label>
            <label className="block">
              <span className="text-xs text-gray-400 mb-1 block">データセット枚数</span>
              <input
                type="number"
                min="0"
                value={datasetSize}
                onChange={(e) => setDatasetSize(e.target.value)}
                className="w-full bg-gray-800 border border-gray-700 text-gray-100 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-indigo-500"
              />
            </label>
          </div>

          {/* タグ */}
          <div>
            <span className="text-xs text-gray-400 mb-1 block">タグ</span>
            <div className="flex gap-2 mb-2">
              <input
                value={tagInput}
                onChange={(e) => setTagInput(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); addTag(); }}}
                placeholder="タグを入力して Enter"
                className="flex-1 bg-gray-800 border border-gray-700 text-gray-100 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:border-indigo-500"
              />
              <button
                onClick={addTag}
                className="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 text-gray-300 rounded-lg text-sm"
              >
                追加
              </button>
            </div>
            <div className="flex flex-wrap gap-1.5 min-h-[28px]">
              {tags.map((t) => (
                <span
                  key={t}
                  className="flex items-center gap-1 text-xs bg-indigo-950 text-indigo-300 border border-indigo-800 px-2 py-0.5 rounded-full"
                >
                  {t}
                  <button
                    onClick={() => setTags((prev) => prev.filter((x) => x !== t))}
                    className="hover:text-red-400"
                  >
                    <X size={10} />
                  </button>
                </span>
              ))}
              {tags.length === 0 && (
                <span className="text-xs text-gray-600">タグなし</span>
              )}
            </div>
          </div>

          {/* メモ */}
          <label className="block">
            <span className="text-xs text-gray-400 mb-1 block">メモ</span>
            <textarea
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              rows={3}
              placeholder="自由記述"
              className="w-full bg-gray-800 border border-gray-700 text-gray-100 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-indigo-500 resize-none"
            />
          </label>
        </div>

        {/* Footer */}
        <div className="flex justify-end gap-2 px-5 py-4 border-t border-gray-800">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-gray-800 hover:bg-gray-700 text-gray-300 rounded-lg text-sm"
          >
            キャンセル
          </button>
          <button
            onClick={() => void handleSave()}
            disabled={!name.trim() || saving}
            className="flex items-center gap-2 px-4 py-2 bg-indigo-700 hover:bg-indigo-600 disabled:bg-gray-700 disabled:text-gray-500 text-white rounded-lg text-sm font-medium"
          >
            <Check size={14} />
            {saving ? "保存中..." : "保存"}
          </button>
        </div>
      </div>
    </div>
  );
}

// ── アセットカード ────────────────────────────────────────────────────────

function AssetCard({
  asset,
  projectName,
  onEdit,
  onDelete,
}: {
  asset: LoraAsset;
  projectName: string | null;
  onEdit: () => void;
  onDelete: () => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const typeStyle = ASSET_TYPE_COLORS[asset.asset_type] ?? ASSET_TYPE_COLORS.custom;
  const typeLabel = ASSET_TYPE_LABELS[asset.asset_type] ?? asset.asset_type;

  return (
    <div className="bg-gray-800 border border-gray-700 rounded-xl overflow-hidden hover:border-gray-600 transition-colors">
      {/* Card Header */}
      <div className="p-4">
        <div className="flex items-start justify-between gap-2 mb-2">
          <div className="flex items-center gap-2 min-w-0">
            <span className={`text-xs font-bold px-2 py-0.5 rounded border shrink-0 ${typeStyle}`}>
              {typeLabel}
            </span>
            <span className="text-sm font-semibold text-gray-100 truncate">{asset.name}</span>
          </div>
          <div className="flex items-center gap-1 shrink-0">
            <QualityBadge score={asset.quality_score} />
            <button
              onClick={onEdit}
              className="p-1.5 text-gray-500 hover:text-gray-300 hover:bg-gray-700 rounded transition-colors"
            >
              <Edit3 size={13} />
            </button>
            <button
              onClick={onDelete}
              className="p-1.5 text-gray-600 hover:text-red-400 hover:bg-red-950 rounded transition-colors"
            >
              <Trash2 size={13} />
            </button>
          </div>
        </div>

        {/* Meta chips */}
        <div className="flex flex-wrap gap-1.5 text-xs">
          {asset.base_model && (
            <span className="flex items-center gap-1 bg-gray-700 text-gray-400 px-2 py-0.5 rounded-full">
              <Layers size={10} />
              {asset.base_model}
            </span>
          )}
          {asset.profile_name && (
            <span className="flex items-center gap-1 bg-gray-700 text-gray-400 px-2 py-0.5 rounded-full">
              <Star size={10} />
              {asset.profile_name}
            </span>
          )}
          {asset.dataset_size > 0 && (
            <span className="flex items-center gap-1 bg-gray-700 text-gray-400 px-2 py-0.5 rounded-full">
              <FileBox size={10} />
              {asset.dataset_size}枚
            </span>
          )}
          {projectName && (
            <span className="flex items-center gap-1 bg-indigo-950 text-indigo-400 px-2 py-0.5 rounded-full">
              {projectName}
            </span>
          )}
        </div>

        {/* Tags */}
        {asset.tags.length > 0 && (
          <div className="flex flex-wrap gap-1 mt-2">
            {asset.tags.map((t) => (
              <span key={t} className="text-xs bg-gray-700 text-gray-500 px-1.5 py-0.5 rounded">
                #{t}
              </span>
            ))}
          </div>
        )}
      </div>

      {/* LoRA path (collapsible) */}
      {(asset.lora_path || asset.notes) && (
        <div className="border-t border-gray-700">
          <button
            onClick={() => setExpanded((v) => !v)}
            className="w-full flex items-center gap-2 px-4 py-2 text-xs text-gray-500 hover:text-gray-400 hover:bg-gray-700/50 transition-colors text-left"
          >
            {expanded ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
            <Info size={11} />
            詳細
          </button>
          {expanded && (
            <div className="px-4 pb-3 space-y-2">
              {asset.lora_path && (
                <div className="text-xs text-gray-500 font-mono bg-gray-900 rounded px-2 py-1.5 break-all">
                  {asset.lora_path}
                </div>
              )}
              {asset.notes && (
                <p className="text-xs text-gray-400 whitespace-pre-wrap">{asset.notes}</p>
              )}
              <div className="text-xs text-gray-600">
                {new Date(asset.created_at).toLocaleString("ja-JP")}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── メインコンポーネント ───────────────────────────────────────────────────

export default function Library({ projects, selectedProjectId, showError, showNotice }: Props) {
  const [assets, setAssets] = useState<LoraAsset[]>([]);
  const [loading, setLoading] = useState(false);
  const [filterType, setFilterType] = useState<string>("all");
  const [filterProject, setFilterProject] = useState<string>("all");
  const [searchText, setSearchText] = useState("");
  const [modalOpen, setModalOpen] = useState(false);
  const [editingAsset, setEditingAsset] = useState<LoraAsset | null>(null);
  const [stats, setStats] = useState<{ total: number; by_type: Record<string, number>; avg_quality_score: number | null } | null>(null);

  const loadAssets = useCallback(async () => {
    setLoading(true);
    try {
      const [assetList, libStats] = await Promise.all([
        apiGet<LoraAsset[]>("/library/assets"),
        apiGet<{ total: number; by_type: Record<string, number>; avg_quality_score: number | null }>("/library/stats"),
      ]);
      setAssets(assetList);
      setStats(libStats);
    } catch (e) {
      showError(String(e));
    } finally {
      setLoading(false);
    }
  }, [showError]);

  useEffect(() => {
    void loadAssets();
  }, [loadAssets]);

  // フィルタ済みリスト
  const displayAssets = assets.filter((a) => {
    if (filterType !== "all" && a.asset_type !== filterType) return false;
    if (filterProject !== "all") {
      const pid = filterProject === "none" ? null : Number(filterProject);
      if (a.project_id !== pid) return false;
    }
    if (searchText) {
      const q = searchText.toLowerCase();
      return (
        a.name.toLowerCase().includes(q) ||
        a.base_model.toLowerCase().includes(q) ||
        a.tags.some((t) => t.toLowerCase().includes(q)) ||
        a.notes.toLowerCase().includes(q)
      );
    }
    return true;
  });

  const projectMap = Object.fromEntries(projects.map((p) => [p.id, p.name]));

  async function handleSave(payload: LoraAssetPayload) {
    if (editingAsset) {
      await apiPost<LoraAsset>(`/library/assets/${editingAsset.id}`, payload, "PUT");
      showNotice("LoRA 資産を更新しました");
    } else {
      await apiPost<LoraAsset>("/library/assets", payload);
      showNotice("LoRA 資産を追加しました");
    }
    await loadAssets();
  }

  async function handleDelete(asset: LoraAsset) {
    if (!confirm(`「${asset.name}」を削除しますか？（ファイルは削除されません）`)) return;
    try {
      await apiDelete(`/library/assets/${asset.id}`);
      showNotice(`「${asset.name}」を削除しました`);
      await loadAssets();
    } catch (e) {
      showError(String(e));
    }
  }

  function openCreate() {
    setEditingAsset(null);
    setModalOpen(true);
  }

  function openEdit(asset: LoraAsset) {
    setEditingAsset(asset);
    setModalOpen(true);
  }

  return (
    <div className="space-y-5">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-gray-100 flex items-center gap-2">
            <Archive size={20} className="text-indigo-400" />
            LoRA ライブラリ
          </h2>
          <p className="text-sm text-gray-400 mt-0.5">§22-23 学習済み LoRA 資産の管理</p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => void loadAssets()}
            className="p-2 text-gray-500 hover:text-gray-300 hover:bg-gray-800 rounded-lg transition-colors"
            title="更新"
          >
            <RefreshCw size={15} className={loading ? "animate-spin" : ""} />
          </button>
          <button
            onClick={openCreate}
            className="flex items-center gap-2 bg-indigo-700 hover:bg-indigo-600 text-white rounded-lg px-4 py-2 text-sm font-medium transition-colors"
          >
            <Plus size={15} />
            資産を追加
          </button>
        </div>
      </div>

      {/* Stats bar */}
      {stats && (
        <div className="grid grid-cols-4 gap-3">
          {[
            { label: "総資産数", value: stats.total, color: "text-gray-200" },
            { label: "Character", value: stats.by_type.character ?? 0, color: "text-violet-400" },
            { label: "Style", value: stats.by_type.style ?? 0, color: "text-indigo-400" },
            { label: "平均スコア", value: stats.avg_quality_score != null ? `${stats.avg_quality_score}` : "—", color: "text-emerald-400" },
          ].map(({ label, value, color }) => (
            <div key={label} className="bg-gray-800 border border-gray-700 rounded-xl p-3 text-center">
              <div className={`text-2xl font-bold ${color}`}>{value}</div>
              <div className="text-xs text-gray-500 mt-0.5">{label}</div>
            </div>
          ))}
        </div>
      )}

      {/* Filters */}
      <div className="flex flex-wrap gap-2 items-center">
        {/* 検索 */}
        <div className="relative flex-1 min-w-48">
          <Search size={13} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" />
          <input
            value={searchText}
            onChange={(e) => setSearchText(e.target.value)}
            placeholder="名前・タグ・メモで検索..."
            className="w-full bg-gray-800 border border-gray-700 text-gray-100 rounded-lg pl-8 pr-3 py-2 text-sm focus:outline-none focus:border-indigo-500"
          />
          {searchText && (
            <button
              onClick={() => setSearchText("")}
              className="absolute right-2 top-1/2 -translate-y-1/2 text-gray-500 hover:text-gray-300"
            >
              <X size={13} />
            </button>
          )}
        </div>

        {/* タイプフィルタ */}
        <div className="flex gap-1">
          {["all", "character", "style", "hybrid", "custom"].map((t) => (
            <button
              key={t}
              onClick={() => setFilterType(t)}
              className={[
                "text-xs px-3 py-1.5 rounded-lg border transition-colors",
                filterType === t
                  ? "bg-indigo-900 border-indigo-700 text-indigo-300"
                  : "bg-gray-800 border-gray-700 text-gray-500 hover:text-gray-400",
              ].join(" ")}
            >
              {t === "all"
                ? "全て"
                : ASSET_TYPE_LABELS[t] ?? t}
            </button>
          ))}
        </div>

        {/* プロジェクトフィルタ */}
        <select
          value={filterProject}
          onChange={(e) => setFilterProject(e.target.value)}
          className="bg-gray-800 border border-gray-700 text-gray-400 rounded-lg px-3 py-1.5 text-xs focus:outline-none focus:border-indigo-500"
        >
          <option value="all">全プロジェクト</option>
          <option value="none">未関連</option>
          {projects.map((p) => (
            <option key={p.id} value={p.id}>{p.name}</option>
          ))}
        </select>

        <span className="text-xs text-gray-600">{displayAssets.length}件</span>
      </div>

      {/* Asset Grid */}
      {loading ? (
        <div className="text-center py-16 text-gray-500 text-sm">読み込み中...</div>
      ) : displayAssets.length === 0 ? (
        <div className="bg-gray-800 border border-gray-700 rounded-xl p-12 text-center">
          <Archive size={40} className="text-gray-600 mx-auto mb-3" />
          <p className="text-gray-400 text-sm mb-1">
            {assets.length === 0 ? "LoRA 資産がまだありません" : "条件に一致する資産がありません"}
          </p>
          {assets.length === 0 && (
            <p className="text-gray-600 text-xs">
              「資産を追加」ボタンで学習済み LoRA を登録してください
            </p>
          )}
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {displayAssets.map((asset) => (
            <AssetCard
              key={asset.id}
              asset={asset}
              projectName={asset.project_id != null ? (projectMap[asset.project_id] ?? null) : null}
              onEdit={() => openEdit(asset)}
              onDelete={() => void handleDelete(asset)}
            />
          ))}
        </div>
      )}

      {/* Modal */}
      {modalOpen && (
        <AssetModal
          asset={editingAsset}
          projects={projects}
          onClose={() => setModalOpen(false)}
          onSave={handleSave}
        />
      )}
    </div>
  );
}
