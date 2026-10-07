import { useRef, useState } from "react";
import {
  Plus,
  Copy,
  Trash2,
  User,
  Palette,
  Zap,
  CheckCircle2,
  AlertCircle,
  TrendingUp,
  ArrowRight,
  Download,
  Upload,
  FolderOpen,
  ImagePlus,
  Wand2,
  ChevronDown,
} from "lucide-react";
import { API_BASE, apiPost, apiDelete } from "../lib/api";
import type { Project, TrainingStatus, IntegrationStatus } from "../types";

type Props = {
  projects: Project[];
  selectedProjectId: number | null;
  selectedProject: Project | null;
  statuses: Record<number, TrainingStatus>;
  integrationStatus: IntegrationStatus | null;
  onSelectProject: (id: number | null) => void;
  onRefresh: () => Promise<void>;
  showError: (msg: string) => void;
  showNotice: (msg: string) => void;
  clearMessages: () => void;
  onTabChange: (tab: string) => void;
};

function StatusDot({ status }: { status: string }) {
  const cls =
    status === "training" ? "bg-amber-400 animate-pulse"
    : status === "completed" ? "bg-emerald-400"
    : status === "paused" ? "bg-blue-400"
    : "bg-gray-600";
  return <span className={`inline-block w-2 h-2 rounded-full flex-shrink-0 ${cls}`} />;
}

function StatusBadge({ status }: { status: string }) {
  const map: Record<string, { label: string; cls: string }> = {
    idle:      { label: "待機中", cls: "bg-gray-700 text-gray-400" },
    training:  { label: "学習中", cls: "bg-amber-900 text-amber-300" },
    paused:    { label: "一時停止", cls: "bg-blue-900 text-blue-300" },
    completed: { label: "完了", cls: "bg-emerald-900 text-emerald-300" },
  };
  const { label, cls } = map[status] ?? { label: status, cls: "bg-gray-700 text-gray-400" };
  return <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${cls}`}>{label}</span>;
}

export default function Home({
  projects,
  selectedProjectId,
  selectedProject,
  statuses,
  integrationStatus,
  onSelectProject,
  onRefresh,
  showError,
  showNotice,
  onTabChange,
}: Props) {
  const [newName, setNewName] = useState("");
  const [newType, setNewType] = useState<"character" | "style">("character");
  const [creating, setCreating] = useState(false);
  const [deleting, setDeleting] = useState<number | null>(null);
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [importing, setImporting] = useState(false);
  const importRef = useRef<HTMLInputElement>(null);

  const integrationOk = integrationStatus
    ? Object.values(integrationStatus.checks).filter((x) => x.ok).length : 0;
  const integrationTotal = integrationStatus
    ? Object.keys(integrationStatus.checks).length : 6;

  async function handleCreate() {
    if (!newName.trim()) return;
    setCreating(true);
    try {
      const p = await apiPost<Project>("/projects", { name: newName.trim(), project_type: newType });
      setNewName("");
      setShowCreateForm(false);
      onSelectProject(p.id);
      showNotice(`プロジェクト「${p.name}」を作成しました`);
      await onRefresh();
    } catch (e) {
      showError(`作成失敗: ${String(e)}`);
    } finally {
      setCreating(false);
    }
  }

  async function handleDuplicate(projectId: number, srcName: string) {
    try {
      const p = await apiPost<Project>(`/projects/${projectId}/duplicate`, { name: `${srcName}_copy` });
      await onRefresh();
      onSelectProject(p.id);
      showNotice(`「${p.name}」を複製しました`);
    } catch (e) {
      showError(`複製失敗: ${String(e)}`);
    }
  }

  async function handleDelete(projectId: number) {
    if (deleting === projectId) {
      try {
        await apiDelete(`/projects/${projectId}`);
        if (selectedProjectId === projectId) onSelectProject(null);
        await onRefresh();
        showNotice("プロジェクトを削除しました");
      } catch (e) {
        showError(`削除失敗: ${String(e)}`);
      } finally {
        setDeleting(null);
      }
    } else {
      setDeleting(projectId);
      setTimeout(() => setDeleting((d) => (d === projectId ? null : d)), 3000);
    }
  }

  async function handleExport() {
    if (!selectedProjectId) return;
    try {
      const res = await fetch(`${API_BASE}/project-files/${selectedProjectId}/export`);
      if (!res.ok) throw new Error(await res.text());
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${selectedProject?.name ?? "project"}.lbp`;
      a.click();
      URL.revokeObjectURL(url);
      showNotice(".lbp ファイルをエクスポートしました");
    } catch (e) {
      showError(`エクスポート失敗: ${String(e)}`);
    }
  }

  async function handleImport(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setImporting(true);
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch(`${API_BASE}/project-files/import`, { method: "POST", body: form });
      if (!res.ok) throw new Error(await res.text());
      const { project_id, project_name } = await res.json() as { project_id: number; project_name: string };
      await onRefresh();
      onSelectProject(project_id);
      showNotice(`「${project_name}」をインポートしました`);
    } catch (e) {
      showError(`インポート失敗: ${String(e)}`);
    } finally {
      setImporting(false);
      if (importRef.current) importRef.current.value = "";
    }
  }

  // 次のステップロジック
  const nextSteps: { label: string; desc: string; tab: string; icon: React.ReactNode }[] = [];
  if (selectedProject) {
    nextSteps.push({ label: "データセットを準備する", desc: "画像収集・キャプション付けを進めましょう", tab: "dataset", icon: <ImagePlus size={15} /> });
    nextSteps.push({ label: "学習を開始する", desc: "Anima学習設定を確認してRunを作成", tab: "training", icon: <Wand2 size={15} /> });
  }
  if (integrationOk < integrationTotal) {
    nextSteps.push({ label: "外部ツールを設定する", desc: "Anima Backend / WD14 などの接続を確認", tab: "integrations", icon: <AlertCircle size={15} /> });
  }

  return (
    <div className="space-y-6 max-w-5xl">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h2 className="text-xl font-bold text-gray-100 mb-0.5">ホーム</h2>
          <p className="text-sm text-gray-500">プロジェクトを選択してワークフローを開始</p>
        </div>
        <div className="flex gap-2">
          <input ref={importRef} type="file" accept=".lbp" className="hidden" onChange={(e) => void handleImport(e)} />
          <button
            onClick={() => importRef.current?.click()}
            disabled={importing}
            className="flex items-center gap-2 bg-gray-800 hover:bg-gray-700 border border-gray-700 text-gray-300 rounded-lg px-3 py-2 text-sm transition-colors"
          >
            <Upload size={14} />
            {importing ? "読込中..." : ".lbp インポート"}
          </button>
          {selectedProject && (
            <button
              onClick={() => void handleExport()}
              className="flex items-center gap-2 bg-gray-800 hover:bg-gray-700 border border-gray-700 text-gray-300 rounded-lg px-3 py-2 text-sm transition-colors"
            >
              <Download size={14} />
              .lbp 保存
            </button>
          )}
        </div>
      </div>

      {/* 統計（プロジェクトがあるとき） */}
      {projects.length > 0 && (
        <div className="grid grid-cols-4 gap-3">
          {[
            { label: "プロジェクト", value: projects.length, icon: <FolderOpen size={18} className="text-indigo-400" />, color: "bg-indigo-950/60" },
            { label: "学習中", value: Object.values(statuses).filter(s => s.status === "training").length, icon: <Zap size={18} className="text-amber-400" />, color: "bg-amber-950/60" },
            { label: "ツール連携", value: `${integrationOk}/${integrationTotal}`, icon: integrationOk === integrationTotal ? <CheckCircle2 size={18} className="text-emerald-400" /> : <AlertCircle size={18} className="text-amber-400" />, color: integrationOk === integrationTotal ? "bg-emerald-950/60" : "bg-amber-950/60" },
            { label: "完了", value: Object.values(statuses).filter(s => s.status === "completed").length, icon: <TrendingUp size={18} className="text-violet-400" />, color: "bg-violet-950/60" },
          ].map(({ label, value, icon, color }) => (
            <div key={label} className={`${color} border border-gray-700/50 rounded-xl p-4 flex items-center gap-3`}>
              <div className={`p-2 rounded-lg bg-gray-900/60`}>{icon}</div>
              <div>
                <div className="text-2xl font-bold text-gray-100">{value}</div>
                <div className="text-xs text-gray-500">{label}</div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* プロジェクトリスト */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
        <div className="px-5 py-3.5 border-b border-gray-800 flex items-center justify-between">
          <h3 className="text-sm font-semibold text-gray-300">プロジェクト</h3>
          <button
            onClick={() => setShowCreateForm((v) => !v)}
            className="flex items-center gap-1.5 bg-orange-600 hover:bg-orange-500 text-white rounded-lg px-3 py-1.5 text-xs font-medium transition-colors"
          >
            <Plus size={13} />
            新規作成
          </button>
        </div>

        {/* 作成フォーム (展開式) */}
        {showCreateForm && (
          <div className="px-5 py-4 border-b border-gray-800 bg-gray-800/50">
            <div className="flex gap-2">
              <input
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && void handleCreate()}
                placeholder="project_name"
                autoFocus
                className="flex-1 bg-gray-700 border border-gray-600 text-gray-100 placeholder-gray-500 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-orange-500 focus:ring-1 focus:ring-orange-500/30"
              />
              <div className="relative">
                <select
                  value={newType}
                  onChange={(e) => setNewType(e.target.value as "character" | "style")}
                  className="appearance-none bg-gray-700 border border-gray-600 text-gray-100 rounded-lg pl-3 pr-8 py-2 text-sm focus:outline-none focus:border-orange-500"
                >
                  <option value="character">Character</option>
                  <option value="style">Style</option>
                </select>
                <ChevronDown size={13} className="absolute right-2 top-1/2 -translate-y-1/2 text-gray-400 pointer-events-none" />
              </div>
              <button
                onClick={() => void handleCreate()}
                disabled={!newName.trim() || creating}
                className="flex items-center gap-1.5 bg-orange-600 hover:bg-orange-500 disabled:bg-gray-700 disabled:text-gray-500 text-white rounded-lg px-4 py-2 text-sm font-medium transition-colors"
              >
                {creating ? "作成中..." : "作成"}
              </button>
            </div>
          </div>
        )}

        {/* プロジェクト行 */}
        {projects.length === 0 ? (
          <div className="py-14 text-center">
            <FolderOpen size={36} className="text-gray-700 mx-auto mb-3" />
            <p className="text-gray-500 text-sm mb-1">プロジェクトがありません</p>
            <p className="text-gray-600 text-xs">「新規作成」からLoRAプロジェクトを作成してください</p>
          </div>
        ) : (
          <div className="divide-y divide-gray-800">
            {projects.map((p) => {
              const st = statuses[p.id];
              const isSelected = p.id === selectedProjectId;
              return (
                <div
                  key={p.id}
                  className={[
                    "px-5 py-3.5 flex items-center gap-3 transition-colors cursor-pointer",
                    isSelected ? "bg-orange-950/20 border-l-2 border-orange-500" : "hover:bg-gray-800/60 border-l-2 border-transparent",
                  ].join(" ")}
                  onClick={() => onSelectProject(p.id)}
                >
                  <StatusDot status={st?.status ?? "idle"} />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-semibold text-gray-100 truncate">{p.name}</span>
                      {isSelected && <span className="text-[10px] bg-orange-900 text-orange-300 px-1.5 py-0.5 rounded font-medium">選択中</span>}
                    </div>
                    <div className="flex items-center gap-2 mt-0.5 flex-wrap">
                      {p.project_type === "character" ? <User size={11} className="text-gray-500" /> : <Palette size={11} className="text-gray-500" />}
                      <span className="text-xs text-gray-500">{p.project_type}</span>
                      {/* フェーズミニバー */}
                      {(() => {
                        const status = st?.status ?? "idle";
                        const phases = [
                          { label: "Dataset", done: status !== "idle" || true },
                          { label: "Training", done: status === "training" || status === "completed" || status === "paused" },
                          { label: "Done", done: status === "completed" },
                        ];
                        return (
                          <div className="flex items-center gap-0.5">
                            {phases.map((ph, i) => (
                              <span key={ph.label} className="flex items-center gap-0.5">
                                <span className={["text-[10px] px-1.5 py-0.5 rounded", ph.done ? "bg-emerald-900/60 text-emerald-400" : "bg-gray-800 text-gray-600"].join(" ")}>{ph.label}</span>
                                {i < phases.length - 1 && <span className="text-gray-700 text-[10px]">›</span>}
                              </span>
                            ))}
                          </div>
                        );
                      })()}
                      {st?.status === "training" && st.progress_percent != null && (
                        <span className="text-xs text-amber-400">{st.epoch}/{st.total_epochs} ep · {st.progress_percent.toFixed(0)}%</span>
                      )}
                    </div>
                  </div>
                  <StatusBadge status={st?.status ?? "idle"} />
                  {/* Actions */}
                  <div className="flex items-center gap-0.5" onClick={(e) => e.stopPropagation()}>
                    <button
                      onClick={() => void handleDuplicate(p.id, p.name)}
                      className="p-1.5 rounded-lg text-gray-500 hover:text-gray-300 hover:bg-gray-700 transition-colors"
                      title="複製"
                    >
                      <Copy size={14} />
                    </button>
                    <button
                      onClick={() => void handleDelete(p.id)}
                      className={["p-1.5 rounded-lg transition-colors", deleting === p.id ? "bg-red-900/60 text-red-300 animate-pulse" : "text-gray-500 hover:text-red-400 hover:bg-red-950/60"].join(" ")}
                      title={deleting === p.id ? "もう一度クリックで削除" : "削除"}
                    >
                      <Trash2 size={14} />
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* 次のステップ */}
      {nextSteps.length > 0 && (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
          <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">次のステップ</h3>
          <div className="space-y-2">
            {nextSteps.slice(0, 2).map((step) => (
              <button
                key={step.tab}
                onClick={() => onTabChange(step.tab)}
                className="flex items-center gap-3 w-full text-left bg-gray-800/60 border border-gray-700/60 hover:border-orange-700/50 hover:bg-orange-950/20 rounded-xl px-4 py-3 transition-colors"
              >
                <div className="text-orange-400">{step.icon}</div>
                <div className="flex-1">
                  <div className="text-sm font-semibold text-gray-200">{step.label}</div>
                  <div className="text-xs text-gray-500 mt-0.5">{step.desc}</div>
                </div>
                <ArrowRight size={14} className="text-gray-600" />
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
