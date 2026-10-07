import { useState } from "react";
import {
  Plus,
  Copy,
  Trash2,
  FolderOpen,
  User,
  Palette,
  ChevronDown,
} from "lucide-react";
import { apiPost, apiDelete } from "../lib/api";
import type { Project, TrainingStatus } from "../types";

type Props = {
  projects: Project[];
  selectedProjectId: number | null;
  statuses: Record<number, TrainingStatus>;
  onSelectProject: (id: number) => void;
  onRefresh: () => Promise<void>;
  showError: (msg: string) => void;
  showNotice: (msg: string) => void;
};

function StatusDot({ status }: { status: string }) {
  const cls =
    status === "training"
      ? "bg-amber-400 animate-pulse"
      : status === "completed"
        ? "bg-emerald-400"
        : status === "paused"
          ? "bg-blue-400"
          : "bg-gray-500";
  return <span className={`inline-block w-2 h-2 rounded-full ${cls}`} />;
}

export default function Projects({
  projects,
  selectedProjectId,
  statuses,
  onSelectProject,
  onRefresh,
  showError,
  showNotice,
}: Props) {
  const [newName, setNewName] = useState("");
  const [newType, setNewType] = useState<"character" | "style">("character");
  const [deleting, setDeleting] = useState<number | null>(null);

  async function handleCreate() {
    if (!newName.trim()) return;
    try {
      const p = await apiPost<Project>("/projects", {
        name: newName.trim(),
        project_type: newType,
      });
      setNewName("");
      onSelectProject(p.id);
      showNotice(`プロジェクト「${p.name}」を作成しました`);
      await onRefresh();
    } catch (e) {
      showError(`作成失敗: ${String(e)}`);
    }
  }

  async function handleTypeChange(
    projectId: number,
    t: "character" | "style",
  ) {
    try {
      await apiPost(`/projects/${projectId}`, { project_type: t }, "PATCH");
      await onRefresh();
      showNotice(`種別を ${t} に変更`);
    } catch (e) {
      showError(`種別変更失敗: ${String(e)}`);
    }
  }

  async function handleDuplicate(projectId: number, srcName: string) {
    try {
      const p = await apiPost<Project>(`/projects/${projectId}/duplicate`, {
        name: `${srcName}_copy`,
      });
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

  return (
    <div className="space-y-6 max-w-4xl">
      <div>
        <h2 className="text-xl font-bold text-gray-100 mb-1">プロジェクト管理</h2>
        <p className="text-sm text-gray-400">
          作成したプロジェクトがデータセット作成・学習タブに同期されます
        </p>
      </div>

      {/* Create Form */}
      <div className="bg-gray-800 border border-gray-700 rounded-xl p-5">
        <h3 className="text-sm font-semibold text-gray-300 mb-4">
          新規プロジェクト作成
        </h3>
        <div className="flex gap-3">
          <input
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && void handleCreate()}
            placeholder="project_name"
            className="flex-1 bg-gray-700 border border-gray-600 text-gray-100 placeholder-gray-500 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
          />
          <select
            value={newType}
            onChange={(e) => setNewType(e.target.value as "character" | "style")}
            className="bg-gray-700 border border-gray-600 text-gray-100 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-indigo-500"
          >
            <option value="character">Character LoRA</option>
            <option value="style">Style LoRA</option>
          </select>
          <button
            onClick={() => void handleCreate()}
            disabled={!newName.trim()}
            className="flex items-center gap-2 bg-indigo-600 hover:bg-indigo-500 disabled:bg-gray-700 disabled:text-gray-500 text-white rounded-lg px-4 py-2 text-sm font-medium transition-colors"
          >
            <Plus size={16} />
            作成
          </button>
        </div>
      </div>

      {/* Project List */}
      <div className="space-y-3">
        {projects.length === 0 && (
          <div className="bg-gray-800 border border-gray-700 border-dashed rounded-xl p-10 text-center">
            <FolderOpen size={36} className="text-gray-600 mx-auto mb-2" />
            <p className="text-gray-400 text-sm">プロジェクトがありません</p>
          </div>
        )}
        {projects.map((p) => {
          const st = statuses[p.id];
          const isSelected = p.id === selectedProjectId;
          return (
            <div
              key={p.id}
              className={[
                "bg-gray-800 border rounded-xl p-4 transition-colors",
                isSelected
                  ? "border-indigo-500 ring-1 ring-indigo-500/30"
                  : "border-gray-700 hover:border-gray-600",
              ].join(" ")}
            >
              <div className="flex items-start gap-3">
                {/* Select button */}
                <button
                  onClick={() => onSelectProject(p.id)}
                  className="flex-1 text-left"
                >
                  <div className="flex items-center gap-2 mb-1">
                    <StatusDot status={st?.status ?? "idle"} />
                    <span className="text-sm font-semibold text-gray-100">
                      {p.name}
                    </span>
                    {isSelected && (
                      <span className="text-xs bg-indigo-900 text-indigo-300 px-1.5 py-0.5 rounded">
                        選択中
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-3 text-xs text-gray-400">
                    <span className="flex items-center gap-1">
                      {p.project_type === "character" ? (
                        <User size={11} />
                      ) : (
                        <Palette size={11} />
                      )}
                      {p.project_type}
                    </span>
                    <span>状態: {st?.status ?? "idle"}</span>
                    {st?.status === "training" && st.progress_percent != null && (
                      <span className="text-amber-400">
                        {st.epoch}/{st.total_epochs} epoch (
                        {st.progress_percent.toFixed(0)}%)
                      </span>
                    )}
                  </div>
                </button>

                {/* Actions */}
                <div className="flex items-center gap-1 flex-shrink-0">
                  <div className="relative">
                    <select
                      value={p.project_type}
                      onChange={(e) =>
                        void handleTypeChange(
                          p.id,
                          e.target.value as "character" | "style",
                        )
                      }
                      className="appearance-none bg-gray-700 border border-gray-600 text-gray-300 rounded-lg pl-2 pr-6 py-1.5 text-xs cursor-pointer focus:outline-none"
                    >
                      <option value="character">character</option>
                      <option value="style">style</option>
                    </select>
                    <ChevronDown
                      size={12}
                      className="absolute right-1.5 top-1/2 -translate-y-1/2 text-gray-400 pointer-events-none"
                    />
                  </div>
                  <button
                    onClick={() => void handleDuplicate(p.id, p.name)}
                    className="p-1.5 rounded-lg text-gray-400 hover:text-gray-200 hover:bg-gray-700 transition-colors"
                    title="複製"
                  >
                    <Copy size={15} />
                  </button>
                  <button
                    onClick={() => void handleDelete(p.id)}
                    className={[
                      "p-1.5 rounded-lg transition-colors",
                      deleting === p.id
                        ? "bg-red-900 text-red-300 animate-pulse"
                        : "text-gray-400 hover:text-red-400 hover:bg-red-950",
                    ].join(" ")}
                    title={deleting === p.id ? "もう一度クリックで削除" : "削除"}
                  >
                    <Trash2 size={15} />
                  </button>
                </div>
              </div>

              {/* Training progress bar */}
              {st?.status === "training" && st.progress_percent != null && (
                <div className="mt-3">
                  <div className="h-1.5 bg-gray-700 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-amber-500 rounded-full transition-all duration-500"
                      style={{
                        width: `${Math.min(100, st.progress_percent)}%`,
                      }}
                    />
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
