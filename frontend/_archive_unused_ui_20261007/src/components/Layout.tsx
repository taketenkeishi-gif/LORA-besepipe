import { type ReactNode, useEffect, useState } from "react";
import {
  Home,
  LayoutDashboard,
  ImagePlus,
  Wand2,
  Settings2,
  BookOpen,
  Archive,
  RefreshCw,
  Wifi,
  WifiOff,
  AlertCircle,
  CheckCircle2,
  Sparkles,
  X,
  ChevronRight,
  Activity,
  GitCompare,
} from "lucide-react";
import type { TabId, Project, TrainingStatus, SystemStatus } from "../types";
import { API_BASE } from "../lib/api";

type Props = {
  tab: TabId;
  onTabChange: (tab: TabId) => void;
  apiHealth: string;
  systemStatus?: SystemStatus | null;
  error: string;
  notice: string;
  onRefresh: () => void;
  selectedProjectName: string | null;
  coverImages?: string[];
  openProjects?: Project[];
  selectedProjectId?: number | null;
  statuses?: Record<number, TrainingStatus>;
  onSelectProject?: (id: number) => void;
  onCloseProject?: (id: number) => void;
  children: ReactNode;
};

type Toast = { id: number; type: "error" | "notice"; message: string };

export default function Layout({
  tab,
  onTabChange,
  apiHealth,
  systemStatus = null,
  error,
  notice,
  onRefresh,
  selectedProjectName,
  coverImages = [],
  openProjects = [],
  selectedProjectId = null,
  statuses = {},
  onSelectProject,
  onCloseProject,
  children,
}: Props) {
  const isOnline = apiHealth === "接続OK";
  const isChecking = apiHealth === "確認中";
  const hasProject = selectedProjectName !== null;

  const [toasts, setToasts] = useState<Toast[]>([]);
  const [nextId, setNextId] = useState(0);

  const dismissToast = (id: number) =>
    setToasts((prev) => prev.filter((t) => t.id !== id));

  useEffect(() => {
    if (!error && !notice) {
      setToasts((prev) => prev.filter((toast) => toast.type !== "error"));
      return;
    }
    const id = nextId;
    setNextId((n) => n + 1);
    const type = error ? "error" : "notice";
    const message = error || notice;
    setToasts((prev) => [...prev.slice(-2), { id, type, message }]);
    if (type === "notice") {
      const timer = setTimeout(() => dismissToast(id), 5000);
      return () => clearTimeout(timer);
    }
  }, [error, notice]);

  const projectTabs: { id: TabId; label: string; icon: ReactNode }[] = [
    { id: "overview", label: "Overview", icon: <LayoutDashboard size={15} /> },
    { id: "dataset", label: "データセット", icon: <ImagePlus size={15} /> },
    { id: "training", label: "学習制御", icon: <Wand2 size={15} /> },
    { id: "runs", label: "Runs", icon: <Activity size={15} /> },
    { id: "compare", label: "Compare", icon: <GitCompare size={15} /> },
    { id: "library", label: "LoRAライブラリ", icon: <Archive size={15} /> },
  ];

  return (
    <div className="flex h-screen bg-gray-950 text-gray-100 overflow-hidden">
      {/* Sidebar */}
      <aside className="w-56 flex-shrink-0 bg-gray-900 border-r border-gray-800 flex flex-col">
        {/* Logo */}
        <div className="px-4 py-3.5 border-b border-gray-800">
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-indigo-500 to-indigo-700 flex items-center justify-center flex-shrink-0 shadow-md shadow-indigo-900/40">
              <Sparkles size={14} className="text-white" />
            </div>
            <div>
              <div className="text-sm font-bold text-white leading-tight">LoRA Basepipe</div>
              <div className="text-[10px] text-indigo-300/60 leading-tight font-medium">制作ワークベンチ</div>
            </div>
          </div>
        </div>

        {/* Nav */}
        <nav className="flex-1 px-2 py-2 overflow-y-auto space-y-0.5">

          {/* ホーム */}
          <button
            onClick={() => onTabChange("home")}
            className={[
              "w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm font-medium transition-colors text-left",
              tab === "home"
                ? "bg-indigo-600 text-white shadow-sm shadow-indigo-900/40"
                : "text-gray-400 hover:text-gray-100 hover:bg-gray-800",
            ].join(" ")}
          >
            <Home size={16} className="flex-shrink-0" />
            <span>ホーム</span>
          </button>

          {/* プロジェクトタブ */}
          {openProjects.length > 0 && (
            <div className="mt-1">
              <div className="px-3 pt-1.5 pb-1 text-[10px] text-gray-500 font-semibold uppercase tracking-wide">
                プロジェクト
              </div>
              <div className="space-y-0.5">
                {openProjects.map((project) => {
                  const isActive = project.id === selectedProjectId;
                  const isTraining = statuses[project.id]?.status === "training";
                  return (
                    <div key={project.id}>
                      {/* プロジェクトタブ行 */}
                      <div className={[
                        "group flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg transition-colors cursor-pointer",
                        isActive
                          ? "bg-indigo-600/25 border border-indigo-700/40"
                          : "hover:bg-gray-800 border border-transparent",
                      ].join(" ")}
                        onClick={() => onSelectProject?.(project.id)}
                      >
                        {/* 学習中インジケーター */}
                        <span className={[
                          "w-1.5 h-1.5 rounded-full flex-shrink-0",
                          isTraining ? "bg-emerald-400 animate-pulse" : isActive ? "bg-indigo-400" : "bg-gray-600",
                        ].join(" ")} />
                        <span className={[
                          "flex-1 text-xs font-medium truncate",
                          isActive ? "text-indigo-200" : "text-gray-400 group-hover:text-gray-200",
                        ].join(" ")}>
                          {project.name}
                        </span>
                        {/* 閉じるボタン */}
                        <button
                          onClick={(e) => { e.stopPropagation(); onCloseProject?.(project.id); }}
                          className="opacity-0 group-hover:opacity-50 hover:!opacity-100 p-0.5 rounded hover:bg-gray-600 transition-all flex-shrink-0"
                          title="閉じる"
                        >
                          <X size={10} />
                        </button>
                      </div>
                      {/* アクティブなプロジェクトのサブナビ */}
                      {isActive && (
                        <div className="ml-3 pl-2.5 border-l border-indigo-800/40 mt-0.5 space-y-0.5">
                          {/* カバー画像ミニグリッド */}
                          {coverImages.length > 0 && (
                            <div className="pb-1.5">
                              <div className="grid grid-cols-4 gap-0.5 h-8 rounded overflow-hidden border border-indigo-800/30">
                                {Array.from({ length: 4 }).map((_, i) => {
                                  const src = coverImages[i % coverImages.length];
                                  return (
                                    <div key={i} className="bg-gray-900 overflow-hidden">
                                      {src && (
                                        <img
                                          src={`${API_BASE}/collector/thumbnail?path=${encodeURIComponent(src)}&size=80`}
                                          alt=""
                                          className="w-full h-full object-cover"
                                          draggable={false}
                                        />
                                      )}
                                    </div>
                                  );
                                })}
                              </div>
                            </div>
                          )}
                          {projectTabs.map((item) => (
                            <button
                              key={item.id}
                              onClick={() => onTabChange(item.id)}
                              className={[
                                "w-full flex items-center gap-2 px-2 py-1.5 rounded-lg text-xs font-medium transition-colors text-left",
                                tab === item.id
                                  ? "bg-gray-700 text-gray-100"
                                  : "text-gray-500 hover:text-gray-200 hover:bg-gray-800",
                              ].join(" ")}
                            >
                              <span className="flex-shrink-0 text-gray-400">{item.icon}</span>
                              <span>{item.label}</span>
                            </button>
                          ))}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </nav>

        {/* Footer icons */}
        <div className="px-2 pb-2 flex gap-1 border-t border-gray-800 pt-2">
          {(["integrations", "guide"] as const).map((id) => {
            const meta = {
              integrations: { icon: <Settings2 size={13} />, label: "設定" },
              guide: { icon: <BookOpen size={13} />, label: "ガイド" },
            }[id];
            return (
              <button
                key={id}
                onClick={() => onTabChange(id)}
                className={[
                  "flex-1 flex items-center justify-center gap-1 py-2 rounded-lg text-xs transition-colors",
                  tab === id ? "bg-gray-700 text-gray-100" : "text-gray-500 hover:text-gray-300 hover:bg-gray-800",
                ].join(" ")}
              >
                {meta.icon}
                <span>{meta.label}</span>
              </button>
            );
          })}
        </div>

        {/* API Status */}
        <div className="px-2 pb-3">
           <div className="bg-gray-800 rounded-lg p-2 border border-gray-700">
            <div className="flex items-center justify-between">
            <div className="flex items-center gap-1.5 text-xs">
              {isOnline ? (
                <Wifi size={11} className="text-emerald-400" />
              ) : isChecking ? (
                <Wifi size={11} className="text-gray-500 animate-pulse" />
              ) : (
                <WifiOff size={11} className="text-red-400" />
              )}
              <span className={isOnline ? "text-emerald-400" : isChecking ? "text-gray-500" : "text-red-400"}>
                {apiHealth}
              </span>
            </div>
            <button onClick={onRefresh} className="p-1 rounded hover:bg-gray-700 text-gray-500 hover:text-gray-300 transition-colors">
              <RefreshCw size={11} />
            </button>
            </div>
            {systemStatus && (
              <div className="mt-1.5 flex gap-2 text-[10px] text-gray-500" title="システム接続状態">
                {[
                  ["Anima", systemStatus.services.anima?.state],
                  ["ComfyUI", systemStatus.services.comfyui?.state],
                  ["GPU1", systemStatus.gpu.target_state ?? systemStatus.gpu.state],
                ].map(([label, state]) => {
                  const ready = state === "ready";
                  const offline = state === "offline" || state === "not_configured";
                  const stateLabel = ready ? "準備完了" : offline ? "オフライン" : state === "busy" ? "使用中/空き不足" : "要確認";
                  return (
                    <span key={label} className={ready ? "text-emerald-400" : offline ? "text-gray-500" : "text-amber-400"}>
                      {label} {stateLabel}
                    </span>
                  );
                })}
              </div>
            )}
           </div>
        </div>
      </aside>

      {/* Main */}
      <main className="flex-1 flex flex-col overflow-hidden">
        {/* Toasts */}
        {toasts.length > 0 && (
          <div className="fixed top-4 right-4 z-50 flex flex-col gap-2 pointer-events-none">
            {toasts.map((t) => (
              <div
                key={t.id}
                className={[
                  "flex items-center gap-2 px-4 py-3 rounded-lg shadow-lg text-sm max-w-sm pointer-events-auto",
                  t.type === "error"
                    ? "bg-red-900 border border-red-700 text-red-200"
                    : "bg-emerald-900 border border-emerald-700 text-emerald-200",
                ].join(" ")}
              >
                {t.type === "error" ? <AlertCircle size={14} className="flex-shrink-0" /> : <CheckCircle2 size={14} className="flex-shrink-0" />}
                <span className="flex-1">{t.message}</span>
                <button onClick={() => dismissToast(t.id)} className="ml-1 opacity-60 hover:opacity-100 transition-opacity">
                  <X size={12} />
                </button>
              </div>
            ))}
          </div>
        )}

        <div className="flex-1 overflow-y-auto p-4 md:p-6">
          {children}
        </div>
      </main>
    </div>
  );
}
