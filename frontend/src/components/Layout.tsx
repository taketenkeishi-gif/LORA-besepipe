import type { ReactNode } from "react";
import {
  LayoutDashboard,
  FolderKanban,
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
} from "lucide-react";
import type { TabId } from "../types";

type NavItem = { id: TabId; label: string; icon: ReactNode };

const NAV_ITEMS: NavItem[] = [
  { id: "dashboard", label: "ダッシュボード", icon: <LayoutDashboard size={18} /> },
  { id: "projects", label: "プロジェクト管理", icon: <FolderKanban size={18} /> },
  { id: "dataset", label: "データセット作成", icon: <ImagePlus size={18} /> },
  { id: "training", label: "学習制御", icon: <Wand2 size={18} /> },
  { id: "library", label: "LoRA ライブラリ", icon: <Archive size={18} /> },
  { id: "integrations", label: "外部連携設定", icon: <Settings2 size={18} /> },
  { id: "guide", label: "使い方ガイド", icon: <BookOpen size={18} /> },
];

type Props = {
  tab: TabId;
  onTabChange: (tab: TabId) => void;
  apiHealth: string;
  error: string;
  notice: string;
  onRefresh: () => void;
  children: ReactNode;
};

export default function Layout({
  tab,
  onTabChange,
  apiHealth,
  error,
  notice,
  onRefresh,
  children,
}: Props) {
  const isOnline = apiHealth === "接続OK";
  const isChecking = apiHealth === "確認中";

  return (
    <div className="flex h-screen bg-gray-950 text-gray-100 overflow-hidden">
      {/* Sidebar */}
      <aside className="w-64 flex-shrink-0 bg-gray-900 border-r border-gray-800 flex flex-col">
        {/* Logo */}
        <div className="px-5 py-5 border-b border-gray-800 bg-gradient-to-r from-gray-900 to-gray-800">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-orange-500 to-amber-600 flex items-center justify-center flex-shrink-0 shadow-lg">
              <Sparkles size={16} className="text-white" />
            </div>
            <div>
              <div className="text-sm font-bold text-white leading-tight">LoRA Basepipe</div>
              <div className="text-xs text-amber-400/70 leading-tight font-semibold">制作ワークベンチ</div>
            </div>
          </div>
        </div>

        {/* Nav */}
        <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto scrollbar-thin">
          {NAV_ITEMS.map(({ id, label, icon }) => (
            <button
              key={id}
              onClick={() => onTabChange(id)}
              className={[
                "w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors text-left",
                tab === id
                  ? "bg-gradient-to-r from-orange-600 to-amber-600 text-white shadow-md"
                  : "text-gray-400 hover:text-gray-100 hover:bg-gray-800",
              ].join(" ")}
            >
              <span className="flex-shrink-0">{icon}</span>
              <span>{label}</span>
            </button>
          ))}
        </nav>

        {/* API Status */}
        <div className="px-3 pb-4">
          <div className="bg-gray-800 rounded-lg p-3 border border-gray-700">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2 text-xs">
                {isOnline ? (
                  <Wifi size={13} className="text-emerald-400" />
                ) : isChecking ? (
                  <Wifi size={13} className="text-gray-500 animate-pulse" />
                ) : (
                  <WifiOff size={13} className="text-red-400" />
                )}
                <span
                  className={
                    isOnline
                      ? "text-emerald-400"
                      : isChecking
                        ? "text-gray-500"
                        : "text-red-400"
                  }
                >
                  {apiHealth}
                </span>
              </div>
              <button
                onClick={onRefresh}
                className="p-1 rounded hover:bg-gray-700 text-gray-500 hover:text-gray-300 transition-colors"
                title="更新"
              >
                <RefreshCw size={13} />
              </button>
            </div>
            <div className="text-xs text-gray-600">127.0.0.1:8000</div>
          </div>
        </div>
      </aside>

      {/* Main */}
      <main className="flex-1 flex flex-col overflow-hidden">
        {/* Notification bar */}
        {(error || notice) && (
          <div
            className={[
              "px-6 py-3 flex items-center gap-2 text-sm flex-shrink-0",
              error
                ? "bg-red-950 border-b border-red-900 text-red-300"
                : "bg-emerald-950 border-b border-emerald-900 text-emerald-300",
            ].join(" ")}
          >
            {error ? (
              <AlertCircle size={15} className="flex-shrink-0" />
            ) : (
              <CheckCircle2 size={15} className="flex-shrink-0" />
            )}
            <span>{error || notice}</span>
          </div>
        )}

        {/* Content */}
        <div className="flex-1 overflow-y-auto scrollbar-thin p-6">
          {children}
        </div>
      </main>
    </div>
  );
}
