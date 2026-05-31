import { FolderKanban, Zap, CheckCircle2, Clock, AlertCircle, TrendingUp } from "lucide-react";
import type { Project, TrainingStatus, IntegrationStatus } from "../types";

type Props = {
  projects: Project[];
  statuses: Record<number, TrainingStatus>;
  integrationStatus: IntegrationStatus | null;
};

function StatCard({
  label,
  value,
  sub,
  icon,
  color,
}: {
  label: string;
  value: string | number;
  sub?: string;
  icon: React.ReactNode;
  color: string;
}) {
  return (
    <div className="bg-gray-800 border border-gray-700 rounded-xl p-5">
      <div className="flex items-start justify-between">
        <div>
          <div className="text-xs text-gray-400 mb-1">{label}</div>
          <div className="text-3xl font-bold text-gray-100">{value}</div>
          {sub && <div className="text-xs text-gray-500 mt-1">{sub}</div>}
        </div>
        <div className={`p-2.5 rounded-lg ${color}`}>{icon}</div>
      </div>
    </div>
  );
}

export default function Dashboard({ projects, statuses, integrationStatus }: Props) {
  const trainingCount = Object.values(statuses).filter(
    (s) => s.status === "training",
  ).length;
  const integrationOk = integrationStatus
    ? Object.values(integrationStatus.checks).filter((x) => x.ok).length
    : 0;
  const integrationTotal = integrationStatus
    ? Object.keys(integrationStatus.checks).length
    : 6;

  const recentProjects = projects.slice(-5).reverse();

  return (
    <div className="space-y-6 max-w-5xl">
      <div>
        <h2 className="text-xl font-bold text-gray-100 mb-1">ダッシュボード</h2>
        <p className="text-sm text-gray-400">LoRA制作ワークベンチの概要</p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <StatCard
          label="プロジェクト総数"
          value={projects.length}
          icon={<FolderKanban size={20} className="text-indigo-400" />}
          color="bg-indigo-950"
        />
        <StatCard
          label="学習中"
          value={trainingCount}
          sub={trainingCount > 0 ? "進行中" : "待機中"}
          icon={<Zap size={20} className={trainingCount > 0 ? "text-amber-400" : "text-gray-500"} />}
          color={trainingCount > 0 ? "bg-amber-950" : "bg-gray-700"}
        />
        <StatCard
          label="連携OK"
          value={`${integrationOk}/${integrationTotal}`}
          sub="ツール接続状態"
          icon={
            integrationOk === integrationTotal ? (
              <CheckCircle2 size={20} className="text-emerald-400" />
            ) : (
              <AlertCircle size={20} className="text-amber-400" />
            )
          }
          color={integrationOk === integrationTotal ? "bg-emerald-950" : "bg-amber-950"}
        />
        <StatCard
          label="完了プロジェクト"
          value={
            Object.values(statuses).filter((s) => s.status === "completed")
              .length
          }
          icon={<TrendingUp size={20} className="text-violet-400" />}
          color="bg-violet-950"
        />
      </div>

      {/* Project Status Table */}
      {projects.length > 0 && (
        <div className="bg-gray-800 border border-gray-700 rounded-xl overflow-hidden">
          <div className="px-5 py-4 border-b border-gray-700">
            <h3 className="text-sm font-semibold text-gray-200">プロジェクト一覧</h3>
          </div>
          <div className="divide-y divide-gray-700">
            {recentProjects.map((p) => {
              const st = statuses[p.id];
              return (
                <div key={p.id} className="px-5 py-3 flex items-center gap-4">
                  <div className="flex-1 min-w-0">
                    <div className="text-sm font-medium text-gray-200 truncate">
                      {p.name}
                    </div>
                    <div className="text-xs text-gray-500">{p.project_type}</div>
                  </div>
                  <StatusBadge status={st?.status ?? "idle"} />
                  {st?.status === "training" && st.progress_percent != null && (
                    <div className="w-24">
                      <div className="h-1.5 bg-gray-700 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-indigo-500 rounded-full transition-all"
                          style={{ width: `${Math.min(100, st.progress_percent)}%` }}
                        />
                      </div>
                      <div className="text-xs text-gray-500 mt-0.5 text-right">
                        {st.progress_percent.toFixed(0)}%
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {projects.length === 0 && (
        <div className="bg-gray-800 border border-gray-700 border-dashed rounded-xl p-12 text-center">
          <FolderKanban size={40} className="text-gray-600 mx-auto mb-3" />
          <p className="text-gray-400 text-sm">
            プロジェクトがありません。「プロジェクト管理」から作成してください。
          </p>
        </div>
      )}
    </div>
  );
}

function StatusBadge({ status }: { status: string }) {
  const map: Record<string, { label: string; cls: string }> = {
    idle: { label: "待機中", cls: "bg-gray-700 text-gray-400" },
    training: { label: "学習中", cls: "bg-amber-900 text-amber-300 animate-pulse" },
    paused: { label: "一時停止", cls: "bg-blue-900 text-blue-300" },
    completed: { label: "完了", cls: "bg-emerald-900 text-emerald-300" },
  };
  const { label, cls } = map[status] ?? { label: status, cls: "bg-gray-700 text-gray-400" };
  return (
    <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${cls}`}>
      {label}
    </span>
  );
}
