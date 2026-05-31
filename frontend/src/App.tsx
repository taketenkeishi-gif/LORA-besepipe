import { useEffect, useMemo, useState } from "react";
import Layout from "./components/Layout";
import Dashboard from "./components/Dashboard";
import Projects from "./components/Projects";
import Dataset from "./components/Dataset";
import Training from "./components/Training";
import Library from "./components/Library";
import Integrations from "./components/Integrations";
import { apiGet, apiPost } from "./lib/api";
import type {
  TabId,
  Project,
  TrainingStatus,
  ToolPaths,
  IntegrationStatus,
  PreviewPrompts,
} from "./types";

export default function App() {
  const [tab, setTab] = useState<TabId>("dataset");
  const [apiHealth, setApiHealth] = useState("確認中");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  const [projects, setProjects] = useState<Project[]>([]);
  const [selectedProjectId, setSelectedProjectId] = useState<number | null>(null);
  const [statuses, setStatuses] = useState<Record<number, TrainingStatus>>({});

  const [toolPaths, setToolPaths] = useState<ToolPaths>({
    python_exe: "",
    kohya_root: "",
    comfyui_root: "",
    wd14_script: "",
    temp_dir: "",
    dataset_base_dir: "",
  });
  const [integrationStatus, setIntegrationStatus] =
    useState<IntegrationStatus | null>(null);
  const [prompts, setPrompts] = useState<PreviewPrompts>({
    positive_prompt: "",
    negative_prompt: "",
  });

  const selectedProject = useMemo(
    () => projects.find((p) => p.id === selectedProjectId) ?? null,
    [projects, selectedProjectId],
  );

  const showError = (msg: string) => {
    setError(msg);
    setNotice("");
  };
  const showNotice = (msg: string) => {
    setNotice(msg);
    setError("");
  };
  const clearMessages = () => {
    setError("");
    setNotice("");
  };

  async function refreshRuntime() {
    try {
      const health = await apiGet<{ status: string }>("/health");
      setApiHealth(health.status === "ok" ? "接続OK" : health.status);
      const list = await apiGet<Project[]>("/projects");
      setProjects(list);
      const entries = await Promise.all(
        list.map(
          async (p) =>
            [
              p.id,
              await apiGet<TrainingStatus>(
                `/training/status?project_id=${p.id}`,
              ),
            ] as const,
        ),
      );
      setStatuses(Object.fromEntries(entries));
      if (selectedProjectId === null && list.length > 0)
        setSelectedProjectId(list[0].id);
    } catch {
      setApiHealth("offline");
    }
  }

  async function loadIntegrations() {
    try {
      setToolPaths(await apiGet<ToolPaths>("/settings/tool-paths"));
      setIntegrationStatus(
        await apiGet<IntegrationStatus>("/settings/integrations/status"),
      );
    } catch {}
  }

  async function loadPreviewPrompts() {
    try {
      setPrompts(await apiGet<PreviewPrompts>("/settings/preview-prompts"));
    } catch {}
  }

  async function refreshAll() {
    await Promise.all([
      refreshRuntime(),
      loadIntegrations(),
      loadPreviewPrompts(),
    ]);
  }

  useEffect(() => {
    void refreshAll();
    const timer = setInterval(() => void refreshRuntime(), 2000);
    return () => clearInterval(timer);
  }, []);

  const commonProps = {
    projects,
    selectedProjectId,
    selectedProject,
    statuses,
    onSelectProject: setSelectedProjectId,
    onRefresh: refreshRuntime,
    showError,
    showNotice,
    clearMessages,
  };

  return (
    <Layout
      tab={tab}
      onTabChange={(t) => {
        setTab(t);
        clearMessages();
      }}
      apiHealth={apiHealth}
      error={error}
      notice={notice}
      onRefresh={() => void refreshAll()}
    >
      {tab === "dashboard" && (
        <Dashboard
          projects={projects}
          statuses={statuses}
          integrationStatus={integrationStatus}
        />
      )}
      {tab === "projects" && <Projects {...commonProps} />}
      {tab === "dataset" && <Dataset {...commonProps} />}
      {tab === "training" && (
        <Training
          {...commonProps}
          prompts={prompts}
          onPromptsChange={setPrompts}
        />
      )}
      {tab === "library" && (
        <Library
          projects={projects}
          selectedProjectId={selectedProjectId}
          showError={showError}
          showNotice={showNotice}
        />
      )}
      {tab === "integrations" && (
        <Integrations
          toolPaths={toolPaths}
          onToolPathsChange={setToolPaths}
          integrationStatus={integrationStatus}
          showError={showError}
          showNotice={showNotice}
          onReload={loadIntegrations}
        />
      )}
      {tab === "guide" && (
        <div className="max-w-2xl space-y-4">
          <div>
            <h2 className="text-xl font-bold text-gray-100 mb-1">使い方ガイド</h2>
            <p className="text-sm text-gray-400">LoRA制作ワークフロー</p>
          </div>
          <div className="bg-gray-800 border border-gray-700 rounded-xl p-6">
            <ol className="space-y-4">
              {[
                ["プロジェクト作成", "「プロジェクト管理」タブでLoRAの種別(character/style)を選んで作成"],
                ["候補画像収集", "「データセット作成」でURLまたはD&Dで画像を収集し、フィルタ・ソートで絞り込む"],
                ["フォルダ作成と取り込み", "「2) Createフォルダー」→「3) 取り込み」→「4) タグ生成」の順に実行"],
                ["学習実行", "「学習制御」タブでKOHYAパラメータを設定し「学習開始」。Epochごとにプレビュー生成"],
                ["外部ツール設定", "「外部連携設定」でPython/kohya/ComfyUI/WD14のパスを設定"],
              ].map(([title, desc], i) => (
                <li key={i} className="flex gap-4">
                  <span className="flex-shrink-0 w-7 h-7 rounded-full bg-indigo-900 text-indigo-300 text-sm font-bold flex items-center justify-center">
                    {i + 1}
                  </span>
                  <div>
                    <div className="text-sm font-semibold text-gray-200 mb-0.5">{title}</div>
                    <div className="text-sm text-gray-400">{desc}</div>
                  </div>
                </li>
              ))}
            </ol>
          </div>
        </div>
      )}
    </Layout>
  );
}
