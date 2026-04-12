import { useEffect, useMemo, useState } from "react";

const API_BASE = "http://127.0.0.1:8000";

type TabId = "dashboard" | "projects" | "workflow" | "integrations";

type Project = {
  id: number;
  name: string;
  status: string;
  base_dir: string;
  dataset_dir: string;
  captions_dir: string;
  outputs_dir: string;
};

type TrainingStatus = {
  run_id: number | null;
  status: string;
  stop_mode: string | null;
  epoch: number;
  step: number;
  total_epochs: number;
  steps_per_epoch: number;
};

type ScanItem = {
  id: number;
  title: string;
  width: number;
  height: number;
  aspect: string;
};

type ToolPaths = {
  python_exe: string;
  kohya_root: string;
  comfyui_root: string;
  wd14_script: string;
};

type IntegrationStatus = {
  paths: ToolPaths;
  checks: Record<string, { ok: boolean; reason: string }>;
};

type PreviewTimelineItem = {
  checkpoint_id: number;
  epoch: number;
  step: number;
  mark: string;
  created_at: string;
  samples: Record<string, string>;
};

export default function App() {
  const [tab, setTab] = useState<TabId>("dashboard");
  const [apiHealth, setApiHealth] = useState<string>("checking");
  const [error, setError] = useState<string>("");

  const [projects, setProjects] = useState<Project[]>([]);
  const [newProjectName, setNewProjectName] = useState<string>("");
  const [selectedProjectId, setSelectedProjectId] = useState<number | null>(null);
  const [statuses, setStatuses] = useState<Record<number, TrainingStatus>>({});

  const [scanUrl, setScanUrl] = useState<string>("https://example.com/mock-collection");
  const [scanItems, setScanItems] = useState<ScanItem[]>([]);
  const [selectedScanIds, setSelectedScanIds] = useState<number[]>([]);
  const [namingTemplate, setNamingTemplate] = useState<string>("{title}_{index}");
  const [timeline, setTimeline] = useState<PreviewTimelineItem[]>([]);
  const [epochs, setEpochs] = useState<number>(5);
  const [stepsPerEpoch, setStepsPerEpoch] = useState<number>(20);

  const [toolPaths, setToolPaths] = useState<ToolPaths>({
    python_exe: "",
    kohya_root: "",
    comfyui_root: "",
    wd14_script: ""
  });
  const [integrationStatus, setIntegrationStatus] = useState<IntegrationStatus | null>(null);

  const selectedProject = useMemo(
    () => projects.find((p) => p.id === selectedProjectId) ?? null,
    [projects, selectedProjectId]
  );

  useEffect(() => {
    void refreshAll();
    const timer = window.setInterval(() => {
      void refreshRuntime();
    }, 2000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    if (selectedProjectId !== null) {
      void loadProjectRuntime(selectedProjectId);
    }
  }, [selectedProjectId]);

  async function apiGet<T>(path: string): Promise<T> {
    const res = await fetch(`${API_BASE}${path}`);
    if (!res.ok) throw new Error(`GET ${path} failed`);
    return (await res.json()) as T;
  }

  async function apiPost<T>(path: string, body: unknown, method: "POST" | "PUT" = "POST"): Promise<T> {
    const res = await fetch(`${API_BASE}${path}`, {
      method,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });
    const json = await res.json();
    if (!res.ok) throw new Error(json?.detail ?? `${method} ${path} failed`);
    return json as T;
  }

  async function refreshAll() {
    await Promise.all([refreshRuntime(), loadIntegrations()]);
  }

  async function refreshRuntime() {
    try {
      const health = await apiGet<{ status: string }>("/health");
      setApiHealth(health.status);
      const list = await apiGet<Project[]>("/projects");
      setProjects(list);

      const ids = list.map((x) => x.id);
      const statsEntries = await Promise.all(
        ids.map(async (id) => [id, await apiGet<TrainingStatus>(`/training/status?project_id=${id}`)] as const)
      );
      setStatuses(Object.fromEntries(statsEntries));

      if (selectedProjectId === null && list.length > 0) {
        setSelectedProjectId(list[0].id);
      }
    } catch (e) {
      setApiHealth("offline");
      setError(`API接続エラー: ${String(e)}`);
    }
  }

  async function loadProjectRuntime(projectId: number) {
    try {
      const p = await apiGet<{ timeline: PreviewTimelineItem[] }>(`/previews/${projectId}`);
      setTimeline(p.timeline);
    } catch {
      setTimeline([]);
    }
  }

  async function loadIntegrations() {
    try {
      const paths = await apiGet<ToolPaths>("/settings/tool-paths");
      setToolPaths(paths);
      const status = await apiGet<IntegrationStatus>("/settings/integrations/status");
      setIntegrationStatus(status);
    } catch {
      setIntegrationStatus(null);
    }
  }

  async function createProject() {
    if (!newProjectName.trim()) return;
    setError("");
    try {
      const created = await apiPost<Project>("/projects", { name: newProjectName.trim() });
      setNewProjectName("");
      setSelectedProjectId(created.id);
      await refreshRuntime();
    } catch (e) {
      setError(String(e));
    }
  }

  async function startTraining(projectId: number) {
    setError("");
    try {
      await apiPost("/training/start", {
        project_id: projectId,
        total_epochs: epochs,
        steps_per_epoch: stepsPerEpoch
      });
      await refreshRuntime();
    } catch (e) {
      setError(String(e));
    }
  }

  async function trainingAction(path: string, projectId: number) {
    setError("");
    try {
      await apiPost(path, { project_id: projectId });
      await refreshRuntime();
    } catch (e) {
      setError(String(e));
    }
  }

  async function runScan() {
    if (!selectedProject) return;
    setError("");
    try {
      const result = await apiPost<{ items: ScanItem[] }>("/collector/scan", {
        project_id: selectedProject.id,
        url: scanUrl
      });
      setScanItems(result.items);
      setSelectedScanIds(result.items.slice(0, 6).map((x) => x.id));
    } catch (e) {
      setError(String(e));
    }
  }

  async function runImport() {
    if (!selectedProject) return;
    setError("");
    try {
      await apiPost("/collector/import", {
        project_id: selectedProject.id,
        selected_ids: selectedScanIds,
        naming_template: namingTemplate
      });
      await refreshRuntime();
    } catch (e) {
      setError(String(e));
    }
  }

  async function runTags() {
    if (!selectedProject) return;
    setError("");
    try {
      await apiPost("/tags/generate", { project_id: selectedProject.id });
      await refreshRuntime();
    } catch (e) {
      setError(String(e));
    }
  }

  function toggleScanSelection(id: number) {
    setSelectedScanIds((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]));
  }

  async function saveToolPaths() {
    setError("");
    try {
      const saved = await apiPost<ToolPaths>("/settings/tool-paths", toolPaths, "PUT");
      setToolPaths(saved);
      await loadIntegrations();
    } catch (e) {
      setError(String(e));
    }
  }

  async function autoDetectToolPaths() {
    setError("");
    try {
      const detected = await apiPost<ToolPaths>("/settings/tool-paths/autodetect", {});
      setToolPaths(detected);
      await loadIntegrations();
    } catch (e) {
      setError(String(e));
    }
  }

  return (
    <div className="layout">
      <aside className="sidebar">
        <h1 className="logo">LoRA Workbench</h1>
        <nav className="menu">
          <button className={tab === "dashboard" ? "menuBtn active" : "menuBtn"} onClick={() => setTab("dashboard")}>
            Dashboard
          </button>
          <button className={tab === "projects" ? "menuBtn active" : "menuBtn"} onClick={() => setTab("projects")}>
            Projects
          </button>
          <button className={tab === "workflow" ? "menuBtn active" : "menuBtn"} onClick={() => setTab("workflow")}>
            Workflow
          </button>
          <button
            className={tab === "integrations" ? "menuBtn active" : "menuBtn"}
            onClick={() => setTab("integrations")}
          >
            Integrations
          </button>
        </nav>
        <div className="healthCard">
          <div>API: {apiHealth}</div>
          <button className="btn secondary small" onClick={() => void refreshAll()}>
            Refresh
          </button>
        </div>
      </aside>

      <main className="content">
        {error && <div className="errorBanner">{error}</div>}

        {tab === "dashboard" && (
          <section className="panel">
            <h2>Dashboard</h2>
            <div className="statsGrid">
              <div className="statBox">
                <div className="statLabel">Projects</div>
                <div className="statValue">{projects.length}</div>
              </div>
              <div className="statBox">
                <div className="statLabel">Training Active</div>
                <div className="statValue">
                  {Object.values(statuses).filter((s) => s.status === "training").length}
                </div>
              </div>
              <div className="statBox">
                <div className="statLabel">Integrations Ready</div>
                <div className="statValue">
                  {integrationStatus
                    ? Object.values(integrationStatus.checks).filter((x) => x.ok).length
                    : 0}
                  /4
                </div>
              </div>
            </div>
          </section>
        )}

        {tab === "projects" && (
          <section className="panel">
            <h2>Projects</h2>
            <div className="row">
              <input
                value={newProjectName}
                onChange={(e) => setNewProjectName(e.target.value)}
                placeholder="New project name"
              />
              <button className="btn primary" onClick={createProject}>
                Create
              </button>
            </div>

            <div className="projectTable">
              {projects.map((p) => (
                <button
                  key={p.id}
                  className={selectedProjectId === p.id ? "projectRow selected" : "projectRow"}
                  onClick={() => setSelectedProjectId(p.id)}
                >
                  <span>{p.name}</span>
                  <span>{statuses[p.id]?.status ?? "idle"}</span>
                </button>
              ))}
            </div>
          </section>
        )}

        {tab === "workflow" && (
          <section className="panel">
            <h2>Workflow</h2>
            {!selectedProject ? (
              <p className="muted">Projectを選択してください。</p>
            ) : (
              <>
                <p className="muted">Current: {selectedProject.name}</p>
                <div className="workflowGrid">
                  <div className="card">
                    <h3>Dataset Collection</h3>
                    <input value={scanUrl} onChange={(e) => setScanUrl(e.target.value)} placeholder="source URL" />
                    <button className="btn info" onClick={runScan}>
                      Scan
                    </button>
                    <div className="chips">
                      {scanItems.map((i) => (
                        <button
                          key={i.id}
                          className={selectedScanIds.includes(i.id) ? "chip active" : "chip"}
                          onClick={() => toggleScanSelection(i.id)}
                        >
                          {i.title}
                        </button>
                      ))}
                    </div>
                    <div className="row">
                      <input
                        value={namingTemplate}
                        onChange={(e) => setNamingTemplate(e.target.value)}
                        placeholder="{title}_{index}"
                      />
                      <button className="btn primary" onClick={runImport}>
                        Import
                      </button>
                    </div>
                    <button className="btn accent" onClick={runTags}>
                      Generate Tags
                    </button>
                  </div>

                  <div className="card">
                    <h3>Training Control</h3>
                    <div className="row">
                      <input
                        type="number"
                        value={epochs}
                        onChange={(e) => setEpochs(Number(e.target.value || 1))}
                        placeholder="epochs"
                      />
                      <input
                        type="number"
                        value={stepsPerEpoch}
                        onChange={(e) => setStepsPerEpoch(Number(e.target.value || 1))}
                        placeholder="steps/epoch"
                      />
                    </div>
                    <div className="row wrap">
                      <button className="btn primary" onClick={() => startTraining(selectedProject.id)}>
                        Start
                      </button>
                      <button className="btn secondary" onClick={() => trainingAction("/training/stop-at-epoch", selectedProject.id)}>
                        Stop@Epoch
                      </button>
                      <button className="btn warning" onClick={() => trainingAction("/training/stop-now", selectedProject.id)}>
                        StopNow
                      </button>
                      <button className="btn info" onClick={() => trainingAction("/training/resume", selectedProject.id)}>
                        Resume
                      </button>
                    </div>
                    <p className="muted">
                      status: {statuses[selectedProject.id]?.status ?? "idle"} / progress:{" "}
                      {statuses[selectedProject.id]?.epoch ?? 0}/{statuses[selectedProject.id]?.total_epochs ?? 0}
                    </p>
                  </div>
                </div>

                <div className="card">
                  <h3>Preview Timeline</h3>
                  {timeline.length === 0 ? (
                    <p className="muted">まだpreviewはありません。</p>
                  ) : (
                    <div className="timeline">
                      {timeline.map((t) => (
                        <div key={t.checkpoint_id} className="timelineItem">
                          <strong>Epoch {t.epoch}</strong>
                          <span className="muted">mark: {t.mark}</span>
                          <span className="muted">samples: {Object.keys(t.samples).join(", ")}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </>
            )}
          </section>
        )}

        {tab === "integrations" && (
          <section className="panel">
            <h2>Integrations</h2>
            <p className="muted">外部ツールのパスを設定して保存し、接続状態を確認します。</p>
            <div className="card">
              <div className="formGrid">
                <label>
                  Python Executable
                  <input
                    value={toolPaths.python_exe}
                    onChange={(e) => setToolPaths({ ...toolPaths, python_exe: e.target.value })}
                    placeholder="C:\\Python310\\python.exe"
                  />
                </label>
                <label>
                  kohya Root
                  <input
                    value={toolPaths.kohya_root}
                    onChange={(e) => setToolPaths({ ...toolPaths, kohya_root: e.target.value })}
                    placeholder="D:\\tools\\kohya_ss"
                  />
                </label>
                <label>
                  ComfyUI Root
                  <input
                    value={toolPaths.comfyui_root}
                    onChange={(e) => setToolPaths({ ...toolPaths, comfyui_root: e.target.value })}
                    placeholder="D:\\ComfyUI"
                  />
                </label>
                <label>
                  WD14 Script Path
                  <input
                    value={toolPaths.wd14_script}
                    onChange={(e) => setToolPaths({ ...toolPaths, wd14_script: e.target.value })}
                    placeholder="...\\tag_images_by_wd14_tagger.py"
                  />
                </label>
              </div>
              <div className="row wrap">
                <button className="btn primary" onClick={saveToolPaths}>
                  Save Paths
                </button>
                <button className="btn info" onClick={autoDetectToolPaths}>
                  Auto Detect
                </button>
                <button className="btn secondary" onClick={() => void loadIntegrations()}>
                  Check Status
                </button>
              </div>
            </div>
            <div className="card">
              <h3>Connection Status</h3>
              {!integrationStatus ? (
                <p className="muted">status not loaded</p>
              ) : (
                <ul className="statusList">
                  {Object.entries(integrationStatus.checks).map(([key, v]) => (
                    <li key={key}>
                      <strong>{key}</strong>: {v.ok ? "OK" : "NG"} ({v.reason})
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </section>
        )}
      </main>
    </div>
  );
}
