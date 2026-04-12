import { useEffect, useMemo, useState } from "react";

const API_BASE = "http://127.0.0.1:8000";

type TabId = "dashboard" | "projects" | "dataset" | "training" | "integrations" | "guide";

type Project = {
  id: number;
  name: string;
  project_type: "character" | "style";
  status: string;
  base_dir: string;
  dataset_dir: string;
  captions_dir: string;
  outputs_dir: string;
  library_dir: string;
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
  thumbnail_url?: string;
  tags?: string[];
};

type ToolPaths = {
  python_exe: string;
  kohya_root: string;
  comfyui_root: string;
  wd14_script: string;
  temp_dir: string;
  dataset_base_dir: string;
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
  sample_previews?: Record<string, string>;
};

export default function App() {
  const [tab, setTab] = useState<TabId>("dashboard");
  const [apiHealth, setApiHealth] = useState<string>("確認中");
  const [error, setError] = useState<string>("");
  const [notice, setNotice] = useState<string>("");

  const [projects, setProjects] = useState<Project[]>([]);
  const [newProjectName, setNewProjectName] = useState<string>("");
  const [newProjectType, setNewProjectType] = useState<"character" | "style">("character");
  const [selectedProjectId, setSelectedProjectId] = useState<number | null>(null);
  const [statuses, setStatuses] = useState<Record<number, TrainingStatus>>({});

  const [scanUrl, setScanUrl] = useState<string>("https://www.pixiv.net/artworks/143382932");
  const [scanKeyword, setScanKeyword] = useState<string>("");
  const [aspectFilter, setAspectFilter] = useState<string>("all");
  const [minW, setMinW] = useState<number>(0);
  const [minH, setMinH] = useState<number>(0);
  const [tagFilter, setTagFilter] = useState<string>("");
  const [scanItems, setScanItems] = useState<ScanItem[]>([]);
  const [selectedScanIds, setSelectedScanIds] = useState<number[]>([]);
  const [namingTemplate, setNamingTemplate] = useState<string>("{title}_{index}");

  const [repeatFolderTitle, setRepeatFolderTitle] = useState<string>("subject");
  const [repeatCount, setRepeatCount] = useState<number>(5);

  const [timeline, setTimeline] = useState<PreviewTimelineItem[]>([]);
  const [epochs, setEpochs] = useState<number>(5);
  const [alpha, setAlpha] = useState<number>(4);
  const [trainRepeats, setTrainRepeats] = useState<number>(5);

  const [toolPaths, setToolPaths] = useState<ToolPaths>({
    python_exe: "",
    kohya_root: "",
    comfyui_root: "",
    wd14_script: "",
    temp_dir: "",
    dataset_base_dir: ""
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

  function clearMessages() {
    setError("");
    setNotice("");
  }

  async function refreshAll() {
    await Promise.all([refreshRuntime(), loadIntegrations()]);
  }

  async function refreshRuntime() {
    try {
      const health = await apiGet<{ status: string }>("/health");
      setApiHealth(health.status === "ok" ? "接続OK" : health.status);
      const list = await apiGet<Project[]>("/projects");
      setProjects(list);
      const entries = await Promise.all(
        list.map(async (p) => [p.id, await apiGet<TrainingStatus>(`/training/status?project_id=${p.id}`)] as const)
      );
      setStatuses(Object.fromEntries(entries));
      if (selectedProjectId === null && list.length > 0) setSelectedProjectId(list[0].id);
    } catch (e) {
      setApiHealth("オフライン");
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
    clearMessages();
    try {
      const created = await apiPost<Project>("/projects", {
        name: newProjectName.trim(),
        project_type: newProjectType
      });
      setNewProjectName("");
      setSelectedProjectId(created.id);
      setNotice(`プロジェクト「${created.name}」を作成しました。`);
      await refreshRuntime();
    } catch (e) {
      setError(`作成失敗: ${String(e)}`);
    }
  }

  function buildKeyword() {
    const terms: string[] = [];
    if (scanKeyword.trim()) terms.push(scanKeyword.trim());
    if (tagFilter.trim()) terms.push(`tag:${tagFilter.trim()}`);
    if (aspectFilter !== "all") terms.push(`aspect:${aspectFilter}`);
    if (minW > 0) terms.push(`minw:${minW}`);
    if (minH > 0) terms.push(`minh:${minH}`);
    return terms.join(" ");
  }

  async function runScan() {
    if (!selectedProject) return;
    clearMessages();
    try {
      const result = await apiPost<{ items: ScanItem[]; detected: number; mode: string }>("/collector/scan", {
        project_id: selectedProject.id,
        url: scanUrl,
        keyword: buildKeyword(),
        limit: 60
      });
      setScanItems(result.items);
      setSelectedScanIds(result.items.slice(0, 12).map((x) => x.id));
      setNotice(`候補画像 ${result.detected} 件を取得 (${result.mode})。サムネイルを見て選択してください。`);
    } catch (e) {
      setError(`取得失敗: ${String(e)}`);
    }
  }

  async function runImport() {
    if (!selectedProject) return;
    clearMessages();
    try {
      await apiPost("/collector/import", {
        project_id: selectedProject.id,
        selected_ids: selectedScanIds,
        naming_template: namingTemplate
      });
      setNotice("取り込み完了。必要なら繰り返しフォルダを作成してください。");
      await refreshRuntime();
    } catch (e) {
      setError(`取り込み失敗: ${String(e)}`);
    }
  }

  async function prepareRepeatFolder() {
    if (!selectedProject) return;
    clearMessages();
    try {
      const res = await apiPost<{ folder: string; copied_images: number }>("/collector/prepare-repeat-folder", {
        project_id: selectedProject.id,
        repeats: repeatCount,
        folder_title: repeatFolderTitle
      });
      setNotice(`繰り返しフォルダを作成しました: ${res.folder} (${res.copied_images}枚コピー)`);
    } catch (e) {
      setError(`繰り返しフォルダ作成失敗: ${String(e)}`);
    }
  }

  async function runTags() {
    if (!selectedProject) return;
    clearMessages();
    try {
      const res = await apiPost<{ generated_count: number }>("/tags/generate", { project_id: selectedProject.id });
      setNotice(`タグ生成完了: ${res.generated_count} 件`);
      await refreshRuntime();
    } catch (e) {
      setError(`タグ生成失敗: ${String(e)}`);
    }
  }

  function toggleScanSelection(id: number) {
    setSelectedScanIds((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]));
  }

  async function startTraining(projectId: number) {
    clearMessages();
    try {
      const res = await apiPost<{ steps_per_epoch: number }>("/training/start", {
        project_id: projectId,
        epochs,
        repeats: trainRepeats,
        alpha
      });
      setNotice(`学習開始。1epochあたりstepは自動計算: ${res.steps_per_epoch}`);
      await refreshRuntime();
    } catch (e) {
      setError(`学習開始失敗: ${String(e)}`);
    }
  }

  async function trainingAction(path: string, okMessage: string) {
    if (!selectedProject) return;
    clearMessages();
    try {
      await apiPost(path, { project_id: selectedProject.id });
      setNotice(okMessage);
      await refreshRuntime();
    } catch (e) {
      setError(String(e));
    }
  }

  async function saveToolPaths() {
    clearMessages();
    try {
      await apiPost<ToolPaths>("/settings/tool-paths", toolPaths, "PUT");
      setNotice("設定保存完了");
      await loadIntegrations();
    } catch (e) {
      setError(String(e));
    }
  }

  async function autoDetectToolPaths() {
    clearMessages();
    try {
      const detected = await apiPost<ToolPaths>("/settings/tool-paths/autodetect", {});
      setToolPaths(detected);
      setNotice("自動検出完了");
      await loadIntegrations();
    } catch (e) {
      setError(String(e));
    }
  }

  function integrationRow(label: string, keyName: keyof ToolPaths) {
    const check = integrationStatus?.checks?.[keyName];
    const ok = !!check?.ok;
    return (
      <li key={keyName} className="statusRow">
        <span>{label}</span>
        <span className={ok ? "statusOk" : "statusNg"}>{ok ? "OK" : "NG"}</span>
        <span className="muted">{check?.reason ?? "未確認"}</span>
      </li>
    );
  }

  return (
    <div className="layout">
      <aside className="sidebar">
        <h1 className="logo">LoRA制作ワークベンチ</h1>
        <nav className="menu">
          <button className={tab === "dashboard" ? "menuBtn active" : "menuBtn"} onClick={() => setTab("dashboard")}>
            ダッシュボード
          </button>
          <button className={tab === "projects" ? "menuBtn active" : "menuBtn"} onClick={() => setTab("projects")}>
            プロジェクト管理
          </button>
          <button className={tab === "dataset" ? "menuBtn active" : "menuBtn"} onClick={() => setTab("dataset")}>
            データセット作成
          </button>
          <button className={tab === "training" ? "menuBtn active" : "menuBtn"} onClick={() => setTab("training")}>
            学習制御
          </button>
          <button
            className={tab === "integrations" ? "menuBtn active" : "menuBtn"}
            onClick={() => setTab("integrations")}
          >
            外部連携設定
          </button>
          <button className={tab === "guide" ? "menuBtn active" : "menuBtn"} onClick={() => setTab("guide")}>
            使い方ガイド
          </button>
        </nav>
        <div className="healthCard">
          <div>API状態: {apiHealth}</div>
          <button className="btn secondary small" onClick={() => void refreshAll()}>
            最新状態に更新
          </button>
        </div>
      </aside>

      <main className="content">
        {error && <div className="errorBanner">{error}</div>}
        {notice && <div className="noticeBanner">{notice}</div>}

        {tab === "dashboard" && (
          <section className="panel">
            <h2>ダッシュボード</h2>
            <div className="statsGrid">
              <div className="statBox">
                <div className="statLabel">登録プロジェクト</div>
                <div className="statValue">{projects.length}</div>
              </div>
              <div className="statBox">
                <div className="statLabel">学習中ジョブ</div>
                <div className="statValue">{Object.values(statuses).filter((s) => s.status === "training").length}</div>
              </div>
              <div className="statBox">
                <div className="statLabel">連携チェックOK</div>
                <div className="statValue">{integrationStatus ? Object.values(integrationStatus.checks).filter((x) => x.ok).length : 0}</div>
              </div>
            </div>
          </section>
        )}

        {tab === "projects" && (
          <section className="panel">
            <h2>プロジェクト管理</h2>
            <div className="row">
              <input value={newProjectName} onChange={(e) => setNewProjectName(e.target.value)} placeholder="例: style_demo" />
              <select value={newProjectType} onChange={(e) => setNewProjectType(e.target.value as "character" | "style")}>
                <option value="character">Character LoRA</option>
                <option value="style">Style LoRA</option>
              </select>
              <button className="btn primary" onClick={createProject}>
                作成
              </button>
            </div>
            <div className="projectTable">
              {projects.map((p) => (
                <button key={p.id} className={selectedProjectId === p.id ? "projectRow selected" : "projectRow"} onClick={() => setSelectedProjectId(p.id)}>
                  <span>{p.name}</span>
                  <span>
                    {p.project_type} / {statuses[p.id]?.status ?? "idle"}
                  </span>
                </button>
              ))}
            </div>
          </section>
        )}

        {tab === "dataset" && (
          <section className="panel">
            <h2>データセット作成</h2>
            {!selectedProject ? (
              <p className="muted">プロジェクトを選択してください。</p>
            ) : (
              <>
                <p className="guideLine">対象: {selectedProject.name} ({selectedProject.project_type})</p>
                <div className="card">
                  <label>収集元URL<input value={scanUrl} onChange={(e) => setScanUrl(e.target.value)} /></label>
                  <div className="row">
                    <label>キーワード<input value={scanKeyword} onChange={(e) => setScanKeyword(e.target.value)} placeholder="face closeup texture" /></label>
                    <label>タグ絞り込み<input value={tagFilter} onChange={(e) => setTagFilter(e.target.value)} placeholder="face など" /></label>
                  </div>
                  <div className="row">
                    <label>アスペクト
                      <select value={aspectFilter} onChange={(e) => setAspectFilter(e.target.value)}>
                        <option value="all">all</option>
                        <option value="portrait">portrait</option>
                        <option value="landscape">landscape</option>
                        <option value="square">square</option>
                      </select>
                    </label>
                    <label>最小幅<input type="number" value={minW} onChange={(e) => setMinW(Number(e.target.value || 0))} /></label>
                    <label>最小高さ<input type="number" value={minH} onChange={(e) => setMinH(Number(e.target.value || 0))} /></label>
                  </div>
                  <button className="btn info" onClick={runScan}>1) 候補画像を取得</button>
                  <div className="row wrap">
                    <button className="btn secondary" onClick={() => setSelectedScanIds(scanItems.map((x) => x.id))}>すべて選択</button>
                    <button className="btn secondary" onClick={() => setSelectedScanIds([])}>すべて解除</button>
                    <span className="muted">選択中: {selectedScanIds.length} 件</span>
                  </div>
                  <div className="chips">
                    {scanItems.map((i) => (
                      <button key={i.id} className={selectedScanIds.includes(i.id) ? "thumbCard active" : "thumbCard"} onClick={() => toggleScanSelection(i.id)}>
                        {i.thumbnail_url ? <img src={i.thumbnail_url} alt={i.title} /> : <div className="thumbFallback">NO IMAGE</div>}
                        <span className="thumbTitle">{i.title}</span>
                        <span className="thumbMeta">{i.width}x{i.height} / {i.aspect}</span>
                        <span className="thumbMeta">{(i.tags || []).join(", ")}</span>
                      </button>
                    ))}
                  </div>
                  <div className="row">
                    <input value={namingTemplate} onChange={(e) => setNamingTemplate(e.target.value)} placeholder="{title}_{index}" />
                    <button className="btn primary" onClick={runImport}>2) 取り込み</button>
                    <button className="btn accent" onClick={runTags}>3) タグ生成</button>
                  </div>
                  <div className="row">
                    <label>繰り返し数<input type="number" value={repeatCount} onChange={(e) => setRepeatCount(Number(e.target.value || 1))} /></label>
                    <label>フォルダ名<input value={repeatFolderTitle} onChange={(e) => setRepeatFolderTitle(e.target.value)} placeholder="character_name" /></label>
                    <button className="btn primary" onClick={prepareRepeatFolder}>4) 繰り返しフォルダ作成</button>
                  </div>
                </div>
              </>
            )}
          </section>
        )}

        {tab === "training" && (
          <section className="panel">
            <h2>学習制御</h2>
            {!selectedProject ? (
              <p className="muted">プロジェクトを選択してください。</p>
            ) : (
              <>
                <div className="card">
                  <p className="guideLine">steps/epoch は自動計算です（画像枚数 × repeats）。手動入力は不要です。</p>
                  <div className="row">
                    <label>epochs<input type="number" value={epochs} onChange={(e) => setEpochs(Number(e.target.value || 1))} /></label>
                    <label>alpha<input type="number" step="0.1" value={alpha} onChange={(e) => setAlpha(Number(e.target.value || 1))} /></label>
                    <label>repeats<input type="number" value={trainRepeats} onChange={(e) => setTrainRepeats(Number(e.target.value || 1))} /></label>
                  </div>
                  <div className="row wrap">
                    <button className="btn primary" onClick={() => startTraining(selectedProject.id)}>学習開始</button>
                    <button className="btn secondary" onClick={() => trainingAction("/training/stop-at-epoch", "epoch終了後に停止します")}>epoch区切り停止</button>
                    <button className="btn warning" onClick={() => trainingAction("/training/stop-now", "すぐ停止しました")}>すぐ停止</button>
                    <button className="btn info" onClick={() => trainingAction("/training/resume", "再開しました")}>再開</button>
                  </div>
                  <p className="muted">
                    状態: {statuses[selectedProject.id]?.status ?? "idle"} / 進捗: {statuses[selectedProject.id]?.epoch ?? 0}/
                    {statuses[selectedProject.id]?.total_epochs ?? 0} / step {statuses[selectedProject.id]?.step ?? 0}/
                    {statuses[selectedProject.id]?.steps_per_epoch ?? 0}
                  </p>
                </div>
                <div className="card">
                  <h3>軽量プレビュー履歴</h3>
                  {timeline.length === 0 ? (
                    <p className="muted">まだプレビューがありません。</p>
                  ) : (
                    <div className="timeline">
                      {timeline.map((t) => (
                        <div key={t.checkpoint_id} className="timelineItem">
                          <strong>Epoch {t.epoch}</strong>
                          <span className="muted">mark: {t.mark}</span>
                          <div className="thumbGrid4">
                            {Object.entries(t.sample_previews || {}).map(([slot, img]) => (
                              <div key={slot} className="smallThumb">
                                <img src={img} alt={`${slot}`} />
                                <span>{slot}</span>
                              </div>
                            ))}
                          </div>
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
            <h2>外部連携設定</h2>
            <div className="card">
              <div className="formGrid">
                <label>Python<input value={toolPaths.python_exe} onChange={(e) => setToolPaths({ ...toolPaths, python_exe: e.target.value })} /></label>
                <label>kohya<input value={toolPaths.kohya_root} onChange={(e) => setToolPaths({ ...toolPaths, kohya_root: e.target.value })} /></label>
                <label>ComfyUI<input value={toolPaths.comfyui_root} onChange={(e) => setToolPaths({ ...toolPaths, comfyui_root: e.target.value })} /></label>
                <label>WD14<input value={toolPaths.wd14_script} onChange={(e) => setToolPaths({ ...toolPaths, wd14_script: e.target.value })} /></label>
                <label>temp_dir<input value={toolPaths.temp_dir} onChange={(e) => setToolPaths({ ...toolPaths, temp_dir: e.target.value })} /></label>
                <label>dataset_base_dir<input value={toolPaths.dataset_base_dir} onChange={(e) => setToolPaths({ ...toolPaths, dataset_base_dir: e.target.value })} /></label>
              </div>
              <div className="row wrap">
                <button className="btn info" onClick={autoDetectToolPaths}>自動検出</button>
                <button className="btn primary" onClick={saveToolPaths}>保存</button>
                <button className="btn secondary" onClick={() => void loadIntegrations()}>再確認</button>
              </div>
            </div>
            <div className="card">
              <h3>状態</h3>
              {!integrationStatus ? (
                <p className="muted">未取得</p>
              ) : (
                <ul className="statusList">
                  {integrationRow("Python", "python_exe")}
                  {integrationRow("kohya", "kohya_root")}
                  {integrationRow("ComfyUI", "comfyui_root")}
                  {integrationRow("WD14", "wd14_script")}
                  {integrationRow("temp_dir", "temp_dir")}
                  {integrationRow("dataset_base", "dataset_base_dir")}
                </ul>
              )}
            </div>
          </section>
        )}

        {tab === "guide" && (
          <section className="panel">
            <h2>使い方ガイド</h2>
            <ol className="guideList">
              <li>外部連携設定で `dataset_base_dir` と `temp_dir` を確認する</li>
              <li>プロジェクト作成で Character/Style を選ぶ</li>
              <li>データセット作成でフィルター/タグ絞り込みして取り込み</li>
              <li>繰り返しフォルダ（例: `5_subject`）を作成する</li>
              <li>学習制御で epochs / alpha / repeats を設定して開始する</li>
            </ol>
          </section>
        )}
      </main>
    </div>
  );
}
