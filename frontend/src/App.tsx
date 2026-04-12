import { useEffect, useMemo, useState } from "react";

const API_BASE = "http://127.0.0.1:8000";

type TabId = "dashboard" | "projects" | "workflow" | "integrations" | "guide";

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

  const [scanUrl, setScanUrl] = useState<string>("https://example.com/mock-collection");
  const [scanKeyword, setScanKeyword] = useState<string>("");
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
      setError(`APIへ接続できません。start_web.bat を再実行してください。詳細: ${String(e)}`);
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
      setError(`プロジェクト作成に失敗しました: ${String(e)}`);
    }
  }

  async function startTraining(projectId: number) {
    clearMessages();
    try {
      await apiPost("/training/start", {
        project_id: projectId,
        total_epochs: epochs,
        steps_per_epoch: stepsPerEpoch
      });
      setNotice("学習を開始しました。進捗は2秒ごとに自動更新されます。");
      await refreshRuntime();
    } catch (e) {
      setError(`学習開始に失敗しました: ${String(e)}`);
    }
  }

  async function trainingAction(path: string, projectId: number, okMessage: string) {
    clearMessages();
    try {
      await apiPost(path, { project_id: projectId });
      setNotice(okMessage);
      await refreshRuntime();
    } catch (e) {
      setError(String(e));
    }
  }

  async function runScan() {
    if (!selectedProject) return;
    clearMessages();
    try {
      const result = await apiPost<{ items: ScanItem[] }>("/collector/scan", {
        project_id: selectedProject.id,
        url: scanUrl,
        keyword: scanKeyword,
        limit: 48
      });
      setScanItems(result.items);
      setSelectedScanIds(result.items.slice(0, 6).map((x) => x.id));
      setNotice(`候補画像を ${result.items.length} 件取得しました。必要な画像だけ選択してください。`);
    } catch (e) {
      setError(`画像候補の取得に失敗しました: ${String(e)}`);
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
      setNotice("選択画像をデータセットへ取り込みました。次はタグ生成を実行してください。");
      await refreshRuntime();
    } catch (e) {
      setError(`取り込みに失敗しました: ${String(e)}`);
    }
  }

  async function runTags() {
    if (!selectedProject) return;
    clearMessages();
    try {
      const res = await apiPost<{ generated_count: number }>("/tags/generate", { project_id: selectedProject.id });
      setNotice(`タグファイルを ${res.generated_count} 件生成しました。`);
      await refreshRuntime();
    } catch (e) {
      setError(`タグ生成に失敗しました: ${String(e)}`);
    }
  }

  function toggleScanSelection(id: number) {
    setSelectedScanIds((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]));
  }

  async function saveToolPaths() {
    clearMessages();
    try {
      await apiPost<ToolPaths>("/settings/tool-paths", toolPaths, "PUT");
      setNotice("連携パスを保存しました。続けて「状態確認」で接続可否を確認してください。");
      await loadIntegrations();
    } catch (e) {
      setError(`パス保存に失敗しました: ${String(e)}`);
    }
  }

  async function autoDetectToolPaths() {
    clearMessages();
    try {
      const detected = await apiPost<ToolPaths>("/settings/tool-paths/autodetect", {});
      setToolPaths(detected);
      setNotice("自動検出を実行しました。必要に応じて手入力で修正してください。");
      await loadIntegrations();
    } catch (e) {
      setError(`自動検出に失敗しました: ${String(e)}`);
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
          <button className={tab === "workflow" ? "menuBtn active" : "menuBtn"} onClick={() => setTab("workflow")}>
            制作ワークフロー
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
            <p className="guideLine">まずは「外部連携設定」でパス確認後、「プロジェクト管理」へ進んでください。</p>
            <div className="statsGrid">
              <div className="statBox">
                <div className="statLabel">登録プロジェクト数</div>
                <div className="statValue">{projects.length}</div>
              </div>
              <div className="statBox">
                <div className="statLabel">学習中ジョブ数</div>
                <div className="statValue">{Object.values(statuses).filter((s) => s.status === "training").length}</div>
              </div>
              <div className="statBox">
                <div className="statLabel">連携チェックOK</div>
                <div className="statValue">
                  {integrationStatus ? Object.values(integrationStatus.checks).filter((x) => x.ok).length : 0}/4
                </div>
              </div>
            </div>
          </section>
        )}

        {tab === "projects" && (
          <section className="panel">
            <h2>プロジェクト管理</h2>
            <p className="guideLine">手順1: 新しいプロジェクト名を入力して作成し、下の一覧で対象を選択します。</p>
            <div className="row">
              <input
                value={newProjectName}
                onChange={(e) => setNewProjectName(e.target.value)}
                placeholder="例: キャラ名_衣装A_v1"
              />
              <select value={newProjectType} onChange={(e) => setNewProjectType(e.target.value as "character" | "style")}>
                <option value="character">Character LoRA</option>
                <option value="style">Style LoRA</option>
              </select>
              <button className="btn primary" onClick={createProject}>
                プロジェクト作成
              </button>
            </div>

            <div className="projectTable">
              {projects.length === 0 && <p className="muted">まだプロジェクトがありません。</p>}
              {projects.map((p) => (
                <button
                  key={p.id}
                  className={selectedProjectId === p.id ? "projectRow selected" : "projectRow"}
                  onClick={() => setSelectedProjectId(p.id)}
                >
                  <span>{p.name}</span>
                  <span>
                    {p.project_type} / {statuses[p.id]?.status ?? "idle"}
                  </span>
                </button>
              ))}
            </div>
          </section>
        )}

        {tab === "workflow" && (
          <section className="panel">
            <h2>制作ワークフロー</h2>
            {!selectedProject ? (
              <p className="muted">先に「プロジェクト管理」から対象プロジェクトを選択してください。</p>
            ) : (
              <>
                <p className="guideLine">
                  手順2: 画像収集→取り込み→タグ生成→学習開始の順に進めると迷いません。現在の対象:{" "}
                  <strong>{selectedProject.name}</strong>
                </p>
                <div className="workflowGrid">
                  <div className="card">
                    <h3>データセット作成</h3>
                    <label>
                      収集元URL
                      <input value={scanUrl} onChange={(e) => setScanUrl(e.target.value)} placeholder="https://..." />
                    </label>
                    <label>
                      キーワード（任意）
                      <input value={scanKeyword} onChange={(e) => setScanKeyword(e.target.value)} placeholder="例: face, closeup, texture" />
                    </label>
                    <button className="btn info" onClick={runScan}>
                      1) 候補画像を取得
                    </button>
                    <p className="muted">取得後、必要な画像だけ選択してください（青が選択中）。</p>
                    <div className="row wrap">
                      <button className="btn secondary" onClick={() => setSelectedScanIds(scanItems.map((x) => x.id))}>
                        すべて選択
                      </button>
                      <button className="btn secondary" onClick={() => setSelectedScanIds([])}>
                        すべて解除
                      </button>
                      <span className="muted">選択中: {selectedScanIds.length} 件</span>
                    </div>
                    <div className="chips">
                      {scanItems.map((i) => (
                        <button key={i.id} className={selectedScanIds.includes(i.id) ? "thumbCard active" : "thumbCard"} onClick={() => toggleScanSelection(i.id)}>
                          {i.thumbnail_url ? <img src={i.thumbnail_url} alt={i.title} /> : <div className="thumbFallback">NO IMAGE</div>}
                          <span className="thumbTitle">{i.title}</span>
                          <span className="thumbMeta">
                            {i.width}x{i.height} / {i.aspect}
                          </span>
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
                        2) 選択画像を取り込み
                      </button>
                    </div>
                    <button className="btn accent" onClick={runTags}>
                      3) タグを生成
                    </button>
                  </div>

                  <div className="card">
                    <h3>学習制御</h3>
                    <p className="muted">手順3: 設定して学習開始。停止は「即時」か「epoch区切り」を選べます。</p>
                    <div className="row">
                      <label>
                        epoch数
                        <input type="number" value={epochs} onChange={(e) => setEpochs(Number(e.target.value || 1))} />
                      </label>
                      <label>
                        1epochあたりstep
                        <input
                          type="number"
                          value={stepsPerEpoch}
                          onChange={(e) => setStepsPerEpoch(Number(e.target.value || 1))}
                        />
                      </label>
                    </div>
                    <div className="row wrap">
                      <button className="btn primary" onClick={() => startTraining(selectedProject.id)}>
                        学習開始
                      </button>
                      <button
                        className="btn secondary"
                        onClick={() => trainingAction("/training/stop-at-epoch", selectedProject.id, "現在epoch終了後に停止します。")}
                      >
                        epoch区切りで停止
                      </button>
                      <button
                        className="btn warning"
                        onClick={() => trainingAction("/training/stop-now", selectedProject.id, "学習を停止しました。")}
                      >
                        すぐ停止
                      </button>
                      <button
                        className="btn info"
                        onClick={() => trainingAction("/training/resume", selectedProject.id, "学習を再開しました。")}
                      >
                        再開
                      </button>
                    </div>
                    <p className="muted">
                      状態: {statuses[selectedProject.id]?.status ?? "idle"} / 進捗:{" "}
                      {statuses[selectedProject.id]?.epoch ?? 0}/{statuses[selectedProject.id]?.total_epochs ?? 0} epoch
                    </p>
                  </div>
                </div>

                <div className="card">
                  <h3>プレビュー履歴</h3>
                  <p className="muted">手順4: epochごとの差分確認に使います。気になるepochをメモしてください。</p>
                  {timeline.length === 0 ? (
                    <p className="muted">まだプレビューはありません（学習が1epoch進むと表示されます）。</p>
                  ) : (
                    <div className="timeline">
                      {timeline.map((t) => (
                        <div key={t.checkpoint_id} className="timelineItem">
                          <strong>Epoch {t.epoch}</strong>
                          <span className="muted">マーク: {t.mark}</span>
                          <span className="muted">サンプル: {Object.keys(t.samples).join(", ")}</span>
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
            <p className="guideLine">手順0: 最初にここを設定。自動検出 → 保存 → 状態確認 の順で実施してください。</p>
            <div className="card">
              <div className="formGrid">
                <label>
                  Python実行ファイル
                  <input
                    value={toolPaths.python_exe}
                    onChange={(e) => setToolPaths({ ...toolPaths, python_exe: e.target.value })}
                    placeholder="C:\\Python310\\python.exe"
                  />
                </label>
                <label>
                  kohyaディレクトリ
                  <input
                    value={toolPaths.kohya_root}
                    onChange={(e) => setToolPaths({ ...toolPaths, kohya_root: e.target.value })}
                    placeholder="D:\\tools\\kohya_ss"
                  />
                </label>
                <label>
                  ComfyUIディレクトリ
                  <input
                    value={toolPaths.comfyui_root}
                    onChange={(e) => setToolPaths({ ...toolPaths, comfyui_root: e.target.value })}
                    placeholder="D:\\ComfyUI"
                  />
                </label>
                <label>
                  WD14スクリプト
                  <input
                    value={toolPaths.wd14_script}
                    onChange={(e) => setToolPaths({ ...toolPaths, wd14_script: e.target.value })}
                    placeholder="...\\tag_images_by_wd14_tagger.py"
                  />
                </label>
                <label>
                  一時保存ディレクトリ
                  <input
                    value={toolPaths.temp_dir}
                    onChange={(e) => setToolPaths({ ...toolPaths, temp_dir: e.target.value })}
                    placeholder="C:\\...\\.runtime\\tmp"
                  />
                </label>
                <label>
                  データセット補完ベース
                  <input
                    value={toolPaths.dataset_base_dir}
                    onChange={(e) => setToolPaths({ ...toolPaths, dataset_base_dir: e.target.value })}
                    placeholder="C:\\ポートフォリオ\\SDXL\\LoRA_Traning\\dataset"
                  />
                </label>
              </div>
              <div className="row wrap">
                <button className="btn info" onClick={autoDetectToolPaths}>
                  自動検出
                </button>
                <button className="btn primary" onClick={saveToolPaths}>
                  パスを保存
                </button>
                <button className="btn secondary" onClick={() => void loadIntegrations()}>
                  状態確認
                </button>
              </div>
            </div>
            <div className="card">
              <h3>接続ステータス</h3>
              {!integrationStatus ? (
                <p className="muted">まだ確認できていません。</p>
              ) : (
                <ul className="statusList">
                  {integrationRow("Python", "python_exe")}
                  {integrationRow("kohya", "kohya_root")}
                  {integrationRow("ComfyUI", "comfyui_root")}
                  {integrationRow("WD14", "wd14_script")}
                  {integrationRow("TempDir", "temp_dir")}
                  {integrationRow("DatasetBase", "dataset_base_dir")}
                </ul>
              )}
            </div>
          </section>
        )}

        {tab === "guide" && (
          <section className="panel">
            <h2>使い方ガイド（画面内版）</h2>
            <ol className="guideList">
              <li>まず `外部連携設定` で自動検出し、保存後に状態確認します。</li>
              <li>`プロジェクト管理` で新規プロジェクトを作成し、対象を選択します。</li>
              <li>`制作ワークフロー` で候補取得→取り込み→タグ生成を順に実行します。</li>
              <li>学習設定を入力して `学習開始`。必要に応じて停止/再開します。</li>
              <li>プレビュー履歴でepoch差分を確認し、良いcheckpointを判断します。</li>
            </ol>
            <p className="muted">詳細版はリポジトリの「使い方ガイド」ファイルを参照してください。</p>
          </section>
        )}
      </main>
    </div>
  );
}
