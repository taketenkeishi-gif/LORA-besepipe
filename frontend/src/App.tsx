import { DragEvent, useEffect, useMemo, useState } from "react";

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
  total_steps?: number;
  done_steps?: number;
  progress_percent?: number;
  eta_seconds?: number | null;
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

type PreviewPrompts = {
  positive_prompt: string;
  negative_prompt: string;
};

type ApiCapabilities = {
  dropFiles: boolean;
  dropUrl: boolean;
};

export default function App() {
  const [tab, setTab] = useState<TabId>("dashboard");
  const [apiHealth, setApiHealth] = useState<string>("確認中");
  const [error, setError] = useState<string>("");
  const [notice, setNotice] = useState<string>("");
  const [dragActive, setDragActive] = useState<boolean>(false);
  const [apiCaps, setApiCaps] = useState<ApiCapabilities>({ dropFiles: true, dropUrl: true });

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
  const [expandedTagCardId, setExpandedTagCardId] = useState<number | null>(null);
  const [namingTemplate, setNamingTemplate] = useState<string>("{title}_{index}");

  const [repeatFolderTitle, setRepeatFolderTitle] = useState<string>("subject");
  const [repeatCount, setRepeatCount] = useState<number>(5);

  const [timeline, setTimeline] = useState<PreviewTimelineItem[]>([]);
  const [epochs, setEpochs] = useState<number>(5);
  const [alpha, setAlpha] = useState<number>(4);
  const [rank, setRank] = useState<number>(16);
  const [saveEvery, setSaveEvery] = useState<number>(1);
  const [trainRepeats, setTrainRepeats] = useState<number>(5);
  const [outputName, setOutputName] = useState<string>("lora_output");
  const [baseCkpt, setBaseCkpt] = useState<string>("");
  const [trainDir, setTrainDir] = useState<string>("");
  const [regDir, setRegDir] = useState<string>("");
  const [resolution, setResolution] = useState<number>(512);

  const [prompts, setPrompts] = useState<PreviewPrompts>({
    positive_prompt: "",
    negative_prompt: ""
  });

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

  useEffect(() => {
    if (!selectedProject) return;
    if (!trainDir) setTrainDir(selectedProject.dataset_dir);
    if (!regDir) setRegDir(selectedProject.captions_dir);
  }, [selectedProject, trainDir, regDir]);

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
    await Promise.all([refreshRuntime(), loadIntegrations(), loadPreviewPrompts(), loadApiCapabilities()]);
  }

  async function loadApiCapabilities() {
    try {
      const res = await fetch(`${API_BASE}/openapi.json`);
      if (!res.ok) return;
      const doc = (await res.json()) as { paths?: Record<string, unknown> };
      const paths = doc.paths ?? {};
      setApiCaps({
        dropFiles: Boolean(paths["/collector/drop-files"] || paths["/collector/drop_files"]),
        dropUrl: Boolean(paths["/collector/drop-url"] || paths["/collector/drop_url"])
      });
    } catch {
      // ignore
    }
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
      if (selectedProjectId !== null) {
        await loadProjectRuntime(selectedProjectId);
      }
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

  async function loadPreviewPrompts() {
    try {
      const p = await apiGet<PreviewPrompts>("/settings/preview-prompts");
      setPrompts(p);
    } catch {
      // ignore
    }
  }

  async function savePreviewPrompts() {
    clearMessages();
    try {
      await apiPost<PreviewPrompts>("/settings/preview-prompts", prompts, "PUT");
      setNotice("軽量プレビュー用プロンプトを保存しました。");
    } catch (e) {
      setError(String(e));
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
      setExpandedTagCardId(null);
      setSelectedScanIds(result.items.slice(0, 12).map((x) => x.id));
      setNotice(`候補画像 ${result.detected} 件を取得 (${result.mode})`);
    } catch (e) {
      setError(`取得失敗: ${String(e)}`);
    }
  }

  async function onDropFiles(ev: DragEvent<HTMLDivElement>) {
    if (!selectedProject) return;
    ev.preventDefault();
    setDragActive(false);
    clearMessages();
    const files = Array.from(ev.dataTransfer.files).filter((f) => {
      if (f.type?.startsWith("image/")) return true;
      return /\.(png|jpe?g|webp|bmp|gif)$/i.test(f.name);
    });
    if (files.length === 0) {
      const maybeUrl = extractDroppedUrl(ev);
      if (maybeUrl) {
        await dropUrlAsCandidate(selectedProject.id, maybeUrl.trim());
      } else {
        setNotice("画像ファイルまたは画像URLをドロップしてください。");
      }
      return;
    }
    if (!apiCaps.dropFiles) {
      setError("このバックエンドにはD&Dファイル追加APIがありません。`start_web.bat` で再起動して最新版を起動してください。");
      return;
    }
    try {
      const form = new FormData();
      form.append("project_id", String(selectedProject.id));
      for (const f of files) form.append("files", f);
      let res = await fetch(`${API_BASE}/collector/drop-files`, { method: "POST", body: form });
      if (res.status === 404) {
        // 互換ルート
        res = await fetch(`${API_BASE}/collector/drop_files`, { method: "POST", body: form });
      }
      const json = await res.json();
      if (!res.ok) throw new Error(json?.detail ?? "drop failed");
      setScanItems(json.items || []);
      setExpandedTagCardId(null);
      setNotice(`${json.added_count ?? files.length} 件を候補に追加しました。`);
    } catch (e) {
      setError(`ドラッグ&ドロップ追加失敗: ${String(e)}`);
    }
  }

  function extractDroppedUrl(ev: DragEvent<HTMLDivElement>): string {
    const uriList = ev.dataTransfer.getData("text/uri-list");
    if (uriList?.trim()) return uriList.trim().split("\n")[0].trim();

    const plain = ev.dataTransfer.getData("text/plain");
    if (plain?.trim().startsWith("http")) return plain.trim();

    const html = ev.dataTransfer.getData("text/html");
    if (html?.trim()) {
      const m = html.match(/https?:\/\/[^"' >]+/i);
      if (m?.[0]) return m[0];
    }
    return "";
  }

  async function dropUrlAsCandidate(projectId: number, url: string) {
    if (!apiCaps.dropUrl) {
      setScanUrl(url);
      setNotice("URLを収集欄に反映しました。`候補画像取得` を押してください。");
      return;
    }
    try {
      let json: { items: ScanItem[]; added_count: number } | null = null;
      try {
        json = await apiPost<{ items: ScanItem[]; added_count: number }>("/collector/drop-url", { project_id: projectId, url });
      } catch {
        json = await apiPost<{ items: ScanItem[]; added_count: number }>("/collector/drop_url", { project_id: projectId, url });
      }
      setScanItems(json.items || []);
      setExpandedTagCardId(null);
      setNotice("URL画像を候補に追加しました。");
    } catch (e) {
      // ページURLなどで直接追加できない場合はscanにフォールバック
      setScanUrl(url);
      try {
        const result = await apiPost<{ items: ScanItem[]; detected: number; mode: string }>("/collector/scan", {
          project_id: projectId,
          url,
          keyword: buildKeyword(),
          limit: 60
        });
        setScanItems(result.items);
        setExpandedTagCardId(null);
        setSelectedScanIds(result.items.slice(0, 12).map((x) => x.id));
        setNotice(`URLドロップを収集にフォールバック: ${result.detected} 件 (${result.mode})`);
      } catch (inner) {
        setNotice("URLは収集欄に反映しました。");
        setError(`URL追加失敗: ${String(e)} / scan失敗: ${String(inner)}`);
      }
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
      setNotice("取り込み完了。");
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
      setNotice(`繰り返しフォルダ作成: ${res.folder} (${res.copied_images}枚)`);
    } catch (e) {
      setError(`失敗: ${String(e)}`);
    }
  }

  async function runTags() {
    if (!selectedProject) return;
    clearMessages();
    try {
      const res = await apiPost<{ generated_count: number }>("/tags/generate", { project_id: selectedProject.id });
      setNotice(`タグ生成完了: ${res.generated_count} 件`);
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
        alpha,
        rank,
        save_every_n_epochs: saveEvery,
        output_name: outputName,
        base_checkpoint_path: baseCkpt,
        train_data_dir: trainDir,
        reg_data_dir: regDir,
        resolution
      });
      setNotice(`学習開始。step/epoch自動計算: ${res.steps_per_epoch}`);
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

  function etaText(sec?: number | null) {
    if (sec == null) return "計算中";
    const h = Math.floor(sec / 3600);
    const m = Math.floor((sec % 3600) / 60);
    return `${h}h ${m}m`;
  }

  const currentStatus = selectedProject ? statuses[selectedProject.id] : undefined;

  return (
    <div className="layout">
      <aside className="sidebar">
        <h1 className="logo">LoRA制作ワークベンチ</h1>
        <nav className="menu">
          {[
            ["dashboard", "ダッシュボード"],
            ["projects", "プロジェクト管理"],
            ["dataset", "データセット作成"],
            ["training", "学習制御"],
            ["integrations", "外部連携設定"],
            ["guide", "使い方ガイド"]
          ].map(([id, label]) => (
            <button key={id} className={tab === id ? "menuBtn active" : "menuBtn"} onClick={() => setTab(id as TabId)}>
              {label}
            </button>
          ))}
        </nav>
        <div className="healthCard">
          <div>API状態: {apiHealth}</div>
          <button className="btn secondary small" onClick={() => void refreshAll()}>
            更新
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
                <div className="statLabel">学習中</div>
                <div className="statValue">{Object.values(statuses).filter((s) => s.status === "training").length}</div>
              </div>
              <div className="statBox">
                <div className="statLabel">連携OK</div>
                <div className="statValue">{integrationStatus ? Object.values(integrationStatus.checks).filter((x) => x.ok).length : 0}</div>
              </div>
            </div>
          </section>
        )}

        {tab === "projects" && (
          <section className="panel">
            <h2>プロジェクト管理</h2>
            <div className="row">
              <input value={newProjectName} onChange={(e) => setNewProjectName(e.target.value)} placeholder="project_name" />
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
                  <span>{p.project_type} / {statuses[p.id]?.status ?? "idle"}</span>
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
              <div className="card">
                <label>収集元URL<input value={scanUrl} onChange={(e) => setScanUrl(e.target.value)} /></label>
                <div className="row">
                  <label>キーワード<input value={scanKeyword} onChange={(e) => setScanKeyword(e.target.value)} /></label>
                  <label>タグ絞り込み<input value={tagFilter} onChange={(e) => setTagFilter(e.target.value)} /></label>
                </div>
                <div className="row">
                  <label>Aspect
                    <select value={aspectFilter} onChange={(e) => setAspectFilter(e.target.value)}>
                      <option value="all">all</option>
                      <option value="portrait">portrait</option>
                      <option value="landscape">landscape</option>
                      <option value="square">square</option>
                    </select>
                  </label>
                  <label>minW<input type="number" value={minW} onChange={(e) => setMinW(Number(e.target.value || 0))} /></label>
                  <label>minH<input type="number" value={minH} onChange={(e) => setMinH(Number(e.target.value || 0))} /></label>
                </div>
                <div className="ctaRow">
                  <button className="btn cta info" onClick={runScan}>1) 候補画像取得</button>
                  <button className="btn cta primary" onClick={runImport}>2) 取り込み</button>
                  <button className="btn cta accent" onClick={runTags}>3) タグ生成</button>
                  <button className="btn cta secondary" onClick={() => selectedProject && dropUrlAsCandidate(selectedProject.id, scanUrl)}>URLを候補追加</button>
                </div>
                <div
                  className={dragActive ? "dropZone active" : "dropZone"}
                  onDrop={onDropFiles}
                  onDragEnter={(e) => {
                    e.preventDefault();
                    setDragActive(true);
                  }}
                  onDragLeave={() => setDragActive(false)}
                  onDragOver={(e) => {
                    e.preventDefault();
                    setDragActive(true);
                  }}
                >
                  ここに画像ファイルをドラッグ&ドロップ。Webは画像URLのドロップにも対応。
                </div>
                {!apiCaps.dropFiles && <p className="errorInline">D&D APIが未対応のバックエンドです。`start_web.bat` で再起動してください。</p>}
                <div className="row wrap">
                  <button className="btn secondary" onClick={() => setSelectedScanIds(scanItems.map((x) => x.id))}>すべて選択</button>
                  <button className="btn secondary" onClick={() => setSelectedScanIds([])}>すべて解除</button>
                  <span className="muted">選択中: {selectedScanIds.length} 件</span>
                </div>
                <div className="chips">
                  {scanItems.map((i) => (
                    <button
                      key={i.id}
                      className={selectedScanIds.includes(i.id) ? "thumbCard active" : "thumbCard"}
                      onClick={() => toggleScanSelection(i.id)}
                      onDoubleClick={() => setExpandedTagCardId((prev) => (prev === i.id ? null : i.id))}
                    >
                      <div className="thumbViewport">
                        {i.thumbnail_url ? <img src={i.thumbnail_url} alt={i.title} onError={(e) => {
                          e.currentTarget.onerror = null;
                          e.currentTarget.src = "data:image/svg+xml;utf8,%3Csvg xmlns='http://www.w3.org/2000/svg' width='320' height='200'%3E%3Crect width='100%25' height='100%25' fill='%23dbe6f4'/%3E%3Ctext x='16' y='104' fill='%234c5f7a' font-size='14'%3Epreview unavailable%3C/text%3E%3C/svg%3E";
                        }} /> : <div className="thumbFallback">NO IMAGE</div>}
                      </div>
                      <span className="thumbTitle">{i.title}</span>
                      <span className="thumbMeta">{i.width}x{i.height} / {i.aspect}</span>
                      <span className="thumbMeta">ダブルクリックでタグ表示</span>
                      {expandedTagCardId === i.id && (
                        <div className="tagFlow">
                          {(i.tags || []).map((t) => (
                            <span key={`${i.id}-${t}`} className="tagChip">{t}</span>
                          ))}
                        </div>
                      )}
                    </button>
                  ))}
                </div>
                <div className="row">
                  <input value={namingTemplate} onChange={(e) => setNamingTemplate(e.target.value)} placeholder="{title}_{index}" />
                </div>
                <div className="row">
                  <label>繰り返し数<input type="number" value={repeatCount} onChange={(e) => setRepeatCount(Number(e.target.value || 1))} /></label>
                  <label>フォルダ名<input value={repeatFolderTitle} onChange={(e) => setRepeatFolderTitle(e.target.value)} /></label>
                  <button className="btn primary" onClick={prepareRepeatFolder}>4) {`<repeats>_<title>`} 作成</button>
                </div>
              </div>
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
                  <div className="trainingSummary">
                    <div className="metricBox">
                      <div className="metricLabel">状態</div>
                      <div className="metricValue">{currentStatus?.status ?? "idle"}</div>
                    </div>
                    <div className="metricBox">
                      <div className="metricLabel">Epoch</div>
                      <div className="metricValue">{currentStatus?.epoch ?? 0} / {currentStatus?.total_epochs ?? 0}</div>
                    </div>
                    <div className="metricBox">
                      <div className="metricLabel">Step</div>
                      <div className="metricValue">{currentStatus?.done_steps ?? 0} / {currentStatus?.total_steps ?? 0}</div>
                    </div>
                    <div className="metricBox">
                      <div className="metricLabel">残り目安</div>
                      <div className="metricValue">{etaText(currentStatus?.eta_seconds)}</div>
                    </div>
                  </div>
                  <div className="progressBar big">
                    <div className="progressFill" style={{ width: `${Math.min(100, Math.max(0, currentStatus?.progress_percent ?? 0))}%` }} />
                  </div>
                  <p className="muted">進捗: {currentStatus?.progress_percent ?? 0}%</p>
                  <div className="row wrap">
                    <label>epochs<input type="number" value={epochs} onChange={(e) => setEpochs(Number(e.target.value || 1))} /></label>
                    <label>rank<input type="number" value={rank} onChange={(e) => setRank(Number(e.target.value || 1))} /></label>
                    <label>alpha<input type="number" step="0.1" value={alpha} onChange={(e) => setAlpha(Number(e.target.value || 1))} /></label>
                    <label>repeats<input type="number" value={trainRepeats} onChange={(e) => setTrainRepeats(Number(e.target.value || 1))} /></label>
                    <label>保存間隔(epoch)<input type="number" value={saveEvery} onChange={(e) => setSaveEvery(Number(e.target.value || 1))} /></label>
                    <label>解像度<input type="number" value={resolution} onChange={(e) => setResolution(Number(e.target.value || 512))} /></label>
                  </div>
                  <div className="row wrap">
                    <label>出力名<input value={outputName} onChange={(e) => setOutputName(e.target.value)} /></label>
                    <label>学習元Checkpoint<input value={baseCkpt} onChange={(e) => setBaseCkpt(e.target.value)} /></label>
                  </div>
                  <div className="row wrap">
                    <label>教師画像フォルダ<input value={trainDir} onChange={(e) => setTrainDir(e.target.value)} /></label>
                    <label>正則化画像フォルダ<input value={regDir} onChange={(e) => setRegDir(e.target.value)} /></label>
                  </div>
                  <div className="row wrap">
                    <label>プレビュー Positive<input value={prompts.positive_prompt} onChange={(e) => setPrompts({ ...prompts, positive_prompt: e.target.value })} /></label>
                    <label>プレビュー Negative<input value={prompts.negative_prompt} onChange={(e) => setPrompts({ ...prompts, negative_prompt: e.target.value })} /></label>
                    <button className="btn secondary" onClick={savePreviewPrompts}>プロンプト保存</button>
                  </div>
                  <div className="row wrap">
                    <button className="btn cta xl primary" onClick={() => startTraining(selectedProject.id)}>学習開始</button>
                    <button className="btn cta secondary" onClick={() => trainingAction("/training/stop-at-epoch", "epoch区切り停止予約")}>epoch区切り停止</button>
                    <button className="btn cta warning" onClick={() => trainingAction("/training/stop-now", "すぐ停止")}>すぐ停止</button>
                    <button className="btn cta info" onClick={() => trainingAction("/training/resume", "再開")}>再開</button>
                  </div>
                </div>
                <div className="card">
                  <h3>軽量プレビュー履歴（ライブ）</h3>
                  {timeline.length === 0 ? (
                    <p className="muted">まだプレビューがありません。</p>
                  ) : (
                    <div className="timeline">
                      {timeline.map((t) => (
                        <div key={t.checkpoint_id} className="timelineItem">
                          <strong>Epoch {t.epoch}</strong>
                          <div className="thumbGrid4">
                            {Object.entries(t.sample_previews || {}).map(([slot, img]) => (
                              <div key={slot} className="smallThumb">
                                <img src={img} alt={slot} />
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
              {!integrationStatus ? <p className="muted">未取得</p> : <ul className="statusList">{integrationRow("Python", "python_exe")}{integrationRow("kohya", "kohya_root")}{integrationRow("ComfyUI", "comfyui_root")}{integrationRow("WD14", "wd14_script")}{integrationRow("temp_dir", "temp_dir")}{integrationRow("dataset_base", "dataset_base_dir")}</ul>}
            </div>
          </section>
        )}

        {tab === "guide" && (
          <section className="panel">
            <h2>使い方ガイド</h2>
            <ol className="guideList">
              <li>外部連携設定で各パスを確認</li>
              <li>データセット作成ページで候補収集・D&D・絞り込み・取り込み</li>
              <li>繰り返しフォルダ（例: 5_subject）を作成</li>
              <li>学習制御ページでパラメータ設定・開始・ライブ監視</li>
            </ol>
          </section>
        )}
      </main>
    </div>
  );
}
