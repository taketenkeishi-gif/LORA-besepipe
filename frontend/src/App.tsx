import { DragEvent, useEffect, useMemo, useState } from "react";

const API_BASE = "http://127.0.0.1:8000";

type TabId = "dashboard" | "projects" | "dataset" | "training" | "integrations" | "guide";

type Project = { id: number; name: string; project_type: "character" | "style"; status: string; dataset_dir: string; captions_dir: string };
type TrainingStatus = { status: string; epoch: number; total_epochs: number; total_steps?: number; done_steps?: number; progress_percent?: number; eta_seconds?: number | null };
type ScanItem = { id: number; title: string; width: number; height: number; aspect: string; thumbnail_url?: string; tags?: string[] };
type ToolPaths = { python_exe: string; kohya_root: string; comfyui_root: string; wd14_script: string; temp_dir: string; dataset_base_dir: string };
type IntegrationStatus = { checks: Record<string, { ok: boolean; reason: string }> };
type PreviewTimelineItem = { checkpoint_id: number; epoch: number; sample_previews?: Record<string, string> };
type PreviewPrompts = { positive_prompt: string; negative_prompt: string };
type DatasetPreview = { image_path: string | null; thumbnail_url: string | null };

export default function App() {
  const [tab, setTab] = useState<TabId>("dataset");
  const [apiHealth, setApiHealth] = useState("確認中");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [dragActive, setDragActive] = useState(false);

  const [projects, setProjects] = useState<Project[]>([]);
  const [selectedProjectId, setSelectedProjectId] = useState<number | null>(null);
  const [statuses, setStatuses] = useState<Record<number, TrainingStatus>>({});
  const [newProjectName, setNewProjectName] = useState("");
  const [newProjectType, setNewProjectType] = useState<"character" | "style">("character");

  const [scanUrl, setScanUrl] = useState("https://www.pixiv.net/artworks/143382932");
  const [scanKeyword, setScanKeyword] = useState("");
  const [tagFilter, setTagFilter] = useState("");
  const [aspectFilter, setAspectFilter] = useState("all");
  const [minW, setMinW] = useState(0);
  const [minH, setMinH] = useState(0);
  const [scanItems, setScanItems] = useState<ScanItem[]>([]);
  const [selectedScanIds, setSelectedScanIds] = useState<number[]>([]);
  const [expandedTagCardId, setExpandedTagCardId] = useState<number | null>(null);
  const [namingTemplate, setNamingTemplate] = useState("{title}_{index}");
  const [repeatCount, setRepeatCount] = useState(5);
  const [repeatFolderTitle, setRepeatFolderTitle] = useState("subject");

  const [epochs, setEpochs] = useState(5);
  const [rank, setRank] = useState(16);
  const [alpha, setAlpha] = useState(4);
  const [trainRepeats, setTrainRepeats] = useState(5);
  const [saveEvery, setSaveEvery] = useState(1);
  const [resolution, setResolution] = useState(512);
  const [outputName, setOutputName] = useState("lora_output");
  const [baseCkpt, setBaseCkpt] = useState("");
  const [trainDir, setTrainDir] = useState("");
  const [regDir, setRegDir] = useState("");
  const [timeline, setTimeline] = useState<PreviewTimelineItem[]>([]);
  const [prompts, setPrompts] = useState<PreviewPrompts>({ positive_prompt: "", negative_prompt: "" });
  const [datasetPreview, setDatasetPreview] = useState<DatasetPreview | null>(null);

  const [toolPaths, setToolPaths] = useState<ToolPaths>({ python_exe: "", kohya_root: "", comfyui_root: "", wd14_script: "", temp_dir: "", dataset_base_dir: "" });
  const [integrationStatus, setIntegrationStatus] = useState<IntegrationStatus | null>(null);

  const selectedProject = useMemo(() => projects.find((p) => p.id === selectedProjectId) ?? null, [projects, selectedProjectId]);
  const currentStatus = selectedProject ? statuses[selectedProject.id] : undefined;

  const clearMessages = () => { setError(""); setNotice(""); };

  async function apiGet<T>(path: string): Promise<T> { const res = await fetch(`${API_BASE}${path}`); if (!res.ok) throw new Error(path); return await res.json(); }
  async function apiPost<T>(path: string, body: unknown, method: "POST" | "PUT" | "PATCH" = "POST"): Promise<T> {
    const res = await fetch(`${API_BASE}${path}`, { method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const json = await res.json(); if (!res.ok) throw new Error(json?.detail ?? path); return json;
  }

  async function refreshRuntime() {
    try {
      const health = await apiGet<{ status: string }>("/health");
      setApiHealth(health.status === "ok" ? "接続OK" : health.status);
      const list = await apiGet<Project[]>("/projects");
      setProjects(list);
      const entries = await Promise.all(list.map(async (p) => [p.id, await apiGet<TrainingStatus>(`/training/status?project_id=${p.id}`)] as const));
      setStatuses(Object.fromEntries(entries));
      if (selectedProjectId === null && list.length > 0) setSelectedProjectId(list[0].id);
    } catch (e) { setApiHealth("offline"); setError(`API接続エラー: ${String(e)}`); }
  }

  async function refreshAll() { await Promise.all([refreshRuntime(), loadIntegrations(), loadPreviewPrompts()]); }
  async function loadIntegrations() { try { setToolPaths(await apiGet<ToolPaths>("/settings/tool-paths")); setIntegrationStatus(await apiGet<IntegrationStatus>("/settings/integrations/status")); } catch {} }
  async function loadPreviewPrompts() { try { setPrompts(await apiGet<PreviewPrompts>("/settings/preview-prompts")); } catch {} }

  useEffect(() => {
    void refreshAll();
    const timer = setInterval(() => void refreshRuntime(), 2000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => { if (selectedProject) { if (!trainDir) setTrainDir(selectedProject.dataset_dir); if (!regDir) setRegDir(selectedProject.captions_dir); } }, [selectedProject]);
  useEffect(() => { if (selectedProject) void loadDatasetPreview(selectedProject.id, trainDir || selectedProject.dataset_dir); }, [selectedProject, trainDir]);

  function buildKeyword() {
    const terms: string[] = [];
    if (scanKeyword.trim()) terms.push(scanKeyword.trim());
    if (tagFilter.trim()) terms.push(`tag:${tagFilter.trim()}`);
    if (aspectFilter !== "all") terms.push(`aspect:${aspectFilter}`);
    if (minW > 0) terms.push(`minw:${minW}`);
    if (minH > 0) terms.push(`minh:${minH}`);
    return terms.join(" ");
  }
  async function createProject() {
    if (!newProjectName.trim()) return;
    clearMessages();
    try { const p = await apiPost<Project>("/projects", { name: newProjectName.trim(), project_type: newProjectType }); setSelectedProjectId(p.id); setNewProjectName(""); setNotice(`プロジェクト作成: ${p.name}`); await refreshRuntime(); }
    catch (e) { setError(`作成失敗: ${String(e)}`); }
  }
  async function updateProjectType(projectId: number, projectType: "character" | "style") {
    try { await apiPost<Project>(`/projects/${projectId}`, { project_type: projectType }, "PATCH"); await refreshRuntime(); setNotice(`種別を ${projectType} に変更`); }
    catch (e) { setError(`種別変更失敗: ${String(e)}`); }
  }
  async function duplicateProject(projectId: number, srcName: string) {
    const name = `${srcName}_copy`;
    try { const p = await apiPost<Project>(`/projects/${projectId}/duplicate`, { name }); await refreshRuntime(); setSelectedProjectId(p.id); setNotice(`複製作成: ${name}`); }
    catch (e) { setError(`複製失敗: ${String(e)}`); }
  }
  async function deleteProject(projectId: number) {
    try {
      const res = await fetch(`${API_BASE}/projects/${projectId}`, { method: "DELETE" });
      if (!res.ok) throw new Error("delete failed");
      await refreshRuntime();
      if (selectedProjectId === projectId) setSelectedProjectId(null);
      setNotice("プロジェクト削除完了");
    }
    catch (e) { setError(`削除失敗: ${String(e)}`); }
  }

  async function runScan() {
    if (!selectedProject) return;
    clearMessages();
    try {
      const r = await apiPost<{ items: ScanItem[]; detected: number; mode: string }>("/collector/scan", { project_id: selectedProject.id, url: scanUrl, keyword: buildKeyword(), limit: 60 });
      setScanItems(r.items); setSelectedScanIds(r.items.slice(0, 12).map((x) => x.id)); setExpandedTagCardId(null); setNotice(`候補取得: ${r.detected}件 (${r.mode})`);
    } catch (e) { setError(`取得失敗: ${String(e)}`); }
  }

  function extractDroppedUrl(ev: DragEvent<HTMLDivElement>) {
    const uri = ev.dataTransfer.getData("text/uri-list"); if (uri?.trim()) return uri.trim().split("\n")[0].trim();
    const plain = ev.dataTransfer.getData("text/plain"); if (plain?.trim().startsWith("http")) return plain.trim();
    const html = ev.dataTransfer.getData("text/html"); const m = html?.match(/https?:\/\/[^"' >]+/i); return m?.[0] ?? "";
  }

  async function dropUrlAsCandidate(projectId: number, url: string) {
    try {
      const r = await apiPost<{ items: ScanItem[] }>("/collector/drop-url", { project_id: projectId, url });
      setScanItems(r.items || []); setNotice("URLを候補に追加しました");
    } catch {
      await runScan();
    }
  }

  async function onDropFiles(ev: DragEvent<HTMLDivElement>) {
    if (!selectedProject) return;
    ev.preventDefault(); setDragActive(false); clearMessages();
    const files = Array.from(ev.dataTransfer.files).filter((f) => f.type?.startsWith("image/") || /\.(png|jpe?g|webp|bmp|gif)$/i.test(f.name));
    if (files.length === 0) { const url = extractDroppedUrl(ev); if (url) await dropUrlAsCandidate(selectedProject.id, url); return; }
    try {
      const form = new FormData(); form.append("project_id", String(selectedProject.id)); files.forEach((f) => form.append("files", f));
      const res = await fetch(`${API_BASE}/collector/drop-files`, { method: "POST", body: form });
      const json = await res.json(); if (!res.ok) throw new Error(json?.detail ?? "drop");
      setScanItems(json.items || []); setNotice(`${json.added_count ?? files.length}件追加`);
    } catch (e) { setError(`ドラッグ&ドロップ追加失敗: ${String(e)}`); }
  }

  async function runImport() {
    if (!selectedProject) return;
    clearMessages();
    try { await apiPost("/collector/import", { project_id: selectedProject.id, selected_ids: selectedScanIds, naming_template: namingTemplate }); setNotice("取り込み完了"); }
    catch (e) { setError(`取り込み失敗: ${String(e)}`); }
  }

  async function removeCandidates(ids: number[]) {
    if (!selectedProject || ids.length === 0) return;
    try {
      const r = await apiPost<{ items: ScanItem[]; removed_count: number }>("/collector/remove-candidates", { project_id: selectedProject.id, candidate_ids: ids });
      setScanItems(r.items || []); setSelectedScanIds((prev) => prev.filter((id) => !ids.includes(id))); setNotice(`${r.removed_count ?? ids.length}件削除`);
    } catch (e) { setError(`候補削除に失敗: ${String(e)}`); }
  }

  async function prepareRepeatFolder() {
    if (!selectedProject) return;
    try {
      const r = await apiPost<{ folder: string; copied_images: number }>("/collector/prepare-repeat-folder", {
        project_id: selectedProject.id,
        repeats: repeatCount,
        folder_title: repeatFolderTitle
      });
      setNotice(`学習フォルダを作成: ${r.folder} (${r.copied_images}件コピー)`);
    }
    catch (e) { setError(`作成失敗: ${String(e)}`); }
  }

  async function runTags() { if (!selectedProject) return; try { await apiPost("/tags/generate", { project_id: selectedProject.id }); setNotice("タグ生成完了"); } catch (e) { setError(String(e)); } }
  function toggleScanSelection(id: number) { setSelectedScanIds((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id])); }

  async function startTraining(projectId: number) {
    try {
      await apiPost("/training/start", { project_id: projectId, epochs, repeats: trainRepeats, alpha, rank, save_every_n_epochs: saveEvery, output_name: outputName, base_checkpoint_path: baseCkpt, train_data_dir: trainDir, reg_data_dir: regDir, resolution });
      setNotice("学習を開始しました");
    } catch (e) { setError(`学習開始失敗: ${String(e)}`); }
  }
  async function loadDatasetPreview(projectId: number, dir: string) {
    try {
      const q = `project_id=${projectId}&train_data_dir=${encodeURIComponent(dir)}`;
      const r = await apiGet<DatasetPreview>(`/training/dataset-preview?${q}`);
      setDatasetPreview(r);
    } catch {
      setDatasetPreview(null);
    }
  }
  async function trainingAction(path: string, okMessage: string) { if (!selectedProject) return; try { await apiPost(path, { project_id: selectedProject.id }); setNotice(okMessage); } catch (e) { setError(String(e)); } }
  async function savePreviewPrompts() { try { await apiPost("/settings/preview-prompts", prompts, "PUT"); setNotice("プレビュー用プロンプト保存"); } catch (e) { setError(String(e)); } }
  async function saveToolPaths() { try { await apiPost("/settings/tool-paths", toolPaths, "PUT"); setNotice("連携設定を保存"); await loadIntegrations(); } catch (e) { setError(String(e)); } }
  async function autoDetectToolPaths() { try { const d = await apiPost<ToolPaths>("/settings/tool-paths/autodetect", {}); setToolPaths(d); setNotice("自動検出を実行"); } catch (e) { setError(String(e)); } }

  function etaText(sec?: number | null) { if (sec == null) return "計算中"; const h = Math.floor(sec / 3600); const m = Math.floor((sec % 3600) / 60); return `${h}h ${m}m`; }
  function integrationRow(label: string, keyName: keyof ToolPaths) {
    const c = integrationStatus?.checks?.[keyName];
    return <li key={keyName} className="statusRow"><span>{label}</span><span className={c?.ok ? "statusOk" : "statusNg"}>{c?.ok ? "OK" : "NG"}</span><span className="muted">{c?.reason ?? "未確認"}</span></li>;
  }

  return (
    <div className="layout">
      <aside className="sidebar">
        <h1 className="logo">LoRA制作ワークベンチ</h1>
        <nav className="menu">
          {[ ["dashboard", "ダッシュボード"], ["projects", "プロジェクト管理"], ["dataset", "データセット作成"], ["training", "学習制御"], ["integrations", "外部連携設定"], ["guide", "使い方ガイド"] ].map(([id, label]) => (
            <button key={id} className={tab === id ? "menuBtn active" : "menuBtn"} onClick={() => setTab(id as TabId)}>{label}</button>
          ))}
        </nav>
        <div className="healthCard"><div>API状態: {apiHealth}</div><button className="btn secondary small" onClick={() => void refreshAll()}>更新</button></div>
      </aside>
      <main className="content">
        {error && <div className="errorBanner">{error}</div>}
        {notice && <div className="noticeBanner">{notice}</div>}
        {tab === "dashboard" && <section className="panel"><h2>ダッシュボード</h2><div className="statsGrid"><div className="statBox"><div className="statLabel">有効プロジェクト</div><div className="statValue">{projects.length}</div></div><div className="statBox"><div className="statLabel">学習中</div><div className="statValue">{Object.values(statuses).filter((s) => s.status === "training").length}</div></div><div className="statBox"><div className="statLabel">連携OK</div><div className="statValue">{integrationStatus ? Object.values(integrationStatus.checks).filter((x) => x.ok).length : 0}</div></div></div></section>}

        {tab === "projects" && <section className="panel"><h2>プロジェクト管理</h2><div className="row"><input value={newProjectName} onChange={(e) => setNewProjectName(e.target.value)} placeholder="project_name" /><select value={newProjectType} onChange={(e) => setNewProjectType(e.target.value as "character" | "style")}><option value="character">Character LoRA</option><option value="style">Style LoRA</option></select><button className="btn primary" onClick={createProject}>作成</button></div><div className="projectTable">{projects.map((p) => <div key={p.id} className={selectedProjectId === p.id ? "projectRow selected projectRowBlock" : "projectRow projectRowBlock"}><button className="projectPickBtn" onClick={() => setSelectedProjectId(p.id)}><span>{p.name}</span><span>{p.project_type} / {statuses[p.id]?.status ?? "idle"}</span></button><div className="projectActions"><select value={p.project_type} onChange={(e) => void updateProjectType(p.id, e.target.value as "character" | "style")}><option value="character">character</option><option value="style">style</option></select><button className="btn secondary small" onClick={() => void duplicateProject(p.id, p.name)}>複製</button><button className="btn danger small" onClick={() => void deleteProject(p.id)}>削除</button></div></div>)}</div></section>}

        {tab === "dataset" && <section className="panel"><h2>データセット作成</h2>{!selectedProject ? <p className="muted">プロジェクトを選択してください。</p> : <div className={dragActive ? "card dropBoard active" : "card dropBoard"} onDrop={onDropFiles} onDragEnter={(e) => { e.preventDefault(); setDragActive(true); }} onDragOver={(e) => { e.preventDefault(); setDragActive(true); }} onDragLeave={() => setDragActive(false)}><label>収集元URL<input value={scanUrl} onChange={(e) => setScanUrl(e.target.value)} /></label><div className="row"><label>キーワード<input value={scanKeyword} onChange={(e) => setScanKeyword(e.target.value)} /></label><label>タグ絞り込み<input value={tagFilter} onChange={(e) => setTagFilter(e.target.value)} /></label></div><div className="row"><label>Aspect<select value={aspectFilter} onChange={(e) => setAspectFilter(e.target.value)}><option value="all">all</option><option value="portrait">portrait</option><option value="landscape">landscape</option><option value="square">square</option></select></label><label>minW<input type="number" value={minW} onChange={(e) => setMinW(Number(e.target.value || 0))} /></label><label>minH<input type="number" value={minH} onChange={(e) => setMinH(Number(e.target.value || 0))} /></label></div><div className="ctaRow"><button className="btn cta info" onClick={runScan}>1) 候補画像取得</button><button className="btn cta primary" onClick={runImport}>2) 取り込み</button><button className="btn cta accent" onClick={runTags}>3) タグ生成</button><button className="btn cta secondary" onClick={() => selectedProject && dropUrlAsCandidate(selectedProject.id, scanUrl)}>URLを候補追加</button></div><div className="row wrap"><button className="btn secondary" onClick={() => setSelectedScanIds(scanItems.map((x) => x.id))}>すべて選択</button><button className="btn secondary" onClick={() => setSelectedScanIds([])}>すべて解除</button><button className="btn danger" onClick={() => void removeCandidates(selectedScanIds)} disabled={selectedScanIds.length === 0}>選択候補を削除</button><span className="muted">選択中: {selectedScanIds.length}</span></div><div className="chips">{scanItems.map((i) => <button key={i.id} className={selectedScanIds.includes(i.id) ? "thumbCard active" : "thumbCard"} title={i.title} onClick={() => toggleScanSelection(i.id)} onDoubleClick={() => setExpandedTagCardId((prev) => (prev === i.id ? null : i.id))}><span className="thumbDelete" title={`${i.title} remove`} onClick={(e) => { e.preventDefault(); e.stopPropagation(); void removeCandidates([i.id]); }}>x</span><div className="thumbViewport">{i.thumbnail_url ? <img src={i.thumbnail_url} alt={i.title} /> : <div className="thumbFallback">NO IMAGE</div>}</div><span className="thumbMeta">{i.width}x{i.height} / {i.aspect}</span>{expandedTagCardId === i.id && <div className="tagFlow">{(i.tags || []).map((t) => <span key={`${i.id}-${t}`} className="tagChip">{t}</span>)}</div>}</button>)}</div><div className="row"><input value={namingTemplate} onChange={(e) => setNamingTemplate(e.target.value)} placeholder="{title}_{index}" /></div><div className="repeatRow"><label>Repeat<input type="number" value={repeatCount} onChange={(e) => setRepeatCount(Number(e.target.value || 1))} /></label><label>Folder<input value={repeatFolderTitle} onChange={(e) => setRepeatFolderTitle(e.target.value)} /></label></div><button className="btn cta primary repeatBtn" onClick={prepareRepeatFolder}>4) Create repeat folder ({repeatCount}_{repeatFolderTitle || "title"})</button></div>}</section>}

        {tab === "training" && <section className="panel"><h2>学習制御</h2>{!selectedProject ? <p className="muted">プロジェクトを選択してください。</p> : <><div className="card"><div className="trainingSummary"><div className="metricBox"><div className="metricLabel">状態</div><div className="metricValue">{currentStatus?.status ?? "idle"}</div></div><div className="metricBox"><div className="metricLabel">Epoch</div><div className="metricValue">{currentStatus?.epoch ?? 0} / {currentStatus?.total_epochs ?? 0}</div></div><div className="metricBox"><div className="metricLabel">Step</div><div className="metricValue">{currentStatus?.done_steps ?? 0} / {currentStatus?.total_steps ?? 0}</div></div><div className="metricBox"><div className="metricLabel">残り目安</div><div className="metricValue">{etaText(currentStatus?.eta_seconds)}</div></div></div><div className="progressBar big"><div className="progressFill" style={{ width: `${Math.min(100, Math.max(0, currentStatus?.progress_percent ?? 0))}%` }} /></div><div className="trainingSection"><h4>KOHYA params</h4><div className="paramPairGrid"><label>epochs<input type="number" value={epochs} onChange={(e) => setEpochs(Number(e.target.value || 1))} /></label><label>repeats<input type="number" value={trainRepeats} onChange={(e) => setTrainRepeats(Number(e.target.value || 1))} /></label><label>rank<input type="number" value={rank} onChange={(e) => setRank(Number(e.target.value || 1))} /></label><label>alpha<input type="number" step="0.1" value={alpha} onChange={(e) => setAlpha(Number(e.target.value || 1))} /></label></div><div className="row wrap"><label>save_every<input type="number" value={saveEvery} onChange={(e) => setSaveEvery(Number(e.target.value || 1))} /></label><label>resolution<input type="number" value={resolution} onChange={(e) => setResolution(Number(e.target.value || 512))} /></label></div></div><div className="trainingSection"><h4>Paths</h4><div className="row wrap"><label>output<input value={outputName} onChange={(e) => setOutputName(e.target.value)} /></label><label>checkpoint<input value={baseCkpt} onChange={(e) => setBaseCkpt(e.target.value)} /></label><label>train dir<input value={trainDir} onChange={(e) => setTrainDir(e.target.value)} /></label><label>reg dir<input value={regDir} onChange={(e) => setRegDir(e.target.value)} /></label></div><div className="datasetPreviewBox"><div className="muted">Dataset thumbnail</div>{datasetPreview?.thumbnail_url ? <img className="datasetPreviewImg" src={datasetPreview.thumbnail_url} alt="dataset preview" /> : <div className="thumbFallback">NO IMAGE</div>}<div className="muted">{datasetPreview?.image_path ?? "No image found"}</div></div></div><div className="trainingSection"><h4>Preview prompts</h4><div className="row wrap"><label>Positive<input value={prompts.positive_prompt} onChange={(e) => setPrompts({ ...prompts, positive_prompt: e.target.value })} /></label><label>Negative<input value={prompts.negative_prompt} onChange={(e) => setPrompts({ ...prompts, negative_prompt: e.target.value })} /></label><button className="btn secondary" onClick={savePreviewPrompts}>Save</button></div></div><div className="trainingSection"><h4>Run</h4><div className="row wrap"><button className="btn cta xl primary" onClick={() => startTraining(selectedProject.id)}>学習開始</button><button className="btn cta secondary" onClick={() => trainingAction("/training/stop-at-epoch", "epoch区切りで停止")}>epoch区切り停止</button><button className="btn cta warning" onClick={() => trainingAction("/training/stop-now", "すぐ停止")}>すぐ停止</button><button className="btn cta info" onClick={() => trainingAction("/training/resume", "再開")}>再開</button></div></div></div><div className="card"><h3>軽量プレビュー履歴</h3>{timeline.length === 0 ? <p className="muted">まだプレビューがありません。</p> : <div className="timeline">{timeline.map((t) => <div key={t.checkpoint_id} className="timelineItem"><strong>Epoch {t.epoch}</strong><div className="thumbGrid4">{Object.entries(t.sample_previews || {}).map(([slot, img]) => <div key={slot} className="smallThumb"><img src={img} alt={slot} /><span>{slot}</span></div>)}</div></div>)}</div>}</div></>}</section>}

        {tab === "integrations" && <section className="panel"><h2>外部連携設定</h2><div className="card"><div className="formGrid"><label>Python<input value={toolPaths.python_exe} onChange={(e) => setToolPaths({ ...toolPaths, python_exe: e.target.value })} /></label><label>kohya<input value={toolPaths.kohya_root} onChange={(e) => setToolPaths({ ...toolPaths, kohya_root: e.target.value })} /></label><label>ComfyUI<input value={toolPaths.comfyui_root} onChange={(e) => setToolPaths({ ...toolPaths, comfyui_root: e.target.value })} /></label><label>WD14<input value={toolPaths.wd14_script} onChange={(e) => setToolPaths({ ...toolPaths, wd14_script: e.target.value })} /></label><label>temp_dir<input value={toolPaths.temp_dir} onChange={(e) => setToolPaths({ ...toolPaths, temp_dir: e.target.value })} /></label><label>dataset_base_dir<input value={toolPaths.dataset_base_dir} onChange={(e) => setToolPaths({ ...toolPaths, dataset_base_dir: e.target.value })} /></label></div><div className="row wrap"><button className="btn info" onClick={autoDetectToolPaths}>自動検出</button><button className="btn primary" onClick={saveToolPaths}>保存</button><button className="btn secondary" onClick={() => void loadIntegrations()}>再確認</button></div></div><div className="card">{!integrationStatus ? <p className="muted">未確認</p> : <ul className="statusList">{integrationRow("Python", "python_exe")}{integrationRow("kohya", "kohya_root")}{integrationRow("ComfyUI", "comfyui_root")}{integrationRow("WD14", "wd14_script")}{integrationRow("temp_dir", "temp_dir")}{integrationRow("dataset_base", "dataset_base_dir")}</ul>}</div></section>}

        {tab === "guide" && <section className="panel"><h2>使い方ガイド</h2><ol className="guideList"><li>外部連携設定で必要パスを確認します。</li><li>データセット作成で候補画像を取得し、URL追加やドラッグ&ドロップで候補を増やします。</li><li>不要候補を削除し、必要画像だけ取り込みます。</li><li>学習制御でKOHYA寄りの項目を設定して学習を進めます。</li></ol></section>}
      </main>
    </div>
  );
}

