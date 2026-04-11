import { useEffect, useState } from "react";

type Health = {
  status: string;
};

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
  message: string;
};

type ScanResult = {
  detected: number;
  items: { id: number }[];
};

const API_BASE = "http://127.0.0.1:8000";

export default function App() {
  const [health, setHealth] = useState<string>("checking");
  const [projects, setProjects] = useState<Project[]>([]);
  const [statuses, setStatuses] = useState<Record<number, TrainingStatus>>({});
  const [name, setName] = useState<string>("");
  const [error, setError] = useState<string>("");

  useEffect(() => {
    void refresh();
  }, []);

  async function refresh() {
    setError("");
    try {
      const h = await fetch(`${API_BASE}/health`);
      const hJson: Health = await h.json();
      setHealth(hJson.status);

      const p = await fetch(`${API_BASE}/projects`);
      const pJson: Project[] = await p.json();
      setProjects(pJson);
      await loadStatuses(pJson);
    } catch (e) {
      setHealth("offline");
      setError(`API接続エラー: ${String(e)}`);
    }
  }

  async function loadStatuses(items: Project[]) {
    const entries = await Promise.all(
      items.map(async (p) => {
        try {
          const res = await fetch(`${API_BASE}/training/status?project_id=${p.id}`);
          const json: TrainingStatus = await res.json();
          return [p.id, json] as const;
        } catch {
          return [
            p.id,
            { run_id: null, status: "unknown", stop_mode: null, message: "fetch failed" }
          ] as const;
        }
      })
    );
    setStatuses(Object.fromEntries(entries));
  }

  async function createProject() {
    if (!name.trim()) return;
    setError("");
    try {
      const res = await fetch(`${API_BASE}/projects`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: name.trim() })
      });
      if (!res.ok) {
        const body = await res.json();
        throw new Error(body?.detail ?? "作成に失敗しました");
      }
      setName("");
      await refresh();
    } catch (e) {
      setError(String(e));
    }
  }

  async function callTrainingAction(path: string, projectId: number) {
    setError("");
    try {
      const res = await fetch(`${API_BASE}${path}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ project_id: projectId })
      });
      if (!res.ok) {
        const body = await res.json();
        throw new Error(body?.detail ?? "操作に失敗しました");
      }
      await refresh();
    } catch (e) {
      setError(String(e));
    }
  }

  async function quickScanImportAndTag(projectId: number) {
    setError("");
    try {
      const scanRes = await fetch(`${API_BASE}/collector/scan`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          project_id: projectId,
          url: "https://example.com/mock-collection"
        })
      });
      if (!scanRes.ok) throw new Error("scan failed");
      const scanJson: ScanResult = await scanRes.json();
      const selected = scanJson.items.slice(0, 5).map((x) => x.id);

      const importRes = await fetch(`${API_BASE}/collector/import`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          project_id: projectId,
          selected_ids: selected,
          naming_template: "{title}_{index}"
        })
      });
      if (!importRes.ok) throw new Error("import failed");

      const tagRes = await fetch(`${API_BASE}/tags/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ project_id: projectId })
      });
      if (!tagRes.ok) throw new Error("tags generate failed");

      await refresh();
    } catch (e) {
      setError(String(e));
    }
  }

  return (
    <main className="page">
      <section className="card">
        <h1>LoRA制作ワークベンチ</h1>
        <p className="muted">API状態: {health}</p>
        <button onClick={refresh} className="btn secondary">
          再読み込み
        </button>
      </section>

      <section className="card">
        <h2>プロジェクト作成</h2>
        <div className="row">
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="LoRA名を入力"
          />
          <button onClick={createProject} className="btn primary">
            作成
          </button>
        </div>
        {error && <p className="error">{error}</p>}
      </section>

      <section className="card">
        <h2>プロジェクト一覧</h2>
        {projects.length === 0 ? (
          <p className="muted">まだプロジェクトはありません</p>
        ) : (
          <ul className="projectList">
            {projects.map((p) => (
              <li key={p.id} className="projectItem">
                <div className="projectTop">
                  <strong>{p.name}</strong>
                  <span className="muted">project status: {p.status}</span>
                </div>
                <div className="projectMeta">
                  <span className="muted">
                    training: {statuses[p.id]?.status ?? "loading"} / stop_mode:{" "}
                    {statuses[p.id]?.stop_mode ?? "-"}
                  </span>
                </div>
                <div className="row actions">
                  <button
                    onClick={() => callTrainingAction("/training/start", p.id)}
                    className="btn primary"
                  >
                    Start
                  </button>
                  <button
                    onClick={() => callTrainingAction("/training/stop-at-epoch", p.id)}
                    className="btn secondary"
                  >
                    Stop@Epoch
                  </button>
                  <button
                    onClick={() => callTrainingAction("/training/stop-now", p.id)}
                    className="btn warning"
                  >
                    StopNow
                  </button>
                  <button
                    onClick={() => callTrainingAction("/training/resume", p.id)}
                    className="btn info"
                  >
                    Resume
                  </button>
                  <button
                    onClick={() => quickScanImportAndTag(p.id)}
                    className="btn accent"
                  >
                    Quick Scan+Import+Tag
                  </button>
                </div>
              </li>
            ))}
          </ul>
        )}
      </section>
    </main>
  );
}
