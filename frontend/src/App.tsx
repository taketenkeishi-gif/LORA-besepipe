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

const API_BASE = "http://127.0.0.1:8000";

export default function App() {
  const [health, setHealth] = useState<string>("checking");
  const [projects, setProjects] = useState<Project[]>([]);
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
    } catch (e) {
      setHealth("offline");
      setError(`API接続エラー: ${String(e)}`);
    }
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
                <strong>{p.name}</strong>
                <span className="muted">status: {p.status}</span>
              </li>
            ))}
          </ul>
        )}
      </section>
    </main>
  );
}

