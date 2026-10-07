import { useMemo, useState } from "react";
import { ArrowRight, Boxes, Plus, Sparkles, UserRound, Palette, X } from "lucide-react";
import { apiPost } from "../../lib/api";
import type { Project, TrainingStatus } from "../../types";

type Props = {
  projects: Project[];
  statuses: Record<number, TrainingStatus>;
  onOpen: (project: Project) => void;
  onRefresh: () => Promise<void>;
  showError: (message: string) => void;
  showNotice: (message: string) => void;
};

export default function ProjectHome({ projects, statuses, onOpen, onRefresh, showError, showNotice }: Props) {
  const [createOpen, setCreateOpen] = useState(false);
  const [name, setName] = useState("");
  const [type, setType] = useState<"character" | "style">("character");
  const [creating, setCreating] = useState(false);
  const counts = useMemo(() => ({
    running: Object.values(statuses).filter((status) => status.status === "training").length,
    completed: Object.values(statuses).filter((status) => status.status === "completed").length,
  }), [statuses]);

  async function createProject() {
    if (!name.trim()) return;
    setCreating(true);
    try {
      const project = await apiPost<Project>("/projects", { name: name.trim(), project_type: type });
      await onRefresh();
      showNotice(`「${project.name}」を作成しました`);
      setCreateOpen(false);
      setName("");
      onOpen(project);
    } catch (error) {
      showError(error instanceof Error ? error.message : "プロジェクト作成に失敗しました");
    } finally {
      setCreating(false);
    }
  }

  return (
    <div className="studio-home">
      <section className="studio-home-hero">
        <div>
          <span className="studio-kicker"><Sparkles size={13} /> LoRA production workspace</span>
          <h1>作ることに集中できる<br />LoRAワークスペース</h1>
          <p>素材の整理から学習、比較、Exportまでをひとつの流れで管理します。</p>
        </div>
        <button className="studio-primary-button" onClick={() => setCreateOpen(true)}><Plus size={17} /> 新しいプロジェクト</button>
      </section>

      <div className="studio-home-metrics" aria-label="プロジェクト概要">
        <div><span>Projects</span><strong>{projects.length}</strong></div>
        <div><span>Running now</span><strong>{counts.running}</strong></div>
        <div><span>Completed</span><strong>{counts.completed}</strong></div>
        <div><span>Workflow</span><strong>4 stages</strong></div>
      </div>

      <section className="studio-section">
        <div className="studio-section-heading">
          <div><p className="studio-eyebrow">RECENT WORK</p><h2>プロジェクト</h2></div>
          <p>続きから開くと、現在の状態に応じた次の操作を表示します。</p>
        </div>
        <div className="studio-project-grid">
          {projects.map((project) => {
            const status = statuses[project.id];
            const Icon = project.project_type === "style" ? Palette : UserRound;
            return (
              <button key={project.id} className="studio-project-card" onClick={() => onOpen(project)}>
                <span className={`studio-project-icon is-${project.project_type}`}><Icon size={21} /></span>
                <span className="studio-project-copy">
                  <small>{project.project_type === "style" ? "STYLE LORA" : "CHARACTER LORA"}</small>
                  <strong>{project.name}</strong>
                  <span className={`studio-status-text is-${status?.status ?? project.status}`}>
                    {status?.status === "training" ? `学習中 · ${Math.round(status.progress_percent ?? 0)}%` : status?.status === "completed" ? "学習結果あり" : "準備を続ける"}
                  </span>
                </span>
                <ArrowRight size={18} />
              </button>
            );
          })}
          {projects.length === 0 && (
            <div className="studio-empty-state"><Boxes size={26} /><h3>プロジェクトはまだありません</h3><p>CharacterまたはStyle LoRAから始められます。</p><button onClick={() => setCreateOpen(true)}>最初のプロジェクトを作る</button></div>
          )}
        </div>
      </section>

      {createOpen && (
        <div className="studio-dialog-backdrop" role="presentation" onMouseDown={() => setCreateOpen(false)}>
          <section className="studio-modal" role="dialog" aria-modal="true" aria-labelledby="create-project-title" onMouseDown={(event) => event.stopPropagation()}>
            <header><div><p className="studio-eyebrow">NEW PROJECT</p><h2 id="create-project-title">何を学習しますか？</h2></div><button className="studio-icon-button" onClick={() => setCreateOpen(false)} aria-label="閉じる"><X size={18} /></button></header>
            <div className="studio-choice-grid">
              <button className={type === "character" ? "is-selected" : ""} onClick={() => setType("character")}><UserRound size={21} /><strong>Character</strong><small>人物・衣装・Identity</small></button>
              <button className={type === "style" ? "is-selected" : ""} onClick={() => setType("style")}><Palette size={21} /><strong>Style</strong><small>画風・質感・構図</small></button>
            </div>
            <label className="studio-field"><span>プロジェクト名</span><input autoFocus value={name} onChange={(event) => setName(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter") void createProject(); }} placeholder="例: Melusine identity" /></label>
            <footer><button className="studio-secondary-button" onClick={() => setCreateOpen(false)}>キャンセル</button><button className="studio-primary-button" disabled={!name.trim() || creating} onClick={() => void createProject()}>{creating ? "作成中…" : "プロジェクトを作成"}<ArrowRight size={16} /></button></footer>
          </section>
        </div>
      )}
    </div>
  );
}
