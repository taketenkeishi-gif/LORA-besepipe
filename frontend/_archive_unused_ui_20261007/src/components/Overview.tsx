import { useEffect, useState } from "react";
import { ArrowRight, CheckCircle2, Circle, AlertTriangle, Layers3, LockKeyhole } from "lucide-react";
import { apiGet } from "../lib/api";
import type { Project, TrainingStatus } from "../types";

type Props = { project: Project; status?: TrainingStatus; onNavigate: (tab: "dataset" | "training" | "runs" | "compare" | "library") => void };
type DatasetAudit = { selected: number; caption_rate: number; missing_caption: number; missing_files: number; warnings: string[] };
type WorkspaceConcept = { name: string; trigger_token: string; concept_type?: string };
type WorkspaceAsset = { review_status?: string; user_rating?: number | null };

export default function Overview({ project, status, onNavigate }: Props) {
  const [workspace, setWorkspace] = useState<{ concepts: WorkspaceConcept[]; assets: WorkspaceAsset[]; snapshots: Array<{ name: string; item_count: number; snapshot_hash: string }> } | null>(null);
  const [audit, setAudit] = useState<DatasetAudit | null>(null);
  useEffect(() => {
    let active = true;
    void Promise.all([
      apiGet<typeof workspace>(`/basepipe/projects/${project.id}/workspace`),
      apiGet<DatasetAudit>(`/basepipe/projects/${project.id}/dataset-audit`),
    ]).then(([data, nextAudit]) => { if (active) { setWorkspace(data); setAudit(nextAudit); } }).catch(() => { if (active) { setWorkspace(null); setAudit(null); } });
    return () => { active = false; };
  }, [project.id]);
  const state = status?.status ?? "idle";
  const failed = state === "error";
  const completed = state === "completed";
  const hasConcept = Boolean(workspace?.concepts.length);
  const hasSnapshot = Boolean(workspace?.snapshots.length);
  const datasetReady = Boolean(audit && audit.selected > 0 && audit.caption_rate === 1 && audit.missing_files === 0 && audit.missing_caption === 0);
  const qualityUnassessed = workspace?.assets.filter((asset) => asset.user_rating == null || asset.review_status !== "approved").length ?? 0;
  const nextAction = hasSnapshot ? "Train画面でAnima RunをReview" : !hasConcept ? "DatasetでConceptとTriggerを保存" : !datasetReady ? "Datasetで画像・captionをReview" : "DatasetでSnapshotをsealed";
  const steps = [
    { label: "Dataset", done: datasetReady, warning: Boolean(audit && !datasetReady) },
    { label: "Prepare", done: hasSnapshot },
    { label: "Train", done: completed, warning: failed },
    { label: "Compare", done: false },
    { label: "Export", done: false },
  ];
  return <div className="space-y-5">
    <div><h2 className="text-xl font-bold text-gray-100">{project.name}</h2><p className="text-sm text-gray-500">プロジェクトの現在位置</p></div>
    <div className="grid gap-4 md:grid-cols-3">
      <section className="rounded-xl border border-indigo-900/70 bg-indigo-950/20 p-5"><div className="flex items-center gap-2 text-indigo-300"><Layers3 size={16}/><p className="text-xs font-semibold uppercase tracking-wide">Anima First</p></div><p className="mt-2 text-lg font-semibold text-gray-100">Anima Base v1.0</p><p className="mt-1 text-xs text-gray-500">物理GPU1を既定経路に固定（実機状態はTraining / Operationsで確認）</p><p className="mt-2 text-[11px] text-indigo-200">Primary Intent: {project.project_type === "style" ? "Style LoRA" : "Character LoRA"}</p></section>
      <section className="rounded-xl border border-gray-800 bg-gray-900/50 p-5"><div className="flex items-center justify-between"><div><p className="text-xs text-gray-500">Dataset</p><p className="mt-1 text-2xl font-semibold text-gray-100">入力データ</p><p className="mt-1 text-xs text-gray-500">詳細はDatasetで確認できます</p></div><button onClick={() => onNavigate("dataset")} className="rounded-lg border border-gray-700 p-2 text-gray-400 hover:bg-gray-800"><ArrowRight size={15}/></button></div></section>
      <section className="rounded-xl border border-gray-800 bg-gray-900/50 p-5"><div className="flex items-center justify-between"><div><p className="text-xs text-gray-500">Latest Run</p><p className={`mt-1 text-2xl font-semibold ${failed ? "text-red-300" : completed ? "text-emerald-300" : "text-gray-100"}`}>{status?.run_id ? `Run #${status.run_id}` : "未実行"}</p><p className="mt-1 text-xs text-gray-500">{failed ? status?.failure?.message ?? "失敗理由未取得" : status?.message ?? "次のRunを設定できます"}</p></div><button onClick={() => onNavigate("runs")} className="rounded-lg border border-gray-700 p-2 text-gray-400 hover:bg-gray-800"><ArrowRight size={15}/></button></div></section>
    </div>
    <section className="rounded-xl border border-gray-800 bg-gray-900/50 p-5"><div className="flex items-center justify-between"><div><div className="flex items-center gap-2"><LockKeyhole size={15} className="text-emerald-400"/><h3 className="text-sm font-semibold text-gray-300">Basepipe Lineage</h3></div><p className="mt-1 text-xs text-gray-500">AssetのSHA-256と不変Dataset SnapshotをRunへ接続します</p></div><button onClick={() => onNavigate("dataset")} className="rounded-lg border border-gray-700 p-2 text-gray-400 hover:bg-gray-800"><ArrowRight size={15}/></button></div><div className="mt-4 grid gap-3 sm:grid-cols-3"><div><p className="text-[11px] text-gray-500">Concept</p><p className="mt-1 text-lg font-semibold text-gray-100">{workspace?.concepts.length ?? 0}</p><p className="text-[11px] text-gray-600">{workspace?.concepts[0]?.trigger_token ?? "trigger未設定"}</p></div><div><p className="text-[11px] text-gray-500">Assets</p><p className="mt-1 text-lg font-semibold text-gray-100">{workspace?.assets.length ?? 0}</p><p className="text-[11px] text-gray-600">content hash付き</p></div><div><p className="text-[11px] text-gray-500">Snapshots</p><p className="mt-1 text-lg font-semibold text-gray-100">{workspace?.snapshots.length ?? 0}</p><p className="text-[11px] text-gray-600">{workspace?.snapshots[0] ? `${workspace.snapshots[0].item_count} items / sealed` : "未作成"}</p></div></div><div className="mt-4 flex flex-wrap gap-1">{workspace?.concepts.slice(0, 6).map((concept) => <span key={concept.name} className="rounded bg-fuchsia-950/40 px-2 py-1 text-[10px] text-fuchsia-200">{concept.name}: {concept.trigger_token}</span>)}</div><div className="mt-4 flex flex-wrap items-center justify-between gap-2 rounded-lg border border-indigo-900/50 bg-indigo-950/20 px-3 py-2"><span className="text-xs text-indigo-200">次の操作: {nextAction}</span><button onClick={() => onNavigate(hasSnapshot ? "training" : "dataset")} className="rounded border border-indigo-800 px-2 py-1 text-[11px] text-indigo-200 hover:bg-indigo-900/40">開く →</button></div>{audit && <p className="mt-2 text-[10px] text-gray-600">Dataset監査: {audit.selected}件 · caption {Math.round(audit.caption_rate * 100)}% · 未検出 {audit.missing_files}件{audit.warnings.length ? ` · 警告 ${audit.warnings.length}件` : ""}</p>}</section>
    <section className="rounded-xl border border-amber-900/50 bg-amber-950/10 p-4"><div className="flex items-center justify-between"><div><h3 className="text-sm font-semibold text-amber-200">Quality Review</h3><p className="mt-1 text-xs text-gray-500">Character / Styleの審美評価が未確定のAsset</p></div><span className="text-2xl font-semibold text-amber-300">{qualityUnassessed}</span></div><button onClick={() => onNavigate("dataset")} className="mt-3 rounded border border-amber-800/70 px-2 py-1 text-[11px] text-amber-200 hover:bg-amber-900/30">Dataset Reviewを開く →</button></section>
    <section className="rounded-xl border border-gray-800 bg-gray-900/50 p-5"><div className="mb-4 flex items-center justify-between"><h3 className="text-sm font-semibold text-gray-300">工程</h3><button onClick={() => onNavigate("training")} className="text-xs text-indigo-300 hover:text-indigo-200">New Training Run →</button></div><div className="grid grid-cols-5 gap-2">{steps.map((step) => <button key={step.label} onClick={() => onNavigate(step.label === "Dataset" ? "dataset" : step.label === "Train" ? "training" : step.label === "Compare" ? "compare" : step.label === "Export" ? "library" : "dataset")} className="text-left"><div className="flex items-center gap-1 text-xs text-gray-400">{step.warning ? <AlertTriangle size={13} className="text-red-400"/> : step.done ? <CheckCircle2 size={13} className="text-emerald-400"/> : <Circle size={13} className="text-gray-600"/>}{step.label}</div><div className={`mt-2 h-1 rounded ${step.warning ? "bg-red-500" : step.done ? "bg-emerald-500" : "bg-gray-700"}`}/></button>)}</div></section>
  </div>;
}
