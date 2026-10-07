import { type KeyboardEvent as ReactKeyboardEvent, type ReactNode, useEffect, useMemo, useRef, useState } from "react";
import {
  BookOpen,
  Boxes,
  Check,
  ChevronDown,
  CircleHelp,
  Command,
  Gauge,
  ImagePlus,
  Library,
  Menu,
  RefreshCw,
  Search,
  Settings2,
  Sparkles,
  X,
} from "lucide-react";
import type { Project, SystemStatus, TabId, TrainingStatus } from "../../types";

type Props = {
  tab: TabId;
  projects: Project[];
  selectedProject: Project | null;
  statuses: Record<number, TrainingStatus>;
  apiHealth: string;
  systemStatus: SystemStatus | null;
  refreshing: boolean;
  error: string;
  notice: string;
  onDismissMessage: () => void;
  onRefresh: () => void;
  onSelectProject: (id: number) => void;
  onNavigate: (tab: TabId) => void;
  children: ReactNode;
};

const workflow = [
  { id: "dataset" as const, label: "データセット", hint: "画像・キャプション", icon: ImagePlus, shortcut: "Alt+1" },
  { id: "training" as const, label: "学習", hint: "モデル・設定", icon: Gauge, shortcut: "Alt+2" },
  { id: "runs" as const, label: "結果", hint: "出力・比較", icon: Boxes, shortcut: "Alt+3" },
  { id: "library" as const, label: "ライブラリ", hint: "保存したLoRA", icon: Library, shortcut: "Alt+4" },
];

export default function StudioShell({
  tab,
  projects,
  selectedProject,
  statuses,
  apiHealth,
  systemStatus,
  refreshing,
  error,
  notice,
  onDismissMessage,
  onRefresh,
  onSelectProject,
  onNavigate,
  children,
}: Props) {
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [mobileNavOpen, setMobileNavOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [activeCommandIndex, setActiveCommandIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const paletteRef = useRef<HTMLElement>(null);
  const paletteTriggerRef = useRef<HTMLButtonElement>(null);
  const paletteWasOpenRef = useRef(false);
  const activeRun = selectedProject ? statuses[selectedProject.id] : undefined;
  const targetGpu = systemStatus?.gpu?.devices?.find(
    (gpu) => gpu.index === systemStatus.gpu.target_physical_index,
  );

  const commands = useMemo(
    () => [
      ...workflow.map((item) => ({
        label: `Workspace: ${item.label}を開く`,
        detail: item.hint,
        shortcut: item.shortcut,
        action: () => onNavigate(item.id),
        disabled: !selectedProject,
      })),
      {
        label: "Project: プロジェクト一覧を開く",
        detail: "別のLoRA制作へ切り替える",
        shortcut: "",
        action: () => onNavigate("home"),
        disabled: false,
      },
      {
        label: "System: 接続状態を更新",
        detail: "API・GPU・外部サービスを再取得",
        shortcut: "",
        action: onRefresh,
        disabled: false,
      },
      {
        label: "Settings: ツール設定を開く",
        detail: "Anima / ComfyUI / 保存先",
        shortcut: "",
        action: () => onNavigate("integrations"),
        disabled: false,
      },
    ],
    [onNavigate, onRefresh, selectedProject],
  );

  const filteredCommands = commands.filter((command) =>
    `${command.label} ${command.detail}`.toLowerCase().includes(query.toLowerCase()),
  );

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setPaletteOpen((value) => !value);
      }
      if (event.key === "Escape") {
        setPaletteOpen(false);
        setMobileNavOpen(false);
      }
      if (event.altKey && /^[1-4]$/.test(event.key) && selectedProject) {
        event.preventDefault();
        onNavigate(workflow[Number(event.key) - 1].id);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onNavigate, selectedProject]);

  useEffect(() => {
    if (paletteOpen) {
      paletteWasOpenRef.current = true;
      setQuery("");
      setActiveCommandIndex(0);
      inputRef.current?.focus();
      return;
    }
    if (paletteWasOpenRef.current) {
      paletteWasOpenRef.current = false;
      window.setTimeout(() => paletteTriggerRef.current?.focus(), 0);
    }
  }, [paletteOpen]);

  useEffect(() => {
    setActiveCommandIndex((current) => Math.max(0, Math.min(current, filteredCommands.length - 1)));
  }, [query, filteredCommands.length]);

  function execute(action: () => void, disabled: boolean) {
    if (disabled) return;
    setPaletteOpen(false);
    action();
  }

  function commandKeyDown(event: ReactKeyboardEvent<HTMLInputElement>) {
    if (event.key === "ArrowDown") {
      event.preventDefault();
      setActiveCommandIndex((current) => Math.min(filteredCommands.length - 1, current + 1));
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      setActiveCommandIndex((current) => Math.max(0, current - 1));
    } else if (event.key === "Enter") {
      event.preventDefault();
      const command = filteredCommands[activeCommandIndex];
      if (command) execute(command.action, command.disabled);
    }
  }

  function paletteKeyDown(event: ReactKeyboardEvent<HTMLElement>) {
    if (event.key === "Escape") {
      event.preventDefault();
      event.stopPropagation();
      setPaletteOpen(false);
      return;
    }
    if (event.key !== "Tab" || !paletteRef.current) return;
    const focusable = Array.from(paletteRef.current.querySelectorAll<HTMLElement>(
      'input, button:not(:disabled), a[href], [tabindex]:not([tabindex="-1"])',
    )).filter((element) => element.getClientRects().length > 0);
    if (!focusable.length) return;
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (event.shiftKey && (document.activeElement === first || !paletteRef.current.contains(document.activeElement))) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  }

  return (
    <div className="studio-app studio-refined">
      <aside className={`studio-rail ${mobileNavOpen ? "is-open" : ""}`} aria-label="制作ワークフロー">
        <button className="studio-brand" onClick={() => onNavigate("home")} aria-label="LoRA Basepipe ホーム">
          <span className="studio-brand-mark"><Sparkles size={17} /></span>
          <span className="studio-brand-copy"><strong>Basepipe</strong></span>
        </button>

        <nav className="studio-workflow-nav">
          
          {workflow.map((item) => {
            const Icon = item.icon;
            const active = tab === item.id || (tab === "compare" && item.id === "runs");
            return (
              <button
                key={item.id}
                className={`studio-nav-item ${active ? "is-active" : ""}`}
                disabled={!selectedProject}
                onClick={() => { onNavigate(item.id); setMobileNavOpen(false); }}
                aria-current={active ? "page" : undefined}
              >
                
                <Icon size={18} />
                <span><strong>{item.label}</strong></span>
              </button>
            );
          })}
        </nav>

        <div className="studio-rail-footer">
          <button className={tab === "integrations" ? "is-active" : ""} onClick={() => onNavigate("integrations")}>
            <Settings2 size={17} /><span>設定</span>
          </button>
          <button className={tab === "guide" ? "is-active" : ""} onClick={() => onNavigate("guide")}>
            <CircleHelp size={17} /><span>ガイド</span>
          </button>
        </div>
      </aside>

      <div className="studio-frame">
        <header className="studio-topbar">
          <button className="studio-mobile-menu" onClick={() => setMobileNavOpen((value) => !value)} aria-label="メニュー">
            {mobileNavOpen ? <X size={20} /> : <Menu size={20} />}
          </button>
          <div className="studio-project-switcher">
            <span className="studio-project-type">{selectedProject?.project_type === "style" ? "STYLE" : "CHARACTER"}</span>
            <select
              aria-label="プロジェクトを切り替え"
              value={selectedProject?.id ?? ""}
              onChange={(event) => onSelectProject(Number(event.target.value))}
            >
              <option value="" disabled>プロジェクトを選択</option>
              {projects.map((project) => <option key={project.id} value={project.id}>{project.name}</option>)}
            </select>
            <ChevronDown size={14} aria-hidden="true" />
          </div>

          <button ref={paletteTriggerRef} className="studio-command-trigger" aria-label="操作を検索" title="操作を検索 · Ctrl K" onClick={() => setPaletteOpen(true)} aria-haspopup="dialog" aria-expanded={paletteOpen}>
            <Search size={15} /><span>操作を検索</span><kbd>Ctrl K</kbd>
          </button>

          <div className="studio-topbar-status">
            {activeRun?.status === "training" && (
              <button className="studio-run-pill" onClick={() => onNavigate("runs")}>
                <span className="studio-live-dot" />Run #{activeRun.run_id} · {Math.round(activeRun.progress_percent ?? 0)}%
              </button>
            )}
            <span className={`studio-health ${apiHealth === "接続OK" ? "is-online" : "is-offline"}`}>
              <span />{apiHealth === "接続OK" ? "Connected" : apiHealth}
            </span>
            {targetGpu && <span className="studio-gpu" title={`VRAM ${targetGpu.vram_pct}%`}>GPU {targetGpu.index} · {Math.round(targetGpu.vram_pct)}%</span>}
            <button className="studio-icon-button" onClick={onRefresh} disabled={refreshing} aria-label="すべて更新">
              <RefreshCw size={16} className={refreshing ? "spin" : ""} />
            </button>
          </div>
        </header>

        {(error || notice) && (
          <div className={`studio-banner ${error ? "is-error" : "is-success"}`} role={error ? "alert" : "status"}>
            <span>{error || notice}</span>
            <button type="button" onClick={onDismissMessage} aria-label="通知を閉じる"><X size={14} /></button>
          </div>
        )}

        <main className="studio-main">{children}</main>
      </div>

      {paletteOpen && (
        <div className="studio-dialog-backdrop" role="presentation" onMouseDown={() => setPaletteOpen(false)}>
          <section ref={paletteRef} className="studio-command-palette" role="dialog" aria-modal="true" aria-label="コマンドパレット" onMouseDown={(event) => event.stopPropagation()} onKeyDown={paletteKeyDown}>
            <div className="studio-command-input">
              <Command size={18} />
              <input ref={inputRef} autoFocus value={query} onChange={(event) => setQuery(event.target.value)} onKeyDown={commandKeyDown} placeholder="画面や操作を検索…" aria-label="操作を検索" aria-controls="studio-command-list" />
              <kbd>Esc</kbd>
            </div>
            <div className="studio-command-list" id="studio-command-list">
              {filteredCommands.map((command, index) => (
                <button key={command.label} className={index === activeCommandIndex ? "is-active" : ""} disabled={command.disabled} onMouseEnter={() => setActiveCommandIndex(index)} onClick={() => execute(command.action, command.disabled)}>
                  <span><strong>{command.label}</strong><small>{command.disabled ? "先にプロジェクトを選択してください" : command.detail}</small></span>
                  {command.shortcut && <kbd>{command.shortcut}</kbd>}
                </button>
              ))}
              {filteredCommands.length === 0 && <p className="studio-empty-line">一致する操作はありません</p>}
            </div>
            <footer><span><Check size={12} /> Enterで実行</span><span><BookOpen size={12} /> GPU処理はPreflightを迂回しません</span></footer>
          </section>
        </div>
      )}
    </div>
  );
}
