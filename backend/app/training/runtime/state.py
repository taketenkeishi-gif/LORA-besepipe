"""学習 run の実行時状態（スレッド・プロセス管理）。

Backend 実装（musubi/sdxl/simulated）はこのモジュールの RUNNER_LOCK /
RUNNER_PROCESSES を使ってサブプロセスハンドルを共有する。API 層
（app.routers.training）はこのモジュールを直接いじらず、ensure_runner() /
terminate_run() を通じてのみ操作する。
"""
from __future__ import annotations

import subprocess
import threading
from pathlib import Path
from typing import Callable

from ...db import get_conn

RUNNER_LOCK = threading.Lock()
RUNNER_THREADS: dict[int, threading.Thread] = {}
RUNNER_PROCESSES: dict[int, "subprocess.Popen[str]"] = {}

# ── 学習時間予測（リアルタイム計測 + キャリブレーション） ───────────────────
# project_id -> {"ema": float, "last_step": int, "last_t": float}
STEP_TIMING: dict[int, dict] = {}

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}

_RUNS_ROOT = Path(__file__).resolve().parents[4] / ".runtime" / "runs"


def run_dir(run_id: int) -> Path:
    return _RUNS_ROOT / str(run_id)


def latest_run(conn, project_id: int):
    """project の最新 training_run 行を返す（Backend・API 両方から共通利用）。"""
    return conn.execute(
        """
        SELECT id, status, stop_mode, current_epoch, current_step, total_epochs,
               steps_per_epoch, config_json, log_path
        FROM training_runs
        WHERE project_id = ? ORDER BY id DESC LIMIT 1
        """,
        (project_id,),
    ).fetchone()


def set_project_status(conn, project_id: int, status: str) -> None:
    conn.execute("UPDATE projects SET status = ? WHERE id = ?", (status, project_id))


def ensure_runner(project_id: int, loop_fn: Callable[[int], None]) -> None:
    """project_id に対応する監視スレッドが無ければ起動する。

    loop_fn はディスパッチャ（app.routers.training._runner_loop 相当）を
    呼び出し側から注入してもらう。ここで import すると循環 import になるため。
    """
    with RUNNER_LOCK:
        alive = RUNNER_THREADS.get(project_id)
        if alive is not None and alive.is_alive():
            return
        t = threading.Thread(target=loop_fn, args=(project_id,), daemon=True)
        RUNNER_THREADS[project_id] = t
        t.start()


def _terminate_process_tree(pid: int) -> None:
    """指定PIDとその子孫プロセス全てを終了する(Windows対応)。

    kohya_ss は accelerate 経由で子プロセス(データローダーワーカー等)を
    spawn することがあり、親プロセスのみに proc.terminate()/kill() を送っても
    子プロセスが取り残される(ゾンビ・孫プロセス残存によるVRAM解放不全・
    ポート/ファイルロック残留のリスク)。psutil.children(recursive=True) で
    子孫を列挙し、まず全体にterminateを送り、生存していればkillへ
    escalationする(tail_monitor側の待機ロジックと対になる、木構造版の
    段階的終了)。
    """
    try:
        import psutil
    except ImportError:
        return
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    children = parent.children(recursive=True)
    procs = children + [parent]
    for p in procs:
        try:
            p.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    _, alive = psutil.wait_procs(procs, timeout=5)
    for p in alive:
        try:
            p.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    if alive:
        psutil.wait_procs(alive, timeout=5)


def terminate_run(run_id: int, proc: "subprocess.Popen | None" = None) -> None:
    """run の学習プロセスを停止する。proc 保有時はそのPIDから、無ければ config
    パスでOS検索したPIDから、プロセスツリー全体(子孫含む)を終了する。
    """
    if proc is not None:
        try:
            _terminate_process_tree(proc.pid)
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass
        return
    cfg_path = str(run_dir(run_id) / "config.toml")
    try:
        import psutil
        for p in psutil.process_iter(["pid", "cmdline"]):
            try:
                cl = " ".join(str(a) for a in (p.info.get("cmdline") or []))
                if cfg_path in cl:
                    _terminate_process_tree(p.info["pid"])
            except (psutil.NoSuchProcess, psutil.AccessDenied, Exception):
                continue
    except Exception:
        pass
