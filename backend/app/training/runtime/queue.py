"""GPU単一共有リソースを前提としたTraining FIFO Queue。

新規テーブルは追加せず、既存 training_runs.status に 'queued' を追加して
FIFOの待ち行列を表現する。挿入順序は INTEGER PRIMARY KEY AUTOINCREMENT の
id が保証するため、"ORDER BY id ASC" がそのままFIFO順になる（新規の
queue_position列は不要）。

用語について: 本モジュールが導入するのは 'queued' 状態のみ。既存の
'training'(実行中)/'completed'(完了)/'error'(失敗)/'paused'(中断・キャンセル済み)
という語彙は変更しない（Frontend/既存テスト/既存DBとの後方互換を最優先するため）。
"starting" という中間状態は永続化しない — dispatch_next() がqueued行を
'training' へ昇格させ、同一トランザクション内でensure_runner呼び出しまで
行うため、位置づけとしては一瞬で完了する（別途DB状態として観測可能な
"starting" ウィンドウを作らない）。

GPU単一実行の保証は、同じSQLiteトランザクション内で training_runs と
basepipe_character_generation_runs の双方を確認して判定する。
"""
from __future__ import annotations

import threading
from typing import Callable

from ...db import get_conn

# dispatch_next() の「is_gpu_busy確認 → queued行の昇格」はSELECT+UPDATEの
# 2ステップに分かれており、この間にロックを取らないと、ほぼ同時に複数スレッドから
# dispatch_next()が呼ばれた場合(例: 1つのrunが完了した直後の finally と、
# 同時に別projectから投げられた/startのdispatchが競合する等)、両方が
# is_gpu_busy()==Falseを観測してから昇格しようとし、2つのrunが同時に
# 'training'へ昇格してしまうレースコンディションが理論上あり得る
# (単一プロセス内の複数スレッド間の競合。プロセスをまたぐ排他はそもそも
# 単一Backendプロセス構成のため考慮不要)。
# 同一プロセス内で使い回されている RUNNER_LOCK ではなく専用のロックを使う
# (RUNNER_LOCK は RUNNER_PROCESSES 辞書アクセス用の別関心事のロックであり、
# 混用すると意図しないデッドロック要因になり得るため独立させる)。
_DISPATCH_LOCK = threading.Lock()


def is_gpu_busy(conn) -> bool:
    """TrainingまたはCharacter GenerationがGPUを所有中か。"""
    row = conn.execute("SELECT 1 FROM training_runs WHERE status = 'training' LIMIT 1").fetchone()
    if row is not None:
        return True
    if conn.execute("SELECT 1 FROM preview_jobs WHERE status='running' LIMIT 1").fetchone() is not None:
        return True
    row = conn.execute(
        "SELECT 1 FROM basepipe_character_generation_runs "
        "WHERE status IN ('queued','running','running_qwen') LIMIT 1"
    ).fetchone()
    return row is not None


def queue_position(conn, run_id: int) -> int:
    """指定run_idのqueued行がFIFOの何番目(1始まり)かを返す。queuedでなければ0。"""
    row = conn.execute("SELECT id, status FROM training_runs WHERE id = ?", (run_id,)).fetchone()
    if row is None or row["status"] != "queued":
        return 0
    count = conn.execute(
        "SELECT COUNT(*) AS c FROM training_runs WHERE status = 'queued' AND id < ?",
        (run_id,),
    ).fetchone()["c"]
    return int(count) + 1


def list_queue(conn) -> list[dict]:
    """queued状態の全runをFIFO順(position付き)で返す。"""
    rows = conn.execute(
        "SELECT id, project_id, config_json, started_at FROM training_runs "
        "WHERE status = 'queued' ORDER BY id ASC"
    ).fetchall()
    result = []
    for i, r in enumerate(rows):
        result.append({
            "run_id": r["id"],
            "project_id": r["project_id"],
            "queue_position": i + 1,
            "queued_at": r["started_at"],
        })
    return result


def current_running(conn) -> dict | None:
    row = conn.execute(
        "SELECT id, project_id FROM training_runs WHERE status = 'training' "
        "ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return None
    return {"run_id": row["id"], "project_id": row["project_id"]}


def dispatch_next(
    ensure_runner_fn: Callable[[int], None],
    set_project_status_fn,
    admission_fn: Callable[[dict], bool] | None = None,
) -> int | None:
    """GPUが空いていれば、FIFO先頭のqueued runを 'training' へ昇格し起動する。

    呼び出しどころ:
      1. /training/start でqueuedへ挿入した直後（即座に空いていれば起動）
      2. /training/resume でqueuedへ挿入した直後
      3. 現在のrunが完了/失敗/キャンセルされた直後(_run_training_backendのfinally)
      4. Backend起動時（再起動後、runningが無くqueuedが残っている場合の復旧）

    戻り値: 昇格させたrun_id。何も昇格しなかった場合はNone。
    """
    with _DISPATCH_LOCK:
        conn = get_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            if is_gpu_busy(conn):
                return None
            row = conn.execute(
                "SELECT id, project_id, config_json FROM training_runs WHERE status = 'queued' "
                "ORDER BY id ASC LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            if admission_fn is not None and not admission_fn(dict(row)):
                return None
            run_id = int(row["id"])
            project_id = int(row["project_id"])
            conn.execute(
                "UPDATE training_runs SET status = 'training', updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (run_id,),
            )
            set_project_status_fn(conn, project_id, "training")
            conn.commit()
        finally:
            conn.close()

    ensure_runner_fn(project_id)
    return run_id


def cancel_queued(conn, run_id: int) -> bool:
    """まだ開始していない(status='queued')runをキャンセルする。

    subprocessは一切起動していないため、terminate等は不要。単純にDB状態を
    'paused'(既存語彙でのキャンセル済み相当)へ遷移させるだけでよい。
    後続のqueued runのqueue_positionはCOUNT(*)ベースで動的に算出するため、
    別途position再計算処理は不要（次回list_queue()呼び出し時に自動的に
    正しい値になる）。
    """
    row = conn.execute("SELECT status FROM training_runs WHERE id = ?", (run_id,)).fetchone()
    if row is None or row["status"] != "queued":
        return False
    conn.execute(
        "UPDATE training_runs SET status = 'paused', updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        (run_id,),
    )
    conn.commit()
    return True
