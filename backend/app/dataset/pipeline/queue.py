"""複数プロジェクトの前処理パイプラインを順番に処理するキュー。

同時実行は不要という要件のため、ワーカースレッドは常に1つだけ。複数
プロジェクトから同時にパイプラインが要求されても、キューに積まれた順に
1件ずつ run_pipeline() を実行する（同一プロセス内の待ち行列であり、
永続化やプロセス跨ぎの分散処理は行わない）。
"""
from __future__ import annotations

import logging
import queue as _queue
import threading
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_JOB_QUEUE: "_queue.Queue[_PipelineJob]" = _queue.Queue()
_QUEUED_PROJECT_IDS: set[int] = set()
_CANCELLED_PROJECT_IDS: set[int] = set()
_WORKER_LOCK = threading.Lock()
_WORKER_STARTED = False


@dataclass
class _PipelineJob:
    project_id: int
    args: tuple[Any, ...] = field(default_factory=tuple)
    kwargs: dict[str, Any] = field(default_factory=dict)


def _worker_loop() -> None:
    from .runner import _PIPELINE_STATUS, run_pipeline  # noqa: PLC0415 循環回避（遅延import）

    while True:
        job = _JOB_QUEUE.get()
        try:
            _QUEUED_PROJECT_IDS.discard(job.project_id)
            if job.project_id in _CANCELLED_PROJECT_IDS:
                _CANCELLED_PROJECT_IDS.discard(job.project_id)
                _PIPELINE_STATUS[job.project_id] = {
                    "status": "cancelled",
                    "message": "キュー内でキャンセルされました",
                    "done": 0, "total": 0, "failed": 0, "skipped": 0, "success": 0,
                    "current_step": "", "current_step_name": "", "current_image": "",
                    "items": [],
                }
                continue
            run_pipeline(job.project_id, *job.args, **job.kwargs)
        except Exception:  # noqa: BLE001 — ワーカースレッド自体は止めない
            logger.exception("pipeline queue worker failed for project %s", job.project_id)
        finally:
            _JOB_QUEUE.task_done()


def _ensure_worker() -> None:
    global _WORKER_STARTED
    with _WORKER_LOCK:
        if not _WORKER_STARTED:
            t = threading.Thread(target=_worker_loop, daemon=True, name="pipeline-queue-worker")
            t.start()
            _WORKER_STARTED = True


def enqueue(project_id: int, *args: Any, **kwargs: Any) -> int:
    """パイプラインジョブをキューへ積む。戻り値はキュー内の待ち件数（自分を含む）。"""
    from .runner import mark_queued  # noqa: PLC0415

    _ensure_worker()
    _QUEUED_PROJECT_IDS.add(project_id)
    _JOB_QUEUE.put(_PipelineJob(project_id=project_id, args=args, kwargs=kwargs))
    position = _JOB_QUEUE.qsize()
    mark_queued(project_id, position)
    return position


def is_queued(project_id: int) -> bool:
    return project_id in _QUEUED_PROJECT_IDS


def cancel_queued(project_id: int) -> bool:
    """まだ実行開始していないキュー内ジョブをキャンセルする。実行中/未キュー時は False。"""
    if project_id in _QUEUED_PROJECT_IDS:
        _CANCELLED_PROJECT_IDS.add(project_id)
        return True
    return False


def queue_size() -> int:
    return _JOB_QUEUE.qsize()
