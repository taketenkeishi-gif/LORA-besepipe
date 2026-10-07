"""学習 run の実行時基盤（プロセス管理・監視・推定・成果物登録）。

Backend 実装（app.training.backends.*）はここを利用してサブプロセスを
起動・監視する。API 層（app.routers.training）はこのモジュールを直接操作
せず、Backend 経由でのみ関与する。
"""
from __future__ import annotations

from . import estimation, log_parser, monitor, queue, state
from .estimation import (
    DEFAULT_BASE_SIT,
    DEFAULT_BATCH,
    GPU_BASE_SIT,
    OPT_FACTOR,
    backfill_calib_from_runs,
    best_sec_per_step,
    calib_key,
    db_sec_per_step,
    detect_gpu_name,
    gpu_base_sit,
    live_sec_per_step,
    load_calib,
    record_step_timing,
    save_calib,
    sec_factor_for_run,
)
from .lora_export import auto_register_lora_asset, export_lora_to_output_dir
from .queue import (
    cancel_queued,
    current_running,
    dispatch_next,
    is_gpu_busy,
    list_queue,
    queue_position,
)
from .monitor import (
    MUSUBI_PREVIEW_SUPPORTED_FAMILIES,
    fail_run,
    proc_alive,
    reattach_active_runs,
    reconstruct_run_state,
    retry_preview_for_checkpoint,
    scan_and_register_artifacts,
    tail_monitor,
)
from .state import (
    IMAGE_EXTS,
    RUNNER_LOCK,
    RUNNER_PROCESSES,
    RUNNER_THREADS,
    STEP_TIMING,
    ensure_runner,
    latest_run,
    run_dir,
    set_project_status,
    terminate_run,
)

__all__ = [
    "estimation", "log_parser", "monitor", "queue", "state",
    # state
    "RUNNER_LOCK", "RUNNER_THREADS", "RUNNER_PROCESSES", "STEP_TIMING", "IMAGE_EXTS",
    "ensure_runner", "terminate_run", "run_dir", "latest_run", "set_project_status",
    # monitor
    "MUSUBI_PREVIEW_SUPPORTED_FAMILIES",
    "fail_run", "proc_alive", "tail_monitor", "reattach_active_runs",
    "reconstruct_run_state", "scan_and_register_artifacts", "retry_preview_for_checkpoint",
    # estimation
    "DEFAULT_BATCH", "DEFAULT_BASE_SIT", "GPU_BASE_SIT", "OPT_FACTOR",
    "detect_gpu_name", "gpu_base_sit", "calib_key", "load_calib", "save_calib",
    "backfill_calib_from_runs", "record_step_timing", "live_sec_per_step",
    "best_sec_per_step", "db_sec_per_step", "sec_factor_for_run",
    # lora_export
    "auto_register_lora_asset", "export_lora_to_output_dir",
    # queue
    "is_gpu_busy", "queue_position", "list_queue", "current_running",
    "dispatch_next", "cancel_queued",
]
