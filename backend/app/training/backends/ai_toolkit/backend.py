"""Ostris AI Toolkit backend used from the shared Trainer GUI.

The toolkit's own web UI is not part of the user workflow here; this backend
generates the same YAML job contract and streams its stdout into the existing
run monitor, queue, log viewer, and artifact registry.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import yaml

from ....db import get_conn
from ...base import PreparedRun, TrainingBackend, TrainingContext
from ...runtime import RUNNER_LOCK, RUNNER_PROCESSES, IMAGE_EXTS, detect_gpu_name, tail_monitor
from ...runtime.gpu_mapping import resolve_cuda_index
from ..shared.helpers import persist_dataset_snapshot, prepare_run_dir, resolve_dataset_source


def _toolkit_python(root: str) -> str:
    p = Path(root)
    for candidate in (p / ".venv" / "Scripts" / "python.exe", p / "venv" / "Scripts" / "python.exe"):
        if candidate.exists():
            return str(candidate)
    return ""


def is_ai_toolkit_ready(settings: dict[str, str]) -> bool:
    root = (settings.get("ai_toolkit_root") or "").strip()
    return bool(root and Path(root, "run.py").exists() and Path(_toolkit_python(root)).exists())


class AiToolkitBackend(TrainingBackend):
    display_name = "AI Toolkit"

    def is_ready(self, settings: dict[str, str]) -> bool:
        return is_ai_toolkit_ready(settings)

    def prepare(self, ctx: TrainingContext) -> PreparedRun:
        cfg, settings = ctx.cfg, ctx.settings
        root = settings.get("ai_toolkit_root", "").strip()
        python_exe = _toolkit_python(root)
        if not python_exe:
            raise RuntimeError("AI Toolkit の .venv\\Scripts\\python.exe が見つかりません")

        conn = get_conn()
        project = conn.execute(
            "SELECT name, dataset_dir FROM projects WHERE id = ?", (ctx.project_id,)
        ).fetchone()
        conn.close()
        if not project:
            raise RuntimeError("プロジェクトが見つかりません")
        dataset_dir, snapshot = resolve_dataset_source(
            project["dataset_dir"] or cfg.get("train_data_dir", ""),
            str(cfg.get("dataset_source") or "original"),
        )
        run_dir = prepare_run_dir(ctx.run_id, dataset_dir, int(cfg.get("repeats", 5)), project["name"])
        train_dir = run_dir / "train_data"
        image_count = sum(1 for p in train_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        if image_count == 0:
            raise RuntimeError(f"train_data に画像が見つかりません (dataset_dir={dataset_dir})")
        persist_dataset_snapshot(ctx.run_id, {**snapshot, "dataset_staged_image_count": image_count})

        # AI Toolkit counts optimizer steps, while this application's UI uses epochs.
        steps = max(1, (image_count * int(cfg.get("repeats", 5)) + max(1, int(cfg.get("train_batch_size", 1))) - 1)
                    // max(1, int(cfg.get("train_batch_size", 1)))) * int(cfg.get("epochs", 5))
        model = cfg.get("base_checkpoint_path", "")
        process: dict[str, Any] = {
            "type": "sd_trainer",
            "training_folder": str(run_dir / "output"),
            "device": "cuda:0",
            "network": {"type": "lora", "linear": int(cfg.get("rank", 16)), "linear_alpha": float(cfg.get("alpha", 8))},
            "save": {"dtype": str(cfg.get("save_precision", "fp16")), "save_every": max(1, steps // max(1, int(cfg.get("epochs", 5))))},
            "datasets": [{"folder_path": str(train_dir), "caption_ext": "txt", "cache_latents_to_disk": True,
                          "resolution": [int(cfg.get("resolution", 1024))]}],
            "train": {"batch_size": max(1, int(cfg.get("train_batch_size", 1))), "steps": steps,
                      "gradient_checkpointing": bool(cfg.get("gradient_checkpointing", True)),
                      "optimizer": str(cfg.get("optimizer", "adamw8bit")).lower(),
                      "lr": float(cfg.get("learning_rate", 1e-4)), "dtype": str(cfg.get("mixed_precision", "bf16")),
                      "noise_scheduler": "flowmatch", "disable_sampling": True},
            "model": {"name_or_path": model, "quantize": True, "quantize_te": True, "low_vram": True},
        }
        if ctx.model_family == "krea2" and "turbo" in Path(model).name.lower():
            process["model"]["assistant_lora_path"] = "ostris/krea2_turbo_training_adapter/krea2_turbo_training_adapter_v1.safetensors"
        config_path = run_dir / "ai_toolkit_config.yaml"
        config_path.write_text(yaml.safe_dump({"job": "extension", "config": {"name": cfg.get("output_name", "lora_output"), "process": [process]}}, sort_keys=False), encoding="utf-8")
        log_path = run_dir / "logs" / "training.log"
        conn = get_conn()
        conn.execute("UPDATE training_runs SET log_path = ? WHERE id = ?", (str(log_path), ctx.run_id))
        conn.commit(); conn.close()
        gpu_device_id = int(cfg.get("gpu_device_id", 1))
        gpu_mapping = resolve_cuda_index(gpu_device_id, python_exe)
        env = {
            **os.environ,
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": str(gpu_mapping["cuda_index"]),
        }
        return PreparedRun(run_dir=run_dir, config_path=config_path, log_path=log_path,
                           cmd=[python_exe, str(Path(root) / "run.py"), str(config_path)], cwd=root,
                           python_exe=python_exe, env=env,
                           extra={"gpu_device_id": gpu_device_id, "gpu_mapping": gpu_mapping, "gpu_mapping_verified": bool(gpu_mapping["verified"])})

    def train(self, ctx: TrainingContext, prepared: PreparedRun) -> None:
        gpu_id = int(prepared.extra.get("gpu_device_id", 1))
        prepared.log_path.parent.mkdir(parents=True, exist_ok=True)
        prepared.log_path.touch(exist_ok=True)
        try:
            if not prepared.extra.get("gpu_mapping_verified", False):
                raise RuntimeError(f"GPU物理順序とCUDA列挙順を照合できないため開始を拒否: {prepared.extra.get('gpu_mapping', {})}")
            prepared.log_path.write_text("[AI_TOOLKIT] cmd=" + " ".join(prepared.cmd) + "\n", encoding="utf-8")
            with prepared.log_path.open("ab") as log:
                proc = subprocess.Popen(prepared.cmd, stdout=log, stderr=subprocess.STDOUT,
                                        cwd=prepared.cwd, env=prepared.env)
                with RUNNER_LOCK:
                    RUNNER_PROCESSES[ctx.project_id] = proc
                tail_monitor(ctx.run_id, ctx.project_id, proc, 1.0, detect_gpu_name(gpu_id), ctx.model_family)
        except Exception as exc:
            prepared.log_path.write_text(prepared.log_path.read_text(encoding="utf-8") + f"\n[ERROR] {exc}\n", encoding="utf-8")
            conn = get_conn(); conn.execute("UPDATE training_runs SET status='error', failure_code='TRAINER_LAUNCH_FAILED', failure_message=?, failure_stage='training', failure_at=CURRENT_TIMESTAMP WHERE id=?", (str(exc), ctx.run_id)); conn.execute("UPDATE projects SET status='idle' WHERE id=?", (ctx.project_id,)); conn.commit(); conn.close()
