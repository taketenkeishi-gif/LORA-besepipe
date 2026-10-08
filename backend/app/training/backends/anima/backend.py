"""ANIMA エンジン — CircleStone Labs製アニメ特化DiTモデル。

SDXL/SD1.xと同じkohya_ss(sd-scripts)ツールを使うが、`anima_train_network.py`は
`sdxl_train_network.py`/`train_network.py`とは非互換の専用引数体系
(DiT本体 + Qwen3-0.6Bテキストエンコーダ + Qwen-Image VAE + LLM Adapter構成、
network_module=networks.lora_anima)を持つため、SDXLBackendとは別の専用Backend
として実装する(参照: sd-scripts/docs/anima_train_network.md)。

kohya_ss自体はSDXLBackendが使うものと同じインストール(python_exe/kohya_root)を
再利用する(is_kohya_ready/resolve_pythonはsdxl/backend.pyの実装をそのまま使う。
kohya_ss全体に対する判定であり、SDXL固有のロジックではないため)。
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from ....db import get_conn
from ...base import PreparedRun, TrainingBackend, TrainingContext
from ...snapshot_inputs import fetch_snapshot_inputs, verify_snapshot_input
from ...runtime import (
    RUNNER_LOCK,
    RUNNER_PROCESSES,
    STEP_TIMING,
    IMAGE_EXTS,
    detect_gpu_name,
    tail_monitor,
)
from ...runtime.gpu_mapping import resolve_cuda_index
from ..shared.helpers import persist_dataset_snapshot, prepare_run_dir, resolve_dataset_source
from ..sdxl.backend import is_kohya_ready, resolve_python

_ANIMA_MIN_FREE_VRAM_MB = 20_000


def _uses_adaptive_unit_lr(optimizer_name: str) -> bool:
    normalized = optimizer_name.strip().lower()
    return normalized.startswith("prodigy") or normalized.startswith("dadapt")


def _assert_anima_gpu_admission(physical_index: int = 1, memory_mode: str = "standard", allow_interactive_sharing: bool = False) -> None:
    from ...runtime.anima_admission import required_free_mb,query_occupancy,wait_reason
    if physical_index != 1:raise RuntimeError("Anima学習は物理GPU1で実行します")
    free=int(subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True,timeout=5).strip())
    if free < required_free_mb(memory_mode):raise RuntimeError(f"省VRAM設定の起動時確認容量が不足しています: 空き{free}MiB")
    reason=wait_reason(query_occupancy(1),allow_interactive_sharing)
    if reason:raise RuntimeError(reason)


def get_anima_script(kohya_root: str) -> str | None:
    p = Path(kohya_root)
    for candidate in (p / "anima_train_network.py", p / "sd-scripts" / "anima_train_network.py"):
        if candidate.exists():
            return str(candidate)
    return None


def write_anima_toml_config(run_dir: Path, cfg: dict) -> Path:
    """anima_train_network.py 用 TOML 設定ファイルを生成する。

    train_network.py がベースのため train_data_dir 方式のデータセット指定
    (SDXLBackendと同じ簡易形式)が引き続き使える。Anima固有のキーのみ追加する
    (qwen3/vae/timestep_sampling/discrete_flow_shift、network_module=lora_anima)。
    """
    config_path = run_dir / "config.toml"

    def q(v: str) -> str:
        return v.replace("\\", "\\\\").replace('"', '\\"')

    resolution = int(cfg["resolution"])
    lr = cfg.get("learning_rate", 1e-4)

    qwen3_path = cfg.get("text_encoder_path", "").strip()
    vae_path = cfg.get("vae_path", "").strip()
    if not qwen3_path:
        raise RuntimeError("ANIMA学習にはQwen3-0.6Bテキストエンコーダのパス指定が必須です(テキストエンコーダパス欄)")
    if not vae_path:
        raise RuntimeError("ANIMA学習にはQwen-Image VAEのパス指定が必須です(VAEパス欄)")

    lines = [
        f'pretrained_model_name_or_path = "{q(cfg["base_checkpoint_path"])}"',
        f'qwen3 = "{q(qwen3_path)}"',
        f'vae = "{q(vae_path)}"',
        f'train_data_dir = "{q(cfg["train_data_dir"])}"',
        f'output_dir = "{q(cfg["output_dir"])}"',
        f'output_name = "{q(cfg["output_name"])}"',
        'save_model_as = "safetensors"',
        f'resolution = "{resolution},{resolution}"',
        'network_module = "networks.lora_anima"',
        *([f"blocks_to_swap = {12 if cfg.get('training_memory_mode') == 'balanced' else 26}"] if cfg.get('training_memory_mode', 'standard') != 'standard' else []),
        f'network_dim = {int(cfg["rank"])}',
        f'network_alpha = {float(cfg["alpha"])}',
        *([f'network_weights = "{q(cfg["resume_from_checkpoint"])}"'] if cfg.get("resume_from_checkpoint") else []),
        f'max_train_epochs = {int(cfg["epochs"])}',
        f'save_every_n_epochs = {int(cfg["save_every_n_epochs"])}',
        f'train_batch_size = {int(cfg.get("train_batch_size", 2))}',
        'caption_extension = ".txt"',
        f'optimizer_type = "{q(cfg.get("optimizer", "AdamW8bit"))}"',
        f'lr_scheduler = "{q(cfg.get("scheduler", "constant"))}"',
        f'learning_rate = {lr}',
        'timestep_sampling = "sigmoid"',
        'discrete_flow_shift = 1.0',
        f'mixed_precision = "{cfg.get("mixed_precision", "bf16")}"',
        f'save_precision = "{cfg.get("save_precision", "bf16")}"',
        f'gradient_checkpointing = {"true" if cfg.get("gradient_checkpointing", True) else "false"}',
        f'cache_latents = {"true" if cfg.get("cache_latents", True) else "false"}',
        f'cache_latents_to_disk = {"true" if cfg.get("cache_latents", True) and cfg.get("cache_latents_to_disk", False) else "false"}',
        f'network_train_unet_only = {"true" if cfg.get("network_train_unet_only", True) else "false"}',
        *(
            ['cache_text_encoder_outputs = true', 'cache_text_encoder_outputs_to_disk = true']
            if cfg.get("network_train_unet_only", True) else []
        ),
        'vae_disable_cache = true',
        # The trainer documents this still-image path as numerically equivalent
        # while materially reducing Qwen-Image VAE peak memory.
        'qwen_image_vae_2d = true',
        f'persistent_data_loader_workers = {"true" if cfg.get("persistent_data_loader_workers", True) else "false"}',
        f'max_data_loader_n_workers = {int(cfg.get("max_data_loader_n_workers", 4))}',
        'enable_bucket = true',
        'min_bucket_reso = 256',
        'max_bucket_reso = 2048',
        'bucket_reso_steps = 64',
        f'logging_dir = "{q(cfg["logs_dir"])}"',
    ]

    llm_adapter_path = cfg.get("llm_adapter_path", "").strip()
    if llm_adapter_path:
        lines.append(f'llm_adapter_path = "{q(llm_adapter_path)}"')
    t5_tokenizer_path = cfg.get("t5_tokenizer_path", "").strip()
    if t5_tokenizer_path:
        lines.append(f't5_tokenizer_path = "{q(t5_tokenizer_path)}"')

    from ...advanced import apply_to_toml
    lines = apply_to_toml(lines, cfg, "anima")
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return config_path


class AnimaBackend(TrainingBackend):
    display_name = "Anima Engine"

    def is_ready(self, settings: dict[str, str]) -> bool:
        if not is_kohya_ready(settings):
            return False
        kohya_root = settings.get("kohya_root", "").strip()
        return get_anima_script(kohya_root) is not None

    def supports_resume(self) -> bool:
        """anima_train_network.pyもtrain_network.py同様--network_weightsによる
        既存LoRA重みからの継続学習に対応している(sd-scripts/docs/
        anima_train_network.md: 引数体系はtrain_network.pyベースと明記)。"""
        return True

    @staticmethod
    def _resolve_dependency(settings: dict[str, str], configured: str, filename: str, subdirs: tuple[str, ...]) -> str:
        """Resolve Anima's Qwen dependencies without silently substituting a model."""
        explicit = configured.strip()
        if explicit:
            if Path(explicit).is_file():
                return explicit
            raise RuntimeError(f"指定されたAnima依存ファイルが見つかりません: {explicit}")
        comfy_root = Path(settings.get("comfyui_root", "").strip())
        for subdir in subdirs:
            candidate = comfy_root / "models" / subdir / filename
            if candidate.is_file():
                return str(candidate)
        raise RuntimeError(f"Anima依存ファイルが見つかりません: {filename} (ComfyUI modelsを確認してください)")

    def prepare(self, ctx: TrainingContext) -> PreparedRun:
        project_id, run_id, settings, cfg = ctx.project_id, ctx.run_id, ctx.settings, ctx.cfg
        from ...learning_rates import resolve_learning_rates
        resolve_learning_rates(cfg, reject_conflict=True)
        python_exe = resolve_python(settings)
        kohya_root = settings.get("kohya_root", "")

        train_script = get_anima_script(kohya_root)
        if not train_script:
            raise RuntimeError("anima_train_network.py が見つかりません(kohya_ssをANIMA対応版に更新してください)")

        conn_p = get_conn()
        project_row = conn_p.execute(
            "SELECT name, dataset_dir, outputs_dir FROM projects WHERE id = ?", (project_id,)
        ).fetchone()
        conn_p.close()

        project_name = project_row["name"]
        dataset_source = str(cfg.get("dataset_source") or "original")
        dataset_dir, dataset_snapshot = resolve_dataset_source(
            project_row["dataset_dir"] or cfg.get("train_data_dir", ""), dataset_source
        )
        repeats = int(cfg.get("repeats", 5))
        snapshot_id = cfg.get("dataset_snapshot_id")
        snapshot_asset_paths: list[tuple[str, str, str]] | None = None
        snapshot_info: dict = {}
        if snapshot_id is not None:
            conn_snapshot = get_conn()
            snapshot_row = conn_snapshot.execute(
                "SELECT snapshot_hash, status, item_count FROM basepipe_dataset_snapshots WHERE id = ? AND project_id = ?",
                (int(snapshot_id), project_id),
            ).fetchone()
            snapshot_rows = fetch_snapshot_inputs(conn_snapshot, int(snapshot_id))
            conn_snapshot.close()
            if snapshot_row is None or snapshot_row["status"] != "sealed":
                raise RuntimeError(f"Dataset Snapshot #{snapshot_id} がsealed状態ではありません")
            for snapshot_input in snapshot_rows:
                integrity_ok, integrity_detail = verify_snapshot_input(snapshot_input)
                if not integrity_ok:
                    raise RuntimeError(
                        f"Dataset Snapshot #{snapshot_id} input #{snapshot_input['asset_id']} integrity error: {integrity_detail}"
                    )
            snapshot_asset_paths = [
                (
                    str(row["file_path"]),
                    str(row["caption_at_snapshot"] or ""),
                    str(row.get("snapshot_sha256") or row["content_sha256"]),
                )
                for row in snapshot_rows
            ]
            snapshot_info = {
                "dataset_snapshot_id": int(snapshot_id),
                "dataset_snapshot_hash": snapshot_row["snapshot_hash"],
                "dataset_snapshot_item_count": int(snapshot_row["item_count"]),
            }

        def _has_images(d: str) -> bool:
            p = Path(d)
            return p.exists() and any(
                f.suffix.lower() in IMAGE_EXTS for f in p.iterdir() if f.is_file()
            )

        if dataset_source == "original" and not _has_images(dataset_dir):
            conn_items = get_conn()
            rows = conn_items.execute(
                "SELECT file_path FROM dataset_items WHERE project_id = ? LIMIT 200", (project_id,)
            ).fetchall()
            conn_items.close()
            existing = [r["file_path"] for r in rows if Path(r["file_path"]).exists()]
            if existing:
                dataset_dir = str(Path(existing[0]).parent)
                dataset_snapshot = {"dataset_source": "original", "resolved_dataset_dir": dataset_dir}

        from ...runtime.state import run_dir as canonical_run_dir
        run_dir = prepare_run_dir(run_id, dataset_dir, repeats, project_name, asset_paths=snapshot_asset_paths, runtime_root=canonical_run_dir(run_id).parent)
        train_data_root = run_dir / "train_data"
        image_count = sum(
            1 for p in train_data_root.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
        if image_count == 0:
            raise RuntimeError(f"train_data に画像が見つかりません (dataset_dir={dataset_dir})")
        if snapshot_id is not None:
            expected_count = int(snapshot_info["dataset_snapshot_item_count"])
            caption_count = sum(1 for p in train_data_root.rglob("*.txt") if p.is_file())
            if image_count != expected_count or caption_count != expected_count:
                raise RuntimeError(
                    f"Dataset Snapshot staging count mismatch: expected={expected_count}, "
                    f"images={image_count}, captions={caption_count}"
                )
        persist_dataset_snapshot(run_id, {**dataset_snapshot, **snapshot_info, "dataset_staged_image_count": image_count})

        optimizer_name = cfg.get("optimizer", "AdamW8bit")
        raw_lr = cfg.get("learning_rate", 1e-4)
        effective_lr = 1.0 if _uses_adaptive_unit_lr(str(optimizer_name)) else raw_lr

        text_encoder_path = self._resolve_dependency(
            settings, str(cfg.get("text_encoder_path", "")), "qwen_3_06b_base.safetensors", ("text_encoders", "clip")
        )
        vae_path = self._resolve_dependency(
            settings, str(cfg.get("vae_path", "")), "qwen_image_vae.safetensors", ("vae", "diffusion_models")
        )
        snapshot_info.update({
            "resolved_text_encoder_path": text_encoder_path,
            "resolved_vae_path": vae_path,
        })
        data_workers = int(cfg.get("advanced", {}).get("max_data_loader_n_workers", 0 if cfg.get("cache_latents",True) and cfg.get("network_train_unet_only",True) else cfg.get("max_data_loader_n_workers",4)))
        anima_cfg = {
            "advanced": cfg.get("advanced", {}),
            "base_checkpoint_path": cfg.get("base_checkpoint_path", ""),
            "text_encoder_path": text_encoder_path,
            "vae_path": vae_path,
            "train_data_dir": str(run_dir / "train_data"),
            "output_dir": str(run_dir / "output"),
            "output_name": cfg.get("output_name", "lora_output"),
            "resolution": int(cfg.get("resolution", 1024)),
            "train_batch_size": int(cfg.get("train_batch_size", 2) or 2),
            "learning_rate": effective_lr,
            "rank": int(cfg.get("rank", 16)),
            "alpha": float(cfg.get("alpha", 1)),
            "epochs": int(cfg.get("epochs", 10)),
            "save_every_n_epochs": int(cfg.get("save_every_n_epochs", 1)),
            "optimizer": optimizer_name,
            "scheduler": cfg.get("scheduler", "constant"),
            "mixed_precision": cfg.get("mixed_precision", "bf16"),
            "save_precision": cfg.get("save_precision", "bf16"),
            "gradient_checkpointing": cfg.get("gradient_checkpointing", True),
            "network_train_unet_only": bool(cfg.get("network_train_unet_only", True)),
            "cache_text_encoder_outputs": bool(cfg.get("network_train_unet_only", True)),
            "cache_text_encoder_outputs_to_disk": bool(cfg.get("network_train_unet_only", True)),
            "training_memory_mode": cfg.get("training_memory_mode", "standard"),
            "max_data_loader_n_workers": data_workers,
            "persistent_data_loader_workers": data_workers > 0 and bool(cfg.get("persistent_data_loader_workers",True)),
            "cache_latents": bool(cfg.get("cache_latents", True)),
            "cache_latents_to_disk": bool(cfg.get("cache_latents_to_disk", False)),
            "qwen_image_vae_2d": True,
            "logs_dir": str(run_dir / "logs"),
            "resume_from_checkpoint": cfg.get("resume_from_checkpoint", ""),
        }

        if "advanced" not in cfg:
            anima_cfg.pop("advanced", None)
        config_path = write_anima_toml_config(run_dir, anima_cfg)
        # Preview inference is delegated to ComfyUI at the existing epoch callback.
        cfg['preview_backend']='comfyui'
        persist_dataset_snapshot(run_id, {'preview_backend':'comfyui'})
        log_path = run_dir / "logs" / "training.log"

        gpu_device_id = int(cfg.get("gpu_device_id", 1))
        gpu_mapping = resolve_cuda_index(gpu_device_id, python_exe)
        env = {
            **os.environ,
            # gpu_device_id is the physical nvidia-smi index; CUDA can enumerate differently.
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": subprocess.check_output(["nvidia-smi",f"--id={gpu_device_id}","--query-gpu=uuid","--format=csv,noheader"],text=True,timeout=5).strip(),
        }

        conn = get_conn()
        conn.execute(
            "UPDATE training_runs SET log_path = ? WHERE id = ?",
            (str(log_path), run_id),
        )
        conn.commit()
        conn.close()

        cmd = [python_exe, train_script, "--config_file", str(config_path)]
        from ...runtime import preview_gpu
        preview_gpu.set_mode(run_dir, str(cfg.get("preview_gpu") or "gpu0"))  # read by the epoch hook: 3060 = keep training while previews render
        if cfg.get("training_memory_mode", "standard") in ("low_vram","balanced","standard"):
            # Fraction is a cap, not a reservation. Allocations grow with actual demand.
            launcher = run_dir / "bounded_training_entry.py"
            launcher.write_text("import sys,runpy,torch,json,subprocess,os\n" +
                "assert torch.cuda.device_count()==1\n" +
                f"assert str(torch.cuda.get_device_properties(0).uuid).lower().replace('gpu-','')=={env['CUDA_VISIBLE_DEVICES'].lower().replace('gpu-','')!r}\n" +
                "free,total=torch.cuda.mem_get_info(0)\n" +
                "system_free=int(subprocess.check_output(['nvidia-smi','--id='+os.environ['CUDA_VISIBLE_DEVICES'],'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True,timeout=5).strip())*1024**2\n" +
                "free=min(free,system_free)\n" +
                "budget=min(free-2*1024**3,int(total*0.75))\n" +
                "assert budget>=6*1024**3, 'Insufficient free VRAM for bounded offload profile'\n" +
                "torch.cuda.set_per_process_memory_fraction(budget/total,0)\n" +
                "torch.set_num_threads(4)\n" +
                "print('VRAM_BUDGET '+json.dumps({'initial_free':free,'limit':budget,'reservation':False}),flush=True)\n" +
                f"sys.path.insert(0,{str(Path(train_script).parent)!r})\n" +
                "import importlib.util\n" +
                f"hook_spec=importlib.util.spec_from_file_location('studio_comfy_epoch',{str(Path(__file__).resolve().parents[2] / 'runtime' / 'comfy_epoch_hook.py')!r})\n" +
                "hook=importlib.util.module_from_spec(hook_spec);hook_spec.loader.exec_module(hook)\n" +
                f"hook.install({str(run_dir)!r},{run_id},'http://127.0.0.1:5175')\n" +
                f"sys.argv=[{train_script!r},'--config_file',{str(config_path)!r}]\n" +
                f"try:\n runpy.run_path({train_script!r},run_name='__main__')\nfinally:\n print('VRAM_RESULT '+json.dumps({{'peak_allocated':torch.cuda.max_memory_allocated(0),'peak_reserved':torch.cuda.max_memory_reserved(0)}}),flush=True)\n",encoding="utf8")
            cmd = [python_exe, str(launcher), "--config_file", str(config_path)]
            env.update(OMP_NUM_THREADS="4", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")

        return PreparedRun(
            run_dir=run_dir, config_path=config_path, log_path=log_path,
            cmd=cmd, cwd=kohya_root, python_exe=python_exe, env=env,
            extra={
                "base_checkpoint_path": cfg.get("base_checkpoint_path", ""),
                "trainer_script_path": train_script,
                "python_executable": python_exe,
                "gpu_device_id": gpu_device_id,
                "gpu_mapping": gpu_mapping,
                "gpu_mapping_verified": bool(gpu_mapping["verified"]),
                "anima_cfg": anima_cfg,
            },
        )

    def train(self, ctx: TrainingContext, prepared: PreparedRun) -> None:
        project_id, run_id = ctx.project_id, ctx.run_id
        base_ckpt = prepared.extra.get("base_checkpoint_path", "")
        log_path = prepared.log_path

        header = (
            f"[ANIMA] python={prepared.python_exe}\n"
            f"[ANIMA] script={prepared.cmd[1] if len(prepared.cmd) > 1 else ''}\n"
            f"[ANIMA] base_checkpoint={base_ckpt}\n"
            f"[ANIMA] text_encoder={prepared.extra.get('anima_cfg', {}).get('text_encoder_path', '')}\n"
            f"[ANIMA] vae={prepared.extra.get('anima_cfg', {}).get('vae_path', '')}\n"
            f"[ANIMA] config={prepared.config_path}\n"
            f"[ANIMA] CUDA_DEVICE_ORDER={prepared.env.get('CUDA_DEVICE_ORDER') if prepared.env else ''}\n"
            f"[ANIMA] CUDA_VISIBLE_DEVICES={prepared.env.get('CUDA_VISIBLE_DEVICES') if prepared.env else ''}\n"
            f"[ANIMA] cmd={' '.join(prepared.cmd)}\n\n"
        )
        try:
            if not prepared.extra.get("gpu_mapping_verified", False):
                raise RuntimeError(f"GPU物理順序とCUDA列挙順を照合できないため開始を拒否: {prepared.extra.get('gpu_mapping', {})}")
            with open(log_path, "w", encoding="utf-8") as lf:
                lf.write(header)
        except OSError:
            pass

        gpu_name = detect_gpu_name(int(prepared.extra.get("gpu_device_id", 1)))
        STEP_TIMING.pop(project_id, None)

        log_fh = None
        proc: subprocess.Popen | None = None
        try:
            _assert_anima_gpu_admission(int(prepared.extra.get("gpu_device_id", 1)),str(ctx.cfg.get("training_memory_mode", "standard")),bool(ctx.cfg.get("allow_interactive_gpu_sharing")))
            log_fh = open(log_path, "ab")
            proc = subprocess.Popen(
                prepared.cmd,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                cwd=prepared.cwd,
                env=prepared.env,
                creationflags=(subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS) if os.name=='nt' and ctx.cfg.get('allow_interactive_gpu_sharing') else 0,
            )
            with RUNNER_LOCK:
                RUNNER_PROCESSES[project_id] = proc
            tail_monitor(run_id, project_id, proc, 1.0, gpu_name, "anima")
        except Exception as exc:
            try:
                with open(log_path, "a", encoding="utf-8") as lf:
                    lf.write(f"\n[ERROR] {exc}\n")
            except OSError:
                pass
            conn = get_conn()
            conn.execute(
                "UPDATE training_runs SET status='error', updated_at=CURRENT_TIMESTAMP, "
                "failure_code='TRAINER_LAUNCH_FAILED', failure_message=?, failure_stage='training', "
                "failure_at=CURRENT_TIMESTAMP WHERE id=?",
                (str(exc), run_id),
            )
            conn.execute("UPDATE projects SET status='idle' WHERE id=?", (project_id,))
            conn.commit()
            conn.close()
            with RUNNER_LOCK:
                RUNNER_PROCESSES.pop(project_id, None)
        finally:
            if log_fh:
                try:
                    log_fh.close()
                except Exception:
                    pass
