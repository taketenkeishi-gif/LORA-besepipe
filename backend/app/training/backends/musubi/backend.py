"""Musubi エンジン — Flux / Qwen-Image / KREA2 / Wan2.1 / HunyuanVideo / Z-Image 等。

内部実装は musubi-tuner を利用するが、これは実装詳細でありこの Backend の外
（API・UI）には露出しない。
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from ....db import get_conn
from ....trainers import get_runner, get_spec, list_specs
from ...base import PreparedRun, TrainingBackend, TrainingContext
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

_AUTO_LR_OPTIMIZERS = {"Prodigy", "DAdaptAdam", "DAdaptAdaGrad", "DAdaptSGD"}


def is_musubi_ready(settings: dict[str, str]) -> bool:
    musubi_root = settings.get("musubi_root", "").strip()
    if not musubi_root:
        return False
    p = Path(musubi_root)
    # registry に登録済みのスクリプト名を検索対象にする
    known_scripts = {s.train_script for s in list_specs() if s.backend == "musubi"}
    known_scripts.add("train_network.py")  # fallback
    for script in known_scripts:
        for base in (p, p / "src"):
            if (base / script).exists():
                return True
    return False


def get_musubi_script(musubi_root: str, model_family: str) -> str | None:
    """registry からスクリプト名を取得してパスを解決する。"""
    spec = get_spec(model_family)
    script_name = spec.train_script if spec else "train_network.py"
    p = Path(musubi_root)
    for cand in [p / script_name, p / "src" / script_name]:
        if cand.exists():
            return str(cand)
    # fallback
    for cand in [p / "train_network.py", p / "src" / "train_network.py"]:
        if cand.exists():
            return str(cand)
    return None


def resolve_musubi_python(settings: dict[str, str]) -> str:
    musubi_root = (settings.get("musubi_root") or "").strip()
    if musubi_root:
        for cand in (
            Path(musubi_root) / "venv" / "Scripts" / "python.exe",
            Path(musubi_root) / ".venv" / "Scripts" / "python.exe",
        ):
            if cand.exists():
                return str(cand)
    return (settings.get("python_exe") or "").strip()


def write_musubi_dataset_json(
    run_dir: Path, train_data_dir: str, repeats: int, resolution: int, batch_size: int = 1,
) -> Path:
    """musubi-tuner 用データセット JSON ファイルを生成する。

    重要: musubi-tuner の BaseDatasetParams(dataset/config_utils.py)は
    batch_size をデータセット単位のフィールドとして持ち(既定値1)、
    kohya_ss のようにTOML側のtrain_batch_sizeでは制御しない。以前は
    batch_size をこの dataset_config に一切含めていなかったため、UIで
    どんな値を設定してもmusubi側は既定値1のまま使われ、
    「bucket: total batches: (images*repeats)」となり、ユーザーが設定した
    batch_sizeがRuntimeのstep数に一切反映されない実バグがあった
    (実行中Runの実ログ"total batches: 147"(=49枚×repeats3、batch_size=6は
    未適用)と、musubi-tuner本体のdataset/config_utils.py:38
    `batch_size: int = 1`および dataset/bucket.py の
    `num_batches = math.ceil(len(bucket) / self.batch_size)` を突き合わせて
    確定した)。
    """
    cache_dir = run_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    # musubi-tuner の dataset config スキーマ(ConfigSanitizer.IMAGE_DATASET_DISTINCT_SCHEMA /
    # DATASET_ASCENDABLE_SCHEMA)には shuffle_caption / caption_separator が存在しない
    # (kohya_ss 固有のオプション)。含めると voluptuous の extra keys エラーで即失敗する。
    dataset_config = {
        "datasets": [{
            "image_directory": train_data_dir,
            "caption_extension": ".txt",
            "num_repeats": repeats,
            "resolution": [resolution, resolution],
            "cache_directory": str(cache_dir),
            "batch_size": max(1, int(batch_size)),
        }]
    }
    json_path = run_dir / "dataset.json"
    json_path.write_text(json.dumps(dataset_config, indent=2, ensure_ascii=False), encoding="utf-8")
    return json_path


def write_musubi_toml_config(run_dir: Path, cfg: dict, model_family: str) -> Path:
    """musubi-tuner 用 TOML 設定ファイルを生成する（registry 経由）。"""
    spec = get_spec(model_family)
    runner = get_runner("musubi")
    config_path = run_dir / "musubi_config.toml"

    if spec is None or runner is None:
        # fallback: 旧来のシンプル生成（registry 未登録モデル用）
        def q(v: str) -> str:
            return v.replace("\\", "\\\\").replace('"', '\\"')
        lr = cfg.get("learning_rate", 1e-4)
        lines = [
            f'dit = "{q(cfg.get("base_checkpoint_path", ""))}"',
            f'dataset_config = "{q(cfg.get("dataset_config", ""))}"',
            f'output_dir = "{q(cfg.get("output_dir", ""))}"',
            f'output_name = "{q(cfg.get("output_name", "lora_output"))}"',
            'network_module = "networks.lora"',
            f'network_dim = {int(cfg.get("rank", 32))}',
            f'network_alpha = {float(cfg.get("alpha", 16))}',
            f'max_train_epochs = {int(cfg.get("epochs", 10))}',
            f'save_every_n_epochs = {int(cfg.get("save_every_n_epochs", 1))}',
            f'optimizer_type = "{q(cfg.get("optimizer", "AdamW8bit"))}"',
            f'lr_scheduler = "{q(cfg.get("scheduler", "cosine_with_restarts"))}"',
            'lr_scheduler_num_cycles = 1',
            f'learning_rate = {lr}',
            f'mixed_precision = "{cfg.get("mixed_precision", "bf16")}"',
            f'save_precision = "{cfg.get("save_precision", "bf16")}"',
            f'gradient_checkpointing = {"true" if cfg.get("gradient_checkpointing", True) else "false"}',
            f'logging_dir = "{q(cfg.get("logs_dir", ""))}"',
        ]
        config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return config_path

    common = runner.common_toml(spec, cfg)
    resolved = {rm.key: cfg.get(rm.toml_key, cfg.get(f"{rm.key}_path", ""))
                for rm in spec.required_models if rm.toml_key}
    toml_dict = runner.build_toml(spec, common, resolved)
    config_path.write_text(runner.toml_serialize(toml_dict), encoding="utf-8")
    return config_path


class MusubiBackend(TrainingBackend):
    display_name = "Extended Diffusion Engine"

    def is_ready(self, settings: dict[str, str]) -> bool:
        return is_musubi_ready(settings)

    def supports_resume(self) -> bool:
        """musubi-tunerのtrainer_base.pyは--network_weights指定時に
        network.load_weights()で既存LoRA重みを初期値として読み込む
        (kohyaのtrain_network.pyと同じ仕組み。utils/train_utils.pyの
        引数リストとtrainer_base.py:1603-1606で実装を確認済み)。
        optimizer状態・epochカウンタは継続されない(重み継続のみ)。"""
        return True

    def prepare(self, ctx: TrainingContext) -> PreparedRun:
        """train_dir 準備・dataset.json・TOML 生成まで（学習開始前の準備）。

        準備できない場合は RuntimeError を送出する（呼び出し側が run を error 化する）。
        """
        project_id, run_id, settings, cfg = ctx.project_id, ctx.run_id, ctx.settings, ctx.cfg
        model_family = cfg.get("model_family", "wan21")
        musubi_root = settings.get("musubi_root", "")
        train_script = get_musubi_script(musubi_root, model_family)
        if not train_script:
            raise RuntimeError("musubi-tuner の学習スクリプトが見つかりません")

        python_exe = resolve_musubi_python(settings)
        if not python_exe or not Path(python_exe).exists():
            raise RuntimeError("musubi-tuner 用 Python が見つかりません")

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
        _spec_for_res = get_spec(model_family)
        _default_res = _spec_for_res.default_resolution if _spec_for_res else 512
        resolution = int(cfg.get("resolution", _default_res))
        gpu_device_id = int(cfg.get("gpu_device_id", 1))

        def _has_images(d: str) -> bool:
            p = Path(d)
            return p.exists() and any(
                f.suffix.lower() in IMAGE_EXTS for f in p.iterdir() if f.is_file()
            )

        if dataset_source == "original" and not _has_images(dataset_dir):
            # processed モード選択時は暗黙フォールバックしない（resolve_dataset_source が
            # 既に検証済みのため、ここに来る場合は明確なエラーとして扱うべき）。
            conn_items = get_conn()
            rows = conn_items.execute(
                "SELECT file_path FROM dataset_items WHERE project_id = ? LIMIT 200", (project_id,)
            ).fetchall()
            conn_items.close()
            existing = [r["file_path"] for r in rows if Path(r["file_path"]).exists()]
            if existing:
                dataset_dir = str(Path(existing[0]).parent)
                dataset_snapshot = {"dataset_source": "original", "resolved_dataset_dir": dataset_dir}

        run_dir = prepare_run_dir(run_id, dataset_dir, repeats, project_name)
        train_data_root = run_dir / "train_data"
        image_count = sum(
            1 for p in train_data_root.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
        if image_count == 0:
            raise RuntimeError(f"train_data に画像が見つかりません (dataset_dir={dataset_dir})")
        persist_dataset_snapshot(run_id, {**dataset_snapshot, "dataset_staged_image_count": image_count})

        # musubi-tuner の ImageDirectoryDatasource はサブディレクトリを再帰的に走査しない。
        # prepare_run_dir が作る kohya_ss 形式 train_data/{repeats}_{name}/ の実画像フォルダを
        # 直接 image_directory に指定する必要がある（train_data 直下だと 0 件になる）。
        safe_name = re.sub(r"[^\w\-]", "_", project_name)
        musubi_image_dir = run_dir / "train_data" / f"{repeats}_{safe_name}"
        batch_size = int(cfg.get("train_batch_size", 1) or 1)
        dataset_json_path = write_musubi_dataset_json(
            run_dir, str(musubi_image_dir), repeats, resolution, batch_size,
        )

        optimizer_name = cfg.get("optimizer", "AdamW8bit")
        raw_lr = cfg.get("learning_rate", 1e-4)
        effective_lr = 1.0 if optimizer_name in _AUTO_LR_OPTIMIZERS else raw_lr

        # registry からスペックとランナーを取得してモデルパスを解決
        _musubi_spec = get_spec(model_family)
        _musubi_runner = get_runner("musubi")
        _resolved_models: dict[str, str] = {}
        if _musubi_spec and _musubi_runner:
            _resolved_models = _musubi_runner.resolve_models(_musubi_spec, settings, cfg)

        musubi_cfg = {
            "base_checkpoint_path": cfg.get("base_checkpoint_path", ""),
            "dataset_config": str(dataset_json_path),
            "output_dir": str(run_dir / "output"),
            "output_name": cfg.get("output_name", "lora_output"),
            "resolution": resolution,
            "train_batch_size": int(cfg.get("train_batch_size", 1) or 1),
            "learning_rate": effective_lr,
            "rank": int(cfg.get("rank", 32)),
            "alpha": float(cfg.get("alpha", 16)),
            "epochs": int(cfg.get("epochs", 10)),
            "save_every_n_epochs": int(cfg.get("save_every_n_epochs", 1)),
            "optimizer": optimizer_name,
            "scheduler": cfg.get("scheduler", "cosine_with_restarts"),
            "mixed_precision": cfg.get("mixed_precision", "bf16"),
            "save_precision": cfg.get("save_precision", "bf16"),
            "gradient_checkpointing": cfg.get("gradient_checkpointing", True),
            "logs_dir": str(run_dir / "logs"),
            # 解決済みモデルパスを cfg に注入（write_musubi_toml_config が参照）
            **{f"{k}_path": v for k, v in _resolved_models.items() if v},
            # Resume: /resume が config_json に書き込んだ既存LoRA重みパスを
            # そのままMusubiRunner.common_tomlへ橋渡しする(network_weightsとして
            # TOMLへ出力される。kohyaのresume_from_checkpoint配線と同じ形)。
            "resume_from_checkpoint": cfg.get("resume_from_checkpoint", ""),
        }

        config_path = write_musubi_toml_config(run_dir, musubi_cfg, model_family)
        log_path = run_dir / "logs" / "training.log"

        conn = get_conn()
        conn.execute(
            "UPDATE training_runs SET log_path = ? WHERE id = ?",
            (str(log_path), run_id),
        )
        conn.commit()
        conn.close()

        gpu_mapping = resolve_cuda_index(gpu_device_id, python_exe)
        env = {
            **os.environ,
            # Match nvidia-smi's physical PCI order before masking devices.
            # Without this, CUDA ordinal 1 can resolve to the RTX 3060 on
            # Windows hosts where the fastest GPU is enumerated first.
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": str(gpu_mapping["cuda_index"]),
        }
        cmd = [python_exe, train_script, "--config_file", str(config_path)]

        return PreparedRun(
            run_dir=run_dir, config_path=config_path, log_path=log_path,
            cmd=cmd, cwd=musubi_root, python_exe=python_exe, env=env,
            extra={
                "model_family": model_family,
                "musubi_cfg": musubi_cfg,
                "gpu_device_id": gpu_device_id,
                "gpu_mapping": gpu_mapping,
                "gpu_mapping_verified": bool(gpu_mapping["verified"]),
                "spec": _musubi_spec,
                "runner": _musubi_runner,
                "resolved_models": _resolved_models,
                "dataset_json_path": str(dataset_json_path),
            },
        )

    def cache(self, ctx: TrainingContext, prepared: PreparedRun) -> bool:
        """事前キャッシュ生成（registry の cache_steps を汎用的に実行）。不要なら no-op(True)。"""
        spec = prepared.extra.get("spec")
        runner = prepared.extra.get("runner")
        if not (spec and runner and spec.cache_steps):
            return True
        if not prepared.extra.get("gpu_mapping_verified", False):
            return False
        return runner.run_cache_steps(
            spec,
            prepared.extra.get("resolved_models", {}),
            dataset_json_path=prepared.extra.get("dataset_json_path", ""),
            python_exe=prepared.python_exe,
            tool_root=prepared.cwd,
            log_path=prepared.log_path,
            env=prepared.env,
        )

    def train(self, ctx: TrainingContext, prepared: PreparedRun) -> None:
        """サブプロセス起動 + 監視（ブロッキング）。DB への状態反映まで含む。"""
        project_id, run_id = ctx.project_id, ctx.run_id
        model_family = prepared.extra.get("model_family", "")
        musubi_cfg: dict[str, Any] = prepared.extra.get("musubi_cfg", {})
        gpu_device_id = prepared.extra.get("gpu_device_id", 1)
        log_path = prepared.log_path

        header = (
            f"[MUSUBI] model_family={model_family}\n"
            f"[MUSUBI] python={prepared.python_exe}\n"
            f"[MUSUBI] script={prepared.cmd[1] if len(prepared.cmd) > 1 else ''}\n"
            f"[MUSUBI] base_checkpoint={musubi_cfg.get('base_checkpoint_path', '')}\n"
            f"[MUSUBI] config={prepared.config_path}\n"
            f"[MUSUBI] CUDA_VISIBLE_DEVICES={gpu_device_id}\n"
            f"[MUSUBI] cmd={' '.join(prepared.cmd)}\n\n"
        )
        try:
            with open(log_path, "w", encoding="utf-8") as lf:
                lf.write(header)
        except OSError:
            pass

        gpu_name = detect_gpu_name(gpu_device_id)
        STEP_TIMING.pop(project_id, None)

        log_fh = None
        proc: subprocess.Popen | None = None
        try:
            if not prepared.extra.get("gpu_mapping_verified", False):
                raise RuntimeError(f"GPU物理順序とCUDA列挙順を照合できないため開始を拒否: {prepared.extra.get('gpu_mapping', {})}")
            log_fh = open(log_path, "ab")
            proc = subprocess.Popen(
                prepared.cmd,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                cwd=prepared.cwd,
                env=prepared.env,
            )
            with RUNNER_LOCK:
                RUNNER_PROCESSES[project_id] = proc
            tail_monitor(run_id, project_id, proc, 1.0, gpu_name, model_family)
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
