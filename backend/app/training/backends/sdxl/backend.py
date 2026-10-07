"""SDXL エンジン — Stable Diffusion 1.x/2.x/XL 系。

内部実装は kohya_ss (sd-scripts) を利用するが、これは実装詳細でありこの
Backend の外（API・UI）には露出しない。cache() はこのエンジンでは不要
（kohya_ss 自身が latent キャッシュ等を内包する）ため既定の no-op を使う。
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from ....db import get_conn
from ....services.preview_prompts import (
    DEFAULT_PREVIEW_RESOLUTION,
    build_sample_line,
    default_prompt_item,
)
from ....trainers import get_spec
from ...base import PreparedRun, TrainingBackend, TrainingContext
from ...runtime import (
    RUNNER_LOCK,
    RUNNER_PROCESSES,
    STEP_TIMING,
    IMAGE_EXTS,
    sec_factor_for_run,
    tail_monitor,
)
from ..shared.helpers import persist_dataset_snapshot, prepare_run_dir, resolve_dataset_source

_AUTO_LR_OPTIMIZERS = {"Prodigy", "DAdaptAdam", "DAdaptAdaGrad", "DAdaptSGD"}


def get_train_script(kohya_root: str, sdxl: bool = False) -> str | None:
    """学習スクリプトのパスを返す。sdxl=True なら SDXL 用スクリプトを優先する。"""
    p = Path(kohya_root)
    name = "sdxl_train_network.py" if sdxl else "train_network.py"
    candidates = [
        p / name,
        p / "sd-scripts" / name,
    ]
    if sdxl:
        # SDXL スクリプトが無い環境では SD1.x 用にフォールバック
        candidates += [p / "train_network.py", p / "sd-scripts" / "train_network.py"]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def resolve_python(settings: dict[str, str]) -> str:
    """学習に使う Python を決定する。

    kohya_root 配下に専用 venv があればそれを最優先する。
    設定の python_exe（バックエンド venv 等）は kohya 依存・CUDA torch を
    持たないことがあるため、kohya venv を優先することで誤実行を防ぐ。
    """
    kohya_root = (settings.get("kohya_root") or "").strip()
    if kohya_root:
        for cand in (
            Path(kohya_root) / "venv" / "Scripts" / "python.exe",
            Path(kohya_root) / ".venv" / "Scripts" / "python.exe",
        ):
            if cand.exists():
                return str(cand)
    return settings.get("python_exe", "").strip()


def detect_sdxl(checkpoint_path: str) -> bool:
    """safetensors のヘッダ（テンソル名）から SDXL かどうかを判定する。

    SDXL は 2 つ目のテキストエンコーダ（conditioner.embedders.1）を持つ。
    判定不能な場合はファイル名の 'xl' ヒューリスティックにフォールバックする。
    """
    path = Path(checkpoint_path)
    try:
        if path.suffix.lower() == ".safetensors" and path.exists():
            with open(path, "rb") as f:
                header_len = int.from_bytes(f.read(8), "little")
                # ヘッダは過大なら読まない（破損対策）
                if 0 < header_len < 50_000_000:
                    header = json.loads(f.read(header_len).decode("utf-8", "replace"))
                    keys = header.keys()
                    for k in keys:
                        if k.startswith("conditioner.embedders.1") or (
                            "label_emb" in k and "diffusion_model" in k
                        ):
                            return True
                    # conditioner.embedders 系があり cond_stage_model が無ければ SDXL
                    has_cond = any(k.startswith("conditioner.embedders") for k in keys)
                    has_legacy = any(k.startswith("cond_stage_model") for k in keys)
                    if has_cond and not has_legacy:
                        return True
                    if has_legacy:
                        return False
    except (OSError, ValueError, json.JSONDecodeError):
        pass
    # フォールバック: ファイル名
    return "xl" in path.name.lower()


def is_kohya_ready(settings: dict[str, str]) -> bool:
    python_exe = resolve_python(settings)
    kohya_root = settings.get("kohya_root", "").strip()
    if not python_exe or not kohya_root:
        return False
    if not Path(python_exe).exists():
        return False
    return get_train_script(kohya_root) is not None


def kohya_not_ready_reason(settings: dict[str, str]) -> str:
    kohya_root = settings.get("kohya_root", "").strip()
    python_exe = resolve_python(settings)
    if not python_exe and not kohya_root:
        return "python_exe と kohya_root が未設定"
    if not kohya_root:
        return "kohya_root が未設定"
    if not python_exe:
        return "学習用 Python が未設定（kohya venv も未構築）"
    if not Path(python_exe).exists():
        return f"学習用 Python が見つかりません: {python_exe}"
    train_script = get_train_script(kohya_root)
    if train_script is None:
        return f"kohya_root 内に train_network.py / sdxl_train_network.py が見つかりません: {kohya_root}"
    return ""


def write_toml_config(run_dir: Path, cfg: dict, sdxl: bool = False) -> Path:
    """kohya_ss 用 TOML 設定ファイルを生成する。

    sdxl=True のとき SDXL 向けパラメータ（1024 解像度・no_half_vae 等）を付与する。
    GPU をフル活用するため bf16 / xformers / latent キャッシュを有効化する。
    """
    config_path = run_dir / "config.toml"

    def q(v: str) -> str:
        return v.replace("\\", "\\\\").replace('"', '\\"')

    # 設定された解像度をそのまま使う（フォーム値＝予測＝実学習を一致させる）
    resolution = int(cfg["resolution"])

    lr = cfg.get("learning_rate", 1e-4)

    lines = [
        f'pretrained_model_name_or_path = "{q(cfg["base_checkpoint_path"])}"',
        f'train_data_dir = "{q(cfg["train_data_dir"])}"',
        f'output_dir = "{q(cfg["output_dir"])}"',
        f'output_name = "{q(cfg["output_name"])}"',
        'save_model_as = "safetensors"',
        f'resolution = "{resolution},{resolution}"',
        'network_module = "networks.lora"',
        f'network_dim = {int(cfg["rank"])}',
        f'network_alpha = {float(cfg["alpha"])}',
        *([f'network_weights = "{q(cfg["resume_from_checkpoint"])}"'] if cfg.get("resume_from_checkpoint") else []),
        f'max_train_epochs = {int(cfg["epochs"])}',
        f'save_every_n_epochs = {int(cfg["save_every_n_epochs"])}',
        f'train_batch_size = {int(cfg.get("train_batch_size", 2))}',
        'caption_extension = ".txt"',
        'shuffle_caption = true',
        'keep_tokens = 1',
        f'optimizer_type = "{q(cfg.get("optimizer", "AdamW8bit"))}"',
        f'lr_scheduler = "{q(cfg.get("scheduler", "cosine_with_restarts"))}"',
        'lr_scheduler_num_cycles = 1',
        f'learning_rate = {lr}',
        # ── GPU フル活用 ──
        f'mixed_precision = "{cfg.get("mixed_precision", "bf16")}"',
        f'save_precision = "{cfg.get("save_precision", "fp16")}"',
        f'gradient_checkpointing = {"true" if cfg.get("gradient_checkpointing", True) else "false"}',
        f'xformers = {"true" if cfg.get("xformers", True) else "false"}',
        f'cache_latents = {"true" if cfg.get("cache_latents", True) else "false"}',
        f'cache_latents_to_disk = {"true" if cfg.get("cache_latents_to_disk", False) else "false"}',
        f'persistent_data_loader_workers = {"true" if cfg.get("persistent_data_loader_workers", True) else "false"}',
        f'max_data_loader_n_workers = {int(cfg.get("max_data_loader_n_workers", 4))}',
        *((['network_train_unet_only = true']) if cfg.get('network_train_unet_only', False) else []),
        # ── Aspect ratio bucketing ──
        'enable_bucket = true',
        'min_bucket_reso = 256',
        'max_bucket_reso = 2048',
        'bucket_reso_steps = 64',
        f'logging_dir = "{q(cfg["logs_dir"])}"',
    ]

    if sdxl:
        # SDXL VAE は fp16 で NaN になるため half を無効化
        lines.append("no_half_vae = true")

    # ── エポックごとのサンプル画像生成（プレビュー用） ──
    # kohya 本体に sample 生成させ output_dir/sample/*.png を出力させる。
    # 別途 ComfyUI に依存せず、学習と同一モデルでプレビューを得る。
    # 複数プレビュープロンプトに対応（preview_prompts_json があれば各プロンプトで1枚ずつ生成）
    prompt_items: list[dict] = []
    frozen_preview = None
    try:
        pconn = get_conn()
        jrow = pconn.execute(
            "SELECT value FROM app_settings WHERE key='preview_prompts_json'"
        ).fetchone()
        prows = pconn.execute(
            "SELECT key, value FROM app_settings "
            "WHERE key IN ('positive_prompt','negative_prompt','preview_resolution',"
            "'preview_sampler','preview_cfg','preview_steps')"
        ).fetchall()
        frozen_row = pconn.execute("SELECT payload_json FROM basepipe_preview_profile_snapshots WHERE id=? AND project_id=?", (cfg.get('preview_profile_snapshot_id'),cfg.get('project_id'))).fetchone()
        frozen_preview = json.loads(frozen_row['payload_json']) if frozen_row else None
        pconn.close()
        if jrow and jrow["value"]:
            parsed = json.loads(jrow["value"])
            if isinstance(parsed, list):
                prompt_items = [
                    p for p in parsed
                    if isinstance(p, dict)
                    and (
                        str(p.get("quality", "")).strip()
                        or str(p.get("positive", "")).strip()
                        or str(p.get("trigger_words", "")).strip()
                    )
                ]
        pmap = {str(r["key"]): str(r["value"]) for r in prows}
    except Exception:
        pmap = {}
    if frozen_preview:
        prompt_items=[{'positive':frozen_preview['prompt'],'negative':frozen_preview.get('negative_prompt','')}]
        pmap.update(preview_resolution=str(frozen_preview.get('resolution',512)),preview_sampler=frozen_preview.get('sampler') or 'euler',preview_cfg=str(cfg.get('preview_cfg') if cfg.get('preview_cfg') is not None else frozen_preview.get('cfg',7)),preview_steps=str(cfg.get('preview_steps') if cfg.get('preview_steps') is not None else frozen_preview.get('steps',20)))
    if not prompt_items:
        item = default_prompt_item()
        if pmap.get("positive_prompt"):
            item["positive"] = pmap["positive_prompt"]
        if pmap.get("negative_prompt"):
            item["negative"] = pmap["negative_prompt"]
        prompt_items = [item]

    # プレビューのサンプラー/cfg/steps/解像度既定値はモデル世代 (spec) から取得する。
    # kohya バックエンドは現状 sd / sdxl のみのため、sdxl フラグからスペックを引く。
    # ユーザーが設定画面で明示的に上書きしていれば（preview_sampler/cfg/steps）そちらを優先する。
    preview_spec = get_spec("sdxl" if sdxl else "sd")
    spec_preview_res = preview_spec.default_resolution if preview_spec else DEFAULT_PREVIEW_RESOLUTION
    preview_sampler = pmap.get("preview_sampler") or (preview_spec.preview_sampler if preview_spec else "euler_a")
    try:
        preview_cfg = float(pmap["preview_cfg"]) if pmap.get("preview_cfg") else (
            preview_spec.preview_cfg if preview_spec else 7.0
        )
    except (ValueError, TypeError):
        preview_cfg = preview_spec.preview_cfg if preview_spec else 7.0
    try:
        preview_steps = int(float(pmap["preview_steps"])) if pmap.get("preview_steps") else (
            preview_spec.preview_steps if preview_spec else 20
        )
    except (ValueError, TypeError):
        preview_steps = preview_spec.preview_steps if preview_spec else 20

    # プレビュー解像度は学習解像度と独立（既定はモデル spec の default_resolution）
    try:
        preview_res = max(256, min(2048, int(pmap.get("preview_resolution") or spec_preview_res)))
    except (ValueError, TypeError):
        preview_res = spec_preview_res
    sample_lines = [
        build_sample_line(p, preview_res, steps=preview_steps, cfg=preview_cfg, seed=int(frozen_preview.get("seed",42)) if frozen_preview else 42)
        for p in prompt_items
    ]
    prompts_path = run_dir / "sample_prompts.txt"
    prompts_path.write_text("\n".join(sample_lines) + "\n", encoding="utf-8")
    every = max(1, int(cfg.get("save_every_n_epochs", 1)))
    lines.append(f'sample_prompts = "{q(str(prompts_path))}"')
    lines.append(f"sample_every_n_epochs = {every}")
    lines.append(f'sample_sampler = "{q(preview_sampler)}"')

    min_snr = cfg.get("min_snr_gamma")
    if min_snr is not None:
        lines.append(f"min_snr_gamma = {int(min_snr)}")
    reg_dir = cfg.get("reg_data_dir", "").strip()
    # 正則化フォルダは画像が実際に存在するときだけ指定する
    if reg_dir and Path(reg_dir).exists() and any(
        p.suffix.lower() in IMAGE_EXTS for p in Path(reg_dir).rglob("*") if p.is_file()
    ):
        lines.append(f'reg_data_dir = "{q(reg_dir)}"')

    from ...advanced import apply_to_toml
    lines = apply_to_toml(lines, cfg, "sdxl")
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return config_path


class SDXLBackend(TrainingBackend):
    display_name = "Standard Diffusion Engine"

    def is_ready(self, settings: dict[str, str]) -> bool:
        return is_kohya_ready(settings)

    def supports_resume(self) -> bool:
        """kohyaのtrain_network.pyは--network_weightsで既存LoRA重みからの
        継続学習に対応している(train_network.py: args.network_weightsで
        load_weights()を呼ぶ実装を確認済み)。"""
        return True

    def prepare(self, ctx: TrainingContext) -> PreparedRun:
        """train_dir 準備・TOML 生成まで（学習開始前の準備）。

        準備できない場合は RuntimeError を送出する（呼び出し側が run を error 化する）。
        """
        project_id, run_id, settings, cfg = ctx.project_id, ctx.run_id, ctx.settings, ctx.cfg
        from ...learning_rates import resolve_learning_rates
        resolve_learning_rates(cfg, reject_conflict=True)
        python_exe = resolve_python(settings)
        kohya_root = settings.get("kohya_root", "")

        # ベースモデルが SDXL かを判定し、適切な学習スクリプトを選択する
        base_ckpt = cfg.get("base_checkpoint_path", "")
        is_sdxl = detect_sdxl(base_ckpt) if base_ckpt else False
        train_script = get_train_script(kohya_root, sdxl=is_sdxl)

        if not train_script:
            raise RuntimeError("学習スクリプトが見つかりません")

        conn_p = get_conn()
        project_row = conn_p.execute(
            "SELECT name, dataset_dir, outputs_dir FROM projects WHERE id = ?", (project_id,)
        ).fetchone()
        conn_p.close()

        project_name = project_row["name"]
        # dataset_source ("original"/"processed") を明示的に解決する。processed 指定時は
        # Dataset Builder Pipeline 完了マニフェストが無ければ即エラー（暗黙フォールバック禁止）。
        dataset_source = str(cfg.get("dataset_source") or "original")
        dataset_dir, dataset_snapshot = resolve_dataset_source(
            project_row["dataset_dir"] or cfg.get("train_data_dir", ""), dataset_source
        )
        repeats = int(cfg.get("repeats", 5))

        # dataset_dir が空か存在しない・画像がない場合は dataset_items の file_path から割り出す
        # （dataset_source=processed のときは resolve_dataset_source が既に検証済みのため対象外）
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
                # 共通の親ディレクトリを dataset_dir とする
                dataset_dir = str(Path(existing[0]).parent)
                dataset_snapshot = {"dataset_source": "original", "resolved_dataset_dir": dataset_dir}

        run_dir = prepare_run_dir(run_id, dataset_dir, repeats, project_name)
        # 画像が 0 枚なら即エラー
        train_data_root = run_dir / "train_data"
        image_count = sum(
            1 for p in train_data_root.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
        if image_count == 0:
            raise RuntimeError(f"train_data に画像が見つかりません (dataset_dir={dataset_dir})")
        persist_dataset_snapshot(run_id, {**dataset_snapshot, "dataset_staged_image_count": image_count})

        optimizer_name = cfg.get("optimizer", "AdamW8bit")
        raw_lr = cfg.get("learning_rate", 1e-4)
        # Prodigy / DAdaptAdam は lr=1.0 で自動適応する設計。
        # 小さい値（例: 1e-4）を渡すと適応が止まりロスが収束しない。
        effective_lr = 1.0 if optimizer_name in _AUTO_LR_OPTIMIZERS else raw_lr

        kohya_cfg = {
            "advanced": cfg.get("advanced", {}),
            "base_checkpoint_path": cfg.get("base_checkpoint_path", ""),
            "train_data_dir": str(run_dir / "train_data"),
            "output_dir": str(run_dir / "output"),
            "output_name": cfg.get("output_name", "lora_output"),
            "resolution": int(cfg.get("resolution", 512)),
            "train_batch_size": int(cfg.get("train_batch_size", 2) or 2),
            "learning_rate": effective_lr,
            "rank": int(cfg.get("rank", 16)),
            "alpha": float(cfg.get("alpha", 8)),
            "epochs": int(cfg.get("epochs", 10)),
            "save_every_n_epochs": int(cfg.get("save_every_n_epochs", 1)),
            "optimizer": optimizer_name,
            "scheduler": cfg.get("scheduler", "cosine_with_restarts"),
            "cache_latents": bool(cfg.get("cache_latents", True)),
            "cache_latents_to_disk": bool(cfg.get("cache_latents", True) and cfg.get("cache_latents_to_disk", False)),
            "min_snr_gamma": cfg.get("min_snr_gamma", 5),
            "reg_data_dir": cfg.get("reg_data_dir", ""),
            "logs_dir": str(run_dir / "logs"),
            # Resume（既存LoRA重みからの継続学習）: /resume APIが直近checkpointの
            # パスをconfig_json["resume_from_checkpoint"]へ書き込んでから再起動する。
            # kohyaのnetwork_weightsはLoRA重みの初期値として使われるのみで、
            # optimizer状態・epochカウンタ自体は継続されない（真の状態Resumeではなく
            # 「既存重みからの継続学習」であることをUI/ドキュメントで明示する必要がある）。
            "resume_from_checkpoint": cfg.get("resume_from_checkpoint", ""),
        }

        for key in ("mixed_precision", "save_precision", "gradient_checkpointing", "xformers", "persistent_data_loader_workers", "max_data_loader_n_workers", "network_train_unet_only"):
            if key in cfg:
                kohya_cfg[key] = cfg[key]
        if "advanced" not in cfg:
            kohya_cfg.pop("advanced", None)
        kohya_cfg.update(project_id=project_id,preview_profile_snapshot_id=cfg.get('preview_profile_snapshot_id'),preview_steps=cfg.get('preview_steps'),preview_cfg=cfg.get('preview_cfg'))
        config_path = write_toml_config(run_dir, kohya_cfg, sdxl=is_sdxl)
        if is_sdxl:
            lines=config_path.read_text(encoding='utf8').splitlines()
            config_path.write_text('\n'.join(line for line in lines if line.split('=',1)[0].strip() not in {'sample_prompts','sample_every_n_epochs','sample_every_n_steps','sample_at_first','sample_sampler'})+'\n',encoding='utf8')
            persist_dataset_snapshot(run_id,{'preview_backend':'comfyui'})
        log_path = run_dir / "logs" / "training.log"

        conn = get_conn()
        conn.execute(
            "UPDATE training_runs SET log_path = ? WHERE id = ?",
            (str(log_path), run_id),
        )
        conn.commit()
        conn.close()

        # accelerate launch で GPU を使って学習する（単機・単GPU・bf16）
        # 単一GPU構成では accelerate launch ラッパーは不要。
        # （Python 3.12.x の argparse バグで accelerate CLI が落ちるため回避）
        # 学習スクリプトを直接実行する。GPU / mixed_precision(bf16) は config.toml から適用される。
        cmd = [python_exe, train_script, "--config_file", str(config_path)]
        gpu_id=int(cfg.get('gpu_device_id',1))
        if gpu_id!=1:raise RuntimeError('この環境の学習は物理GPU1を指定してください')
        gpu_uuid=subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=5).strip()
        environment={**os.environ,'CUDA_DEVICE_ORDER':'PCI_BUS_ID','CUDA_VISIBLE_DEVICES':gpu_uuid,'OMP_NUM_THREADS':'4','USE_TF':'0'}
        if is_sdxl:
            launcher=run_dir/'comfy_training_entry.py'
            hook_path=Path(__file__).resolve().parents[2]/'runtime'/'comfy_epoch_hook.py'
            launcher.write_text('import sys,runpy,torch,importlib.util\n'+f'sys.path.insert(0,{str(Path(train_script).parent)!r})\n'+"torch.set_num_threads(4)\n"+f"spec=importlib.util.spec_from_file_location('studio_comfy_epoch',{str(hook_path)!r})\n"+"hook=importlib.util.module_from_spec(spec);spec.loader.exec_module(hook)\n"+f"hook.install({str(run_dir)!r},{run_id},'http://127.0.0.1:5175')\n"+f"sys.argv=[{train_script!r},'--config_file',{str(config_path)!r}]\nrunpy.run_path({train_script!r},run_name='__main__')\n",encoding='utf8')
            cmd=[python_exe,str(launcher)]


        return PreparedRun(
            run_dir=run_dir, config_path=config_path, log_path=log_path,
            cmd=cmd, cwd=kohya_root, python_exe=python_exe, env=environment,
            extra={
                "is_sdxl": is_sdxl,
                "base_checkpoint_path": base_ckpt,
                "kohya_cfg": kohya_cfg,
            },
        )

    def train(self, ctx: TrainingContext, prepared: PreparedRun) -> None:
        """サブプロセス起動 + 監視（ブロッキング）。DB への状態反映まで含む。"""
        project_id, run_id = ctx.project_id, ctx.run_id
        is_sdxl = prepared.extra.get("is_sdxl", False)
        base_ckpt = prepared.extra.get("base_checkpoint_path", "")
        kohya_cfg: dict[str, Any] = prepared.extra.get("kohya_cfg", {})
        log_path = prepared.log_path

        # ── kohya 起動: stdout をログファイルへ直接リダイレクト ──
        # PIPE を使わずファイルへ書かせることで、監視スレッド（uvicornワーカー）が
        # reload/クラッシュしてもサブプロセスはブロックせず書き込み続ける。
        # 監視は tail_monitor がログを tail して行うため、再起動後も再アタッチできる。
        header = (
            f"[KOHYA] model_type={'SDXL' if is_sdxl else 'SD1.x/2.x'}\n"
            f"[KOHYA] python={prepared.python_exe}\n"
            f"[KOHYA] script={prepared.cmd[1] if len(prepared.cmd) > 1 else ''}\n"
            f"[KOHYA] base_checkpoint={base_ckpt}\n"
            f"[KOHYA] config={prepared.config_path}\n"
            f"[KOHYA] cmd={' '.join(prepared.cmd)}\n\n"
        )
        try:
            with open(log_path, "w", encoding="utf-8") as lf:
                lf.write(header)
        except OSError:
            pass

        sec_factor, gpu_name, model_family = sec_factor_for_run(
            {**kohya_cfg, "train_batch_size": kohya_cfg.get("train_batch_size", 2)},
            detect_sdxl,
        )
        STEP_TIMING.pop(project_id, None)

        log_fh = None
        proc: subprocess.Popen | None = None
        try:
            log_fh = open(log_path, "ab")  # subprocess 用に raw バイト追記
            proc = subprocess.Popen(
                prepared.cmd,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                cwd=prepared.cwd,
                env=prepared.env,
            )
            with RUNNER_LOCK:
                RUNNER_PROCESSES[project_id] = proc
            # 同期的に tail 監視（このスレッドは ensure_runner が起動したワーカースレッド）
            tail_monitor(run_id, project_id, proc, sec_factor, gpu_name, model_family)
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
