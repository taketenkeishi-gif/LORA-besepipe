from __future__ import annotations

import json
import base64
import os
import re
import shutil
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import IO

from fastapi import APIRouter, HTTPException
from PIL import Image, ImageDraw

from ..db import get_conn
from ..schemas import TrainingControlIn, TrainingStartIn

router = APIRouter(prefix="/training", tags=["training"])

RUNNER_LOCK = threading.Lock()
RUNNER_THREADS: dict[int, threading.Thread] = {}
RUNNER_PROCESSES: dict[int, "subprocess.Popen[str]"] = {}

SLOTS = ["face", "bust", "full", "bg"]
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}

# ── kohya stdout パターン ──────────────────────────────────────────────────
_RE_EPOCH = re.compile(r"epoch\s+(\d+)\s*/\s*(\d+)", re.IGNORECASE)
_RE_TQDM = re.compile(
    r"(\d+)/(\d+)\s*\[.*?(?:avr_loss|loss)\s*=\s*([0-9.eE+\-]+)", re.IGNORECASE
)
_RE_STEP_LOSS = re.compile(
    r"#?step\s*(\d+)[:/]\s*(?:train\s+)?loss\s*=\s*([0-9.eE+\-]+)", re.IGNORECASE
)
_RE_SAVE = re.compile(r"saving (?:checkpoint|model)", re.IGNORECASE)


# ── ヘルパー ──────────────────────────────────────────────────────────────────

def _ensure_project(project_id: int) -> dict:
    conn = get_conn()
    row = conn.execute(
        "SELECT id, name, outputs_dir, dataset_dir FROM projects WHERE id = ?",
        (project_id,),
    ).fetchone()
    conn.close()
    if row is None:
        raise HTTPException(status_code=404, detail=f"project not found: {project_id}")
    return dict(row)


def _set_project_status(conn, project_id: int, status: str) -> None:
    conn.execute("UPDATE projects SET status = ? WHERE id = ?", (status, project_id))


def _latest_run(conn, project_id: int):
    return conn.execute(
        """
        SELECT id, status, stop_mode, current_epoch, current_step, total_epochs,
               steps_per_epoch, config_json, log_path
        FROM training_runs
        WHERE project_id = ? ORDER BY id DESC LIMIT 1
        """,
        (project_id,),
    ).fetchone()


def _ensure_runner(project_id: int) -> None:
    with RUNNER_LOCK:
        alive = RUNNER_THREADS.get(project_id)
        if alive is not None and alive.is_alive():
            return
        t = threading.Thread(target=_runner_loop, args=(project_id,), daemon=True)
        RUNNER_THREADS[project_id] = t
        t.start()


# ── アプリ設定取得 ──────────────────────────────────────────────────────────

def _get_app_settings() -> dict[str, str]:
    conn = get_conn()
    rows = conn.execute("SELECT key, value FROM app_settings").fetchall()
    conn.close()
    return {str(r["key"]): str(r["value"]) for r in rows}


def _get_train_script(kohya_root: str) -> str | None:
    p = Path(kohya_root)
    for candidate in [
        p / "train_network.py",
        p / "sd-scripts" / "train_network.py",
    ]:
        if candidate.exists():
            return str(candidate)
    return None


def _is_kohya_ready(settings: dict[str, str]) -> bool:
    python_exe = settings.get("python_exe", "").strip()
    kohya_root = settings.get("kohya_root", "").strip()
    if not python_exe or not kohya_root:
        return False
    if not Path(python_exe).exists():
        return False
    return _get_train_script(kohya_root) is not None


# ── kohya 学習ディレクトリ準備 ────────────────────────────────────────────

def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _prepare_run_dir(
    run_id: int,
    dataset_dir: str,
    repeats: int,
    project_name: str,
) -> Path:
    """kohya_ss 形式のディレクトリ構造を .runtime/runs/{run_id}/ 以下に作成する"""
    run_dir = (
        Path(__file__).resolve().parents[3]
        / ".runtime"
        / "runs"
        / str(run_id)
    )
    safe_name = re.sub(r"[^\w\-]", "_", project_name)
    train_sub = run_dir / "train_data" / f"{repeats}_{safe_name}"
    train_sub.mkdir(parents=True, exist_ok=True)
    (run_dir / "output").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)

    src = Path(dataset_dir)
    if src.exists():
        for f in src.iterdir():
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                _link_or_copy(f.resolve(), train_sub / f.name)
                txt = f.with_suffix(".txt")
                if txt.exists():
                    _link_or_copy(txt.resolve(), train_sub / txt.name)

    return run_dir


def _write_toml_config(run_dir: Path, cfg: dict) -> Path:
    """kohya_ss 用 TOML 設定ファイルを生成する"""
    config_path = run_dir / "config.toml"

    def q(v: str) -> str:
        return v.replace("\\", "\\\\").replace('"', '\\"')

    lines = [
        f'pretrained_model_name_or_path = "{q(cfg["base_checkpoint_path"])}"',
        f'train_data_dir = "{q(cfg["train_data_dir"])}"',
        f'output_dir = "{q(cfg["output_dir"])}"',
        f'output_name = "{q(cfg["output_name"])}"',
        f'resolution = "{cfg["resolution"]},{cfg["resolution"]}"',
        'network_module = "networks.lora"',
        f'network_dim = {int(cfg["rank"])}',
        f'network_alpha = {float(cfg["alpha"])}',
        f'max_train_epochs = {int(cfg["epochs"])}',
        f'save_every_n_epochs = {int(cfg["save_every_n_epochs"])}',
        'caption_extension = ".txt"',
        'shuffle_caption = true',
        f'optimizer_type = "{q(cfg.get("optimizer", "AdamW8bit"))}"',
        f'lr_scheduler = "{q(cfg.get("scheduler", "cosine_with_restarts"))}"',
        'lr_scheduler_num_cycles = 1',
        'learning_rate = 1e-4',
        'gradient_checkpointing = true',
        'mixed_precision = "fp16"',
        f'logging_dir = "{q(cfg["logs_dir"])}"',
    ]

    min_snr = cfg.get("min_snr_gamma")
    if min_snr is not None:
        lines.append(f"min_snr_gamma = {int(min_snr)}")
    reg_dir = cfg.get("reg_data_dir", "").strip()
    if reg_dir:
        lines.append(f'reg_data_dir = "{q(reg_dir)}"')

    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return config_path


# ── kohya stdout パース ────────────────────────────────────────────────────

def _parse_kohya_line(line: str) -> dict:
    result: dict = {}
    m = _RE_EPOCH.search(line)
    if m:
        result["epoch"] = int(m.group(1))
        result["total_epochs"] = int(m.group(2))
    m = _RE_TQDM.search(line)
    if m:
        result["step"] = int(m.group(1))
        result["steps_total"] = int(m.group(2))
        try:
            result["loss"] = float(m.group(3))
        except ValueError:
            pass
    m = _RE_STEP_LOSS.search(line)
    if m:
        result["step"] = int(m.group(1))
        try:
            result["loss"] = float(m.group(2))
        except ValueError:
            pass
    return result


# ── シミュレーション学習ループ ─────────────────────────────────────────────

def _write_preview_and_checkpoint(conn, project_id: int, epoch: int) -> str:
    prompt_row = conn.execute(
        "SELECT key, value FROM app_settings WHERE key IN ('positive_prompt','negative_prompt')"
    ).fetchall()
    prompt_map = {str(r["key"]): str(r["value"]) for r in prompt_row}
    pos = prompt_map.get("positive_prompt", "")
    neg = prompt_map.get("negative_prompt", "")

    row = conn.execute(
        "SELECT outputs_dir FROM projects WHERE id = ?", (project_id,)
    ).fetchone()
    outputs_dir = Path(row["outputs_dir"])
    checkpoints_dir = outputs_dir / "checkpoints"
    previews_dir = outputs_dir / "previews"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = checkpoints_dir / f"project_{project_id}_e{epoch}.safetensors"
    ckpt_path.write_text("placeholder checkpoint", encoding="utf-8")
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO checkpoints(project_id, file_path, epoch, step, mark)
        VALUES (?, ?, ?, ?, 'none')
        """,
        (project_id, str(ckpt_path), epoch, 0),
    )
    checkpoint_id = cur.lastrowid
    for slot in SLOTS:
        p = previews_dir / f"e{epoch}_{slot}.png"
        img = Image.new(
            "RGB",
            (512, 512),
            color=(30 + epoch * 10 % 200, 40 + len(slot) * 20, 90),
        )
        d = ImageDraw.Draw(img)
        d.text((24, 24), f"Project {project_id}", fill=(240, 240, 240))
        d.text((24, 56), f"Epoch {epoch}", fill=(240, 240, 240))
        d.text((24, 88), f"Slot {slot}", fill=(240, 240, 240))
        d.text((24, 120), f"+ {pos[:48]}", fill=(220, 240, 220))
        d.text((24, 148), f"- {neg[:48]}", fill=(240, 220, 220))
        img.save(p, format="PNG")
        cur.execute(
            """
            INSERT INTO preview_samples(checkpoint_id, slot, image_path)
            VALUES (?, ?, ?)
            """,
            (checkpoint_id, slot, str(p)),
        )
    return str(ckpt_path)


def _runner_loop_simulated(project_id: int) -> None:
    """シミュレーション学習ループ（kohya未接続時のフォールバック）"""
    while True:
        conn = get_conn()
        row = _latest_run(conn, project_id)
        if row is None or row["status"] != "training":
            conn.close()
            return

        epoch = int(row["current_epoch"])
        step = int(row["current_step"])
        total_epochs = int(row["total_epochs"])
        steps_per_epoch = int(row["steps_per_epoch"])
        stop_mode = row["stop_mode"]

        if epoch >= total_epochs:
            conn.execute(
                """
                UPDATE training_runs
                SET status = 'completed', updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (row["id"],),
            )
            _set_project_status(conn, project_id, "completed")
            conn.commit()
            conn.close()
            return

        # 疑似学習: 1 step 進める
        time.sleep(0.5)
        step += 1
        fake_loss = round(0.25 - (epoch * steps_per_epoch + step) / (total_epochs * steps_per_epoch) * 0.18, 4)
        if step >= steps_per_epoch:
            epoch += 1
            step = 0
            latest_ckpt = _write_preview_and_checkpoint(conn, project_id, epoch)
            conn.execute(
                """
                UPDATE training_runs
                SET latest_checkpoint_path = ?, current_epoch = ?, current_step = ?,
                    loss = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (latest_ckpt, epoch, step, fake_loss, row["id"]),
            )
            if stop_mode == "epoch":
                conn.execute(
                    """
                    UPDATE training_runs
                    SET status = 'paused', updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (row["id"],),
                )
                _set_project_status(conn, project_id, "paused")
                conn.commit()
                conn.close()
                return
        else:
            conn.execute(
                """
                UPDATE training_runs
                SET current_epoch = ?, current_step = ?, loss = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (epoch, step, fake_loss, row["id"]),
            )

        conn.commit()
        conn.close()


# ── kohya_ss 実接続学習ループ ─────────────────────────────────────────────

def _runner_loop_kohya(project_id: int, run_id: int, settings: dict, cfg: dict) -> None:
    """kohya_ss サブプロセスを起動して学習を実行する"""
    python_exe = settings["python_exe"]
    kohya_root = settings.get("kohya_root", "")
    train_script = _get_train_script(kohya_root)

    if not train_script:
        # フォールバック
        _runner_loop_simulated(project_id)
        return

    conn_p = get_conn()
    project_row = conn_p.execute(
        "SELECT name, dataset_dir, outputs_dir FROM projects WHERE id = ?", (project_id,)
    ).fetchone()
    conn_p.close()

    project_name = project_row["name"]
    dataset_dir = cfg.get("train_data_dir") or project_row["dataset_dir"]
    outputs_dir = project_row["outputs_dir"]
    repeats = int(cfg.get("repeats", 5))

    try:
        run_dir = _prepare_run_dir(run_id, dataset_dir, repeats, project_name)
    except Exception as exc:
        conn = get_conn()
        conn.execute(
            "UPDATE training_runs SET status = 'error', updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (run_id,),
        )
        conn.execute("UPDATE projects SET status = 'idle' WHERE id = ?", (project_id,))
        conn.commit()
        conn.close()
        return

    kohya_cfg = {
        "base_checkpoint_path": cfg.get("base_checkpoint_path", ""),
        "train_data_dir": str(run_dir / "train_data"),
        "output_dir": str(run_dir / "output"),
        "output_name": cfg.get("output_name", "lora_output"),
        "resolution": int(cfg.get("resolution", 512)),
        "rank": int(cfg.get("rank", 16)),
        "alpha": float(cfg.get("alpha", 8)),
        "epochs": int(cfg.get("epochs", 10)),
        "save_every_n_epochs": int(cfg.get("save_every_n_epochs", 1)),
        "optimizer": cfg.get("optimizer", "AdamW8bit"),
        "scheduler": cfg.get("scheduler", "cosine_with_restarts"),
        "min_snr_gamma": cfg.get("min_snr_gamma", 5),
        "reg_data_dir": cfg.get("reg_data_dir", ""),
        "logs_dir": str(run_dir / "logs"),
    }

    config_path = _write_toml_config(run_dir, kohya_cfg)
    log_path = run_dir / "logs" / "training.log"

    conn = get_conn()
    conn.execute(
        "UPDATE training_runs SET log_path = ? WHERE id = ?",
        (str(log_path), run_id),
    )
    conn.commit()
    conn.close()

    cmd = [python_exe, train_script, "--config_file", str(config_path)]

    log_file: IO[str] | None = None
    proc: subprocess.Popen[str] | None = None
    try:
        log_file = open(log_path, "w", encoding="utf-8", buffering=1)
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=kohya_root,
        )
        with RUNNER_LOCK:
            RUNNER_PROCESSES[project_id] = proc

        current_epoch = 0

        for line in iter(proc.stdout.readline, ""):  # type: ignore[union-attr]
            if not line:
                break
            log_file.write(line)
            log_file.flush()

            parsed = _parse_kohya_line(line)
            if not parsed:
                continue

            conn = get_conn()
            db_row = conn.execute(
                "SELECT status, stop_mode FROM training_runs WHERE id = ?", (run_id,)
            ).fetchone()

            if db_row and db_row["status"] in ("paused",):
                conn.close()
                proc.terminate()
                break

            updates: list[str] = []
            params: list = []

            if "epoch" in parsed:
                current_epoch = parsed["epoch"]
                updates.append("current_epoch = ?")
                params.append(current_epoch)
            if "step" in parsed:
                updates.append("current_step = ?")
                params.append(parsed["step"])
            if "loss" in parsed:
                updates.append("loss = ?")
                params.append(parsed["loss"])

            if updates:
                updates.append("updated_at = CURRENT_TIMESTAMP")
                conn.execute(
                    f"UPDATE training_runs SET {', '.join(updates)} WHERE id = ?",
                    [*params, run_id],
                )
                conn.commit()
            conn.close()

        proc.wait()

    except Exception as exc:
        if log_file:
            log_file.write(f"\n[ERROR] {exc}\n")
        returncode = -1
    else:
        returncode = proc.returncode if proc else -1
    finally:
        if log_file:
            log_file.close()
        with RUNNER_LOCK:
            RUNNER_PROCESSES.pop(project_id, None)

    conn = get_conn()
    run_check = conn.execute(
        "SELECT status FROM training_runs WHERE id = ?", (run_id,)
    ).fetchone()
    current_db_status = run_check["status"] if run_check else "unknown"

    if current_db_status not in ("paused",):
        if returncode == 0:
            conn.execute(
                "UPDATE training_runs SET status = 'completed', updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (run_id,),
            )
            conn.execute("UPDATE projects SET status = 'completed' WHERE id = ?", (project_id,))
        else:
            conn.execute(
                "UPDATE training_runs SET status = 'error', updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (run_id,),
            )
            conn.execute("UPDATE projects SET status = 'idle' WHERE id = ?", (project_id,))

    conn.commit()
    conn.close()


# ── メインランナーループ（dispatcher） ─────────────────────────────────────

def _runner_loop(project_id: int) -> None:
    settings = _get_app_settings()
    conn = get_conn()
    row = _latest_run(conn, project_id)
    conn.close()

    if row is None or row["status"] != "training":
        return

    run_id = int(row["id"])
    config_json = {}
    try:
        config_json = json.loads(row["config_json"] or "{}")
    except Exception:
        pass

    if _is_kohya_ready(settings):
        _runner_loop_kohya(project_id, run_id, settings, config_json)
    else:
        _runner_loop_simulated(project_id)


# ── API エンドポイント ────────────────────────────────────────────────────

@router.get("/mode")
def get_training_mode() -> dict:
    """kohya_ss 接続状態を返す"""
    settings = _get_app_settings()
    kohya_ready = _is_kohya_ready(settings)
    kohya_root = settings.get("kohya_root", "")
    train_script = _get_train_script(kohya_root) if kohya_root else None
    return {
        "mode": "kohya" if kohya_ready else "simulated",
        "kohya_ready": kohya_ready,
        "kohya_root": kohya_root,
        "train_script": train_script,
        "message": "kohya_ss 接続済み" if kohya_ready else "シミュレーションモード（kohya_ss 未接続）",
    }


@router.get("/logs/{project_id}")
def get_logs(project_id: int, lines: int = 80) -> dict:
    """最新実行の学習ログ末尾を返す"""
    _ensure_project(project_id)
    conn = get_conn()
    row = conn.execute(
        "SELECT log_path FROM training_runs WHERE project_id = ? ORDER BY id DESC LIMIT 1",
        (project_id,),
    ).fetchone()
    conn.close()

    if row is None or not row["log_path"]:
        return {"project_id": project_id, "lines": [], "log_path": None, "total_lines": 0}

    log_path = Path(row["log_path"])
    if not log_path.exists():
        return {"project_id": project_id, "lines": [], "log_path": str(log_path), "total_lines": 0}

    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.readlines()
        recent = content[-lines:]
        return {
            "project_id": project_id,
            "lines": [line.rstrip("\n") for line in recent],
            "log_path": str(log_path),
            "total_lines": len(content),
        }
    except OSError:
        return {"project_id": project_id, "lines": [], "log_path": str(log_path), "total_lines": 0}


@router.post("/start")
def start(payload: TrainingStartIn) -> dict:
    _ensure_project(payload.project_id)
    conn = get_conn()
    dataset_count = conn.execute(
        "SELECT COUNT(*) AS c FROM dataset_items WHERE project_id = ? AND selected = 1",
        (payload.project_id,),
    ).fetchone()["c"]
    steps_per_epoch = max(1, int(dataset_count) * int(payload.repeats))
    conn.close()

    config_data = {
        "preset_id": payload.preset_id,
        "epochs": payload.epochs,
        "repeats": payload.repeats,
        "alpha": payload.alpha,
        "rank": payload.rank,
        "resolution": payload.resolution,
        "save_every_n_epochs": payload.save_every_n_epochs,
        "output_name": payload.output_name,
        "base_checkpoint_path": payload.base_checkpoint_path,
        "train_data_dir": payload.train_data_dir,
        "reg_data_dir": payload.reg_data_dir,
        "optimizer": payload.optimizer,
        "scheduler": payload.scheduler,
        "min_snr_gamma": payload.min_snr_gamma,
    }

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO training_runs(
            project_id, status, stop_mode, latest_checkpoint_path, config_json,
            current_epoch, current_step, total_epochs, steps_per_epoch
        )
        VALUES (?, 'training', NULL, NULL, ?, 0, 0, ?, ?)
        """,
        (
            payload.project_id,
            json.dumps(config_data, ensure_ascii=False),
            payload.epochs,
            steps_per_epoch,
        ),
    )
    run_id = cur.lastrowid
    _set_project_status(conn, payload.project_id, "training")
    conn.commit()
    conn.close()

    settings = _get_app_settings()
    mode = "kohya" if _is_kohya_ready(settings) else "simulated"

    _ensure_runner(payload.project_id)
    return {
        "run_id": run_id,
        "project_id": payload.project_id,
        "preset_id": payload.preset_id,
        "status": "training",
        "mode": mode,
        "total_epochs": payload.epochs,
        "steps_per_epoch": steps_per_epoch,
        "dataset_images": dataset_count,
        "repeats": payload.repeats,
        "alpha": payload.alpha,
        "message": f"学習を開始しました（{mode} モード）",
    }


@router.post("/stop-now")
def stop_now(payload: TrainingControlIn) -> dict:
    _ensure_project(payload.project_id)
    conn = get_conn()
    row = _latest_run(conn, payload.project_id)
    if row is None:
        conn.close()
        raise HTTPException(status_code=400, detail="no training run")
    conn.execute(
        """
        UPDATE training_runs
        SET status = 'paused', stop_mode = 'now', updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (row["id"],),
    )
    _set_project_status(conn, payload.project_id, "paused")
    conn.commit()
    conn.close()

    # kohya サブプロセスを終了
    with RUNNER_LOCK:
        proc = RUNNER_PROCESSES.get(payload.project_id)
        if proc and proc.poll() is None:
            proc.terminate()

    return {
        "project_id": payload.project_id,
        "mode": "stop_now",
        "status": "paused",
        "message": "stopped at nearest safe point",
    }


@router.post("/stop-at-epoch")
def stop_at_epoch(payload: TrainingControlIn) -> dict:
    _ensure_project(payload.project_id)
    conn = get_conn()
    row = _latest_run(conn, payload.project_id)
    if row is None:
        conn.close()
        raise HTTPException(status_code=400, detail="no training run")
    if row["status"] != "training":
        conn.close()
        raise HTTPException(status_code=400, detail="run is not training")
    conn.execute(
        """
        UPDATE training_runs
        SET stop_mode = 'epoch', updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (row["id"],),
    )
    conn.commit()
    conn.close()
    return {
        "project_id": payload.project_id,
        "mode": "stop_at_epoch",
        "status": "training",
        "message": "epoch stop reserved",
    }


@router.post("/resume")
def resume(payload: TrainingControlIn) -> dict:
    _ensure_project(payload.project_id)
    conn = get_conn()
    row = _latest_run(conn, payload.project_id)
    if row is None:
        conn.close()
        raise HTTPException(status_code=400, detail="no training run")
    if row["status"] == "completed":
        conn.close()
        raise HTTPException(status_code=400, detail="run already completed")
    conn.execute(
        """
        UPDATE training_runs
        SET status = 'training', stop_mode = NULL, updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (row["id"],),
    )
    _set_project_status(conn, payload.project_id, "training")
    conn.commit()
    conn.close()
    _ensure_runner(payload.project_id)
    return {
        "project_id": payload.project_id,
        "status": "training",
        "message": "resume accepted",
    }


@router.get("/status")
def status(project_id: int) -> dict:
    _ensure_project(project_id)
    conn = get_conn()
    row = conn.execute(
        """
        SELECT id, status, stop_mode, latest_checkpoint_path, started_at, updated_at,
               current_epoch, current_step, total_epochs, steps_per_epoch, loss, log_path
        FROM training_runs
        WHERE project_id = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (project_id,),
    ).fetchone()
    conn.close()

    settings = _get_app_settings()
    mode = "kohya" if _is_kohya_ready(settings) else "simulated"

    if row is None:
        return {
            "project_id": project_id,
            "run_id": None,
            "status": "idle",
            "epoch": 0,
            "step": 0,
            "total_epochs": 0,
            "steps_per_epoch": 0,
            "stop_mode": None,
            "latest_checkpoint_path": None,
            "loss": None,
            "mode": mode,
            "message": "no run yet",
        }

    total_steps = int(row["total_epochs"]) * int(row["steps_per_epoch"])
    done_steps = (
        int(row["current_epoch"]) * int(row["steps_per_epoch"])
        + int(row["current_step"])
    )
    progress_percent = (done_steps / total_steps * 100.0) if total_steps > 0 else 0.0

    eta_seconds = None
    try:
        started_at = datetime.fromisoformat(str(row["started_at"]).replace(" ", "T"))
        elapsed = max(1.0, (datetime.now() - started_at).total_seconds())
        speed = done_steps / elapsed if done_steps > 0 else 0.0
        if speed > 0:
            eta_seconds = int(max(0.0, (total_steps - done_steps) / speed))
    except ValueError:
        eta_seconds = None

    return {
        "project_id": project_id,
        "run_id": row["id"],
        "status": row["status"],
        "epoch": row["current_epoch"],
        "step": row["current_step"],
        "total_epochs": row["total_epochs"],
        "steps_per_epoch": row["steps_per_epoch"],
        "total_steps": total_steps,
        "done_steps": done_steps,
        "progress_percent": round(progress_percent, 2),
        "eta_seconds": eta_seconds,
        "stop_mode": row["stop_mode"],
        "latest_checkpoint_path": row["latest_checkpoint_path"],
        "loss": row["loss"],
        "mode": mode,
        "log_path": row["log_path"],
        "started_at": row["started_at"],
        "updated_at": row["updated_at"],
        "message": f"status ({mode} mode)",
    }


@router.get("/resources")
def get_resources() -> dict:
    """
    §18 Resource Monitor — CPU / RAM / GPU 使用状況を返す。
    psutil で CPU/RAM、nvidia-smi で GPU を取得する。
    """
    # ── CPU / RAM (psutil) ────────────────────────────────────────────────
    try:
        import psutil
        cpu_pct = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        ram_used_gb = round(mem.used / 1024 ** 3, 2)
        ram_total_gb = round(mem.total / 1024 ** 3, 2)
        ram_pct = round(mem.percent, 1)
    except ImportError:
        cpu_pct = 0.0
        ram_used_gb = 0.0
        ram_total_gb = 0.0
        ram_pct = 0.0

    # ── GPU (pynvml → nvidia-smi フォールバック) ──────────────────────────
    gpu_list: list[dict] = []
    gpu_available = False

    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except Exception:
                temp = None
            vram_total_mb = mem_info.total // (1024 * 1024)
            vram_used_mb = mem_info.used // (1024 * 1024)
            gpu_list.append({
                "index": i,
                "name": name,
                "vram_used_mb": vram_used_mb,
                "vram_total_mb": vram_total_mb,
                "vram_pct": round(vram_used_mb / vram_total_mb * 100, 1) if vram_total_mb else 0.0,
                "gpu_util_pct": float(util.gpu),
                "temperature": temp,
            })
        gpu_available = len(gpu_list) > 0
        pynvml.nvmlShutdown()
    except Exception:
        # pynvml 未インストール or GPU なし → nvidia-smi で試みる
        try:
            result = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                for i, line in enumerate(result.stdout.strip().splitlines()):
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 5:
                        vram_used_mb = int(parts[1])
                        vram_total_mb = int(parts[2])
                        try:
                            temp_val: int | None = int(parts[4])
                        except ValueError:
                            temp_val = None
                        gpu_list.append({
                            "index": i,
                            "name": parts[0],
                            "vram_used_mb": vram_used_mb,
                            "vram_total_mb": vram_total_mb,
                            "vram_pct": round(vram_used_mb / vram_total_mb * 100, 1) if vram_total_mb else 0.0,
                            "gpu_util_pct": float(parts[3]),
                            "temperature": temp_val,
                        })
                gpu_available = len(gpu_list) > 0
        except Exception:
            pass

    return {
        "cpu_pct": cpu_pct,
        "ram_used_gb": ram_used_gb,
        "ram_total_gb": ram_total_gb,
        "ram_pct": ram_pct,
        "gpu": gpu_list,
        "gpu_available": gpu_available,
    }


@router.get("/dataset-preview")
def dataset_preview(project_id: int, train_data_dir: str = "") -> dict:
    _ensure_project(project_id)
    target = train_data_dir.strip()
    if not target:
        conn = get_conn()
        row = conn.execute(
            "SELECT dataset_dir FROM projects WHERE id = ?", (project_id,)
        ).fetchone()
        conn.close()
        target = str(row["dataset_dir"]) if row else ""
    d = Path(target)
    if not d.exists():
        return {"project_id": project_id, "image_path": None, "thumbnail_url": None}
    files = [p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if not files:
        return {"project_id": project_id, "image_path": None, "thumbnail_url": None}
    first = sorted(files)[0]
    try:
        with Image.open(first) as im:
            im = im.convert("RGB")
            im.thumbnail((320, 320))
            from io import BytesIO
            buf = BytesIO()
            im.save(buf, format="JPEG", quality=82)
            thumb = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except OSError:
        thumb = None
    return {"project_id": project_id, "image_path": str(first), "thumbnail_url": thumb}
