"""シミュレーションエンジン — 学習ツール未接続時のフォールバック。

実際のサブプロセスを起動せず、DBを直接更新しながら疑似的な学習進行と
プレースホルダのチェックポイント/プレビュー画像を生成する。UI 上は他の
エンジンと同じ Training フローで動作して見える。
"""
from __future__ import annotations

import time
from pathlib import Path

from PIL import Image, ImageDraw

from ....db import get_conn
from ...base import PreparedRun, TrainingBackend, TrainingContext
from ...runtime import auto_register_lora_asset, latest_run, set_project_status

SLOTS = ["face", "bust", "full", "bg"]


def _write_preview_and_checkpoint(conn, project_id: int, run_id: int, epoch: int) -> str:
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
        INSERT INTO checkpoints(project_id, run_id, file_path, epoch, step, mark, validation_status)
        VALUES (?, ?, ?, ?, ?, 'none', 'simulated')
        """,
        (project_id, run_id, str(ckpt_path), epoch, 0),
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


class SimulatedBackend(TrainingBackend):
    display_name = "Simulation"

    def is_ready(self, settings: dict[str, str]) -> bool:
        # シミュレーションは常に実行可能（他エンジン未接続時の最終フォールバック）。
        return True

    def prepare(self, ctx: TrainingContext) -> PreparedRun:
        # 実際のサブプロセスを起動しないため、train_dir 等の準備は不要。
        # PreparedRun はインターフェース互換のためのプレースホルダ。
        return PreparedRun(
            run_dir=Path("."), config_path=Path("."), log_path=Path("."),
            cmd=[], cwd=".", python_exe="", env=None, extra={},
        )

    def train(self, ctx: TrainingContext, prepared: PreparedRun) -> None:
        """疑似学習ループ（DB を直接更新しながら進行する）。"""
        project_id = ctx.project_id
        while True:
            conn = get_conn()
            row = latest_run(conn, project_id)
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
                set_project_status(conn, project_id, "completed")
                auto_register_lora_asset(conn, project_id, int(row["id"]))
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
                latest_ckpt = _write_preview_and_checkpoint(conn, project_id, int(row["id"]), epoch)
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
                    set_project_status(conn, project_id, "paused")
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
