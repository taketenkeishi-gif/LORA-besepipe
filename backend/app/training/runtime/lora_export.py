"""学習完了時のLoRA candidate登録。外部ExportはFinal選択後の明示操作だけで行う。"""
from __future__ import annotations

import json
import logging
import hashlib
from pathlib import Path

logger = logging.getLogger(__name__)


def export_lora_to_output_dir(conn, lora_path: str) -> None:
    """app_settings の lora_output_dir へ LoRA ファイルをコピーする。"""
    try:
        row = conn.execute(
            "SELECT value FROM app_settings WHERE key='lora_output_dir'"
        ).fetchone()
        dest_dir = (row["value"] if row else "").strip()
        if not dest_dir or not lora_path:
            return
        src = Path(lora_path)
        if not src.exists():
            return
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(src, dest / src.name)
        logger.info(f"LoRA exported: {src.name} → {dest}")
    except Exception as e:
        logger.warning(f"LoRA export failed: {e}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def auto_register_lora_asset(conn, project_id: int, run_id: int) -> int:
    """学習完了時に lora_assets へ自動登録する（重複スキップ）。"""
    existing = conn.execute(
        "SELECT id FROM lora_assets WHERE project_id = ? AND training_run_id = ?",
        (project_id, run_id),
    ).fetchone()
    if existing:
        return int(existing["id"])

    run = conn.execute(
        "SELECT latest_checkpoint_path, config_json, loss, dataset_snapshot_id FROM training_runs WHERE id = ?",
        (run_id,),
    ).fetchone()
    proj = conn.execute("SELECT name FROM projects WHERE id = ?", (project_id,)).fetchone()
    if not run or not proj:
        raise RuntimeError(f"RunまたはProjectが見つかりません: run={run_id}, project={project_id}")

    cfg = json.loads(run["config_json"] or "{}")
    checkpoint = conn.execute(
        "SELECT id, file_path, epoch, step, validation_status FROM checkpoints "
        "WHERE run_id = ? ORDER BY epoch DESC, id DESC LIMIT 1",
        (run_id,),
    ).fetchone()
    if checkpoint is None:
        raise RuntimeError(f"Run #{run_id} に登録済みCheckpointがありません")
    source = Path(str(checkpoint["file_path"] or ""))
    if not source.is_file():
        raise RuntimeError(f"Checkpointファイルが見つかりません: {source}")
    if str(checkpoint["validation_status"] or "none") == "invalid":
        raise RuntimeError(f"Checkpoint #{checkpoint['id']} はvalidation_status=invalidです")
    lora_path = str(source)
    snapshot = conn.execute(
        "SELECT item_count, snapshot_hash FROM basepipe_dataset_snapshots WHERE id = ? AND project_id = ?",
        (run["dataset_snapshot_id"], project_id),
    ).fetchone() if run["dataset_snapshot_id"] is not None else None
    dataset_size = int(snapshot["item_count"]) if snapshot is not None else 0
    cfg["library_source"] = {
        "checkpoint_id": int(checkpoint["id"]),
        "checkpoint_path": lora_path,
        "checkpoint_sha256": _sha256(source),
        "checkpoint_epoch": int(checkpoint["epoch"] or 0),
        "checkpoint_step": int(checkpoint["step"] or 0),
        "checkpoint_validation_status": str(checkpoint["validation_status"] or "none"),
        "dataset_snapshot_id": int(run["dataset_snapshot_id"]) if run["dataset_snapshot_id"] is not None else None,
        "dataset_snapshot_hash": snapshot["snapshot_hash"] if snapshot is not None else None,
        "preview_profile_snapshot_id": cfg.get("preview_profile_snapshot_id"),
    }

    preview_row = conn.execute(
        "SELECT output_path FROM preview_jobs WHERE checkpoint_id = ? AND status = 'succeeded' "
        "ORDER BY prompt_index, instance_index LIMIT 1",
        (checkpoint["id"],),
    ).fetchone()
    preview_path = (preview_row["output_path"] if preview_row else None) or ""

    cur = conn.execute(
        """INSERT INTO lora_assets (
                project_id, training_run_id, dataset_snapshot_id, name, lora_path, base_model,
                dataset_size, profile_name, tags_json, notes,
                preview_path, training_config_json, quality_score, asset_type, library_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                project_id,
                run_id,
                run["dataset_snapshot_id"],
                proj["name"],
                lora_path,
                cfg.get("base_model", ""),
                dataset_size,
                cfg.get("preset_name", ""),
                "[]",
                f"自動登録 — 学習完了 (run {run_id})",
                preview_path,
                json.dumps(cfg, ensure_ascii=False),
                None,
                "lora",
                "candidate",
            ),
        )
    logger.info("LoRA candidate registered: asset=%s run=%s checkpoint=%s", cur.lastrowid, run_id, checkpoint["id"])
    return int(cur.lastrowid)
