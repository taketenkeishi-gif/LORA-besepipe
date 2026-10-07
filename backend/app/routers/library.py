"""
§22-23 Asset Library — LoRA 資産管理 CRUD
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel

from ..db import get_conn

router = APIRouter(prefix="/library", tags=["library"])


# ── Pydantic スキーマ ─────────────────────────────────────────────────────────

class AssetCreateIn(BaseModel):
    project_id: int | None = None
    name: str
    lora_path: str = ""
    base_model: str = ""
    dataset_size: int = 0
    profile_name: str = ""
    tags: list[str] = []
    notes: str = ""
    preview_path: str = ""
    training_config: dict[str, Any] = {}
    quality_score: int | None = None
    asset_type: str = "character"


class AssetUpdateIn(BaseModel):
    name: str | None = None
    lora_path: str | None = None
    base_model: str | None = None
    dataset_size: int | None = None
    profile_name: str | None = None
    tags: list[str] | None = None
    notes: str | None = None
    preview_path: str | None = None
    training_config: dict[str, Any] | None = None
    quality_score: int | None = None
    asset_type: str | None = None


class LibraryExportIn(BaseModel):
    asset_id: int


class LibrarySelectFinalIn(BaseModel):
    human_confirmed: bool = False
    note: str = ""


AESTHETIC_UNREVIEWED = "AESTHETIC_UNREVIEWED"
USER_APPROVED = "USER_APPROVED"


# ── ヘルパー ──────────────────────────────────────────────────────────────────

def _row_to_dict(row) -> dict:
    d = dict(row)
    # tags_json → tags list
    try:
        d["tags"] = json.loads(d.pop("tags_json", "[]"))
    except Exception:
        d["tags"] = []
    # training_config_json → dict
    try:
        d["training_config"] = json.loads(d.pop("training_config_json", "{}"))
    except Exception:
        d["training_config"] = {}
    return d


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_name(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z._-]+", "_", value).strip("._")
    return value or "lora_asset"


def _lineage_metadata(conn, row) -> dict[str, Any]:
    """Return the navigable Run → Snapshot → Checkpoint → Preview lineage."""
    config = json.loads(row["training_config_json"] or "{}")
    source_run = conn.execute(
        "SELECT status FROM training_runs WHERE id = ?",
        (row["training_run_id"],),
    ).fetchone() if row["training_run_id"] is not None else None
    source_assets = []
    if row["dataset_snapshot_id"] is not None:
        source_assets = conn.execute(
            "SELECT a.id, a.asset_key FROM basepipe_snapshot_entries e "
            "JOIN basepipe_assets a ON a.id = e.asset_id "
            "WHERE e.snapshot_id = ? ORDER BY e.ordinal",
            (row["dataset_snapshot_id"],),
        ).fetchall()
    snapshot = conn.execute(
        "SELECT snapshot_hash, item_count, status, manifest_json FROM basepipe_dataset_snapshots WHERE id = ?",
        (row["dataset_snapshot_id"],),
    ).fetchone() if row["dataset_snapshot_id"] is not None else None
    snapshot_manifest: dict[str, Any] = {}
    if snapshot is not None:
        try:
            parsed_manifest = json.loads(snapshot["manifest_json"] or "{}")
            if isinstance(parsed_manifest, dict):
                snapshot_manifest = parsed_manifest
        except (TypeError, ValueError):
            snapshot_manifest = {}
    source_binding = config.get("library_source") if isinstance(config.get("library_source"), dict) else {}
    bound_checkpoint_id = source_binding.get("checkpoint_id")
    if bound_checkpoint_id is not None and row["training_run_id"] is not None:
        checkpoint = conn.execute(
            "SELECT id, file_path, epoch, step, validation_status FROM checkpoints "
            "WHERE id = ? AND run_id = ? AND project_id = ?",
            (bound_checkpoint_id, row["training_run_id"], row["project_id"]),
        ).fetchone()
    else:
        # Legacy candidates created before exact Checkpoint binding existed.
        checkpoint = conn.execute(
            "SELECT id, file_path, epoch, step, validation_status FROM checkpoints "
            "WHERE run_id = ? ORDER BY epoch DESC, id DESC LIMIT 1",
            (row["training_run_id"],),
        ).fetchone() if row["training_run_id"] is not None else None
    bound_profile_id = source_binding.get("preview_profile_snapshot_id") or config.get("preview_profile_snapshot_id")
    if bound_profile_id is not None and row["project_id"] is not None:
        profile = conn.execute(
            "SELECT id, snapshot_hash, payload_json FROM basepipe_preview_profile_snapshots "
            "WHERE id = ? AND project_id = ?",
            (bound_profile_id, row["project_id"]),
        ).fetchone()
    else:
        profile = conn.execute(
            "SELECT id, snapshot_hash, payload_json FROM basepipe_preview_profile_snapshots "
            "WHERE project_id = ? ORDER BY id DESC LIMIT 1",
            (row["project_id"],),
        ).fetchone() if row["project_id"] is not None else None
    concept = conn.execute(
        "SELECT c.trigger_token FROM basepipe_concepts c JOIN basepipe_dataset_snapshots s ON s.concept_id = c.id WHERE s.id = ?",
        (row["dataset_snapshot_id"],),
    ).fetchone() if row["dataset_snapshot_id"] is not None else None
    preview_count = 0
    preview_success_count = 0
    jobs = []
    if checkpoint is not None:
        jobs = conn.execute(
            "SELECT id, prompt_index, instance_index, status, output_path, conditions_json, preview_snapshot_json "
            "FROM preview_jobs WHERE checkpoint_id = ? ORDER BY prompt_index, instance_index",
            (checkpoint["id"],),
        ).fetchall()
        preview_count = len(jobs)
        if preview_count == 0:
            # Legacy runs may only have sample rows; preserve their evidence
            # without presenting sample-row count as preview-job count.
            preview_count = int(conn.execute(
                "SELECT COUNT(*) FROM preview_samples WHERE checkpoint_id = ?", (checkpoint["id"],)
            ).fetchone()[0])
        preview_success_count = sum(
            1 for job in jobs
            if job["status"] == "succeeded" and Path(str(job["output_path"] or "")).is_file()
        )
    evaluation = None
    if checkpoint is not None:
        accepted_rows = conn.execute(
            "SELECT preview_slot, preview_profile_snapshot_id, accepted, review_state, reviewed_by FROM basepipe_evaluations "
            "WHERE checkpoint_id = ? AND project_id = ? AND accepted = 1 "
            "AND review_state = ? AND reviewed_by = 'user' ORDER BY updated_at DESC",
            (checkpoint["id"], row["project_id"], USER_APPROVED),
        ).fetchall()
        jobs_by_slot = {
            f"p{job['prompt_index']}_i{job['instance_index']}": job for job in jobs
        }
        for accepted in accepted_rows:
            accepted_job = jobs_by_slot.get(str(accepted["preview_slot"]))
            if accepted_job is None or accepted_job["status"] != "succeeded":
                continue
            if not Path(str(accepted_job["output_path"] or "")).is_file():
                continue
            try:
                accepted_snapshot = json.loads(accepted_job["preview_snapshot_json"] or "{}")
            except (TypeError, ValueError):
                continue
            if accepted_snapshot.get("profile_snapshot_id") != accepted["preview_profile_snapshot_id"]:
                continue
            evaluation = accepted
            break
    comparison_profile = profile
    if evaluation and evaluation["preview_profile_snapshot_id"] is not None:
        comparison_profile = conn.execute(
            "SELECT id, snapshot_hash, payload_json FROM basepipe_preview_profile_snapshots "
            "WHERE id = ? AND project_id = ?",
            (evaluation["preview_profile_snapshot_id"], row["project_id"]),
        ).fetchone()
    preview_condition_mismatch_count = 0
    preview_condition_mismatch_fields: list[str] = []
    preview_condition_match = False
    if checkpoint is not None and preview_count > 0 and comparison_profile is not None:
        try:
            expected = json.loads(comparison_profile["payload_json"] or "{}")
        except (TypeError, ValueError):
            expected = {}
        if len(jobs) == preview_count:
            for job in jobs:
                try:
                    preview_snapshot = json.loads(job["preview_snapshot_json"] or "{}")
                    conditions = json.loads(job["conditions_json"] or "{}")
                except (TypeError, ValueError):
                    preview_condition_mismatch_count += 1
                    preview_condition_mismatch_fields.append(f"job:{job['id']} parse")
                    continue
                actual_resolution = conditions.get("resolution") or preview_snapshot.get("width")
                comparisons = (
                    ("prompt", expected.get("prompt"), preview_snapshot.get("prompt")),
                    ("negative_prompt", expected.get("negative_prompt"), preview_snapshot.get("negative_prompt")),
                    ("seed", expected.get("seed"), preview_snapshot.get("seed")),
                    ("resolution", expected.get("resolution"), actual_resolution),
                    # conditions_json is the immutable requested/frozen Preview
                    # evidence. Do not let an older snapshot hide a mismatch.
                    ("steps", expected.get("steps"), conditions.get("steps") or preview_snapshot.get("steps")),
                    ("cfg", expected.get("cfg"), conditions.get("cfg") or preview_snapshot.get("cfg")),
                    ("sampler", expected.get("sampler"), conditions.get("sampler") or preview_snapshot.get("sampler")),
                    ("scheduler", expected.get("scheduler"), conditions.get("scheduler") or preview_snapshot.get("scheduler")),
                    ("model_family", expected.get("model_family"), conditions.get("model_family")),
                )
                mismatch_keys: list[str] = []
                for key, expected_value, actual_value in comparisons:
                    if expected_value is None:
                        continue
                    if key == "cfg":
                        try:
                            equal = abs(float(expected_value) - float(actual_value)) < 1e-6
                        except (TypeError, ValueError):
                            equal = False
                    else:
                        equal = expected_value == actual_value
                    if not equal:
                        mismatch_keys.append(key)
                if job["status"] != "succeeded":
                    mismatch_keys.append("status")
                if preview_snapshot.get("profile_snapshot_id") != comparison_profile["id"]:
                    mismatch_keys.append("profile_snapshot_id")
                if mismatch_keys:
                    preview_condition_mismatch_count += 1
                    preview_condition_mismatch_fields.extend(
                        f"job:{job['id']} {key}" for key in mismatch_keys
                    )
            preview_condition_match = preview_condition_mismatch_count == 0
    export_info = config.get("library_export") or {}
    final_selection = config.get("final_selection") if isinstance(config.get("final_selection"), dict) else {}
    export_path = export_info.get("export_path")
    source_path = Path(str(row["lora_path"] or ""))
    checkpoint_path = Path(str(checkpoint["file_path"] or "")) if checkpoint is not None else None
    source_artifact_match = bool(
        checkpoint_path is not None
        and source_path.is_file()
        and checkpoint_path.is_file()
        and os.path.normcase(str(source_path.resolve())) == os.path.normcase(str(checkpoint_path.resolve()))
    )
    expected_checkpoint_hash = str(source_binding.get("checkpoint_sha256") or "")
    if source_artifact_match and expected_checkpoint_hash:
        source_artifact_match = _sha256(source_path) == expected_checkpoint_hash
    lineage_gate = {
        "run_completed": bool(source_run and source_run["status"] == "completed"),
        "snapshot_sealed": bool(snapshot and snapshot["status"] == "sealed"),
        "checkpoint_present": checkpoint is not None,
        "source_artifact_match": source_artifact_match,
        "preview_complete": preview_count > 0 and preview_success_count == preview_count,
        "preview_conditions_match": preview_condition_match,
        "human_evaluation": evaluation is not None,
        "final_selected": bool(
            str(row["library_status"] or "candidate") == "accepted"
            and final_selection.get("aesthetic_review_state") == USER_APPROVED
            and final_selection.get("reviewed_by") == "user"
        ),
        "exported": bool(export_path and Path(str(export_path)).is_file()),
    }
    selected_profile = comparison_profile or profile
    return {
        "source_run_id": row["training_run_id"],
        "source_run_status": source_run["status"] if source_run else None,
        "source_snapshot_id": row["dataset_snapshot_id"],
        "source_snapshot_hash": snapshot["snapshot_hash"] if snapshot else config.get("dataset_snapshot_hash"),
        "source_snapshot_item_count": snapshot["item_count"] if snapshot else config.get("dataset_snapshot_item_count"),
        "outfit_profiles": snapshot_manifest.get("outfit_profiles", []),
        "source_asset_ids": [int(asset["id"]) for asset in source_assets],
        "source_asset_keys": [str(asset["asset_key"]) for asset in source_assets],
        "source_checkpoint_id": checkpoint["id"] if checkpoint else None,
        "source_checkpoint_path": checkpoint["file_path"] if checkpoint else None,
        "source_checkpoint_sha256": expected_checkpoint_hash or None,
        "source_checkpoint_epoch": checkpoint["epoch"] if checkpoint else None,
        "checkpoint_validation_status": checkpoint["validation_status"] if checkpoint else None,
        "model_family": config.get("model_family"),
        "rank": config.get("rank"),
        "alpha": config.get("alpha"),
        "trigger_token": concept["trigger_token"] if concept else export_info.get("trigger_token"),
        "preview_profile_snapshot_id": selected_profile["id"] if selected_profile else export_info.get("preview_profile_snapshot_id"),
        "preview_profile_snapshot_hash": selected_profile["snapshot_hash"] if selected_profile else export_info.get("preview_profile_snapshot_hash"),
        "accepted_evaluation_preview_snapshot_id": evaluation["preview_profile_snapshot_id"] if evaluation else None,
        "preview_count": preview_count,
        "preview_success_count": preview_success_count,
        "preview_condition_mismatch_count": preview_condition_mismatch_count,
        "preview_condition_mismatch_fields": preview_condition_mismatch_fields,
        "export_path": export_path,
        "export_sha256": export_info.get("export_sha256"),
        "lineage_gate": lineage_gate,
    }


# ── エンドポイント ────────────────────────────────────────────────────────────

@router.get("/assets")
def list_assets(project_id: int | None = None, asset_type: str | None = None) -> list[dict]:
    """LoRA 資産一覧を取得する。project_id / asset_type でフィルタ可能。"""
    conn = get_conn()
    query = "SELECT * FROM lora_assets WHERE 1=1"
    params: list = []
    if project_id is not None:
        query += " AND project_id = ?"
        params.append(project_id)
    if asset_type is not None:
        query += " AND asset_type = ?"
        params.append(asset_type)
    query += " ORDER BY created_at DESC"
    rows = conn.execute(query, params).fetchall()
    result = [dict(_row_to_dict(r), lineage=_lineage_metadata(conn, r)) for r in rows]
    conn.close()
    return result


@router.post("/export")
def export_library_asset(payload: LibraryExportIn) -> dict:
    """Export an explicitly accepted asset while preserving its complete lineage."""
    conn = get_conn()
    temp_destination: Path | None = None
    backup_destination: Path | None = None
    destination: Path | None = None
    replaced_existing = False
    try:
        row = conn.execute("SELECT * FROM lora_assets WHERE id = ?", (payload.asset_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail=f"asset not found: {payload.asset_id}")
        if str(row["library_status"] or "candidate") != "accepted":
            raise HTTPException(status_code=400, detail="Final採用済みAssetだけをExportできます")
        lineage = _lineage_metadata(conn, row)
        if row["training_run_id"] is not None:
            lineage_gate = lineage["lineage_gate"]
            required_gates = (
                "run_completed", "snapshot_sealed", "checkpoint_present",
                "source_artifact_match",
                "preview_complete", "preview_conditions_match", "human_evaluation",
                "final_selected",
            )
            missing_gates = [key for key in required_gates if not lineage_gate.get(key)]
            if missing_gates:
                raise HTTPException(
                    status_code=400,
                    detail=f"Export前のLineage Gate未通過: {', '.join(missing_gates)}",
                )
        source = Path(str(row["lora_path"] or ""))
        if not source.is_file():
            raise HTTPException(status_code=400, detail=f"LoRA source file not found: {source}")
        if row["project_id"] is None:
            raise HTTPException(status_code=400, detail="プロジェクト未関連のAssetはExportできません")

        project = conn.execute("SELECT id, base_dir FROM projects WHERE id = ?", (row["project_id"],)).fetchone()
        if project is None:
            raise HTTPException(status_code=404, detail="source project not found")

        # Export先はプロジェクト配下に固定し、旧設定の外部datasetパスには書き込まない。
        export_dir = Path(str(project["base_dir"])) / "library"
        export_dir.mkdir(parents=True, exist_ok=True)
        destination = export_dir / f"{_safe_name(str(row['name']))}_asset{int(row['id'])}{source.suffix.lower()}"
        temp_destination = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        backup_destination = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.bak")
        shutil.copy2(source, temp_destination)
        export_hash = _sha256(temp_destination)

        checkpoint = conn.execute(
            "SELECT id, validation_status FROM checkpoints WHERE id = ?",
            (lineage["source_checkpoint_id"],),
        ).fetchone() if lineage.get("source_checkpoint_id") is not None else None
        snapshot = conn.execute(
            "SELECT id, snapshot_hash FROM basepipe_dataset_snapshots WHERE id = ?",
            (row["dataset_snapshot_id"],),
        ).fetchone() if row["dataset_snapshot_id"] is not None else None
        profile_snapshot = None
        if lineage.get("preview_profile_snapshot_id") is not None:
            profile_snapshot = conn.execute(
                "SELECT id, snapshot_hash FROM basepipe_preview_profile_snapshots WHERE id = ? AND project_id = ?",
                (lineage["preview_profile_snapshot_id"], row["project_id"]),
            ).fetchone()
        concept = conn.execute(
            "SELECT c.trigger_token FROM basepipe_concepts c JOIN basepipe_dataset_snapshots s ON s.concept_id = c.id WHERE s.id = ?",
            (row["dataset_snapshot_id"],),
        ).fetchone() if row["dataset_snapshot_id"] is not None else None

        config = json.loads(row["training_config_json"] or "{}")
        export_info = {
            "export_path": str(destination),
            "export_sha256": export_hash,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "source_run_id": int(row["training_run_id"]) if row["training_run_id"] is not None else None,
            "source_snapshot_id": int(row["dataset_snapshot_id"]) if row["dataset_snapshot_id"] is not None else None,
            "source_snapshot_hash": snapshot["snapshot_hash"] if snapshot else None,
            "source_checkpoint_id": int(checkpoint["id"]) if checkpoint else None,
            "preview_profile_snapshot_id": int(profile_snapshot["id"]) if profile_snapshot else lineage.get("preview_profile_snapshot_id"),
            "preview_profile_snapshot_hash": profile_snapshot["snapshot_hash"] if profile_snapshot else lineage.get("preview_profile_snapshot_hash"),
            "trigger_token": concept["trigger_token"] if concept else None,
            "outfit_profiles": lineage.get("outfit_profiles", []),
        }
        config["library_export"] = export_info
        replaced_existing = destination.exists()
        if replaced_existing:
            os.replace(destination, backup_destination)
        os.replace(temp_destination, destination)
        temp_destination = None
        conn.execute(
            "UPDATE lora_assets SET training_config_json = ? WHERE id = ?",
            (json.dumps(config, ensure_ascii=False), payload.asset_id),
        )
        exported = conn.execute("SELECT * FROM lora_assets WHERE id = ?", (payload.asset_id,)).fetchone()
        result = _row_to_dict(exported)
        result["lineage"] = _lineage_metadata(conn, exported)
        conn.commit()
        try:
            if backup_destination.exists():
                backup_destination.unlink()
        except OSError:
            # The committed export and lineage are authoritative. A stale
            # internal backup is recoverable cleanup, not a failed export.
            pass
        return result
    except HTTPException:
        conn.rollback()
        raise
    except Exception as exc:
        conn.rollback()
        if destination is not None and destination.exists() and backup_destination is not None:
            destination.unlink(missing_ok=True)
        if replaced_existing and backup_destination is not None and backup_destination.exists() and destination is not None:
            os.replace(backup_destination, destination)
        if temp_destination is not None:
            temp_destination.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Library export failed: {exc}") from exc
    finally:
        conn.close()


@router.post("/assets", status_code=201)
def create_asset(payload: AssetCreateIn) -> dict:
    """LoRA 資産を新規作成する。"""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO lora_assets (
            project_id, name, lora_path, base_model, dataset_size,
            profile_name, tags_json, notes, preview_path,
            training_config_json, quality_score, asset_type
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            payload.project_id,
            payload.name,
            payload.lora_path,
            payload.base_model,
            payload.dataset_size,
            payload.profile_name,
            json.dumps(payload.tags, ensure_ascii=False),
            payload.notes,
            payload.preview_path,
            json.dumps(payload.training_config, ensure_ascii=False),
            payload.quality_score,
            payload.asset_type,
        ),
    )
    asset_id = cur.lastrowid
    conn.commit()
    row = conn.execute("SELECT * FROM lora_assets WHERE id = ?", (asset_id,)).fetchone()
    result = _row_to_dict(row)
    result["lineage"] = _lineage_metadata(conn, row)
    conn.close()
    return result


@router.get("/assets/{asset_id}")
def get_asset(asset_id: int) -> dict:
    """LoRA 資産を 1 件取得する。"""
    conn = get_conn()
    row = conn.execute("SELECT * FROM lora_assets WHERE id = ?", (asset_id,)).fetchone()
    if row is None:
        conn.close()
        raise HTTPException(status_code=404, detail=f"asset not found: {asset_id}")
    result = _row_to_dict(row)
    result["lineage"] = _lineage_metadata(conn, row)
    conn.close()
    return result


@router.get("/assets/{asset_id}/lineage")
def get_asset_lineage(asset_id: int) -> dict:
    """Return lineage and machine-readable acceptance gates for one Asset."""
    conn = get_conn()
    try:
        row = conn.execute("SELECT * FROM lora_assets WHERE id = ?", (asset_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail=f"asset not found: {asset_id}")
        return {
            "asset_id": asset_id,
            "library_status": row["library_status"] or "candidate",
            "lineage": _lineage_metadata(conn, row),
        }
    finally:
        conn.close()


@router.post("/assets/{asset_id}/select-final")
def select_final_asset(asset_id: int, payload: LibrarySelectFinalIn = LibrarySelectFinalIn()) -> dict:
    """Promote a trained candidate only after an explicit human Compare evaluation."""
    if not payload.human_confirmed:
        raise HTTPException(
            status_code=400,
            detail="Final選択は実Previewを確認したユーザー本人の明示確認が必要です",
        )
    conn = get_conn()
    try:
        row = conn.execute("SELECT * FROM lora_assets WHERE id = ?", (asset_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail=f"asset not found: {asset_id}")
        run_id = row["training_run_id"]
        if run_id is None:
            raise HTTPException(status_code=400, detail="Source RunがないAssetはFinal選択できません")
        lineage = _lineage_metadata(conn, row)
        gate = lineage["lineage_gate"]
        if not gate["run_completed"]:
            raise HTTPException(status_code=400, detail="Source Run完了後にFinal選択してください")
        if not gate["snapshot_sealed"]:
            raise HTTPException(status_code=400, detail="Dataset Snapshot封印後にFinal選択してください")
        if not gate["checkpoint_present"]:
            raise HTTPException(status_code=400, detail="Checkpoint生成後にFinal選択してください")
        if not gate["source_artifact_match"]:
            raise HTTPException(status_code=400, detail="評価対象CheckpointとLibrary Assetの実ファイルが一致しません")
        if not gate["preview_complete"]:
            raise HTTPException(status_code=400, detail="Preview全件成功後にFinal選択してください")
        if not gate["preview_conditions_match"]:
            raise HTTPException(status_code=400, detail="固定Preview条件と全Previewの一致確認後にFinal選択してください")
        if not gate["human_evaluation"]:
            raise HTTPException(status_code=400, detail="固定Preview条件でCompare人間評価を保存してからFinal選択してください")
        config = json.loads(row["training_config_json"] or "{}")
        config["final_selection"] = {
            "aesthetic_review_state": USER_APPROVED,
            "reviewed_by": "user",
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
            "note": payload.note,
            "source_checkpoint_id": lineage.get("source_checkpoint_id"),
            "preview_profile_snapshot_id": lineage.get("accepted_evaluation_preview_snapshot_id"),
        }
        conn.execute(
            "UPDATE lora_assets SET library_status = 'accepted', training_config_json = ? WHERE id = ?",
            (json.dumps(config, ensure_ascii=False), asset_id),
        )
        conn.commit()
        updated = conn.execute("SELECT * FROM lora_assets WHERE id = ?", (asset_id,)).fetchone()
        result = _row_to_dict(updated)
        result["lineage"] = _lineage_metadata(conn, updated)
        return result
    except HTTPException:
        conn.rollback()
        raise
    finally:
        conn.close()


@router.put("/assets/{asset_id}")
def update_asset(asset_id: int, payload: AssetUpdateIn) -> dict:
    """LoRA 資産を更新する（指定フィールドのみ上書き）。"""
    conn = get_conn()
    row = conn.execute("SELECT * FROM lora_assets WHERE id = ?", (asset_id,)).fetchone()
    if row is None:
        conn.close()
        raise HTTPException(status_code=404, detail=f"asset not found: {asset_id}")
    if row["training_run_id"] is not None:
        if payload.lora_path is not None and payload.lora_path != str(row["lora_path"] or ""):
            conn.close()
            raise HTTPException(status_code=409, detail="学習Run由来AssetのCheckpointパスは不変です")
        if payload.training_config is not None:
            conn.close()
            raise HTTPException(status_code=409, detail="学習Run由来AssetのLineage設定は不変です")

    updates: list[str] = []
    params: list = []

    if payload.name is not None:
        updates.append("name = ?"); params.append(payload.name)
    if payload.lora_path is not None:
        updates.append("lora_path = ?"); params.append(payload.lora_path)
    if payload.base_model is not None:
        updates.append("base_model = ?"); params.append(payload.base_model)
    if payload.dataset_size is not None:
        updates.append("dataset_size = ?"); params.append(payload.dataset_size)
    if payload.profile_name is not None:
        updates.append("profile_name = ?"); params.append(payload.profile_name)
    if payload.tags is not None:
        updates.append("tags_json = ?"); params.append(json.dumps(payload.tags, ensure_ascii=False))
    if payload.notes is not None:
        updates.append("notes = ?"); params.append(payload.notes)
    if payload.preview_path is not None:
        updates.append("preview_path = ?"); params.append(payload.preview_path)
    if payload.training_config is not None:
        updates.append("training_config_json = ?"); params.append(json.dumps(payload.training_config, ensure_ascii=False))
    if payload.quality_score is not None:
        updates.append("quality_score = ?"); params.append(payload.quality_score)
    if payload.asset_type is not None:
        updates.append("asset_type = ?"); params.append(payload.asset_type)

    if updates:
        params.append(asset_id)
        conn.execute(f"UPDATE lora_assets SET {', '.join(updates)} WHERE id = ?", params)
        conn.commit()

    row = conn.execute("SELECT * FROM lora_assets WHERE id = ?", (asset_id,)).fetchone()
    conn.close()
    return _row_to_dict(row)


@router.delete("/assets/{asset_id}")
def delete_asset(asset_id: int) -> Response:
    """LoRA 資産を削除する（ファイルは削除しない）。"""
    conn = get_conn()
    row = conn.execute("SELECT id FROM lora_assets WHERE id = ?", (asset_id,)).fetchone()
    if row is None:
        conn.close()
        raise HTTPException(status_code=404, detail=f"asset not found: {asset_id}")
    conn.execute("DELETE FROM lora_assets WHERE id = ?", (asset_id,))
    conn.commit()
    conn.close()
    return Response(status_code=204)


@router.get("/stats")
def library_stats() -> dict:
    """ライブラリ全体の統計情報を返す。"""
    conn = get_conn()
    total = conn.execute("SELECT COUNT(*) FROM lora_assets").fetchone()[0]
    by_type = conn.execute(
        "SELECT asset_type, COUNT(*) AS cnt FROM lora_assets GROUP BY asset_type"
    ).fetchall()
    avg_qs = conn.execute(
        "SELECT AVG(quality_score) FROM lora_assets WHERE quality_score IS NOT NULL"
    ).fetchone()[0]
    conn.close()
    return {
        "total": total,
        "by_type": {row["asset_type"]: row["cnt"] for row in by_type},
        "avg_quality_score": round(avg_qs, 1) if avg_qs is not None else None,
    }
