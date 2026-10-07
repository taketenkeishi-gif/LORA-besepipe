"""Select resume weights from the current run or a provably compatible sealed input."""
from __future__ import annotations
import json
from pathlib import Path
from .registry import detect_model_family, resolve_training_backend


def _identity(config: dict) -> tuple | None:
    model = str(config.get('base_checkpoint_path') or '').strip()
    output = str(config.get('output_name') or '').strip()
    snapshot = config.get('dataset_snapshot_id')
    # A matching output label or mutable dataset directory does not establish lineage.
    if not model or not output or snapshot is None or config.get('rank') is None or config.get('alpha') is None:
        return None
    family = detect_model_family(model, str(config.get('model_family') or 'auto'))
    rank = int(config['rank'])
    return (family, resolve_training_backend(family, config.get('training_engine', 'auto')),
            str(Path(model).resolve()).casefold(), output, snapshot,
            config.get('dataset_snapshot_hash'), rank, float(config['alpha']))


def select_resume_checkpoint(conn, project_id: int, run_id: int, config: dict):
    target = _identity(config)
    rows = conn.execute(
        "SELECT c.id, c.file_path, c.run_id, tr.config_json FROM checkpoints c "
        "JOIN training_runs tr ON tr.id=c.run_id "
        "WHERE c.project_id=? AND tr.project_id=? "
        "ORDER BY CASE WHEN c.run_id=? THEN 0 ELSE 1 END, c.run_id DESC, "
        "COALESCE(c.step,0) DESC, c.epoch DESC, c.id DESC",
        (project_id, project_id, run_id),
    ).fetchall()
    for row in rows:
        if not Path(str(row['file_path'])).is_file():
            continue
        if int(row['run_id']) == run_id:
            return row
        try:
            candidate = json.loads(row['config_json'] or '{}')
            identity = _identity(candidate) if isinstance(candidate, dict) else None
        except (ValueError, TypeError, KeyError):
            continue
        if target is not None and identity == target:
            return row
    return None
