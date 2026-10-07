"""Persistent, read-only evidence helpers for training runs."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .state import run_dir


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_run_manifest(
    run_id: int,
    project_id: int,
    requested: dict,
    resolved: dict,
    dataset: dict,
    environment: dict,
    concepts: list[dict] | None = None,
) -> Path:
    """Create the immutable run manifest once, before the runner is dispatched."""
    target_dir = run_dir(run_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = target_dir / "run_manifest.json"
    if manifest_path.exists():
        return manifest_path
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "project_id": project_id,
        "created_at": _utc_now(),
        # The manifest records the backend that will actually execute the Run.
        # For Anima, the UI may request "auto", but the resolved backend is
        # still the canonical engine identity required by Basepipe lineage.
        "engine": resolved.get("resolved_training_backend") or resolved.get("training_engine"),
        "model_family": resolved.get("model_family"),
        "concepts": concepts or [],
        "dataset": dataset,
        "requested_config_path": "requested_config.json",
        "resolved_config_path": "resolved_config.json",
        "generated_configs": [],
        "environment": environment,
    }
    for name, value in (
        ("requested_config.json", requested),
        ("resolved_config.json", resolved),
    ):
        (target_dir / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest_path


def load_run_manifest(run_id: int) -> dict | None:
    path = run_dir(run_id) / "run_manifest.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError, TypeError):
        return None


def record_prepared_evidence(run_id: int, prepared) -> None:
    """Persist the concrete config path and backend-resolved values after prepare()."""
    target_dir = run_dir(run_id)
    resolved_path = target_dir / "resolved_config.json"
    try:
        resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
        if not isinstance(resolved, dict):
            resolved = {}
    except (OSError, ValueError, TypeError):
        resolved = {}
    resolved["prepared_config_path"] = str(prepared.config_path)
    resolved["prepared_log_path"] = str(prepared.log_path)
    extra = getattr(prepared, "extra", {}) or {}
    generated_configs = [str(prepared.config_path)]
    if extra.get("dataset_json_path"):
        generated_configs.append(str(extra["dataset_json_path"]))
    resolved["generated_configs"] = generated_configs
    if isinstance(extra.get("musubi_cfg"), dict):
        resolved["backend_resolved"] = extra["musubi_cfg"]
    if isinstance(extra.get("anima_cfg"), dict):
        anima_cfg = extra["anima_cfg"]
        resolved["backend_resolved"] = anima_cfg
        resolved["resolved_text_encoder_path"] = anima_cfg.get("text_encoder_path")
        resolved["resolved_vae_path"] = anima_cfg.get("vae_path")
    if extra.get("trainer_script_path"):
        resolved["trainer_script_path"] = extra["trainer_script_path"]
    if extra.get("python_executable"):
        resolved["python_executable"] = extra["python_executable"]
    if extra.get("gpu_device_id") is not None:
        resolved["resolved_gpu_device_id"] = extra["gpu_device_id"]
    if isinstance(extra.get("gpu_mapping"), dict):
        resolved["gpu_mapping"] = extra["gpu_mapping"]
    env = getattr(prepared, "env", {}) or {}
    resolved["cuda_visible_devices"] = env.get("CUDA_VISIBLE_DEVICES")
    resolved_path.write_text(json.dumps(resolved, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_path = target_dir / "run_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(manifest, dict):
            manifest["generated_configs"] = generated_configs
            manifest["resolved_config_path"] = str(resolved_path)
            env_snapshot = manifest.setdefault("environment", {})
            if isinstance(env_snapshot, dict):
                env_snapshot["cuda_visible_devices"] = env.get("CUDA_VISIBLE_DEVICES")
                if isinstance(extra.get("gpu_mapping"), dict):
                    env_snapshot["gpu_mapping"] = extra["gpu_mapping"]
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    except (OSError, ValueError, TypeError):
        pass

    from ...db import get_conn
    conn = get_conn()
    conn.execute(
        "UPDATE training_runs SET resolved_config_json=?, manifest_path=COALESCE(manifest_path, ?) WHERE id=?",
        (json.dumps(resolved, ensure_ascii=False), str(manifest_path), run_id),
    )
    conn.commit()
    conn.close()


def parameter_evidence(requested: dict, resolved: dict, observed: dict | None = None) -> list[dict]:
    observed = observed or {}
    keys = (
        ("batch_size", "train_batch_size"),
        ("learning_rate", "learning_rate"),
        ("rank", "rank"),
        ("alpha", "alpha"),
        ("resolution", "resolution"),
        ("epochs", "epochs"),
        ("steps", "total_steps"),
        ("gpu_device_id", "gpu_device_id"),
    )
    result = []
    for key, source_key in keys:
        req = requested.get(source_key)
        res = resolved.get(source_key)
        obs = observed.get(key)
        result.append({
            "key": key,
            "requested": req,
            "resolved": res,
            "observed": obs,
            "observed_source": "trainer" if obs is not None else "unknown",
            "mismatch": obs is not None and res != obs,
        })
    return result
