"""Trainer-owned epoch samples: no second GPU model/process during training."""
from pathlib import Path
import json
from ....db import get_conn


def configure_native_preview(run_dir, config_path, cfg, project_id):
    base = Path(cfg.get("base_checkpoint_path", ""))
    preview = Path(cfg.get("preview_base_checkpoint_path") or base)
    if not base.is_file() or not preview.is_file() or not base.samefile(preview):
        return False
    conn = get_conn()
    try:
        row = conn.execute("SELECT payload_json FROM basepipe_preview_profile_snapshots WHERE id=? AND project_id=?",
                           (cfg.get("preview_profile_snapshot_id"), project_id)).fetchone()
    finally:
        conn.close()
    if row is None:
        return False
    profile = json.loads(row["payload_json"])
    # This trainer supplies Euler flow sampling only. Other selections keep the
    # original Comfy workflow, rather than silently ignoring the selected sampler.
    if profile.get("sampler", "euler") != "euler" or profile.get("scheduler", "simple") != "simple":
        return False
    resolution = int(profile.get("resolution") or 512)
    prompt = {"prompt": profile.get("prompt", ""), "negative_prompt": profile.get("negative_prompt", ""),
              "seed": int(profile.get("seed", 42)), "width": resolution, "height": resolution,
              "sample_steps": int(cfg.get("preview_steps") if cfg.get("preview_steps") is not None else profile.get("steps", 20)),
              "scale": float(cfg.get("preview_cfg") if cfg.get("preview_cfg") is not None else profile.get("cfg", 5)),
              "flow_shift": 3.0}
    prompt_path = Path(run_dir) / "native_sample_prompts.json"
    prompt_path.write_text(json.dumps([prompt], ensure_ascii=False), encoding="utf8")
    with Path(config_path).open("a", encoding="utf8") as f:
        f.write("sample_prompts = " + json.dumps(str(prompt_path)) + "\n")
        f.write(f"sample_every_n_epochs = {max(1, int(cfg.get('save_every_n_epochs', 1)))}\n")
    return True
