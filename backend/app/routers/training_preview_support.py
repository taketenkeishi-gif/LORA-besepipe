from __future__ import annotations

import json

from ..training.runtime.preview_jobs import outfit_negative_tags, outfit_signature_tags


def outfit_defaults(conn, project_id: int) -> list[dict]:
    rows = conn.execute("SELECT name, trigger_token FROM basepipe_concepts WHERE project_id=? AND concept_type='outfit' AND TRIM(trigger_token)<>'' ORDER BY id", (project_id,)).fetchall()
    triggers = [str(r["trigger_token"]).strip() for r in rows]
    snapshot_id = None
    try:
        draft = conn.execute("SELECT training_config_json FROM projects WHERE id=?", (project_id,)).fetchone()
        snapshot_id = (json.loads(draft["training_config_json"] or "{}").get("dataset_snapshot_id") if draft else None)
    except Exception:
        snapshot_id = None
    if snapshot_id is None:
        latest = conn.execute("SELECT id FROM basepipe_dataset_snapshots WHERE project_id=? ORDER BY id DESC LIMIT 1", (project_id,)).fetchone()
        snapshot_id = latest["id"] if latest else None
    captions = [str(r["caption_at_snapshot"] or "") for r in conn.execute("SELECT caption_at_snapshot FROM basepipe_snapshot_entries WHERE snapshot_id=?", (snapshot_id,))] if snapshot_id else []
    signature = outfit_signature_tags(captions, triggers) if triggers else {}
    negatives = outfit_negative_tags(captions, triggers) if triggers else {}
    out = []
    for r in rows:
        trig = str(r["trigger_token"]).strip()
        out.append({"name": str(r["name"]), "trigger": trig, "tags": signature.get(trig, []), "negative_tags": negatives.get(trig, []),
                    "images": sum(1 for c in captions if trig.casefold() in {t.strip().casefold() for t in c.split(",")})})
    return out
