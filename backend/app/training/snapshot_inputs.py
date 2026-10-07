"""Resolve the immutable file/caption pair sealed into a Dataset Snapshot.

Basepipe Assets retain the original source image. A Snapshot entry may point
at a reviewed derivative (for example a Qwen correction) through its frozen
``metadata_json.snapshot`` block. Training must consume that frozen path, not
the mutable Asset row or whichever derivative happens to be latest later.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any


def _metadata(value: Any) -> dict:
    try:
        parsed = json.loads(value or "{}")
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def fetch_snapshot_inputs(conn, snapshot_id: int) -> list[dict]:
    rows = conn.execute(
        "SELECT e.asset_id, a.id AS source_registration_id, a.file_path AS source_file_path, "
        "a.content_sha256 AS source_content_sha256, a.asset_type, "
        "a.origin_kind, a.review_status, a.training_enabled, e.caption_at_snapshot, "
        "e.metadata_json AS entry_metadata_json "
        "FROM basepipe_snapshot_entries e "
        "LEFT JOIN basepipe_assets a ON a.id = e.asset_id "
        "WHERE e.snapshot_id = ? ORDER BY e.ordinal",
        (snapshot_id,),
    ).fetchall()
    resolved: list[dict] = []
    for raw in rows:
        row = dict(raw)
        row['source_registration_missing'] = row.pop('source_registration_id') is None
        snapshot = _metadata(row.pop("entry_metadata_json", "")).get("snapshot", {})
        snapshot = snapshot if isinstance(snapshot, dict) else {}
        row["file_path"] = str(snapshot.get("path") or row["source_file_path"])
        row["content_sha256"] = str(snapshot.get("sha256") or row["source_content_sha256"])
        # Snapshot作成時点でAssetはapprovedかつtraining_enabledであることを
        # 強制している。以後のAssetレビューは次のSnapshot用の編集であり、
        # 封印済みSnapshotの適格性を変えてはならない。旧Snapshotには
        # review_statusがないため、entryの存在自体をapprovedの証拠として扱う。
        row["review_status"] = str(snapshot.get("review_status") or "approved")
        row["training_enabled"] = 1 if bool(snapshot.get("training_enabled", True)) else 0
        row["asset_type"] = str(snapshot.get("asset_type") or row["asset_type"])
        row["origin_kind"] = str(snapshot.get("origin_kind") or row["origin_kind"] or "")
        row["input_source"] = str(snapshot.get("input_source") or "source_asset")
        row["version_id"] = snapshot.get("version_id")
        row["requires_user_aesthetic_approval"] = bool(
            snapshot.get("requires_user_aesthetic_approval", row["origin_kind"] == "h3_frame")
        )
        row["identity_review_state"] = snapshot.get("identity_review_state")
        row["identity_reviewed_by"] = snapshot.get("identity_reviewed_by")
        row["training_input_review_state"] = snapshot.get("training_input_review_state")
        row["training_input_reviewed_by"] = snapshot.get("training_input_reviewed_by")
        resolved.append(row)
    return resolved


def snapshot_user_approval_issue(entry: dict) -> str:
    """Return a blocking reason when an H3-derived entry lacks frozen user approval.

    Missing evidence is intentionally not promoted.  Older snapshots may have
    ``review_status=approved`` but no proof that the user made the aesthetic
    Frame and Training Input decisions.
    """
    if not bool(entry.get("requires_user_aesthetic_approval")):
        return ""
    if entry.get("identity_review_state") != "USER_APPROVED" or entry.get("identity_reviewed_by") != "user":
        return "H3 Frame選別のユーザー審美承認証拠が欠落"
    if entry.get("training_input_review_state") != "USER_APPROVED" or entry.get("training_input_reviewed_by") != "user":
        return "Training Input選択のユーザー審美承認証拠が欠落"
    return ""


def verify_snapshot_input(entry: dict) -> tuple[bool, str]:
    if entry.get('source_registration_missing'):
        return False, "学習対象の画像登録が欠落"
    path = Path(str(entry.get("file_path") or ""))
    if not path.is_file():
        return False, "ファイル欠落"
    expected = str(entry.get("content_sha256") or "").strip().lower()
    if not expected:
        return False, "SHA-256欠落"
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    if digest != expected:
        return False, "SHA-256不一致"
    return True, ""
