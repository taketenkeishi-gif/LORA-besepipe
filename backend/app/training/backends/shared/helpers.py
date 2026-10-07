"""SDXL/Musubi 両バックエンドが共有する学習ディレクトリ準備処理。"""
from __future__ import annotations

import json
import hashlib
import os
import re
import shutil
from pathlib import Path

from ....dataset.pipeline import read_manifest
from ....db import get_conn
from ...runtime.state import IMAGE_EXTS, run_dir as canonical_run_dir


class DatasetSourceError(RuntimeError):
    """dataset_source の解決に失敗したことを示す（暗黙フォールバックせず即エラーにする）。"""


def resolve_dataset_source(project_dataset_dir: str, dataset_source: str) -> tuple[str, dict]:
    """Training が読む Dataset のディレクトリを明示的に解決する。

    dataset_source:
      - "original"（既定・後方互換）: project.dataset_dir 直下を使う。
      - "processed": <project.dataset_dir>/processed/ を使う。ただし
        Dataset Builder Pipeline が成功完了した証跡（.manifest.json、
        success 件数 > 0）が無ければエラーにする — フォルダの存在だけで
        「処理済み」と判定しない。

    戻り値: (実際に使用するディレクトリ, snapshot_info)
    snapshot_info は config_json へ保存し、この run がどの Dataset
    Artifact/Snapshot を使用したかを後から追跡できるようにする。
    """
    if dataset_source not in ("original", "processed"):
        raise DatasetSourceError(f"未知の dataset_source です: {dataset_source!r}")

    if dataset_source == "original":
        return project_dataset_dir, {"dataset_source": "original", "resolved_dataset_dir": project_dataset_dir}

    processed_dir = Path(project_dataset_dir) / "processed"
    if not processed_dir.exists():
        raise DatasetSourceError(
            f"dataset_source=processed が指定されましたが、{processed_dir} が存在しません。"
            f"先に Dataset Builder で前処理パイプラインを実行してください。"
        )
    manifest = read_manifest(processed_dir)
    if manifest is None:
        raise DatasetSourceError(
            f"dataset_source=processed が指定されましたが、{processed_dir} に有効な"
            f"完了済みマニフェスト(.manifest.json)がありません（成功件数0、または未実行）。"
            f"processed/ フォルダの存在だけでは Training 入力として使用できません。"
        )
    return str(processed_dir), {
        "dataset_source": "processed",
        "resolved_dataset_dir": str(processed_dir),
        "dataset_snapshot_generated_at": manifest.get("generated_at"),
        "dataset_snapshot_success_count": manifest.get("success"),
    }


def persist_dataset_snapshot(run_id: int, dataset_snapshot: dict) -> None:
    """この run が実際に使用した Dataset Snapshot 情報を training_runs.config_json へ
    マージ保存する。「どの Dataset Artifact/Snapshot を入力として使ったか」を
    run 作成後からでも追跡できるようにする（新規テーブルは追加しない）。
    """
    conn = get_conn()
    try:
        row = conn.execute("SELECT config_json FROM training_runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            return
        try:
            cfg = json.loads(row["config_json"] or "{}")
        except (ValueError, TypeError):
            cfg = {}
        cfg.update(dataset_snapshot)
        conn.execute(
            "UPDATE training_runs SET config_json = ? WHERE id = ?",
            (json.dumps(cfg, ensure_ascii=False), run_id),
        )
        conn.commit()
    finally:
        conn.close()


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_run_dir(
    run_id: int,
    dataset_dir: str,
    repeats: int,
    project_name: str,
    asset_paths: list[tuple[str, str, str]] | None = None,
    runtime_root: Path | None = None,
) -> Path:
    """kohya_ss 形式のディレクトリ構造を .runtime/runs/{run_id}/ 以下に作成する。

    musubi-tuner もこの構造(train_data/{repeats}_{name}/)をそのまま使う
    （ImageDirectoryDatasource がサブディレクトリを直接指定できるため）。
    """
    # A snapshot is authoritative for every engine, not only Anima callers
    # which already provide asset_paths explicitly. Never reinterpret it as
    # a hint to scan the original directory (which can contain holdout images).
    conn = get_conn()
    try:
        run = conn.execute('SELECT project_id,config_json FROM training_runs WHERE id=?',(run_id,)).fetchone()
        run_cfg = json.loads(run['config_json'] or '{}') if run else {}
        if asset_paths is None and run_cfg.get('dataset_snapshot_id') is not None:
            from ...snapshot_inputs import fetch_snapshot_inputs, verify_snapshot_input
            sid = int(run_cfg['dataset_snapshot_id'])
            snapshot = conn.execute('SELECT item_count,status FROM basepipe_dataset_snapshots WHERE id=? AND project_id=?',(sid,run['project_id'])).fetchone()
            if snapshot is None or snapshot['status']!='sealed':
                raise DatasetSourceError('The selected training snapshot is not sealed or belongs to another project')
            entries = fetch_snapshot_inputs(conn,sid)
            if not entries or len(entries)!=int(snapshot['item_count']):
                raise DatasetSourceError('Training snapshot image count mismatch')
            asset_paths = []
            for entry in entries:
                valid,detail=verify_snapshot_input(entry)
                if not valid:raise DatasetSourceError(detail)
                asset_paths.append((str(entry['file_path']),str(entry['caption_at_snapshot'] or ''),str(entry.get('snapshot_sha256') or entry['content_sha256'])))
        reference = run_cfg.get('evaluation_reference')
        if reference:
            if asset_paths is None:
                raise DatasetSourceError('Evaluation holdout requires an explicit sealed training snapshot')
            if any(sha==reference['sha256'] or _sha256(Path(path))==reference['sha256'] for path,caption,sha in asset_paths):
                raise DatasetSourceError('Evaluation reference cannot be staged as training input')
    finally:
        conn.close()
    runs_root = runtime_root or canonical_run_dir(run_id).parent
    run_dir = runs_root / str(run_id)
    safe_name = re.sub(r"[^\w\-]", "_", project_name)
    train_sub = run_dir / "train_data" / f"{repeats}_{safe_name}"
    # Every prepare is a deterministic rebuild. Leaving files from a failed
    # attempt can silently train on assets outside the sealed Snapshot.
    if train_sub.exists():
        shutil.rmtree(train_sub)
    train_sub.mkdir(parents=True, exist_ok=True)
    (run_dir / "output").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)

    if asset_paths is not None:
        # A sealed Basepipe Snapshot is an explicit file list, not a folder
        # hint. Stage only those files and preserve the caption at seal time.
        for ordinal, (raw_path, caption, expected_sha256) in enumerate(asset_paths):
            f = Path(raw_path)
            if not f.is_file() or f.suffix.lower() not in IMAGE_EXTS:
                raise DatasetSourceError(f"sealed Snapshot image is missing: {f}")
            # The ordinal prevents same-name files from different source
            # directories from collapsing into one staging entry.
            staged_stem = f"{ordinal:05d}_{expected_sha256[:12]}"
            staged_image = train_sub / f"{staged_stem}{f.suffix.lower()}"
            # Snapshot staging must be independent from its mutable source.
            # Hardlinks would let a later source edit mutate trainer input.
            shutil.copy2(f.resolve(), staged_image)
            staged_sha256 = _sha256(staged_image)
            if staged_sha256 != expected_sha256:
                staged_image.unlink(missing_ok=True)
                raise DatasetSourceError(
                    f"sealed Snapshot copy hash mismatch: expected={expected_sha256} actual={staged_sha256}"
                )
            # Never prefer a mutable neighbouring .txt file. The caption frozen
            # in basepipe_snapshot_entries is the only trainer input.
            (train_sub / f"{staged_stem}.txt").write_text(caption, encoding="utf-8")
    else:
        src = Path(dataset_dir)
        if src.exists():
            for f in src.iterdir():
                if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                    link_or_copy(f.resolve(), train_sub / f.name)
                    txt = f.with_suffix(".txt")
                    if txt.exists():
                        link_or_copy(txt.resolve(), train_sub / txt.name)

    return run_dir
