"""学習成果物(checkpoint/LoRA)の互換性検証。

Training成功とPreview成功を無関係な状態にしないための橋渡し。checkpoint が
DB登録されただけでは「Preview側で実際に使える」ことを何も保証しない
（安全なfp32/フォーマット破損チェックすら行われていなかった）。

状態は既存の checkpoints テーブルへ列追加した validation_status で表現する
（新規テーブルは追加しない）。論理的な完了ステートマシン:

    TRAINING_SUCCEEDED      … training_runs.status == 'completed'（既存）
    ARTIFACT_REGISTERED     … checkpoints 行が存在する（既存）
    ARTIFACT_VALIDATED      … validation_status == 'validated'（本モジュール）
    PREVIEW_SUCCEEDED       … validation_status == 'preview_succeeded'
    PREVIEW_FAILED          … validation_status == 'preview_failed'
    (invalid)               … validation_status == 'invalid'（壊れている・不整合）
"""
from __future__ import annotations

import logging
from pathlib import Path

from ...db import get_conn

logger = logging.getLogger(__name__)

# model_family ごとの ss_base_model_version / ss_network_module 期待値。
# 完全一致までは要求しない（学習ツールのバージョン差異があるため）が、
# 明らかな不一致（例: sdxl の LoRA を krea2 として使おうとしている）は検出する。
_EXPECTED_BASE_MODEL_VERSION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "krea2": ("krea2", "krea_2", "krea-2"),
    "flux": ("flux",),
    "sdxl": ("sdxl", "sd_xl", "xl"),
}


def validate_artifact(file_path: str, model_family: str = "") -> tuple[bool, str]:
    """checkpoint (.safetensors) が Preview 側で読み込み可能な状態かを検証する。

    確認する内容:
      1. file exists
      2. non-zero size
      3. safetensors として読取可能（ヘッダ破損検出）
      4. metadata 取得可能
      5. 対象 model_family との整合性（metadata の ss_base_model_version から判定可能な範囲）
      6. 先頭テンソルに NaN/Inf が無いか（軽量サンプリング、全走査はしない）

    戻り値: (ok, detail) — detail は成功/失敗理由の短い説明。
    """
    p = Path(file_path)
    if not p.exists():
        return False, f"file not found: {file_path}"

    size = p.stat().st_size
    if size <= 0:
        return False, f"file is empty (0 bytes): {file_path}"

    try:
        from safetensors import safe_open
    except ImportError:
        return False, "safetensors package not installed — cannot validate"

    try:
        with safe_open(str(p), framework="pt") as f:
            keys = list(f.keys())
            if not keys:
                return False, "safetensors file has zero tensors (empty checkpoint)"

            metadata = f.metadata() or {}
            base_model_version = str(metadata.get("ss_base_model_version", "")).lower()

            if model_family and model_family in _EXPECTED_BASE_MODEL_VERSION_KEYWORDS:
                expected_kw = _EXPECTED_BASE_MODEL_VERSION_KEYWORDS[model_family]
                if base_model_version and not any(kw in base_model_version for kw in expected_kw):
                    return False, (
                        f"model_family mismatch: expected one of {expected_kw}, "
                        f"but checkpoint metadata says ss_base_model_version={base_model_version!r}"
                    )

            # 軽量サンプリング: 先頭最大20テンソルのみ NaN/Inf チェック（全走査は大きいLoRAで重い）
            import torch
            bad_tensors: list[str] = []
            for k in keys[:20]:
                t = f.get_tensor(k)
                if torch.isnan(t).any() or torch.isinf(t).any():
                    bad_tensors.append(k)
            if bad_tensors:
                return False, f"NaN/Inf detected in tensors: {bad_tensors[:5]}"

    except Exception as exc:  # noqa: BLE001 — safetensors破損は多様な例外形で来る
        return False, f"safetensors read failed: {exc}"

    return True, f"OK: {len(keys)} tensors, {size} bytes, base_model_version={base_model_version or 'unknown'}"


def validate_and_record(checkpoint_id: int, file_path: str, model_family: str = "") -> bool:
    """validate_artifact を実行し、結果を checkpoints.validation_status へ記録する。"""
    ok, detail = validate_artifact(file_path, model_family)
    conn = get_conn()
    try:
        conn.execute(
            "UPDATE checkpoints SET validation_status = ?, validation_detail = ? WHERE id = ?",
            ("validated" if ok else "invalid", detail, checkpoint_id),
        )
        conn.commit()
    finally:
        conn.close()
    if not ok:
        logger.warning("artifact validation failed for checkpoint %s (%s): %s", checkpoint_id, file_path, detail)
    return ok


def record_preview_outcome(
    checkpoint_id: int, succeeded: bool, detail: str = "",
    *, succeeded_count: int | None = None, expected_count: int | None = None,
) -> None:
    """Preview生成試行の結果を checkpoints.validation_status へ記録する。

    validate_and_record で既に 'invalid' と判定されている場合は上書きしない
    （壊れているという事実の方が Preview結果より重要な情報のため）。

    succeeded_count/expected_count を渡した場合、複数Prompt/複数Instanceの
    一部だけ成功したケースを "preview_partial" として区別する。以前は
    1件でも成功すれば checkpoint 全体を "preview_succeeded" として記録して
    おり、実運用で「3件要求したのに1件しか生成されないのに完了扱いになる」
    実バグがあった(実データ監査で発見: run 39 epoch1、preview_samples 1件のみ
    なのに validation_status='preview_succeeded')。succeeded_count/
    expected_count を渡さない既存呼び出しは、従来通りの二値判定を維持する
    (後方互換)。
    """
    if expected_count is not None and expected_count > 0:
        n_ok = succeeded_count if succeeded_count is not None else (expected_count if succeeded else 0)
        if n_ok >= expected_count:
            status = "preview_succeeded"
        elif n_ok > 0:
            status = "preview_partial"
        else:
            status = "preview_failed"
        if not detail:
            detail = f"{n_ok}/{expected_count} succeeded"
    else:
        status = "preview_succeeded" if succeeded else "preview_failed"

    conn = get_conn()
    try:
        row = conn.execute("SELECT validation_status FROM checkpoints WHERE id = ?", (checkpoint_id,)).fetchone()
        if row is not None and row["validation_status"] == "invalid":
            return
        conn.execute(
            "UPDATE checkpoints SET validation_status = ?, validation_detail = ? WHERE id = ?",
            (status, detail, checkpoint_id),
        )
        conn.commit()
    finally:
        conn.close()
