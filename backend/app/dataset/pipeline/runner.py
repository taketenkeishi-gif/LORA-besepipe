"""前処理パイプラインのオーケストレーション（Resize -> Upscale -> Cleanup -> Caption -> Save）。

画像ごとに PipelineStep を順に適用する。画像単位の失敗は握りつぶして
継続する（1枚の失敗が残り全体を止めない）。進捗状態はプロジェクト単位で
_PIPELINE_STATUS に保持し、GET /dataset/preprocess-pipeline-status/{id} が
参照する。複数プロジェクトからの同時要求は queue.py が直列化する
（本モジュールの run_pipeline() 自体は1プロジェクト分の処理のみを担う）。

パイプラインが成功完了すると processed/ 直下に .manifest.json を書き出す。
これは Training 側が「processed/ が本当に完了済みの Dataset Snapshot か」を
判定するための唯一の根拠であり、processed/ フォルダの単純な存在チェックでは
代替できない（画像が1枚も無いのに空フォルダだけ存在する等の誤判定を防ぐ）。
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from ...db import get_conn
from .base import PipelineContext, PipelineStep
from .steps import CaptionStep, CleanupStep, ResizeStep, SaveStep, UpscaleStep

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = ".manifest.json"


def _write_manifest(
    processed_dir: Path,
    project_id: int,
    items: list[dict],
    success: int,
    failed: int,
    use_esrgan: bool,
    use_qwen: bool,
    use_caption: bool,
) -> None:
    """processed/ が Training から利用可能な Dataset Snapshot であることの証跡を書く。

    file exists だけで完成判定しないための最小限のメタデータ（新規テーブルは作らない）。
    """
    manifest = {
        "project_id": project_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total": len(items),
        "success": success,
        "failed": failed,
        "steps": {"resize": True, "upscale": use_esrgan, "cleanup": use_qwen, "caption": use_caption},
        "success_item_ids": [it["item_id"] for it in items if it.get("status") == "success"],
    }
    try:
        (processed_dir / MANIFEST_FILENAME).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError as exc:  # noqa: BLE001 — マニフェスト書き込み失敗は前処理自体の成否に影響させない
        logger.warning("manifest write failed for %s: %s", processed_dir, exc)


def read_manifest(processed_dir: Path) -> dict | None:
    """processed/ の Dataset Snapshot マニフェストを読む。存在しない/壊れていれば None。"""
    manifest_path = processed_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        return None
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or int(data.get("success", 0)) <= 0:
            return None
        return data
    except (OSError, ValueError, TypeError):
        return None

_PIPELINE_STATUS: dict[int, dict] = {}
_PIPELINE_CANCEL: set[int] = set()

# 既定のステップ順序。新しいステップを追加する場合はここに挿入するだけでよい。
DEFAULT_STEPS: list[PipelineStep] = [
    ResizeStep(), UpscaleStep(), CleanupStep(), CaptionStep(), SaveStep(),
]


def _empty_status(status: str, message: str) -> dict:
    return {
        "status": status, "message": message,
        "done": 0, "total": 0, "failed": 0, "skipped": 0, "success": 0,
        "current_step": "", "current_step_name": "", "current_image": "",
        "items": [], "processed_dir": "",
    }


def get_pipeline_status(project_id: int) -> dict:
    return _PIPELINE_STATUS.get(project_id, _empty_status("idle", "待機中"))


def is_running(project_id: int) -> bool:
    return _PIPELINE_STATUS.get(project_id, {}).get("status") == "running"


def mark_queued(project_id: int, position: int) -> None:
    """queue.py からの呼び出し専用 — キュー投入直後の状態を反映する。"""
    _PIPELINE_STATUS[project_id] = _empty_status(
        "queued", f"キュー待機中（{position} 番目）",
    )


def request_cancel(project_id: int) -> bool:
    """実行中パイプラインへキャンセルを要求する。実行中でなければ False。"""
    if not is_running(project_id):
        return False
    _PIPELINE_CANCEL.add(project_id)
    return True


def run_pipeline(
    project_id: int,
    target_size: int,
    resize_mode: str,
    use_esrgan: bool,
    esrgan_model: str,
    use_qwen: bool,
    item_ids: list[int] | None,
    use_caption: bool = False,
    caption_general_thresh: float = 0.35,
    caption_character_thresh: float = 0.85,
    caption_remove_character_tags: bool = False,
    steps: list[PipelineStep] | None = None,
) -> None:
    """統合前処理パイプライン: Resize → Upscale(任意) → Cleanup(任意) → Caption(任意) → Save。

    元画像は変更しない（Caption有効時のみ、既存タグ編集UIとの互換のため
    dataset_items.caption と元画像の .txt サイドカーを更新する — 画像本体は
    引き続き変更しない）。処理結果は <dataset_dir>/processed/ に PNG (+ .txt)
    で保存する。画像ごとに失敗しても残りの処理は継続する。
    """
    from ...services.comfyui_client import ComfyUIClient, ComfyUIError  # noqa: PLC0415

    pipeline_steps = steps if steps is not None else DEFAULT_STEPS

    _PIPELINE_STATUS[project_id] = _empty_status("running", "初期化中...")

    try:
        conn = get_conn()
        prow = conn.execute("SELECT dataset_dir FROM projects WHERE id = ?", (project_id,)).fetchone()
        if prow is None:
            conn.close()
            _PIPELINE_STATUS[project_id] = _empty_status("failed", "プロジェクトが見つかりません")
            return

        dataset_dir = prow["dataset_dir"] or ""

        if item_ids:
            placeholders = ",".join("?" * len(item_ids))
            rows = conn.execute(
                f"SELECT id, file_path FROM dataset_items WHERE project_id = ? AND id IN ({placeholders})",
                (project_id, *item_ids),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, file_path FROM dataset_items WHERE project_id = ?",
                (project_id,),
            ).fetchall()
        conn.close()

        targets = [(r["id"], Path(r["file_path"])) for r in rows if Path(r["file_path"]).exists()]
        total = len(targets)

        if total == 0:
            _PIPELINE_STATUS[project_id] = _empty_status(
                "failed", "対象画像がありません（画像を収集してから実行してください）",
            )
            return

        # 出力先: dataset_dir/processed/
        if dataset_dir:
            processed_dir = Path(dataset_dir) / "processed"
        else:
            # dataset_dir 未設定時は最初の画像と同じ親に processed/ を作る
            processed_dir = targets[0][1].parent / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)

        # ComfyUI クライアント（Upscale / Cleanup 使用時）
        comfy_client: ComfyUIClient | None = None
        actual_esrgan_model = esrgan_model

        if use_esrgan or use_qwen:
            try:
                comfy_client = ComfyUIClient()
                st = comfy_client.check_status()
                if not st.get("reachable"):
                    if use_qwen:
                        _PIPELINE_STATUS[project_id] = _empty_status(
                            "failed", f"ComfyUI に接続できません: {st.get('message', '')}",
                        )
                        return
                    logger.warning("Upscale requested but ComfyUI unreachable: %s", st.get("message"))
                    comfy_client = None
                else:
                    if use_esrgan and not actual_esrgan_model:
                        actual_esrgan_model = st.get("upscale_model") or ""
            except Exception as exc:  # noqa: BLE001
                logger.warning("ComfyUI init failed: %s", exc)
                if use_qwen:
                    _PIPELINE_STATUS[project_id] = _empty_status("failed", f"ComfyUI 初期化失敗: {exc}")
                    return
                comfy_client = None

        _PIPELINE_STATUS[project_id].update({
            "total": total, "message": f"0/{total} 件処理中...", "processed_dir": str(processed_dir),
        })

        options = {
            "target_size": target_size,
            "resize_mode": resize_mode,
            "use_esrgan": use_esrgan,
            "esrgan_model": actual_esrgan_model,
            "use_qwen": use_qwen,
            "use_caption": use_caption,
            "caption_general_thresh": caption_general_thresh,
            "caption_character_thresh": caption_character_thresh,
            "caption_remove_character_tags": caption_remove_character_tags,
            "processed_dir": processed_dir,
        }

        done = 0
        failed = 0
        items: list[dict] = []

        def _remaining_skipped(from_index: int) -> list[dict]:
            return [
                {"item_id": iid, "file_name": p.name, "status": "skipped", "error": None}
                for iid, p in targets[from_index:]
            ]

        for idx, (item_id, src_path) in enumerate(targets):
            if project_id in _PIPELINE_CANCEL:
                _PIPELINE_CANCEL.discard(project_id)
                items.extend(_remaining_skipped(idx))
                _PIPELINE_STATUS[project_id] = {
                    **_empty_status(
                        "cancelled",
                        f"キャンセルしました: 成功 {done - failed} 件 / 失敗 {failed} 件 / 残り {total - done} 件",
                    ),
                    "done": done, "total": total, "failed": failed, "skipped": total - done,
                    "success": done - failed, "items": items,
                }
                return
            error_message: str | None = None
            out_path: str = ""
            try:
                with Image.open(src_path) as im:
                    ctx = PipelineContext(
                        src_path=src_path,
                        image=im.copy(),
                        width=im.width,
                        height=im.height,
                        item_id=item_id,
                        options=options,
                        extra={"comfy_client": comfy_client},
                    )
                for step in pipeline_steps:
                    if not step.should_run(ctx):
                        continue
                    _PIPELINE_STATUS[project_id]["current_step"] = f"{step.name}: {src_path.name}"
                    _PIPELINE_STATUS[project_id]["current_step_name"] = step.name
                    _PIPELINE_STATUS[project_id]["current_image"] = src_path.name
                    ctx = step.process(ctx)
                if ctx.extra.get("out_path"):
                    out_path = str(ctx.extra["out_path"])
            except Exception as exc:  # noqa: BLE001
                logger.warning("pipeline failed for %s: %s", src_path, exc, exc_info=True)
                error_message = str(exc)
                failed += 1

            items.append({
                "item_id": item_id,
                "file_name": src_path.name,
                "status": "failed" if error_message is not None else "success",
                "error": error_message,
                "output_path": out_path,
            })

            done += 1
            _PIPELINE_STATUS[project_id].update({
                "done": done,
                "failed": failed,
                "success": done - failed,
                "items": items,
                "message": f"{done}/{total} 件処理中..." + (f"（失敗 {failed} 件）" if failed else ""),
            })

        _PIPELINE_CANCEL.discard(project_id)
        ok = done - failed
        step_names = []
        step_names.append("リサイズ")
        if use_esrgan:
            step_names.append("ESRGAN")
        if use_qwen:
            step_names.append("Qwen Cleanup")
        if use_caption:
            step_names.append("Caption")

        _write_manifest(processed_dir, project_id, items, ok, failed, use_esrgan, use_qwen, use_caption)

        _PIPELINE_STATUS[project_id] = {
            **_empty_status(
                "done",
                f"完了（{'→'.join(step_names)}）: 成功 {ok} 件 / 失敗 {failed} 件 → {processed_dir}",
            ),
            "done": done,
            "total": total,
            "failed": failed,
            "skipped": 0,
            "success": ok,
            "items": items,
            "processed_dir": str(processed_dir),
        }

    except Exception as e:  # noqa: BLE001
        logger.error("run_pipeline failed: %s", e, exc_info=True)
        _PIPELINE_CANCEL.discard(project_id)
        _PIPELINE_STATUS[project_id] = _empty_status("failed", f"パイプラインエラー: {e}")
