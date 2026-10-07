"""学習プロセスの監視・成果物スキャン・状態確定。

Backend.train() はサブプロセスを起動した後、tail_monitor() を呼んでブロッキング
監視する。ログ tail・DB 状態反映・チェックポイント検出・プレビュー生成の
トリガーはすべてここに集約される。プレビュー生成（PreviewProvider 経由）は
学習基盤とは独立した責務であり、失敗しても本モジュールの状態確定には影響しない。
"""
from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

from ...db import get_conn
from ...trainers import get_spec
from .artifact_validation import validate_and_record, validate_artifact
from .estimation import record_step_timing
from .log_parser import (
    RE_CKPT_EPOCH,
    RE_SAMPLE_EPOCH,
    RE_SAMPLE_EPOCH_IDX,
    RE_SAVE,
    parse_log_state,
    parse_train_line,
)
from .lora_export import auto_register_lora_asset
from .state import RUNNER_LOCK, RUNNER_PROCESSES, run_dir, terminate_run
from .stages import set_training_stage

logger = logging.getLogger(__name__)
_CKPT_REGISTER_LOCK = threading.Lock()

# musubi バックエンドのうち、ComfyUI プレビュー生成に対応するモデルファミリー。
# wan21 / hunyuanvideo（動画系）と custom は対象外（T2I 静止画グラフでは表現できない）。
MUSUBI_PREVIEW_SUPPORTED_FAMILIES = {"flux", "qwen_image", "qwen_image_edit", "krea2", "zimage", "anima"}


def _create_preview_jobs_for_new_checkpoints(
    run_id: int, project_id: int, conn, new_ckpts: list[tuple[Path, int]], checkpoint_ids: dict[str, int],
) -> int:
    """新規checkpoint検出時にPreview Job(SSOT)を冪等に作成する。

    _generate_previews_for_checkpoints(旧方式)と同じ早期終了条件(model_family
    がMUSUBI_PREVIEW_SUPPORTED_FAMILIESでない/ModelSpec不明)を踏襲しつつ、
    実際のPreview生成は行わずJob(preview_jobs)の作成のみを行う。実生成は
    dispatch_pending_preview_jobs()がGPU Safe Scheduling下で別途担当する。
    """
    from ...services.preview_prompts import load_active_prompt_items
    from .preview_jobs import ensure_jobs_for_checkpoint, get_instances_per_prompt, sync_checkpoint_status

    row = conn.execute("SELECT config_json FROM training_runs WHERE id=?", (run_id,)).fetchone()
    if not row or not row["config_json"]:
        return 0
    run_cfg = json.loads(row["config_json"])
    model_family = str(run_cfg.get("model_family") or "")
    if model_family not in MUSUBI_PREVIEW_SUPPORTED_FAMILIES and not (model_family=="sdxl" and run_cfg.get("preview_backend")=="comfyui"):
        return 0
    spec = get_spec(model_family)
    if spec is None:
        return 0

    prompt_items, _resolution = load_active_prompt_items(conn)
    instances_per_prompt = get_instances_per_prompt(conn)

    total_created = 0
    for f, epoch in new_ckpts:
        checkpoint_id = checkpoint_ids.get(str(f))
        if checkpoint_id is None:
            continue
        created = ensure_jobs_for_checkpoint(
            conn, checkpoint_id=checkpoint_id, run_id=run_id, project_id=project_id, epoch=epoch,
            prompt_items=prompt_items, instances_per_prompt=instances_per_prompt, model_family=model_family,
        )
        total_created += created
        sync_checkpoint_status(conn, checkpoint_id)
    return total_created


def fail_run(project_id: int, run_id: int, message: str) -> None:
    """run を error 状態にし、理由を training.log に書き込む（フォールバック起動前の即時失敗用）。"""
    log_path = run_dir(run_id) / "logs" / "training.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as lf:
        lf.write(f"[ERROR] {message}\n")
    conn = get_conn()
    conn.execute(
        "UPDATE training_runs SET status = 'error', updated_at = CURRENT_TIMESTAMP, log_path = ?, "
        "failure_code = ?, failure_message = ?, failure_stage = ?, failure_at = CURRENT_TIMESTAMP "
        "WHERE id = ?",
        (str(log_path), "BACKEND_START_FAILED", message, "prepare", run_id),
    )
    conn.execute("UPDATE projects SET status = 'idle' WHERE id = ?", (project_id,))
    conn.commit()
    conn.close()


def proc_alive(run_id: int) -> bool:
    """この run の学習サブプロセスが OS 上で生存しているか（worker再起動を跨いで判定）。"""
    cfg_path = str(run_dir(run_id) / "config.toml")
    run_dir_str = str(run_dir(run_id))
    try:
        import psutil
        for proc in psutil.process_iter(["cmdline"]):
            try:
                cl = proc.info.get("cmdline") or []
                joined = " ".join(str(a) for a in cl)
                if cfg_path in joined or (run_dir_str in joined and "train_network" in joined):
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied, Exception):
                continue
    except Exception:
        pass
    return False


def _generate_previews_for_checkpoints(
    run_id: int,
    project_id: int,
    out_dir: Path,
    new_ckpts: list[tuple[Path, int]],
    conn,
    checkpoint_ids: dict[str, int] | None = None,
) -> None:
    """新規チェックポイントに対して、モデルスペックの preview_backend に応じた
    プレビュー画像を生成し、kohya と同じ命名規則
    (name_e{epoch:06d}_{idx:02d}_{seed}.png) で out_dir/sample/ に書き出す。

    プレビュー生成は学習基盤（app.training）とは独立した PreviewProvider の
    責務であり、失敗しても学習 run の成否には一切影響しない。
    app.preview.registry.generate_preview_safe が Provider 未接続・生成エラーを
    内部ですべて捕捉するため、本関数から例外が外に伝播することはない
    （呼び出し元 scan_and_register_artifacts の try/except は二重の安全策）。

    checkpoint_ids が渡されれば、各チェックポイントについて Preview 生成前に
    ARTIFACT_VALIDATED（safetensors読取・metadata・NaN/Inf・model_family整合性）を
    行い、生成後は PREVIEW_SUCCEEDED/PREVIEW_FAILED を記録する
    （Training成功とPreview成功を無関係な状態にしないため）。
    """
    from ...preview import generate_preview_safe
    from ...preview.base import PreviewRequest
    from ...services.preview_prompts import load_active_prompt_items, resolve_preview_params
    from .artifact_validation import record_preview_outcome

    row = conn.execute(
        "SELECT config_json FROM training_runs WHERE id=?", (run_id,)
    ).fetchone()
    if not row or not row["config_json"]:
        return
    run_cfg = json.loads(row["config_json"])
    model_family = str(run_cfg.get("model_family") or "")
    if model_family not in MUSUBI_PREVIEW_SUPPORTED_FAMILIES:
        return

    spec = get_spec(model_family)
    if spec is None:
        return

    settings_rows = conn.execute(
        "SELECT key, value FROM app_settings WHERE key IN ('comfyui_root','comfyui_url')"
    ).fetchall()
    settings = {str(r["key"]): str(r["value"]) for r in settings_rows}

    # GPU Safe Scheduling: 学習processが生存中はPreview生成を今すぐ実行しない。
    # musubi系(krea2/flux/qwen_image等)の学習はDiT本体を同一GPUへロードし続けて
    # おり、Preview生成が同時にモデルロード/サンプリングを行うとVRAM枯渇で
    # Training/ComfyUIいずれかが例外もログも残さず強制終了するリスクが実運用で
    # 確認された(run 39: 学習processがstep途中で無言終了、同時刻にComfyUIも
    # 接続不能になっていた)。checkpointの登録(ARTIFACT_VALIDATED)はそのまま行い、
    # Preview生成だけを次回scan(学習終了後のpolling、またはtail_monitorの
    # 定期scan)まで保留する。checkpoint.validation_status は 'validated' の
    # ままとなり、UIには「Preview待機中」として表示できる。
    if proc_alive(run_id):
        logger.info(
            "preview: run %s は学習process生存中のためPreview生成を保留(GPU Safe Scheduling)",
            run_id,
        )
        return

    prompt_items, resolution = load_active_prompt_items(conn)
    preview_sampler, preview_cfg, preview_steps = resolve_preview_params(
        conn,
        spec_sampler=spec.preview_sampler,
        spec_cfg=spec.preview_cfg,
        spec_steps=spec.preview_steps,
    )
    sample_dir = out_dir / "sample"
    sample_dir.mkdir(parents=True, exist_ok=True)

    for ckpt_path, epoch in new_ckpts:
        checkpoint_id = checkpoint_ids.get(str(ckpt_path)) if checkpoint_ids else None
        _generate_preview_for_one_checkpoint(
            run_id=run_id, checkpoint_id=checkpoint_id, ckpt_path=ckpt_path, epoch=epoch,
            model_family=model_family, spec=spec, prompt_items=prompt_items, resolution=resolution,
            settings=settings, preview_sampler=preview_sampler, preview_cfg=preview_cfg,
            preview_steps=preview_steps, sample_dir=sample_dir,
        )


def _generate_preview_for_one_checkpoint(
    *, run_id: int, checkpoint_id: int | None, ckpt_path: Path, epoch: int,
    model_family: str, spec, prompt_items: list, resolution: int, settings: dict,
    preview_sampler: str, preview_cfg: float, preview_steps: int, sample_dir: Path,
) -> tuple[bool, str]:
    """1 checkpoint ぶんのPreview生成(ARTIFACT_VALIDATED確認 → 生成 → 結果記録)。

    学習完了直後の自動生成(_generate_previews_for_checkpoints)と、ユーザーが
    手動で再試行するRetry(app.routers.previews.retry_preview)の両方から
    共有される。戻り値: (成功したか, 最後のエラーメッセージ)。
    """
    from ...preview import generate_preview_safe
    from ...preview.base import PreviewRequest
    from .artifact_validation import record_preview_outcome, validate_and_record

    # ARTIFACT_VALIDATED: safetensors読取・metadata・NaN/Inf・model_family整合性を確認する。
    # 壊れているcheckpointでPreview生成を試みてもエラーの切り分けができないため、
    # 検証NGなら Preview 自体を試みずスキップする。
    #
    # validate_and_record() は bool のみを返す(タプルではない)。以前は
    # `ok, detail = validate_and_record(...)` という誤ったunpackがあり、
    # checkpoint_idがNoneでない(=学習完了後の通常経路で必ず該当する)場合に
    # 必ず TypeError: cannot unpack non-iterable bool object でクラッシュしていた
    # 実バグをこのセッションの実Runtime検証(ComfyUI到達不能状態でのRetry呼び出し)
    # で発見した。scan_and_register_artifacts()の外側try/exceptに握り潰されるため、
    # musubi/KREA2系の自動Preview生成がこれまで気づかれずに毎回失敗していた
    # 可能性が高い(直前セッションのKrea2実Preview検証はgenerate_preview_safe()を
    # 直接呼ぶ経路だったため、この自動Preview経路自体は通っていなかった)。
    if checkpoint_id is not None:
        ok = validate_and_record(checkpoint_id, str(ckpt_path), model_family)
        if not ok:
            conn = get_conn()
            row = conn.execute(
                "SELECT validation_detail FROM checkpoints WHERE id = ?", (checkpoint_id,)
            ).fetchone()
            conn.close()
            detail = row["validation_detail"] if row else "artifact validation failed"
            logger.warning("artifact invalid, skipping preview (run %s, epoch %s): %s", run_id, epoch, detail)
            return False, detail

    request = PreviewRequest(
        model_family=model_family,
        checkpoint_path=str(ckpt_path),
        run_id=run_id,
        epoch=epoch,
        prompt_items=prompt_items,
        resolution=resolution,
        settings=settings,
        sampler=preview_sampler,
        cfg=preview_cfg,
        steps=preview_steps,
    )
    results = generate_preview_safe(spec.preview_backend, request)
    any_succeeded = False
    succeeded_count = 0
    last_error = ""
    for result in results:
        if result.image_bytes is None:
            logger.warning(
                "preview: 生成失敗（run %s, epoch %s, prompt %s）: %s",
                run_id, epoch, result.prompt_index, result.error,
            )
            last_error = result.error or "no image_bytes"
            continue
        any_succeeded = True
        succeeded_count += 1
        out_name = f"preview_e{epoch:06d}_{result.prompt_index:02d}_{result.seed}.png"
        (sample_dir / out_name).write_bytes(result.image_bytes)

    if checkpoint_id is not None:
        # expected_count は実際にディスパッチされたprompt数(results数)。
        # 複数Prompt中の一部失敗を"preview_succeeded"へ丸めない(succeeded_count参照)。
        record_preview_outcome(
            checkpoint_id, any_succeeded, "" if any_succeeded else last_error,
            succeeded_count=succeeded_count, expected_count=len(results),
        )
    return any_succeeded, last_error


def retry_preview_for_checkpoint(checkpoint_id: int) -> tuple[bool, str]:
    """ユーザーが手動でPreview再生成を要求したときのエントリポイント
    (app.routers.previews から呼ばれる)。Trainingの再実行は一切行わない。

    checkpoints行からmodel_family(training_runs経由)・run_id・epochを解決し、
    _generate_preview_for_one_checkpoint と同じ経路で再生成する。
    """
    from ...services.preview_prompts import load_active_prompt_items, resolve_preview_params

    conn = get_conn()
    try:
        ckpt = conn.execute(
            "SELECT id, project_id, run_id, file_path, epoch FROM checkpoints WHERE id = ?",
            (checkpoint_id,),
        ).fetchone()
        if ckpt is None:
            return False, "checkpoint not found"
        run_id = ckpt["run_id"]
        if run_id is None:
            return False, "この checkpoint は run_id が不明なため Preview を再生成できません"

        run_row = conn.execute(
            "SELECT config_json FROM training_runs WHERE id = ?", (run_id,)
        ).fetchone()
        if run_row is None or not run_row["config_json"]:
            return False, "training run の設定が見つかりません"
        run_cfg = json.loads(run_row["config_json"])
        # config_json.model_family が未解決の "auto" のまま保存されている古いrun
        # (model_family解決バグ修正 前に作られたrun)にも対応するため、
        # /training/start・/training/resumeと同じ detect_model_family() で
        # 都度解決する(保存値をそのまま信用しない)。実Runtime検証で、
        # config_json.model_family="auto" のまま保存された既存run(修正前に
        # 作成されたrun)に対してRetryを呼ぶと本来krea2であるべきところが
        # "auto"として扱われ誤って対象外エラーになることを発見し、この修正に至った。
        from ...training import detect_model_family

        base_ckpt = str(run_cfg.get("base_checkpoint_path") or "")
        model_family = detect_model_family(base_ckpt, str(run_cfg.get("model_family") or "auto"))
        if model_family not in MUSUBI_PREVIEW_SUPPORTED_FAMILIES:
            return False, (
                f"model_family={model_family} はComfyUI経由の手動Preview再生成に対応していません"
                f"（kohya系はTraining自体が内蔵サンプル生成を行うため、この経路の対象外です）"
            )
        spec = get_spec(model_family)
        if spec is None:
            return False, f"model_family={model_family} のModelSpecが見つかりません"

        settings_rows = conn.execute(
            "SELECT key, value FROM app_settings WHERE key IN ('comfyui_root','comfyui_url')"
        ).fetchall()
        settings = {str(r["key"]): str(r["value"]) for r in settings_rows}

        prompt_items, resolution = load_active_prompt_items(conn)
        preview_sampler, preview_cfg, preview_steps = resolve_preview_params(
            conn, spec_sampler=spec.preview_sampler, spec_cfg=spec.preview_cfg, spec_steps=spec.preview_steps,
        )
        sample_dir = run_dir(int(run_id)) / "output" / "sample"
        sample_dir.mkdir(parents=True, exist_ok=True)
    finally:
        conn.close()

    return _generate_preview_for_one_checkpoint(
        run_id=int(run_id), checkpoint_id=checkpoint_id, ckpt_path=Path(ckpt["file_path"]), epoch=int(ckpt["epoch"]),
        model_family=model_family, spec=spec, prompt_items=prompt_items, resolution=resolution,
        settings=settings, preview_sampler=preview_sampler, preview_cfg=preview_cfg,
        preview_steps=preview_steps, sample_dir=sample_dir,
    )


def scan_and_register_artifacts(run_id: int, project_id: int, *, strict: bool = False, dispatch: bool = True) -> int:
    """output_dir の .safetensors と sample 画像を DB に登録する（冪等）。新規checkpoint数を返す。"""
    out = run_dir(run_id) / "output"
    if not out.exists():
        return 0
    conn = get_conn()
    added = 0
    try:
        run_meta = conn.execute(
            "SELECT total_epochs, config_json FROM training_runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        total_epochs = int(run_meta["total_epochs"] or 0) if run_meta else 0
        try:
            run_config = json.loads(run_meta["config_json"] or "{}") if run_meta else {}
        except (TypeError, ValueError):
            run_config = {}
        final_output_name = str(run_config.get("output_name") or "")
        existing_ckpt = {
            r["file_path"]: r["id"]
            for r in conn.execute(
                "SELECT id, file_path FROM checkpoints WHERE project_id=?", (project_id,)
            )
        }
        ckpts = sorted(out.glob("*.safetensors"))
        # 保存日時・設定・環境などをLoRAのヘッダへ埋め込む(冪等・失敗しても登録は続行)。
        # DB登録・sha256・プレビュー生成より前に行い、以後は埋め込み後のファイルを正とする。
        try:
            from ..lora_metadata import stamp_run_outputs
            stamp_run_outputs(conn, run_id, project_id, ckpts)
        except Exception as exc:  # noqa: BLE001
            logger.warning("lora metadata stamping skipped (run %s): %s", run_id, exc)
        new_ckpts: list[tuple[Path, int]] = []
        with _CKPT_REGISTER_LOCK:  # two scanners (monitor loop / status refresh) registering the same file made duplicate rows and doubled the previews
            existing_ckpt = {
                row["file_path"]: row["id"] for row in conn.execute("SELECT id, file_path FROM checkpoints WHERE project_id=?", (project_id,))
            }
            for f in ckpts:
                fp = str(f)
                if fp in existing_ckpt:
                    conn.execute(
                        "UPDATE checkpoints SET run_id=? WHERE id=? AND (run_id IS NULL OR run_id<>?)",
                        (run_id, existing_ckpt[fp], run_id),
                    )
                    continue
                m = RE_CKPT_EPOCH.search(f.name)
                epoch = int(m.group(1)) if m else (
                    total_epochs if final_output_name and f.stem == final_output_name else 0
                )
                if run_config.get('preview_backend')=='comfyui' and f.stem==final_output_name and conn.execute('SELECT 1 FROM checkpoints WHERE run_id=? AND epoch=?',(run_id,epoch)).fetchone():
                    continue  # Final file aliases the already sampled last-epoch weights.
                cur = conn.cursor()
                cur.execute(
                    "INSERT INTO checkpoints(project_id, run_id, file_path, epoch, step, mark) "
                    "VALUES (?,?,?,?,0,'none')",
                    (project_id, run_id, fp, epoch),
                )
                existing_ckpt[fp] = cur.lastrowid
                added += 1
                new_ckpts.append((f, epoch))

            # 上記INSERT/UPDATEをここで確定させる。validate_and_record / _generate_previews_for_checkpoints
            # (record_preview_outcome含む)はそれぞれ独自にget_conn()で別接続を開くため、
            # この commit を挟まないと「同一DBファイルへ書き込み中の未コミットトランザクション」と
            # 衝突し sqlite3.OperationalError: database is locked で毎回サイレントに失敗する
            # (呼び出し元の except Exception: pass で握り潰され、added の値だけが返っていた)。
            # model_family非依存の共通層のバグであり、SDXL固有の修正ではない。
            conn.commit()

        # ARTIFACT_VALIDATED: musubi/kohya を問わず、新規チェックポイントは全て
        # safetensors読取・metadata・NaN/Inf・model_family整合性を検証する。
        # (musubi系はこの後 _generate_previews_for_checkpoints 内でも呼ばれるが、
        #  validate_and_record は同一checkpointに対して冪等 — 二重実行しても結果は変わらない)
        if new_ckpts:
            cfg_row = conn.execute("SELECT config_json FROM training_runs WHERE id=?", (run_id,)).fetchone()
            run_model_family = ""
            if cfg_row and cfg_row["config_json"]:
                try:
                    run_model_family = str(json.loads(cfg_row["config_json"]).get("model_family") or "")
                except (ValueError, TypeError):
                    pass
            for f, _epoch in new_ckpts:
                validate_and_record(existing_ckpt[str(f)], str(f), run_model_family)

        # musubi系（Flux/Qwen-Image/KREA2/Z-Image）は kohya と違いサンプル生成機能を
        # 内蔵しないため、新規チェックポイントが出たタイミングでPreview Job(SSOT)を
        # 作成する。Preview Job単位の永続状態(preview_jobs)がSingle Source of Truthで
        # あり、checkpoints.validation_statusはそこからの派生集計値。
        #
        # 実装Job数は「保留(pending)」のままDBに残り続けるため、GPU Safe Scheduling
        # (学習process生存中はdispatchしない)や、学習終了後の自動再開(Missing Preview
        # Recovery)も、new_ckptsに含まれるかどうかに依存しない(以前のcheckpoint単位
        # 方式では、保留されたcheckpointが"new"でなくなり永久にPreview欠落する実バグが
        # あった)。
        if new_ckpts:
            set_training_stage(run_id, "checkpoint", "completed", f"{len(new_ckpts)} checkpoint(s) detected")
            try:
                _create_preview_jobs_for_new_checkpoints(run_id, project_id, conn, new_ckpts, existing_ckpt)
            except Exception as exc:  # noqa: BLE001
                logger.warning("preview job creation skipped (run %s): %s", run_id, exc)

        try:
            # Real preview state is set only by an admitted Comfy epoch window.
            from .preview_jobs import dispatch_pending_preview_jobs, ensure_preview_wait_monitor

            if dispatch:
                dispatch_pending_preview_jobs(conn, max_jobs=1)
                ensure_preview_wait_monitor()
        except Exception as exc:  # noqa: BLE001
            logger.warning("preview job dispatch skipped (run %s): %s", run_id, exc)

        epoch_to_ckpt: dict[int, int] = {}
        latest_ckpt_id: int | None = None
        for r in conn.execute(
            "SELECT id, epoch FROM checkpoints WHERE project_id=? AND run_id=? ORDER BY epoch",
            (project_id, run_id),
        ):
            epoch_to_ckpt[int(r["epoch"])] = int(r["id"])
            latest_ckpt_id = int(r["id"])

        sample_dir = out / "sample"
        if sample_dir.exists() and epoch_to_ckpt:
            registered_imgs = {
                r["image_path"]
                for r in conn.execute(
                    "SELECT ps.image_path FROM preview_samples ps "
                    "JOIN checkpoints c ON c.id = ps.checkpoint_id WHERE c.project_id=?",
                    (project_id,),
                )
            }
            for img in sorted(sample_dir.glob("*.png")):
                ip = str(img)
                if ip in registered_imgs:
                    continue
                mi = RE_SAMPLE_EPOCH_IDX.search(img.name)
                if mi:
                    s_epoch = int(mi.group(1))
                    slot = str(int(mi.group(2)))
                else:
                    me = RE_SAMPLE_EPOCH.search(img.name)
                    s_epoch = int(me.group(1)) if me else None
                    slot = "0"
                target = epoch_to_ckpt.get(s_epoch) if s_epoch is not None else latest_ckpt_id
                if target is None:
                    target = latest_ckpt_id
                if target is None:
                    continue
                conn.execute(
                    "INSERT INTO preview_samples(checkpoint_id, slot, image_path) VALUES (?,?,?)",
                    (target, slot, ip),
                )
                registered_imgs.add(ip)

        if ckpts:
            latest_registered = conn.execute(
                "SELECT file_path FROM checkpoints WHERE run_id = ? ORDER BY epoch DESC, id DESC LIMIT 1",
                (run_id,),
            ).fetchone()
            conn.execute(
                "UPDATE training_runs SET latest_checkpoint_path=? WHERE id=?",
                (str(latest_registered["file_path"]) if latest_registered else str(ckpts[-1]), run_id),
            )
        conn.commit()
    except Exception:
        # 完全な握りつぶし禁止: このtry節はcheckpoints/preview_samples registrationと
        # ARTIFACT_VALIDATED処理をまとめて保護する「学習ループを壊さないための安全策」だが、
        # 従来は`pass`のみでログすら残さず、DB書き込み失敗(preview_samples INSERT失敗等)が
        # 完全にサイレントになっていた実バグがあった(実コード監査で発見)。
        # record_preview_outcome()は別トランザクションでchekpoints.validation_statusを
        # 'preview_succeeded'に更新済みのため、ここが失敗すると「検証済み扱いなのに
        # サムネイルが出ない」という説明不能な不整合をユーザーに見せてしまう。
        # 学習ループへの例外伝播は引き続き防ぎつつ、原因をログへ必ず残す。
        logger.exception("scan_and_register_artifacts failed (run=%s, project=%s)", run_id, project_id)
        if strict:
            raise
    finally:
        conn.close()
    return added


def reconstruct_run_state(row: dict) -> dict:
    """DB行 + ログ + ディスク + プロセス生存から、表示用の実態状態を再構成する。

    返値: status / epoch / step / total_epochs / steps_per_epoch / loss /
          done_steps / total_steps / alive / source の統合ビュー。
    DB が古い場合はログ由来の値で上書きし、status の不整合（training なのに
    プロセス停止）を reconcile する。
    """
    run_id = int(row["id"])
    project_id = int(row["project_id"]) if "project_id" in row.keys() else None
    db_status = row["status"]
    steps_per_epoch = int(row["steps_per_epoch"] or 0)
    total_epochs = int(row["total_epochs"] or 0)

    log_state = parse_log_state(row["log_path"])

    epoch = log_state["epoch"] if log_state["epoch"] is not None else int(row["current_epoch"] or 0)
    loss = log_state["loss"] if log_state["loss"] is not None else row["loss"]
    if log_state["total_epochs"]:
        total_epochs = log_state["total_epochs"]

    if log_state["gtotal"]:
        total_steps = int(log_state["gtotal"])
        done_steps = int(log_state["gstep"] or 0)
        steps_per_epoch = (total_steps // total_epochs) if total_epochs > 0 else steps_per_epoch
        computed_epoch = (done_steps // steps_per_epoch) if steps_per_epoch > 0 else epoch
        computed_epoch = min(computed_epoch, total_epochs)
        if computed_epoch > epoch:
            epoch = computed_epoch
        step = done_steps - epoch * steps_per_epoch if steps_per_epoch else done_steps
    else:
        total_steps = total_epochs * steps_per_epoch
        step = int(row["current_step"] or 0)
        done_steps = epoch * steps_per_epoch + step

    alive = proc_alive(run_id)

    if project_id is not None:
        scan_and_register_artifacts(run_id, project_id)

    status = db_status
    note = None
    if db_status == "training":
        has_progress = (log_state["gstep"] or 0) > 0 or bool(log_state["done_marker"])
        age: float | None = None
        try:
            keys = row.keys()
            base_ts = (row["updated_at"] if "updated_at" in keys else None) \
                or (row["started_at"] if "started_at" in keys else None)
            if base_ts:
                base_at = datetime.fromisoformat(str(base_ts).replace(" ", "T"))
                age = (datetime.utcnow() - base_at).total_seconds()
        except Exception:
            age = None
        GRACE = 180.0  # 起動～モデルロードの猶予

        if alive:
            status = "training"
        elif log_state["done_marker"] or (total_epochs > 0 and epoch >= total_epochs):
            status = "completed"
        elif not has_progress and (age is None or age < GRACE):
            # 起動準備中（テキストエンコーダ/latentキャッシュ生成の子プロセスは
            # proc_alive のパターンに一致しないため、この間 alive=False になる）。
            status = "training"
            note = "学習プロセスを起動中…（モデルロード中）"
        elif log_state["error"]:
            status = "error"
            note = log_state["error"]
        elif has_progress:
            status = "paused"
            note = "プロセスが停止しています（中断）。再開または再学習してください。"
        else:
            status = "error"
            note = "学習プロセスの起動に失敗しました。ログを確認してください。"
        if status != db_status and status != "training" and project_id is not None:
            try:
                conn = get_conn()
                conn.execute(
                    "UPDATE training_runs SET status=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                    (status, run_id),
                )
                proj_status = "completed" if status == "completed" else ("idle" if status == "error" else "paused")
                conn.execute("UPDATE projects SET status=? WHERE id=?", (proj_status, project_id))
                conn.commit()
                conn.close()
            except Exception:
                pass

    return {
        "status": status, "epoch": epoch, "step": max(0, step), "loss": loss,
        "total_epochs": total_epochs, "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps, "done_steps": done_steps,
        "alive": alive, "sit": log_state["sit"], "note": note,
    }


def epoch_stop_checkpoint(run_id: int, stable: dict) -> int | None:
    """Only accept a newly saved, stable, readable checkpoint after the request."""
    conn = get_conn()
    try:
        run = conn.execute("SELECT config_json,stop_mode,status FROM training_runs WHERE id=?", (run_id,)).fetchone()
        if not run or run["status"] != "training" or run["stop_mode"] != "epoch":
            return None
        cfg = json.loads(run["config_json"] or "{}")
        # Older requests have no durable boundary; establish it rather than stop on an old file.
        if "epoch_stop_after_checkpoint_id" not in cfg:
            last = conn.execute("SELECT COALESCE(MAX(id),0) AS id FROM checkpoints WHERE run_id=?", (run_id,)).fetchone()
            cfg["epoch_stop_after_checkpoint_id"] = int(last["id"])
            conn.execute("UPDATE training_runs SET config_json=? WHERE id=?", (json.dumps(cfg),run_id))
            conn.commit()
            return None
        rows = conn.execute("SELECT id,file_path,epoch FROM checkpoints WHERE run_id=? AND id>? AND epoch>0 ORDER BY id DESC", (run_id,int(cfg["epoch_stop_after_checkpoint_id"]))).fetchall()
    finally:
        conn.close()
    for row in rows:
        try:
            path = Path(row["file_path"]); stat = path.stat()
            if stat.st_mtime_ns <= int(cfg.get("epoch_stop_requested_at_ns", 0)):
                continue
            signature = (stat.st_size, stat.st_mtime_ns)
            previous = stable.get(row["id"])
            stable[row["id"]] = signature
            if signature != previous or not stat.st_size:
                continue
            ok, _detail = validate_artifact(str(path),str(cfg.get("model_family") or ""))
            latest = path.stat()
            if ok and (latest.st_size,latest.st_mtime_ns) == signature:
                return int(row["id"])
        except OSError:
            continue
    return None


def tail_monitor(
    run_id: int, project_id: int, proc: "subprocess.Popen | None",
    sec_factor: float, gpu_name: str | None, model_family: str,
) -> None:
    """ログファイルを tail して DB 更新・成果物登録・終了時の status 確定を行う。

    proc 保有（新規起動）→ proc を監視。proc=None（再アタッチ）→ OS プロセス生存を監視。
    学習プロセスの stdout はログファイルに直接リダイレクトされるため、ワーカーが死んでも
    サブプロセスはブロックせず書き込み続け、再アタッチ時はログ末尾から追従できる。
    """
    log_path = run_dir(run_id) / "logs" / "training.log"
    pos = 0
    epoch_stop_stable: dict = {}
    while True:
        chunk = ""
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                f.seek(pos)
                chunk = f.read()
                pos = f.tell()
        except OSError:
            pass

        saw_save = False
        for line in chunk.splitlines():
            parsed = parse_train_line(line)
            if parsed:
                updates: list[str] = []
                params: list = []
                if "epoch" in parsed:
                    updates.append("current_epoch = ?")
                    params.append(parsed["epoch"])
                if "step" in parsed:
                    updates.append("current_step = ?")
                    params.append(parsed["step"])
                    record_step_timing(
                        project_id, int(parsed["step"]), sec_factor,
                        gpu_name, model_family, run_id, line,
                    )
                if "loss" in parsed:
                    updates.append("loss = ?")
                    params.append(parsed["loss"])
                if updates:
                    try:
                        c = get_conn()
                        updates.append("updated_at = CURRENT_TIMESTAMP")
                        c.execute(
                            f"UPDATE training_runs SET {', '.join(updates)} WHERE id = ?",
                            [*params, run_id],
                        )
                        c.commit()
                        c.close()
                    except Exception:
                        pass
            if RE_SAVE.search(line):
                saw_save = True
        if saw_save:
            scan_and_register_artifacts(run_id, project_id)

        # 停止要求: ユーザーが明示停止したとき(stop_mode='now')のみ即終了する。
        try:
            c = get_conn()
            st = c.execute(
                "SELECT status, stop_mode FROM training_runs WHERE id = ?", (run_id,)
            ).fetchone()
            c.close()
        except Exception:
            st = None
        if st and st["stop_mode"] == "epoch":
            scan_and_register_artifacts(run_id, project_id)
            saved_id = epoch_stop_checkpoint(run_id, epoch_stop_stable)
            if saved_id is not None:
                c = get_conn()
                try:
                    changed = c.execute("UPDATE training_runs SET status='paused',updated_at=CURRENT_TIMESTAMP WHERE id=? AND status='training' AND stop_mode='epoch'", (run_id,)).rowcount
                    if changed:
                        c.execute("UPDATE projects SET status='paused' WHERE id=?", (project_id,))
                    c.commit()
                finally:
                    c.close()
                if changed:
                    logger.info("Epoch stop after verified checkpoint %s (run %s)", saved_id, run_id)
                    terminate_run(run_id, proc)
                    break
        if st and st["stop_mode"] == "now":
            terminate_run(run_id, proc)
            break

        alive = (proc.poll() is None) if proc is not None else proc_alive(run_id)
        if not alive:
            break
        time.sleep(1.0)

    if proc is not None:
        # 固定5秒待機で「完了扱い」にするだけでは、CUDAコンテキスト解放等で
        # 5秒を超えるケースでプロセスが実際には生きたままDB状態だけが
        # 確定してしまう(次のqueued runがGPU競合を起こす・ゾンビプロセスが
        # 残る等のリスク)。TimeoutExpiredを握り潰さず、まず追加10秒待ち、
        # それでも終了しなければ強制終了(kill)してから再度終了を確認する。
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("training subprocess (run %s) が5秒以内に終了しなかったため追加待機します", run_id)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning(
                    "training subprocess (run %s) が15秒経過しても終了しないため強制終了します", run_id
                )
                try:
                    proc.kill()
                    proc.wait(timeout=5)
                except Exception as exc:
                    logger.error("training subprocess (run %s) の強制終了に失敗: %s", run_id, exc)
        except Exception:
            pass

    # 終了時の最終スキャンと status 確定。Trainerのreturn code=0だけでは
    # 成功にしない。Checkpointの登録・検証・Library candidate化までが必要。
    artifact_error: str | None = None
    try:
        scan_and_register_artifacts(run_id, project_id, strict=True)
    except Exception as exc:  # noqa: BLE001
        artifact_error = f"artifact scan failed: {exc}"
    log_state = parse_log_state(log_path)
    try:
        c = get_conn()
        cur = c.execute(
            "SELECT status, total_epochs FROM training_runs WHERE id = ?", (run_id,)
        ).fetchone()
        completed_stage = False
        cur_status = cur["status"] if cur else "unknown"
        total_epochs = int(cur["total_epochs"]) if cur and cur["total_epochs"] else 0
        if cur_status != "paused":
            if proc is not None:
                done = proc.returncode == 0
            else:
                done = bool(log_state["done_marker"]) or (
                    log_state["epoch"] is not None and total_epochs > 0
                    and log_state["epoch"] >= total_epochs
                )
            checkpoint = c.execute(
                "SELECT id, file_path, validation_status FROM checkpoints WHERE run_id = ? "
                "ORDER BY epoch DESC, id DESC LIMIT 1",
                (run_id,),
            ).fetchone()
            if done and artifact_error is None and checkpoint is None:
                artifact_error = "trainer exited successfully but no Checkpoint was registered"
            if done and artifact_error is None and (
                not Path(str(checkpoint["file_path"] or "")).is_file()
                or str(checkpoint["validation_status"] or "none") == "invalid"
            ):
                artifact_error = (
                    f"Checkpoint #{checkpoint['id']} is missing or invalid "
                    f"(status={checkpoint['validation_status']})"
                )
            if done and artifact_error is None:
                try:
                    auto_register_lora_asset(c, project_id, run_id)
                except Exception as exc:  # noqa: BLE001
                    artifact_error = f"Library candidate registration failed: {exc}"
            if done and artifact_error is None:
                c.execute(
                    "UPDATE training_runs SET status='completed', failure_code=NULL, failure_message=NULL, "
                    "failure_stage=NULL, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                    (run_id,),
                )
                c.execute("UPDATE projects SET status='completed' WHERE id=?", (project_id,))
                completed_stage = True
            elif done:
                c.execute(
                    "UPDATE training_runs SET status='error', updated_at=CURRENT_TIMESTAMP, "
                    "failure_code='FINAL_ARTIFACT_FAILED', failure_message=?, failure_stage='finalize', "
                    "failure_at=CURRENT_TIMESTAMP WHERE id=?",
                    (artifact_error, run_id),
                )
                c.execute("UPDATE projects SET status='idle' WHERE id=?", (project_id,))
            else:
                new = "error" if log_state["error"] else "paused"
                failure_code = "TRAINER_PROCESS_FAILED" if new == "error" else None
                failure_message = str(log_state.get("error") or "trainer process stopped without a completion marker") if new == "error" else None
                c.execute(
                    "UPDATE training_runs SET status=?, updated_at=CURRENT_TIMESTAMP, "
                    "failure_code=?, failure_message=?, failure_stage=?, failure_at=CASE WHEN ? IS NOT NULL THEN CURRENT_TIMESTAMP ELSE failure_at END "
                    "WHERE id=?",
                    (new, failure_code, failure_message, "training" if failure_code else None, failure_code, run_id),
                )
                c.execute(
                    "UPDATE projects SET status=? WHERE id=?",
                    ("idle" if new == "error" else "paused", project_id),
                )
        c.commit()
        c.close()
        if completed_stage:
            set_training_stage(run_id, "training", "completed")
            set_training_stage(run_id, "finalize", "completed")
    except Exception:
        logger.exception("final training reconciliation failed (run=%s, project=%s)", run_id, project_id)
    with RUNNER_LOCK:
        RUNNER_PROCESSES.pop(project_id, None)


def reattach_active_runs(sec_factor_fn, on_run_finished=None) -> None:
    """起動時に status='training' の run を検出し、生存中なら tail 監視を再開、
    死亡していれば実態に合わせて status を reconcile する。

    sec_factor_fn: cfg(dict) -> (sec_factor, gpu_name, model_family) を返す関数
    （backend 実装から注入。循環 import 回避のため）。
    on_run_finished: 引数なしで呼ばれるコールバック（省略可）。再アタッチした
    tail_monitor が完了した直後に呼ばれる。Training Queueの自動進行
    (_dispatch_next_queued_job)を、通常経路(_run_training_backendのfinally)
    だけでなく、Backend再起動によるreattach経路でも確実に発火させるために
    使う（reattach後はtail_monitorを直接呼ぶため、通常経路のfinallyフックを
    素通りしてしまうことの補完）。
    """
    try:
        c = get_conn()
        rows = c.execute(
            "SELECT id, project_id, config_json FROM training_runs WHERE status = 'training'"
        ).fetchall()
        c.close()
    except Exception:
        return
    for r in rows:
        rid = int(r["id"])
        pid = int(r["project_id"])
        try:
            cfg = json.loads(r["config_json"] or "{}")
        except Exception:
            cfg = {}
        sec_factor, gpu_name, model_family = sec_factor_fn(cfg)
        if proc_alive(rid):
            import threading

            def _run_and_dispatch(rid=rid, pid=pid, sec_factor=sec_factor,
                                   gpu_name=gpu_name, model_family=model_family):
                try:
                    tail_monitor(rid, pid, None, sec_factor, gpu_name, model_family)
                finally:
                    if on_run_finished is not None:
                        on_run_finished()

            t = threading.Thread(target=_run_and_dispatch, daemon=True)
            t.start()
        else:
            try:
                c = get_conn()
                # started_at/updated_at/step1_at を含めないと reconstruct_run_state() が
                # age を計算できず(age=None)、「起動直後の猶予期間」分岐に恒久的に
                # 該当してしまい、本当に古い(前回セッションから取り残された)runが
                # いつまでも 'training' のまま reconcile されない実バグがあった
                # (Training Queueのdispatch_next()がGPU busyと誤判定し続け、
                # queued runが永久に開始されない事象として顕在化する)。
                row = c.execute(
                    "SELECT id, project_id, status, current_epoch, current_step, "
                    "total_epochs, steps_per_epoch, loss, log_path, started_at, updated_at, step1_at "
                    "FROM training_runs WHERE id=?",
                    (rid,),
                ).fetchone()
                c.close()
                if row:
                    reconstruct_run_state(row)
            except Exception:
                pass
    # 生存プロセスが1件もなかった(=全runが即座にreconcileされた)場合でも、
    # 「runningが無くqueuedが残っている」状態からの復旧としてdispatchを試みる。
    if on_run_finished is not None and not any(proc_alive(int(r["id"])) for r in rows):
        on_run_finished()
