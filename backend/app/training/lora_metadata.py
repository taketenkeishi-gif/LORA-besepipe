"""保存済みLoRA(.safetensors)へ「いつ・どう・どの環境で学習したか」を埋め込む。

kohya系トレーナー(sd-scripts)は自前の ss_* メタデータを書くが、
  * ComfyUIエポックプレビュー用フックが保存するファイルは ss_* が8個しかない
  * 日時はUNIX秒、トリガー/インスタンス/データセット・スナップショット/GPU/環境/所要時間は無い
ので、成果物登録時にこのモジュールが **ヘッダの __metadata__ だけ** を書き換えて補う。

方針:
  * トレーナーが書いた ss_* は上書きしない(欠けているキーだけ補完)。
  * アプリ固有の情報は lssn_* 名前空間へ必ず書く(lssn_schema があれば処理済み=冪等)。
  * テンソル部は1バイトも変えない。ストリーミングコピーで一時ファイルへ書き、os.replaceで差し替える。
  * ファイルが書き込み途中(ヘッダのオフセットとファイルサイズが合わない)なら触らない。
"""
from __future__ import annotations

import json
import logging
import os
import re
import struct
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SCHEMA = "1"
APP_NAME = "LoRA-Studio-Next"
APP_VERSION = "0.1.0"
JST = timezone(timedelta(hours=9))
_MAX_HEADER = 100 * 1024 * 1024
_CHUNK = 8 * 1024 * 1024
_EPOCH_RE = re.compile(r"-(\d{6})\.safetensors$", re.IGNORECASE)


# ---------------------------------------------------------------- header I/O
def read_header(path: str | Path) -> tuple[dict, int, int]:
    """(header_dict, header_len, file_size)。テンソルは読まない。"""
    p = Path(path)
    size = p.stat().st_size
    with p.open("rb") as fh:
        raw = fh.read(8)
        if len(raw) < 8:
            raise ValueError("file too small")
        n = struct.unpack("<Q", raw)[0]
        if n <= 0 or n > _MAX_HEADER or 8 + n > size:
            raise ValueError(f"invalid safetensors header length {n}")
        header = json.loads(fh.read(n).decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError("header is not an object")
    return header, n, size


def read_metadata(path: str | Path) -> dict[str, str]:
    header, _n, _size = read_header(path)
    meta = header.get("__metadata__") or {}
    return {str(k): str(v) for k, v in meta.items()}


def _data_is_complete(header: dict, header_len: int, file_size: int) -> bool:
    end = 0
    for key, info in header.items():
        if key == "__metadata__":
            continue
        offsets = info.get("data_offsets") if isinstance(info, dict) else None
        if not offsets or len(offsets) != 2:
            return False
        end = max(end, int(offsets[1]))
    return 8 + header_len + end == file_size


def rewrite_metadata(path: str | Path, updates: dict[str, str], *, overwrite: bool = False) -> bool:
    """__metadata__ に updates を追加(overwrite=Falseなら既存キーは保持)。変更があれば True。

    テンソルバイトはストリーミングコピーで完全に保持する。
    """
    p = Path(path)
    header, n, size = read_header(p)
    if not _data_is_complete(header, n, size):
        raise ValueError("tensor data does not match header (file may still be written)")
    meta = {str(k): str(v) for k, v in (header.get("__metadata__") or {}).items()}
    changed = False
    for key, value in updates.items():
        value = str(value)
        if key in meta and not overwrite:
            continue
        if meta.get(key) != value:
            meta[key] = value
            changed = True
    if not changed:
        return False
    new_header = {"__metadata__": meta}
    new_header.update({k: v for k, v in header.items() if k != "__metadata__"})
    blob = json.dumps(new_header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    blob += b" " * ((8 - len(blob) % 8) % 8)
    tmp = p.with_name(p.name + ".lssn-tmp")
    try:
        with p.open("rb") as src, tmp.open("wb") as dst:
            dst.write(struct.pack("<Q", len(blob)))
            dst.write(blob)
            src.seek(8 + n)
            while True:
                chunk = src.read(_CHUNK)
                if not chunk:
                    break
                dst.write(chunk)
            dst.flush()
            os.fsync(dst.fileno())
        if tmp.stat().st_size != 8 + len(blob) + (size - 8 - n):
            raise OSError("rewritten file size mismatch")
        last: Exception | None = None
        for _ in range(5):
            try:
                os.replace(tmp, p)
                last = None
                break
            except PermissionError as exc:  # 他プロセスが開いている
                last = exc
                time.sleep(0.5)
        if last is not None:
            raise last
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    return True


# ------------------------------------------------------------ context build
def _json(value: Any) -> dict:
    try:
        parsed = json.loads(value or "{}")
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _jst(ts: float | None = None, text_utc: str | None = None) -> str:
    if text_utc:
        s = str(text_utc).strip().replace("T", " ")
        try:
            dt = datetime.fromisoformat(s.split("+")[0].strip()).replace(tzinfo=timezone.utc)
            return dt.astimezone(JST).isoformat(timespec="seconds")
        except ValueError:
            return ""
    if ts is None:
        return ""
    return datetime.fromtimestamp(ts, JST).isoformat(timespec="seconds")


def _utc_epoch(text_utc: str | None) -> float | None:
    try:
        s = str(text_utc or "").strip().replace("T", " ").split("+")[0].strip()
        return datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return None


def _trainer_env(resolved: dict) -> dict[str, str]:
    """学習venvのパッケージ情報をdist-infoのフォルダ名から読む(サブプロセスなし)。"""
    out: dict[str, str] = {}
    exe = str(resolved.get("python_executable") or "")
    if not exe:
        return out
    try:
        venv = Path(exe).resolve().parent.parent
        cfg = venv / "pyvenv.cfg"
        if cfg.is_file():
            for line in cfg.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.lower().startswith("version"):
                    out["python"] = line.split("=", 1)[1].strip()
                    break
        site = venv / "Lib" / "site-packages"
        if site.is_dir():
            for pkg in ("torch", "torchvision", "bitsandbytes", "accelerate", "transformers", "diffusers"):
                hit = next(iter(site.glob(f"{pkg}-*.dist-info")), None)
                if hit is not None:
                    out[pkg] = hit.name[len(pkg) + 1:-len(".dist-info")]
    except OSError:
        pass
    return out


def build_run_context(conn, run_id: int, project_id: int) -> dict[str, Any]:
    """Run 全体で共通の値を集める(チェックポイントごとの値は build_checkpoint_metadata)。"""
    row = conn.execute(
        "SELECT started_at, total_epochs, steps_per_epoch, config_json, resolved_config_json, "
        "requested_config_json, dataset_snapshot_id FROM training_runs WHERE id=?",
        (run_id,),
    ).fetchone()
    ctx: dict[str, Any] = {"run_id": run_id, "project_id": project_id}
    if row is None:
        return ctx
    cfg = _json(row["config_json"])
    cfg.update({k: v for k, v in _json(row["requested_config_json"]).items() if k not in cfg})
    resolved = _json(row["resolved_config_json"])
    cfg.update({k: v for k, v in resolved.items() if k not in cfg})
    ctx["cfg"] = cfg
    ctx["resolved"] = resolved
    ctx["started_at_text"] = row["started_at"]
    ctx["started_ts"] = _utc_epoch(row["started_at"])
    ctx["total_epochs"] = int(row["total_epochs"] or 0)
    ctx["steps_per_epoch"] = int(row["steps_per_epoch"] or 0)
    ctx["snapshot_id"] = row["dataset_snapshot_id"]
    proj = conn.execute("SELECT name FROM projects WHERE id=?", (project_id,)).fetchone()
    ctx["project_name"] = str(proj["name"]) if proj else ""
    ctx["env"] = _trainer_env(resolved)

    captions: list[str] = []
    snap_row = None
    if row["dataset_snapshot_id"] is not None:
        snap_row = conn.execute(
            "SELECT id, name, snapshot_hash, item_count FROM basepipe_dataset_snapshots WHERE id=?",
            (row["dataset_snapshot_id"],),
        ).fetchone()
        try:
            from .snapshot_inputs import fetch_snapshot_inputs
            captions = [str(r.get("caption_at_snapshot") or "") for r in fetch_snapshot_inputs(conn, int(row["dataset_snapshot_id"]))]
        except Exception as exc:  # noqa: BLE001
            logger.warning("snapshot captions unavailable for run %s: %s", run_id, exc)
    ctx["snapshot"] = dict(snap_row) if snap_row else {}
    tag_lists = [[t.strip() for t in re.split(r"[,\n]", c) if t.strip()] for c in captions]
    ctx["tag_lists"] = tag_lists

    concepts = conn.execute(
        "SELECT name, concept_type, trigger_token FROM basepipe_concepts WHERE project_id=? ORDER BY id",
        (project_id,),
    ).fetchall()
    folded = [{x.casefold() for x in t} for t in tag_lists]
    triggers, instances = [], []
    for c in concepts:
        tok = str(c["trigger_token"] or "").strip()
        if not tok:
            continue
        matched = sum(tok.casefold() in f for f in folded)
        entry = {"name": c["name"], "type": c["concept_type"], "trigger": tok, "images": matched}
        (instances if c["concept_type"] == "outfit" else triggers).append(entry)
    ctx["triggers"], ctx["instances"] = triggers, instances
    ctx["instance_hints"] = _instance_hints(captions, instances)
    return ctx


def _instance_hints(captions: list[str], instances: list[dict]) -> dict[str, dict[str, list[str]]]:
    """Per outfit trigger: the tags that mark the outfit and the tags that other outfits show but this one never does.

    Generation tools can use it so a feature that belongs to one outfit (a hair ribbon) is not drawn for the others without typing negatives each time.
    """
    triggers = [str(i["trigger"]) for i in instances if i.get("trigger")]
    if len(triggers) < 2 or not captions:
        return {}
    try:
        from .runtime.preview_jobs import outfit_negative_tags, outfit_signature_tags
        pos = outfit_signature_tags(captions, triggers)
        neg = outfit_negative_tags(captions, triggers)
    except Exception as exc:  # noqa: BLE001
        logger.warning("instance hints skipped: %s", exc)
        return {}
    # "separable_tags" are NOT part of the outfit itself: they may still be drawn together with it when the prompt asks for them
    # (e.g. jacket + white hair ribbon). Tools should leave them out by default and stop leaving them out once the prompt names the tag.
    # "negative_tags" repeats the same list for older readers.
    return {t: {"tags": pos.get(t, []), "separable_tags": neg.get(t, []), "negative_tags": neg.get(t, []),
                "rule": "default_negative_unless_in_positive_prompt"} for t in triggers}


def _s(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def build_checkpoint_metadata(ctx: dict[str, Any], ckpt: Path, existing: dict[str, str]) -> tuple[dict[str, str], dict[str, str]]:
    """(lssn_* 全部, 欠けていれば補う ss_*) を返す。値は全て str。"""
    cfg: dict = ctx.get("cfg") or {}
    resolved: dict = ctx.get("resolved") or {}
    backend_cfg: dict = resolved.get("backend_resolved") or {}
    m = _EPOCH_RE.search(ckpt.name)
    epoch = int(existing.get("ss_epoch") or (m.group(1) if m else 0) or 0) or ctx.get("total_epochs", 0)
    step = existing.get("ss_steps") or (str(epoch * ctx["steps_per_epoch"]) if ctx.get("steps_per_epoch") else "")
    try:
        mtime = ckpt.stat().st_mtime
    except OSError:
        mtime = time.time()
    started = ctx.get("started_ts")
    snap = ctx.get("snapshot") or {}
    gpu = resolved.get("gpu_mapping") or {}
    env = ctx.get("env") or {}
    base_path = str(cfg.get("base_checkpoint_path") or "")
    base_stat = None
    try:
        if base_path and Path(base_path).is_file():
            base_stat = Path(base_path).stat()
    except OSError:
        pass
    total_steps = resolved.get("total_steps") or existing.get("ss_max_train_steps") or ""

    lssn: dict[str, str] = {
        "lssn_schema": SCHEMA,
        "lssn_app": APP_NAME,
        "lssn_app_version": APP_VERSION,
        "lssn_app_build": _build_hash(),
        "lssn_run_id": str(ctx.get("run_id")),
        "lssn_project": ctx.get("project_name", ""),
        "lssn_trained_started_at": _jst(text_utc=ctx.get("started_at_text")),
        "lssn_saved_at": _jst(mtime),
        "lssn_elapsed_seconds_at_save": str(int(mtime - started)) if started else "",
        "lssn_epoch": str(epoch),
        "lssn_total_epochs": str(ctx.get("total_epochs") or cfg.get("epochs") or ""),
        "lssn_step": str(step),
        "lssn_total_steps": _s(total_steps),
        "lssn_steps_per_epoch": str(ctx.get("steps_per_epoch") or ""),
        "lssn_model_family": _s(cfg.get("model_family")),
        "lssn_backend": _s(resolved.get("resolved_training_backend")),
        "lssn_learning_rate": _s(cfg.get("learning_rate")),
        "lssn_optimizer": _s(cfg.get("optimizer")),
        "lssn_scheduler": _s(cfg.get("scheduler")),
        "lssn_batch_size": _s(cfg.get("train_batch_size")),
        "lssn_resolution": _s(cfg.get("resolution")),
        "lssn_repeats": _s(cfg.get("repeats")),
        "lssn_network_rank": _s(cfg.get("rank")),
        "lssn_network_alpha": _s(cfg.get("alpha")),
        "lssn_mixed_precision": _s(cfg.get("mixed_precision")),
        "lssn_save_precision": _s(cfg.get("save_precision")),
        "lssn_seed": existing.get("ss_seed", ""),
        "lssn_advanced": _s(cfg.get("advanced") or {}),
        "lssn_base_model_name": Path(base_path).name if base_path else "",
        "lssn_base_model_size": str(base_stat.st_size) if base_stat else "",
        "lssn_base_model_hash": existing.get("ss_new_sd_model_hash") or existing.get("ss_sd_model_hash", ""),
        "lssn_vae_name": Path(str(backend_cfg.get("vae_path") or resolved.get("resolved_vae_path") or "")).name,
        "lssn_text_encoder_name": Path(str(backend_cfg.get("text_encoder_path") or resolved.get("resolved_text_encoder_path") or "")).name,
        "lssn_dataset_snapshot_id": _s(snap.get("id") or ctx.get("snapshot_id")),
        "lssn_dataset_snapshot_hash": _s(snap.get("snapshot_hash") or resolved.get("dataset_snapshot_hash")),
        "lssn_dataset_image_count": _s(snap.get("item_count") or resolved.get("dataset_item_count")),
        "lssn_trigger_words": _s(ctx.get("triggers") or resolved.get("concepts") or []),
        "lssn_instances": _s(ctx.get("instances") or []),
        "lssn_instance_hints": _s(ctx.get("instance_hints") or {}),
        "lssn_gpu": _s(gpu.get("physical_name") or gpu.get("cuda_name")),
        "lssn_gpu_vram_mb": _s(gpu.get("physical_total_mb")),
        "lssn_python": env.get("python", ""),
        "lssn_torch": env.get("torch", ""),
        "lssn_trainer_packages": _s({k: v for k, v in env.items() if k != "python"}),
        "lssn_trainer_script": Path(str(resolved.get("trainer_script_path") or "")).name,
    }
    lssn = {k: v for k, v in lssn.items() if v != "" or k in ("lssn_schema",)}

    ss: dict[str, str] = {}
    def fill(key: str, value: Any) -> None:
        text = _s(value)
        if text != "" and key not in existing:
            ss[key] = text
    fill("ss_epoch", epoch)
    fill("ss_steps", step)
    fill("ss_num_epochs", ctx.get("total_epochs") or cfg.get("epochs"))
    fill("ss_max_train_steps", total_steps)
    fill("ss_learning_rate", cfg.get("learning_rate"))
    fill("ss_optimizer", cfg.get("optimizer"))
    fill("ss_lr_scheduler", cfg.get("scheduler"))
    fill("ss_total_batch_size", cfg.get("train_batch_size"))
    fill("ss_network_dim", cfg.get("rank"))
    fill("ss_network_alpha", cfg.get("alpha"))
    fill("ss_mixed_precision", cfg.get("mixed_precision"))
    fill("ss_output_name", cfg.get("output_name"))
    fill("ss_sd_model_name", Path(base_path).name if base_path else "")
    fill("ss_vae_name", lssn.get("lssn_vae_name"))
    fill("ss_num_train_images", (snap.get("item_count") or 0) * int(cfg.get("repeats") or 1) if snap.get("item_count") else "")
    res = cfg.get("resolution")
    if res:
        fill("ss_resolution", f"({res}, {res})")
    if started:
        fill("ss_training_started_at", started)
    tag_lists = ctx.get("tag_lists") or []
    if tag_lists:
        freq = Counter(t for tags in tag_lists for t in tags)
        label = f"{int(cfg.get('repeats') or 1)}_{ctx.get('project_name') or 'dataset'}"
        fill("ss_tag_frequency", {label: dict(freq.most_common(300))})
    return lssn, ss


_BUILD: str | None = None


def _build_hash() -> str:
    """このモジュール自身のSHA256先頭12桁(gitなしビルド識別子)。"""
    global _BUILD
    if _BUILD is None:
        import hashlib
        try:
            _BUILD = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
        except OSError:
            _BUILD = "unknown"
    return _BUILD


# ------------------------------------------------------------------ public
def stamp_checkpoint(ctx: dict[str, Any], ckpt: str | Path) -> bool:
    """1ファイルへ埋め込む。処理済み(lssn_schemaあり)なら何もしない。変更したら True。"""
    p = Path(ckpt)
    existing = read_metadata(p)
    if "lssn_schema" in existing:
        return False
    lssn, ss = build_checkpoint_metadata(ctx, p, existing)
    return rewrite_metadata(p, {**ss, **lssn})


def stamp_run_outputs(conn, run_id: int, project_id: int, files: list[Path]) -> int:
    """出力フォルダ内の未処理チェックポイントへ埋め込む。失敗は握りつぶしてログのみ(次回スキャンで再試行)。"""
    pending = []
    for f in files:
        try:
            if "lssn_schema" not in read_metadata(f):
                pending.append(f)
        except Exception:  # noqa: BLE001  書き込み途中・非safetensors
            continue
    if not pending:
        return 0
    try:
        ctx = build_run_context(conn, run_id, project_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("lora metadata context failed (run %s): %s", run_id, exc)
        return 0
    done = 0
    for f in pending:
        try:
            if stamp_checkpoint(ctx, f):
                done += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("lora metadata stamp skipped (%s): %s", f.name, exc)
    return done
