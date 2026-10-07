"""学習時間の推定（GPU 検出・キャリブレーション・実測 EMA）。

Backend の train() 実行中に呼ばれ、estimate API（学習開始前の見積もり）と
実測値を橋渡しする。DB（app_settings.train_calib::*, training_runs.sec_per_step_ema）
と GPU 問い合わせ以外の外部依存はない。
"""
from __future__ import annotations

import subprocess
import time
from datetime import datetime
from pathlib import Path

from ...db import get_conn
from .log_parser import RE_SIT, RE_TQDM_SPEED
from .state import STEP_TIMING

# 既知 GPU の「512px / SD1.x / batch1 / 画像1枚あたり」基準 sec/step。
# 実測キャリブレーションが無いときの初期推定に使う（部分一致で照合）。
GPU_BASE_SIT: dict[str, float] = {
    "5090": 0.085, "4090": 0.115, "4080 super": 0.16, "4080": 0.17,
    "4070 ti super": 0.19, "4070 ti": 0.20, "4070 super": 0.23, "4070": 0.26,
    "4060 ti": 0.34, "4060": 0.42,
    "3090 ti": 0.19, "3090": 0.21, "3080 ti": 0.23, "3080": 0.26,
    "3070 ti": 0.31, "3070": 0.33, "3060 ti": 0.36, "3060": 0.46, "3050": 0.70,
    "2080 ti": 0.38, "2080": 0.45, "2070": 0.55, "2060": 0.70,
    "a100": 0.10, "a6000": 0.18, "a5000": 0.24, "a4000": 0.40,
    "l40": 0.14, "l4": 0.40, "t4": 0.95, "v100": 0.30,
    "1080 ti": 0.85, "1070": 1.2, "1660": 1.4, "1650": 1.8,
}
DEFAULT_BASE_SIT = 0.40  # 未知 GPU のフォールバック

# optimizer ごとの相対係数（state 計算量の差）
OPT_FACTOR: dict[str, float] = {
    "AdamW8bit": 1.0, "AdamW": 1.05, "Lion8bit": 0.97, "Lion": 1.0,
    "Adafactor": 1.10, "DAdaptAdam": 1.18, "Prodigy": 1.15,
}

# 学習バッチサイズ既定値（UI未露出のため固定）。start / estimate / 各backend 全経路で統一。
DEFAULT_BATCH = 2


def detect_gpu_name(device_id: int = 0) -> str | None:
    """指定 GPU の名前を返す（pynvml → nvidia-smi フォールバック）。

    device_id: 学習に実際使用する CUDA デバイス番号（gpu_device_id 設定値）。
    """
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        try:
            count = pynvml.nvmlDeviceGetCount()
            if count > 0:
                idx = device_id if 0 <= device_id < count else 0
                handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                n = pynvml.nvmlDeviceGetName(handle)
                return n.decode("utf-8") if isinstance(n, bytes) else str(n)
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        pass
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            lines = [ln.strip() for ln in r.stdout.strip().splitlines() if ln.strip()]
            if lines:
                idx = device_id if 0 <= device_id < len(lines) else 0
                return lines[idx]
    except Exception:
        pass
    return None


def gpu_base_sit(gpu_name: str | None) -> float:
    """GPU 名から基準 sec/step（512/SD1.x/画像1枚）を引く。部分一致。"""
    if not gpu_name:
        return DEFAULT_BASE_SIT
    low = gpu_name.lower()
    for key in sorted(GPU_BASE_SIT, key=len, reverse=True):
        if key in low:
            return GPU_BASE_SIT[key]
    return DEFAULT_BASE_SIT


def calib_key(gpu_name: str | None, model_family: str) -> str:
    g = (gpu_name or "unknown").lower().replace(" ", "_")[:48]
    return f"train_calib::{g}::{model_family}"


def load_calib(gpu_name: str | None, model_family: str, settings: dict[str, str]) -> float | None:
    raw = settings.get(calib_key(gpu_name, model_family))
    if raw:
        try:
            v = float(raw)
            return v if v > 0 else None
        except ValueError:
            return None
    return None


def save_calib(gpu_name: str | None, model_family: str, base_sit: float) -> None:
    """正規化済み基準 sec/step を EMA でキャリブレーション値として保存する。"""
    if base_sit < 0.05 or base_sit > 30:
        return
    key = calib_key(gpu_name, model_family)
    conn = get_conn()
    try:
        row = conn.execute("SELECT value FROM app_settings WHERE key = ?", (key,)).fetchone()
        if row is not None:
            try:
                prev = float(row["value"])
            except (ValueError, TypeError):
                prev = base_sit
            val = prev * 0.7 + base_sit * 0.3
        else:
            val = base_sit
        conn.execute(
            "INSERT OR REPLACE INTO app_settings(key, value, updated_at) "
            "VALUES (?, ?, CURRENT_TIMESTAMP)",
            (key, f"{val:.6f}"),
        )
        conn.commit()
    finally:
        conn.close()


def backfill_calib_from_runs(detect_sdxl_fn) -> None:
    """過去の学習 run の実測 sec_per_step_ema から calibration をシードする。

    各 run を1度だけシードする（marker で冪等化し、再起動による EMA ドリフトを防ぐ）。
    detect_sdxl_fn: base_checkpoint_path から SDXL か判定する関数（sdxl backend から注入）。
    """
    import json

    try:
        conn = get_conn()
        seeded_raw = conn.execute(
            "SELECT value FROM app_settings WHERE key = 'train_calib_seeded_runs'"
        ).fetchone()
        seeded: set[int] = set()
        if seeded_raw and seeded_raw["value"]:
            try:
                seeded = {int(x) for x in str(seeded_raw["value"]).split(",") if x.strip()}
            except ValueError:
                seeded = set()

        rows = conn.execute(
            "SELECT id, project_id, config_json, steps_per_epoch, sec_per_step_ema "
            "FROM training_runs WHERE sec_per_step_ema IS NOT NULL"
        ).fetchall()
        conn.close()
    except Exception:
        return

    newly: list[int] = []
    for row in rows:
        run_id = int(row["id"])
        if run_id in seeded:
            continue
        try:
            ema = float(row["sec_per_step_ema"])
            cfg = json.loads(row["config_json"] or "{}")
            gpu_name = detect_gpu_name(int(cfg.get("gpu_device_id", 0) or 0))
            ckpt = cfg.get("base_checkpoint_path", "") or ""
            is_sdxl = detect_sdxl_fn(ckpt) if ckpt.strip() else False
            model_family = "sdxl" if is_sdxl else "sd"
            eff_res = int(cfg.get("resolution", 512))
            batch = cfg.get("train_batch_size")
            if not batch:
                images = 0
                try:
                    c2 = get_conn()
                    images = int(c2.execute(
                        "SELECT COUNT(*) FROM dataset_items WHERE project_id=? AND selected=1",
                        (int(row["project_id"]),),
                    ).fetchone()[0])
                    c2.close()
                except Exception:
                    images = 0
                repeats = int(cfg.get("repeats", 1)) or 1
                spe = int(row["steps_per_epoch"] or 0)
                batch = max(1, round(images * repeats / spe)) if spe > 0 and images > 0 else 1
            batch = max(1, int(batch))
            model_factor = 2.8 if is_sdxl else 1.0
            opt_factor = OPT_FACTOR.get(cfg.get("optimizer", "AdamW8bit"), 1.0)
            sec_factor = max(0.001, (batch ** 0.6) * (eff_res / 512.0) ** 2 * model_factor * opt_factor)
            base_sit = ema / sec_factor
            save_calib(gpu_name, model_family, base_sit)
            newly.append(run_id)
        except Exception:
            continue

    if newly:
        try:
            merged = sorted(seeded | set(newly))
            conn = get_conn()
            conn.execute(
                "INSERT OR REPLACE INTO app_settings(key, value, updated_at) "
                "VALUES('train_calib_seeded_runs', ?, CURRENT_TIMESTAMP)",
                (",".join(str(i) for i in merged),),
            )
            conn.commit()
            conn.close()
        except Exception:
            pass


def record_step_timing(project_id: int, step: int, sec_factor: float,
                        gpu_name: str | None, model_family: str,
                        run_id: int | None = None,
                        tqdm_line: str = "") -> None:
    """tqdm行の s/it 値から実測 sec/step を取得してEMAで平滑化する。
    タイムスタンプ差分は使わない（tqdmが同一stepを複数行出力するため不正確）。"""
    tqdm_sps: float | None = None
    m_spd = RE_TQDM_SPEED.search(tqdm_line)
    if m_spd:
        val = float(m_spd.group(1))
        unit = m_spd.group(2).lower()
        tqdm_sps = val if unit == "s/it" else (1.0 / val if val > 0 else None)

    if tqdm_sps is None or not (0.05 < tqdm_sps < 60):
        return

    rec = STEP_TIMING.get(project_id) or {}
    ema = rec.get("ema")
    new_ema = tqdm_sps if ema is None else ema * 0.85 + tqdm_sps * 0.15
    rec["ema"] = new_ema
    rec["last_step"] = step
    rec["last_t"] = time.time()
    STEP_TIMING[project_id] = rec

    if sec_factor > 0:
        save_calib(gpu_name, model_family, tqdm_sps / sec_factor)

    if run_id is not None:
        try:
            conn = get_conn()
            conn.execute(
                "UPDATE training_runs SET sec_per_step_ema=? WHERE id=?",
                (new_ema, run_id),
            )
            if step <= 2:
                conn.execute(
                    "UPDATE training_runs SET step1_at=? WHERE id=? AND step1_at IS NULL",
                    (datetime.utcnow().isoformat(), run_id),
                )
            conn.commit()
            conn.close()
        except Exception:
            pass


def live_sec_per_step(project_id: int) -> float | None:
    """直近 120 秒以内に実測がある学習中プロジェクトの sec/step を返す。"""
    rec = STEP_TIMING.get(project_id)
    if rec and rec.get("ema") and (time.time() - rec.get("last_t", 0)) < 120:
        return float(rec["ema"])
    return None


def db_sec_per_step(run_id: int) -> float | None:
    """DB 永続化 EMA → ログファイル解析 の順でフォールバックし sec/step を返す。"""
    try:
        conn = get_conn()
        row = conn.execute(
            "SELECT sec_per_step_ema, log_path FROM training_runs WHERE id=?", (run_id,)
        ).fetchone()
        conn.close()
        if row and row["sec_per_step_ema"]:
            return float(row["sec_per_step_ema"])
        log_path = row["log_path"] if row else None
        if log_path and Path(log_path).exists():
            with open(log_path, "rb") as f:
                f.seek(max(0, f.seek(0, 2) - 4096), 0)
                tail = f.read().decode("utf-8", errors="replace")
            matches = RE_SIT.findall(tail)
            if matches:
                sps = float(matches[-1])
                try:
                    c2 = get_conn()
                    c2.execute("UPDATE training_runs SET sec_per_step_ema=? WHERE id=?", (sps, run_id))
                    c2.commit()
                    c2.close()
                except Exception:
                    pass
                return sps
    except Exception:
        pass
    return None


def best_sec_per_step(project_id: int, run_id: int | None) -> float | None:
    """学習中 run の sec/step を単一の優先順位で返す（status と estimate を一致させる）。
    優先: in-memory EMA(直近実測) → DB永続EMA(再起動耐性) → ログ末尾の s/it。"""
    v = live_sec_per_step(project_id)
    if v:
        return v
    if run_id is not None:
        v = db_sec_per_step(run_id)
        if v:
            return v
    return None


def sec_factor_for_run(cfg: dict, detect_sdxl_fn) -> tuple[float, str | None, str]:
    """run の config からキャリブレーション用 sec_factor / gpu / model_family を算出する。"""
    base_ckpt = cfg.get("base_checkpoint_path", "") or ""
    is_sdxl = detect_sdxl_fn(base_ckpt) if base_ckpt.strip() else False
    model_family = "sdxl" if is_sdxl else "sd"
    eff_res = int(cfg.get("resolution", 512))
    batch = int(cfg.get("train_batch_size", 2) or 2)
    opt_factor = OPT_FACTOR.get(cfg.get("optimizer", "AdamW8bit"), 1.0)
    model_factor = 2.8 if is_sdxl else 1.0
    sec_factor = max(0.001, (batch ** 0.6) * (eff_res / 512.0) ** 2 * model_factor * opt_factor)
    gpu_device_id = int(cfg.get("gpu_device_id", 0) or 0)
    return sec_factor, detect_gpu_name(gpu_device_id), model_family
