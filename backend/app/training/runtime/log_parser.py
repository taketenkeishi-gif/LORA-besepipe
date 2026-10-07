"""学習ログ（kohya_ss/musubi-tuner いずれも同じ tqdm 形式）のパース。

DB・プロセスに一切依存しない純粋関数群。Backend 間で共通のログ形式を
前提にしているため、将来 Backend が増えてもここは変更不要なことが多い。
"""
from __future__ import annotations

import re
from pathlib import Path

# ── ログ内パターン ──────────────────────────────────────────────────────────
RE_EPOCH = re.compile(r"epoch\s+(\d+)\s*/\s*(\d+)", re.IGNORECASE)
RE_TQDM = re.compile(
    r"(\d+)/(\d+)\s*\[.*?(?:avr_loss|loss)\s*=\s*([0-9.eE+\-]+)", re.IGNORECASE
)
# tqdm の速度フィールド: "2.18s/it" または "1.23it/s"
RE_TQDM_SPEED = re.compile(r",\s*([0-9.]+)(s/it|it/s)", re.IGNORECASE)
RE_STEP_LOSS = re.compile(
    r"#?step\s*(\d+)[:/]\s*(?:train\s+)?loss\s*=\s*([0-9.eE+\-]+)", re.IGNORECASE
)
RE_SAVE = re.compile(r"saving (?:checkpoint|model)", re.IGNORECASE)
RE_SIT = re.compile(r"(\d+\.\d+)s/it")

# チェックポイント/サンプル命名規則
RE_CKPT_EPOCH = re.compile(r"-0*(\d+)\.safetensors$", re.IGNORECASE)
RE_SAMPLE_EPOCH = re.compile(r"_e0*(\d+)_", re.IGNORECASE)
RE_SAMPLE_EPOCH_IDX = re.compile(r"_e0*(\d+)_(\d+)_", re.IGNORECASE)
RE_TRAIN_DONE = re.compile(r"model saved\.|training (?:is )?finished", re.IGNORECASE)

# xformers 等が内部 try/except で握りつぶす optional 依存の欠如。ログには
# Traceback 付きで出力されるがプロセスは継続する（致命的エラーではない）。
BENIGN_LOG_ERROR_SUBSTRINGS = (
    "no module named 'triton'",
    "no module named 'sageattention'",
    "no module named 'flash_attn'",
    "no module named 'apex'",
    "failed to import triton",
    "failed to import sageattention",
)


def parse_train_line(line: str) -> dict:
    """1行から epoch/step/loss を抽出する（tqdm形式・stepログ形式の両対応）。"""
    result: dict = {}
    m = RE_EPOCH.search(line)
    if m:
        result["epoch"] = int(m.group(1))
        result["total_epochs"] = int(m.group(2))
    m = RE_TQDM.search(line)
    if m:
        result["step"] = int(m.group(1))
        result["steps_total"] = int(m.group(2))
        try:
            result["loss"] = float(m.group(3))
        except ValueError:
            pass
    m = RE_STEP_LOSS.search(line)
    if m:
        result["step"] = int(m.group(1))
        try:
            result["loss"] = float(m.group(2))
        except ValueError:
            pass
    return result


def parse_log_state(log_path: str | Path | None) -> dict:
    """ログファイル全体を走査し学習の実態を返す。ワーカーの生死に依存しない。"""
    state: dict = {
        "epoch": None, "total_epochs": None, "gstep": None, "gtotal": None,
        "loss": None, "sit": None, "saving": 0, "done_marker": False, "error": None,
    }
    if not log_path:
        return state
    p = Path(log_path)
    if not p.exists():
        return state
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return state

    err_line: str | None = None
    for line in text.splitlines():
        m = RE_EPOCH.search(line)
        if m:
            state["epoch"] = int(m.group(1))
            state["total_epochs"] = int(m.group(2))
        m = RE_TQDM.search(line)
        if m:
            state["gstep"] = int(m.group(1))
            state["gtotal"] = int(m.group(2))
            try:
                state["loss"] = float(m.group(3))
            except ValueError:
                pass
        m = RE_TQDM_SPEED.search(line)
        if m:
            v = float(m.group(1))
            u = m.group(2).lower()
            state["sit"] = v if u == "s/it" else (1.0 / v if v > 0 else None)
        if RE_SAVE.search(line):
            state["saving"] += 1
        if RE_TRAIN_DONE.search(line):
            state["done_marker"] = True
        low = line.lower()
        # "error:" の緩い部分一致は "ModuleNotFoundError:" のような無害な語にも
        # 誤爆する（例: "founderror:" が "error:" を含む）ため、既知の非致命的
        # パターンは除外する。
        is_benign = any(p in low for p in BENIGN_LOG_ERROR_SUBSTRINGS)
        if not is_benign and (
            low.startswith("[error]") or "error:" in low or "runtimeerror" in low
            or "torch.cuda.outofmemoryerror" in low
        ):
            err_line = line.strip()[:300]
    state["error"] = err_line
    return state
