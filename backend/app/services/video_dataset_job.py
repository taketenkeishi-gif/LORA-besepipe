"""Video -> per-character dataset: job runner (subprocess) and output-folder operations (rename / merge).

The heavy pipeline lives in ``tools/video_dataset/video_to_dataset.py`` and runs with ComfyUI's embedded Python
(its torch / onnxruntime-gpu / ultralytics are what the models need).  This module only starts it, watches it and
edits the folder it produces.  Nothing here touches the GPU or the user's dataset.
"""
from __future__ import annotations

import json
import os
import tempfile
import re
import shutil
import subprocess
import threading
import time
import uuid
from collections import Counter
from pathlib import Path

from ..db import get_conn

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
SCENE_THRESHOLDS = {"fine": 0.15, "normal": 0.3, "coarse": 0.5}
KNOWN_COMFY_ROOT = Path(r"C:\Users\Keishi\Portfolio\Toolchains-Physical\ComfyUI-Portable\ComfyUI_windows_portable\ComfyUI")
JOBS_DIRNAME = ".video-datasets"
RESTART_MESSAGE = "アプリの再起動で中断されました"

_NOISE = ("ON_DETACH", "model_patcher", "ModelPatcher")
_EXC_LINE = re.compile(r"^[\w.]*(Error|Exception)\b")
_OUTFIT_RE = re.compile(r"^outfit_(\d{2})(.*)$")
_RESERVED = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}

_lock = threading.RLock()  # guards _live
_ops_lock = threading.RLock()  # serialises rename / merge / import
_live: dict[str, dict] = {}  # job_id -> {"proc","cancel","dir","log"} for processes started by this app instance


class VideoDatasetError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


# ------------------------------------------------------------------ small helpers
def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None


def write_json(path: Path, data) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def jobs_root_for(dataset_root: Path) -> Path:
    return dataset_root.parent / JOBS_DIRNAME


def script_path() -> Path:
    return Path(__file__).resolve().parents[3] / "tools" / "video_dataset" / "video_to_dataset.py"


def _setting(key: str) -> str:
    conn = get_conn()
    try:
        row = conn.execute("SELECT value FROM app_settings WHERE key=?", (key,)).fetchone()
    finally:
        conn.close()
    return (row["value"] if row else "") or ""


def find_comfy(settings_root: str | None = None) -> tuple[Path, Path]:
    """(python.exe, ComfyUI root): app setting -> env LORA_STUDIO_COMFY_PYTHON -> known install path."""
    root_setting = _setting("comfyui_root") if settings_root is None else settings_root
    candidates: list[tuple[Path, Path]] = []
    if root_setting.strip():
        root = Path(root_setting.strip())
        candidates.append((root.parent / "python_embeded" / "python.exe", root))
    env_python = os.environ.get("LORA_STUDIO_COMFY_PYTHON", "").strip()
    if env_python:
        python = Path(env_python)
        candidates.append((python, python.parent.parent / "ComfyUI"))
    candidates.append((KNOWN_COMFY_ROOT.parent / "python_embeded" / "python.exe", KNOWN_COMFY_ROOT))
    for python, root in candidates:
        if python.is_file() and root.is_dir():
            return python, root
    raise VideoDatasetError(409, "ComfyUI の Python（python_embeded）が見つかりません。設定で ComfyUI のフォルダを指定するか、環境変数 LORA_STUDIO_COMFY_PYTHON に python.exe のパスを設定してください")


def pick_gpu() -> int:
    """The GPU (nvidia-smi / PCI order) with the most free VRAM."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=15).stdout
        best, best_free = 0, -1
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit() and int(parts[1]) > best_free:
                best, best_free = int(parts[0]), int(parts[1])
        return best
    except (OSError, subprocess.SubprocessError):
        return 0


# ------------------------------------------------------------------ parameter validation (Japanese 422 messages)
def validate_params(payload) -> dict:
    p = payload if isinstance(payload, dict) else payload.model_dump()

    def number(name: str, label: str, lo: float, hi: float, integer: bool = False):
        value = p.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise VideoDatasetError(422, f"{label}は数値で指定してください")
        if integer and float(value) != int(value):
            raise VideoDatasetError(422, f"{label}は整数で指定してください")
        if not (lo <= value <= hi):
            raise VideoDatasetError(422, f"{label}は {lo:g} から {hi:g} の間で指定してください")
        return int(value) if integer else float(value)

    sensitivity = p.get("scene_sensitivity")
    if sensitivity not in SCENE_THRESHOLDS:
        raise VideoDatasetError(422, "シーンの検出感度は「細かく・標準・粗く」（fine / normal / coarse）から選んでください")
    others = p.get("others")
    if others not in ("keep", "mask", "exclude"):
        raise VideoDatasetError(422, "他のキャラクターの扱いは keep（残す）/ mask（灰色で隠す）/ exclude（除外）から選んでください")
    path = str(p.get("video_path") or "").strip().strip('"')
    if not path:
        raise VideoDatasetError(422, "動画のパスを指定してください")
    if Path(path).suffix.lower() not in VIDEO_EXTENSIONS:
        raise VideoDatasetError(422, "対応している動画は mp4 / mov / mkv / webm / avi / m4v です")
    long_side = number("long_side", "長辺", 0, 4096, True)
    if 0 < long_side < 256:
        raise VideoDatasetError(422, "長辺は 0（変更しない）か 256 以上で指定してください")
    gpu = p.get("gpu")
    if gpu is not None:
        if isinstance(gpu, bool) or not isinstance(gpu, int) or not (0 <= gpu <= 15):
            raise VideoDatasetError(422, "GPU は 0 から 15 の整数か、自動（未指定）にしてください")
    return {
        "video_path": path, "scene_sensitivity": sensitivity,
        "frame_interval": number("frame_interval", "フレーム間隔（秒）", 0.2, 60),
        "max_frames_per_scene": number("max_frames_per_scene", "1シーンあたりの最大フレーム数", 1, 50, True),
        "others": others,
        "overlap_threshold": number("overlap_threshold", "他の人物とみなす重なり", 0, 1),
        "margin": number("margin", "人物まわりの余白", 0, 1),
        "long_side": long_side,
        "min_area": number("min_area", "人物の最小面積", 0, 1),
        "min_short": number("min_short", "短辺の最小ピクセル", 16, 4000, True),
        "min_char_crops": number("min_char_crops", "キャラとして扱う最小枚数", 1, 1000, True),
        "merge_quantile": number("merge_quantile", "統合のしきい値", 0.5, 0.9999),
        "group_merge_quantile": number("group_merge_quantile", "断片グループの統合しきい値", 0, 0.99),
        "rescue_quantile": number("rescue_quantile", "未仕分けを寄せるしきい値", 0, 0.99),
        "yolo_conf": number("yolo_conf", "人物検出の確度の下限", 0.01, 0.9),
        "cascade_margin": number("cascade_margin", "寄せるときの2番手との差", 0, 5),
        "early_dup_bits": number("early_dup_bits", "ほぼ同じ画像を先に捨てる厳しさ（ビット差）", 0, 16, True),
        "use_names": bool(p.get("use_names", False)),
        "gpu": gpu,
        "limit_seconds": number("limit_seconds", "解析する長さ（秒）", 0, 86400),
    }


def build_command(python: Path, params: dict, out_dir: Path, progress_file: Path) -> list[str]:
    cmd = [str(python), "-u", str(script_path()), params["video_path"], str(out_dir),
           "--threshold", str(SCENE_THRESHOLDS[params["scene_sensitivity"]]),
           "--frame-interval", str(params["frame_interval"]),
           "--max-frames-per-scene", str(params["max_frames_per_scene"]),
           "--others", params["others"],
           "--overlap-threshold", str(params["overlap_threshold"]),
           "--margin", str(params["margin"]),
           "--long-side", str(params["long_side"]),
           "--min-area", str(params["min_area"]),
           "--min-short", str(params["min_short"]),
           "--min-char-crops", str(params["min_char_crops"]),
           "--merge-quantile", str(params["merge_quantile"]),
           "--group-merge-quantile", str(params["group_merge_quantile"]),
           "--rescue-quantile", str(params["rescue_quantile"]),
           "--yolo-conf", str(params["yolo_conf"]),
           "--cascade-margin", str(params["cascade_margin"]),
           "--early-dup-bits", str(params["early_dup_bits"]),
           "--limit-seconds", str(params["limit_seconds"]),
           "--progress-file", str(progress_file)]
    if params["use_names"]:
        cmd.append("--use-names")
    return cmd


# ------------------------------------------------------------------ process control
def kill_tree(proc) -> None:
    if proc is None or proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, timeout=30)
    else:
        proc.kill()


MIN_COMMIT_HEADROOM_MB = 14000


def commit_headroom_mb() -> int | None:
    """Windows commit charge still available (physical + page file). A job that starts while this is nearly used up dies with MemoryError halfway."""
    if os.name != "nt":
        return None
    try:
        import ctypes

        class _Status(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        st = _Status(); st.dwLength = ctypes.sizeof(_Status)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
        return int(st.ullAvailPageFile // (1024 * 1024))
    except Exception:  # noqa: BLE001
        return None


def any_running(gpu: int | None = None) -> bool:
    """Is a job running (on this GPU, when given)? One job per GPU: two GPUs can analyse two videos at the same time."""
    with _lock:
        return any(j["proc"].poll() is None and (gpu is None or j.get("gpu") == gpu) for j in _live.values())


def local_work_dir(job_id: str) -> Path:
    """Scratch frames live OUTSIDE the (often OneDrive-synced) dataset tree: sync locks made temp-file deletes fail mid-run."""
    base = Path(os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()) / "LoRA-Studio-Next" / "video-work"
    return base / job_id


def start_job(project_id: int, dataset_root: Path, params: dict) -> str:
    video = Path(params["video_path"])
    if not video.is_file():
        raise VideoDatasetError(404, "動画ファイルが見つかりません。パスを確認してください")
    with _lock:
        headroom = commit_headroom_mb()
        if headroom is not None and headroom < MIN_COMMIT_HEADROOM_MB:
            raise VideoDatasetError(409, f"メモリに余裕がありません（空き {headroom // 1024}GB < 必要 {MIN_COMMIT_HEADROOM_MB // 1024}GB）。ほかの解析やアプリを閉じるか、少し待ってから開始してください")
        wanted_gpu = params["gpu"] if params["gpu"] is not None else pick_gpu()
        if any_running(wanted_gpu):
            raise VideoDatasetError(409, "このGPUでは別の動画解析が実行中です。終わるか中止してから開始してください（別のGPUなら同時に実行できます）")
        python, comfy_root = find_comfy()
        if not script_path().is_file():
            raise VideoDatasetError(500, f"解析スクリプトが見つかりません: {script_path()}")
        resolved = dict(params)
        resolved["gpu"] = wanted_gpu
        job_id = uuid.uuid4().hex[:16]
        job_dir = jobs_root_for(dataset_root) / job_id
        job_dir.mkdir(parents=True, exist_ok=False)
        env = os.environ.copy()
        env.update({"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "GPU": str(resolved["gpu"]), "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1",
                    "COMFY_ROOT": str(comfy_root), "WORK_DIR": str(local_work_dir(job_id))})
        command = build_command(python, resolved, job_dir, job_dir / "progress.json")
        log = open(job_dir / "run.log", "wb")
        flags = (getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(subprocess, "CREATE_NO_WINDOW", 0)) if os.name == "nt" else 0
        try:
            proc = subprocess.Popen(command, cwd=str(job_dir), stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env, creationflags=flags)
        except OSError as exc:
            log.close()
            shutil.rmtree(job_dir, ignore_errors=True)
            raise VideoDatasetError(500, f"解析プロセスを起動できませんでした: {exc}") from exc
        now = time.time()
        meta = {"job_id": job_id, "project_id": project_id, "status": "running", "created_at": now, "started_at": now, "finished_at": None,
                "video": str(video), "params": resolved, "output_dir": str(job_dir), "pid": proc.pid, "error": "", "images_written": None}
        write_json(job_dir / "job.json", meta)
        _live[job_id] = {"proc": proc, "cancel": False, "dir": job_dir, "log": log, "gpu": wanted_gpu}
    threading.Thread(target=_watch, args=(job_id,), daemon=True, name=f"video-dataset-{job_id}").start()
    return job_id


def _watch(job_id: str) -> None:
    with _lock:
        live = _live.get(job_id)
    if live is None:
        return
    code = live["proc"].wait()
    job_dir: Path = live["dir"]
    try:
        live["log"].close()
    except OSError:
        pass
    meta = load_json(job_dir / "job.json") or {}
    report = load_json(job_dir / "report.json")
    if live["cancel"]:
        meta.update(status="cancelled", error="")
    elif isinstance(report, dict) and "images_written" in report:
        meta.update(status="done", error="", images_written=report.get("images_written"))
    else:
        meta.update(status="error", error=_failure_message(job_dir, code))
    meta["finished_at"] = time.time()
    shutil.rmtree(job_dir / "_work", ignore_errors=True)
    shutil.rmtree(local_work_dir(job_dir.name), ignore_errors=True)
    write_json(job_dir / "job.json", meta)


def _failure_message(job_dir: Path, code: int) -> str:
    lines = filtered_log_lines(job_dir / "run.log", 200)
    for line in reversed(lines):
        if line.startswith("[video_dataset] ERROR:"):
            return line[len("[video_dataset] ERROR:"):].strip()[:500]
    for line in reversed(lines):
        if _EXC_LINE.match(line):
            return line[:500]
    return f"解析が異常終了しました（終了コード {code}）。ログを確認してください"


def cancel_job(job_id: str) -> bool:
    with _lock:
        live = _live.get(job_id)
    if live is None or live["proc"].poll() is not None:
        return False
    live["cancel"] = True
    kill_tree(live["proc"])
    return True


def cancel_adopted(job_dir: Path) -> bool:
    """Cancel a job that was started by an earlier instance of the app (restart): kill its process tree by the recorded PID."""
    meta = load_json(job_dir / "job.json")
    if not isinstance(meta, dict) or meta.get("status") != "running":
        return False
    pid = int(meta.get("pid") or 0)
    if not _pid_alive(pid):
        return False
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, timeout=30)
    else:
        os.kill(pid, 9)
    meta.update(status="cancelled", error="", finished_at=time.time())
    write_json(job_dir / "job.json", meta)
    return True


# ------------------------------------------------------------------ status
def filtered_log_lines(log_path: Path, limit: int = 30) -> list[str]:
    try:
        with log_path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 96 * 1024))
            text = f.read().decode("utf-8", errors="replace")
    except OSError:
        return []
    out: list[str] = []
    skipping = False
    for raw in re.split(r"[\r\n]+", text):
        line = raw.rstrip()
        if not line.strip():
            continue
        if line.startswith("Exception ignored in"):
            skipping = True
            continue
        if skipping:
            if line.startswith((" ", "\t", "Traceback")):
                continue
            skipping = False
            if _EXC_LINE.match(line) or any(n in line for n in _NOISE):
                continue
        if any(n in line for n in _NOISE):
            continue
        out.append(line)
    return out[-limit:]


def refresh_meta(job_dir: Path) -> dict | None:
    """job.json with a 'running' entry that has no live process (app restart) reported as an error."""
    meta = load_json(job_dir / "job.json")
    if not isinstance(meta, dict):
        return None
    if meta.get("status") == "running":
        with _lock:
            live = _live.get(meta.get("job_id"))
        if live is None:
            # Started by an earlier instance of the app (restart): a still-running process is adopted, a finished one is judged by its result file.
            if _pid_alive(int(meta.get("pid") or 0)):
                return meta
            report = load_json(job_dir / "report.json")
            if isinstance(report, dict) and "images_written" in report:
                meta.update(status="done", error="", images_written=report.get("images_written"), finished_at=meta.get("finished_at") or time.time())
            else:
                meta.update(status="error", error=RESTART_MESSAGE, finished_at=meta.get("finished_at") or time.time())
            write_json(job_dir / "job.json", meta)
    return meta


def _pid_alive(pid: int) -> bool:
    """True only for a live process that really is the analysis script (a recycled PID must not keep a dead job 'running')."""
    if pid <= 0:
        return False
    try:
        import psutil
        proc = psutil.Process(pid)
        return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE and "video_to_dataset" in " ".join(proc.cmdline())
    except Exception:  # noqa: BLE001 - gone, denied, or psutil missing
        return False


def status_payload(job_dir: Path) -> dict:
    meta = refresh_meta(job_dir)
    if meta is None:
        raise VideoDatasetError(404, "解析ジョブが見つかりません")
    status = meta["status"]
    progress = load_json(job_dir / "progress.json") or {}
    percent = int(progress.get("percent") or 0)
    stage = str(progress.get("stage") or "準備中")
    finished = meta.get("finished_at")
    elapsed = max(0.0, (finished or time.time()) - float(meta.get("started_at") or meta.get("created_at") or time.time()))
    if status == "done":
        percent, stage = 100, "完了"
    elif status == "cancelled":
        stage = "中止しました"
    elif status == "error":
        stage = "エラー"
    eta = None
    if status == "running" and percent >= 3:
        eta = round(elapsed * (100 - percent) / percent, 1)
    elif status == "done":
        eta = 0.0
    return {"job_id": meta["job_id"], "status": status, "stage": stage, "percent": percent, "elapsed_s": round(elapsed, 1), "eta_s": eta,
            "error": meta.get("error") or "", "log_tail": filtered_log_lines(job_dir / "run.log", 30), "params": meta.get("params") or {},
            "video": meta.get("video") or "", "created_at": meta.get("created_at"), "output_dir": str(job_dir),
            "report": compose_report(job_dir) if status == "done" else None}


def list_jobs(dataset_root: Path, project_id: int, limit: int = 30) -> list[dict]:
    root = jobs_root_for(dataset_root)
    rows = []
    if root.is_dir():
        for d in root.iterdir():
            if not d.is_dir():
                continue
            meta = refresh_meta(d)
            if not meta or meta.get("project_id") != project_id:
                continue
            written = meta.get("images_written")
            if written is None and meta.get("status") == "done":
                written = (load_json(d / "report.json") or {}).get("images_written")
            rows.append({"job_id": meta["job_id"], "status": meta["status"], "created_at": meta.get("created_at"), "video": meta.get("video") or "", "images_written": written})
    rows.sort(key=lambda r: r.get("created_at") or 0, reverse=True)
    return rows[:limit]


# ------------------------------------------------------------------ report
def load_report(job_dir: Path) -> dict:
    report = load_json(job_dir / "report.json")
    if not isinstance(report, dict) or not isinstance(report.get("characters"), list):
        raise VideoDatasetError(409, "結果のレポートがありません（解析が完了していません）")
    return report


def _images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def _spread(items: list, count: int) -> list:
    if len(items) <= count:
        return list(items)
    return [items[round(i * (len(items) - 1) / (count - 1))] for i in range(count)]


def compose_report(job_dir: Path) -> dict | None:
    base = load_json(job_dir / "report.json")
    if not isinstance(base, dict):
        return None
    report = dict(base)
    characters = []
    for ch in base.get("characters", []):
        files = _images(job_dir / ch["folder"])
        item = dict(ch)
        item["images_total"] = len(files)
        item["sample_files"] = [p.relative_to(job_dir).as_posix() for p in _spread(files, 12)]
        item["all_files"] = [p.relative_to(job_dir).as_posix() for p in files]
        outfits = []
        for o in ch.get("outfits", []):
            entry = dict(o)
            entry["images"] = len(_images(job_dir / ch["folder"] / o["folder"])) if (job_dir / ch["folder"] / o["folder"]).is_dir() else o.get("images", 0)
            outfits.append(entry)
        item["outfits"] = outfits
        characters.append(item)
    report["characters"] = characters
    unassigned = [d for d in job_dir.iterdir() if d.is_dir() and d.name.startswith("_仕分け不能")]
    unassigned_files = _images(unassigned[0]) if unassigned else []
    report["unassigned_sample_files"] = [p.relative_to(job_dir).as_posix() for p in _spread(unassigned_files, 24)]
    report["unassigned_files"] = [p.relative_to(job_dir).as_posix() for p in unassigned_files]
    report["unassigned_images"] = len(unassigned_files)
    rejected = [d for d in job_dir.iterdir() if d.is_dir() and d.name.startswith("_画質フィルタ")]
    report["rejected_frame_examples"] = [p.relative_to(job_dir).as_posix() for p in _spread(_images(rejected[0]) if rejected else [], 12)]
    return report


# ------------------------------------------------------------------ folder operations
def safe_folder_name(name: str) -> str:
    name = (name or "").strip()
    if not name:
        raise VideoDatasetError(422, "名前を入力してください")
    if len(name) > 80:
        raise VideoDatasetError(422, "名前は80文字以内にしてください")
    if re.search(r'[<>:"/\\|?*\x00-\x1f]', name):
        raise VideoDatasetError(422, '名前に使えない文字（ < > : " / \\ | ? * ）が含まれています')
    if name.startswith((".", "_")) or name.endswith((".", " ")):
        raise VideoDatasetError(422, "名前の先頭に「.」「_」は使えず、末尾に「.」や空白も使えません")
    if name.split(".")[0].lower() in _RESERVED:
        raise VideoDatasetError(422, "Windows の予約名は使えません")
    return name


def _character(report: dict, folder: str) -> dict:
    for ch in report["characters"]:
        if ch["folder"] == folder:
            return ch
    raise VideoDatasetError(404, f"キャラクターのフォルダが見つかりません: {folder}")


def _char_dir(job_dir: Path, folder: str) -> Path:
    if not folder or re.search(r'[\\/:]', folder) or folder in (".", "..") or folder.startswith("_"):
        raise VideoDatasetError(400, "不正なフォルダ名です")
    path = (job_dir / folder).resolve()
    if path.parent != job_dir.resolve() or not path.is_dir():
        raise VideoDatasetError(404, f"フォルダが見つかりません: {folder}")
    return path


def rename_character(job_dir: Path, folder: str, new_name: str) -> dict:
    new_name = safe_folder_name(new_name)
    with _ops_lock:
        report = load_report(job_dir)
        ch = _character(report, folder)
        src = _char_dir(job_dir, folder)
        if new_name == folder:
            return compose_report(job_dir)
        for other in job_dir.iterdir():
            if other.name.lower() == new_name.lower() and other.name.lower() != folder.lower():
                raise VideoDatasetError(409, "同じ名前のフォルダが既にあります")
        dst = job_dir / new_name
        os.rename(src, dst)
        original = (job_dir / "report.json").read_bytes()
        try:
            ch["folder"] = new_name
            write_json(job_dir / "report.json", report)
        except OSError as exc:
            os.rename(dst, src)
            (job_dir / "report.json").write_bytes(original)
            raise VideoDatasetError(500, f"レポートを更新できませんでした: {exc}") from exc
        sheet = job_dir / "_contact_sheets" / f"{folder}.jpg"
        if sheet.is_file():
            try:
                os.replace(sheet, sheet.with_name(f"{new_name}.jpg"))
            except OSError:
                pass
    return compose_report(job_dir)


def rewrite_caption(text: str, old_char: str, new_char: str, old_outfit: str | None, new_outfit: str) -> str:
    """Replace the first (character) and second (outfit) trigger tokens; the other tags are kept as they are."""
    tokens = text.split(", ")
    if tokens and tokens[0].strip() == old_char:
        tokens[0] = new_char
    if len(tokens) > 1 and old_outfit and tokens[1].strip() == old_outfit:
        tokens[1] = new_outfit
    return ", ".join(tokens)


def _unique_stem(directory: Path, stem: str, reserved: set[Path]) -> str:
    candidate, n = stem, 1
    while any(((directory / f"{candidate}{ext}") in reserved or (directory / f"{candidate}{ext}").exists()) for ext in (".png", ".jpg", ".jpeg", ".txt")):
        n += 1
        candidate = f"{stem}_m{n}"
    return candidate


def merge_characters(job_dir: Path, target: str, sources: list[str]) -> dict:
    if not sources:
        raise VideoDatasetError(422, "統合元のキャラクターを選んでください")
    if len(set(sources)) != len(sources):
        raise VideoDatasetError(422, "統合元が重複しています")
    if target in sources:
        raise VideoDatasetError(422, "統合先を統合元に含めることはできません")
    with _ops_lock:
        report = load_report(job_dir)
        tgt = _character(report, target)
        tdir = _char_dir(job_dir, target)
        used = set()
        for sub in tdir.iterdir():
            m = _OUTFIT_RE.match(sub.name) if sub.is_dir() else None
            if m:
                used.add(int(m.group(1)))
        for o in tgt.get("outfits", []):
            m = re.search(r"_o(\d+)$", o.get("trigger", ""))
            if m:
                used.add(int(m.group(1)))
        src_chars = []
        for s in sources:
            src_chars.append((_character(report, s), _char_dir(job_dir, s)))

        # ---- validate / plan (nothing is touched yet)
        plan: list[tuple[Path, Path, str | None]] = []  # (src file, dst file, new text for TXT or None to move as is)
        dst_dirs: list[Path] = []
        new_outfits: list[dict] = []
        reserved: set[Path] = set()
        views = Counter(tgt.get("views", {}))
        crops = int(tgt.get("crops", 0))
        for sc, sdir in src_chars:
            by_folder = {o["folder"]: o for o in sc.get("outfits", [])}
            for od in sorted(d for d in sdir.iterdir() if d.is_dir()):
                m = _OUTFIT_RE.match(od.name)
                number = int(m.group(1)) if m else None
                if number is None or number in used or number == 99:
                    n = 1
                    while n in used or n == 99:
                        n += 1
                    if n > 98:
                        raise VideoDatasetError(409, "衣装の数が多すぎて統合できません")
                else:
                    n = number
                used.add(n)
                new_name = f"outfit_{n:02d}" + (m.group(2) if m else f"_{od.name}")
                new_trigger = f"{tgt['trigger']}_o{n:02d}"
                entry = by_folder.get(od.name, {})
                old_trigger = entry.get("trigger") or (f"{sc['trigger']}_o{number:02d}" if number is not None else None)
                dst_dir = tdir / new_name
                if dst_dir.exists():
                    raise VideoDatasetError(409, f"統合先に同名の衣装フォルダがあります: {new_name}")
                dst_dirs.append(dst_dir)
                files = sorted(p for p in od.rglob("*") if p.is_file())
                stems_done: dict[tuple[Path, str], str] = {}
                kept_images = 0
                for f in files:
                    rel_dir = f.parent.relative_to(od)
                    key = (rel_dir, f.stem)
                    if key not in stems_done:
                        stems_done[key] = _unique_stem(dst_dir / rel_dir, f.stem, reserved)
                    dst = dst_dir / rel_dir / f"{stems_done[key]}{f.suffix}"
                    reserved.add(dst)
                    if f.suffix.lower() == ".txt":
                        try:
                            text = f.read_text(encoding="utf-8-sig")
                        except (OSError, UnicodeError) as exc:
                            raise VideoDatasetError(422, f"TXT を読み込めません: {f.name}") from exc
                        plan.append((f, dst, rewrite_caption(text, sc["trigger"], tgt["trigger"], old_trigger, new_trigger)))
                    else:
                        plan.append((f, dst, None))
                        if f.suffix.lower() in IMAGE_EXTENSIONS:
                            kept_images += 1
                new_outfits.append({"folder": new_name, "trigger": new_trigger, "images": kept_images, "dropped_duplicates": int(entry.get("dropped_duplicates", 0)),
                                    "top_clothing": entry.get("top_clothing", [])})
            views.update(sc.get("views", {}))
            crops += int(sc.get("crops", 0))

        # ---- execute with rollback
        report_path = job_dir / "report.json"
        original_report = report_path.read_bytes()
        done: list[tuple[str, Path, Path, bytes | None]] = []
        created: list[Path] = []
        try:
            for d in dst_dirs:
                d.mkdir(parents=True, exist_ok=False)
                created.append(d)
            for src, dst, text in plan:
                dst.parent.mkdir(parents=True, exist_ok=True)
                if text is None:
                    os.replace(src, dst)
                    done.append(("move", src, dst, None))
                else:
                    original = src.read_bytes()
                    dst.write_text(text, encoding="utf-8", newline="")
                    src.unlink()
                    done.append(("txt", src, dst, original))
            tgt["crops"] = crops
            tgt["views"] = dict(views)
            tgt["outfits"] = sorted(list(tgt.get("outfits", [])) + new_outfits, key=lambda o: o["folder"])
            report["characters"] = [c for c in report["characters"] if c["folder"] not in sources]
            write_json(report_path, report)
        except Exception as exc:  # noqa: BLE001 - roll everything back, then report
            for kind, src, dst, original in reversed(done):
                try:
                    if kind == "move":
                        os.replace(dst, src)
                    else:
                        src.write_bytes(original or b"")
                        dst.unlink(missing_ok=True)
                except OSError:
                    pass
            for d in reversed(created):
                shutil.rmtree(d, ignore_errors=True)
            report_path.write_bytes(original_report)
            raise VideoDatasetError(500, f"統合に失敗したため元に戻しました: {exc}") from exc

        for _, sdir in src_chars:
            shutil.rmtree(sdir, ignore_errors=True)
        for s in sources:
            (job_dir / "_contact_sheets" / f"{s}.jpg").unlink(missing_ok=True)
    return compose_report(job_dir)


# ------------------------------------------------------------------ manual placement of unassigned crops
MANUAL_OUTFIT = "outfit_98_手動で追加"


def _unassigned_dir(job_dir: Path) -> Path:
    found = [d for d in job_dir.iterdir() if d.is_dir() and d.name.startswith("_仕分け不能")]
    if not found:
        raise VideoDatasetError(404, "未仕分けの画像はありません")
    return found[0]


def _next_free_trigger(report: dict) -> str:
    used = {str(c.get("trigger", "")).casefold() for c in report["characters"]}
    n = 1
    while f"ch{n:02d}".casefold() in used:
        n += 1
    return f"ch{n:02d}"


def _caption_without_triggers(txt: Path) -> list[str]:
    if not txt.is_file():
        return []
    tags = [t.strip() for t in txt.read_text(encoding="utf-8-sig").split(",") if t.strip()]
    return [t for t in tags if not re.fullmatch(r"ch\d+(_o\d+)?", t)]


def assign_unassigned(job_dir: Path, files: list[str], target: str | None, new_name: str | None) -> dict:
    """Move chosen crops out of the unassigned folder into an existing character (or a brand-new one), keeping PNG+TXT together."""
    if not files:
        raise VideoDatasetError(422, "移す画像を選んでください")
    with _ops_lock:
        report = load_report(job_dir)
        src_dir = _unassigned_dir(job_dir).resolve()
        sources: list[Path] = []
        for rel in dict.fromkeys(files):
            path = (job_dir / rel).resolve()
            if path.parent != src_dir or path.suffix.lower() not in IMAGE_EXTENSIONS or not path.is_file():
                raise VideoDatasetError(400, f"未仕分けの画像ではありません: {rel}")
            sources.append(path)
        if target:
            ch = _character(report, target)
            char_dir = _char_dir(job_dir, target)
        else:
            name = safe_folder_name(new_name or "")
            if any(d.name.lower() == name.lower() for d in job_dir.iterdir()):
                raise VideoDatasetError(409, "同じ名前のフォルダが既にあります")
            ch = {"folder": name, "trigger": _next_free_trigger(report), "crops": 0, "views": {}, "outfits": []}
            char_dir = job_dir / name
            report["characters"].append(ch)
        trig = ch["trigger"]
        otrig = f"{trig}_o98"
        dest = char_dir / MANUAL_OUTFIT
        dest.mkdir(parents=True, exist_ok=True)
        moved: list[tuple[Path, Path, Path | None, Path | None]] = []
        try:
            for src in sources:
                txt = src.with_suffix(".txt")
                dst = dest / src.name
                if dst.exists():
                    dst = dest / f"{src.stem}_m{len(moved)}{src.suffix}"
                tags = _caption_without_triggers(txt)
                shutil.move(str(src), str(dst))
                dst_txt = dst.with_suffix(".txt")
                dst_txt.write_text(", ".join([trig, otrig] + [t for t in tags if t != "solo"]), encoding="utf-8")
                if txt.is_file():
                    txt.unlink()
                moved.append((src, dst, txt, dst_txt))
        except OSError as exc:
            for src, dst, _txt, dst_txt in reversed(moved):
                shutil.move(str(dst), str(src))
                if dst_txt and dst_txt.exists():
                    dst_txt.unlink()
            raise VideoDatasetError(500, f"画像を移せませんでした: {exc}") from exc
        ch["crops"] = int(ch.get("crops", 0)) + len(moved)
        views = dict(ch.get("views") or {})
        for _src, dst, _t, _d in moved:
            m = re.search(r"s\d+_\d+m\d+s_([a-z]+-[a-z-]+)_p", dst.stem)
            if m:
                views[m.group(1)] = views.get(m.group(1), 0) + 1
        ch["views"] = views
        entry = next((o for o in ch.get("outfits", []) if o["folder"] == MANUAL_OUTFIT), None)
        if entry is None:
            entry = {"folder": MANUAL_OUTFIT, "trigger": otrig, "images": 0, "dropped_duplicates": 0, "top_clothing": []}
            ch.setdefault("outfits", []).append(entry)
        entry["images"] = len(_images(dest))
        report["unassigned_images"] = max(0, int(report.get("unassigned_images", 0)) - len(moved))
        report["images_written"] = int(report.get("images_written", 0)) + len(moved)
        write_json(job_dir / "report.json", report)
        return compose_report(job_dir)


def suggest_for_character(job_dir: Path, target: str) -> list[dict]:
    """Rank the unassigned crops by how much their tags (hair, eyes, clothing...) resemble the crops already in ``target``."""
    import math
    from collections import Counter
    report = load_report(job_dir)
    _character(report, target)
    char_dir = _char_dir(job_dir, target)
    rest = _unassigned_dir(job_dir)

    def tag_set(png: Path) -> set[str]:
        return set(_caption_without_triggers(png.with_suffix(".txt"))) - {"solo", "1girl", "1boy", "multiple girls", "looking at viewer", "simple background"}

    members = [tag_set(p) for p in _images(char_dir)]
    candidates = [(p, tag_set(p)) for p in _images(rest)]
    everything = members + [t for _, t in candidates]
    df = Counter(t for tags in everything for t in tags)
    n = max(1, len(everything))
    idf = {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}
    centre = Counter(t for tags in members for t in tags)
    out = []
    for p, tags in candidates:
        if not tags or not members:
            score = 0.0
        else:
            num = sum(idf[t] * centre[t] / len(members) for t in tags)
            den = sum(idf[t] for t in tags) + 1e-9
            score = num / den
        out.append({"file": p.relative_to(job_dir).as_posix(), "score": round(score, 4)})
    out.sort(key=lambda r: -r["score"])
    return out
