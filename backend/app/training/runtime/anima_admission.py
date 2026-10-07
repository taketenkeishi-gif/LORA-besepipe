"""Shared preflight, queue and final-spawn admission for Anima."""
import json,subprocess,time
from ...db import get_conn
ANIMA_MIN_FREE_VRAM_MB=20_000

def _get_app_settings():
    conn=get_conn()
    try:return {r['key']:r['value'] for r in conn.execute('SELECT key,value FROM app_settings')}
    finally:conn.close()

def required_free_mb(mode: str = "standard") -> int:
    # Conservative first-run admission for 26-block swapping, not a claim of peak usage.
    # Prior 256px rank4 offloaded smoke measured +1313MiB; larger settings remain bounded by the worker cap.
    return 8192 if mode in ("low_vram", "balanced") else 12288

def query_occupancy(physical_index: int) -> dict:
    """Resolve compute apps by physical GPU UUID for the Anima admission gate.

    VRAM alone is insufficient: an interactive DCC can leave enough free VRAM
    while still owning CUDA resources.  Every compute app on the target GPU
    is treated as foreign at admission time; process-name markers are unsafe
    because ComfyUI, Python tools, or unknown CUDA clients can still race a
    training start.
    """
    result: dict = {"physical_index": physical_index, "compute_processes": [], "foreign_processes": [], "query_ok": False}
    try:
        gpu = subprocess.run(
            [
                "nvidia-smi", f"--id={physical_index}",
                "--query-gpu=uuid", "--format=csv,noheader,nounits",
            ], capture_output=True, text=True, timeout=5, check=False,
        )
        gpu_uuid = gpu.stdout.strip().splitlines()[0].strip() if gpu.returncode == 0 and gpu.stdout.strip() else ""
        if not gpu_uuid:
            return result
        apps = subprocess.run(
            [
                "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name",
                "--format=csv,noheader,nounits",
            ], capture_output=True, text=True, timeout=5, check=False,
        )
        if apps.returncode != 0:
            return result
        for line in apps.stdout.splitlines():
            parts = [part.strip() for part in line.split(",", 2)]
            if len(parts) != 3 or parts[0] != gpu_uuid:
                continue
            try:
                pid = int(parts[1])
            except ValueError:
                continue
            process = {"pid": pid, "process_name": parts[2]}
            result["compute_processes"].append(process)
            result["foreign_processes"].append(process)
        # A WDDM client entry is not evidence of active CUDA computation.
        # Sample actual GPU work and inspect the bound ComfyUI queue without modifying other apps.
        import urllib.request
        loads = []
        for _ in range(3):
            probe = subprocess.run(["nvidia-smi",f"--id={physical_index}","--query-gpu=utilization.gpu","--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=5,check=False)
            if probe.returncode != 0:
                return result
            loads.append(int(probe.stdout.strip()))
            time.sleep(0.2)
        result["utilization_samples"] = loads
        settings = _get_app_settings()
        url = str(settings.get("comfyui_url") or "http://127.0.0.1:8188").rstrip('/')
        comfy_busy = False
        if any('comfy' in str(p['process_name']).lower() for p in result['compute_processes']):
            try:
                with urllib.request.urlopen(url+'/queue',timeout=2) as response:
                    queue = json.load(response)
                comfy_busy = bool(queue.get('queue_running') or queue.get('queue_pending'))
            except Exception:
                result['reason'] = 'ComfyUIの実行キューを確認できません'
                return result
        result['comfy_busy'] = comfy_busy
        import psutil
        trainers=[]
        for item in result['compute_processes']:
            try:
                command=' '.join(psutil.Process(item['pid']).cmdline()).lower()
                if any(marker in command for marker in ('train_network.py','bounded_training_entry.py','musubi-tuner')):
                    trainers.append(item)
            except psutil.NoSuchProcess:pass
            except psutil.AccessDenied:
                result['reason']='GPUプロセスの実行内容を確認できません';return result
        result['active_trainers']=trainers
        if max(loads) <= 10 and not comfy_busy and not trainers:
            result['idle_resident_processes'] = result['foreign_processes']
            result['foreign_processes'] = []
        result["query_ok"] = True
        return result
    except (OSError, ValueError, subprocess.SubprocessError):
        return result


def wait_reason(occupancy: dict, allow_interactive_sharing: bool = False) -> str | None:
    if not occupancy.get('query_ok'):
        return str(occupancy.get('reason') or 'GPUの計算負荷を確認できません。再確認します。')
    trainers=occupancy.get('active_trainers') or []
    if trainers:
        return '別の学習が準備・実行中です（PID '+', '.join(str(p['pid']) for p in trainers)+'）。終了後に再評価します。'
    if occupancy.get('comfy_busy'):
        return 'ComfyUIに生成中・生成待ちの処理があります。終了後に再評価します。'
    if allow_interactive_sharing:
        # Explicit per-Run consent covers the observed game/desktop apps, not
        # other trainers or queued Comfy generation (checked above).
        from pathlib import PureWindowsPath
        interactive={'monsterhunterwilds.exe','chatgpt.exe','clipstudiopaint.exe','dwm.exe','explorer.exe'}
        unknown=[p for p in occupancy.get('foreign_processes',[]) if PureWindowsPath(p.get('process_name','')).name.lower() not in interactive and 'comfy' not in p.get('process_name','').lower()]
        if unknown:
            return '並行実行の対象外のGPUプロセスがあります：'+', '.join(PureWindowsPath(p.get('process_name','')).name for p in unknown)
        return None
    load=max(occupancy.get('utilization_samples') or [0])
    if load>10:
        return f'GPUの計算負荷が高い状態です（直近の最大 {load}%）。落ち着くまで待機します。'
    if occupancy.get('foreign_processes'):
        return 'GPUの使用状況を再確認しています。'
    return None

