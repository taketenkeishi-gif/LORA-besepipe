"""System status API for the GUI shell.

The endpoint intentionally reports each dependency independently.  A failed
project query or optional tool must not collapse the whole header to offline.
"""
from __future__ import annotations

import requests
from fastapi import APIRouter

from ..services.comfyui_client import get_comfyui_url
from .settings import _check_dir, _check_exe, _check_file, _read_paths
from .training import get_resources
from ..training import get_training_backend

router = APIRouter(prefix="/system", tags=["system"])


def _state(check: dict) -> str:
    if check.get("ok"):
        return "ready"
    reason = str(check.get("reason") or "unknown")
    return "not_configured" if reason == "not configured" else "offline"


def _check_comfyui_runtime(configured_root: str) -> dict:
    """Return live ComfyUI reachability without starting or probing a workflow."""
    if not str(configured_root or "").strip():
        return {"state": "not_configured", "detail": "ComfyUI root is not configured"}
    url = get_comfyui_url()
    try:
        response = requests.get(f"{url}/system_stats", timeout=0.75)
        response.raise_for_status()
    except requests.RequestException as exc:
        return {"state": "offline", "detail": f"ComfyUI is not reachable at {url}: {exc}"}
    return {"state": "ready", "detail": f"ComfyUI responding at {url}"}


@router.get("/status")
def system_status() -> dict:
    paths = _read_paths()
    checks = {
        "python": _check_exe(paths.get("python_exe", "")),
        "kohya": _check_dir(paths.get("kohya_root", "")),
        "musubi": _check_dir(paths.get("musubi_root", "")),
        "ai_toolkit": _check_dir(paths.get("ai_toolkit_root", "")),
        "comfyui": _check_dir(paths.get("comfyui_root", "")),
        "wd14": _check_file(paths.get("wd14_script", "")),
    }
    anima_backend = get_training_backend("anima")
    anima_ready = bool(anima_backend and anima_backend.is_ready(paths))
    resources = get_resources()
    services = {
        key: {"state": _state(value), "detail": value.get("reason", "")}
        for key, value in checks.items()
    }
    services["comfyui"] = _check_comfyui_runtime(paths.get("comfyui_root", ""))
    services["anima"] = {
        "state": "ready" if anima_ready else "offline",
        "detail": "Anima専用Backend ready" if anima_ready else "Anima専用スクリプトまたはKohya設定を確認してください",
    }
    anima_gpu = next((device for device in resources.get("gpu", []) if int(device.get("index", -1)) == 1), None)
    anima_free_vram = None
    if anima_gpu is not None:
        # NVML/nvidia-smi expose an authoritative free counter.  Do not derive
        # it from total-used: driver-reserved memory makes that projection
        # optimistic and can disagree with the admission gate by hundreds of MiB.
        measured_free = anima_gpu.get("vram_free_mb")
        anima_free_vram = (
            int(measured_free)
            if measured_free is not None
            else int(anima_gpu["vram_total_mb"]) - int(anima_gpu["vram_used_mb"])
        )
    # Anima/H3の安全な投入条件と同じ20,000 MiBと外部process gateをUIにも公開する。
    # `state=ready` はGPUハードウェアの検出状態、`target_state` はAnima対象GPU1の投入可否。
    occupancy = str(anima_gpu.get("occupancy_status") or "unknown") if anima_gpu else "unknown"
    if anima_free_vram is None or occupancy == "unknown":
        target_state = "unknown"
    elif occupancy == "blocked":
        target_state = "blocked"
    elif anima_free_vram < 20_000:
        target_state = "busy"
    else:
        target_state = "ready"
    return {
        "api": {"state": "ready", "detail": "API responding"},
        "services": services,
        "gpu": {
            "state": "ready" if resources.get("gpu_available") else "unknown",
            "devices": resources.get("gpu", []),
            "target_physical_index": 1,
            "target_device": anima_gpu["name"] if anima_gpu else None,
            "target_free_vram_mb": anima_free_vram,
            "target_min_free_vram_mb": 20_000,
            "target_state": target_state,
        },
    }
