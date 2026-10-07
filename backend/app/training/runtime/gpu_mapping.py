"""Physical NVIDIA-SMI GPU to target-Python CUDA ordinal mapping."""
from __future__ import annotations

import json
import os
import subprocess
from functools import lru_cache
from pathlib import Path


def _nvidia_smi_devices() -> list[dict[str, int | str]]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10, check=True,
    )
    devices: list[dict[str, int | str]] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 3:
            devices.append({"physical_index": int(parts[0]), "name": parts[1], "total_mb": int(parts[2])})
    if not devices:
        raise RuntimeError("nvidia-smiがGPUを返しません")
    return devices


@lru_cache(maxsize=8)
def _cuda_devices(python_exe: str) -> list[dict[str, int | str]]:
    probe = (
        "import json,torch; print(json.dumps([{'cuda_index':i,'name':torch.cuda.get_device_name(i),"
        "'total_mb':torch.cuda.get_device_properties(i).total_memory//(1024*1024)} "
        "for i in range(torch.cuda.device_count())]))"
    )
    # The Anima/Musubi launchers use PCI_BUS_ID before applying
    # CUDA_VISIBLE_DEVICES.  The probe must use the identical ordering;
    # otherwise a machine whose default CUDA enumeration is reversed can
    # report physical GPU1 as CUDA0 and launch on physical GPU0.
    probe_env = {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
    result = subprocess.run(
        [python_exe, "-c", probe], capture_output=True, text=True,
        timeout=45, check=True, env=probe_env,
    )
    value = json.loads(result.stdout.strip())
    if not isinstance(value, list) or not value:
        raise RuntimeError("対象PythonのCUDAデバイスが空です")
    return value


def resolve_cuda_index(physical_index: int, python_exe: str) -> dict[str, object]:
    """Resolve CUDA ordinal for a physical nvidia-smi index.

    Preparation paths may fall back, but execution callers must require
    ``verified`` to prevent a wrong physical GPU from being used.
    """
    try:
        physical = next(row for row in _nvidia_smi_devices() if int(row["physical_index"]) == physical_index)
        cuda_devices = _cuda_devices(str(Path(python_exe)))
        matches = [row for row in cuda_devices if str(row.get("name")) == str(physical["name"])]
        if not matches:
            raise RuntimeError(f"CUDA列挙に対象GPUがありません: {physical['name']}")
        match = min(matches, key=lambda row: abs(int(row.get("total_mb", 0)) - int(physical["total_mb"])))
        return {
            "physical_index": physical_index,
            "cuda_index": int(match["cuda_index"]),
            "physical_name": str(physical["name"]),
            "physical_total_mb": int(physical["total_mb"]),
            "cuda_name": str(match["name"]),
            "cuda_total_mb": int(match["total_mb"]),
            "verified": True,
            "reason": "nvidia-smiと対象Pythonのtorch.cudaを名称・VRAMで照合",
        }
    except Exception as exc:  # noqa: BLE001
        return {"physical_index": physical_index, "cuda_index": physical_index, "physical_name": "", "cuda_name": "", "verified": False, "reason": str(exc)}
