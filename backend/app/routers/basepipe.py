from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
import subprocess
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image, ImageDraw, ImageFont

from ..db import get_conn
from ..schemas import BasepipeAestheticApprovalIn, BasepipeAssetBulkReviewIn, BasepipeAssetCreate, BasepipeAssetReviewIn, BasepipeAssetTransformIn, BasepipeAssetGroupCreate, BasepipeAssetGroupMembersIn, BasepipeCaptionBulkOperationIn, BasepipeCaptionBulkPreviewIn, BasepipeCaptionLineageUpdate, BasepipeConceptCreate, BasepipeDatasetBulkReviewIn, BasepipeEvaluationIn, BasepipeSnapshotCreate, BasepipeTrainingInputChoiceIn, CharacterGenerationPreflightIn, CharacterGenerationStartIn, CharacterGenerationStageIn, CharacterGenerationFrameExtractionIn, CharacterGenerationEnhancementIn, CharacterGenerationRunAestheticReviewIn, CharacterGenerationIdentityReviewIn, CharacterGenerationQwenCorrectionIn, CharacterGenerationCaptionIn
from ..training.runtime import IMAGE_EXTS
from ..training.runtime.gpu_mapping import resolve_cuda_index

router = APIRouter(prefix="/basepipe", tags=["basepipe"])

# Canonical H3 runtime: a project-owned ComfyUI instance on port 8189.  Port
# 8188 belongs to the user's general ComfyUI session and must never be reused,
# stopped, or reconfigured by a Character Factory Run.
H3_WORKFLOW_ROOT = Path(r"C:\Users\Keishi\AI_tools\ComfyUI-Portable\ComfyUI_windows_portable\ComfyUI\user\default\workflows")
H3_COMFYUI_PORT = 8189
H3_COMFYUI_URL = f"http://127.0.0.1:{H3_COMFYUI_PORT}"
GENERAL_COMFYUI_URL = "http://127.0.0.1:8188"
# Keep the API gate aligned with the dedicated runtime so an explicit
# execute=true request cannot bypass physical-GPU admission.
H3_MIN_FREE_VRAM_MB = 20_000
# Qwen-Image-Edit uses ComfyUI's dynamic VRAM loader. Runtime evidence records
# the largest indivisible text-encoder stage at 7,910 MiB while the 19,548 MiB
# diffusion model is patch-loaded/offloaded. Keep 590 MiB of headroom above
# that hard stage instead of reusing H3's full-load floor.
QWEN_MIN_FREE_VRAM_MB = 8_500
# WDDM keeps idle GUI/Python clients in nvidia-smi's compute-app list even
# after their model allocations have been released.  Treat current activity,
# not mere process residency, as the admission signal.  The aggregate free
# VRAM threshold remains the hard capacity guard.
H3_MAX_IDLE_UTILIZATION_PERCENT = 5
H3_WAIT_POLL_SECONDS = 10.0
QWEN_WAIT_POLL_SECONDS = 10.0
ENHANCEMENT_WAIT_POLL_SECONDS = 10.0
SELECTIVE_HQ_PROFILE = "selected-frame-qwen-identity-repair-v1"
QWEN_DATASET_REGEN_PROFILE = "four-view-qwen-dataset-regeneration-v1"
QWEN_CONSISTENCY_PROFILE = "qwen-2511-multiple-angles-adjacent-dual-anchor-v2"
QWEN_MULTIPLE_ANGLES_LORA = "qwen-image-edit-2511-multiple-angles-lora.safetensors"
FLASHVSR_DATASET_PROFILE = {
    "profile_id": "flashvsr-v1.1-official-4x-dataset-v1",
    "model": "FlashVSR-v1.1",
    "mode": "full",
    "vae": "Wan2.1",
    "scale": 4,
    "precision": "bf16",
    "attention_mode": "sparse_sage_attention",
    "steps": 1,
    "cfg": 1.0,
    "sparse_ratio": 2.0,
    "kv_ratio": 3.0,
    "local_range_candidates": [9, 11],
    "color_fix": "wavelet",
    "tiled_vae": False,
    "tiled_dit": False,
    "resize_factor": 1.0,
    "frame_window_rule": "8n+1 contiguous frames around each selected yaw",
    "output_rule": "retain native 4x output and derive the training-size image with a recorded high-quality downsample",
    "required_runtime_proof": [
        "v1.1 weights loaded",
        "LCSA or equivalent locality-constrained sparse attention active",
        "input and output dimensions recorded",
        "no post-VSR resize back to the source dimensions",
    ],
}
REALESRGAN_DATASET_BASELINE = {
    "profile_id": "realesrgan-x2plus-frame-baseline-v1",
    "model": "RealESRGAN_x2plus.pth",
    "scale": 2,
    "temporal_context": False,
}
AESTHETIC_UNREVIEWED = "AESTHETIC_UNREVIEWED"
USER_APPROVED = "USER_APPROVED"
USER_REJECTED = "USER_REJECTED"
CHARACTER_GENERATION_STAGES = (
    "planning", "h3_generation", "frame_extraction", "blur_filter", "duplicate_filter",
    "technical_selection", "enhancement", "identity_review", "qwen_correction", "balancing", "captioning", "snapshot",
)


def _require_user_aesthetic_confirmation(confirmed: bool) -> None:
    if not confirmed:
        raise HTTPException(
            status_code=400,
            detail="審美判断はユーザー本人の明示確認が必要です。自動処理・指標・Agent判断では確定できません",
        )


def _record_training_input_review(metadata: dict, *, choice: str, version_id: int | None, note: str) -> None:
    metadata["training_input_review_state"] = USER_APPROVED
    metadata["training_input_reviewed_by"] = "user"
    metadata["training_input_reviewed_at"] = time.time()
    metadata["training_input_review_note"] = note.strip()
    metadata["training_input_review_choice"] = choice
    metadata["training_input_review_version_id"] = version_id
_CARDINAL_REFERENCE_ROLES = {"front", "left", "back", "right"}
_ALLOWED_REFERENCE_ROLES = _CARDINAL_REFERENCE_ROLES | {"detail"}
_REFERENCE_ROLE_ORDER = {role: index for index, role in enumerate(("front", "left", "back", "right", "detail"))}
_H3_COVERAGE_SHOTS = (
    {
        "prompt": "full-body front to three-quarter to left-side turnaround, neutral standing pose, entire character visible",
        "coverage": {"views": ["front", "three-quarter", "left"], "shots": ["full-body"], "poses": ["neutral-standing"], "expressions": ["neutral"], "backgrounds": ["simple"]},
    },
    {
        "prompt": "full-body right-side to back turnaround, neutral standing pose, clearly reveal rear hair, outfit and accessories",
        "coverage": {"views": ["right", "back"], "shots": ["full-body"], "poses": ["neutral-standing"], "expressions": ["neutral"], "backgrounds": ["simple"]},
    },
    {
        "prompt": "face and bust coverage, front and three-quarter views, neutral expression then subtle smile",
        "coverage": {"views": ["front", "three-quarter"], "shots": ["face", "bust"], "poses": ["upright"], "expressions": ["neutral", "subtle-smile"], "backgrounds": ["simple"]},
    },
    {
        "prompt": "half-body and full-body pose variation, arms separated from torso, clean silhouette",
        "coverage": {"views": ["three-quarter"], "shots": ["half-body", "full-body"], "poses": ["arms-separated"], "expressions": ["neutral"], "backgrounds": ["simple"]},
    },
    {
        "prompt": "dynamic but anatomically clear standing pose, looking back, preserve every identity feature",
        "coverage": {"views": ["back-three-quarter"], "shots": ["full-body"], "poses": ["dynamic-standing", "looking-back"], "expressions": ["neutral"], "backgrounds": ["simple"]},
    },
    {
        "prompt": "seated and low-angle coverage with unobstructed face, hands, footwear and character-specific equipment",
        "coverage": {"views": ["three-quarter", "low-angle"], "shots": ["full-body"], "poses": ["seated"], "expressions": ["neutral"], "backgrounds": ["simple"]},
    },
)
_H3_TURNTABLE_SHOT = {
    "prompt": (
        "a locked-camera studio turntable shot; the character holds one perfectly still neutral standing pose "
        "while the turntable rotates clockwise through one complete 360-degree revolution at constant speed; "
        "no walking, no limb motion, no pose change, no expression change, no camera movement, no zoom, "
        "no cuts, no background change; expose front, left, back, right and every intermediate yaw angle"
    ),
    "coverage": {
        "views": ["front", "front-left", "left", "back-left", "back", "back-right", "right", "front-right"],
        "shots": ["full-body"],
        "poses": ["locked-neutral-standing"],
        "expressions": ["locked-neutral"],
        "backgrounds": ["locked-simple-studio"],
    },
}
_H3_COVERAGE_PRESET_CLIPS = {"turntable": 1, "essential": 2, "balanced": 3, "complete": 6}
_H3_IDENTITY_CONSTRAINTS = {
    "face": "preserve exact facial identity and face shape",
    "hair": "preserve exact hairstyle and hair color",
    "eyes": "preserve exact eye color and eye design",
    "body": "preserve body proportions and silhouette",
    "outfit": "preserve exact outfit construction and markings",
    "accessories": "preserve every character-specific accessory and equipment detail",
    "palette": "preserve the exact character color palette",
}

logger = logging.getLogger(__name__)
_H3_WAIT_MONITOR_LOCK = threading.Lock()
_H3_WAIT_MONITOR_THREAD: threading.Thread | None = None
_QWEN_WAIT_MONITOR_LOCK = threading.Lock()
_QWEN_WAIT_MONITOR_THREAD: threading.Thread | None = None
_ENHANCEMENT_WAIT_MONITOR_LOCK = threading.Lock()
_ENHANCEMENT_WAIT_MONITOR_THREAD: threading.Thread | None = None


def _validate_reference_roles(asset_ids: list[int], reference_roles: dict[str, str]) -> tuple[bool, str]:
    """Validate the visible four-view contract without guessing missing roles."""
    if not reference_roles:
        return True, "Role指定なし（互換入力）"
    roles = [str(reference_roles.get(str(asset_id), "")).strip().lower() for asset_id in asset_ids]
    if any(not role for role in roles):
        return False, "すべてのReferenceにRoleを指定してください"
    invalid = sorted({role for role in roles if role not in _ALLOWED_REFERENCE_ROLES})
    if invalid:
        return False, f"未対応のReference Role: {', '.join(invalid)}"
    cardinal = [role for role in roles if role in _CARDINAL_REFERENCE_ROLES]
    if len(cardinal) != len(set(cardinal)):
        return False, "Front / Left / Back / Right は重複できません"
    if "front" not in cardinal:
        return False, "Identity基準となるFront Referenceが必要です"
    if len(asset_ids) == 4 and set(cardinal) != _CARDINAL_REFERENCE_ROLES:
        missing = ", ".join(sorted(_CARDINAL_REFERENCE_ROLES - set(cardinal)))
        return False, f"4枚構成には Front / Left / Back / Right が必要です（不足: {missing}）"
    return True, " / ".join(role.title() for role in roles)


def _order_reference_assets(assets: list, reference_roles: dict[str, str]) -> list:
    """Order model inputs by semantic view while keeping legacy inputs stable."""
    if not reference_roles:
        return assets
    return sorted(
        assets,
        key=lambda row: (
            _REFERENCE_ROLE_ORDER.get(str(reference_roles.get(str(row["id"]), "")).strip().lower(), len(_REFERENCE_ROLE_ORDER)),
            int(row["id"]),
        ),
    )


def _h3_clip_plans(payload: CharacterGenerationStartIn, reference_count: int, ordered_roles: list[str] | None = None) -> list[dict]:
    """Freeze deterministic, diverse H3 clip graphs inside one immutable Run."""
    from ..services.comfyui_client import (
        ComfyUIClient,
        H3_DATASET_ANCHOR_PROFILE,
        H3_DATASET_ANCHOR_PROMPT_ID,
        H3_DATASET_ANCHOR_WORKFLOW_SHA256,
    )

    identity_constraints = [
        constraint
        for key, constraint in _H3_IDENTITY_CONSTRAINTS.items()
        if bool(payload.identity_lock.get(key))
    ]
    lock_prompt = f", identity constraints: {'; '.join(identity_constraints)}" if identity_constraints else ""
    role_names = {
        "front": "front identity view",
        "left": "left profile view",
        "back": "back identity view",
        "right": "right profile view",
        "detail": "identity detail view",
    }
    picture_map = [
        f"<Picture {index}> is the {role_names.get(role, 'identity reference')}"
        for index, role in enumerate(ordered_roles or ["reference"] * reference_count, start=1)
    ]
    reference_prompt = (
        "all reference pictures show exactly the same character; "
        + "; ".join(picture_map)
        + "; preserve identity consistently across every view"
    )
    plans: list[dict] = []
    effective_clip_count = _H3_COVERAGE_PRESET_CLIPS.get(payload.coverage_preset, payload.clip_count)
    shots = (_H3_TURNTABLE_SHOT,) if payload.coverage_preset == "turntable" else _H3_COVERAGE_SHOTS[:effective_clip_count]
    for clip_index, shot in enumerate(shots, start=1):
        clip_prompt = f"{reference_prompt}, {payload.prompt.strip().rstrip(',')}, {shot['prompt']}, simple background{lock_prompt}".strip(", ")
        clip_seed = int(payload.seed) + (clip_index - 1) * 9_973
        graph = ComfyUIClient.build_h3_reference_to_video_graph(
            reference_image=[f"<uploaded-reference-image-{index}>" for index in range(reference_count)],
            prompt=clip_prompt,
            seed=clip_seed,
            width=payload.width,
            height=payload.height,
            length=payload.length,
            ref_image_size=payload.ref_image_size,
        )
        plans.append({
            "clip_id": f"clip_{clip_index:02d}",
            "clip_index": clip_index,
            "seed": clip_seed,
            "prompt": clip_prompt,
            "coverage_preset": payload.coverage_preset,
            "coverage": shot["coverage"],
            "identity_constraints": identity_constraints,
            "candidate_generation_profile": H3_DATASET_ANCHOR_PROFILE,
            "anchor_provenance": {
                "source_prompt_id": H3_DATASET_ANCHOR_PROMPT_ID,
                "source_workflow_sha256": H3_DATASET_ANCHOR_WORKFLOW_SHA256,
                "derivation": "parameterized_identity_dataset_candidate_graph",
                "aesthetic_state": AESTHETIC_UNREVIEWED,
            },
            "graph": graph,
        })
    return plans


def _runtime_dir() -> Path:
    path = Path(__file__).resolve().parents[3] / ".runtime"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _managed_h3_process_ids() -> set[int]:
    """Return the dedicated H3 ComfyUI PID and every live descendant.

    The PID file is only a hint.  Ownership is accepted after the live command
    line proves that it is ComfyUI main.py on our port and isolated user root.
    """
    pid_file = _runtime_dir() / "comfyui-h3.pid"
    if not pid_file.is_file():
        return set()
    try:
        import psutil

        root = psutil.Process(int(pid_file.read_text(encoding="utf-8").strip()))
        command = [str(part).lower() for part in root.cmdline()]
        expected_user_root = str((_runtime_dir() / "comfyui-h3-user-8189").resolve()).lower()
        has_main = any(Path(part).name.lower() == "main.py" for part in command)
        has_port = any(
            part == "--port" and index + 1 < len(command) and command[index + 1] == str(H3_COMFYUI_PORT)
            for index, part in enumerate(command)
        )
        has_user_root = any(part == expected_user_root for part in command)
        if not (has_main and has_port and has_user_root):
            return set()
        return {root.pid, *(child.pid for child in root.children(recursive=True))}
    except Exception:  # noqa: BLE001 - stale ownership evidence must fail closed
        return set()


def _h3_port_owner_process_ids() -> set[int]:
    """Return listeners for the dedicated H3 ComfyUI port.

    A GPU label alone is not ownership evidence.  A different ComfyUI may use
    the same physical GPU, and submitting to it would interfere with work that
    is outside this Run.  Unknown ownership therefore fails closed.
    """
    try:
        import psutil

        owners: set[int] = set()
        for connection in psutil.net_connections(kind="tcp"):
            local = connection.laddr
            if not local or int(local.port) != H3_COMFYUI_PORT:
                continue
            if connection.status != psutil.CONN_LISTEN or connection.pid is None:
                continue
            owners.add(int(connection.pid))
        return owners
    except Exception:  # noqa: BLE001 - unknown port ownership must fail closed
        return set()


def _query_external_comfyui_queue() -> dict[str, object]:
    """Observe the user's general ComfyUI queue without mutating it.

    Port 8188 is outside this Run's ownership boundary.  The queue probe exists
    only so the GUI can distinguish an active external render from idle model
    residency; it is never used to submit, interrupt, unload, or restart work.
    """
    try:
        import requests

        response = requests.get(f"{GENERAL_COMFYUI_URL}/queue", timeout=1.0)
        response.raise_for_status()
        payload = response.json()
        return {
            "base_url": GENERAL_COMFYUI_URL,
            "reachable": True,
            "running": len(payload.get("queue_running") or []),
            "pending": len(payload.get("queue_pending") or []),
            "read_only": True,
        }
    except Exception as exc:  # noqa: BLE001 - external service state is evidence, not a gate failure
        return {
            "base_url": GENERAL_COMFYUI_URL,
            "reachable": False,
            "running": None,
            "pending": None,
            "read_only": True,
            "error": str(exc),
        }


def _evaluate_h3_gpu_admission(
    *,
    free_mb: int,
    utilization_percent: int,
    foreign_processes: list[dict[str, int | str | bool]],
    external_comfyui: dict[str, object],
    required_free_mb: int = H3_MIN_FREE_VRAM_MB,
) -> dict[str, object]:
    """Classify live GPU activity without confusing idle WDDM clients for work.

    On Windows, ``nvidia-smi --query-compute-apps`` can continue to list an
    application after its model VRAM has been released, and per-process VRAM is
    commonly reported as N/A.  A process-name-only single-owner check therefore
    leaves a queued Run stuck forever.  Admission still fails closed when the
    capacity floor is missed, utilization is active, the external ComfyUI queue
    has work, or an observed external ComfyUI owner cannot be queried.
    """
    blockers: list[str] = []
    if free_mb < required_free_mb:
        blockers.append(f"GPU1 free VRAM {free_mb} MiB < {required_free_mb} MiB")
    if utilization_percent > H3_MAX_IDLE_UTILIZATION_PERCENT:
        blockers.append(
            f"GPU1 utilization {utilization_percent}% > idle limit {H3_MAX_IDLE_UTILIZATION_PERCENT}%"
        )

    foreign_comfyui = [
        item for item in foreign_processes
        if "comfyui" in str(item.get("process_name", "")).lower()
    ]
    queue_reachable = external_comfyui.get("reachable") is True
    queue_running = external_comfyui.get("running")
    queue_pending = external_comfyui.get("pending")
    if queue_reachable:
        running = int(queue_running or 0)
        pending = int(queue_pending or 0)
        if running or pending:
            blockers.append(f"external ComfyUI 8188 queue active: running={running}, pending={pending}")
    elif foreign_comfyui:
        blockers.append("external ComfyUI owner detected but queue state is unknown")

    return {
        "admission_ok": not blockers,
        "activity_blockers": blockers,
        "idle_foreign_processes": foreign_processes if not blockers else [],
        "idle_utilization_limit_percent": H3_MAX_IDLE_UTILIZATION_PERCENT,
        "required_free_mb": required_free_mb,
    }


def _qwen_runtime_external_conflict(memory: dict[str, object]) -> str | None:
    """Find foreign work without reapplying pre-launch limits to managed 8189."""
    external = memory.get("external_comfyui")
    if not isinstance(external, dict):
        external = {}
    if external.get("reachable") is True:
        running = int(external.get("running") or 0)
        pending = int(external.get("pending") or 0)
        if running or pending:
            return f"external ComfyUI 8188 queue active: running={running}, pending={pending}"
        return None
    foreign = memory.get("foreign_processes")
    if isinstance(foreign, list) and any(
        "comfyui" in str(item.get("process_name", "")).lower()
        for item in foreign
        if isinstance(item, dict)
    ):
        return "external ComfyUI owner detected but queue state is unknown"
    return None


def _runtime_exception_detail(exc: Exception) -> str:
    """Persist FastAPI/background exception evidence even when str(exc) is empty."""
    detail = getattr(exc, "detail", None)
    if detail not in (None, ""):
        return f"{type(exc).__name__}: {detail}"
    message = str(exc).strip()
    return f"{type(exc).__name__}: {message or repr(exc)}"


def _query_h3_gpu_memory(*, required_free_mb: int = H3_MIN_FREE_VRAM_MB) -> dict[str, object]:
    """Return live physical GPU1 evidence for a workload-specific admission gate."""
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--id=1",
                "--query-gpu=index,uuid,name,memory.used,memory.free,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        row = next((line.strip() for line in completed.stdout.splitlines() if line.strip()), "")
        index, uuid, name, used, free, total, utilization = [part.strip() for part in row.split(",")]
        managed_pids = _managed_h3_process_ids()

        processes: list[dict[str, int | str | bool]] = []
        managed_processes: list[dict[str, int | str | bool]] = []
        process_probe = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        for process_line in process_probe.stdout.splitlines():
            values = [part.strip() for part in process_line.split(",", 2)]
            if len(values) != 3 or values[0] != uuid or not values[1].isdigit():
                continue
            item = {
                "pid": int(values[1]),
                "process_name": values[2],
                "memory_used_mb": 0,
                "managed": int(values[1]) in managed_pids,
            }
            processes.append(item)
            if item["managed"]:
                managed_processes.append(item)
        # Preserve every observed client for evidence.  The admission evaluator
        # distinguishes idle WDDM residency from active work using aggregate
        # capacity/utilization and the read-only external ComfyUI queue.
        # Project-owned 8189 is still identified by PID + command line.
        foreign = [item for item in processes if not item["managed"]]
        external_comfyui = _query_external_comfyui_queue()
        admission = _evaluate_h3_gpu_admission(
            free_mb=int(free),
            utilization_percent=int(utilization),
            foreign_processes=foreign,
            external_comfyui=external_comfyui,
            required_free_mb=required_free_mb,
        )
        return {
            "physical_index": int(index),
            "name": name,
            "used_mb": int(used),
            "free_mb": int(free),
            "total_mb": int(total),
            "utilization_percent": int(utilization),
            "threshold_mb": required_free_mb,
            "compute_processes": processes,
            "managed_processes": managed_processes,
            "foreign_processes": foreign,
            "external_comfyui": external_comfyui,
            "source": "nvidia-smi --id=1",
            **admission,
        }
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return {
            "physical_index": 1,
            "threshold_mb": required_free_mb,
            "admission_ok": False,
            "external_comfyui": _query_external_comfyui_queue(),
            "source": "nvidia-smi --id=1",
            "error": str(exc),
        }


def _h3_gpu_check(memory: dict[str, object]) -> dict[str, object]:
    required_free_mb = int(memory.get("required_free_mb") or memory.get("threshold_mb") or H3_MIN_FREE_VRAM_MB)
    activity_blockers = memory.get("activity_blockers") or []
    if activity_blockers:
        return {"label": "GPU1占有", "ok": False, "detail": " / ".join(str(item) for item in activity_blockers) + "。H3/Qwenを開始しません。"}
    if memory.get("admission_ok"):
        idle_count = len(memory.get("idle_foreign_processes") or [])
        idle_detail = f" / 待機中の外部クライアント {idle_count}件" if idle_count else ""
        return {
            "label": "GPU VRAM",
            "ok": True,
            "detail": f"GPU1 空き {memory['free_mb']} MiB / 必要 {required_free_mb} MiB / 使用率 {memory.get('utilization_percent', 'unknown')}%{idle_detail}",
        }
    if "free_mb" in memory:
        return {
            "label": "GPU VRAM",
            "ok": False,
            "detail": f"GPU1 空き {memory['free_mb']} MiB / 必要 {required_free_mb} MiB。GPU処理を開始しません。",
        }
    return {"label": "GPU VRAM", "ok": False, "detail": f"GPU1のVRAM実測に失敗: {memory.get('error', 'unknown')}"}


def _query_qwen_gpu_memory() -> dict[str, object]:
    """Use Qwen's measured dynamic-loader floor instead of H3's full-load floor."""
    return _query_h3_gpu_memory(required_free_mb=QWEN_MIN_FREE_VRAM_MB)


def _h3_comfyui_ready() -> tuple[bool, str]:
    """Verify that port 8189 is our managed ComfyUI on physical GPU1."""
    try:
        import requests

        response = requests.get(f"{H3_COMFYUI_URL}/system_stats", timeout=3.0)
        response.raise_for_status()
        owners = _h3_port_owner_process_ids()
        managed = _managed_h3_process_ids()
        if not owners or not owners.issubset(managed):
            return False, f"ComfyUI {H3_COMFYUI_PORT}は外部または所有者不明です (owner PIDs={sorted(owners) or ['unknown']})"
        devices = response.json().get("devices", [])
        labels = [str(device.get("name", "")) for device in devices]
        if not any("RTX 3090 Ti" in label for label in labels):
            return False, f"{H3_COMFYUI_PORT}のGPUが物理GPU1ではありません: {labels or ['device unknown']}"
        object_response = requests.get(f"{H3_COMFYUI_URL}/object_info", timeout=30.0)
        object_response.raise_for_status()
        object_info = object_response.json()
        from ..services.comfyui_client import ComfyUIClient

        probe_graph = ComfyUIClient.build_h3_reference_to_video_graph(
            reference_image=["<front>", "<left>", "<back>", "<right>"],
            prompt="contract probe",
            seed=0,
            width=512,
            height=768,
            length=56,
        )
        graph_contract = ComfyUIClient.validate_graph_against_object_info(probe_graph, object_info)
        if not graph_contract["ok"]:
            return False, f"H3 API graph contract mismatch: {graph_contract}"
        required_models = {
            ("UNETLoader", "unet_name"): probe_graph["model"]["inputs"]["unet_name"],
            ("CLIPLoader", "clip_name"): probe_graph["clip"]["inputs"]["clip_name"],
            ("VAELoader", "vae_name"): {
                probe_graph["vae"]["inputs"]["vae_name"],
                probe_graph["audio_vae"]["inputs"]["vae_name"],
            },
        }
        missing_models: list[str] = []
        for (node_name, field), expected in required_models.items():
            available = set(ComfyUIClient._extract_combo(object_info, node_name, field))
            expected_names = expected if isinstance(expected, set) else {expected}
            missing_models.extend(sorted(str(name) for name in expected_names if name not in available))
        if missing_models:
            return False, f"H3 model contract mismatch: missing {', '.join(missing_models)}"
        return True, f"ComfyUI {H3_COMFYUI_PORT} / physical GPU1 / H3 graph contract ready"
    except Exception as exc:  # noqa: BLE001
        return False, f"ComfyUI {H3_COMFYUI_PORT}未接続: {exc}"


def _h3_cuda_mapping() -> dict[str, object]:
    """Resolve the physical GPU1 ordinal in the exact ComfyUI Python."""
    portable = H3_WORKFLOW_ROOT.parents[3]
    python_exe = portable / "python_embeded" / "python.exe"
    if not python_exe.is_file():
        return {
            "physical_index": 1,
            "cuda_index": 1,
            "physical_name": "",
            "cuda_name": "",
            "verified": False,
            "reason": f"ComfyUI Pythonが見つかりません: {python_exe}",
            "python_executable": str(python_exe),
        }
    mapping = resolve_cuda_index(1, str(python_exe))
    return {**mapping, "python_executable": str(python_exe)}


def _verify_masked_cuda_target(python_exe: Path, mapping: dict[str, object], env: dict[str, str]) -> dict[str, object]:
    """Fail closed unless the masked process sees only the verified GPU1."""
    probe = (
        "import json,torch; p=torch.cuda.get_device_properties(0); "
        "print(json.dumps({'count':torch.cuda.device_count(),'name':torch.cuda.get_device_name(0),"
        "'total_mb':p.total_memory//(1024*1024)}))"
    )
    result = subprocess.run(
        [str(python_exe), "-c", probe],
        capture_output=True,
        text=True,
        timeout=45,
        check=True,
        env=env,
    )
    observed = json.loads(result.stdout.strip())
    expected_name = str(mapping.get("physical_name") or mapping.get("cuda_name") or "")
    expected_total = int(mapping.get("physical_total_mb") or mapping.get("cuda_total_mb") or 0)
    observed_total = int(observed.get("total_mb") or 0)
    if int(observed.get("count") or 0) != 1:
        raise RuntimeError(f"masked CUDA device countが1ではありません: {observed}")
    if expected_name and str(observed.get("name")) != expected_name:
        raise RuntimeError(f"masked CUDA GPU名が物理GPU1と不一致です: expected={expected_name}, observed={observed}")
    if expected_total and abs(observed_total - expected_total) > 256:
        raise RuntimeError(f"masked CUDA VRAMが物理GPU1と不一致です: expected={expected_total}, observed={observed}")
    return observed


def _build_h3_comfyui_launch() -> tuple[list[str], str, dict[str, str], dict[str, object]]:
    """Build a dedicated launch only after target-Python GPU verification."""
    portable = H3_WORKFLOW_ROOT.parents[3]
    comfyui_root = portable / "ComfyUI"
    python_exe = portable / "python_embeded" / "python.exe"
    main_py = comfyui_root / "main.py"
    if not python_exe.is_file() or not main_py.is_file():
        raise RuntimeError(f"ComfyUI runtimeが見つかりません: {main_py}")

    mapping = _h3_cuda_mapping()
    if not mapping.get("verified"):
        raise RuntimeError(f"物理GPU1とComfyUI CUDA列挙を照合できません: {mapping.get('reason', 'unknown')}")

    gate_token_path = comfyui_root / "user" / "instance_gate.token"
    try:
        gate_token = gate_token_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise RuntimeError(f"ComfyUI managed-launch tokenを読めません: {gate_token_path}") from exc
    if not gate_token:
        raise RuntimeError(f"ComfyUI managed-launch tokenが空です: {gate_token_path}")

    runtime_root = _runtime_dir()
    user_root = (runtime_root / "comfyui-h3-user-8189").resolve()
    input_root = (runtime_root / "comfyui-h3-input").resolve()
    output_root = (runtime_root / "comfyui-h3-output").resolve()
    temp_root = (runtime_root / "comfyui-h3-temp").resolve()
    for path in (user_root, input_root, output_root, temp_root):
        path.mkdir(parents=True, exist_ok=True)
    database_path = user_root / "comfyui.db"
    extra_model_paths = (runtime_root / "managed-comfy-extra-model-paths.yaml").resolve()

    command = [
        str(python_exe),
        "-s",
        str(main_py),
        "--windows-standalone-build",
        "--enable-cors-header",
        "--disable-auto-launch",
        "--port",
        str(H3_COMFYUI_PORT),
        "--reserve-vram",
        "1.5",
        "--disable-pinned-memory",
        "--user-directory",
        str(user_root),
        "--database-url",
        f"sqlite:///{database_path.as_posix()}",
        "--input-directory",
        str(input_root),
        "--output-directory",
        str(output_root),
        "--temp-directory",
        str(temp_root),
    ]
    if extra_model_paths.is_file():
        command.extend(["--extra-model-paths-config", str(extra_model_paths)])
    env = {
        **os.environ,
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        # ComfyUI's --cuda-device rewrites CUDA_VISIBLE_DEVICES using the
        # machine-global ordinal and therefore must not be passed here.
        # Mask to the verified target in the parent environment; the child
        # then sees that sole adapter as process-local cuda:0.
        "CUDA_VISIBLE_DEVICES": str(mapping["cuda_index"]),
        "COMFY_INSTANCE_GATE_TOKEN": gate_token,
        "COMFY_SINGLE_PORT_MODE": "1",
    }
    mapping = {**mapping, "masked_probe": _verify_masked_cuda_target(python_exe, mapping, env)}
    return command, str(portable), env, mapping


def _ensure_h3_comfyui() -> tuple[bool, str]:
    """Start the project-owned isolated ComfyUI only after GPU1 admission.

    This function never stops an existing process and never chooses GPU0.  It
    is called only by an explicitly armed waiting Run.
    """
    ready, detail = _h3_comfyui_ready()
    if ready:
        return ready, detail
    port_owners = _h3_port_owner_process_ids()
    if port_owners:
        return False, detail

    log_path = _runtime_dir() / "comfyui-h3.log"
    try:
        command, cwd, env, mapping = _build_h3_comfyui_launch()
        creationflags = getattr(subprocess, "DETACHED_PROCESS", 0x00000008) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
        with log_path.open("a", encoding="utf-8") as log_handle:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=env,
                creationflags=creationflags,
                close_fds=True,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
        (_runtime_dir() / "comfyui-h3.pid").write_text(str(process.pid), encoding="utf-8")
        logger.info(
            "Started managed H3 ComfyUI pid=%s port=%s physical_gpu=1 cuda_index=%s",
            process.pid,
            H3_COMFYUI_PORT,
            mapping["cuda_index"],
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"H3専用ComfyUI起動失敗: {exc}"

    deadline = time.monotonic() + 120.0
    while time.monotonic() < deadline:
        ready, detail = _h3_comfyui_ready()
        if ready:
            return True, detail
        if process.poll() is not None:
            return False, f"H3専用ComfyUIが終了しました (exit={process.returncode}); log={log_path}"
        time.sleep(2.0)
    return False, f"H3 ComfyUI起動タイムアウト; log={log_path}"


def _stop_managed_h3_comfyui(reason: str) -> dict[str, object]:
    """Release only the isolated ComfyUI process owned by this project.

    The PID file alone is never sufficient authority.  The live command line,
    isolated user root, and port ownership must still match the same process
    tree.  An unrelated ComfyUI or any other external process is never stopped.
    """
    pid_file = _runtime_dir() / "comfyui-h3.pid"
    if not pid_file.is_file():
        return {"stopped": False, "reason": reason, "detail": "managed H3 ComfyUI is not running"}
    try:
        root_pid = int(pid_file.read_text(encoding="utf-8").strip())
    except (OSError, ValueError) as exc:
        return {"stopped": False, "reason": reason, "detail": f"invalid managed PID evidence: {exc}"}

    managed = _managed_h3_process_ids()
    owners = _h3_port_owner_process_ids()
    if root_pid not in managed:
        return {"stopped": False, "reason": reason, "detail": "managed PID command-line evidence did not match"}
    if owners and not owners.issubset(managed):
        return {"stopped": False, "reason": reason, "detail": f"port {H3_COMFYUI_PORT} has an external owner; no process was stopped"}

    conn = get_conn()
    try:
        active = _active_gpu_work(conn)
    finally:
        conn.close()
    if active is not None:
        return {"stopped": False, "reason": reason, "detail": f"GPU work still active: {active['kind']} Run #{active['id']}"}

    try:
        import psutil

        root = psutil.Process(root_pid)
        processes = [*root.children(recursive=True), root]
        target_pids = [int(process.pid) for process in processes]
        for process in processes:
            try:
                process.terminate()
            except psutil.NoSuchProcess:
                pass
        _gone, alive = psutil.wait_procs(processes, timeout=10.0)
        for process in alive:
            try:
                process.kill()
            except psutil.NoSuchProcess:
                pass
        if alive:
            _gone_after_kill, alive = psutil.wait_procs(alive, timeout=5.0)
        if alive:
            return {"stopped": False, "reason": reason, "detail": f"managed process did not exit: {[process.pid for process in alive]}"}
        pid_file.unlink(missing_ok=True)
        logger.info("Stopped managed H3 ComfyUI pids=%s reason=%s", target_pids, reason)
        return {"stopped": True, "reason": reason, "pids": target_pids, "detail": "managed H3 ComfyUI released"}
    except Exception as exc:  # noqa: BLE001 - fail closed; never broaden the stop target
        return {"stopped": False, "reason": reason, "detail": f"managed H3 ComfyUI release failed: {exc}"}


def _release_managed_h3_after_wait_race(reason: str, wait_detail: str) -> str:
    """Release our isolated runtime when admission changes after launch."""
    release = _stop_managed_h3_comfyui(reason)
    return f"{wait_detail}; managed 8189 release: {release.get('detail', 'unknown')}"


def _dataset_repair_candidates(project, rows) -> dict[str, str]:
    """Find unique basename matches in configured and repository-local Dataset roots."""
    candidate_paths: dict[str, list[str]] = {}
    roots = [
        Path(str(project["dataset_dir"] or "")),
        Path(str(project["base_dir"] or "")) / "dataset",
        Path(__file__).resolve().parents[3] / "projects" / str(project["name"]) / "dataset",
    ]
    for root in roots:
        if not root.is_dir():
            continue
        for candidate in root.rglob("*"):
            if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTS:
                candidate_paths.setdefault(candidate.name, []).append(str(candidate))
    requested_names = {Path(str(row["file_path"])).name for row in rows}
    return {name: paths[0] for name, paths in candidate_paths.items() if name in requested_names and len(set(paths)) == 1}


def _project_or_404(conn, project_id: int):
    row = conn.execute("SELECT id FROM projects WHERE id = ?", (project_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"project not found: {project_id}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _active_gpu_work(conn) -> dict | None:
    training = conn.execute(
        "SELECT id FROM training_runs WHERE status='training' ORDER BY id LIMIT 1"
    ).fetchone()
    if training is not None:
        return {"kind": "training", "id": int(training["id"])}
    generation = conn.execute(
        "SELECT id FROM basepipe_character_generation_runs "
        "WHERE status IN ('queued','running','running_qwen','running_enhancement') ORDER BY id LIMIT 1"
    ).fetchone()
    if generation is not None:
        return {"kind": "character_generation", "id": int(generation["id"])}
    preview = conn.execute(
        "SELECT id FROM preview_jobs WHERE status='running' ORDER BY id LIMIT 1"
    ).fetchone()
    if preview is not None:
        return {"kind": "preview", "id": int(preview["id"])}
    return None


def _h3_reference_integrity(project_id: int, resolved: dict) -> dict:
    """Verify every frozen H3 reference against its asset record and bytes."""
    asset_ids = [int(asset_id) for asset_id in resolved.get("reference_asset_ids", [])]
    paths = [Path(str(path)) for path in resolved.get("reference_asset_paths", [])]
    frozen_hashes = {str(key): str(value) for key, value in (resolved.get("reference_asset_hashes") or {}).items()}
    if not asset_ids or len(asset_ids) != len(paths):
        return {"ok": False, "verified_count": 0, "expected_count": len(asset_ids), "detail": "Reference IDとpathの対応が壊れています"}
    conn = get_conn()
    try:
        marks = ",".join("?" for _ in asset_ids)
        rows = conn.execute(
            f"SELECT id, file_path, content_sha256 FROM basepipe_assets WHERE project_id = ? AND id IN ({marks})",
            [project_id, *asset_ids],
        ).fetchall()
    finally:
        conn.close()
    by_id = {int(row["id"]): row for row in rows}
    verified = []
    for asset_id, path in zip(asset_ids, paths, strict=True):
        row = by_id.get(asset_id)
        if row is None:
            return {"ok": False, "verified_count": len(verified), "expected_count": len(asset_ids), "detail": f"Reference Asset #{asset_id}が見つかりません"}
        if Path(str(row["file_path"])).resolve() != path.resolve():
            return {"ok": False, "verified_count": len(verified), "expected_count": len(asset_ids), "detail": f"Reference Asset #{asset_id}のpathがRun作成時から変わっています"}
        if not path.is_file():
            return {"ok": False, "verified_count": len(verified), "expected_count": len(asset_ids), "detail": f"Reference Asset #{asset_id}のfileがありません"}
        expected = frozen_hashes.get(str(asset_id)) or str(row["content_sha256"] or "")
        actual = _sha256(path)
        if not expected or actual != expected or str(row["content_sha256"] or "") != expected:
            return {"ok": False, "verified_count": len(verified), "expected_count": len(asset_ids), "detail": f"Reference Asset #{asset_id}のSHA-256が一致しません"}
        verified.append({"asset_id": asset_id, "sha256": actual})
    return {
        "ok": True,
        "verified_count": len(verified),
        "expected_count": len(asset_ids),
        "legacy_hash_baseline": not bool(frozen_hashes),
        "assets": verified,
        "detail": f"{len(verified)}/{len(asset_ids)}件のpathとSHA-256が一致",
    }


def _qwen_master_references(project_id: int, run_id: int) -> list[dict]:
    """Resolve all immutable cardinal identity anchors for Qwen regeneration."""
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT resolved_json FROM basepipe_character_generation_runs WHERE id=? AND project_id=?",
            (run_id, project_id),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        raise HTTPException(status_code=404, detail="Qwen補正元のGeneration Runが見つかりません")
    resolved = json.loads(row["resolved_json"] or "{}")
    integrity = _h3_reference_integrity(project_id, resolved)
    if not integrity.get("ok"):
        raise HTTPException(status_code=409, detail=f"Qwen Master Referenceを照合できません: {integrity['detail']}")
    roles = {str(key): str(value) for key, value in (resolved.get("reference_roles") or {}).items()}
    asset_ids = [int(asset_id) for asset_id in resolved.get("reference_asset_ids", [])]
    paths = [str(path) for path in resolved.get("reference_asset_paths", [])]
    hashes = {str(item["asset_id"]): str(item["sha256"]) for item in integrity.get("assets", [])}
    priority = {"front": 0, "back": 1, "left": 2, "right": 3, "detail": 4}
    references = [
        {"asset_id": asset_id, "role": roles.get(str(asset_id), "detail"), "path": path, "sha256": hashes.get(str(asset_id), "")}
        for asset_id, path in zip(asset_ids, paths, strict=True)
    ]
    references.sort(key=lambda item: (priority.get(str(item["role"]), 99), int(item["asset_id"])))
    return references


def _qwen_identity_reference_pack(run_id: int, references: list[dict]) -> dict:
    """Create a deterministic four-view sheet so every cardinal view reaches Qwen."""
    by_role = {str(reference.get("role")): reference for reference in references}
    roles = ("front", "left", "back", "right")
    missing = [role for role in roles if role not in by_role]
    if missing:
        raise HTTPException(status_code=409, detail=f"Qwen四面Identity Sheetに必要な参照がありません: {', '.join(missing)}")
    pack_dir = _runtime_dir() / "qwen_identity_packs" / f"run_{run_id}"
    pack_dir.mkdir(parents=True, exist_ok=True)
    output = pack_dir / "front_left_back_right.png"
    canvas = Image.new("RGB", (1536, 1536), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=28)
    cells = {
        "front": (0, 0),
        "left": (768, 0),
        "back": (0, 768),
        "right": (768, 768),
    }
    for role in roles:
        source_path = Path(str(by_role[role]["path"]))
        with Image.open(source_path) as opened:
            image = opened.convert("RGB")
            image.thumbnail((704, 676), Image.Resampling.LANCZOS)
        cell_x, cell_y = cells[role]
        x = cell_x + (768 - image.width) // 2
        y = cell_y + 64 + (676 - image.height) // 2
        canvas.paste(image, (x, y))
        draw.text((cell_x + 24, cell_y + 18), role.upper(), fill="black", font=font)
        draw.rectangle((cell_x, cell_y, cell_x + 767, cell_y + 767), outline=(160, 160, 160), width=2)
    canvas.save(output, format="PNG", optimize=True)
    return {
        "path": str(output),
        "sha256": _sha256(output),
        "layout": "front,left/back,right",
        "roles": list(roles),
        "source_assets": [int(by_role[role]["asset_id"]) for role in roles],
    }


def _qwen_regeneration_inputs(run_id: int, assets: list, references: list[dict]) -> dict[int, dict]:
    """Bind each technical yaw guide to its nearest cardinal view and the four-view pack."""
    pack = _qwen_identity_reference_pack(run_id, references)
    cardinal_order = ("front", "left", "back", "right")
    by_role = {str(reference.get("role")): reference for reference in references}
    ordered = sorted(assets, key=lambda asset: int(asset["id"]))
    denominator = max(1, len(ordered) - 1)
    result: dict[int, dict] = {}
    for index, asset in enumerate(ordered):
        role = cardinal_order[int(round((index / denominator) * 4.0)) % 4]
        nearest = by_role[role]
        result[int(asset["id"])] = {
            "yaw_index": index,
            "yaw_fraction": round(index / denominator, 6),
            "nearest_role": role,
            "nearest_reference": nearest,
            "identity_pack": pack,
            "runtime_reference_paths": [str(nearest["path"]), str(pack["path"])],
        }
    return result


def _qwen_consistency_inputs(run_id: int, assets: list, references: list[dict]) -> dict[int, dict]:
    """Bind each diagonal target to its two adjacent immutable cardinal views.

    A single cardinal image leaves the hidden half of asymmetric clothes and
    props underdetermined.  A four-character sheet and rejected H3 guide have
    already caused duplicate bodies and ornament drift, so use exactly the two
    neighboring cardinal masters as separate Qwen inputs instead.
    """
    by_role = {str(reference.get("role")): reference for reference in references}
    targets = {
        1: ("front", "left", "front-left quarter view"),
        3: ("back", "left", "back-left quarter view"),
        5: ("back", "right", "back-right quarter view"),
        7: ("front", "right", "front-right quarter view"),
    }
    result: dict[int, dict] = {}
    for asset in assets:
        metadata = json.loads(asset["metadata_json"] or "{}")
        lineage = metadata.get("lineage") if isinstance(metadata.get("lineage"), dict) else {}
        quality = metadata.get("quality") if isinstance(metadata.get("quality"), dict) else {}
        ordinal = int(lineage.get("frame_ordinal", quality.get("ordinal", -1)))
        if ordinal not in targets:
            raise HTTPException(status_code=400, detail=f"Consistency比較は中間yaw ordinal 1/3/5/7だけです: Asset #{asset['id']} ordinal={ordinal}")
        source_role, adjacent_role, target_yaw = targets[ordinal]
        missing = [role for role in (source_role, adjacent_role) if role not in by_role]
        if missing:
            raise HTTPException(status_code=409, detail=f"{target_yaw}用の参照が不足しています: {', '.join(missing)}")
        source_reference = by_role[source_role]
        adjacent_reference = by_role[adjacent_role]
        result[int(asset["id"])] = {
            "yaw_ordinal": ordinal,
            "target_yaw": target_yaw,
            "source_role": source_role,
            "adjacent_role": adjacent_role,
            "source_reference": source_reference,
            "adjacent_reference": adjacent_reference,
            "runtime_source_path": str(source_reference["path"]),
            "runtime_reference_paths": [str(adjacent_reference["path"])],
            "angle_lora": QWEN_MULTIPLE_ANGLES_LORA,
            "angle_lora_strength": 0.9,
            "prompt_contract": f"<sks> {target_yaw} eye-level shot medium shot",
            "sampler_profile": {"steps": 4, "cfg": 1.0, "lightning": True},
        }
    return result


def _qwen_resolved_prompt(requested_prompt: str, master_references: list[dict]) -> str:
    """Describe the exact Qwen multi-image contract in the executed prompt."""
    anchors = [
        f"input image {index} is the immutable {str(reference.get('role') or 'identity')} identity anchor"
        for index, reference in enumerate(master_references, start=2)
    ]
    prefix = (
        "Edit only input image 1; preserve its exact camera viewpoint, body orientation, pose, framing, "
        "silhouette, prop placement, and background; never rotate or normalize a side/back view into a front view; "
        + "; ".join(anchors)
        + "; use anchors for identity details only and do not copy their pose or background"
    )
    return f"{prefix}; {requested_prompt.strip()}".strip("; ")


def _qwen_dataset_regeneration_prompt(requested_prompt: str, binding: dict) -> str:
    """Describe the three causal inputs used for one dataset yaw regeneration."""
    nearest_role = str(binding["nearest_role"])
    return (
        "Regenerate a clean training-dataset still image. Input image 1 is a rejected H3 yaw guide: "
        "use only its camera viewpoint, body orientation, neutral standing pose, framing, silhouette, and prop placement; "
        "do not preserve its face, fingers, pedestal, gray background, hallucinated ornaments, or generated noise. "
        f"Input image 2 is the immutable {nearest_role} identity reference: use its exact face, hair, eyes, anatomy, outfit construction, accessories, markings, and color palette. "
        "Input image 3 is an immutable four-view identity sheet arranged FRONT top-left, LEFT top-right, BACK bottom-left, RIGHT bottom-right: "
        "cross-check every character-specific detail against all four views. Output exactly one full-body character on a uniform pure #FFFFFF background, "
        "with no floor, no pedestal, no platform, no cast shadow, no gradient, no scenery, no text, no border, and no new decoration. "
        "Render a sharply readable face and two anatomically coherent hands with five distinct fingers where visible. "
        "Do not add, remove, redesign, duplicate, or relocate any clothing part, instrument part, strap, marking, tail feature, or ornament. "
        f"{requested_prompt.strip()}"
    ).strip()


def _qwen_consistency_prompt(requested_prompt: str, binding: dict) -> str:
    angle_prompt = binding["prompt_contract"]
    source_role = binding["source_role"]
    adjacent_role = binding["adjacent_role"]
    return (
        f"{angle_prompt}. Input image 1 is the immutable {source_role} identity reference. "
        f"Input image 2 is the immutable adjacent {adjacent_role} identity reference of the same character. "
        "Fuse these two views into one subject and change only the camera azimuth requested by the <sks> camera-control trigger. Preserve one head, two fox ears, one body, two arms, two hands, two legs, exactly one orange fox tail, and exactly one keytar. "
        "Cross-check the face, hair, eye design, outfit topology, straps, buckles, skirt marking, stocking pattern, boots, keytar front/back construction, key layout, palette, and charm against both references. "
        "Do not invent, duplicate, remove, relocate, or redesign any ornament, limb, tail, instrument, strap, buckle, marking, or accessory. "
        "Use a uniform pure #FFFFFF background with no floor, pedestal, platform, shadow, gradient, scenery, text, or border. "
        "Keep the face and every visible finger sharply readable. Output exactly one full-body character. "
        f"{requested_prompt.strip()}"
    ).strip()


def _selected_qwen_version(conn, asset_row, *, require_choice: bool = True):
    """Resolve the user-selected immutable Qwen correction for one H3 Frame."""
    versions = conn.execute(
        "SELECT id, file_path, content_sha256, version_kind FROM basepipe_asset_versions "
        "WHERE asset_id=? AND status='ready' AND version_kind='qwen_correction' ORDER BY id DESC",
        (int(asset_row["id"]),),
    ).fetchall()
    if not versions:
        return None
    metadata = json.loads(asset_row["metadata_json"] or "{}")
    selected_id = metadata.get("selected_qwen_version_id")
    if selected_id is not None:
        selected = next((version for version in versions if int(version["id"]) == int(selected_id)), None)
        if selected is None:
            raise HTTPException(status_code=409, detail=f"Frame #{asset_row['id']} の選択Qwen Versionがready状態ではありません")
        return selected
    if len(versions) == 1:
        return versions[0]
    if require_choice:
        raise HTTPException(status_code=409, detail=f"Frame #{asset_row['id']} に複数のQwen Versionがあります。使用Versionを選択してください")
    return versions[0]


def _training_input_selection(conn, asset_row) -> dict | None:
    """Resolve only an explicitly user-accepted immutable training input."""
    metadata = json.loads(asset_row["metadata_json"] or "{}")
    if metadata.get("training_input_review_state") != USER_APPROVED:
        return None
    if metadata.get("training_input_reviewed_by") != "user":
        return None
    choice = str(metadata.get("training_input_choice") or "")
    if metadata.get("training_input_review_choice") != choice:
        return None
    if choice == "original":
        if metadata.get("training_input_review_version_id") is not None:
            return None
        path = Path(str(asset_row["file_path"] or ""))
        digest = str(asset_row["content_sha256"] or "")
        if not path.is_file() or not digest or _sha256(path) != digest:
            raise HTTPException(status_code=409, detail=f"Frame #{asset_row['id']} のOriginal fileをSHA-256照合できません")
        return {"input_source": "original", "file_path": str(path), "content_sha256": digest, "version_id": None}
    if choice == "qwen":
        selected_id = metadata.get("selected_qwen_version_id")
        if selected_id is None or metadata.get("training_input_review_version_id") != selected_id:
            return None
        version = _selected_qwen_version(conn, asset_row, require_choice=True)
        if version is None:
            raise HTTPException(status_code=409, detail=f"Frame #{asset_row['id']} のQwen Training Inputが見つかりません")
        path = Path(str(version["file_path"] or ""))
        digest = str(version["content_sha256"] or "")
        if not path.is_file() or not digest or _sha256(path) != digest:
            raise HTTPException(status_code=409, detail=f"Frame #{asset_row['id']} のQwen fileをSHA-256照合できません")
        return {"input_source": "qwen_version", **dict(version)}
    if choice == "enhanced":
        selected_id = metadata.get("selected_version_id")
        if selected_id is None or metadata.get("training_input_review_version_id") != selected_id:
            return None
        version = conn.execute(
            "SELECT id, file_path, content_sha256, version_kind FROM basepipe_asset_versions "
            "WHERE id=? AND asset_id=? AND status='ready'",
            (int(selected_id), int(asset_row["id"])),
        ).fetchone()
        allowed = {"flashvsr_lr9_2x", "flashvsr_lr11_2x", "realesrgan_x2"}
        if version is None or str(version["version_kind"]) not in allowed:
            raise HTTPException(status_code=409, detail=f"Frame #{asset_row['id']} のEnhanced Training Inputが見つかりません")
        path = Path(str(version["file_path"] or ""))
        digest = str(version["content_sha256"] or "")
        if not path.is_file() or not digest or _sha256(path) != digest:
            raise HTTPException(status_code=409, detail=f"Frame #{asset_row['id']} のEnhanced fileをSHA-256照合できません")
        return {"input_source": "enhanced_version", **dict(version)}
    return None


@router.post("/projects/{project_id}/character-generation/preflight")
def character_generation_preflight(project_id: int, payload: CharacterGenerationPreflightIn) -> dict:
    """Verify the real H3 generation prerequisites without starting a GPU job."""
    conn = get_conn()
    try:
        _project_or_404(conn, project_id)
        assets = conn.execute("SELECT id, file_path, review_status FROM basepipe_assets WHERE project_id = ? AND id IN ({}) ORDER BY id".format(",".join("?" for _ in payload.asset_ids) or "NULL"), [project_id, *payload.asset_ids]).fetchall() if payload.asset_ids else []
        workflow_names = (
            ("MiniMax H3 - R2V Validation.json", "H3 Re2Va.json")
            if payload.mode in ("training", "variation")
            else ("MiniMax H3 - Unified Director 4 Modes.json", "H3 Re2Va.json")
        )
        workflow = next((H3_WORKFLOW_ROOT / name for name in workflow_names if (H3_WORKFLOW_ROOT / name).is_file()), H3_WORKFLOW_ROOT / workflow_names[0])
        workflow_ok = workflow.is_file()
        model_root = H3_WORKFLOW_ROOT.parents[2] / "models" / "diffusion_models"
        model_candidates = list(model_root.glob("*h3*")) if model_root.is_dir() else []
        try:
            import urllib.request
            with urllib.request.urlopen(f"{H3_COMFYUI_URL}/system_stats", timeout=1.5) as response:
                comfy = json.load(response)
            comfy_ok = bool(comfy.get("devices"))
            comfy_detail = "ComfyUI接続OK"
        except Exception as exc:  # noqa: BLE001
            comfy_ok = False
            comfy_detail = f"ComfyUI未接続: {exc}"
        gpu_memory = _query_h3_gpu_memory()
        gpu_mapping = _h3_cuda_mapping()
        roles_ok, roles_detail = _validate_reference_roles(payload.asset_ids, payload.reference_roles)
        checks = [
            {"label": "GPU", "ok": payload.gpu_device_id == 1, "detail": "GPU1（RTX 3090 Ti）固定" if payload.gpu_device_id == 1 else "GPU0は禁止"},
            {
                "label": "GPU1 Mapping",
                "ok": bool(gpu_mapping.get("verified")),
                "detail": (
                    f"物理GPU1 {gpu_mapping.get('physical_name') or 'unknown'} → CUDA {gpu_mapping.get('cuda_index')} "
                    f"（{gpu_mapping.get('cuda_name') or 'unknown'}）"
                    if gpu_mapping.get("verified")
                    else str(gpu_mapping.get("reason") or "CUDA列挙を照合できません")
                ),
            },
            _h3_gpu_check(gpu_memory),
            {"label": "Reference Asset", "ok": len(assets) >= 1 and all(Path(row["file_path"]).is_file() for row in assets), "detail": f"{len(assets)}件 / 最大4件"},
            {"label": "Reference Roles", "ok": roles_ok, "detail": roles_detail},
            {"label": "H3 Workflow", "ok": workflow_ok, "detail": str(workflow)},
            {"label": "H3 Model", "ok": bool(model_candidates), "detail": f"{len(model_candidates)}候補"},
            {"label": "ComfyUI", "ok": comfy_ok, "detail": comfy_detail},
        ]
        ready = all(check["ok"] for check in checks)
        requested = {"asset_ids": payload.asset_ids, "mode": payload.mode, "gpu_device_id": payload.gpu_device_id, "reference_roles": payload.reference_roles}
        resolved = {"workflow": str(workflow), "model_candidates": [str(path) for path in model_candidates], "gpu_mapping": gpu_mapping}
        # Persist the exact checks, not only the derived ready/blocked status.
        # The GUI and an external agent must be able to reconstruct the same
        # admission evidence after a reload without re-running the probe.
        observed = {
            "record_kind": "preflight_probe",
            "comfyui_connected": comfy_ok,
            "comfyui_detail": comfy_detail,
            "gpu_memory": gpu_memory,
            "gpu_mapping": gpu_mapping,
            "checks": checks,
        }
        cur = conn.execute("INSERT INTO basepipe_character_generation_runs(project_id, mode, status, requested_json, resolved_json, observed_json) VALUES (?, ?, ?, ?, ?, ?)", (project_id, payload.mode, "ready" if ready else "blocked", json.dumps(requested, ensure_ascii=False), json.dumps(resolved, ensure_ascii=False), json.dumps(observed, ensure_ascii=False)))
        conn.commit()
        return {"run_id": int(cur.lastrowid), "project_id": project_id, "status": "ready" if ready else "blocked", "checks": checks, "requested": requested, "resolved": resolved, "observed": observed, "message": "H3生成を開始できる前提が揃っています" if ready else "H3生成は開始していません。未充足の前提を確認してください。"}
    finally:
        conn.close()


def _is_character_preflight_probe(item: dict) -> bool:
    """Keep admission probes auditable without presenting them as production Runs.

    Older probes predate ``record_kind``.  A real prepared/executed generation
    request always freezes ``prompt`` and ``execute``; a preflight request does
    not.  This fallback hides historical probe rows such as blocked Run #23
    without deleting their evidence.
    """
    observed = item.get("observed") if isinstance(item.get("observed"), dict) else {}
    if observed.get("record_kind") == "preflight_probe":
        return True
    requested = item.get("requested") if isinstance(item.get("requested"), dict) else {}
    return (
        item.get("status") in {"ready", "blocked"}
        and "execute" not in requested
        and "prompt" not in requested
    )


@router.get("/projects/{project_id}/character-generation/runs")
def list_character_generation_runs(
    project_id: int,
    limit: int = 50,
    include_preflight_probes: bool = False,
) -> dict:
    conn = get_conn()
    try:
        _project_or_404(conn, project_id)
        requested_limit = max(1, min(limit, 200))
        # Filtering after a caller-sized SQL LIMIT could let newer probes hide
        # older production Runs. Scan the bounded history window first, then
        # apply the public limit to the visible result.
        fetch_limit = requested_limit if include_preflight_probes else 200
        waiting_queues = {
            status: [
                int(waiting_row["id"])
                for waiting_row in conn.execute(
                    "SELECT id FROM basepipe_character_generation_runs WHERE status = ? ORDER BY id",
                    (status,),
                ).fetchall()
            ]
            for status in ("waiting", "waiting_qwen", "waiting_enhancement")
        }
        rows = conn.execute(
            "SELECT id, project_id, mode, status, requested_json, resolved_json, observed_json, output_manifest_json, current_stage, stage_history_json, error_detail, created_at, updated_at "
            "FROM basepipe_character_generation_runs WHERE project_id = ? ORDER BY id DESC LIMIT ?",
            (project_id, fetch_limit),
        ).fetchall()
        runs = []
        for row in rows:
            item = dict(row)
            for key in ("requested_json", "resolved_json", "observed_json", "output_manifest_json", "stage_history_json"):
                item[key.removesuffix("_json")] = json.loads(item.pop(key) or "{}")
            item["record_kind"] = (
                "preflight_probe" if _is_character_preflight_probe(item) else "generation_run"
            )
            if not include_preflight_probes and item["record_kind"] == "preflight_probe":
                continue
            queue = waiting_queues.get(str(item["status"]), [])
            if int(item["id"]) in queue:
                position = queue.index(int(item["id"])) + 1
                item["queue_position"] = position
                item["queue_head"] = position == 1
                item["ahead_run_id"] = queue[position - 2] if position > 1 else None
            else:
                item["queue_position"] = None
                item["queue_head"] = False
                item["ahead_run_id"] = None
            runs.append(item)
            if len(runs) >= requested_limit:
                break
        return {"project_id": project_id, "runs": runs}
    finally:
        conn.close()


@router.get("/projects/{project_id}/character-generation/runs/{run_id}")
def get_character_generation_run(project_id: int, run_id: int) -> dict:
    conn = get_conn()
    try:
        row = conn.execute("SELECT * FROM basepipe_character_generation_runs WHERE id = ? AND project_id = ?", (run_id, project_id)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="character generation run not found")
        result = dict(row)
        for key in ("requested_json", "resolved_json", "observed_json", "output_manifest_json", "stage_history_json"):
            result[key.removesuffix("_json")] = json.loads(result.pop(key) or "{}")
        result["record_kind"] = (
            "preflight_probe" if _is_character_preflight_probe(result) else "generation_run"
        )
        if result["status"] in {"waiting", "waiting_qwen", "waiting_enhancement"}:
            queue = [
                int(waiting_row["id"])
                for waiting_row in conn.execute(
                    "SELECT id FROM basepipe_character_generation_runs WHERE status = ? ORDER BY id",
                    (result["status"],),
                ).fetchall()
            ]
            position = queue.index(run_id) + 1
            result["queue_position"] = position
            result["queue_head"] = position == 1
            result["ahead_run_id"] = queue[position - 2] if position > 1 else None
        else:
            result["queue_position"] = None
            result["queue_head"] = False
            result["ahead_run_id"] = None
        return result
    finally:
        conn.close()


def _update_character_generation_run(run_id: int, *, status: str, observed: dict | None = None, manifest: dict | None = None, error: str = "", current_stage: str | None = None) -> None:
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT status, stage_history_json, observed_json, output_manifest_json FROM basepipe_character_generation_runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        history = json.loads(row["stage_history_json"] or "[]") if row else []
        merged_observed = json.loads(row["observed_json"] or "{}") if row else {}
        if observed:
            merged_observed.update(observed)
        merged_manifest = manifest if manifest is not None else (json.loads(row["output_manifest_json"] or "{}") if row else {})
        user_rejected = bool(row and row["status"] == "user_rejected")
        effective_status = "user_rejected" if user_rejected else status
        effective_error = "" if user_rejected else error
        effective_stage = None if user_rejected else current_stage
        if effective_stage:
            history.append({"stage": current_stage, "status": "completed" if status == "completed" else status, "artifact_manifest": manifest or {}})
        conn.execute(
            "UPDATE basepipe_character_generation_runs SET status = ?, observed_json = ?, output_manifest_json = ?, error_detail = ?, current_stage = COALESCE(?, current_stage), stage_history_json = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (effective_status, json.dumps(merged_observed, ensure_ascii=False), json.dumps(merged_manifest, ensure_ascii=False), effective_error, effective_stage, json.dumps(history, ensure_ascii=False), run_id),
        )
        conn.commit()
    finally:
        conn.close()


def _character_generation_status(run_id: int) -> str:
    conn = get_conn()
    try:
        row = conn.execute("SELECT status FROM basepipe_character_generation_runs WHERE id = ?", (run_id,)).fetchone()
        return str(row["status"]) if row is not None else "failed"
    finally:
        conn.close()


def _run_h3_generation(run_id: int, project_id: int, source_paths: list[Path], clip_plans: list[dict], outputs_dir: Path) -> None:
    """Execute the frozen H3 coverage clips sequentially on managed GPU1."""
    try:
        import requests
        from ..services.comfyui_client import ComfyUIClient

        client = ComfyUIClient(base_url=H3_COMFYUI_URL, timeout=20.0)
        with requests.get(f"{H3_COMFYUI_URL}/system_stats", timeout=3.0) as response:
            response.raise_for_status()
            stats = response.json()
        if not stats.get("devices"):
            raise RuntimeError("ComfyUIのGPUデバイスが確認できません")
        uploaded_references = [client.upload_image(path) for path in source_paths]
        _update_character_generation_run(
            run_id,
            status="running",
            observed={
                "queued": True,
                "waiting": False,
                "uploaded_references": uploaded_references,
                "gpu_device_id": 1,
                "clip_progress": {"completed": 0, "total": len(clip_plans)},
            },
        )
        output_dir = outputs_dir / "h3" / f"run_{run_id}"
        output_dir.mkdir(parents=True, exist_ok=True)
        clip_manifests: list[dict] = []
        for ordinal, frozen_plan in enumerate(clip_plans, start=1):
            graph = json.loads(json.dumps(frozen_plan["graph"]))
            for index, uploaded in enumerate(uploaded_references):
                node_id = "ref" if index == 0 else f"ref_{index}"
                graph[node_id]["inputs"]["image"] = uploaded
            prompt_id = client.submit(graph)
            # Persist liveness evidence as soon as ComfyUI accepts the prompt.
            # Waiting until history completes makes a legitimately running GPU
            # job look stalled in the GUI for its entire sampling/decode time.
            _update_character_generation_run(
                run_id,
                status="running",
                observed={
                    "queued": True,
                    "waiting": False,
                    "prompt_id": prompt_id,
                    "gpu_device_id": 1,
                    "clip_progress": {
                        "completed": len(clip_manifests),
                        "total": len(clip_plans),
                        "current_clip_id": str(frozen_plan.get("clip_id") or f"clip_{ordinal:02d}"),
                    },
                },
            )
            entry = client.wait_history(prompt_id, timeout=1800.0, poll=2.0)
            media_ref = client.first_output_video(entry)
            content = client.download_media(media_ref)
            suffix = Path(str(media_ref.get("filename", ""))).suffix.lower() or ".mp4"
            clip_id = str(frozen_plan.get("clip_id") or f"clip_{ordinal:02d}")
            output = output_dir / f"{clip_id}_{prompt_id}{suffix}"
            output.write_bytes(content)
            digest = _sha256(output)
            clip_manifest = {
                "clip_id": clip_id,
                "clip_index": int(frozen_plan.get("clip_index", ordinal)),
                "prompt_id": prompt_id,
                "prompt": str(frozen_plan.get("prompt", "")),
                "seed": int(frozen_plan.get("seed", 0)),
                "coverage_preset": str(frozen_plan.get("coverage_preset", "custom")),
                "coverage": frozen_plan.get("coverage") if isinstance(frozen_plan.get("coverage"), dict) else {},
                "media": media_ref,
                "file_path": str(output),
                "content_sha256": digest,
                "review_status": "pending",
            }
            clip_manifests.append(clip_manifest)
            conn = get_conn()
            try:
                conn.execute(
                    "INSERT INTO basepipe_assets(project_id, asset_key, file_path, content_sha256, asset_type, origin_kind, source_ref, review_status, metadata_json) VALUES (?, ?, ?, ?, 'video', 'h3_generated', ?, 'pending', ?)",
                    (project_id, f"h3_run_{run_id}_{clip_id}", str(output), digest, f"character_generation:{run_id}", json.dumps(clip_manifest, ensure_ascii=False)),
                )
                conn.commit()
            finally:
                conn.close()
            partial_manifest = {
                "schema": "basepipe.character_generation_output.v2",
                "run_id": run_id,
                "clips": clip_manifests,
                "clip_count": len(clip_plans),
                "completed_clip_count": len(clip_manifests),
                "source_asset_paths": [str(path) for path in source_paths],
                "reference_count": len(source_paths),
                "gpu_device_id": 1,
            }
            if clip_manifests:
                partial_manifest.update({
                    "file_path": clip_manifests[0]["file_path"],
                    "content_sha256": clip_manifests[0]["content_sha256"],
                })
            _update_character_generation_run(
                run_id,
                status="running",
                observed={
                    "queued": True,
                    "waiting": False,
                    "prompt_id": prompt_id,
                    "gpu_device_id": 1,
                    "clip_progress": {"completed": len(clip_manifests), "total": len(clip_plans), "current_clip_id": clip_id},
                },
                manifest=partial_manifest,
            )
        manifest = {
            "schema": "basepipe.character_generation_output.v2",
            "run_id": run_id,
            "clips": clip_manifests,
            "clip_count": len(clip_plans),
            "completed_clip_count": len(clip_manifests),
            "file_path": clip_manifests[0]["file_path"],
            "content_sha256": clip_manifests[0]["content_sha256"],
            "review_status": "pending",
            "gpu_device_id": 1,
            "source_asset_paths": [str(path) for path in source_paths],
            "reference_count": len(source_paths),
            "pipeline_stages": {stage: ("completed" if stage == "h3_generation" else "pending") for stage in CHARACTER_GENERATION_STAGES},
        }
        _update_character_generation_run(
            run_id,
            status="completed",
            observed={
                "queued": False,
                "waiting": False,
                "gpu_device_id": 1,
                "clip_progress": {"completed": len(clip_manifests), "total": len(clip_plans)},
            },
            manifest=manifest,
            current_stage="h3_generation",
        )
        try:
            extraction = extract_character_generation_frames(
                project_id,
                run_id,
                CharacterGenerationFrameExtractionIn(),
            )
            extraction_manifest = extraction.get("manifest") or {}
            _update_character_generation_run(
                run_id,
                status="completed",
                observed={
                    "auto_frame_extraction": {
                        "status": "completed",
                        "frame_count": int(extraction_manifest.get("frame_count", 0)),
                        "clip_count": int(extraction_manifest.get("clip_count", len(clip_manifests))),
                    },
                },
            )
        except Exception as extraction_exc:  # noqa: BLE001 - H3 artifact remains valid and retryable
            logger.exception("Automatic frame extraction failed for H3 Run #%s", run_id)
            detail = extraction_exc.detail if isinstance(extraction_exc, HTTPException) else str(extraction_exc)
            _update_character_generation_run(
                run_id,
                status="completed",
                observed={"auto_frame_extraction": {"status": "failed", "detail": str(detail)}},
            )
    except Exception as exc:  # noqa: BLE001 — preserve the real runtime failure on the Run
        _update_character_generation_run(run_id, status="failed", observed={"queued": False, "waiting": False, "gpu_device_id": 1}, error=str(exc))
    finally:
        release = _stop_managed_h3_comfyui(f"H3 Run #{run_id} finished")
        _update_character_generation_run(run_id, status=_character_generation_status(run_id), observed={"comfyui_release": release})


def _record_h3_wait_probe(run_id: int, memory: dict, detail: str, *, launch_detail: str = "", reference_integrity: dict | None = None, gpu_mapping: dict | None = None) -> None:
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT observed_json FROM basepipe_character_generation_runs WHERE id = ? AND status = 'waiting'",
            (run_id,),
        ).fetchone()
        if row is None:
            return
        observed = json.loads(row["observed_json"] or "{}")
        observed.update({
            "waiting": True,
            "queued": False,
            "wait_reason": detail,
            "last_admission_probe_at": time.time(),
            "gpu_memory": memory,
        })
        if launch_detail:
            observed["comfyui_launch_detail"] = launch_detail
        if reference_integrity is not None:
            observed["reference_integrity"] = reference_integrity
        if gpu_mapping is not None:
            observed["gpu_mapping"] = gpu_mapping
        observed["admission_probe_count"] = int(observed.get("admission_probe_count", 0)) + 1
        conn.execute(
            "UPDATE basepipe_character_generation_runs SET observed_json = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ? AND status = 'waiting'",
            (json.dumps(observed, ensure_ascii=False), run_id),
        )
        conn.commit()
    finally:
        conn.close()


def _h3_wait_monitor() -> None:
    """Advance explicitly armed H3 Runs when physical GPU1 becomes safe.

    The monitor is FIFO and single-flight.  It never changes the selected GPU,
    never kills a process, and only launches the allowlisted isolated ComfyUI
    after the same physical-GPU gate used by Preflight reports safe.
    """
    global _H3_WAIT_MONITOR_THREAD
    try:
        while True:
            conn = get_conn()
            try:
                waiting = conn.execute(
                    "SELECT * FROM basepipe_character_generation_runs WHERE status = 'waiting' ORDER BY id LIMIT 1"
                ).fetchone()
                active = _active_gpu_work(conn)
            finally:
                conn.close()
            if waiting is None:
                return
            if active is not None:
                time.sleep(H3_WAIT_POLL_SECONDS)
                continue

            run_id = int(waiting["id"])
            resolved = json.loads(waiting["resolved_json"] or "{}")
            observed = json.loads(waiting["observed_json"] or "{}")
            gpu_mapping = observed.get("gpu_mapping")
            if not isinstance(gpu_mapping, dict) or not gpu_mapping.get("verified"):
                gpu_mapping = _h3_cuda_mapping()
            if not gpu_mapping.get("verified"):
                memory = _query_h3_gpu_memory()
                _record_h3_wait_probe(
                    run_id,
                    memory,
                    f"物理GPU1とH3 CUDA列挙を照合できません: {gpu_mapping.get('reason', 'unknown')}",
                    gpu_mapping=gpu_mapping,
                )
                time.sleep(H3_WAIT_POLL_SECONDS)
                continue
            reference_integrity = _h3_reference_integrity(int(waiting["project_id"]), resolved)
            if not reference_integrity.get("ok"):
                _update_character_generation_run(
                    run_id,
                    status="failed",
                    observed={"waiting": False, "reference_integrity": reference_integrity},
                    error=f"Reference integrity failed: {reference_integrity['detail']}",
                )
                continue
            memory = _query_h3_gpu_memory()
            gpu_check = _h3_gpu_check(memory)
            if not memory.get("admission_ok"):
                _record_h3_wait_probe(run_id, memory, str(gpu_check["detail"]), reference_integrity=reference_integrity, gpu_mapping=gpu_mapping)
                time.sleep(H3_WAIT_POLL_SECONDS)
                continue

            comfy_ready, comfy_detail = _ensure_h3_comfyui()
            if not comfy_ready:
                wait_detail = _release_managed_h3_after_wait_race(
                    f"H3 Run #{run_id} runtime preparation failed",
                    "GPU1は空きましたがH3 ComfyUIを準備できません",
                )
                _record_h3_wait_probe(run_id, memory, wait_detail, launch_detail=comfy_detail, reference_integrity=reference_integrity, gpu_mapping=gpu_mapping)
                time.sleep(H3_WAIT_POLL_SECONDS)
                continue

            # The launcher now owns GPU1. Re-run admission and require that all
            # reported compute clients belong to its process tree.
            admitted = _query_h3_gpu_memory()
            if not admitted.get("admission_ok"):
                wait_detail = _release_managed_h3_after_wait_race(
                    f"H3 Run #{run_id} post-launch admission changed",
                    str(_h3_gpu_check(admitted)["detail"]),
                )
                _record_h3_wait_probe(run_id, admitted, wait_detail, launch_detail=comfy_detail, reference_integrity=reference_integrity, gpu_mapping=gpu_mapping)
                time.sleep(H3_WAIT_POLL_SECONDS)
                continue

            source_paths = [Path(str(path)) for path in resolved.get("reference_asset_paths", [])]
            graph = resolved.get("graph")
            clip_plans = resolved.get("clip_plans") or [{
                "clip_id": "clip_01",
                "clip_index": 1,
                "seed": int(graph.get("noise", {}).get("inputs", {}).get("noise_seed", 0)) if isinstance(graph, dict) else 0,
                "prompt": str(graph.get("conditioning", {}).get("inputs", {}).get("prompt", "")) if isinstance(graph, dict) else "",
                "coverage_preset": "legacy",
                "graph": graph,
            }]
            if not source_paths or not clip_plans or any(not isinstance(item.get("graph"), dict) for item in clip_plans) or not all(path.is_file() for path in source_paths):
                release = _stop_managed_h3_comfyui(f"H3 Run #{run_id} invalid waiting payload")
                _update_character_generation_run(run_id, status="failed", observed={"waiting": False}, error="待機RunのReferenceまたはH3 graphが失われています")
                _update_character_generation_run(run_id, status="failed", observed={"comfyui_release": release})
                continue

            conn = get_conn()
            try:
                conn.execute("BEGIN IMMEDIATE")
                observed = json.loads(waiting["observed_json"] or "{}")
                observed.update({
                    "waiting": False,
                    "queued": True,
                    "admitted_at": time.time(),
                    "gpu_memory": admitted,
                    "gpu_mapping": gpu_mapping,
                    "comfyui_launch_detail": comfy_detail,
                })
                active_now = _active_gpu_work(conn)
                changed = 0 if active_now is not None else conn.execute(
                    "UPDATE basepipe_character_generation_runs SET status = 'queued', observed_json = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ? AND status = 'waiting'",
                    (json.dumps(observed, ensure_ascii=False), run_id),
                ).rowcount
                conn.commit()
                project = conn.execute("SELECT outputs_dir FROM projects WHERE id = ?", (int(waiting["project_id"]),)).fetchone()
            finally:
                conn.close()
            if not changed:
                _stop_managed_h3_comfyui(f"H3 Run #{run_id} lost admission transaction")
                continue
            outputs_dir = Path(str(project["outputs_dir"] or "")) if project else Path.cwd() / ".runtime" / "h3"
            logger.info("H3 waiting Run #%s admitted on physical GPU1", run_id)
            _run_h3_generation(run_id, int(waiting["project_id"]), source_paths, clip_plans, outputs_dir)
    finally:
        with _H3_WAIT_MONITOR_LOCK:
            _H3_WAIT_MONITOR_THREAD = None
        # A Run can be inserted after the loop observed an empty queue but
        # before this thread clears its handle. The inserter would then see a
        # live thread and skip spawning another one. Re-check after clearing
        # the handle so an explicitly armed wait never becomes orphaned.
        try:
            ensure_h3_wait_monitor()
        except Exception as exc:  # noqa: BLE001 - do not hide the original exit
            logger.warning("H3 waiting monitor restart skipped: %s", exc)


def ensure_h3_wait_monitor() -> bool:
    """Start the bounded-poll FIFO monitor if at least one armed Run exists."""
    global _H3_WAIT_MONITOR_THREAD
    conn = get_conn()
    try:
        exists = conn.execute("SELECT 1 FROM basepipe_character_generation_runs WHERE status = 'waiting' LIMIT 1").fetchone() is not None
    finally:
        conn.close()
    if not exists:
        return False
    with _H3_WAIT_MONITOR_LOCK:
        if _H3_WAIT_MONITOR_THREAD and _H3_WAIT_MONITOR_THREAD.is_alive():
            return True
        _H3_WAIT_MONITOR_THREAD = threading.Thread(target=_h3_wait_monitor, daemon=True, name="h3-wait-monitor")
        _H3_WAIT_MONITOR_THREAD.start()
    return True


def _record_qwen_wait_probe(run_id: int, memory: dict, detail: str) -> None:
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT observed_json FROM basepipe_character_generation_runs WHERE id = ? AND status = 'waiting_qwen'",
            (run_id,),
        ).fetchone()
        if row is None:
            return
        observed = json.loads(row["observed_json"] or "{}")
        observed.update({
            "waiting": True,
            "queued": False,
            "wait_stage": "qwen_correction",
            "wait_reason": detail,
            "last_qwen_admission_probe_at": time.time(),
            "gpu_memory": memory,
        })
        observed["qwen_admission_probe_count"] = int(observed.get("qwen_admission_probe_count", 0)) + 1
        conn.execute(
            "UPDATE basepipe_character_generation_runs SET observed_json=?, updated_at=CURRENT_TIMESTAMP WHERE id=? AND status='waiting_qwen'",
            (json.dumps(observed, ensure_ascii=False), run_id),
        )
        conn.commit()
    finally:
        conn.close()


def _execute_qwen_correction_runtime(
    run_id: int,
    project_id: int,
    payload: CharacterGenerationQwenCorrectionIn,
    memory: dict,
) -> dict:
    """Execute or safely resume one admitted Qwen 2511 correction/regeneration batch."""
    regeneration = payload.mode == "dataset_regeneration"
    consistency = payload.mode == "consistency_regeneration"
    quality_profile = QWEN_CONSISTENCY_PROFILE if consistency else (QWEN_DATASET_REGEN_PROFILE if regeneration else SELECTIVE_HQ_PROFILE)
    conn = get_conn()
    try:
        marks = ",".join("?" for _ in payload.asset_ids)
        assets = conn.execute(
            f"SELECT id, file_path, review_status, metadata_json FROM basepipe_assets WHERE project_id=? AND source_ref=? AND origin_kind='h3_frame' AND id IN ({marks}) ORDER BY id",
            [project_id, f"character_generation:{run_id}", *payload.asset_ids],
        ).fetchall()
        if len(assets) != len(set(payload.asset_ids)):
            raise HTTPException(status_code=400, detail="Qwen対象Frameが現行Runと一致しません")
        if not (regeneration or consistency) and any(asset["review_status"] != "approved" for asset in assets):
            raise HTTPException(status_code=400, detail="Qwen補正対象はこのRunのapproved H3 Frameだけです")
    finally:
        conn.close()

    master_references = _qwen_master_references(project_id, run_id)
    regeneration_bindings = _qwen_regeneration_inputs(run_id, list(assets), master_references) if regeneration else {}
    consistency_bindings = _qwen_consistency_inputs(run_id, list(assets), master_references) if consistency else {}
    version_kind = "qwen_consistency_regeneration" if consistency else ("qwen_dataset_regeneration" if regeneration else "qwen_correction")
    resolved_prompts: dict[str, str] = {}
    executed: list[dict] = []
    for asset in assets:
        asset_id = int(asset["id"])
        binding = consistency_bindings.get(asset_id) or regeneration_bindings.get(asset_id)
        resolved_prompt = (
            _qwen_consistency_prompt(payload.prompt, binding)
            if consistency
            else _qwen_dataset_regeneration_prompt(payload.prompt, binding)
            if regeneration
            else _qwen_resolved_prompt(payload.prompt, master_references[:2])
        )
        resolved_prompts[str(asset_id)] = resolved_prompt
        conn = get_conn()
        try:
            existing = conn.execute(
                "SELECT id, file_path, content_sha256, parent_version_id FROM basepipe_asset_versions "
                "WHERE asset_id=? AND version_kind=? AND status='ready' AND prompt=? AND seed=? ORDER BY id DESC LIMIT 1",
                (asset_id, version_kind, resolved_prompt, payload.seed),
            ).fetchone()
        finally:
            conn.close()
        if existing is not None and Path(str(existing["file_path"])).is_file():
            executed.append({
                "asset_id": asset_id,
                "project_id": project_id,
                "version_id": int(existing["id"]),
                "version_kind": version_kind,
                "status": "ready",
                "file_path": existing["file_path"],
                "content_sha256": existing["content_sha256"],
                "parent_version_id": existing["parent_version_id"],
                "resumed": True,
            })
            continue
        executed.append(_run_asset_transform(
            asset_id,
            project_id,
            version_kind,
            BasepipeAssetTransformIn(prompt=resolved_prompt, seed=payload.seed, gpu_device_id=1, execute=True),
            admission_verified=True,
            master_references=master_references,
            runtime_reference_paths=[Path(path) for path in binding["runtime_reference_paths"]] if binding else None,
            reference_contract=binding or {},
            runtime_source_path=Path(binding["runtime_source_path"]) if consistency else None,
            qwen_use_lightning=True if consistency else None,
            qwen_angle_lora_name=QWEN_MULTIPLE_ANGLES_LORA if consistency else None,
            qwen_angle_lora_strength=float(binding["angle_lora_strength"]) if consistency else 0.9,
        ))

    conn = get_conn()
    try:
        current = conn.execute(
            "SELECT stage_history_json, output_manifest_json FROM basepipe_character_generation_runs WHERE id=? AND project_id=?",
            (run_id, project_id),
        ).fetchone()
        history = json.loads(current["stage_history_json"] or "[]") if current else []
        history = [entry for entry in history if entry.get("stage") != "qwen_correction"]
        manifest = {
            "asset_count": len(executed),
            "version_ids": [int(item["version_id"]) for item in executed],
            "engine": "Qwen-Image-Edit-2511",
            "quality_profile": quality_profile,
            "operation_mode": payload.mode,
            "selection_gate": (
                "user-directed consistency comparison; immutable cardinal input plus dedicated Multiple-Angles camera-control LoRA"
                if consistency
                else "user-approved regeneration route; rejected H3 pixels are structural guides only"
                if regeneration
                else "user-approved frame asset IDs only"
            ),
            "selection_scope": [int(asset["id"]) for asset in assets],
            "aesthetic_state": AESTHETIC_UNREVIEWED,
            "source_frames_immutable": True,
            "master_references": master_references,
            "resolved_prompts": resolved_prompts,
            "regeneration_bindings": regeneration_bindings,
            "consistency_bindings": consistency_bindings,
        }
        history.append({"stage": "qwen_correction", "status": "completed", "artifact_manifest": manifest})
        output_manifest = json.loads(current["output_manifest_json"] or "{}") if current else {}
        output_manifest["qwen_correction"] = manifest
        observed = {
            "waiting": False,
            "queued": False,
            "wait_stage": "",
            "wait_reason": "",
            "qwen_completed_at": time.time(),
            "qwen_physical_gpu": memory,
        }
        row = conn.execute("SELECT observed_json FROM basepipe_character_generation_runs WHERE id=?", (run_id,)).fetchone()
        merged_observed = json.loads(row["observed_json"] or "{}") if row else {}
        merged_observed.update(observed)
        conn.execute(
            "UPDATE basepipe_character_generation_runs SET status='completed', current_stage='qwen_correction', stage_history_json=?, output_manifest_json=?, observed_json=?, error_detail='', updated_at=CURRENT_TIMESTAMP WHERE id=? AND project_id=?",
            (json.dumps(history, ensure_ascii=False), json.dumps(output_manifest, ensure_ascii=False), json.dumps(merged_observed, ensure_ascii=False), run_id, project_id),
        )
        conn.commit()
    finally:
        conn.close()
    return {
        "run_id": run_id,
        "project_id": project_id,
        "status": "completed",
        "execute": True,
        "gpu_device_id": 1,
        "physical_gpu": memory,
        "versions": executed,
        "current_stage": "qwen_correction",
        "message": "Qwen-Image-Edit-2511補正を完了し、元Frameを変更せずready Versionを登録しました。",
    }


def _qwen_wait_monitor() -> None:
    global _QWEN_WAIT_MONITOR_THREAD
    try:
        while True:
            conn = get_conn()
            try:
                waiting = conn.execute(
                    "SELECT * FROM basepipe_character_generation_runs WHERE status='waiting_qwen' ORDER BY id LIMIT 1"
                ).fetchone()
                active = _active_gpu_work(conn)
            finally:
                conn.close()
            if waiting is None:
                return
            if active is not None:
                time.sleep(QWEN_WAIT_POLL_SECONDS)
                continue
            run_id = int(waiting["id"])
            memory = _query_qwen_gpu_memory()
            if not memory.get("admission_ok"):
                wait_detail = _release_managed_h3_after_wait_race(
                    f"Qwen Run #{run_id} post-launch admission changed",
                    str(_h3_gpu_check(memory)["detail"]),
                )
                _record_qwen_wait_probe(run_id, memory, wait_detail)
                time.sleep(QWEN_WAIT_POLL_SECONDS)
                continue
            comfy_ready, comfy_detail = _ensure_h3_comfyui()
            if not comfy_ready:
                wait_detail = _release_managed_h3_after_wait_race(
                    f"Qwen Run #{run_id} runtime preparation failed",
                    comfy_detail,
                )
                _record_qwen_wait_probe(run_id, memory, wait_detail)
                time.sleep(QWEN_WAIT_POLL_SECONDS)
                continue
            memory = _query_qwen_gpu_memory()
            if not memory.get("admission_ok"):
                wait_detail = _release_managed_h3_after_wait_race(
                    f"Qwen Run #{run_id} post-launch admission changed",
                    str(_h3_gpu_check(memory)["detail"]),
                )
                _record_qwen_wait_probe(run_id, memory, wait_detail)
                time.sleep(QWEN_WAIT_POLL_SECONDS)
                continue
            from ..services.comfyui_client import ComfyUIClient
            qwen_status = ComfyUIClient(base_url=H3_COMFYUI_URL, timeout=0.75).check_qwen_status()
            if not qwen_status.get("ready"):
                wait_detail = _release_managed_h3_after_wait_race(
                    f"Qwen Run #{run_id} runtime is not ready",
                    str(qwen_status.get("message") or "Qwen-Image-Edit-2511未準備"),
                )
                _record_qwen_wait_probe(run_id, memory, wait_detail)
                time.sleep(QWEN_WAIT_POLL_SECONDS)
                continue
            resolved = json.loads(waiting["resolved_json"] or "{}")
            request = resolved.get("qwen_request")
            if not isinstance(request, dict):
                release = _stop_managed_h3_comfyui(f"Qwen Run #{run_id} invalid waiting payload")
                _update_character_generation_run(run_id, status="failed", observed={"waiting": False}, error="待機RunのQwen requestが失われています")
                _update_character_generation_run(run_id, status="failed", observed={"comfyui_release": release})
                continue
            payload = CharacterGenerationQwenCorrectionIn(**request)
            conn = get_conn()
            try:
                conn.execute("BEGIN IMMEDIATE")
                observed = json.loads(waiting["observed_json"] or "{}")
                observed.update({"waiting": False, "queued": True, "wait_stage": "qwen_correction", "wait_reason": "", "qwen_admitted_at": time.time(), "gpu_memory": memory})
                active_now = _active_gpu_work(conn)
                changed = 0 if active_now is not None else conn.execute(
                    "UPDATE basepipe_character_generation_runs SET status='running_qwen', observed_json=?, updated_at=CURRENT_TIMESTAMP WHERE id=? AND status='waiting_qwen'",
                    (json.dumps(observed, ensure_ascii=False), run_id),
                ).rowcount
                conn.commit()
            finally:
                conn.close()
            if not changed:
                _stop_managed_h3_comfyui(f"Qwen Run #{run_id} lost admission transaction")
                continue
            try:
                _execute_qwen_correction_runtime(run_id, int(waiting["project_id"]), payload, memory)
            except Exception as exc:  # noqa: BLE001 - preserve exact Runtime failure
                from ..services.comfyui_client import ComfyUIUnavailableError

                detail = _runtime_exception_detail(exc)
                retryable = (
                    isinstance(exc, ComfyUIUnavailableError)
                    or (
                        isinstance(exc, HTTPException)
                        and exc.status_code == 409
                        and "external ComfyUI" in str(exc.detail)
                    )
                )
                if retryable:
                    _update_character_generation_run(
                        run_id,
                        status="waiting_qwen",
                        observed={
                            "waiting": True,
                            "queued": False,
                            "wait_reason": detail,
                            "qwen_last_retryable_error": detail,
                        },
                    )
                else:
                    _update_character_generation_run(run_id, status="failed", observed={"waiting": False, "queued": False}, error=detail)
            finally:
                release = _stop_managed_h3_comfyui(f"Qwen Run #{run_id} finished")
                _update_character_generation_run(run_id, status=_character_generation_status(run_id), observed={"comfyui_release": release})
    finally:
        with _QWEN_WAIT_MONITOR_LOCK:
            _QWEN_WAIT_MONITOR_THREAD = None
        try:
            ensure_qwen_wait_monitor()
        except Exception as exc:  # noqa: BLE001 - do not hide the original exit
            logger.warning("Qwen waiting monitor restart skipped: %s", exc)


def recover_stale_qwen_runs_on_startup() -> dict[str, object]:
    """Re-arm Qwen work whose in-process monitor vanished with the backend."""
    conn = get_conn()
    recovered: list[int] = []
    try:
        rows = conn.execute(
            "SELECT id, observed_json FROM basepipe_character_generation_runs "
            "WHERE status='running_qwen' ORDER BY id"
        ).fetchall()
        for row in rows:
            observed = json.loads(row["observed_json"] or "{}")
            observed.update({
                "waiting": True,
                "queued": False,
                "wait_stage": "qwen_correction",
                "wait_reason": "Backend再起動後にQwen安全ゲートから再開します",
                "qwen_recovered_after_restart_at": time.time(),
            })
            conn.execute(
                "UPDATE basepipe_character_generation_runs SET status='waiting_qwen', "
                "observed_json=?, error_detail='', updated_at=CURRENT_TIMESTAMP WHERE id=? AND status='running_qwen'",
                (json.dumps(observed, ensure_ascii=False), int(row["id"])),
            )
            recovered.append(int(row["id"]))
        conn.commit()
    finally:
        conn.close()
    release = (
        _stop_managed_h3_comfyui("backend restart Qwen recovery")
        if recovered
        else {"stopped": False, "detail": "no stale Qwen Run"}
    )
    return {"recovered_run_ids": recovered, "comfyui_release": release}


def ensure_qwen_wait_monitor() -> bool:
    global _QWEN_WAIT_MONITOR_THREAD
    conn = get_conn()
    try:
        exists = conn.execute("SELECT 1 FROM basepipe_character_generation_runs WHERE status='waiting_qwen' LIMIT 1").fetchone() is not None
    finally:
        conn.close()
    if not exists:
        return False
    with _QWEN_WAIT_MONITOR_LOCK:
        if _QWEN_WAIT_MONITOR_THREAD and _QWEN_WAIT_MONITOR_THREAD.is_alive():
            return True
        _QWEN_WAIT_MONITOR_THREAD = threading.Thread(target=_qwen_wait_monitor, daemon=True, name="qwen-wait-monitor")
        _QWEN_WAIT_MONITOR_THREAD.start()
    return True


@router.post("/projects/{project_id}/character-generation/runs")
@router.post("/projects/{project_id}/character-generation/start")
def character_generation_start(project_id: int, payload: CharacterGenerationStartIn) -> dict:
    """Prepare an H3 R2V graph; only execute=true may enqueue a GPU job.

    The preparation path is intentionally useful while ComfyUI is offline: it
    records the requested/resolved graph and leaves the Run in ``prepared``.
    No upload, queue submission, or training restart occurs in that path.
    """
    conn = get_conn()
    try:
        _project_or_404(conn, project_id)
        if payload.gpu_device_id != 1:
            raise HTTPException(status_code=400, detail="GPU0は禁止です。H3はGPU1固定です")
        assets = conn.execute(
            "SELECT id, file_path, content_sha256 FROM basepipe_assets WHERE project_id = ? AND id IN ({}) ORDER BY id".format(
                ",".join("?" for _ in payload.asset_ids) or "NULL"
            ),
            [project_id, *payload.asset_ids],
        ).fetchall() if payload.asset_ids else []
        source_paths = [Path(str(row["file_path"])) for row in assets]
        if not source_paths or not all(path.is_file() for path in source_paths):
            raise HTTPException(status_code=400, detail="実在するReference Assetを1〜4件指定してください")
        roles_ok, roles_detail = _validate_reference_roles([int(row["id"]) for row in assets], payload.reference_roles)
        if not roles_ok:
            raise HTTPException(status_code=400, detail=f"Reference Roleが不正です: {roles_detail}")
        assets = _order_reference_assets(list(assets), payload.reference_roles)
        source_paths = [Path(str(row["file_path"])) for row in assets]

        ordered_roles = [
            str(payload.reference_roles.get(str(row["id"]), "reference")).strip().lower()
            for row in assets
        ]
        clip_plans = _h3_clip_plans(payload, len(source_paths), ordered_roles)
        graph = clip_plans[0]["graph"]
        gpu_mapping = _h3_cuda_mapping()
        requested = payload.model_dump()
        resolved = {
            "engine": "ComfyUI-H3-R2V",
            "gpu_device_id": 1,
            "gpu_mapping": gpu_mapping,
            "reference_asset_ids": [int(row["id"]) for row in assets],
            "reference_asset_paths": [str(path) for path in source_paths],
            "reference_asset_hashes": {str(row["id"]): str(row["content_sha256"]) for row in assets},
            "reference_roles": payload.reference_roles,
            "reference_input_order": [
                {
                    "role": str(payload.reference_roles.get(str(row["id"]), "unspecified")).strip().lower(),
                    "asset_id": int(row["id"]),
                    "file_path": str(row["file_path"]),
                    "content_sha256": row["content_sha256"],
                }
                for row in assets
            ],
            "reference_prompt_map": [
                {"picture_tag": f"<Picture {index}>", "role": role, "asset_id": int(row["id"])}
                for index, (row, role) in enumerate(zip(assets, ordered_roles, strict=True), start=1)
            ],
            "graph": graph,
            "clip_plans": clip_plans,
            "clip_count": len(clip_plans),
            "coverage_preset": payload.coverage_preset,
            "identity_constraints": clip_plans[0]["identity_constraints"],
            "candidate_generation_profile": clip_plans[0]["candidate_generation_profile"],
            "anchor_provenance": clip_plans[0]["anchor_provenance"],
            "quality_strategy": {
                "candidate_pass": "anchor-derived H3 RefMod/Turbo/Spectrum at requested draft dimensions",
                "technical_selection": "deterministic yaw coverage plus blur, duplicate and structural integrity gates run automatically",
                "selective_enhancement": "only automatically selected yaw neighborhoods enter FlashVSR or RealESRGAN comparison",
                "flashvsr": FLASHVSR_DATASET_PROFILE,
                "realesrgan_baseline": REALESRGAN_DATASET_BASELINE,
                "training_input": "user reviews the completed contact sheet and explicitly approves the final dataset inputs",
            },
            "lineage_policy": "technical curation and enhancement are automatic; only the completed contact sheet requires user aesthetic approval before Snapshot",
        }
        status = "prepared"
        observed = {"queued": False, "execution_requested": bool(payload.execute), "gpu_mapping": gpu_mapping}
        if payload.execute and not payload.queue_if_busy:
            gpu_memory = _query_h3_gpu_memory()
            if not gpu_memory.get("admission_ok"):
                detail = _h3_gpu_check(gpu_memory)["detail"]
                raise HTTPException(status_code=409, detail=f"H3実ジョブを開始しません: {detail}")
            comfy_ready, comfy_detail = _ensure_h3_comfyui()
            if not comfy_ready:
                raise HTTPException(status_code=409, detail=f"H3実ジョブを開始できません: {comfy_detail}")
            admitted = _query_h3_gpu_memory()
            if not admitted.get("admission_ok"):
                raise HTTPException(status_code=409, detail=f"H3実ジョブを開始しません: {_h3_gpu_check(admitted)['detail']}")
            status = "queued"
            observed["queued"] = True
        elif payload.execute and payload.queue_if_busy:
            gpu_memory = _query_h3_gpu_memory()
            status = "waiting"
            observed.update({
                "queued": False,
                "waiting": True,
                "auto_start_when_ready": True,
                "gpu_memory": gpu_memory,
                "wait_reason": (
                    "GPU1安全条件とH3 ComfyUIを再評価しています"
                    if gpu_memory.get("admission_ok") and gpu_mapping.get("verified")
                    else str(_h3_gpu_check(gpu_memory)["detail"])
                    if not gpu_memory.get("admission_ok")
                    else f"物理GPU1とH3 CUDA列挙を照合できません: {gpu_mapping.get('reason', 'unknown')}"
                ),
                "admission_probe_count": 0,
            })
        if status == "queued":
            conn.execute("BEGIN IMMEDIATE")
            active_now = _active_gpu_work(conn)
            if active_now is not None:
                conn.rollback()
                raise HTTPException(status_code=409, detail=f"GPU1は{active_now['kind']} Run #{active_now['id']}が使用中です")
        cur = conn.execute(
            "INSERT INTO basepipe_character_generation_runs(project_id, mode, status, requested_json, resolved_json, observed_json) VALUES (?, ?, ?, ?, ?, ?)",
            (project_id, payload.mode, status, json.dumps(requested, ensure_ascii=False), json.dumps(resolved, ensure_ascii=False), json.dumps(observed, ensure_ascii=False)),
        )
        conn.commit()
        run_id = int(cur.lastrowid)
        if payload.execute and not payload.queue_if_busy:
            project = conn.execute("SELECT outputs_dir FROM projects WHERE id = ?", (project_id,)).fetchone()
            outputs_dir = Path(str(project["outputs_dir"] or "")) if project else Path.cwd() / ".runtime" / "h3"
            threading.Thread(target=_run_h3_generation, args=(run_id, project_id, source_paths, clip_plans, outputs_dir), daemon=True, name=f"h3-run-{run_id}").start()
        elif status == "waiting":
            ensure_h3_wait_monitor()
        message = (
            "H3 R2Vグラフを準備しました。GPUジョブは開始していません。"
            if not payload.execute
            else "H3 RunをGPU1待機キューへ追加しました。安全条件を満たすまで外部プロセスには触れません。"
            if status == "waiting"
            else "H3 R2VジョブをGPU1へ投入しました。Runを監視しています。"
        )
        return {"run_id": run_id, "project_id": project_id, "status": status, "queued": status in {"waiting", "queued", "running"}, "requested": requested, "resolved": resolved, "observed": observed, "message": message}
    finally:
        conn.close()


@router.post("/projects/{project_id}/character-generation/runs/{run_id}/cancel-waiting")
def cancel_waiting_character_generation(project_id: int, run_id: int) -> dict:
    """Cancel an H3 or Qwen waiting Run before GPU work starts."""
    conn = get_conn()
    try:
        changed = conn.execute(
            "UPDATE basepipe_character_generation_runs SET status = 'cancelled', error_detail = '', updated_at = CURRENT_TIMESTAMP "
            "WHERE id = ? AND project_id = ? AND status IN ('waiting','waiting_qwen','waiting_enhancement')",
            (run_id, project_id),
        ).rowcount
        if not changed:
            row = conn.execute(
                "SELECT status FROM basepipe_character_generation_runs WHERE id = ? AND project_id = ?",
                (run_id, project_id),
            ).fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail="character generation run not found")
            raise HTTPException(status_code=409, detail=f"GPU待機中のRunだけ取消できます（現在: {row['status']}）")
        conn.commit()
        return {"run_id": run_id, "project_id": project_id, "status": "cancelled", "gpu_process_started": False}
    finally:
        conn.close()


@router.post("/projects/{project_id}/character-generation/runs/{run_id}/aesthetic-review")
def review_character_generation_run(project_id: int, run_id: int, payload: CharacterGenerationRunAestheticReviewIn) -> dict:
    """Record only an explicit human decision and safely release a rejected managed job."""
    _require_user_aesthetic_confirmation(payload.human_confirmed)
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT status, observed_json, output_manifest_json FROM basepipe_character_generation_runs WHERE id=? AND project_id=?",
            (run_id, project_id),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="character generation run not found")
        prior_status = str(row["status"])
        if payload.decision == "approved" and prior_status != "completed":
            raise HTTPException(status_code=409, detail="完了した実成果物だけ承認できます")
        aesthetic_status = USER_APPROVED if payload.decision == "approved" else USER_REJECTED
        observed = json.loads(row["observed_json"] or "{}")
        manifest = json.loads(row["output_manifest_json"] or "{}")
        review = {
            "status": aesthetic_status,
            "decision": payload.decision,
            "note": payload.note,
            "human_confirmed": True,
            "reviewed_at": time.time(),
        }
        observed.update({"aesthetic_status": aesthetic_status, "aesthetic_review": review})
        manifest.update({"aesthetic_status": aesthetic_status, "aesthetic_review": review})
        next_status = "user_rejected" if payload.decision == "rejected" else prior_status
        conn.execute(
            "UPDATE basepipe_character_generation_runs SET status=?, observed_json=?, output_manifest_json=?, error_detail='', updated_at=CURRENT_TIMESTAMP WHERE id=?",
            (next_status, json.dumps(observed, ensure_ascii=False), json.dumps(manifest, ensure_ascii=False), run_id),
        )
        conn.commit()
    finally:
        conn.close()

    release = {"stopped": False, "detail": "no active managed process release required"}
    if payload.decision == "rejected" and prior_status in {"running", "queued", "waiting", "running_enhancement", "waiting_enhancement", "running_qwen", "waiting_qwen"}:
        try:
            import requests
            requests.post(f"{H3_COMFYUI_URL}/interrupt", timeout=3.0)
        except Exception:  # noqa: BLE001 - process ownership gate remains authoritative
            pass
        release = _stop_managed_h3_comfyui(f"Run #{run_id} was explicitly rejected by the user")
    return {
        "run_id": run_id,
        "project_id": project_id,
        "prior_status": prior_status,
        "status": next_status,
        "aesthetic_status": aesthetic_status,
        "managed_runtime_release": release,
    }


@router.post("/projects/{project_id}/character-generation/runs/{run_id}/execute")
def execute_prepared_character_generation(project_id: int, run_id: int) -> dict:
    """Promote one immutable prepared H3 plan into the safe GPU1 wait queue."""
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT status, current_stage, requested_json, resolved_json, observed_json FROM basepipe_character_generation_runs WHERE id = ? AND project_id = ?",
            (run_id, project_id),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="character generation run not found")
        if row["status"] != "prepared" or row["current_stage"] != "planning":
            raise HTTPException(status_code=409, detail=f"prepared状態のH3計画だけ実行待機へ移せます（現在: {row['status']} / {row['current_stage']}）")
        requested = json.loads(row["requested_json"] or "{}")
        resolved = json.loads(row["resolved_json"] or "{}")
        source_paths = [Path(str(path)) for path in resolved.get("reference_asset_paths", [])]
        if not source_paths or not all(path.is_file() for path in source_paths):
            raise HTTPException(status_code=409, detail="保存済みRunのReference fileを確認できません。Runは開始していません")
        if int(requested.get("gpu_device_id", -1)) != 1 or int(resolved.get("gpu_device_id", -1)) != 1:
            raise HTTPException(status_code=409, detail="保存済みRunが物理GPU1へ固定されていません。GPU0では開始しません")
        graph = resolved.get("graph")
        if not isinstance(graph, dict) or not graph:
            raise HTTPException(status_code=409, detail="保存済みH3 graphがありません。Runは開始していません")
        reference_integrity = _h3_reference_integrity(project_id, resolved)
        if not reference_integrity.get("ok"):
            raise HTTPException(status_code=409, detail=f"保存済みReferenceを照合できません: {reference_integrity['detail']}")
        gpu_memory = _query_h3_gpu_memory()
        gpu_mapping = _h3_cuda_mapping()
        observed = json.loads(row["observed_json"] or "{}")
        observed.update({
            "queued": False,
            "execution_requested": True,
            "execution_requested_from_prepared": True,
            "waiting": True,
            "auto_start_when_ready": True,
            "gpu_memory": gpu_memory,
            "gpu_mapping": gpu_mapping,
            "reference_integrity": reference_integrity,
            "wait_reason": (
                "GPU1安全条件とH3 ComfyUIを再評価しています"
                if gpu_memory.get("admission_ok") and gpu_mapping.get("verified")
                else str(_h3_gpu_check(gpu_memory)["detail"])
                if not gpu_memory.get("admission_ok")
                else f"物理GPU1とH3 CUDA列挙を照合できません: {gpu_mapping.get('reason', 'unknown')}"
            ),
            "admission_probe_count": 0,
        })
        changed = conn.execute(
            "UPDATE basepipe_character_generation_runs SET status='waiting', observed_json=?, error_detail='', updated_at=CURRENT_TIMESTAMP WHERE id=? AND project_id=? AND status='prepared' AND current_stage='planning'",
            (json.dumps(observed, ensure_ascii=False), run_id, project_id),
        ).rowcount
        if changed != 1:
            raise HTTPException(status_code=409, detail="Run状態が更新されたため待機キューへ追加できませんでした")
        conn.commit()
        ensure_h3_wait_monitor()
        return {
            "run_id": run_id,
            "project_id": project_id,
            "status": "waiting",
            "queued": True,
            "requested": requested,
            "resolved": resolved,
            "observed": observed,
            "message": "保存済みH3計画をGPU1待機キューへ追加しました。Requested/Resolvedは変更していません。",
        }
    except HTTPException:
        conn.rollback()
        raise
    finally:
        conn.close()


@router.post("/projects/{project_id}/character-generation/runs/{run_id}/stage")
def advance_character_generation_stage(project_id: int, run_id: int, payload: CharacterGenerationStageIn) -> dict:
    """Record one truthful, ordered Dataset Generation Run stage.

    This endpoint records evidence; it does not manufacture frames or approve assets.
    Each completed stage must carry an artifact manifest, and stages cannot be skipped.
    """
    if payload.stage not in CHARACTER_GENERATION_STAGES:
        raise HTTPException(status_code=400, detail=f"unknown generation stage: {payload.stage}")
    conn = get_conn()
    try:
        row = conn.execute("SELECT * FROM basepipe_character_generation_runs WHERE id = ? AND project_id = ?", (run_id, project_id)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="character generation run not found")
        current = row["current_stage"] or "planning"
        target_index = CHARACTER_GENERATION_STAGES.index(payload.stage)
        current_index = CHARACTER_GENERATION_STAGES.index(current) if current in CHARACTER_GENERATION_STAGES else 0
        if target_index != current_index + 1:
            raise HTTPException(status_code=409, detail=f"stage skip is forbidden: {current} -> {payload.stage}")
        if payload.status != "completed":
            raise HTTPException(status_code=409, detail="stage evidence endpointは実際に完了した段階だけを記録します。未完了状態はRuntime Runが更新します")
        if payload.stage == "h3_generation" and payload.status == "completed":
            if row["status"] != "completed" or not json.loads(row["output_manifest_json"] or "{}"):
                raise HTTPException(status_code=409, detail="H3生成完了は実Runtimeのoutput manifestがあるRunだけ記録できます")
        if payload.status == "completed" and not payload.artifact_manifest:
            raise HTTPException(status_code=400, detail="completed stage requires artifact_manifest")
        history = json.loads(row["stage_history_json"] or "[]")
        event = {"stage": payload.stage, "status": payload.status, "artifact_manifest": payload.artifact_manifest, "note": payload.note}
        history.append(event)
        new_current = payload.stage
        conn.execute("UPDATE basepipe_character_generation_runs SET current_stage = ?, stage_history_json = ?, status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?", (new_current, json.dumps(history, ensure_ascii=False), "completed" if payload.stage == "snapshot" else row["status"], run_id))
        conn.commit()
        return {"run_id": run_id, "project_id": project_id, "current_stage": new_current, "stage": event, "stage_history": history}
    except HTTPException:
        conn.rollback()
        raise
    finally:
        conn.close()


@router.post("/projects/{project_id}/character-generation/runs/{run_id}/extract-frames")
def extract_character_generation_frames(project_id: int, run_id: int, payload: CharacterGenerationFrameExtractionIn) -> dict:
    """Decode H3 output into pending frame Assets; never sends frames to training."""
    conn = get_conn()
    try:
        row = conn.execute("SELECT * FROM basepipe_character_generation_runs WHERE id = ? AND project_id = ?", (run_id, project_id)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="character generation run not found")
        if row["current_stage"] != "h3_generation" or row["status"] != "completed":
            raise HTTPException(status_code=409, detail="実H3生成が完了したRunだけFrame抽出できます")
        manifest = json.loads(row["output_manifest_json"] or "{}")
        clip_manifests = manifest.get("clips") or [{
            "clip_id": "clip_01",
            "clip_index": 1,
            "file_path": manifest.get("file_path", ""),
            "prompt": json.loads(row["requested_json"] or "{}").get("prompt", ""),
            "seed": json.loads(row["requested_json"] or "{}").get("seed", 0),
        }]
        if not clip_manifests or any(not Path(str(clip.get("file_path", ""))).is_file() for clip in clip_manifests):
            raise HTTPException(status_code=409, detail="H3 output clipが見つかりません")
        from ..services.h3_dataset import classify_frames, extract_frames
        frames: list[Path] = []
        frame_origins: dict[str, dict] = {}
        for clip_position, clip in enumerate(clip_manifests, start=1):
            video_path = Path(str(clip["file_path"]))
            clip_id = str(clip.get("clip_id") or f"clip_{clip_position:02d}")
            frame_dir = video_path.parent / clip_id / "frames"
            clip_frames = extract_frames(video_path, frame_dir, fps=payload.fps, max_frames=payload.max_frames)
            for clip_frame_index, frame in enumerate(clip_frames):
                frame_origins[str(frame)] = {
                    "clip_id": clip_id,
                    "clip_index": int(clip.get("clip_index", clip_position)),
                    "clip_frame_index": clip_frame_index,
                    "timestamp_seconds": round(clip_frame_index / payload.fps, 4),
                    "source_video": str(video_path),
                    "generation_seed": int(clip.get("seed", 0)),
                    "generation_prompt": str(clip.get("prompt", "")),
                    "coverage": clip.get("coverage") if isinstance(clip.get("coverage"), dict) else {},
                }
            frames.extend(clip_frames)
        classifications = classify_frames(frames, blur_threshold=payload.blur_threshold, duplicate_distance=payload.duplicate_distance)
        created_assets: list[dict] = []
        for item in classifications:
            frame = Path(item["file_path"])
            origin = frame_origins[str(frame)]
            cur = conn.execute(
                "INSERT INTO basepipe_assets(project_id, asset_key, file_path, content_sha256, asset_type, origin_kind, source_ref, review_status, training_enabled, metadata_json) VALUES (?, ?, ?, ?, 'image', 'h3_frame', ?, ?, 0, ?)",
                (
                    project_id,
                    f"h3_run_{run_id}_{origin['clip_id']}_frame_{origin['clip_frame_index']:04d}",
                    str(frame),
                    item["content_sha256"],
                    f"character_generation:{run_id}",
                    item["review_status"],
                    json.dumps({"lineage": {"run_id": run_id, "frame_ordinal": item["ordinal"], **origin}, "quality": item}, ensure_ascii=False),
                ),
            )
            created_assets.append({"asset_id": int(cur.lastrowid), **item, **origin})
        stage_manifest = {
            "frame_count": len(created_assets),
            "clip_count": len(clip_manifests),
            "max_frames_per_clip": payload.max_frames,
            "accepted_candidates": sum(item["review_status"] == "needs_review" for item in created_assets),
            "rejected": sum(item["review_status"] == "rejected" for item in created_assets),
            "frames": created_assets,
            "source_clips": [str(clip["file_path"]) for clip in clip_manifests],
        }
        history = json.loads(row["stage_history_json"] or "[]")
        history.extend([
            {"stage": "frame_extraction", "status": "completed", "artifact_manifest": stage_manifest},
            {"stage": "blur_filter", "status": "completed", "artifact_manifest": {"threshold": payload.blur_threshold, "rejected": stage_manifest["rejected"]}},
            {"stage": "duplicate_filter", "status": "completed", "artifact_manifest": {"distance": payload.duplicate_distance, "rejected": sum(item["rejection_reason"] == "duplicate" for item in created_assets)}},
        ])
        conn.execute("UPDATE basepipe_character_generation_runs SET current_stage='duplicate_filter', stage_history_json=?, output_manifest_json=?, updated_at=CURRENT_TIMESTAMP WHERE id=?", (json.dumps(history, ensure_ascii=False), json.dumps({**manifest, "frame_extraction": stage_manifest}, ensure_ascii=False), run_id))
        conn.commit()
        return {"run_id": run_id, "project_id": project_id, "current_stage": "duplicate_filter", "manifest": stage_manifest}
    except HTTPException:
        conn.rollback()
        raise
    except Exception as exc:  # noqa: BLE001
        conn.rollback()
        raise HTTPException(status_code=400, detail=f"H3 Frame抽出に失敗しました: {exc}") from exc
    finally:
        conn.close()


def _enhancement_source_video(manifest: dict) -> Path:
    extraction = manifest.get("frame_extraction") if isinstance(manifest.get("frame_extraction"), dict) else {}
    candidates = extraction.get("source_clips") or [clip.get("file_path") for clip in manifest.get("clips", []) if isinstance(clip, dict)]
    paths = [Path(str(item)) for item in candidates if item]
    if len(paths) != 1 or not paths[0].is_file():
        raise RuntimeError("Turntable enhancement requires exactly one existing source clip")
    return paths[0]


def _persist_enhancement_progress(run_id: int, progress: dict, *, status: str = "running_enhancement") -> None:
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT status, observed_json, output_manifest_json FROM basepipe_character_generation_runs WHERE id=?",
            (run_id,),
        ).fetchone()
        if row is None:
            return
        if row["status"] == "user_rejected":
            return
        observed = json.loads(row["observed_json"] or "{}")
        observed["enhancement_progress"] = progress
        manifest = json.loads(row["output_manifest_json"] or "{}")
        manifest["enhancement"] = progress
        conn.execute(
            "UPDATE basepipe_character_generation_runs SET status=?, observed_json=?, output_manifest_json=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
            (status, json.dumps(observed, ensure_ascii=False), json.dumps(manifest, ensure_ascii=False), run_id),
        )
        conn.commit()
    finally:
        conn.close()


def _mark_character_enhancement_failed(run_id: int, detail: str) -> None:
    """Close enhancement state consistently while preserving resume evidence."""
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT status, observed_json, output_manifest_json FROM basepipe_character_generation_runs WHERE id=?",
            (run_id,),
        ).fetchone()
        if row is None:
            return
        if row["status"] == "user_rejected":
            return
        observed = json.loads(row["observed_json"] or "{}")
        progress = observed.get("enhancement_progress")
        if not isinstance(progress, dict):
            progress = {}
        progress.update({"status": "failed", "error": detail, "failed_at": time.time()})
        observed.update({
            "waiting": False,
            "queued": False,
            "enhancement_progress": progress,
            "enhancement_failure": detail,
        })
        observed.pop("wait_reason", None)
        manifest = json.loads(row["output_manifest_json"] or "{}")
        manifest["enhancement"] = progress
        conn.execute(
            "UPDATE basepipe_character_generation_runs SET status='failed', observed_json=?, output_manifest_json=?, error_detail=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
            (json.dumps(observed, ensure_ascii=False), json.dumps(manifest, ensure_ascii=False), detail, run_id),
        )
        conn.commit()
    finally:
        conn.close()


def _execute_character_enhancement(run_id: int, project_id: int, request: dict, memory: dict) -> None:
    """Run deterministic enhancement comparisons; never approve a variant."""
    from ..services.character_enhancement import (
        create_enhancement_comparison_sheet,
        derive_2x_from_4x,
        extract_video_center_frame,
        prepare_circular_yaw_windows,
    )
    from ..services.comfyui_client import ComfyUIClient

    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT output_manifest_json, resolved_json FROM basepipe_character_generation_runs WHERE id=? AND project_id=?",
            (run_id, project_id),
        ).fetchone()
        if row is None:
            raise RuntimeError("character generation run not found")
        manifest = json.loads(row["output_manifest_json"] or "{}")
        resolved = json.loads(row["resolved_json"] or "{}")
    finally:
        conn.close()
    source_video = _enhancement_source_video(manifest)
    workspace = source_video.parent / "enhancement"
    windows = prepare_circular_yaw_windows(
        source_video,
        workspace,
        total_frames=int(request.get("total_frames", 124)),
        slot_count=int(request.get("slot_count", 8)),
    )
    client = ComfyUIClient(base_url=H3_COMFYUI_URL, timeout=20.0)
    progress = {
        "status": "running",
        "source_video": str(source_video),
        "source_sha256": _sha256(source_video),
        "profile": resolved.get("quality_strategy", {}),
        "windows": windows,
        "jobs": [],
        "completed_jobs": 0,
        "total_jobs": len(windows) * (len(request.get("local_ranges", [9, 11])) + (1 if request.get("include_realesrgan", True) else 0)),
        "physical_gpu": memory,
        "aesthetic_status": AESTHETIC_UNREVIEWED,
    }
    _persist_enhancement_progress(run_id, progress)
    comparison_rows: list[dict] = []
    for window in windows:
        slot = int(window["yaw_slot"])
        slot_dir = Path(str(window["window_path"])).parent
        comparison = {
            "yaw_slot": slot,
            "yaw_target_degrees": float(window["yaw_target_degrees"]),
            "original": str(window["original_path"]),
        }
        with Image.open(Path(str(window["original_path"]))) as original_image:
            source_size = original_image.size
        for local_range in request.get("local_ranges", [9, 11]):
            native_path = slot_dir / f"flashvsr_lr{int(local_range)}_4x.png"
            two_x_path = slot_dir / f"flashvsr_lr{int(local_range)}_2x.png"
            video_path = slot_dir / f"flashvsr_lr{int(local_range)}_4x.mp4"
            prompt_id = "resumed"
            if not native_path.is_file() or not two_x_path.is_file():
                graph = ComfyUIClient.build_flashvsr_video_graph(
                    input_video_path=str(window["window_path"]),
                    local_range=int(local_range),
                    filename_prefix=f"lora_basepipe/run_{run_id}/yaw_{slot:02d}_flash_lr{int(local_range)}",
                )
                prompt_id = client.submit(graph)
                _update_character_generation_run(
                    run_id,
                    status="running_enhancement",
                    observed={"enhancement_prompt_id": prompt_id, "enhancement_yaw_slot": slot, "enhancement_variant": f"flashvsr_lr{int(local_range)}"},
                )
                entry = client.wait_history(prompt_id, timeout=3600.0, poll=2.0)
                media_ref = client.first_output_video(entry)
                video_path.write_bytes(client.download_media(media_ref))
                extract_video_center_frame(video_path, native_path)
                derive_2x_from_4x(native_path, two_x_path, source_size=source_size)
            key = f"flashvsr_lr{int(local_range)}_2x"
            comparison[key] = str(two_x_path)
            progress["jobs"].append({
                "yaw_slot": slot, "variant": f"flashvsr_lr{int(local_range)}", "prompt_id": prompt_id,
                "native_4x_path": str(native_path), "native_4x_sha256": _sha256(native_path),
                "derived_2x_path": str(two_x_path), "derived_2x_sha256": _sha256(two_x_path),
                "input_window_sha256": window["window_sha256"], "status": "completed",
            })
            progress["completed_jobs"] = len(progress["jobs"])
            _persist_enhancement_progress(run_id, progress)
        if request.get("include_realesrgan", True):
            real_path = slot_dir / "realesrgan_2x.png"
            real_video = slot_dir / "realesrgan_2x.mp4"
            prompt_id = "resumed"
            if not real_path.is_file():
                graph = ComfyUIClient.build_realesrgan_video_graph(
                    input_video_path=str(window["window_path"]),
                    filename_prefix=f"lora_basepipe/run_{run_id}/yaw_{slot:02d}_realesrgan",
                )
                prompt_id = client.submit(graph)
                _update_character_generation_run(
                    run_id, status="running_enhancement",
                    observed={"enhancement_prompt_id": prompt_id, "enhancement_yaw_slot": slot, "enhancement_variant": "realesrgan_x2"},
                )
                entry = client.wait_history(prompt_id, timeout=1800.0, poll=2.0)
                real_video.write_bytes(client.download_media(client.first_output_video(entry)))
                extract_video_center_frame(real_video, real_path)
            comparison["realesrgan_2x"] = str(real_path)
            progress["jobs"].append({
                "yaw_slot": slot, "variant": "realesrgan_x2", "prompt_id": prompt_id,
                "output_path": str(real_path), "output_sha256": _sha256(real_path),
                "input_window_sha256": window["window_sha256"], "status": "completed",
            })
            progress["completed_jobs"] = len(progress["jobs"])
            _persist_enhancement_progress(run_id, progress)
        comparison_rows.append(comparison)

    sheet = create_enhancement_comparison_sheet(comparison_rows, workspace / "comparison-contact-sheet.png", run_id=run_id)
    progress.update({"status": "completed", "comparison_rows": comparison_rows, "contact_sheet": sheet, "aesthetic_status": AESTHETIC_UNREVIEWED})
    conn = get_conn()
    try:
        row = conn.execute("SELECT stage_history_json, observed_json, output_manifest_json FROM basepipe_character_generation_runs WHERE id=?", (run_id,)).fetchone()
        output_manifest = json.loads(row["output_manifest_json"] or "{}")
        prior_extraction = output_manifest.get("frame_extraction") if isinstance(output_manifest.get("frame_extraction"), dict) else {}
        selected_assets: list[dict] = []
        yaw_names = ("front", "front-left", "left", "back-left", "back", "back-right", "right", "front-right")
        for comparison in comparison_rows:
            slot = int(comparison["yaw_slot"])
            original_path = Path(str(comparison["original"]))
            asset_key = f"h3_run_{run_id}_yaw_{slot:02d}_original"
            existing = conn.execute("SELECT id FROM basepipe_assets WHERE project_id=? AND asset_key=?", (project_id, asset_key)).fetchone()
            metadata = {
                "lineage": {
                    "run_id": run_id, "yaw_slot": slot, "yaw_target_degrees": float(comparison["yaw_target_degrees"]),
                    "source_video": str(source_video), "source_video_sha256": progress["source_sha256"],
                    "selection_policy": "deterministic_equal_yaw_circular_9_frame_windows",
                },
                "quality": {"review_status": "needs_review", "aesthetic_status": AESTHETIC_UNREVIEWED},
            }
            if existing is None:
                cur = conn.execute(
                    "INSERT INTO basepipe_assets(project_id, asset_key, file_path, content_sha256, asset_type, origin_kind, source_ref, review_status, training_enabled, metadata_json) "
                    "VALUES (?, ?, ?, ?, 'image', 'h3_frame', ?, 'needs_review', 0, ?)",
                    (project_id, asset_key, str(original_path), _sha256(original_path), f"character_generation:{run_id}", json.dumps(metadata, ensure_ascii=False)),
                )
                asset_id = int(cur.lastrowid)
            else:
                asset_id = int(existing["id"])
                conn.execute(
                    "UPDATE basepipe_assets SET file_path=?, content_sha256=?, review_status='needs_review', training_enabled=0, metadata_json=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                    (str(original_path), _sha256(original_path), json.dumps(metadata, ensure_ascii=False), asset_id),
                )
            version_specs = (
                ("flashvsr_lr9_2x", Path(str(comparison["flashvsr_lr9_2x"]))),
                ("flashvsr_lr11_2x", Path(str(comparison["flashvsr_lr11_2x"]))),
                ("realesrgan_x2", Path(str(comparison["realesrgan_2x"]))),
            )
            version_ids: dict[str, int] = {}
            for version_kind, version_path in version_specs:
                version = conn.execute("SELECT id FROM basepipe_asset_versions WHERE asset_id=? AND version_kind=? AND status='ready'", (asset_id, version_kind)).fetchone()
                version_metadata = {"run_id": run_id, "yaw_slot": slot, "engine_profile": version_kind, "aesthetic_review_state": AESTHETIC_UNREVIEWED}
                if version is None:
                    cur = conn.execute(
                        "INSERT INTO basepipe_asset_versions(asset_id, project_id, version_kind, file_path, content_sha256, prompt, seed, status, metadata_json) VALUES (?, ?, ?, ?, ?, '', 42, 'ready', ?)",
                        (asset_id, project_id, version_kind, str(version_path), _sha256(version_path), json.dumps(version_metadata, ensure_ascii=False)),
                    )
                    version_ids[version_kind] = int(cur.lastrowid)
                else:
                    version_ids[version_kind] = int(version["id"])
            with Image.open(original_path) as source_image:
                width, height = source_image.size
            selected_assets.append({
                "asset_id": asset_id, "file_path": str(original_path), "content_sha256": _sha256(original_path),
                "width": width, "height": height, "ordinal": slot, "review_status": "needs_review",
                "rejection_reason": "", "blur_status": "pass", "duplicate_status": "pass",
                "yaw_slot": slot, "yaw_target_degrees": float(comparison["yaw_target_degrees"]),
                "coverage": {"views": [yaw_names[slot]], "shots": ["full-body"], "poses": ["locked-neutral-standing"], "expressions": ["locked-neutral"], "backgrounds": ["locked-simple-studio"]},
                "enhancement_version_ids": version_ids, "aesthetic_status": AESTHETIC_UNREVIEWED,
            })
        output_manifest["candidate_frame_extraction"] = prior_extraction
        output_manifest["frame_extraction"] = {
            "frame_count": len(selected_assets), "clip_count": 1, "accepted_candidates": len(selected_assets), "rejected": 0,
            "frames": selected_assets, "source_clips": [str(source_video)], "selection_policy": "deterministic_equal_yaw_circular_9_frame_windows",
        }
        history = json.loads(row["stage_history_json"] or "[]")
        history = [item for item in history if item.get("stage") not in {"technical_selection", "enhancement"}]
        history.extend([
            {"stage": "technical_selection", "status": "completed", "artifact_manifest": {"windows": windows, "policy": "deterministic_equal_yaw_circular_9_frame_windows", "aesthetic_status": AESTHETIC_UNREVIEWED}},
            {"stage": "enhancement", "status": "completed", "artifact_manifest": progress},
        ])
        observed = json.loads(row["observed_json"] or "{}")
        observed.update({"waiting": False, "queued": False, "enhancement_completed_at": time.time(), "enhancement_progress": progress})
        observed.pop("wait_reason", None)
        output_manifest["enhancement"] = progress
        conn.execute(
            "UPDATE basepipe_character_generation_runs SET status='completed', current_stage='enhancement', stage_history_json=?, observed_json=?, output_manifest_json=?, error_detail='', updated_at=CURRENT_TIMESTAMP WHERE id=?",
            (json.dumps(history, ensure_ascii=False), json.dumps(observed, ensure_ascii=False), json.dumps(output_manifest, ensure_ascii=False), run_id),
        )
        conn.commit()
    finally:
        conn.close()


def _record_enhancement_wait_probe(run_id: int, memory: dict, detail: str) -> None:
    conn = get_conn()
    try:
        row = conn.execute("SELECT observed_json FROM basepipe_character_generation_runs WHERE id=? AND status='waiting_enhancement'", (run_id,)).fetchone()
        if row is None:
            return
        observed = json.loads(row["observed_json"] or "{}")
        observed.update({"waiting": True, "queued": False, "wait_stage": "enhancement", "wait_reason": detail, "last_enhancement_admission_probe_at": time.time(), "gpu_memory": memory})
        observed["enhancement_admission_probe_count"] = int(observed.get("enhancement_admission_probe_count", 0)) + 1
        conn.execute("UPDATE basepipe_character_generation_runs SET observed_json=?, updated_at=CURRENT_TIMESTAMP WHERE id=?", (json.dumps(observed, ensure_ascii=False), run_id))
        conn.commit()
    finally:
        conn.close()


def _enhancement_wait_monitor() -> None:
    global _ENHANCEMENT_WAIT_MONITOR_THREAD
    try:
        while True:
            conn = get_conn()
            try:
                waiting = conn.execute("SELECT * FROM basepipe_character_generation_runs WHERE status='waiting_enhancement' ORDER BY id LIMIT 1").fetchone()
                active = _active_gpu_work(conn)
            finally:
                conn.close()
            if waiting is None:
                return
            if active is not None:
                time.sleep(ENHANCEMENT_WAIT_POLL_SECONDS)
                continue
            run_id = int(waiting["id"])
            memory = _query_h3_gpu_memory()
            mapping = _h3_cuda_mapping()
            if not mapping.get("verified") or not memory.get("admission_ok"):
                detail = f"GPU mapping: {mapping.get('reason', 'unknown')}" if not mapping.get("verified") else str(_h3_gpu_check(memory)["detail"])
                _record_enhancement_wait_probe(run_id, memory, detail)
                time.sleep(ENHANCEMENT_WAIT_POLL_SECONDS)
                continue
            comfy_ready, comfy_detail = _ensure_h3_comfyui()
            if not comfy_ready:
                _record_enhancement_wait_probe(run_id, memory, comfy_detail)
                time.sleep(ENHANCEMENT_WAIT_POLL_SECONDS)
                continue
            admitted = _query_h3_gpu_memory()
            if not admitted.get("admission_ok"):
                detail = _release_managed_h3_after_wait_race(f"Enhancement Run #{run_id} post-launch admission changed", str(_h3_gpu_check(admitted)["detail"]))
                _record_enhancement_wait_probe(run_id, admitted, detail)
                time.sleep(ENHANCEMENT_WAIT_POLL_SECONDS)
                continue
            resolved = json.loads(waiting["resolved_json"] or "{}")
            request = resolved.get("enhancement_request")
            if not isinstance(request, dict):
                _update_character_generation_run(run_id, status="failed", observed={"waiting": False}, error="enhancement request is missing")
                continue
            conn = get_conn()
            try:
                conn.execute("BEGIN IMMEDIATE")
                active_now = _active_gpu_work(conn)
                current = conn.execute(
                    "SELECT observed_json FROM basepipe_character_generation_runs WHERE id=? AND status='waiting_enhancement'",
                    (run_id,),
                ).fetchone()
                running_observed = json.loads(current["observed_json"] or "{}") if current is not None else {}
                running_observed.update({
                    "waiting": False,
                    "queued": False,
                    "wait_stage": "enhancement",
                    "enhancement_admitted_at": time.time(),
                    "gpu_memory": admitted,
                })
                running_observed.pop("wait_reason", None)
                changed = 0 if active_now is not None else conn.execute(
                    "UPDATE basepipe_character_generation_runs SET status='running_enhancement', observed_json=?, error_detail='', updated_at=CURRENT_TIMESTAMP WHERE id=? AND status='waiting_enhancement'",
                    (json.dumps(running_observed, ensure_ascii=False), run_id),
                ).rowcount
                conn.commit()
            finally:
                conn.close()
            if not changed:
                _stop_managed_h3_comfyui(f"Enhancement Run #{run_id} lost admission transaction")
                continue
            failure_detail: str | None = None
            try:
                _execute_character_enhancement(run_id, int(waiting["project_id"]), request, admitted)
            except Exception as exc:  # noqa: BLE001
                failure_detail = _runtime_exception_detail(exc)
                _mark_character_enhancement_failed(run_id, failure_detail)
            finally:
                release = _stop_managed_h3_comfyui(f"Enhancement Run #{run_id} finished")
                _update_character_generation_run(
                    run_id,
                    status=_character_generation_status(run_id),
                    observed={"comfyui_release": release},
                    error=failure_detail or "",
                )
    finally:
        with _ENHANCEMENT_WAIT_MONITOR_LOCK:
            _ENHANCEMENT_WAIT_MONITOR_THREAD = None
        try:
            ensure_enhancement_wait_monitor()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Enhancement waiting monitor restart skipped: %s", exc)


def ensure_enhancement_wait_monitor() -> bool:
    global _ENHANCEMENT_WAIT_MONITOR_THREAD
    conn = get_conn()
    try:
        exists = conn.execute("SELECT 1 FROM basepipe_character_generation_runs WHERE status='waiting_enhancement' LIMIT 1").fetchone() is not None
    finally:
        conn.close()
    if not exists:
        return False
    with _ENHANCEMENT_WAIT_MONITOR_LOCK:
        if _ENHANCEMENT_WAIT_MONITOR_THREAD and _ENHANCEMENT_WAIT_MONITOR_THREAD.is_alive():
            return True
        _ENHANCEMENT_WAIT_MONITOR_THREAD = threading.Thread(target=_enhancement_wait_monitor, daemon=True, name="enhancement-wait-monitor")
        _ENHANCEMENT_WAIT_MONITOR_THREAD.start()
    return True


def recover_stale_enhancement_runs_on_startup() -> dict:
    conn = get_conn()
    try:
        rows = conn.execute("SELECT id, observed_json FROM basepipe_character_generation_runs WHERE status='running_enhancement'").fetchall()
        for row in rows:
            observed = json.loads(row["observed_json"] or "{}")
            observed.update({"waiting": True, "queued": False, "wait_stage": "enhancement", "wait_reason": "Backend restart recovery"})
            conn.execute("UPDATE basepipe_character_generation_runs SET status='waiting_enhancement', observed_json=?, updated_at=CURRENT_TIMESTAMP WHERE id=?", (json.dumps(observed, ensure_ascii=False), int(row["id"])))
        conn.commit()
        return {"recovered_run_ids": [int(row["id"]) for row in rows]}
    finally:
        conn.close()


@router.post("/projects/{project_id}/character-generation/runs/{run_id}/enhance")
def queue_character_generation_enhancement(project_id: int, run_id: int, payload: CharacterGenerationEnhancementIn) -> dict:
    """Queue technical enhancement comparisons without making aesthetic choices."""
    if payload.gpu_device_id != 1:
        raise HTTPException(status_code=400, detail="GPU0は禁止です。Enhancementは物理GPU1固定です")
    if sorted(payload.local_ranges) != [9, 11]:
        raise HTTPException(status_code=400, detail="FlashVSR比較はlocal_range 9/11固定です")
    conn = get_conn()
    try:
        row = conn.execute("SELECT * FROM basepipe_character_generation_runs WHERE id=? AND project_id=?", (run_id, project_id)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="character generation run not found")
        if row["status"] in {"waiting_enhancement", "running_enhancement"}:
            return {"run_id": run_id, "project_id": project_id, "status": row["status"], "queued": True, "message": "既存のEnhancement待機/実行を継続しています"}
        if row["status"] not in {"completed", "failed"} or row["current_stage"] not in {"duplicate_filter", "enhancement"}:
            raise HTTPException(status_code=409, detail=f"H3抽出完了後のRunだけEnhancementできます（現在: {row['status']} / {row['current_stage']}）")
        manifest = json.loads(row["output_manifest_json"] or "{}")
        source = _enhancement_source_video(manifest)
        resolved = json.loads(row["resolved_json"] or "{}")
        resolved["enhancement_request"] = payload.model_dump()
        observed = json.loads(row["observed_json"] or "{}")
        prior_progress = observed.get("enhancement_progress")
        if isinstance(prior_progress, dict):
            prior_progress = {**prior_progress, "status": "waiting"}
        observed.update({
            "waiting": True, "queued": False, "wait_stage": "enhancement", "auto_start_when_ready": True,
            "enhancement_admission_probe_count": 0, "enhancement_source_video": str(source),
            "aesthetic_status": AESTHETIC_UNREVIEWED,
        })
        if isinstance(prior_progress, dict):
            observed["enhancement_progress"] = prior_progress
        observed.pop("wait_reason", None)
        observed.pop("enhancement_failure", None)
        conn.execute("UPDATE basepipe_character_generation_runs SET status='waiting_enhancement', resolved_json=?, observed_json=?, error_detail='', updated_at=CURRENT_TIMESTAMP WHERE id=?", (json.dumps(resolved, ensure_ascii=False), json.dumps(observed, ensure_ascii=False), run_id))
        conn.commit()
    finally:
        conn.close()
    ensure_enhancement_wait_monitor()
    return {"run_id": run_id, "project_id": project_id, "status": "waiting_enhancement", "queued": True, "aesthetic_status": AESTHETIC_UNREVIEWED, "message": "技術比較を物理GPU1の永続待機キューへ追加しました"}


def _current_character_generation_frame_ids(run_row) -> set[int] | None:
    """Return the active extraction manifest IDs, or ``None`` for legacy Runs.

    Frame extraction is intentionally retryable. Every retry preserves older
    Asset rows for provenance, so ``source_ref=character_generation:<id>`` is
    broader than the active candidate set. Once a Run has a frame manifest,
    every downstream stage must use that exact manifest rather than stale rows.
    """
    try:
        raw_manifest = run_row["output_manifest_json"]
    except (KeyError, IndexError, TypeError):
        raw_manifest = None
    if isinstance(raw_manifest, str):
        try:
            manifest = json.loads(raw_manifest or "{}")
        except (TypeError, ValueError):
            manifest = {}
    elif isinstance(raw_manifest, dict):
        manifest = raw_manifest
    else:
        manifest = {}
    extraction = manifest.get("frame_extraction")
    if not isinstance(extraction, dict) or "frames" not in extraction:
        return None
    frames = extraction.get("frames")
    if not isinstance(frames, list):
        return set()
    result: set[int] = set()
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        try:
            result.add(int(frame["asset_id"]))
        except (KeyError, TypeError, ValueError):
            continue
    return result


@router.post("/projects/{project_id}/character-generation/runs/{run_id}/identity-review")
def complete_character_generation_identity_review(project_id: int, run_id: int, payload: CharacterGenerationIdentityReviewIn) -> dict:
    """Apply explicit human decisions to every H3 frame and seal Identity Review."""
    _require_user_aesthetic_confirmation(payload.human_confirmed)
    conn = get_conn()
    try:
        row = conn.execute("SELECT * FROM basepipe_character_generation_runs WHERE id = ? AND project_id = ?", (run_id, project_id)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="character generation run not found")
        requested = json.loads(row["requested_json"] or "{}")
        enhancement_required = requested.get("coverage_preset") == "turntable"
        if row["current_stage"] != "enhancement" and not (row["current_stage"] == "duplicate_filter" and not enhancement_required):
            raise HTTPException(status_code=409, detail="Turntable Runは技術選抜とEnhancement比較が完了してからIdentity Reviewしてください")
        decisions = {item.asset_id: item.review_status for item in payload.decisions}
        current_frame_ids = _current_character_generation_frame_ids(row)
        if current_frame_ids is not None and current_frame_ids != set(decisions):
            raise HTTPException(status_code=400, detail="Identity Reviewは現行manifestの全Frame Assetを明示判定してください")
        marks = ",".join("?" for _ in decisions)
        assets = conn.execute(f"SELECT id, review_status, metadata_json FROM basepipe_assets WHERE project_id = ? AND source_ref = ? AND origin_kind = 'h3_frame' AND id IN ({marks})", [project_id, f"character_generation:{run_id}", *decisions]).fetchall()
        if len(assets) != len(decisions):
            raise HTTPException(status_code=400, detail="指定されたFrame AssetがこのRunに属していません")
        if current_frame_ids is None:
            all_frames = conn.execute("SELECT id FROM basepipe_assets WHERE project_id = ? AND source_ref = ? AND origin_kind = 'h3_frame'", (project_id, f"character_generation:{run_id}")).fetchall()
            if {int(item["id"]) for item in all_frames} != set(decisions):
                raise HTTPException(status_code=400, detail="Identity ReviewはRunの全Frame Assetを明示判定してください")
        approved = sum(status == "approved" for status in decisions.values())
        if approved < 1:
            raise HTTPException(status_code=400, detail="Identity Reviewにはapproved Frameが1件以上必要です")
        asset_metadata = {int(asset["id"]): json.loads(asset["metadata_json"] or "{}") for asset in assets}
        reviewed_at = time.time()
        for asset_id, status in decisions.items():
            metadata = asset_metadata[asset_id]
            metadata["identity_review_state"] = USER_APPROVED if status == "approved" else USER_REJECTED
            metadata["identity_reviewed_by"] = "user"
            metadata["identity_reviewed_at"] = reviewed_at
            conn.execute(
                "UPDATE basepipe_assets SET review_status = ?, metadata_json = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (status, json.dumps(metadata, ensure_ascii=False), asset_id),
            )
        history = json.loads(row["stage_history_json"] or "[]")
        manifest = {
            "frame_count": len(decisions),
            "approved": approved,
            "rejected": len(decisions) - approved,
            "decisions": [
                {"asset_id": int(asset_id), "review_status": status}
                for asset_id, status in sorted(decisions.items())
            ],
            "note": payload.note,
            "aesthetic_review_state": USER_APPROVED,
            "reviewed_by": "user",
            "reviewed_at": reviewed_at,
            "policy": "user is the sole aesthetic approver; metrics and agents cannot seal Identity Review",
        }
        history.append({"stage": "identity_review", "status": "completed", "artifact_manifest": manifest})
        output_manifest = json.loads(row["output_manifest_json"] or "{}")
        output_manifest["identity_review"] = manifest
        conn.execute(
            "UPDATE basepipe_character_generation_runs SET current_stage='identity_review', stage_history_json=?, "
            "output_manifest_json=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
            (json.dumps(history, ensure_ascii=False), json.dumps(output_manifest, ensure_ascii=False), run_id),
        )
        conn.commit()
        return {"run_id": run_id, "project_id": project_id, "current_stage": "identity_review", "manifest": manifest}
    except HTTPException:
        conn.rollback()
        raise
    finally:
        conn.close()


@router.post("/projects/{project_id}/character-generation/runs/{run_id}/identity-review/reopen")
def reopen_character_generation_identity_review(project_id: int, run_id: int) -> dict:
    """Return a preserved Run to user-owned Frame review without deleting artifacts."""
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT * FROM basepipe_character_generation_runs WHERE id=? AND project_id=?",
            (run_id, project_id),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="character generation run not found")
        if row["status"] in {"waiting", "queued", "running", "waiting_qwen", "running_qwen"}:
            raise HTTPException(status_code=409, detail="GPU待機・実行中はReviewを開き直せません。先に待機を取り消してください")
        if row["current_stage"] not in {"duplicate_filter", "identity_review", "qwen_correction", "balancing", "captioning"}:
            raise HTTPException(status_code=409, detail=f"現在の{row['current_stage']}段階からIdentity Reviewは開き直せません")

        current_frame_ids = _current_character_generation_frame_ids(row)
        if current_frame_ids is None:
            assets = conn.execute(
                "SELECT id, review_status, metadata_json FROM basepipe_assets WHERE project_id=? AND source_ref=? AND origin_kind='h3_frame' ORDER BY id",
                (project_id, f"character_generation:{run_id}"),
            ).fetchall()
        elif current_frame_ids:
            marks = ",".join("?" for _ in current_frame_ids)
            assets = conn.execute(
                f"SELECT id, review_status, metadata_json FROM basepipe_assets WHERE project_id=? AND source_ref=? AND origin_kind='h3_frame' AND id IN ({marks}) ORDER BY id",
                (project_id, f"character_generation:{run_id}", *sorted(current_frame_ids)),
            ).fetchall()
        else:
            assets = []
        for asset in assets:
            metadata = json.loads(asset["metadata_json"] or "{}")
            previous_review_status = str(asset["review_status"] or "needs_review")
            for key in (
                "identity_review_state", "identity_reviewed_by", "identity_reviewed_at",
                "training_input_choice", "selected_qwen_version_id", "training_input_reviewed_by",
                "training_input_reviewed_at", "training_input_review_note", "training_input_review_choice",
                "training_input_review_version_id",
            ):
                metadata.pop(key, None)
            metadata["identity_review_state"] = AESTHETIC_UNREVIEWED
            metadata["previous_untrusted_review_status"] = previous_review_status
            metadata["previous_untrusted_review_status_authoritative"] = False
            metadata["training_input_review_state"] = AESTHETIC_UNREVIEWED
            conn.execute(
                "UPDATE basepipe_assets SET review_status='needs_review', training_enabled=0, metadata_json=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                (json.dumps(metadata, ensure_ascii=False), int(asset["id"])),
            )

        output_manifest = json.loads(row["output_manifest_json"] or "{}")
        for stale_stage in ("identity_review", "balancing", "captioning", "snapshot"):
            output_manifest.pop(stale_stage, None)
        extraction = output_manifest.get("frame_extraction")
        if isinstance(extraction, dict) and isinstance(extraction.get("frames"), list):
            active_ids = {int(asset["id"]) for asset in assets}
            for frame in extraction["frames"]:
                if not isinstance(frame, dict):
                    continue
                try:
                    frame_id = int(frame["asset_id"])
                except (KeyError, TypeError, ValueError):
                    continue
                if frame_id in active_ids:
                    frame["review_status"] = "needs_review"
        preserved_versions = conn.execute(
            "SELECT COUNT(*) AS count FROM basepipe_asset_versions WHERE project_id=? AND asset_id IN "
            f"({','.join('?' for _ in assets) or 'NULL'}) AND version_kind='qwen_correction' AND status='ready'",
            (project_id, *(int(asset["id"]) for asset in assets)),
        ).fetchone()
        history = json.loads(row["stage_history_json"] or "[]")
        history.append({
            "stage": "identity_review_reopened",
            "status": "completed",
            "artifact_manifest": {
                "aesthetic_review_state": AESTHETIC_UNREVIEWED,
                "frame_count": len(assets),
                "preserved_qwen_version_count": int(preserved_versions["count"] if preserved_versions else 0),
                "policy": "previous decisions are not trusted without current user confirmation",
            },
        })
        observed = json.loads(row["observed_json"] or "{}")
        observed.update({
            "waiting": False,
            "queued": False,
            "auto_start_when_ready": False,
            "wait_stage": "",
            "aesthetic_review_state": AESTHETIC_UNREVIEWED,
        })
        conn.execute(
            "UPDATE basepipe_character_generation_runs SET status='completed', current_stage='duplicate_filter', "
            "output_manifest_json=?, stage_history_json=?, observed_json=?, error_detail='', updated_at=CURRENT_TIMESTAMP WHERE id=?",
            (json.dumps(output_manifest, ensure_ascii=False), json.dumps(history, ensure_ascii=False), json.dumps(observed, ensure_ascii=False), run_id),
        )
        conn.commit()
        return {
            "run_id": run_id,
            "project_id": project_id,
            "status": "completed",
            "current_stage": "duplicate_filter",
            "aesthetic_review_state": AESTHETIC_UNREVIEWED,
            "frame_count": len(assets),
            "preserved_qwen_version_count": int(preserved_versions["count"] if preserved_versions else 0),
        }
    except HTTPException:
        conn.rollback()
        raise
    finally:
        conn.close()


@router.post("/projects/{project_id}/character-generation/runs/{run_id}/qwen-correction")
def prepare_character_generation_qwen_correction(project_id: int, run_id: int, payload: CharacterGenerationQwenCorrectionIn) -> dict:
    """Plan or execute Qwen correction without overwriting H3 frames.

    The explicit ``execute`` path reuses the general Asset transform runtime so
    Character Generation and manual Asset editing have the same GPU/readiness
    gates and immutable-version semantics.
    """
    regeneration = payload.mode == "dataset_regeneration"
    consistency = payload.mode == "consistency_regeneration"
    quality_profile = QWEN_CONSISTENCY_PROFILE if consistency else (QWEN_DATASET_REGEN_PROFILE if regeneration else SELECTIVE_HQ_PROFILE)
    if regeneration or consistency:
        _require_user_aesthetic_confirmation(payload.human_confirmed)
    conn = get_conn()
    try:
        row = conn.execute("SELECT * FROM basepipe_character_generation_runs WHERE id = ? AND project_id = ?", (run_id, project_id)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="character generation run not found")
        if row["status"] in {"waiting_qwen", "running_qwen"}:
            raise HTTPException(status_code=409, detail="このRunのQwen補正は既に待機中または実行中です")
        if row["status"] == "cancelled":
            raise HTTPException(status_code=409, detail="取消済みRunではQwen補正を開始しません。ユーザーがIdentity Reviewを開き直してください")
        allowed_stages = {"duplicate_filter", "identity_review", "qwen_correction", "balancing", "captioning"} if (regeneration or consistency) else {"identity_review", "qwen_correction", "balancing", "captioning"}
        if row["current_stage"] not in allowed_stages:
            raise HTTPException(status_code=409, detail="Snapshot固定前のIdentity/Qwen/Balancing/Captioning段階だけQwen補正を再実行できます")
        identity_review = (json.loads(row["output_manifest_json"] or "{}").get("identity_review") or {})
        if not (regeneration or consistency) and (identity_review.get("aesthetic_review_state") != USER_APPROVED or identity_review.get("reviewed_by") != "user"):
            raise HTTPException(status_code=409, detail="Identity ReviewはAESTHETIC_UNREVIEWEDです。ユーザー本人の再確認が必要です")
        if payload.gpu_device_id != 1:
            raise HTTPException(status_code=400, detail="Qwen補正もGPU1固定です。GPU0は禁止です")
        asset_ids = list(dict.fromkeys(payload.asset_ids))
        current_frame_ids = _current_character_generation_frame_ids(row)
        if current_frame_ids is not None and not set(asset_ids).issubset(current_frame_ids):
            raise HTTPException(status_code=400, detail="Qwen補正対象は現行manifestのapproved H3 Frameだけです")
        marks = ",".join("?" for _ in asset_ids)
        assets = conn.execute(f"SELECT id, file_path, content_sha256, review_status, metadata_json FROM basepipe_assets WHERE project_id = ? AND source_ref = ? AND origin_kind = 'h3_frame' AND id IN ({marks})", [project_id, f"character_generation:{run_id}", *asset_ids]).fetchall()
        if len(assets) != len(asset_ids) or (not (regeneration or consistency) and any(item["review_status"] != "approved" for item in assets)):
            raise HTTPException(status_code=400, detail="Qwen補正対象はこのRunのapproved H3 Frameだけです")
        master_references = _qwen_master_references(project_id, run_id)
        regeneration_bindings = _qwen_regeneration_inputs(run_id, list(assets), master_references) if regeneration else {}
        consistency_bindings = _qwen_consistency_inputs(run_id, list(assets), master_references) if consistency else {}
        resolved_prompt = _qwen_resolved_prompt(payload.prompt, master_references[:2]) if not (regeneration or consistency) else ""
        if row["current_stage"] in {"balancing", "captioning"}:
            history = [
                entry for entry in json.loads(row["stage_history_json"] or "[]")
                if entry.get("stage") not in {"balancing", "captioning", "snapshot"}
            ]
            output_manifest = json.loads(row["output_manifest_json"] or "{}")
            for stale_stage in ("balancing", "captioning", "snapshot"):
                output_manifest.pop(stale_stage, None)
            conn.execute(
                "UPDATE basepipe_character_generation_runs SET current_stage='qwen_correction', status='completed', "
                "stage_history_json=?, output_manifest_json=?, error_detail='', updated_at=CURRENT_TIMESTAMP WHERE id=?",
                (json.dumps(history, ensure_ascii=False), json.dumps(output_manifest, ensure_ascii=False), run_id),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM basepipe_character_generation_runs WHERE id=? AND project_id=?",
                (run_id, project_id),
            ).fetchone()
        if payload.execute and payload.queue_if_busy:
            resolved = json.loads(row["resolved_json"] or "{}")
            resolved["qwen_request"] = payload.model_dump()
            resolved["qwen_master_references"] = master_references
            resolved["qwen_resolved_prompt"] = resolved_prompt
            resolved["qwen_regeneration_bindings"] = regeneration_bindings
            resolved["qwen_consistency_bindings"] = consistency_bindings
            observed = json.loads(row["observed_json"] or "{}")
            observed.update({
                "waiting": True,
                "queued": False,
                "wait_stage": "qwen_correction",
                "wait_reason": "GPU1安全条件、管理対象ComfyUI、Qwen-Image-Edit-2511モデルを再評価しています",
                "qwen_admission_probe_count": 0,
            })
            changed = conn.execute(
                "UPDATE basepipe_character_generation_runs SET status='waiting_qwen', resolved_json=?, observed_json=?, error_detail='', updated_at=CURRENT_TIMESTAMP WHERE id=? AND project_id=? AND current_stage IN ('duplicate_filter','identity_review','qwen_correction') AND status NOT IN ('waiting_qwen','running_qwen')",
                (json.dumps(resolved, ensure_ascii=False), json.dumps(observed, ensure_ascii=False), run_id, project_id),
            ).rowcount
            if changed != 1:
                conn.rollback()
                raise HTTPException(status_code=409, detail="Qwen待機投入前にRun状態が変わりました")
            conn.commit()
            ensure_qwen_wait_monitor()
            return {
                "run_id": run_id,
                "project_id": project_id,
                "status": "waiting",
                "run_status": "waiting_qwen",
                "execute": True,
                "gpu_device_id": 1,
                "versions": [],
                "message": "Qwen-Image-Edit-2511をGPU1待機キューへ追加しました。外部プロセスには触れません。",
            }
        if payload.execute:
            memory = _query_qwen_gpu_memory()
            if not memory.get("admission_ok"):
                detail = memory.get("free_mb")
                suffix = f"GPU1空きVRAM {detail} MiB / 必要 {QWEN_MIN_FREE_VRAM_MB} MiB" if detail is not None else str(memory.get("error", "GPU1 VRAMを実測できません"))
                raise HTTPException(status_code=409, detail=f"Qwen補正を開始しません: {suffix}")
            comfy_ready, comfy_detail = _ensure_h3_comfyui()
            if not comfy_ready:
                raise HTTPException(status_code=409, detail=f"Qwen補正を開始しません: {comfy_detail}")
            memory = _query_qwen_gpu_memory()
            if not memory.get("admission_ok"):
                raise HTTPException(status_code=409, detail=f"Qwen補正を開始しません: {_h3_gpu_check(memory)['detail']}")
            conn.execute("BEGIN IMMEDIATE")
            active_now = _active_gpu_work(conn)
            if active_now is not None:
                conn.rollback()
                raise HTTPException(status_code=409, detail=f"GPU1では{active_now['kind']} Run #{active_now['id']}が実行中です")
            observed = json.loads(row["observed_json"] or "{}")
            observed.update({
                "waiting": False,
                "queued": True,
                "wait_stage": "qwen_correction",
                "wait_reason": "",
                "qwen_admitted_at": time.time(),
                "gpu_memory": memory,
            })
            changed = conn.execute(
                "UPDATE basepipe_character_generation_runs SET status='running_qwen', observed_json=?, updated_at=CURRENT_TIMESTAMP "
                "WHERE id=? AND project_id=? AND current_stage IN ('duplicate_filter','identity_review','qwen_correction') AND status NOT IN ('queued','running','waiting_qwen','running_qwen')",
                (json.dumps(observed, ensure_ascii=False), run_id, project_id),
            ).rowcount
            if changed != 1:
                conn.rollback()
                raise HTTPException(status_code=409, detail="Qwen実行開始前にRun状態が変わりました")
            conn.commit()
            try:
                return _execute_qwen_correction_runtime(run_id, project_id, payload, memory)
            except Exception as exc:  # noqa: BLE001 - persist the exact direct Runtime failure
                _update_character_generation_run(run_id, status="failed", observed={"waiting": False, "queued": False}, error=_runtime_exception_detail(exc))
                raise
            finally:
                release = _stop_managed_h3_comfyui(f"Qwen Run #{run_id} finished")
                _update_character_generation_run(run_id, status=_character_generation_status(run_id), observed={"comfyui_release": release})
        planned = []
        for asset in assets:
            binding = consistency_bindings.get(int(asset["id"])) or regeneration_bindings.get(int(asset["id"]))
            asset_prompt = _qwen_consistency_prompt(payload.prompt, binding) if consistency else (_qwen_dataset_regeneration_prompt(payload.prompt, binding) if regeneration else resolved_prompt)
            plan_kind = "qwen_consistency_regeneration_plan" if consistency else ("qwen_dataset_regeneration_plan" if regeneration else "qwen_correction_plan")
            parent = conn.execute("SELECT id FROM basepipe_asset_versions WHERE asset_id = ? ORDER BY id DESC LIMIT 1", (asset["id"],)).fetchone()
            cur = conn.execute("INSERT INTO basepipe_asset_versions(asset_id, project_id, parent_version_id, version_kind, file_path, content_sha256, prompt, seed, status, metadata_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'planned', ?)", (asset["id"], project_id, int(parent["id"]) if parent else None, plan_kind, asset["file_path"], asset["content_sha256"], asset_prompt, payload.seed, json.dumps({"gpu_device_id": 1, "source_asset": asset["file_path"], "master_references": master_references, "reference_contract": binding or {}, "requested_prompt": payload.prompt, "resolved_prompt": asset_prompt, "operation_mode": payload.mode, "execution": "not_started", "quality_profile": quality_profile, "selection_gate": "user-directed consistency bakeoff" if consistency else ("user-confirmed dataset regeneration route" if regeneration else "user-approved frame asset IDs only"), "aesthetic_state": AESTHETIC_UNREVIEWED, "lineage_policy": "source frame is immutable"}, ensure_ascii=False)))
            planned.append({"asset_id": int(asset["id"]), "version_id": int(cur.lastrowid), "status": "planned", "parent_file_path": asset["file_path"], "prompt": asset_prompt, "seed": payload.seed, "quality_profile": quality_profile, "aesthetic_state": AESTHETIC_UNREVIEWED})
        conn.commit()
        return {"run_id": run_id, "project_id": project_id, "status": "planned", "execute": False, "gpu_device_id": 1, "quality_profile": quality_profile, "selection_scope": asset_ids, "versions": planned, "message": "中間yawの一貫性比較計画を保存しました。H3 RGBと四面図パックを入力せず、単一の正本cardinal画像をMultiple-Angles LoRAで回転します。" if consistency else ("Qwen静止画再生成計画を保存しました。元Frameは構図ガイドとして不変のまま保持されます。" if regeneration else "選択Frameだけの高品質化計画を保存しました。元Frameは変更されず、実行済み扱いにもなりません。")}
    except HTTPException:
        conn.rollback()
        raise
    finally:
        conn.close()


@router.post("/projects/{project_id}/character-generation/runs/{run_id}/balance")
def complete_character_generation_balance(project_id: int, run_id: int) -> dict:
    """Seal balancing evidence only after every approved H3 frame has Qwen output."""
    conn = get_conn()
    try:
        row = conn.execute("SELECT * FROM basepipe_character_generation_runs WHERE id = ? AND project_id = ?", (run_id, project_id)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="character generation run not found")
        if row["status"] in {"waiting_qwen", "running_qwen"}:
            raise HTTPException(status_code=409, detail="Qwen補正の待機・実行中はBalancingできません")
        if row["status"] == "cancelled":
            raise HTTPException(status_code=409, detail="取消済みRunはBalancingできません。ユーザーが審美Reviewを再開してください")
        if row["current_stage"] != "qwen_correction":
            raise HTTPException(status_code=409, detail="Identity Review後のRunだけBalancingできます")
        output_manifest = json.loads(row["output_manifest_json"] or "{}")
        identity_review = output_manifest.get("identity_review") or {}
        if identity_review.get("aesthetic_review_state") != USER_APPROVED or identity_review.get("reviewed_by") != "user":
            raise HTTPException(status_code=409, detail="Identity ReviewはAESTHETIC_UNREVIEWEDです。ユーザー本人の再確認が必要です")
        current_frame_ids = _current_character_generation_frame_ids(row)
        if current_frame_ids is None:
            assets = conn.execute("SELECT id, file_path, content_sha256, review_status, metadata_json FROM basepipe_assets WHERE project_id = ? AND source_ref = ? AND origin_kind = 'h3_frame'", (project_id, f"character_generation:{run_id}")).fetchall()
        elif current_frame_ids:
            marks = ",".join("?" for _ in current_frame_ids)
            assets = conn.execute(
                f"SELECT id, file_path, content_sha256, review_status, metadata_json FROM basepipe_assets WHERE project_id = ? AND source_ref = ? AND origin_kind = 'h3_frame' AND id IN ({marks})",
                (project_id, f"character_generation:{run_id}", *sorted(current_frame_ids)),
            ).fetchall()
        else:
            assets = []
        approved = [asset for asset in assets if asset["review_status"] == "approved"]
        if not approved:
            raise HTTPException(status_code=400, detail="Balancing対象のapproved Frameがありません")
        unconfirmed_identity = [
            int(asset["id"])
            for asset in approved
            if json.loads(asset["metadata_json"] or "{}").get("identity_review_state") != USER_APPROVED
            or json.loads(asset["metadata_json"] or "{}").get("identity_reviewed_by") != "user"
        ]
        if unconfirmed_identity:
            raise HTTPException(status_code=409, detail=f"ユーザー未確認のIdentity Frame: {', '.join(map(str, unconfirmed_identity))}")
        selections = {int(asset["id"]): _training_input_selection(conn, asset) for asset in approved}
        missing = [asset_id for asset_id, selection in selections.items() if selection is None]
        if missing:
            raise HTTPException(status_code=409, detail=f"Training Input未選択のFrame: {', '.join(map(str, missing))}")
        distribution = Counter(str(selection["input_source"]) for selection in selections.values() if selection)
        coverage_distribution: dict[str, Counter] = {}
        for asset in approved:
            metadata = json.loads(asset["metadata_json"] or "{}")
            coverage = (metadata.get("lineage") or {}).get("coverage") or {}
            if not isinstance(coverage, dict):
                continue
            for dimension, values in coverage.items():
                if not isinstance(values, list):
                    continue
                bucket = coverage_distribution.setdefault(str(dimension), Counter())
                bucket.update(str(value) for value in values if str(value).strip())
        manifest = {
            "source_frame_count": len(assets),
            "approved_count": len(approved),
            "rejected_count": len(assets) - len(approved),
            "selections": [
                {
                    "asset_id": int(asset_id),
                    "input_source": str(selection["input_source"]),
                    "file_path": str(selection["file_path"]),
                    "content_sha256": str(selection["content_sha256"]),
                    "qwen_version_id": selection.get("id") if selection["input_source"] == "qwen_version" else None,
                }
                for asset_id, selection in sorted(selections.items())
                if selection is not None
            ],
            "input_distribution": dict(distribution),
            "coverage_distribution": {dimension: dict(counts) for dimension, counts in coverage_distribution.items()},
            "policy": "approved H3 frames with explicit Original or Qwen acceptance",
        }
        history = json.loads(row["stage_history_json"] or "[]")
        history.append({"stage": "balancing", "status": "completed", "artifact_manifest": manifest})
        output_manifest["balancing"] = manifest
        conn.execute(
            "UPDATE basepipe_character_generation_runs SET current_stage='balancing', stage_history_json=?, "
            "output_manifest_json=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
            (json.dumps(history, ensure_ascii=False), json.dumps(output_manifest, ensure_ascii=False), run_id),
        )
        conn.commit()
        return {"run_id": run_id, "project_id": project_id, "current_stage": "balancing", "manifest": manifest}
    except HTTPException:
        conn.rollback()
        raise
    finally:
        conn.close()


@router.post("/projects/{project_id}/character-generation/runs/{run_id}/qwen-correction/skip")
def skip_character_generation_qwen(project_id: int, run_id: int) -> dict:
    """Advance without GPU correction only when every approved frame explicitly uses Original."""
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT current_stage, stage_history_json, output_manifest_json FROM basepipe_character_generation_runs WHERE id=? AND project_id=?",
            (run_id, project_id),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="character generation run not found")
        if row["current_stage"] != "identity_review":
            raise HTTPException(status_code=409, detail="Identity Review直後のRunだけQwenをスキップできます")
        output_manifest = json.loads(row["output_manifest_json"] or "{}")
        identity_review = output_manifest.get("identity_review") or {}
        if identity_review.get("aesthetic_review_state") != USER_APPROVED or identity_review.get("reviewed_by") != "user":
            raise HTTPException(status_code=409, detail="Identity ReviewはAESTHETIC_UNREVIEWEDです。ユーザー本人の再確認が必要です")
        current_frame_ids = _current_character_generation_frame_ids(row)
        if current_frame_ids is None:
            assets = conn.execute(
                "SELECT id, file_path, content_sha256, metadata_json FROM basepipe_assets "
                "WHERE project_id=? AND source_ref=? AND origin_kind='h3_frame' AND review_status='approved' ORDER BY id",
                (project_id, f"character_generation:{run_id}"),
            ).fetchall()
        elif current_frame_ids:
            marks = ",".join("?" for _ in current_frame_ids)
            assets = conn.execute(
                f"SELECT id, file_path, content_sha256, metadata_json FROM basepipe_assets "
                f"WHERE project_id=? AND source_ref=? AND origin_kind='h3_frame' AND review_status='approved' AND id IN ({marks}) ORDER BY id",
                (project_id, f"character_generation:{run_id}", *sorted(current_frame_ids)),
            ).fetchall()
        else:
            assets = []
        if not assets:
            raise HTTPException(status_code=400, detail="approved Frameがありません")
        selections = {int(asset["id"]): _training_input_selection(conn, asset) for asset in assets}
        invalid = [asset_id for asset_id, selection in selections.items() if not selection or selection["input_source"] != "original"]
        if invalid:
            raise HTTPException(status_code=409, detail=f"Original未採用のFrame: {', '.join(map(str, invalid))}")
        manifest = {
            "status": "skipped_by_user",
            "original_count": len(assets),
            "qwen_count": 0,
            "policy": "explicit Original acceptance for every approved frame",
        }
        history = json.loads(row["stage_history_json"] or "[]")
        history.append({"stage": "qwen_correction", "status": "skipped", "artifact_manifest": manifest})
        conn.execute(
            "UPDATE basepipe_character_generation_runs SET current_stage='qwen_correction', stage_history_json=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
            (json.dumps(history, ensure_ascii=False), run_id),
        )
        conn.commit()
        return {"run_id": run_id, "project_id": project_id, "current_stage": "qwen_correction", "manifest": manifest}
    except HTTPException:
        conn.rollback()
        raise
    finally:
        conn.close()


@router.post("/projects/{project_id}/character-generation/runs/{run_id}/captioning")
def complete_character_generation_captioning(project_id: int, run_id: int, payload: CharacterGenerationCaptionIn) -> dict:
    """Persist explicit captions for all approved frames and seal Captioning."""
    conn = get_conn()
    try:
        row = conn.execute("SELECT * FROM basepipe_character_generation_runs WHERE id = ? AND project_id = ?", (run_id, project_id)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="character generation run not found")
        if row["current_stage"] != "balancing":
            raise HTTPException(status_code=409, detail="Balancing完了後にCaptioningできます")
        captions: dict[int, str] = {}
        for item in payload.captions:
            if not isinstance(item, dict) or not isinstance(item.get("asset_id"), int) or not str(item.get("caption", "")).strip():
                raise HTTPException(status_code=400, detail="captioningはasset_idと空でないcaptionが必要です")
            captions[int(item["asset_id"])] = str(item["caption"]).strip()
        marks = ",".join("?" for _ in captions)
        assets = conn.execute(f"SELECT id, review_status FROM basepipe_assets WHERE project_id = ? AND source_ref = ? AND origin_kind = 'h3_frame' AND id IN ({marks})", [project_id, f"character_generation:{run_id}", *captions]).fetchall()
        current_frame_ids = _current_character_generation_frame_ids(row)
        if current_frame_ids is None:
            approved = conn.execute("SELECT id FROM basepipe_assets WHERE project_id = ? AND source_ref = ? AND origin_kind = 'h3_frame' AND review_status = 'approved'", (project_id, f"character_generation:{run_id}")).fetchall()
        elif current_frame_ids:
            current_marks = ",".join("?" for _ in current_frame_ids)
            approved = conn.execute(
                f"SELECT id FROM basepipe_assets WHERE project_id = ? AND source_ref = ? AND origin_kind = 'h3_frame' AND review_status = 'approved' AND id IN ({current_marks})",
                (project_id, f"character_generation:{run_id}", *sorted(current_frame_ids)),
            ).fetchall()
        else:
            approved = []
        if {int(asset["id"]) for asset in assets} != set(captions) or {int(asset["id"]) for asset in approved} != set(captions):
            raise HTTPException(status_code=400, detail="CaptioningはRun内の全approved Frameを指定してください")
        for asset_id, caption in captions.items():
            conn.execute("UPDATE basepipe_assets SET caption=?, caption_source='h3_captioning', caption_original=?, training_input=?, training_input_source='h3_captioning', training_enabled=1, updated_at=CURRENT_TIMESTAMP WHERE id=?", (caption, caption, caption, asset_id))
        manifest = {
            "captioned_count": len(captions),
            "captions": [
                {"asset_id": int(asset_id), "caption": caption}
                for asset_id, caption in sorted(captions.items())
            ],
            "source": "explicit_captioning",
            "note": payload.note,
        }
        history = json.loads(row["stage_history_json"] or "[]")
        history.append({"stage": "captioning", "status": "completed", "artifact_manifest": manifest})
        output_manifest = json.loads(row["output_manifest_json"] or "{}")
        output_manifest["captioning"] = manifest
        conn.execute(
            "UPDATE basepipe_character_generation_runs SET current_stage='captioning', stage_history_json=?, "
            "output_manifest_json=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
            (json.dumps(history, ensure_ascii=False), json.dumps(output_manifest, ensure_ascii=False), run_id),
        )
        conn.commit()
        return {"run_id": run_id, "project_id": project_id, "current_stage": "captioning", "manifest": manifest}
    except HTTPException:
        conn.rollback()
        raise
    finally:
        conn.close()


def _asset_dict(row) -> dict:
    item = dict(row)
    item["metadata"] = json.loads(item.pop("metadata_json") or "{}")
    return item


def _with_asset_groups(conn, items: list[dict], project_id: int) -> list[dict]:
    if not items:
        return items
    ids = [int(item["id"]) for item in items]
    marks = ",".join("?" for _ in ids)
    rows = conn.execute(
        f"SELECT m.asset_id, g.id AS group_id, g.name AS group_name "
        f"FROM basepipe_asset_group_members m JOIN basepipe_asset_groups g ON g.id = m.group_id "
        f"WHERE g.project_id = ? AND m.asset_id IN ({marks}) ORDER BY g.id",
        (project_id, *ids),
    ).fetchall()
    groups: dict[int, list[dict]] = {asset_id: [] for asset_id in ids}
    for row in rows:
        groups[int(row["asset_id"])].append({"id": int(row["group_id"]), "name": row["group_name"]})
    for item in items:
        item["groups"] = groups[int(item["id"])]
        item["group_ids"] = [group["id"] for group in item["groups"]]
    return items


def _lineage_values(original: str, edited: str = "", processed: str = "") -> tuple[str, str, str]:
    """Keep source caption immutable and derive the exact string staged for training."""
    edited = edited or ""
    processed = processed or ""
    if processed.strip():
        return edited, processed, "processed"
    if edited.strip():
        return edited, edited, "edited"
    return edited, original, "original"


@router.get("/projects/{project_id}/workspace")
def workspace(project_id: int) -> dict:
    conn = get_conn()
    try:
        _project_or_404(conn, project_id)
        concepts = [dict(r) for r in conn.execute(
            "SELECT id, project_id, name, concept_type, trigger_token, description, status, created_at, updated_at "
            "FROM basepipe_concepts WHERE project_id = ? ORDER BY id DESC", (project_id,)
        ).fetchall()]
        assets = _with_asset_groups(conn, [_asset_dict(r) for r in conn.execute(
            "SELECT id, project_id, asset_key, file_path, content_sha256, asset_type, origin_kind, source_ref, "
            "caption, caption_source, caption_original, caption_edited, caption_processed, training_input, training_input_source, review_status, user_rating, training_enabled, training_weight, metadata_json, created_at "
            "FROM basepipe_assets WHERE project_id = ? ORDER BY id DESC LIMIT 200", (project_id,)
        ).fetchall()], project_id)
        groups = [dict(row) for row in conn.execute(
            "SELECT g.id, g.project_id, g.name, g.created_at, g.updated_at, COUNT(m.asset_id) AS asset_count "
            "FROM basepipe_asset_groups g LEFT JOIN basepipe_asset_group_members m ON m.group_id = g.id "
            "WHERE g.project_id = ? GROUP BY g.id ORDER BY g.id", (project_id,)
        ).fetchall()]
        snapshots = [dict(r) for r in conn.execute(
            "SELECT id, project_id, concept_id, name, snapshot_hash, status, item_count, created_at "
            "FROM basepipe_dataset_snapshots WHERE project_id = ? ORDER BY id DESC", (project_id,)
        ).fetchall()]
        return {"project_id": project_id, "concepts": concepts, "assets": assets, "groups": groups, "snapshots": snapshots}
    finally:
        conn.close()


@router.get("/projects/{project_id}/dataset-items")
def dataset_items(project_id: int) -> dict:
    """Expose the registered Dataset SSOT for Snapshot selection.

    This deliberately reads dataset_items rather than walking folders, so the
    selected training input and its current Caption Lineage remain explicit.
    """
    conn = get_conn()
    try:
        _project_or_404(conn, project_id)
        rows = conn.execute(
            "SELECT d.id, d.file_path, d.width, d.height, d.aspect, d.caption, d.caption_source, d.selected, "
            "a.id AS basepipe_asset_id, COALESCE(a.review_status, 'pending') AS review_status, "
            "a.caption_original, a.caption_edited, a.caption_processed, a.training_input, a.training_input_source "
            "FROM dataset_items d LEFT JOIN basepipe_assets a ON a.id = ("
            "SELECT a2.id FROM basepipe_assets a2 WHERE a2.project_id = d.project_id AND a2.file_path = d.file_path "
            "ORDER BY a2.id DESC LIMIT 1) WHERE d.project_id = ? ORDER BY d.id", (project_id,)
        ).fetchall()
        return {"project_id": project_id, "items": [dict(row) for row in rows]}
    finally:
        conn.close()


@router.get("/projects/{project_id}/dataset-audit")
def dataset_audit(project_id: int) -> dict:
    """Return a read-only pre-Snapshot audit of the current Dataset SSOT."""
    conn = get_conn()
    try:
        project = conn.execute("SELECT id, name, project_type, base_dir, dataset_dir FROM projects WHERE id = ?", (project_id,)).fetchone()
        if project is None:
            raise HTTPException(status_code=404, detail="project not found")
        rows = conn.execute(
            "SELECT file_path, caption, selected FROM dataset_items WHERE project_id = ? ORDER BY id", (project_id,)
        ).fetchall()
        selected = [row for row in rows if int(row["selected"] or 0) == 1]
        captioned = [row for row in selected if str(row["caption"] or "").strip()]
        tag_counter: Counter[str] = Counter()
        for row in captioned:
            for token in re.split(r"[,，]", str(row["caption"] or "")):
                token = token.strip().lower()
                if token and len(token) >= 2:
                    tag_counter[token] += 1
        top_tags = [
            {"tag": tag, "count": count, "ratio": round(count / len(captioned), 3) if captioned else 0.0}
            for tag, count in tag_counter.most_common(12)
        ]
        missing_files = sum(1 for row in selected if not Path(str(row["file_path"])).is_file())
        candidate_by_name = _dataset_repair_candidates(project, selected)
        repairable = [
            {"current_path": str(row["file_path"]), "candidate_path": candidate_by_name[Path(str(row["file_path"])).name]}
            for row in selected
            if not Path(str(row["file_path"])).is_file() and Path(str(row["file_path"])).name in candidate_by_name
        ]
        missing_caption = len(selected) - len(captioned)
        warnings: list[str] = []
        if missing_files:
            suffix = f"（現行Datasetから復元候補 {len(repairable)}件）" if repairable else ""
            warnings.append(f"入力ファイルが見つからない: {missing_files}件{suffix}")
        if missing_caption:
            warnings.append(f"caption未設定: {missing_caption}件")
        if top_tags and top_tags[0]["ratio"] >= 0.5:
            warnings.append(f"頻出タグが偏っています: {top_tags[0]['tag']} ({top_tags[0]['ratio'] * 100:.0f}%)")
        return {
            "project_id": project_id,
            "project_name": project["name"],
            "project_type": project["project_type"],
            "total": len(rows),
            "selected": len(selected),
            "captioned": len(captioned),
            "caption_rate": round(len(captioned) / len(selected), 3) if selected else 0.0,
            "missing_caption": missing_caption,
            "missing_files": missing_files,
            "repairable_files": len(repairable),
            "repair_candidates": repairable[:100],
            "top_tags": top_tags,
            "warnings": warnings,
        }
    finally:
        conn.close()


@router.post("/projects/{project_id}/repair-dataset-paths")
def repair_dataset_paths(project_id: int) -> dict:
    """Relink missing Dataset SSOT paths to unique local candidates, without moving files."""
    conn = get_conn()
    try:
        project = conn.execute("SELECT id, name, base_dir, dataset_dir FROM projects WHERE id = ?", (project_id,)).fetchone()
        if project is None:
            raise HTTPException(status_code=404, detail="project not found")
        rows = conn.execute("SELECT id, file_path FROM dataset_items WHERE project_id = ?", (project_id,)).fetchall()
        candidates = _dataset_repair_candidates(project, rows)
        updated: list[dict] = []
        for row in rows:
            old_path = str(row["file_path"])
            if Path(old_path).is_file():
                continue
            candidate = candidates.get(Path(old_path).name)
            if not candidate:
                continue
            conn.execute("UPDATE dataset_items SET file_path = ? WHERE id = ? AND project_id = ?", (candidate, row["id"], project_id))
            updated.append({"id": int(row["id"]), "old_path": old_path, "new_path": candidate})
        conn.commit()
        return {"project_id": project_id, "updated_count": len(updated), "updated": updated}
    except HTTPException:
        conn.rollback()
        raise
    except Exception as exc:
        conn.rollback()
        raise HTTPException(status_code=400, detail=f"dataset path repair failed: {exc}") from exc
    finally:
        conn.close()


@router.post("/projects/{project_id}/dataset-items/{item_id}/review")
def review_dataset_item(project_id: int, item_id: int, payload: BasepipeAssetReviewIn) -> dict:
    """Promote a Dataset SSOT item into a reviewable Basepipe Asset."""
    conn = get_conn()
    try:
        item = conn.execute(
            "SELECT * FROM dataset_items WHERE id = ? AND project_id = ?", (item_id, project_id)
        ).fetchone()
        if item is None:
            raise HTTPException(status_code=404, detail="dataset item not found")
        path = Path(item["file_path"])
        if not path.is_file():
            raise HTTPException(status_code=400, detail=f"dataset asset file not found: {path}")
        digest = _sha256(path)
        existing = conn.execute(
            "SELECT * FROM basepipe_assets WHERE project_id = ? AND file_path = ? ORDER BY id DESC LIMIT 1",
            (project_id, str(path)),
        ).fetchone()
        original = (existing["caption_original"] if existing and existing["caption_original"] else (existing["caption"] if existing else item["caption"])) or ""
        edited = (existing["caption_edited"] if existing else "") or (payload.caption if payload.caption is not None else "")
        processed = (existing["caption_processed"] if existing else "") or ""
        edited, training_input, training_input_source = _lineage_values(original, edited, processed)
        caption = training_input
        if existing is None:
            cur = conn.execute(
                "INSERT INTO basepipe_assets(project_id, asset_key, file_path, content_sha256, origin_kind, caption, caption_source, caption_original, caption_edited, caption_processed, training_input, training_input_source, review_status) "
                "VALUES (?, ?, ?, ?, 'dataset_item', ?, ?, ?, ?, ?, ?, ?, ?)",
                (project_id, path.stem, str(path), digest, caption, "human_review" if payload.caption is not None else item["caption_source"], original, edited, processed, training_input, training_input_source, payload.review_status),
            )
            asset_id = int(cur.lastrowid)
        else:
            asset_id = int(existing["id"])
            conn.execute(
                "UPDATE basepipe_assets SET content_sha256 = ?, review_status = ?, caption = ?, caption_source = ?, caption_original = ?, caption_edited = ?, caption_processed = ?, training_input = ?, training_input_source = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (digest, payload.review_status, caption, "human_review" if payload.caption is not None else item["caption_source"], original, edited, processed, training_input, training_input_source, asset_id),
            )
        conn.commit()
        row = conn.execute("SELECT * FROM basepipe_assets WHERE id = ?", (asset_id,)).fetchone()
        return _asset_dict(row)
    except HTTPException:
        conn.rollback()
        raise
    except Exception as exc:
        conn.rollback()
        raise HTTPException(status_code=400, detail=f"dataset review failed: {exc}") from exc
    finally:
        conn.close()


@router.post("/projects/{project_id}/dataset-items/bulk-review")
def bulk_review_dataset_items(project_id: int, payload: BasepipeDatasetBulkReviewIn) -> dict:
    """Apply an explicit human review decision to only the requested Dataset items."""
    conn = get_conn()
    try:
        _project_or_404(conn, project_id)
        item_ids = list(dict.fromkeys(payload.item_ids))
        marks = ",".join("?" for _ in item_ids)
        items = conn.execute(
            f"SELECT * FROM dataset_items WHERE project_id = ? AND id IN ({marks})",
            (project_id, *item_ids),
        ).fetchall()
        found = {int(item["id"]) for item in items}
        missing = [item_id for item_id in item_ids if item_id not in found]
        if missing:
            raise HTTPException(status_code=400, detail=f"dataset item not found: {', '.join(map(str, missing))}")
        updated: list[int] = []
        for item in items:
            path = Path(item["file_path"])
            if not path.is_file():
                raise HTTPException(status_code=400, detail=f"dataset asset file not found: {path}")
            digest = _sha256(path)
            existing = conn.execute(
                "SELECT id FROM basepipe_assets WHERE project_id = ? AND file_path = ? ORDER BY id DESC LIMIT 1",
                (project_id, str(path)),
            ).fetchone()
            if existing is None:
                original = item["caption"] or ""
                edited, training_input, training_input_source = _lineage_values(original)
                cur = conn.execute(
                    "INSERT INTO basepipe_assets(project_id, asset_key, file_path, content_sha256, origin_kind, caption, caption_source, caption_original, caption_edited, caption_processed, training_input, training_input_source, review_status) "
                    "VALUES (?, ?, ?, ?, 'dataset_item', ?, ?, ?, ?, ?, ?, ?, ?)",
                    (project_id, path.stem, str(path), digest, training_input, item["caption_source"], original, edited, "", training_input, training_input_source, payload.review_status),
                )
                updated.append(int(cur.lastrowid))
            else:
                asset_id = int(existing["id"])
                conn.execute(
                    "UPDATE basepipe_assets SET content_sha256 = ?, review_status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (digest, payload.review_status, asset_id),
                )
                updated.append(asset_id)
        conn.commit()
        return {"project_id": project_id, "item_ids": item_ids, "asset_ids": updated, "review_status": payload.review_status, "updated_count": len(updated)}
    except HTTPException:
        conn.rollback()
        raise
    except Exception as exc:
        conn.rollback()
        raise HTTPException(status_code=400, detail=f"bulk dataset review failed: {exc}") from exc
    finally:
        conn.close()


@router.post("/projects/{project_id}/concepts")
def create_concept(project_id: int, payload: BasepipeConceptCreate) -> dict:
    conn = get_conn()
    try:
        _project_or_404(conn, project_id)
        cur = conn.execute(
            "INSERT INTO basepipe_concepts(project_id, name, concept_type, trigger_token, description) VALUES (?, ?, ?, ?, ?)",
            (project_id, payload.name, payload.concept_type, payload.trigger_token, payload.description),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM basepipe_concepts WHERE id = ?", (cur.lastrowid,)).fetchone()
        return dict(row)
    except Exception as exc:
        conn.rollback()
        raise HTTPException(status_code=400, detail=f"concept create failed: {exc}") from exc
    finally:
        conn.close()


@router.get("/projects/{project_id}/concepts")
def list_concepts(project_id: int) -> dict:
    """List Concepts without exposing mutable project-internal state guesses."""
    conn = get_conn()
    try:
        _project_or_404(conn, project_id)
        rows = conn.execute(
            "SELECT id, project_id, name, concept_type, trigger_token, description, status, created_at, updated_at "
            "FROM basepipe_concepts WHERE project_id = ? ORDER BY id DESC",
            (project_id,),
        ).fetchall()
        return {"project_id": project_id, "concepts": [dict(row) for row in rows]}
    finally:
        conn.close()


@router.post("/projects/{project_id}/assets")
def create_asset(project_id: int, payload: BasepipeAssetCreate) -> dict:
    path = Path(payload.file_path).expanduser()
    if not path.is_file():
        raise HTTPException(status_code=400, detail=f"asset file not found: {path}")
    conn = get_conn()
    try:
        _project_or_404(conn, project_id)
        digest = _sha256(path)
        asset_key = payload.asset_key or path.stem
        original = payload.caption or ""
        edited, training_input, training_input_source = _lineage_values(original)
        cur = conn.execute(
            "INSERT INTO basepipe_assets(project_id, asset_key, file_path, content_sha256, asset_type, origin_kind, source_ref, caption, caption_source, caption_original, caption_edited, caption_processed, training_input, training_input_source, review_status, user_rating, training_enabled, training_weight, metadata_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (project_id, asset_key, str(path), digest, payload.asset_type, payload.origin_kind, payload.source_ref,
             training_input, payload.caption_source, original, edited, "", training_input, training_input_source, payload.review_status,
             payload.user_rating, int(payload.training_enabled), payload.training_weight, json.dumps(payload.metadata, ensure_ascii=False)),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM basepipe_assets WHERE id = ?", (cur.lastrowid,)).fetchone()
        return _asset_dict(row)
    except Exception as exc:
        conn.rollback()
        raise HTTPException(status_code=400, detail=f"asset create failed: {exc}") from exc
    finally:
        conn.close()


@router.post("/projects/{project_id}/reference-assets")
async def upload_reference_assets(
    project_id: int,
    files: list[UploadFile] = File(...),
    role: str = Form(default="detail"),
) -> dict:
    """Persist user-supplied character references and register their lineage.

    The generic collector drop zone only creates transient scan candidates. A
    Character Dataset Factory needs durable files that survive browser reloads
    and can be passed directly to H3, so references get a project-owned storage
    location and an immutable SHA-256-backed Asset record here.
    """
    if role not in {"front", "left", "back", "right", "detail"}:
        raise HTTPException(status_code=400, detail="reference role must be front, left, back, right, or detail")
    if not files or len(files) > 4:
        raise HTTPException(status_code=400, detail="reference images must contain 1 to 4 files")

    conn = get_conn()
    written: list[Path] = []
    try:
        project = conn.execute(
            "SELECT id, project_type, dataset_dir FROM projects WHERE id = ?",
            (project_id,),
        ).fetchone()
        if project is None:
            raise HTTPException(status_code=404, detail=f"project not found: {project_id}")
        if project["project_type"] != "character":
            raise HTTPException(status_code=400, detail="reference assets are only available for character projects")

        target_dir = Path(project["dataset_dir"]) / "references"
        target_dir.mkdir(parents=True, exist_ok=True)
        registered: list[dict] = []
        for index, uploaded in enumerate(files):
            raw = await uploaded.read()
            if not raw:
                raise HTTPException(status_code=400, detail=f"empty reference image: {uploaded.filename or index}")
            if len(raw) > 32 * 1024 * 1024:
                raise HTTPException(status_code=413, detail=f"reference image exceeds 32 MiB: {uploaded.filename or index}")
            try:
                with Image.open(io.BytesIO(raw)) as image:
                    image_size = image.size
                    image.verify()
                    image_format = (image.format or "PNG").lower()
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"invalid reference image: {uploaded.filename or index}") from exc

            digest = hashlib.sha256(raw).hexdigest()
            supplied = Path(uploaded.filename or f"reference-{index + 1}.png")
            suffix = supplied.suffix.lower()
            if suffix not in IMAGE_EXTS:
                suffix = ".jpg" if image_format in {"jpeg", "jpg"} else f".{image_format}"
            safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "-", supplied.stem).strip("-._") or f"reference-{index + 1}"
            destination = target_dir / f"{digest[:12]}-{safe_stem}{suffix}"
            if not destination.exists():
                destination.write_bytes(raw)
                written.append(destination)

            existing = conn.execute(
                "SELECT * FROM basepipe_assets WHERE project_id = ? AND content_sha256 = ? AND origin_kind = 'character_reference' ORDER BY id DESC LIMIT 1",
                (project_id, digest),
            ).fetchone()
            metadata = {
                "role": role,
                "reference_order": index,
                "identity_lock": {},
                "source_filename": supplied.name,
                "width": int(image_size[0]),
                "height": int(image_size[1]),
            }
            if existing is None:
                cursor = conn.execute(
                    "INSERT INTO basepipe_assets(project_id, asset_key, file_path, content_sha256, asset_type, origin_kind, source_ref, review_status, training_enabled, training_weight, metadata_json) "
                    "VALUES (?, ?, ?, ?, 'image', 'character_reference', 'user_upload', 'approved', 0, 1.0, ?)",
                    (project_id, safe_stem, str(destination), digest, json.dumps(metadata, ensure_ascii=False)),
                )
                existing = conn.execute("SELECT * FROM basepipe_assets WHERE id = ?", (cursor.lastrowid,)).fetchone()
            else:
                previous = json.loads(existing["metadata_json"] or "{}")
                previous.update(metadata)
                conn.execute(
                    "UPDATE basepipe_assets SET file_path = ?, metadata_json = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (str(destination), json.dumps(previous, ensure_ascii=False), existing["id"]),
                )
                existing = conn.execute("SELECT * FROM basepipe_assets WHERE id = ?", (existing["id"],)).fetchone()
            registered.append(_asset_dict(existing))

        conn.commit()
        return {"project_id": project_id, "assets": registered, "added_count": len(registered)}
    except HTTPException:
        conn.rollback()
        for path in written:
            path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        conn.rollback()
        for path in written:
            path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"reference upload failed: {exc}") from exc
    finally:
        conn.close()


@router.get("/projects/{project_id}/assets")
def list_assets(project_id: int, review_status: str | None = None) -> dict:
    """List immutable source Assets and their caption/review lineage."""
    conn = get_conn()
    try:
        _project_or_404(conn, project_id)
        query = (
            "SELECT id, project_id, asset_key, file_path, content_sha256, asset_type, origin_kind, source_ref, "
            "caption, caption_source, caption_original, caption_edited, caption_processed, training_input, "
            "training_input_source, review_status, user_rating, training_enabled, training_weight, metadata_json, created_at, updated_at "
            "FROM basepipe_assets WHERE project_id = ?"
        )
        params: list[object] = [project_id]
        if review_status:
            query += " AND review_status = ?"
            params.append(review_status)
        query += " ORDER BY id DESC"
        rows = conn.execute(query, params).fetchall()
        return {"project_id": project_id, "assets": _with_asset_groups(conn, [_asset_dict(row) for row in rows], project_id)}
    finally:
        conn.close()


@router.patch("/assets/{asset_id}/review")
def review_asset(asset_id: int, payload: BasepipeAssetReviewIn) -> dict:
    """Update human review metadata without changing the source file."""
    conn = get_conn()
    try:
        row = conn.execute("SELECT * FROM basepipe_assets WHERE id = ?", (asset_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="asset not found")
        if row["origin_kind"] == "dataset_file" and payload.caption is not None:
            from .dataset_files import root_for, image_path, sidecar, atomic_text
            root = root_for(int(row["project_id"]))
            path = image_path(root, str(Path(row["file_path"]).relative_to(root)))
            disk_caption, _ = sidecar(path)
            if disk_caption != (row["caption"] or ""):
                raise HTTPException(status_code=409, detail="外部でTXTが変更されています。画像・キャプション画面で再読込してください")
            atomic_text(path.with_suffix(".txt"), payload.caption)
        metadata = json.loads(row["metadata_json"] or "{}")
        previous_role = str(metadata.get("role") or "").lower()
        if payload.metadata is not None:
            metadata.update(payload.metadata)
        next_role = str(metadata.get("role") or "").lower()
        if row["origin_kind"] == "character_reference" and next_role in _CARDINAL_REFERENCE_ROLES:
            candidates = conn.execute(
                "SELECT id, metadata_json FROM basepipe_assets "
                "WHERE project_id = ? AND origin_kind = 'character_reference' AND id <> ?",
                (row["project_id"], asset_id),
            ).fetchall()
            for candidate in candidates:
                candidate_metadata = json.loads(candidate["metadata_json"] or "{}")
                if str(candidate_metadata.get("role") or "").lower() != next_role:
                    continue
                candidate_metadata["role"] = previous_role if previous_role in _CARDINAL_REFERENCE_ROLES else "detail"
                conn.execute(
                    "UPDATE basepipe_assets SET metadata_json = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (json.dumps(candidate_metadata, ensure_ascii=False), int(candidate["id"])),
                )
        if payload.caption is None:
            conn.execute(
                "UPDATE basepipe_assets SET review_status = ?, user_rating = COALESCE(?, user_rating), "
                "training_enabled = COALESCE(?, training_enabled), training_weight = COALESCE(?, training_weight), "
                "metadata_json = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (payload.review_status, payload.user_rating, None if payload.training_enabled is None else int(payload.training_enabled), payload.training_weight, json.dumps(metadata, ensure_ascii=False), asset_id),
            )
        else:
            # An empty original caption is still an established lineage value.
            # Falling back with ``or`` would promote the latest human edit to
            # "original", making it impossible to clear a caption afterwards.
            original = row["caption_original"] if row["caption_original"] is not None else (row["caption"] or "")
            edited = payload.caption if payload.caption is not None else (row["caption_edited"] or "")
            processed = row["caption_processed"] or ""
            edited, training_input, training_input_source = _lineage_values(original, edited, processed)
            conn.execute(
                "UPDATE basepipe_assets SET review_status = ?, caption = ?, caption_source = 'human_review', caption_original = ?, caption_edited = ?, caption_processed = ?, training_input = ?, training_input_source = ?, user_rating = COALESCE(?, user_rating), training_enabled = COALESCE(?, training_enabled), training_weight = COALESCE(?, training_weight), metadata_json = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (payload.review_status, training_input, original, edited, processed, training_input, training_input_source, payload.user_rating, None if payload.training_enabled is None else int(payload.training_enabled), payload.training_weight, json.dumps(metadata, ensure_ascii=False), asset_id),
            )
        conn.commit()
        updated = conn.execute("SELECT * FROM basepipe_assets WHERE id = ?", (asset_id,)).fetchone()
        return _asset_dict(updated)
    except HTTPException:
        conn.rollback()
        raise
    except Exception as exc:
        conn.rollback()
        raise HTTPException(status_code=400, detail=f"asset review failed: {exc}") from exc
    finally:
        conn.close()


@router.post("/projects/{project_id}/assets/bulk-review")
def bulk_review_assets(project_id: int, payload: BasepipeAssetBulkReviewIn) -> dict:
    """Persist one human review decision for an explicit Asset selection."""
    conn = get_conn()
    try:
        _project_or_404(conn, project_id)
        asset_ids = list(dict.fromkeys(payload.asset_ids))
        marks = ",".join("?" for _ in asset_ids)
        rows = conn.execute(
            f"SELECT id FROM basepipe_assets WHERE project_id = ? AND id IN ({marks})",
            (project_id, *asset_ids),
        ).fetchall()
        found = {int(row["id"]) for row in rows}
        missing = [asset_id for asset_id in asset_ids if asset_id not in found]
        if missing:
            raise HTTPException(status_code=400, detail=f"asset not found in project: {', '.join(map(str, missing))}")
        if payload.training_enabled is None:
            conn.execute(
                f"UPDATE basepipe_assets SET review_status = ?, updated_at = CURRENT_TIMESTAMP WHERE project_id = ? AND id IN ({marks})",
                (payload.review_status, project_id, *asset_ids),
            )
        else:
            conn.execute(
                f"UPDATE basepipe_assets SET review_status = ?, training_enabled = ?, updated_at = CURRENT_TIMESTAMP WHERE project_id = ? AND id IN ({marks})",
                (payload.review_status, int(payload.training_enabled), project_id, *asset_ids),
            )
        conn.commit()
        updated_rows = conn.execute(
            f"SELECT * FROM basepipe_assets WHERE project_id = ? AND id IN ({marks}) ORDER BY id DESC",
            (project_id, *asset_ids),
        ).fetchall()
        assets = _with_asset_groups(conn, [_asset_dict(row) for row in updated_rows], project_id)
        return {"project_id": project_id, "updated_count": len(assets), "asset_ids": asset_ids, "assets": assets}
    except HTTPException:
        conn.rollback()
        raise
    except Exception as exc:
        conn.rollback()
        raise HTTPException(status_code=400, detail=f"asset bulk review failed: {exc}") from exc
    finally:
        conn.close()


@router.get("/projects/{project_id}/groups")
def list_asset_groups(project_id: int) -> dict:
    conn = get_conn()
    try:
        _project_or_404(conn, project_id)
        groups = conn.execute(
            "SELECT g.id, g.project_id, g.name, g.created_at, g.updated_at, COUNT(m.asset_id) AS asset_count "
            "FROM basepipe_asset_groups g LEFT JOIN basepipe_asset_group_members m ON m.group_id = g.id "
            "WHERE g.project_id = ? GROUP BY g.id ORDER BY g.id", (project_id,),
        ).fetchall()
        return {"project_id": project_id, "groups": [dict(row) for row in groups]}
    finally:
        conn.close()


@router.post("/projects/{project_id}/groups")
def create_asset_group(project_id: int, payload: BasepipeAssetGroupCreate) -> dict:
    conn = get_conn()
    try:
        _project_or_404(conn, project_id)
        cur = conn.execute("INSERT INTO basepipe_asset_groups(project_id, name) VALUES (?, ?)", (project_id, payload.name.strip()))
        conn.commit()
        row = conn.execute("SELECT id, project_id, name, created_at, updated_at, 0 AS asset_count FROM basepipe_asset_groups WHERE id = ?", (cur.lastrowid,)).fetchone()
        return dict(row)
    except Exception as exc:
        conn.rollback()
        raise HTTPException(status_code=400, detail=f"group create failed: {exc}") from exc
    finally:
        conn.close()


@router.patch("/projects/{project_id}/groups/{group_id}")
def rename_asset_group(project_id: int, group_id: int, payload: BasepipeAssetGroupCreate) -> dict:
    conn = get_conn()
    try:
        row = conn.execute("SELECT id FROM basepipe_asset_groups WHERE id = ? AND project_id = ?", (group_id, project_id)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="group not found")
        conn.execute("UPDATE basepipe_asset_groups SET name = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?", (payload.name.strip(), group_id))
        conn.commit()
        return dict(conn.execute("SELECT g.id, g.project_id, g.name, g.created_at, g.updated_at, COUNT(m.asset_id) AS asset_count FROM basepipe_asset_groups g LEFT JOIN basepipe_asset_group_members m ON m.group_id = g.id WHERE g.id = ? GROUP BY g.id", (group_id,)).fetchone())
    except HTTPException:
        conn.rollback()
        raise
    except Exception as exc:
        conn.rollback()
        raise HTTPException(status_code=400, detail=f"group rename failed: {exc}") from exc
    finally:
        conn.close()


@router.delete("/projects/{project_id}/groups/{group_id}")
def delete_asset_group(project_id: int, group_id: int) -> dict:
    conn = get_conn()
    try:
        row = conn.execute("SELECT id FROM basepipe_asset_groups WHERE id = ? AND project_id = ?", (group_id, project_id)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="group not found")
        conn.execute("DELETE FROM basepipe_asset_group_members WHERE group_id = ?", (group_id,))
        conn.execute("DELETE FROM basepipe_asset_groups WHERE id = ?", (group_id,))
        conn.commit()
        return {"project_id": project_id, "group_id": group_id, "deleted": True}
    except HTTPException:
        conn.rollback()
        raise
    finally:
        conn.close()


@router.put("/projects/{project_id}/groups/{group_id}/assets")
def assign_asset_group(project_id: int, group_id: int, payload: BasepipeAssetGroupMembersIn) -> dict:
    conn = get_conn()
    try:
        group = conn.execute("SELECT id FROM basepipe_asset_groups WHERE id = ? AND project_id = ?", (group_id, project_id)).fetchone()
        if group is None:
            raise HTTPException(status_code=404, detail="group not found")
        ids = list(dict.fromkeys(payload.asset_ids))
        marks = ",".join("?" for _ in ids)
        found = conn.execute(f"SELECT id FROM basepipe_assets WHERE project_id = ? AND id IN ({marks})", (project_id, *ids)).fetchall()
        if len(found) != len(ids):
            raise HTTPException(status_code=400, detail="one or more assets do not belong to this project")
        conn.executemany("INSERT OR IGNORE INTO basepipe_asset_group_members(group_id, asset_id) VALUES (?, ?)", [(group_id, asset_id) for asset_id in ids])
        conn.commit()
        return {"project_id": project_id, "group_id": group_id, "asset_ids": ids}
    except HTTPException:
        conn.rollback()
        raise
    finally:
        conn.close()


@router.delete("/projects/{project_id}/groups/{group_id}/assets/{asset_id}")
def unassign_asset_group(project_id: int, group_id: int, asset_id: int) -> dict:
    conn = get_conn()
    try:
        group = conn.execute("SELECT id FROM basepipe_asset_groups WHERE id = ? AND project_id = ?", (group_id, project_id)).fetchone()
        if group is None:
            raise HTTPException(status_code=404, detail="group not found")
        conn.execute("DELETE FROM basepipe_asset_group_members WHERE group_id = ? AND asset_id = ?", (group_id, asset_id))
        conn.commit()
        return {"project_id": project_id, "group_id": group_id, "asset_id": asset_id, "unassigned": True}
    finally:
        conn.close()


@router.get("/assets/{asset_id}/caption-lineage")
def get_caption_lineage(asset_id: int) -> dict:
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT id, project_id, caption_original, caption_edited, caption_processed, training_input, training_input_source, updated_at "
            "FROM basepipe_assets WHERE id = ?", (asset_id,)
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="asset not found")
        return {"asset_id": int(row["id"]), "project_id": int(row["project_id"]), "original": row["caption_original"], "edited": row["caption_edited"], "processed": row["caption_processed"], "training_input": row["training_input"], "training_input_source": row["training_input_source"], "updated_at": row["updated_at"]}
    finally:
        conn.close()


@router.delete("/assets/{asset_id}")
def delete_asset(asset_id: int) -> dict:
    """Remove a Basepipe registration and its derived metadata, never the source file."""
    conn = get_conn()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT id, project_id, file_path FROM basepipe_assets WHERE id = ?", (asset_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="asset not found")
        snapshots = conn.execute(
            "SELECT DISTINCT snapshot_id FROM basepipe_snapshot_entries WHERE asset_id=? ORDER BY snapshot_id",
            (asset_id,),
        ).fetchall()
        if snapshots:
            ids = ', '.join(str(r['snapshot_id']) for r in snapshots)
            raise HTTPException(409, f"学習対象Snapshot（{ids}）が参照している画像登録は削除できません。元ファイルと学習履歴を保持します。")
        conn.execute("DELETE FROM basepipe_asset_group_members WHERE asset_id = ?", (asset_id,))
        conn.execute("DELETE FROM basepipe_asset_versions WHERE asset_id = ?", (asset_id,))
        conn.execute("DELETE FROM basepipe_assets WHERE id = ?", (asset_id,))
        conn.commit()
        return {"asset_id": asset_id, "deleted": True, "source_file_preserved": True, "file_path": row["file_path"]}
    finally:
        conn.close()


@router.get("/assets/{asset_id}/versions")
def list_asset_versions(asset_id: int) -> dict:
    conn = get_conn()
    try:
        asset = conn.execute(
            "SELECT id, project_id, origin_kind, source_ref, metadata_json FROM basepipe_assets WHERE id = ?",
            (asset_id,),
        ).fetchone()
        if asset is None:
            raise HTTPException(status_code=404, detail="asset not found")
        rows = conn.execute("SELECT id, asset_id, parent_version_id, version_kind, file_path, content_sha256, prompt, seed, status, metadata_json, created_at FROM basepipe_asset_versions WHERE asset_id = ? ORDER BY id", (asset_id,)).fetchall()
        asset_metadata = json.loads(asset["metadata_json"] or "{}")
        mutable, lock_reason, frozen_snapshot_id = _training_input_mutability(conn, asset, asset_metadata)
        selected_id = asset_metadata.get("selected_version_id") or asset_metadata.get("selected_qwen_version_id")
        result = []
        for row in rows:
            item = dict(row)
            item["metadata"] = json.loads(item.pop("metadata_json") or "{}")
            item["selected"] = selected_id is not None and int(item["id"]) == int(selected_id)
            result.append(item)
        return {
            "asset_id": asset_id,
            "project_id": int(asset["project_id"]),
            "training_input_choice": asset_metadata.get("training_input_choice") or ("qwen" if selected_id is not None else None),
            "selected_version_id": selected_id,
            "selected_qwen_version_id": selected_id,
            "aesthetic_review_state": asset_metadata.get("training_input_review_state") or AESTHETIC_UNREVIEWED,
            "aesthetic_reviewed_by": asset_metadata.get("training_input_reviewed_by"),
            "aesthetic_review_note": asset_metadata.get("training_input_review_note") or "",
            "training_input_mutable": mutable,
            "training_input_lock_reason": lock_reason,
            "frozen_snapshot_id": frozen_snapshot_id,
            "versions": result,
        }
    finally:
        conn.close()


def _training_input_mutability(conn, asset, metadata: dict | None = None) -> tuple[bool, str, int | None]:
    """Return whether an Asset's Training Input may still be changed.

    A sealed Snapshot owns an immutable input choice. H3 Frames are also only
    editable during the explicit Identity/Qwen selection stages; later stages
    must not let the mutable Asset row contradict the Run evidence shown in UI.
    """
    sealed = conn.execute(
        "SELECT s.id FROM basepipe_snapshot_entries e "
        "JOIN basepipe_dataset_snapshots s ON s.id = e.snapshot_id "
        "WHERE e.asset_id = ? AND s.status = 'sealed' ORDER BY s.id DESC LIMIT 1",
        (asset["id"],),
    ).fetchone()
    if sealed is not None:
        snapshot_id = int(sealed["id"])
        return False, f"Dataset Snapshot #{snapshot_id}でTraining Inputは固定済みです", snapshot_id
    if str(asset["origin_kind"] or "") != "h3_frame":
        return True, "", None

    metadata = metadata if isinstance(metadata, dict) else json.loads(asset["metadata_json"] or "{}")
    lineage = metadata.get("lineage") if isinstance(metadata.get("lineage"), dict) else {}
    run_id = lineage.get("run_id")
    if run_id is None:
        source_ref = str(asset["source_ref"] or "")
        match = re.search(r"(\d+)$", source_ref)
        run_id = int(match.group(1)) if match else None
    if run_id is None:
        return False, "H3 FrameのGeneration Run lineageを確認できません", None
    run = conn.execute(
        "SELECT current_stage, output_manifest_json FROM basepipe_character_generation_runs WHERE id = ? AND project_id = ?",
        (run_id, asset["project_id"]),
    ).fetchone()
    if run is None:
        return False, f"Generation Run #{run_id}を確認できません", None
    current_frame_ids = _current_character_generation_frame_ids(run)
    if current_frame_ids is not None and int(asset["id"]) not in current_frame_ids:
        return False, f"Generation Run #{run_id}の現行Frame候補ではありません", None
    if str(run["current_stage"] or "") not in {"identity_review", "qwen_correction"}:
        return False, f"Generation Run #{run_id}は{run['current_stage']}段階でTraining Inputを固定済みです", None
    return True, "", None


def _require_mutable_training_input(conn, asset, metadata: dict | None = None) -> None:
    mutable, reason, _snapshot_id = _training_input_mutability(conn, asset, metadata)
    if not mutable:
        raise HTTPException(status_code=409, detail=reason)


@router.post("/assets/{asset_id}/versions/{version_id}/select")
def select_asset_version(asset_id: int, version_id: int, payload: BasepipeAestheticApprovalIn) -> dict:
    _require_user_aesthetic_confirmation(payload.human_confirmed)
    conn = get_conn()
    try:
        asset = conn.execute(
            "SELECT id, project_id, origin_kind, source_ref, metadata_json FROM basepipe_assets WHERE id=?",
            (asset_id,),
        ).fetchone()
        if asset is None:
            raise HTTPException(status_code=404, detail="asset not found")
        metadata = json.loads(asset["metadata_json"] or "{}")
        _require_mutable_training_input(conn, asset, metadata)
        version = conn.execute(
            "SELECT id, asset_id, file_path, content_sha256, version_kind, status, metadata_json FROM basepipe_asset_versions WHERE id=? AND asset_id=?",
            (version_id, asset_id),
        ).fetchone()
        if version is None:
            raise HTTPException(status_code=404, detail="asset version not found")
        if version["version_kind"] != "qwen_correction" or version["status"] != "ready":
            raise HTTPException(status_code=409, detail="選択できるのはready状態のQwen correction Versionだけです")
        path = Path(str(version["file_path"]))
        if not path.is_file() or _sha256(path) != str(version["content_sha256"]):
            raise HTTPException(status_code=409, detail="Qwen Version fileの存在またはSHA-256を確認できません")
        metadata["selected_qwen_version_id"] = version_id
        metadata["training_input_choice"] = "qwen"
        _record_training_input_review(metadata, choice="qwen", version_id=version_id, note=payload.note)
        version_metadata = json.loads(version["metadata_json"] or "{}")
        version_metadata["aesthetic_review_state"] = USER_APPROVED
        version_metadata["aesthetic_reviewed_by"] = "user"
        version_metadata["aesthetic_reviewed_at"] = metadata["training_input_reviewed_at"]
        version_metadata["aesthetic_review_note"] = payload.note.strip()
        conn.execute(
            "UPDATE basepipe_assets SET metadata_json=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
            (json.dumps(metadata, ensure_ascii=False), asset_id),
        )
        conn.execute(
            "UPDATE basepipe_asset_versions SET metadata_json=? WHERE id=?",
            (json.dumps(version_metadata, ensure_ascii=False), version_id),
        )
        conn.commit()
        return {"asset_id": asset_id, "project_id": int(asset["project_id"]), "training_input_choice": "qwen", "selected_qwen_version_id": version_id, "aesthetic_review_state": USER_APPROVED, "aesthetic_reviewed_by": "user", "file_path": str(path), "content_sha256": version["content_sha256"]}
    finally:
        conn.close()


@router.put("/assets/{asset_id}/training-input")
def choose_asset_training_input(asset_id: int, payload: BasepipeTrainingInputChoiceIn) -> dict:
    _require_user_aesthetic_confirmation(payload.human_confirmed)
    conn = get_conn()
    try:
        asset = conn.execute("SELECT * FROM basepipe_assets WHERE id=?", (asset_id,)).fetchone()
        if asset is None:
            raise HTTPException(status_code=404, detail="asset not found")
        if asset["origin_kind"] != "h3_frame":
            raise HTTPException(status_code=409, detail="Training Input選択はH3 Frameだけに使用できます")
        metadata = json.loads(asset["metadata_json"] or "{}")
        _require_mutable_training_input(conn, asset, metadata)
        if payload.choice == "original":
            path = Path(str(asset["file_path"] or ""))
            digest = str(asset["content_sha256"] or "")
            if not path.is_file() or not digest or _sha256(path) != digest:
                raise HTTPException(status_code=409, detail="Original Frameの存在またはSHA-256を確認できません")
            metadata["training_input_choice"] = "original"
            _record_training_input_review(metadata, choice="original", version_id=None, note=payload.note)
            conn.execute("UPDATE basepipe_assets SET metadata_json=?, updated_at=CURRENT_TIMESTAMP WHERE id=?", (json.dumps(metadata, ensure_ascii=False), asset_id))
            conn.commit()
            return {"asset_id": asset_id, "project_id": int(asset["project_id"]), "training_input_choice": "original", "selected_qwen_version_id": metadata.get("selected_qwen_version_id"), "aesthetic_review_state": USER_APPROVED, "aesthetic_reviewed_by": "user", "file_path": str(path), "content_sha256": digest}
        if payload.version_id is None:
            raise HTTPException(status_code=400, detail="Version Training Inputにはversion_idが必要です")
        version = conn.execute(
            "SELECT id, file_path, content_sha256, version_kind, status, metadata_json FROM basepipe_asset_versions WHERE id=? AND asset_id=?",
            (payload.version_id, asset_id),
        ).fetchone()
        allowed_kinds = {"qwen_correction"} if payload.choice == "qwen" else {"flashvsr_lr9_2x", "flashvsr_lr11_2x", "realesrgan_x2"}
        if version is None or version["version_kind"] not in allowed_kinds or version["status"] != "ready":
            raise HTTPException(status_code=409, detail=f"ready状態の{payload.choice} Versionを指定してください")
        path = Path(str(version["file_path"] or ""))
        if not path.is_file() or _sha256(path) != str(version["content_sha256"]):
            raise HTTPException(status_code=409, detail="Version fileの存在またはSHA-256を確認できません")
        metadata["training_input_choice"] = payload.choice
        metadata["selected_version_id"] = int(version["id"])
        if payload.choice == "qwen":
            metadata["selected_qwen_version_id"] = int(version["id"])
        _record_training_input_review(metadata, choice=payload.choice, version_id=int(version["id"]), note=payload.note)
        version_metadata = json.loads(version["metadata_json"] or "{}")
        version_metadata["aesthetic_review_state"] = USER_APPROVED
        version_metadata["aesthetic_reviewed_by"] = "user"
        version_metadata["aesthetic_reviewed_at"] = metadata["training_input_reviewed_at"]
        version_metadata["aesthetic_review_note"] = payload.note.strip()
        conn.execute("UPDATE basepipe_assets SET metadata_json=?, updated_at=CURRENT_TIMESTAMP WHERE id=?", (json.dumps(metadata, ensure_ascii=False), asset_id))
        conn.execute("UPDATE basepipe_asset_versions SET metadata_json=? WHERE id=?", (json.dumps(version_metadata, ensure_ascii=False), int(version["id"])))
        conn.commit()
        return {"asset_id": asset_id, "project_id": int(asset["project_id"]), "training_input_choice": payload.choice, "selected_version_id": int(version["id"]), "selected_qwen_version_id": int(version["id"]) if payload.choice == "qwen" else metadata.get("selected_qwen_version_id"), "aesthetic_review_state": USER_APPROVED, "aesthetic_reviewed_by": "user", "file_path": str(path), "content_sha256": version["content_sha256"]}
    finally:
        conn.close()


def _run_asset_transform(
    asset_id: int,
    project_id: int,
    version_kind: str,
    payload: BasepipeAssetTransformIn,
    *,
    admission_verified: bool = False,
    master_references: list[dict] | None = None,
    runtime_reference_paths: list[Path] | None = None,
    reference_contract: dict | None = None,
    runtime_source_path: Path | None = None,
    qwen_use_lightning: bool | None = None,
    qwen_angle_lora_name: str | None = None,
    qwen_angle_lora_strength: float = 0.9,
) -> dict:
    """Create an immutable Qwen-derived asset version when explicitly executed."""
    conn = get_conn()
    try:
        asset = conn.execute("SELECT * FROM basepipe_assets WHERE id = ? AND project_id = ?", (asset_id, project_id)).fetchone()
        if asset is None:
            raise HTTPException(status_code=404, detail="asset not found")
        original_source = Path(asset["file_path"])
        source = runtime_source_path or original_source
        if not source.is_file():
            raise HTTPException(status_code=400, detail=f"asset file not found: {source}")
        parent = conn.execute("SELECT id FROM basepipe_asset_versions WHERE asset_id = ? ORDER BY id DESC LIMIT 1", (asset_id,)).fetchone()
        prompt = payload.prompt.strip() or ("remove character-specific ornaments and preserve neutral clothing" if version_kind == "neutralize" else "preserve the subject and correct the image")
        if not payload.execute:
            return {"asset_id": asset_id, "project_id": project_id, "version_kind": version_kind, "status": "ready_for_execution", "gpu_device_id": payload.gpu_device_id, "source_path": str(source), "prompt": prompt, "message": "execute=trueでQwen処理を開始します。元画像は変更されません。"}
        if payload.gpu_device_id != 1:
            raise HTTPException(status_code=400, detail="Qwen Asset処理はGPU1固定です。GPU0は使用しません。")
        memory = _query_qwen_gpu_memory()
        if admission_verified:
            conflict = _qwen_runtime_external_conflict(memory)
            if conflict:
                raise HTTPException(status_code=409, detail=f"Qwen Asset処理を中断しました: {conflict}")
        else:
            if not memory.get("admission_ok"):
                raise HTTPException(status_code=409, detail=f"Qwen Asset処理を開始しません: {_h3_gpu_check(memory)['detail']}")
            comfy_ready, comfy_detail = _ensure_h3_comfyui()
            if not comfy_ready:
                raise HTTPException(status_code=409, detail=f"Qwen Asset処理を開始しません: {comfy_detail}")
            memory = _query_qwen_gpu_memory()
            if not memory.get("admission_ok"):
                raise HTTPException(status_code=409, detail=f"Qwen Asset処理を開始しません: {_h3_gpu_check(memory)['detail']}")
        from ..services.comfyui_client import ComfyUIClient
        client = ComfyUIClient(base_url=H3_COMFYUI_URL, timeout=0.75)
        qwen_status = client.check_qwen_status()
        if not qwen_status.get("ready"):
            raise HTTPException(
                status_code=400,
                detail=f"Qwen Asset処理を開始できません: {qwen_status.get('message') or 'ComfyUI/Qwenの準備が未完了です'}",
            )
        reference_paths = runtime_reference_paths if runtime_reference_paths is not None else [Path(str(item["path"])) for item in (master_references or [])[:2]]
        qwen_kwargs = {"reference_paths": reference_paths, "prompt": prompt, "seed": payload.seed}
        if qwen_use_lightning is not None:
            qwen_kwargs["use_lightning"] = qwen_use_lightning
        if qwen_angle_lora_name is not None:
            if not qwen_status.get("multiple_angles_ok"):
                raise HTTPException(status_code=400, detail=f"Multiple-Angles LoRAがComfyUIにありません: {qwen_angle_lora_name}")
            qwen_kwargs["angle_lora_name"] = qwen_angle_lora_name
            qwen_kwargs["angle_lora_strength"] = qwen_angle_lora_strength
        output_bytes = client.qwen_cleanup(source, **qwen_kwargs)
        digest = hashlib.sha256(output_bytes).hexdigest()
        version_dir = original_source.parent / ".basepipe_versions"
        version_dir.mkdir(parents=True, exist_ok=True)
        output = version_dir / f"asset_{asset_id}_{version_kind}_seed{payload.seed}_{digest[:12]}.png"
        if not output.is_file():
            output.write_bytes(output_bytes)
        runtime = {
            "engine": "Qwen-Image-Edit-2511",
            "gpu_device_id": payload.gpu_device_id,
            "physical_gpu": memory,
            "source_asset": str(original_source),
            "runtime_source_path": str(source),
            "master_references": master_references or [],
            "runtime_reference_paths": [str(path) for path in reference_paths],
            "reference_contract": reference_contract or {},
            "diffusion_model": qwen_status.get("diffusion_model"),
            "text_encoder": qwen_status.get("text_encoder"),
            "vae_model": qwen_status.get("vae_model"),
            "lightning_available": bool(qwen_status.get("lightning_ok")),
            "lightning_used": bool(qwen_status.get("lightning_ok")) if qwen_use_lightning is None else qwen_use_lightning,
            "sampler_profile": {"steps": 4, "cfg": 1.0} if (bool(qwen_status.get("lightning_ok")) if qwen_use_lightning is None else qwen_use_lightning) else {"steps": 40, "cfg": 4.0},
            "angle_lora": qwen_angle_lora_name,
            "angle_lora_strength": qwen_angle_lora_strength if qwen_angle_lora_name else None,
            "aesthetic_review_state": AESTHETIC_UNREVIEWED,
            "aesthetic_reviewed_by": None,
            "aesthetic_policy": "user is the sole aesthetic approver",
        }
        cur = conn.execute("INSERT INTO basepipe_asset_versions(asset_id, project_id, parent_version_id, version_kind, file_path, content_sha256, prompt, seed, status, metadata_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'ready', ?)", (asset_id, project_id, int(parent["id"]) if parent else None, version_kind, str(output), digest, prompt, payload.seed, json.dumps(runtime, ensure_ascii=False)))
        version_id = int(cur.lastrowid)
        selected = False
        conn.commit()
        return {"asset_id": asset_id, "project_id": project_id, "version_id": version_id, "version_kind": version_kind, "status": "ready", "selected": selected, "file_path": str(output), "content_sha256": digest, "parent_version_id": int(parent["id"]) if parent else None}
    finally:
        conn.close()


@router.post("/projects/{project_id}/assets/{asset_id}/qwen-edit")
def qwen_edit_asset(project_id: int, asset_id: int, payload: BasepipeAssetTransformIn) -> dict:
    return _run_asset_transform(asset_id, project_id, "qwen_edit", payload)


@router.post("/projects/{project_id}/assets/{asset_id}/neutralize")
def neutralize_asset(project_id: int, asset_id: int, payload: BasepipeAssetTransformIn) -> dict:
    return _run_asset_transform(asset_id, project_id, "neutralize", payload)


@router.put("/assets/{asset_id}/caption-lineage")
def update_caption_lineage(asset_id: int, payload: BasepipeCaptionLineageUpdate) -> dict:
    conn = get_conn()
    try:
        row = conn.execute("SELECT * FROM basepipe_assets WHERE id = ?", (asset_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="asset not found")
        original = row["caption_original"] or row["caption"] or ""
        edited = row["caption_edited"] if payload.edited is None else payload.edited
        processed = row["caption_processed"] if payload.processed is None else payload.processed
        edited, training_input, training_input_source = _lineage_values(original, edited, processed)
        conn.execute(
            "UPDATE basepipe_assets SET caption = ?, caption_edited = ?, caption_processed = ?, training_input = ?, training_input_source = ?, caption_source = 'human_review', updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (training_input, edited, processed or "", training_input, training_input_source, asset_id),
        )
        conn.commit()
        updated = conn.execute("SELECT * FROM basepipe_assets WHERE id = ?", (asset_id,)).fetchone()
        return _asset_dict(updated)
    except HTTPException:
        conn.rollback()
        raise
    except Exception as exc:
        conn.rollback()
        raise HTTPException(status_code=400, detail=f"caption lineage update failed: {exc}") from exc
    finally:
        conn.close()


@router.post("/projects/{project_id}/captions/bulk-preview")
def caption_bulk_preview(project_id: int, payload: BasepipeCaptionBulkPreviewIn) -> dict:
    """Create a non-destructive caption diff over Basepipe assets."""
    conn = get_conn()
    try:
        _project_or_404(conn, project_id)
        if payload.asset_ids:
            marks = ",".join("?" for _ in payload.asset_ids)
            rows = conn.execute(
                f"SELECT id, caption, caption_original, caption_edited, caption_processed, training_input, training_input_source, caption_source FROM basepipe_assets WHERE project_id = ? AND id IN ({marks}) ORDER BY id",
                [project_id, *payload.asset_ids],
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, caption, caption_original, caption_edited, caption_processed, training_input, training_input_source, caption_source FROM basepipe_assets WHERE project_id = ? ORDER BY id",
                (project_id,),
            ).fetchall()
        changes = []
        for row in rows:
            before = row["caption_processed"] or row["caption_edited"] or row["caption_original"] or ""
            after = before.replace(payload.find, payload.replace)
            if after != before:
                changes.append({
                    "asset_id": int(row["id"]),
                    "before": before,
                    "after": after,
                    "before_lineage": {
                        "caption": row["caption"] or "",
                        "caption_original": row["caption_original"] or "",
                        "caption_edited": row["caption_edited"] or "",
                        "caption_processed": row["caption_processed"] or "",
                        "training_input": row["training_input"] or "",
                        "training_input_source": row["training_input_source"] or "original",
                        "caption_source": row["caption_source"] or "",
                    },
                })
        cur = conn.execute(
            "INSERT INTO basepipe_caption_bulk_operations(project_id, changes_json, status) VALUES (?, ?, 'preview')",
            (project_id, json.dumps(changes, ensure_ascii=False)),
        )
        conn.commit()
        return {"operation_id": int(cur.lastrowid), "project_id": project_id, "find": payload.find, "replace": payload.replace, "affected_count": len(changes), "changes": changes}
    finally:
        conn.close()


@router.post("/projects/{project_id}/captions/bulk-apply")
def caption_bulk_apply(project_id: int, payload: BasepipeCaptionBulkOperationIn) -> dict:
    conn = get_conn()
    try:
        row = conn.execute("SELECT id, changes_json, status FROM basepipe_caption_bulk_operations WHERE id = ? AND project_id = ?", (payload.operation_id, project_id)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="caption bulk operation not found")
        if row["status"] != "preview":
            raise HTTPException(status_code=409, detail="caption bulk operation is not pending")
        changes = json.loads(row["changes_json"] or "[]")
        # The preview is an immutable intent.  Refuse a stale Apply if any
        # selected caption changed after the user inspected the diff.
        stale_assets: list[int] = []
        for change in changes:
            asset = conn.execute(
                "SELECT caption_original, caption_edited, caption_processed FROM basepipe_assets WHERE id = ? AND project_id = ?",
                (change["asset_id"], project_id),
            ).fetchone()
            current = (asset["caption_processed"] or asset["caption_edited"] or asset["caption_original"] or "") if asset else None
            if current != change["before"]:
                stale_assets.append(int(change["asset_id"]))
        if stale_assets:
            raise HTTPException(status_code=409, detail=f"caption changed after preview: asset ids {stale_assets}")

        applied = 0
        for change in changes:
            asset = conn.execute("SELECT caption_original, caption_edited, caption_processed FROM basepipe_assets WHERE id = ? AND project_id = ?", (change["asset_id"], project_id)).fetchone()
            if asset is None:
                continue
            edited = change["after"]
            conn.execute("UPDATE basepipe_assets SET caption = ?, caption_edited = ?, caption_processed = '', training_input = ?, training_input_source = 'edited', caption_source = 'human_bulk_edit', updated_at = CURRENT_TIMESTAMP WHERE id = ?", (edited, edited, edited, change["asset_id"]))
            applied += 1
        conn.execute("UPDATE basepipe_caption_bulk_operations SET status = 'applied', applied_at = CURRENT_TIMESTAMP WHERE id = ?", (payload.operation_id,))
        conn.commit()
        return {"operation_id": payload.operation_id, "project_id": project_id, "applied_count": applied, "status": "applied"}
    finally:
        conn.close()


@router.post("/projects/{project_id}/captions/undo")
def caption_bulk_undo(project_id: int, payload: BasepipeCaptionBulkOperationIn) -> dict:
    conn = get_conn()
    try:
        row = conn.execute("SELECT id, changes_json, status FROM basepipe_caption_bulk_operations WHERE id = ? AND project_id = ?", (payload.operation_id, project_id)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="caption bulk operation not found")
        if row["status"] != "applied":
            raise HTTPException(status_code=409, detail="only an applied caption bulk operation can be undone")
        changes = json.loads(row["changes_json"] or "[]")
        undone = 0
        for change in changes:
            before_lineage = change.get("before_lineage")
            if isinstance(before_lineage, dict):
                cursor = conn.execute(
                    "UPDATE basepipe_assets SET caption = ?, caption_original = ?, caption_edited = ?, caption_processed = ?, training_input = ?, training_input_source = ?, caption_source = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ? AND project_id = ?",
                    (
                        before_lineage.get("caption", ""),
                        before_lineage.get("caption_original", ""),
                        before_lineage.get("caption_edited", ""),
                        before_lineage.get("caption_processed", ""),
                        before_lineage.get("training_input", ""),
                        before_lineage.get("training_input_source", "original"),
                        before_lineage.get("caption_source", ""),
                        change["asset_id"],
                        project_id,
                    ),
                )
            else:
                # Backward compatibility for preview records created before
                # full lineage snapshots were stored.
                cursor = conn.execute("UPDATE basepipe_assets SET caption = ?, caption_edited = '', caption_processed = '', training_input = ?, training_input_source = 'original', caption_source = 'undo', updated_at = CURRENT_TIMESTAMP WHERE id = ? AND project_id = ?", (change["before"], change["before"], change["asset_id"], project_id))
            undone += cursor.rowcount
        conn.execute("UPDATE basepipe_caption_bulk_operations SET status = 'undone', undone_at = CURRENT_TIMESTAMP WHERE id = ?", (payload.operation_id,))
        conn.commit()
        return {"operation_id": payload.operation_id, "project_id": project_id, "undone_count": undone, "status": "undone"}
    finally:
        conn.close()


@router.post("/projects/{project_id}/snapshots")
@router.post("/projects/{project_id}/dataset-snapshots")
def create_snapshot(project_id: int, payload: BasepipeSnapshotCreate) -> dict:
    conn = get_conn()
    try:
        _project_or_404(conn, project_id)
        if payload.concept_id is not None:
            concept = conn.execute("SELECT id FROM basepipe_concepts WHERE id = ? AND project_id = ?", (payload.concept_id, project_id)).fetchone()
            if concept is None:
                raise HTTPException(status_code=400, detail="concept does not belong to project")

        outfit_profiles: list[dict] = []
        outfit_row = conn.execute(
            "SELECT profiles_json FROM project_outfit_profiles WHERE project_id = ?",
            (project_id,),
        ).fetchone()
        if outfit_row is not None:
            try:
                parsed_profiles = json.loads(outfit_row["profiles_json"] or "[]")
                if isinstance(parsed_profiles, list):
                    outfit_profiles = [profile for profile in parsed_profiles if isinstance(profile, dict)]
            except (TypeError, ValueError):
                outfit_profiles = []

        asset_ids = list(dict.fromkeys(payload.asset_ids))
        for item_id in payload.dataset_item_ids:
            item = conn.execute("SELECT * FROM dataset_items WHERE id = ? AND project_id = ?", (item_id, project_id)).fetchone()
            if item is None:
                raise HTTPException(status_code=400, detail=f"dataset item not found: {item_id}")
            path = Path(item["file_path"])
            if not path.is_file():
                raise HTTPException(status_code=400, detail=f"dataset asset file not found: {path}")
            digest = _sha256(path)
            existing = conn.execute(
                "SELECT id FROM basepipe_assets WHERE project_id = ? AND asset_key = ? AND content_sha256 = ?",
                (project_id, path.stem, digest),
            ).fetchone()
            if existing is None:
                cur = conn.execute(
                    "INSERT INTO basepipe_assets(project_id, asset_key, file_path, content_sha256, origin_kind, caption, caption_source, review_status) VALUES (?, ?, ?, ?, 'dataset_item', ?, ?, 'pending')",
                    (project_id, path.stem, str(path), digest, item["caption"], item["caption_source"]),
                )
                asset_ids.append(int(cur.lastrowid))
            else:
                asset_ids.append(int(existing["id"]))

        if not asset_ids:
            raise HTTPException(status_code=400, detail="snapshot requires at least one asset")
        marks = ",".join("?" for _ in asset_ids)
        assets = conn.execute(
            f"SELECT id, project_id, file_path, content_sha256, asset_type, origin_kind, source_ref, caption, caption_source, review_status, user_rating, training_enabled, training_weight, metadata_json FROM basepipe_assets WHERE project_id = ? AND id IN ({marks})",
            (project_id, *asset_ids),
        ).fetchall()
        if len(assets) != len(asset_ids):
            raise HTTPException(status_code=400, detail="one or more assets do not belong to project")
        from .evaluation import read_reference,digest
        evaluation=read_reference(project_id,conn)
        if evaluation:
            assets=[row for row in assets if digest(row['file_path'])!=evaluation['sha256']]
            asset_ids=[int(row['id']) for row in assets]
            if not assets:raise HTTPException(400,'評価画像を除くと学習対象が0枚になります')
        pending = [str(row["id"]) for row in assets if row["review_status"] != "approved"]
        if pending:
            raise HTTPException(status_code=400, detail=f"Snapshotにはapproved Assetだけを指定してください: {', '.join(pending)}")
        training_input_by_asset: dict[int, dict] = {}
        aesthetic_approval_by_asset: dict[int, dict] = {}
        for row in assets:
            if row["asset_type"] != "image":
                raise HTTPException(status_code=400, detail=f"学習Snapshotには画像Assetだけを指定してください: {row['id']}")
            if row["origin_kind"] == "h3_frame":
                source_ref = str(row["source_ref"] or "")
                try:
                    generation_run_id = int(source_ref.rsplit(":", 1)[1])
                except (IndexError, ValueError):
                    raise HTTPException(status_code=400, detail=f"H3 Frame #{row['id']} のRun lineageが不正です")
                generation_run = conn.execute("SELECT current_stage, output_manifest_json FROM basepipe_character_generation_runs WHERE id = ? AND project_id = ?", (generation_run_id, project_id)).fetchone()
                if generation_run is None:
                    raise HTTPException(status_code=400, detail=f"H3 Frame #{row['id']} のGeneration Runが見つかりません")
                generation_output = json.loads(generation_run["output_manifest_json"] or "{}")
                identity_review = generation_output.get("identity_review") if isinstance(generation_output.get("identity_review"), dict) else {}
                source_metadata = json.loads(row["metadata_json"] or "{}")
                current_frame_ids = _current_character_generation_frame_ids(generation_run)
                if current_frame_ids is not None and int(row["id"]) not in current_frame_ids:
                    raise HTTPException(status_code=400, detail=f"H3 Frame #{row['id']} はGeneration Runの現行manifestに含まれません")
                generation_stage = generation_run["current_stage"] or "planning"
                if CHARACTER_GENERATION_STAGES.index(generation_stage) < CHARACTER_GENERATION_STAGES.index("captioning"):
                    raise HTTPException(status_code=400, detail=f"H3 Frame #{row['id']} はQwen補正・Balancing・Captioning完了前のためSnapshot化できません")
                if identity_review.get("aesthetic_review_state") != USER_APPROVED or identity_review.get("reviewed_by") != "user":
                    raise HTTPException(status_code=400, detail=f"H3 Frame #{row['id']} はFrame選別のユーザーIdentity Reviewが未承認です")
                if source_metadata.get("identity_review_state") != USER_APPROVED or source_metadata.get("identity_reviewed_by") != "user":
                    raise HTTPException(status_code=400, detail=f"H3 Frame #{row['id']} はAsset側のユーザーIdentity Review証拠が欠落しています")
                training_input = _training_input_selection(conn, row)
                if training_input is None:
                    raise HTTPException(status_code=400, detail=f"H3 Frame #{row['id']} はOriginal / Enhanced Training Inputが未選択です")
                training_input_by_asset[int(row["id"])] = training_input
                aesthetic_approval_by_asset[int(row["id"])] = {
                    "identity_review_state": source_metadata.get("identity_review_state"),
                    "identity_reviewed_by": source_metadata.get("identity_reviewed_by"),
                    "identity_reviewed_at": source_metadata.get("identity_reviewed_at"),
                    "training_input_review_state": source_metadata.get("training_input_review_state"),
                    "training_input_reviewed_by": source_metadata.get("training_input_reviewed_by"),
                    "training_input_reviewed_at": source_metadata.get("training_input_reviewed_at"),
                }
                if not str(row["caption"] or "").strip() or row["caption_source"] != "h3_captioning":
                    raise HTTPException(status_code=400, detail=f"H3 Frame #{row['id']} はCaptioning済みの学習入力が必要です")
        disabled = [str(row["id"]) for row in assets if not int(row["training_enabled"] or 0)]
        if disabled:
            raise HTTPException(status_code=400, detail=f"学習無効AssetはSnapshotへ追加できません: {', '.join(disabled)}")
        sorted_assets = sorted(assets, key=lambda row: int(row["id"]))
        manifest_assets = []
        for row in sorted_assets:
            input_path = str(row["file_path"])
            input_sha256 = str(row["content_sha256"])
            entry = {
                "id": int(row["id"]),
                "path": input_path,
                "sha256": input_sha256,
                "caption": row["caption"],
                "user_rating": row["user_rating"],
                "training_enabled": bool(row["training_enabled"]),
                "training_weight": float(row["training_weight"] or 1.0),
            }
            if row["origin_kind"] == "h3_frame":
                training_input = training_input_by_asset[int(row["id"])]
                approval = aesthetic_approval_by_asset[int(row["id"])]
                source_metadata = json.loads(row["metadata_json"] or "{}")
                source_lineage = source_metadata.get("lineage") if isinstance(source_metadata.get("lineage"), dict) else {}
                entry["path"] = str(training_input["file_path"])
                entry["sha256"] = str(training_input["content_sha256"])
                entry["lineage"] = {
                    "source_frame_asset_id": int(row["id"]),
                    "source_frame_path": input_path,
                    "source_frame_sha256": input_sha256,
                    "input_source": training_input["input_source"],
                    "qwen_version_id": int(training_input["id"]) if training_input["input_source"] == "qwen_version" else None,
                    "qwen_file_path": training_input["file_path"] if training_input["input_source"] == "qwen_version" else None,
                    "qwen_sha256": training_input["content_sha256"] if training_input["input_source"] == "qwen_version" else None,
                    "qwen_version_kind": training_input.get("version_kind") if training_input["input_source"] == "qwen_version" else None,
                    "enhanced_version_id": int(training_input["id"]) if training_input["input_source"] == "enhanced_version" else None,
                    "enhanced_file_path": training_input["file_path"] if training_input["input_source"] == "enhanced_version" else None,
                    "enhanced_sha256": training_input["content_sha256"] if training_input["input_source"] == "enhanced_version" else None,
                    "enhanced_version_kind": training_input.get("version_kind") if training_input["input_source"] == "enhanced_version" else None,
                    "generation_run_id": source_lineage.get("run_id"),
                    "clip_id": source_lineage.get("clip_id"),
                    "clip_index": source_lineage.get("clip_index"),
                    "clip_frame_index": source_lineage.get("clip_frame_index"),
                    "timestamp_seconds": source_lineage.get("timestamp_seconds"),
                    "generation_seed": source_lineage.get("generation_seed"),
                    "generation_prompt": source_lineage.get("generation_prompt"),
                    "coverage": source_lineage.get("coverage") if isinstance(source_lineage.get("coverage"), dict) else {},
                    "aesthetic_approval": approval,
                }
            manifest_assets.append(entry)
        manifest = {
            "schema": "basepipe.snapshot.v1",
            "project_id": project_id,
            "concept_id": payload.concept_id,
            "outfit_profiles": outfit_profiles,
            "assets": manifest_assets,
        }
        snapshot_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

        def handoff_training_draft(target_snapshot_id: int, target_profile_snapshot_id: int | None) -> dict:
            project_row = conn.execute(
                "SELECT training_config_json FROM projects WHERE id=?",
                (project_id,),
            ).fetchone()
            try:
                draft = json.loads(project_row["training_config_json"] or "{}") if project_row else {}
                if not isinstance(draft, dict):
                    draft = {}
            except (TypeError, ValueError):
                draft = {}
            draft.update({
                "dataset_snapshot_id": target_snapshot_id,
                "preview_profile_snapshot_id": target_profile_snapshot_id,
                "model_family": "anima",
                "training_engine": "auto",
                "gpu_device_id": 1,
            })
            conn.execute(
                "UPDATE projects SET training_config_json=? WHERE id=?",
                (json.dumps(draft, ensure_ascii=False), project_id),
            )
            return draft

        def ensure_anima_preview_profile() -> dict:
            existing = conn.execute(
                "SELECT id, name FROM preview_profiles WHERE project_id=? AND model_family='anima' "
                "ORDER BY updated_at DESC, id DESC LIMIT 1",
                (project_id,),
            ).fetchone()
            if existing is not None:
                return {"id": int(existing["id"]), "name": str(existing["name"]), "created": False}
            concept_row = conn.execute(
                "SELECT trigger_token FROM basepipe_concepts WHERE id=? AND project_id=?",
                (payload.concept_id, project_id),
            ).fetchone() if payload.concept_id is not None else None
            trigger = str(concept_row["trigger_token"] or "character") if concept_row else str((conn.execute("SELECT trigger_token FROM basepipe_concepts WHERE project_id=? AND TRIM(trigger_token)!='' ORDER BY id LIMIT 1",(project_id,)).fetchone() or ["character"])[0])
            profile_name = f"{payload.name} · Identity Baseline"
            profile_cur = conn.execute(
                "INSERT INTO preview_profiles(project_id, name, prompt, negative_prompt, seed, resolution, steps, cfg, sampler, scheduler, model_family) "
                "VALUES (?, ?, ?, ?, 42, 1024, 7, 1.0, 'euler', 'simple', 'anima')",
                (
                    project_id,
                    profile_name,
                    f"{trigger}, solo, full body, neutral pose, simple background",
                    "low quality, blurry, duplicate, extra limbs, deformed hands",
                ),
            )
            return {"id": int(profile_cur.lastrowid), "name": profile_name, "created": True}

        existing_snapshot = conn.execute(
            "SELECT id, name, item_count FROM basepipe_dataset_snapshots "
            "WHERE project_id=? AND snapshot_hash=? AND status='sealed' ORDER BY id DESC LIMIT 1",
            (project_id, snapshot_hash),
        ).fetchone()
        if existing_snapshot is not None:
            snapshot_id = int(existing_snapshot["id"])
            profile_info = ensure_anima_preview_profile()
            preview_profile_snapshot = _freeze_preview_profile_row(conn, project_id, int(profile_info["id"]))
            draft = handoff_training_draft(
                snapshot_id,
                int(preview_profile_snapshot["id"]) if preview_profile_snapshot else None,
            )
            conn.commit()
            return {
                "id": snapshot_id,
                "project_id": project_id,
                "concept_id": payload.concept_id,
                "name": str(existing_snapshot["name"]),
                "snapshot_hash": snapshot_hash,
                "status": "sealed",
                "item_count": int(existing_snapshot["item_count"]),
                "manifest": manifest,
                "preview_profile": (
                    profile_info
                ),
                "preview_profile_snapshot": preview_profile_snapshot,
                "training_draft": draft,
                "reused": True,
            }
        cur = conn.execute(
            "INSERT INTO basepipe_dataset_snapshots(project_id, concept_id, name, snapshot_hash, item_count, manifest_json) VALUES (?, ?, ?, ?, ?, ?)",
            (project_id, payload.concept_id, payload.name, snapshot_hash, len(assets), json.dumps(manifest, ensure_ascii=False)),
        )
        snapshot_id = int(cur.lastrowid)
        for ordinal, (row, manifest_entry) in enumerate(zip(sorted_assets, manifest_assets)):
            # The snapshot INSERT holds the write transaction: a concurrent deletion
            # cannot invalidate registrations after this recheck and before commit.
            if conn.execute("SELECT id FROM basepipe_assets WHERE id=?", (row['id'],)).fetchone() is None:
                raise HTTPException(409, "Snapshot作成中に画像登録が変更されました。画像一覧を再読込してください。")
            entry_metadata = json.loads(row["metadata_json"] or "{}")
            entry_metadata["snapshot"] = {
                "path": manifest_entry["path"],
                "sha256": manifest_entry["sha256"],
                "source_path": row["file_path"],
                "source_sha256": row["content_sha256"],
                "input_source": (manifest_entry.get("lineage") or {}).get("input_source") if row["origin_kind"] == "h3_frame" else "source_asset",
                "version_id": (manifest_entry.get("lineage") or {}).get("qwen_version_id"),
                "asset_type": row["asset_type"],
                "origin_kind": row["origin_kind"],
                "review_status": row["review_status"],
                "user_rating": row["user_rating"],
                "training_enabled": bool(row["training_enabled"]),
                "training_weight": float(row["training_weight"] or 1.0),
            }
            if row["origin_kind"] == "h3_frame":
                approval = aesthetic_approval_by_asset[int(row["id"])]
                entry_metadata["snapshot"].update({
                    "requires_user_aesthetic_approval": True,
                    **approval,
                })
            conn.execute(
                "INSERT INTO basepipe_snapshot_entries(snapshot_id, asset_id, ordinal, caption_at_snapshot, metadata_json) VALUES (?, ?, ?, ?, ?)",
                (snapshot_id, int(row["id"]), ordinal, row["caption"], json.dumps(entry_metadata, ensure_ascii=False)),
            )
        generation_run_ids = set()
        for row in assets:
            if row["origin_kind"] == "h3_frame":
                try:
                    generation_run_ids.add(int(str(row["source_ref"]).rsplit(":", 1)[1]))
                except (IndexError, ValueError):
                    pass
        preview_profile: dict | None = None
        if len(generation_run_ids) == 1:
            generation_run_id = next(iter(generation_run_ids))
            generation_run = conn.execute(
                "SELECT stage_history_json, output_manifest_json, current_stage FROM basepipe_character_generation_runs "
                "WHERE id = ? AND project_id = ?",
                (generation_run_id, project_id),
            ).fetchone()
            if generation_run is not None:
                history = json.loads(generation_run["stage_history_json"] or "[]")
                snapshot_evidence = {"snapshot_id": snapshot_id, "snapshot_hash": snapshot_hash, "item_count": len(assets)}
                history.append({"stage": "snapshot", "status": "completed", "artifact_manifest": snapshot_evidence})
                output_manifest = json.loads(generation_run["output_manifest_json"] or "{}")
                output_manifest["snapshot"] = snapshot_evidence
                conn.execute(
                    "UPDATE basepipe_character_generation_runs SET status='completed', current_stage='snapshot', "
                    "stage_history_json=?, output_manifest_json=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                    (json.dumps(history, ensure_ascii=False), json.dumps(output_manifest, ensure_ascii=False), generation_run_id),
                )
            existing_profile = conn.execute(
                "SELECT id, name FROM preview_profiles WHERE project_id=? AND model_family='anima' ORDER BY updated_at DESC, id DESC LIMIT 1",
                (project_id,),
            ).fetchone()
            if existing_profile is None:
                concept_row = conn.execute(
                    "SELECT trigger_token FROM basepipe_concepts WHERE id=? AND project_id=?",
                    (payload.concept_id, project_id),
                ).fetchone() if payload.concept_id is not None else None
                trigger = str(concept_row["trigger_token"] or "character") if concept_row else str((conn.execute("SELECT trigger_token FROM basepipe_concepts WHERE project_id=? AND TRIM(trigger_token)!='' ORDER BY id LIMIT 1",(project_id,)).fetchone() or ["character"])[0])
                profile_cur = conn.execute(
                    "INSERT INTO preview_profiles(project_id, name, prompt, negative_prompt, seed, resolution, steps, cfg, sampler, scheduler, model_family) "
                    "VALUES (?, ?, ?, ?, 42, 1024, 7, 1.0, 'euler', 'simple', 'anima')",
                    (
                        project_id,
                        f"{payload.name} · Identity Baseline",
                        f"{trigger}, solo, full body, neutral pose, simple background",
                        "low quality, blurry, duplicate, extra limbs, deformed hands",
                    ),
                )
                preview_profile = {"id": int(profile_cur.lastrowid), "name": f"{payload.name} · Identity Baseline", "created": True}
            else:
                preview_profile = {"id": int(existing_profile["id"]), "name": str(existing_profile["name"]), "created": False}
        if preview_profile is None:
            preview_profile = ensure_anima_preview_profile()
        preview_profile_snapshot = (
            _freeze_preview_profile_row(conn, project_id, int(preview_profile["id"]))
            if preview_profile else None
        )
        training_draft = handoff_training_draft(
            snapshot_id,
            int(preview_profile_snapshot["id"]) if preview_profile_snapshot else None,
        )
        conn.commit()
        return {"id": snapshot_id, "project_id": project_id, "concept_id": payload.concept_id, "name": payload.name,
                "snapshot_hash": snapshot_hash, "status": "sealed", "item_count": len(assets), "manifest": manifest,
                "preview_profile": preview_profile, "preview_profile_snapshot": preview_profile_snapshot,
                "training_draft": training_draft, "reused": False}
    except HTTPException:
        conn.rollback()
        raise
    except Exception as exc:
        conn.rollback()
        raise HTTPException(status_code=400, detail=f"snapshot create failed: {exc}") from exc
    finally:
        conn.close()


@router.get("/projects/{project_id}/snapshots")
@router.get("/projects/{project_id}/dataset-snapshots")
def list_snapshots(project_id: int) -> dict:
    """List sealed Dataset Snapshots for project-level lineage navigation."""
    conn = get_conn()
    try:
        _project_or_404(conn, project_id)
        rows = conn.execute(
            "SELECT id, project_id, concept_id, name, snapshot_hash, status, item_count, created_at "
            "FROM basepipe_dataset_snapshots WHERE project_id = ? ORDER BY id DESC",
            (project_id,),
        ).fetchall()
        return {"project_id": project_id, "snapshots": [dict(row) for row in rows]}
    finally:
        conn.close()


@router.get("/snapshots/{snapshot_id}")
def get_snapshot(snapshot_id: int) -> dict:
    conn = get_conn()
    try:
        row = conn.execute("SELECT * FROM basepipe_dataset_snapshots WHERE id = ?", (snapshot_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="snapshot not found")
        entries = [dict(r) for r in conn.execute(
            "SELECT e.snapshot_id, e.asset_id, e.ordinal, e.caption_at_snapshot, e.metadata_json, "
            "a.file_path AS source_current_path, a.content_sha256 AS source_current_sha256 "
            "FROM basepipe_snapshot_entries e JOIN basepipe_assets a ON a.id = e.asset_id WHERE e.snapshot_id = ? ORDER BY e.ordinal", (snapshot_id,)
        ).fetchall()]
        for entry in entries:
            entry["metadata"] = json.loads(entry.pop("metadata_json") or "{}")
            frozen = entry["metadata"].get("snapshot")
            frozen = frozen if isinstance(frozen, dict) else {}
            entry["file_path"] = str(frozen.get("path") or entry["source_current_path"])
            entry["content_sha256"] = str(frozen.get("sha256") or entry["source_current_sha256"])
        result = dict(row)
        result["manifest"] = json.loads(result.pop("manifest_json"))
        result["entries"] = entries
        return result
    finally:
        conn.close()


@router.get("/projects/{project_id}/preview-profile-snapshots")
def list_preview_profile_snapshots(project_id: int) -> dict:
    conn = get_conn()
    try:
        _project_or_404(conn, project_id)
        rows = conn.execute(
            "SELECT id, project_id, source_profile_id, name, snapshot_hash, payload_json, created_at "
            "FROM basepipe_preview_profile_snapshots WHERE project_id = ? ORDER BY id DESC", (project_id,)
        ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["payload"] = json.loads(item.pop("payload_json") or "{}")
            result.append(item)
        return {"project_id": project_id, "snapshots": result}
    finally:
        conn.close()


@router.get("/projects/{project_id}/evaluations")
def list_evaluations(project_id: int) -> dict:
    conn = get_conn()
    try:
        _project_or_404(conn, project_id)
        rows = conn.execute("SELECT * FROM basepipe_evaluations WHERE project_id = ? ORDER BY updated_at DESC", (project_id,)).fetchall()
        return {"project_id": project_id, "evaluations": [dict(row) for row in rows]}
    finally:
        conn.close()


@router.post("/projects/{project_id}/evaluations")
def save_evaluation(project_id: int, payload: BasepipeEvaluationIn) -> dict:
    if payload.accepted:
        _require_user_aesthetic_confirmation(payload.human_confirmed)
    conn = get_conn()
    try:
        _project_or_404(conn, project_id)
        checkpoint = conn.execute("SELECT id FROM checkpoints WHERE id = ? AND project_id = ?", (payload.checkpoint_id, project_id)).fetchone()
        if checkpoint is None:
            raise HTTPException(status_code=400, detail="checkpoint does not belong to project")
        if payload.preview_profile_snapshot_id is not None:
            frozen = conn.execute(
                "SELECT id FROM basepipe_preview_profile_snapshots WHERE id = ? AND project_id = ?",
                (payload.preview_profile_snapshot_id, project_id),
            ).fetchone()
            if frozen is None:
                raise HTTPException(status_code=400, detail="Preview SnapshotがこのProjectに属していません")
        elif payload.accepted:
            raise HTTPException(status_code=400, detail="採用評価にはFrozen Preview Snapshotが必要です")
        if payload.accepted:
            slot_match = re.fullmatch(r"p(\d+)_i(\d+)", payload.preview_slot)
            if slot_match is None:
                raise HTTPException(status_code=400, detail="採用評価のPreview slotが実Job形式ではありません")
            job = conn.execute(
                "SELECT status, output_path, preview_snapshot_json FROM preview_jobs "
                "WHERE checkpoint_id = ? AND prompt_index = ? AND instance_index = ?",
                (payload.checkpoint_id, int(slot_match.group(1)), int(slot_match.group(2))),
            ).fetchone()
            if job is None or job["status"] != "succeeded" or not Path(str(job["output_path"] or "")).is_file():
                raise HTTPException(status_code=400, detail="実画像が存在する成功済みPreview Jobだけを採用評価できます")
            try:
                job_snapshot = json.loads(job["preview_snapshot_json"] or "{}")
            except (TypeError, ValueError):
                job_snapshot = {}
            if job_snapshot.get("profile_snapshot_id") != payload.preview_profile_snapshot_id:
                raise HTTPException(status_code=409, detail="表示中Previewと固定Profile Snapshotが一致しません。再読み込みしてください")
        review_state = USER_APPROVED if payload.accepted else AESTHETIC_UNREVIEWED
        reviewed_by = "user" if payload.accepted else ""
        reviewed_at = datetime.now(timezone.utc).isoformat() if payload.accepted else None
        values = (project_id, payload.checkpoint_id, payload.preview_slot, payload.preview_profile_snapshot_id, payload.identity, payload.outfit_separation, payload.style_durability, payload.style_quality, payload.prompt_flexibility, payload.artifact, int(payload.accepted), payload.note, review_state, reviewed_by, reviewed_at)
        conn.execute(
            "INSERT INTO basepipe_evaluations(project_id, checkpoint_id, preview_slot, preview_profile_snapshot_id, identity, outfit_separation, style_durability, style_quality, prompt_flexibility, artifact, accepted, note, review_state, reviewed_by, reviewed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(project_id, checkpoint_id, preview_slot) DO UPDATE SET preview_profile_snapshot_id=excluded.preview_profile_snapshot_id, identity=excluded.identity, outfit_separation=excluded.outfit_separation, style_durability=excluded.style_durability, style_quality=excluded.style_quality, prompt_flexibility=excluded.prompt_flexibility, artifact=excluded.artifact, accepted=excluded.accepted, note=excluded.note, review_state=excluded.review_state, reviewed_by=excluded.reviewed_by, reviewed_at=excluded.reviewed_at, updated_at=CURRENT_TIMESTAMP",
            values,
        )
        conn.commit()
        row = conn.execute("SELECT * FROM basepipe_evaluations WHERE project_id = ? AND checkpoint_id = ? AND preview_slot = ?", (project_id, payload.checkpoint_id, payload.preview_slot)).fetchone()
        return dict(row)
    except HTTPException:
        conn.rollback()
        raise
    except Exception as exc:
        conn.rollback()
        raise HTTPException(status_code=400, detail=f"evaluation save failed: {exc}") from exc
    finally:
        conn.close()


def _freeze_preview_profile_row(conn, project_id: int, profile_id: int) -> dict:
    has_outfits = any(r[1] == "outfits_json" for r in conn.execute("PRAGMA table_info(preview_profiles)"))  # older DB files migrate on server start
    profile = conn.execute(
        "SELECT id, project_id, name, prompt, negative_prompt, seed, resolution, steps, cfg, sampler, scheduler, model_family"
        + (", outfits_json " if has_outfits else " ") +
        "FROM preview_profiles WHERE id = ? AND (project_id = ? OR project_id IS NULL)",
        (profile_id, project_id),
    ).fetchone()
    if profile is None:
        raise HTTPException(status_code=404, detail="preview profile not found")
    payload = dict(profile)
    payload.pop("project_id", None)
    outfits_raw = payload.pop("outfits_json", "") or ""
    if outfits_raw:  # profiles without per-outfit settings keep their old hash/payload shape
        payload["outfits"] = json.loads(outfits_raw)
    snapshot_hash = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    existing = conn.execute(
        "SELECT id FROM basepipe_preview_profile_snapshots WHERE project_id=? AND source_profile_id=? AND snapshot_hash=? "
        "ORDER BY id DESC LIMIT 1",
        (project_id, profile_id, snapshot_hash),
    ).fetchone()
    if existing is not None:
        return {"id": int(existing["id"]), "project_id": project_id, "source_profile_id": profile_id,
                "name": profile["name"], "snapshot_hash": snapshot_hash, "payload": payload, "created": False}
    cur = conn.execute(
        "INSERT INTO basepipe_preview_profile_snapshots(project_id, source_profile_id, name, snapshot_hash, payload_json) VALUES (?, ?, ?, ?, ?)",
        (project_id, profile_id, profile["name"], snapshot_hash, json.dumps(payload, ensure_ascii=False)),
    )
    return {"id": int(cur.lastrowid), "project_id": project_id, "source_profile_id": profile_id,
            "name": profile["name"], "snapshot_hash": snapshot_hash, "payload": payload, "created": True}


@router.post("/projects/{project_id}/preview-profile-snapshots/{profile_id}")
def freeze_preview_profile(project_id: int, profile_id: int) -> dict:
    """Freeze a Preview Profile's exact condition for reproducible Compare runs."""
    conn = get_conn()
    try:
        _project_or_404(conn, project_id)
        result = _freeze_preview_profile_row(conn, project_id, profile_id)
        conn.commit()
        return result
    except HTTPException:
        conn.rollback()
        raise
    except Exception as exc:
        conn.rollback()
        raise HTTPException(status_code=400, detail=f"preview profile freeze failed: {exc}") from exc
    finally:
        conn.close()
