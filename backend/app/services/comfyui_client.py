"""ComfyUI HTTP client — SPEC §14 Dataset Refinery エンジン接続。

LoRA Basepipe 自身は画像処理エンジンを持たない（SPEC §6 OSS First / §7 Non Goals）。
文字削除などの前処理は ComfyUI Workflow をエンジンとして実行する。

文字削除パイプライン（全て ComfyUI 内で完結・バックエンド追加依存なし）:

    LoadImage
      ├─→ ApplyCLIPSeg(prompt="text", threshold, dilate) ─→ MASK
      │                                                       ↓
      └───────────────────→ INPAINT_InpaintWithModel(big-lama) ─→ PreviewImage

出力は SaveImage ではなく PreviewImage(temp 保存) を使う。これにより LoRA アプリの
前処理画像が ComfyUI の output フォルダ（ユーザーのギャラリー）を汚染しない。

CLIPSeg をテキスト領域検出に用いる。Florence2 系ノードは ComfyUI 同梱 transformers と
非互換（forced_bos_token_id エラー）でモデルを問わずロード不能のため不採用。
CLIPSeg は標準 transformers 互換モデル (CIDAS/clipseg) を使用し安定動作する。
"""
from __future__ import annotations

import io
import json
import logging
import os
import shutil
import time
import urllib.parse
import uuid
from pathlib import Path

import requests

from ..db import get_conn

logger = logging.getLogger(__name__)

DEFAULT_COMFYUI_URL = "http://127.0.0.1:8188"
DEFAULT_COMFY_TASK_GATE_PATH = Path(
    r"C:\Users\Keishi\AI_tools\ComfyUI-Portable\ComfyUI_windows_portable\ComfyUI\user\instance_gate.token"
)

# 文字削除ワークフローが要求するカスタムノード。
# テキスト領域検出は CLIPSeg(ComfyUI_essentials)、穴埋めは LaMa(ComfyUI-Inpaint-Nodes)。
# 出力は PreviewImage(コア・temp保存) を使い ComfyUI の output フォルダを汚染しない。
REQUIRED_NODES_TEXT_REMOVAL = (
    "LoadImage",
    "PreviewImage",
    "LoadCLIPSegModels+",
    "ApplyCLIPSeg+",
    "INPAINT_LoadInpaintModel",
    "INPAINT_InpaintWithModel",
)

LAMA_MODEL_NAME = "big-lama.pt"

# CLIPSeg のテキスト検出プロンプト（既定）
DEFAULT_TEXT_PROMPT = "text"

# アップスケール（解像度標準化の拡大）用ノード・モデル
UPSCALE_NODES = ("LoadImage", "PreviewImage", "UpscaleModelLoader", "ImageUpscaleWithModel")
# アニメ／イラスト向けを優先（LoRA データセット用途）
UPSCALE_PREFERRED = (
    "realesrganX4plusAnime_v1.pth",
    "4x-UltraSharp.pth",
    "RealESRGAN_x4plus.pth",
    "OmniSR_X4_DIV2K.safetensors",
)

# Qwen-Image-Edit-2511 native ComfyUI workflow constants.  Keep the preferred
# order explicit so Runtime Evidence records the actual weight variant instead
# of silently falling back to the old Qwen-Rapid checkpoint.
QWEN_EDIT_2511_MODELS = (
    "qwen_image_edit_2511_int8_convrot.safetensors",
    "qwen_image_edit_2511_fp8mixed.safetensors",
    "qwen_image_edit_2511_bf16.safetensors",
)
QWEN_EDIT_2511_CLIP = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
QWEN_EDIT_2511_VAE = "qwen_image_vae.safetensors"
QWEN_EDIT_2511_LIGHTNING = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
QWEN_EDIT_2511_MULTIPLE_ANGLES = "qwen-image-edit-2511-multiple-angles-lora.safetensors"
QWEN_CLEANUP_PROMPT = (
    "学習データセットにしたいので、背景や吹き出し、ノイズを消して\nキャラクターはそのままで"
)

# Character Dataset Factory H3 candidate-generation profile.  This is a
# parameterised derivative of the user-selected rendering-engine anchor:
# ``anime_action_refmod_directed_api.json`` (prompt
# 6dc04bdf-505c-4727-bbeb-90fce01aed64).  Keep the provenance explicit: the
# original workflow is read-only and this application only reproduces its
# identity-retention / sampler stack with project-owned references and prompts.
H3_DATASET_ANCHOR_PROFILE = "h3-refmod-turbo-spectrum-v1"
H3_DATASET_ANCHOR_PROMPT_ID = "6dc04bdf-505c-4727-bbeb-90fce01aed64"
H3_DATASET_ANCHOR_WORKFLOW_SHA256 = "f55fcb8d3b022329d1404924c97cc3d79eb57a662b7cdc3b0c7f6e719ceb8818"
H3_TURBO_LORA = "minimax_h3_turbo_v4_step600_ema.safetensors"
QWEN_REQUIRED_NODES = (
    "UNETLoader",
    "CLIPLoader",
    "VAELoader",
    "LoadImage",
    "TextEncodeQwenImageEditPlus",
    "FluxKontextImageScale",
    "FluxKontextMultiReferenceLatentMethod",
    "ModelSamplingAuraFlow",
    "CFGNorm",
    "VAEEncode",
    "KSampler",
    "VAEDecode",
    "SaveImage",
)


def get_comfyui_url() -> str:
    """app_settings の comfyui_url を読む。未設定なら既定 (localhost:8188)。"""
    try:
        conn = get_conn()
        row = conn.execute(
            "SELECT value FROM app_settings WHERE key = ?", ("comfyui_url",)
        ).fetchone()
        conn.close()
        if row and str(row["value"]).strip():
            return str(row["value"]).strip().rstrip("/")
    except Exception as exc:  # noqa: BLE001
        logger.warning("get_comfyui_url failed: %s", exc)
    return DEFAULT_COMFYUI_URL


class ComfyUIError(RuntimeError):
    """ComfyUI 接続・実行エラー。"""


class ComfyUIUnavailableError(ComfyUIError):
    """A previously reachable local ComfyUI process disappeared mid-job."""


class ComfyUIClient:
    def __init__(self, base_url: str | None = None, timeout: float = 15.0) -> None:
        self.base_url = (base_url or get_comfyui_url()).rstrip("/")
        self.timeout = timeout
        self.client_id = str(uuid.uuid4())

    @staticmethod
    def _task_router_headers() -> dict[str, str]:
        """Attach the local managed-instance admission proof to mutations.

        The token is read only at request time and is never logged or persisted
        by this application.  Read-only endpoints remain available without it.
        """
        configured = os.environ.get("COMFY_INSTANCE_GATE_TOKEN_PATH", "").strip()
        token_path = Path(configured) if configured else DEFAULT_COMFY_TASK_GATE_PATH
        try:
            token = token_path.read_text(encoding="utf-8").strip()
        except OSError:
            return {}
        if not token:
            return {}
        return {"X-Comfy-Task-Gate": token, "X-Comfy-Task-Router": "1"}

    # ── 疎通・能力チェック（doctor） ──────────────────────────────────
    def check_status(self) -> dict:
        """ComfyUI の到達性・必要ノード・LaMa モデルの有無を返す。"""
        result: dict = {
            "reachable": False,
            "url": self.base_url,
            "version": None,
            "missing_nodes": [],
            "nodes_ok": False,
            "lama_model": None,
            "lama_ok": False,
            "ready": False,
            "message": "",
        }
        try:
            r = requests.get(f"{self.base_url}/system_stats", timeout=self.timeout)
            r.raise_for_status()
            stats = r.json()
            result["reachable"] = True
            result["version"] = stats.get("system", {}).get("comfyui_version")
        except Exception as exc:  # noqa: BLE001
            result["message"] = f"ComfyUI に接続できません ({self.base_url}): {exc}"
            return result

        try:
            r = requests.get(f"{self.base_url}/object_info", timeout=30)
            r.raise_for_status()
            info = r.json()
        except Exception as exc:  # noqa: BLE001
            result["message"] = f"object_info 取得失敗: {exc}"
            return result

        missing = [n for n in REQUIRED_NODES_TEXT_REMOVAL if n not in info]
        result["missing_nodes"] = missing
        result["nodes_ok"] = not missing

        # LaMa モデル
        lama_list = self._extract_combo(info, "INPAINT_LoadInpaintModel", "model_name")
        if lama_list:
            picked = LAMA_MODEL_NAME if LAMA_MODEL_NAME in lama_list else lama_list[0]
            result["lama_model"] = picked
            result["lama_ok"] = bool(picked)

        # アップスケール能力（任意・解像度標準化の拡大用）
        up_missing = [n for n in UPSCALE_NODES if n not in info]
        result["upscale_nodes_ok"] = not up_missing
        up_list = self._extract_combo(info, "UpscaleModelLoader", "model_name")
        result["upscale_models"] = up_list
        if up_list:
            result["upscale_model"] = next((m for m in UPSCALE_PREFERRED if m in up_list), up_list[0])
        else:
            result["upscale_model"] = None

        result["ready"] = bool(result["reachable"] and result["nodes_ok"] and result["lama_ok"])
        if result["ready"]:
            result["message"] = "ComfyUI 準備完了"
        elif missing:
            result["message"] = "必要なカスタムノードが不足しています: " + ", ".join(missing)
        elif not result["lama_ok"]:
            result["message"] = f"LaMa モデル ({LAMA_MODEL_NAME}) が見つかりません"
        return result

    @staticmethod
    def _extract_combo(info: dict, node: str, field: str) -> list[str]:
        try:
            req = info[node]["input"]["required"]
            spec = req.get(field)
            if isinstance(spec, list) and spec:
                # 形式1: [["a","b",...], {opts}]
                if isinstance(spec[0], list):
                    return [str(x) for x in spec[0]]
                # 形式2: ["COMBO", {"options": ["a","b",...]}]
                if len(spec) > 1 and isinstance(spec[1], dict) and isinstance(spec[1].get("options"), list):
                    return [str(x) for x in spec[1]["options"]]
        except Exception:  # noqa: BLE001
            pass
        return []

    # ── 画像アップロード ──────────────────────────────────────────────
    def upload_image(self, path: Path) -> str:
        """画像を ComfyUI input にアップロードし、サーバ側ファイル名を返す。"""
        with open(path, "rb") as f:
            files = {"image": (path.name, f, "application/octet-stream")}
            data = {"overwrite": "true", "type": "input"}
            r = requests.post(
                f"{self.base_url}/upload/image", files=files, data=data,
                headers=self._task_router_headers(), timeout=60
            )
        r.raise_for_status()
        body = r.json()
        name = body.get("name")
        if not name:
            raise ComfyUIError(f"upload_image: 予期しない応答 {body}")
        subfolder = body.get("subfolder", "")
        return f"{subfolder}/{name}" if subfolder else name

    @staticmethod
    def build_h3_reference_to_video_graph(
        *,
        reference_image: str | list[str],
        prompt: str,
        seed: int,
        width: int = 768,
        height: int = 512,
        length: int = 56,
        ref_image_size: str = "match",
        model_name: str = "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
        clip_name: str = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        video_vae_name: str = "minimax_h3_video_vae_fp16.safetensors",
        audio_vae_name: str = "minimax_h3_audio_vae_fp32.safetensors",
        anchor_profile: str = H3_DATASET_ANCHOR_PROFILE,
        turbo_lora_name: str = H3_TURBO_LORA,
        turbo_lora_strength: float = 1.0,
        refmod_retention: float = 0.7,
        spectrum_blend_weight: float = 0.5,
    ) -> dict:
        """Build the local ComfyUI API graph for H3 reference-to-video.

        The graph deliberately uses the R2V path and one to four explicit
        reference images.  A character turnaround must not silently collapse to
        its first image: every uploaded view gets its own LoadImage node and
        keeps a stable index in ``ref_images``.
        The model path intentionally mirrors the user's selected H3 anchor:
        Turbo V4 EMA -> Sage -> memory-efficient Sage -> ChunkFFN -> Spectrum,
        with the front reference extracted as the canonical RefMod identity.
        H3 outputs are review artifacts and are not treated as training assets;
        the caller must record explicit user review/lineage before Snapshot
        adoption or selective high-quality correction.
        """
        if length < 22 or (length - 5) % 17:
            raise ValueError("H3 length must be 17n+5 (for example 56 or 124)")
        reference_images = [reference_image] if isinstance(reference_image, str) else list(reference_image)
        if not 1 <= len(reference_images) <= 4:
            raise ValueError("H3 reference images must contain 1 to 4 items")
        if anchor_profile != H3_DATASET_ANCHOR_PROFILE:
            raise ValueError(f"unsupported H3 anchor profile: {anchor_profile}")
        if not 0.0 <= refmod_retention <= 1.0:
            raise ValueError("H3 RefMod retention must be between 0 and 1")
        if not 0.0 <= spectrum_blend_weight <= 1.0:
            raise ValueError("H3 Spectrum blend weight must be between 0 and 1")
        reference_nodes = {
            ("ref" if index == 0 else f"ref_{index}"): {
                "class_type": "LoadImage",
                "inputs": {"image": image},
            }
            for index, image in enumerate(reference_images)
        }
        reference_links = {
            f"ref_image_{index}": [("ref" if index == 0 else f"ref_{index}"), 0]
            for index in range(len(reference_images))
        }
        return {
            **reference_nodes,
            "model": {"class_type": "UNETLoader", "inputs": {"unet_name": model_name, "weight_dtype": "default"}},
            "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip_name, "type": "minimax", "device": "default"}},
            "vae": {"class_type": "VAELoader", "inputs": {"vae_name": video_vae_name}},
            "audio_vae": {"class_type": "VAELoader", "inputs": {"vae_name": audio_vae_name}},
            "turbo_lora": {
                "class_type": "MiniMaxH3TurboLoRA",
                "inputs": {
                    "model": ["model", 0],
                    "lora_name": turbo_lora_name,
                    "strength": turbo_lora_strength,
                    "low_vram": False,
                },
            },
            "sage": {
                "class_type": "PathchSageAttentionKJ",
                "inputs": {"model": ["turbo_lora", 0], "sage_attention": "auto", "allow_compile": False},
            },
            "memory_sage": {
                "class_type": "MiniMaxH3MemoryEfficientSageAttentionPatch",
                "inputs": {"model": ["sage", 0]},
            },
            "chunk_ffn": {
                "class_type": "MiniMaxChunkFeedForward",
                "inputs": {"model": ["memory_sage", 0], "chunks": 2, "seq_threshold": 4096},
            },
            "spectrum": {
                "class_type": "SpectrumApplyMiniMaxH3",
                "inputs": {
                    "model": ["chunk_ffn", 0],
                    "enabled": True,
                    "blend_weight": spectrum_blend_weight,
                    "degree": 1,
                    "ridge_lambda": 0.1,
                    "window_size": 2.0,
                    "flex_window": 0.75,
                    "warmup_steps": 1,
                    "tail_actual_steps": 1,
                    "max_history": 8,
                    "debug": True,
                    "history_storage": "system_ram",
                    "bootstrap_first_forecast": True,
                    "anchor_residual_feedback": False,
                    "selective_rollback_correction": False,
                    "offline_smoothing_replay": True,
                    "audio_blend_weight": 0.0,
                    "offline_archive_storage": "system_ram",
                    "model_aware_mode": "off",
                    "model_aware_risk_threshold": 0.65,
                    "model_aware_trust_shrinkage": False,
                    "model_aware_replay_generic_correction": False,
                    "generic_correction_mode": "coordinate_rls",
                    "generic_correction_limiter": "hard_clip",
                    "generic_correction_limit": 0.4,
                    "generic_correction_attenuation": "no_attenuation",
                },
            },
            "conditioning": {
                "class_type": "MiniMaxH3ReferenceToVideo",
                "inputs": {
                    "clip": ["clip", 0],
                    "vae": ["vae", 0],
                    "audio_vae": ["audio_vae", 0],
                    "prompt": prompt,
                    "width": width,
                    "height": height,
                    "length": length,
                    "ref_image_size": ref_image_size,
                    "ref_images": reference_links,
                    "ref_videos": {},
                    "ref_video_audios": {},
                    "ref_audios": {},
                },
            },
            "refmod_extract": {
                "class_type": "MiniMaxH3RefModExtract",
                "inputs": {
                    "name": "lora_basepipe_character_identity",
                    "mode": "encode",
                    "concept_type": "identity",
                    "background_retention": 0.0,
                    "ref_resolution": 768,
                    "pool_h": 16,
                    "pool_w": 16,
                    "latent_frames": 1,
                    "identity": 500,
                    "multiplier": 1,
                    "max_tokens": 5120,
                    "description": "canonical character identity, face, hair, body, outfit and accessories",
                    "save": False,
                    "ref_image_1": ["ref", 0],
                    "vae": ["vae", 0],
                },
            },
            "refmod_apply": {
                "class_type": "MiniMaxH3RefModApply",
                "inputs": {
                    "conditioning": ["conditioning", 0],
                    "mods": ["refmod_extract", 0],
                    "retention": refmod_retention,
                    "curve_direction": "decrease",
                    "curve_shape": "ease",
                    "curve_value": 1.0,
                },
            },
            "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
            "guider": {"class_type": "BasicGuider", "inputs": {"model": ["spectrum", 0], "conditioning": ["refmod_apply", 0]}},
            "sampler": {"class_type": "MiniMaxH3TurboSampler", "inputs": {}},
            "scheduler": {"class_type": "BasicScheduler", "inputs": {"model": ["spectrum", 0], "scheduler": "simple", "steps": 8, "denoise": 1.0}},
            "sample": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["sampler", 0], "sigmas": ["scheduler", 0], "latent_image": ["conditioning", 1]}},
            # H3 returns a joint audio/video latent (NestedTensor).  It must be
            # split before either VAE sees it; direct VAEDecode is a verified
            # runtime failure path (``Tensor.to(NestedTensor)``).
            "split_av": {"class_type": "MiniMaxH3LatentLabSplitAV", "inputs": {"av_latent": ["sample", 0]}},
            "decode": {
                "class_type": "VAEDecodeTiled",
                "inputs": {
                    "samples": ["split_av", 0], "vae": ["vae", 0],
                    "tile_size": 1024, "overlap": 64,
                    "temporal_size": 16, "temporal_overlap": 4,
                },
            },
            "decode_audio": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["split_av", 1], "vae": ["audio_vae", 0]}},
            "video": {"class_type": "CreateVideo", "inputs": {"images": ["decode", 0], "audio": ["decode_audio", 0], "fps": 24}},
            "out": {
                "class_type": "SaveVideo",
                "inputs": {
                    "video": ["video", 0],
                    "filename_prefix": "lora_basepipe/h3",
                    "format": "auto",
                    "codec": "auto",
                },
            },
        }

    @staticmethod
    def validate_graph_against_object_info(graph: dict, object_info: dict) -> dict:
        """Validate API-node presence and required inputs without queueing work."""
        missing_nodes: list[str] = []
        missing_inputs: list[str] = []
        for node_id, node in graph.items():
            class_type = str(node.get("class_type", ""))
            schema = object_info.get(class_type)
            if not isinstance(schema, dict):
                missing_nodes.append(class_type or f"<node:{node_id}>")
                continue
            required = schema.get("input", {}).get("required", {})
            inputs = node.get("inputs", {})
            for field in required:
                if field not in inputs:
                    missing_inputs.append(f"{node_id}:{class_type}.{field}")
        return {
            "ok": not missing_nodes and not missing_inputs,
            "missing_nodes": sorted(set(missing_nodes)),
            "missing_inputs": missing_inputs,
        }

    # ── ワークフロー投入・完了待ち ────────────────────────────────────
    def submit(self, graph: dict) -> str:
        payload = {"prompt": graph, "client_id": self.client_id}
        r = requests.post(
            f"{self.base_url}/prompt", json=payload,
            headers=self._task_router_headers(), timeout=self.timeout,
        )
        if r.status_code != 200:
            raise ComfyUIError(f"/prompt 失敗 ({r.status_code}): {r.text[:400]}")
        body = r.json()
        prompt_id = body.get("prompt_id")
        if not prompt_id:
            raise ComfyUIError(f"prompt_id が返りませんでした: {body}")
        return prompt_id

    def _queue_stage_hint(self, prompt_id: str) -> str:
        """タイムアウト時の診断用: prompt_id がQueueのどの状態にあるか（実行中/待機中/
        どちらにも見当たらない=クラッシュや取りこぼしの疑い）を1回だけ問い合わせる。

        この呼び出し自体が失敗しても診断情報が得られないだけで、本来のタイムアウト
        エラーの送出は妨げない（例外を握りつぶして "unknown" を返す）。
        """
        try:
            r = requests.get(f"{self.base_url}/queue", timeout=self.timeout)
            r.raise_for_status()
            q = r.json()
            running_ids = {item[1] for item in q.get("queue_running", [])}
            pending_ids = {item[1] for item in q.get("queue_pending", [])}
            if prompt_id in running_ids:
                return "queue_running(実行中だがhistoryに未反映=モデルロード/サンプリング継続中の可能性)"
            if prompt_id in pending_ids:
                return f"queue_pending(他ジョブ待ち, pending={len(pending_ids)}件)"
            return "queueに見当たらない(ComfyUI側でクラッシュ/再起動/取りこぼしの疑い)"
        except requests.exceptions.RequestException as exc:
            return f"queue状態取得失敗(ComfyUI応答なしの可能性): {exc}"

    def wait_history(self, prompt_id: str, timeout: float = 300.0, poll: float = 1.0) -> dict:
        deadline = time.time() + timeout
        refused_count = 0
        while time.time() < deadline:
            try:
                # 個々のポーリングリクエストの接続タイムアウトには self.timeout（既定 15秒）
                # を使う。ComfyUI が学習プロセスと GPU を共有している場合、単発の応答が
                # 遅延することがあるが、それは致命的エラーではなく次のポーリングで
                # 回収されるべき。ここで即座に例外化すると全体が失敗する。
                r = requests.get(f"{self.base_url}/history/{prompt_id}", timeout=self.timeout)
            except requests.exceptions.ConnectionError as exc:
                refused_count += 1
                if refused_count >= 3:
                    raise ComfyUIUnavailableError(
                        f"ComfyUI became unreachable during prompt {prompt_id}: {exc}"
                    ) from exc
                time.sleep(poll)
                continue
            except requests.exceptions.RequestException:
                time.sleep(poll)
                continue
            refused_count = 0
            if r.status_code == 200:
                hist = r.json()
                if prompt_id in hist:
                    entry = hist[prompt_id]
                    status = entry.get("status", {})
                    if status.get("status_str") == "error":
                        raise ComfyUIError(f"ワークフロー実行エラー: {json.dumps(status)[:400]}")
                    if entry.get("outputs"):
                        return entry
            time.sleep(poll)
        stage_hint = self._queue_stage_hint(prompt_id)
        raise ComfyUIError(f"タイムアウト ({timeout}s): prompt {prompt_id} — 状態: {stage_hint}")

    def first_output_image(self, history_entry: dict) -> dict:
        """history の outputs から最初の画像 {filename, subfolder, type} を返す。"""
        for _node_id, out in history_entry.get("outputs", {}).items():
            images = out.get("images") or []
            if images:
                return images[0]
        raise ComfyUIError("出力画像が見つかりません")

    def first_output_video(self, history_entry: dict) -> dict:
        """Return the first video reference emitted by SaveVideo/PreviewVideo."""
        preferred = ("videos", "gifs", "video")
        for _node_id, out in history_entry.get("outputs", {}).items():
            for key in preferred:
                refs = out.get(key) or []
                if isinstance(refs, dict):
                    refs = [refs]
                if refs:
                    return refs[0]
            for value in out.values():
                if not isinstance(value, list):
                    continue
                for ref in value:
                    if isinstance(ref, dict) and Path(str(ref.get("filename", ""))).suffix.lower() in {".mp4", ".webm", ".mov", ".gif"}:
                        return ref
        raise ComfyUIError("出力動画が見つかりません")

    def download(self, image_ref: dict) -> bytes:
        params = {
            "filename": image_ref.get("filename", ""),
            "subfolder": image_ref.get("subfolder", ""),
            "type": image_ref.get("type", "output"),
        }
        url = f"{self.base_url}/view?" + urllib.parse.urlencode(params)
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        content = r.content
        # ComfyUIが200を返しても本文が空/壊れている場合がある(ディスク書き込み失敗・
        # 途中切断等)。検証なしに返すと、呼び出し元は「生成成功」として扱い、壊れた
        # 画像をそのままpreview_samplesへ保存してしまう(実コード監査で発見した
        # Preview Failure Matrix上の未処理ケース: Invalid image)。ここで最小限の
        # デコード検証を行い、壊れている場合はComfyUIErrorとして明示的に失敗させる。
        if not content:
            raise ComfyUIError(f"画像データが空です(filename={params['filename']})")
        try:
            from PIL import Image

            with Image.open(io.BytesIO(content)) as img:
                img.verify()
        except Exception as exc:  # noqa: BLE001 — デコード失敗は全てImage不正として扱う
            raise ComfyUIError(
                f"画像データが破損しています(filename={params['filename']}): {exc}"
            ) from exc
        return content

    def download_media(self, media_ref: dict) -> bytes:
        """Download a ComfyUI media reference without applying image decoding."""
        params = {
            "filename": media_ref.get("filename", ""),
            "subfolder": media_ref.get("subfolder", ""),
            "type": media_ref.get("type", "output"),
        }
        r = requests.get(f"{self.base_url}/view?" + urllib.parse.urlencode(params), timeout=120)
        r.raise_for_status()
        if not r.content:
            raise ComfyUIError(f"メディアデータが空です(filename={params['filename']})")
        return r.content

    # ── 文字削除ワークフロー構築 ──────────────────────────────────────
    @staticmethod
    def build_text_removal_graph(
        image_name: str,
        lama_model: str = LAMA_MODEL_NAME,
        prompt: str = DEFAULT_TEXT_PROMPT,
        threshold: float = 0.4,
        grow: int = 8,
    ) -> dict:
        """API 形式 (prompt) のワークフローグラフを構築。

        ApplyCLIPSeg+ でテキスト領域の MASK を生成（dilate=grow でマスク拡張）し、
        INPAINT_InpaintWithModel(big-lama) で穴埋めする。
        出力は PreviewImage(temp 保存) のため ComfyUI output フォルダを汚さない。
        """
        return {
            "1": {
                "class_type": "LoadImage",
                "inputs": {"image": image_name},
            },
            "2": {
                "class_type": "LoadCLIPSegModels+",
                "inputs": {},
            },
            "3": {
                "class_type": "ApplyCLIPSeg+",
                "inputs": {
                    "clip_seg": ["2", 0],
                    "image": ["1", 0],
                    "prompt": prompt,
                    "threshold": threshold,
                    "smooth": 9,
                    "dilate": max(-32, min(32, grow)),
                    "blur": 0,
                },
            },
            "4": {
                "class_type": "INPAINT_LoadInpaintModel",
                "inputs": {"model_name": lama_model},
            },
            "5": {
                "class_type": "INPAINT_InpaintWithModel",
                "inputs": {
                    "inpaint_model": ["4", 0],
                    "image": ["1", 0],
                    "mask": ["3", 0],
                    "seed": 0,
                },
            },
            "6": {
                "class_type": "PreviewImage",
                "inputs": {"images": ["5", 0]},
            },
        }

    def remove_text(
        self,
        src_path: Path,
        lama_model: str = LAMA_MODEL_NAME,
        prompt: str = DEFAULT_TEXT_PROMPT,
        threshold: float = 0.4,
        grow: int = 8,
        timeout: float = 300.0,
    ) -> bytes:
        """1 枚の画像に対し文字削除を実行し、処理後画像のバイト列を返す。"""
        server_name = self.upload_image(src_path)
        graph = self.build_text_removal_graph(
            image_name=server_name,
            lama_model=lama_model,
            prompt=prompt,
            threshold=threshold,
            grow=grow,
        )
        prompt_id = self.submit(graph)
        entry = self.wait_history(prompt_id, timeout=timeout)
        image_ref = self.first_output_image(entry)
        return self.download(image_ref)

    # ── アップスケール（解像度標準化の拡大） ───────────────────────────
    @staticmethod
    def build_upscale_graph(
        image_name: str,
        model_name: str,
    ) -> dict:
        """UpscaleModelLoader → ImageUpscaleWithModel(ESRGAN) → PreviewImage。

        出力は PreviewImage(temp 保存) のため ComfyUI output フォルダを汚さない。
        """
        return {
            "1": {"class_type": "LoadImage", "inputs": {"image": image_name}},
            "2": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": model_name}},
            "3": {
                "class_type": "ImageUpscaleWithModel",
                "inputs": {"upscale_model": ["2", 0], "image": ["1", 0]},
            },
            "4": {
                "class_type": "PreviewImage",
                "inputs": {"images": ["3", 0]},
            },
        }

    @staticmethod
    def build_flashvsr_video_graph(
        input_video_path: str,
        local_range: int,
        filename_prefix: str,
    ) -> dict:
        """Build the Character Dataset FlashVSR v1.1 comparison graph.

        The input contract is one contiguous nine-frame yaw neighbourhood.  A
        frame cap is kept in the graph as a safety invariant so a caller cannot
        accidentally turn a selective dataset enhancement into a full-clip GPU
        job.  ``local_range`` deliberately exposes only the two Stable-node
        comparison settings: 9 favours detail, while 11 favours consistency.

        The output uses ``VHS_VideoCombine`` so its history entry is compatible
        with :meth:`first_output_video`.
        """
        if local_range not in {9, 11}:
            raise ValueError("FlashVSR local_range must be 9 or 11")
        if not str(input_video_path).strip():
            raise ValueError("FlashVSR input_video_path must not be empty")
        if not str(filename_prefix).strip():
            raise ValueError("FlashVSR filename_prefix must not be empty")

        return {
            "load_video": {
                "class_type": "VHS_LoadVideoPath",
                "inputs": {
                    "video": str(input_video_path),
                    "force_rate": 0,
                    "custom_width": 0,
                    "custom_height": 0,
                    "frame_load_cap": 9,
                    "skip_first_frames": 0,
                    "select_every_nth": 1,
                    "format": "None",
                },
            },
            "flashvsr_pipe": {
                "class_type": "FlashVSRInitPipe",
                "inputs": {
                    "model": "FlashVSR-v1.1",
                    "mode": "full",
                    "vae_model": "Wan2.1",
                    "force_offload": True,
                    "precision": "bf16",
                    "device": "auto",
                    "attention_mode": "sparse_sage_attention",
                },
            },
            "flashvsr": {
                "class_type": "FlashVSRNodeAdv",
                "inputs": {
                    "pipe": ["flashvsr_pipe", 0],
                    "frames": ["load_video", 0],
                    "scale": 4,
                    "color_fix": True,
                    "tiled_vae": False,
                    "tiled_dit": False,
                    "tile_size": 256,
                    "tile_overlap": 24,
                    "unload_dit": False,
                    "sparse_ratio": 2.0,
                    "kv_ratio": 3.0,
                    "local_range": local_range,
                    "seed": 73910521,
                    "frame_chunk_size": 0,
                    "enable_debug": True,
                    "keep_models_on_cpu": True,
                    "resize_factor": 1.0,
                },
            },
            "out": {
                "class_type": "VHS_VideoCombine",
                "inputs": {
                    "images": ["flashvsr", 0],
                    "frame_rate": 24.0,
                    "loop_count": 0,
                    "filename_prefix": filename_prefix,
                    "format": "video/nvenc_hevc-mp4",
                    "pix_fmt": "yuv420p",
                    "bitrate": 20,
                    "megabit": True,
                    "save_metadata": False,
                    "pingpong": False,
                    "save_output": True,
                },
            },
        }

    @staticmethod
    def build_realesrgan_video_graph(
        input_video_path: str,
        filename_prefix: str,
    ) -> dict:
        """Build the nine-frame RealESRGAN x2plus video baseline graph.

        This baseline follows the same VHS input/output contract as FlashVSR,
        which keeps comparison lineage and :meth:`first_output_video`
        extraction identical between enhancement engines.
        """
        if not str(input_video_path).strip():
            raise ValueError("RealESRGAN input_video_path must not be empty")
        if not str(filename_prefix).strip():
            raise ValueError("RealESRGAN filename_prefix must not be empty")

        return {
            "load_video": {
                "class_type": "VHS_LoadVideoPath",
                "inputs": {
                    "video": str(input_video_path),
                    "force_rate": 0,
                    "custom_width": 0,
                    "custom_height": 0,
                    "frame_load_cap": 9,
                    "skip_first_frames": 0,
                    "select_every_nth": 1,
                    "format": "None",
                },
            },
            "upscale_model": {
                "class_type": "UpscaleModelLoader",
                "inputs": {"model_name": "RealESRGAN_x2plus.pth"},
            },
            "upscale": {
                "class_type": "ImageUpscaleWithModel",
                "inputs": {
                    "upscale_model": ["upscale_model", 0],
                    "image": ["load_video", 0],
                },
            },
            "out": {
                "class_type": "VHS_VideoCombine",
                "inputs": {
                    "images": ["upscale", 0],
                    "frame_rate": 24.0,
                    "loop_count": 0,
                    "filename_prefix": filename_prefix,
                    "format": "video/nvenc_hevc-mp4",
                    "pix_fmt": "yuv420p",
                    "bitrate": 20,
                    "megabit": True,
                    "save_metadata": False,
                    "pingpong": False,
                    "save_output": True,
                },
            },
        }

    def upscale_image(self, src_path: Path, model_name: str, timeout: float = 300.0) -> bytes:
        """1 枚の画像を ESRGAN 等でアップスケールし、処理後画像のバイト列を返す。"""
        server_name = self.upload_image(src_path)
        graph = self.build_upscale_graph(server_name, model_name)
        prompt_id = self.submit(graph)
        entry = self.wait_history(prompt_id, timeout=timeout)
        image_ref = self.first_output_image(entry)
        return self.download(image_ref)

    # ── Qwen-Image-Edit-2511 native ComfyUI workflow ─────────────────

    @staticmethod
    def build_qwen_cleanup_graph(
        image_name: str,
        reference_image_names: list[str] | None = None,
        prompt: str = QWEN_CLEANUP_PROMPT,
        seed: int = 0,
        model_name: str = QWEN_EDIT_2511_MODELS[0],
        clip_name: str = QWEN_EDIT_2511_CLIP,
        vae_name: str = QWEN_EDIT_2511_VAE,
        use_lightning: bool = False,
        lightning_name: str = QWEN_EDIT_2511_LIGHTNING,
        angle_lora_name: str | None = None,
        angle_lora_strength: float = 0.9,
        denoise: float = 1.0,
    ) -> dict:
        """Build the native Qwen-Image-Edit-2511 graph with identity anchors."""
        graph = {
            "ref": {"class_type": "LoadImage", "inputs": {"image": image_name}},
            "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": model_name, "weight_dtype": "default"}},
            "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip_name, "type": "qwen_image", "device": "default"}},
            "vae": {"class_type": "VAELoader", "inputs": {"vae_name": vae_name}},
            "sampling": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["unet", 0], "shift": 3.1}},
            "cfg_norm": {"class_type": "CFGNorm", "inputs": {"model": ["sampling", 0], "strength": 1.0, "pre_cfg": False}},
            "scale": {"class_type": "FluxKontextImageScale", "inputs": {"image": ["ref", 0]}},
            "positive": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"clip": ["clip", 0], "vae": ["vae", 0], "image1": ["scale", 0], "prompt": prompt}},
            "negative": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"clip": ["clip", 0], "vae": ["vae", 0], "image1": ["scale", 0], "prompt": " "}},
            "positive_method": {"class_type": "FluxKontextMultiReferenceLatentMethod", "inputs": {"conditioning": ["positive", 0], "reference_latents_method": "index_timestep_zero"}},
            "negative_method": {"class_type": "FluxKontextMultiReferenceLatentMethod", "inputs": {"conditioning": ["negative", 0], "reference_latents_method": "index_timestep_zero"}},
            "latent": {"class_type": "VAEEncode", "inputs": {"pixels": ["scale", 0], "vae": ["vae", 0]}},
            "sample": {"class_type": "KSampler", "inputs": {"model": ["cfg_norm", 0], "positive": ["positive_method", 0], "negative": ["negative_method", 0], "latent_image": ["latent", 0], "seed": seed, "steps": 40, "cfg": 4.0, "sampler_name": "euler", "scheduler": "simple", "denoise": max(0.0, min(1.0, float(denoise)))}},
            "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}},
            "out": {"class_type": "SaveImage", "inputs": {"images": ["decode", 0], "filename_prefix": "lora_basepipe/qwen_2511"}},
        }
        model_output = ["cfg_norm", 0]
        if angle_lora_name:
            graph["angle_lora"] = {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {"model": model_output, "lora_name": angle_lora_name, "strength_model": angle_lora_strength},
            }
            model_output = ["angle_lora", 0]
        if use_lightning:
            graph["lightning"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": model_output, "lora_name": lightning_name, "strength_model": 1.0}}
            graph["sample"]["inputs"].update({"model": ["lightning", 0], "steps": 4, "cfg": 1.0})
        else:
            graph["sample"]["inputs"]["model"] = model_output
        for index, reference_name in enumerate((reference_image_names or [])[:2], start=2):
            ref_key = f"master_ref_{index - 1}"
            scale_key = f"master_scale_{index - 1}"
            graph[ref_key] = {"class_type": "LoadImage", "inputs": {"image": reference_name}}
            graph[scale_key] = {"class_type": "FluxKontextImageScale", "inputs": {"image": [ref_key, 0]}}
            graph["positive"]["inputs"][f"image{index}"] = [scale_key, 0]
            graph["negative"]["inputs"][f"image{index}"] = [scale_key, 0]
        return graph

    def qwen_cleanup(
        self,
        src_path: Path,
        reference_paths: list[Path] | None = None,
        width: int | None = None,
        height: int | None = None,
        prompt: str = QWEN_CLEANUP_PROMPT,
        seed: int = 0,
        timeout: float = 600.0,
        use_lightning: bool | None = None,
        angle_lora_name: str | None = None,
        angle_lora_strength: float = 0.9,
        denoise: float = 1.0,
    ) -> bytes:
        """Apply canonical Qwen-Image-Edit-2511 without mutating the source.

        ``width`` and ``height`` remain accepted for Dataset Pipeline API
        compatibility.  The official FluxKontextImageScale node selects the
        nearest supported aspect-preserving resolution.
        """
        status = self.check_qwen_status()
        if not status.get("ready"):
            raise ComfyUIError(status.get("message") or "Qwen-Image-Edit-2511 is not ready")
        server_name = self.upload_image(src_path)
        reference_names = [self.upload_image(path) for path in (reference_paths or [])[:2]]
        graph = self.build_qwen_cleanup_graph(
            image_name=server_name,
            reference_image_names=reference_names,
            prompt=prompt,
            seed=seed,
            model_name=str(status["diffusion_model"]),
            clip_name=str(status["text_encoder"]),
            vae_name=str(status["vae_model"]),
            use_lightning=bool(status.get("lightning_ok")) if use_lightning is None else use_lightning,
            angle_lora_name=angle_lora_name,
            angle_lora_strength=angle_lora_strength,
            denoise=denoise,
        )
        prompt_id = self.submit(graph)
        entry = self.wait_history(prompt_id, timeout=timeout)
        image_ref = self.first_output_image(entry)
        return self.download(image_ref)

    # ── 学習中プレビュー生成（musubi系 + Anima） ──
    #
    # musubi-tuner 自体はサンプル画像生成機能を持たないため（kohya_ss と異なる）、
    # 学習中の LoRA チェックポイントを ComfyUI にロードしてプレビューを生成する。
    # 使用ノードはすべて ComfyUI コア標準ノード（カスタムノード不要）。
    # モデルファイル名は ComfyUI 自身の object_info コンボから checkpoint_patterns で
    # 部分一致検索する（ベストエフォート。一致しなければ生成をスキップする）。

    # Flux は CLIP-L + T5-XXL の DualCLIPLoader が標準（単一 CLIPLoader ではない）。
    # clip_type は実機 ComfyUI (object_info/CLIPLoader の type コンボ) で確認済みの値。
    # KREA2 は VAE/テキストエンコーダファイルこそ Qwen-Image を流用するが、
    # CLIPLoader の type は専用の "krea2" が必要（"qwen_image" ではエラーになる。
    # 実機検証: "Krea2 expects conditioning with 12x2560=30720 features
    # (a 12-layer Qwen3-VL stack) but got 2560. Load the text encoder with
    # CLIPLoader type 'krea2'." — 2026-07-02 実機 ComfyUI で確認）。
    # zimage の type は未検証（実機にモデル未導入のため確認不能。要再検証）。
    PREVIEW_UNET_LOADER_FAMILIES = {
        "flux": {"clip_mode": "dual", "clip_type": "flux"},
        "qwen_image": {"clip_mode": "single", "clip_type": "qwen_image"},
        "qwen_image_edit": {"clip_mode": "single", "clip_type": "qwen_image"},
        "krea2": {"clip_mode": "single", "clip_type": "krea2"},
        "zimage": {"clip_mode": "single", "clip_type": "qwen_image"},  # 未検証
    }

    # VAE / CLIP はモデル世代ごとに専用ファイル名の慣習が異なり checkpoint_patterns
    # （UNET/checkpoint ファイル名向け）とは語彙が一致しないため、家系ごとに別途
    # キーワードを持つ。存在しない/曖昧な場合は None を返し、呼び出し側はスキップする
    # （誤ったファイルを推測で選ぶより、プレビュー生成をスキップする方が安全）。
    PREVIEW_VAE_KEYWORDS = {
        "flux": ["ae.safetensors", "flux_vae", "flux-ae"],
        "qwen_image": ["qwen"],
        "qwen_image_edit": ["qwen"],
        "krea2": ["qwen"],  # KREA2 は Qwen-Image VAE を流用（specs/krea2.py 参照）
        "zimage": ["qwen", "zimage", "z_image", "z-image"],
    }
    PREVIEW_CLIP_KEYWORDS = {
        "qwen_image": ["qwen"],
        "qwen_image_edit": ["qwen"],
        "krea2": ["qwen"],  # KREA2 は Qwen3-VL テキストエンコーダを要求（specs/krea2.py 参照）
        "zimage": ["qwen", "zimage", "z_image", "z-image"],
    }
    # UNET 候補から確実に除外すべきキーワード（VAE/CLIP/エンコーダファイルの誤認識防止）
    _UNET_EXCLUDE_KEYWORDS = ("vae", "clip", "text_encoder", "encoder")

    def resolve_preview_base_model(self, model_family: str, checkpoint_patterns: list[str]) -> dict | None:
        """ComfyUI 側に存在する unet/clip/vae 名を照合して返す。

        該当ファイルが見つからない場合は None（呼び出し側は生成をスキップすること）。
        Flux は clip_name1(T5-XXL) / clip_name2(CLIP-L) の2本、それ以外は clip_name 1本。
        VAE/CLIP は checkpoint_patterns ではなく家系別キーワード（PREVIEW_*_KEYWORDS）で
        照合する。誤ったファイルを推測で選ぶくらいなら生成をスキップする方針。

        重要: ComfyUI自体への接続失敗(タイムアウト・接続拒否等)は「該当ファイルが
        見つからない」とは全く異なる事象であり、ComfyUIError を送出する
        (呼び出し元のgenerate_preview_safe()が捕捉してerror detailへ反映する)。
        以前はここで例外を握り潰してNoneを返していたため、ComfyUI停止時に
        Preview失敗の理由が「ComfyUI未接続」ではなく空文字列としてしか
        記録されない実バグがあった(実Runtime検証で発見)。
        """
        try:
            r = requests.get(f"{self.base_url}/object_info", timeout=30)
            r.raise_for_status()
            info = r.json()
        except requests.exceptions.RequestException as exc:
            reason = f"ComfyUIへの接続に失敗しました({self.base_url}): {exc}"
            logger.warning("resolve_preview_base_model: %s", reason)
            raise ComfyUIError(reason) from exc
        except Exception as exc:  # noqa: BLE001 — JSON decode等の非ネットワーク系異常
            reason = f"ComfyUIのobject_info応答が不正です: {exc}"
            logger.warning("resolve_preview_base_model: %s", reason)
            raise ComfyUIError(reason) from exc

        patterns = [p.lower() for p in checkpoint_patterns]

        def _match(
            names: list[str], keywords: list[str], *, exclude: tuple[str, ...] = (), prefer_fp8: bool = False,
        ) -> str | None:
            """keywords に一致する最初の候補を返す。

            prefer_fp8=True の場合のみ、複数候補があれば fp8 量子化済みファイルを優先する。
            これは「同一モデルの精度違いリリース」が一般的なテキストエンコーダ等でのみ安全
            （例: qwen3vl_4b_bf16 と qwen3vl_4b_fp8_scaled は同一モデルの精度違い）。
            UNET（diffusion model）は "fp8" を含むファイル名が必ずしも同一ベースモデルの
            量子化版とは限らない（例: turbo/distill 版は別ファインチューンであり、
            学習時に使ったベースモデルと重みが異なるため LoRA との組み合わせが不正になる）。
            そのため UNET 選択では適用しない（呼び出し側で prefer_fp8=False を渡す）。
            """
            candidates = []
            for n in names:
                low = n.lower()
                if exclude and any(x in low for x in exclude):
                    continue
                if any(k in low for k in keywords):
                    candidates.append(n)
            if not candidates:
                return None
            if prefer_fp8:
                fp8_candidates = [n for n in candidates if "fp8" in n.lower()]
                if fp8_candidates:
                    return fp8_candidates[0]
            return candidates[0]

        # Anima は ComfyUI の標準 UNETLoader/CLIPLoader 直列ではなく、
        # 実機に導入済みの UnifiedModelStackLoader が diffusion model +
        # Qwen CLIP + VAE + LoRA stack を一つの正規経路として扱う。
        # ここを標準ローダーとして誤認すると、生成時に Anima のモデル構造と
        # テキストエンコーダの組み合わせが崩れるため、実機 object_info で
        # ノードの存在と候補を確認して専用グラフへ渡す。
        if model_family == "anima":
            stack_ckpts = self._extract_combo(info, "UnifiedModelStackLoader", "ckpt_name")
            stack_clips = self._extract_combo(info, "UnifiedModelStackLoader", "clip_name")
            stack_vaes = self._extract_combo(info, "UnifiedModelStackLoader", "vae_name")
            if not stack_clips or not stack_vaes:
                return None
            # UnifiedModelStackLoader exposes ckpt_name as a STRING widget, so
            # object_info cannot enumerate its diffusion_models candidates.
            # The Anima Base v1.0 file is nevertheless the same concrete base
            # model used by the training backend and verified on this ComfyUI
            # instance. Keep the name explicit rather than selecting the
            # unrelated WAI-ANIMA checkpoint shown as the widget default.
            ckpt = (
                "anima-base-v1.0.safetensors"
                if any("anima" in pattern for pattern in patterns)
                else _match(stack_ckpts, patterns, exclude=self._UNET_EXCLUDE_KEYWORDS)
            )
            clip = next((n for n in stack_clips if "qwen_3_06b_base" in n.lower()), None)
            vae = next((n for n in stack_vaes if "qwen_image_vae" in n.lower()), None)
            if not ckpt or not clip or not vae:
                return None
            return {
                "loader_class_type": "UnifiedModelStackLoader",
                "ckpt_name": ckpt,
                "clip_name": clip,
                "clip_type": "qwen_image",
                "vae_name": vae,
            }

        unet_list = self._extract_combo(info, "UNETLoader", "unet_name")
        unet = _match(unet_list, patterns, exclude=self._UNET_EXCLUDE_KEYWORDS)
        if not unet:
            return None

        vae_list = self._extract_combo(info, "VAELoader", "vae_name")
        vae_keywords = self.PREVIEW_VAE_KEYWORDS.get(model_family, patterns)
        vae = _match(vae_list, vae_keywords) or _match(vae_list, ["vae"])
        if not vae:
            return None

        clip_mode = self.PREVIEW_UNET_LOADER_FAMILIES.get(model_family, {}).get("clip_mode", "single")
        if clip_mode == "dual":
            dual_list = self._extract_combo(info, "DualCLIPLoader", "clip_name1")
            t5 = next((n for n in dual_list if "t5" in n.lower()), None)
            clip_l = next((n for n in dual_list if "clip_l" in n.lower() or "clip-l" in n.lower()), None)
            if not t5 or not clip_l or t5 == clip_l:
                return None
            return {"unet_name": unet, "vae_name": vae, "clip_name1": t5, "clip_name2": clip_l}

        clip_list = self._extract_combo(info, "CLIPLoader", "clip_name")
        clip_keywords = self.PREVIEW_CLIP_KEYWORDS.get(model_family)
        clip = _match(clip_list, clip_keywords, prefer_fp8=True) if clip_keywords else None
        if not clip:
            return None
        return {"unet_name": unet, "vae_name": vae, "clip_name": clip}

    def stage_lora_for_comfyui(self, comfyui_root: str, run_id: int, lora_path: Path) -> str | None:
        """学習中の LoRA を ComfyUI の models/loras 配下にコピーし、相対名を返す。

        ComfyUI の LoraLoaderModelOnly は自身の loras フォルダ内のファイルしか
        参照できないため、学習出力（.runtime/runs/{run_id}/output/*.safetensors）を
        都度コピーする。ファイルサイズ・更新時刻が同じならコピーをスキップする。

        ComfyUI はサブフォルダ内 LoRA を os.sep 区切り（Windows では "\\"）で
        object_info に返す。ここで "/" 固定にすると Windows 上で value_not_in_list
        エラーになり、プレビュー生成が常に失敗する（LoRA ファイル自体は存在するのに）。
        """
        if not comfyui_root:
            return None
        dest_dir = Path(comfyui_root) / "models" / "loras" / "lora_basepipe" / str(run_id)
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / lora_path.name
            if not dest.exists() or dest.stat().st_mtime < lora_path.stat().st_mtime:
                shutil.copy2(lora_path, dest)
            return os.sep.join(["lora_basepipe", str(run_id), lora_path.name])
        except OSError as exc:
            logger.warning("stage_lora_for_comfyui failed: %s", exc)
            return None

    @staticmethod
    def build_musubi_preview_graph(
        *,
        model_family: str,
        base_models: dict,
        lora_name: str,
        positive: str,
        negative: str,
        width: int,
        height: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        seed: int,
        unet_weight_dtype: str = "default",
    ) -> dict:
        """UNETLoader + CLIP(Dual) + VAELoader + LoraLoaderModelOnly の共通グラフ。

        Flux / Qwen-Image / KREA2 / Z-Image は musubi-tuner で学習される DiT 系モデルで、
        いずれも ComfyUI 上では UNETLoader(diffusion model) + 個別 CLIP + VAE の
        分割ロードが標準（CheckpointLoaderSimple 形式ではない）。
        Flux のみ CLIP-L + T5-XXL の DualCLIPLoader が必要（それ以外は単一 CLIPLoader）。
        """
        if model_family == "anima":
            # Anima の実機ワークフローは UnifiedModelStackLoader が LoRA を
            # loader 内で適用し、UnifiedTagEditorNode が Qwen CLIP 用の
            # conditioning を生成する。標準 CLIPTextEncodeへ置換しない。
            lora_stack = json.dumps([{
                "enabled": True,
                "bypassed": False,
                "lora_name": lora_name,
                "strength_model": 1.0,
                "strength_clip": 1.0,
            }], ensure_ascii=False)
            editor_state = json.dumps({
                "prefix": [],
                "mainTags": [{"raw": positive}],
                "suffix": [],
                "blocklist": [],
                "negative": [{"raw": negative}],
                "clipSkip": 1,
            }, ensure_ascii=False)
            anima_sampler = sampler_name if sampler_name not in {"", "auto", "euler_a"} else "euler"
            return {
                "loader": {
                    "class_type": "UnifiedModelStackLoader",
                    "inputs": {
                        "ckpt_name": base_models["ckpt_name"],
                        "clip_name": base_models["clip_name"],
                        "clip_type": base_models["clip_type"],
                        "vae_name": base_models["vae_name"],
                        "lora_stack": lora_stack,
                    },
                },
                "tags": {
                    "class_type": "UnifiedTagEditorNode",
                    "inputs": {
                        "clip": ["loader", 1],
                        "editor_state": editor_state,
                    },
                },
                "latent": {
                    "class_type": "EmptyLatentImage",
                    "inputs": {"width": width, "height": height, "batch_size": 1},
                },
                "sampler": {
                    "class_type": "KSampler Adv. (Efficient)",
                    "inputs": {
                        "model": ["loader", 0],
                        "positive": ["tags", 0],
                        "negative": ["tags", 1],
                        "latent_image": ["latent", 0],
                        "optional_vae": ["loader", 2],
                        "add_noise": "enable",
                        "noise_seed": seed,
                        "steps": steps,
                        "cfg": cfg,
                        "sampler_name": anima_sampler,
                        "scheduler": "simple",
                        "start_at_step": 0,
                        "end_at_step": steps,
                        "return_with_leftover_noise": "disable",
                        "preview_method": "none",
                        "vae_decode": "true",
                    },
                },
                "out": {
                    "class_type": "SaveImage",
                    "inputs": {
                        "images": ["sampler", 5],
                        "filename_prefix": "lora_basepipe/preview",
                    },
                },
            }

        unet_name = base_models["unet_name"]
        vae_name = base_models["vae_name"]
        clip_mode = ComfyUIClient.PREVIEW_UNET_LOADER_FAMILIES.get(model_family, {}).get(
            "clip_mode", "single"
        )
        clip_type = ComfyUIClient.PREVIEW_UNET_LOADER_FAMILIES.get(model_family, {}).get(
            "clip_type", "qwen_image"
        )
        if clip_mode == "dual":
            clip_node = {
                "class_type": "DualCLIPLoader",
                "inputs": {
                    "clip_name1": base_models["clip_name1"],
                    "clip_name2": base_models["clip_name2"],
                    "type": clip_type,
                },
            }
        else:
            clip_node = {
                "class_type": "CLIPLoader",
                "inputs": {"clip_name": base_models["clip_name"], "type": clip_type},
            }
        return {
            "unet": {
                "class_type": "UNETLoader",
                "inputs": {"unet_name": unet_name, "weight_dtype": unet_weight_dtype},
            },
            "clip": clip_node,
            "vae": {
                "class_type": "VAELoader",
                "inputs": {"vae_name": vae_name},
            },
            "lora": {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {"model": ["unet", 0], "lora_name": lora_name, "strength_model": 1.0},
            },
            "pos": {
                "class_type": "CLIPTextEncode",
                "inputs": {"clip": ["clip", 0], "text": positive},
            },
            "neg": {
                "class_type": "CLIPTextEncode",
                "inputs": {"clip": ["clip", 0], "text": negative},
            },
            "latent": {
                "class_type": "EmptyLatentImage",
                "inputs": {"width": width, "height": height, "batch_size": 1},
            },
            "sampler": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["lora", 0],
                    "positive": ["pos", 0],
                    "negative": ["neg", 0],
                    "latent_image": ["latent", 0],
                    "seed": seed,
                    "steps": steps,
                    "cfg": cfg,
                    "sampler_name": sampler_name,
                    "scheduler": "normal",
                    "denoise": 1.0,
                },
            },
            "decode": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["sampler", 0], "vae": ["vae", 0]},
            },
            "out": {
                "class_type": "SaveImage",
                "inputs": {"images": ["decode", 0], "filename_prefix": "lora_basepipe/preview"},
            },
        }

    def generate_musubi_preview(
        self,
        *,
        model_family: str,
        base_models: dict,
        lora_name: str,
        positive: str,
        negative: str,
        width: int,
        height: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        seed: int,
        unet_weight_dtype: str = "default",
        # 既定 300s は大型DiT系(RAW MMDiT等)では実測上不足する。実測(Krea2, RTX 3090 Ti,
        # 512px/10steps/fp8): 444.4秒で完了。300s既定のままだと処理が継続しているにも
        # 関わらず監視側がタイムアウト扱いにしてしまうため、余裕を見て600sを既定にする。
        timeout: float = 600.0,
    ) -> bytes:
        graph = self.build_musubi_preview_graph(
            model_family=model_family,
            base_models=base_models,
            lora_name=lora_name,
            positive=positive,
            negative=negative,
            width=width,
            height=height,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            seed=seed,
            unet_weight_dtype=unet_weight_dtype,
        )
        prompt_id = self.submit(graph)
        entry = self.wait_history(prompt_id, timeout=timeout)
        image_ref = self.first_output_image(entry)
        return self.download(image_ref)

    def check_qwen_status(self) -> dict:
        """Check the canonical native Qwen-Image-Edit-2511 graph and weights."""
        result: dict = {
            "reachable": False,
            "qwen_nodes_ok": False,
            "missing_nodes": [],
            "diffusion_model": None,
            "text_encoder": None,
            "vae_model": None,
            "lightning_ok": False,
            "multiple_angles_ok": False,
            "ready": False,
            "message": "",
        }
        try:
            r = requests.get(f"{self.base_url}/system_stats", timeout=self.timeout)
            r.raise_for_status()
            result["reachable"] = True
        except Exception as exc:  # noqa: BLE001
            result["message"] = f"ComfyUI に接続できません: {exc}"
            return result

        try:
            r = requests.get(f"{self.base_url}/object_info", timeout=30)
            r.raise_for_status()
            info = r.json()
        except Exception as exc:  # noqa: BLE001
            result["message"] = f"object_info 取得失敗: {exc}"
            return result

        missing = [n for n in QWEN_REQUIRED_NODES if n not in info]
        result["missing_nodes"] = missing
        result["qwen_nodes_ok"] = not missing

        unets = self._extract_combo(info, "UNETLoader", "unet_name")
        clips = self._extract_combo(info, "CLIPLoader", "clip_name")
        vaes = self._extract_combo(info, "VAELoader", "vae_name")
        loras = self._extract_combo(info, "LoraLoaderModelOnly", "lora_name")
        result["diffusion_model"] = next((name for name in QWEN_EDIT_2511_MODELS if name in unets), None)
        result["text_encoder"] = QWEN_EDIT_2511_CLIP if QWEN_EDIT_2511_CLIP in clips else None
        result["vae_model"] = QWEN_EDIT_2511_VAE if QWEN_EDIT_2511_VAE in vaes else None
        result["lightning_ok"] = QWEN_EDIT_2511_LIGHTNING in loras
        result["multiple_angles_ok"] = QWEN_EDIT_2511_MULTIPLE_ANGLES in loras
        graph_contract = {"ok": False, "missing_nodes": [], "missing_inputs": []}
        if result["diffusion_model"] and result["text_encoder"] and result["vae_model"]:
            probe_graph = self.build_qwen_cleanup_graph(
                "<source>",
                ["<front-anchor>", "<back-anchor>"],
                prompt="contract probe",
                model_name=str(result["diffusion_model"]),
                clip_name=str(result["text_encoder"]),
                vae_name=str(result["vae_model"]),
                use_lightning=bool(result["lightning_ok"]),
            )
            graph_contract = self.validate_graph_against_object_info(probe_graph, info)
        result["graph_contract"] = graph_contract
        result["ready"] = bool(result["reachable"] and result["qwen_nodes_ok"] and result["diffusion_model"] and result["text_encoder"] and result["vae_model"] and graph_contract["ok"])
        if result["ready"]:
            mode = "4-step Lightning" if result["lightning_ok"] else "40-step base"
            result["message"] = f"Qwen-Image-Edit-2511 準備完了 ({mode})"
        elif missing:
            result["message"] = "必要なノードが不足: " + ", ".join(missing)
        elif not graph_contract["ok"] and (graph_contract["missing_nodes"] or graph_contract["missing_inputs"]):
            result["message"] = f"Qwen API graph contract mismatch: {graph_contract}"
        else:
            absent = []
            if not result["diffusion_model"]:
                absent.append(QWEN_EDIT_2511_MODELS[0])
            if not result["text_encoder"]:
                absent.append(QWEN_EDIT_2511_CLIP)
            if not result["vae_model"]:
                absent.append(QWEN_EDIT_2511_VAE)
            result["message"] = "Qwen-Image-Edit-2511 model missing: " + ", ".join(absent)
        return result
