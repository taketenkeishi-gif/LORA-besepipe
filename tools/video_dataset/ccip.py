"""CCIP (deepghs/ccip_onnx) identity features for anime characters.

feat(images) -> (N,768); dist(F) -> (N,N) CCIP difference matrix (lower = more likely the same character).
Feed single-character crops (mask others out first). Default same-character threshold from the model card: 0.178475.
Model files live in the shared reference library (read-only).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image

MODEL_DIR = Path(os.environ.get("CCIP_MODEL_DIR", r"C:\Users\Keishi\Krita\reference_sources\models\ccip_onnx\ccip-caformer-24-randaug-pruned"))
THRESHOLD = 0.178475
_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
_STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)


class Ccip:
    def __init__(self, device_id: int | None = None):
        import onnxruntime as ort

        try:  # onnxruntime-gpu needs torch's CUDA DLLs on the search path
            import torch

            os.add_dll_directory(str(Path(torch.__file__).parent / "lib"))
        except Exception:
            pass
        prov = [("CUDAExecutionProvider", {"device_id": device_id}), "CPUExecutionProvider"] if device_id is not None else ["CPUExecutionProvider"]
        self._feat = ort.InferenceSession(str(MODEL_DIR / "model_feat.onnx"), providers=prov)
        self._metric = ort.InferenceSession(str(MODEL_DIR / "model_metrics.onnx"), providers=["CPUExecutionProvider"])

    @staticmethod
    def _prep(im: Image.Image) -> np.ndarray:
        im = im.convert("RGB").resize((384, 384), Image.BILINEAR)
        return ((np.asarray(im, np.float32) / 255.0 - _MEAN) / _STD).transpose(2, 0, 1)

    def feat(self, images: list[Image.Image | Path | str], batch: int = 16) -> np.ndarray:
        out = []
        for i in range(0, len(images), batch):
            chunk = [im if isinstance(im, Image.Image) else Image.open(im) for im in images[i:i + batch]]
            out.append(self._feat.run(["output"], {"input": np.stack([self._prep(c) for c in chunk])})[0])
        return np.concatenate(out) if out else np.zeros((0, 768), np.float32)

    def dist(self, feats: np.ndarray) -> np.ndarray:
        return self._metric.run(["output"], {"input": feats.astype(np.float32)})[0]
