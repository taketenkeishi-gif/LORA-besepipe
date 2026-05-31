from __future__ import annotations

import csv
import io
import threading
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False

try:
    from huggingface_hub import hf_hub_download
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False

MODEL_REPO = "SmilingWolf/wd-vit-large-tagger-v3"
MODEL_FILENAME = "model.onnx"
TAGS_FILENAME = "selected_tags.csv"
CACHE_DIR = Path(__file__).resolve().parents[4] / ".runtime" / "models" / "wd14"

IMAGE_SIZE = 448
GENERAL_THRESHOLD = 0.35
CHARACTER_THRESHOLD = 0.85

# モジュールレベルキャッシュ
_session: Optional[object] = None
_tags_data: Optional[list[dict]] = None
_load_lock = threading.Lock()


class TaggerUnavailableError(RuntimeError):
    pass


def _ensure_model() -> tuple[object, list[dict]]:
    global _session, _tags_data
    if _session is not None and _tags_data is not None:
        return _session, _tags_data

    with _load_lock:
        if _session is not None and _tags_data is not None:
            return _session, _tags_data

        if not _ORT_AVAILABLE:
            raise TaggerUnavailableError("onnxruntime not installed. Run: pip install onnxruntime-gpu")
        if not _HF_AVAILABLE:
            raise TaggerUnavailableError("huggingface-hub not installed. Run: pip install huggingface-hub")

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        model_path = CACHE_DIR / MODEL_FILENAME
        tags_path = CACHE_DIR / TAGS_FILENAME

        if not model_path.exists():
            model_path_str = hf_hub_download(
                repo_id=MODEL_REPO,
                filename=MODEL_FILENAME,
                local_dir=str(CACHE_DIR),
            )
            model_path = Path(model_path_str)

        if not tags_path.exists():
            tags_path_str = hf_hub_download(
                repo_id=MODEL_REPO,
                filename=TAGS_FILENAME,
                local_dir=str(CACHE_DIR),
            )
            tags_path = Path(tags_path_str)

        providers = (
            [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
            if _ORT_AVAILABLE and "CUDAExecutionProvider" in ort.get_available_providers()
            else ["CPUExecutionProvider"]
        )
        sess = ort.InferenceSession(str(model_path), providers=providers)

        tags: list[dict] = []
        with open(tags_path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tags.append(row)

        _session = sess
        _tags_data = tags
        return _session, _tags_data


def _preprocess(image_path: str) -> np.ndarray:
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    # WD14はBGR順 + batch次元
    arr = arr[:, :, ::-1]
    return arr[np.newaxis, ...]


def predict(
    image_path: str,
    general_thresh: float = GENERAL_THRESHOLD,
    character_thresh: float = CHARACTER_THRESHOLD,
    remove_character_tags: bool = False,
) -> str:
    """画像のパスを受け取り、コンマ区切りのタグ文字列を返す。

    Args:
        image_path: 画像ファイルパス
        general_thresh: 一般タグの信頼度しきい値 (0.05–0.95, デフォルト 0.35)
        character_thresh: キャラクタータグの信頼度しきい値 (0.05–0.99, デフォルト 0.85)
        remove_character_tags: Trueならキャラクタータグ(category=4)を除外する
    """
    session, tags_data = _ensure_model()

    inp = _preprocess(image_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    preds = session.run([output_name], {input_name: inp})[0][0]

    result: list[str] = []
    for i, tag in enumerate(tags_data):
        if i >= len(preds):
            break
        cat = int(tag.get("category", 9))
        score = float(preds[i])
        name = tag.get("name", "").replace("_", " ")

        if cat == 9:  # rating
            continue
        if cat == 5:  # copyright — LoRAに混入させない
            continue
        if cat == 4 and remove_character_tags:  # キャラクタータグ除外オプション
            continue
        threshold = character_thresh if cat == 4 else general_thresh
        if score >= threshold and name:
            result.append(name)

    return ", ".join(result)


def is_available() -> bool:
    return _ORT_AVAILABLE and _HF_AVAILABLE


def model_cached() -> bool:
    return (CACHE_DIR / MODEL_FILENAME).exists() and (CACHE_DIR / TAGS_FILENAME).exists()
