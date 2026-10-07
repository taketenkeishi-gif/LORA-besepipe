from __future__ import annotations

import csv
import io
import os
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


# 既にディスクにある同系統のWD14 v3モデルを読み取り専用で再利用する（ダウンロード・複製を避ける）。
# 環境変数 WD14_FALLBACK_DIRS（os.pathsep区切り）で追加できる。
_FALLBACK_DIRS = [Path(p) for p in os.environ.get("WD14_FALLBACK_DIRS", "").split(os.pathsep) if p] + [
    Path(r"C:\Users\Keishi\Portfolio\Toolchains-Physical\ComfyUI-Portable\ComfyUI_windows_portable\ComfyUI\custom_nodes\comfyui-wd14-tagger\models"),
]
_FALLBACK_MODELS = ("wd-eva02-large-tagger-v3", "wd-vit-large-tagger-v3", "wd-vit-tagger-v3", "wd-swinv2-tagger-v3")


def _local_model() -> tuple[Path, Path] | None:
    for directory in _FALLBACK_DIRS:
        for stem in _FALLBACK_MODELS:
            model, tags = directory / f"{stem}.onnx", directory / f"{stem}.csv"
            if model.is_file() and tags.is_file():
                return model, tags
    return None


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
        if not (model_path.exists() and tags_path.exists()):
            local = _local_model()
            if local is not None:
                model_path, tags_path = local

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
            [("CUDAExecutionProvider", {"device_id": 1}), "CPUExecutionProvider"]
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


def _preprocess_image(img: Image.Image) -> np.ndarray:
    # WD14モデル(SmilingWolf系)はcv2.imread由来のBGR順で学習されている
    # (kohya_ssのtag_images_by_wd14_tagger.py: `image = image[:, :, ::-1]  # RGB->BGR`
    # と同じ変換が必須)。RGBのまま渡すとR/Bチャンネルが入れ替わり、赤系統の色が
    # 青と誤判定される等、色タグが体系的に誤って付与される実バグがあった
    # (紫髪・青肌等、画像に存在しない色タグが混入する症状と一致)。
    img = img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    arr = arr[:, :, ::-1]  # RGB -> BGR
    return arr[np.newaxis, ...]


def _preprocess(image_path: str) -> np.ndarray:
    with Image.open(image_path) as img:
        return _preprocess_image(img)


def predict_image(
    image: Image.Image,
    general_thresh: float = GENERAL_THRESHOLD,
    character_thresh: float = CHARACTER_THRESHOLD,
    remove_character_tags: bool = False,
) -> str:
    """PIL.Image を受け取り、コンマ区切りのタグ文字列を返す（ファイル書き出し不要）。

    Dataset Pipeline の CaptionStep など、画像がまだメモリ上にしかない場面で使う。
    ロジックは predict() と同一（同じモデル・同じカテゴリ除外規則）。

    Args:
        image: 推論対象の画像（任意のモード。内部で RGB に変換する）
        general_thresh: 一般タグの信頼度しきい値 (0.05–0.95, デフォルト 0.35)
        character_thresh: キャラクタータグの信頼度しきい値 (0.05–0.99, デフォルト 0.85)
        remove_character_tags: Trueならキャラクタータグ(category=4)を除外する
    """
    session, tags_data = _ensure_model()

    inp = _preprocess_image(image)
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
    with Image.open(image_path) as img:
        return predict_image(
            img,
            general_thresh=general_thresh,
            character_thresh=character_thresh,
            remove_character_tags=remove_character_tags,
        )


def is_available() -> bool:
    return _ORT_AVAILABLE and _HF_AVAILABLE


def model_cached() -> bool:
    return ((CACHE_DIR / MODEL_FILENAME).exists() and (CACHE_DIR / TAGS_FILENAME).exists()) or _local_model() is not None
