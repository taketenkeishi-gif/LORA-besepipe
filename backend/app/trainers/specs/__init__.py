# このモジュールを import するだけで全スペックが registry に登録される。
from . import krea2, wan21, hunyuanvideo, flux, sdxl, sd, qwen_image, qwen_image_edit, zimage, anima

__all__ = [
    "krea2", "wan21", "hunyuanvideo", "flux",
    "sdxl", "sd", "anima",
    "qwen_image", "qwen_image_edit", "zimage",
]
