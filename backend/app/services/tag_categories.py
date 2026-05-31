"""
SPEC §12 Caption Intelligence — カテゴリ分類
タグ名のパターンマッチングで SPEC 定義カテゴリに自動分類する。
"""
from __future__ import annotations

# ── SPEC §12 カテゴリ定義 ───────────────────────────────────────────────────
CATEGORIES = ["FACE", "HAIR", "COSTUME", "ACCESSORY", "BACKGROUND", "BODY", "COLOR", "STYLE", "GENERAL"]

# 各カテゴリのキーワード（タグ名に部分一致）
_PATTERNS: dict[str, list[str]] = {
    "HAIR": [
        "hair", "twintails", "ponytail", "braid", "ahoge", "bangs",
        "bob cut", "pixie cut", "drill", "ringlet", "locks", "strand",
        "blonde", "brunette", "silver hair", "white hair", "black hair",
        "brown hair", "red hair", "pink hair", "blue hair", "green hair",
        "purple hair", "orange hair", "grey hair", "gray hair", "multicolored hair",
        "two-tone hair", "gradient hair",
    ],
    "FACE": [
        "eyes", " eye", "iris", "pupil", "heterochromia",
        "smile", "laughing", "crying", "tears", "angry", "blush",
        "open mouth", "closed mouth", "expression", "grin", "frown",
        "nose", "lips", "mouth", "face", "cheek", "ear", "forehead",
        "serious", "surprised", "embarrassed", "scared", "sleepy",
        "pout", "smirk",
    ],
    "COSTUME": [
        "uniform", "dress", "skirt", "shirt", "jacket", "coat", "outfit",
        "costume", "clothes", "clothing", "wearing", "top", "bottom",
        "pants", "shorts", "blouse", "sweater", "hoodie", "cardigan",
        "apron", "kimono", "yukata", "maid", "suit", "tie", "bow tie",
        "scarf", "cape", "cloak", "armor", "swimsuit", "bikini",
        "leotard", "bodysuit", "school uniform", "sailor uniform",
        "military uniform", "sports uniform",
    ],
    "ACCESSORY": [
        "ribbon", "bow", "hairpin", "hair clip", "hair ornament",
        "headband", "hairband", "flower", "hat", "cap", "beret",
        "crown", "tiara", "glasses", "sunglasses", "earring", "necklace",
        "bracelet", "ring", "gloves", "stockings", "socks", "shoes",
        "boots", "bag", "backpack", "umbrella", "wand", "sword",
        "weapon", "wings", "tail", "cat ears", "animal ears",
    ],
    "BACKGROUND": [
        "background", "indoors", "outdoors", "sky", "ocean", "sea",
        "forest", "nature", "cityscape", "city", "school", "classroom",
        "park", "garden", "hallway", "corridor", "room", "bedroom",
        "kitchen", "bathroom", "night", "day", "sunset", "sunrise",
        "stars", "moon", "clouds", "rain", "snow", "wind",
        "simple background", "white background", "black background",
        "gradient background", "abstract background", "bokeh",
    ],
    "BODY": [
        "1girl", "1boy", "2girls", "3girls", "multiple", "solo",
        "standing", "sitting", "lying", "running", "walking", "jumping",
        "crouching", "kneeling", "floating", "dancing",
        "full body", "upper body", "lower body", "bust", "portrait",
        "close-up", "cowboy shot", "from above", "from below",
        "from side", "from behind",
    ],
    "COLOR": [
        "monochrome", "greyscale", "grayscale", "black and white",
        "colorful", "pale color", "vibrant", "pastel",
        "sepia", "red theme", "blue theme",
    ],
    "STYLE": [
        "realistic", "anime", "manga", "sketch", "lineart",
        "flat color", "cel shading", "watercolor", "oil painting",
        "pixel art", "chibi", "silhouette", "detailed", "highres",
        "low resolution", "blurry", "noise", "compression",
        "speech bubble", "text", "watermark", "signature", "logo",
    ],
}

# §11 Character Leak — 髪色・眼色・髪型パターン（偏り検出用）
HAIR_COLORS = [
    "blonde hair", "black hair", "brown hair", "white hair", "silver hair",
    "red hair", "pink hair", "blue hair", "green hair", "purple hair",
    "orange hair", "grey hair", "gray hair", "multicolored hair",
    "two-tone hair", "gradient hair",
]

HAIR_STYLES = [
    "long hair", "short hair", "medium hair", "very long hair",
    "twintails", "ponytail", "braid", "braided ponytail",
    "side ponytail", "low ponytail", "high ponytail",
    "bob cut", "pixie cut", "drill hair", "ahoge",
    "hime cut", "wavy hair", "curly hair", "straight hair",
    "messy hair", "hair down",
]

EYE_COLORS = [
    "blue eyes", "red eyes", "green eyes", "brown eyes", "purple eyes",
    "gold eyes", "yellow eyes", "orange eyes", "pink eyes", "grey eyes",
    "gray eyes", "black eyes", "silver eyes", "white eyes",
    "heterochromia",
]

COSTUME_TYPES = [
    "school uniform", "sailor uniform", "maid outfit", "military uniform",
    "swimsuit", "bikini", "casual", "dress", "kimono", "yukata",
    "suit", "sportswear", "pajamas", "apron",
]

# §11 character leak — キャラ固有タグとして登録される可能性が高い組み合わせ
LEAK_RISK_COMBOS = [
    ("HAIR_COLOR", HAIR_COLORS, 0.70),
    ("HAIR_STYLE", HAIR_STYLES, 0.70),
    ("EYE_COLOR", EYE_COLORS, 0.75),
    ("COSTUME", COSTUME_TYPES, 0.65),
]


def classify_tag(tag: str) -> str:
    """タグを SPEC §12 カテゴリに分類する。"""
    tag_lower = tag.lower()
    for category, patterns in _PATTERNS.items():
        if any(p in tag_lower for p in patterns):
            return category
    return "GENERAL"


def classify_caption(caption: str) -> dict[str, list[str]]:
    """キャプションを解析してカテゴリ別タグ辞書を返す。"""
    tags = [t.strip() for t in caption.split(",") if t.strip()]
    result: dict[str, list[str]] = {cat: [] for cat in CATEGORIES}
    for tag in tags:
        cat = classify_tag(tag)
        result[cat].append(tag)
    return result


def detect_dominant_feature(tags_list: list[str], feature_group: list[str], threshold: float) -> tuple[str | None, float]:
    """feature_group のいずれかが threshold 以上の割合で出現していれば (dominant, ratio) を返す。"""
    if not tags_list:
        return None, 0.0
    counts: dict[str, int] = {}
    for tag in tags_list:
        tag_lower = tag.lower()
        for feature in feature_group:
            if feature in tag_lower:
                counts[feature] = counts.get(feature, 0) + 1
                break
    if not counts:
        return None, 0.0
    dominant = max(counts, key=lambda k: counts[k])
    ratio = counts[dominant] / len(tags_list)
    if ratio >= threshold:
        return dominant, ratio
    return None, ratio
