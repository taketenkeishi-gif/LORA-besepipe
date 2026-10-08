"""Short Japanese display names for what the video pipeline produced (folders keep their real names; only the UI label changes).

character("grey-hair-purple-eyes") -> "灰髪・紫瞳"; outfit("dress-black-dress") -> "黒ドレス"; tags too ("blonde hair", "purple eyes").
"""
from __future__ import annotations

import re

COLOURS = {"blonde": "金", "brown": "茶", "light brown": "明るい茶", "black": "黒", "blue": "青", "light blue": "水色", "dark blue": "紺",
           "aqua": "水色", "pink": "ピンク", "purple": "紫", "light purple": "薄紫", "green": "緑", "white": "白", "grey": "灰", "gray": "灰",
           "silver": "銀", "red": "赤", "orange": "オレンジ", "yellow": "黄", "multicolored": "多色", "two-tone": "ツートン"}
ITEMS = {"shirt": "シャツ", "dress": "ドレス", "skirt": "スカート", "jacket": "ジャケット", "coat": "コート", "uniform": "制服",
         "school uniform": "制服", "serafuku": "セーラー服", "necktie": "ネクタイ", "bowtie": "蝶ネクタイ", "ribbon": "リボン", "gloves": "手袋",
         "fingerless gloves": "指なし手袋", "hoodie": "パーカー", "sweater": "セーター", "vest": "ベスト", "kimono": "着物", "swimsuit": "水着",
         "bikini": "ビキニ", "apron": "エプロン", "pants": "ズボン", "shorts": "ショートパンツ", "cape": "マント", "hat": "帽子", "armor": "鎧",
         "pajamas": "パジャマ", "t-shirt": "Tシャツ", "blouse": "ブラウス", "cardigan": "カーディガン", "leotard": "レオタード", "suit": "スーツ",
         "collared shirt": "襟付きシャツ", "sailor collar": "セーラー襟", "hairband": "カチューシャ", "boots": "ブーツ", "thighhighs": "ニーソ",
         "pantyhose": "タイツ", "scarf": "マフラー", "headphones": "ヘッドホン", "glasses": "眼鏡", "frills": "フリル", "wings": "羽", "crown": "王冠",
         "tiara": "ティアラ", "maid": "メイド服", "nurse": "ナース服", "jersey": "ジャージ", "overalls": "オーバーオール", "robe": "ローブ"}


def _words(slug: str) -> list[str]:
    return [w for w in re.split(r"[-_\s]+", slug.lower()) if w]


def _colour_at(ws: list[str], i: int) -> tuple[str, int]:
    two = " ".join(ws[i:i + 2])
    if two in COLOURS:
        return COLOURS[two], 2
    return (COLOURS[ws[i]], 1) if ws[i] in COLOURS else ("", 0)


def _item_at(ws: list[str], i: int) -> tuple[str, int]:
    two = " ".join(ws[i:i + 2])
    if two in ITEMS:
        return ITEMS[two], 2
    return (ITEMS[ws[i]], 1) if i < len(ws) and ws[i] in ITEMS else ("", 0)


def character(slug: str = "", hair: str = "", eyes: str = "") -> str:
    """Hair and eye colour in Japanese, e.g. 灰髪・紫瞳.  Empty when nothing is known."""
    ws = _words(slug) if slug else _words(hair) + _words(eyes)
    parts = []
    for i, w in enumerate(ws):
        if w in ("hair", "eyes") and i > 0:
            j = i - 1
            c2 = " ".join(ws[max(0, i - 2):i])
            colour = COLOURS.get(c2) if c2 in COLOURS else COLOURS.get(ws[j])
            if colour:
                parts.append(colour + ("髪" if w == "hair" else "瞳"))
    seen = []
    for p in parts:
        if p not in seen:
            seen.append(p)
    return "・".join(seen)


def outfit(slug: str) -> str:
    """Clothing in Japanese, e.g. 黒ドレス・白シャツ.  Unknown words are dropped; empty when nothing is recognised."""
    ws = _words(slug)
    out: list[str] = []
    i = 0
    while i < len(ws):
        colour, n = _colour_at(ws, i)
        if colour:
            item, m = _item_at(ws, i + n)
            if item:
                out.append(colour + item)
                i += n + m
                continue
            i += n
            continue
        item, m = _item_at(ws, i)
        if item:
            if not any(o.endswith(item) for o in out):
                out.append(item)
            i += m
            continue
        i += 1
    dedup = []
    for o in out:  # keep the specific name only: ドレス + 黒ドレス -> 黒ドレス
        if o not in dedup and not any(p != o and p.endswith(o) for p in out):
            dedup.append(o)
    return "・".join(dedup[:3])


def folder_label(name: str, parent_is_job: bool, in_character: bool, video_name: str = "") -> str | None:
    """Display label for a pipeline folder, or None = hide it (diagnostic folders). Unrelated folders keep their name."""
    if parent_is_job and (name.startswith("_contact_sheets") or name.startswith("_画質フィルタ")):
        return None
    if parent_is_job and name.startswith("_仕分け不能"):
        return "未分類"
    m = re.fullmatch(r"char_(\d+)_(.*)", name)
    if parent_is_job and m:
        return character(m.group(2)) or f"キャラ{int(m.group(1))}"
    m = re.fullmatch(r"outfit_(\d+)_(.*)", name)
    if in_character and m:
        if m.group(1) in ("00",) or m.group(2).startswith("不明"):
            return "衣装不明"
        if m.group(1) == "99" or m.group(2).startswith("その他"):
            return "その他の衣装"
        return outfit(m.group(2)) or f"衣装{int(m.group(1))}"
    if video_name:
        return video_name
    return name
