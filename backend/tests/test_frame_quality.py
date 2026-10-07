"""frame_quality must flag damaged middle frames and leave clean bursts alone (synthetic, deterministic)."""
from __future__ import annotations

import io
import random
import unittest

from PIL import Image, ImageDraw, ImageFilter


def _scene(seed: int, shift: int = 0) -> Image.Image:
    """Flat anime-like art: colour blocks plus thin dark lines (the hard case for block detection)."""
    import numpy as np

    rng = random.Random(seed)
    # smooth gradient background with faint texture: codec blocking is visible there, unlike on pure flat fills
    xs = np.linspace(0, 1, 640, dtype=np.float32)[None, :]
    ys = np.linspace(0, 1, 360, dtype=np.float32)[:, None]
    base = np.stack([200 + 40 * np.sin(3 * xs + ys), 190 + 40 * np.cos(2 * ys + xs), 210 + 30 * np.sin(xs * ys * 4)], axis=-1)
    base += np.random.default_rng(seed).normal(0, 2.0, base.shape).astype(np.float32)
    image = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    for _ in range(14):
        x, y = rng.randint(0, 560), rng.randint(0, 300)
        draw.rectangle([x + shift, y, x + shift + rng.randint(30, 90), y + rng.randint(30, 90)],
                       fill=(rng.randint(40, 255), rng.randint(40, 255), rng.randint(40, 255)), outline=(20, 20, 30), width=2)
    return image


def _jpeg(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=quality)
    return Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")


class FrameQualityTests(unittest.TestCase):
    def setUp(self):
        from app.services import frame_quality
        self.fq = frame_quality
        # three consecutive frames of a held shot (edges keep their position on the 8-px codec grid)
        self.prev, self.cur, self.nxt = _scene(1, 0), _scene(1, 0), _scene(1, 0)

    def test_clean_burst_is_usable(self):
        self.assertEqual(self.fq.flags(self.fq.assess(self.prev, self.cur, self.nxt)), [])

    def test_codec_blocking_is_flagged(self):
        self.assertIn("blocky", self.fq.flags(self.fq.assess(self.prev, _jpeg(self.cur, 8), self.nxt)))

    def test_blur_is_flagged(self):
        self.assertIn("blurry", self.fq.flags(self.fq.assess(self.prev, self.cur.filter(ImageFilter.GaussianBlur(3)), self.nxt)))

    def test_foreign_frame_is_jitter(self):
        # a completely different picture sitting between two similar frames
        self.assertIn("jitter", self.fq.flags(self.fq.assess(self.prev, _scene(99, 0), self.nxt)))

    def test_blend_of_two_different_neighbours_is_ghost(self):
        far = _scene(1, 40)
        ghost = Image.blend(self.prev, far, 0.5)
        self.assertIn("ghost", self.fq.flags(self.fq.assess(self.prev, ghost, far)))

    def test_missing_next_frame_does_not_crash(self):
        self.assertIsInstance(self.fq.assess(self.prev, self.cur, None)["sharpness"], float)

    def test_quality_score_prefers_the_clean_candidate(self):
        good = self.fq.quality_score(self.fq.assess(self.prev, self.cur, self.nxt))
        bad = self.fq.quality_score(self.fq.assess(self.prev, _jpeg(self.cur, 8), self.nxt))
        self.assertGreater(good, bad)


if __name__ == "__main__":
    unittest.main()
