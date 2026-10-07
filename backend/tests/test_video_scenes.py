"""Scene detection + one-representative-per-scene on a synthetic 3-cut video."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from PIL import Image


class VideoScenesTests(unittest.TestCase):
    def setUp(self):
        from app.services import video_scenes
        self.vs = video_scenes
        self.work = Path(tempfile.mkdtemp())
        self.video = self.work / "cuts.mp4"
        ffmpeg = video_scenes._ffmpeg()
        inputs = []
        for color in ("black", "white", "0x808080"):  # scene score is luma-based: pick cuts with a clear brightness change
            inputs += ["-f", "lavfi", "-i", f"color=c={color}:s=160x120:r=10:d=2"]
        subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", *inputs,
                        "-filter_complex", "[0][1][2]concat=n=3:v=1:a=0", "-pix_fmt", "yuv420p", "-y", str(self.video)],
                       check=True, timeout=120)

    def tearDown(self):
        shutil.rmtree(self.work, ignore_errors=True)

    def test_three_cuts_give_three_scenes_with_matching_colors(self):
        vs = self.vs
        self.assertAlmostEqual(vs.probe_duration(self.video), 6.0, delta=0.3)
        scenes = vs.build_scenes(vs.detect_cuts(self.video, 0.3), vs.probe_duration(self.video), 0.6)
        self.assertEqual(len(scenes), 3, scenes)
        items = vs.analyze_video(self.video, self.work / "out", threshold=0.3, min_seconds=0.6, samples=3,
                                 max_scenes=50, signature=[], tag_frame=None,
                                 progress=lambda *_: None, cancelled=lambda: False)
        self.assertEqual(len(items), 3)
        means = [round(sum(Image.open(self.work / "out" / item["file"]).convert("L").resize((1, 1)).getdata()) / 1) for item in items]
        self.assertLess(means[0], 20)
        self.assertGreater(means[1], 235)
        self.assertTrue(110 < means[2] < 150, means)
        # exactly one file per scene is kept (the losing samples are deleted)
        self.assertEqual(len(list((self.work / "out").glob("*.png"))), 3)

    def test_short_scenes_merge_and_signature_scoring(self):
        vs = self.vs
        # the 0.2s scene (1.0-1.2) is merged into the scene before it
        self.assertEqual(vs.build_scenes([1.0, 1.2, 3.0], 5.0, 0.6), [(0.0, 1.2), (1.2, 3.0), (3.0, 5.0)])
        signature = vs.character_signature(["1girl, orange hair, twintails", "1girl, orange hair, smile", "1girl, blue eyes"],
                                           exclude={"trigger"})
        self.assertEqual(signature, ["orange hair"])
        score, matched = vs.score_character("1girl, orange hair, smile", ["orange hair", "twintails"])
        self.assertEqual((score, matched), (0.5, ["orange hair"]))


if __name__ == "__main__":
    unittest.main()
