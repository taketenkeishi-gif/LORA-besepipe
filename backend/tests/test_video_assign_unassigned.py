import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from app.services import video_dataset_job as svc


def _png(path: Path) -> None:
    Image.new("RGB", (8, 8), (10, 20, 30)).save(path)


class AssignUnassignedTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.job = Path(self.tmp.name)
        (self.job / "char_01_a" / "outfit_01_x").mkdir(parents=True)
        _png(self.job / "char_01_a" / "outfit_01_x" / "v_s0001_00m01s_front-upper_p0.png")
        rest = self.job / "_仕分け不能（髪色が判定できない・少数キャラ）"
        rest.mkdir()
        for n in range(3):
            _png(rest / f"v_s000{n}_00m0{n}s_front-other_p0.png")
            (rest / f"v_s000{n}_00m0{n}s_front-other_p0.txt").write_text("purple eyes, solo, ch09", encoding="utf-8")
        svc.write_json(self.job / "report.json", {"characters": [{"folder": "char_01_a", "trigger": "ch01", "crops": 1, "views": {}, "outfits": [{"folder": "outfit_01_x", "trigger": "ch01_o01", "images": 1, "dropped_duplicates": 0, "top_clothing": []}]}],
                                                  "unassigned_images": 3, "images_written": 1})
        self.files = sorted(p.relative_to(self.job).as_posix() for p in rest.glob("*.png"))

    def tearDown(self):
        self.tmp.cleanup()

    def test_move_to_existing_character(self):
        rep = svc.assign_unassigned(self.job, self.files[:2], "char_01_a", None)
        self.assertEqual(rep["unassigned_images"], 1)
        ch = rep["characters"][0]
        self.assertEqual(ch["crops"], 3)
        self.assertEqual(len(ch["all_files"]), 3)
        moved = next(self.job.glob("char_01_a/outfit_98_*/*.txt"))
        self.assertEqual(moved.read_text(encoding="utf-8"), "ch01, ch01_o98, purple eyes")
        self.assertEqual(len(rep["unassigned_files"]), 1)

    def test_new_character(self):
        rep = svc.assign_unassigned(self.job, self.files, None, "Nanoha")
        self.assertEqual(rep["unassigned_images"], 0)
        new = [c for c in rep["characters"] if c["folder"] == "Nanoha"][0]
        self.assertEqual(new["trigger"], "ch02")
        self.assertEqual(len(new["all_files"]), 3)

    def test_suggest_ranks_similar_tags_first(self):
        char = self.job / "char_01_a" / "outfit_01_x" / "v_s0001_00m01s_front-upper_p0"
        char.with_suffix(".txt").write_text("ch01, ch01_o01, purple eyes, brown hair, white jacket", encoding="utf-8")
        rest = next(self.job.glob("_仕分け不能*"))
        files = sorted(rest.glob("*.txt"))
        files[0].write_text("blonde hair, red eyes, black dress", encoding="utf-8")
        files[1].write_text("purple eyes, brown hair, white jacket", encoding="utf-8")
        ranked = svc.suggest_for_character(self.job, "char_01_a")
        self.assertEqual(Path(ranked[0]["file"]).stem, files[1].stem)
        self.assertGreater(ranked[0]["score"], ranked[-1]["score"])

    def test_rejects_foreign_paths_and_empty(self):
        with self.assertRaises(svc.VideoDatasetError):
            svc.assign_unassigned(self.job, ["char_01_a/outfit_01_x/v_s0001_00m01s_front-upper_p0.png"], "char_01_a", None)
        with self.assertRaises(svc.VideoDatasetError):
            svc.assign_unassigned(self.job, [], "char_01_a", None)


if __name__ == "__main__":
    unittest.main()
