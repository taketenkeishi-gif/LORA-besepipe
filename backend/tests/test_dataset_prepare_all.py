"""Prepare-all / summary / folder instance assignment (copied DB + temp dataset)."""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from fastapi import HTTPException
from PIL import Image


class DatasetPrepareAllTests(unittest.TestCase):
    def setUp(self):
        from app import db
        self.db = db
        self.original = db.DB_PATH
        self.work = Path(tempfile.mkdtemp())
        self.temp = self.work / "workspace.db"
        shutil.copy2(self.original, self.temp)
        db.DB_PATH = self.temp
        self.root = self.work / "dataset"
        self.root.mkdir()
        conn = db.get_conn()
        self.pid = int(conn.execute("SELECT id FROM projects ORDER BY id LIMIT 1").fetchone()[0])
        conn.execute("UPDATE projects SET dataset_dir=?, training_config_json='{}' WHERE id=?", (str(self.root), self.pid))
        conn.execute("DELETE FROM app_settings WHERE key=?", (f"evaluation_reference:{self.pid}",))
        conn.commit()
        conn.close()
        self.n = 0

    def tearDown(self):
        self.db.DB_PATH = self.original
        shutil.rmtree(self.work, ignore_errors=True)

    def _image(self, rel: str, caption="tag"):
        self.n += 1
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (4 + self.n, 4), (self.n * 7 % 256, self.n * 13 % 256, 50)).save(path)
        path.with_suffix(".txt").write_text(caption, encoding="utf-8")
        return path

    def test_summary_prepare_all_and_folder(self):
        from app.routers import dataset_files as d
        self._image("a.png"); self._image("a2.png")
        self._image("outfit_A/b.png"); self._image("outfit_A/c.png"); self._image("outfit_B/deep/e.png")
        self._image(".hidden/x.png")
        (self.root / "empty").mkdir()
        s = d.summary(self.pid)
        self.assertEqual(s["totals"], {"images": 5, "training_targets": 0, "folders": 3, "folders_with_targets": 0})
        by = {f["relative"]: f for f in s["folders"]}
        self.assertEqual(by["outfit_A"]["images"], 2)
        self.assertEqual(by["outfit_B"]["images"], 1)  # recursive
        self.assertNotIn(".hidden", by)
        # one folder only
        prepared = d.prepare_all(self.pid, d.PrepareScope(folder="outfit_A"))
        self.assertEqual(prepared["item_count"], 2)
        self.assertEqual(len(prepared["entries"]), 2)
        s = d.summary(self.pid)
        by = {f["relative"]: f for f in s["folders"]}
        self.assertEqual((by["outfit_A"]["training_targets"], by["outfit_A"]["images"]), (2, 2))
        self.assertEqual(by["outfit_B"]["training_targets"], 0)
        self.assertEqual(s["totals"]["training_targets"], 2)
        self.assertEqual(s["totals"]["folders_with_targets"], 1)
        # everything: one snapshot containing all 5 (hidden excluded)
        prepared = d.prepare_all(self.pid, d.PrepareScope())
        self.assertEqual(prepared["item_count"], 5)
        self.assertTrue(all(".hidden" not in e["file_path"] for e in prepared["entries"]))
        s = d.summary(self.pid)
        self.assertEqual(s["totals"], {"images": 5, "training_targets": 5, "folders": 3, "folders_with_targets": 3})
        self.assertEqual(s["snapshot_id"], prepared["id"])

    def test_evaluation_reference_excluded_and_empty_refused(self):
        from app.routers import dataset_files as d
        from app.routers import evaluation as ev
        with self.assertRaises(HTTPException) as e:
            d.prepare_all(self.pid, d.PrepareScope())
        self.assertEqual(e.exception.status_code, 422)
        self._image("a.png"); self._image("b.png"); self._image("c.png")
        ev.choose(self.pid, ev.Choice(relative="b.png"))
        prepared = d.prepare_all(self.pid, d.PrepareScope())
        names = sorted(Path(x["file_path"]).name for x in prepared["entries"])
        self.assertEqual(names, ["a.png", "c.png"])
        self.assertEqual(d.summary(self.pid)["totals"]["images"], 2)

    def test_assign_instance_to_folder(self):
        from app.routers import dataset_files as d
        from app.routers import training as t
        t.create_training_instance(self.pid, {"name": "制服", "trigger_token": "zz_uniform"})
        t.create_training_instance(self.pid, {"name": "私服", "trigger_token": "zz_casual"})
        a = self._image("A/1.png", "1girl, zz_casual, smile")
        b = self._image("A/2.png", "zz_uniform, solo")
        c = self._image("B/3.png", "untouched")
        r = d.assign_instance(self.pid, d.AssignInstance(folder="A", token="zz_uniform"))
        self.assertEqual(r, {"changed": 1, "unchanged": 1, "total": 2})
        self.assertEqual(a.with_suffix(".txt").read_text(encoding="utf-8"), "zz_uniform, 1girl, smile")
        self.assertEqual(b.with_suffix(".txt").read_text(encoding="utf-8"), "zz_uniform, solo")
        self.assertEqual(c.with_suffix(".txt").read_text(encoding="utf-8"), "untouched")
        with self.assertRaises(HTTPException) as e:
            d.assign_instance(self.pid, d.AssignInstance(folder="A", token="not_an_instance"))
        self.assertEqual(e.exception.status_code, 422)
        # a non-UTF8 sidecar aborts before ANY write
        a.with_suffix(".txt").write_text("1girl, zz_casual", encoding="utf-8")
        b.with_suffix(".txt").write_bytes(b"\xff\xfe\x00bad")
        with self.assertRaises(HTTPException):
            d.assign_instance(self.pid, d.AssignInstance(folder="A", token="zz_uniform"))
        self.assertEqual(a.with_suffix(".txt").read_text(encoding="utf-8"), "1girl, zz_casual")


if __name__ == "__main__":
    unittest.main()
