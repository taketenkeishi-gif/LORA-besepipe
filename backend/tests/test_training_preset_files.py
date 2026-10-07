"""Training preset file management against a temporary preset folder (the real folder is never touched)."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from fastapi import HTTPException

REAL_DIR = Path(__file__).resolve().parents[2] / "user-presets" / "training"


def make_payload(name="テスト", **config):
    base = {"model_family": "anima", "rank": 16, "alpha": 16, "epochs": 6, "learning_rate": 0.0001}
    return {"format": "lora-studio-training-preset", "version": 1, "name": name, "category": "character",
            "config": {**base, **config}}


class TrainingPresetFileTests(unittest.TestCase):
    def setUp(self):
        from app.routers import training_preset_files as t
        self.t = t
        self.work = Path(tempfile.mkdtemp())
        self.dir = self.work / "training"
        self.dir.mkdir()
        self.old_env = os.environ.get("LORA_STUDIO_PRESETS_DIR")
        os.environ["LORA_STUDIO_PRESETS_DIR"] = str(self.dir)
        self.launched = []
        self.old_launch = t._launch_explorer
        t._launch_explorer = self.launched.append
        self.real_before = sorted((p.name, p.stat().st_mtime) for p in REAL_DIR.glob("*.json")) if REAL_DIR.exists() else []

    def tearDown(self):
        self.t._launch_explorer = self.old_launch
        if self.old_env is None:
            os.environ.pop("LORA_STUDIO_PRESETS_DIR", None)
        else:
            os.environ["LORA_STUDIO_PRESETS_DIR"] = self.old_env
        shutil.rmtree(self.work, ignore_errors=True)
        real_after = sorted((p.name, p.stat().st_mtime) for p in REAL_DIR.glob("*.json")) if REAL_DIR.exists() else []
        self.assertEqual(self.real_before, real_after, "real preset folder must stay untouched")

    def save(self, name="A", **kw):
        return self.t.save_training_file(self.t.SaveBody(name=name, payload=make_payload(name), **kw))

    def test_save_list_and_conflict_then_overwrite_keeps_extra_keys(self):
        first = self.save("A")
        self.assertEqual(first["id"], "preset-A.json")
        listing = self.t.list_training_files()
        self.assertEqual([e["name"] for e in listing], ["A"])
        with self.assertRaises(HTTPException) as ctx:
            self.save("A")
        self.assertEqual(ctx.exception.status_code, 409)
        # an extra key (like 'recommendation') survives overwrite
        path = self.dir / "preset-A.json"
        data = json.loads(path.read_text(encoding="utf8"))
        data["recommendation"] = {"basis": "x"}
        path.write_text(json.dumps(data), encoding="utf8")
        result = self.t.save_training_file(self.t.SaveBody(name="A", payload=make_payload("A", rank=64), overwrite=True))
        self.assertTrue(result["overwritten"])
        after = json.loads(path.read_text(encoding="utf8"))
        self.assertEqual(after["config"]["rank"], 64)
        self.assertEqual(after["recommendation"], {"basis": "x"})

    def test_rejects_bad_names_and_payload(self):
        for bad in ["a/b", "a:b", "con", "trail.", "a*"]:
            with self.assertRaises(HTTPException, msg=bad):
                self.t.save_training_file(self.t.SaveBody(name=bad, payload=make_payload()))
        with self.assertRaises(HTTPException):
            self.t.save_training_file(self.t.SaveBody(name="X", payload={"format": "other"}))

    def test_rename_duplicate(self):
        self.save("A")
        self.save("B")
        renamed = self.t.rename_training_file("preset-A.json", self.t.NameBody(name="A2"))
        self.assertEqual(renamed["id"], "preset-A2.json")
        self.assertFalse((self.dir / "preset-A.json").exists())
        self.assertEqual(json.loads((self.dir / "preset-A2.json").read_text(encoding="utf8"))["name"], "A2")
        with self.assertRaises(HTTPException) as ctx:
            self.t.rename_training_file("preset-A2.json", self.t.NameBody(name="B"))
        self.assertEqual(ctx.exception.status_code, 409)
        copy1 = self.t.duplicate_training_file("preset-B.json")
        copy2 = self.t.duplicate_training_file("preset-B.json")
        self.assertEqual(copy1["name"], "B のコピー")
        self.assertEqual(copy2["name"], "B のコピー (2)")
        self.assertEqual(len(self.t.list_training_files()), 4)

    def test_delete_goes_to_recycle_bin(self):
        self.save("Gone")
        path = self.dir / "preset-Gone.json"
        self.t.delete_training_file("preset-Gone.json")
        self.assertFalse(path.exists())
        with self.assertRaises(HTTPException) as ctx:
            self.t.delete_training_file("preset-Gone.json")
        self.assertEqual(ctx.exception.status_code, 404)

    def test_path_traversal_refused(self):
        outside = self.work / "secret.json"
        outside.write_text(json.dumps(make_payload()), encoding="utf8")
        for evil in ["../secret.json", "..\\secret.json", str(outside), "sub/x.json", "notjson.txt", ""]:
            with self.assertRaises(HTTPException, msg=evil):
                self.t.delete_training_file(evil)
            if evil:  # an empty id means "open the folder" by design
                with self.assertRaises(HTTPException, msg=evil):
                    self.t.reveal_training_file(self.t.RevealBody(id=evil))
        self.assertTrue(outside.exists())
        self.assertEqual(self.launched, [])

    def test_reveal_builds_explorer_select_and_folder(self):
        self.save("A")
        self.t.reveal_training_file(self.t.RevealBody(id="preset-A.json"))
        self.assertEqual(self.launched[-1][0], "explorer.exe")
        self.assertTrue(self.launched[-1][1].startswith("/select,") and self.launched[-1][1].endswith("preset-A.json"))
        self.t.reveal_training_file(self.t.RevealBody())
        self.assertEqual(self.launched[-1], ["explorer.exe", str(self.dir)])

    def test_broken_file_does_not_hide_others(self):
        self.save("Good")
        (self.dir / "broken.json").write_text("{not json", encoding="utf8")
        (self.dir / "foreign.json").write_text(json.dumps({"nodes": []}), encoding="utf8")
        rows = {e["id"]: e for e in self.t.list_training_files()}
        self.assertIsNone(rows["preset-Good.json"]["error"])
        self.assertIsNotNone(rows["broken.json"]["error"])
        self.assertIsNone(rows["broken.json"]["payload"])
        self.assertIsNotNone(rows["foreign.json"]["error"])

    def test_existing_real_presets_are_readable(self):
        os.environ["LORA_STUDIO_PRESETS_DIR"] = str(REAL_DIR)
        rows = self.t.list_training_files()
        self.assertTrue(all(r["error"] is None for r in rows), [r["error"] for r in rows if r["error"]])


if __name__ == "__main__":
    unittest.main()
