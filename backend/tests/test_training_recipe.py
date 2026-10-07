"""Copied-DB regression checks for the common recipe API surface."""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
import hashlib


class TrainingRecipeTests(unittest.TestCase):
    def setUp(self):
        from app import db
        self.db = db
        self.original = db.DB_PATH
        self.temp = Path(tempfile.mkdtemp()) / "workspace.db"
        shutil.copy2(self.original, self.temp)
        db.DB_PATH = self.temp

    def tearDown(self):
        self.db.DB_PATH = self.original
        shutil.rmtree(self.temp.parent, ignore_errors=True)

    def test_common_merge_override_and_revision_invalidation(self):
        from app.routers import training_recipe as recipe
        from app.db import get_conn
        conn = get_conn()
        project_id = int(conn.execute("SELECT id FROM projects ORDER BY id LIMIT 1").fetchone()[0])
        conn.close()
        config = {
            "model_family": "anima", "base_checkpoint_path": "C:/models/anima.safetensors",
            "rank": 32, "alpha": 16.0, "epochs": 10, "repeats": 1, "resolution": 1024,
            "learning_rate": 0.0002, "train_batch_size": 1, "save_every_n_epochs": 1,
            "optimizer": "AdamW8bit", "scheduler": "cosine", "advanced": {},
            "preview_steps": 20, "preview_cfg": 5.0, "preview_lora_strength": 1.0,
        }
        self.assertFalse(recipe.get_common_recipe()["available"])
        common = recipe.put_common_recipe({"config": config})
        self.assertTrue(common["available"])
        recipe.put_project_recipe(project_id, {**config, "rank": 48, "output_name": "kept-per-project", "train_data_dir": "C:/dataset"})
        merged = recipe.get_project_recipe(project_id)
        self.assertEqual(merged["config"]["rank"], 48)
        self.assertEqual(merged["config"]["output_name"], "kept-per-project")
        self.assertEqual(merged["source"], "common_with_project_override")
        recipe.put_common_recipe({"config": {**config, "rank": 64}})
        revised = recipe.get_project_recipe(project_id)
        self.assertEqual(revised["config"]["rank"], 64)
        self.assertEqual(revised["source"], "common")

    def test_rejects_nonportable_and_invalid_family(self):
        from fastapi import HTTPException
        from app.routers import training_recipe as recipe
        with self.assertRaises(HTTPException):
            recipe.put_common_recipe({"config": {"model_family": "auto", "base_checkpoint_path": "x", "output_name": "no"}})

    def test_prepare_keeps_source_and_freezes_db_trigger_caption(self):
        from PIL import Image
        from app.db import get_conn
        from app.routers import training_recipe as recipe
        root = self.temp.parent / "dataset"
        root.mkdir()
        image = root / "sample.png"
        Image.new("RGB", (32, 32), (20, 30, 40)).save(image)
        txt = root / "sample.txt"
        txt.write_text("1girl, blue hair", encoding="utf-8")
        before_hash = hashlib.sha256(image.read_bytes()).hexdigest()
        before_txt = txt.read_bytes()
        conn = get_conn()
        cur = conn.execute(
            "INSERT INTO projects(name,project_type,base_dir,dataset_dir,captions_dir,outputs_dir,library_dir,status,training_config_json) VALUES(?,?,?,?,?,?,?,?,?)",
            ("recipe-fixture", "character", str(root), str(root), str(root), str(root / "out"), "", "idle", "{}"),
        )
        project_id = int(cur.lastrowid)
        conn.commit(); conn.close()
        result = recipe.prepare_project_dataset(project_id, {"folder": "", "trigger_token": "FIXTURE"})
        self.assertEqual(hashlib.sha256(image.read_bytes()).hexdigest(), before_hash)
        self.assertEqual(txt.read_bytes(), before_txt)
        self.assertEqual(result["snapshot"]["entries"][0]["caption_at_snapshot"], "FIXTURE, 1girl, blue hair")
        again = recipe.prepare_project_dataset(project_id, {"folder": "", "trigger_token": "FIXTURE"})
        self.assertEqual(result["snapshot"]["snapshot_hash"], again["snapshot"]["snapshot_hash"])


if __name__ == "__main__":
    unittest.main()
