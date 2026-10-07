"""Recycle-Bin delete + folder creation for the dataset editor (temp project root, DB copy)."""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from fastapi import HTTPException
from PIL import Image


class DatasetTrashTests(unittest.TestCase):
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
        conn.execute("UPDATE projects SET dataset_dir=? WHERE id=?", (str(self.root), self.pid))
        conn.commit()
        conn.close()

    def tearDown(self):
        self.db.DB_PATH = self.original
        shutil.rmtree(self.work, ignore_errors=True)

    def _image(self, rel: str):
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (4, 4), "red").save(path)
        path.with_suffix(".txt").write_text("tag", encoding="utf-8")
        return path

    def test_trash_images_with_sidecar_then_folder_and_create(self):
        from app.routers import dataset_files as d
        a, b = self._image("a.png"), self._image("sub/b.png")
        self.assertEqual(d.trash(self.pid, d.Paths(relatives=["a.png"]))["count"], 1)
        self.assertFalse(a.exists() or a.with_suffix(".txt").exists())
        self.assertEqual(d.trash_folder(self.pid, d.FolderPath(folder="sub"))["count"], 1)
        self.assertFalse((self.root / "sub").exists())
        made = d.create_folder(self.pid, d.NewFolder(parent="", name="outfit_A"))
        self.assertTrue((self.root / made["relative"]).is_dir())
        with self.assertRaises(HTTPException) as dup:
            d.create_folder(self.pid, d.NewFolder(parent="", name="outfit_A"))
        self.assertEqual(dup.exception.status_code, 409)
        for bad in ("../x", "a/b", "..", ".hidden", "x?"):
            with self.assertRaises(HTTPException, msg=bad):
                d.create_folder(self.pid, d.NewFolder(parent="", name=bad))
        with self.assertRaises(HTTPException):
            d.trash_folder(self.pid, d.FolderPath(folder="."))


if __name__ == "__main__":
    unittest.main()
