"""Copied-DB checks for per-image training instances (outfit concepts)."""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from fastapi import HTTPException


class TrainingInstanceTests(unittest.TestCase):
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

    def _project(self):
        conn = self.db.get_conn()
        pid = int(conn.execute("SELECT id FROM projects ORDER BY id LIMIT 1").fetchone()[0])
        conn.close()
        return pid

    def test_instance_lifecycle_and_token_collision(self):
        from app.routers import training as t
        pid = self._project()
        state = t.create_training_instance(pid, {"name": "制服", "trigger_token": "zz_inst_uniform"})
        self.assertEqual([i["trigger_token"] for i in state["instances"]], ["zz_inst_uniform"])
        state = t.create_training_instance(pid, {"name": "私服", "trigger_token": "zz_inst_casual"})
        self.assertEqual(len(state["instances"]), 2)
        with self.assertRaises(HTTPException) as dup:
            t.create_training_instance(pid, {"name": "x", "trigger_token": "ZZ_INST_UNIFORM"})
        self.assertEqual(dup.exception.status_code, 409)
        with self.assertRaises(HTTPException) as bad:
            t.create_training_instance(pid, {"name": "x", "trigger_token": "a, b"})
        self.assertEqual(bad.exception.status_code, 400)
        first = state["instances"][0]["id"]
        state = t.update_training_instance(pid, first, {"name": "制服2", "trigger_token": "zz_inst_uniform2"})
        self.assertEqual(state["instances"][0]["name"], "制服2")
        state = t.delete_training_instance(pid, first)
        self.assertEqual([i["trigger_token"] for i in state["instances"]], ["zz_inst_casual"])
        # the shared trigger is untouched by instance operations
        before = t._trigger_state(pid)["trigger_token"]
        t.create_training_instance(pid, {"name": "水着", "trigger_token": "zz_inst_swim"})
        self.assertEqual(t._trigger_state(pid)["trigger_token"], before)


if __name__ == "__main__":
    unittest.main()
