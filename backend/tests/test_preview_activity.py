"""Copied-DB checks for the read-only preview_activity status block."""
from __future__ import annotations

import json
import shutil
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

RUN = 987654


def _ts(offset: float) -> str:
    return datetime.fromtimestamp(time.time() + offset, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


class PreviewActivityTests(unittest.TestCase):
    def setUp(self):
        from app import db
        self.db = db
        self.original = db.DB_PATH
        self.tmp = Path(tempfile.mkdtemp())
        self.temp = self.tmp / "workspace.db"
        shutil.copy2(self.original, self.temp)
        db.DB_PATH = self.temp
        self.run_dir = self.tmp / "run"
        (self.run_dir / "output" / "sample").mkdir(parents=True)

    def tearDown(self):
        self.db.DB_PATH = self.original
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _jobs(self, statuses, epoch=3, started_ago=40):
        conn = self.db.get_conn()
        conn.execute("DELETE FROM preview_jobs WHERE run_id=?", (RUN,))
        for i, status in enumerate(statuses):
            conn.execute(
                "INSERT INTO preview_jobs(checkpoint_id,run_id,project_id,epoch,prompt_index,instance_index,seed,status,"
                "error_detail,started_at,completed_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (1, RUN, 1, epoch, i, 0, 42 + i, status,
                 "GPU1待機: テスト" if status == "pending" else "",
                 _ts(-started_ago) if status in ("running", "succeeded") else None,
                 _ts(-started_ago + 5) if status == "succeeded" else None),
            )
        conn.commit()
        return conn

    def _collector(self, **fields):
        folder = self.run_dir / "output" / "sample" / "native-abc"
        folder.mkdir(exist_ok=True)
        (folder / "job.json").write_text(json.dumps({"status": "running", **fields}), encoding="utf8")

    def _build(self, conn, status="training", window=None):
        from app.training.runtime import preview_activity as pa
        with mock.patch("app.training.runtime.comfy_epoch_hook.active_window", return_value=window):
            return pa.build_preview_activity(conn, RUN, status, self.run_dir)

    def test_no_jobs_is_inactive(self):
        conn = self._jobs([])
        self.assertFalse(self._build(conn)["active"])

    def test_generating_with_step_progress_and_paused_training(self):
        conn = self._jobs(["succeeded", "succeeded", "succeeded", "running", "pending", "pending", "pending", "pending"])
        self._collector(progress_value=12, progress_max=20, message="生成中 12 / 20")
        window = {"epoch": 3, "step": 300, "deadline": time.time() + 1700, "state": "waiting"}
        a = self._build(conn, window=window)
        self.assertTrue(a["active"])
        self.assertEqual((a["phase"], a["done"], a["total"], a["pending"]), ("generating", 3, 8, 4))
        self.assertTrue(a["training_paused"])
        self.assertEqual((a["current_job"]["step"], a["current_job"]["step_max"]), (12, 20))
        self.assertFalse(a["stalled"])
        self.assertEqual(len(a["jobs"]), 8)

    def test_legacy_message_progress_is_parsed(self):
        conn = self._jobs(["running"])
        self._collector(message="生成中 5 / 25")
        a = self._build(conn)
        self.assertEqual((a["current_job"]["step"], a["current_job"]["step_max"]), (5, 25))
        self.assertFalse(a["training_paused"])

    def test_waiting_for_gpu_after_training(self):
        conn = self._jobs(["pending", "pending"])
        a = self._build(conn, status="completed")
        self.assertEqual(a["phase"], "waiting_gpu")
        self.assertIn("GPU1待機", a["waiting_reason"])

    def test_deferred_while_training_continues(self):
        conn = self._jobs(["pending"])
        a = self._build(conn, status="training")
        self.assertEqual(a["phase"], "deferred_until_training_ends")
        self.assertFalse(a["training_paused"])

    def test_stalled_when_collector_silent(self):
        conn = self._jobs(["running"], started_ago=900)
        self._collector(progress_value=3, progress_max=20)
        path = self.run_dir / "output" / "sample" / "native-abc" / "job.json"
        old = time.time() - 600
        import os
        os.utime(path, (old, old))
        a = self._build(conn)
        self.assertTrue(a["stalled"])

    def test_failed_job_reported_and_resolved_epoch_hidden(self):
        conn = self._jobs(["failed", "pending"])
        a = self._build(conn, status="completed")
        self.assertEqual(a["failed"], 1)
        conn = self._jobs(["succeeded", "failed"])
        self.assertFalse(self._build(conn, status="completed")["active"])


if __name__ == "__main__":
    unittest.main()
