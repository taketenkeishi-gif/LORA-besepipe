"""RTX 3060 preview lane: the run-dir marker, and the idle stop that never leaves the instance resident."""
import json
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from app.training.runtime import preview_gpu


class PreviewGpuTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.state = self.tmp / "preview-3060.json"
        self.p = mock.patch.object(preview_gpu, "_STATE", self.state)
        self.p.start()
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("CREATE TABLE training_runs(id INTEGER, status TEXT)")
        self.conn.execute("CREATE TABLE preview_jobs(run_id INTEGER, status TEXT)")

    def tearDown(self):
        self.p.stop()

    def test_marker_switches_both_ways(self):
        run = self.tmp / "runs" / "5"
        preview_gpu.set_mode(run, "gpu0")
        self.assertTrue(preview_gpu.parallel(run))
        self.assertEqual(json.loads((run / preview_gpu.MARKER).read_text(encoding="utf-8"))["url"], preview_gpu.PREVIEW_URL)
        preview_gpu.set_mode(run, "gpu1")
        self.assertFalse(preview_gpu.parallel(run))

    def _instance(self, idle_for: float):
        self.state.write_text(json.dumps({"pid": 4242, "started": time.time() - 999, "last_used": time.time() - idle_for}), encoding="utf-8")

    def test_idle_stop_waits_for_training_and_pending_previews(self):
        run = self.tmp / "runs" / "7"
        preview_gpu.set_mode(run, "gpu0")
        with mock.patch.object(preview_gpu, "_alive", return_value=True), mock.patch("app.training.runtime.state.run_dir", lambda i: self.tmp / "runs" / str(i)), \
                mock.patch.object(preview_gpu, "stop", return_value=True) as stop:
            self._instance(idle_for=10_000)
            self.conn.execute("INSERT INTO training_runs VALUES (7,'training')")
            self.assertFalse(preview_gpu.stop_if_idle(self.conn))           # its run is still training
            self.conn.execute("UPDATE training_runs SET status='completed'")
            self.conn.execute("INSERT INTO preview_jobs VALUES (7,'pending')")
            self.assertFalse(preview_gpu.stop_if_idle(self.conn))           # a preview still waits for it
            self.conn.execute("UPDATE preview_jobs SET status='succeeded'")
            self._instance(idle_for=30)
            self.assertFalse(preview_gpu.stop_if_idle(self.conn))           # idle, but not for 5 minutes yet
            self._instance(idle_for=preview_gpu.IDLE_STOP_SECONDS + 1)
            self.assertTrue(preview_gpu.stop_if_idle(self.conn))            # idle long enough: stopped
            stop.assert_called_once()


if __name__ == "__main__":
    unittest.main()
