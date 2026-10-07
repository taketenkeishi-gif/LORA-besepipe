"""Video -> per-character dataset job API: validation, files, rename/merge/import, cancel, listing.

No GPU and no real pipeline: the output folder is faked with PIL images + TXT, the process is a dummy python sleep.
Uses a COPIED database and temp folders (pattern of test_dataset_trash.py).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

from fastapi import HTTPException
from PIL import Image


def _payload(**over):
    from app.routers import dataset_video_chars as r
    base = dict(video_path="x.mp4")
    base.update(over)
    return r.StartPayload(**base)


class VideoDatasetJobTests(unittest.TestCase):
    def setUp(self):
        from app import db
        from app.routers import projects
        from app.services import video_dataset_job as svc
        self.db, self.projects, self.svc = db, projects, svc
        self.original_db = db.DB_PATH
        self.original_root = projects.PROJECTS_ROOT
        self.original_base = projects._dataset_base_dir
        self.work = Path(tempfile.mkdtemp())
        shutil.copy2(self.original_db, self.work / "workspace.db")
        db.DB_PATH = self.work / "workspace.db"
        projects.PROJECTS_ROOT = self.work / "projects"
        projects._dataset_base_dir = lambda: self.work / "library"
        self.root = self.work / "proj" / "dataset"
        self.root.mkdir(parents=True)
        conn = db.get_conn()
        self.pid = int(conn.execute("SELECT id FROM projects ORDER BY id LIMIT 1").fetchone()[0])
        conn.execute("UPDATE projects SET dataset_dir=? WHERE id=?", (str(self.root), self.pid))
        conn.execute("DELETE FROM basepipe_concepts WHERE project_id=?", (self.pid,))
        conn.commit()
        conn.close()
        self.job_id = "a1b2c3d4e5f60718"
        self.job = self.root.parent / ".video-datasets" / self.job_id
        self.job.mkdir(parents=True)
        self._fake_output()

    def tearDown(self):
        self.db.DB_PATH = self.original_db
        self.projects.PROJECTS_ROOT = self.original_root
        self.projects._dataset_base_dir = self.original_base
        with self.svc._lock:
            for live in self.svc._live.values():
                self.svc.kill_tree(live["proc"])
                try:
                    live["proc"].wait(10)
                    live["log"].close()
                except Exception:  # noqa: BLE001
                    pass
            self.svc._live.clear()
        shutil.rmtree(self.work, ignore_errors=True)

    # ---------------------------------------------------------------- fixtures
    def _outfit(self, char: str, trig: str, outfit: str, otrig: str, names: list[str]):
        folder = self.job / char / outfit
        folder.mkdir(parents=True, exist_ok=True)
        for n in names:
            Image.new("RGB", (8, 8), "red").save(folder / f"{n}.png")
            (folder / f"{n}.txt").write_text(f"{trig}, {otrig}, 1girl, blue hair", encoding="utf-8")

    def _fake_output(self):
        self._outfit("char_01_blue-hair", "ch01", "outfit_01_school-uniform", "ch01_o01", ["v_s0001_a", "v_s0002_a", "v_s0003_a"])
        self._outfit("char_01_blue-hair", "ch01", "outfit_00_不明（衣装が写らない）", "ch01_o00", ["v_s0004_a"])
        self._outfit("char_02_red-hair", "ch02", "outfit_01_school-uniform", "ch02_o01", ["v_s0001_a", "v_s0009_a"])
        self._outfit("char_03_green-hair", "ch03", "outfit_02_dress", "ch03_o02", ["v_s0020_a"])
        (self.job / "_仕分け不能（髪色が判定できない・少数キャラ）").mkdir()
        Image.new("RGB", (8, 8), "gray").save(self.job / "_仕分け不能（髪色が判定できない・少数キャラ）" / "u1.png")
        (self.job / "_contact_sheets").mkdir()
        (self.job / "_contact_sheets" / "char_02_red-hair.jpg").write_bytes(b"x")
        report = {"video": "v.mp4", "images_written": 7, "unassigned_images": 1, "scenes": 3, "characters": [
            {"folder": "char_01_blue-hair", "trigger": "ch01", "crops": 5, "views": {"front-upper": 4, "back-full": 1}, "outfits": [
                {"folder": "outfit_01_school-uniform", "trigger": "ch01_o01", "images": 3, "dropped_duplicates": 1, "top_clothing": ["school uniform"]},
                {"folder": "outfit_00_不明（衣装が写らない）", "trigger": "ch01_o00", "images": 1, "dropped_duplicates": 0, "top_clothing": []}]},
            {"folder": "char_02_red-hair", "trigger": "ch02", "crops": 2, "views": {"front-upper": 2}, "outfits": [
                {"folder": "outfit_01_school-uniform", "trigger": "ch02_o01", "images": 2, "dropped_duplicates": 0, "top_clothing": ["school uniform"]}]},
            {"folder": "char_03_green-hair", "trigger": "ch03", "crops": 1, "views": {"back-full": 1}, "outfits": [
                {"folder": "outfit_02_dress", "trigger": "ch03_o02", "images": 1, "dropped_duplicates": 0, "top_clothing": ["dress"]}]}]}
        (self.job / "report.json").write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
        self._meta("done")

    def _meta(self, status: str, **extra):
        meta = {"job_id": self.job_id, "project_id": self.pid, "status": status, "created_at": time.time(), "started_at": time.time() - 5, "finished_at": time.time(),
                "video": "v.mp4", "params": {"gpu": 1}, "output_dir": str(self.job), "pid": 0, "error": "", "images_written": 7}
        meta.update(extra)
        (self.job / "job.json").write_text(json.dumps(meta), encoding="utf-8")

    def _report(self):
        return json.loads((self.job / "report.json").read_text(encoding="utf-8"))

    # ---------------------------------------------------------------- contract validation
    def test_start_validation_messages_are_japanese(self):
        from app.routers import dataset_video_chars as r
        cases = [dict(scene_sensitivity="weird"), dict(others="all"), dict(frame_interval=0), dict(max_frames_per_scene=0), dict(overlap_threshold=2),
                 dict(long_side=100), dict(long_side=99999), dict(min_char_crops=0), dict(merge_quantile=0.1), dict(gpu=99), dict(limit_seconds=-1),
                 dict(video_path="clip.gif"), dict(video_path="  ")]
        for over in cases:
            with self.assertRaises(HTTPException, msg=str(over)) as ctx:
                r.start(self.pid, _payload(**over))
            self.assertEqual(ctx.exception.status_code, 422, str(over))
            self.assertTrue(any(ord(c) > 0x3000 for c in ctx.exception.detail), ctx.exception.detail)
        with self.assertRaises(HTTPException) as missing:  # valid params but no such video
            r.start(self.pid, _payload(video_path=str(self.work / "nope.mp4")))
        self.assertEqual(missing.exception.status_code, 404)

    def test_defaults_and_scene_sensitivity_mapping(self):
        params = self.svc.validate_params(_payload())
        self.assertEqual((params["scene_sensitivity"], params["long_side"], params["gpu"], params["limit_seconds"]), ("normal", 1024, None, 0))
        for name, thr in (("fine", "0.15"), ("normal", "0.3"), ("coarse", "0.5")):
            cmd = self.svc.build_command(Path("py.exe"), self.svc.validate_params(_payload(scene_sensitivity=name, long_side=0, use_names=True)), self.job, self.job / "progress.json")
            self.assertEqual(cmd[cmd.index("--threshold") + 1], thr)
            self.assertEqual(cmd[cmd.index("--long-side") + 1], "0")
            self.assertIn("--use-names", cmd)
            self.assertEqual(cmd[cmd.index("--progress-file") + 1], str(self.job / "progress.json"))

    def test_restart_adopts_running_job_and_judges_finished_ones(self):
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "video_to_dataset"])
        try:
            self._meta("running", pid=proc.pid)
            self.assertEqual(self.svc.refresh_meta(self.job)["status"], "running")      # still alive: adopted, not an error
        finally:
            proc.kill(); proc.wait(10)
        self._meta("running", pid=proc.pid)                                              # now dead, and report.json exists -> done
        self.assertEqual(self.svc.refresh_meta(self.job)["status"], "done")
        (self.job / "report.json").rename(self.job / "report.bak")
        self._meta("running", pid=proc.pid)                                              # dead and no result -> error
        meta = self.svc.refresh_meta(self.job)
        self.assertEqual(meta["status"], "error")
        (self.job / "report.bak").rename(self.job / "report.json")

    def test_cancel_adopted_job_after_restart(self):
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "video_to_dataset"])
        try:
            self._meta("running", pid=proc.pid)
            self.assertTrue(self.svc.cancel_adopted(self.job))
            proc.wait(15)
            self.assertEqual(self.svc.load_json(self.job / "job.json")["status"], "cancelled")
            self.assertFalse(self.svc.cancel_adopted(self.job))   # nothing left to cancel
        finally:
            if proc.poll() is None:
                proc.kill(); proc.wait(10)

    def test_only_one_job_at_a_time(self):
        video = self.work / "v.mp4"
        video.write_bytes(b"x")
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        self.svc._live["running1"] = {"proc": proc, "cancel": False, "dir": self.job, "log": open(self.work / "l.log", "wb"), "gpu": 0}
        from app.routers import dataset_video_chars as r
        self.assertTrue(self.svc.any_running(0))
        self.assertFalse(self.svc.any_running(1))  # another GPU may analyse another video at the same time
        with self.assertRaises(HTTPException) as ctx:
            r.start(self.pid, _payload(video_path=str(video), gpu=0))
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertIn("実行中", ctx.exception.detail)

    def test_missing_comfy_python_is_clear_409(self):
        saved = self.svc.KNOWN_COMFY_ROOT
        self.svc.KNOWN_COMFY_ROOT = self.work / "no" / "ComfyUI"
        try:
            with self.assertRaises(self.svc.VideoDatasetError) as ctx:
                self.svc.find_comfy("")
        finally:
            self.svc.KNOWN_COMFY_ROOT = saved
        self.assertEqual(ctx.exception.status, 409)
        self.assertIn("python_embeded", ctx.exception.message)

    # ---------------------------------------------------------------- status / report / files
    def test_status_report_shape_and_extras(self):
        from app.routers import dataset_video_chars as r
        s = r.status(self.pid, self.job_id)
        for key in ("status", "stage", "percent", "elapsed_s", "eta_s", "error", "log_tail", "params", "output_dir", "report"):
            self.assertIn(key, s)
        self.assertEqual((s["status"], s["percent"], s["eta_s"]), ("done", 100, 0.0))
        rep = s["report"]
        ch = rep["characters"][0]
        self.assertEqual(ch["images_total"], 4)
        self.assertLessEqual(len(ch["sample_files"]), 12)
        self.assertTrue(all((self.job / p).is_file() for p in ch["sample_files"]))
        self.assertEqual(len(rep["unassigned_sample_files"]), 1)
        self.assertEqual(rep["rejected_frame_examples"], [])

    def test_running_status_uses_progress_file_and_eta(self):
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        self.svc._live[self.job_id] = {"proc": proc, "cancel": False, "dir": self.job, "log": open(self.work / "l.log", "wb")}
        self._meta("running", finished_at=None, started_at=time.time() - 100)
        (self.job / "progress.json").write_text(json.dumps({"stage": "人物検出", "percent": 50, "detail": "", "updated_at": time.time()}), encoding="utf-8")
        (self.job / "run.log").write_text("hello\nException ignored in: <function ModelPatcher.__del__>\nTraceback (most recent call last):\n  File \"x\", line 1\nAttributeError: ON_DETACH\nlast line\n", encoding="utf-8")
        s = self.svc.status_payload(self.job)
        self.assertEqual((s["status"], s["stage"], s["percent"]), ("running", "人物検出", 50))
        self.assertAlmostEqual(s["eta_s"], 100, delta=3)
        self.assertEqual(s["log_tail"], ["hello", "last line"])  # ON_DETACH noise filtered
        self.assertIsNone(s["report"])

    def test_file_endpoint_refuses_traversal_and_bad_types(self):
        from app.routers import dataset_video_chars as r
        good = "char_01_blue-hair/outfit_01_school-uniform/v_s0001_a.png"
        resp = r.file(self.pid, self.job_id, good)
        self.assertEqual(Path(resp.path).name, "v_s0001_a.png")
        (self.job / "secret.png").write_bytes(b"x")
        (self.work / "outside.png").write_bytes(b"x")
        for bad in ("../outside.png", "..\\outside.png", "char_01_blue-hair/../../outside.png", str(self.work / "outside.png"), "/etc/x.png", "C:/x.png",
                    "report.json", "char_01_blue-hair/outfit_01_school-uniform/v_s0001_a.txt", "job.json", "a//b.png", "./x.png"):
            with self.assertRaises(HTTPException, msg=bad) as ctx:
                r.file(self.pid, self.job_id, bad)
            self.assertIn(ctx.exception.status_code, (400, 404), bad)
        with self.assertRaises(HTTPException) as nf:
            r.file(self.pid, self.job_id, "char_01_blue-hair/nope.png")
        self.assertEqual(nf.exception.status_code, 404)
        with self.assertRaises(HTTPException):
            r.status(self.pid, "../../x")

    def test_file_route_through_http(self):
        try:
            from fastapi.testclient import TestClient
        except Exception:  # noqa: BLE001
            self.skipTest("TestClient unavailable")
        from app.main import app
        client = TestClient(app)
        base = f"/dataset-files/{self.pid}/video-dataset"
        self.assertEqual(client.get(f"{base}/jobs").json()[0]["job_id"], self.job_id)  # /jobs is not swallowed by /{job_id}
        ok = client.get(f"{base}/{self.job_id}/file", params={"rel": "char_03_green-hair/outfit_02_dress/v_s0020_a.png"})
        self.assertEqual((ok.status_code, ok.headers["content-type"]), (200, "image/png"))
        self.assertEqual(client.get(f"{base}/{self.job_id}/file", params={"rel": "../x.png"}).status_code, 400)
        self.assertEqual(client.get(f"{base}/{self.job_id}").json()["status"], "done")

    # ---------------------------------------------------------------- rename
    def test_rename_character(self):
        from app.routers import dataset_video_chars as r
        rep = r.rename(self.pid, self.job_id, r.RenamePayload(folder="char_02_red-hair", new_name="アカリ"))
        self.assertTrue((self.job / "アカリ" / "outfit_01_school-uniform" / "v_s0001_a.png").is_file())
        self.assertFalse((self.job / "char_02_red-hair").exists())
        self.assertEqual([c["folder"] for c in rep["characters"]][1], "アカリ")
        self.assertEqual(self._report()["characters"][1]["folder"], "アカリ")
        self.assertIn("ch02, ch02_o01", (self.job / "アカリ" / "outfit_01_school-uniform" / "v_s0001_a.txt").read_text(encoding="utf-8"))  # triggers untouched
        self.assertTrue((self.job / "_contact_sheets" / "アカリ.jpg").is_file())
        for bad in ("../x", "a/b", "", "  ", ".hid", "_x", "CON", "bad?", "char_01_blue-hair", "x" * 90):
            with self.assertRaises(HTTPException, msg=bad):
                r.rename(self.pid, self.job_id, r.RenamePayload(folder="char_03_green-hair", new_name=bad))
        with self.assertRaises(HTTPException) as nf:
            r.rename(self.pid, self.job_id, r.RenamePayload(folder="nope", new_name="x"))
        self.assertEqual(nf.exception.status_code, 404)

    def test_mutations_require_finished_job(self):
        from app.routers import dataset_video_chars as r
        self._meta("running")
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        self.svc._live[self.job_id] = {"proc": proc, "cancel": False, "dir": self.job, "log": open(self.work / "l.log", "wb")}
        with self.assertRaises(HTTPException) as ctx:
            r.rename(self.pid, self.job_id, r.RenamePayload(folder="char_02_red-hair", new_name="x"))
        self.assertEqual(ctx.exception.status_code, 409)

    # ---------------------------------------------------------------- merge
    def test_merge_rewrites_triggers_and_renumbers_outfits(self):
        from app.routers import dataset_video_chars as r
        rep = r.merge(self.pid, self.job_id, r.MergePayload(target="char_01_blue-hair", sources=["char_02_red-hair", "char_03_green-hair"]))
        self.assertEqual([c["folder"] for c in rep["characters"]], ["char_01_blue-hair"])
        self.assertFalse((self.job / "char_02_red-hair").exists() or (self.job / "char_03_green-hair").exists())
        target = rep["characters"][0]
        triggers = [o["trigger"] for o in target["outfits"]]
        self.assertEqual(len(triggers), len(set(triggers)), triggers)
        self.assertEqual(sorted(triggers), ["ch01_o00", "ch01_o01", "ch01_o02", "ch01_o03"])
        self.assertEqual(target["crops"], 5 + 2 + 1)
        self.assertEqual(target["views"], {"front-upper": 6, "back-full": 2})
        self.assertEqual(target["images_total"], 4 + 2 + 1)
        # every TXT: first token = target trigger, second = trigger that matches its own outfit folder
        for outfit in (self.job / "char_01_blue-hair").iterdir():
            number = outfit.name[len("outfit_"):len("outfit_") + 2]
            for txt in outfit.glob("*.txt"):
                tokens = txt.read_text(encoding="utf-8").split(", ")
                self.assertEqual(tokens[:2], ["ch01", f"ch01_o{number}"], txt)
                self.assertEqual(tokens[2:], ["1girl", "blue hair"])
            self.assertEqual(len(list(outfit.glob("*.png"))), len(list(outfit.glob("*.txt"))))
        self.assertFalse((self.job / "_contact_sheets" / "char_02_red-hair.jpg").exists())
        self.assertEqual(len(self._report()["characters"]), 1)

    def test_merge_validation_and_rollback(self):
        from app.routers import dataset_video_chars as r
        for payload in (r.MergePayload(target="char_01_blue-hair", sources=[]), r.MergePayload(target="char_01_blue-hair", sources=["char_01_blue-hair"]),
                        r.MergePayload(target="char_01_blue-hair", sources=["char_02_red-hair", "char_02_red-hair"]), r.MergePayload(target="nope", sources=["char_02_red-hair"]),
                        r.MergePayload(target="char_01_blue-hair", sources=["nope"])):
            with self.assertRaises(HTTPException):
                r.merge(self.pid, self.job_id, payload)
        before = sorted(p.relative_to(self.job).as_posix() for p in self.job.rglob("*"))
        report_before = (self.job / "report.json").read_bytes()
        original = self.svc.write_json
        calls = {"n": 0}

        def failing(path, data):
            if path.name == "report.json":
                calls["n"] += 1
                raise OSError("disk full")
            return original(path, data)
        self.svc.write_json = failing
        try:
            with self.assertRaises(HTTPException) as ctx:
                r.merge(self.pid, self.job_id, r.MergePayload(target="char_01_blue-hair", sources=["char_02_red-hair"]))
        finally:
            self.svc.write_json = original
        self.assertEqual(ctx.exception.status_code, 500)
        self.assertEqual(sorted(p.relative_to(self.job).as_posix() for p in self.job.rglob("*")), before)
        self.assertEqual((self.job / "report.json").read_bytes(), report_before)
        self.assertIn("ch02, ch02_o01", (self.job / "char_02_red-hair" / "outfit_01_school-uniform" / "v_s0001_a.txt").read_text(encoding="utf-8"))

    # ---------------------------------------------------------------- import
    def _concepts(self, project_id):
        conn = self.db.get_conn()
        try:
            return [(r["name"], r["concept_type"], r["trigger_token"]) for r in conn.execute(
                "SELECT name,concept_type,trigger_token FROM basepipe_concepts WHERE project_id=? ORDER BY id", (project_id,))]
        finally:
            conn.close()

    def test_import_current_project_copies_and_registers_without_clashes(self):
        from app.routers import dataset_video_chars as r
        before = sorted(p.relative_to(self.job).as_posix() for p in self.job.rglob("*"))
        out = r.import_characters(self.pid, self.job_id, r.ImportPayload(characters=["char_01_blue-hair", "char_02_red-hair", "ghost"], mode="current_project"))
        self.assertEqual([i["folder"] for i in out["imported"]], ["char_01_blue-hair", "char_02_red-hair"])
        self.assertEqual(out["skipped"][0]["folder"], "ghost")
        self.assertEqual([i["images"] for i in out["imported"]], [4, 2])
        self.assertTrue((self.root / "char_01_blue-hair" / "outfit_01_school-uniform" / "v_s0001_a.png").is_file())
        concepts = self._concepts(self.pid)
        self.assertEqual({c[2] for c in concepts}, {"ch01", "ch01_o01", "ch01_o00", "ch02", "ch02_o01"})
        self.assertEqual(sum(c[1] == "outfit" for c in concepts), 3)
        self.assertEqual(sum(c[1] == "character" for c in concepts), 2)
        # second import of the same character: never overwrites, triggers are remapped so none collide
        again = r.import_characters(self.pid, self.job_id, r.ImportPayload(characters=["char_01_blue-hair"], mode="current_project"))
        self.assertTrue((self.root / "char_01_blue-hair_2").is_dir())
        tokens = [c[2].casefold() for c in self._concepts(self.pid)]
        self.assertEqual(len(tokens), len(set(tokens)), tokens)
        new_char = next(c for c in again["imported"][0]["concepts"] if c["concept_type"] == "character")["trigger_token"]
        self.assertNotIn(new_char, ("ch01", "ch02"))
        txt = next((self.root / "char_01_blue-hair_2" / "outfit_01_school-uniform").glob("*.txt")).read_text(encoding="utf-8").split(", ")
        self.assertEqual(txt[0], new_char)
        self.assertEqual(txt[1], f"{new_char}_o01")
        # the job output is untouched
        self.assertEqual(sorted(p.relative_to(self.job).as_posix() for p in self.job.rglob("*")), before)

    def test_compose_merges_groups_of_several_jobs_into_one_project(self):
        from app.routers import dataset_video_link as link
        # a second finished job with the SAME looking character under another folder
        job2 = self.root.parent / ".video-datasets" / "b2c3d4e5f6071829"
        folder = job2 / "char_07_blue-hair" / "outfit_01_dress"
        folder.mkdir(parents=True)
        for n in ("w_s0001_a", "w_s0002_a"):
            Image.new("RGB", (8, 8), "blue").save(folder / f"{n}.png")
            (folder / f"{n}.txt").write_text("ch07, ch07_o01, 1girl, blue hair, dress", encoding="utf-8")
        (job2 / "report.json").write_text(json.dumps({"video": "w.mp4", "images_written": 2, "unassigned_images": 0, "characters": [
            {"folder": "char_07_blue-hair", "trigger": "ch07", "crops": 2, "views": {}, "outfits": [
                {"folder": "outfit_01_dress", "trigger": "ch07_o01", "images": 2, "dropped_duplicates": 0, "top_clothing": ["dress"]}]}]}), encoding="utf-8")
        (job2 / "job.json").write_text(json.dumps({"job_id": "b2c3d4e5f6071829", "project_id": self.pid, "status": "done", "video": "w.mp4", "params": {}, "output_dir": str(job2)}), encoding="utf-8")
        out = link.compose(self.pid, link.ComposePayload(name="Blue girl", members=[
            link.Member(job_id=self.job_id, folder="char_01_blue-hair"), link.Member(job_id="b2c3d4e5f6071829", folder="char_07_blue-hair")]))
        self.assertEqual((out["groups"], out["images"]), (2, 6))
        conn = self.db.get_conn()
        dataset = Path(conn.execute("SELECT dataset_dir FROM projects WHERE id=?", (out["project_id"],)).fetchone()[0])
        conn.close()
        concepts = self._concepts(out["project_id"])
        self.assertEqual(sum(c[1] == "character" for c in concepts), 1)          # ONE character...
        self.assertEqual(sum(c[1] == "outfit" for c in concepts), 3)             # ...three instances (2 outfits of the first + 1 of the second)
        tokens = [c[2] for c in concepts]
        self.assertEqual(len(tokens), len(set(tokens)), tokens)
        captions = [t.read_text(encoding="utf-8").split(", ")[0] for t in dataset.rglob("*.txt")]
        self.assertEqual(set(captions), {"ch01"})                                  # every image carries the one shared trigger
        with self.assertRaises(HTTPException):
            link.compose(self.pid, link.ComposePayload(name="x", members=[link.Member(job_id=self.job_id, folder="ghost")]))
        with self.assertRaises(HTTPException):
            link.compose(self.pid, link.ComposePayload(name="x", members=[]))

    def test_import_new_projects(self):
        from app.routers import dataset_video_chars as r
        out = r.import_characters(self.pid, self.job_id, r.ImportPayload(characters=["char_01_blue-hair", "char_03_green-hair"], mode="new_projects"))
        self.assertEqual(len(out["imported"]), 2)
        ids = [i["project_id"] for i in out["imported"]]
        self.assertEqual(len(set(ids)), 2)
        self.assertNotIn(self.pid, ids)
        conn = self.db.get_conn()
        rows = {row["id"]: dict(row) for row in conn.execute("SELECT id,name,dataset_dir FROM projects WHERE id IN (?,?)", ids)}
        conn.close()
        first = rows[ids[0]]
        self.assertEqual(first["name"], "char_01_blue-hair")
        self.assertTrue((Path(first["dataset_dir"]) / "outfit_01_school-uniform" / "v_s0001_a.png").is_file())
        self.assertTrue(str(first["dataset_dir"]).startswith(str(self.work)))  # sandboxed by the test patch
        self.assertEqual({c[2] for c in self._concepts(ids[0])}, {"ch01", "ch01_o01", "ch01_o00"})
        self.assertEqual({c[2] for c in self._concepts(ids[1])}, {"ch03", "ch03_o02"})
        # the instances feature sees the outfits
        from app.routers import training
        state = training._trigger_state(ids[0])
        self.assertEqual(state["trigger_token"], "ch01")
        self.assertEqual({i["trigger_token"] for i in state["instances"]}, {"ch01_o01", "ch01_o00"})
        for bad in (r.ImportPayload(characters=[], mode="new_projects"), r.ImportPayload(characters=["char_01_blue-hair"], mode="x")):
            with self.assertRaises(HTTPException):
                r.import_characters(self.pid, self.job_id, bad)

    # ---------------------------------------------------------------- reveal
    def test_reveal_arguments_and_call(self):
        from app.routers import dataset_video_chars as r
        self.assertEqual(r.explorer_args(Path(r"C:\a b\x")), ["explorer.exe", str(Path(r"C:\a b\x"))])
        calls = []
        original = subprocess.Popen
        subprocess.Popen = lambda args, **kw: calls.append((args, kw))
        try:
            out = r.reveal(self.pid, self.job_id)
        finally:
            subprocess.Popen = original
        self.assertEqual(out["path"], str(self.job))
        self.assertEqual(calls[0][0], ["explorer.exe", str(self.job)])
        self.assertNotIn("shell", calls[0][1])

    # ---------------------------------------------------------------- cancel / listing / restart
    def test_cancel_kills_process_tree_and_marks_cancelled(self):
        from app.routers import dataset_video_chars as r
        self._meta("running", finished_at=None)
        (self.job / "report.json").unlink()
        child = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)']); time.sleep(120)"
        proc = subprocess.Popen([sys.executable, "-c", child])
        log = open(self.job / "run.log", "wb")
        self.svc._live[self.job_id] = {"proc": proc, "cancel": False, "dir": self.job, "log": log}
        watcher = threading.Thread(target=self.svc._watch, args=(self.job_id,), daemon=True)
        watcher.start()
        time.sleep(1.0)
        self.assertEqual(r.status(self.pid, self.job_id)["status"], "running")
        self.assertTrue(r.cancel(self.pid, self.job_id)["cancelled"])
        watcher.join(20)
        self.assertFalse(watcher.is_alive())
        self.assertIsNotNone(proc.poll())
        s = r.status(self.pid, self.job_id)
        self.assertEqual(s["status"], "cancelled")
        with self.assertRaises(HTTPException) as again:
            r.cancel(self.pid, self.job_id)
        self.assertEqual(again.exception.status_code, 409)

    def test_failed_process_reports_error_from_log(self):
        self._meta("running", finished_at=None)
        (self.job / "report.json").unlink()
        log = open(self.job / "run.log", "wb")
        proc = subprocess.Popen([sys.executable, "-c", "import sys; print('[video_dataset] ERROR: RuntimeError: 動画の長さを読み取れませんでした', file=sys.stderr); sys.exit(1)"], stderr=log,
                                env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
        self.svc._live[self.job_id] = {"proc": proc, "cancel": False, "dir": self.job, "log": log}
        self.svc._watch(self.job_id)
        s = self.svc.status_payload(self.job)
        self.assertEqual(s["status"], "error")
        self.assertIn("動画の長さ", s["error"])

    def test_success_even_with_nonzero_exit_when_report_exists(self):
        self._meta("running", finished_at=None)
        log = open(self.job / "run.log", "wb")
        proc = subprocess.Popen([sys.executable, "-c", "import sys; sys.exit(3)"], stderr=log)
        self.svc._live[self.job_id] = {"proc": proc, "cancel": False, "dir": self.job, "log": log}
        self.svc._watch(self.job_id)
        self.assertEqual(self.svc.status_payload(self.job)["status"], "done")

    def test_jobs_listing_and_restart_interruption(self):
        from app.routers import dataset_video_chars as r
        second = self.root.parent / ".video-datasets" / "ffffffff00000001"
        second.mkdir()
        (second / "job.json").write_text(json.dumps({"job_id": "ffffffff00000001", "project_id": self.pid, "status": "running", "created_at": time.time() + 10,
                                                    "started_at": time.time(), "finished_at": None, "video": "w.mp4", "images_written": None}), encoding="utf-8")
        other = self.root.parent / ".video-datasets" / "eeeeeeee00000002"
        other.mkdir()
        (other / "job.json").write_text(json.dumps({"job_id": "eeeeeeee00000002", "project_id": self.pid + 999, "status": "done", "created_at": 1, "video": "z.mp4"}), encoding="utf-8")
        rows = r.jobs(self.pid)
        self.assertEqual([x["job_id"] for x in rows], ["ffffffff00000001", self.job_id])  # newest first, other project's job hidden
        self.assertEqual(rows[0]["status"], "error")  # was 'running' with no live process
        self.assertEqual(rows[1]["images_written"], 7)
        s = r.status(self.pid, "ffffffff00000001")
        self.assertEqual((s["status"], s["error"]), ("error", "アプリの再起動で中断されました"))
        with self.assertRaises(HTTPException) as ctx:
            r.status(self.pid, "eeeeeeee00000002")
        self.assertEqual(ctx.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
