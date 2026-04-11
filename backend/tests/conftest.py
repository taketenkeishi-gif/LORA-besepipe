from __future__ import annotations

import time
from pathlib import Path
import sys

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import db
from app.main import app
from app.routers import projects, training


@pytest.fixture()
def client(tmp_path: Path):
    db.DB_PATH = tmp_path / "test_workspace.db"
    projects.PROJECTS_ROOT = tmp_path / "projects"
    db.init_db()

    with TestClient(app) as c:
        yield c

    # テスト終了時に学習状態を止めて、ワーカースレッドの残留を避ける
    conn = db.get_conn()
    conn.execute("UPDATE training_runs SET status = 'paused'")
    conn.execute("UPDATE projects SET status = 'paused' WHERE status = 'training'")
    conn.commit()
    conn.close()
    time.sleep(0.6)

    for t in list(training.RUNNER_THREADS.values()):
        t.join(timeout=0.2)
