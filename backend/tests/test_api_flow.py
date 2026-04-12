from __future__ import annotations

import base64
import time
from pathlib import Path


def test_health_and_projects(client):
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    created = client.post("/projects", json={"name": "test_project", "project_type": "character"})
    assert created.status_code == 200
    project = created.json()
    assert project["name"] == "test_project"
    assert project["project_type"] == "character"
    assert "character" in project["library_dir"]

    listed = client.get("/projects")
    assert listed.status_code == 200
    items = listed.json()
    assert len(items) == 1
    assert items[0]["name"] == "test_project"


def test_collection_import_and_tags(client):
    created = client.post("/projects", json={"name": "collect_project", "project_type": "style"})
    project_id = created.json()["id"]
    dataset_dir = Path(created.json()["dataset_dir"])
    captions_dir = Path(created.json()["captions_dir"])

    scan = client.post(
        "/collector/scan",
        json={"project_id": project_id, "url": "https://example.com/user/mock", "keyword": "mock", "limit": 12},
    )
    assert scan.status_code == 200
    scan_json = scan.json()
    items = scan_json["items"]
    # URL取得不可時は無関係候補を返さない仕様のため、0件ならD&Dで候補を作る
    if scan_json["detected"] == 0:
        png_1x1 = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7ZbHkAAAAASUVORK5CYII="
        )
        dropped = client.post(
            "/collector/drop-files",
            data={"project_id": str(project_id)},
            files={"files": ("sample.png", png_1x1, "image/png")},
        )
        assert dropped.status_code == 200
        items = dropped.json()["items"]
    assert len(items) >= 1
    thumb = items[0]["thumbnail_url"]
    assert thumb.startswith("data:image/svg+xml;utf8,") or thumb.startswith("data:image/jpeg;base64,")

    selected_ids = [x["id"] for x in items[:4]]
    imported = client.post(
        "/collector/import",
        json={
            "project_id": project_id,
            "selected_ids": selected_ids,
            "naming_template": "{title}_{index}",
        },
    )
    assert imported.status_code == 200
    assert imported.json()["imported_count"] == len(selected_ids)

    for p in imported.json()["files"]:
        assert Path(p).exists()
        assert str(dataset_dir) in p

    generated = client.post("/tags/generate", json={"project_id": project_id})
    assert generated.status_code == 200
    assert generated.json()["generated_count"] == len(selected_ids)
    for p in generated.json()["files"]:
        assert Path(p).exists()
        assert str(captions_dir) in p


def test_training_progress_and_previews(client):
    created = client.post("/projects", json={"name": "training_project", "project_type": "character"})
    project_id = created.json()["id"]

    started = client.post(
        "/training/start",
        json={"project_id": project_id, "epochs": 2, "repeats": 2, "alpha": 4.0},
    )
    assert started.status_code == 200

    deadline = time.time() + 12
    status_payload = None
    while time.time() < deadline:
        status_resp = client.get(f"/training/status?project_id={project_id}")
        assert status_resp.status_code == 200
        status_payload = status_resp.json()
        if status_payload["status"] == "completed":
            break
        time.sleep(0.5)

    assert status_payload is not None
    assert status_payload["status"] == "completed"
    assert status_payload["epoch"] == 2

    repeat_folder = client.post(
        "/collector/prepare-repeat-folder",
        json={"project_id": project_id, "repeats": 5, "folder_title": "subject"},
    )
    assert repeat_folder.status_code == 200

    previews = client.get(f"/previews/{project_id}")
    assert previews.status_code == 200
    timeline = previews.json()["timeline"]
    assert len(timeline) >= 2
    # epoch 1/2でサンプルが作られていること
    all_slots = set()
    for item in timeline:
        all_slots.update(item["samples"].keys())
    assert {"face", "bust", "full", "bg"}.issubset(all_slots)


def test_settings_paths_and_status(client, tmp_path):
    current = client.get("/settings/tool-paths")
    assert current.status_code == 200
    assert set(current.json().keys()) == {
        "python_exe",
        "kohya_root",
        "comfyui_root",
        "wd14_script",
        "temp_dir",
        "dataset_base_dir",
    }

    tmp_kohya = (tmp_path / "kohya").resolve()
    tmp_comfy = (tmp_path / "comfy").resolve()
    tmp_wd14 = (tmp_path / "wd14.py").resolve()
    tmp_tmp = (tmp_path / "tmp").resolve()
    tmp_dataset = (tmp_path / "dataset").resolve()
    tmp_kohya.mkdir(parents=True, exist_ok=True)
    tmp_comfy.mkdir(parents=True, exist_ok=True)
    tmp_dataset.mkdir(parents=True, exist_ok=True)
    tmp_wd14.write_text("# dummy", encoding="utf-8")

    updated = client.put(
        "/settings/tool-paths",
        json={
            "python_exe": "",
            "kohya_root": str(tmp_kohya),
            "comfyui_root": str(tmp_comfy),
            "wd14_script": str(tmp_wd14),
            "temp_dir": str(tmp_tmp),
            "dataset_base_dir": str(tmp_dataset),
        },
    )
    assert updated.status_code == 200
    assert updated.json()["kohya_root"] == str(tmp_kohya)
    assert updated.json()["temp_dir"] == str(tmp_tmp)

    status = client.get("/settings/integrations/status")
    assert status.status_code == 200
    payload = status.json()
    assert "paths" in payload
    assert "checks" in payload
    assert set(payload["checks"].keys()) == {
        "python_exe",
        "kohya_root",
        "comfyui_root",
        "wd14_script",
        "temp_dir",
        "dataset_base_dir",
    }
