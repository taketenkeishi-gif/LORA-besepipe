"""
.lbp (LoRA Basepipe Project) file format — ZIP-based archive.

Layout:
  manifest.json  — format version, project metadata
  settings.json  — training preset / tool paths snapshot
  notes.md       — free-form memo
  previews/      — preview images (copied from preview_samples)
  dataset_cache/ — thumbnail cache (optional, skipped if large)
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from ..db import get_conn

router = APIRouter(prefix="/project-files", tags=["project-files"])

FORMAT_VERSION = 1


# ── Export ─────────────────────────────────────────────────────────────────


@router.get("/{project_id}/export")
def export_project(project_id: int) -> StreamingResponse:
    conn = get_conn()
    project = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # manifest.json
        manifest = {
            "format": "lbp",
            "version": FORMAT_VERSION,
            "project_name": project["name"],
            "project_type": project["project_type"],
        }
        zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))

        # settings.json — active preset + training run summary
        preset_id = project["preset_id"]
        preset_row = None
        if preset_id:
            preset_row = conn.execute("SELECT * FROM presets WHERE id = ?", (preset_id,)).fetchone()
        settings_data: dict = {
            "preset": dict(preset_row) if preset_row else None,
            "latest_run": None,
        }
        latest_run = conn.execute(
            "SELECT * FROM training_runs WHERE project_id = ? ORDER BY id DESC LIMIT 1",
            (project_id,),
        ).fetchone()
        if latest_run:
            settings_data["latest_run"] = dict(latest_run)
        zf.writestr("settings.json", json.dumps(settings_data, ensure_ascii=False, indent=2))

        # notes.md — placeholder
        zf.writestr("notes.md", f"# {project['name']}\n\nLoRA project notes\n")

        # previews/ — copy preview images
        previews_dir = Path(project["outputs_dir"]) / "previews" if project["outputs_dir"] else None
        if previews_dir and previews_dir.exists():
            for img_path in list(previews_dir.glob("*.png"))[:50]:  # cap at 50
                try:
                    zf.write(img_path, f"previews/{img_path.name}")
                except Exception:
                    pass

    buf.seek(0)
    filename = f"{project['name']}.lbp"
    return StreamingResponse(
        buf,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ── Import ─────────────────────────────────────────────────────────────────


@router.post("/import")
async def import_project(file: UploadFile) -> dict:
    content = await file.read()
    try:
        buf = io.BytesIO(content)
        with zipfile.ZipFile(buf, "r") as zf:
            names = zf.namelist()
            if "manifest.json" not in names:
                raise HTTPException(status_code=400, detail="Invalid .lbp file: missing manifest.json")

            manifest = json.loads(zf.read("manifest.json"))
            if manifest.get("format") != "lbp":
                raise HTTPException(status_code=400, detail="Not a valid .lbp file")

            project_name = manifest.get("project_name", "imported_project")
            project_type = manifest.get("project_type", "character")

            # Ensure unique name
            conn = get_conn()
            existing = conn.execute("SELECT id FROM projects WHERE name = ?", (project_name,)).fetchone()
            if existing:
                project_name = f"{project_name}_imported"

            # Create project via same logic as projects router
            base = Path(__file__).resolve().parents[3] / "workspace" / project_name
            base.mkdir(parents=True, exist_ok=True)
            dataset_dir = str(base / "dataset")
            captions_dir = str(base / "captions")
            outputs_dir = str(base / "outputs")
            library_dir = str(base / "library")
            for d in [dataset_dir, captions_dir, outputs_dir, library_dir]:
                Path(d).mkdir(parents=True, exist_ok=True)

            cur = conn.execute(
                """INSERT INTO projects (name, project_type, base_dir, dataset_dir, captions_dir, outputs_dir, library_dir)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (project_name, project_type, str(base), dataset_dir, captions_dir, outputs_dir, library_dir),
            )
            conn.commit()
            new_id = cur.lastrowid

            # Restore previews
            preview_out = Path(outputs_dir) / "previews"
            preview_out.mkdir(parents=True, exist_ok=True)
            for name in names:
                if name.startswith("previews/") and name != "previews/":
                    data = zf.read(name)
                    dest = preview_out / Path(name).name
                    dest.write_bytes(data)

        return {"project_id": new_id, "project_name": project_name}

    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="File is not a valid ZIP archive")
