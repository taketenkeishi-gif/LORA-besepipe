from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

from ..db import get_conn
from ..schemas import ProjectCreate, ProjectOut

router = APIRouter(prefix="/projects", tags=["projects"])
PROJECTS_ROOT = Path(__file__).resolve().parents[3] / "projects"


def _dataset_base_dir() -> Path:
    conn = get_conn()
    row = conn.execute("SELECT value FROM app_settings WHERE key = 'dataset_base_dir'").fetchone()
    conn.close()
    if row and row["value"]:
        return Path(str(row["value"]))
    return Path(r"C:\ポートフォリオ\SDXL\LoRA_Traning\dataset")


@router.post("", response_model=ProjectOut)
def create_project(payload: ProjectCreate) -> ProjectOut:
    PROJECTS_ROOT.mkdir(parents=True, exist_ok=True)
    base_dir = PROJECTS_ROOT / payload.name
    dataset_dir = base_dir / "dataset"
    captions_dir = base_dir / "captions"
    outputs_dir = base_dir / "outputs"
    library_dir = _dataset_base_dir() / payload.project_type / payload.name

    for d in [base_dir, dataset_dir, captions_dir, outputs_dir, library_dir]:
        d.mkdir(parents=True, exist_ok=True)

    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            INSERT INTO projects(name, project_type, base_dir, dataset_dir, captions_dir, outputs_dir, library_dir, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'idle')
            """,
            (
                payload.name,
                payload.project_type,
                str(base_dir),
                str(dataset_dir),
                str(captions_dir),
                str(outputs_dir),
                str(library_dir),
            ),
        )
        conn.commit()
    except Exception as exc:
        conn.close()
        raise HTTPException(status_code=400, detail=f"create failed: {exc}") from exc

    row = cur.execute(
        "SELECT id, name, project_type, status, base_dir, dataset_dir, captions_dir, outputs_dir, library_dir FROM projects WHERE name = ?",
        (payload.name,),
    ).fetchone()
    conn.close()
    return ProjectOut(**dict(row))


@router.get("", response_model=list[ProjectOut])
def list_projects() -> list[ProjectOut]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, name, project_type, status, base_dir, dataset_dir, captions_dir, outputs_dir, library_dir FROM projects ORDER BY id DESC"
    ).fetchall()
    conn.close()
    return [ProjectOut(**dict(r)) for r in rows]
