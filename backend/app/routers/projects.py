from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException

from ..db import get_conn
from ..schemas import ProjectCreate, ProjectDuplicateIn, ProjectOut, ProjectUpdateIn

router = APIRouter(prefix="/projects", tags=["projects"])
PROJECTS_ROOT = Path(__file__).resolve().parents[3] / "projects"


def _dataset_base_dir() -> Path:
    conn = get_conn()
    row = conn.execute("SELECT value FROM app_settings WHERE key = 'dataset_base_dir'").fetchone()
    conn.close()
    if row and row["value"]:
        return Path(str(row["value"]))
    return Path(r"C:\ポートフォリオ\SDXL\LoRA_Traning\dataset")


def _project_row(project_id: int):
    conn = get_conn()
    row = conn.execute(
        "SELECT id, name, project_type, status, base_dir, dataset_dir, captions_dir, outputs_dir, library_dir FROM projects WHERE id = ?",
        (project_id,),
    ).fetchone()
    conn.close()
    return row


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


@router.get("/{project_id}/cover")
def project_cover(project_id: int) -> dict:
    """プロジェクトアイコン用に、データセットから代表画像を最大4枚返す（2x2ボード用）。
    全体から均等サンプリングして偏りを減らす。"""
    conn = get_conn()
    rows = conn.execute(
        "SELECT file_path FROM dataset_items WHERE project_id = ? AND selected = 1 ORDER BY id",
        (project_id,),
    ).fetchall()
    conn.close()
    paths = [str(r["file_path"]) for r in rows if r["file_path"]]
    # 存在するファイルのみ
    paths = [p for p in paths if Path(p).exists()]
    if not paths:
        return {"project_id": project_id, "images": []}
    if len(paths) <= 4:
        picked = paths
    else:
        # 均等サンプリングで4枚
        step = len(paths) / 4
        picked = [paths[int(i * step)] for i in range(4)]
    return {"project_id": project_id, "images": picked}


@router.patch("/{project_id}", response_model=ProjectOut)
def update_project(project_id: int, payload: ProjectUpdateIn) -> ProjectOut:
    row = _project_row(project_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"project not found: {project_id}")
    row_d = dict(row)
    new_library = _dataset_base_dir() / payload.project_type / row_d["name"]
    new_library.mkdir(parents=True, exist_ok=True)

    conn = get_conn()
    conn.execute(
        "UPDATE projects SET project_type = ?, library_dir = ? WHERE id = ?",
        (payload.project_type, str(new_library), project_id),
    )
    conn.commit()
    conn.close()
    updated = _project_row(project_id)
    return ProjectOut(**dict(updated))


@router.post("/{project_id}/duplicate", response_model=ProjectOut)
def duplicate_project(project_id: int, payload: ProjectDuplicateIn) -> ProjectOut:
    row = _project_row(project_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"project not found: {project_id}")
    src = dict(row)
    created = create_project(ProjectCreate(name=payload.name, project_type=src["project_type"]))

    src_dataset = Path(src["dataset_dir"])
    src_captions = Path(src["captions_dir"])
    src_library = Path(src["library_dir"])
    dst_dataset = Path(created.dataset_dir)
    dst_captions = Path(created.captions_dir)
    dst_library = Path(created.library_dir)

    for p in src_dataset.glob("*"):
        if p.is_file():
            shutil.copy2(p, dst_dataset / p.name)
    for p in src_captions.glob("*"):
        if p.is_file():
            shutil.copy2(p, dst_captions / p.name)
    for p in src_library.glob("*"):
        if p.is_file():
            shutil.copy2(p, dst_library / p.name)
    return created


@router.delete("/{project_id}")
def delete_project(project_id: int) -> dict:
    row = _project_row(project_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"project not found: {project_id}")
    d = dict(row)
    base_dir = Path(d["base_dir"])
    library_dir = Path(d["library_dir"])
    dataset_base = _dataset_base_dir().resolve()

    conn = get_conn()
    cur = conn.cursor()
    ckpt_ids = [int(r["id"]) for r in cur.execute("SELECT id FROM checkpoints WHERE project_id = ?", (project_id,)).fetchall()]
    if ckpt_ids:
        marks = ",".join("?" for _ in ckpt_ids)
        cur.execute(f"DELETE FROM preview_samples WHERE checkpoint_id IN ({marks})", tuple(ckpt_ids))
    cur.execute("DELETE FROM checkpoints WHERE project_id = ?", (project_id,))
    cur.execute("DELETE FROM training_runs WHERE project_id = ?", (project_id,))
    cur.execute("DELETE FROM dataset_items WHERE project_id = ?", (project_id,))
    cur.execute("DELETE FROM projects WHERE id = ?", (project_id,))
    conn.commit()
    conn.close()

    if base_dir.exists() and PROJECTS_ROOT.resolve() in base_dir.resolve().parents:
        shutil.rmtree(base_dir, ignore_errors=True)
    if library_dir.exists() and dataset_base in library_dir.resolve().parents:
        shutil.rmtree(library_dir, ignore_errors=True)
    return {"project_id": project_id, "deleted": True}
