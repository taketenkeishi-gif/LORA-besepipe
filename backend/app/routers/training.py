from __future__ import annotations

from fastapi import APIRouter

from ..schemas import TrainingControlIn, TrainingStartIn

router = APIRouter(prefix="/training", tags=["training"])


@router.post("/start")
def start(payload: TrainingStartIn) -> dict:
    return {
        "project_id": payload.project_id,
        "preset_id": payload.preset_id,
        "message": "TODO: kohya CLI起動を実装",
    }


@router.post("/stop-now")
def stop_now(payload: TrainingControlIn) -> dict:
    return {
        "project_id": payload.project_id,
        "mode": "stop_now",
        "message": "TODO: 最短安全地点停止を実装",
    }


@router.post("/stop-at-epoch")
def stop_at_epoch(payload: TrainingControlIn) -> dict:
    return {
        "project_id": payload.project_id,
        "mode": "stop_at_epoch",
        "message": "TODO: epoch区切り停止を実装",
    }


@router.post("/resume")
def resume(payload: TrainingControlIn) -> dict:
    return {
        "project_id": payload.project_id,
        "message": "TODO: 最新checkpointから再開を実装",
    }


@router.get("/status")
def status(project_id: int) -> dict:
    return {
        "project_id": project_id,
        "status": "idle",
        "epoch": 0,
        "step": 0,
        "message": "TODO: 実ジョブ状態を返す",
    }

