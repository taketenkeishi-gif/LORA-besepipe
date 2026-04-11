from __future__ import annotations

from pydantic import BaseModel, Field


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)


class ProjectOut(BaseModel):
    id: int
    name: str
    status: str
    base_dir: str
    dataset_dir: str
    captions_dir: str
    outputs_dir: str


class CollectorScanIn(BaseModel):
    project_id: int
    url: str


class CollectorImportIn(BaseModel):
    project_id: int
    selected_ids: list[int]
    naming_template: str = "{title}_{index}"


class TrainingStartIn(BaseModel):
    project_id: int
    preset_id: int | None = None
    total_epochs: int = Field(default=5, ge=1, le=1000)
    steps_per_epoch: int = Field(default=20, ge=1, le=100000)


class TrainingControlIn(BaseModel):
    project_id: int


class GenerateTagsIn(BaseModel):
    project_id: int
