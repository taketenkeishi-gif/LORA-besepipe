from __future__ import annotations

from pydantic import BaseModel, Field


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    project_type: str = Field(default="character", pattern="^(character|style)$")


class ProjectOut(BaseModel):
    id: int
    name: str
    project_type: str
    status: str
    base_dir: str
    dataset_dir: str
    captions_dir: str
    outputs_dir: str
    library_dir: str


class CollectorScanIn(BaseModel):
    project_id: int
    url: str
    keyword: str = ""
    limit: int = Field(default=24, ge=1, le=200)


class CollectorImportIn(BaseModel):
    project_id: int
    selected_ids: list[int]
    naming_template: str = "{title}_{index}"


class RepeatFolderIn(BaseModel):
    project_id: int
    repeats: int = Field(default=5, ge=1, le=1000)
    folder_title: str = Field(default="default", min_length=1, max_length=120)


class DropUrlIn(BaseModel):
    project_id: int
    url: str = Field(min_length=5, max_length=2000)


class TrainingStartIn(BaseModel):
    project_id: int
    preset_id: int | None = None
    epochs: int = Field(default=5, ge=1, le=1000)
    repeats: int = Field(default=5, ge=1, le=1000)
    alpha: float = Field(default=4.0, ge=0.1, le=128.0)
    rank: int = Field(default=16, ge=1, le=512)
    save_every_n_epochs: int = Field(default=1, ge=1, le=1000)
    output_name: str = Field(default="lora_output", min_length=1, max_length=120)
    base_checkpoint_path: str = ""
    train_data_dir: str = ""
    reg_data_dir: str = ""
    resolution: int = Field(default=512, ge=256, le=2048)


class TrainingControlIn(BaseModel):
    project_id: int


class GenerateTagsIn(BaseModel):
    project_id: int


class ToolPathsIn(BaseModel):
    python_exe: str = ""
    kohya_root: str = ""
    comfyui_root: str = ""
    wd14_script: str = ""
    temp_dir: str = ""
    dataset_base_dir: str = ""


class ToolPathsOut(BaseModel):
    python_exe: str
    kohya_root: str
    comfyui_root: str
    wd14_script: str
    temp_dir: str
    dataset_base_dir: str


class PreviewPromptsIn(BaseModel):
    positive_prompt: str = ""
    negative_prompt: str = ""


class PreviewPromptsOut(BaseModel):
    positive_prompt: str
    negative_prompt: str
