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


class ProjectUpdateIn(BaseModel):
    project_type: str = Field(pattern="^(character|style)$")


class ProjectDuplicateIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)


class CollectorScanIn(BaseModel):
    project_id: int
    url: str
    keyword: str = ""
    limit: int = Field(default=24, ge=1, le=200)


class CollectorImportIn(BaseModel):
    project_id: int
    selected_ids: list[int]
    naming_template: str = "{title}_{index}"
    import_dir: str = ""


class RepeatFolderIn(BaseModel):
    project_id: int
    repeats: int = Field(default=5, ge=1, le=1000)
    folder_title: str = Field(default="default", min_length=1, max_length=120)


class DropUrlIn(BaseModel):
    project_id: int
    url: str = Field(min_length=5, max_length=2000)


class CandidateRemoveIn(BaseModel):
    project_id: int
    candidate_ids: list[int] = Field(default_factory=list)


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
    optimizer: str = Field(default="AdamW8bit", max_length=64)
    scheduler: str = Field(default="cosine_with_restarts", max_length=64)
    min_snr_gamma: int | None = Field(default=5, ge=0, le=20)


class TrainingControlIn(BaseModel):
    project_id: int


class GenerateTagsIn(BaseModel):
    project_id: int
    overwrite: bool = False
    general_thresh: float = Field(default=0.35, ge=0.05, le=0.95)
    character_thresh: float = Field(default=0.85, ge=0.05, le=0.99)
    remove_character_tags: bool = False


class CaptionItem(BaseModel):
    id: int
    file_path: str
    width: int
    height: int
    aspect: str
    caption: str
    caption_source: str


class CaptionEditIn(BaseModel):
    caption: str


class BatchReplaceIn(BaseModel):
    project_id: int
    find: str
    replace: str
    item_ids: list[int] = Field(default_factory=list)


class BatchRemoveIn(BaseModel):
    project_id: int
    tags: list[str]
    item_ids: list[int] = Field(default_factory=list)


class TaggerStatusOut(BaseModel):
    project_id: int
    status: str
    total: int
    done: int
    message: str


class PresetCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    payload: dict


class PresetUpdateIn(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    payload: dict | None = None


class ToolPathsIn(BaseModel):
    python_exe: str = ""
    kohya_root: str = ""
    comfyui_root: str = ""
    wd14_script: str = ""
    temp_dir: str = ""
    dataset_base_dir: str = ""
    pixiv_session: str = ""


class ToolPathsOut(BaseModel):
    python_exe: str
    kohya_root: str
    comfyui_root: str
    wd14_script: str
    temp_dir: str
    dataset_base_dir: str
    pixiv_session: str = ""


class PreviewPromptsIn(BaseModel):
    positive_prompt: str = ""
    negative_prompt: str = ""


class PreviewPromptsOut(BaseModel):
    positive_prompt: str
    negative_prompt: str
