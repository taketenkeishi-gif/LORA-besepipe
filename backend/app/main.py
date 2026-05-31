from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .db import init_db
from .routers import collector, dataset, previews, presets, projects, settings, tags, training

app = FastAPI(title="LoRA Workbench API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(projects.router)
app.include_router(collector.router)
app.include_router(dataset.router)
app.include_router(tags.router)
app.include_router(training.router)
app.include_router(previews.router)
app.include_router(settings.router)
app.include_router(presets.router)


@app.on_event("startup")
def on_startup() -> None:
    init_db()
    presets._seed_defaults_internal()  # デフォルトプロファイルを初回自動追加


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def root() -> dict[str, str]:
    return {"name": "LoRA Workbench API", "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)
