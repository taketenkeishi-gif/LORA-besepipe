"""Single-port entry for the separately authorized LoRA Studio copy."""
from contextlib import asynccontextmanager
from pathlib import Path
import hashlib
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from app.db import init_db
from app.main import app as api
from app.routers import dataset_files

# Capture fingerprints once at import time. Reading the edited files on every
# request would incorrectly claim that an old running process loaded new code.
BACKEND_FILES = [
    "app/training/learning_rates.py",
    "app/training/runtime/comfy_epoch_hook.py", "app/training/runtime/queue.py",
    "app/routers/evaluation.py", "app/routers/presets.py", "app/main.py",
    "app/training/draft_estimate.py", "app/training/run_cleanup.py",
    'app/training/advanced.py', 'app/training/advanced_catalog.json',
    'app/routers/dataset_files.py', 'app/routers/basepipe.py',
    'app/routers/training.py', 'app/routers/collector.py', 'app/routers/dataset.py',
    'app/training/snapshot_inputs.py', 'app/training/resume_checkpoint.py',
    'app/training/registry.py', 'app/training/runtime/metrics.py', 'app/routers/previews.py', 'app/training/runtime/monitor.py', 'app/training/runtime/anima_admission.py', 'app/schemas.py', 'app/training/runtime/preview_jobs.py', 'app/preview/providers/comfyui/provider.py', 'app/desktop_comfy.py', 'app/training/backends/shared/helpers.py',
    'app/training/backends/anima/backend.py', 'app/training/backends/anima/native_preview.py', 'app/training/backends/sdxl/backend.py',
]
BACKEND_BUILD = {name: hashlib.sha256((ROOT / 'backend' / name).read_bytes()).hexdigest()
                 for name in BACKEND_FILES}


@asynccontextmanager
async def lifespan(app):
    # This new copy must not resume historical GPU jobs from the database copy.
    # Jobs can only be started through the user's normal explicit API actions.
    init_db()
    yield


app = FastAPI(title='LoRA Studio Next', lifespan=lifespan)
app.mount('/api', api)
app.mount('/assets', StaticFiles(directory=ROOT / 'frontend/dist/assets'), name='assets')


@app.get('/instance')
def instance():
    path = Path(dataset_files.__file__)
    return {'application': 'LoRA Studio Next', 'root': str(ROOT), 'port': 5175,
            'dataset_api_sha256': BACKEND_BUILD['app/routers/dataset_files.py'],
            'backend_build': BACKEND_BUILD,
            'historical_jobs_auto_resumed': False,
            'python_executable': sys.executable, 'python_prefix': sys.prefix}


@app.get('/{page:path}')
def page(page: str):
    return FileResponse(ROOT / 'frontend/dist/index.html', headers={'Cache-Control': 'no-cache'})
