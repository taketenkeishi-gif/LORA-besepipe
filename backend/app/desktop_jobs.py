"""Atomic desktop job state with bounded Windows reader-sharing retry."""
import time
from pathlib import Path
from .routers.dataset_files import atomic_text

def write_state(path:Path,text:str):
    for attempt in range(20):
        try:return atomic_text(path,text)
        except OSError as error:
            if getattr(error,'winerror',None) not in (5,32,33) or attempt==19:raise
            time.sleep(.025)
