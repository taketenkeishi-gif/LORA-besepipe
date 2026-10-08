# LoRA-Trainer UI Refresh Agent Contract

## GPU execution

- Physical GPU1 (RTX 3090 Ti) is the only GPU authorized for this pipeline. GPU0 is prohibited.
  - Exception (user decision 2026-10-08): training-epoch PREVIEWS may run on physical GPU0 (RTX 3060) through the app's private
    ComfyUI on port 8189 (`backend/app/training/runtime/preview_gpu.py`), per run setting `preview_gpu` (default `gpu0`, switchable to
    `gpu1`). The app starts it only when a preview needs it and stops it after 5 idle minutes. Nothing else may use GPU0 or port 8189.
- NVIDIA SMI physical ordering and the target process's CUDA ordering must be compared by UUID/name/VRAM before execution. A process-local `cuda:0` may map to physical GPU1; never infer physical identity from the CUDA index alone.
- Authorized H3, Qwen, Anima, Preview, FlashVSR, RealESRGAN, ComfyUI, and Python GPU work must use the persistent serialized queue. GPU Busy is a normal `waiting` state, not a blocker or turn-ending condition.
- Before creating work, search for an equivalent `waiting`, `queued`, or `running` Run and resume it. Do not create duplicate H3 or training Runs.
- For a new authorized GPU Run, use `queue_if_busy=true`. Before yielding, verify Run ID, physical GPU1, `waiting`/`queued`, auto-start intent, and increasing admission probe or queue position.
- Never stop, restart, unload, move, or change an external process or its GPU assignment. The monitor must re-evaluate admission and start the same Run when GPU1 is safe.
- Human aesthetic decisions remain with the user. Queueing and technical dataset selection do not grant authority to approve frames, Qwen versions, previews, or final exports.

## Canonical continuation

Read `HANDOFF.md` and the current gate document it references before acting. Runtime evidence and durable Run state are authoritative; chat recollection is not.
