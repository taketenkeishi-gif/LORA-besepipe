import os
import sys
import time
from pathlib import Path

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # RTX 3090 Ti
import torch  # noqa: F401  (loads CUDA 13 / cuDNN 9 DLLs)

torch_lib = Path(torch.__file__).parent / "lib"
os.add_dll_directory(str(torch_lib))
os.environ["PATH"] = str(torch_lib) + os.pathsep + os.environ["PATH"]

import numpy as np
import onnxruntime as ort

root = Path(r"C:\Users\Keishi\Portfolio\Toolchains-Physical\ComfyUI-Portable\ComfyUI_windows_portable\ComfyUI")
model = root / "custom_nodes" / "comfyui-wd14-tagger" / "models" / "wd-eva02-large-tagger-v3.onnx"
t = time.time()
sess = ort.InferenceSession(str(model), providers=[("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"])
print("providers", sess.get_providers(), "load", round(time.time() - t, 1), flush=True)
name = sess.get_inputs()[0].name
print("input", sess.get_inputs()[0].shape, flush=True)
for batch in (1, 8):
    x = np.random.rand(batch, 448, 448, 3).astype(np.float32) * 255
    try:
        sess.run(None, {name: x})  # warm-up
        t = time.time()
        for _ in range(3):
            sess.run(None, {name: x})
        per = (time.time() - t) / 3 / batch
        print(f"wd14 batch={batch}: {per:.3f}s per image", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"wd14 batch={batch} failed: {str(exc)[:160]}", flush=True)

# upscaler through spandrel (what ComfyUI itself uses)
sys.path.insert(0, str(root))
os.chdir(root)
from spandrel import ModelLoader

t = time.time()
up = ModelLoader().load_from_file(str(root / "models" / "upscale_models" / "4x-UltraSharp.pth")).model.eval().cuda().half()
print("upscaler load", round(time.time() - t, 1), flush=True)
for h, w in ((300, 200), (600, 400)):
    x = torch.rand(1, 3, h, w, device="cuda").half()
    with torch.no_grad():
        up(x)
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(3):
            up(x)
        torch.cuda.synchronize()
    print(f"4x-UltraSharp {w}x{h}: {(time.time() - t) / 3:.3f}s", flush=True)
print("vram used GB", round(torch.cuda.max_memory_allocated() / 2**30, 2))
