"""General image features (CLIP ViT-H, ComfyUI's clip_vision_h) for every crop of a crop_features.py output.

python clip_features.py FEAT_PREFIX COMFY_ROOT
Writes FEAT_PREFIX.clip.npy (float16, L2-normalised, rows in the order of FEAT_PREFIX.json).  CCIP compares faces; CLIP sees the
whole picture (outfit, colours, props), which is what back views and close-ups still show.  Rows already computed for the same
crop (path + size + mtime key) are reused.
"""
import json, os, sys, time
from pathlib import Path

import numpy as np

prefix, comfy_root = Path(sys.argv[1]), Path(sys.argv[2])
items = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
root = Path(json.loads(prefix.with_name(prefix.name + ".root").read_text(encoding="utf-8"))) if prefix.with_name(prefix.name + ".root").is_file() else Path(sys.argv[3])
out = prefix.with_name(prefix.name + ".clip.npy")
keys = prefix.with_name(prefix.name + ".clip.keys.json")
old = {}
if out.is_file() and keys.is_file():
    E0 = np.load(out)
    old = {k: E0[r] for r, k in enumerate(json.loads(keys.read_text(encoding="utf-8")))}
want = [f"{it['rel']}|{it.get('key', '')}" for it in items]
todo = [k for k, w in enumerate(want) if w not in old]
print(f"crops {len(items)}; reuse {len(items) - len(todo)}, embed {len(todo)}", flush=True)
E = np.zeros((len(items), 1024), np.float16)
for k, w in enumerate(want):
    if w in old:
        E[k] = old[w]
if todo:
    os.chdir(comfy_root)
    sys.path.insert(0, str(comfy_root))
    sys.argv = ["x"]
    import torch  # noqa: E402
    from PIL import Image  # noqa: E402
    import comfy.clip_vision  # noqa: E402

    model = comfy.clip_vision.load(str(comfy_root / "models" / "clip_vision" / "clip_vision_h.safetensors"))
    t0 = time.time()
    B = 64
    for a in range(0, len(todo), B):
        idx = todo[a:a + B]
        ims = [torch.from_numpy(np.asarray(Image.open(root / items[k]["rel"]).convert("RGB").resize((224, 224)), np.float32) / 255) for k in idx]
        with torch.no_grad():
            e = model.encode_image(torch.stack(ims)).image_embeds.float()
        e = torch.nn.functional.normalize(e, dim=-1).cpu().numpy()
        E[idx] = e.astype(np.float16)
        if (a // B) % 20 == 0:
            print(f"{a + len(idx)}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
np.save(out, E)
keys.write_text(json.dumps(want), encoding="utf-8")
print(f"done -> {out}", flush=True)
os._exit(0)  # skip ComfyUI model teardown noise at exit
