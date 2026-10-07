"""Short training benchmark (sd-scripts Anima): seconds per step and peak VRAM for config variants.

python bench_train_speed.py RUN_DIR OUT.json [STEPS]
Copies RUN_DIR/config.toml to a temp dir per variant (outputs never touch the run), runs STEPS steps on the 3090 Ti,
reads the steady-state s/it from the progress bar and the peak VRAM from torch.
"""
import json, os, re, subprocess, sys, tempfile, time
from pathlib import Path

run, out_json = Path(sys.argv[1]), Path(sys.argv[2])
steps = int(sys.argv[3]) if len(sys.argv) > 3 else 40
PY = r"C:\kohya_ss\venv\Scripts\python.exe"
SCRIPT = r"C:\kohya_ss\sd-scripts\anima_train_network.py"
base = (run / "config.toml").read_text(encoding="utf-8")
VARIANTS = {
    "current (checkpointing on, swap 12)": {},
    "checkpointing off, swap 12": {"gradient_checkpointing": "false"},
    "checkpointing off, swap 0": {"gradient_checkpointing": "false", "blocks_to_swap": "0"},
    "checkpointing on, swap 0": {"blocks_to_swap": "0"},
    "checkpointing on, swap 26": {"blocks_to_swap": "26"},
}
if os.environ.get("ONLY"):
    VARIANTS = {k: v for k, v in VARIANTS.items() if os.environ["ONLY"] in k}
ENTRY = r'''
import sys, runpy, json, torch
sys.path.insert(0, r"C:\kohya_ss\sd-scripts")
sys.argv = [r"{script}", "--config_file", r"{cfg}"]
try:
    runpy.run_path(r"{script}", run_name="__main__")
finally:
    print("VRAM_RESULT " + json.dumps({{"peak_reserved": torch.cuda.max_memory_reserved(0)}}), flush=True)
'''
results = {}
for name, edits in VARIANTS.items():
    tmp = Path(tempfile.mkdtemp(prefix="trainbench_"))
    cfg = base
    for k, v in {**edits, "output_dir": json.dumps(str(tmp / "out")), "logging_dir": json.dumps(str(tmp / "logs")), "save_every_n_epochs": "999"}.items():
        cfg, n = re.subn(rf"(?m)^{k} = .*$", lambda _m, k=k, v=v: f"{k} = {v}", cfg)  # function: keep Windows-path backslashes literal
        if not n:
            cfg += f"\n{k} = {v}"
    cfg = re.sub(r"(?m)^max_train_epochs = .*$", f"max_train_steps = {steps}", cfg)
    (tmp / "config.toml").write_text(cfg, encoding="utf-8")
    (tmp / "entry.py").write_text(ENTRY.format(script=SCRIPT, cfg=tmp / "config.toml"), encoding="utf-8")
    env = os.environ.copy(); env.update({"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "1", "PYTHONIOENCODING": "utf-8"})
    t0 = time.time()
    r = subprocess.run([PY, str(tmp / "entry.py")], env=env, cwd=r"C:\kohya_ss\sd-scripts", capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=3600)
    log = (r.stdout or "") + (r.stderr or "")
    # only the training progress bar ("steps: ..."); caching bars also print it/s
    bar = [seg for seg in re.split(r"[\r\n]", log) if "steps:" in seg and f"/{steps}" in seg]
    rate = None
    for seg in reversed(bar):
        m = re.search(r"([0-9.]+)(s/it|it/s)", seg)
        if m:
            rate = float(m.group(1)) if m.group(2) == "s/it" else 1 / float(m.group(1))
            break
    sp = rate
    vram = re.search(r"VRAM_RESULT (\{.*\})", log)
    oom = "out of memory" in log.lower()
    results[name] = {"exit": r.returncode, "s_per_step": sp, "peak_reserved_gb": round(json.loads(vram.group(1))["peak_reserved"] / 1024**3, 2) if vram else None,
                     "oom": oom, "wall_s": round(time.time() - t0), "tail": log[-300:] if r.returncode else ""}
    print(name, {k: v for k, v in results[name].items() if k != "tail"}, flush=True)
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
