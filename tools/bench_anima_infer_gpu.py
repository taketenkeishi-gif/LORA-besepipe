"""Anima preview load on one GPU (sd-scripts anima_minimal_inference, model loaded once, N prompts).

python bench_anima_infer_gpu.py GPU_UUID GPU_SMI_INDEX LORA OUT_DIR N
Writes OUT_DIR/result.json: seconds per image (from output-file times), and nvidia-smi samples of that GPU every 0.5 s.
"""
import json, os, subprocess, sys, threading, time
from pathlib import Path

uuid, smi_id, lora, out, n = sys.argv[1], sys.argv[2], sys.argv[3], Path(sys.argv[4]), int(sys.argv[5])
out.mkdir(parents=True, exist_ok=True)
M = Path(r"C:\Users\Keishi\Portfolio\Toolchains-Physical\ComfyUI-Portable\ComfyUI_windows_portable\ComfyUI\models")
prompt = ("EXCEEDS, ex_nanoha_uniform, shirt, white shirt, necktie, tie clip, blue necktie, uniform, black skirt, collared shirt, skirt, 1girl, solo, "
          "looking at viewer, standing, cowboy shot, brown hair, side ponytail, purple eyes, indoors")
neg = "low quality, blurry, duplicate, extra limbs, deformed hands, jacket"
(out / "prompts.txt").write_text("\n".join(f"{prompt} --n {neg} --d {1000 + i} --s 30 --g 6 --w 1024 --h 1024" for i in range(n)) + "\n", encoding="utf-8")
samples, stop = [], threading.Event()


def sampler():
    while not stop.is_set():
        try:
            u, m, p = [float(x) for x in subprocess.check_output(["nvidia-smi", f"--id={smi_id}", "--query-gpu=utilization.gpu,memory.used,power.draw",
                                                                 "--format=csv,noheader,nounits"], text=True, timeout=5).strip().split(",")]
            samples.append({"t": time.time(), "util": u, "mem_mib": m, "power_w": p})
        except Exception:
            pass
        stop.wait(0.5)


th = threading.Thread(target=sampler, daemon=True); th.start()
time.sleep(5)
env = os.environ.copy(); env.update({"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": uuid, "PYTHONIOENCODING": "utf-8"})
t0 = time.time()
r = subprocess.run([r"C:\kohya_ss\venv\Scripts\python.exe", r"C:\kohya_ss\sd-scripts\anima_minimal_inference.py",
                    "--dit", str(M / "diffusion_models" / "anima-base-v1.0.safetensors"), "--vae", str(M / "vae" / "qwen_image_vae.safetensors"),
                    "--text_encoder", str(M / "text_encoders" / "qwen_3_06b_base.safetensors"), "--lora_weight", lora, "--lora_multiplier", "1.0",
                    "--vae_disable_cache", "--qwen_image_vae_2d", "--attn_mode", "torch", "--from_file", str(out / "prompts.txt"), "--save_path", str(out / "img")],
                   cwd=r"C:\kohya_ss\sd-scripts", env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=3000)
t1 = time.time()
time.sleep(5); stop.set(); th.join()
imgs = sorted((out / "img").rglob("*.png"), key=lambda p: p.stat().st_mtime)
marks = [p.stat().st_mtime for p in imgs]
gaps = [round(b - a, 1) for a, b in zip(marks, marks[1:])]
busy = [s for s in samples if t0 <= s["t"] <= t1]
idle = [s for s in samples if s["t"] < t0]
gen = [s for s in samples if marks and marks[0] - (gaps[0] if gaps else 0) <= s["t"] <= marks[-1]]
res = {"exit": r.returncode, "images": len(imgs), "total_s": round(t1 - t0, 1), "first_image_after_s": round(marks[0] - t0, 1) if marks else None,
       "per_image_gaps_s": gaps, "window": [marks[0] - (gaps[0] if gaps else 0), marks[-1]] if marks else None,
       "gpu_idle": {"util_mean": round(sum(s["util"] for s in idle) / max(1, len(idle)), 1), "mem_mib": max((s["mem_mib"] for s in idle), default=None)},
       "gpu_generating": {"util_mean": round(sum(s["util"] for s in gen) / max(1, len(gen)), 1), "util_min": min((s["util"] for s in gen), default=None),
                          "mem_peak_mib": max((s["mem_mib"] for s in busy), default=None), "power_mean_w": round(sum(s["power_w"] for s in gen) / max(1, len(gen)), 1)},
       "tail": (r.stdout + r.stderr)[-600:] if r.returncode else ""}
(out / "result.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print(json.dumps(res, indent=1), flush=True)
