"""Preview generation load test: run a saved preview graph N times on a ComfyUI instance and sample that GPU.

python bench_preview_gpu.py GRAPH.json COMFY_URL GPU_UUID N OUT.json
Reports seconds per image (queue -> done) and, from nvidia-smi every 0.5 s, the GPU's utilisation and memory while generating.
"""
import json, subprocess, sys, threading, time, urllib.request, uuid
from pathlib import Path

graph_path, url, gpu, n, out = sys.argv[1], sys.argv[2].rstrip("/"), sys.argv[3], int(sys.argv[4]), Path(sys.argv[5])
graph = json.loads(Path(graph_path).read_text(encoding="utf-8"))
samples, stop = [], threading.Event()


def sampler():
    while not stop.is_set():
        try:
            line = subprocess.check_output(["nvidia-smi", f"--id={gpu}", "--query-gpu=utilization.gpu,memory.used,power.draw", "--format=csv,noheader,nounits"], text=True, timeout=5)
            u, m, p = [float(x) for x in line.strip().split(",")]
            samples.append({"t": time.time(), "util": u, "mem_mib": m, "power_w": p})
        except Exception:
            pass
        stop.wait(0.5)


def post(path, body):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=30).read())


def get(path):
    return json.loads(urllib.request.urlopen(url + path, timeout=30).read())


th = threading.Thread(target=sampler, daemon=True); th.start()
time.sleep(5)  # idle baseline
t_start = time.time()
times = []
for i in range(n):
    for node in graph.values():
        if "noise_seed" in node["inputs"]:
            node["inputs"]["noise_seed"] = 1000 + i
    t0 = time.time()
    pid = post("/prompt", {"prompt": graph, "client_id": uuid.uuid4().hex})["prompt_id"]
    while True:
        h = get(f"/history/{pid}")
        if pid in h and h[pid].get("status", {}).get("completed") is not None:
            st = h[pid]["status"]
            break
        time.sleep(0.25)
    times.append({"i": i, "s": round(time.time() - t0, 2), "ok": st.get("status_str") == "success"})
    print(times[-1], flush=True)
t_end = time.time()
time.sleep(5)
stop.set(); th.join()
busy = [s for s in samples if t_start <= s["t"] <= t_end]
idle = [s for s in samples if s["t"] < t_start]
res = {"per_image_s": times, "first_s": times[0]["s"] if times else None,
       "steady_median_s": sorted(x["s"] for x in times[1:])[len(times[1:]) // 2] if len(times) > 1 else None,
       "gpu_idle": {"util_mean": round(sum(s["util"] for s in idle) / max(1, len(idle)), 1), "mem_mib": max((s["mem_mib"] for s in idle), default=None)},
       "gpu_busy": {"util_mean": round(sum(s["util"] for s in busy) / max(1, len(busy)), 1), "util_p90": sorted(s["util"] for s in busy)[int(0.9 * len(busy))] if busy else None,
                    "mem_peak_mib": max((s["mem_mib"] for s in busy), default=None), "power_mean_w": round(sum(s["power_w"] for s in busy) / max(1, len(busy)), 1)},
       "window": [t_start, t_end]}
out.write_text(json.dumps(res, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in res.items() if k != "per_image_s"}, indent=1))
