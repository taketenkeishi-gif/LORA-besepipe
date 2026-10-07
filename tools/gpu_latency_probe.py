"""GPU responsiveness probe: time a tiny kernel on one GPU every 50 ms (what a compositor frame also has to wait for).

python gpu_latency_probe.py GPU_UUID SECONDS OUT.json
Writes per-sample [unix_time, latency_ms]. Also samples dwm.exe 3D-engine GPU use (Windows counters) once per second.
"""
import json, os, subprocess, sys, threading, time

uuid, secs, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
os.environ.update({"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": uuid})
import torch

x = torch.randn(256, 256, device="cuda")
torch.cuda.synchronize()
lat, dwm, stop = [], [], threading.Event()


def dwm_sampler():
    ps = ("$p=(Get-Process dwm).Id; (Get-Counter \"\\GPU Engine(pid_${p}*engtype_3D)\\Utilization Percentage\" -ErrorAction SilentlyContinue)"
          ".CounterSamples | Measure-Object CookedValue -Sum | ForEach-Object Sum")
    while not stop.is_set():
        try:
            v = subprocess.check_output(["powershell", "-NoProfile", "-Command", ps], text=True, timeout=10).strip()
            dwm.append([time.time(), float(v or 0)])
        except Exception:
            pass


th = threading.Thread(target=dwm_sampler, daemon=True); th.start()
end = time.time() + secs
while time.time() < end:
    t = time.perf_counter()
    (x @ x).sum().item()  # forces a GPU round trip
    lat.append([time.time(), round((time.perf_counter() - t) * 1000, 2)])
    time.sleep(0.05)
stop.set()
json.dump({"latency_ms": lat, "dwm_3d_percent": dwm}, open(out, "w"))
