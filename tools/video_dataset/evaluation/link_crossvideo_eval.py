"""Cross-video identity check with real labels: which group score separates "same character in ANOTHER video" from "someone else"?

Labels come from a curated dataset (images picked by hand for one character, possibly renamed/re-encoded):
every output group of the source runs is labelled positive when >= --min-share of its images match a curated image (perceptual hash).

python link_crossvideo_eval.py EXP_ROOT CURATED_DIR OUT.json JOB [JOB ...]
Runs link_characters.py twice (with / without CCIP, --dump-score) and reports AUC over cross-job pairs only.
"""
import argparse, json, os, subprocess, sys
from pathlib import Path

import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("exp_root"); ap.add_argument("curated"); ap.add_argument("out"); ap.add_argument("jobs", nargs="+")
ap.add_argument("--min-share", type=float, default=0.3)
args = ap.parse_args()
root = Path(args.exp_root)


def phash(p: Path) -> int:
    g = np.asarray(Image.open(p).convert("L").resize((17, 16), Image.BILINEAR), np.float32)
    return int("".join("1" if b else "0" for b in (g[:, 1:] > g[:, :-1]).flatten()), 2)


cur = [phash(p) for p in Path(args.curated).rglob("*.png")]
cur_arr = np.array(cur, dtype=object)
print("curated images", len(cur), flush=True)


def matched(h: int) -> bool:
    return any(bin(h ^ c).count("1") <= 12 for c in cur)


labels = {}
for j in args.jobs:
    for cdir in sorted((root / j).glob("char_*")):
        pngs = sorted(cdir.rglob("*.png"))
        if not pngs:
            continue
        hits = sum(matched(phash(p)) for p in pngs)
        labels[f"{j}/{cdir.name}"] = (hits / len(pngs), hits, len(pngs))
pos_ids = {k for k, (s, h, n) in labels.items() if s >= args.min_share and h >= 2}
print("positive groups", sorted(pos_ids), flush=True)

py = r"C:\Users\Keishi\Portfolio\Toolchains-Physical\ComfyUI-Portable\ComfyUI_windows_portable\python_embeded\python.exe"
tool = Path(__file__).resolve().parents[1] / "link_characters.py"
env = os.environ.copy(); env.update({"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"})
res = {"labels": {k: list(v) for k, v in labels.items()}, "positives": sorted(pos_ids), "methods": {}}
tmp = Path(os.environ.get("TEMP", ".")) / "xv_eval"
tmp.mkdir(exist_ok=True)
r = subprocess.run([py, "-X", "utf8", str(tool), str(tmp / "links.json"), str(root), *args.jobs, "--q", "0.75", "--dump-score", str(tmp / "s.npz")],
                   env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=3000)
if r.returncode:
    sys.exit(r.stderr[-800:])
d = np.load(tmp / "s.npz", allow_pickle=True)
ids = list(d["ids"]); k = len(ids)
job_of = [i.split("/")[0] for i in ids]
isp = np.array([i in pos_ids for i in ids])
iu = np.triu_indices(k, 1)


def z(M):
    v = M[iu]
    return (M - v.mean()) / (v.std() + 1e-9)


def auc(sp, sn):
    allv = np.concatenate([sp, sn]); rk = allv.argsort().argsort()[: len(sp)] + 1
    return float((rk.sum() - len(sp) * (len(sp) + 1) / 2) / (len(sp) * len(sn)))


P, N = [], []
for a, b in zip(*iu):
    if job_of[a] == job_of[b]:
        continue  # cross-video pairs only
    if isp[a] and isp[b]:
        P.append((a, b))
    elif isp[a] or isp[b]:
        N.append((a, b))
print("cross-video pairs: same", len(P), "different", len(N), flush=True)
mats = {"CLIP+tags (current)": z(d["cos"]) + z(d["tag"]), "CLIP only": d["cos"], "tags only": d["tag"]}
if np.any(d["ccip"]):
    mats["CCIP only (median of crop pairs)"] = d["ccip"]
    mats["CLIP+tags+CCIP"] = z(d["cos"]) + z(d["tag"]) + z(d["ccip"])
    mats["tags+CCIP"] = z(d["tag"]) + z(d["ccip"])
for name, M in mats.items():
    sp = np.array([M[a, b] for a, b in P]); sn = np.array([M[a, b] for a, b in N])
    # best achievable: the share of same pairs ranked above the strongest different pair
    res["methods"][name] = {"auc": round(auc(sp, sn), 4), "same_pairs_above_all_different": round(float((sp > sn.max()).mean()), 3)}
    print(name, res["methods"][name], flush=True)
Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
