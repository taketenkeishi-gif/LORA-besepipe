"""Estimate the next run from its sealed inputs and comparable measured runs."""
import json, math, statistics
from pathlib import Path
from ..db import get_conn
from .snapshot_inputs import fetch_snapshot_inputs
from .runtime.metrics import run_metrics


def estimate_draft(cfg: dict) -> dict:
    conn = get_conn()
    try:
        snapshot = cfg.get('dataset_snapshot_id')
        owned = conn.execute('SELECT id FROM basepipe_dataset_snapshots WHERE id=? AND project_id=? AND status=?', (snapshot, cfg['project_id'], 'sealed')).fetchone() if snapshot else None
        images = sum(Path(r['file_path']).is_file() for r in fetch_snapshot_inputs(conn, snapshot)) if owned else 0
        history = conn.execute("SELECT id,config_json FROM training_runs WHERE status='completed' ORDER BY id DESC LIMIT 100").fetchall()
    finally:
        conn.close()
    accum = int(cfg.get('advanced', {}).get('gradient_accumulation_steps', 1))
    batch = int(cfg['train_batch_size'])
    steps = math.ceil(math.ceil(images * int(cfg['repeats']) / batch) / accum) if images else 0
    total = steps * int(cfg['epochs'])
    result = dict(images=images, steps_per_epoch=steps, total_steps=total, eta_seconds=None, source='unavailable', reference_run_id=None, note='バケット端数により実際のstep数は変わります。時間はモデル読込・キャッシュ準備・プレビュー生成を除く目安です。')
    candidates = []
    for row in history:
        old = json.loads(row['config_json'] or '{}')
        if old.get('model_family') != cfg.get('model_family') or int(old.get('gpu_device_id', -1)) != int(cfg.get('gpu_device_id', 1)):
            continue
        categorical = {'optimizer':'AdamW8bit','training_memory_mode':'standard','mixed_precision':'bf16','gradient_checkpointing':True,'cache_latents':True,'network_train_unet_only':True}
        if any(old.get(k, default) != cfg.get(k, default) for k, default in categorical.items()):
            continue  # Do not invent a speed multiplier for a different offload/precision strategy.
        keys = ('resolution','rank','train_batch_size','optimizer','training_memory_mode','mixed_precision','gradient_checkpointing','cache_latents','network_train_unet_only')
        difference = sum(old.get(k) != cfg.get(k) for k in keys)
        candidates.append((difference, -int(row['id']), row, old))
    for difference, _, row, old in sorted(candidates)[:3]:
        metrics = run_metrics(int(row['id']))
        speeds = [v for _, v in metrics['series'].get('speed/seconds', []) if v > 0]
        if len(speeds) < 5:
            continue
        previous_accum = max(1, int(old.get('advanced', {}).get('gradient_accumulation_steps', 1)))
        ratio = (int(cfg['resolution']) / int(old.get('resolution',512))) ** 2
        ratio *= (batch / max(1,int(old.get('train_batch_size',1)))) ** .6
        ratio *= max(.5,int(cfg['rank']) / max(1,int(old.get('rank',16))))
        ratio *= accum / previous_accum
        seconds = statistics.median(speeds) * ratio
        result.update(eta_seconds=round(total*seconds), sec_per_step=round(seconds,3), reference_run_id=int(row['id']), source='history' if difference == 0 and accum == previous_accum else 'adjusted_history', samples=len(speeds))
        break
    return result
