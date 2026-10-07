"""Read actual TensorBoard scalars, with explicitly labelled text-log fallback."""
from pathlib import Path
import json,math,re,threading,tomllib
from collections import OrderedDict
from ...db import get_conn
from .log_parser import RE_TQDM,RE_TQDM_SPEED
_cache=OrderedDict()
_lock=threading.Lock()

def run_metrics(run_id:int):
    conn=get_conn()
    try:row=conn.execute('SELECT * FROM training_runs WHERE id=?',(run_id,)).fetchone()
    finally:conn.close()
    if row is None:raise ValueError('Runが見つかりません')
    cfg=json.loads(row['config_json'] or '{}');log=Path(row['log_path']) if row['log_path'] else None
    files=sorted(log.parent.rglob('events.out.tfevents.*'))[:32] if log and log.parent.is_dir() else []
    series={};times={};source='none';warnings=[]
    if len(files)>1:
        files=[max(files,key=lambda p:p.stat().st_mtime_ns)]
        warnings.append('再開前のログとは混ぜず、最新セッションを表示しています')
    for path in files:
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            with _lock:
                key=str(path.resolve())
                if key not in _cache:
                    _cache[key]=EventAccumulator(key,size_guidance={'scalars':20000})
                    while len(_cache)>8:_cache.popitem(last=False)
                accumulator=_cache[key];_cache.move_to_end(key);accumulator.Reload()
                for tag in accumulator.Tags()['scalars']:
                    if tag not in ('loss/current','loss/average') and not tag.startswith('lr/'):continue
                    events=accumulator.Scalars(tag)
                    if len(events)>=20000:warnings.append('長い履歴のため最大20000点で表示しています')
                    target=series.setdefault(tag,{})
                    for e in events:
                        if math.isfinite(e.value):
                            target[e.step]=float(e.value)
                            if tag=='loss/current' or e.step not in times:times[e.step]=e.wall_time
                source='tensorboard'
        except Exception as exc:warnings.append('TensorBoard読込: '+str(exc))
    speed={}
    if not series and log and log.is_file():
        with log.open('rb') as f:
            size=log.stat().st_size
            if size>16*1024*1024:f.seek(size-16*1024*1024);f.readline();warnings.append('ログ末尾16MBを表示しています')
            lines=f.read().decode('utf8',errors='replace').splitlines()
        for line in lines:
            m=RE_TQDM.search(line)
            if not m:continue
            step=int(m[1]);value=float(m[3])
            if math.isfinite(value):series.setdefault('loss/average',{})[step]=value
            v=RE_TQDM_SPEED.search(line)
            if v and float(v[1])>0:speed[step]=float(v[1]) if v[2]=='s/it' else 1/float(v[1])
        source='text_log' if series else 'none'
    total=int(row['total_epochs'] or 0)*int(row['steps_per_epoch'] or 0)
    # Bucketed batches can differ from the configured estimate; prefer the trainer's reported total.
    if log and log.is_file():
        with log.open('rb') as stream:
            stream.seek(max(0,log.stat().st_size-131072))
            tail=stream.read().decode('utf8',errors='replace').splitlines()
        for line in reversed(tail):
            match=RE_TQDM.search(line)
            if match:
                total=int(match[2]);break
    progress={}
    ordered=sorted(times)
    if ordered:
        first=times[ordered[0]]
        for i,step in enumerate(ordered):
            if total:progress[max(0,times[step]-first)]=min(100,step*100/total)
            if i and step>ordered[i-1]:
                dt=times[step]-times[ordered[i-1]]
                if dt>0:speed[step]=dt/(step-ordered[i-1])
    series['speed/seconds']=speed;series['progress/percent']=progress
    result={k:[[x,v] for x,v in sorted(points.items())] for k,points in series.items() if points}
    signature='|'.join([str(row['updated_at'])]+[str(p.stat().st_mtime_ns)+':'+str(p.stat().st_size) for p in files]+([str(log.stat().st_mtime_ns)] if log and log.is_file() else []))
    parameters={k:cfg.get(k) for k in ('base_checkpoint_path','model_family','epochs','repeats','rank','alpha','resolution','learning_rate','train_batch_size','optimizer','scheduler','training_memory_mode','preview_base_checkpoint_path')}
    from ..advanced import FIELDS
    advanced_parameters={FIELDS[k]['label']:v for k,v in cfg.get('advanced',{}).items() if k in FIELDS}
    from ..learning_rates import resolve_learning_rates
    rates=resolve_learning_rates(cfg)
    parameter_source='requested_config'
    config_path=log.parent.parent/'config.toml' if log else None
    if config_path and config_path.is_file():
        try:
            actual=tomllib.loads(config_path.read_text(encoding='utf8'))
            for target,key in {'base_checkpoint_path':'pretrained_model_name_or_path','epochs':'max_train_epochs','rank':'network_dim','alpha':'network_alpha','resolution':'resolution','learning_rate':'learning_rate','train_batch_size':'train_batch_size','optimizer':'optimizer_type','scheduler':'lr_scheduler'}.items():
                if key in actual:parameters[target]=actual[key]
            advanced_parameters={FIELDS[k]['label']:v for k,v in actual.items() if k in FIELDS}
            rates=resolve_learning_rates(actual)
            parameter_source='trainer_config'
        except (OSError,ValueError):pass
    parameters['body_learning_rate']=rates['body']
    parameters['requested_learning_rate']=rates['requested']
    lr_points=result.get('lr/unet',[])
    parameters['observed_initial_learning_rate']=lr_points[0][1] if lr_points else None
    parameters['observed_latest_learning_rate']=lr_points[-1][1] if lr_points else None
    if rates['conflict']:
        warnings.append(f"このRunの基本学習率は{rates['requested']:g}ですが、旧上書き設定により本体の開始値は{rates['body']:g}です")
    return {'run_id':run_id,'source':source,'series':result,'total_steps':total,'parameters':parameters,'parameter_source':parameter_source,'advanced_parameters':advanced_parameters,'warnings':list(dict.fromkeys(warnings)),'fingerprint':signature,'speed_source':'TensorBoard記録間隔' if times else 'ログ表示速度','time_origin':'最初の記録済みstep（モデル読込時間は含まない）' if times else None}


from functools import lru_cache
@lru_cache(maxsize=512)
def _metadata_step(path: str, size: int, modified: int):
    from safetensors import safe_open
    try:
        with safe_open(path,framework='np') as f:
            value=(f.metadata() or {}).get('ss_steps')
            return int(value) if value is not None and int(value)>=0 else None
    except Exception:return None

def checkpoint_progress(path: str, recorded):
    if recorded is not None and int(recorded)>0:return int(recorded),'recorded'
    try:
        stat=Path(path).stat();value=_metadata_step(path,stat.st_size,stat.st_mtime_ns)
    except OSError:value=None
    return (value,'checkpoint_metadata') if value is not None else (None,'unknown')
