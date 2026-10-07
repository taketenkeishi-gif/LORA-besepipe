"""Scoped disposable files; never reset counters, logs, weights or source datasets."""
from pathlib import Path
from ..db import get_conn
from .runtime.state import run_dir

ACTIVE = {'queued','preparing','caching','training','running','waiting','waiting_gpu','stopping'}


def files_for(conn, run_id, kind):
    root = run_dir(run_id).resolve()
    if kind == 'training':
        candidates = list((root/'cache').rglob('*')) if (root/'cache').is_dir() else []
        candidates += list((root/'train_data').rglob('*.npz')) if (root/'train_data').is_dir() else []
    elif kind == 'weights':
        candidates=list((root/'output').glob('*.safetensors'))+list((root/'output').glob('*.ckpt'))
    elif kind == 'preview':
        candidates = [Path(r[0]) for r in conn.execute('SELECT ps.image_path FROM preview_samples ps JOIN checkpoints c ON c.id=ps.checkpoint_id WHERE c.run_id=?', (run_id,)) if r[0]]
        candidates += [Path(r[0]) for r in conn.execute('SELECT output_path FROM preview_jobs WHERE run_id=?', (run_id,)) if r[0]]
        candidates += [Path(r[0]) for r in conn.execute('SELECT h.image_path FROM preview_history h JOIN checkpoints c ON c.id=h.checkpoint_id WHERE c.run_id=?',(run_id,)) if r[0]]
        if (root/'output/sample').is_dir():candidates += [p for p in (root/'output/sample').rglob('*') if p.suffix.lower() in {'.png','.jpg','.jpeg','.webp'}]
    else:
        raise ValueError('Unknown cache kind')
    files = []
    for path in candidates:
        if not path.is_file():
            continue
        if not path.resolve().is_relative_to(root) or path.is_symlink():
            raise ValueError('Runフォルダ外を参照するファイルが含まれています。削除していません。')
        if kind == 'preview' and path.suffix.lower() not in {'.png','.jpg','.jpeg','.webp'}:
            raise ValueError('プレビュー以外のファイルが含まれています')
        files.append(path)
    return list(dict.fromkeys(files))


def inspect_cleanup(run_id):
    conn = get_conn()
    try:
        row = conn.execute('SELECT status FROM training_runs WHERE id=?',(run_id,)).fetchone()
        if row is None:raise ValueError('Runが見つかりません')
        busy = row['status'] in ACTIVE or bool(conn.execute("SELECT 1 FROM preview_jobs WHERE run_id=? AND status IN ('pending','running') LIMIT 1",(run_id,)).fetchone())
        groups = {}
        for kind in ('training','preview'):
            paths = files_for(conn,run_id,kind)
            groups[kind] = {'files':len(paths),'bytes':sum(p.stat().st_size for p in paths)}
        return {'run_id':run_id,'busy':busy,**groups}
    finally:conn.close()


def cleanup(run_id,kind):
    if kind not in ('training','preview'):raise ValueError('LoRA削除はRun一括整理で対象を選択してください')
    conn=get_conn()
    try:row=conn.execute('SELECT project_id FROM training_runs WHERE id=?',(run_id,)).fetchone()
    finally:conn.close()
    if row is None:raise ValueError('Runが見つかりません')
    result=cleanup_project_runs(row['project_id'],[run_id],[kind])
    return {**result,'deleted_bytes':result['target_bytes'],'checkpoints_preserved':True,'logs_preserved':True}


KINDS=('training','preview','weights')

def _busy(conn, row):
    from .runtime.monitor import proc_alive
    return row['status'] in ACTIVE or proc_alive(row['id']) or bool(conn.execute("SELECT 1 FROM preview_jobs WHERE run_id=? AND status IN ('pending','running') LIMIT 1",(row['id'],)).fetchone())

def _groups(conn,row):
    groups={}
    for kind in KINDS:
        paths=files_for(conn,row['id'],kind)
        groups[kind]={'files':len(paths),'bytes':sum(p.stat().st_size for p in paths)}
    return {'run_id':row['id'],'status':row['status'],'epoch':row['current_epoch'],'busy':_busy(conn,row),**groups}

def project_cleanup_inventory(project_id):
    conn=get_conn()
    try:
        rows=conn.execute('SELECT * FROM training_runs WHERE project_id=? ORDER BY id DESC',(project_id,)).fetchall()
        return {'project_id':project_id,'runs':[_groups(conn,row) for row in rows]}
    finally:conn.close()

def cleanup_project_runs(project_id,run_ids,kinds):
    ids=list(dict.fromkeys(run_ids));kinds=list(dict.fromkeys(kinds))
    if not ids or len(ids)>200 or not kinds or any(k not in KINDS for k in kinds):raise ValueError('Runと削除する種類を選択してください')
    conn=get_conn()
    try:
        conn.execute('BEGIN IMMEDIATE')
        rows=[];paths=[]
        for run_id in ids:
            row=conn.execute('SELECT * FROM training_runs WHERE id=? AND project_id=?',(run_id,project_id)).fetchone()
            if row is None:raise ValueError('別プロジェクトのRunまたは存在しないRunです')
            if _busy(conn,row):raise ValueError(f'Run #{run_id}は実行中または生成待ちです。削除していません')
            rows.append(row)
            for kind in kinds:paths.extend(files_for(conn,run_id,kind))
        if 'weights' in kinds:
            import json
            selected_roots=[run_dir(i).resolve() for i in ids]
            for row in conn.execute("SELECT config_json FROM training_runs WHERE status IN ('queued','training','running','waiting','stopping')"):
                cfg=json.loads(row['config_json'] or '{}');source=cfg.get('resume_from_checkpoint')
                if source and any(Path(source).resolve().is_relative_to(root) for root in selected_roots):raise ValueError('実行中の学習が継続元に使っています。削除していません')
            setting=conn.execute("SELECT value FROM app_settings WHERE key='comfyui_root'").fetchone()
            if setting and setting[0]:
                managed=Path(setting[0])/'models/loras/LoRA-Studio-preview'
                for row in rows:
                    for source in files_for(conn,row['id'],'weights'):
                        link=managed/f"{row['id']}-{source.stat().st_ino:x}"/source.name
                        if link.exists():
                            if link.is_symlink() or not link.resolve().is_relative_to(managed.resolve()) or not link.samefile(source):raise ValueError('共有LoRAリンクの同一性を確認できません')
                            paths.append(link)
        paths=list(dict.fromkeys(paths));unique={}
        for path in paths:
            st=path.stat();unique[(st.st_dev,st.st_ino)]=st.st_size
        # Every selected Run, path and active dependency has passed before the first unlink.
        for path in paths:path.unlink()
        for row in rows:
            run_id=row['id']
            if 'preview' in kinds:
                conn.execute('DELETE FROM preview_history WHERE checkpoint_id IN (SELECT id FROM checkpoints WHERE run_id=?)',(run_id,))
                conn.execute('DELETE FROM preview_samples WHERE checkpoint_id IN (SELECT id FROM checkpoints WHERE run_id=?)',(run_id,))
                conn.execute("UPDATE preview_jobs SET status='cancelled',output_path='',error_detail='プレビュー削除済み' WHERE run_id=?",(run_id,))
                conn.execute("UPDATE checkpoints SET validation_status='validated',validation_detail='プレビュー削除済み（LoRA保持）' WHERE run_id=? AND validation_status NOT IN ('invalid','deleted')",(run_id,))
            if 'weights' in kinds:
                conn.execute('UPDATE training_runs SET latest_checkpoint_path=NULL WHERE id=?',(run_id,))
                conn.execute("UPDATE checkpoints SET validation_status='deleted',validation_detail='ユーザーが学習済みLoRAを削除しました' WHERE run_id=?",(run_id,))
        if 'weights' in kinds:
            import json
            row=conn.execute('SELECT training_config_json FROM projects WHERE id=?',(project_id,)).fetchone()
            cfg=json.loads(row[0] or '{}')
            selected=cfg.get('resume_checkpoint_id')
            if selected and conn.execute("SELECT 1 FROM checkpoints WHERE id=? AND validation_status='deleted'",(selected,)).fetchone():
                cfg['resume_checkpoint_id']=None;conn.execute('UPDATE projects SET training_config_json=? WHERE id=?',(json.dumps(cfg,ensure_ascii=False),project_id))
        conn.commit()
        return {'run_ids':ids,'kinds':kinds,'deleted_files':len(paths),'target_bytes':sum(unique.values()),'source_dataset_preserved':True,'logs_preserved':True}
    finally:conn.close()

