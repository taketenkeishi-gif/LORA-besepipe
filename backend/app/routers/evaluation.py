"""Frozen evaluation reference, kept out of newly sealed training snapshots."""
from __future__ import annotations
import hashlib,json,secrets,shutil,re
from pathlib import Path
from fastapi import APIRouter,HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from ..db import get_conn
from .dataset_files import root_for,image_path,EXTENSIONS,sidecar
router=APIRouter(prefix='/evaluation',tags=['evaluation'])

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def read_reference(project_id,conn=None):
    own=conn is None;conn=conn or get_conn()
    try:
        row=conn.execute('SELECT value FROM app_settings WHERE key=?',(f'evaluation_reference:{project_id}',)).fetchone()
        return json.loads(row['value']) if row else None
    finally:
        if own:conn.close()

def store(project_id,value):
    conn=get_conn()
    try:
        conn.execute('INSERT INTO app_settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value',(f'evaluation_reference:{project_id}',json.dumps(value,ensure_ascii=False)));conn.commit()
    finally:conn.close()

@router.get('/{project_id}')
def state(project_id:int,run_id:int|None=None):
    root_for(project_id)
    conn=get_conn()
    try:
        row=conn.execute('SELECT config_json FROM training_runs WHERE id=? AND project_id=?',(run_id,project_id)).fetchone() if run_id else None
        frozen=json.loads(row['config_json']).get('evaluation_reference') if row else None
    finally:conn.close()
    return {'reference':read_reference(project_id),'run_reference':frozen}

class Choice(BaseModel):
    relative:str|None=None

@router.post('/{project_id}')
def choose(project_id:int,payload:Choice):
    root=root_for(project_id)
    if payload.relative is None:
        candidates=[p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS and not any(part.startswith('.') for part in p.relative_to(root).parts) and p.resolve().is_relative_to(root)]
        if len(candidates)<2:raise HTTPException(400,'評価用と学習用を分けるには2枚以上の画像が必要です')
        current=read_reference(project_id)
        if current:
            alternatives=[p for p in candidates if p.relative_to(root).as_posix()!=current.get('source_relative')]
            secrets.SystemRandom().shuffle(alternatives)
            different=next((p for p in alternatives if digest(p)!=current.get('sha256')),None)
            if different is None:raise HTTPException(400,'別の内容の画像がありません。現在の選択を保持します')
            candidates=[different]
        path=secrets.choice(candidates)
    else:path=image_path(root,payload.relative)
    sha=digest(path);directory=root/'.evaluation-reference';directory.mkdir(exist_ok=True);target=directory/(sha+path.suffix.lower())
    if not target.exists():shutil.copyfile(path,target)
    if digest(target)!=sha:raise HTTPException(409,'評価画像の内容が変わりました。再選択してください')
    caption,revision=sidecar(path)
    value={'sha256':sha,'source_relative':path.relative_to(root).as_posix(),'reference_relative':target.relative_to(root).as_posix(),'name':path.name,'caption':caption,'prompt':None,'profile_snapshot_id':None,'image_url':f'/api/evaluation/{project_id}/image/{sha}'}
    previous=read_reference(project_id)
    # Rebuild only the explicitly selected current training set; preserve old Runs/snapshots.
    conn=get_conn()
    try:
        row=conn.execute('SELECT training_config_json FROM projects WHERE id=?',(project_id,)).fetchone();draft=json.loads(row['training_config_json'] or '{}');current_snapshot=draft.get('dataset_snapshot_id')
        snapshot=previous.get('training_source_snapshot_id') if previous and previous.get('derived_snapshot_id')==current_snapshot else current_snapshot
        snapshot=snapshot or current_snapshot
        entries=conn.execute('SELECT a.id,a.file_path FROM basepipe_snapshot_entries e JOIN basepipe_assets a ON a.id=e.asset_id WHERE e.snapshot_id=?',(snapshot,)).fetchall() if snapshot else []
        kept=[int(e['id']) for e in entries if digest(e['file_path'])!=sha]
    finally:conn.close()
    value.update(training_source_snapshot_id=snapshot,derived_snapshot_id=current_snapshot)
    store(project_id,value)
    prepared=None
    if entries and (len(kept)!=len(entries) or snapshot!=current_snapshot):
        if not kept:
            conn=get_conn();draft['dataset_snapshot_id']=None;value['derived_snapshot_id']=None;conn.execute('UPDATE projects SET training_config_json=? WHERE id=?',(json.dumps(draft),project_id));conn.commit();conn.close()
        else:
            from .basepipe import create_snapshot
            from ..schemas import BasepipeSnapshotCreate
            prepared=create_snapshot(project_id,BasepipeSnapshotCreate(name='評価画像を除いた学習対象',asset_ids=kept))
            conn=get_conn();row=conn.execute('SELECT training_config_json FROM projects WHERE id=?',(project_id,)).fetchone();draft=json.loads(row['training_config_json'] or '{}');draft['dataset_snapshot_id']=prepared['id'];value['derived_snapshot_id']=prepared['id'];conn.execute('UPDATE projects SET training_config_json=? WHERE id=?',(json.dumps(draft,ensure_ascii=False),project_id));conn.commit();conn.close()
    store(project_id,value)
    return {'reference':value,'prepared':prepared,'excluded_from_current_set':len(entries)-len(kept)}

@router.delete('/{project_id}')
def clear(project_id:int):
    root_for(project_id);conn=get_conn()
    try:
        reference=read_reference(project_id,conn)
        row=conn.execute('SELECT training_config_json FROM projects WHERE id=?',(project_id,)).fetchone()
        draft=json.loads(row['training_config_json'] or '{}')
        if reference and draft.get('dataset_snapshot_id')==reference.get('derived_snapshot_id'):
            draft['dataset_snapshot_id']=reference.get('training_source_snapshot_id')
            conn.execute('UPDATE projects SET training_config_json=? WHERE id=?',(json.dumps(draft,ensure_ascii=False),project_id))
        conn.execute('DELETE FROM app_settings WHERE key=?',(f'evaluation_reference:{project_id}',));conn.commit()
        return {'reference':None,'dataset_snapshot_id':draft.get('dataset_snapshot_id')}
    finally:conn.close()


@router.get('/{project_id}/image/{sha}')
def image(project_id:int,sha:str):
    if not re.fullmatch('[0-9a-f]{64}',sha):raise HTTPException(400,'画像IDが不正です')
    directory=root_for(project_id)/'.evaluation-reference';paths=[p for p in directory.glob(sha+'.*') if p.suffix.lower() in EXTENSIONS]
    if not paths:raise HTTPException(404,'評価画像がありません')
    return FileResponse(paths[0])

class Tags(BaseModel):
    job_id:str
    model_family:str='anima'

@router.post('/{project_id}/use-tags')
def use_tags(project_id:int,payload:Tags):
    from ..dataset_tagging import status
    from .preview_profiles import create_profile,PreviewProfileIn
    from .basepipe import freeze_preview_profile
    value=read_reference(project_id)
    if not value:raise HTTPException(400,'先に評価画像を選択してください')
    job=status({'project_id':project_id})
    if job.get('id')!=payload.job_id or job.get('status') not in ('completed','done'):raise HTTPException(409,'指定したタグ生成が完了していません')
    result=next((r for r in job.get('results',[]) if r['relative']==value['reference_relative']),None)
    original=next((r for r in job.get('inputs',[]) if r['relative']==value['reference_relative']),None)
    if not result or not original or original.get('sha256')!=value['sha256']:raise HTTPException(409,'タグと評価画像の対応を確認できません')
    conn=get_conn()
    try:triggers=[r['trigger_token'] for r in conn.execute('SELECT trigger_token FROM basepipe_concepts WHERE project_id=?',(project_id,)) if r['trigger_token']]
    finally:conn.close()
    tags=list(dict.fromkeys(triggers+[t.strip() for t in result['caption'].split(',') if t.strip()]));prompt=', '.join(tags)
    profile=create_profile(PreviewProfileIn(project_id=project_id,name='評価画像・'+value['name'],prompt=prompt,negative_prompt='low quality, blurry',seed=42,resolution=1024,steps=20,cfg=5,sampler='euler',scheduler='simple',model_family=payload.model_family))
    frozen=freeze_preview_profile(project_id,profile['profile']['id']);value.update(prompt=prompt,profile_snapshot_id=frozen['id'],tag_model=job.get('model',{}).get('name'));store(project_id,value)
    return {'reference':value,'profile':frozen}
