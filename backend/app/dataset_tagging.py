"""Persistent CPU tagging previews. Only the existing caption-save route writes TXT."""
from __future__ import annotations
import csv,json,os,subprocess,sys,time,uuid
from pathlib import Path
import numpy as np
import psutil
from PIL import Image,ImageOps
from . import dataset_desktop as desktop
from .routers.dataset_files import database_connection,root_for,image_path,sidecar,atomic_text
from .desktop_jobs import write_state

def models():
    with database_connection() as conn:
        row=conn.execute("SELECT value FROM app_settings WHERE key='comfyui_root'").fetchone()
    roots=[Path(row['value'])] if row and row['value'] else [Path.home()/'AI_tools/ComfyUI-Portable/ComfyUI_windows_portable/ComfyUI']
    result=[];seen=set()
    for root in roots:
        for directory in (root/'custom_nodes/comfyui-wd14-tagger/models',root/'models/wd14_tagger'):
            for path in sorted(directory.glob('*.onnx')):
                path=path.resolve();csv_path=path.with_suffix('.csv')
                if csv_path.is_file() and str(path).casefold() not in seen:
                    seen.add(str(path).casefold());result.append({'id':str(path),'name':path.stem,'bytes':path.stat().st_size})
    return {'models':result,'provider':'CPUExecutionProvider','downloads':False}

def directory(project_id):
    root_for(project_id)
    target=desktop.ROOT/'.runtime/tagging'/str(project_id);target.mkdir(parents=True,exist_ok=True)
    return target

def status(payload):
    target=directory(int(payload['project_id']))/'job.json'
    if not target.exists():return {'status':'idle'}
    job=json.loads(target.read_text(encoding='utf8'))
    if job['status'] in ('queued','running'):
        if job.get('pid'):
            try:alive=abs(psutil.Process(job['pid']).create_time()-job['process_created'])<1
            except psutil.Error:alive=False
        else:alive=time.time()-job['created']<30
        if not alive:
            job.update(status='interrupted',message='処理が中断されました。再生成できます。');atomic_text(target,json.dumps(job,ensure_ascii=False))
    return job

def start(payload):
    project_id=int(payload['project_id']);target=directory(project_id);lock=target/'start.lock'
    try:fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:raise ValueError('タグ生成の開始処理中です。少し待って再試行してください')
    try:
        os.close(fd)
        if status(payload)['status'] in ('queued','running'):raise ValueError('このプロジェクトのタグ生成は実行中です')
        available=models()['models'];model=next((m for m in available if m['id']==payload.get('model')),None)
        if not model:raise ValueError('ComfyUI側にあるタグ付けモデルを選択してください')
        threshold=float(payload.get('threshold',.35));character=float(payload.get('character_threshold',.85))
        if not (.05<=threshold<=.95 and .05<=character<=.99):raise ValueError('しきい値の範囲が不正です')
        mode=payload.get('mode','missing')
        if mode not in ('missing','append','replace'):raise ValueError('タグの保存方法が不正です')
        root=root_for(project_id);relatives=list(dict.fromkeys(payload.get('relatives') or []))
        if not 1<=len(relatives)<=500:raise ValueError('1〜500枚の画像を選択してください')
        inputs=[]
        for rel in relatives:
            path=image_path(root,rel);caption,revision=sidecar(path)
            if mode=='missing' and caption.strip():continue
            inputs.append({'relative':rel,'path':str(path),'caption':caption,'revision':revision,'sha256':desktop.digest(path)})
        if not inputs:raise ValueError('対象に未タグ付け画像がありません。「既存タグに追加」などを選んでください')
        job={'id':uuid.uuid4().hex,'project_id':project_id,'created':time.time(),'status':'queued','message':'開始待ち','model':model,'provider':'CPUExecutionProvider','threshold':threshold,'character_threshold':character,'remove_characters':bool(payload.get('remove_characters',True)),'mode':mode,'prefix':str(payload.get('prefix','')),'blocked':str(payload.get('blocked','')),'inputs':inputs,'results':[],'errors':[],'total':len(inputs),'done':0}
        job.update(requested_count=len(relatives),requested_relatives=relatives,skipped_existing_count=len(relatives)-len(inputs),scope=payload.get('scope','selected'))
        atomic_text(target/'job.json',json.dumps(job,ensure_ascii=False))
        script=Path(__file__).resolve().parents[1]/'desktop_actions.py'
        proc=subprocess.Popen([sys.executable,'-X','utf8',str(script),'tag-worker'],stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,cwd=script.parent,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),env=os.environ.copy())
        proc.stdin.write(json.dumps({'project_id':project_id,'job_id':job['id']}).encode());proc.stdin.close()
        return job
    finally:lock.unlink(missing_ok=True)

def stop(payload):
    job=status(payload)
    if job['status'] in ('queued','running'):(directory(int(payload['project_id']))/(job['id']+'.cancel')).touch()
    return {'requested':job['status'] in ('queued','running')}

def _tags(text):return [t.strip() for t in text.replace('\n',',').split(',') if t.strip()]

def load_predictor(model_path):
    import onnxruntime as ort
    options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(model_path),sess_options=options,providers=['CPUExecutionProvider'])
    with Path(model_path).with_suffix('.csv').open(encoding='utf-8-sig',newline='') as stream:tags=list(csv.DictReader(stream))
    return session,tags

def predict_caption(image,session,tags,options,previous='',return_tags=False):
    size=int(session.get_inputs()[0].shape[1]);img=ImageOps.exif_transpose(image).convert('RGBA')
    white=Image.new('RGB',img.size,'white');white.paste(img,mask=img.getchannel('A'))
    padded=ImageOps.pad(white,(size,size),method=Image.Resampling.LANCZOS,color='white',centering=(.5,.5))
    array=np.asarray(padded,dtype=np.float32)[:,:,::-1][None,...];scores=session.run(None,{session.get_inputs()[0].name:array})[0][0]
    predicted=[]
    for score,tag in zip(scores,tags):
        category=int(tag['category'])
        if category not in (0,4) or (category==4 and options.get('remove_characters',True)):continue
        if float(score)>=(options.get('character_threshold',.85) if category==4 else options.get('threshold',.35)):predicted.append(tag['name'].replace('_',' '))
    blocked={t.casefold() for t in _tags(options.get('blocked',''))};seen=set();combined=[]
    for value in _tags(options.get('prefix',''))+(_tags(previous) if options.get('mode')=='append' else [])+predicted:
        key=value.casefold()
        if key not in seen and key not in blocked:seen.add(key);combined.append(value)
    return (', '.join(combined),predicted) if return_tags else ', '.join(combined)

def run(payload):
    project_id=int(payload['project_id']);target=directory(project_id);job=status(payload)
    if job.get('id')!=payload['job_id']:return
    def persist():write_state(target/'job.json',json.dumps(job,ensure_ascii=False))
    cancel=target/(job['id']+'.cancel')
    try:
        job.update(status='running',pid=os.getpid(),process_created=psutil.Process().create_time(),message='CPUでモデルを読み込み中');persist()
        session,tags=load_predictor(job['model']['id']);job['providers']=session.get_providers();deadline=time.monotonic()+1800
        for item in job['inputs']:
            if cancel.exists():job.update(status='stopped',message='停止しました。生成済みの分は確認・保存できます。');break
            if time.monotonic()>deadline:raise ValueError('30分の処理上限に達しました。枚数を分けて再実行してください')
            try:
                path=image_path(root_for(project_id),item['relative'])
                if desktop.digest(path)!=item['sha256']:raise ValueError('生成前に画像が変更されました')
                with Image.open(path) as raw:caption,predicted=predict_caption(raw,session,tags,job,item['caption'],return_tags=True)
                if desktop.digest(path)!=item['sha256']:raise ValueError('生成中に画像が変更されました')
                job['results'].append({'relative':item['relative'],'caption':caption,'revision':item['revision'],'previous':item['caption'],'raw_tags':predicted,'tag_count':len(_tags(caption))})
            except Exception as error:job['errors'].append({'relative':item['relative'],'message':str(error)})
            job['done']+=1;job['message']=f"{job['done']} / {job['total']}枚を生成";persist()
        else:job.update(status='done',message='生成が終わりました。内容を確認してTXTへ保存してください。')
        persist()
    except Exception as error:job.update(status='failed',message=str(error));persist()
    finally:cancel.unlink(missing_ok=True)
    return {'status':job['status']}
