"""File-backed preprocessing jobs: resize, optional real upscale, output-only tags."""
from __future__ import annotations
import base64,io,json,os,shutil,subprocess,sys,time,uuid
from pathlib import Path
from filelock import FileLock
import psutil
from PIL import Image
from . import dataset_desktop as images,dataset_tagging as tagging,desktop_comfy as comfy
from .desktop_jobs import write_state
from .routers.dataset_files import root_for,sidecar
from .services.comfyui_client import ComfyUIClient

def _directory(project_id):
    root_for(project_id);target=images.ROOT/'.runtime/preprocessing'/str(project_id);target.mkdir(parents=True,exist_ok=True);return target

def capabilities(payload):
    result={'tag_models':tagging.models()['models'],'upscale_models':[],'comfy_ready':False,'text_removal_available':False,'text_removal_reason':'文字除去は未接続','masking_available':False,'masking_reason':'特徴マスキングは旧版から処理未実装'}
    try:
        catalog=comfy.inspect({'include_schema':True});info=catalog['object_info']
        result['comfy_ready']=any('RTX 3090 Ti' in d.get('name','') for d in catalog['devices'])
        result['url']=catalog['url']
        if result['comfy_ready'] and all(n in info for n in ['ETN_LoadImageBase64','UpscaleModelLoader','ImageUpscaleWithModel','SaveImageWebsocket']):result['upscale_models']=ComfyUIClient._extract_combo(info,'UpscaleModelLoader','model_name')
        lama=ComfyUIClient._extract_combo(info,'INPAINT_LoadInpaintModel','model_name')
        # The legacy screen advertised CLIPSeg/LaMa but ran a different Qwen cleanup.
        # Do not expose that generative rewrite as verified text-only removal.
        result['text_removal_reason']='文字除去：LaMaモデル未導入' if not lama else '文字除去：旧UIと実処理の不一致があり、専用経路は未検証'
        if not result['comfy_ready']:result['comfy_error']='許可されたGPU1のComfyUIへ接続してください'
    except Exception as error:result['comfy_error']=str(error);result['text_removal_reason']='文字除去：ComfyUI未接続'
    return result

def status(payload):
    directory=_directory(int(payload['project_id']));path=directory/'latest.json'
    if not path.exists():return {'status':'idle'}
    job=json.loads(path.read_text(encoding='utf8'))
    if job['status'] in ('starting','running') and time.time()-job['created']>30:
        try:alive=abs(psutil.Process(job['pid']).create_time()-job['process_created'])<1
        except (psutil.Error,KeyError):alive=False
        if not alive:job.update(status='interrupted',message='前処理が中断されました。原本は変更していません。処理記録を確認してください。')
    if job.get('results'):
        path=Path(job['results'][0]['path'])
        if path.is_file():
            with Image.open(path) as image:job['after_preview']=images._thumbnail(image)
    return job

def start(payload):
    project_id=int(payload['project_id']);directory=_directory(project_id)
    with FileLock(str(directory/'operation.lock'),timeout=0):
        if status(payload)['status'] in ('starting','running'):raise ValueError('このプロジェクトの前処理は実行中です')
        mode,width,height,upscale,alignment=images._settings(payload);root,inputs=images._inputs(payload)
        expected={i['relative']:i for i in payload.get('expected',[])}
        if len(expected)!=len(inputs):raise ValueError('寸法のプレビューを確認してください')
        prepared=[]
        for relative,path,caption,revision in inputs:
            sha=images.digest(path)
            if expected.get(relative,{}).get('sha256')!=sha or expected[relative].get('caption_revision')!=revision:raise ValueError('画像またはTXTが変わりました。プレビューを更新してください')
            prepared.append({'relative':relative,'path':str(path),'sha256':sha,'caption':caption,'caption_revision':revision})
        caps=capabilities(payload) if payload.get('use_ai') else {'tag_models':tagging.models()['models']}
        if payload.get('use_ai') and (not upscale or payload.get('upscale_model') not in caps.get('upscale_models',[])):raise ValueError('AI拡大には拡大許可と接続先の拡大モデルを指定してください')
        tag_options=payload.get('tag_options') or {};tag_model=payload.get('tag_model')
        if payload.get('use_caption'):
            if tag_model not in {m['id'] for m in caps['tag_models']}:raise ValueError('既存のタグ付けモデルを選択してください')
            if not (.05<=float(tag_options.get('threshold',.35))<=.95 and .05<=float(tag_options.get('character_threshold',.85))<=.99):raise ValueError('タグしきい値の範囲が不正です')
        token=uuid.uuid4().hex
        job={'id':token,'project_id':project_id,'created':time.time(),'status':'starting','message':'前処理を開始しています','settings':{k:v for k,v in payload.items() if k not in ('expected','relatives')},'inputs':prepared,'total':len(prepared),'done':0,'failed':0,'results':[],'errors':[],'native_jobs':[],'folder':'processed-'+token[:10],'before_preview':images._thumbnail(images._load(inputs[0][1])),'comfy_url':caps.get('url')}
        write_state(directory/'latest.json',json.dumps(job,ensure_ascii=False))
        script=Path(__file__).resolve().parents[1]/'desktop_actions.py'
        with (directory/(token+'.log')).open('ab') as log:
            proc=subprocess.Popen([sys.executable,'-X','utf8',str(script),'preprocess-worker'],stdin=subprocess.PIPE,stdout=log,stderr=log,cwd=script.parent,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),env=os.environ.copy())
        proc.stdin.write(json.dumps({'project_id':project_id,'job_id':token}).encode());proc.stdin.close();return job

def cancel(payload):
    job=status(payload)
    if job['status'] in ('starting','running'):(_directory(int(payload['project_id']))/(job['id']+'.cancel')).touch()
    return {'requested':job['status'] in ('starting','running')}

def run(payload):
    project_id=int(payload['project_id']);directory=_directory(project_id);job=json.loads((directory/'latest.json').read_text(encoding='utf8'))
    if job['id']!=payload['job_id']:return
    config=job['settings'];root=root_for(project_id);stage=root/('.preprocess-'+job['id']);stop=directory/(job['id']+'.cancel')
    def persist():write_state(directory/'latest.json',json.dumps(job,ensure_ascii=False))
    try:
        job.update(status='running',pid=os.getpid(),process_created=psutil.Process().create_time(),message='処理を準備中');persist();stage.mkdir(exist_ok=False)
        mode,width,height,upscale,alignment=images._settings(config);predictor=None
        if config.get('use_caption'):
            job['message']='CPUタグモデルを読み込み中';persist();predictor=tagging.load_predictor(config['tag_model'])
        used=set();deadline=time.monotonic()+1800;cancelled=False
        for item in job['inputs']:
            if stop.exists() or time.monotonic()>deadline:cancelled=True;break
            output=None;native_path=None
            try:
                source=images.image_path(root,item['relative'])
                if images.digest(source)!=item['sha256'] or sidecar(source)[1]!=item['caption_revision']:raise ValueError('開始後に元画像またはTXTが変わりました')
                image=images._load(source);source_size=list(image.size);_,expected_size,scale=images._geometry(image.size,mode,width,height,upscale,alignment)
                job['message']=f"{job['done']+1}/{job['total']}：{source.name} を処理中";persist()
                ai_applied=False
                if config.get('use_ai') and scale>1:
                    stream=io.BytesIO();white=Image.new('RGB',image.size,'white');white.paste(image,mask=image.getchannel('A'));white.save(stream,format='PNG')
                    graph=ComfyUIClient.build_upscale_graph('',config['upscale_model']);graph['1']={'class_type':'ETN_LoadImageBase64','inputs':{'image':base64.b64encode(stream.getvalue()).decode()}};graph['4']={'class_type':'SaveImageWebsocket','inputs':{'images':['3',0]}}
                    graph_path=directory/f"{job['id']}-{job['done']}.json";write_state(graph_path,json.dumps(graph))
                    native_path=directory/f"{job['id']}-{job['done']}.png"
                    native={'url':job['comfy_url'],'graph_path':str(graph_path),'output_nodes':['4'],'expected_artifact':str(native_path),'collection_required':True,'collection_state':'PENDING','status':'starting'}
                    def progress():
                        job['message']=f"{job['done']+1}/{job['total']} AI拡大：{native.get('message','送信中')}";job['current_prompt_id']=native.get('prompt_id');persist()
                    comfy.collect_graph(native,progress);job['native_jobs'].append({k:v for k,v in native.items() if k!='graph'})
                    if native['status']!='done':raise ValueError(native.get('message','AI拡大に失敗しました'))
                    image=Image.open(native_path).convert('RGBA');ai_applied=True
                # Use the original computed output geometry even after a 2x/4x model pass.
                if not ai_applied:result=images._render(image,mode,width,height,upscale,alignment)
                elif mode=='crop':result=images._render(image,'crop',expected_size[0],expected_size[1],True,1)
                elif mode=='pad':result=images._render(image,'pad',expected_size[0],expected_size[1],True,1)
                else:result=image.convert('RGB').resize(expected_size,Image.Resampling.LANCZOS)
                name=source.stem+'.png';suffix=2
                while name.casefold() in used:name=f'{source.stem}_{suffix}.png';suffix+=1
                used.add(name.casefold());output=stage/name;result.save(output)
                if predictor:
                    text=tagging.predict_caption(result,*predictor,config.get('tag_options') or {'mode':'append'},item['caption']);output.with_suffix('.txt').write_text(text,encoding='utf8',newline='')
                elif source.with_suffix('.txt').exists():shutil.copyfile(source.with_suffix('.txt'),output.with_suffix('.txt'))
                if images.digest(source)!=item['sha256'] or sidecar(source)[1]!=item['caption_revision']:raise ValueError('処理中に元画像またはTXTが変わりました')
                job['results'].append({'source':item['relative'],'source_sha256':item['sha256'],'relative':job['folder']+'/'+name,'path':str(output),'source_size':source_size,'output_size':list(result.size),'bytes':output.stat().st_size,'sha256':images.digest(output),'ai_applied':ai_applied,'caption_generated':bool(predictor),'has_txt':output.with_suffix('.txt').exists()})
            except Exception as error:
                job['failed']+=1;job['errors'].append({'relative':item['relative'],'message':str(error)})
                if output and output.parent==stage:output.unlink(missing_ok=True);output.with_suffix('.txt').unlink(missing_ok=True)
            job['done']+=1;persist()
        write_state(stage/'.preprocess.json',json.dumps({'settings':config,'results':job['results'],'errors':job['errors'],'cancelled':cancelled},ensure_ascii=False,indent=2))
        final=root/job['folder'];stage.rename(final)
        job['folder_committed']=True
        for row in job['results']:row['path']=str(root/row['relative'])
        job.update(status='cancelled' if cancelled else 'done_with_errors' if job['failed'] else 'done',message=f"{'停止' if cancelled else '処理終了'}：成功 {len(job['results'])}枚・失敗 {job['failed']}枚・未処理 {job['total']-job['done']}枚。原本は保持しています。")
        persist();write_state(directory/(job['id']+'.json'),json.dumps(job,ensure_ascii=False,indent=2))
    except Exception as error:
        import traceback
        traceback.print_exc();job.update(status='failed',message=str(error));persist()
    finally:stop.unlink(missing_ok=True)
    return {'status':job['status']}
