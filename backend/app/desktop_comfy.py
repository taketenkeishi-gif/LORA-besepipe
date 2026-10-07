"""ComfyUI binding and persisted native preview collection for the desktop app."""
from __future__ import annotations
import base64,copy,io,json,os,subprocess,sys,time,uuid
from pathlib import Path
from urllib.parse import urlparse
import requests,psutil
from filelock import FileLock,Timeout
from PIL import Image
from . import dataset_desktop as desktop
from .routers.dataset_files import database_connection,root_for,atomic_text
from .services.comfyui_client import ComfyUIClient
from .desktop_jobs import write_state

def _url(value):
    url=str(value).strip().rstrip('/');parsed=urlparse(url)
    if parsed.scheme not in ('http','https') or parsed.hostname not in ('localhost','127.0.0.1','::1') or parsed.username or parsed.password:raise ValueError('このPCのComfyUI接続先を指定してください（例 http://127.0.0.1:8188）')
    return url

def settings():
    with database_connection() as conn:values={r['key']:r['value'] for r in conn.execute("SELECT key,value FROM app_settings WHERE key IN ('comfyui_root','comfyui_url')")}
    return {'root':values.get('comfyui_root') or str(Path.home()/'AI_tools/ComfyUI-Portable/ComfyUI_windows_portable/ComfyUI'),'url':values.get('comfyui_url') or 'http://127.0.0.1:8188'}

def inspect(payload):
    current=settings();url=_url(payload.get('url') or current['url']);root=Path(payload.get('root') or current['root']).expanduser().resolve()
    if not (root/'main.py').is_file() or not (root/'models').is_dir():raise ValueError('ComfyUIのmain.pyとmodelsがある本体フォルダを選択してください')
    try:
        response=requests.get(url+'/system_stats',timeout=4);response.raise_for_status();stats=response.json()
    except Exception as error:raise ValueError(f'ComfyUIに接続できません：{url}。起動中のインスタンスを選び直してください。') from error
    argv=stats.get('system',{}).get('argv',[])
    if not argv or Path(argv[0]).resolve()!=(root/'main.py').resolve():raise ValueError('接続先インスタンスと本体フォルダが一致しません')
    response=requests.get(url+'/object_info',timeout=20);response.raise_for_status();info=response.json()
    models=[]
    for category,node,field in [('checkpoints','CheckpointLoaderSimple','ckpt_name'),('diffusion_models','UNETLoader','unet_name')]:
        for name in ComfyUIClient._extract_combo(info,node,field):
            candidates=[root/'models'/category/name]
            if category=='diffusion_models':candidates.append(root/'models/unet'/name)
            path=next((p.resolve() for p in candidates if p.is_file()),None)
            models.append({'id':category+':'+name,'name':name,'category':category,'path':str(path) if path else None})
    workflows=[]
    for folder in (root/'user/default/workflows',root/'workflows'):
        if folder.is_dir():
            for path in sorted(folder.rglob('*.json'))[:200]:workflows.append({'path':str(path.resolve()),'name':str(path.relative_to(folder))})
    result={'root':str(root),'url':url,'connected':True,'devices':stats.get('devices',[]),'models':models,'loras':ComfyUIClient._extract_combo(info,'LoraLoader','lora_name'),'workflows':workflows,'nodes':len(info)}
    if payload.get('include_schema'):result['object_info']=info
    return result

def discover(payload):
    current=settings();candidates=[]
    for url in dict.fromkeys([current['url'],'http://127.0.0.1:8188','http://127.0.0.1:8189']):
        try:
            stats=requests.get(_url(url)+'/system_stats',timeout=1).json();argv=stats.get('system',{}).get('argv',[])
            if argv:candidates.append({'url':url,'root':str(Path(argv[0]).resolve().parent),'device':', '.join(d.get('name','') for d in stats.get('devices',[]))})
        except Exception:pass
    return {**current,'instances':candidates}

def bind(payload):
    result=inspect(payload)
    with database_connection() as conn:
        for key,value in [('comfyui_root',result['root']),('comfyui_url',result['url'])]:conn.execute("INSERT INTO app_settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_at=CURRENT_TIMESTAMP",(key,value))
    return result

def workflow(payload):
    root=Path(settings()['root']).resolve();path=Path(payload['path']).resolve()
    if not any(path.is_relative_to(root/folder) for folder in ['user/default/workflows','workflows']) or path.suffix.lower()!='.json':raise ValueError('接続したComfyUIのワークフローを選択してください')
    if path.stat().st_size>10*1024*1024:raise ValueError('ワークフローが大きすぎます')
    return {'data':json.loads(path.read_text(encoding='utf-8-sig')),'url':_url(settings()['url']),'path':str(path)}

def checkpoints(payload):
    with database_connection() as conn:
        rows=conn.execute('SELECT id,file_path,epoch,run_id FROM checkpoints WHERE project_id=? ORDER BY id DESC',(int(payload['project_id']),)).fetchall()
    return {'checkpoints':[dict(r) for r in rows if Path(r['file_path']).is_file()]}

def _directory(project_id):
    root_for(project_id);target=desktop.ROOT/'.runtime/comfy-previews'/str(project_id);target.mkdir(parents=True,exist_ok=True);return target

def _share_checkpoint(comfy_root,run_id,source):
    source=Path(source).resolve(strict=True)
    target_dir=Path(comfy_root)/'models/loras/LoRA-Studio-preview'/f'{run_id}-{source.stat().st_ino:x}'
    target_dir.mkdir(parents=True,exist_ok=True);target=target_dir/source.name
    if target.exists():
        if not target.samefile(source):raise ValueError('共有先に別のLoRAファイルがあります。上書きは行いません')
    else:
        try:os.link(source,target)
        except OSError as error:raise ValueError('学習LoRAの共有リンクを作成できません。同じドライブ上の学習出力とComfyUIを使用してください') from error
    return str(target.relative_to(Path(comfy_root)/'models/loras'))

def preview_status(payload):
    target=_directory(int(payload['project_id']))/'latest.json'
    if not target.exists():return {'status':'idle'}
    value=json.loads(target.read_text(encoding='utf8'))
    if value['status'] in ('starting','queued','running') and time.time()-value['created']>30:
        try:alive=abs(psutil.Process(value['pid']).create_time()-value['process_created'])<1
        except (psutil.Error,KeyError):alive=False
        if not alive:value.update(status='interrupted',message='プレビューの受信が中断されました。接続先の履歴と生成記録を確認してください。');write_state(target,json.dumps(value,ensure_ascii=False))
    if value.get('image_path'):
        path=Path(value['image_path'])
        if path.is_file():
            with Image.open(path) as raw:
                img=raw.copy();img.thumbnail((768,768));stream=io.BytesIO();img.save(stream,format='PNG');value['image']='data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode()
    return value

def _graph(payload,catalog):
    info=catalog['object_info'];family=payload.get('family','anima');model=next((m for m in catalog['models'] if m['id']==payload.get('model')),None)
    imported=payload.get('graph') is not None
    if not model and not imported:raise ValueError('接続先のベースモデルを選択してください')
    width=int(payload.get('width',1024));height=int(payload.get('height',1024));steps=int(payload.get('steps',20));cfg=float(payload.get('cfg',5));seed=int(payload.get('seed',42));strength=float(payload.get('strength',1))
    if not imported and not (256<=width<=2048 and 256<=height<=2048 and width%64==0 and height%64==0 and 1<=steps<=100 and 0<=cfg<=30 and 0<=seed<2**63 and -2<=strength<=2):raise ValueError('生成サイズ・ステップ・CFG・Seed・LoRA強度の範囲を確認してください')
    lora='' if imported else str(payload.get('lora') or '');checkpoint_id=None if imported else payload.get('checkpoint_id')
    if checkpoint_id:
        with database_connection() as conn:row=conn.execute('SELECT * FROM checkpoints WHERE id=? AND project_id=?',(int(checkpoint_id),int(payload['project_id']))).fetchone()
        if row is None or not Path(row['file_path']).is_file():raise ValueError('このプロジェクトの学習LoRAが見つかりません')
        lora=_share_checkpoint(catalog['root'],row['run_id'],Path(row['file_path']))
        if not lora:raise ValueError('学習LoRAをComfyUIへ配置できませんでした')
    elif lora and lora not in catalog['loras']:raise ValueError('接続先にないLoRAです')
    extra_loras=payload.get('extra_loras') or []
    if not isinstance(extra_loras,list) or len(extra_loras)>8:raise ValueError('追加LoRAは8件まで指定できます')
    stack=[{'enabled':True,'lora_name':lora,'strength_model':strength,'strength_clip':strength}] if lora else []
    for extra in extra_loras:
        name=str(extra.get('name') or '');weight=float(extra.get('strength',1))
        if name not in catalog['loras'] or not -2<=weight<=2:raise ValueError('追加LoRAの名前または強度が不正です')
        candidate=Path(catalog['root'])/'models'/'loras'/name
        if candidate.is_file() and candidate.suffix.lower()=='.safetensors':
            from safetensors import safe_open
            with safe_open(str(candidate),framework='numpy') as reader:
                version=str((reader.metadata() or {}).get('ss_base_model_version','')).lower()
            if (family=='anima' and any(x in version for x in ('sdxl','sd_v','flux'))) or (family=='sdxl' and 'anima' in version):raise ValueError(f'モデル系統が異なる追加LoRAです：{name}')
        stack.append({'enabled':True,'lora_name':name,'strength_model':weight,'strength_clip':weight})
    positive=str(payload.get('positive') or '').strip();negative=str(payload.get('negative') or '')
    if not positive and not imported:raise ValueError('プレビューのプロンプトを入力してください')
    if imported:
        graph=copy.deepcopy(payload['graph'])
        if not isinstance(graph,dict) or not graph or len(graph)>300:raise ValueError('ワークフローのAPI形式が不正です')
        # Imported graph owns its model/conditioning. Do not silently rewrite its internals.
    elif family=='anima':
        clips=ComfyUIClient._extract_combo(info,'UnifiedModelStackLoader','clip_name');vaes=ComfyUIClient._extract_combo(info,'UnifiedModelStackLoader','vae_name')
        clip=next((n for n in clips if Path(n).name=='qwen_3_06b_base.safetensors'),None);vae=next((n for n in vaes if Path(n).name=='qwen_image_vae.safetensors'),None)
        if not clip or not vae:raise ValueError('接続先にAnima用QwenテキストエンコーダーとVAEが見つかりません')
        graph=ComfyUIClient.build_musubi_preview_graph(model_family='anima',base_models={'ckpt_name':model['name'],'clip_name':clip,'clip_type':'qwen_image','vae_name':vae},lora_name=lora,positive=positive,negative=negative,width=width,height=height,steps=steps,cfg=cfg,sampler_name=str(payload.get('sampler') or 'euler'),seed=seed)
        graph['loader']['inputs']['lora_stack']=json.dumps(stack)
        graph['sampler']['inputs']['scheduler']=str(payload.get('scheduler') or 'simple')
    elif family=='sdxl':
        if model['category']!='checkpoints':raise ValueError('SDXLはCheckpoint形式のモデルを選択してください')
        graph={'loader':{'class_type':'CheckpointLoaderSimple','inputs':{'ckpt_name':model['name']}},'positive':{'class_type':'CLIPTextEncode','inputs':{'clip':['loader',1],'text':positive}},'negative':{'class_type':'CLIPTextEncode','inputs':{'clip':['loader',1],'text':negative}},'latent':{'class_type':'EmptyLatentImage','inputs':{'width':width,'height':height,'batch_size':1}},'sampler':{'class_type':'KSampler','inputs':{'model':['loader',0],'positive':['positive',0],'negative':['negative',0],'latent_image':['latent',0],'seed':seed,'steps':steps,'cfg':cfg,'sampler_name':'euler','scheduler':'normal','denoise':1}},'decode':{'class_type':'VAEDecode','inputs':{'samples':['sampler',0],'vae':['loader',2]}},'out':{'class_type':'SaveImage','inputs':{'images':['decode',0],'filename_prefix':'unused'}}}
        if lora:
            graph['lora']={'class_type':'LoraLoader','inputs':{'model':['loader',0],'clip':['loader',1],'lora_name':lora,'strength_model':strength,'strength_clip':strength}};graph['sampler']['inputs']['model']=['lora',0]
            for node in ['positive','negative']:graph[node]['inputs']['clip']=['lora',1]
    else:raise ValueError('このモデル系統は接続先の保存ワークフローを選択してください')
    if not imported and family=='sdxl' and stack:
        graph.pop('lora',None)
        current_model=['loader',0];current_clip=['loader',1]
        for index,entry in enumerate(stack):
            node=f'lora_stack_{index}'
            graph[node]={'class_type':'LoraLoader','inputs':{'model':current_model,'clip':current_clip,'lora_name':entry['lora_name'],'strength_model':entry['strength_model'],'strength_clip':entry['strength_clip']}}
            current_model=[node,0];current_clip=[node,1]
        graph['sampler']['inputs']['model']=current_model
        graph['positive']['inputs']['clip']=current_clip;graph['negative']['inputs']['clip']=current_clip
    output_nodes=[]
    for key,node in graph.items():
        if node.get('class_type') in ('SaveImage','PreviewImage','SaveImageWebsocket'):
            node['class_type']='SaveImageWebsocket';node['inputs']={'images':node['inputs']['images']};output_nodes.append(key)
        elif node.get('class_type')=='KSampler Adv. (Efficient)':node['inputs']['preview_method']='none'
        elif info.get(node.get('class_type'),{}).get('output_node'):raise ValueError(f"画像以外の出力ノードは実行しません：{node.get('class_type')}")
    if not output_nodes:raise ValueError('ワークフローに画像の出力ノードがありません')
    validation=ComfyUIClient.validate_graph_against_object_info(graph,info)
    if not validation['ok']:raise ValueError('接続先で不足するノード/入力：'+str(validation))
    return graph,output_nodes

def preview_start(payload):
    try:
        with FileLock(str(_directory(int(payload['project_id']))/'start.lock'),timeout=0):return _preview_start(payload)
    except Timeout:raise ValueError('プレビューの開始処理中です')

def _preview_start(payload):
    project_id=int(payload['project_id']);target=_directory(project_id)
    if preview_status(payload)['status'] in ('starting','queued','running'):raise ValueError('このプロジェクトのプレビューが実行中です')
    catalog=inspect({'include_schema':True})
    gpu=next((d for d in catalog['devices'] if 'RTX 3090 Ti' in d.get('name','')),None)
    if not gpu:raise ValueError('接続先の使用GPUが許可されたRTX 3090 Tiと一致しません')
    graph,outputs=_graph(payload,catalog);job_id=uuid.uuid4().hex;jobdir=target/job_id;jobdir.mkdir()
    atomic_text(jobdir/'workflow-api.json',json.dumps(graph,ensure_ascii=False,indent=2))
    job={'id':job_id,'project_id':project_id,'status':'starting','created':time.time(),'url':catalog['url'],'graph_path':str(jobdir/'workflow-api.json'),'output_nodes':outputs,'image_path':None,'expected_artifact':str(jobdir/'preview.png'),'collection_required':True,'collection_state':'PENDING','message':'接続先キューへ送信中','devices':catalog['devices'],'settings':payload}
    write_state(target/'latest.json',json.dumps(job,ensure_ascii=False))
    script=Path(__file__).resolve().parents[1]/'desktop_actions.py'
    with (jobdir/'worker.log').open('ab') as log:
        proc=subprocess.Popen([sys.executable,'-X','utf8',str(script),'comfy-worker'],stdin=subprocess.PIPE,stdout=log,stderr=log,cwd=script.parent,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),env=os.environ.copy())
    proc.stdin.write(json.dumps({'project_id':project_id,'job_id':job_id}).encode());proc.stdin.close()
    return job

def preview_run(payload):
    import websocket
    project_id=int(payload['project_id']);target=_directory(project_id);job=json.loads((target/'latest.json').read_text(encoding='utf8'))
    if job['id']!=payload['job_id']:return
    def persist():write_state(target/'latest.json',json.dumps(job,ensure_ascii=False))
    return collect_graph(job,persist)

def collect_graph(job,persist):
    import websocket
    ws=None
    try:
        job.update(pid=os.getpid(),process_created=psutil.Process().create_time());persist()
        client=ComfyUIClient(job['url'],timeout=8);ws=websocket.create_connection(job['url'].replace('http','ws',1)+'/ws?clientId='+client.client_id,timeout=8)
        graph=json.loads(Path(job['graph_path']).read_text(encoding='utf8'));prompt_id=client.submit(graph);job.update(prompt_id=prompt_id,status='queued',message='ComfyUIのキューへ登録しました');persist()
        current_node=None;deadline=time.monotonic()+1800;ws.settimeout(5);received=[];completion_seen=False
        job['transport_events']=[]
        while time.monotonic()<deadline:
            try:message=ws.recv()
            except websocket.WebSocketTimeoutException:
                try:
                    history=requests.get(job['url']+'/history/'+prompt_id,timeout=5).json().get(prompt_id)
                except requests.RequestException:
                    # Keep receiving the already submitted prompt; a busy HTTP
                    # endpoint must not discard its websocket image stream.
                    continue
                if history:
                    state=history.get('status',{})
                    if state.get('status_str')=='error':raise ValueError('ComfyUIで生成に失敗しました：'+json.dumps(state,ensure_ascii=False)[:800])
                    if state.get('completed'):
                        if received:break
                        if not completion_seen:
                            completion_seen=True;deadline=min(deadline,time.monotonic()+10)
                        # Completion and binary delivery can be observed in
                        # different order. Drain this prompt's websocket first.
                continue
            if isinstance(message,str):
                event=json.loads(message);data=event.get('data',{})
                if event.get('type') in ('executing','execution_success','execution_error'):
                    job['transport_events']=(job['transport_events']+[{'type':event['type'],'node':data.get('node'),'prompt_id':data.get('prompt_id')}])[-40:]
                if data.get('prompt_id') not in (None,prompt_id):continue
                if event.get('type')=='executing' and data.get('prompt_id')==prompt_id:
                    if data.get('node') is None:
                        if received:break
                        if not completion_seen:
                            completion_seen=True;deadline=min(deadline,time.monotonic()+10)
                        continue
                    current_node=data.get('node')
                    job.update(status='running',message='接続先でプレビューを生成中');persist()
                elif event.get('type')=='execution_error':raise ValueError(data.get('exception_message') or 'ComfyUIの生成エラー')
                elif event.get('type')=='progress':job['message']=f"生成中 {data.get('value',0)} / {data.get('max',0)}";job.update(progress_value=data.get('value',0),progress_max=data.get('max',0),progress_at=time.time());persist()
            elif current_node in job['output_nodes']:
                job['transport_events']=(job['transport_events']+[{'type':'binary','bytes':len(message),'node':current_node}])[-40:]
                with Image.open(io.BytesIO(message[8:])) as raw:
                    raw.load();path=Path(job['expected_artifact']) if not received else Path(job['expected_artifact']).with_name(f'preview-{len(received)+1}.png');raw.save(path);received.append(str(path))
                if completion_seen:break
        if not received:raise ValueError('生成画像を受信できませんでした。接続先の履歴を確認してください。')
        job.update(status='done',message='プレビュー画像を保存しました',image_path=received[0],images=received,collection_state='COLLECTED');persist()
    except Exception as error:
        import traceback
        traceback.print_exc();job.update(status='failed',message=str(error));persist()
    finally:
        if ws:ws.close()
    return {'status':job['status']}

def reveal(payload):
    job=preview_status(payload);path=Path(job.get('image_path') or '')
    if not path.is_file():raise ValueError('プレビュー画像がありません')
    subprocess.Popen(['explorer.exe','/select,',str(path)]);return {'path':str(path)}
