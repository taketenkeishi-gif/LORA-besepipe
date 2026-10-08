"""Cooperative epoch handoff. Only ComfyUI performs image generation."""
from pathlib import Path
import json,os,time,uuid,types,urllib.request


def _write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False),encoding='utf8');os.replace(temp,path)


def active_window(run_directory):
    try:
        import psutil
        root=Path(run_directory).resolve();v=json.loads((root/'comfy-preview-window.json').read_text(encoding='utf8'))
        p=psutil.Process(int(v['pid']))
        return v if v['state']=='waiting' and v['deadline']>time.time() and abs(p.create_time()-v['process_created'])<1 and any(str(root).casefold() in str(arg).casefold() for arg in p.cmdline()) else None
    except (OSError,ValueError,KeyError,ImportError):return None
    except Exception:return None


def install(run_directory,run_id,backend_url):
    import torch,train_network,psutil,importlib
    directory=Path(run_directory);original=train_network.NetworkTrainer.__init__;holder={}
    for module_name in ('networks.lora_anima','networks.lora'):
        module=importlib.import_module(module_name)
        for function_name in ('create_network','create_network_from_weights'):
            original_factory=getattr(module,function_name)
            def factory(*args,_original=original_factory,**kwargs):
                result=_original(*args,**kwargs);holder['network']=result[0] if isinstance(result,tuple) else result;return result
            setattr(module,function_name,factory)
    def sample(self,accelerator,args,epoch,global_step,device,vae,tokenizer,text_encoder,unet):
        if epoch is None or epoch==0 or epoch%max(1,int(args.save_every_n_epochs or 1)) or not accelerator.is_main_process:return
        torch.cuda.synchronize();torch.cuda.empty_cache()
        checkpoint=directory/'output'/f'{args.output_name}-{int(epoch):06d}.safetensors'
        if not checkpoint.exists():
            network=accelerator.unwrap_model(holder['network'])
            dtype={'bf16':torch.bfloat16,'fp16':torch.float16,'float':torch.float32}.get(args.save_precision,torch.float32)
            metadata={'ss_epoch':str(epoch),'ss_steps':str(global_step),'ss_network_module':str(args.network_module),'ss_network_dim':str(args.network_dim),'ss_network_alpha':str(args.network_alpha),'ss_base_model_version':'anima' if 'anima' in args.network_module else 'sdxl'}
            network.save_weights(str(checkpoint),dtype,metadata)
        if (directory/'preview-parallel.json').is_file():  # previews run on the RTX 3060 beside training: do not pause (read every epoch)
            print(f'COMFY_PREVIEW_PARALLEL epoch={epoch} step={global_step}',flush=True);return
        value={'state':'waiting','epoch':int(epoch),'step':int(global_step),'pid':os.getpid(),'process_created':psutil.Process().create_time(),'request_id':uuid.uuid4().hex,'deadline':time.time()+1800}
        path=directory/'comfy-preview-window.json';_write(path,value)
        print(f'COMFY_PREVIEW_WAIT epoch={epoch} step={global_step}',flush=True)
        try:
            while time.time()<value['deadline']:
                request=urllib.request.Request(backend_url+f'/api/previews/epoch-window/{run_id}',data=json.dumps({'request_id':value['request_id']}).encode(),headers={'Content-Type':'application/json'},method='POST')
                try:
                    with urllib.request.urlopen(request,timeout=12) as response:result=json.load(response)
                    if result.get('state')=='done':
                        print('COMFY_PREVIEW_DONE '+json.dumps(result),flush=True);return
                    if result.get('state')=='rejected':raise RuntimeError(result.get('message','Preview window rejected'))
                except (OSError,ValueError) as exc:
                    print('COMFY_PREVIEW_WAIT_CONNECTION '+str(exc),flush=True)
                time.sleep(2)
            raise RuntimeError('ComfyUI preview did not reach a confirmed terminal state within30minutes; last checkpoint is preserved')
        finally:
            value.update(state='closed',closed_at=time.time());_write(path,value)
    def initialize(self,*args,**kwargs):
        original(self,*args,**kwargs)
        if type(self).__name__ in ('AnimaNetworkTrainer','SdxlNetworkTrainer'):self.sample_images=types.MethodType(sample,self)
    train_network.NetworkTrainer.__init__=initialize
