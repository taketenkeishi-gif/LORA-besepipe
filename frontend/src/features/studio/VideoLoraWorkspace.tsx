import {useCallback,useEffect,useRef,useState} from 'react';
import {Button,Checkbox,Select,Spinner,Text,TextField} from '@radix-ui/themes';
import {Check,Film,FolderOpen,Users,Sparkles} from 'lucide-react';
import {apiGet,apiPost} from '../../lib/api';
import {desktop} from './desktopBridge';
import CharacterEditor,{type SetState} from './CharacterEditor';
import './character-sets.css';

// "動画からLoRA": one workspace per folder of videos.  ① 動画 - every episode is extracted in a queue, then grouped into
// characters (backend services/video_lora_pipeline.py).  ② キャラ - check and fix the characters.  ③ LoRA - the ticked
// characters become projects, their outfits instances; the new project opens.
type Video={path:string;name:string;job:string|null;status:string;percent:number;eta_s:number|null;error:string};
type Pipeline={stage?:string;stage_percent?:number;stage_eta_s?:number|null;error?:string;running?:boolean;videos?:Video[]};
type Full=SetState&{pipeline?:Pipeline};
type Ws={set_id:string;folder:string;name:string;videos:number;stage:string;running:boolean};
type Outfit={name:string;files:string[]};
const LAST='videolora:last';
const STATUS:Record<string,string>={done:'完了',running:'処理中',queued:'待機中',waiting:'GPUの空き待ち',error:'エラー',cancelled:'中止'};
const STAGE:Record<string,string>={videos:'動画から人物を切り出しています',features:'キャラを見分ける特徴を計算しています',characters:'全動画をまたいでキャラに分けています',outfits:'仕上げ中',ready:'完了',error:'エラー',cancelled:'中止しました'};
const time=(s?:number|null)=>s==null?'':s<60?`残り${Math.max(1,Math.round(s))}秒`:s<3600?`残り${Math.round(s/60)}分`:`残り${(s/3600).toFixed(1)}時間`;

export default function VideoLoraWorkspace(){
 const [list,setList]=useState<Ws[]>([]),[sid,setSid]=useState<string>(()=>{try{return localStorage.getItem(LAST)||'';}catch{return '';}});
 const [set,setSet]=useState<Full|null>(null),[step,setStep]=useState<'videos'|'chars'|'lora'>('chars');
 const [error,setError]=useState(''),[busy,setBusy]=useState(false);
 const [picked,setPickedRaw]=useState<Set<number>>(new Set());
 const pickKey=`videolora:picked:${sid}`;
 useEffect(()=>{try{setPickedRaw(new Set(JSON.parse(localStorage.getItem(pickKey)||'[]')));}catch{setPickedRaw(new Set());}},[pickKey]);
 const setPicked=(f:(s:Set<number>)=>Set<number>)=>setPickedRaw(s=>{const n=f(s);try{localStorage.setItem(pickKey,JSON.stringify([...n]));}catch{}return n;});

 const loadList=useCallback(async()=>{try{const r=await apiGet<{workspaces:Ws[]}>('/character-sets/workspaces',30000);setList(r.workspaces);if(!sid&&r.workspaces[0])setSid(r.workspaces[0].set_id);}catch(e){setError(String(e));}},[sid]);
 useEffect(()=>{void loadList();},[]);
 const loadSet=useCallback(async()=>{if(!sid)return null;const s=await apiGet<Full>(`/character-sets/${encodeURIComponent(sid)}`,30000);setSet(s);return s;},[sid]);
 // the first step is chosen once per workspace from its state; a step the user clicked is never overridden by a later load
 const chose=useRef(false);
 useEffect(()=>{if(!sid)return;try{localStorage.setItem(LAST,sid);}catch{}setSet(null);chose.current=false;
  void loadSet().then(s=>{if(s&&!chose.current)setStep(s.status==='ready'?'chars':'videos');}).catch(e=>setError(String(e)));},[sid,loadSet]);
 const running=!!set?.pipeline?.running;
 useEffect(()=>{if(!running&&set?.status!=='computing')return;const t=setInterval(()=>void loadSet().catch(()=>{}),2000);return()=>clearInterval(t);},[running,set?.status,loadSet]);

 async function chooseFolder(){setError('');try{const folder=await desktop()?.chooseVideoFolder?.();if(!folder)return;setBusy(true);
  const r=await apiPost<{set_id:string}>('/character-sets/workspace',{folder},'POST',undefined,60000);await loadList();setSid(r.set_id);chose.current=true;setStep('videos');}
  catch(e){setError(e instanceof Error?e.message:String(e));}finally{setBusy(false);}}
 async function rerun(){if(!set?.folder)return;setBusy(true);setError('');try{await apiPost('/character-sets/workspace',{folder:set.folder},'POST',undefined,60000);await loadSet();}catch(e){setError(e instanceof Error?e.message:String(e));}finally{setBusy(false);}}
 async function cancel(){try{await apiPost(`/character-sets/${encodeURIComponent(sid)}/pipeline/cancel`,{},'POST');await loadSet();}catch(e){setError(String(e));}}

 const p=set?.pipeline??{};
 const videos=p.videos??[];
 const done=videos.filter(v=>v.status==='done').length;
 const mains=(set?.characters??[]).filter(c=>!c.pending);
 const ready=set?.status==='ready';
 const stepBtn=(id:typeof step,icon:JSX.Element,label:string,sub:string,enabled=true)=>
  <button type="button" className={`vl-step${step===id?' on':''}`} disabled={!enabled} onClick={()=>{chose.current=true;setStep(id);}}>{icon}<span><b>{label}</b><small>{sub}</small></span></button>;

 return <div className="vl-root">
  <div className="vl-bar">
   <Select.Root value={sid||undefined} onValueChange={setSid}><Select.Trigger aria-label="動画フォルダ" placeholder="動画フォルダ"/>
    <Select.Content>{list.map(w=><Select.Item key={w.set_id} value={w.set_id}>{w.name}（{w.videos}本）</Select.Item>)}</Select.Content></Select.Root>
   {desktop()?.chooseVideoFolder&&<Button size="1" variant="soft" color="gray" disabled={busy} onClick={()=>void chooseFolder()}><FolderOpen size={14}/>動画フォルダを選ぶ</Button>}
   <span className="vl-steps">
    {stepBtn('videos',running&&p.stage==='videos'?<Spinner/>:<Film size={16}/>,'① 動画',videos.length?`${done}/${videos.length}本${running?`・${STAGE[p.stage||'']?'処理中':''} ${p.stage_percent??0}%`:''}`:'—')}
    {stepBtn('chars',running&&p.stage!=='videos'?<Spinner/>:<Users size={16}/>,'② キャラ',ready?`${mains.length}人`:running?`${p.stage_percent??0}%`:'—',ready)}
    {stepBtn('lora',<Sparkles size={16}/>,'③ LoRA',picked.size?`${picked.size}人を選択`:'キャラを選ぶ',ready)}
   </span>
  </div>
  {error&&<Text as="div" size="1" color="red" className="vl-error">{error}</Text>}
  {!sid?<div className="vl-empty"><Text size="3">動画のフォルダを選ぶと、全話を順番に処理してキャラに分けます</Text>{desktop()?.chooseVideoFolder&&<Button size="3" onClick={()=>void chooseFolder()}><FolderOpen size={16}/>動画フォルダを選ぶ</Button>}</div>
  :!set?<div className="vl-empty"><Spinner size="3"/></div>
  :step==='videos'?<VideosStep set={set} p={p} running={running} busy={busy} onRerun={()=>void rerun()} onCancel={()=>void cancel()} onNext={()=>setStep('chars')}/>
  :step==='chars'&&ready?<CharacterEditor set={set} reload={async()=>(await loadSet())!} picked={picked} setPicked={setPicked}/>
  :step==='lora'&&ready?<LoraStep set={set} picked={picked} setPicked={setPicked}/>
  :<div className="vl-empty"><Spinner size="3"/><Text>{STAGE[p.stage||'']||'準備中'}　{p.stage_percent??0}%　{time(p.stage_eta_s)}</Text></div>}
 </div>;
}

function VideosStep({set,p,running,busy,onRerun,onCancel,onNext}:{set:Full;p:Pipeline;running:boolean;busy:boolean;onRerun:()=>void;onCancel:()=>void;onNext:()=>void}){
 const videos=p.videos??[];
 const left=videos.filter(v=>v.status!=='done').length;
 return <div className="vl-videos">
  <div className="vl-stage">
   {running?<><Spinner/><Text size="2" weight="bold">{STAGE[p.stage||'']}</Text><Text size="2">{p.stage_percent??0}%</Text><Text size="2" color="gray">{time(p.stage_eta_s)}</Text>
     <span className="cs-grow"/><Button size="1" variant="soft" color="red" onClick={onCancel}>中止</Button></>
   :<><Text size="2" weight="bold">{p.stage==='error'?'エラーで止まりました':p.stage==='cancelled'?'中止しました':left?`${left}本が未処理です`:'すべての動画を処理しました'}</Text>
     {p.error&&<Text size="1" color="red">{p.error}</Text>}<span className="cs-grow"/>
     {left>0&&<Button size="1" disabled={busy} onClick={onRerun}>未処理の{left}本を処理する</Button>}
     {set.status==='ready'&&<Button size="1" variant="soft" onClick={onNext}>キャラを確かめる</Button>}</>}
  </div>
  <Text as="div" size="1" color="gray" className="vl-folder">{set.folder}</Text>
  <div className="vl-video-list">{videos.map(v=><div key={v.path} className={`vl-video ${v.status}`}>
   <span className="vl-video-name" title={v.path}>{v.name}</span>
   <span className="vl-video-status">{v.status==='done'?<Check size={14}/>:v.status==='running'?<Spinner size="1"/>:null}{STATUS[v.status]||v.status}</span>
   <span className="vl-meter"><span style={{width:`${v.status==='done'?100:v.percent}%`}}/></span>
   <span className="vl-video-eta">{v.status==='running'?time(v.eta_s):v.error?<span className="vl-video-error" title={v.error}>{v.error}</span>:''}</span></div>)}</div>
 </div>;
}

function LoraStep({set,picked,setPicked}:{set:Full;picked:Set<number>;setPicked:(f:(s:Set<number>)=>Set<number>)=>void}){
 const base=`/character-sets/${encodeURIComponent(set.set_id)}`;
 const chosen=(set.characters??[]).filter(c=>!c.pending&&picked.has(c.id));
 const [names,setNames]=useState<Record<number,string>>({});
 const [outfits,setOutfits]=useState<Record<number,(Outfit&{use:boolean})[]>>({});
 const [busy,setBusy]=useState(false),[error,setError]=useState(''),[made,setMade]=useState<{project_id:number;name:string;images:number;instances:{name:string;trigger:string;images:number}[]}[]>([]);
 useEffect(()=>{for(const c of chosen)if(!outfits[c.id])void apiGet<{outfits:Outfit[]}>(`${base}/outfits?character=${c.id}`,60000).then(r=>setOutfits(o=>({...o,[c.id]:r.outfits.map(x=>({...x,use:x.name!=='その他の衣装'}))}))).catch(e=>setError(String(e)));},[chosen.map(c=>c.id).join(',')]);
 async function make(){setBusy(true);setError('');
  try{const r=await apiPost<{projects:typeof made}>(`${base}/compose`,{characters:chosen.map(c=>({id:c.id,name:(names[c.id]??c.name).trim()||c.name,outfits:outfits[c.id]??[]}))},'POST',undefined,600000);setMade(r.projects);}
  catch(e){setError(e instanceof Error?e.message:String(e));}finally{setBusy(false);}}
 if(!chosen.length)return <div className="vl-empty"><Text size="3">② キャラで、LoRAにするキャラにチェックを付けてください</Text></div>;
 return <div className="vl-lora">
  {chosen.map(c=>{const os=outfits[c.id];return <section key={c.id} className="vl-lora-char">
   <div className="vl-lora-head"><Checkbox checked onCheckedChange={()=>setPicked(s=>{const n=new Set(s);n.delete(c.id);return n;})} aria-label="選択を外す"/>
    <TextField.Root size="2" value={names[c.id]??c.name} onChange={e=>setNames(n=>({...n,[c.id]:e.target.value}))} aria-label="LoRAの名前"/><Text size="1" color="gray">{c.count}枚</Text></div>
   {!os?<Spinner/>:<div className="vl-lora-outfits">{os.map((o,k)=><label key={o.name} className={`vl-outfit${o.use&&o.files.length>=30&&o.name!=='その他の衣装'?' inst':''}`}>
     <Checkbox checked={o.use} disabled={o.name==='その他の衣装'} onCheckedChange={v=>setOutfits(m=>({...m,[c.id]:m[c.id].map((x,j)=>j===k?{...x,use:v===true}:x)}))}/>
     <span>{o.name}</span><small>{o.files.length}枚{o.name==='その他の衣装'?'・キャラのみ':o.use&&o.files.length>=30?'・インスタンス':o.files.length<30?'・少ないのでキャラのみ':'・キャラのみ'}</small></label>)}</div>}
  </section>;})}
  <div className="vl-lora-go">
   <Button size="3" disabled={busy||chosen.some(c=>!outfits[c.id])} onClick={()=>void make()}>{busy?<Spinner/>:<Sparkles size={16}/>}{chosen.length}人をLoRAのプロジェクトにする</Button>
   {error&&<Text size="2" color="red">{error}</Text>}
  </div>
  {made.length>0&&<div className="vl-made">{made.map(m=><div key={m.project_id} className="vl-made-row">
   <Text size="2" weight="bold">{m.name}</Text><Text size="1" color="gray">{m.images}枚・インスタンス {m.instances.length}（{m.instances.map(i=>`${i.trigger} ${i.name}`).join(' / ')||'なし'}）</Text>
   <Button size="1" onClick={()=>window.dispatchEvent(new CustomEvent('open-project',{detail:{id:m.project_id}}))}>開いて学習へ</Button></div>)}</div>}
 </div>;
}
