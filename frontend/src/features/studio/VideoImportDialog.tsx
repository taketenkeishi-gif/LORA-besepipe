import {useEffect,useMemo,useRef,useState} from 'react';
import {Badge,Button,Callout,Checkbox,Dialog,Flex,Progress,SegmentedControl,Spinner,Text,TextField} from '@radix-ui/themes';
import {API_BASE,apiGet,apiPost} from '../../lib/api';
import {desktop} from './desktopBridge';

type Item={index:number;start:number;end:number;timestamp:number;sharpness:number;score:number|null;matched:string[];tags:string};
type Job={status:'running'|'done'|'cancelled'|'error';stage:string;done:number;total:number;error:string;video:string;items:Item[];ranking:{signature:string[];enabled:boolean;reason:string}};
// 大きいほど「大きなカットだけ」を区切りにする（ffmpegのscene閾値）
const SENSITIVITY:Record<string,number>={fine:0.15,normal:0.3,coarse:0.5};
const AUTO_SELECT_SCORE=0.4;
const mmss=(s:number)=>`${Math.floor(s/60)}:${String(Math.floor(s%60)).padStart(2,'0')}`;

export default function VideoImportDialog({open,onOpenChange,projectId,folder,onAdded}:{open:boolean;onOpenChange:(open:boolean)=>void;projectId:number;folder:string;onAdded:(count:number,skipped:number)=>void}){
 const mode='quick';
 const [path,setPath]=useState(''),[sensitivity,setSensitivity]=useState('normal'),[rank,setRank]=useState(true);
 const [jobId,setJobId]=useState(''),[job,setJob]=useState<Job|null>(null),[picked,setPicked]=useState<Set<number>>(new Set()),[busy,setBusy]=useState(false),[error,setError]=useState('');
 const autoSelected=useRef('');
 const running=job?.status==='running'||(busy&&!job);

 useEffect(()=>{if(!open){setJobId('');setJob(null);setPicked(new Set());setError('');autoSelected.current='';}},[open]);
 useEffect(()=>{
  if(!jobId)return;let live=true;
  const poll=async()=>{try{const s=await apiGet<Job>(`/dataset-files/${projectId}/video/${jobId}`);if(!live)return;setJob(s);if(s.status==='error')setError(s.error);}catch(e){if(live)setError(e instanceof Error?e.message:String(e));}};
  void poll();const timer=window.setInterval(()=>{if(job?.status!=='running'&&job)return;void poll();},1500);
  return()=>{live=false;window.clearInterval(timer);};
 },[jobId,projectId,job?.status]);
 useEffect(()=>{
  // 解析が終わった最初の1回だけ、キャラらしい候補（キャラ判定なしなら全部）を選んでおく
  if(!job||job.status!=='done'||autoSelected.current===jobId)return;autoSelected.current=jobId;
  setPicked(new Set(job.items.filter(i=>job.ranking.enabled?(i.score??0)>=AUTO_SELECT_SCORE:true).map(i=>i.index)));
 },[job,jobId]);

 async function choose(){try{const p=await desktop()?.chooseVideo?.();if(p)setPath(p);}catch(e){setError(e instanceof Error?e.message:String(e));}}
 async function start(){
  setBusy(true);setError('');setJob(null);setPicked(new Set());autoSelected.current='';
  try{const r=await apiPost<{job_id:string}>(`/dataset-files/${projectId}/video/analyze`,{video_path:path,threshold:SENSITIVITY[sensitivity],rank_character:rank});setJobId(r.job_id);}
  catch(e){setError(e instanceof Error?e.message:String(e));}finally{setBusy(false);}
 }
 async function stop(){try{await apiPost(`/dataset-files/${projectId}/video/${jobId}/cancel`,{});}catch{/* 既に終了している */}}
 async function add(){
  setBusy(true);setError('');
  try{const r=await apiPost<{count:number;skipped:string[]}>(`/dataset-files/${projectId}/video/${jobId}/accept`,{indices:[...picked],folder},'POST',undefined,60000);onAdded(r.count,r.skipped.length);onOpenChange(false);}
  catch(e){setError(e instanceof Error?e.message:String(e));}finally{setBusy(false);}
 }
 const items=job?.items??[];
 const toggle=(i:number)=>setPicked(prev=>{const next=new Set(prev);if(next.has(i))next.delete(i);else next.add(i);return next;});
 const finished=job&&(job.status==='done'||job.status==='cancelled');
 const progressValue=useMemo(()=>job&&job.total>0?Math.round(job.done/job.total*100):0,[job]);

 return <Dialog.Root open={open} onOpenChange={onOpenChange}><Dialog.Content maxWidth="980px" aria-describedby={undefined}>
  <Dialog.Title>動画からシーンごとに1枚ずつ追加</Dialog.Title>
  <Flex gap="2" align="center" mb="3"><Text size="1" color="gray">キャラごとに分けてLoRAを作るときは</Text>
   <Button size="1" variant="soft" onClick={()=>{onOpenChange(false);window.dispatchEvent(new CustomEvent('open-character-sets'));}}>動画からLoRA を開く</Button></Flex>
  {mode==='quick'&&<>
  <Text as="p" size="2" color="gray" mb="3">動画をカットごとに区切り、各カットのいちばんくっきりした1枚だけを候補にします。全フレームは取り込みません。追加先：{folder||'データセット'}</Text>
  <Flex gap="2" align="end" wrap="wrap">
   <label style={{flex:1,minWidth:280}}><Text size="1" color="gray">動画のファイル</Text>
    <TextField.Root aria-label="動画のパス" placeholder={String.raw`C:\Videos\example.mp4`} value={path} disabled={running} onChange={e=>setPath(e.target.value)}/></label>
   {desktop()?.chooseVideo&&<Button variant="soft" color="gray" disabled={running} onClick={()=>void choose()}>ファイルを選ぶ</Button>}
  </Flex>
  <Flex gap="4" align="center" wrap="wrap" mt="3">
   <Flex gap="2" align="center"><Text size="1" color="gray">カットの区切り方</Text>
    <SegmentedControl.Root size="1" value={sensitivity} onValueChange={setSensitivity} aria-label="カットの区切り方"><SegmentedControl.Item value="fine">細かく</SegmentedControl.Item><SegmentedControl.Item value="normal">標準</SegmentedControl.Item><SegmentedControl.Item value="coarse">大きなカットだけ</SegmentedControl.Item></SegmentedControl.Root></Flex>
   <Text as="label" size="2"><Flex gap="2" align="center"><Checkbox checked={rank} disabled={running} onCheckedChange={v=>setRank(v===true)}/>今のデータセットのキャラに近い順に並べる</Flex></Text>
   <Button disabled={!path.trim()||running} onClick={()=>void start()}>{running?<Spinner/>:null}{job?'もう一度解析':'解析する'}</Button>
   {job?.status==='running'&&<Button variant="soft" color="gray" onClick={()=>void stop()}>やめる</Button>}
  </Flex>
  {running&&<div style={{marginTop:12}}><Text size="1" color="gray">{job?.stage||'準備中'}（{job?.done??0}/{job?.total??0}）</Text><Progress value={progressValue} mt="1"/></div>}
  {error&&<Callout.Root color="red" size="1" mt="3"><Callout.Text>{error}</Callout.Text></Callout.Root>}
  {finished&&<>
   <Flex gap="3" align="center" wrap="wrap" mt="4"><Text size="2" weight="bold">{items.length}シーン</Text><Text size="2" color="gray">{picked.size}枚を選択中</Text>
    <Button size="1" variant="ghost" color="gray" onClick={()=>setPicked(new Set(items.map(i=>i.index)))}>すべて選ぶ</Button>
    <Button size="1" variant="ghost" color="gray" onClick={()=>setPicked(new Set())}>選択を解除</Button>
    {job.ranking.enabled&&<Button size="1" variant="ghost" onClick={()=>setPicked(new Set(items.filter(i=>(i.score??0)>=AUTO_SELECT_SCORE).map(i=>i.index)))}>キャラに近い候補だけ選ぶ</Button>}
    {job.status==='cancelled'&&<Badge color="amber">途中でやめたため、一部のシーンだけです</Badge>}</Flex>
   {!job.ranking.enabled&&rank&&job.ranking.reason&&<Text as="p" size="1" color="amber" mt="1">キャラ判定は使えませんでした：{job.ranking.reason}。シーン順に並べています。</Text>}
   {job.ranking.enabled&&<Text as="p" size="1" color="gray" mt="1">キャラ判定の基準（今のデータセットの共通タグ）：{job.ranking.signature.slice(0,8).join('、')}。一致が多いほど上に並びます。タグ判定は目安なので、追加前に目で確認してください。</Text>}
   <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fill,minmax(150px,1fr))',gap:8,marginTop:12,maxHeight:'46vh',overflow:'auto'}}>
    {items.map(i=><button type="button" key={i.index} onClick={()=>toggle(i.index)} aria-pressed={picked.has(i.index)} title={i.matched.length?`一致タグ：${i.matched.join('、')}`:undefined}
      style={{position:'relative',padding:0,border:picked.has(i.index)?'2px solid var(--accent-9)':'2px solid transparent',borderRadius:6,background:'var(--gray-3)',cursor:'pointer',textAlign:'left'}}>
     <img src={`${API_BASE}/dataset-files/${projectId}/video/${jobId}/image/${i.index}`} alt={`シーン${i.index} ${mmss(i.timestamp)}`} loading="lazy" style={{width:'100%',aspectRatio:'1',objectFit:'cover',borderRadius:4,display:'block'}}/>
     <Flex justify="between" px="2" py="1"><Text size="1">{mmss(i.timestamp)}</Text>{i.score!==null&&<Badge size="1" color={i.score>=AUTO_SELECT_SCORE?'green':'gray'}>{Math.round(i.score*100)}%</Badge>}</Flex>
     {picked.has(i.index)&&<span aria-hidden style={{position:'absolute',top:6,right:6,background:'var(--accent-9)',color:'white',borderRadius:10,fontSize:11,padding:'0 6px'}}>選択</span>}
    </button>)}
   </div>
  </>}
  <Flex gap="3" justify="end" mt="4"><Button variant="soft" color="gray" onClick={()=>onOpenChange(false)}>閉じる</Button><Button disabled={!finished||!picked.size||busy} onClick={()=>void add()}>{busy&&finished?<Spinner/>:null}{picked.size}枚をデータセットに追加</Button></Flex>
  </>}
 </Dialog.Content></Dialog.Root>;
}
