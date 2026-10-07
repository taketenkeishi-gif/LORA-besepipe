import {useCallback,useEffect,useMemo,useState} from 'react';
import {Badge,Button,Callout,Checkbox,Flex,Spinner,Text,TextField} from '@radix-ui/themes';
import {API_BASE,apiGet,apiPost} from '../../lib/api';

// 複数の動画をまたいで「同じキャラ」のグループを集め、1つのLoRA（新規プロジェクト）にまとめる。
// 自動の候補（確かなもの／要確認）が初期値で、人が足し引きして作成する。
type LinkMember={job:string;video:string;folder:string;images:number;outfits:string[];sample:string[]};
type Instance={label:string;images:number;sample:string[]};
type Cluster={id:number;groups:number;images:number;hair:string;eyes:string;members:LinkMember[];instances?:Instance[];videos?:number;video_names?:string[];tier?:'recommended'|'candidate'|'few'};
type Metrics={groups:number;jobs:number;clusters:number;multi_group_clusters:number;merged_conflicts_must_be_0:number;hair_agreement_inside_merged_clusters:number|null;merge_bar:number};
type Result={state:{running:boolean;error:string;log:string};auto?:{clusters:Cluster[];metrics:Metrics};suggest?:{clusters:Cluster[];metrics:Metrics}};
type Group={job:string;video:string;folder:string;images:number;sample:string[];outfits:string[]};

const base=(pid:number)=>`/dataset-files/${pid}/video-links`;
const img=(pid:number,rel:string)=>`${API_BASE}${base(pid)}/file?rel=${encodeURIComponent(rel)}`;
const key=(m:{job:string;folder:string})=>`${m.job}/${m.folder}`;

function Strip({pid,files}:{pid:number;files:string[]}){
 return <Flex gap="1" style={{overflowX:'auto'}}>{files.slice(0,5).map(f=><img key={f} src={img(pid,f)} alt="" loading="lazy" style={{width:64,height:64,objectFit:'cover',borderRadius:4,flex:'0 0 auto'}}/>)}</Flex>;
}

export default function CharacterLinkPanel({projectId,onCreated}:{projectId:number;onCreated:(projectId:number,images:number)=>void}){
 const [res,setRes]=useState<Result|null>(null),[groups,setGroups]=useState<Group[]>([]),[picked,setPicked]=useState<Set<string>>(new Set()),[name,setName]=useState('');
 const [busy,setBusy]=useState(false),[error,setError]=useState(''),[created,setCreated]=useState<string[]>([]);
 const load=useCallback(async()=>{try{setRes(await apiGet<Result>(`${base(projectId)}/result`));setGroups((await apiGet<{groups:Group[]}>(`${base(projectId)}/groups`)).groups);}catch(e){setError(e instanceof Error?e.message:String(e));}},[projectId]);
 useEffect(()=>{void load();},[load]);
 useEffect(()=>{if(!res?.state.running)return;const t=setInterval(()=>void load(),4000);return()=>clearInterval(t);},[res?.state.running,load]);
 async function compute(){setError('');try{await apiPost(`${base(projectId)}/compute`,{});await load();}catch(e){setError(e instanceof Error?e.message:String(e));}}
 const toggle=(ks:string[],on:boolean)=>setPicked(p=>{const n=new Set(p);ks.forEach(k=>on?n.add(k):n.delete(k));return n;});
 const byKey=useMemo(()=>new Map(groups.map(g=>[key(g),g])),[groups]);
 const chosen=[...picked].map(k=>byKey.get(k)).filter(Boolean) as Group[];
 const totalImages=chosen.reduce((a,g)=>a+g.images,0);
 async function create(){
  setBusy(true);setError('');
  try{const r=await apiPost<{project_id:number;name:string;images:number;groups:number}>(`${base(projectId)}/compose`,{name:name.trim(),members:chosen.map(g=>({job_id:g.job,folder:g.folder}))},'POST',undefined,300000);
   setCreated(c=>[...c,`${r.name}（${r.groups}グループ・${r.images}枚）`]);setPicked(new Set());setName('');onCreated(r.project_id,r.images);}
  catch(e){setError(e instanceof Error?e.message:String(e));}finally{setBusy(false);}
 }
 const freq=(c:Cluster)=>c.images*(1+0.25*((c.videos??1)-1));
 const autoClusters=(res?.auto?.clusters??[]).filter(c=>c.groups>1).slice().sort((a,b)=>freq(b)-freq(a));
 const autoKeys=new Set(autoClusters.flatMap(c=>c.members.map(key)));
 const suggestOnly=(res?.suggest?.clusters??[]).filter(c=>c.groups>1&&c.members.some(m=>!autoKeys.has(key(m))));
 const tierLabel:Record<string,[string,'green'|'blue'|'amber']>={recommended:['LoRA推奨','green'],candidate:['候補','blue'],few:['枚数少なめ','amber']};
 const card=(c:Cluster,tag:string,color:'green'|'amber')=><div key={`${tag}${c.id}`} style={{border:'1px solid var(--gray-6)',borderRadius:8,padding:10,minWidth:0}}>
  <Flex gap="2" align="center" wrap="wrap"><Badge color={color}>{tag}</Badge>{c.tier&&<Badge color={tierLabel[c.tier][1]} variant="soft">{tierLabel[c.tier][0]}</Badge>}<Text size="2" weight="medium">{c.images}枚・{c.videos??'?'}本の動画・{c.groups}グループ</Text><Text size="1" color="gray">{c.hair||'髪色不明'} / {c.eyes||'瞳色不明'}</Text>
   <Button size="1" variant="soft" onClick={()=>toggle(c.members.map(key),true)}>この候補をすべて選ぶ</Button></Flex>
  {c.instances&&c.instances.length>0&&<div style={{marginTop:8}}><Text as="div" size="1" color="gray" mb="1">服装（インスタンス）の候補：{c.instances.length}種類</Text>
   <Flex gap="2" style={{overflowX:'auto',paddingBottom:4}}>{c.instances.map(o=><div key={o.label} style={{flex:'0 0 auto',width:118}}><Flex gap="1">{o.sample.map(f=><img key={f} src={img(projectId,f)} alt={o.label} loading="lazy" style={{width:57,height:57,objectFit:'cover',borderRadius:4}}/>)}</Flex>
    <Text as="div" size="1" style={{overflow:'hidden',textOverflow:'ellipsis',whiteSpace:'nowrap'}} title={o.label}>{o.images}枚・{o.label}</Text></div>)}</Flex></div>}
  <Flex direction="column" gap="2" mt="2">{c.members.map(m=><Flex key={key(m)} gap="2" align="center" wrap="wrap" style={{minWidth:0}}>
   <Checkbox checked={picked.has(key(m))} onCheckedChange={v=>toggle([key(m)],v===true)} aria-label={`${m.video} ${m.folder}を選ぶ`}/>
   <Text size="1" style={{width:150,overflowWrap:'anywhere'}}>{m.video}<br/><span style={{color:'var(--gray-10)'}}>{m.images}枚・{m.folder.slice(8,34)}</span></Text><Strip pid={projectId} files={m.sample}/></Flex>)}</Flex></div>;

 return <div>
  <Text as="p" size="2">複数の動画に分かれて出てきた同じキャラを集めて、1つのLoRA（新規プロジェクト）にまとめます。自動の候補を初期値にして、足し引きできます。各グループの衣装フォルダは、そのキャラの服装として引き継がれます。</Text>
  <Flex gap="3" align="center" wrap="wrap" mt="2"><Button disabled={res?.state.running} onClick={()=>void compute()}>{res?.state.running?<Spinner/>:null}{res?.auto?'結合候補を計算し直す':'結合候補を自動で計算する'}</Button>
   <Text size="1" color="gray">{res?.state.running?'計算中…（画像の特徴を比べています）':res?.state.error?`失敗: ${res.state.error}`:''}</Text></Flex>
  {res?.auto&&<Text as="p" size="1" color="gray" mt="2">測定：{res.auto.metrics.groups}グループ（{res.auto.metrics.jobs}本）→ 確かな結合 {res.auto.metrics.multi_group_clusters}件。同じフレームの別人を結合した数 {res.auto.metrics.merged_conflicts_must_be_0}（0であること）。結合内の髪色一致率 {res.auto.metrics.hair_agreement_inside_merged_clusters??'—'}。</Text>}
  {error&&<Callout.Root color="red" size="1" mt="2"><Callout.Text>{error}</Callout.Text></Callout.Root>}
  {created.length>0&&<Callout.Root color="green" size="1" mt="2"><Callout.Text>作成しました：{created.join('、')}。プロジェクト一覧から開けます。</Callout.Text></Callout.Root>}

  <div style={{position:'sticky',top:0,background:'var(--color-panel-solid)',zIndex:2,padding:'8px 0',borderBottom:'1px solid var(--gray-6)',marginTop:12}}>
   <Flex gap="2" align="center" wrap="wrap"><Text size="2" weight="medium">選択中 {chosen.length}グループ・{totalImages}枚</Text>
    <TextField.Root size="1" style={{width:220}} placeholder="新しいキャラの名前" aria-label="新しいキャラの名前" value={name} onChange={e=>setName(e.target.value)}/>
    <Button disabled={!chosen.length||!name.trim()||busy} onClick={()=>void create()}>{busy?<Spinner/>:null}1つのキャラとして新規プロジェクトを作る</Button>
    <Button size="1" variant="ghost" color="gray" disabled={!picked.size} onClick={()=>setPicked(new Set())}>選択を解除</Button></Flex></div>

  {autoClusters.length>0&&<><Text as="div" size="3" weight="bold" mt="3">LoRAにするキャラの候補（動画をまたいで結合・枚数と出現の多い順）</Text>
   <div style={{display:'grid',gap:8,marginTop:6}}>{autoClusters.map(c=>card(c,'確か','green'))}</div></>}
  {suggestOnly.length>0&&<><Text as="div" size="3" weight="bold" mt="3">要確認の候補（基準を緩めて増えた分）</Text>
   <Text as="p" size="1" color="gray">似た色の別キャラが混ざることがあります。画像を見て、選ぶものだけにチェックを入れてください。</Text>
   <div style={{display:'grid',gap:8,marginTop:6}}>{suggestOnly.map(c=>card(c,'要確認','amber'))}</div></>}

  <details style={{marginTop:14}}><summary style={{cursor:'pointer'}}><Text size="2" weight="medium">すべてのグループから手で選ぶ（{groups.length}）</Text></summary>
   <div style={{display:'grid',gap:6,marginTop:8}}>{groups.map(g=><Flex key={key(g)} gap="2" align="center" wrap="wrap" style={{minWidth:0}}>
    <Checkbox checked={picked.has(key(g))} onCheckedChange={v=>toggle([key(g)],v===true)} aria-label={`${g.video} ${g.folder}を選ぶ`}/>
    <Text size="1" style={{width:150,overflowWrap:'anywhere'}}>{g.video}<br/><span style={{color:'var(--gray-10)'}}>{g.images}枚・{g.folder.slice(8,34)}</span></Text><Strip pid={projectId} files={g.sample}/></Flex>)}</div></details>
 </div>;
}
