import {useCallback,useEffect,useMemo,useState} from 'react';
import {Badge,Button,Callout,Checkbox,Flex,Spinner,Text,TextField} from '@radix-ui/themes';
import {API_BASE,apiGet,apiPost} from '../../lib/api';

// 複数の動画をまたいで「同じキャラ」のグループを集め、1つのLoRA（新規プロジェクト）にまとめる。
// 自動の候補（確かなもの／要確認）が初期値で、人が足し引きして作成する。
type LinkMember={job:string;video:string;folder:string;images:number;outfits:string[];sample:string[]};
type Instance={label:string;images:number;sample:string[]};
type Cluster={id:number;groups:number;images:number;hair:string;eyes:string;members:LinkMember[];instances?:Instance[];videos?:number;video_names?:string[];tier?:'recommended'|'candidate'|'few'};
type Metrics={groups:number;jobs:number;clusters:number;multi_group_clusters:number;merged_conflicts_must_be_0:number;hair_agreement_inside_merged_clusters:number|null;merge_bar:number};
type PropInstance={label:string;images:number;videos:number;sample:string[]};
type PropGroup={job:string;folder:string;video:string;images:number;sample:string[];why?:string};
type Proposal={cluster:number;name:string;images:number;videos:number;video_names:string[];groups:number;hair:string;eyes:string;members:PropGroup[];near:PropGroup[];instances:PropInstance[];character_only_images:number;min_instance_images:number};

// One group (one character of one video) with its thumbnails and an include checkbox: the unit the user judges.
function ReviewRow({pid,g,on,onChange}:{pid:number;g:PropGroup;on:boolean;onChange:(v:boolean)=>void}){
 return <Flex gap="2" align="center" style={{minWidth:0,opacity:on?1:.55}}>
  <Checkbox checked={on} onCheckedChange={v=>onChange(v===true)} aria-label={`${g.video} ${g.folder}を含める`}/>
  <Text size="1" style={{width:130,flex:'0 0 auto',overflowWrap:'anywhere'}}>{g.video}<br/><span style={{color:'var(--gray-10)'}}>{g.images}枚{g.why?`・${g.why}`:''}</span></Text>
  <Flex gap="1" style={{overflowX:'auto'}}>{g.sample.map(f=><img key={f} src={img(pid,f)} alt="" loading="lazy" style={{width:72,height:72,objectFit:'cover',borderRadius:4,flex:'0 0 auto'}}/>)}</Flex></Flex>;
}
type Result={state:{running:boolean;error:string;log:string};auto?:{clusters:Cluster[];metrics:Metrics};suggest?:{clusters:Cluster[];metrics:Metrics};proposals?:Proposal[]};

// One ready-to-make LoRA: who, how many images, and which instances (same outfit across videos = one instance).
function ProposalCard({pid,p,rank,similar,busy,onCreate,onAdjust}:{pid:number;p:Proposal;rank:number;similar:(Proposal&{rank:number})[];busy:boolean;onCreate:(name:string,members:{job:string;folder:string}[])=>void;onAdjust:(members:{job:string;folder:string}[])=>void}){
 const [name,setName]=useState(p.name),[withSimilar,setWithSimilar]=useState(false);
 const [off,setOff]=useState<Set<string>>(new Set()),[added,setAdded]=useState<Set<string>>(new Set()),[saved,setSaved]=useState('');
 const all=[...p.members,...(withSimilar?similar.flatMap(s=>s.members):[]),...p.near.filter(g=>added.has(key(g)))];
 const uniq=[...new Map(all.map(g=>[key(g),g])).values()];
 const members=uniq.filter(g=>!off.has(key(g)));
 const images=members.reduce((a,g)=>a+g.images,0);
 const flip=(set:Set<string>,setter:(s:Set<string>)=>void,k:string,v:boolean)=>{const n=new Set(set);v?n.add(k):n.delete(k);setter(n);setSaved('');};
 async function saveJudgement(){
  try{await apiPost(`${base(pid)}/feedback`,{cluster:p.cluster,kept:members.map(key),excluded:p.members.filter(g=>off.has(key(g))).map(key),added:[...added]});setSaved(`評価を保存しました（除外 ${p.members.filter(g=>off.has(key(g))).length}・追加 ${added.size}）`);}
  catch(e){setSaved(e instanceof Error?e.message:String(e));}
 }
 return <div style={{border:'1px solid var(--gray-6)',borderRadius:8,padding:10,minWidth:0}}>
  <Flex gap="2" align="center" wrap="wrap"><Badge color="green">提案 {rank}</Badge><Text size="2" weight="bold">{p.images}枚・{p.videos}本の動画に登場</Text><Text size="1" color="gray">{p.hair||'髪色不明'} / {p.eyes||'瞳色不明'}・{p.groups}グループ</Text></Flex>
  <Text as="div" size="1" color="gray" mt="1">インスタンス（衣装ごとのトリガー）{p.instances.length}種類・どれにも入らない {p.character_only_images}枚はキャラ本体のトリガーだけで学習（{p.min_instance_images}枚未満の衣装・衣装不明）</Text>
  <Flex gap="2" mt="2" style={{overflowX:'auto',paddingBottom:4}}>{p.instances.map(o=><div key={o.label} style={{flex:'0 0 auto',width:178}}>
   <Flex gap="1">{o.sample.map(f=><img key={f} src={img(pid,f)} alt={o.label} loading="lazy" style={{width:57,height:57,objectFit:'cover',borderRadius:4}}/>)}</Flex>
   <Text as="div" size="1" style={{overflow:'hidden',textOverflow:'ellipsis',whiteSpace:'nowrap'}} title={o.label}>{o.label}</Text>
   <Text as="div" size="1" color="gray">{o.images}枚・{o.videos}本</Text></div>)}
   {!p.instances.length&&<Text size="1" color="gray">衣装ごとに十分な枚数がないため、キャラ本体のみのLoRAになります。</Text>}</Flex>
  {similar.length>0&&<Flex gap="2" align="center" mt="1"><Checkbox checked={withSimilar} onCheckedChange={v=>setWithSimilar(v===true)} aria-label="似た候補もまとめる"/>
   <Text size="1">髪と瞳が同じ候補（提案 {similar.map(s=>s.rank).join('・')}、計 {similar.reduce((a,s)=>a+s.images,0)}枚）も同じキャラとしてまとめる</Text></Flex>}
  <details style={{marginTop:8}}><summary style={{cursor:'pointer'}}><Text size="2" weight="medium">中身を確認・調整（含まれる {uniq.length}グループ／近いが入っていない {p.near.length}グループ）</Text></summary>
   <Text as="p" size="1" color="gray" mt="1">別キャラが混ざっていたらチェックを外し、同じキャラなのに入っていないものはチェックを入れてください。この判断は結合の精度を測る正解として保存され、自動の結合の改善に使います。</Text>
   <Text as="div" size="2" weight="bold" mt="2">含まれるグループ（別キャラなら外す）</Text>
   <Flex direction="column" gap="2" mt="1">{uniq.map(g=><ReviewRow key={key(g)} pid={pid} g={g} on={!off.has(key(g))} onChange={v=>flip(off,setOff,key(g),!v)}/>)}</Flex>
   {p.near.length>0&&<><Text as="div" size="2" weight="bold" mt="3">近いが入っていないグループ（同じキャラなら入れる）</Text>
    <Flex direction="column" gap="2" mt="1">{p.near.filter(g=>!added.has(key(g))).map(g=><ReviewRow key={key(g)} pid={pid} g={g} on={false} onChange={v=>flip(added,setAdded,key(g),v)}/>)}</Flex></>}
   <Flex gap="2" align="center" mt="2"><Button size="1" variant="soft" onClick={()=>void saveJudgement()}>この判断を評価として保存</Button><Text size="1" color="gray">{saved}</Text></Flex>
  </details>
  <Flex gap="2" align="center" wrap="wrap" mt="2">
   <TextField.Root size="1" style={{width:200}} aria-label="新しいキャラの名前" value={name} onChange={e=>setName(e.target.value)}/>
   <Button size="1" disabled={busy||!name.trim()||!members.length} onClick={()=>{void saveJudgement();onCreate(name.trim(),members);}}>{busy?<Spinner/>:null}この内容で新規プロジェクトを作る（{images}枚・{members.length}グループ）</Button>
   <Button size="1" variant="soft" color="gray" onClick={()=>onAdjust(members)}>下で足し引きする</Button></Flex>
 </div>;
}
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
 const [loading,setLoading]=useState(true),[groupsOpen,setGroupsOpen]=useState(false);
 // candidates alone are enough to pick from; the full group list is fetched only when "pick by hand" is opened
 const load=useCallback(async()=>{try{setRes(await apiGet<Result>(`${base(projectId)}/result`,60000));}catch(e){setError(e instanceof Error?e.message:String(e));}finally{setLoading(false);}},[projectId]);
 const loadGroups=useCallback(async()=>{try{setGroups((await apiGet<{groups:Group[]}>(`${base(projectId)}/groups`,60000)).groups);}catch(e){setError(e instanceof Error?e.message:String(e));}},[projectId]);
 useEffect(()=>{void load();},[load]);
 useEffect(()=>{if(groupsOpen&&!groups.length)void loadGroups();},[groupsOpen,groups.length,loadGroups]);
 useEffect(()=>{if(!res?.state.running)return;const t=setInterval(()=>void load(),4000);return()=>clearInterval(t);},[res?.state.running,load]);
 async function compute(){setError('');try{await apiPost(`${base(projectId)}/compute`,{});await load();}catch(e){setError(e instanceof Error?e.message:String(e));}}
 const toggle=(ks:string[],on:boolean)=>setPicked(p=>{const n=new Set(p);ks.forEach(k=>on?n.add(k):n.delete(k));return n;});
 const byKey=useMemo(()=>{
  const m=new Map<string,Group>();
  for(const tag of ['auto','suggest'] as const)for(const c of res?.[tag]?.clusters??[])for(const g of c.members)m.set(key(g),g);
  for(const g of groups)m.set(key(g),g);
  return m;
 },[res,groups]);
 const chosen=[...picked].map(k=>byKey.get(k)).filter(Boolean) as Group[];
 const totalImages=chosen.reduce((a,g)=>a+g.images,0);
 async function compose(newName:string,members:{job:string;folder:string}[]){
  setBusy(true);setError('');
  try{const r=await apiPost<{project_id:number;name:string;images:number;groups:number;instances:{label:string;images:number;trigger:string}[]}>(`${base(projectId)}/compose`,{name:newName,members:members.map(g=>({job_id:g.job,folder:g.folder}))},'POST',undefined,300000);
   setCreated(c=>[...c,`${r.name}（${r.groups}グループ・${r.images}枚・インスタンス${r.instances?.length??0}種類）`]);setPicked(new Set());setName('');onCreated(r.project_id,r.images);}
  catch(e){setError(e instanceof Error?e.message:String(e));}finally{setBusy(false);}
 }
 const create=()=>compose(name.trim(),chosen);
 const proposals=(res?.proposals??[]).map((p,i)=>({...p,rank:i+1}));
 const sameLook=(a:Proposal,b:Proposal)=>!!a.hair&&!!a.eyes&&a.hair===b.hair&&a.eyes===b.eyes;
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

  {proposals.length>0&&<><Text as="div" size="3" weight="bold" mt="3">LoRAの提案（登場の多いキャラ順）</Text>
   <Text as="p" size="1" color="gray">登場の多いキャラごとに、作るLoRAのデータセット案です。同じ衣装は動画をまたいで1つのインスタンスにまとめます。そのまま作るか、「下で足し引きする」で調整してください。</Text>
   <div style={{display:'grid',gap:8,marginTop:6}}>{proposals.map(p=><ProposalCard key={p.cluster} pid={projectId} p={p} rank={p.rank} busy={busy}
    similar={proposals.filter(o=>o.cluster!==p.cluster&&sameLook(o,p))}
    onCreate={(n,m)=>void compose(n,m)} onAdjust={m=>{setPicked(new Set(m.map(key)));setName(p.name);}}/>)}</div></>}
  {autoClusters.length>0&&<><Text as="div" size="3" weight="bold" mt="4">動画をまたいだ結合の一覧（調整用）</Text>
   <div style={{display:'grid',gap:8,marginTop:6}}>{autoClusters.map(c=>card(c,'確か','green'))}</div></>}
  {suggestOnly.length>0&&<><Text as="div" size="3" weight="bold" mt="3">要確認の候補（基準を緩めて増えた分）</Text>
   <Text as="p" size="1" color="gray">似た色の別キャラが混ざることがあります。画像を見て、選ぶものだけにチェックを入れてください。</Text>
   <div style={{display:'grid',gap:8,marginTop:6}}>{suggestOnly.map(c=>card(c,'要確認','amber'))}</div></>}

  {loading&&<Flex gap="2" align="center" mt="3"><Spinner/><Text size="2" color="gray">結合の結果を読み込んでいます…</Text></Flex>}
  {!loading&&res&&!res.auto&&!res.state.running&&<Callout.Root size="1" mt="3"><Callout.Text>まだ結合を計算していません。「結合候補を自動で計算する」を押すと、取り込んだすべての動画をまとめて比べます。</Callout.Text></Callout.Root>}
  <details style={{marginTop:14}} onToggle={e=>setGroupsOpen((e.currentTarget as HTMLDetailsElement).open)}><summary style={{cursor:'pointer'}}><Text size="2" weight="medium">すべてのグループから手で選ぶ（{groups.length||res?.auto?.metrics.groups||0}）</Text></summary>
   {groupsOpen&&!groups.length&&<Flex gap="2" align="center" mt="2"><Spinner/><Text size="1" color="gray">一覧を読み込んでいます…</Text></Flex>}
   <div style={{display:'grid',gap:6,marginTop:8}}>{groups.map(g=><Flex key={key(g)} gap="2" align="center" wrap="wrap" style={{minWidth:0}}>
    <Checkbox checked={picked.has(key(g))} onCheckedChange={v=>toggle([key(g)],v===true)} aria-label={`${g.video} ${g.folder}を選ぶ`}/>
    <Text size="1" style={{width:150,overflowWrap:'anywhere'}}>{g.video}<br/><span style={{color:'var(--gray-10)'}}>{g.images}枚・{g.folder.slice(8,34)}</span></Text><Strip pid={projectId} files={g.sample}/></Flex>)}</div></details>
 </div>;
}
