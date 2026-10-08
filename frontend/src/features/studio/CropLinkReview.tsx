import {useEffect,useState} from 'react';
import {Badge,Button,Callout,Checkbox,Flex,Spinner,Text} from '@radix-ui/themes';
import {API_BASE,apiGet,apiPost} from '../../lib/api';

// Review of the image-level (CCIP only) character clusters: the user judges the result here.
// Click an image = enlarge (handled by the surrounding panel); "×" = not this character; "同じキャラ" on a near cluster = merge.
type Near={id:number;dist:number;rank:number|null;images:number;videos:number;hair:string;eyes:string;sample:string[]};
type Prop={id:number;rank:number;images:number;videos:number;hair:string;eyes:string;hair_agreement:number|null;from_unassigned:number;outfits:[string,number][];
 by_video:{job:string;video:string;count:number;sample:string[]}[];hair_disagrees:string[];nearest:Near[]};
type Res={available:boolean;t?:number;crops?:number;videos?:number;clusters?:number;clusters_20_plus?:number;proposals?:Prop[]};

const base=(pid:number)=>`/dataset-files/${pid}/video-links`;
const src=(pid:number,rel:string)=>`${API_BASE}${base(pid)}/file?rel=${encodeURIComponent(rel)}`;

function Tile({pid,rel,off,onToggle}:{pid:number;rel:string;off:boolean;onToggle:()=>void}){
 return <div style={{position:'relative',flex:'0 0 auto'}}>
  <img src={src(pid,rel)} alt="" loading="lazy" style={{width:96,height:96,objectFit:'cover',borderRadius:4,display:'block',outline:off?'3px solid var(--red-9)':'none',opacity:off?.5:1,cursor:'zoom-in'}}/>
  <button type="button" title={off?'別キャラの印を外す':'別キャラとして印を付ける'} aria-label="別キャラ" data-zoom-ignore
   onClick={e=>{e.stopPropagation();onToggle();}}
   style={{position:'absolute',top:2,right:2,width:22,height:22,borderRadius:11,border:'none',cursor:'pointer',fontSize:13,lineHeight:'22px',padding:0,
    background:off?'var(--red-9)':'rgba(0,0,0,.55)',color:'#fff'}}>×</button></div>;
}

function Card({pid,p}:{pid:number;p:Prop}){
 const [off,setOff]=useState<Set<string>>(new Set()),[same,setSame]=useState<Set<number>>(new Set()),[saved,setSaved]=useState('');
 const flip=(rel:string)=>{setOff(s=>{const n=new Set(s);n.has(rel)?n.delete(rel):n.add(rel);return n;});setSaved('');};
 async function save(){
  try{await apiPost(`${base(pid)}/feedback`,{cluster:p.id,kept:[],excluded:[...off],added:[...same].map(i=>`cluster:${i}`),note:'crop-level review'});
   setSaved(`保存しました（別キャラ ${off.size}枚・同じキャラの塊 ${same.size}件）`);}catch(e){setSaved(e instanceof Error?e.message:String(e));}
 }
 const agree=p.hair_agreement==null?'—':`${Math.round(p.hair_agreement*100)}%`;
 return <div style={{border:'1px solid var(--gray-6)',borderRadius:8,padding:10,minWidth:0}}>
  <Flex gap="2" align="center" wrap="wrap"><Badge color="green">提案 {p.rank}</Badge><Text size="2" weight="bold">{p.images}枚・{p.videos}本の動画</Text>
   <Text size="1" color="gray">多数派 {p.hair||'髪色不明'} / {p.eyes||'瞳色不明'}・髪色タグ一致 {agree}・仕分け不能から {p.from_unassigned}枚</Text></Flex>
  <Text as="div" size="1" color="gray" mt="1">衣装フォルダ: {p.outfits.map(([k,v])=>`${k||'衣装なし・仕分け不能'}（${v}）`).join(' / ')||'—'}</Text>
  <Text as="div" size="2" weight="bold" mt="2">この塊の画像（動画ごと・最大16枚） <Text size="1" color="gray" weight="regular">別キャラには「×」</Text></Text>
  <Flex direction="column" gap="2" mt="1">{p.by_video.map(v=><Flex key={v.job} gap="2" align="start" style={{minWidth:0}}>
   <Text size="1" style={{width:110,flex:'0 0 auto'}}><b>{v.video}</b><br/><span style={{color:'var(--gray-10)'}}>{v.count}枚</span></Text>
   <div data-zoomset style={{display:'flex',flexWrap:'wrap',gap:4,minWidth:0}}>{v.sample.map(r=><Tile key={r} pid={pid} rel={r} off={off.has(r)} onToggle={()=>flip(r)}/>)}</div></Flex>)}</Flex>
  {p.hair_disagrees.length>0&&<><Text as="div" size="2" weight="bold" mt="3">髪色タグが多数派と違う画像（{p.hair_disagrees.length}枚） <Text size="1" color="gray" weight="regular">混入か、変身などで髪色が変わった同じキャラか</Text></Text>
   <div data-zoomset style={{display:'flex',flexWrap:'wrap',gap:4,marginTop:4}}>{p.hair_disagrees.map(r=><Tile key={r} pid={pid} rel={r} off={off.has(r)} onToggle={()=>flip(r)}/>)}</div></>}
  {p.nearest.length>0&&<><Text as="div" size="2" weight="bold" mt="3">近い別の塊 <Text size="1" color="gray" weight="regular">同じキャラなら「同じキャラ」</Text></Text>
   <Flex direction="column" gap="2" mt="1">{p.nearest.map(n=><Flex key={n.id} gap="2" align="start" style={{minWidth:0}}>
    <Flex direction="column" gap="1" style={{width:110,flex:'0 0 auto'}}><Text size="1"><b>{n.images}枚・{n.videos}本</b><br/><span style={{color:'var(--gray-10)'}}>{n.hair} / {n.eyes}<br/>距離 {n.dist}{n.rank?`・提案 ${n.rank}`:''}</span></Text>
     <Text as="label" size="1"><Checkbox checked={same.has(n.id)} onCheckedChange={v=>{setSame(s=>{const x=new Set(s);v===true?x.add(n.id):x.delete(n.id);return x;});setSaved('');}}/> 同じキャラ</Text></Flex>
    <div data-zoomset style={{display:'flex',flexWrap:'wrap',gap:4,minWidth:0}}>{n.sample.map(r=><img key={r} src={src(pid,r)} alt="" loading="lazy" style={{width:96,height:96,objectFit:'cover',borderRadius:4,cursor:'zoom-in'}}/>)}</div></Flex>)}</Flex></>}
  <Flex gap="2" align="center" mt="3"><Button size="1" onClick={()=>void save()}>この評価を保存</Button><Text size="1" color="gray">{saved}</Text></Flex>
 </div>;
}

export default function CropLinkReview({projectId}:{projectId:number}){
 const [res,setRes]=useState<Res|null>(null),[error,setError]=useState('');
 useEffect(()=>{apiGet<Res>(`${base(projectId)}/crop-result`,60000).then(setRes).catch(e=>setError(e instanceof Error?e.message:String(e)));},[projectId]);
 if(error)return <Callout.Root color="red" size="1" mt="2"><Callout.Text>{error}</Callout.Text></Callout.Root>;
 if(!res)return <Flex gap="2" align="center" mt="2"><Spinner/><Text size="2" color="gray">画像単位の結合結果を読み込んでいます…</Text></Flex>;
 if(!res.available)return <Text as="p" size="1" color="gray" mt="2">画像単位の結合結果はまだありません。</Text>;
 return <div>
  <Text as="div" size="3" weight="bold" mt="2">LoRAの提案（画像単位・キャラ判別モデルのみ）</Text>
  <Text as="p" size="1" color="gray">{res.crops}枚（仕分け不能を含む）を1枚ずつ、キャラ判別専用モデル（CCIP）の距離だけでまとめました（しきい値 {res.t}、同じ画面の2人は別人。髪・瞳のタグは判定に不使用）。20枚以上の塊 {res.clusters_20_plus}件のうち、上位{res.proposals?.length}件。画像はクリックで拡大（← → で前後、Esc で閉じる）。</Text>
  <div style={{display:'grid',gap:8,marginTop:6}}>{res.proposals?.map(p=><Card key={p.id} pid={projectId} p={p}/>)}</div>
 </div>;
}
