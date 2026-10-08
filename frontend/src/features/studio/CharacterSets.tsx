import {useCallback,useEffect,useMemo,useRef,useState} from 'react';
import {createPortal} from 'react-dom';
import {Button,Select,Spinner,Text} from '@radix-ui/themes';
import {Plus,Trash2,Undo2} from 'lucide-react';
import {API_BASE,apiGet,apiPost} from '../../lib/api';
import './character-sets.css';

// Merge editor for the characters found in a folder of videos.
// Select images (click / ctrl / shift / drag-box free), drag them onto a character, the bin or "+"; drag a character onto
// another to merge; Delete = to the bin; Ctrl+Z = undo; double-click = enlarge; double-click a name = rename.
type Source={folder:string;name:string;videos:number;set_id:string;characters:number|null;computing:boolean};
type Char={id:number;name:string;count:number;cover:string[]};
type SetState={set_id:string;status:'ready'|'computing'|'error';error?:string;folder?:string;videos?:number;characters?:Char[];excluded?:number;can_undo?:boolean};
const IMG='application/x-lora-images',CHAR='application/x-lora-character';
const LAST='charsets:last-folder';

export default function CharacterSets(){
 const [sources,setSources]=useState<Source[]>([]),[folder,setFolder]=useState<string>(()=>{try{return localStorage.getItem(LAST)||'';}catch{return '';}});
 const [set,setSet]=useState<SetState|null>(null),[current,setCurrent]=useState<string>('');
 const [images,setImages]=useState<string[]>([]),[sel,setSel]=useState<Set<string>>(new Set()),[anchor,setAnchor]=useState<number>(-1);
 const [over,setOver]=useState<string>(''),[zoom,setZoom]=useState<number|null>(null),[renaming,setRenaming]=useState<number|null>(null);
 const [error,setError]=useState(''),[busy,setBusy]=useState(false);
 const sid=set?.set_id;
 const base=sid?`/character-sets/${encodeURIComponent(sid)}`:'';
 const thumb=(rel:string)=>`${API_BASE}${base}/thumb?s=192&rel=${encodeURIComponent(rel)}`;
 const full=(rel:string)=>`${API_BASE}${base}/file?rel=${encodeURIComponent(rel)}`;

 useEffect(()=>{apiGet<{sources:Source[]}>('/character-sets/sources',30000).then(r=>{setSources(r.sources);if(!folder&&r.sources[0])setFolder(r.sources[0].folder);}).catch(e=>setError(String(e)));},[]);
 const loadSet=useCallback(async(id:string)=>{const s=await apiGet<SetState>(`/character-sets/${encodeURIComponent(id)}`,30000);setSet(s);return s;},[]);
 useEffect(()=>{if(!folder)return;try{localStorage.setItem(LAST,folder);}catch{}
  setSet(null);setCurrent('');setImages([]);
  apiPost<{set_id:string}>('/character-sets/open',{folder},'POST',undefined,60000).then(r=>loadSet(r.set_id)).catch(e=>setError(String(e)));},[folder,loadSet]);
 useEffect(()=>{if(set?.status!=='computing')return;const t=setInterval(()=>void loadSet(set.set_id),5000);return()=>clearInterval(t);},[set?.status,set?.set_id,loadSet]);
 useEffect(()=>{if(set?.status==='ready'&&!current&&set.characters?.[0])setCurrent(String(set.characters[0].id));},[set,current]);
 const loadImages=useCallback(async()=>{if(!sid||!current)return;const r=await apiGet<{images:string[]}>(`${base}/images?character=${current}`,30000);setImages(r.images);setSel(new Set());setAnchor(-1);},[sid,current,base]);
 useEffect(()=>{void loadImages().catch(e=>setError(String(e)));},[loadImages]);

 async function act(path:string,body:unknown){if(!sid)return;setBusy(true);setError('');
  try{await apiPost(`${base}${path}`,body,'POST',undefined,60000);const s=await loadSet(sid);
   if(current!=='excluded'&&!s.characters?.some(c=>String(c.id)===current))setCurrent(String(s.characters?.[0]?.id??''));else await loadImages();}
  catch(e){setError(e instanceof Error?e.message:String(e));}finally{setBusy(false);}}
 const move=(imgs:string[],to:string)=>{if(imgs.length&&to!==current)void act('/move',{images:imgs,to});};
 const undo=()=>void act('/undo',{});

 function click(e:React.MouseEvent,i:number){const rel=images[i];
  if(e.shiftKey&&anchor>=0){const [a,b]=[Math.min(anchor,i),Math.max(anchor,i)];setSel(new Set(images.slice(a,b+1)));return;}
  if(e.ctrlKey||e.metaKey){setSel(s=>{const n=new Set(s);n.has(rel)?n.delete(rel):n.add(rel);return n;});setAnchor(i);return;}
  setSel(new Set([rel]));setAnchor(i);}
 function dragImages(e:React.DragEvent,i:number){const rel=images[i];const list=sel.has(rel)?[...sel]:[rel];if(!sel.has(rel))setSel(new Set([rel]));
  e.dataTransfer.setData(IMG,JSON.stringify(list));e.dataTransfer.effectAllowed='move';}
 function drop(e:React.DragEvent,to:string){e.preventDefault();setOver('');
  const imgs=e.dataTransfer.getData(IMG);if(imgs){move(JSON.parse(imgs),to);return;}
  const ch=e.dataTransfer.getData(CHAR);if(ch&&to!=='excluded'&&to!=='new'&&ch!==to)void act('/merge',{source:Number(ch),target:Number(to)});}
 const target=(to:string)=>({onDragOver:(e:React.DragEvent)=>{e.preventDefault();setOver(to);},onDragLeave:()=>setOver(o=>o===to?'':o),onDrop:(e:React.DragEvent)=>drop(e,to)});

 const keyRef=useRef<(e:KeyboardEvent)=>void>(()=>{});
 keyRef.current=(e:KeyboardEvent)=>{
  if(e.target instanceof Element&&e.target.closest('input,textarea,[role=combobox],[role=listbox]'))return;
  if(zoom!==null){if(e.key==='Escape')setZoom(null);if(e.key==='ArrowRight')setZoom(z=>z===null?z:(z+1)%images.length);if(e.key==='ArrowLeft')setZoom(z=>z===null?z:(z-1+images.length)%images.length);return;}
  if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='z'){e.preventDefault();undo();}
  else if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='a'){e.preventDefault();setSel(new Set(images));}
  else if((e.key==='Delete'||e.key==='Backspace')&&sel.size){e.preventDefault();move([...sel],'excluded');}};
 useEffect(()=>{const k=(e:KeyboardEvent)=>keyRef.current(e);window.addEventListener('keydown',k);return()=>window.removeEventListener('keydown',k);},[]);

 const chars=set?.characters??[];
 const src=sources.find(s=>s.folder===folder);
 const currentName=useMemo(()=>current==='excluded'?'除外':chars.find(c=>String(c.id)===current)?.name??'',[current,chars]);
 return <div className="cs-root">
  <div className="cs-bar">
   <Select.Root value={folder||undefined} onValueChange={setFolder}><Select.Trigger aria-label="動画フォルダ" placeholder="動画フォルダ"/>
    <Select.Content>{sources.map(s=><Select.Item key={s.folder} value={s.folder}>{s.name}（{s.videos}本）</Select.Item>)}</Select.Content></Select.Root>
   {set?.status==='ready'&&<Text size="1" color="gray">{chars.length}キャラ</Text>}
   <span className="cs-grow"/>
   {busy&&<Spinner/>}
   <Button size="1" variant="soft" color="gray" disabled={!set?.can_undo||busy} onClick={undo} title="元に戻す（Ctrl+Z）"><Undo2 size={14}/></Button>
  </div>
  {error&&<Text as="div" size="1" color="red" className="cs-error">{error}</Text>}
  {!set||set.status==='computing'?<div className="cs-wait"><Spinner/><Text size="2" color="gray">{src?`${src.name}（${src.videos}本）`:''}のキャラを解析しています</Text></div>
  :set.status==='error'?<div className="cs-wait"><Text size="2" color="red">{set.error}</Text></div>
  :<div className="cs-body">
   <div className="cs-list">
    {chars.map(c=><div key={c.id} className={`cs-char${String(c.id)===current?' on':''}${over===String(c.id)?' over':''}`} draggable
      onDragStart={e=>{e.dataTransfer.setData(CHAR,String(c.id));e.dataTransfer.effectAllowed='move';}}
      onClick={()=>setCurrent(String(c.id))} {...target(String(c.id))}>
     {c.cover[0]&&<img src={thumb(c.cover[0])} alt="" draggable={false}/>}
     {renaming===c.id?<input autoFocus defaultValue={c.name} onBlur={e=>{setRenaming(null);if(e.target.value.trim()&&e.target.value!==c.name)void act('/rename',{id:c.id,name:e.target.value});}}
       onKeyDown={e=>{if(e.key==='Enter')(e.target as HTMLInputElement).blur();if(e.key==='Escape')setRenaming(null);}}/>
      :<span className="cs-name" onDoubleClick={e=>{e.stopPropagation();setRenaming(c.id);}}>{c.name}</span>}
     <span className="cs-count">{c.count}</span></div>)}
   </div>
   <div className="cs-side">
    <div className={`cs-drop${over==='new'?' over':''}`} {...target('new')} title="ここにドロップで新しいキャラ"><Plus size={16}/></div>
    <div className={`cs-drop${over==='excluded'?' over':''}${current==='excluded'?' on':''}`} {...target('excluded')} onClick={()=>setCurrent('excluded')} title="除外（Delete）"><Trash2 size={16}/><span>{set.excluded}</span></div>
   </div>
   <div className="cs-grid" onClick={e=>{if(e.target===e.currentTarget)setSel(new Set());}}>
    <Text as="div" size="1" color="gray" className="cs-grid-head">{currentName}・{images.length}{sel.size?`（${sel.size}選択）`:''}</Text>
    {images.map((rel,i)=><img key={rel} src={thumb(rel)} alt="" loading="lazy" draggable className={sel.has(rel)?'sel':''}
      onClick={e=>click(e,i)} onDoubleClick={()=>setZoom(i)} onDragStart={e=>dragImages(e,i)}/>)}
   </div>
  </div>}
  {zoom!==null&&images[zoom]&&createPortal(<div className="cs-zoom" onClick={e=>{if(e.target===e.currentTarget)setZoom(null);}}>
   <img src={full(images[zoom])} alt=""/><Text size="1">{zoom+1} / {images.length}</Text></div>,document.body)}
 </div>;
}
