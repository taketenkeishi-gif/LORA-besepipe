import {Fragment,useCallback,useEffect,useLayoutEffect,useMemo,useRef,useState} from 'react';
import {createPortal} from 'react-dom';
import {Button,Checkbox,Select,Spinner,Text} from '@radix-ui/themes';
import {Plus,Trash2,Undo2} from 'lucide-react';
import {API_BASE,apiGet,apiPost} from '../../lib/api';
import './character-sets.css';

// ② Characters of a video workspace.  Left: characters (tick = make a LoRA) and the pending piles.  Right:
//  - a character: its images by outfit (a suggestion),
//  - a pending pile: rows to compare - each row with the character it most likely belongs to and that character's reference shots,
//  - the bin: a plain grid.
// Everywhere: click / Ctrl / Shift / Ctrl+A / drag a box from empty space to select; drag onto a character, "+" or the bin;
// Delete = bin; Ctrl+Z = undo; double-click = enlarge; double-click a name = rename.
export type Char={id:number;name:string;count:number;cover:string[];pending?:boolean;section?:string};
export type SetState={set_id:string;status:'ready'|'computing'|'error';error?:string;folder?:string;set_folder?:string;characters?:Char[];excluded?:number;can_undo?:boolean};
type Outfit={name:string;files:string[]};
type Row={dest:number|null;video:string;files:string[];score:number};
type Rows={rows:Row[];refs:Record<string,string[]>;characters:{id:number;name:string}[]};
type Box={x0:number;y0:number;x1:number;y1:number;base:Set<string>};
const IMG='application/x-lora-images',CHAR='application/x-lora-character';

export default function CharacterEditor({set,reload,picked,setPicked}:{set:SetState;reload:()=>Promise<SetState>;picked:Set<number>;setPicked:(f:(s:Set<number>)=>Set<number>)=>void}){
 const [current,setCurrent]=useState<string>('');
 const [images,setImages]=useState<string[]>([]),[sel,setSel]=useState<Set<string>>(new Set()),[anchor,setAnchor]=useState<string>('');
 const [outfits,setOutfits]=useState<Outfit[]|null>(null),[rows,setRows]=useState<Rows|null>(null),[rowDest,setRowDest]=useState<Record<number,string>>({});
 const [over,setOver]=useState(''),[zoom,setZoom]=useState<number|null>(null),[renaming,setRenaming]=useState<number|null>(null);
 const [error,setError]=useState(''),[busy,setBusy]=useState(false),[notice,setNotice]=useState('');
 const base=`/character-sets/${encodeURIComponent(set.set_id)}`;
 const thumb=(rel:string)=>`${API_BASE}${base}/thumb?s=192&rel=${encodeURIComponent(rel)}`;
 const full=(rel:string)=>`${API_BASE}${base}/file?rel=${encodeURIComponent(rel)}`;
 const chars=set.characters??[];
 const here=chars.find(c=>String(c.id)===current);
 const kind=current==='excluded'?'bin':here?.pending?'pile':'character';
 const nameOf=(id:string)=>id==='excluded'?'除外':id==='new'?'新しいキャラ':chars.find(c=>String(c.id)===id)?.name??'';
 useEffect(()=>{if(!current&&chars[0])setCurrent(String(chars[0].id));else if(current&&current!=='excluded'&&!chars.some(c=>String(c.id)===current)&&chars[0])setCurrent(String(chars[0].id));},[chars,current]);

 // what the right side shows, loaded per kind; an older reply never overwrites a newer choice
 const want=useRef('');
 const load=useCallback(async()=>{if(!current)return;const key=`${set.set_id}|${current}|${chars.find(c=>String(c.id)===current)?.count}`;want.current=key;
  try{const r=await apiGet<{images:string[]}>(`${base}/images?character=${current}`,30000);if(want.current!==key)return;setImages(r.images);
   if(current!=='excluded'&&!chars.find(c=>String(c.id)===current)?.pending){const o=await apiGet<{outfits:Outfit[]}>(`${base}/outfits?character=${current}`,60000);if(want.current===key){setOutfits(o.outfits);setRows(null);}}
   else if(current!=='excluded'){const r2=await apiGet<Rows>(`${base}/rows?character=${current}`,60000);if(want.current===key){setRows(r2);setOutfits(null);}}
   else{setOutfits(null);setRows(null);}}
  catch(e){setError(e instanceof Error?e.message:String(e));}},[current,base,set.set_id,chars]);
 // the selection is cleared only when another character is opened; a reload after an edit keeps what is still there
 useEffect(()=>{void load();},[load]);
 useEffect(()=>{setImages([]);setOutfits(null);setRows(null);setNotice('');setRowDest({});setSel(new Set());setAnchor('');},[current]);

 // display order (for Shift ranges and the enlarged view): outfits / rows / plain list
 const order=useMemo(()=>outfits?outfits.flatMap(o=>o.files):rows?rows.rows.flatMap(r=>r.files):images,[outfits,rows,images]);
 const indexOf=useMemo(()=>new Map(order.map((r,i)=>[r,i])),[order]);
 useEffect(()=>{setSel(s=>{if(!s.size)return s;const n=new Set([...s].filter(r=>indexOf.has(r)));return n.size===s.size?s:n;});},[indexOf]);

 async function act(path:string,body:unknown,done?:string){setBusy(true);setError('');
  try{await apiPost(`${base}${path}`,body,'POST',undefined,60000);await reload();if(done)setNotice(done);}
  catch(e){setError(e instanceof Error?e.message:String(e));}finally{setBusy(false);}}
 const move=(imgs:string[],to:string)=>{if(imgs.length&&to!==current)void act('/move',{images:imgs,to},to==='excluded'?`${imgs.length}枚を除外しました`:`${imgs.length}枚を${nameOf(to)}へ移しました`);};
 const undo=()=>void act('/undo',{},'元に戻しました');

 function click(e:React.MouseEvent,rel:string){
  if(e.shiftKey&&anchor&&indexOf.has(anchor)){const [a,b]=[indexOf.get(anchor)!,indexOf.get(rel)!].sort((x,y)=>x-y);const range=order.slice(a,b+1);
   setSel(s=>e.ctrlKey||e.metaKey?new Set([...s,...range]):new Set(range));return;}
  if(e.ctrlKey||e.metaKey){setSel(s=>{const n=new Set(s);n.has(rel)?n.delete(rel):n.add(rel);return n;});setAnchor(rel);return;}
  setSel(new Set([rel]));setAnchor(rel);}
 function dragImages(e:React.DragEvent,rel:string){const list=sel.has(rel)?[...sel]:[rel];if(!sel.has(rel))setSel(new Set([rel]));
  e.dataTransfer.setData(IMG,JSON.stringify(list));e.dataTransfer.effectAllowed='move';}
 function drop(e:React.DragEvent,to:string){e.preventDefault();setOver('');
  const imgs=e.dataTransfer.getData(IMG);if(imgs){move(JSON.parse(imgs),to);return;}
  const ch=e.dataTransfer.getData(CHAR);if(ch&&to!=='excluded'&&to!=='new'&&ch!==to)void act('/merge',{source:Number(ch),target:Number(to)},`${nameOf(ch)}を${nameOf(to)}に結合しました`);}
 const target=(to:string)=>({onDragOver:(e:React.DragEvent)=>{e.preventDefault();e.dataTransfer.dropEffect='move';setOver(to);},onDragLeave:()=>setOver(o=>o===to?'':o),onDrop:(e:React.DragEvent)=>drop(e,to)});

 const keyRef=useRef<(e:KeyboardEvent)=>void>(()=>{});
 keyRef.current=(e:KeyboardEvent)=>{
  if(e.target instanceof Element&&e.target.closest('input,textarea,[role=combobox],[role=listbox],[role=option]'))return;
  if(zoom!==null){if(e.key==='Escape')setZoom(null);if(e.key==='ArrowRight')setZoom(z=>z===null?z:(z+1)%order.length);if(e.key==='ArrowLeft')setZoom(z=>z===null?z:(z-1+order.length)%order.length);return;}
  if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='z'){e.preventDefault();undo();}
  else if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='a'){e.preventDefault();setSel(new Set(order));}
  else if(e.key==='Escape'&&sel.size)setSel(new Set());
  else if((e.key==='Delete'||e.key==='Backspace')&&sel.size){e.preventDefault();move([...sel],'excluded');}};
 useEffect(()=>{const k=(e:KeyboardEvent)=>keyRef.current(e);window.addEventListener('keydown',k);return()=>window.removeEventListener('keydown',k);},[]);

 // drag a box from empty space: every image it touches is selected (Ctrl/Shift adds)
 const paneRef=useRef<HTMLDivElement>(null);
 const [box,setBox]=useState<Box|null>(null);
 const boxed=useRef(false);
 function boxStart(e:React.MouseEvent){if(e.button!==0||(e.target as Element).closest('img,button,input,select,[role=combobox]'))return;e.preventDefault();
  boxed.current=false;setBox({x0:e.clientX,y0:e.clientY,x1:e.clientX,y1:e.clientY,base:e.ctrlKey||e.metaKey||e.shiftKey?new Set(sel):new Set()});}
 useEffect(()=>{if(!box)return;
  const moveBox=(e:MouseEvent)=>{setBox(b=>{if(!b)return b;const nb={...b,x1:e.clientX,y1:e.clientY};
    const [l,r,t,bt]=[Math.min(nb.x0,nb.x1),Math.max(nb.x0,nb.x1),Math.min(nb.y0,nb.y1),Math.max(nb.y0,nb.y1)];
    if(r-l>4||bt-t>4){boxed.current=true;const hit=new Set(nb.base);
     paneRef.current?.querySelectorAll<HTMLImageElement>('img[data-rel]').forEach(im=>{const q=im.getBoundingClientRect();if(q.right>l&&q.left<r&&q.bottom>t&&q.top<bt)hit.add(im.dataset.rel!);});
     setSel(hit);}
    return nb;});};
  const end=()=>setBox(null);
  window.addEventListener('mousemove',moveBox);window.addEventListener('mouseup',end);
  return()=>{window.removeEventListener('mousemove',moveBox);window.removeEventListener('mouseup',end);};},[box!==null]);

 const tile=(rel:string)=><img key={rel} data-rel={rel} src={thumb(rel)} alt="" loading="lazy" draggable className={sel.has(rel)?'sel':''}
  onClick={e=>click(e,rel)} onDoubleClick={()=>setZoom(indexOf.get(rel)??0)} onDragStart={e=>dragImages(e,rel)}/>;

 // the bin: plain grid, only the rows in view exist (it can hold thousands)
 const [view,setView]=useState({top:0,w:0,h:0});
 useLayoutEffect(()=>{const g=paneRef.current;if(!g)return;const read=()=>setView(v=>({...v,w:g.clientWidth,h:g.clientHeight}));read();const ro=new ResizeObserver(read);ro.observe(g);return()=>ro.disconnect();},[]);
 useLayoutEffect(()=>{const g=paneRef.current;if(g){g.scrollTop=0;setView({top:0,w:g.clientWidth,h:g.clientHeight});}},[current]);
 const win=useMemo(()=>{const PAD=8,GAP=4,MIN=124,HEAD=34;const inner=Math.max(MIN,view.w-PAD*2);
  const cols=Math.max(1,Math.floor((inner+GAP)/(MIN+GAP)));const row=(inner-GAP*(cols-1))/cols+GAP;const rowsN=Math.ceil(images.length/cols);
  if(!view.h)return {from:0,to:Math.min(images.length,cols*8),before:0,after:0};
  const r0=Math.max(0,Math.floor((view.top-HEAD)/row)-2),r1=Math.min(rowsN,Math.ceil((view.top+view.h)/row)+2);
  return {from:r0*cols,to:Math.min(images.length,r1*cols),before:r0?r0*row-GAP:0,after:r1<rowsN?(rowsN-r1)*row-GAP:0};},[view,images.length]);

 const mains=chars.filter(c=>!c.pending);
 const head=<div className="ce-head">
  <Text size="2" weight="bold">{current==='excluded'?'除外':here?.name}</Text>
  <Text size="1" color="gray">{order.length}枚{sel.size?`・${sel.size}枚選択中`:''}</Text>
  {kind==='character'&&here&&<Text as="label" size="1"><Checkbox checked={picked.has(here.id)} onCheckedChange={v=>setPicked(s=>{const n=new Set(s);v===true?n.add(here.id):n.delete(here.id);return n;})}/> LoRAにする</Text>}
  <span className="cs-grow"/>
  {busy&&<Spinner/>}
  {notice&&<Text size="1" color="green">{notice}</Text>}
  <Button size="1" variant="soft" color="gray" disabled={!set.can_undo||busy} onClick={undo} title="元に戻す（Ctrl+Z）"><Undo2 size={14}/></Button>
 </div>;

 return <div className="ce-root">
  <div className="cs-list">
   {chars.map((c,k)=><Fragment key={c.id}>{c.pending&&!chars[k-1]?.pending&&<div className="cs-section">保留</div>}
    <div className={`cs-char${c.pending?' held':''}${String(c.id)===current?' on':''}${over===String(c.id)?' over':''}`} draggable
     onDragStart={e=>{e.dataTransfer.setData(CHAR,String(c.id));e.dataTransfer.effectAllowed='move';}}
     onClick={()=>setCurrent(String(c.id))} {...target(String(c.id))}>
     {!c.pending&&<span onClick={e=>e.stopPropagation()}><Checkbox aria-label={`${c.name}をLoRAにする`} checked={picked.has(c.id)} onCheckedChange={v=>setPicked(s=>{const n=new Set(s);v===true?n.add(c.id):n.delete(c.id);return n;})}/></span>}
     {c.cover[0]&&<img src={thumb(c.cover[0])} alt="" loading="lazy" draggable={false}/>}
     {renaming===c.id?<input autoFocus defaultValue={c.name} onBlur={e=>{setRenaming(null);if(e.target.value.trim()&&e.target.value!==c.name)void act('/rename',{id:c.id,name:e.target.value});}}
       onKeyDown={e=>{if(e.key==='Enter')(e.target as HTMLInputElement).blur();if(e.key==='Escape')setRenaming(null);}}/>
      :<span className="cs-name" onDoubleClick={e=>{e.stopPropagation();setRenaming(c.id);}}>{c.name}</span>}
     <span className="cs-count">{c.count}</span></div></Fragment>)}
  </div>
  <div className="cs-side">
   <div className={`cs-drop${over==='new'?' over':''}`} {...target('new')} title="ここにドロップで新しいキャラ"><Plus size={16}/></div>
   <div className={`cs-drop${over==='excluded'?' over':''}${current==='excluded'?' on':''}`} {...target('excluded')} onClick={()=>setCurrent('excluded')} title="除外（Delete）"><Trash2 size={16}/><span>{set.excluded}</span></div>
  </div>
  <div ref={paneRef} className={`ce-pane${sel.size?' has-sel':''}`} onMouseDown={boxStart}
   onScroll={e=>{const t=e.currentTarget.scrollTop;if(kind==='bin')setView(v=>v.top===t?v:{...v,top:t});}}
   onClick={e=>{if(!boxed.current&&!(e.target as Element).closest('img,button,input,label,[role=combobox]'))setSel(new Set());boxed.current=false;}}>
   {head}
   {error&&<Text as="div" size="1" color="red">{error}</Text>}
   {kind==='character'&&(outfits?outfits.map(o=><section key={o.name} className="ce-outfit">
     <Text as="div" size="1" weight="bold" className="ce-outfit-name">{o.name}<span>{o.files.length}</span></Text>
     <div className="ce-grid">{o.files.map(tile)}</div></section>)
    :<div className="ce-wait"><Spinner/></div>)}
   {kind==='pile'&&(rows?rows.rows.map((r,k)=>{const dest=rowDest[k]??(r.dest!=null?String(r.dest):'');const destName=nameOf(dest);
     return <section key={k} className="ce-row">
      <div className="ce-row-dest">
       <Select.Root value={dest||undefined} onValueChange={v=>setRowDest(d=>({...d,[k]:v}))}>
        <Select.Trigger placeholder="帰属先" aria-label="帰属先"/>
        <Select.Content>{mains.map(c=><Select.Item key={c.id} value={String(c.id)}>{c.name}</Select.Item>)}</Select.Content></Select.Root>
       <div className="ce-refs">{(rows.refs[dest]??(dest?chars.find(c=>String(c.id)===dest)?.cover:[])??[]).map(f=><img key={f} src={thumb(f)} alt="" loading="lazy" draggable={false} onDoubleClick={()=>window.open(full(f))}/>)}</div>
       <div className="ce-row-actions">
        <Button size="1" disabled={busy||!dest} onClick={()=>move(r.files.filter(f=>!sel.has(f)),dest)}>{sel.size&&r.files.some(f=>sel.has(f))?'選択以外を':'この行を'}{destName||'…'}へ</Button>
        <Button size="1" variant="soft" color="red" disabled={busy} onClick={()=>move(r.files,'excluded')}>この行を除外</Button></div>
       <Text size="1" color="gray">{r.video}・{r.files.length}枚</Text>
      </div>
      <div className="ce-grid">{r.files.map(tile)}</div></section>;})
    :<div className="ce-wait"><Spinner/></div>)}
   {kind==='bin'&&<div className="ce-grid">
     {win.before>0&&<div className="cs-spacer" style={{height:win.before}}/>}
     {images.slice(win.from,win.to).map(tile)}
     {win.after>0&&<div className="cs-spacer" style={{height:win.after}}/>}</div>}
  </div>
  {box&&boxed.current&&createPortal(<div className="cs-box" style={{left:Math.min(box.x0,box.x1),top:Math.min(box.y0,box.y1),width:Math.abs(box.x1-box.x0),height:Math.abs(box.y1-box.y0)}}/>,document.body)}
  {zoom!==null&&order[zoom]&&createPortal(<div className="cs-zoom" onClick={e=>{if(e.target===e.currentTarget)setZoom(null);}}>
   <img src={full(order[zoom])} alt=""/><Text size="1">{zoom+1} / {order.length}</Text></div>,document.body)}
 </div>;
}
