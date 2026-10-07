import {useEffect,useMemo,useRef,useState} from 'react';
import {Button,Popover,TextField,Text,IconButton} from '@radix-ui/themes';
import {Check,ChevronDown,RefreshCw,Search} from 'lucide-react';
import './editor-components.css';
export type ModelChoice={name:string;path:string;size_mb?:number;source?:string};
type Props={label?:string;value:string;choices:ModelChoice[];onChange:(v:string)=>void;onRefresh:()=>Promise<void>};
const normalized=(s:string)=>s.replace(/\\/g,'/').toLowerCase();
export default function ModelPicker({value,choices,onChange,onRefresh,label="ベースモデル"}:Props){
 const [open,setOpen]=useState(false),[query,setQuery]=useState(''),[active,setActive]=useState(0),[refreshing,setRefreshing]=useState(false),[error,setError]=useState('');
 const input=useRef<HTMLInputElement>(null),list=useRef<HTMLDivElement>(null);
 const unique=useMemo(()=>Array.from(new Map(choices.map(c=>[normalized(c.path),c])).values()),[choices]);
 const shown=useMemo(()=>{const q=query.trim().toLowerCase();return unique.filter(c=>`${c.name} ${c.path} ${c.source||''}`.toLowerCase().includes(q));},[query,unique]);
 useEffect(()=>setActive(0),[query]);
 useEffect(()=>{list.current?.querySelector(`[data-model-index="${active}"]`)?.scrollIntoView({block:'nearest'});},[active]);
 const select=(path:string)=>{onChange(path);setOpen(false);};
 return <div className="model-field"><Text as="label" size="1" color="gray">{label}</Text><Popover.Root open={open} onOpenChange={v=>{setOpen(v);if(v){setQuery('');setActive(0);}}}>
  <Popover.Trigger><Button variant="surface" color="gray" className="model-trigger" aria-label={`${label}を一覧から選択`} title={value}><span>{unique.find(c=>normalized(c.path)===normalized(value))?.name||(value?value.split(/[\\/]/).pop():'モデルを選ぶ')}</span><ChevronDown size={15}/></Button></Popover.Trigger>
  <Popover.Content className="model-picker" align="end" sideOffset={5} onOpenAutoFocus={e=>{e.preventDefault();input.current?.focus();}}>
   <div className="model-search"><TextField.Root ref={input} aria-label="モデルを検索" placeholder="名前・フォルダで検索" value={query} onChange={e=>setQuery(e.target.value)} role="combobox" aria-controls="model-options" aria-expanded={true} aria-activedescendant={shown[active]?`model-choice-${active}`:undefined} onKeyDown={e=>{if(e.nativeEvent.isComposing)return;if(e.key==='ArrowDown'||e.key==='ArrowUp'){e.preventDefault();setActive(n=>Math.max(0,Math.min(shown.length-1,n+(e.key==='ArrowDown'?1:-1))));}if(e.key==='Enter'&&shown[active]){e.preventDefault();select(shown[active].path);}}}><TextField.Slot><Search size={15}/></TextField.Slot></TextField.Root><IconButton aria-label="モデルを再スキャン" title="モデルを再スキャン" variant="soft" disabled={refreshing} onClick={async()=>{setRefreshing(true);setError('');try{await onRefresh();}catch(e){setError(String(e));}finally{setRefreshing(false);}}}><RefreshCw size={15}/></IconButton></div>
   <div className="model-count">{shown.length} / {unique.length}件</div>
   {error&&<Text as="p" color="red" size="1">{error}</Text>}
   <div id="model-options" role="listbox" aria-label="モデル候補" className="model-options" ref={list}>{shown.map((c,i)=>{const selected=normalized(value)===normalized(c.path);const parts=c.path.replace(/\\/g,'/').split('/');const folder=parts.slice(-3,-1).join(' / ');return <button type="button" role="option" aria-selected={selected} id={`model-choice-${i}`} data-model-index={i} className={`model-option ${i===active?'is-active':''}`} key={c.path} title={c.path} onMouseEnter={()=>setActive(i)} onClick={()=>select(c.path)}><span className="model-check">{selected&&<Check size={16}/>}</span><span className="model-info"><span className="model-name">{c.name}</span><span className="model-location">{folder}</span></span><span className="model-size">{c.size_mb===undefined?'':c.size_mb>=1024?`${(c.size_mb/1024).toFixed(1)} GB`:`${Math.round(c.size_mb)} MB`}</span></button>})}{!shown.length&&<p className="model-empty">一致するモデルがありません。パスを直接指定できます。</p>}</div>
   {shown[active]&&<div className="model-detail" aria-label="候補の保存場所">{shown[active].path}</div>}
  </Popover.Content>
 </Popover.Root></div>;
}
