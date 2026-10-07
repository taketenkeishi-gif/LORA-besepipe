import {useEffect,useRef,useState} from 'react';
import {Button,Callout,Dialog,Flex,Spinner,Text} from '@radix-ui/themes';

// ご自身のカスタムノード（SDXL_tag）のタグエディターを、そのままこのアプリに埋め込む。
// 部品は /vendor/sdxl-tag-editor/ にコピーしてあり、素のDOMで動く（ComfyUI本体に依存しない）。
type Editor={getJSON():string;setJSON(json:string):void;destroy?:()=>void};
type EditorCtor=new(root:HTMLElement,onChange:(json:string)=>void,getGraphNodes?:()=>unknown[],getHostNode?:()=>unknown)=>Editor;
type Chip={raw?:string;type?:string;bypassed?:boolean};

const splitTags=(text:string)=>text.split(/[,\n]/).map(t=>t.trim()).filter(Boolean);
export function stateFromPrompts(positive:string,negative:string){
 return JSON.stringify({prefix:[],mainTags:splitTags(positive).map(raw=>({raw})),suffix:[],blocklist:[],negative:splitTags(negative).map(raw=>({raw})),stash:[],characters:[]});
}
const norm=(t:string)=>t.trim().toLowerCase().replace(/\s+/g,' ');
const raws=(list?:Chip[])=>(list??[]).filter(c=>c&&c.type!=='divider'&&!c.bypassed&&c.raw&&c.raw.trim()).map(c=>c.raw!.trim());
/** エディターの状態 → 実際に使うプロンプト（先頭・本文・末尾をつなぎ、ブロックリストのタグを除く） */
export function promptsFromState(json:string){
 const s=JSON.parse(json) as {prefix?:Chip[];mainTags?:Chip[];suffix?:Chip[];blocklist?:Chip[];negative?:Chip[]};
 const block=new Set(raws(s.blocklist).map(norm));
 const seen=new Set<string>();const positive=[...raws(s.prefix),...raws(s.mainTags),...raws(s.suffix)].filter(t=>{const k=norm(t);if(block.has(k)||seen.has(k))return false;seen.add(k);return true;});
 return {positive:positive.join(', '),negative:raws(s.negative).join(', ')};
}

export default function TagEditorDialog({open,positive,negative,onApply,onClose}:{open:boolean;positive:string;negative:string;onApply:(positive:string,negative:string)=>void;onClose:()=>void}){
 const host=useRef<HTMLDivElement|null>(null),editor=useRef<Editor|null>(null),latest=useRef('');
 const [error,setError]=useState(''),[ready,setReady]=useState(false);
 useEffect(()=>{
  if(!open)return;let live=true;setError('');setReady(false);
  const timer=setTimeout(async()=>{
   try{
    const url='/vendor/sdxl-tag-editor/editor.js';
    const nativeImport=new Function('u','return import(u)') as (u:string)=>Promise<unknown>; // バンドラーを通さず、コピーしたファイルをそのまま読む
    const mod=await nativeImport(url) as {TagEditor:EditorCtor};
    if(!live||!host.current)return;
    host.current.innerHTML='';
    const ed=new mod.TagEditor(host.current,json=>{latest.current=json;},()=>[],()=>null);
    const initial=stateFromPrompts(positive,negative);ed.setJSON(initial);latest.current=initial;editor.current=ed;setReady(true);
   }catch(e){if(live)setError(e instanceof Error?e.message:String(e));}
  },50);
  return()=>{live=false;clearTimeout(timer);try{editor.current?.destroy?.();}catch{/* 破棄に失敗しても画面には影響しない */}editor.current=null;};
 },[open]);// eslint-disable-line react-hooks/exhaustive-deps
 function apply(){try{const json=editor.current?.getJSON()??latest.current;const r=promptsFromState(json);onApply(r.positive,r.negative);onClose();}catch(e){setError(e instanceof Error?e.message:String(e));}}
 return <Dialog.Root open={open} onOpenChange={v=>{if(!v)onClose();}}><Dialog.Content maxWidth="min(1100px,96vw)" style={{padding:12}}>
  <Flex justify="between" align="center" gap="3" wrap="wrap"><Dialog.Title size="3" style={{margin:0}}>タグエディター</Dialog.Title>
   <Flex gap="2"><Button variant="soft" color="gray" onClick={onClose}>キャンセル</Button><Button disabled={!ready} onClick={apply}>この内容をプロンプトに反映</Button></Flex></Flex>
  <Dialog.Description size="1" color="gray" mb="2">ComfyUIのカスタムノード（SDXL_tag）のタグエディターと同じ部品です。ドラッグで並べ替え、右クリックで重みや無効化ができます。</Dialog.Description>
  {error&&<Callout.Root color="red" size="1" mb="2"><Callout.Text>{error}</Callout.Text></Callout.Root>}
  {!ready&&!error&&<Flex align="center" gap="2"><Spinner/><Text size="1">読み込み中…</Text></Flex>}
  <div ref={host} style={{minHeight:560,maxHeight:'78vh',overflow:'auto'}}/>
 </Dialog.Content></Dialog.Root>;
}
