import DatasetImageViewer from './DatasetImageViewer';
import CaptionEditor,{splitTags,uniqueTags} from './CaptionEditor';
import PreprocessDialog from './PreprocessDialog';
import BatchNamingDialog from './BatchNamingDialog';
import AutoTagDialog from './AutoTagDialog';
import {desktop,desktopError} from './desktopBridge';
import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import VideoImportDialog from './VideoImportDialog';
import { Film, FolderOpen, Folder, FolderPlus, Upload, RefreshCw, Save, Search, ArrowUp, Grid2X2, List, Check, ImagePlus, FileText, SlidersHorizontal, ClipboardPaste } from 'lucide-react';
import { API_BASE, apiPost, apiGet } from '../../lib/api';
import type { PreparedDataset } from './workbenchTypes';
import type { Project } from '../../types';
import { Button, IconButton, TextField, TextArea, Select, SegmentedControl, Dialog, Popover, Flex, Text, Badge, Card, Checkbox, Tooltip, Spinner, ContextMenu, DropdownMenu } from '@radix-ui/themes';
import './dataset-files.css';
import {useDatasetDrag,insertBlock,type ImportFile} from './useDatasetDrag';

type Item = {requires_training_copy?:boolean;protected_source?:boolean;name:string; relative:string; path:string; caption_path:string; caption:string; revision:string; width:number; height:number; bytes:number; modified:number; asset?:{id:number}|null};
type Listing = {root:string;directory:string;folder:string;folders:{name:string;relative:string}[];items:Item[];errors:string[];order_revision:string};
type Draft = {text:string;revision:string};
type FolderStat = {relative:string;images:number;training_targets:number;direct:number;direct_targets:number};
type Summary = {snapshot_id:number|null;folders:FolderStat[];root:FolderStat;totals:{images:number;training_targets:number;folders:number;folders_with_targets:number}};
export type ToolsContext = {busy:boolean;prepareAll:()=>Promise<boolean>;summaryText:string;summaryColor:'gray'|'green'|'amber'};
const tone = (s?:{images:number;training_targets:number}):'gray'|'green'|'amber' => !s||!s.images||!s.training_targets?'gray':s.training_targets>=s.images?'green':'amber';
type Props = {project:Project;active?:boolean;openRootRequest?:number;showError:(s:string)=>void;showNotice:(s:string)=>void;onReview:()=>void;trainingPaths?:string[];extraTools?:ReactNode|((ctx:ToolsContext)=>ReactNode);onPrepared?:(v:PreparedDataset)=>void;onDatasetChanged?:()=>void};
const read = (key:string, fallback:any) => {try{return JSON.parse(localStorage.getItem(key)||'null') ?? fallback;}catch{return fallback;}};

export default function DatasetFiles({project,active=true,openRootRequest=0,onReview,trainingPaths,extraTools,onPrepared,onDatasetChanged}:Props) {
  const key = `dataset-files:${project.id}`;
  const [folder,setFolder] = useState<string>(()=>(window.location.pathname.match(/\/project\/(\d+)/)?.[1]===String(project.id)?new URLSearchParams(window.location.search).get('folder'):null) || read(key,{folder:''}).folder||'');
  const [data,setData] = useState<Listing|null>(null);
  const [selected,setSelected] = useState<string[]>(()=>read(`${key}:selection`,[]));
  const [anchor,setAnchor] = useState('');
  const [query,setQuery] = useState<string>(()=>read(`${key}:query`,''));
  const [sort,setSort] = useState<string>(()=>read(key,{sort:'manual'}).sort||'manual');
  const [undoOrder,setUndoOrder] = useState<{folder:string;order:string[];revision:string}|null>(null);
  const [onlyEmpty,setOnlyEmpty] = useState(false);
  const [onlyMissingTrigger,setOnlyMissingTrigger]=useState(false);
  const [editorWidth,setEditorWidth]=useState<number>(()=>Math.max(240,Math.min(680,Number(read('dataset-editor-width',330))||330)));
  const resizeStart=useRef<{x:number;width:number}|null>(null);
  const resizeEditor=(v:number)=>setEditorWidth(Math.max(240,Math.min(680,window.innerWidth-470,v)));
  useEffect(()=>{localStorage.setItem('dataset-editor-width',JSON.stringify(editorWidth));},[editorWidth]);
  const [displayOpen,setDisplayOpen] = useState(false);
  const [view,setView] = useState<string>(()=>read(key,{view:'grid'}).view||'grid');
  const [size,setSize] = useState<number>(()=>read(key,{size:180}).size||180);
  const [drafts,setDrafts] = useState<Record<string,Draft>>(()=>read(`${key}:drafts`,{}));
  const [busy,setBusy] = useState(false);
  const [normalizeOpen,setNormalizeOpen]=useState(false);
  const [namingOpen,setNamingOpen]=useState(false);
  const [namingUndo,setNamingUndo]=useState<string>(()=>read(`${key}:naming-undo`, ""));
  const [autoTagOpen,setAutoTagOpen]=useState(false);
  const [webOpen,setWebOpen]=useState(false),[webUrl,setWebUrl]=useState('');
  const [error,setError] = useState('');
  const [message,setMessage] = useState('');
  const [viewing,setViewing]=useState<Item|null>(null);
  const viewerReturn=useRef<HTMLElement|null>(null);
  function openImage(item:Item,opener:HTMLElement){viewerReturn.current=opener;setViewing(item);}
  const [rename,setRename] = useState<string|null>(null);
  const [bulk,setBulk] = useState<string|null>(null);
  const [bulkMode,setBulkMode] = useState('add');
  const [bulkFind,setBulkFind] = useState('');
  const [triggerTags,setTriggerTags] = useState<string[]>([]);
  const [sharedTags,setSharedTags] = useState<string[]>([]);
  const [instances,setInstances] = useState<{name:string;token:string}[]>([]);
  const [bulkUndo,setBulkUndo] = useState<Record<string,{before:Draft|undefined;after:Draft}>|null>(null);
  useEffect(()=>{let alive=true;void apiGet<{concepts:{name:string;trigger_token:string;concept_type:string}[]}>(`/basepipe/projects/${project.id}/concepts`).then(r=>{if(!alive)return;const live=r.concepts.filter(c=>c.trigger_token);setTriggerTags(live.map(c=>c.trigger_token));setSharedTags(live.filter(c=>c.concept_type!=='outfit').map(c=>c.trigger_token));setInstances(live.filter(c=>c.concept_type==='outfit').map(c=>({name:c.name,token:c.trigger_token})).reverse());}).catch(()=>{});return()=>{alive=false;};},[project.id,active]);
  const [lastExcluded,setLastExcluded] = useState<string>(()=>read(`${key}:excluded`,''));
  const [excludeOpen,setExcludeOpen] = useState(false);
  const [videoOpen,setVideoOpen] = useState(false);
  const [newFolder,setNewFolder] = useState<string|null>(null);
  const [folderToDelete,setFolderToDelete] = useState<string|null>(null);
  const [summary,setSummary] = useState<Summary|null>(null);
  const [assignFolder,setAssignFolder] = useState<{folder:string;token:string}|null>(null);
  const summaryRequest = useRef(0);
  async function loadSummary(){
    const seq=++summaryRequest.current;
    try{const next=await apiGet<Summary>(`/dataset-files/${project.id}/summary`);if(seq===summaryRequest.current)setSummary(next);}catch{/* 件数は補助表示。失敗しても編集は続けられる */}
  }
  const statByFolder=useMemo(()=>new Map((summary?.folders||[]).map(f=>[f.relative,f])),[summary]);
  const statText=(s?:FolderStat)=>!s?'':!s.images?'画像なし':`学習対象 ${s.training_targets}/${s.images}枚`;
  const totals=summary?.totals;
  const summaryText=totals?`全体：学習対象 ${totals.training_targets}/${totals.images}枚・フォルダ${totals.folders}個中 ${totals.folders_with_targets}個が学習対象`:'';
  const input = useRef<HTMLInputElement>(null);
  const bulkTrigger = useRef<HTMLButtonElement>(null);
  const excludeTrigger = useRef<HTMLButtonElement>(null);
  const request = useRef(0);
  const fingerprints = useRef<Record<string,string>>({});
  useEffect(()=>{if(openRootRequest){setFolder('');setSelected([]);setQuery('');setOnlyEmpty(false);}},[openRootRequest]);
  const sortedItems=useMemo(()=>[...(data?.items||[])].sort((a,b)=>sort==='manual'?0:sort==='modified'?b.modified-a.modified:sort==='size'?b.bytes-a.bytes:a.name.localeCompare(b.name,'ja',{numeric:true})),[data,sort]);
  // 共通トリガーは全画像に必要。インスタンス（衣装など）は画像ごとにどれか1つあればよい。
  const hasTag=(caption:string,t:string)=>splitTags(caption).some(tag=>tag.toLowerCase()===t.toLowerCase());
  const hasTriggers=(caption:string)=>sharedTags.every(t=>hasTag(caption,t))&&(!instances.length||instances.some(i=>hasTag(caption,i.token)));
  const triggerCount=(data?.items||[]).filter(i=>hasTriggers(i.caption)).length;
  const items=useMemo(()=>sortedItems.filter(i=>(!onlyEmpty||!i.caption.trim())&&(!onlyMissingTrigger||!hasTriggers(i.caption))&&`${i.name} ${i.caption}`.toLowerCase().includes(query.toLowerCase())),[sortedItems,query,onlyEmpty,onlyMissingTrigger,triggerTags]);
  const current = data?.items.find(i=>i.relative===selected[selected.length-1]);
  const draft = current ? drafts[current.relative] : undefined;
  const dirtyCount = Object.keys(drafts).length;
  useEffect(()=>{if(dirtyCount>0)onDatasetChanged?.();},[dirtyCount]);
  const report = (e:unknown) => {const s=e instanceof Error?e.message:String(e); setError(s);};
  const notice = (s:string) => {setMessage(s);};

  async function refresh() {
    const seq=++request.current;
    setBusy(true);setError('');
    try {
      const next=await apiPost<Listing>(`/dataset-files/${project.id}/rescan`,{folder},"POST",undefined,30000);
      if(seq!==request.current)return;
      const fingerprint=JSON.stringify(next.items.map(i=>[i.relative,i.caption,i.modified]));
      if(fingerprints.current[folder] && fingerprints.current[folder]!==fingerprint)onDatasetChanged?.();
      fingerprints.current[folder]=fingerprint;
      setData(next);void loadSummary();
      setSelected(prev=>prev.filter(p=>next.items.some(i=>i.relative===p)));
      setMessage('');
    }catch(e){if(seq===request.current)report(e);}finally{if(seq===request.current)setBusy(false);}
  }
  useEffect(()=>{void refresh();return()=>{request.current++;};},[folder]);
  useEffect(()=>{if(active)void loadSummary();},[active,trainingPaths]);
  useEffect(()=>{localStorage.setItem(key,JSON.stringify({folder,view,size,sort}));},[key,folder,view,size,sort]);
  useEffect(()=>{if(active)window.history.replaceState({},'',`/project/${project.id}/dataset/images${folder?'?folder='+encodeURIComponent(folder):''}`);},[active,folder,project.id]);
  useEffect(()=>{localStorage.setItem(`${key}:selection`,JSON.stringify(selected));localStorage.setItem(`${key}:query`,JSON.stringify(query));},[key,selected,query]);
  useEffect(()=>{localStorage.setItem(`${key}:drafts`,JSON.stringify(drafts));},[drafts,key]);
  useEffect(()=>{
    const warn=(e:BeforeUnloadEvent)=>{if(dirtyCount){e.preventDefault();e.returnValue='';}};
    window.addEventListener('beforeunload',warn);return()=>window.removeEventListener('beforeunload',warn);
  },[dirtyCount]);

  function select(item:Item,e:React.MouseEvent) {
    if(dnd.ignoreClick.current&&e.detail>0)return;
    if(e.shiftKey && anchor) {
      const a=items.findIndex(i=>i.relative===anchor),b=items.indexOf(item);
      if(a>=0){setSelected(items.slice(Math.min(a,b),Math.max(a,b)+1).map(i=>i.relative));return;}
    }
    setSelected(prev=>e.ctrlKey||e.metaKey ? prev.includes(item.relative)?prev.filter(p=>p!==item.relative):[...prev,item.relative]:[item.relative]);
    setAnchor(item.relative);setRename(null);
  }
  function update(text:string) {
    if(!current)return;
    onDatasetChanged?.();
    setDrafts(prev=>({...prev,[current.relative]:{text,revision:prev[current.relative]?.revision||current.revision}}));
  }
  async function save(edits:{relative:string;caption:string;revision:string}[]) {
    setBusy(true);setError('');
    try {
      await apiPost(`/dataset-files/${project.id}/captions`,{edits});
      // Keep the displayed value stable while clearing the saved draft; the rescan supplies new revisions.
      setData(prev=>prev?{...prev,items:prev.items.map(item=>{const edit=edits.find(e=>e.relative===item.relative);return edit?{...item,caption:edit.caption}:item;})}:prev);
      setDrafts(prev=>{const next={...prev};edits.forEach(e=>delete next[e.relative]);return next;});
      await refresh();notice(`${edits.length}件 保存済み`);
      return true;
    }catch(e){report(e);return false;}finally{setBusy(false);}
  }
  async function upload(files:FileList|File[]|ImportFile[],destination=folder,selectImported=false) {
    setBusy(true);setError('');
    try {
      const form=new FormData();Array.from(files as ArrayLike<File|ImportFile>).forEach(f=>{const entry='file' in f?f:{file:f,name:f.webkitRelativePath||f.name};form.append('files',entry.file,entry.name);});
      const response=await fetch(`${API_BASE}/dataset-files/${project.id}/upload?folder=${encodeURIComponent(destination)}`,{method:'POST',body:form});
      const result=await response.json();if(!response.ok)throw Error(result.detail||'追加できませんでした');
      if(destination!==folder){setFolder(destination);setSelected([]);onDatasetChanged?.();}else await refresh();if(!result.saved.length&&result.rejected?.length)throw Error('対応する画像・UTF-8のTXTがありません');if(selectImported&&result.saved.length){setQuery('');setOnlyEmpty(false);setOnlyMissingTrigger(false);setSelected(result.saved.map((name:string)=>destination?`${destination}/${name}`:name));onDatasetChanged?.();}notice(`${result.saved.length}件を追加しました${result.skipped?`（同名・非対応 ${result.skipped}件は上書きせずスキップ）`:''}`);
    }catch(e){report(e);}finally{setBusy(false);if(input.current)input.current.value='';}
  }
  const pasteInFlight=useRef(false);
  async function pasteImages(files:File[],destination=folder){
    if(busy||pasteInFlight.current){notice('追加処理中です。完了後にもう一度貼り付けてください');return;}
    const images=files.filter(f=>f.type.startsWith('image/')||/\.(png|jpe?g|webp|bmp|gif|avif)$/i.test(f.name));
    if(!images.length){notice('クリップボードに画像がありません');return;}
    pasteInFlight.current=true;
    try{const named=images.map(f=>{const ext:Record<string,string>={'image/png':'png','image/jpeg':'jpg','image/webp':'webp','image/bmp':'bmp','image/gif':'gif','image/avif':'avif'};const suffix=ext[f.type]||f.name.split('.').pop()||'png';return new File([f],`clipboard-${new Date().toISOString().replace(/[:.]/g,'-')}-${crypto.randomUUID().slice(0,8)}.${suffix}`,{type:f.type});});await upload(named,destination,true);}finally{pasteInFlight.current=false;}
  }
  async function readClipboardImages(){const destination=folder;try{const entries=await navigator.clipboard.read();const files:File[]=[];for(const item of entries){const type=item.types.find(t=>t.startsWith('image/'));if(type){const blob=await item.getType(type);files.push(new File([blob],'clipboard',{type}));}}await pasteImages(files,destination);}catch{report('クリップボードを読み取れませんでした。ギャラリー上でCtrl+Vを押してください');}}
  useEffect(()=>{
    if(!active)return;
    const paste=(event:ClipboardEvent)=>{
      const target=event.target instanceof Element?event.target:null;
      if(event.defaultPrevented||target?.closest('input,textarea,[contenteditable="true"],[role="textbox"],.tag-list,[role="dialog"]')||document.querySelector('[role="dialog"]'))return;
      const files=Array.from(event.clipboardData?.files||[]);
      if(!files.some(f=>f.type.startsWith('image/')||/\.(png|jpe?g|webp|bmp|gif|avif)$/i.test(f.name)))return;
      event.preventDefault();void pasteImages(files);
    };
    window.addEventListener('paste',paste);return()=>window.removeEventListener('paste',paste);
  },[active,folder,busy]);
  async function undoNaming(){if(!namingUndo)return;setBusy(true);setError("");try{const r=await desktop()!.undoNaming({project_id:project.id,token:namingUndo});setSelected(r.relatives);if(r.folder!==folder)setFolder(r.folder);else await refresh();setNamingUndo("");localStorage.removeItem(`${key}:naming-undo`);onDatasetChanged?.();notice("画像とTXTの名前を戻しました");}catch(e){report(desktopError(e));}finally{setBusy(false);}}
  async function reveal(relative:string) {try{await apiPost(`/dataset-files/${project.id}/reveal`,{relative});notice('エクスプローラーに保存場所を開きました');}catch(e){report(e);}}
  async function importUrl(url:string,destination:string){setBusy(true);setError('');setMessage('');try{await desktop()!.importWebImage({project_id:project.id,folder:destination,url});onDatasetChanged?.();if(destination!==folder){setFolder(destination);setSelected([]);}else await refresh();setWebOpen(false);setWebUrl('');notice('Web画像をPNGとして保存しました');}catch(e){report(desktopError(e));}finally{setBusy(false);}}
  async function applyRename() {
    if(!current||!rename)return;setBusy(true);
    try{const r=await apiPost<{relative:string}>(`/dataset-files/${project.id}/rename`,{relative:current.relative,name:rename});setSelected([r.relative]);setRename(null);await refresh();notice('画像と同名TXTを一緒に変更しました');}catch(e){report(e);}finally{setBusy(false);}
  }
  async function copyForTraining() {
    setBusy(true);try{const r=await apiPost<{folder:string;relatives:string[]}>(`/dataset-files/${project.id}/copy-for-training`,{relatives:selected});setFolder(r.folder);setSelected(r.relatives);onDatasetChanged?.();notice('原本を保持して学習用に複製しました');}catch(e){report(e);}finally{setBusy(false);}
  }
  async function register() {
    setBusy(true);
    try{const r=await apiPost<{asset_ids:number[]}>(`/dataset-files/${project.id}/register`,{relatives:selected});if(onPrepared){
        await apiPost(`/basepipe/projects/${project.id}/assets/bulk-review`,{asset_ids:r.asset_ids,review_status:'approved',training_enabled:true});
        const prepared=await apiPost<PreparedDataset>(`/basepipe/projects/${project.id}/snapshots`,{name:`学習対象 ${new Date().toLocaleString('ja-JP')}`,asset_ids:r.asset_ids},'POST',undefined,60000);
        prepared.entries=(await apiGet<PreparedDataset>(`/basepipe/snapshots/${prepared.id}`)).entries;await refresh();onPrepared(prepared);notice(`${prepared.item_count}枚を学習対象にしました`);
      }else{await refresh();notice(`${r.asset_ids.length}枚を学習素材のレビューに登録しました`);}}catch(e){report(e);}finally{setBusy(false);}
  }
  // 全画像（folder=''）またはフォルダ配下すべてを、登録→承認→1つの確定セットにする。選択は不要。
  async function prepareScope(scope:string):Promise<boolean> {
    if(!onPrepared||busy)return false;
    const dirty=Object.keys(drafts).filter(r=>!scope||r.startsWith(`${scope}/`));
    if(dirty.length){report(`未保存の下書きが${dirty.length}件あります。先に保存するか元に戻してから学習対象にしてください`);return false;}
    setBusy(true);setError('');
    try{
      const prepared=await apiPost<PreparedDataset>(`/dataset-files/${project.id}/prepare-all`,{folder:scope},'POST',undefined,300000);
      await refresh();onPrepared(prepared);
      notice(`${scope?`フォルダ「${scope}」の`:'全'}${prepared.item_count}枚を学習対象にしました`);
      return true;
    }catch(e){report(e);return false;}finally{setBusy(false);}
  }
  async function assignInstanceToFolder() {
    if(!assignFolder||!assignFolder.token)return;
    const {folder:scope,token}=assignFolder;
    const dirty=Object.keys(drafts).filter(r=>r.startsWith(`${scope}/`));
    if(dirty.length){report(`未保存の下書きが${dirty.length}件あります。先に保存するか元に戻してください`);setAssignFolder(null);return;}
    setBusy(true);setError('');
    try{
      const r=await apiPost<{changed:number;unchanged:number;total:number}>(`/dataset-files/${project.id}/assign-instance`,{folder:scope,token});
      setAssignFolder(null);await refresh();onDatasetChanged?.();
      notice(`フォルダ「${scope}」の${r.total}枚にインスタンス「${token}」を設定しました（変更${r.changed}枚・既に設定済み${r.unchanged}枚）。学習対象は再確定してください`);
    }catch(e){report(e);}finally{setBusy(false);}
  }
  async function excludeFiles() {
    setBusy(true);
    try { const r=await apiPost<{token:string;archive:string}>(`/dataset-files/${project.id}/exclude`,{relatives:selected});setLastExcluded(r.token);localStorage.setItem(`${key}:excluded`,JSON.stringify(r.token));setExcludeOpen(false);setSelected([]);await refresh();notice(`削除せず退避しました：${r.archive}`); }catch(e){report(e);}finally{setBusy(false);}
  }
  async function trashFiles() {
    setBusy(true);
    try { const r=await apiPost<{count:number}>(`/dataset-files/${project.id}/trash`,{relatives:selected});setExcludeOpen(false);setSelected([]);await refresh();onDatasetChanged?.();notice(`${r.count}枚をごみ箱へ移動しました。Windowsのごみ箱から戻せます`); }catch(e){report(e);}finally{setBusy(false);}
  }
  async function createFolder() {
    if(newFolder===null||!newFolder.trim())return;
    setBusy(true);
    try { const r=await apiPost<{relative:string}>(`/dataset-files/${project.id}/create-folder`,{parent:folder,name:newFolder.trim()});setNewFolder(null);await refresh();notice(`フォルダ「${r.relative.split('/').pop()}」を作りました`); }catch(e){report(e);}finally{setBusy(false);}
  }
  async function trashFolder() {
    if(!folderToDelete)return;
    setBusy(true);
    try { const r=await apiPost<{count:number}>(`/dataset-files/${project.id}/trash-folder`,{folder:folderToDelete});const gone=folderToDelete;setFolderToDelete(null);if(folder===gone||folder.startsWith(`${gone}/`)){setFolder(gone.split('/').slice(0,-1).join('/'));setSelected([]);}else await refresh();onDatasetChanged?.();notice(`フォルダと画像${r.count}枚をごみ箱へ移動しました。Windowsのごみ箱から戻せます`); }catch(e){report(e);}finally{setBusy(false);}
  }
  async function restoreFiles() {
    setBusy(true);
    try{await apiPost(`/dataset-files/${project.id}/restore`,{token:lastExcluded});setLastExcluded('');localStorage.removeItem(`${key}:excluded`);await refresh();notice('元の場所に戻しました。レビューの採用状態は再確認してください');}catch(e){report(e);}finally{setBusy(false);}
  }
  async function reorderFiles(moving:string[],before:string|null) {
    if(!data)return;
    const oldOrder=sortedItems.map(i=>i.relative);const next=insertBlock(oldOrder,moving,before);
    if(JSON.stringify(next)===JSON.stringify(oldOrder))return;
    setBusy(true);setError('');
    try{const result=await apiPost<Listing>(`/dataset-files/${project.id}/order`,{folder,order:next,revision:data.order_revision});setData(result);setSort('manual');setUndoOrder({folder,order:oldOrder,revision:result.order_revision});onDatasetChanged?.();notice(`${moving.length}枚を並べ替えました`);}catch(e){report(e);}finally{setBusy(false);}
  }
  async function undoReorder() {
    if(!undoOrder)return;setBusy(true);
    try{const result=await apiPost<Listing>(`/dataset-files/${project.id}/order`,undoOrder);setData(result);setSort('manual');setUndoOrder(null);onDatasetChanged?.();notice('並び順を戻しました');}catch(e){report(e);}finally{setBusy(false);}
  }
  async function moveFiles(ids:string[],target:string){
    setBusy(true);setError('');
    try{await apiPost(`/dataset-files/${project.id}/move`,{relatives:ids,target_folder:target});setSelected(s=>s.filter(id=>!ids.includes(id)));await refresh();onDatasetChanged?.();notice(`${ids.length}枚とTXTを移動しました`);}catch(e){report(e);}finally{setBusy(false);}
  }
  function keyboardReorder(id:string,direction:number){
    const order=sortedItems.map(i=>i.relative),moving=selected.includes(id)?order.filter(p=>selected.includes(p)):[id];
    const indices=moving.map(p=>order.indexOf(p));const edge=direction<0?Math.min(...indices)-1:Math.max(...indices)+1;
    if(edge<0||edge>=order.length)return;
    const before=direction<0?order[edge]:order[edge+1]??null;void reorderFiles(moving,before);
  }
  const dnd=useDatasetDrag({active,folder,items:sortedItems,visible:items.map(i=>i.relative),selected,dirty:Object.keys(drafts),view,busy:busy||!data?.order_revision,folders:(data?.folders||[]).map(f=>f.relative),onSelect:setSelected,onReorder:(ids,before)=>void reorderFiles(ids,before),onMove:(ids,target)=>void moveFiles(ids,target),onImport:(files,target)=>void upload(files,target),onImportUrl:desktop()?(url,target)=>void importUrl(url,target):undefined,onReadStart:()=>setBusy(true),onReadError:message=>{setBusy(false);report(message);},onError:report});
  const marker=dnd.state?.kind==='reorder'?dnd.state.hit:undefined;
  const dropClass=(path:string)=>dnd.state?.target===path?(dnd.state.denied?'df-folder-denied':'df-folder-drop'):'';
  useEffect(()=>{setUndoOrder(null);},[folder]);
  const photo = (i:Item) => `${API_BASE}/collector/thumbnail?path=${encodeURIComponent(i.path)}&size=512&v=${i.modified}`;

  const selectedDrafts = selected.filter(p=>drafts[p]);
  const needsCopy = (data?.items||[]).some(i=>selected.includes(i.relative)&&(i.requires_training_copy??i.protected_source));
  const hasProtected = (data?.items||[]).some(i=>selected.includes(i.relative)&&i.protected_source);
  const bulkChanges=(data?.items||[]).filter(i=>selected.includes(i.relative)).map(i=>{
    const text=drafts[i.relative]?.text??i.caption;
    const tags=splitTags(text),incoming=splitTags(bulk||'');
    const matching=new Set(incoming.map(t=>t.toLowerCase()));
    const existing=new Set(tags.map(t=>t.toLowerCase()));
    if(bulkMode==='instance'){
      const chosen=(bulk||'').trim(),all=new Set(instances.map(x=>x.token.toLowerCase()));
      if(!chosen)return {item:i,before:text,after:text};
      const rest=tags.filter(t=>!all.has(t.toLowerCase()));
      const shared=rest.filter(t=>sharedTags.some(s=>s.toLowerCase()===t.toLowerCase())),others=rest.filter(t=>!shared.includes(t));
      return {item:i,before:text,after:[...shared,chosen,...others].join(', ')};
    }
    const applies=bulkMode==='add'?incoming.some(t=>!existing.has(t.toLowerCase())):bulkMode==='remove'?tags.some(t=>matching.has(t.toLowerCase())):!!bulkFind.trim()&&tags.some(t=>t.toLowerCase()===bulkFind.trim().toLowerCase());
    const next=bulkMode==='add'?uniqueTags([...tags,...incoming]):bulkMode==='remove'?tags.filter(t=>!matching.has(t.toLowerCase())):uniqueTags(tags.flatMap(t=>t.toLowerCase()===bulkFind.trim().toLowerCase()?incoming:[t]));
    return {item:i,before:text,after:applies?next.join(', '):text};
  }).filter(c=>c.before!==c.after && (bulkMode!=='replace'||bulkFind.trim()));
  const canUndoBulk=!!bulkUndo&&Object.entries(bulkUndo).every(([rel,v])=>drafts[rel]?.text===v.after.text&&drafts[rel]?.revision===v.after.revision);
  function applyBulk(){
    if(!bulkChanges.length)return;
    const undo:Record<string,{before:Draft|undefined;after:Draft}>={};
    const next={...drafts};
    bulkChanges.forEach(c=>{const rel=c.item.relative;const after={text:c.after,revision:drafts[rel]?.revision||c.item.revision};undo[rel]={before:drafts[rel],after};next[rel]=after;});
    setBulkUndo(undo);setDrafts(next);setBulk(null);onDatasetChanged?.();notice(`${bulkChanges.length}枚の下書きに反映しました。選択分を保存できます。`);
  }
  function undoBulk(){if(!bulkUndo||!canUndoBulk)return;setDrafts(prev=>{const next={...prev};Object.entries(bulkUndo).forEach(([rel,v])=>{if(v.before)next[rel]=v.before;else delete next[rel];});return next;});setBulkUndo(null);notice('一括タグ編集を戻しました');}
  const selectedTagSets=(data?.items||[]).filter(i=>selected.includes(i.relative)).map(i=>splitTags(drafts[i.relative]?.text??i.caption));
  const commonTags=(selectedTagSets[0]||[]).filter(t=>selectedTagSets.every(tags=>tags.some(v=>v.toLowerCase()===t.toLowerCase())));
  const saveSelected = () => save(selectedDrafts.map(relative=>({relative,caption:drafts[relative].text,revision:drafts[relative].revision})));
  return <section ref={dnd.rootRef} className="df-workspace" aria-busy={busy} aria-label="データセット編集">
    <div className="df-location"><FolderOpen size={16}/><code title={data?.directory}>{data?.directory||project.dataset_dir}</code><Button variant="ghost" color="gray" onClick={()=>void reveal(folder)}><FolderOpen size={15}/>エクスプローラーで開く</Button></div>
    <div className="df-toolbar">
      <Button disabled={busy} title={`「${folder||'データセット'}」フォルダに画像とTXTを追加します`} onClick={()=>input.current?.click()}><Upload size={16}/>画像・TXTを追加</Button>
      <Button variant="soft" color="gray" disabled={busy||!navigator.clipboard?.read} title="現在のフォルダに画像を保存（Ctrl+V）" onClick={()=>void readClipboardImages()}><ClipboardPaste size={16}/>画像を貼り付け</Button>
      <Button variant="soft" color="gray" disabled={busy} title="動画をカットごとに区切り、1カット1枚だけ候補にします" onClick={()=>setVideoOpen(true)}><Film size={16}/>動画から追加</Button>
      {desktop()&&<Button variant="soft" color="gray" disabled={busy} onClick={()=>{setError('');setWebOpen(true);}}>Web画像</Button>}
      <DropdownMenu.Root><DropdownMenu.Trigger><Button variant="soft" color="gray" disabled={busy||!items.length}>整理・前処理<DropdownMenu.TriggerIcon/></Button></DropdownMenu.Trigger><DropdownMenu.Content>{desktop()?.previewNaming&&<DropdownMenu.Item disabled={!!(selected.length?selectedDrafts.length:items.filter(i=>drafts[i.relative]).length)} onSelect={()=>{if(!selected.length)setSelected(items.map(i=>i.relative));setNamingOpen(true);}}>名前をまとめて整理</DropdownMenu.Item>}{desktop()&&<DropdownMenu.Item disabled={!!(selected.length?selectedDrafts.length:items.filter(i=>drafts[i.relative]).length)} onSelect={()=>{if(!selected.length)setSelected(items.map(i=>i.relative));setNormalizeOpen(true);}}>画像サイズを揃える</DropdownMenu.Item>}{namingUndo&&desktop()?.undoNaming&&<DropdownMenu.Item onSelect={()=>void undoNaming()}>直前の命名を戻す</DropdownMenu.Item>}</DropdownMenu.Content></DropdownMenu.Root>
      <input ref={input} type="file" multiple accept="image/png,image/jpeg,image/webp,image/bmp,.txt" hidden onChange={e=>e.target.files&&void upload(e.target.files)}/>
      <Tooltip content="外部で変更した画像・TXTを再読込"><Button variant="soft" color="gray" disabled={busy} onClick={()=>void refresh()}>{busy?<Spinner/>:<RefreshCw size={15}/>}再読込</Button></Tooltip>
      <TextField.Root className="df-search" aria-label="ファイル名・キャプションを検索" placeholder="名前・キャプションで検索" value={query} onChange={e=>setQuery(e.target.value)}><TextField.Slot><Search size={15}/></TextField.Slot></TextField.Root>
      <Select.Root value={sort} onValueChange={setSort}><Select.Trigger variant="soft" color="gray" aria-label="並び順"/><Select.Content><Select.Item value="manual">手動順</Select.Item><Select.Item value="name">名前順</Select.Item><Select.Item value="modified">更新が新しい順</Select.Item><Select.Item value="size">容量が大きい順</Select.Item></Select.Content></Select.Root>
      <Popover.Root open={displayOpen} onOpenChange={setDisplayOpen}><Popover.Trigger><IconButton variant="soft" color="gray" aria-label="表示設定"><SlidersHorizontal size={16}/></IconButton></Popover.Trigger><Popover.Content width="240"><Flex direction="column" gap="3"><Text size="2" weight="medium">画像サイズ</Text><SegmentedControl.Root value={String(size)} onValueChange={v=>setSize(Number(v))}><SegmentedControl.Item value="120">小</SegmentedControl.Item><SegmentedControl.Item value="180">中</SegmentedControl.Item><SegmentedControl.Item value="250">大</SegmentedControl.Item></SegmentedControl.Root></Flex></Popover.Content></Popover.Root>
      <SegmentedControl.Root value={view} onValueChange={setView} aria-label="表示形式"><SegmentedControl.Item value="grid" aria-label="サムネイル表示"><Grid2X2 size={16}/></SegmentedControl.Item><SegmentedControl.Item value="list" aria-label="詳細一覧表示"><List size={16}/></SegmentedControl.Item></SegmentedControl.Root>
    </div>
    {error&&<div className="df-error" role="alert">{error}</div>}
    <div className="df-body" style={{'--editor-width':`${editorWidth}px`} as React.CSSProperties}>
      <aside className="df-folders">{summaryText&&<Text size="1" weight="medium" color={tone(totals)} className="df-summary" role="status" aria-label={summaryText} title={summaryText}>{totals?<>学習対象 {totals.training_targets}/{totals.images}枚<br/>フォルダ {totals.folders_with_targets}/{totals.folders}個</>:summaryText}</Text>}<Button disabled={busy} data-dnd-zone="folder" data-drop-folder="" className={`${dropClass('')} df-folder-btn`} variant={!folder?'soft':'ghost'} color="gray" onClick={()=>{setFolder('');setSelected([]);}}><Folder size={16}/><span className="df-folder-label"><span>データセット</span>{summary&&<Text size="1" color={tone(summary.root)} data-folder-stat="">{statText(summary.root)}</Text>}</span></Button>{folder&&<Button disabled={busy} data-dnd-zone="folder" data-drop-folder={folder.split('/').slice(0,-1).join('/')} className={dropClass(folder.split('/').slice(0,-1).join('/'))} variant="ghost" color="gray" onClick={()=>{setFolder(folder.split('/').slice(0,-1).join('/'));setSelected([]);}}><ArrowUp size={16}/>ひとつ上へ</Button>}{data?.folders.map(f=><ContextMenu.Root key={f.relative}><ContextMenu.Trigger><Button disabled={busy} data-dnd-zone="folder" data-drop-folder={f.relative} className={`${dropClass(f.relative)} df-folder-btn`} variant="ghost" color="gray" onClick={()=>{setFolder(f.relative);setSelected([]);}} title={`${f.name}（右クリックで操作）`}><Folder size={16}/><span className="df-folder-label"><span>{f.name}</span>{statByFolder.has(f.relative)&&<Text size="1" color={tone(statByFolder.get(f.relative))} data-folder-stat="">{statText(statByFolder.get(f.relative))}</Text>}</span></Button></ContextMenu.Trigger><ContextMenu.Content size="2" aria-label="フォルダの操作"><ContextMenu.Item disabled={busy||!onPrepared||!statByFolder.get(f.relative)?.images} onSelect={()=>void prepareScope(f.relative)}>このフォルダの全画像を学習対象にする</ContextMenu.Item><ContextMenu.Item disabled={busy||!instances.length||!statByFolder.get(f.relative)?.images} onSelect={()=>{setError('');setAssignFolder({folder:f.relative,token:''});}}>このフォルダの全画像にインスタンスを割り当てる…{instances.length?'':'（インスタンス未作成）'}</ContextMenu.Item><ContextMenu.Separator/><ContextMenu.Item color="red" onSelect={()=>setFolderToDelete(f.relative)}>フォルダを削除…</ContextMenu.Item></ContextMenu.Content></ContextMenu.Root>)}<Button disabled={busy} variant="ghost" color="gray" onClick={()=>setNewFolder('')}><FolderPlus size={16}/><span>新しいフォルダ</span></Button><hr/>{triggerTags.length>0&&<><Text size="1" color={triggerCount===data?.items.length?'green':'amber'}>トリガー保存済み {triggerCount}/{data?.items.length||0}枚（このフォルダー）</Text><Text as="label" size="1"><Flex gap="2" align="center"><Checkbox aria-label="トリガーが足りない画像だけ表示" checked={onlyMissingTrigger} onCheckedChange={v=>setOnlyMissingTrigger(v===true)}/>トリガー不足</Flex></Text></>}<Text as="label" size="1"><Flex gap="2" align="center"><Checkbox aria-label="キャプションが空の画像だけ表示" checked={onlyEmpty} onCheckedChange={v=>setOnlyEmpty(v===true)}/>キャプションなし</Flex></Text>{dirtyCount>0&&<Badge color="amber" mt="4">下書き {dirtyCount}件</Badge>}</aside>
      <main className="df-browser">
        <div className={`df-selection ${selected.length?'has-selection':''}`}><Text size="1" color="gray">{folder||'データセット'} · {items.length}枚</Text>{selected.length>0&&<Badge>{selected.length}枚 選択</Badge>}<div className="df-selection-actions"><Button size="1" variant="ghost" color="gray" disabled={!items.length} onClick={()=>setSelected(items.map(i=>i.relative))}>すべて選択</Button>{selected.length>0&&<><Button size="1" variant="ghost" color="gray" onClick={()=>setSelected([])}>解除</Button>{selectedDrafts.length>0&&<Button size="1" disabled={busy} onClick={()=>void saveSelected()}><Save size={13}/>選択分を保存</Button>}<DropdownMenu.Root><DropdownMenu.Trigger><Button size="1" variant="soft" color="gray" disabled={busy} ref={bulkTrigger}>タグ編集<DropdownMenu.TriggerIcon/></Button></DropdownMenu.Trigger><DropdownMenu.Content><DropdownMenu.Item onSelect={()=>{setBulkMode('add');setBulkFind('');setBulk('');}}>選択した画像のタグを編集…{instances.length>0?'（インスタンス割り当て含む）':''}</DropdownMenu.Item>{desktop()&&<DropdownMenu.Item disabled={!!selectedDrafts.length} onSelect={()=>setAutoTagOpen(true)}>画像からタグを自動生成…</DropdownMenu.Item>}</DropdownMenu.Content></DropdownMenu.Root>{selectedDrafts.length===0&&<><Button size="1" variant="soft" disabled={busy} title={needsCopy?"他の工程が参照している原本なので、コピーを作って学習に使います":"選択した画像だけを学習対象にします（フォルダ全体なら右の「全画像を学習対象にして学習設定へ」）"} onClick={()=>void (needsCopy?copyForTraining():register())}>{needsCopy?"学習用に複製":onPrepared?"学習対象にする":"学習素材に登録"}</Button><Button size="1" variant="ghost" color="gray" disabled={busy||hasProtected} title={hasProtected?"他の工程で参照中の原本です":undefined} ref={excludeTrigger} onClick={()=>setExcludeOpen(true)}>削除…</Button></>}</>}{canUndoBulk&&<Button size="1" variant="ghost" disabled={busy} onClick={undoBulk}>一括編集を戻す</Button>}{lastExcluded&&<Button size="1" variant="ghost" color="gray" disabled={busy} onClick={()=>void restoreFiles()}>退避を戻す</Button>}</div></div>
        <ContextMenu.Root><ContextMenu.Trigger><div ref={dnd.gridRef} onContextMenu={e=>{const card=(e.target as HTMLElement).closest<HTMLElement>("[data-dnd-id]");if(card){const id=card.dataset.dndId!;setSelected(values=>values.includes(id)?[...values.filter(v=>v!==id),id]:[id]);setAnchor(id);}else setSelected([]);}} data-dnd-zone="grid" className={`df-files ${view==='list'?'df-list':''} ${dnd.state?.kind==='external'&&dnd.state.target===folder?'df-drop':''}`} style={{'--thumb':`${size}px`} as React.CSSProperties} role="listbox" aria-label="画像ファイル" aria-multiselectable tabIndex={0} onKeyDown={e=>{if((e.ctrlKey||e.metaKey)&&e.key==='a'){e.preventDefault();setSelected(items.map(i=>i.relative));}}}>
          {items.map(i=><Card asChild key={i.relative} className={`df-file ${selected.includes(i.relative)?'selected':''} ${current?.relative===i.relative?'editing':''} ${dnd.dragIds.includes(i.relative)?'df-dragging':''}`}><button data-dnd-id={i.relative} role="option" aria-selected={selected.includes(i.relative)} onPointerDown={()=>{dnd.ignoreClick.current=false;}} onKeyDown={e=>{if(e.key==='Enter'&&!e.altKey&&!e.ctrlKey&&!e.metaKey){e.preventDefault();openImage(i,e.currentTarget);return;}if(e.altKey&&['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(e.key)){e.preventDefault();keyboardReorder(i.relative,['ArrowLeft','ArrowUp'].includes(e.key)?-1:1);}}} onClick={e=>select(i,e)} onDoubleClick={e=>{if(!dnd.ignoreClick.current)openImage(i,e.currentTarget);}} title={`${i.path} · ダブルクリックで拡大`}><img draggable={false} src={photo(i)} alt="" loading="lazy"/><div><Text as="div" size="1" weight="medium" truncate>{i.name}</Text><Text as="div" size="1" color="gray">{i.width} × {i.height}</Text><Text as="div" size="1" color={drafts[i.relative]?'amber':'gray'} truncate>{drafts[i.relative]?'● 未保存':i.caption||'キャプションなし'}</Text></div>{trainingPaths?.some(p=>p.toLowerCase()===i.path.toLowerCase())?<Badge data-training="true" className="df-registered" size="1">学習対象</Badge>:!trainingPaths&&i.asset?<Badge className="df-registered" color="gray" size="1">学習素材</Badge>:null}{selected.includes(i.relative)&&<span className="df-selected-mark"><Check size={12}/></span>}</button></Card>)}
          {marker&&<div aria-hidden="true" data-drop-before={marker.before??'END'} className={`df-insertion df-insertion-${marker.axis}`} style={{left:marker.x,top:marker.y,...(marker.axis==='vertical'?{height:marker.length}:{width:marker.length})}}/>}
          {!items.length&&<div className="df-empty"><ImagePlus size={32}/><Text as="p" size="2">{query||onlyEmpty?'条件に合う画像がありません':'画像をドロップして追加'}</Text>{!query&&!onlyEmpty&&<Button variant="soft" onClick={()=>input.current?.click()}>ファイルを選択</Button>}</div>}
        </div>
        </ContextMenu.Trigger><ContextMenu.Content size="2" aria-label="画像の操作">
          <ContextMenu.Label>{selected.length?`${selected.length}枚の画像`:`${items.length}枚の一覧`}</ContextMenu.Label>{(selected.length?selectedDrafts.length:items.filter(i=>drafts[i.relative]).length)>0&&<ContextMenu.Label>未保存のタグがあります。先にTXTへ保存してください。</ContextMenu.Label>}
          {desktop()&&<ContextMenu.Item disabled={busy||!!(selected.length?selectedDrafts.length:items.filter(i=>drafts[i.relative]).length)||(!selected.length&&!items.length)} onSelect={()=>{if(!selected.length)setSelected(items.map(i=>i.relative));setAutoTagOpen(true);}}>一括自動タグ生成・TXT出力</ContextMenu.Item>}
          <ContextMenu.Item disabled={busy||!selected.length} onSelect={()=>{setBulkMode("add");setBulkFind("");setBulk("");}}>タグを一括編集</ContextMenu.Item>
          <ContextMenu.Item disabled={busy||!selectedDrafts.length} onSelect={()=>void saveSelected()}>下書きをTXTへ一括保存</ContextMenu.Item>
          <ContextMenu.Separator/><ContextMenu.Item disabled={!current} onSelect={()=>{if(current)openImage(current,document.activeElement as HTMLElement);}}>画像を拡大表示</ContextMenu.Item>
          <ContextMenu.Item disabled={busy||selected.length!==1} onSelect={()=>void reveal(current!.relative)}>エクスプローラーで表示</ContextMenu.Item>
          <ContextMenu.Item disabled={busy||!navigator.clipboard?.read} onSelect={()=>void readClipboardImages()}>画像を貼り付け（Ctrl+V）</ContextMenu.Item>
          <ContextMenu.Item disabled={!selected.length} onSelect={()=>{void navigator.clipboard.writeText((data?.items||[]).filter(i=>selected.includes(i.relative)).map(i=>i.path).join("\n")).then(()=>notice("パスをコピーしました")).catch(report);}}>ファイルのパスをコピー</ContextMenu.Item>
          {desktop()?.previewNaming&&<ContextMenu.Item disabled={busy||!selected.length||!!selectedDrafts.length} onSelect={()=>setNamingOpen(true)}>名前をまとめて整理</ContextMenu.Item>}
          <ContextMenu.Item disabled={busy||selected.length!==1||!!draft||hasProtected} onSelect={()=>setRename(current!.name)}>名前を変更</ContextMenu.Item>
          {desktop()&&<ContextMenu.Item disabled={busy||!selected.length||!!selectedDrafts.length} onSelect={()=>setNormalizeOpen(true)}>画像サイズを揃える</ContextMenu.Item>}
          <ContextMenu.Item disabled={busy||!selected.length||!!selectedDrafts.length} onSelect={()=>void copyForTraining()}>学習用コピーを作成</ContextMenu.Item>
          <ContextMenu.Item color="red" disabled={busy||!selected.length||hasProtected||!!selectedDrafts.length} onSelect={()=>setExcludeOpen(true)}>退避して外す…</ContextMenu.Item>
          <ContextMenu.Separator/><ContextMenu.Item disabled={!items.length} onSelect={()=>setSelected(items.map(i=>i.relative))}>すべて選択</ContextMenu.Item><ContextMenu.Item disabled={busy} onSelect={()=>void refresh()}>再読込</ContextMenu.Item>
        </ContextMenu.Content></ContextMenu.Root>
        <footer>{namingUndo&&desktop()?.undoNaming&&<Button size="1" variant="ghost" disabled={busy} onClick={()=>void undoNaming()}>命名を戻す</Button>}<span>Ctrl：複数選択 · Shift：範囲選択 · ドラッグ：移動 · Alt＋矢印：並べ替え</span>{undoOrder&&<Button size="1" variant="ghost" disabled={busy} onClick={()=>void undoReorder()}>並べ替えを戻す</Button>}<span>{items.length} images</span>{typeof extraTools==='function'?extraTools({busy,prepareAll:()=>prepareScope(''),summaryText:totals?`学習対象 ${totals.training_targets}/${totals.images}枚・フォルダ ${totals.folders_with_targets}/${totals.folders}`:'',summaryColor:tone(totals)}):extraTools}</footer>
      </main>
      <div className="df-editor-resize" role="separator" aria-label="画像詳細パネルの幅" aria-orientation="vertical" aria-valuemin={240} aria-valuemax={680} aria-valuenow={editorWidth} tabIndex={0} onPointerDown={e=>{resizeStart.current={x:e.clientX,width:editorWidth};e.currentTarget.setPointerCapture(e.pointerId);e.preventDefault();}} onPointerMove={e=>{if(resizeStart.current)resizeEditor(resizeStart.current.width+resizeStart.current.x-e.clientX);}} onPointerUp={e=>{resizeStart.current=null;if(e.currentTarget.hasPointerCapture(e.pointerId))e.currentTarget.releasePointerCapture(e.pointerId);}} onPointerCancel={()=>{resizeStart.current=null;}} onLostPointerCapture={()=>{resizeStart.current=null;}} onKeyDown={e=>{if(e.key==='ArrowLeft'||e.key==='ArrowRight'){e.preventDefault();resizeEditor(editorWidth+(e.key==='ArrowLeft'?20:-20));}if(e.key==='Home'){e.preventDefault();resizeEditor(330);}}}/>
      <div className="df-side"><aside className={`df-editor ${current?'has-file':''}`}>
        {current?<>{selected.length>1&&<Badge color="gray" mb="2">個別編集 · この1枚</Badge>}<Text as="div" size="2" weight="medium" mb="3" truncate>{current.name}</Text><button type="button" className="df-preview df-preview-open" aria-label={`${current.name}を拡大表示`} onClick={e=>openImage(current,e.currentTarget)} title="画像を拡大表示"><img src={photo(current)} alt={current.name}/><span className="df-preview-open-label">拡大表示</span></button><Flex gap="3" my="3" wrap="wrap"><Button variant="ghost" color="gray" size="1" onClick={()=>void reveal(current.relative)}><FolderOpen size={14}/>実ファイルを表示</Button><Button variant="ghost" color="gray" size="1" disabled={!!draft||busy||current.protected_source} title={current.protected_source?"他の工程で参照中の原本です":undefined} onClick={()=>setRename(current.name)}>名前を変更</Button></Flex>
        {rename!==null&&<form className="df-rename" onSubmit={e=>{e.preventDefault();void applyRename();}}><TextField.Root aria-label="新しいファイル名" value={rename} onChange={e=>setRename(e.target.value)}/><Button size="1" disabled={busy} type="submit">変更</Button><Button size="1" variant="soft" color="gray" type="button" onClick={()=>setRename(null)}>取消</Button></form>}
        <Flex justify="between" align="center" mt="5" mb="3"><Text as="div" size="2">キャプション</Text><Badge color={draft?'amber':current.revision.endsWith(':missing')?'gray':'green'} variant="soft">{draft?'未保存':current.revision.endsWith(':missing')?'TXTなし':<><Check size={12}/>保存済み</>}</Badge></Flex>
        {triggerTags.length>0&&<Flex direction="column" gap="1" mb="2" aria-label="この画像のトリガーワード">{instances.length>0&&(()=>{const text=draft?.text??current.caption;const own=instances.filter(i=>hasTag(text,i.token));return <Text size="1" color={own.length===1?'green':'amber'}>インスタンス：{own.length===0?'未割当（タグ編集 → インスタンスで割り当て）':own.length>1?`複数に割当（${own.map(i=>i.name).join('・')}）`:own[0].name}</Text>;})()}{sharedTags.map(t=>{const saved=splitTags(current.caption).some(tag=>tag.toLowerCase()===t.toLowerCase());const edited=splitTags(draft?.text??current.caption).some(tag=>tag.toLowerCase()===t.toLowerCase());return <Text key={t} size="1" color={saved?'green':'amber'}>{t}：{saved?'TXTに含まれています':edited?'下書きのみ・未保存':'TXTにありません'}</Text>;})}</Flex>}
        <Button size="1" variant="soft" disabled={busy} mb="2" onClick={async()=>{setBusy(true);try{await apiPost(`/evaluation/${project.id}`,{relative:current.relative},'POST',undefined,60000);window.dispatchEvent(new CustomEvent('evaluation-reference-changed',{detail:{projectId:project.id}}));notice('評価用に固定し、学習対象から除外しました');}catch(e){report(e);}finally{setBusy(false);}}}>評価用画像にする</Button>
        <CaptionEditor key={current.relative} storageKey={`${key}:caption-history:${current.relative}`} value={draft?.text??current.caption} disabled={busy} triggerTags={triggerTags} onChange={update} onSave={()=>{if(draft&&!busy)void save([{relative:current.relative,caption:draft.text,revision:draft.revision}]);}}/>
        {draft&&draft.revision!==current.revision&&<Text as="p" size="1" color="amber">外部の変更あり。下書きは保持しています。</Text>}
        <Flex gap="3" my="3"><Button disabled={!draft||busy} onClick={()=>draft&&void save([{relative:current.relative,caption:draft.text,revision:draft.revision}])}><Save size={15}/>TXTに保存</Button><Button variant="ghost" color="gray" disabled={!draft||busy} onClick={()=>setDrafts(prev=>{const n={...prev};delete n[current.relative];return n;})}>保存済みの内容に戻す</Button></Flex><Tooltip content={current.caption_path}><Button className="df-caption-file" size="1" variant="ghost" color="gray" disabled={current.revision.endsWith(':missing')} onClick={()=>void reveal(current.relative.replace(/\.[^.]+$/,'.txt'))}><FileText size={13}/>{current.name.replace(/\.[^.]+$/,'.txt')}</Button></Tooltip>{current.protected_source&&<Text as="p" size="1" color="gray" mt="3">確定済みの学習・生成で参照中の原本</Text>}{current.asset&&<Button className="df-review-link" size="1" variant="soft" onClick={onReview}>採用を確認 →</Button>}</>:<div className="df-empty"><FileText size={30}/><Text as="p" size="2" color="gray">画像を選択</Text></div>}
      </aside></div>
    </div>
    <div className="df-status" role="status">{dnd.state?.denied|| (dnd.dragIds.length?`${dnd.dragIds.length}枚を移動中`:busy?'処理中…':message||`${data?.items.length||0}枚`)}{data?.errors.length?` · 読込エラー ${data.errors.length}件`:''}</div>
    {data?.errors.length?<details className="df-errors"><summary>読込エラー</summary>{data.errors.map(e=><p key={e}>{e}</p>)}</details>:null}
    {viewing&&<DatasetImageViewer projectId={project.id} item={viewing} items={items} onChange={item=>setViewing(items.find(i=>i.relative===item.relative)||null)} onClose={()=>setViewing(null)} returnFocus={viewerReturn.current}/>}
    {autoTagOpen&&<AutoTagDialog projectId={project.id} relatives={selected} folderItems={data?.items||[]} dirtyRelatives={Object.keys(drafts)} onClose={()=>setAutoTagOpen(false)} onSave={async edits=>{if(edits.some(e=>drafts[e.relative])){report("未保存の下書きがあります。先に保存してください");return false;}return save(edits);}}/>}
    {namingOpen&&<BatchNamingDialog projectId={project.id} title={project.name} relatives={sortedItems.filter(i=>selected.includes(i.relative)).map(i=>i.relative)} protectedSource={hasProtected} onClose={()=>setNamingOpen(false)} onApplied={r=>{setQuery("");setSelected(r.relatives);if(r.folder!==folder)setFolder(r.folder);else void refresh();if(r.undoable){setNamingUndo(r.token);localStorage.setItem(`${key}:naming-undo`,JSON.stringify(r.token));}onDatasetChanged?.();notice("命名整理を反映しました");}}/>}
    {normalizeOpen&&<PreprocessDialog projectId={project.id} relatives={selected} onClose={()=>setNormalizeOpen(false)} onCreated={(path,relatives)=>{setQuery('');setOnlyEmpty(false);setFolder(path);setSelected(relatives);onDatasetChanged?.();}}/>}
    <Dialog.Root open={webOpen} onOpenChange={value=>{if(!busy)setWebOpen(value);}}><Dialog.Content maxWidth="520px"><Dialog.Title>Web画像を追加</Dialog.Title><Dialog.Description size="2" mb="4">Webページの画像を一覧へドラッグするか、画像のURLを貼り付けてください。現在のフォルダへPNGで保存します。</Dialog.Description><form onSubmit={e=>{e.preventDefault();void importUrl(webUrl.trim(),folder);}}><TextField.Root autoFocus aria-label="画像のURL" placeholder="https://…/image.png" value={webUrl} disabled={busy} onChange={e=>setWebUrl(e.target.value)}/><Text as="p" size="1" color="gray" mt="3">公開画像に対応。ページ自体のURL、ログインが必要な画像、保存を制限しているサイトは取り込めません。</Text>{error&&<Text as="p" size="2" color="red" role="alert">{error}</Text>}<Flex justify="end" gap="3" mt="4"><Button type="button" variant="soft" color="gray" disabled={busy} onClick={()=>setWebOpen(false)}>キャンセル</Button><Button type="submit" disabled={busy||!/^https?:\/\//i.test(webUrl.trim())}>{busy?<Spinner/>:null}保存して追加</Button></Flex></form></Dialog.Content></Dialog.Root>
    <Dialog.Root open={excludeOpen} onOpenChange={setExcludeOpen}><Dialog.Content maxWidth="440px" onCloseAutoFocus={e=>{e.preventDefault();excludeTrigger.current?.focus();}}><Dialog.Title>選択した {selected.length}枚を削除</Dialog.Title><Dialog.Description size="2" mb="5">画像と同名のTXTをまとめて処理します。どちらも元に戻せます（ごみ箱はWindowsのごみ箱、退避はこの画面の「退避を戻す」）。</Dialog.Description><Flex gap="3" justify="end" wrap="wrap"><Dialog.Close><Button variant="soft" color="gray">キャンセル</Button></Dialog.Close><Button variant="soft" color="gray" disabled={busy} onClick={()=>void excludeFiles()}>退避だけする</Button><Button color="red" disabled={busy} onClick={()=>void trashFiles()}>ごみ箱へ移動</Button></Flex></Dialog.Content></Dialog.Root>
    <VideoImportDialog open={videoOpen} onOpenChange={setVideoOpen} projectId={project.id} folder={folder} onAdded={(count,skipped)=>{void refresh();onDatasetChanged?.();notice(`動画から${count}枚を追加しました${skipped?`（同名の${skipped}枚は追加済みのためそのまま）`:''}。キャラ判定は目安なので、画像を見て不要なものは削除してください`);}}/>
    <Dialog.Root open={newFolder!==null}onOpenChange={open=>{if(!open)setNewFolder(null);}}><Dialog.Content maxWidth="400px"><Dialog.Title>新しいフォルダ</Dialog.Title><Dialog.Description size="2" mb="3">「{folder||'データセット'}」の中に作ります。衣装・インスタンスごとに分けると整理しやすくなります。</Dialog.Description><form onSubmit={e=>{e.preventDefault();void createFolder();}}><TextField.Root aria-label="フォルダ名" autoFocus placeholder="例：outfit_uniform" value={newFolder||''} disabled={busy} onChange={e=>setNewFolder(e.target.value)}/><Flex gap="3" justify="end" mt="4"><Button type="button" variant="soft" color="gray" onClick={()=>setNewFolder(null)}>キャンセル</Button><Button type="submit" disabled={busy||!(newFolder||'').trim()}>作成</Button></Flex></form></Dialog.Content></Dialog.Root>
    <Dialog.Root open={assignFolder!==null} onOpenChange={open=>{if(!open&&!busy)setAssignFolder(null);}}><Dialog.Content maxWidth="460px"><Dialog.Title>フォルダ「{(assignFolder?.folder||'').split('/').pop()}」にインスタンスを割り当て</Dialog.Title><Dialog.Description size="2" mb="3">フォルダ内（下の階層も含む）{statByFolder.get(assignFolder?.folder||'')?.images??0}枚のTXTに、選んだインスタンスのトリガーを入れます。他のインスタンスのトリガーは外し、共通トリガーとその他のタグはそのまま残します。</Dialog.Description><form onSubmit={e=>{e.preventDefault();void assignInstanceToFolder();}}><Select.Root value={assignFolder?.token||''} onValueChange={v=>setAssignFolder(a=>a?{...a,token:v}:a)}><Select.Trigger style={{width:'100%'}} aria-label="割り当てるインスタンス" placeholder="インスタンスを選ぶ"/><Select.Content>{instances.map(i=><Select.Item key={i.token} value={i.token}>{i.name===i.token?i.token:`${i.name}（${i.token}）`}</Select.Item>)}</Select.Content></Select.Root><Text as="p" size="1" color="gray" mt="3">TXTを直接書き換えます。確定済みの学習対象には影響しませんが、再度「学習対象にする」で確定し直してください。</Text>{error&&<Text as="p" size="2" color="red" role="alert">{error}</Text>}<Flex gap="3" justify="end" mt="4"><Button type="button" variant="soft" color="gray" disabled={busy} onClick={()=>setAssignFolder(null)}>キャンセル</Button><Button type="submit" disabled={busy||!assignFolder?.token}>{busy?<Spinner/>:null}割り当てる</Button></Flex></form></Dialog.Content></Dialog.Root>
    <Dialog.Root open={folderToDelete!==null} onOpenChange={open=>{if(!open)setFolderToDelete(null);}}><Dialog.Content maxWidth="440px"><Dialog.Title>フォルダ「{(folderToDelete||'').split('/').pop()}」を削除</Dialog.Title><Dialog.Description size="2" mb="5">フォルダの中の画像とTXTをまとめてWindowsのごみ箱へ移動します。確定した学習対象などで使われている画像があるときは削除できません。</Dialog.Description><Flex gap="3" justify="end"><Button variant="soft" color="gray" onClick={()=>setFolderToDelete(null)}>キャンセル</Button><Button color="red" disabled={busy} onClick={()=>void trashFolder()}>ごみ箱へ移動</Button></Flex></Dialog.Content></Dialog.Root>
    <Dialog.Root open={bulk!==null} onOpenChange={open=>{if(!open)setBulk(null);}}><Dialog.Content maxWidth="480px" onCloseAutoFocus={e=>{e.preventDefault();bulkTrigger.current?.focus();}}><Dialog.Title>選択した {selected.length}枚のタグ編集</Dialog.Title><Dialog.Description size="2" mb="4">下書きに反映してから、選択分をまとめて保存できます。</Dialog.Description><Text as="div" size="1" color="gray" mb="2">選択画像に共通するタグ（{commonTags.length}）</Text><Flex gap="1" wrap="wrap" mb="3" style={{maxHeight:120,overflow:"auto"}}>{commonTags.length?commonTags.map(tag=><Badge key={tag} color="gray">{tag}</Badge>):<Text size="1" color="gray">共通タグはありません</Text>}</Flex><form onSubmit={e=>{e.preventDefault();applyBulk();}}><SegmentedControl.Root value={bulkMode} onValueChange={setBulkMode} aria-label="一括タグ操作"><SegmentedControl.Item value="add">追加</SegmentedControl.Item><SegmentedControl.Item value="replace">置換</SegmentedControl.Item><SegmentedControl.Item value="remove">削除</SegmentedControl.Item>{instances.length>0&&<SegmentedControl.Item value="instance">インスタンス</SegmentedControl.Item>}</SegmentedControl.Root>{bulkMode==='replace'&&<TextField.Root mt="3" aria-label="置換するタグ" placeholder="置換するタグ（完全一致）" value={bulkFind} onChange={e=>setBulkFind(e.target.value)}/>}{bulkMode==='instance'?<Select.Root value={bulk||''} onValueChange={setBulk}><Select.Trigger mt="3" style={{width:'100%'}} aria-label="割り当てるインスタンス" placeholder="インスタンスを選ぶ"/><Select.Content>{instances.map(i=><Select.Item key={i.token} value={i.token}>{i.name===i.token?i.token:`${i.name}（${i.token}）`}</Select.Item>)}</Select.Content></Select.Root>:<TextArea mt="3" disabled={busy} aria-label={bulkMode==='add'?'追加するタグ':bulkMode==='remove'?'削除するタグ':'置換後のタグ'} value={bulk||''} onChange={e=>setBulk(e.target.value)} placeholder={bulkMode==='replace'?'置換後のタグ（空欄なら削除）':'カンマ・改行で複数指定'}/>}<Text as="p" size="1" color="gray" mt="3">{bulkChanges.length} / {selected.length}枚を変更</Text><Flex gap="3" justify="end" mt="5"><Button type="button" variant="soft" color="gray" onClick={()=>setBulk(null)}>キャンセル</Button><Button type="submit" disabled={!bulkChanges.length||busy}>下書きに反映</Button></Flex></form></Dialog.Content></Dialog.Root>
  </section>;
}
