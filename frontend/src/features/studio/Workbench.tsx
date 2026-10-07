import {useEffect,useState,useRef} from 'react';
import {Button,Dialog,Flex,Text,TextField,Spinner,Callout,Tabs,Popover,IconButton} from '@radix-ui/themes';
import {Plus,ListChecks,Layers,X,FolderOpen,Check,ArrowLeft,SlidersHorizontal,RefreshCw} from 'lucide-react';
import {apiGet,apiPost} from '../../lib/api';
import type {Project} from '../../types';
import type {PreparedDataset} from './workbenchTypes';
type HistoryDetail = PreparedDataset & {entries:{file_path:string;caption_at_snapshot:string}[]};
import DatasetFiles from './DatasetFiles';
import TrainingDock from './TrainingDock';
import './workbench.css';
import {desktop} from './desktopBridge';

function ProjectCanvas({project,active,openRootRequest}:{project:Project;active:boolean;openRootRequest:number}){
  const [workspace,setWorkspace]=useState(()=>localStorage.getItem(`workbench:view:${project.id}`)==='training'?'training':'dataset');
  const backButton=useRef<HTMLButtonElement>(null),trainingButton=useRef<HTMLButtonElement>(null);
  const switchWorkspace=(value:string)=>{setWorkspace(value);requestAnimationFrame(()=>{(value==='training'?backButton:trainingButton).current?.focus();});};
  useEffect(()=>{localStorage.setItem(`workbench:view:${project.id}`,workspace);},[workspace,project.id]);
  useEffect(()=>{if(openRootRequest)setWorkspace('dataset');},[openRootRequest]);
  useEffect(()=>{const update=(e:Event)=>{if((e as CustomEvent).detail.projectId!==project.id)return;void apiGet<{config:{dataset_snapshot_id?:number}}>(`/training/config-draft/${project.id}`).then(async r=>{if(r.config.dataset_snapshot_id){await chooseHistory(r.config.dataset_snapshot_id);}else{setPrepared(null);setTrainingPaths([]);setChanged(true);}}).catch(e=>setError(String(e)));};window.addEventListener('evaluation-reference-changed',update);return()=>window.removeEventListener('evaluation-reference-changed',update);},[project.id]);
  const [trainingPaths,setTrainingPaths]=useState<string[]>([]);
  const [prepared,setPrepared]=useState<PreparedDataset|null>(null);
  const [changed,setChanged]=useState(()=>localStorage.getItem(`workbench:changed:${project.id}`)==='true');
  useEffect(()=>{localStorage.setItem(`workbench:changed:${project.id}`,String(changed));},[changed,project.id]);
  const [historyDetail,setHistoryDetail]=useState<HistoryDetail|null>(null);
  const [history,setHistory]=useState<PreparedDataset[]|null>(null);
  const [error,setError]=useState('');
  async function openHistory(){setHistoryDetail(null);try{const r=await apiGet<{snapshots:PreparedDataset[]}>(`/basepipe/projects/${project.id}/snapshots`);setHistory(r.snapshots);}catch(e){setError(String(e));}}
  async function chooseHistory(id:number){try{
    const r=await apiGet<PreparedDataset>(`/basepipe/snapshots/${id}`);
    const current=await apiGet<{config:{preview_profile_snapshot_id?:number;dataset_snapshot_id?:number}}>(`/training/config-draft/${project.id}`);
    // Historical snapshots without a preview binding remain visible, but never silently invent a binding.
    if(current.config.dataset_snapshot_id===id && current.config.preview_profile_snapshot_id){const p=await apiGet<{snapshots:{id:number;name:string;snapshot_hash:string}[]}>(`/basepipe/projects/${project.id}/preview-profile-snapshots`);r.preview_profile_snapshot=p.snapshots.find(x=>x.id===current.config.preview_profile_snapshot_id)||null;}
    setPrepared(r);setTrainingPaths(r.entries?.map(e=>e.file_path)||[]);setChanged(false);setHistory(null);
  }catch(e){setError(String(e));}}
  return <>
    {error&&<Callout.Root color="red"><Callout.Text>{error}</Callout.Text></Callout.Root>}
    <div className="wb-dataset-stage" hidden={workspace!=='dataset'}><DatasetFiles project={project} active={active&&workspace==='dataset'} openRootRequest={openRootRequest} showError={setError} showNotice={()=>{}} onReview={()=>void openHistory()}
      trainingPaths={trainingPaths} onPrepared={v=>{setPrepared(v);setTrainingPaths(v.entries?.map(e=>e.file_path)||[]);setChanged(false);}} onDatasetChanged={()=>setChanged(true)}
      extraTools={ctx=><>{ctx.summaryText&&<Text size="1" weight="medium" color={ctx.summaryColor} className="wb-target-summary" aria-label="学習対象の件数">{ctx.summaryText}</Text>}<Button size="2" variant="soft" disabled={ctx.busy} title="データセット内のすべての画像（フォルダの中も含む）を学習対象にして、学習設定を開きます。評価用画像は除外されます" onClick={()=>{void ctx.prepareAll().then(ok=>{if(ok)switchWorkspace('training');});}}><ListChecks size={15}/>全画像を学習対象にして学習設定へ</Button><Button ref={trainingButton} size="2" variant="solid" onClick={()=>switchWorkspace('training')}><SlidersHorizontal size={15}/>学習設定を開く</Button><Button size="1" variant="ghost" color="gray" title="以前確定した学習対象（画像とキャプションの組）を一覧から選び直します" onClick={()=>void openHistory()}>保存済みの学習対象</Button></>}/></div>
    <div className="wb-training-stage" hidden={workspace!=='training'}><div className="wb-stage-tools"><Button ref={backButton} variant="soft" color="gray" onClick={()=>switchWorkspace('dataset')}><ArrowLeft size={16}/>データセットに戻る</Button><Button size="1" variant="ghost" color="gray" title="以前確定した学習対象（画像とキャプションの組）を一覧から選び直します" onClick={()=>void openHistory()}>保存済みの学習対象</Button></div><TrainingDock project={project} prepared={prepared} changed={changed} onTargets={setTrainingPaths}/></div>
    <Dialog.Root open={history!==null} onOpenChange={open=>{if(!open)setHistory(null);}}><Dialog.Content maxWidth="560px"><Dialog.Title>保存済みの学習対象</Dialog.Title><Dialog.Description size="2" mb="4">以前確定した画像とキャプション</Dialog.Description><Flex direction="column" gap="2">{historyDetail?<><Text size="2" weight="medium">{historyDetail.name} · {historyDetail.item_count}枚</Text>{historyDetail.entries.map((entry,i)=><div key={i} style={{padding:"10px",background:"var(--gray-a3)",borderRadius:"var(--radius-2)"}}><Text as="div" size="1" weight="medium">{entry.file_path.split(/[\\/]/).pop()}</Text><Text as="p" size="1" color="gray">{entry.caption_at_snapshot||"キャプションなし"}</Text></div>)}<Flex gap="3" mt="3"><Button variant="soft" color="gray" onClick={()=>setHistoryDetail(null)}>一覧に戻る</Button><Button onClick={()=>void chooseHistory(historyDetail.id)}>このセットを学習対象にする</Button></Flex></>:history?.length?history.map(s=><Button key={s.id} variant="soft" color="gray" onClick={()=>{void apiGet<HistoryDetail>(`/basepipe/snapshots/${s.id}`).then(setHistoryDetail).catch(e=>setError(e.message));}}>{s.name} · {s.item_count}枚</Button>):<Text color="gray" size="2">まだありません</Text>}</Flex><Flex justify="end" mt="4"><Dialog.Close><Button variant="soft" color="gray">閉じる</Button></Dialog.Close></Flex></Dialog.Content></Dialog.Root>
  </>;
}

export default function Workbench(){
  const [reloading,setReloading]=useState(false);
  async function reloadScreen(){setReloading(true);try{const pending:Promise<unknown>[]=[];window.dispatchEvent(new CustomEvent('workbench-before-reload',{detail:{waitUntil:(p:Promise<unknown>)=>pending.push(p)}}));await Promise.race([Promise.all(pending),new Promise((_,reject)=>setTimeout(()=>reject(new Error('設定の保存を確認できませんでした。再読み込みは行っていません')),10000))]);window.location.reload();}catch(e){setError(String(e));setReloading(false);}}
  const [projects,setProjects]=useState<Project[]>([]),[selected,setSelected]=useState<number|null>(null);
  const [openTabs,setOpenTabs]=useState<number[]>([]);
  const [openingFolder,setOpeningFolder]=useState(false);
  const [rootRequests,setRootRequests]=useState<Record<number,number>>({});
  const [loading,setLoading]=useState(true),[error,setError]=useState('');
  const [openMenu,setOpenMenu]=useState(false),[projectQuery,setProjectQuery]=useState('');
  const [create,setCreate]=useState(false),[name,setName]=useState(''),[creating,setCreating]=useState(false);
  async function load(){setLoading(true);try{
    const rows=await apiGet<Project[]>('/projects');setProjects(rows);
    const fromPath=Number(window.location.pathname.match(/\/project\/(\d+)/)?.[1]);
    let saved:{ids?:number[];active?:number|null}={};try{saved=JSON.parse(localStorage.getItem('workbench:tabs:v1')||'{}');}catch{}
    const valid=(id:number|null|undefined)=>rows.some(p=>p.id===id);
    const ids=Array.isArray(saved.ids)?saved.ids.filter((id,index,all)=>valid(id)&&all.indexOf(id)===index):[];
    let stored=0;try{stored=Number(localStorage.getItem('workbench:project'));}catch{}
    const next=valid(fromPath)?fromPath:Array.isArray(saved.ids)?(valid(saved.active)?saved.active!:ids[0]??null):valid(stored)?stored:rows[0]?.id??null;
    if(next!==null&&!ids.includes(next))ids.push(next);
    setOpenTabs(ids);setSelected(next);setError('');
  }catch(e){setError(e instanceof Error?e.message:String(e));}finally{setLoading(false);}}
  useEffect(()=>{void load();},[]);
  useEffect(()=>{if(!loading){localStorage.setItem('workbench:tabs:v1',JSON.stringify({ids:openTabs,active:selected}));localStorage.setItem('workbench:project',selected?String(selected):'');}},[openTabs,selected,loading]);
  function choose(id:number){setOpenTabs(prev=>prev.includes(id)?prev:[...prev,id]);setSelected(id);setOpenMenu(false);}
  async function openFolder(){if(openingFolder)return;setOpeningFolder(true);setError('');try{const p=await desktop()!.openDatasetFolder();if(p){setProjects(rows=>rows.some(row=>row.id===p.id)?rows.map(row=>row.id===p.id?p:row):[...rows,p]);setRootRequests(values=>({...values,[p.id]:(values[p.id]||0)+1}));choose(p.id);}}catch(e){setError(e instanceof Error?e.message:String(e));}finally{setOpeningFolder(false);}}
  function closeTab(id:number){const next=openTabs.filter(v=>v!==id);setOpenTabs(next);if(selected===id){const at=openTabs.indexOf(id);setSelected(next[Math.min(at,next.length-1)]??null);if(!next.length)window.history.replaceState({},'','/');}}
  async function createProject(){setCreating(true);try{const p=await apiPost<Project>('/projects',{name:name.trim(),project_type:'character'});setProjects(v=>[p,...v]);choose(p.id);setCreate(false);setName('');}catch(e){setError(e instanceof Error?e.message:String(e));}finally{setCreating(false);}}
  const opened=openTabs.map(id=>projects.find(p=>p.id===id)).filter((p):p is Project=>!!p);
  return <Tabs.Root className="wb-app" value={selected?String(selected):''} onValueChange={v=>setSelected(Number(v))} onKeyDown={e=>{if((e.ctrlKey||e.metaKey)&&e.key==='Tab'&&(e.target as HTMLElement).closest('[role=menu][data-state=open]')){e.preventDefault();return;}if((e.ctrlKey||e.metaKey)&&e.key==='Tab'&&openTabs.length>1&&!((e.target as HTMLElement).closest('[role="dialog"]'))){e.preventDefault();const index=openTabs.indexOf(selected!);setSelected(openTabs[(index+(e.shiftKey?-1:1)+openTabs.length )%openTabs.length]);requestAnimationFrame(()=>document.querySelector<HTMLButtonElement>('.wb-project-trigger[data-state=active]')?.focus());}}}>
    <header className="wb-bar"><Flex align="center" gap="2"><Layers size={17}/><Text size="2" weight="medium">LoRA Studio</Text></Flex><span className="wb-bar-divider"/>
      <Tabs.List className="wb-project-tabs" aria-label="開いているプロジェクト">{opened.map(p=><div className={`wb-project-tab ${selected===p.id?'is-active':''}`} key={p.id}><Tabs.Trigger value={String(p.id)} title={p.dataset_dir} aria-label={p.name} className="wb-project-trigger"><FolderOpen size={13}/><span>{p.name}</span></Tabs.Trigger><button className="wb-close-tab" type="button" aria-label={`${p.name}のタブを閉じる`} title="タブを閉じる（データは削除されません）" onClick={()=>closeTab(p.id)}><X size={13}/></button></div>)}</Tabs.List>
      <Button size="1" variant="soft" color="gray" disabled={reloading} title="画面だけ更新します。学習とサーバーは停止しません" onClick={()=>void reloadScreen()}>{reloading?<Spinner/>:<RefreshCw size={14}/>}画面を再読み込み</Button>
      <Popover.Root open={openMenu} onOpenChange={v=>{setOpenMenu(v);if(v)setProjectQuery('');}}><Popover.Trigger><IconButton size="1" variant="ghost" color="gray" aria-label="プロジェクトをタブで開く" title="プロジェクトをタブで開く"><Plus size={17}/></IconButton></Popover.Trigger><Popover.Content align="end" className="wb-project-picker"><TextField.Root aria-label="プロジェクトを検索" placeholder="プロジェクトを検索" value={projectQuery} onChange={e=>setProjectQuery(e.target.value)}/><div className="wb-project-choices">{projects.filter(p=>p.name.toLowerCase().includes(projectQuery.toLowerCase())).map(p=><button key={p.id} className="wb-project-choice" type="button" onClick={()=>choose(p.id)}><FolderOpen size={15}/><span><strong>{p.name}</strong><small>{p.dataset_dir}</small></span>{openTabs.includes(p.id)&&<Check size={14}/>}</button>)}</div><Button size="2" variant="soft" onClick={()=>{setOpenMenu(false);setCreate(true);}}><Plus size={14}/>新しいプロジェクト</Button></Popover.Content></Popover.Root>
      {desktop()&&<Button size="1" variant="soft" color="gray" disabled={openingFolder} onClick={()=>void openFolder()} title="既存フォルダをその場所で開く（画像はコピーしません）">{openingFolder?<Spinner/>:<FolderOpen size={14}/>}フォルダを開く</Button>}
    </header>
    {error&&<Callout.Root color="red"><Callout.Text>{error}</Callout.Text><Button size="1" onClick={()=>void load()}>再接続</Button></Callout.Root>}
    <main className="wb-canvas">{loading?<Flex justify="center" align="center" height="300px"><Spinner size="3"/></Flex>:opened.length?opened.map(p=><Tabs.Content key={p.id} value={String(p.id)} forceMount className="wb-project-panel"><ProjectCanvas project={p} active={selected===p.id} openRootRequest={rootRequests[p.id]||0}/></Tabs.Content>):<Flex direction="column" align="center" justify="center" gap="4" height="400px"><Text color="gray">プロジェクトをタブで開いて作業を開始</Text><Button onClick={()=>setOpenMenu(true)}><FolderOpen size={16}/>プロジェクトを開く</Button><Button variant="soft" onClick={()=>setCreate(true)}><Plus size={16}/>新しいプロジェクト</Button></Flex>}</main>
    <Dialog.Root open={create} onOpenChange={setCreate}><Dialog.Content maxWidth="400px"><Dialog.Title>新しいプロジェクト</Dialog.Title><Dialog.Description size="2" mb="4">キャラクターや学習内容が分かる名前</Dialog.Description><form onSubmit={e=>{e.preventDefault();void createProject();}}><TextField.Root autoFocus aria-label="プロジェクト名" value={name} onChange={e=>setName(e.target.value)} placeholder="名前"/><Flex justify="end" gap="3" mt="5"><Dialog.Close><Button type="button" variant="soft" color="gray">キャンセル</Button></Dialog.Close><Button type="submit" disabled={!name.trim()||creating}>作成</Button></Flex></form></Dialog.Content></Dialog.Root>
  </Tabs.Root>;
}
