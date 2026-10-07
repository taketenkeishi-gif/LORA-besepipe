import {useEffect,useRef,useState} from 'react';
import {AlertDialog,Badge,Button,Callout,Dialog,Flex,Select,Text,TextField} from '@radix-ui/themes';
import {ApiError,apiGet,apiPost} from '../../lib/api';
import type {Config} from './TrainingDock';
import {desktop,desktopError} from './desktopBridge';
import PresetManager from './PresetManager';
import {categoryName,differences,downloadJson,pick,readPreset,type Entry,type Preset,type PresetCategory} from './presetModel';

type Prompt={kind:'new'|'import'|'rename'|'duplicate';entry?:Entry;imported?:Preset};
type Confirm={kind:'overwrite'|'delete';entry:Entry};
const BASE='/presets/training-files';
const storageKey=(projectId:number)=>`lora-studio.applied-preset.${projectId}`;
const loadApplied=(projectId:number)=>{try{return localStorage.getItem(storageKey(projectId))||'';}catch{return '';}};
const saveApplied=(projectId:number,id:string)=>{try{if(id)localStorage.setItem(storageKey(projectId),id);else localStorage.removeItem(storageKey(projectId));}catch{/* storage may be unavailable */}};
const message=(e:unknown)=>e instanceof Error?e.message:String(e);
const currentCategory=(config:Config):PresetCategory=>config.training_goal==='art_style'?'style':'character';

export default function TrainingPresets({projectId,config,disabled,onApply}:{projectId:number;config:Config;disabled:boolean;onApply:(patch:Partial<Config>)=>void}){
 const [items,setItems]=useState<Entry[]>([]),[folder,setFolder]=useState('');
 const [appliedId,setAppliedIdState]=useState(()=>loadApplied(projectId));
 const [busy,setBusy]=useState(false),[error,setError]=useState(''),[notice,setNotice]=useState('');
 const [managerOpen,setManagerOpen]=useState(false);
 const [prompt,setPrompt]=useState<Prompt|null>(null),[name,setName]=useState(''),[category,setCategory]=useState<PresetCategory>('character'),[conflict,setConflict]=useState(false),[promptError,setPromptError]=useState('');
 const [confirm,setConfirm]=useState<Confirm|null>(null);
 const fileInput=useRef<HTMLInputElement>(null);

 const setApplied=(id:string)=>{setAppliedIdState(id);saveApplied(projectId,id);};
 useEffect(()=>{setAppliedIdState(loadApplied(projectId));},[projectId]);
 async function refresh(){
  const rows=await apiGet<Entry[]>(BASE);setItems(rows);
  // A preset that was deleted or renamed outside the app can no longer be "applied".
  setAppliedIdState(id=>{if(id&&!rows.some(r=>r.id===id)){saveApplied(projectId,'');return '';}return id;});
 }
 useEffect(()=>{void refresh().catch(e=>setError(message(e)));void apiGet<{path:string}>(BASE+'/folder').then(r=>setFolder(r.path)).catch(()=>{});},[]);
 useEffect(()=>{if(managerOpen)void refresh().catch(e=>setError(message(e)));},[managerOpen]);

 const applied=items.find(i=>i.id===appliedId&&i.payload)||null;
 const diffs=applied?.payload?differences(config,applied.payload):[];

 async function validate(p:Preset){await apiPost('/training/advanced-validate',{...p.config,network_train_unet_only:true,max_data_loader_n_workers:0,persistent_data_loader_workers:false});}
 async function run(work:()=>Promise<void>){setBusy(true);setError('');try{await work();}catch(e){setError(message(e));}finally{setBusy(false);}}
 const apply=(entry:Entry)=>run(async()=>{
  if(!entry.payload)throw Error('このファイルは学習プリセットとして読み込めません');
  const p=readPreset(entry.payload);await validate(p);onApply(p.config);setApplied(entry.id);setNotice(`「${p.name}」を次回の学習設定に適用しました`);
 });

 function openPrompt(next:Prompt){
  setPrompt(next);setConflict(false);setPromptError('');
  setName(next.kind==='rename'?next.entry!.name:next.kind==='duplicate'?`${next.entry!.name} のコピー`:next.imported?.name||'');
  setCategory(next.imported?.category||currentCategory(config));
 }
 async function submitPrompt(){
  if(!prompt)return;setBusy(true);setPromptError('');
  try{
   const trimmed=name.trim();
   if(prompt.kind==='rename'){
    const r=await apiPost<Entry>(`${BASE}/${encodeURIComponent(prompt.entry!.id)}/rename`,{name:trimmed});
    if(appliedId===prompt.entry!.id)setApplied(r.id);
    setNotice(`「${r.name}」に名前を変更しました`);
   }else if(prompt.kind==='duplicate'){
    const r=await apiPost<Entry>(`${BASE}/${encodeURIComponent(prompt.entry!.id)}/duplicate`,{name:trimmed});
    setNotice(`「${r.name}」として複製しました`);
   }else{
    const p=readPreset({...(prompt.imported||{}),format:'lora-studio-training-preset',version:1,name:trimmed,category,config:prompt.imported?.config||pick(config)});
    await validate(p);
    const r=await apiPost<Entry>(BASE,{name:p.name,payload:p,overwrite:conflict});
    if(prompt.imported){onApply(p.config);}
    setApplied(r.id);setNotice(`「${p.name}」を保存しました（${r.file_path}）`);
   }
   await refresh();setPrompt(null);
  }catch(e){
   if(e instanceof ApiError&&e.status===409&&prompt.kind!=='rename'&&prompt.kind!=='duplicate'&&!conflict){setConflict(true);setPromptError('同名のプリセットがあります。上書きする場合は「上書きして保存」を押してください。');}
   else setPromptError(message(e));
  }finally{setBusy(false);}
 }
 const doConfirm=()=>{
  const c=confirm;if(!c)return;setConfirm(null);
  void run(async()=>{
   if(c.kind==='delete'){
    await apiPost(`${BASE}/${encodeURIComponent(c.entry.id)}`,{},'DELETE');
    if(appliedId===c.entry.id)setApplied('');
    setNotice(`「${c.entry.name}」をごみ箱に移動しました（Windowsのごみ箱から元に戻せます）`);
   }else{
    const p=readPreset({format:'lora-studio-training-preset',version:1,name:c.entry.name,category:currentCategory(config),config:pick(config)});
    await validate(p);
    const r=await apiPost<Entry>(BASE,{name:p.name,payload:p,overwrite:true});
    setApplied(r.id);setNotice(`「${p.name}」を現在の設定で上書きしました`);
   }
   await refresh();
  });
 };
 async function reveal(entry?:Entry){
  try{
   const bridge=desktop();
   if(bridge?.revealPreset)await bridge.revealPreset(entry?.id??null);
   else await apiPost(BASE+'/reveal',{id:entry?.id??null});
  }catch(e){setError(desktopError(e));}
 }
 async function importFile(file:File){
  setError('');
  try{if(file.size>1024*1024)throw Error('JSONは1MB以内で指定してください');const p=readPreset(JSON.parse(await file.text()));await validate(p);openPrompt({kind:'import',imported:p});}
  catch(e){setError(message(e));}
 }
 const exportCurrent=()=>downloadJson({format:'lora-studio-training-preset',version:1,name:applied?.name||`${config.model_family}・${categoryName(currentCategory(config))}`,category:currentCategory(config),config:pick(config)});

 const promptTitle={new:'現在の設定を新規保存',import:'JSONをプリセットに登録',rename:'名前を変更',duplicate:'プリセットを複製'};
 const choosable=items.filter(i=>i.payload);
 return <div className="wb-training-presets" style={{marginBottom:12}}>
  <Flex gap="2" wrap="wrap" align="center">
   <Text size="1" color="gray">学習目的</Text>
   <Select.Root value={config.training_goal} onValueChange={v=>onApply({training_goal:v})} disabled={disabled||busy}><Select.Trigger aria-label="学習目的"/><Select.Content><Select.Item value="character_identity">キャラクター</Select.Item><Select.Item value="art_style">スタイル</Select.Item></Select.Content></Select.Root>
   <Text size="1" color="gray" style={{marginLeft:8}}>プリセット</Text>
   <Select.Root value={applied?.id} onOpenChange={o=>{if(o)void refresh().catch(e=>setError(message(e)));}} onValueChange={id=>{const e=items.find(i=>i.id===id);if(e)void apply(e);}} disabled={disabled||busy||!choosable.length}>
    <Select.Trigger aria-label="学習プリセット" placeholder={choosable.length?'選んで適用':'プリセットなし'} style={{flex:1,minWidth:180}}/>
    <Select.Content>{choosable.map(p=><Select.Item key={p.id} value={p.id}>{p.name} · {categoryName(p.payload!.category)} · {p.payload!.config.model_family}</Select.Item>)}</Select.Content>
   </Select.Root>
  </Flex>
  <Flex gap="2" wrap="wrap" mt="2" align="center">
   <Badge color={!applied?'gray':diffs.length?'amber':'green'} title={diffs.map(d=>`${d.label}：${d.current} → ${d.preset}`).join('\n')}>{!applied?'プリセット未適用（手動設定）':diffs.length?`「${applied.name}」から${diffs.length}項目変更`:`「${applied.name}」を適用中`}</Badge>
   {applied&&diffs.length>0&&<Button size="1" variant="ghost" disabled={disabled||busy} onClick={()=>void apply(applied)}>元に戻す</Button>}
   <Flex gap="2" ml="auto"><Button size="1" variant="soft" disabled={disabled||busy} onClick={()=>openPrompt({kind:'new'})}>現在の設定を保存</Button><Button size="1" variant="soft" color="gray" disabled={disabled} onClick={()=>setManagerOpen(true)}>プリセットを管理…</Button></Flex>
  </Flex>
  <input ref={fileInput} type="file" accept=".json,application/json" aria-label="学習設定JSONファイル" hidden onChange={e=>{const f=e.target.files?.[0];e.target.value='';if(f)void importFile(f);}}/>
  {notice&&!managerOpen&&<Text as="p" size="1" color="gray" role="status">{notice}</Text>}
  {error&&!managerOpen&&!prompt&&<Callout.Root color="red" size="1" mt="1"><Callout.Text style={{whiteSpace:'pre-line'}}>{error}</Callout.Text></Callout.Root>}

  <PresetManager open={managerOpen} onOpenChange={o=>{setManagerOpen(o);if(!o)setError('');}} items={items} folder={folder} appliedId={appliedId} differences={diffs} busy={busy} notice={notice} error={error}
   actions={{apply:e=>void apply(e),saveNew:()=>openPrompt({kind:'new'}),importFile:()=>fileInput.current?.click(),exportCurrent,overwrite:e=>setConfirm({kind:'overwrite',entry:e}),rename:e=>openPrompt({kind:'rename',entry:e}),duplicate:e=>openPrompt({kind:'duplicate',entry:e}),remove:e=>setConfirm({kind:'delete',entry:e}),exportEntry:e=>{if(e.payload)downloadJson(e.payload);},reveal:e=>void reveal(e)}}/>

  <Dialog.Root open={!!prompt} onOpenChange={v=>{if(!busy&&!v)setPrompt(null);}}><Dialog.Content maxWidth="520px">
   <Dialog.Title>{prompt?promptTitle[prompt.kind]:''}</Dialog.Title>
   <Dialog.Description size="2">{prompt?.kind==='rename'||prompt?.kind==='duplicate'?'ファイル名に使えない文字（\\ / : * ? " < > |）は使えません。':'モデル・学習パラメータ・詳細設定を保存します。画像選択、保存名、トリガーワードはプロジェクトごとに保持されます。'}</Dialog.Description>
   <TextField.Root mt="3" aria-label="プリセット名" placeholder="例：Anima・キャラクター768" value={name} onChange={e=>{setName(e.target.value);setConflict(false);setPromptError('');}} onKeyDown={e=>{if(e.key==='Enter'&&name.trim()&&!busy)void submitPrompt();}}/>
   {(prompt?.kind==='new'||prompt?.kind==='import')&&<Flex mt="3" align="center" gap="2"><Text size="1" color="gray">分類</Text><Select.Root value={category} onValueChange={v=>setCategory(v as PresetCategory)}><Select.Trigger aria-label="保存するプリセットの分類"/><Select.Content><Select.Item value="character">キャラクター用</Select.Item><Select.Item value="style">スタイル用</Select.Item></Select.Content></Select.Root></Flex>}
   {promptError&&<Callout.Root color={conflict?'amber':'red'} mt="3"><Callout.Text>{promptError}</Callout.Text></Callout.Root>}
   <Flex justify="end" gap="2" mt="4"><Button variant="soft" color="gray" disabled={busy} onClick={()=>setPrompt(null)}>キャンセル</Button><Button color={conflict?'amber':undefined} disabled={busy||!name.trim()} onClick={()=>void submitPrompt()}>{conflict?'上書きして保存':prompt?.kind==='import'?'登録して適用':prompt?.kind==='rename'?'名前を変更':prompt?.kind==='duplicate'?'複製する':'保存する'}</Button></Flex>
  </Dialog.Content></Dialog.Root>

  <AlertDialog.Root open={!!confirm} onOpenChange={v=>{if(!v)setConfirm(null);}}><AlertDialog.Content maxWidth="460px">
   <AlertDialog.Title>{confirm?.kind==='delete'?'プリセットを削除':'プリセットを上書き'}</AlertDialog.Title>
   <AlertDialog.Description size="2">{confirm?.kind==='delete'?`「${confirm.entry.name}」のファイルをWindowsのごみ箱へ移動します。ごみ箱から元に戻せます。`:`「${confirm?.entry.name}」を、いま画面に出ている学習設定で置き換えます。元の内容は戻せません。`}</AlertDialog.Description>
   <Flex justify="end" gap="2" mt="4"><AlertDialog.Cancel><Button variant="soft" color="gray">キャンセル</Button></AlertDialog.Cancel><Button color={confirm?.kind==='delete'?'red':'amber'} onClick={doConfirm}>{confirm?.kind==='delete'?'ごみ箱へ移動':'上書きする'}</Button></Flex>
  </AlertDialog.Content></AlertDialog.Root>
 </div>;
}
