import {useCallback,useEffect,useMemo,useRef,useState} from 'react';
import {AlertDialog,Badge,Button,Callout,Checkbox,Dialog,Flex,Progress,RadioGroup,Select,Slider,Spinner,Text,TextField} from '@radix-ui/themes';
import {API_BASE,apiGet,apiPost} from '../../lib/api';
import {desktop} from './desktopBridge';

// ---- 契約（バックエンド /dataset-files/{pid}/video-dataset）の型 ----
type Outfit={folder:string;trigger:string;images:number;dropped_duplicates:number;top_clothing:string[]};
type Char={folder:string;trigger:string;crops:number;images_total:number;views:Record<string,number>;sample_files:string[];all_files?:string[];outfits:Outfit[]};
type Timing={total_s:number;count:number;per_unit_s:number};
type Report={duration_s:number;scenes:number;frames_kept:number;person_crops:number;unassigned_images:number;images_written:number;wall_clock_s:number;per_output_image_s:number;per_output_image_excl_model_load_s:number;damaged_candidates_tried:number;crops_with_other_characters:number;view_distribution_all_crops:Record<string,number>;timing:Record<string,Timing>;characters:Char[];unassigned_sample_files:string[];unassigned_files?:string[];rejected_frame_examples:string[]};
type Job={status:'running'|'done'|'error'|'cancelled';stage:string;percent:number;elapsed_s:number;eta_s:number|null;error:string;log_tail:string[];params:Record<string,unknown>;output_dir:string;report:Report|null};
type JobSummary={job_id:string;status:string;created_at?:string|number;video?:string;images_written?:number};
type Imported={folder:string;project_id:number;images:number;concepts?:number|string[]};
type ImportResult={imported:Imported[];skipped:unknown[]};

type Settings={video_path:string;scene_sensitivity:'fine'|'normal'|'coarse';frame_interval:number;max_frames_per_scene:number;others:'keep'|'mask'|'exclude';overlap_threshold:number;margin:number;long_side:number;min_area:number;min_short:number;min_char_crops:number;merge_quantile:number;use_names:boolean;gpu:number|null;limit_seconds:number};
const DEFAULTS:Settings={video_path:'',scene_sensitivity:'normal',frame_interval:0.5,max_frames_per_scene:24,others:'keep',overlap_threshold:0.25,margin:0.10,long_side:1024,min_area:0.01,min_short:100,min_char_crops:3,merge_quantile:0.80,use_names:false,gpu:null,limit_seconds:0};
const STORE_KEY='lora-studio.videoDataset.settings.v3';
function loadSettings():Settings{try{const raw=window.localStorage.getItem(STORE_KEY);if(raw){const p=JSON.parse(raw) as Partial<Settings>;return {...DEFAULTS,...p};}}catch{/* 保存できない環境では既定値 */}return {...DEFAULTS};}
function saveSettings(s:Settings){try{window.localStorage.setItem(STORE_KEY,JSON.stringify(s));}catch{/* 無視 */}}

const STAGE_LABELS:Record<string,string>={queued:'順番待ち',starting:'準備中',load:'モデルを読み込み中',loading:'モデルを読み込み中',scenes:'カットを区切っています',scene:'カットを区切っています',extract:'フレームを取り出しています',frames:'フレームを取り出しています',quality:'傷んだフレームを除外中',detect:'人物を探しています',person:'人物を切り抜いています',crop:'人物を切り抜いています',tag:'タグを付けています',tagging:'タグを付けています',identify:'キャラを見分けています',cluster:'キャラごとにグループ分け中',group:'キャラごとにグループ分け中',outfit:'衣装ごとに仕分け中',upscale:'小さい画像を拡大中',write:'画像を書き出しています',export:'画像を書き出しています',report:'結果をまとめています',finalize:'仕上げ中',done:'完了'};
const stageLabel=(s:string)=>s?(STAGE_LABELS[s.toLowerCase()]??s):'準備中';
const TIMING_LABELS:Record<string,string>={model_load:'モデル読み込み',scenes:'カット検出',scene_detect:'カット検出',extract:'フレーム抽出',frames:'フレーム抽出',quality:'画質チェック',detect:'人物検出',person_detect:'人物検出',tag:'タグ付け',tagging:'タグ付け',clip:'見た目の特徴抽出',cluster:'グループ分け',outfit:'衣装仕分け',upscale:'拡大',write:'書き出し',total:'合計'};
const timingLabel=(k:string)=>TIMING_LABELS[k]??k;

const fmtDur=(s:number|null|undefined)=>{if(s==null||!isFinite(s))return '—';const t=Math.round(s);const h=Math.floor(t/3600),m=Math.floor(t%3600/60),sec=t%60;return h?`${h}時間${m}分${sec}秒`:m?`${m}分${sec}秒`:`${sec}秒`;};
const fmtSec=(s:number|null|undefined)=>s==null||!isFinite(s)?'—':`${s.toFixed(2)}秒`;
const errText=(e:unknown)=>e instanceof Error?e.message:String(e);

const base=(pid:number)=>`/dataset-files/${pid}/video-dataset`;
const fileUrl=(pid:number,job:string,rel:string)=>`${API_BASE}${base(pid)}/${job}/file?rel=${encodeURIComponent(rel)}`;

// 向き・構図の内訳（views のキー例: front-upper / back-other / side-full / front-lower-cut）
function viewBadges(views:Record<string,number>){
 let front=0,back=0,side=0,full=0,cut=0,close=0;
 for(const [k,n] of Object.entries(views)){
  if(k.startsWith('back'))back+=n;else if(k.startsWith('side'))side+=n;else if(k.startsWith('front'))front+=n;
  if(k.endsWith('-full'))full+=n;
  if(k.includes('-cut'))cut+=n;
  if(k.endsWith('-close'))close+=n;
 }
 return [['正面',front],['後ろ姿',back],['横向き',side],['全身',full],['見切れ',cut],['アップ',close]] as [string,number][];
}

// ---- 小さな部品 ----
function Field({label,help,children}:{label:string;help?:string;children:React.ReactNode}){
 return <div style={{minWidth:0}}><Text as="div" size="2" weight="medium" mb="1">{label}</Text>{children}{help&&<Text as="p" size="1" color="gray" mt="1" style={{overflowWrap:'anywhere'}}>{help}</Text>}</div>;
}
function NumField({label,help,value,onCommit,min,step,suffix}:{label:string;help?:string;value:number;onCommit:(v:number)=>void;min?:number;step?:number;suffix?:string}){
 const [text,setText]=useState(String(value));
 useEffect(()=>{setText(String(value));},[value]);
 return <Field label={label} help={help}><Flex align="center" gap="2"><TextField.Root style={{width:120}} type="number" inputMode="decimal" min={min} step={step} aria-label={label} value={text}
  onChange={e=>{setText(e.target.value);const v=Number(e.target.value);if(e.target.value.trim()!==''&&isFinite(v))onCommit(v);}} onBlur={()=>setText(String(value))}/>{suffix&&<Text size="1" color="gray">{suffix}</Text>}</Flex></Field>;
}
function SliderField({label,help,value,onChange,min=0,max=1,step=0.01}:{label:string;help?:string;value:number;onChange:(v:number)=>void;min?:number;max?:number;step?:number}){
 return <Field label={`${label}：${value.toFixed(2)}`} help={help}><Slider aria-label={label} min={min} max={max} step={step} value={[value]} onValueChange={v=>onChange(v[0])}/></Field>;
}

// ---- 設定フォーム ----
function SetupForm({s,setS,running,onStart,busy,recent,onOpenJob}:{s:Settings;setS:(f:(p:Settings)=>Settings)=>void;running:boolean;onStart:()=>void;busy:boolean;recent:JobSummary[];onOpenJob:(id:string)=>void}){
 const set=<K extends keyof Settings>(k:K,v:Settings[K])=>setS(p=>({...p,[k]:v}));
 async function choose(){try{const p=await desktop()?.chooseVideo?.();if(p)set('video_path',p);}catch{/* ダイアログのキャンセル等 */}}
 return <div>
  <Flex gap="2" align="end" wrap="wrap">
   <label style={{flex:1,minWidth:240}}><Text size="2" weight="medium">動画のファイル</Text>
    <TextField.Root aria-label="動画のパス" placeholder={String.raw`C:\Videos\example.mp4`} value={s.video_path} onChange={e=>set('video_path',e.target.value)}/></label>
   {desktop()?.chooseVideo&&<Button variant="soft" color="gray" onClick={()=>void choose()}>ファイルを選ぶ</Button>}
  </Flex>
  <Text as="p" size="1" color="gray" mt="1">このPC上の動画ファイルのパスを入力します。動画は変更されません。</Text>

  <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fit,minmax(260px,1fr))',gap:16,marginTop:16}}>
   <Field label="カットの区切り方" help="場面が切り替わる所を探す細かさです。細かくすると短いカットも別の場面として扱います。">
    <RadioGroup.Root value={s.scene_sensitivity} onValueChange={v=>set('scene_sensitivity',v as Settings['scene_sensitivity'])} aria-label="カットの区切り方">
     <RadioGroup.Item value="fine">細かく</RadioGroup.Item><RadioGroup.Item value="normal">標準</RadioGroup.Item><RadioGroup.Item value="coarse">大きなカットだけ</RadioGroup.Item></RadioGroup.Root></Field>
   <NumField label="シーンあたりの抽出間隔（秒）" help="同じカットの中で何秒おきにフレームを取り出すか。短いほど枚数が増え、処理も長くなります。" value={s.frame_interval} min={0.1} step={0.1} suffix="秒" onCommit={v=>set('frame_interval',v)}/>
  </div>

  <div style={{marginTop:16}}><Field label="他のキャラが写っているときの切り抜き" help="複数のキャラが同じ画面にいるとき、注目しているキャラの切り抜きをどう扱うかを選びます。">
   <RadioGroup.Root value={s.others} onValueChange={v=>set('others',v as Settings['others'])} aria-label="他のキャラが写っているときの切り抜き">
    <RadioGroup.Item value="keep">そのまま残す（注目キャラ全体が入ればよい・推奨）</RadioGroup.Item>
    <RadioGroup.Item value="mask">他キャラを灰色で塗る</RadioGroup.Item>
    <RadioGroup.Item value="exclude">他キャラが入る画像は使わない</RadioGroup.Item></RadioGroup.Root>
   <Text as="p" size="1" color="gray" mt="1">「そのまま残す」は枚数が多く集まります。「灰色で塗る」は他キャラが学習に混ざるのを防げますが、塗り跡が残ることがあります。「使わない」は最も安全ですが枚数が減ります。</Text></Field></div>

  <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fit,minmax(260px,1fr))',gap:16,marginTop:16}}>
   <SliderField label="他キャラとみなす重なりの閾値" help="切り抜きの中に、別のキャラの体がこの割合以上入っていたら「他キャラが写っている」とみなします。小さいほど厳しく判定します。" value={s.overlap_threshold} onChange={v=>set('overlap_threshold',v)}/>
   <SliderField label="切り抜きの余白" help="人物の枠の外側にどれだけ余白を付けて切り抜くか（人物の大きさに対する割合）。" value={s.margin} max={0.5} onChange={v=>set('margin',v)}/>
   <NumField label="長辺をそろえる（px）" help="切り抜きの長い辺をこのサイズにそろえます。0 ならそのままのサイズで保存します。" value={s.long_side} min={0} step={64} suffix="px" onCommit={v=>set('long_side',Math.max(0,Math.round(v)))}/>
  </div>

  <details style={{marginTop:16}}><summary style={{cursor:'pointer'}}><Text size="2" weight="medium">詳細設定</Text></summary>
   <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fit,minmax(260px,1fr))',gap:16,marginTop:12}}>
    <SliderField label="人物の最小サイズ（画面に対する面積）" help="画面に占める割合がこれより小さい人物は切り抜きません。遠景の小さな人物を除外します。" value={s.min_area} max={0.3} step={0.005} onChange={v=>set('min_area',v)}/>
    <NumField label="切り抜きの最小の短辺（px）" help="これより小さい切り抜きは使いません。" value={s.min_short} min={0} step={10} suffix="px" onCommit={v=>set('min_short',Math.max(0,Math.round(v)))}/>
    <NumField label="キャラとして認める最小の切り抜き数" help="この枚数に届かない人物は「未仕分け」に回します。モブを除くのに使います。" value={s.min_char_crops} min={1} step={1} suffix="枚" onCommit={v=>set('min_char_crops',Math.max(1,Math.round(v)))}/>
    <NumField label="1シーンあたりの最大枚数" help="1つのカットから取り出すフレームの上限です。" value={s.max_frames_per_scene} min={1} step={1} suffix="枚" onCommit={v=>set('max_frames_per_scene',Math.max(1,Math.round(v)))}/>
    <SliderField label="グループ分けの厳しさ（merge_quantile）" help="ここを上げるとキャラが複数フォルダに割れにくくなる反面、別人が混ざりやすくなります。割れたときは結果画面で「統合」できます。" value={s.merge_quantile} min={0.5} max={0.99} onChange={v=>set('merge_quantile',v)}/>
    <Field label="使うGPU" help="自動なら空いているGPUを選びます。学習中は別のGPUを選ぶか、学習が終わってから実行してください。">
     <Select.Root value={s.gpu==null?'auto':String(s.gpu)} onValueChange={v=>set('gpu',v==='auto'?null:Number(v))}><Select.Trigger aria-label="使うGPU"/><Select.Content><Select.Item value="auto">自動</Select.Item><Select.Item value="0">GPU 0</Select.Item><Select.Item value="1">GPU 1</Select.Item></Select.Content></Select.Root></Field>
    <NumField label="試し実行（先頭N秒だけ）" help="0 なら動画の最後まで処理します。まず 60 秒などで試すと設定の見当が付けられます。" value={s.limit_seconds} min={0} step={30} suffix="秒" onCommit={v=>set('limit_seconds',Math.max(0,v))}/>
   </div>
   <Text as="label" size="2" mt="3" style={{display:'block'}}><Flex gap="2" align="center"><Checkbox checked={s.use_names} onCheckedChange={v=>set('use_names',v===true)}/>WD14が知っているキャラ名があればグループ分けに使う</Flex></Text>
   <Text as="p" size="1" color="gray" ml="5">未知のキャラ（オリジナルなど）には影響しません。名前を知っているキャラだけ、別のカットでもまとまりやすくなります。</Text>
  </details>

  <Flex gap="3" align="center" wrap="wrap" mt="4">
   <Button disabled={!s.video_path.trim()||running||busy} onClick={onStart}>{busy?<Spinner/>:null}{s.limit_seconds>0?`先頭${s.limit_seconds}秒で試し実行`:'キャラ別データセットを作る'}</Button>
   <Button variant="ghost" color="gray" size="1" onClick={()=>setS(p=>({...DEFAULTS,video_path:p.video_path}))}>設定を初期値に戻す</Button>
  </Flex>
  <Text as="p" size="1" color="gray" mt="2">処理中はGPUを使います。完了までこの画面を閉じても処理は続き、開き直すと続きが見られます。</Text>

  {recent.length>0&&<div style={{marginTop:20}}><Text as="div" size="2" weight="medium" mb="1">前回までの結果</Text>
   <Flex direction="column" gap="1">{recent.slice(0,5).map(j=><Flex key={j.job_id} gap="2" align="center" wrap="wrap">
    <Badge color={j.status==='done'?'green':j.status==='running'?'blue':'gray'}>{j.status==='done'?'完了':j.status==='running'?'実行中':j.status==='error'?'失敗':'中止'}</Badge>
    <Text size="1" style={{overflowWrap:'anywhere',minWidth:0,flex:1}}>{(j.video??'').split(/[\\/]/).pop()||j.job_id}{j.images_written!=null?`（${j.images_written}枚）`:''}</Text>
    <Button size="1" variant="soft" color="gray" onClick={()=>onOpenJob(j.job_id)}>開く</Button></Flex>)}</Flex></div>}
 </div>;
}

// ---- 進捗 ----
function ProgressView({job,onCancel,cancelling}:{job:Job;onCancel:()=>void;cancelling:boolean}){
 return <div>
  <Flex justify="between" align="center" gap="3" wrap="wrap"><Text size="2" weight="medium">{stageLabel(job.stage)}</Text><Text size="2">{Math.max(0,Math.min(100,job.percent))}%</Text></Flex>
  <Progress value={job.percent} mt="2" aria-label="進捗"/>
  <Flex gap="4" wrap="wrap" mt="2"><Text size="1" color="gray">経過 {fmtDur(job.elapsed_s)}</Text><Text size="1" color="gray">残り {job.eta_s==null?'計算中':`約${fmtDur(job.eta_s)}`}</Text></Flex>
  <Flex gap="3" mt="3"><Button color="gray" variant="soft" disabled={cancelling} onClick={onCancel}>{cancelling?<Spinner/>:null}中止</Button></Flex>
  <details style={{marginTop:12}}><summary style={{cursor:'pointer'}}><Text size="1" color="gray">詳しい経過（ログ）</Text></summary>
   <pre style={{margin:'8px 0 0',padding:8,background:'var(--gray-3)',borderRadius:6,maxHeight:180,overflow:'auto',whiteSpace:'pre-wrap',overflowWrap:'anywhere',fontSize:11}}>{job.log_tail.length?job.log_tail.join('\n'):'（まだログはありません）'}</pre></details>
 </div>;
}

// ---- 結果 ----
function Thumb({src,alt,onClick,size=88}:{src:string;alt:string;onClick:()=>void;size?:number}){
 return <button type="button" onClick={onClick} title={alt} aria-label={`拡大：${alt}`} style={{padding:0,border:'1px solid var(--gray-6)',borderRadius:4,background:'var(--gray-3)',cursor:'zoom-in',flex:'0 0 auto',width:size,height:size}}>
  <img src={src} alt={alt} loading="lazy" style={{width:'100%',height:'100%',objectFit:'cover',borderRadius:3,display:'block'}}/></button>;
}
function CharCard({c,pid,jobId,checked,onCheck,mergeMode,mergeChecked,onMergeCheck,onRename,onPreview,busy}:{c:Char;pid:number;jobId:string;checked:boolean;onCheck:(v:boolean)=>void;mergeMode:boolean;mergeChecked:boolean;onMergeCheck:(v:boolean)=>void;onRename:(newName:string)=>void;onPreview:(list:string[],idx:number)=>void;busy:boolean}){
 const [name,setName]=useState(c.folder);const [showAll,setShowAll]=useState(false);
 const dirty=name.trim()!==''&&name!==c.folder;
 return <div style={{border:'1px solid var(--gray-6)',borderRadius:8,padding:12,minWidth:0}}>
  <Flex gap="3" align="center" wrap="wrap">
   <Text as="label" size="2"><Flex gap="2" align="center"><Checkbox checked={checked} onCheckedChange={v=>onCheck(v===true)} aria-label={`${c.folder}をインポートに含める`}/>取り込み対象</Flex></Text>
   {mergeMode&&<Text as="label" size="2" color="blue"><Flex gap="2" align="center"><Checkbox checked={mergeChecked} onCheckedChange={v=>onMergeCheck(v===true)} aria-label={`${c.folder}を統合の対象にする`}/>統合する</Flex></Text>}
   <Flex gap="1" align="center" style={{flex:1,minWidth:200}}>
    <TextField.Root style={{flex:1,minWidth:0}} aria-label={`${c.folder}の名前`} value={name} disabled={busy} onChange={e=>setName(e.target.value)} onKeyDown={e=>{if(e.key==='Enter'&&dirty)onRename(name.trim());}}/>
    <Button size="1" variant="soft" disabled={!dirty||busy} onClick={()=>onRename(name.trim())}>名前を変更</Button></Flex>
  </Flex>
  <Flex gap="2" wrap="wrap" mt="2" align="center">
   <Badge color="indigo">トリガー：{c.trigger}</Badge><Badge color="gray">切り抜き {c.crops}</Badge><Badge color="green">出力 {c.images_total}枚</Badge>
   {viewBadges(c.views).filter(([,n])=>n>0).map(([l,n])=><Badge key={l} variant="soft" color="gray">{l} {n}</Badge>)}
  </Flex>
  {c.outfits.length>0&&<div style={{marginTop:8}}><Text as="div" size="1" color="gray" mb="1">衣装ごとのフォルダ</Text>
   <Flex direction="column" gap="1">{c.outfits.map(o=><Flex key={o.folder} gap="2" wrap="wrap" align="center" style={{fontSize:12}}>
    <Text size="1" weight="medium" style={{overflowWrap:'anywhere'}}>{o.folder}</Text><Badge size="1" variant="outline">{o.trigger}</Badge><Text size="1">{o.images}枚</Text>
    {o.dropped_duplicates>0&&<Text size="1" color="gray">（重複 {o.dropped_duplicates}枚を除外）</Text>}
    {o.top_clothing.length>0&&<Text size="1" color="gray" style={{overflowWrap:'anywhere'}}>{o.top_clothing.slice(0,5).join('、')}</Text>}</Flex>)}</Flex></div>}
  {c.sample_files.length>0&&!showAll&&<Flex gap="2" mt="2" style={{overflowX:'auto',paddingBottom:4}}>{c.sample_files.map((f,i)=><Thumb key={f} src={fileUrl(pid,jobId,f)} alt={f.split('/').pop()||f} onClick={()=>onPreview(c.sample_files,i)}/>)}</Flex>}
  {showAll&&<Flex gap="2" wrap="wrap" mt="2">{(c.all_files??c.sample_files).map((f,i,all)=><Thumb key={f} src={fileUrl(pid,jobId,f)} alt={f.split('/').pop()||f} onClick={()=>onPreview(all,i)}/>)}</Flex>}
  {(c.all_files?.length??0)>c.sample_files.length&&<Button size="1" variant="soft" color="gray" mt="2" onClick={()=>setShowAll(v=>!v)}>{showAll?'代表の12枚だけ表示':`全${c.all_files!.length}枚を表示`}</Button>}
 </div>;
}
function Collapsed({title,files,pid,jobId,onPreview,help}:{title:string;files:string[];pid:number;jobId:string;onPreview:(l:string[],i:number)=>void;help:string}){
 if(!files.length)return null;
 return <details style={{marginTop:12}}><summary style={{cursor:'pointer'}}><Text size="2" weight="medium">{title}（{files.length}）</Text></summary>
  <Text as="p" size="1" color="gray" mt="1">{help}</Text>
  <Flex gap="2" wrap="wrap" mt="2">{files.map((f,i)=><Thumb key={f} src={fileUrl(pid,jobId,f)} alt={f.split('/').pop()||f} onClick={()=>onPreview(files,i)}/>)}</Flex></details>;
}

function UnassignedPanel({files,chars,pid,jobId,busy,onAssign,onPreview}:{files:string[];chars:Char[];pid:number;jobId:string;busy:boolean;onAssign:(files:string[],target:string|null,newName:string|null)=>Promise<boolean>;onPreview:(l:string[],i:number)=>void}){
 const [sel,setSel]=useState<Set<string>>(new Set());const [target,setTarget]=useState('');const [newName,setNewName]=useState('');
 const [scores,setScores]=useState<Record<string,number>>({});const [topN,setTopN]=useState(20);const lastClicked=useRef<number>(-1);
 useEffect(()=>{setSel(prev=>new Set([...prev].filter(f=>files.includes(f))));},[files]);
 // 移す先を選ぶと、そのキャラのタグ（髪・目・服）に似ている順に並べ替える
 useEffect(()=>{if(!target||!files.length){setScores({});return;}let live=true;
  apiGet<{ranked:{file:string;score:number}[]}>(`${base(pid)}/${jobId}/suggest?target=${encodeURIComponent(target)}`).then(r=>{if(live)setScores(Object.fromEntries(r.ranked.map(x=>[x.file,x.score])));}).catch(()=>{if(live)setScores({});});
  return()=>{live=false;};},[target,files,pid,jobId]);
 const ordered=useMemo(()=>Object.keys(scores).length?[...files].sort((a,b)=>(scores[b]??0)-(scores[a]??0)):files,[files,scores]);
 if(!files.length)return null;
 const toggle=(f:string,e?:{shiftKey:boolean})=>{const idx=ordered.indexOf(f);
  setSel(p=>{const n=new Set(p);
   if(e?.shiftKey&&lastClicked.current>=0){const [a,b]=[Math.min(lastClicked.current,idx),Math.max(lastClicked.current,idx)];const on=!p.has(f);for(let i=a;i<=b;i++){if(on)n.add(ordered[i]);else n.delete(ordered[i]);}}
   else if(n.has(f))n.delete(f);else n.add(f);return n;});lastClicked.current=idx;};
 const picked=ordered.filter(f=>sel.has(f));
 return <div style={{border:'1px solid var(--amber-7)',borderRadius:8,padding:12,marginTop:12}}>
  <Text as="div" size="3" weight="bold">未仕分けの画像（{files.length}枚）</Text>
  <Text as="p" size="1" color="gray" mt="1">自動でどのキャラにも入れられなかった画像です。画像をクリックして選び、キャラに移せます。画像の右下の「拡大」で大きく見られます。</Text>
  <Flex gap="2" align="center" wrap="wrap" mt="2">
   <Button size="1" variant="ghost" color="gray" onClick={()=>setSel(new Set(files))}>すべて選ぶ</Button>
   <Button size="1" variant="ghost" color="gray" onClick={()=>setSel(new Set())}>選択を解除</Button>
   <Text size="1" color="gray">{picked.length}枚を選択中（Shiftを押しながらクリックで範囲選択）</Text></Flex>
  <Flex gap="2" align="center" wrap="wrap" mt="2">
   <Select.Root value={target||undefined} onValueChange={setTarget}><Select.Trigger placeholder="移す先のキャラを選ぶ" aria-label="移す先のキャラ"/><Select.Content>{chars.map(c=><Select.Item key={c.folder} value={c.folder}>{c.folder}（{c.images_total}枚）</Select.Item>)}</Select.Content></Select.Root>
   {target&&Object.keys(scores).length>0&&<><Text size="1" color="gray">似ている順に並べています</Text>
    <TextField.Root size="1" type="number" style={{width:64}} min={1} value={String(topN)} aria-label="上位から選ぶ枚数" onChange={e=>setTopN(Math.max(1,Number(e.target.value)||1))}/>
    <Button size="1" variant="soft" color="gray" onClick={()=>setSel(new Set(ordered.slice(0,topN)))}>似ている上位{topN}枚を選ぶ</Button></>}
   <Button size="1" disabled={!picked.length||!target||busy} onClick={async()=>{if(await onAssign(picked,target,null))setSel(new Set());}}>選んだ{picked.length}枚をこのキャラに移す</Button>
   <TextField.Root size="1" style={{width:180}} placeholder="新しいキャラの名前" aria-label="新しいキャラの名前" value={newName} onChange={e=>setNewName(e.target.value)}/>
   <Button size="1" variant="soft" disabled={!picked.length||!newName.trim()||busy} onClick={async()=>{if(await onAssign(picked,null,newName.trim())){setSel(new Set());setNewName('');}}}>選んだ画像から新しいキャラを作る</Button></Flex>
  <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fill,minmax(96px,1fr))',gap:6,marginTop:10,maxHeight:560,overflowY:'auto'}}>
   {ordered.map((f,i)=><div key={f} style={{position:'relative'}}>
    <button type="button" onClick={e=>toggle(f,e)} aria-pressed={sel.has(f)} title={f.split('/').pop()} style={{padding:0,width:'100%',aspectRatio:'1',border:sel.has(f)?'3px solid var(--blue-9)':'1px solid var(--gray-6)',borderRadius:4,background:'var(--gray-3)',cursor:'pointer',display:'block'}}>
     <img src={fileUrl(pid,jobId,f)} alt={f.split('/').pop()||f} loading="lazy" style={{width:'100%',height:'100%',objectFit:'cover',borderRadius:3,display:'block',opacity:sel.has(f)?0.75:1}}/></button>
 {scores[f]!=null&&<span style={{position:'absolute',left:2,top:2,fontSize:10,padding:'0 4px',borderRadius:3,background:'rgba(0,0,0,.65)',color:'#fff'}}>{Math.round(scores[f]*100)}%</span>}
    <button type="button" onClick={()=>onPreview(ordered,i)} aria-label="拡大" style={{position:'absolute',right:2,bottom:2,fontSize:10,padding:'1px 4px',borderRadius:3,border:'none',background:'rgba(0,0,0,.65)',color:'#fff',cursor:'zoom-in'}}>拡大</button>
   </div>)}
  </div>
 </div>;
}

const MIN_AUTO_PICK=15;

function ResultView({job,jobId,pid,onReport,onBack,onImported,onError}:{job:Job;jobId:string;pid:number;onReport:(r:Report)=>void;onBack:()=>void;onImported:(mode:'new_projects'|'current_project',total:number)=>void;onError:(m:string)=>void}){
 const report=job.report;
 const chars=useMemo(()=>report?.characters??[],[report]);
 // 学習に使えるだけの枚数があるキャラだけを最初から選んでおく（少数のグループは折りたたみ・未選択）
 const [picked,setPicked]=useState<Set<string>>(()=>new Set(chars.filter(c=>c.images_total>=MIN_AUTO_PICK).map(c=>c.folder)));
 const [mergeMode,setMergeMode]=useState(false),[mergeSel,setMergeSel]=useState<Set<string>>(new Set()),[mergeTarget,setMergeTarget]=useState('');
 const [busy,setBusy]=useState(false),[mode,setMode]=useState<'new_projects'|'current_project'>('new_projects'),[result,setResult]=useState<ImportResult|null>(null),[confirm,setConfirm]=useState(false);
 const [preview,setPreview]=useState<{list:string[];idx:number}|null>(null);
 const seen=useRef(new Set(chars.map(c=>c.folder)));
 // 統合・名前変更で新しく現れたフォルダは取り込み対象に、消えたフォルダは対象から外す
 useEffect(()=>{
  const now=new Set(chars.map(c=>c.folder));const before=seen.current;
  setPicked(prev=>{const next=new Set<string>();for(const f of now){if(prev.has(f)||!before.has(f))next.add(f);}return next;});
  seen.current=now;setMergeSel(prev=>new Set([...prev].filter(f=>now.has(f))));
 },[chars]);
 if(!report)return <Callout.Root color="amber" size="1"><Callout.Text>結果のデータがありません。</Callout.Text></Callout.Root>;

 const call=async<T,>(path:string,body:unknown,ms=30000):Promise<T|null>=>{setBusy(true);try{return await apiPost<T>(`${base(pid)}/${jobId}${path}`,body,'POST',undefined,ms);}catch(e){onError(errText(e));return null;}finally{setBusy(false);}};
 const takeReport=(r:unknown)=>{const rep=(r&&typeof r==='object'&&'report' in r&&(r as {report:unknown}).report?(r as {report:Report}).report:r) as Report;if(rep&&Array.isArray(rep.characters))onReport(rep);};
 async function rename(folder:string,newName:string){const r=await call<unknown>('/rename',{folder,new_name:newName});if(r)takeReport(r);}
 async function merge(){const sources=[...mergeSel].filter(f=>f!==mergeTarget);const r=await call<unknown>('/merge',{target:mergeTarget,sources});if(r){takeReport(r);setMergeMode(false);setMergeSel(new Set());setMergeTarget('');}}
 async function assignFiles(files:string[],target:string|null,newName:string|null){const r=await call<unknown>('/assign',{files,target,new_name:newName},120000);if(r){takeReport(r);return true;}return false;}
 async function reveal(){const r=await call<{path:string}>('/reveal',{});if(r)onError('');}
 async function doImport(){
  setConfirm(false);const folders=chars.filter(c=>picked.has(c.folder)).map(c=>c.folder);
  const r=await call<ImportResult>('/import',{characters:folders,mode},300000);
  if(r){setResult(r);const total=r.imported.reduce((a,x)=>a+(x.images||0),0);onImported(mode,total);}
 }
 const selectedChars=chars.filter(c=>picked.has(c.folder));
 const selectedImages=selectedChars.reduce((a,c)=>a+c.images_total,0);
 const mergeCandidates=chars.filter(c=>mergeSel.has(c.folder));
 const stats:[string,string][]=[['動画の長さ',fmtDur(report.duration_s)],['シーン数',String(report.scenes)],['採用したフレーム',String(report.frames_kept)],['人物の切り抜き数',String(report.person_crops)],['出力枚数',`${report.images_written}枚`],['処理時間',fmtDur(report.wall_clock_s)],['1枚あたりの時間',fmtSec(report.per_output_image_s)],['（モデル読み込み除く）',fmtSec(report.per_output_image_excl_model_load_s)],['画質フィルタで除外した候補',`${report.damaged_candidates_tried}件`],['他キャラが入った切り抜き',`${report.crops_with_other_characters}件`],['未仕分けの画像',`${report.unassigned_images}枚`]];
 const timingRows=Object.entries(report.timing??{});

 const renderCard=(c:(typeof chars)[number])=><CharCard key={c.folder} c={c} pid={pid} jobId={jobId} busy={busy} checked={picked.has(c.folder)} onCheck={v=>setPicked(p=>{const n=new Set(p);if(v)n.add(c.folder);else n.delete(c.folder);return n;})}
  mergeMode={mergeMode} mergeChecked={mergeSel.has(c.folder)} onMergeCheck={v=>setMergeSel(p=>{const n=new Set(p);if(v)n.add(c.folder);else n.delete(c.folder);return n;})}
  onRename={nn=>void rename(c.folder,nn)} onPreview={(list,idx)=>setPreview({list,idx})}/>;

 return <div>
  {job.status==='cancelled'&&<Callout.Root color="amber" size="1" mb="3"><Callout.Text>途中で中止したため、ここまでの結果だけです。</Callout.Text></Callout.Root>}
  <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fit,minmax(170px,1fr))',gap:8}}>
   {stats.map(([k,v])=><div key={k} style={{background:'var(--gray-3)',borderRadius:6,padding:'6px 10px',minWidth:0}}><Text as="div" size="1" color="gray">{k}</Text><Text as="div" size="3" weight="bold" style={{overflowWrap:'anywhere'}}>{v}</Text></div>)}
  </div>
  {timingRows.length>0&&<details style={{marginTop:8}}><summary style={{cursor:'pointer'}}><Text size="1" color="gray">処理ごとの所要時間</Text></summary>
   <div style={{overflowX:'auto'}}><table style={{borderCollapse:'collapse',marginTop:6,fontSize:12,width:'100%'}}><thead><tr style={{textAlign:'left',color:'var(--gray-11)'}}><th style={{padding:'2px 8px 2px 0'}}>処理</th><th style={{padding:'2px 8px'}}>合計</th><th style={{padding:'2px 8px'}}>回数</th><th style={{padding:'2px 8px'}}>1回あたり</th></tr></thead>
    <tbody>{timingRows.map(([k,t])=><tr key={k} style={{borderTop:'1px solid var(--gray-5)'}}><td style={{padding:'2px 8px 2px 0'}}>{timingLabel(k)}</td><td style={{padding:'2px 8px'}}>{fmtDur(t.total_s)}</td><td style={{padding:'2px 8px'}}>{t.count}</td><td style={{padding:'2px 8px'}}>{fmtSec(t.per_unit_s)}</td></tr>)}</tbody></table></div></details>}

  <Flex gap="3" align="center" wrap="wrap" mt="4"><Text size="3" weight="bold">見つかったキャラ（{chars.length}）</Text>
   <Text size="1" color="gray">{selectedChars.length}キャラ・{selectedImages}枚を選択中</Text>
   <Button size="1" variant="ghost" color="gray" onClick={()=>setPicked(new Set(chars.map(c=>c.folder)))}>すべて選ぶ</Button>
   <Button size="1" variant="ghost" color="gray" onClick={()=>setPicked(new Set())}>選択を解除</Button>
   <Button size="1" variant={mergeMode?'solid':'soft'} onClick={()=>{setMergeMode(m=>!m);setMergeSel(new Set());setMergeTarget('');}} disabled={chars.length<2||busy}>{mergeMode?'統合をやめる':'統合する'}</Button>
   <Button size="1" variant="soft" color="gray" disabled={busy} onClick={()=>void reveal()}>Explorerで出力フォルダを開く</Button></Flex>

  {mergeMode&&<div style={{border:'1px dashed var(--blue-8)',borderRadius:8,padding:12,marginTop:8}}>
   <Text as="p" size="2">キャラが複数フォルダに割れているときに使います。統合したいキャラの「統合する」にチェックを入れ（2つ以上）、残す名前を選んでください。</Text>
   {mergeCandidates.length>=2?<>
    <RadioGroup.Root value={mergeTarget} onValueChange={setMergeTarget} aria-label="統合後に残すフォルダ" style={{marginTop:8}}>{mergeCandidates.map(c=><RadioGroup.Item key={c.folder} value={c.folder}>{c.folder}（{c.images_total}枚）をこの名前で残す</RadioGroup.Item>)}</RadioGroup.Root>
    <Button mt="3" disabled={!mergeTarget||busy} onClick={()=>void merge()}>{busy?<Spinner/>:null}{mergeCandidates.length}つを「{mergeTarget||'…'}」に統合</Button></>
    :<Text as="p" size="1" color="gray" mt="1">いま選択中：{mergeCandidates.length}つ</Text>}</div>}

  <div style={{display:'grid',gap:10,marginTop:10}}>
   {chars.filter(c=>c.images_total>=MIN_AUTO_PICK).map(renderCard)}
   {chars.some(c=>c.images_total<MIN_AUTO_PICK)&&<details open={mergeMode} style={{border:'1px solid var(--gray-5)',borderRadius:8,padding:'6px 10px'}}><summary style={{cursor:'pointer'}}><Text size="2">少数のグループ（{chars.filter(c=>c.images_total<MIN_AUTO_PICK).length}）</Text><Text size="1" color="gray"> — {MIN_AUTO_PICK}枚未満。学習には足りないため初期状態では選択していません。同じキャラが割れている場合は「統合する」で大きいグループへ寄せられます</Text></summary>
    <div style={{display:'grid',gap:10,marginTop:8}}>{chars.filter(c=>c.images_total<MIN_AUTO_PICK).map(renderCard)}</div></details>}
   {chars.length===0&&<Callout.Root color="amber" size="1"><Callout.Text>キャラとして認められるグループがありませんでした。「キャラとして認める最小の切り抜き数」を下げるか、「試し実行」を外して全体を処理してみてください。</Callout.Text></Callout.Root>}
  </div>

  <UnassignedPanel files={report.unassigned_files??report.unassigned_sample_files??[]} chars={chars} pid={pid} jobId={jobId} busy={busy} onAssign={assignFiles} onPreview={(l,i)=>setPreview({list:l,idx:i})}/>
  <Collapsed title="画質フィルタで除外したフレームの例" files={report.rejected_frame_examples??[]} pid={pid} jobId={jobId} onPreview={(l,i)=>setPreview({list:l,idx:i})} help="ブロックノイズやボケなどで傷んでいると判定して使わなかったフレームの例です。"/>

  <div style={{borderTop:'1px solid var(--gray-6)',marginTop:16,paddingTop:12}}>
   <Text as="div" size="3" weight="bold" mb="2">データセットに取り込む</Text>
   <RadioGroup.Root value={mode} onValueChange={v=>setMode(v as typeof mode)} aria-label="取り込み先">
    <RadioGroup.Item value="new_projects">選んだキャラごとに新しいプロジェクトを作る（推奨）</RadioGroup.Item>
    <RadioGroup.Item value="current_project">今のプロジェクトのデータセットに入れる</RadioGroup.Item></RadioGroup.Root>
   <Text as="p" size="1" color="gray" mt="1">LoRAは「1キャラ＝1つ」で作るのが基本なので、キャラごとに別プロジェクトにするのがおすすめです。衣装フォルダはインスタンスとして引き継がれます。</Text>
   <Flex gap="3" align="center" wrap="wrap" mt="3">
    <Button disabled={!selectedChars.length||busy||!!result} onClick={()=>setConfirm(true)}>{busy&&!mergeMode?<Spinner/>:null}{selectedChars.length}キャラ（{selectedImages}枚）を取り込む</Button>
    <Button variant="soft" color="gray" onClick={onBack}>別の動画・設定でやり直す</Button></Flex>
   {result&&<Callout.Root color="green" size="1" mt="3"><Callout.Text>
    {result.imported.length}キャラを取り込みました。{mode==='new_projects'?'新しいプロジェクトはプロジェクト一覧から開けます。':'今のプロジェクトのデータセットを更新しました。'}</Callout.Text></Callout.Root>}
   {result&&<ul style={{margin:'8px 0 0',paddingLeft:18,fontSize:13}}>{result.imported.map(x=><li key={x.folder+x.project_id}>{x.folder}：{x.images}枚{mode==='new_projects'?`（新プロジェクト #${x.project_id}）`:''}{typeof x.concepts==='number'?`、衣装 ${x.concepts}`:''}</li>)}
    {(result.skipped??[]).map((s,i)=><li key={`s${i}`} style={{color:'var(--amber-11)'}}>取り込まなかった：{typeof s==='string'?s:JSON.stringify(s)}</li>)}</ul>}
  </div>

  <AlertDialog.Root open={confirm} onOpenChange={setConfirm}><AlertDialog.Content maxWidth="460px">
   <AlertDialog.Title>取り込みの確認</AlertDialog.Title>
   <AlertDialog.Description size="2">{selectedChars.length}キャラ・合計{selectedImages}枚を{mode==='new_projects'?`${selectedChars.length}個の新しいプロジェクトに分けて`:'今のプロジェクトのデータセットに'}取り込みます。元の動画と出力フォルダは変更されません。</AlertDialog.Description>
   <ul style={{fontSize:13,margin:'8px 0',paddingLeft:18}}>{selectedChars.map(c=><li key={c.folder}>{c.folder}：{c.images_total}枚</li>)}</ul>
   <Flex gap="3" justify="end" mt="3"><AlertDialog.Cancel><Button variant="soft" color="gray">戻る</Button></AlertDialog.Cancel><Button onClick={()=>void doImport()}>取り込む</Button></Flex></AlertDialog.Content></AlertDialog.Root>

  <Dialog.Root open={!!preview} onOpenChange={o=>{if(!o)setPreview(null);}}>{preview&&<Dialog.Content maxWidth="min(92vw,900px)" aria-describedby={undefined}
   onKeyDown={e=>{if(e.key==='ArrowLeft')setPreview(p=>p&&{...p,idx:(p.idx-1+p.list.length)%p.list.length});if(e.key==='ArrowRight')setPreview(p=>p&&{...p,idx:(p.idx+1)%p.list.length});}}>
   <Dialog.Title size="2" style={{overflowWrap:'anywhere'}}>{preview.list[preview.idx]}（{preview.idx+1}/{preview.list.length}）</Dialog.Title>
   <img src={fileUrl(pid,jobId,preview.list[preview.idx])} alt={preview.list[preview.idx]} style={{maxWidth:'100%',maxHeight:'70vh',display:'block',margin:'0 auto',objectFit:'contain'}}/>
   <Flex gap="3" justify="between" mt="3"><Flex gap="2"><Button variant="soft" color="gray" onClick={()=>setPreview(p=>p&&{...p,idx:(p.idx-1+p.list.length)%p.list.length})}>前へ</Button><Button variant="soft" color="gray" onClick={()=>setPreview(p=>p&&{...p,idx:(p.idx+1)%p.list.length})}>次へ</Button></Flex>
    <Dialog.Close><Button>閉じる</Button></Dialog.Close></Flex></Dialog.Content>}</Dialog.Root>
 </div>;
}

// ---- 全体 ----
export default function CharacterVideoDataset({open,projectId,onAdded}:{open:boolean;projectId:number;onAdded:(count:number,skipped:number)=>void}){
 const [s,setS]=useState<Settings>(loadSettings);
 const [jobId,setJobId]=useState(''),[job,setJob]=useState<Job|null>(null),[error,setError]=useState(''),[busy,setBusy]=useState(false),[cancelling,setCancelling]=useState(false),[recent,setRecent]=useState<JobSummary[]>([]);
 useEffect(()=>{saveSettings(s);},[s]);

 const refreshJobs=useCallback(async()=>{
  try{const list=await apiGet<JobSummary[]>(`${base(projectId)}/jobs`);setRecent(list);return list;}catch{return [] as JobSummary[];}
 },[projectId]);
 // ダイアログを開いたとき、実行中のジョブがあればそれにつなぐ
 useEffect(()=>{
  if(!open)return;let live=true;
  void refreshJobs().then(list=>{if(!live)return;const run=list.find(j=>j.status==='running');if(run){setJobId(prev=>prev||run.job_id);}});
  return()=>{live=false;};
 },[open,refreshJobs]);
 useEffect(()=>{if(!open){setCancelling(false);}},[open]);

 // ポーリング：実行中だけ1.5秒おき。閉じる・ジョブ変更・終了で必ず止める
 useEffect(()=>{
  if(!open||!jobId)return;let live=true;let timer:number|undefined;
  const tick=async()=>{
   try{const r=await apiGet<Job>(`${base(projectId)}/${jobId}`);if(!live)return;setJob(r);
    if(r.status==='running'){timer=window.setTimeout(()=>void tick(),1500);}else{setCancelling(false);void refreshJobs();}}
   catch(e){if(!live)return;setError(errText(e));timer=window.setTimeout(()=>void tick(),4000);}
  };
  void tick();
  return()=>{live=false;if(timer!==undefined)window.clearTimeout(timer);};
 },[open,jobId,projectId,refreshJobs]);

 async function start(){
  setBusy(true);setError('');setJob(null);
  const body={...s,video_path:s.video_path.trim(),frame_interval:Math.max(0.1,s.frame_interval)};
  try{const r=await apiPost<{job_id:string}>(`${base(projectId)}/start`,body,'POST',undefined,30000);setJobId(r.job_id);}
  catch(e){setError(errText(e));}finally{setBusy(false);}
 }
 async function cancel(){setCancelling(true);try{await apiPost(`${base(projectId)}/${jobId}/cancel`,{});}catch(e){setError(errText(e));setCancelling(false);}}
 function back(){setJobId('');setJob(null);setError('');void refreshJobs();}
 function openJob(id:string){setError('');setJob(null);setJobId(id);}

 const running=job?.status==='running'||(!!jobId&&!job&&!error);
 return <div>
  <Text as="p" size="2" color="gray" mb="3">動画の中の人物をキャラごとに切り抜き、衣装ごとのフォルダに仕分けた「データセットの候補」を作ります。シーンごとの1枚抜きとは違い、登場人物ごとに複数枚を作ります。</Text>
  {error&&<Callout.Root color="red" size="1" mb="3" role="alert"><Callout.Text>{error}</Callout.Text></Callout.Root>}
  {!jobId&&<SetupForm s={s} setS={f=>setS(f)} running={false} busy={busy} onStart={()=>void start()} recent={recent} onOpenJob={openJob}/>}
  {jobId&&!job&&!error&&<Flex gap="2" align="center"><Spinner/><Text size="2">状況を確認しています…</Text></Flex>}
  {jobId&&!job&&error&&<Button variant="soft" color="gray" onClick={back}>設定に戻る</Button>}
  {job&&running&&<ProgressView job={job} onCancel={()=>void cancel()} cancelling={cancelling}/>}
  {job&&job.status==='error'&&<div><Callout.Root color="red" size="1"><Callout.Text>処理に失敗しました：{job.error||'原因不明のエラー'}</Callout.Text></Callout.Root>
   {job.log_tail.length>0&&<details style={{marginTop:8}}><summary style={{cursor:'pointer'}}><Text size="1" color="gray">詳しい経過（ログ）</Text></summary><pre style={{margin:'8px 0 0',padding:8,background:'var(--gray-3)',borderRadius:6,maxHeight:180,overflow:'auto',whiteSpace:'pre-wrap',overflowWrap:'anywhere',fontSize:11}}>{job.log_tail.join('\n')}</pre></details>}
   <Button mt="3" onClick={back}>設定に戻ってやり直す</Button></div>}
  {job&&job.status==='cancelled'&&!job.report&&<div><Callout.Root color="amber" size="1"><Callout.Text>中止しました。</Callout.Text></Callout.Root><Button mt="3" onClick={back}>設定に戻る</Button></div>}
  {job&&(job.status==='done'||(job.status==='cancelled'&&job.report))&&<ResultView key={jobId} job={job} jobId={jobId} pid={projectId} onReport={r=>setJob(j=>j&&{...j,report:r})} onBack={back} onError={setError}
   onImported={(mode,total)=>{if(mode==='current_project')onAdded(total,0);}}/>}
 </div>;
}
