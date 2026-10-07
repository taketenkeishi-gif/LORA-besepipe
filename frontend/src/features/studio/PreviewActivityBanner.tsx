import {useEffect,useRef,useState} from 'react';
import {Callout,Flex,Progress,Text} from '@radix-ui/themes';
import {Loader2,Pause,Clock,AlertTriangle} from 'lucide-react';
import type {PreviewActivity,TrainingStatus} from '../../types';

function useNow(intervalMs=1000){const [now,setNow]=useState(()=>Date.now());useEffect(()=>{const t=setInterval(()=>setNow(Date.now()),intervalMs);return()=>clearInterval(t);},[intervalMs]);return now;}
export function formatDuration(seconds:number|null|undefined){if(seconds==null||!Number.isFinite(seconds))return '—';const s=Math.max(0,Math.round(seconds));if(s<60)return `${s}秒`;const m=Math.floor(s/60);return m<60?`${m}分${String(s%60).padStart(2,'0')}秒`:`${Math.floor(m/60)}時間${m%60}分`;}

/** Seconds since this exact activity snapshot arrived, so server-side ages keep ticking between 5s polls. */
function useSinceReceived(activity:PreviewActivity|undefined){const received=useRef(Date.now());const key=activity?.generated_at;useEffect(()=>{received.current=Date.now();},[key]);const now=useNow();return Math.max(0,(now-received.current)/1000);}

export function previewPhaseLabel(a:PreviewActivity|undefined):string|null{
 if(!a?.active)return null;
 if(a.phase==='generating'||a.phase==='window_waiting')return a.training_paused?'プレビュー画像を生成中（学習計算は一時停止）':'プレビュー画像を生成中';
 if(a.phase==='waiting_gpu')return 'プレビュー生成のGPU待ち';
 if(a.phase==='deferred_until_training_ends')return '学習中（プレビューは学習終了後に生成）';
 return null;
}

export default function PreviewActivityBanner({activity}:{activity?:PreviewActivity}){
 const since=useSinceReceived(activity);
 if(!activity?.active)return null;
 const a=activity,total=a.total||0,done=a.done||0,failed=a.failed||0,cur=a.current_job;
 const stepFrac=cur&&cur.step!=null&&cur.step_max?Math.min(1,cur.step/cur.step_max):0;
 const overall=total?Math.min(100,((done+failed+stepFrac)/total)*100):0;
 const age=a.last_activity_age_seconds==null?null:a.last_activity_age_seconds+since;
 const elapsed=cur?.elapsed_seconds==null?null:cur.elapsed_seconds+since;
 const windowElapsed=a.window?.elapsed_seconds==null?null:a.window.elapsed_seconds+since;
 const windowRemain=a.window?.remaining_seconds==null?null:Math.max(0,a.window.remaining_seconds-since);
 const ordinal=Math.min(total,done+failed+1);
 let title='',tone:'blue'|'amber'|'gray'='blue',icon=<Loader2 size={16} className="wb-spin"/>;
 if(a.phase==='generating'){title=`プレビュー画像を生成中（エポック${a.epoch}の重み）`;}
 else if(a.phase==='window_waiting'){title=`プレビュー画像を生成中（エポック${a.epoch}の重み）・次の1枚を準備中`;}
 else if(a.phase==='waiting_gpu'){title=`GPUを待っています（エポック${a.epoch}のプレビュー）`;tone='amber';icon=<Clock size={16}/>;}
 else{title=`エポック${a.epoch}のプレビューは学習終了後に生成します`;tone='gray';icon=<Clock size={16}/>;}
 const stalled=!!a.stalled;
 return <div className="wb-preview-activity" data-tone={stalled?'amber':tone} data-phase={a.phase} data-stalled={stalled?'1':'0'} role="status" aria-live="polite">
  <Flex align="center" gap="2"><span className="wb-preview-activity-icon">{stalled?<AlertTriangle size={16}/>:icon}</span><Text size="2" weight="bold">{title}</Text></Flex>
  {(a.phase==='generating'||a.phase==='window_waiting')&&<>
   <Flex justify="between" mt="2"><Text size="2">{done}/{total}枚完了{a.phase==='generating'&&ordinal<=total?` ・ ${ordinal}枚目を生成中`:''}{failed?` ・ 失敗${failed}枚`:''}</Text><Text size="2" weight="bold">{overall.toFixed(0)}%</Text></Flex>
   <Progress value={overall} my="1" aria-label="プレビュー生成の全体進捗"/>
   {cur&&<Text as="p" size="1" color="gray">現在の1枚：{cur.step!=null&&cur.step_max?`${cur.step}/${cur.step_max}ステップ（${Math.round(stepFrac*100)}%）`:'ComfyUIの順番待ち・モデル準備中（ステップ数はまだ届いていません）'} ・ 経過 {formatDuration(elapsed)}</Text>}
   {a.training_paused&&<Text as="p" size="1" style={{marginTop:6}}><Pause size={11} style={{verticalAlign:'-1px'}}/> 学習の計算は<b>意図的に一時停止中</b>です（同じGPUでComfyUIが画像を生成するため）。Step・Lossが動かないのは正常で、完了すると自動で学習が再開します。この待機は{formatDuration(windowElapsed)}経過・上限まで残り最大{formatDuration(windowRemain)}。</Text>}
   {!a.training_paused&&<Text as="p" size="1" color="gray" style={{marginTop:6}}>学習は終了済みです。残りの画像をComfyUIで順番に生成しています。</Text>}
  </>}
  {a.phase==='waiting_gpu'&&<Text as="p" size="1" style={{marginTop:6}}>{a.pending}枚が待機中です。{a.waiting_reason||'他のGPU作業の完了を待っています。'} 空き次第、自動で生成を始めます。</Text>}
  {a.phase==='deferred_until_training_ends'&&<Text as="p" size="1" color="gray" style={{marginTop:6}}>学習は止まっていません。同じGPUでの同時実行を避けるため、画像{total}枚は学習が終わってから生成します。</Text>}
  {(a.phase==='generating'||a.phase==='window_waiting')&&<Text as="p" size="1" color="gray" style={{marginTop:4}}>ComfyUIからの最終更新：{age==null?'まだ受信なし':`${formatDuration(age)}前`}</Text>}
  {stalled&&<Callout.Root size="1" color="amber" mt="2"><Callout.Text>応答が止まっています。{formatDuration(age)}の間、ComfyUIからの進捗が更新されていません。最後に確認できた状態：{cur&&cur.step!=null&&cur.step_max?`1枚の生成 ${cur.step}/${cur.step_max}ステップ`:`${done}/${total}枚完了`}。ComfyUIが重い処理中のことがあるため、まず数分待ってください。回復しない場合は停止して保存済みLoRAから継続できます。</Callout.Text></Callout.Root>}
 </div>;
}

/** Heartbeat for the plain training phase: shows when the step counter last moved (client-observed). */
export function RunHeartbeat({status}:{status:TrainingStatus}){
 const last=useRef({steps:status.done_steps??0,at:Date.now(),run:status.run_id});
 if(last.current.run!==status.run_id){last.current={steps:status.done_steps??0,at:Date.now(),run:status.run_id};}
 else if((status.done_steps??0)!==last.current.steps){last.current={...last.current,steps:status.done_steps??0,at:Date.now()};}
 const now=useNow(),idle=Math.max(0,(now-last.current.at)/1000);
 if(status.status!=='training')return null;
 const act=status.preview_activity,paused=!!act?.active&&!!act.training_paused;
 const stuck=!paused&&idle>180;
 return <Text as="p" size="1" color={stuck?'amber':'gray'} data-heartbeat={stuck?'stuck':'ok'}>{paused?'Stepは一時停止中（プレビュー生成待ち）':stuck?`Stepが${formatDuration(idle)}進んでいません（モデル読込・保存・キャッシュ処理中の場合があります）`:`Step最終更新：${formatDuration(idle)}前`}</Text>;
}
