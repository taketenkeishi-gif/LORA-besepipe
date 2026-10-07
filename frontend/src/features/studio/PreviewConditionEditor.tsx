import {useEffect,useState} from 'react';
import {Button,Dialog,Flex,Text,TextArea,Select,Callout,Spinner,Checkbox} from '@radix-ui/themes';
import NumericInput from './NumericInput';
import TagEditorDialog from './TagEditorField';
import {apiGet,apiPost} from '../../lib/api';
export type OutfitPreview={trigger:string;enabled:boolean;tags:string[];negative_tags?:string[]};
export type PreviewCondition={prompt:string;negative_prompt:string;seed:number;resolution:number;steps:number;cfg:number;sampler:string;scheduler:string;outfits?:OutfitPreview[]};
export default function PreviewConditionEditor({projectId,initial,onSaved,onClose,family='anima'}:{family?:string;projectId:number;initial:PreviewCondition;onSaved:(snapshot:{id:number;name:string;payload:PreviewCondition})=>void;onClose:()=>void}){
 const [draft,setDraft]=useState(initial),[busy,setBusy]=useState(false),[error,setError]=useState('');
 const [trigger,setTrigger]=useState<string|null>(null),[triggerError,setTriggerError]=useState('');
 const [tagEditor,setTagEditor]=useState(false);
 // 服装（学習インスタンス）ごとのプレビュー設定。保存済みの設定があればそれを、なければ学習データの字幕から自動で選んだタグを使う
 const [outfits,setOutfits]=useState<(OutfitPreview&{name:string;images:number;auto:string[];autoNeg:string[]})[]>([]);
 useEffect(()=>{let live=true;void apiGet<{outfits:{name:string;trigger:string;tags:string[];negative_tags:string[];images:number}[]}>(`/training/outfit-preview-defaults/${projectId}`).then(r=>{if(!live)return;
  const saved=new Map((initial.outfits??[]).map(o=>[o.trigger,o]));
  setOutfits(r.outfits.map(o=>({name:o.name,images:o.images,auto:o.tags,autoNeg:o.negative_tags,trigger:o.trigger,enabled:saved.get(o.trigger)?.enabled??true,tags:saved.get(o.trigger)?.tags??o.tags,negative_tags:saved.get(o.trigger)?.negative_tags??o.negative_tags})));}).catch(()=>{});return()=>{live=false;};},[projectId]);
 // 実際に生成へ渡すプロンプト（バックエンドの prompt_for_outfit と同じ規則）
 const effective=(o:OutfitPreview)=>{const drop=new Set([...outfits.map(x=>x.trigger),...outfits.filter(x=>x.trigger!==o.trigger).flatMap(x=>x.tags)].map(t=>t.toLowerCase()));
  const base=draft.prompt.split(',').map(t=>t.trim()).filter(t=>t&&!drop.has(t.toLowerCase()));const own=o.tags.filter(t=>!base.some(b=>b.toLowerCase()===t.toLowerCase()));
  const pos=[...base.slice(0,1),o.trigger,...own,...base.slice(1)].join(', ');
  const negExtra=(o.negative_tags??[]).filter(t=>!draft.negative_prompt.toLowerCase().includes(t.toLowerCase()));
  return {pos,neg:[draft.negative_prompt.trim(),...negExtra].filter(Boolean).join(', ')};};
 const setOutfit=(trigger:string,patch:Partial<OutfitPreview>)=>setOutfits(list=>list.map(o=>o.trigger===trigger?{...o,...patch}:o));
 useEffect(()=>{let live=true;void apiGet<{trigger_token:string}>(`/training/trigger/${projectId}`).then(s=>{if(live){const token=s.trigger_token.trim();setTrigger(token);let include=true;try{include=localStorage.getItem(`preview-include-trigger:${projectId}`)!=='false';}catch{}if(include&&token)setDraft(d=>({...d,prompt:[token,...d.prompt.split(/[,\n]/).map(t=>t.trim()).filter(t=>t&&t.toLowerCase()!==token.toLowerCase())].join(', ')}));}}).catch(e=>{if(live)setTriggerError(e.message);});return()=>{live=false;};},[projectId]);
 const containsTrigger=!!trigger&&draft.prompt.split(/[,\n]/).some(tag=>tag.trim().toLocaleLowerCase()===trigger.toLocaleLowerCase());
 function toggleTrigger(include:boolean){if(!trigger)return;setDraft(d=>{const remaining=d.prompt.split(/[,\n]/).filter(tag=>tag.trim().toLocaleLowerCase()!==trigger.toLocaleLowerCase()).map(tag=>tag.trim()).filter(Boolean);return {...d,prompt:(include?[trigger,...remaining]:remaining).join(', ')};});}
 const update=(key:keyof PreviewCondition,value:string|number)=>setDraft(d=>({...d,[key]:value}));
 async function save(){
  if(!draft.prompt.trim()||!Number.isInteger(draft.seed)||draft.seed<0||draft.seed>2147483647||!Number.isInteger(draft.steps)||draft.steps<1||draft.steps>100||draft.cfg<0||draft.cfg>30){setError('プロンプトを入力し、Seedは0〜2147483647、Stepsは1〜100、CFGは0〜30で指定してください。');return;}
  setBusy(true);setError('');try{
   const created=await apiPost<{profile:{id:number}}>('/preview-profiles',{...draft,outfits:outfits.map(o=>({trigger:o.trigger,enabled:o.enabled,tags:o.tags,negative_tags:o.negative_tags??[]})),project_id:projectId,name:`プレビュー条件 ${new Date().toLocaleString('ja-JP')}`,model_family:family});
   const snapshot=await apiPost<{id:number;name:string;payload:PreviewCondition}>(`/basepipe/projects/${projectId}/preview-profile-snapshots/${created.profile.id}`,{});
   if(trigger)localStorage.setItem(`preview-include-trigger:${projectId}`,String(containsTrigger));
   onSaved(snapshot);onClose();
  }catch(e){setError(e instanceof Error?e.message:String(e));}finally{setBusy(false);}
 }
 return <Dialog.Root open onOpenChange={v=>{if(!v&&!busy)onClose();}}><Dialog.Content maxWidth="680px"><Dialog.Title>学習中のプレビュー条件</Dialog.Title><Dialog.Description size="2">次回の学習に使用する条件です。実行済み・実行中のRunや保存画像は変更しません。</Dialog.Description>
  <Flex justify="end" mt="2"><Button size="1" variant="soft" onClick={()=>setTagEditor(true)}>タグエディターで編集</Button></Flex>
  <TagEditorDialog open={tagEditor} positive={draft.prompt} negative={draft.negative_prompt} onClose={()=>setTagEditor(false)} onApply={(p,n)=>setDraft(d=>({...d,prompt:p,negative_prompt:n}))}/>
  <label><Text as="div" size="1" mt="3">プロンプト</Text><TextArea aria-label="自動プレビューのプロンプト" rows={4} value={draft.prompt} onChange={e=>update('prompt',e.target.value)}/></label>
  <Flex align="center" gap="2" mt="2"><Checkbox id="preview-include-trigger" aria-label="登録済みトリガーワードを含める" checked={containsTrigger} disabled={busy||!trigger} onCheckedChange={v=>toggleTrigger(v===true)}/><Text as="label" htmlFor="preview-include-trigger" size="2">登録済みトリガーワードを含める{trigger?`（${trigger}）`:''}</Text></Flex>
  {trigger===''&&<Text as="p" size="1" color="gray">学習画面でトリガーワードを登録すると使えます。</Text>}{triggerError&&<Text as="p" size="1" color="red">トリガーワードを取得できませんでした：{triggerError}</Text>}
  {outfits.length>0&&<div style={{marginTop:16,border:'1px solid var(--gray-6)',borderRadius:8,padding:12}}>
   <Text as="div" size="2" weight="medium">服装（インスタンス）ごとのプレビュー</Text>
   <Text as="p" size="1" color="gray" mt="1">上のプロンプトを共通の土台にして、服装ごとにトリガーと下のタグだけを差し替えて、1エポックごとに服装の数だけ画像を作ります。チェックを外した服装は作りません。</Text>
   <Flex direction="column" gap="3" mt="2">{outfits.map(o=><div key={o.trigger} style={{minWidth:0}}>
    <Text as="label" size="2"><Flex gap="2" align="center"><Checkbox checked={o.enabled} onCheckedChange={v=>setOutfit(o.trigger,{enabled:v===true})} aria-label={`${o.name}のプレビューを作る`}/><span style={{overflowWrap:'anywhere'}}>{o.name}</span><Text size="1" color="gray">{o.trigger} · {o.images}枚</Text></Flex></Text>
    <Flex gap="2" mt="1" align="center"><TextArea size="1" style={{flex:1,minWidth:0}} rows={2} disabled={!o.enabled} aria-label={`${o.name}の服装タグ`} value={o.tags.join(', ')} placeholder="この服装を表すタグ（カンマ区切り）" onChange={e=>setOutfit(o.trigger,{tags:e.target.value.split(',').map(t=>t.trim()).filter(Boolean)})}/>
     <Button size="1" variant="soft" color="gray" disabled={!o.enabled} title="学習データの字幕から、この服装に特徴的なタグを自動で選び直します" onClick={()=>setOutfit(o.trigger,{tags:o.auto,negative_tags:o.autoNeg})}>自動に戻す</Button></Flex>
    <TextArea size="1" mt="1" rows={1} disabled={!o.enabled} aria-label={`${o.name}の特徴ではないタグ（既定では出さない）`} value={(o.negative_tags??[]).join(', ')} placeholder="この服装そのものの特徴ではないタグ（既定では出さない。必要ならプロンプトに足して再現できます。他の服装にだけあるものを自動で入れています）" onChange={e=>setOutfit(o.trigger,{negative_tags:e.target.value.split(',').map(t=>t.trim()).filter(Boolean)})}/>
    {o.enabled&&<details style={{marginTop:4}}><summary style={{cursor:'pointer'}}><Text size="1" color="gray">実際に使われるプロンプトを見る</Text></summary>
     <pre style={{margin:'4px 0 0',padding:8,background:'var(--gray-3)',borderRadius:6,whiteSpace:'pre-wrap',overflowWrap:'anywhere',fontSize:11}}>{`ポジティブ: ${effective(o).pos}
ネガティブ: ${effective(o).neg}`}</pre></details>}
   </div>)}</Flex></div>}
  <label><Text as="div" size="1" mt="3">ネガティブ</Text><TextArea aria-label="自動プレビューのネガティブ" rows={2} value={draft.negative_prompt} onChange={e=>update('negative_prompt',e.target.value)}/></label>
  <div className="wb-pair" style={{marginTop:16}}><label><Text size="1">解像度</Text><Select.Root value={String(draft.resolution)} onValueChange={v=>update('resolution',Number(v))}><Select.Trigger aria-label="自動プレビューの解像度"/><Select.Content>{[512,768,1024,1216,1536].map(n=><Select.Item key={n} value={String(n)}>{n} × {n}</Select.Item>)}</Select.Content></Select.Root></label>{([['seed','Seed',0,2147483647,1],['steps','Steps',1,100,1],['cfg','CFG',0,30,.1]] as const).map(([key,label,min,max,step])=><label key={key}><Text size="1">{label}</Text><NumericInput aria-label={`プレビュー条件 ${label}`} value={draft[key]} min={min} max={max} step={step} onValueChange={v=>update(key,v)}/></label>)}
  {([['sampler','Sampler',['euler','euler_ancestral','dpmpp_2m','dpmpp_2m_sde']],['scheduler','Scheduler',['simple','normal','karras','sgm_uniform']]] as const).map(([key,label,options])=><label key={key}><Text size="1">{label}</Text><Select.Root value={draft[key]} onValueChange={v=>update(key,v)}><Select.Trigger aria-label={`プレビュー条件 ${label}`}/><Select.Content>{[...new Set([...options,draft[key]])].filter(Boolean).map(v=><Select.Item key={v} value={v}>{v}</Select.Item>)}</Select.Content></Select.Root></label>)}</div>
  {error&&<Callout.Root color="red" mt="3"><Callout.Text>{error}</Callout.Text></Callout.Root>}<Flex gap="3" justify="end" mt="4"><Button color="gray" variant="soft" disabled={busy} onClick={onClose}>キャンセル</Button><Button disabled={busy} onClick={()=>void save()}>{busy&&<Spinner/>}次回の学習に使用</Button></Flex>
 </Dialog.Content></Dialog.Root>;
}
