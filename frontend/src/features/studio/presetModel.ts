import type {Config} from './TrainingDock';

export const presetKeys=['preview_backend','preview_lora_strength','preview_extra_loras','training_goal','model_family','training_engine','base_checkpoint_path','training_memory_mode','epochs','repeats','rank','alpha','resolution','learning_rate','train_batch_size','optimizer','scheduler','save_every_n_epochs','cache_latents','cache_latents_to_disk','advanced','preview_base_checkpoint_path','preview_steps','preview_cfg'] as const;
export type PresetCategory='character'|'style';
export type Preset={format:'lora-studio-training-preset';version:1;name:string;category:PresetCategory;config:Partial<Config>;recommendation?:{status?:string;basis?:string;rationale?:string;adjustment?:string}};
export type Entry={id:string;file_path:string;name:string;payload:Preset|null;error:string|null;modified:number;size:number};

export const pick=(config:Config)=>Object.fromEntries(presetKeys.map(k=>[k,config[k]])) as Partial<Config>;

export function readPreset(value:unknown):Preset{
 if(!value||typeof value!=='object')throw Error('学習設定JSONではありません');const p=value as Preset;
 if(p.format!=='lora-studio-training-preset'||p.version!==1)throw Error('LoRA Studioの学習設定JSON（version 1）を選んでください。ComfyUIのワークフローJSONとは別です。');
 if(typeof p.name!=='string'||!p.name.trim()||p.name.length>120||!['character','style'].includes(p.category)||!p.config||typeof p.config!=='object'||Array.isArray(p.config))throw Error('プリセット名・分類・設定の形式が不正です');
 if(Object.keys(p.config).some(k=>!presetKeys.includes(k as typeof presetKeys[number])))throw Error('対応していない設定項目が含まれています');
 const c=p.config;
 if(c.preview_backend!==undefined&&c.preview_backend!=='comfyui')throw Error('生成先はComfyUIを指定してください');
 if(c.preview_lora_strength!==undefined&&(typeof c.preview_lora_strength!=='number'||!Number.isFinite(c.preview_lora_strength)||c.preview_lora_strength<0||c.preview_lora_strength>2))throw Error('LoRA強度は0〜2で指定してください');
 if(c.preview_extra_loras!==undefined&&(!Array.isArray(c.preview_extra_loras)||c.preview_extra_loras.length>8||c.preview_extra_loras.some(e=>typeof e.name!=='string'||!e.name||typeof e.strength!=='number'||!Number.isFinite(e.strength)||e.strength<0||e.strength>2)))throw Error('追加LoRAの形式を確認してください');
 if(!['anima','sdxl','krea2'].includes(c.model_family||''))throw Error('モデル系統が不正です');
 const ranges:Record<string,[number,number]>={epochs:[1,10000],repeats:[1,1000],rank:[1,1024],alpha:[.01,1024],resolution:[256,4096],learning_rate:[.000000001,10],train_batch_size:[1,128],save_every_n_epochs:[1,10000],preview_steps:[1,100],preview_cfg:[0,30]};
 for(const [key,[min,max]] of Object.entries(ranges)){const v=(c as any)[key];if(v==null&&key.startsWith('preview_'))continue;if(typeof v!=='number'||!Number.isFinite(v)||v<min||v>max)throw Error(`${key}の値が不正です`);if(!['alpha','learning_rate','preview_cfg'].includes(key)&&!Number.isInteger(v))throw Error(`${key}は整数で指定してください`);}
 if((c.resolution||0)%64!==0)throw Error('解像度は64の倍数で指定してください');
 for(const key of ['base_checkpoint_path','preview_base_checkpoint_path','optimizer','scheduler','training_engine','training_memory_mode'])if(typeof (c as any)[key]!=='string')throw Error(`${key}の形式が不正です`);
 for(const key of ['cache_latents','cache_latents_to_disk'])if(typeof (c as any)[key]!=='boolean')throw Error(`${key}の形式が不正です`);
 if(!c.advanced||typeof c.advanced!=='object'||Array.isArray(c.advanced))throw Error('詳細設定の形式が不正です');
 return {...p,name:p.name.trim(),config:{...c,training_goal:p.category==='style'?'art_style':'character_identity'}};
}

const familyNames:Record<string,string>={anima:'Anima',sdxl:'Illustrious / SDXL',krea2:'KREA2'};
const memoryNames:Record<string,string>={low_vram:'省VRAM',balanced:'バランス',standard:'速度優先'};
const labels:Record<string,string>={training_goal:'学習目的',model_family:'モデル系統',base_checkpoint_path:'ベースモデル',training_memory_mode:'VRAMの使い方',epochs:'学習回数（エポック）',repeats:'繰り返し',rank:'Rank',alpha:'Alpha',resolution:'解像度',learning_rate:'学習率',train_batch_size:'バッチ',optimizer:'Optimizer',scheduler:'スケジューラ',save_every_n_epochs:'保存間隔（エポック）',cache_latents:'潜在表現キャッシュ',cache_latents_to_disk:'キャッシュをディスクに保存',advanced:'詳細設定',preview_steps:'プレビューSteps',preview_cfg:'プレビューCFG',preview_base_checkpoint_path:'プレビュー用モデル',preview_lora_strength:'プレビューLoRA強度',preview_extra_loras:'プレビュー追加LoRA',training_engine:'学習エンジン',preview_backend:'プレビュー生成先'};
export const labelOf=(key:string)=>labels[key]||key;
const baseName=(p:string)=>p.split(/[\\/]/).filter(Boolean).pop()||'（未指定）';
export function formatValue(key:string,value:unknown):string{
 if(value===undefined||value===null||value==='')return '（未指定）';
 if(key==='training_goal')return value==='art_style'?'スタイル':'キャラクター';
 if(key==='model_family')return familyNames[String(value)]||String(value);
 if(key==='training_memory_mode')return memoryNames[String(value)]||String(value);
 if(key==='base_checkpoint_path'||key==='preview_base_checkpoint_path')return baseName(String(value));
 if(typeof value==='boolean')return value?'ON':'OFF';
 if(key==='advanced'){const n=Object.keys(value as object).length;return `${n}項目`;}
 if(key==='preview_extra_loras')return `${(value as unknown[]).length}件`;
 return String(value);
}
/** Readable one-screen summary of what a preset contains. */
export function summarize(p:Preset):[string,string][]{
 const c=p.config,f=(k:string)=>formatValue(k,(c as any)[k]);
 return [['学習目的',p.category==='style'?'スタイル':'キャラクター'],['モデル系統',f('model_family')],['ベースモデル',f('base_checkpoint_path')],
  ['Rank / Alpha',`${f('rank')} / ${f('alpha')}`],['エポック × 繰り返し',`${f('epochs')} × ${f('repeats')}`],['学習率',f('learning_rate')],['解像度',`${f('resolution')}px`],['バッチ',f('train_batch_size')],
  ['Optimizer / スケジューラ',`${f('optimizer')} / ${f('scheduler')}`],...(c.training_memory_mode&&c.model_family==='anima'?[['VRAMの使い方',f('training_memory_mode')] as [string,string]]:[]),
  ['保存間隔',`${f('save_every_n_epochs')}エポックごと`],['詳細設定',f('advanced')]];
}
function stable(v:unknown):string{if(Array.isArray(v))return '['+v.map(stable).join(',')+']';if(v&&typeof v==='object')return '{'+Object.keys(v).sort().map(k=>JSON.stringify(k)+':'+stable((v as any)[k])).join(',')+'}';return JSON.stringify(v)??'undefined';}
// The dock drops advanced.unet_lr on load (the top-level learning rate is the single source), so ignore it on both sides.
const normalizeValue=(key:string,v:unknown)=>{if(key==='advanced'&&v&&typeof v==='object'){const {unet_lr:_,...rest}=v as Record<string,unknown>;return rest;}return v;};
export type Difference={key:string;label:string;current:string;preset:string};
/** Settings that the preset defines and that currently differ from the live form. */
export function differences(current:Config,p:Preset):Difference[]{
 const out:Difference[]=[];
 for(const key of Object.keys(p.config)){
  const a=normalizeValue(key,(current as any)[key]),b=normalizeValue(key,(p.config as any)[key]);
  if(stable(a)===stable(b))continue;
  if(key==='advanced'){
   const ka=(a||{}) as Record<string,unknown>,kb=(b||{}) as Record<string,unknown>;
   const changed=[...new Set([...Object.keys(ka),...Object.keys(kb)])].filter(k=>stable(ka[k])!==stable(kb[k]));
   out.push({key,label:'詳細設定',current:`${changed.length}項目が異なる`,preset:changed.slice(0,4).join('、')+(changed.length>4?' ほか':'')});continue;
  }
  out.push({key,label:labelOf(key),current:formatValue(key,(current as any)[key]),preset:formatValue(key,(p.config as any)[key])});
 }
 return out;
}
export const safeFileName=(name:string)=>name.replace(/[<>:"/\\|?*\x00-\x1f]/g,'_');
export function downloadJson(p:Preset){
 const url=URL.createObjectURL(new Blob([JSON.stringify(p,null,2)],{type:'application/json'}));
 const a=document.createElement('a');a.href=url;a.download=safeFileName(p.name)+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),10000);
}
export const categoryName=(c:string)=>c==='style'?'スタイル':'キャラクター';
