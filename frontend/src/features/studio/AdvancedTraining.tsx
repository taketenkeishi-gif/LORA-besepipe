import {useEffect,useState} from 'react';
import {Button,Checkbox,Text,Select,TextArea,TextField} from '@radix-ui/themes';
import {RotateCcw} from 'lucide-react';
import NumericInput from './NumericInput';
import {apiGet} from '../../lib/api';

export type AdvancedValues=Record<string,number|string|boolean|string[]>;
type Field={key:string;label:string;group:string;type:string;default:number|string|boolean|string[];min:number|null;max:number|null;choices:(number|string)[]|null;help:string};
export default function AdvancedTraining({family,values,onChange,learningRate,scheduler,memoryMode}:{family:string;values:AdvancedValues;onChange:(v:AdvancedValues)=>void;learningRate:number;scheduler:string;memoryMode:string}){
  const [fields,setFields]=useState<Field[]>([]),[search,setSearch]=useState(''),[error,setError]=useState('');
  useEffect(()=>{let current=true;setFields([]);void apiGet<{fields:Field[]}>(`/training/advanced-catalog?model_family=${family}`).then(r=>{if(current){setFields(r.fields.filter(f=>f.key!=='unet_lr'));setError('');}}).catch(e=>{if(current)setError(String(e));});return()=>{current=false;};},[family]);
  const effective=(f:Field)=>values[f.key]??(f.key==='unet_lr'||f.key==='text_encoder_lr'?learningRate:f.default);
  const value=(k:string)=>values[k]??fields.find(f=>f.key===k)?.default;
  function set(key:string,v:AdvancedValues[string]){const next={...values,[key]:v};if(key==='cache_text_encoder_outputs'&&v===false)next.cache_text_encoder_outputs_to_disk=false;if(key==='network_train_unet_only'&&v===false){next.cache_text_encoder_outputs=false;next.cache_text_encoder_outputs_to_disk=false;}if(key==='loss_type'&&['huber','smooth_l1'].includes(String(v))&&!next.huber_schedule)next.huber_schedule='constant';onChange(next);}
  function reason(f:Field):string{
    if(f.key==='lr_warmup_steps'&&['constant','adafactor'].includes(scheduler))return 'ウォームアップ対応のスケジューラで使用します';
    if(f.key==='text_encoder_lr'&&value('network_train_unet_only'))return '本体のみ学習をオフにすると指定できます';
    if(f.key.startsWith('huber_')&&!['huber','smooth_l1'].includes(String(value('loss_type'))))return 'Huber / smooth_l1を選択すると指定できます';
    if(f.key==='lr_scheduler_power'&&scheduler!=='polynomial')return 'Polynomialで使用します';
    if(f.key==='lr_scheduler_num_cycles'&&scheduler!=='cosine_with_restarts')return 'cosine_with_restartsで使用します';
    if(f.key==='discrete_flow_shift'&&!['sigma','shift'].includes(String(value('timestep_sampling'))))return 'sigma / shiftで使用します';
    if(f.key==='sigmoid_scale'&&!['sigmoid','shift','flux_shift'].includes(String(value('timestep_sampling'))))return 'sigmoid / shift / flux_shiftで使用します';
    if(f.key==='cache_text_encoder_outputs_to_disk'&&!value('cache_text_encoder_outputs'))return 'Text Encoderキャッシュを有効にしてください';
    if(f.key==='persistent_data_loader_workers'&&value('max_data_loader_n_workers')===0)return 'worker数が1以上の場合に使用します';
    return '';
  }
  const visible=fields.filter(f=>`${f.label} ${f.key}`.toLowerCase().includes(search.toLowerCase()));
  return <div className="wb-native-advanced">
    <TextField.Root aria-label="詳細設定を検索" placeholder="設定名で検索…" value={search} onChange={e=>setSearch(e.target.value)}/>
    <Text as="p" size="1" color="gray">変更は次回の学習に保存されます。↶でその項目を既定値に戻せます。{family==='anima'&&`CPU退避は上のVRAM設定に従います（${memoryMode==='standard'?'0':memoryMode==='balanced'?'12':'26'}層）。`}</Text>
    {error&&<Text color="red">設定一覧を読み込めませんでした：{error}</Text>}
    {!fields.length&&!error&&<Text size="1">{['anima','sdxl'].includes(family)?'設定を読み込み中…':'このエンジンの詳細設定には未対応です。Anima / SDXL用の設定は適用しません。'}</Text>}
    {[...new Set(visible.map(f=>f.group))].map(group=><details key={group} className="wb-advanced-group" open={search?true:undefined}><summary>{group}<Text size="1" color="gray">{visible.filter(f=>f.group===group).length}項目</Text></summary><div className="wb-pair">{visible.filter(f=>f.group===group).map(f=>{const v=effective(f),why=reason(f),changed=Object.prototype.hasOwnProperty.call(values,f.key);return <div className="wb-advanced-field" key={f.key} style={f.type==='lines'?{gridColumn:'1 / -1'}:undefined}>
      <div className="wb-advanced-label"><Text as="label" size="1" htmlFor={`advanced-${f.key}`}>{f.label}</Text>{changed&&<Button type="button" size="1" variant="ghost" color="gray" aria-label={`${f.label}を既定値に戻す`} onClick={()=>{const next={...values};delete next[f.key];onChange(next);}}><RotateCcw size={12}/></Button>}</div>
      {f.type==='bool'?<Checkbox id={`advanced-${f.key}`} aria-label={f.label} disabled={!!why} checked={v===true} onCheckedChange={x=>set(f.key,x===true)}/>:f.type==='choice'?<Select.Root disabled={!!why} value={String(v)} onValueChange={x=>set(f.key,typeof f.default==='number'?Number(x):x)}><Select.Trigger id={`advanced-${f.key}`} aria-label={f.label} style={{width:'100%'}}/><Select.Content>{f.choices?.filter(x=>!(family==='anima'&&f.key==='huber_schedule'&&x==='snr')).map(x=><Select.Item key={x} value={String(x)}>{x}</Select.Item>)}</Select.Content></Select.Root>:f.type==='lines'?<TextArea id={`advanced-${f.key}`} aria-label={f.label} value={Array.isArray(v)?v.join('\n'):''} onChange={e=>set(f.key,e.target.value.split('\n'))} onBlur={()=>{if(Array.isArray(v))set(f.key,v.map(x=>x.trim()).filter(Boolean));}}/>:<NumericInput id={`advanced-${f.key}`} aria-label={f.label} disabled={!!why} value={Number(v)} min={f.min??undefined} max={f.max??undefined} step={f.type==='int'?1:'any'} onValueChange={x=>set(f.key,x)}/>}
      {['int','float'].includes(f.type)&&(Number(v)<Number(f.min)||Number(v)>Number(f.max)||(f.type==='int'&&!Number.isInteger(Number(v))))&&<Text as="p" size="1" color="red">{f.min}〜{f.max}の{f.type==='int'?'整数':'数値'}を指定してください</Text>}
      {(why||f.help)&&<Text as="p" size="1" color="gray">{why||f.help}</Text>}
    </div>;})}</div></details>)}
    {family==='anima'&&<Text as="p" size="1" color="gray">Animaで無効になるSDXL専用ノイズ補正・Min-SNR・FP8は表示していません。</Text>}
    <Text as="p" size="1" color="gray">現在の「保存済みLoRAから継続」は重みの再利用です。Optimizerを含む完全な状態復元は未対応です。</Text>
  </div>;
}
