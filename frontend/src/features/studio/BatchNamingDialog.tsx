import NumericInput from './NumericInput';
import {useEffect,useRef,useState} from 'react';
import {Button,Dialog,Flex,Text,TextField,Select,Callout,Spinner,Badge} from '@radix-ui/themes';
import {desktop,desktopError,type NamingPlan,type NamingOptions} from './desktopBridge';
type Props={projectId:number;title:string;relatives:string[];protectedSource:boolean;onClose:()=>void;onApplied:(result:{folder:string;relatives:string[];token:string;undoable:boolean})=>void};
export default function BatchNamingDialog({projectId,title:initialTitle,relatives,protectedSource,onClose,onApplied}:Props){
 const key=`batch-naming:${projectId}`;
 const saved=useRef<any>((()=>{try{return JSON.parse(localStorage.getItem(key)||'{}');}catch{return {};}})()).current;
 const [mode,setMode]=useState<'rename'|'copy'>(protectedSource?'copy':saved.mode||'rename'),[title,setTitle]=useState(saved.title||initialTitle),[template,setTemplate]=useState(saved.template||'{title}_{index}'),[start,setStart]=useState(saved.start??1),[digits,setDigits]=useState(saved.digits||4);
 const [numbering,setNumbering]=useState<'append'|'manual'>(saved.numbering||'append');
 const [plan,setPlan]=useState<NamingPlan|null>(null),[pending,setPending]=useState(false),[error,setError]=useState('');
 const config:NamingOptions={project_id:projectId,relatives,mode,title,template,start,digits,numbering};const signature=JSON.stringify(config),current=useRef(signature);current.current=signature;
 useEffect(()=>{setPlan(null);localStorage.setItem(key,JSON.stringify({mode,title,template,start,digits,numbering}));},[signature]);
 async function preview(){const version=signature;setPending(true);setError('');try{const result=await desktop()!.previewNaming(config);if(current.current===version)setPlan(result);}catch(e){setError(desktopError(e));}finally{setPending(false);}}
 async function apply(){if(!plan)return;setPending(true);setError('');try{const result=await desktop()!.applyNaming({...config,expected:plan});onApplied(result);onClose();}catch(e){setError(desktopError(e));setPlan(null);}finally{setPending(false);}}
 const valid=!!title.trim()&&!!template.trim()&&Number.isInteger(start)&&start>=0&&start<=99999999&&Number.isInteger(digits)&&digits>=1&&digits<=8&&relatives.length<=500;
 return <Dialog.Root open onOpenChange={v=>{if(!v&&!pending)onClose();}}><Dialog.Content maxWidth="760px"><Dialog.Title>名前をまとめて整理</Dialog.Title><Dialog.Description size="2" mb="4">選択した{relatives.length}枚を、現在の一覧順で命名します。画像と同名TXTを一緒に扱います。1回500枚まで。</Dialog.Description><Flex direction="column" gap="3">
 <Select.Root value={mode} onValueChange={v=>setMode(v as 'rename'|'copy')} disabled={pending}><Select.Trigger aria-label="命名整理の方法"/><Select.Content><Select.Item value="rename" disabled={protectedSource}>元ファイルの名前を変更</Select.Item><Select.Item value="copy">原本を残してコピー側を整理</Select.Item></Select.Content></Select.Root>
 {protectedSource&&<Text color="amber" size="1">確定済みの学習が参照しているため、原本を残して別フォルダに作成します。</Text>}
 <Select.Root value={numbering} onValueChange={v=>setNumbering(v as 'append'|'manual')} disabled={pending}><Select.Trigger aria-label="連番の付け方"/><Select.Content><Select.Item value="append">既存番号の続きから自動で付ける</Select.Item><Select.Item value="manual">開始番号を指定して振り直す</Select.Item></Select.Content></Select.Root>
 {numbering==='append'&&<Text size="1" color="gray">命名済みの画像はそのまま残し、追加画像へ続きの番号を付けます。既存の画像・TXTと同じ番号は使いません。</Text>}
 <Flex gap="3" wrap="wrap"><label style={{flex:1}}><Text as="div" size="1">共通の名前</Text><TextField.Root aria-label="共通の名前" value={title} disabled={pending} onChange={e=>setTitle(e.target.value)}/></label><label><Text as="div" size="1">開始番号</Text><NumericInput aria-label="開始番号" style={{width:100}} value={start} min={0} disabled={pending||numbering==='append'} onValueChange={setStart}/></label><label><Text as="div" size="1">番号の桁数</Text><NumericInput aria-label="番号の桁数" style={{width:100}} min={1} max={8} value={digits} disabled={pending} onValueChange={setDigits}/></label></Flex>
 <label><Text as="div" size="1">命名テンプレート</Text><TextField.Root aria-label="命名テンプレート" value={template} disabled={pending} onChange={e=>setTemplate(e.target.value)}/></label><Text size="1" color="gray">{'{title}：共通の名前　{index}：連番　{stem}：元の名前（拡張子を除く）'}</Text>
 {error&&<Callout.Root color="red"><Callout.Text>{error}</Callout.Text></Callout.Root>}
 {plan?.numbering==='append'&&plan.assigned_start!=null&&<Text size="2">追加画像は {plan.assigned_start} 番から命名します。</Text>}
 {plan&&<><Badge color="gray">{plan.changed} / {plan.rows.length}枚を変更</Badge><div className="naming-preview"><table><thead><tr><th>変更前</th><th>変更後</th><th>TXT</th></tr></thead><tbody>{plan.rows.map(r=><tr key={r.before}><td title={r.before}>{r.before}</td><td title={r.after}>{r.after}</td><td>{r.has_txt?'一緒に変更':'なし'}</td></tr>)}</tbody></table></div><Text size="1" color="gray">{mode==='copy'?`作成先：${plan.folder}`:'名前の変更後は「命名を戻す」から復元できます。画像の内容・TXT本文は変更しません。'}</Text></>}
 <Flex justify="end" gap="3" mt="3"><Button variant="soft" color="gray" disabled={pending} onClick={onClose}>キャンセル</Button><Button variant="soft" disabled={pending||!valid} onClick={()=>void preview()}>{pending?<Spinner/>:null}変更をプレビュー</Button><Button disabled={pending||!plan?.changed} onClick={()=>void apply()}>{mode==='copy'?'整理したコピーを作成':'名前を変更'}</Button></Flex>
 </Flex></Dialog.Content></Dialog.Root>;
}
