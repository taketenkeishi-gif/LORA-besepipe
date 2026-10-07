import NumericInput from './NumericInput';
import {useEffect,useRef,useState} from 'react';
import {Button,Dialog,Flex,Text,TextField,Select,Checkbox,Spinner,Callout} from '@radix-ui/themes';
import {desktop,desktopError,type NormalizePreview,type NormalizeSettings} from './desktopBridge';
type Props={projectId:number;relatives:string[];onClose:()=>void;onCreated:(folder:string,relatives:string[])=>void};
export default function NormalizeDialog({projectId,relatives,onClose,onCreated}:Props){
 const [mode,setMode]=useState<'pad'|'longest'>('pad'),[width,setWidth]=useState(1024),[height,setHeight]=useState(1024),[upscale,setUpscale]=useState(false),[pending,setPending]=useState(false),[error,setError]=useState(''),[preview,setPreview]=useState<NormalizePreview|null>(null);
 const config:NormalizeSettings={project_id:projectId,relatives,mode,width,height,allow_upscale:upscale};
 const signature=JSON.stringify(config),current=useRef(signature);current.current=signature;
 useEffect(()=>{setPreview(null);},[signature]);
 const valid=width>=64&&width<=4096&&height>=64&&height<=4096&&Number.isInteger(width)&&Number.isInteger(height);
 async function inspect(){const version=signature;setPending(true);setError('');try{const result=await desktop()!.previewNormalize(config);if(current.current===version)setPreview(result);}catch(e){setError(desktopError(e));}finally{setPending(false);}}
 async function apply(){if(!preview)return;setPending(true);setError('');try{const result=await desktop()!.applyNormalize({...config,expected:preview.items});onCreated(result.folder,result.relatives);onClose();}catch(e){setError(desktopError(e));setPreview(null);}finally{setPending(false);}}
 return <Dialog.Root open onOpenChange={v=>{if(!v&&!pending)onClose();}}><Dialog.Content maxWidth="700px"><Dialog.Title>画像サイズを揃える</Dialog.Title><Dialog.Description size="2" mb="4">選択した{relatives.length}枚を別フォルダへ作成します。原本とTXTは残します。</Dialog.Description>
  <Flex direction="column" gap="3"><Select.Root value={mode} onValueChange={v=>setMode(v as 'pad'|'longest')} disabled={pending}><Select.Trigger aria-label="サイズの揃え方"/><Select.Content><Select.Item value="pad">縦横を揃える・余白で調整</Select.Item><Select.Item value="longest">長辺を揃える・縦横比を維持</Select.Item></Select.Content></Select.Root>
   <Flex gap="3"><label><Text as="div" size="1">{mode==='pad'?'横幅':'長辺'}（px）</Text><NumericInput aria-label={mode==='pad'?'横幅px':'長辺px'} min={64} max={4096} value={width} disabled={pending} onValueChange={setWidth}/></label>{mode==='pad'&&<label><Text as="div" size="1">高さ（px）</Text><NumericInput aria-label="高さpx" min={64} max={4096} value={height} disabled={pending} onValueChange={setHeight}/></label>}<Flex align="end" gap="1">{[512,768,1024].map(v=><Button size="1" variant="soft" key={v} disabled={pending} onClick={()=>{setWidth(v);setHeight(v);}}>{v}</Button>)}</Flex></Flex>
   <Text as="label" size="2"><Flex gap="2" align="center"><Checkbox checked={upscale} disabled={pending} onCheckedChange={v=>setUpscale(v===true)}/>小さい画像も拡大する</Flex></Text><Text size="1" color="gray">画像を切り抜いたり引き伸ばしたりしません。拡大は通常のリサイズで、高画質化モデルは使いません。</Text>
   {error&&<Callout.Root color="red"><Callout.Text>{error}</Callout.Text></Callout.Root>}
   {preview&&<><div className="normalize-comparison"><figure><img src={preview.before} alt="処理前のプレビュー"/><figcaption>処理前 · {preview.items[0].source_size.join(' × ')}</figcaption></figure><figure><img src={preview.after} alt="サイズ調整後のプレビュー"/><figcaption>処理後 · {preview.items[0].output_size.join(' × ')}</figcaption></figure></div><Text size="1" color="gray">先頭画像のプレビューです。{preview.count}枚、{new Set(preview.items.map(i=>i.source_size.join('x'))).size}種類のサイズ → {new Set(preview.items.map(i=>i.output_size.join('x'))).size}種類。{preview.items.filter(i=>i.scale>1).length}枚が拡大対象。</Text></>}
   <Flex justify="end" gap="3" mt="2"><Button variant="soft" color="gray" disabled={pending} onClick={onClose}>キャンセル</Button><Button variant="soft" disabled={!valid||pending} onClick={()=>void inspect()}>{pending?<Spinner/>:null}プレビュー</Button><Button disabled={!preview||pending} onClick={()=>void apply()}>作成して開く</Button></Flex>
  </Flex>
 </Dialog.Content></Dialog.Root>;
}
