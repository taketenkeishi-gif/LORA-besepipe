import {useEffect,useState} from 'react';
import {Button,Dialog,Flex,Text,TextField,Callout,Spinner,Badge} from '@radix-ui/themes';
import {desktop,desktopError} from './desktopBridge';

type Catalog={root:string;url:string;models:{name:string}[];devices:{name:string}[]};
export default function ComfyConnectionDialog({onClose,onBind}:{onClose:()=>void;onBind:()=>void}){
 const [root,setRoot]=useState(''),[url,setUrl]=useState(''),[catalog,setCatalog]=useState<Catalog|null>(null);
 const [instances,setInstances]=useState<{url:string;root:string;device:string}[]>([]);
 const [loading,setLoading]=useState(true),[busy,setBusy]=useState(false),[error,setError]=useState('');
 useEffect(()=>{let live=true;void (async()=>{try{const found=await desktop()!.comfy('discover');if(!live)return;setRoot(found.root);setUrl(found.url);setInstances(found.instances);const result=await desktop()!.comfy('inspect');if(live)setCatalog(result);}catch(e){if(live)setError(desktopError(e));}finally{if(live)setLoading(false);}})();return()=>{live=false;};},[]);
 async function connect(){setBusy(true);setError('');try{setCatalog(await desktop()!.comfy('bind',{root,url}));onBind();}catch(e){setCatalog(null);setError(desktopError(e));}finally{setBusy(false);}}
 return <Dialog.Root open onOpenChange={v=>{if(!v&&!busy)onClose();}}><Dialog.Content maxWidth="680px"><Dialog.Title>ComfyUI接続</Dialog.Title><Dialog.Description size="2" mb="4">学習中の自動プレビューに使うComfyUIと、共有するモデルの保存場所を指定します。</Dialog.Description><Flex direction="column" gap="3">
  {loading?<Flex gap="2"><Spinner/><Text>接続先を確認中…</Text></Flex>:<>
   {!!instances.length&&<Flex gap="2" wrap="wrap">{instances.map(i=><Button key={i.url} size="1" variant="soft" disabled={busy} onClick={()=>{setRoot(i.root);setUrl(i.url);setCatalog(null);}}>起動中 {i.url}</Button>)}</Flex>}
   <label><Text as="div" size="1">ComfyUI本体フォルダ</Text><Flex gap="2"><TextField.Root style={{flex:1}} aria-label="ComfyUI本体フォルダ" value={root} disabled={busy} onChange={e=>{setRoot(e.target.value);setCatalog(null);}}/><Button variant="soft" disabled={busy} onClick={()=>{void desktop()!.chooseComfyRoot().then(path=>{if(path){setRoot(path);setCatalog(null);}}).catch(e=>setError(desktopError(e)));}}>選択</Button></Flex></label>
   <label><Text as="div" size="1">接続先URL</Text><TextField.Root aria-label="ComfyUI接続先URL" value={url} disabled={busy} onChange={e=>{setUrl(e.target.value);setCatalog(null);}}/></label>
   <Button disabled={busy||!root.trim()||!url.trim()} onClick={()=>void connect()}>{busy&&<Spinner/>}接続してモデル一覧を読み込む</Button>
   {catalog&&<><Flex gap="2" align="center"><Badge color="green">接続済み</Badge><Text size="2">{catalog.models.length}モデルを参照</Text></Flex>{catalog.devices.map((d,i)=><Text key={i} size="1" color="gray">{d.name}</Text>)}<Text size="1" color="gray">モデルはこの保存場所から直接参照します。学習画面でモデルと自動プレビューの条件を選択できます。</Text></>}
  </>}
  {error&&<Callout.Root color="red"><Callout.Text>{error}</Callout.Text></Callout.Root>}
  <Flex justify="end"><Button variant="soft" color="gray" disabled={busy} onClick={onClose}>閉じる</Button></Flex>
 </Flex></Dialog.Content></Dialog.Root>;
}
