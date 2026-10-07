import {desktop} from './desktopBridge';
type Catalog={root:string;url:string;models:{name:string}[];devices:{name:string}[]};
let pending:Promise<Catalog>|null=null;
let expires=0;
export function autoComfyConnection(force=false):Promise<Catalog>{
 if(!force&&pending&&Date.now()<expires)return pending;
 const bridge=desktop();
 if(!bridge)return Promise.reject(new Error('デスクトップアプリで接続設定を確認してください'));
 expires=Date.now()+30000;
 pending=(async()=>{
  try{return await bridge.comfy('inspect') as Catalog;}catch{/* Re-resolve moved paths or a changed local port. */}
  const found=await bridge.comfy('discover');
  const candidates=found.instances as {url:string;root:string;device:string}[];
  // Do not silently switch away from the configured endpoint or the designated GPU.
  const same=candidates.filter(i=>i.url===found.url);
  const gpu=candidates.filter(i=>/RTX\s*3090\s*Ti/i.test(i.device));
  const selected=same.length===1?same[0]:gpu.length===1?gpu[0]:null;
  if(!selected)throw new Error(candidates.length?'接続先を一つに特定できませんでした。手動で接続先を指定してください。':'起動中のComfyUIを確認できませんでした。起動後に再検出できます。');
  return await bridge.comfy('bind',{root:selected.root,url:selected.url}) as Catalog;
 })();
 return pending;
}
