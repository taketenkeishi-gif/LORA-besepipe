import {useEffect,useRef,useState} from 'react';
import {draggable,dropTargetForElements,monitorForElements} from '@atlaskit/pragmatic-drag-and-drop/element/adapter';
import {dropTargetForExternal,monitorForExternal} from '@atlaskit/pragmatic-drag-and-drop/external/adapter';
import {containsFiles} from '@atlaskit/pragmatic-drag-and-drop/utils/contains-files';
import {combine} from '@atlaskit/pragmatic-drag-and-drop/combine';
import {setCustomNativeDragPreview} from '@atlaskit/pragmatic-drag-and-drop/utils/set-custom-native-drag-preview';
import {autoScrollForElements} from '@atlaskit/pragmatic-drag-and-drop-auto-scroll/element';
import {autoScrollForExternal} from '@atlaskit/pragmatic-drag-and-drop-auto-scroll/external';

export type ImportFile={file:File;name:string};
type Entry={name:string;isDirectory:boolean;file:(ok:(f:File)=>void,fail:(e:unknown)=>void)=>void;createReader:()=>{readEntries:(ok:(e:Entry[])=>void,fail:(e:unknown)=>void)=>void}};
type Hit={before:string|null;x:number;y:number;length:number;axis:'horizontal'|'vertical'};
type DragState={kind:'reorder'|'folder'|'external';ids:string[];target?:string;hit?:Hit;denied?:string}|null;
type Props={active?:boolean;folder:string;items:{relative:string;protected_source?:boolean}[];visible:string[];selected:string[];dirty:string[];view:string;busy:boolean;folders:string[];onSelect:(ids:string[])=>void;onReorder:(ids:string[],before:string|null)=>void;onMove:(ids:string[],folder:string)=>void;onImport:(files:ImportFile[],folder:string)=>void;onImportUrl?:(url:string,folder:string)=>void;onReadStart:()=>void;onReadError:(message:string)=>void;onError:(message:string)=>void};

/** Insert a block, never exchange cards. Unseen entries remain in the full order. */
export function insertBlock(order:string[],moving:string[],before:string|null){
  const selected=new Set(moving);const block=order.filter(id=>selected.has(id));
  if(before!==null&&selected.has(before))return order;
  const rest=order.filter(id=>!selected.has(id));const at=before===null?rest.length:rest.indexOf(before);
  if(at<0)return order;
  return [...rest.slice(0,at),...block,...rest.slice(at)];
}

/** Measure the actual rendered rows, including padding and both kinds of gaps. */
export function insertionAt(grid:HTMLElement,x:number,y:number,view:string):Hit|null{
  const nodes=Array.from(grid.querySelectorAll<HTMLElement>('[data-dnd-id]'));
  if(!nodes.length)return null;
  const cells=nodes.map(node=>({id:node.dataset.dndId!,r:node.getBoundingClientRect()}));
  const box=grid.getBoundingClientRect();
  const point=(before:string|null,px:number,py:number,length:number,axis:Hit['axis']):Hit=>({before,x:px-box.left+grid.scrollLeft,y:py-box.top+grid.scrollTop,length,axis});
  if(view==='list'){
    const next=cells.find(c=>y<c.r.top+c.r.height/2);
    const last=cells[cells.length-1];
    return point(next?.id??null,box.left+8,next?next.r.top-3:last.r.bottom+3,box.width-16,'horizontal');
  }
  const rows:typeof cells[]=[];
  for(const cell of cells){const row=rows[rows.length-1];if(!row||Math.abs(row[0].r.top-cell.r.top)>4)rows.push([cell]);else row.push(cell);}
  if(y<rows[0][0].r.top)return point(cells[0].id,box.left+8,rows[0][0].r.top-5,box.width-16,'horizontal');
  for(let n=0;n<rows.length;n++){
    const row=rows[n],bottom=Math.max(...row.map(c=>c.r.bottom)),nextRow=rows[n+1];
    if(y<=bottom){
      const next=row.find(c=>x<c.r.left+c.r.width/2);
      if(next){const i=row.indexOf(next),left=i?((row[i-1].r.right+next.r.left)/2):next.r.left-5;return point(next.id,left,row[0].r.top,bottom-row[0].r.top,'vertical');}
      const last=row[row.length-1];return point(nextRow?.[0].id??null,last.r.right+6,row[0].r.top,bottom-row[0].r.top,'vertical');
    }
    if(nextRow&&y<nextRow[0].r.top)return point(nextRow[0].id,box.left+8,(bottom+nextRow[0].r.top)/2,box.width-16,'horizontal');
  }
  const last=rows[rows.length-1];return point(null,box.left+8,Math.max(...last.map(c=>c.r.bottom))+5,box.width-16,'horizontal');
}

// Capture native entries synchronously before the drag data store closes.
async function filesFromItems(items:DataTransferItem[]):Promise<ImportFile[]>{
  const entries=items.filter(i=>i.kind==='file').map(i=>({entry:i.webkitGetAsEntry?.() as unknown as Entry|null,file:i.getAsFile()}));
  const result:ImportFile[]=[];let directories=0;
  async function walk(entry:Entry,prefix:string,depth:number):Promise<void>{
    if(depth>16||directories>256||result.length>=500)throw Error('一度に追加できるのは500ファイル・16階層までです');
    const name=prefix+entry.name;
    if(!entry.isDirectory){const file=await new Promise<File>((ok,fail)=>entry.file(ok,fail));result.push({file,name});return;}
    directories++;const reader=entry.createReader();
    for(;;){const batch=await new Promise<Entry[]>((ok,fail)=>reader.readEntries(ok,fail));if(!batch.length)break;for(const child of batch)await walk(child,name+'/',depth+1);}
  }
  for(const item of entries){if(item.entry)await walk(item.entry,'',0);else if(item.file)result.push({file:item.file,name:item.file.name});}
  if(result.length>500)throw Error('一度に追加できるのは500ファイルまでです');
  return result;
}

export function useDatasetDrag(props:Props){
  const rootRef=useRef<HTMLElement|null>(null),gridRef=useRef<HTMLDivElement|null>(null);
  const latest=useRef(props);latest.current=props;
  const scope=useRef(Symbol('dataset-drag'));
  const ignoreClick=useRef(false);
  const [state,setState]=useState<DragState>(null);
  const [dragIds,setDragIds]=useState<string[]>([]);
  const key=JSON.stringify([props.active,props.folder,props.visible,props.folders]);
  useEffect(()=>{
    if(props.active===false){setState(null);setDragIds([]);ignoreClick.current=false;return;}
    const root=rootRef.current,grid=gridRef.current;if(!root||!grid)return;
    const own=(source:{data:Record<string,unknown>})=>source.data.scope===scope.current;
    const idsOf=(source:{data:Record<string,unknown>})=>source.data.ids as string[];
    const cleanups:(()=>void)[]=[];
    for(const element of grid.querySelectorAll<HTMLElement>('[data-dnd-id]')){
      cleanups.push(draggable({element,canDrag:()=>!latest.current.busy,
        getInitialData:()=>{const p=latest.current,id=element.dataset.dndId!;return {scope:scope.current,ids:p.selected.includes(id)?p.items.filter(i=>p.selected.includes(i.relative)).map(i=>i.relative):[id]};},
        onGenerateDragPreview:({source,nativeSetDragImage})=>{
          setCustomNativeDragPreview({nativeSetDragImage,render:({container})=>{const preview=document.createElement('div');preview.textContent=`${idsOf(source).length}枚を移動`;preview.style.cssText='padding:10px 16px;background:#292d37;color:white;border:1px solid #969ac0;border-radius:8px;font:14px system-ui;box-shadow:0 8px 24px #0008';container.append(preview);}});
        },
        onDragStart:({source})=>{ignoreClick.current=true;const ids=idsOf(source);setDragIds(ids);latest.current.onSelect(ids);},
      }));
    }
    cleanups.push(dropTargetForElements({element:grid,canDrop:({source})=>own(source)&&!latest.current.busy,
      getData:({input})=>({kind:'order',hit:insertionAt(grid,input.clientX,input.clientY,latest.current.view)}),
      onDrag:({self,source})=>setState({kind:'reorder',ids:idsOf(source),hit:self.data.hit as Hit}),
      onDragEnter:({self,source})=>setState({kind:'reorder',ids:idsOf(source),hit:self.data.hit as Hit}),
      onDragLeave:()=>setState(null),
      onDrop:({self,source})=>{const hit=self.data.hit as Hit|null;if(hit)latest.current.onReorder(idsOf(source),hit.before);},
    }));
    const denial=(ids:string[],target:string)=>{
      const p=latest.current;
      if(p.busy)return '処理中です';
      if(target===p.folder)return '現在のフォルダです';
      if(ids.some(id=>p.dirty.includes(id)))return '先にキャプションを保存してください';
      if(p.items.some(i=>ids.includes(i.relative)&&i.protected_source))return '参照原本です。学習用に複製してください';
      return undefined;
    };
    const folderElements=Array.from(root.querySelectorAll<HTMLElement>('[data-drop-folder]'));
    for(const element of folderElements){
      const target=element.dataset.dropFolder!;
      cleanups.push(dropTargetForElements({element,canDrop:({source})=>own(source)&&!denial(idsOf(source),target),getData:()=>({kind:'folder',target}),
        getDropEffect:()=> 'move',
        onDragEnter:({source})=>setState({kind:'folder',ids:idsOf(source),target,denied:denial(idsOf(source),target)}),
        onDrag:({source})=>setState({kind:'folder',ids:idsOf(source),target,denied:denial(idsOf(source),target)}),
        onDragLeave:()=>setState(null),
        onDrop:({source})=>{const reason=denial(idsOf(source),target);if(reason)latest.current.onError(reason);else latest.current.onMove(idsOf(source),target);},
      }));
    }
    for(const element of [grid,...folderElements]){
      const target=element===grid?latest.current.folder:element.dataset.dropFolder!;
      cleanups.push(dropTargetForExternal({element,canDrop:args=>containsFiles(args)&&!latest.current.busy,getData:()=>({kind:'external',target}),getDropEffect:()=> 'copy',
        onDragEnter:()=>setState({kind:'external',ids:[],target}),onDragLeave:()=>setState(null),
        onDrop:({source})=>{latest.current.onReadStart();void filesFromItems(source.items).then(files=>{if(!files.length)throw Error('画像またはTXTファイルをドロップしてください');latest.current.onImport(files,target);}).catch(e=>latest.current.onReadError(e.message));},
      }));
    }
    cleanups.push(monitorForElements({canMonitor:({source})=>own(source),onDrag:({source,location})=>{const el=document.elementFromPoint(location.current.input.clientX,location.current.input.clientY)?.closest<HTMLElement>('[data-drop-folder]');if(el){const target=el.dataset.dropFolder!,reason=denial(idsOf(source),target);if(reason)setState({kind:'folder',ids:idsOf(source),target,denied:reason});}},onDrop:()=>{setState(null);setDragIds([]);}}));
    cleanups.push(monitorForExternal({canMonitor:containsFiles,onDrop:()=>setState(null)}));
    cleanups.push(autoScrollForElements({element:grid,canScroll:({source})=>own(source)&&!latest.current.busy}));
    cleanups.push(autoScrollForExternal({element:grid,canScroll:args=>containsFiles(args)&&!latest.current.busy}));
    const guard=(e:DragEvent)=>{
      const types=Array.from(e.dataTransfer?.types||[]);const files=types.includes('Files');const uri=types.includes('text/uri-list')||types.includes('text/html');
      if(!files&&!uri)return;
      // a drag inside the app (images or characters of the video workspace) is not an external drop: an <img> drag also carries
      // uri-list/html, and refusing it here made every drop target of the app reject the drag (dropEffect none)
      if(types.some(t=>t.startsWith('application/x-lora-')))return;
      if(!files&&(e.target as HTMLElement)?.closest('textarea,input,[contenteditable=true]'))return;
      e.preventDefault();
      const zone=(e.target as HTMLElement)?.closest<HTMLElement>('[data-dnd-zone]');
      const allowed=!!zone&&root.contains(zone)&&!latest.current.busy;
      if(!files&&allowed&&latest.current.onImportUrl){
        if(e.dataTransfer)e.dataTransfer.dropEffect='copy';
        const target=zone.dataset.dropFolder??latest.current.folder;
        if(e.type==='dragover'){setState({kind:'external',ids:[],target});return;}
        setState(null);
        try{
          const html=e.dataTransfer?.getData('text/html')||'';
          const images=html?Array.from(new DOMParser().parseFromString(html,'text/html').querySelectorAll('img')).map(i=>i.getAttribute('src')).filter(Boolean):[];
          const urls=(e.dataTransfer?.getData('text/uri-list')||'').split(/\r?\n/).filter(s=>s.trim()&&!s.startsWith('#'));
          if(images.length>1||(!images.length&&urls.length>1))throw Error('Web画像は1枚ずつドロップしてください');
          const value=images[0]||urls[0]||'';
          const url=new URL(value,urls[0]);
          if(!['http:','https:'].includes(url.protocol))throw Error('公開画像のhttp/https URLをドロップしてください');
          latest.current.onImportUrl(url.href,target);
        }catch(err){latest.current.onError(err instanceof Error?err.message:String(err));}
        return;
      }
      if((!allowed||!files)&&e.dataTransfer)e.dataTransfer.dropEffect='none';
      if(e.type==='drop'&&(!allowed||!files)){setState(null);latest.current.onError(!allowed?'一覧またはフォルダへドロップしてください':'Web画像の取り込みはデスクトップアプリで利用できます');}
    };
    const leave=(e:DragEvent)=>{if(!e.relatedTarget)setState(null);};
    window.addEventListener('dragleave',leave);
    window.addEventListener('dragover',guard);window.addEventListener('drop',guard);
    return combine(...cleanups,()=>{window.removeEventListener('dragleave',leave);window.removeEventListener('dragover',guard);window.removeEventListener('drop',guard);});
  },[key]);
  return {rootRef,gridRef,state,dragIds,ignoreClick};
}
