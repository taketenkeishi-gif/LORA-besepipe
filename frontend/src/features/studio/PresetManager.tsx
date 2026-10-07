import {useEffect,useMemo,useState} from 'react';
import {Badge,Button,Callout,Dialog,Flex,Select,Text,TextField} from '@radix-ui/themes';
import {Copy,Download,FolderOpen,Pencil,Save,Trash2,Upload} from 'lucide-react';
import {categoryName,summarize,type Difference,type Entry} from './presetModel';

export type ManagerActions={
 apply:(e:Entry)=>void;saveNew:()=>void;importFile:()=>void;exportCurrent:()=>void;
 overwrite:(e:Entry)=>void;rename:(e:Entry)=>void;duplicate:(e:Entry)=>void;remove:(e:Entry)=>void;exportEntry:(e:Entry)=>void;
 reveal:(e?:Entry)=>void;
};
type Props={open:boolean;onOpenChange:(v:boolean)=>void;items:Entry[];folder:string;appliedId:string;differences:Difference[];busy:boolean;notice:string;error:string;actions:ManagerActions};
const when=(t:number)=>new Date(t*1000).toLocaleString('ja-JP');

export default function PresetManager({open,onOpenChange,items,folder,appliedId,differences,busy,notice,error,actions}:Props){
 const [query,setQuery]=useState(''),[category,setCategory]=useState('all'),[selectedId,setSelectedId]=useState('');
 const shown=useMemo(()=>{const q=query.trim().toLowerCase();return items.filter(e=>(category==='all'||e.payload?.category===category)&&(!q||e.name.toLowerCase().includes(q)||String(e.payload?.config.model_family||'').includes(q)||e.id.toLowerCase().includes(q)));},[items,query,category]);
 // Keep a valid selection: prefer the applied preset, otherwise the first visible one.
 useEffect(()=>{if(open&&!items.some(e=>e.id===selectedId))setSelectedId(items.find(e=>e.id===appliedId)?.id||items[0]?.id||'');},[open,items,appliedId,selectedId]);
 const selected=items.find(e=>e.id===selectedId);
 const isApplied=!!selected&&selected.id===appliedId;
 return <Dialog.Root open={open} onOpenChange={onOpenChange}><Dialog.Content maxWidth="940px" style={{maxHeight:'88vh',overflow:'auto'}}>
  <Dialog.Title>学習プリセットの管理</Dialog.Title>
  <Dialog.Description size="2">学習設定（モデル・Rank・学習率・解像度・詳細設定など）をJSONファイルとして保存・管理します。画像選択・保存名・トリガーワードはプロジェクトごとに保持され、プリセットには含まれません。</Dialog.Description>
  <Flex align="center" gap="2" mt="3" wrap="wrap"><Text size="1" color="gray" style={{overflowWrap:'anywhere',flex:1,minWidth:200}} title={folder}>保存先：{folder||'…'}</Text><Button size="1" variant="soft" onClick={()=>actions.reveal()}><FolderOpen size={13}/>保存フォルダをエクスプローラーで開く</Button></Flex>
  <Flex gap="2" mt="3" wrap="wrap" align="center">
   <Button size="2" disabled={busy} onClick={actions.saveNew}><Save size={14}/>現在の設定を新規保存</Button>
   <Button size="2" variant="soft" disabled={busy} onClick={actions.importFile}><Upload size={14}/>JSONから読み込む</Button>
   <Button size="2" variant="soft" color="gray" disabled={busy} onClick={actions.exportCurrent}><Download size={14}/>現在の設定をJSONに書き出す</Button>
  </Flex>
  {notice&&<Text as="p" size="1" color="green" role="status" mt="2">{notice}</Text>}
  {error&&<Callout.Root color="red" size="1" mt="2"><Callout.Text style={{whiteSpace:'pre-line'}}>{error}</Callout.Text></Callout.Root>}
  <div style={{display:'grid',gridTemplateColumns:'minmax(240px,1fr) minmax(300px,1.3fr)',gap:16,marginTop:14}}>
   <div style={{minWidth:0}}>
    <Flex gap="2" mb="2"><TextField.Root style={{flex:1}} aria-label="プリセットを検索" placeholder="名前・モデルで検索" value={query} onChange={e=>setQuery(e.target.value)}/><Select.Root value={category} onValueChange={setCategory}><Select.Trigger aria-label="分類で絞り込み"/><Select.Content><Select.Item value="all">すべて</Select.Item><Select.Item value="character">キャラクター</Select.Item><Select.Item value="style">スタイル</Select.Item></Select.Content></Select.Root></Flex>
    <div role="listbox" aria-label="プリセット一覧" style={{display:'flex',flexDirection:'column',gap:4,maxHeight:380,overflow:'auto'}}>
     {shown.length===0&&<Text size="2" color="gray">{items.length?'条件に合うプリセットがありません':'プリセットがまだありません。「現在の設定を新規保存」で作成できます。'}</Text>}
     {shown.map(e=><button key={e.id} type="button" role="option" aria-selected={e.id===selectedId} onClick={()=>setSelectedId(e.id)} data-preset-id={e.id} style={{textAlign:'left',padding:'8px 10px',borderRadius:6,border:'1px solid '+(e.id===selectedId?'var(--accent-8)':'var(--gray-5)'),background:e.id===selectedId?'var(--accent-3)':'var(--gray-2)',color:'inherit',cursor:'pointer'}}>
      <Flex justify="between" align="center" gap="2"><Text size="2" weight="medium" style={{overflowWrap:'anywhere'}}>{e.name}</Text>{e.id===appliedId&&<Badge color="green" size="1">適用中</Badge>}</Flex>
      {e.payload?<Text size="1" color="gray">{categoryName(e.payload.category)} · {e.payload.config.model_family} · rank {e.payload.config.rank} · {e.payload.config.epochs}ep</Text>:<Text size="1" color="red">読み込めません</Text>}
     </button>)}
    </div>
   </div>
   <div style={{minWidth:0}} aria-label="プリセットの詳細">
    {!selected?<Text size="2" color="gray">左の一覧からプリセットを選んでください。</Text>:<>
     <Flex align="center" gap="2" wrap="wrap"><Text size="4" weight="bold" style={{overflowWrap:'anywhere'}}>{selected.name}</Text>{isApplied&&<Badge color={differences.length?'amber':'green'}>{differences.length?`適用中 · ${differences.length}項目を変更済み`:'適用中 · 変更なし'}</Badge>}</Flex>
     <Text as="p" size="1" color="gray" style={{overflowWrap:'anywhere'}}>{selected.id} · 更新 {when(selected.modified)}</Text>
     {selected.error&&<Callout.Root color="red" size="1" my="2"><Callout.Text>このファイルは学習プリセットとして読み込めません：{selected.error}。エクスプローラーで確認するか、削除できます。</Callout.Text></Callout.Root>}
     {selected.payload&&<table style={{borderCollapse:'collapse',width:'100%',marginTop:8}}><tbody>{summarize(selected.payload).map(([k,v])=><tr key={k}><th scope="row" style={{textAlign:'left',fontWeight:400,color:'var(--gray-11)',fontSize:12,padding:'3px 10px 3px 0',whiteSpace:'nowrap',verticalAlign:'top'}}>{k}</th><td style={{fontSize:13,padding:'3px 0',overflowWrap:'anywhere'}}>{v}</td></tr>)}</tbody></table>}
     {selected.payload?.recommendation?.rationale&&<Text as="p" size="1" color="gray" mt="2" style={{overflowWrap:'anywhere'}}>{selected.payload.recommendation.rationale}</Text>}
     {isApplied&&differences.length>0&&<div style={{marginTop:10,padding:8,border:'1px solid var(--amber-6)',borderRadius:6,background:'var(--amber-2)'}}><Text size="1" weight="medium">プリセット適用後に変えた項目（現在 → プリセット）</Text>{differences.map(d=><Text as="p" size="1" key={d.key}>{d.label}：{d.current} → {d.preset}</Text>)}</div>}
     <Flex gap="2" mt="3" wrap="wrap">
      <Button disabled={busy||!selected.payload} onClick={()=>actions.apply(selected)}>{isApplied&&differences.length?'もう一度適用（元に戻す）':'この設定を適用'}</Button>
      <Button variant="soft" disabled={busy||!selected.payload} onClick={()=>actions.overwrite(selected)} title="現在の設定でこのプリセットを置き換えます">現在の設定で上書き…</Button>
     </Flex>
     <Flex gap="2" mt="2" wrap="wrap">
      <Button size="1" variant="soft" color="gray" disabled={busy||!selected.payload} onClick={()=>actions.rename(selected)}><Pencil size={12}/>名前を変更</Button>
      <Button size="1" variant="soft" color="gray" disabled={busy||!selected.payload} onClick={()=>actions.duplicate(selected)}><Copy size={12}/>複製</Button>
      <Button size="1" variant="soft" color="gray" onClick={()=>actions.reveal(selected)}><FolderOpen size={12}/>このファイルの場所を開く</Button>
      <Button size="1" variant="soft" color="gray" disabled={!selected.payload} onClick={()=>actions.exportEntry(selected)}><Download size={12}/>このプリセットをJSONに書き出す</Button>
      <Button size="1" variant="soft" color="red" disabled={busy} onClick={()=>actions.remove(selected)}><Trash2 size={12}/>削除…</Button>
     </Flex>
    </>}
   </div>
  </div>
  <Flex justify="end" mt="4"><Dialog.Close><Button variant="soft" color="gray">閉じる</Button></Dialog.Close></Flex>
 </Dialog.Content></Dialog.Root>;
}
