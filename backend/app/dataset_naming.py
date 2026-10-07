"""Previewed batch naming for image/TXT pairs. Never overwrite another file."""
from __future__ import annotations
import json,re,shutil,uuid
from pathlib import Path
from filelock import FileLock
from . import dataset_desktop as desktop
from .db import get_conn
from .desktop_jobs import write_state
from .routers.dataset_files import root_for,image_path,sidecar,protected_paths,EXTENSIONS,folder_order,write_order,ORDER_FILE

def _name(stem):
    if not stem or stem!=stem.strip() or stem.startswith('.') or stem.endswith('.') or re.search(r'[<>:"/\\|?*\x00-\x1f{}]',stem):raise ValueError('名前に使えない文字、空白だけの名前、未対応の変数が含まれています')
    if stem.split('.')[0].upper() in {'CON','PRN','AUX','NUL',*[f'COM{i}' for i in range(1,10)],*[f'LPT{i}' for i in range(1,10)]}:raise ValueError('Windowsで予約された名前は使えません')
    if len(stem)>150:raise ValueError('名前が長すぎます（150文字まで）')
    return stem

def _journal(project_id):
    directory=desktop.ROOT/'.runtime/naming'/str(project_id);directory.mkdir(parents=True,exist_ok=True);return directory

def preview(payload):
    project_id=int(payload['project_id']);root=root_for(project_id);relatives=list(dict.fromkeys(payload.get('relatives') or []))
    if not 1<=len(relatives)<=500:raise ValueError('1〜500枚の画像を選択してください')
    mode=payload.get('mode','rename')
    if mode not in ('rename','copy'):raise ValueError('命名方法が不正です')
    template=str(payload.get('template') or '{title}_{index}');title=str(payload.get('title') or 'image');start=int(payload.get('start',1));digits=int(payload.get('digits',4))
    if not (0<=start<=99999999 and 1<=digits<=8):raise ValueError('開始番号または桁数の範囲が不正です')
    copy_folder=str(payload.get('copy_folder') or 'organized-'+uuid.uuid4().hex[:8])
    if not re.fullmatch(r'organized-[0-9a-f]{8}',copy_folder):raise ValueError('整理先フォルダが不正です')
    if mode=='copy' and (root/copy_folder).exists():raise ValueError('整理先フォルダが既にあります。プレビューを更新してください')
    conn=get_conn()
    try:protected=protected_paths(conn)
    finally:conn.close()
    numbering=payload.get('numbering','append')
    if numbering not in ('append','manual'):raise ValueError('連番の付け方が不正です')
    pattern=None;next_number={};assigned=[]
    if numbering=='append':
        if template.count('{index}')!=1:raise ValueError('自動連番では{index}を1か所含めてください')
        expression=re.escape(template).replace(re.escape('{title}'),re.escape(title)).replace(re.escape('{stem}'),'.+?').replace(re.escape('{index}'),r'(?P<index>\d+)')
        pattern=re.compile('^'+expression+'$',re.IGNORECASE)
        for relative in relatives:
            source=image_path(root,relative);folder=root/copy_folder if mode=='copy' else source.parent
            if folder not in next_number:
                used=[int(m.group('index')) for candidate in folder.iterdir() if candidate.is_file() and candidate.suffix.lower() in EXTENSIONS|{'.txt'} and (m:=pattern.fullmatch(candidate.stem))] if folder.exists() else []
                next_number[folder]=max(used,default=0)+1
    rows=[];source_keys=set();dest_keys=set();stem_keys=set()
    for index,relative in enumerate(relatives,start):
        path=image_path(root,relative);caption,revision=sidecar(path)
        if mode=='rename' and str(path).casefold() in protected:raise ValueError('確定済みの学習が参照しています。「コピー側を整理」を選んで原本を残してください')
        if mode=='rename' and sum(p.is_file() and p.stem.casefold()==path.stem.casefold() and p.suffix.lower() in EXTENSIONS for p in path.parent.iterdir())>1:raise ValueError('同じTXTを共有する画像があります。「コピー側を整理」を使用してください')
        folder=root/copy_folder if mode=='copy' else path.parent
        existing=pattern.fullmatch(path.stem) if pattern and mode=='rename' else None
        if existing:
            stem=path.stem
        else:
            if numbering=='append':index=next_number[folder];next_number[folder]+=1;assigned.append(index)
            if index>99999999:raise ValueError('連番の上限を超えています')
            stem=_name(template.replace('{title}',title).replace('{index}',str(index).zfill(digits)).replace('{stem}',path.stem))
        target=(root/copy_folder if mode=='copy' else path.parent)/(stem+path.suffix)
        key=str(target).casefold();txt_key=str(target.with_suffix('.txt')).casefold()
        if key in dest_keys or txt_key in stem_keys:raise ValueError('変更後の名前が重複しています。{index}を含めてください')
        dest_keys.add(key);stem_keys.add(txt_key);source_keys.add(str(path).casefold())
        if path.with_suffix('.txt').exists():source_keys.add(str(path.with_suffix('.txt')).casefold())
        rows.append({'before':path.relative_to(root).as_posix(),'after':target.relative_to(root).as_posix(),'sha256':desktop.digest(path),'caption_revision':revision,'has_txt':path.with_suffix('.txt').exists(),'changed':mode=='copy' or path.name!=target.name})
    for row in rows:
        target=root/row['after']
        for path in (target,target.with_suffix('.txt')):
            if path.exists() and str(path).casefold() not in source_keys:raise ValueError(f'既存ファイルと重複します：{path.name}')
    return {'project_id':project_id,'mode':mode,'numbering':numbering,'assigned_start':min(assigned) if assigned else None,'template':template,'title':title,'start':start,'digits':digits,'copy_folder':copy_folder,'rows':rows,'changed':sum(r['changed'] for r in rows),'folder':copy_folder if mode=='copy' else Path(rows[0]['before']).parent.as_posix().replace('.','',1) if Path(rows[0]['before']).parent==Path('.') else Path(rows[0]['before']).parent.as_posix()}

def _execute(project_id,rows,token):
    root=root_for(project_id);conn=get_conn();stage=root/('.rename-stage-'+token);moves=[];orders={};metadata=[]
    stage.mkdir(exist_ok=False)
    write_state(stage/'plan.json',json.dumps({'rows':rows},ensure_ascii=False,indent=2))
    try:
        conn.execute('BEGIN IMMEDIATE');protected=protected_paths(conn)
        for row in rows:
            src=image_path(root,row['before']);dst=root/row['after']
            if not dst.resolve().is_relative_to(root):raise ValueError('データセット外へ変更できません')
            if str(src).casefold() in protected:raise ValueError('確定済み学習の参照が追加されました。コピー側を整理してください')
            if desktop.digest(src)!=row['sha256'] or sidecar(src)[1]!=row['caption_revision']:raise ValueError('画像またはTXTが変わりました。プレビューを更新してください')
            if src.parent not in orders:
                order_path=src.parent/ORDER_FILE;names=[p.relative_to(root).as_posix() for p in src.parent.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS]
                order,_=folder_order(root,src.parent,names);orders[src.parent]=(order,order_path.read_bytes() if order_path.exists() else None)
        # Move all sources out of the way first, permitting swaps and case-only changes.
        for index,row in enumerate(rows):
            if not row['changed']:continue
            src=root/row['before'];dst=root/row['after'];temp=stage/f'{index}{src.suffix}';src.rename(temp);moves.append((src,temp));metadata.append((src,temp,dst))
            if row['has_txt']:
                original=src.with_suffix('.txt');staged=temp.with_suffix('.txt');original.rename(staged);moves.append((original,staged))
        for src,temp,dst in metadata:
            temp.rename(dst);moves.append((temp,dst))
            if temp.with_suffix('.txt').exists():temp.with_suffix('.txt').rename(dst.with_suffix('.txt'));moves.append((temp.with_suffix('.txt'),dst.with_suffix('.txt')))
        # Use temporary identities in the database too, so name swaps don't collide.
        for src,temp,dst in metadata:
            conn.execute("UPDATE basepipe_assets SET file_path=?,asset_key=?,updated_at=CURRENT_TIMESTAMP WHERE project_id=? AND file_path=? AND origin_kind='dataset_file'",(str(temp),token+'-'+temp.name,project_id,str(src)))
            conn.execute('UPDATE dataset_items SET file_path=? WHERE project_id=? AND file_path=?',(str(temp),project_id,str(src)))
        for src,temp,dst in metadata:
            conn.execute("UPDATE basepipe_assets SET file_path=?,asset_key=?,updated_at=CURRENT_TIMESTAMP WHERE project_id=? AND file_path=? AND origin_kind='dataset_file'",(str(dst),dst.name,project_id,str(temp)))
            conn.execute('UPDATE dataset_items SET file_path=? WHERE project_id=? AND file_path=?',(str(dst),project_id,str(temp)))
        mapping={row['before']:row['after'] for row in rows}
        for folder,(order,_) in orders.items():write_order(root,folder,[mapping.get(value,value) for value in order])
        conn.commit()
    except Exception:
        conn.rollback();rollback_errors=[]
        for original,current in reversed(moves):
            try:
                if current.exists():current.rename(original)
            except OSError as error:rollback_errors.append(str(error))
        for folder,(_,raw) in orders.items():
            try:
                order_path=folder/ORDER_FILE
                if raw is None:order_path.unlink(missing_ok=True)
                else:order_path.write_bytes(raw)
            except OSError as error:rollback_errors.append(str(error))
        if rollback_errors:raise RuntimeError(f'復元が必要です。退避先：{stage}。{rollback_errors}')
        raise
    finally:
        conn.close()
        # Remove only our empty staging metadata; preserve files if rollback was incomplete.
        if stage.exists() and all(p.name=='plan.json' for p in stage.iterdir()):(stage/'plan.json').unlink(missing_ok=True);stage.rmdir()

def apply(payload):
    project_id=int(payload['project_id']);root=root_for(project_id);expected=payload.get('expected') or {};token=uuid.uuid4().hex
    with FileLock(str(_journal(project_id)/'operation.lock'),timeout=0):
        plan=preview({**payload,'copy_folder':expected.get('copy_folder')});rows=plan['rows']
        if rows!=expected.get('rows') or plan['mode']!=expected.get('mode'):raise ValueError('対象や設定が変わりました。プレビューを更新してください')
        if not plan['changed']:raise ValueError('変更される名前がありません')
        write_state(_journal(project_id)/(token+'.json'),json.dumps({**plan,'token':token,'status':'prepared'},ensure_ascii=False,indent=2))
        if plan['mode']=='copy':
            stage=root/('.rename-copy-'+token);stage.mkdir()
            try:
                for row in rows:
                    src=root/row['before'];out=stage/Path(row['after']).name
                    with out.open('xb') as writer,src.open('rb') as reader:shutil.copyfileobj(reader,writer)
                    if row['has_txt']:shutil.copyfile(src.with_suffix('.txt'),out.with_suffix('.txt'))
                    if desktop.digest(src)!=row['sha256'] or sidecar(src)[1]!=row['caption_revision']:raise ValueError('処理中に元ファイルが変わりました')
                stage.rename(root/plan['copy_folder'])
            except Exception:
                if stage.exists() and stage.parent==root:shutil.rmtree(stage)
                raise
        else:_execute(project_id,rows,token)
        record={**plan,'token':token,'status':'applied'};write_state(_journal(project_id)/(token+'.json'),json.dumps(record,ensure_ascii=False,indent=2))
        return {'token':token,'folder':plan['folder'],'relatives':[r['after'] for r in rows],'count':plan['changed'],'undoable':plan['mode']=='rename'}

def undo(payload):
    project_id=int(payload['project_id']);token=str(payload.get('token',''))
    if not re.fullmatch('[0-9a-f]{32}',token):raise ValueError('復元記録が不正です')
    directory=_journal(project_id)
    with FileLock(str(directory/'operation.lock'),timeout=0):
        path=directory/(token+'.json');record=json.loads(path.read_text(encoding='utf8'))
        if record['status']!='applied' or record['mode']!='rename':raise ValueError('この命名整理は復元できません')
        root=root_for(project_id);rows=[]
        for row in record['rows']:
            current=image_path(root,row['after'])
            if desktop.digest(current)!=row['sha256']:raise ValueError('命名後に画像が編集・差し替えされています。現在のファイルは変更していません')
            rows.append({**row,'before':row['after'],'after':row['before'],'caption_revision':sidecar(current)[1],'has_txt':current.with_suffix('.txt').exists()})
        _execute(project_id,rows,uuid.uuid4().hex)
        record['status']='undone';write_state(path,json.dumps(record,ensure_ascii=False,indent=2))
        return {'folder':record['folder'],'relatives':[r['after'] for r in rows],'count':record['changed']}
