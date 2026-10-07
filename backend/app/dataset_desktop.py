"""Explicit desktop dataset actions. Originals are never moved or overwritten."""
from __future__ import annotations
import hashlib,io,ipaddress,json,os,shutil,socket,tempfile,uuid
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request,build_opener,HTTPRedirectHandler
from PIL import Image,ImageOps,UnidentifiedImageError
from urllib.error import HTTPError,URLError
from .db import get_conn
from .routers.dataset_files import root_for,image_path,sidecar

ROOT=Path(__file__).resolve().parents[2]
def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def open_folder(folder: str) -> dict:
    dataset=Path(folder).expanduser().resolve(strict=True)
    if not dataset.is_dir():raise ValueError('画像のあるフォルダを選択してください')
    conn=get_conn()
    try:
        conn.execute('BEGIN IMMEDIATE')
        for row in conn.execute('SELECT * FROM projects'):
            if Path(row['dataset_dir']).resolve()==dataset:
                return dict(row)
        label=dataset.name or str(dataset.anchor)
        names={row[0] for row in conn.execute('SELECT name FROM projects')}
        name=label;index=2
        while name in names:name=f'{label} ({index})';index+=1
        managed=ROOT/'projects'/('linked-'+uuid.uuid4().hex[:12])
        for directory in (managed/'outputs',managed/'library'):directory.mkdir(parents=True,exist_ok=False)
        cur=conn.execute("INSERT INTO projects(name,project_type,base_dir,dataset_dir,captions_dir,outputs_dir,library_dir,status) VALUES(?,'character',?,?,?,?,?,'idle')",
                         (name,str(managed),str(dataset),str(dataset),str(managed/'outputs'),str(managed/'library')))
        row=conn.execute('SELECT * FROM projects WHERE id=?',(cur.lastrowid,)).fetchone();conn.commit()
        return dict(row)
    finally:conn.close()

def _settings(payload):
    mode=payload.get('mode','pad')
    if mode not in ('pad','longest','shortest','crop'):raise ValueError('サイズの揃え方が不正です')
    width=int(payload.get('width',1024));height=int(payload.get('height',1024))
    if not (64<=width<=4096 and 64<=height<=4096):raise ValueError('サイズは64〜4096pxで指定してください')
    alignment=int(payload.get('alignment',1))
    if alignment not in (1,8):raise ValueError('寸法の単位が不正です')
    return mode,width,height,bool(payload.get('allow_upscale',False)),alignment

def _geometry(size,mode,width,height,upscale,alignment=1):
    w,h=size
    if mode=='crop':
        crop_w=min(w,h*width/height);crop_h=crop_w*height/width;scale=width/crop_w
        if not upscale:scale=min(scale,1.)
        output=(max(1,round(crop_w*scale)),max(1,round(crop_h*scale)));resized=output
    else:
        scale=min(width/w,height/h) if mode=='pad' else width/(min(w,h) if mode=='shortest' else max(w,h))
        if not upscale:scale=min(scale,1.)
        resized=(max(1,round(w*scale)),max(1,round(h*scale)));output=(width,height) if mode=='pad' else resized
    if alignment>1:
        if min(output)<alignment:raise ValueError('8px単位にできない細い画像です。寸法設定を確認してください')
        output=tuple(v//alignment*alignment for v in output)
        if mode!='pad':resized=output
        else:
            scale=min(output[0]/w,output[1]/h)
            if not upscale:scale=min(scale,1.)
            resized=(max(1,round(w*scale)),max(1,round(h*scale)))
    if max(output)>16384 or output[0]*output[1]>50_000_000:raise ValueError('処理後の画像が大きすぎます。短辺指定や目標サイズを下げてください')
    return resized,output,scale

def _load(path):
    with Image.open(path) as raw:
        if getattr(raw,'n_frames',1)>1:raise ValueError(f'アニメーション画像は未対応です: {path.name}')
        if raw.width*raw.height>50_000_000:raise ValueError(f'画像が大きすぎます: {path.name}')
        return ImageOps.exif_transpose(raw).convert('RGBA')

def _render(image,mode,width,height,upscale,alignment=1):
    image=image.convert('RGBA')
    size,output,_=_geometry(image.size,mode,width,height,upscale,alignment)
    resized=ImageOps.fit(image,output,method=Image.Resampling.LANCZOS,centering=(.5,.5)) if mode=='crop' else image.resize(size,Image.Resampling.LANCZOS)
    canvas=Image.new('RGB',output,'white')
    canvas.paste(resized,((output[0]-size[0])//2,(output[1]-size[1])//2),resized)
    return canvas

def _thumbnail(image):
    import base64
    image=image.copy();image.thumbnail((320,320));buf=io.BytesIO();image.save(buf,format='PNG')
    return 'data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode('ascii')

def _inputs(payload):
    root=root_for(int(payload['project_id']));rels=list(dict.fromkeys(payload.get('relatives') or []))
    if not 1<=len(rels)<=200:raise ValueError('1〜200枚の画像を選択してください')
    result=[]
    for rel in rels:
        path=image_path(root,rel);caption,revision=sidecar(path)
        result.append((rel,path,caption,revision))
    return root,result

def preview_normalize(payload):
    mode,width,height,upscale,alignment=_settings(payload);root,inputs=_inputs(payload);items=[];before=after=None
    for rel,path,caption,revision in inputs:
        image=_load(path);size,out,scale=_geometry(image.size,mode,width,height,upscale,alignment)
        items.append({'relative':rel,'source_size':list(image.size),'output_size':list(out),'content_size':list(size),'scale':round(scale,3),'sha256':digest(path),'caption_revision':revision})
        if before is None:before=_thumbnail(image);after=_thumbnail(_render(image,mode,width,height,upscale,alignment))
    return {'items':items,'before':before,'after':after,'mode':mode,'width':width,'height':height,'allow_upscale':upscale,'alignment':alignment,'count':len(items)}

def apply_normalize(payload):
    mode,width,height,upscale,alignment=_settings(payload);root,inputs=_inputs(payload)
    expected={item['relative']:item for item in payload.get('expected',[])}
    if len(expected)!=len(inputs):raise ValueError('先にプレビューを確認してください')
    for rel,path,caption,revision in inputs:
        check=expected.get(rel,{})
        if check.get('sha256')!=digest(path) or check.get('caption_revision')!=revision:raise ValueError('元画像またはTXTが変わりました。プレビューを更新してください')
    stage=Path(tempfile.mkdtemp(prefix='.normalize-',dir=root));records=[]
    try:
        names=set()
        for index,(rel,path,caption,revision) in enumerate(inputs):
            name=path.stem+'.png'
            suffix=2
            while name.casefold() in names:
                name=f'{path.stem}_{suffix}.png';suffix+=1
            names.add(name.casefold());result=_render(_load(path),mode,width,height,upscale,alignment)
            result.save(stage/name,format='PNG')
            if path.with_suffix('.txt').is_file():shutil.copyfile(path.with_suffix('.txt'),(stage/name).with_suffix('.txt'))
            records.append({'source':str(path),'source_sha256':expected[rel]['sha256'],'file':name,'output_size':list(result.size)})
        for rel,path,caption,revision in inputs:
            if digest(path)!=expected[rel]['sha256'] or sidecar(path)[1]!=revision:raise ValueError('処理中に元画像またはTXTが変わりました。元データは変更していません')
        (stage/'.normalization.json').write_text(json.dumps({'settings':{'mode':mode,'width':width,'height':height,'allow_upscale':upscale,'alignment':alignment},'items':records},ensure_ascii=False,indent=2),encoding='utf-8')
        label=f'normalized-{width}'+(f'x{height}' if mode in ('pad','crop') else '-short' if mode=='shortest' else '-long')+('-crop' if mode=='crop' else '')+'-'+datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:4]
        final=root/label;stage.rename(final)
        return {'folder':label,'count':len(records),'relatives':[f'{label}/{r["file"]}' for r in records],'originals_preserved':True}
    except Exception:
        # Only this operation's new staging directory; never source files.
        if stage.exists() and stage.parent==root:shutil.rmtree(stage)
        raise

def _public_url(url):
    parsed=urlparse(url)
    if parsed.scheme not in ('https','http') or not parsed.hostname or parsed.username or parsed.password:raise ValueError('公開画像のhttp/https URLを指定してください')
    addresses=socket.getaddrinfo(parsed.hostname,parsed.port or (443 if parsed.scheme=='https' else 80),type=socket.SOCK_STREAM)
    if not addresses or any(not ipaddress.ip_address(row[4][0]).is_global for row in addresses):raise ValueError('ローカル・プライベートネットワークのURLは取り込みません')

class PublicRedirects(HTTPRedirectHandler):
    max_redirections=5
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        _public_url(newurl)
        return super().redirect_request(req,fp,code,msg,headers,newurl)

def import_web_image(payload):
    root=root_for(int(payload['project_id']));from .routers.dataset_files import within
    folder=str(payload.get('folder') or '');destination=within(root,folder)
    if not destination.is_dir():raise ValueError('取り込み先のフォルダがありません')
    url=str(payload.get('url') or '');_public_url(url)
    request=Request(url,headers={'User-Agent':'Mozilla/5.0 LoRA-Studio/1.0','Accept':'image/*'})
    try:
        with build_opener(PublicRedirects()).open(request,timeout=20) as response:
            raw=response.read(25*1024*1024+1)
    except HTTPError as error:
        raise ValueError(f'画像を取得できませんでした（HTTP {error.code}）。公開画像のURLとサイトの保存制限を確認してください') from error
    except (URLError,TimeoutError) as error:
        raise ValueError('画像の取得に失敗しました。接続と画像URLを確認して再試行してください') from error
    if len(raw)>25*1024*1024:raise ValueError('画像は25MB以下にしてください')
    try:decoded=Image.open(io.BytesIO(raw))
    except UnidentifiedImageError as error:
        raise ValueError('このURLから画像を読み取れませんでした。WebページのURLではなく、画像のURLを指定してください') from error
    with decoded as image:
        if getattr(image,'n_frames',1)>1:raise ValueError('アニメーション画像はファイル保存後に確認してください')
        if image.width*image.height>50_000_000:raise ValueError('画像が大きすぎます')
        image.load();image=ImageOps.exif_transpose(image).convert('RGBA');width,height=image.size
        name='web-'+datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8]+'.png'
        target=destination/name
        encoded=io.BytesIO();image.save(encoded,format='PNG')
        owned=False
        try:
            with target.open('xb') as stream:
                owned=True;stream.write(encoded.getvalue())
        except Exception:
            if owned:target.unlink(missing_ok=True)
            raise
    return {'name':name,'relative':target.relative_to(root).as_posix(),'width':width,'height':height,'bytes':target.stat().st_size}
