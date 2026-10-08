"""A project-scoped, file-backed dataset editor. Sidecar TXT is authoritative here."""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from PIL import Image

from ..db import get_conn

router = APIRouter(prefix="/dataset-files", tags=["dataset-files"])
EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_lock = threading.RLock()
ORDER_FILE = '.dataset-order.json'


@contextmanager
def database_connection():
    conn = get_conn()
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def folder_order(root: Path, directory: Path, names: list[str]) -> tuple[list[str], str]:
    metadata = within(root, str((directory / ORDER_FILE).relative_to(root)))
    raw = metadata.read_bytes() if metadata.exists() else b''
    saved = []
    if raw:
        try:
            saved = json.loads(raw).get('items', [])
        except (ValueError, AttributeError):
            raise HTTPException(409, f'並べ替え情報を読み込めません: {metadata}')
    known = set(names)
    order = list(dict.fromkeys(n for n in saved if isinstance(n, str) and n in known))
    order.extend(n for n in names if n not in order)
    revision = hashlib.sha256(raw + json.dumps(sorted(names), ensure_ascii=False).encode()).hexdigest()
    return order, revision


def write_order(root: Path, directory: Path, order: list[str]) -> None:
    target = within(root, str((directory / ORDER_FILE).relative_to(root)))
    atomic_text(target, json.dumps({'version': 1, 'items': order}, ensure_ascii=False, indent=2))


def root_for(project_id: int) -> Path:
    with database_connection() as conn:
        row = conn.execute("SELECT dataset_dir FROM projects WHERE id=?", (project_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "プロジェクトがありません")
    root = Path(row["dataset_dir"]).resolve()
    if not root.is_dir():
        raise HTTPException(404, f"保存フォルダがありません: {root}")
    return root


def within(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise HTTPException(400, "データセットの外側は操作できません")
    return path


def image_path(root: Path, relative: str) -> Path:
    path = within(root, relative)
    if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
        raise HTTPException(404, "画像が見つかりません。再読込してください")
    # Sidecars may be links too: never write outside the dataset.
    within(root, str(path.with_suffix('.txt').relative_to(root)))
    return path


def sidecar(path: Path) -> tuple[str, str]:
    txt = path.with_suffix(".txt")
    data = txt.read_bytes() if txt.exists() else b""
    try:
        caption = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(422, f"UTF-8で保存してください: {txt.name}")
    image_digest = hashlib.sha256()
    with path.open('rb') as image_file:
        for block in iter(lambda: image_file.read(1024 * 1024), b''):
            image_digest.update(block)
    # The edit token identifies the image bytes AND its caption, not just the
    # filename or equal empty TXT. Renaming pairs preserves it; swapping images
    # makes stale generated/manual captions conflict before any write.
    revision = hashlib.sha256(data).hexdigest() + ':' + image_digest.hexdigest()
    return caption, revision + (":exists" if txt.exists() else ":missing")


def atomic_text(path: Path, text: str) -> None:
    fd, name = tempfile.mkstemp(prefix=".caption-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def sync_registered(project_id: int, path: Path, caption: str) -> None:
    # Only explicit gallery registration opts into sidecar ownership.
    with database_connection() as conn:
        conn.execute("UPDATE basepipe_assets SET caption=?, caption_edited=?, caption_processed='', training_input=?, training_input_source='edited', caption_source='sidecar', updated_at=CURRENT_TIMESTAMP WHERE project_id=? AND file_path=? AND origin_kind='dataset_file'",
                     (caption, caption, caption, project_id, str(path)))


def protected_paths(conn) -> set[str]:
    paths = {str(Path(row['file_path']).resolve()).casefold() for row in conn.execute("SELECT file_path FROM basepipe_assets WHERE origin_kind<>'dataset_file'")}
    for row in conn.execute("SELECT a.file_path,e.metadata_json FROM basepipe_snapshot_entries e LEFT JOIN basepipe_assets a ON a.id=e.asset_id"):
        try:
            frozen = json.loads(row['metadata_json'] or '{}').get('snapshot', {})
        except (ValueError, AttributeError):
            frozen = {}
        path = frozen.get('path') if isinstance(frozen, dict) else None
        path = path or row['file_path']
        if path:
            paths.add(str(Path(path).resolve()).casefold())
    return paths


def require_editable_source(path: Path, *, allow_snapshot_reference: bool = False) -> None:
    with database_connection() as conn:
        protected = ({str(Path(row['file_path']).resolve()).casefold() for row in conn.execute("SELECT file_path FROM basepipe_assets WHERE origin_kind<>'dataset_file'")}
                     if allow_snapshot_reference else protected_paths(conn))
    if str(path.resolve()).casefold() in protected:
        raise HTTPException(409, "確定した学習対象または生成・レビューが参照している原本です。元ファイルを残して学習用に複製してください")


def _folder_preview(folder: Path, n: int = 4, max_dirs: int = 200) -> list[str]:
    """Up to n images inside a folder (subfolders too, shallow first) so the folder can be shown with thumbnails like Explorer."""
    found: list[str] = []
    seen = 0
    for current, dirs, files in os.walk(folder):
        dirs[:] = sorted(d for d in dirs if not d.startswith('.'))
        for name in sorted(files):
            if Path(name).suffix.lower() in EXTENSIONS:
                found.append(str(Path(current) / name))
                if len(found) >= n:
                    return found
        seen += 1
        if seen >= max_dirs:
            break
    return found


@router.get("/{project_id}")
def listing(project_id: int, folder: str = "") -> dict:
    root = root_for(project_id)
    directory = within(root, folder)
    if not directory.is_dir():
        raise HTTPException(404, "フォルダが見つかりません")
    items, folders, errors = [], [], []
    with database_connection() as conn:
        registered = {row['file_path']: dict(row) for row in conn.execute("SELECT id,file_path,training_enabled,review_status FROM basepipe_assets WHERE project_id=? AND origin_kind='dataset_file'", (project_id,))}
        protected = protected_paths(conn)
        copy_required = {str(Path(row['file_path']).resolve()).casefold() for row in conn.execute("SELECT file_path FROM basepipe_assets WHERE origin_kind<>'dataset_file'")}
    for path in sorted(directory.iterdir(), key=lambda p: p.name.casefold()):
        if path.name.startswith('.') or path.is_symlink() or not path.resolve().is_relative_to(root):
            continue
        relative = path.relative_to(root).as_posix()
        if path.is_dir():
            folders.append({"name": path.name, "relative": relative, "preview": _folder_preview(path)})
        elif path.suffix.lower() in EXTENSIONS:
            try:
                image_path(root, relative)
                caption, revision = sidecar(path)
                with Image.open(path) as im:
                    width, height = im.size
                stat = path.stat()
                items.append({"name": path.name, "relative": relative, "path": str(path), "caption_path": str(path.with_suffix('.txt')), "caption": caption, "revision": revision, "width": width, "height": height, "bytes": stat.st_size, "modified": stat.st_mtime_ns, "asset": registered.get(str(path)), "protected_source": str(path.resolve()).casefold() in protected, "requires_training_copy": str(path.resolve()).casefold() in copy_required})
            except (OSError, ValueError, HTTPException) as exc:
                errors.append(f"{path.name}: {getattr(exc, 'detail', str(exc))}")
    order, revision = folder_order(root, directory, [i['relative'] for i in items])
    by_name = {i['relative']: i for i in items}
    return {"root": str(root), "folder": '' if directory == root else directory.relative_to(root).as_posix(), "directory": str(directory), "folders": folders, "items": [by_name[n] for n in order], "errors": errors, 'order_revision': revision}


class OrderEdit(BaseModel):
    folder: str = ''
    order: list[str] = Field(max_length=10000)
    revision: str


@router.post('/{project_id}/order')
def reorder(project_id: int, payload: OrderEdit):
    with _lock:
        current = listing(project_id, payload.folder)
        names = [i['relative'] for i in current['items']]
        if payload.revision != current['order_revision']:
            raise HTTPException(409, '別の画面または外部で一覧が変わりました。再読込してください')
        if len(set(payload.order)) != len(payload.order) or set(payload.order) != set(names):
            raise HTTPException(409, '非表示の画像を含む全件の順序が一致しません。再読込してください')
        root = root_for(project_id)
        write_order(root, within(root, payload.folder), payload.order)
        return listing(project_id, payload.folder)


class Rescan(BaseModel):
    folder: str = ""


@router.post("/{project_id}/rescan")
def rescan(project_id: int, payload: Rescan):
    with _lock:
        result = listing(project_id, payload.folder)
        for item in result['items']:
            if item['asset']:
                path = Path(item['path'])
                sync_registered(project_id, path, item['caption'])
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                with database_connection() as conn:
                    conn.execute("UPDATE basepipe_assets SET content_sha256=?,review_status='pending',training_enabled=0 WHERE id=? AND content_sha256<>?", (digest, item['asset']['id'], digest))
        with database_connection() as conn:
            rows = conn.execute("SELECT id,file_path FROM basepipe_assets WHERE project_id=? AND origin_kind='dataset_file'", (project_id,)).fetchall()
            for row in rows:
                if not Path(row['file_path']).is_file():
                    conn.execute("UPDATE basepipe_assets SET training_enabled=0,review_status='needs_review' WHERE id=?", (row['id'],))
        return result


@router.get("/{project_id}/image")
def image_file(project_id: int, relative: str):
    return FileResponse(image_path(root_for(project_id), relative))


class CaptionEdit(BaseModel):
    relative: str
    caption: str = Field(max_length=10000)
    revision: str


class CaptionBatch(BaseModel):
    edits: list[CaptionEdit] = Field(min_length=1, max_length=500)


@router.post("/{project_id}/captions")
def save_captions(project_id: int, payload: CaptionBatch):
    with _lock:
        root = root_for(project_id)
        pending = []
        for edit in payload.edits:
            path = image_path(root, edit.relative)
            old, rev = sidecar(path)
            if rev != edit.revision:
                raise HTTPException(409, f"画像またはTXTが変更されています: {path.name}。生成結果・下書きは保存せず保持しています。対象を確認して再読込してください")
            pending.append((path, edit.caption, old))
        if len({p.with_suffix('.txt') for p, _, _ in pending}) != len(pending):
            raise HTTPException(400, "同名のキャプションが重複しています")
        results = []
        for path, caption, _ in pending:
            # Per-file atomic replacement. The response never claims a cross-file transaction.
            atomic_text(path.with_suffix('.txt'), caption)
            sync_registered(project_id, path, caption)
            results.append({"relative": path.relative_to(root).as_posix(), "revision": sidecar(path)[1]})
        return {"saved": results}


class Rename(BaseModel):
    relative: str
    name: str = Field(min_length=1, max_length=200)


@router.post("/{project_id}/rename")
def rename(project_id: int, payload: Rename):
    with _lock:
        root = root_for(project_id)
        source = image_path(root, payload.relative)
        require_editable_source(source)
        if re.search(r'[<>:"/\\|?*\x00-\x1f]', payload.name) or payload.name.endswith((' ', '.')):
            raise HTTPException(400, "ファイル名に使えない文字が含まれています")
        target = source.with_name(payload.name)
        if target.suffix.lower() != source.suffix.lower():
            raise HTTPException(400, "拡張子は変更せず名前を入力してください")
        if target == source:
            return {"relative": payload.relative}
        if target.exists() or target.with_suffix('.txt').exists():
            raise HTTPException(409, "同じ名前の画像またはTXTが既にあります")
        # A shared stem cannot safely move its sidecar.
        if sum(p.is_file() and p.stem == source.stem and p.suffix.lower() in EXTENSIONS for p in source.parent.iterdir()) > 1:
            raise HTTPException(409, "同名・別拡張子の画像があります。TXTの共有を解消してください")
        old_order = [i['relative'] for i in listing(project_id, source.parent.relative_to(root).as_posix())['items']]
        txt = source.with_suffix('.txt')
        txt_moved = order_written = False
        source.rename(target)
        try:
            if txt.exists():
                txt.rename(target.with_suffix('.txt'))
                txt_moved = True
            with database_connection() as conn:
                conn.execute("UPDATE basepipe_assets SET file_path=?,asset_key=?,updated_at=CURRENT_TIMESTAMP WHERE project_id=? AND file_path=? AND origin_kind='dataset_file'", (str(target), target.name, project_id, str(source)))
                conn.execute('UPDATE dataset_items SET file_path=? WHERE project_id=? AND file_path=?', (str(target), project_id, str(source)))
                write_order(root, source.parent, [target.relative_to(root).as_posix() if r == source.relative_to(root).as_posix() else r for r in old_order])
                order_written = True
        except Exception:
            target.rename(source)
            if txt_moved:
                target.with_suffix('.txt').rename(txt)
            if order_written:
                write_order(root, source.parent, old_order)
            raise
        return {"relative": target.relative_to(root).as_posix()}


class Paths(BaseModel):
    relatives: list[str] = Field(min_length=1, max_length=500)


class MoveFiles(Paths):
    target_folder: str


@router.post('/{project_id}/move')
def move_files(project_id: int, payload: MoveFiles):
    with _lock:
        root = root_for(project_id)
        sources = [image_path(root, r) for r in dict.fromkeys(payload.relatives)]
        target_dir = within(root, payload.target_folder)
        if not target_dir.is_dir():
            raise HTTPException(404, '移動先のフォルダがありません')
        if any(p.parent == target_dir for p in sources):
            raise HTTPException(409, '現在のフォルダには移動できません')
        pairs = {}
        for source in sources:
            require_editable_source(source)
            pairs[source] = target_dir / source.name
            txt = source.with_suffix('.txt')
            if txt.exists():
                if any(p != source and p.stem == source.stem and p.suffix.lower() in EXTENSIONS and p not in sources for p in source.parent.iterdir()):
                    raise HTTPException(409, 'TXTを共有する同名画像も一緒に選択してください')
                pairs[txt] = target_dir / txt.name
        targets = list(pairs.values())
        if len(set(targets)) != len(targets) or any(p.exists() for p in targets):
            raise HTTPException(409, '移動先に同名ファイルがあります。上書きせず中止しました')
        # A pre-existing orphan sidecar must not be silently attached to the moved image.
        if any((target_dir / p.name).with_suffix('.txt').exists() for p in sources):
            raise HTTPException(409, '移動先に同名TXTがあります')
        dest_before = listing(project_id, payload.target_folder)
        moved = []
        try:
            for old, new in pairs.items():
                old.rename(new)
                moved.append((old, new))
            with database_connection() as conn:
                for old in sources:
                    new = target_dir / old.name
                    conn.execute("UPDATE basepipe_assets SET file_path=?,updated_at=CURRENT_TIMESTAMP WHERE project_id=? AND file_path=? AND origin_kind='dataset_file'", (str(new), project_id, str(old)))
                    conn.execute('UPDATE dataset_items SET file_path=? WHERE project_id=? AND file_path=?', (str(new), project_id, str(old)))
                order = [i['relative'] for i in dest_before['items']] + [(target_dir / p.name).relative_to(root).as_posix() for p in sources]
                write_order(root, target_dir, order)
        except Exception:
            for old, new in reversed(moved):
                new.rename(old)
            raise
        return {'moved': len(sources), 'target_folder': payload.target_folder, 'relatives': [(target_dir / p.name).relative_to(root).as_posix() for p in sources]}


@router.post("/{project_id}/copy-for-training")
def copy_for_training(project_id: int, payload: Paths):
    root = root_for(project_id)
    with _lock:
        sources = [image_path(root, r) for r in payload.relatives]
        target_dir = within(root, 'training-inputs')
        target_dir.mkdir(exist_ok=True)
        copied = []
        for source in sources:
            target = target_dir / source.name
            if target.exists() or target.with_suffix('.txt').exists():
                target = target.with_name(f'{source.stem}_{uuid.uuid4().hex[:8]}{source.suffix}')
            shutil.copy2(source, target)
            txt = source.with_suffix('.txt')
            if txt.exists():
                shutil.copy2(txt, target.with_suffix('.txt'))
            copied.append(target.relative_to(root).as_posix())
        return {'folder':'training-inputs','relatives':copied}


@router.post("/{project_id}/register")
def register(project_id: int, payload: Paths):
    return register_relatives(project_id, payload.relatives)


def register_relatives(project_id: int, relatives: list[str]) -> dict:
    """Register images as dataset_file assets (no count limit; the HTTP schema limits only the single call)."""
    from .basepipe import create_asset
    from ..schemas import BasepipeAssetCreate
    root = root_for(project_id)
    ids = []
    from .evaluation import read_reference,digest
    reference=read_reference(project_id)
    excluded=[]
    with _lock:
        for relative in relatives:
            path = image_path(root, relative)
            if reference and digest(path)==reference['sha256']:
                excluded.append(relative)
                continue
            # Registration/caption refresh does not move or alter the frozen image.
            require_editable_source(path, allow_snapshot_reference=True)
            caption, _ = sidecar(path)
            with database_connection() as conn:
                found = conn.execute("SELECT id,origin_kind FROM basepipe_assets WHERE project_id=? AND file_path=?", (project_id, str(path))).fetchone()
            if found and found['origin_kind'] != 'dataset_file':
                raise HTTPException(409, f"既存の生成・レビュー素材です。元のレビュー画面で管理してください: {path.name}")
            if found:
                sync_registered(project_id, path, caption)
                ids.append(found['id'])
            else:
                item = create_asset(project_id, BasepipeAssetCreate(file_path=str(path), asset_key=path.name, caption=caption, caption_source='sidecar', origin_kind='dataset_file', review_status='pending'))
                ids.append(item['id'])
    return {"asset_ids": ids, "excluded_evaluation_images": excluded, "message": "レビューに登録しました。採用はレビュー画面で確定してください"}


class Location(BaseModel):
    relative: str = ""


@router.post("/{project_id}/exclude")
def exclude(project_id: int, payload: Paths):
    """Move out of the dataset, not delete. A receipt permits restoration."""
    with _lock:
        root = root_for(project_id)
        paths = [image_path(root, r) for r in payload.relatives]
        for path in paths:
            require_editable_source(path)
        token = uuid.uuid4().hex
        archive = root.parent / '.dataset-excluded' / token
        pairs = []
        for path in paths:
            pairs.append((path, archive / path.relative_to(root)))
            txt = path.with_suffix('.txt')
            if txt.exists() and all(txt != old for old, _ in pairs):
                if any(p != path and p.stem == path.stem and p.suffix.lower() in EXTENSIONS and p not in paths for p in path.parent.iterdir()):
                    raise HTTPException(409, "同名画像がTXTを共有しています。同名画像を一緒に選択してください")
                pairs.append((txt, archive / txt.relative_to(root)))
        moved = []
        try:
            for old, new in pairs:
                new.parent.mkdir(parents=True, exist_ok=True)
                old.rename(new)
                moved.append((old,new))
            atomic_text(archive/'receipt.json', json.dumps({'root':str(root),'paths':[str(p.relative_to(root)) for p,_ in pairs]}, ensure_ascii=False))
        except Exception:
            for old,new in reversed(moved):
                new.rename(old)
            raise
        with database_connection() as conn:
            for path in paths:
                conn.execute("UPDATE basepipe_assets SET training_enabled=0,review_status='needs_review' WHERE project_id=? AND file_path=? AND origin_kind='dataset_file'", (project_id,str(path)))
        return {'token':token,'count':len(paths),'archive':str(archive)}


def recycle(paths: list[Path]) -> None:
    """Send files/folders to the Windows Recycle Bin (restorable); never a silent permanent delete."""
    if os.name != 'nt':
        raise HTTPException(501, "ごみ箱への移動はWindowsのみ対応です")
    import ctypes
    from ctypes import wintypes

    class SHFILEOPSTRUCTW(ctypes.Structure):
        _fields_ = [('hwnd', wintypes.HWND), ('wFunc', wintypes.UINT), ('pFrom', ctypes.c_void_p), ('pTo', ctypes.c_void_p),
                    ('fFlags', ctypes.c_ushort), ('fAnyOperationsAborted', wintypes.BOOL), ('hNameMappings', ctypes.c_void_p),
                    ('lpszProgressTitle', wintypes.LPCWSTR)]
    names = [str(path) for path in paths if path.exists()]
    if not names:
        return
    buffer = ctypes.create_unicode_buffer('\0'.join(names) + '\0\0', sum(len(n) + 1 for n in names) + 1)
    # FO_DELETE | FOF_SILENT | FOF_NOCONFIRMATION | FOF_ALLOWUNDO | FOF_NOERRORUI | FOF_WANTNUKEWARNING
    op = SHFILEOPSTRUCTW(None, 3, ctypes.addressof(buffer), None, 0x0004 | 0x0010 | 0x0040 | 0x0400 | 0x4000, False, None, None)
    code = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(op))
    if code != 0 or op.fAnyOperationsAborted or any(Path(n).exists() for n in names):
        raise HTTPException(500, f"ごみ箱へ移動できませんでした（コード {code}）。ファイルが他のアプリで開かれていないか確認してください")


def _mark_removed(project_id: int, paths: list[Path]) -> None:
    with database_connection() as conn:
        for path in paths:
            conn.execute("UPDATE basepipe_assets SET training_enabled=0,review_status='needs_review' WHERE project_id=? AND file_path=? AND origin_kind='dataset_file'", (project_id, str(path)))


@router.post("/{project_id}/trash")
def trash(project_id: int, payload: Paths):
    """Delete images (and their TXT) into the Recycle Bin. Files referenced by sealed snapshots are refused."""
    with _lock:
        root = root_for(project_id)
        paths = [image_path(root, r) for r in payload.relatives]
        for path in paths:
            require_editable_source(path)
        targets = list(paths)
        for path in paths:
            txt = path.with_suffix('.txt')
            if txt.exists() and txt not in targets:
                if any(p != path and p.stem == path.stem and p.suffix.lower() in EXTENSIONS and p not in paths for p in path.parent.iterdir()):
                    raise HTTPException(409, "同名画像がTXTを共有しています。同名画像を一緒に選択してください")
                targets.append(txt)
        recycle(targets)
        _mark_removed(project_id, paths)
        return {'count': len(paths)}


class FolderPath(BaseModel):
    folder: str = Field(min_length=1, max_length=1024)


@router.post("/{project_id}/trash-folder")
def trash_folder(project_id: int, payload: FolderPath):
    """Delete a dataset subfolder (never the dataset root) into the Recycle Bin."""
    with _lock:
        root = root_for(project_id)
        directory = within(root, payload.folder)
        if directory == root or not directory.is_dir():
            raise HTTPException(404, "フォルダが見つかりません")
        images = [p for p in directory.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS]
        for image in images:
            require_editable_source(image)
        recycle([directory])
        _mark_removed(project_id, images)
        return {'count': len(images)}


class NewFolder(BaseModel):
    parent: str = Field(default='', max_length=1024)
    name: str = Field(min_length=1, max_length=120)


@router.post("/{project_id}/create-folder")
def create_folder(project_id: int, payload: NewFolder):
    name = payload.name.strip()
    if not name or name in {'.', '..'} or re.search(r'[\\/:*?"<>|\x00-\x1f]', name) or name.endswith('.') or name.startswith('.'):
        raise HTTPException(400, "フォルダ名に使えない文字が含まれています")
    with _lock:
        root = root_for(project_id)
        target = within(root, f"{payload.parent}/{name}" if payload.parent else name)
        if target.exists():
            raise HTTPException(409, "同じ名前のフォルダがあります")
        target.mkdir(parents=False)
        return {'relative': target.relative_to(root).as_posix()}


class Restore(BaseModel):
    token: str = Field(pattern=r'^[0-9a-f]{32}$')


@router.post("/{project_id}/restore")
def restore(project_id: int, payload: Restore):
    with _lock:
        root = root_for(project_id)
        archive = root.parent / '.dataset-excluded' / payload.token
        receipt = archive/'receipt.json'
        if not receipt.is_file():
            raise HTTPException(404, "退避記録がありません")
        record = json.loads(receipt.read_text(encoding='utf-8'))
        if record['root'] != str(root):
            raise HTTPException(409, "元の保存場所と一致しません")
        pairs = [(within(archive,r),within(root,r)) for r in record['paths']]
        if any(not a.is_file() or b.exists() for a,b in pairs):
            raise HTTPException(409, "元の場所に同名ファイルがあるか退避ファイルがありません。上書きせず中止しました")
        moved=[]
        try:
            for old,new in pairs:
                new.parent.mkdir(parents=True, exist_ok=True)
                old.rename(new);moved.append((old,new))
        except Exception:
            for old,new in reversed(moved):new.rename(old)
            raise
        return {'restored':len(pairs),'message':'元の場所へ戻しました。採用状態は再確認してください'}


@router.post("/{project_id}/reveal")
def reveal(project_id: int, payload: Location):
    path = within(root_for(project_id), payload.relative)
    if not path.exists():
        raise HTTPException(404, "保存先が見つかりません")
    if os.name != 'nt':
        raise HTTPException(400, "Windowsのみ対応しています")
    args = ['explorer.exe', '/select,', str(path)] if path.is_file() else ['explorer.exe', str(path)]
    subprocess.Popen(args)
    return {"opened": str(path)}


@router.post("/{project_id}/upload")
async def upload(project_id: int, folder: str = "", files: list[UploadFile] = File(...)):
    root = root_for(project_id)
    directory = within(root, folder)
    if not directory.is_dir() or len(files) > 500:
        raise HTTPException(400, "フォルダまたはファイル数が不正です")
    saved, rejected = [], []
    # Create-only: imports never overwrite existing user data.
    for file in files:
        name = (file.filename or '').replace('\\', '/')
        parts = name.split('/')
        if not name or any(p in ('', '.', '..') or p.endswith((' ', '.')) or re.search(r'[<>:"|?*\x00-\x1f]', p) for p in parts):
            raise HTTPException(400, "不正なファイル名です")
        if Path(name).suffix.lower() not in EXTENSIONS | {'.txt'}:
            rejected.append(name)
            continue
        target = within(root, str((directory / name).relative_to(root)))
        data = await file.read(50 * 1024 * 1024 + 1)
        if len(data) > 50 * 1024 * 1024:
            raise HTTPException(413, "1ファイル50MBまでです")
        try:
            if target.suffix.lower() == '.txt':
                data.decode('utf-8-sig')
            else:
                with Image.open(io.BytesIO(data)) as im:
                    im.verify()
        except (ValueError, OSError, UnicodeError):
            rejected.append(name)
            continue
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as f:
                f.write(data)
        except FileExistsError:
            continue
        saved.append(name)
    return {"saved": saved, "skipped": len(files) - len(saved), "rejected": rejected}


def walk_tree(root: Path, folder: str = '') -> tuple[list[str], list[str]]:
    """Recursive (directories, images) below ``folder`` as root-relative posix paths.

    Hidden (dot) folders/files and symlinks are skipped, like the folder listing.
    """
    base = within(root, folder)
    if not base.is_dir():
        raise HTTPException(404, "フォルダが見つかりません")
    dirs, images = [], []
    for dirpath, dirnames, filenames in os.walk(base):
        here = Path(dirpath)
        dirnames[:] = sorted((d for d in dirnames if not d.startswith('.') and not (here / d).is_symlink()), key=str.casefold)
        for d in dirnames:
            dirs.append((here / d).relative_to(root).as_posix())
        for name in sorted(filenames, key=str.casefold):
            path = here / name
            if name.startswith('.') or path.suffix.lower() not in EXTENSIONS or path.is_symlink() or not path.resolve().is_relative_to(root):
                continue
            images.append(path.relative_to(root).as_posix())
    return dirs, images


def current_snapshot_paths(project_id: int) -> tuple[set[str], int | None]:
    """Paths frozen into the project's CURRENT training dataset snapshot."""
    from ..training.snapshot_inputs import fetch_snapshot_inputs
    with database_connection() as conn:
        row = conn.execute("SELECT training_config_json FROM projects WHERE id=?", (project_id,)).fetchone()
        try:
            draft = json.loads(row['training_config_json'] or '{}') if row else {}
        except ValueError:
            draft = {}
        snapshot_id = draft.get('dataset_snapshot_id') if isinstance(draft, dict) else None
        if not snapshot_id:
            return set(), None
        paths = {str(Path(r['file_path']).resolve()).casefold() for r in fetch_snapshot_inputs(conn, int(snapshot_id)) if r.get('file_path')}
    return paths, int(snapshot_id)


@router.get("/{project_id}/summary")
def summary(project_id: int):
    """Per-folder image / training-target counts (recursive) and totals. The evaluation image is not counted."""
    from .evaluation import read_reference, digest
    root = root_for(project_id)
    dirs, images = walk_tree(root)
    targets, snapshot_id = current_snapshot_paths(project_id)
    reference = read_reference(project_id)
    ref_size = None
    if reference:
        ref_file = root / str(reference.get('reference_relative') or '')
        ref_size = ref_file.stat().st_size if ref_file.is_file() else None
    folders = {d: {"relative": d, "images": 0, "training_targets": 0, "direct": 0, "direct_targets": 0} for d in dirs}
    top = {"relative": "", "images": 0, "training_targets": 0, "direct": 0, "direct_targets": 0}
    for relative in images:
        path = root / relative
        if reference and (relative == reference.get('source_relative') or (ref_size is not None and path.stat().st_size == ref_size and digest(path) == reference.get('sha256'))):
            continue
        hit = 1 if str(path.resolve()).casefold() in targets else 0
        parts = relative.split('/')[:-1]
        owners = [top] + [folders["/".join(parts[:i + 1])] for i in range(len(parts))]
        for entry in owners:
            entry["images"] += 1
            entry["training_targets"] += hit
        owners[-1]["direct"] += 1
        owners[-1]["direct_targets"] += hit
    counted = [e for e in [top, *folders.values()] if e["direct"]]
    return {
        "snapshot_id": snapshot_id,
        "folders": list(folders.values()),
        "root": top,
        "totals": {"images": top["images"], "training_targets": top["training_targets"], "folders": len(counted), "folders_with_targets": sum(1 for e in counted if e["direct_targets"])},
    }


class PrepareScope(BaseModel):
    folder: str = Field(default='', max_length=1024)


def prepare_relatives(project_id: int, relatives: list[str], name: str) -> dict:
    """register -> approve -> ONE sealed snapshot. Same chain the selection button uses."""
    from .basepipe import bulk_review_assets, create_snapshot, get_snapshot
    from ..schemas import BasepipeAssetBulkReviewIn, BasepipeSnapshotCreate
    registered = register_relatives(project_id, relatives)
    ids = [int(i) for i in registered["asset_ids"]]
    if not ids:
        raise HTTPException(422, "評価用画像を除くと学習対象にできる画像がありません")
    for start in range(0, len(ids), 500):
        bulk_review_assets(project_id, BasepipeAssetBulkReviewIn(asset_ids=ids[start:start + 500], review_status="approved", training_enabled=True))
    created = create_snapshot(project_id, BasepipeSnapshotCreate(name=name[:160], asset_ids=ids))
    frozen = get_snapshot(int(created["id"]))
    return {**frozen, **created, "entries": frozen["entries"], "excluded_evaluation_images": registered["excluded_evaluation_images"]}


@router.post("/{project_id}/prepare-all")
def prepare_all(project_id: int, payload: PrepareScope):
    """Make every image under the dataset root (or one folder, recursively) the training target in one sealed snapshot."""
    from datetime import datetime
    with _lock:
        root = root_for(project_id)
        _, relatives = walk_tree(root, payload.folder)
        if not relatives:
            raise HTTPException(422, "学習対象にできる画像がありません（画像が1枚もありません）")
        stamp = datetime.now().strftime('%Y-%m-%d %H:%M')
        label = f"フォルダ {payload.folder}" if payload.folder else "全画像"
        return prepare_relatives(project_id, relatives, f"{label} · {stamp}")


class AssignInstance(BaseModel):
    folder: str = Field(default='', max_length=1024)
    token: str = Field(min_length=1, max_length=120)


@router.post("/{project_id}/assign-instance")
def assign_instance(project_id: int, payload: AssignInstance):
    """Give every image under a folder one instance trigger (replacing other instance triggers) in its TXT."""
    def tags_of(text: str) -> list[str]:
        return [t.strip() for t in re.split(r'[,\n]', text) if t.strip()]
    with _lock:
        root = root_for(project_id)
        with database_connection() as conn:
            rows = conn.execute("SELECT concept_type,trigger_token FROM basepipe_concepts WHERE project_id=? AND trigger_token<>''", (project_id,)).fetchall()
        instances = {str(r['trigger_token']).strip().casefold() for r in rows if r['concept_type'] == 'outfit'}
        shared = {str(r['trigger_token']).strip().casefold() for r in rows if r['concept_type'] != 'outfit'}
        chosen = payload.token.strip()
        if chosen.casefold() not in instances:
            raise HTTPException(422, "そのインスタンスは登録されていません")
        _, relatives = walk_tree(root, payload.folder)
        if not relatives:
            raise HTTPException(422, "このフォルダに画像がありません")
        plan = []
        for relative in relatives:  # validate everything before the first write
            path = image_path(root, relative)
            caption, _ = sidecar(path)
            tags = tags_of(caption)
            rest = [t for t in tags if t.casefold() not in instances]
            new = [t for t in rest if t.casefold() in shared] + [chosen] + [t for t in rest if t.casefold() not in shared]
            if new != tags:
                plan.append((path, caption, ', '.join(new)))
        written = []
        try:
            for path, old, new in plan:
                txt = path.with_suffix('.txt')
                original = txt.read_bytes() if txt.exists() else None
                atomic_text(txt, new)
                written.append((txt, original))
                sync_registered(project_id, path, new)
        except Exception:
            for txt, original in reversed(written):
                if original is None:
                    txt.unlink(missing_ok=True)
                else:
                    txt.write_bytes(original)
            raise
        return {"changed": len(plan), "unchanged": len(relatives) - len(plan), "total": len(relatives)}
