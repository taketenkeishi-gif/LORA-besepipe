"""JSON-only desktop adapter; fixed dataset operations, no arbitrary command execution."""
import json,sys,os
from pathlib import Path
if os.environ.get('LORA_STUDIO_TEST_DB'):
    from app import db
    db.DB_PATH=Path(os.environ['LORA_STUDIO_TEST_DB'])
from app.dataset_desktop import open_folder,preview_normalize,apply_normalize,import_web_image
if os.environ.get('LORA_STUDIO_TEST_MANAGED_ROOT'):
    from app import dataset_desktop
    dataset_desktop.ROOT=Path(os.environ['LORA_STUDIO_TEST_MANAGED_ROOT'])
try:
    operation=sys.argv[1];payload=json.loads(sys.stdin.read())
    if operation=='open-folder':result=open_folder(payload['folder'])
    elif operation=='normalize-preview':result=preview_normalize(payload)
    elif operation=='normalize-apply':result=apply_normalize(payload)
    elif operation=='web-image':result=import_web_image(payload)
    elif operation.startswith('naming-'):
        from app import dataset_naming as naming
        actions={'naming-preview':naming.preview,'naming-apply':naming.apply,'naming-undo':naming.undo}
        if operation not in actions:raise ValueError('Unknown naming operation')
        result=actions[operation](payload)
    elif operation.startswith('preprocess-'):
        from app import dataset_preprocess as preprocessing
        actions={'preprocess-capabilities':preprocessing.capabilities,'preprocess-start':preprocessing.start,'preprocess-status':preprocessing.status,'preprocess-cancel':preprocessing.cancel,'preprocess-worker':preprocessing.run}
        if operation not in actions:raise ValueError('Unknown preprocessing operation')
        result=actions[operation](payload)
    elif operation.startswith('comfy-'):
        from app import desktop_comfy as comfy
        actions={'comfy-discover':comfy.discover,'comfy-inspect':comfy.inspect,'comfy-bind':comfy.bind,'comfy-workflow':comfy.workflow,'comfy-checkpoints':comfy.checkpoints,'comfy-preview-start':comfy.preview_start,'comfy-preview-status':comfy.preview_status,'comfy-worker':comfy.preview_run,'comfy-reveal':comfy.reveal}
        if operation not in actions:raise ValueError('Unknown ComfyUI operation')
        result=actions[operation](payload)
    elif operation.startswith('tag-'):
        from app import dataset_tagging as tagging
        actions={'tag-models':lambda value:tagging.models(),'tag-start':tagging.start,'tag-status':tagging.status,'tag-stop':tagging.stop,'tag-worker':tagging.run}
        if operation not in actions:raise ValueError('Unknown tagging operation')
        result=actions[operation](payload)
    else:raise ValueError('Unknown desktop dataset operation')
    print(json.dumps({'ok':True,'result':result},ensure_ascii=False))
except Exception as exc:
    print(json.dumps({'ok':False,'error':str(exc)},ensure_ascii=False));sys.exit(1)
