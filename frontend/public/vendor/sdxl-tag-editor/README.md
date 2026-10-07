Copied unmodified from ComfyUI custom node SDXL_tag (web/js/tag-editor, 2026-10-07). Source of truth stays in custom_nodes/SDXL_tag; re-copy to update.
Host contract: new TagEditor(rootElement, onChange(json), getGraphNodes, getHostNode); getJSON()/setJSON(json). Server routes /unified_tag_editor/* are proxied to ComfyUI (backend/app/routers/comfy_tag_proxy.py).
Only local change: tagger.js BASE = /api/unified_tag_editor.
