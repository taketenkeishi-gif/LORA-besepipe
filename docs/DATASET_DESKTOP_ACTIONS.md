# Dataset desktop actions — 2026-09-20

## User authority and scope
User clarified that opening from Explorer means the app's 「フォルダを開く」 native folder picker, not an Explorer context-menu extension. Preserve the existing dataset/training screen and project tabs. Companion requests: public web image drop and intuitive image-size normalization. No GPU processing, model acquisition, or original replacement.

## Goal fidelity
- USER_UNCERTAINTY: Can an existing dataset be selected directly and edited in place, while the visible Explorer action opens its real location? Can web images be saved into it and sizes normalized visibly?
- REQUIRED_CAUSAL_CAPABILITY: Actual Windows folder dialog → fixed desktop IPC → persisted project; external drag payload → network image decode/write; selected files → Pillow resize/pad → separate real files plus captions.
- DISCRIMINATING_OUTPUT: Existing folder opens without copying images; same folder reuses its tab; normalized files have measured dimensions and unchanged originals; downloaded image appears after reload.
- EXCLUDED_SURROGATES: Mock picker responses, image placeholders, build-only success, GPU training or whole-pipeline completion claims.

## Routes / ownership
| Control | Route | Persistence / verification |
|---|---|---|
| フォルダを開く | Workbench.openFolder → preload → native showOpenDialog → desktop_actions open-folder | workspace.db project refers to chosen dataset directory; React project tab persists |
| エクスプローラーで開く | existing dataset-files reveal API | existing system folder view; label made explicit |
| Web image drop | active useDatasetDrag root → importWebImage IPC → HTTP image → PNG → rescan | writes only chosen dataset folder; no browser cookies; public image URLs only |
| サイズを揃える | selected saved images → NormalizeDialog → previewNormalize → applyNormalize | source hash/TXT revision guard, separate normalized-* folder, byte-copy TXT |

Only native desktop exposes new controls; a plain browser lacks the OS bridge. Folder selection cancellation returns null and creates no project. Names are unique case-insensitively during flattened normalization. Preview changes invalidate application. No crop or distortion; pad uses white background, longest mode preserves aspect. Enlargement is optional and defaults off; without enlargement small images can retain their shorter long-edge size. This is ordinary CPU resizing, not learned super-resolution or automatic subject centering.

## Source changes
- backend/app/dataset_desktop.py: project-directory association, normalization preview/apply, image URL import.
- backend/desktop_actions.py: fixed JSON operation adapter, isolated test DB overrides.
- desktop/preload.cjs and desktop/main.cjs: context-isolated fixed bridge, native dialog, trusted local origin, bounded subprocess, test runtime isolation, normalized root identity.
- scripts/Start-Desktop.ps1: distribute preload with main/package; retained UTF-8 BOM.
- frontend/src/features/studio/{Workbench.tsx,DatasetFiles.tsx,useDatasetDrag.ts,desktopBridge.ts,NormalizeDialog.tsx,dataset-files.css}: folder entry, URL drop, normalization dialog and preview.

## Evidence
Thread work/dataset-entry: real Electron with isolated DB on5192; native dialog inspected by ownerPID/class/label and confirmed with Windows BM_CLICK (not physical mouse). Actual UI →512x512 copies of two images, captions displayed, public python.org logo drop downloaded580x164, three images after reload. ui-result.json, normalize-preview.png, result.png.
Thread work/dataset-actions-review: independent domain test found flattened output filename collision and TXT newline conversion. Both repaired: collision candidate while-loop and byte-copy TXT. Repeated executable test: all checks true, originals same hashes, all four pad/longest+upscale combinations, stale image/TXT rejection, private URL refusal, public image saved/reopened, JSON adapter. Original failure retained as receipt-before.json; updated receipt.json records regression.
Frontend production build passed. Native production window reopened normally with backend left running; desktop-status records desktopBridge=true, folderOpenControl=true, images4/4 loaded, scrollWidth=clientWidth794. Previous backend audit fixes remain NOT production verified. The CLI actions load current code independently; this does not certify old server route code.

## Limits / continuation
Browser-drop tested with real DataTransfer events through production handlers and actual network/files, not physical dragging from every browser/site. Authenticated, anti-hotlink and blob/data images are unsupported; failures must remain visible. Native user-perspective trial recorded separately before operational reporting. User aesthetic status AESTHETIC_UNREVIEWED. Parent pipeline audit remains docs/PARENT_AUDIT.md and is not complete.

## Final user-trial corrections
Independent native UI trial confirmed768x768 two-image copies and byte-identical TXT, reopening originals, public image import and cancel on failure. Found same-root reopen stuck in previous subfolder and web import undiscoverable: repaired root-request signal (ordinary tab switching still retains location), added Web画像 button/dialog supporting URL paste and explaining drag route. Found raw remote/decoder error plus stale success: translated non-image/network failures and cleared previous notice. Later actual UI regression error-regression.json confirms Japanese non-image URL instruction, no stale success, unchanged file count after failed import, cancel returned. No physical cross-browser dragging or aesthetic approval claimed.

