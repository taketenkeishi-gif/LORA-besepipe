# Gallery density correction — 2026-09-21
User requested independent preview panel and less vertical stacking; then asked whether shared UI skill already required this.
Existing product-ui-craft chapter2 density/frequency and chapter3 responsive rules already covered the principle. This was an application omission, not absence of all guidance. Added shared DENSITY-01 to references/11-interaction-requirements.md and cross-reference from chapter3, with source thread01a0b7d6-ceaf-72e0-996b-9965958178f8; turn/user-message UNKNOWN. Requirements target repeated galleries/cards, not all pages; no fixed column count or no-scroll mandate. Skill quick_validate passed under Python UTF8 mode; initial default Windows cp932 read failed without changing skill content.

Production files (root C:/Users/Keishi/Portfolio/Generation/Training/LoRA-Studio-Next):
- frontend/src/features/studio/TrainingDock.tsx: new training-side wrapper, Timeline sibling to results instead of nested.
- TrainingTimeline.tsx: gallery grid; images prioritized, generated settings/path in initially collapsed details; existing image zoom/retry routes preserved; retry uses live display status.
- workbench.css: independent panel border; adaptive columns, details layout; image contain scaling. Panel can close. Existing files/models/previews unchanged.

Runtime work/advanced-training/gallery-result.json: production4 actual epoch images, 2 columns at1550x1100, each card288x308px,4images loaded; actual zoom succeeded, path expansion and gallery close/reopen succeeded. At390px panelclientWidth364 equals scrollWidth364. No before pixel-height measurement; do not invent a percentage reduction. All4 images were screenshot in one683px-high gallery region. Build/index-C4ubB3CF.js loaded by normal native reopen; backend/GPU jobs not restarted. Screenshot outputs/エポック比較ギャラリー.png. AESTHETIC_UNREVIEWED. Whole parent pipeline parity remains incomplete.
