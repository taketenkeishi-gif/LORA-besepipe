# 2026-09-20 UI correction checkpoint

User goal: fixed sidebar with chip display zoom; use SDXL_tag editor behavior as reference; restore training presets, explain workflow JSON, make SDXL preview selection reachable, diagnose disabled preflight.

Modified: frontend/src/features/studio/CaptionEditor.tsx (display preference 60–160%, persistent global zoom/reset; selected-position paste); editor-components.css (scaled layout/hit boxes); TrainingDock.tsx (old option lists, numeric custom escape, clickable missing-input diagnosis, family switch clears incompatible base path, adaptive optimizer LR display); ComfyPreviewDialog.tsx (local preview family, workflow JSON purpose, visible initial loading/errors).
Reference: custom_nodes/SDXL_tag/web/js/tag-editor/paste-position.mjs:1-8; styles.js:119-156; frontend/src/components/Training.tsx:2135-2180. No claim that SDXL_tag already had chip zoom or that full node parity is achieved.
Goal fidelity: measure real chip layout/selection/content/persistence and actual selected config persistence, family catalog, preflight user path. Build alone and fixed mock values excluded.
Evidence root: C:/Users/Keishi/Documents/Codex/2026-09-19/lo/work/preprocess-audit/
- tag-reference.json and training-reference.json IMPLEMENTATION_READY gate passed.
- zoom-result.json: 240px content width equals scrollWidth at 60%/160%; unchanged text and selection; persisted after reload and reset100.
- zoom-independent.md: fresh UI trial selected-position paste/undo, zoom restore, image switching verified, aesthetics unreviewed.
- scope-result.json: real CPU inference results for three fixture images saved to three TXT. Prior single vs all scope scenario resumed after hidden-window screenshot timeout.
- training-ui-result.json: Rank32, Alpha8, resolution768, batch2, scheduler cosine persist across reload; actual Comfy catalog contains WAI Illustrious/HS and SDXL after in-dialog family switch. Missing inputs diagnosed on click. No training job started in this scenario.

UNVERIFIED: production already-open renderer update; complete training start -> epoch preview -> evaluation -> export; comprehensive preprocessing runtime; latest small loading/adaptive changes pending follow-up. Parent audit remains unfinished in PARENT_AUDIT.md. New dialog JSON means ComfyUI image-generation workflow, not JSONL, training-config import or model catalog.
Deployment preserves user unsaved work; do not force-close vetoed windows or kill backend. AESTHETIC_UNREVIEWED.


Final checkpoint: fresh independent training UI trial in training-independent.md confirmed selectors, missing-input diagnosis, WAI Illustrious preview model selection, and workflow-purpose copy, without launching training/generation.
Actual fixture selection of 3 images -> sealed dataset -> real Anima preflight was executed. With production runtime-path settings copied only into isolated test DB, missing engine/Python/encoder/VAE errors disappeared; GPU VRAM/occupancy diagnostics and queue-start option were returned. No new GPU job submitted. preflight-actual.json records actual output. Fixture concept is unset; do not treat fixture warnings as production-specific faults.
Build index-CFNpa-Et.js passed. Production native app PID48816 was closed through normal CloseMainWindow, exited without force, then Start-Desktop.ps1 launched PID48060. Focus-free display1 capture production-window.png observed new preset dropdowns and active preflight control in the actual app. Startup desktop-status.json remains stale: existing startup observer awaits decoding all lazy images and can hang. This telemetry defect remains uncorrected; do not reuse its prior PID/script claim as current evidence.
User data: no production captions/images changed by tests. Production preview worker/server not stopped. AESTHETIC_UNREVIEWED. Full training E2E and expanded preprocessing still outstanding, not claimed complete.
