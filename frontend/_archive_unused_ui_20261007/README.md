# Archived unused UI (2026-10-07)

These files were not reachable from `src/main.tsx` through any import (checked with a script that follows every import/dynamic import), i.e. they were leftovers of older UIs and were never rendered by the running app. They were MOVED here (not deleted) with the same relative paths.

Restore one file: move it back to the same path under `frontend/` (for example `_archive_unused_ui_20261007/src/components/Dataset.tsx` -> `src/components/Dataset.tsx`).
This folder is outside `src`, so it is neither type-checked nor bundled.

Files:
- src/app/router.ts
- src/components/BasepipePanel.tsx
- src/components/Compare.tsx
- src/components/Dashboard.tsx
- src/components/Dataset.tsx
- src/components/Home.tsx
- src/components/Integrations.tsx
- src/components/Layout.tsx
- src/components/Library.tsx
- src/components/Overview.tsx
- src/components/Projects.tsx
- src/components/Runs.tsx
- src/components/Training.tsx
- src/components/TrainPage.tsx
- src/features/reproduction/ReproductionPages.tsx
- src/features/studio/CharacterDatasetFactory.tsx
- src/features/studio/ComfyPreviewDialog.tsx
- src/features/studio/EvaluateWorkspace.tsx
- src/features/studio/LibraryWorkspace.tsx
- src/features/studio/NormalizeDialog.tsx
- src/features/studio/PrepareWorkspace.tsx
- src/features/studio/ProjectHome.tsx
- src/features/studio/StudioShell.tsx
- src/features/studio/TrainWorkspace.tsx
- src/lib/utils.ts
- src/styles.css
