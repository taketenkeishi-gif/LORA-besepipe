# Advanced training and preview/progress correction — 2026-09-21

## Goal and authority
User rejected the sparse advanced panel and said the prior preview/progress UI was superior. Latest wording "9番" was queried; no reply observed, so old-version interpretation remains provisional. Keep the current overall workspace structure. UI craft/capability-linked-ui and installed kohya source comparison used. Conformance packet passed IMPLEMENTATION_READY for individual native settings and old preview/progress behavior; exhaustive application parity is NOT claimed.

## Changes
All paths below are relative to C:/Users/Keishi/Portfolio/Generation/Training/LoRA-Studio-Next.
- backend/app/training/advanced_catalog.json and advanced.py: 54 field definitions, Anima47 / SDXL45, typed bounds, model applicability, cache/augmentation/TE/bucket/attention/Huber/warmup constraints. Native TOML defaults replaced once, no duplicate keys. Arbitrary paths/command flags are not accepted through extra settings.
- backend/app/schemas.py TrainingStartIn.advanced, routers/training.py config-draft/start configuration recording and advanced-catalog/advanced-validate routes. Explicit advanced precision/checkpointing values override prior fixed start values. Gradient accumulation enters the initial approximate step count; native bucket-aware log remains actual authority.
- backend/app/training/backends/anima/backend.py and sdxl/backend.py prepare/writers carry advanced values to TOML; SDXL previously dropped existing precision/worker settings. Old Runs lacking advanced retain old writer behavior when resumed.
- frontend/src/features/studio/AdvancedTraining.tsx and TrainingDock.tsx: searchable grouped controls, applicability guidance, reset, immediate numeric-range message, next-Run draft persistence. Blank numeric editing uses shared NumericInput. Draft saved is distinguished from validated.
- PreviewConditionEditor.tsx: existing project-specific profile creation/freeze APIs edit prompt, negative, resolution, seed, steps, CFG, sampler/scheduler. Explicit next-Run scope; in-flight Run/jobs untouched. Preview model control stays independent of training model.
- TrainingDock.tsx: preview settings moved beside results/history. Overall percent, epoch segments, steps and ETA grouped. Completed training reports 学習完了.
- TrainingCharts.tsx/workbench.css: full-width Loss height240, latest values at rest, hover values, optional extra3charts. Actual advanced Run parameters exposed by runtime/metrics.py; Loss status explicitly log rounded value.
- TrainingTimeline.tsx: live pending/running counts override stale succeeded labels/details, avoiding contradictory status.
- runtime/preview_jobs.py preserves explicit CFG0 in frozen conditions.
- backend/desktop_server.py records21 import-time source fingerprints including new advanced files.

## Verified
- TypeScript + production build succeeded. Current production frontend index-DbXZHWNN.js.
- serialization-result.json: all47 Anima /45 SDXL fields emitted and matched actual TOML; incompatible configurations rejected.
- native-parser-result.json: installed native Anima234 / SDXL199 parser fields; generated68-key configs accepted. Actual native scheduler constructed and advanced on CPU. No new GPU training started for these UI changes.
- UI trial: blank numeric editing, accumulation4 save/reload, reset1; server rejects incompatible settings. Independent fresh-context evaluator confirmed accumulation/seed persistence, resets,390px no overflow, actionable cache conflicts. Numeric range/saved wording defects then repaired and review-regression.json verifies immediate error.
- Preview UI: prompt/seed0/CFG0 retained across save/reload. Actual ensure_jobs_for_checkpoint froze prompt/conditions/seed0/CFG0 in isolated DB transaction, then rollback; no job dispatched. Independent preview trial confirmed prompt/resolution768/seed123456/CFG4.5 and next-Run scope. Its stale-status findings repaired; review-regression.json verifies same pending fixture no longer says saved/succeeded and training-completed stage consistent.
- Graph actual300-step data: latest/hover/leave values verified;481px plotting width, three extra charts expand with no horizontal overflow. Screenshot and prior independent chart review retained.
- Production idle checked before narrowly identified internal server restart; no external process or GPU model unload. Native app normal close/reopen, backend21 hashes and loaded frontend verified by deployment.json/native status.

## Limits and parent task
AESTHETIC_UNREVIEWED. No universal training-combination or GPU-quality claim. Native state/Optimizer resume, alternate LoRA algorithms, block-wise rank/LR, compile/FP8 coverage and multi-prompt automatic previews are not restored to full Kohya parity. Current preview condition editor is Anima/next-Run, not mutation of already-running Run. The original pipeline-wide audit remains incomplete; this result closes the settings/UI correction scope only. Named native EXE source is not proven identical to installed Gradio source. User's exact "9番" reference remains unanswered.

## Evidence
Thread work/advanced-training/: conformance.json, reference-audit.md, reference-inventory.json, serialization-result.json, native-parser-result.json, ui-result.json, preview-ui-result.json, preview-freeze-result.json, review-regression.json, deployment.json, user-trial/report.md, preview-trial/findings.md. Graph evidence work/graph-preview/charts-readable-result.json. No project git repository; did not commit into ancestor home repository.
