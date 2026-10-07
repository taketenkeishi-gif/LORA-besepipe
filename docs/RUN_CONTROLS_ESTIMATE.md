# Run controls and draft estimates — 2026-09-21
User reported omitted old-UI time/step estimation, cache deletion and continuation controls.

Implemented structure (all paths rooted at this project):
- backend/app/training/draft_estimate.py and routers/training.py POST /estimate-draft: sealed snapshot count, repeats/batch/gradient accumulation/epochs determine estimated optimization steps. Completed same-family/GPU runs with compatible optimizer/offload/precision/cache policy provide median recorded step intervals. Resolution/rank/batch/accumulation differences use explicitly approximate scaling. No compatible measurement returns time unavailable instead of fake certainty. Excludes load/cache/preview, bucket remainder caveat shown. Does not use active-run count for future draft.
- TrainingDock.tsx: debounced estimates, distinct check/start controls. Start performs the same checks before submission; current GPU waiting policy retained.
- RunActions.tsx: per-Run saved-epoch selection sets next draft resume_checkpoint_id; current base/rank/alpha compatibility verified through continuation-check and again in start. New Run receives resume_from_checkpoint, which existing native writers serialize as network_weights. Original completed Run is not reset. Legacy paused resume route still exists and is explicitly weights-only; no optimizer-state restoration claimed.
- schemas.py, draft whitelist/config recording carry resume_checkpoint_id. Existing checkpoints remain project-owned and file existence checked.
- run_cleanup.py plus GET /runs/{id}/cleanup and DELETE /runs/{id}/cleanup/{training|preview}: list exact disposable files within resolved owned Run root. Training cache includes .npz under staged train_data plus cache/. Preview deletion targets recorded PNG/JPEG/WebP only, rejects outside-root/symlink paths, refuses active/queued training or preview. No logs/counters/LoRA/input reset. Preview records marked removed/cancelled; no automatic regeneration.
- RunActions dialog shows category/count/bytes and requires explicit category confirmation. apiDelete retains actionable server error detail.
- desktop_server.py import-time23 fingerprints include new backend modules.

Verification:
- Build passed; actual deployed bundle recorded in desktop-status.json.
- work/advanced-training/run-tools-readonly.json: real production75-image snapshot -> 4epochs300steps333sec /6epochs450steps500sec, Run4 speed median1.112 sec over299 intervals. Historical adjusted estimate, not guarantee or new benchmark.
- Same record: valid real checkpoint accepted, mismatching Rank rejected. continuation-config-result.json: actual native TOML network_weights points to chosen saved checkpoint and max_train_epochs6. New GPU continuation execution was NOT performed.
- cleanup-test-result.json: cloned isolated DB and disposable files; cache2files and preview1file removed separately. SHA256 of source image/caption, weights and log unchanged; queued Run rejected. No user's real cache or images deleted.
- run-tools-ui.json: isolated actual UI estimates visible, independent check/start buttons, completed-Run continuation selection/clear. No training started.
- production-run-tools.json: live75images450step estimate and actual cache dialog loaded; deletion cancelled. Real production UI and backend refreshed after confirming no active Runs/jobs; external GPU/apps untouched.

Source comparison: old Training.tsx resetTraining and training.py estimate/resume/cache/reset inspected. Old reset terminates process/clears logs/counters; intentionally not wired as harmless cache deletion. Existing resume was weights-only, not native state resume. Conformance packet individual-route gate passes; no exhaustive Kohya parity claim.
AESTHETIC_UNREVIEWED. Complete optimizer state resume and fresh GPU run with new continuation controls remain unverified; original full pipeline audit remains unfinished.
