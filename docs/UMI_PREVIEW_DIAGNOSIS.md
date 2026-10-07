# UMI preview quality diagnosis — 2026-09-21

User rejected visual similarity to the held-out reference. This is not a quality success. Existing Run7 remains running; do not stop it without answering the pending explicit question about epoch-boundary stop and weights-only continuation (optimizer state is not saved).

Observed:62/62 frozen captions include UMI;62 text caches have at most159Qwen tokens/188T5 tokens, below512. Checkpoints1,3,8 each840finite tensors; LoRA-up RMS grows0.0001763→0.0004886, so weights are changing. Epoch3 was falselyinvalid after an incomplete-file read; finalized file was revalidated and checkpoint19 is nowpreview_succeeded. No claim that weight growth means better style.

Reference822×1200. Frozen preview profile8 generates1024×1024,20steps,CFG5,seed42. Native helper hardcoded flow_shift1; installed ComfyUI supported_models.Anima uses shift3 and multiplier1. Existing native sampler defaults3 too. This is a comparison mismatch, not proof of the visual cause. Current gallery native images have much more high-saturation area than reference, but composition differs; HSV differences are not a style-quality score.

Future source correction: native_preview.py flow_shift3; preview_jobs.py reads the actual native prompt file to record shift rather than assuming1; TrainingDock next-run wording updated. Actual config-generation test produces3 while currentRun7 sample file remains1. Changes have NOT been deployed to running backend/trainer; do not claim current samples fixed. No training-rate or dataset changes made.

Pending comparison requires authorization to stop at next save or wait until training finishes. Proposed controlled comparisons: same saved checkpoint/prompt/seed, LoRA off/on and CFG3/5 in normal Comfy; reference-proportional dimensions separately from original1024square baseline. Do not run another GPU model concurrently with training. Game/training sharing permission does not cover extra simultaneous Comfy inference. No comparison generation has been submitted.
