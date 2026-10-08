# Archived 2026-10-09 (user approved: "動画からLoRA" rebuilt as one workspace)

Replaced by `src/features/studio/VideoLoraWorkspace.tsx` (① 動画 queue, ② キャラ, ③ LoRA) and `CharacterEditor.tsx`.

- CharacterSets.tsx — the first cross-video editor ("動画のキャラ"); its editing lives on in CharacterEditor.
- CharacterVideoDataset.tsx — one-video-at-a-time extraction dialog; replaced by the folder queue (backend services/video_lora_pipeline.py).
- CharacterLinkPanel.tsx / CropLinkReview.tsx — folder-level linking and image-level proposal review; replaced by ② and ③.

Backend endpoints they used (dataset_video_chars, dataset_video_link) are still served. To restore a screen, move the file back to
src/features/studio/ and import it again.
