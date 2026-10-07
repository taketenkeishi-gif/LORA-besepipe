# 学習中UIの旧版照合（2026-09-20）
今回の依頼は確認・照合。ソースと既存監査記録を確認。実学習の再実行・停止・キュー投入は行っていない。

|項目|旧UI|現行TrainingDock|
|---|---|---|
|進捗|Epoch/Step/残り時間/loss (Training.tsx2015-2024)|進捗%/状態/メッセージ/ログのみ|
|保存間隔|save_every_n_epochs入力 (Training.tsx2139)|設定UIなし。API既定1 (schemas.py79)|
|エポック別画像|PreviewTimeline (Training.tsx143-278,2818-2827)|タイムラインなし。Comfy手動プレビューは別経路|
|区切り停止|stop-at-epoch (Training.tsx2082)|即時停止だけ|
|途中再開|resume (Training.tsx2099)|再開ボタンなし|
|キャッシュ|latent/disc設定 (Training.tsx579-580)|start送信cache_latents=true、編集UIなし|

再開API routers/training.py1937-1944は既存LoRA重みからの継続。optimizer状態・epochカウンタの完全復元ではない。旧ラベルの再現だけで完全再開と説明しない。
待機判定: routers/training.py85は20000MiB固定。274-277は空き容量に加えforeign_processesが空である条件。空き容量だけ増えれば始まるとは説明できない。ユーザー前回画像にはChatGPT/CLIP STUDIO/ComfyUIが列挙されている。UIを開いているだけで待機条件が成立しない可能性を含むが、列挙対象の分類の妥当性・本番稼働コード一致は未検証。
結論: 学習中の透明性・途中再開は移植漏れ。親監査は未完了。GPUでの中断再開とエポック自動プレビュー連携は未検証。新旧同等とは報告しない。
