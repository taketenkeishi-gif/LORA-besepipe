# 学習中UIの復旧 — 2026-09-20

## 反映した機能
- TrainingDock.tsx: Epoch/Step/残り時間/Loss、工程、日本語状態、保存済み重みのパス、次の保存エポックで停止、保存済みLoRAから継続（制限説明と取消）、保存間隔・キャッシュ2項目。
- TrainingTimeline.tsx: 現在Runに属するCheckpointだけの一覧、エポック/Step/ファイルパス、プレビュー状態と件数、実画像、Epoch番号付き拡大、既存失敗Preview再生成APIへの接続。
- 次回設定と実行済みRunを説明上で分離。

## 裏側の修正
- backend/app/routers/training.py::stop_at_epoch: 停止予約時の最終Checkpoint IDと要求時刻を永続記録。
- backend/app/training/runtime/monitor.py::epoch_stop_checkpoint,tail_monitor: 予約後の新しい保存ファイルを検出。サイズ/更新時刻が安定しsafetensors検証を通った後、当該Runの所有プロセスだけ停止しpausedへ遷移。以前保存されたファイル、未完ファイル、取消後は停止しない。Preview状態は検証で上書きしない。
- backend/app/training/backends/anima/backend.py,sdxl/backend.py: キャッシュ設定を実行設定へ伝搬。
- backend/desktop_server.py: 監視と両backendの読込時ハッシュを公開。
- desktop/main.cjs: 遅延読込画像のdecodeと撮影を期限付きにし、古い起動証拠のまま停止する不具合を修正。

## 実施した検証
- ビルド成功。配布アプリがindex-B9EGRkO-.jsを読込。配布内部サーバーの起動時11ファイルhashと実ディスク一致。
- 旧ファイル、部分ファイル、変化中ファイルを停止条件から除外し、新しい安定ファイルだけ受理する実DB/実ファイル試験。
- 本物のtail_monitorと所有CPUプロセスを使い、保存後約7.5秒までにpaused・プロセス終了・保存ファイル保持を確認。これは制御試験でありGPU学習成功ではない。
- Anima/SDXL TOML出力で保存間隔3とcache on/offの実伝搬を確認。
- 保存間隔/キャッシュのUI変更・再読込保持、過去の実学習出力に対応する生成画像512x512の読込・拡大、再開説明と取消を実画面で確認。
- 新しい評価者が同じ利用目的をUIから独立検証。指摘された状態名・次回設定の区別・拡大Epoch番号を修正後に実画面再検証。

## 証拠と境界
証拠は C:/Users/Keishi/Documents/Codex/2026-09-19/lo/work/preprocess-audit/ の transparency-ui-result.json, transparency-independent.md, transparency-final-ui.json, epoch-stop-guards.json, monitor-control-result.json, training-config-result.json, production-backend-deployment.json, transparency-production-desktop.json。
表示試験のRun行は隔離UI fixtureで、既存の実LoRAと実生成画像を参照。新規GPU学習・中断→再開のGPU通し試験・失敗Previewの新規再生成を成功したとは主張しない。
再開は既存重みからの継続でありoptimizer内部状態/Epochカウンタの完全復元ではない。
GPU1固定20000MiB/外部process判定は今回変更していない。審美はAESTHETIC_UNREVIEWED。全パイプラインの品質承認とは別。
