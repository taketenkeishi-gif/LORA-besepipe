# GPU判定・トリガー修復と実学習結果（2026-09-20）

## ユーザーの結果
Wasabiの既存予約 Run4 / Project4 を重複作成せず引き継いだ。75画像、768px、Rank32、Alpha16、batch1、4epochs、300stepsがcompleted。学習部分の最終進捗ログは05:41、1.14s/it。4つのcheckpointと各epochの4previewすべて実ファイル・APIで確認。最終ファイルは C:/Users/Keishi/Portfolio/Generation/Training/LoRA-Studio-Next/.runtime/runs/4/output/WASABI_ANIMA.safetensors（91,861,240bytes）。840tensorがfinite、280個のlora_up tensorが非ゼロ。
ユーザーの学習設定（epochs/rank/alpha/resolution/batch/optimizer/確定Dataset）を変更せず、待機判定と実行方式を修復。実行GPUはUUID GPU-325e2e52-6609-0917-c853-e3d2019cdebc、RTX3090Ti。外部アプリの停止/メモリ解除はしていない。

## VRAMと速度
0%はGPUの計算負荷であり、VRAM使用量ではない。常駐プロセス一覧だけの一律拒否をやめ、複数負荷サンプル、実行中trainerの識別、ComfyUI native queueを確認。未知/高負荷/実行中学習/生成キューは引き続き待機する。
起動時のNVML/nvidia-smi実空きとPyTorch空きの小さい値を使用。そこから2GiBを残し、総量75%との小さい方をPyTorch割当上限にする。上限量を予約せず実使用分が増える。WDDMではPyTorchの空きがシステム値より大きかったため片方だけを使わない。
省VRAM26層退避とバランス12層退避の初回確認容量8GiB、速度優先GPU常駐は12GiB。これは全条件の必要量保証ではなく、上限付き初回確認の下限。高解像度/大batch等のOOMが絶対に起こらない保証はしない。
1024x1024/Rank32/Alpha16/batch1/8stepsの比較（AdamW）:
- 26層退避: 約3.2秒/step、PyTorch最大予約2.95GB。
- 12層退避: 約2.4秒/step、最大予約4.89GB。
- GPU常駐＋cached dataset workers0: 約2.1秒/step、最大予約6.61GB。
Wasabi本番（AdamW8bit・768px）は約1.14秒/step、PyTorch最大予約5.67GB。異なる解像度/optimizer/データ条件の値を同一条件比較とは呼ばない。学習ロード時間・プレビュー時間はstep時間と別。

## トリガー
既存Wasabi全75captionの先頭WASABIと、Concept管理未登録の食い違いを確認。元captionを変更せずConceptにWASABIを登録。75/75一致。TrainingTriggerで既存値/候補/一致枚数を表示、その場で登録・修正。登録だけでTXTを変更しないこと、未一致時はタグと確定素材を更新することを表示。不正値拒否・旧値保持・再読込・VRAM選択の保持を実UIで検証。

## 構造差分
backend/app/training/runtime/anima_admission.py: preflight/queue/final spawnの共通判定。
backend/app/routers/training.py: memory modeのRun/下書き保存、trigger GET/PUT、既存queue再評価API。新Job作成しない再評価でRun4を起動。
backend/app/schemas.py: memory mode列挙。
backend/app/training/backends/anima/backend.py: block swap、UUID指定、上限付きentry、実空き小側採用、cached dataset workers0。既存Run rootを使用。
backend/app/training/backends/shared/helpers.py: Run rootの一元化（隔離rootを無視していた不具合修復）。
backend/app/training/runtime/preview_jobs.py: Animaを既存bound ComfyUI native queueへ。pending->runningの条件付きUPDATEで二重claim防止。
backend/app/preview/providers/comfyui/provider.py: 学習時と同じbase、LoRA hardlink、websocket出力。共有Comfy input/outputへ画像を保存しない。
backend/app/desktop_comfy.py: sampler受け渡し。
backend/desktop_server.py:17ファイルの読込hash。
frontend/src/features/studio/TrainingTrigger.tsx,TrainingDock.tsx,workbench.css: trigger直接編集、VRAM選択、設定読込中の入力保護、実学習中の工程表示。

## 検証と失敗履歴
実画面から小規模Run4/5/6/7/8（隔離DB）を実行。Run5のbalanced選択がRun configへ落ちていた欠陥を発見し、保存キーと実行dictへ追加。そのRun5は26層の測定として扱う。Run7の初回64秒は4読込worker起動を含み、2秒の定常部分だけを全体平均としない。workers0のRun8で8step全体を再確認。
初回テストは別の最終spawn20GB判定で開始前に失敗。次の試行はWDDM空き不一致を発見して所有テストだけ停止。全profileでシステム側空きを使う上限へ修復。これらは成功試験に数えない。
隔離Run2準備時、共有helperが隔離rootを無視し旧QA Run2ディレクトリへテストconfig/log/train_dataを出力した。既存checkpointを生成/上書きする段階より前に停止。元QA config/logの事前内容はUNKNOWN。helper修復後の実試験は全てwork/preprocess-audit/live-training-runsに分離。実WasabiのRun4とは別。
Run5のプレビュー重複送信を検出（2回目は白画像）。atomic claim修正後Run6/7/8と本番各epochは各1 native promptで成功。失敗証拠は削除せず保持。
本番完了直前に観測scriptのstatus取得が15秒timeoutしたが、再取得で300/300完走と4preview成功を確認。timeoutの内部原因はUNKNOWN。学習を再作成/再開始していない。

証拠: C:/Users/Keishi/Documents/Codex/2026-09-19/lo/work/preprocess-audit/ の speed-comparison.json,wasabi-completed.json,wasabi-artifact-verified.json,trigger-repair.json,trigger-vram-independent.md,gpu-trigger-final-ui.json,active-trainer-admission.json,gpu-trigger-final-deployment.json。独立評価が登録/修正/復元/再読込を確認。モデル品質の人間承認はAESTHETIC_UNREVIEWED。GPU中断再開や全モデル系統の保証とは区別する。
