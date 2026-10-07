# 学習グラフ・生成モデル分離・数値入力台帳（2026-09-21）

## 利用できる変更
- 進捗率、Loss/平均Loss、学習率、1stepの記録間隔をグラフ表示。ホバーで時点の値を表示。現在の入力欄と別に、そのRunの保存済み実行設定を表示。
- 自動プレビュー用のAnima系モデルを学習ベースと別に選択。Steps/CFGは比較条件から継承し、指定時だけ上書き。設定は次回Runへ固定され、既存画像を勝手に生成し直さない。
- 手動ComfyUIプレビューの入力値/モデル/条件をプロジェクト単位で保存し、閉じて開き直した後に復元。
- 数値入力は編集中の文字列を保持して、blur/Enterで確定。空欄を0に押し戻さない。
- 共通UI要求台帳 product-ui-craft/references/11-interaction-requirements.md の NUMBER-01 に原因、適用範囲、新規アプリの実入力テスト、今回の根拠を登録。将来の全実装での遵守保証ではない。

## 実測と実生成
Wasabi本番Run4のTensorBoard実ログからLoss/current300点、Loss/average300点、lr/unet300点、記録間隔299点を取得。進捗の分母は実ログのtotalを優先。学習率を設定値から架空に描いていない。モデル読込時間はグラフの経過秒から除き、step記録間隔に保存などの待ち時間が含まれることを表示。
独立評価が4グラフの時点値、実行設定の閲覧、学習ベースを維持した生成用モデルの変更とreload保持を確認。1550/900/640pxで横溢れなし、定期読込でCanvas/ホバー値が破棄されないことを別シナリオで測定。
Anima-base-v1.0で学習済みの実WASABI LoRAを、fnMixAnimaTurbo_v31を生成ベースにして実生成。native Comfy prompt d2f5ab8e-270b-4f3d-ada5-4435866a2687、Seed0、8steps、CFG1、512px。学習用baseの変更なし。既存チェックポイントへのhardlinkを隔離fixtureで使い、元の重みは書き換えていない。生成画像をチャットへ提示済み。
既存EpochのStep0は記録欠落で、実LoRA metadataのss_stepsは75/150/225/300。API表示はこの実記録を参照するよう修復。記録が無いものは未記録とし、架空の0を表示しない。
数値編集の独立評価で全消去→768、CFG3.5、Seed0、学習直接入力を確認。追加試験でプレビュー再オープン時の768/3.5/0保持、CFG0とSeed0を生成条件へ保持するユニット試験を確認。

## 構造差分
frontend/src/features/studio/NumericInput.tsx および6利用画面: 文字列draftと確定numberの分離。
TrainingCharts.tsx,workbench.css: [uPlot](https://github.com/leeoniya/uPlot)1.6.32を使用した実系列描画・時点値・レスポンシブ配置。LICENSEをpublic/distへ同梱。
TrainingDock.tsx,ModelPicker.tsx,TrainingTimeline.tsx,types/index.ts: 独立生成モデル/条件、実行設定、生成モデル表示、実Step表示。
ComfyPreviewDialog.tsx: プロジェクト別draft保存と復元、Seed0/CFG0保持。
backend/app/training/runtime/metrics.py: TensorBoard2.19.0の実scalar取得。ログfallbackは区別。再開前ログを混ぜず最新セッションを使用。checkpoint metadataのStep参照をcache。
routers/training.py,schemas.py: metrics API、独立モデル/Steps/CFGの保存・同系統/実在検査。
training/runtime/preview_jobs.py,preview/providers/comfyui/provider.py,routers/previews.py: Run固定の生成条件と生成モデルを保持し、native graphへ渡す。元の比較Profileは書き換えない。
backend/desktop_server.py: 追加ソースの読込hash。

## 証拠・限界
証拠: work/graph-preview/metrics-real-run4.json,chart-preview-ui.json,charts-final-runtime.json,independent-ui.md,preview-draft-result.json,zero-conditions-result.json,turbo-job-before.json,turbo-preview-result.json,research.json。数値検証はwork/preprocess-audit/numeric-input-result.json,numeric-independent.md。
独立評価は新規学習・GPU操作をせず、実生成は親が別に実施。今回新しい学習を開始していない。モデル品質/審美はユーザー未承認。自動プレビュー用の別モデル設定はAnimaに限定して表示。過去の全Runや手動生成の全履歴一覧はこの変更の対象外。
