# 学習パネルの構造変更（2026-09-20）

ユーザー指示: 学習設定をサイドバーへ詰め込まない。既存プロジェクトタブはそのまま、パネル全体を切り替える。Rank/Alphaを同格で隣接させ、意味で分類する。

構造差分:
- frontend/src/features/studio/Workbench.tsx::ProjectCanvas: プロジェクト単位の dataset/training 表示状態と保存を追加。DatasetFilesとTrainingDockを別ステージで常時保持し、表示だけ切り替える。データセットに戻る/学習設定を開くでフォーカス移動。選択・下書き・設定は維持。
- DatasetFiles.tsx: 学習UIをサイドバーへ挿入していたtrainingPanel propと描画を削除。右サイドは画像/キャプション編集。
- TrainingDock.tsx: 設定とRun/Checkpoint/Previewの2領域へ分離。Rank+Alpha、Epoch+repeats/batch+save interval、resolution+LR/optimizer+schedulerを意味単位で配置。Rank/Alphaを詳細設定の開閉に依存させない。
- workbench.css: 950px超では設定/進捗の2列、以下は1列。新しいタブ列は作らない。短い画面遷移、reduced-motion対応。

検証: ビルド成功。panel-switch-result.jsonでRank/Alpha同一行、1250/900/640pxのstage clientWidth==scrollWidth、タグ下書き・設定の往復保持・再読込復元を確認。スクリーンショットtraining-full-panel.pngは隔離fixtureにおける実画面。審美はユーザー未確認。親パイプラインのGPU学習完走などをこのUI試験へ置き換えない。
独立評価はpanel-switch-independent.mdへ記録。検証中に親のRank=8保存が重なったことは評価者へ通知し、設定のリロード結果と分離する。

独立評価完了: keep.png選択と未保存本文、Rank32/Alpha8/保存名の変更が往復後も保持。同一行・同じ高さ・各約387px幅を確認。配布アプリでindex-0vVG4GzG.jsの読込を確認。独立評価者の未実施である再読込永続性は親の別実測で確認し、証拠を混同しない。
