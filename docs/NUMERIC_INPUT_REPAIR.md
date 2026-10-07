# 数値入力の修復と共通要求登録
2026-09-21。NumericInput.tsxを追加し、ComfyPreviewDialog/TrainingDock/AutoTagDialog/PreprocessDialog/BatchNamingDialog/NormalizeDialogの手入力を文字列draft→blur/Enterで数値確定へ変更。空欄は編集中に保持、空欄確定は直前値へ復帰、不要な先頭0は確定時正規化。入力中Number変換を除去。
実UI検証 numeric-input-result.json：全消去→768、0768正規化、空欄確定の復帰、CFG4.5、Seed0、学習解像度768の実設定保存。独立検証numeric-independent.mdも高さ/小数/0/学習直接入力を確認。生成/学習開始なし。ビルドindex-KNr2ng_k.js。共有登録先：product-ui-craft/references/11-interaction-requirements.md NUMBER-01。
独立評価でプレビュー再オープン時に設定が戻ることも観測。別の未解決事項としてグラフ・生成用モデル分離の追加依頼に引き継ぐ。

更新：プレビュー再オープン保持の不成立はComfyPreviewDialogのproject別draft保存で修復し、work/graph-preview/preview-draft-result.jsonで768/3.5/0の復元を実測。NUMBER-01は共通要求表の1行として検査済み。
