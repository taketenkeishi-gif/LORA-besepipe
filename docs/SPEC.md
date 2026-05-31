\# LoRA Basepipe V2 SPEC



\# Part 1 / 3



\# Philosophy / Scope / Core Architecture



\---



\# 1. Project Definition



LoRA Basepipe は LoRA学習ツールではない。



LoRA制作工程全体を最適化し、

再利用可能な知識・データセット・学習設定・LoRA資産を蓄積するための研究基盤である。



本プロジェクトの目的は

単発のLoRAを完成させることではなく、



\* 良質なデータセットを高速に作ること

\* 良質なLoRAを安定して再現すること

\* 制作ノウハウを資産化すること

\* 将来のAI自動化に備えること



である。



\---



\# 2. Project Goal



ユーザーが目指す最終状態



現在



URL収集

↓

画像整理

↓

タグ整理

↓

前処理

↓

学習設定

↓

学習

↓

比較

↓

保存



に大量の手作業が存在する。



LoRA Basepipe V2では



URL貼り付け

↓

候補確認

↓

データセット確認

↓

学習開始



程度の操作量でLoRA制作を進行可能にする。



\---



\# 3. Core Philosophy



\---



\## 3.1 Dataset First



LoRA品質は学習工程ではなく

データセット工程で決まる。



本システムは

Training Firstではなく



Dataset First



を採用する。



優先順位



Dataset

↓

Caption

↓

Refinery

↓

Training

↓

Evaluation

↓

Library



\---



\## 3.2 Manual First



V2では人間が主導する。



AIは補助する。



最終決定権は常にユーザーが持つ。



AIが勝手に



\* 削除

\* 学習

\* 上書き



を行ってはならない。



AIは



提案



警告



分析



のみを行う。



\---



\## 3.3 Future AI Ready



将来的にCloudAI OSとの連携を想定する。



ただしV2では実装しない。



代わりに



すべての主要操作を



API



Task



Job



として実装可能な構造にする。



\---



\## 3.4 Asset First



LoRAは成果物ではない。



LoRAは資産である。



保存対象



\* Dataset

\* Caption

\* Training Config

\* Preview

\* Evaluation

\* LoRA



すべてを資産として管理する。



\---



\# 4. User Workflow



\---



\## Step 1



Project Creation



ユーザーは



\* Character

\* Style

\* Hybrid



のいずれかを選択する。



\---



\## Step 2



Dataset Collection



画像を収集する。



入力可能ソース



\* Pixiv

\* Danbooru

\* Gelbooru

\* URL

\* Folder

\* Drag \& Drop



\---



\## Step 3



Dataset Engineering



データセットを分析する。



ここがシステムの中心となる。



\---



\## Step 4



Training



Kohyaを使用して学習する。



\---



\## Step 5



Evaluation



生成結果を比較する。



\---



\## Step 6



Library



成果物を資産化する。



\---



\# 5. Core Modules



V2では以下を中核モジュールとする。



\---



\## Dataset Module



役割



画像収集



画像管理



画像分析



\---



\## Caption Module



役割



タグ生成



タグ分析



タグ統合



\---



\## Refinery Module



役割



画像前処理



画像補正



不要情報除去



\---



\## Training Module



役割



Kohya制御



進捗監視



ライブプレビュー



\---



\## Evaluation Module



役割



LoRA品質評価



比較表示



\---



\## Library Module



役割



LoRA資産管理



\---



\# 6. OSS First Policy



LoRA Basepipeは

既存OSSを最大限利用する。



再実装を禁止する。



\---



利用候補



Training



\* Kohya SS



\---



Generation



\* ComfyUI



\---



Caption



\* WD14

\* JoyCaption

\* Florence2

\* Qwen-VL

\* DeepSeek-VL



\---



Image Similarity



\* CLIP

\* SigLIP



\---



Face Analysis



\* InsightFace



\---



Duplicate Detection



\* pHash



\---



Image Editing



\* ComfyUI Workflow



\---



\# 7. Non Goals



V2では以下を行わない。



\---



独自学習エンジン開発



\---



独自画像生成エンジン開発



\---



独自タグモデル開発



\---



独自拡散モデル開発



\---



LoRA Basepipeは



オーケストレーション



分析



管理



に集中する。



以上 Part 1 / 3





\# LoRA Basepipe V2 SPEC



\# Part 2 / 3



\# Dataset Engineering / Caption Intelligence / Dataset Refinery



\---



\# 8. Dataset Engineering



Dataset Engineering は

LoRA Basepipe V2 の中核機能である。



本システムは



学習ツール



ではなく



Dataset Engineering Platform



として設計する。



\---



\## 8.1 Purpose



目的



学習前に



\* データセット品質

\* 情報密度

\* 偏り

\* 汚染



を可視化すること。



\---



\## 8.2 Dataset Dashboard



データセット作成画面では



単なる画像一覧ではなく



データセット全体の状態を表示する。



表示項目



\* 総画像数

\* 推定キャラクター数

\* 推定衣装数

\* 推定背景数

\* 解像度分布

\* アスペクト比分布

\* カラー比率

\* 白黒比率

\* 類似画像数

\* 重複画像数



\---



\## 8.3 Dataset Quality Score



データセット品質を数値化する。



例



Dataset Quality



92 / 100



\---



減点対象



\* 重複

\* 極端な低解像度

\* 白黒偏重

\* 同一構図偏重

\* 同一キャラ偏重

\* 同一衣装偏重

\* 吹き出し

\* 擬音

\* 圧縮ノイズ



\---



このスコアは参考値であり



学習可否を制限しない。



\---



\# 9. Dataset Intelligence



AIによる分析機能。



\---



\## 9.1 Similarity Analysis



CLIP系特徴量を利用する。



検出対象



\* 類似画像

\* 近似構図

\* 連番画像

\* 微差分画像



\---



UI表示



類似画像グループ



Aグループ

12枚



Bグループ

8枚



のように表示する。



\---



\## 9.2 Character Distribution Analysis



データセット内のキャラ偏りを分析する。



表示例



Character A

67%



Character B

12%



Character C

8%



Other

13%



\---



Style LoRA作成時は警告を表示する。



\---



\## 9.3 Costume Distribution Analysis



衣装偏りを分析する。



例



School Uniform

72%



Casual

13%



Other

15%



\---



Style Modeでは警告対象。



\---



\## 9.4 Hairstyle Distribution Analysis



髪型偏りを分析する。



例



Twin Tail

63%



Long Hair

18%



Other

19%



\---



Style Modeでは警告対象。



\---



\# 10. Dataset Mixer



LoRA Basepipe独自機能。



\---



\## Purpose



何を学習させ



何を学習させないか



を調整する。



\---



\## Philosophy



Style LoRA



Character LoRA



という固定分類は採用しない。



代わりに



Feature Weight



を採用する。



\---



\## Feature Categories



Face



Expression



Hair



Costume



Accessory



Background



Composition



Line



Color



Lighting



Mood



\---



各項目



0〜100



\---



\## Example



UMI Style



Face

80



Hair

20



Costume

10



Accessory

5



Line

100



Color

100



\---



Character LoRA



Face

100



Hair

100



Costume

100



Accessory

80



Line

40



Color

30



\---



\## Purpose



学習方針を数値化する。



\---



\# 11. Character Leak Analysis



V2最重要機能の一つ。



\---



\## Problem



Style LoRA作成時



WD14タグと

キャラクター固有要素が結びつく問題が存在する。



例



long hair



が



特定キャラの前髪形状



として学習される。



\---



\## Goal



タグだけを見ず



データセット全体から



リーク候補を検出する。



\---



\## Detection Targets



キャラ固有前髪



キャラ固有髪色



キャラ固有衣装



キャラ固有アクセサリ



キャラ固有装飾



\---



\## Output



警告として表示する。



自動削除は行わない。



\---



\# 12. Caption Intelligence



タグ生成を超えた概念。



\---



\## Philosophy



WD14単独に依存しない。



複数モデルの結果を統合する。



\---



\## Sources



WD14



JoyCaption



Florence2



Qwen-VL



DeepSeek-VL



\---



\## Stored Information



Raw WD14



Raw Vision Caption



Merged Caption



Final Caption



\---



\## Category Classification



タグを分類する。



GENERAL



FACE



HAIR



COSTUME



ACCESSORY



BACKGROUND



STYLE



COLOR



LIGHTING



COMPOSITION



\---



\## Purpose



単なるタグ保存ではなく



意味単位で管理する。



\---



\# 13. Caption Editing UX



大量編集を前提とする。



\---



\## Batch Replace



例



school\_uniform



↓



uniform



\---



\## Batch Remove



例



hair\_ornament



全削除



\---



\## Batch Category Edit



HAIRカテゴリ全体を弱化



などを可能にする。



\---



\# 14. Dataset Refinery



画像前処理エンジン。



\---



\## Philosophy



画像を1枚ずつ手作業で修正することを前提にしない。



パイプライン化を前提とする。



\---



\## Engine



ComfyUI Workflow



\---



\## Workflow Presets



Speech Bubble Removal



Text Removal



Background Cleanup



Character Isolation



Style Cleanup



Custom Workflow



\---



\## Batch Processing



複数画像に一括適用可能。



\---



\# 15. Style Extraction Pipeline



将来の重要機能。



\---



\## Goal



画像から



スタイル要素



と



キャラ要素



を分離する。



\---



例



保持



\* 顔つき

\* 線

\* 塗り

\* 色使い

\* 陰影



弱化



\* 衣装

\* 髪飾り

\* 固有小物



除去



\* 吹き出し

\* 擬音

\* テキスト



\---



\## Output



Refined Dataset



\---



\# 16. Human Review Gate



学習前に必ずレビューを行う。



\---



学習開始前



Dataset Report



を生成する。



内容



\* Dataset Quality

\* Character Leak

\* Caption Summary

\* Similarity Report

\* Warning List



\---



ユーザーが承認した場合のみ



Trainingへ進む。



以上 Part 2 / 3



\# LoRA Basepipe V2 SPEC



\# Part 3 / 3



\# Training / Evaluation / Asset Intelligence / Future Expansion



\---



\# 17. Training System



Training Module は

Kohyaを中心とした学習オーケストレーターとして動作する。



LoRA Basepipe自身は

学習エンジンを持たない。



\---



\## 17.1 Training Engine



採用



\* Kohya SS



\---



将来的な拡張



\* OneTrainer

\* Flux Trainer



ただしV2では対象外。



\---



\## 17.2 Training Philosophy



ユーザーは



学習設定を作る



のではなく



学習方針を決める。



\---



システムが



\* Rank

\* Alpha

\* Repeat

\* Epoch

\* Optimizer

\* Scheduler



を提案する。



\---



\## 17.3 Training Profiles



Profile方式を採用する。



\---



Style



Character



Hybrid



\---



さらに



ユーザー独自Profile



保存可能。



\---



例



UMI Style



RANSI Style



TWW Style



Custom



\---



\## 17.4 Advanced Settings



上級者向け。



すべてのKohya設定を編集可能。



\---



ただし通常UIでは非表示。



\---



\# 18. Training Monitor



学習状況をリアルタイムで可視化する。



\---



\## Current Status



表示



\* Running

\* Paused

\* Stopped

\* Finished

\* Error



\---



\## Progress



表示



\* Epoch

\* Step

\* Progress

\* ETA



\---



\## Resource Monitor



表示



\* GPU Usage

\* VRAM Usage

\* RAM Usage

\* Disk Usage



\---



\## Training Log



リアルタイム表示。



検索可能。



\---



\# 19. Live Preview System



V2の主要機能。



\---



\## Goal



学習終了後ではなく



学習中に品質を確認する。



\---



\## Engine



ComfyUI



\---



\## Preview Workflow



固定Workflowを使用する。



\---



固定項目



Seed



Prompt



Negative Prompt



CFG



Sampler



Resolution



\---



変化するのは



Checkpointのみ。



\---



\## Auto Generation



指定Epochごとに



自動生成する。



\---



例



Epoch 1



↓



Preview



↓



Epoch 2



↓



Preview



↓



Epoch 3



↓



Preview



\---



\## Compare Timeline



時系列比較。



\---



Epoch 1



Epoch 2



Epoch 3



Epoch 4



...



を横並び表示。



\---



\# 20. Evaluation System



学習完了後の評価システム。



\---



\## Philosophy



最重要指標は



再現性



ではない。



\---



最重要指標



操縦性



である。



\---



\## Reproducibility



意図した特徴が再現されるか。



\---



\## Controllability



LoRA強度を上げた際



不要情報が増えないか。



\---



\## Character Leak



不要なキャラ特徴が混入していないか。



\---



\## Style Purity



スタイル抽出精度。



\---



\## Overfitting



過学習判定。



\---



\# 21. Evaluation Workspace



比較専用画面。



\---



\## Compare Modes



Side By Side



Slider



Grid



Timeline



\---



\## Fixed Test Suite



毎回同じ条件で評価する。



\---



Prompt Set



Seed Set



ControlNet Set



を保存する。



\---



\## Purpose



LoRA同士の比較を可能にする。



\---



\# 22. Asset Intelligence



V2の重要機能。



\---



\## Philosophy



LoRAは成果物ではない。



資産である。



\---



\## Asset Unit



1 Asset に保存するもの



\---



LoRA



Dataset



Caption



Training Config



Preview



Evaluation



Metadata



\---



\## Benefit



数ヶ月後でも



学習条件を完全再現できる。



\---



\# 23. Asset Library



LoRA管理画面。



\---



\## Stored Metadata



Name



Author



Created Date



Base Model



Dataset Size



Training Profile



Tags



\---



\## Search



Name



Tag



Profile



Model



Date



\---



\## Grouping



Style



Character



Hybrid



Custom



\---



\# 24. Recipe System



V2で最も重要な資産。



\---



\## Dataset Recipe



保存内容



Feature Weight



Caption Rule



Refinery Workflow



Filter Rule



\---



\## Training Recipe



保存内容



Rank



Alpha



Repeat



Epoch



Optimizer



Scheduler



\---



\## Evaluation Recipe



保存内容



Prompt Set



Seed Set



Preview Workflow



\---



\## Goal



成功事例を再利用可能にする。



\---



\# 25. Workflow Automation



V2後半で実装対象。



\---



\## One Click Pipeline



Dataset



↓



Caption



↓



Refinery



↓



Training



↓



Preview



↓



Evaluation



を一括実行。



\---



ただし



各工程で停止可能。



\---



\# 26. Internal API Architecture



将来拡張のため



全処理をジョブ化する。



\---



例



CreateProject



ScanDataset



GenerateCaption



RunRefinery



StartTraining



GeneratePreview



EvaluateAsset



ExportAsset



\---



UIは内部APIを呼び出すだけとする。



\---



\# 27. Future CloudAI Integration



V2では実装しない。



\---



将来的に



CloudAI OS



↓



LoRA Basepipe



連携可能な構造を維持する。



\---



想定フロー



企画生成



↓



必要LoRA分析



↓



LoRA作成依頼



↓



Dataset作成



↓



Training



↓



Asset登録



\---



ただしV2では対象外。



\---



\# 28. Success Criteria



LoRA Basepipe V2 が成功したと判断する条件。



\---



従来



LoRA制作



30〜60分以上



\---



V2



5〜10分以内で学習開始可能



\---



Dataset分析



自動



\---



Caption生成



自動



\---



前処理



半自動



\---



学習監視



リアルタイム



\---



評価



標準化



\---



資産管理



完全再現可能



\---



最終目標



ユーザーは



LoRAを作るために作業する



のではなく



LoRAを設計するだけで済む状態を実現する。



End of Part 3





