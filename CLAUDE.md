# LoRA Basepipe — 開発コンテキスト共有ファイル

> このファイルは **複数アカウントの Claude Code 間での状態共有** を目的としています。
> 実装フェーズが完了するたびに更新してください。

---

## プロジェクト概要

Stable Diffusion LoRA モデル制作の全ワークフローを GUI 一本で管理する Desktop + Web アプリ。

**フルスタック構成:**
- Backend: FastAPI + SQLite (`workspace.db`) + Uvicorn
- Frontend: React 18 + TypeScript 5 + Vite 7 + Tailwind CSS v4 + lucide-react
- Desktop: Electron (`desktop/main.cjs`)

---

## ディレクトリ構成

```
LORA-besepipe-master/
├── CLAUDE.md              ← このファイル（状態共有）
├── backend/
│   ├── app/
│   │   ├── main.py        FastAPI エントリポイント・CORS設定
│   │   ├── db.py          SQLite 初期化・スキーマ・マイグレーション
│   │   ├── schemas.py     Pydantic モデル定義
│   │   ├── services/
│   │   │   └── tagger.py  WD14 ONNX 推論サービス（GPU対応）
│   │   └── routers/
│   │       ├── collector.py  画像収集・スキャン・インポート・サムネイル
│   │       ├── tags.py       キャプション生成・編集・バッチ操作
│   │       ├── training.py   学習制御（シミュレーション）
│   │       ├── previews.py   プレビュー画像管理
│   │       ├── projects.py   プロジェクト CRUD
│   │       └── settings.py   ツールパス設定
│   ├── requirements.txt   依存パッケージ
│   └── .venv/             仮想環境
├── frontend/
│   └── src/
│       ├── types/index.ts    共通型定義
│       ├── lib/
│       │   ├── api.ts        fetch ユーティリティ（ApiError / apiGet / apiPost）
│       │   └── utils.ts      parseIntOr / etaText / extractDroppedUrl
│       ├── components/
│       │   ├── Layout.tsx    サイドバー + メインレイアウト
│       │   ├── Dashboard.tsx 統計ダッシュボード
│       │   ├── Dataset.tsx   画像収集 + キャプション編集（2タブ）
│       │   ├── Training.tsx  学習制御・プレビュー履歴
│       │   └── Integrations.tsx  ツールパス設定
│       └── App.tsx           ルート（状態管理 + ルーティング）
└── desktop/
    └── main.cjs              Electron メインプロセス
```

---

## 起動方法

### ワンクリック起動（推奨）
```
start.bat   ← ルートのこれをダブルクリック
  [1] Web モード  (ブラウザ: http://127.0.0.1:5173)
  [2] Desktop モード  (Electron)
  [3] 全プロセス停止
  [4] デスクトップショートカット作成
```

### 開発時（個別起動）
```powershell
# Backend（Port 8000）
cd backend
.venv\Scripts\uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

# Frontend（Port 5173）
cd frontend
npm run dev
```

### ログ確認
`.runtime\logs\backend.err.log` / `backend.out.log`

---

## DB スキーマ（主要テーブル）

```sql
projects          -- プロジェクト基本情報
dataset_items     -- 画像ファイル（caption, caption_source 列あり）
training_runs     -- 学習ジョブ
checkpoints       -- 学習チェックポイント
preview_samples   -- チェックポイントごとのプレビュー画像
presets           -- 学習プリセット
app_settings      -- ツールパス等の設定
```

---

## 実装済み機能

### Phase 0 — 基盤（2025-05-31）
- [x] プロジェクト管理（作成・一覧・選択）
- [x] 画像収集（URL スキャン・Pixiv対応・D&D）
- [x] クライアントサイドソート・バッチ削除
- [x] 学習制御 UI（シミュレーション）
- [x] プレビュー履歴表示
- [x] ダークネイビー UI テーマ（Tailwind v4 + lucide-react）
- [x] imghdr → Pillow 置き換え（Python 3.13 対応）

### Phase 3 — Character Leak + Caption Categories（2026-05-31）
- [x] `services/tag_categories.py` 新規追加
  - SPEC §12 カテゴリパターン (FACE/HAIR/COSTUME/ACCESSORY/BACKGROUND/BODY/COLOR/STYLE/GENERAL)
  - §11 リーク検出グループ: `HAIR_COLORS`, `HAIR_STYLES`, `EYE_COLORS`, `COSTUME_TYPES`
  - `classify_tag()` / `classify_caption()` / `detect_dominant_feature()` ヘルパー
- [x] `routers/dataset.py` に 2 エンドポイント追加
  - `GET /dataset/character-leak/{project_id}` — §11: リスクスコア・偏り検出・WD14 キャラタグ
  - `GET /dataset/tag-categories/{project_id}` — §12: カテゴリ別カバレッジ・トップタグ
- [x] `types/index.ts` に型追加: `LeakItem`, `CharacterTagLeak`, `CharacterLeakResult`, `CategoryDetail`, `TagCategoriesResult`
- [x] Dashboard タブに §11/§12 UI 追加
  - 「リーク分析実行」ボタン（手動トリガー、SPEC §3.2 Manual First 準拠）
  - リスクレベルバッジ (LOW/MEDIUM/HIGH)・リスクスコア
  - 特徴偏りリスト（feature / dominant / ratio バー / severity）
  - WD14 キャラタグ一覧チップ
  - §12 カテゴリカバレッジバー（全9カテゴリ・トップ3タグ）

### Phase 2 — Dataset Engineering Dashboard（2026-05-31）
- [x] `routers/dataset.py` 新規追加
  - `POST /dataset/analyze/{project_id}` — バックグラウンドpHash分析
  - `GET /dataset/analysis-status/{project_id}` — 分析進捗
  - `GET /dataset/stats/{project_id}` — 統計・Quality Score
  - `GET /dataset/similarity/{project_id}` — 類似グループ
- [x] DB に `dataset_analysis` テーブル追加（結果キャッシュ）
- [x] Quality Score 算出（§8.3）: 低解像度・未キャプション・重複・類似・白黒偏重を減点
- [x] pHash 重複・類似検出（§9.1）: `ImageHash 4.3.2` + `scipy` 使用
- [x] モノクロ判定: RGB 彩度差でグレースケール画像を検出
- [x] Dataset.tsx に「Dashboard」3 タブ目追加
  - Quality Score サークル（色分け: 緑/黄/赤）
  - 統計カード（総数・キャプション率・カラー比・重複数）
  - 警告リスト（SPEC §8.3 減点対象を表示）
  - 解像度分布バー（High/Medium/Low）
  - アスペクト比分布バー（Portrait/Landscape/Square）
  - 類似グループ一覧（折りたたみ表示）
- [x] `ImageHash>=4.3.0` を requirements.txt に追加

### Phase 1 — WD14 + キャプション基盤（2026-05-31）
- [x] `services/tagger.py` — WD14 ONNX 推論（GPU自動検出）
- [x] DB に `caption`, `caption_source` カラム追加
- [x] `POST /tags/generate` — バックグラウンド推論（overwrite対応）
- [x] `GET /tags/status/{project_id}` — 推論進捗ポーリング
- [x] `GET /tags/{project_id}` — キャプション一覧（.txt 同期対応）
- [x] `PATCH /tags/item/{item_id}` — 個別編集（.txt 書き出し）
- [x] `POST /tags/batch-replace` — バッチ置換
- [x] `POST /tags/batch-remove` — タグ単位の一括削除
- [x] `GET /tags/frequency/{project_id}` — タグ出現頻度分析
- [x] `GET /collector/thumbnail` — 動的サムネイル生成
- [x] Dataset.tsx を 2 タブ構成に更新
  - 画像収集タブ（従来機能）
  - キャプション編集タブ（新規）
    - 統計カード・WD14 生成ボタン・プログレスバー
    - キャプション一覧（インライン編集）
    - バッチ置換 / バッチ削除パネル
    - タグ頻度チャート（クリックでバッチ置換に転送）
    - フィルタ（全て / 未キャプション / WD14 / 手動）
- [x] `onnxruntime-gpu 1.26.0`, `huggingface-hub 1.17.0`, `numpy 2.4.6` インストール済み

---

## 未実装（SPEC 参照）

| 機能 | SPEC 章 | 優先度 |
|------|---------|--------|
| ~~Dataset Dashboard（品質スコア・解像度分布）~~ | §8 | ✅ 完了 |
| ~~Similarity Analysis（重複・類似検出）~~ | §9.1 | ✅ 完了 |
| ~~Distribution Analysis（髪色・衣装分布）~~ | §9.2-9.4 | ✅ 完了 |
| ~~Character Leak Analysis~~ | §11 | ✅ 完了 |
| ~~Caption Category Classification~~ | §12 | ✅ 完了 |
| ~~Human Review Gate（学習前承認フロー）~~ | §16 | ✅ 完了 |
| ~~Training Profiles（Style/Character/Hybrid）~~ | §17.3 | ✅ 完了 |
| ~~kohya_ss 実接続（学習のシミュレーション解除）~~ | — | ✅ 完了 |
| ~~Resource Monitor（CPU/RAM/GPU モニタリング）~~ | §18 | ✅ 完了 |
| ~~Asset Library（LoRA 資産管理）~~ | §22-23 | ✅ 完了 |
| JoyCaption / Florence2 統合 | §13 | ⭐⭐ |
| Dataset Refinery（ComfyUI 前処理） | §14 | ⭐⭐ |
| Live Preview（ComfyUI 実接続） | §19 | ⭐ |
| Evaluation Module（LoRA 品質評価） | §20-21 | ⭐ |
| Recipe System | §24 | ⭐ |

---

### Phase 4 — Training Profiles + Review Gate + kohya_ss 実接続（2026-05-31）
- [x] `routers/presets.py` 新規追加
  - CRUD: `GET /presets`, `POST /presets`, `GET /presets/{id}`, `PUT /presets/{id}`, `DELETE /presets/{id}`
  - `POST /presets/seed-defaults` — 起動時に自動シード（Character/Style/Hybrid/Lightweight 4種）
- [x] `routers/dataset.py` に `GET /dataset/report/{project_id}` 追加
  - §16 Human Review Gate: 品質スコア・キャプション率・リーク分析・警告・ブロッカーを一括返却
- [x] `routers/training.py` 大幅改修
  - `GET /training/mode` — kohya_ss 接続状態チェック
  - `GET /training/logs/{project_id}` — 学習ログ末尾取得
  - `_runner_loop_kohya()` — kohya_ss サブプロセス起動・stdout パース・DB更新
  - `_runner_loop_simulated()` — シミュレーション（フォールバック）
  - `_runner_loop()` — kohya 設定有無で自動ディスパッチ
  - `.runtime/runs/{run_id}/` に学習ディレクトリ自動生成（hard link fallback to copy）
  - TOML コンフィグ自動生成
  - stop_now で kohya サブプロセスを terminate
- [x] `db.py` — `training_runs.loss`, `training_runs.log_path` カラム追加
- [x] `schemas.py` — `TrainingStartIn` に optimizer / scheduler / min_snr_gamma 追加; `PresetCreate`, `PresetUpdateIn` 追加
- [x] `main.py` — presets ルーター登録・起動時自動シード
- [x] `types/index.ts` — `Preset`, `PresetPayload`, `DatasetReport`, `TrainingMode` 型追加
- [x] `Training.tsx` 大幅改修
  - 学習プロファイルカード（4種デフォルト + カスタム）— クリックでパラメータ自動適用
  - Review Gate モーダル（学習開始前に Dataset Report 表示・承認必須）
  - kohya_ss 接続状態バッジ（🟢 kohya_ss / ⚡ simulation）
  - Optimizer / LR Scheduler セレクター
  - 学習ログビューア（LIVE ポーリング対応）
  - loss 表示

### Phase 5 — Asset Library + Distribution Analysis + Resource Monitor（2026-05-31）
- [x] `db.py` — `lora_assets` テーブル追加（id/project_id/name/lora_path/base_model/dataset_size/profile_name/tags_json/notes/training_config_json/quality_score/asset_type/created_at）
- [x] `routers/library.py` 新規追加
  - `GET /library/assets` — 一覧（project_id/asset_type フィルタ対応）
  - `POST /library/assets` — 作成
  - `GET /library/assets/{id}` — 取得
  - `PUT /library/assets/{id}` — 更新（差分のみ）
  - `DELETE /library/assets/{id}` — 削除（ファイルは保持）
  - `GET /library/stats` — 統計（総数・タイプ別・平均スコア）
- [x] `routers/dataset.py` に `GET /dataset/distribution/{project_id}` 追加
  - §9.2-9.4: HAIR_COLORS / HAIR_STYLES / EYE_COLORS / COSTUME_TYPES 分布集計
- [x] `routers/training.py` に `GET /training/resources` 追加
  - §18: psutil で CPU%/RAM、pynvml → nvidia-smi フォールバックで GPU VRAM/Util/温度
- [x] `requirements.txt` — `psutil>=5.9.0` 追加・インストール済み
- [x] `types/index.ts` — `LoraAsset`, `LoraAssetPayload`, `DistributionItem`, `DistributionData`, `GpuInfo`, `ResourceStats` 追加
- [x] `Library.tsx` 新規追加
  - 資産カードグリッド（タイプ/プロジェクト/検索フィルタ）
  - 品質スコアバッジ（緑/黄/赤）
  - 編集モーダル（名前・タイプ・パス・ベースモデル・プロファイル・タグ・メモ・スコア）
- [x] `Layout.tsx` / `App.tsx` — `"library"` タブ追加
- [x] `Training.tsx` — §18 リソースモニターカード追加
  - CPU%/RAM バー、GPU VRAM/Util/温度バー
  - 学習中に 3 秒ポーリング
- [x] `Dataset.tsx` — §9.2-9.4 分布分析 UI 追加
  - 髪色・髪型・瞳色・衣装の Top-6 バーチャート

## 既知の制約・注意点

- **kohya_ss 実接続** — `Integrations` タブで `kohya_root` と `python_exe` を設定することで有効化。未設定時はシミュレーションにフォールバック
- **kohya 学習ディレクトリ** — `.runtime/runs/{run_id}/` に自動作成。ハードリンク失敗時はコピー
- **WD14 初回実行** — モデルダウンロードに数分・~600MB 必要
  - キャッシュ先: `.runtime/models/wd14/`
- **Pixiv original 画像** — ログイン必要で取得不可の場合あり
- **SCAN_CACHE はインメモリ** — サーバ再起動でリセット
- **`/collector/thumbnail`** — ローカルパスを直接受け付ける（ローカルツールのため許容）
- **WD14 モデル** — `SmilingWolf/wd-vit-large-tagger-v3`（general 閾値 0.35、character 0.85）
  - copyright タグは意図的に除外（キャラ混入防止）

---

## 次のアクション候補

1. **JoyCaption 統合（§13）** — WD14 に加えた第2のキャプションモデル（テキスト生成型）
2. **Dataset Refinery（§14）** — ComfyUI 前処理ワークフロー
3. **Live Preview（§19）** — ComfyUI 実接続によるリアルタイムプレビュー
4. **Evaluation Module（§20-21）** — LoRA 品質評価
5. **Asset Library（§22-23）** — LoRA 資産管理

---

## 次のアクション候補（残 SPEC）

1. **JoyCaption 統合（§13）** — LLM によるテキスト生成型キャプション（WD14 に代わる第2の選択肢）
2. **Dataset Refinery（§14）** — ComfyUI 前処理ワークフロー連携
3. **Live Preview（§19）** — ComfyUI 実接続によるリアルタイムプレビュー
4. **Evaluation Module（§20-21）** — LoRA 品質評価スコア算出
5. **Recipe System（§24）** — 学習設定の共有・複製フロー

---

*最終更新: 2026-05-31 | Phase 5 完了（§9.2-9.4 Distribution + §18 Resource Monitor + §22-23 Asset Library）*
