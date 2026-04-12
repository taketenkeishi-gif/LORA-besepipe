# LoRA制作ワークベンチ

企画書をベースにした初期実装スキャフォールドです。

## 構成

- `backend/` FastAPI + SQLite + Pythonジョブ制御の入口
- `frontend/` React UI の土台
- `projects/` 実データ保存先（dataset/captions/checkpoints/previews）

## 起動手順

### 1) Backend（FastAPI）

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m app.main
```

起動後のAPIドキュメント:
- `http://127.0.0.1:8000/docs`

### 2) Frontend（Vite + React + TypeScript）

別ターミナルで:

```bash
cd frontend
npm install
npm run dev
```

フロント画面:
- `http://127.0.0.1:5173`

## BATで起動（Windows）

- `start_web.bat`: ブラウザ版を起動（backend + frontend）
- `start_desktop.bat`: デスクトップ版を起動（backend + frontend + Electron）
- `stop_all.bat`: backend / frontend / desktop 関連プロセスを停止
- `create_desktop_shortcut.bat`: デスクトップに非表示起動ショートカットを作成

不安定な時は `stop_all.bat` 実行後に `start_web.bat` または `start_desktop.bat` を実行してください。
起動ログは `.runtime/logs/` に出力されます。

## 現在動く範囲

- プロジェクト作成/一覧
- 画像候補収集（dataset_base優先）とdataset取り込み（import）
  - 候補サムネイル表示
  - 画像ファイルのドラッグ&ドロップ追加（Explorer）
  - 画像URLのドラッグ&ドロップ追加（Web）
  - `tag:` `aspect:` `minw:` `minh:` フィルタ
- タグ生成（WD14スクリプト設定時は既存caption優先）
- 外部連携設定
  - Python / kohya / ComfyUI / WD14 のパス保存
  - 初回アクセス時に自動検出結果を自動入力
  - 一時保存ディレクトリ（`temp_dir`）を設定可能
  - データセット補完ベース（`dataset_base_dir`）を設定可能
  - 自動検出（autodetect）
  - 接続状態チェック（path存在 + pythonバージョン確認）
- 学習ジョブ（疑似実行だが進捗可視化を強化）
  - `start` でepoch/step進行（`rank/alpha/repeats/save_every_n_epochs/output_name/resolution` 設定対応）
  - `stop-now` で即停止
  - `stop-at-epoch` でepoch終端停止予約
  - `resume` で再開
  - `total_steps / done_steps / progress_percent / eta_seconds` 表示
  - epoch終端で checkpoint と preview 画像をタイムラインへ追加
- プレビュー
  - Positive/Negative プロンプトを事前保存
  - エポックごとの軽量プレビュー履歴表示

UIは以下メニューで利用できます。
- Dashboard
- Projects
- Dataset
- Training
- Integrations

詳細な手順は [docs/使い方ガイド.md](docs/使い方ガイド.md) を参照してください。

## テスト

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pytest -q
```

このテストで以下を確認します。
- プロジェクト作成/一覧
- 収集(scan)→取り込み(import)→タグ生成
- 学習進行と完了
- previewタイムライン取得
