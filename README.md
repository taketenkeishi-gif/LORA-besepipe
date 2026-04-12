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

不安定な時は `stop_all.bat` 実行後に `start_web.bat` または `start_desktop.bat` を実行してください。

## 現在動く範囲

- プロジェクト作成/一覧
- mock画像収集（scan）とdataset取り込み（import）
- placeholderタグ生成（captions配下にtxt）
- 外部連携設定
  - Python / kohya / ComfyUI / WD14 のパス保存
  - 自動検出（autodetect）
  - 接続状態チェック（path存在 + pythonバージョン確認）
- 疑似学習ジョブ
  - `start` でepoch/step進行
  - `stop-now` で即停止
  - `stop-at-epoch` でepoch終端停止予約
  - `resume` で再開
  - epoch終端で checkpoint と preview プレースホルダ生成

UIは以下メニューで利用できます。
- Dashboard
- Projects
- Workflow
- Integrations

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
