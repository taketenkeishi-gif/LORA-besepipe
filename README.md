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
