from __future__ import annotations

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[2] / "workspace.db"


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.executescript(
        """
        CREATE TABLE IF NOT EXISTS projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            project_type TEXT NOT NULL DEFAULT 'character',
            base_dir TEXT NOT NULL,
            dataset_dir TEXT NOT NULL,
            captions_dir TEXT NOT NULL,
            outputs_dir TEXT NOT NULL,
            library_dir TEXT NOT NULL DEFAULT '',
            preset_id INTEGER,
            status TEXT NOT NULL DEFAULT 'idle',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS dataset_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER NOT NULL,
            file_path TEXT NOT NULL,
            width INTEGER,
            height INTEGER,
            aspect TEXT,
            selected INTEGER NOT NULL DEFAULT 1,
            FOREIGN KEY(project_id) REFERENCES projects(id)
        );

        CREATE TABLE IF NOT EXISTS checkpoints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER NOT NULL,
            file_path TEXT NOT NULL,
            epoch INTEGER NOT NULL,
            step INTEGER,
            mark TEXT NOT NULL DEFAULT 'none',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(project_id) REFERENCES projects(id)
        );

        CREATE TABLE IF NOT EXISTS preview_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            checkpoint_id INTEGER NOT NULL,
            slot TEXT NOT NULL,
            image_path TEXT NOT NULL,
            FOREIGN KEY(checkpoint_id) REFERENCES checkpoints(id)
        );

        CREATE TABLE IF NOT EXISTS presets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS training_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER NOT NULL,
            status TEXT NOT NULL,
            stop_mode TEXT,
            latest_checkpoint_path TEXT,
            config_json TEXT,
            current_epoch INTEGER NOT NULL DEFAULT 0,
            current_step INTEGER NOT NULL DEFAULT 0,
            total_epochs INTEGER NOT NULL DEFAULT 5,
            steps_per_epoch INTEGER NOT NULL DEFAULT 20,
            started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(project_id) REFERENCES projects(id)
        );

        CREATE TABLE IF NOT EXISTS app_settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS dataset_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER NOT NULL UNIQUE,
            quality_score INTEGER NOT NULL DEFAULT 0,
            stats_json TEXT NOT NULL DEFAULT '{}',
            similarity_json TEXT NOT NULL DEFAULT '[]',
            analyzed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(project_id) REFERENCES projects(id)
        );

        CREATE TABLE IF NOT EXISTS lora_assets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER,
            name TEXT NOT NULL,
            lora_path TEXT NOT NULL DEFAULT '',
            base_model TEXT NOT NULL DEFAULT '',
            dataset_size INTEGER NOT NULL DEFAULT 0,
            profile_name TEXT NOT NULL DEFAULT '',
            tags_json TEXT NOT NULL DEFAULT '[]',
            notes TEXT NOT NULL DEFAULT '',
            preview_path TEXT NOT NULL DEFAULT '',
            training_config_json TEXT NOT NULL DEFAULT '{}',
            quality_score INTEGER,
            asset_type TEXT NOT NULL DEFAULT 'character',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(project_id) REFERENCES projects(id)
        );

        CREATE TABLE IF NOT EXISTS dataset_suggestions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER NOT NULL,
            suggestions_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(project_id) REFERENCES projects(id)
        );

        CREATE TABLE IF NOT EXISTS dataset_mixer (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER NOT NULL UNIQUE,
            feature_weights_json TEXT NOT NULL DEFAULT '{}',
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(project_id) REFERENCES projects(id)
        );

        CREATE TABLE IF NOT EXISTS project_tag_settings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER NOT NULL UNIQUE,
            prefix_tags_json TEXT NOT NULL DEFAULT '[]',
            block_words_json TEXT NOT NULL DEFAULT '[]',
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(project_id) REFERENCES projects(id)
        );

        CREATE TABLE IF NOT EXISTS project_outfit_profiles (
            project_id INTEGER NOT NULL PRIMARY KEY,
            profiles_json TEXT NOT NULL DEFAULT '[]',
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(project_id) REFERENCES projects(id)
        );

        -- Preview Job SSOT: 個々のPreview要求(1 prompt × 1 instance)を永続追跡する。
        -- checkpoints.validation_status(preview_succeeded/failed等)はこのテーブルからの
        -- 派生集計値であり、ここがSingle Source of Truth。以前はcheckpoint単位のstatus
        -- しか無く「複数Prompt中1件成功しただけで全体succeeded」という実バグがあった。
        CREATE TABLE IF NOT EXISTS preview_jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            checkpoint_id INTEGER NOT NULL,
            run_id INTEGER NOT NULL,
            project_id INTEGER NOT NULL,
            epoch INTEGER NOT NULL,
            prompt_index INTEGER NOT NULL,
            instance_index INTEGER NOT NULL,
            prompt TEXT NOT NULL DEFAULT '',
            negative_prompt TEXT NOT NULL DEFAULT '',
            seed INTEGER NOT NULL,
            preview_model_family TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'pending',
            attempts INTEGER NOT NULL DEFAULT 0,
            error_detail TEXT NOT NULL DEFAULT '',
            output_path TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            started_at TEXT,
            completed_at TEXT,
            FOREIGN KEY(checkpoint_id) REFERENCES checkpoints(id),
            UNIQUE(checkpoint_id, prompt_index, instance_index)
        );
        CREATE INDEX IF NOT EXISTS idx_preview_jobs_checkpoint ON preview_jobs(checkpoint_id);
        CREATE INDEX IF NOT EXISTS idx_preview_jobs_status ON preview_jobs(status);
        CREATE INDEX IF NOT EXISTS idx_preview_jobs_run ON preview_jobs(run_id);

        CREATE TABLE IF NOT EXISTS preview_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            checkpoint_id INTEGER NOT NULL,
            job_id INTEGER,
            slot TEXT NOT NULL DEFAULT '',
            image_path TEXT NOT NULL,
            preview_snapshot_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(checkpoint_id) REFERENCES checkpoints(id),
            FOREIGN KEY(job_id) REFERENCES preview_jobs(id)
        );
        CREATE INDEX IF NOT EXISTS idx_preview_history_checkpoint ON preview_history(checkpoint_id);

        CREATE TABLE IF NOT EXISTS preview_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER,
            name TEXT NOT NULL,
            prompt TEXT NOT NULL DEFAULT '',
            negative_prompt TEXT NOT NULL DEFAULT '',
            seed INTEGER NOT NULL DEFAULT 42,
            resolution INTEGER NOT NULL DEFAULT 1024,
            steps INTEGER NOT NULL DEFAULT 20,
            cfg REAL NOT NULL DEFAULT 7.0,
            sampler TEXT NOT NULL DEFAULT '',
            model_family TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(project_id, name),
            FOREIGN KEY(project_id) REFERENCES projects(id)
        );
        CREATE INDEX IF NOT EXISTS idx_preview_profiles_project ON preview_profiles(project_id);

        -- LoRA Basepipe v1 foundation. These records are additive to the legacy
        -- dataset/training tables and keep immutable lineage separate from UI drafts.
        CREATE TABLE IF NOT EXISTS basepipe_concepts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            concept_type TEXT NOT NULL DEFAULT 'character',
            trigger_token TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'draft',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(project_id) REFERENCES projects(id),
            UNIQUE(project_id, name)
        );
        CREATE INDEX IF NOT EXISTS idx_basepipe_concepts_project ON basepipe_concepts(project_id);

        CREATE TABLE IF NOT EXISTS basepipe_assets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER NOT NULL,
            asset_key TEXT NOT NULL,
            file_path TEXT NOT NULL,
            content_sha256 TEXT NOT NULL,
            asset_type TEXT NOT NULL DEFAULT 'image',
            origin_kind TEXT NOT NULL DEFAULT 'imported',
            source_ref TEXT NOT NULL DEFAULT '',
            caption TEXT NOT NULL DEFAULT '',
            caption_source TEXT NOT NULL DEFAULT '',
            review_status TEXT NOT NULL DEFAULT 'pending',
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(project_id) REFERENCES projects(id),
            UNIQUE(project_id, asset_key, content_sha256)
        );
        CREATE INDEX IF NOT EXISTS idx_basepipe_assets_project ON basepipe_assets(project_id);

        CREATE TABLE IF NOT EXISTS basepipe_asset_groups (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(project_id) REFERENCES projects(id),
            UNIQUE(project_id, name)
        );
        CREATE INDEX IF NOT EXISTS idx_basepipe_asset_groups_project ON basepipe_asset_groups(project_id);

        CREATE TABLE IF NOT EXISTS basepipe_asset_group_members (
            group_id INTEGER NOT NULL,
            asset_id INTEGER NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(group_id, asset_id),
            FOREIGN KEY(group_id) REFERENCES basepipe_asset_groups(id),
            FOREIGN KEY(asset_id) REFERENCES basepipe_assets(id)
        );
        CREATE INDEX IF NOT EXISTS idx_basepipe_asset_group_members_asset ON basepipe_asset_group_members(asset_id);

        CREATE TABLE IF NOT EXISTS basepipe_asset_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset_id INTEGER NOT NULL,
            project_id INTEGER NOT NULL,
            parent_version_id INTEGER,
            version_kind TEXT NOT NULL,
            file_path TEXT NOT NULL,
            content_sha256 TEXT NOT NULL,
            prompt TEXT NOT NULL DEFAULT '',
            seed INTEGER,
            status TEXT NOT NULL DEFAULT 'ready',
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(asset_id) REFERENCES basepipe_assets(id),
            FOREIGN KEY(project_id) REFERENCES projects(id),
            FOREIGN KEY(parent_version_id) REFERENCES basepipe_asset_versions(id)
        );
        CREATE INDEX IF NOT EXISTS idx_basepipe_asset_versions_asset ON basepipe_asset_versions(asset_id);

        CREATE TABLE IF NOT EXISTS basepipe_character_generation_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER NOT NULL,
            mode TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'preflight',
            requested_json TEXT NOT NULL DEFAULT '{}',
            resolved_json TEXT NOT NULL DEFAULT '{}',
            observed_json TEXT NOT NULL DEFAULT '{}',
            output_manifest_json TEXT NOT NULL DEFAULT '{}',
            current_stage TEXT NOT NULL DEFAULT 'planning',
            stage_history_json TEXT NOT NULL DEFAULT '[]',
            error_detail TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(project_id) REFERENCES projects(id)
        );
        CREATE INDEX IF NOT EXISTS idx_basepipe_character_runs_project ON basepipe_character_generation_runs(project_id);

        CREATE TABLE IF NOT EXISTS basepipe_caption_bulk_operations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER NOT NULL,
            changes_json TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'preview',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            applied_at TEXT,
            undone_at TEXT,
            FOREIGN KEY(project_id) REFERENCES projects(id)
        );
        CREATE INDEX IF NOT EXISTS idx_basepipe_caption_bulk_project ON basepipe_caption_bulk_operations(project_id);

        CREATE TABLE IF NOT EXISTS basepipe_dataset_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER NOT NULL,
            concept_id INTEGER,
            name TEXT NOT NULL,
            snapshot_hash TEXT NOT NULL UNIQUE,
            status TEXT NOT NULL DEFAULT 'sealed',
            item_count INTEGER NOT NULL DEFAULT 0,
            manifest_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(project_id) REFERENCES projects(id),
            FOREIGN KEY(concept_id) REFERENCES basepipe_concepts(id)
        );
        CREATE INDEX IF NOT EXISTS idx_basepipe_snapshots_project ON basepipe_dataset_snapshots(project_id);

        CREATE TABLE IF NOT EXISTS basepipe_snapshot_entries (
            snapshot_id INTEGER NOT NULL,
            asset_id INTEGER NOT NULL,
            ordinal INTEGER NOT NULL,
            caption_at_snapshot TEXT NOT NULL DEFAULT '',
            metadata_json TEXT NOT NULL DEFAULT '{}',
            PRIMARY KEY(snapshot_id, asset_id),
            FOREIGN KEY(snapshot_id) REFERENCES basepipe_dataset_snapshots(id),
            FOREIGN KEY(asset_id) REFERENCES basepipe_assets(id)
        );

        CREATE TABLE IF NOT EXISTS basepipe_preview_profile_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER NOT NULL,
            source_profile_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            snapshot_hash TEXT NOT NULL UNIQUE,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(project_id) REFERENCES projects(id),
            FOREIGN KEY(source_profile_id) REFERENCES preview_profiles(id)
        );
        CREATE INDEX IF NOT EXISTS idx_basepipe_preview_snapshots_project ON basepipe_preview_profile_snapshots(project_id);

        CREATE TABLE IF NOT EXISTS basepipe_evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER NOT NULL,
            checkpoint_id INTEGER NOT NULL,
            preview_slot TEXT NOT NULL DEFAULT '',
            identity INTEGER,
            outfit_separation INTEGER,
            style_durability INTEGER,
            style_quality INTEGER,
            prompt_flexibility INTEGER,
            artifact INTEGER,
            accepted INTEGER NOT NULL DEFAULT 0,
            note TEXT NOT NULL DEFAULT '',
            review_state TEXT NOT NULL DEFAULT 'AESTHETIC_UNREVIEWED',
            reviewed_by TEXT NOT NULL DEFAULT '',
            reviewed_at TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(project_id) REFERENCES projects(id),
            FOREIGN KEY(checkpoint_id) REFERENCES checkpoints(id),
            UNIQUE(project_id, checkpoint_id, preview_slot)
        );
        CREATE INDEX IF NOT EXISTS idx_basepipe_evaluations_project ON basepipe_evaluations(project_id);
        """
    )
    _ensure_column(cur, "training_runs", "current_epoch", "INTEGER NOT NULL DEFAULT 0")
    _ensure_column(cur, "training_runs", "current_step", "INTEGER NOT NULL DEFAULT 0")
    _ensure_column(cur, "training_runs", "total_epochs", "INTEGER NOT NULL DEFAULT 5")
    _ensure_column(cur, "training_runs", "steps_per_epoch", "INTEGER NOT NULL DEFAULT 20")
    _ensure_column(cur, "training_runs", "loss", "REAL")
    _ensure_column(cur, "training_runs", "log_path", "TEXT")
    _ensure_column(cur, "training_runs", "sec_per_step_ema", "REAL")
    _ensure_column(cur, "training_runs", "step1_at", "TEXT")
    _ensure_column(cur, "training_runs", "requested_config_json", "TEXT NOT NULL DEFAULT '{}'")
    _ensure_column(cur, "training_runs", "resolved_config_json", "TEXT NOT NULL DEFAULT '{}'")
    _ensure_column(cur, "training_runs", "manifest_path", "TEXT")
    _ensure_column(cur, "training_runs", "dataset_snapshot_id", "INTEGER")
    _ensure_column(cur, "training_runs", "failure_code", "TEXT")
    _ensure_column(cur, "training_runs", "failure_message", "TEXT")
    _ensure_column(cur, "training_runs", "failure_stage", "TEXT")
    _ensure_column(cur, "training_runs", "failure_at", "TEXT")
    _ensure_column(cur, "training_runs", "current_stage", "TEXT NOT NULL DEFAULT 'planning'")
    _ensure_column(cur, "training_runs", "stage_history_json", "TEXT NOT NULL DEFAULT '[]'")
    _ensure_column(cur, "projects", "project_type", "TEXT NOT NULL DEFAULT 'character'")
    _ensure_column(cur, "projects", "library_dir", "TEXT NOT NULL DEFAULT ''")
    # Dataset Builder の前処理パイプライン設定（Resize/Upscale/Cleanup/Caption）を
    # プロジェクトごとに保存・復元するための列。
    _ensure_column(cur, "projects", "pipeline_config_json", "TEXT NOT NULL DEFAULT '{}'")
    # Training設定フォーム(rank/learning_rate/optimizer等)をProject単位で保存・復元する列。
    # これは「最後に実行したrunの設定」(training_runs.config_json)とは別概念 —
    # ユーザーがまだ一度も学習を開始していない・設定を変更した直後にProject切替や
    # Page reloadを行った場合でも、直近の編集内容を失わないようにするための下書き保存。
    _ensure_column(cur, "projects", "training_config_json", "TEXT NOT NULL DEFAULT '{}'")
    _ensure_column(cur, "dataset_items", "caption", "TEXT NOT NULL DEFAULT ''")
    _ensure_column(cur, "dataset_items", "caption_source", "TEXT NOT NULL DEFAULT ''")
    _ensure_column(cur, "basepipe_assets", "caption_original", "TEXT NOT NULL DEFAULT ''")
    _ensure_column(cur, "basepipe_assets", "caption_edited", "TEXT NOT NULL DEFAULT ''")
    _ensure_column(cur, "basepipe_assets", "caption_processed", "TEXT NOT NULL DEFAULT ''")
    _ensure_column(cur, "basepipe_assets", "training_input", "TEXT NOT NULL DEFAULT ''")
    _ensure_column(cur, "basepipe_assets", "training_input_source", "TEXT NOT NULL DEFAULT 'original'")
    _ensure_column(cur, "basepipe_assets", "user_rating", "INTEGER")
    _ensure_column(cur, "basepipe_assets", "training_enabled", "INTEGER NOT NULL DEFAULT 1")
    _ensure_column(cur, "basepipe_assets", "training_weight", "REAL NOT NULL DEFAULT 1.0")
    _ensure_column(cur, "basepipe_assets", "updated_at", "TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP")
    _ensure_column(cur, "basepipe_character_generation_runs", "current_stage", "TEXT NOT NULL DEFAULT 'planning'")
    _ensure_column(cur, "basepipe_character_generation_runs", "stage_history_json", "TEXT NOT NULL DEFAULT '[]'")
    # checkpoint がどの学習 run に属するか（run跨ぎのepoch番号衝突を防ぐ）
    _ensure_column(cur, "checkpoints", "run_id", "INTEGER")
    # Artifact検証状態: none(未検証) / invalid(壊れている) / validated(safetensors検証OK) /
    # preview_succeeded / preview_failed。TRAINING_SUCCEEDED(training_runs.status)・
    # ARTIFACT_REGISTERED(このレコードの存在自体)と合わせて状態を区別できるようにする
    # （新規テーブルは追加せず、既存 checkpoints テーブルへ列追加のみで表現する）。
    _ensure_column(cur, "checkpoints", "validation_status", "TEXT NOT NULL DEFAULT 'none'")
    _ensure_column(cur, "checkpoints", "validation_detail", "TEXT NOT NULL DEFAULT ''")
    _ensure_column(cur, "lora_assets", "training_run_id", "INTEGER")
    _ensure_column(cur, "lora_assets", "dataset_snapshot_id", "INTEGER")
    _ensure_column(cur, "lora_assets", "library_status", "TEXT NOT NULL DEFAULT 'accepted'")
    _ensure_column(cur, "basepipe_evaluations", "preview_profile_snapshot_id", "INTEGER")
    _ensure_column(cur, "basepipe_evaluations", "review_state", "TEXT NOT NULL DEFAULT 'AESTHETIC_UNREVIEWED'")
    _ensure_column(cur, "basepipe_evaluations", "reviewed_by", "TEXT NOT NULL DEFAULT ''")
    _ensure_column(cur, "basepipe_evaluations", "reviewed_at", "TEXT")
    # Preview条件の証拠。生成要求時の解決値と、旧sample取り込みの由来を
    # 画像だけから推測せず、Job単位で保持する。
    _ensure_column(cur, "preview_jobs", "conditions_json", "TEXT NOT NULL DEFAULT '{}'")
    _ensure_column(cur, "preview_jobs", "preview_snapshot_json", "TEXT NOT NULL DEFAULT '{}'")
    _ensure_column(cur, "preview_profiles", "scheduler", "TEXT NOT NULL DEFAULT 'simple'")
    _ensure_column(cur, "preview_profiles", "outfits_json", "TEXT NOT NULL DEFAULT ''")

    # Initialize Pixiv session settings if not present
    _init_pixiv_settings(cur)

    conn.commit()
    conn.close()


def _ensure_column(cur: sqlite3.Cursor, table: str, column: str, ddl: str) -> None:
    columns = cur.execute(f"PRAGMA table_info({table})").fetchall()
    names = {row[1] for row in columns}
    if column not in names:
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


def _init_pixiv_settings(cur: sqlite3.Cursor) -> None:
    """Initialize Pixiv session settings if not already present."""
    pixiv_keys = [
        "pixiv_session_expires_at",
        "pixiv_session_last_validated",
        "pixiv_session_user_id"
    ]
    for key in pixiv_keys:
        row = cur.execute("SELECT value FROM app_settings WHERE key = ?", (key,)).fetchone()
        if not row:
            cur.execute(
                "INSERT INTO app_settings(key, value) VALUES (?, ?)",
                (key, "")
            )
