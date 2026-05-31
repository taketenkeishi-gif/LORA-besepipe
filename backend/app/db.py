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
        """
    )
    _ensure_column(cur, "training_runs", "current_epoch", "INTEGER NOT NULL DEFAULT 0")
    _ensure_column(cur, "training_runs", "current_step", "INTEGER NOT NULL DEFAULT 0")
    _ensure_column(cur, "training_runs", "total_epochs", "INTEGER NOT NULL DEFAULT 5")
    _ensure_column(cur, "training_runs", "steps_per_epoch", "INTEGER NOT NULL DEFAULT 20")
    _ensure_column(cur, "training_runs", "loss", "REAL")
    _ensure_column(cur, "training_runs", "log_path", "TEXT")
    _ensure_column(cur, "projects", "project_type", "TEXT NOT NULL DEFAULT 'character'")
    _ensure_column(cur, "projects", "library_dir", "TEXT NOT NULL DEFAULT ''")
    _ensure_column(cur, "dataset_items", "caption", "TEXT NOT NULL DEFAULT ''")
    _ensure_column(cur, "dataset_items", "caption_source", "TEXT NOT NULL DEFAULT ''")
    conn.commit()
    conn.close()


def _ensure_column(cur: sqlite3.Cursor, table: str, column: str, ddl: str) -> None:
    columns = cur.execute(f"PRAGMA table_info({table})").fetchall()
    names = {row[1] for row in columns}
    if column not in names:
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
