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
            base_dir TEXT NOT NULL,
            dataset_dir TEXT NOT NULL,
            captions_dir TEXT NOT NULL,
            outputs_dir TEXT NOT NULL,
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
            started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(project_id) REFERENCES projects(id)
        );
        """
    )
    conn.commit()
    conn.close()

