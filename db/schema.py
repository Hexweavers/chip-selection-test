"""SQLite database schema for chip benchmark results."""

from pathlib import Path
import sqlite3

DB_PATH = Path("benchmark.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id              TEXT PRIMARY KEY,
    name            TEXT,
    created_at      TEXT,
    persona         TEXT,
    prompt_style    TEXT,
    flow            TEXT,
    constraint_type TEXT,
    chip_count      INTEGER
);

CREATE TABLE IF NOT EXISTS results (
    id                  TEXT PRIMARY KEY,
    run_id              TEXT REFERENCES runs(id),
    model               TEXT,
    chips               TEXT,
    tokens_in           INTEGER,
    tokens_out          INTEGER,
    cost_usd            REAL,
    latency_ms          INTEGER,
    situation_count     INTEGER,
    jargon_count        INTEGER,
    role_task_count     INTEGER,
    environment_count   INTEGER,
    rating              INTEGER,
    rated_at            TEXT
);

CREATE INDEX IF NOT EXISTS idx_results_run_id ON results(run_id);
CREATE INDEX IF NOT EXISTS idx_results_model ON results(model);
CREATE INDEX IF NOT EXISTS idx_results_rating ON results(rating);
"""


def init_db(db_path: Path | None = None) -> sqlite3.Connection:
    """Initialize the database with schema. Returns connection."""
    path = db_path or DB_PATH
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    conn.commit()
    return conn
