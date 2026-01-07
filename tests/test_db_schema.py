import sqlite3
import tempfile
from pathlib import Path

import pytest

from db.schema import init_db, DB_PATH


def test_init_db_creates_tables():
    """init_db should create runs and results tables."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        init_db(db_path)

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check runs table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='runs'"
        )
        assert cursor.fetchone() is not None

        # Check results table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='results'"
        )
        assert cursor.fetchone() is not None

        conn.close()


def test_init_db_is_idempotent():
    """Calling init_db twice should not error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        init_db(db_path)
        init_db(db_path)  # Should not raise
