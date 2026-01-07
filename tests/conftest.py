"""Pytest configuration and fixtures."""

import pytest
import tempfile
from pathlib import Path

from db.schema import init_db
from db.repository import Repository


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        conn = init_db(db_path)
        yield conn
        conn.close()


@pytest.fixture
def repo(temp_db):
    """Create a repository with temporary database."""
    return Repository(temp_db)
