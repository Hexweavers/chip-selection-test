from pathlib import Path

import libsql_experimental as libsql
from config import TURSO_DATABASE_URL, TURSO_AUTH_TOKEN


def get_db():
    """Get a connection to the Turso database."""
    if not TURSO_DATABASE_URL:
        raise ValueError("TURSO_DATABASE_URL environment variable is required")

    return libsql.connect(
        TURSO_DATABASE_URL,
        auth_token=TURSO_AUTH_TOKEN or "",
    )


def init_db(db=None):
    """Initialize database schema."""
    if db is None:
        db = get_db()

    schema_path = Path(__file__).parent / "schema.sql"
    schema = schema_path.read_text()

    db.executescript(schema)
    db.commit()
    return db
