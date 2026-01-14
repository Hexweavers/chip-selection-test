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
