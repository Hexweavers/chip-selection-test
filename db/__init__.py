from db.client import get_db, init_db

try:
    from db.repository import Repository
except ImportError:
    Repository = None  # Will be available after Task 5

__all__ = ["get_db", "init_db", "Repository"]
