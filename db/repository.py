from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass

from db.client import get_db


@dataclass
class Run:
    id: str
    started_at: str
    completed_at: str | None
    status: str
    config: dict | None


class Repository:
    def __init__(self, db=None):
        self.db = db or get_db()

    def create_run(self, config: dict | None = None) -> str:
        """Create a new test run. Returns run ID."""
        run_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        self.db.execute(
            """
            INSERT INTO runs (id, started_at, status, config)
            VALUES (?, ?, 'running', ?)
            """,
            (run_id, now, json.dumps(config) if config else None),
        )
        self.db.commit()
        return run_id

    def complete_run(self, run_id: str, status: str = "completed") -> None:
        """Mark a run as completed or failed."""
        now = datetime.now(timezone.utc).isoformat()
        self.db.execute(
            "UPDATE runs SET completed_at = ?, status = ? WHERE id = ?",
            (now, status, run_id),
        )
        self.db.commit()

    def get_run(self, run_id: str) -> Run | None:
        """Get a run by ID."""
        row = self.db.execute(
            "SELECT id, started_at, completed_at, status, config FROM runs WHERE id = ?",
            (run_id,),
        ).fetchone()

        if not row:
            return None

        return Run(
            id=row[0],
            started_at=row[1],
            completed_at=row[2],
            status=row[3],
            config=json.loads(row[4]) if row[4] else None,
        )

    def list_runs(self, limit: int = 50, offset: int = 0) -> list[dict]:
        """List runs with result counts."""
        rows = self.db.execute(
            """
            SELECT r.id, r.started_at, r.completed_at, r.status,
                   COUNT(res.id) as result_count
            FROM runs r
            LEFT JOIN results res ON res.run_id = r.id
            GROUP BY r.id
            ORDER BY r.started_at DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()

        return [
            {
                "id": row[0],
                "started_at": row[1],
                "completed_at": row[2],
                "status": row[3],
                "result_count": row[4],
            }
            for row in rows
        ]
