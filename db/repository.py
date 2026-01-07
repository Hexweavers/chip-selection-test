"""Repository for database operations."""

from datetime import datetime, timezone
import json
import sqlite3
import uuid


class Repository:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def create_run(
        self,
        name: str,
        persona: str,
        prompt_style: str,
        flow: str,
        constraint_type: str,
        chip_count: int,
    ) -> str:
        """Create a new run and return its ID."""
        run_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            """
            INSERT INTO runs (id, name, created_at, persona, prompt_style, flow, constraint_type, chip_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (run_id, name, created_at, persona, prompt_style, flow, constraint_type, chip_count),
        )
        self.conn.commit()
        return run_id

    def get_run(self, run_id: str) -> dict | None:
        """Get a run by ID."""
        cursor = self.conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def list_runs(self, limit: int = 100) -> list[dict]:
        """List runs ordered by creation date (newest first)."""
        cursor = self.conn.execute(
            "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
        )
        return [dict(row) for row in cursor.fetchall()]

    def delete_run(self, run_id: str) -> None:
        """Delete a run and its results."""
        self.conn.execute("DELETE FROM results WHERE run_id = ?", (run_id,))
        self.conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
        self.conn.commit()

    def create_result(
        self,
        run_id: str,
        model: str,
        chips: list[dict],
        tokens_in: int,
        tokens_out: int,
        cost_usd: float,
        latency_ms: int,
        situation_count: int,
        jargon_count: int,
        role_task_count: int,
        environment_count: int,
    ) -> str:
        """Create a result for a run and return its ID."""
        result_id = str(uuid.uuid4())
        chips_json = json.dumps(chips)
        self.conn.execute(
            """
            INSERT INTO results (
                id, run_id, model, chips, tokens_in, tokens_out, cost_usd, latency_ms,
                situation_count, jargon_count, role_task_count, environment_count
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result_id, run_id, model, chips_json, tokens_in, tokens_out, cost_usd,
                latency_ms, situation_count, jargon_count, role_task_count, environment_count,
            ),
        )
        self.conn.commit()
        return result_id

    def get_result(self, result_id: str) -> dict | None:
        """Get a result by ID."""
        cursor = self.conn.execute("SELECT * FROM results WHERE id = ?", (result_id,))
        row = cursor.fetchone()
        if not row:
            return None
        result = dict(row)
        result["chips"] = json.loads(result["chips"])
        return result

    def get_results_for_run(self, run_id: str) -> list[dict]:
        """Get all results for a run."""
        cursor = self.conn.execute(
            "SELECT * FROM results WHERE run_id = ? ORDER BY model", (run_id,)
        )
        results = []
        for row in cursor.fetchall():
            result = dict(row)
            result["chips"] = json.loads(result["chips"])
            results.append(result)
        return results

    def update_rating(self, result_id: str, rating: int) -> None:
        """Update the rating for a result (1-5)."""
        if not 1 <= rating <= 5:
            raise ValueError("Rating must be between 1 and 5")
        rated_at = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            "UPDATE results SET rating = ?, rated_at = ? WHERE id = ?",
            (rating, rated_at, result_id),
        )
        self.conn.commit()
