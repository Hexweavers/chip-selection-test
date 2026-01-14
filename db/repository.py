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

    def save_result(
        self,
        run_id: str,
        model: str,
        persona_id: str,
        sector: str,
        desired_role: str,
        style: str,
        input_type: str,
        constraint_type: str,
        chip_count: int,
        final_chips: list[dict],
        step1_chips: list[dict] | None = None,
        selected_chips: list[dict] | None = None,
        fill_chips: list[dict] | None = None,
        errors: list[str] | None = None,
        latency_ms: int | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cost_usd: float | None = None,
    ) -> str:
        """Save a test result. Returns result ID."""
        result_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        self.db.execute(
            """
            INSERT INTO results (
                id, run_id, model, persona_id, sector, desired_role,
                style, input_type, constraint_type, chip_count,
                final_chips, step1_chips, selected_chips, fill_chips,
                errors, latency_ms, input_tokens, output_tokens,
                cost_usd, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result_id, run_id, model, persona_id, sector, desired_role,
                style, input_type, constraint_type, chip_count,
                json.dumps(final_chips),
                json.dumps(step1_chips) if step1_chips else None,
                json.dumps(selected_chips) if selected_chips else None,
                json.dumps(fill_chips) if fill_chips else None,
                json.dumps(errors) if errors else None,
                latency_ms, input_tokens, output_tokens, cost_usd, now,
            ),
        )
        self.db.commit()
        return result_id

    def result_exists(
        self,
        run_id: str,
        model: str,
        persona_id: str,
        style: str,
        input_type: str,
        constraint_type: str,
        chip_count: int,
    ) -> bool:
        """Check if a result already exists (for resume functionality)."""
        row = self.db.execute(
            """
            SELECT 1 FROM results
            WHERE run_id = ? AND model = ? AND persona_id = ?
              AND style = ? AND input_type = ? AND constraint_type = ?
              AND chip_count = ?
            LIMIT 1
            """,
            (run_id, model, persona_id, style, input_type, constraint_type, chip_count),
        ).fetchone()
        return row is not None

    def get_result(self, result_id: str) -> dict | None:
        """Get a single result with full details."""
        row = self.db.execute(
            """
            SELECT r.*,
                   AVG(rat.rating) as avg_rating,
                   COUNT(rat.id) as rating_count
            FROM results r
            LEFT JOIN ratings rat ON rat.result_id = r.id
            WHERE r.id = ?
            GROUP BY r.id
            """,
            (result_id,),
        ).fetchone()

        if not row:
            return None

        return self._row_to_result_dict(row, full=True)

    def list_results(
        self,
        run_id: str | None = None,
        model: str | None = None,
        persona_id: str | None = None,
        rated_by: str | None = None,
        unrated_by: str | None = None,
        limit: int = 50,
        offset: int = 0,
        user_id: str | None = None,
    ) -> tuple[list[dict], int]:
        """List results with filters. Returns (results, total_count)."""
        conditions = []
        params = []

        if run_id:
            conditions.append("r.run_id = ?")
            params.append(run_id)
        if model:
            conditions.append("r.model = ?")
            params.append(model)
        if persona_id:
            conditions.append("r.persona_id = ?")
            params.append(persona_id)
        if rated_by:
            conditions.append("EXISTS (SELECT 1 FROM ratings WHERE result_id = r.id AND user_id = ?)")
            params.append(rated_by)
        if unrated_by:
            conditions.append("NOT EXISTS (SELECT 1 FROM ratings WHERE result_id = r.id AND user_id = ?)")
            params.append(unrated_by)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        # Get total count
        count_row = self.db.execute(
            f"SELECT COUNT(*) FROM results r {where_clause}",
            params,
        ).fetchone()
        total = count_row[0] if count_row else 0

        # Get results with ratings
        query = f"""
            SELECT r.*,
                   AVG(rat.rating) as avg_rating,
                   COUNT(rat.id) as rating_count
                   {"," if user_id else ""}
                   {f"(SELECT rating FROM ratings WHERE result_id = r.id AND user_id = ?) as my_rating" if user_id else ""}
            FROM results r
            LEFT JOIN ratings rat ON rat.result_id = r.id
            {where_clause}
            GROUP BY r.id
            ORDER BY r.created_at DESC
            LIMIT ? OFFSET ?
        """
        query_params = params.copy()
        if user_id:
            query_params.insert(0, user_id)
        query_params.extend([limit, offset])

        rows = self.db.execute(query, query_params).fetchall()

        return [self._row_to_result_dict(row, full=False, include_my_rating=bool(user_id)) for row in rows], total

    def _row_to_result_dict(self, row, full: bool = False, include_my_rating: bool = False) -> dict:
        """Convert a database row to a result dict."""
        # Column indices based on SELECT order
        result = {
            "id": row[0],
            "run_id": row[1],
            "model": row[2],
            "persona_id": row[3],
            "sector": row[4],
            "desired_role": row[5],
            "style": row[6],
            "input_type": row[7],
            "constraint_type": row[8],
            "chip_count": row[9],
            "cost_usd": row[18],
            "created_at": row[19],
            "avg_rating": row[20],
            "rating_count": row[21],
        }

        if full:
            result.update({
                "final_chips": json.loads(row[10]) if row[10] else [],
                "step1_chips": json.loads(row[11]) if row[11] else None,
                "selected_chips": json.loads(row[12]) if row[12] else None,
                "fill_chips": json.loads(row[13]) if row[13] else None,
                "errors": json.loads(row[14]) if row[14] else None,
                "latency_ms": row[15],
                "input_tokens": row[16],
                "output_tokens": row[17],
            })

        if include_my_rating:
            result["my_rating"] = row[22]

        return result
