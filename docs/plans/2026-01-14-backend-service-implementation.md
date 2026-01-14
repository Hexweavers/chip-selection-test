# Backend Service Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transform the chip generation benchmark into a backend service for hexweavers.io with Turso database and FastAPI Lambda.

**Architecture:** CLI writes test results to Turso (libSQL). FastAPI Lambda serves results and accepts ratings. Both share the `db/` module.

**Tech Stack:** Turso (libSQL), FastAPI, Mangum, Pydantic

---

## Task 1: Add Dependencies

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add new dependencies to pyproject.toml**

```toml
[project]
name = "chip-selection-test"
version = "0.1.0"
description = "Chip generation benchmark"
requires-python = ">=3.11"
dependencies = [
    "httpx>=0.27",
    "python-dotenv>=1.0",
    "tqdm>=4.66",
    "libsql-experimental>=0.0.47",
    "fastapi>=0.115",
    "mangum>=0.19",
    "pydantic>=2.10",
    "uvicorn>=0.34",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

**Step 2: Install dependencies**

Run: `uv sync`
Expected: Dependencies installed successfully

**Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add turso, fastapi, mangum dependencies"
```

---

## Task 2: Add Model Pricing Config

**Files:**
- Modify: `config.py`

**Step 1: Update ModelConfig dataclass with pricing**

In `config.py`, replace the `ModelConfig` dataclass and `MODELS` list:

```python
@dataclass
class ModelConfig:
    id: str
    name: str
    input_cost_per_m: float = 0.0   # $ per 1M input tokens
    output_cost_per_m: float = 0.0  # $ per 1M output tokens

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_cost_per_m +
            output_tokens * self.output_cost_per_m
        ) / 1_000_000


# Models to test (pricing from OpenRouter as of Jan 2026)
MODELS = [
    ModelConfig("anthropic/claude-haiku-4.5", "Claude Haiku 4.5", 1.0, 5.0),
    ModelConfig("openai/gpt-5-mini", "GPT-5 Mini", 0.15, 0.60),
    ModelConfig("meta-llama/llama-scout-4-12b", "Llama Scout 4 12B", 0.05, 0.10),
    ModelConfig("google/gemini-3-flash-preview", "Gemini 3 Flash Preview", 0.075, 0.30),
    ModelConfig("google/gemini-2.5-flash-lite", "Gemini 2.5 Flash Lite", 0.02, 0.08),
    ModelConfig("qwen/qwen3-next-80b-a3b-instruct", "Qwen3 Next 80B", 0.20, 0.60),
    ModelConfig("minimax/minimax-m2.1", "MiniMax M2.1", 0.10, 0.30),
    ModelConfig("deepseek/deepseek-v3.2", "DeepSeek V3.2", 0.14, 0.28),
    ModelConfig("x-ai/grok-4.1-fast", "Grok 4.1 Fast", 5.0, 15.0),
    ModelConfig("mistralai/mistral-nemo", "Mistral Nemo", 0.03, 0.10),
]


def get_model_config(model_id: str) -> ModelConfig | None:
    """Get model config by ID."""
    return next((m for m in MODELS if m.id == model_id), None)
```

**Step 2: Add Turso config**

Add at the end of `config.py`:

```python
# Database settings
TURSO_DATABASE_URL = os.getenv("TURSO_DATABASE_URL")
TURSO_AUTH_TOKEN = os.getenv("TURSO_AUTH_TOKEN")
```

**Step 3: Commit**

```bash
git add config.py
git commit -m "feat(config): add model pricing and turso settings"
```

---

## Task 3: Create Database Client

**Files:**
- Create: `db/__init__.py`
- Create: `db/client.py`

**Step 1: Create db package**

Create `db/__init__.py`:

```python
from db.client import get_db
from db.repository import Repository

__all__ = ["get_db", "Repository"]
```

**Step 2: Create database client**

Create `db/client.py`:

```python
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
```

**Step 3: Commit**

```bash
git add db/
git commit -m "feat(db): add turso client module"
```

---

## Task 4: Create Database Schema

**Files:**
- Create: `db/schema.sql`
- Modify: `db/client.py`

**Step 1: Create schema file**

Create `db/schema.sql`:

```sql
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT DEFAULT 'running',
    config TEXT
);

CREATE TABLE IF NOT EXISTS results (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id),
    model TEXT NOT NULL,
    persona_id TEXT NOT NULL,
    sector TEXT NOT NULL,
    desired_role TEXT NOT NULL,
    style TEXT NOT NULL,
    input_type TEXT NOT NULL,
    constraint_type TEXT NOT NULL,
    chip_count INTEGER NOT NULL,
    final_chips TEXT NOT NULL,
    step1_chips TEXT,
    selected_chips TEXT,
    fill_chips TEXT,
    errors TEXT,
    latency_ms INTEGER,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cost_usd REAL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ratings (
    id TEXT PRIMARY KEY,
    result_id TEXT NOT NULL REFERENCES results(id),
    user_id TEXT NOT NULL,
    rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
    created_at TEXT NOT NULL,
    UNIQUE(result_id, user_id)
);

CREATE INDEX IF NOT EXISTS idx_results_model ON results(model);
CREATE INDEX IF NOT EXISTS idx_results_run_id ON results(run_id);
CREATE INDEX IF NOT EXISTS idx_ratings_result_id ON ratings(result_id);
```

**Step 2: Add init_db function to client.py**

Add to `db/client.py`:

```python
from pathlib import Path


def init_db(db=None):
    """Initialize database schema."""
    if db is None:
        db = get_db()

    schema_path = Path(__file__).parent / "schema.sql"
    schema = schema_path.read_text()

    db.executescript(schema)
    db.commit()
    return db
```

Update `db/__init__.py`:

```python
from db.client import get_db, init_db
from db.repository import Repository

__all__ = ["get_db", "init_db", "Repository"]
```

**Step 3: Commit**

```bash
git add db/
git commit -m "feat(db): add schema and init_db function"
```

---

## Task 5: Create Repository - Runs

**Files:**
- Create: `db/repository.py`

**Step 1: Create repository with run operations**

Create `db/repository.py`:

```python
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
```

**Step 2: Commit**

```bash
git add db/repository.py
git commit -m "feat(db): add repository with run operations"
```

---

## Task 6: Add Repository - Results

**Files:**
- Modify: `db/repository.py`

**Step 1: Add result operations to Repository class**

Add these methods to the `Repository` class in `db/repository.py`:

```python
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
```

**Step 2: Commit**

```bash
git add db/repository.py
git commit -m "feat(db): add result operations to repository"
```

---

## Task 7: Add Repository - Ratings and Stats

**Files:**
- Modify: `db/repository.py`

**Step 1: Add ratings and stats methods to Repository**

Add these methods to the `Repository` class:

```python
    def add_rating(self, result_id: str, user_id: str, rating: int) -> str:
        """Add or update a rating. Returns rating ID."""
        rating_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        # Upsert rating
        self.db.execute(
            """
            INSERT INTO ratings (id, result_id, user_id, rating, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (result_id, user_id) DO UPDATE SET
                rating = excluded.rating,
                created_at = excluded.created_at
            """,
            (rating_id, result_id, user_id, rating, now),
        )
        self.db.commit()
        return rating_id

    def get_ratings(
        self,
        result_id: str | None = None,
        user_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """Get ratings with optional filters."""
        conditions = []
        params = []

        if result_id:
            conditions.append("result_id = ?")
            params.append(result_id)
        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.extend([limit, offset])

        rows = self.db.execute(
            f"""
            SELECT id, result_id, user_id, rating, created_at
            FROM ratings
            {where_clause}
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """,
            params,
        ).fetchall()

        return [
            {
                "id": row[0],
                "result_id": row[1],
                "user_id": row[2],
                "rating": row[3],
                "created_at": row[4],
            }
            for row in rows
        ]

    def get_stats(
        self,
        group_by: str = "model",
        run_id: str | None = None,
    ) -> list[dict]:
        """Get aggregated stats grouped by model, persona_id, style, or input_type."""
        valid_groups = {"model", "persona_id", "style", "input_type"}
        if group_by not in valid_groups:
            raise ValueError(f"group_by must be one of {valid_groups}")

        conditions = []
        params = []

        if run_id:
            conditions.append("r.run_id = ?")
            params.append(run_id)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        rows = self.db.execute(
            f"""
            SELECT r.{group_by},
                   COUNT(r.id) as result_count,
                   COUNT(rat.id) as rated_count,
                   AVG(rat.rating) as avg_rating,
                   SUM(r.cost_usd) as total_cost_usd,
                   AVG(r.latency_ms) as avg_latency_ms,
                   AVG(r.input_tokens) as avg_input_tokens,
                   AVG(r.output_tokens) as avg_output_tokens
            FROM results r
            LEFT JOIN ratings rat ON rat.result_id = r.id
            {where_clause}
            GROUP BY r.{group_by}
            ORDER BY avg_rating DESC NULLS LAST
            """,
            params,
        ).fetchall()

        return [
            {
                group_by: row[0],
                "result_count": row[1],
                "rated_count": row[2],
                "avg_rating": round(row[3], 2) if row[3] else None,
                "total_cost_usd": round(row[4], 6) if row[4] else 0,
                "avg_latency_ms": round(row[5]) if row[5] else None,
                "avg_tokens": {
                    "input": round(row[6]) if row[6] else None,
                    "output": round(row[7]) if row[7] else None,
                },
            }
            for row in rows
        ]
```

**Step 2: Commit**

```bash
git add db/repository.py
git commit -m "feat(db): add ratings and stats to repository"
```

---

## Task 8: Update Runner to Use Database

**Files:**
- Modify: `runner.py`

**Step 1: Update imports and add run management**

Replace the imports section and add run_id handling in `runner.py`:

```python
#!/usr/bin/env python3
"""
Chip Generation Test Runner

Usage:
    python runner.py --model anthropic/claude-haiku-4.5
    python runner.py --all
    python runner.py --model anthropic/claude-haiku-4.5 --persona tech_pm
    python runner.py --model openai/gpt-5-mini --resume
    python runner.py --model anthropic/claude-haiku-4.5 --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from typing import Optional

from tqdm import tqdm

from config import (
    MODELS,
    PROMPT_STYLES,
    CONSTRAINTS,
    INPUT_TYPES,
    CHIP_COUNTS,
    PERSONAS_FILE,
    MIN_CHIPS_PER_TYPE,
    get_model_config,
)
from models.chip import Chip, TestResult, TestMetadata
from services.llm import LLMClient
from services.generator import ChipGenerator
from services.selector import ChipSelector
from db import Repository, init_db
from utils.fill import FillService
```

**Step 2: Update run_test to return cost**

Modify the end of `run_test` function to calculate cost:

```python
    result.final_chips = unique_chips
    result.latency_ms = total_latency
    result.input_tokens = total_input_tokens
    result.output_tokens = total_output_tokens

    # Calculate cost
    model_config = get_model_config(model)
    if model_config:
        result.cost_usd = model_config.calculate_cost(total_input_tokens, total_output_tokens)

    return result
```

**Step 3: Add cost_usd to TestResult**

In `models/chip.py`, add `cost_usd` to `TestResult`:

```python
@dataclass
class TestResult:
    metadata: TestMetadata
    step1_chips: list[Chip] = field(default_factory=list)
    user_selected_chips: list[Chip] = field(default_factory=list)
    step2_chips: list[Chip] = field(default_factory=list)
    fill_chips: list[Chip] = field(default_factory=list)
    final_chips: list[Chip] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
```

And update `to_dict`:

```python
    def to_dict(self) -> dict:
        return {
            "metadata": self.metadata.to_dict(),
            "step1_chips": [c.to_dict() for c in self.step1_chips],
            "user_selected_chips": [c.to_dict() for c in self.user_selected_chips],
            "step2_chips": [c.to_dict() for c in self.step2_chips],
            "fill_chips": [c.to_dict() for c in self.fill_chips],
            "final_chips": [c.to_dict() for c in self.final_chips],
            "errors": self.errors,
            "latency_ms": self.latency_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
        }
```

**Step 4: Update run_model_batch to use Repository**

Replace the `run_model_batch` function signature and body:

```python
def run_model_batch(
    model_id: str,
    personas: list[dict],
    repo: Repository,
    run_id: str,
    generator: ChipGenerator,
    selector: ChipSelector,
    fill_service: FillService,
    resume: bool = False,
    dry_run: bool = False,
    persona_filter: str | None = None,
    pbar: tqdm | None = None,
):
    """Run all tests for a single model."""
    # Filter personas if specified
    if persona_filter:
        personas = [p for p in personas if p["id"] == persona_filter]
        if not personas:
            print(f"Error: Persona '{persona_filter}' not found")
            return

    # Generate all test combinations
    combinations = list(
        product(
            personas,
            PROMPT_STYLES,
            CONSTRAINTS,
            INPUT_TYPES,
            CHIP_COUNTS,
        )
    )

    skipped = 0

    for persona, style, constraint, input_type, chip_count in combinations:
        # Check if result exists (for resume)
        if resume and repo.result_exists(
            run_id, model_id, persona["id"], style, input_type, constraint, chip_count
        ):
            skipped += 1
            if pbar:
                pbar.update(1)
            continue

        # Update progress bar description
        desc = f"{model_id.split('/')[-1]} | {persona['id']} | {style[:5]} | {input_type[:5]} | {chip_count}"
        if pbar:
            pbar.set_description(desc)

        if dry_run:
            if pbar:
                pbar.update(1)
            continue

        result = run_test(
            model=model_id,
            persona=persona,
            style=style,
            constraint=constraint,
            input_type=input_type,
            chip_count=chip_count,
            generator=generator,
            selector=selector,
            fill_service=fill_service,
            dry_run=dry_run,
        )

        # Save result to database
        repo.save_result(
            run_id=run_id,
            model=model_id,
            persona_id=persona["id"],
            sector=persona["sector"],
            desired_role=persona["desired_role"],
            style=style,
            input_type=input_type,
            constraint_type=constraint,
            chip_count=chip_count,
            final_chips=[c.to_dict() for c in result.final_chips],
            step1_chips=[c.to_dict() for c in result.step1_chips] if result.step1_chips else None,
            selected_chips=[c.to_dict() for c in result.user_selected_chips] if result.user_selected_chips else None,
            fill_chips=[c.to_dict() for c in result.fill_chips] if result.fill_chips else None,
            errors=result.errors if result.errors else None,
            latency_ms=result.latency_ms,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cost_usd=result.cost_usd,
        )

        # Update progress bar with result info
        counts = result.count_by_type()
        if result.errors:
            if pbar:
                pbar.set_postfix(
                    chips=len(result.final_chips), errors=len(result.errors)
                )
        else:
            fill_str = f"+{len(result.fill_chips)}" if result.fill_chips else ""
            if pbar:
                pbar.set_postfix(
                    chips=len(result.final_chips),
                    S=counts["situation"],
                    J=counts["jargon"],
                    R=counts["role_task"],
                    E=counts["environment"],
                    fill=fill_str or None,
                )

        if pbar:
            pbar.update(1)

    return skipped
```

**Step 5: Update main() to use Repository**

Replace the `main()` function:

```python
def main():
    parser = argparse.ArgumentParser(description="Run chip generation tests")
    parser.add_argument(
        "--model", type=str, help="Model ID to test (e.g., anthropic/claude-haiku-4.5)"
    )
    parser.add_argument("--all", action="store_true", help="Run all models")
    parser.add_argument(
        "--persona", type=str, help="Run only specific persona (e.g., tech_pm)"
    )
    parser.add_argument("--resume", type=str, help="Resume a previous run by ID")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would run without making API calls",
    )
    parser.add_argument(
        "--list-models", action="store_true", help="List available models"
    )

    args = parser.parse_args()

    if args.list_models:
        print("\nAvailable models:")
        for m in MODELS:
            print(f"  {m.id} ({m.name})")
        return

    if not args.model and not args.all:
        parser.print_help()
        print("\nError: Must specify --model or --all")
        sys.exit(1)

    # Load personas
    personas = load_personas()

    # Filter personas for count calculation
    filtered_personas = personas
    if args.persona:
        filtered_personas = [p for p in personas if p["id"] == args.persona]

    # Calculate total tests
    tests_per_persona = (
        len(PROMPT_STYLES) * len(CONSTRAINTS) * len(INPUT_TYPES) * len(CHIP_COUNTS)
    )
    if args.all:
        total_tests = len(MODELS) * len(filtered_personas) * tests_per_persona
        models_to_run = MODELS
    else:
        total_tests = len(filtered_personas) * tests_per_persona
        models_to_run = [m for m in MODELS if m.id == args.model]

    # Initialize services (skip LLM client for dry run)
    llm_client = None
    generator = None
    selector = None
    fill_service = None

    if not args.dry_run:
        llm_client = LLMClient()
        generator = ChipGenerator(llm_client)
        selector = ChipSelector(llm_client)
        fill_service = FillService(llm_client)

    # Initialize database
    db = init_db()
    repo = Repository(db)

    # Create or resume run
    if args.resume:
        run_id = args.resume
        run = repo.get_run(run_id)
        if not run:
            print(f"Error: Run '{run_id}' not found")
            sys.exit(1)
        print(f"Resuming run: {run_id}")
    else:
        config = {
            "models": [m.id for m in models_to_run],
            "persona_filter": args.persona,
        }
        run_id = repo.create_run(config)
        print(f"Created run: {run_id}")

    try:
        with tqdm(total=total_tests, desc="Starting...", unit="test") as pbar:
            for model in models_to_run:
                run_model_batch(
                    model_id=model.id,
                    personas=personas,
                    repo=repo,
                    run_id=run_id,
                    generator=generator,
                    selector=selector,
                    fill_service=fill_service,
                    resume=bool(args.resume),
                    dry_run=args.dry_run,
                    persona_filter=args.persona,
                    pbar=pbar,
                )

        # Mark run as completed
        if not args.dry_run:
            repo.complete_run(run_id)
            print(f"\nCompleted run: {run_id}")

    except KeyboardInterrupt:
        print(f"\nRun interrupted. Resume with: --resume {run_id}")
        sys.exit(1)
    finally:
        if llm_client:
            llm_client.close()


if __name__ == "__main__":
    main()
```

**Step 6: Commit**

```bash
git add runner.py models/chip.py
git commit -m "feat(runner): use database instead of JSON files"
```

---

## Task 9: Create FastAPI App Structure

**Files:**
- Create: `api/__init__.py`
- Create: `api/main.py`

**Step 1: Create api package**

Create `api/__init__.py`:

```python
from api.main import app

__all__ = ["app"]
```

**Step 2: Create main FastAPI app**

Create `api/main.py`:

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import runs, results, ratings, stats

app = FastAPI(
    title="Chip Benchmark API",
    description="Backend for chip generation benchmark results and ratings",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Internal VPC only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(runs.router, prefix="/runs", tags=["runs"])
app.include_router(results.router, prefix="/results", tags=["results"])
app.include_router(ratings.router, prefix="/ratings", tags=["ratings"])
app.include_router(stats.router, prefix="/stats", tags=["stats"])


@app.get("/health")
def health():
    return {"status": "ok"}
```

**Step 3: Create routes package**

Create `api/routes/__init__.py`:

```python
```

**Step 4: Commit**

```bash
git add api/
git commit -m "feat(api): create fastapi app structure"
```

---

## Task 10: Create API Routes - Runs

**Files:**
- Create: `api/routes/runs.py`

**Step 1: Create runs router**

Create `api/routes/runs.py`:

```python
from fastapi import APIRouter, HTTPException, Query

from db import Repository, init_db

router = APIRouter()


def get_repo():
    db = init_db()
    return Repository(db)


@router.get("")
def list_runs(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """List all test runs."""
    repo = get_repo()
    runs = repo.list_runs(limit=limit, offset=offset)
    return {"runs": runs}


@router.get("/{run_id}")
def get_run(run_id: str):
    """Get a single run by ID."""
    repo = get_repo()
    run = repo.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return {
        "id": run.id,
        "started_at": run.started_at,
        "completed_at": run.completed_at,
        "status": run.status,
        "config": run.config,
    }
```

**Step 2: Commit**

```bash
git add api/routes/runs.py
git commit -m "feat(api): add runs endpoints"
```

---

## Task 11: Create API Routes - Results

**Files:**
- Create: `api/routes/results.py`

**Step 1: Create results router**

Create `api/routes/results.py`:

```python
from fastapi import APIRouter, HTTPException, Query, Header
from typing import Optional

from db import Repository, init_db

router = APIRouter()


def get_repo():
    db = init_db()
    return Repository(db)


@router.get("")
def list_results(
    run_id: Optional[str] = Query(None),
    model: Optional[str] = Query(None),
    persona_id: Optional[str] = Query(None),
    rated_by: Optional[str] = Query(None),
    unrated_by: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    x_user_id: Optional[str] = Header(None),
):
    """List results with filters."""
    repo = get_repo()
    results, total = repo.list_results(
        run_id=run_id,
        model=model,
        persona_id=persona_id,
        rated_by=rated_by,
        unrated_by=unrated_by,
        limit=limit,
        offset=offset,
        user_id=x_user_id,
    )
    return {"results": results, "total": total}


@router.get("/{result_id}")
def get_result(result_id: str):
    """Get a single result with full details."""
    repo = get_repo()
    result = repo.get_result(result_id)
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")
    return result
```

**Step 2: Commit**

```bash
git add api/routes/results.py
git commit -m "feat(api): add results endpoints"
```

---

## Task 12: Create API Routes - Ratings

**Files:**
- Create: `api/routes/ratings.py`
- Modify: `api/routes/results.py`

**Step 1: Create ratings router**

Create `api/routes/ratings.py`:

```python
from fastapi import APIRouter, Query
from typing import Optional

from db import Repository, init_db

router = APIRouter()


def get_repo():
    db = init_db()
    return Repository(db)


@router.get("")
def list_ratings(
    result_id: Optional[str] = Query(None),
    user_id: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """List ratings with filters."""
    repo = get_repo()
    ratings = repo.get_ratings(
        result_id=result_id,
        user_id=user_id,
        limit=limit,
        offset=offset,
    )
    return {"ratings": ratings}
```

**Step 2: Add rating submission to results router**

Add to `api/routes/results.py`:

```python
from pydantic import BaseModel, Field


class RatingCreate(BaseModel):
    rating: int = Field(..., ge=1, le=5)


@router.post("/{result_id}/ratings")
def create_rating(
    result_id: str,
    body: RatingCreate,
    x_user_id: str = Header(...),
):
    """Submit a rating for a result."""
    repo = get_repo()

    # Check result exists
    result = repo.get_result(result_id)
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")

    rating_id = repo.add_rating(result_id, x_user_id, body.rating)

    return {
        "id": rating_id,
        "result_id": result_id,
        "user_id": x_user_id,
        "rating": body.rating,
    }
```

**Step 3: Commit**

```bash
git add api/routes/ratings.py api/routes/results.py
git commit -m "feat(api): add ratings endpoints"
```

---

## Task 13: Create API Routes - Stats

**Files:**
- Create: `api/routes/stats.py`

**Step 1: Create stats router**

Create `api/routes/stats.py`:

```python
from fastapi import APIRouter, Query, HTTPException
from typing import Optional, Literal

from db import Repository, init_db

router = APIRouter()


def get_repo():
    db = init_db()
    return Repository(db)


@router.get("")
def get_stats(
    group_by: Literal["model", "persona_id", "style", "input_type"] = Query("model"),
    run_id: Optional[str] = Query(None),
):
    """Get aggregated stats for dashboard/leaderboard."""
    repo = get_repo()
    try:
        stats = repo.get_stats(group_by=group_by, run_id=run_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"stats": stats}
```

**Step 2: Commit**

```bash
git add api/routes/stats.py
git commit -m "feat(api): add stats endpoint for leaderboard"
```

---

## Task 14: Create Lambda Handler

**Files:**
- Create: `api/lambda_handler.py`

**Step 1: Create Lambda handler with Mangum**

Create `api/lambda_handler.py`:

```python
from mangum import Mangum
from api.main import app

handler = Mangum(app, lifespan="off")
```

**Step 2: Commit**

```bash
git add api/lambda_handler.py
git commit -m "feat(api): add lambda handler with mangum"
```

---

## Task 15: Update .env.example and .gitignore

**Files:**
- Create: `.env.example`
- Modify: `.gitignore`

**Step 1: Create .env.example**

Create `.env.example`:

```bash
# OpenRouter API
OPENROUTER_API_KEY=your_openrouter_api_key

# Turso Database
TURSO_DATABASE_URL=libsql://your-database.turso.io
TURSO_AUTH_TOKEN=your_turso_auth_token
```

**Step 2: Update .gitignore**

Add to `.gitignore`:

```
# Environment
.env

# Database
*.db

# Results (legacy)
results/
```

**Step 3: Commit**

```bash
git add .env.example .gitignore
git commit -m "chore: add env example and update gitignore"
```

---

## Task 16: Clean Up Legacy Storage

**Files:**
- Delete: `utils/storage.py`
- Modify: `utils/__init__.py`

**Step 1: Remove legacy storage module**

Delete `utils/storage.py`:

```bash
rm utils/storage.py
```

**Step 2: Update utils/__init__.py if needed**

If `utils/__init__.py` imports `ResultStorage`, remove that import.

**Step 3: Commit**

```bash
git add -A
git commit -m "chore: remove legacy JSON storage module"
```

---

## Task 17: Final Integration Test

**Step 1: Set up Turso database**

```bash
# Install turso CLI if not installed
# Create database at turso.tech and get credentials
# Add to .env:
# TURSO_DATABASE_URL=libsql://your-db.turso.io
# TURSO_AUTH_TOKEN=your_token
```

**Step 2: Run a dry-run test**

```bash
python runner.py --model anthropic/claude-haiku-4.5 --persona tech_pm --dry-run
```

Expected: Should create a run and complete without errors.

**Step 3: Start API locally**

```bash
uvicorn api.main:app --reload
```

**Step 4: Test endpoints**

```bash
curl http://localhost:8000/health
curl http://localhost:8000/runs
curl http://localhost:8000/stats
```

Expected: All return valid JSON responses.

**Step 5: Final commit**

```bash
git add -A
git commit -m "feat: complete backend service implementation"
```

---

## Summary

| Task | Description |
|------|-------------|
| 1 | Add dependencies (turso, fastapi, mangum) |
| 2 | Add model pricing config |
| 3 | Create database client |
| 4 | Create database schema |
| 5 | Repository - runs |
| 6 | Repository - results |
| 7 | Repository - ratings & stats |
| 8 | Update runner to use database |
| 9 | FastAPI app structure |
| 10 | API routes - runs |
| 11 | API routes - results |
| 12 | API routes - ratings |
| 13 | API routes - stats |
| 14 | Lambda handler |
| 15 | Environment config |
| 16 | Clean up legacy storage |
| 17 | Integration test |
