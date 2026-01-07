# TUI Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transform the chip benchmark CLI into an interactive Textual TUI with configure, monitor, and results screens.

**Architecture:** Textual app with tabbed navigation. Async LLM client for non-blocking API calls. SQLite database for persisting runs, results, and ratings. Event-driven communication between runner and UI via Textual message passing.

**Tech Stack:** Textual (TUI), SQLite (storage), httpx with AsyncClient (async HTTP), pytest (testing)

---

## Task 1: Project Setup

**Files:**
- Modify: `pyproject.toml`
- Create: `tui/__init__.py`
- Create: `db/__init__.py`

**Step 1: Add dependencies to pyproject.toml**

Edit `pyproject.toml` to add textual:

```toml
[project]
name = "chip-selection-test"
version = "0.1.0"
description = "Add your description here"
requires-python = ">=3.13"
dependencies = [
    "httpx>=0.28.1",
    "python-dotenv>=1.2.1",
    "tqdm>=4.67.1",
    "textual>=0.89.1",
]

[dependency-groups]
dev = [
    "black>=25.12.0",
    "pytest>=8.0.0",
    "pytest-asyncio>=0.24.0",
]
```

**Step 2: Create package directories**

```bash
mkdir -p tui/screens tui/widgets db
touch tui/__init__.py tui/screens/__init__.py tui/widgets/__init__.py db/__init__.py
```

**Step 3: Install dependencies**

Run: `uv sync`

**Step 4: Commit**

```bash
git add -A
git commit -m "chore: add textual and pytest dependencies, create package structure"
```

---

## Task 2: Database Schema

**Files:**
- Create: `db/schema.py`
- Create: `tests/test_db_schema.py`

**Step 1: Write the failing test**

Create `tests/test_db_schema.py`:

```python
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_db_schema.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'db.schema'"

**Step 3: Write minimal implementation**

Create `db/schema.py`:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_db_schema.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add db/schema.py tests/test_db_schema.py
git commit -m "feat(db): add SQLite schema for runs and results"
```

---

## Task 3: Database Repository

**Files:**
- Create: `db/repository.py`
- Create: `tests/test_db_repository.py`

**Step 1: Write the failing test**

Create `tests/test_db_repository.py`:

```python
import tempfile
from pathlib import Path
from datetime import datetime
import json

import pytest

from db.schema import init_db
from db.repository import Repository


@pytest.fixture
def repo():
    """Create a repository with a temporary database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        conn = init_db(db_path)
        yield Repository(conn)
        conn.close()


def test_create_run(repo):
    """Should create a run and return its ID."""
    run_id = repo.create_run(
        name="Test Run",
        persona="tech_pm",
        prompt_style="guided",
        flow="enriched",
        constraint_type="2-per-type",
        chip_count=35,
    )
    assert run_id is not None
    assert len(run_id) == 36  # UUID format


def test_get_run(repo):
    """Should retrieve a run by ID."""
    run_id = repo.create_run(
        name="Test Run",
        persona="tech_pm",
        prompt_style="guided",
        flow="enriched",
        constraint_type="2-per-type",
        chip_count=35,
    )
    run = repo.get_run(run_id)
    assert run["name"] == "Test Run"
    assert run["persona"] == "tech_pm"


def test_create_result(repo):
    """Should create a result linked to a run."""
    run_id = repo.create_run(
        name="Test",
        persona="tech_pm",
        prompt_style="guided",
        flow="enriched",
        constraint_type="2-per-type",
        chip_count=35,
    )
    chips = [{"key": "test", "display": "Test", "type": "situation"}]
    result_id = repo.create_result(
        run_id=run_id,
        model="claude-haiku",
        chips=chips,
        tokens_in=100,
        tokens_out=200,
        cost_usd=0.001,
        latency_ms=500,
        situation_count=1,
        jargon_count=0,
        role_task_count=0,
        environment_count=0,
    )
    assert result_id is not None


def test_update_rating(repo):
    """Should update the rating for a result."""
    run_id = repo.create_run(
        name="Test",
        persona="tech_pm",
        prompt_style="guided",
        flow="enriched",
        constraint_type="2-per-type",
        chip_count=35,
    )
    result_id = repo.create_result(
        run_id=run_id,
        model="claude-haiku",
        chips=[],
        tokens_in=100,
        tokens_out=200,
        cost_usd=0.001,
        latency_ms=500,
        situation_count=0,
        jargon_count=0,
        role_task_count=0,
        environment_count=0,
    )
    repo.update_rating(result_id, 5)
    result = repo.get_result(result_id)
    assert result["rating"] == 5
    assert result["rated_at"] is not None


def test_list_runs(repo):
    """Should list runs grouped by date."""
    repo.create_run(
        name="Run 1",
        persona="tech_pm",
        prompt_style="guided",
        flow="enriched",
        constraint_type="2-per-type",
        chip_count=35,
    )
    repo.create_run(
        name="Run 2",
        persona="tech_swe",
        prompt_style="terse",
        flow="basic",
        constraint_type="none",
        chip_count=15,
    )
    runs = repo.list_runs()
    assert len(runs) == 2


def test_get_results_for_run(repo):
    """Should get all results for a run."""
    run_id = repo.create_run(
        name="Test",
        persona="tech_pm",
        prompt_style="guided",
        flow="enriched",
        constraint_type="2-per-type",
        chip_count=35,
    )
    repo.create_result(
        run_id=run_id,
        model="claude-haiku",
        chips=[],
        tokens_in=100,
        tokens_out=200,
        cost_usd=0.001,
        latency_ms=500,
        situation_count=0,
        jargon_count=0,
        role_task_count=0,
        environment_count=0,
    )
    repo.create_result(
        run_id=run_id,
        model="gpt-5-mini",
        chips=[],
        tokens_in=150,
        tokens_out=250,
        cost_usd=0.002,
        latency_ms=600,
        situation_count=0,
        jargon_count=0,
        role_task_count=0,
        environment_count=0,
    )
    results = repo.get_results_for_run(run_id)
    assert len(results) == 2
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_db_repository.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'db.repository'"

**Step 3: Write minimal implementation**

Create `db/repository.py`:

```python
"""Repository for database operations."""

from datetime import datetime
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
        created_at = datetime.utcnow().isoformat()
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
        rated_at = datetime.utcnow().isoformat()
        self.conn.execute(
            "UPDATE results SET rating = ?, rated_at = ? WHERE id = ?",
            (rating, rated_at, result_id),
        )
        self.conn.commit()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_db_repository.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add db/repository.py tests/test_db_repository.py
git commit -m "feat(db): add repository with CRUD operations for runs and results"
```

---

## Task 4: Async LLM Client

**Files:**
- Create: `services/llm_async.py`
- Create: `tests/test_llm_async.py`

**Step 1: Write the failing test**

Create `tests/test_llm_async.py`:

```python
import pytest

from services.llm_async import AsyncLLMClient, LLMResponse


@pytest.mark.asyncio
async def test_async_client_returns_llm_response():
    """AsyncLLMClient.chat should return an LLMResponse."""
    # We'll test with a mock/dry-run approach since we don't want real API calls
    # The actual test verifies the interface exists
    client = AsyncLLMClient()
    # Just verify the client can be instantiated and has the right methods
    assert hasattr(client, "chat")
    assert hasattr(client, "close")
    await client.close()


@pytest.mark.asyncio
async def test_llm_response_dataclass():
    """LLMResponse should have expected fields."""
    response = LLMResponse(
        content="test",
        input_tokens=10,
        output_tokens=20,
        latency_ms=100,
        error=None,
    )
    assert response.content == "test"
    assert response.input_tokens == 10
    assert response.output_tokens == 20
    assert response.latency_ms == 100
    assert response.error is None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm_async.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'services.llm_async'"

**Step 3: Write minimal implementation**

Create `services/llm_async.py`:

```python
"""Async LLM client for non-blocking API calls."""

from __future__ import annotations

import time
from dataclasses import dataclass

import httpx

from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL


@dataclass
class LLMResponse:
    content: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    error: str | None = None


class AsyncLLMClient:
    def __init__(self):
        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY environment variable is required")
        self.client = httpx.AsyncClient(
            base_url=OPENROUTER_BASE_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )

    async def chat(self, model: str, system: str, user: str) -> LLMResponse:
        """Send a chat completion request asynchronously."""
        start = time.time()
        try:
            response = await self.client.post(
                "/chat/completions",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "temperature": 0.7,
                },
            )
            latency_ms = int((time.time() - start) * 1000)

            if response.status_code != 200:
                return LLMResponse(
                    content="",
                    input_tokens=0,
                    output_tokens=0,
                    latency_ms=latency_ms,
                    error=f"HTTP {response.status_code}: {response.text}",
                )

            data = response.json()

            if "error" in data:
                return LLMResponse(
                    content="",
                    input_tokens=0,
                    output_tokens=0,
                    latency_ms=latency_ms,
                    error=data["error"].get("message", str(data["error"])),
                )

            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})

            return LLMResponse(
                content=content,
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
                latency_ms=latency_ms,
            )

        except httpx.TimeoutException:
            latency_ms = int((time.time() - start) * 1000)
            return LLMResponse(
                content="",
                input_tokens=0,
                output_tokens=0,
                latency_ms=latency_ms,
                error="Request timed out",
            )
        except Exception as e:
            latency_ms = int((time.time() - start) * 1000)
            return LLMResponse(
                content="",
                input_tokens=0,
                output_tokens=0,
                latency_ms=latency_ms,
                error=str(e),
            )

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_llm_async.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add services/llm_async.py tests/test_llm_async.py
git commit -m "feat(services): add async LLM client for non-blocking API calls"
```

---

## Task 5: Async Test Runner

**Files:**
- Create: `services/runner_async.py`
- Create: `tests/test_runner_async.py`

**Step 1: Write the failing test**

Create `tests/test_runner_async.py`:

```python
import pytest
from dataclasses import dataclass
from typing import AsyncIterator

from services.runner_async import AsyncRunner, RunConfig, RunEvent, EventType


def test_run_config_creation():
    """RunConfig should hold test configuration."""
    config = RunConfig(
        models=["claude-haiku", "gpt-5-mini"],
        persona_id="tech_pm",
        prompt_style="guided",
        flow="enriched",
        constraint_type="2-per-type",
        chip_count=35,
    )
    assert config.models == ["claude-haiku", "gpt-5-mini"]
    assert config.is_head_to_head is True


def test_run_config_single_model():
    """RunConfig with one model should not be head-to-head."""
    config = RunConfig(
        models=["claude-haiku"],
        persona_id="tech_pm",
        prompt_style="guided",
        flow="enriched",
        constraint_type="2-per-type",
        chip_count=35,
    )
    assert config.is_head_to_head is False


def test_run_event_types():
    """RunEvent should have expected event types."""
    assert EventType.LOG is not None
    assert EventType.PROGRESS is not None
    assert EventType.COMPLETE is not None
    assert EventType.ERROR is not None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_runner_async.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'services.runner_async'"

**Step 3: Write minimal implementation**

Create `services/runner_async.py`:

```python
"""Async test runner with event streaming for TUI integration."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import AsyncIterator, Callable

from config import PROMPTS_FILE, PERSONAS_FILE, MIN_CHIPS_PER_TYPE, CHIP_TYPES
from models.chip import Chip, parse_chips_from_json
from services.llm_async import AsyncLLMClient, LLMResponse


class EventType(Enum):
    LOG = auto()
    PROGRESS = auto()
    COMPLETE = auto()
    ERROR = auto()


@dataclass
class RunEvent:
    type: EventType
    model: str
    message: str
    data: dict = field(default_factory=dict)


@dataclass
class RunConfig:
    models: list[str]
    persona_id: str
    prompt_style: str
    flow: str
    constraint_type: str
    chip_count: int

    @property
    def is_head_to_head(self) -> bool:
        return len(self.models) == 2


@dataclass
class ModelResult:
    model: str
    chips: list[dict]
    tokens_in: int
    tokens_out: int
    latency_ms: int
    cost_usd: float
    situation_count: int
    jargon_count: int
    role_task_count: int
    environment_count: int
    errors: list[str] = field(default_factory=list)


def load_personas() -> dict[str, dict]:
    """Load personas indexed by ID."""
    with open(PERSONAS_FILE) as f:
        data = json.load(f)
    return {p["id"]: p for p in data["personas"]}


def load_prompts() -> dict:
    """Load prompt templates."""
    with open(PROMPTS_FILE) as f:
        return json.load(f)["styles"]


class AsyncRunner:
    """Async test runner that yields events for TUI consumption."""

    def __init__(self):
        self.client: AsyncLLMClient | None = None
        self.prompts = load_prompts()
        self.personas = load_personas()
        self._cancelled = False

    async def start(self):
        """Initialize the LLM client."""
        self.client = AsyncLLMClient()

    async def stop(self):
        """Close the LLM client."""
        if self.client:
            await self.client.close()
            self.client = None

    def cancel(self):
        """Signal cancellation."""
        self._cancelled = True

    async def run(self, config: RunConfig) -> AsyncIterator[RunEvent]:
        """Run tests and yield events."""
        self._cancelled = False

        if not self.client:
            await self.start()

        persona = self.personas.get(config.persona_id)
        if not persona:
            yield RunEvent(
                type=EventType.ERROR,
                model="",
                message=f"Persona not found: {config.persona_id}",
            )
            return

        # Run models in parallel
        tasks = [
            self._run_model(config, model, persona)
            for model in config.models
        ]

        # Collect events from all tasks
        queues: dict[str, asyncio.Queue] = {
            model: asyncio.Queue() for model in config.models
        }
        results: dict[str, ModelResult] = {}

        async def run_and_queue(model: str, coro):
            async for event in coro:
                await queues[model].put(event)
            await queues[model].put(None)  # Sentinel

        # Start all model runs
        runner_tasks = [
            asyncio.create_task(run_and_queue(model, self._run_model(config, model, persona)))
            for model in config.models
        ]

        # Yield events as they come in
        active_queues = set(config.models)
        while active_queues and not self._cancelled:
            for model in list(active_queues):
                try:
                    event = queues[model].get_nowait()
                    if event is None:
                        active_queues.remove(model)
                    else:
                        yield event
                        if event.type == EventType.COMPLETE:
                            results[model] = event.data.get("result")
                except asyncio.QueueEmpty:
                    pass
            await asyncio.sleep(0.01)  # Small delay to prevent busy loop

        # Wait for tasks to complete
        for task in runner_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def _run_model(
        self, config: RunConfig, model: str, persona: dict
    ) -> AsyncIterator[RunEvent]:
        """Run a single model and yield events."""
        total_tokens_in = 0
        total_tokens_out = 0
        total_latency = 0
        errors = []
        all_chips = []

        yield RunEvent(
            type=EventType.LOG,
            model=model,
            message="Starting test run",
        )
        yield RunEvent(
            type=EventType.PROGRESS,
            model=model,
            message="Initializing",
            data={"percent": 0, "step": "init"},
        )

        if self._cancelled:
            return

        # Enriched flow: Step 1 + Selection + Step 2
        if config.flow == "enriched":
            # Step 1
            yield RunEvent(
                type=EventType.LOG,
                model=model,
                message="Step 1: Generating selectable chips",
            )
            yield RunEvent(
                type=EventType.PROGRESS,
                model=model,
                message="Step 1: Generating selectable chips",
                data={"percent": 10, "step": "step1"},
            )

            step1_chips, step1_response = await self._generate_step1(
                model, persona, config.prompt_style, config.constraint_type
            )
            total_tokens_in += step1_response.input_tokens
            total_tokens_out += step1_response.output_tokens
            total_latency += step1_response.latency_ms

            if step1_response.error:
                errors.append(f"Step 1: {step1_response.error}")
                yield RunEvent(
                    type=EventType.ERROR,
                    model=model,
                    message=f"Step 1 error: {step1_response.error}",
                )

            yield RunEvent(
                type=EventType.LOG,
                model=model,
                message=f"Got {len(step1_chips)} selectable chips",
            )

            if self._cancelled:
                return

            # Selection
            yield RunEvent(
                type=EventType.LOG,
                model=model,
                message="Selecting chips as user",
            )
            yield RunEvent(
                type=EventType.PROGRESS,
                model=model,
                message="Selecting chips",
                data={"percent": 30, "step": "selection"},
            )

            selected_chips, select_response = await self._select_chips(
                step1_chips, persona["persona"], config.prompt_style
            )
            total_tokens_in += select_response.input_tokens
            total_tokens_out += select_response.output_tokens
            total_latency += select_response.latency_ms

            if select_response.error:
                errors.append(f"Selection: {select_response.error}")

            yield RunEvent(
                type=EventType.LOG,
                model=model,
                message=f"Selected {len(selected_chips)} chips",
            )

            if self._cancelled:
                return

            # Step 2 (enriched)
            yield RunEvent(
                type=EventType.LOG,
                model=model,
                message="Step 2: Generating final chips",
            )
            yield RunEvent(
                type=EventType.PROGRESS,
                model=model,
                message="Step 2: Generating final chips",
                data={"percent": 60, "step": "step2"},
            )

            step2_chips, step2_response = await self._generate_step2_enriched(
                model, persona, config.prompt_style, config.chip_count, selected_chips
            )
            total_tokens_in += step2_response.input_tokens
            total_tokens_out += step2_response.output_tokens
            total_latency += step2_response.latency_ms

            if step2_response.error:
                errors.append(f"Step 2: {step2_response.error}")

            # Merge chips
            all_chips = selected_chips + step2_chips

        else:
            # Basic flow: Step 2 only
            yield RunEvent(
                type=EventType.LOG,
                model=model,
                message="Step 2: Generating chips (basic flow)",
            )
            yield RunEvent(
                type=EventType.PROGRESS,
                model=model,
                message="Step 2: Generating chips",
                data={"percent": 50, "step": "step2"},
            )

            step2_chips, step2_response = await self._generate_step2_basic(
                model, persona, config.prompt_style, config.chip_count
            )
            total_tokens_in += step2_response.input_tokens
            total_tokens_out += step2_response.output_tokens
            total_latency += step2_response.latency_ms

            if step2_response.error:
                errors.append(f"Step 2: {step2_response.error}")

            all_chips = step2_chips

        if self._cancelled:
            return

        yield RunEvent(
            type=EventType.LOG,
            model=model,
            message=f"Generated {len(all_chips)} chips",
        )

        # Deduplicate
        seen_keys = set()
        unique_chips = []
        for chip in all_chips:
            if chip.key not in seen_keys:
                seen_keys.add(chip.key)
                unique_chips.append(chip)

        # Check coverage and fill if needed
        if config.constraint_type == "2-per-type":
            yield RunEvent(
                type=EventType.PROGRESS,
                model=model,
                message="Checking type coverage",
                data={"percent": 80, "step": "fill"},
            )

            missing = self._get_missing_types(unique_chips)
            if missing:
                yield RunEvent(
                    type=EventType.LOG,
                    model=model,
                    message=f"Filling missing types: {missing}",
                )

                fill_chips, fill_response = await self._fill_missing(
                    model, persona, unique_chips, missing, config.prompt_style
                )
                total_tokens_in += fill_response.input_tokens
                total_tokens_out += fill_response.output_tokens
                total_latency += fill_response.latency_ms

                if fill_response.error:
                    errors.append(f"Fill: {fill_response.error}")

                for chip in fill_chips:
                    if chip.key not in seen_keys:
                        seen_keys.add(chip.key)
                        unique_chips.append(chip)

        # Count by type
        counts = {t: 0 for t in CHIP_TYPES}
        for chip in unique_chips:
            if chip.type in counts:
                counts[chip.type] += 1

        # Estimate cost (rough approximation)
        cost_usd = (total_tokens_in * 0.001 + total_tokens_out * 0.002) / 1000

        result = ModelResult(
            model=model,
            chips=[{"key": c.key, "display": c.display, "type": c.type} for c in unique_chips],
            tokens_in=total_tokens_in,
            tokens_out=total_tokens_out,
            latency_ms=total_latency,
            cost_usd=cost_usd,
            situation_count=counts["situation"],
            jargon_count=counts["jargon"],
            role_task_count=counts["role_task"],
            environment_count=counts["environment"],
            errors=errors,
        )

        yield RunEvent(
            type=EventType.PROGRESS,
            model=model,
            message="Complete",
            data={"percent": 100, "step": "done"},
        )
        yield RunEvent(
            type=EventType.LOG,
            model=model,
            message=f"Finished: {len(unique_chips)} chips, {total_latency}ms, ${cost_usd:.4f}",
        )
        yield RunEvent(
            type=EventType.COMPLETE,
            model=model,
            message="Run complete",
            data={"result": result},
        )

    def _get_prompt(self, style: str, key: str) -> dict:
        return self.prompts[style][key]

    def _format_prompt(self, template: str, **kwargs) -> str:
        result = template
        for key, value in kwargs.items():
            placeholder = "{" + key + "}"
            if placeholder in result:
                result = result.replace(placeholder, str(value))
        return result

    async def _generate_step1(
        self, model: str, persona: dict, style: str, constraint: str
    ) -> tuple[list[Chip], LLMResponse]:
        prompt_key = (
            "step1_user_selectable_with_constraint"
            if constraint == "with_constraint" or constraint == "2-per-type"
            else "step1_user_selectable"
        )
        prompt = self._get_prompt(style, prompt_key)
        system = prompt["system"]
        user = self._format_prompt(
            prompt["user"],
            sector=persona["sector"],
            desired_role=persona["desired_role"],
        )
        response = await self.client.chat(model, system, user)
        if response.error:
            return [], response
        chips, _ = parse_chips_from_json(response.content)
        return chips, response

    async def _select_chips(
        self, chips: list[Chip], persona_text: str, style: str
    ) -> tuple[list[Chip], LLMResponse]:
        from config import SELECTOR_MODEL
        prompt = self._get_prompt(style, "chip_selection")
        chips_json = json.dumps([{"key": c.key, "display": c.display, "type": c.type} for c in chips])
        system = prompt["system"]
        user = self._format_prompt(prompt["user"], persona=persona_text, available_chips=chips_json)
        response = await self.client.chat(SELECTOR_MODEL.id, system, user)
        if response.error:
            return [], response

        # Parse selected keys
        try:
            content = response.content.strip()
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1])
            selected_keys = json.loads(content)
            if isinstance(selected_keys, dict) and "selected" in selected_keys:
                selected_keys = selected_keys["selected"]
            selected = [c for c in chips if c.key in selected_keys]
            return selected, response
        except:
            return [], LLMResponse(
                content="",
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                latency_ms=response.latency_ms,
                error="Failed to parse selection response",
            )

    async def _generate_step2_basic(
        self, model: str, persona: dict, style: str, chip_count: int
    ) -> tuple[list[Chip], LLMResponse]:
        prompt = self._get_prompt(style, "step2_final_generation_basic")
        system = prompt["system"]
        user = self._format_prompt(
            prompt["user"],
            sector=persona["sector"],
            desired_role=persona["desired_role"],
            chip_count=chip_count,
        )
        response = await self.client.chat(model, system, user)
        if response.error:
            return [], response
        chips, _ = parse_chips_from_json(response.content)
        return chips, response

    async def _generate_step2_enriched(
        self, model: str, persona: dict, style: str, chip_count: int, selected_chips: list[Chip]
    ) -> tuple[list[Chip], LLMResponse]:
        prompt = self._get_prompt(style, "step2_final_generation")
        selected_json = json.dumps([{"key": c.key, "display": c.display, "type": c.type} for c in selected_chips])
        system = prompt["system"]
        user = self._format_prompt(
            prompt["user"],
            sector=persona["sector"],
            desired_role=persona["desired_role"],
            chip_count=chip_count,
            user_selected_chips=selected_json,
        )
        response = await self.client.chat(model, system, user)
        if response.error:
            return [], response
        chips, _ = parse_chips_from_json(response.content)
        return chips, response

    def _get_missing_types(self, chips: list[Chip]) -> list[str]:
        counts = {t: 0 for t in CHIP_TYPES}
        for chip in chips:
            if chip.type in counts:
                counts[chip.type] += 1
        return [t for t, count in counts.items() if count < MIN_CHIPS_PER_TYPE]

    async def _fill_missing(
        self, model: str, persona: dict, existing_chips: list[Chip], missing_types: list[str], style: str
    ) -> tuple[list[Chip], LLMResponse]:
        prompt = self._get_prompt(style, "fill_missing_types")
        existing_json = json.dumps([{"key": c.key, "display": c.display, "type": c.type} for c in existing_chips])
        system = prompt["system"]
        user = self._format_prompt(
            prompt["user"],
            sector=persona["sector"],
            desired_role=persona["desired_role"],
            existing_chips=existing_json,
            missing_types=", ".join(missing_types),
        )
        response = await self.client.chat(model, system, user)
        if response.error:
            return [], response
        chips, _ = parse_chips_from_json(response.content)
        return chips, response
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_runner_async.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add services/runner_async.py tests/test_runner_async.py
git commit -m "feat(services): add async test runner with event streaming"
```

---

## Task 6: Basic TUI App Shell

**Files:**
- Create: `tui/app.py`
- Create: `main.py`

**Step 1: Create the basic app structure**

Create `tui/app.py`:

```python
"""Main Textual application."""

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, TabbedContent, TabPane, Static
from textual.binding import Binding


class ChipBenchmarkApp(App):
    """Chip benchmark TUI application."""

    TITLE = "Chip Benchmark"
    CSS = """
    Screen {
        background: $surface;
    }
    TabbedContent {
        height: 100%;
    }
    TabPane {
        padding: 1;
    }
    .placeholder {
        height: 100%;
        content-align: center middle;
        color: $text-muted;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("?", "help", "Help"),
        Binding("1", "switch_tab('configure')", "Configure", show=False),
        Binding("2", "switch_tab('monitor')", "Monitor", show=False),
        Binding("3", "switch_tab('results')", "Results", show=False),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent(initial="configure"):
            with TabPane("Configure", id="configure"):
                yield Static("Configure screen placeholder", classes="placeholder")
            with TabPane("Monitor", id="monitor"):
                yield Static("Monitor screen placeholder", classes="placeholder")
            with TabPane("Results", id="results"):
                yield Static("Results screen placeholder", classes="placeholder")
        yield Footer()

    def action_switch_tab(self, tab_id: str) -> None:
        """Switch to a specific tab."""
        self.query_one(TabbedContent).active = tab_id

    def action_help(self) -> None:
        """Show help."""
        self.notify("Press 1/2/3 to switch tabs, q to quit")


def run():
    """Run the application."""
    app = ChipBenchmarkApp()
    app.run()


if __name__ == "__main__":
    run()
```

**Step 2: Create the entry point**

Create `main.py`:

```python
#!/usr/bin/env python3
"""Entry point for the Chip Benchmark TUI."""

from tui.app import run

if __name__ == "__main__":
    run()
```

**Step 3: Test manually**

Run: `python main.py`
Expected: TUI opens with 3 tabs and placeholder text

**Step 4: Commit**

```bash
git add tui/app.py main.py
git commit -m "feat(tui): add basic app shell with tabbed navigation"
```

---

## Task 7: Model Selector Widget

**Files:**
- Create: `tui/widgets/model_selector.py`
- Create: `tests/test_model_selector.py`

**Step 1: Write the failing test**

Create `tests/test_model_selector.py`:

```python
import pytest
from textual.app import App, ComposeResult

from tui.widgets.model_selector import ModelSelector


class TestApp(App):
    def compose(self) -> ComposeResult:
        yield ModelSelector()


@pytest.mark.asyncio
async def test_model_selector_allows_max_two():
    """ModelSelector should allow selecting at most 2 models."""
    app = TestApp()
    async with app.run_test() as pilot:
        selector = app.query_one(ModelSelector)
        assert len(selector.selected) == 0
        assert selector.max_selections == 2


@pytest.mark.asyncio
async def test_model_selector_shows_models():
    """ModelSelector should display available models."""
    app = TestApp()
    async with app.run_test() as pilot:
        selector = app.query_one(ModelSelector)
        # Should have some model options
        assert len(selector.models) > 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_model_selector.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'tui.widgets.model_selector'"

**Step 3: Write minimal implementation**

Create `tui/widgets/model_selector.py`:

```python
"""Model selector widget for choosing 1-2 models."""

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static, Checkbox
from textual.widget import Widget
from textual.message import Message
from textual.reactive import reactive

from config import MODELS


class ModelSelector(Widget):
    """Widget for selecting 1-2 models for head-to-head comparison."""

    DEFAULT_CSS = """
    ModelSelector {
        height: auto;
        border: solid $primary;
        padding: 1;
    }
    ModelSelector .title {
        text-style: bold;
        margin-bottom: 1;
    }
    ModelSelector .model-grid {
        height: auto;
        layout: grid;
        grid-size: 3;
        grid-gutter: 1;
    }
    ModelSelector .status {
        margin-top: 1;
        text-align: right;
        color: $text-muted;
    }
    ModelSelector .status.head-to-head {
        color: $success;
    }
    """

    selected: reactive[set[str]] = reactive(set, init=False)
    max_selections: int = 2

    class SelectionChanged(Message):
        """Posted when model selection changes."""

        def __init__(self, selected: set[str]) -> None:
            self.selected = selected
            super().__init__()

    def __init__(self) -> None:
        super().__init__()
        self.models = [(m.id, m.name) for m in MODELS]
        self.selected = set()

    def compose(self) -> ComposeResult:
        yield Static("Models (pick 1-2 for head-to-head)", classes="title")
        with VerticalScroll(classes="model-grid"):
            for model_id, model_name in self.models:
                yield Checkbox(model_name, id=f"model-{model_id.replace('/', '-')}", name=model_id)
        yield Static("Selected: 0", classes="status", id="status")

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        """Handle checkbox state changes."""
        model_id = event.checkbox.name

        if event.value:
            if len(self.selected) >= self.max_selections:
                # Deselect the checkbox - can't select more than max
                event.checkbox.value = False
                self.notify(f"Maximum {self.max_selections} models allowed", severity="warning")
                return
            self.selected = self.selected | {model_id}
        else:
            self.selected = self.selected - {model_id}

        self._update_status()
        self.post_message(self.SelectionChanged(self.selected.copy()))

    def _update_status(self) -> None:
        """Update the status display."""
        status = self.query_one("#status", Static)
        count = len(self.selected)
        if count == 2:
            status.update("Selected: 2 (head-to-head)")
            status.add_class("head-to-head")
        else:
            status.update(f"Selected: {count}")
            status.remove_class("head-to-head")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_model_selector.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tui/widgets/model_selector.py tests/test_model_selector.py
git commit -m "feat(tui): add model selector widget with max 2 selection"
```

---

## Task 8: Configure Screen

**Files:**
- Create: `tui/screens/configure.py`
- Modify: `tui/app.py`

**Step 1: Create the configure screen**

Create `tui/screens/configure.py`:

```python
"""Configure screen for setting up test runs."""

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Static, Select, Button, RadioButton, RadioSet
from textual.widget import Widget
from textual.message import Message

from config import MODELS
from tui.widgets.model_selector import ModelSelector


# Load personas for the dropdown
import json
from config import PERSONAS_FILE

def load_persona_options() -> list[tuple[str, str]]:
    with open(PERSONAS_FILE) as f:
        data = json.load(f)
    return [(p["id"], f"{p['desired_role']} ({p['sector']})") for p in data["personas"]]


class ConfigureScreen(Widget):
    """Configuration screen for test setup."""

    DEFAULT_CSS = """
    ConfigureScreen {
        height: 100%;
        padding: 1;
    }
    ConfigureScreen .section {
        margin-bottom: 1;
    }
    ConfigureScreen .section-title {
        text-style: bold;
        margin-bottom: 1;
    }
    ConfigureScreen .options-row {
        height: auto;
        layout: horizontal;
    }
    ConfigureScreen .option-group {
        width: 1fr;
        margin-right: 2;
    }
    ConfigureScreen .option-group-title {
        text-style: bold;
        margin-bottom: 1;
    }
    ConfigureScreen .buttons {
        margin-top: 2;
        height: auto;
        align: center middle;
    }
    ConfigureScreen Button {
        margin: 0 1;
    }
    ConfigureScreen #run-btn {
        background: $success;
    }
    """

    class RunRequested(Message):
        """Posted when user wants to run a test."""

        def __init__(
            self,
            models: list[str],
            persona_id: str,
            prompt_style: str,
            flow: str,
            constraint_type: str,
            chip_count: int,
            dry_run: bool = False,
        ) -> None:
            self.models = models
            self.persona_id = persona_id
            self.prompt_style = prompt_style
            self.flow = flow
            self.constraint_type = constraint_type
            self.chip_count = chip_count
            self.dry_run = dry_run
            super().__init__()

    def compose(self) -> ComposeResult:
        yield ModelSelector()

        yield Static("", classes="section")

        with Horizontal(classes="options-row"):
            with Vertical(classes="option-group"):
                yield Static("Persona", classes="option-group-title")
                yield Select(
                    load_persona_options(),
                    id="persona-select",
                    prompt="Select persona",
                )

            with Vertical(classes="option-group"):
                yield Static("Prompt Style", classes="option-group-title")
                with RadioSet(id="prompt-style"):
                    yield RadioButton("Terse", value=True, id="terse")
                    yield RadioButton("Guided", id="guided")

            with Vertical(classes="option-group"):
                yield Static("Flow", classes="option-group-title")
                with RadioSet(id="flow"):
                    yield RadioButton("Basic", value=True, id="basic")
                    yield RadioButton("Enriched", id="enriched")

        with Horizontal(classes="options-row"):
            with Vertical(classes="option-group"):
                yield Static("Constraints", classes="option-group-title")
                with RadioSet(id="constraint"):
                    yield RadioButton("None", value=True, id="none")
                    yield RadioButton("2-per-type", id="2-per-type")

            with Vertical(classes="option-group"):
                yield Static("Chip Count", classes="option-group-title")
                with RadioSet(id="chip-count"):
                    yield RadioButton("15", value=True, id="15")
                    yield RadioButton("35", id="35")

        with Horizontal(classes="buttons"):
            yield Button("Run Test", id="run-btn", variant="success")
            yield Button("Dry Run", id="dry-run-btn", variant="default")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id in ("run-btn", "dry-run-btn"):
            self._request_run(dry_run=event.button.id == "dry-run-btn")

    def _request_run(self, dry_run: bool = False) -> None:
        """Gather config and post run request."""
        selector = self.query_one(ModelSelector)
        models = list(selector.selected)

        if not models:
            self.notify("Please select at least one model", severity="error")
            return

        persona_select = self.query_one("#persona-select", Select)
        if persona_select.value == Select.BLANK:
            self.notify("Please select a persona", severity="error")
            return

        # Get radio selections
        prompt_style = self._get_radio_value("prompt-style")
        flow = self._get_radio_value("flow")
        constraint = self._get_radio_value("constraint")
        chip_count = int(self._get_radio_value("chip-count"))

        self.post_message(
            self.RunRequested(
                models=models,
                persona_id=str(persona_select.value),
                prompt_style=prompt_style,
                flow=flow,
                constraint_type=constraint,
                chip_count=chip_count,
                dry_run=dry_run,
            )
        )

    def _get_radio_value(self, radio_set_id: str) -> str:
        """Get the selected value from a RadioSet."""
        radio_set = self.query_one(f"#{radio_set_id}", RadioSet)
        pressed = radio_set.pressed_button
        return pressed.id if pressed else ""
```

**Step 2: Update app.py to use the configure screen**

Edit `tui/app.py` to import and use `ConfigureScreen`:

```python
"""Main Textual application."""

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, TabbedContent, TabPane, Static
from textual.binding import Binding

from tui.screens.configure import ConfigureScreen


class ChipBenchmarkApp(App):
    """Chip benchmark TUI application."""

    TITLE = "Chip Benchmark"
    CSS = """
    Screen {
        background: $surface;
    }
    TabbedContent {
        height: 100%;
    }
    TabPane {
        padding: 1;
    }
    .placeholder {
        height: 100%;
        content-align: center middle;
        color: $text-muted;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("?", "help", "Help"),
        Binding("1", "switch_tab('configure')", "Configure", show=False),
        Binding("2", "switch_tab('monitor')", "Monitor", show=False),
        Binding("3", "switch_tab('results')", "Results", show=False),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent(initial="configure"):
            with TabPane("Configure", id="configure"):
                yield ConfigureScreen()
            with TabPane("Monitor", id="monitor"):
                yield Static("Monitor screen placeholder", classes="placeholder")
            with TabPane("Results", id="results"):
                yield Static("Results screen placeholder", classes="placeholder")
        yield Footer()

    def action_switch_tab(self, tab_id: str) -> None:
        """Switch to a specific tab."""
        self.query_one(TabbedContent).active = tab_id

    def action_help(self) -> None:
        """Show help."""
        self.notify("Press 1/2/3 to switch tabs, q to quit")

    def on_configure_screen_run_requested(self, event: ConfigureScreen.RunRequested) -> None:
        """Handle run request from configure screen."""
        models_str = " vs ".join([m.split("/")[-1] for m in event.models])
        mode = "Dry run" if event.dry_run else "Running"
        self.notify(f"{mode}: {models_str}")
        # Switch to monitor tab
        self.action_switch_tab("monitor")


def run():
    """Run the application."""
    app = ChipBenchmarkApp()
    app.run()


if __name__ == "__main__":
    run()
```

**Step 3: Test manually**

Run: `python main.py`
Expected: Configure screen with model selector, dropdowns, radio buttons, and buttons

**Step 4: Commit**

```bash
git add tui/screens/configure.py tui/app.py
git commit -m "feat(tui): add configure screen with model/config selection"
```

---

## Task 9: Log Pane Widget

**Files:**
- Create: `tui/widgets/log_pane.py`
- Create: `tests/test_log_pane.py`

**Step 1: Write the failing test**

Create `tests/test_log_pane.py`:

```python
import pytest
from textual.app import App, ComposeResult

from tui.widgets.log_pane import LogPane


class TestApp(App):
    def compose(self) -> ComposeResult:
        yield LogPane(title="Test Model")


@pytest.mark.asyncio
async def test_log_pane_add_line():
    """LogPane should allow adding log lines."""
    app = TestApp()
    async with app.run_test() as pilot:
        pane = app.query_one(LogPane)
        pane.add_line("Test message")
        assert "Test message" in pane.get_log_text()


@pytest.mark.asyncio
async def test_log_pane_auto_scroll():
    """LogPane should auto-scroll by default."""
    app = TestApp()
    async with app.run_test() as pilot:
        pane = app.query_one(LogPane)
        assert pane.auto_scroll is True
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_log_pane.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'tui.widgets.log_pane'"

**Step 3: Write minimal implementation**

Create `tui/widgets/log_pane.py`:

```python
"""Log pane widget with auto-scroll support."""

from datetime import datetime

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static, RichLog
from textual.widget import Widget
from textual.reactive import reactive


class LogPane(Widget):
    """A log pane with auto-scroll and pause-on-interaction."""

    DEFAULT_CSS = """
    LogPane {
        height: 100%;
        border: solid $primary;
    }
    LogPane .header {
        height: 3;
        padding: 0 1;
        background: $primary;
        color: $text;
    }
    LogPane .header-title {
        text-style: bold;
    }
    LogPane .header-status {
        dock: right;
    }
    LogPane RichLog {
        height: 1fr;
        padding: 0 1;
    }
    """

    auto_scroll: reactive[bool] = reactive(True)

    def __init__(self, title: str = "Log", **kwargs) -> None:
        super().__init__(**kwargs)
        self.title = title
        self._log_lines: list[str] = []

    def compose(self) -> ComposeResult:
        with Static(classes="header"):
            yield Static(self.title, classes="header-title")
            yield Static("[auto]", classes="header-status", id="scroll-status")
        yield RichLog(highlight=True, markup=True, id="log")

    def add_line(self, message: str, timestamp: bool = True) -> None:
        """Add a line to the log."""
        log = self.query_one("#log", RichLog)
        if timestamp:
            ts = datetime.now().strftime("%H:%M:%S")
            line = f"[dim]{ts}[/dim] {message}"
        else:
            line = message
        self._log_lines.append(line)
        log.write(line)
        if self.auto_scroll:
            log.scroll_end(animate=False)

    def get_log_text(self) -> str:
        """Get all log text for testing."""
        return "\n".join(self._log_lines)

    def clear(self) -> None:
        """Clear the log."""
        self._log_lines = []
        log = self.query_one("#log", RichLog)
        log.clear()

    def watch_auto_scroll(self, value: bool) -> None:
        """Update status when auto_scroll changes."""
        status = self.query_one("#scroll-status", Static)
        status.update("[auto]" if value else "[paused]")

    def on_rich_log_scroll(self) -> None:
        """Pause auto-scroll when user scrolls."""
        # This will be called when user interacts with scroll
        pass

    def toggle_auto_scroll(self) -> None:
        """Toggle auto-scroll state."""
        self.auto_scroll = not self.auto_scroll
        if self.auto_scroll:
            log = self.query_one("#log", RichLog)
            log.scroll_end(animate=False)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_log_pane.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tui/widgets/log_pane.py tests/test_log_pane.py
git commit -m "feat(tui): add log pane widget with auto-scroll"
```

---

## Task 10: Monitor Screen

**Files:**
- Create: `tui/screens/monitor.py`
- Modify: `tui/app.py`

**Step 1: Create the monitor screen**

Create `tui/screens/monitor.py`:

```python
"""Monitor screen for watching test runs."""

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Static, Button, ProgressBar
from textual.widget import Widget
from textual.message import Message
from textual.reactive import reactive


from tui.widgets.log_pane import LogPane


class ModelProgress(Widget):
    """Progress display for a single model."""

    DEFAULT_CSS = """
    ModelProgress {
        height: auto;
        padding: 1;
        border: solid $primary;
        margin-bottom: 1;
    }
    ModelProgress .model-name {
        text-style: bold;
        margin-bottom: 1;
    }
    ModelProgress .stats {
        color: $text-muted;
    }
    """

    def __init__(self, model: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model = model

    def compose(self) -> ComposeResult:
        yield Static(self.model.split("/")[-1], classes="model-name")
        yield ProgressBar(total=100, show_eta=False, id="progress")
        yield Static("Tokens: 0 | $0.000", classes="stats", id="stats")

    def update_progress(self, percent: int, step: str = "") -> None:
        """Update progress bar and step text."""
        bar = self.query_one("#progress", ProgressBar)
        bar.update(progress=percent)

    def update_stats(self, tokens: int, cost: float) -> None:
        """Update token and cost stats."""
        stats = self.query_one("#stats", Static)
        stats.update(f"Tokens: {tokens:,} | ${cost:.4f}")


class MonitorScreen(Widget):
    """Monitor screen for watching test runs."""

    DEFAULT_CSS = """
    MonitorScreen {
        height: 100%;
        padding: 1;
    }
    MonitorScreen .header {
        height: auto;
        margin-bottom: 1;
    }
    MonitorScreen .run-info {
        text-style: bold;
    }
    MonitorScreen .config-info {
        color: $text-muted;
    }
    MonitorScreen .progress-area {
        height: auto;
        margin-bottom: 1;
    }
    MonitorScreen .logs-area {
        height: 1fr;
        layout: horizontal;
    }
    MonitorScreen .logs-area LogPane {
        width: 1fr;
        margin-right: 1;
    }
    MonitorScreen .logs-area LogPane:last-child {
        margin-right: 0;
    }
    MonitorScreen .buttons {
        height: auto;
        dock: bottom;
        padding: 1;
        align: center middle;
    }
    MonitorScreen Button {
        margin: 0 1;
    }
    MonitorScreen .idle-message {
        height: 100%;
        content-align: center middle;
        color: $text-muted;
    }
    """

    is_running: reactive[bool] = reactive(False)

    class CancelRequested(Message):
        """Posted when user requests cancellation."""
        pass

    class ViewResultsRequested(Message):
        """Posted when user wants to view results."""
        pass

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.models: list[str] = []
        self.run_name: str = ""
        self.config_str: str = ""

    def compose(self) -> ComposeResult:
        yield Static("No run in progress. Configure and start a test.", classes="idle-message", id="idle")
        with Container(id="running-view"):
            with Container(classes="header"):
                yield Static("", classes="run-info", id="run-info")
                yield Static("", classes="config-info", id="config-info")
            with Horizontal(classes="progress-area", id="progress-area"):
                pass  # Progress bars added dynamically
            with Horizontal(classes="logs-area", id="logs-area"):
                pass  # Log panes added dynamically
            with Horizontal(classes="buttons"):
                yield Button("Pause", id="pause-btn", variant="warning")
                yield Button("Cancel", id="cancel-btn", variant="error")
                yield Button("View Results", id="results-btn", variant="success", disabled=True)

    def on_mount(self) -> None:
        """Hide running view initially."""
        self.query_one("#running-view").display = False

    def setup_run(self, models: list[str], run_name: str, config_str: str) -> None:
        """Set up the monitor for a new run."""
        self.models = models
        self.run_name = run_name
        self.config_str = config_str

        # Update header
        self.query_one("#run-info", Static).update(f"Run: {run_name}")
        self.query_one("#config-info", Static).update(config_str)

        # Clear and add progress bars
        progress_area = self.query_one("#progress-area")
        progress_area.remove_children()
        for model in models:
            progress_area.mount(ModelProgress(model, id=f"progress-{model.replace('/', '-')}"))

        # Clear and add log panes
        logs_area = self.query_one("#logs-area")
        logs_area.remove_children()
        for model in models:
            model_name = model.split("/")[-1]
            logs_area.mount(LogPane(title=model_name, id=f"log-{model.replace('/', '-')}"))

        # Show running view
        self.query_one("#idle").display = False
        self.query_one("#running-view").display = True
        self.is_running = True
        self.query_one("#results-btn", Button).disabled = True

    def add_log(self, model: str, message: str) -> None:
        """Add a log line for a model."""
        pane_id = f"log-{model.replace('/', '-')}"
        try:
            pane = self.query_one(f"#{pane_id}", LogPane)
            pane.add_line(message)
        except Exception:
            pass  # Pane might not exist yet

    def update_progress(self, model: str, percent: int, step: str = "") -> None:
        """Update progress for a model."""
        progress_id = f"progress-{model.replace('/', '-')}"
        try:
            progress = self.query_one(f"#{progress_id}", ModelProgress)
            progress.update_progress(percent, step)
        except Exception:
            pass

    def update_stats(self, model: str, tokens: int, cost: float) -> None:
        """Update stats for a model."""
        progress_id = f"progress-{model.replace('/', '-')}"
        try:
            progress = self.query_one(f"#{progress_id}", ModelProgress)
            progress.update_stats(tokens, cost)
        except Exception:
            pass

    def mark_complete(self) -> None:
        """Mark the run as complete."""
        self.is_running = False
        self.query_one("#results-btn", Button).disabled = False
        self.query_one("#pause-btn", Button).disabled = True
        self.query_one("#cancel-btn", Button).disabled = True

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "cancel-btn":
            self.post_message(self.CancelRequested())
        elif event.button.id == "results-btn":
            self.post_message(self.ViewResultsRequested())
```

**Step 2: Update app.py to use the monitor screen**

Edit `tui/app.py`:

```python
"""Main Textual application."""

import asyncio
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, TabbedContent, TabPane, Static
from textual.binding import Binding

from tui.screens.configure import ConfigureScreen
from tui.screens.monitor import MonitorScreen
from services.runner_async import AsyncRunner, RunConfig, EventType
from db.schema import init_db
from db.repository import Repository


class ChipBenchmarkApp(App):
    """Chip benchmark TUI application."""

    TITLE = "Chip Benchmark"
    CSS = """
    Screen {
        background: $surface;
    }
    TabbedContent {
        height: 100%;
    }
    TabPane {
        padding: 1;
    }
    .placeholder {
        height: 100%;
        content-align: center middle;
        color: $text-muted;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("?", "help", "Help"),
        Binding("1", "switch_tab('configure')", "Configure", show=False),
        Binding("2", "switch_tab('monitor')", "Monitor", show=False),
        Binding("3", "switch_tab('results')", "Results", show=False),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.runner = AsyncRunner()
        self.db_conn = init_db()
        self.repo = Repository(self.db_conn)
        self.current_run_id: str | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent(initial="configure"):
            with TabPane("Configure", id="configure"):
                yield ConfigureScreen()
            with TabPane("Monitor", id="monitor"):
                yield MonitorScreen()
            with TabPane("Results", id="results"):
                yield Static("Results screen placeholder", classes="placeholder")
        yield Footer()

    def action_switch_tab(self, tab_id: str) -> None:
        """Switch to a specific tab."""
        self.query_one(TabbedContent).active = tab_id

    def action_help(self) -> None:
        """Show help."""
        self.notify("Press 1/2/3 to switch tabs, q to quit")

    async def on_configure_screen_run_requested(self, event: ConfigureScreen.RunRequested) -> None:
        """Handle run request from configure screen."""
        if event.dry_run:
            self.notify("Dry run mode - no API calls")
            return

        # Create run name
        models_short = [m.split("/")[-1] for m in event.models]
        run_name = " vs ".join(models_short) if len(models_short) == 2 else models_short[0]

        # Config string for display
        config_str = f"{event.persona_id} | {event.prompt_style} | {event.flow} | {event.constraint_type} | {event.chip_count} chips"

        # Create run in database
        self.current_run_id = self.repo.create_run(
            name=run_name,
            persona=event.persona_id,
            prompt_style=event.prompt_style,
            flow=event.flow,
            constraint_type=event.constraint_type,
            chip_count=event.chip_count,
        )

        # Setup monitor
        monitor = self.query_one(MonitorScreen)
        monitor.setup_run(event.models, run_name, config_str)

        # Switch to monitor tab
        self.action_switch_tab("monitor")

        # Start the run
        config = RunConfig(
            models=event.models,
            persona_id=event.persona_id,
            prompt_style=event.prompt_style,
            flow=event.flow,
            constraint_type=event.constraint_type,
            chip_count=event.chip_count,
        )

        # Run in background
        asyncio.create_task(self._run_test(config))

    async def _run_test(self, config: RunConfig) -> None:
        """Run the test and update monitor."""
        monitor = self.query_one(MonitorScreen)

        try:
            await self.runner.start()
            async for event in self.runner.run(config):
                if event.type == EventType.LOG:
                    monitor.add_log(event.model, event.message)
                elif event.type == EventType.PROGRESS:
                    percent = event.data.get("percent", 0)
                    step = event.data.get("step", "")
                    monitor.update_progress(event.model, percent, step)
                elif event.type == EventType.ERROR:
                    monitor.add_log(event.model, f"[red]ERROR: {event.message}[/red]")
                elif event.type == EventType.COMPLETE:
                    result = event.data.get("result")
                    if result and self.current_run_id:
                        # Save to database
                        self.repo.create_result(
                            run_id=self.current_run_id,
                            model=result.model,
                            chips=result.chips,
                            tokens_in=result.tokens_in,
                            tokens_out=result.tokens_out,
                            cost_usd=result.cost_usd,
                            latency_ms=result.latency_ms,
                            situation_count=result.situation_count,
                            jargon_count=result.jargon_count,
                            role_task_count=result.role_task_count,
                            environment_count=result.environment_count,
                        )
                        monitor.update_stats(result.model, result.tokens_in + result.tokens_out, result.cost_usd)
        finally:
            await self.runner.stop()
            monitor.mark_complete()
            self.notify("Run complete!")

    def on_monitor_screen_cancel_requested(self, event: MonitorScreen.CancelRequested) -> None:
        """Handle cancel request."""
        self.runner.cancel()
        self.notify("Cancelling run...")

    def on_monitor_screen_view_results_requested(self, event: MonitorScreen.ViewResultsRequested) -> None:
        """Handle view results request."""
        self.action_switch_tab("results")


def run():
    """Run the application."""
    app = ChipBenchmarkApp()
    app.run()


if __name__ == "__main__":
    run()
```

**Step 3: Test manually**

Run: `python main.py`
Expected: Configure a run, click Run Test, see Monitor screen with progress and logs

**Step 4: Commit**

```bash
git add tui/screens/monitor.py tui/app.py
git commit -m "feat(tui): add monitor screen with split logs and progress"
```

---

## Task 11: Results Browser Screen

**Files:**
- Create: `tui/screens/results.py`
- Modify: `tui/app.py`

**Step 1: Create the results browser**

Create `tui/screens/results.py`:

```python
"""Results screen for browsing and comparing runs."""

from datetime import datetime

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.widgets import Static, Button, Select, ListView, ListItem, Label
from textual.widget import Widget
from textual.message import Message
from textual.reactive import reactive

from db.repository import Repository


class RunListItem(ListItem):
    """A list item representing a run."""

    def __init__(self, run: dict, results: list[dict], **kwargs) -> None:
        super().__init__(**kwargs)
        self.run = run
        self.results = results

    def compose(self) -> ComposeResult:
        # Build display
        models = [r["model"].split("/")[-1] for r in self.results]
        models_str = " vs ".join(models) if len(models) == 2 else models[0] if models else "No results"

        # Ratings
        ratings = []
        for r in self.results:
            if r.get("rating"):
                ratings.append("★" * r["rating"] + "☆" * (5 - r["rating"]))
            else:
                ratings.append("--")
        ratings_str = "  ".join(ratings)

        # Parse timestamp
        created = datetime.fromisoformat(self.run["created_at"])
        time_str = created.strftime("%H:%M")

        config = f"{self.run['persona']} | {self.run['prompt_style']} | {self.run['flow']} | {self.run['chip_count']} chips"

        yield Static(f"[bold]{models_str}[/bold]  {ratings_str}")
        yield Static(f"[dim]{config}[/dim]")
        yield Static(f"[dim]{time_str}[/dim]")


class ResultsScreen(Widget):
    """Results screen for browsing runs."""

    DEFAULT_CSS = """
    ResultsScreen {
        height: 100%;
        padding: 1;
    }
    ResultsScreen .filters {
        height: auto;
        margin-bottom: 1;
    }
    ResultsScreen .filters Select {
        width: 20;
        margin-right: 1;
    }
    ResultsScreen ListView {
        height: 1fr;
        border: solid $primary;
    }
    ResultsScreen .buttons {
        height: auto;
        margin-top: 1;
    }
    ResultsScreen Button {
        margin-right: 1;
    }
    ResultsScreen .empty-message {
        height: 100%;
        content-align: center middle;
        color: $text-muted;
    }
    RunListItem {
        padding: 1;
    }
    RunListItem:hover {
        background: $surface-lighten-1;
    }
    """

    class RunSelected(Message):
        """Posted when a run is selected for viewing."""

        def __init__(self, run_id: str) -> None:
            self.run_id = run_id
            super().__init__()

    def __init__(self, repo: Repository, **kwargs) -> None:
        super().__init__(**kwargs)
        self.repo = repo

    def compose(self) -> ComposeResult:
        with Horizontal(classes="filters"):
            yield Select(
                [("All personas", "all")],
                id="persona-filter",
                value="all",
            )
            yield Select(
                [("All models", "all")],
                id="model-filter",
                value="all",
            )
            yield Select(
                [("All ratings", "all"), ("Unrated", "unrated"), ("Rated", "rated")],
                id="rating-filter",
                value="all",
            )
            yield Button("Refresh", id="refresh-btn")

        yield ListView(id="runs-list")

        with Horizontal(classes="buttons"):
            yield Button("Open", id="open-btn", variant="success")
            yield Button("Delete", id="delete-btn", variant="error")
            yield Button("Export CSV", id="export-btn")

    def on_mount(self) -> None:
        """Load runs when mounted."""
        self.refresh_runs()

    def refresh_runs(self) -> None:
        """Refresh the runs list."""
        runs = self.repo.list_runs()
        list_view = self.query_one("#runs-list", ListView)
        list_view.clear()

        if not runs:
            return

        # Group by date
        by_date: dict[str, list[tuple[dict, list[dict]]]] = {}
        for run in runs:
            date = datetime.fromisoformat(run["created_at"]).strftime("%Y-%m-%d")
            results = self.repo.get_results_for_run(run["id"])
            if date not in by_date:
                by_date[date] = []
            by_date[date].append((run, results))

        # Add to list
        for date, items in by_date.items():
            # Date header
            list_view.append(ListItem(Label(f"[bold]{date}[/bold] ({len(items)} runs)")))
            for run, results in items:
                list_view.append(RunListItem(run, results, id=f"run-{run['id']}"))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "refresh-btn":
            self.refresh_runs()
        elif event.button.id == "open-btn":
            self._open_selected()
        elif event.button.id == "delete-btn":
            self._delete_selected()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle list item selection."""
        if isinstance(event.item, RunListItem):
            self.post_message(self.RunSelected(event.item.run["id"]))

    def _open_selected(self) -> None:
        """Open the selected run."""
        list_view = self.query_one("#runs-list", ListView)
        if list_view.highlighted_child and isinstance(list_view.highlighted_child, RunListItem):
            self.post_message(self.RunSelected(list_view.highlighted_child.run["id"]))

    def _delete_selected(self) -> None:
        """Delete the selected run."""
        list_view = self.query_one("#runs-list", ListView)
        if list_view.highlighted_child and isinstance(list_view.highlighted_child, RunListItem):
            run_id = list_view.highlighted_child.run["id"]
            self.repo.delete_run(run_id)
            self.refresh_runs()
            self.notify("Run deleted")
```

**Step 2: Update app.py to use results screen**

Add import and update compose/handlers in `tui/app.py`:

```python
# Add to imports
from tui.screens.results import ResultsScreen

# Update compose() - replace Results TabPane content:
with TabPane("Results", id="results"):
    yield ResultsScreen(repo=self.repo)

# Add handler for run selection
def on_results_screen_run_selected(self, event: ResultsScreen.RunSelected) -> None:
    """Handle run selection from results screen."""
    self.notify(f"Opening run: {event.run_id[:8]}...")
    # TODO: Open comparison view
```

**Step 3: Test manually**

Run: `python main.py`
Expected: Results tab shows list of runs grouped by date, with ratings and config info

**Step 4: Commit**

```bash
git add tui/screens/results.py tui/app.py
git commit -m "feat(tui): add results browser screen with run list"
```

---

## Task 12: Comparison Detail Screen

**Files:**
- Create: `tui/screens/comparison.py`
- Create: `tui/widgets/chip_panel.py`
- Create: `tui/widgets/rating_bar.py`
- Modify: `tui/app.py`

**Step 1: Create chip panel widget**

Create `tui/widgets/chip_panel.py`:

```python
"""Chip panel widget for displaying chips grouped by type."""

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static
from textual.widget import Widget

from config import CHIP_TYPES


class ChipPanel(Widget):
    """Panel displaying chips grouped by type."""

    DEFAULT_CSS = """
    ChipPanel {
        height: 100%;
        border: solid $primary;
        padding: 1;
    }
    ChipPanel .header {
        text-style: bold;
        margin-bottom: 1;
    }
    ChipPanel .type-section {
        margin-bottom: 1;
    }
    ChipPanel .type-header {
        text-style: bold;
        color: $primary;
        text-transform: uppercase;
    }
    ChipPanel .chip {
        padding-left: 2;
    }
    """

    def __init__(self, model: str, chips: list[dict], rating: int | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.chips = chips
        self.rating = rating

    def compose(self) -> ComposeResult:
        model_short = self.model.split("/")[-1]
        rating_str = "★" * self.rating + "☆" * (5 - self.rating) if self.rating else "--"
        yield Static(f"[bold]{model_short}[/bold]  {rating_str}", classes="header")

        with VerticalScroll():
            # Group chips by type
            by_type: dict[str, list[dict]] = {t: [] for t in CHIP_TYPES}
            for chip in self.chips:
                if chip["type"] in by_type:
                    by_type[chip["type"]].append(chip)

            for chip_type in CHIP_TYPES:
                chips = by_type[chip_type]
                type_display = chip_type.replace("_", " ").upper()
                yield Static(f"{type_display} ({len(chips)})", classes="type-header")
                for chip in chips:
                    yield Static(f"• {chip['display']}", classes="chip")
```

**Step 2: Create rating bar widget**

Create `tui/widgets/rating_bar.py`:

```python
"""Rating bar widget for 1-5 star ratings."""

from textual.app import ComposeResult
from textual.widgets import Static, Button
from textual.widget import Widget
from textual.message import Message
from textual.reactive import reactive
from textual.binding import Binding


class RatingBar(Widget):
    """Widget for rating with 1-5 stars using keyboard."""

    DEFAULT_CSS = """
    RatingBar {
        height: auto;
        padding: 1;
        background: $surface;
        border: solid $primary;
        layout: horizontal;
    }
    RatingBar .label {
        margin-right: 2;
    }
    RatingBar .stars {
        margin-right: 2;
    }
    RatingBar .saved {
        color: $success;
    }
    """

    BINDINGS = [
        Binding("1", "rate(1)", "1 star", show=False),
        Binding("2", "rate(2)", "2 stars", show=False),
        Binding("3", "rate(3)", "3 stars", show=False),
        Binding("4", "rate(4)", "4 stars", show=False),
        Binding("5", "rate(5)", "5 stars", show=False),
        Binding("left", "prev_model", "Previous", show=False),
        Binding("right", "next_model", "Next", show=False),
    ]

    current_model: reactive[int] = reactive(0)
    show_saved: reactive[bool] = reactive(False)

    class RatingChanged(Message):
        """Posted when rating changes."""

        def __init__(self, result_id: str, rating: int) -> None:
            self.result_id = result_id
            self.rating = rating
            super().__init__()

    def __init__(self, results: list[dict], **kwargs) -> None:
        super().__init__(**kwargs)
        self.results = results

    def compose(self) -> ComposeResult:
        if not self.results:
            yield Static("No results to rate")
            return

        result = self.results[0]
        model = result["model"].split("/")[-1]
        yield Static(f"Rate: ← {model} →", classes="label", id="label")
        yield Static("[1] [2] [3] [4] [5]", classes="stars")
        yield Static("", classes="saved", id="saved")

    def watch_current_model(self, index: int) -> None:
        """Update label when model changes."""
        if not self.results:
            return
        result = self.results[index]
        model = result["model"].split("/")[-1]
        label = self.query_one("#label", Static)
        label.update(f"Rate: ← {model} →")

    def watch_show_saved(self, value: bool) -> None:
        """Show/hide saved indicator."""
        saved = self.query_one("#saved", Static)
        saved.update("Saved ✓" if value else "")

    def action_rate(self, rating: int) -> None:
        """Rate the current model."""
        if not self.results:
            return
        result = self.results[self.current_model]
        self.post_message(self.RatingChanged(result["id"], rating))
        self.show_saved = True
        self.set_timer(1.5, self._hide_saved)

    def _hide_saved(self) -> None:
        self.show_saved = False

    def action_prev_model(self) -> None:
        """Switch to previous model."""
        if self.current_model > 0:
            self.current_model -= 1

    def action_next_model(self) -> None:
        """Switch to next model."""
        if self.current_model < len(self.results) - 1:
            self.current_model += 1
```

**Step 3: Create comparison screen**

Create `tui/screens/comparison.py`:

```python
"""Comparison detail screen for viewing run results."""

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.widgets import Static, Button, TabbedContent, TabPane
from textual.widget import Widget
from textual.message import Message
from textual.screen import Screen

from db.repository import Repository
from tui.widgets.chip_panel import ChipPanel
from tui.widgets.rating_bar import RatingBar


class ComparisonScreen(Screen):
    """Full-screen comparison view for a run."""

    DEFAULT_CSS = """
    ComparisonScreen {
        background: $surface;
    }
    ComparisonScreen .header {
        height: auto;
        padding: 1;
        background: $primary;
    }
    ComparisonScreen .header-title {
        text-style: bold;
    }
    ComparisonScreen .header-config {
        color: $text-muted;
    }
    ComparisonScreen .back-btn {
        dock: left;
    }
    ComparisonScreen .content {
        height: 1fr;
    }
    ComparisonScreen .chips-area {
        height: 1fr;
        layout: horizontal;
        padding: 1;
    }
    ComparisonScreen .chips-area ChipPanel {
        width: 1fr;
        margin-right: 1;
    }
    ComparisonScreen .chips-area ChipPanel:last-child {
        margin-right: 0;
    }
    ComparisonScreen RatingBar {
        dock: bottom;
    }
    """

    BINDINGS = [
        ("escape", "go_back", "Back"),
        ("backspace", "go_back", "Back"),
    ]

    class BackRequested(Message):
        """Posted when user wants to go back."""
        pass

    def __init__(self, run_id: str, repo: Repository, **kwargs) -> None:
        super().__init__(**kwargs)
        self.run_id = run_id
        self.repo = repo
        self.run: dict | None = None
        self.results: list[dict] = []

    def compose(self) -> ComposeResult:
        self.run = self.repo.get_run(self.run_id)
        self.results = self.repo.get_results_for_run(self.run_id)

        if not self.run:
            yield Static("Run not found")
            return

        models = [r["model"].split("/")[-1] for r in self.results]
        title = " vs ".join(models) if len(models) == 2 else models[0] if models else "No results"
        config = f"{self.run['persona']} | {self.run['prompt_style']} | {self.run['flow']} | {self.run['constraint_type']} | {self.run['chip_count']} chips"

        with Container(classes="header"):
            yield Button("← Back", id="back-btn", classes="back-btn")
            yield Static(f"Results: {title}", classes="header-title")
            yield Static(config, classes="header-config")

        with TabbedContent():
            with TabPane("Chips", id="chips-tab"):
                with Horizontal(classes="chips-area"):
                    for result in self.results:
                        yield ChipPanel(
                            model=result["model"],
                            chips=result["chips"],
                            rating=result.get("rating"),
                        )

            with TabPane("Stats", id="stats-tab"):
                yield self._build_stats_table()

            with TabPane("Raw", id="raw-tab"):
                yield Static("Raw JSON view - TODO")

        yield RatingBar(self.results)

    def _build_stats_table(self) -> Widget:
        """Build a stats comparison table."""
        lines = ["[bold]Metric              " + "  ".join(r["model"].split("/")[-1][:15].ljust(15) for r in self.results) + "[/bold]"]
        lines.append("-" * 60)

        metrics = [
            ("Tokens In", "tokens_in"),
            ("Tokens Out", "tokens_out"),
            ("Cost (USD)", "cost_usd"),
            ("Latency (ms)", "latency_ms"),
            ("Situation", "situation_count"),
            ("Jargon", "jargon_count"),
            ("Role Task", "role_task_count"),
            ("Environment", "environment_count"),
        ]

        for label, key in metrics:
            values = []
            for r in self.results:
                val = r.get(key, 0)
                if key == "cost_usd":
                    values.append(f"${val:.4f}".ljust(15))
                else:
                    values.append(str(val).ljust(15))
            lines.append(f"{label:<20}" + "  ".join(values))

        return Static("\n".join(lines))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back-btn":
            self.action_go_back()

    def action_go_back(self) -> None:
        """Go back to results list."""
        self.app.pop_screen()

    def on_rating_bar_rating_changed(self, event: RatingBar.RatingChanged) -> None:
        """Handle rating change."""
        self.repo.update_rating(event.result_id, event.rating)
        self.notify(f"Rated {event.rating} stars")
```

**Step 4: Update app.py to push comparison screen**

Update `tui/app.py`:

```python
# Add import
from tui.screens.comparison import ComparisonScreen

# Update the handler:
def on_results_screen_run_selected(self, event: ResultsScreen.RunSelected) -> None:
    """Handle run selection from results screen."""
    self.push_screen(ComparisonScreen(event.run_id, self.repo))
```

**Step 5: Test manually**

Run: `python main.py`
Expected: Click a run in Results to see full comparison view with chips, stats, and rating bar

**Step 6: Commit**

```bash
git add tui/widgets/chip_panel.py tui/widgets/rating_bar.py tui/screens/comparison.py tui/app.py
git commit -m "feat(tui): add comparison detail screen with chip panels and ratings"
```

---

## Task 13: Final Polish and Testing

**Files:**
- Modify: `.gitignore`
- Create: `tests/conftest.py`

**Step 1: Update .gitignore**

Add to `.gitignore`:

```
# Database
benchmark.db

# Python
__pycache__/
*.py[cod]
.pytest_cache/

# Environment
.env
.venv/
```

**Step 2: Create pytest configuration**

Create `tests/conftest.py`:

```python
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
```

**Step 3: Run all tests**

Run: `pytest -v`
Expected: All tests pass

**Step 4: Run the full app manually**

Run: `python main.py`
Test the full flow:
1. Configure a run with 2 models
2. Click Run Test
3. Watch logs in Monitor
4. View results in Results tab
5. Open a run and rate it with 1-5 keys

**Step 5: Final commit**

```bash
git add -A
git commit -m "chore: add gitignore and pytest config, final polish"
```

---

## Summary

This plan creates:

1. **Database layer** (Tasks 2-3): SQLite schema and repository
2. **Async services** (Tasks 4-5): Non-blocking LLM client and test runner
3. **TUI app shell** (Task 6): Basic Textual app with tabs
4. **Configure screen** (Tasks 7-8): Model selector and config options
5. **Monitor screen** (Tasks 9-10): Split log panes with progress
6. **Results screen** (Tasks 11-12): Run browser and comparison view
7. **Testing** (Task 13): Final polish and test coverage

Total: ~13 tasks, each with 3-5 steps following TDD.
