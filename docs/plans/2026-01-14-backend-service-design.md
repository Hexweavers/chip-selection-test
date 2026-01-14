# Backend Service Design

Transform the chip generation benchmark from CLI-only to a backend service for hexweavers.io.

## Overview

- **Workflow**: CLI runs batch tests, developers browse and rate results via UI
- **Database**: Turso (libSQL) - hosted edge SQLite
- **Auth**: None (internal VPC only)
- **Framework**: FastAPI + Mangum for Lambda
- **Ratings**: Track rater identity (user_id from frontend)

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  CLI (runner)   │────▶│  Turso (libSQL)  │◀────│     Lambda      │
│  Batch tests    │     │    SQLite DB     │     │    FastAPI      │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                                                          ▼
                                                 ┌─────────────────┐
                                                 │  hexweavers.io  │
                                                 │   Frontend UI   │
                                                 └─────────────────┘
```

The CLI and Lambda share the same database client code, just different entry points.

## Database Schema

```sql
CREATE TABLE runs (
    id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT DEFAULT 'running',  -- running | completed | failed
    config TEXT                     -- JSON: models, personas, filters used
);

CREATE TABLE results (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id),
    model TEXT NOT NULL,
    persona_id TEXT NOT NULL,
    sector TEXT NOT NULL,
    desired_role TEXT NOT NULL,
    style TEXT NOT NULL,            -- terse | guided
    input_type TEXT NOT NULL,       -- basic | enriched
    constraint_type TEXT NOT NULL,  -- with_constraint | no_constraint
    chip_count INTEGER NOT NULL,
    final_chips TEXT NOT NULL,      -- JSON array of chips
    step1_chips TEXT,               -- JSON (enriched flow only)
    selected_chips TEXT,            -- JSON (enriched flow only)
    fill_chips TEXT,                -- JSON (if fill was needed)
    errors TEXT,                    -- JSON array of error strings
    latency_ms INTEGER,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cost_usd REAL,                  -- Calculated at test time
    created_at TEXT NOT NULL
);

CREATE TABLE ratings (
    id TEXT PRIMARY KEY,
    result_id TEXT NOT NULL REFERENCES results(id),
    user_id TEXT NOT NULL,
    rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
    created_at TEXT NOT NULL,
    UNIQUE(result_id, user_id)
);

CREATE INDEX idx_results_model ON results(model);
CREATE INDEX idx_results_run_id ON results(run_id);
CREATE INDEX idx_ratings_result_id ON ratings(result_id);
```

## API Endpoints

### `GET /runs`
List test runs.
```json
{
  "runs": [
    {"id": "abc-123", "started_at": "2025-01-07T10:00:00Z", "status": "completed", "result_count": 48}
  ]
}
```

### `GET /runs/{run_id}`
Get single run details.

### `GET /results`
List results with filters.
```
?run_id=abc-123
?model=anthropic/claude-haiku-4.5
?persona_id=tech_pm
?rated_by=user_123
?unrated_by=user_123
?limit=50&offset=0
```
```json
{
  "results": [
    {
      "id": "def-456",
      "model": "anthropic/claude-haiku-4.5",
      "persona_id": "tech_pm",
      "style": "terse",
      "chip_count": 15,
      "avg_rating": 3.5,
      "rating_count": 2,
      "my_rating": null,
      "cost_usd": 0.0012
    }
  ],
  "total": 128
}
```

### `GET /results/{result_id}`
Full result with all chips.

### `POST /results/{result_id}/ratings`
Submit rating. User ID from `X-User-Id` header.
```json
// Request
{"rating": 4}

// Response
{"id": "rat-789", "rating": 4, "created_at": "..."}
```

### `GET /ratings`
Raw ratings data.
```
?result_id=def-456
?user_id=user_123
```

### `GET /stats`
Aggregated metrics for dashboards/leaderboards.
```
?group_by=model
?run_id=abc-123
```
```json
{
  "stats": [
    {
      "model": "anthropic/claude-haiku-4.5",
      "result_count": 24,
      "rated_count": 18,
      "avg_rating": 4.2,
      "total_cost_usd": 0.048,
      "avg_latency_ms": 1250,
      "avg_tokens": {"input": 850, "output": 420}
    }
  ]
}
```

## CLI Integration

The CLI stays the same interface but writes to Turso instead of JSON files.

### New Modules

**`db/client.py`** - Turso client wrapper:
```python
import libsql_experimental as libsql

def get_db():
    return libsql.connect(
        os.getenv("TURSO_DATABASE_URL"),
        auth_token=os.getenv("TURSO_AUTH_TOKEN")
    )
```

**`db/repository.py`** - Database operations:
```python
def create_run(config: dict) -> str: ...
def save_result(run_id: str, result: TestResult) -> str: ...
def complete_run(run_id: str): ...
def result_exists(...) -> bool: ...  # For --resume
```

### Pricing Config

Extend `config.py`:
```python
@dataclass
class ModelConfig:
    id: str
    name: str
    input_cost_per_m: float   # $ per 1M input tokens
    output_cost_per_m: float  # $ per 1M output tokens
```

Cost calculated at test time:
```
cost = (input_tokens * input_price + output_tokens * output_price) / 1_000_000
```

## Project Structure

```
api/
├── __init__.py
├── main.py              # FastAPI app
├── routes/
│   ├── runs.py
│   ├── results.py
│   ├── ratings.py
│   └── stats.py
└── lambda_handler.py    # Mangum wrapper

db/
├── __init__.py
├── client.py            # Turso connection
├── repository.py        # CRUD operations
└── schema.sql           # Table definitions
```

## Deployment

**Lambda Handler:**
```python
from mangum import Mangum
from api.main import app

handler = Mangum(app, lifespan="off")
```

**Environment Variables:**
```
TURSO_DATABASE_URL=libsql://your-db.turso.io
TURSO_AUTH_TOKEN=eyJ...
```

**Internal VPC:**
- Lambda in private subnet
- API Gateway with VPC endpoint or internal ALB
- No public internet access

**Local Development:**
```bash
uvicorn api.main:app --reload
```

## What Stays the Same

**CLI flags unchanged:**
```bash
python runner.py --model anthropic/claude-haiku-4.5
python runner.py --all
python runner.py --resume
python runner.py --dry-run
python runner.py --persona tech_pm
```

**Core services unchanged:**
- `services/llm.py` - OpenRouter client
- `services/generator.py` - Chip generation logic
- `services/selector.py` - LLM-as-user selection

**What changes:**

| Component | Before | After |
|-----------|--------|-------|
| Storage | JSON files in `results/` | Turso database |
| Output | Local files | DB + optional JSON export |
| New | - | `api/` module for Lambda |
| New | - | `db/` module shared by CLI + API |
