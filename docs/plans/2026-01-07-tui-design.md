# Chip Benchmark TUI Design

## Overview

Transform the chip generation benchmark tool into a fully interactive TUI using Textual. Enables head-to-head model comparison with live monitoring and manual quality review.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Chip Benchmark TUI                              [?] Help  [Q]uit│
├────────────────┬─────────────────┬──────────────────────────────┤
│  ⚙ Configure   │   ▶ Monitor     │   📊 Results                 │
├────────────────┴─────────────────┴──────────────────────────────┤
│                      (Tab content area)                          │
└──────────────────────────────────────────────────────────────────┘
```

- **Framework:** Textual
- **Navigation:** Tabs at top (Configure, Monitor, Results)
- **Storage:** SQLite (`benchmark.db`)
- **Concurrency:** Up to 2 parallel model runs (head-to-head)

## Screens

### Configure Screen

Pick 1-2 models for head-to-head comparison against a single test configuration.

```
┌─────────────────────────────────────────────────────────────────┐
│  ⚙ Configure                                                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Models (pick 1-2 for head-to-head)                             │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ ● Claude Haiku      ○ Llama Scout      ○ DeepSeek       │    │
│  │ ● GPT-5-Mini        ○ Qwen3            ○ Grok           │    │
│  │ ○ Gemini Flash      ○ MiniMax          ○ Mistral        │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                        Selected: 2 (head-to-head)│
│                                                                  │
│  ─────────────────────────────────────────────────────────────  │
│                                                                  │
│  Persona            Prompt Style       Flow                      │
│  ▼ Product Manager  ○ Terse            ○ Basic                   │
│                     ● Guided           ● Enriched                │
│                                                                  │
│  Constraints        Chip Count                                   │
│  ○ None             ○ 15                                         │
│  ● 2-per-type       ● 35                                         │
│                                                                  │
│  ─────────────────────────────────────────────────────────────  │
│                                                                  │
│         [ ▶ Run Test ]    [ 📋 Dry Run ]                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Behavior:**
- Models: Multi-select, max 2. Shows "head-to-head" badge when 2 selected.
- All other options: Single select (radio buttons or dropdowns)
- Run Test: Switches to Monitor tab and begins execution
- Dry Run: Validates config without API calls

### Monitor Screen

Split-pane live logs with per-model progress tracking.

```
┌─────────────────────────────────────────────────────────────────┐
│  ▶ Monitor                                                       │
│  Run: "Claude Haiku vs GPT-5-Mini"         Status: Running ●    │
│  Config: Product Manager · Guided · Enriched · 2-per-type · 35  │
├────────────────────────────────┬────────────────────────────────┤
│  Claude Haiku                  │  GPT-5-Mini                    │
│  ████████████░░░░ 75%          │  ██████████░░░░░░ 62%          │
│  Tokens: 1,247 · $0.003        │  Tokens: 982 · $0.004          │
├────────────────────────────────┼────────────────────────────────┤
│  Log                     [●]   │  Log                     [●]   │
│ ┌────────────────────────────┐ │ ┌────────────────────────────┐ │
│ │ 14:23:01 Step 1: Generate  │ │ │ 14:23:03 Step 1: Generate  │ │
│ │ 14:23:02 Got 8 chips       │ │ │ 14:23:05 Got 9 chips       │ │
│ │ 14:23:02 Selecting 4 chips │ │ │ 14:23:05 Selecting 3 chips │ │
│ │ 14:23:04 Step 2: Final     │ │ │ 14:23:06 Step 2: Final     │ │
│ │ 14:23:05 Generated 35 chips│ │ │ ...                        │ │
│ │ 14:23:05 Checking coverage │ │ │                            │ │
│ │ ▼                          │ │ │ ▼                          │ │
│ └────────────────────────────┘ │ └────────────────────────────┘ │
├────────────────────────────────┴────────────────────────────────┤
│        [ ⏸ Pause ]    [ ⏹ Cancel Run ]    [ → View Results ]    │
└─────────────────────────────────────────────────────────────────┘
```

**Behavior:**
- Split log panes: Each model gets its own scrollable log
- Auto-scroll with pause on interaction: Follows new output, stops when you scroll up
- Independent scroll: Can scroll each log independently
- `[●]` indicator shows auto-scroll state per pane
- Single model run: Full-width log pane

**Keyboard:**
- `1` / `2`: Focus left/right log pane
- `Space`: Toggle auto-scroll on focused pane
- `l`: Link/unlink scroll (both panes scroll together)
- `p`: Pause/resume run
- `Esc`: Cancel (with confirmation)

### Results Screen — Run Browser

Browse and filter completed runs.

```
┌─────────────────────────────────────────────────────────────────┐
│  📊 Results                                                      │
├─────────────────────────────────────────────────────────────────┤
│  Filter: [All personas ▼]  [All models ▼]  [All ratings ▼]      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ▼ Today (3 runs)                                               │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ ● Claude Haiku vs GPT-5-Mini         ★★★★☆  ★★★☆☆         │  │
│  │   Product Manager · Guided · Enriched · 35 chips          │  │
│  │   14:23 · Tokens: 2,229 · $0.007                          │  │
│  ├───────────────────────────────────────────────────────────┤  │
│  │ ○ Gemini Flash vs Llama Scout        --     --            │  │
│  │   Software Engineer · Terse · Basic · 15 chips            │  │
│  │   11:05 · Tokens: 1,102 · $0.003                          │  │
│  ├───────────────────────────────────────────────────────────┤  │
│  │ ○ DeepSeek (solo)                    ★★★★★                │  │
│  │   Nurse · Guided · Enriched · 35 chips                    │  │
│  │   09:30 · Tokens: 1,455 · $0.002                          │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ▶ Yesterday (5 runs)                                           │
│  ▶ Jan 5, 2026 (8 runs)                                         │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│         [Enter] Open selected    [D] Delete    [E] Export CSV   │
└─────────────────────────────────────────────────────────────────┘
```

**Behavior:**
- Grouped by date, expandable sections
- Each row shows: models, ratings, config summary, quick stats
- `Enter` or double-click opens comparison detail view
- Filters narrow the list by persona, model, or rating status
- `j`/`k` or arrows to navigate

### Results Screen — Comparison Detail

Full-width detailed view for reviewing and rating chip quality.

```
┌─────────────────────────────────────────────────────────────────┐
│  📊 Results › Claude Haiku vs GPT-5-Mini              [← Back]  │
│  Product Manager · Guided · Enriched · 2-per-type · 35 chips    │
├─────────────────────────────────────────────────────────────────┤
│  [Chips]  [Stats]  [Raw]                                        │
├────────────────────────────────┬────────────────────────────────┤
│  Claude Haiku           ★★★★☆ │  GPT-5-Mini             ★★★☆☆  │
├────────────────────────────────┼────────────────────────────────┤
│  SITUATION (4)                 │  SITUATION (3)                 │
│  • Deadline Pressure           │  • Time Crunch                 │
│  • Stakeholder Conflict        │  • Cross-team Dependency       │
│  • Resource Constraints        │  • Shifting Priorities         │
│  • Scope Creep                 │                                │
├────────────────────────────────┼────────────────────────────────┤
│  JARGON (8)                    │  JARGON (7)                    │
│  • Sprint Planning             │  • Agile Methodology           │
│  • Backlog Grooming            │  • Scrum Ceremonies            │
│  • Daily Standup               │  • Retrospective               │
│  • Story Points                │  • Velocity Tracking           │
│  • ...                         │  • ...                         │
├────────────────────────────────┼────────────────────────────────┤
│  ROLE TASK (6)                 │  ROLE TASK (5)                 │
│  • Product Roadmapping         │  • Feature Prioritization      │
│  • ...                         │  • ...                         │
├────────────────────────────────┴────────────────────────────────┤
│  Rate: ← Claude Haiku →    [1] [2] [3] [4] [5]       Saved ✓   │
└─────────────────────────────────────────────────────────────────┘
```

**Behavior:**
- Full-width side-by-side comparison, scrollable
- Tabs: Chips (default, grouped by type), Stats (coverage/tokens/cost), Raw (JSON)
- `←`/`→` switches which model you're rating
- `1`-`5` sets rating instantly, saves to SQLite
- `Backspace` or `[← Back]` returns to run browser
- Single model runs: Full-width, no split

## Database Schema

```sql
-- Runs table: one row per test execution
CREATE TABLE runs (
    id              TEXT PRIMARY KEY,  -- UUID
    name            TEXT,              -- "Claude Haiku vs GPT-5-Mini"
    created_at      DATETIME,

    -- Config
    persona         TEXT,              -- "Product Manager"
    prompt_style    TEXT,              -- "guided" | "terse"
    flow            TEXT,              -- "basic" | "enriched"
    constraint_type TEXT,              -- "none" | "2-per-type"
    chip_count      INTEGER            -- 15 | 35
);

-- Results table: one row per model in a run
CREATE TABLE results (
    id              TEXT PRIMARY KEY,
    run_id          TEXT REFERENCES runs(id),
    model           TEXT,              -- "claude-haiku"

    -- Output
    chips           JSON,              -- Full chip array

    -- Stats
    tokens_in       INTEGER,
    tokens_out      INTEGER,
    cost_usd        REAL,
    latency_ms      INTEGER,

    -- Coverage
    situation_count INTEGER,
    jargon_count    INTEGER,
    role_task_count INTEGER,
    environment_count INTEGER,

    -- Rating (nullable until reviewed)
    rating          INTEGER,           -- 1-5, NULL if unrated
    rated_at        DATETIME
);
```

**Design choices:**
- Separate runs/results: Clean 1-to-many for head-to-head runs
- Chips as JSON: Flexible, easy to render, searchable with SQLite JSON functions
- Pre-computed coverage counts: Fast filtering/sorting without parsing JSON
- Rating on results, not runs: Rate each model independently

## File Structure

```
chip-selection-test/
├── tui/
│   ├── __init__.py
│   ├── app.py              # Main Textual app, tab container
│   ├── screens/
│   │   ├── __init__.py
│   │   ├── configure.py    # Model/config selection
│   │   ├── monitor.py      # Split-pane live logs
│   │   ├── results.py      # Run browser
│   │   └── comparison.py   # Detail view for a run
│   └── widgets/
│       ├── __init__.py
│       ├── model_selector.py
│       ├── log_pane.py     # Scrollable log with auto-scroll
│       ├── chip_panel.py   # Chip display grouped by type
│       └── rating_bar.py   # 1-5 star rating widget
├── db/
│   ├── __init__.py
│   ├── schema.py           # SQLite table creation
│   └── repository.py       # CRUD operations
├── services/
│   ├── llm.py              # (existing)
│   ├── generator.py        # (existing)
│   ├── selector.py         # (existing)
│   └── runner.py           # Async test runner (refactored)
├── models/
│   └── chip.py             # (existing)
├── config.py               # (existing)
├── prompts.json            # (existing)
├── test_personas.json      # (existing)
├── main.py                 # New entry point: python main.py
└── benchmark.db            # SQLite database (gitignored)
```

## Dependencies

Add to `pyproject.toml`:
- `textual` — TUI framework

## Key Behaviors Summary

| Area | Key Feature |
|------|-------------|
| Configure | 1-2 models, single test config, head-to-head mode |
| Monitor | Split log panes, auto-scroll with pause, parallel progress |
| Results | Run browser → Comparison detail, full-width chip view |
| Rating | 5-star, `1`-`5` keys, `←`/`→` to switch model, instant save |
| Data | SQLite, runs + results tables, JSON chips |
