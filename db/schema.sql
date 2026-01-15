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

-- Unique constraint for interactive-ui results to prevent race condition duplicates
CREATE UNIQUE INDEX IF NOT EXISTS idx_results_unique_params
ON results(run_id, model, persona_id, style, input_type, constraint_type, chip_count);
