# Chip Generation Test

Benchmark LLM chip generation quality for WinSpeak's sign-up flow.

**Chips** = structured tags (situation, jargon, role_task, environment) personalized to a user's sector + role.

## What This Tests

- **10 models** via OpenRouter (Claude Haiku, GPT-5-Mini, Gemini Flash, etc.)
- **2 prompt styles** (terse vs guided)
- **2 input flows** (basic: sector+role only vs enriched: with user-selected chips)
- **6 test personas** (PM, SWE, Nurse, Analyst, Store Manager, UX Designer)

960 total test runs. Results saved to `results/` as JSON + CSV.

## Setup

```bash
uv sync
echo "OPENROUTER_API_KEY=your-key" > .env
```

## Run

```bash
# Single model
uv run python runner.py --model anthropic/claude-haiku-4.5

# All models
uv run python runner.py --all

# Dry run (no API calls)
uv run python runner.py --model anthropic/claude-haiku-4.5 --dry-run

# Resume interrupted run
uv run python runner.py --model openai/gpt-5-mini --resume

# List available models
uv run python runner.py --list-models
```

## Output

- `results/<model>/` - JSON + CSV per test configuration
- `results/summary.csv` - All results flattened for spreadsheet review

## Cost

~$5-10 for full run across all models.
