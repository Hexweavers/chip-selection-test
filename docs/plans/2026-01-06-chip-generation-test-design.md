# Chip Generation Test Design

## Overview

Test chip generation quality across different LLMs, prompt styles, and input configurations for WinSpeak's sign-up flow.

## Test Variables

| Variable | Values | Count |
|----------|--------|-------|
| Model | Claude Haiku 4.5, GPT-5-Mini, Llama Scout 4 12B, Gemini 3 Flash Preview, Gemini 2.5 Flash-Lite, Qwen3-Next 80B, MiniMax M2.1, DeepSeek V3.2, Grok 4.1 Fast, Mistral Nemo | 10 |
| Prompt style | `terse`, `guided` | 2 |
| Constraint | `with_constraint` (MUST 2 per type), `no_constraint` | 2 |
| Input type | `basic` (sector+role), `enriched` (sector+role+user_selected_chips) | 2 |
| Chip count | 15, 35 | 2 |
| Test personas | tech_pm, tech_swe, healthcare_nurse, finance_analyst, retail_manager, creative_ux | 6 |

**Total Step 2 calls:** 10 × 2 × 2 × 2 × 2 × 6 = 960
**Total estimated calls:** ~1,590 (including Step 1, chip selection, fills)
**Estimated cost:** $3-8 (budget $15-20)

## Chip Schema

```json
{
  "key": "stakeholder_conflict",
  "display": "Stakeholder Conflict",
  "type": "situation"
}
```

**Chip types:** `situation`, `jargon`, `role_task`, `environment`

## Test Flows

### Basic Flow (input_type = "basic")
```
sector + desired_role
  → Step 2: Generate final chips
  → Check type coverage
  → Fill if any type < 2 chips
  → Save results
```

### Enriched Flow (input_type = "enriched")
```
sector + desired_role
  → Step 1: Generate 8-10 selectable chips
  → Chip selector (Gemini 2.5 Flash): Pick 3-5 based on persona
  → Step 2: Generate final chips with selections as context
  → Merge: user_selected + generated
  → Check type coverage
  → Fill if any type < 2 chips
  → Save results
```

## Models

All via OpenRouter:

| Model | OpenRouter ID |
|-------|---------------|
| Claude Haiku 4.5 | `anthropic/claude-haiku-4.5` |
| GPT-5-Mini | `openai/gpt-5-mini` |
| Llama Scout 4 12B | `meta-llama/llama-scout-4-12b` |
| Gemini 3 Flash Preview | `google/gemini-3-flash-preview` |
| Gemini 2.5 Flash-Lite | `google/gemini-2.5-flash-lite` |
| Qwen3-Next 80B | `qwen/qwen3-next-80b-a3b-instruct` |
| MiniMax M2.1 | `minimax/minimax-m2.1` |
| DeepSeek V3.2 | `deepseek/deepseek-v3.2` |
| Grok 4.1 Fast | `x-ai/grok-4.1-fast` |
| Mistral Nemo | `mistralai/mistral-nemo` |

**Chip selector model (fixed):** `google/gemini-2.5-flash` (Gemini 2.5 Flash)

## File Structure

```
chip-selection-test/
  ├── prompts.json              # Prompt templates
  ├── test_personas.json        # 6 test personas
  ├── config.py                 # Models list, OpenRouter settings
  ├── runner.py                 # Main test runner
  ├── models/
  │   └── chip.py               # Chip data model, validation
  ├── services/
  │   ├── llm.py                # OpenRouter API wrapper
  │   ├── generator.py          # Chip generation logic
  │   └── selector.py           # LLM-as-user chip selection
  ├── utils/
  │   ├── storage.py            # Save results to JSON/CSV
  │   └── fill.py               # Fill missing chip types
  └── results/
      ├── anthropic--claude-haiku-4.5/
      │   ├── tech_pm_terse_constrained_basic_15.json
      │   └── ...
      └── summary.csv
```

## Runner CLI

```bash
# Run all tests for a specific model
python runner.py --model anthropic/claude-haiku-4.5

# Run all models sequentially
python runner.py --all

# Run specific persona (debugging)
python runner.py --model anthropic/claude-haiku-4.5 --persona tech_pm

# Resume (skip existing results)
python runner.py --model openai/gpt-5-mini --resume

# Dry run
python runner.py --model anthropic/claude-haiku-4.5 --dry-run
```

## Output Format

### Per-test JSON
```json
{
  "metadata": {
    "model": "anthropic/claude-haiku-4.5",
    "persona_id": "tech_pm",
    "style": "terse",
    "constraint": "with_constraint",
    "input_type": "enriched",
    "chip_count": 15,
    "timestamp": "2026-01-06T..."
  },
  "step1_chips": [],
  "user_selected_chips": [],
  "step2_chips": [],
  "fill_chips": [],
  "final_chips": [],
  "errors": []
}
```

### Summary CSV columns
```
model, persona_id, sector, desired_role, style, constraint, input_type,
chip_count_requested, final_chip_count, situation_count, jargon_count,
role_task_count, environment_count, fill_needed, fill_count,
step1_chips_json, selected_chips_json, final_chips_json,
errors, timestamp, latency_ms, input_tokens, output_tokens
```

## Error Handling

- **Strategy:** Log and continue
- No retries, log failures, continue with next call
- Review failures after batch completes

## Evaluation

After running:
1. Open `results/summary.csv` in Google Sheets
2. Filter/sort by model, persona, style, etc.
3. Manually review chip quality
4. Add scoring columns (relevance 1-5, specificity 1-5, etc.)

## Key Files Already Created

- `prompts.json` - Terse and guided prompt templates for all steps
- `test_personas.json` - 6 detailed work-focused personas
