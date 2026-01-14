import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


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


# OpenRouter API settings
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

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

# Fixed model for chip selection (LLM-as-user)
SELECTOR_MODEL = ModelConfig("google/gemini-2.5-flash", "Gemini 2.5 Flash")

# Test variables
PROMPT_STYLES = ["terse", "guided"]
CONSTRAINTS = ["with_constraint", "no_constraint"]
INPUT_TYPES = ["basic", "enriched"]
CHIP_COUNTS = [15, 35]

# Chip types
CHIP_TYPES = ["situation", "jargon", "role_task", "environment"]
MIN_CHIPS_PER_TYPE = 2

# Step 1 settings
STEP1_CHIP_COUNT_MIN = 8
STEP1_CHIP_COUNT_MAX = 10
USER_SELECTION_MIN = 3
USER_SELECTION_MAX = 5

# File paths
PROMPTS_FILE = "prompts.json"
PERSONAS_FILE = "test_personas.json"
RESULTS_DIR = "results"

# Database settings
TURSO_DATABASE_URL = os.getenv("TURSO_DATABASE_URL")
TURSO_AUTH_TOKEN = os.getenv("TURSO_AUTH_TOKEN")
