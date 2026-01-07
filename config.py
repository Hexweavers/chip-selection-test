import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class ModelConfig:
    id: str
    name: str


# OpenRouter API settings
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Models to test
MODELS = [
    ModelConfig("anthropic/claude-haiku-4.5", "Claude Haiku 4.5"),
    ModelConfig("openai/gpt-5-mini", "GPT-5 Mini"),
    ModelConfig("meta-llama/llama-scout-4-12b", "Llama Scout 4 12B"),
    ModelConfig("google/gemini-3-flash-preview", "Gemini 3 Flash Preview"),
    ModelConfig("google/gemini-2.5-flash-lite", "Gemini 2.5 Flash Lite"),
    ModelConfig("qwen/qwen3-next-80b-a3b-instruct", "Qwen3 Next 80B"),
    ModelConfig("minimax/minimax-m2.1", "MiniMax M2.1"),
    ModelConfig("deepseek/deepseek-v3.2", "DeepSeek V3.2"),
    ModelConfig("x-ai/grok-4.1-fast", "Grok 4.1 Fast"),
    ModelConfig("mistralai/mistral-nemo", "Mistral Nemo"),
]

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
