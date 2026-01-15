"""Lazy-loaded service singletons for Lambda cold start optimization."""

import json
import threading

from config import PERSONAS_FILE
from services.llm import LLMClient
from services.generator import ChipGenerator
from services.selector import ChipSelector
from utils.fill import FillService

_lock = threading.RLock()
_llm_client: LLMClient | None = None
_generator: ChipGenerator | None = None
_selector: ChipSelector | None = None
_fill_service: FillService | None = None
_personas: dict[str, dict] | None = None


def get_llm_client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        with _lock:
            if _llm_client is None:
                _llm_client = LLMClient()
    return _llm_client


def get_generator() -> ChipGenerator:
    global _generator
    if _generator is None:
        with _lock:
            if _generator is None:
                _generator = ChipGenerator(get_llm_client())
    return _generator


def get_selector() -> ChipSelector:
    global _selector
    if _selector is None:
        with _lock:
            if _selector is None:
                _selector = ChipSelector(get_llm_client())
    return _selector


def get_fill_service() -> FillService:
    global _fill_service
    if _fill_service is None:
        with _lock:
            if _fill_service is None:
                _fill_service = FillService(get_llm_client())
    return _fill_service


def get_personas() -> dict[str, dict]:
    """Load and cache personas from file. Returns dict keyed by persona id."""
    global _personas
    if _personas is None:
        with _lock:
            if _personas is None:
                with open(PERSONAS_FILE) as f:
                    personas_list = json.load(f)["personas"]
                _personas = {p["id"]: p for p in personas_list}
    return _personas


def get_personas_list() -> list[dict]:
    """Return personas as a list for the options endpoint."""
    personas = get_personas()
    return list(personas.values())
