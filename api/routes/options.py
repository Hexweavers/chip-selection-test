"""Configuration options endpoint for the UI."""

from fastapi import APIRouter

from config import MODELS, PROMPT_STYLES, CONSTRAINTS, INPUT_TYPES, CHIP_COUNTS
from api.services import get_personas_list

router = APIRouter()


@router.get("")
def get_options() -> dict:
    """Return available configuration options for the UI."""
    personas = get_personas_list()

    return {
        "models": [{"id": m.id, "name": m.name} for m in MODELS],
        "personas": [
            {
                "id": p["id"],
                "sector": p["sector"],
                "desired_role": p["desired_role"],
            }
            for p in personas
        ],
        "styles": PROMPT_STYLES,
        "input_types": INPUT_TYPES,
        "constraint_types": CONSTRAINTS,
        "chip_counts": CHIP_COUNTS,
    }
