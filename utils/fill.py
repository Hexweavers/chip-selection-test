import json

from models.chip import Chip, parse_chips_from_json
from services.llm import LLMClient, LLMResponse
from config import PROMPTS_FILE, MIN_CHIPS_PER_TYPE, CHIP_TYPES


class FillService:
    """Fill missing chip types to ensure minimum coverage."""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self.prompts = self._load_prompts()

    def _load_prompts(self) -> dict:
        with open(PROMPTS_FILE) as f:
            return json.load(f)["styles"]

    def get_missing_types(self, chips: list[Chip], min_per_type: int = MIN_CHIPS_PER_TYPE) -> list[str]:
        """Find chip types that have fewer than min_per_type chips."""
        counts = {t: 0 for t in CHIP_TYPES}
        for chip in chips:
            if chip.type in counts:
                counts[chip.type] += 1
        return [t for t, count in counts.items() if count < min_per_type]

    def fill_missing(
        self,
        model: str,
        sector: str,
        desired_role: str,
        existing_chips: list[Chip],
        missing_types: list[str],
        style: str,
    ) -> tuple[list[Chip], LLMResponse]:
        """Generate chips to fill missing types."""
        if not missing_types:
            return [], LLMResponse(content="", input_tokens=0, output_tokens=0, latency_ms=0)

        prompt = self.prompts[style]["fill_missing_types"]

        # Format existing chips
        existing_formatted = json.dumps(
            [{"key": c.key, "display": c.display, "type": c.type} for c in existing_chips],
            indent=2,
        )

        system = prompt["system"]
        user = (
            prompt["user"]
            .replace("{sector}", sector)
            .replace("{desired_role}", desired_role)
            .replace("{existing_chips}", existing_formatted)
            .replace("{missing_types}", ", ".join(missing_types))
        )

        response = self.llm.chat(model, system, user)

        if response.error:
            return [], response

        chips, errors = parse_chips_from_json(response.content)
        if errors:
            response.error = "; ".join(errors)

        return chips, response
