import json

from models.chip import Chip
from services.llm import LLMClient, LLMResponse
from config import PROMPTS_FILE, SELECTOR_MODEL


class ChipSelector:
    """Simulates user chip selection using LLM-as-user."""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self.prompts = self._load_prompts()
        self.model = SELECTOR_MODEL.id

    def _load_prompts(self) -> dict:
        with open(PROMPTS_FILE) as f:
            return json.load(f)["styles"]

    def select_chips(
        self,
        available_chips: list[Chip],
        persona: str,
        style: str,
    ) -> tuple[list[Chip], LLMResponse]:
        """Select 3-5 chips that a user with this persona would choose."""
        prompt = self.prompts[style]["chip_selector"]

        # Format available chips for display
        chips_formatted = json.dumps(
            [{"key": c.key, "display": c.display, "type": c.type} for c in available_chips],
            indent=2,
        )

        system = prompt["system"]
        user = prompt["user"].replace("{persona}", persona).replace("{available_chips}", chips_formatted)

        response = self.llm.chat(self.model, system, user)

        if response.error:
            return [], response

        # Parse selected chip keys from response
        selected_chips = []
        try:
            cleaned = response.content.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                cleaned = "\n".join(lines[1:-1])

            selected_keys = json.loads(cleaned)

            if not isinstance(selected_keys, list):
                response.error = f"Expected array of keys, got {type(selected_keys).__name__}"
                return [], response

            # Map keys back to chips
            chip_map = {c.key: c for c in available_chips}
            for key in selected_keys:
                if key in chip_map:
                    selected_chips.append(chip_map[key])
                else:
                    # Try to find partial match (in case LLM slightly modified the key)
                    for chip_key, chip in chip_map.items():
                        if key.lower() == chip_key.lower():
                            selected_chips.append(chip)
                            break

        except json.JSONDecodeError as e:
            response.error = f"Failed to parse selection: {str(e)}"

        return selected_chips, response
