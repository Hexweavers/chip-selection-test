import json
from pathlib import Path

from models.chip import Chip, parse_chips_from_json
from services.llm import LLMClient, LLMResponse
from config import PROMPTS_FILE


class ChipGenerator:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self.prompts = self._load_prompts()

    def _load_prompts(self) -> dict:
        with open(PROMPTS_FILE) as f:
            return json.load(f)["styles"]

    def _get_prompt(self, style: str, prompt_key: str) -> dict:
        return self.prompts[style][prompt_key]

    def _format_prompt(self, template: str, **kwargs) -> str:
        result = template
        for key, value in kwargs.items():
            placeholder = "{" + key + "}"
            if placeholder in result:
                result = result.replace(placeholder, str(value))
        return result

    def generate_step1(
        self,
        model: str,
        sector: str,
        desired_role: str,
        style: str,
        constraint: str,
    ) -> tuple[list[Chip], LLMResponse]:
        """Generate 8-10 user-selectable chips (Step 1)."""
        prompt_key = (
            "step1_user_selectable_with_constraint"
            if constraint == "with_constraint"
            else "step1_user_selectable"
        )
        prompt = self._get_prompt(style, prompt_key)

        system = prompt["system"]
        user = self._format_prompt(
            prompt["user"],
            sector=sector,
            desired_role=desired_role,
        )

        response = self.llm.chat(model, system, user)

        if response.error:
            return [], response

        chips, errors = parse_chips_from_json(response.content)
        if errors:
            response.error = "; ".join(errors)

        return chips, response

    def generate_step2_basic(
        self,
        model: str,
        sector: str,
        desired_role: str,
        style: str,
        chip_count: int,
    ) -> tuple[list[Chip], LLMResponse]:
        """Generate final chips without user selections (basic flow)."""
        prompt = self._get_prompt(style, "step2_final_generation_basic")

        system = prompt["system"]
        user = self._format_prompt(
            prompt["user"],
            sector=sector,
            desired_role=desired_role,
            chip_count=chip_count,
        )

        response = self.llm.chat(model, system, user)

        if response.error:
            return [], response

        chips, errors = parse_chips_from_json(response.content)
        if errors:
            response.error = "; ".join(errors)

        return chips, response

    def generate_step2_enriched(
        self,
        model: str,
        sector: str,
        desired_role: str,
        style: str,
        chip_count: int,
        user_selected_chips: list[Chip],
    ) -> tuple[list[Chip], LLMResponse]:
        """Generate final chips with user selections as context (enriched flow)."""
        prompt = self._get_prompt(style, "step2_final_generation")

        # Format user-selected chips for the prompt
        selected_formatted = json.dumps(
            [
                {"key": c.key, "display": c.display, "type": c.type}
                for c in user_selected_chips
            ],
            indent=2,
        )

        system = prompt["system"]
        user = self._format_prompt(
            prompt["user"],
            sector=sector,
            desired_role=desired_role,
            chip_count=chip_count,
            user_selected_chips=selected_formatted,
        )

        response = self.llm.chat(model, system, user)

        if response.error:
            return [], response

        chips, errors = parse_chips_from_json(response.content)
        if errors:
            response.error = "; ".join(errors)

        return chips, response
