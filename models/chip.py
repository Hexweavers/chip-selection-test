from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal, Optional, List, Dict, Tuple
from datetime import datetime
import json

from config import CHIP_TYPES


ChipType = Literal["situation", "jargon", "role_task", "environment"]


@dataclass
class Chip:
    key: str
    display: str
    type: ChipType

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Chip":
        return cls(
            key=data["key"],
            display=data["display"],
            type=data["type"],
        )

    def validate(self) -> list[str]:
        errors = []
        if not self.key:
            errors.append("key is required")
        if not self.display:
            errors.append("display is required")
        if self.type not in CHIP_TYPES:
            errors.append(f"type must be one of {CHIP_TYPES}, got '{self.type}'")
        return errors


@dataclass
class TestMetadata:
    model: str
    persona_id: str
    sector: str
    desired_role: str
    style: str
    constraint: str
    input_type: str
    chip_count: int
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TestResult:
    metadata: TestMetadata
    step1_chips: list[Chip] = field(default_factory=list)
    user_selected_chips: list[Chip] = field(default_factory=list)
    step2_chips: list[Chip] = field(default_factory=list)
    fill_chips: list[Chip] = field(default_factory=list)
    final_chips: list[Chip] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def to_dict(self) -> dict:
        return {
            "metadata": self.metadata.to_dict(),
            "step1_chips": [c.to_dict() for c in self.step1_chips],
            "user_selected_chips": [c.to_dict() for c in self.user_selected_chips],
            "step2_chips": [c.to_dict() for c in self.step2_chips],
            "fill_chips": [c.to_dict() for c in self.fill_chips],
            "final_chips": [c.to_dict() for c in self.final_chips],
            "errors": self.errors,
            "latency_ms": self.latency_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def count_by_type(self, chips: list[Chip] | None = None) -> dict[str, int]:
        chips = chips or self.final_chips
        counts = {t: 0 for t in CHIP_TYPES}
        for chip in chips:
            if chip.type in counts:
                counts[chip.type] += 1
        return counts

    def get_missing_types(self, min_per_type: int = 2) -> list[str]:
        counts = self.count_by_type()
        return [t for t, count in counts.items() if count < min_per_type]


def parse_chips_from_json(json_str: str) -> tuple[list[Chip], list[str]]:
    """Parse chips from LLM JSON response. Returns (chips, errors)."""
    chips = []
    errors = []

    try:
        # Handle potential markdown code blocks
        cleaned = json_str.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first and last lines (```json and ```)
            cleaned = "\n".join(lines[1:-1])

        data = json.loads(cleaned)

        # Handle both array and object with chips key
        if isinstance(data, dict) and "chips" in data:
            data = data["chips"]

        if not isinstance(data, list):
            errors.append(f"Expected array, got {type(data).__name__}")
            return chips, errors

        for i, item in enumerate(data):
            try:
                chip = Chip.from_dict(item)
                validation_errors = chip.validate()
                if validation_errors:
                    errors.append(f"Chip {i}: {', '.join(validation_errors)}")
                else:
                    chips.append(chip)
            except (KeyError, TypeError) as e:
                errors.append(f"Chip {i}: {str(e)}")

    except json.JSONDecodeError as e:
        errors.append(f"JSON parse error: {str(e)}")

    return chips, errors
