"""Async test runner with event streaming for TUI consumption."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, AsyncIterator

from config import (
    CHIP_TYPES,
    MIN_CHIPS_PER_TYPE,
    PERSONAS_FILE,
    PROMPTS_FILE,
    SELECTOR_MODEL,
    USER_SELECTION_MAX,
    USER_SELECTION_MIN,
)
from models.chip import Chip, parse_chips_from_json
from services.llm_async import AsyncLLMClient, LLMResponse


class EventType(Enum):
    """Types of events emitted by the async runner."""

    LOG = auto()
    PROGRESS = auto()
    COMPLETE = auto()
    ERROR = auto()


@dataclass
class RunEvent:
    """Event emitted during test execution."""

    type: EventType
    model: str
    message: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunConfig:
    """Configuration for a test run."""

    models: list[str]
    persona_id: str
    prompt_style: str
    flow: str  # "basic" or "enriched"
    constraint_type: str  # "2-per-type" or "none"
    chip_count: int

    @property
    def is_head_to_head(self) -> bool:
        """Returns True if multiple models are being compared."""
        return len(self.models) > 1


@dataclass
class ModelResult:
    """Result from running a single model."""

    model: str
    chips: list[Chip]
    tokens_in: int
    tokens_out: int
    latency_ms: int
    cost_usd: float
    counts: dict[str, int]
    errors: list[str]


def load_personas() -> dict[str, dict]:
    """Load personas from JSON file."""
    with open(PERSONAS_FILE) as f:
        data = json.load(f)
    return {p["id"]: p for p in data["personas"]}


def load_prompts() -> dict:
    """Load prompts from JSON file."""
    with open(PROMPTS_FILE) as f:
        return json.load(f)["styles"]


class AsyncRunner:
    """Async test runner that yields events for TUI consumption."""

    def __init__(self):
        self._client: AsyncLLMClient | None = None
        self._cancelled = False
        self._running = False
        self._personas = load_personas()
        self._prompts = load_prompts()

    async def start(self):
        """Initialize the runner and LLM client."""
        if self._client is None:
            self._client = AsyncLLMClient()
        self._cancelled = False
        self._running = True

    async def stop(self):
        """Stop the runner and close the LLM client."""
        self._running = False
        if self._client is not None:
            await self._client.close()
            self._client = None

    def cancel(self):
        """Request cancellation of the current run."""
        self._cancelled = True

    async def run(self, config: RunConfig) -> AsyncIterator[RunEvent]:
        """Run tests for all models in config, yielding events."""
        await self.start()

        try:
            # Run models in parallel
            tasks = [
                asyncio.create_task(self._collect_model_events(model, config))
                for model in config.models
            ]

            # As each model completes, yield its events
            for completed_task in asyncio.as_completed(tasks):
                events = await completed_task
                for event in events:
                    yield event
                    if self._cancelled:
                        # Cancel remaining tasks
                        for task in tasks:
                            if not task.done():
                                task.cancel()
                        return
        finally:
            await self.stop()

    async def _collect_model_events(
        self, model: str, config: RunConfig
    ) -> list[RunEvent]:
        """Run a single model and collect all events."""
        events = []
        async for event in self._run_model(model, config):
            events.append(event)
        return events

    async def _run_model(
        self, model: str, config: RunConfig
    ) -> AsyncIterator[RunEvent]:
        """Run the test flow for a single model, yielding events."""
        if self._client is None:
            yield RunEvent(
                type=EventType.ERROR,
                model=model,
                message="Client not initialized",
            )
            return

        persona = self._personas.get(config.persona_id)
        if persona is None:
            yield RunEvent(
                type=EventType.ERROR,
                model=model,
                message=f"Persona not found: {config.persona_id}",
            )
            return

        sector = persona["sector"]
        desired_role = persona["desired_role"]
        persona_text = persona["persona"]

        total_tokens_in = 0
        total_tokens_out = 0
        total_latency_ms = 0
        errors: list[str] = []
        all_chips: list[Chip] = []
        step1_chips: list[Chip] = []
        user_selected_chips: list[Chip] = []

        # Determine constraint for step1
        constraint = (
            "with_constraint" if config.constraint_type == "2-per-type" else "no_constraint"
        )

        try:
            # Step 1: Generate user-selectable chips (only for enriched flow)
            if config.flow == "enriched":
                yield RunEvent(
                    type=EventType.LOG,
                    model=model,
                    message="Generating user-selectable chips (Step 1)...",
                )
                yield RunEvent(
                    type=EventType.PROGRESS,
                    model=model,
                    message="Step 1",
                    data={"percent": 10},
                )

                step1_chips, response = await self._generate_step1(
                    model, sector, desired_role, config.prompt_style, constraint
                )

                total_tokens_in += response.input_tokens
                total_tokens_out += response.output_tokens
                total_latency_ms += response.latency_ms

                if response.error:
                    errors.append(f"Step 1: {response.error}")
                    yield RunEvent(
                        type=EventType.ERROR,
                        model=model,
                        message=f"Step 1 error: {response.error}",
                    )

                if self._cancelled:
                    return

                yield RunEvent(
                    type=EventType.LOG,
                    model=model,
                    message=f"Generated {len(step1_chips)} user-selectable chips",
                )
                yield RunEvent(
                    type=EventType.PROGRESS,
                    model=model,
                    message="Step 1 complete",
                    data={"percent": 25},
                )

                # Simulate user selection using selector model
                if step1_chips:
                    yield RunEvent(
                        type=EventType.LOG,
                        model=model,
                        message="Simulating user chip selection...",
                    )

                    user_selected_chips, response = await self._select_chips(
                        step1_chips, persona_text, config.prompt_style
                    )

                    total_tokens_in += response.input_tokens
                    total_tokens_out += response.output_tokens
                    total_latency_ms += response.latency_ms

                    if response.error:
                        errors.append(f"Selection: {response.error}")

                    yield RunEvent(
                        type=EventType.LOG,
                        model=model,
                        message=f"User selected {len(user_selected_chips)} chips",
                    )

                if self._cancelled:
                    return

                yield RunEvent(
                    type=EventType.PROGRESS,
                    model=model,
                    message="Selection complete",
                    data={"percent": 40},
                )

            # Step 2: Generate final chips
            yield RunEvent(
                type=EventType.LOG,
                model=model,
                message=f"Generating {config.chip_count} final chips (Step 2)...",
            )
            yield RunEvent(
                type=EventType.PROGRESS,
                model=model,
                message="Step 2",
                data={"percent": 50},
            )

            if config.flow == "basic":
                step2_chips, response = await self._generate_step2_basic(
                    model, sector, desired_role, config.prompt_style, config.chip_count
                )
            else:
                step2_chips, response = await self._generate_step2_enriched(
                    model,
                    sector,
                    desired_role,
                    config.prompt_style,
                    config.chip_count,
                    user_selected_chips,
                )

            total_tokens_in += response.input_tokens
            total_tokens_out += response.output_tokens
            total_latency_ms += response.latency_ms

            if response.error:
                errors.append(f"Step 2: {response.error}")
                yield RunEvent(
                    type=EventType.ERROR,
                    model=model,
                    message=f"Step 2 error: {response.error}",
                )

            all_chips = step2_chips

            if self._cancelled:
                return

            yield RunEvent(
                type=EventType.LOG,
                model=model,
                message=f"Generated {len(step2_chips)} chips",
            )
            yield RunEvent(
                type=EventType.PROGRESS,
                model=model,
                message="Step 2 complete",
                data={"percent": 75},
            )

            # Fill missing types if constraint is set
            if config.constraint_type == "2-per-type":
                missing_types = self._get_missing_types(all_chips)
                if missing_types:
                    yield RunEvent(
                        type=EventType.LOG,
                        model=model,
                        message=f"Filling missing types: {missing_types}",
                    )

                    fill_chips, response = await self._fill_missing(
                        model,
                        sector,
                        desired_role,
                        config.prompt_style,
                        all_chips,
                        missing_types,
                    )

                    total_tokens_in += response.input_tokens
                    total_tokens_out += response.output_tokens
                    total_latency_ms += response.latency_ms

                    if response.error:
                        errors.append(f"Fill: {response.error}")

                    all_chips.extend(fill_chips)

                    yield RunEvent(
                        type=EventType.LOG,
                        model=model,
                        message=f"Added {len(fill_chips)} chips to fill missing types",
                    )

            if self._cancelled:
                return

            yield RunEvent(
                type=EventType.PROGRESS,
                model=model,
                message="Finalizing",
                data={"percent": 90},
            )

            # Calculate counts
            counts = self._count_by_type(all_chips)

            # Calculate cost (simplified - in real impl would use model pricing)
            cost_usd = (total_tokens_in * 0.001 + total_tokens_out * 0.002) / 1000

            result = ModelResult(
                model=model,
                chips=all_chips,
                tokens_in=total_tokens_in,
                tokens_out=total_tokens_out,
                latency_ms=total_latency_ms,
                cost_usd=cost_usd,
                counts=counts,
                errors=errors,
            )

            yield RunEvent(
                type=EventType.PROGRESS,
                model=model,
                message="Complete",
                data={"percent": 100},
            )

            yield RunEvent(
                type=EventType.COMPLETE,
                model=model,
                message=f"Completed with {len(all_chips)} chips",
                data={
                    "result": {
                        "model": result.model,
                        "chip_count": len(result.chips),
                        "tokens_in": result.tokens_in,
                        "tokens_out": result.tokens_out,
                        "latency_ms": result.latency_ms,
                        "cost_usd": result.cost_usd,
                        "counts": result.counts,
                        "errors": result.errors,
                    }
                },
            )

        except Exception as e:
            yield RunEvent(
                type=EventType.ERROR,
                model=model,
                message=f"Unexpected error: {str(e)}",
            )

    async def _generate_step1(
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

        response = await self._client.chat(model, system, user)

        if response.error:
            return [], response

        chips, parse_errors = parse_chips_from_json(response.content)
        if parse_errors:
            response.error = "; ".join(parse_errors)

        return chips, response

    async def _select_chips(
        self,
        available_chips: list[Chip],
        persona: str,
        style: str,
    ) -> tuple[list[Chip], LLMResponse]:
        """Simulate user selection of chips using the selector model."""
        prompt = self._get_prompt(style, "chip_selector")

        available_formatted = json.dumps(
            [{"key": c.key, "display": c.display, "type": c.type} for c in available_chips],
            indent=2,
        )

        system = prompt["system"]
        user = self._format_prompt(
            prompt["user"],
            persona=persona,
            available_chips=available_formatted,
        )

        response = await self._client.chat(SELECTOR_MODEL.id, system, user)

        if response.error:
            return [], response

        try:
            # Parse selected keys from response
            cleaned = response.content.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                cleaned = "\n".join(lines[1:-1])

            selected_keys = json.loads(cleaned)

            # Clamp selection count
            if len(selected_keys) < USER_SELECTION_MIN:
                selected_keys = selected_keys[:USER_SELECTION_MIN] if selected_keys else []
            elif len(selected_keys) > USER_SELECTION_MAX:
                selected_keys = selected_keys[:USER_SELECTION_MAX]

            # Map keys back to chips
            key_to_chip = {c.key: c for c in available_chips}
            selected_chips = [
                key_to_chip[k] for k in selected_keys if k in key_to_chip
            ]

            return selected_chips, response

        except (json.JSONDecodeError, TypeError) as e:
            response.error = f"Selection parse error: {str(e)}"
            return [], response

    async def _generate_step2_basic(
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

        response = await self._client.chat(model, system, user)

        if response.error:
            return [], response

        chips, parse_errors = parse_chips_from_json(response.content)
        if parse_errors:
            response.error = "; ".join(parse_errors)

        return chips, response

    async def _generate_step2_enriched(
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

        response = await self._client.chat(model, system, user)

        if response.error:
            return [], response

        chips, parse_errors = parse_chips_from_json(response.content)
        if parse_errors:
            response.error = "; ".join(parse_errors)

        return chips, response

    async def _fill_missing(
        self,
        model: str,
        sector: str,
        desired_role: str,
        style: str,
        existing_chips: list[Chip],
        missing_types: list[str],
    ) -> tuple[list[Chip], LLMResponse]:
        """Generate chips to fill missing types."""
        prompt = self._get_prompt(style, "fill_missing_types")

        existing_formatted = json.dumps(
            [
                {"key": c.key, "display": c.display, "type": c.type}
                for c in existing_chips
            ],
            indent=2,
        )

        system = prompt["system"]
        user = self._format_prompt(
            prompt["user"],
            sector=sector,
            desired_role=desired_role,
            existing_chips=existing_formatted,
            missing_types=", ".join(missing_types),
        )

        response = await self._client.chat(model, system, user)

        if response.error:
            return [], response

        chips, parse_errors = parse_chips_from_json(response.content)
        if parse_errors:
            response.error = "; ".join(parse_errors)

        return chips, response

    def _get_prompt(self, style: str, prompt_key: str) -> dict:
        """Get a prompt template by style and key."""
        return self._prompts[style][prompt_key]

    def _format_prompt(self, template: str, **kwargs) -> str:
        """Format a prompt template with the given values."""
        result = template
        for key, value in kwargs.items():
            placeholder = "{" + key + "}"
            if placeholder in result:
                result = result.replace(placeholder, str(value))
        return result

    def _get_missing_types(
        self, chips: list[Chip], min_per_type: int = MIN_CHIPS_PER_TYPE
    ) -> list[str]:
        """Get chip types that have fewer than min_per_type chips."""
        counts = self._count_by_type(chips)
        return [t for t, count in counts.items() if count < min_per_type]

    def _count_by_type(self, chips: list[Chip]) -> dict[str, int]:
        """Count chips by type."""
        counts = {t: 0 for t in CHIP_TYPES}
        for chip in chips:
            if chip.type in counts:
                counts[chip.type] += 1
        return counts
