#!/usr/bin/env python3
"""
Chip Generation Test Runner

Usage:
    python runner.py --model anthropic/claude-haiku-4.5
    python runner.py --all
    python runner.py --model anthropic/claude-haiku-4.5 --persona tech_pm
    python runner.py --model openai/gpt-5-mini --resume
    python runner.py --model anthropic/claude-haiku-4.5 --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from typing import Optional

from tqdm import tqdm

from config import (
    MODELS,
    PROMPT_STYLES,
    CONSTRAINTS,
    INPUT_TYPES,
    CHIP_COUNTS,
    PERSONAS_FILE,
    MIN_CHIPS_PER_TYPE,
)
from models.chip import Chip, TestResult, TestMetadata
from services.llm import LLMClient
from services.generator import ChipGenerator
from services.selector import ChipSelector
from utils.storage import ResultStorage
from utils.fill import FillService


def load_personas() -> list[dict]:
    with open(PERSONAS_FILE) as f:
        return json.load(f)["personas"]


def run_test(
    model: str,
    persona: dict,
    style: str,
    constraint: str,
    input_type: str,
    chip_count: int,
    generator: ChipGenerator,
    selector: ChipSelector,
    fill_service: FillService,
    dry_run: bool = False,
) -> TestResult:
    """Run a single test configuration."""
    metadata = TestMetadata(
        model=model,
        persona_id=persona["id"],
        sector=persona["sector"],
        desired_role=persona["desired_role"],
        style=style,
        constraint=constraint,
        input_type=input_type,
        chip_count=chip_count,
    )

    result = TestResult(metadata=metadata)

    if dry_run:
        return result

    total_latency = 0
    total_input_tokens = 0
    total_output_tokens = 0

    # Enriched flow: Step 1 + Selection + Step 2
    if input_type == "enriched":
        # Step 1: Generate user-selectable chips
        step1_chips, step1_response = generator.generate_step1(
            model=model,
            sector=persona["sector"],
            desired_role=persona["desired_role"],
            style=style,
            constraint=constraint,
        )
        result.step1_chips = step1_chips
        total_latency += step1_response.latency_ms
        total_input_tokens += step1_response.input_tokens
        total_output_tokens += step1_response.output_tokens

        if step1_response.error:
            result.errors.append(f"Step 1: {step1_response.error}")

        # Chip selection (LLM-as-user)
        if step1_chips:
            selected_chips, select_response = selector.select_chips(
                available_chips=step1_chips,
                persona=persona["persona"],
                style=style,
            )
            result.user_selected_chips = selected_chips
            total_latency += select_response.latency_ms
            total_input_tokens += select_response.input_tokens
            total_output_tokens += select_response.output_tokens

            if select_response.error:
                result.errors.append(f"Selection: {select_response.error}")

        # Step 2: Generate final chips with user selections
        step2_chips, step2_response = generator.generate_step2_enriched(
            model=model,
            sector=persona["sector"],
            desired_role=persona["desired_role"],
            style=style,
            chip_count=chip_count,
            user_selected_chips=result.user_selected_chips,
        )
        result.step2_chips = step2_chips
        total_latency += step2_response.latency_ms
        total_input_tokens += step2_response.input_tokens
        total_output_tokens += step2_response.output_tokens

        if step2_response.error:
            result.errors.append(f"Step 2: {step2_response.error}")

    # Basic flow: Step 2 only
    else:
        step2_chips, step2_response = generator.generate_step2_basic(
            model=model,
            sector=persona["sector"],
            desired_role=persona["desired_role"],
            style=style,
            chip_count=chip_count,
        )
        result.step2_chips = step2_chips
        total_latency += step2_response.latency_ms
        total_input_tokens += step2_response.input_tokens
        total_output_tokens += step2_response.output_tokens

        if step2_response.error:
            result.errors.append(f"Step 2: {step2_response.error}")

    # Merge chips: user_selected + step2
    merged_chips = list(result.user_selected_chips) + list(result.step2_chips)

    # Deduplicate by key
    seen_keys = set()
    unique_chips = []
    for chip in merged_chips:
        if chip.key not in seen_keys:
            seen_keys.add(chip.key)
            unique_chips.append(chip)

    # Check for missing types and fill
    missing_types = fill_service.get_missing_types(unique_chips, MIN_CHIPS_PER_TYPE)
    if missing_types:
        fill_chips, fill_response = fill_service.fill_missing(
            model=model,
            sector=persona["sector"],
            desired_role=persona["desired_role"],
            existing_chips=unique_chips,
            missing_types=missing_types,
            style=style,
        )
        result.fill_chips = fill_chips
        total_latency += fill_response.latency_ms
        total_input_tokens += fill_response.input_tokens
        total_output_tokens += fill_response.output_tokens

        if fill_response.error:
            result.errors.append(f"Fill: {fill_response.error}")

        # Add fill chips to final
        for chip in fill_chips:
            if chip.key not in seen_keys:
                seen_keys.add(chip.key)
                unique_chips.append(chip)

    result.final_chips = unique_chips
    result.latency_ms = total_latency
    result.input_tokens = total_input_tokens
    result.output_tokens = total_output_tokens

    return result


def run_model_batch(
    model_id: str,
    personas: list[dict],
    storage: ResultStorage,
    generator: ChipGenerator,
    selector: ChipSelector,
    fill_service: FillService,
    resume: bool = False,
    dry_run: bool = False,
    persona_filter: str | None = None,
    pbar: tqdm | None = None,
):
    """Run all tests for a single model."""
    # Filter personas if specified
    if persona_filter:
        personas = [p for p in personas if p["id"] == persona_filter]
        if not personas:
            print(f"Error: Persona '{persona_filter}' not found")
            return

    # Generate all test combinations
    combinations = list(
        product(
            personas,
            PROMPT_STYLES,
            CONSTRAINTS,
            INPUT_TYPES,
            CHIP_COUNTS,
        )
    )

    skipped = 0

    for persona, style, constraint, input_type, chip_count in combinations:
        # Check if result exists (for resume)
        if resume and storage.result_exists(
            model_id, persona["id"], style, constraint, input_type, chip_count
        ):
            skipped += 1
            if pbar:
                pbar.update(1)
            continue

        # Update progress bar description
        desc = f"{model_id.split('/')[-1]} | {persona['id']} | {style[:5]} | {input_type[:5]} | {chip_count}"
        if pbar:
            pbar.set_description(desc)

        if dry_run:
            if pbar:
                pbar.update(1)
            continue

        result = run_test(
            model=model_id,
            persona=persona,
            style=style,
            constraint=constraint,
            input_type=input_type,
            chip_count=chip_count,
            generator=generator,
            selector=selector,
            fill_service=fill_service,
            dry_run=dry_run,
        )

        # Save result
        storage.save_result(result)

        # Update progress bar with result info
        counts = result.count_by_type()
        if result.errors:
            if pbar:
                pbar.set_postfix(
                    chips=len(result.final_chips), errors=len(result.errors)
                )
        else:
            fill_str = f"+{len(result.fill_chips)}" if result.fill_chips else ""
            if pbar:
                pbar.set_postfix(
                    chips=len(result.final_chips),
                    S=counts["situation"],
                    J=counts["jargon"],
                    R=counts["role_task"],
                    E=counts["environment"],
                    fill=fill_str or None,
                )

        if pbar:
            pbar.update(1)

    return skipped


def main():
    parser = argparse.ArgumentParser(description="Run chip generation tests")
    parser.add_argument(
        "--model", type=str, help="Model ID to test (e.g., anthropic/claude-haiku-4.5)"
    )
    parser.add_argument("--all", action="store_true", help="Run all models")
    parser.add_argument(
        "--persona", type=str, help="Run only specific persona (e.g., tech_pm)"
    )
    parser.add_argument("--resume", action="store_true", help="Skip existing results")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would run without making API calls",
    )
    parser.add_argument(
        "--list-models", action="store_true", help="List available models"
    )

    args = parser.parse_args()

    if args.list_models:
        print("\nAvailable models:")
        for m in MODELS:
            print(f"  {m.id} ({m.name})")
        return

    if not args.model and not args.all:
        parser.print_help()
        print("\nError: Must specify --model or --all")
        sys.exit(1)

    # Load personas
    personas = load_personas()

    # Filter personas for count calculation
    filtered_personas = personas
    if args.persona:
        filtered_personas = [p for p in personas if p["id"] == args.persona]

    # Calculate total tests
    tests_per_persona = (
        len(PROMPT_STYLES) * len(CONSTRAINTS) * len(INPUT_TYPES) * len(CHIP_COUNTS)
    )
    if args.all:
        total_tests = len(MODELS) * len(filtered_personas) * tests_per_persona
        models_to_run = MODELS
    else:
        total_tests = len(filtered_personas) * tests_per_persona
        models_to_run = [m for m in MODELS if m.id == args.model]

    # Initialize services (skip LLM client for dry run)
    llm_client = None
    generator = None
    selector = None
    fill_service = None

    if not args.dry_run:
        llm_client = LLMClient()
        generator = ChipGenerator(llm_client)
        selector = ChipSelector(llm_client)
        fill_service = FillService(llm_client)

    storage = ResultStorage()

    try:
        with tqdm(total=total_tests, desc="Starting...", unit="test") as pbar:
            for model in models_to_run:
                run_model_batch(
                    model_id=model.id,
                    personas=personas,
                    storage=storage,
                    generator=generator,
                    selector=selector,
                    fill_service=fill_service,
                    resume=args.resume,
                    dry_run=args.dry_run,
                    persona_filter=args.persona,
                    pbar=pbar,
                )
    finally:
        if llm_client:
            llm_client.close()


if __name__ == "__main__":
    main()
