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

    total = len(combinations)
    completed = 0
    skipped = 0

    print(f"\n{'='*60}")
    print(f"Running {total} tests for {model_id}")
    print(f"{'='*60}\n")

    for persona, style, constraint, input_type, chip_count in combinations:
        # Check if result exists (for resume)
        if resume and storage.result_exists(
            model_id, persona["id"], style, constraint, input_type, chip_count
        ):
            skipped += 1
            continue

        completed += 1
        progress = f"[{completed}/{total - skipped}]"

        # Status line
        status = f"{progress} {persona['id']} | {style} | {constraint} | {input_type} | {chip_count}"

        if dry_run:
            print(f"{status} → [DRY RUN]")
            continue

        print(f"{status} → ", end="", flush=True)

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

        # Print result summary
        counts = result.count_by_type()
        type_summary = f"S:{counts['situation']} J:{counts['jargon']} R:{counts['role_task']} E:{counts['environment']}"

        if result.errors:
            print(
                f"⚠ {len(result.final_chips)} chips ({type_summary}) - {len(result.errors)} errors"
            )
        else:
            fill_note = f" +{len(result.fill_chips)} fill" if result.fill_chips else ""
            print(f"✓ {len(result.final_chips)} chips ({type_summary}){fill_note}")

    print(f"\n{'='*60}")
    print(f"Completed: {completed} | Skipped: {skipped}")
    print(f"{'='*60}\n")


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
        if args.all:
            for model in MODELS:
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
                )
        else:
            # Validate model exists
            valid_ids = [m.id for m in MODELS]
            if args.model not in valid_ids:
                print(f"Error: Unknown model '{args.model}'")
                print(f"Use --list-models to see available models")
                sys.exit(1)

            run_model_batch(
                model_id=args.model,
                personas=personas,
                storage=storage,
                generator=generator,
                selector=selector,
                fill_service=fill_service,
                resume=args.resume,
                dry_run=args.dry_run,
                persona_filter=args.persona,
            )
    finally:
        if llm_client:
            llm_client.close()


if __name__ == "__main__":
    main()
