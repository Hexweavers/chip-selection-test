"""Chip generation endpoint with caching."""

from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config import MIN_CHIPS_PER_TYPE, get_model_config
from db import Repository, init_db
from api.services import get_generator, get_selector, get_fill_service, get_personas

router = APIRouter()

INTERACTIVE_RUN_ID = "interactive-ui"


def get_repo():
    db = init_db()
    return Repository(db)


def ensure_interactive_run(repo: Repository) -> None:
    """Create the interactive run if it doesn't exist."""
    run = repo.get_run(INTERACTIVE_RUN_ID)
    if not run:
        now = datetime.now(timezone.utc).isoformat()
        repo.db.execute(
            """
            INSERT OR IGNORE INTO runs (id, started_at, status, config)
            VALUES (?, ?, 'running', '{"source": "interactive-ui"}')
            """,
            (INTERACTIVE_RUN_ID, now),
        )
        repo.db.commit()


def find_cached_result(repo: Repository, params: dict) -> dict | None:
    """Find an exact match in the database."""
    row = repo.db.execute(
        """
        SELECT id FROM results
        WHERE run_id = ? AND model = ? AND persona_id = ?
          AND style = ? AND input_type = ? AND constraint_type = ?
          AND chip_count = ?
        LIMIT 1
        """,
        (
            INTERACTIVE_RUN_ID,
            params["model"],
            params["persona_id"],
            params["style"],
            params["input_type"],
            params["constraint_type"],
            params["chip_count"],
        ),
    ).fetchone()
    return {"id": row[0]} if row else None


class GenerateRequest(BaseModel):
    model: str
    persona_id: str
    style: Literal["terse", "guided"]
    input_type: Literal["basic", "enriched"]
    constraint_type: Literal["with_constraint", "no_constraint"]
    chip_count: Literal[15, 35]


class GenerateResponse(BaseModel):
    cached: bool
    result_id: str
    final_chips: list[dict]
    step1_chips: list[dict] | None = None
    selected_chips: list[dict] | None = None
    fill_chips: list[dict] | None = None
    errors: list[str] | None = None
    latency_ms: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


@router.post("", response_model=GenerateResponse)
def generate_chips(body: GenerateRequest):
    """Generate chips or return cached result if exists."""
    repo = get_repo()
    personas = get_personas()

    if body.persona_id not in personas:
        raise HTTPException(
            status_code=400, detail=f"Unknown persona_id: {body.persona_id}"
        )
    persona = personas[body.persona_id]

    model_config = get_model_config(body.model)
    if not model_config:
        raise HTTPException(status_code=400, detail=f"Unknown model: {body.model}")

    ensure_interactive_run(repo)

    params = {
        "model": body.model,
        "persona_id": body.persona_id,
        "style": body.style,
        "input_type": body.input_type,
        "constraint_type": body.constraint_type,
        "chip_count": body.chip_count,
    }

    cached = find_cached_result(repo, params)
    if cached:
        full_result = repo.get_result(cached["id"])
        return GenerateResponse(
            cached=True,
            result_id=full_result["id"],
            final_chips=full_result["final_chips"],
            step1_chips=full_result.get("step1_chips"),
            selected_chips=full_result.get("selected_chips"),
            fill_chips=full_result.get("fill_chips"),
            errors=full_result.get("errors"),
            latency_ms=full_result.get("latency_ms"),
            input_tokens=full_result.get("input_tokens"),
            output_tokens=full_result.get("output_tokens"),
            cost_usd=full_result.get("cost_usd"),
        )

    generator = get_generator()
    selector = get_selector()
    fill_service = get_fill_service()

    total_latency = 0
    total_input_tokens = 0
    total_output_tokens = 0
    errors = []

    step1_chips = []
    selected_chips = []
    step2_chips = []
    fill_chips = []

    if body.input_type == "enriched":
        step1_result, step1_response = generator.generate_step1(
            model=body.model,
            sector=persona["sector"],
            desired_role=persona["desired_role"],
            style=body.style,
            constraint=body.constraint_type,
        )
        step1_chips = step1_result
        total_latency += step1_response.latency_ms
        total_input_tokens += step1_response.input_tokens
        total_output_tokens += step1_response.output_tokens
        if step1_response.error:
            errors.append(f"Step 1: {step1_response.error}")

        if step1_chips:
            selected_result, select_response = selector.select_chips(
                available_chips=step1_chips,
                persona=persona["persona"],
                style=body.style,
            )
            selected_chips = selected_result
            total_latency += select_response.latency_ms
            total_input_tokens += select_response.input_tokens
            total_output_tokens += select_response.output_tokens
            if select_response.error:
                errors.append(f"Selection: {select_response.error}")

        step2_result, step2_response = generator.generate_step2_enriched(
            model=body.model,
            sector=persona["sector"],
            desired_role=persona["desired_role"],
            style=body.style,
            chip_count=body.chip_count,
            user_selected_chips=selected_chips,
        )
        step2_chips = step2_result
        total_latency += step2_response.latency_ms
        total_input_tokens += step2_response.input_tokens
        total_output_tokens += step2_response.output_tokens
        if step2_response.error:
            errors.append(f"Step 2: {step2_response.error}")

    else:
        step2_result, step2_response = generator.generate_step2_basic(
            model=body.model,
            sector=persona["sector"],
            desired_role=persona["desired_role"],
            style=body.style,
            chip_count=body.chip_count,
        )
        step2_chips = step2_result
        total_latency += step2_response.latency_ms
        total_input_tokens += step2_response.input_tokens
        total_output_tokens += step2_response.output_tokens
        if step2_response.error:
            errors.append(f"Step 2: {step2_response.error}")

    merged_chips = list(selected_chips) + list(step2_chips)
    seen_keys = set()
    unique_chips = []
    for chip in merged_chips:
        if chip.key not in seen_keys:
            seen_keys.add(chip.key)
            unique_chips.append(chip)

    missing_types = fill_service.get_missing_types(unique_chips, MIN_CHIPS_PER_TYPE)
    if missing_types:
        fill_result, fill_response = fill_service.fill_missing(
            model=body.model,
            sector=persona["sector"],
            desired_role=persona["desired_role"],
            existing_chips=unique_chips,
            missing_types=missing_types,
            style=body.style,
        )
        fill_chips = fill_result
        total_latency += fill_response.latency_ms
        total_input_tokens += fill_response.input_tokens
        total_output_tokens += fill_response.output_tokens
        if fill_response.error:
            errors.append(f"Fill: {fill_response.error}")

        for chip in fill_chips:
            if chip.key not in seen_keys:
                seen_keys.add(chip.key)
                unique_chips.append(chip)

    final_chips = unique_chips

    cost_usd = None
    if model_config:
        cost_usd = model_config.calculate_cost(total_input_tokens, total_output_tokens)

    # Try to save, handle race condition where another request inserted first
    try:
        result_id = repo.save_result(
            run_id=INTERACTIVE_RUN_ID,
            model=body.model,
            persona_id=body.persona_id,
            sector=persona["sector"],
            desired_role=persona["desired_role"],
            style=body.style,
            input_type=body.input_type,
            constraint_type=body.constraint_type,
            chip_count=body.chip_count,
            final_chips=[c.to_dict() for c in final_chips],
            step1_chips=[c.to_dict() for c in step1_chips] if step1_chips else None,
            selected_chips=[c.to_dict() for c in selected_chips] if selected_chips else None,
            fill_chips=[c.to_dict() for c in fill_chips] if fill_chips else None,
            errors=errors if errors else None,
            latency_ms=total_latency,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            cost_usd=cost_usd,
        )
    except Exception:
        # Race condition: another request inserted first, return the cached result
        cached = find_cached_result(repo, params)
        if cached:
            full_result = repo.get_result(cached["id"])
            return GenerateResponse(
                cached=True,
                result_id=full_result["id"],
                final_chips=full_result["final_chips"],
                step1_chips=full_result.get("step1_chips"),
                selected_chips=full_result.get("selected_chips"),
                fill_chips=full_result.get("fill_chips"),
                errors=full_result.get("errors"),
                latency_ms=full_result.get("latency_ms"),
                input_tokens=full_result.get("input_tokens"),
                output_tokens=full_result.get("output_tokens"),
                cost_usd=full_result.get("cost_usd"),
            )
        raise  # Re-raise if it wasn't a duplicate key error

    return GenerateResponse(
        cached=False,
        result_id=result_id,
        final_chips=[c.to_dict() for c in final_chips],
        step1_chips=[c.to_dict() for c in step1_chips] if step1_chips else None,
        selected_chips=[c.to_dict() for c in selected_chips] if selected_chips else None,
        fill_chips=[c.to_dict() for c in fill_chips] if fill_chips else None,
        errors=errors if errors else None,
        latency_ms=total_latency,
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
        cost_usd=cost_usd,
    )
