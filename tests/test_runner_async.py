import pytest
from dataclasses import dataclass
from typing import AsyncIterator

from services.runner_async import AsyncRunner, RunConfig, RunEvent, EventType


def test_run_config_creation():
    """RunConfig should hold test configuration."""
    config = RunConfig(
        models=["claude-haiku", "gpt-5-mini"],
        persona_id="tech_pm",
        prompt_style="guided",
        flow="enriched",
        constraint_type="2-per-type",
        chip_count=35,
    )
    assert config.models == ["claude-haiku", "gpt-5-mini"]
    assert config.is_head_to_head is True


def test_run_config_single_model():
    """RunConfig with one model should not be head-to-head."""
    config = RunConfig(
        models=["claude-haiku"],
        persona_id="tech_pm",
        prompt_style="guided",
        flow="enriched",
        constraint_type="2-per-type",
        chip_count=35,
    )
    assert config.is_head_to_head is False


def test_run_event_types():
    """RunEvent should have expected event types."""
    assert EventType.LOG is not None
    assert EventType.PROGRESS is not None
    assert EventType.COMPLETE is not None
    assert EventType.ERROR is not None
