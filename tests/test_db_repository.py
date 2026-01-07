import tempfile
from pathlib import Path
from datetime import datetime
import json

import pytest

from db.schema import init_db
from db.repository import Repository


@pytest.fixture
def repo():
    """Create a repository with a temporary database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        conn = init_db(db_path)
        yield Repository(conn)
        conn.close()


def test_create_run(repo):
    """Should create a run and return its ID."""
    run_id = repo.create_run(
        name="Test Run",
        persona="tech_pm",
        prompt_style="guided",
        flow="enriched",
        constraint_type="2-per-type",
        chip_count=35,
    )
    assert run_id is not None
    assert len(run_id) == 36  # UUID format


def test_get_run(repo):
    """Should retrieve a run by ID."""
    run_id = repo.create_run(
        name="Test Run",
        persona="tech_pm",
        prompt_style="guided",
        flow="enriched",
        constraint_type="2-per-type",
        chip_count=35,
    )
    run = repo.get_run(run_id)
    assert run["name"] == "Test Run"
    assert run["persona"] == "tech_pm"


def test_create_result(repo):
    """Should create a result linked to a run."""
    run_id = repo.create_run(
        name="Test",
        persona="tech_pm",
        prompt_style="guided",
        flow="enriched",
        constraint_type="2-per-type",
        chip_count=35,
    )
    chips = [{"key": "test", "display": "Test", "type": "situation"}]
    result_id = repo.create_result(
        run_id=run_id,
        model="claude-haiku",
        chips=chips,
        tokens_in=100,
        tokens_out=200,
        cost_usd=0.001,
        latency_ms=500,
        situation_count=1,
        jargon_count=0,
        role_task_count=0,
        environment_count=0,
    )
    assert result_id is not None


def test_update_rating(repo):
    """Should update the rating for a result."""
    run_id = repo.create_run(
        name="Test",
        persona="tech_pm",
        prompt_style="guided",
        flow="enriched",
        constraint_type="2-per-type",
        chip_count=35,
    )
    result_id = repo.create_result(
        run_id=run_id,
        model="claude-haiku",
        chips=[],
        tokens_in=100,
        tokens_out=200,
        cost_usd=0.001,
        latency_ms=500,
        situation_count=0,
        jargon_count=0,
        role_task_count=0,
        environment_count=0,
    )
    repo.update_rating(result_id, 5)
    result = repo.get_result(result_id)
    assert result["rating"] == 5
    assert result["rated_at"] is not None


def test_list_runs(repo):
    """Should list runs grouped by date."""
    repo.create_run(
        name="Run 1",
        persona="tech_pm",
        prompt_style="guided",
        flow="enriched",
        constraint_type="2-per-type",
        chip_count=35,
    )
    repo.create_run(
        name="Run 2",
        persona="tech_swe",
        prompt_style="terse",
        flow="basic",
        constraint_type="none",
        chip_count=15,
    )
    runs = repo.list_runs()
    assert len(runs) == 2


def test_get_results_for_run(repo):
    """Should get all results for a run."""
    run_id = repo.create_run(
        name="Test",
        persona="tech_pm",
        prompt_style="guided",
        flow="enriched",
        constraint_type="2-per-type",
        chip_count=35,
    )
    repo.create_result(
        run_id=run_id,
        model="claude-haiku",
        chips=[],
        tokens_in=100,
        tokens_out=200,
        cost_usd=0.001,
        latency_ms=500,
        situation_count=0,
        jargon_count=0,
        role_task_count=0,
        environment_count=0,
    )
    repo.create_result(
        run_id=run_id,
        model="gpt-5-mini",
        chips=[],
        tokens_in=150,
        tokens_out=250,
        cost_usd=0.002,
        latency_ms=600,
        situation_count=0,
        jargon_count=0,
        role_task_count=0,
        environment_count=0,
    )
    results = repo.get_results_for_run(run_id)
    assert len(results) == 2
