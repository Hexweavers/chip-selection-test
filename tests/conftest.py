"""Pytest fixtures for API tests."""

import sqlite3
from pathlib import Path

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from api.main import app
from db.repository import Repository
from models.chip import Chip
from services.llm import LLMResponse


@pytest.fixture
def test_db():
    """In-memory SQLite database for testing."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    schema_path = Path(__file__).parent.parent / "db" / "schema.sql"
    conn.executescript(schema_path.read_text())
    conn.commit()
    return conn


@pytest.fixture
def test_repo(test_db):
    """Repository with in-memory test database."""
    return Repository(db=test_db)


@pytest.fixture
def client(test_repo):
    """FastAPI test client with mocked database."""
    with patch("api.routes.generate.get_repo", return_value=test_repo), \
         patch("api.routes.results.get_repo", return_value=test_repo), \
         patch("api.routes.runs.get_repo", return_value=test_repo), \
         patch("api.routes.ratings.get_repo", return_value=test_repo), \
         patch("api.routes.stats.get_repo", return_value=test_repo):
        yield TestClient(app)


@pytest.fixture
def mock_llm_response():
    """Factory for creating mock LLM responses."""
    def _create(content: str = "[]", error: str | None = None):
        return LLMResponse(
            content=content,
            input_tokens=100,
            output_tokens=50,
            latency_ms=500,
            error=error,
        )
    return _create


@pytest.fixture
def sample_chips():
    """Sample chips for testing."""
    return [
        Chip(key="sprint_planning", display="Sprint Planning", type="situation"),
        Chip(key="okr", display="OKR", type="jargon"),
        Chip(key="roadmap_review", display="Roadmap Review", type="role_task"),
        Chip(key="remote_team", display="Remote Team", type="environment"),
        Chip(key="stakeholder_meeting", display="Stakeholder Meeting", type="situation"),
    ]


@pytest.fixture
def mock_generator(sample_chips, mock_llm_response):
    """Mock ChipGenerator that returns sample chips."""
    generator = MagicMock()
    generator.generate_step1.return_value = (sample_chips[:4], mock_llm_response())
    generator.generate_step2_basic.return_value = (sample_chips, mock_llm_response())
    generator.generate_step2_enriched.return_value = (sample_chips[2:], mock_llm_response())
    return generator


@pytest.fixture
def mock_selector(sample_chips, mock_llm_response):
    """Mock ChipSelector that returns selected chips."""
    selector = MagicMock()
    selector.select_chips.return_value = (sample_chips[:3], mock_llm_response())
    return selector


@pytest.fixture
def mock_fill_service(mock_llm_response):
    """Mock FillService."""
    fill_service = MagicMock()
    fill_service.get_missing_types.return_value = []
    fill_service.fill_missing.return_value = ([], mock_llm_response())
    return fill_service


@pytest.fixture
def mock_services(test_repo, mock_generator, mock_selector, mock_fill_service):
    """Patch all services and database with mocks. Use this instead of client for generate tests."""
    with patch("api.routes.generate.get_repo", return_value=test_repo), \
         patch("api.routes.results.get_repo", return_value=test_repo), \
         patch("api.routes.runs.get_repo", return_value=test_repo), \
         patch("api.routes.ratings.get_repo", return_value=test_repo), \
         patch("api.routes.stats.get_repo", return_value=test_repo), \
         patch("api.routes.generate.get_generator", return_value=mock_generator), \
         patch("api.routes.generate.get_selector", return_value=mock_selector), \
         patch("api.routes.generate.get_fill_service", return_value=mock_fill_service):
        yield {
            "generator": mock_generator,
            "selector": mock_selector,
            "fill_service": mock_fill_service,
            "client": TestClient(app),
        }


@pytest.fixture
def reset_singletons():
    """Reset service singletons between tests."""
    import api.services as services

    # Store original values
    original = {
        "_llm_client": services._llm_client,
        "_generator": services._generator,
        "_selector": services._selector,
        "_fill_service": services._fill_service,
        "_personas": services._personas,
    }

    # Reset to None
    services._llm_client = None
    services._generator = None
    services._selector = None
    services._fill_service = None
    services._personas = None

    yield

    # Restore original values
    services._llm_client = original["_llm_client"]
    services._generator = original["_generator"]
    services._selector = original["_selector"]
    services._fill_service = original["_fill_service"]
    services._personas = original["_personas"]
