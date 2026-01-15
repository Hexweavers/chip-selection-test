"""Tests for POST /generate endpoint."""

import pytest
from unittest.mock import patch, MagicMock


class TestGenerateEndpoint:
    """Tests for the /generate endpoint."""

    @pytest.fixture
    def valid_request(self):
        """Valid generate request payload."""
        return {
            "model": "google/gemini-2.5-flash-lite",
            "persona_id": "tech_pm",
            "style": "guided",
            "input_type": "basic",
            "constraint_type": "no_constraint",
            "chip_count": 15,
        }

    def test_returns_400_for_unknown_persona(self, client):
        """Returns 400 for unknown persona_id."""
        response = client.post("/generate", json={
            "model": "google/gemini-2.5-flash-lite",
            "persona_id": "nonexistent_persona",
            "style": "guided",
            "input_type": "basic",
            "constraint_type": "no_constraint",
            "chip_count": 15,
        })
        assert response.status_code == 400
        assert "Unknown persona_id" in response.json()["detail"]

    def test_returns_400_for_unknown_model(self, client):
        """Returns 400 for unknown model."""
        response = client.post("/generate", json={
            "model": "unknown/model",
            "persona_id": "tech_pm",
            "style": "guided",
            "input_type": "basic",
            "constraint_type": "no_constraint",
            "chip_count": 15,
        })
        assert response.status_code == 400
        assert "Unknown model" in response.json()["detail"]

    def test_returns_422_for_invalid_style(self, client):
        """Returns 422 for invalid style."""
        response = client.post("/generate", json={
            "model": "google/gemini-2.5-flash-lite",
            "persona_id": "tech_pm",
            "style": "invalid_style",
            "input_type": "basic",
            "constraint_type": "no_constraint",
            "chip_count": 15,
        })
        assert response.status_code == 422

    def test_returns_422_for_invalid_input_type(self, client):
        """Returns 422 for invalid input_type."""
        response = client.post("/generate", json={
            "model": "google/gemini-2.5-flash-lite",
            "persona_id": "tech_pm",
            "style": "guided",
            "input_type": "invalid_type",
            "constraint_type": "no_constraint",
            "chip_count": 15,
        })
        assert response.status_code == 422

    def test_returns_422_for_invalid_chip_count(self, client):
        """Returns 422 for invalid chip_count."""
        response = client.post("/generate", json={
            "model": "google/gemini-2.5-flash-lite",
            "persona_id": "tech_pm",
            "style": "guided",
            "input_type": "basic",
            "constraint_type": "no_constraint",
            "chip_count": 99,
        })
        assert response.status_code == 422

    def test_basic_flow_returns_chips(self, valid_request, mock_services):
        """Basic flow generates and returns chips."""
        client = mock_services["client"]
        response = client.post("/generate", json=valid_request)

        assert response.status_code == 200
        data = response.json()

        assert "result_id" in data
        assert "final_chips" in data
        assert "cached" in data
        assert data["cached"] is False

        # Basic flow doesn't have step1 or selected chips
        assert data["step1_chips"] is None
        assert data["selected_chips"] is None

        # Verify generator was called with correct params
        mock_services["generator"].generate_step2_basic.assert_called_once()

    def test_enriched_flow_returns_all_steps(self, mock_services):
        """Enriched flow returns step1, selected, and final chips."""
        client = mock_services["client"]
        request = {
            "model": "google/gemini-2.5-flash-lite",
            "persona_id": "tech_pm",
            "style": "guided",
            "input_type": "enriched",
            "constraint_type": "no_constraint",
            "chip_count": 15,
        }
        response = client.post("/generate", json=request)

        assert response.status_code == 200
        data = response.json()

        assert data["step1_chips"] is not None
        assert data["selected_chips"] is not None
        assert data["final_chips"] is not None

        # Verify enriched flow called all steps
        mock_services["generator"].generate_step1.assert_called_once()
        mock_services["selector"].select_chips.assert_called_once()
        mock_services["generator"].generate_step2_enriched.assert_called_once()

    def test_returns_latency_and_cost(self, valid_request, mock_services):
        """Response includes latency and cost metrics."""
        client = mock_services["client"]
        response = client.post("/generate", json=valid_request)

        assert response.status_code == 200
        data = response.json()

        assert "latency_ms" in data
        assert "input_tokens" in data
        assert "output_tokens" in data
        assert "cost_usd" in data

        assert data["latency_ms"] > 0
        assert data["input_tokens"] > 0

    def test_caching_returns_same_result(self, valid_request, mock_services):
        """Second request with same params returns cached result."""
        client = mock_services["client"]
        # First request
        response1 = client.post("/generate", json=valid_request)
        assert response1.status_code == 200
        data1 = response1.json()
        assert data1["cached"] is False

        # Second request with same params
        response2 = client.post("/generate", json=valid_request)
        assert response2.status_code == 200
        data2 = response2.json()

        assert data2["cached"] is True
        assert data2["result_id"] == data1["result_id"]

    def test_different_params_not_cached(self, mock_services):
        """Different params generate new result, not cached."""
        client = mock_services["client"]
        request1 = {
            "model": "google/gemini-2.5-flash-lite",
            "persona_id": "tech_pm",
            "style": "guided",
            "input_type": "basic",
            "constraint_type": "no_constraint",
            "chip_count": 15,
        }
        request2 = {
            "model": "google/gemini-2.5-flash-lite",
            "persona_id": "tech_swe",  # Different persona
            "style": "guided",
            "input_type": "basic",
            "constraint_type": "no_constraint",
            "chip_count": 15,
        }

        response1 = client.post("/generate", json=request1)
        response2 = client.post("/generate", json=request2)

        assert response1.status_code == 200
        assert response2.status_code == 200

        data1 = response1.json()
        data2 = response2.json()

        assert data1["cached"] is False
        assert data2["cached"] is False
        assert data1["result_id"] != data2["result_id"]

    def test_fill_service_called_when_types_missing(self, mock_services, sample_chips, mock_llm_response):
        """Fill service is called when chip types are missing."""
        client = mock_services["client"]
        # Configure fill service to report missing types
        mock_services["fill_service"].get_missing_types.return_value = ["environment"]
        mock_services["fill_service"].fill_missing.return_value = (
            [sample_chips[3]],  # environment chip
            mock_llm_response(),
        )

        request = {
            "model": "google/gemini-2.5-flash-lite",
            "persona_id": "finance_analyst",  # Use different persona to avoid cache
            "style": "guided",
            "input_type": "basic",
            "constraint_type": "no_constraint",
            "chip_count": 15,
        }

        response = client.post("/generate", json=request)
        assert response.status_code == 200

        mock_services["fill_service"].get_missing_types.assert_called()
        mock_services["fill_service"].fill_missing.assert_called_once()

    def test_errors_collected_from_all_steps(self, mock_services, mock_llm_response):
        """Errors from each step are collected and returned."""
        client = mock_services["client"]
        # Make step2 return an error
        error_response = mock_llm_response(error="LLM parsing failed")
        mock_services["generator"].generate_step2_basic.return_value = ([], error_response)

        request = {
            "model": "google/gemini-2.5-flash-lite",
            "persona_id": "retail_manager",  # Different persona to avoid cache
            "style": "terse",
            "input_type": "basic",
            "constraint_type": "no_constraint",
            "chip_count": 15,
        }

        response = client.post("/generate", json=request)
        assert response.status_code == 200

        data = response.json()
        assert data["errors"] is not None
        assert any("Step 2" in e for e in data["errors"])


class TestGenerateRequestValidation:
    """Tests for request validation."""

    def test_requires_all_fields(self, client):
        """Request requires all fields."""
        response = client.post("/generate", json={})
        assert response.status_code == 422

    def test_requires_model(self, client):
        """Request requires model field."""
        response = client.post("/generate", json={
            "persona_id": "tech_pm",
            "style": "guided",
            "input_type": "basic",
            "constraint_type": "no_constraint",
            "chip_count": 15,
        })
        assert response.status_code == 422

    def test_requires_persona_id(self, client):
        """Request requires persona_id field."""
        response = client.post("/generate", json={
            "model": "google/gemini-2.5-flash-lite",
            "style": "guided",
            "input_type": "basic",
            "constraint_type": "no_constraint",
            "chip_count": 15,
        })
        assert response.status_code == 422
