"""Tests for GET /results endpoint with extended filters."""

import pytest


class TestResultsEndpoint:
    """Tests for the /results endpoint."""

    def test_returns_200(self, client):
        """Results endpoint returns 200 OK."""
        response = client.get("/results")
        assert response.status_code == 200

    def test_returns_results_and_total(self, client):
        """Response includes results array and total count."""
        response = client.get("/results")
        data = response.json()

        assert "results" in data
        assert "total" in data
        assert isinstance(data["results"], list)
        assert isinstance(data["total"], int)

    def test_filter_by_style(self, client):
        """Can filter results by style."""
        response = client.get("/results?style=guided")
        assert response.status_code == 200

        response = client.get("/results?style=terse")
        assert response.status_code == 200

    def test_filter_by_input_type(self, client):
        """Can filter results by input_type."""
        response = client.get("/results?input_type=basic")
        assert response.status_code == 200

        response = client.get("/results?input_type=enriched")
        assert response.status_code == 200

    def test_filter_by_constraint_type(self, client):
        """Can filter results by constraint_type."""
        response = client.get("/results?constraint_type=no_constraint")
        assert response.status_code == 200

        response = client.get("/results?constraint_type=with_constraint")
        assert response.status_code == 200

    def test_filter_by_chip_count(self, client):
        """Can filter results by chip_count."""
        response = client.get("/results?chip_count=15")
        assert response.status_code == 200

        response = client.get("/results?chip_count=35")
        assert response.status_code == 200

    def test_filter_by_model(self, client):
        """Can filter results by model."""
        response = client.get("/results?model=google/gemini-2.5-flash-lite")
        assert response.status_code == 200

    def test_filter_by_persona_id(self, client):
        """Can filter results by persona_id."""
        response = client.get("/results?persona_id=tech_pm")
        assert response.status_code == 200

    def test_multiple_filters_combined(self, client):
        """Can combine multiple filters."""
        response = client.get(
            "/results?model=google/gemini-2.5-flash-lite"
            "&style=guided"
            "&input_type=basic"
            "&chip_count=15"
        )
        assert response.status_code == 200

    def test_pagination_limit(self, client):
        """Can limit number of results."""
        response = client.get("/results?limit=5")
        assert response.status_code == 200

        data = response.json()
        assert len(data["results"]) <= 5

    def test_pagination_offset(self, client):
        """Can offset results for pagination."""
        response = client.get("/results?offset=10")
        assert response.status_code == 200


class TestResultsWithGeneratedData:
    """Tests that verify filters work with generated data."""

    @pytest.fixture
    def generate_test_result(self, client, mock_services):
        """Generate a result for testing filters."""
        request = {
            "model": "google/gemini-2.5-flash-lite",
            "persona_id": "creative_ux",
            "style": "terse",
            "input_type": "basic",
            "constraint_type": "with_constraint",
            "chip_count": 35,
        }
        response = client.post("/generate", json=request)
        assert response.status_code == 200
        return response.json()

    def test_generated_result_appears_in_list(self, client, generate_test_result):
        """Generated result appears in results list."""
        result_id = generate_test_result["result_id"]

        response = client.get("/results")
        data = response.json()

        result_ids = [r["id"] for r in data["results"]]
        assert result_id in result_ids

    def test_filter_finds_generated_result(self, client, generate_test_result):
        """Filters correctly find the generated result."""
        result_id = generate_test_result["result_id"]

        # Filter by the exact params used to generate
        response = client.get(
            "/results?persona_id=creative_ux"
            "&style=terse"
            "&input_type=basic"
            "&constraint_type=with_constraint"
            "&chip_count=35"
        )
        data = response.json()

        result_ids = [r["id"] for r in data["results"]]
        assert result_id in result_ids

    def test_wrong_filter_excludes_result(self, client, generate_test_result):
        """Wrong filter values exclude the result."""
        result_id = generate_test_result["result_id"]

        # Filter by different style (generated was "terse")
        response = client.get("/results?style=guided&persona_id=creative_ux")
        data = response.json()

        result_ids = [r["id"] for r in data["results"]]
        assert result_id not in result_ids


class TestResultDetail:
    """Tests for GET /results/{result_id} endpoint."""

    def test_returns_404_for_unknown_id(self, client):
        """Returns 404 for unknown result_id."""
        response = client.get("/results/nonexistent-uuid")
        assert response.status_code == 404

    def test_returns_full_result(self, client, mock_services):
        """Returns full result details including chips."""
        # First generate a result
        request = {
            "model": "google/gemini-2.5-flash-lite",
            "persona_id": "healthcare_nurse",
            "style": "guided",
            "input_type": "basic",
            "constraint_type": "no_constraint",
            "chip_count": 15,
        }
        gen_response = client.post("/generate", json=request)
        result_id = gen_response.json()["result_id"]

        # Then fetch it
        response = client.get(f"/results/{result_id}")
        assert response.status_code == 200

        data = response.json()
        assert data["id"] == result_id
        assert "final_chips" in data
        assert "model" in data
        assert "persona_id" in data
