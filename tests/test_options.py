"""Tests for GET /options endpoint."""

import pytest


class TestOptionsEndpoint:
    """Tests for the /options endpoint."""

    def test_returns_200(self, client):
        """Options endpoint returns 200 OK."""
        response = client.get("/options")
        assert response.status_code == 200

    def test_returns_models(self, client):
        """Options includes list of models with id and name."""
        response = client.get("/options")
        data = response.json()

        assert "models" in data
        assert len(data["models"]) > 0

        model = data["models"][0]
        assert "id" in model
        assert "name" in model
        assert isinstance(model["id"], str)
        assert isinstance(model["name"], str)

    def test_returns_personas(self, client):
        """Options includes list of personas with id, sector, and desired_role."""
        response = client.get("/options")
        data = response.json()

        assert "personas" in data
        assert len(data["personas"]) > 0

        persona = data["personas"][0]
        assert "id" in persona
        assert "sector" in persona
        assert "desired_role" in persona

    def test_returns_styles(self, client):
        """Options includes available styles."""
        response = client.get("/options")
        data = response.json()

        assert "styles" in data
        assert "terse" in data["styles"]
        assert "guided" in data["styles"]

    def test_returns_input_types(self, client):
        """Options includes available input types."""
        response = client.get("/options")
        data = response.json()

        assert "input_types" in data
        assert "basic" in data["input_types"]
        assert "enriched" in data["input_types"]

    def test_returns_constraint_types(self, client):
        """Options includes available constraint types."""
        response = client.get("/options")
        data = response.json()

        assert "constraint_types" in data
        assert "with_constraint" in data["constraint_types"]
        assert "no_constraint" in data["constraint_types"]

    def test_returns_chip_counts(self, client):
        """Options includes available chip counts."""
        response = client.get("/options")
        data = response.json()

        assert "chip_counts" in data
        assert 15 in data["chip_counts"]
        assert 35 in data["chip_counts"]

    def test_known_personas_exist(self, client):
        """Known personas from test_personas.json are present."""
        response = client.get("/options")
        data = response.json()

        persona_ids = [p["id"] for p in data["personas"]]
        assert "tech_pm" in persona_ids
        assert "tech_swe" in persona_ids
        assert "healthcare_nurse" in persona_ids

    def test_known_models_exist(self, client):
        """Known models from config.py are present."""
        response = client.get("/options")
        data = response.json()

        model_ids = [m["id"] for m in data["models"]]
        assert "anthropic/claude-haiku-4.5" in model_ids
        assert "google/gemini-2.5-flash-lite" in model_ids
