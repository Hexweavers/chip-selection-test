import pytest
from config import ModelConfig, get_model_config, MODELS, CHIP_TYPES


class TestModelConfig:
    def test_calculate_cost_basic(self):
        model = ModelConfig(
            id="test-model",
            name="Test Model",
            input_cost_per_m=1.0,
            output_cost_per_m=5.0,
        )
        cost = model.calculate_cost(input_tokens=1_000_000, output_tokens=0)
        assert cost == 1.0

    def test_calculate_cost_output_only(self):
        model = ModelConfig(
            id="test-model",
            name="Test Model",
            input_cost_per_m=1.0,
            output_cost_per_m=5.0,
        )
        cost = model.calculate_cost(input_tokens=0, output_tokens=1_000_000)
        assert cost == 5.0

    def test_calculate_cost_combined(self):
        model = ModelConfig(
            id="test-model",
            name="Test Model",
            input_cost_per_m=1.0,
            output_cost_per_m=5.0,
        )
        cost = model.calculate_cost(input_tokens=1_000_000, output_tokens=1_000_000)
        assert cost == 6.0

    def test_calculate_cost_fractional(self):
        model = ModelConfig(
            id="test-model",
            name="Test Model",
            input_cost_per_m=1.0,
            output_cost_per_m=5.0,
        )
        cost = model.calculate_cost(input_tokens=500_000, output_tokens=200_000)
        assert cost == pytest.approx(0.5 + 1.0)

    def test_calculate_cost_zero(self):
        model = ModelConfig(
            id="test-model",
            name="Test Model",
            input_cost_per_m=1.0,
            output_cost_per_m=5.0,
        )
        cost = model.calculate_cost(input_tokens=0, output_tokens=0)
        assert cost == 0.0

    def test_calculate_cost_small_tokens(self):
        model = ModelConfig(
            id="test-model",
            name="Test Model",
            input_cost_per_m=1.0,
            output_cost_per_m=5.0,
        )
        cost = model.calculate_cost(input_tokens=1000, output_tokens=500)
        expected = (1000 * 1.0 + 500 * 5.0) / 1_000_000
        assert cost == pytest.approx(expected)


class TestGetModelConfig:
    def test_get_existing_model(self):
        model = get_model_config("anthropic/claude-haiku-4.5")
        assert model is not None
        assert model.name == "Claude Haiku 4.5"

    def test_get_nonexistent_model(self):
        model = get_model_config("nonexistent/model")
        assert model is None

    def test_all_models_have_required_fields(self):
        for model in MODELS:
            assert model.id
            assert model.name
            assert model.input_cost_per_m >= 0
            assert model.output_cost_per_m >= 0


class TestChipTypes:
    def test_all_chip_types_defined(self):
        expected = {"situation", "jargon", "role_task", "environment"}
        assert set(CHIP_TYPES) == expected

    def test_chip_types_count(self):
        assert len(CHIP_TYPES) == 4
