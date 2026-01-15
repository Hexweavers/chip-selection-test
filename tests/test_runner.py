import pytest
from unittest.mock import MagicMock

from models.chip import Chip
from runner import run_test
from services.llm import LLMResponse


@pytest.fixture
def mock_llm_response():
    def _create(error=None):
        return LLMResponse(
            content="[]",
            input_tokens=100,
            output_tokens=50,
            latency_ms=500,
            error=error,
        )
    return _create


@pytest.fixture
def sample_chips():
    return [
        Chip(key="sprint_planning", display="Sprint Planning", type="situation"),
        Chip(key="okr", display="OKR", type="jargon"),
        Chip(key="roadmap_review", display="Roadmap Review", type="role_task"),
        Chip(key="remote_team", display="Remote Team", type="environment"),
        Chip(key="standup", display="Daily Standup", type="situation"),
    ]


@pytest.fixture
def persona():
    return {
        "id": "tech_pm",
        "sector": "Technology",
        "desired_role": "Product Manager",
        "persona": "A product manager at a tech company",
    }


@pytest.fixture
def mock_generator(sample_chips, mock_llm_response):
    generator = MagicMock()
    generator.generate_step1.return_value = (sample_chips[:4], mock_llm_response())
    generator.generate_step2_basic.return_value = (sample_chips, mock_llm_response())
    generator.generate_step2_enriched.return_value = (sample_chips[2:], mock_llm_response())
    return generator


@pytest.fixture
def mock_selector(sample_chips, mock_llm_response):
    selector = MagicMock()
    selector.select_chips.return_value = (sample_chips[:3], mock_llm_response())
    return selector


@pytest.fixture
def mock_fill_service(mock_llm_response):
    fill_service = MagicMock()
    fill_service.get_missing_types.return_value = []
    fill_service.fill_missing.return_value = ([], mock_llm_response())
    return fill_service


class TestRunTestBasicFlow:
    def test_basic_flow_calls_generate_step2_basic(
        self, persona, mock_generator, mock_selector, mock_fill_service
    ):
        result = run_test(
            model="test-model",
            persona=persona,
            style="terse",
            constraint="no_constraint",
            input_type="basic",
            chip_count=15,
            generator=mock_generator,
            selector=mock_selector,
            fill_service=mock_fill_service,
        )

        mock_generator.generate_step2_basic.assert_called_once()
        mock_generator.generate_step1.assert_not_called()
        mock_selector.select_chips.assert_not_called()

    def test_basic_flow_returns_chips(
        self, persona, mock_generator, mock_selector, mock_fill_service
    ):
        result = run_test(
            model="test-model",
            persona=persona,
            style="terse",
            constraint="no_constraint",
            input_type="basic",
            chip_count=15,
            generator=mock_generator,
            selector=mock_selector,
            fill_service=mock_fill_service,
        )

        assert len(result.final_chips) > 0
        assert result.step1_chips == []
        assert result.user_selected_chips == []

    def test_basic_flow_accumulates_tokens(
        self, persona, mock_generator, mock_selector, mock_fill_service
    ):
        result = run_test(
            model="test-model",
            persona=persona,
            style="terse",
            constraint="no_constraint",
            input_type="basic",
            chip_count=15,
            generator=mock_generator,
            selector=mock_selector,
            fill_service=mock_fill_service,
        )

        assert result.input_tokens == 100
        assert result.output_tokens == 50
        assert result.latency_ms == 500


class TestRunTestEnrichedFlow:
    def test_enriched_flow_calls_all_steps(
        self, persona, mock_generator, mock_selector, mock_fill_service
    ):
        result = run_test(
            model="test-model",
            persona=persona,
            style="guided",
            constraint="with_constraint",
            input_type="enriched",
            chip_count=35,
            generator=mock_generator,
            selector=mock_selector,
            fill_service=mock_fill_service,
        )

        mock_generator.generate_step1.assert_called_once()
        mock_selector.select_chips.assert_called_once()
        mock_generator.generate_step2_enriched.assert_called_once()

    def test_enriched_flow_populates_all_chip_lists(
        self, persona, mock_generator, mock_selector, mock_fill_service
    ):
        result = run_test(
            model="test-model",
            persona=persona,
            style="guided",
            constraint="with_constraint",
            input_type="enriched",
            chip_count=35,
            generator=mock_generator,
            selector=mock_selector,
            fill_service=mock_fill_service,
        )

        assert len(result.step1_chips) > 0
        assert len(result.user_selected_chips) > 0
        assert len(result.step2_chips) > 0
        assert len(result.final_chips) > 0

    def test_enriched_flow_accumulates_all_tokens(
        self, persona, mock_generator, mock_selector, mock_fill_service
    ):
        result = run_test(
            model="test-model",
            persona=persona,
            style="guided",
            constraint="with_constraint",
            input_type="enriched",
            chip_count=35,
            generator=mock_generator,
            selector=mock_selector,
            fill_service=mock_fill_service,
        )

        # 3 LLM calls: step1 + selection + step2_enriched
        assert result.input_tokens == 300
        assert result.output_tokens == 150
        assert result.latency_ms == 1500


class TestChipMerging:
    def test_merges_selected_and_step2_chips(
        self, persona, mock_generator, mock_selector, mock_fill_service, mock_llm_response
    ):
        selected = [
            Chip(key="sel1", display="Selected 1", type="situation"),
            Chip(key="sel2", display="Selected 2", type="jargon"),
        ]
        step2 = [
            Chip(key="gen1", display="Generated 1", type="role_task"),
            Chip(key="gen2", display="Generated 2", type="environment"),
        ]

        mock_selector.select_chips.return_value = (selected, mock_llm_response())
        mock_generator.generate_step2_enriched.return_value = (step2, mock_llm_response())

        result = run_test(
            model="test-model",
            persona=persona,
            style="guided",
            constraint="with_constraint",
            input_type="enriched",
            chip_count=15,
            generator=mock_generator,
            selector=mock_selector,
            fill_service=mock_fill_service,
        )

        final_keys = {c.key for c in result.final_chips}
        assert "sel1" in final_keys
        assert "sel2" in final_keys
        assert "gen1" in final_keys
        assert "gen2" in final_keys

    def test_deduplicates_by_key(
        self, persona, mock_generator, mock_selector, mock_fill_service, mock_llm_response
    ):
        selected = [
            Chip(key="duplicate", display="Selected Version", type="situation"),
        ]
        step2 = [
            Chip(key="duplicate", display="Generated Version", type="situation"),
            Chip(key="unique", display="Unique", type="jargon"),
        ]

        mock_selector.select_chips.return_value = (selected, mock_llm_response())
        mock_generator.generate_step2_enriched.return_value = (step2, mock_llm_response())

        result = run_test(
            model="test-model",
            persona=persona,
            style="guided",
            constraint="with_constraint",
            input_type="enriched",
            chip_count=15,
            generator=mock_generator,
            selector=mock_selector,
            fill_service=mock_fill_service,
        )

        keys = [c.key for c in result.final_chips]
        assert keys.count("duplicate") == 1
        # First one wins (from selected)
        dup_chip = next(c for c in result.final_chips if c.key == "duplicate")
        assert dup_chip.display == "Selected Version"


class TestFillLogic:
    def test_fill_service_called_when_types_missing(
        self, persona, mock_generator, mock_selector, mock_fill_service, mock_llm_response
    ):
        mock_fill_service.get_missing_types.return_value = ["environment", "role_task"]
        fill_chips = [
            Chip(key="fill1", display="Fill 1", type="environment"),
            Chip(key="fill2", display="Fill 2", type="role_task"),
        ]
        mock_fill_service.fill_missing.return_value = (fill_chips, mock_llm_response())

        result = run_test(
            model="test-model",
            persona=persona,
            style="terse",
            constraint="no_constraint",
            input_type="basic",
            chip_count=15,
            generator=mock_generator,
            selector=mock_selector,
            fill_service=mock_fill_service,
        )

        mock_fill_service.get_missing_types.assert_called_once()
        mock_fill_service.fill_missing.assert_called_once()
        assert len(result.fill_chips) == 2

    def test_fill_chips_added_to_final(
        self, persona, mock_generator, mock_selector, mock_fill_service, mock_llm_response
    ):
        mock_fill_service.get_missing_types.return_value = ["environment"]
        fill_chips = [Chip(key="filled_env", display="Filled Env", type="environment")]
        mock_fill_service.fill_missing.return_value = (fill_chips, mock_llm_response())

        result = run_test(
            model="test-model",
            persona=persona,
            style="terse",
            constraint="no_constraint",
            input_type="basic",
            chip_count=15,
            generator=mock_generator,
            selector=mock_selector,
            fill_service=mock_fill_service,
        )

        final_keys = {c.key for c in result.final_chips}
        assert "filled_env" in final_keys

    def test_fill_not_called_when_all_types_covered(
        self, persona, mock_generator, mock_selector, mock_fill_service
    ):
        mock_fill_service.get_missing_types.return_value = []

        result = run_test(
            model="test-model",
            persona=persona,
            style="terse",
            constraint="no_constraint",
            input_type="basic",
            chip_count=15,
            generator=mock_generator,
            selector=mock_selector,
            fill_service=mock_fill_service,
        )

        mock_fill_service.fill_missing.assert_not_called()


class TestErrorHandling:
    def test_step1_error_collected(
        self, persona, mock_generator, mock_selector, mock_fill_service
    ):
        mock_generator.generate_step1.return_value = (
            [],
            LLMResponse(content="", input_tokens=0, output_tokens=0, latency_ms=0, error="Step 1 failed"),
        )

        result = run_test(
            model="test-model",
            persona=persona,
            style="guided",
            constraint="with_constraint",
            input_type="enriched",
            chip_count=15,
            generator=mock_generator,
            selector=mock_selector,
            fill_service=mock_fill_service,
        )

        assert any("Step 1" in e for e in result.errors)

    def test_selection_error_collected(
        self, persona, mock_generator, mock_selector, mock_fill_service, sample_chips, mock_llm_response
    ):
        mock_generator.generate_step1.return_value = (sample_chips[:4], mock_llm_response())
        mock_selector.select_chips.return_value = (
            [],
            LLMResponse(content="", input_tokens=0, output_tokens=0, latency_ms=0, error="Selection failed"),
        )

        result = run_test(
            model="test-model",
            persona=persona,
            style="guided",
            constraint="with_constraint",
            input_type="enriched",
            chip_count=15,
            generator=mock_generator,
            selector=mock_selector,
            fill_service=mock_fill_service,
        )

        assert any("Selection" in e for e in result.errors)

    def test_step2_error_collected(
        self, persona, mock_generator, mock_selector, mock_fill_service
    ):
        mock_generator.generate_step2_basic.return_value = (
            [],
            LLMResponse(content="", input_tokens=0, output_tokens=0, latency_ms=0, error="Step 2 failed"),
        )

        result = run_test(
            model="test-model",
            persona=persona,
            style="terse",
            constraint="no_constraint",
            input_type="basic",
            chip_count=15,
            generator=mock_generator,
            selector=mock_selector,
            fill_service=mock_fill_service,
        )

        assert any("Step 2" in e for e in result.errors)

    def test_fill_error_collected(
        self, persona, mock_generator, mock_selector, mock_fill_service
    ):
        mock_fill_service.get_missing_types.return_value = ["environment"]
        mock_fill_service.fill_missing.return_value = (
            [],
            LLMResponse(content="", input_tokens=0, output_tokens=0, latency_ms=0, error="Fill failed"),
        )

        result = run_test(
            model="test-model",
            persona=persona,
            style="terse",
            constraint="no_constraint",
            input_type="basic",
            chip_count=15,
            generator=mock_generator,
            selector=mock_selector,
            fill_service=mock_fill_service,
        )

        assert any("Fill" in e for e in result.errors)


class TestCostCalculation:
    def test_cost_calculated_for_known_model(
        self, persona, mock_generator, mock_selector, mock_fill_service
    ):
        result = run_test(
            model="anthropic/claude-haiku-4.5",
            persona=persona,
            style="terse",
            constraint="no_constraint",
            input_type="basic",
            chip_count=15,
            generator=mock_generator,
            selector=mock_selector,
            fill_service=mock_fill_service,
        )

        assert result.cost_usd is not None
        assert result.cost_usd > 0

    def test_cost_none_for_unknown_model(
        self, persona, mock_generator, mock_selector, mock_fill_service
    ):
        result = run_test(
            model="unknown/model",
            persona=persona,
            style="terse",
            constraint="no_constraint",
            input_type="basic",
            chip_count=15,
            generator=mock_generator,
            selector=mock_selector,
            fill_service=mock_fill_service,
        )

        assert result.cost_usd is None


class TestDryRun:
    def test_dry_run_returns_empty_result(
        self, persona, mock_generator, mock_selector, mock_fill_service
    ):
        result = run_test(
            model="test-model",
            persona=persona,
            style="terse",
            constraint="no_constraint",
            input_type="basic",
            chip_count=15,
            generator=mock_generator,
            selector=mock_selector,
            fill_service=mock_fill_service,
            dry_run=True,
        )

        assert result.final_chips == []
        assert result.latency_ms == 0

    def test_dry_run_no_llm_calls(
        self, persona, mock_generator, mock_selector, mock_fill_service
    ):
        run_test(
            model="test-model",
            persona=persona,
            style="terse",
            constraint="no_constraint",
            input_type="basic",
            chip_count=15,
            generator=mock_generator,
            selector=mock_selector,
            fill_service=mock_fill_service,
            dry_run=True,
        )

        mock_generator.generate_step1.assert_not_called()
        mock_generator.generate_step2_basic.assert_not_called()
        mock_selector.select_chips.assert_not_called()


class TestMetadata:
    def test_metadata_populated(
        self, persona, mock_generator, mock_selector, mock_fill_service
    ):
        result = run_test(
            model="test-model",
            persona=persona,
            style="guided",
            constraint="with_constraint",
            input_type="enriched",
            chip_count=35,
            generator=mock_generator,
            selector=mock_selector,
            fill_service=mock_fill_service,
        )

        assert result.metadata.model == "test-model"
        assert result.metadata.persona_id == "tech_pm"
        assert result.metadata.sector == "Technology"
        assert result.metadata.desired_role == "Product Manager"
        assert result.metadata.style == "guided"
        assert result.metadata.constraint == "with_constraint"
        assert result.metadata.input_type == "enriched"
        assert result.metadata.chip_count == 35
