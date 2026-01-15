import pytest
from unittest.mock import MagicMock, patch

from models.chip import Chip
from services.generator import ChipGenerator
from services.llm import LLMResponse


@pytest.fixture
def mock_llm():
    return MagicMock()


@pytest.fixture
def mock_prompts():
    return {
        "terse": {
            "step1_user_selectable": {
                "system": "System for step1",
                "user": "Generate chips for {sector} {desired_role}",
            },
            "step1_user_selectable_with_constraint": {
                "system": "System for step1 constrained",
                "user": "Generate constrained chips for {sector} {desired_role}",
            },
            "step2_final_generation_basic": {
                "system": "System for step2 basic",
                "user": "Generate {chip_count} chips for {sector} {desired_role}",
            },
            "step2_final_generation": {
                "system": "System for step2 enriched",
                "user": "Generate {chip_count} chips for {sector} {desired_role} using {user_selected_chips}",
            },
        },
        "guided": {
            "step1_user_selectable": {
                "system": "Guided system",
                "user": "Guided user prompt {sector} {desired_role}",
            },
            "step1_user_selectable_with_constraint": {
                "system": "Guided constrained system",
                "user": "Guided constrained {sector} {desired_role}",
            },
            "step2_final_generation_basic": {
                "system": "Guided step2 basic",
                "user": "Guided {chip_count} chips {sector} {desired_role}",
            },
            "step2_final_generation": {
                "system": "Guided step2 enriched",
                "user": "Guided {chip_count} chips {sector} {desired_role} with {user_selected_chips}",
            },
        },
    }


@pytest.fixture
def generator(mock_llm, mock_prompts):
    with patch.object(ChipGenerator, '_load_prompts', return_value=mock_prompts):
        return ChipGenerator(mock_llm)


class TestFormatPrompt:
    def test_replaces_single_placeholder(self, generator):
        result = generator._format_prompt("Hello {name}", name="World")
        assert result == "Hello World"

    def test_replaces_multiple_placeholders(self, generator):
        result = generator._format_prompt(
            "{greeting} {name}!",
            greeting="Hi",
            name="Alice",
        )
        assert result == "Hi Alice!"

    def test_ignores_missing_kwargs(self, generator):
        result = generator._format_prompt("Hello {name}", other="value")
        assert result == "Hello {name}"

    def test_converts_non_strings(self, generator):
        result = generator._format_prompt("Count: {count}", count=42)
        assert result == "Count: 42"


class TestGenerateStep1:
    def test_uses_correct_prompt_without_constraint(self, generator, mock_llm):
        mock_llm.chat.return_value = LLMResponse(
            content='[{"key": "c1", "display": "Chip 1", "type": "situation"}]',
            input_tokens=100,
            output_tokens=50,
            latency_ms=500,
        )

        chips, response = generator.generate_step1(
            model="test-model",
            sector="Tech",
            desired_role="Dev",
            style="terse",
            constraint="no_constraint",
        )

        mock_llm.chat.assert_called_once()
        call_args = mock_llm.chat.call_args
        assert call_args[0][1] == "System for step1"
        assert "Tech" in call_args[0][2]
        assert "Dev" in call_args[0][2]

    def test_uses_constrained_prompt(self, generator, mock_llm):
        mock_llm.chat.return_value = LLMResponse(
            content='[]',
            input_tokens=100,
            output_tokens=50,
            latency_ms=500,
        )

        generator.generate_step1(
            model="test-model",
            sector="Tech",
            desired_role="Dev",
            style="terse",
            constraint="with_constraint",
        )

        call_args = mock_llm.chat.call_args
        assert call_args[0][1] == "System for step1 constrained"

    def test_parses_valid_chips(self, generator, mock_llm):
        mock_llm.chat.return_value = LLMResponse(
            content='[{"key": "c1", "display": "Chip 1", "type": "situation"}]',
            input_tokens=100,
            output_tokens=50,
            latency_ms=500,
        )

        chips, response = generator.generate_step1(
            model="test-model",
            sector="Tech",
            desired_role="Dev",
            style="terse",
            constraint="no_constraint",
        )

        assert len(chips) == 1
        assert chips[0].key == "c1"

    def test_returns_empty_on_llm_error(self, generator, mock_llm):
        mock_llm.chat.return_value = LLMResponse(
            content="",
            input_tokens=0,
            output_tokens=0,
            latency_ms=100,
            error="API Error",
        )

        chips, response = generator.generate_step1(
            model="test-model",
            sector="Tech",
            desired_role="Dev",
            style="terse",
            constraint="no_constraint",
        )

        assert chips == []
        assert response.error == "API Error"

    def test_sets_error_on_parse_failure(self, generator, mock_llm):
        mock_llm.chat.return_value = LLMResponse(
            content='[{"key": "", "display": "Bad", "type": "situation"}]',
            input_tokens=100,
            output_tokens=50,
            latency_ms=500,
        )

        chips, response = generator.generate_step1(
            model="test-model",
            sector="Tech",
            desired_role="Dev",
            style="terse",
            constraint="no_constraint",
        )

        assert chips == []
        assert response.error is not None


class TestGenerateStep2Basic:
    def test_uses_correct_prompt(self, generator, mock_llm):
        mock_llm.chat.return_value = LLMResponse(
            content='[]',
            input_tokens=100,
            output_tokens=50,
            latency_ms=500,
        )

        generator.generate_step2_basic(
            model="test-model",
            sector="Finance",
            desired_role="Analyst",
            style="guided",
            chip_count=15,
        )

        call_args = mock_llm.chat.call_args
        assert call_args[0][1] == "Guided step2 basic"
        assert "15" in call_args[0][2]
        assert "Finance" in call_args[0][2]

    def test_parses_chips(self, generator, mock_llm):
        mock_llm.chat.return_value = LLMResponse(
            content='[{"key": "k1", "display": "K1", "type": "jargon"}]',
            input_tokens=100,
            output_tokens=50,
            latency_ms=500,
        )

        chips, response = generator.generate_step2_basic(
            model="test-model",
            sector="Tech",
            desired_role="Dev",
            style="terse",
            chip_count=15,
        )

        assert len(chips) == 1
        assert chips[0].type == "jargon"


class TestGenerateStep2Enriched:
    def test_includes_selected_chips_in_prompt(self, generator, mock_llm):
        mock_llm.chat.return_value = LLMResponse(
            content='[]',
            input_tokens=100,
            output_tokens=50,
            latency_ms=500,
        )

        selected = [
            Chip(key="sel1", display="Selected 1", type="situation"),
            Chip(key="sel2", display="Selected 2", type="jargon"),
        ]

        generator.generate_step2_enriched(
            model="test-model",
            sector="Tech",
            desired_role="Dev",
            style="terse",
            chip_count=35,
            user_selected_chips=selected,
        )

        call_args = mock_llm.chat.call_args
        user_prompt = call_args[0][2]
        assert "sel1" in user_prompt
        assert "sel2" in user_prompt
        assert "35" in user_prompt

    def test_parses_enriched_chips(self, generator, mock_llm):
        mock_llm.chat.return_value = LLMResponse(
            content='[{"key": "new1", "display": "New 1", "type": "role_task"}]',
            input_tokens=100,
            output_tokens=50,
            latency_ms=500,
        )

        chips, response = generator.generate_step2_enriched(
            model="test-model",
            sector="Tech",
            desired_role="Dev",
            style="terse",
            chip_count=15,
            user_selected_chips=[],
        )

        assert len(chips) == 1
        assert chips[0].key == "new1"
