import pytest
from unittest.mock import MagicMock, patch

from models.chip import Chip
from services.selector import ChipSelector
from services.llm import LLMResponse


@pytest.fixture
def mock_llm():
    return MagicMock()


@pytest.fixture
def mock_prompts():
    return {
        "terse": {
            "chip_selector": {
                "system": "You are simulating a user",
                "user": "As {persona}, select from: {available_chips}",
            },
        },
        "guided": {
            "chip_selector": {
                "system": "Guided selector",
                "user": "Guided {persona} selects from {available_chips}",
            },
        },
    }


@pytest.fixture
def selector(mock_llm, mock_prompts):
    with patch.object(ChipSelector, '_load_prompts', return_value=mock_prompts):
        return ChipSelector(mock_llm)


@pytest.fixture
def available_chips():
    return [
        Chip(key="sprint_planning", display="Sprint Planning", type="situation"),
        Chip(key="okr", display="OKR", type="jargon"),
        Chip(key="roadmap_review", display="Roadmap Review", type="role_task"),
        Chip(key="remote_team", display="Remote Team", type="environment"),
    ]


class TestSelectChips:
    def test_parses_selected_keys(self, selector, mock_llm, available_chips):
        mock_llm.chat.return_value = LLMResponse(
            content='["sprint_planning", "okr"]',
            input_tokens=100,
            output_tokens=50,
            latency_ms=500,
        )

        selected, response = selector.select_chips(
            available_chips=available_chips,
            persona="A tech PM",
            style="terse",
        )

        assert len(selected) == 2
        assert selected[0].key == "sprint_planning"
        assert selected[1].key == "okr"

    def test_handles_markdown_code_block(self, selector, mock_llm, available_chips):
        mock_llm.chat.return_value = LLMResponse(
            content='```json\n["roadmap_review"]\n```',
            input_tokens=100,
            output_tokens=50,
            latency_ms=500,
        )

        selected, response = selector.select_chips(
            available_chips=available_chips,
            persona="A tech PM",
            style="terse",
        )

        assert len(selected) == 1
        assert selected[0].key == "roadmap_review"

    def test_case_insensitive_fallback(self, selector, mock_llm, available_chips):
        mock_llm.chat.return_value = LLMResponse(
            content='["SPRINT_PLANNING", "OKR"]',
            input_tokens=100,
            output_tokens=50,
            latency_ms=500,
        )

        selected, response = selector.select_chips(
            available_chips=available_chips,
            persona="A tech PM",
            style="terse",
        )

        assert len(selected) == 2

    def test_ignores_unknown_keys(self, selector, mock_llm, available_chips):
        mock_llm.chat.return_value = LLMResponse(
            content='["sprint_planning", "nonexistent_key"]',
            input_tokens=100,
            output_tokens=50,
            latency_ms=500,
        )

        selected, response = selector.select_chips(
            available_chips=available_chips,
            persona="A tech PM",
            style="terse",
        )

        assert len(selected) == 1
        assert selected[0].key == "sprint_planning"

    def test_returns_empty_on_llm_error(self, selector, mock_llm, available_chips):
        mock_llm.chat.return_value = LLMResponse(
            content="",
            input_tokens=0,
            output_tokens=0,
            latency_ms=100,
            error="API Error",
        )

        selected, response = selector.select_chips(
            available_chips=available_chips,
            persona="A tech PM",
            style="terse",
        )

        assert selected == []
        assert response.error == "API Error"

    def test_error_on_invalid_json(self, selector, mock_llm, available_chips):
        mock_llm.chat.return_value = LLMResponse(
            content="not valid json",
            input_tokens=100,
            output_tokens=50,
            latency_ms=500,
        )

        selected, response = selector.select_chips(
            available_chips=available_chips,
            persona="A tech PM",
            style="terse",
        )

        assert selected == []
        assert "Failed to parse" in response.error

    def test_error_on_non_array_response(self, selector, mock_llm, available_chips):
        mock_llm.chat.return_value = LLMResponse(
            content='{"key": "sprint_planning"}',
            input_tokens=100,
            output_tokens=50,
            latency_ms=500,
        )

        selected, response = selector.select_chips(
            available_chips=available_chips,
            persona="A tech PM",
            style="terse",
        )

        assert selected == []
        assert "Expected array" in response.error

    def test_formats_prompt_with_persona(self, selector, mock_llm, available_chips):
        mock_llm.chat.return_value = LLMResponse(
            content='[]',
            input_tokens=100,
            output_tokens=50,
            latency_ms=500,
        )

        selector.select_chips(
            available_chips=available_chips,
            persona="A senior developer",
            style="terse",
        )

        call_args = mock_llm.chat.call_args
        user_prompt = call_args[0][2]
        assert "senior developer" in user_prompt

    def test_formats_prompt_with_chips(self, selector, mock_llm, available_chips):
        mock_llm.chat.return_value = LLMResponse(
            content='[]',
            input_tokens=100,
            output_tokens=50,
            latency_ms=500,
        )

        selector.select_chips(
            available_chips=available_chips,
            persona="A tech PM",
            style="terse",
        )

        call_args = mock_llm.chat.call_args
        user_prompt = call_args[0][2]
        assert "sprint_planning" in user_prompt
        assert "okr" in user_prompt

    def test_uses_fixed_selector_model(self, selector, mock_llm, available_chips):
        mock_llm.chat.return_value = LLMResponse(
            content='[]',
            input_tokens=100,
            output_tokens=50,
            latency_ms=500,
        )

        selector.select_chips(
            available_chips=available_chips,
            persona="A tech PM",
            style="terse",
        )

        call_args = mock_llm.chat.call_args
        model_used = call_args[0][0]
        assert model_used == selector.model

    def test_empty_chips_input(self, selector, mock_llm):
        mock_llm.chat.return_value = LLMResponse(
            content='[]',
            input_tokens=100,
            output_tokens=50,
            latency_ms=500,
        )

        selected, response = selector.select_chips(
            available_chips=[],
            persona="A tech PM",
            style="terse",
        )

        assert selected == []
        assert response.error is None

    def test_selects_all_available(self, selector, mock_llm, available_chips):
        import json
        all_keys = [c.key for c in available_chips]
        mock_llm.chat.return_value = LLMResponse(
            content=json.dumps(all_keys),
            input_tokens=100,
            output_tokens=50,
            latency_ms=500,
        )

        selected, response = selector.select_chips(
            available_chips=available_chips,
            persona="A tech PM",
            style="terse",
        )

        assert len(selected) == len(available_chips)
