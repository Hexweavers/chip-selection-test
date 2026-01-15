import pytest
from unittest.mock import Mock, patch
from models.chip import Chip
from utils.fill import FillService
from services.llm import LLMResponse


class TestFillServiceGetMissingTypes:
    def test_all_types_missing(self):
        with patch.object(FillService, '_load_prompts', return_value={}):
            service = FillService(Mock())

        chips = []
        missing = service.get_missing_types(chips, min_per_type=2)
        assert set(missing) == {"situation", "jargon", "role_task", "environment"}

    def test_some_types_missing(self):
        with patch.object(FillService, '_load_prompts', return_value={}):
            service = FillService(Mock())

        chips = [
            Chip(key="s1", display="S1", type="situation"),
            Chip(key="s2", display="S2", type="situation"),
            Chip(key="j1", display="J1", type="jargon"),
            Chip(key="j2", display="J2", type="jargon"),
        ]
        missing = service.get_missing_types(chips, min_per_type=2)
        assert set(missing) == {"role_task", "environment"}

    def test_no_types_missing(self):
        with patch.object(FillService, '_load_prompts', return_value={}):
            service = FillService(Mock())

        chips = [
            Chip(key="s1", display="S1", type="situation"),
            Chip(key="s2", display="S2", type="situation"),
            Chip(key="j1", display="J1", type="jargon"),
            Chip(key="j2", display="J2", type="jargon"),
            Chip(key="r1", display="R1", type="role_task"),
            Chip(key="r2", display="R2", type="role_task"),
            Chip(key="e1", display="E1", type="environment"),
            Chip(key="e2", display="E2", type="environment"),
        ]
        missing = service.get_missing_types(chips, min_per_type=2)
        assert missing == []

    def test_custom_min_per_type(self):
        with patch.object(FillService, '_load_prompts', return_value={}):
            service = FillService(Mock())

        chips = [
            Chip(key="s1", display="S1", type="situation"),
            Chip(key="s2", display="S2", type="situation"),
            Chip(key="s3", display="S3", type="situation"),
        ]
        missing = service.get_missing_types(chips, min_per_type=3)
        assert set(missing) == {"jargon", "role_task", "environment"}

    def test_partial_coverage(self):
        with patch.object(FillService, '_load_prompts', return_value={}):
            service = FillService(Mock())

        chips = [
            Chip(key="s1", display="S1", type="situation"),
            Chip(key="j1", display="J1", type="jargon"),
        ]
        missing = service.get_missing_types(chips, min_per_type=2)
        assert set(missing) == {"situation", "jargon", "role_task", "environment"}


class TestFillServiceFillMissing:
    def test_fill_missing_no_missing_types(self):
        with patch.object(FillService, '_load_prompts', return_value={}):
            service = FillService(Mock())

        chips, response = service.fill_missing(
            model="test-model",
            sector="Tech",
            desired_role="Dev",
            existing_chips=[],
            missing_types=[],
            style="terse",
        )

        assert chips == []
        assert response.input_tokens == 0
        assert response.output_tokens == 0

    def test_fill_missing_with_llm_response(self):
        mock_llm = Mock()
        mock_llm.chat.return_value = LLMResponse(
            content='[{"key": "r1", "display": "Role Task 1", "type": "role_task"}]',
            input_tokens=100,
            output_tokens=50,
            latency_ms=200,
        )

        mock_prompts = {
            "terse": {
                "fill_missing_types": {
                    "system": "System prompt",
                    "user": "Fill {sector} {desired_role} {existing_chips} {missing_types}",
                }
            }
        }

        with patch.object(FillService, '_load_prompts', return_value=mock_prompts):
            service = FillService(mock_llm)

        chips, response = service.fill_missing(
            model="test-model",
            sector="Tech",
            desired_role="Dev",
            existing_chips=[],
            missing_types=["role_task"],
            style="terse",
        )

        assert len(chips) == 1
        assert chips[0].type == "role_task"
        assert response.input_tokens == 100

    def test_fill_missing_with_llm_error(self):
        mock_llm = Mock()
        mock_llm.chat.return_value = LLMResponse(
            content="",
            input_tokens=0,
            output_tokens=0,
            latency_ms=100,
            error="API timeout",
        )

        mock_prompts = {
            "terse": {
                "fill_missing_types": {
                    "system": "System prompt",
                    "user": "Fill {sector} {desired_role} {existing_chips} {missing_types}",
                }
            }
        }

        with patch.object(FillService, '_load_prompts', return_value=mock_prompts):
            service = FillService(mock_llm)

        chips, response = service.fill_missing(
            model="test-model",
            sector="Tech",
            desired_role="Dev",
            existing_chips=[],
            missing_types=["role_task"],
            style="terse",
        )

        assert chips == []
        assert response.error == "API timeout"

    def test_fill_missing_with_parse_errors(self):
        mock_llm = Mock()
        mock_llm.chat.return_value = LLMResponse(
            content='[{"key": "", "display": "Bad", "type": "role_task"}]',
            input_tokens=100,
            output_tokens=50,
            latency_ms=200,
        )

        mock_prompts = {
            "terse": {
                "fill_missing_types": {
                    "system": "System prompt",
                    "user": "Fill {sector} {desired_role} {existing_chips} {missing_types}",
                }
            }
        }

        with patch.object(FillService, '_load_prompts', return_value=mock_prompts):
            service = FillService(mock_llm)

        chips, response = service.fill_missing(
            model="test-model",
            sector="Tech",
            desired_role="Dev",
            existing_chips=[],
            missing_types=["role_task"],
            style="terse",
        )

        assert chips == []
        assert response.error is not None
        assert "key is required" in response.error
