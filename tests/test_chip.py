import pytest
from models.chip import Chip, TestMetadata, TestResult, parse_chips_from_json


class TestChip:
    def test_to_dict(self):
        chip = Chip(key="test_key", display="Test Display", type="situation")
        result = chip.to_dict()
        assert result == {
            "key": "test_key",
            "display": "Test Display",
            "type": "situation",
        }

    def test_from_dict(self):
        data = {"key": "my_key", "display": "My Display", "type": "jargon"}
        chip = Chip.from_dict(data)
        assert chip.key == "my_key"
        assert chip.display == "My Display"
        assert chip.type == "jargon"

    def test_validate_valid_chip(self):
        chip = Chip(key="valid", display="Valid Chip", type="environment")
        errors = chip.validate()
        assert errors == []

    def test_validate_empty_key(self):
        chip = Chip(key="", display="Display", type="situation")
        errors = chip.validate()
        assert "key is required" in errors

    def test_validate_empty_display(self):
        chip = Chip(key="key", display="", type="situation")
        errors = chip.validate()
        assert "display is required" in errors

    def test_validate_invalid_type(self):
        chip = Chip(key="key", display="Display", type="invalid_type")
        errors = chip.validate()
        assert any("type must be one of" in e for e in errors)

    def test_validate_multiple_errors(self):
        chip = Chip(key="", display="", type="bad")
        errors = chip.validate()
        assert len(errors) == 3


class TestParseChipsFromJson:
    def test_parse_valid_array(self):
        json_str = '''[
            {"key": "chip1", "display": "Chip 1", "type": "situation"},
            {"key": "chip2", "display": "Chip 2", "type": "jargon"}
        ]'''
        chips, errors = parse_chips_from_json(json_str)
        assert len(chips) == 2
        assert len(errors) == 0
        assert chips[0].key == "chip1"
        assert chips[1].type == "jargon"

    def test_parse_object_with_chips_key(self):
        json_str = '''{"chips": [
            {"key": "chip1", "display": "Chip 1", "type": "environment"}
        ]}'''
        chips, errors = parse_chips_from_json(json_str)
        assert len(chips) == 1
        assert chips[0].key == "chip1"

    def test_parse_markdown_code_block(self):
        json_str = '''```json
[{"key": "chip1", "display": "Chip 1", "type": "role_task"}]
```'''
        chips, errors = parse_chips_from_json(json_str)
        assert len(chips) == 1
        assert chips[0].type == "role_task"

    def test_parse_markdown_code_block_no_language(self):
        json_str = '''```
[{"key": "chip1", "display": "Chip 1", "type": "situation"}]
```'''
        chips, errors = parse_chips_from_json(json_str)
        assert len(chips) == 1

    def test_parse_invalid_json(self):
        json_str = "not valid json"
        chips, errors = parse_chips_from_json(json_str)
        assert len(chips) == 0
        assert len(errors) == 1
        assert "JSON parse error" in errors[0]

    def test_parse_non_array_response(self):
        json_str = '{"key": "chip1", "display": "Chip 1", "type": "situation"}'
        chips, errors = parse_chips_from_json(json_str)
        assert len(chips) == 0
        assert "Expected array" in errors[0]

    def test_parse_missing_required_field(self):
        json_str = '[{"key": "chip1", "display": "Chip 1"}]'
        chips, errors = parse_chips_from_json(json_str)
        assert len(chips) == 0
        assert len(errors) == 1
        assert "Chip 0" in errors[0]

    def test_parse_invalid_chip_type(self):
        json_str = '[{"key": "chip1", "display": "Chip 1", "type": "invalid"}]'
        chips, errors = parse_chips_from_json(json_str)
        assert len(chips) == 0
        assert len(errors) == 1
        assert "type must be one of" in errors[0]

    def test_parse_mixed_valid_invalid(self):
        json_str = '''[
            {"key": "valid", "display": "Valid", "type": "situation"},
            {"key": "", "display": "Invalid", "type": "situation"},
            {"key": "also_valid", "display": "Also Valid", "type": "jargon"}
        ]'''
        chips, errors = parse_chips_from_json(json_str)
        assert len(chips) == 2
        assert len(errors) == 1
        assert chips[0].key == "valid"
        assert chips[1].key == "also_valid"

    def test_parse_empty_array(self):
        json_str = "[]"
        chips, errors = parse_chips_from_json(json_str)
        assert len(chips) == 0
        assert len(errors) == 0

    def test_parse_whitespace_handling(self):
        json_str = '''
        [{"key": "chip1", "display": "Chip 1", "type": "situation"}]
        '''
        chips, errors = parse_chips_from_json(json_str)
        assert len(chips) == 1

    def test_parse_all_chip_types(self):
        json_str = '''[
            {"key": "s1", "display": "Situation", "type": "situation"},
            {"key": "j1", "display": "Jargon", "type": "jargon"},
            {"key": "r1", "display": "Role Task", "type": "role_task"},
            {"key": "e1", "display": "Environment", "type": "environment"}
        ]'''
        chips, errors = parse_chips_from_json(json_str)
        assert len(chips) == 4
        assert len(errors) == 0
        types = {c.type for c in chips}
        assert types == {"situation", "jargon", "role_task", "environment"}


class TestTestMetadata:
    def test_to_dict(self):
        metadata = TestMetadata(
            model="test-model",
            persona_id="p1",
            sector="Technology",
            desired_role="Developer",
            style="terse",
            constraint="with_constraint",
            input_type="basic",
            chip_count=15,
        )
        result = metadata.to_dict()
        assert result["model"] == "test-model"
        assert result["persona_id"] == "p1"
        assert result["chip_count"] == 15
        assert "timestamp" in result


class TestTestResult:
    def test_to_dict(self):
        metadata = TestMetadata(
            model="test-model",
            persona_id="p1",
            sector="Tech",
            desired_role="Dev",
            style="terse",
            constraint="no_constraint",
            input_type="enriched",
            chip_count=15,
        )
        chip = Chip(key="c1", display="Chip 1", type="situation")
        result = TestResult(
            metadata=metadata,
            final_chips=[chip],
            latency_ms=100,
            cost_usd=0.001,
        )
        data = result.to_dict()
        assert data["metadata"]["model"] == "test-model"
        assert len(data["final_chips"]) == 1
        assert data["latency_ms"] == 100
        assert data["cost_usd"] == 0.001

    def test_count_by_type(self):
        metadata = TestMetadata(
            model="m", persona_id="p", sector="s", desired_role="r",
            style="terse", constraint="c", input_type="basic", chip_count=15,
        )
        chips = [
            Chip(key="s1", display="S1", type="situation"),
            Chip(key="s2", display="S2", type="situation"),
            Chip(key="j1", display="J1", type="jargon"),
        ]
        result = TestResult(metadata=metadata, final_chips=chips)
        counts = result.count_by_type()
        assert counts["situation"] == 2
        assert counts["jargon"] == 1
        assert counts["role_task"] == 0
        assert counts["environment"] == 0

    def test_count_by_type_custom_chips(self):
        metadata = TestMetadata(
            model="m", persona_id="p", sector="s", desired_role="r",
            style="terse", constraint="c", input_type="basic", chip_count=15,
        )
        chips = [Chip(key="e1", display="E1", type="environment")]
        result = TestResult(metadata=metadata, final_chips=[])
        counts = result.count_by_type(chips)
        assert counts["environment"] == 1

    def test_get_missing_types(self):
        metadata = TestMetadata(
            model="m", persona_id="p", sector="s", desired_role="r",
            style="terse", constraint="c", input_type="basic", chip_count=15,
        )
        chips = [
            Chip(key="s1", display="S1", type="situation"),
            Chip(key="s2", display="S2", type="situation"),
            Chip(key="j1", display="J1", type="jargon"),
            Chip(key="j2", display="J2", type="jargon"),
        ]
        result = TestResult(metadata=metadata, final_chips=chips)
        missing = result.get_missing_types(min_per_type=2)
        assert set(missing) == {"role_task", "environment"}

    def test_get_missing_types_all_covered(self):
        metadata = TestMetadata(
            model="m", persona_id="p", sector="s", desired_role="r",
            style="terse", constraint="c", input_type="basic", chip_count=15,
        )
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
        result = TestResult(metadata=metadata, final_chips=chips)
        missing = result.get_missing_types(min_per_type=2)
        assert missing == []
