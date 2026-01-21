"""
Tests for JSON helper utilities.

Tests cover:
- safe_json_loads
- extract_json_from_text
- extract_and_parse_json
- _extract_balanced_json
"""

import pytest

from aragora.utils.json_helpers import (
    safe_json_loads,
    extract_json_from_text,
    extract_and_parse_json,
    _extract_balanced_json,
)


class TestSafeJsonLoads:
    """Tests for safe_json_loads function."""

    def test_valid_json(self):
        """Parses valid JSON."""
        result = safe_json_loads('{"key": "value"}')
        assert result == {"key": "value"}

    def test_valid_json_array(self):
        """Parses valid JSON array."""
        result = safe_json_loads('[1, 2, 3]')
        assert result == [1, 2, 3]

    def test_none_input(self):
        """Returns default for None input."""
        result = safe_json_loads(None)
        assert result == {}

    def test_empty_string(self):
        """Returns default for empty string."""
        result = safe_json_loads("")
        assert result == {}

    def test_invalid_json(self):
        """Returns default for invalid JSON."""
        result = safe_json_loads("not json")
        assert result == {}

    def test_custom_default(self):
        """Uses custom default value."""
        result = safe_json_loads("invalid", default=[])
        assert result == []

    def test_custom_default_none_input(self):
        """Uses custom default for None input."""
        result = safe_json_loads(None, default={"fallback": True})
        assert result == {"fallback": True}

    def test_context_logging(self):
        """Accepts context parameter for logging."""
        # Just verify it doesn't raise
        result = safe_json_loads("invalid", context="test:context")
        assert result == {}

    def test_nested_json(self):
        """Parses nested JSON."""
        data = '{"outer": {"inner": [1, 2, {"deep": true}]}}'
        result = safe_json_loads(data)
        assert result["outer"]["inner"][2]["deep"] is True

    def test_json_with_unicode(self):
        """Parses JSON with unicode characters."""
        result = safe_json_loads('{"emoji": "🎉", "text": "café"}')
        assert result["emoji"] == "🎉"
        assert result["text"] == "café"


class TestExtractBalancedJson:
    """Tests for _extract_balanced_json function."""

    def test_simple_object(self):
        """Extracts simple JSON object."""
        text = 'Some text {"key": "value"} more text'
        result = _extract_balanced_json(text, "{", "}")
        assert result == '{"key": "value"}'

    def test_nested_object(self):
        """Extracts nested JSON object."""
        text = 'Result: {"outer": {"inner": "value"}} done'
        result = _extract_balanced_json(text, "{", "}")
        assert result == '{"outer": {"inner": "value"}}'

    def test_deeply_nested(self):
        """Extracts deeply nested structure."""
        text = '{"a": {"b": {"c": {"d": 1}}}}'
        result = _extract_balanced_json(text, "{", "}")
        assert result == '{"a": {"b": {"c": {"d": 1}}}}'

    def test_array_extraction(self):
        """Extracts JSON array."""
        text = 'Data: [1, 2, [3, 4]] end'
        result = _extract_balanced_json(text, "[", "]")
        assert result == "[1, 2, [3, 4]]"

    def test_braces_in_string(self):
        """Handles braces inside string values."""
        text = '{"text": "contains { and } braces"}'
        result = _extract_balanced_json(text, "{", "}")
        assert result == '{"text": "contains { and } braces"}'

    def test_escaped_quotes(self):
        """Handles escaped quotes in strings."""
        text = r'{"text": "has \"escaped\" quotes"}'
        result = _extract_balanced_json(text, "{", "}")
        assert result is not None
        # Should contain the full object
        assert "escaped" in result

    def test_no_json_found(self):
        """Returns None when no JSON found."""
        text = "No JSON here"
        result = _extract_balanced_json(text, "{", "}")
        assert result is None

    def test_unbalanced_json(self):
        """Returns None for unbalanced braces."""
        text = '{"key": "value"'
        result = _extract_balanced_json(text, "{", "}")
        assert result is None


class TestExtractJsonFromText:
    """Tests for extract_json_from_text function."""

    def test_json_in_code_block(self):
        """Extracts JSON from markdown code block."""
        text = '```json\n{"key": "value"}\n```'
        result = extract_json_from_text(text)
        assert result == '{"key": "value"}'

    def test_json_in_plain_code_block(self):
        """Extracts JSON from plain code block."""
        text = '```\n{"key": "value"}\n```'
        result = extract_json_from_text(text)
        assert result == '{"key": "value"}'

    def test_array_in_code_block(self):
        """Extracts array from code block."""
        text = '```json\n[1, 2, 3]\n```'
        result = extract_json_from_text(text)
        assert result == "[1, 2, 3]"

    def test_raw_json_object(self):
        """Extracts raw JSON object."""
        text = 'Here is the result: {"key": "value"} and more text'
        result = extract_json_from_text(text)
        assert result == '{"key": "value"}'

    def test_raw_json_array(self):
        """Extracts raw JSON array."""
        text = 'The list is [1, 2, 3] as expected'
        result = extract_json_from_text(text)
        assert result == "[1, 2, 3]"

    def test_nested_json(self):
        """Extracts nested JSON correctly."""
        text = 'Result: {"outer": {"inner": [1, 2]}} done'
        result = extract_json_from_text(text)
        assert result == '{"outer": {"inner": [1, 2]}}'

    def test_no_json(self):
        """Returns original text when no JSON found."""
        text = "No JSON here at all"
        result = extract_json_from_text(text)
        assert result == text

    def test_prefers_code_block(self):
        """Prefers JSON in code block over raw JSON."""
        text = '{"raw": true}\n```json\n{"block": true}\n```'
        result = extract_json_from_text(text)
        assert result == '{"block": true}'

    def test_multiline_code_block(self):
        """Handles multiline JSON in code block."""
        text = '''```json
{
  "key1": "value1",
  "key2": "value2"
}
```'''
        result = extract_json_from_text(text)
        assert '"key1"' in result
        assert '"key2"' in result

    def test_llm_style_response(self):
        """Handles typical LLM response format."""
        text = """Here is the analysis:

```json
{
  "conclusion": "approved",
  "confidence": 0.95
}
```

This is based on the evidence provided."""
        result = extract_json_from_text(text)
        parsed = safe_json_loads(result)
        assert parsed["conclusion"] == "approved"
        assert parsed["confidence"] == 0.95


class TestExtractAndParseJson:
    """Tests for extract_and_parse_json function."""

    def test_basic_extraction_and_parse(self):
        """Extracts and parses JSON in one step."""
        text = 'Result: {"key": "value"}'
        result = extract_and_parse_json(text)
        assert result == {"key": "value"}

    def test_code_block_extraction(self):
        """Extracts from code block and parses."""
        text = '```json\n{"status": "ok"}\n```'
        result = extract_and_parse_json(text)
        assert result == {"status": "ok"}

    def test_default_on_failure(self):
        """Returns default when extraction fails."""
        text = "No JSON here"
        result = extract_and_parse_json(text, default={"error": True})
        assert result == {"error": True}

    def test_default_on_invalid_json(self):
        """Returns default when JSON is invalid."""
        text = '{"unclosed": '
        result = extract_and_parse_json(text, default=[])
        assert result == []

    def test_nested_structure(self):
        """Handles nested structures."""
        text = 'Data: {"users": [{"name": "Alice"}, {"name": "Bob"}]}'
        result = extract_and_parse_json(text)
        assert len(result["users"]) == 2
        assert result["users"][0]["name"] == "Alice"

    def test_context_parameter(self):
        """Accepts context for logging."""
        text = "invalid json here"
        result = extract_and_parse_json(text, context="test:123")
        assert result == {}

    def test_array_result(self):
        """Returns array when array extracted."""
        text = "List: [1, 2, 3, 4, 5]"
        result = extract_and_parse_json(text)
        assert result == [1, 2, 3, 4, 5]


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_object(self):
        """Handles empty object."""
        result = extract_and_parse_json("{}")
        assert result == {}

    def test_empty_array(self):
        """Handles empty array."""
        result = extract_and_parse_json("[]")
        assert result == []

    def test_json_with_newlines(self):
        """Handles JSON with embedded newlines."""
        text = '{"text": "line1\\nline2"}'
        result = safe_json_loads(text)
        assert result["text"] == "line1\nline2"

    def test_json_with_tabs(self):
        """Handles JSON with embedded tabs."""
        text = '{"text": "col1\\tcol2"}'
        result = safe_json_loads(text)
        assert result["text"] == "col1\tcol2"

    def test_numeric_values(self):
        """Handles various numeric types."""
        text = '{"int": 42, "float": 3.14, "negative": -10, "scientific": 1.5e10}'
        result = safe_json_loads(text)
        assert result["int"] == 42
        assert result["float"] == 3.14
        assert result["negative"] == -10
        assert result["scientific"] == 1.5e10

    def test_boolean_and_null(self):
        """Handles boolean and null values."""
        text = '{"true": true, "false": false, "null": null}'
        result = safe_json_loads(text)
        assert result["true"] is True
        assert result["false"] is False
        assert result["null"] is None

    def test_multiple_json_objects(self):
        """Extracts first JSON object when multiple present."""
        text = '{"first": 1} {"second": 2}'
        result = extract_json_from_text(text)
        assert result == '{"first": 1}'

    def test_json_at_end_of_text(self):
        """Handles JSON at end of text."""
        text = 'The result is {"answer": 42}'
        result = extract_and_parse_json(text)
        assert result == {"answer": 42}

    def test_json_at_start_of_text(self):
        """Handles JSON at start of text."""
        text = '{"answer": 42} is the result'
        result = extract_and_parse_json(text)
        assert result == {"answer": 42}
