"""Tests for safe_data module."""

from __future__ import annotations

import sys
import types as _types_mod

# Pre-stub Slack modules to prevent import chain failures
_SLACK_ATTRS = [
    "SlackHandler",
    "get_slack_handler",
    "get_slack_integration",
    "get_workspace_store",
    "resolve_workspace",
    "create_tracked_task",
    "_validate_slack_url",
    "SLACK_SIGNING_SECRET",
    "SLACK_BOT_TOKEN",
    "SLACK_WEBHOOK_URL",
    "SLACK_ALLOWED_DOMAINS",
    "SignatureVerifierMixin",
    "CommandsMixin",
    "EventsMixin",
    "init_slack_handler",
]
for _mod_name in (
    "aragora.server.handlers.social.slack.handler",
    "aragora.server.handlers.social.slack",
    "aragora.server.handlers.social._slack_impl",
):
    if _mod_name not in sys.modules:
        _m = _types_mod.ModuleType(_mod_name)
        for _a in _SLACK_ATTRS:
            setattr(_m, _a, None)
        sys.modules[_mod_name] = _m

import json

import pytest

from aragora.server.handlers.utils.safe_data import (
    safe_get,
    safe_get_nested,
    safe_json_parse,
)


# =============================================================================
# Test safe_get
# =============================================================================


class TestSafeGet:
    """Tests for safe_get function."""

    def test_returns_value_when_key_exists(self):
        """Should return value when key exists."""
        data = {"name": "Alice", "age": 30}
        assert safe_get(data, "name") == "Alice"
        assert safe_get(data, "age") == 30

    def test_returns_default_when_key_missing(self):
        """Should return default when key is missing."""
        data = {"name": "Alice"}
        assert safe_get(data, "missing") is None
        assert safe_get(data, "missing", "default") == "default"

    def test_returns_default_when_data_is_none(self):
        """Should return default when data is None."""
        assert safe_get(None, "key") is None
        assert safe_get(None, "key", []) == []

    def test_returns_default_when_data_is_not_dict(self):
        """Should return default when data is not a dict."""
        assert safe_get("string", "key") is None
        assert safe_get(123, "key") is None
        assert safe_get(["list"], "key") is None

    def test_uses_custom_default(self):
        """Should use custom default value."""
        data = {"a": 1}
        assert safe_get(data, "b", default=[]) == []
        assert safe_get(data, "b", default=0) == 0
        assert safe_get(data, "b", default={"empty": True}) == {"empty": True}

    def test_returns_falsy_values(self):
        """Should return falsy values when they exist."""
        data = {"zero": 0, "empty": "", "false": False, "none": None}
        assert safe_get(data, "zero") == 0
        assert safe_get(data, "empty") == ""
        assert safe_get(data, "false") is False
        # Note: None is treated as missing when using .get()
        assert safe_get(data, "none") is None


# =============================================================================
# Test safe_get_nested
# =============================================================================


class TestSafeGetNested:
    """Tests for safe_get_nested function."""

    def test_returns_nested_value(self):
        """Should return nested value when path exists."""
        data = {"outer": {"inner": {"deep": "value"}}}
        assert safe_get_nested(data, ["outer", "inner", "deep"]) == "value"

    def test_returns_default_when_path_missing(self):
        """Should return default when path is missing."""
        data = {"outer": {"inner": {}}}
        assert safe_get_nested(data, ["outer", "missing", "deep"]) is None
        assert safe_get_nested(data, ["outer", "missing", "deep"], "default") == "default"

    def test_returns_default_when_data_is_none(self):
        """Should return default when data is None."""
        assert safe_get_nested(None, ["a", "b"]) is None
        assert safe_get_nested(None, ["a", "b"], []) == []

    def test_returns_default_when_intermediate_is_not_dict(self):
        """Should return default when intermediate value is not a dict."""
        data = {"outer": "not_a_dict"}
        assert safe_get_nested(data, ["outer", "inner"]) is None

    def test_handles_empty_keys_list(self):
        """Should handle empty keys list."""
        data = {"a": 1}
        # Empty keys means return None (since we never navigate)
        assert safe_get_nested(data, []) is None

    def test_uses_custom_default(self):
        """Should use custom default value."""
        data = {"outer": {}}
        assert safe_get_nested(data, ["outer", "inner", "deep"], []) == []
        assert safe_get_nested(data, ["missing"], {"default": True}) == {"default": True}

    def test_returns_intermediate_dict(self):
        """Should return intermediate dict when that's the target."""
        data = {"outer": {"inner": {"key": "value"}}}
        result = safe_get_nested(data, ["outer", "inner"])
        assert result == {"key": "value"}

    def test_handles_deeply_nested(self):
        """Should handle deeply nested structures."""
        data = {"a": {"b": {"c": {"d": {"e": "deep_value"}}}}}
        assert safe_get_nested(data, ["a", "b", "c", "d", "e"]) == "deep_value"


# =============================================================================
# Test safe_json_parse
# =============================================================================


class TestSafeJsonParse:
    """Tests for safe_json_parse function."""

    def test_parses_json_string(self):
        """Should parse valid JSON string."""
        result = safe_json_parse('{"key": "value"}')
        assert result == {"key": "value"}

    def test_parses_json_array(self):
        """Should parse JSON array string."""
        result = safe_json_parse("[1, 2, 3]")
        assert result == [1, 2, 3]

    def test_returns_dict_as_is(self):
        """Should return dict without modification."""
        data = {"already": "parsed"}
        result = safe_json_parse(data)
        assert result == data

    def test_returns_list_as_is(self):
        """Should return list without modification."""
        data = [1, 2, 3]
        result = safe_json_parse(data)
        assert result == data

    def test_returns_default_for_none(self):
        """Should return default for None input."""
        assert safe_json_parse(None) is None
        assert safe_json_parse(None, {}) == {}

    def test_returns_default_for_invalid_json(self):
        """Should return default for invalid JSON."""
        assert safe_json_parse("not valid json") is None
        assert safe_json_parse("not valid json", {"error": True}) == {"error": True}

    def test_returns_default_for_non_string_non_dict(self):
        """Should return default for non-string/non-dict input."""
        assert safe_json_parse(123) is None
        assert safe_json_parse(True) is None

    def test_parses_bytes(self):
        """Should parse JSON from bytes."""
        result = safe_json_parse(b'{"from": "bytes"}')
        assert result == {"from": "bytes"}

    def test_parses_bytearray(self):
        """Should parse JSON from bytearray."""
        result = safe_json_parse(bytearray(b'{"from": "bytearray"}'))
        assert result == {"from": "bytearray"}

    def test_handles_nested_json_string(self):
        """Should parse nested JSON structures."""
        nested = json.dumps({"outer": {"inner": [1, 2, 3]}})
        result = safe_json_parse(nested)
        assert result["outer"]["inner"] == [1, 2, 3]

    def test_handles_unicode_json(self):
        """Should handle unicode in JSON."""
        result = safe_json_parse('{"emoji": "Hello World"}')
        assert result["emoji"] == "Hello World"
