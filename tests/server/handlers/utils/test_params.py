"""Tests for params module."""

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

import pytest

from aragora.server.handlers.utils.params import (
    parse_query_params,
    get_int_param,
    get_float_param,
    get_bool_param,
    get_string_param,
    get_clamped_int_param,
    get_bounded_float_param,
    get_bounded_string_param,
)


# =============================================================================
# Test parse_query_params
# =============================================================================


class TestParseQueryParams:
    """Tests for parse_query_params function."""

    def test_parses_single_values(self):
        """Should parse single values from query string."""
        result = parse_query_params("name=Alice&age=30")
        assert result["name"] == "Alice"
        assert result["age"] == "30"

    def test_parses_multi_values_as_list(self):
        """Should parse multiple values as list."""
        result = parse_query_params("tag=a&tag=b&tag=c")
        assert result["tag"] == ["a", "b", "c"]

    def test_returns_empty_dict_for_empty_string(self):
        """Should return empty dict for empty query string."""
        result = parse_query_params("")
        assert result == {}

    def test_handles_url_encoded_values(self):
        """Should handle URL encoded values."""
        result = parse_query_params("name=Hello%20World")
        assert result["name"] == "Hello World"


# =============================================================================
# Test get_int_param
# =============================================================================


class TestGetIntParam:
    """Tests for get_int_param function."""

    def test_returns_int_value(self):
        """Should return integer value."""
        params = {"limit": "50"}
        assert get_int_param(params, "limit") == 50

    def test_returns_default_when_missing(self):
        """Should return default when key is missing."""
        params = {}
        assert get_int_param(params, "limit", 10) == 10

    def test_returns_default_for_invalid_value(self):
        """Should return default for non-numeric value."""
        params = {"limit": "invalid"}
        assert get_int_param(params, "limit", 10) == 10

    def test_handles_list_values(self):
        """Should handle list values from query strings."""
        params = {"limit": ["50", "100"]}
        assert get_int_param(params, "limit") == 50

    def test_returns_default_for_empty_list(self):
        """Should return default for empty list."""
        params = {"limit": []}
        assert get_int_param(params, "limit", 25) == 25

    def test_handles_negative_values(self):
        """Should handle negative values."""
        params = {"offset": "-10"}
        assert get_int_param(params, "offset") == -10


# =============================================================================
# Test get_float_param
# =============================================================================


class TestGetFloatParam:
    """Tests for get_float_param function."""

    def test_returns_float_value(self):
        """Should return float value."""
        params = {"threshold": "0.75"}
        assert get_float_param(params, "threshold") == 0.75

    def test_returns_default_when_missing(self):
        """Should return default when key is missing."""
        params = {}
        assert get_float_param(params, "threshold", 0.5) == 0.5

    def test_returns_default_for_invalid_value(self):
        """Should return default for non-numeric value."""
        params = {"threshold": "invalid"}
        assert get_float_param(params, "threshold", 0.5) == 0.5

    def test_handles_list_values(self):
        """Should handle list values."""
        params = {"threshold": ["0.8", "0.9"]}
        assert get_float_param(params, "threshold") == 0.8

    def test_handles_integer_strings(self):
        """Should handle integer strings."""
        params = {"threshold": "1"}
        assert get_float_param(params, "threshold") == 1.0


# =============================================================================
# Test get_bool_param
# =============================================================================


class TestGetBoolParam:
    """Tests for get_bool_param function."""

    def test_returns_true_for_true_string(self):
        """Should return True for 'true' string."""
        params = {"active": "true"}
        assert get_bool_param(params, "active") is True

    def test_returns_true_for_1_string(self):
        """Should return True for '1' string."""
        params = {"active": "1"}
        assert get_bool_param(params, "active") is True

    def test_returns_true_for_yes_string(self):
        """Should return True for 'yes' string."""
        params = {"active": "yes"}
        assert get_bool_param(params, "active") is True

    def test_returns_true_for_on_string(self):
        """Should return True for 'on' string."""
        params = {"active": "on"}
        assert get_bool_param(params, "active") is True

    def test_returns_false_for_false_string(self):
        """Should return False for 'false' string."""
        params = {"active": "false"}
        assert get_bool_param(params, "active") is False

    def test_returns_default_when_missing(self):
        """Should return default when key is missing."""
        params = {}
        assert get_bool_param(params, "active", True) is True

    def test_handles_boolean_values(self):
        """Should handle actual boolean values."""
        params = {"active": True}
        assert get_bool_param(params, "active") is True
        params = {"active": False}
        assert get_bool_param(params, "active") is False

    def test_handles_list_values(self):
        """Should handle list values."""
        params = {"active": ["true", "false"]}
        assert get_bool_param(params, "active") is True

    def test_case_insensitive(self):
        """Should be case insensitive."""
        assert get_bool_param({"a": "TRUE"}, "a") is True
        assert get_bool_param({"a": "True"}, "a") is True


# =============================================================================
# Test get_string_param
# =============================================================================


class TestGetStringParam:
    """Tests for get_string_param function."""

    def test_returns_string_value(self):
        """Should return string value."""
        params = {"name": "Alice"}
        assert get_string_param(params, "name") == "Alice"

    def test_returns_default_when_missing(self):
        """Should return default when key is missing."""
        params = {}
        assert get_string_param(params, "name", "default") == "default"

    def test_returns_none_when_missing_no_default(self):
        """Should return None when missing and no default."""
        params = {}
        assert get_string_param(params, "name") is None

    def test_handles_list_values(self):
        """Should handle list values."""
        params = {"name": ["Alice", "Bob"]}
        assert get_string_param(params, "name") == "Alice"

    def test_converts_non_string_to_string(self):
        """Should convert non-string values to string."""
        params = {"count": 123}
        assert get_string_param(params, "count") == "123"


# =============================================================================
# Test get_clamped_int_param
# =============================================================================


class TestGetClampedIntParam:
    """Tests for get_clamped_int_param function."""

    def test_returns_value_within_range(self):
        """Should return value when within range."""
        params = {"limit": "50"}
        assert get_clamped_int_param(params, "limit", 10, 1, 100) == 50

    def test_clamps_to_min(self):
        """Should clamp value to minimum."""
        params = {"limit": "0"}
        assert get_clamped_int_param(params, "limit", 10, 5, 100) == 5

    def test_clamps_to_max(self):
        """Should clamp value to maximum."""
        params = {"limit": "200"}
        assert get_clamped_int_param(params, "limit", 10, 1, 50) == 50

    def test_uses_default_when_missing(self):
        """Should use default when parameter is missing."""
        params = {}
        assert get_clamped_int_param(params, "limit", 25, 1, 100) == 25


# =============================================================================
# Test get_bounded_float_param
# =============================================================================


class TestGetBoundedFloatParam:
    """Tests for get_bounded_float_param function."""

    def test_returns_value_within_range(self):
        """Should return value when within range."""
        params = {"threshold": "0.5"}
        assert get_bounded_float_param(params, "threshold", 0.0, 0.0, 1.0) == 0.5

    def test_bounds_to_min(self):
        """Should bound value to minimum."""
        params = {"threshold": "-0.5"}
        assert get_bounded_float_param(params, "threshold", 0.0, 0.0, 1.0) == 0.0

    def test_bounds_to_max(self):
        """Should bound value to maximum."""
        params = {"threshold": "1.5"}
        assert get_bounded_float_param(params, "threshold", 0.0, 0.0, 1.0) == 1.0


# =============================================================================
# Test get_bounded_string_param
# =============================================================================


class TestGetBoundedStringParam:
    """Tests for get_bounded_string_param function."""

    def test_returns_string_value(self):
        """Should return string value."""
        params = {"query": "search term"}
        assert get_bounded_string_param(params, "query") == "search term"

    def test_truncates_to_max_length(self):
        """Should truncate string to max length."""
        params = {"query": "a" * 1000}
        result = get_bounded_string_param(params, "query", max_length=10)
        assert len(result) == 10

    def test_returns_none_when_missing(self):
        """Should return None when missing."""
        params = {}
        assert get_bounded_string_param(params, "query") is None

    def test_returns_default_when_missing(self):
        """Should return default when missing."""
        params = {}
        assert get_bounded_string_param(params, "query", default="default") == "default"
