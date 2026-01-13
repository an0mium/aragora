"""
Tests for query parameter validation functions.

Verifies the parse_*_param and safe_query_* functions handle:
- Normal values within bounds
- Values exceeding bounds (clamped)
- Invalid/malformed values (fallback to default)
- Empty/missing values
- Both parse_qs format (list values) and aiohttp format (single values)
"""

import pytest

from aragora.server.validation import (
    parse_int_param,
    parse_float_param,
    parse_bool_param,
    parse_string_param,
    safe_query_int,
    safe_query_float,
)


class TestParseIntParam:
    """Tests for parse_int_param (parse_qs format with list values)."""

    def test_normal_value(self):
        """Normal value within bounds."""
        query = {"limit": ["50"]}
        assert parse_int_param(query, "limit", default=20) == 50

    def test_missing_key_returns_default(self):
        """Missing key returns default."""
        query = {}
        assert parse_int_param(query, "limit", default=20) == 20

    def test_empty_list_returns_default(self):
        """Empty list returns default."""
        query = {"limit": []}
        assert parse_int_param(query, "limit", default=20) == 20

    def test_invalid_value_returns_default(self):
        """Invalid (non-numeric) value returns default."""
        query = {"limit": ["abc"]}
        assert parse_int_param(query, "limit", default=20) == 20

    def test_value_below_min_clamped(self):
        """Value below min_val is clamped to min_val."""
        query = {"limit": ["-5"]}
        assert parse_int_param(query, "limit", default=20, min_val=1) == 1

    def test_value_above_max_clamped(self):
        """Value above max_val is clamped to max_val."""
        query = {"limit": ["999"]}
        assert parse_int_param(query, "limit", default=20, max_val=100) == 100

    def test_custom_bounds(self):
        """Custom min/max bounds work correctly."""
        query = {"offset": ["5"]}
        result = parse_int_param(query, "offset", default=0, min_val=0, max_val=1000)
        assert result == 5

    def test_negative_allowed_when_min_negative(self):
        """Negative values allowed when min_val is negative."""
        query = {"delta": ["-10"]}
        result = parse_int_param(query, "delta", default=0, min_val=-100, max_val=100)
        assert result == -10


class TestParseFloatParam:
    """Tests for parse_float_param (parse_qs format with list values)."""

    def test_normal_value(self):
        """Normal float value within bounds."""
        query = {"threshold": ["0.75"]}
        assert parse_float_param(query, "threshold", default=0.5) == 0.75

    def test_missing_key_returns_default(self):
        """Missing key returns default."""
        query = {}
        assert parse_float_param(query, "threshold", default=0.5) == 0.5

    def test_invalid_value_returns_default(self):
        """Invalid value returns default."""
        query = {"threshold": ["not-a-number"]}
        assert parse_float_param(query, "threshold", default=0.5) == 0.5

    def test_value_clamped_to_bounds(self):
        """Value outside bounds is clamped."""
        query = {"threshold": ["1.5"]}
        result = parse_float_param(query, "threshold", default=0.5, max_val=1.0)
        assert result == 1.0


class TestParseBoolParam:
    """Tests for parse_bool_param."""

    def test_true_values(self):
        """Various true values recognized."""
        for val in ["true", "1", "yes", "True", "TRUE", "Yes", "YES"]:
            query = {"enabled": [val]}
            assert parse_bool_param(query, "enabled", default=False) is True

    def test_false_values(self):
        """Various false values recognized."""
        for val in ["false", "0", "no", "False", "FALSE", "No", "NO"]:
            query = {"enabled": [val]}
            assert parse_bool_param(query, "enabled", default=True) is False

    def test_missing_key_returns_default(self):
        """Missing key returns default."""
        query = {}
        assert parse_bool_param(query, "enabled", default=True) is True
        assert parse_bool_param(query, "enabled", default=False) is False

    def test_invalid_value_returns_default(self):
        """Invalid value returns default."""
        query = {"enabled": ["maybe"]}
        assert parse_bool_param(query, "enabled", default=True) is True


class TestParseStringParam:
    """Tests for parse_string_param."""

    def test_normal_value(self):
        """Normal string value."""
        query = {"name": ["test-agent"]}
        assert parse_string_param(query, "name", default="") == "test-agent"

    def test_missing_key_returns_default(self):
        """Missing key returns default."""
        query = {}
        assert parse_string_param(query, "name", default="default-name") == "default-name"

    def test_value_truncated_to_max_length(self):
        """Long value truncated to max_length."""
        query = {"name": ["a" * 100]}
        result = parse_string_param(query, "name", default="", max_length=10)
        assert result == "a" * 10

    def test_allowed_values_enforced(self):
        """Value not in allowed_values returns default."""
        query = {"sort": ["invalid"]}
        result = parse_string_param(query, "sort", default="asc", allowed_values={"asc", "desc"})
        assert result == "asc"

    def test_allowed_value_accepted(self):
        """Value in allowed_values is accepted."""
        query = {"sort": ["desc"]}
        result = parse_string_param(query, "sort", default="asc", allowed_values={"asc", "desc"})
        assert result == "desc"


class TestSafeQueryInt:
    """Tests for safe_query_int (works with both formats)."""

    def test_aiohttp_format(self):
        """Works with aiohttp MultiDict format (single string values)."""

        class FakeMultiDict:
            def __init__(self, d):
                self._d = d

            def get(self, k, default=None):
                return self._d.get(k, default)

        query = FakeMultiDict({"limit": "50"})
        assert safe_query_int(query, "limit", default=20) == 50

    def test_parse_qs_format(self):
        """Also works with parse_qs format (list values)."""
        query = {"limit": ["50"]}
        assert safe_query_int(query, "limit", default=20) == 50

    def test_bounds_applied(self):
        """Min/max bounds are applied."""

        class FakeMultiDict:
            def get(self, k, default=None):
                return "999"

        query = FakeMultiDict()
        result = safe_query_int(query, "limit", default=20, max_val=100)
        assert result == 100


class TestSafeQueryFloat:
    """Tests for safe_query_float (works with both formats)."""

    def test_aiohttp_format(self):
        """Works with aiohttp MultiDict format."""

        class FakeMultiDict:
            def __init__(self, d):
                self._d = d

            def get(self, k, default=None):
                return self._d.get(k, default)

        query = FakeMultiDict({"threshold": "0.75"})
        assert safe_query_float(query, "threshold", default=0.5) == 0.75

    def test_bounds_applied(self):
        """Min/max bounds are applied."""

        class FakeMultiDict:
            def get(self, k, default=None):
                return "1.5"

        query = FakeMultiDict()
        result = safe_query_float(query, "threshold", default=0.5, max_val=1.0)
        assert result == 1.0


class TestQueryParamSecurityScenarios:
    """Security-focused tests for query parameter validation."""

    def test_sql_injection_attempt_returns_default(self):
        """SQL injection attempts return default value."""
        query = {"limit": ["1; DROP TABLE users;"]}
        result = parse_int_param(query, "limit", default=20)
        assert result == 20

    def test_very_large_number_clamped(self):
        """Very large numbers are clamped to max_val."""
        query = {"limit": ["999999999999999999999"]}
        result = parse_int_param(query, "limit", default=20, max_val=100)
        assert result == 100

    def test_float_overflow_handled(self):
        """Float overflow returns default."""
        query = {"threshold": ["1e999"]}
        # Should either clamp or return default (behavior may vary)
        result = parse_float_param(query, "threshold", default=0.5, max_val=1.0)
        assert 0.0 <= result <= 1.0 or result == 0.5

    def test_empty_string_returns_default(self):
        """Empty string returns default."""
        query = {"limit": [""]}
        result = parse_int_param(query, "limit", default=20)
        assert result == 20

    def test_whitespace_only_returns_default(self):
        """Whitespace-only value returns default."""
        query = {"limit": ["   "]}
        result = parse_int_param(query, "limit", default=20)
        assert result == 20

    def test_negative_value_with_positive_min(self):
        """Negative value with positive min_val is clamped."""
        query = {"limit": ["-100"]}
        result = parse_int_param(query, "limit", default=20, min_val=1, max_val=100)
        assert result == 1


# =============================================================================
# Tests for Handler Validation Decorators
# =============================================================================

from aragora.server.validation import (
    validate_request,
    validate_post_body,
    validate_query_params,
    validate_debate_id,
    validate_agent_name,
    DEBATE_START_SCHEMA,
)


class MockHandler:
    """Mock handler class for testing decorators."""

    pass


class TestValidateRequestDecorator:
    """Tests for @validate_request decorator."""

    def test_no_validation_passes(self):
        """Handler without validation requirements passes through."""

        class Handler(MockHandler):
            @validate_request()
            def handle(self, path, query, handler):
                return {"success": True}

        h = Handler()
        result = h.handle("/api/test", {}, None)
        assert result == {"success": True}

    def test_required_param_missing(self):
        """Missing required parameter returns error."""

        class Handler(MockHandler):
            @validate_request(required_params=["task"])
            def handle(self, path, query, handler):
                return {"success": True}

        h = Handler()
        result = h.handle("/api/test", {}, None)
        assert result["status"] == 400
        assert "Missing required parameter: task" in result["error"]

    def test_required_param_present(self):
        """Present required parameter passes."""

        class Handler(MockHandler):
            @validate_request(required_params=["task"])
            def handle(self, path, query, handler):
                return {"success": True}

        h = Handler()
        result = h.handle("/api/test", {"task": "test"}, None)
        assert result == {"success": True}

    def test_required_param_empty_list(self):
        """Empty list for required param returns error."""

        class Handler(MockHandler):
            @validate_request(required_params=["agents"])
            def handle(self, path, query, handler):
                return {"success": True}

        h = Handler()
        result = h.handle("/api/test", {"agents": []}, None)
        assert result["status"] == 400

    def test_path_validator_valid(self):
        """Valid path segment passes validation."""

        class Handler(MockHandler):
            @validate_request(path_validators={"debate_id": validate_debate_id})
            def handle(self, path, query, handler):
                return {"success": True}

        h = Handler()
        result = h.handle("/api/debates/valid-id-123", {}, None)
        assert result == {"success": True}

    def test_path_validator_invalid(self):
        """Invalid path segment returns error."""

        class Handler(MockHandler):
            @validate_request(path_validators={"debate_id": validate_debate_id})
            def handle(self, path, query, handler):
                return {"success": True}

        h = Handler()
        result = h.handle("/api/debates/../../../etc/passwd", {}, None)
        assert result["status"] == 400

    def test_schema_validation_valid(self):
        """Valid body passes schema validation."""

        class Handler(MockHandler):
            @validate_request(schema=DEBATE_START_SCHEMA)
            def handle(self, path, query, body, handler):
                return {"success": True, "task": body["task"]}

        h = Handler()
        result = h.handle("/api/debates", {}, {"task": "Test debate task"}, None)
        assert result["success"] is True
        assert result["task"] == "Test debate task"

    def test_schema_validation_missing_required(self):
        """Missing required field in body returns error."""

        class Handler(MockHandler):
            @validate_request(schema=DEBATE_START_SCHEMA)
            def handle(self, path, query, body, handler):
                return {"success": True}

        h = Handler()
        result = h.handle("/api/debates", {}, {}, None)
        assert result["status"] == 400
        assert "task" in result["error"].lower()


class TestValidatePostBodyDecorator:
    """Tests for @validate_post_body decorator."""

    def test_valid_body_passes(self):
        """Valid body passes validation."""

        class Handler(MockHandler):
            @validate_post_body(DEBATE_START_SCHEMA)
            def handle(self, body, handler):
                return {"success": True}

        h = Handler()
        result = h.handle({"task": "Test task"}, None)
        assert result == {"success": True}

    def test_invalid_body_type_returns_error(self):
        """Non-dict body returns error."""

        class Handler(MockHandler):
            @validate_post_body(DEBATE_START_SCHEMA)
            def handle(self, body, handler):
                return {"success": True}

        h = Handler()
        result = h.handle("not a dict", None)
        assert result["status"] == 400
        assert "JSON object" in result["error"]

    def test_missing_required_field_returns_error(self):
        """Missing required field returns error."""

        class Handler(MockHandler):
            @validate_post_body(DEBATE_START_SCHEMA)
            def handle(self, body, handler):
                return {"success": True}

        h = Handler()
        result = h.handle({"rounds": 3}, None)  # Missing 'task'
        assert result["status"] == 400

    def test_field_too_long_returns_error(self):
        """Field exceeding max length returns error."""

        class Handler(MockHandler):
            @validate_post_body(DEBATE_START_SCHEMA)
            def handle(self, body, handler):
                return {"success": True}

        h = Handler()
        result = h.handle({"task": "x" * 3000}, None)  # Exceeds 2000 char limit
        assert result["status"] == 400


class TestValidateQueryParamsDecorator:
    """Tests for @validate_query_params decorator."""

    def test_required_params_present(self):
        """All required params present passes."""

        class Handler(MockHandler):
            @validate_query_params(required=["agent", "limit"])
            def handle(self, query, handler):
                return {"success": True}

        h = Handler()
        result = h.handle({"agent": "claude", "limit": "10"}, None)
        assert result == {"success": True}

    def test_required_param_missing(self):
        """Missing required param returns error."""

        class Handler(MockHandler):
            @validate_query_params(required=["agent"])
            def handle(self, query, handler):
                return {"success": True}

        h = Handler()
        result = h.handle({}, None)
        assert result["status"] == 400
        assert "agent" in result["error"]

    def test_int_param_in_bounds(self):
        """Int param within bounds passes."""

        class Handler(MockHandler):
            @validate_query_params(int_params={"limit": (10, 1, 100)})
            def handle(self, query, handler):
                return {"success": True}

        h = Handler()
        result = h.handle({"limit": "50"}, None)
        assert result == {"success": True}

    def test_int_param_out_of_bounds(self):
        """Int param out of bounds returns error."""

        class Handler(MockHandler):
            @validate_query_params(int_params={"limit": (10, 1, 100)})
            def handle(self, query, handler):
                return {"success": True}

        h = Handler()
        result = h.handle({"limit": "500"}, None)
        assert result["status"] == 400
        assert "limit" in result["error"]

    def test_int_param_invalid_type(self):
        """Non-integer int param returns error."""

        class Handler(MockHandler):
            @validate_query_params(int_params={"limit": (10, 1, 100)})
            def handle(self, query, handler):
                return {"success": True}

        h = Handler()
        result = h.handle({"limit": "not-a-number"}, None)
        assert result["status"] == 400
        assert "integer" in result["error"]

    def test_string_param_too_long(self):
        """String param exceeding max length returns error."""

        class Handler(MockHandler):
            @validate_query_params(string_params={"sort": ("created_at", 20)})
            def handle(self, query, handler):
                return {"success": True}

        h = Handler()
        result = h.handle({"sort": "x" * 30}, None)
        assert result["status"] == 400
        assert "exceeds" in result["error"]

    def test_string_param_within_length(self):
        """String param within length passes."""

        class Handler(MockHandler):
            @validate_query_params(string_params={"sort": ("created_at", 20)})
            def handle(self, query, handler):
                return {"success": True}

        h = Handler()
        result = h.handle({"sort": "name"}, None)
        assert result == {"success": True}


class TestValidationDecoratorCombinations:
    """Tests for combining multiple validation decorators."""

    def test_combined_validation(self):
        """Multiple validation constraints can be combined."""

        class Handler(MockHandler):
            @validate_request(required_params=["mode"])
            @validate_query_params(int_params={"limit": (10, 1, 100)})
            def handle(self, path, query, handler):
                return {"success": True}

        h = Handler()
        # Missing required param
        result = h.handle("/api/test", {"limit": "50"}, None)
        assert result["status"] == 400

        # Valid params
        result = h.handle("/api/test", {"mode": "test", "limit": "50"}, None)
        assert result == {"success": True}

    def test_decorator_preserves_function_name(self):
        """Decorator preserves the original function name."""

        class Handler(MockHandler):
            @validate_request()
            def my_handler_name(self, path, query, handler):
                return {"success": True}

        h = Handler()
        assert h.my_handler_name.__name__ == "my_handler_name"
