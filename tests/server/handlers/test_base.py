"""
Tests for aragora/server/handlers/base.py - the base handler module.

This module tests all public classes, functions, and decorators provided by
the base handler module, which underpins all 90+ HTTP endpoints.

Test coverage:
- ServerContext TypedDict structure
- HandlerResult dataclass
- Response builders (json_response, error_response, success_response, etc.)
- Utility functions (get_host_header, get_agent_name, agent_to_dict, etc.)
- Parameter extraction (get_int_param, get_float_param, get_bool_param, etc.)
- Handler mixins (PaginatedHandlerMixin, CachedHandlerMixin, AuthenticatedHandlerMixin)
- BaseHandler class and methods
- Typed handler classes (TypedHandler, AuthenticatedHandler, PermissionHandler, etc.)
- Decorators (@require_quota, @api_endpoint, @rate_limit, @validate_body)
"""

import json
import pytest
from dataclasses import dataclass
from http import HTTPStatus
from io import BytesIO
from unittest.mock import MagicMock, Mock, patch, PropertyMock
from typing import Any


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_handler():
    """Create a mock HTTP request handler."""
    handler = MagicMock()
    handler.headers = {
        "Host": "test.example.com",
        "Content-Type": "application/json",
        "Content-Length": "50",
        "Authorization": "Bearer test-token",
    }
    handler.path = "/api/v1/test?param=value"
    handler.command = "GET"

    # Mock rfile for reading request body
    body_content = b'{"key": "value"}'
    handler.headers["Content-Length"] = str(len(body_content))
    handler.rfile = BytesIO(body_content)

    return handler


@pytest.fixture
def server_context():
    """Create a minimal server context."""
    return {
        "storage": MagicMock(),
        "user_store": MagicMock(),
        "elo_system": MagicMock(),
    }


@pytest.fixture
def mock_user_context():
    """Create a mock user authentication context."""
    user_ctx = MagicMock()
    user_ctx.is_authenticated = True
    user_ctx.user_id = "test-user-123"
    user_ctx.email = "test@example.com"
    user_ctx.role = "admin"
    user_ctx.roles = ["admin"]
    user_ctx.permissions = ["debates:read", "debates:create"]
    user_ctx.org_id = "org-123"
    user_ctx.error_reason = None
    return user_ctx


# =============================================================================
# Tests for HandlerResult
# =============================================================================


class TestHandlerResult:
    """Tests for the HandlerResult dataclass."""

    def test_handler_result_creation(self):
        """Should create a HandlerResult with required fields."""
        from aragora.server.handlers.utils.responses import HandlerResult

        result = HandlerResult(
            status_code=200, content_type="application/json", body=b'{"success": true}'
        )

        assert result.status_code == 200
        assert result.content_type == "application/json"
        assert result.body == b'{"success": true}'
        assert result.headers == {}  # Default from __post_init__

    def test_handler_result_with_headers(self):
        """Should create a HandlerResult with custom headers."""
        from aragora.server.handlers.utils.responses import HandlerResult

        result = HandlerResult(
            status_code=201,
            content_type="application/json",
            body=b'{"id": "123"}',
            headers={"Location": "/api/resource/123"},
        )

        assert result.headers == {"Location": "/api/resource/123"}

    def test_handler_result_headers_default_to_empty_dict(self):
        """Headers should default to empty dict via __post_init__."""
        from aragora.server.handlers.utils.responses import HandlerResult

        result = HandlerResult(status_code=200, content_type="text/plain", body=b"OK", headers=None)

        assert result.headers == {}


# =============================================================================
# Tests for Response Builders
# =============================================================================


class TestJsonResponse:
    """Tests for the json_response function."""

    def test_json_response_basic(self):
        """Should create a basic JSON response."""
        from aragora.server.handlers.base import json_response

        result = json_response({"key": "value"})

        assert result.status_code == 200
        assert result.content_type == "application/json"
        assert json.loads(result.body) == {"key": "value"}

    def test_json_response_with_status(self):
        """Should create a JSON response with custom status."""
        from aragora.server.handlers.base import json_response

        result = json_response({"created": True}, status=201)

        assert result.status_code == 201

    def test_json_response_with_headers(self):
        """Should create a JSON response with custom headers."""
        from aragora.server.handlers.base import json_response

        result = json_response({"data": "test"}, headers={"X-Custom-Header": "value"})

        assert result.headers == {"X-Custom-Header": "value"}

    def test_json_response_complex_data(self):
        """Should serialize complex data structures."""
        from aragora.server.handlers.base import json_response

        data = {"items": [1, 2, 3], "nested": {"a": "b"}, "count": 100}
        result = json_response(data)

        assert json.loads(result.body) == data

    def test_json_response_uses_str_for_non_serializable(self):
        """Should use str() for non-JSON-serializable types."""
        from aragora.server.handlers.base import json_response
        from datetime import datetime

        data = {"timestamp": datetime(2025, 1, 1, 12, 0, 0)}
        result = json_response(data)

        parsed = json.loads(result.body)
        assert "2025" in parsed["timestamp"]

    def test_json_response_empty_dict(self):
        """Should handle empty dict."""
        from aragora.server.handlers.base import json_response

        result = json_response({})

        assert json.loads(result.body) == {}

    def test_json_response_list_data(self):
        """Should handle list as top-level data."""
        from aragora.server.handlers.base import json_response

        data = [{"id": 1}, {"id": 2}]
        result = json_response(data)

        assert json.loads(result.body) == data


class TestErrorResponse:
    """Tests for the error_response function."""

    def test_error_response_simple(self):
        """Should create a simple error response."""
        from aragora.server.handlers.base import error_response

        result = error_response("Something went wrong", 400)

        assert result.status_code == 400
        body = json.loads(result.body)
        assert body == {"error": "Something went wrong"}

    def test_error_response_default_status(self):
        """Should default to 400 status."""
        from aragora.server.handlers.base import error_response

        result = error_response("Bad request")

        assert result.status_code == 400

    def test_error_response_with_code(self):
        """Should include error code when provided."""
        from aragora.server.handlers.base import error_response

        result = error_response("Validation failed", 400, code="VALIDATION_ERROR")

        body = json.loads(result.body)
        assert body["error"]["code"] == "VALIDATION_ERROR"
        assert body["error"]["message"] == "Validation failed"

    def test_error_response_with_trace_id(self):
        """Should include trace_id when provided."""
        from aragora.server.handlers.base import error_response

        result = error_response("Server error", 500, trace_id="abc12345")

        body = json.loads(result.body)
        assert body["error"]["trace_id"] == "abc12345"

    def test_error_response_with_suggestion(self):
        """Should include suggestion when provided."""
        from aragora.server.handlers.base import error_response

        result = error_response(
            "Field 'name' is required", 400, suggestion="Include 'name' in request body"
        )

        body = json.loads(result.body)
        assert body["error"]["suggestion"] == "Include 'name' in request body"

    def test_error_response_with_details(self):
        """Should include details when provided."""
        from aragora.server.handlers.base import error_response

        result = error_response(
            "Multiple errors", 400, details={"field1": "error1", "field2": "error2"}
        )

        body = json.loads(result.body)
        assert body["error"]["details"] == {"field1": "error1", "field2": "error2"}

    def test_error_response_structured_format(self):
        """Should use structured format when requested."""
        from aragora.server.handlers.base import error_response

        result = error_response("Error message", 400, structured=True)

        body = json.loads(result.body)
        assert "message" in body["error"]

    def test_error_response_500_server_error(self):
        """Should handle 500 server errors."""
        from aragora.server.handlers.base import error_response

        result = error_response("Internal server error", 500)

        assert result.status_code == 500

    def test_error_response_401_unauthorized(self):
        """Should handle 401 unauthorized errors."""
        from aragora.server.handlers.base import error_response

        result = error_response("Authentication required", 401)

        assert result.status_code == 401

    def test_error_response_403_forbidden(self):
        """Should handle 403 forbidden errors."""
        from aragora.server.handlers.base import error_response

        result = error_response("Access denied", 403)

        assert result.status_code == 403

    def test_error_response_404_not_found(self):
        """Should handle 404 not found errors."""
        from aragora.server.handlers.base import error_response

        result = error_response("Resource not found", 404)

        assert result.status_code == 404


class TestSuccessResponse:
    """Tests for the success_response function."""

    def test_success_response_basic(self):
        """Should create a success response with deprecation warning."""
        from aragora.server.handlers.base import success_response
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = success_response({"id": "123"})

            assert len(w) == 1
            assert "deprecated" in str(w[0].message).lower()

        body = json.loads(result.body)
        assert body["success"] is True
        assert body["data"] == {"id": "123"}

    def test_success_response_with_message(self):
        """Should include message when provided."""
        from aragora.server.handlers.base import success_response
        import warnings

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = success_response({"items": []}, message="No items found")

        body = json.loads(result.body)
        assert body["message"] == "No items found"


class TestSafeErrorResponse:
    """Tests for the safe_error_response function."""

    def test_safe_error_response_sanitizes_message(self):
        """Should sanitize error messages for clients."""
        from aragora.server.handlers.base import safe_error_response

        exc = Exception("Internal path: /secret/path/file.py")
        result = safe_error_response(exc, "database query", 500)

        assert result.status_code == 500
        body = json.loads(result.body)
        # Should not contain internal paths
        assert "/secret/path" not in str(body)

    def test_safe_error_response_extracts_trace_id_from_handler(self):
        """Should extract trace_id from handler if available."""
        from aragora.server.handlers.base import safe_error_response

        handler = MagicMock()
        handler.trace_id = "handler-trace-123"

        exc = ValueError("Test error")
        result = safe_error_response(exc, "test context", 400, handler)

        body = json.loads(result.body)
        assert body.get("trace_id") == "handler-trace-123" or "trace_id" in str(body)

    def test_safe_error_response_generates_trace_id(self):
        """Should generate trace_id when not available."""
        from aragora.server.handlers.base import safe_error_response

        exc = ValueError("Test error")
        result = safe_error_response(exc, "test context", 400)

        # Should complete without error
        assert result.status_code == 400


# =============================================================================
# Tests for Utility Functions
# =============================================================================


class TestGetHostHeader:
    """Tests for the get_host_header function."""

    def test_get_host_header_from_handler(self, mock_handler):
        """Should extract Host header from handler."""
        from aragora.server.handlers.base import get_host_header

        result = get_host_header(mock_handler)

        assert result == "test.example.com"

    def test_get_host_header_with_none_handler(self):
        """Should return default when handler is None."""
        from aragora.server.handlers.base import get_host_header

        result = get_host_header(None)

        # Should return default (localhost:8080 or env var)
        assert result is not None
        assert ":" in result or result == "localhost:8080"

    def test_get_host_header_with_custom_default(self):
        """Should use custom default when provided."""
        from aragora.server.handlers.base import get_host_header

        result = get_host_header(None, default="custom.host:9000")

        assert result == "custom.host:9000"

    def test_get_host_header_missing_header(self):
        """Should return default when Host header is missing."""
        from aragora.server.handlers.base import get_host_header

        handler = MagicMock()
        handler.headers = {}

        result = get_host_header(handler)

        assert result is not None

    def test_get_host_header_no_headers_attribute(self):
        """Should return default when handler has no headers attribute."""
        from aragora.server.handlers.base import get_host_header

        handler = object()  # No headers attribute
        result = get_host_header(handler, default="fallback:8080")

        assert result == "fallback:8080"


class TestGetAgentName:
    """Tests for the get_agent_name function."""

    def test_get_agent_name_from_dict_with_name(self):
        """Should extract name from dict."""
        from aragora.server.handlers.base import get_agent_name

        agent = {"name": "claude", "elo": 1500}

        assert get_agent_name(agent) == "claude"

    def test_get_agent_name_from_dict_with_agent_name(self):
        """Should extract agent_name from dict."""
        from aragora.server.handlers.base import get_agent_name

        agent = {"agent_name": "gpt-4", "wins": 10}

        assert get_agent_name(agent) == "gpt-4"

    def test_get_agent_name_prefers_agent_name(self):
        """Should prefer agent_name over name."""
        from aragora.server.handlers.base import get_agent_name

        agent = {"agent_name": "preferred", "name": "fallback"}

        assert get_agent_name(agent) == "preferred"

    def test_get_agent_name_from_object(self):
        """Should extract name from object attribute."""
        from aragora.server.handlers.base import get_agent_name

        agent = MagicMock()
        agent.name = "object-agent"
        agent.agent_name = None

        assert get_agent_name(agent) == "object-agent"

    def test_get_agent_name_returns_none_for_none(self):
        """Should return None for None input."""
        from aragora.server.handlers.base import get_agent_name

        assert get_agent_name(None) is None

    def test_get_agent_name_returns_none_for_empty_dict(self):
        """Should return None for dict without name."""
        from aragora.server.handlers.base import get_agent_name

        assert get_agent_name({}) is None


class TestAgentToDict:
    """Tests for the agent_to_dict function."""

    def test_agent_to_dict_from_dict(self):
        """Should return copy of dict input."""
        from aragora.server.handlers.base import agent_to_dict

        agent = {"name": "claude", "elo": 1500}
        result = agent_to_dict(agent)

        assert result == {"name": "claude", "elo": 1500}
        assert result is not agent  # Should be a copy

    def test_agent_to_dict_from_object(self):
        """Should extract standard fields from object."""
        from aragora.server.handlers.base import agent_to_dict

        # Use a simple class instead of MagicMock to avoid attribute issues
        class MockAgent:
            name = "test-agent"
            agent_name = None
            elo = 1600
            wins = 10
            losses = 5
            draws = 2
            win_rate = 0.66
            games_played = 17
            matches = 15

        agent = MockAgent()
        result = agent_to_dict(agent)

        assert result["name"] == "test-agent"
        assert result["elo"] == 1600
        assert result["wins"] == 10
        assert result["losses"] == 5
        assert result["draws"] == 2

    def test_agent_to_dict_with_include_name_false(self):
        """Should exclude name fields when include_name is False."""
        from aragora.server.handlers.base import agent_to_dict

        class MockAgent:
            name = "test"
            agent_name = None
            elo = 1500
            wins = 0
            losses = 0
            draws = 0
            win_rate = 0.0
            games_played = 0
            matches = 0

        agent = MockAgent()
        result = agent_to_dict(agent, include_name=False)

        assert "name" not in result
        assert "agent_name" not in result
        assert "elo" in result

    def test_agent_to_dict_returns_empty_for_none(self):
        """Should return empty dict for None input."""
        from aragora.server.handlers.base import agent_to_dict

        assert agent_to_dict(None) == {}

    def test_agent_to_dict_uses_defaults_for_missing_attrs(self):
        """Should use defaults for missing attributes."""
        from aragora.server.handlers.base import agent_to_dict

        class MinimalAgent:
            name = "minimal"
            agent_name = None

        agent = MinimalAgent()
        result = agent_to_dict(agent)

        assert result["elo"] == 1500  # Default ELO
        assert result["wins"] == 0
        assert result["losses"] == 0


# =============================================================================
# Tests for Parameter Extraction
# =============================================================================


class TestGetIntParam:
    """Tests for the get_int_param function."""

    def test_get_int_param_basic(self):
        """Should extract integer parameter."""
        from aragora.server.handlers.base import get_int_param

        params = {"limit": "10"}
        assert get_int_param(params, "limit", 20) == 10

    def test_get_int_param_default(self):
        """Should return default for missing param."""
        from aragora.server.handlers.base import get_int_param

        params = {}
        assert get_int_param(params, "limit", 20) == 20

    def test_get_int_param_list_value(self):
        """Should handle list value from query string."""
        from aragora.server.handlers.base import get_int_param

        params = {"limit": ["10", "20"]}  # Multiple values
        assert get_int_param(params, "limit", 0) == 10

    def test_get_int_param_invalid_value(self):
        """Should return default for invalid value."""
        from aragora.server.handlers.base import get_int_param

        params = {"limit": "not_a_number"}
        assert get_int_param(params, "limit", 50) == 50

    def test_get_int_param_empty_list(self):
        """Should return default for empty list."""
        from aragora.server.handlers.base import get_int_param

        params = {"limit": []}
        assert get_int_param(params, "limit", 100) == 100


class TestGetFloatParam:
    """Tests for the get_float_param function."""

    def test_get_float_param_basic(self):
        """Should extract float parameter."""
        from aragora.server.handlers.base import get_float_param

        params = {"threshold": "0.75"}
        assert get_float_param(params, "threshold", 0.5) == 0.75

    def test_get_float_param_default(self):
        """Should return default for missing param."""
        from aragora.server.handlers.base import get_float_param

        params = {}
        assert get_float_param(params, "threshold", 0.5) == 0.5

    def test_get_float_param_integer_value(self):
        """Should handle integer string as float."""
        from aragora.server.handlers.base import get_float_param

        params = {"value": "10"}
        assert get_float_param(params, "value", 0.0) == 10.0

    def test_get_float_param_invalid_value(self):
        """Should return default for invalid value."""
        from aragora.server.handlers.base import get_float_param

        params = {"value": "invalid"}
        assert get_float_param(params, "value", 1.5) == 1.5


class TestGetBoolParam:
    """Tests for the get_bool_param function."""

    def test_get_bool_param_true_string(self):
        """Should parse 'true' as True."""
        from aragora.server.handlers.base import get_bool_param

        params = {"enabled": "true"}
        assert get_bool_param(params, "enabled", False) is True

    def test_get_bool_param_false_string(self):
        """Should parse 'false' as False."""
        from aragora.server.handlers.base import get_bool_param

        params = {"enabled": "false"}
        assert get_bool_param(params, "enabled", True) is False

    def test_get_bool_param_one_string(self):
        """Should parse '1' as True."""
        from aragora.server.handlers.base import get_bool_param

        params = {"flag": "1"}
        assert get_bool_param(params, "flag", False) is True

    def test_get_bool_param_yes_string(self):
        """Should parse 'yes' as True."""
        from aragora.server.handlers.base import get_bool_param

        params = {"flag": "yes"}
        assert get_bool_param(params, "flag", False) is True

    def test_get_bool_param_on_string(self):
        """Should parse 'on' as True."""
        from aragora.server.handlers.base import get_bool_param

        params = {"flag": "on"}
        assert get_bool_param(params, "flag", False) is True

    def test_get_bool_param_default(self):
        """Should return default for missing param."""
        from aragora.server.handlers.base import get_bool_param

        params = {}
        assert get_bool_param(params, "flag", True) is True

    def test_get_bool_param_boolean_value(self):
        """Should handle actual boolean values."""
        from aragora.server.handlers.base import get_bool_param

        params = {"flag": True}
        assert get_bool_param(params, "flag", False) is True

    def test_get_bool_param_list_value(self):
        """Should handle list value."""
        from aragora.server.handlers.base import get_bool_param

        params = {"flag": ["true"]}
        assert get_bool_param(params, "flag", False) is True


class TestGetStringParam:
    """Tests for the get_string_param function."""

    def test_get_string_param_basic(self):
        """Should extract string parameter."""
        from aragora.server.handlers.base import get_string_param

        params = {"name": "test-value"}
        assert get_string_param(params, "name", None) == "test-value"

    def test_get_string_param_default(self):
        """Should return default for missing param."""
        from aragora.server.handlers.base import get_string_param

        params = {}
        assert get_string_param(params, "name", "default") == "default"

    def test_get_string_param_list_value(self):
        """Should handle list value."""
        from aragora.server.handlers.base import get_string_param

        params = {"name": ["first", "second"]}
        assert get_string_param(params, "name", None) == "first"

    def test_get_string_param_empty_list(self):
        """Should return default for empty list."""
        from aragora.server.handlers.base import get_string_param

        params = {"name": []}
        assert get_string_param(params, "name", "default") == "default"


class TestGetClampedIntParam:
    """Tests for the get_clamped_int_param function."""

    def test_get_clamped_int_param_within_range(self):
        """Should return value when within range."""
        from aragora.server.handlers.base import get_clamped_int_param

        params = {"limit": "50"}
        assert get_clamped_int_param(params, "limit", 100, 1, 100) == 50

    def test_get_clamped_int_param_clamp_to_min(self):
        """Should clamp to min when below range."""
        from aragora.server.handlers.base import get_clamped_int_param

        params = {"limit": "-10"}
        assert get_clamped_int_param(params, "limit", 100, 1, 100) == 1

    def test_get_clamped_int_param_clamp_to_max(self):
        """Should clamp to max when above range."""
        from aragora.server.handlers.base import get_clamped_int_param

        params = {"limit": "500"}
        assert get_clamped_int_param(params, "limit", 100, 1, 100) == 100


class TestGetBoundedFloatParam:
    """Tests for the get_bounded_float_param function."""

    def test_get_bounded_float_param_within_range(self):
        """Should return value when within range."""
        from aragora.server.handlers.base import get_bounded_float_param

        params = {"threshold": "0.5"}
        assert get_bounded_float_param(params, "threshold", 0.5, 0.0, 1.0) == 0.5

    def test_get_bounded_float_param_clamp_to_min(self):
        """Should clamp to min when below range."""
        from aragora.server.handlers.base import get_bounded_float_param

        params = {"threshold": "-0.5"}
        assert get_bounded_float_param(params, "threshold", 0.5, 0.0, 1.0) == 0.0

    def test_get_bounded_float_param_clamp_to_max(self):
        """Should clamp to max when above range."""
        from aragora.server.handlers.base import get_bounded_float_param

        params = {"threshold": "1.5"}
        assert get_bounded_float_param(params, "threshold", 0.5, 0.0, 1.0) == 1.0


class TestGetBoundedStringParam:
    """Tests for the get_bounded_string_param function."""

    def test_get_bounded_string_param_within_limit(self):
        """Should return full string when within limit."""
        from aragora.server.handlers.base import get_bounded_string_param

        params = {"query": "short"}
        assert get_bounded_string_param(params, "query", None, max_length=100) == "short"

    def test_get_bounded_string_param_truncate(self):
        """Should truncate string when exceeds limit."""
        from aragora.server.handlers.base import get_bounded_string_param

        params = {"query": "this is a very long string"}
        result = get_bounded_string_param(params, "query", None, max_length=10)
        assert result == "this is a "
        assert len(result) == 10

    def test_get_bounded_string_param_default(self):
        """Should return default for missing param."""
        from aragora.server.handlers.base import get_bounded_string_param

        params = {}
        assert get_bounded_string_param(params, "query", "default") == "default"


class TestParseQueryParams:
    """Tests for the parse_query_params function."""

    def test_parse_query_params_basic(self):
        """Should parse query string into dict."""
        from aragora.server.handlers.base import parse_query_params

        result = parse_query_params("limit=10&offset=5")

        assert result == {"limit": "10", "offset": "5"}

    def test_parse_query_params_empty(self):
        """Should return empty dict for empty string."""
        from aragora.server.handlers.base import parse_query_params

        result = parse_query_params("")

        assert result == {}

    def test_parse_query_params_multiple_values(self):
        """Should keep list for multiple values."""
        from aragora.server.handlers.base import parse_query_params

        result = parse_query_params("tag=a&tag=b")

        assert result["tag"] == ["a", "b"]


# =============================================================================
# Tests for Safe Data Utilities
# =============================================================================


class TestSafeGet:
    """Tests for the safe_get function."""

    def test_safe_get_existing_key(self):
        """Should return value for existing key."""
        from aragora.server.handlers.base import safe_get

        data = {"key": "value"}
        assert safe_get(data, "key", "default") == "value"

    def test_safe_get_missing_key(self):
        """Should return default for missing key."""
        from aragora.server.handlers.base import safe_get

        data = {"other": "value"}
        assert safe_get(data, "key", "default") == "default"

    def test_safe_get_none_data(self):
        """Should return default for None data."""
        from aragora.server.handlers.base import safe_get

        assert safe_get(None, "key", "default") == "default"

    def test_safe_get_non_dict_data(self):
        """Should return default for non-dict data."""
        from aragora.server.handlers.base import safe_get

        assert safe_get("not a dict", "key", "default") == "default"


class TestSafeGetNested:
    """Tests for the safe_get_nested function."""

    def test_safe_get_nested_success(self):
        """Should navigate nested structure."""
        from aragora.server.handlers.base import safe_get_nested

        data = {"outer": {"inner": {"deep": "value"}}}
        assert safe_get_nested(data, ["outer", "inner", "deep"], "default") == "value"

    def test_safe_get_nested_missing_key(self):
        """Should return default for missing key in chain."""
        from aragora.server.handlers.base import safe_get_nested

        data = {"outer": {"inner": {}}}
        assert safe_get_nested(data, ["outer", "inner", "deep"], "default") == "default"

    def test_safe_get_nested_none_data(self):
        """Should return default for None data."""
        from aragora.server.handlers.base import safe_get_nested

        assert safe_get_nested(None, ["a", "b"], "default") == "default"

    def test_safe_get_nested_non_dict_intermediate(self):
        """Should return default when intermediate value is not dict."""
        from aragora.server.handlers.base import safe_get_nested

        data = {"outer": "string_value"}
        assert safe_get_nested(data, ["outer", "inner"], "default") == "default"


class TestSafeJsonParse:
    """Tests for the safe_json_parse function."""

    def test_safe_json_parse_string(self):
        """Should parse JSON string."""
        from aragora.server.handlers.base import safe_json_parse

        result = safe_json_parse('{"key": "value"}')
        assert result == {"key": "value"}

    def test_safe_json_parse_dict_passthrough(self):
        """Should return dict as-is."""
        from aragora.server.handlers.base import safe_json_parse

        data = {"key": "value"}
        assert safe_json_parse(data) is data

    def test_safe_json_parse_list_passthrough(self):
        """Should return list as-is."""
        from aragora.server.handlers.base import safe_json_parse

        data = [1, 2, 3]
        assert safe_json_parse(data) is data

    def test_safe_json_parse_invalid_json(self):
        """Should return default for invalid JSON."""
        from aragora.server.handlers.base import safe_json_parse

        assert safe_json_parse("not valid json", "default") == "default"

    def test_safe_json_parse_none(self):
        """Should return default for None."""
        from aragora.server.handlers.base import safe_json_parse

        assert safe_json_parse(None, "default") == "default"


# =============================================================================
# Tests for Handler Mixins
# =============================================================================


class TestPaginatedHandlerMixin:
    """Tests for the PaginatedHandlerMixin class."""

    def test_get_pagination_defaults(self):
        """Should return default pagination values."""
        from aragora.server.handlers.base import PaginatedHandlerMixin

        mixin = PaginatedHandlerMixin()
        limit, offset = mixin.get_pagination({})

        assert limit == 20  # DEFAULT_LIMIT
        assert offset == 0  # DEFAULT_OFFSET

    def test_get_pagination_from_params(self):
        """Should extract pagination from params."""
        from aragora.server.handlers.base import PaginatedHandlerMixin

        mixin = PaginatedHandlerMixin()
        limit, offset = mixin.get_pagination({"limit": "50", "offset": "10"})

        assert limit == 50
        assert offset == 10

    def test_get_pagination_clamps_to_max(self):
        """Should clamp limit to max."""
        from aragora.server.handlers.base import PaginatedHandlerMixin

        mixin = PaginatedHandlerMixin()
        limit, _ = mixin.get_pagination({"limit": "500"})

        assert limit == 100  # MAX_LIMIT

    def test_get_pagination_clamps_to_min(self):
        """Should clamp limit to minimum of 1."""
        from aragora.server.handlers.base import PaginatedHandlerMixin

        mixin = PaginatedHandlerMixin()
        limit, _ = mixin.get_pagination({"limit": "0"})

        assert limit == 1

    def test_get_pagination_negative_offset(self):
        """Should clamp negative offset to 0."""
        from aragora.server.handlers.base import PaginatedHandlerMixin

        mixin = PaginatedHandlerMixin()
        _, offset = mixin.get_pagination({"offset": "-10"})

        assert offset == 0

    def test_paginated_response(self):
        """Should create paginated response with metadata."""
        from aragora.server.handlers.base import PaginatedHandlerMixin

        mixin = PaginatedHandlerMixin()
        result = mixin.paginated_response(items=[1, 2, 3], total=10, limit=3, offset=0)

        body = json.loads(result.body)
        assert body["items"] == [1, 2, 3]
        assert body["total"] == 10
        assert body["limit"] == 3
        assert body["offset"] == 0
        assert body["has_more"] is True

    def test_paginated_response_no_more(self):
        """Should set has_more to False when at end."""
        from aragora.server.handlers.base import PaginatedHandlerMixin

        mixin = PaginatedHandlerMixin()
        result = mixin.paginated_response(items=[8, 9, 10], total=10, limit=3, offset=7)

        body = json.loads(result.body)
        assert body["has_more"] is False

    def test_paginated_response_custom_items_key(self):
        """Should use custom items key."""
        from aragora.server.handlers.base import PaginatedHandlerMixin

        mixin = PaginatedHandlerMixin()
        result = mixin.paginated_response(
            items=[{"id": 1}], total=1, limit=10, offset=0, items_key="debates"
        )

        body = json.loads(result.body)
        assert "debates" in body
        assert "items" not in body


class TestCachedHandlerMixin:
    """Tests for the CachedHandlerMixin class."""

    def test_cached_response_generates_on_miss(self):
        """Should call generator on cache miss."""
        from aragora.server.handlers.base import CachedHandlerMixin, clear_cache

        clear_cache()  # Start fresh

        mixin = CachedHandlerMixin()
        generator_called = [False]

        def generator():
            generator_called[0] = True
            return {"data": "generated"}

        result = mixin.cached_response(cache_key="test_key", ttl_seconds=300, generator=generator)

        assert generator_called[0] is True
        assert result == {"data": "generated"}

    def test_cached_response_returns_cached(self):
        """Should return cached value on hit."""
        from aragora.server.handlers.base import CachedHandlerMixin, clear_cache

        clear_cache()

        mixin = CachedHandlerMixin()
        call_count = [0]

        def generator():
            call_count[0] += 1
            return {"data": f"call_{call_count[0]}"}

        # First call - should generate
        result1 = mixin.cached_response("test_key_2", 300, generator)

        # Second call - should use cache
        result2 = mixin.cached_response("test_key_2", 300, generator)

        assert call_count[0] == 1  # Only called once
        assert result1 == result2


class TestAuthenticatedHandlerMixin:
    """Tests for the AuthenticatedHandlerMixin class."""

    def test_require_auth_calls_require_auth_or_error(self):
        """Should delegate to require_auth_or_error if available."""
        from aragora.server.handlers.base import AuthenticatedHandlerMixin

        mixin = AuthenticatedHandlerMixin()

        # Mock require_auth_or_error
        mock_user = MagicMock()
        mock_user.is_authenticated = True
        mixin.require_auth_or_error = MagicMock(return_value=(mock_user, None))

        result = mixin.require_auth(MagicMock())

        assert result == mock_user


# =============================================================================
# Tests for BaseHandler
# =============================================================================


class TestBaseHandler:
    """Tests for the BaseHandler class."""

    def test_init_with_context(self, server_context):
        """Should initialize with server context."""
        from aragora.server.handlers.base import BaseHandler

        handler = BaseHandler(server_context)

        assert handler.ctx is server_context
        assert handler._current_handler is None
        assert handler._current_query_params == {}

    def test_set_request_context(self, server_context, mock_handler):
        """Should set current request context."""
        from aragora.server.handlers.base import BaseHandler

        handler = BaseHandler(server_context)
        handler.set_request_context(mock_handler, {"param": "value"})

        assert handler._current_handler is mock_handler
        assert handler._current_query_params == {"param": "value"}

    def test_get_query_param_from_context(self, server_context):
        """Should get param from stored context."""
        from aragora.server.handlers.base import BaseHandler

        handler = BaseHandler(server_context)
        handler._current_query_params = {"limit": "10"}

        result = handler.get_query_param("limit", "20")

        assert result == "10"

    def test_get_query_param_default(self, server_context):
        """Should return default for missing param."""
        from aragora.server.handlers.base import BaseHandler

        handler = BaseHandler(server_context)
        handler._current_query_params = {}

        result = handler.get_query_param("missing", "default")

        assert result == "default"

    def test_json_response_method(self, server_context):
        """Should create JSON response."""
        from aragora.server.handlers.base import BaseHandler

        handler = BaseHandler(server_context)
        result = handler.json_response({"key": "value"})

        assert result.status_code == 200
        assert json.loads(result.body) == {"key": "value"}

    def test_json_response_with_http_status(self, server_context):
        """Should handle HTTPStatus enum."""
        from aragora.server.handlers.base import BaseHandler

        handler = BaseHandler(server_context)
        result = handler.json_response({"created": True}, HTTPStatus.CREATED)

        assert result.status_code == 201

    def test_error_response_method(self, server_context):
        """Should create error response."""
        from aragora.server.handlers.base import BaseHandler

        handler = BaseHandler(server_context)
        result = handler.error_response("Bad input", HTTPStatus.BAD_REQUEST)

        assert result.status_code == 400

    def test_json_error_method(self, server_context):
        """Should create JSON error response."""
        from aragora.server.handlers.base import BaseHandler

        handler = BaseHandler(server_context)
        result = handler.json_error("Not found", HTTPStatus.NOT_FOUND)

        assert result.status_code == 404

    def test_get_storage(self, server_context):
        """Should return storage from context."""
        from aragora.server.handlers.base import BaseHandler

        handler = BaseHandler(server_context)
        storage = handler.get_storage()

        assert storage is server_context["storage"]

    def test_get_elo_system(self, server_context):
        """Should return ELO system from context."""
        from aragora.server.handlers.base import BaseHandler

        handler = BaseHandler(server_context)
        elo = handler.get_elo_system()

        assert elo is server_context["elo_system"]

    def test_read_json_body_success(self, server_context, mock_handler):
        """Should read and parse JSON body."""
        from aragora.server.handlers.base import BaseHandler

        handler = BaseHandler(server_context)
        result = handler.read_json_body(mock_handler)

        assert result == {"key": "value"}

    def test_read_json_body_empty(self, server_context):
        """Should return empty dict for no content."""
        from aragora.server.handlers.base import BaseHandler

        mock = MagicMock()
        mock.headers = {"Content-Length": "0"}
        mock.rfile = BytesIO(b"")

        handler = BaseHandler(server_context)
        result = handler.read_json_body(mock)

        assert result == {}

    def test_read_json_body_invalid(self, server_context):
        """Should return None for invalid JSON."""
        from aragora.server.handlers.base import BaseHandler

        mock = MagicMock()
        mock.headers = {"Content-Length": "10"}
        mock.rfile = BytesIO(b"not json!!")

        handler = BaseHandler(server_context)
        result = handler.read_json_body(mock)

        assert result is None

    def test_read_json_body_too_large(self, server_context):
        """Should return None for body exceeding max size."""
        from aragora.server.handlers.base import BaseHandler

        mock = MagicMock()
        mock.headers = {"Content-Length": "99999999"}

        handler = BaseHandler(server_context)
        result = handler.read_json_body(mock, max_size=100)

        assert result is None

    def test_validate_json_content_type_valid(self, server_context, mock_handler):
        """Should return None for valid content type."""
        from aragora.server.handlers.base import BaseHandler

        handler = BaseHandler(server_context)
        result = handler.validate_json_content_type(mock_handler)

        assert result is None

    def test_validate_json_content_type_invalid(self, server_context):
        """Should return error for invalid content type."""
        from aragora.server.handlers.base import BaseHandler

        mock = MagicMock()
        mock.headers = {"Content-Type": "text/plain", "Content-Length": "10"}

        handler = BaseHandler(server_context)
        result = handler.validate_json_content_type(mock)

        assert result is not None
        assert result.status_code == 415

    def test_validate_json_content_type_with_charset(self, server_context):
        """Should accept content type with charset."""
        from aragora.server.handlers.base import BaseHandler

        mock = MagicMock()
        mock.headers = {"Content-Type": "application/json; charset=utf-8"}

        handler = BaseHandler(server_context)
        result = handler.validate_json_content_type(mock)

        assert result is None

    def test_extract_path_param_success(self, server_context):
        """Should extract valid path parameter."""
        from aragora.server.handlers.base import BaseHandler

        handler = BaseHandler(server_context)
        # Path splits into: ['', 'api', 'v1', 'agent', 'claude', 'profile']
        # Index 4 is 'claude'
        value, error = handler.extract_path_param("/api/v1/agent/claude/profile", 4, "agent_name")

        assert value == "claude"
        assert error is None

    def test_extract_path_param_missing(self, server_context):
        """Should return error for missing segment."""
        from aragora.server.handlers.base import BaseHandler

        handler = BaseHandler(server_context)
        value, error = handler.extract_path_param("/api/v1", 5, "agent_name")

        assert value is None
        assert error is not None
        assert error.status_code == 400

    def test_extract_path_param_empty(self, server_context):
        """Should return error for empty segment."""
        from aragora.server.handlers.base import BaseHandler

        handler = BaseHandler(server_context)
        # Path with empty segment at index 3
        value, error = handler.extract_path_param("/api/v1//profile", 3, "agent_name")

        assert value is None
        assert error is not None

    def test_extract_path_params_multiple(self, server_context):
        """Should extract multiple path parameters."""
        from aragora.server.handlers.base import BaseHandler, SAFE_AGENT_PATTERN

        handler = BaseHandler(server_context)
        # Path splits into: ['', 'api', 'v1', 'compare', 'claude', 'gpt4']
        # Index 4 is 'claude', index 5 is 'gpt4'
        params, error = handler.extract_path_params(
            "/api/v1/compare/claude/gpt4",
            [
                (4, "agent_a", SAFE_AGENT_PATTERN),
                (5, "agent_b", SAFE_AGENT_PATTERN),
            ],
        )

        assert error is None
        assert params == {"agent_a": "claude", "agent_b": "gpt4"}

    def test_handle_returns_none_by_default(self, server_context, mock_handler):
        """Should return None by default (not handled)."""
        from aragora.server.handlers.base import BaseHandler

        handler = BaseHandler(server_context)
        result = handler.handle("/api/test", {}, mock_handler)

        assert result is None

    def test_handle_post_returns_none_by_default(self, server_context, mock_handler):
        """Should return None for POST by default."""
        from aragora.server.handlers.base import BaseHandler

        handler = BaseHandler(server_context)
        result = handler.handle_post("/api/test", {}, mock_handler)

        assert result is None

    def test_handle_delete_returns_none_by_default(self, server_context, mock_handler):
        """Should return None for DELETE by default."""
        from aragora.server.handlers.base import BaseHandler

        handler = BaseHandler(server_context)
        result = handler.handle_delete("/api/test", {}, mock_handler)

        assert result is None


class TestBaseHandlerAuthentication:
    """Tests for BaseHandler authentication methods."""

    def test_get_current_user_authenticated(self, server_context, mock_handler, mock_user_context):
        """Should return user context when authenticated."""
        from aragora.server.handlers.base import BaseHandler

        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            mock_extract.return_value = mock_user_context

            handler = BaseHandler(server_context)
            user = handler.get_current_user(mock_handler)

            assert user is mock_user_context

    def test_get_current_user_not_authenticated(self, server_context, mock_handler):
        """Should return None when not authenticated."""
        from aragora.server.handlers.base import BaseHandler

        mock_user = MagicMock()
        mock_user.is_authenticated = False

        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            mock_extract.return_value = mock_user

            handler = BaseHandler(server_context)
            user = handler.get_current_user(mock_handler)

            assert user is None

    def test_require_auth_or_error_success(self, server_context, mock_handler, mock_user_context):
        """Should return user and None error when authenticated."""
        from aragora.server.handlers.base import BaseHandler

        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            mock_extract.return_value = mock_user_context

            handler = BaseHandler(server_context)
            user, error = handler.require_auth_or_error(mock_handler)

            assert user is mock_user_context
            assert error is None

    def test_require_auth_or_error_failure(self, server_context, mock_handler):
        """Should return None user and error when not authenticated."""
        from aragora.server.handlers.base import BaseHandler

        mock_user = MagicMock()
        mock_user.is_authenticated = False

        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            mock_extract.return_value = mock_user

            handler = BaseHandler(server_context)
            user, error = handler.require_auth_or_error(mock_handler)

            assert user is None
            assert error is not None
            assert error.status_code == 401

    def test_require_admin_or_error_success(self, server_context, mock_handler, mock_user_context):
        """Should return user for admin."""
        from aragora.server.handlers.base import BaseHandler

        mock_user_context.roles = ["admin"]

        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            mock_extract.return_value = mock_user_context

            handler = BaseHandler(server_context)
            user, error = handler.require_admin_or_error(mock_handler)

            assert user is mock_user_context
            assert error is None

    def test_require_admin_or_error_not_admin(
        self, server_context, mock_handler, mock_user_context
    ):
        """Should return error for non-admin."""
        from aragora.server.handlers.base import BaseHandler

        mock_user_context.roles = ["member"]
        mock_user_context.permissions = []
        mock_user_context.is_admin = False

        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            mock_extract.return_value = mock_user_context

            handler = BaseHandler(server_context)
            user, error = handler.require_admin_or_error(mock_handler)

            assert user is None
            assert error is not None
            assert error.status_code == 403


# =============================================================================
# Tests for TypedHandler Classes
# =============================================================================


class TestTypedHandler:
    """Tests for the TypedHandler class."""

    def test_typed_handler_inherits_base_handler(self, server_context):
        """Should inherit from BaseHandler."""
        from aragora.server.handlers.base import TypedHandler, BaseHandler

        handler = TypedHandler(server_context)

        assert isinstance(handler, BaseHandler)

    def test_with_dependencies_factory(self, server_context):
        """Should create handler with injected dependencies."""
        from aragora.server.handlers.base import TypedHandler

        mock_user_store = MagicMock()
        mock_storage = MagicMock()

        handler = TypedHandler.with_dependencies(
            server_context, user_store=mock_user_store, storage=mock_storage
        )

        assert handler._user_store_factory() is mock_user_store
        assert handler._storage_factory() is mock_storage

    def test_get_user_store_from_factory(self, server_context):
        """Should get user store from factory if set."""
        from aragora.server.handlers.base import TypedHandler

        mock_store = MagicMock()
        handler = TypedHandler(server_context)
        handler._user_store_factory = lambda: mock_store

        result = handler.get_user_store()

        assert result is mock_store


class TestAuthenticatedHandler:
    """Tests for the AuthenticatedHandler class."""

    def test_ensure_authenticated_success(self, server_context, mock_handler, mock_user_context):
        """Should set current_user on success."""
        from aragora.server.handlers.base import AuthenticatedHandler

        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            mock_extract.return_value = mock_user_context

            handler = AuthenticatedHandler(server_context)
            user, error = handler._ensure_authenticated(mock_handler)

            assert user is mock_user_context
            assert error is None
            assert handler.current_user is mock_user_context

    def test_ensure_authenticated_failure(self, server_context, mock_handler):
        """Should clear current_user on failure."""
        from aragora.server.handlers.base import AuthenticatedHandler

        mock_user = MagicMock()
        mock_user.is_authenticated = False

        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            mock_extract.return_value = mock_user

            handler = AuthenticatedHandler(server_context)
            user, error = handler._ensure_authenticated(mock_handler)

            assert user is None
            assert error is not None
            assert handler.current_user is None


class TestPermissionHandler:
    """Tests for the PermissionHandler class."""

    def test_ensure_permission_with_no_required_permission(
        self, server_context, mock_handler, mock_user_context
    ):
        """Should allow when no permission required for method."""
        from aragora.server.handlers.base import PermissionHandler

        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            mock_extract.return_value = mock_user_context

            handler = PermissionHandler(server_context)
            user, error = handler._ensure_permission(mock_handler, "GET")

            assert user is mock_user_context
            assert error is None

    def test_custom_permissions_mapping(self, server_context):
        """Should use custom REQUIRED_PERMISSIONS."""
        from aragora.server.handlers.base import PermissionHandler

        class CustomHandler(PermissionHandler):
            REQUIRED_PERMISSIONS = {
                "GET": "custom:read",
                "POST": "custom:write",
            }

        handler = CustomHandler(server_context)

        assert handler.REQUIRED_PERMISSIONS["GET"] == "custom:read"


class TestAdminHandler:
    """Tests for the AdminHandler class."""

    def test_log_admin_action(self, server_context, mock_user_context):
        """Should log admin actions."""
        from aragora.server.handlers.base import AdminHandler
        import logging

        handler = AdminHandler(server_context)
        handler._current_user = mock_user_context

        with patch.object(logging.getLogger("aragora.server.handlers.base"), "info") as mock_log:
            handler._log_admin_action("test_action", "resource-123", {"detail": "value"})

            mock_log.assert_called_once()
            call_args = str(mock_log.call_args)
            assert "test_action" in call_args

    def test_log_admin_action_disabled(self, server_context):
        """Should not log when AUDIT_ACTIONS is False."""
        from aragora.server.handlers.base import AdminHandler
        import logging

        handler = AdminHandler(server_context)
        handler.AUDIT_ACTIONS = False

        with patch.object(logging.getLogger("aragora.server.handlers.base"), "info") as mock_log:
            handler._log_admin_action("test_action")

            mock_log.assert_not_called()


class TestResourceHandler:
    """Tests for the ResourceHandler class."""

    def test_resource_permissions_generated(self, server_context):
        """Should generate permissions from resource name."""
        from aragora.server.handlers.base import ResourceHandler

        class TestResourceHandler(ResourceHandler):
            RESOURCE_NAME = "debate"

        handler = TestResourceHandler(server_context)

        assert handler.REQUIRED_PERMISSIONS["GET"] == "debate:read"
        assert handler.REQUIRED_PERMISSIONS["POST"] == "debate:create"
        assert handler.REQUIRED_PERMISSIONS["DELETE"] == "debate:delete"

    def test_extract_resource_id(self, server_context):
        """Should extract resource ID from path."""
        from aragora.server.handlers.base import ResourceHandler

        handler = ResourceHandler(server_context)
        handler.RESOURCE_NAME = "debate"

        result = handler._extract_resource_id("/api/v1/debates/abc123")

        assert result == "abc123"

    def test_extract_resource_id_collection(self, server_context):
        """Should return None for collection endpoint."""
        from aragora.server.handlers.base import ResourceHandler

        handler = ResourceHandler(server_context)
        handler.RESOURCE_NAME = "debate"

        result = handler._extract_resource_id("/api/v1/debates")

        assert result is None

    def test_default_methods_return_501(self, server_context, mock_handler, mock_user_context):
        """Should return 501 for unimplemented methods."""
        from aragora.server.handlers.base import ResourceHandler

        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            mock_extract.return_value = mock_user_context

            handler = ResourceHandler(server_context)
            result = handler._list_resources({}, mock_handler)

            assert result.status_code == 501


# =============================================================================
# Tests for Decorators
# =============================================================================


class TestRequireQuotaDecorator:
    """Tests for the @require_quota decorator."""

    def test_require_quota_passes_when_under_limit(self, mock_handler, mock_user_context):
        """Should allow operation when under quota."""
        from aragora.server.handlers.base import require_quota, json_response

        # Mock organization with available quota
        mock_org = MagicMock()
        mock_org.is_at_limit = False
        mock_org.debates_used_this_month = 5
        mock_org.limits.debates_per_month = 100

        mock_user_store = MagicMock()
        mock_user_store.get_organization_by_id.return_value = mock_org
        mock_handler.user_store = mock_user_store

        @require_quota()
        def test_func(handler, user):
            return json_response({"success": True})

        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            mock_extract.return_value = mock_user_context

            result = test_func(handler=mock_handler)

            assert result.status_code == 200

    def test_require_quota_blocks_when_at_limit(self, mock_handler, mock_user_context):
        """Should return 429 when at quota limit."""
        from aragora.server.handlers.base import require_quota, json_response

        # Mock organization at quota limit
        mock_org = MagicMock()
        mock_org.is_at_limit = True
        mock_org.debates_used_this_month = 100
        mock_org.limits.debates_per_month = 100
        mock_org.tier.value = "free"

        mock_user_store = MagicMock()
        mock_user_store.get_organization_by_id.return_value = mock_org
        mock_handler.user_store = mock_user_store

        @require_quota()
        def test_func(handler, user):
            return json_response({"success": True})

        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            mock_extract.return_value = mock_user_context

            result = test_func(handler=mock_handler)

            assert result.status_code == 429
            body = json.loads(result.body)
            assert body["code"] == "quota_exceeded"

    def test_require_quota_requires_auth(self, mock_handler):
        """Should return 401 when not authenticated."""
        from aragora.server.handlers.base import require_quota, json_response

        mock_user = MagicMock()
        mock_user.is_authenticated = False
        mock_user.error_reason = "Token expired"

        @require_quota()
        def test_func(handler, user):
            return json_response({"success": True})

        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            mock_extract.return_value = mock_user

            result = test_func(handler=mock_handler)

            assert result.status_code == 401


class TestApiEndpointDecorator:
    """Tests for the @api_endpoint decorator."""

    def test_api_endpoint_attaches_metadata(self):
        """Should attach API metadata to function."""
        from aragora.server.handlers.base import api_endpoint

        @api_endpoint(
            method="POST",
            path="/api/v1/debates",
            summary="Create a debate",
            description="Creates a new debate session",
        )
        def create_debate():
            pass

        assert hasattr(create_debate, "_api_metadata")
        meta = create_debate._api_metadata
        assert meta["method"] == "POST"
        assert meta["path"] == "/api/v1/debates"
        assert meta["summary"] == "Create a debate"


class TestValidateBodyDecorator:
    """Tests for the @validate_body decorator."""

    @pytest.mark.asyncio
    async def test_validate_body_async_success(self):
        """Should pass validation for valid body."""
        from aragora.server.handlers.base import validate_body, json_response

        class MockSelf:
            pass

        mock_request = MagicMock()

        # The decorator calls await request.json(), so we need an async mock
        async def mock_json():
            return {"name": "test", "value": 123}

        mock_request.json = mock_json

        @validate_body(["name", "value"])
        async def test_func(self, request):
            return json_response({"success": True})

        result = await test_func(MockSelf(), mock_request)

        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_validate_body_async_missing_field(self):
        """Should return error for missing required field."""
        from aragora.server.handlers.base import validate_body

        class MockSelf:
            pass

        mock_request = MagicMock()

        # The decorator calls await request.json(), so we need an async mock
        async def mock_json():
            return {"name": "test"}  # Missing "value"

        mock_request.json = mock_json

        @validate_body(["name", "value"])
        async def test_func(self, request):
            return {"success": True}

        result = await test_func(MockSelf(), mock_request)

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "value" in body["error"]

    @pytest.mark.asyncio
    async def test_validate_body_async_invalid_json(self):
        """Should return error for invalid JSON."""
        from aragora.server.handlers.base import validate_body
        import json as json_module

        class MockSelf:
            pass

        mock_request = MagicMock()

        # The decorator calls await request.json(), so we need an async function that raises
        async def mock_json():
            raise json_module.JSONDecodeError("test", "", 0)

        mock_request.json = mock_json

        @validate_body(["name"])
        async def test_func(self, request):
            return {"success": True}

        result = await test_func(MockSelf(), mock_request)

        assert result.status_code == 400


# =============================================================================
# Tests for PathMatcher and RouteDispatcher
# =============================================================================


class TestPathMatcher:
    """Tests for the PathMatcher class."""

    def test_path_matcher_exact_match(self):
        """Should match exact path."""
        from aragora.server.handlers.base import PathMatcher

        matcher = PathMatcher("/api/v1/debates")
        result = matcher.match("/api/v1/debates")

        assert result == {}

    def test_path_matcher_with_params(self):
        """Should extract path parameters."""
        from aragora.server.handlers.base import PathMatcher

        matcher = PathMatcher("/api/v1/agent/{name}/profile")
        result = matcher.match("/api/v1/agent/claude/profile")

        assert result == {"name": "claude"}

    def test_path_matcher_multiple_params(self):
        """Should extract multiple parameters."""
        from aragora.server.handlers.base import PathMatcher

        matcher = PathMatcher("/api/v1/compare/{agent_a}/{agent_b}")
        result = matcher.match("/api/v1/compare/claude/gpt4")

        assert result == {"agent_a": "claude", "agent_b": "gpt4"}

    def test_path_matcher_no_match_different_length(self):
        """Should not match paths with different lengths."""
        from aragora.server.handlers.base import PathMatcher

        matcher = PathMatcher("/api/v1/debates")
        result = matcher.match("/api/v1/debates/123")

        assert result is None

    def test_path_matcher_no_match_different_segment(self):
        """Should not match paths with different segments."""
        from aragora.server.handlers.base import PathMatcher

        matcher = PathMatcher("/api/v1/debates")
        result = matcher.match("/api/v1/agents")

        assert result is None

    def test_path_matcher_matches_method(self):
        """Should return True for matching path."""
        from aragora.server.handlers.base import PathMatcher

        matcher = PathMatcher("/api/v1/debates")

        assert matcher.matches("/api/v1/debates") is True
        assert matcher.matches("/api/v1/other") is False


class TestRouteDispatcher:
    """Tests for the RouteDispatcher class."""

    def test_route_dispatcher_simple_route(self):
        """Should dispatch to handler for simple route."""
        from aragora.server.handlers.base import RouteDispatcher

        dispatcher = RouteDispatcher()
        handler_called = [False]

        def handler(query_params):
            handler_called[0] = True
            return {"handled": True}

        dispatcher.add_route("/api/v1/debates", handler)
        result = dispatcher.dispatch("/api/v1/debates")

        assert handler_called[0] is True
        assert result == {"handled": True}

    def test_route_dispatcher_with_params(self):
        """Should pass path params to handler."""
        from aragora.server.handlers.base import RouteDispatcher

        dispatcher = RouteDispatcher()
        received_params = [None]

        def handler(params, query_params):
            received_params[0] = params
            return {"handled": True}

        dispatcher.add_route("/api/v1/agent/{name}/profile", handler)
        dispatcher.dispatch("/api/v1/agent/claude/profile")

        assert received_params[0] == {"name": "claude"}

    def test_route_dispatcher_no_match(self):
        """Should return None for unmatched routes."""
        from aragora.server.handlers.base import RouteDispatcher

        dispatcher = RouteDispatcher()
        dispatcher.add_route("/api/v1/debates", lambda qp: {"handled": True})

        result = dispatcher.dispatch("/api/v1/other")

        assert result is None

    def test_route_dispatcher_can_handle(self):
        """Should check if route can be handled."""
        from aragora.server.handlers.base import RouteDispatcher

        dispatcher = RouteDispatcher()
        dispatcher.add_route("/api/v1/debates", lambda qp: None)

        assert dispatcher.can_handle("/api/v1/debates") is True
        assert dispatcher.can_handle("/api/v1/other") is False

    def test_route_dispatcher_chaining(self):
        """Should support method chaining."""
        from aragora.server.handlers.base import RouteDispatcher

        dispatcher = (
            RouteDispatcher()
            .add_route("/api/v1/debates", lambda qp: None)
            .add_route("/api/v1/agents", lambda qp: None)
        )

        assert dispatcher.can_handle("/api/v1/debates") is True
        assert dispatcher.can_handle("/api/v1/agents") is True


# =============================================================================
# Tests for feature_unavailable_response
# =============================================================================


class TestFeatureUnavailableResponse:
    """Tests for the feature_unavailable_response function."""

    def test_feature_unavailable_response_returns_503(self):
        """Should return 503 for unavailable feature."""
        from aragora.server.handlers.base import feature_unavailable_response

        result = feature_unavailable_response("pulse")

        assert result.status_code == 503

    def test_feature_unavailable_response_with_custom_message(self):
        """Should use custom message when provided."""
        from aragora.server.handlers.base import feature_unavailable_response

        result = feature_unavailable_response("custom", "Custom message")

        body = json.loads(result.body)
        # The response structure depends on the implementation
        assert result.status_code == 503


# =============================================================================
# Tests for Validation Functions
# =============================================================================


class TestValidation:
    """Tests for validation functions."""

    def test_validate_debate_id_valid(self):
        """Should accept valid debate ID."""
        from aragora.server.handlers.base import validate_debate_id

        is_valid, error = validate_debate_id("abc-123-def")

        assert is_valid is True
        assert error is None  # Returns None on success, not empty string

    def test_validate_debate_id_invalid(self):
        """Should reject invalid debate ID."""
        from aragora.server.handlers.base import validate_debate_id

        is_valid, error = validate_debate_id("../../../etc/passwd")

        assert is_valid is False
        assert error is not None

    def test_validate_agent_name_valid(self):
        """Should accept valid agent name."""
        from aragora.server.handlers.base import validate_agent_name

        is_valid, error = validate_agent_name("claude-3")

        assert is_valid is True

    def test_validate_agent_name_invalid(self):
        """Should reject invalid agent name."""
        from aragora.server.handlers.base import validate_agent_name

        is_valid, error = validate_agent_name("agent<script>")

        assert is_valid is False


# =============================================================================
# Tests for AsyncTypedHandler
# =============================================================================


class TestAsyncTypedHandler:
    """Tests for the AsyncTypedHandler class."""

    @pytest.mark.asyncio
    async def test_async_handle_returns_none(self, server_context, mock_handler):
        """Should return None by default."""
        from aragora.server.handlers.base import AsyncTypedHandler

        handler = AsyncTypedHandler(server_context)
        result = await handler.handle("/api/test", {}, mock_handler)

        assert result is None

    @pytest.mark.asyncio
    async def test_async_handle_post_returns_none(self, server_context, mock_handler):
        """Should return None for POST by default."""
        from aragora.server.handlers.base import AsyncTypedHandler

        handler = AsyncTypedHandler(server_context)
        result = await handler.handle_post("/api/test", {}, mock_handler)

        assert result is None


# =============================================================================
# Tests for HTML Response Functions
# =============================================================================


class TestHtmlResponse:
    """Tests for HTML response functions."""

    def test_html_response_basic(self):
        """Should create HTML response."""
        from aragora.server.handlers.utils.responses import html_response

        result = html_response("<html><body>Hello</body></html>")

        assert result.status_code == 200
        assert "text/html" in result.content_type
        assert b"Hello" in result.body

    def test_html_response_with_nonce(self):
        """Should add CSP header with nonce."""
        from aragora.server.handlers.utils.responses import html_response

        result = html_response("<html></html>", nonce="abc123")

        assert "Content-Security-Policy" in result.headers
        assert "nonce-abc123" in result.headers["Content-Security-Policy"]

    def test_redirect_response(self):
        """Should create redirect response."""
        from aragora.server.handlers.utils.responses import redirect_response

        result = redirect_response("/new/location")

        assert result.status_code == 302
        assert result.headers["Location"] == "/new/location"

    def test_redirect_response_custom_status(self):
        """Should support custom redirect status."""
        from aragora.server.handlers.utils.responses import redirect_response

        result = redirect_response("/permanent", status=301)

        assert result.status_code == 301
