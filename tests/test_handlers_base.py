"""
Tests for BaseHandler utilities.

Tests cover:
- Response helpers (json_response, error_response)
- Parameter extractors (get_int_param, get_float_param, etc.)
- PathMatcher for URL pattern matching
- RouteDispatcher for request routing
- BaseHandler core methods
"""

from __future__ import annotations

import json
import pytest
from unittest.mock import Mock, MagicMock
from dataclasses import dataclass

from aragora.server.handlers.base import (
    HandlerResult,
    json_response,
    error_response,
    generate_trace_id,
    _map_exception_to_status,
    PathMatcher,
    RouteDispatcher,
    parse_query_params,
    get_int_param,
    get_float_param,
    get_bool_param,
    get_string_param,
    get_clamped_int_param,
    get_bounded_float_param,
    get_bounded_string_param,
    BaseHandler,
)


# ============================================================================
# HandlerResult Tests
# ============================================================================

class TestHandlerResult:
    """Tests for HandlerResult dataclass."""

    def test_create_basic_result(self):
        """Test creating a basic handler result."""
        result = HandlerResult(
            status_code=200,
            content_type="application/json",
            body=b'{"ok": true}'
        )
        assert result.status_code == 200
        assert result.content_type == "application/json"
        assert result.body == b'{"ok": true}'

    def test_headers_default_to_empty_dict(self):
        """Test that headers default to empty dict."""
        result = HandlerResult(
            status_code=200,
            content_type="text/plain",
            body=b"hello"
        )
        assert result.headers == {}

    def test_headers_can_be_provided(self):
        """Test that custom headers can be set."""
        result = HandlerResult(
            status_code=200,
            content_type="application/json",
            body=b"{}",
            headers={"X-Custom": "value"}
        )
        assert result.headers == {"X-Custom": "value"}


# ============================================================================
# json_response Tests
# ============================================================================

class TestJsonResponse:
    """Tests for json_response helper."""

    def test_creates_valid_json(self):
        """Test that json_response creates valid JSON body."""
        result = json_response({"key": "value"})
        body_data = json.loads(result.body)
        assert body_data == {"key": "value"}

    def test_sets_content_type(self):
        """Test that content type is set to application/json."""
        result = json_response({})
        assert result.content_type == "application/json"

    def test_default_status_200(self):
        """Test default status code is 200."""
        result = json_response({})
        assert result.status_code == 200

    def test_custom_status(self):
        """Test custom status code."""
        result = json_response({}, status=201)
        assert result.status_code == 201

    def test_custom_headers(self):
        """Test custom headers are included."""
        result = json_response({}, headers={"X-Custom": "test"})
        assert result.headers == {"X-Custom": "test"}

    def test_handles_nested_data(self):
        """Test handling nested dictionaries and lists."""
        data = {
            "users": [
                {"name": "alice", "scores": [1, 2, 3]},
                {"name": "bob", "scores": [4, 5, 6]},
            ]
        }
        result = json_response(data)
        body_data = json.loads(result.body)
        assert body_data == data

    def test_handles_datetime_via_default_str(self):
        """Test that datetime objects are serialized via default=str."""
        from datetime import datetime
        data = {"timestamp": datetime(2026, 1, 10, 12, 0, 0)}
        result = json_response(data)
        body_data = json.loads(result.body)
        assert "2026" in body_data["timestamp"]


# ============================================================================
# error_response Tests
# ============================================================================

class TestErrorResponse:
    """Tests for error_response helper."""

    def test_simple_error_format(self):
        """Test simple error format for backward compatibility."""
        result = error_response("Something went wrong", 400)
        body_data = json.loads(result.body)
        assert body_data == {"error": "Something went wrong"}

    def test_default_status_400(self):
        """Test default status is 400."""
        result = error_response("Bad request")
        assert result.status_code == 400

    def test_custom_status(self):
        """Test custom status codes."""
        result = error_response("Not found", 404)
        assert result.status_code == 404

    def test_structured_format_with_code(self):
        """Test structured format when code is provided."""
        result = error_response("Field required", 400, code="VALIDATION_ERROR")
        body_data = json.loads(result.body)
        assert body_data["error"]["message"] == "Field required"
        assert body_data["error"]["code"] == "VALIDATION_ERROR"

    def test_structured_format_with_trace_id(self):
        """Test trace_id is included in structured format."""
        result = error_response("Error", 500, trace_id="abc123")
        body_data = json.loads(result.body)
        assert body_data["error"]["trace_id"] == "abc123"

    def test_structured_format_with_suggestion(self):
        """Test suggestion is included in structured format."""
        result = error_response(
            "Missing field",
            400,
            suggestion="Include 'name' in request body"
        )
        body_data = json.loads(result.body)
        assert body_data["error"]["suggestion"] == "Include 'name' in request body"

    def test_structured_format_with_details(self):
        """Test details dict is included in structured format."""
        result = error_response(
            "Validation failed",
            400,
            details={"field": "name", "reason": "too short"}
        )
        body_data = json.loads(result.body)
        assert body_data["error"]["details"]["field"] == "name"

    def test_forced_structured_format(self):
        """Test structured=True forces structured format."""
        result = error_response("Simple error", 400, structured=True)
        body_data = json.loads(result.body)
        assert "message" in body_data["error"]


# ============================================================================
# Trace ID Tests
# ============================================================================

class TestGenerateTraceId:
    """Tests for generate_trace_id."""

    def test_returns_string(self):
        """Test that trace ID is a string."""
        trace_id = generate_trace_id()
        assert isinstance(trace_id, str)

    def test_length_is_8(self):
        """Test trace ID is 8 characters."""
        trace_id = generate_trace_id()
        assert len(trace_id) == 8

    def test_unique_ids(self):
        """Test that multiple calls return unique IDs."""
        ids = [generate_trace_id() for _ in range(100)]
        assert len(set(ids)) == 100


# ============================================================================
# Exception Mapping Tests
# ============================================================================

class TestMapExceptionToStatus:
    """Tests for _map_exception_to_status."""

    def test_file_not_found_returns_404(self):
        """Test FileNotFoundError maps to 404."""
        assert _map_exception_to_status(FileNotFoundError()) == 404

    def test_key_error_returns_404(self):
        """Test KeyError maps to 404."""
        assert _map_exception_to_status(KeyError("key")) == 404

    def test_value_error_returns_400(self):
        """Test ValueError maps to 400."""
        assert _map_exception_to_status(ValueError("bad value")) == 400

    def test_type_error_returns_400(self):
        """Test TypeError maps to 400."""
        assert _map_exception_to_status(TypeError("type issue")) == 400

    def test_permission_error_returns_403(self):
        """Test PermissionError maps to 403."""
        assert _map_exception_to_status(PermissionError("denied")) == 403

    def test_timeout_error_returns_504(self):
        """Test TimeoutError maps to 504."""
        assert _map_exception_to_status(TimeoutError()) == 504

    def test_connection_error_returns_502(self):
        """Test ConnectionError maps to 502."""
        assert _map_exception_to_status(ConnectionError()) == 502

    def test_unknown_exception_returns_default(self):
        """Test unknown exceptions return default 500."""
        class CustomError(Exception):
            pass
        assert _map_exception_to_status(CustomError()) == 500

    def test_custom_default(self):
        """Test custom default status."""
        class CustomError(Exception):
            pass
        assert _map_exception_to_status(CustomError(), default=503) == 503


# ============================================================================
# Parameter Extractor Tests
# ============================================================================

class TestGetIntParam:
    """Tests for get_int_param."""

    def test_returns_default_when_missing(self):
        """Test returns default when key is missing."""
        assert get_int_param({}, "limit", 10) == 10

    def test_parses_valid_int(self):
        """Test parsing valid integer."""
        assert get_int_param({"limit": "50"}, "limit", 10) == 50

    def test_returns_default_on_invalid(self):
        """Test returns default on invalid input."""
        assert get_int_param({"limit": "abc"}, "limit", 10) == 10

    def test_handles_list_value(self):
        """Test handles list values from query strings."""
        assert get_int_param({"limit": ["25"]}, "limit", 10) == 25

    def test_handles_empty_list(self):
        """Test handles empty list."""
        assert get_int_param({"limit": []}, "limit", 10) == 10

    def test_handles_none_value(self):
        """Test handles None value."""
        assert get_int_param({"limit": None}, "limit", 10) == 10


class TestGetFloatParam:
    """Tests for get_float_param."""

    def test_returns_default_when_missing(self):
        """Test returns default when key is missing."""
        assert get_float_param({}, "score", 0.5) == 0.5

    def test_parses_valid_float(self):
        """Test parsing valid float."""
        assert get_float_param({"score": "0.75"}, "score", 0.5) == 0.75

    def test_parses_integer_as_float(self):
        """Test parsing integer as float."""
        assert get_float_param({"score": "3"}, "score", 0.5) == 3.0

    def test_returns_default_on_invalid(self):
        """Test returns default on invalid input."""
        assert get_float_param({"score": "not-a-number"}, "score", 0.5) == 0.5

    def test_handles_list_value(self):
        """Test handles list values."""
        assert get_float_param({"score": ["0.9"]}, "score", 0.5) == 0.9


class TestGetBoolParam:
    """Tests for get_bool_param."""

    def test_true_values(self):
        """Test various true values."""
        for val in ["true", "True", "TRUE", "1", "yes", "Yes", "on", "ON"]:
            assert get_bool_param({"flag": val}, "flag", False) is True

    def test_false_values(self):
        """Test various false values."""
        for val in ["false", "False", "0", "no", "off", ""]:
            assert get_bool_param({"flag": val}, "flag", True) is False

    def test_returns_default_when_missing(self):
        """Test returns default when missing."""
        assert get_bool_param({}, "flag", True) is True
        assert get_bool_param({}, "flag", False) is False


class TestGetStringParam:
    """Tests for get_string_param."""

    def test_returns_string(self):
        """Test returns string value."""
        assert get_string_param({"name": "claude"}, "name") == "claude"

    def test_returns_default_when_missing(self):
        """Test returns default when missing."""
        assert get_string_param({}, "name", "default") == "default"

    def test_returns_none_default(self):
        """Test returns None as default."""
        assert get_string_param({}, "name") is None

    def test_handles_list_value(self):
        """Test handles list values."""
        assert get_string_param({"name": ["alice", "bob"]}, "name") == "alice"

    def test_handles_empty_list(self):
        """Test handles empty list."""
        assert get_string_param({"name": []}, "name", "default") == "default"


class TestGetClampedIntParam:
    """Tests for get_clamped_int_param."""

    def test_clamps_to_min(self):
        """Test value is clamped to minimum."""
        assert get_clamped_int_param({"n": "0"}, "n", 10, 1, 100) == 1

    def test_clamps_to_max(self):
        """Test value is clamped to maximum."""
        assert get_clamped_int_param({"n": "500"}, "n", 10, 1, 100) == 100

    def test_value_in_range(self):
        """Test value in range is unchanged."""
        assert get_clamped_int_param({"n": "50"}, "n", 10, 1, 100) == 50

    def test_uses_default_when_missing(self):
        """Test uses default when key is missing."""
        assert get_clamped_int_param({}, "n", 25, 1, 100) == 25


class TestGetBoundedFloatParam:
    """Tests for get_bounded_float_param."""

    def test_clamps_to_min(self):
        """Test value is clamped to minimum."""
        assert get_bounded_float_param({"f": "-1"}, "f", 0.5, 0.0, 1.0) == 0.0

    def test_clamps_to_max(self):
        """Test value is clamped to maximum."""
        assert get_bounded_float_param({"f": "5"}, "f", 0.5, 0.0, 1.0) == 1.0

    def test_value_in_range(self):
        """Test value in range is unchanged."""
        assert get_bounded_float_param({"f": "0.75"}, "f", 0.5, 0.0, 1.0) == 0.75


class TestGetBoundedStringParam:
    """Tests for get_bounded_string_param."""

    def test_truncates_long_string(self):
        """Test long strings are truncated."""
        result = get_bounded_string_param({"s": "a" * 100}, "s", "", 10)
        assert len(result) == 10
        assert result == "a" * 10

    def test_short_string_unchanged(self):
        """Test short strings are unchanged."""
        result = get_bounded_string_param({"s": "hello"}, "s", "", 10)
        assert result == "hello"

    def test_returns_default_when_missing(self):
        """Test returns default when missing."""
        result = get_bounded_string_param({}, "s", "default", 10)
        assert result == "default"


# ============================================================================
# Parse Query Params Tests
# ============================================================================

class TestParseQueryParams:
    """Tests for parse_query_params."""

    def test_empty_string_returns_empty_dict(self):
        """Test empty string returns empty dict."""
        assert parse_query_params("") == {}

    def test_parses_single_param(self):
        """Test parsing single parameter."""
        result = parse_query_params("name=claude")
        assert result == {"name": "claude"}

    def test_parses_multiple_params(self):
        """Test parsing multiple parameters."""
        result = parse_query_params("name=claude&limit=10")
        assert result["name"] == "claude"
        assert result["limit"] == "10"

    def test_single_value_not_list(self):
        """Test single values are not wrapped in lists."""
        result = parse_query_params("name=claude")
        assert result["name"] == "claude"  # Not ["claude"]

    def test_multiple_values_are_list(self):
        """Test multiple values for same key are lists."""
        result = parse_query_params("tag=a&tag=b&tag=c")
        assert result["tag"] == ["a", "b", "c"]


# ============================================================================
# PathMatcher Tests
# ============================================================================

class TestPathMatcher:
    """Tests for PathMatcher class."""

    def test_exact_match(self):
        """Test exact path matching."""
        matcher = PathMatcher("/api/debates")
        result = matcher.match("/api/debates")
        assert result == {}

    def test_exact_match_with_trailing_slash(self):
        """Test matching ignores trailing slashes."""
        matcher = PathMatcher("/api/debates")
        result = matcher.match("/api/debates/")
        assert result == {}

    def test_captures_single_param(self):
        """Test capturing a single parameter."""
        matcher = PathMatcher("/api/agent/{name}")
        result = matcher.match("/api/agent/claude")
        assert result == {"name": "claude"}

    def test_captures_multiple_params(self):
        """Test capturing multiple parameters."""
        matcher = PathMatcher("/api/agent/{name}/{action}")
        result = matcher.match("/api/agent/claude/profile")
        assert result == {"name": "claude", "action": "profile"}

    def test_no_match_different_length(self):
        """Test no match when path length differs."""
        matcher = PathMatcher("/api/agent/{name}")
        result = matcher.match("/api/agent/claude/extra")
        assert result is None

    def test_no_match_different_prefix(self):
        """Test no match when prefix differs."""
        matcher = PathMatcher("/api/agent/{name}")
        result = matcher.match("/api/debate/test")
        assert result is None

    def test_matches_method(self):
        """Test matches() returns boolean."""
        matcher = PathMatcher("/api/debates")
        assert matcher.matches("/api/debates") is True
        assert matcher.matches("/api/other") is False

    def test_pattern_stored(self):
        """Test pattern is stored."""
        matcher = PathMatcher("/api/test")
        assert matcher.pattern == "/api/test"


# ============================================================================
# RouteDispatcher Tests
# ============================================================================

class TestRouteDispatcher:
    """Tests for RouteDispatcher class."""

    def test_add_route(self):
        """Test adding routes."""
        dispatcher = RouteDispatcher()
        handler = Mock(return_value="result")
        dispatcher.add_route("/api/test", handler)
        assert len(dispatcher.routes) == 1

    def test_add_route_chainable(self):
        """Test add_route is chainable."""
        dispatcher = RouteDispatcher()
        result = dispatcher.add_route("/a", Mock()).add_route("/b", Mock())
        assert result is dispatcher
        assert len(dispatcher.routes) == 2

    def test_dispatch_to_handler(self):
        """Test dispatching to correct handler."""
        dispatcher = RouteDispatcher()
        handler = Mock(return_value="result")
        dispatcher.add_route("/api/test", handler)

        result = dispatcher.dispatch("/api/test")
        assert result == "result"
        handler.assert_called_once()

    def test_dispatch_passes_params(self):
        """Test dispatch passes extracted params as positional args."""
        dispatcher = RouteDispatcher()
        handler = Mock(return_value="result")
        dispatcher.add_route("/api/agent/{name}", handler)

        dispatcher.dispatch("/api/agent/claude", {"limit": "10"})
        handler.assert_called_once()
        # Dispatch calls handler(path_params, query_params)
        call_args = handler.call_args[0]
        assert call_args[0] == {"name": "claude"}  # path_params
        assert call_args[1] == {"limit": "10"}  # query_params

    def test_dispatch_returns_none_for_unknown(self):
        """Test dispatch returns None for unknown paths."""
        dispatcher = RouteDispatcher()
        dispatcher.add_route("/api/known", Mock())

        result = dispatcher.dispatch("/api/unknown")
        assert result is None

    def test_can_handle(self):
        """Test can_handle returns correct boolean."""
        dispatcher = RouteDispatcher()
        dispatcher.add_route("/api/test", Mock())

        assert dispatcher.can_handle("/api/test") is True
        assert dispatcher.can_handle("/api/other") is False


# ============================================================================
# BaseHandler Tests
# ============================================================================

class TestBaseHandler:
    """Tests for BaseHandler class."""

    def test_init_with_server_context(self):
        """Test initialization with server context."""
        ctx = {"storage": Mock(), "elo": Mock()}
        handler = BaseHandler(ctx)
        assert handler.ctx == ctx

    def test_get_storage_returns_ctx_value(self):
        """Test get_storage returns context value."""
        storage_mock = Mock()
        handler = BaseHandler({"storage": storage_mock})
        assert handler.get_storage() == storage_mock

    def test_get_storage_returns_none_when_missing(self):
        """Test get_storage returns None when not in context."""
        handler = BaseHandler({})
        assert handler.get_storage() is None

    def test_get_elo_system_returns_ctx_value(self):
        """Test get_elo_system returns context value."""
        elo_mock = Mock()
        handler = BaseHandler({"elo_system": elo_mock})
        assert handler.get_elo_system() == elo_mock

    def test_get_elo_system_returns_none_when_missing(self):
        """Test get_elo_system returns None when not in context."""
        handler = BaseHandler({})
        assert handler.get_elo_system() is None

    def test_get_nomic_dir_returns_ctx_value(self):
        """Test get_nomic_dir returns context value."""
        handler = BaseHandler({"nomic_dir": "/path/to/.nomic"})
        assert handler.get_nomic_dir() == "/path/to/.nomic"

    def test_extract_path_param_by_index(self):
        """Test extracting path parameter by segment index."""
        handler = BaseHandler({})
        # Path: /api/agent/claude -> segment 2 is "claude"
        value, err = handler.extract_path_param("/api/agent/claude", 2, "name")
        assert value == "claude"
        assert err is None

    def test_extract_path_param_index_out_of_bounds(self):
        """Test extract_path_param returns error when index out of bounds."""
        handler = BaseHandler({})
        value, err = handler.extract_path_param("/api/agent", 5, "name")
        assert value is None
        assert err is not None
        assert err.status_code == 400

    def test_handle_returns_none_by_default(self):
        """Test handle returns None by default."""
        handler = BaseHandler({})
        result = handler.handle("/api/test", {}, Mock())
        assert result is None

    def test_handle_post_returns_none_by_default(self):
        """Test handle_post returns None by default."""
        handler = BaseHandler({})
        result = handler.handle_post("/api/test", {}, Mock())
        assert result is None


# ============================================================================
# Integration Tests
# ============================================================================

class TestBaseHandlerIntegration:
    """Integration tests combining multiple base handler utilities."""

    def test_error_flow(self):
        """Test complete error handling flow."""
        # Create error with trace ID
        trace_id = generate_trace_id()
        result = error_response(
            "Database error",
            503,
            code="DB_ERROR",
            trace_id=trace_id,
            suggestion="Retry in a few seconds"
        )

        assert result.status_code == 503
        body = json.loads(result.body)
        assert body["error"]["message"] == "Database error"
        assert body["error"]["code"] == "DB_ERROR"
        assert body["error"]["trace_id"] == trace_id

    def test_route_dispatch_with_params(self):
        """Test complete route dispatch with parameter extraction."""
        dispatcher = RouteDispatcher()

        def get_agent_handler(path_params: dict, query_params: dict):
            name = path_params.get("name", "unknown")
            limit = int(query_params.get("limit", "10"))
            return json_response({"agent": name, "limit": limit})

        dispatcher.add_route("/api/agent/{name}", get_agent_handler)

        result = dispatcher.dispatch("/api/agent/claude", {"limit": "25"})
        body = json.loads(result.body)
        assert body["agent"] == "claude"
        assert body["limit"] == 25

    def test_param_extraction_pipeline(self):
        """Test parameter extraction pipeline."""
        params = {"limit": "1000", "score": "1.5", "enabled": "true", "name": "test"}

        limit = get_clamped_int_param(params, "limit", 20, 1, 100)
        score = get_bounded_float_param(params, "score", 0.5, 0.0, 1.0)
        enabled = get_bool_param(params, "enabled", False)
        name = get_bounded_string_param(params, "name", "", 50)

        assert limit == 100  # Clamped to max
        assert score == 1.0  # Clamped to max
        assert enabled is True
        assert name == "test"
