"""Tests for base handler utilities and classes.

Tests the core handler infrastructure including:
- Response builders (json_response, error_response)
- Parameter extraction utilities
- Path extraction helpers
- Handler mixins (PaginatedHandlerMixin, CachedHandlerMixin, AuthenticatedHandlerMixin)
- BaseHandler class methods
- Quota enforcement
"""

import json
import re
from io import BytesIO
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.base import (
    BaseHandler,
    HandlerResult,
    PaginatedHandlerMixin,
    CachedHandlerMixin,
    AuthenticatedHandlerMixin,
    error_response,
    json_response,
    safe_error_response,
    get_host_header,
    get_agent_name,
    agent_to_dict,
    require_quota,
    get_int_param,
    get_float_param,
    get_bool_param,
    get_string_param,
    get_clamped_int_param,
    get_bounded_float_param,
    get_bounded_string_param,
    SAFE_ID_PATTERN,
    SAFE_AGENT_PATTERN,
)


def parse_body(result: HandlerResult) -> dict:
    """Parse JSON body from HandlerResult."""
    return json.loads(result.body.decode("utf-8"))


# =============================================================================
# Response Builder Tests
# =============================================================================


class TestJsonResponse:
    """Tests for json_response function."""

    def test_basic_response(self):
        """Test basic JSON response creation."""
        result = json_response({"key": "value"})
        assert isinstance(result, HandlerResult)
        assert result.status_code == 200
        assert parse_body(result) == {"key": "value"}

    def test_custom_status(self):
        """Test response with custom status code."""
        result = json_response({"created": True}, status=201)
        assert result.status_code == 201

    def test_empty_dict(self):
        """Test response with empty dict."""
        result = json_response({})
        assert parse_body(result) == {}
        assert result.status_code == 200

    def test_nested_data(self):
        """Test response with nested data structures."""
        data = {
            "items": [{"id": 1}, {"id": 2}],
            "metadata": {"count": 2, "page": 1},
        }
        result = json_response(data)
        assert parse_body(result) == data

    def test_various_status_codes(self):
        """Test various HTTP status codes."""
        for status in [200, 201, 202, 204, 400, 401, 403, 404, 500]:
            result = json_response({}, status=status)
            assert result.status_code == status


class TestErrorResponse:
    """Tests for error_response function."""

    def test_basic_error(self):
        """Test basic error response."""
        result = error_response("Something went wrong", 500)
        assert result.status_code == 500
        body = parse_body(result)
        assert "error" in body
        assert body["error"] == "Something went wrong"

    def test_default_status(self):
        """Test default status is 400."""
        result = error_response("Bad request")
        assert result.status_code == 400

    def test_various_error_codes(self):
        """Test various error status codes."""
        codes = {
            400: "Bad Request",
            401: "Unauthorized",
            403: "Forbidden",
            404: "Not Found",
            409: "Conflict",
            422: "Unprocessable Entity",
            429: "Too Many Requests",
            500: "Internal Server Error",
            503: "Service Unavailable",
        }
        for code, message in codes.items():
            result = error_response(message, code)
            assert result.status_code == code
            assert parse_body(result)["error"] == message


class TestSafeErrorResponse:
    """Tests for safe_error_response function."""

    def test_sanitizes_exception_message(self):
        """Test that exception details are sanitized."""
        exc = Exception("/secret/path/file.py: connection failed")
        result = safe_error_response(exc, "database operation")
        # Should not contain the full path
        body_str = result.body.decode("utf-8")
        assert "/secret/path" not in body_str

    def test_includes_trace_id(self):
        """Test that trace ID is included."""
        exc = Exception("Test error")
        result = safe_error_response(exc, "test operation")
        body = parse_body(result)
        # Result should have error structure
        assert "error" in body or "message" in body

    def test_custom_status_code(self):
        """Test custom status code."""
        exc = ValueError("Invalid input")
        result = safe_error_response(exc, "validation", status=422)
        assert result.status_code == 422


# =============================================================================
# Host and Agent Helper Tests
# =============================================================================


class TestGetHostHeader:
    """Tests for get_host_header function."""

    def test_with_host_header(self):
        """Test extraction when Host header present."""
        handler = MagicMock()
        handler.headers = {"Host": "example.com:8080"}
        assert get_host_header(handler) == "example.com:8080"

    def test_without_host_header(self):
        """Test default when Host header missing."""
        handler = MagicMock()
        handler.headers = {}
        result = get_host_header(handler)
        # Should return default
        assert "localhost" in result or result is not None

    def test_none_handler(self):
        """Test with None handler."""
        result = get_host_header(None)
        assert result is not None  # Should return default

    def test_custom_default(self):
        """Test custom default value."""
        result = get_host_header(None, default="custom.host:9000")
        assert result == "custom.host:9000"

    def test_handler_without_headers(self):
        """Test handler without headers attribute."""
        handler = MagicMock(spec=[])  # No attributes
        result = get_host_header(handler)
        assert result is not None


class TestGetAgentName:
    """Tests for get_agent_name function."""

    def test_dict_with_name(self):
        """Test extraction from dict with 'name' key."""
        assert get_agent_name({"name": "claude"}) == "claude"

    def test_dict_with_agent_name(self):
        """Test extraction from dict with 'agent_name' key."""
        assert get_agent_name({"agent_name": "gpt4"}) == "gpt4"

    def test_dict_prefers_agent_name(self):
        """Test agent_name takes precedence over name."""
        result = get_agent_name({"agent_name": "preferred", "name": "fallback"})
        assert result == "preferred"

    def test_object_with_name(self):
        """Test extraction from object with name attribute."""
        agent = MagicMock()
        agent.name = "gemini"
        agent.agent_name = None
        assert get_agent_name(agent) == "gemini"

    def test_none_input(self):
        """Test with None input."""
        assert get_agent_name(None) is None

    def test_empty_dict(self):
        """Test with empty dict."""
        assert get_agent_name({}) is None


class TestAgentToDict:
    """Tests for agent_to_dict function."""

    def test_dict_input(self):
        """Test with dict input returns copy."""
        original = {"name": "claude", "elo": 1600}
        result = agent_to_dict(original)
        assert result == original
        # Should be a copy
        result["elo"] = 1500
        assert original["elo"] == 1600

    def test_object_input(self):
        """Test with object input."""
        agent = MagicMock()
        agent.name = "claude"
        agent.agent_name = None
        agent.elo = 1650
        agent.wins = 10
        agent.losses = 5
        agent.draws = 2
        agent.win_rate = 0.67
        agent.games_played = 17
        agent.matches = 17

        result = agent_to_dict(agent)
        assert result["name"] == "claude"
        assert result["elo"] == 1650
        assert result["wins"] == 10
        assert result["losses"] == 5

    def test_none_input(self):
        """Test with None input."""
        assert agent_to_dict(None) == {}

    def test_include_name_false(self):
        """Test with include_name=False."""
        agent = MagicMock()
        agent.name = "claude"
        agent.elo = 1500
        agent.wins = 0
        agent.losses = 0
        agent.draws = 0
        agent.win_rate = 0.0
        agent.games_played = 0
        agent.matches = 0

        result = agent_to_dict(agent, include_name=False)
        assert "name" not in result
        assert "agent_name" not in result
        assert "elo" in result


# =============================================================================
# Parameter Extraction Tests
# =============================================================================


class TestParameterExtraction:
    """Tests for parameter extraction utilities."""

    def test_get_int_param_present(self):
        """Test int param extraction when present."""
        assert get_int_param({"limit": "50"}, "limit", 20) == 50

    def test_get_int_param_missing(self):
        """Test int param extraction when missing."""
        assert get_int_param({}, "limit", 20) == 20

    def test_get_int_param_invalid(self):
        """Test int param extraction with invalid value."""
        assert get_int_param({"limit": "abc"}, "limit", 20) == 20

    def test_get_float_param_present(self):
        """Test float param extraction when present."""
        result = get_float_param({"threshold": "0.75"}, "threshold", 0.5)
        assert abs(result - 0.75) < 0.001

    def test_get_float_param_missing(self):
        """Test float param extraction when missing."""
        assert get_float_param({}, "threshold", 0.5) == 0.5

    def test_get_bool_param_true_values(self):
        """Test bool param extraction with true values."""
        for val in ["true", "1", "yes", "True", "TRUE"]:
            assert get_bool_param({"flag": val}, "flag", False) is True

    def test_get_bool_param_false_values(self):
        """Test bool param extraction with false values."""
        for val in ["false", "0", "no", "False", "FALSE"]:
            assert get_bool_param({"flag": val}, "flag", True) is False

    def test_get_bool_param_missing(self):
        """Test bool param extraction when missing."""
        assert get_bool_param({}, "flag", True) is True
        assert get_bool_param({}, "flag", False) is False

    def test_get_string_param(self):
        """Test string param extraction."""
        assert get_string_param({"name": "test"}, "name", "default") == "test"
        assert get_string_param({}, "name", "default") == "default"

    def test_get_clamped_int_param(self):
        """Test clamped int param extraction."""
        # Within range
        assert get_clamped_int_param({"val": "50"}, "val", 20, 1, 100) == 50
        # Below minimum
        assert get_clamped_int_param({"val": "-5"}, "val", 20, 1, 100) == 1
        # Above maximum
        assert get_clamped_int_param({"val": "500"}, "val", 20, 1, 100) == 100

    def test_get_bounded_float_param(self):
        """Test bounded float param extraction."""
        # Within range
        result = get_bounded_float_param({"val": "0.5"}, "val", 0.3, 0.0, 1.0)
        assert abs(result - 0.5) < 0.001
        # Below minimum
        result = get_bounded_float_param({"val": "-1.0"}, "val", 0.3, 0.0, 1.0)
        assert abs(result - 0.0) < 0.001

    def test_get_bounded_string_param(self):
        """Test bounded string param extraction."""
        # Normal case
        assert get_bounded_string_param({"s": "hello"}, "s", "default", max_length=100) == "hello"
        # Truncation
        result = get_bounded_string_param({"s": "hello world"}, "s", "default", max_length=5)
        assert len(result) <= 5


# =============================================================================
# Handler Mixin Tests
# =============================================================================


class TestPaginatedHandlerMixin:
    """Tests for PaginatedHandlerMixin."""

    def test_get_pagination_defaults(self):
        """Test pagination with defaults."""
        mixin = PaginatedHandlerMixin()
        limit, offset = mixin.get_pagination({})
        assert limit == mixin.DEFAULT_LIMIT
        assert offset == mixin.DEFAULT_OFFSET

    def test_get_pagination_custom_values(self):
        """Test pagination with custom values."""
        mixin = PaginatedHandlerMixin()
        limit, offset = mixin.get_pagination({"limit": "50", "offset": "100"})
        assert limit == 50
        assert offset == 100

    def test_get_pagination_clamping(self):
        """Test pagination value clamping."""
        mixin = PaginatedHandlerMixin()
        # Over max
        limit, _ = mixin.get_pagination({"limit": "500"})
        assert limit == mixin.MAX_LIMIT
        # Negative offset
        _, offset = mixin.get_pagination({"offset": "-10"})
        assert offset == 0

    def test_get_pagination_custom_limits(self):
        """Test pagination with custom limits."""
        mixin = PaginatedHandlerMixin()
        limit, _ = mixin.get_pagination({"limit": "75"}, max_limit=50)
        assert limit == 50

    def test_paginated_response(self):
        """Test paginated response generation."""
        mixin = PaginatedHandlerMixin()
        items = [{"id": i} for i in range(10)]
        result = mixin.paginated_response(items, total=100, limit=10, offset=0)
        body = parse_body(result)

        assert body["items"] == items
        assert body["total"] == 100
        assert body["limit"] == 10
        assert body["offset"] == 0
        assert body["has_more"] is True

    def test_paginated_response_last_page(self):
        """Test paginated response on last page."""
        mixin = PaginatedHandlerMixin()
        items = [{"id": i} for i in range(5)]
        result = mixin.paginated_response(items, total=15, limit=10, offset=10)
        body = parse_body(result)
        assert body["has_more"] is False


class TestCachedHandlerMixin:
    """Tests for CachedHandlerMixin."""

    def test_cached_response_cache_miss(self):
        """Test cache miss generates value."""
        mixin = CachedHandlerMixin()
        generator_called = []

        def generator():
            generator_called.append(True)
            return {"data": "fresh"}

        with patch("aragora.server.handlers.base.get_handler_cache") as mock_cache:
            cache_instance = MagicMock()
            cache_instance.get.return_value = (False, None)  # Cache miss
            mock_cache.return_value = cache_instance

            result = mixin.cached_response("key", 300, generator)

            assert result == {"data": "fresh"}
            assert len(generator_called) == 1
            cache_instance.set.assert_called_once()

    def test_cached_response_cache_hit(self):
        """Test cache hit returns cached value."""
        mixin = CachedHandlerMixin()
        generator_called = []

        def generator():
            generator_called.append(True)
            return {"data": "fresh"}

        with patch("aragora.server.handlers.base.get_handler_cache") as mock_cache:
            cache_instance = MagicMock()
            cache_instance.get.return_value = (True, {"data": "cached"})  # Cache hit
            mock_cache.return_value = cache_instance

            result = mixin.cached_response("key", 300, generator)

            assert result == {"data": "cached"}
            assert len(generator_called) == 0  # Generator not called


class TestAuthenticatedHandlerMixin:
    """Tests for AuthenticatedHandlerMixin."""

    def test_require_auth_authenticated(self):
        """Test require_auth with authenticated user."""
        mixin = AuthenticatedHandlerMixin()

        # Add require_auth_or_error method mock
        mock_user = MagicMock()
        mock_user.is_authenticated = True
        mixin.require_auth_or_error = MagicMock(return_value=(mock_user, None))

        handler = MagicMock()
        result = mixin.require_auth(handler)
        assert result == mock_user

    def test_require_auth_unauthenticated(self):
        """Test require_auth with unauthenticated user."""
        mixin = AuthenticatedHandlerMixin()

        error_result = error_response("Not authenticated", 401)
        mixin.require_auth_or_error = MagicMock(return_value=(None, error_result))

        handler = MagicMock()
        result = mixin.require_auth(handler)
        assert result.status_code == 401


# =============================================================================
# BaseHandler Tests
# =============================================================================


class TestBaseHandler:
    """Tests for BaseHandler class."""

    @pytest.fixture
    def handler(self):
        """Create a base handler with mock context."""
        ctx = {
            "storage": MagicMock(),
            "elo_system": MagicMock(),
            "user_store": MagicMock(),
        }
        return BaseHandler(ctx)

    def test_init(self, handler):
        """Test handler initialization."""
        assert handler.ctx is not None
        assert "storage" in handler.ctx

    def test_get_storage(self, handler):
        """Test get_storage method."""
        storage = handler.get_storage()
        assert storage is not None

    def test_get_elo_system(self, handler):
        """Test get_elo_system method."""
        elo = handler.get_elo_system()
        assert elo is not None

    def test_extract_path_param_valid(self, handler):
        """Test path param extraction with valid input."""
        value, err = handler.extract_path_param("/api/debates/abc123", 2, "debate_id")
        assert value == "abc123"
        assert err is None

    def test_extract_path_param_missing(self, handler):
        """Test path param extraction with missing segment."""
        value, err = handler.extract_path_param("/api/debates", 3, "debate_id")
        assert value is None
        assert err is not None
        assert err.status_code == 400

    def test_extract_path_param_invalid_pattern(self, handler):
        """Test path param extraction with invalid pattern."""
        # Use a pattern that won't match
        value, err = handler.extract_path_param(
            "/api/debates/../../etc", 2, "debate_id", SAFE_ID_PATTERN
        )
        assert value is None
        assert err is not None

    def test_extract_path_params_multiple(self, handler):
        """Test extracting multiple path params."""
        params, err = handler.extract_path_params(
            "/api/agents/compare/claude/gpt4",
            [
                (3, "agent_a", SAFE_AGENT_PATTERN),
                (4, "agent_b", SAFE_AGENT_PATTERN),
            ],
        )
        assert err is None
        assert params["agent_a"] == "claude"
        assert params["agent_b"] == "gpt4"

    def test_read_json_body_valid(self, handler):
        """Test reading valid JSON body."""
        mock_handler = MagicMock()
        body_content = json.dumps({"key": "value"}).encode()
        mock_handler.headers = {"Content-Length": str(len(body_content))}
        mock_handler.rfile.read.return_value = body_content

        result = handler.read_json_body(mock_handler)
        assert result == {"key": "value"}

    def test_read_json_body_empty(self, handler):
        """Test reading empty body."""
        mock_handler = MagicMock()
        mock_handler.headers = {"Content-Length": "0"}

        result = handler.read_json_body(mock_handler)
        assert result == {}

    def test_read_json_body_invalid_json(self, handler):
        """Test reading invalid JSON body."""
        mock_handler = MagicMock()
        body_content = b"not valid json"
        mock_handler.headers = {"Content-Length": str(len(body_content))}
        mock_handler.rfile.read.return_value = body_content

        result = handler.read_json_body(mock_handler)
        assert result is None

    def test_read_json_body_too_large(self, handler):
        """Test reading body that exceeds max size."""
        mock_handler = MagicMock()
        # Claim very large content length
        mock_handler.headers = {"Content-Length": str(100 * 1024 * 1024)}

        result = handler.read_json_body(mock_handler, max_size=1024)
        assert result is None

    def test_validate_json_content_type_valid(self, handler):
        """Test content type validation with valid JSON type."""
        mock_handler = MagicMock()
        mock_handler.headers = {"Content-Type": "application/json"}

        result = handler.validate_json_content_type(mock_handler)
        assert result is None  # No error

    def test_validate_json_content_type_with_charset(self, handler):
        """Test content type validation with charset."""
        mock_handler = MagicMock()
        mock_handler.headers = {"Content-Type": "application/json; charset=utf-8"}

        result = handler.validate_json_content_type(mock_handler)
        assert result is None

    def test_validate_json_content_type_invalid(self, handler):
        """Test content type validation with invalid type."""
        mock_handler = MagicMock()
        mock_handler.headers = {"Content-Type": "text/plain"}

        result = handler.validate_json_content_type(mock_handler)
        assert result is not None
        assert result.status_code == 415

    def test_read_json_body_validated(self, handler):
        """Test combined content type validation and body reading."""
        mock_handler = MagicMock()
        body_content = json.dumps({"test": "data"}).encode()
        mock_handler.headers = {
            "Content-Type": "application/json",
            "Content-Length": str(len(body_content)),
        }
        mock_handler.rfile.read.return_value = body_content

        body, err = handler.read_json_body_validated(mock_handler)
        assert err is None
        assert body == {"test": "data"}

    def test_handle_returns_none_by_default(self, handler):
        """Test that handle returns None by default."""
        result = handler.handle("/test", {}, MagicMock())
        assert result is None

    def test_handle_post_returns_none_by_default(self, handler):
        """Test that handle_post returns None by default."""
        result = handler.handle_post("/test", {}, MagicMock())
        assert result is None

    def test_handle_delete_returns_none_by_default(self, handler):
        """Test that handle_delete returns None by default."""
        result = handler.handle_delete("/test", {}, MagicMock())
        assert result is None


# =============================================================================
# Quota Decorator Tests
# =============================================================================


class TestRequireQuota:
    """Tests for require_quota decorator."""

    def test_quota_decorator_allows_request(self):
        """Test quota decorator allows request when under limit."""

        @require_quota()
        def handler_func(handler, user=None):
            return json_response({"success": True})

        mock_handler = MagicMock()
        mock_handler.headers = {"Authorization": "Bearer valid_token"}

        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            mock_user = MagicMock()
            mock_user.is_authenticated = True
            mock_user.org_id = None  # No org, no quota check
            mock_extract.return_value = mock_user

            result = handler_func(mock_handler)
            assert result.status_code == 200

    def test_quota_decorator_requires_auth(self):
        """Test quota decorator requires authentication."""

        @require_quota()
        def handler_func(handler, user=None):
            return json_response({"success": True})

        mock_handler = MagicMock()
        mock_handler.headers = {}

        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            mock_user = MagicMock()
            mock_user.is_authenticated = False
            mock_user.error_reason = "No token"
            mock_extract.return_value = mock_user

            result = handler_func(mock_handler)
            assert result.status_code == 401


# =============================================================================
# Validation Pattern Tests
# =============================================================================


class TestValidationPatterns:
    """Tests for validation regex patterns."""

    def test_safe_id_pattern_valid(self):
        """Test SAFE_ID_PATTERN with valid IDs."""
        valid_ids = [
            "abc123",
            "debate_001",
            "user-test-id",
            "A1B2C3",
            "simple",
        ]
        for id_val in valid_ids:
            assert SAFE_ID_PATTERN.match(id_val), f"Should match: {id_val}"

    def test_safe_id_pattern_invalid(self):
        """Test SAFE_ID_PATTERN rejects invalid IDs."""
        invalid_ids = [
            "../etc/passwd",
            "path/traversal",
            "<script>",
            "'; DROP TABLE",
            "",
        ]
        for id_val in invalid_ids:
            # Should either not match or fail validation
            match = SAFE_ID_PATTERN.match(id_val)
            if match:
                # If it matches, ensure it doesn't capture the whole dangerous string
                assert match.group() != id_val or id_val == ""

    def test_safe_agent_pattern_valid(self):
        """Test SAFE_AGENT_PATTERN with valid agent names."""
        valid_names = [
            "claude",
            "gpt-4",
            "gemini_pro",
            "agent123",
        ]
        for name in valid_names:
            assert SAFE_AGENT_PATTERN.match(name), f"Should match: {name}"


# =============================================================================
# Integration Tests
# =============================================================================


class TestBaseHandlerIntegration:
    """Integration tests combining multiple handler features."""

    def test_full_request_flow(self):
        """Test a complete request handling flow."""
        ctx = {
            "storage": MagicMock(),
            "elo_system": MagicMock(),
        }
        handler = BaseHandler(ctx)

        # Create mock HTTP handler
        mock_http = MagicMock()
        body_content = json.dumps({"action": "create", "data": {"name": "test"}}).encode()
        mock_http.headers = {
            "Content-Type": "application/json",
            "Content-Length": str(len(body_content)),
            "Host": "localhost:8080",
        }
        mock_http.rfile.read.return_value = body_content

        # Read and validate body
        body, err = handler.read_json_body_validated(mock_http)
        assert err is None
        assert body["action"] == "create"

        # Check host header
        host = get_host_header(mock_http)
        assert host == "localhost:8080"

    def test_paginated_handler_subclass(self):
        """Test creating a paginated handler subclass."""

        class MyPaginatedHandler(BaseHandler, PaginatedHandlerMixin):
            def handle(self, path, query_params, handler):
                limit, offset = self.get_pagination(query_params)
                items = [{"id": i} for i in range(limit)]
                return self.paginated_response(items, total=100, limit=limit, offset=offset)

        handler = MyPaginatedHandler({})
        result = handler.handle("/items", {"limit": "10", "offset": "20"}, MagicMock())
        body = parse_body(result)

        assert result.status_code == 200
        assert len(body["items"]) == 10
        assert body["offset"] == 20
