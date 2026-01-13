"""
Tests for the base handler utilities and classes.

Covers:
- Utility functions (get_host_header, get_agent_name, agent_to_dict)
- Response builders (json_response, error_response, safe_error_response)
- require_quota decorator
- Handler mixins (PaginatedHandlerMixin, CachedHandlerMixin, AuthenticatedHandlerMixin)
- BaseHandler class methods
"""

from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass
from typing import Any, Optional
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from aragora.server.handlers.base import (
    get_host_header,
    get_agent_name,
    agent_to_dict,
    safe_error_response,
    feature_unavailable_response,
    require_quota,
    json_response,
    error_response,
    PaginatedHandlerMixin,
    CachedHandlerMixin,
    AuthenticatedHandlerMixin,
    BaseHandler,
    SAFE_ID_PATTERN,
    SAFE_AGENT_PATTERN,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_handler():
    """Create a mock HTTP request handler."""
    handler = MagicMock()
    handler.headers = {"Host": "example.com:8080", "Content-Type": "application/json"}
    handler.rfile = io.BytesIO(b'{}')
    return handler


@pytest.fixture
def mock_handler_with_body():
    """Create a mock handler with JSON body."""
    def create(body: dict):
        handler = MagicMock()
        body_bytes = json.dumps(body).encode()
        handler.headers = {
            "Host": "example.com:8080",
            "Content-Type": "application/json",
            "Content-Length": str(len(body_bytes)),
        }
        handler.rfile = io.BytesIO(body_bytes)
        return handler
    return create


@pytest.fixture
def server_context():
    """Create a basic server context."""
    return {
        "storage": MagicMock(),
        "elo_system": MagicMock(),
        "debate_embeddings": MagicMock(),
        "critique_store": MagicMock(),
        "nomic_dir": "/tmp/nomic",
    }


@dataclass
class MockAgent:
    """Mock agent object for testing."""
    name: str = "test-agent"
    agent_name: str = "test-agent"
    elo: int = 1600
    wins: int = 10
    losses: int = 5
    draws: int = 2
    win_rate: float = 0.67
    games_played: int = 17
    matches: int = 17


@dataclass
class MockUserContext:
    """Mock user authentication context."""
    is_authenticated: bool = True
    user_id: str = "user_123"
    org_id: str = "org_456"
    email: str = "test@example.com"
    error_reason: Optional[str] = None


@dataclass
class MockOrganization:
    """Mock organization for quota tests."""
    debates_used_this_month: int = 5
    is_at_limit: bool = False

    class limits:
        debates_per_month = 100

    class tier:
        value = "professional"


# =============================================================================
# Utility Function Tests
# =============================================================================


class TestGetHostHeader:
    """Tests for get_host_header utility."""

    def test_returns_host_header(self, mock_handler):
        """Test extracting Host header."""
        result = get_host_header(mock_handler)
        assert result == "example.com:8080"

    def test_returns_default_when_handler_none(self):
        """Test returns default when handler is None."""
        result = get_host_header(None)
        assert result == "localhost:8080"

    def test_returns_custom_default(self):
        """Test custom default value."""
        result = get_host_header(None, default="custom:9000")
        assert result == "custom:9000"

    def test_returns_default_when_no_host_header(self):
        """Test returns default when Host header missing."""
        handler = MagicMock()
        handler.headers = {}
        result = get_host_header(handler)
        assert result == "localhost:8080"

    def test_returns_default_when_no_headers_attr(self):
        """Test returns default when handler has no headers attribute."""
        handler = MagicMock(spec=[])
        result = get_host_header(handler)
        assert result == "localhost:8080"


class TestGetAgentName:
    """Tests for get_agent_name utility."""

    def test_extracts_name_from_dict(self):
        """Test extracting name from dict."""
        agent = {"name": "claude"}
        assert get_agent_name(agent) == "claude"

    def test_prefers_agent_name_key(self):
        """Test agent_name key takes precedence."""
        agent = {"agent_name": "claude-3", "name": "claude"}
        assert get_agent_name(agent) == "claude-3"

    def test_extracts_from_object(self):
        """Test extracting name from object."""
        agent = MockAgent(name="gpt-4", agent_name="gpt-4")
        assert get_agent_name(agent) == "gpt-4"

    def test_prefers_agent_name_attr(self):
        """Test agent_name attribute takes precedence."""
        agent = MockAgent(agent_name="gpt-4-turbo", name="gpt-4")
        assert get_agent_name(agent) == "gpt-4-turbo"

    def test_returns_none_for_none(self):
        """Test returns None for None input."""
        assert get_agent_name(None) is None

    def test_returns_none_for_empty_dict(self):
        """Test returns None for empty dict."""
        assert get_agent_name({}) is None


class TestAgentToDict:
    """Tests for agent_to_dict utility."""

    def test_returns_empty_for_none(self):
        """Test returns empty dict for None."""
        assert agent_to_dict(None) == {}

    def test_copies_dict_input(self):
        """Test copies dict input."""
        agent = {"name": "claude", "elo": 1700}
        result = agent_to_dict(agent)
        assert result == agent
        assert result is not agent  # Should be a copy

    def test_extracts_from_object(self):
        """Test extracting fields from object."""
        agent = MockAgent()
        result = agent_to_dict(agent)

        assert result["name"] == "test-agent"
        assert result["agent_name"] == "test-agent"
        assert result["elo"] == 1600
        assert result["wins"] == 10
        assert result["losses"] == 5
        assert result["draws"] == 2
        assert result["win_rate"] == 0.67
        assert result["games"] == 17
        assert result["matches"] == 17

    def test_exclude_name(self):
        """Test excluding name fields."""
        agent = MockAgent()
        result = agent_to_dict(agent, include_name=False)

        assert "name" not in result
        assert "agent_name" not in result
        assert result["elo"] == 1600

    def test_default_values_for_missing_attrs(self):
        """Test defaults for missing attributes."""
        class MinimalAgent:
            pass

        agent = MinimalAgent()
        result = agent_to_dict(agent)

        assert result["elo"] == 1500
        assert result["wins"] == 0
        assert result["losses"] == 0
        assert result["draws"] == 0
        assert result["win_rate"] == 0.0
        assert result["games"] == 0
        assert result["matches"] == 0


# =============================================================================
# Response Builder Tests
# =============================================================================


class TestJsonResponse:
    """Tests for json_response builder."""

    def test_creates_response_tuple(self):
        """Test creating response as HandlerResult."""
        result = json_response({"status": "ok"})

        assert json.loads(result.body) == {"status": "ok"}
        assert result.status_code == 200
        assert result.content_type == "application/json"

    def test_custom_status_code(self):
        """Test custom status code."""
        result = json_response({"created": True}, status=201)
        assert result.status_code == 201


class TestErrorResponse:
    """Tests for error_response builder."""

    def test_creates_error_tuple(self):
        """Test creating error response."""
        result = error_response("Something went wrong", 400)

        parsed = json.loads(result.body)
        assert parsed["error"] == "Something went wrong"
        assert result.status_code == 400
        assert result.content_type == "application/json"

    def test_default_status_400(self):
        """Test default status is 400."""
        result = error_response("Bad request")
        assert result.status_code == 400


class TestSafeErrorResponse:
    """Tests for safe_error_response with sanitization."""

    def test_sanitizes_exception(self):
        """Test exception message is sanitized."""
        exc = ValueError("Secret database path: /var/db/secrets.db")
        result = safe_error_response(exc, "database lookup")

        parsed = json.loads(result.body)

        # Should not expose path details
        assert "/var/db" not in parsed.get("error", "")
        assert result.status_code == 500

    def test_includes_trace_id(self):
        """Test includes trace ID."""
        exc = ValueError("test error")
        result = safe_error_response(exc, "test operation")

        parsed = json.loads(result.body)

        # trace_id is inside the nested error object
        assert "trace_id" in parsed.get("error", {})

    def test_extracts_trace_from_handler(self):
        """Test extracts trace ID from handler."""
        handler = MagicMock()
        handler.trace_id = "test-trace-123"

        exc = ValueError("test error")
        result = safe_error_response(exc, "test", handler=handler)

        parsed = json.loads(result.body)
        # trace_id is inside the nested error object
        assert parsed.get("error", {}).get("trace_id") == "test-trace-123"

    def test_custom_status(self):
        """Test custom status code."""
        exc = ValueError("not found")
        result = safe_error_response(exc, "lookup", status=404)

        assert result.status_code == 404


class TestFeatureUnavailableResponse:
    """Tests for feature_unavailable_response."""

    def test_returns_503_status(self):
        """Test returns 503 Service Unavailable."""
        from aragora.server.handlers.utils.responses import HandlerResult
        with patch("aragora.server.handlers.features.feature_unavailable_response") as mock:
            mock.return_value = HandlerResult(
                status_code=503,
                content_type="application/json",
                body=b'{"error": "Feature unavailable"}'
            )
            result = feature_unavailable_response("pulse")

            assert result.status_code == 503

    def test_passes_feature_id(self):
        """Test passes feature ID to underlying function."""
        from aragora.server.handlers.utils.responses import HandlerResult
        with patch("aragora.server.handlers.features.feature_unavailable_response") as mock:
            mock.return_value = HandlerResult(
                status_code=503,
                content_type="application/json",
                body=b'{}'
            )
            feature_unavailable_response("genesis")
            mock.assert_called_once_with("genesis", None)


# =============================================================================
# Require Quota Decorator Tests
# =============================================================================


class TestRequireQuota:
    """Tests for require_quota decorator."""

    def test_allows_when_under_quota(self):
        """Test allows request when under quota."""
        mock_user = MockUserContext()
        mock_org = MockOrganization()
        mock_org.debates_used_this_month = 5
        mock_org.is_at_limit = False

        mock_user_store = MagicMock()
        mock_user_store.get_organization_by_id.return_value = mock_org

        @require_quota()
        def handler_func(handler, user=None):
            return json_response({"success": True})

        mock_handler = MagicMock()
        mock_handler.user_store = mock_user_store

        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            mock_extract.return_value = mock_user
            result = handler_func(mock_handler)

        assert result.status_code == 200

    def test_blocks_when_at_limit(self):
        """Test blocks request when at quota limit."""
        mock_user = MockUserContext()
        mock_org = MockOrganization()
        mock_org.is_at_limit = True

        mock_user_store = MagicMock()
        mock_user_store.get_organization_by_id.return_value = mock_org

        @require_quota()
        def handler_func(handler, user=None):
            return json_response({"success": True})

        mock_handler = MagicMock()
        mock_handler.user_store = mock_user_store

        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            mock_extract.return_value = mock_user
            result = handler_func(mock_handler)

        parsed = json.loads(result.body)
        assert result.status_code == 429
        assert parsed["code"] == "quota_exceeded"

    def test_blocks_when_insufficient_quota(self):
        """Test blocks when batch size exceeds remaining quota."""
        mock_user = MockUserContext()
        mock_org = MockOrganization()
        mock_org.debates_used_this_month = 95
        mock_org.is_at_limit = False

        mock_user_store = MagicMock()
        mock_user_store.get_organization_by_id.return_value = mock_org

        @require_quota(debate_count=10)
        def handler_func(handler, user=None):
            return json_response({"success": True})

        mock_handler = MagicMock()
        mock_handler.user_store = mock_user_store

        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            mock_extract.return_value = mock_user
            result = handler_func(mock_handler)

        parsed = json.loads(result.body)
        assert result.status_code == 429
        assert parsed["code"] == "quota_insufficient"
        assert parsed["requested"] == 10
        assert parsed["remaining"] == 5

    def test_requires_authentication(self):
        """Test requires authentication."""
        mock_user = MockUserContext(is_authenticated=False)

        @require_quota()
        def handler_func(handler, user=None):
            return json_response({"success": True})

        mock_handler = MagicMock()
        mock_handler.user_store = MagicMock()

        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            mock_extract.return_value = mock_user
            result = handler_func(mock_handler)

        assert result.status_code == 401

    def test_increments_usage_on_success(self):
        """Test increments organization usage on success."""
        mock_user = MockUserContext()
        mock_org = MockOrganization()
        mock_org.debates_used_this_month = 5

        mock_user_store = MagicMock()
        mock_user_store.get_organization_by_id.return_value = mock_org

        @require_quota()
        def handler_func(handler, user=None):
            return json_response({"success": True})

        mock_handler = MagicMock()
        mock_handler.user_store = mock_user_store

        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            mock_extract.return_value = mock_user
            handler_func(mock_handler)

        mock_user_store.increment_usage.assert_called_once_with("org_456", 1)

    def test_does_not_increment_on_failure(self):
        """Test does not increment usage on error response."""
        mock_user = MockUserContext()
        mock_org = MockOrganization()

        mock_user_store = MagicMock()
        mock_user_store.get_organization_by_id.return_value = mock_org

        @require_quota()
        def handler_func(handler, user=None):
            return error_response("Something failed", 400)

        mock_handler = MagicMock()
        mock_handler.user_store = mock_user_store

        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            mock_extract.return_value = mock_user
            result = handler_func(mock_handler)

        assert result.status_code == 400
        mock_user_store.increment_usage.assert_not_called()


# =============================================================================
# PaginatedHandlerMixin Tests
# =============================================================================


class TestPaginatedHandlerMixin:
    """Tests for PaginatedHandlerMixin."""

    @pytest.fixture
    def mixin(self):
        """Create mixin instance."""
        class Handler(PaginatedHandlerMixin):
            pass
        return Handler()

    def test_get_pagination_defaults(self, mixin):
        """Test default pagination values."""
        limit, offset = mixin.get_pagination({})
        assert limit == 20
        assert offset == 0

    def test_get_pagination_from_params(self, mixin):
        """Test extracting pagination from query params."""
        limit, offset = mixin.get_pagination({"limit": "50", "offset": "100"})
        assert limit == 50
        assert offset == 100

    def test_get_pagination_clamps_limit(self, mixin):
        """Test limit is clamped to max."""
        limit, offset = mixin.get_pagination({"limit": "1000"})
        assert limit == 100  # MAX_LIMIT

    def test_get_pagination_clamps_minimum(self, mixin):
        """Test minimum values are enforced."""
        limit, offset = mixin.get_pagination({"limit": "-5", "offset": "-10"})
        assert limit == 1  # Min limit
        assert offset == 0  # Min offset

    def test_get_pagination_custom_limits(self, mixin):
        """Test custom default and max limits."""
        limit, offset = mixin.get_pagination(
            {}, default_limit=10, max_limit=50
        )
        assert limit == 10

        limit, offset = mixin.get_pagination(
            {"limit": "200"}, max_limit=50
        )
        assert limit == 50

    def test_paginated_response(self, mixin):
        """Test paginated response format."""
        items = [{"id": 1}, {"id": 2}]
        result = mixin.paginated_response(items, total=100, limit=20, offset=0)

        parsed = json.loads(result.body)

        assert parsed["items"] == items
        assert parsed["total"] == 100
        assert parsed["limit"] == 20
        assert parsed["offset"] == 0
        assert parsed["has_more"] is True

    def test_paginated_response_no_more(self, mixin):
        """Test has_more is False when at end."""
        items = [{"id": 1}]
        result = mixin.paginated_response(items, total=1, limit=20, offset=0)

        parsed = json.loads(result.body)
        assert parsed["has_more"] is False

    def test_paginated_response_custom_key(self, mixin):
        """Test custom items key."""
        items = [{"name": "a"}]
        result = mixin.paginated_response(
            items, total=1, limit=20, offset=0, items_key="results"
        )

        parsed = json.loads(result.body)
        assert "results" in parsed
        assert parsed["results"] == items


# =============================================================================
# CachedHandlerMixin Tests
# =============================================================================


class TestCachedHandlerMixin:
    """Tests for CachedHandlerMixin."""

    @pytest.fixture
    def mixin(self):
        """Create mixin instance."""
        class Handler(CachedHandlerMixin):
            pass
        return Handler()

    def test_cached_response_calls_generator(self, mixin):
        """Test generator is called on cache miss."""
        generator = MagicMock(return_value={"data": "test"})

        with patch("aragora.server.handlers.base.get_handler_cache") as mock_cache:
            mock_cache.return_value.get.return_value = (False, None)

            result = mixin.cached_response("key1", 60, generator)

        generator.assert_called_once()
        assert result == {"data": "test"}

    def test_cached_response_uses_cache(self, mixin):
        """Test cached value is used on hit."""
        generator = MagicMock()
        cached_value = {"cached": True}

        with patch("aragora.server.handlers.base.get_handler_cache") as mock_cache:
            mock_cache.return_value.get.return_value = (True, cached_value)

            result = mixin.cached_response("key1", 60, generator)

        generator.assert_not_called()
        assert result == cached_value

    def test_cached_response_stores_result(self, mixin):
        """Test generated value is stored in cache."""
        generator = MagicMock(return_value={"fresh": True})
        mock_cache_instance = MagicMock()
        mock_cache_instance.get.return_value = (False, None)

        with patch("aragora.server.handlers.base.get_handler_cache") as mock_cache:
            mock_cache.return_value = mock_cache_instance
            mixin.cached_response("key2", 120, generator)

        mock_cache_instance.set.assert_called_once_with("key2", {"fresh": True})

    @pytest.mark.asyncio
    async def test_async_cached_response(self, mixin):
        """Test async cached response."""
        async def async_generator():
            return {"async_data": True}

        with patch("aragora.server.handlers.base.get_handler_cache") as mock_cache:
            mock_cache.return_value.get.return_value = (False, None)

            result = await mixin.async_cached_response("key3", 60, async_generator)

        assert result == {"async_data": True}


# =============================================================================
# AuthenticatedHandlerMixin Tests
# =============================================================================


class TestAuthenticatedHandlerMixin:
    """Tests for AuthenticatedHandlerMixin."""

    @pytest.fixture
    def mixin(self):
        """Create mixin instance."""
        class Handler(AuthenticatedHandlerMixin):
            pass
        return Handler()

    def test_require_auth_authenticated(self, mixin):
        """Test returns user context when authenticated."""
        mock_user = MockUserContext()

        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            mock_extract.return_value = mock_user
            mock_handler = MagicMock()

            result = mixin.require_auth(mock_handler)

        assert result == mock_user

    def test_require_auth_not_authenticated(self, mixin):
        """Test returns error when not authenticated."""
        mock_user = MockUserContext(is_authenticated=False)

        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            mock_extract.return_value = mock_user
            mock_handler = MagicMock()

            result = mixin.require_auth(mock_handler)

        # Returns error response as HandlerResult
        assert result.status_code == 401


# =============================================================================
# BaseHandler Tests
# =============================================================================


class TestBaseHandler:
    """Tests for BaseHandler class."""

    @pytest.fixture
    def handler(self, server_context):
        """Create BaseHandler instance."""
        return BaseHandler(server_context)

    # === Initialization ===

    def test_init_stores_context(self, handler, server_context):
        """Test context is stored."""
        assert handler.ctx == server_context

    # === Path Parameter Extraction ===

    def test_extract_path_param_success(self, handler):
        """Test successful path param extraction."""
        value, err = handler.extract_path_param(
            "/api/debates/abc123/status", 2, "debate_id"
        )
        assert value == "abc123"
        assert err is None

    def test_extract_path_param_missing(self, handler):
        """Test error for missing path segment."""
        value, err = handler.extract_path_param(
            "/api/debates", 5, "debate_id"
        )
        assert value is None
        assert err.status_code == 400
        assert "Missing debate_id" in json.loads(err.body)["error"]

    def test_extract_path_param_empty(self, handler):
        """Test error for empty segment."""
        value, err = handler.extract_path_param(
            "/api/debates//status", 2, "debate_id"
        )
        assert value is None
        assert err.status_code == 400

    def test_extract_path_param_invalid_pattern(self, handler):
        """Test error for invalid pattern match."""
        value, err = handler.extract_path_param(
            "/api/debates/../../../etc/passwd/status", 2, "debate_id"
        )
        assert value is None
        assert err.status_code == 400

    def test_extract_path_param_custom_pattern(self, handler):
        """Test custom validation pattern."""
        # Agent names allow hyphens
        value, err = handler.extract_path_param(
            "/api/agents/claude-3", 2, "agent_name", SAFE_AGENT_PATTERN
        )
        assert value == "claude-3"
        assert err is None

    def test_extract_path_params_multiple(self, handler):
        """Test extracting multiple params."""
        params, err = handler.extract_path_params(
            "/api/agents/compare/claude/gpt4",
            [
                (3, "agent_a", SAFE_AGENT_PATTERN),
                (4, "agent_b", SAFE_AGENT_PATTERN),
            ]
        )
        assert err is None
        assert params == {"agent_a": "claude", "agent_b": "gpt4"}

    def test_extract_path_params_fails_on_first_error(self, handler):
        """Test returns first error."""
        params, err = handler.extract_path_params(
            "/api/short",
            [
                (2, "first", None),  # exists
                (10, "second", None),  # missing
            ]
        )
        assert params is None
        assert err is not None

    # === Context Accessors ===

    def test_get_storage(self, handler, server_context):
        """Test get_storage accessor."""
        assert handler.get_storage() == server_context["storage"]

    def test_get_elo_system_from_ctx(self, handler, server_context):
        """Test get_elo_system from context."""
        assert handler.get_elo_system() == server_context["elo_system"]

    def test_get_elo_system_from_class_attr(self, handler):
        """Test get_elo_system from class attribute."""
        mock_elo = MagicMock()
        BaseHandler.elo_system = mock_elo
        try:
            assert handler.get_elo_system() == mock_elo
        finally:
            del BaseHandler.elo_system

    def test_get_debate_embeddings(self, handler, server_context):
        """Test get_debate_embeddings accessor."""
        assert handler.get_debate_embeddings() == server_context["debate_embeddings"]

    def test_get_critique_store(self, handler, server_context):
        """Test get_critique_store accessor."""
        assert handler.get_critique_store() == server_context["critique_store"]

    def test_get_nomic_dir(self, handler, server_context):
        """Test get_nomic_dir accessor."""
        assert handler.get_nomic_dir() == server_context["nomic_dir"]

    # === Authentication ===

    def test_get_current_user_authenticated(self, handler):
        """Test get_current_user when authenticated."""
        mock_user = MockUserContext()
        mock_handler = MagicMock()

        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            mock_extract.return_value = mock_user
            result = handler.get_current_user(mock_handler)

        assert result == mock_user

    def test_get_current_user_not_authenticated(self, handler):
        """Test get_current_user returns None when not authenticated."""
        mock_user = MockUserContext(is_authenticated=False)
        mock_handler = MagicMock()

        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            mock_extract.return_value = mock_user
            result = handler.get_current_user(mock_handler)

        assert result is None

    def test_require_auth_or_error_authenticated(self, handler):
        """Test require_auth_or_error when authenticated."""
        mock_user = MockUserContext()
        mock_handler = MagicMock()

        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            mock_extract.return_value = mock_user
            user, err = handler.require_auth_or_error(mock_handler)

        assert user == mock_user
        assert err is None

    def test_require_auth_or_error_not_authenticated(self, handler):
        """Test require_auth_or_error when not authenticated."""
        mock_user = MockUserContext(is_authenticated=False)
        mock_handler = MagicMock()

        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            mock_extract.return_value = mock_user
            user, err = handler.require_auth_or_error(mock_handler)

        assert user is None
        assert err.status_code == 401

    # === JSON Body Parsing ===

    def test_read_json_body_valid(self, handler, mock_handler_with_body):
        """Test reading valid JSON body."""
        mock_h = mock_handler_with_body({"key": "value"})
        result = handler.read_json_body(mock_h)
        assert result == {"key": "value"}

    def test_read_json_body_empty(self, handler):
        """Test reading empty body."""
        mock_h = MagicMock()
        mock_h.headers = {"Content-Length": "0"}
        result = handler.read_json_body(mock_h)
        assert result == {}

    def test_read_json_body_invalid(self, handler):
        """Test reading invalid JSON."""
        mock_h = MagicMock()
        mock_h.headers = {"Content-Length": "5"}
        mock_h.rfile = io.BytesIO(b"not json")
        result = handler.read_json_body(mock_h)
        assert result is None

    def test_read_json_body_too_large(self, handler):
        """Test body exceeds max size."""
        mock_h = MagicMock()
        mock_h.headers = {"Content-Length": "999999999"}
        result = handler.read_json_body(mock_h, max_size=1000)
        assert result is None

    def test_validate_content_length_valid(self, handler):
        """Test valid content length."""
        mock_h = MagicMock()
        mock_h.headers = {"Content-Length": "500"}
        result = handler.validate_content_length(mock_h)
        assert result == 500

    def test_validate_content_length_invalid(self, handler):
        """Test invalid content length."""
        mock_h = MagicMock()
        mock_h.headers = {"Content-Length": "not-a-number"}
        result = handler.validate_content_length(mock_h)
        assert result is None

    def test_validate_content_length_negative(self, handler):
        """Test negative content length."""
        mock_h = MagicMock()
        mock_h.headers = {"Content-Length": "-100"}
        result = handler.validate_content_length(mock_h)
        assert result is None

    def test_validate_content_length_too_large(self, handler):
        """Test content length exceeds max."""
        mock_h = MagicMock()
        mock_h.headers = {"Content-Length": "999999999"}
        result = handler.validate_content_length(mock_h, max_size=1000)
        assert result is None

    def test_validate_json_content_type_valid(self, handler):
        """Test valid JSON content type."""
        mock_h = MagicMock()
        mock_h.headers = {"Content-Type": "application/json"}
        result = handler.validate_json_content_type(mock_h)
        assert result is None

    def test_validate_json_content_type_with_charset(self, handler):
        """Test JSON content type with charset."""
        mock_h = MagicMock()
        mock_h.headers = {"Content-Type": "application/json; charset=utf-8"}
        result = handler.validate_json_content_type(mock_h)
        assert result is None

    def test_validate_json_content_type_text_json(self, handler):
        """Test text/json is accepted."""
        mock_h = MagicMock()
        mock_h.headers = {"Content-Type": "text/json"}
        result = handler.validate_json_content_type(mock_h)
        assert result is None

    def test_validate_json_content_type_invalid(self, handler):
        """Test invalid content type returns error."""
        mock_h = MagicMock()
        mock_h.headers = {"Content-Type": "text/plain"}
        result = handler.validate_json_content_type(mock_h)

        assert result.status_code == 415
        assert "Unsupported Content-Type" in json.loads(result.body)["error"]

    def test_validate_json_content_type_empty_with_body(self, handler):
        """Test missing content type with body returns error."""
        mock_h = MagicMock()
        mock_h.headers = {"Content-Length": "100"}
        result = handler.validate_json_content_type(mock_h)

        assert result.status_code == 415

    def test_validate_json_content_type_empty_without_body(self, handler):
        """Test empty content type with empty body is OK."""
        mock_h = MagicMock()
        mock_h.headers = {"Content-Length": "0"}
        result = handler.validate_json_content_type(mock_h)
        assert result is None

    def test_read_json_body_validated_success(self, handler, mock_handler_with_body):
        """Test validated JSON read success."""
        mock_h = mock_handler_with_body({"test": "data"})
        body, err = handler.read_json_body_validated(mock_h)

        assert err is None
        assert body == {"test": "data"}

    def test_read_json_body_validated_content_type_error(self, handler):
        """Test validated JSON read with wrong content type."""
        mock_h = MagicMock()
        mock_h.headers = {"Content-Type": "text/plain", "Content-Length": "10"}

        body, err = handler.read_json_body_validated(mock_h)

        assert body is None
        assert err.status_code == 415

    def test_read_json_body_validated_parse_error(self, handler):
        """Test validated JSON read with invalid JSON."""
        mock_h = MagicMock()
        mock_h.headers = {"Content-Type": "application/json", "Content-Length": "10"}
        mock_h.rfile = io.BytesIO(b"not json!!")

        body, err = handler.read_json_body_validated(mock_h)

        assert body is None
        assert err.status_code == 400

    # === Default HTTP Method Handlers ===

    def test_handle_returns_none(self, handler):
        """Test default handle returns None."""
        result = handler.handle("/api/test", {}, MagicMock())
        assert result is None

    def test_handle_post_returns_none(self, handler):
        """Test default handle_post returns None."""
        result = handler.handle_post("/api/test", {}, MagicMock())
        assert result is None

    def test_handle_delete_returns_none(self, handler):
        """Test default handle_delete returns None."""
        result = handler.handle_delete("/api/test", {}, MagicMock())
        assert result is None

    def test_handle_patch_returns_none(self, handler):
        """Test default handle_patch returns None."""
        result = handler.handle_patch("/api/test", {}, MagicMock())
        assert result is None

    def test_handle_put_returns_none(self, handler):
        """Test default handle_put returns None."""
        result = handler.handle_put("/api/test", {}, MagicMock())
        assert result is None


# =============================================================================
# Integration Tests
# =============================================================================


class TestBaseHandlerIntegration:
    """Integration tests combining multiple BaseHandler features."""

    def test_combined_mixin_handler(self, server_context):
        """Test handler combining multiple mixins."""

        class CombinedHandler(BaseHandler, PaginatedHandlerMixin, CachedHandlerMixin):
            def handle(self, path, query_params, handler):
                # Extract path param
                item_id, err = self.extract_path_param(path, 2, "item_id")
                if err:
                    return err

                # Get pagination
                limit, offset = self.get_pagination(query_params)

                # Return paginated response
                items = [{"id": i} for i in range(limit)]
                return self.paginated_response(items, total=100, limit=limit, offset=offset)

        handler = CombinedHandler(server_context)
        mock_http = MagicMock()
        mock_http.headers = {}

        result = handler.handle("/api/items/item123", {"limit": "5"}, mock_http)

        parsed = json.loads(result.body)

        assert result.status_code == 200
        assert len(parsed["items"]) == 5
        assert parsed["limit"] == 5
        assert parsed["has_more"] is True

    def test_handler_with_auth_and_body(self, server_context):
        """Test handler with authentication and JSON body."""

        class AuthHandler(BaseHandler, AuthenticatedHandlerMixin):
            def handle_post(self, path, query_params, handler):
                # Require auth
                user = self.get_current_user(handler)
                if not user:
                    return error_response("Unauthorized", 401)

                # Parse body
                body, err = self.read_json_body_validated(handler)
                if err:
                    return err

                return json_response({
                    "user_id": user.user_id,
                    "received": body,
                })

        handler = AuthHandler(server_context)
        mock_user = MockUserContext()

        # Create mock HTTP handler with body
        body_bytes = json.dumps({"action": "create"}).encode()
        mock_http = MagicMock()
        mock_http.headers = {
            "Content-Type": "application/json",
            "Content-Length": str(len(body_bytes)),
        }
        mock_http.rfile = io.BytesIO(body_bytes)

        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            mock_extract.return_value = mock_user
            result = handler.handle_post("/api/resource", {}, mock_http)

        parsed = json.loads(result.body)

        assert result.status_code == 200
        assert parsed["user_id"] == "user_123"
        assert parsed["received"]["action"] == "create"


class TestBaseHandlerSubclassing:
    """Tests for properly subclassing BaseHandler."""

    def test_custom_handler_with_storage(self, server_context):
        """Test custom handler accessing storage."""

        class DebateHandler(BaseHandler):
            def handle(self, path, query_params, handler):
                storage = self.get_storage()
                if storage:
                    return json_response({"storage": "available"})
                return error_response("No storage", 503)

        handler = DebateHandler(server_context)
        result = handler.handle("/api/debates", {}, MagicMock())

        assert result.status_code == 200
        assert json.loads(result.body)["storage"] == "available"

    def test_custom_handler_without_storage(self):
        """Test handler without storage in context."""

        class DebateHandler(BaseHandler):
            def handle(self, path, query_params, handler):
                storage = self.get_storage()
                if storage:
                    return json_response({"storage": "available"})
                return error_response("No storage", 503)

        handler = DebateHandler({})  # Empty context
        result = handler.handle("/api/debates", {}, MagicMock())

        assert result.status_code == 503
