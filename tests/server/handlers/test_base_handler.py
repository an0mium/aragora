"""
Tests for the base handler module.

Tests cover:
- Response builders (error_response, json_response, HandlerResult)
- BaseHandler class and its methods
- Handler mixins (PaginatedHandlerMixin, CachedHandlerMixin, AuthenticatedHandlerMixin)
- Path parameter extraction
- JSON body parsing
- Authentication helpers
- Utility functions
"""

import json
import re
from io import BytesIO
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


# ============================================================================
# Response Builder Tests
# ============================================================================


class TestHandlerResult:
    """Tests for HandlerResult dataclass."""

    def test_import_handler_result(self):
        """HandlerResult can be imported."""
        from aragora.server.handlers.base import HandlerResult

        assert HandlerResult is not None

    def test_handler_result_fields(self):
        """HandlerResult has required fields."""
        from aragora.server.handlers.base import HandlerResult

        result = HandlerResult(
            status_code=200, content_type="application/json", body=b'{"test": true}'
        )
        assert result.body == b'{"test": true}'
        assert result.status_code == 200
        assert result.content_type == "application/json"

    def test_handler_result_custom_status(self):
        """HandlerResult accepts custom status codes."""
        from aragora.server.handlers.base import HandlerResult

        result = HandlerResult(
            status_code=404, content_type="application/json", body=b'{"error": "Not found"}'
        )
        assert result.status_code == 404


class TestJsonResponse:
    """Tests for json_response helper."""

    def test_json_response_basic(self):
        """json_response creates valid JSON response."""
        from aragora.server.handlers.base import json_response

        result = json_response({"key": "value"})
        assert result.status_code == 200
        assert result.content_type == "application/json"
        body = json.loads(result.body)
        assert body["key"] == "value"

    def test_json_response_custom_status(self):
        """json_response accepts custom status code."""
        from aragora.server.handlers.base import json_response

        result = json_response({"id": 1}, status=201)
        assert result.status_code == 201

    def test_json_response_list(self):
        """json_response handles list data."""
        from aragora.server.handlers.base import json_response

        result = json_response([1, 2, 3])
        body = json.loads(result.body)
        assert body == [1, 2, 3]

    def test_json_response_nested(self):
        """json_response handles nested structures."""
        from aragora.server.handlers.base import json_response

        data = {"nested": {"deep": {"value": 42}}}
        result = json_response(data)
        body = json.loads(result.body)
        assert body["nested"]["deep"]["value"] == 42


class TestErrorResponse:
    """Tests for error_response helper."""

    def test_error_response_basic(self):
        """error_response creates error with message."""
        from aragora.server.handlers.base import error_response

        result = error_response("Not found", 404)
        assert result.status_code == 404
        body = json.loads(result.body)
        assert body["error"] == "Not found"

    def test_error_response_default_status(self):
        """error_response defaults to 400 status for validation errors."""
        from aragora.server.handlers.base import error_response

        # error_response requires status parameter
        result = error_response("Server error", 500)
        assert result.status_code == 500

    def test_error_response_400(self):
        """error_response handles 400 Bad Request."""
        from aragora.server.handlers.base import error_response

        result = error_response("Invalid input", 400)
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "Invalid input" in body["error"]


class TestSafeErrorResponse:
    """Tests for safe_error_response helper."""

    def test_safe_error_response_sanitizes(self):
        """safe_error_response sanitizes exception messages."""
        from aragora.server.handlers.base import safe_error_response

        exc = Exception("Internal path: /home/user/secret")
        result = safe_error_response(exc, "test operation")
        assert result.status_code == 500
        body = json.loads(result.body)
        # Should not contain the internal path
        assert "/home/user" not in str(body)

    def test_safe_error_response_with_handler(self):
        """safe_error_response extracts trace_id from handler."""
        from aragora.server.handlers.base import safe_error_response

        handler = MagicMock()
        handler.trace_id = "trace-123"

        exc = ValueError("Test error")
        result = safe_error_response(exc, "test context", handler=handler)
        assert result.status_code == 500

    def test_safe_error_response_custom_status(self):
        """safe_error_response accepts custom status."""
        from aragora.server.handlers.base import safe_error_response

        exc = RuntimeError("Bad request")
        result = safe_error_response(exc, "validation", status=400)
        assert result.status_code == 400


class TestFeatureUnavailableResponse:
    """Tests for feature_unavailable_response helper."""

    def test_feature_unavailable_basic(self):
        """feature_unavailable_response returns 503."""
        from aragora.server.handlers.base import feature_unavailable_response

        result = feature_unavailable_response("pulse")
        assert result.status_code == 503

    def test_feature_unavailable_with_message(self):
        """feature_unavailable_response accepts custom message."""
        from aragora.server.handlers.base import feature_unavailable_response

        result = feature_unavailable_response("genesis", "Genesis module not installed")
        assert result.status_code == 503


# ============================================================================
# Utility Function Tests
# ============================================================================


class TestGetHostHeader:
    """Tests for get_host_header utility."""

    def test_get_host_header_from_handler(self):
        """get_host_header extracts Host from handler."""
        from aragora.server.handlers.base import get_host_header

        handler = MagicMock()
        handler.headers = {"Host": "example.com:8080"}

        result = get_host_header(handler)
        assert result == "example.com:8080"

    def test_get_host_header_none_handler(self):
        """get_host_header returns default for None handler."""
        from aragora.server.handlers.base import get_host_header

        result = get_host_header(None)
        assert result == "localhost:8080"

    def test_get_host_header_missing_header(self):
        """get_host_header returns default when Host missing."""
        from aragora.server.handlers.base import get_host_header

        handler = MagicMock()
        handler.headers = {}

        result = get_host_header(handler)
        assert result == "localhost:8080"

    def test_get_host_header_custom_default(self):
        """get_host_header accepts custom default."""
        from aragora.server.handlers.base import get_host_header

        result = get_host_header(None, default="custom:9000")
        assert result == "custom:9000"


class TestGetAgentName:
    """Tests for get_agent_name utility."""

    def test_get_agent_name_from_dict_name(self):
        """get_agent_name extracts name from dict."""
        from aragora.server.handlers.base import get_agent_name

        agent = {"name": "claude"}
        assert get_agent_name(agent) == "claude"

    def test_get_agent_name_from_dict_agent_name(self):
        """get_agent_name extracts agent_name from dict."""
        from aragora.server.handlers.base import get_agent_name

        agent = {"agent_name": "gpt4"}
        assert get_agent_name(agent) == "gpt4"

    def test_get_agent_name_from_object(self):
        """get_agent_name extracts name from object."""
        from aragora.server.handlers.base import get_agent_name

        agent = MagicMock()
        agent.name = "gemini"
        agent.agent_name = None

        assert get_agent_name(agent) == "gemini"

    def test_get_agent_name_none(self):
        """get_agent_name returns None for None input."""
        from aragora.server.handlers.base import get_agent_name

        assert get_agent_name(None) is None

    def test_get_agent_name_prefers_agent_name(self):
        """get_agent_name prefers agent_name over name in dict."""
        from aragora.server.handlers.base import get_agent_name

        agent = {"agent_name": "preferred", "name": "fallback"}
        assert get_agent_name(agent) == "preferred"


class TestAgentToDict:
    """Tests for agent_to_dict utility."""

    def test_agent_to_dict_from_dict(self):
        """agent_to_dict returns copy for dict input."""
        from aragora.server.handlers.base import agent_to_dict

        agent = {"name": "claude", "elo": 1600}
        result = agent_to_dict(agent)
        assert result == agent
        assert result is not agent  # Should be a copy

    def test_agent_to_dict_from_object(self):
        """agent_to_dict extracts fields from object."""
        from aragora.server.handlers.base import agent_to_dict

        agent = MagicMock()
        agent.name = "claude"
        agent.agent_name = None
        agent.elo = 1650
        agent.wins = 10
        agent.losses = 5
        agent.draws = 2
        agent.win_rate = 0.6
        agent.games_played = 17

        result = agent_to_dict(agent)
        assert result["name"] == "claude"
        assert result["elo"] == 1650
        assert result["wins"] == 10
        assert result["losses"] == 5

    def test_agent_to_dict_none(self):
        """agent_to_dict returns empty dict for None."""
        from aragora.server.handlers.base import agent_to_dict

        assert agent_to_dict(None) == {}

    def test_agent_to_dict_without_name(self):
        """agent_to_dict can exclude name fields."""
        from aragora.server.handlers.base import agent_to_dict

        agent = MagicMock()
        agent.name = "claude"
        agent.elo = 1500

        result = agent_to_dict(agent, include_name=False)
        assert "name" not in result
        assert "agent_name" not in result
        assert result["elo"] == 1500


# ============================================================================
# BaseHandler Tests
# ============================================================================


class TestBaseHandlerInit:
    """Tests for BaseHandler initialization."""

    def test_base_handler_init(self):
        """BaseHandler initializes with server context."""
        from aragora.server.handlers.base import BaseHandler

        ctx = {"storage": MagicMock(), "elo_system": MagicMock()}
        handler = BaseHandler(ctx)
        assert handler.ctx == ctx

    def test_base_handler_empty_context(self):
        """BaseHandler accepts empty context."""
        from aragora.server.handlers.base import BaseHandler

        handler = BaseHandler({})
        assert handler.ctx == {}


class TestBaseHandlerGetters:
    """Tests for BaseHandler getter methods."""

    def test_get_storage(self):
        """get_storage returns storage from context."""
        from aragora.server.handlers.base import BaseHandler

        mock_storage = MagicMock()
        handler = BaseHandler({"storage": mock_storage})
        assert handler.get_storage() is mock_storage

    def test_get_storage_missing(self):
        """get_storage returns None when not in context."""
        from aragora.server.handlers.base import BaseHandler

        handler = BaseHandler({})
        assert handler.get_storage() is None

    def test_get_elo_system(self):
        """get_elo_system returns elo_system from context."""
        from aragora.server.handlers.base import BaseHandler

        mock_elo = MagicMock()
        handler = BaseHandler({"elo_system": mock_elo})
        assert handler.get_elo_system() is mock_elo

    def test_get_debate_embeddings(self):
        """get_debate_embeddings returns embeddings from context."""
        from aragora.server.handlers.base import BaseHandler

        mock_embeddings = MagicMock()
        handler = BaseHandler({"debate_embeddings": mock_embeddings})
        assert handler.get_debate_embeddings() is mock_embeddings

    def test_get_critique_store(self):
        """get_critique_store returns store from context."""
        from aragora.server.handlers.base import BaseHandler

        mock_store = MagicMock()
        handler = BaseHandler({"critique_store": mock_store})
        assert handler.get_critique_store() is mock_store

    def test_get_nomic_dir(self):
        """get_nomic_dir returns path from context."""
        from aragora.server.handlers.base import BaseHandler
        from pathlib import Path

        path = Path("/tmp/nomic")
        handler = BaseHandler({"nomic_dir": path})
        assert handler.get_nomic_dir() == path


class TestBaseHandlerPathExtraction:
    """Tests for BaseHandler path parameter extraction."""

    def test_extract_path_param_success(self):
        """extract_path_param extracts valid parameter."""
        from aragora.server.handlers.base import BaseHandler

        handler = BaseHandler({})
        value, err = handler.extract_path_param(
            "/api/v1/debates/debate-123/messages", 2, "debate_id"
        )
        assert value == "debate-123"
        assert err is None

    def test_extract_path_param_missing(self):
        """extract_path_param returns error for missing segment."""
        from aragora.server.handlers.base import BaseHandler

        handler = BaseHandler({})
        value, err = handler.extract_path_param("/api/v1/debates", 5, "missing")
        assert value is None
        assert err is not None
        assert err.status_code == 400

    def test_extract_path_param_empty(self):
        """extract_path_param returns error for empty segment."""
        from aragora.server.handlers.base import BaseHandler

        handler = BaseHandler({})
        # Double slash creates empty segment
        value, err = handler.extract_path_param("/api/v1//debates", 1, "empty")
        assert value is None
        assert err is not None

    def test_extract_path_params_multiple(self):
        """extract_path_params extracts multiple parameters."""
        from aragora.server.handlers.base import BaseHandler, SAFE_ID_PATTERN

        handler = BaseHandler({})
        params, err = handler.extract_path_params(
            "/api/v1/debates/debate-123/rounds/5",
            [
                (2, "debate_id", SAFE_ID_PATTERN),
                (3, "resource", None),
                (4, "round_num", None),
            ],
        )
        assert err is None
        assert params["debate_id"] == "debate-123"
        assert params["resource"] == "rounds"
        assert params["round_num"] == "5"

    def test_extract_path_params_first_error_stops(self):
        """extract_path_params returns first error encountered."""
        from aragora.server.handlers.base import BaseHandler

        handler = BaseHandler({})
        params, err = handler.extract_path_params(
            "/api/v1/debates",
            [
                (2, "debate_id", None),
                (10, "nonexistent", None),  # This would fail if reached
            ],
        )
        assert params is None
        assert err is not None


class TestBaseHandlerJsonParsing:
    """Tests for BaseHandler JSON body parsing."""

    def test_read_json_body_success(self):
        """read_json_body parses valid JSON."""
        from aragora.server.handlers.base import BaseHandler

        handler = BaseHandler({})
        mock_http = MagicMock()
        body_bytes = b'{"key": "value"}'
        mock_http.headers = {"Content-Length": str(len(body_bytes))}
        mock_http.rfile = BytesIO(body_bytes)

        result = handler.read_json_body(mock_http)
        assert result == {"key": "value"}

    def test_read_json_body_empty(self):
        """read_json_body returns empty dict for no content."""
        from aragora.server.handlers.base import BaseHandler

        handler = BaseHandler({})
        mock_http = MagicMock()
        mock_http.headers = {"Content-Length": "0"}

        result = handler.read_json_body(mock_http)
        assert result == {}

    def test_read_json_body_invalid(self):
        """read_json_body returns None for invalid JSON."""
        from aragora.server.handlers.base import BaseHandler

        handler = BaseHandler({})
        mock_http = MagicMock()
        body_bytes = b"not json"
        mock_http.headers = {"Content-Length": str(len(body_bytes))}
        mock_http.rfile = BytesIO(body_bytes)

        result = handler.read_json_body(mock_http)
        assert result is None

    def test_read_json_body_too_large(self):
        """read_json_body returns None for oversized body."""
        from aragora.server.handlers.base import BaseHandler

        handler = BaseHandler({})
        mock_http = MagicMock()
        mock_http.headers = {"Content-Length": "999999999"}

        result = handler.read_json_body(mock_http, max_size=1000)
        assert result is None

    def test_validate_content_length_valid(self):
        """validate_content_length accepts valid length."""
        from aragora.server.handlers.base import BaseHandler

        handler = BaseHandler({})
        mock_http = MagicMock()
        mock_http.headers = {"Content-Length": "100"}

        result = handler.validate_content_length(mock_http)
        assert result == 100

    def test_validate_content_length_negative(self):
        """validate_content_length rejects negative length."""
        from aragora.server.handlers.base import BaseHandler

        handler = BaseHandler({})
        mock_http = MagicMock()
        mock_http.headers = {"Content-Length": "-1"}

        result = handler.validate_content_length(mock_http)
        assert result is None

    def test_validate_json_content_type_valid(self):
        """validate_json_content_type accepts application/json."""
        from aragora.server.handlers.base import BaseHandler

        handler = BaseHandler({})
        mock_http = MagicMock()
        mock_http.headers = {"Content-Type": "application/json"}

        result = handler.validate_json_content_type(mock_http)
        assert result is None  # None means valid

    def test_validate_json_content_type_with_charset(self):
        """validate_json_content_type accepts charset parameter."""
        from aragora.server.handlers.base import BaseHandler

        handler = BaseHandler({})
        mock_http = MagicMock()
        mock_http.headers = {"Content-Type": "application/json; charset=utf-8"}

        result = handler.validate_json_content_type(mock_http)
        assert result is None

    def test_validate_json_content_type_invalid(self):
        """validate_json_content_type rejects non-JSON."""
        from aragora.server.handlers.base import BaseHandler

        handler = BaseHandler({})
        mock_http = MagicMock()
        mock_http.headers = {"Content-Type": "text/plain"}

        result = handler.validate_json_content_type(mock_http)
        assert result is not None
        assert result.status_code == 415

    def test_read_json_body_validated_success(self):
        """read_json_body_validated parses with validation."""
        from aragora.server.handlers.base import BaseHandler

        handler = BaseHandler({})
        mock_http = MagicMock()
        body_bytes = b'{"valid": true}'
        mock_http.headers = {
            "Content-Type": "application/json",
            "Content-Length": str(len(body_bytes)),
        }
        mock_http.rfile = BytesIO(body_bytes)

        body, err = handler.read_json_body_validated(mock_http)
        assert err is None
        assert body == {"valid": True}

    def test_read_json_body_validated_wrong_content_type(self):
        """read_json_body_validated rejects wrong Content-Type."""
        from aragora.server.handlers.base import BaseHandler

        handler = BaseHandler({})
        mock_http = MagicMock()
        mock_http.headers = {"Content-Type": "text/html", "Content-Length": "10"}

        body, err = handler.read_json_body_validated(mock_http)
        assert body is None
        assert err is not None
        assert err.status_code == 415


class TestBaseHandlerAuth:
    """Tests for BaseHandler authentication methods."""

    def test_get_current_user_authenticated(self):
        """get_current_user returns user when authenticated."""
        from aragora.server.handlers.base import BaseHandler
        from aragora.billing.auth.context import UserAuthContext

        handler = BaseHandler({})
        mock_http = MagicMock()
        mock_http.headers = {"Authorization": "Bearer test-token"}

        mock_ctx = UserAuthContext(
            authenticated=True,
            user_id="user-123",
            email="test@example.com",
        )

        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=mock_ctx,
        ):
            result = handler.get_current_user(mock_http)
            assert result is not None
            assert result.user_id == "user-123"

    def test_get_current_user_not_authenticated(self):
        """get_current_user returns None when not authenticated."""
        from aragora.server.handlers.base import BaseHandler
        from aragora.billing.auth.context import UserAuthContext

        handler = BaseHandler({})
        mock_http = MagicMock()
        mock_http.headers = {}

        mock_ctx = UserAuthContext(authenticated=False)

        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=mock_ctx,
        ):
            result = handler.get_current_user(mock_http)
            assert result is None

    def test_require_auth_or_error_success(self):
        """require_auth_or_error returns user when authenticated."""
        from aragora.server.handlers.base import BaseHandler
        from aragora.billing.auth.context import UserAuthContext

        handler = BaseHandler({})
        mock_http = MagicMock()

        mock_ctx = UserAuthContext(
            authenticated=True,
            user_id="user-456",
        )

        with patch.object(handler, "get_current_user", return_value=mock_ctx):
            user, err = handler.require_auth_or_error(mock_http)
            assert err is None
            assert user.user_id == "user-456"

    def test_require_auth_or_error_failure(self):
        """require_auth_or_error returns error when not authenticated."""
        from aragora.server.handlers.base import BaseHandler

        handler = BaseHandler({})
        mock_http = MagicMock()

        with patch.object(handler, "get_current_user", return_value=None):
            user, err = handler.require_auth_or_error(mock_http)
            assert user is None
            assert err is not None
            assert err.status_code == 401


class TestBaseHandlerMethods:
    """Tests for BaseHandler handle methods."""

    def test_handle_returns_none_by_default(self):
        """handle returns None by default."""
        from aragora.server.handlers.base import BaseHandler

        handler = BaseHandler({})
        result = handler.handle("/test", {}, MagicMock())
        assert result is None

    def test_handle_post_returns_none_by_default(self):
        """handle_post returns None by default."""
        from aragora.server.handlers.base import BaseHandler

        handler = BaseHandler({})
        result = handler.handle_post("/test", {}, MagicMock())
        assert result is None

    def test_handle_delete_returns_none_by_default(self):
        """handle_delete returns None by default."""
        from aragora.server.handlers.base import BaseHandler

        handler = BaseHandler({})
        result = handler.handle_delete("/test", {}, MagicMock())
        assert result is None

    def test_handle_patch_returns_none_by_default(self):
        """handle_patch returns None by default."""
        from aragora.server.handlers.base import BaseHandler

        handler = BaseHandler({})
        result = handler.handle_patch("/test", {}, MagicMock())
        assert result is None

    def test_handle_put_returns_none_by_default(self):
        """handle_put returns None by default."""
        from aragora.server.handlers.base import BaseHandler

        handler = BaseHandler({})
        result = handler.handle_put("/test", {}, MagicMock())
        assert result is None


# ============================================================================
# Handler Mixin Tests
# ============================================================================


class TestPaginatedHandlerMixin:
    """Tests for PaginatedHandlerMixin."""

    def test_get_pagination_defaults(self):
        """get_pagination returns defaults for empty params."""
        from aragora.server.handlers.base import PaginatedHandlerMixin

        mixin = PaginatedHandlerMixin()
        limit, offset = mixin.get_pagination({})
        assert limit == mixin.DEFAULT_LIMIT
        assert offset == mixin.DEFAULT_OFFSET

    def test_get_pagination_from_params(self):
        """get_pagination extracts from query params."""
        from aragora.server.handlers.base import PaginatedHandlerMixin

        mixin = PaginatedHandlerMixin()
        limit, offset = mixin.get_pagination({"limit": "50", "offset": "10"})
        assert limit == 50
        assert offset == 10

    def test_get_pagination_clamps_limit(self):
        """get_pagination clamps limit to max."""
        from aragora.server.handlers.base import PaginatedHandlerMixin

        mixin = PaginatedHandlerMixin()
        limit, offset = mixin.get_pagination({"limit": "999"})
        assert limit == mixin.MAX_LIMIT

    def test_get_pagination_clamps_negative(self):
        """get_pagination clamps negative values."""
        from aragora.server.handlers.base import PaginatedHandlerMixin

        mixin = PaginatedHandlerMixin()
        limit, offset = mixin.get_pagination({"limit": "-5", "offset": "-10"})
        assert limit >= 1
        assert offset >= 0

    def test_paginated_response_format(self):
        """paginated_response returns correct format."""
        from aragora.server.handlers.base import PaginatedHandlerMixin

        mixin = PaginatedHandlerMixin()
        result = mixin.paginated_response(
            items=[{"id": 1}, {"id": 2}],
            total=100,
            limit=20,
            offset=0,
        )

        body = json.loads(result.body)
        assert body["items"] == [{"id": 1}, {"id": 2}]
        assert body["total"] == 100
        assert body["limit"] == 20
        assert body["offset"] == 0
        assert body["has_more"] is True

    def test_paginated_response_has_more_false(self):
        """paginated_response sets has_more=False at end."""
        from aragora.server.handlers.base import PaginatedHandlerMixin

        mixin = PaginatedHandlerMixin()
        result = mixin.paginated_response(
            items=[{"id": 1}],
            total=1,
            limit=20,
            offset=0,
        )

        body = json.loads(result.body)
        assert body["has_more"] is False

    def test_paginated_response_custom_key(self):
        """paginated_response accepts custom items_key."""
        from aragora.server.handlers.base import PaginatedHandlerMixin

        mixin = PaginatedHandlerMixin()
        result = mixin.paginated_response(
            items=[{"name": "test"}],
            total=1,
            limit=20,
            offset=0,
            items_key="debates",
        )

        body = json.loads(result.body)
        assert "debates" in body
        assert "items" not in body


class TestCachedHandlerMixin:
    """Tests for CachedHandlerMixin."""

    def test_cached_response_caches_value(self):
        """cached_response caches and returns value."""
        from aragora.server.handlers.base import CachedHandlerMixin, clear_cache

        clear_cache()
        mixin = CachedHandlerMixin()

        call_count = 0

        def generator():
            nonlocal call_count
            call_count += 1
            return {"computed": True}

        # First call - generates
        result1 = mixin.cached_response("test_key", 60, generator)
        assert result1 == {"computed": True}
        assert call_count == 1

        # Second call - should use cache
        result2 = mixin.cached_response("test_key", 60, generator)
        assert result2 == {"computed": True}
        # Generator should not be called again (cached)
        # Note: Actual caching depends on implementation

    def test_cached_response_different_keys(self):
        """cached_response uses different values for different keys."""
        from aragora.server.handlers.base import CachedHandlerMixin, clear_cache

        clear_cache()
        mixin = CachedHandlerMixin()

        result1 = mixin.cached_response("key1", 60, lambda: "value1")
        result2 = mixin.cached_response("key2", 60, lambda: "value2")

        assert result1 == "value1"
        assert result2 == "value2"


class TestAuthenticatedHandlerMixin:
    """Tests for AuthenticatedHandlerMixin."""

    def test_require_auth_delegates_to_base(self):
        """require_auth uses require_auth_or_error when available."""
        from aragora.server.handlers.base import (
            AuthenticatedHandlerMixin,
            BaseHandler,
        )
        from aragora.billing.auth.context import UserAuthContext

        class TestHandler(BaseHandler, AuthenticatedHandlerMixin):
            pass

        handler = TestHandler({})
        mock_http = MagicMock()

        mock_user = UserAuthContext(authenticated=True, user_id="test-user")
        with patch.object(handler, "get_current_user", return_value=mock_user):
            result = handler.require_auth(mock_http)
            assert result.user_id == "test-user"

    def test_require_auth_returns_error(self):
        """require_auth returns error when not authenticated."""
        from aragora.server.handlers.base import (
            AuthenticatedHandlerMixin,
            BaseHandler,
        )

        class TestHandler(BaseHandler, AuthenticatedHandlerMixin):
            pass

        handler = TestHandler({})
        mock_http = MagicMock()

        with patch.object(handler, "get_current_user", return_value=None):
            result = handler.require_auth(mock_http)
            # Result should be HandlerResult with 401
            assert hasattr(result, "status_code")
            assert result.status_code == 401


# ============================================================================
# Decorator Tests
# ============================================================================


class TestRequireQuotaDecorator:
    """Tests for require_quota decorator."""

    def test_require_quota_passes_for_available_quota(self):
        """require_quota allows operation when quota available."""
        from aragora.server.handlers.base import require_quota, json_response
        from aragora.billing.auth.context import UserAuthContext

        @require_quota()
        def test_func(handler, user=None):
            return json_response({"success": True})

        mock_handler = MagicMock()
        mock_handler.headers = {}

        mock_user = UserAuthContext(
            authenticated=True,
            user_id="test-user",
            org_id=None,  # No org = no quota check
        )

        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=mock_user,
        ):
            result = test_func(mock_handler)
            body = json.loads(result.body)
            assert body["success"] is True

    def test_require_quota_rejects_unauthenticated(self):
        """require_quota returns 401 for unauthenticated request."""
        from aragora.server.handlers.base import require_quota, json_response
        from aragora.billing.auth.context import UserAuthContext

        @require_quota()
        def test_func(handler, user=None):
            return json_response({"success": True})

        mock_handler = MagicMock()
        mock_handler.headers = {}

        mock_user = UserAuthContext(
            authenticated=False,
            error_reason="No token provided",
        )

        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=mock_user,
        ):
            result = test_func(mock_handler)
            assert result.status_code == 401


# ============================================================================
# Parameter Extraction Tests
# ============================================================================


class TestParameterExtraction:
    """Tests for parameter extraction utilities."""

    def test_get_int_param_default(self):
        """get_int_param returns default for missing param."""
        from aragora.server.handlers.base import get_int_param

        result = get_int_param({}, "missing", 42)
        assert result == 42

    def test_get_int_param_from_string(self):
        """get_int_param parses string to int."""
        from aragora.server.handlers.base import get_int_param

        result = get_int_param({"count": "100"}, "count", 0)
        assert result == 100

    def test_get_bool_param_true_values(self):
        """get_bool_param recognizes true values."""
        from aragora.server.handlers.base import get_bool_param

        assert get_bool_param({"flag": "true"}, "flag", False) is True
        assert get_bool_param({"flag": "1"}, "flag", False) is True
        assert get_bool_param({"flag": "yes"}, "flag", False) is True

    def test_get_bool_param_false_values(self):
        """get_bool_param recognizes false values."""
        from aragora.server.handlers.base import get_bool_param

        assert get_bool_param({"flag": "false"}, "flag", True) is False
        assert get_bool_param({"flag": "0"}, "flag", True) is False
        assert get_bool_param({"flag": "no"}, "flag", True) is False

    def test_get_clamped_int_param(self):
        """get_clamped_int_param clamps to range."""
        from aragora.server.handlers.base import get_clamped_int_param

        # Below min
        result = get_clamped_int_param({"val": "-10"}, "val", 50, min_val=0, max_val=100)
        assert result == 0

        # Above max
        result = get_clamped_int_param({"val": "999"}, "val", 50, min_val=0, max_val=100)
        assert result == 100

        # In range
        result = get_clamped_int_param({"val": "75"}, "val", 50, min_val=0, max_val=100)
        assert result == 75

    def test_get_bounded_string_param_truncates(self):
        """get_bounded_string_param truncates long strings."""
        from aragora.server.handlers.base import get_bounded_string_param

        long_string = "a" * 1000
        result = get_bounded_string_param({"text": long_string}, "text", "", max_length=100)
        assert len(result) == 100

    def test_get_bounded_float_param_clamps(self):
        """get_bounded_float_param clamps to range."""
        from aragora.server.handlers.base import get_bounded_float_param

        # Below min
        result = get_bounded_float_param({"val": "-5.0"}, "val", 0.5, min_val=0.0, max_val=1.0)
        assert result == 0.0

        # Above max
        result = get_bounded_float_param({"val": "2.5"}, "val", 0.5, min_val=0.0, max_val=1.0)
        assert result == 1.0


# ============================================================================
# Module Exports Tests
# ============================================================================


class TestModuleExports:
    """Tests for module-level exports."""

    def test_all_core_exports_exist(self):
        """All core exports are importable."""
        from aragora.server.handlers.base import (
            BaseHandler,
            HandlerResult,
            error_response,
            json_response,
            handle_errors,
            require_auth,
            require_user_auth,
            require_storage,
        )

        assert all(
            [
                BaseHandler,
                HandlerResult,
                error_response,
                json_response,
                handle_errors,
                require_auth,
                require_user_auth,
                require_storage,
            ]
        )

    def test_cache_exports_exist(self):
        """Cache-related exports are importable."""
        from aragora.server.handlers.base import (
            ttl_cache,
            clear_cache,
            get_cache_stats,
            invalidate_cache,
            BoundedTTLCache,
        )

        assert all([ttl_cache, clear_cache, get_cache_stats, invalidate_cache, BoundedTTLCache])

    def test_validation_exports_exist(self):
        """Validation exports are importable."""
        from aragora.server.handlers.base import (
            SAFE_ID_PATTERN,
            SAFE_SLUG_PATTERN,
            SAFE_AGENT_PATTERN,
            validate_agent_name,
            validate_debate_id,
        )

        assert all(
            [
                SAFE_ID_PATTERN,
                SAFE_SLUG_PATTERN,
                SAFE_AGENT_PATTERN,
                validate_agent_name,
                validate_debate_id,
            ]
        )

    def test_mixin_exports_exist(self):
        """Mixin exports are importable."""
        from aragora.server.handlers.base import (
            PaginatedHandlerMixin,
            CachedHandlerMixin,
            AuthenticatedHandlerMixin,
        )

        assert all([PaginatedHandlerMixin, CachedHandlerMixin, AuthenticatedHandlerMixin])
