"""Tests for RLMContextHandler endpoints.

Validates the REST API endpoints for RLM context operations including:
- GET /api/rlm/stats - Compression statistics and cache info
- GET /api/rlm/strategies - List decomposition strategies
- POST /api/rlm/compress - Compress content
- POST /api/rlm/query - Query compressed context
- GET /api/rlm/contexts - List stored contexts
- GET /api/rlm/context/{id} - Get context details
- DELETE /api/rlm/context/{id} - Delete context
"""

import json
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.rlm import RLMContextHandler
from aragora.billing.auth.context import UserAuthContext


def mock_authenticated_user():
    """Create a mock authenticated user context."""
    user_ctx = MagicMock(spec=UserAuthContext)
    user_ctx.is_authenticated = True
    user_ctx.user_id = "test-user-123"
    user_ctx.email = "test@example.com"
    return user_ctx


@pytest.fixture
def rlm_context_handler():
    """Create an RLMContextHandler with mocked dependencies."""
    ctx = {"storage": None}
    handler = RLMContextHandler(ctx)
    return handler


@pytest.fixture
def mock_http_handler():
    """Create a mock HTTP handler with client address."""
    handler = MagicMock()
    handler.client_address = ("127.0.0.1", 12345)
    handler.headers = {}
    handler.command = "GET"
    return handler


def create_request_body(data: dict, method: str = "POST") -> MagicMock:
    """Create a mock HTTP handler with a JSON body."""
    handler = MagicMock()
    handler.client_address = ("127.0.0.1", 12345)
    body = json.dumps(data).encode("utf-8")
    handler.headers = {
        "Content-Length": str(len(body)),
        "Content-Type": "application/json",
    }
    handler.rfile = BytesIO(body)
    handler.command = method
    return handler


def create_auth_request_body(data: dict, method: str = "POST") -> MagicMock:
    """Create a mock HTTP handler with a JSON body and auth token."""
    handler = create_request_body(data, method)
    handler.headers["Authorization"] = "Bearer test-token"
    return handler


def mock_auth_config():
    """Create a mock auth config with a test token."""
    config = MagicMock()
    config.api_token = "test-token"
    config.validate_token = lambda t: t == "test-token"
    return config


class TestRLMContextHandlerCanHandle:
    """Test RLMContextHandler.can_handle method."""

    def test_can_handle_stats(self, rlm_context_handler):
        """Test can_handle returns True for stats endpoint."""
        assert rlm_context_handler.can_handle("/api/v1/rlm/stats")

    def test_can_handle_strategies(self, rlm_context_handler):
        """Test can_handle returns True for strategies endpoint."""
        assert rlm_context_handler.can_handle("/api/v1/rlm/strategies")

    def test_can_handle_compress(self, rlm_context_handler):
        """Test can_handle returns True for compress endpoint."""
        assert rlm_context_handler.can_handle("/api/v1/rlm/compress")

    def test_can_handle_query(self, rlm_context_handler):
        """Test can_handle returns True for query endpoint."""
        assert rlm_context_handler.can_handle("/api/v1/rlm/query")

    def test_can_handle_contexts(self, rlm_context_handler):
        """Test can_handle returns True for contexts endpoint."""
        assert rlm_context_handler.can_handle("/api/v1/rlm/contexts")

    def test_can_handle_context_id(self, rlm_context_handler):
        """Test can_handle returns True for context/{id} endpoint."""
        assert rlm_context_handler.can_handle("/api/v1/rlm/context/ctx_abc123")

    def test_cannot_handle_unknown(self, rlm_context_handler):
        """Test can_handle returns False for unknown endpoint."""
        assert not rlm_context_handler.can_handle("/api/v1/unknown")
        assert not rlm_context_handler.can_handle("/api/v1/rlm")
        assert not rlm_context_handler.can_handle("/api/v1/debates/123/query-rlm")


class TestRLMContextHandlerStats:
    """Test GET /api/rlm/stats endpoint."""

    def test_stats_returns_cache_info(self, rlm_context_handler, mock_http_handler):
        """Test stats endpoint returns cache statistics."""
        result = rlm_context_handler.handle("/api/v1/rlm/stats", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "cache" in body
        assert "contexts" in body
        assert "system" in body
        assert "timestamp" in body

    def test_stats_includes_system_info(self, rlm_context_handler, mock_http_handler):
        """Test stats includes system availability info."""
        result = rlm_context_handler.handle("/api/v1/rlm/stats", {}, mock_http_handler)

        body = json.loads(result.body)
        assert "has_official_rlm" in body["system"]
        assert "compressor_available" in body["system"]
        assert "rlm_available" in body["system"]


class TestRLMContextHandlerStrategies:
    """Test GET /api/rlm/strategies endpoint."""

    def test_strategies_returns_list(self, rlm_context_handler, mock_http_handler):
        """Test strategies endpoint returns strategy list."""
        result = rlm_context_handler.handle("/api/v1/rlm/strategies", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "strategies" in body
        assert "default" in body
        assert body["default"] == "auto"

    def test_strategies_includes_all_types(self, rlm_context_handler, mock_http_handler):
        """Test strategies includes all expected strategy types."""
        result = rlm_context_handler.handle("/api/v1/rlm/strategies", {}, mock_http_handler)

        body = json.loads(result.body)
        strategies = body["strategies"]
        expected_strategies = ["peek", "grep", "partition_map", "summarize", "hierarchical", "auto"]
        for strategy in expected_strategies:
            assert strategy in strategies, f"Missing strategy: {strategy}"
            assert "description" in strategies[strategy]
            assert "use_case" in strategies[strategy]


class TestRLMContextHandlerCompress:
    """Test POST /api/rlm/compress endpoint."""

    def test_compress_requires_auth(self, rlm_context_handler):
        """Test compress endpoint requires authentication."""
        handler = create_request_body({"content": "test content"})

        result = rlm_context_handler.handle_post("/api/v1/rlm/compress", {}, handler)

        # Should require auth
        assert result is not None
        assert result.status_code == 401

    def test_compress_requires_content(self, rlm_context_handler):
        """Test compress requires content field."""
        handler = create_auth_request_body({})

        with patch("aragora.server.auth.auth_config", mock_auth_config()):
            result = rlm_context_handler.handle_post("/api/v1/rlm/compress", {}, handler)

        assert result is not None
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "content" in body["error"].lower()

    def test_compress_validates_source_type(self, rlm_context_handler):
        """Test compress validates source_type parameter."""
        handler = create_auth_request_body(
            {
                "content": "test content",
                "source_type": "invalid",
            }
        )

        with patch("aragora.server.auth.auth_config", mock_auth_config()):
            result = rlm_context_handler.handle_post("/api/v1/rlm/compress", {}, handler)

        assert result is not None
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "source_type" in body["error"].lower()

    def test_compress_validates_levels(self, rlm_context_handler):
        """Test compress validates levels parameter."""
        handler = create_auth_request_body(
            {
                "content": "test content",
                "levels": 10,  # Invalid - max is 5
            }
        )

        with patch("aragora.server.auth.auth_config", mock_auth_config()):
            result = rlm_context_handler.handle_post("/api/v1/rlm/compress", {}, handler)

        assert result is not None
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "levels" in body["error"].lower()

    def test_compress_success(self, rlm_context_handler):
        """Test successful content compression."""
        handler = create_auth_request_body(
            {
                "content": "This is test content for compression.",
                "source_type": "text",
                "levels": 4,
            }
        )

        # Mock the compressor
        mock_context = MagicMock()
        mock_context.original_tokens = 100
        mock_context.total_tokens.return_value = 50
        mock_context.levels = {}

        with patch("aragora.server.auth.auth_config", mock_auth_config()):
            with patch.object(rlm_context_handler, "_get_compressor") as mock_get:
                mock_compressor = MagicMock()
                mock_compressor.compress = AsyncMock(return_value=mock_context)
                mock_get.return_value = mock_compressor

                result = rlm_context_handler.handle_post("/api/v1/rlm/compress", {}, handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "context_id" in body
        assert body["context_id"].startswith("ctx_")
        assert "compression_result" in body
        assert body["compression_result"]["original_tokens"] == 100


class TestRLMContextHandlerQuery:
    """Test POST /api/rlm/query endpoint."""

    def test_query_requires_context_id(self, rlm_context_handler):
        """Test query requires context_id field."""
        handler = create_auth_request_body(
            {
                "query": "What is this about?",
            }
        )

        with patch("aragora.server.auth.auth_config", mock_auth_config()):
            result = rlm_context_handler.handle_post("/api/v1/rlm/query", {}, handler)

        assert result is not None
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "context_id" in body["error"].lower()

    def test_query_requires_query(self, rlm_context_handler):
        """Test query requires query field."""
        handler = create_auth_request_body(
            {
                "context_id": "ctx_abc123",
            }
        )

        with patch("aragora.server.auth.auth_config", mock_auth_config()):
            result = rlm_context_handler.handle_post("/api/v1/rlm/query", {}, handler)

        assert result is not None
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "query" in body["error"].lower()

    def test_query_context_not_found(self, rlm_context_handler):
        """Test query returns 404 for unknown context."""
        handler = create_auth_request_body(
            {
                "context_id": "ctx_nonexistent",
                "query": "What is this about?",
            }
        )

        with patch("aragora.server.auth.auth_config", mock_auth_config()):
            result = rlm_context_handler.handle_post("/api/v1/rlm/query", {}, handler)

        assert result is not None
        assert result.status_code == 404

    def test_query_validates_strategy(self, rlm_context_handler):
        """Test query validates strategy parameter."""
        # First store a context
        rlm_context_handler._contexts["ctx_test"] = {
            "context": MagicMock(),
            "created_at": "2024-01-01T00:00:00",
            "source_type": "text",
            "original_tokens": 100,
        }

        handler = create_auth_request_body(
            {
                "context_id": "ctx_test",
                "query": "What is this about?",
                "strategy": "invalid_strategy",
            }
        )

        with patch("aragora.server.auth.auth_config", mock_auth_config()):
            result = rlm_context_handler.handle_post("/api/v1/rlm/query", {}, handler)

        assert result is not None
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "strategy" in body["error"].lower()


class TestRLMContextHandlerContexts:
    """Test GET /api/rlm/contexts endpoint."""

    def test_list_contexts_empty(self, rlm_context_handler, mock_http_handler):
        """Test list contexts returns empty list when no contexts."""
        result = rlm_context_handler.handle("/api/v1/rlm/contexts", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "contexts" in body
        assert body["contexts"] == []
        assert body["total"] == 0

    def test_list_contexts_with_data(self, rlm_context_handler, mock_http_handler):
        """Test list contexts returns stored contexts."""
        # Add some test contexts
        rlm_context_handler._contexts["ctx_1"] = {
            "context": MagicMock(),
            "created_at": "2024-01-01T00:00:00",
            "source_type": "text",
            "original_tokens": 100,
        }
        rlm_context_handler._contexts["ctx_2"] = {
            "context": MagicMock(),
            "created_at": "2024-01-02T00:00:00",
            "source_type": "code",
            "original_tokens": 200,
        }

        result = rlm_context_handler.handle("/api/v1/rlm/contexts", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["total"] == 2
        assert len(body["contexts"]) == 2

    def test_list_contexts_pagination(self, rlm_context_handler, mock_http_handler):
        """Test list contexts supports pagination."""
        # Add multiple contexts
        for i in range(10):
            rlm_context_handler._contexts[f"ctx_{i}"] = {
                "context": MagicMock(),
                "created_at": f"2024-01-{i + 1:02d}T00:00:00",
                "source_type": "text",
                "original_tokens": 100 * i,
            }

        result = rlm_context_handler.handle(
            "/api/v1/rlm/contexts", {"limit": "3", "offset": "2"}, mock_http_handler
        )

        assert result is not None
        body = json.loads(result.body)
        assert body["total"] == 10
        assert body["limit"] == 3
        assert body["offset"] == 2
        assert len(body["contexts"]) == 3


class TestRLMContextHandlerGetContext:
    """Test GET /api/rlm/context/{id} endpoint."""

    def test_get_context_not_found(self, rlm_context_handler, mock_http_handler):
        """Test get context returns 404 for unknown context."""
        result = rlm_context_handler.handle(
            "/api/v1/rlm/context/ctx_nonexistent", {}, mock_http_handler
        )

        assert result is not None
        assert result.status_code == 404

    def test_get_context_success(self, rlm_context_handler, mock_http_handler):
        """Test get context returns context details."""
        mock_ctx = MagicMock()
        mock_ctx.original_tokens = 1000
        mock_ctx.total_tokens.return_value = 300
        mock_ctx.levels = {}

        rlm_context_handler._contexts["ctx_test123"] = {
            "context": mock_ctx,
            "created_at": "2024-01-15T12:00:00",
            "source_type": "code",
            "original_tokens": 1000,
        }

        result = rlm_context_handler.handle(
            "/api/v1/rlm/context/ctx_test123", {}, mock_http_handler
        )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["id"] == "ctx_test123"
        assert body["source_type"] == "code"
        assert body["original_tokens"] == 1000


class TestRLMContextHandlerDeleteContext:
    """Test DELETE /api/rlm/context/{id} endpoint."""

    def test_delete_context_requires_auth(self, rlm_context_handler):
        """Test delete context requires authentication."""
        rlm_context_handler._contexts["ctx_test"] = {
            "context": MagicMock(),
            "created_at": "2024-01-01T00:00:00",
            "source_type": "text",
            "original_tokens": 100,
        }

        handler = MagicMock()
        handler.client_address = ("127.0.0.1", 12345)
        handler.headers = {}
        handler.command = "DELETE"

        result = rlm_context_handler.handle_delete("/api/v1/rlm/context/ctx_test", {}, handler)

        # Should require auth
        assert result is not None
        assert result.status_code == 401

    def test_delete_context_not_found(self, rlm_context_handler):
        """Test delete context returns 404 for unknown context."""
        handler = MagicMock()
        handler.client_address = ("127.0.0.1", 12345)
        handler.headers = {"Authorization": "Bearer test-token"}
        handler.command = "DELETE"

        with patch("aragora.server.auth.auth_config", mock_auth_config()):
            result = rlm_context_handler.handle_delete(
                "/api/v1/rlm/context/ctx_nonexistent", {}, handler
            )

        assert result is not None
        assert result.status_code == 404

    def test_delete_context_success(self, rlm_context_handler):
        """Test successful context deletion."""
        rlm_context_handler._contexts["ctx_to_delete"] = {
            "context": MagicMock(),
            "created_at": "2024-01-01T00:00:00",
            "source_type": "text",
            "original_tokens": 100,
        }

        handler = MagicMock()
        handler.client_address = ("127.0.0.1", 12345)
        handler.headers = {"Authorization": "Bearer test-token"}
        handler.command = "DELETE"

        with patch("aragora.server.auth.auth_config", mock_auth_config()):
            result = rlm_context_handler.handle_delete(
                "/api/v1/rlm/context/ctx_to_delete", {}, handler
            )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["success"] is True
        assert "ctx_to_delete" not in rlm_context_handler._contexts


class TestRLMContextHandlerIntegration:
    """Integration tests for RLMContextHandler."""

    def test_routes_constant_matches_can_handle(self, rlm_context_handler):
        """Test that ROUTES constant matches can_handle logic."""
        for route in RLMContextHandler.ROUTES:
            assert rlm_context_handler.can_handle(route), (
                f"can_handle should return True for {route}"
            )

    def test_handler_inherits_base(self, rlm_context_handler):
        """Test that RLMContextHandler inherits from BaseHandler."""
        from aragora.server.handlers.base import BaseHandler

        assert isinstance(rlm_context_handler, BaseHandler)

    def test_handler_exported_from_package(self):
        """Test that RLMContextHandler is exported from handlers package."""
        from aragora.server.handlers import RLMContextHandler as ExportedHandler

        assert ExportedHandler is RLMContextHandler

    def test_handler_in_all_handlers(self):
        """Test that RLMContextHandler is in ALL_HANDLERS registry."""
        from aragora.server.handlers import ALL_HANDLERS

        assert RLMContextHandler in ALL_HANDLERS


class TestRLMContextHandlerErrorHandling:
    """Test error handling in RLMContextHandler."""

    def test_compress_handles_compressor_unavailable(self, rlm_context_handler):
        """Test compress handles unavailable compressor gracefully."""
        handler = create_auth_request_body(
            {
                "content": "test content",
                "source_type": "text",
            }
        )

        with patch("aragora.server.auth.auth_config", mock_auth_config()):
            with patch.object(rlm_context_handler, "_get_compressor", return_value=None):
                result = rlm_context_handler.handle_post("/api/v1/rlm/compress", {}, handler)

        assert result is not None
        assert result.status_code == 503  # Service unavailable
        body = json.loads(result.body)
        # Error can be nested or simple depending on error_response format
        error_msg = body.get("error", "")
        if isinstance(error_msg, dict):
            error_msg = error_msg.get("message", "")
        assert "not available" in error_msg.lower()

    def test_content_size_limit(self, rlm_context_handler):
        """Test compress rejects content over size limit.

        Note: The handler limits body size at the read_json_body stage,
        returning 400 "Request body required" for oversized payloads.
        """
        large_content = "x" * (10_000_001)  # Just over 10MB limit
        handler = create_auth_request_body(
            {
                "content": large_content,
                "source_type": "text",
            }
        )

        with patch("aragora.server.auth.auth_config", mock_auth_config()):
            result = rlm_context_handler.handle_post("/api/v1/rlm/compress", {}, handler)

        assert result is not None
        # Large bodies are rejected at the read_json_body stage
        assert result.status_code in (400, 413)
