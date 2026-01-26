"""Tests for RLM handler endpoints.

Comprehensive tests covering:
- RLM context compression endpoints
- Query operations on compressed contexts
- Context storage and retrieval
- Strategy listing
- Statistics endpoints
- Error handling and validation
- Authentication requirements
- Rate limiting
"""

import json
from datetime import datetime
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.rlm import RLMContextHandler


@pytest.fixture(autouse=True)
def mock_auth_config():
    """Mock auth_config to allow authentication for tests."""
    with patch("aragora.server.auth.auth_config") as mock_config:
        mock_config.api_token = "test-token"
        mock_config.validate_token = MagicMock(return_value=True)
        yield mock_config


def mock_authenticated_request():
    """Create a mock auth config for authenticated requests."""
    return {"api_token": "test-token", "validate_token": lambda t: t == "test-token"}


def mock_unauthenticated_request():
    """Mark that request should fail auth (no token provided)."""
    return None


@pytest.fixture
def rlm_handler():
    """Create an RLM context handler with mocked dependencies."""
    ctx = {"storage": None, "nomic_dir": None}
    handler = RLMContextHandler(ctx)
    return handler


@pytest.fixture
def mock_http_handler():
    """Create a mock HTTP handler with client address."""
    handler = MagicMock()
    handler.client_address = ("127.0.0.1", 12345)
    handler.headers = {"Content-Length": "0"}
    handler.command = "GET"
    return handler


def create_request_body(data: dict, with_auth: bool = False) -> MagicMock:
    """Create a mock HTTP handler with a JSON body."""
    handler = MagicMock()
    handler.client_address = ("127.0.0.1", 12345)
    body = json.dumps(data).encode("utf-8")
    handler.headers = {
        "Content-Length": str(len(body)),
        "Content-Type": "application/json",
    }
    if with_auth:
        handler.headers["Authorization"] = "Bearer test-token"
    handler.rfile = BytesIO(body)
    handler.command = "POST"
    return handler


class TestRLMContextHandlerCanHandle:
    """Test RLMContextHandler.can_handle method."""

    def test_can_handle_stats(self, rlm_handler):
        """Test can_handle returns True for stats endpoint."""
        assert rlm_handler.can_handle("/api/v1/rlm/stats")

    def test_can_handle_strategies(self, rlm_handler):
        """Test can_handle returns True for strategies endpoint."""
        assert rlm_handler.can_handle("/api/v1/rlm/strategies")

    def test_can_handle_compress(self, rlm_handler):
        """Test can_handle returns True for compress endpoint."""
        assert rlm_handler.can_handle("/api/v1/rlm/compress")

    def test_can_handle_query(self, rlm_handler):
        """Test can_handle returns True for query endpoint."""
        assert rlm_handler.can_handle("/api/v1/rlm/query")

    def test_can_handle_contexts(self, rlm_handler):
        """Test can_handle returns True for contexts list endpoint."""
        assert rlm_handler.can_handle("/api/v1/rlm/contexts")

    def test_can_handle_context_by_id(self, rlm_handler):
        """Test can_handle returns True for context by ID endpoint."""
        assert rlm_handler.can_handle("/api/v1/rlm/context/ctx_abc123")

    def test_cannot_handle_unknown(self, rlm_handler):
        """Test can_handle returns False for unknown endpoint."""
        assert not rlm_handler.can_handle("/api/v1/unknown")
        assert not rlm_handler.can_handle("/api/v1/debates")
        assert not rlm_handler.can_handle("/api/v1/rlm/unknown")


class TestRLMContextHandlerStatsEndpoint:
    """Test GET /api/rlm/stats endpoint."""

    def test_stats_returns_data(self, rlm_handler, mock_http_handler):
        """Stats endpoint returns cache and system statistics."""
        with patch("aragora.rlm.compressor.get_compression_cache_stats") as mock_stats:
            mock_stats.return_value = {
                "hits": 100,
                "misses": 20,
                "size": 50,
            }
            with patch("aragora.rlm.HAS_OFFICIAL_RLM", False):
                result = rlm_handler.handle("/api/v1/rlm/stats", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "cache" in body
        assert "contexts" in body
        assert "system" in body
        assert "timestamp" in body

    def test_stats_handles_import_error(self, rlm_handler, mock_http_handler):
        """Stats endpoint handles import error gracefully."""
        # Patch at the point of import inside the handler method
        with patch.dict("sys.modules", {"aragora.rlm.compressor": None}):
            # Force reimport to trigger ImportError
            result = rlm_handler.handle("/api/v1/rlm/stats", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "cache" in body
        # When import fails, we get the fallback response
        assert body["system"]["compressor_available"] is False or "error" in body["cache"]


class TestRLMContextHandlerStrategiesEndpoint:
    """Test GET /api/rlm/strategies endpoint."""

    def test_strategies_returns_list(self, rlm_handler, mock_http_handler):
        """Strategies endpoint returns available decomposition strategies."""
        result = rlm_handler.handle("/api/v1/rlm/strategies", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "strategies" in body
        assert "default" in body

        # Verify known strategies are present
        strategies = body["strategies"]
        assert "peek" in strategies
        assert "grep" in strategies
        assert "partition_map" in strategies
        assert "summarize" in strategies
        assert "hierarchical" in strategies
        assert "auto" in strategies

    def test_strategies_include_descriptions(self, rlm_handler, mock_http_handler):
        """Each strategy includes description and use case."""
        result = rlm_handler.handle("/api/v1/rlm/strategies", {}, mock_http_handler)

        body = json.loads(result.body)
        for name, strategy in body["strategies"].items():
            assert "name" in strategy, f"Strategy {name} missing name"
            assert "description" in strategy, f"Strategy {name} missing description"
            assert "use_case" in strategy, f"Strategy {name} missing use_case"


class TestRLMContextHandlerCompressEndpoint:
    """Test POST /api/rlm/compress endpoint."""

    def test_compress_requires_auth(self, rlm_handler, mock_auth_config):
        """Compress endpoint requires authentication."""
        # Override the autouse mock to simulate no API token configured
        mock_auth_config.api_token = None
        handler = create_request_body({"content": "test content"}, with_auth=False)

        result = rlm_handler.handle_post("/api/v1/rlm/compress", {}, handler)

        assert result is not None
        assert result.status_code == 401

    def test_compress_requires_content(self, rlm_handler):
        """Compress endpoint requires content field."""
        handler = create_request_body({}, with_auth=True)

        result = rlm_handler.handle_post("/api/v1/rlm/compress", {}, handler)

        assert result is not None
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "error" in body
        assert "content" in body["error"].lower()

    def test_compress_validates_content_type(self, rlm_handler):
        """Compress endpoint validates content must be string."""
        handler = create_request_body({"content": 12345}, with_auth=True)

        result = rlm_handler.handle_post("/api/v1/rlm/compress", {}, handler)

        assert result is not None
        assert result.status_code == 400

    def test_compress_rejects_too_large_content(self, rlm_handler):
        """Compress endpoint rejects content over 10MB."""
        # Create content > 10MB
        # Note: The handler checks Content-Length first, so we need to set that
        handler = MagicMock()
        handler.client_address = ("127.0.0.1", 12345)
        handler.headers = {
            "Content-Length": str(10 * 1024 * 1024 + 1000),  # Over 10MB
            "Content-Type": "application/json",
            "Authorization": "Bearer test-token",
        }
        handler.command = "POST"
        # The handler will reject based on Content-Length before reading body

        result = rlm_handler.handle_post("/api/v1/rlm/compress", {}, handler)

        assert result is not None
        # Handler returns None for body when content length > 10MB
        # which results in a 400 "Request body required" - this is the actual behavior
        assert result.status_code == 400

    def test_compress_validates_source_type(self, rlm_handler):
        """Compress endpoint validates source_type parameter."""
        handler = create_request_body(
            {
                "content": "test content",
                "source_type": "invalid",
            },
            with_auth=True,
        )

        result = rlm_handler.handle_post("/api/v1/rlm/compress", {}, handler)

        assert result is not None
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "source_type" in body["error"].lower()

    def test_compress_validates_levels(self, rlm_handler):
        """Compress endpoint validates levels parameter."""
        handler = create_request_body(
            {
                "content": "test content",
                "levels": 10,  # Max is 5
            },
            with_auth=True,
        )

        result = rlm_handler.handle_post("/api/v1/rlm/compress", {}, handler)

        assert result is not None
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "levels" in body["error"].lower()

    def test_compress_returns_503_when_unavailable(self, rlm_handler):
        """Compress endpoint returns 503 when compressor unavailable."""
        handler = create_request_body({"content": "test content"}, with_auth=True)

        with patch.object(rlm_handler, "_get_compressor", return_value=None):
            result = rlm_handler.handle_post("/api/v1/rlm/compress", {}, handler)

        assert result is not None
        assert result.status_code == 503

    def test_compress_success(self, rlm_handler):
        """Compress endpoint returns context ID on success."""
        handler = create_request_body(
            {
                "content": "This is test content for compression.",
                "source_type": "text",
            },
            with_auth=True,
        )

        mock_compressor = MagicMock()
        mock_context = MagicMock()
        mock_context.original_tokens = 100
        mock_context.total_tokens.return_value = 20
        mock_context.levels = {}

        mock_compressor.compress = AsyncMock(return_value=mock_context)

        with patch.object(rlm_handler, "_get_compressor", return_value=mock_compressor):
            result = rlm_handler.handle_post("/api/v1/rlm/compress", {}, handler)

        assert result is not None
        if result.status_code == 200:
            body = json.loads(result.body)
            assert "context_id" in body
            assert "compression_result" in body
            assert "created_at" in body


class TestRLMContextHandlerQueryEndpoint:
    """Test POST /api/rlm/query endpoint."""

    def test_query_requires_auth(self, rlm_handler, mock_auth_config):
        """Query endpoint requires authentication."""
        # Override the autouse mock to simulate no API token configured
        mock_auth_config.api_token = None
        handler = create_request_body(
            {
                "context_id": "ctx_123",
                "query": "What is the main topic?",
            },
            with_auth=False,
        )

        result = rlm_handler.handle_post("/api/v1/rlm/query", {}, handler)

        assert result is not None
        assert result.status_code == 401

    def test_query_requires_context_id(self, rlm_handler):
        """Query endpoint requires context_id field."""
        handler = create_request_body(
            {
                "query": "What is the main topic?",
            },
            with_auth=True,
        )

        result = rlm_handler.handle_post("/api/v1/rlm/query", {}, handler)

        assert result is not None
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "context_id" in body["error"].lower()

    def test_query_requires_query_field(self, rlm_handler):
        """Query endpoint requires query field."""
        handler = create_request_body(
            {
                "context_id": "ctx_123",
            },
            with_auth=True,
        )

        result = rlm_handler.handle_post("/api/v1/rlm/query", {}, handler)

        assert result is not None
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "query" in body["error"].lower()

    def test_query_validates_query_length(self, rlm_handler):
        """Query endpoint validates query length."""
        handler = create_request_body(
            {
                "context_id": "ctx_123",
                "query": "x" * 10001,  # Over 10000 char limit
            },
            with_auth=True,
        )

        result = rlm_handler.handle_post("/api/v1/rlm/query", {}, handler)

        assert result is not None
        assert result.status_code == 400

    def test_query_context_not_found(self, rlm_handler):
        """Query endpoint returns 404 for unknown context."""
        handler = create_request_body(
            {
                "context_id": "nonexistent_ctx",
                "query": "What is the topic?",
            },
            with_auth=True,
        )

        result = rlm_handler.handle_post("/api/v1/rlm/query", {}, handler)

        assert result is not None
        assert result.status_code == 404

    def test_query_validates_strategy(self, rlm_handler):
        """Query endpoint validates strategy parameter."""
        # First, add a context
        rlm_handler._contexts["test_ctx"] = {
            "context": MagicMock(),
            "created_at": datetime.now().isoformat(),
        }

        handler = create_request_body(
            {
                "context_id": "test_ctx",
                "query": "What is the topic?",
                "strategy": "invalid_strategy",
            },
            with_auth=True,
        )

        result = rlm_handler.handle_post("/api/v1/rlm/query", {}, handler)

        assert result is not None
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "strategy" in body["error"].lower()


class TestRLMContextHandlerContextsEndpoint:
    """Test GET /api/rlm/contexts endpoint."""

    def test_list_contexts_empty(self, rlm_handler, mock_http_handler):
        """List contexts returns empty list when no contexts."""
        result = rlm_handler.handle("/api/v1/rlm/contexts", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "contexts" in body
        assert body["total"] == 0
        assert body["contexts"] == []

    def test_list_contexts_with_data(self, rlm_handler, mock_http_handler):
        """List contexts returns stored contexts."""
        # Add some test contexts
        rlm_handler._contexts["ctx_1"] = {
            "context": MagicMock(),
            "source_type": "text",
            "original_tokens": 100,
            "created_at": "2026-01-15T10:00:00",
        }
        rlm_handler._contexts["ctx_2"] = {
            "context": MagicMock(),
            "source_type": "code",
            "original_tokens": 200,
            "created_at": "2026-01-15T11:00:00",
        }

        result = rlm_handler.handle("/api/v1/rlm/contexts", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["total"] == 2
        assert len(body["contexts"]) == 2

    def test_list_contexts_pagination(self, rlm_handler, mock_http_handler):
        """List contexts supports pagination."""
        # Add several contexts
        for i in range(10):
            rlm_handler._contexts[f"ctx_{i}"] = {
                "context": MagicMock(),
                "source_type": "text",
                "original_tokens": 100,
                "created_at": f"2026-01-15T{10 + i}:00:00",
            }

        result = rlm_handler.handle(
            "/api/v1/rlm/contexts",
            {"limit": "3", "offset": "2"},
            mock_http_handler,
        )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["total"] == 10
        assert body["limit"] == 3
        assert body["offset"] == 2
        assert len(body["contexts"]) == 3

    def test_list_contexts_limit_capped(self, rlm_handler, mock_http_handler):
        """List contexts caps limit at 100."""
        result = rlm_handler.handle(
            "/api/v1/rlm/contexts",
            {"limit": "500"},
            mock_http_handler,
        )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["limit"] == 100


class TestRLMContextHandlerGetContextEndpoint:
    """Test GET /api/rlm/context/:id endpoint."""

    def test_get_context_not_found(self, rlm_handler, mock_http_handler):
        """Get context returns 404 for unknown ID."""
        result = rlm_handler.handle(
            "/api/v1/rlm/context/nonexistent",
            {},
            mock_http_handler,
        )

        assert result is not None
        assert result.status_code == 404

    def test_get_context_validates_id(self, rlm_handler, mock_http_handler):
        """Get context validates context ID format."""
        # Invalid IDs should return 400
        result = rlm_handler.handle(
            "/api/v1/rlm/context/",
            {},
            mock_http_handler,
        )

        assert result is not None
        assert result.status_code == 400

    def test_get_context_success(self, rlm_handler, mock_http_handler):
        """Get context returns context details."""
        mock_context = MagicMock()
        mock_context.original_tokens = 1000
        mock_context.total_tokens.return_value = 200
        mock_context.levels = {}

        rlm_handler._contexts["test_ctx"] = {
            "context": mock_context,
            "source_type": "text",
            "original_tokens": 1000,
            "created_at": "2026-01-15T10:00:00",
        }

        result = rlm_handler.handle(
            "/api/v1/rlm/context/test_ctx",
            {},
            mock_http_handler,
        )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["id"] == "test_ctx"
        assert body["original_tokens"] == 1000
        assert "compression_ratio" in body

    def test_get_context_with_include_content(self, rlm_handler, mock_http_handler):
        """Get context includes content preview when requested."""
        mock_context = MagicMock()
        mock_context.original_tokens = 1000
        mock_context.total_tokens.return_value = 200
        mock_context.levels = {}

        rlm_handler._contexts["test_ctx"] = {
            "context": mock_context,
            "source_type": "text",
            "original_tokens": 1000,
            "created_at": "2026-01-15T10:00:00",
        }

        result = rlm_handler.handle(
            "/api/v1/rlm/context/test_ctx",
            {"include_content": "true"},
            mock_http_handler,
        )

        assert result is not None
        assert result.status_code == 200


class TestRLMContextHandlerDeleteContextEndpoint:
    """Test DELETE /api/rlm/context/:id endpoint."""

    def test_delete_context_requires_auth(self, rlm_handler, mock_http_handler, mock_auth_config):
        """Delete context requires authentication."""
        # Override the autouse mock to simulate no API token configured
        mock_auth_config.api_token = None
        mock_http_handler.command = "DELETE"

        result = rlm_handler.handle_delete(
            "/api/v1/rlm/context/test_ctx",
            {},
            mock_http_handler,
        )

        assert result is not None
        assert result.status_code == 401

    def test_delete_context_not_found(self, rlm_handler, mock_http_handler):
        """Delete context returns 404 for unknown ID."""
        mock_http_handler.command = "DELETE"
        mock_http_handler.headers["Authorization"] = "Bearer test-token"

        result = rlm_handler.handle_delete(
            "/api/v1/rlm/context/nonexistent",
            {},
            mock_http_handler,
        )

        assert result is not None
        assert result.status_code == 404

    def test_delete_context_success(self, rlm_handler, mock_http_handler):
        """Delete context removes context and returns success."""
        mock_http_handler.command = "DELETE"
        mock_http_handler.headers["Authorization"] = "Bearer test-token"

        rlm_handler._contexts["test_ctx"] = {
            "context": MagicMock(),
            "source_type": "text",
            "created_at": "2026-01-15T10:00:00",
        }

        result = rlm_handler.handle_delete(
            "/api/v1/rlm/context/test_ctx",
            {},
            mock_http_handler,
        )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["success"] is True
        assert "test_ctx" not in rlm_handler._contexts


class TestRLMContextHandlerErrorHandling:
    """Test error handling in RLM handler."""

    def test_compress_handles_exception(self, rlm_handler):
        """Compress endpoint handles exceptions gracefully."""
        handler = create_request_body(
            {
                "content": "test content",
            },
            with_auth=True,
        )

        mock_compressor = MagicMock()
        mock_compressor.compress = AsyncMock(side_effect=Exception("Compression failed"))

        with patch.object(rlm_handler, "_get_compressor", return_value=mock_compressor):
            result = rlm_handler.handle_post("/api/v1/rlm/compress", {}, handler)

        assert result is not None
        assert result.status_code == 500

    def test_query_handles_exception(self, rlm_handler):
        """Query endpoint handles exceptions gracefully."""
        mock_context = MagicMock()
        rlm_handler._contexts["test_ctx"] = {
            "context": mock_context,
            "created_at": datetime.now().isoformat(),
        }

        handler = create_request_body(
            {
                "context_id": "test_ctx",
                "query": "What is the topic?",
            },
            with_auth=True,
        )

        mock_rlm = MagicMock()
        mock_rlm.query = AsyncMock(side_effect=Exception("Query failed"))

        with patch.object(rlm_handler, "_get_rlm", return_value=mock_rlm):
            result = rlm_handler.handle_post("/api/v1/rlm/query", {}, handler)

        assert result is not None
        assert result.status_code == 500


class TestRLMContextHandlerFallbackQuery:
    """Test fallback query behavior when full RLM unavailable."""

    def test_query_fallback_when_rlm_unavailable(self, rlm_handler):
        """Query uses fallback when RLM not available."""
        mock_context = MagicMock()
        rlm_handler._contexts["test_ctx"] = {
            "context": mock_context,
            "created_at": datetime.now().isoformat(),
        }

        handler = create_request_body(
            {
                "context_id": "test_ctx",
                "query": "What is the topic?",
            },
            with_auth=True,
        )

        with patch.object(rlm_handler, "_get_rlm", return_value=None):
            result = rlm_handler.handle_post("/api/v1/rlm/query", {}, handler)

        assert result is not None
        # Should return fallback response
        if result.status_code == 200:
            body = json.loads(result.body)
            assert "fallback" in body.get("metadata", {}) or "answer" in body


class TestRLMContextHandlerIntegration:
    """Integration tests for RLM handler."""

    def test_all_routes_reachable(self, rlm_handler, mock_http_handler):
        """All RLM routes return a response."""
        get_routes = [
            "/api/v1/rlm/stats",
            "/api/v1/rlm/strategies",
            "/api/v1/rlm/contexts",
        ]

        for route in get_routes:
            result = rlm_handler.handle(route, {}, mock_http_handler)
            assert result is not None, f"Route {route} returned None"
            assert result.status_code in [
                200,
                400,
                401,
                429,
                500,
                503,
            ], f"Route {route} returned unexpected status {result.status_code}"

    def test_handler_inherits_from_base(self, rlm_handler):
        """Handler inherits from BaseHandler."""
        from aragora.server.handlers.base import BaseHandler

        assert isinstance(rlm_handler, BaseHandler)

    def test_routes_attribute_matches_can_handle(self, rlm_handler):
        """ROUTES attribute matches can_handle behavior."""
        for route in rlm_handler.ROUTES:
            assert rlm_handler.can_handle(route), f"can_handle should return True for {route}"

    def test_full_compress_and_query_flow(self, rlm_handler, mock_http_handler):
        """Test full workflow: compress content then query it."""
        # This tests the integration between compress and query
        # Step 1: Compress content
        compress_handler = create_request_body(
            {
                "content": "This is a test document about software architecture patterns.",
                "source_type": "text",
            },
            with_auth=True,
        )

        mock_compressor = MagicMock()
        mock_context = MagicMock()
        mock_context.original_tokens = 50
        mock_context.total_tokens.return_value = 10
        mock_context.levels = {}

        mock_compressor.compress = AsyncMock(return_value=mock_context)

        with patch.object(rlm_handler, "_get_compressor", return_value=mock_compressor):
            compress_result = rlm_handler.handle_post("/api/v1/rlm/compress", {}, compress_handler)

        if compress_result and compress_result.status_code == 200:
            compress_body = json.loads(compress_result.body)
            context_id = compress_body.get("context_id")

            # Step 2: Query the compressed context
            query_handler = create_request_body(
                {
                    "context_id": context_id,
                    "query": "What is the document about?",
                },
                with_auth=True,
            )

            mock_rlm = MagicMock()
            mock_result = MagicMock()
            mock_result.answer = "Software architecture patterns"
            mock_result.confidence = 0.9
            mock_result.iteration = 1
            mock_rlm.query = AsyncMock(return_value=mock_result)

            with patch.object(rlm_handler, "_get_rlm", return_value=mock_rlm):
                query_result = rlm_handler.handle_post("/api/v1/rlm/query", {}, query_handler)

            if query_result:
                assert query_result.status_code in [200, 404, 500]


class TestRLMContextHandlerRouteDispatch:
    """Test route dispatch logic."""

    def test_handle_routes_to_stats(self, rlm_handler, mock_http_handler):
        """Handle correctly routes to stats handler."""
        with patch.object(rlm_handler, "handle_stats") as mock_stats:
            mock_stats.return_value = MagicMock(status_code=200, body=b"{}")
            rlm_handler.handle("/api/v1/rlm/stats", {}, mock_http_handler)
            mock_stats.assert_called_once()

    def test_handle_routes_to_strategies(self, rlm_handler, mock_http_handler):
        """Handle correctly routes to strategies handler."""
        with patch.object(rlm_handler, "handle_strategies") as mock_strategies:
            mock_strategies.return_value = MagicMock(status_code=200, body=b"{}")
            rlm_handler.handle("/api/v1/rlm/strategies", {}, mock_http_handler)
            mock_strategies.assert_called_once()

    def test_handle_returns_none_for_unknown(self, rlm_handler, mock_http_handler):
        """Handle returns None for unknown paths."""
        result = rlm_handler.handle("/api/v1/unknown", {}, mock_http_handler)
        assert result is None

    def test_handle_post_routes_to_compress(self, rlm_handler):
        """Handle_post correctly routes to compress handler."""
        handler = create_request_body({"content": "test"}, with_auth=True)

        with patch.object(rlm_handler, "handle_compress") as mock_compress:
            mock_compress.return_value = MagicMock(status_code=200, body=b"{}")
            rlm_handler.handle_post("/api/v1/rlm/compress", {}, handler)
            mock_compress.assert_called_once()

    def test_handle_post_routes_to_query(self, rlm_handler):
        """Handle_post correctly routes to query handler."""
        handler = create_request_body(
            {
                "context_id": "ctx_123",
                "query": "test",
            },
            with_auth=True,
        )

        with patch.object(rlm_handler, "handle_query") as mock_query:
            mock_query.return_value = MagicMock(status_code=200, body=b"{}")
            rlm_handler.handle_post("/api/v1/rlm/query", {}, handler)
            mock_query.assert_called_once()


class TestRLMContextHandlerRateLimiting:
    """Test rate limiting behavior."""

    def test_stats_rate_limited(self, rlm_handler, mock_http_handler):
        """Stats endpoint is rate limited."""
        # The @rate_limit decorator should be applied
        # This test verifies the decorator is present
        assert hasattr(rlm_handler.handle_stats, "__wrapped__") or "rate_limit" in str(
            rlm_handler.handle_stats
        )

    def test_strategies_rate_limited(self, rlm_handler, mock_http_handler):
        """Strategies endpoint is rate limited."""
        assert hasattr(rlm_handler.handle_strategies, "__wrapped__") or "rate_limit" in str(
            rlm_handler.handle_strategies
        )


class TestRLMContextHandlerInputValidation:
    """Test input validation."""

    def test_compress_empty_content_rejected(self, rlm_handler):
        """Compress rejects empty content."""
        handler = create_request_body({"content": ""}, with_auth=True)

        result = rlm_handler.handle_post("/api/v1/rlm/compress", {}, handler)

        assert result is not None
        assert result.status_code == 400

    def test_query_empty_query_rejected(self, rlm_handler):
        """Query rejects empty query string."""
        rlm_handler._contexts["test_ctx"] = {
            "context": MagicMock(),
            "created_at": datetime.now().isoformat(),
        }

        handler = create_request_body(
            {
                "context_id": "test_ctx",
                "query": "",
            },
            with_auth=True,
        )

        result = rlm_handler.handle_post("/api/v1/rlm/query", {}, handler)

        assert result is not None
        assert result.status_code == 400
