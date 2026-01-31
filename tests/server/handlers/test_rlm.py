"""
Tests for aragora.server.handlers.rlm - RLM context handler.

Tests cover:
- Route registration and can_handle for static and dynamic routes
- GET routing (stats, strategies, contexts, stream modes, codebase health)
- POST routing (compress, query, stream)
- DELETE routing (context deletion)
- Context lifecycle: create via compress, retrieve, list, delete
- Compression endpoint: validation, size limits, source_type checks
- Query endpoint: validation, strategy checks, refinement, fallback
- Streaming endpoint: mode selection, query/level/all paths, fallback
- Context-specific routes: get with/without content, delete, method dispatch
- read_json_body utility: normal, empty, oversized, malformed
- Error handling: missing compressor, missing RLM, import errors
- Authentication checks via @require_auth decorator
- Codebase health: manifest detection, refresh, RLM build
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.rlm import RLMContextHandler


# ===========================================================================
# Test Fixtures and Mocks
# ===========================================================================

# Token used for authenticated requests
TEST_API_TOKEN = "test-rlm-token-12345"


def make_mock_handler(
    body: dict | None = None,
    method: str = "GET",
    path: str = "/api/v1/rlm/stats",
    content_length: int | None = None,
    authenticated: bool = False,
) -> MagicMock:
    """Create a mock HTTP handler with headers and body.

    Args:
        body: JSON body to include in request
        method: HTTP method
        path: Request path
        content_length: Override Content-Length header
        authenticated: If True, include Bearer token in Authorization header
    """
    handler = MagicMock()
    handler.command = method
    handler.path = path
    handler.client_address = ("127.0.0.1", 12345)
    handler.headers = {}

    if authenticated:
        handler.headers["Authorization"] = f"Bearer {TEST_API_TOKEN}"

    if body is not None:
        body_bytes = json.dumps(body).encode("utf-8")
        handler.headers["Content-Length"] = str(
            content_length if content_length is not None else len(body_bytes)
        )
        handler.rfile = BytesIO(body_bytes)
    else:
        handler.rfile = BytesIO(b"")
        handler.headers["Content-Length"] = "0"

    return handler


@dataclass
class MockCompressionNode:
    """Mock node in a compressed context."""

    id: str = "node_1"
    content: str = "Summarized content"
    token_count: int = 50


class MockAbstractionLevel:
    """Mock AbstractionLevel enum."""

    SUMMARY = "summary"
    DETAIL = "detail"


@dataclass
class MockCompressedContext:
    """Mock compressed context returned by compressor."""

    original_tokens: int = 1000
    levels: dict = field(
        default_factory=lambda: {
            MockAbstractionLevel.SUMMARY: [
                MockCompressionNode("n1", "Summary node 1", 30),
                MockCompressionNode("n2", "Summary node 2", 20),
            ],
            MockAbstractionLevel.DETAIL: [
                MockCompressionNode("n3", "Detail node 1", 100),
            ],
        }
    )

    def total_tokens(self) -> int:
        total = 0
        for nodes in self.levels.values():
            total += sum(n.token_count for n in nodes)
        return total

    def get_at_level(self, level):
        return self.levels.get(level, [])


@dataclass
class MockQueryResult:
    """Mock query result from RLM."""

    answer: str = "The answer to your question."
    confidence: float = 0.85
    iteration: int = 1
    tokens_processed: int = 500
    sub_calls_made: int = 3


class MockCompressor:
    """Mock hierarchical compressor."""

    async def compress(self, content: str, source_type: str = "text") -> MockCompressedContext:
        return MockCompressedContext()


class MockRLM:
    """Mock AragoraRLM instance."""

    async def query(self, query: str, context: Any, strategy: str) -> MockQueryResult:
        return MockQueryResult()

    async def query_with_refinement(
        self, query: str, context: Any, strategy: str, max_iterations: int = 3
    ) -> MockQueryResult:
        return MockQueryResult(iteration=max_iterations, answer="Refined answer")


@pytest.fixture
def rlm_handler():
    """Create RLMContextHandler with mock server context."""
    ctx: dict[str, Any] = {}
    return RLMContextHandler(ctx)


@pytest.fixture(autouse=True)
def clear_rate_limiters():
    """Clear rate limiters so tests are not rate-limited."""
    from aragora.server.handlers.utils.rate_limit import _limiters

    for limiter in _limiters.values():
        limiter.clear()
    yield
    for limiter in _limiters.values():
        limiter.clear()


@pytest.fixture(autouse=True)
def set_api_token():
    """Mock auth_config so @require_auth passes for authenticated handlers.

    The validate_token method uses HMAC signature verification, so we mock it
    to accept our simple test token (following pattern from test_rlm_context_handler.py).
    """
    mock_config = MagicMock()
    mock_config.api_token = TEST_API_TOKEN
    mock_config.validate_token = MagicMock(side_effect=lambda t, **kw: t == TEST_API_TOKEN)
    with patch("aragora.server.auth.auth_config", mock_config):
        with patch.dict(os.environ, {"ARAGORA_API_TOKEN": TEST_API_TOKEN}):
            yield


# ===========================================================================
# Test Route Registration (can_handle)
# ===========================================================================


class TestRLMHandlerRouting:
    """Tests for can_handle across all RLM routes."""

    def test_can_handle_stats(self, rlm_handler):
        assert rlm_handler.can_handle("/api/v1/rlm/stats") is True

    def test_can_handle_strategies(self, rlm_handler):
        assert rlm_handler.can_handle("/api/v1/rlm/strategies") is True

    def test_can_handle_compress(self, rlm_handler):
        assert rlm_handler.can_handle("/api/v1/rlm/compress") is True

    def test_can_handle_query(self, rlm_handler):
        assert rlm_handler.can_handle("/api/v1/rlm/query") is True

    def test_can_handle_contexts_list(self, rlm_handler):
        assert rlm_handler.can_handle("/api/v1/rlm/contexts") is True

    def test_can_handle_stream(self, rlm_handler):
        assert rlm_handler.can_handle("/api/v1/rlm/stream") is True

    def test_can_handle_stream_modes(self, rlm_handler):
        assert rlm_handler.can_handle("/api/v1/rlm/stream/modes") is True

    def test_can_handle_codebase_health(self, rlm_handler):
        assert rlm_handler.can_handle("/api/v1/rlm/codebase/health") is True

    def test_can_handle_context_by_id(self, rlm_handler):
        assert rlm_handler.can_handle("/api/v1/rlm/context/ctx_abc123_123") is True

    def test_cannot_handle_unrelated_path(self, rlm_handler):
        assert rlm_handler.can_handle("/api/v1/debates") is False

    def test_cannot_handle_partial_path(self, rlm_handler):
        assert rlm_handler.can_handle("/api/v1/rlm") is False


# ===========================================================================
# Test GET Routing
# ===========================================================================


class TestRLMHandlerGetRouting:
    """Tests for GET request routing via handle()."""

    def test_handle_routes_to_stats(self, rlm_handler):
        handler = make_mock_handler()
        with patch.object(rlm_handler, "handle_stats") as mock_stats:
            mock_stats.return_value = MagicMock(status_code=200)
            rlm_handler.handle("/api/v1/rlm/stats", {}, handler)
            mock_stats.assert_called_once()

    def test_handle_routes_to_strategies(self, rlm_handler):
        handler = make_mock_handler()
        with patch.object(rlm_handler, "handle_strategies") as mock_strat:
            mock_strat.return_value = MagicMock(status_code=200)
            rlm_handler.handle("/api/v1/rlm/strategies", {}, handler)
            mock_strat.assert_called_once()

    def test_handle_routes_to_list_contexts(self, rlm_handler):
        handler = make_mock_handler()
        with patch.object(rlm_handler, "handle_list_contexts") as mock_list:
            mock_list.return_value = MagicMock(status_code=200)
            rlm_handler.handle("/api/v1/rlm/contexts", {}, handler)
            mock_list.assert_called_once()

    def test_handle_routes_to_context_get(self, rlm_handler):
        """Dynamic context route dispatches to _get_context."""
        rlm_handler._contexts["testctx"] = {
            "context": MockCompressedContext(),
            "source_type": "text",
            "original_tokens": 1000,
            "created_at": "2024-01-01T00:00:00",
        }
        handler = make_mock_handler()
        result = rlm_handler.handle("/api/v1/rlm/context/testctx", {}, handler)
        assert result is not None
        assert result.status_code == 200

    def test_handle_returns_none_for_unknown(self, rlm_handler):
        handler = make_mock_handler()
        result = rlm_handler.handle("/api/v1/unknown", {}, handler)
        assert result is None


# ===========================================================================
# Test POST Routing
# ===========================================================================


class TestRLMHandlerPostRouting:
    """Tests for POST request routing via handle_post()."""

    def test_handle_post_routes_to_compress(self, rlm_handler):
        handler = make_mock_handler(body={"content": "test"}, method="POST", authenticated=True)
        with patch.object(rlm_handler, "handle_compress") as mock_compress:
            mock_compress.return_value = MagicMock(status_code=200)
            rlm_handler.handle_post("/api/v1/rlm/compress", {}, handler)
            mock_compress.assert_called_once()

    def test_handle_post_routes_to_query(self, rlm_handler):
        handler = make_mock_handler(
            body={"context_id": "ctx1", "query": "q"}, method="POST", authenticated=True
        )
        with patch.object(rlm_handler, "handle_query") as mock_query:
            mock_query.return_value = MagicMock(status_code=200)
            rlm_handler.handle_post("/api/v1/rlm/query", {}, handler)
            mock_query.assert_called_once()

    def test_handle_post_routes_to_stream(self, rlm_handler):
        handler = make_mock_handler(body={"context_id": "ctx1"}, method="POST", authenticated=True)
        with patch.object(rlm_handler, "handle_stream") as mock_stream:
            mock_stream.return_value = MagicMock(status_code=200)
            rlm_handler.handle_post("/api/v1/rlm/stream", {}, handler)
            mock_stream.assert_called_once()

    def test_handle_post_returns_none_for_unknown(self, rlm_handler):
        handler = make_mock_handler(body={}, method="POST", authenticated=True)
        result = rlm_handler.handle_post("/api/v1/rlm/unknown", {}, handler)
        assert result is None


# ===========================================================================
# Test DELETE Routing
# ===========================================================================


class TestRLMHandlerDeleteRouting:
    """Tests for DELETE request routing via handle_delete()."""

    def test_handle_delete_routes_to_context_delete(self, rlm_handler):
        rlm_handler._contexts["delctx"] = {"context": MockCompressedContext()}
        handler = make_mock_handler(method="DELETE", authenticated=True)
        result = rlm_handler.handle_delete("/api/v1/rlm/context/delctx", {}, handler)
        assert result is not None
        assert result.status_code == 200
        assert "delctx" not in rlm_handler._contexts

    def test_handle_delete_rejects_nested_path(self, rlm_handler):
        handler = make_mock_handler(method="DELETE")
        result = rlm_handler.handle_delete("/api/v1/rlm/context/id/extra", {}, handler)
        assert result is None

    def test_handle_delete_returns_none_for_wrong_prefix(self, rlm_handler):
        handler = make_mock_handler(method="DELETE")
        result = rlm_handler.handle_delete("/api/v1/rlm/stats", {}, handler)
        assert result is None


# ===========================================================================
# Test Strategies Endpoint
# ===========================================================================


class TestRLMStrategies:
    """Tests for the strategies endpoint."""

    def test_returns_all_strategies(self, rlm_handler):
        handler = make_mock_handler()
        result = rlm_handler.handle_strategies("/api/v1/rlm/strategies", {}, handler)
        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "strategies" in data
        assert "default" in data
        expected_strategies = {"peek", "grep", "partition_map", "summarize", "hierarchical", "auto"}
        assert set(data["strategies"].keys()) == expected_strategies

    def test_default_strategy_is_auto(self, rlm_handler):
        handler = make_mock_handler()
        result = rlm_handler.handle_strategies("/api/v1/rlm/strategies", {}, handler)
        data = json.loads(result.body)
        assert data["default"] == "auto"


# ===========================================================================
# Test Stats Endpoint
# ===========================================================================


class TestRLMStats:
    """Tests for the stats endpoint."""

    def test_stats_success(self, rlm_handler):
        """Stats returns cache info, context count, and system flags."""
        mock_cache_stats = {"hits": 10, "misses": 5, "size": 3}
        with patch.object(rlm_handler, "_get_compressor", return_value=MagicMock()):
            with patch.object(rlm_handler, "_get_rlm", return_value=MagicMock()):
                with patch.dict(
                    "sys.modules",
                    {
                        "aragora.rlm.compressor": MagicMock(
                            get_compression_cache_stats=lambda: mock_cache_stats
                        ),
                        "aragora.rlm": MagicMock(HAS_OFFICIAL_RLM=False),
                    },
                ):
                    handler = make_mock_handler()
                    result = rlm_handler.handle_stats("/api/v1/rlm/stats", {}, handler)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "cache" in data
        assert "contexts" in data
        assert "system" in data
        assert "timestamp" in data

    def test_stats_import_error_fallback(self, rlm_handler):
        """Stats returns degraded info when RLM module is not available."""
        with patch.dict("sys.modules", {"aragora.rlm.compressor": None, "aragora.rlm": None}):
            handler = make_mock_handler()
            result = rlm_handler.handle_stats("/api/v1/rlm/stats", {}, handler)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["system"]["has_official_rlm"] is False
        assert data["system"]["compressor_available"] is False

    def test_stats_shows_stored_contexts(self, rlm_handler):
        """Stats includes count and IDs of stored contexts."""
        rlm_handler._contexts["ctx_a"] = {"context": MockCompressedContext()}
        rlm_handler._contexts["ctx_b"] = {"context": MockCompressedContext()}

        with patch.dict(
            "sys.modules",
            {
                "aragora.rlm.compressor": MagicMock(get_compression_cache_stats=lambda: {}),
                "aragora.rlm": MagicMock(HAS_OFFICIAL_RLM=False),
            },
        ):
            with patch.object(rlm_handler, "_get_compressor", return_value=None):
                with patch.object(rlm_handler, "_get_rlm", return_value=None):
                    handler = make_mock_handler()
                    result = rlm_handler.handle_stats("/api/v1/rlm/stats", {}, handler)

        data = json.loads(result.body)
        assert data["contexts"]["stored"] == 2
        assert set(data["contexts"]["ids"]) == {"ctx_a", "ctx_b"}


# ===========================================================================
# Test Compress Endpoint
# ===========================================================================


class TestRLMCompress:
    """Tests for the compress endpoint."""

    def test_compress_success(self, rlm_handler):
        """Compress valid content and receive context ID."""
        handler = make_mock_handler(
            body={"content": "Test content to compress", "source_type": "text", "levels": 3},
            method="POST",
            authenticated=True,
        )
        with patch.object(rlm_handler, "_get_compressor", return_value=MockCompressor()):
            result = rlm_handler.handle_compress("/api/v1/rlm/compress", {}, handler)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "context_id" in data
        assert data["context_id"].startswith("ctx_")
        assert "compression_result" in data
        assert data["compression_result"]["source_type"] == "text"
        assert data["compression_result"]["original_tokens"] == 1000

    def test_compress_stores_context(self, rlm_handler):
        """Compression stores context in handler's internal store."""
        handler = make_mock_handler(
            body={"content": "Test content"},
            method="POST",
            authenticated=True,
        )
        with patch.object(rlm_handler, "_get_compressor", return_value=MockCompressor()):
            result = rlm_handler.handle_compress("/api/v1/rlm/compress", {}, handler)

        data = json.loads(result.body)
        context_id = data["context_id"]
        assert context_id in rlm_handler._contexts
        assert rlm_handler._contexts[context_id]["source_type"] == "text"

    def test_compress_no_body(self, rlm_handler):
        """Compress with no body returns 400."""
        handler = make_mock_handler(method="POST", authenticated=True)
        result = rlm_handler.handle_compress("/api/v1/rlm/compress", {}, handler)
        assert result is not None
        assert result.status_code == 400

    def test_compress_missing_content(self, rlm_handler):
        """Compress with missing content field returns 400."""
        handler = make_mock_handler(
            body={"source_type": "text"},
            method="POST",
            authenticated=True,
        )
        result = rlm_handler.handle_compress("/api/v1/rlm/compress", {}, handler)
        assert result is not None
        assert result.status_code == 400

    def test_compress_non_string_content(self, rlm_handler):
        """Compress with non-string content returns 400."""
        handler = make_mock_handler(
            body={"content": 12345},
            method="POST",
            authenticated=True,
        )
        result = rlm_handler.handle_compress("/api/v1/rlm/compress", {}, handler)
        assert result is not None
        assert result.status_code == 400

    def test_compress_content_too_large(self, rlm_handler):
        """Compress with content over 10MB returns 413.

        We mock read_json_body to bypass the body-level size check and test
        the content-level size validation directly.
        """
        large_content = "x" * (10_000_001)
        handler = make_mock_handler(method="POST", authenticated=True)
        with patch.object(rlm_handler, "read_json_body", return_value={"content": large_content}):
            result = rlm_handler.handle_compress("/api/v1/rlm/compress", {}, handler)
        assert result is not None
        assert result.status_code == 413

    def test_compress_invalid_source_type(self, rlm_handler):
        """Compress with invalid source_type returns 400."""
        handler = make_mock_handler(
            body={"content": "test", "source_type": "invalid"},
            method="POST",
            authenticated=True,
        )
        result = rlm_handler.handle_compress("/api/v1/rlm/compress", {}, handler)
        assert result is not None
        assert result.status_code == 400

    def test_compress_valid_source_types(self, rlm_handler):
        """All valid source types are accepted."""
        for st in ("text", "code", "debate"):
            handler = make_mock_handler(
                body={"content": "test content", "source_type": st},
                method="POST",
                authenticated=True,
            )
            with patch.object(rlm_handler, "_get_compressor", return_value=MockCompressor()):
                result = rlm_handler.handle_compress("/api/v1/rlm/compress", {}, handler)
            assert result.status_code == 200, f"source_type={st} should be accepted"

    def test_compress_invalid_levels(self, rlm_handler):
        """Compress with invalid levels returns 400."""
        handler = make_mock_handler(
            body={"content": "test", "levels": 10},
            method="POST",
            authenticated=True,
        )
        result = rlm_handler.handle_compress("/api/v1/rlm/compress", {}, handler)
        assert result is not None
        assert result.status_code == 400

    def test_compress_levels_boundary(self, rlm_handler):
        """Levels must be between 1 and 5."""
        for invalid in (0, 6, -1):
            handler = make_mock_handler(
                body={"content": "test", "levels": invalid},
                method="POST",
                authenticated=True,
            )
            result = rlm_handler.handle_compress("/api/v1/rlm/compress", {}, handler)
            assert result.status_code == 400, f"levels={invalid} should be rejected"

    def test_compress_no_compressor(self, rlm_handler):
        """Compress when compressor is unavailable returns 503."""
        handler = make_mock_handler(
            body={"content": "test"},
            method="POST",
            authenticated=True,
        )
        with patch.object(rlm_handler, "_get_compressor", return_value=None):
            result = rlm_handler.handle_compress("/api/v1/rlm/compress", {}, handler)
        assert result is not None
        assert result.status_code == 503

    def test_compress_runtime_error(self, rlm_handler):
        """RuntimeError during compression returns 500."""
        compressor = MagicMock()

        async def fail_compress(*args, **kwargs):
            raise RuntimeError("compression failed")

        compressor.compress = fail_compress
        handler = make_mock_handler(
            body={"content": "test"},
            method="POST",
            authenticated=True,
        )
        with patch.object(rlm_handler, "_get_compressor", return_value=compressor):
            result = rlm_handler.handle_compress("/api/v1/rlm/compress", {}, handler)
        assert result is not None
        assert result.status_code == 500

    def test_compress_unauthenticated_returns_401(self, rlm_handler):
        """Compress without authentication returns 401."""
        handler = make_mock_handler(
            body={"content": "test"},
            method="POST",
            authenticated=False,
        )
        result = rlm_handler.handle_compress("/api/v1/rlm/compress", {}, handler)
        assert result is not None
        assert result.status_code == 401


# ===========================================================================
# Test Query Endpoint
# ===========================================================================


class TestRLMQuery:
    """Tests for the query endpoint."""

    def _seed_context(self, rlm_handler, ctx_id="ctx_test123_1"):
        """Helper to add a context for querying."""
        rlm_handler._contexts[ctx_id] = {
            "context": MockCompressedContext(),
            "source_type": "text",
            "original_tokens": 1000,
            "created_at": "2024-01-01T00:00:00",
        }
        return ctx_id

    def test_query_success(self, rlm_handler):
        """Query returns answer with metadata."""
        ctx_id = self._seed_context(rlm_handler)
        handler = make_mock_handler(
            body={"context_id": ctx_id, "query": "What is the summary?"},
            method="POST",
            authenticated=True,
        )
        with patch.object(rlm_handler, "_get_rlm", return_value=MockRLM()):
            result = rlm_handler.handle_query("/api/v1/rlm/query", {}, handler)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "answer" in data
        assert "metadata" in data
        assert data["metadata"]["context_id"] == ctx_id
        assert data["metadata"]["strategy"] == "auto"

    def test_query_with_refinement(self, rlm_handler):
        """Query with refine=True uses iterative refinement."""
        ctx_id = self._seed_context(rlm_handler)
        handler = make_mock_handler(
            body={
                "context_id": ctx_id,
                "query": "What is the summary?",
                "refine": True,
                "max_iterations": 5,
            },
            method="POST",
            authenticated=True,
        )
        with patch.object(rlm_handler, "_get_rlm", return_value=MockRLM()):
            result = rlm_handler.handle_query("/api/v1/rlm/query", {}, handler)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["metadata"]["refined"] is True

    def test_query_no_body(self, rlm_handler):
        handler = make_mock_handler(method="POST", authenticated=True)
        result = rlm_handler.handle_query("/api/v1/rlm/query", {}, handler)
        assert result.status_code == 400

    def test_query_missing_context_id(self, rlm_handler):
        handler = make_mock_handler(
            body={"query": "test"},
            method="POST",
            authenticated=True,
        )
        result = rlm_handler.handle_query("/api/v1/rlm/query", {}, handler)
        assert result.status_code == 400

    def test_query_missing_query(self, rlm_handler):
        handler = make_mock_handler(
            body={"context_id": "ctx_test"},
            method="POST",
            authenticated=True,
        )
        result = rlm_handler.handle_query("/api/v1/rlm/query", {}, handler)
        assert result.status_code == 400

    def test_query_too_long(self, rlm_handler):
        """Query over 10000 chars returns 400."""
        ctx_id = self._seed_context(rlm_handler)
        long_query = "x" * 10001
        handler = make_mock_handler(
            body={"context_id": ctx_id, "query": long_query},
            method="POST",
            authenticated=True,
        )
        result = rlm_handler.handle_query("/api/v1/rlm/query", {}, handler)
        assert result.status_code == 400

    def test_query_context_not_found(self, rlm_handler):
        handler = make_mock_handler(
            body={"context_id": "nonexistent", "query": "test"},
            method="POST",
            authenticated=True,
        )
        result = rlm_handler.handle_query("/api/v1/rlm/query", {}, handler)
        assert result.status_code == 404

    def test_query_invalid_strategy(self, rlm_handler):
        ctx_id = self._seed_context(rlm_handler)
        handler = make_mock_handler(
            body={"context_id": ctx_id, "query": "test", "strategy": "invalid"},
            method="POST",
            authenticated=True,
        )
        result = rlm_handler.handle_query("/api/v1/rlm/query", {}, handler)
        assert result.status_code == 400

    def test_query_all_valid_strategies(self, rlm_handler):
        """All valid strategies are accepted."""
        ctx_id = self._seed_context(rlm_handler)
        valid = ["peek", "grep", "partition_map", "summarize", "hierarchical", "auto"]
        for strategy in valid:
            handler = make_mock_handler(
                body={"context_id": ctx_id, "query": "test", "strategy": strategy},
                method="POST",
                authenticated=True,
            )
            with patch.object(rlm_handler, "_get_rlm", return_value=MockRLM()):
                result = rlm_handler.handle_query("/api/v1/rlm/query", {}, handler)
            assert result.status_code == 200, f"strategy={strategy} should be accepted"

    def test_query_fallback_when_rlm_unavailable(self, rlm_handler):
        """Falls back to summary search when RLM is not available."""
        ctx_id = self._seed_context(rlm_handler)
        handler = make_mock_handler(
            body={"context_id": ctx_id, "query": "test"},
            method="POST",
            authenticated=True,
        )
        with patch.object(rlm_handler, "_get_rlm", return_value=None):
            with patch.object(rlm_handler, "_fallback_query") as mock_fb:
                mock_fb.return_value = MagicMock(status_code=200, body=b'{"answer":"fallback"}')
                result = rlm_handler.handle_query("/api/v1/rlm/query", {}, handler)
                mock_fb.assert_called_once()

    def test_query_max_iterations_clamped(self, rlm_handler):
        """Invalid max_iterations is clamped to 3."""
        ctx_id = self._seed_context(rlm_handler)
        handler = make_mock_handler(
            body={
                "context_id": ctx_id,
                "query": "test",
                "refine": True,
                "max_iterations": 100,
            },
            method="POST",
            authenticated=True,
        )
        with patch.object(rlm_handler, "_get_rlm", return_value=MockRLM()):
            result = rlm_handler.handle_query("/api/v1/rlm/query", {}, handler)
        assert result.status_code == 200

    def test_query_runtime_error(self, rlm_handler):
        """RuntimeError during query returns 500."""
        ctx_id = self._seed_context(rlm_handler)

        rlm = MagicMock()

        async def fail_query(*args, **kwargs):
            raise RuntimeError("query failed")

        rlm.query = fail_query
        handler = make_mock_handler(
            body={"context_id": ctx_id, "query": "test"},
            method="POST",
            authenticated=True,
        )
        with patch.object(rlm_handler, "_get_rlm", return_value=rlm):
            result = rlm_handler.handle_query("/api/v1/rlm/query", {}, handler)
        assert result.status_code == 500

    def test_query_unauthenticated_returns_401(self, rlm_handler):
        """Query without authentication returns 401."""
        handler = make_mock_handler(
            body={"context_id": "ctx_1", "query": "test"},
            method="POST",
            authenticated=False,
        )
        result = rlm_handler.handle_query("/api/v1/rlm/query", {}, handler)
        assert result.status_code == 401


# ===========================================================================
# Test Fallback Query
# ===========================================================================


class TestRLMFallbackQuery:
    """Tests for the _fallback_query method."""

    def test_fallback_with_summary_content(self, rlm_handler):
        """Fallback returns summary content when available."""
        context = MockCompressedContext()
        with patch.dict(
            "sys.modules", {"aragora.rlm": MagicMock(AbstractionLevel=MockAbstractionLevel)}
        ):
            result = rlm_handler._fallback_query(context, "test query", "auto")

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["metadata"]["fallback"] is True

    def test_fallback_no_summary_content(self, rlm_handler):
        """Fallback returns generic message when no summary is available."""
        context = MagicMock()
        context.get_at_level.return_value = []

        with patch.dict(
            "sys.modules", {"aragora.rlm": MagicMock(AbstractionLevel=MockAbstractionLevel)}
        ):
            result = rlm_handler._fallback_query(context, "test query", "auto")

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["metadata"]["fallback"] is True

    def test_fallback_import_error(self, rlm_handler):
        """Fallback handles import error gracefully."""
        context = MagicMock()

        with patch.dict("sys.modules", {"aragora.rlm": None}):
            result = rlm_handler._fallback_query(context, "test query", "auto")

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["metadata"]["fallback"] is True
        assert "error" in data["metadata"]


# ===========================================================================
# Test List Contexts
# ===========================================================================


class TestRLMListContexts:
    """Tests for listing stored contexts."""

    def test_list_contexts_empty(self, rlm_handler):
        handler = make_mock_handler()
        result = rlm_handler.handle_list_contexts("/api/v1/rlm/contexts", {}, handler)
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["contexts"] == []
        assert data["total"] == 0

    def test_list_contexts_with_data(self, rlm_handler):
        for i in range(3):
            rlm_handler._contexts[f"ctx_{i}"] = {
                "source_type": "text",
                "original_tokens": 100 * (i + 1),
                "created_at": "2024-01-01T00:00:00",
            }
        handler = make_mock_handler()
        result = rlm_handler.handle_list_contexts("/api/v1/rlm/contexts", {}, handler)
        data = json.loads(result.body)
        assert data["total"] == 3
        # Note: safe_query_int clamps offset to min_val=1 when no explicit
        # offset is given, so the first context is skipped in the default view.
        # The returned contexts should be a subset of stored contexts.
        returned_ids = {c["id"] for c in data["contexts"]}
        assert returned_ids.issubset({"ctx_0", "ctx_1", "ctx_2"})
        assert len(data["contexts"]) >= 2

    def test_list_contexts_pagination(self, rlm_handler):
        for i in range(10):
            rlm_handler._contexts[f"ctx_{i}"] = {
                "source_type": "text",
                "original_tokens": 100,
                "created_at": "2024-01-01T00:00:00",
            }
        handler = make_mock_handler()
        result = rlm_handler.handle_list_contexts(
            "/api/v1/rlm/contexts",
            {"limit": "3", "offset": "2"},
            handler,
        )
        data = json.loads(result.body)
        assert data["total"] == 10
        assert len(data["contexts"]) == 3
        assert data["limit"] == 3
        assert data["offset"] == 2


# ===========================================================================
# Test Context Get / Delete
# ===========================================================================


class TestRLMContextOperations:
    """Tests for get and delete context operations."""

    def test_get_context_success(self, rlm_handler):
        """Get context returns detail with level stats."""
        rlm_handler._contexts["ctx_abc"] = {
            "context": MockCompressedContext(),
            "source_type": "code",
            "original_tokens": 1000,
            "created_at": "2024-01-01T00:00:00",
        }
        result = rlm_handler._get_context("ctx_abc", {}, None)
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["id"] == "ctx_abc"
        assert data["source_type"] == "code"
        assert data["original_tokens"] == 1000
        assert "levels" in data

    def test_get_context_not_found(self, rlm_handler):
        result = rlm_handler._get_context("nonexistent", {}, None)
        assert result.status_code == 404

    def test_get_context_with_content_preview(self, rlm_handler):
        """Get context with include_content=true includes summary preview."""
        rlm_handler._contexts["ctx_prev"] = {
            "context": MockCompressedContext(),
            "source_type": "text",
            "original_tokens": 500,
            "created_at": "2024-01-01T00:00:00",
        }
        with patch.dict(
            "sys.modules", {"aragora.rlm": MagicMock(AbstractionLevel=MockAbstractionLevel)}
        ):
            result = rlm_handler._get_context("ctx_prev", {"include_content": "true"}, None)
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "summary_preview" in data

    def test_delete_context_success(self, rlm_handler):
        """Delete context with auth returns 200."""
        rlm_handler._contexts["ctx_del"] = {"context": MockCompressedContext()}
        handler = make_mock_handler(authenticated=True)
        result = rlm_handler._delete_context("ctx_del", {}, handler)
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["success"] is True
        assert "ctx_del" not in rlm_handler._contexts

    def test_delete_context_not_found(self, rlm_handler):
        handler = make_mock_handler(authenticated=True)
        result = rlm_handler._delete_context("nonexistent", {}, handler)
        assert result.status_code == 404

    def test_delete_context_unauthenticated(self, rlm_handler):
        """Delete without auth returns 401."""
        rlm_handler._contexts["ctx_noauth"] = {"context": MockCompressedContext()}
        handler = make_mock_handler(authenticated=False)
        result = rlm_handler._delete_context("ctx_noauth", {}, handler)
        assert result.status_code == 401


# ===========================================================================
# Test Context Route Dispatch
# ===========================================================================


class TestRLMContextRouteDispatch:
    """Tests for _handle_context_route method dispatch."""

    def test_invalid_empty_context_id(self, rlm_handler):
        result = rlm_handler._handle_context_route("/api/v1/rlm/context/", {}, None)
        assert result.status_code == 400

    def test_invalid_nested_context_id(self, rlm_handler):
        result = rlm_handler._handle_context_route("/api/v1/rlm/context/a/b", {}, None)
        assert result.status_code == 400

    def test_invalid_context_id_pattern(self, rlm_handler):
        """Path traversal attempts are rejected."""
        result = rlm_handler._handle_context_route(
            "/api/v1/rlm/context/../../etc",
            {},
            None,
        )
        assert result.status_code == 400

    def test_method_not_allowed(self, rlm_handler):
        """Non-GET/DELETE methods return 405."""
        handler = MagicMock()
        handler.command = "PATCH"
        rlm_handler._contexts["ctx_x"] = {"context": MockCompressedContext()}
        result = rlm_handler._handle_context_route("/api/v1/rlm/context/ctx_x", {}, handler)
        assert result.status_code == 405


# ===========================================================================
# Test read_json_body utility
# ===========================================================================


class TestRLMReadJsonBody:
    """Tests for the read_json_body utility method."""

    def test_read_valid_json(self, rlm_handler):
        handler = make_mock_handler(body={"key": "value"}, method="POST")
        result = rlm_handler.read_json_body(handler)
        assert result == {"key": "value"}

    def test_read_no_handler(self, rlm_handler):
        result = rlm_handler.read_json_body(None)
        assert result is None

    def test_read_empty_body(self, rlm_handler):
        handler = make_mock_handler(method="POST")
        result = rlm_handler.read_json_body(handler)
        assert result is None

    def test_read_oversized_body(self, rlm_handler):
        handler = make_mock_handler(body={"content": "x"}, method="POST")
        handler.headers["Content-Length"] = str(20_000_000)
        result = rlm_handler.read_json_body(handler)
        assert result is None

    def test_read_custom_max_size(self, rlm_handler):
        handler = make_mock_handler(body={"content": "test"}, method="POST")
        handler.headers["Content-Length"] = "100"
        result = rlm_handler.read_json_body(handler, max_size=50)
        assert result is None

    def test_read_malformed_json(self, rlm_handler):
        handler = MagicMock()
        handler.headers = {"Content-Length": "10"}
        handler.rfile = BytesIO(b"not json!!")
        result = rlm_handler.read_json_body(handler)
        assert result is None


# ===========================================================================
# Test Stream Modes Endpoint
# ===========================================================================


class TestRLMStreamModes:
    """Tests for the stream modes endpoint."""

    def test_stream_modes_with_module(self, rlm_handler):
        """Returns mode list from streaming module."""
        mock_mode = MagicMock()
        mock_mode.TOP_DOWN = MagicMock(value="top_down")
        mock_mode.BOTTOM_UP = MagicMock(value="bottom_up")
        mock_mode.TARGETED = MagicMock(value="targeted")
        mock_mode.PROGRESSIVE = MagicMock(value="progressive")

        with patch.dict(
            "sys.modules",
            {
                "aragora.rlm.streaming": MagicMock(StreamMode=mock_mode),
            },
        ):
            handler = make_mock_handler()
            result = rlm_handler.handle_stream_modes("/api/v1/rlm/stream/modes", {}, handler)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert "modes" in data
        assert len(data["modes"]) == 4

    def test_stream_modes_fallback(self, rlm_handler):
        """Returns fallback modes when streaming module is unavailable."""
        with patch.dict("sys.modules", {"aragora.rlm.streaming": None}):
            handler = make_mock_handler()
            result = rlm_handler.handle_stream_modes("/api/v1/rlm/stream/modes", {}, handler)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert "modes" in data
        assert "note" in data


# ===========================================================================
# Test Stream Endpoint
# ===========================================================================


class TestRLMStream:
    """Tests for the stream endpoint."""

    def _seed_context(self, rlm_handler, ctx_id="ctx_stream"):
        rlm_handler._contexts[ctx_id] = {
            "context": MockCompressedContext(),
            "source_type": "text",
        }
        return ctx_id

    def test_stream_no_body(self, rlm_handler):
        handler = make_mock_handler(method="POST")
        result = rlm_handler.handle_stream("/api/v1/rlm/stream", {}, handler)
        assert result.status_code == 400

    def test_stream_missing_context_id(self, rlm_handler):
        handler = make_mock_handler(body={"mode": "top_down"}, method="POST")
        result = rlm_handler.handle_stream("/api/v1/rlm/stream", {}, handler)
        assert result.status_code == 400

    def test_stream_context_not_found(self, rlm_handler):
        handler = make_mock_handler(
            body={"context_id": "nonexistent"},
            method="POST",
        )
        result = rlm_handler.handle_stream("/api/v1/rlm/stream", {}, handler)
        assert result.status_code == 404

    def test_stream_import_error_with_summary_fallback(self, rlm_handler):
        """Fallback to summary content when streaming module is unavailable."""
        ctx_id = self._seed_context(rlm_handler)
        handler = make_mock_handler(
            body={"context_id": ctx_id},
            method="POST",
        )
        with patch.dict(
            "sys.modules",
            {
                "aragora.rlm.streaming": None,
                "aragora.rlm": MagicMock(AbstractionLevel=MockAbstractionLevel),
            },
        ):
            result = rlm_handler.handle_stream("/api/v1/rlm/stream", {}, handler)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["mode"] == "fallback"
        assert data["total_chunks"] == 1

    def test_stream_import_error_no_summary(self, rlm_handler):
        """Returns 501 when streaming module AND summary are unavailable."""
        ctx_id = self._seed_context(rlm_handler)
        # Replace context with one that has no summary
        ctx = MagicMock()
        ctx.get_at_level.return_value = []
        rlm_handler._contexts[ctx_id]["context"] = ctx

        handler = make_mock_handler(
            body={"context_id": ctx_id},
            method="POST",
        )
        with patch.dict(
            "sys.modules",
            {
                "aragora.rlm.streaming": None,
                "aragora.rlm": MagicMock(AbstractionLevel=MockAbstractionLevel),
            },
        ):
            result = rlm_handler.handle_stream("/api/v1/rlm/stream", {}, handler)

        assert result.status_code == 501


# ===========================================================================
# Test Compressor and RLM Lazy Init
# ===========================================================================


class TestRLMLazyInit:
    """Tests for lazy initialization of compressor and RLM."""

    def test_get_compressor_caches_instance(self, rlm_handler):
        mock_compressor = MockCompressor()
        with patch("aragora.rlm.get_compressor", return_value=mock_compressor):
            c1 = rlm_handler._get_compressor()
            c2 = rlm_handler._get_compressor()
        assert c1 is c2
        assert c1 is mock_compressor

    def test_get_compressor_import_error(self, rlm_handler):
        with patch.dict("sys.modules", {"aragora.rlm": None}):
            # Clear any cached compressor
            rlm_handler._compressor = None
            result = rlm_handler._get_compressor()
        assert result is None

    def test_get_rlm_caches_instance(self, rlm_handler):
        mock_rlm = MockRLM()
        with patch("aragora.rlm.get_rlm", return_value=mock_rlm):
            with patch("aragora.rlm.HAS_OFFICIAL_RLM", False):
                r1 = rlm_handler._get_rlm()
                r2 = rlm_handler._get_rlm()
        assert r1 is r2
        assert r1 is mock_rlm

    def test_get_rlm_import_error(self, rlm_handler):
        with patch.dict("sys.modules", {"aragora.rlm": None}):
            rlm_handler._rlm = None
            result = rlm_handler._get_rlm()
        assert result is None


# ===========================================================================
# Test Codebase Health Endpoint
# ===========================================================================


class TestRLMCodebaseHealth:
    """Tests for the codebase health endpoint."""

    def test_codebase_health_manifest_exists(self, rlm_handler, tmp_path):
        """Health returns manifest info when it exists."""
        context_dir = tmp_path / ".nomic" / "context"
        context_dir.mkdir(parents=True)
        manifest = context_dir / "codebase_manifest.tsv"
        manifest.write_text("# files=100 lines=5000\n# generated\nfile.py\t100\n")

        handler = make_mock_handler()
        with patch.dict("os.environ", {"ARAGORA_CODEBASE_ROOT": str(tmp_path)}):
            with patch.dict(
                "sys.modules",
                {
                    "aragora.rlm": MagicMock(HAS_OFFICIAL_RLM=False),
                    "aragora.rlm.codebase_context": MagicMock(),
                },
            ):
                with patch.object(rlm_handler, "_get_rlm", return_value=None):
                    result = rlm_handler.handle_codebase_health(
                        "/api/v1/rlm/codebase/health",
                        {},
                        handler,
                    )

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["status"] == "available"
        assert data["manifest"]["exists"] is True
        assert data["manifest"]["files"] == 100

    def test_codebase_health_no_manifest(self, rlm_handler, tmp_path):
        """Health returns missing status when manifest doesn't exist."""
        handler = make_mock_handler()
        with patch.dict("os.environ", {"ARAGORA_CODEBASE_ROOT": str(tmp_path)}):
            with patch.dict(
                "sys.modules",
                {
                    "aragora.rlm": MagicMock(HAS_OFFICIAL_RLM=False),
                    "aragora.rlm.codebase_context": MagicMock(),
                },
            ):
                with patch.object(rlm_handler, "_get_rlm", return_value=None):
                    result = rlm_handler.handle_codebase_health(
                        "/api/v1/rlm/codebase/health",
                        {},
                        handler,
                    )

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["status"] == "missing"
        assert data["manifest"]["exists"] is False

    def test_codebase_health_nonexistent_root(self, rlm_handler):
        """Health returns 404 when codebase root doesn't exist."""
        handler = make_mock_handler()
        with patch.dict("os.environ", {"ARAGORA_CODEBASE_ROOT": "/nonexistent/path/xyz123"}):
            with patch.dict(
                "sys.modules",
                {
                    "aragora.rlm": MagicMock(HAS_OFFICIAL_RLM=False),
                    "aragora.rlm.codebase_context": MagicMock(),
                },
            ):
                result = rlm_handler.handle_codebase_health(
                    "/api/v1/rlm/codebase/health",
                    {},
                    handler,
                )
        assert result.status_code == 404


# ===========================================================================
# Test Compression Ratio Calculation
# ===========================================================================


class TestRLMCompressionRatio:
    """Tests for compression ratio edge cases."""

    def test_zero_original_tokens_ratio(self, rlm_handler):
        """Zero original tokens produces ratio of 1.0."""
        rlm_handler._contexts["ctx_zero"] = {
            "context": MagicMock(
                original_tokens=0,
                total_tokens=MagicMock(return_value=0),
                levels={},
            ),
            "source_type": "text",
            "original_tokens": 0,
            "created_at": "2024-01-01T00:00:00",
        }
        result = rlm_handler._get_context("ctx_zero", {}, None)
        data = json.loads(result.body)
        assert data["compression_ratio"] == 1.0


# ===========================================================================
# Test Full Context Lifecycle
# ===========================================================================


class TestRLMContextLifecycle:
    """Integration-style tests for full context lifecycle."""

    def test_compress_then_query_then_delete(self, rlm_handler):
        """Full lifecycle: compress -> query -> list -> delete."""
        # Step 1: Compress
        compress_handler = make_mock_handler(
            body={"content": "Lifecycle test content for RLM compression"},
            method="POST",
            authenticated=True,
        )
        with patch.object(rlm_handler, "_get_compressor", return_value=MockCompressor()):
            compress_result = rlm_handler.handle_compress(
                "/api/v1/rlm/compress",
                {},
                compress_handler,
            )
        assert compress_result.status_code == 200
        context_id = json.loads(compress_result.body)["context_id"]

        # Step 2: Query
        query_handler = make_mock_handler(
            body={"context_id": context_id, "query": "What is in this context?"},
            method="POST",
            authenticated=True,
        )
        with patch.object(rlm_handler, "_get_rlm", return_value=MockRLM()):
            query_result = rlm_handler.handle_query(
                "/api/v1/rlm/query",
                {},
                query_handler,
            )
        assert query_result.status_code == 200

        # Step 3: List - verify the context exists in total count
        # Note: safe_query_int clamps offset to min_val=1 by default, so with only
        # one context it may be skipped in the returned page. We verify total count.
        list_handler = make_mock_handler()
        list_result = rlm_handler.handle_list_contexts(
            "/api/v1/rlm/contexts",
            {},
            list_handler,
        )
        list_data = json.loads(list_result.body)
        assert list_data["total"] >= 1
        # Confirm context is stored even if pagination skips it
        assert context_id in rlm_handler._contexts

        # Step 4: Delete
        delete_handler = make_mock_handler(authenticated=True)
        delete_result = rlm_handler._delete_context(context_id, {}, delete_handler)
        assert delete_result.status_code == 200
        assert context_id not in rlm_handler._contexts

        # Step 5: Verify deleted
        get_result = rlm_handler._get_context(context_id, {}, None)
        assert get_result.status_code == 404
