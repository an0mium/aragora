"""Tests for the RLMContextHandler REST endpoints.

Covers all routes and behavior of the RLMContextHandler class:
- can_handle() route matching (all routes + rejection)
- GET  /api/v1/rlm/stats             - Get RLM compression statistics
- GET  /api/v1/rlm/strategies        - List decomposition strategies
- GET  /api/v1/rlm/contexts          - List stored compressed contexts
- GET  /api/v1/rlm/context/{id}      - Get specific context
- GET  /api/v1/rlm/stream/modes      - Get streaming modes
- GET  /api/v1/rlm/codebase/health   - Get codebase RLM health
- POST /api/v1/rlm/compress          - Compress content
- POST /api/v1/rlm/query             - Query a compressed context
- POST /api/v1/rlm/stream            - Stream context chunks
- DELETE /api/v1/rlm/context/{id}    - Delete a context
- Error handling (400, 404, 405, 413, 500, 501, 503)
- Edge cases (missing params, invalid JSON, empty body)
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from aragora.server.handlers.rlm import RLMContextHandler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if result is None:
        return 0
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


class MockHTTPHandler:
    """Lightweight mock for the HTTP handler passed to handler methods."""

    def __init__(
        self,
        method: str = "GET",
        body: dict[str, Any] | None = None,
        client_address: tuple[str, int] = ("127.0.0.1", 12345),
        auth_token: str | None = "test-valid-token",
    ):
        self.command = method
        self.headers: dict[str, str] = {"User-Agent": "test-agent"}
        self.rfile = MagicMock()
        self.client_address = client_address
        self.path = ""

        if auth_token:
            self.headers["Authorization"] = f"Bearer {auth_token}"

        if body is not None:
            raw = json.dumps(body).encode()
            self.rfile.read.return_value = raw
            self.headers["Content-Length"] = str(len(raw))
            self.headers["Content-Type"] = "application/json"
        else:
            self.rfile.read.return_value = b""
            self.headers["Content-Length"] = "0"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create an RLMContextHandler with empty server context."""
    return RLMContextHandler({})


@pytest.fixture(autouse=True)
def _patch_rate_limit(monkeypatch):
    """Bypass rate limiting for tests."""
    monkeypatch.setenv("ARAGORA_USE_DISTRIBUTED_RATE_LIMIT", "false")


@pytest.fixture(autouse=True)
def _patch_require_auth(monkeypatch):
    """Bypass the @require_auth token check for tests.

    The @require_auth decorator in rlm.py checks ARAGORA_API_TOKEN via auth_config.
    We patch auth_config so that it always validates the test token.
    """
    try:
        from aragora.server import auth as server_auth

        mock_auth_config = MagicMock()
        mock_auth_config.api_token = "test-valid-token"
        mock_auth_config.validate_token.return_value = True
        mock_auth_config.enabled = True
        monkeypatch.setattr(server_auth, "auth_config", mock_auth_config)
    except (ImportError, AttributeError):
        pass


def _make_mock_context(
    original_tokens: int = 1000,
    total_tokens: int = 200,
    levels: dict | None = None,
):
    """Create a mock compression context object."""
    ctx = MagicMock()
    ctx.original_tokens = original_tokens
    ctx.total_tokens.return_value = total_tokens
    if levels is None:
        mock_level = MagicMock()
        mock_level.name = "SUMMARY"
        mock_node = MagicMock()
        mock_node.token_count = 50
        mock_node.id = "node_1"
        mock_node.content = "Summary content here"
        ctx.levels = {mock_level: [mock_node]}
    else:
        ctx.levels = levels
    return ctx


def _add_context(
    handler: RLMContextHandler,
    context_id: str = "ctx_abc123_1234567890",
    source_type: str = "text",
    original_tokens: int = 1000,
    total_tokens: int = 200,
) -> MagicMock:
    """Add a mock context to the handler's internal storage."""
    ctx = _make_mock_context(original_tokens, total_tokens)
    handler._contexts[context_id] = {
        "context": ctx,
        "created_at": datetime.now().isoformat(),
        "source_type": source_type,
        "original_tokens": original_tokens,
    }
    return ctx


# ===========================================================================
# can_handle()
# ===========================================================================


class TestCanHandle:
    """Tests for RLMContextHandler.can_handle() route matching."""

    def test_stats_route(self, handler):
        assert handler.can_handle("/api/v1/rlm/stats") is True

    def test_strategies_route(self, handler):
        assert handler.can_handle("/api/v1/rlm/strategies") is True

    def test_compress_route(self, handler):
        assert handler.can_handle("/api/v1/rlm/compress") is True

    def test_query_route(self, handler):
        assert handler.can_handle("/api/v1/rlm/query") is True

    def test_contexts_route(self, handler):
        assert handler.can_handle("/api/v1/rlm/contexts") is True

    def test_stream_route(self, handler):
        assert handler.can_handle("/api/v1/rlm/stream") is True

    def test_stream_modes_route(self, handler):
        assert handler.can_handle("/api/v1/rlm/stream/modes") is True

    def test_codebase_health_route(self, handler):
        assert handler.can_handle("/api/v1/rlm/codebase/health") is True

    def test_context_by_id_route(self, handler):
        assert handler.can_handle("/api/v1/rlm/context/ctx_abc123") is True

    def test_context_route_prefix_only(self, handler):
        # The prefix without an ID should still match (handler validates later)
        assert handler.can_handle("/api/v1/rlm/context/") is True

    def test_unrelated_route_rejected(self, handler):
        assert handler.can_handle("/api/v1/debates") is False

    def test_partial_route_rejected(self, handler):
        assert handler.can_handle("/api/v1/rlm") is False

    def test_wrong_version_rejected(self, handler):
        assert handler.can_handle("/api/v2/rlm/stats") is False

    def test_empty_path_rejected(self, handler):
        assert handler.can_handle("") is False

    def test_root_path_rejected(self, handler):
        assert handler.can_handle("/") is False


# ===========================================================================
# GET /api/v1/rlm/stats
# ===========================================================================


class TestHandleStats:
    """Tests for the stats endpoint."""

    def test_stats_success_with_rlm_available(self, handler):
        """Stats returns cache info and system status when RLM is available."""
        mock_cache_stats = {"hits": 10, "misses": 5, "size": 3}
        with patch(
            "aragora.server.handlers.rlm.RLMContextHandler._get_compressor",
            return_value=MagicMock(),
        ), patch(
            "aragora.server.handlers.rlm.RLMContextHandler._get_rlm",
            return_value=MagicMock(),
        ), patch.dict(
            "sys.modules",
            {
                "aragora.rlm.compressor": MagicMock(
                    get_compression_cache_stats=MagicMock(return_value=mock_cache_stats)
                ),
                "aragora.rlm": MagicMock(HAS_OFFICIAL_RLM=False),
            },
        ):
            result = handler.handle_stats(
                "/api/v1/rlm/stats", {}, MockHTTPHandler()
            )

        body = _body(result)
        assert _status(result) == 200
        assert "cache" in body
        assert "contexts" in body
        assert "system" in body
        assert "timestamp" in body

    def test_stats_import_error_fallback(self, handler):
        """Stats returns graceful fallback when RLM module unavailable."""
        # Force ImportError by making the import inside handle_stats fail
        with patch.dict("sys.modules", {
            "aragora.rlm.compressor": None,
            "aragora.rlm": None,
        }):
            result = handler.handle_stats(
                "/api/v1/rlm/stats", {}, MockHTTPHandler()
            )
        body = _body(result)
        assert _status(result) == 200
        assert body["system"]["has_official_rlm"] is False
        assert body["system"]["compressor_available"] is False
        assert body["system"]["rlm_available"] is False
        assert body["cache"]["error"] == "RLM module not available"

    def test_stats_with_stored_contexts(self, handler):
        """Stats reports the correct number of stored contexts."""
        _add_context(handler, "ctx_1")
        _add_context(handler, "ctx_2")
        _add_context(handler, "ctx_3")

        with patch(
            "aragora.rlm.compressor.get_compression_cache_stats",
            return_value={"hits": 0},
        ), patch("aragora.rlm.HAS_OFFICIAL_RLM", False):
            result = handler.handle_stats(
                "/api/v1/rlm/stats", {}, MockHTTPHandler()
            )

        body = _body(result)
        assert _status(result) == 200
        assert body["contexts"]["stored"] == 3

    def test_stats_via_handle_dispatch(self, handler):
        """Stats route is dispatched correctly via handle()."""
        with patch.object(handler, "handle_stats", return_value=MagicMock(status_code=200, body=b'{}')) as mock:
            handler.handle("/api/v1/rlm/stats", {}, MockHTTPHandler())
            mock.assert_called_once()


# ===========================================================================
# GET /api/v1/rlm/strategies
# ===========================================================================


class TestHandleStrategies:
    """Tests for the strategies endpoint."""

    def test_strategies_returns_all(self, handler):
        """Strategies returns all 6 known strategies."""
        result = handler.handle_strategies(
            "/api/v1/rlm/strategies", {}, MockHTTPHandler()
        )
        body = _body(result)
        assert _status(result) == 200
        strategies = body["strategies"]
        expected = {"peek", "grep", "partition_map", "summarize", "hierarchical", "auto"}
        assert set(strategies.keys()) == expected

    def test_strategies_default_is_auto(self, handler):
        """Strategies lists 'auto' as the default."""
        result = handler.handle_strategies(
            "/api/v1/rlm/strategies", {}, MockHTTPHandler()
        )
        body = _body(result)
        assert body["default"] == "auto"

    def test_strategies_includes_documentation(self, handler):
        """Strategies includes documentation link."""
        result = handler.handle_strategies(
            "/api/v1/rlm/strategies", {}, MockHTTPHandler()
        )
        body = _body(result)
        assert "documentation" in body
        assert "github.com" in body["documentation"]

    def test_strategy_fields(self, handler):
        """Each strategy has name, description, use_case, and token_reduction."""
        result = handler.handle_strategies(
            "/api/v1/rlm/strategies", {}, MockHTTPHandler()
        )
        body = _body(result)
        for key, strategy in body["strategies"].items():
            assert "name" in strategy, f"Strategy {key} missing 'name'"
            assert "description" in strategy, f"Strategy {key} missing 'description'"
            assert "use_case" in strategy, f"Strategy {key} missing 'use_case'"
            assert "token_reduction" in strategy, f"Strategy {key} missing 'token_reduction'"

    def test_strategies_via_handle_dispatch(self, handler):
        """Strategies route is dispatched correctly via handle()."""
        with patch.object(handler, "handle_strategies", return_value=MagicMock(status_code=200, body=b'{}')) as mock:
            handler.handle("/api/v1/rlm/strategies", {}, MockHTTPHandler())
            mock.assert_called_once()


# ===========================================================================
# GET /api/v1/rlm/contexts
# ===========================================================================


class TestHandleListContexts:
    """Tests for listing contexts."""

    def test_list_empty(self, handler):
        """Returns empty list when no contexts exist."""
        result = handler.handle_list_contexts(
            "/api/v1/rlm/contexts", {}, MockHTTPHandler()
        )
        body = _body(result)
        assert _status(result) == 200
        assert body["contexts"] == []
        assert body["total"] == 0

    def test_list_with_contexts(self, handler):
        """Returns all contexts with metadata."""
        _add_context(handler, "ctx_1", source_type="text")
        _add_context(handler, "ctx_2", source_type="code")

        result = handler.handle_list_contexts(
            "/api/v1/rlm/contexts", {}, MockHTTPHandler()
        )
        body = _body(result)
        assert _status(result) == 200
        assert body["total"] == 2
        assert len(body["contexts"]) == 2

    def test_list_context_fields(self, handler):
        """Each context includes expected fields."""
        _add_context(handler, "ctx_1", source_type="code", original_tokens=500)

        result = handler.handle_list_contexts(
            "/api/v1/rlm/contexts", {}, MockHTTPHandler()
        )
        body = _body(result)
        ctx = body["contexts"][0]
        assert ctx["id"] == "ctx_1"
        assert ctx["source_type"] == "code"
        assert ctx["original_tokens"] == 500
        assert "created_at" in ctx

    def test_list_pagination_limit(self, handler):
        """Limit parameter restricts results."""
        for i in range(10):
            _add_context(handler, f"ctx_{i}")

        result = handler.handle_list_contexts(
            "/api/v1/rlm/contexts", {"limit": "3"}, MockHTTPHandler()
        )
        body = _body(result)
        assert len(body["contexts"]) == 3
        assert body["total"] == 10
        assert body["limit"] == 3

    def test_list_pagination_offset(self, handler):
        """Offset parameter skips results."""
        for i in range(5):
            _add_context(handler, f"ctx_{i}")

        result = handler.handle_list_contexts(
            "/api/v1/rlm/contexts", {"offset": "3"}, MockHTTPHandler()
        )
        body = _body(result)
        assert len(body["contexts"]) == 2
        assert body["offset"] == 3

    def test_list_pagination_combined(self, handler):
        """Limit and offset work together."""
        for i in range(10):
            _add_context(handler, f"ctx_{i}")

        result = handler.handle_list_contexts(
            "/api/v1/rlm/contexts", {"limit": "2", "offset": "5"}, MockHTTPHandler()
        )
        body = _body(result)
        assert len(body["contexts"]) == 2
        assert body["total"] == 10

    def test_list_via_handle_dispatch(self, handler):
        """Contexts route is dispatched correctly via handle()."""
        with patch.object(handler, "handle_list_contexts", return_value=MagicMock(status_code=200, body=b'{}')) as mock:
            handler.handle("/api/v1/rlm/contexts", {}, MockHTTPHandler())
            mock.assert_called_once()


# ===========================================================================
# GET /api/v1/rlm/context/{id}
# ===========================================================================


class TestGetContext:
    """Tests for getting a specific context."""

    def test_get_context_success(self, handler):
        """Returns context details for a valid ID."""
        _add_context(handler, "ctx_abc123", original_tokens=1000, total_tokens=200)

        result = handler._get_context("ctx_abc123", {}, MockHTTPHandler())
        body = _body(result)
        assert _status(result) == 200
        assert body["id"] == "ctx_abc123"
        assert body["original_tokens"] == 1000
        assert body["compressed_tokens"] == 200
        assert "levels" in body

    def test_get_context_not_found(self, handler):
        """Returns 404 for non-existent context."""
        result = handler._get_context("ctx_nonexistent", {}, MockHTTPHandler())
        assert _status(result) == 404

    def test_get_context_with_include_content(self, handler):
        """Returns summary preview when include_content=true."""
        ctx = _add_context(handler, "ctx_abc123")

        # Mock get_at_level for summary content
        mock_node = MagicMock()
        mock_node.id = "n1"
        mock_node.content = "Summary text preview"
        ctx.get_at_level.return_value = [mock_node]

        with patch("aragora.rlm.AbstractionLevel") as mock_level:
            mock_level.SUMMARY = "SUMMARY"
            result = handler._get_context(
                "ctx_abc123", {"include_content": "true"}, MockHTTPHandler()
            )

        body = _body(result)
        assert _status(result) == 200
        assert "summary_preview" in body

    def test_get_context_include_content_false(self, handler):
        """Does not include summary when include_content=false."""
        _add_context(handler, "ctx_abc123")

        result = handler._get_context(
            "ctx_abc123", {"include_content": "false"}, MockHTTPHandler()
        )
        body = _body(result)
        assert "summary_preview" not in body

    def test_get_context_compression_ratio(self, handler):
        """Compression ratio is computed correctly."""
        _add_context(handler, "ctx_abc123", original_tokens=1000, total_tokens=250)

        result = handler._get_context("ctx_abc123", {}, MockHTTPHandler())
        body = _body(result)
        assert body["compression_ratio"] == 0.25

    def test_get_context_zero_original_tokens(self, handler):
        """Handles zero original tokens without division error."""
        _add_context(handler, "ctx_abc123", original_tokens=0, total_tokens=0)

        result = handler._get_context("ctx_abc123", {}, MockHTTPHandler())
        body = _body(result)
        assert body["compression_ratio"] == 1.0

    def test_get_context_via_handle_route(self, handler):
        """Context route dispatched via handle() for GET."""
        _add_context(handler, "ctx_abc123")

        mock_http = MockHTTPHandler(method="GET")
        result = handler.handle(
            "/api/v1/rlm/context/ctx_abc123", {}, mock_http
        )
        body = _body(result)
        assert _status(result) == 200
        assert body["id"] == "ctx_abc123"


# ===========================================================================
# GET /api/v1/rlm/context/{id} - validation
# ===========================================================================


class TestContextRouteValidation:
    """Tests for context route validation."""

    def test_empty_context_id(self, handler):
        """Empty context ID after prefix returns 400."""
        result = handler._handle_context_route(
            "/api/v1/rlm/context/", {}, MockHTTPHandler()
        )
        assert _status(result) == 400

    def test_context_id_with_slash(self, handler):
        """Context ID containing slash returns 400."""
        result = handler._handle_context_route(
            "/api/v1/rlm/context/foo/bar", {}, MockHTTPHandler()
        )
        assert _status(result) == 400

    def test_context_id_valid_format(self, handler):
        """Valid context ID format passes validation."""
        _add_context(handler, "ctx_abc123")
        mock_http = MockHTTPHandler(method="GET")
        result = handler._handle_context_route(
            "/api/v1/rlm/context/ctx_abc123", {}, mock_http
        )
        assert _status(result) == 200

    def test_context_unsupported_method(self, handler):
        """Non GET/DELETE method returns 405."""
        _add_context(handler, "ctx_abc123")
        mock_http = MockHTTPHandler(method="PUT")
        result = handler._handle_context_route(
            "/api/v1/rlm/context/ctx_abc123", {}, mock_http
        )
        assert _status(result) == 405


# ===========================================================================
# DELETE /api/v1/rlm/context/{id}
# ===========================================================================


class TestDeleteContext:
    """Tests for deleting a context."""

    def test_delete_success(self, handler):
        """Successfully deletes an existing context."""
        _add_context(handler, "ctx_abc123")
        assert "ctx_abc123" in handler._contexts

        result = handler._delete_context("ctx_abc123", {}, MockHTTPHandler())
        body = _body(result)
        assert _status(result) == 200
        assert body["success"] is True
        assert body["context_id"] == "ctx_abc123"
        assert "ctx_abc123" not in handler._contexts

    def test_delete_not_found(self, handler):
        """Returns 404 when context does not exist."""
        result = handler._delete_context("ctx_nonexistent", {}, MockHTTPHandler())
        assert _status(result) == 404

    def test_delete_via_handle_delete(self, handler):
        """DELETE dispatch works through handle_delete."""
        _add_context(handler, "ctx_abc123")

        result = handler.handle_delete(
            "/api/v1/rlm/context/ctx_abc123", {}, MockHTTPHandler()
        )
        body = _body(result)
        assert _status(result) == 200
        assert body["success"] is True

    def test_delete_empty_id_returns_none(self, handler):
        """DELETE with empty ID returns None (no handler match)."""
        result = handler.handle_delete(
            "/api/v1/rlm/context/", {}, MockHTTPHandler()
        )
        assert result is None

    def test_delete_id_with_slash_returns_none(self, handler):
        """DELETE with ID containing slash returns None."""
        result = handler.handle_delete(
            "/api/v1/rlm/context/foo/bar", {}, MockHTTPHandler()
        )
        assert result is None

    def test_delete_wrong_prefix_returns_none(self, handler):
        """DELETE on non-context path returns None."""
        result = handler.handle_delete(
            "/api/v1/rlm/stats", {}, MockHTTPHandler()
        )
        assert result is None


# ===========================================================================
# POST /api/v1/rlm/compress
# ===========================================================================


class TestHandleCompress:
    """Tests for the compress endpoint."""

    def test_compress_success(self, handler):
        """Compress returns context_id and compression result."""
        mock_ctx = _make_mock_context(original_tokens=1000, total_tokens=200)

        mock_compressor = MagicMock()
        with patch.object(handler, "_get_compressor", return_value=mock_compressor), \
             patch("aragora.server.handlers.rlm.run_async", return_value=mock_ctx):
            result = handler.handle_compress(
                "/api/v1/rlm/compress",
                {},
                MockHTTPHandler(method="POST", body={"content": "Hello world test content"}),
            )

        body = _body(result)
        assert _status(result) == 200
        assert "context_id" in body
        assert body["context_id"].startswith("ctx_")
        assert "compression_result" in body
        cr = body["compression_result"]
        assert cr["original_tokens"] == 1000
        assert cr["compressed_tokens"] == 200
        assert cr["source_type"] == "text"

    def test_compress_no_body(self, handler):
        """Compress returns 400 when no body provided."""
        result = handler.handle_compress(
            "/api/v1/rlm/compress",
            {},
            MockHTTPHandler(method="POST"),
        )
        assert _status(result) == 400

    def test_compress_missing_content(self, handler):
        """Compress returns 400 when 'content' field missing."""
        result = handler.handle_compress(
            "/api/v1/rlm/compress",
            {},
            MockHTTPHandler(method="POST", body={"source_type": "text"}),
        )
        assert _status(result) == 400

    def test_compress_content_not_string(self, handler):
        """Compress returns 400 when content is not a string."""
        result = handler.handle_compress(
            "/api/v1/rlm/compress",
            {},
            MockHTTPHandler(method="POST", body={"content": 12345}),
        )
        assert _status(result) == 400

    def test_compress_empty_content(self, handler):
        """Compress returns 400 when content is empty string."""
        result = handler.handle_compress(
            "/api/v1/rlm/compress",
            {},
            MockHTTPHandler(method="POST", body={"content": ""}),
        )
        assert _status(result) == 400

    def test_compress_content_too_large(self, handler):
        """Compress returns 413 when content exceeds 10MB."""
        big_content = "x" * (10_000_001)
        # Mock read_json_body to bypass the body-size limit in read_json_body itself
        # so the handler's own content-length check can run
        with patch.object(
            handler, "read_json_body", return_value={"content": big_content}
        ):
            result = handler.handle_compress(
                "/api/v1/rlm/compress",
                {},
                MockHTTPHandler(method="POST"),
            )
        assert _status(result) == 413

    def test_compress_invalid_source_type(self, handler):
        """Compress returns 400 for invalid source_type."""
        result = handler.handle_compress(
            "/api/v1/rlm/compress",
            {},
            MockHTTPHandler(method="POST", body={"content": "test", "source_type": "invalid"}),
        )
        assert _status(result) == 400

    def test_compress_valid_source_types(self, handler):
        """Compress accepts all valid source types."""
        mock_ctx = _make_mock_context()
        mock_compressor = MagicMock()

        for src_type in ("text", "code", "debate"):
            with patch.object(handler, "_get_compressor", return_value=mock_compressor), \
                 patch("aragora.server.handlers.rlm.run_async", return_value=mock_ctx):
                result = handler.handle_compress(
                    "/api/v1/rlm/compress",
                    {},
                    MockHTTPHandler(
                        method="POST",
                        body={"content": "test content", "source_type": src_type},
                    ),
                )
            assert _status(result) == 200, f"Failed for source_type={src_type}"

    def test_compress_invalid_levels_zero(self, handler):
        """Compress returns 400 when levels < 1."""
        result = handler.handle_compress(
            "/api/v1/rlm/compress",
            {},
            MockHTTPHandler(
                method="POST",
                body={"content": "test", "levels": 0},
            ),
        )
        assert _status(result) == 400

    def test_compress_invalid_levels_six(self, handler):
        """Compress returns 400 when levels > 5."""
        result = handler.handle_compress(
            "/api/v1/rlm/compress",
            {},
            MockHTTPHandler(
                method="POST",
                body={"content": "test", "levels": 6},
            ),
        )
        assert _status(result) == 400

    def test_compress_invalid_levels_string(self, handler):
        """Compress returns 400 when levels is not an int."""
        result = handler.handle_compress(
            "/api/v1/rlm/compress",
            {},
            MockHTTPHandler(
                method="POST",
                body={"content": "test", "levels": "three"},
            ),
        )
        assert _status(result) == 400

    def test_compress_valid_levels(self, handler):
        """Compress accepts valid level values 1-5."""
        mock_ctx = _make_mock_context()
        mock_compressor = MagicMock()

        for lvl in (1, 2, 3, 4, 5):
            with patch.object(handler, "_get_compressor", return_value=mock_compressor), \
                 patch("aragora.server.handlers.rlm.run_async", return_value=mock_ctx):
                result = handler.handle_compress(
                    "/api/v1/rlm/compress",
                    {},
                    MockHTTPHandler(
                        method="POST",
                        body={"content": "test", "levels": lvl},
                    ),
                )
            assert _status(result) == 200, f"Failed for levels={lvl}"

    def test_compress_compressor_unavailable(self, handler):
        """Compress returns 503 when compressor is not available."""
        with patch.object(handler, "_get_compressor", return_value=None):
            result = handler.handle_compress(
                "/api/v1/rlm/compress",
                {},
                MockHTTPHandler(
                    method="POST",
                    body={"content": "test content"},
                ),
            )
        assert _status(result) == 503

    def test_compress_runtime_error(self, handler):
        """Compress returns 500 on RuntimeError."""
        mock_compressor = MagicMock()
        with patch.object(handler, "_get_compressor", return_value=mock_compressor), \
             patch("aragora.server.handlers.rlm.run_async", side_effect=RuntimeError("boom")):
            result = handler.handle_compress(
                "/api/v1/rlm/compress",
                {},
                MockHTTPHandler(
                    method="POST",
                    body={"content": "test"},
                ),
            )
        assert _status(result) == 500

    def test_compress_value_error(self, handler):
        """Compress returns 500 on ValueError."""
        mock_compressor = MagicMock()
        with patch.object(handler, "_get_compressor", return_value=mock_compressor), \
             patch("aragora.server.handlers.rlm.run_async", side_effect=ValueError("bad data")):
            result = handler.handle_compress(
                "/api/v1/rlm/compress",
                {},
                MockHTTPHandler(
                    method="POST",
                    body={"content": "test"},
                ),
            )
        assert _status(result) == 500

    def test_compress_stores_context(self, handler):
        """Compress stores the context in handler._contexts."""
        mock_ctx = _make_mock_context()
        mock_compressor = MagicMock()

        with patch.object(handler, "_get_compressor", return_value=mock_compressor), \
             patch("aragora.server.handlers.rlm.run_async", return_value=mock_ctx):
            result = handler.handle_compress(
                "/api/v1/rlm/compress",
                {},
                MockHTTPHandler(
                    method="POST",
                    body={"content": "stored content"},
                ),
            )

        body = _body(result)
        ctx_id = body["context_id"]
        assert ctx_id in handler._contexts
        assert handler._contexts[ctx_id]["source_type"] == "text"

    def test_compress_via_handle_post(self, handler):
        """Compress dispatched correctly via handle_post."""
        with patch.object(handler, "handle_compress", return_value=MagicMock(status_code=200, body=b'{}')) as mock:
            handler.handle_post(
                "/api/v1/rlm/compress", {}, MockHTTPHandler(method="POST")
            )
            mock.assert_called_once()


# ===========================================================================
# POST /api/v1/rlm/query
# ===========================================================================


class TestHandleQuery:
    """Tests for the query endpoint."""

    def test_query_success(self, handler):
        """Query returns answer and metadata."""
        _add_context(handler, "ctx_abc123")

        mock_result = MagicMock()
        mock_result.answer = "The answer is 42"
        mock_result.confidence = 0.95
        mock_result.iteration = 1
        mock_result.tokens_processed = 500
        mock_result.sub_calls_made = 3

        with patch.object(handler, "_get_rlm", return_value=MagicMock()), \
             patch("aragora.server.handlers.rlm.run_async", return_value=mock_result):
            result = handler.handle_query(
                "/api/v1/rlm/query",
                {},
                MockHTTPHandler(
                    method="POST",
                    body={"context_id": "ctx_abc123", "query": "What is the answer?"},
                ),
            )

        body = _body(result)
        assert _status(result) == 200
        assert body["answer"] == "The answer is 42"
        assert body["metadata"]["context_id"] == "ctx_abc123"
        assert body["metadata"]["confidence"] == 0.95

    def test_query_no_body(self, handler):
        """Query returns 400 when no body provided."""
        result = handler.handle_query(
            "/api/v1/rlm/query",
            {},
            MockHTTPHandler(method="POST"),
        )
        assert _status(result) == 400

    def test_query_missing_context_id(self, handler):
        """Query returns 400 when context_id missing."""
        result = handler.handle_query(
            "/api/v1/rlm/query",
            {},
            MockHTTPHandler(
                method="POST",
                body={"query": "What?"},
            ),
        )
        assert _status(result) == 400

    def test_query_missing_query_field(self, handler):
        """Query returns 400 when query missing."""
        _add_context(handler, "ctx_abc123")
        result = handler.handle_query(
            "/api/v1/rlm/query",
            {},
            MockHTTPHandler(
                method="POST",
                body={"context_id": "ctx_abc123"},
            ),
        )
        assert _status(result) == 400

    def test_query_query_not_string(self, handler):
        """Query returns 400 when query is not a string."""
        _add_context(handler, "ctx_abc123")
        result = handler.handle_query(
            "/api/v1/rlm/query",
            {},
            MockHTTPHandler(
                method="POST",
                body={"context_id": "ctx_abc123", "query": 123},
            ),
        )
        assert _status(result) == 400

    def test_query_empty_query(self, handler):
        """Query returns 400 when query is empty."""
        _add_context(handler, "ctx_abc123")
        result = handler.handle_query(
            "/api/v1/rlm/query",
            {},
            MockHTTPHandler(
                method="POST",
                body={"context_id": "ctx_abc123", "query": ""},
            ),
        )
        assert _status(result) == 400

    def test_query_too_long(self, handler):
        """Query returns 400 when query exceeds 10000 characters."""
        _add_context(handler, "ctx_abc123")
        result = handler.handle_query(
            "/api/v1/rlm/query",
            {},
            MockHTTPHandler(
                method="POST",
                body={"context_id": "ctx_abc123", "query": "x" * 10001},
            ),
        )
        assert _status(result) == 400

    def test_query_context_not_found(self, handler):
        """Query returns 404 when context does not exist."""
        result = handler.handle_query(
            "/api/v1/rlm/query",
            {},
            MockHTTPHandler(
                method="POST",
                body={"context_id": "ctx_nonexistent", "query": "What?"},
            ),
        )
        assert _status(result) == 404

    def test_query_invalid_strategy(self, handler):
        """Query returns 400 for invalid strategy."""
        _add_context(handler, "ctx_abc123")
        result = handler.handle_query(
            "/api/v1/rlm/query",
            {},
            MockHTTPHandler(
                method="POST",
                body={
                    "context_id": "ctx_abc123",
                    "query": "What?",
                    "strategy": "invalid_strategy",
                },
            ),
        )
        assert _status(result) == 400

    def test_query_valid_strategies(self, handler):
        """Query accepts all valid strategy names."""
        _add_context(handler, "ctx_abc123")
        mock_result = MagicMock()
        mock_result.answer = "answer"

        valid = ["peek", "grep", "partition_map", "summarize", "hierarchical", "auto"]
        for strategy in valid:
            with patch.object(handler, "_get_rlm", return_value=MagicMock()), \
                 patch("aragora.server.handlers.rlm.run_async", return_value=mock_result):
                result = handler.handle_query(
                    "/api/v1/rlm/query",
                    {},
                    MockHTTPHandler(
                        method="POST",
                        body={
                            "context_id": "ctx_abc123",
                            "query": "What?",
                            "strategy": strategy,
                        },
                    ),
                )
            assert _status(result) == 200, f"Failed for strategy={strategy}"

    def test_query_with_refinement(self, handler):
        """Query uses refinement when refine=True."""
        _add_context(handler, "ctx_abc123")

        mock_rlm = MagicMock()
        mock_result = MagicMock()
        mock_result.answer = "Refined answer"
        mock_result.confidence = 0.99
        mock_result.iteration = 3

        with patch.object(handler, "_get_rlm", return_value=mock_rlm), \
             patch("aragora.server.handlers.rlm.run_async", return_value=mock_result):
            result = handler.handle_query(
                "/api/v1/rlm/query",
                {},
                MockHTTPHandler(
                    method="POST",
                    body={
                        "context_id": "ctx_abc123",
                        "query": "What?",
                        "refine": True,
                        "max_iterations": 5,
                    },
                ),
            )

        body = _body(result)
        assert _status(result) == 200
        assert body["metadata"]["refined"] is True

    def test_query_max_iterations_clamped(self, handler):
        """Invalid max_iterations defaults to 3."""
        _add_context(handler, "ctx_abc123")
        mock_result = MagicMock()
        mock_result.answer = "answer"

        with patch.object(handler, "_get_rlm", return_value=MagicMock()), \
             patch("aragora.server.handlers.rlm.run_async", return_value=mock_result):
            result = handler.handle_query(
                "/api/v1/rlm/query",
                {},
                MockHTTPHandler(
                    method="POST",
                    body={
                        "context_id": "ctx_abc123",
                        "query": "What?",
                        "max_iterations": 999,
                    },
                ),
            )
        # Should not fail - invalid value is silently clamped
        assert _status(result) == 200

    def test_query_fallback_when_rlm_unavailable(self, handler):
        """Query uses fallback when RLM instance is None."""
        ctx = _add_context(handler, "ctx_abc123")

        mock_node = MagicMock()
        mock_node.content = "Fallback summary content"
        ctx.get_at_level.return_value = [mock_node]

        with patch.object(handler, "_get_rlm", return_value=None), \
             patch("aragora.rlm.AbstractionLevel") as mock_level:
            mock_level.SUMMARY = "SUMMARY"
            result = handler.handle_query(
                "/api/v1/rlm/query",
                {},
                MockHTTPHandler(
                    method="POST",
                    body={
                        "context_id": "ctx_abc123",
                        "query": "What?",
                    },
                ),
            )

        body = _body(result)
        assert _status(result) == 200
        assert body["metadata"]["fallback"] is True

    def test_query_fallback_no_summary(self, handler):
        """Fallback query returns generic message when no summary nodes."""
        ctx = _add_context(handler, "ctx_abc123")

        with patch.object(handler, "_get_rlm", return_value=None), \
             patch("aragora.rlm.AbstractionLevel") as mock_level:
            mock_level.SUMMARY = "SUMMARY"
            ctx.get_at_level.return_value = []
            result = handler.handle_query(
                "/api/v1/rlm/query",
                {},
                MockHTTPHandler(
                    method="POST",
                    body={
                        "context_id": "ctx_abc123",
                        "query": "What?",
                    },
                ),
            )

        body = _body(result)
        assert _status(result) == 200
        assert body["metadata"]["fallback"] is True

    def test_query_fallback_import_error(self, handler):
        """Fallback handles ImportError gracefully."""
        _add_context(handler, "ctx_abc123")

        with patch.object(handler, "_get_rlm", return_value=None), \
             patch.dict("sys.modules", {"aragora.rlm": None}):
            result = handler.handle_query(
                "/api/v1/rlm/query",
                {},
                MockHTTPHandler(
                    method="POST",
                    body={
                        "context_id": "ctx_abc123",
                        "query": "What?",
                    },
                ),
            )

        body = _body(result)
        assert _status(result) == 200
        assert "fallback" in body["metadata"]

    def test_query_runtime_error(self, handler):
        """Query returns 500 on RuntimeError."""
        _add_context(handler, "ctx_abc123")

        with patch.object(handler, "_get_rlm", return_value=MagicMock()), \
             patch("aragora.server.handlers.rlm.run_async", side_effect=RuntimeError("boom")):
            result = handler.handle_query(
                "/api/v1/rlm/query",
                {},
                MockHTTPHandler(
                    method="POST",
                    body={"context_id": "ctx_abc123", "query": "What?"},
                ),
            )
        assert _status(result) == 500

    def test_query_value_error(self, handler):
        """Query returns 500 on ValueError."""
        _add_context(handler, "ctx_abc123")

        with patch.object(handler, "_get_rlm", return_value=MagicMock()), \
             patch("aragora.server.handlers.rlm.run_async", side_effect=ValueError("bad")):
            result = handler.handle_query(
                "/api/v1/rlm/query",
                {},
                MockHTTPHandler(
                    method="POST",
                    body={"context_id": "ctx_abc123", "query": "What?"},
                ),
            )
        assert _status(result) == 500

    def test_query_via_handle_post(self, handler):
        """Query dispatched correctly via handle_post."""
        with patch.object(handler, "handle_query", return_value=MagicMock(status_code=200, body=b'{}')) as mock:
            handler.handle_post(
                "/api/v1/rlm/query", {}, MockHTTPHandler(method="POST")
            )
            mock.assert_called_once()


# ===========================================================================
# GET /api/v1/rlm/stream/modes
# ===========================================================================


class TestHandleStreamModes:
    """Tests for stream modes endpoint."""

    def test_stream_modes_with_module(self, handler):
        """Returns streaming modes when module is available."""
        mock_stream_mode = MagicMock()
        mock_stream_mode.TOP_DOWN = MagicMock(value="top_down")
        mock_stream_mode.BOTTOM_UP = MagicMock(value="bottom_up")
        mock_stream_mode.TARGETED = MagicMock(value="targeted")
        mock_stream_mode.PROGRESSIVE = MagicMock(value="progressive")

        with patch.dict(
            "sys.modules",
            {
                "aragora.rlm.streaming": MagicMock(StreamMode=mock_stream_mode),
            },
        ):
            result = handler.handle_stream_modes(
                "/api/v1/rlm/stream/modes", {}, MockHTTPHandler()
            )

        body = _body(result)
        assert _status(result) == 200
        assert "modes" in body
        assert len(body["modes"]) == 4

    def test_stream_modes_fallback(self, handler):
        """Returns fallback modes when streaming module unavailable."""
        with patch.dict("sys.modules", {"aragora.rlm.streaming": None}):
            result = handler.handle_stream_modes(
                "/api/v1/rlm/stream/modes", {}, MockHTTPHandler()
            )

        body = _body(result)
        assert _status(result) == 200
        assert "modes" in body
        assert len(body["modes"]) == 4
        assert "note" in body

    def test_stream_modes_via_handle_dispatch(self, handler):
        """Stream modes route dispatched via handle()."""
        with patch.object(handler, "handle_stream_modes", return_value=MagicMock(status_code=200, body=b'{}')) as mock:
            handler.handle("/api/v1/rlm/stream/modes", {}, MockHTTPHandler())
            mock.assert_called_once()


# ===========================================================================
# POST /api/v1/rlm/stream
# ===========================================================================


class TestHandleStream:
    """Tests for the stream endpoint."""

    def test_stream_no_body(self, handler):
        """Stream returns 400 when no body."""
        result = handler.handle_stream(
            "/api/v1/rlm/stream",
            {},
            MockHTTPHandler(method="POST"),
        )
        assert _status(result) == 400

    def test_stream_missing_context_id(self, handler):
        """Stream returns 400 when context_id missing."""
        result = handler.handle_stream(
            "/api/v1/rlm/stream",
            {},
            MockHTTPHandler(method="POST", body={"mode": "top_down"}),
        )
        assert _status(result) == 400

    def test_stream_context_not_found(self, handler):
        """Stream returns 404 when context does not exist."""
        result = handler.handle_stream(
            "/api/v1/rlm/stream",
            {},
            MockHTTPHandler(
                method="POST",
                body={"context_id": "ctx_nonexistent"},
            ),
        )
        assert _status(result) == 404

    def test_stream_success_with_module(self, handler):
        """Stream returns chunks when streaming module available."""
        _add_context(handler, "ctx_abc123")

        mock_chunk = MagicMock()
        mock_chunk.level = "summary"
        mock_chunk.content = "Chunk content"
        mock_chunk.token_count = 10
        mock_chunk.is_final = True
        mock_chunk.metadata = {"source": "test"}

        # Create a mock async generator
        async def mock_stream_all():
            yield mock_chunk

        mock_streaming_query = MagicMock()
        mock_streaming_query.stream_all = mock_stream_all
        mock_streaming_query.search = mock_stream_all
        mock_streaming_query.drill_down = mock_stream_all

        mock_streaming_module = MagicMock()
        mock_streaming_module.StreamMode.TOP_DOWN = "top_down"
        mock_streaming_module.StreamMode.BOTTOM_UP = "bottom_up"
        mock_streaming_module.StreamMode.TARGETED = "targeted"
        mock_streaming_module.StreamMode.PROGRESSIVE = "progressive"

        with patch.dict("sys.modules", {"aragora.rlm.streaming": mock_streaming_module}), \
             patch("aragora.server.handlers.rlm.run_async") as mock_run:
            # run_async should execute the async collect_chunks function
            mock_run.return_value = None
            result = handler.handle_stream(
                "/api/v1/rlm/stream",
                {},
                MockHTTPHandler(
                    method="POST",
                    body={"context_id": "ctx_abc123", "mode": "top_down"},
                ),
            )

        body = _body(result)
        assert _status(result) == 200
        assert "chunks" in body
        assert body["context_id"] == "ctx_abc123"
        assert body["mode"] == "top_down"

    def test_stream_fallback_with_summary(self, handler):
        """Stream falls back to summary content when module unavailable."""
        ctx = _add_context(handler, "ctx_abc123")

        mock_node = MagicMock()
        mock_node.content = "Summary node content"
        ctx.get_at_level.return_value = [mock_node]

        with patch.dict("sys.modules", {"aragora.rlm.streaming": None}), \
             patch("aragora.rlm.AbstractionLevel") as mock_level:
            mock_level.SUMMARY = "SUMMARY"
            result = handler.handle_stream(
                "/api/v1/rlm/stream",
                {},
                MockHTTPHandler(
                    method="POST",
                    body={"context_id": "ctx_abc123"},
                ),
            )

        body = _body(result)
        assert _status(result) == 200
        assert body["mode"] == "fallback"
        assert body["total_chunks"] == 1

    def test_stream_fallback_no_summary_returns_501(self, handler):
        """Stream returns 501 when no streaming module and no summary."""
        ctx = _add_context(handler, "ctx_abc123")
        ctx.get_at_level.side_effect = ImportError("no module")

        with patch.dict("sys.modules", {"aragora.rlm.streaming": None, "aragora.rlm": None}):
            result = handler.handle_stream(
                "/api/v1/rlm/stream",
                {},
                MockHTTPHandler(
                    method="POST",
                    body={"context_id": "ctx_abc123"},
                ),
            )

        assert _status(result) == 501

    def test_stream_via_handle_post(self, handler):
        """Stream dispatched correctly via handle_post."""
        with patch.object(handler, "handle_stream", return_value=MagicMock(status_code=200, body=b'{}')) as mock:
            handler.handle_post(
                "/api/v1/rlm/stream", {}, MockHTTPHandler(method="POST")
            )
            mock.assert_called_once()


# ===========================================================================
# GET /api/v1/rlm/codebase/health
# ===========================================================================


class TestHandleCodebaseHealth:
    """Tests for the codebase health endpoint."""

    def test_health_basic(self, handler, tmp_path):
        """Returns health info for a valid codebase root."""
        with patch.dict("os.environ", {"ARAGORA_CODEBASE_ROOT": str(tmp_path)}), \
             patch("aragora.rlm.HAS_OFFICIAL_RLM", False), \
             patch("aragora.rlm.codebase_context.CodebaseContextBuilder") as mock_builder_cls, \
             patch.object(handler, "_get_rlm", return_value=None):
            result = handler.handle_codebase_health(
                "/api/v1/rlm/codebase/health", {}, MockHTTPHandler()
            )

        body = _body(result)
        assert _status(result) == 200
        assert "status" in body
        assert "root" in body
        assert "manifest" in body
        assert "rlm" in body

    def test_health_with_manifest(self, handler, tmp_path):
        """Returns manifest info when manifest file exists."""
        context_dir = tmp_path / ".nomic" / "context"
        context_dir.mkdir(parents=True)
        manifest = context_dir / "codebase_manifest.tsv"
        manifest.write_text("# Aragora files=42 lines=1000\nfile\tlines\n")

        with patch.dict("os.environ", {"ARAGORA_CODEBASE_ROOT": str(tmp_path)}), \
             patch("aragora.rlm.HAS_OFFICIAL_RLM", False), \
             patch("aragora.rlm.codebase_context.CodebaseContextBuilder"), \
             patch.object(handler, "_get_rlm", return_value=None):
            result = handler.handle_codebase_health(
                "/api/v1/rlm/codebase/health", {}, MockHTTPHandler()
            )

        body = _body(result)
        assert body["manifest"]["exists"] is True
        assert body["manifest"]["files"] == 42
        assert body["manifest"]["lines"] == 1000
        assert body["status"] == "available"

    def test_health_no_manifest(self, handler, tmp_path):
        """Status is 'missing' when manifest does not exist and no refresh."""
        with patch.dict("os.environ", {"ARAGORA_CODEBASE_ROOT": str(tmp_path)}), \
             patch("aragora.rlm.HAS_OFFICIAL_RLM", False), \
             patch("aragora.rlm.codebase_context.CodebaseContextBuilder"), \
             patch.object(handler, "_get_rlm", return_value=None):
            result = handler.handle_codebase_health(
                "/api/v1/rlm/codebase/health", {}, MockHTTPHandler()
            )

        body = _body(result)
        assert body["manifest"]["exists"] is False
        assert body["status"] == "missing"

    def test_health_with_refresh(self, handler, tmp_path):
        """Refresh triggers index build."""
        mock_index = MagicMock()
        mock_index.total_files = 100
        mock_index.total_lines = 5000
        mock_index.total_bytes = 200000
        mock_index.total_tokens_estimate = 50000
        mock_index.build_time_seconds = 1.5

        mock_builder = MagicMock()
        mock_builder.build_index = MagicMock()

        with patch.dict("os.environ", {"ARAGORA_CODEBASE_ROOT": str(tmp_path)}), \
             patch("aragora.rlm.HAS_OFFICIAL_RLM", False), \
             patch("aragora.rlm.codebase_context.CodebaseContextBuilder", return_value=mock_builder), \
             patch("aragora.server.handlers.rlm.run_async", return_value=mock_index), \
             patch.object(handler, "_get_rlm", return_value=None):
            result = handler.handle_codebase_health(
                "/api/v1/rlm/codebase/health",
                {"refresh": "true"},
                MockHTTPHandler(),
            )

        body = _body(result)
        assert body["index"] is not None
        assert body["index"]["files"] == 100
        assert body["status"] == "available"

    def test_health_with_rlm_build(self, handler, tmp_path):
        """rlm=true triggers RLM context build."""
        mock_builder = MagicMock()

        with patch.dict("os.environ", {"ARAGORA_CODEBASE_ROOT": str(tmp_path)}), \
             patch("aragora.rlm.HAS_OFFICIAL_RLM", True), \
             patch("aragora.rlm.codebase_context.CodebaseContextBuilder", return_value=mock_builder), \
             patch("aragora.server.handlers.rlm.run_async", return_value=MagicMock()), \
             patch.object(handler, "_get_rlm", return_value=MagicMock()):
            result = handler.handle_codebase_health(
                "/api/v1/rlm/codebase/health",
                {"rlm": "true"},
                MockHTTPHandler(),
            )

        body = _body(result)
        assert body["rlm"]["context_ready"] is True
        assert body["rlm"]["has_official_rlm"] is True

    def test_health_via_handle_dispatch(self, handler):
        """Codebase health dispatched via handle()."""
        with patch.object(handler, "handle_codebase_health", return_value=MagicMock(status_code=200, body=b'{}')) as mock:
            handler.handle("/api/v1/rlm/codebase/health", {}, MockHTTPHandler())
            mock.assert_called_once()

    def test_health_env_fallback_to_repo_root(self, handler, tmp_path):
        """Uses ARAGORA_REPO_ROOT when ARAGORA_CODEBASE_ROOT not set."""
        with patch.dict(
            "os.environ",
            {"ARAGORA_REPO_ROOT": str(tmp_path)},
            clear=False,
        ), patch.dict(
            "os.environ",
            {},
            clear=False,
        ), patch("aragora.rlm.HAS_OFFICIAL_RLM", False), \
             patch("aragora.rlm.codebase_context.CodebaseContextBuilder"), \
             patch.object(handler, "_get_rlm", return_value=None):
            # Clear ARAGORA_CODEBASE_ROOT if set
            import os
            old = os.environ.pop("ARAGORA_CODEBASE_ROOT", None)
            try:
                result = handler.handle_codebase_health(
                    "/api/v1/rlm/codebase/health", {}, MockHTTPHandler()
                )
                body = _body(result)
                assert _status(result) == 200
            finally:
                if old is not None:
                    os.environ["ARAGORA_CODEBASE_ROOT"] = old


# ===========================================================================
# handle() dispatch
# ===========================================================================


class TestHandleDispatch:
    """Tests for the main handle() dispatch method."""

    def test_handle_returns_none_for_unknown_route(self, handler):
        """handle() returns None for paths not in ROUTES."""
        result = handler.handle("/api/v1/unknown", {}, MockHTTPHandler())
        assert result is None

    def test_handle_stats(self, handler):
        with patch.object(handler, "handle_stats", return_value=MagicMock(status_code=200)) as mock:
            handler.handle("/api/v1/rlm/stats", {}, MockHTTPHandler())
            mock.assert_called_once()

    def test_handle_strategies(self, handler):
        with patch.object(handler, "handle_strategies", return_value=MagicMock(status_code=200)) as mock:
            handler.handle("/api/v1/rlm/strategies", {}, MockHTTPHandler())
            mock.assert_called_once()

    def test_handle_contexts(self, handler):
        with patch.object(handler, "handle_list_contexts", return_value=MagicMock(status_code=200)) as mock:
            handler.handle("/api/v1/rlm/contexts", {}, MockHTTPHandler())
            mock.assert_called_once()

    def test_handle_stream_modes(self, handler):
        with patch.object(handler, "handle_stream_modes", return_value=MagicMock(status_code=200)) as mock:
            handler.handle("/api/v1/rlm/stream/modes", {}, MockHTTPHandler())
            mock.assert_called_once()

    def test_handle_codebase_health(self, handler):
        with patch.object(handler, "handle_codebase_health", return_value=MagicMock(status_code=200)) as mock:
            handler.handle("/api/v1/rlm/codebase/health", {}, MockHTTPHandler())
            mock.assert_called_once()

    def test_handle_context_id_route(self, handler):
        _add_context(handler, "ctx_abc123")
        with patch.object(handler, "_handle_context_route", return_value=MagicMock(status_code=200)) as mock:
            handler.handle("/api/v1/rlm/context/ctx_abc123", {}, MockHTTPHandler())
            mock.assert_called_once()


class TestHandlePostDispatch:
    """Tests for handle_post() dispatch."""

    def test_post_compress(self, handler):
        with patch.object(handler, "handle_compress", return_value=MagicMock(status_code=200)) as mock:
            handler.handle_post("/api/v1/rlm/compress", {}, MockHTTPHandler(method="POST"))
            mock.assert_called_once()

    def test_post_query(self, handler):
        with patch.object(handler, "handle_query", return_value=MagicMock(status_code=200)) as mock:
            handler.handle_post("/api/v1/rlm/query", {}, MockHTTPHandler(method="POST"))
            mock.assert_called_once()

    def test_post_stream(self, handler):
        with patch.object(handler, "handle_stream", return_value=MagicMock(status_code=200)) as mock:
            handler.handle_post("/api/v1/rlm/stream", {}, MockHTTPHandler(method="POST"))
            mock.assert_called_once()

    def test_post_unknown_returns_none(self, handler):
        result = handler.handle_post("/api/v1/rlm/unknown", {}, MockHTTPHandler(method="POST"))
        assert result is None


class TestHandleDeleteDispatch:
    """Tests for handle_delete() dispatch."""

    def test_delete_context(self, handler):
        _add_context(handler, "ctx_abc123")
        result = handler.handle_delete(
            "/api/v1/rlm/context/ctx_abc123", {}, MockHTTPHandler()
        )
        body = _body(result)
        assert _status(result) == 200
        assert body["success"] is True

    def test_delete_non_context_path(self, handler):
        result = handler.handle_delete("/api/v1/rlm/stats", {}, MockHTTPHandler())
        assert result is None


# ===========================================================================
# read_json_body
# ===========================================================================


class TestReadJsonBody:
    """Tests for the read_json_body utility method."""

    def test_read_valid_body(self, handler):
        mock_http = MockHTTPHandler(body={"key": "value"})
        result = handler.read_json_body(mock_http)
        assert result == {"key": "value"}

    def test_read_none_handler(self, handler):
        result = handler.read_json_body(None)
        assert result is None

    def test_read_zero_content_length(self, handler):
        mock_http = MockHTTPHandler()
        mock_http.headers["Content-Length"] = "0"
        result = handler.read_json_body(mock_http)
        assert result is None

    def test_read_oversized_body(self, handler):
        mock_http = MockHTTPHandler(body={"key": "value"})
        # Pretend content is huge
        mock_http.headers["Content-Length"] = str(20_000_000)
        result = handler.read_json_body(mock_http)
        assert result is None

    def test_read_custom_max_size(self, handler):
        mock_http = MockHTTPHandler(body={"key": "value"})
        mock_http.headers["Content-Length"] = str(500)
        result = handler.read_json_body(mock_http, max_size=100)
        assert result is None

    def test_read_invalid_json(self, handler):
        mock_http = MockHTTPHandler()
        mock_http.headers["Content-Length"] = "10"
        mock_http.rfile.read.return_value = b"not json!!"
        result = handler.read_json_body(mock_http)
        assert result is None

    def test_read_missing_content_length(self, handler):
        mock_http = MockHTTPHandler()
        del mock_http.headers["Content-Length"]
        # headers.get will raise AttributeError because it's a dict,
        # but our mock should handle gracefully
        mock_http.headers = {}
        result = handler.read_json_body(mock_http)
        assert result is None


# ===========================================================================
# _get_compressor / _get_rlm
# ===========================================================================


class TestGetCompressor:
    """Tests for _get_compressor initialization."""

    def test_get_compressor_caches(self, handler):
        """Compressor is cached after first call."""
        mock_compressor = MagicMock()
        with patch("aragora.rlm.get_compressor", return_value=mock_compressor):
            first = handler._get_compressor()
            second = handler._get_compressor()
        assert first is second
        assert first is mock_compressor

    def test_get_compressor_import_error(self, handler):
        """Returns None when module not available."""
        with patch.dict("sys.modules", {"aragora.rlm": None}):
            handler._compressor = None  # Reset
            result = handler._get_compressor()
        assert result is None

    def test_get_rlm_caches(self, handler):
        """RLM instance is cached after first call."""
        mock_rlm = MagicMock()
        with patch("aragora.rlm.get_rlm", return_value=mock_rlm), \
             patch("aragora.rlm.HAS_OFFICIAL_RLM", False):
            first = handler._get_rlm()
            second = handler._get_rlm()
        assert first is second
        assert first is mock_rlm

    def test_get_rlm_import_error(self, handler):
        """Returns None when module not available."""
        with patch.dict("sys.modules", {"aragora.rlm": None}):
            handler._rlm = None  # Reset
            result = handler._get_rlm()
        assert result is None


# ===========================================================================
# Edge Cases & Integration
# ===========================================================================


class TestEdgeCases:
    """Edge cases and integration scenarios."""

    def test_handler_initialization(self, handler):
        """Handler initializes with empty contexts."""
        assert handler._contexts == {}
        assert handler._compressor is None
        assert handler._rlm is None

    def test_context_lifecycle(self, handler):
        """Full lifecycle: add contexts, list, get, delete."""
        # Add
        _add_context(handler, "ctx_1")
        _add_context(handler, "ctx_2")

        # List
        result = handler.handle_list_contexts(
            "/api/v1/rlm/contexts", {}, MockHTTPHandler()
        )
        assert _body(result)["total"] == 2

        # Get
        result = handler._get_context("ctx_1", {}, MockHTTPHandler())
        assert _status(result) == 200

        # Delete
        result = handler._delete_context("ctx_1", {}, MockHTTPHandler())
        assert _status(result) == 200

        # Verify deleted
        result = handler.handle_list_contexts(
            "/api/v1/rlm/contexts", {}, MockHTTPHandler()
        )
        assert _body(result)["total"] == 1

    def test_multiple_compress_different_ids(self, handler):
        """Each compress call generates a unique context_id."""
        mock_ctx = _make_mock_context()
        mock_compressor = MagicMock()
        ids = set()

        for i in range(5):
            with patch.object(handler, "_get_compressor", return_value=mock_compressor), \
                 patch("aragora.server.handlers.rlm.run_async", return_value=mock_ctx):
                result = handler.handle_compress(
                    "/api/v1/rlm/compress",
                    {},
                    MockHTTPHandler(
                        method="POST",
                        body={"content": f"content_{i}"},
                    ),
                )
            body = _body(result)
            ids.add(body["context_id"])

        # Different content should produce different IDs (hash-based)
        assert len(ids) == 5

    def test_handler_routes_attribute(self, handler):
        """ROUTES dictionary contains expected keys."""
        expected_routes = {
            "/api/v1/rlm/stats",
            "/api/v1/rlm/strategies",
            "/api/v1/rlm/compress",
            "/api/v1/rlm/query",
            "/api/v1/rlm/contexts",
            "/api/v1/rlm/stream",
            "/api/v1/rlm/stream/modes",
            "/api/v1/rlm/codebase/health",
        }
        assert set(handler.ROUTES.keys()) == expected_routes

    def test_context_route_prefix(self, handler):
        """Context route prefix is correct."""
        assert handler.CONTEXT_ROUTE_PREFIX == "/api/v1/rlm/context/"

    def test_compress_creates_at_field(self, handler):
        """Compressed result includes created_at timestamp."""
        mock_ctx = _make_mock_context()
        mock_compressor = MagicMock()

        with patch.object(handler, "_get_compressor", return_value=mock_compressor), \
             patch("aragora.server.handlers.rlm.run_async", return_value=mock_ctx):
            result = handler.handle_compress(
                "/api/v1/rlm/compress",
                {},
                MockHTTPHandler(
                    method="POST",
                    body={"content": "test"},
                ),
            )

        body = _body(result)
        assert "created_at" in body

    def test_query_timestamp_in_response(self, handler):
        """Query response includes timestamp."""
        _add_context(handler, "ctx_abc123")
        mock_result = MagicMock()
        mock_result.answer = "answer"

        with patch.object(handler, "_get_rlm", return_value=MagicMock()), \
             patch("aragora.server.handlers.rlm.run_async", return_value=mock_result):
            result = handler.handle_query(
                "/api/v1/rlm/query",
                {},
                MockHTTPHandler(
                    method="POST",
                    body={"context_id": "ctx_abc123", "query": "What?"},
                ),
            )

        body = _body(result)
        assert "timestamp" in body

    def test_context_level_stats_attribute_error(self, handler):
        """Get context handles levels with no .name attribute."""
        ctx = _add_context(handler, "ctx_abc123")
        # Override levels with a level that has no .name attribute
        mock_level = "raw_level"
        mock_node = MagicMock()
        mock_node.token_count = 100
        mock_node.id = "n1"
        ctx.levels = {mock_level: [mock_node]}

        result = handler._get_context("ctx_abc123", {}, MockHTTPHandler())
        body = _body(result)
        assert _status(result) == 200
        assert "raw_level" in body["levels"]
