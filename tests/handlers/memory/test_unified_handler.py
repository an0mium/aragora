"""Tests for the Unified Memory Gateway HTTP handler.

Tests the UnifiedMemoryHandler which wraps MemoryGateway for REST access:
- POST /api/v1/memory/unified/search - Cross-system memory search
- GET /api/v1/memory/unified/stats - Gateway statistics

Covers:
- Constructor initialization (with/without ctx and gateway)
- handle_search: validation, query construction, response serialization
- handle_search: error paths (no gateway, missing query, ImportError, RuntimeError, etc.)
- handle_stats: with/without gateway
- Edge cases: empty results, large limits, float confidence, source filtering
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _status(result) -> int:
    """Extract status code from HandlerResult or infer from dict."""
    if isinstance(result, dict):
        return result.get("status", 200)
    return result.status_code


def _body(result) -> dict[str, Any]:
    """Extract JSON body from HandlerResult or return dict body."""
    if isinstance(result, dict):
        return result.get("body", result)
    try:
        return json.loads(result.body.decode("utf-8"))
    except (json.JSONDecodeError, AttributeError, UnicodeDecodeError):
        return {}


# ---------------------------------------------------------------------------
# Mock data types matching gateway.py dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MockUnifiedResult:
    """Mimics UnifiedMemoryResult for gateway response mocking."""

    id: str
    content: str
    source_system: str
    confidence: float
    surprise_score: float | None = None
    content_hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MockUnifiedResponse:
    """Mimics UnifiedMemoryResponse for gateway response mocking."""

    results: list[MockUnifiedResult] = field(default_factory=list)
    total_found: int = 0
    sources_queried: list[str] = field(default_factory=list)
    duplicates_removed: int = 0
    query_time_ms: float = 0.0
    errors: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_gateway():
    """Create a mock MemoryGateway."""
    gw = AsyncMock()
    gw.get_stats = MagicMock(return_value={
        "enabled": True,
        "sources": ["continuum", "km"],
        "total_queries": 42,
    })
    return gw


@pytest.fixture
def handler_no_gateway():
    """Handler with no gateway configured."""
    from aragora.server.handlers.memory.unified_handler import UnifiedMemoryHandler

    return UnifiedMemoryHandler(ctx={}, gateway=None)


@pytest.fixture
def handler(mock_gateway):
    """Handler with a mock gateway configured."""
    from aragora.server.handlers.memory.unified_handler import UnifiedMemoryHandler

    return UnifiedMemoryHandler(ctx={"key": "val"}, gateway=mock_gateway)


def _make_response(
    results: list[MockUnifiedResult] | None = None,
    total_found: int = 0,
    sources_queried: list[str] | None = None,
    duplicates_removed: int = 0,
    query_time_ms: float = 1.234,
    errors: dict[str, str] | None = None,
) -> MockUnifiedResponse:
    """Build a MockUnifiedResponse with sensible defaults."""
    return MockUnifiedResponse(
        results=results or [],
        total_found=total_found,
        sources_queried=sources_queried or [],
        duplicates_removed=duplicates_removed,
        query_time_ms=query_time_ms,
        errors=errors or {},
    )


# ===================================================================
# Constructor tests
# ===================================================================


class TestUnifiedMemoryHandlerInit:
    """Test UnifiedMemoryHandler constructor."""

    def test_init_default_ctx(self):
        from aragora.server.handlers.memory.unified_handler import UnifiedMemoryHandler

        h = UnifiedMemoryHandler()
        assert h.ctx == {}
        assert h._gateway is None

    def test_init_with_ctx(self):
        from aragora.server.handlers.memory.unified_handler import UnifiedMemoryHandler

        ctx = {"workspace_id": "ws-1"}
        h = UnifiedMemoryHandler(ctx=ctx)
        assert h.ctx is ctx
        assert h._gateway is None

    def test_init_with_gateway(self, mock_gateway):
        from aragora.server.handlers.memory.unified_handler import UnifiedMemoryHandler

        h = UnifiedMemoryHandler(gateway=mock_gateway)
        assert h._gateway is mock_gateway

    def test_init_with_ctx_and_gateway(self, mock_gateway):
        from aragora.server.handlers.memory.unified_handler import UnifiedMemoryHandler

        ctx = {"debug": True}
        h = UnifiedMemoryHandler(ctx=ctx, gateway=mock_gateway)
        assert h.ctx is ctx
        assert h._gateway is mock_gateway

    def test_init_none_ctx_defaults_to_empty_dict(self):
        from aragora.server.handlers.memory.unified_handler import UnifiedMemoryHandler

        h = UnifiedMemoryHandler(ctx=None)
        assert h.ctx == {}


# ===================================================================
# handle_search -- validation
# ===================================================================


class TestHandleSearchValidation:
    """Test handle_search input validation."""

    @pytest.mark.asyncio
    async def test_no_gateway_returns_error(self, handler_no_gateway):
        result = await handler_no_gateway.handle_search({"query": "test"})
        assert result["error"] == "Unified memory gateway not configured"
        assert result["results"] == []

    @pytest.mark.asyncio
    async def test_missing_query_key(self, handler):
        result = await handler.handle_search({})
        assert result["error"] == "Missing required field: query"
        assert result["results"] == []

    @pytest.mark.asyncio
    async def test_empty_query_string(self, handler):
        result = await handler.handle_search({"query": ""})
        assert result["error"] == "Missing required field: query"
        assert result["results"] == []

    @pytest.mark.asyncio
    async def test_query_none_treated_as_missing(self, handler):
        """query=None is falsy, so handler should treat it as missing."""
        result = await handler.handle_search({"query": None})
        assert result["error"] == "Missing required field: query"

    @pytest.mark.asyncio
    async def test_no_gateway_empty_request(self, handler_no_gateway):
        """No gateway should take precedence over missing query."""
        result = await handler_no_gateway.handle_search({})
        assert result["error"] == "Unified memory gateway not configured"


# ===================================================================
# handle_search -- successful paths
# ===================================================================


class TestHandleSearchSuccess:
    """Test handle_search with successful gateway queries."""

    @pytest.mark.asyncio
    async def test_basic_search(self, handler, mock_gateway):
        response = _make_response(
            results=[
                MockUnifiedResult(
                    id="r1",
                    content="Some content",
                    source_system="continuum",
                    confidence=0.95,
                    surprise_score=0.1,
                    metadata={"tag": "test"},
                ),
            ],
            total_found=1,
            sources_queried=["continuum"],
            duplicates_removed=0,
            query_time_ms=5.678,
        )
        mock_gateway.query.return_value = response

        with patch(
            "aragora.server.handlers.memory.unified_handler.UnifiedMemoryQuery",
            create=True,
        ) as MockQuery:
            # Patch the import inside handle_search
            with patch.dict(
                "sys.modules",
                {"aragora.memory.gateway": MagicMock(UnifiedMemoryQuery=MockQuery)},
            ):
                result = await handler.handle_search({"query": "test query"})

        assert "error" not in result
        assert result["total_found"] == 1
        assert result["sources_queried"] == ["continuum"]
        assert result["duplicates_removed"] == 0
        assert result["query_time_ms"] == 5.68
        assert len(result["results"]) == 1

        r = result["results"][0]
        assert r["id"] == "r1"
        assert r["content"] == "Some content"
        assert r["source_system"] == "continuum"
        assert r["confidence"] == 0.95
        assert r["surprise_score"] == 0.1
        assert r["metadata"] == {"tag": "test"}

    @pytest.mark.asyncio
    async def test_search_with_multiple_results(self, handler, mock_gateway):
        results = [
            MockUnifiedResult(
                id=f"r{i}",
                content=f"Content {i}",
                source_system="km" if i % 2 == 0 else "continuum",
                confidence=0.9 - i * 0.1,
            )
            for i in range(5)
        ]
        response = _make_response(
            results=results,
            total_found=5,
            sources_queried=["continuum", "km"],
            duplicates_removed=2,
        )
        mock_gateway.query.return_value = response

        with patch.dict(
            "sys.modules",
            {"aragora.memory.gateway": MagicMock(UnifiedMemoryQuery=MagicMock)},
        ):
            result = await handler.handle_search({"query": "multi"})

        assert len(result["results"]) == 5
        assert result["total_found"] == 5
        assert result["duplicates_removed"] == 2

    @pytest.mark.asyncio
    async def test_search_empty_results(self, handler, mock_gateway):
        mock_gateway.query.return_value = _make_response(
            results=[],
            total_found=0,
            sources_queried=["continuum", "km"],
        )

        with patch.dict(
            "sys.modules",
            {"aragora.memory.gateway": MagicMock(UnifiedMemoryQuery=MagicMock)},
        ):
            result = await handler.handle_search({"query": "nothing here"})

        assert result["results"] == []
        assert result["total_found"] == 0

    @pytest.mark.asyncio
    async def test_search_passes_limit(self, handler, mock_gateway):
        mock_gateway.query.return_value = _make_response()

        captured_args = {}

        def capture_query_cls(**kwargs):
            captured_args.update(kwargs)
            return MagicMock()

        mock_module = MagicMock()
        mock_module.UnifiedMemoryQuery = capture_query_cls

        with patch.dict("sys.modules", {"aragora.memory.gateway": mock_module}):
            await handler.handle_search({"query": "q", "limit": 25})

        assert captured_args["limit"] == 25

    @pytest.mark.asyncio
    async def test_search_passes_min_confidence(self, handler, mock_gateway):
        mock_gateway.query.return_value = _make_response()

        captured_args = {}

        def capture_query_cls(**kwargs):
            captured_args.update(kwargs)
            return MagicMock()

        mock_module = MagicMock()
        mock_module.UnifiedMemoryQuery = capture_query_cls

        with patch.dict("sys.modules", {"aragora.memory.gateway": mock_module}):
            await handler.handle_search({"query": "q", "min_confidence": 0.75})

        assert captured_args["min_confidence"] == 0.75

    @pytest.mark.asyncio
    async def test_search_passes_sources(self, handler, mock_gateway):
        mock_gateway.query.return_value = _make_response()

        captured_args = {}

        def capture_query_cls(**kwargs):
            captured_args.update(kwargs)
            return MagicMock()

        mock_module = MagicMock()
        mock_module.UnifiedMemoryQuery = capture_query_cls

        with patch.dict("sys.modules", {"aragora.memory.gateway": mock_module}):
            await handler.handle_search({"query": "q", "sources": ["km"]})

        assert captured_args["sources"] == ["km"]

    @pytest.mark.asyncio
    async def test_search_passes_dedup_false(self, handler, mock_gateway):
        mock_gateway.query.return_value = _make_response()

        captured_args = {}

        def capture_query_cls(**kwargs):
            captured_args.update(kwargs)
            return MagicMock()

        mock_module = MagicMock()
        mock_module.UnifiedMemoryQuery = capture_query_cls

        with patch.dict("sys.modules", {"aragora.memory.gateway": mock_module}):
            await handler.handle_search({"query": "q", "dedup": False})

        assert captured_args["dedup"] is False

    @pytest.mark.asyncio
    async def test_search_default_limit(self, handler, mock_gateway):
        mock_gateway.query.return_value = _make_response()

        captured_args = {}

        def capture_query_cls(**kwargs):
            captured_args.update(kwargs)
            return MagicMock()

        mock_module = MagicMock()
        mock_module.UnifiedMemoryQuery = capture_query_cls

        with patch.dict("sys.modules", {"aragora.memory.gateway": mock_module}):
            await handler.handle_search({"query": "q"})

        assert captured_args["limit"] == 10

    @pytest.mark.asyncio
    async def test_search_default_min_confidence(self, handler, mock_gateway):
        mock_gateway.query.return_value = _make_response()

        captured_args = {}

        def capture_query_cls(**kwargs):
            captured_args.update(kwargs)
            return MagicMock()

        mock_module = MagicMock()
        mock_module.UnifiedMemoryQuery = capture_query_cls

        with patch.dict("sys.modules", {"aragora.memory.gateway": mock_module}):
            await handler.handle_search({"query": "q"})

        assert captured_args["min_confidence"] == 0.0

    @pytest.mark.asyncio
    async def test_search_default_dedup_true(self, handler, mock_gateway):
        mock_gateway.query.return_value = _make_response()

        captured_args = {}

        def capture_query_cls(**kwargs):
            captured_args.update(kwargs)
            return MagicMock()

        mock_module = MagicMock()
        mock_module.UnifiedMemoryQuery = capture_query_cls

        with patch.dict("sys.modules", {"aragora.memory.gateway": mock_module}):
            await handler.handle_search({"query": "q"})

        assert captured_args["dedup"] is True

    @pytest.mark.asyncio
    async def test_search_default_sources_none(self, handler, mock_gateway):
        mock_gateway.query.return_value = _make_response()

        captured_args = {}

        def capture_query_cls(**kwargs):
            captured_args.update(kwargs)
            return MagicMock()

        mock_module = MagicMock()
        mock_module.UnifiedMemoryQuery = capture_query_cls

        with patch.dict("sys.modules", {"aragora.memory.gateway": mock_module}):
            await handler.handle_search({"query": "q"})

        assert captured_args["sources"] is None

    @pytest.mark.asyncio
    async def test_query_time_rounding(self, handler, mock_gateway):
        """query_time_ms should be rounded to 2 decimal places."""
        mock_gateway.query.return_value = _make_response(query_time_ms=3.14159265)

        with patch.dict(
            "sys.modules",
            {"aragora.memory.gateway": MagicMock(UnifiedMemoryQuery=MagicMock)},
        ):
            result = await handler.handle_search({"query": "pi"})

        assert result["query_time_ms"] == 3.14

    @pytest.mark.asyncio
    async def test_search_with_errors_in_response(self, handler, mock_gateway):
        """Gateway errors dict should be passed through."""
        mock_gateway.query.return_value = _make_response(
            errors={"supermemory": "timeout", "km": "connection refused"},
        )

        with patch.dict(
            "sys.modules",
            {"aragora.memory.gateway": MagicMock(UnifiedMemoryQuery=MagicMock)},
        ):
            result = await handler.handle_search({"query": "errors"})

        assert result["errors"] == {"supermemory": "timeout", "km": "connection refused"}

    @pytest.mark.asyncio
    async def test_result_surprise_score_none(self, handler, mock_gateway):
        """Results with surprise_score=None should serialize correctly."""
        mock_gateway.query.return_value = _make_response(
            results=[
                MockUnifiedResult(
                    id="r1",
                    content="x",
                    source_system="km",
                    confidence=0.5,
                    surprise_score=None,
                ),
            ],
            total_found=1,
        )

        with patch.dict(
            "sys.modules",
            {"aragora.memory.gateway": MagicMock(UnifiedMemoryQuery=MagicMock)},
        ):
            result = await handler.handle_search({"query": "q"})

        assert result["results"][0]["surprise_score"] is None

    @pytest.mark.asyncio
    async def test_result_empty_metadata(self, handler, mock_gateway):
        """Results with empty metadata dict."""
        mock_gateway.query.return_value = _make_response(
            results=[
                MockUnifiedResult(
                    id="r1",
                    content="x",
                    source_system="km",
                    confidence=0.5,
                    metadata={},
                ),
            ],
            total_found=1,
        )

        with patch.dict(
            "sys.modules",
            {"aragora.memory.gateway": MagicMock(UnifiedMemoryQuery=MagicMock)},
        ):
            result = await handler.handle_search({"query": "q"})

        assert result["results"][0]["metadata"] == {}


# ===================================================================
# handle_search -- error handling
# ===================================================================


class TestHandleSearchErrors:
    """Test handle_search exception handling."""

    @pytest.mark.asyncio
    async def test_import_error(self, handler, mock_gateway):
        """ImportError during gateway import should be caught."""
        with patch.dict("sys.modules", {"aragora.memory.gateway": None}):
            result = await handler.handle_search({"query": "test"})

        assert result["error"] == "Unified memory search failed"
        assert result["results"] == []

    @pytest.mark.asyncio
    async def test_runtime_error_from_gateway(self, handler, mock_gateway):
        mock_gateway.query.side_effect = RuntimeError("DB connection lost")

        with patch.dict(
            "sys.modules",
            {"aragora.memory.gateway": MagicMock(UnifiedMemoryQuery=MagicMock)},
        ):
            result = await handler.handle_search({"query": "boom"})

        assert result["error"] == "Unified memory search failed"
        assert result["results"] == []

    @pytest.mark.asyncio
    async def test_value_error_from_gateway(self, handler, mock_gateway):
        mock_gateway.query.side_effect = ValueError("Invalid query parameter")

        with patch.dict(
            "sys.modules",
            {"aragora.memory.gateway": MagicMock(UnifiedMemoryQuery=MagicMock)},
        ):
            result = await handler.handle_search({"query": "bad"})

        assert result["error"] == "Unified memory search failed"
        assert result["results"] == []

    @pytest.mark.asyncio
    async def test_type_error_from_gateway(self, handler, mock_gateway):
        mock_gateway.query.side_effect = TypeError("Unexpected type")

        with patch.dict(
            "sys.modules",
            {"aragora.memory.gateway": MagicMock(UnifiedMemoryQuery=MagicMock)},
        ):
            result = await handler.handle_search({"query": "type_err"})

        assert result["error"] == "Unified memory search failed"
        assert result["results"] == []

    @pytest.mark.asyncio
    async def test_import_error_from_query_class(self, handler, mock_gateway):
        """ImportError when importing UnifiedMemoryQuery specifically."""
        mock_module = MagicMock()
        # Make accessing UnifiedMemoryQuery raise ImportError
        del mock_module.UnifiedMemoryQuery

        def fail_import(*a, **kw):
            raise ImportError("No module named 'aragora.memory.gateway'")

        with patch(
            "builtins.__import__",
            side_effect=fail_import,
        ):
            # Since the handler does `from aragora.memory.gateway import UnifiedMemoryQuery`
            # an ImportError should be caught
            result = await handler.handle_search({"query": "import fail"})

        assert result["error"] == "Unified memory search failed"
        assert result["results"] == []

    @pytest.mark.asyncio
    async def test_runtime_error_in_query_construction(self, handler, mock_gateway):
        """RuntimeError when constructing UnifiedMemoryQuery."""
        mock_module = MagicMock()
        mock_module.UnifiedMemoryQuery.side_effect = RuntimeError("bad construct")

        with patch.dict("sys.modules", {"aragora.memory.gateway": mock_module}):
            result = await handler.handle_search({"query": "construct fail"})

        assert result["error"] == "Unified memory search failed"
        assert result["results"] == []

    @pytest.mark.asyncio
    async def test_value_error_in_query_construction(self, handler, mock_gateway):
        """ValueError when constructing UnifiedMemoryQuery."""
        mock_module = MagicMock()
        mock_module.UnifiedMemoryQuery.side_effect = ValueError("limit must be positive")

        with patch.dict("sys.modules", {"aragora.memory.gateway": mock_module}):
            result = await handler.handle_search({"query": "val fail"})

        assert result["error"] == "Unified memory search failed"
        assert result["results"] == []


# ===================================================================
# handle_stats
# ===================================================================


class TestHandleStats:
    """Test handle_stats endpoint."""

    @pytest.mark.asyncio
    async def test_stats_no_gateway(self, handler_no_gateway):
        result = await handler_no_gateway.handle_stats()
        assert result == {"error": "Unified memory gateway not configured"}

    @pytest.mark.asyncio
    async def test_stats_with_gateway(self, handler, mock_gateway):
        result = await handler.handle_stats()
        assert result == {
            "enabled": True,
            "sources": ["continuum", "km"],
            "total_queries": 42,
        }
        mock_gateway.get_stats.assert_called_once()

    @pytest.mark.asyncio
    async def test_stats_returns_gateway_stats_directly(self):
        """Stats returns exactly what gateway.get_stats() returns."""
        from aragora.server.handlers.memory.unified_handler import UnifiedMemoryHandler

        custom_stats = {
            "enabled": False,
            "sources": [],
            "total_queries": 0,
            "cache_hits": 100,
        }
        gw = MagicMock()
        gw.get_stats.return_value = custom_stats
        h = UnifiedMemoryHandler(gateway=gw)

        result = await h.handle_stats()
        assert result is custom_stats

    @pytest.mark.asyncio
    async def test_stats_empty_stats(self):
        from aragora.server.handlers.memory.unified_handler import UnifiedMemoryHandler

        gw = MagicMock()
        gw.get_stats.return_value = {}
        h = UnifiedMemoryHandler(gateway=gw)

        result = await h.handle_stats()
        assert result == {}


# ===================================================================
# Module-level constants
# ===================================================================


class TestModuleConstants:
    """Test module-level constants."""

    def test_memory_read_permission_value(self):
        from aragora.server.handlers.memory.unified_handler import (
            MEMORY_READ_PERMISSION,
        )

        assert MEMORY_READ_PERMISSION == "memory:read"

    def test_handler_is_secure_handler_subclass(self):
        from aragora.server.handlers.memory.unified_handler import UnifiedMemoryHandler
        from aragora.server.handlers.secure import SecureHandler

        assert issubclass(UnifiedMemoryHandler, SecureHandler)


# ===================================================================
# Edge cases and integration-like tests
# ===================================================================


class TestEdgeCases:
    """Edge case and boundary tests."""

    @pytest.mark.asyncio
    async def test_search_with_large_limit(self, handler, mock_gateway):
        mock_gateway.query.return_value = _make_response()

        captured = {}

        def capture(**kw):
            captured.update(kw)
            return MagicMock()

        mock_mod = MagicMock()
        mock_mod.UnifiedMemoryQuery = capture

        with patch.dict("sys.modules", {"aragora.memory.gateway": mock_mod}):
            await handler.handle_search({"query": "q", "limit": 10000})

        assert captured["limit"] == 10000

    @pytest.mark.asyncio
    async def test_search_with_zero_limit(self, handler, mock_gateway):
        mock_gateway.query.return_value = _make_response()

        captured = {}

        def capture(**kw):
            captured.update(kw)
            return MagicMock()

        mock_mod = MagicMock()
        mock_mod.UnifiedMemoryQuery = capture

        with patch.dict("sys.modules", {"aragora.memory.gateway": mock_mod}):
            await handler.handle_search({"query": "q", "limit": 0})

        assert captured["limit"] == 0

    @pytest.mark.asyncio
    async def test_search_with_multiple_sources(self, handler, mock_gateway):
        mock_gateway.query.return_value = _make_response()

        captured = {}

        def capture(**kw):
            captured.update(kw)
            return MagicMock()

        mock_mod = MagicMock()
        mock_mod.UnifiedMemoryQuery = capture

        with patch.dict("sys.modules", {"aragora.memory.gateway": mock_mod}):
            await handler.handle_search({
                "query": "q",
                "sources": ["continuum", "km", "supermemory"],
            })

        assert captured["sources"] == ["continuum", "km", "supermemory"]

    @pytest.mark.asyncio
    async def test_search_with_min_confidence_one(self, handler, mock_gateway):
        mock_gateway.query.return_value = _make_response()

        captured = {}

        def capture(**kw):
            captured.update(kw)
            return MagicMock()

        mock_mod = MagicMock()
        mock_mod.UnifiedMemoryQuery = capture

        with patch.dict("sys.modules", {"aragora.memory.gateway": mock_mod}):
            await handler.handle_search({"query": "q", "min_confidence": 1.0})

        assert captured["min_confidence"] == 1.0

    @pytest.mark.asyncio
    async def test_search_whitespace_only_query(self, handler):
        """A whitespace-only query is truthy, so it passes validation."""
        # " " is truthy in Python, so the handler will proceed to the gateway
        # We need to mock the gateway to return something
        mock_gw = handler._gateway
        mock_gw.query.return_value = _make_response()

        with patch.dict(
            "sys.modules",
            {"aragora.memory.gateway": MagicMock(UnifiedMemoryQuery=MagicMock)},
        ):
            result = await handler.handle_search({"query": "   "})

        # Should not error since " " is truthy
        assert "error" not in result or result.get("error") != "Missing required field: query"

    @pytest.mark.asyncio
    async def test_search_numeric_query_value(self, handler):
        """Numeric values are truthy but handler uses get(query, '') which returns the number."""
        mock_gw = handler._gateway
        mock_gw.query.return_value = _make_response()

        with patch.dict(
            "sys.modules",
            {"aragora.memory.gateway": MagicMock(UnifiedMemoryQuery=MagicMock)},
        ):
            result = await handler.handle_search({"query": 42})

        # 42 is truthy, handler proceeds
        assert result.get("error") != "Missing required field: query"

    @pytest.mark.asyncio
    async def test_query_time_zero(self, handler, mock_gateway):
        mock_gateway.query.return_value = _make_response(query_time_ms=0.0)

        with patch.dict(
            "sys.modules",
            {"aragora.memory.gateway": MagicMock(UnifiedMemoryQuery=MagicMock)},
        ):
            result = await handler.handle_search({"query": "q"})

        assert result["query_time_ms"] == 0.0

    @pytest.mark.asyncio
    async def test_result_serialization_all_fields(self, handler, mock_gateway):
        """Verify all fields of UnifiedMemoryResult are serialized."""
        r = MockUnifiedResult(
            id="abc-123",
            content="A detailed finding about rate limiting",
            source_system="supermemory",
            confidence=0.87,
            surprise_score=0.42,
            metadata={"debate_id": "d-1", "round": 3, "tags": ["perf"]},
        )
        mock_gateway.query.return_value = _make_response(results=[r], total_found=1)

        with patch.dict(
            "sys.modules",
            {"aragora.memory.gateway": MagicMock(UnifiedMemoryQuery=MagicMock)},
        ):
            result = await handler.handle_search({"query": "rate limiting"})

        assert len(result["results"]) == 1
        serialized = result["results"][0]
        assert serialized["id"] == "abc-123"
        assert serialized["content"] == "A detailed finding about rate limiting"
        assert serialized["source_system"] == "supermemory"
        assert serialized["confidence"] == 0.87
        assert serialized["surprise_score"] == 0.42
        assert serialized["metadata"]["debate_id"] == "d-1"
        assert serialized["metadata"]["round"] == 3
        assert serialized["metadata"]["tags"] == ["perf"]

    @pytest.mark.asyncio
    async def test_extra_request_data_keys_ignored(self, handler, mock_gateway):
        """Extra keys in request_data should be silently ignored."""
        mock_gateway.query.return_value = _make_response()

        with patch.dict(
            "sys.modules",
            {"aragora.memory.gateway": MagicMock(UnifiedMemoryQuery=MagicMock)},
        ):
            result = await handler.handle_search({
                "query": "q",
                "unknown_field": True,
                "extra": [1, 2, 3],
            })

        # Should succeed without error
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_concurrent_searches(self, handler, mock_gateway):
        """Multiple concurrent searches should all succeed."""
        import asyncio

        mock_gateway.query.return_value = _make_response(total_found=1)

        with patch.dict(
            "sys.modules",
            {"aragora.memory.gateway": MagicMock(UnifiedMemoryQuery=MagicMock)},
        ):
            results = await asyncio.gather(*[
                handler.handle_search({"query": f"q{i}"}) for i in range(5)
            ])

        assert all(r.get("total_found") == 1 for r in results)
        assert mock_gateway.query.call_count == 5
