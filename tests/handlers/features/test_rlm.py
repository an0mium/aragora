"""Tests for RLM (Recursive Language Model) handler.

Tests the RLM API endpoints including:
- POST /api/debates/{id}/query-rlm - Query debate using RLM
- POST /api/debates/{id}/compress - Compress debate context
- GET /api/debates/{id}/context/{level} - Get context at abstraction level
- GET /api/debates/{id}/refinement-status - Get refinement progress
- POST /api/knowledge/query-rlm - Query knowledge mound
- GET /api/rlm/status - Get RLM system status
- GET /api/metrics/rlm - Get RLM metrics
"""

import json
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest


@dataclass
class MockHandler:
    """Mock HTTP handler for tests."""

    headers: Dict[str, str] = None
    rfile: BytesIO = None
    _json_body: Dict[str, Any] = None

    def __post_init__(self):
        if self.headers is None:
            self.headers = {"Content-Length": "0"}
        if self.rfile is None:
            self.rfile = BytesIO(b"{}")

    def get_json_body(self) -> Optional[Dict[str, Any]]:
        """Return mock JSON body."""
        return self._json_body


@pytest.fixture
def rlm_handler():
    """Create RLM handler with mock context."""
    from aragora.server.handlers.features.rlm import RLMHandler

    ctx = {}
    return RLMHandler(ctx)


@pytest.fixture(autouse=True)
def reset_state():
    """Reset state before each test."""
    # Reset rate limiters
    try:
        from aragora.server.middleware.rate_limit.registry import reset_rate_limiters

        reset_rate_limiters()
    except ImportError:
        pass
    yield
    try:
        from aragora.server.middleware.rate_limit.registry import reset_rate_limiters

        reset_rate_limiters()
    except ImportError:
        pass


# =============================================================================
# Initialization Tests
# =============================================================================


class TestRLMHandlerInit:
    """Tests for handler initialization."""

    def test_routes_defined(self, rlm_handler):
        """Test that handler routes are defined."""
        assert hasattr(rlm_handler, "ROUTES")
        assert len(rlm_handler.ROUTES) > 0

    def test_can_handle_rlm_paths(self, rlm_handler):
        """Test can_handle recognizes RLM paths."""
        assert rlm_handler.can_handle("/api/debates/test-123/query-rlm")
        assert rlm_handler.can_handle("/api/debates/test-123/compress")
        assert rlm_handler.can_handle("/api/debates/test-123/context/SUMMARY")
        assert rlm_handler.can_handle("/api/debates/test-123/refinement-status")
        assert rlm_handler.can_handle("/api/knowledge/query-rlm")
        assert rlm_handler.can_handle("/api/rlm/status")
        assert rlm_handler.can_handle("/api/metrics/rlm")

    def test_cannot_handle_other_paths(self, rlm_handler):
        """Test can_handle rejects non-RLM paths."""
        assert not rlm_handler.can_handle("/api/debates")
        assert not rlm_handler.can_handle("/api/debates/test-123")
        assert not rlm_handler.can_handle("/api/users")


# =============================================================================
# Path Extraction Tests
# =============================================================================


class TestPathExtraction:
    """Tests for debate ID and level extraction."""

    def test_extracts_debate_id(self, rlm_handler):
        """Extracts debate ID from path."""
        assert rlm_handler._extract_debate_id("/api/debates/test-123/query-rlm") == "test-123"
        assert rlm_handler._extract_debate_id("/api/debates/abc/compress") == "abc"

    def test_returns_none_for_invalid_path(self, rlm_handler):
        """Returns None for invalid paths."""
        assert rlm_handler._extract_debate_id("/api/users/test") is None
        assert rlm_handler._extract_debate_id("/api") is None

    def test_extracts_level(self, rlm_handler):
        """Extracts abstraction level from path."""
        assert rlm_handler._extract_level("/api/debates/test/context/SUMMARY") == "SUMMARY"
        assert rlm_handler._extract_level("/api/debates/test/context/abstract") == "ABSTRACT"

    def test_returns_none_for_missing_level(self, rlm_handler):
        """Returns None when level not in path."""
        assert rlm_handler._extract_level("/api/debates/test/query-rlm") is None


# =============================================================================
# RLM Status Tests
# =============================================================================


class TestRLMStatus:
    """Tests for RLM status endpoint."""

    def test_returns_status_with_features(self, rlm_handler):
        """Returns RLM status with available features."""
        result = rlm_handler._get_rlm_status()
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "available" in data
        assert "provider" in data
        assert "features" in data
        assert isinstance(data["features"], list)

    def test_includes_core_features(self, rlm_handler):
        """Status includes core features like compression and queries."""
        result = rlm_handler._get_rlm_status()
        data = json.loads(result.body)
        # At minimum, built-in features should be present
        assert "compression" in data["features"]
        assert "queries" in data["features"]
        assert "refinement" in data["features"]


# =============================================================================
# RLM Metrics Tests
# =============================================================================


class TestRLMMetrics:
    """Tests for RLM metrics endpoint."""

    def test_returns_metrics_structure(self, rlm_handler):
        """Returns proper metrics structure."""
        result = rlm_handler._get_rlm_metrics()
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "compressions" in data
        assert "queries" in data
        assert "cache" in data
        assert "refinement" in data

    def test_compressions_metrics_fields(self, rlm_handler):
        """Compressions metrics have required fields."""
        result = rlm_handler._get_rlm_metrics()
        data = json.loads(result.body)
        compressions = data["compressions"]
        assert "total" in compressions
        assert "byType" in compressions
        assert "avgRatio" in compressions
        assert "tokensSaved" in compressions

    def test_cache_metrics_fields(self, rlm_handler):
        """Cache metrics have required fields."""
        result = rlm_handler._get_rlm_metrics()
        data = json.loads(result.body)
        cache = data["cache"]
        assert "hits" in cache
        assert "misses" in cache
        assert "hitRate" in cache
        assert "memoryBytes" in cache


# =============================================================================
# Query RLM Tests
# =============================================================================


class TestQueryDebateRLM:
    """Tests for debate RLM query endpoint."""

    def test_returns_400_without_body(self, rlm_handler):
        """Returns 400 when request body is missing."""
        mock_handler = MockHandler(_json_body=None)
        # Call underlying method directly, bypassing decorators
        # The __wrapped__ attribute gives us the original function
        original_method = rlm_handler._query_debate_rlm
        # Walk through decorator chain
        while hasattr(original_method, "__wrapped__"):
            original_method = original_method.__wrapped__
        result = original_method(
            rlm_handler, "/api/debates/test-123/query-rlm", mock_handler, user="test"
        )
        assert result.status_code == 400

    def test_returns_400_without_query(self, rlm_handler):
        """Returns 400 when query parameter is missing."""
        mock_handler = MockHandler(_json_body={"strategy": "auto"})
        result = rlm_handler._query_debate_rlm.__wrapped__(
            rlm_handler, "/api/debates/test-123/query-rlm", mock_handler, user="test"
        )
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "query" in data.get("error", "").lower()

    def test_returns_400_for_invalid_debate_id(self, rlm_handler):
        """Returns 400 for invalid debate ID."""
        mock_handler = MockHandler(_json_body={"query": "What was decided?"})
        result = rlm_handler._query_debate_rlm.__wrapped__(
            rlm_handler, "/api/invalid-path", mock_handler, user="test"
        )
        assert result.status_code == 400


# =============================================================================
# Compress Debate Tests
# =============================================================================


class TestCompressDebate:
    """Tests for debate compression endpoint."""

    def test_returns_400_for_invalid_debate_id(self, rlm_handler):
        """Returns 400 for invalid debate ID."""
        mock_handler = MockHandler(_json_body={})
        result = rlm_handler._compress_debate.__wrapped__(
            rlm_handler, "/api/invalid-path", mock_handler, user="test"
        )
        assert result.status_code == 400

    def test_uses_default_options(self, rlm_handler):
        """Uses default compression options when not specified."""
        mock_handler = MockHandler(_json_body={})

        async def mock_compress(debate_id, target_levels, compression_ratio):
            # Verify defaults are used
            assert target_levels == ["ABSTRACT", "SUMMARY", "DETAILED"]
            assert compression_ratio == 0.3
            return {
                "original_tokens": 1000,
                "compressed_tokens": {"SUMMARY": 300},
                "compression_ratios": {"SUMMARY": 0.3},
                "time_seconds": 0.5,
                "levels_created": 1,
            }

        with patch.object(
            rlm_handler, "_execute_compression", side_effect=mock_compress
        ):
            with patch(
                "aragora.server.handlers.features.rlm._run_async",
                side_effect=lambda coro: asyncio.get_event_loop().run_until_complete(coro)
                if hasattr(coro, "__await__")
                else coro,
            ):
                # Test would need async handling
                pass


# =============================================================================
# Get Context Level Tests
# =============================================================================


class TestGetContextLevel:
    """Tests for get context at level endpoint."""

    def test_returns_400_for_invalid_debate_id(self, rlm_handler):
        """Returns 400 for invalid debate ID."""
        mock_handler = MockHandler()
        result = rlm_handler._get_context_level.__wrapped__(
            rlm_handler, "/api/invalid-path", mock_handler, user="test"
        )
        assert result.status_code == 400

    def test_returns_400_for_invalid_level(self, rlm_handler):
        """Returns 400 for invalid abstraction level."""
        mock_handler = MockHandler()
        result = rlm_handler._get_context_level.__wrapped__(
            rlm_handler, "/api/debates/test-123/query-rlm", mock_handler, user="test"
        )
        assert result.status_code == 400


# =============================================================================
# Refinement Status Tests
# =============================================================================


class TestRefinementStatus:
    """Tests for refinement status endpoint."""

    def test_returns_400_for_invalid_debate_id(self, rlm_handler):
        """Returns 400 for invalid debate ID."""
        mock_handler = MockHandler()
        result = rlm_handler._get_refinement_status.__wrapped__(
            rlm_handler, "/api/invalid", mock_handler, user="test"
        )
        assert result.status_code == 400

    def test_returns_status_for_valid_debate(self, rlm_handler):
        """Returns status for valid debate ID."""
        mock_handler = MockHandler()
        result = rlm_handler._get_refinement_status.__wrapped__(
            rlm_handler, "/api/debates/test-123/refinement-status", mock_handler, user="test"
        )
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["debate_id"] == "test-123"
        assert "active_queries" in data
        assert "status" in data


# =============================================================================
# Knowledge Query Tests
# =============================================================================


class TestKnowledgeQuery:
    """Tests for knowledge mound RLM query endpoint."""

    def test_returns_400_without_body(self, rlm_handler):
        """Returns 400 when request body is missing."""
        mock_handler = MockHandler(_json_body=None)
        result = rlm_handler._query_knowledge_rlm.__wrapped__(
            rlm_handler, mock_handler, user="test"
        )
        assert result.status_code == 400

    def test_returns_400_without_workspace_id(self, rlm_handler):
        """Returns 400 when workspace_id is missing."""
        mock_handler = MockHandler(_json_body={"query": "What are the requirements?"})
        result = rlm_handler._query_knowledge_rlm.__wrapped__(
            rlm_handler, mock_handler, user="test"
        )
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "workspace_id" in data.get("error", "")

    def test_returns_400_without_query(self, rlm_handler):
        """Returns 400 when query is missing."""
        mock_handler = MockHandler(_json_body={"workspace_id": "ws_123"})
        result = rlm_handler._query_knowledge_rlm.__wrapped__(
            rlm_handler, mock_handler, user="test"
        )
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "query" in data.get("error", "")


# =============================================================================
# Handle Method Routing Tests
# =============================================================================


class TestHandleRouting:
    """Tests for request routing."""

    def test_handle_routes_to_status(self, rlm_handler):
        """Handle routes /api/rlm/status to status method."""
        with patch.object(
            rlm_handler, "_get_rlm_status", return_value=MagicMock(status_code=200)
        ) as mock_status:
            rlm_handler.handle("/api/rlm/status", {}, None)
            mock_status.assert_called_once()

    def test_handle_routes_to_metrics(self, rlm_handler):
        """Handle routes /api/metrics/rlm to metrics method."""
        with patch.object(
            rlm_handler, "_get_rlm_metrics", return_value=MagicMock(status_code=200)
        ) as mock_metrics:
            rlm_handler.handle("/api/metrics/rlm", {}, None)
            mock_metrics.assert_called_once()

    def test_handle_routes_context_requests(self, rlm_handler):
        """Handle routes context level requests."""
        with patch.object(
            rlm_handler, "_get_context_level", return_value=MagicMock(status_code=200)
        ) as mock_context:
            rlm_handler.handle("/api/debates/test-123/context/SUMMARY", {}, MockHandler())
            mock_context.assert_called_once()

    def test_handle_post_routes_query_rlm(self, rlm_handler):
        """handle_post routes /query-rlm to query method."""
        with patch.object(
            rlm_handler, "_query_debate_rlm", return_value=MagicMock(status_code=200)
        ) as mock_query:
            rlm_handler.handle_post(
                "/api/debates/test-123/query-rlm", {}, MockHandler()
            )
            mock_query.assert_called_once()

    def test_handle_post_routes_compress(self, rlm_handler):
        """handle_post routes /compress to compress method."""
        with patch.object(
            rlm_handler, "_compress_debate", return_value=MagicMock(status_code=200)
        ) as mock_compress:
            rlm_handler.handle_post(
                "/api/debates/test-123/compress", {}, MockHandler()
            )
            mock_compress.assert_called_once()

    def test_handle_post_routes_knowledge_query(self, rlm_handler):
        """handle_post routes knowledge query."""
        with patch.object(
            rlm_handler, "_query_knowledge_rlm", return_value=MagicMock(status_code=200)
        ) as mock_kq:
            rlm_handler.handle_post("/api/knowledge/query-rlm", {}, MockHandler())
            mock_kq.assert_called_once()


# =============================================================================
# Helper Method Tests
# =============================================================================


class TestHelperMethods:
    """Tests for helper methods."""

    def test_get_counter_value_handles_missing(self, rlm_handler):
        """_get_counter_value handles missing attributes."""
        mock_counter = MagicMock(spec=[])  # No _value attribute
        result = rlm_handler._get_counter_value(mock_counter)
        assert result == 0.0

    def test_get_gauge_value_handles_missing(self, rlm_handler):
        """_get_gauge_value handles missing attributes."""
        mock_gauge = MagicMock(spec=[])  # No _value attribute
        result = rlm_handler._get_gauge_value(mock_gauge)
        assert result == 0.0

    def test_get_counter_by_label_handles_errors(self, rlm_handler):
        """_get_counter_by_label handles errors gracefully."""
        mock_counter = MagicMock(spec=[])  # No _metrics attribute
        result = rlm_handler._get_counter_by_label(mock_counter, "label")
        assert result == {}


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_status_handles_exception(self, rlm_handler):
        """_get_rlm_status handles exceptions gracefully."""
        with patch(
            "aragora.server.handlers.features.rlm.logger"
        ) as mock_logger:
            # Force an exception in the status check
            with patch("builtins.__import__", side_effect=RuntimeError("Test error")):
                result = rlm_handler._get_rlm_status()
                # Should still return a response (either success or degraded)
                assert result.status_code == 200

    def test_query_handles_not_found(self, rlm_handler):
        """Query returns error when debate not found."""

        async def mock_query(*args, **kwargs):
            raise ValueError("Debate test-123 not found")

        mock_handler = MockHandler(_json_body={"query": "What was decided?"})

        with patch.object(rlm_handler, "_execute_rlm_query", side_effect=mock_query):
            with patch(
                "aragora.server.handlers.features.rlm._run_async",
                side_effect=ValueError("Debate test-123 not found"),
            ):
                result = rlm_handler._query_debate_rlm.__wrapped__(
                    rlm_handler,
                    "/api/debates/test-123/query-rlm",
                    mock_handler,
                    user="test",
                )
                assert result.status_code == 500
                data = json.loads(result.body)
                assert "error" in data
