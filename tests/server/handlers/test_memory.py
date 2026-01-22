"""
Tests for aragora.server.handlers.memory - Memory API handlers.

Tests cover:
- MemoryHandler routing
- Rate limiting for different endpoint types
- Authentication for mutating endpoints
- Continuum memory retrieval
- Tier statistics and pressure monitoring
- Memory search and critiques
- Error handling when components unavailable
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from io import BytesIO
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest


# ===========================================================================
# Test Fixtures
# ===========================================================================


class MockMemoryTier(Enum):
    """Mock memory tier enum."""

    FAST = "fast"
    MEDIUM = "medium"
    SLOW = "slow"
    GLACIAL = "glacial"


@dataclass
class MockMemory:
    """Mock memory entry."""

    id: str = "mem-123"
    tier: MockMemoryTier = field(default_factory=lambda: MockMemoryTier.FAST)
    content: str = "Test memory content"
    importance: float = 0.5
    surprise_score: float = 0.3
    consolidation_score: float = 0.2
    update_count: int = 1
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)


@dataclass
class MockCritique:
    """Mock critique entry."""

    agent: str = "gpt-4"
    target_agent: str = "claude"
    issues: list = field(default_factory=lambda: ["Issue 1"])
    suggestions: list = field(default_factory=lambda: ["Suggestion 1"])
    severity: str = "medium"


class MockContinuumMemory:
    """Mock continuum memory for testing."""

    def __init__(self):
        self.memories: list[MockMemory] = []
        self.consolidated = False
        self.cleaned = False

    def retrieve(
        self,
        query: str = "",
        tiers: list = None,
        limit: int = 10,
        min_importance: float = 0.0,
    ) -> list[MockMemory]:
        """Retrieve memories matching criteria."""
        result = self.memories.copy()
        if min_importance > 0:
            result = [m for m in result if m.importance >= min_importance]
        return result[:limit]

    def consolidate(self) -> dict:
        """Simulate consolidation."""
        self.consolidated = True
        return {"processed": 10, "promoted": 3, "consolidated": 2}

    def cleanup_expired_memories(
        self, tier=None, archive: bool = True, max_age_hours: float = None
    ) -> dict:
        """Simulate cleanup."""
        self.cleaned = True
        return {"expired_count": 5, "archived_count": 3}

    def enforce_tier_limits(self, tier=None, archive: bool = True) -> dict:
        """Simulate tier limit enforcement."""
        return {"evicted_count": 2}

    def get_stats(self) -> dict:
        """Get memory statistics."""
        return {
            "by_tier": {
                "FAST": {"count": 50, "avg_importance": 0.6, "avg_surprise": 0.4},
                "MEDIUM": {"count": 200, "avg_importance": 0.5, "avg_surprise": 0.3},
                "SLOW": {"count": 500, "avg_importance": 0.4, "avg_surprise": 0.2},
                "GLACIAL": {"count": 1000, "avg_importance": 0.3, "avg_surprise": 0.1},
            },
            "total_memories": 1750,
            "transitions": [{"from": "fast", "to": "medium", "count": 10}],
        }

    def get_archive_stats(self) -> dict:
        """Get archive statistics."""
        return {"archived_count": 100, "oldest_archive": "2024-01-01"}

    def get_memory_pressure(self) -> float:
        """Get current memory pressure."""
        return 0.65

    def delete(self, memory_id: str) -> bool:
        """Delete a memory."""
        for i, m in enumerate(self.memories):
            if m.id == memory_id:
                self.memories.pop(i)
                return True
        return False


class MockCritiqueStore:
    """Mock critique store for testing."""

    def __init__(self, nomic_dir: str):
        self.nomic_dir = nomic_dir
        self.critiques = [MockCritique()]

    def get_recent(self, limit: int = 100) -> list[MockCritique]:
        """Get recent critiques."""
        return self.critiques[:limit]


@dataclass
class MockAuthContext:
    """Mock authentication context."""

    is_authenticated: bool = True
    user_id: str = "user-123"
    email: str = "user@example.com"


def make_mock_handler(
    body: dict | None = None,
    method: str = "GET",
    headers: dict | None = None,
):
    """Create a mock HTTP handler."""
    handler = MagicMock()
    handler.command = method
    handler.headers = headers or {}
    handler.client_address = ("127.0.0.1", 12345)

    if body is not None:
        body_bytes = json.dumps(body).encode("utf-8")
        handler.headers["Content-Length"] = str(len(body_bytes))
        handler.rfile = BytesIO(body_bytes)
        handler.request_body = body_bytes
    else:
        handler.rfile = BytesIO(b"")
        handler.headers["Content-Length"] = "0"
        handler.request_body = b"{}"

    return handler


def get_status(result) -> int:
    """Extract status code from HandlerResult or tuple."""
    if hasattr(result, "status_code"):
        return result.status_code
    return result[1]


def get_body(result) -> dict:
    """Extract body from HandlerResult or tuple."""
    if hasattr(result, "body"):
        body = result.body
        if isinstance(body, bytes):
            return json.loads(body.decode("utf-8"))
        return json.loads(body)
    body = result[0]
    if isinstance(body, dict):
        return body
    return json.loads(body)


@pytest.fixture
def mock_continuum():
    """Create mock continuum memory."""
    continuum = MockContinuumMemory()
    continuum.memories = [
        MockMemory(id="mem-1", importance=0.8),
        MockMemory(id="mem-2", importance=0.5),
        MockMemory(id="mem-3", importance=0.3),
    ]
    return continuum


@pytest.fixture
def memory_handler(mock_continuum):
    """Create MemoryHandler with mock context."""
    from aragora.server.handlers.memory.memory import MemoryHandler

    ctx = {
        "continuum_memory": mock_continuum,
        "nomic_dir": "/tmp/test_nomic",
    }
    return MemoryHandler(ctx)


@pytest.fixture
def reset_rate_limiters():
    """Reset rate limiters before each test."""
    from aragora.server.handlers.memory import memory

    memory._retrieve_limiter._buckets.clear()
    memory._stats_limiter._buckets.clear()
    memory._mutation_limiter._buckets.clear()
    yield


# ===========================================================================
# Test Routing
# ===========================================================================


class TestMemoryHandlerRouting:
    """Tests for MemoryHandler routing."""

    def test_can_handle_memory_paths(self, memory_handler):
        """Test handler recognizes memory paths."""
        assert memory_handler.can_handle("/api/v1/memory/continuum/retrieve") is True
        assert memory_handler.can_handle("/api/v1/memory/tier-stats") is True
        assert memory_handler.can_handle("/api/v1/memory/pressure") is True
        assert memory_handler.can_handle("/api/v1/memory/tiers") is True
        assert memory_handler.can_handle("/api/v1/memory/search") is True
        assert memory_handler.can_handle("/api/v1/memory/critiques") is True

    def test_can_handle_delete_path(self, memory_handler):
        """Test handler recognizes memory delete paths."""
        assert memory_handler.can_handle("/api/v1/memory/continuum/mem-123") is True
        assert memory_handler.can_handle("/api/v1/memory/continuum/some-id-456") is True

    def test_excludes_known_routes_from_delete_pattern(self, memory_handler):
        """Test known routes are not treated as delete targets."""
        # These are actual routes, not delete targets
        assert memory_handler.can_handle("/api/v1/memory/continuum/retrieve") is True
        assert memory_handler.can_handle("/api/v1/memory/continuum/consolidate") is True
        assert memory_handler.can_handle("/api/v1/memory/continuum/cleanup") is True

    def test_cannot_handle_non_memory_paths(self, memory_handler):
        """Test handler rejects non-memory paths."""
        assert memory_handler.can_handle("/api/v1/debates") is False
        assert memory_handler.can_handle("/api/v1/admin/users") is False


# ===========================================================================
# Test Continuum Retrieve
# ===========================================================================


class TestContinuumRetrieve:
    """Tests for continuum memory retrieval."""

    def test_retrieve_success(self, memory_handler, reset_rate_limiters):
        """Test successful memory retrieval."""
        handler = make_mock_handler()

        result = memory_handler.handle("/api/v1/memory/continuum/retrieve", {}, handler)

        assert result is not None
        assert get_status(result) == 200
        data = get_body(result)
        assert "memories" in data
        assert "count" in data
        assert len(data["memories"]) == 3

    def test_retrieve_with_query(self, memory_handler, reset_rate_limiters):
        """Test retrieval with query parameter."""
        handler = make_mock_handler()

        result = memory_handler.handle(
            "/api/v1/memory/continuum/retrieve",
            {"query": "test search", "limit": "5"},
            handler,
        )

        assert result is not None
        assert get_status(result) == 200
        data = get_body(result)
        assert data["query"] == "test search"

    def test_retrieve_with_min_importance(self, memory_handler, reset_rate_limiters):
        """Test retrieval with minimum importance filter."""
        handler = make_mock_handler()

        result = memory_handler.handle(
            "/api/v1/memory/continuum/retrieve",
            {"min_importance": "0.6"},
            handler,
        )

        assert result is not None
        assert get_status(result) == 200
        data = get_body(result)
        # Only mem-1 has importance >= 0.6
        assert data["count"] == 1

    def test_retrieve_continuum_not_available(self, reset_rate_limiters):
        """Test retrieval when continuum is not available."""
        from aragora.server.handlers.memory.memory import MemoryHandler

        with patch("aragora.server.handlers.memory.memory.CONTINUUM_AVAILABLE", False):
            ctx = {}
            handler_obj = MemoryHandler(ctx)
            http_handler = make_mock_handler()

            result = handler_obj.handle("/api/v1/memory/continuum/retrieve", {}, http_handler)

            assert result is not None
            assert get_status(result) == 503

    def test_retrieve_continuum_not_initialized(self, reset_rate_limiters):
        """Test retrieval when continuum is not initialized."""
        from aragora.server.handlers.memory.memory import MemoryHandler

        ctx = {}  # No continuum_memory
        handler_obj = MemoryHandler(ctx)
        http_handler = make_mock_handler()

        result = handler_obj.handle("/api/v1/memory/continuum/retrieve", {}, http_handler)

        assert result is not None
        assert get_status(result) == 503


# ===========================================================================
# Test Tier Stats
# ===========================================================================


class TestTierStats:
    """Tests for tier statistics endpoints."""

    def test_get_tier_stats_success(self, memory_handler, reset_rate_limiters):
        """Test successful tier stats retrieval."""
        handler = make_mock_handler()

        result = memory_handler.handle("/api/v1/memory/tier-stats", {}, handler)

        assert result is not None
        assert get_status(result) == 200
        data = get_body(result)
        assert "tiers" in data
        assert "total_memories" in data
        assert data["total_memories"] == 1750

    def test_get_all_tiers_success(self, memory_handler, reset_rate_limiters):
        """Test comprehensive tier listing."""
        handler = make_mock_handler()

        result = memory_handler.handle("/api/v1/memory/tiers", {}, handler)

        assert result is not None
        assert get_status(result) == 200
        data = get_body(result)
        assert "tiers" in data
        assert len(data["tiers"]) == 4
        # Check tier metadata
        fast_tier = next(t for t in data["tiers"] if t["id"] == "fast")
        assert fast_tier["name"] == "Fast"
        assert fast_tier["ttl_seconds"] == 60


# ===========================================================================
# Test Memory Pressure
# ===========================================================================


class TestMemoryPressure:
    """Tests for memory pressure endpoint."""

    def test_get_memory_pressure_success(self, memory_handler, reset_rate_limiters):
        """Test memory pressure retrieval."""
        handler = make_mock_handler()

        result = memory_handler.handle("/api/v1/memory/pressure", {}, handler)

        assert result is not None
        assert get_status(result) == 200
        data = get_body(result)
        assert "pressure" in data
        assert "status" in data
        assert "tier_utilization" in data
        assert data["pressure"] == 0.65
        assert data["status"] == "elevated"


# ===========================================================================
# Test Memory Search
# ===========================================================================


class TestMemorySearch:
    """Tests for memory search endpoint."""

    def test_search_success(self, memory_handler, reset_rate_limiters):
        """Test successful memory search."""
        handler = make_mock_handler()

        result = memory_handler.handle("/api/v1/memory/search", {"q": "test query"}, handler)

        assert result is not None
        assert get_status(result) == 200
        data = get_body(result)
        assert "results" in data
        assert "query" in data
        assert data["query"] == "test query"

    def test_search_missing_query(self, memory_handler, reset_rate_limiters):
        """Test search with missing query parameter."""
        handler = make_mock_handler()

        result = memory_handler.handle("/api/v1/memory/search", {}, handler)

        assert result is not None
        assert get_status(result) == 400

    def test_search_with_filters(self, memory_handler, reset_rate_limiters):
        """Test search with various filters."""
        handler = make_mock_handler()

        result = memory_handler.handle(
            "/api/v1/memory/search",
            {
                "q": "test",
                "tier": "fast,medium",
                "min_importance": "0.5",
                "sort": "importance",
                "limit": "10",
            },
            handler,
        )

        assert result is not None
        assert get_status(result) == 200
        data = get_body(result)
        assert "tiers_searched" in data
        assert "filters" in data


# ===========================================================================
# Test Critiques
# ===========================================================================


class TestCritiques:
    """Tests for critique store endpoint."""

    def test_get_critiques_success(self, reset_rate_limiters):
        """Test successful critique retrieval."""
        from aragora.server.handlers.memory.memory import MemoryHandler

        ctx = {"nomic_dir": "/tmp/test_nomic"}
        handler_obj = MemoryHandler(ctx)
        http_handler = make_mock_handler()

        with (
            patch("aragora.server.handlers.memory.memory.CRITIQUE_STORE_AVAILABLE", True),
            patch("aragora.server.handlers.memory.memory.CritiqueStore", MockCritiqueStore),
        ):
            result = handler_obj.handle("/api/v1/memory/critiques", {}, http_handler)

            assert result is not None
            assert get_status(result) == 200
            data = get_body(result)
            assert "critiques" in data
            assert "count" in data

    def test_get_critiques_store_not_available(self, reset_rate_limiters):
        """Test critiques when store is not available."""
        from aragora.server.handlers.memory.memory import MemoryHandler

        with patch("aragora.server.handlers.memory.memory.CRITIQUE_STORE_AVAILABLE", False):
            ctx = {"nomic_dir": "/tmp/test"}
            handler_obj = MemoryHandler(ctx)
            http_handler = make_mock_handler()

            result = handler_obj.handle("/api/v1/memory/critiques", {}, http_handler)

            assert result is not None
            assert get_status(result) == 503

    def test_get_critiques_not_configured(self, reset_rate_limiters):
        """Test critiques when nomic_dir not configured."""
        from aragora.server.handlers.memory.memory import MemoryHandler

        ctx = {}  # No nomic_dir
        handler_obj = MemoryHandler(ctx)
        http_handler = make_mock_handler()

        result = handler_obj.handle("/api/v1/memory/critiques", {}, http_handler)

        assert result is not None
        assert get_status(result) == 503


# ===========================================================================
# Test POST Endpoints (Consolidate/Cleanup)
# ===========================================================================


class TestConsolidateEndpoint:
    """Tests for memory consolidation endpoint."""

    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_consolidate_success(self, mock_auth, memory_handler, reset_rate_limiters):
        """Test successful memory consolidation."""
        mock_auth.return_value = MockAuthContext()
        handler = make_mock_handler(method="POST")

        result = memory_handler.handle_post("/api/v1/memory/continuum/consolidate", {}, handler)

        assert result is not None
        assert get_status(result) == 200
        data = get_body(result)
        assert data["success"] is True
        assert "entries_processed" in data

    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_consolidate_unauthenticated(self, mock_auth, memory_handler, reset_rate_limiters):
        """Test consolidation requires authentication."""
        mock_auth.return_value = MockAuthContext(is_authenticated=False)
        handler = make_mock_handler(method="POST")

        result = memory_handler.handle_post("/api/v1/memory/continuum/consolidate", {}, handler)

        assert result is not None
        assert get_status(result) == 401

    def test_consolidate_get_not_allowed(self, memory_handler, reset_rate_limiters):
        """Test GET not allowed for consolidate endpoint."""
        handler = make_mock_handler(method="GET")

        result = memory_handler.handle("/api/v1/memory/continuum/consolidate", {}, handler)

        assert result is not None
        assert get_status(result) == 405


class TestCleanupEndpoint:
    """Tests for memory cleanup endpoint."""

    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_cleanup_success(self, mock_auth, memory_handler, reset_rate_limiters):
        """Test successful memory cleanup."""
        mock_auth.return_value = MockAuthContext()
        handler = make_mock_handler(method="POST")

        result = memory_handler.handle_post("/api/v1/memory/continuum/cleanup", {}, handler)

        assert result is not None
        assert get_status(result) == 200
        data = get_body(result)
        assert data["success"] is True
        assert "expired" in data

    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_cleanup_with_tier_filter(self, mock_auth, memory_handler, reset_rate_limiters):
        """Test cleanup with specific tier."""
        mock_auth.return_value = MockAuthContext()
        handler = make_mock_handler(method="POST")

        result = memory_handler.handle_post(
            "/api/v1/memory/continuum/cleanup", {"tier": "fast"}, handler
        )

        assert result is not None
        assert get_status(result) == 200

    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_cleanup_invalid_tier(self, mock_auth, memory_handler, reset_rate_limiters):
        """Test cleanup with invalid tier returns error."""
        mock_auth.return_value = MockAuthContext()
        handler = make_mock_handler(method="POST")

        result = memory_handler.handle_post(
            "/api/v1/memory/continuum/cleanup", {"tier": "invalid_tier"}, handler
        )

        assert result is not None
        assert get_status(result) == 400


# ===========================================================================
# Test DELETE Endpoint
# ===========================================================================


class TestDeleteMemory:
    """Tests for memory deletion endpoint."""

    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_delete_success(self, mock_auth, memory_handler, reset_rate_limiters):
        """Test successful memory deletion."""
        mock_auth.return_value = MockAuthContext()
        handler = make_mock_handler(method="DELETE")

        result = memory_handler.handle_delete("/api/v1/memory/continuum/mem-1", {}, handler)

        assert result is not None
        assert get_status(result) == 200
        data = get_body(result)
        assert data["success"] is True

    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_delete_not_found(self, mock_auth, memory_handler, reset_rate_limiters):
        """Test deletion of non-existent memory."""
        mock_auth.return_value = MockAuthContext()
        handler = make_mock_handler(method="DELETE")

        result = memory_handler.handle_delete(
            "/api/v1/memory/continuum/nonexistent-id", {}, handler
        )

        assert result is not None
        assert get_status(result) == 404

    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_delete_unauthenticated(self, mock_auth, memory_handler, reset_rate_limiters):
        """Test deletion requires authentication."""
        mock_auth.return_value = MockAuthContext(is_authenticated=False)
        handler = make_mock_handler(method="DELETE")

        result = memory_handler.handle_delete("/api/v1/memory/continuum/mem-1", {}, handler)

        assert result is not None
        assert get_status(result) == 401


# ===========================================================================
# Test Rate Limiting
# ===========================================================================


class TestRateLimiting:
    """Tests for rate limiting on memory endpoints."""

    def test_retrieve_rate_limit(self, memory_handler):
        """Test rate limiting on retrieve endpoint."""
        from aragora.server.handlers.memory import memory

        # Clear rate limiter
        memory._retrieve_limiter._buckets.clear()

        handler = make_mock_handler()

        # Make 60 requests (the limit)
        for _ in range(60):
            memory_handler.handle("/api/v1/memory/continuum/retrieve", {}, handler)

        # 61st request should be rate limited
        result = memory_handler.handle("/api/v1/memory/continuum/retrieve", {}, handler)

        assert result is not None
        assert get_status(result) == 429

    def test_stats_rate_limit(self, memory_handler):
        """Test rate limiting on stats endpoint."""
        from aragora.server.handlers.memory import memory

        # Clear rate limiter
        memory._stats_limiter._buckets.clear()

        handler = make_mock_handler()

        # Make 30 requests (the limit)
        for _ in range(30):
            memory_handler.handle("/api/v1/memory/tier-stats", {}, handler)

        # 31st request should be rate limited
        result = memory_handler.handle("/api/v1/memory/tier-stats", {}, handler)

        assert result is not None
        assert get_status(result) == 429


# ===========================================================================
# Test Archive Stats
# ===========================================================================


class TestArchiveStats:
    """Tests for archive statistics endpoint."""

    def test_get_archive_stats_success(self, memory_handler, reset_rate_limiters):
        """Test successful archive stats retrieval."""
        handler = make_mock_handler()

        result = memory_handler.handle("/api/v1/memory/archive-stats", {}, handler)

        assert result is not None
        assert get_status(result) == 200
        data = get_body(result)
        assert "archived_count" in data


# ===========================================================================
# Test TTL Formatting
# ===========================================================================


class TestTTLFormatting:
    """Tests for TTL human-readable formatting."""

    def test_format_ttl_seconds(self, memory_handler):
        """Test TTL formatting for seconds."""
        assert memory_handler._format_ttl(30) == "30s"

    def test_format_ttl_minutes(self, memory_handler):
        """Test TTL formatting for minutes."""
        assert memory_handler._format_ttl(120) == "2m"

    def test_format_ttl_hours(self, memory_handler):
        """Test TTL formatting for hours."""
        assert memory_handler._format_ttl(3600) == "1h"
        assert memory_handler._format_ttl(7200) == "2h"

    def test_format_ttl_days(self, memory_handler):
        """Test TTL formatting for days."""
        assert memory_handler._format_ttl(86400) == "1d"
        assert memory_handler._format_ttl(604800) == "7d"
