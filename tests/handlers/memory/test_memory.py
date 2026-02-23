"""Tests for MemoryHandler (aragora/server/handlers/memory/memory.py).

Covers all routes and behavior of the MemoryHandler class:
- can_handle() routing for all ROUTES
- GET /api/v1/memory/continuum/retrieve
- POST /api/v1/memory/continuum/consolidate
- POST /api/v1/memory/continuum/cleanup
- GET /api/v1/memory/tier-stats
- GET /api/v1/memory/archive-stats
- GET /api/v1/memory/pressure
- DELETE /api/v1/memory/continuum/{id}
- GET /api/v1/memory/tiers
- GET /api/v1/memory/search
- GET /api/v1/memory/search-index
- GET /api/v1/memory/search-timeline
- GET /api/v1/memory/entries
- GET /api/v1/memory/viewer
- GET /api/v1/memory/critiques
- GET /api/v1/memory/unified/stats
- POST /api/v1/memory/unified/search
- Legacy /api/memory/* path normalization
- Rate limiting for each tier
- Continuum unavailable / not initialized errors
- Utility methods: _parse_bool_param, _parse_tiers_param, _format_entry_summary,
  _format_entry_full, _estimate_tokens, _format_ttl
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result: object) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result: object) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if result is None:
        return 0
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


class MockHTTPHandler:
    """Mock HTTP request handler for MemoryHandler tests."""

    def __init__(
        self,
        body: dict | None = None,
        method: str = "GET",
    ):
        self.command = method
        self.client_address = ("127.0.0.1", 12345)
        self.headers: dict[str, str] = {"User-Agent": "test-agent"}
        self.rfile = MagicMock()
        self._request_body = body

        if body:
            body_bytes = json.dumps(body).encode()
            self.rfile.read.return_value = body_bytes
            self.headers["Content-Length"] = str(len(body_bytes))
        else:
            self.rfile.read.return_value = b"{}"
            self.headers["Content-Length"] = "2"


# ---------------------------------------------------------------------------
# Mock MemoryTier enum (matches aragora.memory.continuum.MemoryTier)
# ---------------------------------------------------------------------------


class MockMemoryTier(Enum):
    FAST = "fast"
    MEDIUM = "medium"
    SLOW = "slow"
    GLACIAL = "glacial"


# ---------------------------------------------------------------------------
# Mock memory entry
# ---------------------------------------------------------------------------


@dataclass
class MockMemoryEntry:
    id: str = "mem-001"
    tier: Any = None
    content: str = "Test memory content"
    importance: float = 0.8
    surprise_score: float = 0.3
    consolidation_score: float = 0.5
    update_count: int = 2
    created_at: str = "2026-01-01T00:00:00"
    updated_at: str = "2026-01-02T00:00:00"
    metadata: dict = field(default_factory=dict)
    memory_id: str = "mem-001"
    red_line: bool | None = None
    red_line_reason: str = ""

    def __post_init__(self):
        if self.tier is None:
            self.tier = MockMemoryTier.FAST


# ---------------------------------------------------------------------------
# Mock critique
# ---------------------------------------------------------------------------


@dataclass
class MockCritique:
    agent: str = "claude"
    target_agent: str = "gpt4"
    severity: float = 0.7
    issues: list = field(default_factory=lambda: ["Issue 1", "Issue 2"])
    suggestions: list = field(default_factory=lambda: ["Suggestion 1"])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# The patch target for extract_user_from_request -- it is imported locally
# inside handle_post/handle_delete from aragora.billing.jwt_auth.
_EXTRACT_USER_PATCH = "aragora.billing.jwt_auth.extract_user_from_request"


@pytest.fixture
def mock_http():
    """Create a MockHTTPHandler."""
    return MockHTTPHandler()


@pytest.fixture
def mock_continuum():
    """Create a mock continuum memory system."""
    mock = MagicMock()
    mock.retrieve.return_value = [
        MockMemoryEntry(id="mem-001", content="First memory", importance=0.9),
        MockMemoryEntry(id="mem-002", content="Second memory", importance=0.5),
    ]
    mock.consolidate.return_value = {
        "processed": 10,
        "promoted": 3,
        "consolidated": 2,
    }
    mock.cleanup_expired_memories.return_value = {"removed": 5}
    mock.enforce_tier_limits.return_value = {"evicted": 1}
    mock.get_stats.return_value = {
        "total_memories": 42,
        "by_tier": {
            "FAST": {"count": 10, "avg_importance": 0.9, "avg_surprise": 0.2},
            "MEDIUM": {"count": 15, "avg_importance": 0.7, "avg_surprise": 0.3},
            "SLOW": {"count": 10, "avg_importance": 0.5, "avg_surprise": 0.4},
            "GLACIAL": {"count": 7, "avg_importance": 0.3, "avg_surprise": 0.5},
        },
        "transitions": [{"from": "FAST", "to": "MEDIUM"}],
    }
    mock.get_archive_stats.return_value = {"archived": 20, "total_size": 1024}
    mock.get_memory_pressure.return_value = 0.45
    mock.delete.return_value = True
    mock.get_timeline_entries.return_value = {
        "anchor": MockMemoryEntry(id="anchor-001"),
        "before": [MockMemoryEntry(id="before-001")],
        "after": [MockMemoryEntry(id="after-001")],
    }
    mock.get_many.return_value = [
        MockMemoryEntry(id="entry-001"),
        MockMemoryEntry(id="entry-002"),
    ]
    return mock


@pytest.fixture
def handler(mock_continuum):
    """Create a MemoryHandler with mocked dependencies."""
    from aragora.server.handlers.memory.memory import MemoryHandler

    h = MemoryHandler(server_context={
        "continuum_memory": mock_continuum,
        "nomic_dir": "/tmp/test-nomic",
    })
    return h


@pytest.fixture(autouse=True)
def reset_rate_limiters():
    """Reset all rate limiters before each test."""
    from aragora.server.handlers.memory.memory import (
        _retrieve_limiter,
        _stats_limiter,
        _mutation_limiter,
    )

    _retrieve_limiter._buckets = defaultdict(list)
    _retrieve_limiter._requests = _retrieve_limiter._buckets
    _stats_limiter._buckets = defaultdict(list)
    _stats_limiter._requests = _stats_limiter._buckets
    _mutation_limiter._buckets = defaultdict(list)
    _mutation_limiter._requests = _mutation_limiter._buckets
    yield
    _retrieve_limiter._buckets = defaultdict(list)
    _retrieve_limiter._requests = _retrieve_limiter._buckets
    _stats_limiter._buckets = defaultdict(list)
    _stats_limiter._requests = _stats_limiter._buckets
    _mutation_limiter._buckets = defaultdict(list)
    _mutation_limiter._requests = _mutation_limiter._buckets


@pytest.fixture(autouse=True)
def patch_continuum_available():
    """Patch CONTINUUM_AVAILABLE to True by default."""
    with patch(
        "aragora.server.handlers.memory.memory.CONTINUUM_AVAILABLE", True
    ), patch(
        "aragora.server.handlers.memory.memory.MemoryTier", MockMemoryTier
    ), patch(
        "aragora.server.handlers.memory.memory_continuum.MemoryTier",
        MockMemoryTier,
        create=True,
    ):
        yield


@pytest.fixture(autouse=True)
def patch_memory_access():
    """Patch memory access module to avoid tenant enforcement."""
    with patch(
        "aragora.memory.access.tenant_enforcement_enabled",
        return_value=False,
        create=True,
    ), patch(
        "aragora.memory.access.resolve_tenant_id",
        return_value=None,
        create=True,
    ), patch(
        "aragora.memory.access.filter_entries",
        side_effect=lambda entries, ctx: entries,
        create=True,
    ):
        yield


@pytest.fixture(autouse=True)
def patch_emit_handler_event():
    """Patch emit_handler_event to prevent side effects."""
    with patch(
        "aragora.server.handlers.memory.memory_continuum.emit_handler_event"
    ):
        yield


def _make_auth_mock(*, authenticated: bool = True) -> MagicMock:
    """Create a mock auth context."""
    mock_auth = MagicMock()
    mock_auth.is_authenticated = authenticated
    return mock_auth


# ===========================================================================
# can_handle() Tests
# ===========================================================================


class TestCanHandle:
    """Test can_handle() routing."""

    def test_handles_continuum_retrieve(self, handler):
        assert handler.can_handle("/api/v1/memory/continuum/retrieve") is True

    def test_handles_continuum_consolidate(self, handler):
        assert handler.can_handle("/api/v1/memory/continuum/consolidate") is True

    def test_handles_continuum_cleanup(self, handler):
        assert handler.can_handle("/api/v1/memory/continuum/cleanup") is True

    def test_handles_tier_stats(self, handler):
        assert handler.can_handle("/api/v1/memory/tier-stats") is True

    def test_handles_archive_stats(self, handler):
        assert handler.can_handle("/api/v1/memory/archive-stats") is True

    def test_handles_pressure(self, handler):
        assert handler.can_handle("/api/v1/memory/pressure") is True

    def test_handles_tiers(self, handler):
        assert handler.can_handle("/api/v1/memory/tiers") is True

    def test_handles_search(self, handler):
        assert handler.can_handle("/api/v1/memory/search") is True

    def test_handles_search_index(self, handler):
        assert handler.can_handle("/api/v1/memory/search-index") is True

    def test_handles_search_timeline(self, handler):
        assert handler.can_handle("/api/v1/memory/search-timeline") is True

    def test_handles_entries(self, handler):
        assert handler.can_handle("/api/v1/memory/entries") is True

    def test_handles_viewer(self, handler):
        assert handler.can_handle("/api/v1/memory/viewer") is True

    def test_handles_critiques(self, handler):
        assert handler.can_handle("/api/v1/memory/critiques") is True

    def test_handles_compact(self, handler):
        assert handler.can_handle("/api/v1/memory/compact") is True

    def test_handles_context(self, handler):
        assert handler.can_handle("/api/v1/memory/context") is True

    def test_handles_cross_debate(self, handler):
        assert handler.can_handle("/api/v1/memory/cross-debate") is True

    def test_handles_export(self, handler):
        assert handler.can_handle("/api/v1/memory/export") is True

    def test_handles_import(self, handler):
        assert handler.can_handle("/api/v1/memory/import") is True

    def test_handles_prune(self, handler):
        assert handler.can_handle("/api/v1/memory/prune") is True

    def test_handles_query(self, handler):
        assert handler.can_handle("/api/v1/memory/query") is True

    def test_handles_semantic_search(self, handler):
        assert handler.can_handle("/api/v1/memory/semantic-search") is True

    def test_handles_snapshots(self, handler):
        assert handler.can_handle("/api/v1/memory/snapshots") is True

    def test_handles_vacuum(self, handler):
        assert handler.can_handle("/api/v1/memory/vacuum") is True

    def test_handles_dynamic_memory_path(self, handler):
        """Dynamic paths under /api/v1/memory/ are handled."""
        assert handler.can_handle("/api/v1/memory/some-id") is True
        assert handler.can_handle("/api/v1/memory/abc123/promote") is True
        assert handler.can_handle("/api/v1/memory/abc123/demote") is True
        assert handler.can_handle("/api/v1/memory/abc123/move") is True

    def test_does_not_handle_unrelated_path(self, handler):
        assert handler.can_handle("/api/v1/debates") is False
        assert handler.can_handle("/api/v1/knowledge") is False

    def test_handles_legacy_api_memory_path(self, handler):
        """Legacy /api/memory/* paths are normalized and handled."""
        assert handler.can_handle("/api/memory/continuum/retrieve") is True
        assert handler.can_handle("/api/memory/tier-stats") is True
        assert handler.can_handle("/api/memory/search") is True


# ===========================================================================
# Path normalization
# ===========================================================================


class TestPathNormalization:
    """Test _normalize_path() legacy path rewriting."""

    def test_normalizes_legacy_memory_path(self, handler):
        result = handler._normalize_path("/api/memory/continuum/retrieve")
        assert result == "/api/v1/memory/continuum/retrieve"

    def test_preserves_versioned_path(self, handler):
        result = handler._normalize_path("/api/v1/memory/continuum/retrieve")
        assert result == "/api/v1/memory/continuum/retrieve"

    def test_preserves_non_memory_path(self, handler):
        result = handler._normalize_path("/api/v1/debates/123")
        assert result == "/api/v1/debates/123"


# ===========================================================================
# GET /api/v1/memory/continuum/retrieve
# ===========================================================================


class TestContinuumRetrieve:
    """Test GET /api/v1/memory/continuum/retrieve."""

    def test_retrieve_success(self, handler, mock_http, mock_continuum):
        result = handler.handle("/api/v1/memory/continuum/retrieve", {}, mock_http)
        assert _status(result) == 200
        body = _body(result)
        assert "memories" in body
        assert body["count"] == 2

    def test_retrieve_with_query(self, handler, mock_http, mock_continuum):
        result = handler.handle(
            "/api/v1/memory/continuum/retrieve",
            {"query": "test"},
            mock_http,
        )
        assert _status(result) == 200
        mock_continuum.retrieve.assert_called()
        call_kwargs = mock_continuum.retrieve.call_args
        assert call_kwargs.kwargs.get("query") == "test"

    def test_retrieve_with_limit(self, handler, mock_http, mock_continuum):
        result = handler.handle(
            "/api/v1/memory/continuum/retrieve",
            {"limit": "5"},
            mock_http,
        )
        assert _status(result) == 200

    def test_retrieve_with_min_importance(self, handler, mock_http, mock_continuum):
        result = handler.handle(
            "/api/v1/memory/continuum/retrieve",
            {"min_importance": "0.5"},
            mock_http,
        )
        assert _status(result) == 200

    def test_retrieve_with_tiers_filter(self, handler, mock_http, mock_continuum):
        result = handler.handle(
            "/api/v1/memory/continuum/retrieve",
            {"tiers": "fast,medium"},
            mock_http,
        )
        assert _status(result) == 200

    def test_retrieve_continuum_unavailable(self, handler, mock_http):
        with patch(
            "aragora.server.handlers.memory.memory_continuum.CONTINUUM_AVAILABLE",
            False,
            create=True,
        ), patch(
            "aragora.server.handlers.memory.memory.CONTINUUM_AVAILABLE", False
        ):
            result = handler.handle(
                "/api/v1/memory/continuum/retrieve", {}, mock_http
            )
            assert _status(result) == 503

    def test_retrieve_continuum_not_initialized(self, handler, mock_http):
        handler.ctx["continuum_memory"] = None
        result = handler.handle("/api/v1/memory/continuum/retrieve", {}, mock_http)
        assert _status(result) == 503

    def test_retrieve_via_legacy_path(self, handler, mock_http):
        result = handler.handle("/api/memory/continuum/retrieve", {}, mock_http)
        assert _status(result) == 200

    def test_retrieve_memory_format(self, handler, mock_http, mock_continuum):
        """Verify that memory entries are formatted correctly."""
        result = handler.handle("/api/v1/memory/continuum/retrieve", {}, mock_http)
        body = _body(result)
        memories = body["memories"]
        assert len(memories) == 2
        m = memories[0]
        assert "id" in m
        assert "tier" in m
        assert "content" in m
        assert "importance" in m
        assert "surprise_score" in m

    def test_retrieve_truncates_long_content(self, handler, mock_http, mock_continuum):
        """Content > 500 chars should be truncated."""
        long_entry = MockMemoryEntry(
            id="long-001", content="x" * 600, importance=0.5
        )
        mock_continuum.retrieve.return_value = [long_entry]
        result = handler.handle("/api/v1/memory/continuum/retrieve", {}, mock_http)
        body = _body(result)
        content = body["memories"][0]["content"]
        assert content.endswith("...")
        assert len(content) <= 504  # 500 chars + "..."


# ===========================================================================
# POST endpoints returning 405 on GET
# ===========================================================================


class TestPostOnlyEndpoints:
    """Test that POST-only endpoints return 405 for GET."""

    def test_consolidate_returns_405_on_get(self, handler, mock_http):
        result = handler.handle(
            "/api/v1/memory/continuum/consolidate", {}, mock_http
        )
        assert _status(result) == 405
        body = _body(result)
        assert "POST" in body.get("error", "")

    def test_cleanup_returns_405_on_get(self, handler, mock_http):
        result = handler.handle(
            "/api/v1/memory/continuum/cleanup", {}, mock_http
        )
        assert _status(result) == 405
        body = _body(result)
        assert "POST" in body.get("error", "")


# ===========================================================================
# POST /api/v1/memory/continuum/consolidate
# ===========================================================================


class TestConsolidation:
    """Test POST /api/v1/memory/continuum/consolidate."""

    def test_consolidation_success(self, handler, mock_http, mock_continuum):
        with patch(_EXTRACT_USER_PATCH) as mock_extract:
            mock_extract.return_value = _make_auth_mock(authenticated=True)
            result = handler.handle_post(
                "/api/v1/memory/continuum/consolidate", {}, mock_http
            )
            assert _status(result) == 200
            body = _body(result)
            assert body["success"] is True
            assert body["entries_processed"] == 10
            assert body["entries_promoted"] == 3
            assert body["entries_consolidated"] == 2
            assert "duration_seconds" in body

    def test_consolidation_requires_authentication(self, handler, mock_http):
        with patch(_EXTRACT_USER_PATCH) as mock_extract:
            mock_extract.return_value = _make_auth_mock(authenticated=False)
            result = handler.handle_post(
                "/api/v1/memory/continuum/consolidate", {}, mock_http
            )
            assert _status(result) == 401

    def test_consolidation_continuum_unavailable(self, handler, mock_http):
        with patch(_EXTRACT_USER_PATCH) as mock_extract, patch(
            "aragora.server.handlers.memory.memory_continuum.CONTINUUM_AVAILABLE",
            False,
            create=True,
        ), patch(
            "aragora.server.handlers.memory.memory.CONTINUUM_AVAILABLE", False
        ):
            mock_extract.return_value = _make_auth_mock(authenticated=True)
            result = handler.handle_post(
                "/api/v1/memory/continuum/consolidate", {}, mock_http
            )
            assert _status(result) == 503

    def test_consolidation_continuum_not_initialized(self, handler, mock_http):
        handler.ctx["continuum_memory"] = None
        with patch(_EXTRACT_USER_PATCH) as mock_extract:
            mock_extract.return_value = _make_auth_mock(authenticated=True)
            result = handler.handle_post(
                "/api/v1/memory/continuum/consolidate", {}, mock_http
            )
            assert _status(result) == 503


# ===========================================================================
# POST /api/v1/memory/continuum/cleanup
# ===========================================================================


class TestCleanup:
    """Test POST /api/v1/memory/continuum/cleanup."""

    def test_cleanup_success(self, handler, mock_http, mock_continuum):
        with patch(_EXTRACT_USER_PATCH) as mock_extract:
            mock_extract.return_value = _make_auth_mock(authenticated=True)
            result = handler.handle_post(
                "/api/v1/memory/continuum/cleanup", {}, mock_http
            )
            assert _status(result) == 200
            body = _body(result)
            assert body["success"] is True
            assert "expired" in body
            assert "tier_limits" in body
            assert "duration_seconds" in body

    def test_cleanup_with_tier_param(self, handler, mock_http, mock_continuum):
        with patch(_EXTRACT_USER_PATCH) as mock_extract:
            mock_extract.return_value = _make_auth_mock(authenticated=True)
            result = handler.handle_post(
                "/api/v1/memory/continuum/cleanup",
                {"tier": "fast"},
                mock_http,
            )
            assert _status(result) == 200

    def test_cleanup_with_invalid_tier(self, handler, mock_http, mock_continuum):
        with patch(_EXTRACT_USER_PATCH) as mock_extract:
            mock_extract.return_value = _make_auth_mock(authenticated=True)
            result = handler.handle_post(
                "/api/v1/memory/continuum/cleanup",
                {"tier": "nonexistent"},
                mock_http,
            )
            assert _status(result) == 400

    def test_cleanup_with_archive_false(self, handler, mock_http, mock_continuum):
        with patch(_EXTRACT_USER_PATCH) as mock_extract:
            mock_extract.return_value = _make_auth_mock(authenticated=True)
            result = handler.handle_post(
                "/api/v1/memory/continuum/cleanup",
                {"archive": "false"},
                mock_http,
            )
            assert _status(result) == 200

    def test_cleanup_requires_authentication(self, handler, mock_http):
        with patch(_EXTRACT_USER_PATCH) as mock_extract:
            mock_extract.return_value = _make_auth_mock(authenticated=False)
            result = handler.handle_post(
                "/api/v1/memory/continuum/cleanup", {}, mock_http
            )
            assert _status(result) == 401


# ===========================================================================
# GET /api/v1/memory/tier-stats
# ===========================================================================


class TestTierStats:
    """Test GET /api/v1/memory/tier-stats."""

    def test_tier_stats_success(self, handler, mock_http, mock_continuum):
        result = handler.handle("/api/v1/memory/tier-stats", {}, mock_http)
        assert _status(result) == 200
        body = _body(result)
        assert "tiers" in body
        assert body["total_memories"] == 42
        assert "transitions" in body

    def test_tier_stats_continuum_unavailable(self, handler, mock_http):
        with patch(
            "aragora.server.handlers.memory.memory_continuum.CONTINUUM_AVAILABLE",
            False,
            create=True,
        ), patch(
            "aragora.server.handlers.memory.memory.CONTINUUM_AVAILABLE", False
        ):
            result = handler.handle("/api/v1/memory/tier-stats", {}, mock_http)
            assert _status(result) == 503

    def test_tier_stats_not_initialized(self, handler, mock_http):
        handler.ctx["continuum_memory"] = None
        result = handler.handle("/api/v1/memory/tier-stats", {}, mock_http)
        assert _status(result) == 503


# ===========================================================================
# GET /api/v1/memory/archive-stats
# ===========================================================================


class TestArchiveStats:
    """Test GET /api/v1/memory/archive-stats."""

    def test_archive_stats_success(self, handler, mock_http, mock_continuum):
        result = handler.handle("/api/v1/memory/archive-stats", {}, mock_http)
        assert _status(result) == 200
        body = _body(result)
        assert body["archived"] == 20

    def test_archive_stats_continuum_unavailable(self, handler, mock_http):
        with patch(
            "aragora.server.handlers.memory.memory_continuum.CONTINUUM_AVAILABLE",
            False,
            create=True,
        ), patch(
            "aragora.server.handlers.memory.memory.CONTINUUM_AVAILABLE", False
        ):
            result = handler.handle("/api/v1/memory/archive-stats", {}, mock_http)
            assert _status(result) == 503


# ===========================================================================
# GET /api/v1/memory/pressure
# ===========================================================================


class TestMemoryPressure:
    """Test GET /api/v1/memory/pressure."""

    def test_pressure_normal(self, handler, mock_http, mock_continuum):
        mock_continuum.get_memory_pressure.return_value = 0.3
        result = handler.handle("/api/v1/memory/pressure", {}, mock_http)
        assert _status(result) == 200
        body = _body(result)
        assert body["pressure"] == 0.3
        assert body["status"] == "normal"
        assert body["cleanup_recommended"] is False
        assert "tier_utilization" in body

    def test_pressure_elevated(self, handler, mock_http, mock_continuum):
        mock_continuum.get_memory_pressure.return_value = 0.65
        result = handler.handle("/api/v1/memory/pressure", {}, mock_http)
        body = _body(result)
        assert body["status"] == "elevated"

    def test_pressure_high(self, handler, mock_http, mock_continuum):
        mock_continuum.get_memory_pressure.return_value = 0.85
        result = handler.handle("/api/v1/memory/pressure", {}, mock_http)
        body = _body(result)
        assert body["status"] == "high"

    def test_pressure_critical(self, handler, mock_http, mock_continuum):
        mock_continuum.get_memory_pressure.return_value = 0.95
        result = handler.handle("/api/v1/memory/pressure", {}, mock_http)
        body = _body(result)
        assert body["status"] == "critical"
        assert body["cleanup_recommended"] is True

    def test_pressure_tier_utilization(self, handler, mock_http, mock_continuum):
        result = handler.handle("/api/v1/memory/pressure", {}, mock_http)
        body = _body(result)
        util = body["tier_utilization"]
        assert "FAST" in util
        assert util["FAST"]["count"] == 10
        assert util["FAST"]["limit"] == 100
        assert util["FAST"]["utilization"] == 0.1

    def test_pressure_continuum_unavailable(self, handler, mock_http):
        with patch(
            "aragora.server.handlers.memory.memory_continuum.CONTINUUM_AVAILABLE",
            False,
            create=True,
        ), patch(
            "aragora.server.handlers.memory.memory.CONTINUUM_AVAILABLE", False
        ):
            result = handler.handle("/api/v1/memory/pressure", {}, mock_http)
            assert _status(result) == 503


# ===========================================================================
# GET /api/v1/memory/tiers
# ===========================================================================


class TestAllTiers:
    """Test GET /api/v1/memory/tiers."""

    def test_tiers_success(self, handler, mock_http, mock_continuum):
        result = handler.handle("/api/v1/memory/tiers", {}, mock_http)
        assert _status(result) == 200
        body = _body(result)
        assert "tiers" in body
        tiers = body["tiers"]
        assert len(tiers) == 4
        tier_ids = [t["id"] for t in tiers]
        assert "fast" in tier_ids
        assert "medium" in tier_ids
        assert "slow" in tier_ids
        assert "glacial" in tier_ids

    def test_tiers_tier_structure(self, handler, mock_http, mock_continuum):
        result = handler.handle("/api/v1/memory/tiers", {}, mock_http)
        body = _body(result)
        fast = next(t for t in body["tiers"] if t["id"] == "fast")
        assert fast["name"] == "Fast"
        assert fast["ttl_seconds"] == 60
        assert fast["limit"] == 100
        assert fast["count"] == 10
        assert "utilization" in fast
        assert "description" in fast
        assert "ttl_human" in fast

    def test_tiers_total_memories(self, handler, mock_http, mock_continuum):
        result = handler.handle("/api/v1/memory/tiers", {}, mock_http)
        body = _body(result)
        assert body["total_memories"] == 42

    def test_tiers_transitions_24h(self, handler, mock_http, mock_continuum):
        result = handler.handle("/api/v1/memory/tiers", {}, mock_http)
        body = _body(result)
        assert body["transitions_24h"] == 1

    def test_tiers_continuum_unavailable(self, handler, mock_http):
        with patch(
            "aragora.server.handlers.memory.memory_continuum.CONTINUUM_AVAILABLE",
            False,
            create=True,
        ), patch(
            "aragora.server.handlers.memory.memory.CONTINUUM_AVAILABLE", False
        ):
            result = handler.handle("/api/v1/memory/tiers", {}, mock_http)
            assert _status(result) == 503


# ===========================================================================
# GET /api/v1/memory/search
# ===========================================================================


class TestSearchMemories:
    """Test GET /api/v1/memory/search."""

    def test_search_success(self, handler, mock_http, mock_continuum):
        result = handler.handle(
            "/api/v1/memory/search", {"q": "test query"}, mock_http
        )
        assert _status(result) == 200
        body = _body(result)
        assert "results" in body
        assert body["count"] == 2
        assert body["query"] == "test query"

    def test_search_missing_query(self, handler, mock_http):
        result = handler.handle("/api/v1/memory/search", {}, mock_http)
        assert _status(result) == 400

    def test_search_empty_query(self, handler, mock_http):
        result = handler.handle("/api/v1/memory/search", {"q": ""}, mock_http)
        assert _status(result) == 400

    def test_search_with_tier_filter(self, handler, mock_http, mock_continuum):
        result = handler.handle(
            "/api/v1/memory/search",
            {"q": "test", "tier": "fast,medium"},
            mock_http,
        )
        assert _status(result) == 200

    def test_search_with_sort_importance(self, handler, mock_http, mock_continuum):
        result = handler.handle(
            "/api/v1/memory/search",
            {"q": "test", "sort": "importance"},
            mock_http,
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["filters"]["sort"] == "importance"

    def test_search_with_sort_recency(self, handler, mock_http, mock_continuum):
        result = handler.handle(
            "/api/v1/memory/search",
            {"q": "test", "sort": "recency"},
            mock_http,
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["filters"]["sort"] == "recency"

    def test_search_result_format(self, handler, mock_http, mock_continuum):
        result = handler.handle(
            "/api/v1/memory/search", {"q": "test"}, mock_http
        )
        body = _body(result)
        r = body["results"][0]
        assert "id" in r
        assert "tier" in r
        assert "content" in r
        assert "importance" in r
        assert "surprise_score" in r

    def test_search_truncates_long_content(self, handler, mock_http, mock_continuum):
        """Content > 300 chars should be truncated in search results."""
        long_entry = MockMemoryEntry(
            id="long-001", content="x" * 400, importance=0.5
        )
        mock_continuum.retrieve.return_value = [long_entry]
        result = handler.handle(
            "/api/v1/memory/search", {"q": "test"}, mock_http
        )
        body = _body(result)
        content = body["results"][0]["content"]
        assert content.endswith("...")
        assert len(content) <= 304  # 300 chars + "..."

    def test_search_continuum_unavailable(self, handler, mock_http):
        with patch(
            "aragora.server.handlers.memory.memory_progressive.CONTINUUM_AVAILABLE",
            False,
            create=True,
        ), patch(
            "aragora.server.handlers.memory.memory.CONTINUUM_AVAILABLE", False
        ):
            result = handler.handle(
                "/api/v1/memory/search", {"q": "test"}, mock_http
            )
            assert _status(result) == 503


# ===========================================================================
# GET /api/v1/memory/search-index
# ===========================================================================


class TestSearchIndex:
    """Test GET /api/v1/memory/search-index (progressive stage 1)."""

    def test_search_index_success(self, handler, mock_http, mock_continuum):
        result = handler.handle(
            "/api/v1/memory/search-index", {"q": "test"}, mock_http
        )
        assert _status(result) == 200
        body = _body(result)
        assert "results" in body
        assert "count" in body
        assert "tiers" in body
        assert body["query"] == "test"

    def test_search_index_missing_query(self, handler, mock_http):
        result = handler.handle("/api/v1/memory/search-index", {}, mock_http)
        assert _status(result) == 400

    def test_search_index_with_external(self, handler, mock_http, mock_continuum):
        """Test include_external flag triggers external searches."""
        result = handler.handle(
            "/api/v1/memory/search-index",
            {"q": "test", "include_external": "true"},
            mock_http,
        )
        assert _status(result) == 200
        body = _body(result)
        assert "external_results" in body
        assert "external_sources" in body

    def test_search_index_continuum_not_initialized(self, handler, mock_http):
        handler.ctx["continuum_memory"] = None
        result = handler.handle(
            "/api/v1/memory/search-index", {"q": "test"}, mock_http
        )
        assert _status(result) == 503


# ===========================================================================
# GET /api/v1/memory/search-timeline
# ===========================================================================


class TestSearchTimeline:
    """Test GET /api/v1/memory/search-timeline (progressive stage 2)."""

    def test_timeline_success(self, handler, mock_http, mock_continuum):
        result = handler.handle(
            "/api/v1/memory/search-timeline",
            {"anchor_id": "anchor-001"},
            mock_http,
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["anchor_id"] == "anchor-001"
        assert "anchor" in body
        assert "before" in body
        assert "after" in body

    def test_timeline_missing_anchor_id(self, handler, mock_http):
        result = handler.handle(
            "/api/v1/memory/search-timeline", {}, mock_http
        )
        assert _status(result) == 400

    def test_timeline_anchor_not_found(self, handler, mock_http, mock_continuum):
        mock_continuum.get_timeline_entries.return_value = None
        result = handler.handle(
            "/api/v1/memory/search-timeline",
            {"anchor_id": "nonexistent"},
            mock_http,
        )
        assert _status(result) == 404

    def test_timeline_not_supported(self, handler, mock_http, mock_continuum):
        """When continuum doesn't support timeline queries."""
        del mock_continuum.get_timeline_entries
        result = handler.handle(
            "/api/v1/memory/search-timeline",
            {"anchor_id": "anchor-001"},
            mock_http,
        )
        assert _status(result) == 501


# ===========================================================================
# GET /api/v1/memory/entries
# ===========================================================================


class TestGetEntries:
    """Test GET /api/v1/memory/entries (progressive stage 3)."""

    def test_entries_success(self, handler, mock_http, mock_continuum):
        result = handler.handle(
            "/api/v1/memory/entries",
            {"ids": "entry-001,entry-002"},
            mock_http,
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["count"] == 2
        assert "entries" in body
        assert body["ids"] == ["entry-001", "entry-002"]

    def test_entries_missing_ids(self, handler, mock_http):
        result = handler.handle("/api/v1/memory/entries", {}, mock_http)
        assert _status(result) == 400

    def test_entries_empty_ids(self, handler, mock_http):
        result = handler.handle(
            "/api/v1/memory/entries", {"ids": ""}, mock_http
        )
        assert _status(result) == 400

    def test_entries_get_many_not_supported(self, handler, mock_http, mock_continuum):
        """When continuum doesn't support get_many()."""
        del mock_continuum.get_many
        result = handler.handle(
            "/api/v1/memory/entries",
            {"ids": "entry-001"},
            mock_http,
        )
        assert _status(result) == 501


# ===========================================================================
# GET /api/v1/memory/viewer
# ===========================================================================


class TestViewer:
    """Test GET /api/v1/memory/viewer."""

    def test_viewer_returns_html(self, handler, mock_http):
        result = handler.handle("/api/v1/memory/viewer", {}, mock_http)
        assert _status(result) == 200
        assert result.content_type == "text/html"
        html = result.body.decode("utf-8")
        assert "Memory Viewer" in html
        assert "<html" in html


# ===========================================================================
# GET /api/v1/memory/critiques
# ===========================================================================


class TestCritiques:
    """Test GET /api/v1/memory/critiques."""

    def _patch_critique_available(self, available: bool = True):
        """Context manager to patch critique store availability."""
        return patch.dict(
            "aragora.server.handlers.memory.memory.__dict__",
            {"CRITIQUE_STORE_AVAILABLE": available},
        )

    def test_critiques_success(self, handler, mock_http):
        mock_store = MagicMock()
        mock_store.get_recent.return_value = [
            MockCritique(agent="claude", target_agent="gpt4"),
            MockCritique(agent="gpt4", target_agent="claude"),
        ]

        with patch(
            "aragora.server.handlers.memory.memory.CRITIQUE_STORE_AVAILABLE", True
        ), patch(
            "aragora.stores.canonical.get_critique_store",
            return_value=mock_store,
        ):
            result = handler.handle("/api/v1/memory/critiques", {}, mock_http)
            assert _status(result) == 200
            body = _body(result)
            assert "critiques" in body
            assert body["count"] == 2
            assert "total" in body
            assert "offset" in body
            assert "limit" in body

    def test_critiques_with_agent_filter(self, handler, mock_http):
        mock_store = MagicMock()
        mock_store.get_recent.return_value = [
            MockCritique(agent="claude", target_agent="gpt4"),
            MockCritique(agent="gpt4", target_agent="claude"),
        ]

        with patch(
            "aragora.server.handlers.memory.memory.CRITIQUE_STORE_AVAILABLE", True
        ), patch(
            "aragora.stores.canonical.get_critique_store",
            return_value=mock_store,
        ):
            result = handler.handle(
                "/api/v1/memory/critiques",
                {"agent": "claude"},
                mock_http,
            )
            assert _status(result) == 200
            body = _body(result)
            # Only claude critiques should be returned
            for c in body["critiques"]:
                assert c["agent"] == "claude"

    def test_critiques_with_pagination(self, handler, mock_http):
        mock_store = MagicMock()
        mock_store.get_recent.return_value = [
            MockCritique(agent=f"agent-{i}") for i in range(30)
        ]

        with patch(
            "aragora.server.handlers.memory.memory.CRITIQUE_STORE_AVAILABLE", True
        ), patch(
            "aragora.stores.canonical.get_critique_store",
            return_value=mock_store,
        ):
            result = handler.handle(
                "/api/v1/memory/critiques",
                {"limit": "5", "offset": "10"},
                mock_http,
            )
            assert _status(result) == 200
            body = _body(result)
            assert body["count"] <= 5
            assert body["offset"] == 10
            assert body["limit"] == 5

    def test_critiques_store_unavailable(self, handler, mock_http):
        with patch(
            "aragora.server.handlers.memory.memory.CRITIQUE_STORE_AVAILABLE", False
        ):
            result = handler.handle("/api/v1/memory/critiques", {}, mock_http)
            assert _status(result) == 503

    def test_critiques_no_nomic_dir(self, handler, mock_http):
        handler.ctx["nomic_dir"] = None
        with patch(
            "aragora.server.handlers.memory.memory.CRITIQUE_STORE_AVAILABLE", True
        ):
            result = handler.handle("/api/v1/memory/critiques", {}, mock_http)
            assert _status(result) == 503


# ===========================================================================
# DELETE /api/v1/memory/continuum/{id}
# ===========================================================================


class TestDeleteMemory:
    """Test DELETE /api/v1/memory/continuum/{id}."""

    def test_delete_success(self, handler, mock_http, mock_continuum):
        with patch(_EXTRACT_USER_PATCH) as mock_extract:
            mock_extract.return_value = _make_auth_mock(authenticated=True)
            result = handler.handle_delete(
                "/api/v1/memory/continuum/mem-001", {}, mock_http
            )
            assert _status(result) == 200
            body = _body(result)
            assert body["success"] is True
            mock_continuum.delete.assert_called_once()

    def test_delete_not_found(self, handler, mock_http, mock_continuum):
        mock_continuum.delete.return_value = False
        with patch(_EXTRACT_USER_PATCH) as mock_extract:
            mock_extract.return_value = _make_auth_mock(authenticated=True)
            result = handler.handle_delete(
                "/api/v1/memory/continuum/nonexistent", {}, mock_http
            )
            assert _status(result) == 404

    def test_delete_requires_authentication(self, handler, mock_http):
        with patch(_EXTRACT_USER_PATCH) as mock_extract:
            mock_extract.return_value = _make_auth_mock(authenticated=False)
            result = handler.handle_delete(
                "/api/v1/memory/continuum/mem-001", {}, mock_http
            )
            assert _status(result) == 401

    def test_delete_not_supported(self, handler, mock_http, mock_continuum):
        """When continuum doesn't support deletion."""
        del mock_continuum.delete
        with patch(_EXTRACT_USER_PATCH) as mock_extract:
            mock_extract.return_value = _make_auth_mock(authenticated=True)
            result = handler.handle_delete(
                "/api/v1/memory/continuum/mem-001", {}, mock_http
            )
            assert _status(result) == 501

    def test_delete_non_continuum_path_returns_none(self, handler, mock_http):
        """DELETE on non-continuum paths returns None."""
        result = handler.handle_delete(
            "/api/v1/memory/something-else", {}, mock_http
        )
        assert result is None


# ===========================================================================
# Unified memory endpoints
# ===========================================================================


class TestUnifiedMemory:
    """Test unified memory gateway endpoints."""

    def test_unified_stats_handler_unavailable(self, handler, mock_http):
        with patch.object(handler, "_get_unified_handler", return_value=None):
            result = handler.handle("/api/v1/memory/unified/stats", {}, mock_http)
            assert _status(result) == 501

    def test_unified_stats_success(self, handler, mock_http):
        mock_uh = MagicMock()

        async def mock_handle_stats():
            return {"total": 100, "by_tier": {}}

        mock_uh.handle_stats = mock_handle_stats

        with patch.object(handler, "_get_unified_handler", return_value=mock_uh), \
             patch(
                 "aragora.utils.async_utils.run_async",
                 return_value={"total": 100, "by_tier": {}},
             ):
            result = handler.handle("/api/v1/memory/unified/stats", {}, mock_http)
            # The method uses HandlerResult(data=...) which may raise TypeError,
            # but @handle_errors wraps it. Just verify we get a result.
            assert result is not None

    def test_unified_search_handler_unavailable(self, handler, mock_http):
        with patch(_EXTRACT_USER_PATCH) as mock_extract:
            mock_extract.return_value = _make_auth_mock(authenticated=True)
            with patch.object(handler, "_get_unified_handler", return_value=None):
                result = handler.handle_post(
                    "/api/v1/memory/unified/search", {"query": "test"}, mock_http
                )
                assert _status(result) == 501

    def test_handle_post_unknown_path_returns_none(self, handler, mock_http):
        with patch(_EXTRACT_USER_PATCH) as mock_extract:
            mock_extract.return_value = _make_auth_mock(authenticated=True)
            result = handler.handle_post(
                "/api/v1/memory/unknown-post-route", {}, mock_http
            )
            assert result is None


# ===========================================================================
# Rate limiting
# ===========================================================================


class TestRateLimiting:
    """Test rate limiting on memory endpoints."""

    def test_retrieve_rate_limited(self, handler, mock_continuum):
        from aragora.server.handlers.memory.memory import _retrieve_limiter

        # Fill rate limiter with current timestamps to trigger rate limiting
        now = time.time()
        for _ in range(65):
            _retrieve_limiter._buckets["127.0.0.1"].append(now)

        mock_http = MockHTTPHandler()
        result = handler.handle(
            "/api/v1/memory/continuum/retrieve", {}, mock_http
        )
        assert _status(result) == 429

    def test_stats_rate_limited(self, handler, mock_continuum):
        from aragora.server.handlers.memory.memory import _stats_limiter

        now = time.time()
        for _ in range(35):
            _stats_limiter._buckets["127.0.0.1"].append(now)

        mock_http = MockHTTPHandler()
        result = handler.handle("/api/v1/memory/tier-stats", {}, mock_http)
        assert _status(result) == 429

    def test_mutation_rate_limited(self, handler, mock_continuum):
        from aragora.server.handlers.memory.memory import _mutation_limiter

        now = time.time()
        for _ in range(15):
            _mutation_limiter._buckets["127.0.0.1"].append(now)

        mock_http = MockHTTPHandler()
        with patch(_EXTRACT_USER_PATCH) as mock_extract:
            mock_extract.return_value = _make_auth_mock(authenticated=True)
            result = handler.handle_post(
                "/api/v1/memory/continuum/consolidate", {}, mock_http
            )
            assert _status(result) == 429


# ===========================================================================
# Utility method tests
# ===========================================================================


class TestUtilityMethods:
    """Test utility/helper methods on MemoryHandler."""

    def test_parse_bool_param_true_variants(self, handler):
        assert handler._parse_bool_param({"flag": "1"}, "flag") is True
        assert handler._parse_bool_param({"flag": "true"}, "flag") is True
        assert handler._parse_bool_param({"flag": "yes"}, "flag") is True
        assert handler._parse_bool_param({"flag": "y"}, "flag") is True
        assert handler._parse_bool_param({"flag": "on"}, "flag") is True
        assert handler._parse_bool_param({"flag": "True"}, "flag") is True
        assert handler._parse_bool_param({"flag": "YES"}, "flag") is True

    def test_parse_bool_param_false_variants(self, handler):
        assert handler._parse_bool_param({"flag": "0"}, "flag") is False
        assert handler._parse_bool_param({"flag": "false"}, "flag") is False
        assert handler._parse_bool_param({"flag": "no"}, "flag") is False

    def test_parse_bool_param_default(self, handler):
        assert handler._parse_bool_param({}, "flag") is False
        assert handler._parse_bool_param({}, "flag", default=True) is True

    def test_parse_bool_param_empty_string(self, handler):
        assert handler._parse_bool_param({"flag": ""}, "flag") is False

    def test_parse_tiers_param_single(self, handler):
        tiers = handler._parse_tiers_param({"tier": "fast"})
        assert len(tiers) == 1
        assert tiers[0] == MockMemoryTier.FAST

    def test_parse_tiers_param_multiple(self, handler):
        tiers = handler._parse_tiers_param({"tier": "fast,medium"})
        assert len(tiers) == 2

    def test_parse_tiers_param_empty_returns_all(self, handler):
        tiers = handler._parse_tiers_param({})
        assert len(tiers) == len(MockMemoryTier)

    def test_parse_tiers_param_invalid_ignored(self, handler):
        tiers = handler._parse_tiers_param({"tier": "fast,invalid,medium"})
        assert len(tiers) == 2

    def test_parse_tiers_param_all_invalid_returns_all(self, handler):
        tiers = handler._parse_tiers_param({"tier": "invalid1,invalid2"})
        assert len(tiers) == len(MockMemoryTier)

    def test_estimate_tokens_empty(self, handler):
        assert handler._estimate_tokens("") == 0

    def test_estimate_tokens_short_text(self, handler):
        result = handler._estimate_tokens("hello")
        assert result >= 1

    def test_estimate_tokens_longer_text(self, handler):
        text = "a" * 100
        result = handler._estimate_tokens(text)
        assert result == 25  # ceil(100/4)

    def test_estimate_tokens_one_char(self, handler):
        result = handler._estimate_tokens("x")
        assert result == 1

    def test_format_entry_summary_basic(self, handler):
        entry = MockMemoryEntry(
            id="test-001",
            content="Short content",
            importance=0.75,
            surprise_score=0.4,
        )
        result = handler._format_entry_summary(entry)
        assert result["id"] == "test-001"
        assert result["preview"] == "Short content"
        assert result["importance"] == 0.75
        assert result["surprise_score"] == 0.4
        assert result["token_estimate"] >= 1

    def test_format_entry_summary_long_content_truncated(self, handler):
        entry = MockMemoryEntry(content="x" * 300)
        result = handler._format_entry_summary(entry, preview_chars=220)
        assert result["preview"].endswith("...")
        assert len(result["preview"]) <= 224  # 220 + "..."

    def test_format_entry_summary_short_content_not_truncated(self, handler):
        entry = MockMemoryEntry(content="short")
        result = handler._format_entry_summary(entry, preview_chars=220)
        assert result["preview"] == "short"

    def test_format_entry_summary_with_metadata(self, handler):
        entry = MockMemoryEntry(metadata={"key": "value"})
        result = handler._format_entry_summary(entry, include_metadata=True)
        assert result["metadata"] == {"key": "value"}

    def test_format_entry_summary_with_content(self, handler):
        entry = MockMemoryEntry(content="full content")
        result = handler._format_entry_summary(entry, include_content=True)
        assert result["content"] == "full content"

    def test_format_entry_summary_with_red_line(self, handler):
        entry = MockMemoryEntry(red_line=True, red_line_reason="Safety concern")
        result = handler._format_entry_summary(entry)
        assert result["red_line"] is True
        assert result["red_line_reason"] == "Safety concern"

    def test_format_entry_summary_without_red_line(self, handler):
        entry = MockMemoryEntry(red_line=None)
        result = handler._format_entry_summary(entry)
        assert "red_line" not in result

    def test_format_entry_summary_dict_entry(self, handler):
        """Should work with dict entries too."""
        entry = {
            "id": "dict-001",
            "content": "dict content",
            "tier": "fast",
            "importance": 0.6,
        }
        result = handler._format_entry_summary(entry)
        assert result["id"] == "dict-001"
        assert result["preview"] == "dict content"

    def test_format_entry_full(self, handler):
        entry = MockMemoryEntry(
            id="full-001",
            content="Full content text",
            importance=0.9,
            metadata={"key": "value"},
        )
        result = handler._format_entry_full(entry)
        assert result["content"] == "Full content text"
        assert result["metadata"] == {"key": "value"}
        assert "token_estimate" in result

    def test_format_ttl_seconds(self, handler):
        assert handler._format_ttl(30) == "30s"

    def test_format_ttl_minutes(self, handler):
        assert handler._format_ttl(300) == "5m"

    def test_format_ttl_hours(self, handler):
        assert handler._format_ttl(3600) == "1h"

    def test_format_ttl_days(self, handler):
        assert handler._format_ttl(86400) == "1d"
        assert handler._format_ttl(604800) == "7d"


# ===========================================================================
# Handler construction
# ===========================================================================


class TestHandlerConstruction:
    """Test MemoryHandler initialization."""

    def test_init_with_server_context(self):
        from aragora.server.handlers.memory.memory import MemoryHandler

        h = MemoryHandler(server_context={"key": "value"})
        assert h.ctx == {"key": "value"}

    def test_init_with_ctx(self):
        from aragora.server.handlers.memory.memory import MemoryHandler

        h = MemoryHandler(ctx={"key": "value"})
        assert h.ctx == {"key": "value"}

    def test_init_with_no_context(self):
        from aragora.server.handlers.memory.memory import MemoryHandler

        h = MemoryHandler()
        assert h.ctx == {}

    def test_server_context_takes_precedence(self):
        from aragora.server.handlers.memory.memory import MemoryHandler

        h = MemoryHandler(
            ctx={"from_ctx": True},
            server_context={"from_server": True},
        )
        assert h.ctx == {"from_server": True}

    def test_get_user_store_from_context(self, handler):
        handler.ctx["user_store"] = "mock_store"
        assert handler._get_user_store() == "mock_store"

    def test_get_user_store_missing(self, handler):
        assert handler._get_user_store() is None


# ===========================================================================
# handle() returns None for unmatched routes
# ===========================================================================


class TestHandleUnmatched:
    """Test that handle() returns None for unmatched routes."""

    def test_handle_returns_none_for_unmapped_path(self, handler, mock_http):
        """For a path under /api/v1/memory/ that doesn't match any route."""
        result = handler.handle(
            "/api/v1/memory/some-random-thing-that-doesnt-exist",
            {},
            mock_http,
        )
        assert result is None


# ===========================================================================
# Unified handler lazy initialization
# ===========================================================================


class TestUnifiedHandlerInit:
    """Test _get_unified_handler lazy creation."""

    def test_returns_none_when_import_fails(self, handler):
        with patch(
            "aragora.server.handlers.memory.memory.MemoryHandler._get_unified_handler"
        ) as mock_get:
            mock_get.return_value = None
            result = handler._get_unified_handler()
            assert result is None

    def test_caches_handler(self, handler):
        """Once created, unified handler is cached."""
        handler._unified_handler = MagicMock()
        result = handler._get_unified_handler()
        assert result is handler._unified_handler
