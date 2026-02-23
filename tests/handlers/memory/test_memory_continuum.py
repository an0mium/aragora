"""Tests for MemoryContinuumMixin handler methods.

Tests cover all methods in aragora/server/handlers/memory/memory_continuum.py:
- _get_continuum_memories: Retrieve memories with tier/query/importance filtering
- _trigger_consolidation: Trigger memory consolidation
- _trigger_cleanup: Cleanup expired memories with tier/archive/max_age params
- _get_tier_stats: Get per-tier statistics
- _get_archive_stats: Get archive statistics
- _get_memory_pressure: Get pressure status and tier utilization
- _delete_memory: Delete a memory by ID with tenant isolation
- _get_all_tiers: Get comprehensive tier information
- _format_ttl: Format TTL in human-readable form
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Mock MemoryTier used in the handler (mirrors aragora.memory.tier_manager)
# ---------------------------------------------------------------------------


class _MockMemoryTier(str, Enum):
    FAST = "fast"
    MEDIUM = "medium"
    SLOW = "slow"
    GLACIAL = "glacial"


# ---------------------------------------------------------------------------
# Helper to build a mock memory entry
# ---------------------------------------------------------------------------


def _make_memory(
    *,
    id: str = "mem-001",
    tier: _MockMemoryTier = _MockMemoryTier.FAST,
    content: str = "test content",
    importance: float = 0.5,
    surprise_score: float = 0.2,
    consolidation_score: float = 0.3,
    update_count: int = 1,
    created_at: str = "2026-01-01T00:00:00",
    updated_at: str = "2026-01-02T00:00:00",
) -> MagicMock:
    m = MagicMock()
    m.id = id
    m.tier = tier
    m.content = content
    m.importance = importance
    m.surprise_score = surprise_score
    m.consolidation_score = consolidation_score
    m.update_count = update_count
    m.created_at = created_at
    m.updated_at = updated_at
    return m


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_continuum():
    """Create a mock continuum memory backend."""
    cm = MagicMock()
    cm.retrieve.return_value = []
    cm.consolidate.return_value = {"processed": 10, "promoted": 3, "consolidated": 5}
    cm.cleanup_expired_memories.return_value = {"removed": 2, "archived": 1}
    cm.enforce_tier_limits.return_value = {"evicted": 0}
    cm.get_stats.return_value = {
        "by_tier": {
            "FAST": {"count": 20, "avg_importance": 0.6, "avg_surprise": 0.3},
            "MEDIUM": {"count": 50, "avg_importance": 0.7, "avg_surprise": 0.4},
            "SLOW": {"count": 100, "avg_importance": 0.8, "avg_surprise": 0.5},
            "GLACIAL": {"count": 200, "avg_importance": 0.9, "avg_surprise": 0.6},
        },
        "total_memories": 370,
        "transitions": [{"from": "fast", "to": "medium"}],
    }
    cm.get_archive_stats.return_value = {"archived": 15, "total_size_bytes": 4096}
    cm.get_memory_pressure.return_value = 0.45
    cm.delete.return_value = True
    return cm


@pytest.fixture
def handler(mock_continuum):
    """Create a MemoryHandler backed by mock continuum."""
    with (
        patch("aragora.server.handlers.memory.memory.CONTINUUM_AVAILABLE", True),
        patch("aragora.server.handlers.memory.memory.MemoryTier", _MockMemoryTier),
        patch(
            "aragora.server.handlers.memory.memory_continuum.MemoryContinuumMixin.__init_subclass__",
            lambda **kw: None,
            create=True,
        ),
    ):
        from aragora.server.handlers.memory.memory import MemoryHandler

        h = MemoryHandler(server_context={"continuum_memory": mock_continuum})
        h._auth_context = None
        return h


@pytest.fixture
def mock_http_handler():
    """Create a mock HTTP handler."""
    mock = MagicMock()
    mock.client_address = ("127.0.0.1", 12345)
    mock.headers = {}
    mock._auth_context = None
    return mock


@pytest.fixture(autouse=True)
def _patch_continuum_available():
    """Ensure CONTINUUM_AVAILABLE is True in the mixin module."""
    with (
        patch("aragora.server.handlers.memory.memory.CONTINUUM_AVAILABLE", True),
        patch("aragora.server.handlers.memory.memory.MemoryTier", _MockMemoryTier),
    ):
        yield


@pytest.fixture(autouse=True)
def _reset_rate_limiters():
    """Reset module-level rate limiters between tests."""
    try:
        from aragora.server.handlers.memory import memory as mem_mod

        for attr in ("_retrieve_limiter", "_stats_limiter", "_mutation_limiter"):
            limiter = getattr(mem_mod, attr, None)
            if limiter and hasattr(limiter, "_requests"):
                limiter._requests.clear()
            if limiter and hasattr(limiter, "reset"):
                limiter.reset()
    except Exception:
        pass
    yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_body(result) -> dict[str, Any]:
    """Decode a HandlerResult.body to dict."""
    return json.loads(result.body)


# ===========================================================================
# _get_continuum_memories
# ===========================================================================


class TestGetContinuumMemories:
    """Tests for _get_continuum_memories."""

    def test_basic_retrieve_empty(self, handler, mock_http_handler, mock_continuum):
        """Returns empty list when no memories found."""
        mock_continuum.retrieve.return_value = []
        result = handler.handle("/api/v1/memory/continuum/retrieve", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 200
        body = _parse_body(result)
        assert body["memories"] == []
        assert body["count"] == 0

    def test_retrieve_returns_formatted_memories(self, handler, mock_http_handler, mock_continuum):
        """Returns properly formatted memory entries."""
        mem = _make_memory(
            id="m1",
            tier=_MockMemoryTier.MEDIUM,
            content="hello world",
            importance=0.75,
        )
        mock_continuum.retrieve.return_value = [mem]

        result = handler.handle("/api/v1/memory/continuum/retrieve", {}, mock_http_handler)
        assert result.status_code == 200
        body = _parse_body(result)
        assert body["count"] == 1
        entry = body["memories"][0]
        assert entry["id"] == "m1"
        assert entry["tier"] == "medium"
        assert entry["content"] == "hello world"
        assert entry["importance"] == 0.75

    def test_long_content_truncated(self, handler, mock_http_handler, mock_continuum):
        """Content longer than 500 chars is truncated with '...'."""
        long_text = "x" * 600
        mem = _make_memory(content=long_text)
        mock_continuum.retrieve.return_value = [mem]

        result = handler.handle("/api/v1/memory/continuum/retrieve", {}, mock_http_handler)
        body = _parse_body(result)
        content = body["memories"][0]["content"]
        assert content.endswith("...")
        assert len(content) == 503  # 500 + "..."

    def test_query_param_forwarded(self, handler, mock_http_handler, mock_continuum):
        """Query parameter is forwarded to continuum.retrieve."""
        handler.handle(
            "/api/v1/memory/continuum/retrieve",
            {"query": "search term"},
            mock_http_handler,
        )
        call_kwargs = mock_continuum.retrieve.call_args
        assert (
            call_kwargs[1]["query"] == "search term" or call_kwargs.kwargs["query"] == "search term"
        )

    def test_limit_param(self, handler, mock_http_handler, mock_continuum):
        """Limit parameter is forwarded to continuum.retrieve."""
        handler.handle(
            "/api/v1/memory/continuum/retrieve",
            {"limit": "25"},
            mock_http_handler,
        )
        call_kwargs = mock_continuum.retrieve.call_args
        assert call_kwargs.kwargs.get("limit") == 25 or call_kwargs[1].get("limit") == 25

    def test_min_importance_param(self, handler, mock_http_handler, mock_continuum):
        """Min importance parameter is forwarded."""
        handler.handle(
            "/api/v1/memory/continuum/retrieve",
            {"min_importance": "0.7"},
            mock_http_handler,
        )
        call_kwargs = mock_continuum.retrieve.call_args
        assert (
            call_kwargs.kwargs.get("min_importance") == 0.7
            or call_kwargs[1].get("min_importance") == 0.7
        )

    def test_tiers_param_parsed(self, handler, mock_http_handler, mock_continuum):
        """Tiers parameter is parsed and forwarded."""
        handler.handle(
            "/api/v1/memory/continuum/retrieve",
            {"tiers": "fast,slow"},
            mock_http_handler,
        )
        call_kwargs = mock_continuum.retrieve.call_args
        tiers = call_kwargs.kwargs.get("tiers") or call_kwargs[1].get("tiers")
        tier_names = {t.name for t in tiers}
        assert "FAST" in tier_names
        assert "SLOW" in tier_names
        assert "MEDIUM" not in tier_names

    def test_invalid_tier_ignored(self, handler, mock_http_handler, mock_continuum):
        """Invalid tier names are silently ignored."""
        handler.handle(
            "/api/v1/memory/continuum/retrieve",
            {"tiers": "fast,invalid_tier,slow"},
            mock_http_handler,
        )
        call_kwargs = mock_continuum.retrieve.call_args
        tiers = call_kwargs.kwargs.get("tiers") or call_kwargs[1].get("tiers")
        tier_names = {t.name for t in tiers}
        assert "FAST" in tier_names
        assert "SLOW" in tier_names

    def test_all_invalid_tiers_defaults_to_all(self, handler, mock_http_handler, mock_continuum):
        """When all tier names are invalid, defaults to all tiers."""
        handler.handle(
            "/api/v1/memory/continuum/retrieve",
            {"tiers": "bogus,nope"},
            mock_http_handler,
        )
        call_kwargs = mock_continuum.retrieve.call_args
        tiers = call_kwargs.kwargs.get("tiers") or call_kwargs[1].get("tiers")
        assert len(tiers) == 4  # all four tiers

    def test_response_includes_tiers_list(self, handler, mock_http_handler, mock_continuum):
        """Response includes list of requested tier names."""
        mock_continuum.retrieve.return_value = []
        result = handler.handle(
            "/api/v1/memory/continuum/retrieve",
            {"tiers": "fast,medium"},
            mock_http_handler,
        )
        body = _parse_body(result)
        assert "fast" in body["tiers"]
        assert "medium" in body["tiers"]

    def test_response_includes_query(self, handler, mock_http_handler, mock_continuum):
        """Response echoes back the query string."""
        mock_continuum.retrieve.return_value = []
        result = handler.handle(
            "/api/v1/memory/continuum/retrieve",
            {"query": "my search"},
            mock_http_handler,
        )
        body = _parse_body(result)
        assert body["query"] == "my search"

    def test_continuum_unavailable(self, handler, mock_http_handler):
        """Returns 503 when continuum memory system not available."""
        with patch("aragora.server.handlers.memory.memory.CONTINUUM_AVAILABLE", False):
            # Must re-create handler so the mixin's local import picks up the flag
            from aragora.server.handlers.memory.memory import MemoryHandler

            h = MemoryHandler(server_context={"continuum_memory": MagicMock()})
            h._auth_context = None
            result = h.handle("/api/v1/memory/continuum/retrieve", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 503

    def test_continuum_not_initialized(self, mock_http_handler):
        """Returns 503 when continuum_memory not in context."""
        from aragora.server.handlers.memory.memory import MemoryHandler

        h = MemoryHandler(server_context={})
        h._auth_context = None
        result = h.handle("/api/v1/memory/continuum/retrieve", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 503

    def test_tenant_enforcement_no_tenant_returns_400(
        self, handler, mock_http_handler, mock_continuum
    ):
        """Returns 400 when tenant enforcement is on but no tenant in auth context."""
        mock_auth = MagicMock()
        handler._auth_context = mock_auth
        mock_http_handler._auth_context = mock_auth

        # The mixin does a lazy import of aragora.memory.access inside the method,
        # so we must patch the module-level functions that get imported.
        mock_tenant_enabled = MagicMock(return_value=True)
        mock_resolve = MagicMock(return_value=None)
        mock_filter = MagicMock(return_value=[])

        import aragora.memory.access as access_mod

        with (
            patch.object(access_mod, "tenant_enforcement_enabled", mock_tenant_enabled),
            patch.object(access_mod, "resolve_tenant_id", mock_resolve),
            patch.object(access_mod, "filter_entries", mock_filter),
        ):
            result = handler.handle("/api/v1/memory/continuum/retrieve", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 400

    def test_tenant_enforcement_disabled_when_no_auth_context(
        self, handler, mock_http_handler, mock_continuum
    ):
        """When auth context is None and tenant enforcement on, enforcement is disabled."""
        handler._auth_context = None
        mock_http_handler._auth_context = None
        mock_continuum.retrieve.return_value = []

        import aragora.memory.access as access_mod

        with (
            patch.object(access_mod, "tenant_enforcement_enabled", MagicMock(return_value=True)),
            patch.object(access_mod, "resolve_tenant_id", MagicMock(return_value=None)),
            patch.object(access_mod, "filter_entries", MagicMock(return_value=[])),
        ):
            result = handler.handle("/api/v1/memory/continuum/retrieve", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 200

    def test_memory_entry_timestamps(self, handler, mock_http_handler, mock_continuum):
        """Memory entries include created_at and updated_at as strings."""
        mem = _make_memory(created_at="2026-01-01", updated_at="2026-02-01")
        mock_continuum.retrieve.return_value = [mem]

        result = handler.handle("/api/v1/memory/continuum/retrieve", {}, mock_http_handler)
        body = _parse_body(result)
        entry = body["memories"][0]
        assert entry["created_at"] == "2026-01-01"
        assert entry["updated_at"] == "2026-02-01"

    def test_memory_entry_optional_attrs(self, handler, mock_http_handler, mock_continuum):
        """Memory entries include surprise_score, consolidation_score, update_count."""
        mem = _make_memory(surprise_score=0.8, consolidation_score=0.9, update_count=5)
        mock_continuum.retrieve.return_value = [mem]

        result = handler.handle("/api/v1/memory/continuum/retrieve", {}, mock_http_handler)
        body = _parse_body(result)
        entry = body["memories"][0]
        assert entry["surprise_score"] == 0.8
        assert entry["consolidation_score"] == 0.9
        assert entry["update_count"] == 5


# ===========================================================================
# _trigger_consolidation
# ===========================================================================


class TestTriggerConsolidation:
    """Tests for _trigger_consolidation."""

    def test_get_returns_405(self, handler, mock_http_handler):
        """GET to consolidate endpoint returns 405 with Allow header."""
        result = handler.handle("/api/v1/memory/continuum/consolidate", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 405
        assert result.headers.get("Allow") == "POST"

    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_consolidation_success(self, mock_extract, handler, mock_http_handler, mock_continuum):
        """Successful consolidation returns processed/promoted/consolidated counts."""
        mock_auth = MagicMock()
        mock_auth.is_authenticated = True
        mock_extract.return_value = mock_auth

        result = handler.handle_post("/api/v1/memory/continuum/consolidate", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 200
        body = _parse_body(result)
        assert body["success"] is True
        assert body["entries_processed"] == 10
        assert body["entries_promoted"] == 3
        assert body["entries_consolidated"] == 5
        assert "duration_seconds" in body

    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_consolidation_continuum_unavailable(self, mock_extract, mock_http_handler):
        """Returns 503 when continuum not available."""
        mock_auth = MagicMock()
        mock_auth.is_authenticated = True
        mock_extract.return_value = mock_auth

        with patch("aragora.server.handlers.memory.memory.CONTINUUM_AVAILABLE", False):
            from aragora.server.handlers.memory.memory import MemoryHandler

            h = MemoryHandler(server_context={"continuum_memory": MagicMock()})
            h._auth_context = None
            result = h.handle_post("/api/v1/memory/continuum/consolidate", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 503

    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_consolidation_not_initialized(self, mock_extract, mock_http_handler):
        """Returns 503 when continuum_memory not in context."""
        mock_auth = MagicMock()
        mock_auth.is_authenticated = True
        mock_extract.return_value = mock_auth

        from aragora.server.handlers.memory.memory import MemoryHandler

        h = MemoryHandler(server_context={})
        h._auth_context = None
        result = h.handle_post("/api/v1/memory/continuum/consolidate", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 503


# ===========================================================================
# _trigger_cleanup
# ===========================================================================


class TestTriggerCleanup:
    """Tests for _trigger_cleanup."""

    def test_get_returns_405(self, handler, mock_http_handler):
        """GET to cleanup endpoint returns 405."""
        result = handler.handle("/api/v1/memory/continuum/cleanup", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 405

    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_cleanup_success(self, mock_extract, handler, mock_http_handler, mock_continuum):
        """Successful cleanup returns expired/tier_limits/duration."""
        mock_auth = MagicMock()
        mock_auth.is_authenticated = True
        mock_extract.return_value = mock_auth

        result = handler.handle_post(
            "/api/v1/memory/continuum/cleanup",
            {},
            mock_http_handler,
        )
        assert result is not None
        assert result.status_code == 200
        body = _parse_body(result)
        assert body["success"] is True
        assert "expired" in body
        assert "tier_limits" in body
        assert "duration_seconds" in body

    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_cleanup_with_tier_param(
        self, mock_extract, handler, mock_http_handler, mock_continuum
    ):
        """Cleanup respects tier parameter."""
        mock_auth = MagicMock()
        mock_auth.is_authenticated = True
        mock_extract.return_value = mock_auth

        handler.handle_post(
            "/api/v1/memory/continuum/cleanup",
            {"tier": "fast"},
            mock_http_handler,
        )
        cleanup_call = mock_continuum.cleanup_expired_memories.call_args
        tier_arg = cleanup_call.kwargs.get("tier") or cleanup_call[1].get("tier")
        assert tier_arg is not None
        assert tier_arg.name == "FAST"

    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_cleanup_invalid_tier_returns_400(
        self, mock_extract, handler, mock_http_handler, mock_continuum
    ):
        """Returns 400 for invalid tier name."""
        mock_auth = MagicMock()
        mock_auth.is_authenticated = True
        mock_extract.return_value = mock_auth

        result = handler.handle_post(
            "/api/v1/memory/continuum/cleanup",
            {"tier": "nonexistent"},
            mock_http_handler,
        )
        assert result is not None
        assert result.status_code == 400

    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_cleanup_archive_param(self, mock_extract, handler, mock_http_handler, mock_continuum):
        """Archive parameter is forwarded as boolean."""
        mock_auth = MagicMock()
        mock_auth.is_authenticated = True
        mock_extract.return_value = mock_auth

        handler.handle_post(
            "/api/v1/memory/continuum/cleanup",
            {"archive": "false"},
            mock_http_handler,
        )
        cleanup_call = mock_continuum.cleanup_expired_memories.call_args
        assert (
            cleanup_call.kwargs.get("archive") is False or cleanup_call[1].get("archive") is False
        )

    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_cleanup_max_age_param(self, mock_extract, handler, mock_http_handler, mock_continuum):
        """Max age parameter is forwarded."""
        mock_auth = MagicMock()
        mock_auth.is_authenticated = True
        mock_extract.return_value = mock_auth

        handler.handle_post(
            "/api/v1/memory/continuum/cleanup",
            {"max_age_hours": "48"},
            mock_http_handler,
        )
        cleanup_call = mock_continuum.cleanup_expired_memories.call_args
        max_age = cleanup_call.kwargs.get("max_age_hours") or cleanup_call[1].get("max_age_hours")
        assert max_age == 48.0

    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_cleanup_continuum_unavailable(self, mock_extract, mock_http_handler):
        """Returns 503 when continuum not available."""
        mock_auth = MagicMock()
        mock_auth.is_authenticated = True
        mock_extract.return_value = mock_auth

        with patch("aragora.server.handlers.memory.memory.CONTINUUM_AVAILABLE", False):
            from aragora.server.handlers.memory.memory import MemoryHandler

            h = MemoryHandler(server_context={"continuum_memory": MagicMock()})
            h._auth_context = None
            result = h.handle_post("/api/v1/memory/continuum/cleanup", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 503

    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_cleanup_max_age_zero_passes_none(
        self, mock_extract, handler, mock_http_handler, mock_continuum
    ):
        """When max_age_hours is 0, passes None to indicate no limit."""
        mock_auth = MagicMock()
        mock_auth.is_authenticated = True
        mock_extract.return_value = mock_auth

        handler.handle_post(
            "/api/v1/memory/continuum/cleanup",
            {"max_age_hours": "0"},
            mock_http_handler,
        )
        cleanup_call = mock_continuum.cleanup_expired_memories.call_args
        max_age = cleanup_call.kwargs.get("max_age_hours") or cleanup_call[1].get("max_age_hours")
        assert max_age is None


# ===========================================================================
# _get_tier_stats
# ===========================================================================


class TestGetTierStats:
    """Tests for _get_tier_stats."""

    def test_tier_stats_success(self, handler, mock_http_handler, mock_continuum):
        """Returns tier stats from continuum."""
        result = handler.handle("/api/v1/memory/tier-stats", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 200
        body = _parse_body(result)
        assert "tiers" in body
        assert body["total_memories"] == 370
        assert "transitions" in body
        assert len(body["transitions"]) == 1

    def test_tier_stats_not_initialized(self, mock_http_handler):
        """Returns 503 when continuum not in context."""
        from aragora.server.handlers.memory.memory import MemoryHandler

        h = MemoryHandler(server_context={})
        h._auth_context = None
        result = h.handle("/api/v1/memory/tier-stats", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 503


# ===========================================================================
# _get_archive_stats
# ===========================================================================


class TestGetArchiveStats:
    """Tests for _get_archive_stats."""

    def test_archive_stats_success(self, handler, mock_http_handler, mock_continuum):
        """Returns archive stats from continuum."""
        result = handler.handle("/api/v1/memory/archive-stats", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 200
        body = _parse_body(result)
        assert body["archived"] == 15
        assert body["total_size_bytes"] == 4096

    def test_archive_stats_not_initialized(self, mock_http_handler):
        """Returns 503 when continuum not in context."""
        from aragora.server.handlers.memory.memory import MemoryHandler

        h = MemoryHandler(server_context={})
        h._auth_context = None
        result = h.handle("/api/v1/memory/archive-stats", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 503


# ===========================================================================
# _get_memory_pressure
# ===========================================================================


class TestGetMemoryPressure:
    """Tests for _get_memory_pressure."""

    def test_pressure_normal(self, handler, mock_http_handler, mock_continuum):
        """Pressure < 0.5 yields 'normal' status."""
        mock_continuum.get_memory_pressure.return_value = 0.3
        result = handler.handle("/api/v1/memory/pressure", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 200
        body = _parse_body(result)
        assert body["status"] == "normal"
        assert body["pressure"] == 0.3
        assert body["cleanup_recommended"] is False

    def test_pressure_elevated(self, handler, mock_http_handler, mock_continuum):
        """Pressure between 0.5 and 0.8 yields 'elevated' status."""
        mock_continuum.get_memory_pressure.return_value = 0.6
        result = handler.handle("/api/v1/memory/pressure", {}, mock_http_handler)
        body = _parse_body(result)
        assert body["status"] == "elevated"
        assert body["cleanup_recommended"] is False

    def test_pressure_high(self, handler, mock_http_handler, mock_continuum):
        """Pressure between 0.8 and 0.9 yields 'high' status."""
        mock_continuum.get_memory_pressure.return_value = 0.85
        result = handler.handle("/api/v1/memory/pressure", {}, mock_http_handler)
        body = _parse_body(result)
        assert body["status"] == "high"
        assert body["cleanup_recommended"] is False

    def test_pressure_critical(self, handler, mock_http_handler, mock_continuum):
        """Pressure >= 0.9 yields 'critical' status and cleanup_recommended."""
        mock_continuum.get_memory_pressure.return_value = 0.95
        result = handler.handle("/api/v1/memory/pressure", {}, mock_http_handler)
        body = _parse_body(result)
        assert body["status"] == "critical"
        assert body["cleanup_recommended"] is True

    def test_pressure_includes_tier_utilization(self, handler, mock_http_handler, mock_continuum):
        """Response includes per-tier utilization breakdown."""
        result = handler.handle("/api/v1/memory/pressure", {}, mock_http_handler)
        body = _parse_body(result)
        assert "tier_utilization" in body
        util = body["tier_utilization"]
        # FAST: count=20, limit=100 -> utilization=0.2
        assert util["FAST"]["count"] == 20
        assert util["FAST"]["limit"] == 100
        assert util["FAST"]["utilization"] == 0.2

    def test_pressure_total_memories(self, handler, mock_http_handler, mock_continuum):
        """Response includes total_memories count."""
        result = handler.handle("/api/v1/memory/pressure", {}, mock_http_handler)
        body = _parse_body(result)
        assert body["total_memories"] == 370

    def test_pressure_not_initialized(self, mock_http_handler):
        """Returns 503 when continuum not in context."""
        from aragora.server.handlers.memory.memory import MemoryHandler

        h = MemoryHandler(server_context={})
        h._auth_context = None
        result = h.handle("/api/v1/memory/pressure", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 503

    def test_pressure_boundary_exactly_0_5(self, handler, mock_http_handler, mock_continuum):
        """Pressure exactly at 0.5 yields 'elevated'."""
        mock_continuum.get_memory_pressure.return_value = 0.5
        result = handler.handle("/api/v1/memory/pressure", {}, mock_http_handler)
        body = _parse_body(result)
        assert body["status"] == "elevated"

    def test_pressure_boundary_exactly_0_8(self, handler, mock_http_handler, mock_continuum):
        """Pressure exactly at 0.8 yields 'high'."""
        mock_continuum.get_memory_pressure.return_value = 0.8
        result = handler.handle("/api/v1/memory/pressure", {}, mock_http_handler)
        body = _parse_body(result)
        assert body["status"] == "high"

    def test_pressure_boundary_exactly_0_9(self, handler, mock_http_handler, mock_continuum):
        """Pressure exactly at 0.9 yields 'critical' but cleanup_recommended requires > 0.9."""
        mock_continuum.get_memory_pressure.return_value = 0.9
        result = handler.handle("/api/v1/memory/pressure", {}, mock_http_handler)
        body = _parse_body(result)
        assert body["status"] == "critical"
        # cleanup_recommended is `pressure > 0.9`, so exactly 0.9 is False
        assert body["cleanup_recommended"] is False

    def test_pressure_tier_unknown_defaults_limit(self, handler, mock_http_handler, mock_continuum):
        """Unknown tier names in stats get default limit of 1000."""
        mock_continuum.get_stats.return_value = {
            "by_tier": {"UNKNOWN_TIER": {"count": 50}},
            "total_memories": 50,
            "transitions": [],
        }
        result = handler.handle("/api/v1/memory/pressure", {}, mock_http_handler)
        body = _parse_body(result)
        util = body["tier_utilization"]
        assert util["UNKNOWN_TIER"]["limit"] == 1000
        assert util["UNKNOWN_TIER"]["utilization"] == 0.05


# ===========================================================================
# _delete_memory
# ===========================================================================


class TestDeleteMemory:
    """Tests for _delete_memory."""

    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_delete_success(self, mock_extract, handler, mock_http_handler, mock_continuum):
        """Successfully deletes a memory and returns 200."""
        mock_auth = MagicMock()
        mock_auth.is_authenticated = True
        mock_extract.return_value = mock_auth
        mock_continuum.delete.return_value = True

        result = handler.handle_delete("/api/v1/memory/continuum/mem-001", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 200
        body = _parse_body(result)
        assert body["success"] is True

    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_delete_not_found(self, mock_extract, handler, mock_http_handler, mock_continuum):
        """Returns 404 when memory not found."""
        mock_auth = MagicMock()
        mock_auth.is_authenticated = True
        mock_extract.return_value = mock_auth
        mock_continuum.delete.return_value = False

        result = handler.handle_delete("/api/v1/memory/continuum/mem-999", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 404

    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_delete_no_delete_method(
        self, mock_extract, handler, mock_http_handler, mock_continuum
    ):
        """Returns 501 when continuum backend lacks delete method."""
        mock_auth = MagicMock()
        mock_auth.is_authenticated = True
        mock_extract.return_value = mock_auth
        del mock_continuum.delete  # Remove the delete method

        result = handler.handle_delete("/api/v1/memory/continuum/mem-001", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 501

    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_delete_runtime_error_returns_500(
        self, mock_extract, handler, mock_http_handler, mock_continuum
    ):
        """Returns 500 on RuntimeError during deletion."""
        mock_auth = MagicMock()
        mock_auth.is_authenticated = True
        mock_extract.return_value = mock_auth
        mock_continuum.delete.side_effect = RuntimeError("storage failure")

        result = handler.handle_delete("/api/v1/memory/continuum/mem-001", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 500

    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_delete_tenant_enforcement_no_tenant_returns_400(
        self, mock_extract, handler, mock_http_handler, mock_continuum
    ):
        """Returns 400 when tenant enforcement is on but no tenant for delete."""
        mock_auth_ctx = MagicMock()
        mock_auth_ctx.is_authenticated = True
        mock_extract.return_value = mock_auth_ctx
        handler._auth_context = MagicMock()  # Non-None auth context
        mock_http_handler._auth_context = handler._auth_context

        import aragora.memory.access as access_mod

        with (
            patch.object(access_mod, "tenant_enforcement_enabled", MagicMock(return_value=True)),
            patch.object(access_mod, "resolve_tenant_id", MagicMock(return_value=None)),
        ):
            result = handler.handle_delete(
                "/api/v1/memory/continuum/mem-001", {}, mock_http_handler
            )
        assert result is not None
        assert result.status_code == 400

    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_delete_continuum_unavailable(self, mock_extract, mock_http_handler):
        """Returns 503 when continuum not available."""
        mock_auth = MagicMock()
        mock_auth.is_authenticated = True
        mock_extract.return_value = mock_auth

        with patch("aragora.server.handlers.memory.memory.CONTINUUM_AVAILABLE", False):
            from aragora.server.handlers.memory.memory import MemoryHandler

            h = MemoryHandler(server_context={"continuum_memory": MagicMock()})
            h._auth_context = None
            result = h.handle_delete("/api/v1/memory/continuum/mem-001", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 503

    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_delete_unauthenticated_returns_401(self, mock_extract, handler, mock_http_handler):
        """Returns 401 when user is not authenticated."""
        mock_auth = MagicMock()
        mock_auth.is_authenticated = False
        mock_extract.return_value = mock_auth

        result = handler.handle_delete("/api/v1/memory/continuum/mem-001", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 401


# ===========================================================================
# _get_all_tiers
# ===========================================================================


class TestGetAllTiers:
    """Tests for _get_all_tiers."""

    def test_all_tiers_success(self, handler, mock_http_handler, mock_continuum):
        """Returns comprehensive tier information."""
        result = handler.handle("/api/v1/memory/tiers", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 200
        body = _parse_body(result)
        assert len(body["tiers"]) == 4
        assert body["total_memories"] == 370
        assert body["transitions_24h"] == 1

    def test_all_tiers_tier_details(self, handler, mock_http_handler, mock_continuum):
        """Each tier has expected metadata fields."""
        result = handler.handle("/api/v1/memory/tiers", {}, mock_http_handler)
        body = _parse_body(result)
        fast = next(t for t in body["tiers"] if t["id"] == "fast")
        assert fast["name"] == "Fast"
        assert fast["description"] == "Immediate context, very short-term"
        assert fast["ttl_seconds"] == 60
        assert fast["ttl_human"] == "1m"
        assert fast["limit"] == 100
        assert fast["count"] == 20
        assert fast["utilization"] == 0.2
        assert fast["avg_importance"] == 0.6
        assert fast["avg_surprise"] == 0.3

    def test_all_tiers_medium_details(self, handler, mock_http_handler, mock_continuum):
        """Medium tier has correct metadata."""
        result = handler.handle("/api/v1/memory/tiers", {}, mock_http_handler)
        body = _parse_body(result)
        medium = next(t for t in body["tiers"] if t["id"] == "medium")
        assert medium["ttl_seconds"] == 3600
        assert medium["ttl_human"] == "1h"
        assert medium["limit"] == 500

    def test_all_tiers_slow_details(self, handler, mock_http_handler, mock_continuum):
        """Slow tier has correct metadata."""
        result = handler.handle("/api/v1/memory/tiers", {}, mock_http_handler)
        body = _parse_body(result)
        slow = next(t for t in body["tiers"] if t["id"] == "slow")
        assert slow["ttl_seconds"] == 86400
        assert slow["ttl_human"] == "1d"
        assert slow["limit"] == 1000

    def test_all_tiers_glacial_details(self, handler, mock_http_handler, mock_continuum):
        """Glacial tier has correct metadata."""
        result = handler.handle("/api/v1/memory/tiers", {}, mock_http_handler)
        body = _parse_body(result)
        glacial = next(t for t in body["tiers"] if t["id"] == "glacial")
        assert glacial["ttl_seconds"] == 604800
        assert glacial["ttl_human"] == "7d"
        assert glacial["limit"] == 5000

    def test_all_tiers_missing_stats_defaults_to_zero(
        self, handler, mock_http_handler, mock_continuum
    ):
        """Tiers with no stats in continuum default to zero counts."""
        mock_continuum.get_stats.return_value = {
            "by_tier": {},
            "total_memories": 0,
            "transitions": [],
        }
        result = handler.handle("/api/v1/memory/tiers", {}, mock_http_handler)
        body = _parse_body(result)
        for tier in body["tiers"]:
            assert tier["count"] == 0
            assert tier["utilization"] == 0.0

    def test_all_tiers_not_initialized(self, mock_http_handler):
        """Returns 503 when continuum not in context."""
        from aragora.server.handlers.memory.memory import MemoryHandler

        h = MemoryHandler(server_context={})
        h._auth_context = None
        result = h.handle("/api/v1/memory/tiers", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 503


# ===========================================================================
# _format_ttl
# ===========================================================================


class TestFormatTTL:
    """Tests for _format_ttl helper."""

    def test_seconds(self, handler):
        """Formats seconds < 60."""
        assert handler._format_ttl(30) == "30s"
        assert handler._format_ttl(1) == "1s"
        assert handler._format_ttl(59) == "59s"

    def test_minutes(self, handler):
        """Formats seconds in minute range."""
        assert handler._format_ttl(60) == "1m"
        assert handler._format_ttl(120) == "2m"
        assert handler._format_ttl(3599) == "59m"

    def test_hours(self, handler):
        """Formats seconds in hour range."""
        assert handler._format_ttl(3600) == "1h"
        assert handler._format_ttl(7200) == "2h"
        assert handler._format_ttl(86399) == "23h"

    def test_days(self, handler):
        """Formats seconds in day range."""
        assert handler._format_ttl(86400) == "1d"
        assert handler._format_ttl(604800) == "7d"
        assert handler._format_ttl(172800) == "2d"


# ===========================================================================
# Legacy route normalization
# ===========================================================================


class TestLegacyRouteNormalization:
    """Tests for legacy route normalization (path without /v1/)."""

    def test_legacy_retrieve(self, handler, mock_http_handler, mock_continuum):
        """Legacy /api/memory/continuum/retrieve normalizes to /api/v1/memory/continuum/retrieve."""
        mock_continuum.retrieve.return_value = []
        result = handler.handle("/api/memory/continuum/retrieve", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 200

    def test_legacy_tier_stats(self, handler, mock_http_handler, mock_continuum):
        """Legacy /api/memory/tier-stats normalizes correctly."""
        result = handler.handle("/api/memory/tier-stats", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 200

    def test_legacy_archive_stats(self, handler, mock_http_handler, mock_continuum):
        """Legacy /api/memory/archive-stats normalizes correctly."""
        result = handler.handle("/api/memory/archive-stats", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 200

    def test_legacy_pressure(self, handler, mock_http_handler, mock_continuum):
        """Legacy /api/memory/pressure normalizes correctly."""
        result = handler.handle("/api/memory/pressure", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 200

    def test_legacy_tiers(self, handler, mock_http_handler, mock_continuum):
        """Legacy /api/memory/tiers normalizes correctly."""
        result = handler.handle("/api/memory/tiers", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 200


# ===========================================================================
# can_handle tests
# ===========================================================================


class TestCanHandle:
    """Tests for can_handle routing."""

    def test_can_handle_continuum_retrieve(self, handler):
        assert handler.can_handle("/api/v1/memory/continuum/retrieve")

    def test_can_handle_tier_stats(self, handler):
        assert handler.can_handle("/api/v1/memory/tier-stats")

    def test_can_handle_archive_stats(self, handler):
        assert handler.can_handle("/api/v1/memory/archive-stats")

    def test_can_handle_pressure(self, handler):
        assert handler.can_handle("/api/v1/memory/pressure")

    def test_can_handle_tiers(self, handler):
        assert handler.can_handle("/api/v1/memory/tiers")

    def test_can_handle_continuum_consolidate(self, handler):
        assert handler.can_handle("/api/v1/memory/continuum/consolidate")

    def test_can_handle_continuum_cleanup(self, handler):
        assert handler.can_handle("/api/v1/memory/continuum/cleanup")

    def test_can_handle_continuum_delete(self, handler):
        assert handler.can_handle("/api/v1/memory/continuum/some-id")

    def test_cannot_handle_unrelated(self, handler):
        assert not handler.can_handle("/api/v1/debates/list")

    def test_can_handle_legacy_prefix(self, handler):
        assert handler.can_handle("/api/memory/continuum/retrieve")


# ===========================================================================
# Edge cases and error handling
# ===========================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_query_defaults_to_empty_string(self, handler, mock_http_handler, mock_continuum):
        """Empty query parameter defaults to ''."""
        mock_continuum.retrieve.return_value = []
        handler.handle("/api/v1/memory/continuum/retrieve", {}, mock_http_handler)
        call_kwargs = mock_continuum.retrieve.call_args
        assert call_kwargs.kwargs.get("query") == "" or call_kwargs[1].get("query") == ""

    def test_retrieve_multiple_memories(self, handler, mock_http_handler, mock_continuum):
        """Multiple memories are returned correctly."""
        mems = [_make_memory(id=f"m{i}", tier=_MockMemoryTier.FAST) for i in range(5)]
        mock_continuum.retrieve.return_value = mems

        result = handler.handle("/api/v1/memory/continuum/retrieve", {}, mock_http_handler)
        body = _parse_body(result)
        assert body["count"] == 5
        ids = [m["id"] for m in body["memories"]]
        assert ids == [f"m{i}" for i in range(5)]

    def test_content_exactly_500_chars_not_truncated(
        self, handler, mock_http_handler, mock_continuum
    ):
        """Content of exactly 500 chars is not truncated."""
        text = "a" * 500
        mem = _make_memory(content=text)
        mock_continuum.retrieve.return_value = [mem]

        result = handler.handle("/api/v1/memory/continuum/retrieve", {}, mock_http_handler)
        body = _parse_body(result)
        assert body["memories"][0]["content"] == text
        assert not body["memories"][0]["content"].endswith("...")

    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_consolidation_emits_handler_event(
        self, mock_extract, handler, mock_http_handler, mock_continuum
    ):
        """Consolidation emits a handler event on success."""
        mock_auth = MagicMock()
        mock_auth.is_authenticated = True
        mock_extract.return_value = mock_auth

        with patch(
            "aragora.server.handlers.memory.memory_continuum.emit_handler_event"
        ) as mock_emit:
            handler.handle_post("/api/v1/memory/continuum/consolidate", {}, mock_http_handler)
            mock_emit.assert_called_once()
            args = mock_emit.call_args
            assert args[0][0] == "memory"

    def test_pressure_zero(self, handler, mock_http_handler, mock_continuum):
        """Pressure of 0.0 yields 'normal' status."""
        mock_continuum.get_memory_pressure.return_value = 0.0
        result = handler.handle("/api/v1/memory/pressure", {}, mock_http_handler)
        body = _parse_body(result)
        assert body["status"] == "normal"
        assert body["pressure"] == 0.0

    def test_memory_without_optional_attributes(self, handler, mock_http_handler, mock_continuum):
        """Memory entries without optional attributes use defaults."""
        mem = MagicMock()
        mem.id = "m1"
        mem.tier = _MockMemoryTier.FAST
        mem.content = "test"
        mem.importance = 0.5
        # Remove optional attributes so getattr returns defaults
        del mem.surprise_score
        del mem.consolidation_score
        del mem.update_count
        del mem.created_at
        del mem.updated_at
        mock_continuum.retrieve.return_value = [mem]

        result = handler.handle("/api/v1/memory/continuum/retrieve", {}, mock_http_handler)
        body = _parse_body(result)
        entry = body["memories"][0]
        assert entry["surprise_score"] == 0.0
        assert entry["consolidation_score"] == 0.0
        assert entry["update_count"] == 0
        assert entry["created_at"] is None
        assert entry["updated_at"] is None
