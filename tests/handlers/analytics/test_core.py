"""Tests for aragora.server.handlers.analytics.core module.

The core.py module is a re-export hub that consolidates analytics handlers
from _analytics_impl.py and _analytics_metrics_impl.py. These tests verify:

1. All expected symbols are re-exported and accessible via the core module.
2. AnalyticsHandler: routing, RBAC, rate limiting, all analytics endpoints.
3. AnalyticsMetricsHandler: routing, RBAC, rate limiting, demo mode, org access.
4. Utility functions (_parse_time_range, _group_by_time) behave correctly.
5. Constants (VALID_GRANULARITIES, VALID_TIME_RANGES) contain expected values.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

# Import everything through the core module to test re-exports
from aragora.server.handlers.analytics.core import (
    ANALYTICS_PERMISSION,
    AnalyticsHandler,
    _analytics_limiter,
    ANALYTICS_METRICS_PERMISSION,
    AnalyticsMetricsHandler,
    _analytics_metrics_limiter,
    _parse_time_range,
    _group_by_time,
    VALID_GRANULARITIES,
    VALID_TIME_RANGES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


class _MockHTTPHandler:
    """Lightweight mock for the HTTP handler passed to handler.handle."""

    def __init__(self, method: str = "GET", body: dict[str, Any] | None = None):
        self.command = method
        self.headers = {"Content-Length": "0"}
        self.rfile = MagicMock()
        self.client_address = ("127.0.0.1", 12345)

        if body is not None:
            raw = json.dumps(body).encode()
            self.rfile.read.return_value = raw
            self.headers = {"Content-Length": str(len(raw))}
        else:
            self.rfile.read.return_value = b"{}"
            self.headers = {"Content-Length": "2"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_rate_limiters():
    """Reset rate limiter state between tests."""
    _analytics_limiter.clear()
    _analytics_metrics_limiter.clear()
    yield
    _analytics_limiter.clear()
    _analytics_metrics_limiter.clear()


@pytest.fixture
def analytics_handler():
    """Create an AnalyticsHandler with empty context."""
    return AnalyticsHandler({})


@pytest.fixture
def metrics_handler():
    """Create an AnalyticsMetricsHandler with empty context."""
    return AnalyticsMetricsHandler({})


@pytest.fixture
def http_handler():
    """Create a mock HTTP handler."""
    return _MockHTTPHandler()


# ===========================================================================
# Test: Re-exports from core.py
# ===========================================================================


class TestCoreReExports:
    """Verify that analytics.core re-exports all expected symbols."""

    def test_analytics_handler_exported(self):
        assert AnalyticsHandler is not None

    def test_analytics_permission_exported(self):
        assert ANALYTICS_PERMISSION == "analytics:read"

    def test_analytics_limiter_exported(self):
        assert _analytics_limiter is not None

    def test_analytics_metrics_handler_exported(self):
        assert AnalyticsMetricsHandler is not None

    def test_analytics_metrics_permission_exported(self):
        assert ANALYTICS_METRICS_PERMISSION == "analytics:read"

    def test_analytics_metrics_limiter_exported(self):
        assert _analytics_metrics_limiter is not None

    def test_parse_time_range_exported(self):
        assert callable(_parse_time_range)

    def test_group_by_time_exported(self):
        assert callable(_group_by_time)

    def test_valid_granularities_exported(self):
        assert isinstance(VALID_GRANULARITIES, set)

    def test_valid_time_ranges_exported(self):
        assert isinstance(VALID_TIME_RANGES, set)

    def test_all_list_complete(self):
        """Verify __all__ contains all expected symbols."""
        from aragora.server.handlers.analytics import core

        expected = {
            "AnalyticsHandler",
            "ANALYTICS_PERMISSION",
            "_analytics_limiter",
            "AnalyticsMetricsHandler",
            "ANALYTICS_METRICS_PERMISSION",
            "_analytics_metrics_limiter",
            "_parse_time_range",
            "_group_by_time",
            "VALID_GRANULARITIES",
            "VALID_TIME_RANGES",
        }
        assert set(core.__all__) == expected


# ===========================================================================
# Test: Constants
# ===========================================================================


class TestConstants:
    """Verify constant values."""

    def test_valid_granularities_contains_daily(self):
        assert "daily" in VALID_GRANULARITIES

    def test_valid_granularities_contains_weekly(self):
        assert "weekly" in VALID_GRANULARITIES

    def test_valid_granularities_contains_monthly(self):
        assert "monthly" in VALID_GRANULARITIES

    def test_valid_granularities_count(self):
        assert len(VALID_GRANULARITIES) == 3

    def test_valid_time_ranges_contains_7d(self):
        assert "7d" in VALID_TIME_RANGES

    def test_valid_time_ranges_contains_30d(self):
        assert "30d" in VALID_TIME_RANGES

    def test_valid_time_ranges_contains_all(self):
        assert "all" in VALID_TIME_RANGES

    def test_valid_time_ranges_count(self):
        assert len(VALID_TIME_RANGES) == 7


# ===========================================================================
# Test: _parse_time_range
# ===========================================================================


class TestParseTimeRange:
    """Tests for the _parse_time_range utility function."""

    def test_all_returns_none(self):
        assert _parse_time_range("all") is None

    def test_7d_returns_datetime(self):
        result = _parse_time_range("7d")
        assert isinstance(result, datetime)
        assert result.tzinfo is not None

    def test_30d_returns_datetime(self):
        result = _parse_time_range("30d")
        assert isinstance(result, datetime)

    def test_365d_returns_datetime(self):
        result = _parse_time_range("365d")
        assert isinstance(result, datetime)

    def test_invalid_format_returns_default_30d(self):
        """Invalid format falls back to 30 days."""
        result = _parse_time_range("invalid")
        assert isinstance(result, datetime)

    def test_empty_string_returns_default(self):
        result = _parse_time_range("")
        assert isinstance(result, datetime)

    def test_7d_is_approximately_7_days_ago(self):
        result = _parse_time_range("7d")
        now = datetime.now(timezone.utc)
        diff = now - result
        assert 6.9 < diff.total_seconds() / 86400 < 7.1


# ===========================================================================
# Test: _group_by_time
# ===========================================================================


class TestGroupByTime:
    """Tests for the _group_by_time utility function."""

    def test_empty_items(self):
        result = _group_by_time([], "ts", "daily")
        assert result == {}

    def test_daily_grouping_with_string_timestamps(self):
        items = [
            {"ts": "2026-02-01T10:00:00+00:00", "value": 1},
            {"ts": "2026-02-01T15:00:00+00:00", "value": 2},
            {"ts": "2026-02-02T10:00:00+00:00", "value": 3},
        ]
        result = _group_by_time(items, "ts", "daily")
        assert "2026-02-01" in result
        assert len(result["2026-02-01"]) == 2
        assert "2026-02-02" in result
        assert len(result["2026-02-02"]) == 1

    def test_monthly_grouping(self):
        items = [
            {"ts": "2026-01-15T00:00:00+00:00", "val": 1},
            {"ts": "2026-02-20T00:00:00+00:00", "val": 2},
        ]
        result = _group_by_time(items, "ts", "monthly")
        assert "2026-01" in result
        assert "2026-02" in result

    def test_weekly_grouping(self):
        items = [
            {"ts": "2026-02-01T00:00:00+00:00", "val": 1},
            {"ts": "2026-02-08T00:00:00+00:00", "val": 2},
        ]
        result = _group_by_time(items, "ts", "weekly")
        assert len(result) >= 1

    def test_missing_timestamp_key_skips_item(self):
        items = [{"other_key": "no timestamp"}]
        result = _group_by_time(items, "ts", "daily")
        assert result == {}

    def test_invalid_timestamp_string_skips_item(self):
        items = [{"ts": "not-a-date"}]
        result = _group_by_time(items, "ts", "daily")
        assert result == {}

    def test_datetime_objects_as_timestamps(self):
        items = [
            {"ts": datetime(2026, 2, 10, 12, 0, tzinfo=timezone.utc), "val": 1},
        ]
        result = _group_by_time(items, "ts", "daily")
        assert "2026-02-10" in result

    def test_non_datetime_non_string_skips(self):
        items = [{"ts": 12345, "val": 1}]
        result = _group_by_time(items, "ts", "daily")
        assert result == {}


# ===========================================================================
# Test: AnalyticsHandler.can_handle
# ===========================================================================


class TestAnalyticsHandlerCanHandle:
    """Tests for AnalyticsHandler.can_handle route matching."""

    def test_disagreements_route(self, analytics_handler):
        assert analytics_handler.can_handle("/api/analytics/disagreements") is True

    def test_role_rotation_route(self, analytics_handler):
        assert analytics_handler.can_handle("/api/analytics/role-rotation") is True

    def test_early_stops_route(self, analytics_handler):
        assert analytics_handler.can_handle("/api/analytics/early-stops") is True

    def test_consensus_quality_route(self, analytics_handler):
        assert analytics_handler.can_handle("/api/analytics/consensus-quality") is True

    def test_ranking_stats_route(self, analytics_handler):
        assert analytics_handler.can_handle("/api/ranking/stats") is True

    def test_memory_stats_route(self, analytics_handler):
        assert analytics_handler.can_handle("/api/memory/stats") is True

    def test_cross_pollination_route(self, analytics_handler):
        assert analytics_handler.can_handle("/api/analytics/cross-pollination") is True

    def test_versioned_path_accepted(self, analytics_handler):
        assert analytics_handler.can_handle("/api/v1/analytics/disagreements") is True

    def test_unknown_route_rejected(self, analytics_handler):
        assert analytics_handler.can_handle("/api/analytics/unknown") is False


# ===========================================================================
# Test: AnalyticsHandler.handle — route dispatch
# ===========================================================================


class TestAnalyticsHandlerRouting:
    """Test that AnalyticsHandler.handle dispatches to the correct methods."""

    @pytest.mark.asyncio
    async def test_disagreements_returns_stats(self, analytics_handler, http_handler):
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = []
        analytics_handler.ctx = {"storage": mock_storage}

        result = await analytics_handler.handle("/api/analytics/disagreements", {}, http_handler)
        assert result is not None
        assert _status(result) == 200
        body = _body(result)
        assert "stats" in body

    @pytest.mark.asyncio
    async def test_role_rotation_returns_stats(self, analytics_handler, http_handler):
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = []
        analytics_handler.ctx = {"storage": mock_storage}

        result = await analytics_handler.handle("/api/analytics/role-rotation", {}, http_handler)
        assert result is not None
        assert _status(result) == 200
        body = _body(result)
        assert "stats" in body

    @pytest.mark.asyncio
    async def test_early_stops_returns_stats(self, analytics_handler, http_handler):
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = []
        analytics_handler.ctx = {"storage": mock_storage}

        result = await analytics_handler.handle("/api/analytics/early-stops", {}, http_handler)
        assert result is not None
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_consensus_quality_returns_stats(self, analytics_handler, http_handler):
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = []
        analytics_handler.ctx = {"storage": mock_storage}

        result = await analytics_handler.handle(
            "/api/analytics/consensus-quality", {}, http_handler
        )
        assert result is not None
        assert _status(result) == 200
        body = _body(result)
        assert "quality_score" in body

    @pytest.mark.asyncio
    async def test_memory_stats_no_nomic_dir(self, analytics_handler, http_handler):
        """When nomic_dir is not configured, returns empty stats."""
        analytics_handler.ctx = {}

        result = await analytics_handler.handle("/api/memory/stats", {}, http_handler)
        assert result is not None
        assert _status(result) == 200
        body = _body(result)
        assert body["stats"] == {}

    @pytest.mark.asyncio
    async def test_unknown_route_returns_none(self, analytics_handler, http_handler):
        result = await analytics_handler.handle("/api/analytics/nonexistent", {}, http_handler)
        assert result is None

    @pytest.mark.asyncio
    async def test_rate_limit_returns_429(self, analytics_handler, http_handler):
        """When rate limiter denies the request, a 429 is returned."""
        with patch.object(_analytics_limiter, "is_allowed", return_value=False):
            result = await analytics_handler.handle(
                "/api/analytics/disagreements", {}, http_handler
            )
        assert result is not None
        assert _status(result) == 429

    @pytest.mark.asyncio
    async def test_versioned_path_dispatches(self, analytics_handler, http_handler):
        """A versioned path /api/v1/... is correctly stripped and dispatched."""
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = []
        analytics_handler.ctx = {"storage": mock_storage}

        result = await analytics_handler.handle("/api/v1/analytics/disagreements", {}, http_handler)
        assert result is not None
        assert _status(result) == 200


# ===========================================================================
# Test: AnalyticsHandler — storage-backed endpoint behavior
# ===========================================================================


class TestAnalyticsHandlerData:
    """Test AnalyticsHandler methods with mock debate data."""

    @pytest.mark.asyncio
    async def test_disagreement_stats_with_debates(self, analytics_handler, http_handler):
        debates = [
            {
                "result": {
                    "disagreement_report": {"unanimous_critiques": True},
                    "uncertainty_metrics": {"disagreement_type": "mild"},
                }
            },
            {
                "result": {
                    "disagreement_report": {"unanimous_critiques": False},
                    "uncertainty_metrics": {"disagreement_type": "severe"},
                }
            },
        ]
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = debates
        analytics_handler.ctx = {"storage": mock_storage}

        result = await analytics_handler.handle("/api/analytics/disagreements", {}, http_handler)
        body = _body(result)
        stats = body["stats"]
        assert stats["total_debates"] == 2
        assert stats["with_disagreements"] == 1
        assert stats["unanimous"] == 1

    @pytest.mark.asyncio
    async def test_early_stop_stats_with_debates(self, analytics_handler, http_handler):
        debates = [
            {"result": {"rounds_used": 2, "early_stopped": True}},
            {"result": {"rounds_used": 5, "early_stopped": False}},
        ]
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = debates
        analytics_handler.ctx = {"storage": mock_storage}

        result = await analytics_handler.handle("/api/analytics/early-stops", {}, http_handler)
        body = _body(result)
        stats = body["stats"]
        assert stats["total_debates"] == 2
        assert stats["early_stopped"] == 1
        assert stats["full_rounds"] == 1
        assert stats["average_rounds"] == 3.5

    @pytest.mark.asyncio
    async def test_role_rotation_with_messages(self, analytics_handler, http_handler):
        debates = [
            {
                "messages": [
                    {"cognitive_role": "analyst"},
                    {"cognitive_role": "critic"},
                    {"role": "proposer"},
                ]
            },
        ]
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = debates
        analytics_handler.ctx = {"storage": mock_storage}

        result = await analytics_handler.handle("/api/analytics/role-rotation", {}, http_handler)
        body = _body(result)
        stats = body["stats"]
        assert stats["role_assignments"]["analyst"] == 1
        assert stats["role_assignments"]["critic"] == 1
        assert stats["role_assignments"]["proposer"] == 1

    @pytest.mark.asyncio
    async def test_no_storage_returns_empty_stats(self, analytics_handler, http_handler):
        analytics_handler.ctx = {}

        result = await analytics_handler.handle("/api/analytics/disagreements", {}, http_handler)
        body = _body(result)
        assert body["stats"] == {}

    @pytest.mark.asyncio
    async def test_ranking_stats_no_elo(self, analytics_handler, http_handler):
        """When no ELO system is available, returns 503."""
        analytics_handler.ctx = {}

        result = await analytics_handler.handle("/api/ranking/stats", {}, http_handler)
        assert result is not None
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_ranking_stats_with_elo(self, analytics_handler, http_handler):
        mock_entry = MagicMock()
        mock_entry.agent_name = "claude"
        mock_entry.elo = 1600
        mock_entry.debates_count = 10

        mock_elo = MagicMock()
        mock_elo.get_leaderboard.return_value = [mock_entry]
        analytics_handler.ctx = {"elo_system": mock_elo}

        result = await analytics_handler.handle("/api/ranking/stats", {}, http_handler)
        assert _status(result) == 200
        body = _body(result)
        assert body["stats"]["total_agents"] == 1
        assert body["stats"]["top_agent"] == "claude"

    @pytest.mark.asyncio
    async def test_memory_stats_with_nomic_dir(self, analytics_handler, http_handler, tmp_path):
        """When nomic_dir exists, checks for db files."""
        analytics_handler.ctx = {"nomic_dir": tmp_path}

        result = await analytics_handler.handle("/api/memory/stats", {}, http_handler)
        body = _body(result)
        assert body["stats"]["embeddings_db"] is False
        assert body["stats"]["continuum_memory"] is False

    @pytest.mark.asyncio
    async def test_memory_stats_finds_db_files(self, analytics_handler, http_handler, tmp_path):
        (tmp_path / "debate_embeddings.db").touch()
        (tmp_path / "continuum_memory.db").touch()
        analytics_handler.ctx = {"nomic_dir": tmp_path}

        result = await analytics_handler.handle("/api/memory/stats", {}, http_handler)
        body = _body(result)
        assert body["stats"]["embeddings_db"] is True
        assert body["stats"]["continuum_memory"] is True


# ===========================================================================
# Test: AnalyticsHandler — consensus quality details
# ===========================================================================


class TestConsensusQuality:
    """Tests for the consensus quality endpoint behavior."""

    @pytest.mark.asyncio
    async def test_no_debates_returns_insufficient_data(self, analytics_handler, http_handler):
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = []
        analytics_handler.ctx = {"storage": mock_storage}

        result = await analytics_handler.handle(
            "/api/analytics/consensus-quality", {}, http_handler
        )
        body = _body(result)
        assert body["stats"]["trend"] == "insufficient_data"
        assert body["quality_score"] == 0

    @pytest.mark.asyncio
    async def test_high_quality_debates(self, analytics_handler, http_handler):
        debates = [
            {
                "id": f"debate-{i}",
                "timestamp": "2026-02-10T00:00:00Z",
                "result": {"confidence": 0.95, "consensus_reached": True},
            }
            for i in range(10)
        ]
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = debates
        analytics_handler.ctx = {"storage": mock_storage}

        result = await analytics_handler.handle(
            "/api/analytics/consensus-quality", {}, http_handler
        )
        body = _body(result)
        assert body["quality_score"] > 60
        assert body["alert"] is None or body["alert"]["level"] != "critical"

    @pytest.mark.asyncio
    async def test_low_quality_produces_critical_alert(self, analytics_handler, http_handler):
        debates = [
            {
                "id": f"debate-{i}",
                "timestamp": "2026-02-10T00:00:00Z",
                "result": {"confidence": 0.1, "consensus_reached": False},
            }
            for i in range(10)
        ]
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = debates
        analytics_handler.ctx = {"storage": mock_storage}

        result = await analytics_handler.handle(
            "/api/analytics/consensus-quality", {}, http_handler
        )
        body = _body(result)
        assert body["quality_score"] < 40
        assert body["alert"] is not None
        assert body["alert"]["level"] == "critical"


# ===========================================================================
# Test: AnalyticsHandler — cross-pollination endpoints
# ===========================================================================


class TestCrossPollination:
    """Tests for cross-pollination analytics endpoints."""

    @pytest.mark.asyncio
    async def test_cross_pollination_stats_baseline(self, analytics_handler, http_handler):
        """Cross-pollination endpoint returns baseline stats even when RLM unavailable."""
        with patch.dict("sys.modules", {"aragora.rlm.bridge": None}):
            result = await analytics_handler.handle(
                "/api/analytics/cross-pollination", {}, http_handler
            )
        assert _status(result) == 200
        body = _body(result)
        assert "stats" in body
        assert body["version"] == "2.0.3"

    @pytest.mark.asyncio
    async def test_learning_efficiency_no_elo(self, analytics_handler, http_handler):
        with patch.dict("sys.modules", {"aragora.ranking.elo": None}):
            result = await analytics_handler.handle(
                "/api/analytics/learning-efficiency", {"agent": ["claude"]}, http_handler
            )
        body = _body(result)
        assert "error" in body or "agents" in body

    @pytest.mark.asyncio
    async def test_voting_accuracy_no_elo(self, analytics_handler, http_handler):
        with patch.dict("sys.modules", {"aragora.ranking.elo": None}):
            result = await analytics_handler.handle(
                "/api/analytics/voting-accuracy", {"agent": ["claude"]}, http_handler
            )
        body = _body(result)
        assert "error" in body or "agents" in body

    @pytest.mark.asyncio
    async def test_calibration_no_elo(self, analytics_handler, http_handler):
        with patch.dict("sys.modules", {"aragora.ranking.elo": None}):
            result = await analytics_handler.handle(
                "/api/analytics/calibration", {"agent": ["claude"]}, http_handler
            )
        body = _body(result)
        assert "error" in body or "agents" in body


# ===========================================================================
# Test: AnalyticsMetricsHandler.can_handle
# ===========================================================================


class TestMetricsHandlerCanHandle:
    """Tests for AnalyticsMetricsHandler route matching."""

    def test_debates_overview(self, metrics_handler):
        assert metrics_handler.can_handle("/api/analytics/debates/overview") is True

    def test_debates_trends(self, metrics_handler):
        assert metrics_handler.can_handle("/api/analytics/debates/trends") is True

    def test_agents_leaderboard(self, metrics_handler):
        assert metrics_handler.can_handle("/api/analytics/agents/leaderboard") is True

    def test_agent_performance_pattern(self, metrics_handler):
        assert metrics_handler.can_handle("/api/analytics/agents/claude-opus/performance") is True

    def test_usage_tokens(self, metrics_handler):
        assert metrics_handler.can_handle("/api/analytics/usage/tokens") is True

    def test_unknown_route(self, metrics_handler):
        assert metrics_handler.can_handle("/api/analytics/unknown") is False

    def test_versioned_route(self, metrics_handler):
        assert metrics_handler.can_handle("/api/v1/analytics/debates/overview") is True


# ===========================================================================
# Test: AnalyticsMetricsHandler.handle — rate limiting
# ===========================================================================


class TestMetricsHandlerRateLimit:
    """Test rate limit enforcement on AnalyticsMetricsHandler."""

    @pytest.mark.asyncio
    async def test_rate_limit_returns_429(self, metrics_handler, http_handler):
        with patch.object(_analytics_metrics_limiter, "is_allowed", return_value=False):
            result = await metrics_handler.handle(
                "/api/analytics/debates/overview", {}, http_handler
            )
        assert result is not None
        assert _status(result) == 429


# ===========================================================================
# Test: AnalyticsMetricsHandler — demo mode
# ===========================================================================


class TestMetricsHandlerDemoMode:
    """Test demo mode returns sample data."""

    @pytest.mark.asyncio
    async def test_demo_mode_debates_overview(self, metrics_handler, http_handler):
        with patch.dict("os.environ", {"ARAGORA_DEMO_MODE": "true"}):
            result = await metrics_handler.handle(
                "/api/analytics/debates/overview", {}, http_handler
            )
        assert result is not None
        body = _body(result)
        assert body["total_debates"] == 47

    @pytest.mark.asyncio
    async def test_demo_mode_agents_leaderboard(self, metrics_handler, http_handler):
        with patch.dict("os.environ", {"ARAGORA_DEMO_MODE": "true"}):
            result = await metrics_handler.handle(
                "/api/analytics/agents/leaderboard", {}, http_handler
            )
        assert result is not None
        body = _body(result)
        assert "leaderboard" in body


# ===========================================================================
# Test: AnalyticsMetricsHandler._validate_org_access
# ===========================================================================


class TestOrgAccess:
    """Test organization access validation."""

    def test_admin_can_access_any_org(self, metrics_handler):
        ctx = MagicMock()
        ctx.roles = {"admin"}
        ctx.org_id = "org-1"
        org_id, error = metrics_handler._validate_org_access(ctx, "org-2")
        assert error is None
        assert org_id == "org-2"

    def test_platform_admin_can_access_any_org(self, metrics_handler):
        ctx = MagicMock()
        ctx.roles = {"platform_admin"}
        ctx.org_id = "org-1"
        org_id, error = metrics_handler._validate_org_access(ctx, "org-other")
        assert error is None

    def test_regular_user_defaults_to_own_org(self, metrics_handler):
        ctx = MagicMock()
        ctx.roles = {"viewer"}
        ctx.org_id = "org-1"
        org_id, error = metrics_handler._validate_org_access(ctx, None)
        assert error is None
        assert org_id == "org-1"

    def test_regular_user_cannot_access_other_org(self, metrics_handler):
        ctx = MagicMock()
        ctx.roles = {"viewer"}
        ctx.org_id = "org-1"
        org_id, error = metrics_handler._validate_org_access(ctx, "org-2")
        assert error is not None
        assert _status(error) == 403
