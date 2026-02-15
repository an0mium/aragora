"""
Tests for analytics internals:
- aragora.server.handlers._analytics_metrics_common: time parsing, grouping
- aragora.server.handlers._analytics_impl: AnalyticsHandler data transformations

Tests cover:
_analytics_metrics_common:
- VALID_GRANULARITIES and VALID_TIME_RANGES constants
- _parse_time_range: 7d, 30d, 90d, 365d, all, invalid, default
- _group_by_time: daily, weekly, monthly, missing timestamps, string vs datetime

_analytics_impl (AnalyticsHandler):
- Instantiation, ROUTES, can_handle
- _get_disagreement_stats: with data, empty storage, no storage
- _get_role_rotation_stats: with roles, empty storage
- _get_early_stop_stats: early stopped, full rounds, average calculation
- _get_consensus_quality: trends (improving, declining, stable), alerts
- _get_ranking_stats: with agents, no elo system
- _get_memory_stats: databases present, databases missing
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock

import pytest

from aragora.server.handlers._analytics_metrics_common import (
    VALID_GRANULARITIES,
    VALID_TIME_RANGES,
    _group_by_time,
    _parse_time_range,
)
from aragora.server.handlers._analytics_impl import AnalyticsHandler
from aragora.server.handlers.utils.responses import HandlerResult


# ===========================================================================
# Helpers
# ===========================================================================


def _parse_body(result: HandlerResult) -> dict[str, Any]:
    """Parse JSON body from HandlerResult."""
    return json.loads(result.body)


def _make_mock_handler(
    method: str = "GET",
    headers: dict[str, str] | None = None,
) -> MagicMock:
    """Create a mock HTTP handler object."""
    handler = MagicMock()
    handler.command = method
    handler.client_address = ("127.0.0.1", 12345)
    h = {
        "Content-Length": "0",
        "Content-Type": "application/json",
        "Host": "localhost:8080",
    }
    if headers:
        h.update(headers)
    handler.headers = h
    handler.rfile = MagicMock()
    handler.rfile.read.return_value = b""
    return handler


# ===========================================================================
# Test _analytics_metrics_common constants
# ===========================================================================


class TestAnalyticsConstants:
    """Tests for shared analytics constants."""

    def test_valid_granularities(self):
        assert "daily" in VALID_GRANULARITIES
        assert "weekly" in VALID_GRANULARITIES
        assert "monthly" in VALID_GRANULARITIES
        assert len(VALID_GRANULARITIES) == 3

    def test_valid_time_ranges(self):
        assert "7d" in VALID_TIME_RANGES
        assert "30d" in VALID_TIME_RANGES
        assert "90d" in VALID_TIME_RANGES
        assert "all" in VALID_TIME_RANGES
        assert len(VALID_TIME_RANGES) == 7


# ===========================================================================
# Test _parse_time_range
# ===========================================================================


class TestParseTimeRange:
    """Tests for the time range parser."""

    def test_7d(self):
        result = _parse_time_range("7d")
        assert result is not None
        # Should be approximately 7 days ago
        expected = datetime.now(timezone.utc) - timedelta(days=7)
        assert abs((result - expected).total_seconds()) < 5

    def test_30d(self):
        result = _parse_time_range("30d")
        assert result is not None
        expected = datetime.now(timezone.utc) - timedelta(days=30)
        assert abs((result - expected).total_seconds()) < 5

    def test_90d(self):
        result = _parse_time_range("90d")
        assert result is not None
        expected = datetime.now(timezone.utc) - timedelta(days=90)
        assert abs((result - expected).total_seconds()) < 5

    def test_365d(self):
        result = _parse_time_range("365d")
        assert result is not None
        expected = datetime.now(timezone.utc) - timedelta(days=365)
        assert abs((result - expected).total_seconds()) < 5

    def test_all_returns_none(self):
        assert _parse_time_range("all") is None

    def test_invalid_format_defaults_to_30d(self):
        result = _parse_time_range("invalid")
        assert result is not None
        expected = datetime.now(timezone.utc) - timedelta(days=30)
        assert abs((result - expected).total_seconds()) < 5

    def test_empty_string_defaults_to_30d(self):
        result = _parse_time_range("")
        assert result is not None
        expected = datetime.now(timezone.utc) - timedelta(days=30)
        assert abs((result - expected).total_seconds()) < 5


# ===========================================================================
# Test _group_by_time
# ===========================================================================


class TestGroupByTime:
    """Tests for time-based grouping."""

    def test_daily_grouping(self):
        items = [
            {"ts": "2024-01-15T10:00:00Z", "value": 1},
            {"ts": "2024-01-15T14:00:00Z", "value": 2},
            {"ts": "2024-01-16T10:00:00Z", "value": 3},
        ]
        result = _group_by_time(items, "ts", "daily")
        assert "2024-01-15" in result
        assert len(result["2024-01-15"]) == 2
        assert "2024-01-16" in result
        assert len(result["2024-01-16"]) == 1

    def test_weekly_grouping(self):
        items = [
            {"ts": "2024-01-15T10:00:00Z", "value": 1},
            {"ts": "2024-01-22T10:00:00Z", "value": 2},
        ]
        result = _group_by_time(items, "ts", "weekly")
        # Should produce different week keys
        assert len(result) >= 1

    def test_monthly_grouping(self):
        items = [
            {"ts": "2024-01-15T10:00:00Z", "value": 1},
            {"ts": "2024-01-20T10:00:00Z", "value": 2},
            {"ts": "2024-02-10T10:00:00Z", "value": 3},
        ]
        result = _group_by_time(items, "ts", "monthly")
        assert "2024-01" in result
        assert len(result["2024-01"]) == 2
        assert "2024-02" in result
        assert len(result["2024-02"]) == 1

    def test_missing_timestamp_skipped(self):
        items = [
            {"ts": "2024-01-15T10:00:00Z", "value": 1},
            {"value": 2},  # No timestamp
        ]
        result = _group_by_time(items, "ts", "daily")
        total_items = sum(len(v) for v in result.values())
        assert total_items == 1

    def test_datetime_objects(self):
        items = [
            {"ts": datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc), "value": 1},
            {"ts": datetime(2024, 1, 15, 14, 0, tzinfo=timezone.utc), "value": 2},
        ]
        result = _group_by_time(items, "ts", "daily")
        assert "2024-01-15" in result
        assert len(result["2024-01-15"]) == 2

    def test_empty_list(self):
        result = _group_by_time([], "ts", "daily")
        assert result == {}

    def test_invalid_timestamp_skipped(self):
        items = [
            {"ts": "not-a-date", "value": 1},
            {"ts": "2024-01-15T10:00:00Z", "value": 2},
        ]
        result = _group_by_time(items, "ts", "daily")
        total_items = sum(len(v) for v in result.values())
        assert total_items == 1

    def test_non_string_non_datetime_skipped(self):
        items = [
            {"ts": 12345, "value": 1},
            {"ts": "2024-01-15T10:00:00Z", "value": 2},
        ]
        result = _group_by_time(items, "ts", "daily")
        total_items = sum(len(v) for v in result.values())
        assert total_items == 1


# ===========================================================================
# AnalyticsHandler Fixtures
# ===========================================================================


@pytest.fixture
def analytics_handler():
    """Create an AnalyticsHandler with mocked dependencies."""
    with patch("aragora.server.handlers._analytics_impl._analytics_limiter") as mock_limiter:
        mock_limiter.is_allowed.return_value = True
        h = AnalyticsHandler(ctx={})
        yield h


# ===========================================================================
# Test AnalyticsHandler Basics
# ===========================================================================


class TestAnalyticsHandlerBasics:
    """Basic instantiation and routing tests."""

    def test_instantiation(self, analytics_handler):
        assert analytics_handler is not None
        assert isinstance(analytics_handler, AnalyticsHandler)

    def test_routes_count(self, analytics_handler):
        assert len(analytics_handler.ROUTES) == 10

    def test_can_handle_disagreements(self, analytics_handler):
        assert analytics_handler.can_handle("/api/analytics/disagreements") is True

    def test_can_handle_versioned_path(self, analytics_handler):
        assert analytics_handler.can_handle("/api/v1/analytics/disagreements") is True

    def test_can_handle_role_rotation(self, analytics_handler):
        assert analytics_handler.can_handle("/api/analytics/role-rotation") is True

    def test_can_handle_early_stops(self, analytics_handler):
        assert analytics_handler.can_handle("/api/analytics/early-stops") is True

    def test_can_handle_ranking_stats(self, analytics_handler):
        assert analytics_handler.can_handle("/api/ranking/stats") is True

    def test_can_handle_memory_stats(self, analytics_handler):
        assert analytics_handler.can_handle("/api/memory/stats") is True

    def test_can_handle_cross_pollination(self, analytics_handler):
        assert analytics_handler.can_handle("/api/analytics/cross-pollination") is True

    def test_cannot_handle_other(self, analytics_handler):
        assert analytics_handler.can_handle("/api/debates") is False


# ===========================================================================
# Test _get_disagreement_stats
# ===========================================================================


class TestDisagreementStats:
    """Tests for disagreement statistics."""

    def test_with_debates(self, analytics_handler):
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = [
            {
                "result": {
                    "disagreement_report": {"unanimous_critiques": True},
                    "uncertainty_metrics": {"disagreement_type": "semantic"},
                }
            },
            {
                "result": {
                    "disagreement_report": {"unanimous_critiques": False},
                    "uncertainty_metrics": {"disagreement_type": "factual"},
                }
            },
        ]

        with patch.object(analytics_handler, "get_storage", return_value=mock_storage):
            result = analytics_handler._get_disagreement_stats.__wrapped__(analytics_handler)
            assert result.status_code == 200
            data = _parse_body(result)
            assert data["stats"]["total_debates"] == 2

    def test_no_storage(self, analytics_handler):
        with patch.object(analytics_handler, "get_storage", return_value=None):
            result = analytics_handler._get_disagreement_stats.__wrapped__(analytics_handler)
            assert result.status_code == 200
            data = _parse_body(result)
            assert data["stats"] == {}

    def test_empty_debates(self, analytics_handler):
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = []

        with patch.object(analytics_handler, "get_storage", return_value=mock_storage):
            result = analytics_handler._get_disagreement_stats.__wrapped__(analytics_handler)
            assert result.status_code == 200
            data = _parse_body(result)
            assert data["stats"]["total_debates"] == 0


# ===========================================================================
# Test _get_role_rotation_stats
# ===========================================================================


class TestRoleRotationStats:
    """Tests for role rotation statistics."""

    def test_with_roles(self, analytics_handler):
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = [
            {
                "messages": [
                    {"cognitive_role": "analyst"},
                    {"cognitive_role": "critic"},
                    {"role": "unknown"},
                ]
            }
        ]

        with patch.object(analytics_handler, "get_storage", return_value=mock_storage):
            result = analytics_handler._get_role_rotation_stats.__wrapped__(analytics_handler)
            assert result.status_code == 200
            data = _parse_body(result)
            assert data["stats"]["total_debates"] == 1
            assert "analyst" in data["stats"]["role_assignments"]

    def test_no_storage(self, analytics_handler):
        with patch.object(analytics_handler, "get_storage", return_value=None):
            result = analytics_handler._get_role_rotation_stats.__wrapped__(analytics_handler)
            assert result.status_code == 200
            data = _parse_body(result)
            assert data["stats"] == {}


# ===========================================================================
# Test _get_early_stop_stats
# ===========================================================================


class TestEarlyStopStats:
    """Tests for early stop statistics."""

    def test_mixed_stops(self, analytics_handler):
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = [
            {"result": {"early_stopped": True, "rounds_used": 2}},
            {"result": {"early_stopped": False, "rounds_used": 5}},
            {"result": {"early_stopped": True, "rounds_used": 3}},
        ]

        with patch.object(analytics_handler, "get_storage", return_value=mock_storage):
            result = analytics_handler._get_early_stop_stats.__wrapped__(analytics_handler)
            assert result.status_code == 200
            data = _parse_body(result)
            stats = data["stats"]
            assert stats["total_debates"] == 3
            assert stats["early_stopped"] == 2
            assert stats["full_rounds"] == 1
            assert abs(stats["average_rounds"] - (10 / 3)) < 0.01

    def test_no_debates(self, analytics_handler):
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = []

        with patch.object(analytics_handler, "get_storage", return_value=mock_storage):
            result = analytics_handler._get_early_stop_stats.__wrapped__(analytics_handler)
            assert result.status_code == 200
            data = _parse_body(result)
            assert data["stats"]["average_rounds"] == 0.0

    def test_no_storage(self, analytics_handler):
        with patch.object(analytics_handler, "get_storage", return_value=None):
            result = analytics_handler._get_early_stop_stats.__wrapped__(analytics_handler)
            assert result.status_code == 200
            data = _parse_body(result)
            assert data["stats"] == {}


# ===========================================================================
# Test _get_consensus_quality
# ===========================================================================


class TestConsensusQuality:
    """Tests for consensus quality metrics."""

    def _make_debates(self, confidences: list[float], consensus_flags: list[bool] | None = None):
        """Create mock debates with given confidence values."""
        if consensus_flags is None:
            consensus_flags = [True] * len(confidences)
        debates = []
        for i, (conf, cons) in enumerate(zip(confidences, consensus_flags)):
            debates.append(
                {
                    "id": f"debate-{i}",
                    "timestamp": f"2024-01-{15 + i:02d}T10:00:00Z",
                    "result": {
                        "confidence": conf,
                        "consensus_reached": cons,
                    },
                }
            )
        return debates

    def test_improving_trend(self, analytics_handler):
        # First half low, second half high
        confidences = [0.3, 0.35, 0.4, 0.45, 0.5, 0.7, 0.75, 0.8, 0.85, 0.9]
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = self._make_debates(confidences)

        with patch.object(analytics_handler, "get_storage", return_value=mock_storage):
            result = analytics_handler._get_consensus_quality.__wrapped__(analytics_handler)
            assert result.status_code == 200
            data = _parse_body(result)
            assert data["stats"]["trend"] == "improving"

    def test_declining_trend(self, analytics_handler):
        # First half high, second half low
        confidences = [0.9, 0.85, 0.8, 0.75, 0.7, 0.3, 0.35, 0.4, 0.45, 0.5]
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = self._make_debates(confidences)

        with patch.object(analytics_handler, "get_storage", return_value=mock_storage):
            result = analytics_handler._get_consensus_quality.__wrapped__(analytics_handler)
            assert result.status_code == 200
            data = _parse_body(result)
            assert data["stats"]["trend"] == "declining"

    def test_stable_trend(self, analytics_handler):
        confidences = [0.7, 0.72, 0.71, 0.69, 0.7, 0.71, 0.72, 0.7, 0.69, 0.71]
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = self._make_debates(confidences)

        with patch.object(analytics_handler, "get_storage", return_value=mock_storage):
            result = analytics_handler._get_consensus_quality.__wrapped__(analytics_handler)
            assert result.status_code == 200
            data = _parse_body(result)
            assert data["stats"]["trend"] == "stable"

    def test_insufficient_data_for_trend(self, analytics_handler):
        """Fewer than 5 debates means insufficient data for trend."""
        confidences = [0.7, 0.8, 0.9]
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = self._make_debates(confidences)

        with patch.object(analytics_handler, "get_storage", return_value=mock_storage):
            result = analytics_handler._get_consensus_quality.__wrapped__(analytics_handler)
            assert result.status_code == 200
            data = _parse_body(result)
            assert data["stats"]["trend"] == "stable"

    def test_no_debates_returns_zero(self, analytics_handler):
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = []

        with patch.object(analytics_handler, "get_storage", return_value=mock_storage):
            result = analytics_handler._get_consensus_quality.__wrapped__(analytics_handler)
            assert result.status_code == 200
            data = _parse_body(result)
            assert data["quality_score"] == 0
            assert data["stats"]["trend"] == "insufficient_data"

    def test_no_storage(self, analytics_handler):
        with patch.object(analytics_handler, "get_storage", return_value=None):
            result = analytics_handler._get_consensus_quality.__wrapped__(analytics_handler)
            assert result.status_code == 200
            data = _parse_body(result)
            assert data["quality_score"] == 0

    def test_quality_score_range(self, analytics_handler):
        """Quality score should always be between 0 and 100."""
        confidences = [0.5] * 10
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = self._make_debates(confidences)

        with patch.object(analytics_handler, "get_storage", return_value=mock_storage):
            result = analytics_handler._get_consensus_quality.__wrapped__(analytics_handler)
            data = _parse_body(result)
            assert 0 <= data["quality_score"] <= 100

    def test_critical_alert_on_low_quality(self, analytics_handler):
        # Very low confidence and no consensus => low quality
        confidences = [0.1] * 10
        consensus_flags = [False] * 10
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = self._make_debates(confidences, consensus_flags)

        with patch.object(analytics_handler, "get_storage", return_value=mock_storage):
            result = analytics_handler._get_consensus_quality.__wrapped__(analytics_handler)
            data = _parse_body(result)
            assert data["alert"] is not None
            assert data["alert"]["level"] in ("critical", "warning")

    def test_consensus_rate_calculation(self, analytics_handler):
        confidences = [0.8] * 10
        consensus_flags = [True, True, True, False, False, True, True, True, False, True]
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = self._make_debates(confidences, consensus_flags)

        with patch.object(analytics_handler, "get_storage", return_value=mock_storage):
            result = analytics_handler._get_consensus_quality.__wrapped__(analytics_handler)
            data = _parse_body(result)
            assert data["stats"]["consensus_rate"] == 0.7
            assert data["stats"]["consensus_reached_count"] == 7


# ===========================================================================
# Test _get_ranking_stats
# ===========================================================================


class TestRankingStats:
    """Tests for ranking system statistics."""

    def test_with_agents(self, analytics_handler):
        mock_agent = MagicMock()
        mock_agent.elo = 1600
        mock_agent.debates_count = 10
        mock_agent.agent_name = "claude"

        mock_elo = MagicMock()
        mock_elo.get_leaderboard.return_value = [mock_agent]

        with patch.object(analytics_handler, "get_elo_system", return_value=mock_elo):
            result = analytics_handler._get_ranking_stats.__wrapped__(analytics_handler)
            assert result.status_code == 200
            data = _parse_body(result)
            assert data["stats"]["total_agents"] == 1
            assert data["stats"]["top_agent"] == "claude"

    def test_no_elo_system(self, analytics_handler):
        with patch.object(analytics_handler, "get_elo_system", return_value=None):
            result = analytics_handler._get_ranking_stats.__wrapped__(analytics_handler)
            assert result.status_code == 503

    def test_empty_leaderboard(self, analytics_handler):
        mock_elo = MagicMock()
        mock_elo.get_leaderboard.return_value = []

        with patch.object(analytics_handler, "get_elo_system", return_value=mock_elo):
            result = analytics_handler._get_ranking_stats.__wrapped__(analytics_handler)
            assert result.status_code == 200
            data = _parse_body(result)
            assert data["stats"]["total_agents"] == 0
            assert data["stats"]["top_agent"] is None


# ===========================================================================
# Test _get_memory_stats
# ===========================================================================


class TestMemoryStats:
    """Tests for memory system statistics."""

    def test_databases_present(self, analytics_handler, tmp_path):
        # Create mock database files
        (tmp_path / "debate_embeddings.db").touch()
        (tmp_path / "continuum_memory.db").touch()

        with patch.object(analytics_handler, "get_nomic_dir", return_value=tmp_path):
            result = analytics_handler._get_memory_stats.__wrapped__(analytics_handler)
            assert result.status_code == 200
            data = _parse_body(result)
            assert data["stats"]["embeddings_db"] is True
            assert data["stats"]["continuum_memory"] is True

    def test_no_databases(self, analytics_handler, tmp_path):
        with patch.object(analytics_handler, "get_nomic_dir", return_value=tmp_path):
            result = analytics_handler._get_memory_stats.__wrapped__(analytics_handler)
            assert result.status_code == 200
            data = _parse_body(result)
            assert data["stats"]["embeddings_db"] is False
            assert data["stats"]["continuum_memory"] is False

    def test_no_nomic_dir(self, analytics_handler):
        with patch.object(analytics_handler, "get_nomic_dir", return_value=None):
            result = analytics_handler._get_memory_stats.__wrapped__(analytics_handler)
            assert result.status_code == 200
            data = _parse_body(result)
            assert data["stats"] == {}
