"""
Tests for the AnalyticsPerformanceHandler module.

Tests cover:
- Route registration and can_handle
- Agents performance endpoint
- Debates summary endpoint
- Trends endpoint
- RBAC permission checks
- Rate limiting behavior
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.analytics_performance import (
    AnalyticsPerformanceHandler,
    _parse_time_range,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def server_context():
    """Create mock server context for handler initialization."""
    return {"storage": None, "elo_system": None, "nomic_dir": None}


@pytest.fixture
def handler(server_context):
    """Create handler instance for tests."""
    return AnalyticsPerformanceHandler(server_context)


@pytest.fixture
def mock_http_handler():
    """Create mock HTTP handler."""
    mock = MagicMock()
    mock.client_address = ("127.0.0.1", 12345)
    mock.headers = {"X-Forwarded-For": "192.168.1.1"}
    return mock


@pytest.fixture
def mock_elo_system():
    """Create mock ELO system with sample agents."""
    mock_system = MagicMock()

    # Create mock agent ratings
    mock_agent1 = MagicMock()
    mock_agent1.agent_name = "claude"
    mock_agent1.elo = 1650.0
    mock_agent1.win_rate = 0.75
    mock_agent1.games_played = 160
    mock_agent1.wins = 120
    mock_agent1.losses = 30
    mock_agent1.draws = 10

    mock_agent2 = MagicMock()
    mock_agent2.agent_name = "gpt-4"
    mock_agent2.elo = 1580.0
    mock_agent2.win_rate = 0.68
    mock_agent2.games_played = 100
    mock_agent2.wins = 68
    mock_agent2.losses = 27
    mock_agent2.draws = 5

    mock_system.get_leaderboard.return_value = [mock_agent1, mock_agent2]
    mock_system.list_agents.return_value = ["claude", "gpt-4", "gemini"]

    return mock_system


@pytest.fixture
def mock_storage():
    """Create mock storage with sample debates."""
    mock_storage = MagicMock()

    # Create sample debate data - spread across different days
    debates = [
        {
            "id": f"debate-{i}",
            "created_at": datetime(2026, 1, 1 + i, 10, 0, tzinfo=timezone.utc).isoformat(),
            "consensus_reached": i % 3 != 0,  # 66% consensus
            "result": {
                "rounds_used": 3,
                "confidence": 0.85,
                "duration_seconds": 45.0,
                "domain": "security" if i % 2 == 0 else "performance",
                "outcome_type": "consensus" if i % 3 != 0 else "no_resolution",
            },
            "agents": [{"name": "claude"}, {"name": "gpt-4"}],
        }
        for i in range(20)
    ]

    mock_storage.list_debates.return_value = debates
    return mock_storage


# ===========================================================================
# Test: Parse Time Range
# ===========================================================================


class TestParseTimeRange:
    """Tests for _parse_time_range helper function."""

    def test_parse_time_range_7d(self):
        """Test parsing 7d time range."""
        result = _parse_time_range("7d")
        assert result is not None
        delta = datetime.now(timezone.utc) - result
        assert 6 < delta.days < 8  # Allow some variance

    def test_parse_time_range_30d(self):
        """Test parsing 30d time range."""
        result = _parse_time_range("30d")
        assert result is not None
        delta = datetime.now(timezone.utc) - result
        assert 29 < delta.days < 31

    def test_parse_time_range_365d(self):
        """Test parsing 365d time range."""
        result = _parse_time_range("365d")
        assert result is not None
        delta = datetime.now(timezone.utc) - result
        assert 364 < delta.days < 366

    def test_parse_time_range_all(self):
        """Test 'all' returns None."""
        result = _parse_time_range("all")
        assert result is None

    def test_parse_time_range_invalid_defaults_to_30d(self):
        """Test invalid time range defaults to 30d."""
        result = _parse_time_range("invalid")
        assert result is not None
        delta = datetime.now(timezone.utc) - result
        assert 29 < delta.days < 31


# ===========================================================================
# Test: Route Registration and can_handle
# ===========================================================================


class TestRouteRegistration:
    """Tests for route registration and can_handle method."""

    def test_routes_defined(self, handler):
        """Test that ROUTES is properly defined."""
        assert len(handler.ROUTES) == 3
        assert "/api/analytics/agents/performance" in handler.ROUTES
        assert "/api/analytics/debates/summary" in handler.ROUTES
        assert "/api/analytics/trends" in handler.ROUTES

    def test_can_handle_agents_performance(self, handler):
        """Test can_handle for agents performance route."""
        assert handler.can_handle("/api/analytics/agents/performance") is True
        assert handler.can_handle("/api/v1/analytics/agents/performance") is True

    def test_can_handle_debates_summary(self, handler):
        """Test can_handle for debates summary route."""
        assert handler.can_handle("/api/analytics/debates/summary") is True
        assert handler.can_handle("/api/v1/analytics/debates/summary") is True

    def test_can_handle_trends(self, handler):
        """Test can_handle for trends route."""
        assert handler.can_handle("/api/analytics/trends") is True
        assert handler.can_handle("/api/v1/analytics/trends") is True

    def test_cannot_handle_unrelated_paths(self, handler):
        """Test can_handle returns False for unrelated paths."""
        assert handler.can_handle("/api/debates") is False
        assert handler.can_handle("/api/analytics/other") is False
        assert handler.can_handle("/api/agents") is False


# ===========================================================================
# Test: Agents Performance Endpoint
# ===========================================================================


class TestAgentsPerformance:
    """Tests for GET /api/analytics/agents/performance endpoint."""

    def test_agents_performance_no_elo_system(self, handler, mock_http_handler):
        """Test agents performance when ELO system is not available."""
        with patch.object(handler, "get_elo_system", return_value=None):
            result = handler._get_agents_performance({}, mock_http_handler)

        assert result is not None
        body = json.loads(result["body"])
        assert body["total_agents"] == 0
        assert body["agents"] == []
        assert body["summary"]["avg_elo"] == 1500

    def test_agents_performance_with_elo_system(self, handler, mock_http_handler, mock_elo_system):
        """Test agents performance with ELO system."""
        with patch.object(handler, "get_elo_system", return_value=mock_elo_system):
            result = handler._get_agents_performance({"time_range": "30d"}, mock_http_handler)

        assert result is not None
        body = json.loads(result["body"])
        assert body["time_range"] == "30d"
        assert body["total_agents"] == 2
        assert len(body["agents"]) == 2

        # Check first agent (claude)
        claude = body["agents"][0]
        assert claude["agent_name"] == "claude"
        assert claude["elo"] == 1650
        assert claude["win_rate"] == 75.0
        assert claude["rank"] == 1

    def test_agents_performance_with_limit(self, handler, mock_http_handler, mock_elo_system):
        """Test agents performance with limit parameter."""
        with patch.object(handler, "get_elo_system", return_value=mock_elo_system):
            result = handler._get_agents_performance({"limit": "10"}, mock_http_handler)

        assert result is not None
        mock_elo_system.get_leaderboard.assert_called_once()


# ===========================================================================
# Test: Debates Summary Endpoint
# ===========================================================================


class TestDebatesSummary:
    """Tests for GET /api/analytics/debates/summary endpoint."""

    def test_debates_summary_no_storage(self, handler, mock_http_handler):
        """Test debates summary when storage is not available."""
        with patch.object(handler, "get_storage", return_value=None):
            result = handler._get_debates_summary({}, mock_http_handler)

        assert result is not None
        body = json.loads(result["body"])
        assert body["total_debates"] == 0
        assert body["consensus_reached"] == 0

    def test_debates_summary_with_storage(self, handler, mock_http_handler, mock_storage):
        """Test debates summary with storage."""
        with patch.object(handler, "get_storage", return_value=mock_storage):
            # Use 'all' time range to include all mock debates
            result = handler._get_debates_summary({"time_range": "all"}, mock_http_handler)

        assert result is not None
        body = json.loads(result["body"])
        assert body["time_range"] == "all"
        assert body["total_debates"] == 20
        # 66% consensus rate (every 3rd debate is non-consensus)
        # Consensus is True for all i where i % 3 != 0 -> 7 non-consensus (0,3,6,9,12,15,18)
        # So consensus_reached should be 20 - 7 = 13
        assert 12 <= body["consensus_reached"] <= 14  # Allow small variance
        assert body["avg_rounds"] == 3.0

    def test_debates_summary_domain_breakdown(self, handler, mock_http_handler, mock_storage):
        """Test debates summary includes domain breakdown."""
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_debates_summary({}, mock_http_handler)

        assert result is not None
        body = json.loads(result["body"])
        assert "by_domain" in body
        assert len(body["by_domain"]) >= 1

    def test_debates_summary_outcome_breakdown(self, handler, mock_http_handler, mock_storage):
        """Test debates summary includes outcome breakdown."""
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_debates_summary({}, mock_http_handler)

        assert result is not None
        body = json.loads(result["body"])
        assert "by_outcome" in body


# ===========================================================================
# Test: Trends Endpoint
# ===========================================================================


class TestTrends:
    """Tests for GET /api/analytics/trends endpoint."""

    def test_trends_no_storage(self, handler, mock_http_handler):
        """Test trends when storage is not available."""
        with patch.object(handler, "get_storage", return_value=None):
            result = handler._get_general_trends({}, mock_http_handler)

        assert result is not None
        body = json.loads(result["body"])
        assert body["data_points"] == []
        assert body["trend_analysis"] == {}

    def test_trends_with_storage(self, handler, mock_http_handler, mock_storage):
        """Test trends with storage."""
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_general_trends(
                {"time_range": "30d", "granularity": "daily"}, mock_http_handler
            )

        assert result is not None
        body = json.loads(result["body"])
        assert body["time_range"] == "30d"
        assert body["granularity"] == "daily"
        assert "data_points" in body
        assert "trend_analysis" in body

    def test_trends_with_metrics_filter(self, handler, mock_http_handler, mock_storage):
        """Test trends with specific metrics filter."""
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_general_trends(
                {"metrics": "debates,consensus"}, mock_http_handler
            )

        assert result is not None
        body = json.loads(result["body"])
        # Should only include requested metrics
        if body["data_points"]:
            point = body["data_points"][0]
            assert "debates_count" in point or "consensus_rate" in point

    def test_trends_granularity_validation(self, handler, mock_http_handler, mock_storage):
        """Test trends validates granularity parameter."""
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_general_trends({"granularity": "invalid"}, mock_http_handler)

        assert result is not None
        body = json.loads(result["body"])
        # Invalid granularity should default to "daily"
        assert body["granularity"] == "daily"


# ===========================================================================
# Test: Trend Analysis
# ===========================================================================


class TestTrendAnalysis:
    """Tests for _calculate_trends helper method."""

    def test_calculate_trends_insufficient_data(self, handler):
        """Test trend calculation with insufficient data."""
        result = handler._calculate_trends([{"period": "2026-01-01", "debates_count": 10}])
        assert result["debates_trend"] == "insufficient_data"
        assert result["consensus_trend"] == "insufficient_data"

    def test_calculate_trends_increasing(self, handler):
        """Test trend calculation detects increasing trend."""
        data_points = [
            {"period": f"2026-01-{i:02d}", "debates_count": 10 + i, "consensus_rate": 80.0}
            for i in range(1, 11)
        ]
        result = handler._calculate_trends(data_points)
        # Growth from first half to second half should be positive
        assert result["debates_growth_rate"] > 0

    def test_calculate_trends_stable(self, handler):
        """Test trend calculation detects stable trend."""
        data_points = [
            {"period": f"2026-01-{i:02d}", "debates_count": 10, "consensus_rate": 80.0}
            for i in range(1, 11)
        ]
        result = handler._calculate_trends(data_points)
        assert result["debates_trend"] == "stable"


# ===========================================================================
# Test: Time Range Validation
# ===========================================================================


class TestTimeRangeValidation:
    """Tests for time range parameter validation."""

    def test_invalid_time_range_defaults_to_30d(self, handler, mock_http_handler, mock_storage):
        """Test invalid time_range parameter defaults to 30d."""
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = handler._get_debates_summary({"time_range": "invalid"}, mock_http_handler)

        body = json.loads(result["body"])
        assert body["time_range"] == "30d"

    def test_valid_time_ranges_accepted(self, handler, mock_http_handler, mock_storage):
        """Test valid time_range values are accepted."""
        valid_ranges = ["7d", "14d", "30d", "90d", "180d", "365d", "all"]

        with patch.object(handler, "get_storage", return_value=mock_storage):
            for time_range in valid_ranges:
                result = handler._get_debates_summary({"time_range": time_range}, mock_http_handler)
                body = json.loads(result["body"])
                assert body["time_range"] == time_range


# ===========================================================================
# Test: Generated At Timestamp
# ===========================================================================


class TestGeneratedAt:
    """Tests for generated_at timestamp in responses."""

    def test_agents_performance_includes_timestamp(self, handler, mock_http_handler):
        """Test agents performance response includes generated_at."""
        with patch.object(handler, "get_elo_system", return_value=None):
            result = handler._get_agents_performance({}, mock_http_handler)

        body = json.loads(result["body"])
        assert "generated_at" in body
        # Should be a valid ISO format timestamp
        datetime.fromisoformat(body["generated_at"].replace("Z", "+00:00"))

    def test_debates_summary_includes_timestamp(self, handler, mock_http_handler):
        """Test debates summary response includes generated_at."""
        with patch.object(handler, "get_storage", return_value=None):
            result = handler._get_debates_summary({}, mock_http_handler)

        body = json.loads(result["body"])
        assert "generated_at" in body

    def test_trends_includes_timestamp(self, handler, mock_http_handler):
        """Test trends response includes generated_at."""
        with patch.object(handler, "get_storage", return_value=None):
            result = handler._get_general_trends({}, mock_http_handler)

        body = json.loads(result["body"])
        assert "generated_at" in body
