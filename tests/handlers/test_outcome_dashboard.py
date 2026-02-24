"""Tests for Outcome Dashboard handler.

Covers all routes and behaviour of the OutcomeDashboardHandler class:
- GET /api/v1/outcome-dashboard            - Full dashboard data
- GET /api/v1/outcome-dashboard/quality    - Decision quality score
- GET /api/v1/outcome-dashboard/agents     - Agent leaderboard
- GET /api/v1/outcome-dashboard/history    - Decision history
- GET /api/v1/outcome-dashboard/calibration- Calibration curve data

Issue: #281
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.outcome_dashboard import OutcomeDashboardHandler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_response(result) -> dict:
    """Extract data from json_response tuple (body, status, headers)."""
    # Handler methods typically return HandlerResult from json_response().
    if hasattr(result, "body"):
        body = result.body
        if isinstance(body, (bytes, bytearray)):
            body = body.decode("utf-8")
        if isinstance(body, str):
            body = json.loads(body)
        if isinstance(body, dict):
            return body
    if isinstance(result, tuple):
        body = result[0] if len(result) > 0 else {}
        if isinstance(body, str):
            body = json.loads(body)
        return body
    if isinstance(result, dict):
        return result
    return {}


def _get_data(result) -> dict:
    """Extract the 'data' envelope from a response."""
    body = _parse_response(result)
    if isinstance(body, dict) and "data" in body:
        return body["data"]
    return body


# ---------------------------------------------------------------------------
# Mock data objects
# ---------------------------------------------------------------------------


@dataclass
class MockDebateStats:
    """Mock stats from DebateAnalytics."""

    total_debates: int = 100
    completed_debates: int = 95
    failed_debates: int = 5
    consensus_reached: int = 80
    consensus_rate: float = 0.842
    avg_rounds: float = 3.5
    avg_duration_seconds: float = 120.0
    total_messages: int = 500
    total_votes: int = 200


@dataclass
class MockAgentPerformance:
    """Mock agent performance entry."""

    agent_id: str = "agent-001"
    agent_name: str = "claude"
    provider: str = "anthropic"
    model: str = "claude-3-opus"
    debates_participated: int = 50
    messages_sent: int = 200
    avg_response_time_ms: float = 1500.0
    p95_response_time_ms: float = 3000.0
    p99_response_time_ms: float = 5000.0
    error_count: int = 2
    error_rate: float = 0.01
    votes_received: int = 180
    positive_votes: int = 150
    vote_ratio: float = 0.833
    consensus_contributions: int = 40
    total_tokens_in: int = 50000
    total_tokens_out: int = 20000
    current_elo: float = 1650.0
    elo_change_period: float = 25.0
    rank: int = 1


@dataclass
class MockQualityDataPoint:
    """Mock quality trend data point."""

    timestamp: datetime = field(default_factory=lambda: datetime(2026, 2, 1, tzinfo=timezone.utc))
    consensus_rate: float = 0.82
    avg_confidence: float = 0.80
    avg_rounds: float = 3.1
    debate_count: int = 15

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "consensus_rate": round(self.consensus_rate, 4),
            "avg_confidence": round(self.avg_confidence, 4),
            "avg_rounds": round(self.avg_rounds, 2),
            "debate_count": self.debate_count,
        }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_mock_outcome_analytics(
    consensus_rate: float = 0.80,
    avg_rounds: float = 3.2,
    stats: MockDebateStats | None = None,
    trend_points: list | None = None,
):
    """Create a mock OutcomeAnalytics instance."""
    analytics = AsyncMock()
    analytics.get_consensus_rate = AsyncMock(return_value=consensus_rate)
    analytics.get_average_rounds = AsyncMock(return_value=avg_rounds)

    if trend_points is None:
        trend_points = [
            MockQualityDataPoint(
                timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
                consensus_rate=0.75,
                debate_count=10,
            ),
            MockQualityDataPoint(
                timestamp=datetime(2026, 1, 8, tzinfo=timezone.utc),
                consensus_rate=0.82,
                debate_count=15,
            ),
        ]
    analytics.get_decision_quality_trend = AsyncMock(return_value=trend_points)

    return analytics


def _make_mock_debate_analytics(stats: MockDebateStats | None = None):
    """Create a mock DebateAnalytics instance."""
    da = AsyncMock()
    da.get_debate_stats = AsyncMock(return_value=stats or MockDebateStats())
    da.get_agent_leaderboard = AsyncMock(
        return_value=[
            MockAgentPerformance(),
            MockAgentPerformance(
                agent_id="agent-002",
                agent_name="gpt-4",
                provider="openai",
                model="gpt-4-turbo",
                current_elo=1620.0,
                elo_change_period=-10.0,
                rank=2,
            ),
        ]
    )
    da.db_path = ":memory:"
    return da


@pytest.fixture
def handler():
    """Create an OutcomeDashboardHandler instance."""
    return OutcomeDashboardHandler(ctx={})


@pytest.fixture
def mock_backends():
    """Patch both outcome and debate analytics."""
    oa = _make_mock_outcome_analytics()
    da = _make_mock_debate_analytics()

    with (
        patch(
            "aragora.server.handlers.outcome_dashboard._get_outcome_analytics",
            return_value=oa,
        ),
        patch(
            "aragora.server.handlers.outcome_dashboard._get_debate_analytics",
            return_value=da,
        ),
        patch(
            "aragora.server.handlers.outcome_dashboard._parse_period",
            return_value=timedelta(days=30),
        ),
    ):
        yield {"outcome_analytics": oa, "debate_analytics": da}


# ---------------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------------


class TestRoutes:
    """Test ROUTES class attribute."""

    def test_routes_contains_all_endpoints(self):
        expected = [
            "/api/v1/outcome-dashboard",
            "/api/v1/outcome-dashboard/quality",
            "/api/v1/outcome-dashboard/agents",
            "/api/v1/outcome-dashboard/history",
            "/api/v1/outcome-dashboard/calibration",
        ]
        for route in expected:
            assert route in OutcomeDashboardHandler.ROUTES, f"Missing route: {route}"

    def test_can_handle_valid_paths(self, handler):
        for route in OutcomeDashboardHandler.ROUTES:
            assert handler.can_handle(route), f"Should handle: {route}"

    def test_can_handle_rejects_unknown_paths(self, handler):
        assert not handler.can_handle("/api/v1/outcome-dashboard/unknown")
        assert not handler.can_handle("/api/v1/other")


# ---------------------------------------------------------------------------
# GET /api/v1/outcome-dashboard/quality
# ---------------------------------------------------------------------------


class TestQualityScore:
    """Test the quality score endpoint."""

    @pytest.mark.asyncio
    async def test_quality_score_returns_data(self, handler, mock_backends):
        result = await handler._build_quality_score("30d")

        assert isinstance(result, dict)
        assert "quality_score" in result
        assert "consensus_rate" in result
        assert "avg_rounds" in result
        assert "total_decisions" in result
        assert "completion_rate" in result
        assert "trend" in result
        assert result["period"] == "30d"

    @pytest.mark.asyncio
    async def test_quality_score_range(self, handler, mock_backends):
        result = await handler._build_quality_score("30d")

        assert 0 <= result["quality_score"] <= 100
        assert 0 <= result["consensus_rate"] <= 1
        assert 0 <= result["completion_rate"] <= 1

    @pytest.mark.asyncio
    async def test_quality_score_trend_populated(self, handler, mock_backends):
        result = await handler._build_quality_score("30d")

        assert isinstance(result["trend"], list)
        assert len(result["trend"]) >= 1


# ---------------------------------------------------------------------------
# GET /api/v1/outcome-dashboard/agents
# ---------------------------------------------------------------------------


class TestAgentLeaderboard:
    """Test the agent leaderboard endpoint."""

    @pytest.mark.asyncio
    async def test_agent_leaderboard_returns_agents(self, handler, mock_backends):
        result = await handler._build_agent_leaderboard("30d")

        assert isinstance(result, dict)
        assert "agents" in result
        assert "count" in result
        assert result["count"] == 2

    @pytest.mark.asyncio
    async def test_agent_leaderboard_entry_shape(self, handler, mock_backends):
        result = await handler._build_agent_leaderboard("30d")

        agent = result["agents"][0]
        expected_keys = [
            "rank",
            "agent_id",
            "agent_name",
            "provider",
            "model",
            "elo",
            "elo_change",
            "debates",
            "win_rate",
            "brier_score",
            "calibration_accuracy",
            "calibration_count",
        ]
        for key in expected_keys:
            assert key in agent, f"Missing key: {key}"

    @pytest.mark.asyncio
    async def test_agent_leaderboard_elo_values(self, handler, mock_backends):
        result = await handler._build_agent_leaderboard("30d")

        # First agent should be claude with highest ELO
        assert result["agents"][0]["agent_name"] == "claude"
        assert result["agents"][0]["elo"] == 1650.0


# ---------------------------------------------------------------------------
# GET /api/v1/outcome-dashboard/history
# ---------------------------------------------------------------------------


class TestDecisionHistory:
    """Test the decision history endpoint."""

    @pytest.mark.asyncio
    async def test_history_returns_structure(self, handler, mock_backends):
        result = await handler._build_decision_history("30d", limit=50, offset=0)

        assert isinstance(result, dict)
        assert "decisions" in result
        assert "total" in result
        assert "limit" in result
        assert "offset" in result
        assert result["limit"] == 50

    @pytest.mark.asyncio
    async def test_history_empty_when_no_db(self, handler, mock_backends):
        """Should return empty list when no sqlite data available."""
        result = await handler._build_decision_history("30d")

        # With :memory: db and no inserts, decisions should be empty
        assert isinstance(result["decisions"], list)


# ---------------------------------------------------------------------------
# GET /api/v1/outcome-dashboard/calibration
# ---------------------------------------------------------------------------


class TestCalibrationCurve:
    """Test the calibration curve endpoint."""

    @pytest.mark.asyncio
    async def test_calibration_returns_structure(self, handler, mock_backends):
        result = await handler._build_calibration_curve("30d")

        assert isinstance(result, dict)
        assert "points" in result
        assert "total_observations" in result
        assert result["period"] == "30d"

    @pytest.mark.asyncio
    async def test_calibration_has_buckets(self, handler, mock_backends):
        result = await handler._build_calibration_curve("30d")

        assert isinstance(result["points"], list)
        # Should have 10 buckets (0.0-0.1 through 0.9-1.0)
        assert len(result["points"]) == 10

    @pytest.mark.asyncio
    async def test_calibration_point_shape(self, handler, mock_backends):
        result = await handler._build_calibration_curve("30d")

        point = result["points"][0]
        assert "bucket" in point
        assert "predicted" in point
        assert "actual" in point
        assert "count" in point


# ---------------------------------------------------------------------------
# GET /api/v1/outcome-dashboard (full dashboard)
# ---------------------------------------------------------------------------


class TestFullDashboard:
    """Test the full consolidated dashboard endpoint."""

    @pytest.mark.asyncio
    async def test_full_dashboard_has_all_sections(self, handler, mock_backends):
        result = await handler._get_full_dashboard({"period": "30d"})

        body = _parse_response(result)
        data = body.get("data", body)

        assert "quality" in data
        assert "agents" in data
        assert "history" in data
        assert "calibration" in data
        assert data["period"] == "30d"

    @pytest.mark.asyncio
    async def test_full_dashboard_default_period(self, handler, mock_backends):
        result = await handler._get_full_dashboard({})

        body = _parse_response(result)
        data = body.get("data", body)

        assert data["period"] == "30d"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Test graceful degradation when backends fail."""

    @pytest.mark.asyncio
    async def test_quality_score_fallback_on_error(self, handler):
        """Quality score should return zeros when analytics is unavailable."""
        with patch(
            "aragora.server.handlers.outcome_dashboard._get_outcome_analytics",
            side_effect=ImportError("not available"),
        ):
            result = await handler._build_quality_score("30d")

        assert result["quality_score"] == 0.0
        assert result["total_decisions"] == 0
        assert result["trend"] == []

    @pytest.mark.asyncio
    async def test_agent_leaderboard_fallback_on_error(self, handler):
        """Agent leaderboard should return empty list when backend fails."""
        with (
            patch(
                "aragora.server.handlers.outcome_dashboard._parse_period",
                return_value=timedelta(days=30),
            ),
            patch(
                "aragora.server.handlers.outcome_dashboard._get_debate_analytics",
                side_effect=ImportError("not available"),
            ),
        ):
            result = await handler._build_agent_leaderboard("30d")

        assert result["agents"] == []
        assert result["count"] == 0
