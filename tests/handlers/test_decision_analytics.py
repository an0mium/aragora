"""Tests for Decision Analytics handler.

Covers all routes and behavior of the DecisionAnalyticsHandler class:
- GET /api/v1/decision-analytics/overview       - Summary metrics
- GET /api/v1/decision-analytics/trends         - Quality over time
- GET /api/v1/decision-analytics/outcomes       - Decision list with outcomes
- GET /api/v1/decision-analytics/agents         - Per-agent quality metrics
- GET /api/v1/decision-analytics/domains        - Quality by domain

Issue: #281
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.decision_analytics import DecisionAnalyticsHandler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_data(response) -> dict:
    """Extract the 'data' envelope from an aiohttp json_response mock."""
    # web.json_response stores the data; we simulate by reading call args
    if hasattr(response, "body"):
        raw = json.loads(response.body)
    elif hasattr(response, "_json_data"):
        raw = response._json_data
    else:
        raw = {}
    if isinstance(raw, dict) and "data" in raw:
        return raw["data"]
    return raw


def _status(response) -> int:
    """Extract HTTP status code."""
    return getattr(response, "status", 200)


class FakeMultiDict(dict):
    """Minimal stand-in for aiohttp.multidict.CIMultiDictProxy."""

    def get(self, key, default=None):
        return super().get(key, default)


class FakeRequest:
    """Fake aiohttp.web.Request for testing handler methods."""

    def __init__(self, query: dict[str, str] | None = None):
        self.query = FakeMultiDict(query or {})
        self.match_info: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Mock data objects
# ---------------------------------------------------------------------------


@dataclass
class MockDebateStats:
    """Mock stats returned by DebateAnalytics.get_debate_stats()."""

    total_debates: int = 120
    consensus_reached: int = 96
    consensus_rate: float = 0.80
    avg_rounds: float = 3.2


@dataclass
class MockAgentLeaderboardEntry:
    """Mock entry from DebateAnalytics.get_agent_leaderboard()."""

    agent_id: str = "agent-001"
    agent_name: str = "claude"
    debates_participated: int = 40
    consensus_contributions: int = 32
    vote_ratio: float = 0.85


@dataclass
class MockQualityDataPoint:
    """Mock QualityDataPoint from outcome_analytics."""

    timestamp: datetime = field(
        default_factory=lambda: datetime(2026, 2, 1, tzinfo=timezone.utc)
    )
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


@dataclass
class MockAgentContribution:
    """Mock AgentContribution from outcome_analytics."""

    agent_id: str = "agent-001"
    agent_name: str = "claude"
    debates_participated: int = 40
    consensus_contributions: int = 32
    avg_confidence: float = 0.85
    contribution_score: float = 0.78

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "debates_participated": self.debates_participated,
            "consensus_contributions": self.consensus_contributions,
            "avg_confidence": round(self.avg_confidence, 4),
            "contribution_score": round(self.contribution_score, 4),
        }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_mock_analytics(
    consensus_rate: float = 0.80,
    avg_rounds: float = 3.2,
    stats: MockDebateStats | None = None,
    trend_points: list | None = None,
    contributions: dict | None = None,
    topics: dict[str, int] | None = None,
):
    """Create a mock OutcomeAnalytics instance."""
    analytics = AsyncMock()
    analytics.get_consensus_rate = AsyncMock(return_value=consensus_rate)
    analytics.get_average_rounds = AsyncMock(return_value=avg_rounds)

    # Trend points
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

    # Agent contributions
    if contributions is None:
        contributions = {
            "agent-001": MockAgentContribution(),
            "agent-002": MockAgentContribution(
                agent_id="agent-002",
                agent_name="gpt-4",
                debates_participated=30,
                consensus_contributions=24,
                avg_confidence=0.78,
                contribution_score=0.72,
            ),
        }
    analytics.get_agent_contribution_scores = AsyncMock(return_value=contributions)

    # Topics
    if topics is None:
        topics = {"architecture": 25, "security": 15, "general": 10}
    analytics.get_topic_distribution = AsyncMock(return_value=topics)

    # Debate analytics sub-object
    da = AsyncMock()
    da.get_debate_stats = AsyncMock(return_value=stats or MockDebateStats())
    da.db_path = ":memory:"
    analytics._get_debate_analytics = MagicMock(return_value=da)

    return analytics


@pytest.fixture
def handler():
    """Create a DecisionAnalyticsHandler instance."""
    return DecisionAnalyticsHandler(ctx={})


@pytest.fixture
def mock_analytics():
    """Create a default mock analytics and patch it in."""
    analytics = _make_mock_analytics()
    with patch(
        "aragora.server.handlers.decision_analytics._get_outcome_analytics",
        return_value=analytics,
    ):
        yield analytics


# ---------------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------------


class TestRoutes:
    """Test ROUTES class attribute."""

    def test_routes_contains_all_endpoints(self):
        expected_versioned = [
            "/api/v1/decision-analytics/overview",
            "/api/v1/decision-analytics/trends",
            "/api/v1/decision-analytics/outcomes",
            "/api/v1/decision-analytics/agents",
            "/api/v1/decision-analytics/domains",
        ]
        for route in expected_versioned:
            assert route in DecisionAnalyticsHandler.ROUTES

    def test_routes_contains_legacy_paths(self):
        expected_legacy = [
            "/api/decision-analytics/overview",
            "/api/decision-analytics/trends",
            "/api/decision-analytics/outcomes",
            "/api/decision-analytics/agents",
            "/api/decision-analytics/domains",
        ]
        for route in expected_legacy:
            assert route in DecisionAnalyticsHandler.ROUTES


# ---------------------------------------------------------------------------
# GET /api/v1/decision-analytics/overview
# ---------------------------------------------------------------------------


class TestOverview:
    """Tests for the overview endpoint."""

    @pytest.mark.asyncio
    async def test_overview_returns_data_envelope(self, handler, mock_analytics):
        request = FakeRequest(query={"period": "30d"})
        response = await handler.handle_get_overview(request)

        body = json.loads(response.body)
        assert "data" in body
        data = body["data"]
        assert data["total_decisions"] == 120
        assert data["consensus_reached"] == 96
        assert data["consensus_rate"] == 0.80
        assert data["avg_rounds"] == 3.2
        assert data["period"] == "30d"

    @pytest.mark.asyncio
    async def test_overview_default_period(self, handler, mock_analytics):
        request = FakeRequest()
        response = await handler.handle_get_overview(request)

        body = json.loads(response.body)
        assert body["data"]["period"] == "30d"

    @pytest.mark.asyncio
    async def test_overview_custom_period(self, handler, mock_analytics):
        request = FakeRequest(query={"period": "7d"})
        response = await handler.handle_get_overview(request)

        body = json.loads(response.body)
        assert body["data"]["period"] == "7d"
        mock_analytics.get_consensus_rate.assert_called_with(period="7d")

    @pytest.mark.asyncio
    async def test_overview_error_returns_500(self, handler):
        with patch(
            "aragora.server.handlers.decision_analytics._get_outcome_analytics",
            side_effect=RuntimeError("db down"),
        ):
            request = FakeRequest()
            response = await handler.handle_get_overview(request)
            assert response.status == 500

    @pytest.mark.asyncio
    async def test_overview_zero_debates(self, handler):
        analytics = _make_mock_analytics(
            consensus_rate=0.0,
            avg_rounds=0.0,
            stats=MockDebateStats(
                total_debates=0,
                consensus_reached=0,
                consensus_rate=0.0,
                avg_rounds=0.0,
            ),
        )
        with patch(
            "aragora.server.handlers.decision_analytics._get_outcome_analytics",
            return_value=analytics,
        ):
            request = FakeRequest()
            response = await handler.handle_get_overview(request)
            body = json.loads(response.body)
            assert body["data"]["total_decisions"] == 0
            assert body["data"]["consensus_rate"] == 0.0


# ---------------------------------------------------------------------------
# GET /api/v1/decision-analytics/trends
# ---------------------------------------------------------------------------


class TestTrends:
    """Tests for the trends endpoint."""

    @pytest.mark.asyncio
    async def test_trends_returns_data_envelope(self, handler, mock_analytics):
        request = FakeRequest(query={"period": "90d"})
        response = await handler.handle_get_trends(request)

        body = json.loads(response.body)
        assert "data" in body
        data = body["data"]
        assert data["period"] == "90d"
        assert data["count"] == 2
        assert len(data["points"]) == 2

    @pytest.mark.asyncio
    async def test_trends_point_structure(self, handler, mock_analytics):
        request = FakeRequest(query={"period": "90d"})
        response = await handler.handle_get_trends(request)

        body = json.loads(response.body)
        point = body["data"]["points"][0]
        assert "timestamp" in point
        assert "consensus_rate" in point
        assert "avg_confidence" in point
        assert "avg_rounds" in point
        assert "debate_count" in point

    @pytest.mark.asyncio
    async def test_trends_default_period(self, handler, mock_analytics):
        request = FakeRequest()
        response = await handler.handle_get_trends(request)

        body = json.loads(response.body)
        assert body["data"]["period"] == "90d"

    @pytest.mark.asyncio
    async def test_trends_empty_points(self, handler):
        analytics = _make_mock_analytics(trend_points=[])
        with patch(
            "aragora.server.handlers.decision_analytics._get_outcome_analytics",
            return_value=analytics,
        ):
            request = FakeRequest()
            response = await handler.handle_get_trends(request)
            body = json.loads(response.body)
            assert body["data"]["points"] == []
            assert body["data"]["count"] == 0


# ---------------------------------------------------------------------------
# GET /api/v1/decision-analytics/outcomes
# ---------------------------------------------------------------------------


class TestOutcomes:
    """Tests for the outcomes endpoint."""

    @pytest.mark.asyncio
    async def test_outcomes_returns_data_envelope(self, handler, mock_analytics):
        request = FakeRequest(query={"period": "30d", "limit": "10", "offset": "0"})
        response = await handler.handle_get_outcomes(request)

        body = json.loads(response.body)
        assert "data" in body
        data = body["data"]
        assert "outcomes" in data
        assert "total" in data
        assert "limit" in data
        assert "offset" in data
        assert data["period"] == "30d"

    @pytest.mark.asyncio
    async def test_outcomes_default_pagination(self, handler, mock_analytics):
        request = FakeRequest()
        response = await handler.handle_get_outcomes(request)

        body = json.loads(response.body)
        data = body["data"]
        assert data["limit"] == 50
        assert data["offset"] == 0

    @pytest.mark.asyncio
    async def test_outcomes_custom_limit(self, handler, mock_analytics):
        request = FakeRequest(query={"limit": "5"})
        response = await handler.handle_get_outcomes(request)

        body = json.loads(response.body)
        data = body["data"]
        assert data["limit"] == 5


# ---------------------------------------------------------------------------
# GET /api/v1/decision-analytics/agents
# ---------------------------------------------------------------------------


class TestAgents:
    """Tests for the agents endpoint."""

    @pytest.mark.asyncio
    async def test_agents_returns_data_envelope(self, handler, mock_analytics):
        request = FakeRequest(query={"period": "30d"})
        response = await handler.handle_get_agents(request)

        body = json.loads(response.body)
        assert "data" in body
        data = body["data"]
        assert "agents" in data
        assert data["count"] == 2
        assert data["period"] == "30d"

    @pytest.mark.asyncio
    async def test_agents_sorted_by_score(self, handler, mock_analytics):
        request = FakeRequest()
        response = await handler.handle_get_agents(request)

        body = json.loads(response.body)
        agents = body["data"]["agents"]
        assert len(agents) == 2
        # Higher score first
        assert agents[0]["contribution_score"] >= agents[1]["contribution_score"]

    @pytest.mark.asyncio
    async def test_agents_contains_expected_fields(self, handler, mock_analytics):
        request = FakeRequest()
        response = await handler.handle_get_agents(request)

        body = json.loads(response.body)
        agent = body["data"]["agents"][0]
        assert "agent_id" in agent
        assert "agent_name" in agent
        assert "debates_participated" in agent
        assert "consensus_contributions" in agent
        assert "avg_confidence" in agent
        assert "contribution_score" in agent

    @pytest.mark.asyncio
    async def test_agents_empty_list(self, handler):
        analytics = _make_mock_analytics(contributions={})
        with patch(
            "aragora.server.handlers.decision_analytics._get_outcome_analytics",
            return_value=analytics,
        ):
            request = FakeRequest()
            response = await handler.handle_get_agents(request)
            body = json.loads(response.body)
            assert body["data"]["agents"] == []
            assert body["data"]["count"] == 0


# ---------------------------------------------------------------------------
# GET /api/v1/decision-analytics/domains
# ---------------------------------------------------------------------------


class TestDomains:
    """Tests for the domains endpoint."""

    @pytest.mark.asyncio
    async def test_domains_returns_data_envelope(self, handler, mock_analytics):
        request = FakeRequest(query={"period": "30d"})
        response = await handler.handle_get_domains(request)

        body = json.loads(response.body)
        assert "data" in body
        data = body["data"]
        assert "domains" in data
        assert data["total_decisions"] == 50  # 25 + 15 + 10
        assert data["count"] == 3
        assert data["period"] == "30d"

    @pytest.mark.asyncio
    async def test_domains_sorted_by_count(self, handler, mock_analytics):
        request = FakeRequest()
        response = await handler.handle_get_domains(request)

        body = json.loads(response.body)
        domains = body["data"]["domains"]
        assert domains[0]["domain"] == "architecture"
        assert domains[0]["decision_count"] == 25
        assert domains[1]["domain"] == "security"

    @pytest.mark.asyncio
    async def test_domains_percentages(self, handler, mock_analytics):
        request = FakeRequest()
        response = await handler.handle_get_domains(request)

        body = json.loads(response.body)
        domains = body["data"]["domains"]
        total_pct = sum(d["percentage"] for d in domains)
        assert 99.0 <= total_pct <= 101.0  # Allow rounding

    @pytest.mark.asyncio
    async def test_domains_empty(self, handler):
        analytics = _make_mock_analytics(topics={})
        with patch(
            "aragora.server.handlers.decision_analytics._get_outcome_analytics",
            return_value=analytics,
        ):
            request = FakeRequest()
            response = await handler.handle_get_domains(request)
            body = json.loads(response.body)
            assert body["data"]["domains"] == []
            assert body["data"]["total_decisions"] == 0

    @pytest.mark.asyncio
    async def test_domains_contains_expected_fields(self, handler, mock_analytics):
        request = FakeRequest()
        response = await handler.handle_get_domains(request)

        body = json.loads(response.body)
        domain = body["data"]["domains"][0]
        assert "domain" in domain
        assert "decision_count" in domain
        assert "percentage" in domain


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Test error paths across all endpoints."""

    @pytest.mark.asyncio
    async def test_overview_handles_value_error(self, handler):
        with patch(
            "aragora.server.handlers.decision_analytics._get_outcome_analytics",
            side_effect=ValueError("bad period"),
        ):
            request = FakeRequest()
            response = await handler.handle_get_overview(request)
            assert response.status == 500

    @pytest.mark.asyncio
    async def test_trends_handles_import_error(self, handler):
        with patch(
            "aragora.server.handlers.decision_analytics._get_outcome_analytics",
            side_effect=ImportError("no module"),
        ):
            request = FakeRequest()
            response = await handler.handle_get_trends(request)
            assert response.status == 500

    @pytest.mark.asyncio
    async def test_agents_handles_key_error(self, handler):
        with patch(
            "aragora.server.handlers.decision_analytics._get_outcome_analytics",
            side_effect=KeyError("missing"),
        ):
            request = FakeRequest()
            response = await handler.handle_get_agents(request)
            assert response.status == 500

    @pytest.mark.asyncio
    async def test_domains_handles_type_error(self, handler):
        with patch(
            "aragora.server.handlers.decision_analytics._get_outcome_analytics",
            side_effect=TypeError("wrong type"),
        ):
            request = FakeRequest()
            response = await handler.handle_get_domains(request)
            assert response.status == 500

    @pytest.mark.asyncio
    async def test_no_str_e_in_error_responses(self, handler):
        """Verify error responses do not leak exception messages (no str(e))."""
        with patch(
            "aragora.server.handlers.decision_analytics._get_outcome_analytics",
            side_effect=RuntimeError("secret details here"),
        ):
            request = FakeRequest()
            response = await handler.handle_get_overview(request)
            body_text = response.body.decode() if isinstance(response.body, bytes) else str(response.body)
            assert "secret details here" not in body_text
            assert response.status == 500


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------


class TestInit:
    """Test handler initialization."""

    def test_default_ctx(self):
        h = DecisionAnalyticsHandler()
        assert h.ctx == {}

    def test_custom_ctx(self):
        ctx = {"db_path": "/tmp/test.db"}
        h = DecisionAnalyticsHandler(ctx=ctx)
        assert h.ctx == ctx
