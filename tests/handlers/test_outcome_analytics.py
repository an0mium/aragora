"""Tests for Outcome Analytics handler.

Covers all routes and behaviour of the OutcomeAnalyticsHandler class:
- GET /api/analytics/outcomes                    - Full outcome analytics summary
- GET /api/analytics/outcomes/consensus-rate     - Consensus rate for period
- GET /api/analytics/outcomes/average-rounds     - Mean rounds to conclusion
- GET /api/analytics/outcomes/contributions      - Agent contribution scores
- GET /api/analytics/outcomes/quality-trend      - Decision quality over time
- GET /api/analytics/outcomes/topics             - Topic distribution
- GET /api/analytics/outcomes/{debate_id}        - Single debate outcome summary
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.outcome_analytics import (
    OUTCOME_ANALYTICS_PERMISSION,
    OutcomeAnalyticsHandler,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_response(result) -> dict:
    """Extract data from json_response HandlerResult."""
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


def _status_code(result) -> int:
    """Extract status code from HandlerResult."""
    if hasattr(result, "status_code"):
        return result.status_code
    if isinstance(result, tuple) and len(result) > 1:
        return result[1]
    return 200


# ---------------------------------------------------------------------------
# Mock data objects
# ---------------------------------------------------------------------------


@dataclass
class MockContribution:
    """Mock agent contribution scores."""

    proposal_count: int = 10
    critique_count: int = 5
    influence_score: float = 0.85
    consensus_contributions: int = 7

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_count": self.proposal_count,
            "critique_count": self.critique_count,
            "influence_score": self.influence_score,
            "consensus_contributions": self.consensus_contributions,
        }


@dataclass
class MockQualityPoint:
    """Mock quality trend data point."""

    date: str = "2026-02-01"
    quality_score: float = 0.82
    debate_count: int = 15

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "quality_score": self.quality_score,
            "debate_count": self.debate_count,
        }


@dataclass
class MockOutcomeSummary:
    """Mock single debate outcome summary."""

    debate_id: str = "debate-001"
    topic: str = "API rate limiter design"
    consensus_reached: bool = True
    rounds: int = 3
    agents: int = 5
    quality_score: float = 0.88

    def to_dict(self) -> dict[str, Any]:
        return {
            "debate_id": self.debate_id,
            "topic": self.topic,
            "consensus_reached": self.consensus_reached,
            "rounds": self.rounds,
            "agents": self.agents,
            "quality_score": self.quality_score,
        }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_mock_analytics(
    consensus_rate: float = 0.82,
    avg_rounds: float = 3.5,
    topics: list | None = None,
    contributions: dict | None = None,
    trend: list | None = None,
):
    """Create a mock outcome analytics instance."""
    analytics = AsyncMock()
    analytics.get_consensus_rate = AsyncMock(return_value=consensus_rate)
    analytics.get_average_rounds = AsyncMock(return_value=avg_rounds)
    analytics.get_topic_distribution = AsyncMock(
        return_value=topics
        or [
            {"topic": "architecture", "count": 10},
            {"topic": "security", "count": 7},
        ]
    )
    analytics.get_agent_contribution_scores = AsyncMock(
        return_value=contributions
        or {
            "agent-001": MockContribution(),
            "agent-002": MockContribution(proposal_count=8, influence_score=0.72),
        }
    )
    analytics.get_decision_quality_trend = AsyncMock(
        return_value=trend
        or [
            MockQualityPoint(),
            MockQualityPoint(date="2026-02-08", quality_score=0.85),
        ]
    )
    analytics.get_outcome_summary = AsyncMock(return_value=MockOutcomeSummary())
    return analytics


@pytest.fixture
def handler():
    """Create an OutcomeAnalyticsHandler instance."""
    return OutcomeAnalyticsHandler(ctx={})


@pytest.fixture
def mock_analytics():
    """Patch outcome analytics backend."""
    analytics = _make_mock_analytics()
    with patch(
        "aragora.analytics.outcome_analytics.get_outcome_analytics",
        return_value=analytics,
        create=True,
    ):
        yield analytics


@pytest.fixture
def mock_rate_limiter():
    """Bypass rate limiter."""
    with patch("aragora.server.handlers.outcome_analytics._outcome_analytics_limiter") as limiter:
        limiter.is_allowed.return_value = True
        yield limiter


@pytest.fixture
def mock_handler():
    """Create a mock HTTP handler with client info."""
    h = MagicMock()
    h.client_address = ("127.0.0.1", 12345)
    h.headers = {"Authorization": "Bearer test-token"}
    return h


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


class TestRoutes:
    """Test ROUTES and can_handle."""

    def test_routes_contains_all_endpoints(self):
        expected = [
            "/api/analytics/outcomes",
            "/api/analytics/outcomes/consensus-rate",
            "/api/analytics/outcomes/average-rounds",
            "/api/analytics/outcomes/contributions",
            "/api/analytics/outcomes/quality-trend",
            "/api/analytics/outcomes/topics",
        ]
        for route in expected:
            assert route in OutcomeAnalyticsHandler.ROUTES, f"Missing route: {route}"

    def test_can_handle_main_endpoint(self, handler):
        assert handler.can_handle("/api/analytics/outcomes")

    def test_can_handle_consensus_rate(self, handler):
        assert handler.can_handle("/api/analytics/outcomes/consensus-rate")

    def test_can_handle_average_rounds(self, handler):
        assert handler.can_handle("/api/analytics/outcomes/average-rounds")

    def test_can_handle_contributions(self, handler):
        assert handler.can_handle("/api/analytics/outcomes/contributions")

    def test_can_handle_quality_trend(self, handler):
        assert handler.can_handle("/api/analytics/outcomes/quality-trend")

    def test_can_handle_topics(self, handler):
        assert handler.can_handle("/api/analytics/outcomes/topics")

    def test_can_handle_debate_id(self, handler):
        assert handler.can_handle("/api/analytics/outcomes/debate-abc-123")

    def test_can_handle_with_version_prefix(self, handler):
        assert handler.can_handle("/api/v1/analytics/outcomes")
        assert handler.can_handle("/api/v1/analytics/outcomes/consensus-rate")

    def test_can_handle_rejects_unknown(self, handler):
        assert not handler.can_handle("/api/analytics/other")
        assert not handler.can_handle("/api/other/outcomes")

    def test_known_suffixes_not_treated_as_debate_ids(self, handler):
        """Known sub-routes should not match as debate IDs."""
        # They should be handled as regular routes, not debate ID lookups
        for suffix in OutcomeAnalyticsHandler._KNOWN_SUFFIXES:
            path = f"/api/analytics/outcomes/{suffix}"
            # can_handle should return True (routes match)
            assert handler.can_handle(path), f"Should handle: {path}"


# ---------------------------------------------------------------------------
# Permission constant
# ---------------------------------------------------------------------------


class TestPermission:
    """Test permission constant."""

    def test_permission_value(self):
        assert OUTCOME_ANALYTICS_PERMISSION == "analytics:read"


# ---------------------------------------------------------------------------
# GET /api/analytics/outcomes (summary)
# ---------------------------------------------------------------------------


class TestOutcomesSummary:
    """Test full outcome analytics summary endpoint."""

    @pytest.mark.asyncio
    async def test_summary_returns_data(
        self, handler, mock_analytics, mock_rate_limiter, mock_handler
    ):
        result = await handler._get_outcomes_summary({"period": "30d"})

        data = _get_data(result)
        assert "consensus_rate" in data
        assert "average_rounds" in data
        assert "topic_distribution" in data
        assert data["period"] == "30d"

    @pytest.mark.asyncio
    async def test_summary_default_period(
        self, handler, mock_analytics, mock_rate_limiter, mock_handler
    ):
        result = await handler._get_outcomes_summary({})

        data = _get_data(result)
        assert data["period"] == "30d"

    @pytest.mark.asyncio
    async def test_summary_rounds_values(
        self, handler, mock_analytics, mock_rate_limiter, mock_handler
    ):
        result = await handler._get_outcomes_summary({"period": "30d"})

        data = _get_data(result)
        # Values should be rounded
        assert isinstance(data["consensus_rate"], float)
        assert isinstance(data["average_rounds"], float)


# ---------------------------------------------------------------------------
# GET /api/analytics/outcomes/consensus-rate
# ---------------------------------------------------------------------------


class TestConsensusRate:
    """Test consensus rate endpoint."""

    @pytest.mark.asyncio
    async def test_consensus_rate_returns_data(self, handler, mock_analytics, mock_rate_limiter):
        result = await handler._get_consensus_rate({"period": "7d"})

        data = _get_data(result)
        assert "consensus_rate" in data
        assert data["period"] == "7d"

    @pytest.mark.asyncio
    async def test_consensus_rate_default_period(self, handler, mock_analytics, mock_rate_limiter):
        result = await handler._get_consensus_rate({})

        data = _get_data(result)
        assert data["period"] == "30d"

    @pytest.mark.asyncio
    async def test_consensus_rate_value_is_rounded(self, handler, mock_rate_limiter):
        analytics = _make_mock_analytics(consensus_rate=0.823456789)
        with patch(
            "aragora.analytics.outcome_analytics.get_outcome_analytics",
            return_value=analytics,
            create=True,
        ):
            result = await handler._get_consensus_rate({})

        data = _get_data(result)
        # Should be rounded to 4 decimal places
        assert data["consensus_rate"] == round(0.823456789, 4)


# ---------------------------------------------------------------------------
# GET /api/analytics/outcomes/average-rounds
# ---------------------------------------------------------------------------


class TestAverageRounds:
    """Test average rounds endpoint."""

    @pytest.mark.asyncio
    async def test_average_rounds_returns_data(self, handler, mock_analytics, mock_rate_limiter):
        result = await handler._get_average_rounds({"period": "30d"})

        data = _get_data(result)
        assert "average_rounds" in data
        assert data["period"] == "30d"

    @pytest.mark.asyncio
    async def test_average_rounds_default_period(self, handler, mock_analytics, mock_rate_limiter):
        result = await handler._get_average_rounds({})

        data = _get_data(result)
        assert data["period"] == "30d"

    @pytest.mark.asyncio
    async def test_average_rounds_value_is_rounded(self, handler, mock_rate_limiter):
        analytics = _make_mock_analytics(avg_rounds=3.456789)
        with patch(
            "aragora.analytics.outcome_analytics.get_outcome_analytics",
            return_value=analytics,
            create=True,
        ):
            result = await handler._get_average_rounds({})

        data = _get_data(result)
        assert data["average_rounds"] == round(3.456789, 2)


# ---------------------------------------------------------------------------
# GET /api/analytics/outcomes/contributions
# ---------------------------------------------------------------------------


class TestContributions:
    """Test agent contributions endpoint."""

    @pytest.mark.asyncio
    async def test_contributions_returns_data(self, handler, mock_analytics, mock_rate_limiter):
        result = await handler._get_contributions({"period": "30d"})

        data = _get_data(result)
        assert "contributions" in data
        assert data["period"] == "30d"

    @pytest.mark.asyncio
    async def test_contributions_has_agent_entries(
        self, handler, mock_analytics, mock_rate_limiter
    ):
        result = await handler._get_contributions({})

        data = _get_data(result)
        assert "agent-001" in data["contributions"]
        assert "agent-002" in data["contributions"]

    @pytest.mark.asyncio
    async def test_contributions_entry_shape(self, handler, mock_analytics, mock_rate_limiter):
        result = await handler._get_contributions({})

        data = _get_data(result)
        contrib = data["contributions"]["agent-001"]
        assert "proposal_count" in contrib
        assert "critique_count" in contrib
        assert "influence_score" in contrib


# ---------------------------------------------------------------------------
# GET /api/analytics/outcomes/quality-trend
# ---------------------------------------------------------------------------


class TestQualityTrend:
    """Test quality trend endpoint."""

    @pytest.mark.asyncio
    async def test_quality_trend_returns_data(self, handler, mock_analytics, mock_rate_limiter):
        result = await handler._get_quality_trend({"period": "90d"})

        data = _get_data(result)
        assert "trend" in data
        assert data["period"] == "90d"

    @pytest.mark.asyncio
    async def test_quality_trend_default_period(self, handler, mock_analytics, mock_rate_limiter):
        result = await handler._get_quality_trend({})

        data = _get_data(result)
        assert data["period"] == "90d"

    @pytest.mark.asyncio
    async def test_quality_trend_has_points(self, handler, mock_analytics, mock_rate_limiter):
        result = await handler._get_quality_trend({})

        data = _get_data(result)
        assert isinstance(data["trend"], list)
        assert len(data["trend"]) >= 1

    @pytest.mark.asyncio
    async def test_quality_trend_point_shape(self, handler, mock_analytics, mock_rate_limiter):
        result = await handler._get_quality_trend({})

        data = _get_data(result)
        point = data["trend"][0]
        assert "date" in point
        assert "quality_score" in point
        assert "debate_count" in point


# ---------------------------------------------------------------------------
# GET /api/analytics/outcomes/topics
# ---------------------------------------------------------------------------


class TestTopics:
    """Test topic distribution endpoint."""

    @pytest.mark.asyncio
    async def test_topics_returns_data(self, handler, mock_analytics, mock_rate_limiter):
        result = await handler._get_topics({"period": "30d"})

        data = _get_data(result)
        assert "topics" in data
        assert data["period"] == "30d"

    @pytest.mark.asyncio
    async def test_topics_has_entries(self, handler, mock_analytics, mock_rate_limiter):
        result = await handler._get_topics({})

        data = _get_data(result)
        assert isinstance(data["topics"], list)
        assert len(data["topics"]) >= 1

    @pytest.mark.asyncio
    async def test_topics_entry_shape(self, handler, mock_analytics, mock_rate_limiter):
        result = await handler._get_topics({})

        data = _get_data(result)
        topic = data["topics"][0]
        assert "topic" in topic
        assert "count" in topic


# ---------------------------------------------------------------------------
# GET /api/analytics/outcomes/{debate_id}
# ---------------------------------------------------------------------------


class TestDebateOutcome:
    """Test single debate outcome endpoint."""

    @pytest.mark.asyncio
    async def test_debate_outcome_returns_data(self, handler, mock_analytics, mock_rate_limiter):
        result = await handler._get_debate_outcome("debate-001")

        data = _get_data(result)
        assert "debate_id" in data
        assert data["debate_id"] == "debate-001"

    @pytest.mark.asyncio
    async def test_debate_outcome_includes_fields(self, handler, mock_analytics, mock_rate_limiter):
        result = await handler._get_debate_outcome("debate-001")

        data = _get_data(result)
        assert "topic" in data
        assert "consensus_reached" in data
        assert "rounds" in data
        assert "quality_score" in data

    @pytest.mark.asyncio
    async def test_debate_outcome_not_found(self, handler, mock_rate_limiter):
        analytics = _make_mock_analytics()
        analytics.get_outcome_summary = AsyncMock(return_value=None)

        with patch(
            "aragora.analytics.outcome_analytics.get_outcome_analytics",
            return_value=analytics,
            create=True,
        ):
            result = await handler._get_debate_outcome("nonexistent-debate")

        assert _status_code(result) == 404
        body = _parse_response(result)
        assert "not found" in str(body).lower() or "NOT_FOUND" in str(body)


# ---------------------------------------------------------------------------
# Handle method (full dispatch)
# ---------------------------------------------------------------------------


class TestHandleDispatch:
    """Test the main handle() dispatch method."""

    @pytest.mark.asyncio
    async def test_dispatch_to_summary(
        self, handler, mock_analytics, mock_rate_limiter, mock_handler
    ):
        result = await handler.handle("/api/analytics/outcomes", {}, mock_handler)

        assert result is not None
        data = _get_data(result)
        assert "consensus_rate" in data

    @pytest.mark.asyncio
    async def test_dispatch_to_consensus_rate(
        self, handler, mock_analytics, mock_rate_limiter, mock_handler
    ):
        result = await handler.handle("/api/analytics/outcomes/consensus-rate", {}, mock_handler)

        assert result is not None
        data = _get_data(result)
        assert "consensus_rate" in data

    @pytest.mark.asyncio
    async def test_dispatch_to_average_rounds(
        self, handler, mock_analytics, mock_rate_limiter, mock_handler
    ):
        result = await handler.handle("/api/analytics/outcomes/average-rounds", {}, mock_handler)

        assert result is not None
        data = _get_data(result)
        assert "average_rounds" in data

    @pytest.mark.asyncio
    async def test_dispatch_to_contributions(
        self, handler, mock_analytics, mock_rate_limiter, mock_handler
    ):
        result = await handler.handle("/api/analytics/outcomes/contributions", {}, mock_handler)

        assert result is not None
        data = _get_data(result)
        assert "contributions" in data

    @pytest.mark.asyncio
    async def test_dispatch_to_quality_trend(
        self, handler, mock_analytics, mock_rate_limiter, mock_handler
    ):
        result = await handler.handle("/api/analytics/outcomes/quality-trend", {}, mock_handler)

        assert result is not None
        data = _get_data(result)
        assert "trend" in data

    @pytest.mark.asyncio
    async def test_dispatch_to_topics(
        self, handler, mock_analytics, mock_rate_limiter, mock_handler
    ):
        result = await handler.handle("/api/analytics/outcomes/topics", {}, mock_handler)

        assert result is not None
        data = _get_data(result)
        assert "topics" in data

    @pytest.mark.asyncio
    async def test_dispatch_to_debate_id(
        self, handler, mock_analytics, mock_rate_limiter, mock_handler
    ):
        result = await handler.handle("/api/analytics/outcomes/debate-abc-123", {}, mock_handler)

        assert result is not None
        data = _get_data(result)
        assert "debate_id" in data

    @pytest.mark.asyncio
    async def test_dispatch_with_version_prefix(
        self, handler, mock_analytics, mock_rate_limiter, mock_handler
    ):
        result = await handler.handle("/api/v1/analytics/outcomes", {}, mock_handler)

        assert result is not None
        data = _get_data(result)
        assert "consensus_rate" in data


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


class TestRateLimiting:
    """Test rate limiting behaviour."""

    @pytest.mark.asyncio
    async def test_rate_limit_exceeded(self, handler, mock_handler):
        with patch(
            "aragora.server.handlers.outcome_analytics._outcome_analytics_limiter"
        ) as limiter:
            limiter.is_allowed.return_value = False
            result = await handler.handle("/api/analytics/outcomes", {}, mock_handler)

        assert _status_code(result) == 429
        body = _parse_response(result)
        assert "rate limit" in str(body).lower()


# ---------------------------------------------------------------------------
# Authentication / Authorization
# ---------------------------------------------------------------------------


class TestAuth:
    """Test authentication and authorization handling."""

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_unauthenticated_returns_401(self, handler, mock_rate_limiter, mock_handler):
        """When get_auth_context raises UnauthorizedError, return 401."""
        from aragora.server.handlers.secure import UnauthorizedError

        with patch.object(
            handler, "get_auth_context", side_effect=UnauthorizedError("not authenticated")
        ):
            result = await handler.handle("/api/analytics/outcomes", {}, mock_handler)

        assert _status_code(result) == 401

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_forbidden_returns_403(self, handler, mock_rate_limiter, mock_handler):
        """When check_permission raises ForbiddenError, return 403."""
        from aragora.server.handlers.secure import ForbiddenError

        mock_auth_ctx = MagicMock()

        with (
            patch.object(handler, "get_auth_context", return_value=mock_auth_ctx),
            patch.object(
                handler,
                "check_permission",
                side_effect=ForbiddenError("no permission", permission="analytics:read"),
            ),
        ):
            result = await handler.handle("/api/analytics/outcomes", {}, mock_handler)

        assert _status_code(result) == 403


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Test error handling via @handle_errors decorator."""

    @pytest.mark.asyncio
    async def test_summary_error_handling(self, handler, mock_rate_limiter):
        """When the analytics backend raises, @handle_errors should catch."""
        with patch(
            "aragora.analytics.outcome_analytics.get_outcome_analytics",
            side_effect=ImportError("not available"),
            create=True,
        ):
            result = await handler._get_outcomes_summary({})

        # @handle_errors should return an error response, not raise
        assert _status_code(result) >= 400

    @pytest.mark.asyncio
    async def test_consensus_rate_error_handling(self, handler, mock_rate_limiter):
        with patch(
            "aragora.analytics.outcome_analytics.get_outcome_analytics",
            side_effect=RuntimeError("backend down"),
            create=True,
        ):
            result = await handler._get_consensus_rate({})

        assert _status_code(result) >= 400

    @pytest.mark.asyncio
    async def test_debate_outcome_error_handling(self, handler, mock_rate_limiter):
        with patch(
            "aragora.analytics.outcome_analytics.get_outcome_analytics",
            side_effect=ValueError("invalid"),
            create=True,
        ):
            result = await handler._get_debate_outcome("bad-id")

        assert _status_code(result) >= 400
