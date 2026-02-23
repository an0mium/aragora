"""
Tests for AnalyticsMetricsHandler endpoints.

Provides comprehensive test coverage for the analytics dashboard metrics API:

Debate Analytics:
- GET /api/analytics/debates/overview
- GET /api/analytics/debates/trends
- GET /api/analytics/debates/topics
- GET /api/analytics/debates/outcomes

Agent Performance:
- GET /api/analytics/agents/leaderboard
- GET /api/analytics/agents/{agent_id}/performance
- GET /api/analytics/agents/comparison
- GET /api/analytics/agents/trends

Usage Analytics:
- GET /api/analytics/usage/tokens
- GET /api/analytics/usage/costs
- GET /api/analytics/usage/active_users
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Mark entire module for asyncio (contains many async test methods)
pytestmark = pytest.mark.asyncio

from aragora.server.handlers.analytics import (
    AnalyticsMetricsHandler,
    VALID_GRANULARITIES,
    VALID_TIME_RANGES,
    _group_by_time,
    _parse_time_range,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_server_context():
    """Create mock server context for handler initialization."""
    return {"storage": None, "elo_system": None, "user_store": None}


@pytest.fixture
def handler(mock_server_context):
    """Create handler instance with mocked context."""
    return AnalyticsMetricsHandler(mock_server_context)


@pytest.fixture
def mock_http_handler():
    """Create mock HTTP handler with authenticated context."""
    mock = MagicMock()
    mock.client_address = ("127.0.0.1", 12345)
    mock.headers = {"Authorization": "Bearer test-token"}
    return mock


@pytest.fixture
def mock_auth_context():
    """Create mock authorization context with analytics permission."""
    from aragora.rbac.models import AuthorizationContext

    return AuthorizationContext(
        user_id="user-123",
        org_id="org-456",
        workspace_id="ws-789",
        roles={"admin"},
        permissions={"analytics:read"},
    )


@pytest.fixture
def mock_storage():
    """Create mock storage with sample debate data."""
    storage = MagicMock()
    now = datetime.now(timezone.utc)

    debates = [
        {
            "id": "debate-1",
            "created_at": (now - timedelta(days=5)).isoformat(),
            "task": "security analysis",
            "domain": "security",
            "consensus_reached": True,
            "agents": ["claude", "gpt-4", "gemini"],
            "result": {
                "rounds_used": 3,
                "confidence": 0.85,
                "outcome_type": "consensus",
            },
        },
        {
            "id": "debate-2",
            "created_at": (now - timedelta(days=10)).isoformat(),
            "task": "performance review",
            "domain": "performance",
            "consensus_reached": True,
            "agents": ["claude", "gpt-4"],
            "result": {
                "rounds_used": 2,
                "confidence": 0.72,
                "outcome_type": "majority",
            },
        },
        {
            "id": "debate-3",
            "created_at": (now - timedelta(days=15)).isoformat(),
            "task": "budget planning",
            "domain": "finance",
            "consensus_reached": False,
            "agents": ["gemini", "gpt-4", "claude", "mistral"],
            "result": {
                "rounds_used": 5,
                "confidence": 0.45,
                "outcome_type": "dissent",
            },
        },
        {
            "id": "debate-4",
            "created_at": (now - timedelta(days=45)).isoformat(),
            "task": "previous period debate",
            "domain": "general",
            "consensus_reached": True,
            "agents": ["claude"],
            "result": {
                "rounds_used": 1,
                "confidence": 0.95,
            },
        },
    ]

    storage.list_debates.return_value = debates
    return storage


@pytest.fixture
def mock_elo_system():
    """Create mock ELO system with sample agent data."""
    elo = MagicMock()

    # Create mock agent ratings
    def make_agent(name, elo_score, wins, losses, draws):
        agent = MagicMock()
        agent.agent_name = name
        agent.elo = elo_score
        agent.wins = wins
        agent.losses = losses
        agent.draws = draws
        agent.games_played = wins + losses + draws
        agent.win_rate = wins / (wins + losses + draws) if (wins + losses + draws) > 0 else 0
        agent.debates_count = wins + losses + draws
        agent.domain_elos = {"security": elo_score + 50, "performance": elo_score - 20}
        agent.calibration_score = 0.85
        agent.calibration_accuracy = 0.78
        return agent

    agent1 = make_agent("claude", 1650, 120, 30, 10)
    agent2 = make_agent("gpt-4", 1580, 95, 45, 15)
    agent3 = make_agent("gemini", 1520, 80, 55, 20)
    agent4 = make_agent("mistral", 1480, 60, 70, 25)

    elo.get_leaderboard.return_value = [agent1, agent2, agent3, agent4]
    elo.list_agents.return_value = ["claude", "gpt-4", "gemini", "mistral"]
    elo.get_rating.side_effect = lambda name: {
        "claude": agent1,
        "gpt-4": agent2,
        "gemini": agent3,
        "mistral": agent4,
    }.get(name) or (_ for _ in ()).throw(KeyError(f"Agent not found: {name}"))

    # ELO history
    now = datetime.now(timezone.utc)
    elo.get_elo_history.return_value = [
        ((now - timedelta(days=10)).isoformat(), 1620),
        ((now - timedelta(days=5)).isoformat(), 1640),
        (now.isoformat(), 1650),
    ]

    # Recent matches
    elo.get_recent_matches.return_value = [
        {"id": "match-1", "participants": ["claude", "gpt-4"], "winner": "claude"},
        {"id": "match-2", "participants": ["gemini", "claude"], "winner": "claude"},
    ]

    # Head-to-head stats
    elo.get_head_to_head.return_value = {
        "a_wins": 15,
        "b_wins": 10,
        "draws": 5,
        "total": 30,
    }

    return elo


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestParseTimeRange:
    """Tests for _parse_time_range helper function."""

    def test_parse_time_range_7d(self):
        """Parse 7 days time range."""
        result = _parse_time_range("7d")
        assert result is not None
        # Should be approximately 7 days ago
        expected = datetime.now(timezone.utc) - timedelta(days=7)
        assert abs((result - expected).total_seconds()) < 2

    def test_parse_time_range_30d(self):
        """Parse 30 days time range."""
        result = _parse_time_range("30d")
        assert result is not None
        expected = datetime.now(timezone.utc) - timedelta(days=30)
        assert abs((result - expected).total_seconds()) < 2

    def test_parse_time_range_365d(self):
        """Parse 365 days time range."""
        result = _parse_time_range("365d")
        assert result is not None
        expected = datetime.now(timezone.utc) - timedelta(days=365)
        assert abs((result - expected).total_seconds()) < 2

    def test_parse_time_range_all(self):
        """Parse 'all' time range returns None."""
        result = _parse_time_range("all")
        assert result is None

    def test_parse_time_range_invalid_defaults_to_30d(self):
        """Invalid time range defaults to 30 days."""
        result = _parse_time_range("invalid")
        assert result is not None
        expected = datetime.now(timezone.utc) - timedelta(days=30)
        assert abs((result - expected).total_seconds()) < 2


class TestGroupByTime:
    """Tests for _group_by_time helper function."""

    def test_group_by_daily(self):
        """Group items by daily granularity."""
        items = [
            {"ts": "2026-01-15T10:00:00Z", "value": 1},
            {"ts": "2026-01-15T14:00:00Z", "value": 2},
            {"ts": "2026-01-16T09:00:00Z", "value": 3},
        ]
        result = _group_by_time(items, "ts", "daily")
        assert "2026-01-15" in result
        assert "2026-01-16" in result
        assert len(result["2026-01-15"]) == 2
        assert len(result["2026-01-16"]) == 1

    def test_group_by_weekly(self):
        """Group items by weekly granularity."""
        items = [
            {"ts": "2026-01-15T10:00:00Z", "value": 1},
            {"ts": "2026-01-22T10:00:00Z", "value": 2},
        ]
        result = _group_by_time(items, "ts", "weekly")
        assert len(result) == 2

    def test_group_by_monthly(self):
        """Group items by monthly granularity."""
        items = [
            {"ts": "2026-01-15T10:00:00Z", "value": 1},
            {"ts": "2026-02-15T10:00:00Z", "value": 2},
        ]
        result = _group_by_time(items, "ts", "monthly")
        assert "2026-01" in result
        assert "2026-02" in result

    def test_group_by_time_with_datetime_objects(self):
        """Handle datetime objects in items."""
        items = [
            {"ts": datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc), "value": 1},
            {"ts": datetime(2026, 1, 15, 14, 0, 0, tzinfo=timezone.utc), "value": 2},
        ]
        result = _group_by_time(items, "ts", "daily")
        assert "2026-01-15" in result
        assert len(result["2026-01-15"]) == 2

    def test_group_by_time_skips_invalid(self):
        """Skip items with invalid or missing timestamps."""
        items = [
            {"ts": "2026-01-15T10:00:00Z", "value": 1},
            {"ts": None, "value": 2},
            {"ts": "invalid", "value": 3},
            {"other": "field", "value": 4},
        ]
        result = _group_by_time(items, "ts", "daily")
        assert len(result) == 1
        assert len(result["2026-01-15"]) == 1


# =============================================================================
# Handler Routing Tests
# =============================================================================


class TestAnalyticsMetricsHandlerRouting:
    """Tests for handler routing and can_handle method."""

    def test_can_handle_debates_overview(self, handler):
        """Handler can handle debates overview endpoint."""
        assert handler.can_handle("/api/analytics/debates/overview")
        assert handler.can_handle("/api/v1/analytics/debates/overview")

    def test_can_handle_debates_trends(self, handler):
        """Handler can handle debates trends endpoint."""
        assert handler.can_handle("/api/analytics/debates/trends")

    def test_can_handle_debates_topics(self, handler):
        """Handler can handle debates topics endpoint."""
        assert handler.can_handle("/api/analytics/debates/topics")

    def test_can_handle_debates_outcomes(self, handler):
        """Handler can handle debates outcomes endpoint."""
        assert handler.can_handle("/api/analytics/debates/outcomes")

    def test_can_handle_agents_leaderboard(self, handler):
        """Handler can handle agents leaderboard endpoint."""
        assert handler.can_handle("/api/analytics/agents/leaderboard")

    def test_can_handle_agent_performance(self, handler):
        """Handler can handle individual agent performance endpoint."""
        assert handler.can_handle("/api/analytics/agents/claude/performance")
        assert handler.can_handle("/api/analytics/agents/gpt-4/performance")
        assert handler.can_handle("/api/analytics/agents/my_agent_123/performance")

    def test_can_handle_agents_comparison(self, handler):
        """Handler can handle agents comparison endpoint."""
        assert handler.can_handle("/api/analytics/agents/comparison")

    def test_can_handle_agents_trends(self, handler):
        """Handler can handle agents trends endpoint."""
        assert handler.can_handle("/api/analytics/agents/trends")

    def test_can_handle_usage_tokens(self, handler):
        """Handler can handle usage tokens endpoint."""
        assert handler.can_handle("/api/analytics/usage/tokens")

    def test_can_handle_usage_costs(self, handler):
        """Handler can handle usage costs endpoint."""
        assert handler.can_handle("/api/analytics/usage/costs")

    def test_can_handle_usage_active_users(self, handler):
        """Handler can handle usage active users endpoint."""
        assert handler.can_handle("/api/analytics/usage/active_users")

    def test_cannot_handle_unknown_path(self, handler):
        """Handler cannot handle unknown paths."""
        assert not handler.can_handle("/api/analytics/unknown")
        assert not handler.can_handle("/api/debates")
        assert not handler.can_handle("/api/other/endpoint")


# =============================================================================
# Authentication and Authorization Tests
# =============================================================================


class TestAnalyticsMetricsAuth:
    """Tests for authentication and authorization."""

    async def test_unauthenticated_returns_401(self, handler, mock_http_handler):
        """Unauthenticated request returns 401."""
        with patch.object(handler, "get_auth_context") as mock_auth:
            from aragora.server.handlers.utils.auth import UnauthorizedError

            mock_auth.side_effect = UnauthorizedError("Authentication required")

            result = await handler.handle("/api/analytics/debates/overview", {}, mock_http_handler)

            assert result is not None
            assert result.status_code == 401

    async def test_missing_permission_returns_403(self, handler, mock_http_handler):
        """Request without analytics:read permission returns 403."""
        from aragora.rbac.models import AuthorizationContext

        no_perm_context = AuthorizationContext(
            user_id="user-123",
            org_id="org-456",
            roles={"member"},
            permissions=set(),  # No analytics:read permission
        )

        with patch.object(handler, "get_auth_context", return_value=no_perm_context):
            with patch.object(handler, "check_permission") as mock_check:
                from aragora.server.handlers.utils.auth import ForbiddenError

                mock_check.side_effect = ForbiddenError(
                    "Permission denied: analytics:read", permission="analytics:read"
                )

                result = await handler.handle(
                    "/api/analytics/debates/overview", {}, mock_http_handler
                )

                assert result is not None
                assert result.status_code == 403


class TestRateLimiting:
    """Tests for rate limiting."""

    async def test_rate_limit_exceeded_returns_429(self, handler, mock_http_handler):
        """Rate limit exceeded returns 429."""
        from aragora.server.handlers.analytics import _analytics_metrics_limiter

        # Mock rate limiter to deny requests
        with patch.object(_analytics_metrics_limiter, "is_allowed", return_value=False):
            result = await handler.handle("/api/analytics/debates/overview", {}, mock_http_handler)

            assert result is not None
            assert result.status_code == 429
            body = json.loads(result.body)
            assert "Rate limit" in body.get("error", "")


# =============================================================================
# Debate Analytics Endpoint Tests
# =============================================================================


class TestDebatesOverview:
    """Tests for GET /api/analytics/debates/overview."""

    async def test_debates_overview_no_storage(self, handler, mock_http_handler, mock_auth_context):
        """Overview returns empty data when no storage available."""
        with patch.object(handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(handler, "check_permission", return_value=True):
                result = handler._get_debates_overview({})

                assert result is not None
                body = json.loads(result.body)
                assert body["total_debates"] == 0
                assert body["consensus_rate"] == 0.0
                assert "generated_at" in body

    async def test_debates_overview_with_storage(
        self, handler, mock_http_handler, mock_auth_context, mock_storage
    ):
        """Overview returns computed metrics with storage data."""
        handler.ctx["storage"] = mock_storage

        with patch.object(handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(handler, "check_permission", return_value=True):
                result = handler._get_debates_overview({"time_range": "30d"})

                assert result is not None
                body = json.loads(result.body)
                assert body["time_range"] == "30d"
                assert body["total_debates"] == 4
                assert "consensus_rate" in body
                assert "avg_rounds" in body
                assert "avg_agents_per_debate" in body
                assert "avg_confidence" in body

    async def test_debates_overview_filters_by_org_id(
        self, handler, mock_storage, mock_auth_context
    ):
        """Overview filters debates by org_id."""
        handler.ctx["storage"] = mock_storage

        with patch.object(handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(handler, "check_permission", return_value=True):
                handler._get_debates_overview({"org_id": "org-123"})

                mock_storage.list_debates.assert_called_with(limit=10000, org_id="org-123")

    async def test_debates_overview_invalid_time_range_defaults(
        self, handler, mock_storage, mock_auth_context
    ):
        """Invalid time range defaults to 30d."""
        handler.ctx["storage"] = mock_storage

        with patch.object(handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(handler, "check_permission", return_value=True):
                result = handler._get_debates_overview({"time_range": "invalid"})

                body = json.loads(result.body)
                assert body["time_range"] == "30d"


class TestDebatesTrends:
    """Tests for GET /api/analytics/debates/trends."""

    async def test_debates_trends_no_storage(self, handler, mock_auth_context):
        """Trends returns empty data when no storage available."""
        with patch.object(handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(handler, "check_permission", return_value=True):
                result = handler._get_debates_trends({})

                body = json.loads(result.body)
                assert body["data_points"] == []
                assert body["granularity"] == "daily"

    async def test_debates_trends_with_storage(self, handler, mock_storage, mock_auth_context):
        """Trends returns grouped data with storage."""
        handler.ctx["storage"] = mock_storage

        with patch.object(handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(handler, "check_permission", return_value=True):
                result = handler._get_debates_trends({"time_range": "30d", "granularity": "daily"})

                body = json.loads(result.body)
                assert "data_points" in body
                assert body["granularity"] == "daily"
                for dp in body["data_points"]:
                    assert "period" in dp
                    assert "total" in dp
                    assert "consensus_reached" in dp
                    assert "consensus_rate" in dp
                    assert "avg_rounds" in dp

    async def test_debates_trends_weekly_granularity(
        self, handler, mock_storage, mock_auth_context
    ):
        """Trends supports weekly granularity."""
        handler.ctx["storage"] = mock_storage

        with patch.object(handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(handler, "check_permission", return_value=True):
                result = handler._get_debates_trends({"granularity": "weekly"})

                body = json.loads(result.body)
                assert body["granularity"] == "weekly"

    async def test_debates_trends_monthly_granularity(
        self, handler, mock_storage, mock_auth_context
    ):
        """Trends supports monthly granularity."""
        handler.ctx["storage"] = mock_storage

        with patch.object(handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(handler, "check_permission", return_value=True):
                result = handler._get_debates_trends({"granularity": "monthly"})

                body = json.loads(result.body)
                assert body["granularity"] == "monthly"


class TestDebatesTopics:
    """Tests for GET /api/analytics/debates/topics."""

    async def test_debates_topics_no_storage(self, handler, mock_auth_context):
        """Topics returns empty data when no storage available."""
        with patch.object(handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(handler, "check_permission", return_value=True):
                result = handler._get_debates_topics({})

                body = json.loads(result.body)
                assert body["topics"] == []
                assert body["total_debates"] == 0

    async def test_debates_topics_with_storage(self, handler, mock_storage, mock_auth_context):
        """Topics returns topic distribution with storage."""
        handler.ctx["storage"] = mock_storage

        with patch.object(handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(handler, "check_permission", return_value=True):
                result = handler._get_debates_topics({"time_range": "all"})

                body = json.loads(result.body)
                assert "topics" in body
                assert body["total_debates"] > 0
                for topic in body["topics"]:
                    assert "topic" in topic
                    assert "count" in topic
                    assert "percentage" in topic
                    assert "consensus_rate" in topic

    async def test_debates_topics_limit(self, handler, mock_storage, mock_auth_context):
        """Topics respects limit parameter."""
        handler.ctx["storage"] = mock_storage

        with patch.object(handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(handler, "check_permission", return_value=True):
                result = handler._get_debates_topics({"limit": "2"})

                body = json.loads(result.body)
                assert len(body["topics"]) <= 2


class TestDebatesOutcomes:
    """Tests for GET /api/analytics/debates/outcomes."""

    async def test_debates_outcomes_no_storage(self, handler, mock_auth_context):
        """Outcomes returns empty data when no storage available."""
        with patch.object(handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(handler, "check_permission", return_value=True):
                result = handler._get_debates_outcomes({})

                body = json.loads(result.body)
                assert body["outcomes"]["consensus"] == 0
                assert body["outcomes"]["majority"] == 0
                assert body["outcomes"]["dissent"] == 0
                assert body["outcomes"]["no_resolution"] == 0
                assert body["total_debates"] == 0

    async def test_debates_outcomes_with_storage(self, handler, mock_storage, mock_auth_context):
        """Outcomes returns outcome distribution with storage."""
        handler.ctx["storage"] = mock_storage

        with patch.object(handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(handler, "check_permission", return_value=True):
                result = handler._get_debates_outcomes({"time_range": "all"})

                body = json.loads(result.body)
                assert "outcomes" in body
                assert "by_confidence" in body
                assert body["total_debates"] > 0


# =============================================================================
# Agent Performance Endpoint Tests
# =============================================================================


class TestAgentsLeaderboard:
    """Tests for GET /api/analytics/agents/leaderboard."""

    async def test_agents_leaderboard_no_elo_system(self, handler, mock_auth_context):
        """Leaderboard returns empty data when no ELO system available."""
        with patch.object(handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(handler, "check_permission", return_value=True):
                result = handler._get_agents_leaderboard({})

                body = json.loads(result.body)
                assert body["leaderboard"] == []
                assert body["total_agents"] == 0

    async def test_agents_leaderboard_with_elo_system(
        self, handler, mock_elo_system, mock_auth_context
    ):
        """Leaderboard returns ranked agents with ELO system."""
        handler.ctx["elo_system"] = mock_elo_system

        with patch.object(handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(handler, "check_permission", return_value=True):
                result = handler._get_agents_leaderboard({})

                body = json.loads(result.body)
                assert len(body["leaderboard"]) == 4
                assert body["total_agents"] == 4

                # Verify first agent (should be ranked 1)
                first = body["leaderboard"][0]
                assert first["rank"] == 1
                assert first["agent_name"] == "claude"
                assert "elo" in first
                assert "wins" in first
                assert "losses" in first
                assert "win_rate" in first
                assert "calibration_score" in first

    async def test_agents_leaderboard_with_limit(self, handler, mock_elo_system, mock_auth_context):
        """Leaderboard respects limit parameter."""
        handler.ctx["elo_system"] = mock_elo_system

        with patch.object(handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(handler, "check_permission", return_value=True):
                handler._get_agents_leaderboard({"limit": "2"})

                mock_elo_system.get_leaderboard.assert_called_with(limit=2, domain=None)

    async def test_agents_leaderboard_with_domain_filter(
        self, handler, mock_elo_system, mock_auth_context
    ):
        """Leaderboard filters by domain."""
        handler.ctx["elo_system"] = mock_elo_system

        with patch.object(handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(handler, "check_permission", return_value=True):
                handler._get_agents_leaderboard({"domain": "security"})

                mock_elo_system.get_leaderboard.assert_called_with(limit=20, domain="security")


class TestAgentPerformance:
    """Tests for GET /api/analytics/agents/{agent_id}/performance."""

    async def test_agent_performance_no_elo_system(self, handler, mock_auth_context):
        """Performance returns 503 when no ELO system available."""
        with patch.object(handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(handler, "check_permission", return_value=True):
                result = handler._get_agent_performance("claude", {})

                assert result.status_code == 503

    async def test_agent_performance_agent_not_found(
        self, handler, mock_elo_system, mock_auth_context
    ):
        """Performance returns 404 for unknown agent."""
        handler.ctx["elo_system"] = mock_elo_system

        with patch.object(handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(handler, "check_permission", return_value=True):
                result = handler._get_agent_performance("unknown-agent", {})

                assert result.status_code == 404
                body = json.loads(result.body)
                assert "not found" in body.get("error", "").lower()

    async def test_agent_performance_success(self, handler, mock_elo_system, mock_auth_context):
        """Performance returns detailed stats for known agent."""
        handler.ctx["elo_system"] = mock_elo_system

        with patch.object(handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(handler, "check_permission", return_value=True):
                result = handler._get_agent_performance("claude", {})

                assert result.status_code == 200
                body = json.loads(result.body)
                assert body["agent_id"] == "claude"
                assert "elo" in body
                assert "wins" in body
                assert "losses" in body
                assert "win_rate" in body
                assert "domain_performance" in body
                assert "elo_history" in body
                assert "recent_matches" in body


class TestAgentsComparison:
    """Tests for GET /api/analytics/agents/comparison."""

    async def test_agents_comparison_missing_param(self, handler, mock_auth_context):
        """Comparison returns 400 when agents param missing."""
        with patch.object(handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(handler, "check_permission", return_value=True):
                result = handler._get_agents_comparison({})

                assert result.status_code == 400
                body = json.loads(result.body)
                assert "required" in body.get("error", "").lower()

    async def test_agents_comparison_too_few_agents(self, handler, mock_auth_context):
        """Comparison returns 400 when fewer than 2 agents."""
        with patch.object(handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(handler, "check_permission", return_value=True):
                result = handler._get_agents_comparison({"agents": "claude"})

                assert result.status_code == 400
                body = json.loads(result.body)
                assert "2 agents" in body.get("error", "")

    async def test_agents_comparison_too_many_agents(self, handler, mock_auth_context):
        """Comparison returns 400 when more than 10 agents."""
        with patch.object(handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(handler, "check_permission", return_value=True):
                agents = ",".join([f"agent{i}" for i in range(15)])
                result = handler._get_agents_comparison({"agents": agents})

                assert result.status_code == 400
                body = json.loads(result.body)
                assert "10 agents" in body.get("error", "")

    async def test_agents_comparison_no_elo_system(self, handler, mock_auth_context):
        """Comparison returns 503 when no ELO system."""
        with patch.object(handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(handler, "check_permission", return_value=True):
                result = handler._get_agents_comparison({"agents": "claude,gpt-4"})

                assert result.status_code == 503

    async def test_agents_comparison_success(self, handler, mock_elo_system, mock_auth_context):
        """Comparison returns comparison data for multiple agents."""
        handler.ctx["elo_system"] = mock_elo_system

        with patch.object(handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(handler, "check_permission", return_value=True):
                result = handler._get_agents_comparison({"agents": "claude,gpt-4,gemini"})

                assert result.status_code == 200
                body = json.loads(result.body)
                assert body["agents"] == ["claude", "gpt-4", "gemini"]
                assert len(body["comparison"]) == 3
                assert "head_to_head" in body

    async def test_agents_comparison_includes_not_found_agents(
        self, handler, mock_elo_system, mock_auth_context
    ):
        """Comparison includes error for agents not found."""
        handler.ctx["elo_system"] = mock_elo_system

        with patch.object(handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(handler, "check_permission", return_value=True):
                result = handler._get_agents_comparison({"agents": "claude,unknown-agent"})

                assert result.status_code == 200
                body = json.loads(result.body)
                # One should have data, one should have error
                has_error = any("error" in c for c in body["comparison"])
                assert has_error


class TestAgentsTrends:
    """Tests for GET /api/analytics/agents/trends."""

    async def test_agents_trends_no_elo_system(self, handler, mock_auth_context):
        """Trends returns 503 when no ELO system."""
        with patch.object(handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(handler, "check_permission", return_value=True):
                result = handler._get_agents_trends({})

                assert result.status_code == 503

    async def test_agents_trends_default_to_top_5(
        self, handler, mock_elo_system, mock_auth_context
    ):
        """Trends defaults to top 5 agents when no agents param."""
        handler.ctx["elo_system"] = mock_elo_system

        with patch.object(handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(handler, "check_permission", return_value=True):
                result = handler._get_agents_trends({})

                assert result.status_code == 200
                body = json.loads(result.body)
                assert "agents" in body
                assert "trends" in body

    async def test_agents_trends_with_specific_agents(
        self, handler, mock_elo_system, mock_auth_context
    ):
        """Trends returns data for specified agents."""
        handler.ctx["elo_system"] = mock_elo_system

        with patch.object(handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(handler, "check_permission", return_value=True):
                result = handler._get_agents_trends({"agents": "claude,gpt-4"})

                assert result.status_code == 200
                body = json.loads(result.body)
                assert "claude" in body["agents"]
                assert "gpt-4" in body["agents"]


# =============================================================================
# Usage Analytics Endpoint Tests
# =============================================================================


class TestUsageTokens:
    """Tests for GET /api/analytics/usage/tokens."""

    async def test_usage_tokens_missing_org_id(self, handler, mock_auth_context):
        """Tokens returns 400 when org_id missing."""
        with patch.object(handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(handler, "check_permission", return_value=True):
                result = handler._get_usage_tokens({})

                assert result.status_code == 400
                body = json.loads(result.body)
                assert "org_id" in body.get("error", "")

    async def test_usage_tokens_cost_tracker_unavailable(self, handler, mock_auth_context):
        """Tokens returns fallback data when cost tracker unavailable."""
        with patch.object(handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(handler, "check_permission", return_value=True):
                with patch.dict("sys.modules", {"aragora.billing.cost_tracker": None}):
                    result = handler._get_usage_tokens({"org_id": "org-123"})

                    assert result.status_code == 200
                    body = json.loads(result.body)
                    assert body["org_id"] == "org-123"

    async def test_usage_tokens_with_cost_tracker(self, handler, mock_auth_context):
        """Tokens returns data from cost tracker."""
        mock_tracker = MagicMock()
        mock_tracker.get_workspace_stats.return_value = {
            "total_tokens_in": 5000000,
            "total_tokens_out": 1000000,
            "cost_by_agent": {"claude": "80.00"},
            "cost_by_model": {"claude-opus": "80.00"},
        }

        mock_module = MagicMock()
        mock_module.get_cost_tracker.return_value = mock_tracker

        with patch.object(handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(handler, "check_permission", return_value=True):
                with patch.dict(
                    "sys.modules",
                    {"aragora.billing.cost_tracker": mock_module},
                ):
                    result = handler._get_usage_tokens({"org_id": "org-123"})

                    assert result.status_code == 200
                    body = json.loads(result.body)
                    assert body["summary"]["total_tokens_in"] == 5000000
                    assert body["summary"]["total_tokens_out"] == 1000000
                    assert body["summary"]["total_tokens"] == 6000000


class TestUsageCosts:
    """Tests for GET /api/analytics/usage/costs."""

    async def test_usage_costs_missing_org_id(self, handler, mock_auth_context):
        """Costs returns 400 when org_id missing."""
        with patch.object(handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(handler, "check_permission", return_value=True):
                result = handler._get_usage_costs({})

                assert result.status_code == 400

    async def test_usage_costs_with_cost_tracker(self, handler, mock_auth_context):
        """Costs returns cost breakdown from tracker."""
        mock_tracker = MagicMock()
        mock_tracker.get_workspace_stats.return_value = {
            "total_cost_usd": "125.50",
            "total_api_calls": 150,
            "cost_by_agent": {"claude": "80.00", "gpt-4": "45.50"},
            "cost_by_model": {"claude-opus": "80.00"},
        }

        mock_module = MagicMock()
        mock_module.get_cost_tracker.return_value = mock_tracker

        with patch.object(handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(handler, "check_permission", return_value=True):
                with patch.dict(
                    "sys.modules",
                    {"aragora.billing.cost_tracker": mock_module},
                ):
                    result = handler._get_usage_costs({"org_id": "org-123"})

                    assert result.status_code == 200
                    body = json.loads(result.body)
                    assert body["summary"]["total_cost_usd"] == "125.50"
                    assert "by_provider" in body
                    assert "by_model" in body


class TestActiveUsers:
    """Tests for GET /api/analytics/usage/active_users."""

    async def test_active_users_no_user_store(self, handler, mock_auth_context):
        """Active users returns fallback when no user store."""
        with patch.object(handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(handler, "check_permission", return_value=True):
                result = handler._get_active_users({})

                assert result.status_code == 200
                body = json.loads(result.body)
                assert body["active_users"]["daily"] == 0
                assert "message" in body

    async def test_active_users_with_user_store(self, handler, mock_auth_context):
        """Active users returns data from user store."""
        mock_user_store = MagicMock()
        mock_user_store.get_active_user_counts.return_value = {
            "daily": 25,
            "weekly": 85,
            "monthly": 150,
        }
        mock_user_store.get_user_growth.return_value = {
            "new_users": 15,
            "churned_users": 5,
            "net_growth": 10,
        }
        handler.ctx["user_store"] = mock_user_store

        with patch.object(handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(handler, "check_permission", return_value=True):
                result = handler._get_active_users({"org_id": "org-123"})

                assert result.status_code == 200
                body = json.loads(result.body)
                assert body["active_users"]["daily"] == 25
                assert body["active_users"]["weekly"] == 85
                assert body["user_growth"]["new_users"] == 15

    async def test_active_users_invalid_time_range(self, handler, mock_auth_context):
        """Active users defaults invalid time range to 30d."""
        with patch.object(handler, "get_auth_context", return_value=mock_auth_context):
            with patch.object(handler, "check_permission", return_value=True):
                result = handler._get_active_users({"time_range": "invalid"})

                body = json.loads(result.body)
                assert body["time_range"] == "30d"


# =============================================================================
# Integration Tests
# =============================================================================


class TestAnalyticsMetricsIntegration:
    """Integration tests for full request handling flow."""

    async def test_full_handle_flow_debates_overview(
        self, handler, mock_http_handler, mock_auth_context, mock_storage
    ):
        """Full handle flow for debates overview."""
        handler.ctx["storage"] = mock_storage

        from aragora.server.handlers.analytics import _analytics_metrics_limiter

        with patch.object(_analytics_metrics_limiter, "is_allowed", return_value=True):
            with patch.object(handler, "get_auth_context", return_value=mock_auth_context):
                with patch.object(handler, "check_permission", return_value=True):
                    result = await handler.handle(
                        "/api/analytics/debates/overview",
                        {"time_range": "30d"},
                        mock_http_handler,
                    )

                    assert result is not None
                    assert result.status_code == 200
                    body = json.loads(result.body)
                    assert "total_debates" in body

    async def test_full_handle_flow_agent_performance(
        self, handler, mock_http_handler, mock_auth_context, mock_elo_system
    ):
        """Full handle flow for agent performance."""
        handler.ctx["elo_system"] = mock_elo_system

        from aragora.server.handlers.analytics import _analytics_metrics_limiter

        with patch.object(_analytics_metrics_limiter, "is_allowed", return_value=True):
            with patch.object(handler, "get_auth_context", return_value=mock_auth_context):
                with patch.object(handler, "check_permission", return_value=True):
                    result = await handler.handle(
                        "/api/analytics/agents/claude/performance",
                        {},
                        mock_http_handler,
                    )

                    assert result is not None
                    assert result.status_code == 200
                    body = json.loads(result.body)
                    assert body["agent_id"] == "claude"

    async def test_handle_returns_none_for_unknown_path(
        self, handler, mock_http_handler, mock_auth_context
    ):
        """Handle returns None for paths not handled."""
        from aragora.server.handlers.analytics import _analytics_metrics_limiter

        with patch.object(_analytics_metrics_limiter, "is_allowed", return_value=True):
            with patch.object(handler, "get_auth_context", return_value=mock_auth_context):
                with patch.object(handler, "check_permission", return_value=True):
                    result = await handler.handle("/api/other/endpoint", {}, mock_http_handler)

                    assert result is None
