"""
Comprehensive tests for the AnalyticsMetricsHandler module.

Tests cover:
- Handler initialization and routing
- Route handling and can_handle method
- Debate analytics endpoints (overview, trends, topics, outcomes)
- Agent performance endpoints (leaderboard, comparison, trends, individual)
- Usage analytics endpoints (tokens, costs, active_users)
- Rate limiting
- RBAC/authentication checks
- Edge cases and error handling
- Helper functions (_parse_time_range, _group_by_time)
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from aragora.server.handlers._analytics_metrics_impl import (
    AnalyticsMetricsHandler,
    _parse_time_range,
    _group_by_time,
    VALID_GRANULARITIES,
    VALID_TIME_RANGES,
    ANALYTICS_METRICS_PERMISSION,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_server_context():
    """Create mock server context for handler initialization."""
    return {"storage": None, "elo_system": None, "nomic_dir": None, "user_store": None}


@pytest.fixture
def handler(mock_server_context):
    """Create handler instance for tests."""
    return AnalyticsMetricsHandler(mock_server_context)


@pytest.fixture
def mock_http_handler():
    """Create mock HTTP handler for tests."""
    mock = MagicMock()
    mock.headers = {"Authorization": "Bearer test-token"}
    mock.client_address = ("192.168.1.1", 12345)
    return mock


@pytest.fixture
def mock_auth_context():
    """Create mock authorization context."""
    ctx = MagicMock()
    ctx.user_id = "test-user-123"
    ctx.org_id = "test-org-456"
    ctx.workspace_id = "test-workspace-789"
    ctx.roles = {"admin"}
    ctx.permissions = {"analytics:read"}
    return ctx


@pytest.fixture
def mock_storage():
    """Create mock storage with sample debates."""
    storage = MagicMock()

    now = datetime.now(timezone.utc)
    debates = [
        {
            "id": f"debate-{i}",
            "created_at": (now - timedelta(days=i)).isoformat(),
            "consensus_reached": i % 3 != 0,
            "result": {
                "rounds_used": (i % 5) + 1,
                "confidence": 0.7 + (i % 3) * 0.1,
                "outcome_type": "consensus" if i % 3 != 0 else "dissent",
            },
            "agents": ["claude", "gpt-4"] if i % 2 == 0 else ["claude", "gpt-4", "gemini"],
            "task": f"Task about {'security' if i % 2 == 0 else 'performance'}",
            "domain": "security" if i % 2 == 0 else "performance",
        }
        for i in range(50)
    ]
    storage.list_debates.return_value = debates
    return storage


@pytest.fixture
def mock_elo_system():
    """Create mock ELO system with sample agents."""
    elo_system = MagicMock()

    # Create mock agents
    agents = []
    for i, name in enumerate(["claude", "gpt-4", "gemini", "grok", "mistral"]):
        agent = MagicMock()
        agent.agent_name = name
        agent.elo = 1500 + (5 - i) * 50
        agent.wins = 100 - i * 10
        agent.losses = 20 + i * 5
        agent.draws = 10
        agent.win_rate = (100 - i * 10) / (130 - i * 5)
        agent.games_played = 130 - i * 5
        agent.debates_count = 50 - i * 5
        agent.domain_elos = {"security": 1550 - i * 20, "performance": 1480 - i * 20}
        agent.calibration_score = 0.85 - i * 0.05
        agent.calibration_accuracy = 0.82 - i * 0.04
        agents.append(agent)

    elo_system.get_leaderboard.return_value = agents
    elo_system.list_agents.return_value = agents
    elo_system.get_rating.side_effect = lambda name: next(
        (a for a in agents if a.agent_name == name), None
    )

    # ELO history
    now = datetime.now(timezone.utc)
    elo_system.get_elo_history.return_value = [
        ((now - timedelta(days=i)).isoformat(), 1500 + i * 5) for i in range(10)
    ]

    # Recent matches
    elo_system.get_recent_matches.return_value = [
        {"participants": ["claude", "gpt-4"], "winner": "claude"},
        {"participants": ["claude", "gemini"], "winner": "claude"},
        {"participants": ["gpt-4", "gemini"], "winner": "gpt-4"},
    ]

    # Head-to-head
    elo_system.get_head_to_head.return_value = {
        "a_wins": 15,
        "b_wins": 10,
        "draws": 5,
        "total": 30,
    }

    return elo_system


# =============================================================================
# Tests for Helper Functions
# =============================================================================


class TestParseTimeRange:
    """Tests for _parse_time_range helper function."""

    def test_parse_time_range_all_returns_none(self):
        """'all' time range returns None."""
        result = _parse_time_range("all")
        assert result is None

    def test_parse_time_range_7d(self):
        """7d time range returns datetime 7 days ago."""
        result = _parse_time_range("7d")
        expected = datetime.now(timezone.utc) - timedelta(days=7)
        assert abs((result - expected).total_seconds()) < 2

    def test_parse_time_range_30d(self):
        """30d time range returns datetime 30 days ago."""
        result = _parse_time_range("30d")
        expected = datetime.now(timezone.utc) - timedelta(days=30)
        assert abs((result - expected).total_seconds()) < 2

    def test_parse_time_range_90d(self):
        """90d time range returns datetime 90 days ago."""
        result = _parse_time_range("90d")
        expected = datetime.now(timezone.utc) - timedelta(days=90)
        assert abs((result - expected).total_seconds()) < 2

    def test_parse_time_range_365d(self):
        """365d time range returns datetime 365 days ago."""
        result = _parse_time_range("365d")
        expected = datetime.now(timezone.utc) - timedelta(days=365)
        assert abs((result - expected).total_seconds()) < 2

    def test_parse_time_range_invalid_returns_default_30d(self):
        """Invalid time range returns default 30d."""
        result = _parse_time_range("invalid")
        expected = datetime.now(timezone.utc) - timedelta(days=30)
        assert abs((result - expected).total_seconds()) < 2

    def test_parse_time_range_empty_returns_default_30d(self):
        """Empty time range returns default 30d."""
        result = _parse_time_range("")
        expected = datetime.now(timezone.utc) - timedelta(days=30)
        assert abs((result - expected).total_seconds()) < 2


class TestGroupByTime:
    """Tests for _group_by_time helper function."""

    def test_group_by_time_daily(self):
        """Group items by day."""
        items = [
            {"created_at": "2026-01-15T10:00:00Z"},
            {"created_at": "2026-01-15T14:00:00Z"},
            {"created_at": "2026-01-16T10:00:00Z"},
        ]
        result = _group_by_time(items, "created_at", "daily")
        assert "2026-01-15" in result
        assert "2026-01-16" in result
        assert len(result["2026-01-15"]) == 2
        assert len(result["2026-01-16"]) == 1

    def test_group_by_time_weekly(self):
        """Group items by week."""
        items = [
            {"created_at": "2026-01-06T10:00:00Z"},  # Week 01
            {"created_at": "2026-01-07T10:00:00Z"},  # Week 01
            {"created_at": "2026-01-13T10:00:00Z"},  # Week 02
        ]
        result = _group_by_time(items, "created_at", "weekly")
        assert len(result) == 2

    def test_group_by_time_monthly(self):
        """Group items by month."""
        items = [
            {"created_at": "2026-01-15T10:00:00Z"},
            {"created_at": "2026-01-25T10:00:00Z"},
            {"created_at": "2026-02-10T10:00:00Z"},
        ]
        result = _group_by_time(items, "created_at", "monthly")
        assert "2026-01" in result
        assert "2026-02" in result
        assert len(result["2026-01"]) == 2
        assert len(result["2026-02"]) == 1

    def test_group_by_time_with_datetime_objects(self):
        """Group items when timestamps are datetime objects."""
        now = datetime.now(timezone.utc)
        items = [
            {"ts": now},
            {"ts": now - timedelta(hours=2)},
        ]
        result = _group_by_time(items, "ts", "daily")
        assert len(result) == 1

    def test_group_by_time_skips_missing_timestamps(self):
        """Items without timestamp are skipped."""
        items = [
            {"created_at": "2026-01-15T10:00:00Z"},
            {"other_field": "value"},
            {"created_at": None},
        ]
        result = _group_by_time(items, "created_at", "daily")
        assert len(result) == 1
        assert len(result["2026-01-15"]) == 1

    def test_group_by_time_skips_invalid_timestamps(self):
        """Items with invalid timestamps are skipped."""
        items = [
            {"created_at": "2026-01-15T10:00:00Z"},
            {"created_at": "not-a-date"},
            {"created_at": 12345},
        ]
        result = _group_by_time(items, "created_at", "daily")
        assert len(result) == 1

    def test_group_by_time_empty_list(self):
        """Empty list returns empty dict."""
        result = _group_by_time([], "created_at", "daily")
        assert result == {}


# =============================================================================
# Tests for Handler Routing
# =============================================================================


class TestAnalyticsMetricsHandlerRouting:
    """Tests for handler routing."""

    def test_can_handle_debates_overview(self, handler):
        """Handler can handle debates overview endpoint."""
        assert handler.can_handle("/api/analytics/debates/overview")

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

    def test_can_handle_agents_comparison(self, handler):
        """Handler can handle agents comparison endpoint."""
        assert handler.can_handle("/api/analytics/agents/comparison")

    def test_can_handle_agents_trends(self, handler):
        """Handler can handle agents trends endpoint."""
        assert handler.can_handle("/api/analytics/agents/trends")

    def test_can_handle_agent_performance(self, handler):
        """Handler can handle agent performance endpoint with dynamic agent_id."""
        assert handler.can_handle("/api/analytics/agents/claude/performance")
        assert handler.can_handle("/api/analytics/agents/gpt-4/performance")
        assert handler.can_handle("/api/analytics/agents/test_agent-123/performance")

    def test_can_handle_usage_tokens(self, handler):
        """Handler can handle usage tokens endpoint."""
        assert handler.can_handle("/api/analytics/usage/tokens")

    def test_can_handle_usage_costs(self, handler):
        """Handler can handle usage costs endpoint."""
        assert handler.can_handle("/api/analytics/usage/costs")

    def test_can_handle_usage_active_users(self, handler):
        """Handler can handle active users endpoint."""
        assert handler.can_handle("/api/analytics/usage/active_users")

    def test_can_handle_with_version_prefix(self, handler):
        """Handler can handle endpoints with version prefix."""
        assert handler.can_handle("/api/v1/analytics/debates/overview")
        assert handler.can_handle("/api/v2/analytics/agents/leaderboard")

    def test_cannot_handle_unknown_path(self, handler):
        """Handler cannot handle unknown paths."""
        assert not handler.can_handle("/api/analytics/unknown")
        assert not handler.can_handle("/api/v1/other")
        assert not handler.can_handle("/api/debates")
        assert not handler.can_handle("/api/analytics")

    def test_cannot_handle_invalid_agent_performance_path(self, handler):
        """Handler cannot handle malformed agent performance paths."""
        assert not handler.can_handle("/api/analytics/agents/performance")
        assert not handler.can_handle("/api/analytics/agents//performance")

    def test_routes_list_complete(self, handler):
        """ROUTES list contains all expected static endpoints."""
        assert len(handler.ROUTES) >= 10
        assert "/api/analytics/debates/overview" in handler.ROUTES
        assert "/api/analytics/agents/leaderboard" in handler.ROUTES
        assert "/api/analytics/usage/tokens" in handler.ROUTES


# =============================================================================
# Tests for Authentication and Rate Limiting
# =============================================================================


class TestAuthenticationAndRateLimiting:
    """Tests for authentication and rate limiting."""

    @pytest.mark.asyncio
    async def test_returns_401_when_unauthenticated(self, handler, mock_http_handler):
        """Handler returns 401 when not authenticated."""
        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            from aragora.server.handlers.secure import UnauthorizedError

            mock_auth.side_effect = UnauthorizedError("Authentication required")

            result = await handler.handle(
                "/api/analytics/debates/overview",
                {},
                mock_http_handler,
            )

            assert result is not None
            assert result.status_code == 401

    @pytest.mark.asyncio
    async def test_returns_403_when_permission_denied(
        self, handler, mock_http_handler, mock_auth_context
    ):
        """Handler returns 403 when permission denied."""
        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            with patch.object(handler, "check_permission") as mock_check:
                from aragora.server.handlers.secure import ForbiddenError

                mock_auth.return_value = mock_auth_context
                mock_check.side_effect = ForbiddenError(
                    "Permission denied", permission="analytics:read"
                )

                result = await handler.handle(
                    "/api/analytics/debates/overview",
                    {},
                    mock_http_handler,
                )

                assert result is not None
                assert result.status_code == 403

    @pytest.mark.asyncio
    async def test_rate_limit_returns_429(self, handler, mock_http_handler):
        """Handler returns 429 when rate limited."""
        with patch(
            "aragora.server.handlers._analytics_metrics_impl._analytics_metrics_limiter"
        ) as mock_limiter:
            mock_limiter.is_allowed.return_value = False

            result = await handler.handle(
                "/api/analytics/debates/overview",
                {},
                mock_http_handler,
            )

            assert result is not None
            assert result.status_code == 429


# =============================================================================
# Tests for Debate Analytics Endpoints
# =============================================================================


class TestDebatesOverview:
    """Tests for GET /api/analytics/debates/overview."""

    @pytest.mark.asyncio
    async def test_debates_overview_returns_metrics(
        self, handler, mock_http_handler, mock_auth_context, mock_storage
    ):
        """Debates overview returns expected metrics."""
        handler.ctx["storage"] = mock_storage

        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=mock_auth_context
        ):
            with patch.object(handler, "check_permission", return_value=True):
                with patch(
                    "aragora.server.handlers._analytics_metrics_impl._analytics_metrics_limiter"
                ) as mock_limiter:
                    mock_limiter.is_allowed.return_value = True

                    result = await handler.handle(
                        "/api/analytics/debates/overview",
                        {"time_range": "30d"},
                        mock_http_handler,
                    )

                    assert result is not None
                    assert result.status_code == 200
                    data = json.loads(result.body)
                    assert "total_debates" in data
                    assert "consensus_rate" in data
                    assert "avg_rounds" in data
                    assert "generated_at" in data

    @pytest.mark.asyncio
    async def test_debates_overview_no_storage_returns_empty(
        self, handler, mock_http_handler, mock_auth_context
    ):
        """Debates overview returns empty metrics when no storage."""
        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=mock_auth_context
        ):
            with patch.object(handler, "check_permission", return_value=True):
                with patch(
                    "aragora.server.handlers._analytics_metrics_impl._analytics_metrics_limiter"
                ) as mock_limiter:
                    mock_limiter.is_allowed.return_value = True

                    result = await handler.handle(
                        "/api/analytics/debates/overview",
                        {},
                        mock_http_handler,
                    )

                    assert result is not None
                    assert result.status_code == 200
                    data = json.loads(result.body)
                    assert data["total_debates"] == 0

    @pytest.mark.asyncio
    async def test_debates_overview_invalid_time_range_defaults_to_30d(
        self, handler, mock_http_handler, mock_auth_context, mock_storage
    ):
        """Invalid time range defaults to 30d."""
        handler.ctx["storage"] = mock_storage

        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=mock_auth_context
        ):
            with patch.object(handler, "check_permission", return_value=True):
                with patch(
                    "aragora.server.handlers._analytics_metrics_impl._analytics_metrics_limiter"
                ) as mock_limiter:
                    mock_limiter.is_allowed.return_value = True

                    result = await handler.handle(
                        "/api/analytics/debates/overview",
                        {"time_range": "invalid"},
                        mock_http_handler,
                    )

                    assert result is not None
                    assert result.status_code == 200
                    data = json.loads(result.body)
                    assert data["time_range"] == "30d"


class TestDebatesTrends:
    """Tests for GET /api/analytics/debates/trends."""

    @pytest.mark.asyncio
    async def test_debates_trends_returns_data_points(
        self, handler, mock_http_handler, mock_auth_context, mock_storage
    ):
        """Debates trends returns data points by period."""
        handler.ctx["storage"] = mock_storage

        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=mock_auth_context
        ):
            with patch.object(handler, "check_permission", return_value=True):
                with patch(
                    "aragora.server.handlers._analytics_metrics_impl._analytics_metrics_limiter"
                ) as mock_limiter:
                    mock_limiter.is_allowed.return_value = True

                    result = await handler.handle(
                        "/api/analytics/debates/trends",
                        {"time_range": "30d", "granularity": "daily"},
                        mock_http_handler,
                    )

                    assert result is not None
                    assert result.status_code == 200
                    data = json.loads(result.body)
                    assert "data_points" in data
                    assert "granularity" in data
                    assert data["granularity"] == "daily"

    @pytest.mark.asyncio
    async def test_debates_trends_weekly_granularity(
        self, handler, mock_http_handler, mock_auth_context, mock_storage
    ):
        """Debates trends with weekly granularity."""
        handler.ctx["storage"] = mock_storage

        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=mock_auth_context
        ):
            with patch.object(handler, "check_permission", return_value=True):
                with patch(
                    "aragora.server.handlers._analytics_metrics_impl._analytics_metrics_limiter"
                ) as mock_limiter:
                    mock_limiter.is_allowed.return_value = True

                    result = await handler.handle(
                        "/api/analytics/debates/trends",
                        {"granularity": "weekly"},
                        mock_http_handler,
                    )

                    assert result is not None
                    data = json.loads(result.body)
                    assert data["granularity"] == "weekly"

    @pytest.mark.asyncio
    async def test_debates_trends_monthly_granularity(
        self, handler, mock_http_handler, mock_auth_context, mock_storage
    ):
        """Debates trends with monthly granularity."""
        handler.ctx["storage"] = mock_storage

        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=mock_auth_context
        ):
            with patch.object(handler, "check_permission", return_value=True):
                with patch(
                    "aragora.server.handlers._analytics_metrics_impl._analytics_metrics_limiter"
                ) as mock_limiter:
                    mock_limiter.is_allowed.return_value = True

                    result = await handler.handle(
                        "/api/analytics/debates/trends",
                        {"granularity": "monthly"},
                        mock_http_handler,
                    )

                    assert result is not None
                    data = json.loads(result.body)
                    assert data["granularity"] == "monthly"

    @pytest.mark.asyncio
    async def test_debates_trends_invalid_granularity_defaults_to_daily(
        self, handler, mock_http_handler, mock_auth_context, mock_storage
    ):
        """Invalid granularity defaults to daily."""
        handler.ctx["storage"] = mock_storage

        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=mock_auth_context
        ):
            with patch.object(handler, "check_permission", return_value=True):
                with patch(
                    "aragora.server.handlers._analytics_metrics_impl._analytics_metrics_limiter"
                ) as mock_limiter:
                    mock_limiter.is_allowed.return_value = True

                    result = await handler.handle(
                        "/api/analytics/debates/trends",
                        {"granularity": "invalid"},
                        mock_http_handler,
                    )

                    assert result is not None
                    data = json.loads(result.body)
                    assert data["granularity"] == "daily"


class TestDebatesTopics:
    """Tests for GET /api/analytics/debates/topics."""

    @pytest.mark.asyncio
    async def test_debates_topics_returns_topic_distribution(
        self, handler, mock_http_handler, mock_auth_context, mock_storage
    ):
        """Debates topics returns topic distribution."""
        handler.ctx["storage"] = mock_storage

        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=mock_auth_context
        ):
            with patch.object(handler, "check_permission", return_value=True):
                with patch(
                    "aragora.server.handlers._analytics_metrics_impl._analytics_metrics_limiter"
                ) as mock_limiter:
                    mock_limiter.is_allowed.return_value = True

                    result = await handler.handle(
                        "/api/analytics/debates/topics",
                        {"time_range": "30d"},
                        mock_http_handler,
                    )

                    assert result is not None
                    assert result.status_code == 200
                    data = json.loads(result.body)
                    assert "topics" in data
                    assert "total_debates" in data

    @pytest.mark.asyncio
    async def test_debates_topics_respects_limit(
        self, handler, mock_http_handler, mock_auth_context, mock_storage
    ):
        """Debates topics respects the limit parameter."""
        handler.ctx["storage"] = mock_storage

        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=mock_auth_context
        ):
            with patch.object(handler, "check_permission", return_value=True):
                with patch(
                    "aragora.server.handlers._analytics_metrics_impl._analytics_metrics_limiter"
                ) as mock_limiter:
                    mock_limiter.is_allowed.return_value = True

                    result = await handler.handle(
                        "/api/analytics/debates/topics",
                        {"limit": "5"},
                        mock_http_handler,
                    )

                    assert result is not None
                    data = json.loads(result.body)
                    assert len(data["topics"]) <= 5


class TestDebatesOutcomes:
    """Tests for GET /api/analytics/debates/outcomes."""

    @pytest.mark.asyncio
    async def test_debates_outcomes_returns_distribution(
        self, handler, mock_http_handler, mock_auth_context, mock_storage
    ):
        """Debates outcomes returns outcome distribution."""
        handler.ctx["storage"] = mock_storage

        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=mock_auth_context
        ):
            with patch.object(handler, "check_permission", return_value=True):
                with patch(
                    "aragora.server.handlers._analytics_metrics_impl._analytics_metrics_limiter"
                ) as mock_limiter:
                    mock_limiter.is_allowed.return_value = True

                    result = await handler.handle(
                        "/api/analytics/debates/outcomes",
                        {"time_range": "30d"},
                        mock_http_handler,
                    )

                    assert result is not None
                    assert result.status_code == 200
                    data = json.loads(result.body)
                    assert "outcomes" in data
                    assert "consensus" in data["outcomes"]
                    assert "majority" in data["outcomes"]
                    assert "dissent" in data["outcomes"]
                    assert "no_resolution" in data["outcomes"]
                    assert "by_confidence" in data


# =============================================================================
# Tests for Agent Performance Endpoints
# =============================================================================


class TestAgentsLeaderboard:
    """Tests for GET /api/analytics/agents/leaderboard."""

    @pytest.mark.asyncio
    async def test_agents_leaderboard_returns_rankings(
        self, handler, mock_http_handler, mock_auth_context, mock_elo_system
    ):
        """Agents leaderboard returns agent rankings."""
        handler.ctx["elo_system"] = mock_elo_system

        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=mock_auth_context
        ):
            with patch.object(handler, "check_permission", return_value=True):
                with patch(
                    "aragora.server.handlers._analytics_metrics_impl._analytics_metrics_limiter"
                ) as mock_limiter:
                    mock_limiter.is_allowed.return_value = True

                    result = await handler.handle(
                        "/api/analytics/agents/leaderboard",
                        {},
                        mock_http_handler,
                    )

                    assert result is not None
                    assert result.status_code == 200
                    data = json.loads(result.body)
                    assert "leaderboard" in data
                    assert "total_agents" in data
                    assert len(data["leaderboard"]) > 0

                    first_agent = data["leaderboard"][0]
                    assert "rank" in first_agent
                    assert "agent_name" in first_agent
                    assert "elo" in first_agent
                    assert "win_rate" in first_agent

    @pytest.mark.asyncio
    async def test_agents_leaderboard_no_elo_system_returns_empty(
        self, handler, mock_http_handler, mock_auth_context
    ):
        """Agents leaderboard returns empty when no ELO system."""
        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=mock_auth_context
        ):
            with patch.object(handler, "check_permission", return_value=True):
                with patch(
                    "aragora.server.handlers._analytics_metrics_impl._analytics_metrics_limiter"
                ) as mock_limiter:
                    mock_limiter.is_allowed.return_value = True

                    result = await handler.handle(
                        "/api/analytics/agents/leaderboard",
                        {},
                        mock_http_handler,
                    )

                    assert result is not None
                    data = json.loads(result.body)
                    assert data["leaderboard"] == []
                    assert data["total_agents"] == 0

    @pytest.mark.asyncio
    async def test_agents_leaderboard_respects_limit(
        self, handler, mock_http_handler, mock_auth_context, mock_elo_system
    ):
        """Agents leaderboard respects limit parameter."""
        handler.ctx["elo_system"] = mock_elo_system
        mock_elo_system.get_leaderboard.return_value = mock_elo_system.list_agents()[:3]

        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=mock_auth_context
        ):
            with patch.object(handler, "check_permission", return_value=True):
                with patch(
                    "aragora.server.handlers._analytics_metrics_impl._analytics_metrics_limiter"
                ) as mock_limiter:
                    mock_limiter.is_allowed.return_value = True

                    result = await handler.handle(
                        "/api/analytics/agents/leaderboard",
                        {"limit": "3"},
                        mock_http_handler,
                    )

                    assert result is not None
                    data = json.loads(result.body)
                    assert len(data["leaderboard"]) == 3

    @pytest.mark.asyncio
    async def test_agents_leaderboard_filters_by_domain(
        self, handler, mock_http_handler, mock_auth_context, mock_elo_system
    ):
        """Agents leaderboard filters by domain."""
        handler.ctx["elo_system"] = mock_elo_system

        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=mock_auth_context
        ):
            with patch.object(handler, "check_permission", return_value=True):
                with patch(
                    "aragora.server.handlers._analytics_metrics_impl._analytics_metrics_limiter"
                ) as mock_limiter:
                    mock_limiter.is_allowed.return_value = True

                    result = await handler.handle(
                        "/api/analytics/agents/leaderboard",
                        {"domain": "security"},
                        mock_http_handler,
                    )

                    assert result is not None
                    data = json.loads(result.body)
                    assert data.get("domain") == "security"


class TestAgentPerformance:
    """Tests for GET /api/analytics/agents/{agent_id}/performance."""

    @pytest.mark.asyncio
    async def test_agent_performance_returns_stats(
        self, handler, mock_http_handler, mock_auth_context, mock_elo_system
    ):
        """Agent performance returns individual agent stats."""
        handler.ctx["elo_system"] = mock_elo_system

        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=mock_auth_context
        ):
            with patch.object(handler, "check_permission", return_value=True):
                with patch(
                    "aragora.server.handlers._analytics_metrics_impl._analytics_metrics_limiter"
                ) as mock_limiter:
                    mock_limiter.is_allowed.return_value = True

                    result = await handler.handle(
                        "/api/analytics/agents/claude/performance",
                        {},
                        mock_http_handler,
                    )

                    assert result is not None
                    assert result.status_code == 200
                    data = json.loads(result.body)
                    assert data["agent_id"] == "claude"
                    assert "elo" in data
                    assert "wins" in data
                    assert "losses" in data
                    assert "win_rate" in data
                    assert "elo_history" in data

    @pytest.mark.asyncio
    async def test_agent_performance_not_found(
        self, handler, mock_http_handler, mock_auth_context, mock_elo_system
    ):
        """Agent performance returns 404 for unknown agent."""
        handler.ctx["elo_system"] = mock_elo_system
        mock_elo_system.get_rating.side_effect = ValueError("Agent not found")

        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=mock_auth_context
        ):
            with patch.object(handler, "check_permission", return_value=True):
                with patch(
                    "aragora.server.handlers._analytics_metrics_impl._analytics_metrics_limiter"
                ) as mock_limiter:
                    mock_limiter.is_allowed.return_value = True

                    result = await handler.handle(
                        "/api/analytics/agents/unknown_agent/performance",
                        {},
                        mock_http_handler,
                    )

                    assert result is not None
                    assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_agent_performance_no_elo_system_returns_503(
        self, handler, mock_http_handler, mock_auth_context
    ):
        """Agent performance returns 503 when no ELO system."""
        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=mock_auth_context
        ):
            with patch.object(handler, "check_permission", return_value=True):
                with patch(
                    "aragora.server.handlers._analytics_metrics_impl._analytics_metrics_limiter"
                ) as mock_limiter:
                    mock_limiter.is_allowed.return_value = True

                    result = await handler.handle(
                        "/api/analytics/agents/claude/performance",
                        {},
                        mock_http_handler,
                    )

                    assert result is not None
                    assert result.status_code == 503


class TestAgentsComparison:
    """Tests for GET /api/analytics/agents/comparison."""

    @pytest.mark.asyncio
    async def test_agents_comparison_returns_comparison(
        self, handler, mock_http_handler, mock_auth_context, mock_elo_system
    ):
        """Agents comparison returns comparison data."""
        handler.ctx["elo_system"] = mock_elo_system

        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=mock_auth_context
        ):
            with patch.object(handler, "check_permission", return_value=True):
                with patch(
                    "aragora.server.handlers._analytics_metrics_impl._analytics_metrics_limiter"
                ) as mock_limiter:
                    mock_limiter.is_allowed.return_value = True

                    result = await handler.handle(
                        "/api/analytics/agents/comparison",
                        {"agents": "claude,gpt-4,gemini"},
                        mock_http_handler,
                    )

                    assert result is not None
                    assert result.status_code == 200
                    data = json.loads(result.body)
                    assert "agents" in data
                    assert "comparison" in data
                    assert "head_to_head" in data
                    assert len(data["agents"]) == 3

    @pytest.mark.asyncio
    async def test_agents_comparison_requires_agents_param(
        self, handler, mock_http_handler, mock_auth_context, mock_elo_system
    ):
        """Agents comparison returns 400 without agents param."""
        handler.ctx["elo_system"] = mock_elo_system

        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=mock_auth_context
        ):
            with patch.object(handler, "check_permission", return_value=True):
                with patch(
                    "aragora.server.handlers._analytics_metrics_impl._analytics_metrics_limiter"
                ) as mock_limiter:
                    mock_limiter.is_allowed.return_value = True

                    result = await handler.handle(
                        "/api/analytics/agents/comparison",
                        {},
                        mock_http_handler,
                    )

                    assert result is not None
                    assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_agents_comparison_requires_at_least_two_agents(
        self, handler, mock_http_handler, mock_auth_context, mock_elo_system
    ):
        """Agents comparison returns 400 with less than 2 agents."""
        handler.ctx["elo_system"] = mock_elo_system

        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=mock_auth_context
        ):
            with patch.object(handler, "check_permission", return_value=True):
                with patch(
                    "aragora.server.handlers._analytics_metrics_impl._analytics_metrics_limiter"
                ) as mock_limiter:
                    mock_limiter.is_allowed.return_value = True

                    result = await handler.handle(
                        "/api/analytics/agents/comparison",
                        {"agents": "claude"},
                        mock_http_handler,
                    )

                    assert result is not None
                    assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_agents_comparison_limits_to_10_agents(
        self, handler, mock_http_handler, mock_auth_context, mock_elo_system
    ):
        """Agents comparison returns 400 with more than 10 agents."""
        handler.ctx["elo_system"] = mock_elo_system

        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=mock_auth_context
        ):
            with patch.object(handler, "check_permission", return_value=True):
                with patch(
                    "aragora.server.handlers._analytics_metrics_impl._analytics_metrics_limiter"
                ) as mock_limiter:
                    mock_limiter.is_allowed.return_value = True

                    agents = ",".join([f"agent{i}" for i in range(12)])
                    result = await handler.handle(
                        "/api/analytics/agents/comparison",
                        {"agents": agents},
                        mock_http_handler,
                    )

                    assert result is not None
                    assert result.status_code == 400


class TestAgentsTrends:
    """Tests for GET /api/analytics/agents/trends."""

    @pytest.mark.asyncio
    async def test_agents_trends_returns_elo_history(
        self, handler, mock_http_handler, mock_auth_context, mock_elo_system
    ):
        """Agents trends returns ELO history by agent."""
        handler.ctx["elo_system"] = mock_elo_system

        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=mock_auth_context
        ):
            with patch.object(handler, "check_permission", return_value=True):
                with patch(
                    "aragora.server.handlers._analytics_metrics_impl._analytics_metrics_limiter"
                ) as mock_limiter:
                    mock_limiter.is_allowed.return_value = True

                    result = await handler.handle(
                        "/api/analytics/agents/trends",
                        {"time_range": "30d"},
                        mock_http_handler,
                    )

                    assert result is not None
                    assert result.status_code == 200
                    data = json.loads(result.body)
                    assert "agents" in data
                    assert "trends" in data
                    assert "granularity" in data

    @pytest.mark.asyncio
    async def test_agents_trends_with_specific_agents(
        self, handler, mock_http_handler, mock_auth_context, mock_elo_system
    ):
        """Agents trends with specific agents specified."""
        handler.ctx["elo_system"] = mock_elo_system

        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=mock_auth_context
        ):
            with patch.object(handler, "check_permission", return_value=True):
                with patch(
                    "aragora.server.handlers._analytics_metrics_impl._analytics_metrics_limiter"
                ) as mock_limiter:
                    mock_limiter.is_allowed.return_value = True

                    result = await handler.handle(
                        "/api/analytics/agents/trends",
                        {"agents": "claude,gpt-4"},
                        mock_http_handler,
                    )

                    assert result is not None
                    data = json.loads(result.body)
                    assert "claude" in data["agents"] or "gpt-4" in data["agents"]

    @pytest.mark.asyncio
    async def test_agents_trends_no_elo_system_returns_503(
        self, handler, mock_http_handler, mock_auth_context
    ):
        """Agents trends returns 503 when no ELO system."""
        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=mock_auth_context
        ):
            with patch.object(handler, "check_permission", return_value=True):
                with patch(
                    "aragora.server.handlers._analytics_metrics_impl._analytics_metrics_limiter"
                ) as mock_limiter:
                    mock_limiter.is_allowed.return_value = True

                    result = await handler.handle(
                        "/api/analytics/agents/trends",
                        {},
                        mock_http_handler,
                    )

                    assert result is not None
                    assert result.status_code == 503


# =============================================================================
# Tests for Usage Analytics Endpoints
# =============================================================================


class TestUsageTokens:
    """Tests for GET /api/analytics/usage/tokens."""

    @pytest.mark.asyncio
    async def test_usage_tokens_returns_token_stats(
        self, handler, mock_http_handler, mock_auth_context
    ):
        """Usage tokens returns token consumption stats."""
        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=mock_auth_context
        ):
            with patch.object(handler, "check_permission", return_value=True):
                with patch(
                    "aragora.server.handlers._analytics_metrics_impl._analytics_metrics_limiter"
                ) as mock_limiter:
                    mock_limiter.is_allowed.return_value = True

                    with patch(
                        "aragora.server.handlers._analytics_metrics_impl.get_cost_tracker"
                    ) as mock_tracker:
                        mock_tracker.return_value.get_workspace_stats.return_value = {
                            "total_tokens_in": 5000000,
                            "total_tokens_out": 1000000,
                            "cost_by_agent": {},
                            "cost_by_model": {},
                        }

                        result = await handler.handle(
                            "/api/analytics/usage/tokens",
                            {"org_id": "test-org"},
                            mock_http_handler,
                        )

                        assert result is not None
                        assert result.status_code == 200
                        data = json.loads(result.body)
                        assert "summary" in data
                        assert "total_tokens_in" in data["summary"]
                        assert "total_tokens_out" in data["summary"]

    @pytest.mark.asyncio
    async def test_usage_tokens_requires_org_id(
        self, handler, mock_http_handler, mock_auth_context
    ):
        """Usage tokens returns 400 without org_id."""
        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=mock_auth_context
        ):
            with patch.object(handler, "check_permission", return_value=True):
                with patch(
                    "aragora.server.handlers._analytics_metrics_impl._analytics_metrics_limiter"
                ) as mock_limiter:
                    mock_limiter.is_allowed.return_value = True

                    result = await handler.handle(
                        "/api/analytics/usage/tokens",
                        {},
                        mock_http_handler,
                    )

                    assert result is not None
                    assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_usage_tokens_handles_missing_tracker(
        self, handler, mock_http_handler, mock_auth_context
    ):
        """Usage tokens handles missing cost tracker gracefully."""
        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=mock_auth_context
        ):
            with patch.object(handler, "check_permission", return_value=True):
                with patch(
                    "aragora.server.handlers._analytics_metrics_impl._analytics_metrics_limiter"
                ) as mock_limiter:
                    mock_limiter.is_allowed.return_value = True

                    with patch.dict("sys.modules", {"aragora.billing.cost_tracker": None}):
                        result = await handler.handle(
                            "/api/analytics/usage/tokens",
                            {"org_id": "test-org"},
                            mock_http_handler,
                        )

                        assert result is not None
                        assert result.status_code == 200


class TestUsageCosts:
    """Tests for GET /api/analytics/usage/costs."""

    @pytest.mark.asyncio
    async def test_usage_costs_returns_cost_breakdown(
        self, handler, mock_http_handler, mock_auth_context
    ):
        """Usage costs returns cost breakdown."""
        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=mock_auth_context
        ):
            with patch.object(handler, "check_permission", return_value=True):
                with patch(
                    "aragora.server.handlers._analytics_metrics_impl._analytics_metrics_limiter"
                ) as mock_limiter:
                    mock_limiter.is_allowed.return_value = True

                    with patch(
                        "aragora.server.handlers._analytics_metrics_impl.get_cost_tracker"
                    ) as mock_tracker:
                        mock_tracker.return_value.get_workspace_stats.return_value = {
                            "total_cost_usd": "125.50",
                            "total_api_calls": 500,
                            "cost_by_agent": {"claude": "80.00", "gpt-4": "45.50"},
                            "cost_by_model": {},
                        }

                        result = await handler.handle(
                            "/api/analytics/usage/costs",
                            {"org_id": "test-org"},
                            mock_http_handler,
                        )

                        assert result is not None
                        assert result.status_code == 200
                        data = json.loads(result.body)
                        assert "summary" in data
                        assert "total_cost_usd" in data["summary"]
                        assert "by_provider" in data

    @pytest.mark.asyncio
    async def test_usage_costs_requires_org_id(self, handler, mock_http_handler, mock_auth_context):
        """Usage costs returns 400 without org_id."""
        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=mock_auth_context
        ):
            with patch.object(handler, "check_permission", return_value=True):
                with patch(
                    "aragora.server.handlers._analytics_metrics_impl._analytics_metrics_limiter"
                ) as mock_limiter:
                    mock_limiter.is_allowed.return_value = True

                    result = await handler.handle(
                        "/api/analytics/usage/costs",
                        {},
                        mock_http_handler,
                    )

                    assert result is not None
                    assert result.status_code == 400


class TestActiveUsers:
    """Tests for GET /api/analytics/usage/active_users."""

    @pytest.mark.asyncio
    async def test_active_users_returns_counts(self, handler, mock_http_handler, mock_auth_context):
        """Active users returns user counts."""
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

        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=mock_auth_context
        ):
            with patch.object(handler, "check_permission", return_value=True):
                with patch(
                    "aragora.server.handlers._analytics_metrics_impl._analytics_metrics_limiter"
                ) as mock_limiter:
                    mock_limiter.is_allowed.return_value = True

                    result = await handler.handle(
                        "/api/analytics/usage/active_users",
                        {"org_id": "test-org"},
                        mock_http_handler,
                    )

                    assert result is not None
                    assert result.status_code == 200
                    data = json.loads(result.body)
                    assert "active_users" in data
                    assert data["active_users"]["daily"] == 25

    @pytest.mark.asyncio
    async def test_active_users_without_user_store(
        self, handler, mock_http_handler, mock_auth_context
    ):
        """Active users returns empty when no user store."""
        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=mock_auth_context
        ):
            with patch.object(handler, "check_permission", return_value=True):
                with patch(
                    "aragora.server.handlers._analytics_metrics_impl._analytics_metrics_limiter"
                ) as mock_limiter:
                    mock_limiter.is_allowed.return_value = True

                    result = await handler.handle(
                        "/api/analytics/usage/active_users",
                        {},
                        mock_http_handler,
                    )

                    assert result is not None
                    assert result.status_code == 200
                    data = json.loads(result.body)
                    assert data["active_users"]["daily"] == 0

    @pytest.mark.asyncio
    async def test_active_users_invalid_time_range_defaults(
        self, handler, mock_http_handler, mock_auth_context
    ):
        """Active users defaults to 30d for invalid time_range."""
        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=mock_auth_context
        ):
            with patch.object(handler, "check_permission", return_value=True):
                with patch(
                    "aragora.server.handlers._analytics_metrics_impl._analytics_metrics_limiter"
                ) as mock_limiter:
                    mock_limiter.is_allowed.return_value = True

                    result = await handler.handle(
                        "/api/analytics/usage/active_users",
                        {"time_range": "invalid"},
                        mock_http_handler,
                    )

                    assert result is not None
                    data = json.loads(result.body)
                    assert data["time_range"] == "30d"


# =============================================================================
# Tests for Unknown Path Handling
# =============================================================================


class TestUnknownPathHandling:
    """Tests for unknown path handling."""

    @pytest.mark.asyncio
    async def test_unknown_path_returns_none(self, handler, mock_http_handler, mock_auth_context):
        """Unknown path returns None for dispatch to continue."""
        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=mock_auth_context
        ):
            with patch.object(handler, "check_permission", return_value=True):
                with patch(
                    "aragora.server.handlers._analytics_metrics_impl._analytics_metrics_limiter"
                ) as mock_limiter:
                    mock_limiter.is_allowed.return_value = True

                    result = await handler.handle(
                        "/api/v1/other/endpoint",
                        {},
                        mock_http_handler,
                    )

                    assert result is None

    @pytest.mark.asyncio
    async def test_unknown_analytics_path_returns_none(
        self, handler, mock_http_handler, mock_auth_context
    ):
        """Unknown analytics path returns None."""
        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=mock_auth_context
        ):
            with patch.object(handler, "check_permission", return_value=True):
                with patch(
                    "aragora.server.handlers._analytics_metrics_impl._analytics_metrics_limiter"
                ) as mock_limiter:
                    mock_limiter.is_allowed.return_value = True

                    result = await handler.handle(
                        "/api/analytics/nonexistent/path",
                        {},
                        mock_http_handler,
                    )

                    assert result is None


# =============================================================================
# Tests for Constants
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_valid_granularities(self):
        """VALID_GRANULARITIES contains expected values."""
        assert "daily" in VALID_GRANULARITIES
        assert "weekly" in VALID_GRANULARITIES
        assert "monthly" in VALID_GRANULARITIES
        assert len(VALID_GRANULARITIES) == 3

    def test_valid_time_ranges(self):
        """VALID_TIME_RANGES contains expected values."""
        assert "7d" in VALID_TIME_RANGES
        assert "14d" in VALID_TIME_RANGES
        assert "30d" in VALID_TIME_RANGES
        assert "90d" in VALID_TIME_RANGES
        assert "180d" in VALID_TIME_RANGES
        assert "365d" in VALID_TIME_RANGES
        assert "all" in VALID_TIME_RANGES

    def test_analytics_metrics_permission(self):
        """ANALYTICS_METRICS_PERMISSION is correct."""
        assert ANALYTICS_METRICS_PERMISSION == "analytics:read"


# =============================================================================
# Tests for Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_debates_with_object_instead_of_dict(
        self, handler, mock_http_handler, mock_auth_context
    ):
        """Handler handles debates that are objects instead of dicts."""

        class DebateObject:
            def __init__(self):
                self.id = "debate-1"
                self.created_at = datetime.now(timezone.utc).isoformat()
                self.consensus_reached = True
                self.result = {"rounds_used": 3, "confidence": 0.85}
                self.agents = ["claude", "gpt-4"]
                self.task = "Test task"
                self.domain = "security"

        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = [DebateObject()]
        handler.ctx["storage"] = mock_storage

        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=mock_auth_context
        ):
            with patch.object(handler, "check_permission", return_value=True):
                with patch(
                    "aragora.server.handlers._analytics_metrics_impl._analytics_metrics_limiter"
                ) as mock_limiter:
                    mock_limiter.is_allowed.return_value = True

                    result = await handler.handle(
                        "/api/analytics/debates/overview",
                        {},
                        mock_http_handler,
                    )

                    assert result is not None
                    assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_debates_with_datetime_created_at(
        self, handler, mock_http_handler, mock_auth_context
    ):
        """Handler handles debates with datetime objects for created_at."""
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = [
            {
                "id": "debate-1",
                "created_at": datetime.now(timezone.utc),
                "consensus_reached": True,
                "result": {"rounds_used": 3},
                "agents": ["claude"],
                "task": "Test",
            }
        ]
        handler.ctx["storage"] = mock_storage

        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=mock_auth_context
        ):
            with patch.object(handler, "check_permission", return_value=True):
                with patch(
                    "aragora.server.handlers._analytics_metrics_impl._analytics_metrics_limiter"
                ) as mock_limiter:
                    mock_limiter.is_allowed.return_value = True

                    result = await handler.handle(
                        "/api/analytics/debates/overview",
                        {},
                        mock_http_handler,
                    )

                    assert result is not None
                    assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_agent_with_missing_calibration_score(
        self, handler, mock_http_handler, mock_auth_context
    ):
        """Handler handles agents without calibration_score attribute."""
        mock_elo_system = MagicMock()
        agent = MagicMock()
        agent.agent_name = "test-agent"
        agent.elo = 1500
        agent.wins = 50
        agent.losses = 20
        agent.draws = 5
        agent.win_rate = 0.67
        agent.games_played = 75
        # Explicitly remove calibration_score
        del agent.calibration_score

        mock_elo_system.get_leaderboard.return_value = [agent]
        mock_elo_system.list_agents.return_value = [agent]
        handler.ctx["elo_system"] = mock_elo_system

        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=mock_auth_context
        ):
            with patch.object(handler, "check_permission", return_value=True):
                with patch(
                    "aragora.server.handlers._analytics_metrics_impl._analytics_metrics_limiter"
                ) as mock_limiter:
                    mock_limiter.is_allowed.return_value = True

                    result = await handler.handle(
                        "/api/analytics/agents/leaderboard",
                        {},
                        mock_http_handler,
                    )

                    assert result is not None
                    assert result.status_code == 200
                    data = json.loads(result.body)
                    assert "calibration_score" not in data["leaderboard"][0]

    @pytest.mark.asyncio
    async def test_debates_with_missing_result_fields(
        self, handler, mock_http_handler, mock_auth_context
    ):
        """Handler handles debates with missing result fields."""
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = [
            {
                "id": "debate-1",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "consensus_reached": True,
                "result": {},  # Empty result
                "agents": [],
                "task": "Test",
            }
        ]
        handler.ctx["storage"] = mock_storage

        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=mock_auth_context
        ):
            with patch.object(handler, "check_permission", return_value=True):
                with patch(
                    "aragora.server.handlers._analytics_metrics_impl._analytics_metrics_limiter"
                ) as mock_limiter:
                    mock_limiter.is_allowed.return_value = True

                    result = await handler.handle(
                        "/api/analytics/debates/overview",
                        {},
                        mock_http_handler,
                    )

                    assert result is not None
                    assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_head_to_head_error_handling(
        self, handler, mock_http_handler, mock_auth_context, mock_elo_system
    ):
        """Handler handles errors in head-to-head computation."""
        handler.ctx["elo_system"] = mock_elo_system
        mock_elo_system.get_head_to_head.side_effect = Exception("Database error")

        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=mock_auth_context
        ):
            with patch.object(handler, "check_permission", return_value=True):
                with patch(
                    "aragora.server.handlers._analytics_metrics_impl._analytics_metrics_limiter"
                ) as mock_limiter:
                    mock_limiter.is_allowed.return_value = True

                    result = await handler.handle(
                        "/api/analytics/agents/comparison",
                        {"agents": "claude,gpt-4"},
                        mock_http_handler,
                    )

                    # Should still succeed, just without head-to-head data
                    assert result is not None
                    assert result.status_code == 200
                    data = json.loads(result.body)
                    assert "head_to_head" in data

    @pytest.mark.asyncio
    async def test_elo_history_with_string_timestamps(
        self, handler, mock_http_handler, mock_auth_context, mock_elo_system
    ):
        """Handler handles ELO history with string timestamps."""
        handler.ctx["elo_system"] = mock_elo_system
        mock_elo_system.get_elo_history.return_value = [
            ("2026-01-15T10:00:00Z", 1500),
            ("2026-01-14T10:00:00Z", 1490),
        ]

        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=mock_auth_context
        ):
            with patch.object(handler, "check_permission", return_value=True):
                with patch(
                    "aragora.server.handlers._analytics_metrics_impl._analytics_metrics_limiter"
                ) as mock_limiter:
                    mock_limiter.is_allowed.return_value = True

                    result = await handler.handle(
                        "/api/analytics/agents/trends",
                        {},
                        mock_http_handler,
                    )

                    assert result is not None
                    assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_user_store_missing_methods(self, handler, mock_http_handler, mock_auth_context):
        """Handler handles user store without expected methods."""
        mock_user_store = MagicMock(spec=[])  # Empty spec - no methods
        handler.ctx["user_store"] = mock_user_store

        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=mock_auth_context
        ):
            with patch.object(handler, "check_permission", return_value=True):
                with patch(
                    "aragora.server.handlers._analytics_metrics_impl._analytics_metrics_limiter"
                ) as mock_limiter:
                    mock_limiter.is_allowed.return_value = True

                    result = await handler.handle(
                        "/api/analytics/usage/active_users",
                        {},
                        mock_http_handler,
                    )

                    assert result is not None
                    assert result.status_code == 200
                    data = json.loads(result.body)
                    assert data["active_users"]["daily"] == 0

    @pytest.mark.asyncio
    async def test_user_store_raises_exception(self, handler, mock_http_handler, mock_auth_context):
        """Handler handles user store exceptions gracefully."""
        mock_user_store = MagicMock()
        mock_user_store.get_active_user_counts.side_effect = Exception("Database error")
        handler.ctx["user_store"] = mock_user_store

        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=mock_auth_context
        ):
            with patch.object(handler, "check_permission", return_value=True):
                with patch(
                    "aragora.server.handlers._analytics_metrics_impl._analytics_metrics_limiter"
                ) as mock_limiter:
                    mock_limiter.is_allowed.return_value = True

                    result = await handler.handle(
                        "/api/analytics/usage/active_users",
                        {},
                        mock_http_handler,
                    )

                    assert result is not None
                    assert result.status_code == 200
                    data = json.loads(result.body)
                    assert "error" in data


# =============================================================================
# Tests for Response Format Validation
# =============================================================================


class TestResponseFormat:
    """Tests for validating response format consistency."""

    @pytest.mark.asyncio
    async def test_debates_overview_response_format(
        self, handler, mock_http_handler, mock_auth_context, mock_storage
    ):
        """Debates overview has correct response format."""
        handler.ctx["storage"] = mock_storage

        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=mock_auth_context
        ):
            with patch.object(handler, "check_permission", return_value=True):
                with patch(
                    "aragora.server.handlers._analytics_metrics_impl._analytics_metrics_limiter"
                ) as mock_limiter:
                    mock_limiter.is_allowed.return_value = True

                    result = await handler.handle(
                        "/api/analytics/debates/overview",
                        {},
                        mock_http_handler,
                    )

                    data = json.loads(result.body)
                    expected_keys = [
                        "time_range",
                        "total_debates",
                        "debates_this_period",
                        "debates_previous_period",
                        "growth_rate",
                        "consensus_reached",
                        "consensus_rate",
                        "avg_rounds",
                        "avg_agents_per_debate",
                        "avg_confidence",
                        "generated_at",
                    ]
                    for key in expected_keys:
                        assert key in data, f"Missing key: {key}"

    @pytest.mark.asyncio
    async def test_agents_leaderboard_response_format(
        self, handler, mock_http_handler, mock_auth_context, mock_elo_system
    ):
        """Agents leaderboard has correct response format."""
        handler.ctx["elo_system"] = mock_elo_system

        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=mock_auth_context
        ):
            with patch.object(handler, "check_permission", return_value=True):
                with patch(
                    "aragora.server.handlers._analytics_metrics_impl._analytics_metrics_limiter"
                ) as mock_limiter:
                    mock_limiter.is_allowed.return_value = True

                    result = await handler.handle(
                        "/api/analytics/agents/leaderboard",
                        {},
                        mock_http_handler,
                    )

                    data = json.loads(result.body)
                    assert "leaderboard" in data
                    assert "total_agents" in data
                    assert "generated_at" in data

                    if data["leaderboard"]:
                        agent = data["leaderboard"][0]
                        assert "rank" in agent
                        assert "agent_name" in agent
                        assert "elo" in agent
                        assert "wins" in agent
                        assert "losses" in agent
                        assert "draws" in agent
                        assert "win_rate" in agent
                        assert "games_played" in agent

    @pytest.mark.asyncio
    async def test_usage_costs_response_format(self, handler, mock_http_handler, mock_auth_context):
        """Usage costs has correct response format."""
        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=mock_auth_context
        ):
            with patch.object(handler, "check_permission", return_value=True):
                with patch(
                    "aragora.server.handlers._analytics_metrics_impl._analytics_metrics_limiter"
                ) as mock_limiter:
                    mock_limiter.is_allowed.return_value = True

                    with patch(
                        "aragora.server.handlers._analytics_metrics_impl.get_cost_tracker"
                    ) as mock_tracker:
                        mock_tracker.return_value.get_workspace_stats.return_value = {
                            "total_cost_usd": "100.00",
                            "total_api_calls": 100,
                            "cost_by_agent": {},
                            "cost_by_model": {},
                        }

                        result = await handler.handle(
                            "/api/analytics/usage/costs",
                            {"org_id": "test-org"},
                            mock_http_handler,
                        )

                        data = json.loads(result.body)
                        assert "org_id" in data
                        assert "time_range" in data
                        assert "summary" in data
                        assert "by_provider" in data
                        assert "by_model" in data
                        assert "generated_at" in data
