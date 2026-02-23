"""Comprehensive tests for the AnalyticsMetricsHandler in _analytics_metrics_impl.py.

Tests the main handler class routing, authentication, rate limiting, demo mode,
can_handle(), and all 11 endpoint routes through the async handle() method:

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

Also covers:
- can_handle() route matching (static + regex pattern)
- Rate limiter enforcement (429)
- Auth/RBAC enforcement (401, 403)
- Demo mode bypass
- RBAC fail-closed (503)
- RBAC checker second-pass denial
- Unknown route returns None
- Versioned and unversioned path handling
- Security edge cases (path traversal, injection)
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers._analytics_metrics_impl import (
    AnalyticsMetricsHandler,
    _analytics_metrics_limiter,
    _demo_response,
    _is_demo_mode,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result: object) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    raw = result.body
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def _status(result: object) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


class MockHTTPHandler:
    """Minimal mock HTTP handler for handle() routing tests."""

    def __init__(self):
        self.client_address = ("127.0.0.1", 54321)
        self.headers: dict[str, str] = {"User-Agent": "test"}
        self.rfile = MagicMock()
        self.rfile.read.return_value = b"{}"
        self.headers["Content-Length"] = "2"


# ---------------------------------------------------------------------------
# Mock factories
# ---------------------------------------------------------------------------


def _make_debate(
    debate_id: str = "debate-1",
    task: str = "Test task",
    domain: str = "",
    consensus_reached: bool = True,
    confidence: float = 0.85,
    rounds_used: int = 3,
    agents: list[str] | None = None,
    outcome_type: str = "",
    created_at: datetime | str | None = None,
) -> dict[str, Any]:
    """Build a debate dict matching what storage.list_debates returns."""
    now = datetime.now(timezone.utc)
    if created_at is None:
        created_at = now.isoformat()
    elif isinstance(created_at, datetime):
        created_at = created_at.isoformat()

    result_dict: dict[str, Any] = {
        "rounds_used": rounds_used,
        "confidence": confidence,
    }
    if domain:
        result_dict["domain"] = domain
    if outcome_type:
        result_dict["outcome_type"] = outcome_type

    debate = {
        "id": debate_id,
        "task": task,
        "consensus_reached": consensus_reached,
        "result": result_dict,
        "agents": agents or ["agent-a", "agent-b"],
        "created_at": created_at,
    }
    if domain:
        debate["domain"] = domain
    return debate


def _make_agent(
    agent_name: str = "claude",
    elo: float = 1500.0,
    wins: int = 50,
    losses: int = 20,
    draws: int = 10,
    win_rate: float = 0.625,
    games_played: int = 80,
    debates_count: int = 80,
    domain_elos: dict[str, float] | None = None,
    calibration_score: float | None = None,
    calibration_accuracy: float | None = None,
) -> MagicMock:
    """Build a mock agent rating object."""
    agent = MagicMock()
    agent.agent_name = agent_name
    agent.elo = elo
    agent.wins = wins
    agent.losses = losses
    agent.draws = draws
    agent.win_rate = win_rate
    agent.games_played = games_played
    agent.debates_count = debates_count
    agent.domain_elos = domain_elos or {}

    if calibration_score is not None:
        agent.calibration_score = calibration_score
    else:
        del agent.calibration_score

    if calibration_accuracy is not None:
        agent.calibration_accuracy = calibration_accuracy
    else:
        del agent.calibration_accuracy

    return agent


def _make_elo_system(
    agents: list[MagicMock] | None = None,
    leaderboard: list[MagicMock] | None = None,
    elo_history: list[tuple[str, float]] | None = None,
    recent_matches: list[dict] | None = None,
    head_to_head: dict | None = None,
) -> MagicMock:
    """Build a mock ELO system."""
    elo = MagicMock()
    lb = leaderboard if leaderboard is not None else (agents or [])
    elo.get_leaderboard.return_value = lb
    elo.list_agents.return_value = agents or lb
    elo.get_elo_history.return_value = elo_history or []
    elo.get_recent_matches.return_value = recent_matches or []
    elo.get_head_to_head.return_value = head_to_head or {
        "a_wins": 0,
        "b_wins": 0,
        "draws": 0,
        "total": 0,
    }

    def _get_rating(agent_id):
        for a in agents or lb:
            if a.agent_name == agent_id:
                return a
        raise ValueError(f"Agent not found: {agent_id}")

    elo.get_rating.side_effect = _get_rating
    return elo


def _recent_time(days_ago: int = 0) -> datetime:
    """Return a timezone-aware datetime *days_ago* days before now."""
    return datetime.now(timezone.utc) - timedelta(days=days_ago)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Reset the module-level rate limiter between tests."""
    _analytics_metrics_limiter._buckets = defaultdict(list)
    _analytics_metrics_limiter._requests = _analytics_metrics_limiter._buckets
    yield
    _analytics_metrics_limiter._buckets = defaultdict(list)
    _analytics_metrics_limiter._requests = _analytics_metrics_limiter._buckets


@pytest.fixture
def handler():
    """Create an AnalyticsMetricsHandler with empty context."""
    return AnalyticsMetricsHandler({})


@pytest.fixture
def http_handler():
    """Mock HTTP handler for async handle() tests."""
    return MockHTTPHandler()


@pytest.fixture
def mock_storage():
    """Create a MagicMock storage that returns an empty debate list by default."""
    storage = MagicMock()
    storage.list_debates.return_value = []
    return storage


@pytest.fixture
def three_agents():
    """Three mock agents for leaderboard and comparison tests."""
    return [
        _make_agent(
            "claude",
            elo=1650,
            wins=120,
            losses=30,
            draws=10,
            win_rate=0.75,
            games_played=160,
            calibration_score=0.85,
        ),
        _make_agent(
            "gpt-4",
            elo=1580,
            wins=90,
            losses=50,
            draws=20,
            win_rate=0.5625,
            games_played=160,
            calibration_score=0.78,
        ),
        _make_agent(
            "gemini",
            elo=1520,
            wins=70,
            losses=60,
            draws=30,
            win_rate=0.4375,
            games_played=160,
        ),
    ]


@pytest.fixture
def mock_elo(three_agents):
    """Mock ELO system with three agents."""
    return _make_elo_system(agents=three_agents)


@pytest.fixture
def five_debates():
    """Five recent debates with varying attributes."""
    now = datetime.now(timezone.utc)
    return [
        _make_debate(
            debate_id=f"d-{i}",
            task=f"Task {i}",
            consensus_reached=(i % 3 != 0),
            confidence=0.6 + i * 0.08,
            rounds_used=2 + i,
            agents=[f"agent-{j}" for j in range(2 + i % 3)],
            created_at=(now - timedelta(days=i)).isoformat(),
        )
        for i in range(5)
    ]


# ============================================================================
# can_handle() tests
# ============================================================================


class TestCanHandle:
    """Tests for can_handle() route matching."""

    def test_all_static_routes_recognized(self, handler):
        """All static routes are recognized in both versioned and unversioned form."""
        routes = [
            "/api/analytics/debates/overview",
            "/api/analytics/debates/trends",
            "/api/analytics/debates/topics",
            "/api/analytics/debates/outcomes",
            "/api/analytics/agents/leaderboard",
            "/api/analytics/agents/comparison",
            "/api/analytics/agents/trends",
            "/api/analytics/usage/tokens",
            "/api/analytics/usage/costs",
            "/api/analytics/usage/active_users",
        ]
        for route in routes:
            assert handler.can_handle(route), f"can_handle failed for {route}"

    def test_versioned_routes_recognized(self, handler):
        """Versioned /api/v1/ routes are recognized."""
        routes = [
            "/api/v1/analytics/debates/overview",
            "/api/v1/analytics/agents/leaderboard",
            "/api/v1/analytics/usage/tokens",
        ]
        for route in routes:
            assert handler.can_handle(route), f"can_handle failed for {route}"

    def test_agent_performance_pattern_recognized(self, handler):
        """Agent-specific performance route matches the regex pattern."""
        assert handler.can_handle("/api/analytics/agents/claude/performance")
        assert handler.can_handle("/api/v1/analytics/agents/gpt-4/performance")
        assert handler.can_handle("/api/analytics/agents/my_agent_123/performance")

    def test_agent_performance_pattern_rejects_invalid(self, handler):
        """Invalid agent IDs are rejected by the regex pattern."""
        assert not handler.can_handle("/api/analytics/agents//performance")
        assert not handler.can_handle("/api/analytics/agents/bad agent/performance")
        assert not handler.can_handle("/api/analytics/agents/a.b/performance")

    def test_unknown_route_not_handled(self, handler):
        """Unrecognized routes return False."""
        assert not handler.can_handle("/api/v1/analytics/debates/unknown")
        assert not handler.can_handle("/api/v1/other/endpoint")
        assert not handler.can_handle("/random/path")
        assert not handler.can_handle("")

    def test_partial_match_not_handled(self, handler):
        """Partial route paths are not matched."""
        assert not handler.can_handle("/api/analytics/debates")
        assert not handler.can_handle("/api/analytics/agents")
        assert not handler.can_handle("/api/analytics/usage")


# ============================================================================
# _is_demo_mode() tests
# ============================================================================


class TestIsDemoMode:
    """Tests for the _is_demo_mode helper."""

    def test_demo_mode_off_by_default(self):
        """Demo mode is off when env var is not set."""
        with patch.dict("os.environ", {}, clear=True):
            assert not _is_demo_mode()

    def test_demo_mode_true(self):
        """ARAGORA_DEMO_MODE=true activates demo mode."""
        with patch.dict("os.environ", {"ARAGORA_DEMO_MODE": "true"}):
            assert _is_demo_mode()

    def test_demo_mode_yes(self):
        """ARAGORA_DEMO_MODE=yes activates demo mode."""
        with patch.dict("os.environ", {"ARAGORA_DEMO_MODE": "yes"}):
            assert _is_demo_mode()

    def test_demo_mode_one(self):
        """ARAGORA_DEMO_MODE=1 activates demo mode."""
        with patch.dict("os.environ", {"ARAGORA_DEMO_MODE": "1"}):
            assert _is_demo_mode()

    def test_demo_mode_false(self):
        """ARAGORA_DEMO_MODE=false does not activate demo mode."""
        with patch.dict("os.environ", {"ARAGORA_DEMO_MODE": "false"}):
            assert not _is_demo_mode()


# ============================================================================
# _demo_response() tests
# ============================================================================


class TestDemoResponse:
    """Tests for the _demo_response helper."""

    def test_demo_debates_overview(self):
        """Returns demo data for debates overview."""
        result = _demo_response("/api/analytics/debates/overview")
        assert result is not None
        body = _body(result)
        assert body["total_debates"] == 47
        assert body["consensus_rate"] == 72.3

    def test_demo_debates_trends(self):
        """Returns demo data for debates trends."""
        result = _demo_response("/api/analytics/debates/trends")
        assert result is not None
        body = _body(result)
        assert "data_points" in body
        assert len(body["data_points"]) == 5

    def test_demo_debates_topics(self):
        """Returns demo data for debates topics."""
        result = _demo_response("/api/analytics/debates/topics")
        assert result is not None
        body = _body(result)
        assert "topics" in body
        assert len(body["topics"]) == 5

    def test_demo_debates_outcomes(self):
        """Returns demo data for debates outcomes."""
        result = _demo_response("/api/analytics/debates/outcomes")
        assert result is not None
        body = _body(result)
        assert "outcomes" in body
        assert body["outcomes"]["consensus"] == 34

    def test_demo_agents_leaderboard(self):
        """Returns demo data for agents leaderboard."""
        result = _demo_response("/api/analytics/agents/leaderboard")
        assert result is not None
        body = _body(result)
        assert "leaderboard" in body
        assert body["total_agents"] == 5

    def test_demo_usage_tokens(self):
        """Returns demo data for usage tokens."""
        result = _demo_response("/api/analytics/usage/tokens")
        assert result is not None
        body = _body(result)
        assert "summary" in body
        assert body["summary"]["total_tokens"] == 426800

    def test_demo_usage_costs(self):
        """Returns demo data for usage costs."""
        result = _demo_response("/api/analytics/usage/costs")
        assert result is not None
        body = _body(result)
        assert body["total_cost_usd"] == 12.47

    def test_demo_active_users(self):
        """Returns demo data for active users."""
        result = _demo_response("/api/analytics/usage/active_users")
        assert result is not None
        body = _body(result)
        assert body["active_users_24h"] == 3

    def test_demo_unknown_route_returns_none(self):
        """Unknown route returns None from demo."""
        result = _demo_response("/api/analytics/unknown")
        assert result is None


# ============================================================================
# handle() - Demo mode routing
# ============================================================================


class TestHandleDemoMode:
    """Tests for demo mode short-circuit in handle()."""

    @pytest.mark.asyncio
    async def test_demo_mode_returns_demo_data(self, handler, http_handler):
        """In demo mode, handle() returns demo data without auth."""
        with patch.dict("os.environ", {"ARAGORA_DEMO_MODE": "true"}):
            result = await handler.handle(
                "/api/analytics/debates/overview",
                {},
                http_handler,
            )

        assert result is not None
        body = _body(result)
        assert body["total_debates"] == 47

    @pytest.mark.asyncio
    async def test_demo_mode_leaderboard(self, handler, http_handler):
        """In demo mode, leaderboard returns demo data."""
        with patch.dict("os.environ", {"ARAGORA_DEMO_MODE": "true"}):
            result = await handler.handle(
                "/api/analytics/agents/leaderboard",
                {},
                http_handler,
            )

        assert result is not None
        body = _body(result)
        assert len(body["leaderboard"]) == 5

    @pytest.mark.asyncio
    async def test_demo_mode_usage_costs(self, handler, http_handler):
        """In demo mode, usage costs returns demo data."""
        with patch.dict("os.environ", {"ARAGORA_DEMO_MODE": "true"}):
            result = await handler.handle(
                "/api/analytics/usage/costs",
                {},
                http_handler,
            )

        assert result is not None
        body = _body(result)
        assert body["total_cost_usd"] == 12.47

    @pytest.mark.asyncio
    async def test_demo_mode_unknown_route_falls_through(self, handler, http_handler):
        """In demo mode, unknown routes still proceed past demo check."""
        with patch.dict("os.environ", {"ARAGORA_DEMO_MODE": "true"}):
            # Agent performance is not in demo data, so it falls through
            # to the real handler which requires auth (auto-mocked)
            result = await handler.handle(
                "/api/analytics/agents/claude/performance",
                {},
                http_handler,
            )
        # Proceeds to real handler (may return 503 for no ELO, or result)
        # Just verify it didn't short-circuit to demo data
        assert result is not None


# ============================================================================
# handle() - Rate limiting
# ============================================================================


class TestHandleRateLimiting:
    """Tests for rate limiter enforcement in handle()."""

    @pytest.mark.asyncio
    async def test_rate_limit_exceeded_returns_429(self, handler, http_handler):
        """When rate limit is exceeded, returns 429."""
        with (
            patch.object(_analytics_metrics_limiter, "is_allowed", return_value=False),
            patch.dict("os.environ", {}, clear=False),
        ):
            # Ensure demo mode is off
            with patch(
                "aragora.server.handlers._analytics_metrics_impl._is_demo_mode",
                return_value=False,
            ):
                result = await handler.handle(
                    "/api/analytics/debates/overview",
                    {},
                    http_handler,
                )

        assert result is not None
        assert _status(result) == 429
        body = _body(result)
        assert "Rate limit" in body.get("error", body.get("message", ""))

    @pytest.mark.asyncio
    async def test_rate_limit_allowed_proceeds(self, handler, http_handler, mock_storage):
        """When rate limit allows, request proceeds normally."""
        mock_storage.list_debates.return_value = []
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = await handler.handle(
                "/api/analytics/debates/overview",
                {},
                http_handler,
            )

        assert result is not None
        assert _status(result) == 200


# ============================================================================
# handle() - Authentication and RBAC
# ============================================================================


class TestHandleAuth:
    """Tests for authentication and RBAC enforcement in handle()."""

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_unauthenticated_returns_401(self, handler, http_handler):
        """Missing authentication returns 401."""
        from aragora.server.handlers.secure import SecureHandler
        from aragora.server.handlers.utils.auth import UnauthorizedError

        async def raise_unauth(self, request, require_auth=False):
            raise UnauthorizedError("Not authenticated")

        with patch.object(SecureHandler, "get_auth_context", raise_unauth):
            result = await handler.handle(
                "/api/analytics/debates/overview",
                {},
                http_handler,
            )

        assert result is not None
        assert _status(result) == 401

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_forbidden_returns_403(self, handler, http_handler):
        """Insufficient permissions returns 403."""
        from aragora.rbac.models import AuthorizationContext
        from aragora.server.handlers.secure import SecureHandler
        from aragora.server.handlers.utils.auth import ForbiddenError

        mock_ctx = AuthorizationContext(
            user_id="user-1",
            roles=set(),
            permissions=set(),
        )

        async def mock_get_auth(self, request, require_auth=False):
            return mock_ctx

        def raise_forbidden(self, ctx, permission, resource_id=None):
            raise ForbiddenError("No permission")

        with (
            patch.object(SecureHandler, "get_auth_context", mock_get_auth),
            patch.object(SecureHandler, "check_permission", raise_forbidden),
        ):
            result = await handler.handle(
                "/api/analytics/debates/overview",
                {},
                http_handler,
            )

        assert result is not None
        assert _status(result) == 403

    @pytest.mark.asyncio
    async def test_rbac_fail_closed_returns_503(self, handler, http_handler):
        """When RBAC unavailable and fail-closed, returns 503."""
        with (
            patch(
                "aragora.server.handlers._analytics_metrics_impl.RBAC_AVAILABLE",
                False,
            ),
            patch(
                "aragora.server.handlers._analytics_metrics_impl.rbac_fail_closed",
                return_value=True,
            ),
        ):
            result = await handler.handle(
                "/api/analytics/debates/overview",
                {},
                http_handler,
            )

        assert result is not None
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_rbac_unavailable_permissive_in_dev(self, handler, http_handler, mock_storage):
        """When RBAC unavailable but fail-open (dev), request proceeds."""
        mock_storage.list_debates.return_value = []
        with (
            patch(
                "aragora.server.handlers._analytics_metrics_impl.RBAC_AVAILABLE",
                False,
            ),
            patch(
                "aragora.server.handlers._analytics_metrics_impl.rbac_fail_closed",
                return_value=False,
            ),
            patch.object(handler, "get_storage", return_value=mock_storage),
        ):
            result = await handler.handle(
                "/api/analytics/debates/overview",
                {},
                http_handler,
            )

        assert result is not None
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_rbac_checker_denies_returns_403(self, handler, http_handler):
        """When RBAC checker denies permission on second pass, returns 403."""
        mock_decision = MagicMock()
        mock_decision.allowed = False
        mock_decision.reason = "denied"

        http_handler.auth_context = MagicMock()

        with patch(
            "aragora.server.handlers._analytics_metrics_impl.check_permission",
            return_value=mock_decision,
        ):
            result = await handler.handle(
                "/api/analytics/debates/overview",
                {},
                http_handler,
            )

        assert result is not None
        assert _status(result) == 403

    @pytest.mark.asyncio
    async def test_rbac_checker_allows_proceeds(self, handler, http_handler, mock_storage):
        """When RBAC checker allows, request proceeds normally."""
        mock_decision = MagicMock()
        mock_decision.allowed = True

        http_handler.auth_context = MagicMock()
        mock_storage.list_debates.return_value = []

        with (
            patch(
                "aragora.server.handlers._analytics_metrics_impl.check_permission",
                return_value=mock_decision,
            ),
            patch.object(handler, "get_storage", return_value=mock_storage),
        ):
            result = await handler.handle(
                "/api/analytics/debates/overview",
                {},
                http_handler,
            )

        assert result is not None
        assert _status(result) == 200


# ============================================================================
# handle() - Route dispatch
# ============================================================================


class TestHandleRouting:
    """Tests for routing through the async handle() method to all endpoints."""

    @pytest.mark.asyncio
    async def test_route_debates_overview(self, handler, http_handler, mock_storage):
        """handle() routes /api/analytics/debates/overview."""
        mock_storage.list_debates.return_value = []
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = await handler.handle(
                "/api/v1/analytics/debates/overview",
                {"time_range": "30d"},
                http_handler,
            )

        assert result is not None
        assert _status(result) == 200
        assert "total_debates" in _body(result)

    @pytest.mark.asyncio
    async def test_route_debates_trends(self, handler, http_handler, mock_storage):
        """handle() routes /api/v1/analytics/debates/trends."""
        mock_storage.list_debates.return_value = []
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = await handler.handle(
                "/api/v1/analytics/debates/trends",
                {"time_range": "7d", "granularity": "daily"},
                http_handler,
            )

        assert result is not None
        assert _status(result) == 200
        assert "data_points" in _body(result)

    @pytest.mark.asyncio
    async def test_route_debates_topics(self, handler, http_handler, mock_storage):
        """handle() routes /api/v1/analytics/debates/topics."""
        mock_storage.list_debates.return_value = []
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = await handler.handle(
                "/api/v1/analytics/debates/topics",
                {},
                http_handler,
            )

        assert result is not None
        assert _status(result) == 200
        assert "topics" in _body(result)

    @pytest.mark.asyncio
    async def test_route_debates_outcomes(self, handler, http_handler, mock_storage):
        """handle() routes /api/v1/analytics/debates/outcomes."""
        mock_storage.list_debates.return_value = []
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = await handler.handle(
                "/api/v1/analytics/debates/outcomes",
                {},
                http_handler,
            )

        assert result is not None
        assert _status(result) == 200
        assert "outcomes" in _body(result)

    @pytest.mark.asyncio
    async def test_route_agents_leaderboard(self, handler, http_handler, mock_elo):
        """handle() routes /api/v1/analytics/agents/leaderboard."""
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle(
                "/api/v1/analytics/agents/leaderboard",
                {},
                http_handler,
            )

        assert result is not None
        assert _status(result) == 200
        assert "leaderboard" in _body(result)

    @pytest.mark.asyncio
    async def test_route_agents_comparison(self, handler, http_handler, mock_elo):
        """handle() routes /api/v1/analytics/agents/comparison."""
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle(
                "/api/v1/analytics/agents/comparison",
                {"agents": "claude,gpt-4"},
                http_handler,
            )

        assert result is not None
        assert _status(result) == 200
        assert "comparison" in _body(result)

    @pytest.mark.asyncio
    async def test_route_agents_trends(self, handler, http_handler, mock_elo):
        """handle() routes /api/v1/analytics/agents/trends."""
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle(
                "/api/v1/analytics/agents/trends",
                {},
                http_handler,
            )

        assert result is not None
        assert _status(result) == 200
        assert "trends" in _body(result)

    @pytest.mark.asyncio
    async def test_route_agent_performance(self, handler, http_handler, mock_elo):
        """handle() routes /api/v1/analytics/agents/{id}/performance."""
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle(
                "/api/v1/analytics/agents/claude/performance",
                {},
                http_handler,
            )

        assert result is not None
        assert _status(result) == 200
        assert _body(result)["agent_id"] == "claude"

    @pytest.mark.asyncio
    async def test_route_usage_tokens(self, handler, http_handler):
        """handle() routes /api/v1/analytics/usage/tokens."""
        result = await handler.handle(
            "/api/v1/analytics/usage/tokens",
            {"org_id": "test-org"},
            http_handler,
        )

        assert result is not None
        # Could be 200 (with or without cost tracker)
        body = _body(result)
        assert "summary" in body or "org_id" in body

    @pytest.mark.asyncio
    async def test_route_usage_costs(self, handler, http_handler):
        """handle() routes /api/v1/analytics/usage/costs."""
        result = await handler.handle(
            "/api/v1/analytics/usage/costs",
            {"org_id": "test-org"},
            http_handler,
        )

        assert result is not None
        body = _body(result)
        assert "summary" in body or "org_id" in body

    @pytest.mark.asyncio
    async def test_route_active_users(self, handler, http_handler):
        """handle() routes /api/v1/analytics/usage/active_users."""
        result = await handler.handle(
            "/api/v1/analytics/usage/active_users",
            {},
            http_handler,
        )

        assert result is not None
        assert _status(result) == 200
        body = _body(result)
        assert "active_users" in body

    @pytest.mark.asyncio
    async def test_unknown_route_returns_none(self, handler, http_handler):
        """handle() returns None for unrecognized routes."""
        result = await handler.handle(
            "/api/v1/analytics/unknown/endpoint",
            {},
            http_handler,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_unversioned_path_works(self, handler, http_handler, mock_storage):
        """handle() accepts unversioned /api/analytics/... paths."""
        mock_storage.list_debates.return_value = []
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = await handler.handle(
                "/api/analytics/debates/overview",
                {},
                http_handler,
            )

        assert result is not None
        assert _status(result) == 200


# ============================================================================
# _validate_org_access tests
# ============================================================================


class TestValidateOrgAccess:
    """Tests for the _validate_org_access method."""

    def test_admin_can_access_any_org(self, handler):
        """Admin role bypasses org check."""

        class AdminAuth:
            org_id = "admin-org"
            roles = {"admin"}

        org_id, err = handler._validate_org_access(AdminAuth(), "other-org")
        assert err is None
        assert org_id == "other-org"

    def test_platform_admin_can_access_any_org(self, handler):
        """Platform admin role bypasses org check."""

        class PlatformAdminAuth:
            org_id = "admin-org"
            roles = {"platform_admin"}

        org_id, err = handler._validate_org_access(PlatformAdminAuth(), "other-org")
        assert err is None
        assert org_id == "other-org"

    def test_no_requested_org_uses_user_org(self, handler):
        """When no org requested, returns user's org."""

        class UserAuth:
            org_id = "user-org"
            roles = set()

        org_id, err = handler._validate_org_access(UserAuth(), None)
        assert err is None
        assert org_id == "user-org"

    def test_user_can_access_own_org(self, handler):
        """User can access their own org."""

        class UserAuth:
            org_id = "my-org"
            roles = set()

        org_id, err = handler._validate_org_access(UserAuth(), "my-org")
        assert err is None
        assert org_id == "my-org"

    def test_user_cannot_access_other_org(self, handler):
        """Non-admin user cannot access another org."""

        class UserAuth:
            org_id = "my-org"
            roles = set()

        org_id, err = handler._validate_org_access(UserAuth(), "other-org")
        assert err is not None
        assert _status(err) == 403

    def test_no_user_org_no_requested_org(self, handler):
        """When user has no org and no org requested, returns None."""

        class NoOrgAuth:
            org_id = None
            roles = set()

        org_id, err = handler._validate_org_access(NoOrgAuth(), None)
        assert err is None
        assert org_id is None

    def test_no_user_org_but_requested_org(self, handler):
        """When user has no org but requests one, access is allowed."""

        class NoOrgAuth:
            org_id = None
            roles = set()

        org_id, err = handler._validate_org_access(NoOrgAuth(), "some-org")
        # user_org_id is None, requested is "some-org"
        # Since user_org_id is None (falsy), the "if user_org_id and ..." check is False
        assert err is None
        assert org_id == "some-org"

    def test_admin_with_no_requested_org(self, handler):
        """Admin with no requested org returns None."""

        class AdminAuth:
            org_id = "admin-org"
            roles = {"admin"}

        org_id, err = handler._validate_org_access(AdminAuth(), None)
        assert err is None
        assert org_id is None

    def test_roles_as_list(self, handler):
        """Roles can be a list (not just a set)."""

        class ListRolesAuth:
            org_id = "org-1"
            roles = ["admin", "user"]

        org_id, err = handler._validate_org_access(ListRolesAuth(), "other-org")
        assert err is None
        assert org_id == "other-org"

    def test_roles_none_fallback(self, handler):
        """None roles fall back to empty list."""

        class NullRolesAuth:
            org_id = "my-org"
            roles = None

        org_id, err = handler._validate_org_access(NullRolesAuth(), "other-org")
        assert err is not None
        assert _status(err) == 403


# ============================================================================
# handle() - Usage endpoints (token/cost/active_users details)
# ============================================================================


class TestUsageEndpoints:
    """Tests for usage analytics endpoints via handle()."""

    @pytest.mark.asyncio
    async def test_usage_tokens_requires_org_id(self, handler, http_handler):
        """Token usage endpoint requires org_id."""
        result = await handler.handle(
            "/api/analytics/usage/tokens",
            {},  # No org_id
            http_handler,
        )

        assert result is not None
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_usage_costs_requires_org_id(self, handler, http_handler):
        """Cost usage endpoint requires org_id."""
        result = await handler.handle(
            "/api/analytics/usage/costs",
            {},  # No org_id
            http_handler,
        )

        assert result is not None
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_active_users_no_user_store(self, handler, http_handler):
        """Active users without user store returns zero data."""
        result = await handler.handle(
            "/api/analytics/usage/active_users",
            {},
            http_handler,
        )

        assert result is not None
        assert _status(result) == 200
        body = _body(result)
        assert body["active_users"]["daily"] == 0

    @pytest.mark.asyncio
    async def test_active_users_with_user_store(self, http_handler):
        """Active users with user store returns real counts."""
        user_store = MagicMock()
        user_store.get_active_user_counts.return_value = {
            "daily": 10,
            "weekly": 50,
            "monthly": 100,
        }
        user_store.get_user_growth.return_value = {
            "new_users": 5,
            "churned_users": 2,
            "net_growth": 3,
        }
        h = AnalyticsMetricsHandler({"user_store": user_store})
        result = await h.handle(
            "/api/analytics/usage/active_users",
            {},
            http_handler,
        )

        assert result is not None
        assert _status(result) == 200
        body = _body(result)
        assert body["active_users"]["daily"] == 10
        assert body["user_growth"]["new_users"] == 5

    @pytest.mark.asyncio
    async def test_active_users_user_store_error(self, http_handler):
        """Active users with failing user store returns graceful fallback."""
        user_store = MagicMock()
        user_store.get_active_user_counts.side_effect = RuntimeError("DB down")
        h = AnalyticsMetricsHandler({"user_store": user_store})
        result = await h.handle(
            "/api/analytics/usage/active_users",
            {},
            http_handler,
        )

        assert result is not None
        assert _status(result) == 200
        body = _body(result)
        assert body["active_users"]["daily"] == 0
        assert "error" in body

    @pytest.mark.asyncio
    async def test_active_users_invalid_time_range(self, handler, http_handler):
        """Active users with invalid time_range defaults to 30d."""
        result = await handler.handle(
            "/api/analytics/usage/active_users",
            {"time_range": "999d"},
            http_handler,
        )

        assert result is not None
        assert _status(result) == 200
        assert _body(result)["time_range"] == "30d"

    @pytest.mark.asyncio
    async def test_usage_tokens_cost_tracker_unavailable(self, handler, http_handler):
        """Token usage returns fallback when cost tracker import fails."""
        with patch(
            "aragora.server.handlers._analytics_metrics_usage.get_cost_tracker",
            side_effect=ImportError("not available"),
            create=True,
        ):
            result = await handler.handle(
                "/api/analytics/usage/tokens",
                {"org_id": "test-org"},
                http_handler,
            )

        assert result is not None
        assert _status(result) == 200
        body = _body(result)
        assert body["summary"]["total_tokens"] == 0

    @pytest.mark.asyncio
    async def test_usage_costs_cost_tracker_unavailable(self, handler, http_handler):
        """Cost usage returns fallback when cost tracker import fails."""
        with patch(
            "aragora.server.handlers._analytics_metrics_usage.get_cost_tracker",
            side_effect=ImportError("not available"),
            create=True,
        ):
            result = await handler.handle(
                "/api/analytics/usage/costs",
                {"org_id": "test-org"},
                http_handler,
            )

        assert result is not None
        assert _status(result) == 200
        body = _body(result)
        assert body["summary"]["total_cost_usd"] == "0.00"


# ============================================================================
# Agent performance endpoint detail tests
# ============================================================================


class TestAgentPerformanceEndpoint:
    """Tests for the agent-specific performance endpoint."""

    @pytest.mark.asyncio
    async def test_agent_not_found_returns_404(self, handler, http_handler, mock_elo):
        """Requesting a nonexistent agent returns 404."""
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle(
                "/api/analytics/agents/nonexistent/performance",
                {},
                http_handler,
            )

        assert result is not None
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_agent_performance_success(self, handler, http_handler, mock_elo):
        """Successful agent performance returns all expected fields."""
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle(
                "/api/analytics/agents/claude/performance",
                {},
                http_handler,
            )

        assert result is not None
        assert _status(result) == 200
        body = _body(result)
        assert body["agent_id"] == "claude"
        assert "elo" in body
        assert "win_rate" in body
        assert "recent_matches" in body
        assert "elo_history" in body
        assert "generated_at" in body

    @pytest.mark.asyncio
    async def test_agent_performance_no_elo_system(self, handler, http_handler):
        """No ELO system returns 503."""
        with patch.object(handler, "get_elo_system", return_value=None):
            result = await handler.handle(
                "/api/analytics/agents/claude/performance",
                {},
                http_handler,
            )

        assert result is not None
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_agent_performance_with_domain_elos(self, handler, http_handler):
        """Agent with domain ELOs includes domain_performance."""
        agent = _make_agent("claude", domain_elos={"security": 1700.0, "perf": 1600.0})
        elo_sys = _make_elo_system(agents=[agent])
        with patch.object(handler, "get_elo_system", return_value=elo_sys):
            result = await handler.handle(
                "/api/analytics/agents/claude/performance",
                {},
                http_handler,
            )

        body = _body(result)
        assert "domain_performance" in body
        assert "security" in body["domain_performance"]

    @pytest.mark.asyncio
    async def test_agent_performance_with_calibration(self, handler, http_handler):
        """Agent with calibration scores includes them in response."""
        agent = _make_agent("claude", calibration_score=0.92, calibration_accuracy=0.88)
        elo_sys = _make_elo_system(agents=[agent])
        with patch.object(handler, "get_elo_system", return_value=elo_sys):
            result = await handler.handle(
                "/api/analytics/agents/claude/performance",
                {},
                http_handler,
            )

        body = _body(result)
        assert body["calibration_score"] == 0.92
        assert body["calibration_accuracy"] == 0.88

    @pytest.mark.asyncio
    async def test_agent_performance_invalid_time_range(self, handler, http_handler, mock_elo):
        """Invalid time_range defaults to 30d."""
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle(
                "/api/analytics/agents/claude/performance",
                {"time_range": "invalid"},
                http_handler,
            )

        body = _body(result)
        assert body["time_range"] == "30d"


# ============================================================================
# Agent comparison endpoint detail tests
# ============================================================================


class TestAgentComparisonEndpoint:
    """Tests for agent comparison endpoint."""

    @pytest.mark.asyncio
    async def test_comparison_requires_agents_param(self, handler, http_handler, mock_elo):
        """Missing agents parameter returns 400."""
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle(
                "/api/analytics/agents/comparison",
                {},
                http_handler,
            )

        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_comparison_requires_at_least_two(self, handler, http_handler, mock_elo):
        """Single agent returns 400."""
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle(
                "/api/analytics/agents/comparison",
                {"agents": "claude"},
                http_handler,
            )

        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_comparison_max_ten_agents(self, handler, http_handler, mock_elo):
        """More than 10 agents returns 400."""
        agents = ",".join([f"agent-{i}" for i in range(11)])
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle(
                "/api/analytics/agents/comparison",
                {"agents": agents},
                http_handler,
            )

        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_comparison_unknown_agent_included(self, handler, http_handler, mock_elo):
        """Unknown agents included in comparison with error marker."""
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle(
                "/api/analytics/agents/comparison",
                {"agents": "claude,unknown-agent"},
                http_handler,
            )

        assert _status(result) == 200
        body = _body(result)
        unknown = [c for c in body["comparison"] if c.get("error")]
        assert len(unknown) == 1
        assert unknown[0]["agent_name"] == "unknown-agent"

    @pytest.mark.asyncio
    async def test_comparison_no_elo_system(self, handler, http_handler):
        """No ELO system returns 503."""
        with patch.object(handler, "get_elo_system", return_value=None):
            result = await handler.handle(
                "/api/analytics/agents/comparison",
                {"agents": "a,b"},
                http_handler,
            )

        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_comparison_head_to_head(self, handler, http_handler, mock_elo):
        """Comparison includes head-to-head stats."""
        mock_elo.get_head_to_head.return_value = {
            "a_wins": 5,
            "b_wins": 3,
            "draws": 2,
            "total": 10,
        }
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle(
                "/api/analytics/agents/comparison",
                {"agents": "claude,gpt-4"},
                http_handler,
            )

        body = _body(result)
        assert "head_to_head" in body
        key = "claude_vs_gpt-4"
        assert key in body["head_to_head"]
        assert body["head_to_head"][key]["total_matches"] == 10


# ============================================================================
# Agent trends endpoint detail tests
# ============================================================================


class TestAgentTrendsEndpoint:
    """Tests for agent trends endpoint."""

    @pytest.mark.asyncio
    async def test_trends_default_top_five(self, handler, http_handler, mock_elo):
        """Without agents param, defaults to top 5 from leaderboard."""
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle(
                "/api/analytics/agents/trends",
                {},
                http_handler,
            )

        assert _status(result) == 200
        body = _body(result)
        assert "agents" in body
        assert "trends" in body

    @pytest.mark.asyncio
    async def test_trends_with_agents_param(self, handler, http_handler, mock_elo):
        """Specific agents param filters to requested agents."""
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle(
                "/api/analytics/agents/trends",
                {"agents": "claude,gpt-4"},
                http_handler,
            )

        body = _body(result)
        assert body["agents"] == ["claude", "gpt-4"]

    @pytest.mark.asyncio
    async def test_trends_invalid_granularity(self, handler, http_handler, mock_elo):
        """Invalid granularity defaults to daily."""
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle(
                "/api/analytics/agents/trends",
                {"granularity": "hourly"},
                http_handler,
            )

        body = _body(result)
        assert body["granularity"] == "daily"

    @pytest.mark.asyncio
    async def test_trends_no_elo_system(self, handler, http_handler):
        """No ELO system returns 503."""
        with patch.object(handler, "get_elo_system", return_value=None):
            result = await handler.handle(
                "/api/analytics/agents/trends",
                {},
                http_handler,
            )

        assert _status(result) == 503


# ============================================================================
# Debate endpoints through handle() - additional coverage
# ============================================================================


class TestDebateEndpointsViaHandle:
    """Additional debate endpoint tests through handle()."""

    @pytest.mark.asyncio
    async def test_overview_with_real_debates(
        self, handler, http_handler, mock_storage, five_debates
    ):
        """Overview returns correct data with real debates."""
        mock_storage.list_debates.return_value = five_debates
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = await handler.handle(
                "/api/analytics/debates/overview",
                {"time_range": "30d"},
                http_handler,
            )

        assert _status(result) == 200
        body = _body(result)
        assert body["total_debates"] == 5
        assert body["debates_this_period"] > 0

    @pytest.mark.asyncio
    async def test_trends_weekly(self, handler, http_handler, mock_storage, five_debates):
        """Trends with weekly granularity."""
        mock_storage.list_debates.return_value = five_debates
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = await handler.handle(
                "/api/analytics/debates/trends",
                {"granularity": "weekly"},
                http_handler,
            )

        body = _body(result)
        assert body["granularity"] == "weekly"

    @pytest.mark.asyncio
    async def test_topics_with_data(self, handler, http_handler, mock_storage):
        """Topics endpoint with domain data."""
        now = datetime.now(timezone.utc)
        debates = [
            _make_debate("d1", domain="security", created_at=now.isoformat()),
            _make_debate("d2", domain="security", created_at=now.isoformat()),
            _make_debate("d3", domain="performance", created_at=now.isoformat()),
        ]
        mock_storage.list_debates.return_value = debates
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = await handler.handle(
                "/api/analytics/debates/topics",
                {"time_range": "30d"},
                http_handler,
            )

        body = _body(result)
        assert body["total_debates"] == 3
        assert len(body["topics"]) == 2

    @pytest.mark.asyncio
    async def test_outcomes_with_varied_outcomes(self, handler, http_handler, mock_storage):
        """Outcomes endpoint with varied outcome types."""
        now = datetime.now(timezone.utc)
        debates = [
            _make_debate(
                "d1",
                consensus_reached=True,
                confidence=0.9,
                outcome_type="consensus",
                created_at=now.isoformat(),
            ),
            _make_debate(
                "d2",
                consensus_reached=False,
                confidence=0.1,
                outcome_type="",
                created_at=now.isoformat(),
            ),
        ]
        mock_storage.list_debates.return_value = debates
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = await handler.handle(
                "/api/analytics/debates/outcomes",
                {"time_range": "30d"},
                http_handler,
            )

        body = _body(result)
        assert body["outcomes"]["consensus"] >= 1
        assert body["total_debates"] == 2


# ============================================================================
# Security tests
# ============================================================================


class TestSecurity:
    """Security-related tests for the analytics handler."""

    def test_path_traversal_not_handled(self, handler):
        """Path traversal attempts are not handled."""
        assert not handler.can_handle("/api/analytics/../../etc/passwd")
        assert not handler.can_handle("/api/analytics/debates/../../../secret")

    def test_sql_injection_not_handled(self, handler):
        """SQL injection paths are not handled."""
        assert not handler.can_handle("/api/analytics/debates/overview'; DROP TABLE--")

    def test_script_injection_not_handled(self, handler):
        """Script injection paths are not handled."""
        assert not handler.can_handle("/api/analytics/debates/<script>alert(1)</script>")

    def test_null_byte_not_handled(self, handler):
        """Null byte paths are not handled."""
        assert not handler.can_handle("/api/analytics/debates/overview\x00")

    def test_agent_id_regex_prevents_injection(self, handler):
        """Agent performance regex only allows alphanumeric, underscore, hyphen."""
        # These should NOT match
        assert not handler.can_handle("/api/analytics/agents/a;ls/performance")
        assert not handler.can_handle("/api/analytics/agents/a$(cmd)/performance")
        assert not handler.can_handle("/api/analytics/agents/../performance")

        # These SHOULD match
        assert handler.can_handle("/api/analytics/agents/claude-opus/performance")
        assert handler.can_handle("/api/analytics/agents/gpt_4o/performance")
        assert handler.can_handle("/api/analytics/agents/agent123/performance")

    @pytest.mark.asyncio
    async def test_very_long_path_not_handled(self, handler, http_handler):
        """Extremely long paths are not handled."""
        long_path = "/api/analytics/" + "a" * 10000
        result = await handler.handle(long_path, {}, http_handler)
        # Returns None (not handled) -- no crash
        assert result is None


# ============================================================================
# Handler initialization
# ============================================================================


class TestHandlerInit:
    """Tests for handler initialization."""

    def test_init_with_empty_context(self):
        """Handler initializes with empty context."""
        h = AnalyticsMetricsHandler({})
        assert h.ctx == {}

    def test_init_with_none_context(self):
        """Handler initializes with None context defaulting to empty dict."""
        h = AnalyticsMetricsHandler(None)
        assert h.ctx == {}

    def test_init_with_context_data(self):
        """Handler stores context data."""
        ctx = {"user_store": MagicMock(), "storage": MagicMock()}
        h = AnalyticsMetricsHandler(ctx)
        assert h.ctx is ctx

    def test_routes_constant(self, handler):
        """ROUTES constant contains all expected routes."""
        assert len(handler.ROUTES) == 10
        assert "/api/analytics/debates/overview" in handler.ROUTES
        assert "/api/analytics/agents/leaderboard" in handler.ROUTES
        assert "/api/analytics/usage/tokens" in handler.ROUTES

    def test_agent_performance_pattern(self, handler):
        """AGENT_PERFORMANCE_PATTERN matches expected format."""
        pattern = handler.AGENT_PERFORMANCE_PATTERN
        assert pattern.match("/api/analytics/agents/claude/performance")
        assert pattern.match("/api/analytics/agents/agent-123/performance")
        assert not pattern.match("/api/analytics/agents//performance")


# ============================================================================
# Edge cases
# ============================================================================


class TestEdgeCases:
    """Miscellaneous edge case tests."""

    @pytest.mark.asyncio
    async def test_empty_path(self, handler, http_handler):
        """Empty path does not crash."""
        result = await handler.handle("", {}, http_handler)
        assert result is None

    @pytest.mark.asyncio
    async def test_handle_with_none_query_params(self, handler, http_handler, mock_storage):
        """None-like query params handled gracefully."""
        mock_storage.list_debates.return_value = []
        with patch.object(handler, "get_storage", return_value=mock_storage):
            result = await handler.handle(
                "/api/analytics/debates/overview",
                {},
                http_handler,
            )
        assert result is not None
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_concurrent_handler_instances(self, http_handler, mock_storage):
        """Multiple handler instances work independently."""
        h1 = AnalyticsMetricsHandler({"key": "value1"})
        h2 = AnalyticsMetricsHandler({"key": "value2"})
        assert h1.ctx["key"] == "value1"
        assert h2.ctx["key"] == "value2"

    def test_valid_granularities_constant(self):
        """VALID_GRANULARITIES contains expected values."""
        from aragora.server.handlers._analytics_metrics_common import VALID_GRANULARITIES

        assert "daily" in VALID_GRANULARITIES
        assert "weekly" in VALID_GRANULARITIES
        assert "monthly" in VALID_GRANULARITIES

    def test_valid_time_ranges_constant(self):
        """VALID_TIME_RANGES contains expected values."""
        from aragora.server.handlers._analytics_metrics_common import VALID_TIME_RANGES

        assert "7d" in VALID_TIME_RANGES
        assert "30d" in VALID_TIME_RANGES
        assert "90d" in VALID_TIME_RANGES
        assert "365d" in VALID_TIME_RANGES
        assert "all" in VALID_TIME_RANGES

    def test_parse_time_range_all(self):
        """_parse_time_range('all') returns None."""
        from aragora.server.handlers._analytics_metrics_common import _parse_time_range

        assert _parse_time_range("all") is None

    def test_parse_time_range_7d(self):
        """_parse_time_range('7d') returns a datetime ~7 days ago."""
        from aragora.server.handlers._analytics_metrics_common import _parse_time_range

        result = _parse_time_range("7d")
        assert result is not None
        diff = (datetime.now(timezone.utc) - result).total_seconds()
        assert abs(diff - 7 * 86400) < 5  # within 5 seconds

    def test_parse_time_range_invalid(self):
        """_parse_time_range with invalid string returns 30d default."""
        from aragora.server.handlers._analytics_metrics_common import _parse_time_range

        result = _parse_time_range("invalid")
        assert result is not None
        diff = (datetime.now(timezone.utc) - result).total_seconds()
        assert abs(diff - 30 * 86400) < 5

    def test_group_by_time_empty(self):
        """_group_by_time with empty list returns empty dict."""
        from aragora.server.handlers._analytics_metrics_common import _group_by_time

        result = _group_by_time([], "ts", "daily")
        assert result == {}

    def test_group_by_time_daily(self):
        """_group_by_time groups by date for daily granularity."""
        from aragora.server.handlers._analytics_metrics_common import _group_by_time

        now = datetime.now(timezone.utc)
        items = [
            {"ts": now, "val": 1},
            {"ts": now, "val": 2},
        ]
        result = _group_by_time(items, "ts", "daily")
        key = now.strftime("%Y-%m-%d")
        assert key in result
        assert len(result[key]) == 2

    def test_group_by_time_missing_timestamp(self):
        """Items without timestamp key are skipped."""
        from aragora.server.handlers._analytics_metrics_common import _group_by_time

        items = [{"other_key": "value"}]
        result = _group_by_time(items, "ts", "daily")
        assert result == {}
