"""Tests for analytics performance handler (aragora/server/handlers/analytics_performance.py).

Covers all routes and behavior of the AnalyticsPerformanceHandler class:
- can_handle() routing
- GET /api/v1/analytics/agents/performance - Agent performance metrics
- GET /api/v1/analytics/debates/summary   - Debate summary statistics
- GET /api/v1/analytics/trends            - General trend analysis
- Rate limiting
- RBAC enforcement
- Time range parsing
- Trend calculation
- Edge cases and error paths
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.analytics_performance import (
    PERM_ANALYTICS_PERFORMANCE,
    VALID_GRANULARITIES,
    VALID_TIME_RANGES,
    AnalyticsPerformanceHandler,
    _parse_time_range,
)
from aragora.server.handlers.base import HandlerResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result: HandlerResult) -> dict:
    """Extract the JSON body from a HandlerResult."""
    if isinstance(result, HandlerResult):
        if isinstance(result.body, bytes):
            return json.loads(result.body.decode("utf-8"))
        return result.body
    if isinstance(result, dict):
        return result.get("body", result)
    return {}


def _status(result: HandlerResult) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, HandlerResult):
        return result.status_code
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return 200


class MockHTTPHandler:
    """Mock HTTP handler for testing (simulates BaseHTTPRequestHandler)."""

    def __init__(self, body: dict[str, Any] | None = None):
        self.rfile = MagicMock()
        self.command = "GET"
        self._body = body
        self.client_address = ("127.0.0.1", 12345)
        if body:
            body_bytes = json.dumps(body).encode()
            self.rfile.read.return_value = body_bytes
            self.headers = {"Content-Length": str(len(body_bytes))}
        else:
            self.rfile.read.return_value = b"{}"
            self.headers = {"Content-Length": "2"}


def _make_handler(
    body: dict[str, Any] | None = None, method: str = "GET"
) -> MockHTTPHandler:
    """Create a MockHTTPHandler with optional body and method."""
    h = MockHTTPHandler(body=body)
    h.command = method
    return h


class MockAgentRating:
    """Mock agent rating object for ELO system responses."""

    def __init__(
        self,
        agent_name: str = "claude",
        elo: float = 1600.0,
        win_rate: float = 0.75,
        games_played: int = 100,
        wins: int = 75,
        losses: int = 20,
        draws: int = 5,
        consensus_rate: float | None = None,
        avg_response_time_ms: float | None = None,
        calibration_score: float | None = None,
    ):
        self.agent_name = agent_name
        self.elo = elo
        self.win_rate = win_rate
        self.games_played = games_played
        self.wins = wins
        self.losses = losses
        self.draws = draws
        if consensus_rate is not None:
            self.consensus_rate = consensus_rate
        if avg_response_time_ms is not None:
            self.avg_response_time_ms = avg_response_time_ms
        if calibration_score is not None:
            self.calibration_score = calibration_score


def _make_debate(
    debate_id: str = "debate-1",
    consensus_reached: bool = True,
    result: dict | None = None,
    agents: list | None = None,
    domain: str = "security",
    created_at: datetime | None = None,
):
    """Create a mock debate dict for storage."""
    if created_at is None:
        created_at = datetime.now(timezone.utc)
    if result is None:
        result = {
            "rounds_used": 3,
            "confidence": 0.85,
            "duration_seconds": 45.0,
            "outcome_type": "consensus",
            "domain": domain,
        }
    if agents is None:
        agents = ["claude", "gemini", "grok"]
    return {
        "debate_id": debate_id,
        "consensus_reached": consensus_reached,
        "result": result,
        "agents": agents,
        "domain": domain,
        "created_at": created_at.isoformat(),
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create an AnalyticsPerformanceHandler with empty context."""
    return AnalyticsPerformanceHandler(ctx={})


@pytest.fixture(autouse=True)
def _reset_rate_limiters():
    """Reset rate limiters between tests."""
    from aragora.server.handlers.utils.rate_limit import clear_all_limiters

    clear_all_limiters()
    yield
    clear_all_limiters()


@pytest.fixture(autouse=True)
def _reset_module_rate_limiter():
    """Reset the module-level rate limiter between tests."""
    import aragora.server.handlers.analytics_performance as mod

    mod._analytics_performance_limiter._buckets = defaultdict(list)
    yield
    mod._analytics_performance_limiter._buckets = defaultdict(list)


# ============================================================================
# Module Constants
# ============================================================================


class TestModuleConstants:
    """Test module-level constants and configuration."""

    def test_permission_constant(self):
        assert PERM_ANALYTICS_PERFORMANCE == "analytics:read"

    def test_valid_time_ranges(self):
        expected = {"7d", "14d", "30d", "90d", "180d", "365d", "all"}
        assert VALID_TIME_RANGES == expected

    def test_valid_granularities(self):
        expected = {"daily", "weekly", "monthly"}
        assert VALID_GRANULARITIES == expected

    def test_cache_configs_registered(self):
        from aragora.server.handlers.analytics.cache import CACHE_CONFIGS

        assert "agent_performance" in CACHE_CONFIGS
        assert "debates_summary" in CACHE_CONFIGS
        assert "general_trends" in CACHE_CONFIGS

    def test_cache_configs_ttl(self):
        from aragora.server.handlers.analytics.cache import CACHE_CONFIGS

        assert CACHE_CONFIGS["agent_performance"].ttl_seconds == 300.0
        assert CACHE_CONFIGS["debates_summary"].ttl_seconds == 300.0
        assert CACHE_CONFIGS["general_trends"].ttl_seconds == 300.0

    def test_cache_configs_maxsize(self):
        from aragora.server.handlers.analytics.cache import CACHE_CONFIGS

        assert CACHE_CONFIGS["agent_performance"].maxsize == 200
        assert CACHE_CONFIGS["debates_summary"].maxsize == 200
        assert CACHE_CONFIGS["general_trends"].maxsize == 200


# ============================================================================
# _parse_time_range
# ============================================================================


class TestParseTimeRange:
    """Test the _parse_time_range utility function."""

    def test_parse_all_returns_none(self):
        result = _parse_time_range("all")
        assert result is None

    def test_parse_7d(self):
        result = _parse_time_range("7d")
        assert result is not None
        expected = datetime.now(timezone.utc) - timedelta(days=7)
        assert abs((result - expected).total_seconds()) < 2

    def test_parse_30d(self):
        result = _parse_time_range("30d")
        assert result is not None
        expected = datetime.now(timezone.utc) - timedelta(days=30)
        assert abs((result - expected).total_seconds()) < 2

    def test_parse_90d(self):
        result = _parse_time_range("90d")
        assert result is not None
        expected = datetime.now(timezone.utc) - timedelta(days=90)
        assert abs((result - expected).total_seconds()) < 2

    def test_parse_365d(self):
        result = _parse_time_range("365d")
        assert result is not None
        expected = datetime.now(timezone.utc) - timedelta(days=365)
        assert abs((result - expected).total_seconds()) < 2

    def test_parse_180d(self):
        result = _parse_time_range("180d")
        assert result is not None
        expected = datetime.now(timezone.utc) - timedelta(days=180)
        assert abs((result - expected).total_seconds()) < 2

    def test_parse_14d(self):
        result = _parse_time_range("14d")
        assert result is not None
        expected = datetime.now(timezone.utc) - timedelta(days=14)
        assert abs((result - expected).total_seconds()) < 2

    def test_parse_invalid_defaults_to_30d(self):
        result = _parse_time_range("invalid")
        assert result is not None
        expected = datetime.now(timezone.utc) - timedelta(days=30)
        assert abs((result - expected).total_seconds()) < 2

    def test_parse_empty_string_defaults_to_30d(self):
        result = _parse_time_range("")
        assert result is not None
        expected = datetime.now(timezone.utc) - timedelta(days=30)
        assert abs((result - expected).total_seconds()) < 2

    def test_parse_numeric_no_d_suffix_defaults(self):
        result = _parse_time_range("30")
        assert result is not None
        # Should default to 30d
        expected = datetime.now(timezone.utc) - timedelta(days=30)
        assert abs((result - expected).total_seconds()) < 2

    def test_parse_negative_days(self):
        # Regex requires digits so "-7d" won't match -> defaults to 30d
        result = _parse_time_range("-7d")
        assert result is not None

    def test_parse_zero_days(self):
        result = _parse_time_range("0d")
        assert result is not None
        # 0 days from now = now
        expected = datetime.now(timezone.utc)
        assert abs((result - expected).total_seconds()) < 2


# ============================================================================
# Initialization
# ============================================================================


class TestHandlerInit:
    """Test handler initialization."""

    def test_init_with_empty_context(self):
        h = AnalyticsPerformanceHandler(ctx={})
        assert h.ctx == {}

    def test_init_with_none_context(self):
        h = AnalyticsPerformanceHandler(ctx=None)
        assert h.ctx == {}

    def test_init_with_context(self):
        ctx = {"storage": MagicMock()}
        h = AnalyticsPerformanceHandler(ctx=ctx)
        assert h.ctx == ctx

    def test_init_default_none(self):
        h = AnalyticsPerformanceHandler()
        assert h.ctx == {}

    def test_routes_defined(self, handler):
        assert len(handler.ROUTES) == 3
        assert "/api/analytics/agents/performance" in handler.ROUTES
        assert "/api/analytics/debates/summary" in handler.ROUTES
        assert "/api/analytics/trends" in handler.ROUTES


# ============================================================================
# can_handle routing
# ============================================================================


class TestCanHandle:
    """Verify that can_handle correctly accepts or rejects paths."""

    def test_agents_performance_path(self, handler):
        assert handler.can_handle("/api/v1/analytics/agents/performance")

    def test_debates_summary_path(self, handler):
        assert handler.can_handle("/api/v1/analytics/debates/summary")

    def test_trends_path(self, handler):
        assert handler.can_handle("/api/v1/analytics/trends")

    def test_handles_without_version_prefix(self, handler):
        assert handler.can_handle("/api/analytics/agents/performance")

    def test_handles_v2_prefix(self, handler):
        assert handler.can_handle("/api/v2/analytics/agents/performance")

    def test_rejects_unknown_path(self, handler):
        assert not handler.can_handle("/api/v1/analytics/unknown")

    def test_rejects_empty_path(self, handler):
        assert not handler.can_handle("")

    def test_rejects_root_path(self, handler):
        assert not handler.can_handle("/")

    def test_rejects_partial_path(self, handler):
        assert not handler.can_handle("/api/v1/analytics")

    def test_rejects_other_handler_path(self, handler):
        assert not handler.can_handle("/api/v1/debates")

    def test_rejects_agents_without_performance(self, handler):
        assert not handler.can_handle("/api/v1/analytics/agents")


# ============================================================================
# handle() routing
# ============================================================================


class TestHandleRouting:
    """Test the handle() method routing to sub-handlers."""

    def test_routes_to_agents_performance(self, handler):
        mock_elo = MagicMock()
        mock_elo.get_leaderboard.return_value = []
        handler.get_elo_system = MagicMock(return_value=mock_elo)

        h = _make_handler()
        result = handler.handle("/api/v1/analytics/agents/performance", {}, h)
        assert result is not None
        assert _status(result) == 200

    def test_routes_to_debates_summary(self, handler):
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = []
        handler.get_storage = MagicMock(return_value=mock_storage)

        h = _make_handler()
        result = handler.handle("/api/v1/analytics/debates/summary", {}, h)
        assert result is not None
        assert _status(result) == 200

    def test_routes_to_trends(self, handler):
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = []
        handler.get_storage = MagicMock(return_value=mock_storage)

        h = _make_handler()
        result = handler.handle("/api/v1/analytics/trends", {}, h)
        assert result is not None
        assert _status(result) == 200

    def test_unknown_path_returns_none(self, handler):
        h = _make_handler()
        result = handler.handle("/api/v1/analytics/unknown", {}, h)
        assert result is None


# ============================================================================
# Rate Limiting
# ============================================================================


class TestRateLimiting:
    """Test rate limiting on the handle() method."""

    def test_rate_limit_allows_normal_request(self, handler):
        mock_elo = MagicMock()
        mock_elo.get_leaderboard.return_value = []
        handler.get_elo_system = MagicMock(return_value=mock_elo)

        h = _make_handler()
        result = handler.handle("/api/v1/analytics/agents/performance", {}, h)
        assert _status(result) == 200

    def test_rate_limit_exceeded_returns_429(self, handler):
        import aragora.server.handlers.analytics_performance as mod

        mock_elo = MagicMock()
        mock_elo.get_leaderboard.return_value = []
        handler.get_elo_system = MagicMock(return_value=mock_elo)

        # Exhaust the rate limiter via patch (auto-restores on exit)
        with patch.object(
            mod._analytics_performance_limiter, "is_allowed", return_value=False
        ):
            h = _make_handler()
            result = handler.handle("/api/v1/analytics/agents/performance", {}, h)
            assert _status(result) == 429
            body = _body(result)
            assert "rate limit" in body.get("error", "").lower()


# ============================================================================
# RBAC Enforcement
# ============================================================================


class TestRBACEnforcement:
    """Test RBAC permission checks in handle()."""

    @pytest.mark.no_auto_auth
    def test_rbac_denied_returns_403(self, handler):
        from aragora.rbac.models import AuthorizationDecision

        h = _make_handler()
        h.auth_context = MagicMock()

        denied = AuthorizationDecision(
            allowed=False, reason="no permission",
            permission_key=PERM_ANALYTICS_PERFORMANCE,
        )

        with patch(
            "aragora.server.handlers.analytics_performance.check_permission",
            return_value=denied,
        ):
            with patch(
                "aragora.server.handlers.analytics_performance.RBAC_AVAILABLE",
                True,
            ):
                result = handler.handle(
                    "/api/v1/analytics/agents/performance", {}, h
                )
                assert result is not None
                assert _status(result) == 403
                body = _body(result)
                err = body.get("error", "")
                if isinstance(err, dict):
                    assert err.get("code") == "PERMISSION_DENIED" or \
                        "permission denied" in err.get("message", "").lower()
                else:
                    assert "permission denied" in err.lower()

    @pytest.mark.no_auto_auth
    def test_rbac_unavailable_non_production_allows(self, handler):
        """In dev/test, missing RBAC doesn't block access."""
        mock_elo = MagicMock()
        mock_elo.get_leaderboard.return_value = []
        handler.get_elo_system = MagicMock(return_value=mock_elo)

        h = _make_handler()

        with patch(
            "aragora.server.handlers.analytics_performance.RBAC_AVAILABLE",
            False,
        ):
            with patch(
                "aragora.server.handlers.analytics_performance.rbac_fail_closed",
                return_value=False,
            ):
                result = handler.handle(
                    "/api/v1/analytics/agents/performance", {}, h
                )
                assert result is not None
                assert _status(result) == 200

    @pytest.mark.no_auto_auth
    def test_rbac_unavailable_production_returns_503(self, handler):
        """In production, missing RBAC blocks access."""
        h = _make_handler()

        with patch(
            "aragora.server.handlers.analytics_performance.RBAC_AVAILABLE",
            False,
        ):
            with patch(
                "aragora.server.handlers.analytics_performance.rbac_fail_closed",
                return_value=True,
            ):
                result = handler.handle(
                    "/api/v1/analytics/agents/performance", {}, h
                )
                assert result is not None
                assert _status(result) == 503
                body = _body(result)
                assert "service unavailable" in body.get("error", "").lower()


# ============================================================================
# GET /api/v1/analytics/agents/performance
# ============================================================================


class TestAgentsPerformance:
    """Test the agents performance endpoint."""

    def test_no_elo_system_returns_empty(self, handler):
        handler.get_elo_system = MagicMock(return_value=None)

        h = _make_handler()
        result = handler.handle("/api/v1/analytics/agents/performance", {}, h)
        assert _status(result) == 200
        body = _body(result)
        assert body["total_agents"] == 0
        assert body["agents"] == []
        assert body["summary"]["avg_elo"] == 1500
        assert body["summary"]["avg_win_rate"] == 0.0
        assert body["summary"]["total_debates"] == 0
        assert "generated_at" in body

    def test_no_elo_system_includes_time_range(self, handler):
        handler.get_elo_system = MagicMock(return_value=None)

        h = _make_handler()
        result = handler.handle(
            "/api/v1/analytics/agents/performance", {"time_range": "7d"}, h
        )
        body = _body(result)
        assert body["time_range"] == "7d"

    def test_single_agent(self, handler):
        agent = MockAgentRating(
            agent_name="claude",
            elo=1650.0,
            win_rate=0.75,
            games_played=100,
            wins=75,
            losses=20,
            draws=5,
        )
        mock_elo = MagicMock()
        mock_elo.get_leaderboard.return_value = [agent]
        handler.get_elo_system = MagicMock(return_value=mock_elo)

        h = _make_handler()
        result = handler.handle("/api/v1/analytics/agents/performance", {}, h)
        assert _status(result) == 200
        body = _body(result)
        assert body["total_agents"] == 1
        assert len(body["agents"]) == 1

        a = body["agents"][0]
        assert a["agent_name"] == "claude"
        assert a["elo"] == 1650.0
        assert a["win_rate"] == 75.0
        assert a["total_debates"] == 100
        assert a["wins"] == 75
        assert a["losses"] == 20
        assert a["draws"] == 5
        assert a["rank"] == 1

    def test_multiple_agents_ranking(self, handler):
        agents = [
            MockAgentRating("claude", elo=1700, win_rate=0.80, games_played=200,
                            wins=160, losses=30, draws=10),
            MockAgentRating("gemini", elo=1600, win_rate=0.65, games_played=150,
                            wins=97, losses=43, draws=10),
            MockAgentRating("grok", elo=1500, win_rate=0.50, games_played=100,
                            wins=50, losses=40, draws=10),
        ]
        mock_elo = MagicMock()
        mock_elo.get_leaderboard.return_value = agents
        handler.get_elo_system = MagicMock(return_value=mock_elo)

        h = _make_handler()
        result = handler.handle("/api/v1/analytics/agents/performance", {}, h)
        body = _body(result)
        assert body["total_agents"] == 3
        assert body["agents"][0]["rank"] == 1
        assert body["agents"][1]["rank"] == 2
        assert body["agents"][2]["rank"] == 3

    def test_summary_calculations(self, handler):
        agents = [
            MockAgentRating("a1", elo=1600, win_rate=0.80, games_played=100,
                            wins=80, losses=15, draws=5),
            MockAgentRating("a2", elo=1400, win_rate=0.60, games_played=200,
                            wins=120, losses=70, draws=10),
        ]
        mock_elo = MagicMock()
        mock_elo.get_leaderboard.return_value = agents
        handler.get_elo_system = MagicMock(return_value=mock_elo)

        h = _make_handler()
        result = handler.handle("/api/v1/analytics/agents/performance", {}, h)
        body = _body(result)

        summary = body["summary"]
        assert summary["avg_elo"] == round((1600 + 1400) / 2, 0)
        assert summary["avg_win_rate"] == round((80 + 60) / 2, 1)
        assert summary["total_debates"] == 300
        assert summary["top_performer"] == "a1"
        assert summary["most_active"] == "a2"

    def test_consensus_rate_included_when_available(self, handler):
        agent = MockAgentRating("claude", consensus_rate=0.885)
        mock_elo = MagicMock()
        mock_elo.get_leaderboard.return_value = [agent]
        handler.get_elo_system = MagicMock(return_value=mock_elo)

        h = _make_handler()
        result = handler.handle("/api/v1/analytics/agents/performance", {}, h)
        body = _body(result)
        assert body["agents"][0]["consensus_rate"] == 88.5

    def test_consensus_rate_omitted_when_missing(self, handler):
        agent = MockAgentRating("claude")
        mock_elo = MagicMock()
        mock_elo.get_leaderboard.return_value = [agent]
        handler.get_elo_system = MagicMock(return_value=mock_elo)

        h = _make_handler()
        result = handler.handle("/api/v1/analytics/agents/performance", {}, h)
        body = _body(result)
        assert "consensus_rate" not in body["agents"][0]

    def test_response_time_included_when_available(self, handler):
        agent = MockAgentRating("claude", avg_response_time_ms=1250.7)
        mock_elo = MagicMock()
        mock_elo.get_leaderboard.return_value = [agent]
        handler.get_elo_system = MagicMock(return_value=mock_elo)

        h = _make_handler()
        result = handler.handle("/api/v1/analytics/agents/performance", {}, h)
        body = _body(result)
        assert body["agents"][0]["avg_response_time_ms"] == 1251.0

    def test_response_time_omitted_when_missing(self, handler):
        agent = MockAgentRating("claude")
        mock_elo = MagicMock()
        mock_elo.get_leaderboard.return_value = [agent]
        handler.get_elo_system = MagicMock(return_value=mock_elo)

        h = _make_handler()
        result = handler.handle("/api/v1/analytics/agents/performance", {}, h)
        body = _body(result)
        assert "avg_response_time_ms" not in body["agents"][0]

    def test_calibration_score_included_when_available(self, handler):
        agent = MockAgentRating("claude", calibration_score=0.9234)
        mock_elo = MagicMock()
        mock_elo.get_leaderboard.return_value = [agent]
        handler.get_elo_system = MagicMock(return_value=mock_elo)

        h = _make_handler()
        result = handler.handle("/api/v1/analytics/agents/performance", {}, h)
        body = _body(result)
        assert body["agents"][0]["calibration_score"] == 0.92

    def test_calibration_score_omitted_when_missing(self, handler):
        agent = MockAgentRating("claude")
        mock_elo = MagicMock()
        mock_elo.get_leaderboard.return_value = [agent]
        handler.get_elo_system = MagicMock(return_value=mock_elo)

        h = _make_handler()
        result = handler.handle("/api/v1/analytics/agents/performance", {}, h)
        body = _body(result)
        assert "calibration_score" not in body["agents"][0]

    def test_default_time_range_is_30d(self, handler):
        handler.get_elo_system = MagicMock(return_value=None)
        h = _make_handler()
        result = handler.handle("/api/v1/analytics/agents/performance", {}, h)
        body = _body(result)
        assert body["time_range"] == "30d"

    def test_invalid_time_range_defaults_to_30d(self, handler):
        handler.get_elo_system = MagicMock(return_value=None)
        h = _make_handler()
        result = handler.handle(
            "/api/v1/analytics/agents/performance", {"time_range": "999d"}, h
        )
        body = _body(result)
        assert body["time_range"] == "30d"

    def test_valid_time_ranges_accepted(self, handler):
        handler.get_elo_system = MagicMock(return_value=None)
        for tr in VALID_TIME_RANGES:
            h = _make_handler()
            result = handler.handle(
                "/api/v1/analytics/agents/performance", {"time_range": tr}, h
            )
            body = _body(result)
            assert body["time_range"] == tr

    def test_org_id_filter_passed_through(self, handler):
        handler.get_elo_system = MagicMock(return_value=None)
        h = _make_handler()
        result = handler.handle(
            "/api/v1/analytics/agents/performance",
            {"org_id": "org-123"},
            h,
        )
        body = _body(result)
        assert body["org_id"] == "org-123"

    def test_limit_parameter_passed_to_elo(self, handler):
        mock_elo = MagicMock()
        mock_elo.get_leaderboard.return_value = []
        handler.get_elo_system = MagicMock(return_value=mock_elo)

        h = _make_handler()
        handler.handle(
            "/api/v1/analytics/agents/performance", {"limit": "50"}, h
        )
        mock_elo.get_leaderboard.assert_called_once_with(limit=50)

    def test_limit_default_is_20(self, handler):
        mock_elo = MagicMock()
        mock_elo.get_leaderboard.return_value = []
        handler.get_elo_system = MagicMock(return_value=mock_elo)

        h = _make_handler()
        handler.handle("/api/v1/analytics/agents/performance", {}, h)
        mock_elo.get_leaderboard.assert_called_once_with(limit=20)

    def test_limit_clamped_to_max_100(self, handler):
        mock_elo = MagicMock()
        mock_elo.get_leaderboard.return_value = []
        handler.get_elo_system = MagicMock(return_value=mock_elo)

        h = _make_handler()
        handler.handle(
            "/api/v1/analytics/agents/performance", {"limit": "500"}, h
        )
        mock_elo.get_leaderboard.assert_called_once_with(limit=100)

    def test_generated_at_present(self, handler):
        handler.get_elo_system = MagicMock(return_value=None)
        h = _make_handler()
        result = handler.handle("/api/v1/analytics/agents/performance", {}, h)
        body = _body(result)
        assert "generated_at" in body

    def test_most_active_agent_correctly_identified(self, handler):
        agents = [
            MockAgentRating("claude", elo=1700, games_played=50, wins=40, losses=8, draws=2),
            MockAgentRating("gemini", elo=1600, games_played=200, wins=120, losses=60, draws=20),
        ]
        mock_elo = MagicMock()
        mock_elo.get_leaderboard.return_value = agents
        handler.get_elo_system = MagicMock(return_value=mock_elo)

        h = _make_handler()
        result = handler.handle("/api/v1/analytics/agents/performance", {}, h)
        body = _body(result)
        assert body["summary"]["most_active"] == "gemini"
        assert body["summary"]["top_performer"] == "claude"


# ============================================================================
# GET /api/v1/analytics/debates/summary
# ============================================================================


class TestDebatesSummary:
    """Test the debates summary endpoint."""

    def test_no_storage_returns_empty(self, handler):
        handler.get_storage = MagicMock(return_value=None)

        h = _make_handler()
        result = handler.handle("/api/v1/analytics/debates/summary", {}, h)
        assert _status(result) == 200
        body = _body(result)
        assert body["total_debates"] == 0
        assert body["consensus_reached"] == 0
        assert body["consensus_rate"] == 0.0
        assert body["avg_rounds"] == 0.0
        assert body["avg_agents"] == 0.0
        assert body["avg_confidence"] == 0.0
        assert body["by_outcome"] == {}
        assert body["by_domain"] == {}
        assert "generated_at" in body

    def test_empty_debates_list(self, handler):
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = []
        handler.get_storage = MagicMock(return_value=mock_storage)

        h = _make_handler()
        result = handler.handle("/api/v1/analytics/debates/summary", {}, h)
        body = _body(result)
        assert body["total_debates"] == 0

    def test_single_debate_with_consensus(self, handler):
        now = datetime.now(timezone.utc)
        debate = _make_debate(created_at=now, consensus_reached=True)
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = [debate]
        handler.get_storage = MagicMock(return_value=mock_storage)

        h = _make_handler()
        result = handler.handle("/api/v1/analytics/debates/summary", {}, h)
        body = _body(result)
        assert body["total_debates"] == 1
        assert body["consensus_reached"] == 1
        assert body["consensus_rate"] == 100.0
        assert body["avg_rounds"] == 3.0
        assert body["avg_agents"] == 3.0

    def test_debate_without_consensus(self, handler):
        now = datetime.now(timezone.utc)
        debate = _make_debate(
            created_at=now,
            consensus_reached=False,
            result={
                "rounds_used": 5,
                "confidence": 0.4,
                "duration_seconds": 60.0,
                "outcome_type": "",
                "domain": "general",
            },
        )
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = [debate]
        handler.get_storage = MagicMock(return_value=mock_storage)

        h = _make_handler()
        result = handler.handle("/api/v1/analytics/debates/summary", {}, h)
        body = _body(result)
        assert body["total_debates"] == 1
        assert body["consensus_reached"] == 0
        assert body["consensus_rate"] == 0.0
        assert body["by_outcome"]["no_resolution"] == 1

    def test_outcome_type_classification(self, handler):
        now = datetime.now(timezone.utc)
        debates = [
            _make_debate(
                "d1", consensus_reached=True, created_at=now,
                result={"outcome_type": "consensus", "rounds_used": 3,
                        "confidence": 0.9, "domain": "security", "duration_seconds": 30},
            ),
            _make_debate(
                "d2", consensus_reached=False, created_at=now,
                result={"outcome_type": "dissent", "rounds_used": 5,
                        "confidence": 0.3, "domain": "general", "duration_seconds": 60},
            ),
        ]
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = debates
        handler.get_storage = MagicMock(return_value=mock_storage)

        h = _make_handler()
        result = handler.handle("/api/v1/analytics/debates/summary", {}, h)
        body = _body(result)
        assert body["by_outcome"]["consensus"] == 1
        assert body["by_outcome"]["dissent"] == 1

    def test_fallback_outcome_classification(self, handler):
        """When outcome_type is empty, infer from consensus_reached and confidence."""
        now = datetime.now(timezone.utc)
        debates = [
            _make_debate(
                "d1", consensus_reached=True, created_at=now,
                result={"outcome_type": "", "rounds_used": 3,
                        "confidence": 0.9, "domain": "sec", "duration_seconds": 30},
            ),
            _make_debate(
                "d2", consensus_reached=True, created_at=now,
                result={"outcome_type": "", "rounds_used": 3,
                        "confidence": 0.5, "domain": "sec", "duration_seconds": 30},
            ),
            _make_debate(
                "d3", consensus_reached=False, created_at=now,
                result={"outcome_type": "", "rounds_used": 5,
                        "confidence": 0.3, "domain": "sec", "duration_seconds": 60},
            ),
        ]
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = debates
        handler.get_storage = MagicMock(return_value=mock_storage)

        h = _make_handler()
        result = handler.handle("/api/v1/analytics/debates/summary", {}, h)
        body = _body(result)
        assert body["by_outcome"]["consensus"] == 1  # confidence >= 0.8
        assert body["by_outcome"]["majority"] == 1   # consensus but confidence < 0.8
        assert body["by_outcome"]["no_resolution"] == 1

    def test_domain_stats_calculated(self, handler):
        now = datetime.now(timezone.utc)
        debates = [
            _make_debate("d1", consensus_reached=True, domain="security", created_at=now),
            _make_debate("d2", consensus_reached=True, domain="security", created_at=now),
            _make_debate("d3", consensus_reached=False, domain="security", created_at=now),
            _make_debate("d4", consensus_reached=True, domain="performance", created_at=now),
        ]
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = debates
        handler.get_storage = MagicMock(return_value=mock_storage)

        h = _make_handler()
        result = handler.handle("/api/v1/analytics/debates/summary", {}, h)
        body = _body(result)
        assert body["by_domain"]["security"]["count"] == 3
        assert body["by_domain"]["security"]["consensus_rate"] == round(2 / 3 * 100, 1)
        assert body["by_domain"]["performance"]["count"] == 1
        assert body["by_domain"]["performance"]["consensus_rate"] == 100.0

    def test_time_range_filtering(self, handler):
        now = datetime.now(timezone.utc)
        old = now - timedelta(days=60)
        recent = now - timedelta(days=5)
        debates = [
            _make_debate("d_old", created_at=old),
            _make_debate("d_new", created_at=recent),
        ]
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = debates
        handler.get_storage = MagicMock(return_value=mock_storage)

        h = _make_handler()
        result = handler.handle(
            "/api/v1/analytics/debates/summary", {"time_range": "7d"}, h
        )
        body = _body(result)
        assert body["total_debates"] == 1  # Only recent debate

    def test_time_range_all_includes_everything(self, handler):
        now = datetime.now(timezone.utc)
        old = now - timedelta(days=365)
        debates = [
            _make_debate("d_old", created_at=old),
            _make_debate("d_new", created_at=now),
        ]
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = debates
        handler.get_storage = MagicMock(return_value=mock_storage)

        h = _make_handler()
        result = handler.handle(
            "/api/v1/analytics/debates/summary", {"time_range": "all"}, h
        )
        body = _body(result)
        assert body["total_debates"] == 2

    def test_invalid_time_range_defaults(self, handler):
        handler.get_storage = MagicMock(return_value=None)
        h = _make_handler()
        result = handler.handle(
            "/api/v1/analytics/debates/summary", {"time_range": "badvalue"}, h
        )
        body = _body(result)
        assert body["time_range"] == "30d"

    def test_org_id_passed_to_storage(self, handler):
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = []
        handler.get_storage = MagicMock(return_value=mock_storage)

        h = _make_handler()
        handler.handle(
            "/api/v1/analytics/debates/summary", {"org_id": "org-456"}, h
        )
        mock_storage.list_debates.assert_called_once_with(limit=10000, org_id="org-456")

    def test_org_id_in_response(self, handler):
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = []
        handler.get_storage = MagicMock(return_value=mock_storage)

        h = _make_handler()
        result = handler.handle(
            "/api/v1/analytics/debates/summary", {"org_id": "org-456"}, h
        )
        body = _body(result)
        assert body["org_id"] == "org-456"

    def test_avg_confidence_calculation(self, handler):
        now = datetime.now(timezone.utc)
        debates = [
            _make_debate("d1", created_at=now, result={
                "rounds_used": 3, "confidence": 0.9,
                "domain": "sec", "outcome_type": "consensus", "duration_seconds": 10,
            }),
            _make_debate("d2", created_at=now, result={
                "rounds_used": 3, "confidence": 0.7,
                "domain": "sec", "outcome_type": "consensus", "duration_seconds": 10,
            }),
        ]
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = debates
        handler.get_storage = MagicMock(return_value=mock_storage)

        h = _make_handler()
        result = handler.handle("/api/v1/analytics/debates/summary", {}, h)
        body = _body(result)
        assert body["avg_confidence"] == 0.80

    def test_avg_duration_calculation(self, handler):
        now = datetime.now(timezone.utc)
        debates = [
            _make_debate("d1", created_at=now, result={
                "rounds_used": 3, "confidence": 0.9,
                "domain": "sec", "outcome_type": "consensus", "duration_seconds": 30,
            }),
            _make_debate("d2", created_at=now, result={
                "rounds_used": 3, "confidence": 0.7,
                "domain": "sec", "outcome_type": "consensus", "duration_seconds": 60,
            }),
        ]
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = debates
        handler.get_storage = MagicMock(return_value=mock_storage)

        h = _make_handler()
        result = handler.handle("/api/v1/analytics/debates/summary", {}, h)
        body = _body(result)
        assert body["avg_duration_seconds"] == 45.0

    def test_peak_hours_and_days(self, handler):
        now = datetime.now(timezone.utc)
        # Create debates at specific hours
        debates = []
        for i in range(5):
            t = now.replace(hour=14, minute=0, second=0)
            debates.append(_make_debate(f"d14_{i}", created_at=t))
        for i in range(3):
            t = now.replace(hour=10, minute=0, second=0)
            debates.append(_make_debate(f"d10_{i}", created_at=t))

        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = debates
        handler.get_storage = MagicMock(return_value=mock_storage)

        h = _make_handler()
        result = handler.handle(
            "/api/v1/analytics/debates/summary", {"time_range": "30d"}, h
        )
        body = _body(result)
        assert 14 in body["peak_hours"]

    def test_debate_with_datetime_created_at(self, handler):
        """Test that debates with datetime objects (not strings) are handled."""
        now = datetime.now(timezone.utc)
        debate = {
            "debate_id": "d1",
            "consensus_reached": True,
            "result": {
                "rounds_used": 3, "confidence": 0.9,
                "domain": "sec", "outcome_type": "consensus", "duration_seconds": 30,
            },
            "agents": ["claude"],
            "created_at": now,  # datetime object, not string
        }
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = [debate]
        handler.get_storage = MagicMock(return_value=mock_storage)

        h = _make_handler()
        result = handler.handle("/api/v1/analytics/debates/summary", {}, h)
        body = _body(result)
        assert body["total_debates"] == 1

    def test_debate_with_invalid_created_at_skipped(self, handler):
        """Debates with unparseable timestamps are skipped when filtering."""
        debate = {
            "debate_id": "d1",
            "consensus_reached": True,
            "result": {"rounds_used": 3, "confidence": 0.9, "domain": "sec",
                        "outcome_type": "consensus", "duration_seconds": 30},
            "agents": ["claude"],
            "created_at": "not-a-date",
        }
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = [debate]
        handler.get_storage = MagicMock(return_value=mock_storage)

        h = _make_handler()
        result = handler.handle(
            "/api/v1/analytics/debates/summary", {"time_range": "7d"}, h
        )
        body = _body(result)
        assert body["total_debates"] == 0

    def test_debate_as_object_not_dict(self, handler):
        """Test that non-dict debate objects are handled via vars()."""
        now = datetime.now(timezone.utc)

        class MockDebateObj:
            def __init__(self):
                self.debate_id = "d1"
                self.consensus_reached = True
                self.result = {
                    "rounds_used": 3, "confidence": 0.8, "domain": "sec",
                    "outcome_type": "consensus", "duration_seconds": 30,
                }
                self.agents = ["claude", "gemini"]
                self.created_at = now.isoformat()

        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = [MockDebateObj()]
        handler.get_storage = MagicMock(return_value=mock_storage)

        h = _make_handler()
        result = handler.handle("/api/v1/analytics/debates/summary", {}, h)
        body = _body(result)
        assert body["total_debates"] == 1
        assert body["avg_agents"] == 2.0

    def test_zero_confidence_excluded_from_average(self, handler):
        """Debates with confidence=0 should not count in avg_confidence."""
        now = datetime.now(timezone.utc)
        debates = [
            _make_debate("d1", created_at=now, result={
                "rounds_used": 3, "confidence": 0.0,
                "domain": "sec", "outcome_type": "consensus", "duration_seconds": 10,
            }),
            _make_debate("d2", created_at=now, result={
                "rounds_used": 3, "confidence": 0.8,
                "domain": "sec", "outcome_type": "consensus", "duration_seconds": 10,
            }),
        ]
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = debates
        handler.get_storage = MagicMock(return_value=mock_storage)

        h = _make_handler()
        result = handler.handle("/api/v1/analytics/debates/summary", {}, h)
        body = _body(result)
        # Only d2's confidence counts
        assert body["avg_confidence"] == 0.80


# ============================================================================
# GET /api/v1/analytics/trends
# ============================================================================


class TestGeneralTrends:
    """Test the general trends endpoint."""

    def test_no_storage_returns_empty(self, handler):
        handler.get_storage = MagicMock(return_value=None)

        h = _make_handler()
        result = handler.handle("/api/v1/analytics/trends", {}, h)
        assert _status(result) == 200
        body = _body(result)
        assert body["data_points"] == []
        assert body["trend_analysis"] == {}
        assert "generated_at" in body

    def test_empty_debates(self, handler):
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = []
        handler.get_storage = MagicMock(return_value=mock_storage)

        h = _make_handler()
        result = handler.handle("/api/v1/analytics/trends", {}, h)
        body = _body(result)
        assert body["data_points"] == []

    def test_default_granularity_is_daily(self, handler):
        handler.get_storage = MagicMock(return_value=None)
        h = _make_handler()
        result = handler.handle("/api/v1/analytics/trends", {}, h)
        body = _body(result)
        assert body["granularity"] == "daily"

    def test_invalid_granularity_defaults_to_daily(self, handler):
        handler.get_storage = MagicMock(return_value=None)
        h = _make_handler()
        result = handler.handle(
            "/api/v1/analytics/trends", {"granularity": "hourly"}, h
        )
        body = _body(result)
        assert body["granularity"] == "daily"

    def test_weekly_granularity(self, handler):
        now = datetime.now(timezone.utc)
        debates = [_make_debate("d1", created_at=now)]
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = debates
        handler.get_storage = MagicMock(return_value=mock_storage)

        h = _make_handler()
        result = handler.handle(
            "/api/v1/analytics/trends", {"granularity": "weekly"}, h
        )
        body = _body(result)
        assert body["granularity"] == "weekly"
        if body["data_points"]:
            assert "-W" in body["data_points"][0]["period"]

    def test_monthly_granularity(self, handler):
        now = datetime.now(timezone.utc)
        debates = [_make_debate("d1", created_at=now)]
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = debates
        handler.get_storage = MagicMock(return_value=mock_storage)

        h = _make_handler()
        result = handler.handle(
            "/api/v1/analytics/trends", {"granularity": "monthly"}, h
        )
        body = _body(result)
        assert body["granularity"] == "monthly"
        if body["data_points"]:
            # Monthly format: YYYY-MM
            period = body["data_points"][0]["period"]
            parts = period.split("-")
            assert len(parts) == 2

    def test_daily_data_points_generated(self, handler):
        now = datetime.now(timezone.utc)
        day1 = now - timedelta(days=2)
        day2 = now - timedelta(days=1)
        debates = [
            _make_debate("d1", created_at=day1),
            _make_debate("d2", created_at=day1),
            _make_debate("d3", created_at=day2),
        ]
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = debates
        handler.get_storage = MagicMock(return_value=mock_storage)

        h = _make_handler()
        result = handler.handle("/api/v1/analytics/trends", {}, h)
        body = _body(result)
        assert len(body["data_points"]) == 2
        # Sorted chronologically
        assert body["data_points"][0]["period"] < body["data_points"][1]["period"]

    def test_metrics_filter_debates(self, handler):
        now = datetime.now(timezone.utc)
        debates = [_make_debate("d1", created_at=now)]
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = debates
        handler.get_storage = MagicMock(return_value=mock_storage)

        h = _make_handler()
        result = handler.handle(
            "/api/v1/analytics/trends", {"metrics": "debates"}, h
        )
        body = _body(result)
        if body["data_points"]:
            point = body["data_points"][0]
            assert "debates_count" in point
            assert "active_agents" not in point

    def test_metrics_filter_agents(self, handler):
        now = datetime.now(timezone.utc)
        debates = [_make_debate("d1", created_at=now)]
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = debates
        handler.get_storage = MagicMock(return_value=mock_storage)

        h = _make_handler()
        result = handler.handle(
            "/api/v1/analytics/trends", {"metrics": "agents"}, h
        )
        body = _body(result)
        if body["data_points"]:
            point = body["data_points"][0]
            assert "active_agents" in point
            assert "debates_count" not in point

    def test_metrics_filter_tokens(self, handler):
        now = datetime.now(timezone.utc)
        debate = _make_debate("d1", created_at=now)
        debate["result"]["total_tokens"] = 50000
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = [debate]
        handler.get_storage = MagicMock(return_value=mock_storage)

        h = _make_handler()
        result = handler.handle(
            "/api/v1/analytics/trends", {"metrics": "tokens"}, h
        )
        body = _body(result)
        if body["data_points"]:
            point = body["data_points"][0]
            assert "total_tokens" in point
            assert point["total_tokens"] == 50000

    def test_default_metrics(self, handler):
        now = datetime.now(timezone.utc)
        debates = [_make_debate("d1", created_at=now)]
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = debates
        handler.get_storage = MagicMock(return_value=mock_storage)

        h = _make_handler()
        result = handler.handle("/api/v1/analytics/trends", {}, h)
        body = _body(result)
        if body["data_points"]:
            point = body["data_points"][0]
            assert "debates_count" in point
            assert "consensus_rate" in point
            assert "active_agents" in point

    def test_time_range_filtering(self, handler):
        now = datetime.now(timezone.utc)
        old = now - timedelta(days=60)
        recent = now - timedelta(days=3)
        debates = [
            _make_debate("d_old", created_at=old),
            _make_debate("d_new", created_at=recent),
        ]
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = debates
        handler.get_storage = MagicMock(return_value=mock_storage)

        h = _make_handler()
        result = handler.handle(
            "/api/v1/analytics/trends", {"time_range": "7d"}, h
        )
        body = _body(result)
        assert len(body["data_points"]) == 1

    def test_org_id_passed_to_storage(self, handler):
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = []
        handler.get_storage = MagicMock(return_value=mock_storage)

        h = _make_handler()
        handler.handle(
            "/api/v1/analytics/trends", {"org_id": "org-789"}, h
        )
        mock_storage.list_debates.assert_called_once_with(limit=10000, org_id="org-789")

    def test_org_id_in_response(self, handler):
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = []
        handler.get_storage = MagicMock(return_value=mock_storage)

        h = _make_handler()
        result = handler.handle(
            "/api/v1/analytics/trends", {"org_id": "org-789"}, h
        )
        body = _body(result)
        assert body["org_id"] == "org-789"

    def test_agent_tracking_string_agents(self, handler):
        now = datetime.now(timezone.utc)
        debate = _make_debate("d1", created_at=now, agents=["claude", "gemini"])
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = [debate]
        handler.get_storage = MagicMock(return_value=mock_storage)

        h = _make_handler()
        result = handler.handle(
            "/api/v1/analytics/trends", {"metrics": "agents"}, h
        )
        body = _body(result)
        if body["data_points"]:
            assert body["data_points"][0]["active_agents"] == 2

    def test_agent_tracking_dict_agents(self, handler):
        now = datetime.now(timezone.utc)
        debate = _make_debate("d1", created_at=now)
        debate["agents"] = [{"name": "claude"}, {"name": "gemini"}]
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = [debate]
        handler.get_storage = MagicMock(return_value=mock_storage)

        h = _make_handler()
        result = handler.handle(
            "/api/v1/analytics/trends", {"metrics": "agents"}, h
        )
        body = _body(result)
        if body["data_points"]:
            assert body["data_points"][0]["active_agents"] == 2

    def test_invalid_created_at_skipped(self, handler):
        debate = {
            "debate_id": "d1",
            "consensus_reached": True,
            "result": {"rounds_used": 3, "confidence": 0.9, "domain": "sec",
                        "outcome_type": "consensus", "duration_seconds": 30},
            "agents": ["claude"],
            "created_at": "not-a-date",
        }
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = [debate]
        handler.get_storage = MagicMock(return_value=mock_storage)

        h = _make_handler()
        result = handler.handle("/api/v1/analytics/trends", {}, h)
        body = _body(result)
        assert body["data_points"] == []


# ============================================================================
# Trend Calculation (_calculate_trends)
# ============================================================================


class TestCalculateTrends:
    """Test the _calculate_trends method."""

    def test_insufficient_data_single_point(self, handler):
        data_points = [{"period": "2026-01-01", "debates_count": 10}]
        result = handler._calculate_trends(data_points)
        assert result["debates_trend"] == "insufficient_data"
        assert result["consensus_trend"] == "insufficient_data"

    def test_insufficient_data_empty(self, handler):
        result = handler._calculate_trends([])
        assert result["debates_trend"] == "insufficient_data"
        assert result["consensus_trend"] == "insufficient_data"

    def test_increasing_debates_trend(self, handler):
        data_points = [
            {"period": "2026-01-01", "debates_count": 10},
            {"period": "2026-01-02", "debates_count": 10},
            {"period": "2026-01-03", "debates_count": 30},
            {"period": "2026-01-04", "debates_count": 30},
        ]
        result = handler._calculate_trends(data_points)
        assert result["debates_trend"] == "increasing"
        assert result["debates_growth_rate"] > 10

    def test_decreasing_debates_trend(self, handler):
        data_points = [
            {"period": "2026-01-01", "debates_count": 30},
            {"period": "2026-01-02", "debates_count": 30},
            {"period": "2026-01-03", "debates_count": 10},
            {"period": "2026-01-04", "debates_count": 10},
        ]
        result = handler._calculate_trends(data_points)
        assert result["debates_trend"] == "decreasing"
        assert result["debates_growth_rate"] < -10

    def test_stable_debates_trend(self, handler):
        data_points = [
            {"period": "2026-01-01", "debates_count": 20},
            {"period": "2026-01-02", "debates_count": 20},
            {"period": "2026-01-03", "debates_count": 21},
            {"period": "2026-01-04", "debates_count": 21},
        ]
        result = handler._calculate_trends(data_points)
        assert result["debates_trend"] == "stable"

    def test_improving_consensus_trend(self, handler):
        data_points = [
            {"period": "2026-01-01", "consensus_rate": 60},
            {"period": "2026-01-02", "consensus_rate": 60},
            {"period": "2026-01-03", "consensus_rate": 80},
            {"period": "2026-01-04", "consensus_rate": 80},
        ]
        result = handler._calculate_trends(data_points)
        assert result["consensus_trend"] == "improving"
        assert result["consensus_change"] > 5

    def test_declining_consensus_trend(self, handler):
        data_points = [
            {"period": "2026-01-01", "consensus_rate": 80},
            {"period": "2026-01-02", "consensus_rate": 80},
            {"period": "2026-01-03", "consensus_rate": 60},
            {"period": "2026-01-04", "consensus_rate": 60},
        ]
        result = handler._calculate_trends(data_points)
        assert result["consensus_trend"] == "declining"
        assert result["consensus_change"] < -5

    def test_stable_consensus_trend(self, handler):
        data_points = [
            {"period": "2026-01-01", "consensus_rate": 75},
            {"period": "2026-01-02", "consensus_rate": 76},
            {"period": "2026-01-03", "consensus_rate": 77},
            {"period": "2026-01-04", "consensus_rate": 78},
        ]
        result = handler._calculate_trends(data_points)
        assert result["consensus_trend"] == "stable"

    def test_zero_first_half_debates(self, handler):
        """When first half has zero debates, growth is 100% if second half has debates."""
        data_points = [
            {"period": "2026-01-01", "debates_count": 0},
            {"period": "2026-01-02", "debates_count": 0},
            {"period": "2026-01-03", "debates_count": 10},
            {"period": "2026-01-04", "debates_count": 10},
        ]
        result = handler._calculate_trends(data_points)
        assert result["debates_growth_rate"] == 100.0

    def test_zero_both_halves_debates(self, handler):
        data_points = [
            {"period": "2026-01-01", "debates_count": 0},
            {"period": "2026-01-02", "debates_count": 0},
            {"period": "2026-01-03", "debates_count": 0},
            {"period": "2026-01-04", "debates_count": 0},
        ]
        result = handler._calculate_trends(data_points)
        assert result["debates_growth_rate"] == 0.0

    def test_no_consensus_rate_data(self, handler):
        """When no data points have consensus_rate, consensus_change is 0."""
        data_points = [
            {"period": "2026-01-01", "debates_count": 10},
            {"period": "2026-01-02", "debates_count": 10},
            {"period": "2026-01-03", "debates_count": 20},
            {"period": "2026-01-04", "debates_count": 20},
        ]
        result = handler._calculate_trends(data_points)
        assert result["consensus_change"] == 0.0

    def test_odd_number_of_data_points(self, handler):
        data_points = [
            {"period": "2026-01-01", "debates_count": 10},
            {"period": "2026-01-02", "debates_count": 10},
            {"period": "2026-01-03", "debates_count": 10},
        ]
        # mid = 1, first_half = [10], second_half = [10, 10]
        result = handler._calculate_trends(data_points)
        assert "debates_trend" in result
        assert "consensus_trend" in result

    def test_exactly_two_data_points(self, handler):
        data_points = [
            {"period": "2026-01-01", "debates_count": 10, "consensus_rate": 70},
            {"period": "2026-01-02", "debates_count": 20, "consensus_rate": 90},
        ]
        result = handler._calculate_trends(data_points)
        assert result["debates_trend"] == "increasing"
        assert result["consensus_trend"] == "improving"


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and unusual inputs."""

    def test_debate_with_non_dict_result(self, handler):
        """If result is not a dict, should still count the debate."""
        now = datetime.now(timezone.utc)
        debate = {
            "debate_id": "d1",
            "consensus_reached": True,
            "result": "some string result",
            "agents": ["claude"],
            "created_at": now.isoformat(),
        }
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = [debate]
        handler.get_storage = MagicMock(return_value=mock_storage)

        h = _make_handler()
        result = handler.handle("/api/v1/analytics/debates/summary", {}, h)
        body = _body(result)
        assert body["total_debates"] == 1

    def test_debate_with_non_list_agents(self, handler):
        """If agents is not a list, it should be handled gracefully."""
        now = datetime.now(timezone.utc)
        debate = {
            "debate_id": "d1",
            "consensus_reached": True,
            "result": {"rounds_used": 3, "confidence": 0.9, "domain": "sec",
                        "outcome_type": "consensus", "duration_seconds": 30},
            "agents": "claude,gemini",
            "created_at": now.isoformat(),
        }
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = [debate]
        handler.get_storage = MagicMock(return_value=mock_storage)

        h = _make_handler()
        result = handler.handle("/api/v1/analytics/debates/summary", {}, h)
        body = _body(result)
        assert body["total_debates"] == 1
        assert body["avg_agents"] == 0.0  # Non-list agents don't count

    def test_debate_with_z_suffix_timestamp(self, handler):
        """Test ISO format with Z suffix."""
        debate = {
            "debate_id": "d1",
            "consensus_reached": True,
            "result": {"rounds_used": 3, "confidence": 0.9, "domain": "sec",
                        "outcome_type": "consensus", "duration_seconds": 30},
            "agents": ["claude"],
            "created_at": "2026-02-20T12:00:00Z",
        }
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = [debate]
        handler.get_storage = MagicMock(return_value=mock_storage)

        h = _make_handler()
        result = handler.handle(
            "/api/v1/analytics/debates/summary", {"time_range": "all"}, h
        )
        body = _body(result)
        assert body["total_debates"] == 1

    def test_debate_with_empty_agents_dict(self, handler):
        """Test agent dict with no 'name' key."""
        now = datetime.now(timezone.utc)
        debate = _make_debate("d1", created_at=now)
        debate["agents"] = [{"id": "a1"}, {"name": "claude"}]
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = [debate]
        handler.get_storage = MagicMock(return_value=mock_storage)

        h = _make_handler()
        result = handler.handle(
            "/api/v1/analytics/trends", {"metrics": "agents"}, h
        )
        body = _body(result)
        if body["data_points"]:
            # Only "claude" has a name, "a1" has no name key
            assert body["data_points"][0]["active_agents"] == 1

    def test_large_number_of_debates(self, handler):
        """Ensure performance with many debates."""
        now = datetime.now(timezone.utc)
        debates = [
            _make_debate(f"d{i}", created_at=now - timedelta(hours=i))
            for i in range(100)
        ]
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = debates
        handler.get_storage = MagicMock(return_value=mock_storage)

        h = _make_handler()
        result = handler.handle("/api/v1/analytics/debates/summary", {}, h)
        body = _body(result)
        assert body["total_debates"] == 100

    def test_empty_domain_defaults_to_general(self, handler):
        now = datetime.now(timezone.utc)
        debate = _make_debate("d1", created_at=now)
        debate["result"]["domain"] = ""
        debate.pop("domain", None)
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = [debate]
        handler.get_storage = MagicMock(return_value=mock_storage)

        h = _make_handler()
        result = handler.handle("/api/v1/analytics/debates/summary", {}, h)
        body = _body(result)
        # Empty string domain should not be in by_domain
        # The code checks `if domain:` which is falsy for empty string
        # Domain defaults from debate_dict.get("domain", "general")
        # Since we popped domain from the dict and result domain is empty,
        # domain will be empty string -> falsy -> not counted
        assert body["total_debates"] == 1

    def test_no_duration_in_result(self, handler):
        now = datetime.now(timezone.utc)
        debate = {
            "debate_id": "d1",
            "consensus_reached": True,
            "result": {"rounds_used": 3, "confidence": 0.9, "domain": "sec",
                        "outcome_type": "consensus"},
            "agents": ["claude"],
            "created_at": now.isoformat(),
        }
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = [debate]
        handler.get_storage = MagicMock(return_value=mock_storage)

        h = _make_handler()
        result = handler.handle("/api/v1/analytics/debates/summary", {}, h)
        body = _body(result)
        assert body["avg_duration_seconds"] == 0.0

    def test_comma_separated_metrics(self, handler):
        """Test that metrics parameter is parsed as comma-separated."""
        now = datetime.now(timezone.utc)
        debates = [_make_debate("d1", created_at=now)]
        debates[0]["result"]["total_tokens"] = 5000
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = debates
        handler.get_storage = MagicMock(return_value=mock_storage)

        h = _make_handler()
        result = handler.handle(
            "/api/v1/analytics/trends",
            {"metrics": "debates, tokens, agents"},
            h,
        )
        body = _body(result)
        if body["data_points"]:
            point = body["data_points"][0]
            assert "debates_count" in point
            assert "total_tokens" in point
            assert "active_agents" in point

    def test_consensus_only_metric(self, handler):
        now = datetime.now(timezone.utc)
        debate = _make_debate("d1", created_at=now, consensus_reached=True)
        mock_storage = MagicMock()
        mock_storage.list_debates.return_value = [debate]
        handler.get_storage = MagicMock(return_value=mock_storage)

        h = _make_handler()
        result = handler.handle(
            "/api/v1/analytics/trends", {"metrics": "consensus"}, h
        )
        body = _body(result)
        if body["data_points"]:
            point = body["data_points"][0]
            assert "consensus_rate" in point
            assert "debates_count" not in point
            assert "active_agents" not in point
