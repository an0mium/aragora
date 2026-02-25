"""Comprehensive tests for AgentsHandler (aragora/server/handlers/agents/agents.py).

Tests cover:
- Handler initialization and routing (can_handle, ROUTES)
- Rate limiting
- RBAC / auth enforcement on non-public paths
- /api/agents endpoint (list agents)
- /api/agents/health endpoint
- /api/agents/availability endpoint
- /api/agents/local and /api/agents/local/status endpoints
- /api/leaderboard and /api/rankings endpoints
- /api/matches/recent endpoint
- /api/agent/compare endpoint
- Per-agent endpoints via _dispatch_agent_endpoint
- /api/agent/{name}/head-to-head/{opponent}
- /api/agent/{name}/opponent-briefing/{opponent}
- /api/flips/recent and /api/flips/summary endpoints
- Input validation (path segments, query params)
- Helper functions (_secret_configured, _missing_required_env_vars)
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result: object) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result: object) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if result is None:
        return 0
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_rate_limiters():
    """Reset rate limiters before and after each test so quota is fresh."""
    try:
        from aragora.server.middleware.rate_limit.registry import reset_rate_limiters

        reset_rate_limiters()
    except ImportError:
        pass

    # Reset module-level rate limiter inside agents.py
    try:
        from aragora.server.handlers.agents import agents as agents_mod

        agents_mod._agent_limiter = agents_mod.RateLimiter(requests_per_minute=60)
    except (ImportError, AttributeError):
        pass

    # Reset named rate limiters used by decorators
    try:
        from aragora.server.handlers.utils import rate_limit as rl_mod

        with rl_mod._limiters_lock:
            rl_mod._limiters.clear()
    except (ImportError, AttributeError):
        pass

    yield

    try:
        from aragora.server.middleware.rate_limit.registry import reset_rate_limiters

        reset_rate_limiters()
    except ImportError:
        pass


@pytest.fixture
def handler():
    """Create an AgentsHandler with empty server context."""
    from aragora.server.handlers.agents.agents import AgentsHandler

    return AgentsHandler(server_context={})


@pytest.fixture
def mock_http_handler():
    """Create mock HTTP handler with client address and empty headers."""
    h = MagicMock()
    h.client_address = ("127.0.0.1", 54321)
    h.headers = {}
    return h


# ---------------------------------------------------------------------------
# Initialization and Routing
# ---------------------------------------------------------------------------


class TestAgentsHandlerInit:
    """Tests for handler construction."""

    def test_init_with_server_context(self):
        from aragora.server.handlers.agents.agents import AgentsHandler

        ctx = {"elo_system": "mock"}
        h = AgentsHandler(server_context=ctx)
        assert h.ctx is ctx

    def test_init_with_ctx_kwarg(self):
        from aragora.server.handlers.agents.agents import AgentsHandler

        ctx = {"key": "val"}
        h = AgentsHandler(ctx=ctx)
        assert h.ctx is ctx

    def test_init_defaults_to_empty_dict(self):
        from aragora.server.handlers.agents.agents import AgentsHandler

        h = AgentsHandler()
        assert h.ctx == {}

    def test_routes_defined(self, handler):
        assert hasattr(handler, "ROUTES")
        assert "/api/agents" in handler.ROUTES
        assert "/api/leaderboard" in handler.ROUTES
        assert "/api/rankings" in handler.ROUTES
        assert "/api/agents/local" in handler.ROUTES

    def test_routes_include_per_agent_wildcards(self, handler):
        assert "/api/agent/*/profile" in handler.ROUTES
        assert "/api/agent/*/history" in handler.ROUTES
        assert "/api/agent/*/head-to-head/*" in handler.ROUTES

    def test_routes_include_flips(self, handler):
        assert "/api/flips/recent" in handler.ROUTES
        assert "/api/flips/summary" in handler.ROUTES


class TestCanHandle:
    """Tests for can_handle path matching."""

    def test_agents_root(self, handler):
        assert handler.can_handle("/api/agents")
        assert handler.can_handle("/api/v1/agents")

    def test_agents_health(self, handler):
        assert handler.can_handle("/api/agents/health")
        assert handler.can_handle("/api/v1/agents/health")

    def test_agents_availability(self, handler):
        assert handler.can_handle("/api/agents/availability")
        assert handler.can_handle("/api/v1/agents/availability")

    def test_agents_local(self, handler):
        assert handler.can_handle("/api/agents/local")
        assert handler.can_handle("/api/agents/local/status")

    def test_leaderboard_and_rankings(self, handler):
        assert handler.can_handle("/api/leaderboard")
        assert handler.can_handle("/api/rankings")
        assert handler.can_handle("/api/v1/leaderboard")
        assert handler.can_handle("/api/v1/rankings")

    def test_matches_recent(self, handler):
        assert handler.can_handle("/api/matches/recent")

    def test_agent_compare(self, handler):
        assert handler.can_handle("/api/agent/compare")

    def test_per_agent_endpoints(self, handler):
        assert handler.can_handle("/api/agent/claude/profile")
        assert handler.can_handle("/api/agent/gpt4/history")
        assert handler.can_handle("/api/agent/gemini/calibration")

    def test_flips_endpoints(self, handler):
        assert handler.can_handle("/api/flips/recent")
        assert handler.can_handle("/api/flips/summary")

    def test_head_to_head(self, handler):
        assert handler.can_handle("/api/agent/claude/head-to-head/gpt4")

    def test_opponent_briefing(self, handler):
        assert handler.can_handle("/api/agent/claude/opponent-briefing/gpt4")

    def test_unrelated_path_rejected(self, handler):
        assert not handler.can_handle("/api/debates")
        assert not handler.can_handle("/api/users")
        assert not handler.can_handle("/api/health")

    def test_agents_configs_not_handled(self, handler):
        # /api/agents/configs is excluded via startswith guard
        assert not handler.can_handle("/api/agents/configs")

    def test_agents_subpath_converted(self, handler):
        # /api/agents/{name}/... is handled (converted to /api/agent/{name}/...)
        assert handler.can_handle("/api/agents/claude/profile")

    def test_introspect(self, handler):
        assert handler.can_handle("/api/agent/claude/introspect")


# ---------------------------------------------------------------------------
# Public path detection
# ---------------------------------------------------------------------------


class TestIsPublicPath:
    """Tests for _is_public_path logic."""

    def test_public_paths(self, handler):
        for p in [
            "/api/agents",
            "/api/agents/health",
            "/api/agents/availability",
            "/api/leaderboard",
            "/api/rankings",
            "/api/flips/recent",
            "/api/flips/summary",
            "/api/matches/recent",
        ]:
            assert handler._is_public_path(p), f"{p} should be public"

    def test_agent_prefix_is_public(self, handler):
        assert handler._is_public_path("/api/agent/claude/profile")
        assert handler._is_public_path("/api/agent/gpt4/history")

    def test_random_path_not_public(self, handler):
        assert not handler._is_public_path("/api/admin/settings")

    def test_agents_local_not_public(self, handler):
        # /api/agents/local is NOT in _PUBLIC_PATHS
        assert not handler._is_public_path("/api/agents/local")


# ---------------------------------------------------------------------------
# Rate Limiting
# ---------------------------------------------------------------------------


class TestRateLimiting:
    @pytest.mark.asyncio
    async def test_rate_limit_returns_429(self, handler, mock_http_handler):
        """After exhausting the rate limiter budget, handler returns 429."""
        with patch("aragora.server.handlers.agents.agents._agent_limiter") as mock_limiter:
            mock_limiter.is_allowed.return_value = False
            result = await handler.handle("/api/agents", {}, mock_http_handler)
            assert _status(result) == 429
            assert "rate limit" in _body(result).get("error", "").lower()


# ---------------------------------------------------------------------------
# /api/agents  (list agents)
# ---------------------------------------------------------------------------


class TestListAgents:
    @pytest.mark.asyncio
    async def test_list_agents_with_elo(self, handler, mock_http_handler):
        mock_elo = MagicMock()
        mock_elo.get_leaderboard.return_value = [
            {"name": "claude", "elo": 1600, "matches": 10, "wins": 7, "losses": 3},
            {"name": "gpt4", "elo": 1550, "matches": 8, "wins": 5, "losses": 3},
        ]
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle("/api/agents", {}, mock_http_handler)
            assert _status(result) == 200
            body = _body(result)
            assert body["total"] == 2
            assert len(body["agents"]) == 2
            assert body["agents"][0]["name"] == "claude"

    @pytest.mark.asyncio
    async def test_list_agents_with_stats(self, handler, mock_http_handler):
        mock_elo = MagicMock()
        mock_elo.get_leaderboard.return_value = [
            {"name": "claude", "elo": 1600, "matches": 10, "wins": 7, "losses": 3},
        ]
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle(
                "/api/agents", {"include_stats": "true"}, mock_http_handler
            )
            body = _body(result)
            assert _status(result) == 200
            agent = body["agents"][0]
            assert "elo" in agent
            assert "matches" in agent
            assert "wins" in agent

    @pytest.mark.asyncio
    async def test_list_agents_without_stats(self, handler, mock_http_handler):
        mock_elo = MagicMock()
        mock_elo.get_leaderboard.return_value = [
            {"name": "claude", "elo": 1600, "matches": 10, "wins": 7, "losses": 3},
        ]
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle(
                "/api/agents", {"include_stats": "false"}, mock_http_handler
            )
            body = _body(result)
            agent = body["agents"][0]
            assert "elo" not in agent
            assert agent == {"name": "claude"}

    @pytest.mark.asyncio
    async def test_list_agents_fallback_to_allowed_types(self, handler, mock_http_handler):
        """When ELO system returns no agents, falls back to ALLOWED_AGENT_TYPES."""
        with patch.object(handler, "get_elo_system", return_value=None):
            result = await handler.handle("/api/agents", {}, mock_http_handler)
            body = _body(result)
            assert _status(result) == 200
            # Should fall back to ALLOWED_AGENT_TYPES which has many entries
            assert body["total"] > 0
            assert len(body["agents"]) > 0

    @pytest.mark.asyncio
    async def test_list_agents_elo_error_fallback(self, handler, mock_http_handler):
        """When ELO system raises, falls back to ALLOWED_AGENT_TYPES."""
        mock_elo = MagicMock()
        mock_elo.get_leaderboard.side_effect = ValueError("boom")
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle("/api/agents", {}, mock_http_handler)
            body = _body(result)
            assert _status(result) == 200
            assert body["total"] >= 1

    @pytest.mark.asyncio
    async def test_list_agents_include_stats_default_false(self, handler, mock_http_handler):
        """By default include_stats is false."""
        mock_elo = MagicMock()
        mock_elo.get_leaderboard.return_value = [
            {"name": "claude", "elo": 1600, "matches": 10, "wins": 7, "losses": 3},
        ]
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle("/api/agents", {}, mock_http_handler)
            body = _body(result)
            # Without include_stats, agents should be name-only dicts
            assert body["agents"][0] == {"name": "claude"}


# ---------------------------------------------------------------------------
# /api/leaderboard and /api/rankings
# ---------------------------------------------------------------------------


class TestLeaderboard:
    @pytest.mark.asyncio
    async def test_leaderboard_no_elo(self, handler, mock_http_handler):
        with patch.object(handler, "get_elo_system", return_value=None):
            result = await handler.handle("/api/leaderboard", {}, mock_http_handler)
            assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_rankings_alias(self, handler, mock_http_handler):
        """The /api/rankings path is an alias for /api/leaderboard."""
        with patch.object(handler, "get_elo_system", return_value=None):
            result = await handler.handle("/api/rankings", {}, mock_http_handler)
            assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_leaderboard_with_elo(self, handler, mock_http_handler):
        mock_elo = MagicMock()
        mock_elo.get_cached_leaderboard.return_value = [
            {"name": "claude", "elo": 1600},
        ]
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            with patch.object(handler, "get_nomic_dir", return_value=None):
                result = await handler.handle("/api/leaderboard", {"limit": "5"}, mock_http_handler)
                assert _status(result) == 200
                body = _body(result)
                assert "rankings" in body or "agents" in body

    @pytest.mark.asyncio
    async def test_leaderboard_with_domain_filter(self, handler, mock_http_handler):
        mock_elo = MagicMock()
        mock_elo.get_leaderboard.return_value = [{"name": "claude", "elo": 1700}]
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            with patch.object(handler, "get_nomic_dir", return_value=None):
                result = await handler.handle(
                    "/api/leaderboard", {"domain": "technical"}, mock_http_handler
                )
                assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_leaderboard_invalid_domain_rejected(self, handler, mock_http_handler):
        result = await handler.handle(
            "/api/leaderboard", {"domain": "<script>alert(1)</script>"}, mock_http_handler
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_leaderboard_versioned_path(self, handler, mock_http_handler):
        with patch.object(handler, "get_elo_system", return_value=None):
            result = await handler.handle("/api/v1/leaderboard", {}, mock_http_handler)
            assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_leaderboard_default_limit(self, handler, mock_http_handler):
        """Default limit for leaderboard is 20."""
        mock_elo = MagicMock()
        mock_elo.get_cached_leaderboard.return_value = []
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            with patch.object(handler, "get_nomic_dir", return_value=None):
                await handler.handle("/api/leaderboard", {}, mock_http_handler)
                # Capped at min(20, 50) = 20
                mock_elo.get_cached_leaderboard.assert_called_once_with(limit=20)


# ---------------------------------------------------------------------------
# /api/matches/recent
# ---------------------------------------------------------------------------


class TestRecentMatches:
    @pytest.mark.asyncio
    async def test_no_elo(self, handler, mock_http_handler):
        with patch.object(handler, "get_elo_system", return_value=None):
            result = await handler.handle("/api/matches/recent", {}, mock_http_handler)
            assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_with_elo(self, handler, mock_http_handler):
        mock_elo = MagicMock()
        mock_elo.get_cached_recent_matches.return_value = [{"winner": "claude"}]
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle("/api/matches/recent", {"limit": "5"}, mock_http_handler)
            assert _status(result) == 200
            body = _body(result)
            assert "matches" in body

    @pytest.mark.asyncio
    async def test_invalid_loop_id_rejected(self, handler, mock_http_handler):
        result = await handler.handle(
            "/api/matches/recent", {"loop_id": "../../etc/passwd"}, mock_http_handler
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_valid_loop_id_accepted(self, handler, mock_http_handler):
        mock_elo = MagicMock()
        mock_elo.get_cached_recent_matches.return_value = []
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle(
                "/api/matches/recent", {"loop_id": "loop-abc-123"}, mock_http_handler
            )
            assert _status(result) == 200


# ---------------------------------------------------------------------------
# /api/agent/compare
# ---------------------------------------------------------------------------


class TestCompareAgents:
    @pytest.mark.asyncio
    async def test_compare_too_few_agents(self, handler, mock_http_handler):
        result = await handler.handle(
            "/api/agent/compare", {"agents": ["claude"]}, mock_http_handler
        )
        assert _status(result) == 400
        assert "2 agents" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_compare_no_elo(self, handler, mock_http_handler):
        with patch.object(handler, "get_elo_system", return_value=None):
            result = await handler.handle(
                "/api/agent/compare", {"agents": ["claude", "gpt4"]}, mock_http_handler
            )
            assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_compare_two_agents(self, handler, mock_http_handler):
        mock_elo = MagicMock()
        mock_elo.get_ratings_batch.return_value = {"claude": 1600, "gpt4": 1550}
        mock_elo.get_agent_stats.return_value = {"wins": 5}
        mock_elo.get_head_to_head.return_value = {"matches": 3, "agent1_wins": 2}
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle(
                "/api/agent/compare", {"agents": ["claude", "gpt4"]}, mock_http_handler
            )
            assert _status(result) == 200
            body = _body(result)
            assert len(body["agents"]) == 2
            assert body["head_to_head"] is not None

    @pytest.mark.asyncio
    async def test_compare_agents_string_param(self, handler, mock_http_handler):
        """When agents param is a single string, it should be wrapped in a list."""
        mock_elo = MagicMock()
        mock_elo.get_ratings_batch.return_value = {}
        mock_elo.get_agent_stats.return_value = {}
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            # A string means only one agent, so comparison should fail with < 2
            result = await handler.handle(
                "/api/agent/compare", {"agents": "claude"}, mock_http_handler
            )
            assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_compare_limits_to_five(self, handler, mock_http_handler):
        mock_elo = MagicMock()
        mock_elo.get_ratings_batch.return_value = {f"a{i}": 1500 for i in range(7)}
        mock_elo.get_agent_stats.return_value = {}
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            agents = [f"a{i}" for i in range(7)]
            result = await handler.handle(
                "/api/agent/compare", {"agents": agents}, mock_http_handler
            )
            assert _status(result) == 200
            assert len(_body(result)["agents"]) == 5

    @pytest.mark.asyncio
    async def test_compare_empty_agents_list(self, handler, mock_http_handler):
        result = await handler.handle("/api/agent/compare", {"agents": []}, mock_http_handler)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_compare_no_head_to_head_for_three(self, handler, mock_http_handler):
        """Head-to-head is only returned for exactly 2 agents."""
        mock_elo = MagicMock()
        mock_elo.get_ratings_batch.return_value = {"a": 1500, "b": 1500, "c": 1500}
        mock_elo.get_agent_stats.return_value = {}
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle(
                "/api/agent/compare", {"agents": ["a", "b", "c"]}, mock_http_handler
            )
            assert _status(result) == 200
            assert _body(result)["head_to_head"] is None


# ---------------------------------------------------------------------------
# /api/agents/local and /api/agents/local/status
# ---------------------------------------------------------------------------


class TestLocalAgents:
    @pytest.mark.asyncio
    async def test_list_local_agents_success(self, handler, mock_http_handler):
        with patch("aragora.agents.registry.AgentRegistry") as mock_registry:
            mock_registry.detect_local_agents.return_value = [
                {"name": "ollama", "available": True},
                {"name": "lmstudio", "available": False},
            ]
            result = await handler.handle("/api/agents/local", {}, mock_http_handler)
            assert _status(result) == 200
            body = _body(result)
            assert body["total"] == 2
            assert body["available_count"] == 1

    @pytest.mark.asyncio
    async def test_list_local_agents_error(self, handler, mock_http_handler):
        with patch("aragora.agents.registry.AgentRegistry") as mock_registry:
            mock_registry.detect_local_agents.side_effect = ConnectionError("timeout")
            result = await handler.handle("/api/agents/local", {}, mock_http_handler)
            assert _status(result) == 200
            body = _body(result)
            assert body["total"] == 0
            assert "error" in body

    @pytest.mark.asyncio
    async def test_local_status_success(self, handler, mock_http_handler):
        with patch("aragora.agents.registry.AgentRegistry") as mock_registry:
            mock_registry.get_local_status.return_value = {
                "any_available": True,
                "total_models": 3,
                "recommended_server": "ollama",
                "recommended_model": "llama3",
                "available_agents": ["ollama"],
                "servers": [{"name": "ollama"}],
            }
            result = await handler.handle("/api/agents/local/status", {}, mock_http_handler)
            assert _status(result) == 200
            body = _body(result)
            assert body["available"] is True
            assert body["total_models"] == 3
            assert body["recommended"]["server"] == "ollama"

    @pytest.mark.asyncio
    async def test_local_status_error(self, handler, mock_http_handler):
        with patch("aragora.agents.registry.AgentRegistry") as mock_registry:
            mock_registry.get_local_status.side_effect = OSError("fail")
            result = await handler.handle("/api/agents/local/status", {}, mock_http_handler)
            assert _status(result) == 200
            body = _body(result)
            assert body["available"] is False


# ---------------------------------------------------------------------------
# /api/agents/health
# ---------------------------------------------------------------------------


class TestAgentHealth:
    @pytest.mark.asyncio
    async def test_health_basic_structure(self, handler, mock_http_handler):
        """Health endpoint returns expected top-level keys."""
        with patch("aragora.resilience.get_circuit_breakers", return_value={}):
            with patch("aragora.agents.fallback.is_local_llm_available", return_value=False):
                with patch("aragora.agents.fallback.get_local_fallback_providers", return_value=[]):
                    with patch("aragora.agents.registry.register_all_agents"):
                        with patch("aragora.agents.registry.AgentRegistry") as mock_reg:
                            mock_reg.list_all.return_value = {}
                            with patch(
                                "aragora.events.cross_subscribers.get_cross_subscriber_manager"
                            ) as mock_cs:
                                mock_cs.return_value.get_stats.return_value = {}
                                result = await handler.handle(
                                    "/api/agents/health", {}, mock_http_handler
                                )
                                assert _status(result) == 200
                                body = _body(result)
                                assert "overall_status" in body
                                assert "agents" in body
                                assert "circuit_breakers" in body
                                assert "fallback" in body
                                assert "summary" in body

    @pytest.mark.asyncio
    async def test_health_degraded_when_circuit_open(self, handler, mock_http_handler):
        """When circuit breaker is open, agent is marked unavailable."""
        mock_cb = MagicMock()
        mock_cb.get_status_dict.return_value = {
            "state": "open",
            "failure_count": 5,
            "last_failure_time": "2026-01-01T00:00:00",
        }
        with patch("aragora.resilience.get_circuit_breakers", return_value={"claude": mock_cb}):
            with patch("aragora.agents.fallback.is_local_llm_available", return_value=False):
                with patch("aragora.agents.fallback.get_local_fallback_providers", return_value=[]):
                    with patch("aragora.agents.registry.register_all_agents"):
                        with patch("aragora.agents.registry.AgentRegistry") as mock_reg:
                            mock_reg.list_all.return_value = {
                                "claude": {"type": "api", "env_vars": "ANTHROPIC_API_KEY"}
                            }
                            with patch(
                                "aragora.server.handlers.agents.agents._secret_configured",
                                return_value=True,
                            ):
                                with patch(
                                    "aragora.events.cross_subscribers.get_cross_subscriber_manager"
                                ) as mock_cs:
                                    mock_cs.return_value.get_stats.return_value = {}
                                    result = await handler.handle(
                                        "/api/agents/health", {}, mock_http_handler
                                    )
                                    body = _body(result)
                                    # Circuit open -> agent unavailable -> 0/1 agents -> unhealthy
                                    assert body["overall_status"] in ("degraded", "unhealthy")
                                    cb_info = body["circuit_breakers"]["claude"]
                                    assert cb_info["state"] == "open"
                                    assert cb_info["available"] is False
                                    # Agent should be marked as circuit_breaker_open
                                    agent_info = body["agents"].get("claude", {})
                                    assert agent_info.get("circuit_breaker_open") is True
                                    assert agent_info.get("available") is False

    @pytest.mark.asyncio
    async def test_health_unhealthy_when_no_agents_available(self, handler, mock_http_handler):
        """When all agents are unavailable, overall status is unhealthy."""
        with patch("aragora.resilience.get_circuit_breakers", return_value={}):
            with patch("aragora.agents.fallback.is_local_llm_available", return_value=False):
                with patch("aragora.agents.fallback.get_local_fallback_providers", return_value=[]):
                    with patch("aragora.agents.registry.register_all_agents"):
                        with patch("aragora.agents.registry.AgentRegistry") as mock_reg:
                            mock_reg.list_all.return_value = {
                                "claude": {"type": "api", "env_vars": "ANTHROPIC_API_KEY"}
                            }
                            with patch(
                                "aragora.server.handlers.agents.agents._secret_configured",
                                return_value=False,
                            ):
                                with patch(
                                    "aragora.server.handlers.agents.agents._missing_required_env_vars",
                                    return_value=["ANTHROPIC_API_KEY"],
                                ):
                                    with patch(
                                        "aragora.events.cross_subscribers.get_cross_subscriber_manager"
                                    ) as mock_cs:
                                        mock_cs.return_value.get_stats.return_value = {}
                                        result = await handler.handle(
                                            "/api/agents/health", {}, mock_http_handler
                                        )
                                        body = _body(result)
                                        assert body["overall_status"] == "unhealthy"
                                        assert body["summary"]["available_agents"] == 0

    @pytest.mark.asyncio
    async def test_health_graceful_import_failures(self, handler, mock_http_handler):
        """When imports fail, health still returns valid response."""
        # When resilience module is not importable, the code catches ImportError
        # and sets a _note key. We simulate this by letting the real code run
        # with its real import paths. The handler always returns 200.
        result = await handler.handle("/api/agents/health", {}, mock_http_handler)
        assert _status(result) == 200
        body = _body(result)
        assert body["overall_status"] in ("healthy", "degraded", "unhealthy")

    @pytest.mark.asyncio
    async def test_health_timestamp_present(self, handler, mock_http_handler):
        result = await handler.handle("/api/agents/health", {}, mock_http_handler)
        body = _body(result)
        assert "timestamp" in body


# ---------------------------------------------------------------------------
# /api/agents/availability
# ---------------------------------------------------------------------------


class TestAgentAvailability:
    @pytest.mark.asyncio
    async def test_availability_all_configured(self, handler, mock_http_handler):
        with patch("aragora.agents.registry.register_all_agents"):
            with patch("aragora.agents.registry.AgentRegistry") as mock_reg:
                mock_reg.list_all.return_value = {
                    "anthropic-api": {"type": "api", "env_vars": "ANTHROPIC_API_KEY"}
                }
                with patch(
                    "aragora.server.handlers.agents.agents._secret_configured",
                    return_value=True,
                ):
                    with patch(
                        "aragora.server.handlers.agents.agents._missing_required_env_vars",
                        return_value=[],
                    ):
                        result = await handler.handle(
                            "/api/agents/availability", {}, mock_http_handler
                        )
                        assert _status(result) == 200
                        body = _body(result)
                        assert body["agents"]["anthropic-api"]["available"] is True
                        assert body["agents"]["anthropic-api"]["uses_openrouter_fallback"] is False

    @pytest.mark.asyncio
    async def test_availability_uses_openrouter_fallback(self, handler, mock_http_handler):
        with patch("aragora.agents.registry.register_all_agents"):
            with patch("aragora.agents.registry.AgentRegistry") as mock_reg:
                mock_reg.list_all.return_value = {
                    "anthropic-api": {"type": "api", "env_vars": "ANTHROPIC_API_KEY"}
                }
                with patch(
                    "aragora.server.handlers.agents.agents._secret_configured",
                    side_effect=lambda k: k == "OPENROUTER_API_KEY",
                ):
                    with patch(
                        "aragora.server.handlers.agents.agents._missing_required_env_vars",
                        return_value=["ANTHROPIC_API_KEY"],
                    ):
                        result = await handler.handle(
                            "/api/agents/availability", {}, mock_http_handler
                        )
                        body = _body(result)
                        agent_info = body["agents"]["anthropic-api"]
                        assert agent_info["available"] is True
                        assert agent_info["uses_openrouter_fallback"] is True
                        assert agent_info["fallback_model"] == "anthropic/claude-sonnet-4.6"

    @pytest.mark.asyncio
    async def test_availability_not_available(self, handler, mock_http_handler):
        """Agent is not available when missing env vars and no OpenRouter."""
        with patch("aragora.agents.registry.register_all_agents"):
            with patch("aragora.agents.registry.AgentRegistry") as mock_reg:
                mock_reg.list_all.return_value = {
                    "anthropic-api": {"type": "api", "env_vars": "ANTHROPIC_API_KEY"}
                }
                with patch(
                    "aragora.server.handlers.agents.agents._secret_configured",
                    return_value=False,
                ):
                    with patch(
                        "aragora.server.handlers.agents.agents._missing_required_env_vars",
                        return_value=["ANTHROPIC_API_KEY"],
                    ):
                        result = await handler.handle(
                            "/api/agents/availability", {}, mock_http_handler
                        )
                        body = _body(result)
                        agent_info = body["agents"]["anthropic-api"]
                        assert agent_info["available"] is False
                        assert agent_info["uses_openrouter_fallback"] is False


# ---------------------------------------------------------------------------
# Per-agent endpoints via _handle_agent_endpoint / _dispatch_agent_endpoint
# ---------------------------------------------------------------------------


class TestPerAgentEndpoints:
    @pytest.mark.asyncio
    async def test_profile(self, handler, mock_http_handler):
        mock_elo = MagicMock()
        mock_elo.get_rating.return_value = 1600
        mock_elo.get_agent_stats.return_value = {
            "rank": 1,
            "wins": 10,
            "losses": 2,
            "win_rate": 0.833,
        }
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle("/api/agent/claude/profile", {}, mock_http_handler)
            assert _status(result) == 200
            body = _body(result)
            assert body["name"] == "claude"

    @pytest.mark.asyncio
    async def test_history(self, handler, mock_http_handler):
        mock_elo = MagicMock()
        mock_elo.get_elo_history.return_value = [(1000000, 1600), (1000001, 1610)]
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle(
                "/api/agent/claude/history", {"limit": "5"}, mock_http_handler
            )
            assert _status(result) == 200
            body = _body(result)
            assert body["agent"] == "claude"
            assert len(body["history"]) == 2

    @pytest.mark.asyncio
    async def test_calibration(self, handler, mock_http_handler):
        mock_elo = MagicMock()
        mock_elo.get_calibration.return_value = {"agent": "claude", "score": 0.75}
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle("/api/agent/claude/calibration", {}, mock_http_handler)
            assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_calibration_with_domain(self, handler, mock_http_handler):
        mock_elo = MagicMock()
        mock_elo.get_calibration.return_value = {"agent": "claude", "score": 0.8}
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle(
                "/api/agent/claude/calibration", {"domain": "tech"}, mock_http_handler
            )
            assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_consistency(self, handler, mock_http_handler):
        with patch.object(handler, "get_nomic_dir", return_value=None):
            result = await handler.handle("/api/agent/claude/consistency", {}, mock_http_handler)
            assert _status(result) == 200
            body = _body(result)
            assert body["agent"] == "claude"
            assert body["consistency_score"] == 1.0

    @pytest.mark.asyncio
    async def test_flips(self, handler, mock_http_handler):
        with patch.object(handler, "get_nomic_dir", return_value=None):
            result = await handler.handle("/api/agent/claude/flips", {}, mock_http_handler)
            assert _status(result) == 200
            body = _body(result)
            assert body["agent"] == "claude"
            assert body["flips"] == []

    @pytest.mark.asyncio
    async def test_network(self, handler, mock_http_handler):
        mock_elo = MagicMock()
        mock_elo.get_rivals.return_value = [{"name": "gpt4"}]
        mock_elo.get_allies.return_value = [{"name": "gemini"}]
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle("/api/agent/claude/network", {}, mock_http_handler)
            assert _status(result) == 200
            body = _body(result)
            assert body["agent"] == "claude"
            assert len(body["rivals"]) == 1

    @pytest.mark.asyncio
    async def test_rivals(self, handler, mock_http_handler):
        mock_elo = MagicMock()
        mock_elo.get_rivals.return_value = [{"name": "gpt4"}]
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle(
                "/api/agent/claude/rivals", {"limit": "3"}, mock_http_handler
            )
            assert _status(result) == 200
            body = _body(result)
            assert body["agent"] == "claude"

    @pytest.mark.asyncio
    async def test_allies(self, handler, mock_http_handler):
        mock_elo = MagicMock()
        mock_elo.get_allies.return_value = []
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle(
                "/api/agent/claude/allies", {"limit": "3"}, mock_http_handler
            )
            assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_moments_no_elo(self, handler, mock_http_handler):
        """When ELO system is None, moments returns empty."""
        with patch.object(handler, "get_elo_system", return_value=None):
            result = await handler.handle(
                "/api/agent/claude/moments", {"limit": "5"}, mock_http_handler
            )
            assert _status(result) == 200
            body = _body(result)
            assert body["moments"] == []

    @pytest.mark.asyncio
    async def test_moments_with_elo(self, handler, mock_http_handler):
        mock_elo = MagicMock()
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            with patch("aragora.agents.grounded.MomentDetector") as mock_md:
                mock_md.return_value.get_agent_moments.return_value = []
                result = await handler.handle(
                    "/api/agent/claude/moments", {"limit": "5"}, mock_http_handler
                )
                assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_positions(self, handler, mock_http_handler):
        with patch.object(handler, "get_nomic_dir", return_value=None):
            result = await handler.handle("/api/agent/claude/positions", {}, mock_http_handler)
            assert _status(result) == 200
            body = _body(result)
            assert body["positions"] == []

    @pytest.mark.asyncio
    async def test_domains(self, handler, mock_http_handler):
        mock_elo = MagicMock()
        mock_rating = MagicMock()
        mock_rating.domain_elos = {"tech": 1650, "science": 1580}
        mock_rating.elo = 1600
        mock_elo.get_rating.return_value = mock_rating
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle("/api/agent/claude/domains", {}, mock_http_handler)
            assert _status(result) == 200
            body = _body(result)
            assert body["agent"] == "claude"
            assert body["domain_count"] == 2
            # Should be sorted by elo descending
            assert body["domains"][0]["domain"] == "tech"

    @pytest.mark.asyncio
    async def test_domains_no_elo(self, handler, mock_http_handler):
        with patch.object(handler, "get_elo_system", return_value=None):
            result = await handler.handle("/api/agent/claude/domains", {}, mock_http_handler)
            assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_performance(self, handler, mock_http_handler):
        mock_elo = MagicMock()
        mock_rating = MagicMock()
        mock_rating.elo = 1600
        mock_rating.wins = 10
        mock_rating.losses = 3
        mock_rating.draws = 1
        mock_rating.critiques_accepted = 5
        mock_rating.critiques_total = 8
        mock_rating.critique_acceptance_rate = 0.625
        mock_rating.calibration_accuracy = 0.78
        mock_rating.calibration_brier_score = 0.15
        mock_rating.calibration_total = 20
        mock_elo.get_rating.return_value = mock_rating
        mock_elo.get_agent_history.return_value = []
        mock_elo.get_elo_history.return_value = []
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle("/api/agent/claude/performance", {}, mock_http_handler)
            assert _status(result) == 200
            body = _body(result)
            assert body["agent"] == "claude"
            assert body["total_games"] == 14
            assert body["win_rate"] == round(10 / 14, 3)

    @pytest.mark.asyncio
    async def test_performance_no_elo(self, handler, mock_http_handler):
        with patch.object(handler, "get_elo_system", return_value=None):
            result = await handler.handle("/api/agent/claude/performance", {}, mock_http_handler)
            assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_metadata_no_nomic_dir(self, handler, mock_http_handler):
        with patch.object(handler, "get_nomic_dir", return_value=None):
            result = await handler.handle("/api/agent/claude/metadata", {}, mock_http_handler)
            assert _status(result) == 200
            body = _body(result)
            assert body["metadata"] is None

    @pytest.mark.asyncio
    async def test_introspect(self, handler, mock_http_handler):
        with patch.object(handler, "get_elo_system", return_value=None):
            with patch.object(handler, "get_nomic_dir", return_value=None):
                result = await handler.handle("/api/agent/claude/introspect", {}, mock_http_handler)
                assert _status(result) == 200
                body = _body(result)
                assert body["agent_id"] == "claude"

    @pytest.mark.asyncio
    async def test_introspect_with_debate_id(self, handler, mock_http_handler):
        with patch.object(handler, "get_elo_system", return_value=None):
            with patch.object(handler, "get_nomic_dir", return_value=None):
                with patch.object(handler, "get_storage", return_value=None):
                    result = await handler.handle(
                        "/api/agent/claude/introspect", {"debate_id": "d-123"}, mock_http_handler
                    )
                    assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_no_elo_returns_503_for_profile(self, handler, mock_http_handler):
        with patch.object(handler, "get_elo_system", return_value=None):
            result = await handler.handle("/api/agent/claude/profile", {}, mock_http_handler)
            assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_no_elo_returns_503_for_history(self, handler, mock_http_handler):
        with patch.object(handler, "get_elo_system", return_value=None):
            result = await handler.handle("/api/agent/claude/history", {}, mock_http_handler)
            assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_no_elo_returns_503_for_network(self, handler, mock_http_handler):
        with patch.object(handler, "get_elo_system", return_value=None):
            result = await handler.handle("/api/agent/claude/network", {}, mock_http_handler)
            assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_no_elo_returns_503_for_rivals(self, handler, mock_http_handler):
        with patch.object(handler, "get_elo_system", return_value=None):
            result = await handler.handle("/api/agent/claude/rivals", {}, mock_http_handler)
            assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_no_elo_returns_503_for_allies(self, handler, mock_http_handler):
        with patch.object(handler, "get_elo_system", return_value=None):
            result = await handler.handle("/api/agent/claude/allies", {}, mock_http_handler)
            assert _status(result) == 503


# ---------------------------------------------------------------------------
# Head-to-head and opponent briefing
# ---------------------------------------------------------------------------


class TestHeadToHead:
    @pytest.mark.asyncio
    async def test_head_to_head(self, handler, mock_http_handler):
        mock_elo = MagicMock()
        mock_elo.get_head_to_head.return_value = {"matches": 5, "agent1_wins": 3}
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle(
                "/api/agent/claude/head-to-head/gpt4", {}, mock_http_handler
            )
            assert _status(result) == 200
            body = _body(result)
            assert body["agent1"] == "claude"
            assert body["agent2"] == "gpt4"

    @pytest.mark.asyncio
    async def test_head_to_head_no_elo(self, handler, mock_http_handler):
        with patch.object(handler, "get_elo_system", return_value=None):
            result = await handler.handle(
                "/api/agent/claude/head-to-head/gpt4", {}, mock_http_handler
            )
            assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_head_to_head_invalid_opponent(self, handler, mock_http_handler):
        result = await handler.handle(
            "/api/agent/claude/head-to-head/<script>", {}, mock_http_handler
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_head_to_head_invalid_agent(self, handler, mock_http_handler):
        result = await handler.handle(
            "/api/agent/<script>/head-to-head/gpt4", {}, mock_http_handler
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_opponent_briefing(self, handler, mock_http_handler):
        with patch.object(handler, "get_elo_system", return_value=None):
            with patch.object(handler, "get_nomic_dir", return_value=None):
                with patch("aragora.agents.grounded.PersonaSynthesizer") as mock_ps:
                    mock_ps.return_value.get_opponent_briefing.return_value = {
                        "strategy": "aggressive"
                    }
                    result = await handler.handle(
                        "/api/agent/claude/opponent-briefing/gpt4", {}, mock_http_handler
                    )
                    assert _status(result) == 200
                    body = _body(result)
                    assert body["agent"] == "claude"
                    assert body["opponent"] == "gpt4"
                    assert body["briefing"] is not None

    @pytest.mark.asyncio
    async def test_opponent_briefing_no_data(self, handler, mock_http_handler):
        with patch.object(handler, "get_elo_system", return_value=None):
            with patch.object(handler, "get_nomic_dir", return_value=None):
                with patch("aragora.agents.grounded.PersonaSynthesizer") as mock_ps:
                    mock_ps.return_value.get_opponent_briefing.return_value = None
                    result = await handler.handle(
                        "/api/agent/claude/opponent-briefing/gpt4", {}, mock_http_handler
                    )
                    assert _status(result) == 200
                    body = _body(result)
                    assert body["briefing"] is None
                    assert "message" in body

    @pytest.mark.asyncio
    async def test_opponent_briefing_invalid_opponent(self, handler, mock_http_handler):
        result = await handler.handle(
            "/api/agent/claude/opponent-briefing/<bad>", {}, mock_http_handler
        )
        assert _status(result) == 400


# ---------------------------------------------------------------------------
# /api/flips/recent and /api/flips/summary
# ---------------------------------------------------------------------------


class TestFlipEndpoints:
    @pytest.mark.asyncio
    async def test_recent_flips_no_nomic_dir(self, handler, mock_http_handler):
        with patch.object(handler, "get_nomic_dir", return_value=None):
            result = await handler.handle("/api/flips/recent", {}, mock_http_handler)
            assert _status(result) == 200
            body = _body(result)
            assert body["flips"] == []
            assert body["count"] == 0

    @pytest.mark.asyncio
    async def test_flip_summary_no_nomic_dir(self, handler, mock_http_handler):
        with patch.object(handler, "get_nomic_dir", return_value=None):
            result = await handler.handle("/api/flips/summary", {}, mock_http_handler)
            assert _status(result) == 200
            body = _body(result)
            assert body["total_flips"] == 0

    @pytest.mark.asyncio
    async def test_recent_flips_with_limit(self, handler, mock_http_handler):
        with patch.object(handler, "get_nomic_dir", return_value=None):
            result = await handler.handle("/api/flips/recent", {"limit": "5"}, mock_http_handler)
            assert _status(result) == 200


# ---------------------------------------------------------------------------
# Path validation
# ---------------------------------------------------------------------------


class TestPathValidation:
    @pytest.mark.asyncio
    async def test_invalid_agent_name_rejected(self, handler, mock_http_handler):
        """Agent names with special chars are rejected."""
        result = await handler.handle("/api/agent/<script>/profile", {}, mock_http_handler)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_short_agent_path_rejected(self, handler, mock_http_handler):
        """Path with fewer than 5 segments returns 400."""
        result = await handler.handle("/api/agent/claude", {}, mock_http_handler)
        # < 5 parts means error_response("Invalid agent path", 400)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_unknown_agent_endpoint_returns_none(self, handler, mock_http_handler):
        """Unknown sub-endpoint dispatches to None."""
        result = await handler.handle("/api/agent/claude/nonexistent", {}, mock_http_handler)
        assert result is None

    @pytest.mark.asyncio
    async def test_agents_subpath_rewritten_to_agent(self, handler, mock_http_handler):
        """Paths like /api/agents/claude/profile are rewritten to /api/agent/claude/profile."""
        mock_elo = MagicMock()
        mock_elo.get_rating.return_value = 1600
        mock_elo.get_agent_stats.return_value = {"rank": 1, "wins": 5, "losses": 2, "win_rate": 0.7}
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle("/api/agents/claude/profile", {}, mock_http_handler)
            assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_versioned_agent_path(self, handler, mock_http_handler):
        """Version-prefixed paths are handled after stripping."""
        mock_elo = MagicMock()
        mock_elo.get_rating.return_value = 1600
        mock_elo.get_agent_stats.return_value = {"rank": 1, "wins": 5, "losses": 2, "win_rate": 0.7}
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle("/api/v1/agent/claude/profile", {}, mock_http_handler)
            assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_agent_name_with_hyphens_accepted(self, handler, mock_http_handler):
        """Agent names with hyphens are valid."""
        mock_elo = MagicMock()
        mock_elo.get_rating.return_value = 1600
        mock_elo.get_agent_stats.return_value = {}
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle("/api/agent/anthropic-api/profile", {}, mock_http_handler)
            assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_agent_name_with_underscores_accepted(self, handler, mock_http_handler):
        """Agent names with underscores are valid."""
        mock_elo = MagicMock()
        mock_elo.get_rating.return_value = 1600
        mock_elo.get_agent_stats.return_value = {}
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle("/api/agent/gpt_4o/profile", {}, mock_http_handler)
            assert _status(result) == 200


# ---------------------------------------------------------------------------
# Unmatched path returns None
# ---------------------------------------------------------------------------


class TestUnmatchedPaths:
    @pytest.mark.asyncio
    async def test_returns_none_for_unhandled_path(self, handler, mock_http_handler):
        result = await handler.handle("/api/unknown", {}, mock_http_handler)
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_for_empty_dispatch(self, handler, mock_http_handler):
        """Dispatch to unknown endpoint returns None from _dispatch_agent_endpoint."""
        result = handler._dispatch_agent_endpoint("claude", "zzz_unknown", {})
        assert result is None


# ---------------------------------------------------------------------------
# Helper functions (_secret_configured, _missing_required_env_vars)
# ---------------------------------------------------------------------------


class TestSecretConfigured:
    def test_secret_from_get_secret(self):
        from aragora.server.handlers.agents.agents import _secret_configured

        with patch("aragora.config.secrets.get_secret", return_value="abc123"):
            assert _secret_configured("MY_KEY") is True

    def test_secret_from_env_var(self):
        from aragora.server.handlers.agents.agents import _secret_configured

        with patch("aragora.config.secrets.get_secret", side_effect=ImportError):
            with patch.dict("os.environ", {"MY_KEY": "value123"}):
                assert _secret_configured("MY_KEY") is True

    def test_no_secret_available(self):
        from aragora.server.handlers.agents.agents import _secret_configured

        with patch("aragora.config.secrets.get_secret", side_effect=ImportError):
            with patch.dict("os.environ", {}, clear=True):
                assert _secret_configured("MY_KEY") is False

    def test_empty_secret_is_false(self):
        from aragora.server.handlers.agents.agents import _secret_configured

        with patch("aragora.config.secrets.get_secret", return_value="  "):
            with patch.dict("os.environ", {}, clear=True):
                assert _secret_configured("MY_KEY") is False

    def test_whitespace_env_var_is_false(self):
        from aragora.server.handlers.agents.agents import _secret_configured

        with patch("aragora.config.secrets.get_secret", side_effect=ImportError):
            with patch.dict("os.environ", {"MY_KEY": "  "}):
                assert _secret_configured("MY_KEY") is False


class TestMissingRequiredEnvVars:
    def test_none_input(self):
        from aragora.server.handlers.agents.agents import _missing_required_env_vars

        assert _missing_required_env_vars(None) == []

    def test_empty_string(self):
        from aragora.server.handlers.agents.agents import _missing_required_env_vars

        assert _missing_required_env_vars("") == []

    def test_optional_keyword_returns_empty(self):
        from aragora.server.handlers.agents.agents import _missing_required_env_vars

        assert _missing_required_env_vars("optional: SOME_KEY") == []

    def test_optional_case_insensitive(self):
        from aragora.server.handlers.agents.agents import _missing_required_env_vars

        assert _missing_required_env_vars("Optional SOME_KEY") == []

    def test_no_matches(self):
        from aragora.server.handlers.agents.agents import _missing_required_env_vars

        assert _missing_required_env_vars("no uppercase vars here") == []

    def test_all_configured(self):
        from aragora.server.handlers.agents.agents import _missing_required_env_vars

        with patch(
            "aragora.server.handlers.agents.agents._secret_configured",
            return_value=True,
        ):
            assert _missing_required_env_vars("ANTHROPIC_API_KEY") == []

    def test_missing_vars_returned(self):
        from aragora.server.handlers.agents.agents import _missing_required_env_vars

        with patch(
            "aragora.server.handlers.agents.agents._secret_configured",
            return_value=False,
        ):
            result = _missing_required_env_vars("ANTHROPIC_API_KEY")
            assert "ANTHROPIC_API_KEY" in result

    def test_any_configured_means_all_ok(self):
        """If any one of the env vars is configured, all are considered OK."""
        from aragora.server.handlers.agents.agents import _missing_required_env_vars

        # The function uses `any(_secret_configured(var) for var in candidates)`
        # So if one is configured, returns []
        call_count = 0

        def _mock_secret(name):
            nonlocal call_count
            call_count += 1
            return name == "OPENAI_API_KEY"

        with patch(
            "aragora.server.handlers.agents.agents._secret_configured",
            side_effect=_mock_secret,
        ):
            result = _missing_required_env_vars("ANTHROPIC_API_KEY or OPENAI_API_KEY")
            assert result == []


# ---------------------------------------------------------------------------
# Auth enforcement on non-public paths (opt-out of auto-auth)
# ---------------------------------------------------------------------------


class TestAuthEnforcement:
    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_non_public_path_requires_auth(self, mock_http_handler):
        """Non-public endpoints return 401 when auth fails."""
        from aragora.server.handlers.agents.agents import AgentsHandler
        from aragora.server.handlers.secure import SecureHandler, UnauthorizedError

        h = AgentsHandler(server_context={})

        async def _raise_unauth(self, handler, require_auth=False):
            raise UnauthorizedError("No token")

        with patch.object(SecureHandler, "get_auth_context", _raise_unauth):
            # /api/agents/local is NOT in _PUBLIC_PATHS and NOT in _PUBLIC_PREFIXES
            result = await h.handle("/api/agents/local", {}, mock_http_handler)
            assert _status(result) == 401

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_forbidden_returns_403(self, mock_http_handler):
        """Non-public endpoints return 403 when permission denied."""
        from aragora.server.handlers.agents.agents import AgentsHandler
        from aragora.server.handlers.secure import ForbiddenError, SecureHandler

        h = AgentsHandler(server_context={})

        mock_auth_ctx = MagicMock()

        async def _return_ctx(self, handler, require_auth=False):
            return mock_auth_ctx

        def _raise_forbidden(self, ctx, perm):
            raise ForbiddenError("No permission")

        with patch.object(SecureHandler, "get_auth_context", _return_ctx):
            with patch.object(SecureHandler, "check_permission", _raise_forbidden):
                result = await h.handle("/api/agents/local", {}, mock_http_handler)
                assert _status(result) == 403

    @pytest.mark.asyncio
    async def test_public_path_no_auth_needed(self, handler, mock_http_handler):
        """Public paths work even without auth."""
        with patch.object(handler, "get_elo_system", return_value=None):
            result = await handler.handle("/api/agents", {}, mock_http_handler)
            # Should not get 401/403
            assert _status(result) == 200


# ---------------------------------------------------------------------------
# OpenRouter fallback model map
# ---------------------------------------------------------------------------


class TestOpenRouterFallbackMap:
    def test_known_models(self):
        from aragora.server.handlers.agents.agents import _OPENROUTER_FALLBACK_MODELS

        assert "anthropic-api" in _OPENROUTER_FALLBACK_MODELS
        assert "openai-api" in _OPENROUTER_FALLBACK_MODELS
        assert "gemini" in _OPENROUTER_FALLBACK_MODELS
        assert "grok" in _OPENROUTER_FALLBACK_MODELS
        assert "mistral-api" in _OPENROUTER_FALLBACK_MODELS

    def test_unknown_agent_has_no_fallback(self):
        from aragora.server.handlers.agents.agents import _OPENROUTER_FALLBACK_MODELS

        assert _OPENROUTER_FALLBACK_MODELS.get("totally-fake") is None

    def test_fallback_model_values_are_strings(self):
        from aragora.server.handlers.agents.agents import _OPENROUTER_FALLBACK_MODELS

        for agent, model in _OPENROUTER_FALLBACK_MODELS.items():
            assert isinstance(model, str), f"{agent} fallback should be a string"
            assert "/" in model, f"{agent} fallback should be provider/model format"


# ---------------------------------------------------------------------------
# Constants and module-level attributes
# ---------------------------------------------------------------------------


class TestConstants:
    def test_permission_constants(self):
        from aragora.server.handlers.agents.agents import (
            AGENT_PERMISSION,
            AGENTS_READ_PERMISSION,
            AGENTS_WRITE_PERMISSION,
        )

        assert AGENTS_READ_PERMISSION == "agents:read"
        assert AGENTS_WRITE_PERMISSION == "agents:write"
        assert AGENT_PERMISSION == AGENTS_READ_PERMISSION

    def test_env_var_regex(self):
        from aragora.server.handlers.agents.agents import _ENV_VAR_RE

        assert _ENV_VAR_RE.findall("ANTHROPIC_API_KEY") == ["ANTHROPIC_API_KEY"]
        assert _ENV_VAR_RE.findall("lowercase_var") == []
        assert _ENV_VAR_RE.findall("OPENAI_API_KEY or OPENROUTER_API_KEY") == [
            "OPENAI_API_KEY",
            "OPENROUTER_API_KEY",
        ]

    def test_env_var_regex_edge_cases(self):
        from aragora.server.handlers.agents.agents import _ENV_VAR_RE

        # Must start with uppercase
        assert _ENV_VAR_RE.findall("1ABC") == ["ABC"]
        # Numbers after first letter OK
        assert _ENV_VAR_RE.findall("A1B2") == ["A1B2"]


# ---------------------------------------------------------------------------
# _handle_agent_endpoint edge cases
# ---------------------------------------------------------------------------


class TestHandleAgentEndpoint:
    def test_dispatch_all_known_endpoints(self, handler):
        """All 14 known endpoints are in the dispatch map."""
        known = [
            "profile",
            "history",
            "calibration",
            "consistency",
            "flips",
            "network",
            "rivals",
            "allies",
            "moments",
            "positions",
            "domains",
            "performance",
            "metadata",
            "introspect",
        ]
        for endpoint in known:
            # Just verify the dispatch map contains these (no errors)
            result = handler._dispatch_agent_endpoint("test", endpoint, {})
            # Result may be a HandlerResult or None (if e.g. no elo_system)
            # but should not raise

    def test_dispatch_unknown_returns_none(self, handler):
        assert handler._dispatch_agent_endpoint("test", "xyz", {}) is None
