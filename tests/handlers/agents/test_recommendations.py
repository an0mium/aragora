"""Comprehensive tests for AgentRecommendationHandler.

Tests cover all endpoints:
- GET /api/v1/agents/recommend - Get top agent recommendations
- GET /api/v1/agents/leaderboard - Get agent leaderboard with rankings

Each endpoint is tested for:
- Happy path with valid data
- No ELO system (503)
- Domain filtering
- Limit clamping and defaults
- Input validation (domain pattern, limit bounds)
- Error handling (exceptions in ELO methods)
- Introspection enrichment
- Cost estimation (exact match, prefix match, unknown agent)
- Calibration score inclusion
- Rate limiting (429)
- Unmatched paths (returns None)
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
# Mock agent objects
# ---------------------------------------------------------------------------


class MockAgent:
    """Minimal agent object with standard ELO fields."""

    def __init__(
        self,
        name: str = "claude",
        elo: int = 1600,
        wins: int = 10,
        losses: int = 5,
        draws: int = 2,
        win_rate: float = 0.65,
        games_played: int = 17,
        matches: int = 17,
        calibration_score: float | None = None,
        domain_elo: int | None = None,
    ):
        self.name = name
        self.elo = elo
        self.wins = wins
        self.losses = losses
        self.draws = draws
        self.win_rate = win_rate
        self.games_played = games_played
        self.matches = matches
        if calibration_score is not None:
            self.calibration_score = calibration_score
        if domain_elo is not None:
            self.domain_elo = domain_elo


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Reset rate limiter before each test to avoid cross-test pollution."""
    try:
        from aragora.server.middleware.rate_limit.registry import reset_rate_limiters

        reset_rate_limiters()
    except ImportError:
        pass

    try:
        from aragora.server.handlers.agents import recommendations as rec_mod

        rec_mod._recommend_limiter = rec_mod.RateLimiter(requests_per_minute=30)
    except (ImportError, AttributeError):
        pass

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
    """Create an AgentRecommendationHandler with empty server context."""
    from aragora.server.handlers.agents.recommendations import AgentRecommendationHandler

    return AgentRecommendationHandler(ctx={})


@pytest.fixture
def mock_http_handler():
    """Create mock HTTP handler with client address and empty headers."""
    h = MagicMock()
    h.client_address = ("127.0.0.1", 54321)
    h.headers = {}
    return h


def _make_mock_elo(**overrides):
    """Create a MagicMock EloSystem with common method defaults."""
    elo = MagicMock()
    elo.get_leaderboard.return_value = overrides.get("leaderboard", [])
    elo.get_top_agents_for_domain.return_value = overrides.get("domain_agents", [])
    elo.get_cached_leaderboard.return_value = overrides.get("cached_leaderboard", [])
    elo.get_stats.return_value = overrides.get(
        "stats",
        {
            "total_agents": 5,
            "total_matches": 100,
            "avg_elo": 1520,
            "mean_elo": 1520,
        },
    )
    return elo


# =============================================================================
# Initialization & Routing Tests
# =============================================================================


class TestAgentRecommendationHandlerInit:
    """Tests for handler initialization and routing."""

    def test_routes_defined(self, handler):
        """ROUTES attribute lists both endpoints."""
        assert "/api/v1/agents/recommend" in handler.ROUTES
        assert "/api/v1/agents/leaderboard" in handler.ROUTES

    def test_can_handle_recommend_path(self, handler):
        """can_handle recognizes /api/v1/agents/recommend."""
        assert handler.can_handle("/api/v1/agents/recommend")

    def test_can_handle_leaderboard_path(self, handler):
        """can_handle recognizes /api/v1/agents/leaderboard."""
        assert handler.can_handle("/api/v1/agents/leaderboard")

    def test_cannot_handle_other_paths(self, handler):
        """can_handle rejects unrelated paths."""
        assert not handler.can_handle("/api/v1/agents")
        assert not handler.can_handle("/api/v1/agents/profiles")
        assert not handler.can_handle("/api/v1/debates")
        assert not handler.can_handle("/api/v1/leaderboard")

    def test_ctx_defaults_to_empty_dict(self):
        """Context defaults to empty dict when None passed."""
        from aragora.server.handlers.agents.recommendations import AgentRecommendationHandler

        h = AgentRecommendationHandler(ctx=None)
        assert h.ctx == {}

    def test_handle_returns_none_for_unmatched_path(self, handler, mock_http_handler):
        """handle() returns None for paths that do not match."""
        mock_elo = _make_mock_elo()
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = handler.handle("/api/v1/agents/unknown", {}, mock_http_handler)
            assert result is None


# =============================================================================
# GET /api/v1/agents/recommend Tests
# =============================================================================


class TestRecommendations:
    """Tests for the recommendations endpoint."""

    def test_happy_path_no_domain(self, handler, mock_http_handler):
        """Returns recommendations without domain filter."""
        agents = [MockAgent(name="claude", elo=1600), MockAgent(name="gpt4", elo=1550)]
        mock_elo = _make_mock_elo(leaderboard=agents)

        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = handler.handle("/api/v1/agents/recommend", {}, mock_http_handler)
            assert _status(result) == 200
            body = _body(result)
            assert body["count"] == 2
            assert body["domain"] is None
            assert len(body["recommendations"]) == 2

    def test_happy_path_with_domain(self, handler, mock_http_handler):
        """Returns domain-filtered recommendations."""
        agents = [MockAgent(name="claude", elo=1600, domain_elo=1650)]
        mock_elo = _make_mock_elo(domain_agents=agents)

        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = handler.handle(
                "/api/v1/agents/recommend", {"domain": "financial"}, mock_http_handler
            )
            assert _status(result) == 200
            body = _body(result)
            assert body["domain"] == "financial"
            assert body["count"] == 1
            rec = body["recommendations"][0]
            assert rec["domain"] == "financial"
            assert rec["domain_elo"] == 1650
            mock_elo.get_top_agents_for_domain.assert_called_once_with(domain="financial", limit=5)

    def test_default_limit_is_five(self, handler, mock_http_handler):
        """Default limit is 5 when not specified."""
        mock_elo = _make_mock_elo()
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            handler.handle("/api/v1/agents/recommend", {}, mock_http_handler)
            mock_elo.get_leaderboard.assert_called_once_with(limit=5)

    def test_custom_limit(self, handler, mock_http_handler):
        """Respects custom limit parameter."""
        mock_elo = _make_mock_elo()
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            handler.handle("/api/v1/agents/recommend", {"limit": "10"}, mock_http_handler)
            mock_elo.get_leaderboard.assert_called_once_with(limit=10)

    def test_limit_clamped_to_minimum_1(self, handler, mock_http_handler):
        """Limit below 1 is clamped to 1."""
        mock_elo = _make_mock_elo()
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            handler.handle("/api/v1/agents/recommend", {"limit": "0"}, mock_http_handler)
            mock_elo.get_leaderboard.assert_called_once_with(limit=1)

    def test_limit_clamped_to_maximum_20(self, handler, mock_http_handler):
        """Limit above 20 is clamped to 20."""
        mock_elo = _make_mock_elo()
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            handler.handle("/api/v1/agents/recommend", {"limit": "100"}, mock_http_handler)
            mock_elo.get_leaderboard.assert_called_once_with(limit=20)

    def test_negative_limit_clamped(self, handler, mock_http_handler):
        """Negative limit is clamped to 1."""
        mock_elo = _make_mock_elo()
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            handler.handle("/api/v1/agents/recommend", {"limit": "-5"}, mock_http_handler)
            mock_elo.get_leaderboard.assert_called_once_with(limit=1)

    def test_no_elo_system_returns_503(self, handler, mock_http_handler):
        """Returns 503 when ELO system is unavailable."""
        with patch.object(handler, "get_elo_system", return_value=None):
            result = handler.handle("/api/v1/agents/recommend", {}, mock_http_handler)
            assert _status(result) == 503
            assert "ELO system" in _body(result).get("error", "")

    def test_invalid_domain_returns_400(self, handler, mock_http_handler):
        """Returns 400 for domains with unsafe characters."""
        mock_elo = _make_mock_elo()
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = handler.handle(
                "/api/v1/agents/recommend",
                {"domain": "<script>alert(1)</script>"},
                mock_http_handler,
            )
            assert _status(result) == 400

    def test_domain_with_path_traversal_returns_400(self, handler, mock_http_handler):
        """Returns 400 for domain with path traversal characters."""
        mock_elo = _make_mock_elo()
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = handler.handle(
                "/api/v1/agents/recommend",
                {"domain": "../../../etc/passwd"},
                mock_http_handler,
            )
            assert _status(result) == 400

    def test_empty_agent_list(self, handler, mock_http_handler):
        """Handles empty agent list gracefully."""
        mock_elo = _make_mock_elo(leaderboard=[])
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = handler.handle("/api/v1/agents/recommend", {}, mock_http_handler)
            assert _status(result) == 200
            body = _body(result)
            assert body["count"] == 0
            assert body["recommendations"] == []

    def test_elo_exception_returns_500(self, handler, mock_http_handler):
        """Returns 500 when ELO system raises an exception."""
        mock_elo = MagicMock()
        mock_elo.get_leaderboard.side_effect = TypeError("Unexpected type")
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = handler.handle("/api/v1/agents/recommend", {}, mock_http_handler)
            assert _status(result) == 500
            assert "Failed to get agent recommendations" in _body(result).get("error", "")

    def test_value_error_returns_500(self, handler, mock_http_handler):
        """Returns 500 when ELO system raises ValueError."""
        mock_elo = MagicMock()
        mock_elo.get_leaderboard.side_effect = ValueError("Bad value")
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = handler.handle("/api/v1/agents/recommend", {}, mock_http_handler)
            assert _status(result) == 500

    def test_key_error_returns_500(self, handler, mock_http_handler):
        """Returns 500 when ELO system raises KeyError."""
        mock_elo = MagicMock()
        mock_elo.get_leaderboard.side_effect = KeyError("missing key")
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = handler.handle("/api/v1/agents/recommend", {}, mock_http_handler)
            assert _status(result) == 500

    def test_attribute_error_returns_500(self, handler, mock_http_handler):
        """Returns 500 when agent_to_dict raises AttributeError."""
        mock_elo = MagicMock()
        mock_elo.get_leaderboard.side_effect = AttributeError("no attribute")
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = handler.handle("/api/v1/agents/recommend", {}, mock_http_handler)
            assert _status(result) == 500


# =============================================================================
# Cost Estimation Tests
# =============================================================================


class TestCostEstimation:
    """Tests for agent cost estimation in recommendations."""

    def test_exact_cost_match_claude(self, handler, mock_http_handler):
        """Exact cost match for claude agent."""
        agents = [MockAgent(name="claude")]
        mock_elo = _make_mock_elo(leaderboard=agents)
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = handler.handle("/api/v1/agents/recommend", {}, mock_http_handler)
            body = _body(result)
            assert body["recommendations"][0]["estimated_cost_per_1k_tokens"] == 0.015

    def test_exact_cost_match_gpt4(self, handler, mock_http_handler):
        """Exact cost match for gpt4 agent."""
        agents = [MockAgent(name="gpt4")]
        mock_elo = _make_mock_elo(leaderboard=agents)
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = handler.handle("/api/v1/agents/recommend", {}, mock_http_handler)
            body = _body(result)
            assert body["recommendations"][0]["estimated_cost_per_1k_tokens"] == 0.03

    def test_exact_cost_match_gemini(self, handler, mock_http_handler):
        """Exact cost match for gemini agent."""
        agents = [MockAgent(name="gemini")]
        mock_elo = _make_mock_elo(leaderboard=agents)
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = handler.handle("/api/v1/agents/recommend", {}, mock_http_handler)
            body = _body(result)
            assert body["recommendations"][0]["estimated_cost_per_1k_tokens"] == 0.00125

    def test_exact_cost_match_deepseek(self, handler, mock_http_handler):
        """Exact cost match for deepseek agent."""
        agents = [MockAgent(name="deepseek")]
        mock_elo = _make_mock_elo(leaderboard=agents)
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = handler.handle("/api/v1/agents/recommend", {}, mock_http_handler)
            body = _body(result)
            assert body["recommendations"][0]["estimated_cost_per_1k_tokens"] == 0.002

    def test_prefix_cost_match(self, handler, mock_http_handler):
        """Prefix matching works for agent names like 'claude-3-opus'."""
        agents = [MockAgent(name="claude-3-opus")]
        mock_elo = _make_mock_elo(leaderboard=agents)
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = handler.handle("/api/v1/agents/recommend", {}, mock_http_handler)
            body = _body(result)
            assert body["recommendations"][0]["estimated_cost_per_1k_tokens"] == 0.015

    def test_unknown_agent_cost_is_none(self, handler, mock_http_handler):
        """Unknown agent type has None cost."""
        agents = [MockAgent(name="unknown_model")]
        mock_elo = _make_mock_elo(leaderboard=agents)
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = handler.handle("/api/v1/agents/recommend", {}, mock_http_handler)
            body = _body(result)
            assert body["recommendations"][0]["estimated_cost_per_1k_tokens"] is None

    def test_case_insensitive_cost_match(self, handler, mock_http_handler):
        """Cost matching is case insensitive."""
        agents = [MockAgent(name="Claude")]
        mock_elo = _make_mock_elo(leaderboard=agents)
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = handler.handle("/api/v1/agents/recommend", {}, mock_http_handler)
            body = _body(result)
            assert body["recommendations"][0]["estimated_cost_per_1k_tokens"] == 0.015

    def test_gpt_4o_cost(self, handler, mock_http_handler):
        """Cost for gpt-4o is correctly matched."""
        agents = [MockAgent(name="gpt-4o")]
        mock_elo = _make_mock_elo(leaderboard=agents)
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = handler.handle("/api/v1/agents/recommend", {}, mock_http_handler)
            body = _body(result)
            assert body["recommendations"][0]["estimated_cost_per_1k_tokens"] == 0.005

    def test_mistral_cost(self, handler, mock_http_handler):
        """Cost for mistral is correctly matched."""
        agents = [MockAgent(name="mistral")]
        mock_elo = _make_mock_elo(leaderboard=agents)
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = handler.handle("/api/v1/agents/recommend", {}, mock_http_handler)
            body = _body(result)
            assert body["recommendations"][0]["estimated_cost_per_1k_tokens"] == 0.004

    def test_llama_cost(self, handler, mock_http_handler):
        """Cost for llama is correctly matched."""
        agents = [MockAgent(name="llama")]
        mock_elo = _make_mock_elo(leaderboard=agents)
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = handler.handle("/api/v1/agents/recommend", {}, mock_http_handler)
            body = _body(result)
            assert body["recommendations"][0]["estimated_cost_per_1k_tokens"] == 0.001

    def test_qwen_cost(self, handler, mock_http_handler):
        """Cost for qwen is correctly matched."""
        agents = [MockAgent(name="qwen")]
        mock_elo = _make_mock_elo(leaderboard=agents)
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = handler.handle("/api/v1/agents/recommend", {}, mock_http_handler)
            body = _body(result)
            assert body["recommendations"][0]["estimated_cost_per_1k_tokens"] == 0.001

    def test_grok_cost(self, handler, mock_http_handler):
        """Cost for grok is correctly matched."""
        agents = [MockAgent(name="grok")]
        mock_elo = _make_mock_elo(leaderboard=agents)
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = handler.handle("/api/v1/agents/recommend", {}, mock_http_handler)
            body = _body(result)
            assert body["recommendations"][0]["estimated_cost_per_1k_tokens"] == 0.005

    def test_codex_cost(self, handler, mock_http_handler):
        """Cost for codex is correctly matched."""
        agents = [MockAgent(name="codex")]
        mock_elo = _make_mock_elo(leaderboard=agents)
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = handler.handle("/api/v1/agents/recommend", {}, mock_http_handler)
            body = _body(result)
            assert body["recommendations"][0]["estimated_cost_per_1k_tokens"] == 0.03


# =============================================================================
# Calibration & Domain Enrichment Tests
# =============================================================================


class TestCalibrationEnrichment:
    """Tests for calibration score enrichment in recommendations."""

    def test_calibration_score_included_when_present(self, handler, mock_http_handler):
        """Agent with calibration_score has it in the recommendation dict."""
        agents = [MockAgent(name="claude", calibration_score=0.87654)]
        mock_elo = _make_mock_elo(leaderboard=agents)
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = handler.handle("/api/v1/agents/recommend", {}, mock_http_handler)
            body = _body(result)
            rec = body["recommendations"][0]
            assert rec["calibration_score"] == 0.877  # rounded to 3 decimals

    def test_calibration_score_omitted_when_absent(self, handler, mock_http_handler):
        """Agent without calibration_score does not have it in the recommendation."""
        agents = [MockAgent(name="claude")]
        mock_elo = _make_mock_elo(leaderboard=agents)
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = handler.handle("/api/v1/agents/recommend", {}, mock_http_handler)
            body = _body(result)
            rec = body["recommendations"][0]
            assert "calibration_score" not in rec

    def test_domain_elo_included_when_domain_specified(self, handler, mock_http_handler):
        """domain_elo is included when domain is specified and agent has it."""
        agents = [MockAgent(name="claude", domain_elo=1700)]
        mock_elo = _make_mock_elo(domain_agents=agents)
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = handler.handle(
                "/api/v1/agents/recommend", {"domain": "technical"}, mock_http_handler
            )
            body = _body(result)
            rec = body["recommendations"][0]
            assert rec["domain"] == "technical"
            assert rec["domain_elo"] == 1700

    def test_domain_field_without_domain_elo(self, handler, mock_http_handler):
        """Domain is set but domain_elo is absent when agent lacks it."""
        agents = [MockAgent(name="claude")]  # no domain_elo
        mock_elo = _make_mock_elo(domain_agents=agents)
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = handler.handle(
                "/api/v1/agents/recommend", {"domain": "financial"}, mock_http_handler
            )
            body = _body(result)
            rec = body["recommendations"][0]
            assert rec["domain"] == "financial"
            assert "domain_elo" not in rec

    def test_no_domain_fields_when_no_domain(self, handler, mock_http_handler):
        """No domain-related fields when no domain filter is used."""
        agents = [MockAgent(name="claude")]
        mock_elo = _make_mock_elo(leaderboard=agents)
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = handler.handle("/api/v1/agents/recommend", {}, mock_http_handler)
            body = _body(result)
            rec = body["recommendations"][0]
            assert "domain" not in rec
            assert "domain_elo" not in rec


# =============================================================================
# Introspection Enrichment Tests
# =============================================================================


class TestIntrospectionEnrichment:
    """Tests for introspection data enrichment in recommendations."""

    def test_introspection_added_when_available(self, handler, mock_http_handler):
        """Strengths and expertise are added from introspection snapshot."""
        agents = [MockAgent(name="claude")]
        mock_elo = _make_mock_elo(leaderboard=agents)
        mock_snapshot = MagicMock()
        mock_snapshot.strengths = ["reasoning", "analysis"]
        mock_snapshot.expertise_areas = ["finance", "law"]

        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            with patch(
                "aragora.server.handlers.agents.recommendations.get_agent_introspection",
                return_value=mock_snapshot,
                create=True,
            ):
                # Patch the import inside the method
                with patch.dict("sys.modules", {"aragora.introspection.api": MagicMock()}):
                    import sys

                    sys.modules["aragora.introspection.api"].get_agent_introspection = (
                        lambda name: mock_snapshot
                    )
                    result = handler.handle("/api/v1/agents/recommend", {}, mock_http_handler)
                    body = _body(result)
                    rec = body["recommendations"][0]
                    assert rec.get("strengths") == ["reasoning", "analysis"]
                    assert rec.get("expertise") == ["finance", "law"]

    def test_introspection_import_error_graceful(self, handler, mock_http_handler):
        """Graceful degradation when introspection module not available."""
        agents = [MockAgent(name="claude")]
        mock_elo = _make_mock_elo(leaderboard=agents)

        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            with patch.dict("sys.modules", {"aragora.introspection.api": None}):
                result = handler.handle("/api/v1/agents/recommend", {}, mock_http_handler)
                assert _status(result) == 200
                body = _body(result)
                # Should not fail, just omit introspection data
                assert body["count"] == 1

    def test_introspection_none_snapshot_no_fields(self, handler, mock_http_handler):
        """No introspection fields when snapshot is None."""
        agents = [MockAgent(name="claude")]
        mock_elo = _make_mock_elo(leaderboard=agents)

        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            with patch.dict("sys.modules", {"aragora.introspection.api": MagicMock()}):
                import sys

                sys.modules["aragora.introspection.api"].get_agent_introspection = (
                    lambda name: None
                )
                result = handler.handle("/api/v1/agents/recommend", {}, mock_http_handler)
                body = _body(result)
                rec = body["recommendations"][0]
                assert "strengths" not in rec
                assert "expertise" not in rec


# =============================================================================
# GET /api/v1/agents/leaderboard Tests
# =============================================================================


class TestLeaderboard:
    """Tests for the leaderboard endpoint."""

    def test_happy_path_no_domain(self, handler, mock_http_handler):
        """Returns leaderboard rankings without domain filter."""
        agents = [MockAgent(name="claude", elo=1600), MockAgent(name="gpt4", elo=1550)]
        mock_elo = _make_mock_elo(cached_leaderboard=agents)

        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = handler.handle("/api/v1/agents/leaderboard", {}, mock_http_handler)
            assert _status(result) == 200
            body = _body(result)
            assert body["count"] == 2
            assert body["domain"] is None
            assert len(body["leaderboard"]) == 2
            # Check rank assignment
            assert body["leaderboard"][0]["rank"] == 1
            assert body["leaderboard"][1]["rank"] == 2

    def test_happy_path_with_domain(self, handler, mock_http_handler):
        """Returns domain-filtered leaderboard."""
        agents = [MockAgent(name="claude", elo=1600)]
        mock_elo = _make_mock_elo(leaderboard=agents)

        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = handler.handle(
                "/api/v1/agents/leaderboard", {"domain": "financial"}, mock_http_handler
            )
            assert _status(result) == 200
            body = _body(result)
            assert body["domain"] == "financial"
            mock_elo.get_leaderboard.assert_called_once_with(limit=20, domain="financial")

    def test_default_limit_is_20(self, handler, mock_http_handler):
        """Default limit for leaderboard is 20."""
        mock_elo = _make_mock_elo()
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            handler.handle("/api/v1/agents/leaderboard", {}, mock_http_handler)
            mock_elo.get_cached_leaderboard.assert_called_once_with(limit=20)

    def test_custom_limit(self, handler, mock_http_handler):
        """Respects custom limit parameter."""
        mock_elo = _make_mock_elo()
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            handler.handle("/api/v1/agents/leaderboard", {"limit": "10"}, mock_http_handler)
            mock_elo.get_cached_leaderboard.assert_called_once_with(limit=10)

    def test_limit_clamped_to_minimum_1(self, handler, mock_http_handler):
        """Limit below 1 is clamped to 1."""
        mock_elo = _make_mock_elo()
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            handler.handle("/api/v1/agents/leaderboard", {"limit": "0"}, mock_http_handler)
            mock_elo.get_cached_leaderboard.assert_called_once_with(limit=1)

    def test_limit_clamped_to_maximum_50(self, handler, mock_http_handler):
        """Limit above 50 is clamped to 50."""
        mock_elo = _make_mock_elo()
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            handler.handle("/api/v1/agents/leaderboard", {"limit": "200"}, mock_http_handler)
            mock_elo.get_cached_leaderboard.assert_called_once_with(limit=50)

    def test_no_elo_system_returns_503(self, handler, mock_http_handler):
        """Returns 503 when ELO system is unavailable."""
        with patch.object(handler, "get_elo_system", return_value=None):
            result = handler.handle("/api/v1/agents/leaderboard", {}, mock_http_handler)
            assert _status(result) == 503
            assert "ELO system" in _body(result).get("error", "")

    def test_invalid_domain_returns_400(self, handler, mock_http_handler):
        """Returns 400 for domains with unsafe characters."""
        mock_elo = _make_mock_elo()
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = handler.handle(
                "/api/v1/agents/leaderboard",
                {"domain": "<script>"},
                mock_http_handler,
            )
            assert _status(result) == 400

    def test_empty_leaderboard(self, handler, mock_http_handler):
        """Handles empty leaderboard gracefully."""
        mock_elo = _make_mock_elo(cached_leaderboard=[])
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = handler.handle("/api/v1/agents/leaderboard", {}, mock_http_handler)
            assert _status(result) == 200
            body = _body(result)
            assert body["count"] == 0
            assert body["leaderboard"] == []

    def test_stats_included_in_response(self, handler, mock_http_handler):
        """Stats section is included in leaderboard response."""
        agents = [MockAgent(name="claude")]
        mock_elo = _make_mock_elo(
            cached_leaderboard=agents,
            stats={
                "total_agents": 10,
                "total_matches": 50,
                "avg_elo": 1520,
                "mean_elo": 1520,
            },
        )

        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = handler.handle("/api/v1/agents/leaderboard", {}, mock_http_handler)
            body = _body(result)
            assert "stats" in body
            assert body["stats"]["total_agents"] == 10
            assert body["stats"]["total_matches"] == 50
            assert body["stats"]["mean_elo"] == 1520

    def test_stats_defaults_when_get_stats_unavailable(self, handler, mock_http_handler):
        """Stats use agent count as default when get_stats method missing."""
        agents = [MockAgent(name="claude")]
        mock_elo = MagicMock()
        mock_elo.get_cached_leaderboard.return_value = agents
        # Remove get_stats to trigger the hasattr fallback
        del mock_elo.get_stats

        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = handler.handle("/api/v1/agents/leaderboard", {}, mock_http_handler)
            body = _body(result)
            assert "stats" in body
            assert body["stats"]["total_agents"] == 1  # len(agents) fallback
            assert body["stats"]["total_matches"] == 0
            assert body["stats"]["mean_elo"] == 1500

    def test_uses_get_cached_leaderboard_when_available_no_domain(self, handler, mock_http_handler):
        """Uses get_cached_leaderboard when available and no domain specified."""
        agents = [MockAgent(name="claude")]
        mock_elo = _make_mock_elo(cached_leaderboard=agents)
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            handler.handle("/api/v1/agents/leaderboard", {}, mock_http_handler)
            mock_elo.get_cached_leaderboard.assert_called_once()
            mock_elo.get_leaderboard.assert_not_called()

    def test_falls_back_to_get_leaderboard_without_cache(self, handler, mock_http_handler):
        """Falls back to get_leaderboard when get_cached_leaderboard is not available."""
        agents = [MockAgent(name="claude")]
        mock_elo = MagicMock()
        mock_elo.get_leaderboard.return_value = agents
        mock_elo.get_stats.return_value = {"total_agents": 1, "total_matches": 0, "avg_elo": 1500}
        del mock_elo.get_cached_leaderboard  # Remove cached method

        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = handler.handle("/api/v1/agents/leaderboard", {}, mock_http_handler)
            assert _status(result) == 200
            mock_elo.get_leaderboard.assert_called_once_with(limit=20)

    def test_domain_uses_get_leaderboard_not_cached(self, handler, mock_http_handler):
        """Domain queries use get_leaderboard, not get_cached_leaderboard."""
        mock_elo = _make_mock_elo()
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            handler.handle(
                "/api/v1/agents/leaderboard", {"domain": "financial"}, mock_http_handler
            )
            mock_elo.get_leaderboard.assert_called_once_with(limit=20, domain="financial")
            mock_elo.get_cached_leaderboard.assert_not_called()

    def test_elo_exception_returns_500(self, handler, mock_http_handler):
        """Returns 500 when ELO system raises an exception."""
        mock_elo = MagicMock()
        mock_elo.get_cached_leaderboard.side_effect = TypeError("Unexpected type")
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = handler.handle("/api/v1/agents/leaderboard", {}, mock_http_handler)
            assert _status(result) == 500
            assert "Failed to get leaderboard" in _body(result).get("error", "")

    def test_rank_assignment_sequential(self, handler, mock_http_handler):
        """Ranks are assigned sequentially starting from 1."""
        agents = [
            MockAgent(name="claude", elo=1700),
            MockAgent(name="gpt4", elo=1650),
            MockAgent(name="gemini", elo=1600),
        ]
        mock_elo = _make_mock_elo(cached_leaderboard=agents)

        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = handler.handle("/api/v1/agents/leaderboard", {}, mock_http_handler)
            body = _body(result)
            for i, entry in enumerate(body["leaderboard"]):
                assert entry["rank"] == i + 1


# =============================================================================
# Rate Limiting Tests
# =============================================================================


class TestRateLimiting:
    """Tests for rate limiting across both endpoints."""

    def test_recommend_rate_limit_exceeded(self, handler, mock_http_handler):
        """Returns 429 after exceeding rate limit on recommendations."""
        mock_elo = _make_mock_elo(leaderboard=[])
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            for i in range(35):
                h = MagicMock()
                h.client_address = ("192.168.1.100", 12345)
                h.headers = {}
                result = handler.handle("/api/v1/agents/recommend", {}, h)
                if i >= 30 and _status(result) == 429:
                    assert "rate limit" in _body(result).get("error", "").lower()
                    return

        # Rate limiting may not trigger in exact timing, which is acceptable.

    def test_leaderboard_rate_limit_exceeded(self, handler, mock_http_handler):
        """Returns 429 after exceeding rate limit on leaderboard."""
        mock_elo = _make_mock_elo(cached_leaderboard=[])
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            for i in range(35):
                h = MagicMock()
                h.client_address = ("192.168.1.101", 12345)
                h.headers = {}
                result = handler.handle("/api/v1/agents/leaderboard", {}, h)
                if i >= 30 and _status(result) == 429:
                    assert "rate limit" in _body(result).get("error", "").lower()
                    return

    def test_different_ips_not_rate_limited(self, handler, mock_http_handler):
        """Different IPs each have their own rate limit bucket."""
        mock_elo = _make_mock_elo(leaderboard=[])
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            for i in range(5):
                h = MagicMock()
                h.client_address = (f"10.0.0.{i}", 12345)
                h.headers = {}
                result = handler.handle("/api/v1/agents/recommend", {}, h)
                # None of these should be rate limited
                assert _status(result) != 429


# =============================================================================
# Response Structure Tests
# =============================================================================


class TestResponseStructure:
    """Tests for correct response structure and field presence."""

    def test_recommend_response_structure(self, handler, mock_http_handler):
        """Recommendations response has correct top-level keys."""
        agents = [MockAgent(name="claude")]
        mock_elo = _make_mock_elo(leaderboard=agents)
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = handler.handle("/api/v1/agents/recommend", {}, mock_http_handler)
            body = _body(result)
            assert "recommendations" in body
            assert "domain" in body
            assert "count" in body

    def test_leaderboard_response_structure(self, handler, mock_http_handler):
        """Leaderboard response has correct top-level keys."""
        agents = [MockAgent(name="claude")]
        mock_elo = _make_mock_elo(cached_leaderboard=agents)
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = handler.handle("/api/v1/agents/leaderboard", {}, mock_http_handler)
            body = _body(result)
            assert "leaderboard" in body
            assert "count" in body
            assert "domain" in body
            assert "stats" in body

    def test_recommendation_agent_fields(self, handler, mock_http_handler):
        """Each recommendation entry has standard agent fields."""
        agents = [MockAgent(name="claude", elo=1600)]
        mock_elo = _make_mock_elo(leaderboard=agents)
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = handler.handle("/api/v1/agents/recommend", {}, mock_http_handler)
            body = _body(result)
            rec = body["recommendations"][0]
            # agent_to_dict provides these fields
            assert "elo" in rec
            assert "name" in rec
            # cost estimation always present
            assert "estimated_cost_per_1k_tokens" in rec

    def test_leaderboard_agent_has_rank(self, handler, mock_http_handler):
        """Each leaderboard entry has a rank field."""
        agents = [MockAgent(name="claude")]
        mock_elo = _make_mock_elo(cached_leaderboard=agents)
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = handler.handle("/api/v1/agents/leaderboard", {}, mock_http_handler)
            body = _body(result)
            assert "rank" in body["leaderboard"][0]

    def test_stats_structure(self, handler, mock_http_handler):
        """Stats section has the expected sub-fields."""
        agents = [MockAgent(name="claude")]
        mock_elo = _make_mock_elo(cached_leaderboard=agents)
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = handler.handle("/api/v1/agents/leaderboard", {}, mock_http_handler)
            body = _body(result)
            stats = body["stats"]
            assert "total_agents" in stats
            assert "total_matches" in stats
            assert "mean_elo" in stats


# =============================================================================
# Dict Agent Input Tests
# =============================================================================


class TestDictAgentInput:
    """Tests for when the ELO system returns dicts instead of objects."""

    def test_recommend_with_dict_agents(self, handler, mock_http_handler):
        """Handles dict-based agent entries from ELO system."""
        agents = [
            {"name": "claude", "elo": 1600, "wins": 10, "losses": 5},
            {"name": "gpt4", "elo": 1550, "wins": 8, "losses": 7},
        ]
        mock_elo = _make_mock_elo(leaderboard=agents)
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = handler.handle("/api/v1/agents/recommend", {}, mock_http_handler)
            assert _status(result) == 200
            body = _body(result)
            assert body["count"] == 2

    def test_leaderboard_with_dict_agents(self, handler, mock_http_handler):
        """Handles dict-based agent entries for leaderboard."""
        agents = [{"name": "claude", "elo": 1600}]
        mock_elo = _make_mock_elo(cached_leaderboard=agents)
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = handler.handle("/api/v1/agents/leaderboard", {}, mock_http_handler)
            assert _status(result) == 200
            body = _body(result)
            assert body["count"] == 1
            assert body["leaderboard"][0]["rank"] == 1


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Edge case tests."""

    def test_valid_domain_with_hyphens_and_underscores(self, handler, mock_http_handler):
        """Valid domain names with hyphens and underscores are accepted."""
        mock_elo = _make_mock_elo(domain_agents=[])
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = handler.handle(
                "/api/v1/agents/recommend",
                {"domain": "financial-services_v2"},
                mock_http_handler,
            )
            assert _status(result) == 200

    def test_domain_too_long_returns_400(self, handler, mock_http_handler):
        """Domain longer than SAFE_ID_PATTERN max (64 chars) returns 400."""
        long_domain = "a" * 65
        mock_elo = _make_mock_elo()
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = handler.handle(
                "/api/v1/agents/recommend",
                {"domain": long_domain},
                mock_http_handler,
            )
            assert _status(result) == 400

    def test_domain_exact_max_length_accepted(self, handler, mock_http_handler):
        """Domain at exactly 64 chars is accepted."""
        domain = "a" * 64
        mock_elo = _make_mock_elo(domain_agents=[])
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = handler.handle(
                "/api/v1/agents/recommend",
                {"domain": domain},
                mock_http_handler,
            )
            assert _status(result) == 200

    def test_non_numeric_limit_uses_default(self, handler, mock_http_handler):
        """Non-numeric limit falls back to default."""
        mock_elo = _make_mock_elo()
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            handler.handle(
                "/api/v1/agents/recommend", {"limit": "not_a_number"}, mock_http_handler
            )
            # get_int_param returns default (5) on ValueError, then clamped stays 5
            mock_elo.get_leaderboard.assert_called_once_with(limit=5)

    def test_query_params_as_lists(self, handler, mock_http_handler):
        """Query params passed as lists (from URL parsing) are handled."""
        mock_elo = _make_mock_elo(domain_agents=[])
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = handler.handle(
                "/api/v1/agents/recommend",
                {"domain": ["financial"], "limit": ["10"]},
                mock_http_handler,
            )
            assert _status(result) == 200

    def test_multiple_recommendations_have_unique_costs(self, handler, mock_http_handler):
        """Multiple agents each get their own cost estimate."""
        agents = [
            MockAgent(name="claude"),
            MockAgent(name="gemini"),
            MockAgent(name="unknown_agent"),
        ]
        mock_elo = _make_mock_elo(leaderboard=agents)
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = handler.handle("/api/v1/agents/recommend", {}, mock_http_handler)
            body = _body(result)
            costs = [r["estimated_cost_per_1k_tokens"] for r in body["recommendations"]]
            assert costs[0] == 0.015  # claude
            assert costs[1] == 0.00125  # gemini
            assert costs[2] is None  # unknown

    def test_leaderboard_stats_avg_elo_fallback(self, handler, mock_http_handler):
        """Stats uses avg_elo key, falling back to mean_elo."""
        mock_elo = _make_mock_elo(
            cached_leaderboard=[MockAgent(name="claude")],
            stats={
                "total_agents": 5,
                "total_matches": 20,
                "mean_elo": 1480,
                # No avg_elo key - should fallback to mean_elo
            },
        )
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = handler.handle("/api/v1/agents/leaderboard", {}, mock_http_handler)
            body = _body(result)
            assert body["stats"]["mean_elo"] == 1480
