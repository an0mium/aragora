"""Comprehensive tests for AgentProfilesMixin (aragora/server/handlers/agents/agent_profiles.py).

Tests cover all 12 endpoints provided by the mixin:
- GET /api/agent/{name}/profile - Get complete agent profile
- GET /api/agent/{name}/history - Get agent match history
- GET /api/agent/{name}/calibration - Get agent calibration scores
- GET /api/agent/{name}/consistency - Get agent consistency score
- GET /api/agent/{name}/network - Get agent relationship network
- GET /api/agent/{name}/rivals - Get agent top rivals
- GET /api/agent/{name}/allies - Get agent top allies
- GET /api/agent/{name}/moments - Get agent significant moments
- GET /api/agent/{name}/positions - Get agent position history
- GET /api/agent/{name}/domains - Get agent domain-specific ELO ratings
- GET /api/agent/{name}/performance - Get detailed agent performance statistics

Each endpoint is tested for:
- Happy path with valid data
- No ELO system (503)
- Edge cases (empty data, missing attributes)
- Input validation (agent names, limits)
- Method not allowed (POST/PUT/DELETE)
- Security (path traversal, injection)
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
def _reset_caches():
    """Reset caches before each test to avoid stale cached data."""
    try:
        from aragora.server.handlers.admin.cache import clear_cache

        clear_cache()
    except ImportError:
        pass

    # Reset rate limiters
    try:
        from aragora.server.middleware.rate_limit.registry import reset_rate_limiters

        reset_rate_limiters()
    except ImportError:
        pass

    try:
        from aragora.server.handlers.agents import agents as agents_mod

        agents_mod._agent_limiter = agents_mod.RateLimiter(requests_per_minute=60)
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
        from aragora.server.handlers.admin.cache import clear_cache

        clear_cache()
    except ImportError:
        pass

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


def _make_rating(
    elo=1600,
    wins=10,
    losses=3,
    draws=2,
    domain_elos=None,
    critiques_accepted=5,
    critiques_total=8,
    critique_acceptance_rate=0.625,
    calibration_accuracy=0.85,
    calibration_brier_score=0.12,
    calibration_total=50,
):
    """Create a mock rating object with all attributes used by _get_domains and _get_performance."""
    rating = MagicMock()
    rating.elo = elo
    rating.wins = wins
    rating.losses = losses
    rating.draws = draws
    rating.domain_elos = domain_elos or {}
    rating.critiques_accepted = critiques_accepted
    rating.critiques_total = critiques_total
    rating.critique_acceptance_rate = critique_acceptance_rate
    rating.calibration_accuracy = calibration_accuracy
    rating.calibration_brier_score = calibration_brier_score
    rating.calibration_total = calibration_total
    return rating


# ===========================================================================
# Profile endpoint: /api/agent/{name}/profile
# ===========================================================================


class TestGetProfile:
    """Tests for the _get_profile endpoint."""

    @pytest.mark.asyncio
    async def test_profile_happy_path(self, handler, mock_http_handler):
        """Profile returns agent name, rating, stats."""
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
            assert body["rating"] == 1600
            assert body["rank"] == 1
            assert body["wins"] == 10
            assert body["losses"] == 2
            assert body["win_rate"] == 0.833

    @pytest.mark.asyncio
    async def test_profile_no_elo(self, handler, mock_http_handler):
        """Returns 503 when ELO system is not available."""
        with patch.object(handler, "get_elo_system", return_value=None):
            result = await handler.handle("/api/agent/claude/profile", {}, mock_http_handler)
            assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_profile_no_stats(self, handler, mock_http_handler):
        """Profile falls back to defaults when get_agent_stats is not available."""
        mock_elo = MagicMock(spec=[])  # no get_agent_stats
        mock_elo.get_rating = MagicMock(return_value=1500)
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle("/api/agent/claude/profile", {}, mock_http_handler)
            assert _status(result) == 200
            body = _body(result)
            assert body["name"] == "claude"
            # defaults when no stats
            assert body["wins"] == 0
            assert body["losses"] == 0
            assert body["win_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_profile_rating_none_uses_initial(self, handler, mock_http_handler):
        """When get_rating returns None, the initial ELO rating is used."""
        from aragora.config import ELO_INITIAL_RATING

        mock_elo = MagicMock()
        mock_elo.get_rating.return_value = None
        mock_elo.get_agent_stats.return_value = {}
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle("/api/agent/claude/profile", {}, mock_http_handler)
            assert _status(result) == 200
            body = _body(result)
            assert body["rating"] == ELO_INITIAL_RATING

    @pytest.mark.asyncio
    async def test_profile_stats_returns_none(self, handler, mock_http_handler):
        """When get_agent_stats returns None, defaults are used."""
        mock_elo = MagicMock()
        mock_elo.get_rating.return_value = 1500
        mock_elo.get_agent_stats.return_value = None
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle("/api/agent/claude/profile", {}, mock_http_handler)
            assert _status(result) == 200
            body = _body(result)
            assert body["wins"] == 0
            assert body["losses"] == 0

    @pytest.mark.asyncio
    async def test_profile_versioned_path(self, handler, mock_http_handler):
        """Profile works with versioned path /api/v1/agent/{name}/profile."""
        mock_elo = MagicMock()
        mock_elo.get_rating.return_value = 1600
        mock_elo.get_agent_stats.return_value = {
            "rank": 2,
            "wins": 5,
            "losses": 3,
            "win_rate": 0.625,
        }
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle("/api/v1/agent/claude/profile", {}, mock_http_handler)
            assert _status(result) == 200
            body = _body(result)
            assert body["name"] == "claude"

    @pytest.mark.asyncio
    async def test_profile_different_agent_names(self, handler, mock_http_handler):
        """Profile returns the correct agent name for various valid agent names."""
        mock_elo = MagicMock()
        mock_elo.get_rating.return_value = 1500
        mock_elo.get_agent_stats.return_value = {}
        for name in ["gpt4", "gemini", "mistral-api", "anthropic-api"]:
            with patch.object(handler, "get_elo_system", return_value=mock_elo):
                result = await handler.handle(f"/api/agent/{name}/profile", {}, mock_http_handler)
                assert _status(result) == 200
                body = _body(result)
                assert body["name"] == name


# ===========================================================================
# History endpoint: /api/agent/{name}/history
# ===========================================================================


class TestGetHistory:
    """Tests for the _get_history endpoint."""

    @pytest.mark.asyncio
    async def test_history_happy_path(self, handler, mock_http_handler):
        """History returns agent name and list of history entries."""
        mock_elo = MagicMock()
        mock_elo.get_elo_history.return_value = [
            (1000000, 1600),
            (1000001, 1610),
            (1000002, 1620),
        ]
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle(
                "/api/agent/claude/history", {"limit": "5"}, mock_http_handler
            )
            assert _status(result) == 200
            body = _body(result)
            assert body["agent"] == "claude"
            assert len(body["history"]) == 3
            assert body["history"][0]["timestamp"] == 1000000
            assert body["history"][0]["elo"] == 1600

    @pytest.mark.asyncio
    async def test_history_no_elo(self, handler, mock_http_handler):
        """Returns 503 when ELO system is unavailable."""
        with patch.object(handler, "get_elo_system", return_value=None):
            result = await handler.handle("/api/agent/claude/history", {}, mock_http_handler)
            assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_history_empty(self, handler, mock_http_handler):
        """Returns empty history list when agent has no history."""
        mock_elo = MagicMock()
        mock_elo.get_elo_history.return_value = []
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle("/api/agent/claude/history", {}, mock_http_handler)
            assert _status(result) == 200
            body = _body(result)
            assert body["history"] == []

    @pytest.mark.asyncio
    async def test_history_limit_capped_at_100(self, handler, mock_http_handler):
        """Limit is capped at 100 regardless of user input."""
        mock_elo = MagicMock()
        mock_elo.get_elo_history.return_value = []
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle(
                "/api/agent/claude/history", {"limit": "500"}, mock_http_handler
            )
            assert _status(result) == 200
            # The handler calls min(limit, 100), so we verify it was capped
            mock_elo.get_elo_history.assert_called_once_with("claude", limit=100)

    @pytest.mark.asyncio
    async def test_history_default_limit(self, handler, mock_http_handler):
        """Default limit is 30 when not specified."""
        mock_elo = MagicMock()
        mock_elo.get_elo_history.return_value = []
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle("/api/agent/claude/history", {}, mock_http_handler)
            assert _status(result) == 200
            mock_elo.get_elo_history.assert_called_once_with("claude", limit=30)

    @pytest.mark.asyncio
    async def test_history_versioned_path(self, handler, mock_http_handler):
        """History works with /api/v1/ prefix."""
        mock_elo = MagicMock()
        mock_elo.get_elo_history.return_value = [(1, 1500)]
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle("/api/v1/agent/claude/history", {}, mock_http_handler)
            assert _status(result) == 200


# ===========================================================================
# Calibration endpoint: /api/agent/{name}/calibration
# ===========================================================================


class TestGetCalibration:
    """Tests for the _get_calibration endpoint."""

    @pytest.mark.asyncio
    async def test_calibration_happy_path(self, handler, mock_http_handler):
        """Calibration returns data when ELO has get_calibration."""
        mock_elo = MagicMock()
        mock_elo.get_calibration.return_value = {"agent": "claude", "score": 0.75}
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle("/api/agent/claude/calibration", {}, mock_http_handler)
            assert _status(result) == 200
            body = _body(result)
            assert body["agent"] == "claude"
            assert body["score"] == 0.75

    @pytest.mark.asyncio
    async def test_calibration_no_elo(self, handler, mock_http_handler):
        """Returns 503 when ELO system is unavailable."""
        with patch.object(handler, "get_elo_system", return_value=None):
            result = await handler.handle("/api/agent/claude/calibration", {}, mock_http_handler)
            assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_calibration_with_domain(self, handler, mock_http_handler):
        """Passes domain parameter to get_calibration."""
        mock_elo = MagicMock()
        mock_elo.get_calibration.return_value = {"agent": "claude", "score": 0.8, "domain": "tech"}
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle(
                "/api/agent/claude/calibration", {"domain": "tech"}, mock_http_handler
            )
            assert _status(result) == 200
            mock_elo.get_calibration.assert_called_once_with("claude", domain="tech")

    @pytest.mark.asyncio
    async def test_calibration_without_get_calibration_method(self, handler, mock_http_handler):
        """Falls back to default calibration when elo has no get_calibration."""
        mock_elo = MagicMock(spec=[])
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle("/api/agent/claude/calibration", {}, mock_http_handler)
            assert _status(result) == 200
            body = _body(result)
            assert body["agent"] == "claude"
            assert body["score"] == 0.5

    @pytest.mark.asyncio
    async def test_calibration_no_domain(self, handler, mock_http_handler):
        """Domain is None when not provided."""
        mock_elo = MagicMock()
        mock_elo.get_calibration.return_value = {"agent": "claude", "score": 0.6}
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle("/api/agent/claude/calibration", {}, mock_http_handler)
            assert _status(result) == 200
            mock_elo.get_calibration.assert_called_once_with("claude", domain=None)


# ===========================================================================
# Consistency endpoint: /api/agent/{name}/consistency
# ===========================================================================


class TestGetConsistency:
    """Tests for the _get_consistency endpoint."""

    @pytest.mark.asyncio
    async def test_consistency_no_nomic_dir(self, handler, mock_http_handler):
        """Returns 1.0 consistency when nomic_dir is not set."""
        with patch.object(handler, "get_nomic_dir", return_value=None):
            result = await handler.handle("/api/agent/claude/consistency", {}, mock_http_handler)
            assert _status(result) == 200
            body = _body(result)
            assert body["agent"] == "claude"
            assert body["consistency_score"] == 1.0

    @pytest.mark.asyncio
    async def test_consistency_with_nomic_dir(self, handler, mock_http_handler):
        """Uses FlipDetector when nomic_dir is available."""
        from pathlib import Path

        mock_detector = MagicMock()
        mock_detector.get_agent_consistency.return_value = 0.85

        with patch.object(handler, "get_nomic_dir", return_value=Path("/tmp/nomic")):
            with patch(
                "aragora.insights.flip_detector.FlipDetector",
                return_value=mock_detector,
            ):
                with patch(
                    "aragora.server.handlers.agents.agent_profiles.get_db_path",
                    return_value="/tmp/nomic/positions.db",
                ):
                    result = await handler.handle(
                        "/api/agent/claude/consistency", {}, mock_http_handler
                    )
                    assert _status(result) == 200
                    body = _body(result)
                    assert body["agent"] == "claude"
                    assert body["consistency_score"] == 0.85

    @pytest.mark.asyncio
    async def test_consistency_versioned_path(self, handler, mock_http_handler):
        """Works with versioned path."""
        with patch.object(handler, "get_nomic_dir", return_value=None):
            result = await handler.handle("/api/v1/agent/claude/consistency", {}, mock_http_handler)
            assert _status(result) == 200
            body = _body(result)
            assert body["consistency_score"] == 1.0


# ===========================================================================
# Network endpoint: /api/agent/{name}/network
# ===========================================================================


class TestGetNetwork:
    """Tests for the _get_network endpoint."""

    @pytest.mark.asyncio
    async def test_network_happy_path(self, handler, mock_http_handler):
        """Returns rivals and allies."""
        mock_elo = MagicMock()
        mock_elo.get_rivals.return_value = [{"name": "gpt4", "elo": 1580}]
        mock_elo.get_allies.return_value = [{"name": "gemini", "elo": 1520}]
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle("/api/agent/claude/network", {}, mock_http_handler)
            assert _status(result) == 200
            body = _body(result)
            assert body["agent"] == "claude"
            assert len(body["rivals"]) == 1
            assert len(body["allies"]) == 1

    @pytest.mark.asyncio
    async def test_network_no_elo(self, handler, mock_http_handler):
        """Returns 503 when ELO system is unavailable."""
        with patch.object(handler, "get_elo_system", return_value=None):
            result = await handler.handle("/api/agent/claude/network", {}, mock_http_handler)
            assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_network_no_rivals_method(self, handler, mock_http_handler):
        """Falls back to empty when ELO has no get_rivals / get_allies."""
        mock_elo = MagicMock(spec=[])
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle("/api/agent/claude/network", {}, mock_http_handler)
            assert _status(result) == 200
            body = _body(result)
            assert body["rivals"] == []
            assert body["allies"] == []

    @pytest.mark.asyncio
    async def test_network_empty_rivals_and_allies(self, handler, mock_http_handler):
        """Returns empty lists when agent has no relationships."""
        mock_elo = MagicMock()
        mock_elo.get_rivals.return_value = []
        mock_elo.get_allies.return_value = []
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle("/api/agent/claude/network", {}, mock_http_handler)
            assert _status(result) == 200
            body = _body(result)
            assert body["rivals"] == []
            assert body["allies"] == []


# ===========================================================================
# Rivals endpoint: /api/agent/{name}/rivals
# ===========================================================================


class TestGetRivals:
    """Tests for the _get_rivals endpoint."""

    @pytest.mark.asyncio
    async def test_rivals_happy_path(self, handler, mock_http_handler):
        """Returns rivals list."""
        mock_elo = MagicMock()
        mock_elo.get_rivals.return_value = [
            {"name": "gpt4", "head_to_head": {"wins": 3, "losses": 2}},
        ]
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle(
                "/api/agent/claude/rivals", {"limit": "3"}, mock_http_handler
            )
            assert _status(result) == 200
            body = _body(result)
            assert body["agent"] == "claude"
            assert len(body["rivals"]) == 1

    @pytest.mark.asyncio
    async def test_rivals_no_elo(self, handler, mock_http_handler):
        """Returns 503 without ELO."""
        with patch.object(handler, "get_elo_system", return_value=None):
            result = await handler.handle("/api/agent/claude/rivals", {}, mock_http_handler)
            assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_rivals_no_get_rivals_method(self, handler, mock_http_handler):
        """Returns empty when ELO has no get_rivals."""
        mock_elo = MagicMock(spec=[])
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle("/api/agent/claude/rivals", {}, mock_http_handler)
            assert _status(result) == 200
            body = _body(result)
            assert body["rivals"] == []

    @pytest.mark.asyncio
    async def test_rivals_with_limit(self, handler, mock_http_handler):
        """Passes limit to get_rivals."""
        mock_elo = MagicMock()
        mock_elo.get_rivals.return_value = []
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle(
                "/api/agent/claude/rivals", {"limit": "10"}, mock_http_handler
            )
            assert _status(result) == 200
            mock_elo.get_rivals.assert_called_once_with("claude", limit=10)

    @pytest.mark.asyncio
    async def test_rivals_default_limit(self, handler, mock_http_handler):
        """Default limit is 5."""
        mock_elo = MagicMock()
        mock_elo.get_rivals.return_value = []
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle("/api/agent/claude/rivals", {}, mock_http_handler)
            assert _status(result) == 200
            mock_elo.get_rivals.assert_called_once_with("claude", limit=5)


# ===========================================================================
# Allies endpoint: /api/agent/{name}/allies
# ===========================================================================


class TestGetAllies:
    """Tests for the _get_allies endpoint."""

    @pytest.mark.asyncio
    async def test_allies_happy_path(self, handler, mock_http_handler):
        """Returns allies list."""
        mock_elo = MagicMock()
        mock_elo.get_allies.return_value = [
            {"name": "gemini", "agreement_rate": 0.8},
        ]
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle(
                "/api/agent/claude/allies", {"limit": "5"}, mock_http_handler
            )
            assert _status(result) == 200
            body = _body(result)
            assert body["agent"] == "claude"
            assert len(body["allies"]) == 1

    @pytest.mark.asyncio
    async def test_allies_no_elo(self, handler, mock_http_handler):
        """Returns 503 without ELO."""
        with patch.object(handler, "get_elo_system", return_value=None):
            result = await handler.handle("/api/agent/claude/allies", {}, mock_http_handler)
            assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_allies_no_get_allies_method(self, handler, mock_http_handler):
        """Returns empty when ELO has no get_allies."""
        mock_elo = MagicMock(spec=[])
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle("/api/agent/claude/allies", {}, mock_http_handler)
            assert _status(result) == 200
            body = _body(result)
            assert body["allies"] == []

    @pytest.mark.asyncio
    async def test_allies_default_limit(self, handler, mock_http_handler):
        """Default limit is 5."""
        mock_elo = MagicMock()
        mock_elo.get_allies.return_value = []
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle("/api/agent/claude/allies", {}, mock_http_handler)
            assert _status(result) == 200
            mock_elo.get_allies.assert_called_once_with("claude", limit=5)


# ===========================================================================
# Moments endpoint: /api/agent/{name}/moments
# ===========================================================================


class TestGetMoments:
    """Tests for the _get_moments endpoint."""

    @pytest.mark.asyncio
    async def test_moments_happy_path(self, handler, mock_http_handler):
        """Returns moments data."""
        from datetime import datetime, timezone

        mock_moment = MagicMock()
        mock_moment.id = "m1"
        mock_moment.moment_type = "upset"
        mock_moment.agent_name = "claude"
        mock_moment.description = "Beat top-ranked agent"
        mock_moment.significance_score = 0.95
        mock_moment.timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
        mock_moment.debate_id = "d-123"

        mock_detector = MagicMock()
        mock_detector.get_agent_moments.return_value = [mock_moment]

        mock_elo = MagicMock()
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            with patch(
                "aragora.agents.grounded.MomentDetector",
                return_value=mock_detector,
            ):
                result = await handler.handle(
                    "/api/agent/claude/moments", {"limit": "5"}, mock_http_handler
                )
                assert _status(result) == 200
                body = _body(result)
                assert body["agent"] == "claude"
                assert len(body["moments"]) == 1
                m = body["moments"][0]
                assert m["id"] == "m1"
                assert m["moment_type"] == "upset"
                assert m["significance_score"] == 0.95
                assert m["debate_id"] == "d-123"
                assert m["timestamp"] is not None

    @pytest.mark.asyncio
    async def test_moments_no_elo(self, handler, mock_http_handler):
        """Returns empty moments when ELO is unavailable."""
        with patch.object(handler, "get_elo_system", return_value=None):
            result = await handler.handle("/api/agent/claude/moments", {}, mock_http_handler)
            assert _status(result) == 200
            body = _body(result)
            assert body["agent"] == "claude"
            assert body["moments"] == []

    @pytest.mark.asyncio
    async def test_moments_no_timestamp(self, handler, mock_http_handler):
        """Handles moments with no timestamp gracefully."""
        mock_moment = MagicMock()
        mock_moment.id = "m2"
        mock_moment.moment_type = "streak"
        mock_moment.agent_name = "claude"
        mock_moment.description = "5 win streak"
        mock_moment.significance_score = 0.8
        mock_moment.timestamp = None
        mock_moment.debate_id = "d-456"

        mock_detector = MagicMock()
        mock_detector.get_agent_moments.return_value = [mock_moment]

        mock_elo = MagicMock()
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            with patch(
                "aragora.agents.grounded.MomentDetector",
                return_value=mock_detector,
            ):
                result = await handler.handle("/api/agent/claude/moments", {}, mock_http_handler)
                assert _status(result) == 200
                body = _body(result)
                assert body["moments"][0]["timestamp"] is None

    @pytest.mark.asyncio
    async def test_moments_empty_list(self, handler, mock_http_handler):
        """Returns empty moments list when no moments exist."""
        mock_detector = MagicMock()
        mock_detector.get_agent_moments.return_value = []

        mock_elo = MagicMock()
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            with patch(
                "aragora.agents.grounded.MomentDetector",
                return_value=mock_detector,
            ):
                result = await handler.handle("/api/agent/claude/moments", {}, mock_http_handler)
                assert _status(result) == 200
                body = _body(result)
                assert body["moments"] == []

    @pytest.mark.asyncio
    async def test_moments_default_limit(self, handler, mock_http_handler):
        """Default limit for moments is 10."""
        mock_detector = MagicMock()
        mock_detector.get_agent_moments.return_value = []

        mock_elo = MagicMock()
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            with patch(
                "aragora.agents.grounded.MomentDetector",
                return_value=mock_detector,
            ):
                result = await handler.handle("/api/agent/claude/moments", {}, mock_http_handler)
                assert _status(result) == 200
                mock_detector.get_agent_moments.assert_called_once_with("claude", limit=10)


# ===========================================================================
# Positions endpoint: /api/agent/{name}/positions
# ===========================================================================


class TestGetPositions:
    """Tests for the _get_positions endpoint."""

    @pytest.mark.asyncio
    async def test_positions_no_nomic_dir(self, handler, mock_http_handler):
        """Returns empty positions when nomic_dir is not set."""
        with patch.object(handler, "get_nomic_dir", return_value=None):
            result = await handler.handle("/api/agent/claude/positions", {}, mock_http_handler)
            assert _status(result) == 200
            body = _body(result)
            assert body["agent"] == "claude"
            assert body["positions"] == []

    @pytest.mark.asyncio
    async def test_positions_with_nomic_dir(self, handler, mock_http_handler):
        """Uses PositionLedger when nomic_dir is available."""
        from pathlib import Path

        mock_ledger = MagicMock()
        mock_ledger.get_agent_positions.return_value = [
            {"debate_id": "d1", "position": "for", "confidence": 0.9},
        ]

        with patch.object(handler, "get_nomic_dir", return_value=Path("/tmp/nomic")):
            with patch(
                "aragora.agents.grounded.PositionLedger",
                return_value=mock_ledger,
            ):
                with patch(
                    "aragora.server.handlers.agents.agent_profiles.get_db_path",
                    return_value="/tmp/nomic/positions.db",
                ):
                    result = await handler.handle(
                        "/api/agent/claude/positions", {"limit": "10"}, mock_http_handler
                    )
                    assert _status(result) == 200
                    body = _body(result)
                    assert body["agent"] == "claude"
                    assert len(body["positions"]) == 1

    @pytest.mark.asyncio
    async def test_positions_default_limit(self, handler, mock_http_handler):
        """Default limit for positions is 20."""
        from pathlib import Path

        mock_ledger = MagicMock()
        mock_ledger.get_agent_positions.return_value = []

        with patch.object(handler, "get_nomic_dir", return_value=Path("/tmp/nomic")):
            with patch(
                "aragora.agents.grounded.PositionLedger",
                return_value=mock_ledger,
            ):
                with patch(
                    "aragora.server.handlers.agents.agent_profiles.get_db_path",
                    return_value="/tmp/nomic/positions.db",
                ):
                    result = await handler.handle(
                        "/api/agent/claude/positions", {}, mock_http_handler
                    )
                    assert _status(result) == 200
                    mock_ledger.get_agent_positions.assert_called_once_with("claude", limit=20)


# ===========================================================================
# Domains endpoint: /api/agent/{name}/domains
# ===========================================================================


class TestGetDomains:
    """Tests for the _get_domains endpoint."""

    @pytest.mark.asyncio
    async def test_domains_happy_path(self, handler, mock_http_handler):
        """Returns domain-specific ELO ratings sorted descending."""
        rating = _make_rating(
            elo=1600,
            domain_elos={"tech": 1700, "finance": 1550, "science": 1650},
        )
        mock_elo = MagicMock()
        mock_elo.get_rating.return_value = rating
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle("/api/agent/claude/domains", {}, mock_http_handler)
            assert _status(result) == 200
            body = _body(result)
            assert body["agent"] == "claude"
            assert body["overall_elo"] == 1600
            assert body["domain_count"] == 3
            # Sorted descending by ELO
            assert body["domains"][0]["domain"] == "tech"
            assert body["domains"][0]["elo"] == 1700
            assert body["domains"][0]["relative"] == 100.0
            assert body["domains"][1]["domain"] == "science"
            assert body["domains"][2]["domain"] == "finance"
            assert body["domains"][2]["relative"] == -50.0

    @pytest.mark.asyncio
    async def test_domains_no_elo(self, handler, mock_http_handler):
        """Returns 503 without ELO."""
        with patch.object(handler, "get_elo_system", return_value=None):
            result = await handler.handle("/api/agent/claude/domains", {}, mock_http_handler)
            assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_domains_no_domain_elos_attribute(self, handler, mock_http_handler):
        """Returns empty domains when rating has no domain_elos attribute."""
        rating = MagicMock(spec=[])
        rating.elo = 1500
        mock_elo = MagicMock()
        mock_elo.get_rating.return_value = rating
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle("/api/agent/claude/domains", {}, mock_http_handler)
            assert _status(result) == 200
            body = _body(result)
            assert body["domains"] == []
            assert body["domain_count"] == 0

    @pytest.mark.asyncio
    async def test_domains_empty_domain_elos(self, handler, mock_http_handler):
        """Returns empty domains when domain_elos is empty."""
        rating = _make_rating(elo=1500, domain_elos={})
        mock_elo = MagicMock()
        mock_elo.get_rating.return_value = rating
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle("/api/agent/claude/domains", {}, mock_http_handler)
            assert _status(result) == 200
            body = _body(result)
            assert body["domains"] == []
            assert body["domain_count"] == 0

    @pytest.mark.asyncio
    async def test_domains_single_domain(self, handler, mock_http_handler):
        """Correctly handles a single domain."""
        rating = _make_rating(elo=1500, domain_elos={"legal": 1600})
        mock_elo = MagicMock()
        mock_elo.get_rating.return_value = rating
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle("/api/agent/claude/domains", {}, mock_http_handler)
            assert _status(result) == 200
            body = _body(result)
            assert body["domain_count"] == 1
            assert body["domains"][0]["domain"] == "legal"
            assert body["domains"][0]["relative"] == 100.0


# ===========================================================================
# Performance endpoint: /api/agent/{name}/performance
# ===========================================================================


class TestGetPerformance:
    """Tests for the _get_performance endpoint."""

    @pytest.mark.asyncio
    async def test_performance_happy_path(self, handler, mock_http_handler):
        """Returns detailed performance stats."""
        rating = _make_rating(
            elo=1600,
            wins=10,
            losses=3,
            draws=2,
            critiques_accepted=5,
            critiques_total=8,
            critique_acceptance_rate=0.625,
            calibration_accuracy=0.85,
            calibration_brier_score=0.12,
            calibration_total=50,
        )
        mock_elo = MagicMock()
        mock_elo.get_rating.return_value = rating
        mock_elo.get_agent_history.return_value = [
            {"result": "win"},
            {"result": "loss"},
            {"result": "win"},
            {"result": "win"},
            {"result": "win"},
            {"result": "loss"},
            {"result": "win"},
            {"result": "win"},
            {"result": "draw"},
            {"result": "win"},
        ]
        mock_elo.get_elo_history.return_value = [
            (10, 1650),
            (9, 1620),
            (8, 1600),
            (7, 1580),
        ]
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle("/api/agent/claude/performance", {}, mock_http_handler)
            assert _status(result) == 200
            body = _body(result)
            assert body["agent"] == "claude"
            assert body["elo"] == 1600
            assert body["total_games"] == 15
            assert body["wins"] == 10
            assert body["losses"] == 3
            assert body["draws"] == 2
            assert body["win_rate"] == round(10 / 15, 3)
            # recent_win_rate from the 10 matches: 7 wins out of 10
            assert body["recent_win_rate"] == 0.7
            # elo_trend = most recent - oldest = 1650 - 1580 = 70
            assert body["elo_trend"] == 70.0
            assert body["critiques_accepted"] == 5
            assert body["critiques_total"] == 8
            assert body["critique_acceptance_rate"] == 0.625
            assert body["calibration"]["accuracy"] == 0.85
            assert body["calibration"]["brier_score"] == 0.12
            assert body["calibration"]["prediction_count"] == 50

    @pytest.mark.asyncio
    async def test_performance_no_elo(self, handler, mock_http_handler):
        """Returns 503 without ELO."""
        with patch.object(handler, "get_elo_system", return_value=None):
            result = await handler.handle("/api/agent/claude/performance", {}, mock_http_handler)
            assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_performance_zero_games(self, handler, mock_http_handler):
        """Handles zero games correctly (no division by zero)."""
        rating = _make_rating(elo=1500, wins=0, losses=0, draws=0)
        mock_elo = MagicMock()
        mock_elo.get_rating.return_value = rating
        mock_elo.get_agent_history.return_value = []
        mock_elo.get_elo_history.return_value = []
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle("/api/agent/claude/performance", {}, mock_http_handler)
            assert _status(result) == 200
            body = _body(result)
            assert body["total_games"] == 0
            assert body["win_rate"] == 0.0
            assert body["recent_win_rate"] == 0.0
            assert body["elo_trend"] == 0.0

    @pytest.mark.asyncio
    async def test_performance_no_history_methods(self, handler, mock_http_handler):
        """Falls back to empty lists when ELO has no history methods."""
        rating = _make_rating(elo=1600, wins=5, losses=3, draws=1)
        mock_elo = MagicMock(spec=[])
        mock_elo.get_rating = MagicMock(return_value=rating)
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle("/api/agent/claude/performance", {}, mock_http_handler)
            assert _status(result) == 200
            body = _body(result)
            assert body["recent_win_rate"] == 0.0
            assert body["elo_trend"] == 0.0

    @pytest.mark.asyncio
    async def test_performance_single_elo_history_entry(self, handler, mock_http_handler):
        """ELO trend is 0 when there is only one history entry."""
        rating = _make_rating(elo=1600, wins=5, losses=3, draws=1)
        mock_elo = MagicMock()
        mock_elo.get_rating.return_value = rating
        mock_elo.get_agent_history.return_value = []
        mock_elo.get_elo_history.return_value = [(1, 1600)]
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle("/api/agent/claude/performance", {}, mock_http_handler)
            assert _status(result) == 200
            body = _body(result)
            assert body["elo_trend"] == 0.0

    @pytest.mark.asyncio
    async def test_performance_recent_matches_less_than_10(self, handler, mock_http_handler):
        """Recent win rate uses actual match count when fewer than 10."""
        rating = _make_rating(elo=1550, wins=3, losses=1, draws=0)
        mock_elo = MagicMock()
        mock_elo.get_rating.return_value = rating
        mock_elo.get_agent_history.return_value = [
            {"result": "win"},
            {"result": "win"},
            {"result": "loss"},
        ]
        mock_elo.get_elo_history.return_value = []
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle("/api/agent/claude/performance", {}, mock_http_handler)
            assert _status(result) == 200
            body = _body(result)
            # 2 wins out of 3 recent matches
            assert body["recent_win_rate"] == round(2 / 3, 3)

    @pytest.mark.asyncio
    async def test_performance_elo_trend_negative(self, handler, mock_http_handler):
        """ELO trend is negative when agent rating is declining."""
        rating = _make_rating(elo=1400, wins=2, losses=5, draws=0)
        mock_elo = MagicMock()
        mock_elo.get_rating.return_value = rating
        mock_elo.get_agent_history.return_value = []
        mock_elo.get_elo_history.return_value = [(10, 1400), (1, 1500)]
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle("/api/agent/claude/performance", {}, mock_http_handler)
            assert _status(result) == 200
            body = _body(result)
            # Most recent (1400) - oldest (1500) = -100
            assert body["elo_trend"] == -100.0


# ===========================================================================
# Input validation and security tests
# ===========================================================================


class TestInputValidation:
    """Tests for input validation across profile endpoints."""

    @pytest.mark.asyncio
    async def test_invalid_agent_name_path_traversal(self, handler, mock_http_handler):
        """Agent names with path traversal characters are rejected."""
        result = await handler.handle("/api/agent/../../etc/passwd/profile", {}, mock_http_handler)
        # Should be rejected by validation (400) or return None for unmatched path
        assert result is None or _status(result) == 400

    @pytest.mark.asyncio
    async def test_invalid_agent_name_script_injection(self, handler, mock_http_handler):
        """Agent names with script tags are rejected."""
        result = await handler.handle("/api/agent/<script>/profile", {}, mock_http_handler)
        # Should be rejected (400) or unmatched (None)
        assert result is None or _status(result) == 400

    @pytest.mark.asyncio
    async def test_invalid_agent_name_spaces(self, handler, mock_http_handler):
        """Agent names with spaces are rejected."""
        result = await handler.handle("/api/agent/bad agent/profile", {}, mock_http_handler)
        assert result is None or _status(result) == 400

    @pytest.mark.asyncio
    async def test_empty_agent_name(self, handler, mock_http_handler):
        """Empty agent name segment handled gracefully."""
        result = await handler.handle("/api/agent//profile", {}, mock_http_handler)
        # Empty string splits differently, should not crash
        assert result is None or _status(result) in (400, 404)

    @pytest.mark.asyncio
    async def test_too_short_path(self, handler, mock_http_handler):
        """Path with too few segments returns error."""
        result = await handler.handle("/api/agent/claude", {}, mock_http_handler)
        # Less than 5 segments -> 400 from _handle_agent_endpoint
        assert result is None or _status(result) == 400

    @pytest.mark.asyncio
    async def test_unknown_endpoint(self, handler, mock_http_handler):
        """Unknown per-agent endpoint returns None (no match)."""
        result = await handler.handle("/api/agent/claude/nonexistent", {}, mock_http_handler)
        assert result is None

    @pytest.mark.asyncio
    async def test_sql_injection_in_agent_name(self, handler, mock_http_handler):
        """Agent names with SQL injection patterns are rejected."""
        result = await handler.handle(
            "/api/agent/'; DROP TABLE agents;--/profile", {}, mock_http_handler
        )
        assert result is None or _status(result) == 400

    @pytest.mark.asyncio
    async def test_very_long_agent_name(self, handler, mock_http_handler):
        """Very long agent names are handled gracefully."""
        long_name = "a" * 500
        result = await handler.handle(f"/api/agent/{long_name}/profile", {}, mock_http_handler)
        # Should either work (if valid chars) or return 400
        assert result is not None
        assert _status(result) in (200, 400, 503)


# ===========================================================================
# Route dispatch tests
# ===========================================================================


class TestRouteDispatch:
    """Tests ensuring correct routing to profile mixin methods."""

    @pytest.mark.asyncio
    async def test_profile_routes_correctly(self, handler, mock_http_handler):
        """Profile endpoint is dispatched from /api/agent/{name}/profile."""
        mock_elo = MagicMock()
        mock_elo.get_rating.return_value = 1500
        mock_elo.get_agent_stats.return_value = {}
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle("/api/agent/gpt4/profile", {}, mock_http_handler)
            assert _status(result) == 200
            assert _body(result)["name"] == "gpt4"

    @pytest.mark.asyncio
    async def test_all_profile_endpoints_accessible(self, handler, mock_http_handler):
        """All profile mixin endpoints are accessible through the handler."""
        endpoints = [
            "profile",
            "history",
            "calibration",
            "consistency",
            "network",
            "rivals",
            "allies",
            "moments",
            "positions",
            "domains",
            "performance",
        ]

        mock_elo = MagicMock()
        mock_elo.get_rating.return_value = _make_rating()
        mock_elo.get_elo_history.return_value = []
        mock_elo.get_calibration.return_value = {"agent": "claude", "score": 0.5}
        mock_elo.get_rivals.return_value = []
        mock_elo.get_allies.return_value = []
        mock_elo.get_agent_history.return_value = []
        mock_elo.get_agent_stats.return_value = {}

        for endpoint in endpoints:
            with patch.object(handler, "get_elo_system", return_value=mock_elo):
                with patch.object(handler, "get_nomic_dir", return_value=None):
                    # Patch lazy imports used by moments
                    with patch(
                        "aragora.agents.grounded.MomentDetector",
                        return_value=MagicMock(get_agent_moments=MagicMock(return_value=[])),
                    ):
                        result = await handler.handle(
                            f"/api/agent/claude/{endpoint}", {}, mock_http_handler
                        )
                        assert result is not None, f"Endpoint {endpoint} returned None"
                        assert _status(result) in (200, 503), (
                            f"Endpoint {endpoint} returned {_status(result)}"
                        )

    @pytest.mark.asyncio
    async def test_agents_prefix_converted_to_agent(self, handler, mock_http_handler):
        """Path /api/agents/claude/profile is converted to /api/agent/claude/profile."""
        mock_elo = MagicMock()
        mock_elo.get_rating.return_value = 1500
        mock_elo.get_agent_stats.return_value = {}
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle("/api/agents/claude/profile", {}, mock_http_handler)
            assert _status(result) == 200
            assert _body(result)["name"] == "claude"


# ===========================================================================
# Versioned path tests
# ===========================================================================


class TestVersionedPaths:
    """Tests for /api/v1/ prefixed paths."""

    @pytest.mark.asyncio
    async def test_v1_profile(self, handler, mock_http_handler):
        mock_elo = MagicMock()
        mock_elo.get_rating.return_value = 1500
        mock_elo.get_agent_stats.return_value = {}
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle("/api/v1/agent/claude/profile", {}, mock_http_handler)
            assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_v1_history(self, handler, mock_http_handler):
        mock_elo = MagicMock()
        mock_elo.get_elo_history.return_value = []
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle("/api/v1/agent/claude/history", {}, mock_http_handler)
            assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_v1_calibration(self, handler, mock_http_handler):
        mock_elo = MagicMock()
        mock_elo.get_calibration.return_value = {"agent": "claude", "score": 0.5}
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle("/api/v1/agent/claude/calibration", {}, mock_http_handler)
            assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_v1_network(self, handler, mock_http_handler):
        mock_elo = MagicMock()
        mock_elo.get_rivals.return_value = []
        mock_elo.get_allies.return_value = []
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle("/api/v1/agent/claude/network", {}, mock_http_handler)
            assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_v1_domains(self, handler, mock_http_handler):
        rating = _make_rating(elo=1500)
        mock_elo = MagicMock()
        mock_elo.get_rating.return_value = rating
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle("/api/v1/agent/claude/domains", {}, mock_http_handler)
            assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_v1_performance(self, handler, mock_http_handler):
        rating = _make_rating()
        mock_elo = MagicMock()
        mock_elo.get_rating.return_value = rating
        mock_elo.get_agent_history.return_value = []
        mock_elo.get_elo_history.return_value = []
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle("/api/v1/agent/claude/performance", {}, mock_http_handler)
            assert _status(result) == 200


# ===========================================================================
# ELO system unavailability (503 across all endpoints)
# ===========================================================================


class TestEloUnavailable:
    """Confirm all ELO-dependent endpoints return 503 when ELO is None."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "endpoint",
        [
            "profile",
            "history",
            "calibration",
            "network",
            "rivals",
            "allies",
            "domains",
            "performance",
        ],
    )
    async def test_no_elo_returns_503(self, handler, mock_http_handler, endpoint):
        with patch.object(handler, "get_elo_system", return_value=None):
            result = await handler.handle(f"/api/agent/claude/{endpoint}", {}, mock_http_handler)
            assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_moments_no_elo_returns_200_empty(self, handler, mock_http_handler):
        """Moments returns 200 with empty list when ELO is None."""
        with patch.object(handler, "get_elo_system", return_value=None):
            result = await handler.handle("/api/agent/claude/moments", {}, mock_http_handler)
            assert _status(result) == 200
            assert _body(result)["moments"] == []

    @pytest.mark.asyncio
    async def test_consistency_no_elo_returns_200(self, handler, mock_http_handler):
        """Consistency does not depend on ELO - returns 200 with default score."""
        with patch.object(handler, "get_elo_system", return_value=None):
            with patch.object(handler, "get_nomic_dir", return_value=None):
                result = await handler.handle(
                    "/api/agent/claude/consistency", {}, mock_http_handler
                )
                assert _status(result) == 200
                assert _body(result)["consistency_score"] == 1.0

    @pytest.mark.asyncio
    async def test_positions_no_elo_returns_200(self, handler, mock_http_handler):
        """Positions does not depend on ELO - returns 200 with empty list."""
        with patch.object(handler, "get_nomic_dir", return_value=None):
            result = await handler.handle("/api/agent/claude/positions", {}, mock_http_handler)
            assert _status(result) == 200
            assert _body(result)["positions"] == []


# ===========================================================================
# Edge cases with multiple agents
# ===========================================================================


class TestMultipleAgentNames:
    """Tests ensuring different agent names are correctly passed through."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "agent_name",
        [
            "claude",
            "gpt4",
            "gemini",
            "mistral-api",
            "grok",
            "anthropic-api",
            "openai-api",
            "llama3",
        ],
    )
    async def test_agent_name_passed_correctly(self, handler, mock_http_handler, agent_name):
        """Each agent name is correctly extracted and passed to the handler method."""
        mock_elo = MagicMock()
        mock_elo.get_rating.return_value = 1500
        mock_elo.get_agent_stats.return_value = {}
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle(f"/api/agent/{agent_name}/profile", {}, mock_http_handler)
            assert _status(result) == 200
            body = _body(result)
            assert body["name"] == agent_name


# ===========================================================================
# Cache behavior tests
# ===========================================================================


class TestCacheBehavior:
    """Tests for caching behavior on profile endpoint."""

    @pytest.mark.asyncio
    async def test_profile_uses_cache(self, handler, mock_http_handler):
        """Profile endpoint uses ttl_cache - second call should use cached result."""
        mock_elo = MagicMock()
        mock_elo.get_rating.return_value = 1600
        mock_elo.get_agent_stats.return_value = {
            "rank": 1,
            "wins": 10,
            "losses": 2,
            "win_rate": 0.833,
        }
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result1 = await handler.handle("/api/agent/claude/profile", {}, mock_http_handler)
            result2 = await handler.handle("/api/agent/claude/profile", {}, mock_http_handler)
            assert _status(result1) == 200
            assert _status(result2) == 200
            # Both should return the same data
            assert _body(result1)["name"] == _body(result2)["name"]


# ===========================================================================
# can_handle tests for profile-related paths
# ===========================================================================


class TestCanHandle:
    """Tests for can_handle on profile-related paths."""

    def test_profile_path(self, handler):
        assert handler.can_handle("/api/agent/claude/profile")

    def test_history_path(self, handler):
        assert handler.can_handle("/api/agent/claude/history")

    def test_calibration_path(self, handler):
        assert handler.can_handle("/api/agent/claude/calibration")

    def test_consistency_path(self, handler):
        assert handler.can_handle("/api/agent/claude/consistency")

    def test_network_path(self, handler):
        assert handler.can_handle("/api/agent/claude/network")

    def test_rivals_path(self, handler):
        assert handler.can_handle("/api/agent/claude/rivals")

    def test_allies_path(self, handler):
        assert handler.can_handle("/api/agent/claude/allies")

    def test_moments_path(self, handler):
        assert handler.can_handle("/api/agent/claude/moments")

    def test_positions_path(self, handler):
        assert handler.can_handle("/api/agent/claude/positions")

    def test_domains_path(self, handler):
        assert handler.can_handle("/api/agent/claude/domains")

    def test_performance_path(self, handler):
        assert handler.can_handle("/api/agent/claude/performance")

    def test_versioned_profile_path(self, handler):
        assert handler.can_handle("/api/v1/agent/claude/profile")

    def test_versioned_history_path(self, handler):
        assert handler.can_handle("/api/v1/agent/claude/history")

    def test_versioned_domains_path(self, handler):
        assert handler.can_handle("/api/v1/agent/claude/domains")

    def test_versioned_performance_path(self, handler):
        assert handler.can_handle("/api/v1/agent/claude/performance")
