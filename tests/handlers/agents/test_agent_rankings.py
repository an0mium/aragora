"""Comprehensive tests for AgentRankingsMixin (aragora/server/handlers/agents/agent_rankings.py).

Tests cover all 4 endpoint methods provided by the mixin:
- GET /api/leaderboard (or /api/rankings) - Get agent rankings leaderboard
- GET /api/calibration/leaderboard - Get calibration leaderboard
- GET /api/matches/recent - Get recent agent matches
- GET /api/agent/compare - Compare multiple agents

Each endpoint is tested for:
- Happy path with valid data
- No ELO system (503)
- Edge cases (empty data, missing attributes)
- Input validation (limits, domains, agents list)
- Error handling (exceptions in ELO methods)
- Consistency scoring (FlipDetector integration)
- Caching behavior
"""

from __future__ import annotations

import json
from pathlib import Path
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
    """Reset caches and rate limiters before each test."""
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


def _make_mock_elo(**overrides):
    """Create a MagicMock EloSystem with common method defaults."""
    elo = MagicMock()
    elo.get_cached_leaderboard.return_value = overrides.get("cached_leaderboard", [])
    elo.get_leaderboard.return_value = overrides.get("leaderboard", [])
    elo.get_recent_matches.return_value = overrides.get("recent_matches", [])
    elo.get_cached_recent_matches.return_value = overrides.get("cached_recent_matches", [])
    elo.get_ratings_batch.return_value = overrides.get("ratings_batch", {})
    elo.get_agent_stats.return_value = overrides.get("agent_stats", {})
    elo.get_head_to_head.return_value = overrides.get("head_to_head", {})
    return elo


# ===========================================================================
# Leaderboard endpoint: /api/leaderboard and /api/rankings
# ===========================================================================

class TestGetLeaderboard:
    """Tests for the _get_leaderboard endpoint."""

    @pytest.mark.asyncio
    async def test_leaderboard_happy_path_no_domain(self, handler, mock_http_handler):
        """Leaderboard returns rankings when domain is not specified (uses cache)."""
        mock_elo = _make_mock_elo(cached_leaderboard=[
            {"name": "claude", "elo": 1650, "wins": 10, "losses": 3},
            {"name": "gpt4", "elo": 1600, "wins": 8, "losses": 5},
        ])
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            with patch.object(handler, "get_nomic_dir", return_value=None):
                result = await handler.handle(
                    "/api/leaderboard", {}, mock_http_handler
                )
                assert _status(result) == 200
                body = _body(result)
                assert "rankings" in body
                assert "agents" in body
                assert len(body["rankings"]) == 2
                mock_elo.get_cached_leaderboard.assert_called_once_with(limit=20)

    @pytest.mark.asyncio
    async def test_leaderboard_with_domain(self, handler, mock_http_handler):
        """Leaderboard uses get_leaderboard with domain filter when domain is specified."""
        mock_elo = _make_mock_elo(leaderboard=[
            {"name": "claude", "elo": 1700},
        ])
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            with patch.object(handler, "get_nomic_dir", return_value=None):
                result = await handler.handle(
                    "/api/leaderboard", {"domain": "technical"}, mock_http_handler
                )
                assert _status(result) == 200
                body = _body(result)
                assert len(body["rankings"]) == 1
                mock_elo.get_leaderboard.assert_called_once_with(
                    limit=20, domain="technical"
                )

    @pytest.mark.asyncio
    async def test_leaderboard_with_custom_limit(self, handler, mock_http_handler):
        """Leaderboard respects limit parameter."""
        mock_elo = _make_mock_elo(cached_leaderboard=[])
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            with patch.object(handler, "get_nomic_dir", return_value=None):
                result = await handler.handle(
                    "/api/leaderboard", {"limit": "10"}, mock_http_handler
                )
                assert _status(result) == 200
                mock_elo.get_cached_leaderboard.assert_called_once_with(limit=10)

    @pytest.mark.asyncio
    async def test_leaderboard_limit_capped_at_50(self, handler, mock_http_handler):
        """Leaderboard caps limit at 50."""
        mock_elo = _make_mock_elo(cached_leaderboard=[])
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            with patch.object(handler, "get_nomic_dir", return_value=None):
                result = await handler.handle(
                    "/api/leaderboard", {"limit": "200"}, mock_http_handler
                )
                assert _status(result) == 200
                mock_elo.get_cached_leaderboard.assert_called_once_with(limit=50)

    @pytest.mark.asyncio
    async def test_leaderboard_no_elo(self, handler, mock_http_handler):
        """Returns 503 when ELO system is not available."""
        with patch.object(handler, "get_elo_system", return_value=None):
            result = await handler.handle(
                "/api/leaderboard", {}, mock_http_handler
            )
            assert _status(result) == 503
            body = _body(result)
            assert "elo" in body.get("error", "").lower() or "not available" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_leaderboard_via_rankings_alias(self, handler, mock_http_handler):
        """The /api/rankings path is an alias for /api/leaderboard."""
        mock_elo = _make_mock_elo(cached_leaderboard=[
            {"name": "claude", "elo": 1600},
        ])
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            with patch.object(handler, "get_nomic_dir", return_value=None):
                result = await handler.handle(
                    "/api/rankings", {}, mock_http_handler
                )
                assert _status(result) == 200
                body = _body(result)
                assert len(body["rankings"]) == 1

    @pytest.mark.asyncio
    async def test_leaderboard_versioned_path(self, handler, mock_http_handler):
        """Leaderboard works with /api/v1/ prefix."""
        mock_elo = _make_mock_elo(cached_leaderboard=[])
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            with patch.object(handler, "get_nomic_dir", return_value=None):
                result = await handler.handle(
                    "/api/v1/leaderboard", {}, mock_http_handler
                )
                assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_leaderboard_empty_rankings(self, handler, mock_http_handler):
        """Leaderboard returns empty list when no agents exist."""
        mock_elo = _make_mock_elo(cached_leaderboard=[])
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            with patch.object(handler, "get_nomic_dir", return_value=None):
                result = await handler.handle(
                    "/api/leaderboard", {}, mock_http_handler
                )
                assert _status(result) == 200
                body = _body(result)
                assert body["rankings"] == []
                assert body["agents"] == []

    @pytest.mark.asyncio
    async def test_leaderboard_elo_exception(self, handler, mock_http_handler):
        """Returns 500 when ELO system raises an exception."""
        mock_elo = MagicMock()
        mock_elo.get_cached_leaderboard.side_effect = RuntimeError("DB error")
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle(
                "/api/leaderboard", {}, mock_http_handler
            )
            assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_leaderboard_invalid_domain_rejected(self, handler, mock_http_handler):
        """Leaderboard rejects unsafe domain parameter."""
        result = await handler.handle(
            "/api/leaderboard", {"domain": "<script>alert(1)</script>"}, mock_http_handler
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_leaderboard_rankings_and_agents_same(self, handler, mock_http_handler):
        """Both 'rankings' and 'agents' keys contain the same data."""
        mock_elo = _make_mock_elo(cached_leaderboard=[
            {"name": "claude", "elo": 1600},
        ])
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            with patch.object(handler, "get_nomic_dir", return_value=None):
                result = await handler.handle(
                    "/api/leaderboard", {}, mock_http_handler
                )
                assert _status(result) == 200
                body = _body(result)
                assert body["rankings"] == body["agents"]


# ===========================================================================
# Leaderboard consistency enrichment
# ===========================================================================

class TestLeaderboardConsistency:
    """Tests for consistency score enrichment in the leaderboard."""

    @pytest.mark.asyncio
    async def test_consistency_enrichment_with_flip_detector(self, handler, mock_http_handler):
        """Leaderboard adds consistency scores when FlipDetector is available."""
        mock_elo = _make_mock_elo(cached_leaderboard=[
            {"name": "claude", "elo": 1600},
            {"name": "gpt4", "elo": 1550},
        ])

        mock_score_claude = MagicMock()
        mock_score_claude.total_flips = 2
        mock_score_claude.total_positions = 10

        mock_score_gpt4 = MagicMock()
        mock_score_gpt4.total_flips = 5
        mock_score_gpt4.total_positions = 10

        mock_detector = MagicMock()
        mock_detector.get_agents_consistency_batch.return_value = {
            "claude": mock_score_claude,
            "gpt4": mock_score_gpt4,
        }

        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            with patch.object(handler, "get_nomic_dir", return_value=Path("/tmp/nomic")):
                with patch(
                    "aragora.insights.flip_detector.FlipDetector",
                    return_value=mock_detector,
                ):
                    with patch(
                        "aragora.server.handlers.agents.agent_rankings.get_db_path",
                        return_value="/tmp/nomic/positions.db",
                    ):
                        result = await handler.handle(
                            "/api/leaderboard", {}, mock_http_handler
                        )
                        assert _status(result) == 200
                        body = _body(result)
                        rankings = body["rankings"]

                        # claude: 1.0 - (2/10) = 0.8 -> high
                        claude_entry = next(r for r in rankings if r.get("name") == "claude")
                        assert claude_entry["consistency"] == 0.8
                        assert claude_entry["consistency_class"] == "high"

                        # gpt4: 1.0 - (5/10) = 0.5 -> low
                        gpt4_entry = next(r for r in rankings if r.get("name") == "gpt4")
                        assert gpt4_entry["consistency"] == 0.5
                        assert gpt4_entry["consistency_class"] == "low"

    @pytest.mark.asyncio
    async def test_consistency_medium_class(self, handler, mock_http_handler):
        """Agents with consistency between 0.6 and 0.8 get 'medium' class."""
        mock_elo = _make_mock_elo(cached_leaderboard=[
            {"name": "gemini", "elo": 1500},
        ])

        mock_score = MagicMock()
        mock_score.total_flips = 3
        mock_score.total_positions = 10  # consistency = 0.7 -> medium

        mock_detector = MagicMock()
        mock_detector.get_agents_consistency_batch.return_value = {
            "gemini": mock_score,
        }

        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            with patch.object(handler, "get_nomic_dir", return_value=Path("/tmp/nomic")):
                with patch(
                    "aragora.insights.flip_detector.FlipDetector",
                    return_value=mock_detector,
                ):
                    with patch(
                        "aragora.server.handlers.agents.agent_rankings.get_db_path",
                        return_value="/tmp/nomic/positions.db",
                    ):
                        result = await handler.handle(
                            "/api/leaderboard", {}, mock_http_handler
                        )
                        body = _body(result)
                        gemini_entry = body["rankings"][0]
                        assert gemini_entry["consistency"] == 0.7
                        assert gemini_entry["consistency_class"] == "medium"

    @pytest.mark.asyncio
    async def test_consistency_zero_positions_fallback(self, handler, mock_http_handler):
        """When total_positions is 0, uses max(total_positions, 1) to avoid division by zero."""
        mock_elo = _make_mock_elo(cached_leaderboard=[
            {"name": "claude", "elo": 1600},
        ])

        mock_score = MagicMock()
        mock_score.total_flips = 0
        mock_score.total_positions = 0  # max(0,1) = 1, consistency = 1.0 - 0/1 = 1.0

        mock_detector = MagicMock()
        mock_detector.get_agents_consistency_batch.return_value = {
            "claude": mock_score,
        }

        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            with patch.object(handler, "get_nomic_dir", return_value=Path("/tmp/nomic")):
                with patch(
                    "aragora.insights.flip_detector.FlipDetector",
                    return_value=mock_detector,
                ):
                    with patch(
                        "aragora.server.handlers.agents.agent_rankings.get_db_path",
                        return_value="/tmp/nomic/positions.db",
                    ):
                        result = await handler.handle(
                            "/api/leaderboard", {}, mock_http_handler
                        )
                        body = _body(result)
                        claude_entry = body["rankings"][0]
                        assert claude_entry["consistency"] == 1.0
                        assert claude_entry["consistency_class"] == "high"

    @pytest.mark.asyncio
    async def test_consistency_no_nomic_dir(self, handler, mock_http_handler):
        """When nomic_dir is None, no consistency scores are added."""
        mock_elo = _make_mock_elo(cached_leaderboard=[
            {"name": "claude", "elo": 1600},
        ])
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            with patch.object(handler, "get_nomic_dir", return_value=None):
                result = await handler.handle(
                    "/api/leaderboard", {}, mock_http_handler
                )
                assert _status(result) == 200
                body = _body(result)
                # No consistency keys should be present (no nomic_dir means no FlipDetector)
                assert len(body["rankings"]) == 1

    @pytest.mark.asyncio
    async def test_consistency_flipdetector_import_error(self, handler, mock_http_handler):
        """When FlipDetector cannot be imported, degraded flags are set."""
        mock_elo = _make_mock_elo(cached_leaderboard=[
            {"name": "claude", "elo": 1600},
        ])
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            with patch.object(handler, "get_nomic_dir", return_value=Path("/tmp/nomic")):
                with patch.dict("sys.modules", {"aragora.insights.flip_detector": None}):
                    with patch(
                        "aragora.server.handlers.agents.agent_rankings.get_db_path",
                        return_value="/tmp/nomic/positions.db",
                    ):
                        result = await handler.handle(
                            "/api/leaderboard", {}, mock_http_handler
                        )
                        assert _status(result) == 200
                        body = _body(result)
                        claude_entry = body["rankings"][0]
                        assert claude_entry.get("degraded") is True
                        assert claude_entry.get("consistency") is None
                        assert claude_entry.get("consistency_class") == "unknown"

    @pytest.mark.asyncio
    async def test_consistency_batch_lookup_error(self, handler, mock_http_handler):
        """When batch consistency lookup fails, fallback with degraded flag is used."""
        mock_elo = _make_mock_elo(cached_leaderboard=[
            {"name": "claude", "elo": 1600},
        ])

        mock_detector = MagicMock()
        mock_detector.get_agents_consistency_batch.side_effect = ValueError("DB corrupt")

        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            with patch.object(handler, "get_nomic_dir", return_value=Path("/tmp/nomic")):
                with patch(
                    "aragora.insights.flip_detector.FlipDetector",
                    return_value=mock_detector,
                ):
                    with patch(
                        "aragora.server.handlers.agents.agent_rankings.get_db_path",
                        return_value="/tmp/nomic/positions.db",
                    ):
                        result = await handler.handle(
                            "/api/leaderboard", {}, mock_http_handler
                        )
                        assert _status(result) == 200
                        body = _body(result)
                        claude_entry = body["rankings"][0]
                        assert claude_entry.get("degraded") is True
                        assert claude_entry.get("consistency") == 1.0
                        assert claude_entry.get("consistency_class") == "high"

    @pytest.mark.asyncio
    async def test_consistency_agents_with_no_names(self, handler, mock_http_handler):
        """Rankings with None agent names don't break consistency lookup."""
        mock_elo = _make_mock_elo(cached_leaderboard=[
            {"elo": 1500},  # no name key
        ])
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            with patch.object(handler, "get_nomic_dir", return_value=Path("/tmp/nomic")):
                with patch(
                    "aragora.insights.flip_detector.FlipDetector",
                    return_value=MagicMock(get_agents_consistency_batch=MagicMock(return_value={})),
                ):
                    with patch(
                        "aragora.server.handlers.agents.agent_rankings.get_db_path",
                        return_value="/tmp/nomic/positions.db",
                    ):
                        result = await handler.handle(
                            "/api/leaderboard", {}, mock_http_handler
                        )
                        assert _status(result) == 200
                        body = _body(result)
                        assert len(body["rankings"]) == 1


# ===========================================================================
# Calibration leaderboard: _get_calibration_leaderboard (direct method call)
# ===========================================================================

class TestGetCalibrationLeaderboard:
    """Tests for the _get_calibration_leaderboard method.

    This method is defined in AgentRankingsMixin but routing to it goes
    through CalibrationHandler. We test the method directly.
    """

    def test_calibration_leaderboard_happy_path(self, handler):
        """Returns rankings from ELO system."""
        mock_elo = _make_mock_elo(leaderboard=[
            {"name": "claude", "elo": 1650, "calibration_score": 0.92},
            {"name": "gpt4", "elo": 1600, "calibration_score": 0.88},
        ])
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = handler._get_calibration_leaderboard(20)
            assert _status(result) == 200
            body = _body(result)
            assert "rankings" in body
            assert len(body["rankings"]) == 2

    def test_calibration_leaderboard_no_elo(self, handler):
        """Returns 503 when ELO system is not available."""
        with patch.object(handler, "get_elo_system", return_value=None):
            result = handler._get_calibration_leaderboard(20)
            assert _status(result) == 503

    def test_calibration_leaderboard_limit_capped_at_50(self, handler):
        """Limit is capped at 50."""
        mock_elo = _make_mock_elo(leaderboard=[])
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            handler._get_calibration_leaderboard(200)
            mock_elo.get_leaderboard.assert_called_once_with(limit=50)

    def test_calibration_leaderboard_small_limit(self, handler):
        """Small limits are passed through unchanged."""
        mock_elo = _make_mock_elo(leaderboard=[])
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            handler._get_calibration_leaderboard(5)
            mock_elo.get_leaderboard.assert_called_once_with(limit=5)

    def test_calibration_leaderboard_elo_exception(self, handler):
        """Returns 500 when ELO raises an exception."""
        mock_elo = MagicMock()
        mock_elo.get_leaderboard.side_effect = RuntimeError("DB error")
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = handler._get_calibration_leaderboard(20)
            assert _status(result) == 500

    def test_calibration_leaderboard_empty(self, handler):
        """Returns empty rankings when no agents exist."""
        mock_elo = _make_mock_elo(leaderboard=[])
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = handler._get_calibration_leaderboard(20)
            assert _status(result) == 200
            body = _body(result)
            assert body["rankings"] == []

    def test_calibration_leaderboard_value_error(self, handler):
        """Returns 500 on ValueError."""
        mock_elo = MagicMock()
        mock_elo.get_leaderboard.side_effect = ValueError("Invalid data")
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = handler._get_calibration_leaderboard(10)
            assert _status(result) == 500

    def test_calibration_leaderboard_os_error(self, handler):
        """Returns 500 on OSError."""
        mock_elo = MagicMock()
        mock_elo.get_leaderboard.side_effect = OSError("File not found")
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = handler._get_calibration_leaderboard(10)
            assert _status(result) == 500


# ===========================================================================
# Recent matches: /api/matches/recent
# ===========================================================================

class TestGetRecentMatches:
    """Tests for the _get_recent_matches endpoint."""

    @pytest.mark.asyncio
    async def test_recent_matches_happy_path(self, handler, mock_http_handler):
        """Returns recent matches when ELO has get_cached_recent_matches."""
        mock_elo = _make_mock_elo(cached_recent_matches=[
            {"winner": "claude", "loser": "gpt4", "timestamp": 1000000},
            {"winner": "gemini", "loser": "claude", "timestamp": 1000001},
        ])
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle(
                "/api/matches/recent", {}, mock_http_handler
            )
            assert _status(result) == 200
            body = _body(result)
            assert "matches" in body
            assert len(body["matches"]) == 2

    @pytest.mark.asyncio
    async def test_recent_matches_no_elo(self, handler, mock_http_handler):
        """Returns 503 when ELO system is not available."""
        with patch.object(handler, "get_elo_system", return_value=None):
            result = await handler.handle(
                "/api/matches/recent", {}, mock_http_handler
            )
            assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_recent_matches_with_limit(self, handler, mock_http_handler):
        """Limit parameter is passed through."""
        mock_elo = _make_mock_elo(cached_recent_matches=[])
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle(
                "/api/matches/recent", {"limit": "5"}, mock_http_handler
            )
            assert _status(result) == 200
            mock_elo.get_cached_recent_matches.assert_called_once_with(limit=5)

    @pytest.mark.asyncio
    async def test_recent_matches_limit_capped_at_50(self, handler, mock_http_handler):
        """Limit is capped at 50."""
        mock_elo = _make_mock_elo(cached_recent_matches=[])
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle(
                "/api/matches/recent", {"limit": "500"}, mock_http_handler
            )
            assert _status(result) == 200
            mock_elo.get_cached_recent_matches.assert_called_once_with(limit=50)

    @pytest.mark.asyncio
    async def test_recent_matches_default_limit(self, handler, mock_http_handler):
        """Default limit is 10 when not specified."""
        mock_elo = _make_mock_elo(cached_recent_matches=[])
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle(
                "/api/matches/recent", {}, mock_http_handler
            )
            assert _status(result) == 200
            mock_elo.get_cached_recent_matches.assert_called_once_with(limit=10)

    @pytest.mark.asyncio
    async def test_recent_matches_fallback_to_uncached(self, handler, mock_http_handler):
        """Falls back to get_recent_matches when get_cached_recent_matches is not available."""
        mock_elo = MagicMock(spec=["get_recent_matches"])
        mock_elo.get_recent_matches.return_value = [
            {"winner": "claude", "loser": "gpt4"},
        ]
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle(
                "/api/matches/recent", {}, mock_http_handler
            )
            assert _status(result) == 200
            body = _body(result)
            assert len(body["matches"]) == 1
            mock_elo.get_recent_matches.assert_called_once_with(limit=10)

    @pytest.mark.asyncio
    async def test_recent_matches_empty(self, handler, mock_http_handler):
        """Returns empty list when no matches exist."""
        mock_elo = _make_mock_elo(cached_recent_matches=[])
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle(
                "/api/matches/recent", {}, mock_http_handler
            )
            assert _status(result) == 200
            body = _body(result)
            assert body["matches"] == []

    @pytest.mark.asyncio
    async def test_recent_matches_elo_exception(self, handler, mock_http_handler):
        """Returns 500 when ELO raises an exception."""
        mock_elo = MagicMock()
        mock_elo.get_cached_recent_matches.side_effect = RuntimeError("DB locked")
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle(
                "/api/matches/recent", {}, mock_http_handler
            )
            assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_recent_matches_versioned_path(self, handler, mock_http_handler):
        """Works with /api/v1/matches/recent."""
        mock_elo = _make_mock_elo(cached_recent_matches=[])
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle(
                "/api/v1/matches/recent", {}, mock_http_handler
            )
            assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_recent_matches_with_loop_id(self, handler, mock_http_handler):
        """Passes loop_id parameter through."""
        mock_elo = _make_mock_elo(cached_recent_matches=[])
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle(
                "/api/matches/recent", {"loop_id": "loop-abc123"}, mock_http_handler
            )
            assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_recent_matches_invalid_loop_id(self, handler, mock_http_handler):
        """Rejects loop_id with unsafe characters."""
        result = await handler.handle(
            "/api/matches/recent",
            {"loop_id": "../../../etc/passwd"},
            mock_http_handler,
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_recent_matches_os_error(self, handler, mock_http_handler):
        """Returns 500 on OSError."""
        mock_elo = MagicMock()
        mock_elo.get_cached_recent_matches.side_effect = OSError("Disk failure")
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle(
                "/api/matches/recent", {}, mock_http_handler
            )
            assert _status(result) == 500


# ===========================================================================
# Agent comparison: /api/agent/compare
# ===========================================================================

class TestCompareAgents:
    """Tests for the _compare_agents endpoint."""

    @pytest.mark.asyncio
    async def test_compare_two_agents_happy_path(self, handler, mock_http_handler):
        """Comparing two agents returns profiles and head-to-head data."""
        mock_elo = _make_mock_elo(
            ratings_batch={"claude": 1650, "gpt4": 1600},
            agent_stats={"wins": 10, "losses": 5},
            head_to_head={"claude_wins": 3, "gpt4_wins": 2},
        )
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle(
                "/api/agent/compare",
                {"agents": ["claude", "gpt4"]},
                mock_http_handler,
            )
            assert _status(result) == 200
            body = _body(result)
            assert "agents" in body
            assert len(body["agents"]) == 2
            assert body["agents"][0]["name"] == "claude"
            assert body["agents"][0]["rating"] == 1650
            assert body["agents"][1]["name"] == "gpt4"
            assert body["agents"][1]["rating"] == 1600
            assert body["head_to_head"] is not None

    @pytest.mark.asyncio
    async def test_compare_three_agents(self, handler, mock_http_handler):
        """Comparing 3+ agents returns profiles without head-to-head."""
        mock_elo = _make_mock_elo(
            ratings_batch={"claude": 1650, "gpt4": 1600, "gemini": 1550},
            agent_stats={},
        )
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle(
                "/api/agent/compare",
                {"agents": ["claude", "gpt4", "gemini"]},
                mock_http_handler,
            )
            assert _status(result) == 200
            body = _body(result)
            assert len(body["agents"]) == 3
            assert body["head_to_head"] is None

    @pytest.mark.asyncio
    async def test_compare_fewer_than_two(self, handler, mock_http_handler):
        """Returns 400 when fewer than 2 agents are provided."""
        with patch.object(handler, "get_elo_system", return_value=_make_mock_elo()):
            result = await handler.handle(
                "/api/agent/compare",
                {"agents": ["claude"]},
                mock_http_handler,
            )
            assert _status(result) == 400
            body = _body(result)
            assert "2" in body.get("error", "")

    @pytest.mark.asyncio
    async def test_compare_empty_agents_list(self, handler, mock_http_handler):
        """Returns 400 when agents list is empty."""
        with patch.object(handler, "get_elo_system", return_value=_make_mock_elo()):
            result = await handler.handle(
                "/api/agent/compare",
                {"agents": []},
                mock_http_handler,
            )
            assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_compare_no_elo(self, handler, mock_http_handler):
        """Returns 503 when ELO system is not available."""
        with patch.object(handler, "get_elo_system", return_value=None):
            result = await handler.handle(
                "/api/agent/compare",
                {"agents": ["claude", "gpt4"]},
                mock_http_handler,
            )
            assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_compare_limits_to_5_agents(self, handler, mock_http_handler):
        """Only the first 5 agents are compared even if more are provided."""
        agents = ["a1", "a2", "a3", "a4", "a5", "a6", "a7"]
        mock_elo = _make_mock_elo(
            ratings_batch={f"a{i}": 1500 for i in range(1, 8)},
            agent_stats={},
        )
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle(
                "/api/agent/compare",
                {"agents": agents},
                mock_http_handler,
            )
            assert _status(result) == 200
            body = _body(result)
            assert len(body["agents"]) == 5

    @pytest.mark.asyncio
    async def test_compare_uses_initial_rating_for_missing(self, handler, mock_http_handler):
        """Agents not in ratings_batch get the initial ELO rating."""
        from aragora.config import ELO_INITIAL_RATING
        mock_elo = _make_mock_elo(
            ratings_batch={"claude": 1650},  # gpt4 missing
            agent_stats={},
        )
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle(
                "/api/agent/compare",
                {"agents": ["claude", "gpt4"]},
                mock_http_handler,
            )
            assert _status(result) == 200
            body = _body(result)
            gpt4_profile = body["agents"][1]
            assert gpt4_profile["rating"] == ELO_INITIAL_RATING

    @pytest.mark.asyncio
    async def test_compare_no_agent_stats_method(self, handler, mock_http_handler):
        """Works when ELO has no get_agent_stats method."""
        mock_elo = MagicMock(spec=["get_ratings_batch"])
        mock_elo.get_ratings_batch.return_value = {"claude": 1650, "gpt4": 1600}
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle(
                "/api/agent/compare",
                {"agents": ["claude", "gpt4"]},
                mock_http_handler,
            )
            assert _status(result) == 200
            body = _body(result)
            assert len(body["agents"]) == 2

    @pytest.mark.asyncio
    async def test_compare_head_to_head_failure(self, handler, mock_http_handler):
        """Head-to-head failure is gracefully handled."""
        mock_elo = _make_mock_elo(
            ratings_batch={"claude": 1650, "gpt4": 1600},
            agent_stats={},
        )
        mock_elo.get_head_to_head.side_effect = RuntimeError("H2H not available")
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle(
                "/api/agent/compare",
                {"agents": ["claude", "gpt4"]},
                mock_http_handler,
            )
            assert _status(result) == 200
            body = _body(result)
            # h2h failure is caught, comparison still works
            assert body["head_to_head"] is None
            assert len(body["agents"]) == 2

    @pytest.mark.asyncio
    async def test_compare_elo_exception(self, handler, mock_http_handler):
        """Returns 500 when ELO raises during comparison."""
        mock_elo = MagicMock()
        mock_elo.get_ratings_batch.side_effect = RuntimeError("ELO system error")
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle(
                "/api/agent/compare",
                {"agents": ["claude", "gpt4"]},
                mock_http_handler,
            )
            assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_compare_versioned_path(self, handler, mock_http_handler):
        """Works with /api/v1/agent/compare."""
        mock_elo = _make_mock_elo(
            ratings_batch={"claude": 1650, "gpt4": 1600},
            agent_stats={},
        )
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle(
                "/api/v1/agent/compare",
                {"agents": ["claude", "gpt4"]},
                mock_http_handler,
            )
            assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_compare_agents_stats_merged(self, handler, mock_http_handler):
        """Stats from get_agent_stats are merged into the profile."""
        mock_elo = _make_mock_elo(
            ratings_batch={"claude": 1650, "gpt4": 1600},
            agent_stats={"wins": 10, "losses": 3, "win_rate": 0.77},
        )
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle(
                "/api/agent/compare",
                {"agents": ["claude", "gpt4"]},
                mock_http_handler,
            )
            assert _status(result) == 200
            body = _body(result)
            for profile in body["agents"]:
                assert profile["wins"] == 10
                assert profile["losses"] == 3
                assert profile["win_rate"] == 0.77

    @pytest.mark.asyncio
    async def test_compare_agents_string_param(self, handler, mock_http_handler):
        """When agents parameter is a single string, it is wrapped in a list (< 2 agents)."""
        with patch.object(handler, "get_elo_system", return_value=_make_mock_elo()):
            result = await handler.handle(
                "/api/agent/compare",
                {"agents": "claude"},
                mock_http_handler,
            )
            assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_compare_agents_rating_none_uses_default(self, handler, mock_http_handler):
        """When ratings_batch returns None for an agent, uses ELO_INITIAL_RATING."""
        from aragora.config import ELO_INITIAL_RATING
        mock_elo = _make_mock_elo(
            ratings_batch={"claude": None, "gpt4": 1600},
            agent_stats={},
        )
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle(
                "/api/agent/compare",
                {"agents": ["claude", "gpt4"]},
                mock_http_handler,
            )
            assert _status(result) == 200
            body = _body(result)
            claude_profile = body["agents"][0]
            assert claude_profile["rating"] == ELO_INITIAL_RATING

    @pytest.mark.asyncio
    async def test_compare_key_error(self, handler, mock_http_handler):
        """Returns 500 on KeyError from ELO."""
        mock_elo = MagicMock()
        mock_elo.get_ratings_batch.side_effect = KeyError("missing key")
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle(
                "/api/agent/compare",
                {"agents": ["claude", "gpt4"]},
                mock_http_handler,
            )
            assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_compare_type_error(self, handler, mock_http_handler):
        """Returns 500 on TypeError from ELO."""
        mock_elo = MagicMock()
        mock_elo.get_ratings_batch.side_effect = TypeError("bad type")
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle(
                "/api/agent/compare",
                {"agents": ["claude", "gpt4"]},
                mock_http_handler,
            )
            assert _status(result) == 500


# ===========================================================================
# can_handle tests for ranking-related paths
# ===========================================================================

class TestCanHandle:
    """Tests for can_handle on ranking-related paths."""

    def test_leaderboard_path(self, handler):
        assert handler.can_handle("/api/leaderboard")

    def test_leaderboard_versioned(self, handler):
        assert handler.can_handle("/api/v1/leaderboard")

    def test_rankings_path(self, handler):
        assert handler.can_handle("/api/rankings")

    def test_rankings_versioned(self, handler):
        assert handler.can_handle("/api/v1/rankings")

    def test_matches_recent_path(self, handler):
        assert handler.can_handle("/api/matches/recent")

    def test_matches_recent_versioned(self, handler):
        assert handler.can_handle("/api/v1/matches/recent")

    def test_agent_compare_path(self, handler):
        assert handler.can_handle("/api/agent/compare")

    def test_agent_compare_versioned(self, handler):
        assert handler.can_handle("/api/v1/agent/compare")

    def test_unrelated_path(self, handler):
        assert not handler.can_handle("/api/debates")


# ===========================================================================
# Error exception type coverage
# ===========================================================================

class TestErrorExceptionCoverage:
    """Tests ensuring various exception types are caught correctly."""

    @pytest.mark.asyncio
    async def test_leaderboard_value_error(self, handler, mock_http_handler):
        """ValueError in leaderboard returns 500."""
        mock_elo = MagicMock()
        mock_elo.get_cached_leaderboard.side_effect = ValueError("bad value")
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle(
                "/api/leaderboard", {}, mock_http_handler
            )
            assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_leaderboard_key_error(self, handler, mock_http_handler):
        """KeyError in leaderboard returns 500."""
        mock_elo = MagicMock()
        mock_elo.get_cached_leaderboard.side_effect = KeyError("missing")
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle(
                "/api/leaderboard", {}, mock_http_handler
            )
            assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_leaderboard_type_error(self, handler, mock_http_handler):
        """TypeError in leaderboard returns 500."""
        mock_elo = MagicMock()
        mock_elo.get_cached_leaderboard.side_effect = TypeError("wrong type")
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle(
                "/api/leaderboard", {}, mock_http_handler
            )
            assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_leaderboard_os_error(self, handler, mock_http_handler):
        """OSError in leaderboard returns 500."""
        mock_elo = MagicMock()
        mock_elo.get_cached_leaderboard.side_effect = OSError("disk full")
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle(
                "/api/leaderboard", {}, mock_http_handler
            )
            assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_recent_matches_value_error(self, handler, mock_http_handler):
        """ValueError in recent_matches returns 500."""
        mock_elo = MagicMock()
        mock_elo.get_cached_recent_matches.side_effect = ValueError("bad")
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle(
                "/api/matches/recent", {}, mock_http_handler
            )
            assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_compare_os_error(self, handler, mock_http_handler):
        """OSError in compare returns 500."""
        mock_elo = MagicMock()
        mock_elo.get_ratings_batch.side_effect = OSError("disk")
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle(
                "/api/agent/compare",
                {"agents": ["claude", "gpt4"]},
                mock_http_handler,
            )
            assert _status(result) == 500


# ===========================================================================
# Edge cases
# ===========================================================================

class TestEdgeCases:
    """Miscellaneous edge case tests."""

    @pytest.mark.asyncio
    async def test_leaderboard_agent_with_agent_name_key(self, handler, mock_http_handler):
        """Agents with 'agent_name' key (not 'name') are handled correctly."""
        mock_elo = _make_mock_elo(cached_leaderboard=[
            {"agent_name": "claude-v2", "elo": 1650},
        ])
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            with patch.object(handler, "get_nomic_dir", return_value=None):
                result = await handler.handle(
                    "/api/leaderboard", {}, mock_http_handler
                )
                assert _status(result) == 200
                body = _body(result)
                assert len(body["rankings"]) == 1

    @pytest.mark.asyncio
    async def test_leaderboard_object_agents(self, handler, mock_http_handler):
        """Agents that are objects (not dicts) are handled via agent_to_dict."""
        mock_agent = MagicMock()
        mock_agent.name = "claude"
        mock_agent.agent_name = None
        mock_agent.elo = 1650
        mock_agent.wins = 10
        mock_agent.losses = 3
        mock_agent.draws = 2
        mock_agent.domain_elos = {}
        mock_agent.calibration_brier_score = 0.15
        mock_agent.calibration_total = 30

        mock_elo = _make_mock_elo(cached_leaderboard=[mock_agent])
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            with patch.object(handler, "get_nomic_dir", return_value=None):
                result = await handler.handle(
                    "/api/leaderboard", {}, mock_http_handler
                )
                assert _status(result) == 200
                body = _body(result)
                assert len(body["rankings"]) == 1

    @pytest.mark.asyncio
    async def test_compare_exactly_five_agents(self, handler, mock_http_handler):
        """Comparing exactly 5 agents returns all 5 profiles without head-to-head."""
        agents = ["a1", "a2", "a3", "a4", "a5"]
        mock_elo = _make_mock_elo(
            ratings_batch={a: 1500 + i * 10 for i, a in enumerate(agents)},
            agent_stats={},
        )
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle(
                "/api/agent/compare",
                {"agents": agents},
                mock_http_handler,
            )
            assert _status(result) == 200
            body = _body(result)
            assert len(body["agents"]) == 5
            assert body["head_to_head"] is None

    @pytest.mark.asyncio
    async def test_compare_exactly_two_agents_h2h_called(self, handler, mock_http_handler):
        """With exactly 2 agents, head-to-head lookup is attempted."""
        mock_elo = _make_mock_elo(
            ratings_batch={"claude": 1600, "gpt4": 1550},
            agent_stats={},
            head_to_head={"wins_a": 5, "wins_b": 3},
        )
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle(
                "/api/agent/compare",
                {"agents": ["claude", "gpt4"]},
                mock_http_handler,
            )
            assert _status(result) == 200
            body = _body(result)
            mock_elo.get_head_to_head.assert_called_once_with("claude", "gpt4")
            assert body["head_to_head"] is not None

    @pytest.mark.asyncio
    async def test_leaderboard_domain_valid_patterns(self, handler, mock_http_handler):
        """Various valid domain patterns are accepted."""
        mock_elo = _make_mock_elo(leaderboard=[])
        for domain in ["technical", "finance", "legal-review", "science123"]:
            with patch.object(handler, "get_elo_system", return_value=mock_elo):
                with patch.object(handler, "get_nomic_dir", return_value=None):
                    result = await handler.handle(
                        "/api/leaderboard", {"domain": domain}, mock_http_handler
                    )
                    assert _status(result) == 200, f"Domain '{domain}' should be valid"

    @pytest.mark.asyncio
    async def test_leaderboard_limit_zero(self, handler, mock_http_handler):
        """Limit of 0 is passed through (min(0, 50) = 0)."""
        mock_elo = _make_mock_elo(cached_leaderboard=[])
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            with patch.object(handler, "get_nomic_dir", return_value=None):
                result = await handler.handle(
                    "/api/leaderboard", {"limit": "0"}, mock_http_handler
                )
                assert _status(result) == 200
                mock_elo.get_cached_leaderboard.assert_called_once_with(limit=0)

    @pytest.mark.asyncio
    async def test_leaderboard_negative_limit(self, handler, mock_http_handler):
        """Negative limit is passed through (min(-5, 50) = -5)."""
        mock_elo = _make_mock_elo(cached_leaderboard=[])
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            with patch.object(handler, "get_nomic_dir", return_value=None):
                result = await handler.handle(
                    "/api/leaderboard", {"limit": "-5"}, mock_http_handler
                )
                assert _status(result) == 200
                mock_elo.get_cached_leaderboard.assert_called_once_with(limit=-5)

    @pytest.mark.asyncio
    async def test_recent_matches_key_error(self, handler, mock_http_handler):
        """KeyError in recent_matches returns 500."""
        mock_elo = MagicMock()
        mock_elo.get_cached_recent_matches.side_effect = KeyError("no key")
        with patch.object(handler, "get_elo_system", return_value=mock_elo):
            result = await handler.handle(
                "/api/matches/recent", {}, mock_http_handler
            )
            assert _status(result) == 500
