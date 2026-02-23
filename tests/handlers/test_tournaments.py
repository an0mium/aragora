"""
Tests for Tournament Handler.

Tests cover:
- Handler routing (can_handle) for all tournament endpoints
- GET /api/tournaments - List tournaments
- POST /api/tournaments - Create tournament
- GET /api/tournaments/{id} - Get tournament details
- GET /api/tournaments/{id}/standings - Get standings
- GET /api/tournaments/{id}/bracket - Get bracket structure
- GET /api/tournaments/{id}/matches - Get match history
- POST /api/tournaments/{id}/advance - Advance tournament round
- POST /api/tournaments/{id}/matches/{match_id}/result - Record match result
- Rate limiting
- API versioning (strip_version_prefix)
- Input validation and path traversal prevention
- Error handling when tournament system not available
- Error handling when nomic_dir not configured
- Edge cases for missing/corrupted tournaments
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.tournaments import (
    TournamentHandler,
    TOURNAMENT_AVAILABLE,
    _tournament_limiter,
)


# ============================================================================
# Dataclasses for mock tournament objects
# ============================================================================


@dataclass
class MockStanding:
    """Mock tournament standing."""

    agent: str
    wins: int
    losses: int
    draws: int
    points: float
    total_score: float
    win_rate: float


@dataclass
class MockTournament:
    """Mock tournament object."""

    tournament_id: str
    name: str
    bracket_type: str
    participants: list[str]
    rounds_completed: int
    total_rounds: int
    status: str
    created_at: str


@dataclass
class MockMatch:
    """Mock tournament match."""

    match_id: str
    round_num: int
    agent1: str
    agent2: str
    winner: str | None
    score1: float
    score2: float
    debate_id: str | None
    created_at: str
    completed_at: str | None


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_http_handler():
    """Create mock HTTP handler with client_address for rate limiter."""
    mock = MagicMock()
    mock.client_address = ("127.0.0.1", 12345)
    mock.headers = {}
    return mock


@pytest.fixture
def handler_no_ctx():
    """Handler with no context."""
    return TournamentHandler(ctx={})


@pytest.fixture
def handler_with_ctx(tmp_path):
    """Handler with valid nomic_dir context."""
    nomic_dir = tmp_path / "nomic"
    nomic_dir.mkdir()
    return TournamentHandler(ctx={"nomic_dir": str(nomic_dir)})


@pytest.fixture(autouse=True)
def clear_rate_limits():
    """Clear rate limiter state between tests."""
    _tournament_limiter._buckets.clear()
    yield
    _tournament_limiter._buckets.clear()


def _parse_body(result) -> dict[str, Any]:
    """Parse JSON body from HandlerResult."""
    return json.loads(result.body.decode("utf-8"))


# ============================================================================
# Routing Tests (can_handle)
# ============================================================================


class TestCanHandle:
    """Tests for TournamentHandler.can_handle routing."""

    def test_handles_tournaments_list(self, handler_no_ctx):
        assert handler_no_ctx.can_handle("/api/tournaments") is True

    def test_handles_tournaments_list_versioned(self, handler_no_ctx):
        assert handler_no_ctx.can_handle("/api/v1/tournaments") is True

    def test_handles_tournaments_list_v2(self, handler_no_ctx):
        assert handler_no_ctx.can_handle("/api/v2/tournaments") is True

    def test_handles_tournament_by_id(self, handler_no_ctx):
        assert handler_no_ctx.can_handle("/api/tournaments/abc123") is True

    def test_handles_tournament_by_id_versioned(self, handler_no_ctx):
        assert handler_no_ctx.can_handle("/api/v1/tournaments/abc123") is True

    def test_handles_standings(self, handler_no_ctx):
        assert handler_no_ctx.can_handle("/api/tournaments/abc123/standings") is True

    def test_handles_standings_versioned(self, handler_no_ctx):
        assert handler_no_ctx.can_handle("/api/v1/tournaments/abc123/standings") is True

    def test_handles_bracket(self, handler_no_ctx):
        assert handler_no_ctx.can_handle("/api/tournaments/abc123/bracket") is True

    def test_handles_bracket_versioned(self, handler_no_ctx):
        assert handler_no_ctx.can_handle("/api/v1/tournaments/abc123/bracket") is True

    def test_handles_matches(self, handler_no_ctx):
        assert handler_no_ctx.can_handle("/api/tournaments/abc123/matches") is True

    def test_handles_advance(self, handler_no_ctx):
        assert handler_no_ctx.can_handle("/api/tournaments/abc123/advance") is True

    def test_handles_match_result(self, handler_no_ctx):
        assert handler_no_ctx.can_handle(
            "/api/tournaments/abc123/matches/m1/result"
        ) is True

    def test_handles_match_result_versioned(self, handler_no_ctx):
        assert handler_no_ctx.can_handle(
            "/api/v1/tournaments/abc123/matches/m1/result"
        ) is True

    def test_rejects_unrelated_path(self, handler_no_ctx):
        assert handler_no_ctx.can_handle("/api/debates") is False

    def test_rejects_partial_tournament_path(self, handler_no_ctx):
        assert handler_no_ctx.can_handle("/api/tournament") is False

    def test_rejects_unknown_sub_path(self, handler_no_ctx):
        assert handler_no_ctx.can_handle("/api/tournaments/abc/unknown") is False

    def test_rejects_too_short_path(self, handler_no_ctx):
        assert handler_no_ctx.can_handle("/api") is False

    def test_rejects_wrong_sub_resource(self, handler_no_ctx):
        assert handler_no_ctx.can_handle("/api/tournaments/abc/stats") is False

    def test_rejects_match_result_wrong_format(self, handler_no_ctx):
        # /api/tournaments/{id}/matches/{match_id} without /result
        assert handler_no_ctx.can_handle("/api/tournaments/abc/matches/m1") is False

    def test_rejects_empty_path(self, handler_no_ctx):
        assert handler_no_ctx.can_handle("") is False

    def test_rejects_root_path(self, handler_no_ctx):
        assert handler_no_ctx.can_handle("/") is False


# ============================================================================
# GET /api/tournaments (List)
# ============================================================================


class TestListTournaments:
    """Tests for listing tournaments."""

    @patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", False)
    def test_list_unavailable(self, handler_with_ctx, mock_http_handler):
        h = TournamentHandler(ctx=handler_with_ctx.ctx)
        result = h.handle("/api/tournaments", {}, mock_http_handler)
        assert result.status_code == 503
        body = _parse_body(result)
        assert "not available" in body["error"]

    def test_list_no_nomic_dir(self, mock_http_handler):
        h = TournamentHandler(ctx={})
        with patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True):
            result = h.handle("/api/tournaments", {}, mock_http_handler)
        assert result.status_code == 503
        body = _parse_body(result)
        assert "not configured" in body["error"]

    def test_list_empty_tournaments_dir(self, handler_with_ctx, mock_http_handler, tmp_path):
        """No tournaments directory yet returns empty list."""
        with patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True):
            result = handler_with_ctx.handle("/api/tournaments", {}, mock_http_handler)
        assert result.status_code == 200
        body = _parse_body(result)
        assert body["tournaments"] == []
        assert body["count"] == 0

    def test_list_with_tournaments(self, handler_with_ctx, mock_http_handler):
        """List returns tournament summaries from db files."""
        nomic_dir = Path(handler_with_ctx.ctx["nomic_dir"])
        tournaments_dir = nomic_dir / "tournaments"
        tournaments_dir.mkdir(parents=True, exist_ok=True)
        # Create a fake .db file
        (tournaments_dir / "tourney1.db").write_text("")

        mock_standings = [
            MockStanding("agent_a", 3, 1, 0, 9.0, 15.0, 0.75),
            MockStanding("agent_b", 1, 3, 0, 3.0, 8.0, 0.25),
        ]
        mock_manager = MagicMock()
        mock_manager.get_current_standings.return_value = mock_standings

        with patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True), \
             patch("aragora.server.handlers.tournaments._TournamentManager", return_value=mock_manager):
            result = handler_with_ctx.handle("/api/tournaments", {}, mock_http_handler)

        assert result.status_code == 200
        body = _parse_body(result)
        assert body["count"] == 1
        assert body["tournaments"][0]["tournament_id"] == "tourney1"
        assert body["tournaments"][0]["participants"] == 2
        assert body["tournaments"][0]["top_agent"] == "agent_a"

    def test_list_skips_corrupted_db(self, handler_with_ctx, mock_http_handler):
        """Corrupted db files are skipped gracefully."""
        nomic_dir = Path(handler_with_ctx.ctx["nomic_dir"])
        tournaments_dir = nomic_dir / "tournaments"
        tournaments_dir.mkdir(parents=True, exist_ok=True)
        (tournaments_dir / "bad.db").write_text("")

        mock_manager = MagicMock()
        mock_manager.get_current_standings.side_effect = RuntimeError("corrupt")

        with patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True), \
             patch("aragora.server.handlers.tournaments._TournamentManager", return_value=mock_manager):
            result = handler_with_ctx.handle("/api/tournaments", {}, mock_http_handler)

        assert result.status_code == 200
        body = _parse_body(result)
        assert body["count"] == 0
        assert body["tournaments"] == []

    def test_list_empty_standings(self, handler_with_ctx, mock_http_handler):
        """Tournament with no standings shows null top_agent."""
        nomic_dir = Path(handler_with_ctx.ctx["nomic_dir"])
        tournaments_dir = nomic_dir / "tournaments"
        tournaments_dir.mkdir(parents=True, exist_ok=True)
        (tournaments_dir / "empty.db").write_text("")

        mock_manager = MagicMock()
        mock_manager.get_current_standings.return_value = []

        with patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True), \
             patch("aragora.server.handlers.tournaments._TournamentManager", return_value=mock_manager):
            result = handler_with_ctx.handle("/api/tournaments", {}, mock_http_handler)

        assert result.status_code == 200
        body = _parse_body(result)
        assert body["count"] == 1
        assert body["tournaments"][0]["top_agent"] is None
        assert body["tournaments"][0]["participants"] == 0

    def test_list_versioned_path(self, handler_with_ctx, mock_http_handler):
        """Versioned path is stripped before routing."""
        with patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True):
            result = handler_with_ctx.handle("/api/v1/tournaments", {}, mock_http_handler)
        assert result.status_code == 200

    def test_list_max_tournaments_limit(self, handler_with_ctx, mock_http_handler):
        """Directory scanning is bounded by MAX_TOURNAMENTS_TO_LIST."""
        nomic_dir = Path(handler_with_ctx.ctx["nomic_dir"])
        tournaments_dir = nomic_dir / "tournaments"
        tournaments_dir.mkdir(parents=True, exist_ok=True)
        # Create 105 fake db files
        for i in range(105):
            (tournaments_dir / f"t{i:03d}.db").write_text("")

        mock_manager = MagicMock()
        mock_manager.get_current_standings.return_value = []

        with patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True), \
             patch("aragora.server.handlers.tournaments._TournamentManager", return_value=mock_manager):
            result = handler_with_ctx.handle("/api/tournaments", {}, mock_http_handler)

        assert result.status_code == 200
        body = _parse_body(result)
        assert body["count"] <= 100


# ============================================================================
# GET /api/tournaments/{id} (Get Details)
# ============================================================================


class TestGetTournament:
    """Tests for getting tournament details."""

    @patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", False)
    def test_get_unavailable(self, handler_with_ctx, mock_http_handler):
        h = TournamentHandler(ctx=handler_with_ctx.ctx)
        result = h.handle("/api/tournaments/abc", {}, mock_http_handler)
        assert result.status_code == 503

    def test_get_no_nomic_dir(self, mock_http_handler):
        h = TournamentHandler(ctx={})
        with patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True):
            result = h.handle("/api/tournaments/abc", {}, mock_http_handler)
        assert result.status_code == 503

    def test_get_not_found(self, handler_with_ctx, mock_http_handler):
        """Tournament not found returns 404."""
        with patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True):
            result = handler_with_ctx.handle("/api/tournaments/nonexistent", {}, mock_http_handler)
        assert result.status_code == 404

    def test_get_success(self, handler_with_ctx, mock_http_handler):
        """Successfully get tournament details."""
        nomic_dir = Path(handler_with_ctx.ctx["nomic_dir"])
        (nomic_dir / "tournaments").mkdir(parents=True, exist_ok=True)
        (nomic_dir / "tournaments" / "abc.db").write_text("")

        mock_tournament = MockTournament(
            tournament_id="abc",
            name="Test Tournament",
            bracket_type="round_robin",
            participants=["a", "b", "c"],
            rounds_completed=1,
            total_rounds=3,
            status="active",
            created_at="2026-01-01T00:00:00",
        )
        mock_manager = MagicMock()
        mock_manager.get_tournament.return_value = mock_tournament

        with patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True), \
             patch("aragora.server.handlers.tournaments._TournamentManager", return_value=mock_manager):
            result = handler_with_ctx.handle("/api/tournaments/abc", {}, mock_http_handler)

        assert result.status_code == 200
        body = _parse_body(result)
        assert body["tournament_id"] == "abc"
        assert body["name"] == "Test Tournament"
        assert body["bracket_type"] == "round_robin"
        assert body["status"] == "active"
        assert body["rounds_completed"] == 1
        assert body["total_rounds"] == 3

    def test_get_fallback_to_list(self, handler_with_ctx, mock_http_handler):
        """When get_tournament returns None, falls back to list_tournaments."""
        nomic_dir = Path(handler_with_ctx.ctx["nomic_dir"])
        (nomic_dir / "tournaments").mkdir(parents=True, exist_ok=True)
        (nomic_dir / "tournaments" / "abc.db").write_text("")

        mock_tournament = MockTournament(
            tournament_id="other_id",
            name="Fallback Tournament",
            bracket_type="single_elimination",
            participants=["x", "y"],
            rounds_completed=0,
            total_rounds=1,
            status="pending",
            created_at="2026-02-01T00:00:00",
        )
        mock_manager = MagicMock()
        mock_manager.get_tournament.return_value = None
        mock_manager.list_tournaments.return_value = [mock_tournament]

        with patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True), \
             patch("aragora.server.handlers.tournaments._TournamentManager", return_value=mock_manager):
            result = handler_with_ctx.handle("/api/tournaments/abc", {}, mock_http_handler)

        assert result.status_code == 200
        body = _parse_body(result)
        assert body["tournament_id"] == "other_id"

    def test_get_fallback_also_none(self, handler_with_ctx, mock_http_handler):
        """When both get_tournament and list_tournaments return nothing, 404."""
        nomic_dir = Path(handler_with_ctx.ctx["nomic_dir"])
        (nomic_dir / "tournaments").mkdir(parents=True, exist_ok=True)
        (nomic_dir / "tournaments" / "abc.db").write_text("")

        mock_manager = MagicMock()
        mock_manager.get_tournament.return_value = None
        mock_manager.list_tournaments.return_value = []

        with patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True), \
             patch("aragora.server.handlers.tournaments._TournamentManager", return_value=mock_manager):
            result = handler_with_ctx.handle("/api/tournaments/abc", {}, mock_http_handler)

        assert result.status_code == 404

    def test_get_versioned_path(self, handler_with_ctx, mock_http_handler):
        """Versioned path is handled correctly."""
        with patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True):
            result = handler_with_ctx.handle("/api/v1/tournaments/nonexistent", {}, mock_http_handler)
        assert result.status_code == 404


# ============================================================================
# GET /api/tournaments/{id}/standings
# ============================================================================


class TestGetStandings:
    """Tests for getting tournament standings."""

    @patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", False)
    def test_standings_unavailable(self, handler_with_ctx, mock_http_handler):
        h = TournamentHandler(ctx=handler_with_ctx.ctx)
        result = h.handle("/api/tournaments/abc/standings", {}, mock_http_handler)
        assert result.status_code == 503

    def test_standings_no_nomic_dir(self, mock_http_handler):
        h = TournamentHandler(ctx={})
        with patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True):
            result = h.handle("/api/tournaments/abc/standings", {}, mock_http_handler)
        assert result.status_code == 503

    def test_standings_not_found(self, handler_with_ctx, mock_http_handler):
        with patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True):
            result = handler_with_ctx.handle(
                "/api/tournaments/missing/standings", {}, mock_http_handler
            )
        assert result.status_code == 404

    def test_standings_success(self, handler_with_ctx, mock_http_handler):
        """Successfully retrieve standings."""
        nomic_dir = Path(handler_with_ctx.ctx["nomic_dir"])
        (nomic_dir / "tournaments").mkdir(parents=True, exist_ok=True)
        (nomic_dir / "tournaments" / "t1.db").write_text("")

        mock_standings = [
            MockStanding("agent_a", 5, 1, 1, 16.0, 20.0, 0.71),
            MockStanding("agent_b", 3, 3, 1, 10.0, 14.0, 0.43),
        ]
        mock_manager = MagicMock()
        mock_manager.get_current_standings.return_value = mock_standings

        with patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True), \
             patch("aragora.server.handlers.tournaments._TournamentManager", return_value=mock_manager):
            result = handler_with_ctx.handle(
                "/api/tournaments/t1/standings", {}, mock_http_handler
            )

        assert result.status_code == 200
        body = _parse_body(result)
        assert body["tournament_id"] == "t1"
        assert body["count"] == 2
        assert body["standings"][0]["agent"] == "agent_a"
        assert body["standings"][0]["wins"] == 5
        assert body["standings"][0]["losses"] == 1
        assert body["standings"][0]["draws"] == 1
        assert body["standings"][0]["points"] == 16.0
        assert body["standings"][0]["win_rate"] == 0.71
        assert body["standings"][1]["agent"] == "agent_b"


# ============================================================================
# GET /api/tournaments/{id}/bracket
# ============================================================================


class TestGetBracket:
    """Tests for getting tournament bracket."""

    @patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", False)
    def test_bracket_unavailable(self, handler_with_ctx, mock_http_handler):
        h = TournamentHandler(ctx=handler_with_ctx.ctx)
        result = h.handle("/api/tournaments/abc/bracket", {}, mock_http_handler)
        assert result.status_code == 503

    def test_bracket_no_nomic_dir(self, mock_http_handler):
        h = TournamentHandler(ctx={})
        with patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True):
            result = h.handle("/api/tournaments/abc/bracket", {}, mock_http_handler)
        assert result.status_code == 503

    def test_bracket_not_found(self, handler_with_ctx, mock_http_handler):
        with patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True):
            result = handler_with_ctx.handle(
                "/api/tournaments/missing/bracket", {}, mock_http_handler
            )
        assert result.status_code == 404

    def test_bracket_success(self, handler_with_ctx, mock_http_handler):
        """Successfully retrieve bracket with matches organized by round."""
        nomic_dir = Path(handler_with_ctx.ctx["nomic_dir"])
        (nomic_dir / "tournaments").mkdir(parents=True, exist_ok=True)
        (nomic_dir / "tournaments" / "t2.db").write_text("")

        mock_tournament = MockTournament(
            tournament_id="t2",
            name="Bracket Test",
            bracket_type="single_elimination",
            participants=["a", "b", "c", "d"],
            rounds_completed=1,
            total_rounds=2,
            status="active",
            created_at="2026-01-15T00:00:00",
        )
        mock_matches = [
            MockMatch("m1", 1, "a", "b", "a", 1.0, 0.0, "d1", "2026-01-15", "2026-01-15"),
            MockMatch("m2", 1, "c", "d", "c", 1.0, 0.0, "d2", "2026-01-15", "2026-01-15"),
            MockMatch("m3", 2, "a", "c", None, 0.0, 0.0, None, "2026-01-16", None),
        ]
        mock_manager = MagicMock()
        mock_manager.get_tournament.return_value = mock_tournament
        mock_manager.get_matches.return_value = mock_matches

        with patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True), \
             patch("aragora.server.handlers.tournaments._TournamentManager", return_value=mock_manager):
            result = handler_with_ctx.handle(
                "/api/tournaments/t2/bracket", {}, mock_http_handler
            )

        assert result.status_code == 200
        body = _parse_body(result)
        assert body["tournament_id"] == "t2"
        assert body["name"] == "Bracket Test"
        assert body["bracket_type"] == "single_elimination"
        assert body["total_rounds"] == 2
        assert "1" in body["rounds"]
        assert "2" in body["rounds"]
        assert len(body["rounds"]["1"]) == 2
        assert len(body["rounds"]["2"]) == 1
        # Round 1 matches are completed
        assert body["rounds"]["1"][0]["completed"] is True
        # Round 2 match is not completed
        assert body["rounds"]["2"][0]["completed"] is False

    def test_bracket_tournament_not_found_in_db(self, handler_with_ctx, mock_http_handler):
        """Bracket falls back to list_tournaments when get_tournament returns None."""
        nomic_dir = Path(handler_with_ctx.ctx["nomic_dir"])
        (nomic_dir / "tournaments").mkdir(parents=True, exist_ok=True)
        (nomic_dir / "tournaments" / "t3.db").write_text("")

        mock_tournament = MockTournament(
            tournament_id="t3_real",
            name="Fallback Bracket",
            bracket_type="round_robin",
            participants=["x", "y"],
            rounds_completed=0,
            total_rounds=1,
            status="pending",
            created_at="2026-02-01T00:00:00",
        )
        mock_manager = MagicMock()
        mock_manager.get_tournament.return_value = None
        mock_manager.list_tournaments.return_value = [mock_tournament]
        mock_manager.get_matches.return_value = []

        with patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True), \
             patch("aragora.server.handlers.tournaments._TournamentManager", return_value=mock_manager):
            result = handler_with_ctx.handle(
                "/api/tournaments/t3/bracket", {}, mock_http_handler
            )

        assert result.status_code == 200
        body = _parse_body(result)
        assert body["tournament_id"] == "t3_real"

    def test_bracket_no_tournament_at_all(self, handler_with_ctx, mock_http_handler):
        """Bracket returns 404 when neither get_tournament nor list returns anything."""
        nomic_dir = Path(handler_with_ctx.ctx["nomic_dir"])
        (nomic_dir / "tournaments").mkdir(parents=True, exist_ok=True)
        (nomic_dir / "tournaments" / "t4.db").write_text("")

        mock_manager = MagicMock()
        mock_manager.get_tournament.return_value = None
        mock_manager.list_tournaments.return_value = []

        with patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True), \
             patch("aragora.server.handlers.tournaments._TournamentManager", return_value=mock_manager):
            result = handler_with_ctx.handle(
                "/api/tournaments/t4/bracket", {}, mock_http_handler
            )

        assert result.status_code == 404


# ============================================================================
# GET /api/tournaments/{id}/matches
# ============================================================================


class TestGetMatches:
    """Tests for getting tournament matches."""

    @patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", False)
    def test_matches_unavailable(self, handler_with_ctx, mock_http_handler):
        h = TournamentHandler(ctx=handler_with_ctx.ctx)
        result = h.handle("/api/tournaments/abc/matches", {}, mock_http_handler)
        assert result.status_code == 503

    def test_matches_no_nomic_dir(self, mock_http_handler):
        h = TournamentHandler(ctx={})
        with patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True):
            result = h.handle("/api/tournaments/abc/matches", {}, mock_http_handler)
        assert result.status_code == 503

    def test_matches_not_found(self, handler_with_ctx, mock_http_handler):
        with patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True):
            result = handler_with_ctx.handle(
                "/api/tournaments/missing/matches", {}, mock_http_handler
            )
        assert result.status_code == 404

    def test_matches_success(self, handler_with_ctx, mock_http_handler):
        """Successfully retrieve match history."""
        nomic_dir = Path(handler_with_ctx.ctx["nomic_dir"])
        (nomic_dir / "tournaments").mkdir(parents=True, exist_ok=True)
        (nomic_dir / "tournaments" / "t5.db").write_text("")

        mock_matches = [
            MockMatch("m1", 1, "a", "b", "a", 1.0, 0.0, "d1", "2026-01-01", "2026-01-01"),
            MockMatch("m2", 1, "c", "d", None, 0.5, 0.5, None, "2026-01-01", "2026-01-01"),
        ]
        mock_manager = MagicMock()
        mock_manager.get_matches.return_value = mock_matches

        with patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True), \
             patch("aragora.server.handlers.tournaments._TournamentManager", return_value=mock_manager):
            result = handler_with_ctx.handle(
                "/api/tournaments/t5/matches", {}, mock_http_handler
            )

        assert result.status_code == 200
        body = _parse_body(result)
        assert body["tournament_id"] == "t5"
        assert body["count"] == 2
        assert body["matches"][0]["match_id"] == "m1"
        assert body["matches"][0]["winner"] == "a"
        assert body["matches"][1]["winner"] is None

    def test_matches_with_round_filter(self, handler_with_ctx, mock_http_handler):
        """Filter matches by round number."""
        nomic_dir = Path(handler_with_ctx.ctx["nomic_dir"])
        (nomic_dir / "tournaments").mkdir(parents=True, exist_ok=True)
        (nomic_dir / "tournaments" / "t6.db").write_text("")

        mock_matches = [
            MockMatch("m3", 2, "a", "c", "a", 1.0, 0.0, "d3", "2026-01-02", "2026-01-02"),
        ]
        mock_manager = MagicMock()
        mock_manager.get_matches.return_value = mock_matches

        with patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True), \
             patch("aragora.server.handlers.tournaments._TournamentManager", return_value=mock_manager):
            result = handler_with_ctx.handle(
                "/api/tournaments/t6/matches", {"round": "2"}, mock_http_handler
            )

        assert result.status_code == 200
        body = _parse_body(result)
        assert body["count"] == 1
        mock_manager.get_matches.assert_called_once_with(tournament_id="t6", round_num=2)

    def test_matches_empty_result(self, handler_with_ctx, mock_http_handler):
        """No matches returns empty list."""
        nomic_dir = Path(handler_with_ctx.ctx["nomic_dir"])
        (nomic_dir / "tournaments").mkdir(parents=True, exist_ok=True)
        (nomic_dir / "tournaments" / "t7.db").write_text("")

        mock_manager = MagicMock()
        mock_manager.get_matches.return_value = []

        with patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True), \
             patch("aragora.server.handlers.tournaments._TournamentManager", return_value=mock_manager):
            result = handler_with_ctx.handle(
                "/api/tournaments/t7/matches", {}, mock_http_handler
            )

        assert result.status_code == 200
        body = _parse_body(result)
        assert body["count"] == 0
        assert body["matches"] == []


# ============================================================================
# POST /api/tournaments (Create)
# ============================================================================


class TestCreateTournament:
    """Tests for creating tournaments."""

    @patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", False)
    def test_create_unavailable(self, handler_with_ctx, mock_http_handler):
        h = TournamentHandler(ctx=handler_with_ctx.ctx)
        result = h.handle_post(
            "/api/tournaments", {"name": "T", "participants": ["a", "b"]}, mock_http_handler
        )
        assert result.status_code == 503

    def test_create_no_nomic_dir(self, mock_http_handler):
        h = TournamentHandler(ctx={})
        with patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True):
            result = h.handle_post(
                "/api/tournaments", {"name": "T", "participants": ["a", "b"]}, mock_http_handler
            )
        assert result.status_code == 503

    def test_create_missing_name(self, handler_with_ctx, mock_http_handler):
        with patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True):
            result = handler_with_ctx.handle_post(
                "/api/tournaments", {"participants": ["a", "b"]}, mock_http_handler
            )
        assert result.status_code == 400
        body = _parse_body(result)
        assert "name" in body["error"].lower()

    def test_create_empty_name(self, handler_with_ctx, mock_http_handler):
        with patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True):
            result = handler_with_ctx.handle_post(
                "/api/tournaments", {"name": "", "participants": ["a", "b"]}, mock_http_handler
            )
        assert result.status_code == 400

    def test_create_no_participants(self, handler_with_ctx, mock_http_handler):
        with patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True):
            result = handler_with_ctx.handle_post(
                "/api/tournaments", {"name": "T"}, mock_http_handler
            )
        assert result.status_code == 400
        body = _parse_body(result)
        assert "participant" in body["error"].lower()

    def test_create_too_few_participants(self, handler_with_ctx, mock_http_handler):
        with patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True):
            result = handler_with_ctx.handle_post(
                "/api/tournaments", {"name": "T", "participants": ["a"]}, mock_http_handler
            )
        assert result.status_code == 400

    def test_create_empty_participants_list(self, handler_with_ctx, mock_http_handler):
        with patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True):
            result = handler_with_ctx.handle_post(
                "/api/tournaments", {"name": "T", "participants": []}, mock_http_handler
            )
        assert result.status_code == 400

    def test_create_invalid_bracket_type(self, handler_with_ctx, mock_http_handler):
        with patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True):
            result = handler_with_ctx.handle_post(
                "/api/tournaments",
                {"name": "T", "participants": ["a", "b"], "bracket_type": "invalid"},
                mock_http_handler,
            )
        assert result.status_code == 400
        body = _parse_body(result)
        assert "bracket_type" in body["error"]

    def test_create_success_round_robin(self, handler_with_ctx, mock_http_handler):
        """Successfully create a round_robin tournament."""
        mock_tournament = MockTournament(
            tournament_id="new_t",
            name="My Tournament",
            bracket_type="round_robin",
            participants=["a", "b", "c"],
            rounds_completed=0,
            total_rounds=3,
            status="pending",
            created_at="2026-02-20T12:00:00",
        )
        mock_manager = MagicMock()
        mock_manager.create_tournament.return_value = mock_tournament

        with patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True), \
             patch("aragora.server.handlers.tournaments._TournamentManager", return_value=mock_manager):
            result = handler_with_ctx.handle_post(
                "/api/tournaments",
                {"name": "My Tournament", "participants": ["a", "b", "c"]},
                mock_http_handler,
            )

        assert result.status_code == 201
        body = _parse_body(result)
        assert body["tournament_id"] == "new_t"
        assert body["name"] == "My Tournament"
        assert body["bracket_type"] == "round_robin"
        assert body["status"] == "pending"

    def test_create_success_single_elimination(self, handler_with_ctx, mock_http_handler):
        """Create with explicit single_elimination bracket type."""
        mock_tournament = MockTournament(
            tournament_id="se_t",
            name="Elimination",
            bracket_type="single_elimination",
            participants=["a", "b"],
            rounds_completed=0,
            total_rounds=1,
            status="pending",
            created_at="2026-02-20T12:00:00",
        )
        mock_manager = MagicMock()
        mock_manager.create_tournament.return_value = mock_tournament

        with patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True), \
             patch("aragora.server.handlers.tournaments._TournamentManager", return_value=mock_manager):
            result = handler_with_ctx.handle_post(
                "/api/tournaments",
                {
                    "name": "Elimination",
                    "participants": ["a", "b"],
                    "bracket_type": "single_elimination",
                },
                mock_http_handler,
            )

        assert result.status_code == 201
        body = _parse_body(result)
        assert body["bracket_type"] == "single_elimination"

    def test_create_success_double_elimination(self, handler_with_ctx, mock_http_handler):
        """Create with double_elimination bracket type."""
        mock_tournament = MockTournament(
            tournament_id="de_t",
            name="Double Elim",
            bracket_type="double_elimination",
            participants=["a", "b", "c", "d"],
            rounds_completed=0,
            total_rounds=3,
            status="pending",
            created_at="2026-02-20T12:00:00",
        )
        mock_manager = MagicMock()
        mock_manager.create_tournament.return_value = mock_tournament

        with patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True), \
             patch("aragora.server.handlers.tournaments._TournamentManager", return_value=mock_manager):
            result = handler_with_ctx.handle_post(
                "/api/tournaments",
                {
                    "name": "Double Elim",
                    "participants": ["a", "b", "c", "d"],
                    "bracket_type": "double_elimination",
                },
                mock_http_handler,
            )

        assert result.status_code == 201
        body = _parse_body(result)
        assert body["bracket_type"] == "double_elimination"

    def test_create_default_bracket_type(self, handler_with_ctx, mock_http_handler):
        """Default bracket_type is round_robin when not specified."""
        mock_tournament = MockTournament(
            tournament_id="def_t",
            name="Default",
            bracket_type="round_robin",
            participants=["a", "b"],
            rounds_completed=0,
            total_rounds=1,
            status="pending",
            created_at="2026-02-20T12:00:00",
        )
        mock_manager = MagicMock()
        mock_manager.create_tournament.return_value = mock_tournament

        with patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True), \
             patch("aragora.server.handlers.tournaments._TournamentManager", return_value=mock_manager):
            result = handler_with_ctx.handle_post(
                "/api/tournaments",
                {"name": "Default", "participants": ["a", "b"]},
                mock_http_handler,
            )

        assert result.status_code == 201
        mock_manager.create_tournament.assert_called_once_with(
            name="Default",
            participants=["a", "b"],
            bracket_type="round_robin",
        )

    def test_create_versioned_path(self, handler_with_ctx, mock_http_handler):
        """Versioned path works for POST."""
        with patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True):
            result = handler_with_ctx.handle_post(
                "/api/v1/tournaments",
                {"name": "T", "participants": ["a"]},
                mock_http_handler,
            )
        # Should fail validation (too few participants), but confirms routing works
        assert result.status_code == 400


# ============================================================================
# POST /api/tournaments/{id}/advance
# ============================================================================


class TestAdvanceTournament:
    """Tests for advancing tournament rounds."""

    @patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", False)
    def test_advance_unavailable(self, handler_with_ctx, mock_http_handler):
        h = TournamentHandler(ctx=handler_with_ctx.ctx)
        result = h.handle("/api/tournaments/abc/advance", {}, mock_http_handler)
        assert result.status_code == 503

    def test_advance_no_nomic_dir(self, mock_http_handler):
        h = TournamentHandler(ctx={})
        with patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True):
            result = h.handle("/api/tournaments/abc/advance", {}, mock_http_handler)
        assert result.status_code == 503

    def test_advance_not_found(self, handler_with_ctx, mock_http_handler):
        with patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True):
            result = handler_with_ctx.handle(
                "/api/tournaments/missing/advance", {}, mock_http_handler
            )
        assert result.status_code == 404

    def test_advance_success(self, handler_with_ctx, mock_http_handler):
        """Successfully advance tournament."""
        nomic_dir = Path(handler_with_ctx.ctx["nomic_dir"])
        (nomic_dir / "tournaments").mkdir(parents=True, exist_ok=True)
        (nomic_dir / "tournaments" / "t_adv.db").write_text("")

        mock_tournament = MockTournament(
            tournament_id="t_adv",
            name="Advance Test",
            bracket_type="single_elimination",
            participants=["a", "b", "c", "d"],
            rounds_completed=1,
            total_rounds=2,
            status="active",
            created_at="2026-01-01",
        )
        mock_manager = MagicMock()
        mock_manager.advance_round.return_value = True
        mock_manager.get_tournament.return_value = mock_tournament

        with patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True), \
             patch("aragora.server.handlers.tournaments._TournamentManager", return_value=mock_manager):
            result = handler_with_ctx.handle(
                "/api/tournaments/t_adv/advance", {}, mock_http_handler
            )

        assert result.status_code == 200
        body = _parse_body(result)
        assert body["success"] is True
        assert "next round" in body["message"]
        assert body["rounds_completed"] == 1

    def test_advance_cannot_advance(self, handler_with_ctx, mock_http_handler):
        """Cannot advance when round not finished."""
        nomic_dir = Path(handler_with_ctx.ctx["nomic_dir"])
        (nomic_dir / "tournaments").mkdir(parents=True, exist_ok=True)
        (nomic_dir / "tournaments" / "t_no.db").write_text("")

        mock_manager = MagicMock()
        mock_manager.advance_round.return_value = False

        with patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True), \
             patch("aragora.server.handlers.tournaments._TournamentManager", return_value=mock_manager):
            result = handler_with_ctx.handle(
                "/api/tournaments/t_no/advance", {}, mock_http_handler
            )

        assert result.status_code == 200
        body = _parse_body(result)
        assert body["success"] is False
        assert "Cannot advance" in body["message"]

    def test_advance_via_post(self, handler_with_ctx, mock_http_handler):
        """Advance also works via handle_post."""
        nomic_dir = Path(handler_with_ctx.ctx["nomic_dir"])
        (nomic_dir / "tournaments").mkdir(parents=True, exist_ok=True)
        (nomic_dir / "tournaments" / "t_post.db").write_text("")

        mock_manager = MagicMock()
        mock_manager.advance_round.return_value = False

        with patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True), \
             patch("aragora.server.handlers.tournaments._TournamentManager", return_value=mock_manager):
            result = handler_with_ctx.handle_post(
                "/api/tournaments/t_post/advance", {}, mock_http_handler
            )

        assert result.status_code == 200


# ============================================================================
# POST /api/tournaments/{id}/matches/{match_id}/result
# ============================================================================


class TestRecordMatchResult:
    """Tests for recording match results."""

    @patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", False)
    def test_result_unavailable(self, handler_with_ctx, mock_http_handler):
        h = TournamentHandler(ctx=handler_with_ctx.ctx)
        result = h.handle(
            "/api/tournaments/abc/matches/m1/result",
            {"winner": "a"},
            mock_http_handler,
        )
        assert result.status_code == 503

    def test_result_no_nomic_dir(self, mock_http_handler):
        h = TournamentHandler(ctx={})
        with patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True):
            result = h.handle(
                "/api/tournaments/abc/matches/m1/result",
                {"winner": "a"},
                mock_http_handler,
            )
        assert result.status_code == 503

    def test_result_not_found(self, handler_with_ctx, mock_http_handler):
        with patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True):
            result = handler_with_ctx.handle(
                "/api/tournaments/missing/matches/m1/result",
                {"winner": "a"},
                mock_http_handler,
            )
        assert result.status_code == 404

    def test_result_success_with_winner(self, handler_with_ctx, mock_http_handler):
        """Record a match result with a winner."""
        nomic_dir = Path(handler_with_ctx.ctx["nomic_dir"])
        (nomic_dir / "tournaments").mkdir(parents=True, exist_ok=True)
        (nomic_dir / "tournaments" / "t_res.db").write_text("")

        mock_manager = MagicMock()

        with patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True), \
             patch("aragora.server.handlers.tournaments._TournamentManager", return_value=mock_manager):
            result = handler_with_ctx.handle(
                "/api/tournaments/t_res/matches/m1/result",
                {"winner": "agent_a", "score1": "1.0", "score2": "0.5", "debate_id": "d99"},
                mock_http_handler,
            )

        assert result.status_code == 200
        body = _parse_body(result)
        assert body["success"] is True
        assert body["match_id"] == "m1"
        assert body["winner"] == "agent_a"
        assert body["score1"] == 1.0
        assert body["score2"] == 0.5
        mock_manager.record_match_result.assert_called_once_with(
            match_id="m1",
            winner="agent_a",
            score1=1.0,
            score2=0.5,
            debate_id="d99",
        )

    def test_result_draw(self, handler_with_ctx, mock_http_handler):
        """Record a draw (no winner)."""
        nomic_dir = Path(handler_with_ctx.ctx["nomic_dir"])
        (nomic_dir / "tournaments").mkdir(parents=True, exist_ok=True)
        (nomic_dir / "tournaments" / "t_draw.db").write_text("")

        mock_manager = MagicMock()

        with patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True), \
             patch("aragora.server.handlers.tournaments._TournamentManager", return_value=mock_manager):
            result = handler_with_ctx.handle(
                "/api/tournaments/t_draw/matches/m2/result",
                {"score1": "0.5", "score2": "0.5"},
                mock_http_handler,
            )

        assert result.status_code == 200
        body = _parse_body(result)
        assert body["winner"] is None

    def test_result_default_scores(self, handler_with_ctx, mock_http_handler):
        """Default scores are 0.0 when not provided."""
        nomic_dir = Path(handler_with_ctx.ctx["nomic_dir"])
        (nomic_dir / "tournaments").mkdir(parents=True, exist_ok=True)
        (nomic_dir / "tournaments" / "t_def.db").write_text("")

        mock_manager = MagicMock()

        with patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True), \
             patch("aragora.server.handlers.tournaments._TournamentManager", return_value=mock_manager):
            result = handler_with_ctx.handle(
                "/api/tournaments/t_def/matches/m3/result",
                {"winner": "agent_b"},
                mock_http_handler,
            )

        assert result.status_code == 200
        body = _parse_body(result)
        assert body["score1"] == 0.0
        assert body["score2"] == 0.0

    def test_result_via_post(self, handler_with_ctx, mock_http_handler):
        """Record result via handle_post."""
        nomic_dir = Path(handler_with_ctx.ctx["nomic_dir"])
        (nomic_dir / "tournaments").mkdir(parents=True, exist_ok=True)
        (nomic_dir / "tournaments" / "t_post_r.db").write_text("")

        mock_manager = MagicMock()

        with patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True), \
             patch("aragora.server.handlers.tournaments._TournamentManager", return_value=mock_manager):
            result = handler_with_ctx.handle_post(
                "/api/tournaments/t_post_r/matches/m5/result",
                {"winner": "x", "score1": "3.0", "score2": "1.0"},
                mock_http_handler,
            )

        assert result.status_code == 200
        body = _parse_body(result)
        assert body["match_id"] == "m5"
        assert body["winner"] == "x"


# ============================================================================
# Path Traversal Validation
# ============================================================================


class TestPathTraversalValidation:
    """Tests for path traversal prevention in tournament IDs."""

    def test_handle_rejects_double_dot(self, handler_with_ctx, mock_http_handler):
        with patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True):
            result = handler_with_ctx.handle(
                "/api/tournaments/../etc", {}, mock_http_handler
            )
        assert result.status_code == 400
        body = _parse_body(result)
        assert "Invalid" in body["error"]

    def test_handle_rejects_semicolon(self, handler_with_ctx, mock_http_handler):
        with patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True):
            result = handler_with_ctx.handle(
                "/api/tournaments/abc;rm+-rf", {}, mock_http_handler
            )
        assert result.status_code == 400

    def test_handle_post_rejects_double_dot(self, handler_with_ctx, mock_http_handler):
        with patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True):
            result = handler_with_ctx.handle_post(
                "/api/tournaments/../bad/advance", {}, mock_http_handler
            )
        assert result.status_code == 400

    def test_handle_post_rejects_semicolon(self, handler_with_ctx, mock_http_handler):
        with patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True):
            result = handler_with_ctx.handle_post(
                "/api/tournaments/bad;/advance", {}, mock_http_handler
            )
        assert result.status_code == 400


# ============================================================================
# Rate Limiting
# ============================================================================


class TestRateLimiting:
    """Tests for rate limiting on tournament endpoints."""

    def test_rate_limit_on_handle(self, handler_with_ctx, mock_http_handler):
        """Exceeding rate limit returns 429."""
        with patch.object(_tournament_limiter, "is_allowed", return_value=False):
            result = handler_with_ctx.handle("/api/tournaments", {}, mock_http_handler)
        assert result.status_code == 429
        body = _parse_body(result)
        assert "Rate limit" in body["error"]

    def test_rate_limit_on_handle_post(self, handler_with_ctx, mock_http_handler):
        """POST also rate limited."""
        with patch.object(_tournament_limiter, "is_allowed", return_value=False):
            result = handler_with_ctx.handle_post(
                "/api/tournaments",
                {"name": "T", "participants": ["a", "b"]},
                mock_http_handler,
            )
        assert result.status_code == 429

    def test_rate_limit_allows_normal_traffic(self, handler_with_ctx, mock_http_handler):
        """Normal traffic is not rate limited."""
        with patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True):
            result = handler_with_ctx.handle("/api/tournaments", {}, mock_http_handler)
        assert result.status_code != 429


# ============================================================================
# Handle routing (return None for non-matching)
# ============================================================================


class TestHandleRouting:
    """Tests for handle() routing returning None for non-matching paths."""

    def test_handle_non_tournament_returns_none(self, handler_with_ctx, mock_http_handler):
        result = handler_with_ctx.handle("/api/debates", {}, mock_http_handler)
        assert result is None

    def test_handle_post_non_matching_returns_400(self, handler_with_ctx, mock_http_handler):
        """POST to non-tournament path falls through to 'Tournament ID required'."""
        with patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True):
            result = handler_with_ctx.handle_post(
                "/api/debates",
                {"name": "T"},
                mock_http_handler,
            )
        # handle_post doesn't have an early exit for non-tournament paths,
        # so it falls through to the tournament_id check and returns 400
        assert result.status_code == 400

    def test_handle_post_unknown_sub_resource_returns_none(self, handler_with_ctx, mock_http_handler):
        """POST to /api/tournaments/{id}/unknown returns None (no matching sub-route)."""
        with patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True):
            result = handler_with_ctx.handle_post(
                "/api/tournaments/abc/unknown",
                {},
                mock_http_handler,
            )
        assert result is None


# ============================================================================
# Handler initialization
# ============================================================================


class TestHandlerInit:
    """Tests for handler initialization."""

    def test_init_with_context(self):
        ctx = {"nomic_dir": "/tmp/test"}
        h = TournamentHandler(ctx=ctx)
        assert h.ctx == ctx

    def test_init_without_context(self):
        h = TournamentHandler()
        assert h.ctx == {}

    def test_init_with_none_context(self):
        h = TournamentHandler(ctx=None)
        assert h.ctx == {}

    def test_routes_defined(self):
        """Handler has expected ROUTES attribute."""
        assert len(TournamentHandler.ROUTES) == 7
        assert "/api/tournaments" in TournamentHandler.ROUTES


# ============================================================================
# Exception handling via @handle_errors decorator
# ============================================================================


class TestExceptionHandling:
    """Tests for @handle_errors exception handling."""

    def test_manager_exception_caught(self, handler_with_ctx, mock_http_handler):
        """Exceptions from TournamentManager are caught by @handle_errors."""
        nomic_dir = Path(handler_with_ctx.ctx["nomic_dir"])
        (nomic_dir / "tournaments").mkdir(parents=True, exist_ok=True)
        (nomic_dir / "tournaments" / "err.db").write_text("")

        mock_manager = MagicMock()
        mock_manager.get_current_standings.side_effect = RuntimeError("database error")

        with patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True), \
             patch("aragora.server.handlers.tournaments._TournamentManager", return_value=mock_manager):
            result = handler_with_ctx.handle(
                "/api/tournaments/err/standings", {}, mock_http_handler
            )

        assert result.status_code == 500

    def test_bracket_exception_caught(self, handler_with_ctx, mock_http_handler):
        """Exception retrieving bracket is caught."""
        nomic_dir = Path(handler_with_ctx.ctx["nomic_dir"])
        (nomic_dir / "tournaments").mkdir(parents=True, exist_ok=True)
        (nomic_dir / "tournaments" / "err2.db").write_text("")

        mock_manager = MagicMock()
        mock_manager.get_tournament.side_effect = RuntimeError("bad data")

        with patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True), \
             patch("aragora.server.handlers.tournaments._TournamentManager", return_value=mock_manager):
            result = handler_with_ctx.handle(
                "/api/tournaments/err2/bracket", {}, mock_http_handler
            )

        assert result.status_code == 500

    def test_create_exception_caught(self, handler_with_ctx, mock_http_handler):
        """Exception during creation is caught by @handle_errors on handle_post."""
        mock_manager = MagicMock()
        mock_manager.create_tournament.side_effect = RuntimeError("creation failed")

        with patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True), \
             patch("aragora.server.handlers.tournaments._TournamentManager", return_value=mock_manager):
            result = handler_with_ctx.handle_post(
                "/api/tournaments",
                {"name": "T", "participants": ["a", "b"]},
                mock_http_handler,
            )

        assert result.status_code == 500

    def test_advance_exception_caught(self, handler_with_ctx, mock_http_handler):
        """Exception during advancement is caught."""
        nomic_dir = Path(handler_with_ctx.ctx["nomic_dir"])
        (nomic_dir / "tournaments").mkdir(parents=True, exist_ok=True)
        (nomic_dir / "tournaments" / "err3.db").write_text("")

        mock_manager = MagicMock()
        mock_manager.advance_round.side_effect = RuntimeError("advance failed")

        with patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True), \
             patch("aragora.server.handlers.tournaments._TournamentManager", return_value=mock_manager):
            result = handler_with_ctx.handle(
                "/api/tournaments/err3/advance", {}, mock_http_handler
            )

        assert result.status_code == 500

    def test_record_result_exception_caught(self, handler_with_ctx, mock_http_handler):
        """Exception during match result recording is caught."""
        nomic_dir = Path(handler_with_ctx.ctx["nomic_dir"])
        (nomic_dir / "tournaments").mkdir(parents=True, exist_ok=True)
        (nomic_dir / "tournaments" / "err4.db").write_text("")

        mock_manager = MagicMock()
        mock_manager.record_match_result.side_effect = RuntimeError("record failed")

        with patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True), \
             patch("aragora.server.handlers.tournaments._TournamentManager", return_value=mock_manager):
            result = handler_with_ctx.handle(
                "/api/tournaments/err4/matches/m1/result",
                {"winner": "a"},
                mock_http_handler,
            )

        assert result.status_code == 500
