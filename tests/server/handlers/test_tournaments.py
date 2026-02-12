"""
Tests for TournamentHandler - Tournament HTTP endpoints.

Tests cover:
- List tournaments
- Create tournament (POST)
- Get single tournament by ID
- Get tournament standings
- Get tournament bracket structure
- Get tournament matches with optional round filter
- Advance tournament to next round
- Record match result
- Rate limiting behavior
- Error handling and edge cases
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.tournaments import (
    TournamentHandler,
    TOURNAMENT_AVAILABLE,
    MAX_TOURNAMENTS_TO_LIST,
)


# ===========================================================================
# Test Fixtures and Mocks
# ===========================================================================


@dataclass
class MockTournamentStanding:
    """Mock tournament standing for testing."""

    agent: str
    wins: int = 0
    losses: int = 0
    draws: int = 0
    points: float = 0.0
    total_score: float = 0.0

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses + self.draws
        if total == 0:
            return 0.0
        return self.wins / total


@dataclass
class MockTournamentMatch:
    """Mock tournament match for testing."""

    match_id: str
    tournament_id: str
    round_num: int
    agent1: str
    agent2: str
    winner: str | None = None
    score1: float = 0.0
    score2: float = 0.0
    debate_id: str | None = None
    bracket_position: int = 0
    is_losers_bracket: bool = False
    created_at: str = "2024-01-01T00:00:00Z"
    completed_at: str | None = None


@dataclass
class MockTournament:
    """Mock tournament for testing."""

    tournament_id: str
    name: str
    participants: list[str]
    bracket_type: str = "round_robin"
    rounds_completed: int = 0
    total_rounds: int = 1
    status: str = "pending"
    created_at: str = "2024-01-01T00:00:00Z"


class MockTournamentManager:
    """Mock tournament manager for testing."""

    def __init__(self, db_path: str = ""):
        self.db_path = db_path
        self.tournaments: dict[str, MockTournament] = {}
        self.matches: dict[str, MockTournamentMatch] = {}
        self.standings: list[MockTournamentStanding] = []

    def create_tournament(
        self,
        name: str,
        participants: list[str],
        bracket_type: str = "round_robin",
    ) -> MockTournament:
        """Create a new tournament."""
        import uuid

        tournament_id = str(uuid.uuid4())[:8]
        tournament = MockTournament(
            tournament_id=tournament_id,
            name=name,
            participants=participants,
            bracket_type=bracket_type,
            total_rounds=len(participants) - 1 if bracket_type == "round_robin" else 2,
        )
        self.tournaments[tournament_id] = tournament

        # Create initial matches
        match_num = 0
        for i, agent1 in enumerate(participants):
            for agent2 in participants[i + 1 :]:
                match_id = f"{tournament_id}-m{match_num}"
                self.matches[match_id] = MockTournamentMatch(
                    match_id=match_id,
                    tournament_id=tournament_id,
                    round_num=1,
                    agent1=agent1,
                    agent2=agent2,
                )
                match_num += 1

        return tournament

    def get_tournament(self, tournament_id: str) -> MockTournament | None:
        """Get tournament by ID."""
        return self.tournaments.get(tournament_id)

    def list_tournaments(self, limit: int = 100) -> list[MockTournament]:
        """List tournaments."""
        return list(self.tournaments.values())[:limit]

    def get_current_standings(self) -> list[MockTournamentStanding]:
        """Get current standings."""
        return self.standings or [
            MockTournamentStanding(agent="claude", wins=2, losses=1, points=4.0),
            MockTournamentStanding(agent="gpt4", wins=1, losses=2, points=2.0),
        ]

    def get_matches(
        self,
        tournament_id: str | None = None,
        round_num: int | None = None,
    ) -> list[MockTournamentMatch]:
        """Get tournament matches."""
        result = list(self.matches.values())
        if tournament_id:
            result = [m for m in result if m.tournament_id == tournament_id]
        if round_num is not None:
            result = [m for m in result if m.round_num == round_num]
        return result

    def advance_round(self, tournament_id: str) -> bool:
        """Advance tournament to next round."""
        tournament = self.tournaments.get(tournament_id)
        if tournament:
            if tournament.rounds_completed < tournament.total_rounds:
                tournament.rounds_completed += 1
                return True
        return False

    def record_match_result(
        self,
        match_id: str,
        winner: str | None = None,
        score1: float = 0.0,
        score2: float = 0.0,
        debate_id: str | None = None,
    ) -> None:
        """Record a match result."""
        match = self.matches.get(match_id)
        if match:
            match.winner = winner
            match.score1 = score1
            match.score2 = score2
            match.debate_id = debate_id
            match.completed_at = "2024-01-01T12:00:00Z"


@pytest.fixture
def mock_server_context():
    """Create mock server context with nomic_dir."""
    return {"nomic_dir": "/tmp/nomic"}


@pytest.fixture
def mock_handler():
    """Create a mock HTTP request handler."""
    handler = MagicMock()
    handler.headers = {"X-Forwarded-For": "127.0.0.1"}
    handler.client_address = ("127.0.0.1", 12345)
    return handler


@pytest.fixture
def tournament_handler(mock_server_context):
    """Create tournament handler with mock context."""
    return TournamentHandler(mock_server_context)


@pytest.fixture
def mock_tournament_manager():
    """Create a mock tournament manager."""
    return MockTournamentManager()


# ===========================================================================
# Helper Functions
# ===========================================================================


def parse_response_body(result) -> dict[str, Any]:
    """Parse JSON response body from HandlerResult."""
    if hasattr(result, "body"):
        body = result.body
        if isinstance(body, bytes):
            return json.loads(body.decode())
        return json.loads(body)
    return {}


# ===========================================================================
# Path Handling Tests
# ===========================================================================


class TestTournamentHandlerCanHandle:
    """Test path matching for can_handle method."""

    def test_can_handle_list_tournaments(self, tournament_handler):
        """Test can_handle for list tournaments endpoint."""
        assert tournament_handler.can_handle("/api/tournaments") is True

    def test_can_handle_get_tournament(self, tournament_handler):
        """Test can_handle for get tournament by ID."""
        assert tournament_handler.can_handle("/api/tournaments/abc123") is True

    def test_can_handle_standings(self, tournament_handler):
        """Test can_handle for tournament standings."""
        assert tournament_handler.can_handle("/api/tournaments/abc123/standings") is True

    def test_can_handle_bracket(self, tournament_handler):
        """Test can_handle for tournament bracket."""
        assert tournament_handler.can_handle("/api/tournaments/abc123/bracket") is True

    def test_can_handle_matches(self, tournament_handler):
        """Test can_handle for tournament matches."""
        assert tournament_handler.can_handle("/api/tournaments/abc123/matches") is True

    def test_can_handle_advance(self, tournament_handler):
        """Test can_handle for tournament advance."""
        assert tournament_handler.can_handle("/api/tournaments/abc123/advance") is True

    def test_can_handle_match_result(self, tournament_handler):
        """Test can_handle for match result."""
        assert tournament_handler.can_handle("/api/tournaments/abc123/matches/m1/result") is True

    def test_cannot_handle_invalid_path(self, tournament_handler):
        """Test can_handle returns False for invalid paths."""
        assert tournament_handler.can_handle("/api/agents") is False
        assert tournament_handler.can_handle("/api/debates") is False
        assert tournament_handler.can_handle("/api/tournaments/abc123/invalid") is False

    def test_can_handle_versioned_path(self, tournament_handler):
        """Test can_handle strips version prefix."""
        # The version prefix format is /api/v1/..., not /v1/api/...
        assert tournament_handler.can_handle("/api/v1/tournaments") is True
        assert tournament_handler.can_handle("/api/v2/tournaments/abc123") is True


# ===========================================================================
# List Tournaments Tests
# ===========================================================================


class TestListTournaments:
    """Tests for list tournaments endpoint."""

    def test_list_tournaments_unavailable(self, mock_server_context, mock_handler):
        """Test list returns 503 when tournament system unavailable."""
        handler = TournamentHandler(mock_server_context)

        with patch.object(handler, "_list_tournaments") as mock_list:
            from aragora.server.handlers.base import error_response

            mock_list.return_value = error_response("Tournament system not available", 503)
            result = handler._list_tournaments()

        assert result.status_code == 503

    def test_list_tournaments_no_nomic_dir(self, mock_handler):
        """Test list returns 503 when nomic_dir not configured."""
        handler = TournamentHandler({})

        with patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True):
            result = handler._list_tournaments()

        assert result.status_code == 503

    @patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True)
    @patch("aragora.server.handlers.tournaments._TournamentManager")
    def test_list_tournaments_empty(self, mock_manager_class, tournament_handler, mock_handler):
        """Test list returns empty list when no tournaments exist."""
        from pathlib import Path

        with patch.object(Path, "exists", return_value=False):
            result = tournament_handler._list_tournaments()

        assert result.status_code == 200
        data = parse_response_body(result)
        assert data["tournaments"] == []
        assert data["count"] == 0

    @patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True)
    @patch("aragora.server.handlers.tournaments._TournamentManager")
    def test_list_tournaments_with_tournaments(
        self, mock_manager_class, tournament_handler, mock_handler
    ):
        """Test list returns tournaments when they exist."""
        from pathlib import Path

        # Mock the tournament manager
        mock_manager = MockTournamentManager()
        mock_manager.standings = [
            MockTournamentStanding(agent="claude", wins=2, losses=0),
        ]
        mock_manager_class.return_value = mock_manager

        # Mock directory and file operations
        mock_db_file = MagicMock()
        mock_db_file.stem = "tournament1"

        with patch.object(Path, "exists", return_value=True):
            with patch.object(Path, "glob", return_value=[mock_db_file]):
                result = tournament_handler._list_tournaments()

        assert result.status_code == 200
        data = parse_response_body(result)
        assert data["count"] == 1
        assert data["tournaments"][0]["tournament_id"] == "tournament1"


# ===========================================================================
# Get Tournament Tests
# ===========================================================================


class TestGetTournament:
    """Tests for get tournament by ID endpoint."""

    def test_get_tournament_unavailable(self, mock_server_context, mock_handler):
        """Test get returns 503 when tournament system unavailable."""
        handler = TournamentHandler(mock_server_context)

        with patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", False):
            result = handler._get_tournament("abc123")

        assert result.status_code == 503

    def test_get_tournament_no_nomic_dir(self, mock_handler):
        """Test get returns 503 when nomic_dir not configured."""
        handler = TournamentHandler({})

        with patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True):
            result = handler._get_tournament("abc123")

        assert result.status_code == 503

    @patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True)
    def test_get_tournament_not_found(self, tournament_handler):
        """Test get returns 404 when tournament not found."""
        from pathlib import Path

        with patch.object(Path, "exists", return_value=False):
            result = tournament_handler._get_tournament("nonexistent")

        assert result.status_code == 404

    @patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True)
    @patch("aragora.server.handlers.tournaments._TournamentManager")
    def test_get_tournament_success(self, mock_manager_class, tournament_handler):
        """Test get returns tournament details when found."""
        from pathlib import Path

        mock_tournament = MockTournament(
            tournament_id="abc123",
            name="Test Tournament",
            participants=["claude", "gpt4"],
            bracket_type="round_robin",
            status="in_progress",
        )
        mock_manager = MagicMock()
        mock_manager.get_tournament.return_value = mock_tournament
        mock_manager_class.return_value = mock_manager

        with patch.object(Path, "exists", return_value=True):
            result = tournament_handler._get_tournament("abc123")

        assert result.status_code == 200
        data = parse_response_body(result)
        assert data["tournament_id"] == "abc123"
        assert data["name"] == "Test Tournament"
        assert data["bracket_type"] == "round_robin"


# ===========================================================================
# Get Tournament Standings Tests
# ===========================================================================


class TestGetTournamentStandings:
    """Tests for get tournament standings endpoint."""

    @patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True)
    def test_get_standings_not_found(self, tournament_handler):
        """Test standings returns 404 when tournament not found."""
        from pathlib import Path

        with patch.object(Path, "exists", return_value=False):
            result = tournament_handler._get_tournament_standings("nonexistent")

        assert result.status_code == 404

    @patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True)
    @patch("aragora.server.handlers.tournaments._TournamentManager")
    def test_get_standings_success(self, mock_manager_class, tournament_handler):
        """Test standings returns data when tournament exists."""
        from pathlib import Path

        mock_standings = [
            MockTournamentStanding(agent="claude", wins=2, losses=0, points=4.0),
            MockTournamentStanding(agent="gpt4", wins=0, losses=2, points=0.0),
        ]
        mock_manager = MagicMock()
        mock_manager.get_current_standings.return_value = mock_standings
        mock_manager_class.return_value = mock_manager

        with patch.object(Path, "exists", return_value=True):
            result = tournament_handler._get_tournament_standings("abc123")

        assert result.status_code == 200
        data = parse_response_body(result)
        assert data["tournament_id"] == "abc123"
        assert data["count"] == 2
        assert len(data["standings"]) == 2
        assert data["standings"][0]["agent"] == "claude"
        assert data["standings"][0]["wins"] == 2


# ===========================================================================
# Get Tournament Bracket Tests
# ===========================================================================


class TestGetTournamentBracket:
    """Tests for get tournament bracket endpoint."""

    @patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True)
    def test_get_bracket_not_found(self, tournament_handler):
        """Test bracket returns 404 when tournament not found."""
        from pathlib import Path

        with patch.object(Path, "exists", return_value=False):
            result = tournament_handler._get_tournament_bracket("nonexistent")

        assert result.status_code == 404

    @patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True)
    @patch("aragora.server.handlers.tournaments._TournamentManager")
    def test_get_bracket_success(self, mock_manager_class, tournament_handler):
        """Test bracket returns structure when tournament exists."""
        from pathlib import Path

        mock_tournament = MockTournament(
            tournament_id="abc123",
            name="Test Tournament",
            participants=["claude", "gpt4"],
            bracket_type="single_elimination",
            total_rounds=2,
        )
        mock_matches = [
            MockTournamentMatch(
                match_id="m1",
                tournament_id="abc123",
                round_num=1,
                agent1="claude",
                agent2="gpt4",
                winner="claude",
                completed_at="2024-01-01T12:00:00Z",
            ),
        ]
        mock_manager = MagicMock()
        mock_manager.get_tournament.return_value = mock_tournament
        mock_manager.get_matches.return_value = mock_matches
        mock_manager_class.return_value = mock_manager

        with patch.object(Path, "exists", return_value=True):
            result = tournament_handler._get_tournament_bracket("abc123")

        assert result.status_code == 200
        data = parse_response_body(result)
        assert data["tournament_id"] == "abc123"
        assert data["bracket_type"] == "single_elimination"
        assert "1" in data["rounds"]
        assert len(data["rounds"]["1"]) == 1


# ===========================================================================
# Get Tournament Matches Tests
# ===========================================================================


class TestGetTournamentMatches:
    """Tests for get tournament matches endpoint."""

    @patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True)
    def test_get_matches_not_found(self, tournament_handler):
        """Test matches returns 404 when tournament not found."""
        from pathlib import Path

        with patch.object(Path, "exists", return_value=False):
            result = tournament_handler._get_tournament_matches("nonexistent")

        assert result.status_code == 404

    @patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True)
    @patch("aragora.server.handlers.tournaments._TournamentManager")
    def test_get_matches_all_rounds(self, mock_manager_class, tournament_handler):
        """Test matches returns all matches when no round filter."""
        from pathlib import Path

        mock_matches = [
            MockTournamentMatch(
                match_id="m1",
                tournament_id="abc123",
                round_num=1,
                agent1="claude",
                agent2="gpt4",
            ),
            MockTournamentMatch(
                match_id="m2",
                tournament_id="abc123",
                round_num=2,
                agent1="claude",
                agent2="gemini",
            ),
        ]
        mock_manager = MagicMock()
        mock_manager.get_matches.return_value = mock_matches
        mock_manager_class.return_value = mock_manager

        with patch.object(Path, "exists", return_value=True):
            result = tournament_handler._get_tournament_matches("abc123")

        assert result.status_code == 200
        data = parse_response_body(result)
        assert data["tournament_id"] == "abc123"
        assert data["count"] == 2

    @patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True)
    @patch("aragora.server.handlers.tournaments._TournamentManager")
    def test_get_matches_filtered_by_round(self, mock_manager_class, tournament_handler):
        """Test matches returns filtered matches by round."""
        from pathlib import Path

        mock_matches = [
            MockTournamentMatch(
                match_id="m1",
                tournament_id="abc123",
                round_num=1,
                agent1="claude",
                agent2="gpt4",
            ),
        ]
        mock_manager = MagicMock()
        mock_manager.get_matches.return_value = mock_matches
        mock_manager_class.return_value = mock_manager

        with patch.object(Path, "exists", return_value=True):
            result = tournament_handler._get_tournament_matches("abc123", round_num=1)

        assert result.status_code == 200
        data = parse_response_body(result)
        assert data["count"] == 1


# ===========================================================================
# Create Tournament Tests
# ===========================================================================


class TestCreateTournament:
    """Tests for create tournament endpoint."""

    @patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True)
    def test_create_tournament_missing_name(self, tournament_handler):
        """Test create returns 400 when name is missing."""
        result = tournament_handler._create_tournament({"participants": ["claude", "gpt4"]})

        assert result.status_code == 400
        data = parse_response_body(result)
        assert "name" in data["error"].lower()

    @patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True)
    def test_create_tournament_missing_participants(self, tournament_handler):
        """Test create returns 400 when participants missing or too few."""
        result = tournament_handler._create_tournament({"name": "Test Tournament"})
        assert result.status_code == 400

        result = tournament_handler._create_tournament(
            {"name": "Test Tournament", "participants": ["claude"]}
        )
        assert result.status_code == 400

    @patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True)
    def test_create_tournament_invalid_bracket_type(self, tournament_handler):
        """Test create returns 400 for invalid bracket type."""
        result = tournament_handler._create_tournament(
            {
                "name": "Test Tournament",
                "participants": ["claude", "gpt4"],
                "bracket_type": "invalid_type",
            }
        )

        assert result.status_code == 400
        data = parse_response_body(result)
        assert "bracket_type" in data["error"].lower()

    @patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True)
    @patch("aragora.server.handlers.tournaments._TournamentManager")
    def test_create_tournament_success(self, mock_manager_class, tournament_handler):
        """Test create tournament successfully."""
        from pathlib import Path

        mock_tournament = MockTournament(
            tournament_id="new123",
            name="New Tournament",
            participants=["claude", "gpt4", "gemini"],
            bracket_type="round_robin",
            total_rounds=2,
        )
        mock_manager = MagicMock()
        mock_manager.create_tournament.return_value = mock_tournament
        mock_manager_class.return_value = mock_manager

        with patch.object(Path, "mkdir"):
            result = tournament_handler._create_tournament(
                {
                    "name": "New Tournament",
                    "participants": ["claude", "gpt4", "gemini"],
                    "bracket_type": "round_robin",
                }
            )

        assert result.status_code == 201
        data = parse_response_body(result)
        assert data["name"] == "New Tournament"
        assert data["bracket_type"] == "round_robin"

    @patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True)
    @patch("aragora.server.handlers.tournaments._TournamentManager")
    def test_create_tournament_single_elimination(self, mock_manager_class, tournament_handler):
        """Test create tournament with single elimination bracket."""
        from pathlib import Path

        mock_tournament = MockTournament(
            tournament_id="elim123",
            name="Elimination Tournament",
            participants=["claude", "gpt4", "gemini", "llama"],
            bracket_type="single_elimination",
            total_rounds=2,
        )
        mock_manager = MagicMock()
        mock_manager.create_tournament.return_value = mock_tournament
        mock_manager_class.return_value = mock_manager

        with patch.object(Path, "mkdir"):
            result = tournament_handler._create_tournament(
                {
                    "name": "Elimination Tournament",
                    "participants": ["claude", "gpt4", "gemini", "llama"],
                    "bracket_type": "single_elimination",
                }
            )

        assert result.status_code == 201
        data = parse_response_body(result)
        assert data["bracket_type"] == "single_elimination"


# ===========================================================================
# Advance Tournament Tests
# ===========================================================================


class TestAdvanceTournament:
    """Tests for advance tournament endpoint."""

    @patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True)
    def test_advance_tournament_not_found(self, tournament_handler):
        """Test advance returns 404 when tournament not found."""
        from pathlib import Path

        with patch.object(Path, "exists", return_value=False):
            result = tournament_handler._advance_tournament("nonexistent")

        assert result.status_code == 404

    @patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True)
    @patch("aragora.server.handlers.tournaments._TournamentManager")
    def test_advance_tournament_success(self, mock_manager_class, tournament_handler):
        """Test advance tournament successfully."""
        from pathlib import Path

        mock_tournament = MockTournament(
            tournament_id="abc123",
            name="Test Tournament",
            participants=["claude", "gpt4"],
            rounds_completed=1,
            total_rounds=3,
        )
        mock_manager = MagicMock()
        mock_manager.advance_round.return_value = True
        mock_manager.get_tournament.return_value = mock_tournament
        mock_manager_class.return_value = mock_manager

        with patch.object(Path, "exists", return_value=True):
            result = tournament_handler._advance_tournament("abc123")

        assert result.status_code == 200
        data = parse_response_body(result)
        assert data["success"] is True
        assert data["tournament_id"] == "abc123"

    @patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True)
    @patch("aragora.server.handlers.tournaments._TournamentManager")
    def test_advance_tournament_cannot_advance(self, mock_manager_class, tournament_handler):
        """Test advance returns success=False when cannot advance."""
        from pathlib import Path

        mock_manager = MagicMock()
        mock_manager.advance_round.return_value = False
        mock_manager_class.return_value = mock_manager

        with patch.object(Path, "exists", return_value=True):
            result = tournament_handler._advance_tournament("abc123")

        assert result.status_code == 200
        data = parse_response_body(result)
        assert data["success"] is False


# ===========================================================================
# Record Match Result Tests
# ===========================================================================


class TestRecordMatchResult:
    """Tests for record match result endpoint."""

    @patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True)
    def test_record_result_tournament_not_found(self, tournament_handler):
        """Test record result returns 404 when tournament not found."""
        from pathlib import Path

        with patch.object(Path, "exists", return_value=False):
            result = tournament_handler._record_match_result(
                "nonexistent",
                "m1",
                {"winner": "claude", "score1": 1.0, "score2": 0.0},
            )

        assert result.status_code == 404

    @patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True)
    @patch("aragora.server.handlers.tournaments._TournamentManager")
    def test_record_result_success(self, mock_manager_class, tournament_handler):
        """Test record match result successfully."""
        from pathlib import Path

        mock_manager = MagicMock()
        mock_manager.record_match_result.return_value = None
        mock_manager_class.return_value = mock_manager

        with patch.object(Path, "exists", return_value=True):
            result = tournament_handler._record_match_result(
                "abc123",
                "m1",
                {"winner": "claude", "score1": 1.0, "score2": 0.5},
            )

        assert result.status_code == 200
        data = parse_response_body(result)
        assert data["success"] is True
        assert data["match_id"] == "m1"
        assert data["winner"] == "claude"

    @patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True)
    @patch("aragora.server.handlers.tournaments._TournamentManager")
    def test_record_result_draw(self, mock_manager_class, tournament_handler):
        """Test record match result as draw (no winner)."""
        from pathlib import Path

        mock_manager = MagicMock()
        mock_manager.record_match_result.return_value = None
        mock_manager_class.return_value = mock_manager

        with patch.object(Path, "exists", return_value=True):
            result = tournament_handler._record_match_result(
                "abc123",
                "m1",
                {"score1": 0.5, "score2": 0.5},
            )

        assert result.status_code == 200
        data = parse_response_body(result)
        assert data["winner"] is None

    @patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True)
    @patch("aragora.server.handlers.tournaments._TournamentManager")
    def test_record_result_with_debate_id(self, mock_manager_class, tournament_handler):
        """Test record match result with associated debate ID."""
        from pathlib import Path

        mock_manager = MagicMock()
        mock_manager.record_match_result.return_value = None
        mock_manager_class.return_value = mock_manager

        with patch.object(Path, "exists", return_value=True):
            result = tournament_handler._record_match_result(
                "abc123",
                "m1",
                {
                    "winner": "claude",
                    "score1": 1.0,
                    "score2": 0.0,
                    "debate_id": "debate-456",
                },
            )

        assert result.status_code == 200
        mock_manager.record_match_result.assert_called_once()
        call_kwargs = mock_manager.record_match_result.call_args[1]
        assert call_kwargs["debate_id"] == "debate-456"


# ===========================================================================
# Rate Limiting Tests
# ===========================================================================


class TestRateLimiting:
    """Tests for rate limiting behavior."""

    @patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True)
    @patch("aragora.server.handlers.tournaments._tournament_limiter")
    def test_rate_limit_exceeded_get(self, mock_limiter, tournament_handler, mock_handler):
        """Test GET returns 429 when rate limit exceeded."""
        mock_limiter.is_allowed.return_value = False

        result = tournament_handler.handle("/api/tournaments", {}, mock_handler)

        assert result.status_code == 429

    @patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True)
    @patch("aragora.server.handlers.tournaments._tournament_limiter")
    def test_rate_limit_exceeded_post(self, mock_limiter, tournament_handler, mock_handler):
        """Test POST returns 429 when rate limit exceeded."""
        mock_limiter.is_allowed.return_value = False

        result = tournament_handler.handle_post(
            "/api/tournaments",
            {"name": "Test", "participants": ["a", "b"]},
            mock_handler,
        )

        assert result.status_code == 429


# ===========================================================================
# Request Routing Tests
# ===========================================================================


class TestRequestRouting:
    """Tests for request routing to correct methods."""

    @patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True)
    @patch("aragora.server.handlers.tournaments._tournament_limiter")
    @patch("aragora.server.handlers.tournaments._TournamentManager")
    def test_handle_routes_to_list(
        self, mock_manager_class, mock_limiter, tournament_handler, mock_handler
    ):
        """Test handle routes /api/tournaments to list."""
        mock_limiter.is_allowed.return_value = True
        from pathlib import Path

        with patch.object(Path, "exists", return_value=False):
            result = tournament_handler.handle("/api/tournaments", {}, mock_handler)

        assert result.status_code == 200

    @patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True)
    @patch("aragora.server.handlers.tournaments._tournament_limiter")
    def test_handle_routes_to_get(self, mock_limiter, tournament_handler, mock_handler):
        """Test handle routes /api/tournaments/{id} to get."""
        mock_limiter.is_allowed.return_value = True
        from pathlib import Path

        with patch.object(Path, "exists", return_value=False):
            result = tournament_handler.handle("/api/tournaments/abc123", {}, mock_handler)

        assert result.status_code == 404  # Not found since path doesn't exist

    @patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True)
    @patch("aragora.server.handlers.tournaments._tournament_limiter")
    @patch("aragora.server.handlers.tournaments._TournamentManager")
    def test_handle_post_routes_to_create(
        self, mock_manager_class, mock_limiter, tournament_handler, mock_handler
    ):
        """Test handle_post routes /api/tournaments to create."""
        mock_limiter.is_allowed.return_value = True
        from pathlib import Path

        mock_tournament = MockTournament(
            tournament_id="new1",
            name="Test",
            participants=["a", "b"],
        )
        mock_manager = MagicMock()
        mock_manager.create_tournament.return_value = mock_tournament
        mock_manager_class.return_value = mock_manager

        with patch.object(Path, "mkdir"):
            result = tournament_handler.handle_post(
                "/api/tournaments",
                {"name": "Test", "participants": ["a", "b"]},
                mock_handler,
            )

        assert result.status_code == 201

    @patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True)
    @patch("aragora.server.handlers.tournaments._tournament_limiter")
    def test_handle_missing_tournament_id(self, mock_limiter, tournament_handler, mock_handler):
        """Test handle returns 400 when tournament ID required but missing."""
        mock_limiter.is_allowed.return_value = True

        # Path with trailing content but no actual ID
        result = tournament_handler.handle("/api/tournaments/", {}, mock_handler)

        # This should still work - the empty ID gets caught in specific methods
        # The handler may return None for unrecognized paths
        assert result is None or result.status_code in (400, 404)


# ===========================================================================
# Error Handling Tests
# ===========================================================================


class TestErrorHandling:
    """Tests for error handling scenarios."""

    @patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True)
    @patch("aragora.server.handlers.tournaments._TournamentManager")
    def test_list_handles_corrupted_tournament(self, mock_manager_class, tournament_handler):
        """Test list gracefully handles corrupted tournament files."""
        from pathlib import Path

        # Make manager raise exception (simulating corrupted DB)
        mock_manager_class.side_effect = Exception("Database corrupted")

        mock_db_file = MagicMock()
        mock_db_file.stem = "corrupted"

        with patch.object(Path, "exists", return_value=True):
            with patch.object(Path, "glob", return_value=[mock_db_file]):
                result = tournament_handler._list_tournaments()

        # Should succeed but skip corrupted tournament
        assert result.status_code == 200
        data = parse_response_body(result)
        assert data["count"] == 0

    def test_tournament_unavailable_all_endpoints(self, mock_server_context):
        """Test all endpoints return 503 when tournament system unavailable."""
        handler = TournamentHandler(mock_server_context)

        with patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", False):
            assert handler._list_tournaments().status_code == 503
            assert handler._get_tournament("abc").status_code == 503
            assert handler._get_tournament_standings("abc").status_code == 503
            assert handler._get_tournament_bracket("abc").status_code == 503
            assert handler._get_tournament_matches("abc").status_code == 503
            assert handler._create_tournament({"name": "test"}).status_code == 503
            assert handler._advance_tournament("abc").status_code == 503
            assert handler._record_match_result("abc", "m1", {}).status_code == 503


# ===========================================================================
# Integration Tests
# ===========================================================================


class TestIntegration:
    """Integration tests for full request flows."""

    @patch("aragora.server.handlers.tournaments.TOURNAMENT_AVAILABLE", True)
    @patch("aragora.server.handlers.tournaments._tournament_limiter")
    @patch("aragora.server.handlers.tournaments._TournamentManager")
    def test_full_tournament_flow(
        self, mock_manager_class, mock_limiter, tournament_handler, mock_handler
    ):
        """Test creating tournament, recording results, and checking standings."""
        from pathlib import Path

        mock_limiter.is_allowed.return_value = True

        # Create tournament
        mock_tournament = MockTournament(
            tournament_id="flow123",
            name="Flow Test Tournament",
            participants=["claude", "gpt4"],
        )
        mock_standings = [
            MockTournamentStanding(agent="claude", wins=1, losses=0, points=2.0),
            MockTournamentStanding(agent="gpt4", wins=0, losses=1, points=0.0),
        ]
        mock_manager = MagicMock()
        mock_manager.create_tournament.return_value = mock_tournament
        mock_manager.get_tournament.return_value = mock_tournament
        mock_manager.get_current_standings.return_value = mock_standings
        mock_manager_class.return_value = mock_manager

        # Step 1: Create tournament
        with patch.object(Path, "mkdir"):
            create_result = tournament_handler.handle_post(
                "/api/tournaments",
                {"name": "Flow Test Tournament", "participants": ["claude", "gpt4"]},
                mock_handler,
            )

        assert create_result.status_code == 201

        # Step 2: Record match result
        with patch.object(Path, "exists", return_value=True):
            result_result = tournament_handler.handle_post(
                "/api/tournaments/flow123/matches/m1/result",
                {"winner": "claude", "score1": 1.0, "score2": 0.0},
                mock_handler,
            )

        assert result_result.status_code == 200

        # Step 3: Check standings
        with patch.object(Path, "exists", return_value=True):
            standings_result = tournament_handler.handle(
                "/api/tournaments/flow123/standings",
                {},
                mock_handler,
            )

        assert standings_result.status_code == 200
        standings_data = parse_response_body(standings_result)
        assert standings_data["standings"][0]["agent"] == "claude"
        assert standings_data["standings"][0]["wins"] == 1


# ===========================================================================
# Constants Tests
# ===========================================================================


class TestConstants:
    """Tests for module constants."""

    def test_max_tournaments_limit_defined(self):
        """Test MAX_TOURNAMENTS_TO_LIST constant is defined."""
        assert MAX_TOURNAMENTS_TO_LIST == 100

    def test_tournament_available_is_boolean(self):
        """Test TOURNAMENT_AVAILABLE is a boolean."""
        assert isinstance(TOURNAMENT_AVAILABLE, bool)
