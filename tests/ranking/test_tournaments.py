"""
Tests for Tournament Management System.

Tests cover:
- Tournament creation and validation
- Match recording and results
- Standings calculation
- Round-robin bracket generation
- Single elimination brackets
- Double elimination brackets
- Tournament history tracking
- State transitions
- Error handling
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from aragora.ranking.tournaments import (
    MAX_PARTICIPANTS,
    MAX_TOURNAMENT_NAME_LENGTH,
    MIN_PARTICIPANTS,
    InvalidStateError,
    MatchNotFoundError,
    Tournament,
    TournamentEvent,
    TournamentHistoryEntry,
    TournamentManager,
    TournamentMatch,
    TournamentNotFoundError,
    TournamentStanding,
    TournamentStatus,
    ValidationError,
)


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_tournaments.db"
        yield str(db_path)


@pytest.fixture
def manager(temp_db):
    """Create a TournamentManager with temporary database."""
    return TournamentManager(db_path=temp_db)


class TestTournamentStanding:
    """Tests for TournamentStanding dataclass."""

    def test_standing_defaults(self):
        """Test default values for standing."""
        standing = TournamentStanding(agent="alice")
        assert standing.wins == 0
        assert standing.losses == 0
        assert standing.draws == 0
        assert standing.points == 0.0
        assert standing.total_score == 0.0

    def test_win_rate_no_games(self):
        """Test win rate with no games played."""
        standing = TournamentStanding(agent="alice")
        assert standing.win_rate == 0.0

    def test_win_rate_calculation(self):
        """Test win rate calculation."""
        standing = TournamentStanding(agent="alice", wins=3, losses=1, draws=1)
        # 3 wins out of 5 games = 60%
        assert standing.win_rate == pytest.approx(0.6, rel=1e-6)

    def test_win_rate_all_wins(self):
        """Test win rate with all wins."""
        standing = TournamentStanding(agent="alice", wins=5, losses=0, draws=0)
        assert standing.win_rate == pytest.approx(1.0, rel=1e-6)


class TestTournamentMatch:
    """Tests for TournamentMatch dataclass."""

    def test_match_defaults(self):
        """Test default values for match."""
        match = TournamentMatch(
            match_id="m1",
            tournament_id="t1",
            round_num=1,
            agent1="alice",
            agent2="bob",
        )
        assert match.winner is None
        assert match.score1 == 0.0
        assert match.score2 == 0.0
        assert match.debate_id is None
        assert match.bracket_position == 0
        assert match.is_losers_bracket is False
        assert match.completed_at is None


class TestTournamentHistoryEntry:
    """Tests for TournamentHistoryEntry dataclass."""

    def test_to_dict(self):
        """Test serialization to dictionary."""
        entry = TournamentHistoryEntry(
            entry_id="e1",
            tournament_id="t1",
            event_type=TournamentEvent.CREATED,
            timestamp="2024-01-01T00:00:00",
            details={"name": "Test"},
        )
        result = entry.to_dict()

        assert result["entry_id"] == "e1"
        assert result["tournament_id"] == "t1"
        assert result["event_type"] == "created"
        assert result["details"]["name"] == "Test"


class TestTournamentValidation:
    """Tests for tournament input validation."""

    def test_validate_empty_name(self, manager):
        """Test validation of empty tournament name."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            manager.create_tournament(name="", participants=["a", "b"])

    def test_validate_whitespace_name(self, manager):
        """Test validation of whitespace-only name."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            manager.create_tournament(name="   ", participants=["a", "b"])

    def test_validate_name_too_long(self, manager):
        """Test validation of overly long name."""
        long_name = "a" * (MAX_TOURNAMENT_NAME_LENGTH + 1)
        with pytest.raises(ValidationError, match="exceeds"):
            manager.create_tournament(name=long_name, participants=["a", "b"])

    def test_validate_empty_participants(self, manager):
        """Test validation of empty participants list."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            manager.create_tournament(name="Test", participants=[])

    def test_validate_too_few_participants(self, manager):
        """Test validation with too few participants."""
        with pytest.raises(ValidationError, match=f"at least {MIN_PARTICIPANTS}"):
            manager.create_tournament(name="Test", participants=["solo"])

    def test_validate_too_many_participants(self, manager):
        """Test validation with too many participants."""
        many_participants = [f"agent_{i}" for i in range(MAX_PARTICIPANTS + 1)]
        with pytest.raises(ValidationError, match=f"Cannot exceed {MAX_PARTICIPANTS}"):
            manager.create_tournament(name="Test", participants=many_participants)

    def test_validate_invalid_participant_name(self, manager):
        """Test validation of invalid participant name."""
        with pytest.raises(ValidationError, match="Invalid participant name"):
            manager.create_tournament(name="Test", participants=["valid", "invalid name!"])

    def test_validate_duplicate_participants(self, manager):
        """Test validation of duplicate participants."""
        with pytest.raises(ValidationError, match="Duplicate participant"):
            manager.create_tournament(name="Test", participants=["alice", "bob", "alice"])

    def test_validate_invalid_bracket_type(self, manager):
        """Test validation of invalid bracket type."""
        with pytest.raises(ValidationError, match="Invalid bracket type"):
            manager.create_tournament(
                name="Test",
                participants=["alice", "bob"],
                bracket_type="invalid_bracket",
            )


class TestTournamentCreation:
    """Tests for tournament creation."""

    def test_create_round_robin_tournament(self, manager):
        """Test creating a round-robin tournament."""
        tournament = manager.create_tournament(
            name="Test RR",
            participants=["alice", "bob", "charlie"],
            bracket_type="round_robin",
        )

        assert tournament.name == "Test RR"
        assert tournament.bracket_type == "round_robin"
        assert len(tournament.participants) == 3
        assert tournament.status == TournamentStatus.PENDING.value
        # Round-robin: n-1 rounds for n participants
        assert tournament.total_rounds == 2

    def test_create_single_elimination_tournament(self, manager):
        """Test creating a single elimination tournament."""
        tournament = manager.create_tournament(
            name="Test SE",
            participants=["a", "b", "c", "d"],
            bracket_type="single_elimination",
        )

        assert tournament.bracket_type == "single_elimination"
        # 4 participants = 2 rounds (log2(4) = 2)
        assert tournament.total_rounds == 2

    def test_create_double_elimination_tournament(self, manager):
        """Test creating a double elimination tournament."""
        tournament = manager.create_tournament(
            name="Test DE",
            participants=["a", "b", "c", "d"],
            bracket_type="double_elimination",
        )

        assert tournament.bracket_type == "double_elimination"
        # Double elimination: 2 * log2(n) rounds
        assert tournament.total_rounds == 4

    def test_tournament_id_generated(self, manager):
        """Test that tournament ID is auto-generated."""
        tournament = manager.create_tournament(
            name="Test",
            participants=["alice", "bob"],
        )

        assert tournament.tournament_id.startswith("tournament_")
        assert len(tournament.tournament_id) > len("tournament_")

    def test_tournament_logged_in_history(self, manager):
        """Test that tournament creation is logged."""
        tournament = manager.create_tournament(
            name="Test",
            participants=["alice", "bob"],
        )

        history = manager.get_tournament_history(tournament.tournament_id)
        assert len(history) >= 1
        assert history[-1].event_type == TournamentEvent.CREATED


class TestMatchGeneration:
    """Tests for match generation."""

    def test_round_robin_generates_correct_matches(self, manager):
        """Test that round-robin generates all pairs."""
        tournament = manager.create_tournament(
            name="Test",
            participants=["a", "b", "c"],
            bracket_type="round_robin",
        )

        matches = manager.get_matches(tournament_id=tournament.tournament_id)

        # 3 participants = 3 matches (a-b, a-c, b-c)
        assert len(matches) == 3

        # Verify all pairs exist
        pairs = set()
        for m in matches:
            pairs.add(frozenset([m.agent1, m.agent2]))
        assert pairs == {
            frozenset(["a", "b"]),
            frozenset(["a", "c"]),
            frozenset(["b", "c"]),
        }

    def test_single_elimination_generates_first_round(self, manager):
        """Test single elimination generates first round matches."""
        tournament = manager.create_tournament(
            name="Test",
            participants=["a", "b", "c", "d"],
            bracket_type="single_elimination",
        )

        matches = manager.get_matches(tournament_id=tournament.tournament_id, round_num=1)

        # 4 participants = 2 first round matches
        assert len(matches) == 2

    def test_single_elimination_pads_with_byes(self, manager):
        """Test single elimination adds BYEs for non-power-of-2."""
        tournament = manager.create_tournament(
            name="Test",
            participants=["a", "b", "c"],  # 3 participants
            bracket_type="single_elimination",
        )

        matches = manager.get_matches(tournament_id=tournament.tournament_id, round_num=1)

        # Padded to 4, so 2 matches with one BYE
        assert len(matches) == 2
        all_agents = [m.agent1 for m in matches] + [m.agent2 for m in matches]
        assert "BYE" in all_agents


class TestMatchRecording:
    """Tests for recording match results."""

    def test_record_match_result(self, manager):
        """Test recording a match result."""
        tournament = manager.create_tournament(
            name="Test",
            participants=["alice", "bob"],
        )
        matches = manager.get_matches(tournament_id=tournament.tournament_id)
        match = matches[0]

        manager.record_match_result(
            match_id=match.match_id,
            winner="alice",
            score1=1.0,
            score2=0.5,
        )

        updated = manager.get_matches(tournament_id=tournament.tournament_id)[0]
        assert updated.winner == "alice"
        assert updated.score1 == 1.0
        assert updated.score2 == 0.5
        assert updated.completed_at is not None

    def test_record_match_with_debate_id(self, manager):
        """Test recording match result with debate ID."""
        tournament = manager.create_tournament(
            name="Test",
            participants=["alice", "bob"],
        )
        matches = manager.get_matches(tournament_id=tournament.tournament_id)
        match = matches[0]

        manager.record_match_result(
            match_id=match.match_id,
            winner="alice",
            debate_id="debate_123",
        )

        updated = manager.get_matches(tournament_id=tournament.tournament_id)[0]
        assert updated.debate_id == "debate_123"

    def test_record_draw(self, manager):
        """Test recording a draw (no winner)."""
        tournament = manager.create_tournament(
            name="Test",
            participants=["alice", "bob"],
        )
        matches = manager.get_matches(tournament_id=tournament.tournament_id)
        match = matches[0]

        manager.record_match_result(
            match_id=match.match_id,
            winner=None,  # Draw
            score1=0.5,
            score2=0.5,
        )

        updated = manager.get_matches(tournament_id=tournament.tournament_id)[0]
        assert updated.winner is None

    def test_record_nonexistent_match(self, manager):
        """Test recording result for nonexistent match."""
        with pytest.raises(MatchNotFoundError):
            manager.record_match_result(
                match_id="nonexistent",
                winner="alice",
            )

    def test_record_invalid_winner(self, manager):
        """Test recording with invalid winner."""
        tournament = manager.create_tournament(
            name="Test",
            participants=["alice", "bob"],
        )
        matches = manager.get_matches(tournament_id=tournament.tournament_id)
        match = matches[0]

        with pytest.raises(ValidationError, match="must be one of"):
            manager.record_match_result(
                match_id=match.match_id,
                winner="charlie",  # Not a participant
            )

    def test_record_negative_score(self, manager):
        """Test recording with negative score."""
        tournament = manager.create_tournament(
            name="Test",
            participants=["alice", "bob"],
        )
        matches = manager.get_matches(tournament_id=tournament.tournament_id)
        match = matches[0]

        with pytest.raises(ValidationError, match="cannot be negative"):
            manager.record_match_result(
                match_id=match.match_id,
                winner="alice",
                score1=-1.0,
            )

    def test_match_logged_in_history(self, manager):
        """Test that match completion is logged."""
        tournament = manager.create_tournament(
            name="Test",
            participants=["alice", "bob"],
        )
        matches = manager.get_matches(tournament_id=tournament.tournament_id)
        match = matches[0]

        manager.record_match_result(match_id=match.match_id, winner="alice")

        history = manager.get_tournament_history(tournament.tournament_id)
        match_events = [h for h in history if h.event_type == TournamentEvent.MATCH_COMPLETED]
        assert len(match_events) >= 1


class TestStandings:
    """Tests for standings calculation."""

    def test_standings_empty_tournament(self, manager):
        """Test standings with no completed matches."""
        tournament = manager.create_tournament(
            name="Test",
            participants=["alice", "bob", "charlie"],
        )

        standings = manager.get_current_standings(tournament.tournament_id)

        assert len(standings) == 3
        for s in standings:
            assert s.points == 0.0
            assert s.wins == 0

    def test_standings_after_matches(self, manager):
        """Test standings after some matches."""
        tournament = manager.create_tournament(
            name="Test",
            participants=["alice", "bob", "charlie"],
        )
        matches = manager.get_matches(tournament_id=tournament.tournament_id)

        # Record all matches with alice winning
        for m in matches:
            if "alice" in (m.agent1, m.agent2):
                winner = "alice"
            else:
                winner = m.agent1  # Arbitrary for bob vs charlie
            manager.record_match_result(match_id=m.match_id, winner=winner, score1=1.0)

        standings = manager.get_current_standings(tournament.tournament_id)

        # Alice should be first (won all her matches)
        assert standings[0].agent == "alice"
        assert standings[0].wins == 2

    def test_standings_sorted_by_points(self, manager):
        """Test standings are sorted by points descending."""
        tournament = manager.create_tournament(
            name="Test",
            participants=["a", "b", "c"],
        )
        matches = manager.get_matches(tournament_id=tournament.tournament_id)

        # a wins 2, b wins 1, c wins 0
        for m in matches:
            if m.agent1 == "a" or m.agent2 == "a":
                winner = "a"
            elif m.agent1 == "b" or m.agent2 == "b":
                winner = "b"
            else:
                winner = m.agent1
            manager.record_match_result(match_id=m.match_id, winner=winner)

        standings = manager.get_current_standings(tournament.tournament_id)

        points = [s.points for s in standings]
        assert points == sorted(points, reverse=True)

    def test_standings_draw_gives_half_point(self, manager):
        """Test that draws give 0.5 points each."""
        tournament = manager.create_tournament(
            name="Test",
            participants=["alice", "bob"],
        )
        matches = manager.get_matches(tournament_id=tournament.tournament_id)

        manager.record_match_result(match_id=matches[0].match_id, winner=None)

        standings = manager.get_current_standings(tournament.tournament_id)

        for s in standings:
            assert s.draws == 1
            assert s.points == pytest.approx(0.5, rel=1e-6)


class TestTournamentRetrieval:
    """Tests for tournament retrieval."""

    def test_get_tournament(self, manager):
        """Test getting tournament by ID."""
        created = manager.create_tournament(
            name="Test",
            participants=["alice", "bob"],
        )

        retrieved = manager.get_tournament(created.tournament_id)

        assert retrieved is not None
        assert retrieved.tournament_id == created.tournament_id
        assert retrieved.name == "Test"

    def test_get_nonexistent_tournament(self, manager):
        """Test getting nonexistent tournament returns None."""
        result = manager.get_tournament("nonexistent")
        assert result is None

    def test_list_tournaments(self, manager):
        """Test listing all tournaments."""
        manager.create_tournament(name="T1", participants=["a", "b"])
        manager.create_tournament(name="T2", participants=["c", "d"])

        tournaments = manager.list_tournaments()

        assert len(tournaments) >= 2
        names = [t.name for t in tournaments]
        assert "T1" in names
        assert "T2" in names

    def test_list_tournaments_by_status(self, manager):
        """Test listing tournaments filtered by status."""
        t1 = manager.create_tournament(name="Pending", participants=["a", "b"])
        t2 = manager.create_tournament(name="Completed", participants=["c", "d"])

        # Complete one tournament
        matches = manager.get_matches(tournament_id=t2.tournament_id)
        for m in matches:
            manager.record_match_result(match_id=m.match_id, winner=m.agent1)
        manager.advance_round(t2.tournament_id)

        pending = manager.list_tournaments(status=TournamentStatus.PENDING.value)
        assert any(t.tournament_id == t1.tournament_id for t in pending)


class TestAdvanceRound:
    """Tests for round advancement."""

    def test_advance_single_elimination(self, manager):
        """Test advancing rounds in single elimination."""
        tournament = manager.create_tournament(
            name="Test",
            participants=["a", "b", "c", "d"],
            bracket_type="single_elimination",
        )

        # Complete round 1
        round1_matches = manager.get_matches(tournament_id=tournament.tournament_id, round_num=1)
        for m in round1_matches:
            manager.record_match_result(match_id=m.match_id, winner=m.agent1)

        # Advance to round 2
        result = manager.advance_round(tournament.tournament_id)
        assert result is True

        # Should have round 2 matches now
        round2_matches = manager.get_matches(tournament_id=tournament.tournament_id, round_num=2)
        assert len(round2_matches) == 1

    def test_advance_incomplete_round_fails(self, manager):
        """Test that advancing incomplete round raises error."""
        tournament = manager.create_tournament(
            name="Test",
            participants=["a", "b", "c", "d"],
            bracket_type="single_elimination",
        )

        # Only complete one match
        matches = manager.get_matches(tournament_id=tournament.tournament_id, round_num=1)
        manager.record_match_result(match_id=matches[0].match_id, winner=matches[0].agent1)

        with pytest.raises(InvalidStateError, match="not complete"):
            manager.advance_round(tournament.tournament_id)

    def test_advance_nonexistent_tournament(self, manager):
        """Test advancing nonexistent tournament."""
        with pytest.raises(TournamentNotFoundError):
            manager.advance_round("nonexistent")

    def test_tournament_completes_after_final(self, manager):
        """Test tournament completes after final round."""
        tournament = manager.create_tournament(
            name="Test",
            participants=["a", "b"],  # Only 2 = 1 round
            bracket_type="single_elimination",
        )

        # Complete the only match
        matches = manager.get_matches(tournament_id=tournament.tournament_id)
        manager.record_match_result(match_id=matches[0].match_id, winner="a")

        # Try to advance - should complete instead
        result = manager.advance_round(tournament.tournament_id)
        assert result is False

        # Check tournament is completed
        updated = manager.get_tournament(tournament.tournament_id)
        assert updated.status == TournamentStatus.COMPLETED.value

    def test_round_robin_completion(self, manager):
        """Test round-robin tournament completion."""
        tournament = manager.create_tournament(
            name="Test",
            participants=["a", "b", "c"],
            bracket_type="round_robin",
        )

        # Complete all matches
        matches = manager.get_matches(tournament_id=tournament.tournament_id)
        for m in matches:
            manager.record_match_result(match_id=m.match_id, winner=m.agent1)

        # Advance should mark as completed
        manager.advance_round(tournament.tournament_id)

        updated = manager.get_tournament(tournament.tournament_id)
        assert updated.status == TournamentStatus.COMPLETED.value


class TestCancelTournament:
    """Tests for tournament cancellation."""

    def test_cancel_tournament(self, manager):
        """Test cancelling a tournament."""
        tournament = manager.create_tournament(
            name="Test",
            participants=["alice", "bob"],
        )

        result = manager.cancel_tournament(tournament.tournament_id)
        assert result is True

        updated = manager.get_tournament(tournament.tournament_id)
        assert updated.status == TournamentStatus.CANCELLED.value

    def test_cancel_completed_tournament(self, manager):
        """Test cancelling already completed tournament returns False."""
        tournament = manager.create_tournament(
            name="Test",
            participants=["a", "b"],
            bracket_type="single_elimination",
        )

        # Complete tournament
        matches = manager.get_matches(tournament_id=tournament.tournament_id)
        manager.record_match_result(match_id=matches[0].match_id, winner="a")
        manager.advance_round(tournament.tournament_id)

        result = manager.cancel_tournament(tournament.tournament_id)
        assert result is False

    def test_cancel_nonexistent_tournament(self, manager):
        """Test cancelling nonexistent tournament."""
        with pytest.raises(TournamentNotFoundError):
            manager.cancel_tournament("nonexistent")

    def test_cancel_logged_in_history(self, manager):
        """Test that cancellation is logged."""
        tournament = manager.create_tournament(
            name="Test",
            participants=["alice", "bob"],
        )

        manager.cancel_tournament(tournament.tournament_id)

        history = manager.get_tournament_history(tournament.tournament_id)
        cancel_events = [h for h in history if h.event_type == TournamentEvent.CANCELLED]
        assert len(cancel_events) >= 1


class TestTournamentHistory:
    """Tests for tournament history tracking."""

    def test_get_history_order(self, manager):
        """Test history contains expected events."""
        tournament = manager.create_tournament(
            name="Test",
            participants=["alice", "bob"],
        )
        matches = manager.get_matches(tournament_id=tournament.tournament_id)
        manager.record_match_result(match_id=matches[0].match_id, winner="alice")

        history = manager.get_tournament_history(tournament.tournament_id)

        # History should contain both CREATED and MATCH_COMPLETED events
        event_types = [h.event_type for h in history]
        assert TournamentEvent.CREATED in event_types
        assert TournamentEvent.MATCH_COMPLETED in event_types
        assert len(history) >= 2

    def test_history_limit(self, manager):
        """Test history respects limit parameter."""
        tournament = manager.create_tournament(
            name="Test",
            participants=["alice", "bob"],
        )

        history = manager.get_tournament_history(tournament.tournament_id, limit=1)
        assert len(history) == 1


class TestMatchFiltering:
    """Tests for match filtering."""

    def test_filter_completed_only(self, manager):
        """Test filtering completed matches only."""
        tournament = manager.create_tournament(
            name="Test",
            participants=["a", "b", "c"],
        )
        matches = manager.get_matches(tournament_id=tournament.tournament_id)

        # Complete only one match
        manager.record_match_result(match_id=matches[0].match_id, winner=matches[0].agent1)

        completed = manager.get_matches(tournament_id=tournament.tournament_id, completed_only=True)
        assert len(completed) == 1

    def test_filter_by_round(self, manager):
        """Test filtering by round number."""
        tournament = manager.create_tournament(
            name="Test",
            participants=["a", "b", "c", "d"],
            bracket_type="single_elimination",
        )

        # Complete round 1
        round1 = manager.get_matches(tournament_id=tournament.tournament_id, round_num=1)
        for m in round1:
            manager.record_match_result(match_id=m.match_id, winner=m.agent1)

        manager.advance_round(tournament.tournament_id)

        round2 = manager.get_matches(tournament_id=tournament.tournament_id, round_num=2)
        assert len(round2) == 1
        assert round2[0].round_num == 2
