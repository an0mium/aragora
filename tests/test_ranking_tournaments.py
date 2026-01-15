"""
Tests for the Tournament Management System.

Covers:
- Tournament creation with different bracket types
- Match generation (round-robin, single/double elimination)
- Match result recording
- Standings calculation
- Round advancement
- Validation logic
- Error handling
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch

from aragora.ranking.tournaments import (
    TournamentManager,
    Tournament,
    TournamentMatch,
    TournamentStanding,
    TournamentStatus,
    TournamentEvent,
    TournamentError,
    ValidationError,
    TournamentNotFoundError,
    MatchNotFoundError,
    InvalidStateError,
    VALID_BRACKET_TYPES,
    MAX_PARTICIPANTS,
    MIN_PARTICIPANTS,
    MAX_TOURNAMENT_NAME_LENGTH,
)


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_tournaments.db"
        yield str(db_path)


@pytest.fixture
def manager(temp_db):
    """Create a TournamentManager with temp database."""
    return TournamentManager(db_path=temp_db)


@pytest.fixture
def sample_participants():
    """Sample participant list."""
    return ["claude", "gpt4", "gemini", "grok"]


class TestTournamentStanding:
    """Tests for TournamentStanding dataclass."""

    def test_win_rate_with_matches(self):
        """Win rate should be calculated correctly."""
        standing = TournamentStanding(agent="claude", wins=3, losses=1, draws=0)
        assert standing.win_rate == 0.75

    def test_win_rate_with_draws(self):
        """Win rate should account for draws."""
        standing = TournamentStanding(agent="claude", wins=2, losses=1, draws=1)
        assert standing.win_rate == 0.5

    def test_win_rate_no_matches(self):
        """Win rate should be 0 with no matches."""
        standing = TournamentStanding(agent="claude")
        assert standing.win_rate == 0.0

    def test_win_rate_all_losses(self):
        """Win rate should be 0 with all losses."""
        standing = TournamentStanding(agent="claude", wins=0, losses=5)
        assert standing.win_rate == 0.0


class TestTournamentManagerValidation:
    """Tests for TournamentManager validation methods."""

    def test_validate_empty_tournament_name(self, manager):
        """Empty tournament name should raise ValidationError."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            manager._validate_tournament_name("")

    def test_validate_whitespace_tournament_name(self, manager):
        """Whitespace-only tournament name should raise ValidationError."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            manager._validate_tournament_name("   ")

    def test_validate_long_tournament_name(self, manager):
        """Too-long tournament name should raise ValidationError."""
        long_name = "x" * (MAX_TOURNAMENT_NAME_LENGTH + 1)
        with pytest.raises(ValidationError, match="exceeds"):
            manager._validate_tournament_name(long_name)

    def test_validate_valid_tournament_name(self, manager):
        """Valid tournament name should pass."""
        manager._validate_tournament_name("Q1 2024 Championship")  # No exception

    def test_validate_empty_participants(self, manager):
        """Empty participant list should raise ValidationError."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            manager._validate_participants([])

    def test_validate_too_few_participants(self, manager):
        """Less than MIN_PARTICIPANTS should raise ValidationError."""
        with pytest.raises(ValidationError, match=f"at least {MIN_PARTICIPANTS}"):
            manager._validate_participants(["claude"])

    def test_validate_too_many_participants(self, manager):
        """More than MAX_PARTICIPANTS should raise ValidationError."""
        many_participants = [f"agent_{i}" for i in range(MAX_PARTICIPANTS + 1)]
        with pytest.raises(ValidationError, match=f"Cannot exceed {MAX_PARTICIPANTS}"):
            manager._validate_participants(many_participants)

    def test_validate_duplicate_participants(self, manager):
        """Duplicate participants should raise ValidationError."""
        with pytest.raises(ValidationError, match="Duplicate participant"):
            manager._validate_participants(["claude", "gpt4", "claude"])

    def test_validate_invalid_participant_name(self, manager):
        """Invalid participant name should raise ValidationError."""
        with pytest.raises(ValidationError, match="Invalid participant name"):
            manager._validate_participants(["claude", "gpt 4"])  # Space not allowed

    def test_validate_empty_participant_name(self, manager):
        """Empty participant name should raise ValidationError."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            manager._validate_participants(["claude", ""])

    def test_validate_valid_participants(self, manager, sample_participants):
        """Valid participant list should pass."""
        manager._validate_participants(sample_participants)  # No exception

    def test_validate_invalid_bracket_type(self, manager):
        """Invalid bracket type should raise ValidationError."""
        with pytest.raises(ValidationError, match="Invalid bracket type"):
            manager._validate_bracket_type("invalid_bracket")

    def test_validate_valid_bracket_types(self, manager):
        """All valid bracket types should pass."""
        for bracket_type in VALID_BRACKET_TYPES:
            manager._validate_bracket_type(bracket_type)  # No exception

    def test_validate_negative_score(self, manager):
        """Negative score should raise ValidationError."""
        with pytest.raises(ValidationError, match="cannot be negative"):
            manager._validate_score(-1.0)

    def test_validate_non_numeric_score(self, manager):
        """Non-numeric score should raise ValidationError."""
        with pytest.raises(ValidationError, match="must be a number"):
            manager._validate_score("high")  # type: ignore

    def test_validate_valid_score(self, manager):
        """Valid scores should be normalized to float."""
        assert manager._validate_score(5) == 5.0
        assert manager._validate_score(3.5) == 3.5
        assert manager._validate_score(0) == 0.0


class TestTournamentCreation:
    """Tests for tournament creation."""

    def test_create_round_robin_tournament(self, manager, sample_participants):
        """Should create a round-robin tournament."""
        tournament = manager.create_tournament(
            name="Test Tournament",
            participants=sample_participants,
            bracket_type="round_robin",
        )

        assert tournament.tournament_id.startswith("tournament_")
        assert tournament.name == "Test Tournament"
        assert tournament.participants == sample_participants
        assert tournament.bracket_type == "round_robin"
        assert tournament.status == TournamentStatus.PENDING.value
        # Round-robin: n-1 rounds
        assert tournament.total_rounds == len(sample_participants) - 1

    def test_create_single_elimination_tournament(self, manager, sample_participants):
        """Should create a single elimination tournament."""
        tournament = manager.create_tournament(
            name="Elimination Cup",
            participants=sample_participants,
            bracket_type="single_elimination",
        )

        assert tournament.bracket_type == "single_elimination"
        # Single elimination: ceil(log2(n)) rounds
        assert tournament.total_rounds == 2  # ceil(log2(4)) = 2

    def test_create_double_elimination_tournament(self, manager, sample_participants):
        """Should create a double elimination tournament."""
        tournament = manager.create_tournament(
            name="Double Elim",
            participants=sample_participants,
            bracket_type="double_elimination",
        )

        assert tournament.bracket_type == "double_elimination"
        # Double elimination: ceil(log2(n)) * 2 rounds
        assert tournament.total_rounds == 4  # ceil(log2(4)) * 2 = 4

    def test_create_tournament_generates_matches(self, manager, sample_participants):
        """Creating a tournament should generate initial matches."""
        tournament = manager.create_tournament(
            name="Test",
            participants=sample_participants,
            bracket_type="round_robin",
        )

        matches = manager.get_matches(tournament_id=tournament.tournament_id)
        # Round-robin: n(n-1)/2 matches for n participants
        expected_matches = len(sample_participants) * (len(sample_participants) - 1) // 2
        assert len(matches) == expected_matches

    def test_create_tournament_logs_event(self, manager, sample_participants):
        """Creating a tournament should log a CREATED event."""
        tournament = manager.create_tournament(
            name="Test",
            participants=sample_participants,
            bracket_type="round_robin",
        )

        history = manager.get_tournament_history(tournament.tournament_id)
        assert len(history) == 1
        assert history[0].event_type == TournamentEvent.CREATED


class TestMatchGeneration:
    """Tests for match generation."""

    def test_round_robin_matches_cover_all_pairs(self, manager):
        """Round-robin should generate matches for all participant pairs."""
        participants = ["a", "b", "c"]
        tournament = manager.create_tournament(
            name="Test", participants=participants, bracket_type="round_robin"
        )

        matches = manager.get_matches(tournament_id=tournament.tournament_id)
        pairs = {(m.agent1, m.agent2) for m in matches}

        # Should have a-b, a-c, b-c
        expected_pairs = {("a", "b"), ("a", "c"), ("b", "c")}
        assert pairs == expected_pairs

    def test_single_elimination_pads_to_power_of_two(self, manager):
        """Single elimination should pad bracket to power of 2."""
        # 3 participants should get padded to 4 (with one BYE)
        participants = ["a", "b", "c"]
        tournament = manager.create_tournament(
            name="Test", participants=participants, bracket_type="single_elimination"
        )

        matches = manager.get_matches(tournament_id=tournament.tournament_id)
        # First round: 2 matches (a vs b, c vs BYE)
        assert len(matches) == 2
        agents_in_matches = {m.agent1 for m in matches} | {m.agent2 for m in matches}
        assert "BYE" in agents_in_matches

    def test_double_elimination_generates_winners_bracket(self, manager, sample_participants):
        """Double elimination should generate winners bracket first round."""
        tournament = manager.create_tournament(
            name="Test", participants=sample_participants, bracket_type="double_elimination"
        )

        matches = manager.get_matches(tournament_id=tournament.tournament_id, losers_bracket=False)
        # First round of winners bracket
        assert all(m.round_num == 1 for m in matches)
        assert all(not m.is_losers_bracket for m in matches)


class TestMatchResults:
    """Tests for recording match results."""

    def test_record_match_result(self, manager, sample_participants):
        """Should record match result correctly."""
        tournament = manager.create_tournament(
            name="Test", participants=sample_participants, bracket_type="round_robin"
        )

        matches = manager.get_matches(tournament_id=tournament.tournament_id)
        match = matches[0]

        manager.record_match_result(
            match_id=match.match_id,
            winner=match.agent1,
            score1=10.0,
            score2=5.0,
            debate_id="debate_123",
        )

        updated_matches = manager.get_matches(tournament_id=tournament.tournament_id)
        updated_match = next(m for m in updated_matches if m.match_id == match.match_id)

        assert updated_match.winner == match.agent1
        assert updated_match.score1 == 10.0
        assert updated_match.score2 == 5.0
        assert updated_match.debate_id == "debate_123"
        assert updated_match.completed_at is not None

    def test_record_match_result_draw(self, manager, sample_participants):
        """Should record draw (no winner)."""
        tournament = manager.create_tournament(
            name="Test", participants=sample_participants, bracket_type="round_robin"
        )

        matches = manager.get_matches(tournament_id=tournament.tournament_id)
        match = matches[0]

        manager.record_match_result(
            match_id=match.match_id,
            winner=None,  # Draw
            score1=5.0,
            score2=5.0,
        )

        updated_matches = manager.get_matches(tournament_id=tournament.tournament_id)
        updated_match = next(m for m in updated_matches if m.match_id == match.match_id)
        assert updated_match.winner is None

    def test_record_invalid_winner(self, manager, sample_participants):
        """Should reject winner not in match."""
        tournament = manager.create_tournament(
            name="Test", participants=sample_participants, bracket_type="round_robin"
        )

        matches = manager.get_matches(tournament_id=tournament.tournament_id)
        match = matches[0]

        with pytest.raises(ValidationError, match="must be one of"):
            manager.record_match_result(
                match_id=match.match_id,
                winner="nonexistent_agent",
                score1=10.0,
                score2=5.0,
            )

    def test_record_nonexistent_match(self, manager):
        """Should raise MatchNotFoundError for nonexistent match."""
        with pytest.raises(MatchNotFoundError):
            manager.record_match_result(
                match_id="nonexistent_match",
                winner="claude",
                score1=10.0,
                score2=5.0,
            )

    def test_record_match_logs_event(self, manager, sample_participants):
        """Recording a match should log MATCH_COMPLETED event."""
        tournament = manager.create_tournament(
            name="Test", participants=sample_participants, bracket_type="round_robin"
        )

        matches = manager.get_matches(tournament_id=tournament.tournament_id)
        match = matches[0]

        manager.record_match_result(
            match_id=match.match_id,
            winner=match.agent1,
            score1=10.0,
            score2=5.0,
        )

        history = manager.get_tournament_history(tournament.tournament_id)
        match_events = [e for e in history if e.event_type == TournamentEvent.MATCH_COMPLETED]
        assert len(match_events) == 1


class TestStandings:
    """Tests for standings calculation."""

    def test_initial_standings(self, manager, sample_participants):
        """Initial standings should have all participants with 0 points."""
        tournament = manager.create_tournament(
            name="Test", participants=sample_participants, bracket_type="round_robin"
        )

        standings = manager.get_current_standings(tournament.tournament_id)
        assert len(standings) == len(sample_participants)
        for s in standings:
            assert s.wins == 0
            assert s.losses == 0
            assert s.draws == 0
            assert s.points == 0.0

    def test_standings_after_wins(self, manager, sample_participants):
        """Standings should reflect wins correctly."""
        tournament = manager.create_tournament(
            name="Test", participants=sample_participants, bracket_type="round_robin"
        )

        matches = manager.get_matches(tournament_id=tournament.tournament_id)

        # Agent claude wins first match
        first_match = matches[0]
        manager.record_match_result(
            match_id=first_match.match_id,
            winner=first_match.agent1,
            score1=10.0,
            score2=5.0,
        )

        standings = manager.get_current_standings(tournament.tournament_id)
        winner_standing = next(s for s in standings if s.agent == first_match.agent1)
        loser_standing = next(s for s in standings if s.agent == first_match.agent2)

        assert winner_standing.wins == 1
        assert winner_standing.points == 1.0
        assert loser_standing.losses == 1
        assert loser_standing.points == 0.0

    def test_standings_after_draw(self, manager, sample_participants):
        """Standings should reflect draws correctly."""
        tournament = manager.create_tournament(
            name="Test", participants=sample_participants, bracket_type="round_robin"
        )

        matches = manager.get_matches(tournament_id=tournament.tournament_id)
        first_match = matches[0]

        manager.record_match_result(
            match_id=first_match.match_id,
            winner=None,  # Draw
            score1=5.0,
            score2=5.0,
        )

        standings = manager.get_current_standings(tournament.tournament_id)
        agent1_standing = next(s for s in standings if s.agent == first_match.agent1)
        agent2_standing = next(s for s in standings if s.agent == first_match.agent2)

        assert agent1_standing.draws == 1
        assert agent1_standing.points == 0.5
        assert agent2_standing.draws == 1
        assert agent2_standing.points == 0.5

    def test_standings_sorted_by_points(self, manager):
        """Standings should be sorted by points descending."""
        participants = ["a", "b", "c"]
        tournament = manager.create_tournament(
            name="Test", participants=participants, bracket_type="round_robin"
        )

        matches = manager.get_matches(tournament_id=tournament.tournament_id)

        # Make a win all matches
        for match in matches:
            winner = "a" if "a" in (match.agent1, match.agent2) else match.agent1
            manager.record_match_result(
                match_id=match.match_id, winner=winner, score1=10.0, score2=5.0
            )

        standings = manager.get_current_standings(tournament.tournament_id)
        # a should be first with 2 wins
        assert standings[0].agent == "a"
        assert standings[0].wins == 2


class TestRoundAdvancement:
    """Tests for advancing tournament rounds."""

    def test_advance_single_elimination(self, manager):
        """Should advance single elimination tournament."""
        participants = ["a", "b", "c", "d"]
        tournament = manager.create_tournament(
            name="Test", participants=participants, bracket_type="single_elimination"
        )

        # Complete round 1
        matches = manager.get_matches(tournament_id=tournament.tournament_id, round_num=1)
        for match in matches:
            manager.record_match_result(
                match_id=match.match_id, winner=match.agent1, score1=10.0, score2=5.0
            )

        # Advance to round 2
        advanced = manager.advance_round(tournament.tournament_id)
        assert advanced is True

        # Should have round 2 matches
        round2_matches = manager.get_matches(tournament_id=tournament.tournament_id, round_num=2)
        assert len(round2_matches) == 1  # Finals

    def test_advance_completes_tournament(self, manager):
        """Advancing final round should complete tournament."""
        participants = ["a", "b"]
        tournament = manager.create_tournament(
            name="Test", participants=participants, bracket_type="single_elimination"
        )

        # Complete the only match
        matches = manager.get_matches(tournament_id=tournament.tournament_id)
        manager.record_match_result(
            match_id=matches[0].match_id, winner="a", score1=10.0, score2=5.0
        )

        # Try to advance - should complete tournament
        advanced = manager.advance_round(tournament.tournament_id)
        assert advanced is False  # No more rounds

        updated = manager.get_tournament(tournament.tournament_id)
        assert updated.status == TournamentStatus.COMPLETED.value

    def test_advance_incomplete_round_raises(self, manager):
        """Should raise InvalidStateError if round not complete."""
        participants = ["a", "b", "c", "d"]
        tournament = manager.create_tournament(
            name="Test", participants=participants, bracket_type="single_elimination"
        )

        # Don't complete any matches
        with pytest.raises(InvalidStateError, match="not complete"):
            manager.advance_round(tournament.tournament_id)

    def test_advance_nonexistent_tournament(self, manager):
        """Should raise TournamentNotFoundError."""
        with pytest.raises(TournamentNotFoundError):
            manager.advance_round("nonexistent")


class TestTournamentManagement:
    """Tests for tournament management operations."""

    def test_get_tournament(self, manager, sample_participants):
        """Should retrieve tournament by ID."""
        created = manager.create_tournament(
            name="Test", participants=sample_participants, bracket_type="round_robin"
        )

        retrieved = manager.get_tournament(created.tournament_id)
        assert retrieved is not None
        assert retrieved.tournament_id == created.tournament_id
        assert retrieved.name == "Test"

    def test_get_nonexistent_tournament(self, manager):
        """Should return None for nonexistent tournament."""
        result = manager.get_tournament("nonexistent")
        assert result is None

    def test_list_tournaments(self, manager, sample_participants):
        """Should list all tournaments."""
        manager.create_tournament(
            name="Tournament 1", participants=sample_participants, bracket_type="round_robin"
        )
        manager.create_tournament(
            name="Tournament 2", participants=sample_participants, bracket_type="round_robin"
        )

        tournaments = manager.list_tournaments()
        assert len(tournaments) == 2

    def test_list_tournaments_by_status(self, manager, sample_participants):
        """Should filter tournaments by status."""
        manager.create_tournament(
            name="Test", participants=sample_participants, bracket_type="round_robin"
        )

        pending = manager.list_tournaments(status=TournamentStatus.PENDING.value)
        assert len(pending) == 1

        completed = manager.list_tournaments(status=TournamentStatus.COMPLETED.value)
        assert len(completed) == 0

    def test_cancel_tournament(self, manager, sample_participants):
        """Should cancel tournament."""
        tournament = manager.create_tournament(
            name="Test", participants=sample_participants, bracket_type="round_robin"
        )

        result = manager.cancel_tournament(tournament.tournament_id)
        assert result is True

        updated = manager.get_tournament(tournament.tournament_id)
        assert updated.status == TournamentStatus.CANCELLED.value

    def test_cancel_nonexistent_tournament(self, manager):
        """Should raise TournamentNotFoundError."""
        with pytest.raises(TournamentNotFoundError):
            manager.cancel_tournament("nonexistent")

    def test_cancel_already_completed(self, manager):
        """Should return False for already completed tournament."""
        participants = ["a", "b"]
        tournament = manager.create_tournament(
            name="Test", participants=participants, bracket_type="single_elimination"
        )

        # Complete the tournament
        matches = manager.get_matches(tournament_id=tournament.tournament_id)
        manager.record_match_result(
            match_id=matches[0].match_id, winner="a", score1=10.0, score2=5.0
        )
        manager.advance_round(tournament.tournament_id)

        # Try to cancel
        result = manager.cancel_tournament(tournament.tournament_id)
        assert result is False


class TestMatchFiltering:
    """Tests for match filtering."""

    def test_filter_by_round(self, manager):
        """Should filter matches by round number."""
        participants = ["a", "b", "c", "d"]
        tournament = manager.create_tournament(
            name="Test", participants=participants, bracket_type="single_elimination"
        )

        round1_matches = manager.get_matches(tournament_id=tournament.tournament_id, round_num=1)
        assert all(m.round_num == 1 for m in round1_matches)

    def test_filter_completed_only(self, manager, sample_participants):
        """Should filter to completed matches only."""
        tournament = manager.create_tournament(
            name="Test", participants=sample_participants, bracket_type="round_robin"
        )

        # No completed matches yet
        completed = manager.get_matches(tournament_id=tournament.tournament_id, completed_only=True)
        assert len(completed) == 0

        # Complete one match
        matches = manager.get_matches(tournament_id=tournament.tournament_id)
        manager.record_match_result(
            match_id=matches[0].match_id, winner=matches[0].agent1, score1=10.0, score2=5.0
        )

        # Now should have one completed
        completed = manager.get_matches(tournament_id=tournament.tournament_id, completed_only=True)
        assert len(completed) == 1


class TestTournamentHistory:
    """Tests for tournament history/event logging."""

    def test_history_tracks_all_events(self, manager, sample_participants):
        """History should track tournament lifecycle events."""
        tournament = manager.create_tournament(
            name="Test", participants=sample_participants, bracket_type="round_robin"
        )

        # Record a match
        matches = manager.get_matches(tournament_id=tournament.tournament_id)
        manager.record_match_result(
            match_id=matches[0].match_id, winner=matches[0].agent1, score1=10.0, score2=5.0
        )

        history = manager.get_tournament_history(tournament.tournament_id)

        event_types = {e.event_type for e in history}
        assert TournamentEvent.CREATED in event_types
        assert TournamentEvent.MATCH_COMPLETED in event_types

    def test_history_contains_events_in_order(self, manager, sample_participants):
        """History should contain all events (ordering by timestamp may vary at same-second granularity)."""
        tournament = manager.create_tournament(
            name="Test", participants=sample_participants, bracket_type="round_robin"
        )

        matches = manager.get_matches(tournament_id=tournament.tournament_id)
        manager.record_match_result(
            match_id=matches[0].match_id, winner=matches[0].agent1, score1=10.0, score2=5.0
        )

        history = manager.get_tournament_history(tournament.tournament_id)

        # Should have both events
        assert len(history) == 2
        event_types = {e.event_type for e in history}
        assert TournamentEvent.CREATED in event_types
        assert TournamentEvent.MATCH_COMPLETED in event_types

        # Timestamps should be in descending order (or equal)
        assert history[0].timestamp >= history[1].timestamp

    def test_history_respects_limit(self, manager, sample_participants):
        """History should respect limit parameter."""
        tournament = manager.create_tournament(
            name="Test", participants=sample_participants, bracket_type="round_robin"
        )

        # Create multiple events
        matches = manager.get_matches(tournament_id=tournament.tournament_id)
        for match in matches[:3]:
            manager.record_match_result(
                match_id=match.match_id, winner=match.agent1, score1=10.0, score2=5.0
            )

        history = manager.get_tournament_history(tournament.tournament_id, limit=2)
        assert len(history) == 2
