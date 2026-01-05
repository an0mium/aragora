"""
Tests for ELO Ranking System.

Tests the ELO-based agent ranking including:
- Rating calculations and updates
- Match recording and history
- Domain-specific ratings
- Calibration scoring
- Relationship tracking and metrics
- Leaderboard functionality
"""

import os
import tempfile
import pytest

from aragora.ranking.elo import EloSystem


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    os.unlink(path)


@pytest.fixture
def elo(temp_db):
    """Create an EloSystem instance with temp database."""
    return EloSystem(db_path=temp_db)


class TestEloBasics:
    """Test basic ELO operations."""

    def test_initial_rating(self, elo):
        """Test that new agents start with default rating."""
        rating = elo.get_rating("new_agent")
        assert rating.elo == 1500  # Default ELO
        assert rating.games_played == 0

    def test_record_match_winner(self, elo):
        """Test recording a match updates winner rating."""
        elo.record_match(
            debate_id="match_1",
            participants=["winner", "loser"],
            scores={"winner": 1.0, "loser": 0.0}
        )

        winner_rating = elo.get_rating("winner")
        loser_rating = elo.get_rating("loser")

        assert winner_rating.elo > 1500
        assert loser_rating.elo < 1500
        assert winner_rating.games_played == 1
        assert loser_rating.games_played == 1

    def test_record_match_draw(self, elo):
        """Test recording a draw."""
        elo.record_match(
            debate_id="match_draw",
            participants=["agent_a", "agent_b"],
            scores={"agent_a": 0.5, "agent_b": 0.5}
        )

        rating_a = elo.get_rating("agent_a")
        rating_b = elo.get_rating("agent_b")

        # Both should stay at 1500 for first draw
        assert rating_a.games_played == 1
        assert rating_b.games_played == 1

    def test_rating_changes_based_on_expected(self, elo):
        """Test that upset wins result in bigger rating changes."""
        # Create a strong agent
        for i in range(5):
            elo.record_match(
                debate_id=f"match_{i}",
                participants=["strong", "weak"],
                scores={"strong": 1.0, "weak": 0.0}
            )

        strong_rating = elo.get_rating("strong").elo
        weak_rating = elo.get_rating("weak").elo

        # Now let weak beat strong - should be big upset
        elo.record_match(
            debate_id="upset_match",
            participants=["strong", "weak"],
            scores={"weak": 1.0, "strong": 0.0}
        )

        new_weak_rating = elo.get_rating("weak").elo
        rating_gain = new_weak_rating - weak_rating

        # Upset should result in significant rating gain
        assert rating_gain > 20


class TestDomainRatings:
    """Test domain-specific ratings."""

    def test_domain_rating_tracking(self, elo):
        """Test that domain ratings are tracked separately."""
        elo.record_match(
            debate_id="sec_match",
            participants=["agent_a", "agent_b"],
            scores={"agent_a": 1.0, "agent_b": 0.0},
            domain="security"
        )
        elo.record_match(
            debate_id="perf_match",
            participants=["agent_a", "agent_b"],
            scores={"agent_b": 1.0, "agent_a": 0.0},
            domain="performance"
        )

        # Agent A should be higher in security, lower in performance
        rating_a = elo.get_rating("agent_a")
        a_security = rating_a.domain_elos.get("security", 1500)
        a_performance = rating_a.domain_elos.get("performance", 1500)

        assert a_security > 1500
        assert a_performance < 1500

    def test_overall_rating_uses_all_games(self, elo):
        """Test that overall rating considers all domain games."""
        elo.record_match(
            debate_id="sec_match",
            participants=["agent_a", "agent_b"],
            scores={"agent_a": 1.0, "agent_b": 0.0},
            domain="security"
        )
        elo.record_match(
            debate_id="perf_match",
            participants=["agent_a", "agent_b"],
            scores={"agent_a": 1.0, "agent_b": 0.0},
            domain="performance"
        )

        rating = elo.get_rating("agent_a")
        assert rating.games_played == 2


class TestMatchHistory:
    """Test match history functionality."""

    def test_get_match_history(self, elo):
        """Test retrieving match history."""
        elo.record_match(
            debate_id="match_1",
            participants=["agent_a", "agent_b"],
            scores={"agent_a": 1.0, "agent_b": 0.0}
        )
        elo.record_match(
            debate_id="match_2",
            participants=["agent_a", "agent_c"],
            scores={"agent_c": 1.0, "agent_a": 0.0}
        )

        history = elo.get_recent_matches(limit=10)
        assert len(history) == 2

    def test_head_to_head(self, elo):
        """Test head-to-head statistics."""
        elo.record_match(
            debate_id="match_1",
            participants=["agent_a", "agent_b"],
            scores={"agent_a": 1.0, "agent_b": 0.0}
        )
        elo.record_match(
            debate_id="match_2",
            participants=["agent_a", "agent_b"],
            scores={"agent_a": 1.0, "agent_b": 0.0}
        )
        elo.record_match(
            debate_id="match_3",
            participants=["agent_a", "agent_b"],
            scores={"agent_b": 1.0, "agent_a": 0.0}
        )

        h2h = elo.get_head_to_head("agent_a", "agent_b")
        assert h2h["matches"] == 3
        assert h2h["agent_a_wins"] == 2
        assert h2h["agent_b_wins"] == 1


class TestCalibration:
    """Test calibration scoring."""

    def test_record_prediction(self, elo):
        """Test recording calibration predictions."""
        elo.record_domain_prediction("agent_a", "security", 0.8, True)
        elo.record_domain_prediction("agent_a", "security", 0.7, False)

        cal = elo.get_domain_calibration("agent_a", "security")
        assert cal is not None
        assert "total" in cal  # Top level has 'total', not 'predictions'

    def test_calibration_score(self, elo):
        """Test calibration score calculation."""
        # Perfect calibration: 80% confidence, 80% correct
        for _ in range(8):
            elo.record_domain_prediction("calibrated", "general", 0.8, True)
        for _ in range(2):
            elo.record_domain_prediction("calibrated", "general", 0.8, False)

        # This agent should have good calibration
        # (actual accuracy matches stated confidence)

    def test_overconfident_detection(self, elo):
        """Test detecting overconfident predictions."""
        # High confidence, low accuracy
        for _ in range(10):
            elo.record_domain_prediction("overconfident", "general", 0.9, False)

        # Should have poor calibration


class TestRelationships:
    """Test relationship tracking and metrics."""

    def test_relationship_recorded(self, elo):
        """Test that relationships are recorded via update_relationship."""
        # record_match updates ELO, but update_relationship tracks agent pairs
        elo.update_relationship(
            agent_a="agent_a",
            agent_b="agent_b",
            debate_increment=1,
            agreement_increment=0,
            critique_a_to_b=0,
            critique_b_to_a=0,
            a_win=True,
            b_win=False,
        )

        rel = elo.get_relationship_raw("agent_a", "agent_b")
        assert rel is not None
        assert rel["debate_count"] >= 1

    def test_rivalry_score(self, elo):
        """Test rivalry score computation."""
        # Create competitive relationship
        for i in range(5):
            winner = "agent_a" if i % 2 == 0 else "agent_b"
            loser = "agent_b" if i % 2 == 0 else "agent_a"
            elo.record_match(
                debate_id=f"match_{i}",
                participants=["agent_a", "agent_b"],
                scores={winner: 1.0, loser: 0.0}
            )

        metrics = elo.compute_relationship_metrics("agent_a", "agent_b")
        assert "rivalry_score" in metrics
        assert "alliance_score" in metrics
        assert "relationship" in metrics

    def test_get_rivals(self, elo):
        """Test getting agent's rivals."""
        # Create some matches
        for i in range(5):
            winner = "agent_a" if i % 2 == 0 else "rival_1"
            loser = "rival_1" if i % 2 == 0 else "agent_a"
            elo.record_match(
                debate_id=f"rival_match_{i}",
                participants=["agent_a", "rival_1"],
                scores={winner: 1.0, loser: 0.0}
            )
            elo.record_match(
                debate_id=f"easy_match_{i}",
                participants=["agent_a", "rival_2"],
                scores={"agent_a": 1.0, "rival_2": 0.0}  # Always wins
            )

        rivals = elo.get_rivals("agent_a", limit=5)
        assert isinstance(rivals, list)

    def test_get_allies(self, elo):
        """Test getting agent's allies."""
        allies = elo.get_allies("agent_a", limit=5)
        assert isinstance(allies, list)

    def test_no_relationship_returns_empty(self, elo):
        """Test that unknown relationship returns appropriate defaults."""
        metrics = elo.compute_relationship_metrics("unknown_a", "unknown_b")
        assert metrics["rivalry_score"] == 0.0
        assert metrics["alliance_score"] == 0.0
        assert metrics["relationship"] == "unknown"


class TestLeaderboard:
    """Test leaderboard functionality."""

    def test_leaderboard_ranking(self, elo):
        """Test that leaderboard ranks by rating."""
        elo.record_match(
            debate_id="match_1",
            participants=["best", "worst"],
            scores={"best": 1.0, "worst": 0.0}
        )
        elo.record_match(
            debate_id="match_2",
            participants=["best", "middle"],
            scores={"best": 1.0, "middle": 0.0}
        )
        elo.record_match(
            debate_id="match_3",
            participants=["middle", "worst"],
            scores={"middle": 1.0, "worst": 0.0}
        )

        leaderboard = elo.get_leaderboard(limit=10)

        # Best should be first, worst should be last
        agents = [entry.agent_name for entry in leaderboard]
        assert "best" in agents
        assert "worst" in agents
        assert agents.index("best") < agents.index("worst")

    def test_leaderboard_by_domain(self, elo):
        """Test domain-specific leaderboard."""
        elo.record_match(
            debate_id="sec_match",
            participants=["security_pro", "general"],
            scores={"security_pro": 1.0, "general": 0.0},
            domain="security"
        )
        elo.record_match(
            debate_id="perf_match",
            participants=["perf_pro", "general"],
            scores={"perf_pro": 1.0, "general": 0.0},
            domain="performance"
        )

        security_board = elo.get_leaderboard(domain="security", limit=10)
        # security_pro should rank high in security domain
        agents = [entry.agent_name for entry in security_board]
        assert "security_pro" in agents


class TestWinRate:
    """Test win rate calculations."""

    def test_win_rate_calculation(self, elo):
        """Test win rate is calculated correctly."""
        # 3 wins, 2 losses = 60% win rate
        elo.record_match("m1", ["agent", "opp1"], {"agent": 1.0, "opp1": 0.0})
        elo.record_match("m2", ["agent", "opp2"], {"agent": 1.0, "opp2": 0.0})
        elo.record_match("m3", ["agent", "opp3"], {"agent": 1.0, "opp3": 0.0})
        elo.record_match("m4", ["agent", "opp4"], {"opp4": 1.0, "agent": 0.0})
        elo.record_match("m5", ["agent", "opp5"], {"opp5": 1.0, "agent": 0.0})

        rating = elo.get_rating("agent")
        assert rating.games_played == 5
        assert rating.win_rate == pytest.approx(0.6, rel=0.01)


class TestAtomicWrites:
    """Test database atomicity."""

    def test_concurrent_updates(self, elo):
        """Test that concurrent updates don't corrupt data."""
        # Record many matches
        for i in range(20):
            agent_a = f"agent_{i % 5}"
            agent_b = f"agent_{(i+1) % 5}"
            elo.record_match(
                debate_id=f"match_{i}",
                participants=[agent_a, agent_b],
                scores={agent_a: 1.0, agent_b: 0.0}
            )

        # All agents should have valid ratings
        for i in range(5):
            rating = elo.get_rating(f"agent_{i}")
            assert rating.elo > 0


class TestConcurrentAccess:
    """Test thread-safe concurrent database access."""

    def test_threaded_record_match(self, elo):
        """Test concurrent record_match calls from multiple threads."""
        import threading

        errors = []
        results = []
        lock = threading.Lock()

        def record_matches(thread_id: int):
            try:
                for i in range(10):
                    debate_id = f"thread_{thread_id}_match_{i}"
                    # Use unique opponents per thread to avoid contention
                    elo.record_match(
                        debate_id=debate_id,
                        participants=[f"agent_{thread_id}", f"opponent_{thread_id}"],
                        scores={f"agent_{thread_id}": 1.0, f"opponent_{thread_id}": 0.0}
                    )
                    with lock:
                        results.append(debate_id)
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [threading.Thread(target=record_matches, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors: {errors}"
        assert len(results) == 50  # 5 threads * 10 matches

        # Verify each thread's agent has correct game count
        for i in range(5):
            rating = elo.get_rating(f"agent_{i}")
            assert rating.games_played == 10

    def test_concurrent_read_write(self, elo):
        """Test concurrent reads while writes are happening."""
        import threading
        import time

        errors = []
        read_results = []

        def write_matches():
            try:
                for i in range(20):
                    elo.record_match(
                        debate_id=f"rw_match_{i}",
                        participants=["writer_agent", "opponent"],
                        scores={"writer_agent": 1.0, "opponent": 0.0}
                    )
                    time.sleep(0.01)  # Small delay to interleave
            except Exception as e:
                errors.append(("write", e))

        def read_ratings():
            try:
                for _ in range(30):
                    rating = elo.get_rating("writer_agent")
                    read_results.append(rating.elo)
                    time.sleep(0.005)
            except Exception as e:
                errors.append(("read", e))

        writer = threading.Thread(target=write_matches)
        reader = threading.Thread(target=read_ratings)

        writer.start()
        reader.start()
        writer.join()
        reader.join()

        assert len(errors) == 0, f"Errors: {errors}"
        # ELO should generally increase (we're winning all matches)
        assert read_results[-1] >= read_results[0]


class TestDataConsistency:
    """Test data consistency after various operations."""

    def test_elo_sum_is_conserved(self, elo):
        """Test that total ELO is roughly conserved (zero-sum game)."""
        initial_total = 1500 * 4  # 4 agents, each starts at 1500

        agents = ["a", "b", "c", "d"]
        for i in range(20):
            winner = agents[i % 4]
            loser = agents[(i + 1) % 4]
            elo.record_match(
                debate_id=f"conservation_{i}",
                participants=[winner, loser],
                scores={winner: 1.0, loser: 0.0}
            )

        final_total = sum(elo.get_rating(a).elo for a in agents)

        # ELO is approximately zero-sum (small deviations due to K-factor adjustments)
        assert abs(final_total - initial_total) < 100

    def test_wins_plus_losses_equals_games(self, elo):
        """Test that wins + losses + draws equals games played."""
        for i in range(15):
            if i % 3 == 0:
                scores = {"agent": 1.0, "opp": 0.0}  # Win
            elif i % 3 == 1:
                scores = {"agent": 0.0, "opp": 1.0}  # Loss
            else:
                scores = {"agent": 0.5, "opp": 0.5}  # Draw

            elo.record_match(
                debate_id=f"consistency_{i}",
                participants=["agent", "opp"],
                scores=scores
            )

        rating = elo.get_rating("agent")
        total_games = rating.wins + rating.losses + rating.draws
        assert total_games == rating.games_played
        assert rating.games_played == 15

    def test_match_history_count_matches_games(self, elo):
        """Test that match history count matches total games recorded."""
        for i in range(10):
            elo.record_match(
                debate_id=f"history_{i}",
                participants=["agent_x", "agent_y"],
                scores={"agent_x": 1.0, "agent_y": 0.0}
            )

        history = elo.get_recent_matches(limit=100)
        assert len(history) == 10

        rating_x = elo.get_rating("agent_x")
        rating_y = elo.get_rating("agent_y")
        assert rating_x.games_played == 10
        assert rating_y.games_played == 10


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_participant_ignored(self, elo):
        """Test that matches with single participant are ignored."""
        result = elo.record_match(
            debate_id="single",
            participants=["solo"],
            scores={"solo": 1.0}
        )

        assert result == {}
        rating = elo.get_rating("solo")
        assert rating.games_played == 0

    def test_empty_participants_ignored(self, elo):
        """Test that matches with no participants are ignored."""
        result = elo.record_match(
            debate_id="empty",
            participants=[],
            scores={}
        )

        assert result == {}

    def test_duplicate_debate_id_updates(self, elo):
        """Test that recording same debate_id updates existing record."""
        elo.record_match(
            debate_id="duplicate",
            participants=["a", "b"],
            scores={"a": 1.0, "b": 0.0}
        )
        rating_a_first = elo.get_rating("a").elo

        # Record same debate again (should update/replace)
        elo.record_match(
            debate_id="duplicate",
            participants=["a", "b"],
            scores={"a": 1.0, "b": 0.0}
        )
        rating_a_second = elo.get_rating("a").elo

        # Rating should have changed (match recorded twice)
        assert rating_a_second > rating_a_first

    def test_confidence_weight_clamping(self, elo):
        """Test that confidence_weight is clamped to valid range."""
        # Test with weight below minimum
        elo.record_match(
            debate_id="low_confidence",
            participants=["a", "b"],
            scores={"a": 1.0, "b": 0.0},
            confidence_weight=0.0  # Should be clamped to 0.1
        )

        # Test with weight above maximum
        elo.record_match(
            debate_id="high_confidence",
            participants=["c", "d"],
            scores={"c": 1.0, "d": 0.0},
            confidence_weight=2.0  # Should be clamped to 1.0
        )

        # Both should have recorded without error
        assert elo.get_rating("a").games_played == 1
        assert elo.get_rating("c").games_played == 1

    def test_multiway_match(self, elo):
        """Test that 3+ agent matches are handled."""
        elo.record_match(
            debate_id="multiway",
            participants=["first", "second", "third"],
            scores={"first": 1.0, "second": 0.5, "third": 0.0}
        )

        # All should have played
        assert elo.get_rating("first").games_played == 1
        assert elo.get_rating("second").games_played == 1
        assert elo.get_rating("third").games_played == 1

        # First should have highest rating (won against both)
        first_elo = elo.get_rating("first").elo
        third_elo = elo.get_rating("third").elo
        assert first_elo > third_elo

    def test_zero_scores_handled(self, elo):
        """Test that zero scores for all agents defaults to draw."""
        elo.record_match(
            debate_id="zero_scores",
            participants=["a", "b"],
            scores={"a": 0.0, "b": 0.0}
        )

        # Both should have played
        assert elo.get_rating("a").games_played == 1
        # Should count as draw
        assert elo.get_rating("a").draws == 1

    def test_negative_scores_handled(self, elo):
        """Test that negative scores are handled correctly."""
        elo.record_match(
            debate_id="negative_scores",
            participants=["a", "b"],
            scores={"a": -1.0, "b": -2.0}  # Both negative, a is "less bad"
        )

        # Both should have played
        assert elo.get_rating("a").games_played == 1
        assert elo.get_rating("b").games_played == 1

        # a should have won (higher score)
        assert elo.get_rating("a").wins == 1
        assert elo.get_rating("b").losses == 1


class TestEloHistory:
    """Test ELO history tracking."""

    def test_elo_history_recorded(self, elo):
        """Test that ELO history is recorded after each match."""
        for i in range(5):
            # Use different opponents to avoid ELO interaction effects
            elo.record_match(
                debate_id=f"history_test_{i}",
                participants=["tracked", f"opponent_{i}"],
                scores={"tracked": 1.0, f"opponent_{i}": 0.0}
            )

        history = elo.get_elo_history("tracked", limit=10)
        assert len(history) == 5  # Exactly 5 entries

        # History is list of (created_at, elo) tuples, sorted DESC by created_at
        # Verify structure is correct
        assert all(isinstance(entry, tuple) and len(entry) == 2 for entry in history)

        # ELO should be above starting rating (winning all matches)
        newest_elo = history[0][1]  # Most recent
        assert newest_elo > 1500  # Should have gained ELO from wins


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
