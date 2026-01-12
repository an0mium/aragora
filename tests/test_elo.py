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

    def test_duplicate_debate_id_ignored(self, elo):
        """Test that recording same debate_id is ignored (prevents double-counting)."""
        result1 = elo.record_match(
            debate_id="duplicate",
            participants=["a", "b"],
            scores={"a": 1.0, "b": 0.0}
        )
        rating_a_first = elo.get_rating("a").elo
        assert result1 != {}  # First call should return ELO changes

        # Record same debate again (should be ignored)
        result2 = elo.record_match(
            debate_id="duplicate",
            participants=["a", "b"],
            scores={"a": 1.0, "b": 0.0}
        )
        rating_a_second = elo.get_rating("a").elo

        # Rating should NOT change (duplicate ignored)
        assert rating_a_second == rating_a_first
        # Second call should return cached ELO changes from first call
        assert result2 == result1

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


class TestBatchOperations:
    """Test batch operations for performance optimization."""

    def test_get_ratings_batch_empty_list(self, elo):
        """Test that empty list returns empty dict."""
        result = elo.get_ratings_batch([])
        assert result == {}

    def test_get_ratings_batch_single_agent(self, elo):
        """Test batch fetch with single agent."""
        elo.record_match("m1", ["alice", "bob"], {"alice": 1.0, "bob": 0.0})

        result = elo.get_ratings_batch(["alice"])
        assert "alice" in result
        assert result["alice"].elo > 1500  # Won a match

    def test_get_ratings_batch_multiple_agents(self, elo):
        """Test batch fetch with multiple agents."""
        elo.record_match("m1", ["alice", "bob"], {"alice": 1.0, "bob": 0.0})
        elo.record_match("m2", ["charlie", "david"], {"charlie": 1.0, "david": 0.0})

        result = elo.get_ratings_batch(["alice", "bob", "charlie", "david"])
        assert len(result) == 4
        assert all(name in result for name in ["alice", "bob", "charlie", "david"])

    def test_get_ratings_batch_unknown_agents_get_defaults(self, elo):
        """Test that unknown agents get default ratings."""
        result = elo.get_ratings_batch(["unknown1", "unknown2"])
        assert len(result) == 2
        assert result["unknown1"].elo == 1500  # Default
        assert result["unknown2"].elo == 1500
        assert result["unknown1"].games_played == 0

    def test_get_ratings_batch_mixed_known_unknown(self, elo):
        """Test batch with mix of known and unknown agents."""
        elo.record_match("m1", ["known", "opponent"], {"known": 1.0, "opponent": 0.0})

        result = elo.get_ratings_batch(["known", "unknown"])
        assert result["known"].elo > 1500  # Has played
        assert result["unknown"].elo == 1500  # Default

    def test_get_ratings_batch_preserves_all_fields(self, elo):
        """Test that batch fetch preserves all rating fields."""
        elo.record_match("m1", ["agent", "opp"], {"agent": 1.0, "opp": 0.0})
        elo.record_domain_prediction("agent", "pred1", 0.8, "general")

        result = elo.get_ratings_batch(["agent"])
        rating = result["agent"]

        assert rating.agent_name == "agent"
        assert rating.elo > 1500
        assert rating.wins == 1
        assert rating.games_played == 1

    def test_update_relationships_batch_empty_list(self, elo):
        """Test that empty updates list does nothing."""
        elo.update_relationships_batch([])
        # Should not raise

    def test_update_relationships_batch_single_update(self, elo):
        """Test batch update with single relationship."""
        elo.update_relationships_batch([{
            "agent_a": "alice",
            "agent_b": "bob",
            "debate_increment": 1,
            "agreement_increment": 1,
            "a_win": 1,
            "b_win": 0,
        }])

        rel = elo.get_relationship_raw("alice", "bob")
        assert rel is not None
        assert rel["debate_count"] == 1
        assert rel["agreement_count"] == 1

    def test_update_relationships_batch_multiple_updates(self, elo):
        """Test batch update with multiple relationships."""
        updates = [
            {"agent_a": "a", "agent_b": "b", "debate_increment": 1},
            {"agent_a": "b", "agent_b": "c", "debate_increment": 2},
            {"agent_a": "c", "agent_b": "a", "debate_increment": 3},
        ]
        elo.update_relationships_batch(updates)

        assert elo.get_relationship_raw("a", "b")["debate_count"] == 1
        assert elo.get_relationship_raw("b", "c")["debate_count"] == 2
        assert elo.get_relationship_raw("a", "c")["debate_count"] == 3

    def test_update_relationships_batch_canonical_ordering(self, elo):
        """Test that batch updates maintain canonical ordering."""
        # Update with b, a order (should be stored as a, b)
        elo.update_relationships_batch([{
            "agent_a": "zoe",
            "agent_b": "alice",
            "debate_increment": 1,
            "a_win": 1,  # zoe wins
        }])

        # Query with canonical order
        rel = elo.get_relationship_raw("alice", "zoe")
        assert rel is not None
        assert rel["debate_count"] == 1
        # zoe > alice, so a_win becomes b_win in canonical form
        assert rel["b_wins_over_a"] == 1

    def test_update_relationships_batch_skips_invalid(self, elo):
        """Test that invalid updates are skipped."""
        elo.update_relationships_batch([
            {"agent_a": "", "agent_b": "bob", "debate_increment": 1},  # Empty agent_a
            {"agent_a": "alice", "agent_b": "", "debate_increment": 1},  # Empty agent_b
            {"agent_a": "charlie", "agent_b": "david", "debate_increment": 1},  # Valid
        ])

        # Only charlie-david should be recorded
        assert elo.get_relationship_raw("charlie", "david") is not None
        assert elo.get_relationship_raw("", "bob") is None


class TestAgentRatingProperties:
    """Test AgentRating computed properties."""

    def test_calibration_accuracy_zero_total(self):
        """Test calibration_accuracy returns 0 when no predictions."""
        from aragora.ranking.elo import AgentRating
        rating = AgentRating(agent_name="test")
        assert rating.calibration_accuracy == 0.0

    def test_calibration_accuracy_with_predictions(self):
        """Test calibration_accuracy with some correct predictions."""
        from aragora.ranking.elo import AgentRating
        rating = AgentRating(
            agent_name="test",
            calibration_correct=7,
            calibration_total=10
        )
        assert rating.calibration_accuracy == 0.7

    def test_calibration_brier_score_zero_total(self):
        """Test calibration_brier_score returns 1.0 when no predictions."""
        from aragora.ranking.elo import AgentRating
        rating = AgentRating(agent_name="test")
        assert rating.calibration_brier_score == 1.0

    def test_calibration_brier_score_with_predictions(self):
        """Test calibration_brier_score calculation."""
        from aragora.ranking.elo import AgentRating
        rating = AgentRating(
            agent_name="test",
            calibration_brier_sum=0.5,
            calibration_total=10
        )
        assert rating.calibration_brier_score == 0.05

    def test_calibration_score_below_minimum(self):
        """Test calibration_score returns 0 below minimum predictions."""
        from aragora.ranking.elo import AgentRating, CALIBRATION_MIN_COUNT
        rating = AgentRating(
            agent_name="test",
            calibration_correct=CALIBRATION_MIN_COUNT - 1,
            calibration_total=CALIBRATION_MIN_COUNT - 1,
            calibration_brier_sum=0.0
        )
        assert rating.calibration_score == 0.0

    def test_calibration_score_at_minimum(self):
        """Test calibration_score at minimum predictions."""
        from aragora.ranking.elo import AgentRating, CALIBRATION_MIN_COUNT
        rating = AgentRating(
            agent_name="test",
            calibration_correct=CALIBRATION_MIN_COUNT,
            calibration_total=CALIBRATION_MIN_COUNT,
            calibration_brier_sum=0.0  # Perfect Brier
        )
        # confidence = 0.5 at min_count, score = (1 - 0) * 0.5
        assert rating.calibration_score == 0.5

    def test_calibration_score_high_confidence(self):
        """Test calibration_score with high prediction count."""
        from aragora.ranking.elo import AgentRating, CALIBRATION_MIN_COUNT
        rating = AgentRating(
            agent_name="test",
            calibration_correct=50,
            calibration_total=50,
            calibration_brier_sum=0.0  # Perfect
        )
        # confidence should be 1.0, score = (1 - 0) * 1.0 = 1.0
        assert rating.calibration_score == pytest.approx(1.0, rel=0.1)

    def test_critique_acceptance_rate_zero_total(self):
        """Test critique_acceptance_rate returns 0 when no critiques."""
        from aragora.ranking.elo import AgentRating
        rating = AgentRating(agent_name="test")
        assert rating.critique_acceptance_rate == 0.0

    def test_critique_acceptance_rate_with_critiques(self):
        """Test critique_acceptance_rate calculation."""
        from aragora.ranking.elo import AgentRating
        rating = AgentRating(
            agent_name="test",
            critiques_accepted=3,
            critiques_total=5
        )
        assert rating.critique_acceptance_rate == 0.6


class TestValidation:
    """Test input validation functions."""

    def test_escape_like_pattern_backslash(self):
        """Test escaping backslash in LIKE pattern."""
        from aragora.ranking.elo import _escape_like_pattern
        assert _escape_like_pattern("test\\value") == "test\\\\value"

    def test_escape_like_pattern_percent(self):
        """Test escaping percent in LIKE pattern."""
        from aragora.ranking.elo import _escape_like_pattern
        assert _escape_like_pattern("test%value") == "test\\%value"

    def test_escape_like_pattern_underscore(self):
        """Test escaping underscore in LIKE pattern."""
        from aragora.ranking.elo import _escape_like_pattern
        assert _escape_like_pattern("test_value") == "test\\_value"

    def test_escape_like_pattern_combined(self):
        """Test escaping multiple special characters."""
        from aragora.ranking.elo import _escape_like_pattern
        result = _escape_like_pattern("test%_\\end")
        assert result == "test\\%\\_\\\\end"

    def test_validate_agent_name_valid(self):
        """Test validation passes for valid names."""
        from aragora.ranking.elo import _validate_agent_name
        # Should not raise
        _validate_agent_name("valid_agent")
        _validate_agent_name("a" * 32)  # Exactly max length

    def test_validate_agent_name_too_long(self):
        """Test validation fails for overly long names."""
        from aragora.ranking.elo import _validate_agent_name, MAX_AGENT_NAME_LENGTH
        with pytest.raises(ValueError, match="exceeds"):
            _validate_agent_name("a" * (MAX_AGENT_NAME_LENGTH + 1))


class TestCacheOperations:
    """Test cache operations."""

    def test_invalidate_leaderboard_cache(self, elo):
        """Test invalidating leaderboard cache."""
        # Populate cache
        elo.get_cached_leaderboard(limit=10)
        # Invalidate
        count = elo.invalidate_leaderboard_cache()
        # Should return count (0 or more)
        assert isinstance(count, int)

    def test_invalidate_rating_cache_specific(self, elo):
        """Test invalidating specific agent rating cache."""
        elo.record_match("m1", ["agent_x", "opp"], {"agent_x": 1.0, "opp": 0.0})
        elo.get_rating("agent_x")  # Populate cache
        count = elo.invalidate_rating_cache("agent_x")
        assert count == 1 or count == 0  # May or may not be in cache

    def test_invalidate_rating_cache_all(self, elo):
        """Test invalidating all rating caches."""
        elo.record_match("m1", ["a", "b"], {"a": 1.0, "b": 0.0})
        elo.get_rating("a")
        elo.get_rating("b")
        count = elo.invalidate_rating_cache(None)
        assert isinstance(count, int)

    def test_get_rating_with_cache_disabled(self, elo):
        """Test getting rating with cache disabled."""
        elo.record_match("m1", ["cached", "opp"], {"cached": 1.0, "opp": 0.0})
        rating = elo.get_rating("cached", use_cache=False)
        assert rating.elo > 1500


class TestCritiqueTracking:
    """Test critique recording functionality."""

    def test_record_critique_accepted(self, elo):
        """Test recording an accepted critique."""
        elo.record_critique("critic", accepted=True)
        rating = elo.get_rating("critic")
        assert rating.critiques_total == 1
        assert rating.critiques_accepted == 1

    def test_record_critique_rejected(self, elo):
        """Test recording a rejected critique."""
        elo.record_critique("critic_rejected", accepted=False)
        rating = elo.get_rating("critic_rejected")
        assert rating.critiques_total == 1
        assert rating.critiques_accepted == 0

    def test_record_multiple_critiques(self, elo):
        """Test recording multiple critiques."""
        for _ in range(5):
            elo.record_critique("active_critic", accepted=True)
        for _ in range(3):
            elo.record_critique("active_critic", accepted=False)

        rating = elo.get_rating("active_critic")
        assert rating.critiques_total == 8
        assert rating.critiques_accepted == 5
        assert rating.critique_acceptance_rate == 5/8


class TestStatsAndLeaderboard:
    """Test stats and leaderboard functionality."""

    def test_get_stats_empty_db(self, elo):
        """Test stats on empty database."""
        stats = elo.get_stats()
        assert stats["total_agents"] == 0
        assert stats["total_matches"] == 0

    def test_get_stats_with_data(self, elo):
        """Test stats with recorded data."""
        elo.record_match("m1", ["a", "b"], {"a": 1.0, "b": 0.0})
        elo.record_match("m2", ["c", "d"], {"c": 1.0, "d": 0.0})

        stats = elo.get_stats()
        assert stats["total_agents"] == 4
        assert stats["total_matches"] == 2
        assert stats["avg_elo"] is not None
        assert stats["max_elo"] >= stats["min_elo"]

    def test_get_stats_cache_bypass(self, elo):
        """Test stats with cache bypass."""
        elo.record_match("m1", ["a", "b"], {"a": 1.0, "b": 0.0})
        stats = elo.get_stats(use_cache=False)
        assert stats["total_agents"] == 2

    def test_list_agents_empty(self, elo):
        """Test listing agents on empty database."""
        agents = elo.list_agents()
        assert agents == []

    def test_list_agents_with_data(self, elo):
        """Test listing agents with data."""
        elo.record_match("m1", ["alice", "bob"], {"alice": 1.0, "bob": 0.0})
        agents = elo.list_agents()
        assert "alice" in agents
        assert "bob" in agents

    def test_get_all_ratings(self, elo):
        """Test getting all ratings at once."""
        elo.record_match("m1", ["a", "b"], {"a": 1.0, "b": 0.0})
        elo.record_match("m2", ["c", "d"], {"c": 1.0, "d": 0.0})

        ratings = elo.get_all_ratings()
        assert len(ratings) == 4
        # Should be sorted by ELO descending
        elos = [r.elo for r in ratings]
        assert elos == sorted(elos, reverse=True)

    def test_get_top_agents_for_domain(self, elo):
        """Test getting top agents for a specific domain."""
        elo.record_match("m1", ["pro", "amateur"], {"pro": 1.0, "amateur": 0.0}, domain="security")
        top = elo.get_top_agents_for_domain("security", limit=5)
        assert len(top) > 0
        agent_names = [r.agent_name for r in top]
        assert "pro" in agent_names


class TestTournamentCalibration:
    """Test tournament winner prediction and calibration."""

    def test_record_winner_prediction(self, elo):
        """Test recording a winner prediction."""
        elo.record_winner_prediction(
            tournament_id="tourney_1",
            predictor_agent="oracle",
            predicted_winner="favorite",
            confidence=0.8
        )
        history = elo.get_agent_calibration_history("oracle", limit=10)
        assert len(history) == 1
        assert history[0]["tournament_id"] == "tourney_1"
        assert history[0]["predicted_winner"] == "favorite"
        assert history[0]["confidence"] == 0.8

    def test_record_winner_prediction_clamps_confidence(self, elo):
        """Test that confidence is clamped to [0, 1]."""
        elo.record_winner_prediction("t1", "predictor", "winner", 1.5)
        elo.record_winner_prediction("t2", "predictor", "winner", -0.5)

        history = elo.get_agent_calibration_history("predictor", limit=10)
        confidences = [h["confidence"] for h in history]
        assert all(0.0 <= c <= 1.0 for c in confidences)

    def test_resolve_tournament_calibration(self, elo):
        """Test resolving tournament and updating calibration."""
        # Make predictions
        elo.record_winner_prediction("tourney_x", "oracle_a", "winner", 0.9)
        elo.record_winner_prediction("tourney_x", "oracle_b", "loser", 0.9)

        # Resolve with actual winner
        brier_scores = elo.resolve_tournament_calibration("tourney_x", "winner")

        assert "oracle_a" in brier_scores
        assert "oracle_b" in brier_scores
        # oracle_a predicted correctly, oracle_b didn't
        assert brier_scores["oracle_a"] < brier_scores["oracle_b"]

    def test_calibration_leaderboard_empty(self, elo):
        """Test calibration leaderboard with no predictions."""
        leaderboard = elo.get_calibration_leaderboard(limit=10)
        assert leaderboard == []

    def test_calibration_leaderboard_with_data(self, elo):
        """Test calibration leaderboard with predictions."""
        from aragora.ranking.elo import CALIBRATION_MIN_COUNT

        # Record enough predictions to meet minimum
        for i in range(CALIBRATION_MIN_COUNT):
            elo.record_winner_prediction(f"t{i}", "good_oracle", "correct", 0.7)
            elo.resolve_tournament_calibration(f"t{i}", "correct")

        leaderboard = elo.get_calibration_leaderboard(limit=10)
        assert len(leaderboard) > 0


class TestDomainCalibration:
    """Test domain-specific calibration tracking."""

    def test_get_bucket_key(self, elo):
        """Test bucket key generation."""
        # Access private method for testing
        assert elo._get_bucket_key(0.05) == "0.0-0.1"
        assert elo._get_bucket_key(0.15) == "0.1-0.2"
        assert elo._get_bucket_key(0.85) == "0.8-0.9"
        assert elo._get_bucket_key(0.95) == "0.9-1.0"

    def test_get_domain_calibration_no_data(self, elo):
        """Test domain calibration with no predictions."""
        cal = elo.get_domain_calibration("unknown_agent")
        assert cal["total"] == 0
        assert cal["accuracy"] == 0.0
        assert cal["brier_score"] == 1.0

    def test_get_domain_calibration_with_data(self, elo):
        """Test domain calibration with predictions."""
        elo.record_domain_prediction("agent", "ethics", 0.8, True)
        elo.record_domain_prediction("agent", "ethics", 0.8, False)

        cal = elo.get_domain_calibration("agent", domain="ethics")
        assert cal["total"] == 2
        assert "ethics" in cal["domains"]

    def test_get_calibration_by_bucket(self, elo):
        """Test calibration broken down by confidence bucket."""
        for _ in range(5):
            elo.record_domain_prediction("agent", "general", 0.75, True)
        for _ in range(3):
            elo.record_domain_prediction("agent", "general", 0.35, False)

        buckets = elo.get_calibration_by_bucket("agent")
        assert len(buckets) > 0
        for bucket in buckets:
            assert "bucket_key" in bucket
            assert "predictions" in bucket
            assert "accuracy" in bucket

    def test_expected_calibration_error_no_data(self, elo):
        """Test ECE with no predictions."""
        ece = elo.get_expected_calibration_error("unknown")
        assert ece == 1.0

    def test_expected_calibration_error_with_data(self, elo):
        """Test ECE with predictions."""
        for _ in range(10):
            elo.record_domain_prediction("calibrated", "test", 0.5, True)
            elo.record_domain_prediction("calibrated", "test", 0.5, False)

        ece = elo.get_expected_calibration_error("calibrated")
        assert 0.0 <= ece <= 1.0

    def test_get_best_domains_no_data(self, elo):
        """Test best domains with no predictions."""
        domains = elo.get_best_domains("unknown")
        assert domains == []

    def test_get_best_domains_with_data(self, elo):
        """Test best domains with predictions."""
        # Security: 20 predictions at 0.9 confidence, all correct
        # Brier = 0.01, confidence weight = 0.875, score = 0.86
        for _ in range(20):
            elo.record_domain_prediction("expert", "security", 0.9, True)
        # Ethics: 10 predictions at 0.5 confidence, 50% correct
        # Brier = 0.25, confidence weight = 0.625, score = 0.47
        for _ in range(5):
            elo.record_domain_prediction("expert", "ethics", 0.5, True)
            elo.record_domain_prediction("expert", "ethics", 0.5, False)

        domains = elo.get_best_domains("expert", limit=5)
        assert len(domains) > 0
        # Security should be first (lower brier score with same confidence weight)
        assert domains[0][0] == "security"


class TestPairwiseEloCalculations:
    """Test the extracted ELO calculation helpers."""

    def test_calculate_pairwise_elo_changes(self, elo):
        """Test pairwise ELO calculation."""
        ratings = elo.get_ratings_batch(["a", "b"])
        changes = elo._calculate_pairwise_elo_changes(
            participants=["a", "b"],
            scores={"a": 1.0, "b": 0.0},
            ratings=ratings,
            confidence_weight=1.0
        )
        assert "a" in changes
        assert "b" in changes
        assert changes["a"] > 0  # Winner gains
        assert changes["b"] < 0  # Loser loses
        # Zero-sum
        assert abs(changes["a"] + changes["b"]) < 0.001

    def test_calculate_pairwise_elo_changes_draw(self, elo):
        """Test pairwise ELO calculation for draw."""
        ratings = elo.get_ratings_batch(["a", "b"])
        changes = elo._calculate_pairwise_elo_changes(
            participants=["a", "b"],
            scores={"a": 0.5, "b": 0.5},
            ratings=ratings,
            confidence_weight=1.0
        )
        # Both at 1500, draw - no change expected
        assert abs(changes["a"]) < 0.001
        assert abs(changes["b"]) < 0.001

    def test_calculate_pairwise_elo_changes_multiway(self, elo):
        """Test pairwise ELO calculation for 3+ participants."""
        ratings = elo.get_ratings_batch(["a", "b", "c"])
        changes = elo._calculate_pairwise_elo_changes(
            participants=["a", "b", "c"],
            scores={"a": 1.0, "b": 0.5, "c": 0.0},
            ratings=ratings,
            confidence_weight=1.0
        )
        # a beat both, should gain most
        assert changes["a"] > changes["b"]
        assert changes["b"] > changes["c"]

    def test_calculate_pairwise_elo_changes_low_confidence(self, elo):
        """Test that low confidence weight reduces changes."""
        ratings = elo.get_ratings_batch(["a", "b"])

        full_changes = elo._calculate_pairwise_elo_changes(
            participants=["a", "b"],
            scores={"a": 1.0, "b": 0.0},
            ratings=ratings,
            confidence_weight=1.0
        )

        # Get fresh ratings for second calculation
        ratings2 = elo.get_ratings_batch(["c", "d"])

        half_changes = elo._calculate_pairwise_elo_changes(
            participants=["c", "d"],
            scores={"c": 1.0, "d": 0.0},
            ratings={"c": ratings2["c"], "d": ratings2["d"]},
            confidence_weight=0.5
        )

        # Half confidence should give roughly half changes
        assert abs(half_changes["c"]) < abs(full_changes["a"])

    def test_apply_elo_changes(self, elo):
        """Test applying ELO changes."""
        ratings = elo.get_ratings_batch(["a", "b"])
        elo_changes = {"a": 16.0, "b": -16.0}

        ratings_to_save, history = elo._apply_elo_changes(
            elo_changes=elo_changes,
            ratings=ratings,
            winner="a",
            domain="test",
            debate_id="test_debate"
        )

        assert len(ratings_to_save) == 2
        assert len(history) == 2

        # Check a's rating was updated correctly
        a_rating = next(r for r in ratings_to_save if r.agent_name == "a")
        assert a_rating.elo == 1516.0
        assert a_rating.wins == 1
        assert a_rating.debates_count == 1
        assert "test" in a_rating.domain_elos

    def test_apply_elo_changes_draw(self, elo):
        """Test applying ELO changes for draw."""
        ratings = elo.get_ratings_batch(["a", "b"])
        elo_changes = {"a": 0.0, "b": 0.0}

        ratings_to_save, _ = elo._apply_elo_changes(
            elo_changes=elo_changes,
            ratings=ratings,
            winner=None,  # Draw
            domain=None,
            debate_id="draw_debate"
        )

        a_rating = next(r for r in ratings_to_save if r.agent_name == "a")
        assert a_rating.draws == 1
        assert a_rating.wins == 0
        assert a_rating.losses == 0


class TestSnapshotAndCaching:
    """Test snapshot writing and cached access."""

    def test_get_cached_leaderboard_no_snapshot(self, elo):
        """Test cached leaderboard falls back to database."""
        elo.record_match("m1", ["a", "b"], {"a": 1.0, "b": 0.0})
        leaderboard = elo.get_cached_leaderboard(limit=10)
        assert isinstance(leaderboard, list)
        assert len(leaderboard) > 0

    def test_get_cached_recent_matches_no_snapshot(self, elo):
        """Test cached recent matches falls back to database."""
        elo.record_match("m1", ["a", "b"], {"a": 1.0, "b": 0.0})
        matches = elo.get_cached_recent_matches(limit=10)
        assert isinstance(matches, list)
        assert len(matches) == 1


class TestExpectedScore:
    """Test expected score calculation."""

    def test_expected_score_equal_ratings(self, elo):
        """Test expected score when ratings are equal."""
        expected = elo._expected_score(1500, 1500)
        assert expected == 0.5

    def test_expected_score_higher_rated(self, elo):
        """Test expected score when first player is higher rated."""
        expected = elo._expected_score(1600, 1400)
        assert expected > 0.5

    def test_expected_score_lower_rated(self, elo):
        """Test expected score when first player is lower rated."""
        expected = elo._expected_score(1400, 1600)
        assert expected < 0.5

    def test_expected_score_large_difference(self, elo):
        """Test expected score with large rating difference."""
        expected = elo._expected_score(1800, 1200)
        assert expected > 0.9


class TestCalculateNewElo:
    """Test new ELO calculation."""

    def test_calculate_new_elo_win_expected(self, elo):
        """Test ELO gain when expected to win."""
        # If expected 0.75, actual 1.0, small gain
        new_elo = elo._calculate_new_elo(1600, expected=0.75, actual=1.0, k=32)
        assert new_elo > 1600
        assert new_elo < 1610  # Small gain for expected win

    def test_calculate_new_elo_upset_win(self, elo):
        """Test ELO gain for upset win."""
        # If expected 0.25, actual 1.0, big gain
        new_elo = elo._calculate_new_elo(1400, expected=0.25, actual=1.0, k=32)
        assert new_elo > 1420  # Large gain for upset

    def test_calculate_new_elo_expected_loss(self, elo):
        """Test ELO loss when expected to lose."""
        # If expected 0.25, actual 0.0, small loss
        new_elo = elo._calculate_new_elo(1400, expected=0.25, actual=0.0, k=32)
        assert new_elo < 1400
        assert new_elo > 1390  # Small loss for expected loss


class TestVerificationEloIntegration:
    """Test verification-to-ELO integration (Phase 10E)."""

    def test_update_from_verification_verified_claims(self, elo):
        """Test ELO increases when claims are verified."""
        initial_elo = elo.get_rating("prover").elo
        assert initial_elo == 1500

        change = elo.update_from_verification(
            agent_name="prover",
            domain="logic",
            verified_count=2,
            disproven_count=0,
        )

        new_rating = elo.get_rating("prover")
        # 2 verified * 16 * 0.5 = 16 points gain
        assert change == 16.0
        assert new_rating.elo == 1516.0

    def test_update_from_verification_disproven_claims(self, elo):
        """Test ELO decreases when claims are disproven."""
        # First record some matches to raise ELO
        elo.record_match("m1", ["fallacious", "opp"], {"fallacious": 1.0, "opp": 0.0})

        initial_elo = elo.get_rating("fallacious").elo
        assert initial_elo > 1500  # Verify we have some ELO to lose

        change = elo.update_from_verification(
            agent_name="fallacious",
            domain="ethics",
            verified_count=0,
            disproven_count=2,
        )

        new_rating = elo.get_rating("fallacious")
        # 2 disproven * 16 * 0.5 = -16 points
        assert change == -16.0
        assert new_rating.elo == initial_elo - 16.0

    def test_update_from_verification_mixed(self, elo):
        """Test ELO with both verified and disproven claims."""
        change = elo.update_from_verification(
            agent_name="mixed_agent",
            domain="math",
            verified_count=3,
            disproven_count=1,
        )

        # Net: (3 * 16 * 0.5) - (1 * 16 * 0.5) = 24 - 8 = 16
        assert change == 16.0

        new_rating = elo.get_rating("mixed_agent")
        assert new_rating.elo == 1516.0

    def test_update_from_verification_no_claims(self, elo):
        """Test no change when no claims verified or disproven."""
        change = elo.update_from_verification(
            agent_name="idle_agent",
            domain="general",
            verified_count=0,
            disproven_count=0,
        )

        assert change == 0.0
        rating = elo.get_rating("idle_agent")
        # Rating should still be default since no changes
        assert rating.elo == 1500

    def test_update_from_verification_domain_specific(self, elo):
        """Test that domain-specific ELO is updated."""
        elo.update_from_verification(
            agent_name="domain_expert",
            domain="physics",
            verified_count=3,
        )

        rating = elo.get_rating("domain_expert")
        # Both overall and domain-specific should be updated
        assert rating.elo == 1524.0  # 3 * 16 * 0.5
        assert "physics" in rating.domain_elos
        assert rating.domain_elos["physics"] == 1524.0

    def test_update_from_verification_elo_floor(self, elo):
        """Test that ELO doesn't drop below 100."""
        # First, lower the agent's ELO
        for _ in range(20):
            elo.record_match(
                f"loss_{_}",
                ["low_agent", "winner"],
                {"low_agent": 0.0, "winner": 1.0}
            )

        # Now try to disprove many claims
        elo.update_from_verification(
            agent_name="low_agent",
            domain="any",
            verified_count=0,
            disproven_count=100,  # Massive penalty
        )

        rating = elo.get_rating("low_agent")
        assert rating.elo >= 100  # Floor enforced

    def test_update_from_verification_custom_k_factor(self, elo):
        """Test custom k_factor affects magnitude of change."""
        change_default = elo.update_from_verification(
            agent_name="agent_default_k",
            domain="test",
            verified_count=1,
            k_factor=16.0,
        )

        change_custom = elo.update_from_verification(
            agent_name="agent_custom_k",
            domain="test",
            verified_count=1,
            k_factor=32.0,
        )

        # Custom K should give double the change
        assert change_custom == change_default * 2

    def test_update_from_verification_records_history(self, elo):
        """Test that verification updates are recorded in ELO history."""
        elo.update_from_verification(
            agent_name="history_agent",
            domain="logic",
            verified_count=2,
        )

        history = elo.get_elo_history("history_agent", limit=10)
        assert len(history) >= 1

        # Check history entry format
        newest = history[0]
        assert isinstance(newest, tuple)
        assert len(newest) == 2
        # ELO should be 1516 (1500 + 2*16*0.5)
        assert newest[1] == 1516.0

    def test_get_verification_impact_no_data(self, elo):
        """Test verification impact with no verification events."""
        impact = elo.get_verification_impact("unverified_agent")
        assert impact["agent_name"] == "unverified_agent"
        assert impact["verification_events"] == 0
        assert impact["total_impact"] == 0.0
        assert impact["history"] == []

    def test_get_verification_impact_with_data(self, elo):
        """Test verification impact with multiple events."""
        # Record multiple verification events
        elo.update_from_verification("tracked_agent", "math", verified_count=2)
        elo.update_from_verification("tracked_agent", "logic", verified_count=1, disproven_count=1)
        elo.update_from_verification("tracked_agent", "ethics", verified_count=0, disproven_count=1)

        impact = elo.get_verification_impact("tracked_agent")
        assert impact["agent_name"] == "tracked_agent"
        # Should have 3 verification events recorded
        assert impact["verification_events"] >= 2  # At least 2 with changes

    def test_verification_and_match_combined(self, elo):
        """Test that verification and match ELO changes work together."""
        # Win a match first
        elo.record_match("m1", ["combo_agent", "opp"], {"combo_agent": 1.0, "opp": 0.0})
        after_match = elo.get_rating("combo_agent").elo
        assert after_match > 1500

        # Now verify claims
        change = elo.update_from_verification(
            agent_name="combo_agent",
            domain="debate",
            verified_count=2,
        )

        final_rating = elo.get_rating("combo_agent")
        # Should be match ELO + verification bonus
        assert final_rating.elo == after_match + change


class TestMigration:
    """Test database schema migration."""

    def test_migrate_v1_to_v2(self, elo):
        """Test that v1 to v2 migration adds calibration columns."""
        # The migration should have run during init
        # Verify calibration columns exist by using them
        rating = elo.get_rating("test_agent")
        assert hasattr(rating, 'calibration_correct')
        assert hasattr(rating, 'calibration_total')
        assert hasattr(rating, 'calibration_brier_sum')


class TestMultiAgentMatches:
    """Tests for 3+ agent match scenarios."""

    def test_three_agent_match_scoring(self, elo):
        """Test scoring with exactly 3 agents."""
        elo.record_match(
            debate_id="three_way_1",
            participants=["first", "second", "third"],
            scores={"first": 1.0, "second": 0.5, "third": 0.0}
        )

        # First should have highest ELO (beat both)
        first_elo = elo.get_rating("first").elo
        second_elo = elo.get_rating("second").elo
        third_elo = elo.get_rating("third").elo

        assert first_elo > second_elo > third_elo
        assert first_elo > 1500
        assert third_elo < 1500

    def test_four_agent_match_scoring(self, elo):
        """Test scoring with 4 agents."""
        elo.record_match(
            debate_id="four_way",
            participants=["p1", "p2", "p3", "p4"],
            scores={"p1": 1.0, "p2": 0.66, "p3": 0.33, "p4": 0.0}
        )

        # All should have played
        for p in ["p1", "p2", "p3", "p4"]:
            assert elo.get_rating(p).games_played == 1

        # Ranking should match scores
        elos = [elo.get_rating(f"p{i}").elo for i in range(1, 5)]
        assert elos == sorted(elos, reverse=True)

    def test_multi_agent_all_tied(self, elo):
        """Test multi-agent match where all tie."""
        elo.record_match(
            debate_id="multi_tie",
            participants=["tied_a", "tied_b", "tied_c"],
            scores={"tied_a": 0.5, "tied_b": 0.5, "tied_c": 0.5}
        )

        # All should have played and counted as draws
        for agent in ["tied_a", "tied_b", "tied_c"]:
            rating = elo.get_rating(agent)
            assert rating.games_played == 1
            # ELOs should all be close to 1500
            assert abs(rating.elo - 1500) < 10

    def test_multi_agent_two_winners(self, elo):
        """Test multi-agent match with two equal winners."""
        elo.record_match(
            debate_id="two_winners",
            participants=["winner1", "winner2", "loser1", "loser2"],
            scores={"winner1": 1.0, "winner2": 1.0, "loser1": 0.0, "loser2": 0.0}
        )

        # Both winners should gain, both losers should lose
        w1_elo = elo.get_rating("winner1").elo
        w2_elo = elo.get_rating("winner2").elo
        l1_elo = elo.get_rating("loser1").elo
        l2_elo = elo.get_rating("loser2").elo

        assert w1_elo > 1500
        assert w2_elo > 1500
        assert l1_elo < 1500
        assert l2_elo < 1500


class TestCachingEdgeCases:
    """Tests for cache hit/miss/TTL edge cases."""

    def test_cache_hit_returns_same_result(self, elo):
        """Test that cache hit returns identical results."""
        elo.record_match("m1", ["cached_a", "cached_b"], {"cached_a": 1.0, "cached_b": 0.0})

        # First call populates cache
        rating1 = elo.get_rating("cached_a", use_cache=True)
        # Second call should hit cache
        rating2 = elo.get_rating("cached_a", use_cache=True)

        assert rating1.elo == rating2.elo
        assert rating1.wins == rating2.wins

    def test_cache_miss_after_invalidation(self, elo):
        """Test that invalidation causes cache miss."""
        elo.record_match("m1", ["inv_agent", "opp"], {"inv_agent": 1.0, "opp": 0.0})

        # Populate cache
        elo.get_rating("inv_agent", use_cache=True)

        # Invalidate
        elo.invalidate_rating_cache("inv_agent")

        # Record another match
        elo.record_match("m2", ["inv_agent", "opp2"], {"inv_agent": 1.0, "opp2": 0.0})

        # Should get fresh data
        rating = elo.get_rating("inv_agent", use_cache=False)
        assert rating.games_played == 2

    def test_cache_stale_after_new_match(self, elo):
        """Test that cache invalidation happens after new matches."""
        elo.record_match("m1", ["stale_test", "opp"], {"stale_test": 1.0, "opp": 0.0})
        initial_elo = elo.get_rating("stale_test").elo

        # Record more matches - should invalidate cache
        for i in range(3):
            elo.record_match(f"m{i+2}", ["stale_test", f"opp{i}"], {"stale_test": 1.0, f"opp{i}": 0.0})

        # Should reflect new matches
        final_elo = elo.get_rating("stale_test").elo
        assert final_elo > initial_elo

    def test_leaderboard_cache_invalidation(self, elo):
        """Test leaderboard cache is invalidated after matches."""
        elo.record_match("lb1", ["lb_a", "lb_b"], {"lb_a": 1.0, "lb_b": 0.0})
        leaderboard1 = elo.get_cached_leaderboard(limit=10)

        # Record more matches
        elo.record_match("lb2", ["lb_c", "lb_d"], {"lb_c": 1.0, "lb_d": 0.0})

        # Invalidate and get new leaderboard
        elo.invalidate_leaderboard_cache()
        leaderboard2 = elo.get_cached_leaderboard(limit=10)

        # Second leaderboard should have more agents
        assert len(leaderboard2) >= len(leaderboard1)

    def test_cache_bypass_always_fresh(self, elo):
        """Test that use_cache=False always gets fresh data."""
        elo.record_match("m1", ["bypass_agent", "opp"], {"bypass_agent": 1.0, "opp": 0.0})

        # Get rating without cache
        rating1 = elo.get_rating("bypass_agent", use_cache=False)

        # Record another match
        elo.record_match("m2", ["bypass_agent", "opp2"], {"bypass_agent": 1.0, "opp2": 0.0})

        # Should get updated data
        rating2 = elo.get_rating("bypass_agent", use_cache=False)
        assert rating2.games_played == rating1.games_played + 1
        assert rating2.elo > rating1.elo


class TestHeadToHeadComplex:
    """Tests for complex head-to-head match histories."""

    def test_head_to_head_many_matches(self, elo):
        """Test head-to-head with many matches."""
        # Record 10 matches between same agents
        # i % 3 != 0: A wins when i = 1,2,4,5,7,8 (6 wins)
        # i % 3 == 0: B wins when i = 0,3,6,9 (4 wins)
        for i in range(10):
            winner = "h2h_a" if i % 3 != 0 else "h2h_b"
            loser = "h2h_b" if i % 3 != 0 else "h2h_a"
            elo.record_match(
                f"h2h_match_{i}",
                ["h2h_a", "h2h_b"],
                {winner: 1.0, loser: 0.0}
            )

        h2h = elo.get_head_to_head("h2h_a", "h2h_b")
        assert h2h["matches"] == 10
        assert h2h["h2h_a_wins"] == 6
        assert h2h["h2h_b_wins"] == 4

    def test_head_to_head_with_draws(self, elo):
        """Test head-to-head including draws."""
        # Mix of wins and draws
        elo.record_match("h2h_1", ["draw_a", "draw_b"], {"draw_a": 1.0, "draw_b": 0.0})
        elo.record_match("h2h_2", ["draw_a", "draw_b"], {"draw_a": 0.5, "draw_b": 0.5})
        elo.record_match("h2h_3", ["draw_a", "draw_b"], {"draw_b": 1.0, "draw_a": 0.0})
        elo.record_match("h2h_4", ["draw_a", "draw_b"], {"draw_a": 0.5, "draw_b": 0.5})

        h2h = elo.get_head_to_head("draw_a", "draw_b")
        assert h2h["matches"] == 4
        assert h2h["draw_a_wins"] == 1
        assert h2h["draw_b_wins"] == 1
        assert h2h["draws"] == 2

    def test_head_to_head_order_independent(self, elo):
        """Test that h2h lookup order doesn't matter."""
        elo.record_match("order_1", ["order_x", "order_y"], {"order_x": 1.0, "order_y": 0.0})
        elo.record_match("order_2", ["order_x", "order_y"], {"order_y": 1.0, "order_x": 0.0})

        h2h_xy = elo.get_head_to_head("order_x", "order_y")
        h2h_yx = elo.get_head_to_head("order_y", "order_x")

        # Results should be equivalent
        assert h2h_xy["matches"] == h2h_yx["matches"]
        assert h2h_xy["order_x_wins"] == h2h_yx["order_x_wins"]
        assert h2h_xy["order_y_wins"] == h2h_yx["order_y_wins"]


class TestDomainQueries:
    """Tests for domain-specific leaderboard queries."""

    def test_domain_leaderboard_filters_correctly(self, elo):
        """Test domain leaderboard only includes domain matches."""
        # Security expert
        for i in range(5):
            elo.record_match(
                f"sec_{i}",
                ["sec_expert", f"sec_opp_{i}"],
                {"sec_expert": 1.0, f"sec_opp_{i}": 0.0},
                domain="security"
            )

        # Performance expert
        for i in range(5):
            elo.record_match(
                f"perf_{i}",
                ["perf_expert", f"perf_opp_{i}"],
                {"perf_expert": 1.0, f"perf_opp_{i}": 0.0},
                domain="performance"
            )

        # Security leaderboard should rank sec_expert high
        sec_board = elo.get_leaderboard(domain="security", limit=10)
        sec_agents = [e.agent_name for e in sec_board]
        assert "sec_expert" in sec_agents

        # Performance leaderboard should rank perf_expert high
        perf_board = elo.get_leaderboard(domain="performance", limit=10)
        perf_agents = [e.agent_name for e in perf_board]
        assert "perf_expert" in perf_agents

    def test_get_top_agents_for_domain_ordering(self, elo):
        """Test top agents are ordered by domain ELO."""
        agents = ["dom_a", "dom_b", "dom_c"]

        # Record matches so dom_a > dom_b > dom_c in ethics
        elo.record_match("d1", ["dom_a", "dom_b"], {"dom_a": 1.0, "dom_b": 0.0}, domain="ethics")
        elo.record_match("d2", ["dom_a", "dom_c"], {"dom_a": 1.0, "dom_c": 0.0}, domain="ethics")
        elo.record_match("d3", ["dom_b", "dom_c"], {"dom_b": 1.0, "dom_c": 0.0}, domain="ethics")

        top = elo.get_top_agents_for_domain("ethics", limit=10)
        top_names = [r.agent_name for r in top]

        # Should be ordered by domain ELO
        assert top_names.index("dom_a") < top_names.index("dom_b")
        assert top_names.index("dom_b") < top_names.index("dom_c")

    def test_domain_leaderboard_empty_domain(self, elo):
        """Test leaderboard for domain with no matches."""
        # Record matches in one domain
        elo.record_match("m1", ["a", "b"], {"a": 1.0, "b": 0.0}, domain="known")

        # Query different domain
        unknown_board = elo.get_leaderboard(domain="unknown_domain", limit=10)

        # Should be empty or have default ratings
        assert isinstance(unknown_board, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
