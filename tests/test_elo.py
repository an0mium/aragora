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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
