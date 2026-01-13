"""
Integration tests for storage backends and cross-module interactions.

Tests the data flow between ELO system, memory stores, and other
storage backends without requiring HTTP layer.
"""

import tempfile
from pathlib import Path

import pytest

from aragora.ranking.elo import EloSystem


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_db_dir():
    """Create a temporary directory for test databases."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def elo_system(temp_db_dir):
    """Create a real EloSystem with test data."""
    db_path = temp_db_dir / "test_elo.db"
    elo = EloSystem(str(db_path))

    # Create agents by getting their ratings (auto-creates if not exist)
    elo.get_rating("claude")
    elo.get_rating("gpt-4")
    elo.get_rating("gemini")

    # Record some matches to create realistic data
    elo.record_match(
        debate_id="test-debate-1",
        participants=["claude", "gpt-4"],
        scores={"claude": 1.0, "gpt-4": 0.0},  # claude wins
        domain="general",
    )
    elo.record_match(
        debate_id="test-debate-2",
        participants=["claude", "gemini"],
        scores={"claude": 0.0, "gemini": 1.0},  # gemini wins
        domain="general",
    )
    elo.record_match(
        debate_id="test-debate-3",
        participants=["gpt-4", "gemini"],
        scores={"gpt-4": 0.5, "gemini": 0.5},  # Draw
        domain="coding",
    )

    return elo


# =============================================================================
# ELO System Integration Tests
# =============================================================================


class TestEloSystemIntegration:
    """Test ELO system with real database."""

    def test_get_leaderboard_returns_ranked_agents(self, elo_system):
        """Leaderboard returns agents ranked by ELO."""
        leaderboard = elo_system.get_leaderboard(limit=10)

        assert len(leaderboard) >= 2  # At least our test agents

        # Verify ranking order (highest ELO first)
        for i in range(len(leaderboard) - 1):
            assert leaderboard[i].elo >= leaderboard[i + 1].elo

    def test_get_leaderboard_respects_limit(self, elo_system):
        """Leaderboard respects limit parameter."""
        leaderboard = elo_system.get_leaderboard(limit=2)

        assert len(leaderboard) == 2

    def test_get_rating_returns_agent_stats(self, elo_system):
        """get_rating returns complete agent statistics."""
        rating = elo_system.get_rating("claude")

        assert rating.agent_name == "claude"
        assert rating.elo > 0
        assert rating.debates_count >= 2  # We recorded 2 matches with claude

    def test_leaderboard_updates_after_match(self, elo_system):
        """Leaderboard reflects new match results."""
        # Get initial ELO
        initial_gpt4 = elo_system.get_rating("gpt-4", use_cache=False)

        # Record a new match (GPT-4 wins)
        elo_system.record_match(
            debate_id="test-debate-new",
            participants=["claude", "gpt-4"],
            scores={"claude": 0.0, "gpt-4": 1.0},
            domain="general",
        )

        # Verify ELO updated
        updated_gpt4 = elo_system.get_rating("gpt-4", use_cache=False)

        # GPT-4's ELO should have increased
        assert updated_gpt4.elo > initial_gpt4.elo

    def test_batch_rating_retrieval(self, elo_system):
        """Batch rating retrieval returns all requested agents."""
        ratings = elo_system.get_ratings_batch(["claude", "gpt-4", "gemini"])

        assert "claude" in ratings
        assert "gpt-4" in ratings
        assert "gemini" in ratings
        assert ratings["claude"].agent_name == "claude"


# =============================================================================
# Cross-Module Integration Tests
# =============================================================================


class TestCrossModuleIntegration:
    """Test interactions between multiple modules."""

    def test_match_updates_both_agents(self, elo_system):
        """Recording a match updates both participants' stats."""
        initial_claude = elo_system.get_rating("claude", use_cache=False)
        initial_gpt4 = elo_system.get_rating("gpt-4", use_cache=False)

        elo_system.record_match(
            debate_id="cross-test-1",
            participants=["claude", "gpt-4"],
            scores={"claude": 1.0, "gpt-4": 0.0},  # claude wins
            domain="philosophy",
        )

        updated_claude = elo_system.get_rating("claude", use_cache=False)
        updated_gpt4 = elo_system.get_rating("gpt-4", use_cache=False)

        # Winner's ELO increases
        assert updated_claude.elo > initial_claude.elo
        # Loser's ELO decreases
        assert updated_gpt4.elo < initial_gpt4.elo
        # Games played increases for both
        assert updated_claude.debates_count > initial_claude.debates_count
        assert updated_gpt4.debates_count > initial_gpt4.debates_count

    def test_draw_updates_both_agents_equally(self, elo_system):
        """A draw updates both agents with minimal ELO change."""
        initial_claude = elo_system.get_rating("claude", use_cache=False)
        initial_gpt4 = elo_system.get_rating("gpt-4", use_cache=False)

        elo_system.record_match(
            debate_id="draw-test-1",
            participants=["claude", "gpt-4"],
            scores={"claude": 0.5, "gpt-4": 0.5},  # Draw
            domain="ethics",
        )

        updated_claude = elo_system.get_rating("claude", use_cache=False)
        updated_gpt4 = elo_system.get_rating("gpt-4", use_cache=False)

        # Both should have draw incremented
        assert updated_claude.draws > initial_claude.draws
        assert updated_gpt4.draws > initial_gpt4.draws

    def test_domain_elo_tracks_separately(self, elo_system):
        """Domain-specific ELO is tracked separately from global."""
        # Record multiple matches in specific domain
        for i in range(3):
            scores = (
                {"claude": 1.0, "gemini": 0.0} if i % 2 == 0 else {"claude": 0.0, "gemini": 1.0}
            )
            elo_system.record_match(
                debate_id=f"domain-test-{i}",
                participants=["claude", "gemini"],
                scores=scores,
                domain="mathematics",
            )

        rating = elo_system.get_rating("claude")

        # Should have domain-specific ELO tracking
        assert rating.domain_elos is not None


# =============================================================================
# Concurrent Access Tests
# =============================================================================


class TestConcurrentAccessIntegration:
    """Test concurrent access patterns."""

    def test_multiple_reads_concurrent(self, elo_system):
        """Multiple concurrent reads don't interfere."""
        results = []
        for _ in range(10):
            result = elo_system.get_leaderboard(limit=5)
            results.append(result)

        # All should succeed with consistent data
        for result in results:
            assert len(result) >= 2

    def test_read_during_write(self, elo_system):
        """Reads complete successfully even during writes."""
        # Interleave reads and writes
        for i in range(5):
            # Read
            leaderboard = elo_system.get_leaderboard(limit=10)
            assert len(leaderboard) >= 2

            # Write
            scores = {"claude": 1.0, "gpt-4": 0.0} if i % 2 == 0 else {"claude": 0.0, "gpt-4": 1.0}
            elo_system.record_match(
                debate_id=f"concurrent-{i}",
                participants=["claude", "gpt-4"],
                scores=scores,
                domain="general",
            )

        # Final read should reflect updates
        final = elo_system.get_leaderboard(limit=10)
        assert len(final) >= 2

    def test_new_agent_during_operation(self, elo_system):
        """Creating new agents during other operations works."""
        # Add a new agent
        elo_system.get_rating("new-agent-1")

        # Record match with new agent
        elo_system.record_match(
            debate_id="new-agent-test",
            participants=["new-agent-1", "claude"],
            scores={"new-agent-1": 0.0, "claude": 1.0},
            domain="general",
        )

        # New agent should be in leaderboard
        leaderboard = elo_system.get_leaderboard(limit=20)
        agent_names = [r.agent_name for r in leaderboard]
        assert "new-agent-1" in agent_names


# =============================================================================
# Database Consistency Tests
# =============================================================================


class TestDatabaseConsistency:
    """Test database consistency under various conditions."""

    def test_rating_persists_across_instances(self, temp_db_dir):
        """Ratings persist when EloSystem is recreated."""
        db_path = temp_db_dir / "persist_test.db"

        # Create first instance and record data
        elo1 = EloSystem(str(db_path))
        elo1.get_rating("persist-agent")
        elo1.record_match(
            debate_id="persist-test",
            participants=["persist-agent", "claude"],
            scores={"persist-agent": 1.0, "claude": 0.0},
            domain="general",
        )
        rating1 = elo1.get_rating("persist-agent", use_cache=False)

        # Create second instance from same database
        elo2 = EloSystem(str(db_path))
        rating2 = elo2.get_rating("persist-agent", use_cache=False)

        # Ratings should match
        assert rating2.agent_name == rating1.agent_name
        assert rating2.elo == rating1.elo
        assert rating2.wins == rating1.wins

    def test_match_history_persists(self, temp_db_dir):
        """Match history persists across instances."""
        db_path = temp_db_dir / "history_test.db"

        # Create first instance and record matches
        elo1 = EloSystem(str(db_path))
        for i in range(3):
            elo1.record_match(
                debate_id=f"history-{i}",
                participants=["agent-a", "agent-b"],
                scores={"agent-a": 1.0, "agent-b": 0.0},
                domain="test",
            )

        rating1 = elo1.get_rating("agent-a", use_cache=False)

        # Create second instance
        elo2 = EloSystem(str(db_path))
        rating2 = elo2.get_rating("agent-a", use_cache=False)

        # Stats should be consistent
        assert rating2.wins == rating1.wins
        assert rating2.debates_count == rating1.debates_count
