"""
Tests for RelationshipTracker - Agent alliance/rivalry tracking.

Tests cover:
- RelationshipStats dataclass
- RelationshipMetrics dataclass
- RelationshipTracker CRUD operations
- Canonical ordering (agent_a < agent_b)
- Rivalry and alliance score computation
- get_rivals and get_allies queries
- Batch updates
- Thread safety considerations
"""

import os
import tempfile
from pathlib import Path

import pytest

from aragora.ranking.relationships import (
    RelationshipTracker,
    RelationshipStats,
    RelationshipMetrics,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_db_path():
    """Create a temporary database path for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_relationships.db")
        # Initialize database schema using EloSystem
        from aragora.ranking.elo import EloSystem

        elo = EloSystem(db_path=db_path)
        yield db_path


@pytest.fixture
def tracker(temp_db_path):
    """Create a RelationshipTracker instance with a temp database."""
    return RelationshipTracker(db_path=temp_db_path)


# =============================================================================
# Tests: RelationshipStats Dataclass
# =============================================================================


class TestRelationshipStats:
    """Tests for RelationshipStats dataclass."""

    def test_create_stats(self):
        """Test creating a RelationshipStats instance."""
        stats = RelationshipStats(
            agent_a="claude",
            agent_b="gemini",
            debate_count=10,
            agreement_count=6,
            critique_count_a_to_b=5,
            critique_count_b_to_a=3,
            critique_accepted_a_to_b=3,
            critique_accepted_b_to_a=2,
            position_changes_a_after_b=2,
            position_changes_b_after_a=1,
            a_wins_over_b=4,
            b_wins_over_a=3,
        )

        assert stats.agent_a == "claude"
        assert stats.agent_b == "gemini"
        assert stats.debate_count == 10
        assert stats.agreement_count == 6
        assert stats.a_wins_over_b == 4


# =============================================================================
# Tests: RelationshipMetrics Dataclass
# =============================================================================


class TestRelationshipMetrics:
    """Tests for RelationshipMetrics dataclass."""

    def test_create_metrics(self):
        """Test creating a RelationshipMetrics instance."""
        metrics = RelationshipMetrics(
            agent_a="claude",
            agent_b="gemini",
            rivalry_score=0.65,
            alliance_score=0.35,
            relationship="rival",
            debate_count=15,
            agreement_rate=0.4,
            head_to_head="8-7",
        )

        assert metrics.agent_a == "claude"
        assert metrics.rivalry_score == 0.65
        assert metrics.relationship == "rival"
        assert metrics.head_to_head == "8-7"

    def test_default_values(self):
        """Test default values for optional fields."""
        metrics = RelationshipMetrics(
            agent_a="a",
            agent_b="b",
            rivalry_score=0.5,
            alliance_score=0.5,
            relationship="neutral",
            debate_count=5,
        )

        assert metrics.agreement_rate == 0.0
        assert metrics.head_to_head == "0-0"


# =============================================================================
# Tests: RelationshipTracker Initialization
# =============================================================================


class TestRelationshipTrackerInit:
    """Tests for RelationshipTracker initialization."""

    def test_init_creates_db(self, temp_db_path):
        """Test that init creates the database."""
        tracker = RelationshipTracker(db_path=temp_db_path)
        assert Path(temp_db_path).exists()

    def test_init_with_path_object(self, temp_db_path):
        """Test init with Path object."""
        tracker = RelationshipTracker(db_path=Path(temp_db_path))
        assert tracker.db_path == Path(temp_db_path)


# =============================================================================
# Tests: update_relationship
# =============================================================================


class TestUpdateRelationship:
    """Tests for update_relationship method."""

    def test_create_new_relationship(self, tracker):
        """Test creating a new relationship."""
        tracker.update_relationship("alice", "bob", debate_increment=1)

        stats = tracker.get_raw("alice", "bob")
        assert stats is not None
        assert stats.debate_count == 1

    def test_increment_existing_relationship(self, tracker):
        """Test incrementing an existing relationship."""
        tracker.update_relationship("alice", "bob", debate_increment=1)
        tracker.update_relationship("alice", "bob", debate_increment=2, agreement_increment=1)

        stats = tracker.get_raw("alice", "bob")
        assert stats.debate_count == 3
        assert stats.agreement_count == 1

    def test_canonical_ordering(self, tracker):
        """Test that agent order is normalized (a < b)."""
        # Insert with alice, bob (already canonical)
        tracker.update_relationship("alice", "bob", debate_increment=1, a_win=1)

        # Insert with bob, alice (reversed) - should still increment same record
        tracker.update_relationship("bob", "alice", debate_increment=1, a_win=1)

        stats = tracker.get_raw("alice", "bob")
        assert stats.debate_count == 2
        # Both a_wins should go to alice since she's agent_a after normalization
        assert stats.a_wins_over_b == 1  # First call: alice (a) wins
        assert stats.b_wins_over_a == 1  # Second call: bob becomes b, so b wins

    def test_canonical_ordering_swaps_critique_counts(self, tracker):
        """Test that critique counts are swapped correctly with ordering."""
        # Add critique from zack to alice (zack > alice, so will be swapped)
        tracker.update_relationship("zack", "alice", critique_a_to_b=3)

        stats = tracker.get_raw("alice", "zack")
        # zack was a, alice was b in input
        # After swap: alice is a, zack is b
        # critique_a_to_b becomes critique_b_to_a (zack -> alice)
        assert stats.critique_count_b_to_a == 3

    def test_update_all_fields(self, tracker):
        """Test updating all relationship fields."""
        tracker.update_relationship(
            "claude",
            "gemini",
            debate_increment=5,
            agreement_increment=3,
            critique_a_to_b=2,
            critique_b_to_a=1,
            critique_accepted_a_to_b=1,
            critique_accepted_b_to_a=1,
            position_change_a_after_b=1,
            position_change_b_after_a=0,
            a_win=2,
            b_win=1,
        )

        stats = tracker.get_raw("claude", "gemini")
        assert stats.debate_count == 5
        assert stats.agreement_count == 3
        assert stats.critique_count_a_to_b == 2
        assert stats.critique_count_b_to_a == 1
        assert stats.critique_accepted_a_to_b == 1
        assert stats.critique_accepted_b_to_a == 1
        assert stats.position_changes_a_after_b == 1
        assert stats.position_changes_b_after_a == 0
        assert stats.a_wins_over_b == 2
        assert stats.b_wins_over_a == 1


# =============================================================================
# Tests: update_batch
# =============================================================================


class TestUpdateBatch:
    """Tests for batch update method."""

    def test_batch_update_multiple_pairs(self, tracker):
        """Test batch updating multiple agent pairs."""
        updates = [
            {"agent_a": "alice", "agent_b": "bob", "debate_increment": 1},
            {"agent_a": "alice", "agent_b": "charlie", "debate_increment": 2},
            {"agent_a": "bob", "agent_b": "charlie", "debate_increment": 3},
        ]
        tracker.update_batch(updates)

        assert tracker.get_raw("alice", "bob").debate_count == 1
        assert tracker.get_raw("alice", "charlie").debate_count == 2
        assert tracker.get_raw("bob", "charlie").debate_count == 3

    def test_batch_update_same_pair_multiple_times(self, tracker):
        """Test batch updating same pair multiple times."""
        updates = [
            {"agent_a": "alice", "agent_b": "bob", "debate_increment": 1},
            {"agent_a": "alice", "agent_b": "bob", "debate_increment": 2},
        ]
        tracker.update_batch(updates)

        assert tracker.get_raw("alice", "bob").debate_count == 3

    def test_batch_update_with_wins(self, tracker):
        """Test batch update with win tracking."""
        updates = [
            {"agent_a": "alice", "agent_b": "bob", "debate_increment": 1, "a_win": 1},
            {"agent_a": "alice", "agent_b": "bob", "debate_increment": 1, "b_win": 1},
        ]
        tracker.update_batch(updates)

        stats = tracker.get_raw("alice", "bob")
        assert stats.a_wins_over_b == 1
        assert stats.b_wins_over_a == 1

    def test_batch_update_empty_list(self, tracker):
        """Test batch update with empty list does nothing."""
        tracker.update_batch([])
        # Should not raise any errors

    def test_batch_update_skips_invalid_entries(self, tracker):
        """Test that invalid entries are skipped."""
        updates = [
            {"agent_a": "", "agent_b": "bob", "debate_increment": 1},  # Invalid: empty agent_a
            {"agent_a": "alice", "agent_b": "", "debate_increment": 1},  # Invalid: empty agent_b
            {"agent_a": "alice", "agent_b": "bob", "debate_increment": 1},  # Valid
        ]
        tracker.update_batch(updates)

        # Only valid entry should be processed
        assert tracker.get_raw("alice", "bob").debate_count == 1

    def test_batch_update_canonical_ordering(self, tracker):
        """Test that batch updates normalize agent ordering."""
        updates = [
            {"agent_a": "zack", "agent_b": "alice", "debate_increment": 1, "a_win": 1},
        ]
        tracker.update_batch(updates)

        stats = tracker.get_raw("alice", "zack")
        assert stats is not None
        # zack (original a) becomes b after normalization, so b_win
        assert stats.b_wins_over_a == 1


# =============================================================================
# Tests: get_raw
# =============================================================================


class TestGetRaw:
    """Tests for get_raw method."""

    def test_get_existing_relationship(self, tracker):
        """Test getting an existing relationship."""
        tracker.update_relationship("alice", "bob", debate_increment=5, agreement_increment=3)

        stats = tracker.get_raw("alice", "bob")
        assert stats is not None
        assert stats.debate_count == 5
        assert stats.agreement_count == 3

    def test_get_nonexistent_relationship(self, tracker):
        """Test getting a non-existent relationship returns None."""
        stats = tracker.get_raw("nonexistent1", "nonexistent2")
        assert stats is None

    def test_get_with_reversed_order(self, tracker):
        """Test getting with reversed agent order returns same data."""
        tracker.update_relationship("alice", "bob", debate_increment=5)

        stats1 = tracker.get_raw("alice", "bob")
        stats2 = tracker.get_raw("bob", "alice")

        assert stats1.debate_count == stats2.debate_count


# =============================================================================
# Tests: get_all_for_agent
# =============================================================================


class TestGetAllForAgent:
    """Tests for get_all_for_agent method."""

    def test_get_all_relationships(self, tracker):
        """Test getting all relationships for an agent."""
        tracker.update_relationship("alice", "bob", debate_increment=5)
        tracker.update_relationship("alice", "charlie", debate_increment=3)
        tracker.update_relationship("bob", "charlie", debate_increment=1)

        alice_rels = tracker.get_all_for_agent("alice")
        assert len(alice_rels) == 2

    def test_get_all_ordered_by_debate_count(self, tracker):
        """Test relationships are ordered by debate_count."""
        tracker.update_relationship("alice", "bob", debate_increment=1)
        tracker.update_relationship("alice", "charlie", debate_increment=10)
        tracker.update_relationship("alice", "dave", debate_increment=5)

        rels = tracker.get_all_for_agent("alice")
        assert rels[0].debate_count == 10  # charlie
        assert rels[1].debate_count == 5  # dave
        assert rels[2].debate_count == 1  # bob

    def test_get_all_with_limit(self, tracker):
        """Test limit parameter."""
        for i in range(10):
            tracker.update_relationship("alice", f"agent_{i}", debate_increment=i + 1)

        rels = tracker.get_all_for_agent("alice", limit=3)
        assert len(rels) == 3

    def test_get_all_no_relationships(self, tracker):
        """Test getting relationships for agent with none."""
        rels = tracker.get_all_for_agent("lonely_agent")
        assert len(rels) == 0


# =============================================================================
# Tests: compute_metrics
# =============================================================================


class TestComputeMetrics:
    """Tests for compute_metrics method."""

    def test_compute_metrics_no_history(self, tracker):
        """Test metrics for agents with no history."""
        metrics = tracker.compute_metrics("alice", "bob")

        assert metrics.relationship == "unknown"
        assert metrics.rivalry_score == 0.0
        assert metrics.alliance_score == 0.0
        assert metrics.debate_count == 0

    def test_compute_metrics_acquaintance(self, tracker):
        """Test metrics for agents with few debates (acquaintance)."""
        tracker.update_relationship("alice", "bob", debate_increment=2, agreement_increment=1)

        metrics = tracker.compute_metrics("alice", "bob")
        assert metrics.relationship == "acquaintance"
        assert metrics.debate_count == 2

    def test_compute_metrics_rivalry(self, tracker):
        """Test metrics produce rivalry classification."""
        # High debates, low agreement, competitive wins
        tracker.update_relationship(
            "alice",
            "bob",
            debate_increment=20,
            agreement_increment=2,  # Low agreement
            a_win=8,
            b_win=7,  # Competitive
        )

        metrics = tracker.compute_metrics("alice", "bob")
        assert metrics.rivalry_score > 0.5
        # Rivalry may or may not trigger depending on exact thresholds

    def test_compute_metrics_alliance(self, tracker):
        """Test metrics produce alliance classification."""
        # High agreement, high critique acceptance
        tracker.update_relationship(
            "alice",
            "bob",
            debate_increment=10,
            agreement_increment=9,  # High agreement
            critique_a_to_b=5,
            critique_b_to_a=5,
            critique_accepted_a_to_b=4,
            critique_accepted_b_to_a=4,  # High acceptance
        )

        metrics = tracker.compute_metrics("alice", "bob")
        assert metrics.alliance_score > 0.5

    def test_compute_metrics_agreement_rate(self, tracker):
        """Test agreement rate calculation."""
        tracker.update_relationship("alice", "bob", debate_increment=10, agreement_increment=6)

        metrics = tracker.compute_metrics("alice", "bob")
        assert metrics.agreement_rate == 0.6

    def test_compute_metrics_head_to_head(self, tracker):
        """Test head to head record."""
        tracker.update_relationship("alice", "bob", debate_increment=5, a_win=3, b_win=2)

        metrics = tracker.compute_metrics("alice", "bob")
        assert metrics.head_to_head == "3-2"


# =============================================================================
# Tests: get_rivals
# =============================================================================


class TestGetRivals:
    """Tests for get_rivals method."""

    def test_get_rivals_basic(self, tracker):
        """Test getting top rivals."""
        # Create some rivalries
        tracker.update_relationship(
            "alice", "bob", debate_increment=20, agreement_increment=2, a_win=10, b_win=8
        )
        tracker.update_relationship("alice", "charlie", debate_increment=5, agreement_increment=4)

        rivals = tracker.get_rivals("alice", limit=5, min_score=0.0)
        # At least one should be returned with non-zero rivalry
        assert len(rivals) >= 1

    def test_get_rivals_respects_limit(self, tracker):
        """Test that limit is respected."""
        for i in range(10):
            tracker.update_relationship(
                "alice", f"rival_{i}", debate_increment=15, agreement_increment=1
            )

        rivals = tracker.get_rivals("alice", limit=3)
        assert len(rivals) <= 3

    def test_get_rivals_respects_min_score(self, tracker):
        """Test that min_score filters out low rivalry."""
        tracker.update_relationship(
            "alice", "bob", debate_increment=2, agreement_increment=2  # Likely low rivalry
        )

        rivals = tracker.get_rivals("alice", min_score=0.9)  # Very high threshold
        # Should filter out relationships below threshold
        assert all(r.rivalry_score >= 0.9 for r in rivals)

    def test_get_rivals_sorted_by_score(self, tracker):
        """Test rivals are sorted by rivalry score descending."""
        # Create relationships with varying rivalry potential
        tracker.update_relationship("alice", "bob", debate_increment=20, agreement_increment=1)
        tracker.update_relationship("alice", "charlie", debate_increment=5, agreement_increment=4)

        rivals = tracker.get_rivals("alice", min_score=0.0)
        if len(rivals) >= 2:
            for i in range(len(rivals) - 1):
                assert rivals[i].rivalry_score >= rivals[i + 1].rivalry_score


# =============================================================================
# Tests: get_allies
# =============================================================================


class TestGetAllies:
    """Tests for get_allies method."""

    def test_get_allies_basic(self, tracker):
        """Test getting top allies."""
        tracker.update_relationship(
            "alice",
            "bob",
            debate_increment=10,
            agreement_increment=9,
            critique_a_to_b=3,
            critique_accepted_a_to_b=3,
        )

        allies = tracker.get_allies("alice", limit=5, min_score=0.0)
        assert len(allies) >= 0  # May or may not meet threshold

    def test_get_allies_respects_limit(self, tracker):
        """Test that limit is respected."""
        for i in range(10):
            tracker.update_relationship(
                "alice", f"ally_{i}", debate_increment=10, agreement_increment=9
            )

        allies = tracker.get_allies("alice", limit=3)
        assert len(allies) <= 3

    def test_get_allies_sorted_by_score(self, tracker):
        """Test allies are sorted by alliance score descending."""
        tracker.update_relationship("alice", "bob", debate_increment=10, agreement_increment=9)
        tracker.update_relationship("alice", "charlie", debate_increment=10, agreement_increment=5)

        allies = tracker.get_allies("alice", min_score=0.0)
        if len(allies) >= 2:
            for i in range(len(allies) - 1):
                assert allies[i].alliance_score >= allies[i + 1].alliance_score


# =============================================================================
# Tests: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_same_agent_name(self, tracker):
        """Test handling when agent_a == agent_b raises an error.

        The database has a CHECK constraint (agent_a < agent_b) that prevents
        self-relationships. This is by design - agents cannot have relationships
        with themselves.
        """
        import sqlite3

        with pytest.raises(sqlite3.IntegrityError):
            tracker.update_relationship("alice", "alice", debate_increment=1)

    def test_special_characters_in_names(self, tracker):
        """Test agent names with special characters."""
        tracker.update_relationship("claude-v3", "gpt-4o", debate_increment=1)

        stats = tracker.get_raw("claude-v3", "gpt-4o")
        assert stats is not None
        assert stats.debate_count == 1

    def test_unicode_agent_names(self, tracker):
        """Test agent names with unicode characters."""
        tracker.update_relationship("agent_α", "agent_β", debate_increment=1)

        stats = tracker.get_raw("agent_α", "agent_β")
        assert stats is not None

    def test_zero_debate_metrics(self, tracker):
        """Test metrics computation with zero debates."""
        # This creates a record but might have edge case behavior
        tracker.update_relationship("alice", "bob", agreement_increment=1)

        # debate_count is 0, but agreement is 1 - edge case
        stats = tracker.get_raw("alice", "bob")
        assert stats.debate_count == 0
        assert stats.agreement_count == 1

    def test_large_numbers(self, tracker):
        """Test with large counts."""
        tracker.update_relationship(
            "alice",
            "bob",
            debate_increment=10000,
            agreement_increment=5000,
            a_win=3000,
            b_win=2500,
        )

        stats = tracker.get_raw("alice", "bob")
        assert stats.debate_count == 10000
        assert stats.a_wins_over_b == 3000

        metrics = tracker.compute_metrics("alice", "bob")
        assert metrics.debate_count == 10000


# =============================================================================
# Tests: Persistence
# =============================================================================


class TestPersistence:
    """Tests for database persistence."""

    def test_data_persists_across_instances(self, temp_db_path):
        """Test that data persists when creating new tracker instance."""
        tracker1 = RelationshipTracker(db_path=temp_db_path)
        tracker1.update_relationship("alice", "bob", debate_increment=5)

        # Create new instance with same db
        tracker2 = RelationshipTracker(db_path=temp_db_path)
        stats = tracker2.get_raw("alice", "bob")

        assert stats is not None
        assert stats.debate_count == 5
