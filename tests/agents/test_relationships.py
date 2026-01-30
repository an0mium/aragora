"""
Tests for agent relationship tracking module.

This module re-exports functionality from aragora.ranking.relationships and adds
convenience methods for agent-specific use cases. Tests cover:
- RelationshipTracker initialization and database setup
- _canonical_pair ordering
- update_from_debate with various debate scenarios
- get_relationship and AgentRelationship properties
- get_all_relationships retrieval
- get_influence_network analysis
- Error handling and edge cases
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from aragora.agents.relationships import (
    AgentRelationship,
    RelationshipTracker,
)
from aragora.ranking.relationships import RelationshipStats


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_db():
    """Create a temporary database file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        yield Path(f.name)


@pytest.fixture
def tracker(temp_db):
    """Create a RelationshipTracker instance with a temp database."""
    return RelationshipTracker(temp_db)


@pytest.fixture
def populated_tracker(tracker):
    """Create a tracker with pre-populated relationship data."""
    # Create several debates worth of relationships
    # claude vs gemini: 10 debates, 6 agreements, claude wins 4, gemini wins 3
    tracker.update_from_debate(
        debate_id="debate1",
        participants=["claude", "gemini"],
        winner="claude",
        votes={"claude": "yes", "gemini": "yes"},
        critiques=[
            {"agent": "claude", "target": "gemini"},
            {"agent": "gemini", "target": "claude"},
        ],
    )
    for i in range(9):
        winner = "claude" if i % 3 == 0 else ("gemini" if i % 3 == 1 else None)
        agree = i % 2 == 0
        tracker.update_from_debate(
            debate_id=f"debate_{i + 2}",
            participants=["claude", "gemini"],
            winner=winner,
            votes={"claude": "yes", "gemini": "yes" if agree else "no"},
            critiques=[{"agent": "claude", "target": "gemini"}],
        )

    # claude vs gpt4: 5 debates, high agreement (allies)
    for i in range(5):
        tracker.update_from_debate(
            debate_id=f"ally_debate_{i}",
            participants=["claude", "gpt4"],
            winner="claude" if i == 0 else None,
            votes={"claude": "yes", "gpt4": "yes"},  # Always agree
            critiques=[],
        )

    return tracker


# ============================================================================
# Initialization Tests
# ============================================================================


class TestRelationshipTrackerInit:
    """Tests for RelationshipTracker initialization."""

    def test_init_with_path(self, temp_db):
        """Test initializing tracker with explicit path."""
        tracker = RelationshipTracker(temp_db)
        assert tracker.db_path == temp_db

    def test_init_creates_tables(self, temp_db):
        """Test that initialization creates required tables."""
        tracker = RelationshipTracker(temp_db)

        # Verify table exists by querying it
        with tracker._db.connection() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='agent_relationships'"
            )
            result = cursor.fetchone()
        assert result is not None
        assert result[0] == "agent_relationships"

    def test_init_creates_indexes(self, temp_db):
        """Test that initialization creates required indexes."""
        tracker = RelationshipTracker(temp_db)

        with tracker._db.connection() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_relationships_%'"
            )
            indexes = cursor.fetchall()

        index_names = [idx[0] for idx in indexes]
        assert "idx_relationships_a" in index_names
        assert "idx_relationships_b" in index_names

    @patch("aragora.agents.relationships.get_db_path")
    def test_init_with_default_path(self, mock_get_db_path, temp_db):
        """Test initializing tracker with default path."""
        mock_get_db_path.return_value = temp_db
        tracker = RelationshipTracker()
        mock_get_db_path.assert_called_once()


# ============================================================================
# Canonical Pair Tests
# ============================================================================


class TestCanonicalPair:
    """Tests for _canonical_pair method."""

    def test_canonical_pair_already_ordered(self, tracker):
        """Test canonical pair when agents are already in order."""
        result = tracker._canonical_pair("alice", "bob")
        assert result == ("alice", "bob")

    def test_canonical_pair_needs_swap(self, tracker):
        """Test canonical pair when agents need swapping."""
        result = tracker._canonical_pair("charlie", "bob")
        assert result == ("bob", "charlie")

    def test_canonical_pair_same_agent(self, tracker):
        """Test canonical pair with same agent name."""
        result = tracker._canonical_pair("alice", "alice")
        assert result == ("alice", "alice")

    def test_canonical_pair_numeric_names(self, tracker):
        """Test canonical pair with numeric-like names."""
        result = tracker._canonical_pair("agent2", "agent1")
        assert result == ("agent1", "agent2")


# ============================================================================
# Update From Debate Tests
# ============================================================================


class TestUpdateFromDebate:
    """Tests for update_from_debate method."""

    def test_update_creates_relationship(self, tracker):
        """Test that update_from_debate creates new relationships."""
        tracker.update_from_debate(
            debate_id="test1",
            participants=["alice", "bob"],
            winner="alice",
            votes={"alice": "yes", "bob": "no"},
            critiques=[],
        )

        rel = tracker.get_relationship("alice", "bob")
        assert rel.debate_count == 1

    def test_update_increments_debate_count(self, tracker):
        """Test that multiple debates increment the count."""
        for i in range(3):
            tracker.update_from_debate(
                debate_id=f"test{i}",
                participants=["alice", "bob"],
                winner=None,
                votes={},
                critiques=[],
            )

        rel = tracker.get_relationship("alice", "bob")
        assert rel.debate_count == 3

    def test_update_tracks_agreement(self, tracker):
        """Test that agreement is tracked when agents vote the same."""
        tracker.update_from_debate(
            debate_id="test1",
            participants=["alice", "bob"],
            winner=None,
            votes={"alice": "yes", "bob": "yes"},
            critiques=[],
        )

        rel = tracker.get_relationship("alice", "bob")
        assert rel.agreement_count == 1

    def test_update_no_agreement_different_votes(self, tracker):
        """Test that disagreement is not counted as agreement."""
        tracker.update_from_debate(
            debate_id="test1",
            participants=["alice", "bob"],
            winner=None,
            votes={"alice": "yes", "bob": "no"},
            critiques=[],
        )

        rel = tracker.get_relationship("alice", "bob")
        assert rel.agreement_count == 0

    def test_update_tracks_winner_a(self, tracker):
        """Test tracking when first agent (alphabetically) wins."""
        tracker.update_from_debate(
            debate_id="test1",
            participants=["alice", "bob"],
            winner="alice",
            votes={},
            critiques=[],
        )

        rel = tracker.get_relationship("alice", "bob")
        assert rel.a_wins_over_b == 1
        assert rel.b_wins_over_a == 0

    def test_update_tracks_winner_b(self, tracker):
        """Test tracking when second agent (alphabetically) wins."""
        tracker.update_from_debate(
            debate_id="test1",
            participants=["alice", "bob"],
            winner="bob",
            votes={},
            critiques=[],
        )

        rel = tracker.get_relationship("alice", "bob")
        assert rel.a_wins_over_b == 0
        assert rel.b_wins_over_a == 1

    def test_update_tracks_critiques(self, tracker):
        """Test that critiques are tracked correctly."""
        tracker.update_from_debate(
            debate_id="test1",
            participants=["alice", "bob"],
            winner=None,
            votes={},
            critiques=[
                {"agent": "alice", "target": "bob"},
                {"agent": "alice", "target": "bob"},
                {"agent": "bob", "target": "alice"},
            ],
        )

        rel = tracker.get_relationship("alice", "bob")
        assert rel.critique_count_a_to_b == 2
        assert rel.critique_count_b_to_a == 1

    def test_update_critiques_alternate_field_names(self, tracker):
        """Test critiques with critic/target_agent field names."""
        tracker.update_from_debate(
            debate_id="test1",
            participants=["alice", "bob"],
            winner=None,
            votes={},
            critiques=[
                {"critic": "alice", "target_agent": "bob"},
            ],
        )

        rel = tracker.get_relationship("alice", "bob")
        assert rel.critique_count_a_to_b == 1

    def test_update_ignores_self_critiques(self, tracker):
        """Test that self-critiques are ignored."""
        tracker.update_from_debate(
            debate_id="test1",
            participants=["alice", "bob"],
            winner=None,
            votes={},
            critiques=[
                {"agent": "alice", "target": "alice"},  # Self-critique
            ],
        )

        rel = tracker.get_relationship("alice", "bob")
        assert rel.critique_count_a_to_b == 0
        assert rel.critique_count_b_to_a == 0

    def test_update_handles_multiple_participants(self, tracker):
        """Test update with more than two participants."""
        tracker.update_from_debate(
            debate_id="test1",
            participants=["alice", "bob", "charlie"],
            winner="bob",
            votes={"alice": "yes", "bob": "yes", "charlie": "no"},
            critiques=[],
        )

        # Should create three relationships
        rel_ab = tracker.get_relationship("alice", "bob")
        rel_ac = tracker.get_relationship("alice", "charlie")
        rel_bc = tracker.get_relationship("bob", "charlie")

        assert rel_ab.debate_count == 1
        assert rel_ac.debate_count == 1
        assert rel_bc.debate_count == 1

        # Check agreement (alice and bob both voted yes)
        assert rel_ab.agreement_count == 1
        assert rel_ac.agreement_count == 0
        assert rel_bc.agreement_count == 0

    def test_update_handles_empty_participants(self, tracker):
        """Test update with empty participants list."""
        tracker.update_from_debate(
            debate_id="test1",
            participants=[],
            winner=None,
            votes={},
            critiques=[],
        )
        # Should not raise, just do nothing

    def test_update_handles_single_participant(self, tracker):
        """Test update with single participant."""
        tracker.update_from_debate(
            debate_id="test1",
            participants=["alice"],
            winner=None,
            votes={},
            critiques=[],
        )
        # Should not raise, just do nothing


# ============================================================================
# Get Relationship Tests
# ============================================================================


class TestGetRelationship:
    """Tests for get_relationship method."""

    def test_get_nonexistent_relationship(self, tracker):
        """Test getting a relationship that doesn't exist."""
        rel = tracker.get_relationship("alice", "bob")

        assert rel.agent_a == "alice"
        assert rel.agent_b == "bob"
        assert rel.debate_count == 0

    def test_get_relationship_canonical_order(self, tracker):
        """Test that get_relationship uses canonical ordering."""
        tracker.update_from_debate(
            debate_id="test1",
            participants=["alice", "bob"],
            winner="alice",
            votes={},
            critiques=[],
        )

        # Should return same relationship regardless of order
        rel1 = tracker.get_relationship("alice", "bob")
        rel2 = tracker.get_relationship("bob", "alice")

        assert rel1.debate_count == rel2.debate_count == 1

    def test_get_relationship_returns_agent_relationship(self, tracker):
        """Test that get_relationship returns AgentRelationship type."""
        tracker.update_from_debate(
            debate_id="test1",
            participants=["alice", "bob"],
            winner="alice",
            votes={},
            critiques=[],
        )

        rel = tracker.get_relationship("alice", "bob")
        assert isinstance(rel, AgentRelationship)


# ============================================================================
# AgentRelationship Property Tests
# ============================================================================


class TestAgentRelationshipProperties:
    """Tests for AgentRelationship computed properties."""

    def test_rivalry_score_insufficient_debates(self):
        """Test rivalry score returns 0 with fewer than 3 debates."""
        rel = AgentRelationship(
            agent_a="alice",
            agent_b="bob",
            debate_count=2,
            agreement_count=0,
        )
        assert rel.rivalry_score == 0.0

    def test_rivalry_score_calculation(self):
        """Test rivalry score calculation with sufficient data."""
        rel = AgentRelationship(
            agent_a="alice",
            agent_b="bob",
            debate_count=20,
            agreement_count=5,  # 25% agreement, 75% disagreement
            a_wins_over_b=5,
            b_wins_over_a=5,  # Competitive (equal wins)
        )
        # disagreement_rate = 0.75
        # competitiveness = 1.0 (equal wins)
        # frequency_factor = 1.0 (20 debates)
        # rivalry = 0.75 * 1.0 * 1.0 = 0.75
        assert rel.rivalry_score == 0.75

    def test_alliance_score_insufficient_debates(self):
        """Test alliance score returns 0 with fewer than 3 debates."""
        rel = AgentRelationship(
            agent_a="alice",
            agent_b="bob",
            debate_count=2,
            agreement_count=2,
        )
        assert rel.alliance_score == 0.0

    def test_alliance_score_calculation(self):
        """Test alliance score calculation with sufficient data."""
        rel = AgentRelationship(
            agent_a="alice",
            agent_b="bob",
            debate_count=10,
            agreement_count=8,  # 80% agreement
            critique_count_a_to_b=5,
            critique_count_b_to_a=5,
            critique_accepted_a_to_b=3,
            critique_accepted_b_to_a=4,  # 70% acceptance
        )
        # agreement_rate = 0.8
        # acceptance_rate = 0.7
        # alliance = 0.8 * 0.6 + 0.7 * 0.4 = 0.48 + 0.28 = 0.76
        assert abs(rel.alliance_score - 0.76) < 0.01

    def test_influence_a_on_b_no_debates(self):
        """Test influence returns 0 with no debates."""
        rel = AgentRelationship(
            agent_a="alice",
            agent_b="bob",
            debate_count=0,
        )
        assert rel.influence_a_on_b == 0.0

    def test_influence_a_on_b_calculation(self):
        """Test influence A on B calculation."""
        rel = AgentRelationship(
            agent_a="alice",
            agent_b="bob",
            debate_count=10,
            position_changes_b_after_a=4,
        )
        assert rel.influence_a_on_b == 0.4

    def test_influence_b_on_a_calculation(self):
        """Test influence B on A calculation."""
        rel = AgentRelationship(
            agent_a="alice",
            agent_b="bob",
            debate_count=10,
            position_changes_a_after_b=3,
        )
        assert rel.influence_b_on_a == 0.3

    def test_get_influence_from_agent_a(self):
        """Test get_influence when querying from agent A."""
        rel = AgentRelationship(
            agent_a="alice",
            agent_b="bob",
            debate_count=10,
            position_changes_b_after_a=4,
        )
        assert rel.get_influence("alice") == 0.4

    def test_get_influence_from_agent_b(self):
        """Test get_influence when querying from agent B."""
        rel = AgentRelationship(
            agent_a="alice",
            agent_b="bob",
            debate_count=10,
            position_changes_a_after_b=3,
        )
        assert rel.get_influence("bob") == 0.3

    def test_get_influence_unknown_agent(self):
        """Test get_influence with unknown agent returns 0."""
        rel = AgentRelationship(
            agent_a="alice",
            agent_b="bob",
            debate_count=10,
        )
        assert rel.get_influence("charlie") == 0.0


# ============================================================================
# Get All Relationships Tests
# ============================================================================


class TestGetAllRelationships:
    """Tests for get_all_relationships method."""

    def test_get_all_relationships_empty(self, tracker):
        """Test getting relationships when none exist."""
        rels = tracker.get_all_relationships("alice")
        assert rels == []

    def test_get_all_relationships(self, populated_tracker):
        """Test getting all relationships for an agent."""
        rels = populated_tracker.get_all_relationships("claude")

        assert len(rels) >= 2
        agent_names = []
        for rel in rels:
            other = rel.agent_b if rel.agent_a == "claude" else rel.agent_a
            agent_names.append(other)

        assert "gemini" in agent_names
        assert "gpt4" in agent_names

    def test_get_all_relationships_returns_agent_relationship(self, populated_tracker):
        """Test that get_all_relationships returns AgentRelationship objects."""
        rels = populated_tracker.get_all_relationships("claude")

        for rel in rels:
            assert isinstance(rel, AgentRelationship)


# ============================================================================
# Get Influence Network Tests
# ============================================================================


class TestGetInfluenceNetwork:
    """Tests for get_influence_network method."""

    def test_get_influence_network_empty(self, tracker):
        """Test influence network when no relationships exist."""
        network = tracker.get_influence_network("alice")

        assert network["influences"] == []
        assert network["influenced_by"] == []

    def test_get_influence_network_structure(self, tracker):
        """Test influence network returns correct structure."""
        # Create relationship with position changes
        with tracker._db.connection() as conn:
            conn.execute(
                """
                INSERT INTO agent_relationships (
                    agent_a, agent_b, debate_count,
                    position_changes_a_after_b, position_changes_b_after_a
                ) VALUES (?, ?, ?, ?, ?)
                """,
                ("alice", "bob", 10, 3, 5),
            )
            conn.commit()

        network = tracker.get_influence_network("alice")

        assert "influences" in network
        assert "influenced_by" in network
        assert isinstance(network["influences"], list)
        assert isinstance(network["influenced_by"], list)

    def test_get_influence_network_influences(self, tracker):
        """Test that influences are correctly identified."""
        # Alice influences Bob (Bob changes position after Alice)
        with tracker._db.connection() as conn:
            conn.execute(
                """
                INSERT INTO agent_relationships (
                    agent_a, agent_b, debate_count,
                    position_changes_a_after_b, position_changes_b_after_a
                ) VALUES (?, ?, ?, ?, ?)
                """,
                ("alice", "bob", 10, 0, 5),  # Bob changes after Alice
            )
            conn.commit()

        network = tracker.get_influence_network("alice")

        # alice is agent_a, position_changes_b_after_a = 5 means alice influences bob
        assert len(network["influences"]) == 1
        assert network["influences"][0][0] == "bob"
        assert network["influences"][0][1] == 0.5

    def test_get_influence_network_influenced_by(self, tracker):
        """Test that influenced_by is correctly identified."""
        # Alice is influenced by Bob (Alice changes position after Bob)
        with tracker._db.connection() as conn:
            conn.execute(
                """
                INSERT INTO agent_relationships (
                    agent_a, agent_b, debate_count,
                    position_changes_a_after_b, position_changes_b_after_a
                ) VALUES (?, ?, ?, ?, ?)
                """,
                ("alice", "bob", 10, 4, 0),  # Alice changes after Bob
            )
            conn.commit()

        network = tracker.get_influence_network("alice")

        # alice is agent_a, position_changes_a_after_b = 4 means bob influences alice
        assert len(network["influenced_by"]) == 1
        assert network["influenced_by"][0][0] == "bob"
        assert network["influenced_by"][0][1] == 0.4

    def test_get_influence_network_sorted(self, tracker):
        """Test that influence lists are sorted by score descending."""
        # Create multiple relationships with different influence scores
        with tracker._db.connection() as conn:
            conn.execute(
                """
                INSERT INTO agent_relationships (
                    agent_a, agent_b, debate_count,
                    position_changes_a_after_b, position_changes_b_after_a
                ) VALUES (?, ?, ?, ?, ?)
                """,
                ("alice", "bob", 10, 0, 3),  # alice influences bob: 0.3
            )
            conn.execute(
                """
                INSERT INTO agent_relationships (
                    agent_a, agent_b, debate_count,
                    position_changes_a_after_b, position_changes_b_after_a
                ) VALUES (?, ?, ?, ?, ?)
                """,
                ("alice", "charlie", 10, 0, 7),  # alice influences charlie: 0.7
            )
            conn.commit()

        network = tracker.get_influence_network("alice")

        # Should be sorted by influence score descending
        assert len(network["influences"]) == 2
        assert network["influences"][0][0] == "charlie"  # 0.7
        assert network["influences"][1][0] == "bob"  # 0.3


# ============================================================================
# AgentRelationship.from_stats Tests
# ============================================================================


class TestAgentRelationshipFromStats:
    """Tests for AgentRelationship.from_stats class method."""

    def test_from_stats_basic(self):
        """Test creating AgentRelationship from RelationshipStats."""
        stats = RelationshipStats(
            agent_a="alice",
            agent_b="bob",
            debate_count=10,
            agreement_count=7,
            critique_count_a_to_b=5,
            critique_count_b_to_a=3,
            critique_accepted_a_to_b=2,
            critique_accepted_b_to_a=1,
            position_changes_a_after_b=4,
            position_changes_b_after_a=2,
            a_wins_over_b=6,
            b_wins_over_a=4,
        )

        rel = AgentRelationship.from_stats(stats)

        assert rel.agent_a == "alice"
        assert rel.agent_b == "bob"
        assert rel.debate_count == 10
        assert rel.agreement_count == 7
        assert rel.critique_count_a_to_b == 5
        assert rel.critique_count_b_to_a == 3
        assert rel.a_wins_over_b == 6
        assert rel.b_wins_over_a == 4


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


class TestEdgeCasesAndErrors:
    """Tests for edge cases and error handling."""

    def test_empty_votes_dict(self, tracker):
        """Test handling of empty votes dictionary."""
        tracker.update_from_debate(
            debate_id="test1",
            participants=["alice", "bob"],
            winner=None,
            votes={},
            critiques=[],
        )

        rel = tracker.get_relationship("alice", "bob")
        assert rel.debate_count == 1
        assert rel.agreement_count == 0

    def test_missing_vote_keys(self, tracker):
        """Test handling when vote keys don't match participants."""
        tracker.update_from_debate(
            debate_id="test1",
            participants=["alice", "bob"],
            winner=None,
            votes={"charlie": "yes"},  # Wrong agent
            critiques=[],
        )

        rel = tracker.get_relationship("alice", "bob")
        assert rel.agreement_count == 0

    def test_empty_critiques_list(self, tracker):
        """Test handling of empty critiques list."""
        tracker.update_from_debate(
            debate_id="test1",
            participants=["alice", "bob"],
            winner="alice",
            votes={},
            critiques=[],
        )

        rel = tracker.get_relationship("alice", "bob")
        assert rel.critique_count_a_to_b == 0
        assert rel.critique_count_b_to_a == 0

    def test_critique_missing_fields(self, tracker):
        """Test critiques with missing required fields."""
        tracker.update_from_debate(
            debate_id="test1",
            participants=["alice", "bob"],
            winner=None,
            votes={},
            critiques=[
                {"agent": "alice"},  # Missing target
                {"target": "bob"},  # Missing agent
                {},  # Missing both
            ],
        )

        rel = tracker.get_relationship("alice", "bob")
        assert rel.critique_count_a_to_b == 0

    def test_special_characters_in_names(self, tracker):
        """Test handling of special characters in agent names."""
        tracker.update_from_debate(
            debate_id="test1",
            participants=["agent-1", "agent_2"],
            winner="agent-1",
            votes={"agent-1": "yes", "agent_2": "no"},
            critiques=[],
        )

        rel = tracker.get_relationship("agent-1", "agent_2")
        assert rel.debate_count == 1

    def test_unicode_agent_names(self, tracker):
        """Test handling of unicode characters in agent names."""
        tracker.update_from_debate(
            debate_id="test1",
            participants=["claude", "gpt4"],
            winner="claude",
            votes={},
            critiques=[],
        )

        rel = tracker.get_relationship("claude", "gpt4")
        assert rel.debate_count == 1

    def test_very_long_agent_names(self, tracker):
        """Test handling of very long agent names."""
        long_name_a = "a" * 100
        long_name_b = "b" * 100

        tracker.update_from_debate(
            debate_id="test1",
            participants=[long_name_a, long_name_b],
            winner=long_name_a,
            votes={},
            critiques=[],
        )

        rel = tracker.get_relationship(long_name_a, long_name_b)
        assert rel.debate_count == 1

    def test_none_winner(self, tracker):
        """Test handling when winner is None (draw/no winner)."""
        tracker.update_from_debate(
            debate_id="test1",
            participants=["alice", "bob"],
            winner=None,
            votes={},
            critiques=[],
        )

        rel = tracker.get_relationship("alice", "bob")
        assert rel.a_wins_over_b == 0
        assert rel.b_wins_over_a == 0

    def test_winner_not_in_participants(self, tracker):
        """Test handling when winner is not in participants list."""
        tracker.update_from_debate(
            debate_id="test1",
            participants=["alice", "bob"],
            winner="charlie",  # Not a participant
            votes={},
            critiques=[],
        )

        rel = tracker.get_relationship("alice", "bob")
        # Should not crash, winner just won't be tracked
        assert rel.a_wins_over_b == 0
        assert rel.b_wins_over_a == 0


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_full_debate_cycle(self, tracker):
        """Test a complete debate cycle with all features."""
        # First debate
        tracker.update_from_debate(
            debate_id="debate1",
            participants=["claude", "gemini", "gpt4"],
            winner="claude",
            votes={"claude": "yes", "gemini": "no", "gpt4": "yes"},
            critiques=[
                {"agent": "gemini", "target": "claude"},
                {"agent": "gpt4", "target": "claude"},
                {"agent": "claude", "target": "gemini"},
            ],
        )

        # Second debate
        tracker.update_from_debate(
            debate_id="debate2",
            participants=["claude", "gemini", "gpt4"],
            winner="gemini",
            votes={"claude": "yes", "gemini": "yes", "gpt4": "yes"},
            critiques=[
                {"agent": "claude", "target": "gemini"},
            ],
        )

        # Check claude-gemini relationship
        rel_cg = tracker.get_relationship("claude", "gemini")
        assert rel_cg.debate_count == 2
        assert rel_cg.agreement_count == 1  # Only agreed in debate2
        assert rel_cg.a_wins_over_b == 1  # claude won debate1
        assert rel_cg.b_wins_over_a == 1  # gemini won debate2

        # Check claude-gpt4 relationship
        rel_cg4 = tracker.get_relationship("claude", "gpt4")
        assert rel_cg4.debate_count == 2
        assert rel_cg4.agreement_count == 2  # Agreed in both

    def test_relationship_scores_evolution(self, tracker):
        """Test how relationship scores evolve over multiple debates."""
        # Create a rivalry through disagreement
        for i in range(10):
            tracker.update_from_debate(
                debate_id=f"rivalry_{i}",
                participants=["alice", "rival"],
                winner="alice" if i % 2 == 0 else "rival",
                votes={"alice": "yes", "rival": "no"},  # Always disagree
                critiques=[],
            )

        # Create an alliance through agreement
        for i in range(10):
            tracker.update_from_debate(
                debate_id=f"alliance_{i}",
                participants=["alice", "ally"],
                winner="alice" if i == 0 else None,
                votes={"alice": "yes", "ally": "yes"},  # Always agree
                critiques=[],
            )

        rival_rel = tracker.get_relationship("alice", "rival")
        ally_rel = tracker.get_relationship("alice", "ally")

        # Rival should have higher rivalry score
        assert rival_rel.rivalry_score > ally_rel.rivalry_score

        # Ally should have higher alliance score
        assert ally_rel.alliance_score > rival_rel.alliance_score

    def test_influence_network_after_debates(self, tracker):
        """Test influence network is correctly populated."""
        # Create relationships with position changes
        with tracker._db.connection() as conn:
            # alice influences bob and charlie
            conn.execute(
                """
                INSERT INTO agent_relationships (
                    agent_a, agent_b, debate_count,
                    position_changes_a_after_b, position_changes_b_after_a
                ) VALUES (?, ?, ?, ?, ?)
                """,
                ("alice", "bob", 10, 1, 5),
            )
            conn.execute(
                """
                INSERT INTO agent_relationships (
                    agent_a, agent_b, debate_count,
                    position_changes_a_after_b, position_changes_b_after_a
                ) VALUES (?, ?, ?, ?, ?)
                """,
                ("alice", "charlie", 10, 2, 3),
            )
            # alice is influenced by dave
            conn.execute(
                """
                INSERT INTO agent_relationships (
                    agent_a, agent_b, debate_count,
                    position_changes_a_after_b, position_changes_b_after_a
                ) VALUES (?, ?, ?, ?, ?)
                """,
                ("alice", "dave", 10, 6, 1),
            )
            conn.commit()

        network = tracker.get_influence_network("alice")

        # alice influences bob (0.5) and charlie (0.3)
        influences = {agent: score for agent, score in network["influences"]}
        assert "bob" in influences
        assert "charlie" in influences
        assert influences["bob"] == 0.5
        assert influences["charlie"] == 0.3

        # alice is influenced by dave (0.6) and bob (0.1)
        influenced_by = {agent: score for agent, score in network["influenced_by"]}
        assert "dave" in influenced_by
        assert influenced_by["dave"] == 0.6
