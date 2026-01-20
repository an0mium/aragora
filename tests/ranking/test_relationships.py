"""Tests for agent relationship tracking."""

import pytest
from unittest.mock import MagicMock, patch

from aragora.ranking.relationships import (
    RelationshipStats,
    RelationshipMetrics,
    AgentRelationship,
    RelationshipTracker,
    MAX_RELATIONSHIP_LIMIT,
)


class TestRelationshipStats:
    """Test RelationshipStats dataclass."""

    def test_create_stats(self):
        """Test creating relationship stats."""
        stats = RelationshipStats(
            agent_a="claude",
            agent_b="gpt",
            debate_count=10,
            agreement_count=7,
            critique_count_a_to_b=5,
            critique_count_b_to_a=4,
            critique_accepted_a_to_b=3,
            critique_accepted_b_to_a=2,
            position_changes_a_after_b=2,
            position_changes_b_after_a=1,
            a_wins_over_b=4,
            b_wins_over_a=3,
        )

        assert stats.agent_a == "claude"
        assert stats.agent_b == "gpt"
        assert stats.debate_count == 10
        assert stats.agreement_count == 7
        assert stats.critique_count_a_to_b == 5
        assert stats.critique_count_b_to_a == 4
        assert stats.a_wins_over_b == 4
        assert stats.b_wins_over_a == 3

    def test_zero_stats(self):
        """Test stats with all zeros."""
        stats = RelationshipStats(
            agent_a="a",
            agent_b="b",
            debate_count=0,
            agreement_count=0,
            critique_count_a_to_b=0,
            critique_count_b_to_a=0,
            critique_accepted_a_to_b=0,
            critique_accepted_b_to_a=0,
            position_changes_a_after_b=0,
            position_changes_b_after_a=0,
            a_wins_over_b=0,
            b_wins_over_a=0,
        )
        assert stats.debate_count == 0


class TestRelationshipMetrics:
    """Test RelationshipMetrics dataclass."""

    def test_create_metrics(self):
        """Test creating relationship metrics."""
        metrics = RelationshipMetrics(
            agent_a="claude",
            agent_b="gpt",
            rivalry_score=0.75,
            alliance_score=0.25,
            relationship="rival",
            debate_count=20,
            agreement_rate=0.3,
            head_to_head="12-8",
        )

        assert metrics.agent_a == "claude"
        assert metrics.rivalry_score == 0.75
        assert metrics.alliance_score == 0.25
        assert metrics.relationship == "rival"
        assert metrics.head_to_head == "12-8"

    def test_default_values(self):
        """Test default values."""
        metrics = RelationshipMetrics(
            agent_a="a",
            agent_b="b",
            rivalry_score=0.0,
            alliance_score=0.0,
            relationship="neutral",
            debate_count=0,
        )

        assert metrics.agreement_rate == 0.0
        assert metrics.head_to_head == "0-0"


class TestAgentRelationship:
    """Test AgentRelationship dataclass with computed properties."""

    def test_create_relationship(self):
        """Test creating an agent relationship."""
        rel = AgentRelationship(
            agent_a="claude",
            agent_b="gpt",
            debate_count=20,
            agreement_count=5,
            a_wins_over_b=10,
            b_wins_over_a=8,
        )

        assert rel.agent_a == "claude"
        assert rel.agent_b == "gpt"
        assert rel.debate_count == 20

    def test_rivalry_score_few_debates(self):
        """Test rivalry score with few debates returns 0."""
        rel = AgentRelationship(
            agent_a="a",
            agent_b="b",
            debate_count=2,  # Less than 3
            agreement_count=0,
            a_wins_over_b=1,
            b_wins_over_a=1,
        )
        assert rel.rivalry_score == 0.0

    def test_rivalry_score_high(self):
        """Test high rivalry score with many debates, low agreement, competitive wins."""
        rel = AgentRelationship(
            agent_a="a",
            agent_b="b",
            debate_count=20,
            agreement_count=2,  # 10% agreement (90% disagreement)
            a_wins_over_b=10,
            b_wins_over_a=9,  # Competitive
        )

        score = rel.rivalry_score
        assert score > 0.5  # Should be high rivalry

    def test_rivalry_score_low(self):
        """Test low rivalry score with high agreement."""
        rel = AgentRelationship(
            agent_a="a",
            agent_b="b",
            debate_count=20,
            agreement_count=18,  # 90% agreement
            a_wins_over_b=15,
            b_wins_over_a=2,  # Not competitive
        )

        score = rel.rivalry_score
        assert score < 0.3  # Should be low rivalry

    def test_alliance_score_few_debates(self):
        """Test alliance score with few debates returns 0."""
        rel = AgentRelationship(
            agent_a="a",
            agent_b="b",
            debate_count=2,
            agreement_count=2,
        )
        assert rel.alliance_score == 0.0

    def test_alliance_score_high(self):
        """Test high alliance score with high agreement and critique acceptance."""
        rel = AgentRelationship(
            agent_a="a",
            agent_b="b",
            debate_count=20,
            agreement_count=18,  # 90% agreement
            critique_count_a_to_b=10,
            critique_count_b_to_a=10,
            critique_accepted_a_to_b=9,
            critique_accepted_b_to_a=9,
        )

        score = rel.alliance_score
        assert score > 0.7  # Should be high alliance

    def test_alliance_score_low(self):
        """Test low alliance score with low agreement."""
        rel = AgentRelationship(
            agent_a="a",
            agent_b="b",
            debate_count=20,
            agreement_count=2,  # 10% agreement
            critique_count_a_to_b=10,
            critique_count_b_to_a=10,
            critique_accepted_a_to_b=1,
            critique_accepted_b_to_a=1,
        )

        score = rel.alliance_score
        assert score < 0.3  # Should be low alliance

    def test_influence_a_on_b(self):
        """Test influence calculation from A to B."""
        rel = AgentRelationship(
            agent_a="a",
            agent_b="b",
            debate_count=10,
            position_changes_b_after_a=5,
        )

        assert rel.influence_a_on_b == 0.5  # 5/10

    def test_influence_b_on_a(self):
        """Test influence calculation from B to A."""
        rel = AgentRelationship(
            agent_a="a",
            agent_b="b",
            debate_count=10,
            position_changes_a_after_b=3,
        )

        assert rel.influence_b_on_a == 0.3  # 3/10

    def test_influence_zero_debates(self):
        """Test influence with zero debates."""
        rel = AgentRelationship(
            agent_a="a",
            agent_b="b",
            debate_count=0,
        )

        assert rel.influence_a_on_b == 0.0
        assert rel.influence_b_on_a == 0.0

    def test_get_influence(self):
        """Test get_influence method."""
        rel = AgentRelationship(
            agent_a="claude",
            agent_b="gpt",
            debate_count=10,
            position_changes_a_after_b=3,
            position_changes_b_after_a=5,
        )

        assert rel.get_influence("claude") == 0.5  # influence_a_on_b
        assert rel.get_influence("gpt") == 0.3  # influence_b_on_a
        assert rel.get_influence("unknown") == 0.0

    def test_from_stats(self):
        """Test creating AgentRelationship from RelationshipStats."""
        stats = RelationshipStats(
            agent_a="claude",
            agent_b="gpt",
            debate_count=10,
            agreement_count=7,
            critique_count_a_to_b=5,
            critique_count_b_to_a=4,
            critique_accepted_a_to_b=3,
            critique_accepted_b_to_a=2,
            position_changes_a_after_b=2,
            position_changes_b_after_a=1,
            a_wins_over_b=4,
            b_wins_over_a=3,
        )

        rel = AgentRelationship.from_stats(stats)

        assert rel.agent_a == "claude"
        assert rel.agent_b == "gpt"
        assert rel.debate_count == 10
        assert rel.agreement_count == 7
        assert rel.a_wins_over_b == 4


class TestRelationshipTracker:
    """Test RelationshipTracker class."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database."""
        mock = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=None)
        mock_conn.cursor.return_value = mock_cursor
        mock.connection.return_value = mock_conn
        return mock, mock_cursor

    def test_init(self):
        """Test initialization."""
        with patch("aragora.ranking.relationships.EloDatabase"):
            tracker = RelationshipTracker("/path/to/db")
            assert tracker.db_path.name == "db"

    def test_update_relationship_basic(self, mock_db):
        """Test basic relationship update."""
        mock, mock_cursor = mock_db

        with patch("aragora.ranking.relationships.EloDatabase", return_value=mock):
            tracker = RelationshipTracker("/path/to/db")
            tracker.update_relationship("claude", "gpt", debate_increment=1, a_win=1)

            mock_cursor.execute.assert_called_once()
            # Should have called with INSERT OR UPDATE
            sql = mock_cursor.execute.call_args[0][0]
            assert "INSERT INTO agent_relationships" in sql

    def test_update_relationship_canonical_ordering(self, mock_db):
        """Test that agent names are canonically ordered (a < b)."""
        mock, mock_cursor = mock_db

        with patch("aragora.ranking.relationships.EloDatabase", return_value=mock):
            tracker = RelationshipTracker("/path/to/db")
            # Pass names in reverse order
            tracker.update_relationship("zulu", "alpha", debate_increment=1, a_win=1)

            # Values should be reordered (alpha < zulu)
            args = mock_cursor.execute.call_args[0][1]
            assert args[0] == "alpha"  # agent_a
            assert args[1] == "zulu"  # agent_b
            # a_win should now be b_win (0, since we swapped)
            # The win goes to original "zulu" which is now agent_b

    def test_update_batch_empty(self, mock_db):
        """Test batch update with empty list."""
        mock, mock_cursor = mock_db

        with patch("aragora.ranking.relationships.EloDatabase", return_value=mock):
            tracker = RelationshipTracker("/path/to/db")
            tracker.update_batch([])

            # Should not execute anything
            mock_cursor.execute.assert_not_called()

    def test_update_batch(self, mock_db):
        """Test batch update with multiple updates."""
        mock, mock_cursor = mock_db

        with patch("aragora.ranking.relationships.EloDatabase", return_value=mock):
            tracker = RelationshipTracker("/path/to/db")
            tracker.update_batch(
                [
                    {"agent_a": "claude", "agent_b": "gpt", "debate_increment": 1},
                    {"agent_a": "claude", "agent_b": "gemini", "debate_increment": 1, "a_win": 1},
                ]
            )

            # Should execute twice (once per update)
            assert mock_cursor.execute.call_count == 2

    def test_update_batch_skips_invalid(self, mock_db):
        """Test batch update skips invalid entries."""
        mock, mock_cursor = mock_db

        with patch("aragora.ranking.relationships.EloDatabase", return_value=mock):
            tracker = RelationshipTracker("/path/to/db")
            tracker.update_batch(
                [
                    {"agent_a": "claude", "agent_b": "gpt", "debate_increment": 1},
                    {"agent_a": "", "agent_b": "gemini"},  # Invalid - empty agent_a
                    {"agent_a": "claude"},  # Invalid - missing agent_b
                ]
            )

            # Should only execute once (for valid entry)
            assert mock_cursor.execute.call_count == 1

    def test_get_raw_found(self, mock_db):
        """Test getting raw relationship data."""
        mock, mock_cursor = mock_db
        mock_cursor.fetchone.return_value = (10, 7, 5, 4, 3, 2, 2, 1, 4, 3)

        with patch("aragora.ranking.relationships.EloDatabase", return_value=mock):
            tracker = RelationshipTracker("/path/to/db")
            stats = tracker.get_raw("claude", "gpt")

            assert stats is not None
            assert stats.agent_a == "claude"
            assert stats.agent_b == "gpt"
            assert stats.debate_count == 10
            assert stats.agreement_count == 7

    def test_get_raw_not_found(self, mock_db):
        """Test getting non-existent relationship."""
        mock, mock_cursor = mock_db
        mock_cursor.fetchone.return_value = None

        with patch("aragora.ranking.relationships.EloDatabase", return_value=mock):
            tracker = RelationshipTracker("/path/to/db")
            stats = tracker.get_raw("claude", "gpt")

            assert stats is None

    def test_get_raw_canonical_ordering(self, mock_db):
        """Test get_raw uses canonical ordering."""
        mock, mock_cursor = mock_db
        mock_cursor.fetchone.return_value = (10, 7, 5, 4, 3, 2, 2, 1, 4, 3)

        with patch("aragora.ranking.relationships.EloDatabase", return_value=mock):
            tracker = RelationshipTracker("/path/to/db")
            # Pass in reverse order
            tracker.get_raw("zulu", "alpha")

            # Query should use canonical ordering
            args = mock_cursor.execute.call_args[0][1]
            assert args[0] == "alpha"
            assert args[1] == "zulu"

    def test_get_all_for_agent(self, mock_db):
        """Test getting all relationships for an agent."""
        mock, mock_cursor = mock_db
        mock_cursor.fetchall.return_value = [
            ("claude", "gemini", 10, 7, 5, 4, 3, 2, 2, 1, 4, 3),
            ("claude", "gpt", 5, 3, 2, 2, 1, 1, 1, 0, 2, 2),
        ]

        with patch("aragora.ranking.relationships.EloDatabase", return_value=mock):
            tracker = RelationshipTracker("/path/to/db")
            relationships = tracker.get_all_for_agent("claude")

            assert len(relationships) == 2
            assert relationships[0].debate_count == 10
            assert relationships[1].debate_count == 5

    def test_get_all_for_agent_respects_limit(self, mock_db):
        """Test that limit is enforced."""
        mock, mock_cursor = mock_db
        mock_cursor.fetchall.return_value = []

        with patch("aragora.ranking.relationships.EloDatabase", return_value=mock):
            tracker = RelationshipTracker("/path/to/db")
            # Request more than MAX_RELATIONSHIP_LIMIT
            tracker.get_all_for_agent("claude", limit=MAX_RELATIONSHIP_LIMIT + 100)

            # Should cap at MAX_RELATIONSHIP_LIMIT
            args = mock_cursor.execute.call_args[0][1]
            assert args[2] == MAX_RELATIONSHIP_LIMIT

    def test_compute_metrics_no_history(self, mock_db):
        """Test computing metrics with no history."""
        mock, mock_cursor = mock_db
        mock_cursor.fetchone.return_value = None

        with patch("aragora.ranking.relationships.EloDatabase", return_value=mock):
            tracker = RelationshipTracker("/path/to/db")
            metrics = tracker.compute_metrics("claude", "gpt")

            assert metrics.relationship == "unknown"
            assert metrics.debate_count == 0
            assert metrics.rivalry_score == 0.0
            assert metrics.alliance_score == 0.0

    def test_compute_metrics_rival(self, mock_db):
        """Test computing metrics for rivals."""
        mock, mock_cursor = mock_db
        # High debates, low agreement, competitive wins = rival
        mock_cursor.fetchone.return_value = (
            25,  # debate_count
            3,  # agreement_count (12% = low)
            10,  # critique_count_a_to_b
            10,  # critique_count_b_to_a
            2,  # critique_accepted_a_to_b
            2,  # critique_accepted_b_to_a
            1,  # position_changes_a_after_b
            1,  # position_changes_b_after_a
            12,  # a_wins_over_b
            11,  # b_wins_over_a (competitive)
        )

        with patch("aragora.ranking.relationships.EloDatabase", return_value=mock):
            tracker = RelationshipTracker("/path/to/db")
            metrics = tracker.compute_metrics("claude", "gpt")

            assert metrics.rivalry_score > 0.5
            assert metrics.relationship == "rival"

    def test_compute_metrics_ally(self, mock_db):
        """Test computing metrics for allies."""
        mock, mock_cursor = mock_db
        # High agreement, high critique acceptance = ally
        mock_cursor.fetchone.return_value = (
            25,  # debate_count
            22,  # agreement_count (88% = high)
            10,  # critique_count_a_to_b
            10,  # critique_count_b_to_a
            9,  # critique_accepted_a_to_b
            9,  # critique_accepted_b_to_a
            5,  # position_changes_a_after_b
            5,  # position_changes_b_after_a
            15,  # a_wins_over_b
            3,  # b_wins_over_a (not competitive)
        )

        with patch("aragora.ranking.relationships.EloDatabase", return_value=mock):
            tracker = RelationshipTracker("/path/to/db")
            metrics = tracker.compute_metrics("claude", "gpt")

            assert metrics.alliance_score > 0.5
            assert metrics.relationship == "ally"

    def test_compute_metrics_acquaintance(self, mock_db):
        """Test computing metrics for acquaintances (few debates)."""
        mock, mock_cursor = mock_db
        mock_cursor.fetchone.return_value = (2, 1, 1, 1, 0, 0, 0, 0, 1, 1)  # debate_count (< 3)

        with patch("aragora.ranking.relationships.EloDatabase", return_value=mock):
            tracker = RelationshipTracker("/path/to/db")
            metrics = tracker.compute_metrics("claude", "gpt")

            assert metrics.relationship == "acquaintance"

    def test_compute_metrics_head_to_head(self, mock_db):
        """Test head-to-head in metrics."""
        mock, mock_cursor = mock_db
        mock_cursor.fetchone.return_value = (10, 5, 3, 3, 2, 2, 1, 1, 7, 3)

        with patch("aragora.ranking.relationships.EloDatabase", return_value=mock):
            tracker = RelationshipTracker("/path/to/db")
            metrics = tracker.compute_metrics("claude", "gpt")

            assert metrics.head_to_head == "7-3"

    def test_get_rivals(self, mock_db):
        """Test getting rivals."""
        mock, mock_cursor = mock_db
        # Return relationships with varying rivalry potential
        mock_cursor.fetchall.return_value = [
            # High rivalry potential
            ("claude", "rival1", 25, 3, 10, 10, 2, 2, 1, 1, 12, 11),
            # Lower rivalry
            ("claude", "neutral", 10, 7, 5, 5, 3, 3, 2, 2, 6, 3),
        ]

        with patch("aragora.ranking.relationships.EloDatabase", return_value=mock):
            tracker = RelationshipTracker("/path/to/db")
            rivals = tracker.get_rivals("claude", limit=5, min_score=0.3)

            # Should return at least one rival with score > 0.3
            assert len(rivals) >= 1
            # Should be sorted by rivalry_score descending
            if len(rivals) > 1:
                assert rivals[0].rivalry_score >= rivals[1].rivalry_score

    def test_get_allies(self, mock_db):
        """Test getting allies."""
        mock, mock_cursor = mock_db
        # Return relationships with varying alliance potential
        mock_cursor.fetchall.return_value = [
            # High alliance potential
            ("ally1", "claude", 25, 22, 10, 10, 9, 9, 5, 5, 15, 3),
            # Lower alliance
            ("claude", "neutral", 10, 4, 5, 5, 2, 2, 2, 2, 5, 4),
        ]

        with patch("aragora.ranking.relationships.EloDatabase", return_value=mock):
            tracker = RelationshipTracker("/path/to/db")
            allies = tracker.get_allies("claude", limit=5, min_score=0.3)

            # Should return at least one ally with score > 0.3
            assert len(allies) >= 1
            # Should be sorted by alliance_score descending
            if len(allies) > 1:
                assert allies[0].alliance_score >= allies[1].alliance_score

    def test_get_rivals_respects_min_score(self, mock_db):
        """Test get_rivals filters by min_score."""
        mock, mock_cursor = mock_db
        # Return low rivalry relationships
        mock_cursor.fetchall.return_value = [
            ("claude", "low", 5, 4, 2, 2, 1, 1, 1, 1, 3, 1),
        ]

        with patch("aragora.ranking.relationships.EloDatabase", return_value=mock):
            tracker = RelationshipTracker("/path/to/db")
            rivals = tracker.get_rivals("claude", min_score=0.8)

            # High min_score should filter out low rivalry
            assert len(rivals) == 0


class TestRelationshipTrackerMetricComputation:
    """Test metric computation edge cases."""

    @pytest.fixture
    def tracker(self):
        """Create tracker with mocked database."""
        with patch("aragora.ranking.relationships.EloDatabase"):
            return RelationshipTracker("/path/to/db")

    def test_compute_metrics_zero_debates(self, tracker):
        """Test metrics with zero debates."""
        stats = RelationshipStats(
            agent_a="a",
            agent_b="b",
            debate_count=0,
            agreement_count=0,
            critique_count_a_to_b=0,
            critique_count_b_to_a=0,
            critique_accepted_a_to_b=0,
            critique_accepted_b_to_a=0,
            position_changes_a_after_b=0,
            position_changes_b_after_a=0,
            a_wins_over_b=0,
            b_wins_over_a=0,
        )

        metrics = tracker._compute_metrics_from_stats("a", "b", stats)

        assert metrics.relationship == "no_history"
        assert metrics.debate_count == 0

    def test_compute_metrics_zero_wins(self, tracker):
        """Test metrics with debates but no wins."""
        stats = RelationshipStats(
            agent_a="a",
            agent_b="b",
            debate_count=10,
            agreement_count=5,
            critique_count_a_to_b=5,
            critique_count_b_to_a=5,
            critique_accepted_a_to_b=3,
            critique_accepted_b_to_a=3,
            position_changes_a_after_b=2,
            position_changes_b_after_a=2,
            a_wins_over_b=0,
            b_wins_over_a=0,
        )

        metrics = tracker._compute_metrics_from_stats("a", "b", stats)

        # Should still compute without division by zero
        assert metrics.debate_count == 10
        assert metrics.agreement_rate == 0.5

    def test_compute_metrics_zero_critiques(self, tracker):
        """Test metrics with no critiques."""
        stats = RelationshipStats(
            agent_a="a",
            agent_b="b",
            debate_count=10,
            agreement_count=5,
            critique_count_a_to_b=0,
            critique_count_b_to_a=0,
            critique_accepted_a_to_b=0,
            critique_accepted_b_to_a=0,
            position_changes_a_after_b=2,
            position_changes_b_after_a=2,
            a_wins_over_b=5,
            b_wins_over_a=4,
        )

        metrics = tracker._compute_metrics_from_stats("a", "b", stats)

        # Should use default critique acceptance rate
        assert metrics.debate_count == 10
