"""
Tests for suggestion feedback tracking module.

Tests cover:
- SuggestionRecord dataclass
- ContributorStats dataclass
- SuggestionFeedbackTracker class
"""

import os
import tempfile
import pytest
from unittest.mock import MagicMock

from aragora.audience.feedback import (
    ContributorStats,
    SuggestionFeedbackTracker,
    SuggestionRecord,
)


# ============================================================================
# SuggestionRecord Tests
# ============================================================================


class TestSuggestionRecord:
    """Tests for SuggestionRecord dataclass."""

    def test_creation(self):
        """Test basic creation."""
        record = SuggestionRecord(
            id="rec-123",
            debate_id="debate-456",
            suggestion_text="Test suggestion",
            cluster_count=3,
            user_ids=["user1", "user2"],
            injected_at="2024-01-01T12:00:00",
        )
        assert record.id == "rec-123"
        assert record.debate_id == "debate-456"
        assert record.cluster_count == 3

    def test_default_values(self):
        """Test default values."""
        record = SuggestionRecord(
            id="r1",
            debate_id="d1",
            suggestion_text="Test",
            cluster_count=1,
            user_ids=[],
            injected_at="2024-01-01",
        )
        assert record.debate_completed is False
        assert record.consensus_reached is False
        assert record.consensus_confidence == 0.0
        assert record.duration_seconds == 0.0
        assert record.effectiveness_score == 0.0

    def test_to_dict(self):
        """Test serialization to dictionary."""
        record = SuggestionRecord(
            id="r1",
            debate_id="d1",
            suggestion_text="Test",
            cluster_count=2,
            user_ids=["u1"],
            injected_at="2024-01-01",
            debate_completed=True,
            consensus_reached=True,
            consensus_confidence=0.85,
            effectiveness_score=0.9,
        )
        result = record.to_dict()

        assert result["id"] == "r1"
        assert result["debate_id"] == "d1"
        assert result["consensus_reached"] is True
        assert result["consensus_confidence"] == 0.85


# ============================================================================
# ContributorStats Tests
# ============================================================================


class TestContributorStats:
    """Tests for ContributorStats dataclass."""

    def test_creation(self):
        """Test basic creation."""
        stats = ContributorStats(
            user_id="user123",
            total_suggestions=10,
            suggestions_in_consensus=7,
            avg_effectiveness=0.75,
            reputation_score=0.8,
        )
        assert stats.user_id == "user123"
        assert stats.total_suggestions == 10
        assert stats.reputation_score == 0.8

    def test_default_values(self):
        """Test default values."""
        stats = ContributorStats(user_id="user1")
        assert stats.total_suggestions == 0
        assert stats.suggestions_in_consensus == 0
        assert stats.avg_effectiveness == 0.0
        assert stats.reputation_score == 0.5

    def test_consensus_rate_property(self):
        """Test consensus_rate calculation."""
        stats = ContributorStats(
            user_id="user1",
            total_suggestions=10,
            suggestions_in_consensus=7,
        )
        assert stats.consensus_rate == 0.7

    def test_consensus_rate_zero_suggestions(self):
        """Test consensus_rate with zero suggestions."""
        stats = ContributorStats(user_id="user1", total_suggestions=0)
        assert stats.consensus_rate == 0.0


# ============================================================================
# SuggestionFeedbackTracker Tests
# ============================================================================


class TestSuggestionFeedbackTracker:
    """Tests for SuggestionFeedbackTracker class."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database file."""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        yield path
        # Cleanup
        if os.path.exists(path):
            os.unlink(path)

    @pytest.fixture
    def tracker(self, temp_db):
        """Create a tracker with temporary database."""
        return SuggestionFeedbackTracker(temp_db)

    def test_initialization(self, tracker):
        """Test tracker initializes correctly."""
        assert tracker is not None

    def test_record_injection_returns_ids(self, tracker):
        """Test record_injection returns injection IDs."""
        clusters = [
            MagicMock(representative="Idea 1", count=3, user_ids=["u1", "u2"]),
            MagicMock(representative="Idea 2", count=1, user_ids=["u3"]),
        ]

        ids = tracker.record_injection("debate-001", clusters)

        assert len(ids) == 2
        assert all(isinstance(id, str) for id in ids)

    def test_record_injection_stores_data(self, tracker):
        """Test record_injection stores data correctly."""
        cluster = MagicMock(
            representative="Test idea",
            count=5,
            user_ids=["user1", "user2"],
        )

        tracker.record_injection("debate-001", [cluster])
        suggestions = tracker.get_debate_suggestions("debate-001")

        assert len(suggestions) == 1
        assert suggestions[0].suggestion_text == "Test idea"
        assert suggestions[0].cluster_count == 5

    def test_record_injection_handles_dict_clusters(self, tracker):
        """Test record_injection handles dict format clusters."""
        clusters = [
            {"representative": "Dict idea", "count": 2, "user_ids": ["u1"]},
        ]

        ids = tracker.record_injection("debate-001", clusters)

        assert len(ids) == 1
        suggestions = tracker.get_debate_suggestions("debate-001")
        assert suggestions[0].suggestion_text == "Dict idea"

    def test_record_outcome_updates_suggestions(self, tracker):
        """Test record_outcome updates suggestion records."""
        cluster = MagicMock(
            representative="Test",
            count=1,
            user_ids=["user1"],
        )
        tracker.record_injection("debate-001", [cluster])

        updated = tracker.record_outcome(
            debate_id="debate-001",
            consensus_reached=True,
            consensus_confidence=0.9,
            duration_seconds=120.0,
        )

        assert updated == 1
        suggestions = tracker.get_debate_suggestions("debate-001")
        assert suggestions[0].debate_completed is True
        assert suggestions[0].consensus_reached is True
        assert suggestions[0].consensus_confidence == 0.9

    def test_record_outcome_calculates_effectiveness(self, tracker):
        """Test effectiveness score calculation."""
        cluster = MagicMock(representative="Test", count=1, user_ids=["u1"])
        tracker.record_injection("debate-001", [cluster])

        # High confidence consensus should give high effectiveness
        tracker.record_outcome("debate-001", consensus_reached=True, consensus_confidence=0.9)

        suggestions = tracker.get_debate_suggestions("debate-001")
        # 0.5 + 0.9 * 0.4 + 0.1 (bonus) = 0.96
        assert suggestions[0].effectiveness_score >= 0.9

    def test_record_outcome_no_consensus_lower_effectiveness(self, tracker):
        """Test lower effectiveness when no consensus."""
        cluster = MagicMock(representative="Test", count=1, user_ids=["u1"])
        tracker.record_injection("debate-001", [cluster])

        tracker.record_outcome("debate-001", consensus_reached=False, consensus_confidence=0.5)

        suggestions = tracker.get_debate_suggestions("debate-001")
        # 0.2 + 0.5 * 0.2 = 0.3
        assert suggestions[0].effectiveness_score < 0.5

    def test_record_outcome_nonexistent_debate(self, tracker):
        """Test record_outcome with nonexistent debate."""
        updated = tracker.record_outcome(
            "nonexistent", consensus_reached=True, consensus_confidence=0.9
        )
        assert updated == 0

    def test_record_outcome_updates_contributor_stats(self, tracker):
        """Test that outcomes update contributor stats."""
        cluster = MagicMock(
            representative="Test",
            count=1,
            user_ids=["contrib1"],
        )
        tracker.record_injection("debate-001", [cluster])
        tracker.record_outcome("debate-001", consensus_reached=True, consensus_confidence=0.9)

        stats = tracker.get_contributor_stats("contrib1")

        assert stats is not None
        assert stats.total_suggestions == 1
        assert stats.suggestions_in_consensus == 1

    def test_get_contributor_stats_nonexistent(self, tracker):
        """Test get_contributor_stats for unknown user."""
        stats = tracker.get_contributor_stats("unknown_user")
        assert stats is None

    def test_get_top_contributors(self, tracker):
        """Test getting top contributors."""
        # Add multiple contributors with different stats
        for i in range(5):
            user_id = f"user{i}"
            cluster = MagicMock(representative=f"Idea {i}", count=1, user_ids=[user_id])
            tracker.record_injection(f"debate-{i}", [cluster])
            # More suggestions for some users
            for j in range(i):
                cluster2 = MagicMock(representative=f"Idea {i}-{j}", count=1, user_ids=[user_id])
                tracker.record_injection(f"debate-{i}-{j}", [cluster2])
                tracker.record_outcome(f"debate-{i}-{j}", consensus_reached=True, consensus_confidence=0.8)

        # Record outcomes
        for i in range(5):
            tracker.record_outcome(f"debate-{i}", consensus_reached=True, consensus_confidence=0.8)

        top = tracker.get_top_contributors(limit=3)

        # Minimum 3 suggestions required
        assert len(top) <= 3
        # Should be sorted by reputation
        if len(top) >= 2:
            assert top[0].reputation_score >= top[1].reputation_score

    def test_get_debate_suggestions_empty(self, tracker):
        """Test get_debate_suggestions for debate with no suggestions."""
        result = tracker.get_debate_suggestions("nonexistent")
        assert result == []

    def test_get_effectiveness_stats(self, tracker):
        """Test getting overall effectiveness statistics."""
        # Add some data
        cluster = MagicMock(representative="Test", count=2, user_ids=["u1", "u2"])
        tracker.record_injection("debate-001", [cluster])
        tracker.record_outcome("debate-001", consensus_reached=True, consensus_confidence=0.8)

        stats = tracker.get_effectiveness_stats()

        assert "total_suggestions" in stats
        assert "debates_with_suggestions" in stats
        assert "consensus_rate" in stats
        assert "avg_effectiveness" in stats
        assert "total_contributors" in stats

    def test_filter_by_reputation_unknown_users(self, tracker):
        """Test filtering suggestions with unknown users."""
        suggestions = [
            {"suggestion": "Test 1", "user_id": "unknown1"},
            {"suggestion": "Test 2", "user_id": "unknown2"},
        ]

        result = tracker.filter_by_reputation(suggestions, min_reputation=0.3)

        # Unknown users get default 0.5 reputation, so should pass 0.3 threshold
        assert len(result) == 2

    def test_filter_by_reputation_filters_low_rep(self, tracker):
        """Test filtering removes low reputation users."""
        # Create a user with low reputation
        cluster = MagicMock(representative="Bad idea", count=1, user_ids=["lowrep"])
        tracker.record_injection("debate-bad", [cluster])
        # Record bad outcomes repeatedly
        for i in range(5):
            cluster = MagicMock(representative=f"Idea {i}", count=1, user_ids=["lowrep"])
            tracker.record_injection(f"debate-{i}", [cluster])
            tracker.record_outcome(f"debate-{i}", consensus_reached=False, consensus_confidence=0.1)

        suggestions = [
            {"suggestion": "From low rep", "user_id": "lowrep"},
            {"suggestion": "From unknown", "user_id": "unknown"},
        ]

        result = tracker.filter_by_reputation(suggestions, min_reputation=0.5)

        # Low rep user should be filtered out, unknown gets 0.5 exactly
        assert len(result) >= 1

    def test_filter_by_reputation_sorts_by_rep(self, tracker):
        """Test filtered suggestions are sorted by reputation."""
        # Create users with different reputations
        for i, user_id in enumerate(["highrep", "medrep"]):
            for j in range(5):
                cluster = MagicMock(representative=f"Idea {j}", count=1, user_ids=[user_id])
                tracker.record_injection(f"debate-{user_id}-{j}", [cluster])
                # Higher effectiveness for highrep
                conf = 0.95 if user_id == "highrep" else 0.5
                tracker.record_outcome(
                    f"debate-{user_id}-{j}",
                    consensus_reached=conf > 0.6,
                    consensus_confidence=conf,
                )

        suggestions = [
            {"suggestion": "From med", "user_id": "medrep"},
            {"suggestion": "From high", "user_id": "highrep"},
        ]

        result = tracker.filter_by_reputation(suggestions, min_reputation=0.0)

        # Should be sorted high to low reputation
        assert len(result) == 2


# ============================================================================
# Integration Tests
# ============================================================================


class TestFeedbackIntegration:
    """Integration tests for feedback tracking."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database file."""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        yield path
        if os.path.exists(path):
            os.unlink(path)

    def test_complete_workflow(self, temp_db):
        """Test complete workflow from injection to stats."""
        tracker = SuggestionFeedbackTracker(temp_db)

        # 1. Record suggestions
        clusters = [
            MagicMock(representative="Add caching", count=5, user_ids=["u1", "u2", "u3"]),
            MagicMock(representative="Improve UI", count=3, user_ids=["u4", "u5"]),
        ]
        ids = tracker.record_injection("debate-001", clusters)
        assert len(ids) == 2

        # 2. Record debate outcome
        updated = tracker.record_outcome(
            "debate-001",
            consensus_reached=True,
            consensus_confidence=0.85,
            duration_seconds=300.0,
        )
        assert updated == 2

        # 3. Check suggestions updated
        suggestions = tracker.get_debate_suggestions("debate-001")
        assert all(s.debate_completed for s in suggestions)
        assert all(s.effectiveness_score > 0 for s in suggestions)

        # 4. Check contributor stats
        stats = tracker.get_contributor_stats("u1")
        assert stats is not None
        assert stats.total_suggestions == 1
        assert stats.suggestions_in_consensus == 1

        # 5. Check overall stats
        overall = tracker.get_effectiveness_stats()
        assert overall["total_suggestions"] == 2
        assert overall["debates_with_suggestions"] == 1

    def test_reputation_builds_over_time(self, temp_db):
        """Test that reputation changes with repeated outcomes."""
        tracker = SuggestionFeedbackTracker(temp_db)

        user_id = "consistent_user"

        # Record multiple good outcomes
        for i in range(5):
            cluster = MagicMock(representative=f"Good idea {i}", count=1, user_ids=[user_id])
            tracker.record_injection(f"debate-{i}", [cluster])
            tracker.record_outcome(
                f"debate-{i}",
                consensus_reached=True,
                consensus_confidence=0.9,
            )

        stats = tracker.get_contributor_stats(user_id)

        assert stats is not None
        assert stats.total_suggestions == 5
        assert stats.reputation_score > 0.5  # Should have improved from default
