"""Tests for memory tier analytics.

Tests the TierAnalyticsTracker which tracks memory usage
patterns and promotion effectiveness for ROI analysis.
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from aragora.memory.tier_manager import MemoryTier


class TestTierStats:
    """Test TierStats dataclass."""

    def test_tier_stats_creation(self):
        """Test creating TierStats with defaults."""
        from aragora.memory.tier_analytics import TierStats

        stats = TierStats(tier=MemoryTier.FAST)
        assert stats.tier == MemoryTier.FAST
        assert stats.entries == 0
        assert stats.total_hits == 0
        assert stats.avg_hits == 0.0

    def test_tier_stats_to_dict(self):
        """Test TierStats serialization."""
        from aragora.memory.tier_analytics import TierStats

        stats = TierStats(
            tier=MemoryTier.MEDIUM,
            entries=10,
            total_hits=50,
            avg_hits=5.0,
            total_quality_impact=0.5,
            avg_quality_impact=0.01,
            promotions_in=3,
            promotions_out=2,
        )
        d = stats.to_dict()

        assert d["tier"] == "medium"
        assert d["entries"] == 10
        assert d["total_hits"] == 50
        assert d["avg_hits"] == 5.0
        assert d["promotions_in"] == 3


class TestMemoryUsageEvent:
    """Test MemoryUsageEvent dataclass."""

    def test_quality_impact_positive(self):
        """Test positive quality impact calculation."""
        from aragora.memory.tier_analytics import MemoryUsageEvent

        event = MemoryUsageEvent(
            memory_id="mem_1",
            tier=MemoryTier.FAST,
            debate_id="debate_1",
            quality_before=0.7,
            quality_after=0.85,
        )
        assert event.quality_impact == pytest.approx(0.15)

    def test_quality_impact_negative(self):
        """Test negative quality impact."""
        from aragora.memory.tier_analytics import MemoryUsageEvent

        event = MemoryUsageEvent(
            memory_id="mem_1",
            tier=MemoryTier.FAST,
            debate_id="debate_1",
            quality_before=0.85,
            quality_after=0.7,
        )
        assert event.quality_impact == pytest.approx(-0.15)

    def test_auto_timestamp(self):
        """Test automatic timestamp generation."""
        from aragora.memory.tier_analytics import MemoryUsageEvent

        event = MemoryUsageEvent(
            memory_id="mem_1",
            tier=MemoryTier.FAST,
            debate_id="debate_1",
            quality_before=0.7,
            quality_after=0.85,
        )
        assert event.used_at  # Should be auto-populated


class TestTierMovement:
    """Test TierMovement dataclass."""

    def test_tier_movement_creation(self):
        """Test creating TierMovement."""
        from aragora.memory.tier_analytics import TierMovement

        movement = TierMovement(
            memory_id="mem_1",
            from_tier=MemoryTier.FAST,
            to_tier=MemoryTier.MEDIUM,
            reason="promotion",
        )
        assert movement.memory_id == "mem_1"
        assert movement.from_tier == MemoryTier.FAST
        assert movement.to_tier == MemoryTier.MEDIUM
        assert movement.reason == "promotion"
        assert movement.moved_at  # Auto-generated


class TestMemoryAnalytics:
    """Test MemoryAnalytics dataclass."""

    def test_memory_analytics_to_dict(self):
        """Test MemoryAnalytics serialization."""
        from aragora.memory.tier_analytics import MemoryAnalytics, TierStats

        tier_stats = {
            "fast": TierStats(tier=MemoryTier.FAST, entries=5, total_hits=10),
            "medium": TierStats(tier=MemoryTier.MEDIUM, entries=3, total_hits=6),
        }

        analytics = MemoryAnalytics(
            tier_stats=tier_stats,
            promotion_effectiveness=0.75,
            learning_velocity=2.5,
            total_entries=8,
            total_hits=16,
            overall_quality_impact=0.1,
            recommendations=["Test recommendation"],
        )

        d = analytics.to_dict()
        assert d["promotion_effectiveness"] == 0.75
        assert d["learning_velocity"] == 2.5
        assert d["total_entries"] == 8
        assert "fast" in d["tier_stats"]
        assert d["recommendations"] == ["Test recommendation"]


class TestTierAnalyticsTracker:
    """Test TierAnalyticsTracker class."""

    @pytest.fixture
    def tracker(self):
        """Create a tracker with temporary database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_analytics.db"
            from aragora.memory.tier_analytics import TierAnalyticsTracker

            tracker = TierAnalyticsTracker(str(db_path))
            yield tracker

    def test_tracker_initialization(self, tracker):
        """Test tracker initializes database."""
        assert tracker.db_path.exists()

    def test_record_usage(self, tracker):
        """Test recording memory usage."""
        # Should not raise
        tracker.record_usage(
            memory_id="mem_1",
            tier=MemoryTier.FAST,
            debate_id="debate_1",
            quality_before=0.7,
            quality_after=0.85,
        )

        # Record another usage
        tracker.record_usage(
            memory_id="mem_2",
            tier=MemoryTier.FAST,
            debate_id="debate_2",
            quality_before=0.6,
            quality_after=0.75,
        )

    def test_record_usage_duplicate(self, tracker):
        """Test duplicate usage is handled gracefully."""
        tracker.record_usage(
            memory_id="mem_1",
            tier=MemoryTier.FAST,
            debate_id="debate_1",
            quality_before=0.7,
            quality_after=0.85,
        )

        # Same memory + debate combination should be ignored
        tracker.record_usage(
            memory_id="mem_1",
            tier=MemoryTier.FAST,
            debate_id="debate_1",
            quality_before=0.8,
            quality_after=0.9,
        )

    def test_record_tier_movement(self, tracker):
        """Test recording tier movement."""
        tracker.record_tier_movement(
            memory_id="mem_1",
            from_tier=MemoryTier.FAST,
            to_tier=MemoryTier.MEDIUM,
            reason="promotion",
        )

        tracker.record_tier_movement(
            memory_id="mem_2",
            from_tier=MemoryTier.MEDIUM,
            to_tier=MemoryTier.FAST,
            reason="demotion",
        )

    def test_get_tier_stats(self, tracker):
        """Test retrieving tier statistics."""
        # Record some usage
        tracker.record_usage(
            memory_id="mem_1",
            tier=MemoryTier.FAST,
            debate_id="debate_1",
            quality_before=0.7,
            quality_after=0.85,
        )
        tracker.record_usage(
            memory_id="mem_1",
            tier=MemoryTier.FAST,
            debate_id="debate_2",
            quality_before=0.6,
            quality_after=0.75,
        )

        stats = tracker.get_tier_stats(MemoryTier.FAST)

        assert stats.tier == MemoryTier.FAST
        assert stats.entries == 1  # One unique memory
        assert stats.total_hits == 2  # Two usages

    def test_get_tier_stats_empty(self, tracker):
        """Test tier stats with no data."""
        stats = tracker.get_tier_stats(MemoryTier.GLACIAL)

        assert stats.entries == 0
        assert stats.total_hits == 0
        assert stats.avg_hits == 0.0

    def test_get_promotion_effectiveness_no_data(self, tracker):
        """Test promotion effectiveness with no data."""
        effectiveness = tracker.get_promotion_effectiveness()
        assert effectiveness == 0.5  # Neutral default

    def test_get_promotion_effectiveness_with_data(self, tracker):
        """Test promotion effectiveness calculation."""
        # Record a promotion
        tracker.record_tier_movement(
            memory_id="mem_1",
            from_tier=MemoryTier.FAST,
            to_tier=MemoryTier.MEDIUM,
            reason="promotion",
        )

        # Record positive usage after promotion
        tracker.record_usage(
            memory_id="mem_1",
            tier=MemoryTier.MEDIUM,
            debate_id="debate_1",
            quality_before=0.7,
            quality_after=0.85,
        )

        effectiveness = tracker.get_promotion_effectiveness()
        assert effectiveness == 1.0  # 100% effective (only one with positive impact)

    def test_get_learning_velocity(self, tracker):
        """Test learning velocity calculation."""
        # Record some usage
        tracker.record_usage(
            memory_id="mem_1",
            tier=MemoryTier.FAST,
            debate_id="debate_1",
            quality_before=0.7,
            quality_after=0.85,
        )
        tracker.record_usage(
            memory_id="mem_2",
            tier=MemoryTier.FAST,
            debate_id="debate_2",
            quality_before=0.6,
            quality_after=0.75,
        )

        velocity = tracker.get_learning_velocity(days=7)
        # 2 new memories in 7 days = ~0.29 per day
        assert velocity > 0

    def test_generate_recommendations_balanced(self, tracker):
        """Test recommendations for balanced system."""
        analytics = {
            "tier_stats": {
                "fast": {"entries": 10, "avg_hits": 2.0},
                "medium": {"entries": 8, "avg_hits": 1.5},
                "slow": {"entries": 5, "avg_hits": 1.0},
                "glacial": {"entries": 3, "avg_hits": 0.5},
            },
            "promotion_effectiveness": 0.7,
            "learning_velocity": 2.0,
        }

        recommendations = tracker.generate_recommendations(analytics)
        assert "No action needed" in recommendations[0]

    def test_generate_recommendations_low_effectiveness(self, tracker):
        """Test recommendations for low promotion effectiveness."""
        analytics = {
            "tier_stats": {
                "fast": {"entries": 10, "avg_hits": 2.0},
                "medium": {"entries": 8, "avg_hits": 1.5},
            },
            "promotion_effectiveness": 0.3,
            "learning_velocity": 2.0,
        }

        recommendations = tracker.generate_recommendations(analytics)
        assert any("Low promotion effectiveness" in r for r in recommendations)

    def test_generate_recommendations_high_effectiveness(self, tracker):
        """Test recommendations for high promotion effectiveness."""
        analytics = {
            "tier_stats": {
                "fast": {"entries": 10, "avg_hits": 2.0},
                "medium": {"entries": 8, "avg_hits": 1.5},
            },
            "promotion_effectiveness": 0.9,
            "learning_velocity": 2.0,
        }

        recommendations = tracker.generate_recommendations(analytics)
        assert any("aggressive promotion" in r for r in recommendations)

    def test_generate_recommendations_underutilized_fast(self, tracker):
        """Test recommendations for underutilized fast tier."""
        analytics = {
            "tier_stats": {
                "fast": {"entries": 10, "avg_hits": 0.5},
                "medium": {"entries": 8, "avg_hits": 1.5},
            },
            "promotion_effectiveness": 0.7,
            "learning_velocity": 2.0,
        }

        recommendations = tracker.generate_recommendations(analytics)
        assert any("underutilized" in r for r in recommendations)

    def test_get_analytics(self, tracker):
        """Test generating comprehensive analytics."""
        # Add some data
        tracker.record_usage(
            memory_id="mem_1",
            tier=MemoryTier.FAST,
            debate_id="debate_1",
            quality_before=0.7,
            quality_after=0.85,
        )

        analytics = tracker.get_analytics()

        assert isinstance(analytics.tier_stats, dict)
        assert MemoryTier.FAST.value in analytics.tier_stats
        assert analytics.promotion_effectiveness >= 0
        assert analytics.learning_velocity >= 0
        assert len(analytics.recommendations) > 0

    def test_take_snapshot(self, tracker):
        """Test taking daily snapshot."""
        # Record some data
        tracker.record_usage(
            memory_id="mem_1",
            tier=MemoryTier.FAST,
            debate_id="debate_1",
            quality_before=0.7,
            quality_after=0.85,
        )

        # Should not raise
        tracker.take_snapshot()

        # Taking snapshot again should handle duplicate gracefully
        tracker.take_snapshot()


class TestTierAnalyticsIntegration:
    """Integration tests for tier analytics with realistic scenarios."""

    @pytest.fixture
    def tracker(self):
        """Create a tracker with temporary database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_analytics.db"
            from aragora.memory.tier_analytics import TierAnalyticsTracker

            tracker = TierAnalyticsTracker(str(db_path))
            yield tracker

    def test_full_lifecycle(self, tracker):
        """Test full memory lifecycle with analytics."""
        # 1. Memory starts in fast tier
        tracker.record_usage(
            memory_id="pattern_1",
            tier=MemoryTier.FAST,
            debate_id="debate_1",
            quality_before=0.6,
            quality_after=0.75,
        )

        # 2. Memory is used again successfully
        tracker.record_usage(
            memory_id="pattern_1",
            tier=MemoryTier.FAST,
            debate_id="debate_2",
            quality_before=0.65,
            quality_after=0.8,
        )

        # 3. Memory is promoted to medium tier
        tracker.record_tier_movement(
            memory_id="pattern_1",
            from_tier=MemoryTier.FAST,
            to_tier=MemoryTier.MEDIUM,
            reason="promotion",
        )

        # 4. Memory continues to be useful in medium tier
        tracker.record_usage(
            memory_id="pattern_1",
            tier=MemoryTier.MEDIUM,
            debate_id="debate_3",
            quality_before=0.7,
            quality_after=0.85,
        )

        # 5. Check analytics
        analytics = tracker.get_analytics()

        assert analytics.total_entries > 0
        assert analytics.total_hits > 0
        assert analytics.promotion_effectiveness >= 0
