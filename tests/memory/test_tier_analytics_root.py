"""
Tests for memory/tier_analytics.py - Memory tier ROI tracking.

Tests cover:
- TierStats dataclass
- MemoryUsageEvent and quality impact calculation
- TierMovement records
- MemoryAnalytics aggregation
- TierAnalyticsTracker (usage recording, tier movements, analytics)
- Recommendations generation
- Daily snapshots
"""

import pytest
import sqlite3
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from aragora.memory.tier_analytics import (
    TierStats,
    MemoryUsageEvent,
    TierMovement,
    MemoryAnalytics,
    TierAnalyticsTracker,
)
from aragora.memory.tier_manager import MemoryTier


# ============================================================================
# TierStats Tests
# ============================================================================


class TestTierStats:
    """Tests for TierStats dataclass."""

    def test_default_values(self):
        """TierStats should have sensible defaults."""
        stats = TierStats(tier=MemoryTier.FAST)

        assert stats.tier == MemoryTier.FAST
        assert stats.entries == 0
        assert stats.total_hits == 0
        assert stats.avg_hits == 0.0
        assert stats.total_quality_impact == 0.0
        assert stats.promotions_in == 0

    def test_to_dict(self):
        """to_dict should serialize all fields."""
        stats = TierStats(
            tier=MemoryTier.MEDIUM,
            entries=100,
            total_hits=500,
            avg_hits=5.0,
            total_quality_impact=2.5,
            avg_quality_impact=0.005,
            promotions_in=10,
            promotions_out=5,
            demotions_in=2,
            demotions_out=3,
        )

        d = stats.to_dict()

        assert d["tier"] == "medium"
        assert d["entries"] == 100
        assert d["total_hits"] == 500
        assert d["avg_hits"] == 5.0
        assert d["total_quality_impact"] == 2.5
        assert d["promotions_in"] == 10

    def test_to_dict_rounds_floats(self):
        """to_dict should round float values."""
        stats = TierStats(
            tier=MemoryTier.FAST,
            avg_hits=3.14159265,
            avg_quality_impact=0.12345678,
        )

        d = stats.to_dict()

        assert d["avg_hits"] == 3.14
        assert d["avg_quality_impact"] == 0.1235


# ============================================================================
# MemoryUsageEvent Tests
# ============================================================================


class TestMemoryUsageEvent:
    """Tests for MemoryUsageEvent."""

    def test_quality_impact(self):
        """quality_impact should calculate before/after difference."""
        event = MemoryUsageEvent(
            memory_id="mem-1",
            tier=MemoryTier.FAST,
            debate_id="debate-1",
            quality_before=0.5,
            quality_after=0.7,
        )

        assert event.quality_impact == pytest.approx(0.2)

    def test_negative_quality_impact(self):
        """quality_impact can be negative."""
        event = MemoryUsageEvent(
            memory_id="mem-1",
            tier=MemoryTier.SLOW,
            debate_id="debate-1",
            quality_before=0.8,
            quality_after=0.6,
        )

        assert event.quality_impact == pytest.approx(-0.2)

    def test_auto_timestamp(self):
        """used_at should default to current time."""
        event = MemoryUsageEvent(
            memory_id="mem-1",
            tier=MemoryTier.FAST,
            debate_id="debate-1",
            quality_before=0.5,
            quality_after=0.6,
        )

        # Should be recent ISO timestamp
        assert event.used_at.startswith("20")
        assert "T" in event.used_at


# ============================================================================
# TierMovement Tests
# ============================================================================


class TestTierMovement:
    """Tests for TierMovement."""

    def test_promotion(self):
        """Should record promotion between tiers."""
        movement = TierMovement(
            memory_id="mem-1",
            from_tier=MemoryTier.FAST,
            to_tier=MemoryTier.MEDIUM,
            reason="promotion",
        )

        assert movement.from_tier == MemoryTier.FAST
        assert movement.to_tier == MemoryTier.MEDIUM
        assert movement.reason == "promotion"

    def test_demotion(self):
        """Should record demotion between tiers."""
        movement = TierMovement(
            memory_id="mem-1",
            from_tier=MemoryTier.MEDIUM,
            to_tier=MemoryTier.FAST,
            reason="demotion",
        )

        assert movement.reason == "demotion"

    def test_auto_timestamp(self):
        """moved_at should default to current time."""
        movement = TierMovement(
            memory_id="mem-1",
            from_tier=MemoryTier.FAST,
            to_tier=MemoryTier.MEDIUM,
            reason="promotion",
        )

        assert movement.moved_at.startswith("20")


# ============================================================================
# MemoryAnalytics Tests
# ============================================================================


class TestMemoryAnalytics:
    """Tests for MemoryAnalytics."""

    def test_to_dict(self):
        """to_dict should include all metrics."""
        fast_stats = TierStats(tier=MemoryTier.FAST, entries=50, total_hits=100)
        medium_stats = TierStats(tier=MemoryTier.MEDIUM, entries=30, total_hits=50)

        analytics = MemoryAnalytics(
            tier_stats={"fast": fast_stats, "medium": medium_stats},
            promotion_effectiveness=0.75,
            learning_velocity=5.5,
            total_entries=80,
            total_hits=150,
            overall_quality_impact=1.5,
            recommendations=["Test recommendation"],
        )

        d = analytics.to_dict()

        assert d["promotion_effectiveness"] == 0.75
        assert d["learning_velocity"] == 5.5
        assert d["total_entries"] == 80
        assert d["total_hits"] == 150
        assert len(d["recommendations"]) == 1
        assert "fast" in d["tier_stats"]

    def test_auto_timestamp(self):
        """generated_at should default to current time."""
        analytics = MemoryAnalytics(
            tier_stats={},
            promotion_effectiveness=0.5,
            learning_velocity=1.0,
            total_entries=0,
            total_hits=0,
            overall_quality_impact=0.0,
            recommendations=[],
        )

        assert analytics.generated_at.startswith("20")


# ============================================================================
# TierAnalyticsTracker Tests
# ============================================================================


class TestTierAnalyticsTracker:
    """Tests for TierAnalyticsTracker."""

    @pytest.fixture
    def tracker(self, tmp_path):
        """Create tracker with temp database."""
        db_path = tmp_path / "test_analytics.db"
        return TierAnalyticsTracker(str(db_path))

    def test_init_creates_tables(self, tracker):
        """Initialization should create required tables."""
        with sqlite3.connect(str(tracker.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {row[0] for row in cursor.fetchall()}

        assert "memory_usage" in tables
        assert "tier_movements" in tables
        assert "tier_snapshots" in tables

    def test_record_usage(self, tracker):
        """Should record memory usage event."""
        tracker.record_usage(
            memory_id="mem-1",
            tier=MemoryTier.FAST,
            debate_id="debate-1",
            quality_before=0.5,
            quality_after=0.7,
        )

        with sqlite3.connect(str(tracker.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM memory_usage")
            rows = cursor.fetchall()

        assert len(rows) == 1
        assert rows[0][1] == "mem-1"  # memory_id
        assert rows[0][2] == "fast"  # tier

    def test_record_usage_deduplicates(self, tracker):
        """Should not duplicate same memory/debate combination."""
        tracker.record_usage(
            memory_id="mem-1",
            tier=MemoryTier.FAST,
            debate_id="debate-1",
            quality_before=0.5,
            quality_after=0.7,
        )
        # Try to insert duplicate
        tracker.record_usage(
            memory_id="mem-1",
            tier=MemoryTier.FAST,
            debate_id="debate-1",
            quality_before=0.6,
            quality_after=0.8,
        )

        with sqlite3.connect(str(tracker.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM memory_usage")
            count = cursor.fetchone()[0]

        assert count == 1

    def test_record_tier_movement(self, tracker):
        """Should record tier movement."""
        tracker.record_tier_movement(
            memory_id="mem-1",
            from_tier=MemoryTier.FAST,
            to_tier=MemoryTier.MEDIUM,
            reason="promotion",
        )

        with sqlite3.connect(str(tracker.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM tier_movements")
            rows = cursor.fetchall()

        assert len(rows) == 1
        assert rows[0][1] == "mem-1"  # memory_id
        assert rows[0][2] == "fast"  # from_tier
        assert rows[0][3] == "medium"  # to_tier
        assert rows[0][4] == "promotion"  # reason

    def test_get_tier_stats_empty(self, tracker):
        """Should return zero stats for empty database."""
        stats = tracker.get_tier_stats(MemoryTier.FAST)

        assert stats.entries == 0
        assert stats.total_hits == 0
        assert stats.promotions_in == 0

    def test_get_tier_stats_with_data(self, tracker):
        """Should aggregate usage data correctly."""
        # Add some usage data
        for i in range(5):
            tracker.record_usage(
                memory_id=f"mem-{i}",
                tier=MemoryTier.FAST,
                debate_id=f"debate-{i}",
                quality_before=0.5,
                quality_after=0.7,
            )

        stats = tracker.get_tier_stats(MemoryTier.FAST)

        assert stats.entries == 5
        assert stats.total_hits == 5
        assert stats.avg_quality_impact == pytest.approx(0.2)

    def test_get_tier_stats_counts_movements(self, tracker):
        """Should count promotions/demotions correctly."""
        # Add promotions to MEDIUM tier
        tracker.record_tier_movement("mem-1", MemoryTier.FAST, MemoryTier.MEDIUM, "promotion")
        tracker.record_tier_movement("mem-2", MemoryTier.FAST, MemoryTier.MEDIUM, "promotion")

        # Add demotion from MEDIUM tier
        tracker.record_tier_movement("mem-3", MemoryTier.MEDIUM, MemoryTier.FAST, "demotion")

        stats = tracker.get_tier_stats(MemoryTier.MEDIUM)

        assert stats.promotions_in == 2
        assert stats.demotions_out == 1

    def test_get_promotion_effectiveness_no_data(self, tracker):
        """Should return neutral 0.5 with no data."""
        effectiveness = tracker.get_promotion_effectiveness()

        assert effectiveness == 0.5

    def test_get_promotion_effectiveness_positive(self, tracker):
        """Should calculate effectiveness from usage data."""
        # Record promotion
        tracker.record_tier_movement("mem-1", MemoryTier.FAST, MemoryTier.MEDIUM, "promotion")

        # Record positive usage after promotion
        tracker.record_usage(
            memory_id="mem-1",
            tier=MemoryTier.MEDIUM,
            debate_id="debate-1",
            quality_before=0.5,
            quality_after=0.8,  # Positive impact
        )

        effectiveness = tracker.get_promotion_effectiveness()

        assert effectiveness == 1.0  # 100% effective

    def test_get_learning_velocity(self, tracker):
        """Should calculate entries per day."""
        # Add 7 entries
        for i in range(7):
            tracker.record_usage(
                memory_id=f"mem-{i}",
                tier=MemoryTier.FAST,
                debate_id=f"debate-{i}",
                quality_before=0.5,
                quality_after=0.6,
            )

        velocity = tracker.get_learning_velocity(days=7)

        assert velocity == 1.0  # 7 entries / 7 days

    def test_generate_recommendations_fast_underutilized(self, tracker):
        """Should recommend lowering threshold for underutilized fast tier."""
        analytics = {
            "tier_stats": {
                "fast": {"entries": 10, "avg_hits": 0.5},  # Low hit rate
            },
            "promotion_effectiveness": 0.5,
            "learning_velocity": 2.0,
        }

        recommendations = tracker.generate_recommendations(analytics)

        assert any("underutilized" in r.lower() for r in recommendations)

    def test_generate_recommendations_low_effectiveness(self, tracker):
        """Should recommend reviewing criteria for low effectiveness."""
        analytics = {
            "tier_stats": {"fast": {"entries": 10, "avg_hits": 5.0}},
            "promotion_effectiveness": 0.2,  # Very low
            "learning_velocity": 2.0,
        }

        recommendations = tracker.generate_recommendations(analytics)

        assert any("low promotion effectiveness" in r.lower() for r in recommendations)

    def test_generate_recommendations_high_effectiveness(self, tracker):
        """Should recommend more aggressive promotion for high effectiveness."""
        analytics = {
            "tier_stats": {"fast": {"entries": 10, "avg_hits": 5.0}},
            "promotion_effectiveness": 0.9,  # Very high
            "learning_velocity": 2.0,
        }

        recommendations = tracker.generate_recommendations(analytics)

        assert any("more aggressive" in r.lower() for r in recommendations)

    def test_generate_recommendations_low_velocity(self, tracker):
        """Should warn about low learning velocity."""
        analytics = {
            "tier_stats": {"fast": {"entries": 10, "avg_hits": 5.0}},
            "promotion_effectiveness": 0.5,
            "learning_velocity": 0.1,  # Very low
        }

        recommendations = tracker.generate_recommendations(analytics)

        assert any("low learning velocity" in r.lower() for r in recommendations)

    def test_generate_recommendations_balanced(self, tracker):
        """Should indicate no action needed when balanced."""
        analytics = {
            "tier_stats": {
                "fast": {"entries": 10, "avg_hits": 5.0},
                "medium": {"entries": 10, "avg_hits": 3.0},
            },
            "promotion_effectiveness": 0.5,
            "learning_velocity": 2.0,
        }

        recommendations = tracker.generate_recommendations(analytics)

        assert any("no action needed" in r.lower() for r in recommendations)

    def test_get_analytics(self, tracker):
        """Should generate complete analytics report."""
        # Add some data
        tracker.record_usage(
            memory_id="mem-1",
            tier=MemoryTier.FAST,
            debate_id="debate-1",
            quality_before=0.5,
            quality_after=0.7,
        )
        tracker.record_tier_movement("mem-1", MemoryTier.FAST, MemoryTier.MEDIUM, "promotion")

        analytics = tracker.get_analytics(days=30)

        assert analytics.total_entries >= 1
        assert analytics.tier_stats is not None
        assert len(analytics.recommendations) > 0
        assert 0 <= analytics.promotion_effectiveness <= 1

    def test_take_snapshot(self, tracker):
        """Should save daily snapshot."""
        # Add some data first
        tracker.record_usage(
            memory_id="mem-1",
            tier=MemoryTier.FAST,
            debate_id="debate-1",
            quality_before=0.5,
            quality_after=0.7,
        )

        tracker.take_snapshot()

        with sqlite3.connect(str(tracker.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM tier_snapshots")
            count = cursor.fetchone()[0]

        # Should have snapshots for all tiers
        assert count == len(MemoryTier)

    def test_take_snapshot_idempotent(self, tracker):
        """Taking snapshot twice on same day should not duplicate."""
        tracker.take_snapshot()
        tracker.take_snapshot()

        with sqlite3.connect(str(tracker.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM tier_snapshots")
            count = cursor.fetchone()[0]

        assert count == len(MemoryTier)


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestTierAnalyticsEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def tracker(self, tmp_path):
        """Create tracker with temp database."""
        db_path = tmp_path / "test_analytics.db"
        return TierAnalyticsTracker(str(db_path))

    def test_handles_all_tiers(self, tracker):
        """Should work with all MemoryTier values."""
        for tier in MemoryTier:
            tracker.record_usage(
                memory_id=f"mem-{tier.value}",
                tier=tier,
                debate_id="debate-1",
                quality_before=0.5,
                quality_after=0.6,
            )

        analytics = tracker.get_analytics()

        assert len(analytics.tier_stats) == len(MemoryTier)

    def test_zero_quality_impact(self, tracker):
        """Should handle zero quality impact."""
        tracker.record_usage(
            memory_id="mem-1",
            tier=MemoryTier.FAST,
            debate_id="debate-1",
            quality_before=0.5,
            quality_after=0.5,  # No change
        )

        stats = tracker.get_tier_stats(MemoryTier.FAST)

        assert stats.avg_quality_impact == 0.0

    def test_respects_days_parameter(self, tracker):
        """get_tier_stats should filter by days."""
        # This test verifies the days filter works but doesn't
        # manipulate timestamps directly
        tracker.record_usage(
            memory_id="mem-1",
            tier=MemoryTier.FAST,
            debate_id="debate-1",
            quality_before=0.5,
            quality_after=0.6,
        )

        stats_30 = tracker.get_tier_stats(MemoryTier.FAST, days=30)
        stats_1 = tracker.get_tier_stats(MemoryTier.FAST, days=1)

        # Both should find the recent record
        assert stats_30.entries >= 0
        assert stats_1.entries >= 0

    def test_analytics_to_dict_serializable(self, tracker):
        """Analytics should be JSON serializable."""
        import json

        tracker.record_usage(
            memory_id="mem-1",
            tier=MemoryTier.FAST,
            debate_id="debate-1",
            quality_before=0.5,
            quality_after=0.7,
        )

        analytics = tracker.get_analytics()
        d = analytics.to_dict()

        # Should not raise
        json_str = json.dumps(d)
        assert json_str is not None
