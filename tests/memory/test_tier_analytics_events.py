"""Tests for tier analytics event emission.

Verifies that TierAnalyticsTracker emits MEMORY_TIER_PROMOTION and
MEMORY_TIER_DEMOTION events via the cross-subscriber manager when
tier movements are recorded.
"""

import tempfile
from unittest.mock import MagicMock, patch

import pytest

from aragora.memory.tier_manager import MemoryTier


class TestIsPromotion:
    """Test the _is_promotion static helper."""

    def test_slow_to_fast_is_promotion(self):
        from aragora.memory.tier_analytics import TierAnalyticsTracker

        assert TierAnalyticsTracker._is_promotion(MemoryTier.SLOW, MemoryTier.FAST) is True

    def test_glacial_to_medium_is_promotion(self):
        from aragora.memory.tier_analytics import TierAnalyticsTracker

        assert TierAnalyticsTracker._is_promotion(MemoryTier.GLACIAL, MemoryTier.MEDIUM) is True

    def test_glacial_to_slow_is_promotion(self):
        from aragora.memory.tier_analytics import TierAnalyticsTracker

        assert TierAnalyticsTracker._is_promotion(MemoryTier.GLACIAL, MemoryTier.SLOW) is True

    def test_medium_to_fast_is_promotion(self):
        from aragora.memory.tier_analytics import TierAnalyticsTracker

        assert TierAnalyticsTracker._is_promotion(MemoryTier.MEDIUM, MemoryTier.FAST) is True

    def test_fast_to_slow_is_demotion(self):
        from aragora.memory.tier_analytics import TierAnalyticsTracker

        assert TierAnalyticsTracker._is_promotion(MemoryTier.FAST, MemoryTier.SLOW) is False

    def test_fast_to_glacial_is_demotion(self):
        from aragora.memory.tier_analytics import TierAnalyticsTracker

        assert TierAnalyticsTracker._is_promotion(MemoryTier.FAST, MemoryTier.GLACIAL) is False

    def test_medium_to_slow_is_demotion(self):
        from aragora.memory.tier_analytics import TierAnalyticsTracker

        assert TierAnalyticsTracker._is_promotion(MemoryTier.MEDIUM, MemoryTier.SLOW) is False

    def test_same_tier_is_not_promotion(self):
        from aragora.memory.tier_analytics import TierAnalyticsTracker

        assert TierAnalyticsTracker._is_promotion(MemoryTier.FAST, MemoryTier.FAST) is False

    def test_glacial_to_fast_is_promotion(self):
        from aragora.memory.tier_analytics import TierAnalyticsTracker

        assert TierAnalyticsTracker._is_promotion(MemoryTier.GLACIAL, MemoryTier.FAST) is True


class TestTierMovementEventEmission:
    """Test that record_tier_movement emits correct events."""

    @pytest.fixture()
    def tracker(self, tmp_path):
        """Create a tracker with a temporary database."""
        db_path = str(tmp_path / "test_analytics.db")
        with patch("aragora.memory.tier_analytics.resolve_db_path", return_value=db_path):
            from aragora.memory.tier_analytics import TierAnalyticsTracker

            return TierAnalyticsTracker(db_path=db_path)

    def test_promotion_emits_promotion_event(self, tracker):
        """MEMORY_TIER_PROMOTION emitted when moving to a faster tier."""
        from aragora.events.types import StreamEventType

        mock_manager = MagicMock()
        with patch(
            "aragora.events.cross_subscribers.get_cross_subscriber_manager",
            return_value=mock_manager,
        ):
            tracker.record_tier_movement(
                memory_id="mem_1",
                from_tier=MemoryTier.SLOW,
                to_tier=MemoryTier.FAST,
                reason="high_usage",
            )

        mock_manager.dispatch.assert_called_once()
        event = mock_manager.dispatch.call_args[0][0]
        assert event.type == StreamEventType.MEMORY_TIER_PROMOTION
        assert event.data["memory_id"] == "mem_1"
        assert event.data["from_tier"] == "slow"
        assert event.data["to_tier"] == "fast"
        assert event.data["reason"] == "high_usage"

    def test_demotion_emits_demotion_event(self, tracker):
        """MEMORY_TIER_DEMOTION emitted when moving to a slower tier."""
        from aragora.events.types import StreamEventType

        mock_manager = MagicMock()
        with patch(
            "aragora.events.cross_subscribers.get_cross_subscriber_manager",
            return_value=mock_manager,
        ):
            tracker.record_tier_movement(
                memory_id="mem_2",
                from_tier=MemoryTier.FAST,
                to_tier=MemoryTier.SLOW,
                reason="low_usage",
            )

        mock_manager.dispatch.assert_called_once()
        event = mock_manager.dispatch.call_args[0][0]
        assert event.type == StreamEventType.MEMORY_TIER_DEMOTION
        assert event.data["memory_id"] == "mem_2"
        assert event.data["from_tier"] == "fast"
        assert event.data["to_tier"] == "slow"
        assert event.data["reason"] == "low_usage"

    def test_glacial_to_medium_emits_promotion(self, tracker):
        """Multi-tier jump still classifies correctly as promotion."""
        from aragora.events.types import StreamEventType

        mock_manager = MagicMock()
        with patch(
            "aragora.events.cross_subscribers.get_cross_subscriber_manager",
            return_value=mock_manager,
        ):
            tracker.record_tier_movement(
                memory_id="mem_3",
                from_tier=MemoryTier.GLACIAL,
                to_tier=MemoryTier.MEDIUM,
                reason="revival",
            )

        event = mock_manager.dispatch.call_args[0][0]
        assert event.type == StreamEventType.MEMORY_TIER_PROMOTION

    def test_medium_to_glacial_emits_demotion(self, tracker):
        """Multi-tier jump downward classifies as demotion."""
        from aragora.events.types import StreamEventType

        mock_manager = MagicMock()
        with patch(
            "aragora.events.cross_subscribers.get_cross_subscriber_manager",
            return_value=mock_manager,
        ):
            tracker.record_tier_movement(
                memory_id="mem_4",
                from_tier=MemoryTier.MEDIUM,
                to_tier=MemoryTier.GLACIAL,
                reason="decay",
            )

        event = mock_manager.dispatch.call_args[0][0]
        assert event.type == StreamEventType.MEMORY_TIER_DEMOTION

    def test_db_write_succeeds_even_when_event_emission_fails(self, tracker):
        """Database write completes even if event dispatch raises."""
        with patch(
            "aragora.events.cross_subscribers.get_cross_subscriber_manager",
            side_effect=RuntimeError("no manager"),
        ):
            tracker.record_tier_movement(
                memory_id="mem_5",
                from_tier=MemoryTier.SLOW,
                to_tier=MemoryTier.FAST,
                reason="test",
            )

        # Verify the DB write still happened
        with tracker._db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT memory_id FROM tier_movements WHERE memory_id = ?",
                ("mem_5",),
            )
            row = cursor.fetchone()
            assert row is not None
            assert row[0] == "mem_5"


class TestEventEmissionGracefulDegradation:
    """Test that event emission failures are handled gracefully."""

    @pytest.fixture()
    def tracker(self, tmp_path):
        db_path = str(tmp_path / "test_analytics.db")
        with patch("aragora.memory.tier_analytics.resolve_db_path", return_value=db_path):
            from aragora.memory.tier_analytics import TierAnalyticsTracker

            return TierAnalyticsTracker(db_path=db_path)

    def test_graceful_on_import_error(self, tracker):
        """Event emission gracefully handles ImportError."""
        with patch(
            "aragora.memory.tier_analytics.TierAnalyticsTracker._is_promotion",
            side_effect=ImportError("module not found"),
        ):
            # Should not raise
            tracker.record_tier_movement(
                memory_id="mem_ie",
                from_tier=MemoryTier.SLOW,
                to_tier=MemoryTier.FAST,
                reason="test",
            )

        # DB write still succeeded
        with tracker._db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT memory_id FROM tier_movements WHERE memory_id = ?",
                ("mem_ie",),
            )
            assert cursor.fetchone() is not None

    def test_graceful_on_dispatcher_runtime_error(self, tracker):
        """Event emission gracefully handles RuntimeError from dispatcher."""
        with patch(
            "aragora.events.cross_subscribers.get_cross_subscriber_manager",
            side_effect=RuntimeError("dispatcher not initialized"),
        ):
            # Should not raise
            tracker.record_tier_movement(
                memory_id="mem_re",
                from_tier=MemoryTier.FAST,
                to_tier=MemoryTier.SLOW,
                reason="test",
            )

        with tracker._db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT memory_id FROM tier_movements WHERE memory_id = ?",
                ("mem_re",),
            )
            assert cursor.fetchone() is not None

    def test_graceful_on_attribute_error(self, tracker):
        """Event emission gracefully handles AttributeError (e.g., dispatch missing)."""
        mock_manager = MagicMock()
        mock_manager.dispatch = None  # Will raise AttributeError when called
        del mock_manager.dispatch  # Remove entirely so attribute access fails

        # Create a manager that raises AttributeError on dispatch
        bad_manager = MagicMock(spec=[])  # No attributes at all
        with patch(
            "aragora.events.cross_subscribers.get_cross_subscriber_manager",
            return_value=bad_manager,
        ):
            # Should not raise
            tracker.record_tier_movement(
                memory_id="mem_ae",
                from_tier=MemoryTier.MEDIUM,
                to_tier=MemoryTier.GLACIAL,
                reason="test",
            )

        with tracker._db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT memory_id FROM tier_movements WHERE memory_id = ?",
                ("mem_ae",),
            )
            assert cursor.fetchone() is not None
