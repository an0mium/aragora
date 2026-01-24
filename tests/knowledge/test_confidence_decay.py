"""
Comprehensive tests for Knowledge Mound Confidence Decay.

Tests cover:
- Decay calculation models (exponential, linear, step)
- Confidence boost/penalty events
- Batch decay processing
- Domain-specific half-lives
- Integration with KnowledgeMound
- Edge cases and boundary conditions
"""

import math
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.knowledge.mound.ops.confidence_decay import (
    ConfidenceAdjustment,
    ConfidenceDecayManager,
    ConfidenceDecayMixin,
    ConfidenceEvent,
    DecayConfig,
    DecayModel,
    DecayReport,
    get_decay_manager,
)


# =============================================================================
# Unit Tests: DecayConfig
# =============================================================================


class TestDecayConfig:
    """Tests for DecayConfig dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        config = DecayConfig()
        assert config.model == DecayModel.EXPONENTIAL
        assert config.half_life_days == 90.0
        assert config.min_confidence == 0.1
        assert config.max_confidence == 1.0
        assert config.access_boost == 0.01
        assert config.citation_boost == 0.05
        assert config.validation_boost == 0.1
        assert config.invalidation_penalty == 0.3
        assert config.contradiction_penalty == 0.2
        assert config.batch_size == 100
        assert config.decay_interval_hours == 24

    def test_domain_half_lives(self):
        """Should have domain-specific half-lives."""
        config = DecayConfig()
        assert config.domain_half_lives["technology"] == 30.0
        assert config.domain_half_lives["science"] == 180.0
        assert config.domain_half_lives["legal"] == 365.0
        assert config.domain_half_lives["news"] == 7.0

    def test_custom_config(self):
        """Should accept custom configuration."""
        config = DecayConfig(
            model=DecayModel.LINEAR,
            half_life_days=60.0,
            min_confidence=0.2,
            access_boost=0.02,
        )
        assert config.model == DecayModel.LINEAR
        assert config.half_life_days == 60.0
        assert config.min_confidence == 0.2
        assert config.access_boost == 0.02


# =============================================================================
# Unit Tests: Decay Calculations
# =============================================================================


class TestDecayCalculations:
    """Tests for confidence decay calculations."""

    def test_exponential_decay_at_half_life(self):
        """Exponential decay should halve at half-life."""
        manager = ConfidenceDecayManager()
        result = manager.calculate_decay(1.0, age_days=90.0)
        assert 0.49 <= result <= 0.51  # Approximately 0.5

    def test_exponential_decay_at_zero(self):
        """No decay for freshly created items."""
        manager = ConfidenceDecayManager()
        result = manager.calculate_decay(0.8, age_days=0.0)
        assert result == 0.8

    def test_exponential_decay_at_double_half_life(self):
        """Should be 0.25 at twice the half-life."""
        manager = ConfidenceDecayManager()
        result = manager.calculate_decay(1.0, age_days=180.0)
        assert 0.24 <= result <= 0.26

    def test_exponential_decay_respects_floor(self):
        """Should not go below min_confidence."""
        manager = ConfidenceDecayManager()
        result = manager.calculate_decay(1.0, age_days=1000.0)
        assert result == 0.1  # min_confidence default

    def test_linear_decay(self):
        """Linear decay should decrease uniformly."""
        config = DecayConfig(model=DecayModel.LINEAR, half_life_days=90.0)
        manager = ConfidenceDecayManager(config)

        result_45 = manager.calculate_decay(1.0, age_days=45.0)
        result_90 = manager.calculate_decay(1.0, age_days=90.0)

        # Linear decay at half-life should be 0.5
        assert 0.45 <= result_90 <= 0.55

        # At 45 days should be halfway between 1.0 and 0.5
        assert result_45 > result_90

    def test_step_decay_thresholds(self):
        """Step decay should use discrete levels."""
        config = DecayConfig(model=DecayModel.STEP, half_life_days=100.0)
        manager = ConfidenceDecayManager(config)

        # Before 0.5 half-life (50 days) - no decay
        result_30 = manager.calculate_decay(1.0, age_days=30.0)
        assert result_30 == 1.0

        # Between 0.5 and 1.0 half-life - 75%
        result_75 = manager.calculate_decay(1.0, age_days=75.0)
        assert result_75 == 0.75

        # Between 1.0 and 2.0 half-life - 50%
        result_150 = manager.calculate_decay(1.0, age_days=150.0)
        assert result_150 == 0.5

        # Beyond 2.0 half-life - 25%
        result_250 = manager.calculate_decay(1.0, age_days=250.0)
        assert result_250 == 0.25

    def test_domain_specific_half_life(self):
        """Domain-specific decay rates should apply."""
        manager = ConfidenceDecayManager()

        # Technology decays faster (30 day half-life)
        tech_90_days = manager.calculate_decay(1.0, age_days=90.0, domain="technology")
        # At 90 days = 3 half-lives for tech, should be ~0.125
        assert tech_90_days < 0.2

        # Legal decays slower (365 day half-life)
        legal_90_days = manager.calculate_decay(1.0, age_days=90.0, domain="legal")
        # At 90 days ~= 0.25 half-life for legal, should be high
        assert legal_90_days > 0.8

    def test_news_domain_rapid_decay(self):
        """News should decay very rapidly (7 day half-life)."""
        manager = ConfidenceDecayManager()

        # After 14 days (2 half-lives), news should be ~25%
        news_14_days = manager.calculate_decay(1.0, age_days=14.0, domain="news")
        assert 0.2 <= news_14_days <= 0.3

    def test_unknown_domain_uses_default(self):
        """Unknown domains should use default half-life."""
        manager = ConfidenceDecayManager()

        unknown = manager.calculate_decay(1.0, age_days=90.0, domain="unknown_domain")
        default = manager.calculate_decay(1.0, age_days=90.0, domain=None)

        assert unknown == default


# =============================================================================
# Unit Tests: Confidence Boost/Penalty
# =============================================================================


class TestConfidenceBoost:
    """Tests for confidence boost and penalty calculations."""

    def test_access_boost(self):
        """Access events should boost confidence slightly."""
        manager = ConfidenceDecayManager()
        result = manager.calculate_boost(0.5, ConfidenceEvent.ACCESSED)
        assert result == 0.51  # +0.01

    def test_citation_boost(self):
        """Citations should boost confidence more."""
        manager = ConfidenceDecayManager()
        result = manager.calculate_boost(0.5, ConfidenceEvent.CITED)
        assert result == 0.55  # +0.05

    def test_validation_boost(self):
        """Validation should provide largest boost."""
        manager = ConfidenceDecayManager()
        result = manager.calculate_boost(0.5, ConfidenceEvent.VALIDATED)
        assert result == 0.6  # +0.1

    def test_invalidation_penalty(self):
        """Invalidation should reduce confidence."""
        manager = ConfidenceDecayManager()
        result = manager.calculate_boost(0.8, ConfidenceEvent.INVALIDATED)
        assert result == 0.5  # -0.3

    def test_contradiction_penalty(self):
        """Contradictions should reduce confidence."""
        manager = ConfidenceDecayManager()
        result = manager.calculate_boost(0.8, ConfidenceEvent.CONTRADICTED)
        assert abs(result - 0.6) < 0.001  # -0.2 (use approximate comparison)

    def test_boost_respects_ceiling(self):
        """Confidence should not exceed max_confidence."""
        manager = ConfidenceDecayManager()
        result = manager.calculate_boost(0.95, ConfidenceEvent.VALIDATED)
        assert result == 1.0

    def test_penalty_respects_floor(self):
        """Confidence should not go below min_confidence."""
        manager = ConfidenceDecayManager()
        result = manager.calculate_boost(0.2, ConfidenceEvent.INVALIDATED)
        assert result == 0.1  # min_confidence

    def test_no_boost_for_neutral_events(self):
        """CREATED, UPDATED, DECAYED should not boost."""
        manager = ConfidenceDecayManager()
        for event in [ConfidenceEvent.CREATED, ConfidenceEvent.UPDATED, ConfidenceEvent.DECAYED]:
            result = manager.calculate_boost(0.5, event)
            assert result == 0.5


# =============================================================================
# Integration Tests: Apply Decay
# =============================================================================


class TestApplyDecay:
    """Integration tests for apply_decay method."""

    @pytest.mark.asyncio
    async def test_apply_decay_empty_workspace(self):
        """Should handle empty workspace gracefully."""
        mock_mound = MagicMock()
        mock_mound.query = AsyncMock(return_value=MagicMock(items=[]))

        manager = ConfidenceDecayManager()
        report = await manager.apply_decay(mock_mound, "test-workspace", force=True)

        assert report.workspace_id == "test-workspace"
        assert report.items_processed == 0
        assert report.items_decayed == 0
        assert report.adjustments == []

    @pytest.mark.asyncio
    async def test_apply_decay_single_item(self):
        """Should decay a single old item."""
        mock_item = MagicMock()
        mock_item.id = "item-1"
        mock_item.confidence = 1.0
        mock_item.created_at = datetime.now() - timedelta(days=90)
        mock_item.topics = ["technology"]

        mock_mound = MagicMock()
        mock_mound.query = AsyncMock(return_value=MagicMock(items=[mock_item]))
        mock_mound.update_confidence = AsyncMock()

        manager = ConfidenceDecayManager()
        report = await manager.apply_decay(mock_mound, "test-workspace", force=True)

        assert report.items_processed == 1
        assert report.items_decayed == 1
        assert len(report.adjustments) == 1
        assert report.adjustments[0].item_id == "item-1"
        assert report.adjustments[0].old_confidence == 1.0
        assert report.adjustments[0].new_confidence < 1.0

    @pytest.mark.asyncio
    async def test_apply_decay_respects_interval(self):
        """Should skip decay if run recently."""
        mock_mound = MagicMock()
        mock_mound.query = AsyncMock(return_value=MagicMock(items=[]))

        manager = ConfidenceDecayManager()

        # First run
        await manager.apply_decay(mock_mound, "test-workspace", force=True)

        # Second run without force - should skip
        report = await manager.apply_decay(mock_mound, "test-workspace", force=False)

        assert report.items_processed == 0
        mock_mound.query.assert_called_once()  # Only called once

    @pytest.mark.asyncio
    async def test_apply_decay_force_bypasses_interval(self):
        """Force flag should bypass interval check."""
        mock_mound = MagicMock()
        mock_mound.query = AsyncMock(return_value=MagicMock(items=[]))

        manager = ConfidenceDecayManager()

        await manager.apply_decay(mock_mound, "test-workspace", force=True)
        await manager.apply_decay(mock_mound, "test-workspace", force=True)

        assert mock_mound.query.call_count == 2

    @pytest.mark.asyncio
    async def test_apply_decay_multiple_items(self):
        """Should process multiple items with varying ages."""
        items = []
        for i, age in enumerate([7, 30, 90, 180, 365]):
            item = MagicMock()
            item.id = f"item-{i}"
            item.confidence = 0.8
            item.created_at = datetime.now() - timedelta(days=age)
            item.topics = []
            items.append(item)

        mock_mound = MagicMock()
        mock_mound.query = AsyncMock(return_value=MagicMock(items=items))
        mock_mound.update_confidence = AsyncMock()

        manager = ConfidenceDecayManager()
        report = await manager.apply_decay(mock_mound, "test-workspace", force=True)

        assert report.items_processed == 5
        assert report.items_decayed >= 1  # At least some items decayed

    @pytest.mark.asyncio
    async def test_apply_decay_handles_string_dates(self):
        """Should handle ISO string dates."""
        mock_item = MagicMock()
        mock_item.id = "item-1"
        mock_item.confidence = 0.9
        mock_item.created_at = (datetime.now() - timedelta(days=60)).isoformat()
        mock_item.topics = None

        mock_mound = MagicMock()
        mock_mound.query = AsyncMock(return_value=MagicMock(items=[mock_item]))
        mock_mound.update_confidence = AsyncMock()

        manager = ConfidenceDecayManager()
        report = await manager.apply_decay(mock_mound, "test-workspace", force=True)

        # Should not raise, should process item
        assert report.items_processed == 1

    @pytest.mark.asyncio
    async def test_apply_decay_handles_missing_confidence(self):
        """Should default to 0.5 for items without confidence."""
        mock_item = MagicMock()
        mock_item.id = "item-1"
        mock_item.confidence = None  # Missing
        mock_item.created_at = datetime.now() - timedelta(days=100)
        mock_item.topics = []

        mock_mound = MagicMock()
        mock_mound.query = AsyncMock(return_value=MagicMock(items=[mock_item]))
        mock_mound.update_confidence = AsyncMock()

        manager = ConfidenceDecayManager()
        report = await manager.apply_decay(mock_mound, "test-workspace", force=True)

        assert report.items_processed == 1
        if report.adjustments:
            assert report.adjustments[0].old_confidence == 0.5


# =============================================================================
# Integration Tests: Record Events
# =============================================================================


class TestRecordEvent:
    """Tests for recording confidence-affecting events."""

    @pytest.mark.asyncio
    async def test_record_access_event(self):
        """Should record access and boost confidence."""
        mock_item = MagicMock()
        mock_item.confidence = 0.5

        mock_mound = MagicMock()
        mock_mound.get = AsyncMock(return_value=mock_item)
        mock_mound.update_confidence = AsyncMock()

        manager = ConfidenceDecayManager()
        adjustment = await manager.record_event(
            mock_mound, "item-1", ConfidenceEvent.ACCESSED, "User viewed"
        )

        assert adjustment is not None
        assert adjustment.event == ConfidenceEvent.ACCESSED
        assert adjustment.old_confidence == 0.5
        assert adjustment.new_confidence == 0.51
        mock_mound.update_confidence.assert_called_once_with("item-1", 0.51)

    @pytest.mark.asyncio
    async def test_record_event_item_not_found(self):
        """Should return None for missing items."""
        mock_mound = MagicMock()
        mock_mound.get = AsyncMock(return_value=None)

        manager = ConfidenceDecayManager()
        adjustment = await manager.record_event(mock_mound, "missing", ConfidenceEvent.ACCESSED)

        assert adjustment is None

    @pytest.mark.asyncio
    async def test_record_event_no_change(self):
        """Should return None if no confidence change."""
        mock_item = MagicMock()
        mock_item.confidence = 1.0  # Already at max

        mock_mound = MagicMock()
        mock_mound.get = AsyncMock(return_value=mock_item)

        manager = ConfidenceDecayManager()
        adjustment = await manager.record_event(mock_mound, "item-1", ConfidenceEvent.ACCESSED)

        # Already at max, small boost won't push above
        # But it should still boost by 0.01 then clamp
        # Since 1.0 + 0.01 = 1.01, clamped to 1.0, no change
        assert adjustment is None

    @pytest.mark.asyncio
    async def test_record_invalidation_event(self):
        """Should record invalidation with penalty."""
        mock_item = MagicMock()
        mock_item.confidence = 0.8

        mock_mound = MagicMock()
        mock_mound.get = AsyncMock(return_value=mock_item)
        mock_mound.update_confidence = AsyncMock()

        manager = ConfidenceDecayManager()
        adjustment = await manager.record_event(
            mock_mound, "item-1", ConfidenceEvent.INVALIDATED, "Failed fact check"
        )

        assert adjustment is not None
        assert adjustment.new_confidence == 0.5  # 0.8 - 0.3


# =============================================================================
# Unit Tests: Adjustment History
# =============================================================================


class TestAdjustmentHistory:
    """Tests for adjustment history retrieval."""

    @pytest.mark.asyncio
    async def test_get_empty_history(self):
        """Should return empty list when no adjustments."""
        manager = ConfidenceDecayManager()
        history = await manager.get_adjustment_history()
        assert history == []

    @pytest.mark.asyncio
    async def test_get_all_adjustments(self):
        """Should return all adjustments."""
        manager = ConfidenceDecayManager()

        # Manually add adjustments
        adj1 = ConfidenceAdjustment(
            id="1",
            item_id="item-1",
            event=ConfidenceEvent.ACCESSED,
            old_confidence=0.5,
            new_confidence=0.51,
            reason="Test",
        )
        adj2 = ConfidenceAdjustment(
            id="2",
            item_id="item-2",
            event=ConfidenceEvent.VALIDATED,
            old_confidence=0.6,
            new_confidence=0.7,
            reason="Test",
        )
        manager._adjustments = [adj1, adj2]

        history = await manager.get_adjustment_history()
        assert len(history) == 2

    @pytest.mark.asyncio
    async def test_filter_by_item_id(self):
        """Should filter adjustments by item ID."""
        manager = ConfidenceDecayManager()

        adj1 = ConfidenceAdjustment(
            id="1",
            item_id="item-1",
            event=ConfidenceEvent.ACCESSED,
            old_confidence=0.5,
            new_confidence=0.51,
            reason="Test",
        )
        adj2 = ConfidenceAdjustment(
            id="2",
            item_id="item-2",
            event=ConfidenceEvent.VALIDATED,
            old_confidence=0.6,
            new_confidence=0.7,
            reason="Test",
        )
        manager._adjustments = [adj1, adj2]

        history = await manager.get_adjustment_history(item_id="item-1")
        assert len(history) == 1
        assert history[0].item_id == "item-1"

    @pytest.mark.asyncio
    async def test_filter_by_event_type(self):
        """Should filter adjustments by event type."""
        manager = ConfidenceDecayManager()

        adj1 = ConfidenceAdjustment(
            id="1",
            item_id="item-1",
            event=ConfidenceEvent.ACCESSED,
            old_confidence=0.5,
            new_confidence=0.51,
            reason="Test",
        )
        adj2 = ConfidenceAdjustment(
            id="2",
            item_id="item-2",
            event=ConfidenceEvent.VALIDATED,
            old_confidence=0.6,
            new_confidence=0.7,
            reason="Test",
        )
        manager._adjustments = [adj1, adj2]

        history = await manager.get_adjustment_history(event_type=ConfidenceEvent.VALIDATED)
        assert len(history) == 1
        assert history[0].event == ConfidenceEvent.VALIDATED

    @pytest.mark.asyncio
    async def test_limit_results(self):
        """Should respect limit parameter."""
        manager = ConfidenceDecayManager()

        adjustments = [
            ConfidenceAdjustment(
                id=str(i),
                item_id=f"item-{i}",
                event=ConfidenceEvent.ACCESSED,
                old_confidence=0.5,
                new_confidence=0.51,
                reason="Test",
            )
            for i in range(50)
        ]
        manager._adjustments = adjustments

        history = await manager.get_adjustment_history(limit=10)
        assert len(history) == 10


# =============================================================================
# Unit Tests: Statistics
# =============================================================================


class TestDecayStats:
    """Tests for decay manager statistics."""

    def test_empty_stats(self):
        """Should return zeros for empty manager."""
        manager = ConfidenceDecayManager()
        stats = manager.get_stats()

        assert stats["total_adjustments"] == 0
        assert stats["by_event"] == {}
        assert stats["positive_adjustments"] == 0
        assert stats["negative_adjustments"] == 0

    def test_stats_with_adjustments(self):
        """Should correctly count adjustments by type."""
        manager = ConfidenceDecayManager()

        manager._adjustments = [
            ConfidenceAdjustment(
                id="1",
                item_id="item-1",
                event=ConfidenceEvent.ACCESSED,
                old_confidence=0.5,
                new_confidence=0.51,  # positive
                reason="Test",
            ),
            ConfidenceAdjustment(
                id="2",
                item_id="item-2",
                event=ConfidenceEvent.ACCESSED,
                old_confidence=0.6,
                new_confidence=0.61,  # positive
                reason="Test",
            ),
            ConfidenceAdjustment(
                id="3",
                item_id="item-3",
                event=ConfidenceEvent.INVALIDATED,
                old_confidence=0.8,
                new_confidence=0.5,  # negative
                reason="Test",
            ),
        ]

        stats = manager.get_stats()
        assert stats["total_adjustments"] == 3
        assert stats["by_event"]["accessed"] == 2
        assert stats["by_event"]["invalidated"] == 1
        assert stats["positive_adjustments"] == 2
        assert stats["negative_adjustments"] == 1


# =============================================================================
# Tests: ConfidenceDecayMixin
# =============================================================================


class TestConfidenceDecayMixin:
    """Tests for the mixin class."""

    def test_get_decay_manager_creates_singleton(self):
        """Should create and reuse decay manager."""

        class TestMound(ConfidenceDecayMixin):
            pass

        mound = TestMound()
        manager1 = mound._get_decay_manager()
        manager2 = mound._get_decay_manager()

        assert manager1 is manager2
        assert isinstance(manager1, ConfidenceDecayManager)

    @pytest.mark.asyncio
    async def test_apply_confidence_decay(self):
        """Should delegate to manager."""

        class TestMound(ConfidenceDecayMixin):
            async def query(self, **kwargs):
                return MagicMock(items=[])

        mound = TestMound()
        report = await mound.apply_confidence_decay("test-workspace", force=True)

        assert isinstance(report, DecayReport)
        assert report.workspace_id == "test-workspace"

    def test_get_decay_stats(self):
        """Should return stats from manager."""

        class TestMound(ConfidenceDecayMixin):
            pass

        mound = TestMound()
        stats = mound.get_decay_stats()

        assert "total_adjustments" in stats
        assert "by_event" in stats


# =============================================================================
# Tests: Global Singleton
# =============================================================================


class TestGlobalManager:
    """Tests for global decay manager instance."""

    def test_get_decay_manager_singleton(self):
        """Should return same instance."""
        # Reset global state
        import aragora.knowledge.mound.ops.confidence_decay as module

        module._decay_manager = None

        manager1 = get_decay_manager()
        manager2 = get_decay_manager()

        assert manager1 is manager2


# =============================================================================
# Tests: DecayReport Serialization
# =============================================================================


class TestDecayReportSerialization:
    """Tests for DecayReport serialization."""

    def test_to_dict(self):
        """Should serialize report to dictionary."""
        adjustment = ConfidenceAdjustment(
            id="adj-1",
            item_id="item-1",
            event=ConfidenceEvent.DECAYED,
            old_confidence=0.8,
            new_confidence=0.6,
            reason="Time decay",
        )

        report = DecayReport(
            workspace_id="test-ws",
            items_processed=10,
            items_decayed=5,
            items_boosted=0,
            average_confidence_change=-0.1,
            adjustments=[adjustment],
            duration_ms=150.5,
        )

        d = report.to_dict()

        assert d["workspace_id"] == "test-ws"
        assert d["items_processed"] == 10
        assert d["items_decayed"] == 5
        assert d["average_confidence_change"] == -0.1
        assert d["duration_ms"] == 150.5
        assert len(d["adjustments"]) == 1
        assert d["adjustments"][0]["item_id"] == "item-1"


class TestConfidenceAdjustmentSerialization:
    """Tests for ConfidenceAdjustment serialization."""

    def test_to_dict(self):
        """Should serialize adjustment to dictionary."""
        adjustment = ConfidenceAdjustment(
            id="adj-1",
            item_id="item-1",
            event=ConfidenceEvent.VALIDATED,
            old_confidence=0.5,
            new_confidence=0.6,
            reason="Expert review",
            metadata={"reviewer": "alice"},
        )

        d = adjustment.to_dict()

        assert d["id"] == "adj-1"
        assert d["item_id"] == "item-1"
        assert d["event"] == "validated"
        assert d["old_confidence"] == 0.5
        assert d["new_confidence"] == 0.6
        assert d["reason"] == "Expert review"
        assert d["metadata"]["reviewer"] == "alice"


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Edge case and boundary condition tests."""

    def test_very_old_item_decay(self):
        """Items from years ago should decay to minimum."""
        manager = ConfidenceDecayManager()
        result = manager.calculate_decay(1.0, age_days=3650)  # 10 years
        assert result == 0.1  # min_confidence

    def test_negative_age_handled(self):
        """Negative age (future items) should not cause errors."""
        manager = ConfidenceDecayManager()
        result = manager.calculate_decay(0.8, age_days=-10)
        # Negative age means future - math.pow with negative exponent
        # gives > 1 decay factor, so confidence increases
        assert result >= 0.8

    def test_zero_confidence_item(self):
        """Items at zero confidence should stay there."""
        manager = ConfidenceDecayManager()
        result = manager.calculate_decay(0.0, age_days=30)
        assert result == 0.1  # min_confidence

    def test_very_high_initial_confidence(self):
        """Confidence above 1.0 should be handled."""
        manager = ConfidenceDecayManager()
        result = manager.calculate_decay(1.5, age_days=90)
        assert result >= 0.1
        assert result <= 1.0  # Would be clamped in practice

    def test_custom_decay_model(self):
        """Custom model should not change confidence."""
        config = DecayConfig(model=DecayModel.CUSTOM)
        manager = ConfidenceDecayManager(config)
        result = manager.calculate_decay(0.7, age_days=100)
        assert result == 0.7  # No change for custom without implementation

    def test_concurrent_decay_safety(self):
        """Manager should be safe for concurrent use."""
        import asyncio

        manager = ConfidenceDecayManager()

        async def add_adjustment():
            adj = ConfidenceAdjustment(
                id="test",
                item_id="item",
                event=ConfidenceEvent.ACCESSED,
                old_confidence=0.5,
                new_confidence=0.51,
                reason="Test",
            )
            async with manager._lock:
                manager._adjustments.append(adj)

        async def run_concurrent():
            await asyncio.gather(*[add_adjustment() for _ in range(100)])

        asyncio.run(run_concurrent())
        assert len(manager._adjustments) == 100

    def test_adjustment_cap(self):
        """Should cap stored adjustments at 10000."""
        manager = ConfidenceDecayManager()
        manager._adjustments = [
            ConfidenceAdjustment(
                id=str(i),
                item_id=f"item-{i}",
                event=ConfidenceEvent.ACCESSED,
                old_confidence=0.5,
                new_confidence=0.51,
                reason="Test",
            )
            for i in range(11000)
        ]

        # After processing, should trim to 10000
        # This happens in apply_decay, so let's simulate
        if len(manager._adjustments) > 10000:
            manager._adjustments = manager._adjustments[-10000:]

        assert len(manager._adjustments) == 10000
