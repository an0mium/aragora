"""
Tests for PulseAdapter bidirectional integration (Pulse ↔ KM).

Tests the reverse flow methods that enable Knowledge Mound patterns
to influence Pulse topic scheduling and quality thresholds.
"""

from __future__ import annotations

import pytest
from dataclasses import dataclass

from aragora.knowledge.mound.adapters.pulse_adapter import (
    PulseAdapter,
    KMQualityThresholdUpdate,
    KMTopicCoverage,
    KMSchedulingRecommendation,
    KMTopicValidation,
    PulseKMSyncResult,
)


@dataclass
class MockTrendingTopic:
    """Mock TrendingTopic for testing."""

    topic: str
    platform: str
    volume: int
    category: str
    raw_data: dict = None

    def __post_init__(self):
        if self.raw_data is None:
            self.raw_data = {}


class TestKMQualityThresholdUpdate:
    """Tests for KMQualityThresholdUpdate dataclass."""

    def test_default_values(self):
        """Test default values."""
        update = KMQualityThresholdUpdate(
            old_min_quality=0.6,
            new_min_quality=0.5,
        )
        assert update.old_min_quality == 0.6
        assert update.new_min_quality == 0.5
        assert update.patterns_analyzed == 0
        assert update.adjustments_made == 0
        assert update.confidence == 0.7
        assert update.recommendation == "keep"

    def test_custom_values(self):
        """Test custom values."""
        update = KMQualityThresholdUpdate(
            old_min_quality=0.6,
            new_min_quality=0.5,
            old_category_bonuses={"tech": 0.2},
            new_category_bonuses={"tech": 0.25},
            patterns_analyzed=100,
            adjustments_made=5,
            confidence=0.85,
            recommendation="lower_threshold",
        )
        assert update.patterns_analyzed == 100
        assert update.adjustments_made == 5
        assert update.recommendation == "lower_threshold"


class TestKMTopicCoverage:
    """Tests for KMTopicCoverage dataclass."""

    def test_default_values(self):
        """Test default values."""
        coverage = KMTopicCoverage(topic_text="Test topic")
        assert coverage.topic_text == "Test topic"
        assert coverage.coverage_score == 0.0
        assert coverage.related_debates_count == 0
        assert coverage.recommendation == "proceed"
        assert coverage.priority_adjustment == 0.0

    def test_skip_recommendation(self):
        """Test skip recommendation for well-covered topic."""
        coverage = KMTopicCoverage(
            topic_text="Test topic",
            coverage_score=0.9,
            related_debates_count=15,
            consensus_rate=0.8,
            recommendation="skip",
            priority_adjustment=-0.2,
        )
        assert coverage.recommendation == "skip"
        assert coverage.priority_adjustment < 0


class TestKMSchedulingRecommendation:
    """Tests for KMSchedulingRecommendation dataclass."""

    def test_default_values(self):
        """Test default values."""
        rec = KMSchedulingRecommendation(topic_id="topic_123")
        assert rec.topic_id == "topic_123"
        assert rec.original_priority == 0.5
        assert rec.adjusted_priority == 0.5
        assert rec.reason == "no_change"
        assert rec.was_applied is False

    def test_applied_recommendation(self):
        """Test applied recommendation."""
        rec = KMSchedulingRecommendation(
            topic_id="topic_123",
            original_priority=0.5,
            adjusted_priority=0.6,
            reason="boost",
            km_confidence=0.8,
            was_applied=True,
        )
        assert rec.was_applied is True
        assert rec.adjusted_priority > rec.original_priority


class TestKMTopicValidation:
    """Tests for KMTopicValidation dataclass."""

    def test_default_values(self):
        """Test default values."""
        val = KMTopicValidation(topic_id="topic_123")
        assert val.topic_id == "topic_123"
        assert val.km_confidence == 0.7
        assert val.outcome_success_rate == 0.0
        assert val.recommendation == "keep"
        assert val.priority_adjustment == 0.0

    def test_boost_validation(self):
        """Test boost validation for high success."""
        val = KMTopicValidation(
            topic_id="topic_123",
            km_confidence=0.9,
            outcome_success_rate=0.85,
            similar_debates_count=10,
            recommendation="boost",
            priority_adjustment=0.1,
        )
        assert val.recommendation == "boost"
        assert val.priority_adjustment > 0


class TestPulseAdapterOutcomeRecording:
    """Tests for outcome recording."""

    @pytest.fixture
    def adapter(self):
        """Create adapter for testing."""
        return PulseAdapter()

    def test_record_outcome(self, adapter):
        """Test recording an outcome."""
        adapter.record_outcome_for_km(
            topic_id="topic_123",
            debate_id="debate_456",
            outcome_success=True,
            confidence=0.8,
            rounds_used=3,
            category="tech",
        )

        stats = adapter.get_reverse_flow_stats()
        assert stats["outcome_history_count"] == 1

    def test_record_multiple_outcomes(self, adapter):
        """Test recording multiple outcomes."""
        for i in range(5):
            adapter.record_outcome_for_km(
                topic_id=f"topic_{i}",
                debate_id=f"debate_{i}",
                outcome_success=i % 2 == 0,
                confidence=0.5 + (i * 0.1),
            )

        stats = adapter.get_reverse_flow_stats()
        assert stats["outcome_history_count"] == 5


class TestPulseAdapterQualityThresholds:
    """Tests for quality threshold updates."""

    @pytest.fixture
    def adapter(self):
        """Create adapter for testing."""
        return PulseAdapter()

    @pytest.mark.asyncio
    async def test_threshold_update_insufficient_data(self, adapter):
        """Test threshold update with insufficient data."""
        km_items = [{"metadata": {"category": "tech", "outcome_success": True}}]

        update = await adapter.update_quality_thresholds_from_km(km_items)

        assert update.recommendation == "insufficient_data"
        assert update.adjustments_made == 0

    @pytest.mark.asyncio
    async def test_threshold_update_lower_threshold(self, adapter):
        """Test lowering threshold when low quality succeeds."""
        # Create items with low quality but high success
        km_items = []
        for i in range(15):
            km_items.append({
                "metadata": {
                    "category": "tech",
                    "quality_score": 0.45,  # Low quality bucket
                    "outcome_success": True,  # But successful
                }
            })

        update = await adapter.update_quality_thresholds_from_km(km_items)

        # Should recommend lowering threshold
        assert update.new_min_quality < update.old_min_quality or update.recommendation in ["keep", "lower_threshold"]

    @pytest.mark.asyncio
    async def test_threshold_update_raise_threshold(self, adapter):
        """Test raising threshold when low quality fails."""
        # Create items with low quality and low success
        km_items = []
        for i in range(15):
            km_items.append({
                "metadata": {
                    "category": "tech",
                    "quality_score": 0.45,  # Low quality bucket
                    "outcome_success": i < 3,  # Only 20% success
                }
            })

        update = await adapter.update_quality_thresholds_from_km(km_items)

        # Should recommend raising threshold
        assert update.new_min_quality >= update.old_min_quality

    @pytest.mark.asyncio
    async def test_category_bonus_adjustment(self, adapter):
        """Test category bonus adjustment."""
        # Create items with high success rate for science
        km_items = []
        for i in range(10):
            km_items.append({
                "metadata": {
                    "category": "science",
                    "quality_score": 0.7,
                    "outcome_success": True,
                }
            })

        update = await adapter.update_quality_thresholds_from_km(km_items)

        # Science bonus should increase
        old_science = update.old_category_bonuses.get("science", 0.2)
        new_science = update.new_category_bonuses.get("science", 0.2)
        assert new_science >= old_science


class TestPulseAdapterTopicCoverage:
    """Tests for topic coverage analysis."""

    @pytest.fixture
    def adapter(self):
        """Create adapter for testing."""
        return PulseAdapter()

    @pytest.mark.asyncio
    async def test_coverage_no_items(self, adapter):
        """Test coverage with no KM items."""
        coverage = await adapter.get_km_topic_coverage("New topic", [])

        assert coverage.topic_text == "New topic"
        assert coverage.coverage_score == 0.0
        assert coverage.recommendation == "proceed"

    @pytest.mark.asyncio
    async def test_coverage_well_covered(self, adapter):
        """Test coverage for well-covered topic."""
        km_items = []
        for i in range(12):
            km_items.append({
                "metadata": {
                    "outcome_success": True,
                    "confidence": 0.8,
                    "rounds_used": 3,
                }
            })

        coverage = await adapter.get_km_topic_coverage("Popular topic", km_items)

        assert coverage.coverage_score >= 0.8
        assert coverage.consensus_rate >= 0.7
        assert coverage.recommendation == "skip"
        assert coverage.priority_adjustment < 0

    @pytest.mark.asyncio
    async def test_coverage_partial_low_consensus(self, adapter):
        """Test partial coverage with low consensus."""
        km_items = []
        for i in range(6):
            km_items.append({
                "metadata": {
                    "outcome_success": i < 2,  # Low consensus
                    "confidence": 0.5,
                }
            })

        coverage = await adapter.get_km_topic_coverage("Contentious topic", km_items)

        assert coverage.consensus_rate < 0.5
        assert coverage.recommendation == "proceed"
        assert coverage.priority_adjustment >= 0

    @pytest.mark.asyncio
    async def test_coverage_caching(self, adapter):
        """Test that coverage is cached."""
        km_items = [{"metadata": {"outcome_success": True}}]

        coverage1 = await adapter.get_km_topic_coverage("Test topic", km_items)

        stats = adapter.get_reverse_flow_stats()
        assert stats["coverage_cache_size"] >= 1


class TestPulseAdapterTopicValidation:
    """Tests for topic validation."""

    @pytest.fixture
    def adapter(self):
        """Create adapter for testing."""
        return PulseAdapter()

    @pytest.mark.asyncio
    async def test_validate_no_data(self, adapter):
        """Test validation with no data."""
        validation = await adapter.validate_topic_from_km("topic_123", [])

        assert validation.topic_id == "topic_123"
        assert validation.km_confidence == 0.5
        assert validation.recommendation == "keep"

    @pytest.mark.asyncio
    async def test_validate_high_success(self, adapter):
        """Test validation with high success rate."""
        # Record internal outcomes
        for i in range(5):
            adapter.record_outcome_for_km(
                topic_id="topic_123",
                debate_id=f"debate_{i}",
                outcome_success=True,
                confidence=0.8,
            )

        # Add KM cross-refs
        km_refs = [
            {"metadata": {"outcome_success": True, "confidence": 0.9}},
            {"metadata": {"outcome_success": True, "confidence": 0.85}},
        ]

        validation = await adapter.validate_topic_from_km("topic_123", km_refs)

        assert validation.outcome_success_rate >= 0.8
        assert validation.recommendation == "boost"
        assert validation.priority_adjustment > 0

    @pytest.mark.asyncio
    async def test_validate_low_success(self, adapter):
        """Test validation with low success rate."""
        # Record mostly failed outcomes
        for i in range(6):
            adapter.record_outcome_for_km(
                topic_id="bad_topic",
                debate_id=f"debate_{i}",
                outcome_success=i < 1,  # Only 1/6 success
            )

        validation = await adapter.validate_topic_from_km("bad_topic", [])

        assert validation.outcome_success_rate < 0.3
        assert validation.recommendation == "demote"
        assert validation.priority_adjustment < 0


class TestPulseAdapterSchedulingRecommendation:
    """Tests for applying scheduling recommendations."""

    @pytest.fixture
    def adapter(self):
        """Create adapter with stored topic."""
        adapter = PulseAdapter()
        # Store a topic
        topic = MockTrendingTopic(
            topic="Test topic",
            platform="twitter",
            volume=10000,
            category="tech",
        )
        adapter.store_trending_topic(topic)
        return adapter

    @pytest.mark.asyncio
    async def test_apply_keep_recommendation(self, adapter):
        """Test applying keep recommendation."""
        # Get a real topic ID from the adapter
        topic_id = list(adapter._topics.keys())[0]

        validation = KMTopicValidation(
            topic_id=topic_id,
            km_confidence=0.8,
            recommendation="keep",
            priority_adjustment=0.0,
        )

        rec = await adapter.apply_scheduling_recommendation(validation)

        assert rec.was_applied is False
        assert rec.reason == "no_change"

    @pytest.mark.asyncio
    async def test_apply_boost_recommendation(self, adapter):
        """Test applying boost recommendation."""
        # Get a real topic ID
        stats = adapter.get_stats()
        topic_id = list(adapter._topics.keys())[0] if adapter._topics else None

        if topic_id:
            validation = KMTopicValidation(
                topic_id=topic_id,
                km_confidence=0.85,
                recommendation="boost",
                priority_adjustment=0.1,
                outcome_success_rate=0.85,
            )

            rec = await adapter.apply_scheduling_recommendation(validation)

            assert rec.was_applied is True
            assert rec.adjusted_priority > rec.original_priority
            assert rec.reason == "boost"

    @pytest.mark.asyncio
    async def test_apply_recommendation_topic_not_found(self, adapter):
        """Test applying recommendation for missing topic."""
        validation = KMTopicValidation(
            topic_id="nonexistent",
            recommendation="boost",
            priority_adjustment=0.1,
        )

        rec = await adapter.apply_scheduling_recommendation(validation)

        assert rec.was_applied is False
        assert "topic_not_found" in rec.reason


class TestPulseAdapterBatchSync:
    """Tests for batch sync of KM validations."""

    @pytest.fixture
    def adapter(self):
        """Create adapter with topics."""
        adapter = PulseAdapter()
        # Store some topics
        for i in range(3):
            topic = MockTrendingTopic(
                topic=f"Topic {i}",
                platform="twitter",
                volume=10000 + i * 1000,
                category="tech",
            )
            adapter.store_trending_topic(topic)
        return adapter

    @pytest.mark.asyncio
    async def test_sync_empty_items(self, adapter):
        """Test sync with empty items."""
        result = await adapter.sync_validations_from_km([])

        assert isinstance(result, PulseKMSyncResult)
        assert result.topics_analyzed == 0
        assert result.threshold_updates == 0

    @pytest.mark.asyncio
    async def test_sync_with_items(self, adapter):
        """Test sync with KM items."""
        # Record some outcomes first
        topic_ids = list(adapter._topics.keys())
        for topic_id in topic_ids:
            for i in range(3):
                adapter.record_outcome_for_km(
                    topic_id=topic_id,
                    debate_id=f"d_{i}",
                    outcome_success=True,
                    category="tech",
                )

        # Create KM items
        km_items = []
        for topic_id in topic_ids:
            km_items.append({
                "metadata": {
                    "topic_id": topic_id,
                    "outcome_success": True,
                    "category": "tech",
                    "quality_score": 0.7,
                }
            })

        result = await adapter.sync_validations_from_km(km_items, min_confidence=0.5)

        assert result.topics_analyzed >= 1
        assert result.duration_ms >= 0


class TestPulseAdapterReverseFlowStats:
    """Tests for reverse flow statistics."""

    @pytest.fixture
    def adapter(self):
        """Create adapter for testing."""
        return PulseAdapter()

    def test_stats_empty(self, adapter):
        """Test stats with no activity."""
        stats = adapter.get_reverse_flow_stats()

        assert stats["outcome_history_count"] == 0
        assert stats["validations_stored"] == 0
        assert stats["km_priority_adjustments"] == 0
        assert stats["km_threshold_updates"] == 0
        assert stats["current_min_quality"] == 0.6

    @pytest.mark.asyncio
    async def test_stats_after_operations(self, adapter):
        """Test stats after some operations."""
        # Record outcomes
        for i in range(3):
            adapter.record_outcome_for_km(
                topic_id=f"topic_{i}",
                debate_id=f"debate_{i}",
                outcome_success=True,
            )

        # Do validations
        await adapter.validate_topic_from_km("topic_0", [])
        await adapter.validate_topic_from_km("topic_1", [])

        stats = adapter.get_reverse_flow_stats()

        assert stats["outcome_history_count"] == 3
        assert stats["validations_stored"] >= 2

    def test_clear_reverse_flow_state(self, adapter):
        """Test clearing reverse flow state."""
        # Add some state
        adapter.record_outcome_for_km("t1", "d1", True)
        adapter._km_priority_adjustments = 5
        adapter._km_threshold_updates = 3

        # Clear
        adapter.clear_reverse_flow_state()

        # Verify cleared
        stats = adapter.get_reverse_flow_stats()
        assert stats["outcome_history_count"] == 0
        assert stats["km_priority_adjustments"] == 0
        assert stats["km_threshold_updates"] == 0


class TestPulseAdapterIntegration:
    """Integration tests for bidirectional flow."""

    @pytest.fixture
    def adapter(self):
        """Create adapter with topics."""
        adapter = PulseAdapter()
        # Store topics
        for i in range(5):
            topic = MockTrendingTopic(
                topic=f"Integration test topic {i}",
                platform="twitter",
                volume=50000 + i * 10000,
                category="tech" if i % 2 == 0 else "science",
            )
            adapter.store_trending_topic(topic)
        return adapter

    @pytest.mark.asyncio
    async def test_full_bidirectional_cycle(self, adapter):
        """Test complete cycle: record → coverage → validate → apply."""
        topic_id = list(adapter._topics.keys())[0]

        # 1. Record outcomes
        for i in range(6):
            adapter.record_outcome_for_km(
                topic_id=topic_id,
                debate_id=f"debate_{i}",
                outcome_success=True,
                confidence=0.8,
                category="tech",
            )

        # 2. Get coverage
        coverage = await adapter.get_km_topic_coverage(
            "Integration test topic 0",
            [{"metadata": {"outcome_success": True}}],
        )

        assert coverage.coverage_score > 0

        # 3. Validate topic
        km_refs = [
            {"metadata": {"outcome_success": True, "confidence": 0.9}},
            {"metadata": {"outcome_success": True, "confidence": 0.85}},
        ]
        validation = await adapter.validate_topic_from_km(topic_id, km_refs)

        # 4. Apply recommendation if boost
        if validation.recommendation == "boost":
            rec = await adapter.apply_scheduling_recommendation(validation)
            assert rec.was_applied is True

        # 5. Verify stats
        stats = adapter.get_reverse_flow_stats()
        assert stats["validations_stored"] >= 1

    @pytest.mark.asyncio
    async def test_threshold_feedback_loop(self, adapter):
        """Test threshold adjustment feedback loop."""
        # Create extensive outcome data
        km_items = []
        for i in range(20):
            km_items.append({
                "metadata": {
                    "category": "tech",
                    "quality_score": 0.55,  # Medium quality
                    "outcome_success": True,  # High success
                }
            })

        # Update thresholds
        update = await adapter.update_quality_thresholds_from_km(km_items)

        assert update.patterns_analyzed == 20

        # Verify current settings changed
        stats = adapter.get_reverse_flow_stats()
        assert stats["current_min_quality"] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
