"""
Tests for InsightsAdapter bidirectional integration (Insights/Flip ↔ KM).

Tests the reverse flow methods that enable Knowledge Mound patterns
to influence flip detection thresholds and agent baselines.
"""

from __future__ import annotations

import pytest
from dataclasses import dataclass

from aragora.knowledge.mound.adapters.insights_adapter import (
    InsightsAdapter,
    KMFlipThresholdUpdate,
    KMAgentFlipBaseline,
    KMFlipValidation,
    InsightThresholdSyncResult,
)


@dataclass
class MockFlipEvent:
    """Mock FlipEvent for testing."""

    id: str
    agent_name: str
    original_claim: str
    new_claim: str
    original_confidence: float = 0.8
    new_confidence: float = 0.7
    original_debate_id: str = "debate_1"
    new_debate_id: str = "debate_2"
    original_position_id: str = "pos_1"
    new_position_id: str = "pos_2"
    similarity_score: float = 0.75
    flip_type: str = "contradiction"
    domain: str = "testing"
    detected_at: str = "2024-01-01T00:00:00"


@pytest.fixture
def adapter():
    """Create an InsightsAdapter for testing."""
    return InsightsAdapter()


@pytest.fixture
def adapter_with_flips():
    """Create an adapter with some flips stored."""
    adapter = InsightsAdapter()

    # Store some flips
    for i in range(5):
        flip = MockFlipEvent(
            id=f"flip_{i}",
            agent_name="claude",
            original_claim=f"Original claim {i}",
            new_claim=f"New claim {i}",
            similarity_score=0.7 + (i * 0.05),
            flip_type="contradiction" if i % 2 == 0 else "refinement",
            domain="testing" if i < 3 else "security",
        )
        adapter.store_flip(flip)

    return adapter


class TestKMFlipThresholdUpdate:
    """Tests for KMFlipThresholdUpdate dataclass."""

    def test_default_values(self):
        """Test default values."""
        update = KMFlipThresholdUpdate(
            old_similarity_threshold=0.7,
            new_similarity_threshold=0.65,
            old_confidence_threshold=0.6,
            new_confidence_threshold=0.55,
        )
        assert update.patterns_analyzed == 0
        assert update.adjustments_made == 0
        assert update.confidence == 0.7
        assert update.recommendation == "keep"

    def test_custom_values(self):
        """Test custom values."""
        update = KMFlipThresholdUpdate(
            old_similarity_threshold=0.7,
            new_similarity_threshold=0.55,
            old_confidence_threshold=0.6,
            new_confidence_threshold=0.5,
            patterns_analyzed=50,
            adjustments_made=1,
            confidence=0.9,
            recommendation="decrease",
        )
        assert update.patterns_analyzed == 50
        assert update.recommendation == "decrease"


class TestKMAgentFlipBaseline:
    """Tests for KMAgentFlipBaseline dataclass."""

    def test_default_values(self):
        """Test default values."""
        baseline = KMAgentFlipBaseline(
            agent_name="claude",
            expected_flip_rate=0.2,
        )
        assert baseline.flip_type_distribution == {}
        assert baseline.domain_flip_rates == {}
        assert baseline.sample_count == 0
        assert baseline.confidence == 0.7

    def test_with_distributions(self):
        """Test with flip type distributions."""
        baseline = KMAgentFlipBaseline(
            agent_name="gemini",
            expected_flip_rate=0.3,
            flip_type_distribution={"contradiction": 0.6, "refinement": 0.4},
            domain_flip_rates={"security": 0.2, "testing": 0.1},
            sample_count=15,
            confidence=0.75,
        )
        assert baseline.flip_type_distribution["contradiction"] == 0.6
        assert "security" in baseline.domain_flip_rates


class TestKMFlipValidation:
    """Tests for KMFlipValidation dataclass."""

    def test_default_values(self):
        """Test default values."""
        validation = KMFlipValidation(
            flip_id="fl_123",
            km_confidence=0.8,
        )
        assert validation.is_expected is False
        assert validation.pattern_match_score == 0.0
        assert validation.recommendation == "keep"
        assert validation.adjustment == 0.0

    def test_expected_flip(self):
        """Test validation for expected flip."""
        validation = KMFlipValidation(
            flip_id="fl_123",
            km_confidence=0.85,
            is_expected=True,
            pattern_match_score=0.6,
            recommendation="keep",
            adjustment=-0.02,
        )
        assert validation.is_expected is True
        assert validation.adjustment == -0.02


class TestInsightsAdapterThresholdUpdates:
    """Tests for threshold updates from KM patterns."""

    @pytest.mark.asyncio
    async def test_update_thresholds_empty_items(self, adapter):
        """Test threshold update with empty items."""
        update = await adapter.update_flip_thresholds_from_km([])

        assert update.patterns_analyzed == 0
        assert update.old_similarity_threshold == 0.7
        assert update.new_similarity_threshold == 0.7  # No change
        assert update.recommendation == "keep"

    @pytest.mark.asyncio
    async def test_update_thresholds_with_accuracy_data(self, adapter):
        """Test threshold update with accuracy data."""
        km_items = []

        # 0.7-0.8 bucket: 80% accuracy
        for i in range(10):
            km_items.append(
                {
                    "metadata": {
                        "similarity_score": 0.75,
                        "was_accurate": i < 8,  # 8/10 = 80%
                    }
                }
            )

        # 0.8-0.9 bucket: 90% accuracy
        for i in range(10):
            km_items.append(
                {
                    "metadata": {
                        "similarity_score": 0.85,
                        "was_accurate": i < 9,
                    }
                }
            )

        # Add more items for high confidence
        for _ in range(30):
            km_items.append(
                {
                    "metadata": {
                        "similarity_score": 0.9,
                        "was_accurate": True,
                    }
                }
            )

        update = await adapter.update_flip_thresholds_from_km(km_items, min_confidence=0.7)

        assert update.patterns_analyzed == 50
        assert update.confidence >= 0.7

    @pytest.mark.asyncio
    async def test_threshold_decrease_on_good_accuracy(self, adapter):
        """Test threshold decreases when lower thresholds have good accuracy."""
        km_items = []

        # 0.6-0.7 bucket: 80% accuracy (should enable lowering)
        for i in range(5):
            km_items.append(
                {
                    "metadata": {
                        "similarity_score": 0.65,
                        "was_accurate": i < 4,
                    }
                }
            )

        # 0.7-0.8 bucket: 75% accuracy
        for i in range(8):
            km_items.append(
                {
                    "metadata": {
                        "similarity_score": 0.75,
                        "was_accurate": i < 6,
                    }
                }
            )

        # Add more items for confidence
        for _ in range(37):
            km_items.append(
                {
                    "metadata": {
                        "similarity_score": 0.85,
                        "was_accurate": True,
                    }
                }
            )

        update = await adapter.update_flip_thresholds_from_km(km_items, min_confidence=0.7)

        assert update.confidence >= 0.7
        assert update.patterns_analyzed == 50


class TestInsightsAdapterAgentBaselines:
    """Tests for agent flip baselines."""

    @pytest.mark.asyncio
    async def test_get_baselines_empty(self, adapter):
        """Test getting baselines for unknown agent."""
        baseline = await adapter.get_agent_flip_baselines("unknown_agent")

        assert baseline.agent_name == "unknown_agent"
        assert baseline.expected_flip_rate == 0.0
        assert baseline.sample_count == 0

    @pytest.mark.asyncio
    async def test_get_baselines_with_flips(self, adapter_with_flips):
        """Test getting baselines from stored flips."""
        baseline = await adapter_with_flips.get_agent_flip_baselines("claude")

        assert baseline.agent_name == "claude"
        assert baseline.expected_flip_rate > 0
        assert baseline.sample_count == 5
        assert "contradiction" in baseline.flip_type_distribution
        assert "refinement" in baseline.flip_type_distribution

    @pytest.mark.asyncio
    async def test_baselines_cached(self, adapter_with_flips):
        """Test that baselines are cached."""
        baseline1 = await adapter_with_flips.get_agent_flip_baselines("claude")
        baseline2 = await adapter_with_flips.get_agent_flip_baselines("claude")

        assert baseline1.agent_name == baseline2.agent_name
        assert baseline1.expected_flip_rate == baseline2.expected_flip_rate

    @pytest.mark.asyncio
    async def test_get_baselines_with_km_items(self, adapter):
        """Test getting baselines with KM items."""
        km_items = [
            {"metadata": {"agent_name": "claude", "flip_type": "contradiction", "debate_id": "d1"}},
            {"metadata": {"agent_name": "claude", "flip_type": "contradiction", "debate_id": "d2"}},
            {"metadata": {"agent_name": "claude", "flip_type": "refinement", "debate_id": "d3"}},
            {"metadata": {"agent_name": "gemini", "flip_type": "contradiction", "debate_id": "d4"}},
        ]

        baseline = await adapter.get_agent_flip_baselines("claude", km_items)

        assert baseline.agent_name == "claude"
        assert baseline.sample_count == 3  # Only claude items


class TestInsightsAdapterFlipValidation:
    """Tests for flip validation from KM."""

    @pytest.mark.asyncio
    async def test_validate_unknown_flip(self, adapter):
        """Test validation of unknown flip."""
        validation = await adapter.validate_flip_from_km("unknown_flip", [])

        assert validation.recommendation == "keep"
        assert validation.km_confidence == 0.0
        assert "flip_not_found" in validation.metadata.get("error", "")

    @pytest.mark.asyncio
    async def test_validate_expected_flip(self, adapter_with_flips):
        """Test validation of expected flip."""
        # Store more flips to establish pattern
        for i in range(10):
            flip = MockFlipEvent(
                id=f"extra_flip_{i}",
                agent_name="claude",
                original_claim=f"Claim {i}",
                new_claim=f"New {i}",
                flip_type="contradiction",
                similarity_score=0.85,
            )
            adapter_with_flips.store_flip(flip)

        # Validate one of the flips
        patterns = [
            {"metadata": {"flip_type": "contradiction", "relationship": "supports"}},
            {"metadata": {"flip_type": "contradiction", "relationship": "supports"}},
        ]

        validation = await adapter_with_flips.validate_flip_from_km("fl_flip_0", patterns)

        assert validation.flip_id == "fl_flip_0"
        assert validation.is_expected is True  # Common flip type for this agent

    @pytest.mark.asyncio
    async def test_validate_unexpected_flip(self, adapter_with_flips):
        """Test validation of unexpected flip type."""
        # All stored flips are contradiction or refinement
        # Add a flip with different type
        unusual_flip = MockFlipEvent(
            id="unusual_flip",
            agent_name="claude",
            original_claim="Claim",
            new_claim="Opposite claim",
            flip_type="reversal",  # Unusual type
            similarity_score=0.8,
        )
        adapter_with_flips.store_flip(unusual_flip)

        validation = await adapter_with_flips.validate_flip_from_km(
            "fl_unusual_flip",
            [],  # No supporting patterns
        )

        # Reversal type is unexpected for this agent
        assert validation.flip_id == "fl_unusual_flip"

    @pytest.mark.asyncio
    async def test_validate_low_similarity_ignored(self, adapter):
        """Test that low similarity flips are ignored."""
        flip = MockFlipEvent(
            id="low_sim_flip",
            agent_name="claude",
            original_claim="Claim",
            new_claim="Different claim",
            similarity_score=0.5,  # Below threshold
        )
        adapter.store_flip(flip)

        validation = await adapter.validate_flip_from_km("fl_low_sim_flip", [])

        assert validation.recommendation == "ignore"
        assert validation.adjustment == 0.0


class TestInsightsAdapterApplyValidation:
    """Tests for applying KM validations."""

    @pytest.mark.asyncio
    async def test_apply_validation_success(self, adapter_with_flips):
        """Test successful validation application."""
        validation = KMFlipValidation(
            flip_id="fl_flip_0",
            km_confidence=0.85,
            is_expected=True,
            recommendation="keep",
            adjustment=-0.02,
        )

        result = await adapter_with_flips.apply_km_validation(validation)

        assert result is True

        # Check flip was updated
        flip = adapter_with_flips.get_flip("fl_flip_0")
        assert flip["km_validated"] is True
        assert flip["km_confidence"] == 0.85
        assert flip["km_is_expected"] is True

    @pytest.mark.asyncio
    async def test_apply_validation_unknown_flip(self, adapter):
        """Test applying validation to unknown flip."""
        validation = KMFlipValidation(
            flip_id="unknown",
            km_confidence=0.8,
            recommendation="keep",
        )

        result = await adapter.apply_km_validation(validation)
        assert result is False


class TestInsightsAdapterBatchSync:
    """Tests for batch sync of KM validations."""

    @pytest.mark.asyncio
    async def test_sync_empty_items(self, adapter):
        """Test sync with empty items."""
        result = await adapter.sync_validations_from_km([])

        assert isinstance(result, InsightThresholdSyncResult)
        assert result.flips_analyzed == 0
        assert result.insights_analyzed == 0

    @pytest.mark.asyncio
    async def test_sync_with_flip_items(self, adapter_with_flips):
        """Test sync with flip items."""
        km_items = [
            {
                "metadata": {
                    "flip_id": "fl_flip_0",
                    "flip_type": "contradiction",
                    "agent_name": "claude",
                    "similarity_score": 0.8,
                }
            },
            {
                "metadata": {
                    "flip_id": "fl_flip_1",
                    "flip_type": "refinement",
                    "agent_name": "claude",
                    "similarity_score": 0.75,
                }
            },
        ]

        result = await adapter_with_flips.sync_validations_from_km(km_items)

        assert result.flips_analyzed >= 2
        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_sync_updates_baselines(self, adapter):
        """Test that sync updates agent baselines."""
        km_items = [
            {
                "metadata": {
                    "agent_name": "claude",
                    "flip_type": "contradiction",
                    "similarity_score": 0.8,
                }
            },
            {
                "metadata": {
                    "agent_name": "gemini",
                    "flip_type": "refinement",
                    "similarity_score": 0.75,
                }
            },
        ]

        result = await adapter.sync_validations_from_km(km_items)

        # Should have baseline updates for both agents
        assert len(result.baseline_updates) >= 2


class TestInsightsAdapterOutcomeRecording:
    """Tests for outcome recording."""

    def test_record_outcome(self, adapter):
        """Test recording a flip detection outcome."""
        adapter.record_outcome("fl_123", "debate_1", True, 0.9)

        stats = adapter.get_reverse_flow_stats()
        assert stats["outcome_history_size"] == 1

    def test_record_multiple_outcomes(self, adapter):
        """Test recording multiple outcomes."""
        adapter.record_outcome("fl_123", "debate_1", True)
        adapter.record_outcome("fl_123", "debate_2", False)
        adapter.record_outcome("fl_456", "debate_3", True)

        stats = adapter.get_reverse_flow_stats()
        assert stats["outcome_history_size"] == 3


class TestInsightsAdapterStats:
    """Tests for reverse flow statistics."""

    def test_stats_empty(self, adapter):
        """Test stats with no activity."""
        stats = adapter.get_reverse_flow_stats()

        assert stats["km_validations_applied"] == 0
        assert stats["km_threshold_updates"] == 0
        assert stats["km_baselines_computed"] == 0

    @pytest.mark.asyncio
    async def test_stats_after_operations(self, adapter_with_flips):
        """Test stats after some operations."""
        # Do some validations
        await adapter_with_flips.validate_flip_from_km("fl_flip_0", [])
        await adapter_with_flips.validate_flip_from_km("fl_flip_1", [])

        # Get baselines
        await adapter_with_flips.get_agent_flip_baselines("claude")

        stats = adapter_with_flips.get_reverse_flow_stats()

        assert stats["km_validations_applied"] >= 2
        assert stats["km_baselines_computed"] >= 1
        assert stats["validations_stored"] >= 2

    def test_clear_reverse_flow_state(self, adapter_with_flips):
        """Test clearing reverse flow state."""
        # Add some state
        adapter_with_flips.record_outcome("fl_123", "d1", True)
        adapter_with_flips._km_validations_applied = 5
        adapter_with_flips._km_threshold_updates = 2

        # Clear
        adapter_with_flips.clear_reverse_flow_state()

        # Verify cleared
        stats = adapter_with_flips.get_reverse_flow_stats()
        assert stats["km_validations_applied"] == 0
        assert stats["km_threshold_updates"] == 0
        assert stats["outcome_history_size"] == 0

    def test_get_stats_includes_km_data(self, adapter):
        """Test that get_stats includes KM data."""
        stats = adapter.get_stats()

        assert "km_validations_applied" in stats
        assert "km_threshold_updates" in stats
        assert "km_baselines_computed" in stats


class TestInsightsAdapterIntegration:
    """Integration tests for bidirectional flow."""

    @pytest.mark.asyncio
    async def test_full_bidirectional_cycle(self, adapter):
        """Test complete bidirectional cycle: store → validate → update."""
        # 1. Store some flips
        for i in range(5):
            flip = MockFlipEvent(
                id=f"int_flip_{i}",
                agent_name="claude",
                original_claim=f"Claim {i}",
                new_claim=f"New claim {i}",
                similarity_score=0.75 + (i * 0.05),
                flip_type="contradiction" if i % 2 == 0 else "refinement",
            )
            adapter.store_flip(flip)

        # 2. Record outcomes
        adapter.record_outcome("fl_int_flip_0", "d1", True)
        adapter.record_outcome("fl_int_flip_1", "d2", True)
        adapter.record_outcome("fl_int_flip_2", "d3", False)

        # 3. Get baseline
        baseline = await adapter.get_agent_flip_baselines("claude")
        assert baseline.sample_count >= 5

        # 4. Validate flips
        validation = await adapter.validate_flip_from_km(
            "fl_int_flip_0",
            [{"metadata": {"flip_type": "contradiction"}}],
        )

        # 5. Apply validation
        await adapter.apply_km_validation(validation)

        # 6. Verify
        flip = adapter.get_flip("fl_int_flip_0")
        assert flip["km_validated"] is True

        stats = adapter.get_reverse_flow_stats()
        assert stats["km_validations_applied"] >= 1
        assert stats["km_baselines_computed"] >= 1

    @pytest.mark.asyncio
    async def test_threshold_adaptation(self, adapter):
        """Test that thresholds adapt based on accuracy patterns."""
        # Initialize reverse flow state by calling stats
        stats = adapter.get_reverse_flow_stats()
        original_threshold = stats["current_similarity_threshold"]

        # Generate items with accuracy data
        km_items = []
        for i in range(60):
            similarity = 0.55 + (i % 5) * 0.1  # 0.55, 0.65, 0.75, 0.85, 0.95
            km_items.append(
                {
                    "metadata": {
                        "similarity_score": similarity,
                        "was_accurate": similarity >= 0.65,  # Accurate above 0.65
                    }
                }
            )

        update = await adapter.update_flip_thresholds_from_km(km_items, min_confidence=0.5)

        assert update.patterns_analyzed == 60
        assert update.old_similarity_threshold == original_threshold


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
