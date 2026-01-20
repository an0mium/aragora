"""
Tests for KMOutcomeBridge bidirectional integration (Outcome ↔ KM).

Tests the bridge that validates Knowledge Mound entries based on
debate outcomes.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass

from aragora.debate.km_outcome_bridge import (
    KMOutcomeBridge,
    KMOutcomeBridgeConfig,
    OutcomeValidation,
    PropagationResult,
)


@dataclass
class MockConsensusOutcome:
    """Mock ConsensusOutcome for testing."""
    debate_id: str
    consensus_text: str
    consensus_confidence: float
    implementation_attempted: bool
    implementation_succeeded: bool


@pytest.fixture
def bridge():
    """Create a KMOutcomeBridge for testing."""
    return KMOutcomeBridge()


@pytest.fixture
def bridge_with_mocks():
    """Create a bridge with mock tracker and mound."""
    mock_tracker = MagicMock()
    mock_mound = MagicMock()

    # Mock async methods
    mock_mound.get = AsyncMock(return_value={"id": "km_123", "confidence": 0.7})
    mock_mound.update_confidence = AsyncMock(return_value=True)
    mock_mound.get_relationships = AsyncMock(return_value=[])

    return KMOutcomeBridge(
        outcome_tracker=mock_tracker,
        knowledge_mound=mock_mound,
    )


class TestOutcomeValidation:
    """Tests for OutcomeValidation dataclass."""

    def test_default_values(self):
        """Test default values."""
        validation = OutcomeValidation(
            km_item_id="km_123",
            debate_id="debate_456",
            was_successful=True,
            confidence_adjustment=0.1,
            validation_reason="test",
        )
        assert validation.km_item_id == "km_123"
        assert validation.was_successful is True
        assert validation.original_confidence == 0.0
        assert validation.propagated_from is None

    def test_with_propagation(self):
        """Test validation from propagation."""
        validation = OutcomeValidation(
            km_item_id="km_456",
            debate_id="debate_789",
            was_successful=True,
            confidence_adjustment=0.05,
            validation_reason="Propagated",
            propagated_from="km_123",
        )
        assert validation.propagated_from == "km_123"


class TestKMOutcomeBridgeConfig:
    """Tests for configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = KMOutcomeBridgeConfig()
        assert config.success_boost == 0.1
        assert config.failure_penalty == 0.05
        assert config.propagation_decay == 0.5
        assert config.max_propagation_depth == 3
        assert config.auto_propagate is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = KMOutcomeBridgeConfig(
            success_boost=0.2,
            failure_penalty=0.1,
            auto_propagate=False,
        )
        assert config.success_boost == 0.2
        assert config.auto_propagate is False


class TestKMOutcomeBridgeTracking:
    """Tests for KM usage tracking."""

    def test_record_km_usage(self, bridge):
        """Test recording KM usage in debates."""
        bridge.record_km_usage("debate_1", ["km_a", "km_b", "km_c"])

        usage = bridge.get_km_usage("debate_1")
        assert len(usage) == 3
        assert "km_a" in usage
        assert "km_b" in usage
        assert "km_c" in usage

    def test_record_km_usage_deduplicates(self, bridge):
        """Test that duplicate KM IDs are deduplicated."""
        bridge.record_km_usage("debate_1", ["km_a", "km_b"])
        bridge.record_km_usage("debate_1", ["km_b", "km_c"])

        usage = bridge.get_km_usage("debate_1")
        assert len(usage) == 3  # km_a, km_b, km_c (deduplicated)

    def test_get_usage_empty(self, bridge):
        """Test getting usage for unknown debate."""
        usage = bridge.get_km_usage("unknown_debate")
        assert usage == []

    def test_tracking_disabled(self):
        """Test that tracking can be disabled."""
        config = KMOutcomeBridgeConfig(track_usage=False)
        bridge = KMOutcomeBridge(config=config)

        bridge.record_km_usage("debate_1", ["km_a", "km_b"])

        usage = bridge.get_km_usage("debate_1")
        assert usage == []


class TestKMOutcomeBridgeValidation:
    """Tests for outcome-based validation."""

    @pytest.mark.asyncio
    async def test_validate_successful_outcome(self, bridge_with_mocks):
        """Test validation for successful outcome."""
        bridge_with_mocks.record_km_usage("debate_1", ["km_123"])

        outcome = MockConsensusOutcome(
            debate_id="debate_1",
            consensus_text="Test consensus",
            consensus_confidence=0.85,
            implementation_attempted=True,
            implementation_succeeded=True,
        )

        validations = await bridge_with_mocks.validate_knowledge_from_outcome(outcome)

        assert len(validations) == 1
        assert validations[0].was_successful is True
        assert validations[0].confidence_adjustment > 0  # Boost for success

    @pytest.mark.asyncio
    async def test_validate_failed_outcome(self, bridge_with_mocks):
        """Test validation for failed outcome."""
        bridge_with_mocks.record_km_usage("debate_2", ["km_456"])

        outcome = MockConsensusOutcome(
            debate_id="debate_2",
            consensus_text="Test consensus",
            consensus_confidence=0.7,
            implementation_attempted=True,
            implementation_succeeded=False,
        )

        validations = await bridge_with_mocks.validate_knowledge_from_outcome(outcome)

        assert len(validations) == 1
        assert validations[0].was_successful is False
        assert validations[0].confidence_adjustment < 0  # Penalty for failure

    @pytest.mark.asyncio
    async def test_validate_with_explicit_item_ids(self, bridge_with_mocks):
        """Test validation with explicit KM item IDs."""
        outcome = MockConsensusOutcome(
            debate_id="debate_3",
            consensus_text="Test",
            consensus_confidence=0.9,
            implementation_attempted=True,
            implementation_succeeded=True,
        )

        validations = await bridge_with_mocks.validate_knowledge_from_outcome(
            outcome,
            km_item_ids=["km_a", "km_b"],
        )

        # Should validate both explicitly provided items
        assert len(validations) == 2

    @pytest.mark.asyncio
    async def test_validate_no_km_items(self, bridge_with_mocks):
        """Test validation with no KM items to validate."""
        outcome = MockConsensusOutcome(
            debate_id="debate_no_km",
            consensus_text="Test",
            consensus_confidence=0.8,
            implementation_attempted=True,
            implementation_succeeded=True,
        )

        validations = await bridge_with_mocks.validate_knowledge_from_outcome(outcome)

        assert validations == []

    @pytest.mark.asyncio
    async def test_validate_no_mound(self, bridge):
        """Test validation without knowledge mound configured."""
        bridge.record_km_usage("debate_1", ["km_123"])

        outcome = MockConsensusOutcome(
            debate_id="debate_1",
            consensus_text="Test",
            consensus_confidence=0.8,
            implementation_attempted=True,
            implementation_succeeded=True,
        )

        validations = await bridge.validate_knowledge_from_outcome(outcome)

        assert validations == []

    @pytest.mark.asyncio
    async def test_validation_confidence_adjustment_calculation(self, bridge_with_mocks):
        """Test confidence adjustment calculation."""
        config = KMOutcomeBridgeConfig(
            success_boost=0.15,
            failure_penalty=0.08,
            auto_propagate=False,
        )
        bridge = KMOutcomeBridge(
            knowledge_mound=bridge_with_mocks._knowledge_mound,
            config=config,
        )
        bridge.record_km_usage("debate_1", ["km_123"])

        # High confidence success
        outcome = MockConsensusOutcome(
            debate_id="debate_1",
            consensus_text="Test",
            consensus_confidence=0.9,
            implementation_attempted=True,
            implementation_succeeded=True,
        )

        validations = await bridge.validate_knowledge_from_outcome(outcome)

        # Adjustment should be success_boost * confidence = 0.15 * 0.9 = 0.135
        assert len(validations) == 1
        assert 0.13 <= validations[0].confidence_adjustment <= 0.14


class TestKMOutcomeBridgePropagation:
    """Tests for validation propagation."""

    @pytest.mark.asyncio
    async def test_propagate_validation_no_mound(self, bridge):
        """Test propagation without knowledge mound."""
        validation = OutcomeValidation(
            km_item_id="km_123",
            debate_id="debate_1",
            was_successful=True,
            confidence_adjustment=0.1,
            validation_reason="test",
        )

        result = await bridge.propagate_validation(
            km_item_id="km_123",
            validation=validation,
            depth=2,
        )

        assert isinstance(result, PropagationResult)
        assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_propagate_validation_with_relationships(self):
        """Test propagation with related items."""
        mock_mound = MagicMock()
        mock_mound.get = AsyncMock(return_value={"id": "km_456", "confidence": 0.7})
        mock_mound.update_confidence = AsyncMock(return_value=True)
        mock_mound.get_relationships = AsyncMock(return_value=[
            {"target_id": "km_456"},
            {"target_id": "km_789"},
        ])

        bridge = KMOutcomeBridge(knowledge_mound=mock_mound)

        validation = OutcomeValidation(
            km_item_id="km_123",
            debate_id="debate_1",
            was_successful=True,
            confidence_adjustment=0.1,
            validation_reason="test",
        )

        result = await bridge.propagate_validation(
            km_item_id="km_123",
            validation=validation,
            depth=1,
        )

        assert result.items_updated >= 0
        assert result.root_item_id == "km_123"

    @pytest.mark.asyncio
    async def test_propagation_decay(self):
        """Test that propagation decays with depth."""
        mock_mound = MagicMock()
        mock_mound.get = AsyncMock(return_value={"id": "test", "confidence": 0.5})
        mock_mound.update_confidence = AsyncMock(return_value=True)
        mock_mound.get_relationships = AsyncMock(return_value=[
            {"target_id": "km_related"},
        ])

        config = KMOutcomeBridgeConfig(propagation_decay=0.5)
        bridge = KMOutcomeBridge(knowledge_mound=mock_mound, config=config)

        validation = OutcomeValidation(
            km_item_id="km_123",
            debate_id="debate_1",
            was_successful=True,
            confidence_adjustment=0.1,  # Original adjustment
            validation_reason="test",
        )

        result = await bridge.propagate_validation(
            km_item_id="km_123",
            validation=validation,
            depth=2,
        )

        # Propagated validations should have decayed adjustments
        for v in result.validations:
            # At depth 1, adjustment should be 0.1 * 0.5 = 0.05
            assert abs(v.confidence_adjustment) <= abs(validation.confidence_adjustment)


class TestKMOutcomeBridgeStats:
    """Tests for statistics."""

    def test_get_validation_stats_empty(self, bridge):
        """Test stats with no validations."""
        stats = bridge.get_validation_stats()

        assert stats["total_validations"] == 0
        assert stats["success_validations"] == 0
        assert stats["failure_validations"] == 0
        assert stats["debates_tracked"] == 0

    @pytest.mark.asyncio
    async def test_get_validation_stats_with_data(self, bridge_with_mocks):
        """Test stats after some validations."""
        bridge_with_mocks.record_km_usage("debate_1", ["km_123"])
        bridge_with_mocks.record_km_usage("debate_2", ["km_456"])

        # Success
        outcome1 = MockConsensusOutcome(
            debate_id="debate_1",
            consensus_text="Test",
            consensus_confidence=0.9,
            implementation_attempted=True,
            implementation_succeeded=True,
        )
        await bridge_with_mocks.validate_knowledge_from_outcome(outcome1)

        # Failure
        outcome2 = MockConsensusOutcome(
            debate_id="debate_2",
            consensus_text="Test",
            consensus_confidence=0.7,
            implementation_attempted=True,
            implementation_succeeded=False,
        )
        await bridge_with_mocks.validate_knowledge_from_outcome(outcome2)

        stats = bridge_with_mocks.get_validation_stats()

        assert stats["total_validations"] >= 2
        assert stats["success_validations"] >= 1
        assert stats["failure_validations"] >= 1
        assert stats["debates_tracked"] >= 2

    def test_clear_tracking(self, bridge):
        """Test clearing tracking data."""
        bridge.record_km_usage("debate_1", ["km_a", "km_b"])
        bridge._validations_applied.append(
            OutcomeValidation(
                km_item_id="km_a",
                debate_id="debate_1",
                was_successful=True,
                confidence_adjustment=0.1,
                validation_reason="test",
            )
        )
        bridge._total_validations = 1

        bridge.clear_tracking()

        assert bridge.get_km_usage("debate_1") == []
        assert len(bridge._validations_applied) == 0
        assert bridge._total_validations == 0


class TestKMOutcomeBridgeSetters:
    """Tests for setter methods."""

    def test_set_outcome_tracker(self, bridge):
        """Test setting outcome tracker."""
        mock_tracker = MagicMock()
        bridge.set_outcome_tracker(mock_tracker)
        assert bridge.outcome_tracker is mock_tracker

    def test_set_knowledge_mound(self, bridge):
        """Test setting knowledge mound."""
        mock_mound = MagicMock()
        bridge.set_knowledge_mound(mock_mound)
        assert bridge.knowledge_mound is mock_mound


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
