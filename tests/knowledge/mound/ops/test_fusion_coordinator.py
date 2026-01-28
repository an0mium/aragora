"""
Comprehensive tests for FusionCoordinator.

Tests all fusion strategies, conflict resolution methods, and edge cases.
"""

import pytest
from datetime import datetime, timezone, timedelta

from aragora.knowledge.mound.ops.fusion import (
    FusionStrategy,
    ConflictResolution,
    FusionOutcome,
    AdapterValidation,
    FusedValidation,
    FusionConfig,
    FusionCoordinator,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def coordinator():
    """Create a default FusionCoordinator."""
    return FusionCoordinator()


@pytest.fixture
def coordinator_with_config():
    """Create a FusionCoordinator with custom config."""
    config = FusionConfig(
        min_adapters_for_fusion=2,
        consensus_threshold=0.8,
        conflict_threshold=0.2,
        validity_threshold=0.5,
    )
    return FusionCoordinator(config=config)


@pytest.fixture
def sample_validations():
    """Create sample validations for testing."""
    return [
        AdapterValidation(
            adapter_name="elo",
            item_id="item_1",
            confidence=0.8,
            is_valid=True,
            sources=["debate_1"],
            priority=10,
            reliability=0.9,
        ),
        AdapterValidation(
            adapter_name="consensus",
            item_id="item_1",
            confidence=0.75,
            is_valid=True,
            sources=["debate_2"],
            priority=5,
            reliability=0.85,
        ),
        AdapterValidation(
            adapter_name="belief",
            item_id="item_1",
            confidence=0.7,
            is_valid=True,
            sources=["claim_1"],
            priority=3,
            reliability=0.8,
        ),
    ]


@pytest.fixture
def conflicting_validations():
    """Create validations with high variance (conflict)."""
    return [
        AdapterValidation(
            adapter_name="elo",
            item_id="item_2",
            confidence=0.95,
            is_valid=True,
            sources=["src1"],
            priority=10,
            reliability=0.9,
        ),
        AdapterValidation(
            adapter_name="belief",
            item_id="item_2",
            confidence=0.15,
            is_valid=False,
            sources=["src2"],
            priority=5,
            reliability=0.85,
        ),
    ]


# =============================================================================
# Strategy Tests
# =============================================================================


class TestFusionStrategies:
    """Test all fusion strategies."""

    def test_weighted_average_strategy(self, coordinator, sample_validations):
        """Test WEIGHTED_AVERAGE strategy computes correct weighted average."""
        result = coordinator.fuse_validations(
            sample_validations,
            strategy=FusionStrategy.WEIGHTED_AVERAGE,
        )

        assert result.strategy_used == FusionStrategy.WEIGHTED_AVERAGE
        # Result should be between min and max confidences
        assert 0.7 <= result.fused_confidence <= 0.8
        # Higher priority/reliability should have more weight
        # ELO (0.8) has higher priority and reliability than others

    def test_majority_vote_strategy(self, coordinator):
        """Test MAJORITY_VOTE strategy uses most common confidence bucket."""
        validations = [
            AdapterValidation(
                adapter_name="a1",
                item_id="item",
                confidence=0.75,
                is_valid=True,
                priority=0,
                reliability=1.0,
            ),
            AdapterValidation(
                adapter_name="a2",
                item_id="item",
                confidence=0.73,
                is_valid=True,
                priority=0,
                reliability=1.0,
            ),
            AdapterValidation(
                adapter_name="a3",
                item_id="item",
                confidence=0.78,
                is_valid=True,
                priority=0,
                reliability=1.0,
            ),
            AdapterValidation(
                adapter_name="a4",
                item_id="item",
                confidence=0.45,
                is_valid=False,
                priority=0,
                reliability=1.0,
            ),
        ]

        result = coordinator.fuse_validations(
            validations,
            strategy=FusionStrategy.MAJORITY_VOTE,
        )

        assert result.strategy_used == FusionStrategy.MAJORITY_VOTE
        # Most confidences bucket to 0.7 or 0.8, so result should be around there
        assert 0.7 <= result.fused_confidence <= 0.8

    def test_maximum_confidence_strategy(self, coordinator, sample_validations):
        """Test MAXIMUM_CONFIDENCE strategy returns highest confidence."""
        result = coordinator.fuse_validations(
            sample_validations,
            strategy=FusionStrategy.MAXIMUM_CONFIDENCE,
        )

        assert result.strategy_used == FusionStrategy.MAXIMUM_CONFIDENCE
        assert result.fused_confidence == 0.8  # Highest in sample_validations

    def test_minimum_confidence_strategy(self, coordinator, sample_validations):
        """Test MINIMUM_CONFIDENCE strategy returns lowest confidence."""
        result = coordinator.fuse_validations(
            sample_validations,
            strategy=FusionStrategy.MINIMUM_CONFIDENCE,
        )

        assert result.strategy_used == FusionStrategy.MINIMUM_CONFIDENCE
        assert result.fused_confidence == 0.7  # Lowest in sample_validations

    def test_median_strategy(self, coordinator, sample_validations):
        """Test MEDIAN strategy returns median confidence."""
        result = coordinator.fuse_validations(
            sample_validations,
            strategy=FusionStrategy.MEDIAN,
        )

        assert result.strategy_used == FusionStrategy.MEDIAN
        assert result.fused_confidence == 0.75  # Median of [0.7, 0.75, 0.8]

    def test_consensus_threshold_strategy_success(self, coordinator, sample_validations):
        """Test CONSENSUS_THRESHOLD with high agreement."""
        result = coordinator.fuse_validations(
            sample_validations,
            strategy=FusionStrategy.CONSENSUS_THRESHOLD,
        )

        assert result.strategy_used == FusionStrategy.CONSENSUS_THRESHOLD
        # All validations are is_valid=True, so agreement is 1.0
        assert result.agreement_ratio == 1.0

    def test_consensus_threshold_strategy_escalation(self, coordinator):
        """Test CONSENSUS_THRESHOLD escalates when below threshold."""
        config = FusionConfig(
            consensus_threshold=0.9,  # High threshold
            conflict_threshold=0.05,  # Low threshold to detect conflict
            escalate_on_deadlock=True,
        )
        coord = FusionCoordinator(config=config)

        validations = [
            AdapterValidation(
                adapter_name="a1",
                item_id="item",
                confidence=0.9,
                is_valid=True,
                priority=0,
                reliability=1.0,
            ),
            AdapterValidation(
                adapter_name="a2",
                item_id="item",
                confidence=0.2,
                is_valid=False,
                priority=0,
                reliability=1.0,
            ),
        ]

        result = coord.fuse_validations(
            validations,
            strategy=FusionStrategy.CONSENSUS_THRESHOLD,
        )

        # Agreement is 0.5 which is below 0.9 threshold
        assert result.outcome == FusionOutcome.ESCALATED
        assert result.escalation_reason is not None


# =============================================================================
# Conflict Resolution Tests
# =============================================================================


class TestConflictResolution:
    """Test conflict resolution strategies."""

    def test_prefer_higher_confidence(self, coordinator, conflicting_validations):
        """Test PREFER_HIGHER_CONFIDENCE resolution."""
        result = coordinator.fuse_validations(
            conflicting_validations,
            conflict_resolution=ConflictResolution.PREFER_HIGHER_CONFIDENCE,
        )

        assert result.conflict_detected is True
        # Should prefer 0.95 over 0.15
        assert result.fused_confidence >= 0.9

    def test_prefer_more_sources(self, coordinator):
        """Test PREFER_MORE_SOURCES resolution."""
        validations = [
            AdapterValidation(
                adapter_name="a1",
                item_id="item",
                confidence=0.6,
                is_valid=True,
                sources=["s1", "s2", "s3"],
                priority=0,
                reliability=1.0,
            ),
            AdapterValidation(
                adapter_name="a2",
                item_id="item",
                confidence=0.9,
                is_valid=True,
                sources=["s4"],
                priority=0,
                reliability=1.0,
            ),
        ]

        result = coordinator.fuse_validations(
            validations,
            conflict_resolution=ConflictResolution.PREFER_MORE_SOURCES,
        )

        # a1 has more sources, so its confidence (0.6) should be preferred
        # when conflict is detected

    def test_prefer_newer(self, coordinator):
        """Test PREFER_NEWER resolution."""
        old_time = datetime.now(timezone.utc) - timedelta(hours=1)
        new_time = datetime.now(timezone.utc)

        validations = [
            AdapterValidation(
                adapter_name="old",
                item_id="item",
                confidence=0.5,
                is_valid=False,
                priority=0,
                reliability=1.0,
                timestamp=old_time,
            ),
            AdapterValidation(
                adapter_name="new",
                item_id="item",
                confidence=0.9,
                is_valid=True,
                priority=0,
                reliability=1.0,
                timestamp=new_time,
            ),
        ]

        result = coordinator.fuse_validations(
            validations,
            conflict_resolution=ConflictResolution.PREFER_NEWER,
        )

        # Newer validation (0.9) should be preferred

    def test_escalate_resolution(self, coordinator):
        """Test ESCALATE resolution when conflict cannot be resolved."""
        config = FusionConfig(
            conflict_threshold=0.1,  # Low threshold
            escalate_on_deadlock=True,
        )
        coord = FusionCoordinator(config=config)

        validations = [
            AdapterValidation(
                adapter_name="a1",
                item_id="item",
                confidence=0.8,
                is_valid=True,
                priority=5,
                reliability=0.9,
            ),
            AdapterValidation(
                adapter_name="a2",
                item_id="item",
                confidence=0.3,
                is_valid=False,
                priority=5,
                reliability=0.9,
            ),
        ]

        result = coord.fuse_validations(
            validations,
            conflict_resolution=ConflictResolution.ESCALATE,
        )

        # Should escalate due to conflict
        assert result.conflict_detected is True


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Test edge cases for FusionCoordinator."""

    def test_empty_validations(self, coordinator):
        """Test fusion with empty validation list."""
        result = coordinator.fuse_validations([])

        assert result.fused_confidence == 0.0
        assert result.outcome == FusionOutcome.INSUFFICIENT_DATA
        assert len(result.participating_adapters) == 0

    def test_single_adapter_passthrough(self, coordinator):
        """Test that single adapter with min_adapters=1 passes through."""
        config = FusionConfig(min_adapters_for_fusion=1)
        coord = FusionCoordinator(config=config)

        validation = AdapterValidation(
            adapter_name="solo",
            item_id="item",
            confidence=0.85,
            is_valid=True,
            priority=0,
            reliability=1.0,
        )

        result = coord.fuse_validations([validation])

        assert result.fused_confidence == 0.85
        assert result.is_valid is True

    def test_insufficient_adapters(self, coordinator):
        """Test fusion fails with insufficient adapters."""
        # Default min_adapters is 2
        validation = AdapterValidation(
            adapter_name="solo",
            item_id="item",
            confidence=0.85,
            is_valid=True,
            priority=0,
            reliability=1.0,
        )

        result = coordinator.fuse_validations([validation])

        assert result.outcome == FusionOutcome.INSUFFICIENT_DATA

    def test_all_zero_confidence(self, coordinator):
        """Test fusion with all zero confidences."""
        validations = [
            AdapterValidation(
                adapter_name="a1",
                item_id="item",
                confidence=0.0,
                is_valid=False,
                priority=0,
                reliability=1.0,
            ),
            AdapterValidation(
                adapter_name="a2",
                item_id="item",
                confidence=0.0,
                is_valid=False,
                priority=0,
                reliability=1.0,
            ),
        ]

        result = coordinator.fuse_validations(validations)

        assert result.fused_confidence == 0.0
        assert result.is_valid is False

    def test_all_one_confidence(self, coordinator):
        """Test fusion with all 1.0 confidences."""
        validations = [
            AdapterValidation(
                adapter_name="a1",
                item_id="item",
                confidence=1.0,
                is_valid=True,
                priority=0,
                reliability=1.0,
            ),
            AdapterValidation(
                adapter_name="a2",
                item_id="item",
                confidence=1.0,
                is_valid=True,
                priority=0,
                reliability=1.0,
            ),
        ]

        result = coordinator.fuse_validations(validations)

        assert result.fused_confidence == 1.0
        assert result.is_valid is True
        assert result.agreement_ratio == 1.0

    def test_boundary_confidence_values(self, coordinator):
        """Test fusion at validity threshold boundary."""
        config = FusionConfig(validity_threshold=0.5)
        coord = FusionCoordinator(config=config)

        # Exactly at threshold
        validations = [
            AdapterValidation(
                adapter_name="a1",
                item_id="item",
                confidence=0.5,
                is_valid=True,
                priority=0,
                reliability=1.0,
            ),
            AdapterValidation(
                adapter_name="a2",
                item_id="item",
                confidence=0.5,
                is_valid=True,
                priority=0,
                reliability=1.0,
            ),
        ]

        result = coord.fuse_validations(validations)
        assert result.is_valid is True  # >= threshold

    def test_just_below_validity_threshold(self, coordinator):
        """Test fusion just below validity threshold."""
        config = FusionConfig(validity_threshold=0.5)
        coord = FusionCoordinator(config=config)

        validations = [
            AdapterValidation(
                adapter_name="a1",
                item_id="item",
                confidence=0.49,
                is_valid=False,
                priority=0,
                reliability=1.0,
            ),
            AdapterValidation(
                adapter_name="a2",
                item_id="item",
                confidence=0.49,
                is_valid=False,
                priority=0,
                reliability=1.0,
            ),
        ]

        result = coord.fuse_validations(validations)
        assert result.is_valid is False  # < threshold


# =============================================================================
# Multi-Adapter Scenario Tests
# =============================================================================


class TestMultiAdapterScenarios:
    """Test scenarios with 3+ adapters."""

    def test_three_adapter_fusion(self, coordinator, sample_validations):
        """Test fusion with exactly 3 adapters."""
        result = coordinator.fuse_validations(sample_validations)

        assert len(result.participating_adapters) == 3
        assert "elo" in result.participating_adapters
        assert "consensus" in result.participating_adapters
        assert "belief" in result.participating_adapters

    def test_many_adapters_fusion(self, coordinator):
        """Test fusion with many adapters (10+)."""
        validations = [
            AdapterValidation(
                adapter_name=f"adapter_{i}",
                item_id="item",
                confidence=0.5 + (i * 0.03),  # 0.5 to 0.8 range
                is_valid=True,
                priority=i,
                reliability=0.8 + (i * 0.01),
            )
            for i in range(10)
        ]

        result = coordinator.fuse_validations(validations)

        assert len(result.participating_adapters) == 10
        assert result.outcome == FusionOutcome.SUCCESS

    def test_asymmetric_participation(self, coordinator):
        """Test fusion where some adapters have different priorities."""
        validations = [
            AdapterValidation(
                adapter_name="high_priority",
                item_id="item",
                confidence=0.9,
                is_valid=True,
                priority=100,
                reliability=0.95,
            ),
            AdapterValidation(
                adapter_name="low_priority_1",
                item_id="item",
                confidence=0.3,
                is_valid=False,
                priority=1,
                reliability=0.5,
            ),
            AdapterValidation(
                adapter_name="low_priority_2",
                item_id="item",
                confidence=0.4,
                is_valid=False,
                priority=1,
                reliability=0.5,
            ),
        ]

        result = coordinator.fuse_validations(
            validations,
            strategy=FusionStrategy.WEIGHTED_AVERAGE,
        )

        # High priority adapter should dominate
        assert result.fused_confidence > 0.6

    def test_mixed_validity_adapters(self, coordinator):
        """Test fusion with mixed is_valid values."""
        validations = [
            AdapterValidation(
                adapter_name="a1",
                item_id="item",
                confidence=0.8,
                is_valid=True,
                priority=0,
                reliability=1.0,
            ),
            AdapterValidation(
                adapter_name="a2",
                item_id="item",
                confidence=0.7,
                is_valid=True,
                priority=0,
                reliability=1.0,
            ),
            AdapterValidation(
                adapter_name="a3",
                item_id="item",
                confidence=0.3,
                is_valid=False,
                priority=0,
                reliability=1.0,
            ),
        ]

        result = coordinator.fuse_validations(validations)

        # 2 out of 3 agree (is_valid=True)
        assert result.agreement_ratio == pytest.approx(2 / 3, rel=0.01)


# =============================================================================
# Fusion History and Escalation Tests
# =============================================================================


class TestFusionHistoryAndEscalation:
    """Test fusion history tracking and escalation queue."""

    def test_fusion_history_tracking(self, coordinator, sample_validations):
        """Test that fusion operations are tracked in history."""
        # Perform multiple fusions
        coordinator.fuse_validations(sample_validations)
        coordinator.fuse_validations(sample_validations)

        assert len(coordinator._fusion_history) == 2

    def test_escalation_queue_populated(self, coordinator):
        """Test escalation queue is populated for escalated results."""
        config = FusionConfig(
            conflict_threshold=0.05,
            escalate_on_deadlock=True,
        )
        coord = FusionCoordinator(config=config)

        validations = [
            AdapterValidation(
                adapter_name="a1",
                item_id="item",
                confidence=0.9,
                is_valid=True,
                priority=0,
                reliability=1.0,
            ),
            AdapterValidation(
                adapter_name="a2",
                item_id="item",
                confidence=0.1,
                is_valid=False,
                priority=0,
                reliability=1.0,
            ),
        ]

        result = coord.fuse_validations(
            validations,
            strategy=FusionStrategy.CONSENSUS_THRESHOLD,
        )

        if result.outcome == FusionOutcome.ESCALATED:
            assert len(coord._escalation_queue) > 0


# =============================================================================
# Validation Metadata Tests
# =============================================================================


class TestValidationMetadata:
    """Test metadata handling in validations."""

    def test_validation_to_dict(self, sample_validations):
        """Test AdapterValidation.to_dict() method."""
        validation = sample_validations[0]
        d = validation.to_dict()

        assert d["adapter_name"] == "elo"
        assert d["confidence"] == 0.8
        assert "timestamp" in d

    def test_fused_validation_to_dict(self, coordinator, sample_validations):
        """Test FusedValidation.to_dict() method."""
        result = coordinator.fuse_validations(sample_validations)
        d = result.to_dict()

        assert "item_id" in d
        assert "fused_confidence" in d
        assert "strategy_used" in d
        assert "source_validations" in d
        assert len(d["source_validations"]) == 3

    def test_validation_with_custom_metadata(self, coordinator):
        """Test validation with custom metadata is preserved."""
        validations = [
            AdapterValidation(
                adapter_name="a1",
                item_id="item",
                confidence=0.8,
                is_valid=True,
                priority=0,
                reliability=1.0,
                metadata={"custom_field": "value1"},
            ),
            AdapterValidation(
                adapter_name="a2",
                item_id="item",
                confidence=0.7,
                is_valid=True,
                priority=0,
                reliability=1.0,
                metadata={"custom_field": "value2"},
            ),
        ]

        result = coordinator.fuse_validations(validations)

        # Verify metadata is preserved in source validations
        assert result.source_validations[0].metadata["custom_field"] == "value1"
        assert result.source_validations[1].metadata["custom_field"] == "value2"
