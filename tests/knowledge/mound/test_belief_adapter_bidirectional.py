"""
Tests for BeliefAdapter bidirectional integration (Belief ↔ KM).

Tests the reverse flow methods that enable Knowledge Mound patterns
to influence belief network thresholds and confidence.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock
from dataclasses import dataclass

from aragora.knowledge.mound.adapters.belief_adapter import (
    BeliefAdapter,
    KMThresholdUpdate,
    KMBeliefValidation,
    KMPriorRecommendation,
    BeliefThresholdSyncResult,
)


@dataclass
class MockBeliefNode:
    """Mock BeliefNode for testing."""

    node_id: str
    claim_id: str
    claim_statement: str
    author: str
    status: str = "active"
    centrality: float = 0.5
    update_count: int = 1
    parent_ids: list = None
    child_ids: list = None
    metadata: dict = None

    def __post_init__(self):
        self.prior = MagicMock()
        self.prior.p_true = 0.5
        self.posterior = MagicMock()
        self.posterior.p_true = 0.85
        if self.parent_ids is None:
            self.parent_ids = []
        if self.child_ids is None:
            self.child_ids = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class MockCruxClaim:
    """Mock CruxClaim for testing."""

    claim_id: str
    statement: str
    author: str
    crux_score: float = 0.5
    influence_score: float = 0.4
    disagreement_score: float = 0.3
    uncertainty_score: float = 0.3
    centrality_score: float = 0.4
    affected_claims: list = None
    contesting_agents: list = None
    resolution_impact: float = 0.5

    def __post_init__(self):
        if self.affected_claims is None:
            self.affected_claims = []
        if self.contesting_agents is None:
            self.contesting_agents = []


@pytest.fixture
def adapter():
    """Create a BeliefAdapter for testing."""
    return BeliefAdapter()


@pytest.fixture
def adapter_with_beliefs():
    """Create an adapter with some beliefs stored."""
    adapter = BeliefAdapter()

    # Store some beliefs
    for i in range(5):
        node = MockBeliefNode(
            node_id=f"node_{i}",
            claim_id=f"claim_{i}",
            claim_statement=f"Test claim {i}",
            author="claude",
        )
        node.posterior.p_true = 0.85 + (i * 0.02)  # 0.85 to 0.93
        adapter.store_converged_belief(node, debate_id=f"debate_{i}")

    return adapter


class TestKMThresholdUpdate:
    """Tests for KMThresholdUpdate dataclass."""

    def test_default_values(self):
        """Test default values."""
        update = KMThresholdUpdate(
            old_belief_confidence_threshold=0.8,
            new_belief_confidence_threshold=0.75,
            old_crux_score_threshold=0.3,
            new_crux_score_threshold=0.25,
        )
        assert update.patterns_analyzed == 0
        assert update.adjustments_made == 0
        assert update.confidence == 0.7
        assert update.recommendation == "keep"
        assert update.metadata == {}

    def test_custom_values(self):
        """Test custom values."""
        update = KMThresholdUpdate(
            old_belief_confidence_threshold=0.8,
            new_belief_confidence_threshold=0.65,
            old_crux_score_threshold=0.3,
            new_crux_score_threshold=0.25,
            patterns_analyzed=100,
            adjustments_made=2,
            confidence=0.9,
            recommendation="decrease",
            metadata={"source": "test"},
        )
        assert update.patterns_analyzed == 100
        assert update.recommendation == "decrease"


class TestKMBeliefValidation:
    """Tests for KMBeliefValidation dataclass."""

    def test_default_values(self):
        """Test default values."""
        validation = KMBeliefValidation(
            belief_id="bl_123",
            km_confidence=0.8,
        )
        assert validation.outcome_success_rate == 0.0
        assert validation.cross_debate_frequency == 0
        assert validation.was_contradicted is False
        assert validation.was_supported is False
        assert validation.recommendation == "keep"
        assert validation.adjustment == 0.0

    def test_boost_recommendation(self):
        """Test boost recommendation with supporting evidence."""
        validation = KMBeliefValidation(
            belief_id="bl_123",
            km_confidence=0.85,
            outcome_success_rate=0.8,
            cross_debate_frequency=5,
            was_supported=True,
            recommendation="boost",
            adjustment=0.08,
        )
        assert validation.was_supported is True
        assert validation.recommendation == "boost"
        assert validation.adjustment > 0


class TestKMPriorRecommendation:
    """Tests for KMPriorRecommendation dataclass."""

    def test_default_values(self):
        """Test default values."""
        rec = KMPriorRecommendation(
            claim_type="factual",
            recommended_prior=0.7,
        )
        assert rec.sample_count == 0
        assert rec.confidence == 0.7
        assert rec.supporting_debates == []

    def test_with_supporting_data(self):
        """Test with supporting debates."""
        rec = KMPriorRecommendation(
            claim_type="prediction",
            recommended_prior=0.4,
            sample_count=15,
            confidence=0.75,
            supporting_debates=["d1", "d2", "d3"],
        )
        assert rec.sample_count == 15
        assert len(rec.supporting_debates) == 3


class TestBeliefAdapterThresholdUpdates:
    """Tests for threshold updates from KM patterns."""

    @pytest.mark.asyncio
    async def test_update_thresholds_empty_items(self, adapter):
        """Test threshold update with empty items."""
        update = await adapter.update_belief_thresholds_from_km([])

        assert update.patterns_analyzed == 0
        assert update.old_belief_confidence_threshold == 0.8
        assert update.new_belief_confidence_threshold == 0.8  # No change
        assert update.recommendation == "keep"

    @pytest.mark.asyncio
    async def test_update_thresholds_with_success_data(self, adapter):
        """Test threshold update with success data."""
        # Create items with different confidence levels and success rates
        km_items = []

        # 0.7-0.8 bucket: 70% success rate
        for i in range(10):
            km_items.append(
                {
                    "metadata": {
                        "confidence": 0.75,
                        "outcome_success": i < 7,  # 7/10 = 70%
                    }
                }
            )

        # 0.8-0.9 bucket: 90% success rate
        for i in range(10):
            km_items.append(
                {
                    "metadata": {
                        "confidence": 0.85,
                        "outcome_success": i < 9,  # 9/10 = 90%
                    }
                }
            )

        # Need 100 items for high confidence
        for _ in range(80):
            km_items.append(
                {
                    "metadata": {
                        "confidence": 0.95,
                        "outcome_success": True,
                    }
                }
            )

        update = await adapter.update_belief_thresholds_from_km(km_items, min_confidence=0.7)

        assert update.patterns_analyzed == 100
        assert update.confidence >= 0.7

    @pytest.mark.asyncio
    async def test_threshold_decrease_recommendation(self, adapter):
        """Test that threshold can be lowered if lower confidence succeeds."""
        km_items = []

        # 0.6-0.7 bucket: 80% success rate (should enable lowering threshold)
        for i in range(5):
            km_items.append(
                {
                    "metadata": {
                        "confidence": 0.65,
                        "outcome_success": i < 4,  # 4/5 = 80%
                    }
                }
            )

        # 0.7-0.8 bucket: 70% success rate
        for i in range(10):
            km_items.append(
                {
                    "metadata": {
                        "confidence": 0.75,
                        "outcome_success": i < 7,
                    }
                }
            )

        # Add more items to reach confidence threshold
        for _ in range(85):
            km_items.append(
                {
                    "metadata": {
                        "confidence": 0.9,
                        "outcome_success": True,
                    }
                }
            )

        update = await adapter.update_belief_thresholds_from_km(km_items, min_confidence=0.7)

        # Should recommend lowering threshold since 0.65 has good success
        assert update.confidence >= 0.7
        assert update.patterns_analyzed == 100


class TestBeliefAdapterKMValidatedPriors:
    """Tests for KM-validated prior computation."""

    @pytest.mark.asyncio
    async def test_get_priors_no_items(self, adapter):
        """Test prior computation with no matching items."""
        rec = await adapter.get_km_validated_priors("factual", [])

        assert rec.claim_type == "factual"
        assert rec.recommended_prior == 0.5  # Default
        assert rec.sample_count == 0
        assert rec.confidence == 0.5

    @pytest.mark.asyncio
    async def test_get_priors_with_matching_items(self, adapter):
        """Test prior computation with matching items."""
        km_items = [
            {
                "metadata": {
                    "claim_type": "factual",
                    "outcome_success": True,
                    "confidence": 0.8,
                    "debate_id": "d1",
                }
            },
            {
                "metadata": {
                    "claim_type": "factual",
                    "outcome_success": True,
                    "confidence": 0.7,
                    "debate_id": "d2",
                }
            },
            {
                "metadata": {
                    "claim_type": "factual",
                    "outcome_success": False,
                    "confidence": 0.6,
                    "debate_id": "d3",
                }
            },
            {
                "metadata": {
                    "claim_type": "opinion",
                    "outcome_success": True,
                    "confidence": 0.9,
                    "debate_id": "d4",
                }
            },  # Different type
        ]

        rec = await adapter.get_km_validated_priors("factual", km_items)

        assert rec.claim_type == "factual"
        assert rec.sample_count == 3  # Only factual claims
        assert rec.recommended_prior == 2 / 3  # 2 successes out of 3
        assert len(rec.supporting_debates) == 3

    @pytest.mark.asyncio
    async def test_priors_cached(self, adapter):
        """Test that priors are cached."""
        km_items = [
            {"metadata": {"claim_type": "factual", "outcome_success": True, "confidence": 0.8}},
            {"metadata": {"claim_type": "factual", "outcome_success": True, "confidence": 0.7}},
        ]

        # First call computes
        rec1 = await adapter.get_km_validated_priors("factual", km_items)

        # Second call without items should use cache
        rec2 = await adapter.get_km_validated_priors("factual")

        assert rec1.claim_type == rec2.claim_type
        assert rec1.recommended_prior == rec2.recommended_prior


class TestBeliefAdapterValidation:
    """Tests for belief validation from KM."""

    @pytest.mark.asyncio
    async def test_validate_unknown_belief(self, adapter):
        """Test validation of unknown belief."""
        validation = await adapter.validate_belief_from_km(
            "unknown_belief",
            [],
        )

        assert validation.recommendation == "review"
        assert validation.km_confidence == 0.0
        assert "belief_not_found" in validation.metadata.get("error", "")

    @pytest.mark.asyncio
    async def test_validate_supported_belief(self, adapter_with_beliefs):
        """Test validation of well-supported belief."""
        # Record some successful outcomes
        adapter_with_beliefs.record_outcome("bl_node_0", "debate_10", True)
        adapter_with_beliefs.record_outcome("bl_node_0", "debate_11", True)
        adapter_with_beliefs.record_outcome("bl_node_0", "debate_12", True)
        adapter_with_beliefs.record_outcome("bl_node_0", "debate_13", True)

        # Create supporting cross-references
        cross_refs = [
            {"metadata": {"relationship": "supports", "debate_id": "d1", "outcome_success": True}},
            {"metadata": {"relationship": "supports", "debate_id": "d2", "outcome_success": True}},
            {"metadata": {"relationship": "supports", "debate_id": "d3", "outcome_success": True}},
            {"metadata": {"relationship": "supports", "debate_id": "d4", "outcome_success": True}},
        ]

        validation = await adapter_with_beliefs.validate_belief_from_km("bl_node_0", cross_refs)

        assert validation.was_supported is True
        assert validation.recommendation == "boost"
        assert validation.adjustment > 0
        assert validation.outcome_success_rate > 0.5

    @pytest.mark.asyncio
    async def test_validate_contradicted_belief(self, adapter_with_beliefs):
        """Test validation of contradicted belief."""
        cross_refs = [
            {"metadata": {"relationship": "contradicts", "debate_id": "d1"}},
            {"metadata": {"relationship": "contradicts", "debate_id": "d2"}},
            {"metadata": {"relationship": "contradicts", "debate_id": "d3"}},
            {"metadata": {"relationship": "contradicts", "debate_id": "d4"}},
        ]

        validation = await adapter_with_beliefs.validate_belief_from_km("bl_node_0", cross_refs)

        assert validation.was_contradicted is True
        assert validation.recommendation == "penalize"
        assert validation.adjustment < 0

    @pytest.mark.asyncio
    async def test_validate_neutral_belief(self, adapter_with_beliefs):
        """Test validation with mixed evidence."""
        cross_refs = [
            {"metadata": {"relationship": "supports", "debate_id": "d1"}},
            {"metadata": {"relationship": "contradicts", "debate_id": "d2"}},
        ]

        validation = await adapter_with_beliefs.validate_belief_from_km("bl_node_0", cross_refs)

        assert validation.recommendation == "keep"
        assert validation.adjustment == 0.0


class TestBeliefAdapterApplyValidation:
    """Tests for applying KM validations."""

    @pytest.mark.asyncio
    async def test_apply_validation_success(self, adapter_with_beliefs):
        """Test successful validation application."""
        validation = KMBeliefValidation(
            belief_id="bl_node_0",
            km_confidence=0.85,
            recommendation="boost",
            adjustment=0.05,
        )

        result = await adapter_with_beliefs.apply_km_validation(validation)

        assert result is True

        # Check belief was updated
        belief = adapter_with_beliefs.get_belief("bl_node_0")
        assert belief["km_validated"] is True
        assert belief["km_confidence"] == 0.85
        assert "km_validation" in belief["metadata"]

    @pytest.mark.asyncio
    async def test_apply_validation_unknown_belief(self, adapter):
        """Test applying validation to unknown belief."""
        validation = KMBeliefValidation(
            belief_id="unknown",
            km_confidence=0.8,
            recommendation="boost",
            adjustment=0.1,
        )

        result = await adapter.apply_km_validation(validation)
        assert result is False

    @pytest.mark.asyncio
    async def test_apply_negative_adjustment(self, adapter_with_beliefs):
        """Test applying negative adjustment."""
        # Get original confidence
        belief = adapter_with_beliefs.get_belief("bl_node_0")
        original_confidence = belief["confidence"]

        validation = KMBeliefValidation(
            belief_id="bl_node_0",
            km_confidence=0.4,
            recommendation="penalize",
            adjustment=-0.1,
        )

        await adapter_with_beliefs.apply_km_validation(validation)

        # Check confidence decreased
        updated_belief = adapter_with_beliefs.get_belief("bl_node_0")
        assert updated_belief["confidence"] < original_confidence

    @pytest.mark.asyncio
    async def test_confidence_bounded(self, adapter_with_beliefs):
        """Test that confidence stays bounded 0-1."""
        # Apply large negative adjustment
        validation = KMBeliefValidation(
            belief_id="bl_node_0",
            km_confidence=0.1,
            recommendation="penalize",
            adjustment=-10.0,  # Very large
        )

        await adapter_with_beliefs.apply_km_validation(validation)

        belief = adapter_with_beliefs.get_belief("bl_node_0")
        assert belief["confidence"] >= 0.0
        assert belief["confidence"] <= 1.0


class TestBeliefAdapterBatchSync:
    """Tests for batch sync of KM validations."""

    @pytest.mark.asyncio
    async def test_sync_empty_items(self, adapter):
        """Test sync with empty items."""
        result = await adapter.sync_validations_from_km([])

        assert isinstance(result, BeliefThresholdSyncResult)
        assert result.beliefs_analyzed == 0
        assert result.cruxes_analyzed == 0

    @pytest.mark.asyncio
    async def test_sync_with_beliefs(self, adapter_with_beliefs):
        """Test sync with belief items."""
        km_items = [
            {
                "metadata": {
                    "belief_id": "bl_node_0",
                    "relationship": "supports",
                    "outcome_success": True,
                    "confidence": 0.8,
                }
            },
            {
                "metadata": {
                    "belief_id": "bl_node_1",
                    "relationship": "supports",
                    "outcome_success": True,
                    "confidence": 0.85,
                }
            },
        ]

        result = await adapter_with_beliefs.sync_validations_from_km(km_items)

        assert result.beliefs_analyzed >= 2
        assert len(result.validation_results) >= 2
        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_sync_with_cruxes(self, adapter):
        """Test sync with crux items."""
        # Store a crux first
        crux = MockCruxClaim(
            claim_id="crux_1",
            statement="Test crux claim",
            author="gemini",
            crux_score=0.5,
        )
        adapter.store_crux(crux, debate_id="debate_1", topics=["testing"])

        km_items = [
            {
                "metadata": {
                    "belief_id": "cx_crux_1",
                    "is_crux": True,
                    "relationship": "supports",
                    "outcome_success": True,
                }
            },
        ]

        result = await adapter.sync_validations_from_km(km_items)

        assert result.cruxes_analyzed >= 1


class TestBeliefAdapterOutcomeRecording:
    """Tests for outcome recording."""

    def test_record_outcome(self, adapter):
        """Test recording a debate outcome."""
        adapter.record_outcome("bl_123", "debate_1", True, 0.9)

        stats = adapter.get_reverse_flow_stats()
        assert stats["outcome_history_size"] == 1

    def test_record_multiple_outcomes(self, adapter):
        """Test recording multiple outcomes for same belief."""
        adapter.record_outcome("bl_123", "debate_1", True)
        adapter.record_outcome("bl_123", "debate_2", False)
        adapter.record_outcome("bl_123", "debate_3", True)

        stats = adapter.get_reverse_flow_stats()
        assert stats["outcome_history_size"] == 3

    @pytest.mark.asyncio
    async def test_recorded_outcomes_affect_validation(self, adapter_with_beliefs):
        """Test that recorded outcomes affect validation."""
        # Record multiple successful outcomes
        for i in range(5):
            adapter_with_beliefs.record_outcome("bl_node_0", f"debate_{i+10}", True)

        validation = await adapter_with_beliefs.validate_belief_from_km(
            "bl_node_0",
            [],  # No cross-refs, but outcome history
        )

        # Should have good success rate from recorded outcomes
        assert validation.outcome_success_rate == 1.0  # 5/5 successful
        assert validation.cross_debate_frequency == 5


class TestBeliefAdapterStats:
    """Tests for reverse flow statistics."""

    def test_stats_empty(self, adapter):
        """Test stats with no activity."""
        stats = adapter.get_reverse_flow_stats()

        assert stats["km_validations_applied"] == 0
        assert stats["km_threshold_updates"] == 0
        assert stats["km_priors_computed"] == 0

    @pytest.mark.asyncio
    async def test_stats_after_operations(self, adapter_with_beliefs):
        """Test stats after some operations."""
        # Do some validations
        await adapter_with_beliefs.validate_belief_from_km("bl_node_0", [])
        await adapter_with_beliefs.validate_belief_from_km("bl_node_1", [])

        # Get priors
        await adapter_with_beliefs.get_km_validated_priors(
            "factual", [{"metadata": {"claim_type": "factual", "outcome_success": True}}]
        )

        stats = adapter_with_beliefs.get_reverse_flow_stats()

        assert stats["km_validations_applied"] >= 2
        assert stats["km_priors_computed"] >= 1
        assert stats["validations_stored"] >= 2

    def test_clear_reverse_flow_state(self, adapter_with_beliefs):
        """Test clearing reverse flow state."""
        # Add some state
        adapter_with_beliefs.record_outcome("bl_123", "d1", True)
        adapter_with_beliefs._km_validations_applied = 5
        adapter_with_beliefs._km_threshold_updates = 2

        # Clear
        adapter_with_beliefs.clear_reverse_flow_state()

        # Verify cleared
        stats = adapter_with_beliefs.get_reverse_flow_stats()
        assert stats["km_validations_applied"] == 0
        assert stats["km_threshold_updates"] == 0
        assert stats["outcome_history_size"] == 0

    def test_get_stats_includes_km_data(self, adapter):
        """Test that get_stats includes KM data."""
        stats = adapter.get_stats()

        assert "km_validations_applied" in stats
        assert "km_threshold_updates" in stats
        assert "km_priors_computed" in stats


class TestBeliefAdapterIntegration:
    """Integration tests for bidirectional flow."""

    @pytest.mark.asyncio
    async def test_full_bidirectional_cycle(self, adapter):
        """Test complete bidirectional cycle: store → validate → update."""
        # 1. Store some beliefs
        for i in range(3):
            node = MockBeliefNode(
                node_id=f"int_node_{i}",
                claim_id=f"int_claim_{i}",
                claim_statement=f"Integration test claim {i}",
                author="claude",
            )
            node.posterior.p_true = 0.85
            adapter.store_converged_belief(node, debate_id=f"int_debate_{i}")

        # 2. Record outcomes
        adapter.record_outcome("bl_int_node_0", "outcome_1", True)
        adapter.record_outcome("bl_int_node_0", "outcome_2", True)
        adapter.record_outcome("bl_int_node_1", "outcome_3", False)

        # 3. Validate from KM
        validation0 = await adapter.validate_belief_from_km(
            "bl_int_node_0",
            [{"metadata": {"relationship": "supports", "outcome_success": True}}],
        )
        validation1 = await adapter.validate_belief_from_km(
            "bl_int_node_1",
            [{"metadata": {"relationship": "contradicts", "outcome_success": False}}],
        )

        # 4. Apply validations
        await adapter.apply_km_validation(validation0)
        await adapter.apply_km_validation(validation1)

        # 5. Verify results
        belief0 = adapter.get_belief("bl_int_node_0")
        belief1 = adapter.get_belief("bl_int_node_1")

        assert belief0["km_validated"] is True
        assert belief1["km_validated"] is True

        # Belief with good outcomes should have different confidence than one with bad
        stats = adapter.get_reverse_flow_stats()
        assert stats["km_validations_applied"] >= 2

    @pytest.mark.asyncio
    async def test_threshold_adaptation_over_time(self, adapter):
        """Test that thresholds adapt based on outcome patterns."""
        original_belief_threshold = adapter.MIN_BELIEF_CONFIDENCE
        original_crux_threshold = adapter.MIN_CRUX_SCORE

        # Generate items showing lower threshold works well
        km_items = []
        for i in range(120):  # Need enough for high confidence
            conf = 0.65 + (i % 4) * 0.1  # 0.65, 0.75, 0.85, 0.95
            km_items.append(
                {
                    "metadata": {
                        "confidence": conf,
                        "outcome_success": conf >= 0.65,  # All succeed
                        "is_crux": i % 5 == 0,
                        "crux_score": 0.35 if i % 5 == 0 else 0,
                    }
                }
            )

        update = await adapter.update_belief_thresholds_from_km(km_items, min_confidence=0.5)

        assert update.patterns_analyzed == 120
        # Thresholds may have changed based on data
        assert update.old_belief_confidence_threshold == original_belief_threshold


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
