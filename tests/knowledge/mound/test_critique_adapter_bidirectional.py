"""
Tests for CritiqueAdapter bidirectional integration (Critique ↔ KM).

Tests the reverse flow methods that enable Knowledge Mound patterns
to influence critique pattern boosting and agent reputation.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock
from dataclasses import dataclass

from aragora.knowledge.mound.adapters.critique_adapter import (
    CritiqueAdapter,
    KMPatternBoost,
    KMReputationAdjustment,
    KMPatternValidation,
    CritiqueKMSyncResult,
)


@dataclass
class MockPattern:
    """Mock Pattern for testing."""

    id: str
    issue_type: str
    issue_text: str
    suggestion_text: str = "Test suggestion"
    success_count: int = 5
    failure_count: int = 2
    avg_severity: float = 0.5
    example_task: str = "Test task"
    created_at: str = "2024-01-01T00:00:00"
    updated_at: str = "2024-01-02T00:00:00"

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0


@dataclass
class MockAgentReputation:
    """Mock AgentReputation for testing."""

    agent_name: str
    reputation_score: float = 0.7
    critique_count: int = 10
    acceptance_rate: float = 0.7


@pytest.fixture
def mock_store():
    """Create a mock CritiqueStore."""
    store = MagicMock()

    # Create some mock patterns
    patterns = [
        MockPattern(
            id=f"pattern_{i}",
            issue_type="performance" if i % 2 == 0 else "security",
            issue_text=f"Performance issue {i}",
            success_count=5 + i,
            failure_count=2,
        )
        for i in range(5)
    ]

    # Mock retrieve_patterns
    def retrieve_patterns(issue_type=None, min_success=1, limit=100):
        result = patterns
        if issue_type:
            result = [p for p in result if p.issue_type == issue_type]
        return result[:limit]

    store.retrieve_patterns.side_effect = retrieve_patterns

    # Mock get_reputation
    reputations = {
        "claude": MockAgentReputation("claude", 0.8, 20, 0.75),
        "gemini": MockAgentReputation("gemini", 0.7, 15, 0.65),
    }
    store.get_reputation.side_effect = lambda name: reputations.get(name)
    store.get_all_reputations.return_value = list(reputations.values())

    # Mock record_pattern_outcome
    store.record_pattern_outcome.return_value = None

    # Mock update_reputation
    store.update_reputation.return_value = None

    # Store patterns for lookup
    store._patterns = patterns

    return store


@pytest.fixture
def adapter(mock_store):
    """Create a CritiqueAdapter with mock store."""
    return CritiqueAdapter(mock_store)


class TestKMPatternBoost:
    """Tests for KMPatternBoost dataclass."""

    def test_default_values(self):
        """Test default values."""
        boost = KMPatternBoost(pattern_id="pat_123")
        assert boost.boost_amount == 0
        assert boost.km_confidence == 0.7
        assert boost.source_debates == []
        assert boost.was_applied is False

    def test_custom_values(self):
        """Test custom values."""
        boost = KMPatternBoost(
            pattern_id="pat_123",
            boost_amount=3,
            km_confidence=0.9,
            source_debates=["d1", "d2", "d3"],
            was_applied=True,
        )
        assert boost.boost_amount == 3
        assert len(boost.source_debates) == 3


class TestKMReputationAdjustment:
    """Tests for KMReputationAdjustment dataclass."""

    def test_default_values(self):
        """Test default values."""
        adj = KMReputationAdjustment(agent_name="claude")
        assert adj.adjustment == 0.0
        assert adj.pattern_contributions == 0
        assert adj.km_confidence == 0.7
        assert adj.recommendation == "keep"

    def test_boost_adjustment(self):
        """Test boost adjustment."""
        adj = KMReputationAdjustment(
            agent_name="claude",
            adjustment=0.1,
            pattern_contributions=5,
            km_confidence=0.85,
            recommendation="boost",
        )
        assert adj.adjustment == 0.1
        assert adj.recommendation == "boost"


class TestKMPatternValidation:
    """Tests for KMPatternValidation dataclass."""

    def test_default_values(self):
        """Test default values."""
        val = KMPatternValidation(pattern_id="pat_123")
        assert val.km_confidence == 0.7
        assert val.cross_debate_usage == 0
        assert val.outcome_success_rate == 0.0
        assert val.recommendation == "keep"
        assert val.boost_amount == 0

    def test_boost_validation(self):
        """Test boost validation."""
        val = KMPatternValidation(
            pattern_id="pat_123",
            km_confidence=0.9,
            cross_debate_usage=5,
            outcome_success_rate=0.85,
            recommendation="boost",
            boost_amount=3,
        )
        assert val.recommendation == "boost"
        assert val.boost_amount == 3


class TestCritiqueAdapterPatternValidation:
    """Tests for pattern validation from KM."""

    @pytest.mark.asyncio
    async def test_validate_pattern_empty_refs(self, adapter):
        """Test validation with no cross-references."""
        validation = await adapter.validate_pattern_from_km("pat_123", [])

        assert validation.pattern_id == "pat_123"
        assert validation.cross_debate_usage == 0
        assert validation.recommendation == "keep"

    @pytest.mark.asyncio
    async def test_validate_pattern_high_success(self, adapter):
        """Test validation with high success rate."""
        # Record some successful usage
        for i in range(6):
            adapter.record_pattern_usage(f"pattern_{i}", f"debate_{i}", True)

        cross_refs = [
            {"metadata": {"debate_id": "d7", "outcome_success": True}},
            {"metadata": {"debate_id": "d8", "outcome_success": True}},
        ]

        validation = await adapter.validate_pattern_from_km("pattern_0", cross_refs)

        assert validation.pattern_id == "pattern_0"
        # Should recommend boost for high success
        # 6 recorded + 2 cross-refs = 8 outcomes, all successful = 100%
        assert validation.outcome_success_rate >= 0.8 or validation.cross_debate_usage >= 1

    @pytest.mark.asyncio
    async def test_validate_pattern_low_success(self, adapter):
        """Test validation with low success rate."""
        # Record mostly failures
        for i in range(6):
            adapter.record_pattern_usage(
                "bad_pattern", f"debate_{i}", i < 1
            )  # 1 success, 5 failures

        validation = await adapter.validate_pattern_from_km("bad_pattern", [])

        assert validation.pattern_id == "bad_pattern"
        assert validation.outcome_success_rate < 0.3
        assert validation.recommendation == "archive"


class TestCritiqueAdapterPatternBoost:
    """Tests for applying pattern boosts."""

    @pytest.mark.asyncio
    async def test_apply_boost_no_recommendation(self, adapter):
        """Test that non-boost recommendations don't apply."""
        validation = KMPatternValidation(
            pattern_id="pattern_0",
            km_confidence=0.8,
            recommendation="keep",
            boost_amount=0,
        )

        boost = await adapter.apply_pattern_boost(validation)

        assert boost.was_applied is False

    @pytest.mark.asyncio
    async def test_apply_boost_pattern_not_found(self, adapter, mock_store):
        """Test boost fails when pattern not found."""
        # Clear patterns so get() returns None
        mock_store.retrieve_patterns.return_value = []

        validation = KMPatternValidation(
            pattern_id="nonexistent",
            km_confidence=0.8,
            recommendation="boost",
            boost_amount=3,
        )

        boost = await adapter.apply_pattern_boost(validation)

        assert boost.was_applied is False
        assert "pattern_not_found" in boost.metadata.get("error", "")

    @pytest.mark.asyncio
    async def test_apply_boost_success(self, adapter, mock_store):
        """Test successful boost application."""
        validation = KMPatternValidation(
            pattern_id="pattern_0",
            km_confidence=0.8,
            recommendation="boost",
            boost_amount=3,
        )

        boost = await adapter.apply_pattern_boost(validation)

        assert boost.was_applied is True
        assert boost.boost_amount == 3
        # Should have called record_pattern_outcome 3 times
        assert mock_store.record_pattern_outcome.call_count == 3


class TestCritiqueAdapterReputationAdjustment:
    """Tests for reputation adjustments."""

    @pytest.mark.asyncio
    async def test_compute_adjustment_no_data(self, adapter):
        """Test adjustment with no data."""
        adjustment = await adapter.compute_reputation_adjustment("unknown", [])

        assert adjustment.agent_name == "unknown"
        assert adjustment.pattern_contributions == 0
        assert adjustment.recommendation == "keep"

    @pytest.mark.asyncio
    async def test_compute_adjustment_high_success(self, adapter):
        """Test adjustment with high success rate."""
        km_items = [
            {"metadata": {"agent_name": "claude", "outcome_success": True}},
            {"metadata": {"agent_name": "claude", "outcome_success": True}},
            {"metadata": {"agent_name": "claude", "outcome_success": True}},
            {"metadata": {"agent_name": "claude", "outcome_success": True}},
            {"metadata": {"agent_name": "claude", "outcome_success": True}},
        ]

        adjustment = await adapter.compute_reputation_adjustment("claude", km_items)

        assert adjustment.agent_name == "claude"
        assert adjustment.pattern_contributions == 5
        assert adjustment.metadata.get("success_rate") >= 0.8
        assert adjustment.recommendation == "boost"
        assert adjustment.adjustment > 0

    @pytest.mark.asyncio
    async def test_compute_adjustment_low_success(self, adapter):
        """Test adjustment with low success rate."""
        km_items = [
            {"metadata": {"agent_name": "claude", "outcome_success": False}},
            {"metadata": {"agent_name": "claude", "outcome_success": False}},
            {"metadata": {"agent_name": "claude", "outcome_success": False}},
            {"metadata": {"agent_name": "claude", "outcome_success": False}},
            {"metadata": {"agent_name": "claude", "outcome_success": True}},
        ]

        adjustment = await adapter.compute_reputation_adjustment("claude", km_items)

        assert adjustment.metadata.get("success_rate") < 0.3
        assert adjustment.recommendation == "penalize"
        assert adjustment.adjustment < 0

    @pytest.mark.asyncio
    async def test_apply_adjustment_no_reputation(self, adapter, mock_store):
        """Test applying adjustment when agent has no reputation."""
        mock_store.get_reputation.return_value = None

        adjustment = KMReputationAdjustment(
            agent_name="unknown",
            adjustment=0.1,
            recommendation="boost",
        )

        result = await adapter.apply_reputation_adjustment(adjustment)

        assert result is False

    @pytest.mark.asyncio
    async def test_apply_adjustment_success(self, adapter, mock_store):
        """Test successful adjustment application."""
        adjustment = KMReputationAdjustment(
            agent_name="claude",
            adjustment=0.1,
            recommendation="boost",
        )

        result = await adapter.apply_reputation_adjustment(adjustment)

        assert result is True
        # Should have called update_reputation
        assert mock_store.update_reputation.call_count > 0


class TestCritiqueAdapterBatchSync:
    """Tests for batch sync of KM validations."""

    @pytest.mark.asyncio
    async def test_sync_empty_items(self, adapter):
        """Test sync with empty items."""
        result = await adapter.sync_validations_from_km([])

        assert isinstance(result, CritiqueKMSyncResult)
        assert result.patterns_analyzed == 0
        assert result.agents_analyzed == 0

    @pytest.mark.asyncio
    async def test_sync_with_patterns(self, adapter):
        """Test sync with pattern items."""
        # Record some usage first
        for i in range(6):
            adapter.record_pattern_usage("pattern_0", f"d_{i}", True)

        km_items = [
            {
                "metadata": {
                    "pattern_id": "pattern_0",
                    "outcome_success": True,
                    "agent_name": "claude",
                }
            },
            {
                "metadata": {
                    "pattern_id": "pattern_0",
                    "outcome_success": True,
                    "agent_name": "claude",
                }
            },
        ]

        result = await adapter.sync_validations_from_km(km_items, min_confidence=0.5)

        assert result.patterns_analyzed >= 1
        assert result.agents_analyzed >= 1
        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_sync_with_multiple_agents(self, adapter):
        """Test sync with multiple agents."""
        km_items = [
            {"metadata": {"agent_name": "claude", "outcome_success": True}},
            {"metadata": {"agent_name": "gemini", "outcome_success": True}},
            {"metadata": {"agents_involved": ["claude", "gemini"], "outcome_success": True}},
        ]

        result = await adapter.sync_validations_from_km(km_items)

        assert result.agents_analyzed >= 2


class TestCritiqueAdapterUsageRecording:
    """Tests for pattern usage recording."""

    def test_record_usage(self, adapter):
        """Test recording pattern usage."""
        adapter.record_pattern_usage("pat_123", "debate_1", True, 0.9)

        stats = adapter.get_reverse_flow_stats()
        assert stats["total_usage_records"] == 1
        assert stats["patterns_tracked"] == 1

    def test_record_multiple_usage(self, adapter):
        """Test recording multiple usages."""
        adapter.record_pattern_usage("pat_123", "debate_1", True)
        adapter.record_pattern_usage("pat_123", "debate_2", False)
        adapter.record_pattern_usage("pat_456", "debate_3", True)

        stats = adapter.get_reverse_flow_stats()
        assert stats["total_usage_records"] == 3
        assert stats["patterns_tracked"] == 2


class TestCritiqueAdapterStats:
    """Tests for reverse flow statistics."""

    def test_stats_empty(self, adapter):
        """Test stats with no activity."""
        stats = adapter.get_reverse_flow_stats()

        assert stats["km_boosts_applied"] == 0
        assert stats["km_reputation_adjustments"] == 0
        assert stats["validations_stored"] == 0

    @pytest.mark.asyncio
    async def test_stats_after_operations(self, adapter):
        """Test stats after some operations."""
        # Do some validations
        await adapter.validate_pattern_from_km("pattern_0", [])
        await adapter.validate_pattern_from_km("pattern_1", [])

        stats = adapter.get_reverse_flow_stats()

        assert stats["validations_stored"] >= 2

    def test_clear_reverse_flow_state(self, adapter):
        """Test clearing reverse flow state."""
        # Add some state
        adapter.record_pattern_usage("pat_123", "d1", True)
        adapter._km_boosts_applied = 5
        adapter._km_reputation_adjustments = 3

        # Clear
        adapter.clear_reverse_flow_state()

        # Verify cleared
        stats = adapter.get_reverse_flow_stats()
        assert stats["km_boosts_applied"] == 0
        assert stats["km_reputation_adjustments"] == 0
        assert stats["total_usage_records"] == 0


class TestCritiqueAdapterIntegration:
    """Integration tests for bidirectional flow."""

    @pytest.mark.asyncio
    async def test_full_bidirectional_cycle(self, adapter, mock_store):
        """Test complete bidirectional cycle: record → validate → boost."""
        # 1. Record successful pattern usage
        for i in range(6):
            adapter.record_pattern_usage("pattern_0", f"debate_{i}", True)

        # 2. Validate pattern from KM
        cross_refs = [
            {"metadata": {"debate_id": "d7", "outcome_success": True}},
            {"metadata": {"debate_id": "d8", "outcome_success": True}},
        ]

        validation = await adapter.validate_pattern_from_km("pattern_0", cross_refs)

        # 3. Verify validation
        assert validation.outcome_success_rate >= 0.8

        # 4. If boost recommended, apply it
        if validation.recommendation == "boost":
            boost = await adapter.apply_pattern_boost(validation)
            # Boost should succeed
            assert boost.was_applied is True

        # 5. Verify stats
        stats = adapter.get_reverse_flow_stats()
        assert stats["validations_stored"] >= 1

    @pytest.mark.asyncio
    async def test_reputation_feedback_loop(self, adapter, mock_store):
        """Test reputation adjustment feedback loop."""
        # 1. Create KM items showing agent success
        km_items = []
        for i in range(8):
            km_items.append(
                {
                    "metadata": {
                        "agent_name": "claude",
                        "outcome_success": i < 7,  # 87.5% success
                    }
                }
            )

        # 2. Compute adjustment
        adjustment = await adapter.compute_reputation_adjustment("claude", km_items)

        # 3. Should recommend boost
        assert adjustment.recommendation == "boost"
        assert adjustment.adjustment > 0

        # 4. Apply adjustment
        result = await adapter.apply_reputation_adjustment(adjustment)
        assert result is True

        # 5. Verify stats
        stats = adapter.get_reverse_flow_stats()
        assert stats["km_reputation_adjustments"] >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
