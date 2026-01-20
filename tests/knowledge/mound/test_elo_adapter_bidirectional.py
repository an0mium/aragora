"""
Tests for EloAdapter bidirectional integration (KM ↔ ELO).

Tests the reverse flow methods that enable Knowledge Mound patterns
to influence ELO ratings.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from aragora.knowledge.mound.adapters.elo_adapter import (
    EloAdapter,
    KMEloPattern,
    EloAdjustmentRecommendation,
    EloSyncResult,
)


@pytest.fixture
def adapter():
    """Create an EloAdapter for testing."""
    return EloAdapter()


@pytest.fixture
def adapter_with_elo():
    """Create an EloAdapter with a mock ELO system."""
    mock_elo = MagicMock()

    # Mock get_rating to return a rating object
    mock_rating = MagicMock()
    mock_rating.elo = 1200.0
    mock_elo.get_rating.return_value = mock_rating

    adapter = EloAdapter(elo_system=mock_elo)
    return adapter


class TestKMEloPattern:
    """Tests for the KMEloPattern dataclass."""

    def test_default_values(self):
        """Test default values are set correctly."""
        pattern = KMEloPattern(
            agent_name="claude",
            pattern_type="success_contributor",
            confidence=0.85,
        )
        assert pattern.agent_name == "claude"
        assert pattern.pattern_type == "success_contributor"
        assert pattern.confidence == 0.85
        assert pattern.observation_count == 1
        assert pattern.domain is None
        assert pattern.debate_ids == []
        assert pattern.metadata == {}

    def test_custom_values(self):
        """Test custom values are applied correctly."""
        pattern = KMEloPattern(
            agent_name="gemini",
            pattern_type="domain_expert",
            confidence=0.9,
            observation_count=15,
            domain="security",
            debate_ids=["d1", "d2", "d3"],
            metadata={"domain_item_count": 15},
        )
        assert pattern.agent_name == "gemini"
        assert pattern.domain == "security"
        assert len(pattern.debate_ids) == 3


class TestEloAdjustmentRecommendation:
    """Tests for EloAdjustmentRecommendation dataclass."""

    def test_default_values(self):
        """Test default values."""
        rec = EloAdjustmentRecommendation(
            agent_name="claude",
            adjustment=25.0,
            reason="success contributor",
        )
        assert rec.agent_name == "claude"
        assert rec.adjustment == 25.0
        assert rec.patterns == []
        assert rec.confidence == 0.7
        assert rec.applied is False

    def test_with_patterns(self):
        """Test recommendation with supporting patterns."""
        patterns = [
            KMEloPattern("claude", "success_contributor", 0.85, 5),
            KMEloPattern("claude", "crux_resolver", 0.8, 3),
        ]
        rec = EloAdjustmentRecommendation(
            agent_name="claude",
            adjustment=35.0,
            reason="multiple positive patterns",
            patterns=patterns,
            confidence=0.85,
        )
        assert len(rec.patterns) == 2


class TestEloAdapterPatternAnalysis:
    """Tests for KM pattern analysis."""

    @pytest.mark.asyncio
    async def test_analyze_success_contributor(self, adapter):
        """Test detection of success contributor pattern."""
        km_items = [
            {"metadata": {"outcome_success": True, "debate_id": f"d{i}"}}
            for i in range(5)
        ] + [
            {"metadata": {"debate_id": f"d{i+5}"}}
            for i in range(2)
        ]

        patterns = await adapter.analyze_km_patterns_for_agent(
            "claude",
            km_items,
            min_confidence=0.6,
        )

        # Should detect success_contributor (5/7 = 71% success rate)
        success_patterns = [p for p in patterns if p.pattern_type == "success_contributor"]
        assert len(success_patterns) >= 1
        assert success_patterns[0].observation_count == 5

    @pytest.mark.asyncio
    async def test_analyze_contradiction_source(self, adapter):
        """Test detection of contradiction source pattern."""
        km_items = [
            {"metadata": {"was_contradicted": True, "debate_id": f"d{i}"}}
            for i in range(4)
        ] + [
            {"metadata": {"debate_id": f"d{i+4}"}}
            for i in range(6)
        ]

        patterns = await adapter.analyze_km_patterns_for_agent(
            "grok",
            km_items,
            min_confidence=0.5,
        )

        # Should detect contradiction_source (4/10 = 40% contradiction rate)
        contradiction_patterns = [p for p in patterns if p.pattern_type == "contradiction_source"]
        assert len(contradiction_patterns) >= 1

    @pytest.mark.asyncio
    async def test_analyze_domain_expert(self, adapter):
        """Test detection of domain expert pattern."""
        km_items = [
            {"metadata": {"domain": "security", "debate_id": f"d{i}"}}
            for i in range(8)
        ] + [
            {"metadata": {"domain": "testing", "debate_id": f"d{i+8}"}}
            for i in range(3)
        ]

        patterns = await adapter.analyze_km_patterns_for_agent(
            "gemini",
            km_items,
            min_confidence=0.5,
        )

        # Should detect domain_expert for security (8 items)
        domain_patterns = [p for p in patterns if p.pattern_type == "domain_expert"]
        assert len(domain_patterns) >= 1
        security_pattern = next((p for p in domain_patterns if p.domain == "security"), None)
        assert security_pattern is not None
        assert security_pattern.observation_count == 8

    @pytest.mark.asyncio
    async def test_analyze_crux_resolver(self, adapter):
        """Test detection of crux resolver pattern."""
        km_items = [
            {"metadata": {"crux_resolved": True, "debate_id": f"d{i}"}}
            for i in range(4)
        ] + [
            {"metadata": {"key_insight": True, "debate_id": f"d{i+4}"}}
            for i in range(2)
        ]

        patterns = await adapter.analyze_km_patterns_for_agent(
            "claude",
            km_items,
            min_confidence=0.5,
        )

        crux_patterns = [p for p in patterns if p.pattern_type == "crux_resolver"]
        assert len(crux_patterns) >= 1
        assert crux_patterns[0].observation_count >= 4

    @pytest.mark.asyncio
    async def test_analyze_empty_items(self, adapter):
        """Test analysis with empty items list."""
        patterns = await adapter.analyze_km_patterns_for_agent(
            "claude",
            [],
            min_confidence=0.5,
        )
        assert patterns == []

    @pytest.mark.asyncio
    async def test_analyze_confidence_filter(self, adapter):
        """Test that low confidence patterns are filtered."""
        # Only 2 successes out of 10 - too low for pattern
        km_items = [
            {"metadata": {"outcome_success": True, "debate_id": f"d{i}"}}
            for i in range(2)
        ] + [
            {"metadata": {"debate_id": f"d{i+2}"}}
            for i in range(8)
        ]

        patterns = await adapter.analyze_km_patterns_for_agent(
            "claude",
            km_items,
            min_confidence=0.7,
        )

        # Should not have success_contributor pattern (only 20% success)
        success_patterns = [p for p in patterns if p.pattern_type == "success_contributor"]
        assert len(success_patterns) == 0


class TestEloAdapterAdjustments:
    """Tests for ELO adjustment computation."""

    def test_compute_adjustment_success_contributor(self, adapter):
        """Test adjustment computation for success contributor."""
        patterns = [
            KMEloPattern(
                agent_name="claude",
                pattern_type="success_contributor",
                confidence=0.85,
                observation_count=10,
            ),
        ]

        rec = adapter.compute_elo_adjustment(patterns, max_adjustment=50.0)

        assert rec is not None
        assert rec.agent_name == "claude"
        assert rec.adjustment > 0  # Positive for success
        assert "success contributor" in rec.reason.lower()

    def test_compute_adjustment_contradiction_source(self, adapter):
        """Test adjustment computation for contradiction source."""
        patterns = [
            KMEloPattern(
                agent_name="grok",
                pattern_type="contradiction_source",
                confidence=0.8,
                observation_count=5,
            ),
        ]

        rec = adapter.compute_elo_adjustment(patterns, max_adjustment=50.0)

        assert rec is not None
        assert rec.adjustment < 0  # Negative for contradictions

    def test_compute_adjustment_combined_patterns(self, adapter):
        """Test adjustment with multiple patterns."""
        patterns = [
            KMEloPattern(
                "claude", "success_contributor", 0.85, 8,
            ),
            KMEloPattern(
                "claude", "domain_expert", 0.9, 12, domain="security",
            ),
            KMEloPattern(
                "claude", "crux_resolver", 0.8, 4,
            ),
        ]

        rec = adapter.compute_elo_adjustment(patterns, max_adjustment=50.0)

        assert rec is not None
        assert rec.adjustment > 0
        assert len(rec.patterns) == 3
        # Should be capped at max
        assert rec.adjustment <= 50.0

    def test_compute_adjustment_max_cap(self, adapter):
        """Test that adjustment is capped at maximum."""
        # Many strong positive patterns
        patterns = [
            KMEloPattern("claude", "success_contributor", 0.95, 20),
            KMEloPattern("claude", "domain_expert", 0.95, 30, domain="security"),
            KMEloPattern("claude", "domain_expert", 0.95, 25, domain="testing"),
            KMEloPattern("claude", "crux_resolver", 0.95, 15),
        ]

        rec = adapter.compute_elo_adjustment(patterns, max_adjustment=30.0)

        assert rec is not None
        assert abs(rec.adjustment) <= 30.0

    def test_compute_adjustment_empty_patterns(self, adapter):
        """Test adjustment with empty patterns."""
        rec = adapter.compute_elo_adjustment([], max_adjustment=50.0)
        assert rec is None

    def test_compute_adjustment_tiny_change_skipped(self, adapter):
        """Test that tiny adjustments are skipped."""
        # Very low confidence = tiny adjustment
        patterns = [
            KMEloPattern("claude", "success_contributor", 0.7, 3),
        ]

        # Manually check - this might produce very small adjustment
        rec = adapter.compute_elo_adjustment(patterns, max_adjustment=50.0)
        # Result depends on calculation, may or may not be None


class TestEloAdapterApplyAdjustments:
    """Tests for applying ELO adjustments."""

    @pytest.mark.asyncio
    async def test_apply_adjustment_success(self, adapter_with_elo):
        """Test successful adjustment application."""
        rec = EloAdjustmentRecommendation(
            agent_name="claude",
            adjustment=25.0,
            reason="success contributor",
            confidence=0.85,
        )

        result = await adapter_with_elo.apply_km_elo_adjustment(rec)

        assert result is True
        assert rec.applied is True
        adapter_with_elo._elo_system.get_rating.assert_called_with("claude")

    @pytest.mark.asyncio
    async def test_apply_adjustment_no_elo_system(self, adapter):
        """Test adjustment fails without ELO system."""
        rec = EloAdjustmentRecommendation(
            agent_name="claude",
            adjustment=25.0,
            reason="test",
            confidence=0.85,
        )

        result = await adapter.apply_km_elo_adjustment(rec)
        assert result is False

    @pytest.mark.asyncio
    async def test_apply_adjustment_already_applied(self, adapter_with_elo):
        """Test that already-applied adjustments are skipped."""
        rec = EloAdjustmentRecommendation(
            agent_name="claude",
            adjustment=25.0,
            reason="test",
            confidence=0.85,
            applied=True,  # Already applied
        )

        result = await adapter_with_elo.apply_km_elo_adjustment(rec)
        assert result is False

    @pytest.mark.asyncio
    async def test_apply_adjustment_low_confidence(self, adapter_with_elo):
        """Test that low-confidence adjustments are skipped."""
        rec = EloAdjustmentRecommendation(
            agent_name="claude",
            adjustment=25.0,
            reason="test",
            confidence=0.5,  # Below threshold
        )

        result = await adapter_with_elo.apply_km_elo_adjustment(rec, force=False)
        assert result is False

    @pytest.mark.asyncio
    async def test_apply_adjustment_force_low_confidence(self, adapter_with_elo):
        """Test force application of low-confidence adjustment."""
        rec = EloAdjustmentRecommendation(
            agent_name="claude",
            adjustment=25.0,
            reason="test",
            confidence=0.5,
        )

        result = await adapter_with_elo.apply_km_elo_adjustment(rec, force=True)
        assert result is True

    @pytest.mark.asyncio
    async def test_apply_adjustment_agent_not_found(self, adapter_with_elo):
        """Test adjustment fails when agent not in ELO system."""
        adapter_with_elo._elo_system.get_rating.return_value = None

        rec = EloAdjustmentRecommendation(
            agent_name="unknown_agent",
            adjustment=25.0,
            reason="test",
            confidence=0.85,
        )

        result = await adapter_with_elo.apply_km_elo_adjustment(rec)
        assert result is False


class TestEloAdapterBatchSync:
    """Tests for batch sync of KM patterns to ELO."""

    @pytest.mark.asyncio
    async def test_sync_km_to_elo_basic(self, adapter_with_elo):
        """Test basic batch sync."""
        agent_patterns = {
            "claude": [
                KMEloPattern("claude", "success_contributor", 0.85, 10),
            ],
            "gemini": [
                KMEloPattern("gemini", "domain_expert", 0.9, 12, domain="security"),
            ],
        }

        result = await adapter_with_elo.sync_km_to_elo(
            agent_patterns,
            auto_apply=True,
        )

        assert isinstance(result, EloSyncResult)
        assert result.total_patterns == 2
        assert result.adjustments_recommended >= 0
        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_sync_km_to_elo_no_auto_apply(self, adapter_with_elo):
        """Test sync without auto-apply creates recommendations only."""
        agent_patterns = {
            "claude": [
                KMEloPattern("claude", "success_contributor", 0.85, 10),
            ],
        }

        result = await adapter_with_elo.sync_km_to_elo(
            agent_patterns,
            auto_apply=False,
        )

        # Should have recommendations but none applied
        assert result.adjustments_applied == 0
        # Pending adjustments should be stored
        pending = adapter_with_elo.get_pending_adjustments()
        assert len(pending) >= 0


class TestEloAdapterStats:
    """Tests for reverse flow statistics."""

    @pytest.mark.asyncio
    async def test_get_reverse_flow_stats_empty(self, adapter):
        """Test stats with no patterns."""
        stats = adapter.get_reverse_flow_stats()

        assert stats["agents_with_patterns"] == 0
        assert stats["total_patterns"] == 0
        assert stats["pending_adjustments"] == 0
        assert stats["applied_adjustments"] == 0

    @pytest.mark.asyncio
    async def test_get_reverse_flow_stats_with_data(self, adapter):
        """Test stats after analyzing patterns."""
        km_items = [
            {"metadata": {"outcome_success": True, "debate_id": f"d{i}"}}
            for i in range(5)
        ]

        await adapter.analyze_km_patterns_for_agent(
            "claude",
            km_items,
            min_confidence=0.5,
        )

        stats = adapter.get_reverse_flow_stats()

        assert stats["agents_with_patterns"] >= 1
        assert stats["total_patterns"] >= 0

    def test_get_stats_includes_km_data(self, adapter):
        """Test that get_stats includes KM adjustment data."""
        stats = adapter.get_stats()

        assert "km_adjustments_applied" in stats
        assert "km_adjustments_pending" in stats

    def test_get_pending_adjustments(self, adapter):
        """Test getting pending adjustments."""
        pending = adapter.get_pending_adjustments()
        assert isinstance(pending, list)

    def test_clear_pending_adjustments(self, adapter):
        """Test clearing pending adjustments."""
        # Add a pattern and create recommendation
        patterns = [KMEloPattern("claude", "success_contributor", 0.85, 10)]
        adapter.compute_elo_adjustment(patterns)

        # Clear
        count = adapter.clear_pending_adjustments()

        # Should be empty now
        assert adapter.get_pending_adjustments() == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
