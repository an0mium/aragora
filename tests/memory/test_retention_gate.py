"""Tests for the Titans/MIRAS-inspired Retention Gate.

Covers RetentionGate evaluation logic, adaptive decay rates,
batch processing, TierManager integration, ConfidenceDecayManager
integration, and MemoryCoordinator wiring.
"""

from __future__ import annotations

import pytest

from aragora.memory.retention_gate import (
    RetentionDecision,
    RetentionGate,
    RetentionGateConfig,
)
from aragora.memory.surprise import SurpriseScorer
from aragora.memory.tier_manager import MemoryTier, TierManager
from aragora.memory.coordinator import CoordinatorOptions, MemoryCoordinator


# ---------------------------------------------------------------------------
# RetentionGateConfig defaults
# ---------------------------------------------------------------------------


class TestRetentionGateConfig:
    """Test RetentionGateConfig dataclass defaults."""

    def test_defaults(self):
        """Config should default to opt-in disabled with sane thresholds."""
        cfg = RetentionGateConfig()
        assert cfg.enable_surprise_driven_decay is False
        assert cfg.forget_threshold == 0.15
        assert cfg.consolidate_threshold == 0.7
        assert cfg.adaptive_decay_enabled is True
        assert cfg.max_decay_rate == 0.01
        assert cfg.min_decay_rate == 0.0001
        assert cfg.red_line_protection is True


# ---------------------------------------------------------------------------
# RetentionDecision dataclass
# ---------------------------------------------------------------------------


class TestRetentionDecision:
    """Test RetentionDecision dataclass fields."""

    def test_fields(self):
        """RetentionDecision stores all required fields."""
        d = RetentionDecision(
            item_id="abc",
            source_system="km",
            surprise_score=0.8,
            retention_score=0.6,
            action="consolidate",
            decay_rate_override=0.002,
            reason="test reason",
        )
        assert d.item_id == "abc"
        assert d.source_system == "km"
        assert d.surprise_score == 0.8
        assert d.retention_score == 0.6
        assert d.action == "consolidate"
        assert d.decay_rate_override == 0.002
        assert d.reason == "test reason"

    def test_default_decay_override_is_none(self):
        """decay_rate_override defaults to None."""
        d = RetentionDecision(
            item_id="x",
            source_system="continuum",
            surprise_score=0.5,
            retention_score=0.5,
            action="retain",
        )
        assert d.decay_rate_override is None
        assert d.reason == ""


# ---------------------------------------------------------------------------
# RetentionGate.evaluate()
# ---------------------------------------------------------------------------


class TestRetentionGateEvaluate:
    """Test evaluate() for different surprise/confidence combinations."""

    def test_high_surprise_consolidate(self):
        """High surprise should produce 'consolidate' action."""
        gate = RetentionGate()
        decision = gate.evaluate(
            item_id="item1",
            source="km",
            content="novel discovery",
            outcome_surprise=0.85,
            current_confidence=0.6,
        )
        assert decision.action == "consolidate"
        assert decision.surprise_score == 0.85
        assert "High surprise" in decision.reason

    def test_low_surprise_low_confidence_forget(self):
        """Low surprise + low confidence should produce 'forget' action."""
        gate = RetentionGate()
        decision = gate.evaluate(
            item_id="item2",
            source="continuum",
            content="routine data",
            outcome_surprise=0.1,
            current_confidence=0.2,
        )
        assert decision.action == "forget"
        assert "Low surprise" in decision.reason
        assert "low confidence" in decision.reason

    def test_low_surprise_ok_confidence_demote(self):
        """Low surprise + adequate confidence should produce 'demote' action."""
        gate = RetentionGate()
        decision = gate.evaluate(
            item_id="item3",
            source="km",
            content="predictable info",
            outcome_surprise=0.1,
            current_confidence=0.5,
        )
        assert decision.action == "demote"
        assert "Low surprise" in decision.reason

    def test_medium_surprise_retain(self):
        """Medium surprise should produce 'retain' action."""
        gate = RetentionGate()
        decision = gate.evaluate(
            item_id="item4",
            source="supermemory",
            content="moderate content",
            outcome_surprise=0.4,
            current_confidence=0.6,
        )
        assert decision.action == "retain"
        assert "normal range" in decision.reason

    def test_red_line_protection_always_retain(self):
        """Red-line entries should always be retained regardless of surprise."""
        gate = RetentionGate()
        decision = gate.evaluate(
            item_id="critical1",
            source="km",
            content="critical policy",
            outcome_surprise=0.01,
            current_confidence=0.1,
            is_red_line=True,
        )
        assert decision.action == "retain"
        assert decision.retention_score == 1.0
        assert "Red-line" in decision.reason

    def test_red_line_disabled_in_config(self):
        """When red_line_protection is off, red_line flag is ignored."""
        cfg = RetentionGateConfig(red_line_protection=False)
        gate = RetentionGate(config=cfg)
        decision = gate.evaluate(
            item_id="critical2",
            source="km",
            content="critical policy",
            outcome_surprise=0.05,
            current_confidence=0.1,
            is_red_line=True,
        )
        # Should be forget since surprise and confidence are low
        assert decision.action == "forget"

    def test_retention_score_bounded(self):
        """Retention score should be clamped between 0 and 1."""
        gate = RetentionGate()
        decision = gate.evaluate(
            item_id="bounded",
            source="km",
            content="test",
            outcome_surprise=1.0,
            current_confidence=1.0,
            access_count=100,
        )
        assert 0.0 <= decision.retention_score <= 1.0

    def test_access_count_bonus(self):
        """Access count should contribute up to 0.2 bonus to retention score."""
        gate = RetentionGate()
        d_no_access = gate.evaluate(
            item_id="a1",
            source="km",
            content="",
            outcome_surprise=0.4,
            current_confidence=0.5,
        )
        gate.clear_decisions()
        d_high_access = gate.evaluate(
            item_id="a2",
            source="km",
            content="",
            outcome_surprise=0.4,
            current_confidence=0.5,
            access_count=20,
        )
        assert d_high_access.retention_score > d_no_access.retention_score

    def test_decisions_are_tracked(self):
        """Each evaluate call should append to internal decisions list."""
        gate = RetentionGate()
        gate.evaluate("i1", "km", "", 0.5, 0.5)
        gate.evaluate("i2", "km", "", 0.8, 0.5)
        assert len(gate.get_decisions()) == 2


# ---------------------------------------------------------------------------
# RetentionGate.compute_adaptive_decay_rate()
# ---------------------------------------------------------------------------


class TestAdaptiveDecayRate:
    """Test compute_adaptive_decay_rate calculations."""

    def test_high_surprise_low_decay(self):
        """High surprise should produce low decay rate (preserve longer)."""
        gate = RetentionGate()
        rate = gate.compute_adaptive_decay_rate(surprise=1.0, current_confidence=0.5)
        assert rate == gate.config.min_decay_rate

    def test_low_surprise_high_decay(self):
        """Low surprise should produce high decay rate (forget faster)."""
        gate = RetentionGate()
        rate = gate.compute_adaptive_decay_rate(surprise=0.0, current_confidence=0.5)
        assert rate == gate.config.max_decay_rate

    def test_mid_surprise_mid_decay(self):
        """Medium surprise should produce a decay rate between min and max."""
        gate = RetentionGate()
        rate = gate.compute_adaptive_decay_rate(surprise=0.5, current_confidence=0.5)
        assert gate.config.min_decay_rate < rate < gate.config.max_decay_rate

    def test_decay_rate_with_adaptive_disabled(self):
        """When adaptive decay is disabled, evaluate should not set override."""
        cfg = RetentionGateConfig(adaptive_decay_enabled=False)
        gate = RetentionGate(config=cfg)
        decision = gate.evaluate("x", "km", "", 0.5, 0.5)
        assert decision.decay_rate_override is None


# ---------------------------------------------------------------------------
# RetentionGate.batch_evaluate()
# ---------------------------------------------------------------------------


class TestBatchEvaluate:
    """Test batch_evaluate processes multiple items."""

    def test_batch_processes_all_items(self):
        """batch_evaluate should return one decision per item."""
        gate = RetentionGate()
        items = [
            {"item_id": "b1", "source": "km", "outcome_surprise": 0.9, "current_confidence": 0.5},
            {
                "item_id": "b2",
                "source": "continuum",
                "outcome_surprise": 0.1,
                "current_confidence": 0.2,
            },
            {"item_id": "b3", "source": "km", "outcome_surprise": 0.4, "current_confidence": 0.6},
        ]
        decisions = gate.batch_evaluate(items)
        assert len(decisions) == 3
        assert decisions[0].action == "consolidate"
        assert decisions[1].action == "forget"
        assert decisions[2].action == "retain"

    def test_batch_with_red_line(self):
        """batch_evaluate should respect red_line flag in item dicts."""
        gate = RetentionGate()
        items = [
            {
                "item_id": "r1",
                "source": "km",
                "outcome_surprise": 0.01,
                "current_confidence": 0.1,
                "is_red_line": True,
            },
        ]
        decisions = gate.batch_evaluate(items)
        assert decisions[0].action == "retain"
        assert decisions[0].retention_score == 1.0


# ---------------------------------------------------------------------------
# RetentionGate.get_stats() and clear_decisions()
# ---------------------------------------------------------------------------


class TestRetentionGateStats:
    """Test stats tracking and clearing."""

    def test_get_stats_counts(self):
        """get_stats should return correct action counts."""
        gate = RetentionGate()
        gate.evaluate("s1", "km", "", 0.9, 0.5)  # consolidate
        gate.evaluate("s2", "km", "", 0.1, 0.2)  # forget
        gate.evaluate("s3", "km", "", 0.4, 0.5)  # retain

        stats = gate.get_stats()
        assert stats["total_decisions"] == 3
        assert stats["by_action"]["consolidate"] == 1
        assert stats["by_action"]["forget"] == 1
        assert stats["by_action"]["retain"] == 1

    def test_clear_decisions(self):
        """clear_decisions should empty the history."""
        gate = RetentionGate()
        gate.evaluate("c1", "km", "", 0.5, 0.5)
        assert len(gate.get_decisions()) == 1
        gate.clear_decisions()
        assert len(gate.get_decisions()) == 0
        assert gate.get_stats()["total_decisions"] == 0


# ---------------------------------------------------------------------------
# SurpriseScorer -> RetentionGate integration
# ---------------------------------------------------------------------------


class TestSurpriseScorerIntegration:
    """Test that SurpriseScorer can be injected into RetentionGate."""

    def test_custom_scorer(self):
        """RetentionGate should use the provided SurpriseScorer."""
        scorer = SurpriseScorer(alpha=0.5)
        gate = RetentionGate(scorer=scorer)
        assert gate.scorer is scorer
        assert gate.scorer.alpha == 0.5

    def test_default_scorer_created(self):
        """Without a scorer, RetentionGate creates one with config alpha."""
        cfg = RetentionGateConfig(surprise_alpha=0.7)
        gate = RetentionGate(config=cfg)
        assert gate.scorer.alpha == 0.7


# ---------------------------------------------------------------------------
# TierManager.apply_retention_decision()
# ---------------------------------------------------------------------------


class TestTierManagerRetention:
    """Test TierManager.apply_retention_decision for each action."""

    def setup_method(self):
        self.tm = TierManager()

    def test_consolidate_promotes(self):
        """Consolidate should promote to faster tier."""
        new_tier, reason = self.tm.apply_retention_decision(MemoryTier.MEDIUM, "consolidate")
        assert new_tier == MemoryTier.FAST
        assert "promoted" in reason.lower()

    def test_consolidate_at_fastest(self):
        """Consolidate at fastest tier should return None."""
        new_tier, reason = self.tm.apply_retention_decision(MemoryTier.FAST, "consolidate")
        assert new_tier is None
        assert "fastest" in reason.lower()

    def test_demote_demotes(self):
        """Demote should move to slower tier."""
        new_tier, reason = self.tm.apply_retention_decision(MemoryTier.MEDIUM, "demote")
        assert new_tier == MemoryTier.SLOW
        assert "demoted" in reason.lower()

    def test_demote_at_slowest(self):
        """Demote at slowest tier should return None."""
        new_tier, reason = self.tm.apply_retention_decision(MemoryTier.GLACIAL, "demote")
        assert new_tier is None
        assert "slowest" in reason.lower()

    def test_forget_goes_to_glacial(self):
        """Forget should send entry to glacial tier."""
        new_tier, reason = self.tm.apply_retention_decision(MemoryTier.FAST, "forget")
        assert new_tier == MemoryTier.GLACIAL
        assert "forgotten" in reason.lower()

    def test_forget_already_glacial(self):
        """Forget at glacial should return None."""
        new_tier, reason = self.tm.apply_retention_decision(MemoryTier.GLACIAL, "forget")
        assert new_tier is None
        assert "glacial" in reason.lower()

    def test_retain_no_change(self):
        """Retain should not change tier."""
        new_tier, reason = self.tm.apply_retention_decision(MemoryTier.MEDIUM, "retain")
        assert new_tier is None
        assert "retained" in reason.lower()

    def test_consolidate_records_promotion_metric(self):
        """Consolidate should record a promotion metric."""
        self.tm.reset_metrics()
        self.tm.apply_retention_decision(MemoryTier.SLOW, "consolidate")
        metrics = self.tm.get_metrics_dict()
        assert metrics["total_promotions"] == 1

    def test_forget_records_demotion_metric(self):
        """Forget should record a demotion metric."""
        self.tm.reset_metrics()
        self.tm.apply_retention_decision(MemoryTier.FAST, "forget")
        metrics = self.tm.get_metrics_dict()
        assert metrics["total_demotions"] == 1


# ---------------------------------------------------------------------------
# ConfidenceDecayManager.apply_surprise_driven_decay()
# ---------------------------------------------------------------------------


class TestConfidenceDecayManagerSurpriseDriven:
    """Test apply_surprise_driven_decay with mocked mound."""

    @pytest.mark.asyncio
    async def test_forget_sets_min_confidence(self):
        """Forget action should set confidence to min_confidence."""
        from aragora.knowledge.mound.ops.confidence_decay import (
            ConfidenceDecayManager,
            DecayConfig,
        )

        class FakeItem:
            def __init__(self, confidence):
                self.confidence = confidence

        class FakeMound:
            def __init__(self):
                self._updated = {}

            async def get(self, node_id):
                return FakeItem(confidence=0.8)

            async def update_confidence(self, node_id, new_confidence):
                self._updated[node_id] = new_confidence
                return True

        mound = FakeMound()
        manager = ConfidenceDecayManager(config=DecayConfig(min_confidence=0.1))

        decision = RetentionDecision(
            item_id="forget1",
            source_system="km",
            surprise_score=0.05,
            retention_score=0.1,
            action="forget",
            reason="test forget",
        )

        report = await manager.apply_surprise_driven_decay(mound, "ws1", [decision])
        assert report.items_decayed == 1
        assert report.items_boosted == 0
        assert mound._updated["forget1"] == 0.1

    @pytest.mark.asyncio
    async def test_consolidate_boosts_confidence(self):
        """Consolidate action should boost confidence by validation_boost."""
        from aragora.knowledge.mound.ops.confidence_decay import (
            ConfidenceDecayManager,
            DecayConfig,
        )

        class FakeItem:
            def __init__(self, confidence):
                self.confidence = confidence

        class FakeMound:
            def __init__(self):
                self._updated = {}

            async def get(self, node_id):
                return FakeItem(confidence=0.7)

            async def update_confidence(self, node_id, new_confidence):
                self._updated[node_id] = new_confidence
                return True

        mound = FakeMound()
        config = DecayConfig(validation_boost=0.1, max_confidence=1.0)
        manager = ConfidenceDecayManager(config=config)

        decision = RetentionDecision(
            item_id="cons1",
            source_system="km",
            surprise_score=0.9,
            retention_score=0.8,
            action="consolidate",
            reason="test consolidate",
        )

        report = await manager.apply_surprise_driven_decay(mound, "ws1", [decision])
        assert report.items_boosted == 1
        assert mound._updated["cons1"] == pytest.approx(0.8)

    @pytest.mark.asyncio
    async def test_missing_item_skipped(self):
        """Items not found in mound should be skipped gracefully."""
        from aragora.knowledge.mound.ops.confidence_decay import (
            ConfidenceDecayManager,
        )

        class FakeMound:
            async def get(self, node_id):
                return None

        mound = FakeMound()
        manager = ConfidenceDecayManager()

        decision = RetentionDecision(
            item_id="missing1",
            source_system="km",
            surprise_score=0.5,
            retention_score=0.5,
            action="forget",
            reason="test",
        )

        report = await manager.apply_surprise_driven_decay(mound, "ws1", [decision])
        assert report.items_processed == 1
        assert report.items_decayed == 0
        assert report.items_boosted == 0


# ---------------------------------------------------------------------------
# CoordinatorOptions.enable_retention_gate
# ---------------------------------------------------------------------------


class TestCoordinatorOptionsRetention:
    """Test CoordinatorOptions retention gate field."""

    def test_default_disabled(self):
        """enable_retention_gate should default to False."""
        opts = CoordinatorOptions()
        assert opts.enable_retention_gate is False


# ---------------------------------------------------------------------------
# MemoryCoordinator.evaluate_retention()
# ---------------------------------------------------------------------------


class TestMemoryCoordinatorRetention:
    """Test MemoryCoordinator.evaluate_retention method."""

    @pytest.mark.asyncio
    async def test_returns_none_when_gate_not_configured(self):
        """Should return None when no retention gate is set."""
        from aragora.memory.coordinator import MemoryTransaction

        coord = MemoryCoordinator()
        txn = MemoryTransaction(id="t1", debate_id="d1")
        result = await coord.evaluate_retention(txn)
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_gate_disabled(self):
        """Should return None when gate exists but option is disabled."""
        from aragora.memory.coordinator import MemoryTransaction

        gate = RetentionGate()
        coord = MemoryCoordinator(
            retention_gate=gate,
            options=CoordinatorOptions(enable_retention_gate=False),
        )
        txn = MemoryTransaction(id="t2", debate_id="d2")
        result = await coord.evaluate_retention(txn)
        assert result is None

    @pytest.mark.asyncio
    async def test_produces_decisions_for_successful_ops(self):
        """Should produce RetentionDecision for successful continuum/mound ops."""
        from aragora.memory.coordinator import (
            MemoryTransaction,
            WriteOperation,
            WriteStatus,
        )

        gate = RetentionGate()
        coord = MemoryCoordinator(
            retention_gate=gate,
            options=CoordinatorOptions(enable_retention_gate=True),
        )

        txn = MemoryTransaction(id="t3", debate_id="d3")
        op = WriteOperation(
            id="op1",
            target="mound",
            status=WriteStatus.SUCCESS,
            result="item-123",
            data={"confidence": 0.8, "task": "Test debate task"},
        )
        txn.operations.append(op)

        decisions = await coord.evaluate_retention(txn)
        assert decisions is not None
        assert len(decisions) == 1
        assert decisions[0].item_id == "item-123"
        assert decisions[0].source_system == "mound"

    @pytest.mark.asyncio
    async def test_skips_non_memory_targets(self):
        """Should skip operations targeting critique or consensus."""
        from aragora.memory.coordinator import (
            MemoryTransaction,
            WriteOperation,
            WriteStatus,
        )

        gate = RetentionGate()
        coord = MemoryCoordinator(
            retention_gate=gate,
            options=CoordinatorOptions(enable_retention_gate=True),
        )

        txn = MemoryTransaction(id="t4", debate_id="d4")
        op = WriteOperation(
            id="op2",
            target="critique",
            status=WriteStatus.SUCCESS,
            result="crit-1",
            data={"confidence": 0.9},
        )
        txn.operations.append(op)

        decisions = await coord.evaluate_retention(txn)
        assert decisions is None
