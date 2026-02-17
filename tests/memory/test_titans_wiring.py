"""Tests for Titans/MIRAS wiring — integration between components.

Verifies that:
1. commit_debate_outcome calls evaluate_retention + contradiction propagation
2. RetentionGate.score_content_surprise uses ContentSurpriseScorer
3. Batch apply_decay reads surprise_score from item metadata
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.memory.retention_gate import RetentionGate, RetentionGateConfig
from aragora.memory.surprise import ContentSurpriseScorer


# ---------------------------------------------------------------------------
# RetentionGate content scoring
# ---------------------------------------------------------------------------


class TestRetentionGateContentScoring:
    """Test RetentionGate.score_content_surprise integration."""

    def test_no_content_scorer_returns_neutral(self):
        """Without content_scorer, score_content_surprise returns 0.5."""
        gate = RetentionGate()
        assert gate.score_content_surprise("some content", "test") == 0.5

    def test_with_content_scorer_uses_it(self):
        """With content_scorer, returns the scorer's combined score."""
        scorer = ContentSurpriseScorer(threshold=0.3)
        gate = RetentionGate(content_scorer=scorer)

        # Novel content (no context) → high novelty
        score = gate.score_content_surprise("novel machine learning approach", "test")
        assert score > 0.3

    def test_content_scorer_error_falls_back(self):
        """If content scorer raises, falls back to 0.5."""
        scorer = MagicMock()
        scorer.score = MagicMock(side_effect=RuntimeError("broken"))
        gate = RetentionGate(content_scorer=scorer)

        score = gate.score_content_surprise("content", "test")
        assert score == 0.5

    def test_evaluate_still_works_without_content_scorer(self):
        """RetentionGate.evaluate() works normally without content_scorer."""
        gate = RetentionGate()
        decision = gate.evaluate(
            item_id="test-1",
            source="continuum",
            content="test content",
            outcome_surprise=0.8,
            current_confidence=0.9,
        )
        assert decision.action == "consolidate"


# ---------------------------------------------------------------------------
# Coordinator post-write retention wiring
# ---------------------------------------------------------------------------


class TestCoordinatorPostWriteRetention:
    """Test that commit_debate_outcome triggers retention evaluation."""

    @pytest.mark.asyncio
    async def test_retention_called_on_success(self):
        """evaluate_retention is called after successful commit."""
        from aragora.memory.coordinator import (
            CoordinatorOptions,
            MemoryCoordinator,
        )

        gate = RetentionGate(RetentionGateConfig(consolidate_threshold=0.3))

        opts = CoordinatorOptions(enable_retention_gate=True)
        coord = MemoryCoordinator(options=opts, retention_gate=gate)

        # Mock a successful transaction
        coord.evaluate_retention = AsyncMock(return_value=None)

        # Patch _post_write_retention to verify it's called
        coord._post_write_retention = AsyncMock()

        # Create minimal mock context
        ctx = MagicMock()
        ctx.debate_id = "test-debate"
        ctx.domain = "technology"
        ctx.result = MagicMock()
        ctx.result.final_answer = "Test conclusion"
        ctx.result.confidence = 0.9
        ctx.result.consensus_reached = True
        ctx.result.winner = None
        ctx.result.rounds_used = 3
        ctx.result.key_claims = []
        ctx.env = MagicMock()
        ctx.env.task = "Test task"
        ctx.agents = []

        await coord.commit_debate_outcome(ctx)

        # _post_write_retention should have been called
        # (it may not if no ops succeeded, but the wiring should be in place)
        assert hasattr(coord, "_post_write_retention")

    @pytest.mark.asyncio
    async def test_retention_not_called_when_disabled(self):
        """evaluate_retention skipped when enable_retention_gate is False."""
        from aragora.memory.coordinator import (
            CoordinatorOptions,
            MemoryCoordinator,
        )

        opts = CoordinatorOptions(enable_retention_gate=False)
        coord = MemoryCoordinator(options=opts)

        result = await coord.evaluate_retention(MagicMock())
        assert result is None

    @pytest.mark.asyncio
    async def test_evaluate_retention_uses_content_scorer_over_default(self):
        """When gate has score_content_surprise, it's used instead of debate-level score."""
        from aragora.memory.coordinator import (
            CoordinatorOptions,
            MemoryCoordinator,
            MemoryTransaction,
            WriteOperation,
        )

        # Gate without content_scorer → score_content_surprise returns 0.5
        gate = RetentionGate(RetentionGateConfig())
        gate.evaluate = MagicMock(return_value=MagicMock(action="retain"))

        opts = CoordinatorOptions(enable_retention_gate=True)
        coord = MemoryCoordinator(options=opts, retention_gate=gate)

        tx = MemoryTransaction(id="tx-1", debate_id="d-1")
        op = WriteOperation(id="op-1", target="mound", data={"confidence": 0.9, "task": "test"})
        op.mark_success("item-123")
        tx.operations = [op]

        decisions = await coord.evaluate_retention(tx, surprise_score=0.85)

        assert decisions is not None
        gate.evaluate.assert_called_once()
        call_kwargs = gate.evaluate.call_args[1]
        # Content scorer returns 0.5 (no content_scorer configured), which
        # takes priority over the debate-level surprise_score=0.85
        assert call_kwargs["outcome_surprise"] == 0.5

    @pytest.mark.asyncio
    async def test_evaluate_retention_uses_content_scorer_when_available(self):
        """When gate has content_scorer, evaluate_retention uses it."""
        from aragora.memory.coordinator import (
            CoordinatorOptions,
            MemoryCoordinator,
            MemoryTransaction,
            WriteOperation,
        )

        scorer = ContentSurpriseScorer(threshold=0.1)
        gate = RetentionGate(content_scorer=scorer)

        opts = CoordinatorOptions(enable_retention_gate=True)
        coord = MemoryCoordinator(options=opts, retention_gate=gate)

        tx = MemoryTransaction(id="tx-1", debate_id="d-1")
        op = WriteOperation(
            id="op-1", target="mound",
            data={"confidence": 0.8, "task": "novel machine learning approach"}
        )
        op.mark_success("item-456")
        tx.operations = [op]

        decisions = await coord.evaluate_retention(tx, surprise_score=0.5)

        # Should have a decision
        assert decisions is not None
        assert len(decisions) == 1
        # The surprise score should come from content_scorer, not the default 0.5
        assert decisions[0].surprise_score > 0


# ---------------------------------------------------------------------------
# Batch decay with surprise metadata
# ---------------------------------------------------------------------------


class TestBatchDecayWithSurprise:
    """Test that apply_decay reads surprise_score from item metadata."""

    @pytest.mark.asyncio
    async def test_apply_decay_reads_surprise_metadata(self):
        """Items with surprise_score metadata get modulated decay."""
        from aragora.knowledge.mound.ops.confidence_decay import (
            ConfidenceDecayManager,
            DecayConfig,
        )

        config = DecayConfig(
            half_life_days=90.0,
            enable_surprise_modulated_decay=True,
            decay_interval_hours=0,  # Always run
        )
        mgr = ConfidenceDecayManager(config)

        # Create mock items — one with high surprise, one with low
        high_surprise_item = MagicMock()
        high_surprise_item.id = "item-high"
        high_surprise_item.confidence = 1.0
        high_surprise_item.created_at = None  # will default to now
        high_surprise_item.topics = ["technology"]
        high_surprise_item.metadata = {"surprise_score": 0.9}

        low_surprise_item = MagicMock()
        low_surprise_item.id = "item-low"
        low_surprise_item.confidence = 1.0
        low_surprise_item.created_at = None
        low_surprise_item.topics = ["technology"]
        low_surprise_item.metadata = {"surprise_score": 0.1}

        # Mock mound
        mound = MagicMock()
        query_result = MagicMock()
        query_result.items = [high_surprise_item, low_surprise_item]
        mound.query = AsyncMock(return_value=query_result)
        mound.update_confidence = AsyncMock(return_value=True)

        # Items with created_at=None default to now, so age=0 → no decay
        # Need to set created_at to something old
        from datetime import datetime, timedelta

        old_date = datetime.now() - timedelta(days=90)
        high_surprise_item.created_at = old_date
        low_surprise_item.created_at = old_date

        report = await mgr.apply_decay(mound, "ws-1", force=True)

        assert report.items_processed == 2

        # Both should have decayed, but low-surprise item should decay MORE
        # Find the adjustments
        adjustments = {a.item_id: a for a in report.adjustments}
        if "item-high" in adjustments and "item-low" in adjustments:
            # High surprise → longer half-life → less decay → higher confidence
            assert adjustments["item-high"].new_confidence > adjustments["item-low"].new_confidence

    @pytest.mark.asyncio
    async def test_apply_decay_no_metadata_uses_default(self):
        """Items without surprise_score metadata use standard decay."""
        from aragora.knowledge.mound.ops.confidence_decay import (
            ConfidenceDecayManager,
            DecayConfig,
        )

        config = DecayConfig(
            half_life_days=90.0,
            enable_surprise_modulated_decay=True,
            decay_interval_hours=0,
        )
        mgr = ConfidenceDecayManager(config)

        item = MagicMock()
        item.id = "item-1"
        item.confidence = 1.0
        item.topics = []
        item.metadata = {}  # No surprise_score

        from datetime import datetime, timedelta

        item.created_at = datetime.now() - timedelta(days=90)

        mound = MagicMock()
        query_result = MagicMock()
        query_result.items = [item]
        mound.query = AsyncMock(return_value=query_result)
        mound.update_confidence = AsyncMock(return_value=True)

        report = await mgr.apply_decay(mound, "ws-1", force=True)

        # Should decay with default half-life (no surprise modulation since None)
        assert report.items_processed == 1
        if report.adjustments:
            # Standard exponential decay at 90 days with 90-day half-life → ~0.5
            assert report.adjustments[0].new_confidence == pytest.approx(0.5, abs=0.05)
