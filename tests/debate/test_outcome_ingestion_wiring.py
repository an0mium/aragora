"""Tests for debate outcome ingestion to KnowledgeMound.

Verifies that:
1. Outcome ingestion is called after successful debate when KM is configured
2. Ingestion failure does not affect debate result
3. Ingestion is skipped when KM is not configured
4. Ingestion is skipped when confidence is below threshold
5. Ingestion is opt-in via PostDebateConfig.auto_ingest_outcome
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.debate.post_debate_coordinator import (
    PostDebateConfig,
    PostDebateCoordinator,
)


def _make_debate_result(
    confidence: float = 0.9,
    final_answer: str = "The consensus conclusion",
    consensus_reached: bool = True,
) -> MagicMock:
    """Create a mock debate result."""
    result = MagicMock()
    result.confidence = confidence
    result.final_answer = final_answer
    result.consensus_reached = consensus_reached
    result.id = "debate-123"
    result.rounds_used = 3
    result.participants = ["claude", "gpt4"]
    result.winner = "claude"
    result.messages = []
    result.debate_cruxes = None
    result.metadata = {}
    return result


class TestOutcomeIngestionInCoordinator:
    """Test outcome ingestion step in PostDebateCoordinator."""

    def test_ingestion_called_when_km_configured(self):
        """Outcome ingestion fires when knowledge_mound is provided."""
        km = MagicMock()
        km.workspace_id = "default"
        km.store = AsyncMock(return_value=MagicMock(node_id="node-1", success=True))

        config = PostDebateConfig(
            auto_explain=False,
            auto_create_plan=False,
            auto_notify=False,
            auto_execute_plan=False,
            auto_persist_receipt=False,
            auto_gauntlet_validate=False,
            auto_queue_improvement=False,
            auto_outcome_feedback=False,
            auto_trigger_canvas=False,
            auto_execution_bridge=False,
            enforce_execution_safety_gate=False,
            auto_ingest_outcome=True,
        )
        coordinator = PostDebateCoordinator(config=config, knowledge_mound=km)

        result = coordinator.run(
            debate_id="d-1",
            debate_result=_make_debate_result(),
            confidence=0.9,
            task="Design a rate limiter",
        )

        assert result.outcome_ingested is True

    def test_ingestion_skipped_when_km_not_configured(self):
        """Outcome ingestion is skipped when no knowledge_mound."""
        config = PostDebateConfig(
            auto_explain=False,
            auto_create_plan=False,
            auto_notify=False,
            auto_execute_plan=False,
            auto_persist_receipt=False,
            auto_gauntlet_validate=False,
            auto_queue_improvement=False,
            auto_outcome_feedback=False,
            auto_trigger_canvas=False,
            auto_execution_bridge=False,
            enforce_execution_safety_gate=False,
            auto_ingest_outcome=True,
        )
        coordinator = PostDebateCoordinator(config=config, knowledge_mound=None)

        result = coordinator.run(
            debate_id="d-2",
            debate_result=_make_debate_result(),
            confidence=0.9,
            task="Test task",
        )

        assert result.outcome_ingested is False

    def test_ingestion_skipped_when_disabled(self):
        """Outcome ingestion is skipped when auto_ingest_outcome=False."""
        km = MagicMock()
        config = PostDebateConfig(
            auto_explain=False,
            auto_create_plan=False,
            auto_notify=False,
            auto_execute_plan=False,
            auto_persist_receipt=False,
            auto_gauntlet_validate=False,
            auto_queue_improvement=False,
            auto_outcome_feedback=False,
            auto_trigger_canvas=False,
            auto_execution_bridge=False,
            enforce_execution_safety_gate=False,
            auto_ingest_outcome=False,
        )
        coordinator = PostDebateCoordinator(config=config, knowledge_mound=km)

        result = coordinator.run(
            debate_id="d-3",
            debate_result=_make_debate_result(),
            confidence=0.9,
            task="Test task",
        )

        assert result.outcome_ingested is False

    def test_ingestion_skipped_below_confidence_threshold(self):
        """Outcome ingestion is skipped when confidence < threshold."""
        km = MagicMock()
        config = PostDebateConfig(
            auto_explain=False,
            auto_create_plan=False,
            auto_notify=False,
            auto_execute_plan=False,
            auto_persist_receipt=False,
            auto_gauntlet_validate=False,
            auto_queue_improvement=False,
            auto_outcome_feedback=False,
            auto_trigger_canvas=False,
            auto_execution_bridge=False,
            enforce_execution_safety_gate=False,
            auto_ingest_outcome=True,
            ingest_outcome_min_confidence=0.85,
        )
        coordinator = PostDebateCoordinator(config=config, knowledge_mound=km)

        result = coordinator.run(
            debate_id="d-4",
            debate_result=_make_debate_result(confidence=0.5),
            confidence=0.5,
            task="Low confidence task",
        )

        assert result.outcome_ingested is False

    def test_ingestion_failure_does_not_affect_result(self):
        """Ingestion error is caught and does not block pipeline."""
        km = MagicMock()
        km.workspace_id = "default"
        km.store = AsyncMock(side_effect=RuntimeError("KM unavailable"))

        config = PostDebateConfig(
            auto_explain=False,
            auto_create_plan=False,
            auto_notify=False,
            auto_execute_plan=False,
            auto_persist_receipt=False,
            auto_gauntlet_validate=False,
            auto_queue_improvement=False,
            auto_outcome_feedback=False,
            auto_trigger_canvas=False,
            auto_execution_bridge=False,
            enforce_execution_safety_gate=False,
            auto_ingest_outcome=True,
        )
        coordinator = PostDebateCoordinator(config=config, knowledge_mound=km)

        result = coordinator.run(
            debate_id="d-5",
            debate_result=_make_debate_result(),
            confidence=0.9,
            task="Error task",
        )

        # Pipeline completes successfully even if ingestion fails
        assert result.success is True

    def test_config_defaults(self):
        """auto_ingest_outcome defaults to True."""
        config = PostDebateConfig()
        assert config.auto_ingest_outcome is True
        assert config.ingest_outcome_min_confidence == 0.85


class TestOutcomeIngestionOps:
    """Test KnowledgeMoundOperations.ingest_debate_outcome directly."""

    @pytest.mark.asyncio
    async def test_ingest_stores_high_confidence_result(self):
        """High-confidence results are stored in KM."""
        from aragora.debate.knowledge_mound_ops import KnowledgeMoundOperations

        km = MagicMock()
        km.workspace_id = "ws-1"
        km.store = AsyncMock(return_value=MagicMock(node_id="n-1", success=True))

        ops = KnowledgeMoundOperations(knowledge_mound=km, enable_ingestion=True)
        result = _make_debate_result(confidence=0.9)

        await ops.ingest_debate_outcome(result)

        km.store.assert_called_once()
        call_args = km.store.call_args
        request = call_args[0][0]
        assert "Debate Conclusion" in request.content

    @pytest.mark.asyncio
    async def test_ingest_skips_low_confidence(self):
        """Low-confidence results are not stored."""
        from aragora.debate.knowledge_mound_ops import KnowledgeMoundOperations

        km = MagicMock()
        km.workspace_id = "ws-1"
        km.store = AsyncMock()

        ops = KnowledgeMoundOperations(knowledge_mound=km, enable_ingestion=True)
        result = _make_debate_result(confidence=0.5)

        await ops.ingest_debate_outcome(result)

        km.store.assert_not_called()

    @pytest.mark.asyncio
    async def test_ingest_skips_no_final_answer(self):
        """Results without final_answer are not stored."""
        from aragora.debate.knowledge_mound_ops import KnowledgeMoundOperations

        km = MagicMock()
        km.workspace_id = "ws-1"
        km.store = AsyncMock()

        ops = KnowledgeMoundOperations(knowledge_mound=km, enable_ingestion=True)
        result = _make_debate_result(final_answer="")

        await ops.ingest_debate_outcome(result)

        km.store.assert_not_called()

    @pytest.mark.asyncio
    async def test_ingest_skips_when_disabled(self):
        """Ingestion is skipped when enable_ingestion=False."""
        from aragora.debate.knowledge_mound_ops import KnowledgeMoundOperations

        km = MagicMock()
        km.workspace_id = "ws-1"
        km.store = AsyncMock()

        ops = KnowledgeMoundOperations(knowledge_mound=km, enable_ingestion=False)
        result = _make_debate_result(confidence=0.9)

        await ops.ingest_debate_outcome(result)

        km.store.assert_not_called()

    @pytest.mark.asyncio
    async def test_ingest_skips_when_no_km(self):
        """Ingestion is skipped when knowledge_mound is None."""
        from aragora.debate.knowledge_mound_ops import KnowledgeMoundOperations

        ops = KnowledgeMoundOperations(knowledge_mound=None, enable_ingestion=True)
        result = _make_debate_result(confidence=0.9)

        await ops.ingest_debate_outcome(result)
        # No error, just silently skipped

    @pytest.mark.asyncio
    async def test_ingest_handles_store_error_gracefully(self):
        """Store errors are caught and logged, not raised."""
        from aragora.debate.knowledge_mound_ops import KnowledgeMoundOperations

        km = MagicMock()
        km.workspace_id = "ws-1"
        km.store = AsyncMock(side_effect=RuntimeError("DB down"))

        ops = KnowledgeMoundOperations(knowledge_mound=km, enable_ingestion=True)
        result = _make_debate_result(confidence=0.9)

        # Should not raise
        await ops.ingest_debate_outcome(result)
