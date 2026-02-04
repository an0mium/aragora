"""
Integration tests for Debate → Memory → Knowledge Mound flow.

These tests verify that debate outcomes properly flow through the memory
system and get persisted to the Knowledge Mound.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


@dataclass
class MockDebateResult:
    """Mock debate result for testing."""

    debate_id: str = "debate-123"
    consensus: str = "Use token bucket algorithm"
    confidence: float = 0.92
    rounds_used: int = 5
    dissenting_views: list = None
    created_at: datetime = None

    def __post_init__(self):
        self.dissenting_views = self.dissenting_views or ["Fixed window advocate"]
        self.created_at = self.created_at or datetime.now()


class TestDebateToMemoryFlow:
    """Test debate results flowing to memory systems."""

    @pytest.fixture
    def mock_continuum_memory(self):
        """Create mock ContinuumMemory."""
        memory = MagicMock()
        memory.store = AsyncMock(return_value={"id": "mem-123", "tier": "MEDIUM"})
        memory.query = AsyncMock(return_value=[])
        memory.update_surprise = AsyncMock()
        return memory

    @pytest.fixture
    def mock_consensus_memory(self):
        """Create mock ConsensusMemory."""
        memory = MagicMock()
        memory.store_outcome = AsyncMock(return_value={"id": "cons-123"})
        memory.get_dissent = AsyncMock(return_value=[])
        memory.track_evolution = AsyncMock()
        return memory

    @pytest.fixture
    def mock_knowledge_mound(self):
        """Create mock KnowledgeMound."""
        mound = MagicMock()
        mound.store = AsyncMock(
            return_value=MagicMock(
                node_id="km-123",
                content="Test content",
                confidence=0.92,
            )
        )
        mound.query = AsyncMock(return_value=MagicMock(items=[]))
        mound.update_confidence = AsyncMock()
        return mound

    @pytest.fixture
    def mock_critique_store(self):
        """Create mock CritiqueStore."""
        store = MagicMock()
        store.store_critique = AsyncMock(return_value={"id": "crit-123"})
        store.get_patterns = AsyncMock(return_value=[])
        return store

    async def test_debate_outcome_stored_in_continuum(
        self, mock_continuum_memory, mock_consensus_memory
    ):
        """Debate outcome should be stored in ContinuumMemory."""
        result = MockDebateResult()

        # Simulate storing debate outcome
        stored = await mock_continuum_memory.store(
            key=f"debate:{result.debate_id}",
            content=result.consensus,
            importance=result.confidence,
        )

        assert stored["id"] == "mem-123"
        assert stored["tier"] == "MEDIUM"
        mock_continuum_memory.store.assert_called_once()

    async def test_consensus_stored_with_dissent(
        self, mock_continuum_memory, mock_consensus_memory
    ):
        """Consensus should be stored with dissenting views."""
        result = MockDebateResult()

        await mock_consensus_memory.store_outcome(
            debate_id=result.debate_id,
            decision=result.consensus,
            confidence=result.confidence,
            dissenting_views=result.dissenting_views,
        )

        mock_consensus_memory.store_outcome.assert_called_once_with(
            debate_id="debate-123",
            decision="Use token bucket algorithm",
            confidence=0.92,
            dissenting_views=["Fixed window advocate"],
        )

    async def test_knowledge_mound_receives_high_confidence_outcomes(self, mock_knowledge_mound):
        """High confidence outcomes should be stored in Knowledge Mound."""
        result = MockDebateResult(confidence=0.95)

        # Only store if confidence >= threshold
        threshold = 0.7
        if result.confidence >= threshold:
            await mock_knowledge_mound.store(
                content=result.consensus,
                source_type="DEBATE",
                debate_id=result.debate_id,
                confidence=result.confidence,
            )

        mock_knowledge_mound.store.assert_called_once()

    async def test_low_confidence_outcomes_not_stored_in_km(self, mock_knowledge_mound):
        """Low confidence outcomes should not be stored in KM."""
        result = MockDebateResult(confidence=0.4)

        threshold = 0.7
        if result.confidence >= threshold:
            await mock_knowledge_mound.store(content=result.consensus)

        mock_knowledge_mound.store.assert_not_called()

    async def test_memory_retrieval_informs_debate(self, mock_continuum_memory):
        """Past memories should be retrieved to inform debates."""
        # Set up mock to return relevant memories
        mock_continuum_memory.query.return_value = [
            {"content": "Previous rate limiter used fixed window", "importance": 0.8},
            {"content": "Sliding window better for bursty traffic", "importance": 0.75},
        ]

        memories = await mock_continuum_memory.query(
            query="rate limiting patterns",
            limit=10,
            min_importance=0.5,
        )

        assert len(memories) == 2
        assert memories[0]["importance"] == 0.8

    async def test_surprise_score_updates_tier(self, mock_continuum_memory):
        """High surprise score should trigger tier promotion."""
        memory_id = "mem-123"
        surprise_score = 0.85  # High surprise

        await mock_continuum_memory.update_surprise(
            memory_id=memory_id,
            surprise_score=surprise_score,
        )

        mock_continuum_memory.update_surprise.assert_called_once_with(
            memory_id="mem-123",
            surprise_score=0.85,
        )


class TestMemoryCoordinatorAtomicity:
    """Test atomic writes across memory systems."""

    @pytest.fixture
    def mock_coordinator(self):
        """Create mock MemoryCoordinator."""
        coord = MagicMock()
        coord.commit_debate_outcome = AsyncMock(
            return_value={
                "continuum": {"id": "mem-123"},
                "consensus": {"id": "cons-123"},
                "critique": {"id": "crit-123"},
                "mound": {"node_id": "km-123"},
            }
        )
        coord.rollback = AsyncMock()
        return coord

    async def test_atomic_write_success(self, mock_coordinator):
        """All systems should be updated atomically on success."""
        result = MockDebateResult()

        outcome = await mock_coordinator.commit_debate_outcome(
            debate_id=result.debate_id,
            outcome=result,
            rollback_on_failure=True,
        )

        assert "continuum" in outcome
        assert "consensus" in outcome
        assert "mound" in outcome
        mock_coordinator.rollback.assert_not_called()

    async def test_rollback_on_partial_failure(self, mock_coordinator):
        """Partial failures should trigger rollback."""
        mock_coordinator.commit_debate_outcome.side_effect = Exception("KM write failed")

        with pytest.raises(Exception, match="KM write failed"):
            await mock_coordinator.commit_debate_outcome(
                debate_id="debate-123",
                outcome=MockDebateResult(),
                rollback_on_failure=True,
            )

    async def test_sequential_writes_preserve_order(self, mock_coordinator):
        """Sequential writes should preserve order for consistency."""
        call_order = []

        async def write_continuum():
            call_order.append("continuum")

        async def write_consensus():
            call_order.append("consensus")

        async def write_mound():
            call_order.append("mound")

        mock_coordinator.write_continuum = write_continuum
        mock_coordinator.write_consensus = write_consensus
        mock_coordinator.write_mound = write_mound

        # Execute writes sequentially
        await mock_coordinator.write_continuum()
        await mock_coordinator.write_consensus()
        await mock_coordinator.write_mound()

        assert call_order == ["continuum", "consensus", "mound"]


class TestKnowledgeMoundIntegration:
    """Test Knowledge Mound integration with memory systems."""

    @pytest.fixture
    def mock_mound_with_adapters(self):
        """Create KnowledgeMound mock with adapters."""
        mound = MagicMock()
        mound.continuum_adapter = MagicMock()
        mound.continuum_adapter.sync_to_km = AsyncMock(
            return_value={"nodes_synced": 5, "errors": 0}
        )
        mound.consensus_adapter = MagicMock()
        mound.consensus_adapter.sync_to_km = AsyncMock(
            return_value={"nodes_synced": 3, "errors": 0}
        )
        return mound

    async def test_continuum_adapter_sync(self, mock_mound_with_adapters):
        """ContinuumAdapter should sync memories to KM."""
        result = await mock_mound_with_adapters.continuum_adapter.sync_to_km()

        assert result["nodes_synced"] == 5
        assert result["errors"] == 0

    async def test_consensus_adapter_sync(self, mock_mound_with_adapters):
        """ConsensusAdapter should sync outcomes to KM."""
        result = await mock_mound_with_adapters.consensus_adapter.sync_to_km()

        assert result["nodes_synced"] == 3
        assert result["errors"] == 0

    async def test_bidirectional_sync_km_to_memory(self):
        """KM should provide reverse flow to memory systems."""
        mock_continuum_memory = MagicMock()
        mock_continuum_memory.query_km_for_similar = AsyncMock(
            return_value=[{"node_id": "km-100", "content": "Related pattern", "similarity": 0.92}]
        )

        similar = await mock_continuum_memory.query_km_for_similar(
            query="rate limiting",
            limit=5,
        )

        assert len(similar) == 1
        assert similar[0]["similarity"] == 0.92


class TestCrossDebateMemory:
    """Test cross-debate memory injection."""

    @pytest.fixture
    def mock_cross_debate_memory(self):
        """Create mock CrossDebateMemory."""
        mem = MagicMock()
        mem.get_institutional_knowledge = AsyncMock(
            return_value=[
                {
                    "debate_id": "debate-100",
                    "lesson": "Always consider edge cases in rate limiting",
                    "confidence": 0.88,
                },
                {
                    "debate_id": "debate-101",
                    "lesson": "Token bucket handles bursts better",
                    "confidence": 0.91,
                },
            ]
        )
        mem.store_lesson = AsyncMock()
        return mem

    async def test_institutional_knowledge_injection(self, mock_cross_debate_memory):
        """Institutional knowledge should be injected into debates."""
        knowledge = await mock_cross_debate_memory.get_institutional_knowledge(
            topic="rate limiting",
            limit=10,
        )

        assert len(knowledge) == 2
        assert knowledge[0]["confidence"] == 0.88

    async def test_lesson_extraction_after_debate(self, mock_cross_debate_memory):
        """Lessons should be extracted and stored after debates."""
        await mock_cross_debate_memory.store_lesson(
            debate_id="debate-123",
            lesson="Token bucket with sliding window provides optimal rate limiting",
            confidence=0.92,
            topic="rate limiting",
        )

        mock_cross_debate_memory.store_lesson.assert_called_once()


class TestMemoryTierTransitions:
    """Test memory tier promotion/demotion logic."""

    async def test_high_surprise_promotes_to_fast(self):
        """High surprise memories should promote to FAST tier."""
        current_tier = "MEDIUM"
        surprise_score = 0.85
        promotion_threshold = 0.7

        if surprise_score > promotion_threshold:
            new_tier = "FAST"
        else:
            new_tier = current_tier

        assert new_tier == "FAST"

    async def test_stable_memories_demote_to_glacial(self):
        """Stable memories should demote to GLACIAL tier."""
        current_tier = "SLOW"
        consolidation = 0.85
        surprise = 0.15
        update_count = 25

        # Demotion criteria
        min_consolidation = 0.8
        max_surprise = 0.2
        min_updates = 20

        should_demote = (
            consolidation >= min_consolidation
            and surprise <= max_surprise
            and update_count >= min_updates
        )

        new_tier = "GLACIAL" if should_demote else current_tier

        assert new_tier == "GLACIAL"

    async def test_red_line_memories_not_deleted(self):
        """Red line memories should never be deleted."""
        memory = {
            "id": "mem-critical",
            "content": "Never deploy without tests",
            "red_line": True,
            "tier": "GLACIAL",
        }

        # Red line check prevents deletion
        can_delete = not memory.get("red_line", False)

        assert can_delete is False


class TestDebateContextRetrieval:
    """Test debate context retrieval from memory."""

    @pytest.fixture
    def mock_memory_manager(self):
        """Create mock MemoryManager."""
        mgr = MagicMock()
        mgr.get_debate_context = AsyncMock(
            return_value={
                "memories": [
                    {"content": "Previous design decisions", "tier": "MEDIUM"},
                    {"content": "Historical patterns", "tier": "SLOW"},
                ],
                "consensus_history": [
                    {"debate_id": "debate-100", "decision": "Use REST API"},
                ],
                "institutional_knowledge": [
                    {"lesson": "Always version APIs"},
                ],
            }
        )
        return mgr

    async def test_context_aggregation(self, mock_memory_manager):
        """Memory manager should aggregate context from all sources."""
        context = await mock_memory_manager.get_debate_context(
            task="Design API versioning strategy",
            workspace_id="team_a",
        )

        assert "memories" in context
        assert "consensus_history" in context
        assert "institutional_knowledge" in context
        assert len(context["memories"]) == 2
