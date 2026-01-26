"""
Tests for ConsensusAdapter - Bridges ConsensusMemory to Knowledge Mound.

Tests cover:
- Search by topic functionality
- Consensus to KnowledgeItem conversion
- Dissent handling
- Risk warnings and contrarian views
- Statistics
- Event emission
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, AsyncMock


class TestConsensusSearchResult:
    """Tests for ConsensusSearchResult dataclass."""

    def test_create_search_result(self):
        """Should create ConsensusSearchResult."""
        from aragora.knowledge.mound.adapters import ConsensusSearchResult

        mock_record = MagicMock()
        result = ConsensusSearchResult(
            record=mock_record,
            similarity=0.85,
            dissents=[MagicMock()],
        )

        assert result.record is mock_record
        assert result.similarity == 0.85
        assert len(result.dissents) == 1

    def test_default_dissents(self):
        """Should default dissents to empty list."""
        from aragora.knowledge.mound.adapters import ConsensusSearchResult

        mock_record = MagicMock()
        result = ConsensusSearchResult(record=mock_record)

        assert result.dissents == []


class TestConsensusAdapterInit:
    """Tests for ConsensusAdapter initialization."""

    def test_init(self):
        """Should initialize with consensus memory."""
        from aragora.knowledge.mound.adapters.consensus_adapter import ConsensusAdapter

        mock_consensus = MagicMock()
        adapter = ConsensusAdapter(mock_consensus)

        assert adapter.consensus is mock_consensus
        assert adapter._enable_dual_write is False
        assert adapter._event_callback is None

    def test_init_with_options(self):
        """Should accept optional parameters."""
        from aragora.knowledge.mound.adapters.consensus_adapter import ConsensusAdapter

        mock_consensus = MagicMock()
        callback = MagicMock()
        adapter = ConsensusAdapter(
            mock_consensus,
            enable_dual_write=True,
            event_callback=callback,
        )

        assert adapter._enable_dual_write is True
        assert adapter._event_callback is callback


class TestEventCallback:
    """Tests for event callback functionality."""

    def test_set_event_callback(self):
        """Should set event callback."""
        from aragora.knowledge.mound.adapters.consensus_adapter import ConsensusAdapter

        mock_consensus = MagicMock()
        adapter = ConsensusAdapter(mock_consensus)

        callback = MagicMock()
        adapter.set_event_callback(callback)

        assert adapter._event_callback is callback

    def test_emit_event(self):
        """Should emit event via callback."""
        from aragora.knowledge.mound.adapters.consensus_adapter import ConsensusAdapter

        mock_consensus = MagicMock()
        callback = MagicMock()
        adapter = ConsensusAdapter(mock_consensus, event_callback=callback)

        adapter._emit_event("test_event", {"key": "value"})

        callback.assert_called_once_with("test_event", {"key": "value"})

    def test_emit_event_handles_callback_error(self):
        """Should handle callback errors gracefully."""
        from aragora.knowledge.mound.adapters.consensus_adapter import ConsensusAdapter

        mock_consensus = MagicMock()
        callback = MagicMock(side_effect=Exception("Callback failed"))
        adapter = ConsensusAdapter(mock_consensus, event_callback=callback)

        # Should not raise
        adapter._emit_event("test_event", {})


class TestSearchByTopic:
    """Tests for search_by_topic method."""

    @pytest.mark.asyncio
    async def test_search_by_topic(self):
        """Should search consensus memory by topic."""
        from aragora.knowledge.mound.adapters.consensus_adapter import ConsensusAdapter

        mock_consensus = MagicMock()
        mock_debate = MagicMock()
        mock_debate.consensus = MagicMock()
        mock_debate.similarity = 0.9
        mock_debate.dissents = []
        mock_consensus.find_similar_debates = MagicMock(return_value=[mock_debate])

        adapter = ConsensusAdapter(mock_consensus)

        results = await adapter.search_by_topic("rate limiting", limit=10)

        assert len(results) == 1
        assert results[0].similarity == 0.9
        mock_consensus.find_similar_debates.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_with_domain_filter(self):
        """Should pass domain filter to consensus memory."""
        from aragora.knowledge.mound.adapters.consensus_adapter import ConsensusAdapter

        mock_consensus = MagicMock()
        mock_consensus.find_similar_debates = MagicMock(return_value=[])

        adapter = ConsensusAdapter(mock_consensus)

        await adapter.search_by_topic("test", domain="security")

        call_kwargs = mock_consensus.find_similar_debates.call_args[1]
        assert call_kwargs["domain"] == "security"

    @pytest.mark.asyncio
    async def test_search_excludes_dissents_when_requested(self):
        """Should exclude dissents when include_dissents=False."""
        from aragora.knowledge.mound.adapters.consensus_adapter import ConsensusAdapter

        mock_consensus = MagicMock()
        mock_debate = MagicMock()
        mock_debate.consensus = MagicMock()
        mock_debate.similarity = 0.9
        mock_debate.dissents = [MagicMock()]  # Has dissent
        mock_consensus.find_similar_debates = MagicMock(return_value=[mock_debate])

        adapter = ConsensusAdapter(mock_consensus)

        results = await adapter.search_by_topic("test", include_dissents=False)

        assert results[0].dissents == []


class TestGetConsensus:
    """Tests for get method."""

    def test_get_consensus_by_id(self):
        """Should get consensus record by ID."""
        from aragora.knowledge.mound.adapters.consensus_adapter import ConsensusAdapter

        mock_record = MagicMock()
        mock_consensus = MagicMock()
        mock_consensus.get_consensus = MagicMock(return_value=mock_record)

        adapter = ConsensusAdapter(mock_consensus)

        result = adapter.get("debate-123")

        assert result is mock_record
        mock_consensus.get_consensus.assert_called_once_with("debate-123")

    def test_get_strips_mound_prefix(self):
        """Should strip cs_ prefix from mound IDs."""
        from aragora.knowledge.mound.adapters.consensus_adapter import ConsensusAdapter

        mock_consensus = MagicMock()
        mock_consensus.get_consensus = MagicMock(return_value=MagicMock())

        adapter = ConsensusAdapter(mock_consensus)
        adapter.get("cs_debate-123")

        mock_consensus.get_consensus.assert_called_once_with("debate-123")

    @pytest.mark.asyncio
    async def test_get_async(self):
        """Should provide async version of get."""
        from aragora.knowledge.mound.adapters.consensus_adapter import ConsensusAdapter

        mock_record = MagicMock()
        mock_consensus = MagicMock()
        mock_consensus.get_consensus = MagicMock(return_value=mock_record)

        adapter = ConsensusAdapter(mock_consensus)

        result = await adapter.get_async("debate-123")

        assert result is mock_record


class TestToKnowledgeItem:
    """Tests for consensus to KnowledgeItem conversion."""

    def test_convert_consensus_record(self):
        """Should convert ConsensusRecord to KnowledgeItem."""
        from aragora.knowledge.mound.adapters.consensus_adapter import ConsensusAdapter
        from enum import Enum

        # Create mock consensus record with required attributes
        class MockStrength(Enum):
            STRONG = "strong"

        mock_record = MagicMock()
        mock_record.id = "debate-123"
        mock_record.topic = "Rate limiting discussion"
        mock_record.conclusion = "We should use token bucket algorithm"
        mock_record.strength = MockStrength.STRONG
        mock_record.domain = "architecture"
        mock_record.tags = ["api", "rate-limiting"]
        mock_record.confidence = 0.9
        mock_record.timestamp = datetime.now(timezone.utc)
        mock_record.participating_agents = ["claude-3", "gpt-4"]
        mock_record.agreeing_agents = ["claude-3"]
        mock_record.dissenting_agents = ["gpt-4"]
        mock_record.key_claims = []
        mock_record.supporting_evidence = []
        mock_record.dissent_ids = []
        mock_record.rounds = 3
        mock_record.debate_duration_seconds = 120
        mock_record.compute_agreement_ratio = MagicMock(return_value=0.5)
        mock_record.supersedes = None
        mock_record.superseded_by = None

        mock_consensus = MagicMock()
        adapter = ConsensusAdapter(mock_consensus)

        with patch.object(adapter, "_record_metric"):
            item = adapter.to_knowledge_item(mock_record)

        assert item.id == "cs_debate-123"
        assert "token bucket" in item.content

    def test_convert_search_result(self):
        """Should convert ConsensusSearchResult to KnowledgeItem."""
        from aragora.knowledge.mound.adapters.consensus_adapter import (
            ConsensusAdapter,
            ConsensusSearchResult,
        )
        from enum import Enum

        class MockStrength(Enum):
            UNANIMOUS = "unanimous"

        mock_record = MagicMock()
        mock_record.id = "debate-456"
        mock_record.topic = "Testing topic"
        mock_record.conclusion = "Testing conclusion"
        mock_record.strength = MockStrength.UNANIMOUS
        mock_record.domain = "testing"
        mock_record.tags = []
        mock_record.confidence = 0.95
        mock_record.timestamp = datetime.now(timezone.utc)
        mock_record.participating_agents = []
        mock_record.agreeing_agents = []
        mock_record.dissenting_agents = []
        mock_record.key_claims = []
        mock_record.supporting_evidence = []
        mock_record.dissent_ids = []
        mock_record.rounds = 2
        mock_record.debate_duration_seconds = 60
        mock_record.compute_agreement_ratio = MagicMock(return_value=1.0)
        mock_record.supersedes = None
        mock_record.superseded_by = None

        search_result = ConsensusSearchResult(
            record=mock_record,
            similarity=0.85,
        )

        mock_consensus = MagicMock()
        adapter = ConsensusAdapter(mock_consensus)

        item = adapter.to_knowledge_item(search_result)

        assert item.metadata["similarity"] == 0.85


class TestDissentConversion:
    """Tests for dissent to KnowledgeItem conversion."""

    def test_convert_dissent_record(self):
        """Should convert DissentRecord to KnowledgeItem."""
        from aragora.knowledge.mound.adapters.consensus_adapter import ConsensusAdapter
        from enum import Enum

        class MockDissentType(Enum):
            RISK_WARNING = "risk_warning"

        mock_dissent = MagicMock()
        mock_dissent.id = "dissent-789"
        mock_dissent.debate_id = "debate-123"
        mock_dissent.agent_id = "claude-3"
        mock_dissent.content = "This approach has security risks"
        mock_dissent.dissent_type = MockDissentType.RISK_WARNING
        mock_dissent.reasoning = "Buffer overflow possible"
        mock_dissent.acknowledged = False
        mock_dissent.rebuttal = None
        mock_dissent.timestamp = datetime.now(timezone.utc)
        mock_dissent.confidence = 0.7

        mock_consensus = MagicMock()
        adapter = ConsensusAdapter(mock_consensus)

        item = adapter.dissent_to_knowledge_item(mock_dissent)

        assert item.id == "ds_dissent-789"
        assert "security risks" in item.content


class TestDissentQueries:
    """Tests for dissent query methods."""

    def test_get_dissents_for_topic(self):
        """Should retrieve dissents for a topic."""
        from aragora.knowledge.mound.adapters.consensus_adapter import ConsensusAdapter

        mock_dissent = MagicMock()
        mock_consensus = MagicMock()
        mock_consensus.find_relevant_dissent = MagicMock(return_value=[mock_dissent])

        adapter = ConsensusAdapter(mock_consensus)

        dissents = adapter.get_dissents_for_topic("security", limit=10)

        assert len(dissents) == 1
        mock_consensus.find_relevant_dissent.assert_called_once_with(topic="security", limit=10)

    def test_get_risk_warnings(self):
        """Should retrieve risk warnings."""
        from aragora.knowledge.mound.adapters.consensus_adapter import ConsensusAdapter

        mock_warning = MagicMock()
        mock_consensus = MagicMock()
        mock_consensus.find_risk_warnings = MagicMock(return_value=[mock_warning])

        adapter = ConsensusAdapter(mock_consensus)

        warnings = adapter.get_risk_warnings(topic="auth", limit=5)

        assert len(warnings) == 1
        mock_consensus.find_risk_warnings.assert_called_once_with(topic="auth", limit=5)

    def test_get_contrarian_views(self):
        """Should retrieve contrarian views."""
        from aragora.knowledge.mound.adapters.consensus_adapter import ConsensusAdapter

        mock_view = MagicMock()
        mock_consensus = MagicMock()
        mock_consensus.find_contrarian_views = MagicMock(return_value=[mock_view])

        adapter = ConsensusAdapter(mock_consensus)

        views = adapter.get_contrarian_views(limit=5)

        assert len(views) == 1


class TestSearchSimilar:
    """Tests for search_similar (reverse flow) method."""

    def test_search_similar(self):
        """Should search for similar consensus records."""
        from aragora.knowledge.mound.adapters.consensus_adapter import ConsensusAdapter
        from enum import Enum

        class MockStrength(Enum):
            MODERATE = "moderate"

        mock_debate = MagicMock()
        mock_debate.consensus = MagicMock()
        mock_debate.consensus.id = "debate-123"
        mock_debate.consensus.topic = "Test topic"
        mock_debate.consensus.conclusion = "Test conclusion"
        mock_debate.consensus.strength = MockStrength.MODERATE
        mock_debate.consensus.confidence = 0.8
        mock_debate.consensus.domain = "general"
        mock_debate.consensus.timestamp = datetime.now(timezone.utc)
        mock_debate.similarity = 0.9

        mock_consensus = MagicMock()
        mock_consensus.find_similar_debates = MagicMock(return_value=[mock_debate])

        adapter = ConsensusAdapter(mock_consensus)

        results = adapter.search_similar("test query", limit=5)

        assert len(results) == 1
        assert results[0]["similarity"] == 0.9

    def test_search_similar_emits_event(self):
        """Should emit event for reverse flow query."""
        from aragora.knowledge.mound.adapters.consensus_adapter import ConsensusAdapter

        mock_consensus = MagicMock()
        mock_consensus.find_similar_debates = MagicMock(return_value=[])

        callback = MagicMock()
        adapter = ConsensusAdapter(mock_consensus, event_callback=callback)

        adapter.search_similar("test query")

        # Check that event was emitted
        assert callback.call_count >= 1


class TestSemanticSearch:
    """Tests for semantic_search method."""

    @pytest.mark.asyncio
    async def test_semantic_search_fallback(self):
        """Should fall back to keyword search when semantic not available."""
        from aragora.knowledge.mound.adapters.consensus_adapter import ConsensusAdapter
        from enum import Enum

        class MockStrength(Enum):
            STRONG = "strong"

        mock_debate = MagicMock()
        mock_debate.consensus = MagicMock()
        mock_debate.consensus.id = "debate-123"
        mock_debate.consensus.topic = "Test"
        mock_debate.consensus.conclusion = "Test conclusion"
        mock_debate.consensus.strength = MockStrength.STRONG
        mock_debate.consensus.confidence = 0.8
        mock_debate.consensus.domain = "general"
        mock_debate.consensus.timestamp = datetime.now(timezone.utc)
        mock_debate.similarity = 0.85

        mock_consensus = MagicMock()
        mock_consensus.find_similar_debates = MagicMock(return_value=[mock_debate])

        adapter = ConsensusAdapter(mock_consensus)

        results = await adapter.semantic_search("test query", limit=5)

        assert len(results) == 1


class TestStoreConsensus:
    """Tests for store_consensus (forward flow) method."""

    def test_store_consensus_marks_for_sync(self):
        """Should mark consensus for KM sync."""
        from aragora.knowledge.mound.adapters.consensus_adapter import ConsensusAdapter

        mock_record = MagicMock()
        mock_record.metadata = {}
        mock_record.topic = "Test topic"
        mock_record.id = "debate-123"
        mock_record.confidence = 0.8

        mock_consensus = MagicMock()
        adapter = ConsensusAdapter(mock_consensus)

        adapter.store_consensus(mock_record)

        assert mock_record.metadata["km_sync_pending"] is True
        assert "km_sync_requested_at" in mock_record.metadata

    def test_store_consensus_emits_event(self):
        """Should emit event for forward sync."""
        from aragora.knowledge.mound.adapters.consensus_adapter import ConsensusAdapter
        from enum import Enum

        class MockStrength(Enum):
            STRONG = "strong"

        mock_record = MagicMock()
        mock_record.metadata = {}
        mock_record.topic = "Test topic"
        mock_record.id = "debate-123"
        mock_record.confidence = 0.8
        mock_record.strength = MockStrength.STRONG

        callback = MagicMock()
        mock_consensus = MagicMock()
        adapter = ConsensusAdapter(mock_consensus, event_callback=callback)

        adapter.store_consensus(mock_record)

        callback.assert_called()
        event_type = callback.call_args[0][0]
        assert event_type == "km_adapter_forward_sync"


class TestStatistics:
    """Tests for statistics methods."""

    def test_get_stats(self):
        """Should return statistics from consensus memory."""
        from aragora.knowledge.mound.adapters.consensus_adapter import ConsensusAdapter

        mock_stats = {
            "total_records": 100,
            "total_dissents": 20,
        }
        mock_consensus = MagicMock()
        mock_consensus.get_stats = MagicMock(return_value=mock_stats)

        adapter = ConsensusAdapter(mock_consensus)

        stats = adapter.get_stats()

        assert stats["total_records"] == 100
        mock_consensus.get_stats.assert_called_once()
