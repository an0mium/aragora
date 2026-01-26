"""Tests for EvidenceAdapter error handling paths."""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime

from aragora.knowledge.mound.adapters.evidence_adapter import (
    EvidenceAdapter,
    EvidenceAdapterError,
    EvidenceStoreUnavailableError,
    EvidenceNotFoundError,
    EvidenceSearchResult,
)


@pytest.fixture
def mock_store():
    """Create a mock evidence store."""
    store = MagicMock()
    store.search_evidence = MagicMock(return_value=[])
    store.get_evidence = MagicMock(return_value=None)
    store.get_evidence_by_hash = MagicMock(return_value=None)
    store.save_evidence = MagicMock(return_value="ev_123")
    store.get_stats = MagicMock(return_value={"total": 100})
    store.get_debate_evidence = MagicMock(return_value=[])
    store.mark_used_in_consensus = MagicMock()
    store.update_evidence = MagicMock()
    return store


class TestStoreUnavailability:
    """Tests for operations when store is not configured."""

    def test_search_by_topic_without_store(self):
        """Test search_by_topic raises error when store is None."""
        adapter = EvidenceAdapter(store=None)

        with pytest.raises(EvidenceStoreUnavailableError) as exc_info:
            adapter.search_by_topic("test query")

        assert "not configured" in str(exc_info.value)

    def test_search_similar_without_store(self):
        """Test search_similar raises error when store is None."""
        adapter = EvidenceAdapter(store=None)

        with pytest.raises(EvidenceStoreUnavailableError):
            adapter.search_similar("test content")

    def test_get_without_store(self):
        """Test get raises error when store is None."""
        adapter = EvidenceAdapter(store=None)

        with pytest.raises(EvidenceStoreUnavailableError):
            adapter.get("ev_123")

    def test_store_without_store(self):
        """Test store raises error when store is None."""
        adapter = EvidenceAdapter(store=None)

        with pytest.raises(EvidenceStoreUnavailableError):
            adapter.store(
                evidence_id="ev_123",
                source="test",
                title="Test",
                snippet="Test content",
            )

    def test_get_stats_without_store(self):
        """Test get_stats raises error when store is None."""
        adapter = EvidenceAdapter(store=None)

        with pytest.raises(EvidenceStoreUnavailableError):
            adapter.get_stats()

    def test_get_debate_evidence_without_store(self):
        """Test get_debate_evidence raises error when store is None."""
        adapter = EvidenceAdapter(store=None)

        with pytest.raises(EvidenceStoreUnavailableError):
            adapter.get_debate_evidence("debate_123")

    def test_mark_used_in_consensus_without_store(self):
        """Test mark_used_in_consensus raises error when store is None."""
        adapter = EvidenceAdapter(store=None)

        with pytest.raises(EvidenceStoreUnavailableError):
            adapter.mark_used_in_consensus("ev_123", "debate_123")

    @pytest.mark.asyncio
    async def test_update_reliability_without_store(self):
        """Test update_reliability_from_km raises error when store is None."""
        adapter = EvidenceAdapter(store=None)

        with pytest.raises(EvidenceStoreUnavailableError):
            await adapter.update_reliability_from_km("ev_123", {"confidence": 0.9})


class TestSearchErrorHandling:
    """Tests for search operation error handling."""

    def test_search_by_topic_store_exception(self, mock_store):
        """Test search_by_topic wraps store exceptions."""
        mock_store.search_evidence.side_effect = RuntimeError("Database error")
        adapter = EvidenceAdapter(store=mock_store)

        with pytest.raises(EvidenceAdapterError) as exc_info:
            adapter.search_by_topic("test query")

        assert "Search failed" in str(exc_info.value)
        assert exc_info.value.__cause__ is not None

    def test_search_similar_hash_lookup_fallback(self, mock_store):
        """Test search_similar falls back to text search on hash error."""
        mock_store.get_evidence_by_hash.side_effect = RuntimeError("Hash error")
        mock_store.search_evidence.return_value = [{"id": "ev_456"}]

        adapter = EvidenceAdapter(store=mock_store)
        results = adapter.search_similar("test content for similarity")

        # Should fall back to text search
        assert len(results) == 1
        mock_store.search_evidence.assert_called_once()

    def test_search_similar_no_hash_method(self, mock_store):
        """Test search_similar handles store without hash lookup."""
        del mock_store.get_evidence_by_hash  # Remove method
        mock_store.search_evidence.return_value = [{"id": "ev_789"}]

        adapter = EvidenceAdapter(store=mock_store)
        results = adapter.search_similar("test content")

        assert len(results) == 1


class TestGetErrorHandling:
    """Tests for get operation error handling."""

    def test_get_strips_prefix(self, mock_store):
        """Test get strips ev_ prefix from ID."""
        mock_store.get_evidence.return_value = {"id": "123", "snippet": "test"}

        adapter = EvidenceAdapter(store=mock_store)
        result = adapter.get("ev_123")

        mock_store.get_evidence.assert_called_with("123")
        assert result is not None

    def test_get_store_exception(self, mock_store):
        """Test get wraps store exceptions."""
        mock_store.get_evidence.side_effect = RuntimeError("Database error")
        adapter = EvidenceAdapter(store=mock_store)

        with pytest.raises(EvidenceAdapterError) as exc_info:
            adapter.get("ev_123")

        assert "Failed to get evidence" in str(exc_info.value)


class TestStoreErrorHandling:
    """Tests for store operation error handling."""

    def test_store_success_with_event(self, mock_store):
        """Test successful store emits event."""
        mock_store.save_evidence.return_value = "ev_new"
        callback = MagicMock()

        adapter = EvidenceAdapter(store=mock_store, event_callback=callback)
        result = adapter.store(
            evidence_id="ev_new",
            source="test",
            title="Test Evidence",
            snippet="Test content",
        )

        assert result == "ev_new"
        callback.assert_called_once()
        event_type, event_data = callback.call_args[0]
        assert event_type == "knowledge_indexed"
        assert event_data["evidence_id"] == "ev_new"

    def test_store_exception(self, mock_store):
        """Test store wraps exceptions."""
        mock_store.save_evidence.side_effect = RuntimeError("Storage error")
        adapter = EvidenceAdapter(store=mock_store)

        with pytest.raises(EvidenceAdapterError) as exc_info:
            adapter.store(
                evidence_id="ev_fail",
                source="test",
                title="Test",
                snippet="Content",
            )

        assert "Storage failed" in str(exc_info.value)


class TestMarkConsensusErrorHandling:
    """Tests for mark_used_in_consensus error handling."""

    def test_mark_consensus_success(self, mock_store):
        """Test successful consensus marking."""
        adapter = EvidenceAdapter(store=mock_store)
        adapter.mark_used_in_consensus("ev_123", "debate_456")

        mock_store.mark_used_in_consensus.assert_called_with("debate_456", "ev_123")

    def test_mark_consensus_no_method(self, mock_store):
        """Test handles store without mark_used_in_consensus."""
        del mock_store.mark_used_in_consensus
        adapter = EvidenceAdapter(store=mock_store)

        # Should not raise
        adapter.mark_used_in_consensus("ev_123", "debate_456")

    def test_mark_consensus_exception(self, mock_store):
        """Test mark_used_in_consensus wraps exceptions."""
        mock_store.mark_used_in_consensus.side_effect = RuntimeError("DB error")
        adapter = EvidenceAdapter(store=mock_store)

        with pytest.raises(EvidenceAdapterError) as exc_info:
            adapter.mark_used_in_consensus("ev_123", "debate_456")

        assert "Failed to mark consensus usage" in str(exc_info.value)


class TestReliabilityUpdateErrorHandling:
    """Tests for update_reliability_from_km error handling."""

    @pytest.mark.asyncio
    async def test_update_reliability_not_found(self, mock_store):
        """Test update raises error when evidence not found."""
        mock_store.get_evidence.return_value = None
        adapter = EvidenceAdapter(store=mock_store)

        with pytest.raises(EvidenceNotFoundError):
            await adapter.update_reliability_from_km("ev_missing", {"confidence": 0.9})

    @pytest.mark.asyncio
    async def test_update_reliability_success(self, mock_store):
        """Test successful reliability update."""
        mock_store.get_evidence.return_value = {
            "id": "123",
            "reliability_score": 0.5,
        }
        adapter = EvidenceAdapter(store=mock_store)

        await adapter.update_reliability_from_km(
            "ev_123",
            {"confidence": 0.9, "validation_count": 3},
        )

        mock_store.update_evidence.assert_called_once()
        call_args = mock_store.update_evidence.call_args
        assert call_args[0][0] == "123"  # evidence_id
        assert "reliability_score" in call_args[1]

    @pytest.mark.asyncio
    async def test_update_reliability_exception(self, mock_store):
        """Test update wraps exceptions."""
        mock_store.get_evidence.return_value = {"id": "123", "reliability_score": 0.5}
        mock_store.update_evidence.side_effect = RuntimeError("Update error")
        adapter = EvidenceAdapter(store=mock_store)

        with pytest.raises(EvidenceAdapterError) as exc_info:
            await adapter.update_reliability_from_km("ev_123", {"confidence": 0.9})

        assert "Reliability update failed" in str(exc_info.value)


class TestStatsErrorHandling:
    """Tests for get_stats error handling."""

    def test_get_stats_success(self, mock_store):
        """Test successful stats retrieval."""
        mock_store.get_stats.return_value = {"total": 100, "by_source": {"web": 50}}
        adapter = EvidenceAdapter(store=mock_store)

        stats = adapter.get_stats()

        assert stats["total"] == 100

    def test_get_stats_no_method(self, mock_store):
        """Test handles store without get_stats."""
        del mock_store.get_stats
        adapter = EvidenceAdapter(store=mock_store)

        stats = adapter.get_stats()

        assert "error" in stats
        assert "not supported" in stats["error"]

    def test_get_stats_exception(self, mock_store):
        """Test get_stats wraps exceptions."""
        mock_store.get_stats.side_effect = RuntimeError("Stats error")
        adapter = EvidenceAdapter(store=mock_store)

        with pytest.raises(EvidenceAdapterError) as exc_info:
            adapter.get_stats()

        assert "Stats retrieval failed" in str(exc_info.value)


class TestDebateEvidenceErrorHandling:
    """Tests for get_debate_evidence error handling."""

    def test_get_debate_evidence_success(self, mock_store):
        """Test successful debate evidence retrieval."""
        mock_store.get_debate_evidence.return_value = [
            {"id": "ev_1", "snippet": "Evidence 1"},
            {"id": "ev_2", "snippet": "Evidence 2"},
        ]
        adapter = EvidenceAdapter(store=mock_store)

        evidence = adapter.get_debate_evidence("debate_123")

        assert len(evidence) == 2

    def test_get_debate_evidence_no_method(self, mock_store):
        """Test handles store without get_debate_evidence."""
        del mock_store.get_debate_evidence
        adapter = EvidenceAdapter(store=mock_store)

        evidence = adapter.get_debate_evidence("debate_123")

        assert evidence == []

    def test_get_debate_evidence_exception(self, mock_store):
        """Test get_debate_evidence wraps exceptions."""
        mock_store.get_debate_evidence.side_effect = RuntimeError("Query error")
        adapter = EvidenceAdapter(store=mock_store)

        with pytest.raises(EvidenceAdapterError) as exc_info:
            adapter.get_debate_evidence("debate_123")

        assert "Debate evidence retrieval failed" in str(exc_info.value)


class TestJSONParsingErrorHandling:
    """Tests for JSON parsing error handling."""

    def test_invalid_quality_scores_json(self, mock_store):
        """Test handles invalid quality_scores_json gracefully."""
        mock_store.get_evidence.return_value = {
            "id": "123",
            "snippet": "Test",
            "quality_scores_json": "not valid json{",
            "reliability_score": 0.8,
            "created_at": "2024-01-01T00:00:00Z",
        }
        adapter = EvidenceAdapter(store=mock_store)

        # Should not raise, just log warning
        item = adapter.to_knowledge_item(mock_store.get_evidence.return_value)

        assert item.metadata["quality_scores"] == {}

    def test_invalid_enriched_metadata_json(self, mock_store):
        """Test handles invalid enriched_metadata_json gracefully."""
        mock_store.get_evidence.return_value = {
            "id": "456",
            "snippet": "Test",
            "enriched_metadata_json": 12345,  # Wrong type
            "reliability_score": 0.7,
            "created_at": "2024-01-01T00:00:00Z",
        }
        adapter = EvidenceAdapter(store=mock_store)

        # Should not raise, just log warning
        item = adapter.to_knowledge_item(mock_store.get_evidence.return_value)

        assert item.metadata["enriched"] == {}

    def test_valid_json_parsing(self, mock_store):
        """Test valid JSON is parsed correctly."""
        mock_store.get_evidence.return_value = {
            "id": "789",
            "snippet": "Test content",
            "quality_scores_json": '{"accuracy": 0.9, "relevance": 0.8}',
            "enriched_metadata_json": '{"source_type": "academic"}',
            "reliability_score": 0.85,
            "created_at": "2024-01-15T10:30:00Z",
        }
        adapter = EvidenceAdapter(store=mock_store)

        item = adapter.to_knowledge_item(mock_store.get_evidence.return_value)

        assert item.metadata["quality_scores"]["accuracy"] == 0.9
        assert item.metadata["enriched"]["source_type"] == "academic"


class TestEventCallbackErrorHandling:
    """Tests for event callback error handling."""

    def test_event_callback_exception_logged(self, mock_store):
        """Test event callback exceptions are caught and logged."""

        def failing_callback(event_type, data):
            raise RuntimeError("Callback failed")

        mock_store.save_evidence.return_value = "ev_123"
        adapter = EvidenceAdapter(store=mock_store, event_callback=failing_callback)

        # Should not raise despite callback failure
        result = adapter.store(
            evidence_id="ev_123",
            source="test",
            title="Test",
            snippet="Content",
        )

        assert result == "ev_123"


class TestEvidenceSearchResult:
    """Tests for EvidenceSearchResult dataclass."""

    def test_default_matched_topics(self):
        """Test matched_topics defaults to empty list."""
        result = EvidenceSearchResult(
            evidence={"id": "123"},
            relevance_score=0.8,
        )

        assert result.matched_topics == []

    def test_with_matched_topics(self):
        """Test matched_topics can be set."""
        result = EvidenceSearchResult(
            evidence={"id": "123"},
            relevance_score=0.9,
            matched_topics=["topic1", "topic2"],
        )

        assert result.matched_topics == ["topic1", "topic2"]
