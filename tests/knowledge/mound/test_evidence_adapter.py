"""Tests for the EvidenceAdapter."""

import pytest
from unittest.mock import Mock, MagicMock
from datetime import datetime

from aragora.knowledge.mound.adapters.evidence_adapter import (
    EvidenceAdapter,
    EvidenceSearchResult,
)
from aragora.knowledge.unified.types import ConfidenceLevel, KnowledgeSource


class TestEvidenceSearchResult:
    """Tests for EvidenceSearchResult dataclass."""

    def test_basic_creation(self):
        """Create a basic search result."""
        result = EvidenceSearchResult(
            evidence={"id": "ev1", "snippet": "test"},
            relevance_score=0.8,
        )
        assert result.evidence["id"] == "ev1"
        assert result.relevance_score == 0.8
        assert result.matched_topics == []

    def test_with_topics(self):
        """Create result with matched topics."""
        result = EvidenceSearchResult(
            evidence={"id": "ev1"},
            matched_topics=["legal", "contracts"],
        )
        assert result.matched_topics == ["legal", "contracts"]


class TestEvidenceAdapterInit:
    """Tests for EvidenceAdapter initialization."""

    def test_init_without_store(self):
        """Initialize without a store."""
        adapter = EvidenceAdapter()
        assert adapter.evidence_store is None

    def test_init_with_store(self):
        """Initialize with a store."""
        mock_store = Mock()
        adapter = EvidenceAdapter(store=mock_store)
        assert adapter.evidence_store is mock_store

    def test_init_with_dual_write(self):
        """Initialize with dual write enabled."""
        adapter = EvidenceAdapter(enable_dual_write=True)
        assert adapter._enable_dual_write is True

    def test_constants(self):
        """Verify adapter constants."""
        assert EvidenceAdapter.ID_PREFIX == "ev_"
        assert EvidenceAdapter.MIN_RELIABILITY == 0.6
        assert EvidenceAdapter.MIN_QUALITY == 0.7


class TestEvidenceAdapterSearchByTopic:
    """Tests for search_by_topic method."""

    def test_search_basic(self):
        """Basic topic search."""
        mock_store = Mock()
        mock_store.search_evidence.return_value = [
            {"id": "ev1", "snippet": "test", "source": "web"},
            {"id": "ev2", "snippet": "test2", "source": "github"},
        ]

        adapter = EvidenceAdapter(store=mock_store)
        results = adapter.search_by_topic("test query", limit=10)

        mock_store.search_evidence.assert_called_once_with(
            query="test query",
            limit=10,
            min_reliability=0.0,
        )
        assert len(results) == 2

    def test_search_with_source_filter(self):
        """Search with source filter."""
        mock_store = Mock()
        mock_store.search_evidence.return_value = [
            {"id": "ev1", "snippet": "test", "source": "web"},
            {"id": "ev2", "snippet": "test2", "source": "github"},
        ]

        adapter = EvidenceAdapter(store=mock_store)
        results = adapter.search_by_topic("test", source="web")

        assert len(results) == 1
        assert results[0]["source"] == "web"

    def test_search_with_min_reliability(self):
        """Search with minimum reliability."""
        mock_store = Mock()
        mock_store.search_evidence.return_value = []

        adapter = EvidenceAdapter(store=mock_store)
        adapter.search_by_topic("test", min_reliability=0.8)

        mock_store.search_evidence.assert_called_once_with(
            query="test",
            limit=10,
            min_reliability=0.8,
        )


class TestEvidenceAdapterSearchSimilar:
    """Tests for search_similar method."""

    def test_search_similar_exact_match(self):
        """Find exact match by hash."""
        mock_store = Mock()
        mock_store.get_evidence_by_hash.return_value = {
            "id": "ev1",
            "snippet": "exact content",
        }

        adapter = EvidenceAdapter(store=mock_store)
        results = adapter.search_similar("exact content")

        assert len(results) == 1
        assert results[0]["id"] == "ev1"

    def test_search_similar_no_exact_fallback_to_text(self):
        """Fall back to text search when no exact match."""
        mock_store = Mock()
        mock_store.get_evidence_by_hash.return_value = None
        mock_store.search_evidence.return_value = [
            {"id": "ev1", "snippet": "similar content"},
        ]

        adapter = EvidenceAdapter(store=mock_store)
        results = adapter.search_similar("some content here")

        mock_store.search_evidence.assert_called_once()
        assert len(results) == 1


class TestEvidenceAdapterGet:
    """Tests for get method."""

    def test_get_with_prefix(self):
        """Get evidence with ev_ prefix."""
        mock_store = Mock()
        mock_store.get_evidence.return_value = {"id": "123", "snippet": "test"}

        adapter = EvidenceAdapter(store=mock_store)
        result = adapter.get("ev_123")

        mock_store.get_evidence.assert_called_once_with("123")
        assert result["id"] == "123"

    def test_get_without_prefix(self):
        """Get evidence without prefix."""
        mock_store = Mock()
        mock_store.get_evidence.return_value = {"id": "123", "snippet": "test"}

        adapter = EvidenceAdapter(store=mock_store)
        result = adapter.get("123")

        mock_store.get_evidence.assert_called_once_with("123")

    def test_get_not_found(self):
        """Get returns None when not found."""
        mock_store = Mock()
        mock_store.get_evidence.return_value = None

        adapter = EvidenceAdapter(store=mock_store)
        result = adapter.get("nonexistent")

        assert result is None


class TestEvidenceAdapterToKnowledgeItem:
    """Tests for to_knowledge_item method."""

    def test_convert_high_reliability(self):
        """Convert evidence with high reliability."""
        adapter = EvidenceAdapter()
        evidence = {
            "id": "123",
            "snippet": "Test content",
            "source": "web",
            "title": "Test Title",
            "url": "https://example.com",
            "reliability_score": 0.95,
            "created_at": "2024-01-01T00:00:00Z",
        }

        item = adapter.to_knowledge_item(evidence)

        assert item.id == "ev_123"
        assert item.content == "Test content"
        assert item.source == KnowledgeSource.EVIDENCE
        assert item.confidence == ConfidenceLevel.VERIFIED
        assert item.importance == 0.95

    def test_convert_medium_reliability(self):
        """Convert evidence with medium reliability."""
        adapter = EvidenceAdapter()
        evidence = {
            "id": "123",
            "snippet": "Test",
            "reliability_score": 0.6,
        }

        item = adapter.to_knowledge_item(evidence)
        assert item.confidence == ConfidenceLevel.MEDIUM

    def test_convert_low_reliability(self):
        """Convert evidence with low reliability."""
        adapter = EvidenceAdapter()
        evidence = {
            "id": "123",
            "snippet": "Test",
            "reliability_score": 0.35,
        }

        item = adapter.to_knowledge_item(evidence)
        assert item.confidence == ConfidenceLevel.LOW

    def test_convert_with_quality_scores(self):
        """Convert evidence with quality scores JSON."""
        adapter = EvidenceAdapter()
        evidence = {
            "id": "123",
            "snippet": "Test",
            "reliability_score": 0.8,
            "quality_scores_json": '{"accuracy": 0.9, "relevance": 0.85}',
        }

        item = adapter.to_knowledge_item(evidence)
        assert item.metadata["quality_scores"]["accuracy"] == 0.9

    def test_convert_with_invalid_json(self):
        """Handle invalid JSON gracefully."""
        adapter = EvidenceAdapter()
        evidence = {
            "id": "123",
            "snippet": "Test",
            "reliability_score": 0.8,
            "quality_scores_json": "invalid json",
        }

        item = adapter.to_knowledge_item(evidence)
        assert item.metadata["quality_scores"] == {}


class TestEvidenceAdapterStore:
    """Tests for store method."""

    def test_store_evidence(self):
        """Store evidence via adapter."""
        mock_store = Mock()
        mock_store.save_evidence.return_value = "ev_123"

        adapter = EvidenceAdapter(store=mock_store)
        result = adapter.store(
            evidence_id="123",
            source="web",
            title="Test",
            snippet="Content",
            url="https://example.com",
            reliability_score=0.8,
        )

        mock_store.save_evidence.assert_called_once()
        assert result == "ev_123"

    def test_store_with_metadata(self):
        """Store evidence with metadata."""
        mock_store = Mock()
        mock_store.save_evidence.return_value = "ev_123"

        adapter = EvidenceAdapter(store=mock_store)
        adapter.store(
            evidence_id="123",
            source="web",
            title="Test",
            snippet="Content",
            metadata={"custom": "value"},
        )

        call_kwargs = mock_store.save_evidence.call_args[1]
        assert call_kwargs["metadata"]["custom"] == "value"


class TestEvidenceAdapterFromIngestionRequest:
    """Tests for from_ingestion_request method."""

    def test_convert_ingestion_request(self):
        """Convert ingestion request to evidence params."""
        from aragora.knowledge.mound.types import IngestionRequest, KnowledgeSource

        adapter = EvidenceAdapter()
        request = IngestionRequest(
            content="Test content",
            workspace_id="ws1",
            source_type=KnowledgeSource.EVIDENCE,
            confidence=0.85,
            debate_id="debate1",
            metadata={"source": "github", "title": "Test", "url": "https://github.com"},
        )

        params = adapter.from_ingestion_request(request)

        assert params["snippet"] == "Test content"
        assert params["reliability_score"] == 0.85
        assert params["source"] == "github"
        assert "evidence_id" in params

    def test_convert_with_explicit_id(self):
        """Convert with explicit evidence ID."""
        from aragora.knowledge.mound.types import IngestionRequest

        adapter = EvidenceAdapter()
        request = IngestionRequest(
            content="Test",
            workspace_id="ws1",
        )

        params = adapter.from_ingestion_request(request, evidence_id="custom_123")
        assert params["evidence_id"] == "custom_123"


class TestEvidenceAdapterMarkUsedInConsensus:
    """Tests for mark_used_in_consensus method."""

    def test_mark_used(self):
        """Mark evidence as used in consensus."""
        mock_store = Mock()

        adapter = EvidenceAdapter(store=mock_store)
        adapter.mark_used_in_consensus("ev_123", "debate_456")

        mock_store.mark_used_in_consensus.assert_called_once_with("debate_456", "ev_123")


class TestEvidenceAdapterGetStats:
    """Tests for get_stats method."""

    def test_get_stats(self):
        """Get statistics from store."""
        mock_store = Mock()
        mock_store.get_stats.return_value = {
            "total_evidence": 100,
            "by_source": {"web": 60, "github": 40},
        }

        adapter = EvidenceAdapter(store=mock_store)
        stats = adapter.get_stats()

        assert stats["total_evidence"] == 100


class TestEvidenceAdapterGetDebateEvidence:
    """Tests for get_debate_evidence method."""

    def test_get_debate_evidence(self):
        """Get evidence for a debate."""
        mock_store = Mock()
        mock_store.get_debate_evidence.return_value = [
            {"id": "ev1", "relevance": 0.9},
            {"id": "ev2", "relevance": 0.7},
        ]

        adapter = EvidenceAdapter(store=mock_store)
        results = adapter.get_debate_evidence("debate_123", min_relevance=0.5)

        mock_store.get_debate_evidence.assert_called_once_with("debate_123", 0.5)
        assert len(results) == 2
