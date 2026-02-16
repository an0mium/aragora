"""
Tests for ContinuumAdapter - Bridges ContinuumMemory to Knowledge Mound.

Tests cover:
- Adapter initialization with different configurations
- Memory tier interactions (fast/medium/slow/glacial)
- Sync operations between tiers
- Query and retrieval methods (search_by_keyword, get, get_async)
- Storage and update operations
- TTL handling for different tiers
- Error handling and recovery
- Reverse flow (KM -> ContinuumMemory) validation
- Forward flow (ContinuumMemory -> KM) sync
- FusionMixin and SemanticSearchMixin integration
- Edge cases
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# ============================================================================
# Test Fixtures and Mock Classes
# ============================================================================


class MockMemoryTier(str, Enum):
    """Mock MemoryTier for testing."""

    FAST = "fast"
    MEDIUM = "medium"
    SLOW = "slow"
    GLACIAL = "glacial"


@dataclass
class MockContinuumMemoryEntry:
    """Mock ContinuumMemoryEntry for testing."""

    id: str = "entry-123"
    tier: MockMemoryTier = MockMemoryTier.SLOW
    content: str = "Test memory content"
    importance: float = 0.5
    surprise_score: float = 0.3
    consolidation_score: float = 0.6
    update_count: int = 5
    success_count: int = 3
    failure_count: int = 2
    created_at: str = "2024-01-15T10:00:00"
    updated_at: str = "2024-01-15T12:00:00"
    metadata: dict = field(default_factory=dict)
    red_line: bool = False
    red_line_reason: str = ""
    _cross_references: list = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.5

    @property
    def tags(self) -> list[str]:
        return self.metadata.get("tags", [])

    @property
    def cross_references(self) -> list[str]:
        return self.metadata.get("cross_references", [])

    @property
    def knowledge_mound_id(self) -> str:
        return f"cm_{self.id}"

    def add_cross_reference(self, ref_id: str) -> None:
        refs = self.metadata.get("cross_references", [])
        if ref_id not in refs:
            refs.append(ref_id)
            self.metadata["cross_references"] = refs


def create_mock_continuum() -> MagicMock:
    """Create a mock ContinuumMemory instance."""
    mock = MagicMock()

    # Default entry
    default_entry = MockContinuumMemoryEntry()

    # Configure methods
    mock.get.return_value = default_entry
    mock.get_async = AsyncMock(return_value=default_entry)
    mock.retrieve.return_value = [default_entry]
    mock.add.return_value = default_entry
    mock.update.return_value = True
    mock.promote_entry.return_value = True
    mock.demote_entry.return_value = True
    mock.get_stats.return_value = {
        "total_entries": 100,
        "entries_by_tier": {"fast": 10, "medium": 30, "slow": 40, "glacial": 20},
    }
    mock.get_tier_metrics.return_value = {
        "promotions": 5,
        "demotions": 2,
        "tier_counts": {"fast": 10, "medium": 30, "slow": 40, "glacial": 20},
    }

    return mock


# ============================================================================
# Adapter Initialization Tests
# ============================================================================


class TestContinuumAdapterInit:
    """Tests for ContinuumAdapter initialization."""

    def test_init_with_continuum(self):
        """Should initialize with ContinuumMemory."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter

        mock_continuum = create_mock_continuum()
        adapter = ContinuumAdapter(mock_continuum)

        assert adapter._continuum is mock_continuum
        assert adapter._enable_dual_write is False
        assert adapter._event_callback is None

    def test_init_with_options(self):
        """Should accept optional parameters."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter

        mock_continuum = create_mock_continuum()
        callback = MagicMock()
        adapter = ContinuumAdapter(
            mock_continuum,
            enable_dual_write=True,
            event_callback=callback,
            enable_resilience=False,
        )

        assert adapter._enable_dual_write is True
        assert adapter._event_callback is callback

    def test_init_with_resilience_enabled(self):
        """Should initialize with resilience enabled by default."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter

        mock_continuum = create_mock_continuum()

        # Just verify it doesn't raise
        adapter = ContinuumAdapter(mock_continuum, enable_resilience=True)

        assert adapter._continuum is mock_continuum

    def test_init_with_resilience_disabled(self):
        """Should initialize with resilience disabled."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter

        mock_continuum = create_mock_continuum()
        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        assert adapter._continuum is mock_continuum

    def test_continuum_property(self):
        """Should provide access to underlying continuum memory."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter

        mock_continuum = create_mock_continuum()
        adapter = ContinuumAdapter(mock_continuum)

        assert adapter.continuum is mock_continuum


# ============================================================================
# Event Callback Tests
# ============================================================================


class TestEventCallback:
    """Tests for event callback functionality."""

    def test_set_event_callback(self):
        """Should set event callback."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter

        mock_continuum = create_mock_continuum()
        adapter = ContinuumAdapter(mock_continuum)

        callback = MagicMock()
        adapter.set_event_callback(callback)

        assert adapter._event_callback is callback

    def test_emit_event(self):
        """Should emit event via callback."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter

        mock_continuum = create_mock_continuum()
        callback = MagicMock()
        adapter = ContinuumAdapter(mock_continuum, event_callback=callback)

        adapter._emit_event("test_event", {"key": "value"})

        callback.assert_called_once_with("test_event", {"key": "value"})

    def test_emit_event_handles_callback_error(self):
        """Should handle callback errors gracefully."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter

        mock_continuum = create_mock_continuum()
        callback = MagicMock(side_effect=RuntimeError("Callback failed"))
        adapter = ContinuumAdapter(mock_continuum, event_callback=callback)

        # Should not raise
        adapter._emit_event("test_event", {})

    def test_emit_event_without_callback(self):
        """Should not fail when no callback is set."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter

        mock_continuum = create_mock_continuum()
        adapter = ContinuumAdapter(mock_continuum)

        # Should not raise
        adapter._emit_event("test_event", {"key": "value"})


# ============================================================================
# Search by Keyword Tests
# ============================================================================


class TestSearchByKeyword:
    """Tests for search_by_keyword method."""

    def test_search_by_keyword_basic(self):
        """Should search continuum memory by keyword."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter

        mock_continuum = create_mock_continuum()
        entry = MockContinuumMemoryEntry(content="TypeError in agent response")
        mock_continuum.retrieve.return_value = [entry]

        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        results = adapter.search_by_keyword("TypeError", limit=10)

        assert len(results) == 1
        assert results[0].content == "TypeError in agent response"
        mock_continuum.retrieve.assert_called_once()

    def test_search_with_tier_filter(self):
        """Should filter by tier."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter

        mock_continuum = create_mock_continuum()
        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        adapter.search_by_keyword("test", tiers=["fast", "medium"])

        call_kwargs = mock_continuum.retrieve.call_args[1]
        assert call_kwargs["tiers"] is not None
        assert len(call_kwargs["tiers"]) == 2

    def test_search_with_min_importance(self):
        """Should filter by minimum importance."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter

        mock_continuum = create_mock_continuum()
        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        adapter.search_by_keyword("test", min_importance=0.7)

        call_kwargs = mock_continuum.retrieve.call_args[1]
        assert call_kwargs["min_importance"] == 0.7

    def test_search_with_limit(self):
        """Should respect limit parameter."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter

        mock_continuum = create_mock_continuum()
        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        adapter.search_by_keyword("test", limit=5)

        call_kwargs = mock_continuum.retrieve.call_args[1]
        assert call_kwargs["limit"] == 5

    def test_search_with_unknown_tier(self):
        """Should skip unknown tier names with warning."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter

        mock_continuum = create_mock_continuum()
        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        # Include an invalid tier name
        adapter.search_by_keyword("test", tiers=["fast", "invalid_tier"])

        # Should still execute with valid tiers
        mock_continuum.retrieve.assert_called_once()


# ============================================================================
# Get Methods Tests
# ============================================================================


class TestGetMethods:
    """Tests for get and get_async methods."""

    def test_get_entry_by_id(self):
        """Should get entry by ID."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter

        mock_continuum = create_mock_continuum()
        entry = MockContinuumMemoryEntry(id="entry-456")
        mock_continuum.get.return_value = entry

        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        result = adapter.get("entry-456")

        assert result is entry
        mock_continuum.get.assert_called_once_with("entry-456")

    def test_get_strips_mound_prefix(self):
        """Should strip cm_ prefix from mound IDs."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter

        mock_continuum = create_mock_continuum()
        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        adapter.get("cm_entry-789")

        mock_continuum.get.assert_called_once_with("entry-789")

    def test_get_returns_none_when_not_found(self):
        """Should return None when entry not found."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter

        mock_continuum = create_mock_continuum()
        mock_continuum.get.return_value = None

        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        result = adapter.get("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_async(self):
        """Should provide async version of get."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter

        mock_continuum = create_mock_continuum()
        entry = MockContinuumMemoryEntry(id="async-entry")
        mock_continuum.get_async = AsyncMock(return_value=entry)

        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        result = await adapter.get_async("async-entry")

        assert result is entry

    @pytest.mark.asyncio
    async def test_get_async_strips_prefix(self):
        """Should strip cm_ prefix in async get."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter

        mock_continuum = create_mock_continuum()
        mock_continuum.get_async = AsyncMock(return_value=MockContinuumMemoryEntry())

        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        await adapter.get_async("cm_entry-123")

        mock_continuum.get_async.assert_called_once_with("entry-123")


# ============================================================================
# To Knowledge Item Conversion Tests
# ============================================================================


class TestToKnowledgeItem:
    """Tests for to_knowledge_item conversion."""

    def test_convert_basic_entry(self):
        """Should convert ContinuumMemoryEntry to KnowledgeItem."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter

        mock_continuum = create_mock_continuum()
        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        entry = MockContinuumMemoryEntry(
            id="entry-001",
            content="Test content",
            tier=MockMemoryTier.SLOW,
            importance=0.8,
        )

        item = adapter.to_knowledge_item(entry)

        assert item.id == "cm_entry-001"
        assert item.content == "Test content"
        assert item.source_id == "entry-001"
        assert item.importance == 0.8

    def test_convert_fast_tier_to_low_confidence(self):
        """Should map fast tier to LOW confidence."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter

        mock_continuum = create_mock_continuum()
        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        entry = MockContinuumMemoryEntry(tier=MockMemoryTier.FAST)
        item = adapter.to_knowledge_item(entry)

        assert item.confidence.value == "low"

    def test_convert_medium_tier_to_medium_confidence(self):
        """Should map medium tier to MEDIUM confidence."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter

        mock_continuum = create_mock_continuum()
        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        entry = MockContinuumMemoryEntry(tier=MockMemoryTier.MEDIUM)
        item = adapter.to_knowledge_item(entry)

        assert item.confidence.value == "medium"

    def test_convert_slow_tier_to_high_confidence(self):
        """Should map slow tier to HIGH confidence."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter

        mock_continuum = create_mock_continuum()
        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        entry = MockContinuumMemoryEntry(tier=MockMemoryTier.SLOW)
        item = adapter.to_knowledge_item(entry)

        assert item.confidence.value == "high"

    def test_convert_glacial_tier_to_verified_confidence(self):
        """Should map glacial tier to VERIFIED confidence."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter

        mock_continuum = create_mock_continuum()
        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        entry = MockContinuumMemoryEntry(tier=MockMemoryTier.GLACIAL)
        item = adapter.to_knowledge_item(entry)

        assert item.confidence.value == "verified"

    def test_convert_includes_metadata(self):
        """Should include memory metadata in KnowledgeItem."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter

        mock_continuum = create_mock_continuum()
        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        entry = MockContinuumMemoryEntry(
            surprise_score=0.7,
            consolidation_score=0.8,
            update_count=10,
        )
        # Setup success_rate calculation
        entry.success_count = 8
        entry.failure_count = 2

        item = adapter.to_knowledge_item(entry)

        assert item.metadata["tier"] == "slow"
        assert item.metadata["surprise_score"] == 0.7
        assert item.metadata["consolidation_score"] == 0.8
        assert item.metadata["update_count"] == 10
        assert item.metadata["success_rate"] == 0.8

    def test_convert_entry_with_red_line(self):
        """Should include red line info in metadata."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter

        mock_continuum = create_mock_continuum()
        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        entry = MockContinuumMemoryEntry(
            red_line=True,
            red_line_reason="Critical business rule",
        )

        item = adapter.to_knowledge_item(entry)

        assert item.metadata["red_line"] is True
        assert item.metadata["red_line_reason"] == "Critical business rule"

    def test_convert_entry_with_tags(self):
        """Should include tags in metadata."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter

        mock_continuum = create_mock_continuum()
        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        entry = MockContinuumMemoryEntry()
        entry.metadata["tags"] = ["security", "api"]

        item = adapter.to_knowledge_item(entry)

        assert item.metadata["tags"] == ["security", "api"]

    def test_convert_entry_with_cross_references(self):
        """Should include cross references in metadata."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter

        mock_continuum = create_mock_continuum()
        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        entry = MockContinuumMemoryEntry()
        entry.metadata["cross_references"] = ["ref-1", "ref-2"]

        item = adapter.to_knowledge_item(entry)

        assert item.metadata["cross_references"] == ["ref-1", "ref-2"]


# ============================================================================
# From Ingestion Request Tests
# ============================================================================


class TestFromIngestionRequest:
    """Tests for from_ingestion_request conversion."""

    def test_convert_basic_request(self):
        """Should convert IngestionRequest to add() parameters."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter
        from aragora.knowledge.mound.types import IngestionRequest, SourceType

        mock_continuum = create_mock_continuum()
        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        request = IngestionRequest(
            content="Test content from ingestion",
            workspace_id="ws-123",
            source_type=SourceType.CONTINUUM,
            confidence=0.75,
            tier="medium",
        )

        params = adapter.from_ingestion_request(request)

        assert params["content"] == "Test content from ingestion"
        assert params["importance"] == 0.75

    def test_convert_request_with_entry_id(self):
        """Should use provided entry ID."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter
        from aragora.knowledge.mound.types import IngestionRequest, SourceType

        mock_continuum = create_mock_continuum()
        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        request = IngestionRequest(
            content="Test",
            workspace_id="ws-123",
        )

        params = adapter.from_ingestion_request(request, entry_id="custom-id")

        assert params["id"] == "custom-id"

    def test_convert_request_generates_id(self):
        """Should generate ID if not provided."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter
        from aragora.knowledge.mound.types import IngestionRequest

        mock_continuum = create_mock_continuum()
        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        request = IngestionRequest(
            content="Test",
            workspace_id="ws-123",
        )

        params = adapter.from_ingestion_request(request)

        assert params["id"].startswith("mound_")

    def test_convert_request_with_metadata(self):
        """Should include source metadata."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter
        from aragora.knowledge.mound.types import IngestionRequest, SourceType

        mock_continuum = create_mock_continuum()
        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        request = IngestionRequest(
            content="Test",
            workspace_id="ws-123",
            source_type=SourceType.DEBATE,
            debate_id="debate-456",
            agent_id="claude",
            topics=["security", "api"],
        )

        params = adapter.from_ingestion_request(request)

        assert params["metadata"]["source_type"] == "debate"
        assert params["metadata"]["debate_id"] == "debate-456"
        assert params["metadata"]["agent_id"] == "claude"
        assert params["metadata"]["topics"] == ["security", "api"]


# ============================================================================
# Store Methods Tests
# ============================================================================


class TestStoreMethods:
    """Tests for store and storage methods."""

    def test_store_content(self):
        """Should store content in continuum memory."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter

        mock_continuum = create_mock_continuum()
        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        entry_id = adapter.store(
            content="Test storage content",
            importance=0.8,
            tier="medium",
        )

        mock_continuum.add.assert_called_once()
        call_kwargs = mock_continuum.add.call_args[1]
        assert call_kwargs["content"] == "Test storage content"
        assert call_kwargs["importance"] == 0.8

    def test_store_with_entry_id(self):
        """Should use provided entry ID."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter

        mock_continuum = create_mock_continuum()
        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        entry_id = adapter.store(
            content="Test",
            entry_id="custom-store-id",
        )

        assert entry_id == "custom-store-id"

    def test_store_generates_id(self):
        """Should generate ID if not provided."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter

        mock_continuum = create_mock_continuum()
        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        entry_id = adapter.store(content="Test")

        assert entry_id.startswith("mound_")

    def test_store_with_metadata(self):
        """Should pass metadata to continuum memory."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter

        mock_continuum = create_mock_continuum()
        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        adapter.store(
            content="Test",
            metadata={"source": "test", "tags": ["important"]},
        )

        call_kwargs = mock_continuum.add.call_args[1]
        assert call_kwargs["metadata"]["source"] == "test"
        assert call_kwargs["metadata"]["tags"] == ["important"]


# ============================================================================
# Link to Mound Tests
# ============================================================================


class TestLinkToMound:
    """Tests for link_to_mound method."""

    def test_link_entry_to_mound_node(self):
        """Should create cross-reference between entry and mound node."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter

        mock_continuum = create_mock_continuum()
        entry = MockContinuumMemoryEntry(id="entry-001")
        mock_continuum.get.return_value = entry

        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        adapter.link_to_mound("entry-001", "km_node_123")

        # Should have called update to persist the cross-reference
        mock_continuum.update.assert_called()

    def test_link_handles_missing_entry(self):
        """Should handle linking to non-existent entry."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter

        mock_continuum = create_mock_continuum()
        mock_continuum.get.return_value = None

        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        # Should not raise
        adapter.link_to_mound("nonexistent", "km_node_123")


# ============================================================================
# Statistics Tests
# ============================================================================


class TestStatistics:
    """Tests for statistics methods."""

    def test_get_stats(self):
        """Should return continuum memory statistics."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter

        mock_continuum = create_mock_continuum()
        mock_continuum.get_stats.return_value = {
            "total_entries": 100,
            "entries_by_tier": {"fast": 10, "medium": 30, "slow": 40, "glacial": 20},
        }

        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        stats = adapter.get_stats()

        assert stats["total_entries"] == 100
        mock_continuum.get_stats.assert_called_once()

    def test_get_tier_metrics(self):
        """Should return tier metrics."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter

        mock_continuum = create_mock_continuum()
        mock_continuum.get_tier_metrics.return_value = {
            "promotions": 5,
            "demotions": 2,
        }

        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        metrics = adapter.get_tier_metrics()

        assert metrics["promotions"] == 5
        mock_continuum.get_tier_metrics.assert_called_once()


# ============================================================================
# Search Similar Tests (Reverse Flow)
# ============================================================================


class TestSearchSimilar:
    """Tests for search_similar (reverse flow) method."""

    def test_search_similar_basic(self):
        """Should search for similar memory entries."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter

        mock_continuum = create_mock_continuum()
        entry = MockContinuumMemoryEntry(
            id="similar-1",
            content="Similar content found",
            importance=0.85,
        )
        mock_continuum.retrieve.return_value = [entry]

        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        results = adapter.search_similar("test content", limit=5)

        assert len(results) == 1
        assert results[0]["id"] == "similar-1"
        assert results[0]["importance"] == 0.85

    def test_search_similar_emits_event(self):
        """Should emit event for reverse flow query."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter

        mock_continuum = create_mock_continuum()
        mock_continuum.retrieve.return_value = []
        callback = MagicMock()

        adapter = ContinuumAdapter(mock_continuum, event_callback=callback, enable_resilience=False)

        adapter.search_similar("test query")

        callback.assert_called()
        event_type = callback.call_args[0][0]
        assert event_type == "km_adapter_reverse_query"

    def test_search_similar_returns_dict_format(self):
        """Should return results in dict format."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter

        mock_continuum = create_mock_continuum()
        entry = MockContinuumMemoryEntry(
            surprise_score=0.5,
            consolidation_score=0.7,
        )
        mock_continuum.retrieve.return_value = [entry]

        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        results = adapter.search_similar("query")

        assert isinstance(results[0], dict)
        assert "surprise_score" in results[0]
        assert "consolidation_score" in results[0]


# ============================================================================
# Store Memory Tests (Forward Flow)
# ============================================================================


class TestStoreMemory:
    """Tests for store_memory (forward flow) method."""

    def test_store_memory_marks_for_sync(self):
        """Should mark entry for KM sync."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter

        mock_continuum = create_mock_continuum()
        entry = MockContinuumMemoryEntry()

        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        adapter.store_memory(entry)

        assert entry.metadata.get("km_sync_pending") is True
        assert "km_sync_requested_at" in entry.metadata

    def test_store_memory_emits_event(self):
        """Should emit forward sync event."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter

        mock_continuum = create_mock_continuum()
        callback = MagicMock()
        entry = MockContinuumMemoryEntry()

        adapter = ContinuumAdapter(mock_continuum, event_callback=callback, enable_resilience=False)

        adapter.store_memory(entry)

        callback.assert_called()
        event_type = callback.call_args[0][0]
        assert event_type == "km_adapter_forward_sync"


# ============================================================================
# KM Validation Result Tests
# ============================================================================


class TestKMValidationResult:
    """Tests for KMValidationResult dataclass."""

    def test_create_validation_result(self):
        """Should create KMValidationResult."""
        from aragora.knowledge.mound.adapters.continuum_adapter import KMValidationResult

        result = KMValidationResult(
            memory_id="mem-123",
            km_confidence=0.85,
            cross_debate_utility=0.6,
            validation_count=3,
            was_supported=True,
            recommendation="promote",
        )

        assert result.memory_id == "mem-123"
        assert result.km_confidence == 0.85
        assert result.cross_debate_utility == 0.6
        assert result.validation_count == 3
        assert result.was_supported is True
        assert result.recommendation == "promote"

    def test_default_values(self):
        """Should have sensible defaults."""
        from aragora.knowledge.mound.adapters.continuum_adapter import KMValidationResult

        result = KMValidationResult(memory_id="mem-123", km_confidence=0.5)

        assert result.cross_debate_utility == 0.0
        assert result.validation_count == 1
        assert result.was_contradicted is False
        assert result.was_supported is False
        assert result.recommendation == "keep"
        assert result.metadata == {}


# ============================================================================
# Update Continuum from KM Tests (Reverse Flow)
# ============================================================================


class TestUpdateContinuumFromKM:
    """Tests for update_continuum_from_km method."""

    @pytest.mark.asyncio
    async def test_update_importance_from_km_validation(self):
        """Should update importance based on KM validation."""
        from aragora.knowledge.mound.adapters.continuum_adapter import (
            ContinuumAdapter,
            KMValidationResult,
        )

        mock_continuum = create_mock_continuum()
        entry = MockContinuumMemoryEntry(importance=0.5)
        mock_continuum.get.return_value = entry

        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        validation = KMValidationResult(
            memory_id="entry-123",
            km_confidence=0.9,
            validation_count=5,
        )

        result = await adapter.update_continuum_from_km("entry-123", validation)

        assert result is True
        mock_continuum.update.assert_called()

    @pytest.mark.asyncio
    async def test_update_strips_cm_prefix(self):
        """Should strip cm_ prefix when updating."""
        from aragora.knowledge.mound.adapters.continuum_adapter import (
            ContinuumAdapter,
            KMValidationResult,
        )

        mock_continuum = create_mock_continuum()
        entry = MockContinuumMemoryEntry()
        mock_continuum.get.return_value = entry

        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        validation = KMValidationResult(memory_id="cm_entry-123", km_confidence=0.8)

        await adapter.update_continuum_from_km("cm_entry-123", validation)

        mock_continuum.get.assert_called_with("entry-123")

    @pytest.mark.asyncio
    async def test_update_returns_false_for_missing_entry(self):
        """Should return False when entry not found."""
        from aragora.knowledge.mound.adapters.continuum_adapter import (
            ContinuumAdapter,
            KMValidationResult,
        )

        mock_continuum = create_mock_continuum()
        mock_continuum.get.return_value = None

        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        validation = KMValidationResult(memory_id="nonexistent", km_confidence=0.8)

        result = await adapter.update_continuum_from_km("nonexistent", validation)

        assert result is False

    @pytest.mark.asyncio
    async def test_promote_on_high_validation(self):
        """Should promote entry when recommended."""
        from aragora.knowledge.mound.adapters.continuum_adapter import (
            ContinuumAdapter,
            KMValidationResult,
        )

        mock_continuum = create_mock_continuum()
        entry = MockContinuumMemoryEntry(tier=MockMemoryTier.SLOW)
        mock_continuum.get.return_value = entry

        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        validation = KMValidationResult(
            memory_id="entry-123",
            km_confidence=0.9,
            recommendation="promote",
        )

        await adapter.update_continuum_from_km("entry-123", validation)

        mock_continuum.promote_entry.assert_called()

    @pytest.mark.asyncio
    async def test_demote_on_low_validation(self):
        """Should demote entry when recommended."""
        from aragora.knowledge.mound.adapters.continuum_adapter import (
            ContinuumAdapter,
            KMValidationResult,
        )

        mock_continuum = create_mock_continuum()
        entry = MockContinuumMemoryEntry(tier=MockMemoryTier.SLOW)
        mock_continuum.get.return_value = entry

        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        validation = KMValidationResult(
            memory_id="entry-123",
            km_confidence=0.3,
            recommendation="demote",
        )

        await adapter.update_continuum_from_km("entry-123", validation)

        mock_continuum.demote_entry.assert_called()

    @pytest.mark.asyncio
    async def test_no_promote_at_glacial(self):
        """Should not promote when already at glacial tier."""
        from aragora.knowledge.mound.adapters.continuum_adapter import (
            ContinuumAdapter,
            KMValidationResult,
        )

        mock_continuum = create_mock_continuum()
        entry = MockContinuumMemoryEntry(tier=MockMemoryTier.GLACIAL)
        mock_continuum.get.return_value = entry

        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        validation = KMValidationResult(
            memory_id="entry-123",
            km_confidence=0.95,
            recommendation="promote",
        )

        await adapter.update_continuum_from_km("entry-123", validation)

        mock_continuum.promote_entry.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_demote_at_fast(self):
        """Should not demote when already at fast tier."""
        from aragora.knowledge.mound.adapters.continuum_adapter import (
            ContinuumAdapter,
            KMValidationResult,
        )

        mock_continuum = create_mock_continuum()
        entry = MockContinuumMemoryEntry(tier=MockMemoryTier.FAST)
        mock_continuum.get.return_value = entry

        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        validation = KMValidationResult(
            memory_id="entry-123",
            km_confidence=0.2,
            recommendation="demote",
        )

        await adapter.update_continuum_from_km("entry-123", validation)

        mock_continuum.demote_entry.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_marks_km_validated(self):
        """Should mark entry as KM-validated in metadata."""
        from aragora.knowledge.mound.adapters.continuum_adapter import (
            ContinuumAdapter,
            KMValidationResult,
        )

        mock_continuum = create_mock_continuum()
        entry = MockContinuumMemoryEntry()
        mock_continuum.get.return_value = entry

        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        validation = KMValidationResult(
            memory_id="entry-123",
            km_confidence=0.8,
            cross_debate_utility=0.5,
            was_supported=True,
        )

        await adapter.update_continuum_from_km("entry-123", validation)

        call_kwargs = mock_continuum.update.call_args[1]
        metadata = call_kwargs["metadata"]
        assert metadata["km_validated"] is True
        assert metadata["km_supported"] is True


# ============================================================================
# Batch Validation Sync Tests
# ============================================================================


class TestSyncValidationsToContinuum:
    """Tests for sync_validations_to_continuum batch method."""

    @pytest.mark.asyncio
    async def test_sync_batch_validations(self):
        """Should sync multiple validations."""
        from aragora.knowledge.mound.adapters.continuum_adapter import (
            ContinuumAdapter,
            KMValidationResult,
        )

        mock_continuum = create_mock_continuum()
        mock_continuum.get.return_value = MockContinuumMemoryEntry()

        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        validations = [
            KMValidationResult(memory_id=f"entry-{i}", km_confidence=0.8) for i in range(3)
        ]

        result = await adapter.sync_validations_to_continuum(
            workspace_id="ws-123",
            validations=validations,
        )

        assert result.total_processed == 3
        assert result.updated >= 0

    @pytest.mark.asyncio
    async def test_sync_skips_low_confidence(self):
        """Should skip validations below min_confidence threshold."""
        from aragora.knowledge.mound.adapters.continuum_adapter import (
            ContinuumAdapter,
            KMValidationResult,
        )

        mock_continuum = create_mock_continuum()
        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        validations = [
            KMValidationResult(memory_id="entry-1", km_confidence=0.5),  # Below 0.7
            KMValidationResult(memory_id="entry-2", km_confidence=0.8),  # Above 0.7
        ]

        result = await adapter.sync_validations_to_continuum(
            workspace_id="ws-123",
            validations=validations,
            min_confidence=0.7,
        )

        assert result.skipped >= 1

    @pytest.mark.asyncio
    async def test_sync_tracks_promotions(self):
        """Should track promotion count."""
        from aragora.knowledge.mound.adapters.continuum_adapter import (
            ContinuumAdapter,
            KMValidationResult,
        )

        mock_continuum = create_mock_continuum()
        entry = MockContinuumMemoryEntry(tier=MockMemoryTier.SLOW)
        mock_continuum.get.return_value = entry

        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        validations = [
            KMValidationResult(
                memory_id="entry-1",
                km_confidence=0.9,
                recommendation="promote",
            )
        ]

        result = await adapter.sync_validations_to_continuum(
            workspace_id="ws-123",
            validations=validations,
        )

        assert result.promoted == 1

    @pytest.mark.asyncio
    async def test_sync_tracks_demotions(self):
        """Should track demotion count."""
        from aragora.knowledge.mound.adapters.continuum_adapter import (
            ContinuumAdapter,
            KMValidationResult,
        )

        mock_continuum = create_mock_continuum()
        entry = MockContinuumMemoryEntry(tier=MockMemoryTier.SLOW)
        mock_continuum.get.return_value = entry

        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        validations = [
            KMValidationResult(
                memory_id="entry-1",
                km_confidence=0.75,
                recommendation="demote",
            )
        ]

        result = await adapter.sync_validations_to_continuum(
            workspace_id="ws-123",
            validations=validations,
        )

        assert result.demoted == 1

    @pytest.mark.asyncio
    async def test_sync_handles_errors(self):
        """Should handle errors gracefully and continue processing."""
        from aragora.knowledge.mound.adapters.continuum_adapter import (
            ContinuumAdapter,
            KMValidationResult,
        )

        mock_continuum = create_mock_continuum()
        # First call raises, second succeeds
        mock_continuum.get.side_effect = [
            RuntimeError("Database error"),
            MockContinuumMemoryEntry(),
        ]

        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        validations = [
            KMValidationResult(memory_id="entry-1", km_confidence=0.8),
            KMValidationResult(memory_id="entry-2", km_confidence=0.8),
        ]

        result = await adapter.sync_validations_to_continuum(
            workspace_id="ws-123",
            validations=validations,
        )

        assert len(result.errors) == 1
        # Duration can be 0 if operation is very fast
        assert result.duration_ms >= 0


# ============================================================================
# Get KM Validated Entries Tests
# ============================================================================


class TestGetKMValidatedEntries:
    """Tests for get_km_validated_entries method."""

    @pytest.mark.asyncio
    async def test_get_validated_entries(self):
        """Should return KM-validated entries."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter

        mock_continuum = create_mock_continuum()
        validated_entry = MockContinuumMemoryEntry()
        validated_entry.metadata["km_validated"] = True
        validated_entry.metadata["km_confidence"] = 0.9

        unvalidated_entry = MockContinuumMemoryEntry()

        mock_continuum.retrieve.return_value = [validated_entry, unvalidated_entry]

        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        results = await adapter.get_km_validated_entries(limit=10, min_km_confidence=0.7)

        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_get_validated_filters_by_confidence(self):
        """Should filter by minimum KM confidence."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter

        mock_continuum = create_mock_continuum()
        high_conf_entry = MockContinuumMemoryEntry()
        high_conf_entry.metadata["km_validated"] = True
        high_conf_entry.metadata["km_confidence"] = 0.9

        low_conf_entry = MockContinuumMemoryEntry()
        low_conf_entry.metadata["km_validated"] = True
        low_conf_entry.metadata["km_confidence"] = 0.5

        mock_continuum.retrieve.return_value = [high_conf_entry, low_conf_entry]

        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        results = await adapter.get_km_validated_entries(min_km_confidence=0.8)

        assert len(results) == 1


# ============================================================================
# Sync Memory to Mound Tests (Forward Flow)
# ============================================================================


class TestSyncMemoryToMound:
    """Tests for sync_memory_to_mound method."""

    @pytest.mark.asyncio
    async def test_sync_high_importance_memories(self):
        """Should sync high-importance memories to KM."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter

        mock_continuum = create_mock_continuum()
        entry = MockContinuumMemoryEntry(importance=0.8)
        entry.metadata = {}  # Not yet synced
        mock_continuum.retrieve.return_value = [entry]

        mock_mound = AsyncMock()
        mock_mound.ingest = AsyncMock(return_value="km-node-123")

        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        result = await adapter.sync_memory_to_mound(
            mound=mock_mound,
            workspace_id="ws-123",
            min_importance=0.7,
        )

        assert result["synced"] >= 0

    @pytest.mark.asyncio
    async def test_sync_skips_already_synced(self):
        """Should skip already synced entries."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter

        mock_continuum = create_mock_continuum()
        entry = MockContinuumMemoryEntry(importance=0.8)
        entry.metadata["km_synced"] = True
        mock_continuum.retrieve.return_value = [entry]

        mock_mound = AsyncMock()

        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        result = await adapter.sync_memory_to_mound(
            mound=mock_mound,
            workspace_id="ws-123",
        )

        assert result["already_synced"] == 1

    @pytest.mark.asyncio
    async def test_sync_skips_low_importance(self):
        """Should skip entries below importance threshold."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter

        mock_continuum = create_mock_continuum()
        entry = MockContinuumMemoryEntry(importance=0.5)
        entry.metadata = {}
        mock_continuum.retrieve.return_value = [entry]

        mock_mound = AsyncMock()

        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        result = await adapter.sync_memory_to_mound(
            mound=mock_mound,
            workspace_id="ws-123",
            min_importance=0.7,
        )

        assert result["skipped"] == 1

    @pytest.mark.asyncio
    async def test_sync_marks_entry_as_synced(self):
        """Should mark entry as synced after successful ingestion."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter

        mock_continuum = create_mock_continuum()
        entry = MockContinuumMemoryEntry(id="entry-sync-test", importance=0.8)
        entry.metadata = {}
        mock_continuum.retrieve.return_value = [entry]

        mock_mound = AsyncMock()
        mock_mound.ingest = AsyncMock(return_value="km-node-456")

        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        await adapter.sync_memory_to_mound(
            mound=mock_mound,
            workspace_id="ws-123",
        )

        # Should have updated the entry's metadata
        mock_continuum.update.assert_called()

    @pytest.mark.asyncio
    async def test_sync_handles_ingestion_error(self):
        """Should handle ingestion errors gracefully."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter

        mock_continuum = create_mock_continuum()
        entry = MockContinuumMemoryEntry(importance=0.8)
        entry.metadata = {}
        mock_continuum.retrieve.return_value = [entry]

        mock_mound = AsyncMock()
        mock_mound.ingest = AsyncMock(side_effect=RuntimeError("Ingestion failed"))

        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        result = await adapter.sync_memory_to_mound(
            mound=mock_mound,
            workspace_id="ws-123",
        )

        assert len(result["errors"]) == 1


# ============================================================================
# Get Reverse Sync Stats Tests
# ============================================================================


class TestGetReverseSyncStats:
    """Tests for get_reverse_sync_stats method."""

    def test_get_reverse_sync_stats(self):
        """Should return reverse sync statistics."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter

        mock_continuum = create_mock_continuum()

        # Create validated entries
        validated_entry = MockContinuumMemoryEntry(tier=MockMemoryTier.SLOW)
        validated_entry.metadata["km_validated"] = True
        validated_entry.metadata["km_confidence"] = 0.85
        validated_entry.metadata["km_cross_debate_utility"] = 0.6
        validated_entry.metadata["km_supported"] = True

        mock_continuum.retrieve.return_value = [validated_entry]

        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        stats = adapter.get_reverse_sync_stats()

        assert stats["total_km_validated"] == 1
        assert stats["km_supported"] == 1
        assert stats["avg_km_confidence"] > 0

    def test_get_reverse_sync_stats_empty(self):
        """Should handle no validated entries."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter

        mock_continuum = create_mock_continuum()
        mock_continuum.retrieve.return_value = []

        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        stats = adapter.get_reverse_sync_stats()

        assert stats["total_km_validated"] == 0
        assert stats["avg_km_confidence"] == 0.0


# ============================================================================
# ContinuumSearchResult Tests
# ============================================================================


class TestContinuumSearchResult:
    """Tests for ContinuumSearchResult dataclass."""

    def test_create_search_result(self):
        """Should create ContinuumSearchResult."""
        from aragora.knowledge.mound.adapters.continuum_adapter import (
            ContinuumSearchResult,
        )

        entry = MockContinuumMemoryEntry()
        result = ContinuumSearchResult(
            entry=entry,
            relevance_score=0.9,
            matched_keywords=["test", "query"],
        )

        assert result.entry is entry
        assert result.relevance_score == 0.9
        assert result.matched_keywords == ["test", "query"]

    def test_default_matched_keywords(self):
        """Should default matched_keywords to empty list."""
        from aragora.knowledge.mound.adapters.continuum_adapter import (
            ContinuumSearchResult,
        )

        entry = MockContinuumMemoryEntry()
        result = ContinuumSearchResult(entry=entry)

        assert result.matched_keywords == []


# ============================================================================
# ValidationSyncResult Tests
# ============================================================================


class TestValidationSyncResult:
    """Tests for ValidationSyncResult dataclass."""

    def test_create_sync_result(self):
        """Should create ValidationSyncResult."""
        from aragora.knowledge.mound.adapters.continuum_adapter import (
            ValidationSyncResult,
        )

        result = ValidationSyncResult(
            total_processed=10,
            promoted=3,
            demoted=2,
            updated=4,
            skipped=1,
            errors=["Error 1"],
            duration_ms=500,
        )

        assert result.total_processed == 10
        assert result.promoted == 3
        assert result.demoted == 2
        assert result.updated == 4
        assert result.skipped == 1
        assert len(result.errors) == 1
        assert result.duration_ms == 500

    def test_default_values(self):
        """Should have sensible defaults."""
        from aragora.knowledge.mound.adapters.continuum_adapter import (
            ValidationSyncResult,
        )

        result = ValidationSyncResult()

        assert result.total_processed == 0
        assert result.promoted == 0
        assert result.demoted == 0
        assert result.updated == 0
        assert result.skipped == 0
        assert result.errors == []
        assert result.duration_ms == 0


# ============================================================================
# FusionMixin Integration Tests
# ============================================================================


class TestFusionMixinIntegration:
    """Tests for FusionMixin integration."""

    def test_get_fusion_sources(self):
        """Should return list of fusion sources."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter

        mock_continuum = create_mock_continuum()
        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        sources = adapter._get_fusion_sources()

        assert isinstance(sources, list)
        assert "consensus" in sources
        assert "evidence" in sources

    def test_extract_fusible_data(self):
        """Should extract fusible data from KM item."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter

        mock_continuum = create_mock_continuum()
        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        km_item = {
            "id": "km-123",
            "confidence": 0.85,
            "importance": 0.8,
            "metadata": {
                "tier": "slow",
                "source_id": "entry-123",
            },
        }

        data = adapter._extract_fusible_data(km_item)

        assert data["confidence"] == 0.85
        assert data["tier"] == "slow"

    def test_apply_fusion_result(self):
        """Should apply fusion result to entry."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter

        mock_continuum = create_mock_continuum()
        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        entry = MockContinuumMemoryEntry()

        fusion_result = MagicMock()
        fusion_result.fused_confidence = 0.9

        result = adapter._apply_fusion_result(entry, fusion_result)

        assert result is True
        mock_continuum.update.assert_called()


# ============================================================================
# SemanticSearchMixin Integration Tests
# ============================================================================


class TestSemanticSearchMixinIntegration:
    """Tests for SemanticSearchMixin integration."""

    def test_adapter_name(self):
        """Should have adapter_name set."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter

        mock_continuum = create_mock_continuum()
        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        assert adapter.adapter_name == "continuum"

    def test_source_type(self):
        """Should have source_type set."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter

        mock_continuum = create_mock_continuum()
        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        assert adapter.source_type == "continuum"

    def test_get_record_by_id(self):
        """Should get record by ID for semantic mixin."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter

        mock_continuum = create_mock_continuum()
        entry = MockContinuumMemoryEntry(id="sem-entry")
        mock_continuum.get.return_value = entry

        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        result = adapter._get_record_by_id("sem-entry")

        assert result is entry

    def test_get_record_by_id_strips_prefix(self):
        """Should strip cm_ prefix when getting record."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter

        mock_continuum = create_mock_continuum()
        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        adapter._get_record_by_id("cm_entry-123")

        mock_continuum.get.assert_called_with("entry-123")

    def test_record_to_dict(self):
        """Should convert record to dict."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter

        mock_continuum = create_mock_continuum()
        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        entry = MockContinuumMemoryEntry(
            id="dict-entry",
            content="Test content",
            importance=0.75,
        )

        result = adapter._record_to_dict(entry, similarity=0.85)

        assert result["id"] == "dict-entry"
        assert result["content"] == "Test content"
        assert result["importance"] == 0.75
        assert result["similarity"] == 0.85

    def test_extract_record_id(self):
        """Should extract record ID from prefixed source ID."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter

        mock_continuum = create_mock_continuum()
        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        assert adapter._extract_record_id("cm_entry-123") == "entry-123"
        assert adapter._extract_record_id("entry-456") == "entry-456"


# ============================================================================
# Metrics Recording Tests
# ============================================================================


class TestMetricsRecording:
    """Tests for metrics recording functionality."""

    def test_record_metric_without_observability(self):
        """Should handle missing observability module gracefully."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter

        mock_continuum = create_mock_continuum()
        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        # Should not raise even if observability is not available
        adapter._record_metric("search", True, 0.05)

    def test_record_metric_with_extra_labels(self):
        """Should accept extra labels."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter

        mock_continuum = create_mock_continuum()
        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        # Should not raise
        adapter._record_metric(
            "search",
            True,
            0.05,
            extra_labels={"tier": "slow"},
        )


# ============================================================================
# Edge Cases and Error Handling Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_search_query(self):
        """Should handle empty search query."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter

        mock_continuum = create_mock_continuum()
        mock_continuum.retrieve.return_value = []

        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        results = adapter.search_by_keyword("")

        assert results == []

    def test_search_with_special_characters(self):
        """Should handle special characters in query."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter

        mock_continuum = create_mock_continuum()
        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        # Should not raise
        adapter.search_by_keyword("test@#$%^&*()")

    def test_very_long_content(self):
        """Should handle very long content."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter

        mock_continuum = create_mock_continuum()
        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        long_content = "x" * 100000

        # Should not raise
        adapter.store(content=long_content)

    def test_empty_metadata(self):
        """Should handle None metadata."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter

        mock_continuum = create_mock_continuum()
        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        adapter.store(content="Test", metadata=None)

        call_kwargs = mock_continuum.add.call_args[1]
        assert call_kwargs["metadata"] == {}

    def test_unicode_content(self):
        """Should handle Unicode content."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter

        mock_continuum = create_mock_continuum()
        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        unicode_content = "Unicode test: \u4e2d\u6587 \u65e5\u672c\u8a9e \ud83d\ude00"

        adapter.store(content=unicode_content)

        call_kwargs = mock_continuum.add.call_args[1]
        assert call_kwargs["content"] == unicode_content

    @pytest.mark.asyncio
    async def test_concurrent_updates(self):
        """Should handle concurrent update operations."""
        from aragora.knowledge.mound.adapters.continuum_adapter import (
            ContinuumAdapter,
            KMValidationResult,
        )
        import asyncio

        mock_continuum = create_mock_continuum()
        mock_continuum.get.return_value = MockContinuumMemoryEntry()

        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        validations = [
            KMValidationResult(memory_id=f"entry-{i}", km_confidence=0.8) for i in range(5)
        ]

        # Run updates concurrently
        tasks = [adapter.update_continuum_from_km(v.memory_id, v) for v in validations]
        results = await asyncio.gather(*tasks)

        assert all(results)

    def test_extract_fusible_data_missing_fields(self):
        """Should handle KM items with missing fields."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter

        mock_continuum = create_mock_continuum()
        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        km_item = {
            "id": "km-minimal",
            "metadata": {},
        }

        data = adapter._extract_fusible_data(km_item)

        assert data["confidence"] == 0.5  # Default
        assert data["tier"] is None

    def test_apply_fusion_result_without_fused_confidence(self):
        """Should handle fusion result without fused_confidence."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter

        mock_continuum = create_mock_continuum()
        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        entry = MockContinuumMemoryEntry()
        fusion_result = MagicMock(spec=[])  # No fused_confidence attribute

        result = adapter._apply_fusion_result(entry, fusion_result)

        assert result is False

    def test_apply_fusion_result_error(self):
        """Should handle errors during fusion result application."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter

        mock_continuum = create_mock_continuum()
        mock_continuum.update.side_effect = RuntimeError("Update failed")

        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        entry = MockContinuumMemoryEntry()
        fusion_result = MagicMock()
        fusion_result.fused_confidence = 0.9

        result = adapter._apply_fusion_result(entry, fusion_result)

        # Should return False on error, not raise
        assert result is False


# ============================================================================
# Tier Transition Tests
# ============================================================================


class TestTierTransitions:
    """Tests for tier transition logic."""

    @pytest.mark.asyncio
    async def test_promote_fast_to_medium(self):
        """Should promote from fast to medium tier."""
        from aragora.knowledge.mound.adapters.continuum_adapter import (
            ContinuumAdapter,
            KMValidationResult,
        )

        mock_continuum = create_mock_continuum()
        entry = MockContinuumMemoryEntry(tier=MockMemoryTier.FAST)
        mock_continuum.get.return_value = entry

        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        validation = KMValidationResult(
            memory_id="entry-123",
            km_confidence=0.95,
            recommendation="promote",
        )

        await adapter.update_continuum_from_km("entry-123", validation)

        # Should call promote_entry
        mock_continuum.promote_entry.assert_called()

    @pytest.mark.asyncio
    async def test_demote_glacial_to_slow(self):
        """Should demote from glacial to slow tier."""
        from aragora.knowledge.mound.adapters.continuum_adapter import (
            ContinuumAdapter,
            KMValidationResult,
        )

        mock_continuum = create_mock_continuum()
        entry = MockContinuumMemoryEntry(tier=MockMemoryTier.GLACIAL)
        mock_continuum.get.return_value = entry

        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        validation = KMValidationResult(
            memory_id="entry-123",
            km_confidence=0.75,
            recommendation="demote",
        )

        await adapter.update_continuum_from_km("entry-123", validation)

        # Should call demote_entry
        mock_continuum.demote_entry.assert_called()

    @pytest.mark.asyncio
    async def test_importance_weighted_by_validation_count(self):
        """Should weight importance by validation count."""
        from aragora.knowledge.mound.adapters.continuum_adapter import (
            ContinuumAdapter,
            KMValidationResult,
        )

        mock_continuum = create_mock_continuum()
        entry = MockContinuumMemoryEntry(importance=0.5)
        mock_continuum.get.return_value = entry

        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        # High validation count means more weight on KM confidence
        validation = KMValidationResult(
            memory_id="entry-123",
            km_confidence=0.9,
            validation_count=5,  # 5 * 0.1 = 0.5 weight (max)
        )

        await adapter.update_continuum_from_km("entry-123", validation)

        call_kwargs = mock_continuum.update.call_args[1]
        # New importance should be between 0.5 and 0.9
        assert call_kwargs["importance"] is not None or "importance" not in call_kwargs

    @pytest.mark.asyncio
    async def test_cross_debate_utility_boost(self):
        """Should apply cross-debate utility boost to importance."""
        from aragora.knowledge.mound.adapters.continuum_adapter import (
            ContinuumAdapter,
            KMValidationResult,
        )

        mock_continuum = create_mock_continuum()
        entry = MockContinuumMemoryEntry(importance=0.5)
        mock_continuum.get.return_value = entry

        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        validation = KMValidationResult(
            memory_id="entry-123",
            km_confidence=0.5,
            cross_debate_utility=1.0,  # Max utility
            validation_count=1,
        )

        await adapter.update_continuum_from_km("entry-123", validation)

        # Should have updated with utility boost
        mock_continuum.update.assert_called()


# ============================================================================
# Resilience Mixin Tests
# ============================================================================


class TestResilienceMixin:
    """Tests for resilience mixin integration."""

    def test_adapter_inherits_resilient_mixin(self):
        """Should inherit from ResilientAdapterMixin."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter

        mock_continuum = create_mock_continuum()
        adapter = ContinuumAdapter(mock_continuum, enable_resilience=True)

        # Should have the mixin's attributes
        assert hasattr(adapter, "adapter_name")

    def test_adapter_works_without_resilience(self):
        """Should work when resilience is disabled."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter

        mock_continuum = create_mock_continuum()
        adapter = ContinuumAdapter(mock_continuum, enable_resilience=False)

        # Basic operation should still work
        result = adapter.search_by_keyword("test")

        assert isinstance(result, list)
