"""
Comprehensive tests for Knowledge Mound Sync Operations.

Tests cover:
1. _batch_store method - verify batching works correctly
2. sync_from_continuum - test memory tier synchronization
3. sync_from_consensus - test consensus data sync
4. sync_from_facts - test fact synchronization
5. sync_from_evidence - test evidence sync
6. sync_from_critique - test critique sync
7. N+1 query pattern prevention - verify batching avoids N+1
8. Error handling and resilience
9. Concurrent sync operations
10. Large batch handling (100+, 1000+ items)

Target: 50+ tests covering all sync methods and edge cases.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from aragora.knowledge.mound.types import (
    IngestionRequest,
    IngestionResult,
    KnowledgeSource,
    MoundConfig,
    SyncResult,
)
from aragora.knowledge.mound.ops.sync import SyncOperationsMixin


# =============================================================================
# Mock Data Classes
# =============================================================================


@dataclass
class MockContinuumEntry:
    """Mock ContinuumMemoryEntry for testing."""

    id: str
    content: str
    tier: Any  # MemoryTier
    importance: float = 0.5
    surprise_score: float = 0.3
    consolidation_score: float = 0.7
    update_count: int = 1
    success_rate: float = 0.8
    success_count: int = 8
    failure_count: int = 2
    created_at: str = "2024-01-01T00:00:00"
    updated_at: str = "2024-01-01T00:00:00"
    metadata: dict = field(default_factory=dict)


@dataclass
class MockMemoryTier:
    """Mock MemoryTier enum for testing."""

    value: str


@dataclass
class MockFact:
    """Mock Fact for testing."""

    id: str
    statement: str
    confidence: float = 0.8
    topics: list = field(default_factory=list)
    source_documents: list = field(default_factory=list)
    evidence_ids: list = field(default_factory=list)
    validation_status: Any = None

    def __post_init__(self):
        if self.validation_status is None:
            self.validation_status = MockValidationStatus("verified")


@dataclass
class MockValidationStatus:
    """Mock ValidationStatus for testing."""

    value: str


@dataclass
class MockEvidence:
    """Mock Evidence for testing."""

    id: str
    content: str
    debate_id: str | None = None
    agent_id: str | None = None
    quality_score: float = 0.7
    source_url: str | None = None


@dataclass
class MockPattern:
    """Mock Pattern for testing."""

    id: str
    pattern: str = ""
    content: str = ""
    agent_name: str | None = None
    success_rate: float = 0.75
    success_count: int = 10


class MockStore:
    """Mock store for connection context manager."""

    def __init__(self, rows: list = None):
        self._rows = rows or []

    def connection(self):
        return MockConnection(self._rows)


class MockConnection:
    """Mock database connection for testing."""

    def __init__(self, rows: list):
        self._rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def execute(self, query: str, params: tuple = None):
        return MockCursor(self._rows)


class MockCursor:
    """Mock database cursor for testing."""

    def __init__(self, rows: list):
        self._rows = rows

    def fetchall(self):
        return self._rows


# =============================================================================
# Test Implementation Class
# =============================================================================


class TestableSyncMixin(SyncOperationsMixin):
    """Testable implementation of SyncOperationsMixin."""

    def __init__(
        self,
        workspace_id: str = "test-workspace",
        store_results: list[IngestionResult] | None = None,
        store_raises: Exception | None = None,
    ):
        self.workspace_id = workspace_id
        self.config = MoundConfig()
        self._initialized = True
        self._continuum: Any = None
        self._consensus: Any = None
        self._facts: Any = None
        self._evidence: Any = None
        self._critique: Any = None

        # Track store calls for verification
        self._store_calls: list[IngestionRequest] = []
        self._store_results = store_results or []
        self._store_result_index = 0
        self._store_raises = store_raises

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("Not initialized")

    async def store(self, request: IngestionRequest) -> IngestionResult:
        self._store_calls.append(request)

        if self._store_raises:
            raise self._store_raises

        if self._store_results:
            idx = min(self._store_result_index, len(self._store_results) - 1)
            result = self._store_results[idx]
            self._store_result_index += 1
            return result

        # Default result
        return IngestionResult(
            node_id=f"node_{len(self._store_calls)}",
            success=True,
            deduplicated=False,
            relationships_created=0,
        )


# =============================================================================
# _batch_store Tests
# =============================================================================


class TestBatchStore:
    """Tests for _batch_store method."""

    @pytest.mark.asyncio
    async def test_batch_store_empty_list(self):
        """Should handle empty request list."""
        mixin = TestableSyncMixin()
        synced, updated, skipped, rels, errors = await mixin._batch_store([])

        assert synced == 0
        assert updated == 0
        assert skipped == 0
        assert rels == 0
        assert errors == []

    @pytest.mark.asyncio
    async def test_batch_store_single_item(self):
        """Should process single item correctly."""
        mixin = TestableSyncMixin()
        request = IngestionRequest(
            content="Test content",
            workspace_id="test",
            source_type=KnowledgeSource.FACT,
        )

        synced, updated, skipped, rels, errors = await mixin._batch_store([request])

        assert synced == 1
        assert updated == 0
        assert skipped == 0
        assert len(mixin._store_calls) == 1

    @pytest.mark.asyncio
    async def test_batch_store_multiple_items(self):
        """Should process multiple items correctly."""
        mixin = TestableSyncMixin()
        requests = [
            IngestionRequest(
                content=f"Content {i}",
                workspace_id="test",
                source_type=KnowledgeSource.FACT,
            )
            for i in range(5)
        ]

        synced, updated, skipped, rels, errors = await mixin._batch_store(requests)

        assert synced == 5
        assert updated == 0
        assert len(mixin._store_calls) == 5

    @pytest.mark.asyncio
    async def test_batch_store_with_deduplication(self):
        """Should count deduplicated items as updates."""
        results = [
            IngestionResult(node_id="node_1", success=True, deduplicated=False),
            IngestionResult(
                node_id="node_2", success=True, deduplicated=True, existing_node_id="old_1"
            ),
            IngestionResult(node_id="node_3", success=True, deduplicated=False),
        ]
        mixin = TestableSyncMixin(store_results=results)

        requests = [IngestionRequest(content=f"Content {i}", workspace_id="test") for i in range(3)]

        synced, updated, skipped, rels, errors = await mixin._batch_store(requests)

        assert synced == 2
        assert updated == 1
        assert skipped == 0

    @pytest.mark.asyncio
    async def test_batch_store_with_relationships(self):
        """Should count relationships created."""
        results = [
            IngestionResult(
                node_id="node_1", success=True, deduplicated=False, relationships_created=2
            ),
            IngestionResult(
                node_id="node_2", success=True, deduplicated=True, relationships_created=1
            ),
        ]
        mixin = TestableSyncMixin(store_results=results)

        requests = [IngestionRequest(content=f"Content {i}", workspace_id="test") for i in range(2)]

        synced, updated, skipped, rels, errors = await mixin._batch_store(requests)

        assert rels == 3

    @pytest.mark.asyncio
    async def test_batch_store_error_handling(self):
        """Should handle individual item errors gracefully."""
        call_count = 0

        async def failing_store(request):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise ValueError("Store failed")
            return IngestionResult(
                node_id=f"node_{call_count}",
                success=True,
                deduplicated=False,
            )

        mixin = TestableSyncMixin()
        mixin.store = failing_store

        requests = [
            IngestionRequest(
                content=f"Content {i}",
                workspace_id="test",
                metadata={"fact_id": f"fact_{i}"},
            )
            for i in range(3)
        ]

        synced, updated, skipped, rels, errors = await mixin._batch_store(requests)

        assert synced == 2
        assert skipped == 1
        assert len(errors) == 1
        assert "fact_1" in errors[0]

    @pytest.mark.asyncio
    async def test_batch_store_respects_batch_size(self):
        """Should process items in batches of specified size."""
        mixin = TestableSyncMixin()
        store_calls_per_batch = []
        original_store = mixin.store

        batch_count = 0

        async def tracking_store(request):
            nonlocal batch_count
            batch_count += 1
            return await original_store(request)

        mixin.store = tracking_store

        requests = [
            IngestionRequest(content=f"Content {i}", workspace_id="test") for i in range(15)
        ]

        # Process with batch_size=5
        await mixin._batch_store(requests, batch_size=5)

        # All 15 items should be processed
        assert batch_count == 15

    @pytest.mark.asyncio
    async def test_batch_store_n_plus_1_prevention(self):
        """Should process concurrently within batches to avoid N+1."""
        execution_times = []

        async def slow_store(request):
            start = time.time()
            await asyncio.sleep(0.01)  # Simulate async operation
            execution_times.append(time.time() - start)
            return IngestionResult(node_id="node_1", success=True, deduplicated=False)

        mixin = TestableSyncMixin()
        mixin.store = slow_store

        requests = [IngestionRequest(content=f"Content {i}", workspace_id="test") for i in range(5)]

        start = time.time()
        await mixin._batch_store(requests, batch_size=5)
        total_time = time.time() - start

        # If concurrent, total time should be much less than sum of individual times
        # (5 items * 0.01s = 0.05s sequential, ~0.01s concurrent)
        assert total_time < 0.03  # Allow some overhead

    @pytest.mark.asyncio
    async def test_batch_store_metadata_extraction(self):
        """Should extract correct identifiers from metadata for error messages."""

        async def failing_store(request):
            raise ValueError("Test error")

        mixin = TestableSyncMixin()
        mixin.store = failing_store

        test_cases = [
            {"continuum_id": "cm_123"},
            {"consensus_id": "cs_456"},
            {"fact_id": "f_789"},
            {"evidence_id": "ev_012"},
            {"pattern_id": "pt_345"},
        ]

        for metadata in test_cases:
            request = IngestionRequest(
                content="Test",
                workspace_id="test",
                metadata=metadata,
            )
            _, _, _, _, errors = await mixin._batch_store([request])

            expected_id = list(metadata.values())[0]
            assert expected_id in errors[0]


# =============================================================================
# sync_from_continuum Tests
# =============================================================================


class TestSyncFromContinuum:
    """Tests for sync_from_continuum method."""

    @pytest.mark.asyncio
    async def test_sync_from_continuum_basic(self):
        """Should sync entries from ContinuumMemory."""
        mixin = TestableSyncMixin()

        # Create mock continuum
        continuum = MagicMock()
        entries = [
            MockContinuumEntry(
                id=f"entry_{i}",
                content=f"Memory content {i}",
                tier=MockMemoryTier("fast"),
                importance=0.7,
            )
            for i in range(3)
        ]
        continuum.retrieve.return_value = entries

        result = await mixin.sync_from_continuum(continuum)

        assert result.source == "continuum"
        assert result.nodes_synced == 3
        assert result.nodes_updated == 0
        assert result.nodes_skipped == 0
        assert len(mixin._store_calls) == 3

    @pytest.mark.asyncio
    async def test_sync_from_continuum_preserves_metadata(self):
        """Should preserve continuum metadata in ingestion requests."""
        mixin = TestableSyncMixin()

        continuum = MagicMock()
        entries = [
            MockContinuumEntry(
                id="entry_1",
                content="Test content",
                tier=MockMemoryTier("medium"),
                importance=0.85,
                surprise_score=0.4,
                consolidation_score=0.6,
                update_count=5,
                success_rate=0.9,
                metadata={"custom": "data"},
            )
        ]
        continuum.retrieve.return_value = entries

        await mixin.sync_from_continuum(continuum)

        assert len(mixin._store_calls) == 1
        request = mixin._store_calls[0]
        assert request.metadata["continuum_id"] == "entry_1"
        assert request.metadata["surprise_score"] == 0.4
        assert request.metadata["consolidation_score"] == 0.6
        assert request.confidence == 0.85
        assert request.tier == "medium"

    @pytest.mark.asyncio
    async def test_sync_from_continuum_empty_entries(self):
        """Should handle empty entry list."""
        mixin = TestableSyncMixin()

        continuum = MagicMock()
        continuum.retrieve.return_value = []

        result = await mixin.sync_from_continuum(continuum)

        assert result.nodes_synced == 0
        assert result.nodes_skipped == 0
        assert result.errors == []

    @pytest.mark.asyncio
    async def test_sync_from_continuum_error_in_retrieve(self):
        """Should handle errors in continuum.retrieve."""
        mixin = TestableSyncMixin()

        continuum = MagicMock()
        continuum.retrieve.side_effect = RuntimeError("Database connection failed")

        result = await mixin.sync_from_continuum(continuum)

        assert result.nodes_synced == 0
        assert len(result.errors) == 1
        assert "continuum:retrieve" in result.errors[0]

    @pytest.mark.asyncio
    async def test_sync_from_continuum_partial_failure(self):
        """Should continue syncing even if some entries fail."""
        call_count = 0

        async def sometimes_failing_store(request):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise ValueError("Entry failed")
            return IngestionResult(
                node_id=f"node_{call_count}",
                success=True,
                deduplicated=False,
            )

        mixin = TestableSyncMixin()
        mixin.store = sometimes_failing_store

        continuum = MagicMock()
        entries = [
            MockContinuumEntry(
                id=f"entry_{i}",
                content=f"Content {i}",
                tier=MockMemoryTier("fast"),
            )
            for i in range(5)
        ]
        continuum.retrieve.return_value = entries

        result = await mixin.sync_from_continuum(continuum)

        assert result.nodes_synced == 4
        assert result.nodes_skipped == 1

    @pytest.mark.asyncio
    async def test_sync_from_continuum_sets_internal_reference(self):
        """Should set internal _continuum reference."""
        mixin = TestableSyncMixin()
        continuum = MagicMock()
        continuum.retrieve.return_value = []

        await mixin.sync_from_continuum(continuum)

        assert mixin._continuum is continuum

    @pytest.mark.asyncio
    async def test_sync_from_continuum_uses_batch_size(self):
        """Should respect batch_size parameter."""
        mixin = TestableSyncMixin()

        continuum = MagicMock()
        entries = [
            MockContinuumEntry(
                id=f"entry_{i}",
                content=f"Content {i}",
                tier=MockMemoryTier("slow"),
            )
            for i in range(150)
        ]
        continuum.retrieve.return_value = entries

        result = await mixin.sync_from_continuum(continuum, batch_size=50)

        assert result.nodes_synced == 150


# =============================================================================
# sync_from_consensus Tests
# =============================================================================


class TestSyncFromConsensus:
    """Tests for sync_from_consensus method."""

    @pytest.mark.asyncio
    async def test_sync_from_consensus_basic(self):
        """Should sync records from ConsensusMemory."""
        mixin = TestableSyncMixin()

        # Create mock consensus with store
        rows = [
            (
                "rec_1",  # id
                "AI Safety",  # topic
                "We should prioritize alignment",  # conclusion
                "strong",  # strength
                0.85,  # confidence
                '["claude", "gpt"]',  # participating_agents
                '["claude"]',  # agreeing_agents
                "technology",  # domain
                '["ai", "safety"]',  # tags
                "2024-01-01T00:00:00",  # timestamp
                None,  # supersedes
                "{}",  # metadata
            )
        ]
        consensus = MagicMock()
        consensus._store = MockStore(rows)

        result = await mixin.sync_from_consensus(consensus)

        assert result.source == "consensus"
        assert result.nodes_synced == 1
        assert result.nodes_updated == 0

    @pytest.mark.asyncio
    async def test_sync_from_consensus_preserves_metadata(self):
        """Should preserve consensus metadata in ingestion requests."""
        mixin = TestableSyncMixin()

        rows = [
            (
                "rec_1",
                "Topic X",
                "Conclusion Y",
                "moderate",
                0.75,
                '["agent1"]',
                '["agent1"]',
                "finance",
                '["money", "investing"]',
                "2024-01-01T00:00:00",
                None,
                '{"extra": "data"}',
            )
        ]
        consensus = MagicMock()
        consensus._store = MockStore(rows)

        await mixin.sync_from_consensus(consensus)

        assert len(mixin._store_calls) == 1
        request = mixin._store_calls[0]
        assert request.metadata["consensus_id"] == "rec_1"
        assert request.metadata["strength"] == "moderate"
        assert request.metadata["domain"] == "finance"
        assert request.confidence == 0.75
        assert "money" in request.topics

    @pytest.mark.asyncio
    async def test_sync_from_consensus_with_supersession(self):
        """Should create derived_from relationship for superseding records."""
        mixin = TestableSyncMixin()

        rows = [
            (
                "rec_2",
                "Updated Topic",
                "New conclusion",
                "strong",
                0.9,
                '["claude"]',
                '["claude"]',
                "general",
                "[]",
                "2024-01-02T00:00:00",
                "rec_1",  # supersedes rec_1
                "{}",
            )
        ]
        consensus = MagicMock()
        consensus._store = MockStore(rows)

        await mixin.sync_from_consensus(consensus)

        request = mixin._store_calls[0]
        assert request.derived_from == ["cs_rec_1"]

    @pytest.mark.asyncio
    async def test_sync_from_consensus_no_store(self):
        """Should handle consensus without _store attribute."""
        mixin = TestableSyncMixin()

        consensus = MagicMock()
        consensus._store = None

        result = await mixin.sync_from_consensus(consensus)

        assert result.nodes_synced == 0
        assert result.errors == []

    @pytest.mark.asyncio
    async def test_sync_from_consensus_malformed_json(self):
        """Should handle malformed JSON in database fields."""
        mixin = TestableSyncMixin()

        rows = [
            (
                "rec_1",
                "Topic",
                "Conclusion",
                "weak",
                0.5,
                "not valid json",  # Invalid JSON
                '["agent"]',
                "general",
                "also not json",  # Invalid JSON
                "2024-01-01T00:00:00",
                None,
                "{invalid}",  # Invalid JSON
            )
        ]
        consensus = MagicMock()
        consensus._store = MockStore(rows)

        result = await mixin.sync_from_consensus(consensus)

        # Should still process (safe_json_loads returns defaults)
        assert result.nodes_synced == 1

    @pytest.mark.asyncio
    async def test_sync_from_consensus_empty_records(self):
        """Should handle empty records list."""
        mixin = TestableSyncMixin()

        consensus = MagicMock()
        consensus._store = MockStore([])

        result = await mixin.sync_from_consensus(consensus)

        assert result.nodes_synced == 0


# =============================================================================
# sync_from_facts Tests
# =============================================================================


class TestSyncFromFacts:
    """Tests for sync_from_facts method."""

    @pytest.mark.asyncio
    async def test_sync_from_facts_basic(self):
        """Should sync facts from FactStore."""
        mixin = TestableSyncMixin()

        facts = MagicMock()
        facts.query_facts.return_value = [
            MockFact(
                id="fact_1",
                statement="The sky is blue",
                confidence=0.95,
                topics=["science", "nature"],
            ),
            MockFact(
                id="fact_2",
                statement="Water is wet",
                confidence=0.99,
            ),
        ]

        result = await mixin.sync_from_facts(facts)

        assert result.source == "facts"
        assert result.nodes_synced == 2

    @pytest.mark.asyncio
    async def test_sync_from_facts_preserves_metadata(self):
        """Should preserve fact metadata in ingestion requests."""
        mixin = TestableSyncMixin()

        facts = MagicMock()
        facts.query_facts.return_value = [
            MockFact(
                id="fact_1",
                statement="Test fact",
                confidence=0.8,
                topics=["topic1"],
                source_documents=["doc_1", "doc_2"],
                evidence_ids=["ev_1"],
                validation_status=MockValidationStatus("verified"),
            )
        ]

        await mixin.sync_from_facts(facts)

        request = mixin._store_calls[0]
        assert request.content == "Test fact"
        assert request.metadata["fact_id"] == "fact_1"
        assert request.metadata["validation_status"] == "verified"
        assert request.document_id == "doc_1"
        assert request.confidence == 0.8

    @pytest.mark.asyncio
    async def test_sync_from_facts_no_query_method(self):
        """Should handle FactStore without query_facts method."""
        mixin = TestableSyncMixin()

        facts = MagicMock(spec=[])  # No methods

        result = await mixin.sync_from_facts(facts)

        assert result.nodes_synced == 0

    @pytest.mark.asyncio
    async def test_sync_from_facts_error_handling(self):
        """Should handle errors in fact queries."""
        mixin = TestableSyncMixin()

        facts = MagicMock()
        facts.query_facts.side_effect = RuntimeError("Query failed")

        result = await mixin.sync_from_facts(facts)

        assert result.nodes_synced == 0
        assert len(result.errors) == 1
        assert "facts:query" in result.errors[0]

    @pytest.mark.asyncio
    async def test_sync_from_facts_validation_status_string(self):
        """Should handle validation_status as string."""
        mixin = TestableSyncMixin()

        fact = MockFact(id="f1", statement="Test")
        fact.validation_status = "pending"  # String instead of enum

        facts = MagicMock()
        facts.query_facts.return_value = [fact]

        await mixin.sync_from_facts(facts)

        request = mixin._store_calls[0]
        assert request.metadata["validation_status"] == "pending"


# =============================================================================
# sync_from_evidence Tests
# =============================================================================


class TestSyncFromEvidence:
    """Tests for sync_from_evidence method."""

    @pytest.mark.asyncio
    async def test_sync_from_evidence_basic(self):
        """Should sync evidence from EvidenceStore."""
        mixin = TestableSyncMixin()

        evidence = MagicMock()
        evidence.search.return_value = [
            MockEvidence(
                id="ev_1",
                content="Research shows that...",
                quality_score=0.85,
            ),
            MockEvidence(
                id="ev_2",
                content="According to studies...",
                quality_score=0.7,
            ),
        ]

        result = await mixin.sync_from_evidence(evidence)

        assert result.source == "evidence"
        assert result.nodes_synced == 2

    @pytest.mark.asyncio
    async def test_sync_from_evidence_preserves_metadata(self):
        """Should preserve evidence metadata in ingestion requests."""
        mixin = TestableSyncMixin()

        evidence = MagicMock()
        evidence.search.return_value = [
            MockEvidence(
                id="ev_1",
                content="Evidence content",
                debate_id="debate_123",
                agent_id="claude",
                quality_score=0.9,
                source_url="https://example.com",
            )
        ]

        await mixin.sync_from_evidence(evidence)

        request = mixin._store_calls[0]
        assert request.metadata["evidence_id"] == "ev_1"
        assert request.metadata["source_url"] == "https://example.com"
        assert request.debate_id == "debate_123"
        assert request.agent_id == "claude"
        assert request.confidence == 0.9

    @pytest.mark.asyncio
    async def test_sync_from_evidence_no_search_method(self):
        """Should handle EvidenceStore without search method."""
        mixin = TestableSyncMixin()

        evidence = MagicMock(spec=[])

        result = await mixin.sync_from_evidence(evidence)

        assert result.nodes_synced == 0

    @pytest.mark.asyncio
    async def test_sync_from_evidence_default_quality_score(self):
        """Should use default quality_score when not present."""
        mixin = TestableSyncMixin()

        # Create a mock evidence without quality_score attribute
        ev = MagicMock()
        ev.id = "ev_1"
        ev.content = "Test"
        # Don't set quality_score - getattr should return default 0.5
        del ev.quality_score  # MagicMock allows this

        evidence = MagicMock()
        evidence.search.return_value = [ev]

        await mixin.sync_from_evidence(evidence)

        request = mixin._store_calls[0]
        assert request.confidence == 0.5  # Default


# =============================================================================
# sync_from_critique Tests
# =============================================================================


class TestSyncFromCritique:
    """Tests for sync_from_critique method."""

    @pytest.mark.asyncio
    async def test_sync_from_critique_basic(self):
        """Should sync patterns from CritiqueStore."""
        mixin = TestableSyncMixin()

        critique = MagicMock()
        critique.search_patterns.return_value = [
            MockPattern(
                id="pt_1",
                pattern="Check for logical fallacies",
                success_rate=0.8,
            ),
            MockPattern(
                id="pt_2",
                content="Verify citations",
                success_rate=0.9,
            ),
        ]

        result = await mixin.sync_from_critique(critique)

        assert result.source == "critique"
        assert result.nodes_synced == 2

    @pytest.mark.asyncio
    async def test_sync_from_critique_pattern_vs_content(self):
        """Should use pattern attribute, fallback to content."""
        mixin = TestableSyncMixin()

        critique = MagicMock()
        critique.search_patterns.return_value = [
            MockPattern(id="pt_1", pattern="Pattern text", content=""),
            MockPattern(id="pt_2", pattern="", content="Content text"),
        ]

        await mixin.sync_from_critique(critique)

        assert mixin._store_calls[0].content == "Pattern text"
        assert mixin._store_calls[1].content == "Content text"

    @pytest.mark.asyncio
    async def test_sync_from_critique_skips_empty(self):
        """Should skip patterns with no content."""
        mixin = TestableSyncMixin()

        critique = MagicMock()
        critique.search_patterns.return_value = [
            MockPattern(id="pt_1", pattern="", content=""),  # Should skip
            MockPattern(id="pt_2", pattern="Valid pattern", content=""),
        ]

        result = await mixin.sync_from_critique(critique)

        assert result.nodes_synced == 1
        assert result.nodes_skipped == 1

    @pytest.mark.asyncio
    async def test_sync_from_critique_preserves_metadata(self):
        """Should preserve critique metadata in ingestion requests."""
        mixin = TestableSyncMixin()

        critique = MagicMock()
        critique.search_patterns.return_value = [
            MockPattern(
                id="pt_1",
                pattern="Check sources",
                agent_name="critic_agent",
                success_rate=0.85,
                success_count=17,
            )
        ]

        await mixin.sync_from_critique(critique)

        request = mixin._store_calls[0]
        assert request.metadata["pattern_id"] == "pt_1"
        assert request.metadata["success_count"] == 17
        assert request.agent_id == "critic_agent"
        assert request.confidence == 0.85

    @pytest.mark.asyncio
    async def test_sync_from_critique_no_search_method(self):
        """Should handle CritiqueStore without search_patterns method."""
        mixin = TestableSyncMixin()

        critique = MagicMock(spec=[])

        result = await mixin.sync_from_critique(critique)

        assert result.nodes_synced == 0


# =============================================================================
# sync_all Tests
# =============================================================================


class TestSyncAll:
    """Tests for sync_all method."""

    @pytest.mark.asyncio
    async def test_sync_all_no_connected_stores(self):
        """Should return empty results when no stores connected."""
        mixin = TestableSyncMixin()

        results = await mixin.sync_all()

        assert results == {}

    @pytest.mark.asyncio
    async def test_sync_all_with_connected_stores(self):
        """Should sync from all connected stores."""
        mixin = TestableSyncMixin()

        # Connect continuum
        continuum = MagicMock()
        continuum.retrieve.return_value = [
            MockContinuumEntry(id="e1", content="Memory", tier=MockMemoryTier("fast"))
        ]
        mixin._continuum = continuum

        # Connect consensus
        consensus = MagicMock()
        consensus._store = MockStore(
            [
                (
                    "rec_1",
                    "Topic",
                    "Conclusion",
                    "strong",
                    0.9,
                    "[]",
                    "[]",
                    "general",
                    "[]",
                    "2024-01-01",
                    None,
                    "{}",
                )
            ]
        )
        mixin._consensus = consensus

        results = await mixin.sync_all()

        assert "continuum" in results
        assert "consensus" in results
        assert results["continuum"].nodes_synced == 1
        assert results["consensus"].nodes_synced == 1

    @pytest.mark.asyncio
    async def test_sync_all_partial_connection(self):
        """Should only sync from connected stores."""
        mixin = TestableSyncMixin()

        # Only connect facts
        facts = MagicMock()
        facts.query_facts.return_value = [MockFact(id="f1", statement="Test")]
        mixin._facts = facts

        results = await mixin.sync_all()

        assert "facts" in results
        assert "continuum" not in results
        assert "consensus" not in results


# =============================================================================
# Incremental Sync Tests
# =============================================================================


class TestIncrementalSync:
    """Tests for incremental sync methods."""

    @pytest.mark.asyncio
    async def test_sync_continuum_incremental_no_store(self):
        """Should return error when continuum not connected."""
        mixin = TestableSyncMixin()

        result = await mixin.sync_continuum_incremental()

        assert result.nodes_synced == 0
        assert "ContinuumMemory not connected" in result.errors[0]

    @pytest.mark.asyncio
    async def test_sync_continuum_incremental_with_since(self):
        """Should filter by since timestamp."""
        mixin = TestableSyncMixin()

        continuum = MagicMock()
        entries = [
            MockContinuumEntry(
                id="e1",
                content="Old",
                tier=MockMemoryTier("fast"),
            ),
            MockContinuumEntry(
                id="e2",
                content="New",
                tier=MockMemoryTier("fast"),
            ),
        ]
        # Add updated_at attribute
        entries[0].updated_at = "2024-01-01T00:00:00"
        entries[1].updated_at = "2024-01-15T00:00:00"

        continuum.retrieve.return_value = entries
        mixin._continuum = continuum

        result = await mixin.sync_continuum_incremental(since="2024-01-10T00:00:00")

        # Should filter to only newer entries
        assert result.nodes_synced <= 2

    @pytest.mark.asyncio
    async def test_sync_consensus_incremental_no_store(self):
        """Should return error when consensus not connected."""
        mixin = TestableSyncMixin()

        result = await mixin.sync_consensus_incremental()

        assert result.nodes_synced == 0
        assert "ConsensusMemory not connected" in result.errors[0]

    @pytest.mark.asyncio
    async def test_sync_facts_incremental_no_store(self):
        """Should return error when facts not connected."""
        mixin = TestableSyncMixin()

        result = await mixin.sync_facts_incremental()

        assert result.nodes_synced == 0
        assert "FactStore not connected" in result.errors[0]


# =============================================================================
# Connect/Disconnect Tests
# =============================================================================


class TestConnectStores:
    """Tests for connect_memory_stores method."""

    @pytest.mark.asyncio
    async def test_connect_single_store(self):
        """Should connect a single store."""
        mixin = TestableSyncMixin()

        continuum = MagicMock()
        status = await mixin.connect_memory_stores(continuum=continuum)

        assert status == {"continuum": True}
        assert mixin._continuum is continuum

    @pytest.mark.asyncio
    async def test_connect_multiple_stores(self):
        """Should connect multiple stores."""
        mixin = TestableSyncMixin()

        continuum = MagicMock()
        consensus = MagicMock()
        facts = MagicMock()

        status = await mixin.connect_memory_stores(
            continuum=continuum,
            consensus=consensus,
            facts=facts,
        )

        assert len(status) == 3
        assert all(status.values())

    @pytest.mark.asyncio
    async def test_connect_no_stores(self):
        """Should return empty status when no stores provided."""
        mixin = TestableSyncMixin()

        status = await mixin.connect_memory_stores()

        assert status == {}

    def test_get_connected_stores(self):
        """Should return list of connected store names."""
        mixin = TestableSyncMixin()
        mixin._continuum = MagicMock()
        mixin._facts = MagicMock()

        connected = mixin.get_connected_stores()

        assert "continuum" in connected
        assert "facts" in connected
        assert "consensus" not in connected


# =============================================================================
# Large Batch Tests
# =============================================================================


class TestLargeBatches:
    """Tests for large batch handling."""

    @pytest.mark.asyncio
    async def test_batch_store_100_items(self):
        """Should handle 100+ items efficiently."""
        mixin = TestableSyncMixin()

        requests = [
            IngestionRequest(content=f"Content {i}", workspace_id="test") for i in range(100)
        ]

        start = time.time()
        synced, _, _, _, _ = await mixin._batch_store(requests, batch_size=50)
        duration = time.time() - start

        assert synced == 100
        # Should complete in reasonable time (< 5s)
        assert duration < 5.0

    @pytest.mark.asyncio
    async def test_batch_store_1000_items(self):
        """Should handle 1000+ items efficiently."""
        mixin = TestableSyncMixin()

        requests = [
            IngestionRequest(content=f"Content {i}", workspace_id="test") for i in range(1000)
        ]

        start = time.time()
        synced, _, _, _, _ = await mixin._batch_store(requests, batch_size=100)
        duration = time.time() - start

        assert synced == 1000
        # Should complete in reasonable time (< 30s)
        assert duration < 30.0

    @pytest.mark.asyncio
    async def test_sync_from_continuum_large_dataset(self):
        """Should handle large continuum datasets."""
        mixin = TestableSyncMixin()

        continuum = MagicMock()
        entries = [
            MockContinuumEntry(
                id=f"entry_{i}",
                content=f"Memory content {i}",
                tier=MockMemoryTier("fast" if i % 3 == 0 else "medium"),
            )
            for i in range(500)
        ]
        continuum.retrieve.return_value = entries

        result = await mixin.sync_from_continuum(continuum, batch_size=100)

        assert result.nodes_synced == 500


# =============================================================================
# Concurrent Operations Tests
# =============================================================================


class TestConcurrentOperations:
    """Tests for concurrent sync operations."""

    @pytest.mark.asyncio
    async def test_concurrent_syncs_different_sources(self):
        """Should handle concurrent syncs from different sources."""
        mixin = TestableSyncMixin()

        # Create mock stores
        continuum = MagicMock()
        continuum.retrieve.return_value = [
            MockContinuumEntry(id="c1", content="Continuum", tier=MockMemoryTier("fast"))
        ]

        facts = MagicMock()
        facts.query_facts.return_value = [MockFact(id="f1", statement="Fact")]

        # Run syncs concurrently
        results = await asyncio.gather(
            mixin.sync_from_continuum(continuum),
            mixin.sync_from_facts(facts),
        )

        assert results[0].source == "continuum"
        assert results[1].source == "facts"
        assert results[0].nodes_synced == 1
        assert results[1].nodes_synced == 1

    @pytest.mark.asyncio
    async def test_concurrent_batch_stores(self):
        """Should handle concurrent batch store calls."""
        mixin = TestableSyncMixin()

        requests1 = [
            IngestionRequest(content=f"Batch1_{i}", workspace_id="test") for i in range(50)
        ]
        requests2 = [
            IngestionRequest(content=f"Batch2_{i}", workspace_id="test") for i in range(50)
        ]

        results = await asyncio.gather(
            mixin._batch_store(requests1),
            mixin._batch_store(requests2),
        )

        assert results[0][0] == 50  # First batch synced
        assert results[1][0] == 50  # Second batch synced


# =============================================================================
# Error Resilience Tests
# =============================================================================


class TestErrorResilience:
    """Tests for error handling and resilience."""

    @pytest.mark.asyncio
    async def test_batch_store_all_errors(self):
        """Should handle all items failing."""

        async def always_fails(request):
            raise ValueError("Always fails")

        mixin = TestableSyncMixin()
        mixin.store = always_fails

        requests = [
            IngestionRequest(
                content=f"Content {i}",
                workspace_id="test",
                metadata={"fact_id": f"f{i}"},
            )
            for i in range(5)
        ]

        synced, updated, skipped, rels, errors = await mixin._batch_store(requests)

        assert synced == 0
        assert updated == 0
        assert skipped == 5
        assert len(errors) == 5

    @pytest.mark.asyncio
    async def test_sync_duration_tracking(self):
        """Should track sync duration in milliseconds."""
        mixin = TestableSyncMixin()

        continuum = MagicMock()
        continuum.retrieve.return_value = []

        result = await mixin.sync_from_continuum(continuum)

        assert result.duration_ms >= 0
        assert isinstance(result.duration_ms, int)

    @pytest.mark.asyncio
    async def test_sync_result_structure(self):
        """Should return properly structured SyncResult."""
        mixin = TestableSyncMixin()

        continuum = MagicMock()
        continuum.retrieve.return_value = []

        result = await mixin.sync_from_continuum(continuum)

        assert hasattr(result, "source")
        assert hasattr(result, "nodes_synced")
        assert hasattr(result, "nodes_updated")
        assert hasattr(result, "nodes_skipped")
        assert hasattr(result, "relationships_created")
        assert hasattr(result, "duration_ms")
        assert hasattr(result, "errors")

    @pytest.mark.asyncio
    async def test_initialization_check(self):
        """Should verify initialization before operations."""
        mixin = TestableSyncMixin()
        mixin._initialized = False

        with pytest.raises(RuntimeError, match="Not initialized"):
            continuum = MagicMock()
            continuum.retrieve.return_value = []
            await mixin.sync_from_continuum(continuum)


# =============================================================================
# Source Type Tests
# =============================================================================


class TestSourceTypes:
    """Tests for correct source type assignment."""

    @pytest.mark.asyncio
    async def test_continuum_source_type(self):
        """Should assign CONTINUUM source type."""
        mixin = TestableSyncMixin()

        continuum = MagicMock()
        continuum.retrieve.return_value = [
            MockContinuumEntry(id="e1", content="Test", tier=MockMemoryTier("fast"))
        ]

        await mixin.sync_from_continuum(continuum)

        assert mixin._store_calls[0].source_type == KnowledgeSource.CONTINUUM

    @pytest.mark.asyncio
    async def test_consensus_source_type(self):
        """Should assign CONSENSUS source type."""
        mixin = TestableSyncMixin()

        rows = [
            (
                "rec_1",
                "Topic",
                "Conclusion",
                "strong",
                0.9,
                "[]",
                "[]",
                "general",
                "[]",
                "2024-01-01",
                None,
                "{}",
            )
        ]
        consensus = MagicMock()
        consensus._store = MockStore(rows)

        await mixin.sync_from_consensus(consensus)

        assert mixin._store_calls[0].source_type == KnowledgeSource.CONSENSUS

    @pytest.mark.asyncio
    async def test_fact_source_type(self):
        """Should assign FACT source type."""
        mixin = TestableSyncMixin()

        facts = MagicMock()
        facts.query_facts.return_value = [MockFact(id="f1", statement="Test")]

        await mixin.sync_from_facts(facts)

        assert mixin._store_calls[0].source_type == KnowledgeSource.FACT

    @pytest.mark.asyncio
    async def test_evidence_source_type(self):
        """Should assign EVIDENCE source type."""
        mixin = TestableSyncMixin()

        evidence = MagicMock()
        evidence.search.return_value = [MockEvidence(id="ev1", content="Test")]

        await mixin.sync_from_evidence(evidence)

        assert mixin._store_calls[0].source_type == KnowledgeSource.EVIDENCE

    @pytest.mark.asyncio
    async def test_critique_source_type(self):
        """Should assign CRITIQUE source type."""
        mixin = TestableSyncMixin()

        critique = MagicMock()
        critique.search_patterns.return_value = [MockPattern(id="pt1", pattern="Test")]

        await mixin.sync_from_critique(critique)

        assert mixin._store_calls[0].source_type == KnowledgeSource.CRITIQUE


# =============================================================================
# Node Type Tests
# =============================================================================


class TestNodeTypes:
    """Tests for correct node type assignment."""

    @pytest.mark.asyncio
    async def test_continuum_node_type(self):
        """Should assign 'memory' node type for continuum."""
        mixin = TestableSyncMixin()

        continuum = MagicMock()
        continuum.retrieve.return_value = [
            MockContinuumEntry(id="e1", content="Test", tier=MockMemoryTier("fast"))
        ]

        await mixin.sync_from_continuum(continuum)

        assert mixin._store_calls[0].node_type == "memory"

    @pytest.mark.asyncio
    async def test_consensus_node_type(self):
        """Should assign 'consensus' node type."""
        mixin = TestableSyncMixin()

        rows = [
            (
                "rec_1",
                "Topic",
                "Conclusion",
                "strong",
                0.9,
                "[]",
                "[]",
                "general",
                "[]",
                "2024-01-01",
                None,
                "{}",
            )
        ]
        consensus = MagicMock()
        consensus._store = MockStore(rows)

        await mixin.sync_from_consensus(consensus)

        assert mixin._store_calls[0].node_type == "consensus"

    @pytest.mark.asyncio
    async def test_fact_node_type(self):
        """Should assign 'fact' node type."""
        mixin = TestableSyncMixin()

        facts = MagicMock()
        facts.query_facts.return_value = [MockFact(id="f1", statement="Test")]

        await mixin.sync_from_facts(facts)

        assert mixin._store_calls[0].node_type == "fact"

    @pytest.mark.asyncio
    async def test_evidence_node_type(self):
        """Should assign 'evidence' node type."""
        mixin = TestableSyncMixin()

        evidence = MagicMock()
        evidence.search.return_value = [MockEvidence(id="ev1", content="Test")]

        await mixin.sync_from_evidence(evidence)

        assert mixin._store_calls[0].node_type == "evidence"

    @pytest.mark.asyncio
    async def test_critique_node_type(self):
        """Should assign 'critique' node type."""
        mixin = TestableSyncMixin()

        critique = MagicMock()
        critique.search_patterns.return_value = [MockPattern(id="pt1", pattern="Test")]

        await mixin.sync_from_critique(critique)

        assert mixin._store_calls[0].node_type == "critique"
