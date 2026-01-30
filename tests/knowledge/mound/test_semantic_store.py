"""
Comprehensive tests for the SemanticStore.

Tests cover:
1. Vector operations (pack/unpack, cosine similarity)
2. Similarity scoring
3. Embedding generation
4. Index operations (add, remove, update)
5. Search operations
6. Batch operations
7. Persistence and loading
8. Concurrent access
9. Memory management
10. Error handling edge cases

Run with: pytest tests/knowledge/mound/test_semantic_store.py -v
"""

from __future__ import annotations

import asyncio
import hashlib
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.knowledge.mound.semantic_store import (
    SemanticIndexEntry,
    SemanticSearchResult,
    SemanticStore,
)
from aragora.knowledge.mound.types import KnowledgeSource
from aragora.memory.embeddings import (
    EmbeddingProvider,
    cosine_similarity,
    pack_embedding,
    unpack_embedding,
)


# ============================================================================
# Test Fixtures
# ============================================================================


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider that generates deterministic embeddings."""

    def __init__(self, dimension: int = 128):
        super().__init__(dimension)
        self._call_count = 0
        self._embed_history: list[str] = []

    async def embed(self, text: str) -> list[float]:
        """Generate a deterministic embedding based on text hash."""
        self._call_count += 1
        self._embed_history.append(text)
        # Use hash to generate deterministic but unique embeddings
        h = hashlib.sha256(text.encode()).digest()
        embedding = []
        for i in range(self.dimension):
            # Take 4 bytes at a time, cycling through hash
            idx = (i * 4) % len(h)
            val = int.from_bytes(h[idx : idx + 4], "little", signed=True)
            embedding.append(val / (2**31))
        return embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        return [await self.embed(t) for t in texts]


@pytest.fixture
def mock_provider():
    """Create a mock embedding provider."""
    return MockEmbeddingProvider(dimension=128)


@pytest.fixture
def temp_db_path():
    """Create a temporary database path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test_semantic.db"


@pytest.fixture
async def semantic_store(temp_db_path, mock_provider):
    """Create a SemanticStore with mock provider."""
    store = SemanticStore(
        db_path=temp_db_path,
        embedding_provider=mock_provider,
        default_tenant_id="test_tenant",
    )
    yield store


@pytest.fixture
def sample_content():
    """Sample content for testing."""
    return {
        "fact1": "All contracts require a 90-day notice period.",
        "fact2": "Employment contracts must include termination clauses.",
        "fact3": "The legal team reviews all contracts before signing.",
        "policy1": "Company policy requires two approvals for purchases over $10,000.",
        "process1": "Invoice processing takes 5 business days on average.",
    }


# ============================================================================
# Vector Operations Tests
# ============================================================================


class TestVectorOperations:
    """Tests for vector pack/unpack and similarity calculations."""

    def test_pack_unpack_embedding_roundtrip(self):
        """Test that pack/unpack produces identical embeddings."""
        original = [0.1, -0.5, 0.999, -0.001, 0.0]
        packed = pack_embedding(original)
        unpacked = unpack_embedding(packed)

        assert len(unpacked) == len(original)
        for o, u in zip(original, unpacked):
            assert abs(o - u) < 1e-6

    def test_pack_embedding_produces_bytes(self):
        """Test that pack_embedding returns bytes."""
        embedding = [0.5] * 10
        packed = pack_embedding(embedding)
        assert isinstance(packed, bytes)
        assert len(packed) == len(embedding) * 4  # float32 = 4 bytes

    def test_unpack_empty_embedding(self):
        """Test unpacking empty bytes."""
        unpacked = unpack_embedding(b"")
        assert unpacked == []

    def test_cosine_similarity_identical_vectors(self):
        """Test cosine similarity of identical vectors."""
        vec = [0.5, 0.3, -0.2, 0.8]
        similarity = cosine_similarity(vec, vec)
        assert abs(similarity - 1.0) < 1e-6

    def test_cosine_similarity_opposite_vectors(self):
        """Test cosine similarity of opposite vectors."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [-1.0, 0.0, 0.0]
        similarity = cosine_similarity(vec1, vec2)
        assert abs(similarity - (-1.0)) < 1e-6

    def test_cosine_similarity_orthogonal_vectors(self):
        """Test cosine similarity of orthogonal vectors."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        similarity = cosine_similarity(vec1, vec2)
        assert abs(similarity) < 1e-6

    def test_cosine_similarity_similar_vectors(self):
        """Test cosine similarity of similar but not identical vectors."""
        vec1 = [0.9, 0.1, 0.0, 0.0]
        vec2 = [0.95, 0.05, 0.0, 0.0]
        similarity = cosine_similarity(vec1, vec2)
        assert similarity > 0.99  # Very similar


# ============================================================================
# Similarity Scoring Tests
# ============================================================================


class TestSimilarityScoring:
    """Tests for similarity score calculations in search."""

    @pytest.mark.asyncio
    async def test_similarity_score_range(self, semantic_store, sample_content):
        """Test that similarity scores are in valid range [-1, 1]."""
        # Index items
        for name, content in list(sample_content.items())[:3]:
            await semantic_store.index_item(
                source_type=KnowledgeSource.FACT,
                source_id=name,
                content=content,
            )

        # Search
        results = await semantic_store.search_similar(
            query="contract notice period",
            min_similarity=-1.0,  # Accept all
        )

        for result in results:
            assert -1.0 <= result.similarity <= 1.0

    @pytest.mark.asyncio
    async def test_exact_content_highest_similarity(self, semantic_store):
        """Test that searching for exact content returns highest similarity."""
        content = "This is a very specific test document."
        await semantic_store.index_item(
            source_type=KnowledgeSource.FACT,
            source_id="exact_match",
            content=content,
        )

        results = await semantic_store.search_similar(query=content, limit=1)

        assert len(results) == 1
        assert results[0].source_id == "exact_match"
        assert results[0].similarity > 0.99  # Near perfect match


# ============================================================================
# Embedding Generation Tests
# ============================================================================


class TestEmbeddingGeneration:
    """Tests for embedding generation."""

    @pytest.mark.asyncio
    async def test_embedding_generated_on_index(self, semantic_store, mock_provider):
        """Test that embedding is generated when indexing."""
        km_id = await semantic_store.index_item(
            source_type=KnowledgeSource.FACT,
            source_id="test1",
            content="Test content for embedding",
        )

        entry = await semantic_store.get_entry(km_id)
        assert entry is not None
        assert len(entry.embedding) == mock_provider.dimension
        assert mock_provider._call_count == 1

    @pytest.mark.asyncio
    async def test_embedding_dimension_matches_provider(self, temp_db_path, mock_provider):
        """Test that stored embedding has correct dimension."""
        store = SemanticStore(db_path=temp_db_path, embedding_provider=mock_provider)

        km_id = await store.index_item(
            source_type=KnowledgeSource.FACT,
            source_id="test1",
            content="Test content",
        )

        entry = await store.get_entry(km_id)
        assert len(entry.embedding) == mock_provider.dimension
        assert store.embedding_dimension == mock_provider.dimension

    @pytest.mark.asyncio
    async def test_auto_detect_provider_fallback(self, temp_db_path):
        """Test that auto-detect falls back to hash-based embeddings."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("socket.socket"):
                store = SemanticStore(db_path=temp_db_path)
                assert store.embedding_provider is not None
                assert store.embedding_dimension > 0


# ============================================================================
# Index Operations Tests
# ============================================================================


class TestIndexOperations:
    """Tests for add, remove, update operations."""

    @pytest.mark.asyncio
    async def test_index_item_returns_km_id(self, semantic_store):
        """Test that index_item returns a Knowledge Mound ID."""
        km_id = await semantic_store.index_item(
            source_type=KnowledgeSource.FACT,
            source_id="fact_123",
            content="Test fact content",
        )

        assert km_id.startswith("km_")
        assert len(km_id) > 3

    @pytest.mark.asyncio
    async def test_index_item_with_all_params(self, semantic_store):
        """Test indexing with all optional parameters."""
        km_id = await semantic_store.index_item(
            source_type=KnowledgeSource.CONSENSUS,
            source_id="consensus_456",
            content="Consensus decision content",
            tenant_id="enterprise_team",
            domain="legal/contracts",
            importance=0.9,
            metadata={"debate_id": "debate_789"},
        )

        entry = await semantic_store.get_entry(km_id)
        assert entry is not None
        assert entry.source_type == "consensus"
        assert entry.domain == "legal/contracts"
        assert entry.importance == 0.9
        assert entry.metadata["debate_id"] == "debate_789"

    @pytest.mark.asyncio
    async def test_index_item_deduplication_by_hash(self, semantic_store):
        """Test that duplicate content returns existing ID."""
        content = "Duplicate content for testing"

        km_id1 = await semantic_store.index_item(
            source_type=KnowledgeSource.FACT,
            source_id="fact_1",
            content=content,
        )

        km_id2 = await semantic_store.index_item(
            source_type=KnowledgeSource.FACT,
            source_id="fact_2",
            content=content,
        )

        assert km_id1 == km_id2

    @pytest.mark.asyncio
    async def test_update_source_id(self, semantic_store):
        """Test updating source_id after indexing."""
        km_id = await semantic_store.index_item(
            source_type=KnowledgeSource.FACT,
            source_id="temp_id",
            content="Content to be updated",
        )

        result = await semantic_store.update_source_id(km_id, "permanent_id")
        assert result is True

        entry = await semantic_store.get_entry(km_id)
        assert entry.source_id == "permanent_id"

    @pytest.mark.asyncio
    async def test_update_source_id_nonexistent(self, semantic_store):
        """Test updating source_id for non-existent entry."""
        result = await semantic_store.update_source_id("km_nonexistent", "new_id")
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_entry_with_archive(self, semantic_store):
        """Test deleting entry with archiving."""
        km_id = await semantic_store.index_item(
            source_type=KnowledgeSource.FACT,
            source_id="to_delete",
            content="Content to be deleted",
        )

        result = await semantic_store.delete_entry(km_id, archive=True, reason="obsolete")
        assert result is True

        # Should not be retrievable
        entry = await semantic_store.get_entry(km_id)
        assert entry is None

    @pytest.mark.asyncio
    async def test_delete_entry_without_archive(self, semantic_store):
        """Test deleting entry without archiving."""
        km_id = await semantic_store.index_item(
            source_type=KnowledgeSource.FACT,
            source_id="to_delete",
            content="Content to be deleted",
        )

        result = await semantic_store.delete_entry(km_id, archive=False)
        assert result is True

        entry = await semantic_store.get_entry(km_id)
        assert entry is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_entry(self, semantic_store):
        """Test deleting non-existent entry."""
        result = await semantic_store.delete_entry("km_nonexistent")
        assert result is False


# ============================================================================
# Search Operations Tests
# ============================================================================


class TestSearchOperations:
    """Tests for semantic search functionality."""

    @pytest.mark.asyncio
    async def test_search_similar_basic(self, semantic_store, sample_content):
        """Test basic semantic search."""
        # Index items
        for name, content in sample_content.items():
            await semantic_store.index_item(
                source_type=KnowledgeSource.FACT,
                source_id=name,
                content=content,
            )

        results = await semantic_store.search_similar(
            query="contract requirements",
            limit=5,
            min_similarity=0.0,  # Accept all similarities for basic test
        )

        assert len(results) > 0
        assert all(isinstance(r, SemanticSearchResult) for r in results)

    @pytest.mark.asyncio
    async def test_search_with_domain_filter(self, semantic_store):
        """Test search with domain filtering."""
        await semantic_store.index_item(
            source_type=KnowledgeSource.FACT,
            source_id="legal_1",
            content="Legal contract clause",
            domain="legal/contracts",
        )
        await semantic_store.index_item(
            source_type=KnowledgeSource.FACT,
            source_id="hr_1",
            content="HR policy document",
            domain="hr/policies",
        )

        results = await semantic_store.search_similar(
            query="document",
            domain_filter="legal",
            min_similarity=0.0,
        )

        assert all(r.domain.startswith("legal") for r in results)

    @pytest.mark.asyncio
    async def test_search_with_source_type_filter(self, semantic_store):
        """Test search with source type filtering."""
        await semantic_store.index_item(
            source_type=KnowledgeSource.FACT,
            source_id="fact_1",
            content="A factual statement",
        )
        await semantic_store.index_item(
            source_type=KnowledgeSource.CONSENSUS,
            source_id="consensus_1",
            content="A consensus decision",
        )

        results = await semantic_store.search_similar(
            query="statement",
            source_types=["fact"],
            min_similarity=0.0,
        )

        assert all(r.source_type == "fact" for r in results)

    @pytest.mark.asyncio
    async def test_search_respects_min_similarity(self, semantic_store):
        """Test that search filters by minimum similarity."""
        await semantic_store.index_item(
            source_type=KnowledgeSource.FACT,
            source_id="test1",
            content="Very specific technical documentation",
        )

        # High threshold should return no results for unrelated query
        results = await semantic_store.search_similar(
            query="completely unrelated cooking recipe",
            min_similarity=0.99,
        )

        # With deterministic mock embeddings, unrelated content won't match
        # at 0.99 threshold
        assert len(results) == 0 or all(r.similarity >= 0.99 for r in results)

    @pytest.mark.asyncio
    async def test_search_respects_limit(self, semantic_store):
        """Test that search respects limit parameter."""
        # Index many items
        for i in range(20):
            await semantic_store.index_item(
                source_type=KnowledgeSource.FACT,
                source_id=f"fact_{i}",
                content=f"Content number {i} about testing",
            )

        results = await semantic_store.search_similar(
            query="testing content",
            limit=5,
            min_similarity=0.0,
        )

        assert len(results) <= 5

    @pytest.mark.asyncio
    async def test_search_tenant_isolation(self, semantic_store):
        """Test that search respects tenant isolation."""
        await semantic_store.index_item(
            source_type=KnowledgeSource.FACT,
            source_id="tenant_a_fact",
            content="Content for tenant A",
            tenant_id="tenant_a",
        )
        await semantic_store.index_item(
            source_type=KnowledgeSource.FACT,
            source_id="tenant_b_fact",
            content="Content for tenant B",
            tenant_id="tenant_b",
        )

        results = await semantic_store.search_similar(
            query="content",
            tenant_id="tenant_a",
            min_similarity=0.0,
        )

        assert all(r.tenant_id == "tenant_a" for r in results)


# ============================================================================
# Batch Operations Tests
# ============================================================================


class TestBatchOperations:
    """Tests for batch indexing operations."""

    @pytest.mark.asyncio
    async def test_index_batch_basic(self, semantic_store):
        """Test basic batch indexing."""
        items = [
            ("fact", "fact_1", "First fact content"),
            ("fact", "fact_2", "Second fact content"),
            ("consensus", "consensus_1", "Consensus content"),
        ]

        km_ids = await semantic_store.index_batch(items)

        assert len(km_ids) == 3
        assert all(id.startswith("km_") for id in km_ids)

    @pytest.mark.asyncio
    async def test_index_batch_with_domain(self, semantic_store):
        """Test batch indexing with domain."""
        items = [
            ("fact", "fact_1", "First fact"),
            ("fact", "fact_2", "Second fact"),
        ]

        km_ids = await semantic_store.index_batch(
            items,
            domain="legal/contracts",
        )

        for km_id in km_ids:
            entry = await semantic_store.get_entry(km_id)
            assert entry.domain == "legal/contracts"

    @pytest.mark.asyncio
    async def test_index_batch_deduplication(self, semantic_store):
        """Test that batch indexing handles duplicates."""
        items = [
            ("fact", "fact_1", "Same content"),
            ("fact", "fact_2", "Same content"),  # Duplicate
        ]

        km_ids = await semantic_store.index_batch(items)

        # Both should return the same ID due to deduplication
        assert km_ids[0] == km_ids[1]


# ============================================================================
# Persistence and Loading Tests
# ============================================================================


class TestPersistence:
    """Tests for data persistence across store instances."""

    @pytest.mark.asyncio
    async def test_data_persists_across_instances(self, temp_db_path, mock_provider):
        """Test that data persists when store is reopened."""
        # First instance - write data
        store1 = SemanticStore(
            db_path=temp_db_path,
            embedding_provider=mock_provider,
        )
        km_id = await store1.index_item(
            source_type=KnowledgeSource.FACT,
            source_id="persistent_fact",
            content="This should persist",
        )

        # Second instance - read data
        store2 = SemanticStore(
            db_path=temp_db_path,
            embedding_provider=mock_provider,
        )
        entry = await store2.get_entry(km_id)

        assert entry is not None
        assert entry.source_id == "persistent_fact"

    @pytest.mark.asyncio
    async def test_stats_persist(self, temp_db_path, mock_provider):
        """Test that statistics are accurate after reopening."""
        # First instance
        store1 = SemanticStore(
            db_path=temp_db_path,
            embedding_provider=mock_provider,
        )
        for i in range(5):
            await store1.index_item(
                source_type=KnowledgeSource.FACT,
                source_id=f"fact_{i}",
                content=f"Unique content {i}",
            )

        # Second instance
        store2 = SemanticStore(
            db_path=temp_db_path,
            embedding_provider=mock_provider,
        )
        stats = await store2.get_stats()

        assert stats["total_entries"] == 5


# ============================================================================
# Concurrent Access Tests
# ============================================================================


class TestConcurrentAccess:
    """Tests for concurrent read/write operations."""

    @pytest.mark.asyncio
    async def test_concurrent_writes_no_corruption(self, semantic_store):
        """Test that concurrent writes don't corrupt data."""

        async def write_item(i: int):
            return await semantic_store.index_item(
                source_type=KnowledgeSource.FACT,
                source_id=f"concurrent_{i}",
                content=f"Unique concurrent content {i}",
            )

        # Run many concurrent writes
        tasks = [write_item(i) for i in range(50)]
        km_ids = await asyncio.gather(*tasks)

        # All should succeed with unique IDs
        assert len(km_ids) == 50
        assert len(set(km_ids)) == 50  # All unique

    @pytest.mark.asyncio
    async def test_concurrent_reads_during_writes(self, semantic_store):
        """Test that reads work correctly during concurrent writes."""
        # Pre-populate some data
        for i in range(10):
            await semantic_store.index_item(
                source_type=KnowledgeSource.FACT,
                source_id=f"preexisting_{i}",
                content=f"Pre-existing content {i}",
            )

        async def write_new():
            for i in range(10):
                await semantic_store.index_item(
                    source_type=KnowledgeSource.FACT,
                    source_id=f"new_{i}",
                    content=f"New content {i}",
                )

        async def read_existing():
            results = []
            for _ in range(5):
                search_results = await semantic_store.search_similar(
                    query="content",
                    limit=5,
                    min_similarity=0.0,
                )
                results.append(search_results)
                await asyncio.sleep(0.01)
            return results

        # Run both concurrently
        write_task = asyncio.create_task(write_new())
        read_task = asyncio.create_task(read_existing())

        await write_task
        read_results = await read_task

        # All reads should have returned valid results
        assert len(read_results) == 5
        assert all(len(r) > 0 for r in read_results)

    @pytest.mark.asyncio
    async def test_concurrent_searches(self, semantic_store):
        """Test multiple concurrent searches."""
        # Populate data
        for i in range(20):
            await semantic_store.index_item(
                source_type=KnowledgeSource.FACT,
                source_id=f"fact_{i}",
                content=f"Content about topic {i % 5}",
            )

        async def search(query: str):
            return await semantic_store.search_similar(
                query=query,
                limit=10,
                min_similarity=0.0,
            )

        queries = ["topic 0", "topic 1", "topic 2", "content", "fact"]
        tasks = [search(q) for q in queries * 4]  # 20 concurrent searches
        results = await asyncio.gather(*tasks)

        assert len(results) == 20
        assert all(isinstance(r, list) for r in results)


# ============================================================================
# Retrieval Metrics Tests
# ============================================================================


class TestRetrievalMetrics:
    """Tests for retrieval tracking and meta-learning."""

    @pytest.mark.asyncio
    async def test_record_retrieval_updates_count(self, semantic_store):
        """Test that record_retrieval increments count."""
        km_id = await semantic_store.index_item(
            source_type=KnowledgeSource.FACT,
            source_id="tracked_fact",
            content="Fact to track retrievals",
        )

        await semantic_store.record_retrieval(km_id, rank_position=0)
        await semantic_store.record_retrieval(km_id, rank_position=1)
        await semantic_store.record_retrieval(km_id, rank_position=2)

        entry = await semantic_store.get_entry(km_id)
        assert entry.retrieval_count == 3

    @pytest.mark.asyncio
    async def test_record_retrieval_updates_avg_rank(self, semantic_store):
        """Test that average rank is calculated correctly."""
        km_id = await semantic_store.index_item(
            source_type=KnowledgeSource.FACT,
            source_id="ranked_fact",
            content="Fact with rank tracking",
        )

        await semantic_store.record_retrieval(km_id, rank_position=0)
        await semantic_store.record_retrieval(km_id, rank_position=2)
        await semantic_store.record_retrieval(km_id, rank_position=4)

        entry = await semantic_store.get_entry(km_id)
        assert abs(entry.avg_retrieval_rank - 2.0) < 0.01  # (0+2+4)/3 = 2

    @pytest.mark.asyncio
    async def test_retrieval_patterns_analysis(self, semantic_store):
        """Test retrieval pattern analysis."""
        # Index items in different domains
        for domain in ["legal", "hr", "finance"]:
            for i in range(5):
                km_id = await semantic_store.index_item(
                    source_type=KnowledgeSource.FACT,
                    source_id=f"{domain}_fact_{i}",
                    content=f"{domain} content {i}",
                    domain=domain,
                )
                # Simulate retrievals for legal items
                if domain == "legal":
                    for _ in range(10):
                        await semantic_store.record_retrieval(km_id, rank_position=0)

        patterns = await semantic_store.get_retrieval_patterns(min_retrievals=5)

        assert "high_retrieval_domains" in patterns
        assert "low_retrieval_domains" in patterns


# ============================================================================
# Statistics Tests
# ============================================================================


class TestStatistics:
    """Tests for store statistics."""

    @pytest.mark.asyncio
    async def test_get_stats_empty_store(self, semantic_store):
        """Test stats on empty store."""
        stats = await semantic_store.get_stats()

        assert stats["total_entries"] == 0
        assert stats["by_source_type"] == {}

    @pytest.mark.asyncio
    async def test_get_stats_with_data(self, semantic_store):
        """Test stats with populated store."""
        # Add various items
        for i in range(3):
            await semantic_store.index_item(
                source_type=KnowledgeSource.FACT,
                source_id=f"fact_{i}",
                content=f"Fact content {i}",
            )
        for i in range(2):
            await semantic_store.index_item(
                source_type=KnowledgeSource.CONSENSUS,
                source_id=f"consensus_{i}",
                content=f"Consensus content {i}",
            )

        stats = await semantic_store.get_stats()

        assert stats["total_entries"] == 5
        assert stats["by_source_type"]["fact"] == 3
        assert stats["by_source_type"]["consensus"] == 2

    @pytest.mark.asyncio
    async def test_has_source_true(self, semantic_store):
        """Test has_source returns True for existing source."""
        await semantic_store.index_item(
            source_type=KnowledgeSource.FACT,
            source_id="existing_fact",
            content="Content",
        )

        assert semantic_store.has_source("fact", "existing_fact") is True

    @pytest.mark.asyncio
    async def test_has_source_false(self, semantic_store):
        """Test has_source returns False for non-existing source."""
        assert semantic_store.has_source("fact", "nonexistent") is False


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Tests for error handling edge cases."""

    @pytest.mark.asyncio
    async def test_index_empty_content(self, semantic_store):
        """Test indexing empty content still works."""
        km_id = await semantic_store.index_item(
            source_type=KnowledgeSource.FACT,
            source_id="empty_fact",
            content="",
        )

        assert km_id.startswith("km_")
        entry = await semantic_store.get_entry(km_id)
        assert entry is not None

    @pytest.mark.asyncio
    async def test_get_nonexistent_entry(self, semantic_store):
        """Test getting non-existent entry returns None."""
        entry = await semantic_store.get_entry("km_nonexistent123")
        assert entry is None

    @pytest.mark.asyncio
    async def test_search_empty_store(self, semantic_store):
        """Test searching empty store returns empty list."""
        results = await semantic_store.search_similar(
            query="anything",
            min_similarity=0.0,
        )
        assert results == []

    @pytest.mark.asyncio
    async def test_index_with_special_characters(self, semantic_store):
        """Test indexing content with special characters."""
        content = "Content with 'quotes', \"double quotes\", and unicode: "
        km_id = await semantic_store.index_item(
            source_type=KnowledgeSource.FACT,
            source_id="special_chars",
            content=content,
        )

        entry = await semantic_store.get_entry(km_id)
        assert entry is not None

    @pytest.mark.asyncio
    async def test_index_very_long_content(self, semantic_store):
        """Test indexing very long content."""
        content = "Long content. " * 10000
        km_id = await semantic_store.index_item(
            source_type=KnowledgeSource.FACT,
            source_id="long_content",
            content=content,
        )

        entry = await semantic_store.get_entry(km_id)
        assert entry is not None

    @pytest.mark.asyncio
    async def test_record_retrieval_nonexistent(self, semantic_store):
        """Test recording retrieval for non-existent item (should not raise)."""
        # Should not raise, just silently do nothing
        await semantic_store.record_retrieval("km_nonexistent", rank_position=0)


# ============================================================================
# DataClass Tests
# ============================================================================


class TestDataClasses:
    """Tests for SemanticIndexEntry and SemanticSearchResult dataclasses."""

    def test_semantic_index_entry_creation(self):
        """Test creating SemanticIndexEntry."""
        entry = SemanticIndexEntry(
            id="km_test123",
            source_type="fact",
            source_id="fact_1",
            content_hash="abc123",
            embedding=[0.1, 0.2, 0.3],
            embedding_model="MockProvider",
            tenant_id="default",
            domain="general",
            importance=0.5,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        assert entry.id == "km_test123"
        assert entry.retrieval_count == 0
        assert entry.metadata == {}

    def test_semantic_search_result_creation(self):
        """Test creating SemanticSearchResult."""
        result = SemanticSearchResult(
            id="km_test123",
            source_type="fact",
            source_id="fact_1",
            content_hash="abc123",
            similarity=0.95,
            domain="legal",
            importance=0.8,
            tenant_id="default",
        )

        assert result.similarity == 0.95
        assert result.metadata == {}


# ============================================================================
# Property Tests
# ============================================================================


class TestStoreProperties:
    """Tests for store properties."""

    def test_embedding_provider_property(self, temp_db_path, mock_provider):
        """Test embedding_provider property."""
        store = SemanticStore(
            db_path=temp_db_path,
            embedding_provider=mock_provider,
        )
        assert store.embedding_provider is mock_provider

    def test_embedding_dimension_property(self, temp_db_path, mock_provider):
        """Test embedding_dimension property."""
        store = SemanticStore(
            db_path=temp_db_path,
            embedding_provider=mock_provider,
        )
        assert store.embedding_dimension == mock_provider.dimension


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
