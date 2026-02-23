"""
Tests for vector indexing in memory backends.

Tests the VectorIndex class and FAISS-accelerated similarity search
in InMemoryBackend.
"""

import asyncio
import pytest
import numpy as np
from datetime import datetime

from aragora.memory.backends.vector_index import (
    VectorIndex,
    VectorIndexConfig,
    SearchResult,
    HAS_NUMPY,
    HAS_FAISS,
)
from aragora.memory.backends.in_memory import InMemoryBackend
from aragora.memory.protocols import MemoryEntry


class TestVectorIndexBasics:
    """Basic tests for VectorIndex functionality."""

    def test_initialization(self):
        """Test index can be created with dimension."""
        index = VectorIndex(dimension=128)
        assert index.dimension == 128
        assert index.size == 0
        assert not index.is_using_faiss

    def test_add_single_entry(self):
        """Test adding a single embedding."""
        index = VectorIndex(dimension=3)
        index.add("entry1", [1.0, 0.0, 0.0])
        assert index.size == 1

    def test_add_multiple_entries(self):
        """Test adding multiple embeddings."""
        index = VectorIndex(dimension=3)
        index.add("entry1", [1.0, 0.0, 0.0])
        index.add("entry2", [0.0, 1.0, 0.0])
        index.add("entry3", [0.0, 0.0, 1.0])
        assert index.size == 3

    def test_add_updates_existing(self):
        """Test adding entry with same ID updates embedding."""
        index = VectorIndex(dimension=3)
        index.add("entry1", [1.0, 0.0, 0.0])
        index.add("entry1", [0.0, 1.0, 0.0])
        assert index.size == 1

    def test_remove_entry(self):
        """Test removing an entry."""
        index = VectorIndex(dimension=3)
        index.add("entry1", [1.0, 0.0, 0.0])
        assert index.remove("entry1")
        assert index.size == 0

    def test_remove_nonexistent(self):
        """Test removing nonexistent entry returns False."""
        index = VectorIndex(dimension=3)
        assert not index.remove("nonexistent")

    def test_clear(self):
        """Test clearing all entries."""
        index = VectorIndex(dimension=3)
        index.add("entry1", [1.0, 0.0, 0.0])
        index.add("entry2", [0.0, 1.0, 0.0])
        index.clear()
        assert index.size == 0

    def test_dimension_mismatch_raises(self):
        """Test adding wrong dimension raises ValueError."""
        index = VectorIndex(dimension=3)
        with pytest.raises(ValueError, match="dimension"):
            index.add("entry1", [1.0, 0.0])  # Wrong dimension


class TestVectorIndexSearch:
    """Tests for VectorIndex search functionality."""

    def test_search_empty_index(self):
        """Test searching empty index returns empty list."""
        index = VectorIndex(dimension=3)
        results = index.search([1.0, 0.0, 0.0], k=5)
        assert results == []

    def test_search_finds_exact_match(self):
        """Test search finds exact matching embedding."""
        index = VectorIndex(dimension=3)
        index.add("entry1", [1.0, 0.0, 0.0])
        index.add("entry2", [0.0, 1.0, 0.0])

        results = index.search([1.0, 0.0, 0.0], k=2)
        assert len(results) >= 1
        assert results[0].entry_id == "entry1"
        assert results[0].similarity > 0.99

    def test_search_returns_sorted_by_similarity(self):
        """Test results are sorted by similarity descending."""
        index = VectorIndex(dimension=3)
        index.add("exact", [1.0, 0.0, 0.0])
        index.add("similar", [0.9, 0.1, 0.0])
        index.add("different", [0.0, 1.0, 0.0])

        results = index.search([1.0, 0.0, 0.0], k=3)
        assert len(results) == 3

        # Verify sorted order
        for i in range(len(results) - 1):
            assert results[i].similarity >= results[i + 1].similarity

        # Exact match should be first
        assert results[0].entry_id == "exact"

    def test_search_respects_min_similarity(self):
        """Test min_similarity filters results."""
        index = VectorIndex(dimension=3)
        index.add("similar", [0.9, 0.1, 0.0])
        index.add("different", [0.0, 1.0, 0.0])

        results = index.search([1.0, 0.0, 0.0], k=10, min_similarity=0.8)
        # Only "similar" should match (cosine sim > 0.8)
        assert len(results) >= 1
        for r in results:
            assert r.similarity >= 0.8

    def test_search_respects_k_limit(self):
        """Test k parameter limits results."""
        index = VectorIndex(dimension=3)
        for i in range(20):
            # Random embeddings
            np.random.seed(i)
            emb = np.random.randn(3).tolist()
            index.add(f"entry{i}", emb)

        results = index.search([1.0, 0.0, 0.0], k=5)
        assert len(results) <= 5

    def test_search_with_identical_embeddings(self):
        """Test search with multiple identical embeddings."""
        index = VectorIndex(dimension=3)
        embedding = [1.0, 0.0, 0.0]
        index.add("a", embedding)
        index.add("b", embedding)
        index.add("c", embedding)

        results = index.search(embedding, k=5)
        assert len(results) == 3
        for r in results:
            assert r.similarity > 0.99

    def test_search_dimension_mismatch_raises(self):
        """Test searching with wrong dimension raises."""
        index = VectorIndex(dimension=3)
        index.add("entry1", [1.0, 0.0, 0.0])

        with pytest.raises(ValueError, match="dimension"):
            index.search([1.0, 0.0], k=5)


class TestVectorIndexConfig:
    """Tests for VectorIndex configuration."""

    def test_custom_faiss_threshold(self):
        """Test custom FAISS threshold configuration."""
        config = VectorIndexConfig(faiss_threshold=50)
        index = VectorIndex(dimension=3, config=config)
        assert index.config.faiss_threshold == 50

    def test_below_threshold_uses_brute_force(self):
        """Test small index uses brute force not FAISS."""
        config = VectorIndexConfig(faiss_threshold=100)
        index = VectorIndex(dimension=3, config=config)

        # Add 50 entries (below threshold)
        for i in range(50):
            np.random.seed(i)
            index.add(f"entry{i}", np.random.randn(3).tolist())

        # Force index build
        index.rebuild()
        assert not index.is_using_faiss

    def test_above_threshold_uses_faiss(self):
        """Test large index uses FAISS when available (mocked if not installed)."""
        from unittest.mock import MagicMock, patch as _patch

        config = VectorIndexConfig(faiss_threshold=50)
        index = VectorIndex(dimension=3, config=config)

        # Add 100 entries (above threshold)
        for i in range(100):
            np.random.seed(i)
            index.add(f"entry{i}", np.random.randn(3).tolist())

        if HAS_FAISS:
            # Real FAISS available - just rebuild and assert
            index.rebuild()
            assert index.is_using_faiss
        else:
            # Mock faiss so _create_faiss_index succeeds
            mock_faiss = MagicMock()
            mock_faiss_index = MagicMock()
            mock_faiss.IndexFlatIP.return_value = mock_faiss_index
            mock_faiss.get_num_gpus.return_value = 0

            vi_mod = "aragora.memory.backends.vector_index"
            with _patch(f"{vi_mod}.HAS_FAISS", True), _patch(f"{vi_mod}.faiss", mock_faiss):
                index.rebuild()
                assert index._faiss_index is not None
                # is_using_faiss checks the module-level HAS_FAISS which is
                # patched to True inside this context
                assert index.is_using_faiss


class TestVectorIndexStats:
    """Tests for VectorIndex statistics."""

    def test_get_stats_empty(self):
        """Test stats for empty index."""
        index = VectorIndex(dimension=128)
        stats = index.get_stats()

        assert stats["size"] == 0
        assert stats["dimension"] == 128
        assert "faiss_available" in stats
        assert "using_faiss" in stats
        assert stats["index_dirty"] is True

    def test_get_stats_with_entries(self):
        """Test stats with entries."""
        index = VectorIndex(dimension=3)
        index.add("entry1", [1.0, 0.0, 0.0])
        index.add("entry2", [0.0, 1.0, 0.0])

        # Trigger index build
        index.search([1.0, 0.0, 0.0], k=1)

        stats = index.get_stats()
        assert stats["size"] == 2
        assert stats["index_dirty"] is False


class TestVectorIndexAsync:
    """Tests for async search functionality."""

    @pytest.mark.asyncio
    async def test_search_async(self):
        """Test async search method."""
        index = VectorIndex(dimension=3)
        index.add("entry1", [1.0, 0.0, 0.0])
        index.add("entry2", [0.0, 1.0, 0.0])

        results = await index.search_async([1.0, 0.0, 0.0], k=2)
        assert len(results) >= 1
        assert results[0].entry_id == "entry1"

    @pytest.mark.asyncio
    async def test_search_async_concurrent(self):
        """Test concurrent async searches."""
        index = VectorIndex(dimension=3)
        for i in range(10):
            np.random.seed(i)
            index.add(f"entry{i}", np.random.randn(3).tolist())

        # Run concurrent searches
        queries = [np.random.randn(3).tolist() for _ in range(5)]
        tasks = [index.search_async(q, k=3) for q in queries]
        all_results = await asyncio.gather(*tasks)

        assert len(all_results) == 5
        for results in all_results:
            assert len(results) <= 3


class TestInMemoryBackendVectorIndex:
    """Tests for InMemoryBackend with vector index integration."""

    @pytest.mark.asyncio
    async def test_search_similar_creates_index(self):
        """Test similarity search creates vector index lazily."""
        backend = InMemoryBackend()

        entry1 = MemoryEntry(
            id="1",
            content="Test 1",
            embedding=[1.0, 0.0, 0.0],
        )
        entry2 = MemoryEntry(
            id="2",
            content="Test 2",
            embedding=[0.0, 1.0, 0.0],
        )

        await backend.store(entry1)
        await backend.store(entry2)

        results = await backend.search_similar([1.0, 0.0, 0.0], limit=2)
        assert len(results) >= 1
        assert results[0][0].id == "1"

    @pytest.mark.asyncio
    async def test_search_similar_with_tier_filter(self):
        """Test similarity search with tier filter."""
        backend = InMemoryBackend()

        await backend.store(
            MemoryEntry(id="1", content="Fast", tier="fast", embedding=[1.0, 0.0, 0.0])
        )
        await backend.store(
            MemoryEntry(id="2", content="Slow", tier="slow", embedding=[0.9, 0.1, 0.0])
        )

        # Search only in fast tier
        results = await backend.search_similar(
            [1.0, 0.0, 0.0],
            limit=5,
            tier="fast",
        )
        assert len(results) == 1
        assert results[0][0].id == "1"

    @pytest.mark.asyncio
    async def test_search_similar_updates_on_store(self):
        """Test vector index updates when storing entries."""
        backend = InMemoryBackend()

        await backend.store(MemoryEntry(id="1", content="First", embedding=[1.0, 0.0, 0.0]))

        # Trigger index creation
        await backend.search_similar([1.0, 0.0, 0.0], limit=1)

        # Add new entry
        await backend.store(MemoryEntry(id="2", content="Second", embedding=[0.9, 0.1, 0.0]))

        # Should find both
        results = await backend.search_similar([1.0, 0.0, 0.0], limit=5)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_search_similar_updates_on_delete(self):
        """Test vector index updates when deleting entries."""
        backend = InMemoryBackend()

        await backend.store(MemoryEntry(id="1", content="First", embedding=[1.0, 0.0, 0.0]))
        await backend.store(
            MemoryEntry(id="2", content="Second", embedding=[0.9, 0.1, 0.0])  # Similar to entry 1
        )

        # Trigger index creation
        results = await backend.search_similar([1.0, 0.0, 0.0], limit=2, min_similarity=0.5)
        assert len(results) == 2

        # Delete entry
        await backend.delete("1")

        # Should only find remaining entry
        results = await backend.search_similar([1.0, 0.0, 0.0], limit=5, min_similarity=0.5)
        assert len(results) == 1
        assert results[0][0].id == "2"

    @pytest.mark.asyncio
    async def test_search_similar_updates_on_update(self):
        """Test vector index updates when updating entries."""
        backend = InMemoryBackend()

        await backend.store(MemoryEntry(id="1", content="First", embedding=[1.0, 0.0, 0.0]))

        # Trigger index creation - use min_similarity=0 to get result even for low similarity
        results = await backend.search_similar([0.0, 1.0, 0.0], limit=1, min_similarity=0.0)
        assert len(results) == 1
        initial_sim = results[0][1]  # Should be ~0 (orthogonal vectors)

        # Update embedding to match the query
        await backend.update(MemoryEntry(id="1", content="Updated", embedding=[0.0, 1.0, 0.0]))

        # Should now match much better
        results = await backend.search_similar([0.0, 1.0, 0.0], limit=1, min_similarity=0.0)
        assert len(results) == 1
        assert results[0][1] > initial_sim  # Should now be ~1.0

    @pytest.mark.asyncio
    async def test_get_stats_includes_vector_index(self):
        """Test stats include vector index information."""
        backend = InMemoryBackend()

        await backend.store(MemoryEntry(id="1", content="Test", embedding=[1.0, 0.0, 0.0]))
        await backend.search_similar([1.0, 0.0, 0.0], limit=1)

        stats = await backend.get_stats()
        assert "vector_index" in stats
        assert stats["vector_index"]["size"] == 1

    @pytest.mark.asyncio
    async def test_clear_clears_vector_index(self):
        """Test clear also clears vector index."""
        backend = InMemoryBackend()

        await backend.store(MemoryEntry(id="1", content="Test", embedding=[1.0, 0.0, 0.0]))
        await backend.search_similar([1.0, 0.0, 0.0], limit=1)

        await backend.clear()

        results = await backend.search_similar([1.0, 0.0, 0.0], limit=5)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_batch_store_updates_index(self):
        """Test batch store updates vector index."""
        backend = InMemoryBackend()

        # Start from 1 to avoid zero vector
        entries = [
            MemoryEntry(id=f"{i}", content=f"Entry {i}", embedding=[float(i), 0.1, 0.0])
            for i in range(1, 6)
        ]

        await backend.store_batch(entries)
        results = await backend.search_similar([5.0, 0.1, 0.0], limit=5, min_similarity=0.0)
        assert len(results) == 5
        assert results[0][0].id == "5"  # Most similar to [5.0, 0.1, 0.0]

    @pytest.mark.asyncio
    async def test_batch_delete_updates_index(self):
        """Test batch delete updates vector index."""
        backend = InMemoryBackend()

        entries = [
            MemoryEntry(id=f"{i}", content=f"Entry {i}", embedding=[float(i), 0.0, 0.0])
            for i in range(5)
        ]

        await backend.store_batch(entries)
        await backend.search_similar([1.0, 0.0, 0.0], limit=1)  # Trigger index

        await backend.delete_batch(["0", "1", "2"])
        results = await backend.search_similar([1.0, 0.0, 0.0], limit=10)
        assert len(results) == 2


class TestVectorIndexPerformance:
    """Performance and edge case tests."""

    def test_large_dimension(self):
        """Test with high-dimensional embeddings."""
        dimension = 1536  # OpenAI embedding dimension
        index = VectorIndex(dimension=dimension)

        np.random.seed(42)
        for i in range(10):
            index.add(f"entry{i}", np.random.randn(dimension).tolist())

        query = np.random.randn(dimension).tolist()
        results = index.search(query, k=5)
        assert len(results) == 5

    def test_normalized_vs_unnormalized(self):
        """Test that unnormalized embeddings work correctly."""
        index = VectorIndex(dimension=3)

        # Add unnormalized embeddings with different magnitudes
        index.add("small", [0.1, 0.0, 0.0])
        index.add("large", [100.0, 0.0, 0.0])

        # Both should have same cosine similarity to query
        results = index.search([1.0, 0.0, 0.0], k=2)
        assert len(results) == 2
        # Both should be ~1.0 similarity
        assert results[0].similarity > 0.99
        assert results[1].similarity > 0.99

    def test_zero_vector_handling(self):
        """Test handling of zero vectors."""
        index = VectorIndex(dimension=3)
        index.add("zero", [0.0, 0.0, 0.0])
        index.add("nonzero", [1.0, 0.0, 0.0])

        # Searching should not crash
        results = index.search([1.0, 0.0, 0.0], k=5)
        # Zero vector may or may not appear depending on handling
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_entries_without_embeddings(self):
        """Test entries without embeddings are skipped."""
        backend = InMemoryBackend()

        await backend.store(
            MemoryEntry(id="1", content="With embedding", embedding=[1.0, 0.0, 0.0])
        )
        await backend.store(MemoryEntry(id="2", content="No embedding", embedding=None))

        results = await backend.search_similar([1.0, 0.0, 0.0], limit=5)
        assert len(results) == 1
        assert results[0][0].id == "1"


class TestVectorIndexFallback:
    """Tests for graceful fallback when FAISS unavailable."""

    def test_fallback_search_works(self):
        """Test search works without FAISS (numpy fallback)."""
        # This test always uses numpy fallback by keeping below threshold
        config = VectorIndexConfig(faiss_threshold=1000)  # High threshold
        index = VectorIndex(dimension=3, config=config)

        index.add("entry1", [1.0, 0.0, 0.0])
        index.add("entry2", [0.0, 1.0, 0.0])
        index.add("entry3", [0.5, 0.5, 0.0])

        results = index.search([1.0, 0.0, 0.0], k=3)
        assert len(results) == 3
        assert results[0].entry_id == "entry1"

    def test_is_faiss_available_property(self):
        """Test is_faiss_available reflects reality."""
        index = VectorIndex(dimension=3)
        # This just tests the property exists and returns bool
        assert isinstance(index.is_faiss_available, bool)

    def test_stats_reflect_backend(self):
        """Test stats show which backend is being used."""
        config = VectorIndexConfig(faiss_threshold=1000)
        index = VectorIndex(dimension=3, config=config)

        for i in range(10):
            index.add(f"entry{i}", [float(i), 0.0, 0.0])

        index.rebuild()
        stats = index.get_stats()

        assert stats["faiss_available"] == HAS_FAISS
        # Below threshold, so should not use FAISS
        assert stats["using_faiss"] is False
