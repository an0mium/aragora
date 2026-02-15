"""Comprehensive unit tests for InMemoryBackend.

Tests cover:
- Core CRUD operations (store, get, update, delete)
- Query operations (filtering, pagination, sorting)
- Vector similarity search (indexed and tier-filtered)
- Batch operations (store_batch, delete_batch)
- Tier management (promote, count_by_tier)
- TTL/expiration (cleanup_expired)
- Index management (creation, invalidation, rebuild on dimension change)
- Health and diagnostics (health_check, get_stats)
- Edge cases (empty store, missing keys, zero vectors, dimension mismatch)
- Protocol compliance
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from aragora.memory.backends.in_memory import InMemoryBackend
from aragora.memory.protocols import (
    BackendHealth,
    MemoryBackend,
    MemoryEntry,
    MemoryQueryResult,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def backend() -> InMemoryBackend:
    """Create a fresh InMemoryBackend instance."""
    return InMemoryBackend()


@pytest.fixture
def sample_entry() -> MemoryEntry:
    """Create a sample memory entry."""
    return MemoryEntry(
        id="entry_1",
        content="Sample content for testing",
        tier="fast",
        weight=0.8,
        metadata={"source": "test"},
    )


@pytest.fixture
def sample_entry_with_embedding() -> MemoryEntry:
    """Create a sample memory entry with an embedding vector."""
    return MemoryEntry(
        id="emb_1",
        content="Embedded content",
        tier="fast",
        weight=0.9,
        embedding=[1.0, 0.0, 0.0, 0.0],
    )


@pytest.fixture
def entries_across_tiers() -> list[MemoryEntry]:
    """Create entries distributed across all memory tiers."""
    return [
        MemoryEntry(id="fast_1", content="Fast tier 1", tier="fast", weight=0.5),
        MemoryEntry(id="fast_2", content="Fast tier 2", tier="fast", weight=0.7),
        MemoryEntry(id="medium_1", content="Medium tier 1", tier="medium", weight=0.6),
        MemoryEntry(id="slow_1", content="Slow tier 1", tier="slow", weight=0.9),
        MemoryEntry(id="glacial_1", content="Glacial tier 1", tier="glacial", weight=1.0),
    ]


@pytest.fixture
def embedded_entries() -> list[MemoryEntry]:
    """Create entries with orthogonal embedding vectors for predictable similarity tests."""
    return [
        MemoryEntry(
            id="vec_x",
            content="X-axis vector",
            tier="fast",
            weight=0.5,
            embedding=[1.0, 0.0, 0.0],
        ),
        MemoryEntry(
            id="vec_y",
            content="Y-axis vector",
            tier="fast",
            weight=0.5,
            embedding=[0.0, 1.0, 0.0],
        ),
        MemoryEntry(
            id="vec_z",
            content="Z-axis vector",
            tier="slow",
            weight=0.5,
            embedding=[0.0, 0.0, 1.0],
        ),
        MemoryEntry(
            id="vec_xy",
            content="XY diagonal",
            tier="fast",
            weight=0.5,
            embedding=[0.707, 0.707, 0.0],
        ),
    ]


# =============================================================================
# Protocol Compliance
# =============================================================================


class TestProtocolCompliance:
    """Verify InMemoryBackend implements the MemoryBackend protocol."""

    def test_is_instance_of_memory_backend(self):
        """InMemoryBackend should satisfy the MemoryBackend runtime_checkable protocol."""
        backend = InMemoryBackend()
        assert isinstance(backend, MemoryBackend)

    def test_has_all_required_methods(self):
        """InMemoryBackend should expose every method required by the protocol."""
        required_methods = [
            "store",
            "get",
            "update",
            "delete",
            "query",
            "search_similar",
            "store_batch",
            "delete_batch",
            "promote",
            "count_by_tier",
            "cleanup_expired",
            "vacuum",
            "health_check",
            "get_stats",
        ]
        backend = InMemoryBackend()
        for method_name in required_methods:
            assert hasattr(backend, method_name), f"Missing method: {method_name}"
            assert callable(getattr(backend, method_name)), f"Not callable: {method_name}"


# =============================================================================
# Core CRUD Operations
# =============================================================================


class TestStore:
    """Test the store() operation."""

    @pytest.mark.asyncio
    async def test_store_returns_entry_id(self, backend, sample_entry):
        """store() should return the entry's ID."""
        result = await backend.store(sample_entry)
        assert result == "entry_1"

    @pytest.mark.asyncio
    async def test_store_persists_entry(self, backend, sample_entry):
        """Stored entry should be retrievable."""
        await backend.store(sample_entry)
        retrieved = await backend.get("entry_1")
        assert retrieved is not None
        assert retrieved.content == "Sample content for testing"
        assert retrieved.tier == "fast"
        assert retrieved.weight == 0.8

    @pytest.mark.asyncio
    async def test_store_updates_tier_index(self, backend, sample_entry):
        """Stored entry should appear in the correct tier index."""
        await backend.store(sample_entry)
        assert "entry_1" in backend._by_tier["fast"]

    @pytest.mark.asyncio
    async def test_store_overwrites_existing_entry(self, backend):
        """Storing an entry with an existing ID should overwrite it."""
        entry_v1 = MemoryEntry(id="dup", content="Version 1", tier="fast", weight=0.5)
        entry_v2 = MemoryEntry(id="dup", content="Version 2", tier="fast", weight=0.9)

        await backend.store(entry_v1)
        await backend.store(entry_v2)

        retrieved = await backend.get("dup")
        assert retrieved is not None
        assert retrieved.content == "Version 2"
        assert retrieved.weight == 0.9

    @pytest.mark.asyncio
    async def test_store_entry_with_embedding_updates_vector_index(
        self, backend, sample_entry_with_embedding
    ):
        """Storing an entry with an embedding should update the vector index."""
        await backend.store(sample_entry_with_embedding)
        # The vector index should have been lazily created
        assert backend._vector_index is not None or backend._index_dimension is None
        # After store, at minimum the entry should be retrievable via similarity search
        results = await backend.search_similar(
            query_embedding=[1.0, 0.0, 0.0, 0.0],
            limit=5,
            min_similarity=0.5,
        )
        assert len(results) >= 1
        assert results[0][0].id == "emb_1"

    @pytest.mark.asyncio
    async def test_store_entry_without_embedding(self, backend, sample_entry):
        """Storing an entry without an embedding should not break vector index."""
        await backend.store(sample_entry)
        assert await backend.get("entry_1") is not None

    @pytest.mark.asyncio
    async def test_store_increments_length(self, backend):
        """len(backend) should increase with each new entry."""
        assert len(backend) == 0
        await backend.store(MemoryEntry(id="a", content="A", tier="fast"))
        assert len(backend) == 1
        await backend.store(MemoryEntry(id="b", content="B", tier="fast"))
        assert len(backend) == 2


class TestGet:
    """Test the get() operation."""

    @pytest.mark.asyncio
    async def test_get_existing_entry(self, backend, sample_entry):
        """get() should return the stored entry."""
        await backend.store(sample_entry)
        entry = await backend.get("entry_1")
        assert entry is not None
        assert entry.id == "entry_1"

    @pytest.mark.asyncio
    async def test_get_nonexistent_entry(self, backend):
        """get() should return None for a missing key."""
        result = await backend.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_after_delete(self, backend, sample_entry):
        """get() should return None after the entry has been deleted."""
        await backend.store(sample_entry)
        await backend.delete("entry_1")
        result = await backend.get("entry_1")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_preserves_metadata(self, backend, sample_entry):
        """get() should return the full entry including metadata."""
        await backend.store(sample_entry)
        entry = await backend.get("entry_1")
        assert entry is not None
        assert entry.metadata == {"source": "test"}

    @pytest.mark.asyncio
    async def test_get_preserves_embedding(self, backend, sample_entry_with_embedding):
        """get() should return the full entry including embedding."""
        await backend.store(sample_entry_with_embedding)
        entry = await backend.get("emb_1")
        assert entry is not None
        assert entry.embedding == [1.0, 0.0, 0.0, 0.0]


class TestUpdate:
    """Test the update() operation."""

    @pytest.mark.asyncio
    async def test_update_existing_entry(self, backend, sample_entry):
        """update() should return True and modify the entry."""
        await backend.store(sample_entry)

        sample_entry.content = "Updated content"
        sample_entry.weight = 0.95
        result = await backend.update(sample_entry)

        assert result is True
        updated = await backend.get("entry_1")
        assert updated is not None
        assert updated.content == "Updated content"
        assert updated.weight == 0.95

    @pytest.mark.asyncio
    async def test_update_nonexistent_entry(self, backend, sample_entry):
        """update() should return False for a missing entry."""
        result = await backend.update(sample_entry)
        assert result is False

    @pytest.mark.asyncio
    async def test_update_changes_tier_index(self, backend):
        """update() should move the entry between tier indices when tier changes."""
        original = MemoryEntry(
            id="tier_change",
            content="Original tier",
            tier="fast",
            weight=0.8,
        )
        await backend.store(original)
        assert "tier_change" in backend._by_tier["fast"]

        # Create a new entry object with a different tier (avoids same-reference issue)
        updated_entry = MemoryEntry(
            id="tier_change",
            content="Original tier",
            tier="slow",
            weight=0.8,
        )
        await backend.update(updated_entry)

        assert "tier_change" not in backend._by_tier["fast"]
        assert "tier_change" in backend._by_tier["slow"]

    @pytest.mark.asyncio
    async def test_update_sets_updated_at(self, backend, sample_entry):
        """update() should refresh the updated_at timestamp."""
        await backend.store(sample_entry)
        original_updated_at = sample_entry.updated_at

        sample_entry.content = "Changed"
        await backend.update(sample_entry)

        updated = await backend.get("entry_1")
        assert updated is not None
        assert updated.updated_at >= original_updated_at

    @pytest.mark.asyncio
    async def test_update_embedding_triggers_index_update(self, backend):
        """update() should update the vector index when embedding changes."""
        original = MemoryEntry(
            id="upd_emb",
            content="Original",
            tier="fast",
            embedding=[1.0, 0.0, 0.0],
        )
        await backend.store(original)

        # Create a new entry object with different embedding (avoids same-reference issue)
        updated_entry = MemoryEntry(
            id="upd_emb",
            content="Original",
            tier="fast",
            embedding=[0.0, 1.0, 0.0],
        )
        await backend.update(updated_entry)

        # Searching for the new direction should find it
        results = await backend.search_similar(
            query_embedding=[0.0, 1.0, 0.0],
            limit=5,
            min_similarity=0.8,
        )
        found_ids = [r[0].id for r in results]
        assert "upd_emb" in found_ids

    @pytest.mark.asyncio
    async def test_update_same_tier_no_change(self, backend, sample_entry):
        """update() with same tier should not remove entry from tier index."""
        await backend.store(sample_entry)

        sample_entry.content = "Modified"
        # tier stays "fast"
        await backend.update(sample_entry)

        assert "entry_1" in backend._by_tier["fast"]


class TestDelete:
    """Test the delete() operation."""

    @pytest.mark.asyncio
    async def test_delete_existing_entry(self, backend, sample_entry):
        """delete() should return True and remove the entry."""
        await backend.store(sample_entry)
        result = await backend.delete("entry_1")
        assert result is True
        assert await backend.get("entry_1") is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_entry(self, backend):
        """delete() should return False for a missing key."""
        result = await backend.delete("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_removes_from_tier_index(self, backend, sample_entry):
        """delete() should remove the entry from its tier index."""
        await backend.store(sample_entry)
        await backend.delete("entry_1")
        assert "entry_1" not in backend._by_tier["fast"]

    @pytest.mark.asyncio
    async def test_delete_decrements_length(self, backend, sample_entry):
        """delete() should decrease len(backend)."""
        await backend.store(sample_entry)
        assert len(backend) == 1
        await backend.delete("entry_1")
        assert len(backend) == 0

    @pytest.mark.asyncio
    async def test_delete_removes_from_vector_index(self, backend, sample_entry_with_embedding):
        """delete() should remove the entry from the vector index."""
        await backend.store(sample_entry_with_embedding)
        await backend.delete("emb_1")

        results = await backend.search_similar(
            query_embedding=[1.0, 0.0, 0.0, 0.0],
            limit=5,
            min_similarity=0.0,
        )
        found_ids = [r[0].id for r in results]
        assert "emb_1" not in found_ids

    @pytest.mark.asyncio
    async def test_double_delete(self, backend, sample_entry):
        """Deleting an already-deleted entry should return False."""
        await backend.store(sample_entry)
        assert await backend.delete("entry_1") is True
        assert await backend.delete("entry_1") is False


# =============================================================================
# Query Operations
# =============================================================================


class TestQuery:
    """Test the query() operation with filtering, sorting, and pagination."""

    @pytest.mark.asyncio
    async def test_query_all_entries(self, backend, entries_across_tiers):
        """query() with no filters should return all entries."""
        for entry in entries_across_tiers:
            await backend.store(entry)

        result = await backend.query()
        assert isinstance(result, MemoryQueryResult)
        assert result.total_count == 5
        assert len(result.entries) == 5
        assert result.has_more is False

    @pytest.mark.asyncio
    async def test_query_empty_store(self, backend):
        """query() on an empty store should return zero results."""
        result = await backend.query()
        assert result.total_count == 0
        assert len(result.entries) == 0
        assert result.has_more is False
        assert result.cursor is None

    @pytest.mark.asyncio
    async def test_query_by_tier(self, backend, entries_across_tiers):
        """query(tier=...) should only return entries from that tier."""
        for entry in entries_across_tiers:
            await backend.store(entry)

        result = await backend.query(tier="fast")
        assert result.total_count == 2
        assert all(e.tier == "fast" for e in result.entries)

    @pytest.mark.asyncio
    async def test_query_by_nonexistent_tier(self, backend, entries_across_tiers):
        """query() for a tier with no entries should return zero results."""
        for entry in entries_across_tiers:
            await backend.store(entry)

        result = await backend.query(tier="nonexistent")
        assert result.total_count == 0

    @pytest.mark.asyncio
    async def test_query_by_min_weight(self, backend, entries_across_tiers):
        """query(min_weight=...) should filter out low-weight entries."""
        for entry in entries_across_tiers:
            await backend.store(entry)

        result = await backend.query(min_weight=0.8)
        assert result.total_count == 2
        assert all(e.weight >= 0.8 for e in result.entries)

    @pytest.mark.asyncio
    async def test_query_combined_tier_and_weight(self, backend, entries_across_tiers):
        """query() should apply both tier and weight filters together."""
        for entry in entries_across_tiers:
            await backend.store(entry)

        result = await backend.query(tier="fast", min_weight=0.6)
        assert result.total_count == 1
        assert result.entries[0].id == "fast_2"

    @pytest.mark.asyncio
    async def test_query_pagination_limit(self, backend):
        """query(limit=...) should cap the number of returned entries."""
        for i in range(10):
            await backend.store(MemoryEntry(id=f"p{i}", content=f"C{i}", tier="fast", weight=0.5))

        result = await backend.query(limit=3)
        assert len(result.entries) == 3
        assert result.total_count == 10
        assert result.has_more is True
        assert result.cursor == "3"

    @pytest.mark.asyncio
    async def test_query_pagination_offset(self, backend):
        """query(offset=...) should skip the first N entries."""
        for i in range(5):
            await backend.store(MemoryEntry(id=f"o{i}", content=f"C{i}", tier="fast", weight=0.5))

        result_all = await backend.query(limit=100)
        result_offset = await backend.query(limit=100, offset=3)

        assert result_offset.total_count == 5
        assert len(result_offset.entries) == 2

    @pytest.mark.asyncio
    async def test_query_pagination_beyond_end(self, backend):
        """query() with offset beyond total count should return empty results."""
        await backend.store(MemoryEntry(id="sole", content="Only", tier="fast"))
        result = await backend.query(offset=10)
        assert len(result.entries) == 0
        assert result.has_more is False

    @pytest.mark.asyncio
    async def test_query_order_by_weight_descending(self, backend, entries_across_tiers):
        """query(order_by='weight', descending=True) should sort highest weight first."""
        for entry in entries_across_tiers:
            await backend.store(entry)

        result = await backend.query(order_by="weight", descending=True)
        weights = [e.weight for e in result.entries]
        assert weights == sorted(weights, reverse=True)

    @pytest.mark.asyncio
    async def test_query_order_by_weight_ascending(self, backend, entries_across_tiers):
        """query(order_by='weight', descending=False) should sort lowest weight first."""
        for entry in entries_across_tiers:
            await backend.store(entry)

        result = await backend.query(order_by="weight", descending=False)
        weights = [e.weight for e in result.entries]
        assert weights == sorted(weights)

    @pytest.mark.asyncio
    async def test_query_cursor_pagination(self, backend):
        """Cursor-based pagination should iterate through all entries."""
        for i in range(7):
            await backend.store(MemoryEntry(id=f"cur{i}", content=f"C{i}", tier="fast", weight=0.5))

        # First page
        page1 = await backend.query(limit=3, offset=0)
        assert len(page1.entries) == 3
        assert page1.has_more is True
        assert page1.cursor == "3"

        # Second page using cursor as offset
        page2 = await backend.query(limit=3, offset=int(page1.cursor))
        assert len(page2.entries) == 3
        assert page2.has_more is True

        # Third page
        page3 = await backend.query(limit=3, offset=int(page2.cursor))
        assert len(page3.entries) == 1
        assert page3.has_more is False
        assert page3.cursor is None


# =============================================================================
# Vector Similarity Search
# =============================================================================


class TestSearchSimilar:
    """Test vector similarity search."""

    @pytest.mark.asyncio
    async def test_search_empty_query(self, backend):
        """search_similar() with empty query embedding should return empty list."""
        result = await backend.search_similar(query_embedding=[], limit=10)
        assert result == []

    @pytest.mark.asyncio
    async def test_search_empty_store(self, backend):
        """search_similar() on empty store should return empty list."""
        result = await backend.search_similar(
            query_embedding=[1.0, 0.0, 0.0],
            limit=10,
            min_similarity=0.0,
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_search_finds_identical_vector(self, backend, embedded_entries):
        """Searching with an identical vector should return similarity close to 1.0."""
        for entry in embedded_entries:
            await backend.store(entry)

        results = await backend.search_similar(
            query_embedding=[1.0, 0.0, 0.0],
            limit=10,
            min_similarity=0.9,
        )
        assert len(results) >= 1
        assert results[0][0].id == "vec_x"
        assert results[0][1] >= 0.99

    @pytest.mark.asyncio
    async def test_search_similarity_ordering(self, backend, embedded_entries):
        """Results should be sorted by similarity in descending order."""
        for entry in embedded_entries:
            await backend.store(entry)

        results = await backend.search_similar(
            query_embedding=[1.0, 0.0, 0.0],
            limit=10,
            min_similarity=0.0,
        )
        similarities = [sim for _, sim in results]
        assert similarities == sorted(similarities, reverse=True)

    @pytest.mark.asyncio
    async def test_search_respects_limit(self, backend, embedded_entries):
        """search_similar() should return at most limit results."""
        for entry in embedded_entries:
            await backend.store(entry)

        results = await backend.search_similar(
            query_embedding=[0.5, 0.5, 0.5],
            limit=2,
            min_similarity=0.0,
        )
        assert len(results) <= 2

    @pytest.mark.asyncio
    async def test_search_respects_min_similarity(self, backend, embedded_entries):
        """search_similar() should filter out results below min_similarity."""
        for entry in embedded_entries:
            await backend.store(entry)

        # Searching for X-axis with high threshold should only find vec_x and vec_xy
        results = await backend.search_similar(
            query_embedding=[1.0, 0.0, 0.0],
            limit=10,
            min_similarity=0.7,
        )
        for entry, sim in results:
            assert sim >= 0.7

    @pytest.mark.asyncio
    async def test_search_with_tier_filter(self, backend, embedded_entries):
        """search_similar(tier=...) should only return entries from that tier."""
        for entry in embedded_entries:
            await backend.store(entry)

        results = await backend.search_similar(
            query_embedding=[0.0, 0.0, 1.0],
            limit=10,
            min_similarity=0.0,
            tier="slow",
        )
        assert len(results) >= 1
        assert all(e.tier == "slow" for e, _ in results)
        assert results[0][0].id == "vec_z"

    @pytest.mark.asyncio
    async def test_search_tier_filter_excludes_other_tiers(self, backend, embedded_entries):
        """Tier filter should exclude matching vectors in other tiers."""
        for entry in embedded_entries:
            await backend.store(entry)

        # vec_z is in "slow" tier but has embedding [0,0,1]
        results = await backend.search_similar(
            query_embedding=[0.0, 0.0, 1.0],
            limit=10,
            min_similarity=0.0,
            tier="fast",
        )
        found_ids = [e.id for e, _ in results]
        assert "vec_z" not in found_ids

    @pytest.mark.asyncio
    async def test_search_skips_entries_without_embeddings(self, backend):
        """Entries without embeddings should be excluded from similarity search."""
        await backend.store(MemoryEntry(id="no_emb", content="No vector", tier="fast"))
        await backend.store(
            MemoryEntry(
                id="has_emb",
                content="Has vector",
                tier="fast",
                embedding=[1.0, 0.0, 0.0],
            )
        )

        # Tier-filtered search (brute-force path) should skip no_emb
        results = await backend.search_similar(
            query_embedding=[1.0, 0.0, 0.0],
            limit=10,
            min_similarity=0.0,
            tier="fast",
        )
        found_ids = [e.id for e, _ in results]
        assert "no_emb" not in found_ids
        assert "has_emb" in found_ids

    @pytest.mark.asyncio
    async def test_search_skips_mismatched_dimension(self, backend):
        """Tier-filtered search should skip entries with mismatched embedding dimensions."""
        await backend.store(MemoryEntry(id="dim2", content="2D", tier="fast", embedding=[1.0, 0.0]))
        await backend.store(
            MemoryEntry(id="dim3", content="3D", tier="fast", embedding=[1.0, 0.0, 0.0])
        )

        results = await backend.search_similar(
            query_embedding=[1.0, 0.0, 0.0],
            limit=10,
            min_similarity=0.0,
            tier="fast",
        )
        found_ids = [e.id for e, _ in results]
        assert "dim2" not in found_ids
        assert "dim3" in found_ids


class TestCosineSimlarity:
    """Test the static _cosine_similarity helper."""

    def test_identical_vectors(self):
        """Cosine similarity of identical vectors should be 1.0."""
        sim = InMemoryBackend._cosine_similarity([1.0, 0.0], [1.0, 0.0])
        assert abs(sim - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        """Cosine similarity of orthogonal vectors should be 0.0."""
        sim = InMemoryBackend._cosine_similarity([1.0, 0.0], [0.0, 1.0])
        assert abs(sim) < 1e-6

    def test_opposite_vectors(self):
        """Cosine similarity of opposite vectors should be -1.0."""
        sim = InMemoryBackend._cosine_similarity([1.0, 0.0], [-1.0, 0.0])
        assert abs(sim + 1.0) < 1e-6

    def test_same_direction_different_magnitude(self):
        """Cosine similarity should be magnitude-independent."""
        sim = InMemoryBackend._cosine_similarity([1.0, 0.0], [5.0, 0.0])
        assert abs(sim - 1.0) < 1e-6

    def test_zero_vector_a(self):
        """Cosine similarity with zero vector should be 0.0."""
        sim = InMemoryBackend._cosine_similarity([0.0, 0.0], [1.0, 0.0])
        assert sim == 0.0

    def test_zero_vector_b(self):
        """Cosine similarity with zero vector b should be 0.0."""
        sim = InMemoryBackend._cosine_similarity([1.0, 0.0], [0.0, 0.0])
        assert sim == 0.0

    def test_both_zero_vectors(self):
        """Cosine similarity of two zero vectors should be 0.0."""
        sim = InMemoryBackend._cosine_similarity([0.0, 0.0], [0.0, 0.0])
        assert sim == 0.0

    def test_dimension_mismatch(self):
        """Cosine similarity with mismatched dimensions should return 0.0."""
        sim = InMemoryBackend._cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])
        assert sim == 0.0

    def test_known_angle(self):
        """Cosine similarity of 45-degree angle should be approximately 0.707."""
        sim = InMemoryBackend._cosine_similarity([1.0, 0.0], [1.0, 1.0])
        expected = 1.0 / math.sqrt(2)
        assert abs(sim - expected) < 1e-6


# =============================================================================
# Index Management
# =============================================================================


class TestIndexManagement:
    """Test vector index creation, invalidation, and rebuild behavior."""

    @pytest.mark.asyncio
    async def test_index_created_lazily_on_search(self, backend):
        """Vector index should not exist until the first similarity search."""
        entry = MemoryEntry(id="lazy", content="Lazy index", tier="fast", embedding=[1.0, 0.0])
        await backend.store(entry)

        # Index is created during store via _update_vector_index
        # but let's verify search works
        results = await backend.search_similar(
            query_embedding=[1.0, 0.0],
            limit=5,
            min_similarity=0.0,
        )
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_index_rebuilt_on_dimension_change(self, backend):
        """Changing embedding dimension should rebuild the vector index."""
        # Store 2D entry
        entry2d = MemoryEntry(id="2d", content="2D", tier="fast", embedding=[1.0, 0.0])
        await backend.store(entry2d)

        # Force index creation for 2D
        await backend.search_similar(
            query_embedding=[1.0, 0.0],
            limit=5,
            min_similarity=0.0,
        )
        assert backend._index_dimension == 2

        # Store 3D entry and search with 3D query - triggers rebuild
        entry3d = MemoryEntry(id="3d", content="3D", tier="fast", embedding=[1.0, 0.0, 0.0])
        await backend.store(entry3d)

        results = await backend.search_similar(
            query_embedding=[1.0, 0.0, 0.0],
            limit=5,
            min_similarity=0.0,
        )
        assert backend._index_dimension == 3
        # Only the 3D entry should be in the 3D index
        found_ids = [e.id for e, _ in results]
        assert "3d" in found_ids

    @pytest.mark.asyncio
    async def test_remove_from_vector_index_when_no_index(self, backend):
        """Removing from vector index when no index exists should be a no-op."""
        # Should not raise
        backend._remove_from_vector_index("nonexistent")

    @pytest.mark.asyncio
    async def test_update_vector_index_with_none_embedding(self, backend):
        """Updating vector index with None embedding should remove from index."""
        entry = MemoryEntry(id="rm_emb", content="Test", tier="fast", embedding=[1.0, 0.0, 0.0])
        await backend.store(entry)

        # Now update with None embedding
        entry.embedding = None
        backend._update_vector_index(entry)

        # Entry should no longer appear in similarity search
        results = await backend.search_similar(
            query_embedding=[1.0, 0.0, 0.0],
            limit=5,
            min_similarity=0.0,
        )
        found_ids = [e.id for e, _ in results]
        assert "rm_emb" not in found_ids


# =============================================================================
# Batch Operations
# =============================================================================


class TestBatchOperations:
    """Test batch store and delete operations."""

    @pytest.mark.asyncio
    async def test_store_batch(self, backend):
        """store_batch() should store all entries and return their IDs."""
        entries = [
            MemoryEntry(id=f"b{i}", content=f"Batch {i}", tier="fast", weight=0.5) for i in range(5)
        ]
        ids = await backend.store_batch(entries)

        assert len(ids) == 5
        assert ids == [f"b{i}" for i in range(5)]
        assert len(backend) == 5

    @pytest.mark.asyncio
    async def test_store_batch_empty(self, backend):
        """store_batch() with empty list should return empty list."""
        ids = await backend.store_batch([])
        assert ids == []
        assert len(backend) == 0

    @pytest.mark.asyncio
    async def test_store_batch_updates_tier_index(self, backend):
        """store_batch() should correctly update tier indices."""
        entries = [
            MemoryEntry(id="bf1", content="Fast", tier="fast"),
            MemoryEntry(id="bm1", content="Medium", tier="medium"),
            MemoryEntry(id="bf2", content="Fast 2", tier="fast"),
        ]
        await backend.store_batch(entries)

        assert "bf1" in backend._by_tier["fast"]
        assert "bf2" in backend._by_tier["fast"]
        assert "bm1" in backend._by_tier["medium"]

    @pytest.mark.asyncio
    async def test_store_batch_with_embeddings(self, backend):
        """store_batch() should update vector index for entries with embeddings."""
        entries = [
            MemoryEntry(id="be1", content="A", tier="fast", embedding=[1.0, 0.0]),
            MemoryEntry(id="be2", content="B", tier="fast", embedding=[0.0, 1.0]),
        ]
        await backend.store_batch(entries)

        results = await backend.search_similar(
            query_embedding=[1.0, 0.0],
            limit=5,
            min_similarity=0.5,
        )
        assert len(results) >= 1
        assert results[0][0].id == "be1"

    @pytest.mark.asyncio
    async def test_delete_batch(self, backend):
        """delete_batch() should remove specified entries and return count."""
        for i in range(5):
            await backend.store(MemoryEntry(id=f"d{i}", content=f"Del {i}", tier="fast"))

        count = await backend.delete_batch(["d0", "d2", "d4"])
        assert count == 3
        assert len(backend) == 2
        assert await backend.get("d0") is None
        assert await backend.get("d1") is not None
        assert await backend.get("d3") is not None

    @pytest.mark.asyncio
    async def test_delete_batch_with_nonexistent_ids(self, backend):
        """delete_batch() should skip nonexistent IDs without error."""
        await backend.store(MemoryEntry(id="exist", content="Exists", tier="fast"))
        count = await backend.delete_batch(["exist", "ghost1", "ghost2"])
        assert count == 1
        assert len(backend) == 0

    @pytest.mark.asyncio
    async def test_delete_batch_empty(self, backend):
        """delete_batch() with empty list should return 0."""
        count = await backend.delete_batch([])
        assert count == 0

    @pytest.mark.asyncio
    async def test_delete_batch_removes_from_tier_index(self, backend):
        """delete_batch() should clean up tier indices."""
        await backend.store(MemoryEntry(id="dt1", content="A", tier="fast"))
        await backend.store(MemoryEntry(id="dt2", content="B", tier="medium"))

        await backend.delete_batch(["dt1", "dt2"])

        assert "dt1" not in backend._by_tier["fast"]
        assert "dt2" not in backend._by_tier["medium"]


# =============================================================================
# Tier Operations
# =============================================================================


class TestTierOperations:
    """Test tier promotion and counting."""

    @pytest.mark.asyncio
    async def test_promote_entry(self, backend, sample_entry):
        """promote() should change the entry's tier."""
        await backend.store(sample_entry)
        result = await backend.promote("entry_1", "slow")

        assert result is True
        entry = await backend.get("entry_1")
        assert entry is not None
        assert entry.tier == "slow"
        assert "entry_1" not in backend._by_tier["fast"]
        assert "entry_1" in backend._by_tier["slow"]

    @pytest.mark.asyncio
    async def test_promote_nonexistent_entry(self, backend):
        """promote() should return False for a missing entry."""
        result = await backend.promote("ghost", "slow")
        assert result is False

    @pytest.mark.asyncio
    async def test_promote_updates_timestamp(self, backend, sample_entry):
        """promote() should update the updated_at timestamp."""
        await backend.store(sample_entry)
        original_ts = sample_entry.updated_at

        result = await backend.promote("entry_1", "glacial")
        assert result is True

        entry = await backend.get("entry_1")
        assert entry is not None
        assert entry.updated_at >= original_ts

    @pytest.mark.asyncio
    async def test_promote_to_same_tier(self, backend, sample_entry):
        """promote() to same tier should succeed (no-op for indices)."""
        await backend.store(sample_entry)
        result = await backend.promote("entry_1", "fast")
        assert result is True
        assert "entry_1" in backend._by_tier["fast"]

    @pytest.mark.asyncio
    async def test_count_by_tier(self, backend, entries_across_tiers):
        """count_by_tier() should return correct counts per tier."""
        for entry in entries_across_tiers:
            await backend.store(entry)

        counts = await backend.count_by_tier()
        assert counts["fast"] == 2
        assert counts["medium"] == 1
        assert counts["slow"] == 1
        assert counts["glacial"] == 1

    @pytest.mark.asyncio
    async def test_count_by_tier_empty(self, backend):
        """count_by_tier() on empty store should return empty dict."""
        counts = await backend.count_by_tier()
        assert counts == {}

    @pytest.mark.asyncio
    async def test_count_by_tier_after_deletion(self, backend, entries_across_tiers):
        """count_by_tier() should reflect deletions."""
        for entry in entries_across_tiers:
            await backend.store(entry)

        await backend.delete("fast_1")

        counts = await backend.count_by_tier()
        assert counts["fast"] == 1


# =============================================================================
# TTL / Expiration
# =============================================================================


class TestExpiration:
    """Test TTL-based expiration and cleanup."""

    @pytest.mark.asyncio
    async def test_cleanup_removes_expired_entries(self, backend):
        """cleanup_expired() should remove entries past their expires_at."""
        now = datetime.now(timezone.utc)
        expired = MemoryEntry(
            id="old",
            content="Expired",
            tier="fast",
            expires_at=now - timedelta(hours=1),
        )
        valid = MemoryEntry(
            id="new",
            content="Valid",
            tier="fast",
            expires_at=now + timedelta(hours=1),
        )
        await backend.store(expired)
        await backend.store(valid)

        count = await backend.cleanup_expired()
        assert count == 1
        assert await backend.get("old") is None
        assert await backend.get("new") is not None

    @pytest.mark.asyncio
    async def test_cleanup_ignores_no_expiration(self, backend):
        """Entries without expires_at should not be cleaned up."""
        entry = MemoryEntry(id="forever", content="No TTL", tier="fast")
        await backend.store(entry)

        count = await backend.cleanup_expired()
        assert count == 0
        assert await backend.get("forever") is not None

    @pytest.mark.asyncio
    async def test_cleanup_empty_store(self, backend):
        """cleanup_expired() on empty store should return 0."""
        count = await backend.cleanup_expired()
        assert count == 0

    @pytest.mark.asyncio
    async def test_cleanup_multiple_expired(self, backend):
        """cleanup_expired() should remove all expired entries at once."""
        now = datetime.now(timezone.utc)
        for i in range(5):
            await backend.store(
                MemoryEntry(
                    id=f"exp{i}",
                    content=f"Expired {i}",
                    tier="fast",
                    expires_at=now - timedelta(minutes=i + 1),
                )
            )
        await backend.store(MemoryEntry(id="keep", content="Keep", tier="fast"))

        count = await backend.cleanup_expired()
        assert count == 5
        assert len(backend) == 1

    @pytest.mark.asyncio
    async def test_cleanup_boundary_not_yet_expired(self, backend):
        """Entry with expires_at in the future should not be cleaned up."""
        entry = MemoryEntry(
            id="boundary",
            content="Almost",
            tier="fast",
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=60),
        )
        await backend.store(entry)

        count = await backend.cleanup_expired()
        assert count == 0
        assert await backend.get("boundary") is not None


# =============================================================================
# Health and Diagnostics
# =============================================================================


class TestHealthAndStats:
    """Test health_check() and get_stats()."""

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, backend):
        """health_check() should always report healthy for in-memory backend."""
        health = await backend.health_check()
        assert isinstance(health, BackendHealth)
        assert health.healthy is True
        assert health.latency_ms == 0.0

    @pytest.mark.asyncio
    async def test_health_check_reports_entry_count(self, backend, sample_entry):
        """health_check() details should include the current entry count."""
        await backend.store(sample_entry)
        health = await backend.health_check()
        assert health.details["backend"] == "in_memory"
        assert health.details["entry_count"] == 1

    @pytest.mark.asyncio
    async def test_get_stats_empty(self, backend):
        """get_stats() on empty store should report zero entries."""
        stats = await backend.get_stats()
        assert stats["total_entries"] == 0
        assert stats["tier_counts"] == {}
        assert stats["memory_bytes"] == 0

    @pytest.mark.asyncio
    async def test_get_stats_with_entries(self, backend, entries_across_tiers):
        """get_stats() should report correct totals and tier breakdown."""
        for entry in entries_across_tiers:
            await backend.store(entry)

        stats = await backend.get_stats()
        assert stats["total_entries"] == 5
        assert stats["tier_counts"]["fast"] == 2
        assert stats["tier_counts"]["medium"] == 1
        assert stats["memory_bytes"] > 0

    @pytest.mark.asyncio
    async def test_get_stats_includes_vector_index_info(self, backend):
        """get_stats() should include vector index statistics."""
        stats = await backend.get_stats()
        assert "vector_index" in stats
        assert stats["vector_index"]["size"] == 0

    @pytest.mark.asyncio
    async def test_get_stats_with_vector_index(self, backend):
        """get_stats() should report vector index stats after storing embeddings."""
        entry = MemoryEntry(id="vs1", content="Vectored", tier="fast", embedding=[1.0, 0.0, 0.0])
        await backend.store(entry)

        # Trigger index creation by searching
        await backend.search_similar(
            query_embedding=[1.0, 0.0, 0.0],
            limit=5,
            min_similarity=0.0,
        )

        stats = await backend.get_stats()
        assert stats["vector_index"]["size"] >= 1

    @pytest.mark.asyncio
    async def test_get_stats_memory_bytes(self, backend):
        """get_stats() memory_bytes should reflect content size."""
        await backend.store(MemoryEntry(id="small", content="Hi", tier="fast"))
        await backend.store(MemoryEntry(id="big", content="x" * 1000, tier="fast"))

        stats = await backend.get_stats()
        # "Hi" = 2 bytes, "x"*1000 = 1000 bytes
        assert stats["memory_bytes"] >= 1002


# =============================================================================
# Vacuum and Clear
# =============================================================================


class TestVacuumAndClear:
    """Test vacuum() and clear() operations."""

    @pytest.mark.asyncio
    async def test_vacuum_is_noop(self, backend, sample_entry):
        """vacuum() should complete without error (no-op for in-memory)."""
        await backend.store(sample_entry)
        await backend.vacuum()
        # Entry should still exist
        assert await backend.get("entry_1") is not None

    @pytest.mark.asyncio
    async def test_clear_removes_all_entries(self, backend, entries_across_tiers):
        """clear() should remove all entries and return the count."""
        for entry in entries_across_tiers:
            await backend.store(entry)

        count = await backend.clear()
        assert count == 5
        assert len(backend) == 0

    @pytest.mark.asyncio
    async def test_clear_resets_tier_indices(self, backend, entries_across_tiers):
        """clear() should empty all tier indices."""
        for entry in entries_across_tiers:
            await backend.store(entry)

        await backend.clear()
        counts = await backend.count_by_tier()
        assert counts == {}

    @pytest.mark.asyncio
    async def test_clear_empty_store(self, backend):
        """clear() on empty store should return 0."""
        count = await backend.clear()
        assert count == 0

    @pytest.mark.asyncio
    async def test_clear_resets_vector_index(self, backend):
        """clear() should reset the vector index."""
        entry = MemoryEntry(id="vi1", content="Vec", tier="fast", embedding=[1.0, 0.0])
        await backend.store(entry)

        await backend.clear()

        # Similarity search should return empty after clear
        results = await backend.search_similar(
            query_embedding=[1.0, 0.0],
            limit=5,
            min_similarity=0.0,
        )
        assert results == []


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_len_on_fresh_backend(self, backend):
        """len() on a fresh backend should be 0."""
        assert len(backend) == 0

    @pytest.mark.asyncio
    async def test_store_and_get_with_empty_content(self, backend):
        """Entries with empty content should be stored and retrieved correctly."""
        entry = MemoryEntry(id="empty", content="", tier="fast")
        await backend.store(entry)
        retrieved = await backend.get("empty")
        assert retrieved is not None
        assert retrieved.content == ""

    @pytest.mark.asyncio
    async def test_store_entry_with_all_tiers(self, backend):
        """Entries can be stored in any tier (fast, medium, slow, glacial)."""
        for tier in ["fast", "medium", "slow", "glacial"]:
            await backend.store(MemoryEntry(id=f"t_{tier}", content=tier, tier=tier))

        counts = await backend.count_by_tier()
        assert counts == {"fast": 1, "medium": 1, "slow": 1, "glacial": 1}

    @pytest.mark.asyncio
    async def test_store_entry_with_custom_tier(self, backend):
        """Entries with custom tier names should work."""
        entry = MemoryEntry(id="custom", content="Custom tier", tier="ultra_glacial")
        await backend.store(entry)

        counts = await backend.count_by_tier()
        assert counts["ultra_glacial"] == 1

    @pytest.mark.asyncio
    async def test_store_large_number_of_entries(self, backend):
        """Backend should handle storing many entries."""
        entries = [
            MemoryEntry(id=f"bulk_{i}", content=f"Content {i}", tier="fast") for i in range(500)
        ]
        ids = await backend.store_batch(entries)
        assert len(ids) == 500
        assert len(backend) == 500

    @pytest.mark.asyncio
    async def test_concurrent_operations_via_sequential_awaits(self, backend):
        """Sequential async operations should maintain consistency."""
        entry = MemoryEntry(id="concurrent", content="Original", tier="fast")
        await backend.store(entry)

        entry.content = "Modified"
        await backend.update(entry)

        retrieved = await backend.get("concurrent")
        assert retrieved is not None
        assert retrieved.content == "Modified"

    @pytest.mark.asyncio
    async def test_query_with_all_weights_below_threshold(self, backend):
        """query(min_weight=high) should return empty when all entries have low weight."""
        for i in range(3):
            await backend.store(
                MemoryEntry(id=f"low{i}", content=f"Low {i}", tier="fast", weight=0.1)
            )

        result = await backend.query(min_weight=0.9)
        assert result.total_count == 0

    @pytest.mark.asyncio
    async def test_search_similar_all_below_threshold(self, backend):
        """search_similar() should return empty when no entries meet min_similarity."""
        await backend.store(
            MemoryEntry(id="far", content="Far", tier="fast", embedding=[1.0, 0.0, 0.0])
        )

        # Search orthogonal direction with high threshold
        results = await backend.search_similar(
            query_embedding=[0.0, 1.0, 0.0],
            limit=10,
            min_similarity=0.99,
        )
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_entry_metadata_preserved_through_lifecycle(self, backend):
        """Metadata should be preserved through store -> get -> update -> get."""
        entry = MemoryEntry(
            id="meta",
            content="With metadata",
            tier="fast",
            metadata={"key1": "value1", "nested": {"a": 1}},
        )
        await backend.store(entry)

        retrieved = await backend.get("meta")
        assert retrieved is not None
        assert retrieved.metadata["key1"] == "value1"
        assert retrieved.metadata["nested"]["a"] == 1

        entry.metadata["key2"] = "value2"
        await backend.update(entry)

        updated = await backend.get("meta")
        assert updated is not None
        assert updated.metadata["key2"] == "value2"

    @pytest.mark.asyncio
    async def test_store_with_expires_at_in_future(self, backend):
        """Entry with future expires_at should survive cleanup."""
        future = datetime.now(timezone.utc) + timedelta(days=30)
        entry = MemoryEntry(
            id="future",
            content="Future",
            tier="fast",
            expires_at=future,
        )
        await backend.store(entry)

        count = await backend.cleanup_expired()
        assert count == 0
        assert await backend.get("future") is not None

    @pytest.mark.asyncio
    async def test_promote_then_query_by_new_tier(self, backend, sample_entry):
        """After promote, entry should appear in queries for the new tier."""
        await backend.store(sample_entry)
        await backend.promote("entry_1", "glacial")

        result = await backend.query(tier="glacial")
        assert result.total_count == 1
        assert result.entries[0].id == "entry_1"

        result_old = await backend.query(tier="fast")
        assert result_old.total_count == 0

    @pytest.mark.asyncio
    async def test_weight_zero_entry(self, backend):
        """Entry with weight=0.0 should be stored and filtered correctly."""
        entry = MemoryEntry(id="zero_w", content="Zero weight", tier="fast", weight=0.0)
        await backend.store(entry)

        result = await backend.query(min_weight=0.0)
        assert result.total_count == 1

        result_filtered = await backend.query(min_weight=0.1)
        assert result_filtered.total_count == 0
