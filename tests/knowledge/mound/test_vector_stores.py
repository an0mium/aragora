"""
Tests for multi-backend vector store abstraction.

Tests the BaseVectorStore interface and InMemoryVectorStore implementation.
Other backends (Weaviate, Qdrant, Chroma) require external services
and are tested via integration tests.
"""

import pytest
import asyncio
from datetime import datetime

from aragora.knowledge.mound.vector_abstraction import (
    BaseVectorStore,
    VectorBackend,
    VectorSearchResult,
    VectorStoreConfig,
    VectorStoreFactory,
)
from aragora.knowledge.mound.vector_abstraction.memory import InMemoryVectorStore


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def memory_config():
    """Create config for in-memory store."""
    return VectorStoreConfig(
        backend=VectorBackend.MEMORY,
        collection_name="test_collection",
        embedding_dimensions=4,  # Small for testing
    )


@pytest.fixture
async def memory_store(memory_config):
    """Create and connect an in-memory store."""
    store = InMemoryVectorStore(memory_config)
    await store.connect()
    yield store
    await store.disconnect()


@pytest.fixture
def sample_embeddings():
    """Sample embeddings for testing."""
    return {
        "doc1": [1.0, 0.0, 0.0, 0.0],
        "doc2": [0.9, 0.1, 0.0, 0.0],  # Similar to doc1
        "doc3": [0.0, 1.0, 0.0, 0.0],  # Different from doc1
        "doc4": [0.0, 0.0, 1.0, 0.0],
        "query_similar_to_doc1": [0.95, 0.05, 0.0, 0.0],
    }


# ============================================================================
# Factory Tests
# ============================================================================


class TestVectorStoreFactory:
    """Tests for VectorStoreFactory."""

    def test_list_backends(self):
        """Test listing available backends."""
        backends = VectorStoreFactory.list_backends()
        assert VectorBackend.MEMORY in backends
        # Other backends depend on installed packages

    def test_is_registered(self):
        """Test checking if backend is registered."""
        assert VectorStoreFactory.is_registered(VectorBackend.MEMORY)

    def test_create_memory_store(self, memory_config):
        """Test creating in-memory store."""
        store = VectorStoreFactory.create(memory_config)
        assert isinstance(store, InMemoryVectorStore)
        assert store.backend == VectorBackend.MEMORY

    def test_create_unknown_backend_raises(self):
        """Test that unknown backend raises ValueError."""
        # Temporarily unregister memory to test error
        VectorStoreFactory.unregister(VectorBackend.MEMORY)

        config = VectorStoreConfig(backend=VectorBackend.MEMORY)
        with pytest.raises(ValueError, match="Unknown backend"):
            VectorStoreFactory.create(config)

        # Re-register
        VectorStoreFactory.register(VectorBackend.MEMORY, InMemoryVectorStore)


# ============================================================================
# InMemoryVectorStore Tests
# ============================================================================


class TestInMemoryVectorStore:
    """Tests for InMemoryVectorStore."""

    # --------------------------------------------------------------------------
    # Connection Tests
    # --------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_connect_disconnect(self, memory_config):
        """Test connection lifecycle."""
        store = InMemoryVectorStore(memory_config)

        assert not store.is_connected
        await store.connect()
        assert store.is_connected
        await store.disconnect()
        assert not store.is_connected

    @pytest.mark.asyncio
    async def test_context_manager(self, memory_config):
        """Test async context manager."""
        async with InMemoryVectorStore(memory_config) as store:
            assert store.is_connected
        # Should be disconnected after context

    # --------------------------------------------------------------------------
    # Collection Tests
    # --------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_create_collection(self, memory_store):
        """Test creating a collection."""
        await memory_store.create_collection("new_collection")
        assert await memory_store.collection_exists("new_collection")

    @pytest.mark.asyncio
    async def test_delete_collection(self, memory_store):
        """Test deleting a collection."""
        await memory_store.create_collection("to_delete")
        assert await memory_store.collection_exists("to_delete")

        result = await memory_store.delete_collection("to_delete")
        assert result is True
        assert not await memory_store.collection_exists("to_delete")

    @pytest.mark.asyncio
    async def test_list_collections(self, memory_store):
        """Test listing collections."""
        await memory_store.create_collection("coll1")
        await memory_store.create_collection("coll2")

        collections = await memory_store.list_collections()
        assert "coll1" in collections
        assert "coll2" in collections

    # --------------------------------------------------------------------------
    # Upsert Tests
    # --------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_upsert_single(self, memory_store, sample_embeddings):
        """Test upserting a single vector."""
        result_id = await memory_store.upsert(
            id="doc1",
            embedding=sample_embeddings["doc1"],
            content="Test document 1",
            metadata={"category": "test"},
        )

        assert result_id == "doc1"

        # Verify it was stored
        retrieved = await memory_store.get_by_id("doc1")
        assert retrieved is not None
        assert retrieved.content == "Test document 1"
        assert retrieved.metadata["category"] == "test"

    @pytest.mark.asyncio
    async def test_upsert_update(self, memory_store, sample_embeddings):
        """Test that upsert updates existing vectors."""
        await memory_store.upsert(
            id="doc1",
            embedding=sample_embeddings["doc1"],
            content="Original content",
        )

        await memory_store.upsert(
            id="doc1",
            embedding=sample_embeddings["doc1"],
            content="Updated content",
        )

        retrieved = await memory_store.get_by_id("doc1")
        assert retrieved.content == "Updated content"

    @pytest.mark.asyncio
    async def test_upsert_batch(self, memory_store, sample_embeddings):
        """Test batch upsert."""
        items = [
            {"id": "batch1", "embedding": sample_embeddings["doc1"], "content": "Batch 1"},
            {"id": "batch2", "embedding": sample_embeddings["doc2"], "content": "Batch 2"},
            {"id": "batch3", "embedding": sample_embeddings["doc3"], "content": "Batch 3"},
        ]

        ids = await memory_store.upsert_batch(items)

        assert len(ids) == 3
        assert await memory_store.count() == 3

    # --------------------------------------------------------------------------
    # Search Tests
    # --------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_search_basic(self, memory_store, sample_embeddings):
        """Test basic vector search."""
        # Add documents
        await memory_store.upsert("doc1", sample_embeddings["doc1"], "Document 1")
        await memory_store.upsert("doc2", sample_embeddings["doc2"], "Document 2")
        await memory_store.upsert("doc3", sample_embeddings["doc3"], "Document 3")

        # Search for similar to doc1
        results = await memory_store.search(
            embedding=sample_embeddings["query_similar_to_doc1"],
            limit=2,
        )

        assert len(results) == 2
        # doc1 and doc2 should be most similar
        result_ids = [r.id for r in results]
        assert "doc1" in result_ids
        assert "doc2" in result_ids

    @pytest.mark.asyncio
    async def test_search_with_min_score(self, memory_store, sample_embeddings):
        """Test search with minimum score threshold."""
        await memory_store.upsert("doc1", sample_embeddings["doc1"], "Document 1")
        await memory_store.upsert("doc3", sample_embeddings["doc3"], "Document 3")

        # High threshold should only return very similar
        results = await memory_store.search(
            embedding=sample_embeddings["query_similar_to_doc1"],
            limit=10,
            min_score=0.9,
        )

        # Only doc1 should be similar enough
        assert len(results) >= 1
        assert results[0].id == "doc1"

    @pytest.mark.asyncio
    async def test_search_with_filter(self, memory_store, sample_embeddings):
        """Test search with metadata filter."""
        await memory_store.upsert(
            "doc1", sample_embeddings["doc1"], "Document 1",
            metadata={"category": "A"}
        )
        await memory_store.upsert(
            "doc2", sample_embeddings["doc2"], "Document 2",
            metadata={"category": "B"}
        )

        results = await memory_store.search(
            embedding=sample_embeddings["query_similar_to_doc1"],
            limit=10,
            filters={"category": "B"},
        )

        # Should only return doc2 despite doc1 being more similar
        assert len(results) == 1
        assert results[0].id == "doc2"

    @pytest.mark.asyncio
    async def test_search_with_namespace(self, memory_store, sample_embeddings):
        """Test search with namespace isolation."""
        await memory_store.upsert(
            "doc1", sample_embeddings["doc1"], "Namespace A doc",
            namespace="ns_a"
        )
        await memory_store.upsert(
            "doc2", sample_embeddings["doc2"], "Namespace B doc",
            namespace="ns_b"
        )

        # Search in namespace A
        results = await memory_store.search(
            embedding=sample_embeddings["query_similar_to_doc1"],
            limit=10,
            namespace="ns_a",
        )

        assert len(results) == 1
        assert results[0].id == "doc1"

    # --------------------------------------------------------------------------
    # Hybrid Search Tests
    # --------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_hybrid_search(self, memory_store, sample_embeddings):
        """Test hybrid search combining vector and keyword."""
        await memory_store.upsert(
            "doc1", sample_embeddings["doc1"], "python programming language"
        )
        await memory_store.upsert(
            "doc2", sample_embeddings["doc2"], "java programming language"
        )
        await memory_store.upsert(
            "doc3", sample_embeddings["doc3"], "cooking recipes"
        )

        # Search with keyword "python" - should boost doc1
        results = await memory_store.hybrid_search(
            query="python",
            embedding=sample_embeddings["doc3"],  # Vector pointing to doc3
            limit=3,
            alpha=0.7,  # Favor keyword matching
        )

        # With high alpha, keyword match should dominate
        assert results[0].id == "doc1"

    # --------------------------------------------------------------------------
    # Delete Tests
    # --------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_delete_by_id(self, memory_store, sample_embeddings):
        """Test deleting by ID."""
        await memory_store.upsert("doc1", sample_embeddings["doc1"], "Doc 1")
        await memory_store.upsert("doc2", sample_embeddings["doc2"], "Doc 2")

        deleted = await memory_store.delete(["doc1"])

        assert deleted == 1
        assert await memory_store.get_by_id("doc1") is None
        assert await memory_store.get_by_id("doc2") is not None

    @pytest.mark.asyncio
    async def test_delete_by_filter(self, memory_store, sample_embeddings):
        """Test deleting by filter."""
        await memory_store.upsert(
            "doc1", sample_embeddings["doc1"], "Doc 1",
            metadata={"delete_me": True}
        )
        await memory_store.upsert(
            "doc2", sample_embeddings["doc2"], "Doc 2",
            metadata={"delete_me": False}
        )

        deleted = await memory_store.delete_by_filter({"delete_me": True})

        assert deleted == 1
        assert await memory_store.get_by_id("doc1") is None
        assert await memory_store.get_by_id("doc2") is not None

    # --------------------------------------------------------------------------
    # Count Tests
    # --------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_count(self, memory_store, sample_embeddings):
        """Test counting vectors."""
        assert await memory_store.count() == 0

        await memory_store.upsert("doc1", sample_embeddings["doc1"], "Doc 1")
        await memory_store.upsert("doc2", sample_embeddings["doc2"], "Doc 2")

        assert await memory_store.count() == 2

    @pytest.mark.asyncio
    async def test_count_with_filter(self, memory_store, sample_embeddings):
        """Test counting with filter."""
        await memory_store.upsert(
            "doc1", sample_embeddings["doc1"], "Doc 1",
            metadata={"type": "A"}
        )
        await memory_store.upsert(
            "doc2", sample_embeddings["doc2"], "Doc 2",
            metadata={"type": "B"}
        )
        await memory_store.upsert(
            "doc3", sample_embeddings["doc3"], "Doc 3",
            metadata={"type": "A"}
        )

        count_a = await memory_store.count(filters={"type": "A"})
        assert count_a == 2

    # --------------------------------------------------------------------------
    # Health Check Tests
    # --------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_health_check(self, memory_store, sample_embeddings):
        """Test health check."""
        await memory_store.upsert("doc1", sample_embeddings["doc1"], "Doc 1")

        health = await memory_store.health_check()

        assert health["status"] == "healthy"
        assert health["backend"] == "memory"
        assert health["total_vectors"] >= 1

    @pytest.mark.asyncio
    async def test_ping(self, memory_store):
        """Test ping."""
        assert await memory_store.ping() is True


# ============================================================================
# VectorSearchResult Tests
# ============================================================================


class TestVectorSearchResult:
    """Tests for VectorSearchResult dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = VectorSearchResult(
            id="test_id",
            content="test content",
            score=0.95,
            metadata={"key": "value"},
        )

        d = result.to_dict()

        assert d["id"] == "test_id"
        assert d["content"] == "test content"
        assert d["score"] == 0.95
        assert d["metadata"]["key"] == "value"


# ============================================================================
# VectorStoreConfig Tests
# ============================================================================


class TestVectorStoreConfig:
    """Tests for VectorStoreConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = VectorStoreConfig()

        assert config.backend == VectorBackend.MEMORY
        assert config.collection_name == "knowledge_mound"
        assert config.embedding_dimensions == 1536
        assert config.distance_metric == "cosine"

    def test_from_env(self, monkeypatch):
        """Test creating config from environment."""
        monkeypatch.setenv("VECTOR_BACKEND", "memory")
        monkeypatch.setenv("VECTOR_COLLECTION", "test_collection")

        config = VectorStoreConfig.from_env()

        assert config.backend == VectorBackend.MEMORY
        assert config.collection_name == "test_collection"
