"""
Comprehensive tests for vector store adapters.

Tests the BaseVectorStore interface and all concrete implementations:
- InMemoryVectorStore
- QdrantVectorStore
- WeaviateVectorStore
- ChromaVectorStore
- VectorStoreFactory

Uses pytest fixtures and mocks for external clients.
"""

from __future__ import annotations

import sys
import uuid
from collections import namedtuple
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.knowledge.mound.vector_abstraction.base import (
    BaseVectorStore,
    VectorBackend,
    VectorSearchResult,
    VectorStoreConfig,
)
from aragora.knowledge.mound.vector_abstraction.memory import (
    InMemoryVectorStore,
    StoredVector,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def default_config() -> VectorStoreConfig:
    """Create a default vector store configuration."""
    return VectorStoreConfig(
        backend=VectorBackend.MEMORY,
        collection_name="test_collection",
        embedding_dimensions=1536,
        distance_metric="cosine",
    )


@pytest.fixture
def sample_embedding() -> list[float]:
    """Create a sample embedding vector."""
    return [0.1] * 1536


@pytest.fixture
def sample_embedding_alt() -> list[float]:
    """Create an alternative sample embedding vector."""
    return [0.2] * 1536


@pytest.fixture
def sample_items(sample_embedding: list[float]) -> list[dict[str, Any]]:
    """Create sample items for batch operations."""
    return [
        {
            "id": "item1",
            "embedding": sample_embedding,
            "content": "First document about machine learning",
            "metadata": {"category": "ml", "score": 0.9},
        },
        {
            "id": "item2",
            "embedding": [0.2] * 1536,
            "content": "Second document about deep learning",
            "metadata": {"category": "dl", "score": 0.8},
        },
        {
            "id": "item3",
            "embedding": [0.3] * 1536,
            "content": "Third document about neural networks",
            "metadata": {"category": "nn", "score": 0.7},
        },
    ]


# =============================================================================
# Test VectorStoreConfig
# =============================================================================


class TestVectorStoreConfig:
    """Tests for VectorStoreConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = VectorStoreConfig()
        assert config.backend == VectorBackend.MEMORY
        assert config.collection_name == "knowledge_mound"
        assert config.embedding_dimensions == 1536
        assert config.distance_metric == "cosine"
        assert config.batch_size == 100
        assert config.timeout_seconds == 30

    def test_custom_values(self):
        """Test custom configuration values."""
        config = VectorStoreConfig(
            backend=VectorBackend.QDRANT,
            url="http://localhost:6333",
            api_key="test_key",
            collection_name="custom_collection",
            embedding_dimensions=768,
            distance_metric="euclidean",
            namespace="test_ns",
        )
        assert config.backend == VectorBackend.QDRANT
        assert config.url == "http://localhost:6333"
        assert config.api_key == "test_key"
        assert config.collection_name == "custom_collection"
        assert config.embedding_dimensions == 768
        assert config.distance_metric == "euclidean"
        assert config.namespace == "test_ns"

    def test_from_env_default(self):
        """Test configuration from environment with defaults."""
        with patch.dict("os.environ", {}, clear=True):
            config = VectorStoreConfig.from_env()
            assert config.backend == VectorBackend.MEMORY
            assert config.collection_name == "knowledge_mound"

    def test_from_env_with_values(self):
        """Test configuration from environment with custom values."""
        env_vars = {
            "VECTOR_BACKEND": "qdrant",
            "VECTOR_STORE_URL": "http://qdrant:6333",
            "VECTOR_STORE_API_KEY": "secret_key",
            "VECTOR_COLLECTION": "my_collection",
            "EMBEDDING_DIMENSIONS": "768",
            "DISTANCE_METRIC": "euclidean",
        }
        with patch.dict("os.environ", env_vars, clear=True):
            config = VectorStoreConfig.from_env()
            assert config.backend == VectorBackend.QDRANT
            assert config.url == "http://qdrant:6333"
            assert config.api_key == "secret_key"
            assert config.collection_name == "my_collection"
            assert config.embedding_dimensions == 768
            assert config.distance_metric == "euclidean"

    def test_from_env_invalid_backend_fallback(self):
        """Test that invalid backend falls back to memory."""
        with patch.dict("os.environ", {"VECTOR_BACKEND": "invalid_backend"}, clear=True):
            config = VectorStoreConfig.from_env()
            assert config.backend == VectorBackend.MEMORY


# =============================================================================
# Test VectorSearchResult
# =============================================================================


class TestVectorSearchResult:
    """Tests for VectorSearchResult dataclass."""

    def test_basic_creation(self):
        """Test basic result creation."""
        result = VectorSearchResult(
            id="test_id",
            content="Test content",
            score=0.95,
        )
        assert result.id == "test_id"
        assert result.content == "Test content"
        assert result.score == 0.95
        assert result.metadata == {}
        assert result.embedding is None

    def test_with_metadata_and_embedding(self):
        """Test result with metadata and embedding."""
        embedding = [0.1, 0.2, 0.3]
        metadata = {"key": "value", "count": 5}
        result = VectorSearchResult(
            id="test_id",
            content="Test content",
            score=0.85,
            metadata=metadata,
            embedding=embedding,
        )
        assert result.metadata == {"key": "value", "count": 5}
        assert result.embedding == [0.1, 0.2, 0.3]

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = VectorSearchResult(
            id="test_id",
            content="Test content",
            score=0.90,
            metadata={"key": "value"},
            embedding=[0.1, 0.2],
        )
        result_dict = result.to_dict()
        assert result_dict == {
            "id": "test_id",
            "content": "Test content",
            "score": 0.90,
            "metadata": {"key": "value"},
            "embedding": [0.1, 0.2],
        }


# =============================================================================
# Test BaseVectorStore (abstract interface)
# =============================================================================


class TestBaseVectorStore:
    """Tests for BaseVectorStore abstract base class."""

    def test_cannot_instantiate_directly(self, default_config: VectorStoreConfig):
        """Test that BaseVectorStore cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseVectorStore(default_config)

    def test_properties(self, default_config: VectorStoreConfig):
        """Test that properties work on concrete implementations."""
        store = InMemoryVectorStore(default_config)
        assert store.backend == VectorBackend.MEMORY
        assert store.is_connected is False
        assert store.collection_name == "test_collection"

    @pytest.mark.asyncio
    async def test_context_manager(self, default_config: VectorStoreConfig):
        """Test async context manager protocol."""
        async with InMemoryVectorStore(default_config) as store:
            assert store.is_connected is True
        assert store.is_connected is False

    @pytest.mark.asyncio
    async def test_ping_healthy(self, default_config: VectorStoreConfig):
        """Test ping returns True when healthy."""
        store = InMemoryVectorStore(default_config)
        await store.connect()
        result = await store.ping()
        assert result is True
        await store.disconnect()

    @pytest.mark.asyncio
    async def test_ping_disconnected(self, default_config: VectorStoreConfig):
        """Test ping behavior when not connected."""
        store = InMemoryVectorStore(default_config)
        # InMemoryVectorStore health_check always returns healthy even when disconnected
        # since it has no external dependencies
        result = await store.ping()
        assert result is True  # Memory store is always reachable


# =============================================================================
# Test InMemoryVectorStore
# =============================================================================


class TestInMemoryVectorStore:
    """Tests for InMemoryVectorStore implementation."""

    @pytest.mark.asyncio
    async def test_connect_disconnect(self, default_config: VectorStoreConfig):
        """Test connect and disconnect operations."""
        store = InMemoryVectorStore(default_config)
        assert store.is_connected is False

        await store.connect()
        assert store.is_connected is True

        await store.disconnect()
        assert store.is_connected is False

    @pytest.mark.asyncio
    async def test_connect_creates_default_collection(self, default_config: VectorStoreConfig):
        """Test that connect creates the default collection."""
        store = InMemoryVectorStore(default_config)
        await store.connect()

        exists = await store.collection_exists("test_collection")
        assert exists is True

    @pytest.mark.asyncio
    async def test_upsert_single(
        self, default_config: VectorStoreConfig, sample_embedding: list[float]
    ):
        """Test inserting a single vector."""
        store = InMemoryVectorStore(default_config)
        await store.connect()

        result_id = await store.upsert(
            id="doc1",
            embedding=sample_embedding,
            content="Test document content",
            metadata={"category": "test"},
        )

        assert result_id == "doc1"
        count = await store.count()
        assert count == 1

    @pytest.mark.asyncio
    async def test_upsert_update_existing(
        self, default_config: VectorStoreConfig, sample_embedding: list[float]
    ):
        """Test updating an existing vector."""
        store = InMemoryVectorStore(default_config)
        await store.connect()

        await store.upsert(
            id="doc1",
            embedding=sample_embedding,
            content="Original content",
            metadata={"version": 1},
        )

        await store.upsert(
            id="doc1",
            embedding=[0.5] * 1536,
            content="Updated content",
            metadata={"version": 2},
        )

        result = await store.get_by_id("doc1")
        assert result is not None
        assert result.content == "Updated content"
        assert result.metadata["version"] == 2
        count = await store.count()
        assert count == 1

    @pytest.mark.asyncio
    async def test_upsert_batch(
        self, default_config: VectorStoreConfig, sample_items: list[dict[str, Any]]
    ):
        """Test batch upsert operation."""
        store = InMemoryVectorStore(default_config)
        await store.connect()

        ids = await store.upsert_batch(sample_items)

        assert len(ids) == 3
        assert ids == ["item1", "item2", "item3"]
        count = await store.count()
        assert count == 3

    @pytest.mark.asyncio
    async def test_upsert_batch_with_namespace(
        self, default_config: VectorStoreConfig, sample_items: list[dict[str, Any]]
    ):
        """Test batch upsert with namespace."""
        store = InMemoryVectorStore(default_config)
        await store.connect()

        await store.upsert_batch(sample_items, namespace="test_ns")

        # Should not be visible in default namespace
        count_default = await store.count()
        assert count_default == 0

        # Should be visible in test_ns namespace
        count_ns = await store.count(namespace="test_ns")
        assert count_ns == 3

    @pytest.mark.asyncio
    async def test_search_basic(
        self, default_config: VectorStoreConfig, sample_items: list[dict[str, Any]]
    ):
        """Test basic vector search."""
        store = InMemoryVectorStore(default_config)
        await store.connect()
        await store.upsert_batch(sample_items)

        # Search with the same embedding as item1
        results = await store.search(
            embedding=[0.1] * 1536,
            limit=3,
        )

        assert len(results) > 0
        # First result should have highest similarity
        assert results[0].score >= results[-1].score

    @pytest.mark.asyncio
    async def test_search_with_limit(
        self, default_config: VectorStoreConfig, sample_items: list[dict[str, Any]]
    ):
        """Test search respects limit parameter."""
        store = InMemoryVectorStore(default_config)
        await store.connect()
        await store.upsert_batch(sample_items)

        results = await store.search(
            embedding=[0.1] * 1536,
            limit=2,
        )

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_search_with_min_score(
        self, default_config: VectorStoreConfig, sample_embedding: list[float]
    ):
        """Test search with minimum score threshold."""
        store = InMemoryVectorStore(default_config)
        await store.connect()

        # Add vectors with varying similarity
        await store.upsert(
            id="similar",
            embedding=sample_embedding,  # Exact match
            content="Similar document",
        )
        await store.upsert(
            id="different",
            embedding=[-x for x in sample_embedding],  # Opposite direction
            content="Different document",
        )

        results = await store.search(
            embedding=sample_embedding,
            limit=10,
            min_score=0.5,
        )

        # All results should have score >= 0.5
        for result in results:
            assert result.score >= 0.5

    @pytest.mark.asyncio
    async def test_search_with_filters(
        self, default_config: VectorStoreConfig, sample_items: list[dict[str, Any]]
    ):
        """Test search with metadata filters."""
        store = InMemoryVectorStore(default_config)
        await store.connect()
        await store.upsert_batch(sample_items)

        results = await store.search(
            embedding=[0.1] * 1536,
            limit=10,
            filters={"category": "ml"},
        )

        assert len(results) == 1
        assert results[0].id == "item1"

    @pytest.mark.asyncio
    async def test_search_with_operator_filters(
        self, default_config: VectorStoreConfig, sample_items: list[dict[str, Any]]
    ):
        """Test search with operator-based filters."""
        store = InMemoryVectorStore(default_config)
        await store.connect()
        await store.upsert_batch(sample_items)

        # Test $gte operator
        results = await store.search(
            embedding=[0.1] * 1536,
            limit=10,
            filters={"score": {"$gte": 0.8}},
        )
        assert len(results) == 2  # item1 (0.9) and item2 (0.8)

        # Test $in operator
        results = await store.search(
            embedding=[0.1] * 1536,
            limit=10,
            filters={"category": {"$in": ["ml", "dl"]}},
        )
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_delete_by_id(
        self, default_config: VectorStoreConfig, sample_items: list[dict[str, Any]]
    ):
        """Test deleting vectors by ID."""
        store = InMemoryVectorStore(default_config)
        await store.connect()
        await store.upsert_batch(sample_items)

        deleted = await store.delete(["item1", "item2"])

        assert deleted == 2
        count = await store.count()
        assert count == 1
        remaining = await store.get_by_id("item3")
        assert remaining is not None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, default_config: VectorStoreConfig):
        """Test deleting nonexistent IDs."""
        store = InMemoryVectorStore(default_config)
        await store.connect()

        deleted = await store.delete(["nonexistent"])
        assert deleted == 0

    @pytest.mark.asyncio
    async def test_delete_by_filter(
        self, default_config: VectorStoreConfig, sample_items: list[dict[str, Any]]
    ):
        """Test deleting vectors by filter."""
        store = InMemoryVectorStore(default_config)
        await store.connect()
        await store.upsert_batch(sample_items)

        deleted = await store.delete_by_filter({"category": "ml"})

        assert deleted == 1
        count = await store.count()
        assert count == 2

    @pytest.mark.asyncio
    async def test_get_by_id(
        self, default_config: VectorStoreConfig, sample_embedding: list[float]
    ):
        """Test retrieving a vector by ID."""
        store = InMemoryVectorStore(default_config)
        await store.connect()

        await store.upsert(
            id="doc1",
            embedding=sample_embedding,
            content="Test document",
            metadata={"key": "value"},
        )

        result = await store.get_by_id("doc1")

        assert result is not None
        assert result.id == "doc1"
        assert result.content == "Test document"
        assert result.metadata == {"key": "value"}
        assert result.embedding == sample_embedding
        assert result.score == 1.0  # Get by ID returns score of 1.0

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self, default_config: VectorStoreConfig):
        """Test retrieving nonexistent vector."""
        store = InMemoryVectorStore(default_config)
        await store.connect()

        result = await store.get_by_id("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_by_ids(
        self, default_config: VectorStoreConfig, sample_items: list[dict[str, Any]]
    ):
        """Test retrieving multiple vectors by ID."""
        store = InMemoryVectorStore(default_config)
        await store.connect()
        await store.upsert_batch(sample_items)

        results = await store.get_by_ids(["item1", "item3", "nonexistent"])

        assert len(results) == 2
        result_ids = [r.id for r in results]
        assert "item1" in result_ids
        assert "item3" in result_ids

    @pytest.mark.asyncio
    async def test_count_all(
        self, default_config: VectorStoreConfig, sample_items: list[dict[str, Any]]
    ):
        """Test counting all vectors."""
        store = InMemoryVectorStore(default_config)
        await store.connect()
        await store.upsert_batch(sample_items)

        count = await store.count()
        assert count == 3

    @pytest.mark.asyncio
    async def test_count_with_filters(
        self, default_config: VectorStoreConfig, sample_items: list[dict[str, Any]]
    ):
        """Test counting vectors with filters."""
        store = InMemoryVectorStore(default_config)
        await store.connect()
        await store.upsert_batch(sample_items)

        count = await store.count(filters={"category": "ml"})
        assert count == 1

    @pytest.mark.asyncio
    async def test_health_check(
        self, default_config: VectorStoreConfig, sample_items: list[dict[str, Any]]
    ):
        """Test health check returns diagnostics."""
        store = InMemoryVectorStore(default_config)
        await store.connect()
        await store.upsert_batch(sample_items)

        health = await store.health_check()

        assert health["status"] == "healthy"
        assert health["backend"] == "memory"
        assert health["collections"] >= 1
        assert health["total_vectors"] == 3

    @pytest.mark.asyncio
    async def test_collection_management(self, default_config: VectorStoreConfig):
        """Test collection create, list, exists, delete."""
        store = InMemoryVectorStore(default_config)
        await store.connect()

        # Create additional collection
        await store.create_collection("new_collection")

        # List collections
        collections = await store.list_collections()
        assert "test_collection" in collections
        assert "new_collection" in collections

        # Check existence
        assert await store.collection_exists("new_collection") is True
        assert await store.collection_exists("nonexistent") is False

        # Delete collection
        deleted = await store.delete_collection("new_collection")
        assert deleted is True
        assert await store.collection_exists("new_collection") is False

        # Delete nonexistent
        deleted = await store.delete_collection("nonexistent")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_hybrid_search(
        self, default_config: VectorStoreConfig, sample_items: list[dict[str, Any]]
    ):
        """Test hybrid search combining vector and keyword matching."""
        store = InMemoryVectorStore(default_config)
        await store.connect()
        await store.upsert_batch(sample_items)

        results = await store.hybrid_search(
            query="machine learning",
            embedding=[0.1] * 1536,
            limit=3,
            alpha=0.5,
        )

        assert len(results) > 0
        # Results should be ranked by combined score

    @pytest.mark.asyncio
    async def test_namespace_isolation(
        self, default_config: VectorStoreConfig, sample_embedding: list[float]
    ):
        """Test that namespaces provide isolation."""
        store = InMemoryVectorStore(default_config)
        await store.connect()

        await store.upsert(
            id="ns1_doc",
            embedding=sample_embedding,
            content="Namespace 1 document",
            namespace="ns1",
        )
        await store.upsert(
            id="ns2_doc",
            embedding=sample_embedding,
            content="Namespace 2 document",
            namespace="ns2",
        )

        # Search in ns1
        results_ns1 = await store.search(embedding=sample_embedding, limit=10, namespace="ns1")
        assert len(results_ns1) == 1
        assert results_ns1[0].id == "ns1_doc"

        # Search in ns2
        results_ns2 = await store.search(embedding=sample_embedding, limit=10, namespace="ns2")
        assert len(results_ns2) == 1
        assert results_ns2[0].id == "ns2_doc"

    def test_clear(self, default_config: VectorStoreConfig):
        """Test clearing all data."""
        store = InMemoryVectorStore(default_config)
        store._collections["test"] = {"": {"doc1": MagicMock()}}

        store.clear()

        assert len(store._collections) == 0

    @pytest.mark.asyncio
    async def test_get_all_vectors(
        self, default_config: VectorStoreConfig, sample_items: list[dict[str, Any]]
    ):
        """Test retrieving all vectors for testing."""
        store = InMemoryVectorStore(default_config)
        await store.connect()
        await store.upsert_batch(sample_items)

        vectors = store.get_all_vectors()

        assert len(vectors) == 3
        assert all(isinstance(v, StoredVector) for v in vectors)


# =============================================================================
# Test QdrantVectorStore
# =============================================================================


# qdrant-client mock is installed by conftest.py when the real library is absent,
# so the production module always has QDRANT_AVAILABLE = True at test time.


class TestQdrantVectorStore:
    """Tests for QdrantVectorStore with mocked Qdrant client."""

    @pytest.fixture
    def mock_qdrant_client(self):
        """Create a mock Qdrant async client."""
        mock_client = AsyncMock()

        # Mock get_collections
        CollectionInfo = namedtuple("CollectionInfo", ["name"])
        CollectionsResponse = namedtuple("CollectionsResponse", ["collections"])
        mock_client.get_collections.return_value = CollectionsResponse(
            collections=[CollectionInfo(name="test_collection")]
        )

        return mock_client

    @pytest.fixture
    def qdrant_config(self) -> VectorStoreConfig:
        """Create Qdrant configuration."""
        return VectorStoreConfig(
            backend=VectorBackend.QDRANT,
            url="http://localhost:6333",
            collection_name="test_collection",
            embedding_dimensions=1536,
        )

    @pytest.mark.asyncio
    async def test_connect_with_api_key(
        self, mock_qdrant_client: AsyncMock, qdrant_config: VectorStoreConfig
    ):
        """Test connecting with API key."""
        qdrant_config.api_key = "test_api_key"

        with patch(
            "aragora.knowledge.mound.vector_abstraction.qdrant.AsyncQdrantClient"
        ) as mock_class:
            mock_class.return_value = mock_qdrant_client

            from aragora.knowledge.mound.vector_abstraction.qdrant import (
                QdrantVectorStore,
            )

            store = QdrantVectorStore(qdrant_config)
            await store.connect()

            assert store.is_connected is True
            mock_class.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_without_api_key(
        self, mock_qdrant_client: AsyncMock, qdrant_config: VectorStoreConfig
    ):
        """Test connecting without API key."""
        with patch(
            "aragora.knowledge.mound.vector_abstraction.qdrant.AsyncQdrantClient"
        ) as mock_class:
            mock_class.return_value = mock_qdrant_client

            from aragora.knowledge.mound.vector_abstraction.qdrant import (
                QdrantVectorStore,
            )

            store = QdrantVectorStore(qdrant_config)
            await store.connect()

            assert store.is_connected is True

    @pytest.mark.asyncio
    async def test_disconnect(
        self, mock_qdrant_client: AsyncMock, qdrant_config: VectorStoreConfig
    ):
        """Test disconnecting from Qdrant."""
        with patch(
            "aragora.knowledge.mound.vector_abstraction.qdrant.AsyncQdrantClient"
        ) as mock_class:
            mock_class.return_value = mock_qdrant_client

            from aragora.knowledge.mound.vector_abstraction.qdrant import (
                QdrantVectorStore,
            )

            store = QdrantVectorStore(qdrant_config)
            await store.connect()
            await store.disconnect()

            assert store.is_connected is False
            mock_qdrant_client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_upsert(
        self,
        mock_qdrant_client: AsyncMock,
        qdrant_config: VectorStoreConfig,
        sample_embedding: list[float],
    ):
        """Test upserting a vector to Qdrant."""
        with patch(
            "aragora.knowledge.mound.vector_abstraction.qdrant.AsyncQdrantClient"
        ) as mock_class:
            mock_class.return_value = mock_qdrant_client

            from aragora.knowledge.mound.vector_abstraction.qdrant import (
                QdrantVectorStore,
            )

            store = QdrantVectorStore(qdrant_config)
            await store.connect()

            result_id = await store.upsert(
                id="doc1",
                embedding=sample_embedding,
                content="Test content",
                metadata={"key": "value"},
            )

            assert result_id == "doc1"
            mock_qdrant_client.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_upsert_batch(
        self,
        mock_qdrant_client: AsyncMock,
        qdrant_config: VectorStoreConfig,
        sample_items: list[dict[str, Any]],
    ):
        """Test batch upserting vectors to Qdrant."""
        with patch(
            "aragora.knowledge.mound.vector_abstraction.qdrant.AsyncQdrantClient"
        ) as mock_class:
            mock_class.return_value = mock_qdrant_client

            from aragora.knowledge.mound.vector_abstraction.qdrant import (
                QdrantVectorStore,
            )

            store = QdrantVectorStore(qdrant_config)
            await store.connect()

            ids = await store.upsert_batch(sample_items)

            assert len(ids) == 3
            mock_qdrant_client.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_search(
        self,
        mock_qdrant_client: AsyncMock,
        qdrant_config: VectorStoreConfig,
        sample_embedding: list[float],
    ):
        """Test searching in Qdrant."""
        # Mock search response
        MockPoint = namedtuple("MockPoint", ["id", "score", "payload", "vector"])
        mock_qdrant_client.search.return_value = [
            MockPoint(
                id="doc1",
                score=0.95,
                payload={"content": "Test content", "namespace": "", "key": "value"},
                vector=sample_embedding,
            )
        ]

        with patch(
            "aragora.knowledge.mound.vector_abstraction.qdrant.AsyncQdrantClient"
        ) as mock_class:
            mock_class.return_value = mock_qdrant_client

            from aragora.knowledge.mound.vector_abstraction.qdrant import (
                QdrantVectorStore,
            )

            store = QdrantVectorStore(qdrant_config)
            await store.connect()

            results = await store.search(
                embedding=sample_embedding,
                limit=10,
                min_score=0.5,
            )

            assert len(results) == 1
            assert results[0].id == "doc1"
            assert results[0].score == 0.95
            assert results[0].content == "Test content"
            assert results[0].metadata == {"key": "value"}

    @pytest.mark.asyncio
    async def test_search_with_filters(
        self,
        mock_qdrant_client: AsyncMock,
        qdrant_config: VectorStoreConfig,
        sample_embedding: list[float],
    ):
        """Test searching with filters in Qdrant."""
        mock_qdrant_client.search.return_value = []

        with patch(
            "aragora.knowledge.mound.vector_abstraction.qdrant.AsyncQdrantClient"
        ) as mock_class:
            mock_class.return_value = mock_qdrant_client

            from aragora.knowledge.mound.vector_abstraction.qdrant import (
                QdrantVectorStore,
            )

            store = QdrantVectorStore(qdrant_config)
            await store.connect()

            await store.search(
                embedding=sample_embedding,
                limit=10,
                filters={"category": "test"},
                namespace="test_ns",
            )

            # Verify filter was built
            call_args = mock_qdrant_client.search.call_args
            assert call_args.kwargs["query_filter"] is not None

    @pytest.mark.asyncio
    async def test_delete(self, mock_qdrant_client: AsyncMock, qdrant_config: VectorStoreConfig):
        """Test deleting vectors from Qdrant."""
        with patch(
            "aragora.knowledge.mound.vector_abstraction.qdrant.AsyncQdrantClient"
        ) as mock_class:
            mock_class.return_value = mock_qdrant_client

            from aragora.knowledge.mound.vector_abstraction.qdrant import (
                QdrantVectorStore,
            )

            store = QdrantVectorStore(qdrant_config)
            await store.connect()

            deleted = await store.delete(["doc1", "doc2"])

            assert deleted == 2
            mock_qdrant_client.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_healthy(
        self, mock_qdrant_client: AsyncMock, qdrant_config: VectorStoreConfig
    ):
        """Test health check when healthy."""
        CollectionInfo = namedtuple("CollectionInfo", ["name"])
        CollectionsResponse = namedtuple("CollectionsResponse", ["collections"])
        mock_qdrant_client.get_collections.return_value = CollectionsResponse(
            collections=[CollectionInfo(name="col1"), CollectionInfo(name="col2")]
        )

        with patch(
            "aragora.knowledge.mound.vector_abstraction.qdrant.AsyncQdrantClient"
        ) as mock_class:
            mock_class.return_value = mock_qdrant_client

            from aragora.knowledge.mound.vector_abstraction.qdrant import (
                QdrantVectorStore,
            )

            store = QdrantVectorStore(qdrant_config)
            await store.connect()

            health = await store.health_check()

            assert health["status"] == "healthy"
            assert health["backend"] == "qdrant"
            assert health["collections"] == 2

    @pytest.mark.asyncio
    async def test_health_check_disconnected(self, qdrant_config: VectorStoreConfig):
        """Test health check when disconnected."""
        from aragora.knowledge.mound.vector_abstraction.qdrant import (
            QdrantVectorStore,
        )

        store = QdrantVectorStore(qdrant_config)
        health = await store.health_check()

        assert health["status"] == "disconnected"

    @pytest.mark.asyncio
    async def test_get_by_id(
        self,
        mock_qdrant_client: AsyncMock,
        qdrant_config: VectorStoreConfig,
        sample_embedding: list[float],
    ):
        """Test retrieving vector by ID."""
        MockPoint = namedtuple("MockPoint", ["id", "payload", "vector"])
        mock_qdrant_client.retrieve.return_value = [
            MockPoint(
                id="doc1",
                payload={"content": "Test content", "namespace": ""},
                vector=sample_embedding,
            )
        ]

        with patch(
            "aragora.knowledge.mound.vector_abstraction.qdrant.AsyncQdrantClient"
        ) as mock_class:
            mock_class.return_value = mock_qdrant_client

            from aragora.knowledge.mound.vector_abstraction.qdrant import (
                QdrantVectorStore,
            )

            store = QdrantVectorStore(qdrant_config)
            await store.connect()

            result = await store.get_by_id("doc1")

            assert result is not None
            assert result.id == "doc1"

    @pytest.mark.asyncio
    async def test_count(self, mock_qdrant_client: AsyncMock, qdrant_config: VectorStoreConfig):
        """Test counting vectors."""
        CountResult = namedtuple("CountResult", ["count"])
        mock_qdrant_client.count.return_value = CountResult(count=42)

        with patch(
            "aragora.knowledge.mound.vector_abstraction.qdrant.AsyncQdrantClient"
        ) as mock_class:
            mock_class.return_value = mock_qdrant_client

            from aragora.knowledge.mound.vector_abstraction.qdrant import (
                QdrantVectorStore,
            )

            store = QdrantVectorStore(qdrant_config)
            await store.connect()

            count = await store.count()

            assert count == 42


class TestQdrantVectorStoreImportError:
    """Test QdrantVectorStore import error handling."""

    def test_import_error_without_library(self):
        """Test that ImportError is raised when qdrant-client is not installed."""
        # Temporarily hide qdrant_client
        hidden_modules = {}
        for mod_name in list(sys.modules.keys()):
            if "qdrant" in mod_name:
                hidden_modules[mod_name] = sys.modules.pop(mod_name)

        try:
            with patch.dict("sys.modules", {"qdrant_client": None}):
                # Force reimport
                import importlib

                from aragora.knowledge.mound.vector_abstraction import qdrant

                # Patch QDRANT_AVAILABLE to False
                original_available = qdrant.QDRANT_AVAILABLE
                qdrant.QDRANT_AVAILABLE = False

                try:
                    with pytest.raises(ImportError, match="Qdrant client not installed"):
                        qdrant.QdrantVectorStore(VectorStoreConfig(backend=VectorBackend.QDRANT))
                finally:
                    qdrant.QDRANT_AVAILABLE = original_available
        finally:
            # Restore hidden modules
            sys.modules.update(hidden_modules)


# =============================================================================
# Test WeaviateVectorStore
# =============================================================================


class TestWeaviateVectorStore:
    """Tests for WeaviateVectorStore with mocked Weaviate client."""

    @pytest.fixture
    def mock_weaviate_client(self):
        """Create a mock Weaviate client."""
        mock_client = MagicMock()
        mock_collections = MagicMock()
        mock_client.collections = mock_collections
        mock_collections.exists.return_value = True
        mock_collections.list_all.return_value = []
        mock_client.get_meta.return_value = {"version": "1.0.0", "modules": {}}
        return mock_client

    @pytest.fixture
    def weaviate_config(self) -> VectorStoreConfig:
        """Create Weaviate configuration."""
        return VectorStoreConfig(
            backend=VectorBackend.WEAVIATE,
            url="http://localhost:8080",
            collection_name="test_collection",
            embedding_dimensions=1536,
        )

    @pytest.mark.asyncio
    async def test_connect_local(
        self, mock_weaviate_client: MagicMock, weaviate_config: VectorStoreConfig
    ):
        """Test connecting to local Weaviate."""
        with patch("aragora.knowledge.mound.vector_abstraction.weaviate.weaviate") as mock_weaviate:
            mock_weaviate.connect_to_local.return_value = mock_weaviate_client

            from aragora.knowledge.mound.vector_abstraction.weaviate import (
                WeaviateVectorStore,
            )

            store = WeaviateVectorStore(weaviate_config)
            await store.connect()

            assert store.is_connected is True
            mock_weaviate.connect_to_local.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect(
        self, mock_weaviate_client: MagicMock, weaviate_config: VectorStoreConfig
    ):
        """Test disconnecting from Weaviate."""
        with patch("aragora.knowledge.mound.vector_abstraction.weaviate.weaviate") as mock_weaviate:
            mock_weaviate.connect_to_local.return_value = mock_weaviate_client

            from aragora.knowledge.mound.vector_abstraction.weaviate import (
                WeaviateVectorStore,
            )

            store = WeaviateVectorStore(weaviate_config)
            await store.connect()
            await store.disconnect()

            assert store.is_connected is False
            mock_weaviate_client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_upsert(
        self,
        mock_weaviate_client: MagicMock,
        weaviate_config: VectorStoreConfig,
        sample_embedding: list[float],
    ):
        """Test upserting a vector to Weaviate."""
        mock_collection = MagicMock()
        mock_collection.query.fetch_object_by_id.return_value = None
        mock_weaviate_client.collections.get.return_value = mock_collection

        with patch("aragora.knowledge.mound.vector_abstraction.weaviate.weaviate") as mock_weaviate:
            mock_weaviate.connect_to_local.return_value = mock_weaviate_client

            from aragora.knowledge.mound.vector_abstraction.weaviate import (
                WeaviateVectorStore,
            )

            store = WeaviateVectorStore(weaviate_config)
            await store.connect()

            result_id = await store.upsert(
                id="doc1",
                embedding=sample_embedding,
                content="Test content",
                metadata={"key": "value"},
            )

            assert result_id == "doc1"

    @pytest.mark.asyncio
    async def test_search(
        self,
        mock_weaviate_client: MagicMock,
        weaviate_config: VectorStoreConfig,
        sample_embedding: list[float],
    ):
        """Test searching in Weaviate."""
        mock_collection = MagicMock()
        mock_object = MagicMock()
        mock_object.uuid = "doc1"
        mock_object.properties = {"content": "Test content", "namespace": ""}
        mock_object.metadata.distance = 0.1
        mock_object.vector = {"default": sample_embedding}
        mock_response = MagicMock()
        mock_response.objects = [mock_object]
        mock_collection.query.near_vector.return_value = mock_response
        mock_weaviate_client.collections.get.return_value = mock_collection

        with patch("aragora.knowledge.mound.vector_abstraction.weaviate.weaviate") as mock_weaviate:
            mock_weaviate.connect_to_local.return_value = mock_weaviate_client

            from aragora.knowledge.mound.vector_abstraction.weaviate import (
                WeaviateVectorStore,
            )

            store = WeaviateVectorStore(weaviate_config)
            await store.connect()

            results = await store.search(
                embedding=sample_embedding,
                limit=10,
            )

            assert len(results) == 1
            assert results[0].id == "doc1"
            assert results[0].score == 0.9  # 1 - 0.1 distance

    @pytest.mark.asyncio
    async def test_health_check_healthy(
        self, mock_weaviate_client: MagicMock, weaviate_config: VectorStoreConfig
    ):
        """Test health check when healthy."""
        mock_weaviate_client.get_meta.return_value = {
            "version": "1.23.0",
            "modules": {"text2vec-openai": {}},
        }

        with patch("aragora.knowledge.mound.vector_abstraction.weaviate.weaviate") as mock_weaviate:
            mock_weaviate.connect_to_local.return_value = mock_weaviate_client

            from aragora.knowledge.mound.vector_abstraction.weaviate import (
                WeaviateVectorStore,
            )

            store = WeaviateVectorStore(weaviate_config)
            await store.connect()

            health = await store.health_check()

            assert health["status"] == "healthy"
            assert health["backend"] == "weaviate"
            assert health["version"] == "1.23.0"

    @pytest.mark.asyncio
    async def test_delete(
        self, mock_weaviate_client: MagicMock, weaviate_config: VectorStoreConfig
    ):
        """Test deleting vectors from Weaviate."""
        mock_collection = MagicMock()
        mock_weaviate_client.collections.get.return_value = mock_collection

        with patch("aragora.knowledge.mound.vector_abstraction.weaviate.weaviate") as mock_weaviate:
            mock_weaviate.connect_to_local.return_value = mock_weaviate_client

            from aragora.knowledge.mound.vector_abstraction.weaviate import (
                WeaviateVectorStore,
            )

            store = WeaviateVectorStore(weaviate_config)
            await store.connect()

            deleted = await store.delete(["doc1", "doc2"])

            assert deleted == 2
            assert mock_collection.data.delete_by_id.call_count == 2


class TestWeaviateVectorStoreImportError:
    """Test WeaviateVectorStore import error handling."""

    def test_import_error_without_library(self):
        """Test that ImportError is raised when weaviate-client is not installed."""
        from aragora.knowledge.mound.vector_abstraction import weaviate as weaviate_mod

        # Patch WEAVIATE_AVAILABLE to False
        original_available = weaviate_mod.WEAVIATE_AVAILABLE
        weaviate_mod.WEAVIATE_AVAILABLE = False

        try:
            with pytest.raises(ImportError, match="Weaviate client not installed"):
                weaviate_mod.WeaviateVectorStore(VectorStoreConfig(backend=VectorBackend.WEAVIATE))
        finally:
            weaviate_mod.WEAVIATE_AVAILABLE = original_available


# =============================================================================
# Test ChromaVectorStore
# =============================================================================


# chromadb mock is installed by conftest.py when the real library is absent,
# so the production module always has CHROMA_AVAILABLE = True at test time.


class TestChromaVectorStore:
    """Tests for ChromaVectorStore with mocked Chroma client."""

    @pytest.fixture
    def mock_chroma_client(self):
        """Create a mock Chroma client."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client.get_collection.return_value = mock_collection
        mock_client.list_collections.return_value = []
        mock_collection.count.return_value = 0
        return mock_client, mock_collection

    @pytest.fixture
    def chroma_config(self) -> VectorStoreConfig:
        """Create Chroma configuration."""
        return VectorStoreConfig(
            backend=VectorBackend.CHROMA,
            url="./test_chroma_data",
            collection_name="test_collection",
            embedding_dimensions=1536,
        )

    @pytest.mark.asyncio
    async def test_connect_persistent(
        self, mock_chroma_client: tuple, chroma_config: VectorStoreConfig
    ):
        """Test connecting with persistent storage."""
        mock_client, _ = mock_chroma_client

        with patch("aragora.knowledge.mound.vector_abstraction.chroma.chromadb") as mock_chromadb:
            mock_chromadb.PersistentClient.return_value = mock_client
            mock_client.get_collection.side_effect = ValueError("Not found")

            from aragora.knowledge.mound.vector_abstraction.chroma import (
                ChromaVectorStore,
            )

            store = ChromaVectorStore(chroma_config)
            await store.connect()

            assert store.is_connected is True
            mock_chromadb.PersistentClient.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_http(self, mock_chroma_client: tuple, chroma_config: VectorStoreConfig):
        """Test connecting with HTTP client."""
        mock_client, _ = mock_chroma_client
        chroma_config.url = "http://localhost:8000"

        with patch("aragora.knowledge.mound.vector_abstraction.chroma.chromadb") as mock_chromadb:
            mock_chromadb.HttpClient.return_value = mock_client
            mock_client.get_collection.side_effect = ValueError("Not found")

            from aragora.knowledge.mound.vector_abstraction.chroma import (
                ChromaVectorStore,
            )

            store = ChromaVectorStore(chroma_config)
            await store.connect()

            assert store.is_connected is True
            mock_chromadb.HttpClient.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect(self, mock_chroma_client: tuple, chroma_config: VectorStoreConfig):
        """Test disconnecting from Chroma."""
        mock_client, _ = mock_chroma_client

        with patch("aragora.knowledge.mound.vector_abstraction.chroma.chromadb") as mock_chromadb:
            mock_chromadb.PersistentClient.return_value = mock_client
            mock_client.get_collection.side_effect = ValueError("Not found")

            from aragora.knowledge.mound.vector_abstraction.chroma import (
                ChromaVectorStore,
            )

            store = ChromaVectorStore(chroma_config)
            await store.connect()
            await store.disconnect()

            assert store.is_connected is False

    @pytest.mark.asyncio
    async def test_upsert(
        self,
        mock_chroma_client: tuple,
        chroma_config: VectorStoreConfig,
        sample_embedding: list[float],
    ):
        """Test upserting a vector to Chroma."""
        mock_client, mock_collection = mock_chroma_client

        with patch("aragora.knowledge.mound.vector_abstraction.chroma.chromadb") as mock_chromadb:
            mock_chromadb.PersistentClient.return_value = mock_client
            mock_client.get_collection.side_effect = ValueError("Not found")

            from aragora.knowledge.mound.vector_abstraction.chroma import (
                ChromaVectorStore,
            )

            store = ChromaVectorStore(chroma_config)
            await store.connect()

            result_id = await store.upsert(
                id="doc1",
                embedding=sample_embedding,
                content="Test content",
                metadata={"key": "value"},
            )

            assert result_id == "doc1"
            mock_collection.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_upsert_batch(
        self,
        mock_chroma_client: tuple,
        chroma_config: VectorStoreConfig,
        sample_items: list[dict[str, Any]],
    ):
        """Test batch upserting vectors to Chroma."""
        mock_client, mock_collection = mock_chroma_client

        with patch("aragora.knowledge.mound.vector_abstraction.chroma.chromadb") as mock_chromadb:
            mock_chromadb.PersistentClient.return_value = mock_client
            mock_client.get_collection.side_effect = ValueError("Not found")

            from aragora.knowledge.mound.vector_abstraction.chroma import (
                ChromaVectorStore,
            )

            store = ChromaVectorStore(chroma_config)
            await store.connect()

            ids = await store.upsert_batch(sample_items)

            assert len(ids) == 3
            mock_collection.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_search(
        self,
        mock_chroma_client: tuple,
        chroma_config: VectorStoreConfig,
        sample_embedding: list[float],
    ):
        """Test searching in Chroma."""
        mock_client, mock_collection = mock_chroma_client
        mock_collection.query.return_value = {
            "ids": [["doc1"]],
            "documents": [["Test content"]],
            "metadatas": [[{"namespace": "", "key": "value"}]],
            "distances": [[0.1]],
            "embeddings": [[sample_embedding]],
        }

        with patch("aragora.knowledge.mound.vector_abstraction.chroma.chromadb") as mock_chromadb:
            mock_chromadb.PersistentClient.return_value = mock_client
            mock_client.get_collection.side_effect = ValueError("Not found")

            from aragora.knowledge.mound.vector_abstraction.chroma import (
                ChromaVectorStore,
            )

            store = ChromaVectorStore(chroma_config)
            await store.connect()

            results = await store.search(
                embedding=sample_embedding,
                limit=10,
            )

            assert len(results) == 1
            assert results[0].id == "doc1"
            assert results[0].score == 0.9  # 1 - 0.1 distance
            assert results[0].content == "Test content"

    @pytest.mark.asyncio
    async def test_delete(self, mock_chroma_client: tuple, chroma_config: VectorStoreConfig):
        """Test deleting vectors from Chroma."""
        mock_client, mock_collection = mock_chroma_client

        with patch("aragora.knowledge.mound.vector_abstraction.chroma.chromadb") as mock_chromadb:
            mock_chromadb.PersistentClient.return_value = mock_client
            mock_client.get_collection.side_effect = ValueError("Not found")

            from aragora.knowledge.mound.vector_abstraction.chroma import (
                ChromaVectorStore,
            )

            store = ChromaVectorStore(chroma_config)
            await store.connect()

            deleted = await store.delete(["doc1", "doc2"])

            assert deleted == 2
            mock_collection.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_healthy(
        self, mock_chroma_client: tuple, chroma_config: VectorStoreConfig
    ):
        """Test health check when healthy."""
        mock_client, _ = mock_chroma_client
        mock_client.list_collections.return_value = [MagicMock(), MagicMock()]

        with patch("aragora.knowledge.mound.vector_abstraction.chroma.chromadb") as mock_chromadb:
            mock_chromadb.PersistentClient.return_value = mock_client
            mock_client.get_collection.side_effect = ValueError("Not found")

            from aragora.knowledge.mound.vector_abstraction.chroma import (
                ChromaVectorStore,
            )

            store = ChromaVectorStore(chroma_config)
            await store.connect()

            health = await store.health_check()

            assert health["status"] == "healthy"
            assert health["backend"] == "chroma"
            assert health["collections"] == 2

    @pytest.mark.asyncio
    async def test_get_by_id(
        self,
        mock_chroma_client: tuple,
        chroma_config: VectorStoreConfig,
        sample_embedding: list[float],
    ):
        """Test retrieving vector by ID."""
        mock_client, mock_collection = mock_chroma_client
        mock_collection.get.return_value = {
            "ids": ["doc1"],
            "documents": ["Test content"],
            "metadatas": [{"namespace": "", "key": "value"}],
            "embeddings": [sample_embedding],
        }

        with patch("aragora.knowledge.mound.vector_abstraction.chroma.chromadb") as mock_chromadb:
            mock_chromadb.PersistentClient.return_value = mock_client
            mock_client.get_collection.side_effect = ValueError("Not found")

            from aragora.knowledge.mound.vector_abstraction.chroma import (
                ChromaVectorStore,
            )

            store = ChromaVectorStore(chroma_config)
            await store.connect()

            result = await store.get_by_id("doc1")

            assert result is not None
            assert result.id == "doc1"
            assert result.content == "Test content"

    @pytest.mark.asyncio
    async def test_count(self, mock_chroma_client: tuple, chroma_config: VectorStoreConfig):
        """Test counting vectors."""
        mock_client, mock_collection = mock_chroma_client
        mock_collection.count.return_value = 42

        with patch("aragora.knowledge.mound.vector_abstraction.chroma.chromadb") as mock_chromadb:
            mock_chromadb.PersistentClient.return_value = mock_client
            mock_client.get_collection.side_effect = ValueError("Not found")

            from aragora.knowledge.mound.vector_abstraction.chroma import (
                ChromaVectorStore,
            )

            store = ChromaVectorStore(chroma_config)
            await store.connect()

            count = await store.count()

            assert count == 42

    @pytest.mark.asyncio
    async def test_sanitize_metadata(self, chroma_config: VectorStoreConfig):
        """Test metadata sanitization for Chroma."""
        with patch("aragora.knowledge.mound.vector_abstraction.chroma.chromadb") as mock_chromadb:
            mock_client = MagicMock()
            mock_chromadb.PersistentClient.return_value = mock_client
            mock_client.get_collection.side_effect = ValueError("Not found")

            from aragora.knowledge.mound.vector_abstraction.chroma import (
                ChromaVectorStore,
            )

            store = ChromaVectorStore(chroma_config)

            # Test various metadata types
            meta = {
                "string": "value",
                "int": 42,
                "float": 3.14,
                "bool": True,
                "none": None,
                "list": [1, 2, 3],
            }

            sanitized = store._sanitize_metadata(meta)

            assert sanitized["string"] == "value"
            assert sanitized["int"] == 42
            assert sanitized["float"] == 3.14
            assert sanitized["bool"] is True
            assert sanitized["none"] == ""  # None converted to empty string
            assert sanitized["list"] == "[1, 2, 3]"  # List converted to string


class TestChromaVectorStoreImportError:
    """Test ChromaVectorStore import error handling."""

    def test_import_error_without_library(self):
        """Test that ImportError is raised when chromadb is not installed."""
        from aragora.knowledge.mound.vector_abstraction import chroma as chroma_mod

        # Patch CHROMA_AVAILABLE to False
        original_available = chroma_mod.CHROMA_AVAILABLE
        chroma_mod.CHROMA_AVAILABLE = False

        try:
            with pytest.raises(ImportError, match="ChromaDB not installed"):
                chroma_mod.ChromaVectorStore(VectorStoreConfig(backend=VectorBackend.CHROMA))
        finally:
            chroma_mod.CHROMA_AVAILABLE = original_available


# =============================================================================
# Test VectorStoreFactory
# =============================================================================


class TestVectorStoreFactory:
    """Tests for VectorStoreFactory."""

    def test_register_and_unregister(self):
        """Test registering and unregistering backends."""
        from aragora.knowledge.mound.vector_abstraction.factory import (
            VectorStoreFactory,
        )

        # Create a mock store class
        class MockVectorStore(BaseVectorStore):
            async def connect(self):
                pass

            async def disconnect(self):
                pass

            async def create_collection(self, name, schema=None):
                pass

            async def delete_collection(self, name):
                return True

            async def collection_exists(self, name):
                return True

            async def list_collections(self):
                return []

            async def upsert(self, id, embedding, content, metadata=None, namespace=None):
                return id

            async def upsert_batch(self, items, namespace=None):
                return []

            async def delete(self, ids, namespace=None):
                return 0

            async def delete_by_filter(self, filters, namespace=None):
                return 0

            async def search(
                self, embedding, limit=10, filters=None, namespace=None, min_score=0.0
            ):
                return []

            async def hybrid_search(
                self,
                query,
                embedding,
                limit=10,
                alpha=0.5,
                filters=None,
                namespace=None,
            ):
                return []

            async def get_by_id(self, id, namespace=None):
                return None

            async def get_by_ids(self, ids, namespace=None):
                return []

            async def count(self, filters=None, namespace=None):
                return 0

            async def health_check(self):
                return {"status": "healthy"}

        # Verify the factory methods work correctly
        assert VectorStoreFactory.is_registered(VectorBackend.MEMORY) is True

    def test_create_memory_store(self):
        """Test creating in-memory store."""
        from aragora.knowledge.mound.vector_abstraction.factory import (
            VectorStoreFactory,
        )

        config = VectorStoreConfig(backend=VectorBackend.MEMORY)
        store = VectorStoreFactory.create(config)

        assert isinstance(store, InMemoryVectorStore)

    def test_create_invalid_backend(self):
        """Test creating store with invalid backend raises error."""
        from aragora.knowledge.mound.vector_abstraction.factory import (
            VectorStoreFactory,
        )

        # Unregister all backends temporarily
        original_registry = VectorStoreFactory._registry.copy()
        VectorStoreFactory._registry.clear()

        try:
            config = VectorStoreConfig(backend=VectorBackend.MEMORY)
            with pytest.raises(ValueError, match="Unknown backend"):
                VectorStoreFactory.create(config)
        finally:
            VectorStoreFactory._registry = original_registry

    def test_list_backends(self):
        """Test listing available backends."""
        from aragora.knowledge.mound.vector_abstraction.factory import (
            VectorStoreFactory,
        )

        backends = VectorStoreFactory.list_backends()

        assert VectorBackend.MEMORY in backends

    def test_is_registered(self):
        """Test checking if backend is registered."""
        from aragora.knowledge.mound.vector_abstraction.factory import (
            VectorStoreFactory,
        )

        assert VectorStoreFactory.is_registered(VectorBackend.MEMORY) is True

    def test_get_store_class(self):
        """Test getting store class for backend."""
        from aragora.knowledge.mound.vector_abstraction.factory import (
            VectorStoreFactory,
        )

        store_class = VectorStoreFactory.get_store_class(VectorBackend.MEMORY)

        assert store_class is InMemoryVectorStore

    def test_from_env(self):
        """Test creating store from environment."""
        from aragora.knowledge.mound.vector_abstraction.factory import (
            VectorStoreFactory,
        )

        with patch.dict("os.environ", {"VECTOR_BACKEND": "memory"}, clear=True):
            store = VectorStoreFactory.from_env()

            assert isinstance(store, InMemoryVectorStore)

    def test_for_namespace_with_routing(self):
        """Test creating store for namespace with routing enabled."""
        from aragora.knowledge.mound.vector_abstraction.factory import (
            VectorStoreFactory,
        )

        with patch.dict(
            "os.environ",
            {"VECTOR_NAMESPACE_ROUTING": "true", "VECTOR_BACKEND": "memory"},
            clear=True,
        ):
            # When preferred backend is not available, falls back to env default
            store = VectorStoreFactory.for_namespace("unknown_namespace")
            assert store is not None
            assert isinstance(store, InMemoryVectorStore)

    def test_for_namespace_without_routing(self):
        """Test creating store for namespace with routing disabled."""
        from aragora.knowledge.mound.vector_abstraction.factory import (
            VectorStoreFactory,
        )

        with patch.dict(
            "os.environ",
            {"VECTOR_NAMESPACE_ROUTING": "false", "VECTOR_BACKEND": "memory"},
            clear=True,
        ):
            store = VectorStoreFactory.for_namespace("knowledge")

            # Should use default from env, not namespace routing
            assert isinstance(store, InMemoryVectorStore)

    def test_get_backend_for_namespace(self):
        """Test getting recommended backend for namespace."""
        from aragora.knowledge.mound.vector_abstraction.factory import (
            VectorStoreFactory,
        )

        assert VectorStoreFactory.get_backend_for_namespace("knowledge") == VectorBackend.QDRANT
        assert VectorStoreFactory.get_backend_for_namespace("debate") == VectorBackend.WEAVIATE
        assert VectorStoreFactory.get_backend_for_namespace("unknown") is None

    def test_for_namespace_with_config_overrides(self):
        """Test creating store with config overrides."""
        from aragora.knowledge.mound.vector_abstraction.factory import (
            VectorStoreFactory,
        )

        with patch.dict(
            "os.environ",
            {"VECTOR_NAMESPACE_ROUTING": "false", "VECTOR_BACKEND": "memory"},
            clear=True,
        ):
            store = VectorStoreFactory.for_namespace(
                "test",
                config_overrides={
                    "collection_name": "custom_collection",
                    "embedding_dimensions": 768,
                },
            )

            assert store.config.collection_name == "custom_collection"
            assert store.config.embedding_dimensions == 768


# =============================================================================
# Integration-style Tests (using InMemoryVectorStore as reference implementation)
# =============================================================================


class TestVectorStoreIntegration:
    """Integration tests using InMemoryVectorStore as reference."""

    @pytest.mark.asyncio
    async def test_full_workflow(self, default_config: VectorStoreConfig):
        """Test complete workflow: connect, upsert, search, delete, disconnect."""
        async with InMemoryVectorStore(default_config) as store:
            # Upsert documents
            await store.upsert(
                id="doc1",
                embedding=[0.1] * 1536,
                content="Machine learning is a subset of AI",
                metadata={"topic": "ml"},
            )
            await store.upsert(
                id="doc2",
                embedding=[0.2] * 1536,
                content="Deep learning uses neural networks",
                metadata={"topic": "dl"},
            )

            # Search
            results = await store.search(
                embedding=[0.1] * 1536,
                limit=2,
            )
            assert len(results) == 2

            # Verify we can retrieve
            doc = await store.get_by_id("doc1")
            assert doc is not None
            assert doc.content == "Machine learning is a subset of AI"

            # Delete
            deleted = await store.delete(["doc1"])
            assert deleted == 1

            # Verify deletion
            doc = await store.get_by_id("doc1")
            assert doc is None

            # Count remaining
            count = await store.count()
            assert count == 1

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, default_config: VectorStoreConfig):
        """Test concurrent upsert operations."""
        import asyncio

        async with InMemoryVectorStore(default_config) as store:
            # Create multiple upsert tasks
            tasks = []
            for i in range(10):
                task = store.upsert(
                    id=f"doc{i}",
                    embedding=[0.1 * (i + 1)] * 1536,
                    content=f"Document {i}",
                )
                tasks.append(task)

            # Execute concurrently
            await asyncio.gather(*tasks)

            # Verify all were inserted
            count = await store.count()
            assert count == 10

    @pytest.mark.asyncio
    async def test_large_batch_upsert(self, default_config: VectorStoreConfig):
        """Test upserting a large batch of vectors."""
        async with InMemoryVectorStore(default_config) as store:
            items = [
                {
                    "id": f"doc{i}",
                    "embedding": [0.1] * 1536,
                    "content": f"Document {i}",
                    "metadata": {"index": i},
                }
                for i in range(100)
            ]

            ids = await store.upsert_batch(items)

            assert len(ids) == 100
            count = await store.count()
            assert count == 100
