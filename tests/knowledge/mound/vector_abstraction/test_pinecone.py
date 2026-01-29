"""
Tests for Pinecone vector store adapter.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from aragora.knowledge.mound.vector_abstraction.base import (
    VectorBackend,
    VectorSearchResult,
    VectorStoreConfig,
)


class TestPineconeVectorStore:
    """Tests for PineconeVectorStore."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return VectorStoreConfig(
            backend=VectorBackend.PINECONE,
            api_key="test-api-key",
            collection_name="test-index",
            embedding_dimensions=1536,
        )

    @pytest.fixture
    def mock_pinecone(self):
        """Mock Pinecone client."""
        with patch(
            "aragora.knowledge.mound.vector_abstraction.pinecone.PINECONE_AVAILABLE",
            True,
        ):
            with patch("aragora.knowledge.mound.vector_abstraction.pinecone.Pinecone") as mock_pc:
                with patch(
                    "aragora.knowledge.mound.vector_abstraction.pinecone.ServerlessSpec"
                ) as mock_spec:
                    mock_client = MagicMock()
                    mock_pc.return_value = mock_client

                    # Mock list_indexes - return empty so it creates index
                    mock_indexes = MagicMock()
                    mock_indexes.indexes = []
                    mock_client.list_indexes.return_value = mock_indexes

                    # Mock index
                    mock_index = MagicMock()
                    mock_client.Index.return_value = mock_index

                    # Mock ServerlessSpec
                    mock_spec.return_value = MagicMock()

                    yield mock_client, mock_index

    def test_import_check(self):
        """Should check for pinecone library availability."""
        from aragora.knowledge.mound.vector_abstraction import pinecone as pinecone_mod

        # Just verify the module loads
        assert hasattr(pinecone_mod, "PINECONE_AVAILABLE")

    @pytest.mark.asyncio
    async def test_connect(self, config, mock_pinecone):
        """Should connect to Pinecone."""
        mock_client, mock_index = mock_pinecone

        from aragora.knowledge.mound.vector_abstraction.pinecone import (
            PineconeVectorStore,
        )

        store = PineconeVectorStore(config)
        await store.connect()

        assert store.is_connected
        mock_client.list_indexes.assert_called()

    @pytest.mark.asyncio
    async def test_upsert(self, config, mock_pinecone):
        """Should upsert vectors."""
        mock_client, mock_index = mock_pinecone

        from aragora.knowledge.mound.vector_abstraction.pinecone import (
            PineconeVectorStore,
        )

        store = PineconeVectorStore(config)
        store._connected = True
        store._index = mock_index

        result = await store.upsert(
            id="test-id",
            embedding=[0.1] * 1536,
            content="test content",
            metadata={"key": "value"},
        )

        assert result == "test-id"
        mock_index.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_search(self, config, mock_pinecone):
        """Should search for similar vectors."""
        mock_client, mock_index = mock_pinecone

        # Mock search results
        mock_match = MagicMock()
        mock_match.id = "result-1"
        mock_match.score = 0.95
        mock_match.metadata = {"content": "test content", "key": "value"}

        mock_response = MagicMock()
        mock_response.matches = [mock_match]
        mock_index.query.return_value = mock_response

        from aragora.knowledge.mound.vector_abstraction.pinecone import (
            PineconeVectorStore,
        )

        store = PineconeVectorStore(config)
        store._connected = True
        store._index = mock_index

        results = await store.search(
            embedding=[0.1] * 1536,
            limit=10,
        )

        assert len(results) == 1
        assert results[0].id == "result-1"
        assert results[0].score == 0.95
        mock_index.query.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete(self, config, mock_pinecone):
        """Should delete vectors by ID."""
        mock_client, mock_index = mock_pinecone

        from aragora.knowledge.mound.vector_abstraction.pinecone import (
            PineconeVectorStore,
        )

        store = PineconeVectorStore(config)
        store._connected = True
        store._index = mock_index

        count = await store.delete(ids=["id-1", "id-2"])

        assert count == 2
        mock_index.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_by_id(self, config, mock_pinecone):
        """Should get vector by ID."""
        mock_client, mock_index = mock_pinecone

        # Mock fetch result
        mock_vector = MagicMock()
        mock_vector.metadata = {"content": "test content"}
        mock_vector.values = [0.1] * 1536

        mock_response = MagicMock()
        mock_response.vectors = {"test-id": mock_vector}
        mock_index.fetch.return_value = mock_response

        from aragora.knowledge.mound.vector_abstraction.pinecone import (
            PineconeVectorStore,
        )

        store = PineconeVectorStore(config)
        store._connected = True
        store._index = mock_index

        result = await store.get_by_id("test-id")

        assert result is not None
        assert result.id == "test-id"
        assert result.content == "test content"

    @pytest.mark.asyncio
    async def test_count(self, config, mock_pinecone):
        """Should count vectors."""
        mock_client, mock_index = mock_pinecone

        # Mock stats
        mock_stats = MagicMock()
        mock_stats.total_vector_count = 100
        mock_stats.namespaces = {}
        mock_index.describe_index_stats.return_value = mock_stats

        from aragora.knowledge.mound.vector_abstraction.pinecone import (
            PineconeVectorStore,
        )

        store = PineconeVectorStore(config)
        store._connected = True
        store._index = mock_index

        count = await store.count()

        assert count == 100

    @pytest.mark.asyncio
    async def test_health_check(self, config, mock_pinecone):
        """Should return health status."""
        mock_client, mock_index = mock_pinecone

        # Mock stats
        mock_stats = MagicMock()
        mock_stats.total_vector_count = 50
        mock_stats.dimension = 1536
        mock_stats.namespaces = {"default": MagicMock()}
        mock_index.describe_index_stats.return_value = mock_stats

        from aragora.knowledge.mound.vector_abstraction.pinecone import (
            PineconeVectorStore,
        )

        store = PineconeVectorStore(config)
        store._connected = True
        store._index = mock_index
        store._index_name = "test-index"

        health = await store.health_check()

        assert health["status"] == "healthy"
        assert health["backend"] == "pinecone"
        assert health["total_vectors"] == 50


class TestPineconeBackendRegistration:
    """Tests for Pinecone backend registration."""

    def test_backend_enum_exists(self):
        """Should have PINECONE in VectorBackend enum."""
        assert hasattr(VectorBackend, "PINECONE")
        assert VectorBackend.PINECONE.value == "pinecone"
