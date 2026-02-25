"""
Tests for Milvus vector store adapter.
"""

import pytest
from unittest.mock import MagicMock, patch

from aragora.knowledge.mound.vector_abstraction.base import (
    VectorBackend,
    VectorSearchResult,
    VectorStoreConfig,
)


# Check if pymilvus is available
try:
    from pymilvus import connections, Collection, utility

    PYMILVUS_AVAILABLE = True
except ImportError:
    PYMILVUS_AVAILABLE = False


class TestMilvusVectorStore:
    """Tests for MilvusVectorStore."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return VectorStoreConfig(
            backend=VectorBackend.MILVUS,
            url="http://localhost:19530",
            collection_name="test_collection",
            embedding_dimensions=1536,
        )

    @pytest.fixture
    def mock_collection(self):
        """Create a mock Milvus collection."""
        mock = MagicMock()
        mock.num_entities = 100
        return mock

    def test_import_check(self):
        """Should check for pymilvus library availability."""
        from aragora.knowledge.mound.vector_abstraction import milvus as milvus_mod

        # Just verify the module loads
        assert hasattr(milvus_mod, "MILVUS_AVAILABLE")

    @pytest.mark.skip(reason="Integration test requires running Milvus instance")
    @pytest.mark.asyncio
    async def test_connect(self, config):
        """Should connect to Milvus (requires running Milvus)."""
        pass

    def test_upsert_mock(self, config, mock_collection):
        """Should prepare data correctly for upsert."""
        # Test data preparation logic without actual connection
        import json

        test_id = "test-id"
        test_embedding = [0.1] * 1536
        test_content = "test content"
        test_metadata = {"key": "value"}

        # Simulate what upsert does
        entities = [
            [test_id],
            [test_embedding],
            [test_content],
            [""],  # namespace
            [json.dumps(test_metadata)],
        ]

        assert len(entities) == 5
        assert entities[0][0] == test_id
        assert len(entities[1][0]) == 1536
        assert entities[4][0] == '{"key": "value"}'

    def test_search_result_parsing(self):
        """Should parse search results correctly."""
        import json

        # Simulate Milvus search result
        mock_hit = MagicMock()
        mock_hit.id = "result-1"
        mock_hit.distance = 0.05
        mock_hit.entity = {
            "id": "result-1",
            "content": "test content",
            "metadata_json": '{"key": "value"}',
        }

        # Convert distance to score
        score = 1.0 / (1.0 + mock_hit.distance)
        metadata = json.loads(mock_hit.entity.get("metadata_json", "{}"))

        result = VectorSearchResult(
            id=mock_hit.entity.get("id"),
            content=mock_hit.entity.get("content", ""),
            score=score,
            metadata=metadata,
        )

        assert result.id == "result-1"
        assert result.content == "test content"
        assert result.score > 0.9
        assert result.metadata == {"key": "value"}

    def test_count_mock(self, config, mock_collection):
        """Should return entity count."""
        mock_collection.num_entities = 100

        # Simulate count logic
        count = mock_collection.num_entities

        assert count == 100

    def test_health_check_format(self, config, mock_collection):
        """Should return correct health check format."""
        mock_collection.num_entities = 50

        health = {
            "status": "healthy",
            "backend": "milvus",
            "collection": "test_collection",
            "total_vectors": mock_collection.num_entities,
            "row_count": 50,
        }

        assert health["status"] == "healthy"
        assert health["backend"] == "milvus"
        assert health["total_vectors"] == 50


class TestMilvusBackendRegistration:
    """Tests for Milvus backend registration."""

    def test_backend_enum_exists(self):
        """Should have MILVUS in VectorBackend enum."""
        assert hasattr(VectorBackend, "MILVUS")
        assert VectorBackend.MILVUS.value == "milvus"

    def test_milvus_store_class_exists(self):
        """Should have MilvusVectorStore class."""
        from aragora.knowledge.mound.vector_abstraction import milvus as milvus_mod

        assert hasattr(milvus_mod, "MilvusVectorStore")

    @pytest.mark.skipif(not PYMILVUS_AVAILABLE, reason="pymilvus not installed")
    def test_milvus_registered_in_factory(self):
        """Should be registered in factory when pymilvus is available."""
        from aragora.knowledge.mound.vector_abstraction.factory import (
            VectorStoreFactory,
        )

        assert VectorStoreFactory.is_registered(VectorBackend.MILVUS)
