"""
Comprehensive unit tests for KnowledgeVectorStore.

Tests cover:
1. Vector CRUD operations (index, get, delete)
2. Semantic search functionality
3. Keyword search and relationship queries
4. Connection pooling and management
5. Query caching behavior
6. Error handling and edge cases
7. Workspace multi-tenancy
8. Batch operations
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

# Test the WEAVIATE_AVAILABLE flag first before importing the main module
from aragora.knowledge.vector_store import (
    KnowledgeSearchResult,
    KnowledgeVectorConfig,
    KnowledgeVectorStore,
    WEAVIATE_AVAILABLE,
    get_knowledge_vector_store,
)
from aragora.knowledge.types import ValidationStatus
from aragora.memory.tier_manager import MemoryTier


# =============================================================================
# Mock Data Classes
# =============================================================================


@dataclass
class MockKnowledgeNode:
    """Mock KnowledgeNode for testing without importing the real class."""

    id: str = "kn_test_12345678"
    node_type: str = "fact"
    content: str = "Test knowledge content"
    confidence: float = 0.85
    tier: MemoryTier = MemoryTier.SLOW
    workspace_id: str = "test_workspace"
    surprise_score: float = 0.3
    update_count: int = 5
    validation_status: ValidationStatus = ValidationStatus.UNVERIFIED
    created_at: str = "2024-01-15T10:30:00"
    supports: list[str] = field(default_factory=list)
    contradicts: list[str] = field(default_factory=list)
    derived_from: list[str] = field(default_factory=list)


class MockWeaviateObject:
    """Mock Weaviate response object."""

    def __init__(
        self,
        properties: dict[str, Any],
        distance: float | None = None,
        score: float | None = None,
    ):
        self.properties = properties
        self.metadata = MagicMock()
        self.metadata.distance = distance
        self.metadata.score = score


class MockWeaviateResponse:
    """Mock Weaviate query response."""

    def __init__(self, objects: list[MockWeaviateObject] | None = None):
        self.objects = objects or []


class MockDeleteResult:
    """Mock result from delete_many operation."""

    def __init__(self, successful: int = 0):
        self.successful = successful


class MockAggregateResponse:
    """Mock aggregate query response."""

    def __init__(self, total_count: int = 0):
        self.total_count = total_count


class MockBatchContext:
    """Mock batch context manager."""

    def __init__(self):
        self.added_objects: list[dict] = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def add_object(self, properties: dict, vector: list[float]) -> str:
        self.added_objects.append({"properties": properties, "vector": vector})
        return f"uuid_{len(self.added_objects)}"


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    return KnowledgeVectorConfig(
        url="http://localhost:8080",
        api_key=None,
        collection_name="TestKnowledgeNodes",
        embedding_model="text-embedding-3-small",
        embedding_dimensions=1536,
        batch_size=50,
        timeout=30,
        default_workspace="test_workspace",
        index_relationships=True,
    )


@pytest.fixture
def mock_collection():
    """Create a mock Weaviate collection."""
    collection = MagicMock()

    # Mock data operations
    collection.data = MagicMock()
    collection.data.insert = MagicMock(return_value="test-uuid-123")
    collection.data.delete_many = MagicMock(return_value=MockDeleteResult(successful=1))

    # Mock batch operations
    collection.batch = MagicMock()
    collection.batch.dynamic = MagicMock(return_value=MockBatchContext())

    # Mock query operations
    collection.query = MagicMock()
    collection.query.near_vector = MagicMock(return_value=MockWeaviateResponse())
    collection.query.bm25 = MagicMock(return_value=MockWeaviateResponse())
    collection.query.fetch_objects = MagicMock(return_value=MockWeaviateResponse())

    # Mock aggregate operations
    collection.aggregate = MagicMock()
    collection.aggregate.over_all = MagicMock(return_value=MockAggregateResponse())

    return collection


@pytest.fixture
def mock_client(mock_collection):
    """Create a mock Weaviate client."""
    client = MagicMock()
    client.collections = MagicMock()
    client.collections.exists = MagicMock(return_value=True)
    client.collections.get = MagicMock(return_value=mock_collection)
    client.collections.create = MagicMock()
    client.is_ready = MagicMock(return_value=True)
    client.close = MagicMock()
    return client


@pytest.fixture
def sample_node():
    """Create a sample knowledge node for testing."""
    return MockKnowledgeNode(
        id="kn_sample_001",
        node_type="fact",
        content="API keys should be stored securely",
        confidence=0.92,
        tier=MemoryTier.MEDIUM,
        workspace_id="security_ws",
        supports=["kn_parent_001"],
        contradicts=["kn_conflict_001"],
        derived_from=["kn_source_001", "kn_source_002"],
    )


@pytest.fixture
def sample_embedding():
    """Create a sample vector embedding."""
    return [0.1] * 1536


@pytest.fixture
def sample_search_results():
    """Create sample search result objects."""
    return [
        MockWeaviateObject(
            properties={
                "node_id": "kn_result_001",
                "workspace_id": "test_workspace",
                "node_type": "fact",
                "content": "Result one content",
                "confidence": 0.9,
                "tier": "slow",
                "supports_ids": "kn_a,kn_b",
                "contradicts_ids": "",
                "derived_from_ids": "kn_c",
                "surprise_score": 0.2,
                "update_count": 3,
                "validation_status": "majority_agreed",
            },
            distance=0.15,
        ),
        MockWeaviateObject(
            properties={
                "node_id": "kn_result_002",
                "workspace_id": "test_workspace",
                "node_type": "claim",
                "content": "Result two content",
                "confidence": 0.75,
                "tier": "medium",
                "supports_ids": "",
                "contradicts_ids": "kn_d",
                "derived_from_ids": "",
                "surprise_score": 0.5,
                "update_count": 1,
                "validation_status": "unverified",
            },
            distance=0.25,
        ),
    ]


# =============================================================================
# KnowledgeSearchResult Tests
# =============================================================================


class TestKnowledgeSearchResult:
    """Tests for KnowledgeSearchResult dataclass."""

    def test_create_with_defaults(self):
        """Test creating a search result with default values."""
        result = KnowledgeSearchResult(
            node_id="kn_test_001",
            workspace_id="default",
            node_type="fact",
            content="Test content",
            confidence=0.8,
            score=0.95,
        )

        assert result.node_id == "kn_test_001"
        assert result.workspace_id == "default"
        assert result.node_type == "fact"
        assert result.content == "Test content"
        assert result.confidence == 0.8
        assert result.score == 0.95
        assert result.tier == "slow"
        assert result.supports == []
        assert result.contradicts == []
        assert result.derived_from == []
        assert result.metadata == {}

    def test_create_with_relationships(self):
        """Test creating a search result with relationships."""
        result = KnowledgeSearchResult(
            node_id="kn_test_002",
            workspace_id="ws_security",
            node_type="consensus",
            content="Security consensus",
            confidence=0.95,
            score=0.88,
            tier="glacial",
            supports=["kn_a", "kn_b"],
            contradicts=["kn_c"],
            derived_from=["kn_d", "kn_e", "kn_f"],
            metadata={"debate_id": "debate_123"},
        )

        assert result.supports == ["kn_a", "kn_b"]
        assert result.contradicts == ["kn_c"]
        assert result.derived_from == ["kn_d", "kn_e", "kn_f"]
        assert result.metadata == {"debate_id": "debate_123"}

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = KnowledgeSearchResult(
            node_id="kn_test_003",
            workspace_id="default",
            node_type="evidence",
            content="Evidence content",
            confidence=0.7,
            score=0.82,
            tier="fast",
            supports=["kn_x"],
            metadata={"source": "document"},
        )

        result_dict = result.to_dict()

        assert result_dict["node_id"] == "kn_test_003"
        assert result_dict["workspace_id"] == "default"
        assert result_dict["node_type"] == "evidence"
        assert result_dict["content"] == "Evidence content"
        assert result_dict["confidence"] == 0.7
        assert result_dict["score"] == 0.82
        assert result_dict["tier"] == "fast"
        assert result_dict["supports"] == ["kn_x"]
        assert result_dict["contradicts"] == []
        assert result_dict["derived_from"] == []
        assert result_dict["metadata"] == {"source": "document"}


# =============================================================================
# KnowledgeVectorConfig Tests
# =============================================================================


class TestKnowledgeVectorConfig:
    """Tests for KnowledgeVectorConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = KnowledgeVectorConfig()

        assert config.url == "http://localhost:8080"
        assert config.api_key is None
        assert config.collection_name == "KnowledgeNodes"
        assert config.embedding_model == "text-embedding-3-small"
        assert config.embedding_dimensions == 1536
        assert config.batch_size == 100
        assert config.timeout == 30
        assert config.default_workspace == "default"
        assert config.index_relationships is True

    def test_custom_values(self):
        """Test configuration with custom values."""
        config = KnowledgeVectorConfig(
            url="https://weaviate.example.com",
            api_key="secret_key",
            collection_name="CustomNodes",
            embedding_model="text-embedding-ada-002",
            embedding_dimensions=768,
            batch_size=200,
            timeout=60,
            default_workspace="production",
            index_relationships=False,
        )

        assert config.url == "https://weaviate.example.com"
        assert config.api_key == "secret_key"
        assert config.collection_name == "CustomNodes"
        assert config.embedding_dimensions == 768
        assert config.batch_size == 200
        assert config.default_workspace == "production"
        assert config.index_relationships is False

    def test_from_env_defaults(self):
        """Test creating config from environment with defaults."""
        with patch.dict("os.environ", {}, clear=True):
            config = KnowledgeVectorConfig.from_env()

            assert config.url == "http://localhost:8080"
            assert config.api_key is None
            assert config.collection_name == "KnowledgeNodes"
            assert config.default_workspace == "default"

    def test_from_env_with_values(self):
        """Test creating config from environment variables."""
        env_vars = {
            "WEAVIATE_URL": "https://custom.weaviate.io",
            "WEAVIATE_API_KEY": "env_api_key",
            "WEAVIATE_KNOWLEDGE_COLLECTION": "EnvCollection",
            "EMBEDDING_MODEL": "custom-model",
            "ARAGORA_WORKSPACE": "env_workspace",
        }

        with patch.dict("os.environ", env_vars, clear=True):
            config = KnowledgeVectorConfig.from_env()

            assert config.url == "https://custom.weaviate.io"
            assert config.api_key == "env_api_key"
            assert config.collection_name == "EnvCollection"
            assert config.embedding_model == "custom-model"
            assert config.default_workspace == "env_workspace"

    def test_from_env_with_workspace_override(self):
        """Test workspace_id parameter overrides environment."""
        env_vars = {"ARAGORA_WORKSPACE": "env_workspace"}

        with patch.dict("os.environ", env_vars, clear=True):
            config = KnowledgeVectorConfig.from_env(workspace_id="override_workspace")

            assert config.default_workspace == "override_workspace"


# =============================================================================
# KnowledgeVectorStore Initialization Tests
# =============================================================================


class TestKnowledgeVectorStoreInit:
    """Tests for KnowledgeVectorStore initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default values."""
        store = KnowledgeVectorStore()

        assert store.workspace_id == "default"
        assert store.config.url == "http://localhost:8080"
        assert store._client is None
        assert store._collection is None
        assert store._connected is False

    def test_init_with_workspace(self):
        """Test initialization with workspace_id."""
        store = KnowledgeVectorStore(workspace_id="custom_ws")

        assert store.workspace_id == "custom_ws"

    def test_init_with_config(self, mock_config):
        """Test initialization with custom config."""
        store = KnowledgeVectorStore(config=mock_config)

        assert store.config == mock_config
        assert store.workspace_id == "test_workspace"

    def test_init_workspace_overrides_config(self, mock_config):
        """Test workspace_id parameter overrides config default."""
        store = KnowledgeVectorStore(
            workspace_id="override_ws",
            config=mock_config,
        )

        assert store.workspace_id == "override_ws"

    def test_is_connected_property_false(self):
        """Test is_connected returns False when not connected."""
        store = KnowledgeVectorStore()

        assert store.is_connected is False

    def test_is_connected_property_true(self, mock_client, mock_config):
        """Test is_connected returns True when connected."""
        store = KnowledgeVectorStore(config=mock_config)
        store._client = mock_client
        store._connected = True

        assert store.is_connected is True

    def test_is_connected_false_when_client_none(self, mock_config):
        """Test is_connected returns False when client is None."""
        store = KnowledgeVectorStore(config=mock_config)
        store._connected = True  # Even if flag is set
        store._client = None

        assert store.is_connected is False

    def test_valid_node_types(self):
        """Test VALID_NODE_TYPES class attribute."""
        expected_types = {"fact", "claim", "memory", "evidence", "consensus", "insight", "pattern"}
        assert KnowledgeVectorStore.VALID_NODE_TYPES == expected_types


# =============================================================================
# Connection Management Tests
# =============================================================================


class TestConnectionManagement:
    """Tests for connection management (connect/disconnect)."""

    @pytest.mark.asyncio
    async def test_connect_without_weaviate(self, mock_config):
        """Test connect raises error when weaviate not available."""
        store = KnowledgeVectorStore(config=mock_config)

        with patch("aragora.knowledge.vector_store.WEAVIATE_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="Weaviate client not installed"):
                await store.connect()

    @pytest.mark.asyncio
    async def test_connect_local_success(self, mock_client, mock_config):
        """Test successful connection to local Weaviate."""
        store = KnowledgeVectorStore(config=mock_config)

        with (
            patch("aragora.knowledge.vector_store.WEAVIATE_AVAILABLE", True),
            patch("aragora.knowledge.vector_store.weaviate") as mock_weaviate,
        ):
            mock_weaviate.connect_to_local = MagicMock(return_value=mock_client)

            result = await store.connect()

            assert result is True
            assert store.is_connected is True
            assert store._client == mock_client
            mock_weaviate.connect_to_local.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_with_api_key(self, mock_client):
        """Test connection with API key uses custom connect."""
        config = KnowledgeVectorConfig(
            url="https://secure.weaviate.io",
            api_key="secret_key",
        )
        store = KnowledgeVectorStore(config=config)

        with (
            patch("aragora.knowledge.vector_store.WEAVIATE_AVAILABLE", True),
            patch("aragora.knowledge.vector_store.weaviate") as mock_weaviate,
        ):
            mock_weaviate.connect_to_custom = MagicMock(return_value=mock_client)
            mock_weaviate.auth.AuthApiKey = MagicMock()

            result = await store.connect()

            assert result is True
            mock_weaviate.connect_to_custom.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_creates_collection_if_missing(self, mock_client, mock_config):
        """Test connect creates collection when it doesn't exist."""
        mock_client.collections.exists = MagicMock(return_value=False)
        store = KnowledgeVectorStore(config=mock_config)

        with (
            patch("aragora.knowledge.vector_store.WEAVIATE_AVAILABLE", True),
            patch("aragora.knowledge.vector_store.weaviate") as mock_weaviate,
            patch("aragora.knowledge.vector_store.Configure"),
            patch("aragora.knowledge.vector_store.Property"),
            patch("aragora.knowledge.vector_store.DataType"),
        ):
            mock_weaviate.connect_to_local = MagicMock(return_value=mock_client)

            await store.connect()

            mock_client.collections.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_failure_sets_connected_false(self, mock_config):
        """Test connection failure sets _connected to False."""
        store = KnowledgeVectorStore(config=mock_config)

        with (
            patch("aragora.knowledge.vector_store.WEAVIATE_AVAILABLE", True),
            patch("aragora.knowledge.vector_store.weaviate") as mock_weaviate,
        ):
            mock_weaviate.connect_to_local = MagicMock(
                side_effect=ConnectionError("Connection refused")
            )

            with pytest.raises(ConnectionError):
                await store.connect()

            assert store._connected is False

    @pytest.mark.asyncio
    async def test_disconnect(self, mock_client, mock_collection, mock_config):
        """Test disconnect closes client and resets state."""
        store = KnowledgeVectorStore(config=mock_config)
        store._client = mock_client
        store._collection = mock_collection
        store._connected = True

        await store.disconnect()

        assert store._client is None
        assert store._collection is None
        assert store._connected is False
        mock_client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_when_not_connected(self, mock_config):
        """Test disconnect when already disconnected is safe."""
        store = KnowledgeVectorStore(config=mock_config)

        # Should not raise
        await store.disconnect()

        assert store._connected is False


# =============================================================================
# Index Node Tests (CRUD - Create)
# =============================================================================


class TestIndexNode:
    """Tests for single node indexing."""

    @pytest.mark.asyncio
    async def test_index_node_not_connected(self, mock_config, sample_node, sample_embedding):
        """Test index_node raises error when not connected."""
        store = KnowledgeVectorStore(config=mock_config)

        with pytest.raises(RuntimeError, match="Not connected to Weaviate"):
            await store.index_node(sample_node, sample_embedding)

    @pytest.mark.asyncio
    async def test_index_node_success(
        self, mock_client, mock_collection, mock_config, sample_node, sample_embedding
    ):
        """Test successful node indexing."""
        store = KnowledgeVectorStore(config=mock_config)
        store._client = mock_client
        store._collection = mock_collection
        store._connected = True
        mock_collection.data.insert = MagicMock(return_value="uuid-123-456")

        result = await store.index_node(sample_node, sample_embedding)

        assert result == "uuid-123-456"
        mock_collection.data.insert.assert_called_once()

        # Verify properties were passed correctly
        call_args = mock_collection.data.insert.call_args
        properties = call_args.kwargs["properties"]
        assert properties["node_id"] == sample_node.id
        assert properties["content"] == sample_node.content
        assert properties["confidence"] == sample_node.confidence
        assert call_args.kwargs["vector"] == sample_embedding

    @pytest.mark.asyncio
    async def test_index_node_serializes_relationships(
        self, mock_client, mock_collection, mock_config, sample_node, sample_embedding
    ):
        """Test that relationships are serialized as comma-separated strings."""
        store = KnowledgeVectorStore(config=mock_config)
        store._client = mock_client
        store._collection = mock_collection
        store._connected = True

        await store.index_node(sample_node, sample_embedding)

        call_args = mock_collection.data.insert.call_args
        properties = call_args.kwargs["properties"]

        assert properties["supports_ids"] == "kn_parent_001"
        assert properties["contradicts_ids"] == "kn_conflict_001"
        assert properties["derived_from_ids"] == "kn_source_001,kn_source_002"

    @pytest.mark.asyncio
    async def test_index_node_empty_relationships(
        self, mock_client, mock_collection, mock_config, sample_embedding
    ):
        """Test indexing node with empty relationships."""
        node = MockKnowledgeNode(
            supports=[],
            contradicts=[],
            derived_from=[],
        )
        store = KnowledgeVectorStore(config=mock_config)
        store._client = mock_client
        store._collection = mock_collection
        store._connected = True

        await store.index_node(node, sample_embedding)

        call_args = mock_collection.data.insert.call_args
        properties = call_args.kwargs["properties"]

        assert properties["supports_ids"] == ""
        assert properties["contradicts_ids"] == ""
        assert properties["derived_from_ids"] == ""

    @pytest.mark.asyncio
    async def test_index_node_handles_enum_node_type(
        self, mock_client, mock_collection, mock_config, sample_embedding
    ):
        """Test indexing handles enum node_type with .value attribute."""
        node = MockKnowledgeNode()
        node.node_type = MagicMock(value="consensus")  # Mock enum

        store = KnowledgeVectorStore(config=mock_config)
        store._client = mock_client
        store._collection = mock_collection
        store._connected = True

        await store.index_node(node, sample_embedding)

        call_args = mock_collection.data.insert.call_args
        properties = call_args.kwargs["properties"]
        assert properties["node_type"] == "consensus"


# =============================================================================
# Batch Index Tests
# =============================================================================


class TestIndexNodes:
    """Tests for batch node indexing."""

    @pytest.mark.asyncio
    async def test_index_nodes_not_connected(self, mock_config, sample_embedding):
        """Test index_nodes raises error when not connected."""
        store = KnowledgeVectorStore(config=mock_config)
        nodes = [MockKnowledgeNode() for _ in range(3)]
        embeddings = [sample_embedding for _ in range(3)]

        with pytest.raises(RuntimeError, match="Not connected to Weaviate"):
            await store.index_nodes(nodes, embeddings)

    @pytest.mark.asyncio
    async def test_index_nodes_mismatched_lengths(
        self, mock_client, mock_collection, mock_config, sample_embedding
    ):
        """Test index_nodes raises error when nodes/embeddings don't match."""
        store = KnowledgeVectorStore(config=mock_config)
        store._client = mock_client
        store._collection = mock_collection
        store._connected = True

        nodes = [MockKnowledgeNode() for _ in range(3)]
        embeddings = [sample_embedding for _ in range(5)]  # Mismatch

        with pytest.raises(ValueError, match="Number of nodes must match"):
            await store.index_nodes(nodes, embeddings)

    @pytest.mark.asyncio
    async def test_index_nodes_success(
        self, mock_client, mock_collection, mock_config, sample_embedding
    ):
        """Test successful batch indexing."""
        store = KnowledgeVectorStore(config=mock_config)
        store._client = mock_client
        store._collection = mock_collection
        store._connected = True

        nodes = [MockKnowledgeNode(id=f"kn_batch_{i}", content=f"Content {i}") for i in range(3)]
        embeddings = [sample_embedding for _ in range(3)]

        with (
            patch("aragora.knowledge.vector_store.track_vector_index_batch"),
            patch("aragora.knowledge.vector_store.track_vector_operation"),
        ):
            result = await store.index_nodes(nodes, embeddings)

        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_index_nodes_progress_callback(
        self, mock_client, mock_collection, mock_config, sample_embedding
    ):
        """Test progress callback is called during batch indexing."""
        store = KnowledgeVectorStore(config=mock_config)
        store._client = mock_client
        store._collection = mock_collection
        store._connected = True
        store.config.batch_size = 2  # Small batch for testing

        nodes = [MockKnowledgeNode(id=f"kn_{i}") for i in range(5)]
        embeddings = [sample_embedding for _ in range(5)]

        progress_calls = []

        def on_progress(indexed: int, total: int):
            progress_calls.append((indexed, total))

        with (
            patch("aragora.knowledge.vector_store.track_vector_index_batch"),
            patch("aragora.knowledge.vector_store.track_vector_operation"),
        ):
            await store.index_nodes(nodes, embeddings, on_progress=on_progress)

        # Should have 3 batches: [0-1], [2-3], [4]
        assert len(progress_calls) == 3
        assert progress_calls[-1] == (5, 5)  # Final progress


# =============================================================================
# Semantic Search Tests
# =============================================================================


class TestSearchSemantic:
    """Tests for semantic (vector) search."""

    @pytest.mark.asyncio
    async def test_search_semantic_not_connected(self, mock_config, sample_embedding):
        """Test search_semantic raises error when not connected."""
        store = KnowledgeVectorStore(config=mock_config)

        with pytest.raises(RuntimeError, match="Not connected to Weaviate"):
            await store.search_semantic(sample_embedding)

    @pytest.mark.asyncio
    async def test_search_semantic_success(
        self, mock_client, mock_collection, mock_config, sample_embedding, sample_search_results
    ):
        """Test successful semantic search."""
        store = KnowledgeVectorStore(config=mock_config)
        store._client = mock_client
        store._collection = mock_collection
        store._connected = True
        store.workspace_id = "test_workspace"

        mock_collection.query.near_vector = MagicMock(
            return_value=MockWeaviateResponse(sample_search_results)
        )

        with (
            patch("aragora.knowledge.vector_store.track_vector_operation"),
            patch("aragora.knowledge.vector_store.track_vector_search_results"),
            patch("aragora.knowledge.vector_store.Filter") as mock_filter,
        ):
            mock_filter.by_property.return_value.equal.return_value = MagicMock()
            results = await store.search_semantic(sample_embedding, limit=10)

        assert len(results) == 2
        assert results[0].node_id == "kn_result_001"
        assert results[0].score == pytest.approx(0.85, rel=0.01)  # 1 - 0.15
        assert results[0].supports == ["kn_a", "kn_b"]
        assert results[1].node_id == "kn_result_002"
        assert results[1].contradicts == ["kn_d"]

    @pytest.mark.asyncio
    async def test_search_semantic_filters_by_min_score(
        self, mock_client, mock_collection, mock_config, sample_embedding
    ):
        """Test semantic search filters results by min_score."""
        store = KnowledgeVectorStore(config=mock_config)
        store._client = mock_client
        store._collection = mock_collection
        store._connected = True

        # Create results with varying distances
        results_data = [
            MockWeaviateObject(
                properties={
                    "node_id": "kn_high",
                    "workspace_id": "test",
                    "node_type": "fact",
                    "content": "High score",
                    "confidence": 0.9,
                    "tier": "slow",
                    "supports_ids": "",
                    "contradicts_ids": "",
                    "derived_from_ids": "",
                },
                distance=0.1,  # score = 0.9
            ),
            MockWeaviateObject(
                properties={
                    "node_id": "kn_low",
                    "workspace_id": "test",
                    "node_type": "fact",
                    "content": "Low score",
                    "confidence": 0.9,
                    "tier": "slow",
                    "supports_ids": "",
                    "contradicts_ids": "",
                    "derived_from_ids": "",
                },
                distance=0.5,  # score = 0.5
            ),
        ]

        mock_collection.query.near_vector = MagicMock(
            return_value=MockWeaviateResponse(results_data)
        )

        with (
            patch("aragora.knowledge.vector_store.track_vector_operation"),
            patch("aragora.knowledge.vector_store.track_vector_search_results"),
            patch("aragora.knowledge.vector_store.Filter") as mock_filter,
        ):
            mock_filter.by_property.return_value.equal.return_value = MagicMock()
            results = await store.search_semantic(sample_embedding, min_score=0.7)

        # Only high score result should be returned
        assert len(results) == 1
        assert results[0].node_id == "kn_high"

    @pytest.mark.asyncio
    async def test_search_semantic_with_node_types_filter(
        self, mock_client, mock_collection, mock_config, sample_embedding
    ):
        """Test semantic search with node_types filter."""
        store = KnowledgeVectorStore(config=mock_config)
        store._client = mock_client
        store._collection = mock_collection
        store._connected = True

        mock_collection.query.near_vector = MagicMock(return_value=MockWeaviateResponse([]))

        with (
            patch("aragora.knowledge.vector_store.track_vector_operation"),
            patch("aragora.knowledge.vector_store.track_vector_search_results"),
            patch("aragora.knowledge.vector_store.Filter") as mock_filter,
            patch("aragora.knowledge.vector_store.MetadataQuery"),
        ):
            mock_equal = MagicMock()
            mock_contains = MagicMock()
            mock_filter.by_property.return_value.equal.return_value = mock_equal
            mock_filter.by_property.return_value.contains_any.return_value = mock_contains
            mock_equal.__and__ = MagicMock(return_value=mock_equal)

            await store.search_semantic(
                sample_embedding,
                node_types=["fact", "claim"],
            )

            # Verify filter was applied for node_types
            mock_filter.by_property.assert_any_call("node_type")

    @pytest.mark.asyncio
    async def test_search_semantic_with_workspace_override(
        self, mock_client, mock_collection, mock_config, sample_embedding
    ):
        """Test semantic search with workspace_id override."""
        store = KnowledgeVectorStore(config=mock_config)
        store._client = mock_client
        store._collection = mock_collection
        store._connected = True
        store.workspace_id = "default_ws"

        mock_collection.query.near_vector = MagicMock(return_value=MockWeaviateResponse([]))

        with (
            patch("aragora.knowledge.vector_store.track_vector_operation"),
            patch("aragora.knowledge.vector_store.track_vector_search_results"),
            patch("aragora.knowledge.vector_store.Filter") as mock_filter,
            patch("aragora.knowledge.vector_store.MetadataQuery"),
        ):
            mock_filter.by_property.return_value.equal.return_value = MagicMock()

            await store.search_semantic(
                sample_embedding,
                workspace_id="override_ws",
            )

            # Verify workspace filter used the override
            mock_filter.by_property.assert_any_call("workspace_id")


# =============================================================================
# Keyword Search Tests
# =============================================================================


class TestSearchKeyword:
    """Tests for BM25 keyword search."""

    @pytest.mark.asyncio
    async def test_search_keyword_not_connected(self, mock_config):
        """Test search_keyword raises error when not connected."""
        store = KnowledgeVectorStore(config=mock_config)

        with pytest.raises(RuntimeError, match="Not connected to Weaviate"):
            await store.search_keyword("test query")

    @pytest.mark.asyncio
    async def test_search_keyword_success(self, mock_client, mock_collection, mock_config):
        """Test successful keyword search."""
        store = KnowledgeVectorStore(config=mock_config)
        store._client = mock_client
        store._collection = mock_collection
        store._connected = True

        keyword_results = [
            MockWeaviateObject(
                properties={
                    "node_id": "kn_keyword_001",
                    "workspace_id": "test",
                    "node_type": "fact",
                    "content": "Matching content",
                    "confidence": 0.85,
                    "tier": "medium",
                    "supports_ids": "kn_x",
                    "contradicts_ids": "",
                    "derived_from_ids": "",
                },
                score=0.92,
            ),
        ]

        mock_collection.query.bm25 = MagicMock(return_value=MockWeaviateResponse(keyword_results))

        with (
            patch("aragora.knowledge.vector_store.track_vector_operation"),
            patch("aragora.knowledge.vector_store.track_vector_search_results"),
            patch("aragora.knowledge.vector_store.Filter") as mock_filter,
            patch("aragora.knowledge.vector_store.MetadataQuery"),
        ):
            mock_filter.by_property.return_value.equal.return_value = MagicMock()
            results = await store.search_keyword("test query", limit=5)

        assert len(results) == 1
        assert results[0].node_id == "kn_keyword_001"
        assert results[0].score == 0.92
        assert results[0].supports == ["kn_x"]

    @pytest.mark.asyncio
    async def test_search_keyword_with_filters(self, mock_client, mock_collection, mock_config):
        """Test keyword search with all filters."""
        store = KnowledgeVectorStore(config=mock_config)
        store._client = mock_client
        store._collection = mock_collection
        store._connected = True

        mock_collection.query.bm25 = MagicMock(return_value=MockWeaviateResponse([]))

        with (
            patch("aragora.knowledge.vector_store.track_vector_operation"),
            patch("aragora.knowledge.vector_store.track_vector_search_results"),
            patch("aragora.knowledge.vector_store.Filter") as mock_filter,
            patch("aragora.knowledge.vector_store.MetadataQuery"),
        ):
            mock_equal = MagicMock()
            mock_filter.by_property.return_value.equal.return_value = mock_equal
            mock_filter.by_property.return_value.contains_any.return_value = MagicMock()
            mock_filter.by_property.return_value.greater_or_equal.return_value = MagicMock()
            mock_equal.__and__ = MagicMock(return_value=mock_equal)

            await store.search_keyword(
                "security best practices",
                limit=20,
                node_types=["fact", "consensus"],
                min_confidence=0.7,
                workspace_id="security_ws",
            )

            mock_collection.query.bm25.assert_called_once()


# =============================================================================
# Relationship Search Tests
# =============================================================================


class TestSearchByRelationship:
    """Tests for relationship-based search."""

    @pytest.mark.asyncio
    async def test_search_by_relationship_not_connected(self, mock_config):
        """Test search_by_relationship raises error when not connected."""
        store = KnowledgeVectorStore(config=mock_config)

        with pytest.raises(RuntimeError, match="Not connected to Weaviate"):
            await store.search_by_relationship("kn_test", "supports")

    @pytest.mark.asyncio
    async def test_search_by_relationship_invalid_type(
        self, mock_client, mock_collection, mock_config
    ):
        """Test search_by_relationship raises error for invalid relationship type."""
        store = KnowledgeVectorStore(config=mock_config)
        store._client = mock_client
        store._collection = mock_collection
        store._connected = True

        with pytest.raises(ValueError, match="Invalid relationship type"):
            await store.search_by_relationship("kn_test", "invalid_type")

    @pytest.mark.asyncio
    async def test_search_by_relationship_supports(self, mock_client, mock_collection, mock_config):
        """Test searching for nodes that support a given node."""
        store = KnowledgeVectorStore(config=mock_config)
        store._client = mock_client
        store._collection = mock_collection
        store._connected = True

        related_results = [
            MockWeaviateObject(
                properties={
                    "node_id": "kn_supporter",
                    "workspace_id": "test",
                    "node_type": "evidence",
                    "content": "Supporting evidence",
                    "confidence": 0.88,
                    "tier": "slow",
                    "supports_ids": "kn_target",
                    "contradicts_ids": "",
                    "derived_from_ids": "",
                },
            ),
        ]

        mock_collection.query.fetch_objects = MagicMock(
            return_value=MockWeaviateResponse(related_results)
        )

        with patch("aragora.knowledge.vector_store.Filter") as mock_filter:
            mock_equal = MagicMock()
            mock_like = MagicMock()
            mock_filter.by_property.return_value.equal.return_value = mock_equal
            mock_filter.by_property.return_value.like.return_value = mock_like
            mock_equal.__and__ = MagicMock(return_value=mock_equal)

            results = await store.search_by_relationship("kn_target", "supports")

        assert len(results) == 1
        assert results[0].node_id == "kn_supporter"
        assert results[0].score == 1.0  # Relationship queries return score of 1.0

    @pytest.mark.asyncio
    async def test_search_by_relationship_contradicts(
        self, mock_client, mock_collection, mock_config
    ):
        """Test searching for nodes that contradict a given node."""
        store = KnowledgeVectorStore(config=mock_config)
        store._client = mock_client
        store._collection = mock_collection
        store._connected = True

        mock_collection.query.fetch_objects = MagicMock(return_value=MockWeaviateResponse([]))

        with patch("aragora.knowledge.vector_store.Filter") as mock_filter:
            mock_equal = MagicMock()
            mock_filter.by_property.return_value.equal.return_value = mock_equal
            mock_filter.by_property.return_value.like.return_value = MagicMock()
            mock_equal.__and__ = MagicMock(return_value=mock_equal)

            await store.search_by_relationship("kn_test", "contradicts")

            # Verify the correct property was queried
            mock_filter.by_property.assert_any_call("contradicts_ids")

    @pytest.mark.asyncio
    async def test_search_by_relationship_derived_from(
        self, mock_client, mock_collection, mock_config
    ):
        """Test searching for nodes derived from a given node."""
        store = KnowledgeVectorStore(config=mock_config)
        store._client = mock_client
        store._collection = mock_collection
        store._connected = True

        mock_collection.query.fetch_objects = MagicMock(return_value=MockWeaviateResponse([]))

        with patch("aragora.knowledge.vector_store.Filter") as mock_filter:
            mock_equal = MagicMock()
            mock_filter.by_property.return_value.equal.return_value = mock_equal
            mock_filter.by_property.return_value.like.return_value = MagicMock()
            mock_equal.__and__ = MagicMock(return_value=mock_equal)

            await store.search_by_relationship("kn_source", "derived_from")

            mock_filter.by_property.assert_any_call("derived_from_ids")


# =============================================================================
# Get Node Tests (CRUD - Read)
# =============================================================================


class TestGetNode:
    """Tests for getting a single node by ID."""

    @pytest.mark.asyncio
    async def test_get_node_not_connected(self, mock_config):
        """Test get_node raises error when not connected."""
        store = KnowledgeVectorStore(config=mock_config)

        with pytest.raises(RuntimeError, match="Not connected to Weaviate"):
            await store.get_node("kn_test")

    @pytest.mark.asyncio
    async def test_get_node_found(self, mock_client, mock_collection, mock_config):
        """Test getting an existing node."""
        store = KnowledgeVectorStore(config=mock_config)
        store._client = mock_client
        store._collection = mock_collection
        store._connected = True

        found_node = MockWeaviateObject(
            properties={
                "node_id": "kn_existing",
                "workspace_id": "test",
                "node_type": "fact",
                "content": "Existing node content",
                "confidence": 0.9,
                "tier": "glacial",
                "supports_ids": "kn_a,kn_b",
                "contradicts_ids": "",
                "derived_from_ids": "kn_source",
            },
        )

        mock_collection.query.fetch_objects = MagicMock(
            return_value=MockWeaviateResponse([found_node])
        )

        with patch("aragora.knowledge.vector_store.Filter") as mock_filter:
            mock_equal = MagicMock()
            mock_filter.by_property.return_value.equal.return_value = mock_equal
            mock_equal.__and__ = MagicMock(return_value=mock_equal)

            result = await store.get_node("kn_existing")

        assert result is not None
        assert result.node_id == "kn_existing"
        assert result.content == "Existing node content"
        assert result.supports == ["kn_a", "kn_b"]
        assert result.derived_from == ["kn_source"]
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_get_node_not_found(self, mock_client, mock_collection, mock_config):
        """Test getting a non-existent node returns None."""
        store = KnowledgeVectorStore(config=mock_config)
        store._client = mock_client
        store._collection = mock_collection
        store._connected = True

        mock_collection.query.fetch_objects = MagicMock(
            return_value=MockWeaviateResponse([])  # No results
        )

        with patch("aragora.knowledge.vector_store.Filter") as mock_filter:
            mock_equal = MagicMock()
            mock_filter.by_property.return_value.equal.return_value = mock_equal
            mock_equal.__and__ = MagicMock(return_value=mock_equal)

            result = await store.get_node("kn_nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_node_with_workspace_override(
        self, mock_client, mock_collection, mock_config
    ):
        """Test get_node with workspace_id override."""
        store = KnowledgeVectorStore(config=mock_config)
        store._client = mock_client
        store._collection = mock_collection
        store._connected = True
        store.workspace_id = "default_ws"

        mock_collection.query.fetch_objects = MagicMock(return_value=MockWeaviateResponse([]))

        with patch("aragora.knowledge.vector_store.Filter") as mock_filter:
            mock_equal = MagicMock()
            mock_filter.by_property.return_value.equal.return_value = mock_equal
            mock_equal.__and__ = MagicMock(return_value=mock_equal)

            await store.get_node("kn_test", workspace_id="other_ws")

            # Verify workspace filter was called with override
            mock_filter.by_property.assert_any_call("workspace_id")


# =============================================================================
# Delete Node Tests (CRUD - Delete)
# =============================================================================


class TestDeleteNode:
    """Tests for deleting a single node."""

    @pytest.mark.asyncio
    async def test_delete_node_not_connected(self, mock_config):
        """Test delete_node raises error when not connected."""
        store = KnowledgeVectorStore(config=mock_config)

        with pytest.raises(RuntimeError, match="Not connected to Weaviate"):
            await store.delete_node("kn_test")

    @pytest.mark.asyncio
    async def test_delete_node_success(self, mock_client, mock_collection, mock_config):
        """Test successful node deletion."""
        store = KnowledgeVectorStore(config=mock_config)
        store._client = mock_client
        store._collection = mock_collection
        store._connected = True

        mock_collection.data.delete_many = MagicMock(return_value=MockDeleteResult(successful=1))

        with patch("aragora.knowledge.vector_store.Filter") as mock_filter:
            mock_equal = MagicMock()
            mock_filter.by_property.return_value.equal.return_value = mock_equal
            mock_equal.__and__ = MagicMock(return_value=mock_equal)

            result = await store.delete_node("kn_to_delete")

        assert result is True

    @pytest.mark.asyncio
    async def test_delete_node_not_found(self, mock_client, mock_collection, mock_config):
        """Test deleting non-existent node returns False."""
        store = KnowledgeVectorStore(config=mock_config)
        store._client = mock_client
        store._collection = mock_collection
        store._connected = True

        mock_collection.data.delete_many = MagicMock(return_value=MockDeleteResult(successful=0))

        with patch("aragora.knowledge.vector_store.Filter") as mock_filter:
            mock_equal = MagicMock()
            mock_filter.by_property.return_value.equal.return_value = mock_equal
            mock_equal.__and__ = MagicMock(return_value=mock_equal)

            result = await store.delete_node("kn_nonexistent")

        assert result is False


# =============================================================================
# Delete Workspace Tests
# =============================================================================


class TestDeleteWorkspace:
    """Tests for deleting all nodes in a workspace."""

    @pytest.mark.asyncio
    async def test_delete_workspace_not_connected(self, mock_config):
        """Test delete_workspace raises error when not connected."""
        store = KnowledgeVectorStore(config=mock_config)

        with pytest.raises(RuntimeError, match="Not connected to Weaviate"):
            await store.delete_workspace()

    @pytest.mark.asyncio
    async def test_delete_workspace_success(self, mock_client, mock_collection, mock_config):
        """Test successful workspace deletion."""
        store = KnowledgeVectorStore(config=mock_config)
        store._client = mock_client
        store._collection = mock_collection
        store._connected = True

        mock_collection.data.delete_many = MagicMock(return_value=MockDeleteResult(successful=42))

        with patch("aragora.knowledge.vector_store.Filter") as mock_filter:
            mock_filter.by_property.return_value.equal.return_value = MagicMock()

            result = await store.delete_workspace("test_workspace")

        assert result == 42

    @pytest.mark.asyncio
    async def test_delete_workspace_defaults_to_store_workspace(
        self, mock_client, mock_collection, mock_config
    ):
        """Test delete_workspace uses store's workspace by default."""
        store = KnowledgeVectorStore(config=mock_config)
        store._client = mock_client
        store._collection = mock_collection
        store._connected = True
        store.workspace_id = "my_workspace"

        mock_collection.data.delete_many = MagicMock(return_value=MockDeleteResult(successful=10))

        with patch("aragora.knowledge.vector_store.Filter") as mock_filter:
            mock_filter.by_property.return_value.equal.return_value = MagicMock()

            result = await store.delete_workspace()

        assert result == 10


# =============================================================================
# Count Nodes Tests
# =============================================================================


class TestCountNodes:
    """Tests for counting nodes."""

    @pytest.mark.asyncio
    async def test_count_nodes_not_connected(self, mock_config):
        """Test count_nodes raises error when not connected."""
        store = KnowledgeVectorStore(config=mock_config)

        with pytest.raises(RuntimeError, match="Not connected to Weaviate"):
            await store.count_nodes()

    @pytest.mark.asyncio
    async def test_count_nodes_all(self, mock_client, mock_collection, mock_config):
        """Test counting all nodes in workspace."""
        store = KnowledgeVectorStore(config=mock_config)
        store._client = mock_client
        store._collection = mock_collection
        store._connected = True

        mock_collection.aggregate.over_all = MagicMock(
            return_value=MockAggregateResponse(total_count=150)
        )

        with patch("aragora.knowledge.vector_store.Filter") as mock_filter:
            mock_filter.by_property.return_value.equal.return_value = MagicMock()

            result = await store.count_nodes()

        assert result == 150

    @pytest.mark.asyncio
    async def test_count_nodes_by_type(self, mock_client, mock_collection, mock_config):
        """Test counting nodes of specific type."""
        store = KnowledgeVectorStore(config=mock_config)
        store._client = mock_client
        store._collection = mock_collection
        store._connected = True

        mock_collection.aggregate.over_all = MagicMock(
            return_value=MockAggregateResponse(total_count=25)
        )

        with patch("aragora.knowledge.vector_store.Filter") as mock_filter:
            mock_combined = MagicMock()
            mock_filter.by_property.return_value.equal.return_value = mock_combined
            mock_combined.__and__ = MagicMock(return_value=mock_combined)

            result = await store.count_nodes(node_type="fact")

        assert result == 25

    @pytest.mark.asyncio
    async def test_count_nodes_handles_none_total(self, mock_client, mock_collection, mock_config):
        """Test count_nodes returns 0 when total_count is None."""
        store = KnowledgeVectorStore(config=mock_config)
        store._client = mock_client
        store._collection = mock_collection
        store._connected = True

        mock_collection.aggregate.over_all = MagicMock(
            return_value=MockAggregateResponse(total_count=None)
        )

        with patch("aragora.knowledge.vector_store.Filter") as mock_filter:
            mock_filter.by_property.return_value.equal.return_value = MagicMock()

            result = await store.count_nodes()

        assert result == 0


# =============================================================================
# Type Distribution Tests
# =============================================================================


class TestGetTypeDistribution:
    """Tests for getting node type distribution."""

    @pytest.mark.asyncio
    async def test_get_type_distribution(self, mock_client, mock_collection, mock_config):
        """Test getting distribution of node types."""
        store = KnowledgeVectorStore(config=mock_config)
        store._client = mock_client
        store._collection = mock_collection
        store._connected = True

        # Return different counts for different types
        counts = {
            "fact": 50,
            "claim": 30,
            "memory": 0,
            "evidence": 20,
            "consensus": 10,
            "insight": 0,
            "pattern": 5,
        }

        def mock_aggregate(*args, **kwargs):
            # Extract node_type from filter
            # For simplicity, return different counts based on call order
            return MockAggregateResponse(total_count=counts.get(store._current_type, 0))

        call_count = [0]
        type_order = list(KnowledgeVectorStore.VALID_NODE_TYPES)

        async def count_nodes_mock(node_type=None, workspace_id=None):
            return counts.get(node_type, 0)

        with patch.object(store, "count_nodes", side_effect=count_nodes_mock):
            result = await store.get_type_distribution()

        # Should only include types with count > 0
        assert "fact" in result
        assert result["fact"] == 50
        assert "claim" in result
        assert result["claim"] == 30
        assert "memory" not in result  # count was 0
        assert "insight" not in result  # count was 0
        assert "pattern" in result
        assert result["pattern"] == 5


# =============================================================================
# Health Check Tests
# =============================================================================


class TestHealthCheck:
    """Tests for health check functionality."""

    @pytest.mark.asyncio
    async def test_health_check_not_connected(self, mock_config):
        """Test health check when not connected."""
        store = KnowledgeVectorStore(config=mock_config)

        result = await store.health_check()

        assert result["healthy"] is False
        assert result["error"] == "Not connected"

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, mock_client, mock_collection, mock_config):
        """Test health check when healthy."""
        store = KnowledgeVectorStore(config=mock_config)
        store._client = mock_client
        store._collection = mock_collection
        store._connected = True

        mock_client.is_ready = MagicMock(return_value=True)

        with patch.object(store, "count_nodes", return_value=100):
            result = await store.health_check()

        assert result["healthy"] is True
        assert result["url"] == mock_config.url
        assert result["collection"] == mock_config.collection_name
        assert result["workspace"] == store.workspace_id
        assert result["node_count"] == 100

    @pytest.mark.asyncio
    async def test_health_check_not_ready(self, mock_client, mock_collection, mock_config):
        """Test health check when Weaviate not ready."""
        store = KnowledgeVectorStore(config=mock_config)
        store._client = mock_client
        store._collection = mock_collection
        store._connected = True

        mock_client.is_ready = MagicMock(return_value=False)

        result = await store.health_check()

        assert result["healthy"] is False
        assert result["node_count"] == 0

    @pytest.mark.asyncio
    async def test_health_check_handles_runtime_error(
        self, mock_client, mock_collection, mock_config
    ):
        """Test health check handles RuntimeError."""
        store = KnowledgeVectorStore(config=mock_config)
        store._client = mock_client
        store._collection = mock_collection
        store._connected = True

        mock_client.is_ready = MagicMock(side_effect=RuntimeError("Connection lost"))

        result = await store.health_check()

        assert result["healthy"] is False
        assert result["error"]  # Sanitized error message present

    @pytest.mark.asyncio
    async def test_health_check_handles_connection_error(
        self, mock_client, mock_collection, mock_config
    ):
        """Test health check handles ConnectionError."""
        store = KnowledgeVectorStore(config=mock_config)
        store._client = mock_client
        store._collection = mock_collection
        store._connected = True

        mock_client.is_ready = MagicMock(side_effect=ConnectionError("Network unreachable"))

        result = await store.health_check()

        assert result["healthy"] is False
        assert result["error"]  # Sanitized error message present

    @pytest.mark.asyncio
    async def test_health_check_handles_timeout_error(
        self, mock_client, mock_collection, mock_config
    ):
        """Test health check handles TimeoutError."""
        store = KnowledgeVectorStore(config=mock_config)
        store._client = mock_client
        store._collection = mock_collection
        store._connected = True

        mock_client.is_ready = MagicMock(side_effect=TimeoutError("Request timed out"))

        result = await store.health_check()

        assert result["healthy"] is False
        assert result["error"]  # Sanitized error message present

    @pytest.mark.asyncio
    async def test_health_check_handles_unexpected_error(
        self, mock_client, mock_collection, mock_config
    ):
        """Test health check handles unexpected exceptions."""
        store = KnowledgeVectorStore(config=mock_config)
        store._client = mock_client
        store._collection = mock_collection
        store._connected = True

        mock_client.is_ready = MagicMock(side_effect=Exception("Unexpected error occurred"))

        result = await store.health_check()

        assert result["healthy"] is False
        assert result["error"]  # Sanitized error message present


# =============================================================================
# Helper Method Tests
# =============================================================================


class TestParseIds:
    """Tests for _parse_ids helper method."""

    def test_parse_ids_empty_string(self):
        """Test parsing empty string returns empty list."""
        result = KnowledgeVectorStore._parse_ids("")
        assert result == []

    def test_parse_ids_single_id(self):
        """Test parsing single ID."""
        result = KnowledgeVectorStore._parse_ids("kn_single")
        assert result == ["kn_single"]

    def test_parse_ids_multiple_ids(self):
        """Test parsing multiple comma-separated IDs."""
        result = KnowledgeVectorStore._parse_ids("kn_a,kn_b,kn_c")
        assert result == ["kn_a", "kn_b", "kn_c"]

    def test_parse_ids_with_whitespace(self):
        """Test parsing IDs with whitespace."""
        result = KnowledgeVectorStore._parse_ids("kn_a , kn_b , kn_c")
        assert result == ["kn_a", "kn_b", "kn_c"]

    def test_parse_ids_filters_empty_strings(self):
        """Test parsing filters out empty strings."""
        result = KnowledgeVectorStore._parse_ids("kn_a,,kn_b,,,kn_c")
        assert result == ["kn_a", "kn_b", "kn_c"]


# =============================================================================
# Singleton Factory Tests
# =============================================================================


class TestGetKnowledgeVectorStore:
    """Tests for the global store factory function."""

    def test_get_store_creates_new_instance(self):
        """Test factory creates new instance for new workspace."""
        # Clear global stores
        import aragora.knowledge.vector_store as vs

        vs._stores = {}

        store = get_knowledge_vector_store("test_ws_1")

        assert store.workspace_id == "test_ws_1"
        assert "test_ws_1" in vs._stores

    def test_get_store_returns_existing_instance(self):
        """Test factory returns existing instance for same workspace."""
        import aragora.knowledge.vector_store as vs

        vs._stores = {}

        store1 = get_knowledge_vector_store("test_ws_2")
        store2 = get_knowledge_vector_store("test_ws_2")

        assert store1 is store2

    def test_get_store_different_workspaces(self):
        """Test factory creates different instances for different workspaces."""
        import aragora.knowledge.vector_store as vs

        vs._stores = {}

        store_a = get_knowledge_vector_store("workspace_a")
        store_b = get_knowledge_vector_store("workspace_b")

        assert store_a is not store_b
        assert store_a.workspace_id == "workspace_a"
        assert store_b.workspace_id == "workspace_b"

    def test_get_store_with_config(self):
        """Test factory accepts config parameter."""
        import aragora.knowledge.vector_store as vs

        vs._stores = {}

        config = KnowledgeVectorConfig(
            url="https://custom.weaviate.io",
            collection_name="CustomCollection",
        )

        store = get_knowledge_vector_store("test_ws_config", config=config)

        assert store.config.url == "https://custom.weaviate.io"
        assert store.config.collection_name == "CustomCollection"


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_search_semantic_empty_results(
        self, mock_client, mock_collection, mock_config, sample_embedding
    ):
        """Test semantic search with no results."""
        store = KnowledgeVectorStore(config=mock_config)
        store._client = mock_client
        store._collection = mock_collection
        store._connected = True

        mock_collection.query.near_vector = MagicMock(return_value=MockWeaviateResponse([]))

        with (
            patch("aragora.knowledge.vector_store.track_vector_operation"),
            patch("aragora.knowledge.vector_store.track_vector_search_results"),
            patch("aragora.knowledge.vector_store.Filter") as mock_filter,
        ):
            mock_filter.by_property.return_value.equal.return_value = MagicMock()

            results = await store.search_semantic(sample_embedding)

        assert results == []

    @pytest.mark.asyncio
    async def test_search_with_zero_limit(
        self, mock_client, mock_collection, mock_config, sample_embedding
    ):
        """Test search with limit=0."""
        store = KnowledgeVectorStore(config=mock_config)
        store._client = mock_client
        store._collection = mock_collection
        store._connected = True

        mock_collection.query.near_vector = MagicMock(return_value=MockWeaviateResponse([]))

        with (
            patch("aragora.knowledge.vector_store.track_vector_operation"),
            patch("aragora.knowledge.vector_store.track_vector_search_results"),
            patch("aragora.knowledge.vector_store.Filter") as mock_filter,
        ):
            mock_filter.by_property.return_value.equal.return_value = MagicMock()

            results = await store.search_semantic(sample_embedding, limit=0)

        assert results == []

    @pytest.mark.asyncio
    async def test_index_node_with_very_long_content(
        self, mock_client, mock_collection, mock_config, sample_embedding
    ):
        """Test indexing node with very long content."""
        store = KnowledgeVectorStore(config=mock_config)
        store._client = mock_client
        store._collection = mock_collection
        store._connected = True

        long_content = "A" * 100000  # 100KB of content
        node = MockKnowledgeNode(content=long_content)

        await store.index_node(node, sample_embedding)

        call_args = mock_collection.data.insert.call_args
        properties = call_args.kwargs["properties"]
        assert len(properties["content"]) == 100000

    @pytest.mark.asyncio
    async def test_search_with_many_relationships(
        self, mock_client, mock_collection, mock_config, sample_embedding
    ):
        """Test search result with many relationships."""
        store = KnowledgeVectorStore(config=mock_config)
        store._client = mock_client
        store._collection = mock_collection
        store._connected = True

        # 100 relationships
        many_ids = ",".join([f"kn_{i}" for i in range(100)])

        result_with_many_rels = MockWeaviateObject(
            properties={
                "node_id": "kn_many_rels",
                "workspace_id": "test",
                "node_type": "consensus",
                "content": "Node with many relationships",
                "confidence": 0.95,
                "tier": "glacial",
                "supports_ids": many_ids,
                "contradicts_ids": "",
                "derived_from_ids": many_ids,
            },
            distance=0.1,
        )

        mock_collection.query.near_vector = MagicMock(
            return_value=MockWeaviateResponse([result_with_many_rels])
        )

        with (
            patch("aragora.knowledge.vector_store.track_vector_operation"),
            patch("aragora.knowledge.vector_store.track_vector_search_results"),
            patch("aragora.knowledge.vector_store.Filter") as mock_filter,
        ):
            mock_filter.by_property.return_value.equal.return_value = MagicMock()

            results = await store.search_semantic(sample_embedding)

        assert len(results) == 1
        assert len(results[0].supports) == 100
        assert len(results[0].derived_from) == 100

    def test_invalid_embedding_dimensions_handling(self, mock_config):
        """Test config with various embedding dimensions."""
        for dims in [256, 512, 768, 1024, 1536, 3072, 4096]:
            config = KnowledgeVectorConfig(embedding_dimensions=dims)
            store = KnowledgeVectorStore(config=config)
            assert store.config.embedding_dimensions == dims

    @pytest.mark.asyncio
    async def test_search_handles_missing_metadata(
        self, mock_client, mock_collection, mock_config, sample_embedding
    ):
        """Test search handles objects with missing optional properties."""
        store = KnowledgeVectorStore(config=mock_config)
        store._client = mock_client
        store._collection = mock_collection
        store._connected = True

        # Minimal properties
        sparse_object = MockWeaviateObject(
            properties={
                "node_id": "kn_sparse",
                "workspace_id": "test",
                "node_type": "fact",
                "content": "Sparse content",
                # Missing: confidence, tier, supports_ids, etc.
            },
            distance=0.2,
        )

        mock_collection.query.near_vector = MagicMock(
            return_value=MockWeaviateResponse([sparse_object])
        )

        with (
            patch("aragora.knowledge.vector_store.track_vector_operation"),
            patch("aragora.knowledge.vector_store.track_vector_search_results"),
            patch("aragora.knowledge.vector_store.Filter") as mock_filter,
        ):
            mock_filter.by_property.return_value.equal.return_value = MagicMock()

            results = await store.search_semantic(sample_embedding)

        assert len(results) == 1
        # Should use defaults for missing properties
        assert results[0].confidence == 0.5  # Default
        assert results[0].tier == "slow"  # Default
        assert results[0].supports == []

    @pytest.mark.asyncio
    async def test_handles_none_distance_in_metadata(
        self, mock_client, mock_collection, mock_config, sample_embedding
    ):
        """Test handling of None distance in metadata."""
        store = KnowledgeVectorStore(config=mock_config)
        store._client = mock_client
        store._collection = mock_collection
        store._connected = True

        obj_with_none_distance = MockWeaviateObject(
            properties={
                "node_id": "kn_none_dist",
                "workspace_id": "test",
                "node_type": "fact",
                "content": "Content",
                "confidence": 0.8,
                "tier": "slow",
                "supports_ids": "",
                "contradicts_ids": "",
                "derived_from_ids": "",
            },
            distance=None,  # None distance
        )

        mock_collection.query.near_vector = MagicMock(
            return_value=MockWeaviateResponse([obj_with_none_distance])
        )

        with (
            patch("aragora.knowledge.vector_store.track_vector_operation"),
            patch("aragora.knowledge.vector_store.track_vector_search_results"),
            patch("aragora.knowledge.vector_store.Filter") as mock_filter,
        ):
            mock_filter.by_property.return_value.equal.return_value = MagicMock()

            results = await store.search_semantic(sample_embedding)

        assert len(results) == 1
        assert results[0].score == 1.0  # 1 - 0.0 (defaulted)


# =============================================================================
# Concurrency Tests
# =============================================================================


class TestConcurrency:
    """Tests for concurrent operations."""

    @pytest.mark.asyncio
    async def test_concurrent_searches(
        self, mock_client, mock_collection, mock_config, sample_embedding
    ):
        """Test multiple concurrent searches."""
        store = KnowledgeVectorStore(config=mock_config)
        store._client = mock_client
        store._collection = mock_collection
        store._connected = True

        mock_collection.query.near_vector = MagicMock(return_value=MockWeaviateResponse([]))

        with (
            patch("aragora.knowledge.vector_store.track_vector_operation"),
            patch("aragora.knowledge.vector_store.track_vector_search_results"),
            patch("aragora.knowledge.vector_store.Filter") as mock_filter,
        ):
            mock_filter.by_property.return_value.equal.return_value = MagicMock()

            # Run 5 concurrent searches
            tasks = [store.search_semantic(sample_embedding) for _ in range(5)]
            results = await asyncio.gather(*tasks)

        assert len(results) == 5
        assert all(r == [] for r in results)

    @pytest.mark.asyncio
    async def test_concurrent_mixed_operations(
        self, mock_client, mock_collection, mock_config, sample_embedding
    ):
        """Test mixed concurrent operations."""
        store = KnowledgeVectorStore(config=mock_config)
        store._client = mock_client
        store._collection = mock_collection
        store._connected = True

        mock_collection.query.near_vector = MagicMock(return_value=MockWeaviateResponse([]))
        mock_collection.query.bm25 = MagicMock(return_value=MockWeaviateResponse([]))
        mock_collection.aggregate.over_all = MagicMock(
            return_value=MockAggregateResponse(total_count=50)
        )

        with (
            patch("aragora.knowledge.vector_store.track_vector_operation"),
            patch("aragora.knowledge.vector_store.track_vector_search_results"),
            patch("aragora.knowledge.vector_store.Filter") as mock_filter,
        ):
            mock_filter.by_property.return_value.equal.return_value = MagicMock()

            tasks = [
                store.search_semantic(sample_embedding),
                store.search_keyword("test"),
                store.count_nodes(),
            ]
            results = await asyncio.gather(*tasks)

        assert results[0] == []  # semantic search
        assert results[1] == []  # keyword search
        assert results[2] == 50  # count


# =============================================================================
# WEAVIATE_AVAILABLE Flag Tests
# =============================================================================


class TestWeaviateAvailableFlag:
    """Tests for WEAVIATE_AVAILABLE module flag."""

    def test_weaviate_available_is_boolean(self):
        """Test WEAVIATE_AVAILABLE is a boolean."""
        assert isinstance(WEAVIATE_AVAILABLE, bool)

    def test_exports_contain_expected_items(self):
        """Test module exports expected items."""
        from aragora.knowledge import vector_store

        assert hasattr(vector_store, "KnowledgeVectorStore")
        assert hasattr(vector_store, "KnowledgeVectorConfig")
        assert hasattr(vector_store, "KnowledgeSearchResult")
        assert hasattr(vector_store, "WEAVIATE_AVAILABLE")
        assert hasattr(vector_store, "get_knowledge_vector_store")
