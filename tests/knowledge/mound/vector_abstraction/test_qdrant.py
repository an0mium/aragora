"""
Tests for Qdrant vector store adapter.

Covers connection management, CRUD operations, vector similarity search,
filter queries, error handling, and edge cases using mocked qdrant-client.
"""

import pytest
import uuid
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from aragora.knowledge.mound.vector_abstraction.base import (
    VectorBackend,
    VectorSearchResult,
    VectorStoreConfig,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config():
    """Create test configuration for Qdrant."""
    return VectorStoreConfig(
        backend=VectorBackend.QDRANT,
        url="http://localhost:6333",
        collection_name="test_collection",
        embedding_dimensions=1536,
        distance_metric="cosine",
        timeout_seconds=10,
    )


@pytest.fixture
def config_with_api_key():
    """Create test configuration with API key."""
    return VectorStoreConfig(
        backend=VectorBackend.QDRANT,
        url="http://localhost:6333",
        api_key="test-api-key",
        collection_name="test_collection",
        embedding_dimensions=1536,
    )


@pytest.fixture
def mock_qdrant_env():
    """
    Patch Qdrant availability flag and all Qdrant-specific symbols so that
    QdrantVectorStore can be instantiated without the real qdrant-client library.
    Yields (mock_async_client_class, mock_client_instance) for further
    customisation inside tests.
    """
    mock_async_cls = MagicMock()
    mock_client_instance = AsyncMock()
    mock_async_cls.return_value = mock_client_instance

    # Create a lightweight PointStruct that stores args as attributes
    class _PointStruct:
        def __init__(self, id, vector, payload):
            self.id = id
            self.vector = vector
            self.payload = payload

    # Mock Distance enum
    mock_distance = MagicMock()
    mock_distance.COSINE = "Cosine"
    mock_distance.EUCLID = "Euclid"
    mock_distance.DOT = "Dot"

    # Mock qdrant_models
    mock_qdrant_models = MagicMock()

    # Mock Filter that stores conditions
    class _Filter:
        def __init__(self, must=None):
            self.must = must or []

    _mod = "aragora.knowledge.mound.vector_abstraction.qdrant"

    with (
        patch(f"{_mod}.QDRANT_AVAILABLE", True),
        patch(f"{_mod}.AsyncQdrantClient", mock_async_cls, create=True),
        patch(f"{_mod}.PointStruct", _PointStruct, create=True),
        patch(f"{_mod}.VectorParams", MagicMock(), create=True),
        patch(f"{_mod}.Distance", mock_distance, create=True),
        patch(f"{_mod}.Filter", _Filter, create=True),
        patch(f"{_mod}.FieldCondition", MagicMock, create=True),
        patch(f"{_mod}.MatchValue", MagicMock, create=True),
        patch(f"{_mod}.qdrant_models", mock_qdrant_models, create=True),
    ):
        yield mock_async_cls, mock_client_instance


def _make_store(config, mock_client_instance):
    """Helper: create a QdrantVectorStore with pre-wired mock client."""
    from aragora.knowledge.mound.vector_abstraction.qdrant import QdrantVectorStore

    store = QdrantVectorStore(config)
    store._client = mock_client_instance
    store._connected = True
    return store


def _embedding(dim=1536, val=0.1):
    """Helper: create a dummy embedding vector."""
    return [val] * dim


def _named_mock(name_val):
    """Create a MagicMock with a .name attribute (MagicMock's `name` kwarg is reserved)."""
    m = MagicMock()
    m.name = name_val
    return m


# ---------------------------------------------------------------------------
# Connection Management
# ---------------------------------------------------------------------------


class TestQdrantConnection:
    """Tests for Qdrant connection lifecycle."""

    @pytest.mark.asyncio
    async def test_connect_creates_client(self, config, mock_qdrant_env):
        """connect() should create an AsyncQdrantClient and mark as connected."""
        mock_cls, mock_client = mock_qdrant_env

        # collection_exists returns True so it skips create_collection
        mock_collections = MagicMock()
        mock_collections.collections = [_named_mock("test_collection")]
        mock_client.get_collections.return_value = mock_collections

        from aragora.knowledge.mound.vector_abstraction.qdrant import QdrantVectorStore

        store = QdrantVectorStore(config)
        await store.connect()

        assert store.is_connected
        mock_cls.assert_called_once_with(
            url="http://localhost:6333",
            timeout=10,
        )

    @pytest.mark.asyncio
    async def test_connect_with_api_key(self, config_with_api_key, mock_qdrant_env):
        """connect() should pass api_key when configured."""
        mock_cls, mock_client = mock_qdrant_env

        mock_collections = MagicMock()
        mock_collections.collections = [_named_mock("test_collection")]
        mock_client.get_collections.return_value = mock_collections

        from aragora.knowledge.mound.vector_abstraction.qdrant import QdrantVectorStore

        store = QdrantVectorStore(config_with_api_key)
        await store.connect()

        assert store.is_connected
        mock_cls.assert_called_once_with(
            url="http://localhost:6333",
            api_key="test-api-key",
            timeout=30,
        )

    @pytest.mark.asyncio
    async def test_connect_already_connected_is_noop(self, config, mock_qdrant_env):
        """Calling connect() when already connected should be a no-op."""
        _, mock_client = mock_qdrant_env
        store = _make_store(config, mock_client)

        await store.connect()

        # The client constructor should NOT be called again
        # (store was pre-connected by _make_store)

    @pytest.mark.asyncio
    async def test_connect_creates_collection_if_missing(self, config, mock_qdrant_env):
        """connect() should create the default collection when it does not exist."""
        mock_cls, mock_client = mock_qdrant_env

        # First call: collection_exists check in connect -> no collections
        # Second call: create_collection -> still no collections (before creation)
        mock_empty = MagicMock()
        mock_empty.collections = []
        mock_client.get_collections.return_value = mock_empty

        from aragora.knowledge.mound.vector_abstraction.qdrant import QdrantVectorStore

        store = QdrantVectorStore(config)
        await store.connect()

        assert store.is_connected
        mock_client.create_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_failure_raises_connection_error(self, config, mock_qdrant_env):
        """connect() should raise ConnectionError on failure."""
        mock_cls, mock_client = mock_qdrant_env
        mock_cls.side_effect = ConnectionError("refused")

        from aragora.knowledge.mound.vector_abstraction.qdrant import QdrantVectorStore

        store = QdrantVectorStore(config)
        with pytest.raises(ConnectionError, match="Failed to connect to Qdrant"):
            await store.connect()

        assert not store.is_connected

    @pytest.mark.asyncio
    async def test_connect_timeout_raises_connection_error(self, config, mock_qdrant_env):
        """connect() should raise ConnectionError on timeout."""
        mock_cls, mock_client = mock_qdrant_env
        mock_cls.side_effect = TimeoutError("timed out")

        from aragora.knowledge.mound.vector_abstraction.qdrant import QdrantVectorStore

        store = QdrantVectorStore(config)
        with pytest.raises(ConnectionError, match="Failed to connect to Qdrant"):
            await store.connect()

    @pytest.mark.asyncio
    async def test_disconnect(self, config, mock_qdrant_env):
        """disconnect() should close client and reset state."""
        _, mock_client = mock_qdrant_env
        store = _make_store(config, mock_client)

        await store.disconnect()

        mock_client.close.assert_awaited_once()
        assert not store.is_connected
        assert store._client is None

    @pytest.mark.asyncio
    async def test_disconnect_handles_close_error(self, config, mock_qdrant_env):
        """disconnect() should handle errors during close gracefully."""
        _, mock_client = mock_qdrant_env
        mock_client.close.side_effect = RuntimeError("close failed")
        store = _make_store(config, mock_client)

        await store.disconnect()

        assert not store.is_connected
        assert store._client is None

    @pytest.mark.asyncio
    async def test_disconnect_when_not_connected(self, config, mock_qdrant_env):
        """disconnect() with no client should be a no-op."""
        from aragora.knowledge.mound.vector_abstraction.qdrant import QdrantVectorStore

        store = QdrantVectorStore(config)
        await store.disconnect()  # Should not raise

    @pytest.mark.asyncio
    async def test_context_manager(self, config, mock_qdrant_env):
        """Async context manager should connect and disconnect."""
        mock_cls, mock_client = mock_qdrant_env

        mock_collections = MagicMock()
        mock_collections.collections = [_named_mock("test_collection")]
        mock_client.get_collections.return_value = mock_collections

        from aragora.knowledge.mound.vector_abstraction.qdrant import QdrantVectorStore

        async with QdrantVectorStore(config) as store:
            assert store.is_connected

        mock_client.close.assert_awaited_once()


# ---------------------------------------------------------------------------
# Collection Management
# ---------------------------------------------------------------------------


class TestQdrantCollections:
    """Tests for collection CRUD operations."""

    @pytest.mark.asyncio
    async def test_create_collection(self, config, mock_qdrant_env):
        """create_collection() should create with correct vector params."""
        _, mock_client = mock_qdrant_env
        store = _make_store(config, mock_client)

        mock_empty = MagicMock()
        mock_empty.collections = []
        mock_client.get_collections.return_value = mock_empty

        await store.create_collection("my_collection")

        mock_client.create_collection.assert_awaited_once()
        call_kwargs = mock_client.create_collection.call_args
        assert call_kwargs.kwargs["collection_name"] == "my_collection"

    @pytest.mark.asyncio
    async def test_create_collection_already_exists(self, config, mock_qdrant_env):
        """create_collection() should skip if collection already exists."""
        _, mock_client = mock_qdrant_env
        store = _make_store(config, mock_client)

        mock_existing = MagicMock()
        mock_existing.collections = [_named_mock("my_collection")]
        mock_client.get_collections.return_value = mock_existing

        await store.create_collection("my_collection")

        mock_client.create_collection.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_create_collection_not_connected(self, config, mock_qdrant_env):
        """create_collection() should raise ConnectionError if not connected."""
        from aragora.knowledge.mound.vector_abstraction.qdrant import QdrantVectorStore

        store = QdrantVectorStore(config)

        with pytest.raises(ConnectionError, match="Not connected"):
            await store.create_collection("test")

    @pytest.mark.asyncio
    async def test_delete_collection_success(self, config, mock_qdrant_env):
        """delete_collection() should return True on success."""
        _, mock_client = mock_qdrant_env
        store = _make_store(config, mock_client)

        result = await store.delete_collection("my_collection")

        assert result is True
        mock_client.delete_collection.assert_awaited_once_with("my_collection")

    @pytest.mark.asyncio
    async def test_delete_collection_failure(self, config, mock_qdrant_env):
        """delete_collection() should return False on failure."""
        _, mock_client = mock_qdrant_env
        mock_client.delete_collection.side_effect = RuntimeError("not found")
        store = _make_store(config, mock_client)

        result = await store.delete_collection("missing")

        assert result is False

    @pytest.mark.asyncio
    async def test_collection_exists_true(self, config, mock_qdrant_env):
        """collection_exists() should return True when collection present."""
        _, mock_client = mock_qdrant_env
        store = _make_store(config, mock_client)

        mock_resp = MagicMock()
        mock_resp.collections = [_named_mock("test_collection")]
        mock_client.get_collections.return_value = mock_resp

        assert await store.collection_exists("test_collection") is True

    @pytest.mark.asyncio
    async def test_collection_exists_false(self, config, mock_qdrant_env):
        """collection_exists() should return False when collection absent."""
        _, mock_client = mock_qdrant_env
        store = _make_store(config, mock_client)

        mock_resp = MagicMock()
        mock_resp.collections = []
        mock_client.get_collections.return_value = mock_resp

        assert await store.collection_exists("missing") is False

    @pytest.mark.asyncio
    async def test_list_collections(self, config, mock_qdrant_env):
        """list_collections() should return collection names."""
        _, mock_client = mock_qdrant_env
        store = _make_store(config, mock_client)

        mock_resp = MagicMock()
        mock_resp.collections = [_named_mock("a"), _named_mock("b")]
        mock_client.get_collections.return_value = mock_resp

        names = await store.list_collections()
        assert names == ["a", "b"]

    @pytest.mark.asyncio
    async def test_list_collections_not_connected(self, config, mock_qdrant_env):
        """list_collections() should raise when not connected."""
        from aragora.knowledge.mound.vector_abstraction.qdrant import QdrantVectorStore

        store = QdrantVectorStore(config)
        with pytest.raises(ConnectionError):
            await store.list_collections()


# ---------------------------------------------------------------------------
# CRUD Operations
# ---------------------------------------------------------------------------


class TestQdrantUpsert:
    """Tests for insert/update operations."""

    @pytest.mark.asyncio
    async def test_upsert_single(self, config, mock_qdrant_env):
        """upsert() should insert a single vector and return the id."""
        _, mock_client = mock_qdrant_env
        store = _make_store(config, mock_client)

        result = await store.upsert(
            id="vec-1",
            embedding=_embedding(),
            content="test content",
            metadata={"topic": "science"},
        )

        assert result == "vec-1"
        mock_client.upsert.assert_awaited_once()
        call_kwargs = mock_client.upsert.call_args
        assert call_kwargs.kwargs["collection_name"] == "test_collection"

    @pytest.mark.asyncio
    async def test_upsert_with_namespace(self, config, mock_qdrant_env):
        """upsert() should include namespace in payload."""
        _, mock_client = mock_qdrant_env
        store = _make_store(config, mock_client)

        await store.upsert(
            id="vec-2",
            embedding=_embedding(),
            content="namespaced content",
            namespace="tenant-a",
        )

        call_kwargs = mock_client.upsert.call_args
        points = call_kwargs.kwargs["points"]
        assert len(points) == 1
        assert points[0].payload["namespace"] == "tenant-a"

    @pytest.mark.asyncio
    async def test_upsert_without_metadata(self, config, mock_qdrant_env):
        """upsert() should work with no metadata."""
        _, mock_client = mock_qdrant_env
        store = _make_store(config, mock_client)

        result = await store.upsert(
            id="vec-3",
            embedding=_embedding(),
            content="bare content",
        )

        assert result == "vec-3"
        call_kwargs = mock_client.upsert.call_args
        payload = call_kwargs.kwargs["points"][0].payload
        assert payload["content"] == "bare content"
        assert payload["namespace"] == ""

    @pytest.mark.asyncio
    async def test_upsert_with_uuid_id(self, config, mock_qdrant_env):
        """upsert() should handle UUID-format IDs properly."""
        _, mock_client = mock_qdrant_env
        store = _make_store(config, mock_client)

        test_uuid = str(uuid.uuid4())
        result = await store.upsert(
            id=test_uuid,
            embedding=_embedding(),
            content="uuid content",
        )

        assert result == test_uuid

    @pytest.mark.asyncio
    async def test_upsert_not_connected(self, config, mock_qdrant_env):
        """upsert() should raise ConnectionError when not connected."""
        from aragora.knowledge.mound.vector_abstraction.qdrant import QdrantVectorStore

        store = QdrantVectorStore(config)
        with pytest.raises(ConnectionError, match="Not connected"):
            await store.upsert(id="x", embedding=_embedding(), content="c")

    @pytest.mark.asyncio
    async def test_upsert_batch(self, config, mock_qdrant_env):
        """upsert_batch() should insert multiple vectors."""
        _, mock_client = mock_qdrant_env
        store = _make_store(config, mock_client)

        items = [
            {"id": "b1", "embedding": _embedding(), "content": "first", "metadata": {"k": "v1"}},
            {"id": "b2", "embedding": _embedding(), "content": "second", "metadata": {"k": "v2"}},
            {"id": "b3", "embedding": _embedding(), "content": "third"},
        ]

        ids = await store.upsert_batch(items)

        assert ids == ["b1", "b2", "b3"]
        mock_client.upsert.assert_awaited_once()
        call_kwargs = mock_client.upsert.call_args
        points = call_kwargs.kwargs["points"]
        assert len(points) == 3

    @pytest.mark.asyncio
    async def test_upsert_batch_auto_generates_ids(self, config, mock_qdrant_env):
        """upsert_batch() should generate UUIDs for items without ids."""
        _, mock_client = mock_qdrant_env
        store = _make_store(config, mock_client)

        items = [
            {"embedding": _embedding(), "content": "no id item"},
        ]

        ids = await store.upsert_batch(items)

        assert len(ids) == 1
        # Should be a valid UUID4
        uuid.UUID(ids[0])

    @pytest.mark.asyncio
    async def test_upsert_batch_with_namespace(self, config, mock_qdrant_env):
        """upsert_batch() should set namespace in all payloads."""
        _, mock_client = mock_qdrant_env
        store = _make_store(config, mock_client)

        items = [
            {"id": "n1", "embedding": _embedding(), "content": "c1"},
            {"id": "n2", "embedding": _embedding(), "content": "c2"},
        ]

        await store.upsert_batch(items, namespace="tenant-x")

        points = mock_client.upsert.call_args.kwargs["points"]
        for point in points:
            assert point.payload["namespace"] == "tenant-x"


# ---------------------------------------------------------------------------
# Delete Operations
# ---------------------------------------------------------------------------


class TestQdrantDelete:
    """Tests for delete operations."""

    @pytest.mark.asyncio
    async def test_delete_by_ids(self, config, mock_qdrant_env):
        """delete() should delete vectors by ID and return count."""
        _, mock_client = mock_qdrant_env
        store = _make_store(config, mock_client)

        count = await store.delete(ids=["id-1", "id-2", "id-3"])

        assert count == 3
        mock_client.delete.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_delete_empty_ids(self, config, mock_qdrant_env):
        """delete() with empty list should return 0."""
        _, mock_client = mock_qdrant_env
        store = _make_store(config, mock_client)

        count = await store.delete(ids=[])

        assert count == 0

    @pytest.mark.asyncio
    async def test_delete_not_connected(self, config, mock_qdrant_env):
        """delete() should raise ConnectionError when not connected."""
        from aragora.knowledge.mound.vector_abstraction.qdrant import QdrantVectorStore

        store = QdrantVectorStore(config)
        with pytest.raises(ConnectionError):
            await store.delete(ids=["x"])

    @pytest.mark.asyncio
    async def test_delete_by_filter(self, config, mock_qdrant_env):
        """delete_by_filter() should delete matching vectors."""
        _, mock_client = mock_qdrant_env
        store = _make_store(config, mock_client)

        # Mock count to return 5 (before deletion)
        mock_count_result = MagicMock()
        mock_count_result.count = 5
        mock_client.count.return_value = mock_count_result

        count = await store.delete_by_filter(
            filters={"topic": "obsolete"},
            namespace="tenant-a",
        )

        assert count == 5
        mock_client.delete.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_delete_by_filter_not_connected(self, config, mock_qdrant_env):
        """delete_by_filter() should raise when not connected."""
        from aragora.knowledge.mound.vector_abstraction.qdrant import QdrantVectorStore

        store = QdrantVectorStore(config)
        with pytest.raises(ConnectionError):
            await store.delete_by_filter(filters={"k": "v"})


# ---------------------------------------------------------------------------
# Search Operations
# ---------------------------------------------------------------------------


class TestQdrantSearch:
    """Tests for vector similarity search."""

    def _make_point(self, id_val, score, payload, vector=None):
        """Create a mock search result point."""
        point = MagicMock()
        point.id = id_val
        point.score = score
        point.payload = payload
        point.vector = vector
        return point

    @pytest.mark.asyncio
    async def test_search_returns_results(self, config, mock_qdrant_env):
        """search() should return VectorSearchResult list."""
        _, mock_client = mock_qdrant_env
        store = _make_store(config, mock_client)

        mock_client.search.return_value = [
            self._make_point(
                "r1",
                0.95,
                {"content": "first result", "namespace": "ns", "topic": "ai"},
                _embedding(dim=4),
            ),
            self._make_point(
                "r2",
                0.80,
                {"content": "second result", "namespace": "ns"},
            ),
        ]

        results = await store.search(embedding=_embedding(), limit=10)

        assert len(results) == 2
        assert results[0].id == "r1"
        assert results[0].score == 0.95
        assert results[0].content == "first result"
        assert results[0].metadata == {"topic": "ai"}
        assert results[0].embedding == _embedding(dim=4)
        assert results[1].id == "r2"
        assert results[1].metadata == {}

    @pytest.mark.asyncio
    async def test_search_with_filters(self, config, mock_qdrant_env):
        """search() should pass filters to Qdrant."""
        _, mock_client = mock_qdrant_env
        store = _make_store(config, mock_client)
        mock_client.search.return_value = []

        await store.search(
            embedding=_embedding(),
            filters={"topic": "science"},
            namespace="tenant-b",
        )

        call_kwargs = mock_client.search.call_args.kwargs
        assert call_kwargs["query_filter"] is not None

    @pytest.mark.asyncio
    async def test_search_with_min_score(self, config, mock_qdrant_env):
        """search() should pass score_threshold."""
        _, mock_client = mock_qdrant_env
        store = _make_store(config, mock_client)
        mock_client.search.return_value = []

        await store.search(embedding=_embedding(), min_score=0.7)

        call_kwargs = mock_client.search.call_args.kwargs
        assert call_kwargs["score_threshold"] == 0.7

    @pytest.mark.asyncio
    async def test_search_empty_results(self, config, mock_qdrant_env):
        """search() should return empty list when no matches."""
        _, mock_client = mock_qdrant_env
        store = _make_store(config, mock_client)
        mock_client.search.return_value = []

        results = await store.search(embedding=_embedding())

        assert results == []

    @pytest.mark.asyncio
    async def test_search_not_connected(self, config, mock_qdrant_env):
        """search() should raise ConnectionError when not connected."""
        from aragora.knowledge.mound.vector_abstraction.qdrant import QdrantVectorStore

        store = QdrantVectorStore(config)
        with pytest.raises(ConnectionError):
            await store.search(embedding=_embedding())

    @pytest.mark.asyncio
    async def test_search_point_without_vector(self, config, mock_qdrant_env):
        """search() should handle points with no vector gracefully."""
        _, mock_client = mock_qdrant_env
        store = _make_store(config, mock_client)

        mock_client.search.return_value = [
            self._make_point("r1", 0.9, {"content": "text"}, None),
        ]

        results = await store.search(embedding=_embedding())
        assert results[0].embedding is None

    @pytest.mark.asyncio
    async def test_search_point_with_empty_payload(self, config, mock_qdrant_env):
        """search() should handle points with None payload."""
        _, mock_client = mock_qdrant_env
        store = _make_store(config, mock_client)

        point = MagicMock()
        point.id = "r1"
        point.score = 0.8
        point.payload = None
        point.vector = None
        mock_client.search.return_value = [point]

        results = await store.search(embedding=_embedding())
        assert results[0].content == ""
        assert results[0].metadata == {}


# ---------------------------------------------------------------------------
# Hybrid Search
# ---------------------------------------------------------------------------


class TestQdrantHybridSearch:
    """Tests for hybrid vector + keyword search."""

    @pytest.mark.asyncio
    async def test_hybrid_search_reranks(self, config, mock_qdrant_env):
        """hybrid_search() should re-rank results by combining vector and keyword scores."""
        _, mock_client = mock_qdrant_env
        store = _make_store(config, mock_client)

        # Mock the underlying vector search results
        p1 = MagicMock()
        p1.id = "r1"
        p1.score = 0.9
        p1.payload = {"content": "machine learning algorithms", "namespace": ""}
        p1.vector = None

        p2 = MagicMock()
        p2.id = "r2"
        p2.score = 0.85
        p2.payload = {"content": "deep learning neural networks", "namespace": ""}
        p2.vector = None

        mock_client.search.return_value = [p1, p2]

        results = await store.hybrid_search(
            query="machine learning",
            embedding=_embedding(),
            limit=2,
            alpha=0.5,
        )

        assert len(results) <= 2
        # r1 should rank higher because it contains both query terms
        assert results[0].id == "r1"

    @pytest.mark.asyncio
    async def test_hybrid_search_alpha_zero_pure_vector(self, config, mock_qdrant_env):
        """alpha=0 should give pure vector scores."""
        _, mock_client = mock_qdrant_env
        store = _make_store(config, mock_client)

        p1 = MagicMock()
        p1.id = "r1"
        p1.score = 0.9
        p1.payload = {"content": "unrelated words", "namespace": ""}
        p1.vector = None
        mock_client.search.return_value = [p1]

        results = await store.hybrid_search(
            query="machine learning",
            embedding=_embedding(),
            limit=5,
            alpha=0.0,
        )

        # alpha=0 means combined = (1-0)*vector + 0*keyword = vector only
        assert results[0].score == pytest.approx(0.9)

    @pytest.mark.asyncio
    async def test_hybrid_search_empty_query(self, config, mock_qdrant_env):
        """hybrid_search() with empty query should not error."""
        _, mock_client = mock_qdrant_env
        store = _make_store(config, mock_client)
        mock_client.search.return_value = []

        results = await store.hybrid_search(
            query="",
            embedding=_embedding(),
            limit=5,
        )

        assert results == []

    @pytest.mark.asyncio
    async def test_hybrid_search_limits_results(self, config, mock_qdrant_env):
        """hybrid_search() should respect the limit parameter."""
        _, mock_client = mock_qdrant_env
        store = _make_store(config, mock_client)

        points = []
        for i in range(10):
            p = MagicMock()
            p.id = f"r{i}"
            p.score = 0.9 - i * 0.05
            p.payload = {"content": f"content {i}", "namespace": ""}
            p.vector = None
            points.append(p)

        mock_client.search.return_value = points

        results = await store.hybrid_search(
            query="content",
            embedding=_embedding(),
            limit=3,
        )

        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_hybrid_search_not_connected(self, config, mock_qdrant_env):
        """hybrid_search() should raise when not connected."""
        from aragora.knowledge.mound.vector_abstraction.qdrant import QdrantVectorStore

        store = QdrantVectorStore(config)
        with pytest.raises(ConnectionError):
            await store.hybrid_search(query="q", embedding=_embedding())


# ---------------------------------------------------------------------------
# Retrieval Operations
# ---------------------------------------------------------------------------


class TestQdrantRetrieval:
    """Tests for get_by_id and get_by_ids."""

    @pytest.mark.asyncio
    async def test_get_by_id_found(self, config, mock_qdrant_env):
        """get_by_id() should return VectorSearchResult for existing ID."""
        _, mock_client = mock_qdrant_env
        store = _make_store(config, mock_client)

        point = MagicMock()
        point.id = "vec-1"
        point.payload = {"content": "hello world", "namespace": "ns", "topic": "greet"}
        point.vector = _embedding(dim=4)
        mock_client.retrieve.return_value = [point]

        result = await store.get_by_id("vec-1")

        assert result is not None
        assert result.id == "vec-1"
        assert result.content == "hello world"
        assert result.score == 1.0
        assert result.metadata == {"topic": "greet"}
        assert result.embedding == _embedding(dim=4)

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self, config, mock_qdrant_env):
        """get_by_id() should return None for missing ID."""
        _, mock_client = mock_qdrant_env
        store = _make_store(config, mock_client)

        mock_client.retrieve.return_value = []

        result = await store.get_by_id("missing-id")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_by_id_handles_error(self, config, mock_qdrant_env):
        """get_by_id() should return None on retrieval error."""
        _, mock_client = mock_qdrant_env
        store = _make_store(config, mock_client)

        mock_client.retrieve.side_effect = RuntimeError("network error")

        result = await store.get_by_id("error-id")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_by_id_none_payload(self, config, mock_qdrant_env):
        """get_by_id() should handle point with None payload."""
        _, mock_client = mock_qdrant_env
        store = _make_store(config, mock_client)

        point = MagicMock()
        point.id = "vec-1"
        point.payload = None
        point.vector = None
        mock_client.retrieve.return_value = [point]

        result = await store.get_by_id("vec-1")

        assert result is not None
        assert result.content == ""
        assert result.metadata == {}

    @pytest.mark.asyncio
    async def test_get_by_ids(self, config, mock_qdrant_env):
        """get_by_ids() should return multiple results."""
        _, mock_client = mock_qdrant_env
        store = _make_store(config, mock_client)

        p1 = MagicMock()
        p1.id = "v1"
        p1.payload = {"content": "first", "namespace": ""}
        p1.vector = None

        p2 = MagicMock()
        p2.id = "v2"
        p2.payload = {"content": "second", "namespace": ""}
        p2.vector = None

        mock_client.retrieve.return_value = [p1, p2]

        results = await store.get_by_ids(["v1", "v2"])

        assert len(results) == 2
        assert results[0].id == "v1"
        assert results[1].id == "v2"

    @pytest.mark.asyncio
    async def test_get_by_ids_not_connected(self, config, mock_qdrant_env):
        """get_by_ids() should raise when not connected."""
        from aragora.knowledge.mound.vector_abstraction.qdrant import QdrantVectorStore

        store = QdrantVectorStore(config)
        with pytest.raises(ConnectionError):
            await store.get_by_ids(["x"])


# ---------------------------------------------------------------------------
# Count
# ---------------------------------------------------------------------------


class TestQdrantCount:
    """Tests for count operations."""

    @pytest.mark.asyncio
    async def test_count_all(self, config, mock_qdrant_env):
        """count() without filters should return total."""
        _, mock_client = mock_qdrant_env
        store = _make_store(config, mock_client)

        mock_result = MagicMock()
        mock_result.count = 42
        mock_client.count.return_value = mock_result

        count = await store.count()

        assert count == 42

    @pytest.mark.asyncio
    async def test_count_with_filters(self, config, mock_qdrant_env):
        """count() with filters should pass filter to client."""
        _, mock_client = mock_qdrant_env
        store = _make_store(config, mock_client)

        mock_result = MagicMock()
        mock_result.count = 7
        mock_client.count.return_value = mock_result

        count = await store.count(filters={"topic": "ai"}, namespace="ns1")

        assert count == 7
        call_kwargs = mock_client.count.call_args.kwargs
        assert call_kwargs["count_filter"] is not None

    @pytest.mark.asyncio
    async def test_count_not_connected(self, config, mock_qdrant_env):
        """count() should raise when not connected."""
        from aragora.knowledge.mound.vector_abstraction.qdrant import QdrantVectorStore

        store = QdrantVectorStore(config)
        with pytest.raises(ConnectionError):
            await store.count()


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------


class TestQdrantHealth:
    """Tests for health and diagnostics."""

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, config, mock_qdrant_env):
        """health_check() should return healthy status."""
        _, mock_client = mock_qdrant_env
        store = _make_store(config, mock_client)

        mock_info = MagicMock()
        mock_info.collections = [MagicMock(), MagicMock()]
        mock_client.get_collections.return_value = mock_info

        health = await store.health_check()

        assert health["status"] == "healthy"
        assert health["backend"] == "qdrant"
        assert health["collections"] == 2

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, config, mock_qdrant_env):
        """health_check() should return unhealthy on error."""
        _, mock_client = mock_qdrant_env
        store = _make_store(config, mock_client)

        mock_client.get_collections.side_effect = ConnectionError("unreachable")

        health = await store.health_check()

        assert health["status"] == "unhealthy"
        assert health["backend"] == "qdrant"
        assert health["error"] == "Health check failed"

    @pytest.mark.asyncio
    async def test_health_check_disconnected(self, config, mock_qdrant_env):
        """health_check() should return disconnected when no client."""
        from aragora.knowledge.mound.vector_abstraction.qdrant import QdrantVectorStore

        store = QdrantVectorStore(config)

        health = await store.health_check()

        assert health["status"] == "disconnected"
        assert health["backend"] == "qdrant"

    @pytest.mark.asyncio
    async def test_ping_healthy(self, config, mock_qdrant_env):
        """ping() should return True when healthy."""
        _, mock_client = mock_qdrant_env
        store = _make_store(config, mock_client)

        mock_info = MagicMock()
        mock_info.collections = []
        mock_client.get_collections.return_value = mock_info

        assert await store.ping() is True

    @pytest.mark.asyncio
    async def test_ping_unhealthy(self, config, mock_qdrant_env):
        """ping() should return False when unhealthy."""
        _, mock_client = mock_qdrant_env
        store = _make_store(config, mock_client)
        mock_client.get_collections.side_effect = ConnectionError("down")

        assert await store.ping() is False


# ---------------------------------------------------------------------------
# Helper Methods
# ---------------------------------------------------------------------------


class TestQdrantHelpers:
    """Tests for internal helper methods."""

    def test_to_qdrant_id_uuid(self, config, mock_qdrant_env):
        """_to_qdrant_id() should pass through valid UUIDs."""
        from aragora.knowledge.mound.vector_abstraction.qdrant import QdrantVectorStore

        store = QdrantVectorStore(config)
        test_uuid = "550e8400-e29b-41d4-a716-446655440000"
        result = store._to_qdrant_id(test_uuid)
        assert result == test_uuid

    def test_to_qdrant_id_non_uuid(self, config, mock_qdrant_env):
        """_to_qdrant_id() should return raw string for non-UUID ids."""
        from aragora.knowledge.mound.vector_abstraction.qdrant import QdrantVectorStore

        store = QdrantVectorStore(config)
        result = store._to_qdrant_id("my-custom-id")
        assert result == "my-custom-id"

    def test_from_qdrant_id(self, config, mock_qdrant_env):
        """_from_qdrant_id() should convert any ID to string."""
        from aragora.knowledge.mound.vector_abstraction.qdrant import QdrantVectorStore

        store = QdrantVectorStore(config)
        assert store._from_qdrant_id(12345) == "12345"
        assert store._from_qdrant_id("abc") == "abc"

    def test_build_filter_none(self, config, mock_qdrant_env):
        """_build_filter() with no args should return None."""
        from aragora.knowledge.mound.vector_abstraction.qdrant import QdrantVectorStore

        store = QdrantVectorStore(config)
        assert store._build_filter(None, None) is None

    def test_build_filter_namespace_only(self, config, mock_qdrant_env):
        """_build_filter() with namespace should create a filter."""
        from aragora.knowledge.mound.vector_abstraction.qdrant import QdrantVectorStore

        store = QdrantVectorStore(config)
        result = store._build_filter(None, "tenant-a")
        assert result is not None
        assert len(result.must) == 1

    def test_build_filter_multiple(self, config, mock_qdrant_env):
        """_build_filter() should combine namespace and filter conditions."""
        from aragora.knowledge.mound.vector_abstraction.qdrant import QdrantVectorStore

        store = QdrantVectorStore(config)
        result = store._build_filter({"topic": "ai", "lang": "en"}, "ns1")
        assert result is not None
        assert len(result.must) == 3  # namespace + 2 filter keys


# ---------------------------------------------------------------------------
# Backend Registration
# ---------------------------------------------------------------------------


class TestQdrantBackendRegistration:
    """Tests for backend enum and config."""

    def test_backend_enum_exists(self):
        """Should have QDRANT in VectorBackend enum."""
        assert hasattr(VectorBackend, "QDRANT")
        assert VectorBackend.QDRANT.value == "qdrant"

    def test_store_sets_backend(self, config, mock_qdrant_env):
        """QdrantVectorStore should set config backend to QDRANT."""
        from aragora.knowledge.mound.vector_abstraction.qdrant import QdrantVectorStore

        store = QdrantVectorStore(config)
        assert store.backend == VectorBackend.QDRANT

    def test_import_check(self):
        """Module should expose QDRANT_AVAILABLE flag."""
        from aragora.knowledge.mound.vector_abstraction import qdrant as qdrant_mod

        assert hasattr(qdrant_mod, "QDRANT_AVAILABLE")

    def test_collection_name_property(self, config, mock_qdrant_env):
        """collection_name property should return config value."""
        from aragora.knowledge.mound.vector_abstraction.qdrant import QdrantVectorStore

        store = QdrantVectorStore(config)
        assert store.collection_name == "test_collection"
