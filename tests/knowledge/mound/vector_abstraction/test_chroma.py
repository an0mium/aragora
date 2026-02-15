"""
Tests for ChromaDB vector store adapter.

Covers connection management, CRUD operations, vector similarity search,
hybrid search, filter queries, error handling, and edge cases using mocked
chromadb client.
"""

import pytest
import uuid
from unittest.mock import MagicMock, patch

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
    """Create test configuration for Chroma with local persistence."""
    return VectorStoreConfig(
        backend=VectorBackend.CHROMA,
        url="./test_chroma_data",
        collection_name="test_collection",
        embedding_dimensions=1536,
        distance_metric="cosine",
    )


@pytest.fixture
def config_http():
    """Create test configuration for Chroma HTTP client."""
    return VectorStoreConfig(
        backend=VectorBackend.CHROMA,
        url="http://localhost:8000",
        collection_name="test_collection",
        embedding_dimensions=1536,
    )


@pytest.fixture
def mock_chroma_env():
    """
    Patch ChromaDB availability and client factories so ChromaVectorStore
    can be instantiated without the real chromadb library.
    Yields (mock_chromadb, mock_persistent_client, mock_http_client).
    """
    mock_chromadb = MagicMock()
    mock_persistent_client = MagicMock()
    mock_http_client = MagicMock()

    mock_chromadb.PersistentClient.return_value = mock_persistent_client
    mock_chromadb.HttpClient.return_value = mock_http_client

    _mod = "aragora.knowledge.mound.vector_abstraction.chroma"

    with (
        patch(f"{_mod}.CHROMA_AVAILABLE", True),
        patch(f"{_mod}.chromadb", mock_chromadb, create=True),
        patch(f"{_mod}.Settings", MagicMock(), create=True),
    ):
        yield mock_chromadb, mock_persistent_client, mock_http_client


def _make_store(config, mock_client):
    """Helper: create a ChromaVectorStore with pre-wired mock client."""
    from aragora.knowledge.mound.vector_abstraction.chroma import ChromaVectorStore

    store = ChromaVectorStore(config)
    store._client = mock_client
    store._connected = True
    return store


def _embedding(dim=1536, val=0.1):
    """Helper: create a dummy embedding vector."""
    return [val] * dim


# ---------------------------------------------------------------------------
# Connection Management
# ---------------------------------------------------------------------------


class TestChromaConnection:
    """Tests for Chroma connection lifecycle."""

    @pytest.mark.asyncio
    async def test_connect_persistent_client(self, config, mock_chroma_env):
        """connect() should create PersistentClient for local path."""
        mock_chromadb, mock_persistent, _ = mock_chroma_env

        # Make collection_exists return True (collection found)
        mock_persistent.get_collection.return_value = MagicMock()

        from aragora.knowledge.mound.vector_abstraction.chroma import ChromaVectorStore

        store = ChromaVectorStore(config)
        await store.connect()

        assert store.is_connected
        mock_chromadb.PersistentClient.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_http_client(self, config_http, mock_chroma_env):
        """connect() should create HttpClient for http:// URLs."""
        mock_chromadb, _, mock_http = mock_chroma_env

        mock_http.get_collection.return_value = MagicMock()

        from aragora.knowledge.mound.vector_abstraction.chroma import ChromaVectorStore

        store = ChromaVectorStore(config_http)
        await store.connect()

        assert store.is_connected
        mock_chromadb.HttpClient.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_already_connected_noop(self, config, mock_chroma_env):
        """Calling connect() when already connected should be a no-op."""
        _, mock_persistent, _ = mock_chroma_env
        store = _make_store(config, mock_persistent)

        await store.connect()
        # No error, no reconnection

    @pytest.mark.asyncio
    async def test_connect_creates_collection_if_missing(self, config, mock_chroma_env):
        """connect() should create the default collection if it does not exist."""
        mock_chromadb, mock_persistent, _ = mock_chroma_env

        # get_collection raises to indicate missing
        mock_persistent.get_collection.side_effect = ValueError("not found")

        from aragora.knowledge.mound.vector_abstraction.chroma import ChromaVectorStore

        store = ChromaVectorStore(config)
        await store.connect()

        assert store.is_connected
        mock_persistent.get_or_create_collection.assert_called()

    @pytest.mark.asyncio
    async def test_connect_failure_raises_connection_error(self, config, mock_chroma_env):
        """connect() should raise ConnectionError on failure."""
        mock_chromadb, _, _ = mock_chroma_env
        mock_chromadb.PersistentClient.side_effect = Exception("disk error")

        from aragora.knowledge.mound.vector_abstraction.chroma import ChromaVectorStore

        store = ChromaVectorStore(config)
        with pytest.raises(ConnectionError, match="Failed to connect to Chroma"):
            await store.connect()

        assert not store.is_connected

    @pytest.mark.asyncio
    async def test_disconnect(self, config, mock_chroma_env):
        """disconnect() should clear client and reset state."""
        _, mock_persistent, _ = mock_chroma_env
        store = _make_store(config, mock_persistent)
        store._collections["test_collection"] = MagicMock()

        await store.disconnect()

        assert not store.is_connected
        assert store._client is None
        assert store._collections == {}

    @pytest.mark.asyncio
    async def test_disconnect_when_not_connected(self, config, mock_chroma_env):
        """disconnect() with no client should be a no-op."""
        from aragora.knowledge.mound.vector_abstraction.chroma import ChromaVectorStore

        store = ChromaVectorStore(config)
        await store.disconnect()  # Should not raise

    @pytest.mark.asyncio
    async def test_context_manager(self, config, mock_chroma_env):
        """Async context manager should connect and disconnect."""
        _, mock_persistent, _ = mock_chroma_env
        mock_persistent.get_collection.return_value = MagicMock()

        from aragora.knowledge.mound.vector_abstraction.chroma import ChromaVectorStore

        async with ChromaVectorStore(config) as store:
            assert store.is_connected

        assert not store.is_connected


# ---------------------------------------------------------------------------
# Collection Management
# ---------------------------------------------------------------------------


class TestChromaCollections:
    """Tests for collection CRUD operations."""

    @pytest.mark.asyncio
    async def test_create_collection(self, config, mock_chroma_env):
        """create_collection() should call get_or_create_collection."""
        _, mock_persistent, _ = mock_chroma_env
        store = _make_store(config, mock_persistent)

        await store.create_collection("my_collection")

        mock_persistent.get_or_create_collection.assert_called_once_with(
            name="my_collection",
            metadata={"hnsw:space": "cosine"},
        )

    @pytest.mark.asyncio
    async def test_create_collection_euclidean(self, mock_chroma_env):
        """create_collection() should map euclidean to l2."""
        _, mock_persistent, _ = mock_chroma_env
        euc_config = VectorStoreConfig(
            backend=VectorBackend.CHROMA,
            collection_name="test",
            distance_metric="euclidean",
        )
        store = _make_store(euc_config, mock_persistent)

        await store.create_collection("euc_col")

        call_kwargs = mock_persistent.get_or_create_collection.call_args
        assert call_kwargs.kwargs["metadata"]["hnsw:space"] == "l2"

    @pytest.mark.asyncio
    async def test_create_collection_dot_product(self, mock_chroma_env):
        """create_collection() should map dot_product to ip."""
        _, mock_persistent, _ = mock_chroma_env
        dp_config = VectorStoreConfig(
            backend=VectorBackend.CHROMA,
            collection_name="test",
            distance_metric="dot_product",
        )
        store = _make_store(dp_config, mock_persistent)

        await store.create_collection("dp_col")

        call_kwargs = mock_persistent.get_or_create_collection.call_args
        assert call_kwargs.kwargs["metadata"]["hnsw:space"] == "ip"

    @pytest.mark.asyncio
    async def test_create_collection_not_connected(self, config, mock_chroma_env):
        """create_collection() should raise when not connected."""
        from aragora.knowledge.mound.vector_abstraction.chroma import ChromaVectorStore

        store = ChromaVectorStore(config)
        with pytest.raises(ConnectionError, match="Not connected"):
            await store.create_collection("test")

    @pytest.mark.asyncio
    async def test_delete_collection_success(self, config, mock_chroma_env):
        """delete_collection() should return True on success."""
        _, mock_persistent, _ = mock_chroma_env
        store = _make_store(config, mock_persistent)
        store._collections["my_collection"] = MagicMock()

        result = await store.delete_collection("my_collection")

        assert result is True
        mock_persistent.delete_collection.assert_called_once_with("my_collection")
        assert "my_collection" not in store._collections

    @pytest.mark.asyncio
    async def test_delete_collection_failure(self, config, mock_chroma_env):
        """delete_collection() should return False on error."""
        _, mock_persistent, _ = mock_chroma_env
        mock_persistent.delete_collection.side_effect = ValueError("not found")
        store = _make_store(config, mock_persistent)

        result = await store.delete_collection("missing")

        assert result is False

    @pytest.mark.asyncio
    async def test_collection_exists_true(self, config, mock_chroma_env):
        """collection_exists() should return True when collection found."""
        _, mock_persistent, _ = mock_chroma_env
        mock_persistent.get_collection.return_value = MagicMock()
        store = _make_store(config, mock_persistent)

        assert await store.collection_exists("test_collection") is True

    @pytest.mark.asyncio
    async def test_collection_exists_false(self, config, mock_chroma_env):
        """collection_exists() should return False when get_collection raises."""
        _, mock_persistent, _ = mock_chroma_env
        mock_persistent.get_collection.side_effect = ValueError("not found")
        store = _make_store(config, mock_persistent)

        assert await store.collection_exists("missing") is False

    @pytest.mark.asyncio
    async def test_list_collections(self, config, mock_chroma_env):
        """list_collections() should return names of all collections."""
        _, mock_persistent, _ = mock_chroma_env
        c1 = MagicMock()
        c1.name = "col_a"
        c2 = MagicMock()
        c2.name = "col_b"
        mock_persistent.list_collections.return_value = [c1, c2]
        store = _make_store(config, mock_persistent)

        names = await store.list_collections()

        assert names == ["col_a", "col_b"]

    @pytest.mark.asyncio
    async def test_list_collections_not_connected(self, config, mock_chroma_env):
        """list_collections() should raise when not connected."""
        from aragora.knowledge.mound.vector_abstraction.chroma import ChromaVectorStore

        store = ChromaVectorStore(config)
        with pytest.raises(ConnectionError):
            await store.list_collections()


# ---------------------------------------------------------------------------
# CRUD Operations
# ---------------------------------------------------------------------------


class TestChromaUpsert:
    """Tests for insert/update operations."""

    @pytest.mark.asyncio
    async def test_upsert_single(self, config, mock_chroma_env):
        """upsert() should insert a single vector."""
        _, mock_persistent, _ = mock_chroma_env
        store = _make_store(config, mock_persistent)

        mock_collection = MagicMock()
        store._collections["test_collection"] = mock_collection

        result = await store.upsert(
            id="vec-1",
            embedding=_embedding(),
            content="test content",
            metadata={"topic": "science"},
        )

        assert result == "vec-1"
        mock_collection.upsert.assert_called_once()
        call_kwargs = mock_collection.upsert.call_args
        assert call_kwargs.kwargs["ids"] == ["vec-1"]
        assert call_kwargs.kwargs["documents"] == ["test content"]
        assert call_kwargs.kwargs["embeddings"] == [_embedding()]

    @pytest.mark.asyncio
    async def test_upsert_with_namespace(self, config, mock_chroma_env):
        """upsert() should include namespace in metadata."""
        _, mock_persistent, _ = mock_chroma_env
        store = _make_store(config, mock_persistent)

        mock_collection = MagicMock()
        store._collections["test_collection"] = mock_collection

        await store.upsert(
            id="vec-2",
            embedding=_embedding(),
            content="ns content",
            namespace="tenant-a",
        )

        call_kwargs = mock_collection.upsert.call_args
        meta = call_kwargs.kwargs["metadatas"][0]
        assert meta["namespace"] == "tenant-a"

    @pytest.mark.asyncio
    async def test_upsert_without_metadata(self, config, mock_chroma_env):
        """upsert() should work with no metadata."""
        _, mock_persistent, _ = mock_chroma_env
        store = _make_store(config, mock_persistent)

        mock_collection = MagicMock()
        store._collections["test_collection"] = mock_collection

        result = await store.upsert(
            id="vec-3",
            embedding=_embedding(),
            content="bare content",
        )

        assert result == "vec-3"
        call_kwargs = mock_collection.upsert.call_args
        meta = call_kwargs.kwargs["metadatas"][0]
        assert meta["namespace"] == ""

    @pytest.mark.asyncio
    async def test_upsert_sanitizes_metadata(self, config, mock_chroma_env):
        """upsert() should sanitize metadata values for Chroma."""
        _, mock_persistent, _ = mock_chroma_env
        store = _make_store(config, mock_persistent)

        mock_collection = MagicMock()
        store._collections["test_collection"] = mock_collection

        await store.upsert(
            id="vec-4",
            embedding=_embedding(),
            content="meta test",
            metadata={"key_none": None, "key_list": [1, 2, 3], "key_str": "ok"},
        )

        call_kwargs = mock_collection.upsert.call_args
        meta = call_kwargs.kwargs["metadatas"][0]
        # None should become ""
        assert meta["key_none"] == ""
        # list should be stringified
        assert meta["key_list"] == "[1, 2, 3]"
        # str should remain
        assert meta["key_str"] == "ok"

    @pytest.mark.asyncio
    async def test_upsert_not_connected(self, config, mock_chroma_env):
        """upsert() should raise ConnectionError when not connected."""
        from aragora.knowledge.mound.vector_abstraction.chroma import ChromaVectorStore

        store = ChromaVectorStore(config)
        with pytest.raises(ConnectionError):
            await store.upsert(id="x", embedding=_embedding(), content="c")

    @pytest.mark.asyncio
    async def test_upsert_batch(self, config, mock_chroma_env):
        """upsert_batch() should insert multiple vectors at once."""
        _, mock_persistent, _ = mock_chroma_env
        store = _make_store(config, mock_persistent)

        mock_collection = MagicMock()
        store._collections["test_collection"] = mock_collection

        items = [
            {"id": "b1", "embedding": _embedding(), "content": "first", "metadata": {"k": "v1"}},
            {"id": "b2", "embedding": _embedding(), "content": "second", "metadata": {"k": "v2"}},
            {"id": "b3", "embedding": _embedding(), "content": "third"},
        ]

        ids = await store.upsert_batch(items)

        assert ids == ["b1", "b2", "b3"]
        mock_collection.upsert.assert_called_once()
        call_kwargs = mock_collection.upsert.call_args
        assert len(call_kwargs.kwargs["ids"]) == 3
        assert len(call_kwargs.kwargs["embeddings"]) == 3
        assert len(call_kwargs.kwargs["documents"]) == 3

    @pytest.mark.asyncio
    async def test_upsert_batch_auto_generates_ids(self, config, mock_chroma_env):
        """upsert_batch() should auto-generate UUIDs for items without ids."""
        _, mock_persistent, _ = mock_chroma_env
        store = _make_store(config, mock_persistent)

        mock_collection = MagicMock()
        store._collections["test_collection"] = mock_collection

        items = [
            {"embedding": _embedding(), "content": "no id item"},
        ]

        ids = await store.upsert_batch(items)

        assert len(ids) == 1
        # Should be a valid UUID4
        uuid.UUID(ids[0])

    @pytest.mark.asyncio
    async def test_upsert_batch_with_namespace(self, config, mock_chroma_env):
        """upsert_batch() should set namespace in all metadatas."""
        _, mock_persistent, _ = mock_chroma_env
        store = _make_store(config, mock_persistent)

        mock_collection = MagicMock()
        store._collections["test_collection"] = mock_collection

        items = [
            {"id": "n1", "embedding": _embedding(), "content": "c1"},
            {"id": "n2", "embedding": _embedding(), "content": "c2"},
        ]

        await store.upsert_batch(items, namespace="tenant-x")

        call_kwargs = mock_collection.upsert.call_args
        for meta in call_kwargs.kwargs["metadatas"]:
            assert meta["namespace"] == "tenant-x"

    @pytest.mark.asyncio
    async def test_upsert_batch_not_connected(self, config, mock_chroma_env):
        """upsert_batch() should raise when not connected."""
        from aragora.knowledge.mound.vector_abstraction.chroma import ChromaVectorStore

        store = ChromaVectorStore(config)
        with pytest.raises(ConnectionError):
            await store.upsert_batch([{"id": "x", "embedding": [], "content": "c"}])


# ---------------------------------------------------------------------------
# Delete Operations
# ---------------------------------------------------------------------------


class TestChromaDelete:
    """Tests for delete operations."""

    @pytest.mark.asyncio
    async def test_delete_by_ids(self, config, mock_chroma_env):
        """delete() should delete by IDs and return count."""
        _, mock_persistent, _ = mock_chroma_env
        store = _make_store(config, mock_persistent)

        mock_collection = MagicMock()
        store._collections["test_collection"] = mock_collection

        count = await store.delete(ids=["id-1", "id-2", "id-3"])

        assert count == 3
        mock_collection.delete.assert_called_once_with(ids=["id-1", "id-2", "id-3"])

    @pytest.mark.asyncio
    async def test_delete_empty_ids(self, config, mock_chroma_env):
        """delete() with empty list should return 0."""
        _, mock_persistent, _ = mock_chroma_env
        store = _make_store(config, mock_persistent)

        mock_collection = MagicMock()
        store._collections["test_collection"] = mock_collection

        count = await store.delete(ids=[])

        assert count == 0

    @pytest.mark.asyncio
    async def test_delete_not_connected(self, config, mock_chroma_env):
        """delete() should raise when not connected."""
        from aragora.knowledge.mound.vector_abstraction.chroma import ChromaVectorStore

        store = ChromaVectorStore(config)
        with pytest.raises(ConnectionError):
            await store.delete(ids=["x"])

    @pytest.mark.asyncio
    async def test_delete_by_filter(self, config, mock_chroma_env):
        """delete_by_filter() should find matching IDs and delete them."""
        _, mock_persistent, _ = mock_chroma_env
        store = _make_store(config, mock_persistent)

        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": ["id-1", "id-2", "id-3"]}
        store._collections["test_collection"] = mock_collection

        count = await store.delete_by_filter(
            filters={"topic": "obsolete"},
            namespace="tenant-a",
        )

        assert count == 3
        mock_collection.delete.assert_called_once_with(ids=["id-1", "id-2", "id-3"])

    @pytest.mark.asyncio
    async def test_delete_by_filter_no_matches(self, config, mock_chroma_env):
        """delete_by_filter() should return 0 when no matches."""
        _, mock_persistent, _ = mock_chroma_env
        store = _make_store(config, mock_persistent)

        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": []}
        store._collections["test_collection"] = mock_collection

        count = await store.delete_by_filter(filters={"topic": "none"})

        assert count == 0
        mock_collection.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_by_filter_not_connected(self, config, mock_chroma_env):
        """delete_by_filter() should raise when not connected."""
        from aragora.knowledge.mound.vector_abstraction.chroma import ChromaVectorStore

        store = ChromaVectorStore(config)
        with pytest.raises(ConnectionError):
            await store.delete_by_filter(filters={"k": "v"})


# ---------------------------------------------------------------------------
# Search Operations
# ---------------------------------------------------------------------------


class TestChromaSearch:
    """Tests for vector similarity search."""

    @pytest.mark.asyncio
    async def test_search_returns_results(self, config, mock_chroma_env):
        """search() should return VectorSearchResult list."""
        _, mock_persistent, _ = mock_chroma_env
        store = _make_store(config, mock_persistent)

        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [["r1", "r2"]],
            "documents": [["first result", "second result"]],
            "metadatas": [[{"namespace": "ns", "topic": "ai"}, {"namespace": "ns"}]],
            "distances": [[0.1, 0.3]],
            "embeddings": [[_embedding(dim=4), _embedding(dim=4, val=0.2)]],
        }
        store._collections["test_collection"] = mock_collection

        results = await store.search(embedding=_embedding(), limit=10)

        assert len(results) == 2
        assert results[0].id == "r1"
        assert results[0].score == pytest.approx(0.9)  # 1 - 0.1
        assert results[0].content == "first result"
        assert results[0].metadata == {"topic": "ai"}
        assert results[0].embedding == _embedding(dim=4)
        assert results[1].score == pytest.approx(0.7)  # 1 - 0.3

    @pytest.mark.asyncio
    async def test_search_with_min_score(self, config, mock_chroma_env):
        """search() should filter results below min_score."""
        _, mock_persistent, _ = mock_chroma_env
        store = _make_store(config, mock_persistent)

        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [["r1", "r2"]],
            "documents": [["close match", "far match"]],
            "metadatas": [[{"namespace": ""}, {"namespace": ""}]],
            "distances": [[0.05, 0.9]],
            "embeddings": None,
        }
        store._collections["test_collection"] = mock_collection

        results = await store.search(embedding=_embedding(), min_score=0.5)

        assert len(results) == 1
        assert results[0].id == "r1"

    @pytest.mark.asyncio
    async def test_search_with_filters(self, config, mock_chroma_env):
        """search() should pass where filter to Chroma."""
        _, mock_persistent, _ = mock_chroma_env
        store = _make_store(config, mock_persistent)

        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
            "embeddings": None,
        }
        store._collections["test_collection"] = mock_collection

        await store.search(
            embedding=_embedding(),
            filters={"topic": "science"},
            namespace="tenant-b",
        )

        call_kwargs = mock_collection.query.call_args.kwargs
        assert call_kwargs["where"] is not None

    @pytest.mark.asyncio
    async def test_search_empty_results(self, config, mock_chroma_env):
        """search() should return empty list when no matches."""
        _, mock_persistent, _ = mock_chroma_env
        store = _make_store(config, mock_persistent)

        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
            "embeddings": None,
        }
        store._collections["test_collection"] = mock_collection

        results = await store.search(embedding=_embedding())

        assert results == []

    @pytest.mark.asyncio
    async def test_search_no_embeddings_in_response(self, config, mock_chroma_env):
        """search() should handle missing embeddings in response."""
        _, mock_persistent, _ = mock_chroma_env
        store = _make_store(config, mock_persistent)

        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [["r1"]],
            "documents": [["result"]],
            "metadatas": [[{"namespace": ""}]],
            "distances": [[0.2]],
            "embeddings": None,
        }
        store._collections["test_collection"] = mock_collection

        results = await store.search(embedding=_embedding())

        assert results[0].embedding is None

    @pytest.mark.asyncio
    async def test_search_negative_distance_clamped(self, config, mock_chroma_env):
        """search() should clamp score to non-negative values."""
        _, mock_persistent, _ = mock_chroma_env
        store = _make_store(config, mock_persistent)

        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [["r1"]],
            "documents": [["result"]],
            "metadatas": [[{"namespace": ""}]],
            "distances": [[1.5]],  # distance > 1 means score = 1 - 1.5 = -0.5
            "embeddings": None,
        }
        store._collections["test_collection"] = mock_collection

        results = await store.search(embedding=_embedding())

        assert results[0].score == 0  # max(0, 1 - 1.5)

    @pytest.mark.asyncio
    async def test_search_not_connected(self, config, mock_chroma_env):
        """search() should raise ConnectionError when not connected."""
        from aragora.knowledge.mound.vector_abstraction.chroma import ChromaVectorStore

        store = ChromaVectorStore(config)
        with pytest.raises(ConnectionError):
            await store.search(embedding=_embedding())


# ---------------------------------------------------------------------------
# Hybrid Search
# ---------------------------------------------------------------------------


class TestChromaHybridSearch:
    """Tests for hybrid vector + keyword search."""

    @pytest.mark.asyncio
    async def test_hybrid_search_reranks(self, config, mock_chroma_env):
        """hybrid_search() should re-rank results by combined score."""
        _, mock_persistent, _ = mock_chroma_env
        store = _make_store(config, mock_persistent)

        mock_collection = MagicMock()
        # Return two results from the underlying search
        mock_collection.query.return_value = {
            "ids": [["r1", "r2"]],
            "documents": [["machine learning algorithms", "deep learning neural networks"]],
            "metadatas": [[{"namespace": ""}, {"namespace": ""}]],
            "distances": [[0.1, 0.15]],
            "embeddings": None,
        }
        store._collections["test_collection"] = mock_collection

        results = await store.hybrid_search(
            query="machine learning",
            embedding=_embedding(),
            limit=2,
            alpha=0.5,
        )

        assert len(results) <= 2
        # r1 contains "machine" and "learning" (2/2 overlap) -> higher keyword score
        # r2 contains "learning" only (1/2 overlap) -> lower keyword score
        assert results[0].id == "r1"

    @pytest.mark.asyncio
    async def test_hybrid_search_alpha_zero_pure_vector(self, config, mock_chroma_env):
        """alpha=0 should give pure vector scores."""
        _, mock_persistent, _ = mock_chroma_env
        store = _make_store(config, mock_persistent)

        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [["r1"]],
            "documents": [["unrelated content"]],
            "metadatas": [[{"namespace": ""}]],
            "distances": [[0.1]],
            "embeddings": None,
        }
        store._collections["test_collection"] = mock_collection

        results = await store.hybrid_search(
            query="machine learning",
            embedding=_embedding(),
            limit=5,
            alpha=0.0,
        )

        # alpha=0: combined = (1-0)*vector + 0*keyword = vector only = 0.9
        assert results[0].score == pytest.approx(0.9)

    @pytest.mark.asyncio
    async def test_hybrid_search_alpha_one_pure_keyword(self, config, mock_chroma_env):
        """alpha=1 should give pure keyword scores."""
        _, mock_persistent, _ = mock_chroma_env
        store = _make_store(config, mock_persistent)

        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [["r1"]],
            "documents": [["machine learning"]],
            "metadatas": [[{"namespace": ""}]],
            "distances": [[0.1]],
            "embeddings": None,
        }
        store._collections["test_collection"] = mock_collection

        results = await store.hybrid_search(
            query="machine learning",
            embedding=_embedding(),
            limit=5,
            alpha=1.0,
        )

        # alpha=1: combined = 0*vector + 1*keyword; query = {"machine","learning"}
        # content = {"machine","learning"} -> overlap = 2/2 = 1.0
        assert results[0].score == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_hybrid_search_empty_query(self, config, mock_chroma_env):
        """hybrid_search() with empty query should not error."""
        _, mock_persistent, _ = mock_chroma_env
        store = _make_store(config, mock_persistent)

        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
            "embeddings": None,
        }
        store._collections["test_collection"] = mock_collection

        results = await store.hybrid_search(query="", embedding=_embedding(), limit=5)

        assert results == []

    @pytest.mark.asyncio
    async def test_hybrid_search_limits_results(self, config, mock_chroma_env):
        """hybrid_search() should respect the limit parameter."""
        _, mock_persistent, _ = mock_chroma_env
        store = _make_store(config, mock_persistent)

        mock_collection = MagicMock()
        ids = [f"r{i}" for i in range(10)]
        docs = [f"content {i}" for i in range(10)]
        metas = [{"namespace": ""} for _ in range(10)]
        dists = [0.1 + i * 0.05 for i in range(10)]

        mock_collection.query.return_value = {
            "ids": [ids],
            "documents": [docs],
            "metadatas": [metas],
            "distances": [dists],
            "embeddings": None,
        }
        store._collections["test_collection"] = mock_collection

        results = await store.hybrid_search(
            query="content",
            embedding=_embedding(),
            limit=3,
        )

        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_hybrid_search_not_connected(self, config, mock_chroma_env):
        """hybrid_search() should raise when not connected."""
        from aragora.knowledge.mound.vector_abstraction.chroma import ChromaVectorStore

        store = ChromaVectorStore(config)
        with pytest.raises(ConnectionError):
            await store.hybrid_search(query="q", embedding=_embedding())


# ---------------------------------------------------------------------------
# Retrieval Operations
# ---------------------------------------------------------------------------


class TestChromaRetrieval:
    """Tests for get_by_id and get_by_ids."""

    @pytest.mark.asyncio
    async def test_get_by_id_found(self, config, mock_chroma_env):
        """get_by_id() should return VectorSearchResult for existing ID."""
        _, mock_persistent, _ = mock_chroma_env
        store = _make_store(config, mock_persistent)

        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "ids": ["vec-1"],
            "documents": ["hello world"],
            "metadatas": [{"namespace": "ns", "topic": "greet"}],
            "embeddings": [_embedding(dim=4)],
        }
        store._collections["test_collection"] = mock_collection

        result = await store.get_by_id("vec-1")

        assert result is not None
        assert result.id == "vec-1"
        assert result.content == "hello world"
        assert result.score == 1.0
        assert result.metadata == {"topic": "greet"}
        assert result.embedding == _embedding(dim=4)

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self, config, mock_chroma_env):
        """get_by_id() should return None when not found."""
        _, mock_persistent, _ = mock_chroma_env
        store = _make_store(config, mock_persistent)

        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "ids": [],
            "documents": [],
            "metadatas": [],
            "embeddings": [],
        }
        store._collections["test_collection"] = mock_collection

        result = await store.get_by_id("missing-id")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_by_id_handles_error(self, config, mock_chroma_env):
        """get_by_id() should return None on error."""
        _, mock_persistent, _ = mock_chroma_env
        store = _make_store(config, mock_persistent)

        mock_collection = MagicMock()
        mock_collection.get.side_effect = RuntimeError("error")
        store._collections["test_collection"] = mock_collection

        result = await store.get_by_id("error-id")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_by_id_no_embeddings(self, config, mock_chroma_env):
        """get_by_id() should handle missing embeddings."""
        _, mock_persistent, _ = mock_chroma_env
        store = _make_store(config, mock_persistent)

        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "ids": ["vec-1"],
            "documents": ["text"],
            "metadatas": [{"namespace": ""}],
        }
        store._collections["test_collection"] = mock_collection

        result = await store.get_by_id("vec-1")

        assert result is not None
        assert result.embedding is None

    @pytest.mark.asyncio
    async def test_get_by_id_no_metadatas(self, config, mock_chroma_env):
        """get_by_id() should handle empty metadatas."""
        _, mock_persistent, _ = mock_chroma_env
        store = _make_store(config, mock_persistent)

        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "ids": ["vec-1"],
            "documents": ["text"],
            "metadatas": None,
        }
        store._collections["test_collection"] = mock_collection

        result = await store.get_by_id("vec-1")

        assert result is not None
        assert result.metadata == {}

    @pytest.mark.asyncio
    async def test_get_by_ids(self, config, mock_chroma_env):
        """get_by_ids() should return multiple results."""
        _, mock_persistent, _ = mock_chroma_env
        store = _make_store(config, mock_persistent)

        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "ids": ["v1", "v2"],
            "documents": ["first", "second"],
            "metadatas": [{"namespace": ""}, {"namespace": "", "topic": "ai"}],
            "embeddings": None,
        }
        store._collections["test_collection"] = mock_collection

        results = await store.get_by_ids(["v1", "v2"])

        assert len(results) == 2
        assert results[0].id == "v1"
        assert results[1].id == "v2"
        assert results[1].metadata == {"topic": "ai"}

    @pytest.mark.asyncio
    async def test_get_by_ids_empty(self, config, mock_chroma_env):
        """get_by_ids() with empty input should return empty list."""
        _, mock_persistent, _ = mock_chroma_env
        store = _make_store(config, mock_persistent)

        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "ids": [],
            "documents": [],
            "metadatas": [],
        }
        store._collections["test_collection"] = mock_collection

        results = await store.get_by_ids([])

        assert results == []

    @pytest.mark.asyncio
    async def test_get_by_ids_not_connected(self, config, mock_chroma_env):
        """get_by_ids() should raise when not connected."""
        from aragora.knowledge.mound.vector_abstraction.chroma import ChromaVectorStore

        store = ChromaVectorStore(config)
        with pytest.raises(ConnectionError):
            await store.get_by_ids(["x"])


# ---------------------------------------------------------------------------
# Count
# ---------------------------------------------------------------------------


class TestChromaCount:
    """Tests for count operations."""

    @pytest.mark.asyncio
    async def test_count_all(self, config, mock_chroma_env):
        """count() without filters should call collection.count()."""
        _, mock_persistent, _ = mock_chroma_env
        store = _make_store(config, mock_persistent)

        mock_collection = MagicMock()
        mock_collection.count.return_value = 42
        store._collections["test_collection"] = mock_collection

        count = await store.count()

        assert count == 42
        mock_collection.count.assert_called_once()

    @pytest.mark.asyncio
    async def test_count_with_filters(self, config, mock_chroma_env):
        """count() with filters should use get() and count IDs."""
        _, mock_persistent, _ = mock_chroma_env
        store = _make_store(config, mock_persistent)

        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": ["a", "b", "c"]}
        store._collections["test_collection"] = mock_collection

        count = await store.count(filters={"topic": "ai"})

        assert count == 3

    @pytest.mark.asyncio
    async def test_count_with_namespace(self, config, mock_chroma_env):
        """count() with namespace should apply namespace filter."""
        _, mock_persistent, _ = mock_chroma_env
        store = _make_store(config, mock_persistent)

        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": ["a"]}
        store._collections["test_collection"] = mock_collection

        count = await store.count(namespace="ns1")

        assert count == 1
        call_kwargs = mock_collection.get.call_args.kwargs
        assert call_kwargs["where"] is not None

    @pytest.mark.asyncio
    async def test_count_not_connected(self, config, mock_chroma_env):
        """count() should raise when not connected."""
        from aragora.knowledge.mound.vector_abstraction.chroma import ChromaVectorStore

        store = ChromaVectorStore(config)
        with pytest.raises(ConnectionError):
            await store.count()


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------


class TestChromaHealth:
    """Tests for health and diagnostics."""

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, config, mock_chroma_env):
        """health_check() should return healthy status."""
        _, mock_persistent, _ = mock_chroma_env
        store = _make_store(config, mock_persistent)

        c1 = MagicMock()
        c2 = MagicMock()
        mock_persistent.list_collections.return_value = [c1, c2]

        health = await store.health_check()

        assert health["status"] == "healthy"
        assert health["backend"] == "chroma"
        assert health["collections"] == 2

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, config, mock_chroma_env):
        """health_check() should return unhealthy on error."""
        _, mock_persistent, _ = mock_chroma_env
        store = _make_store(config, mock_persistent)
        mock_persistent.list_collections.side_effect = RuntimeError("disk error")

        health = await store.health_check()

        assert health["status"] == "unhealthy"
        assert health["backend"] == "chroma"
        assert health["error"] == "Health check failed"

    @pytest.mark.asyncio
    async def test_health_check_disconnected(self, config, mock_chroma_env):
        """health_check() should return disconnected when no client."""
        from aragora.knowledge.mound.vector_abstraction.chroma import ChromaVectorStore

        store = ChromaVectorStore(config)

        health = await store.health_check()

        assert health["status"] == "disconnected"
        assert health["backend"] == "chroma"

    @pytest.mark.asyncio
    async def test_ping_healthy(self, config, mock_chroma_env):
        """ping() should return True when healthy."""
        _, mock_persistent, _ = mock_chroma_env
        store = _make_store(config, mock_persistent)
        mock_persistent.list_collections.return_value = []

        assert await store.ping() is True

    @pytest.mark.asyncio
    async def test_ping_unhealthy(self, config, mock_chroma_env):
        """ping() should return False when unhealthy."""
        _, mock_persistent, _ = mock_chroma_env
        store = _make_store(config, mock_persistent)
        mock_persistent.list_collections.side_effect = RuntimeError("down")

        assert await store.ping() is False


# ---------------------------------------------------------------------------
# Helper Methods
# ---------------------------------------------------------------------------


class TestChromaHelpers:
    """Tests for internal helper methods."""

    def test_get_collection_caches(self, config, mock_chroma_env):
        """_get_collection() should cache collection references."""
        _, mock_persistent, _ = mock_chroma_env
        store = _make_store(config, mock_persistent)

        mock_col = MagicMock()
        mock_persistent.get_or_create_collection.return_value = mock_col

        col1 = store._get_collection()
        col2 = store._get_collection()

        # Should only call get_or_create_collection once
        mock_persistent.get_or_create_collection.assert_called_once()
        assert col1 is col2

    def test_get_collection_not_connected(self, config, mock_chroma_env):
        """_get_collection() should raise when not connected."""
        from aragora.knowledge.mound.vector_abstraction.chroma import ChromaVectorStore

        store = ChromaVectorStore(config)
        with pytest.raises(ConnectionError):
            store._get_collection()

    def test_build_filter_none(self, config, mock_chroma_env):
        """_build_filter() with no args should return None."""
        _, mock_persistent, _ = mock_chroma_env
        store = _make_store(config, mock_persistent)
        assert store._build_filter(None, None) is None

    def test_build_filter_single_namespace(self, config, mock_chroma_env):
        """_build_filter() with only namespace should return single condition."""
        _, mock_persistent, _ = mock_chroma_env
        store = _make_store(config, mock_persistent)

        result = store._build_filter(None, "tenant-a")
        assert result == {"namespace": {"$eq": "tenant-a"}}

    def test_build_filter_single_filter(self, config, mock_chroma_env):
        """_build_filter() with single filter key should return single condition."""
        _, mock_persistent, _ = mock_chroma_env
        store = _make_store(config, mock_persistent)

        result = store._build_filter({"topic": "ai"}, None)
        assert result == {"topic": {"$eq": "ai"}}

    def test_build_filter_multiple(self, config, mock_chroma_env):
        """_build_filter() should combine with $and for multiple conditions."""
        _, mock_persistent, _ = mock_chroma_env
        store = _make_store(config, mock_persistent)

        result = store._build_filter({"topic": "ai"}, "ns1")
        assert "$and" in result
        assert len(result["$and"]) == 2

    def test_build_filter_multiple_filters(self, config, mock_chroma_env):
        """_build_filter() should handle multiple filter keys."""
        _, mock_persistent, _ = mock_chroma_env
        store = _make_store(config, mock_persistent)

        result = store._build_filter({"topic": "ai", "lang": "en"}, "ns1")
        assert "$and" in result
        assert len(result["$and"]) == 3

    def test_sanitize_metadata_valid_types(self, config, mock_chroma_env):
        """_sanitize_metadata() should pass through valid types."""
        _, mock_persistent, _ = mock_chroma_env
        store = _make_store(config, mock_persistent)

        meta = {"str": "hello", "int": 42, "float": 3.14, "bool": True}
        result = store._sanitize_metadata(meta)

        assert result == meta

    def test_sanitize_metadata_none_to_empty_string(self, config, mock_chroma_env):
        """_sanitize_metadata() should convert None to empty string."""
        _, mock_persistent, _ = mock_chroma_env
        store = _make_store(config, mock_persistent)

        result = store._sanitize_metadata({"key": None})
        assert result == {"key": ""}

    def test_sanitize_metadata_complex_to_string(self, config, mock_chroma_env):
        """_sanitize_metadata() should convert complex types to string."""
        _, mock_persistent, _ = mock_chroma_env
        store = _make_store(config, mock_persistent)

        result = store._sanitize_metadata({"list": [1, 2], "dict": {"a": "b"}})
        assert result["list"] == "[1, 2]"
        assert result["dict"] == "{'a': 'b'}"


# ---------------------------------------------------------------------------
# Backend Registration
# ---------------------------------------------------------------------------


class TestChromaBackendRegistration:
    """Tests for backend enum and config."""

    def test_backend_enum_exists(self):
        """Should have CHROMA in VectorBackend enum."""
        assert hasattr(VectorBackend, "CHROMA")
        assert VectorBackend.CHROMA.value == "chroma"

    def test_store_sets_backend(self, config, mock_chroma_env):
        """ChromaVectorStore should set config backend to CHROMA."""
        from aragora.knowledge.mound.vector_abstraction.chroma import ChromaVectorStore

        store = ChromaVectorStore(config)
        assert store.backend == VectorBackend.CHROMA

    def test_import_check(self):
        """Module should expose CHROMA_AVAILABLE flag."""
        from aragora.knowledge.mound.vector_abstraction import chroma as chroma_mod

        assert hasattr(chroma_mod, "CHROMA_AVAILABLE")

    def test_collection_name_property(self, config, mock_chroma_env):
        """collection_name property should return config value."""
        from aragora.knowledge.mound.vector_abstraction.chroma import ChromaVectorStore

        store = ChromaVectorStore(config)
        assert store.collection_name == "test_collection"
