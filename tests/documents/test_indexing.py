"""
Tests for document indexing and hybrid search.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass

from aragora.documents.indexing.weaviate_store import (
    WeaviateStore,
    WeaviateConfig,
    SearchResult,
    get_weaviate_store,
    WEAVIATE_AVAILABLE,
)
from aragora.documents.indexing.hybrid_search import (
    HybridSearcher,
    HybridSearchConfig,
    HybridResult,
    SimpleEmbedder,
)
from aragora.documents.models import DocumentChunk, ChunkType


class TestWeaviateConfig:
    """Tests for WeaviateConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = WeaviateConfig()
        assert config.url == "http://localhost:8080"
        assert config.api_key is None
        assert config.collection_name == "DocumentChunks"
        assert config.batch_size == 100

    def test_custom_config(self):
        """Test custom configuration."""
        config = WeaviateConfig(
            url="https://weaviate.example.com",
            api_key="test-key",
            collection_name="MyDocuments",
            batch_size=50,
        )
        assert config.url == "https://weaviate.example.com"
        assert config.api_key == "test-key"
        assert config.collection_name == "MyDocuments"
        assert config.batch_size == 50

    def test_from_env(self):
        """Test configuration from environment."""
        with patch.dict(
            "os.environ",
            {
                "WEAVIATE_URL": "http://test:8080",
                "WEAVIATE_API_KEY": "env-key",
                "WEAVIATE_COLLECTION": "EnvCollection",
            },
        ):
            config = WeaviateConfig.from_env()
            assert config.url == "http://test:8080"
            assert config.api_key == "env-key"
            assert config.collection_name == "EnvCollection"


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_create_search_result(self):
        """Test creating a search result."""
        result = SearchResult(
            chunk_id="chunk-123",
            document_id="doc-456",
            content="Test content",
            score=0.95,
            chunk_type="text",
            heading_context="Introduction",
            start_page=1,
            end_page=2,
        )
        assert result.chunk_id == "chunk-123"
        assert result.score == 0.95
        assert result.start_page == 1

    def test_to_dict(self):
        """Test serialization to dictionary."""
        result = SearchResult(
            chunk_id="c1",
            document_id="d1",
            content="Test",
            score=0.8,
        )
        data = result.to_dict()
        assert data["chunk_id"] == "c1"
        assert data["score"] == 0.8
        assert "metadata" in data


class TestWeaviateStore:
    """Tests for WeaviateStore class (unit tests without actual Weaviate)."""

    def test_initialization(self):
        """Test store initialization."""
        config = WeaviateConfig(url="http://test:8080")
        store = WeaviateStore(config)
        assert store.config.url == "http://test:8080"
        assert not store.is_connected

    def test_not_connected_raises_error(self):
        """Test that operations fail when not connected."""
        store = WeaviateStore()
        with pytest.raises(RuntimeError, match="Not connected"):
            # Sync wrapper to test async method
            import asyncio

            asyncio.get_event_loop().run_until_complete(store.search_keyword("test"))

    @pytest.mark.skipif(not WEAVIATE_AVAILABLE, reason="Weaviate client not installed")
    def test_weaviate_available(self):
        """Test that weaviate library is detected when installed."""
        assert WEAVIATE_AVAILABLE is True


class TestHybridSearchConfig:
    """Tests for HybridSearchConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = HybridSearchConfig()
        assert config.vector_weight == 0.7
        assert config.keyword_weight == 0.3
        assert config.rrf_k == 60
        assert config.vector_limit == 50
        assert config.keyword_limit == 50

    def test_custom_weights(self):
        """Test custom weight configuration."""
        config = HybridSearchConfig(
            vector_weight=0.5,
            keyword_weight=0.5,
        )
        assert config.vector_weight == 0.5
        assert config.keyword_weight == 0.5

    def test_score_thresholds(self):
        """Test score threshold configuration."""
        config = HybridSearchConfig(
            min_vector_score=0.5,
            min_keyword_score=0.3,
            min_combined_score=0.4,
        )
        assert config.min_vector_score == 0.5
        assert config.min_combined_score == 0.4


class TestHybridResult:
    """Tests for HybridResult dataclass."""

    def test_create_hybrid_result(self):
        """Test creating a hybrid result."""
        result = HybridResult(
            chunk_id="chunk-1",
            document_id="doc-1",
            content="Test content",
            combined_score=0.85,
            vector_score=0.9,
            keyword_score=0.7,
        )
        assert result.chunk_id == "chunk-1"
        assert result.combined_score == 0.85
        assert result.vector_score == 0.9
        assert result.keyword_score == 0.7

    def test_to_dict(self):
        """Test serialization to dictionary."""
        result = HybridResult(
            chunk_id="c1",
            document_id="d1",
            content="Test",
            combined_score=0.8,
            vector_score=0.85,
            keyword_score=0.75,
        )
        data = result.to_dict()
        assert data["combined_score"] == 0.8
        assert data["vector_score"] == 0.85
        assert data["keyword_score"] == 0.75


class TestHybridSearcher:
    """Tests for HybridSearcher class (with mocked dependencies)."""

    @pytest.fixture
    def mock_store(self):
        """Create a mock Weaviate store."""
        store = MagicMock(spec=WeaviateStore)
        store.is_connected = True
        return store

    @pytest.fixture
    def mock_embedder(self):
        """Create a mock embedder."""
        embedder = MagicMock()
        embedder.embed = AsyncMock(return_value=[0.1] * 1536)
        embedder.embed_batch = AsyncMock(return_value=[[0.1] * 1536])
        return embedder

    @pytest.fixture
    def sample_vector_results(self):
        """Sample vector search results."""
        return [
            SearchResult(
                chunk_id="c1",
                document_id="d1",
                content="First result",
                score=0.95,
            ),
            SearchResult(
                chunk_id="c2",
                document_id="d1",
                content="Second result",
                score=0.85,
            ),
        ]

    @pytest.fixture
    def sample_keyword_results(self):
        """Sample keyword search results."""
        return [
            SearchResult(
                chunk_id="c2",
                document_id="d1",
                content="Second result",
                score=5.0,  # BM25 scores can be > 1
            ),
            SearchResult(
                chunk_id="c3",
                document_id="d1",
                content="Third result",
                score=3.0,
            ),
        ]

    @pytest.mark.asyncio
    async def test_hybrid_search(
        self, mock_store, mock_embedder, sample_vector_results, sample_keyword_results
    ):
        """Test hybrid search combining vector and keyword results."""
        mock_store.search_vector = AsyncMock(return_value=sample_vector_results)
        mock_store.search_keyword = AsyncMock(return_value=sample_keyword_results)

        searcher = HybridSearcher(mock_store, mock_embedder)
        results = await searcher.search("test query", limit=10)

        assert len(results) >= 1
        # c2 should rank high as it appears in both
        chunk_ids = [r.chunk_id for r in results]
        assert "c2" in chunk_ids

    @pytest.mark.asyncio
    async def test_rrf_scoring(
        self, mock_store, mock_embedder, sample_vector_results, sample_keyword_results
    ):
        """Test RRF scoring combines results properly."""
        mock_store.search_vector = AsyncMock(return_value=sample_vector_results)
        mock_store.search_keyword = AsyncMock(return_value=sample_keyword_results)

        config = HybridSearchConfig(vector_weight=0.7, keyword_weight=0.3)
        searcher = HybridSearcher(mock_store, mock_embedder, config)

        results = await searcher.search("test query")

        # Results should have combined scores
        for result in results:
            assert result.combined_score > 0

    @pytest.mark.asyncio
    async def test_vector_only_search(self, mock_store, mock_embedder, sample_vector_results):
        """Test vector-only search."""
        mock_store.search_vector = AsyncMock(return_value=sample_vector_results)

        searcher = HybridSearcher(mock_store, mock_embedder)
        results = await searcher.search_vector_only("test query")

        assert len(results) == 2
        assert results[0].score == 0.95

    @pytest.mark.asyncio
    async def test_keyword_only_search(self, mock_store, mock_embedder, sample_keyword_results):
        """Test keyword-only search."""
        mock_store.search_keyword = AsyncMock(return_value=sample_keyword_results)

        searcher = HybridSearcher(mock_store, mock_embedder)
        results = await searcher.search_keyword_only("test query")

        assert len(results) == 2
        assert results[0].score == 5.0

    @pytest.mark.asyncio
    async def test_document_filter(self, mock_store, mock_embedder):
        """Test filtering by document IDs."""
        mock_store.search_vector = AsyncMock(return_value=[])
        mock_store.search_keyword = AsyncMock(return_value=[])

        searcher = HybridSearcher(mock_store, mock_embedder)
        await searcher.search("test", document_ids=["doc1", "doc2"])

        # Verify filter was passed
        mock_store.search_vector.assert_called_once()
        call_args = mock_store.search_vector.call_args
        assert call_args.kwargs["document_ids"] == ["doc1", "doc2"]


class TestSimpleEmbedder:
    """Tests for SimpleEmbedder class."""

    def test_initialization(self):
        """Test embedder initialization."""
        embedder = SimpleEmbedder(model="text-embedding-3-small")
        assert embedder.model == "text-embedding-3-small"

    def test_fallback_embed(self):
        """Test fallback embedding method."""
        embedder = SimpleEmbedder()
        embedding = embedder._fallback_embed("test text")

        # Should return a normalized vector
        assert len(embedding) == 1536
        assert isinstance(embedding[0], float)

        # Check normalization (sum of squares should be close to 1)
        norm = sum(x * x for x in embedding) ** 0.5
        assert abs(norm - 1.0) < 0.01 or norm == 0  # Either normalized or zero

    def test_fallback_produces_different_vectors(self):
        """Test that different texts produce different embeddings."""
        embedder = SimpleEmbedder()
        emb1 = embedder._fallback_embed("hello world")
        emb2 = embedder._fallback_embed("goodbye world")

        # Should be different vectors
        assert emb1 != emb2


class TestIntegration:
    """Integration tests (require mocking or real Weaviate)."""

    @pytest.mark.asyncio
    async def test_search_result_flow(self):
        """Test the flow from search to result processing."""
        # Create mock data
        vector_results = [
            SearchResult(
                chunk_id="security-1",
                document_id="policy-doc",
                content="Security policy requires encryption at rest.",
                score=0.92,
                heading_context="Security Requirements",
                start_page=5,
                end_page=5,
            )
        ]
        keyword_results = [
            SearchResult(
                chunk_id="security-1",
                document_id="policy-doc",
                content="Security policy requires encryption at rest.",
                score=4.5,
            )
        ]

        # Create mocked searcher
        mock_store = MagicMock()
        mock_store.is_connected = True
        mock_store.search_vector = AsyncMock(return_value=vector_results)
        mock_store.search_keyword = AsyncMock(return_value=keyword_results)

        mock_embedder = MagicMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 1536)

        searcher = HybridSearcher(mock_store, mock_embedder)
        results = await searcher.search("encryption security")

        assert len(results) == 1
        assert results[0].chunk_id == "security-1"
        assert results[0].heading_context == "Security Requirements"
        # Should have both vector and keyword scores
        assert results[0].vector_score > 0
        assert results[0].keyword_score > 0
