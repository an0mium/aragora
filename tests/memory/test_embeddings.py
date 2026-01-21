"""
Tests for aragora.memory.embeddings - Semantic retrieval using embeddings.

Tests cover:
- EmbeddingProvider base class (hash-based fallback)
- OpenAIEmbedding (mocked API)
- GeminiEmbedding (mocked API)
- OllamaEmbedding (mocked API)
- SemanticRetriever (database operations)
- Embedding cache (get/set/stats)
- Utility functions (cosine_similarity, pack/unpack)
"""

from __future__ import annotations

import asyncio
import struct
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.memory.embeddings import (
    EmbeddingProvider,
    GeminiEmbedding,
    OllamaEmbedding,
    OpenAIEmbedding,
    SemanticRetriever,
    cosine_similarity,
    get_embedding_cache,
    get_embedding_cache_stats,
    pack_embedding,
    unpack_embedding,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def temp_db():
    """Create a temporary database file."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        yield f.name
    # Cleanup handled by tempfile


@pytest.fixture
def mock_aiohttp_session():
    """Create a mock aiohttp session."""
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_response)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    return mock_session, mock_response


@pytest.fixture
def force_hash_provider():
    """Force fallback to hash-based provider by removing API keys.

    This avoids rate limiting issues when running tests in parallel.
    """
    with patch.dict("os.environ", {}, clear=True):
        with patch("socket.socket") as mock_socket:
            mock_sock = MagicMock()
            mock_sock.connect_ex.return_value = 1  # Ollama not available
            mock_socket.return_value.__enter__ = MagicMock(return_value=mock_sock)
            mock_socket.return_value.__exit__ = MagicMock(return_value=None)
            yield


# ===========================================================================
# Utility Function Tests
# ===========================================================================


class TestCosingSimilarity:
    """Tests for cosine_similarity function."""

    def test_identical_vectors(self):
        """Identical vectors have similarity 1.0."""
        v = [1.0, 2.0, 3.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        """Orthogonal vectors have similarity 0.0."""
        v1 = [1.0, 0.0, 0.0]
        v2 = [0.0, 1.0, 0.0]
        assert cosine_similarity(v1, v2) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        """Opposite vectors have similarity -1.0."""
        v1 = [1.0, 0.0, 0.0]
        v2 = [-1.0, 0.0, 0.0]
        assert cosine_similarity(v1, v2) == pytest.approx(-1.0)

    def test_similar_vectors(self):
        """Similar vectors have positive similarity."""
        v1 = [1.0, 2.0, 3.0]
        v2 = [1.1, 2.1, 3.1]
        sim = cosine_similarity(v1, v2)
        assert 0.99 < sim <= 1.0

    def test_empty_vectors(self):
        """Empty vectors return 0.0 (avoid division by zero)."""
        v1 = []
        v2 = []
        # Should handle gracefully
        try:
            result = cosine_similarity(v1, v2)
            assert result == 0.0 or result != result  # 0.0 or NaN
        except (ZeroDivisionError, ValueError):
            pass  # Also acceptable


class TestPackUnpackEmbedding:
    """Tests for pack_embedding and unpack_embedding functions."""

    def test_pack_unpack_roundtrip(self):
        """Pack and unpack should be inverses."""
        original = [0.1, 0.2, 0.3, 0.4, 0.5]
        packed = pack_embedding(original)
        unpacked = unpack_embedding(packed)
        assert len(unpacked) == len(original)
        for a, b in zip(original, unpacked):
            assert a == pytest.approx(b)

    def test_pack_returns_bytes(self):
        """pack_embedding returns bytes."""
        embedding = [1.0, 2.0, 3.0]
        packed = pack_embedding(embedding)
        assert isinstance(packed, bytes)

    def test_unpack_returns_list(self):
        """unpack_embedding returns list of floats."""
        embedding = [1.0, 2.0, 3.0]
        packed = pack_embedding(embedding)
        unpacked = unpack_embedding(packed)
        assert isinstance(unpacked, list)
        assert all(isinstance(x, float) for x in unpacked)

    def test_empty_embedding(self):
        """Handle empty embedding."""
        packed = pack_embedding([])
        unpacked = unpack_embedding(packed)
        assert unpacked == []

    def test_large_embedding(self):
        """Handle large embeddings (1536 dimensions like OpenAI)."""
        large = [float(i) / 1000 for i in range(1536)]
        packed = pack_embedding(large)
        unpacked = unpack_embedding(packed)
        assert len(unpacked) == 1536


# ===========================================================================
# EmbeddingProvider Tests
# ===========================================================================


class TestEmbeddingProvider:
    """Tests for base EmbeddingProvider class."""

    def test_default_dimension(self):
        """Default dimension is 256."""
        provider = EmbeddingProvider()
        assert provider.dimension == 256

    def test_custom_dimension(self):
        """Can set custom dimension."""
        provider = EmbeddingProvider(dimension=512)
        assert provider.dimension == 512

    @pytest.mark.asyncio
    async def test_embed_returns_list(self):
        """embed() returns list of floats."""
        provider = EmbeddingProvider()
        result = await provider.embed("test text")
        assert isinstance(result, list)
        assert len(result) == 256
        assert all(isinstance(x, float) for x in result)

    @pytest.mark.asyncio
    async def test_embed_deterministic(self):
        """Same input produces same embedding (hash-based)."""
        provider = EmbeddingProvider()
        result1 = await provider.embed("test text")
        result2 = await provider.embed("test text")
        assert result1 == result2

    @pytest.mark.asyncio
    async def test_embed_different_inputs(self):
        """Different inputs produce different embeddings."""
        provider = EmbeddingProvider()
        result1 = await provider.embed("text one")
        result2 = await provider.embed("text two")
        assert result1 != result2

    @pytest.mark.asyncio
    async def test_embed_values_in_range(self):
        """Embedding values are in [-1, 1] range."""
        provider = EmbeddingProvider()
        result = await provider.embed("test text")
        assert all(-1.0 <= x <= 1.0 for x in result)

    @pytest.mark.asyncio
    async def test_embed_batch(self):
        """embed_batch() returns list of embeddings."""
        provider = EmbeddingProvider()
        texts = ["text one", "text two", "text three"]
        results = await provider.embed_batch(texts)
        assert len(results) == 3
        assert all(len(emb) == 256 for emb in results)

    @pytest.mark.asyncio
    async def test_embed_batch_preserves_order(self):
        """embed_batch() maintains alignment with inputs."""
        provider = EmbeddingProvider()
        texts = ["aaa", "bbb", "ccc"]
        results = await provider.embed_batch(texts)

        # Each should match individual embed
        for text, batch_result in zip(texts, results):
            single_result = await provider.embed(text)
            assert batch_result == single_result


# ===========================================================================
# OpenAIEmbedding Tests
# ===========================================================================


class TestOpenAIEmbedding:
    """Tests for OpenAI embedding provider."""

    def test_initialization(self):
        """Test initialization with API key."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            provider = OpenAIEmbedding()
            assert provider.api_key == "test-key"
            assert provider.model == "text-embedding-3-small"
            assert provider.dimension == 1536

    def test_custom_model(self):
        """Test initialization with custom model."""
        provider = OpenAIEmbedding(api_key="key", model="text-embedding-ada-002")
        assert provider.model == "text-embedding-ada-002"

    @pytest.mark.asyncio
    async def test_embed_success(self):
        """Test successful embedding with mocked API."""
        mock_embedding = [0.1] * 1536

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"data": [{"embedding": mock_embedding}]})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            provider = OpenAIEmbedding(api_key="test-key")
            # Clear cache to ensure API call
            from aragora.memory.embeddings import _get_embedding_cache

            _get_embedding_cache()._cache.clear()

            result = await provider.embed("test text")

        assert result == mock_embedding

    @pytest.mark.asyncio
    async def test_embed_caches_result(self):
        """Test that embeddings are cached."""
        mock_embedding = [0.2] * 1536

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"data": [{"embedding": mock_embedding}]})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            provider = OpenAIEmbedding(api_key="test-key")
            from aragora.memory.embeddings import _get_embedding_cache

            _get_embedding_cache()._cache.clear()

            # First call - should hit API
            result1 = await provider.embed("cached text unique 123")
            # Second call - should use cache
            result2 = await provider.embed("cached text unique 123")

        assert result1 == result2
        # API should only be called once
        assert mock_session.post.call_count == 1

    @pytest.mark.asyncio
    async def test_embed_batch_success(self):
        """Test batch embedding with mocked API."""
        mock_embeddings = [[0.1] * 1536, [0.2] * 1536]

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "data": [
                    {"embedding": mock_embeddings[0], "index": 0},
                    {"embedding": mock_embeddings[1], "index": 1},
                ]
            }
        )
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            provider = OpenAIEmbedding(api_key="test-key")
            result = await provider.embed_batch(["text1", "text2"])

        assert result == mock_embeddings


# ===========================================================================
# GeminiEmbedding Tests
# ===========================================================================


class TestGeminiEmbedding:
    """Tests for Gemini embedding provider."""

    def test_initialization(self):
        """Test initialization with API key."""
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
            provider = GeminiEmbedding()
            assert provider.api_key == "test-key"
            assert provider.model == "text-embedding-004"
            assert provider.dimension == 768

    def test_google_api_key_fallback(self):
        """Test fallback to GOOGLE_API_KEY."""
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "google-key"}, clear=True):
            provider = GeminiEmbedding()
            assert provider.api_key == "google-key"

    @pytest.mark.asyncio
    async def test_embed_success(self):
        """Test successful embedding with mocked API."""
        mock_embedding = [0.1] * 768

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"embedding": {"values": mock_embedding}})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            provider = GeminiEmbedding(api_key="test-key")
            from aragora.memory.embeddings import _get_embedding_cache

            _get_embedding_cache()._cache.clear()

            result = await provider.embed("test text gemini unique")

        assert result == mock_embedding


# ===========================================================================
# OllamaEmbedding Tests
# ===========================================================================


class TestOllamaEmbedding:
    """Tests for Ollama embedding provider."""

    def test_initialization_defaults(self):
        """Test default initialization."""
        provider = OllamaEmbedding()
        assert provider.model == "nomic-embed-text"
        assert provider.base_url == "http://localhost:11434"
        assert provider.dimension == 768

    def test_custom_url(self):
        """Test custom base URL."""
        provider = OllamaEmbedding(base_url="http://custom:8080")
        assert provider.base_url == "http://custom:8080"

    def test_env_url(self):
        """Test URL from environment."""
        with patch.dict("os.environ", {"OLLAMA_HOST": "http://env-host:11434"}):
            provider = OllamaEmbedding()
            assert provider.base_url == "http://env-host:11434"

    @pytest.mark.asyncio
    async def test_embed_success(self):
        """Test successful embedding with mocked API."""
        mock_embedding = [0.1] * 768

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"embedding": mock_embedding})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            provider = OllamaEmbedding()
            result = await provider.embed("test text")

        assert result == mock_embedding


# ===========================================================================
# SemanticRetriever Tests
# ===========================================================================


class TestSemanticRetriever:
    """Tests for SemanticRetriever class."""

    def test_initialization(self, temp_db):
        """Test retriever initialization creates tables."""
        retriever = SemanticRetriever(temp_db)
        assert retriever.db_path == Path(temp_db)
        assert retriever.provider is not None

    def test_auto_detect_provider_openai(self, temp_db):
        """Test auto-detection prefers OpenAI."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            retriever = SemanticRetriever(temp_db)
            assert isinstance(retriever.provider, OpenAIEmbedding)

    def test_auto_detect_provider_gemini(self, temp_db):
        """Test auto-detection uses Gemini when OpenAI not available."""
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}, clear=True):
            retriever = SemanticRetriever(temp_db)
            assert isinstance(retriever.provider, GeminiEmbedding)

    def test_auto_detect_provider_fallback(self, temp_db):
        """Test fallback to hash-based provider."""
        with patch.dict("os.environ", {}, clear=True):
            # Mock socket to indicate Ollama is not available
            with patch("socket.socket") as mock_socket:
                mock_sock = MagicMock()
                mock_sock.connect_ex.return_value = 1  # Connection failed
                mock_socket.return_value.__enter__ = MagicMock(return_value=mock_sock)
                mock_socket.return_value.__exit__ = MagicMock(return_value=None)

                retriever = SemanticRetriever(temp_db)
                assert isinstance(retriever.provider, EmbeddingProvider)
                assert retriever.provider.dimension == 256

    def test_text_hash_deterministic(self, temp_db):
        """Test text hash is deterministic."""
        retriever = SemanticRetriever(temp_db)
        hash1 = retriever._text_hash("test text")
        hash2 = retriever._text_hash("test text")
        assert hash1 == hash2

    def test_text_hash_case_insensitive(self, temp_db):
        """Test text hash is case-insensitive."""
        retriever = SemanticRetriever(temp_db)
        hash1 = retriever._text_hash("Test Text")
        hash2 = retriever._text_hash("test text")
        assert hash1 == hash2

    @pytest.mark.asyncio
    async def test_embed_and_store(self, temp_db, force_hash_provider):
        """Test embedding and storing text."""
        retriever = SemanticRetriever(temp_db)

        embedding = await retriever.embed_and_store("id1", "test text for storage")

        assert isinstance(embedding, list)
        assert len(embedding) == retriever.provider.dimension

    @pytest.mark.asyncio
    async def test_embed_and_store_deduplicates(self, temp_db, force_hash_provider):
        """Test that identical texts are not re-embedded."""
        retriever = SemanticRetriever(temp_db)

        # Store same text with different IDs
        emb1 = await retriever.embed_and_store("id1", "duplicate text")
        emb2 = await retriever.embed_and_store("id2", "duplicate text")

        # Should return same embedding (with float32 precision tolerance from pack/unpack)
        assert len(emb1) == len(emb2)
        for a, b in zip(emb1, emb2):
            assert a == pytest.approx(b, rel=1e-6)

    @pytest.mark.asyncio
    async def test_find_similar(self, temp_db, force_hash_provider):
        """Test finding similar texts."""
        retriever = SemanticRetriever(temp_db)

        # Store some texts
        await retriever.embed_and_store("id1", "the cat sat on the mat")
        await retriever.embed_and_store("id2", "the dog ran in the park")
        await retriever.embed_and_store("id3", "the cat played with yarn")

        # Find similar to cat-related query
        results = await retriever.find_similar("cat behavior", limit=3, min_similarity=0.0)

        assert isinstance(results, list)
        # Results should be sorted by similarity
        if len(results) > 1:
            assert results[0][2] >= results[1][2]

    @pytest.mark.asyncio
    async def test_find_similar_respects_limit(self, temp_db, force_hash_provider):
        """Test that find_similar respects limit parameter."""
        retriever = SemanticRetriever(temp_db)

        # Store multiple texts
        for i in range(10):
            await retriever.embed_and_store(f"id{i}", f"text number {i}")

        results = await retriever.find_similar("text", limit=3, min_similarity=0.0)
        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_find_similar_respects_min_similarity(self, temp_db, force_hash_provider):
        """Test that find_similar respects min_similarity parameter."""
        retriever = SemanticRetriever(temp_db)

        await retriever.embed_and_store("id1", "completely unrelated topic")

        # With very high threshold, should get no results
        results = await retriever.find_similar("different subject", limit=10, min_similarity=0.99)
        # May or may not have results depending on hash-based similarity
        for _, _, similarity in results:
            assert similarity >= 0.99

    @pytest.mark.asyncio
    async def test_find_similar_empty_db(self, temp_db, force_hash_provider):
        """Test find_similar on empty database."""
        retriever = SemanticRetriever(temp_db)

        results = await retriever.find_similar("query", limit=5)
        assert results == []

    def test_get_stats(self, temp_db):
        """Test getting statistics."""
        retriever = SemanticRetriever(temp_db)
        stats = retriever.get_stats()

        assert "total_embeddings" in stats
        assert "by_provider" in stats
        assert stats["total_embeddings"] == 0

    @pytest.mark.asyncio
    async def test_get_stats_after_storage(self, temp_db, force_hash_provider):
        """Test stats after storing embeddings."""
        retriever = SemanticRetriever(temp_db)

        await retriever.embed_and_store("id1", "text one")
        await retriever.embed_and_store("id2", "text two")

        stats = retriever.get_stats()
        assert stats["total_embeddings"] == 2


# ===========================================================================
# Embedding Cache Tests
# ===========================================================================


class TestEmbeddingCache:
    """Tests for embedding cache functions."""

    def test_get_embedding_cache(self):
        """Test getting global cache."""
        cache = get_embedding_cache()
        assert cache is not None

    def test_cache_singleton(self):
        """Test cache is singleton."""
        cache1 = get_embedding_cache()
        cache2 = get_embedding_cache()
        assert cache1 is cache2

    def test_get_embedding_cache_stats(self):
        """Test getting cache stats."""
        stats = get_embedding_cache_stats()

        assert "size" in stats
        assert "valid" in stats
        assert "ttl_seconds" in stats
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats

    def test_cache_set_get(self):
        """Test setting and getting from cache."""
        cache = get_embedding_cache()
        test_embedding = [0.1, 0.2, 0.3]

        cache.set("test_key_unique_12345", test_embedding)
        result = cache.get("test_key_unique_12345")

        assert result == test_embedding

    def test_cache_miss(self):
        """Test cache miss returns None."""
        cache = get_embedding_cache()
        result = cache.get("nonexistent_key_xyz")
        assert result is None


# ===========================================================================
# Integration Tests
# ===========================================================================


class TestEmbeddingIntegration:
    """Integration tests for embedding workflow."""

    @pytest.mark.asyncio
    async def test_full_workflow(self, temp_db, force_hash_provider):
        """Test complete embedding and retrieval workflow."""
        retriever = SemanticRetriever(temp_db)

        # Store documents
        docs = [
            ("doc1", "Python is a programming language"),
            ("doc2", "JavaScript runs in browsers"),
            ("doc3", "Python supports multiple paradigms"),
        ]

        for doc_id, text in docs:
            await retriever.embed_and_store(doc_id, text)

        # Search
        results = await retriever.find_similar("Python programming", limit=2, min_similarity=0.0)

        assert len(results) <= 2
        # All results should have required fields
        for doc_id, text, similarity in results:
            assert isinstance(doc_id, str)
            assert isinstance(text, str)
            assert isinstance(similarity, float)

    @pytest.mark.asyncio
    async def test_provider_fallback_chain(self, temp_db):
        """Test that providers fall back gracefully."""
        # Clear environment to test fallback
        with patch.dict("os.environ", {}, clear=True):
            with patch("socket.socket") as mock_socket:
                mock_sock = MagicMock()
                mock_sock.connect_ex.return_value = 1  # Ollama not available
                mock_socket.return_value.__enter__ = MagicMock(return_value=mock_sock)
                mock_socket.return_value.__exit__ = MagicMock(return_value=None)

                retriever = SemanticRetriever(temp_db)

                # Should still work with hash-based fallback
                embedding = await retriever.embed_and_store("id1", "test")
                assert len(embedding) == 256  # Hash-based dimension

    @pytest.mark.asyncio
    async def test_concurrent_embeddings(self, temp_db, force_hash_provider):
        """Test concurrent embedding operations."""
        retriever = SemanticRetriever(temp_db)

        # Create multiple embedding tasks
        tasks = [retriever.embed_and_store(f"id{i}", f"text {i}") for i in range(10)]

        # Run concurrently
        results = await asyncio.gather(*tasks)

        assert len(results) == 10
        assert all(isinstance(r, list) for r in results)

        # Verify all stored
        stats = retriever.get_stats()
        assert stats["total_embeddings"] == 10
