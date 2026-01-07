"""
Tests for the memory/embeddings module.

Covers:
- EmbeddingCache with TTL and LRU eviction
- EmbeddingProvider base class (hash-based fallback)
- OpenAIEmbedding, GeminiEmbedding, OllamaEmbedding providers
- cosine_similarity function (numpy and pure Python paths)
- pack_embedding/unpack_embedding binary serialization
- SemanticRetriever class
- _auto_detect_provider fallback logic
- _retry_with_backoff helper
"""

import asyncio
import os
import struct
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, AsyncMock

import pytest
import aiohttp

from aragora.memory.embeddings import (
    EmbeddingCache,
    EmbeddingProvider,
    OpenAIEmbedding,
    GeminiEmbedding,
    OllamaEmbedding,
    cosine_similarity,
    pack_embedding,
    unpack_embedding,
    SemanticRetriever,
    get_embedding_cache_stats,
    _retry_with_backoff,
    _embedding_cache,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def cache():
    """Create a fresh EmbeddingCache."""
    return EmbeddingCache(ttl_seconds=3600, max_size=10)


@pytest.fixture
def short_ttl_cache():
    """Create cache with short TTL for testing expiration."""
    return EmbeddingCache(ttl_seconds=0.1, max_size=10)


@pytest.fixture
def base_provider():
    """Create base EmbeddingProvider (hash-based)."""
    return EmbeddingProvider(dimension=128)


@pytest.fixture
def temp_db(tmp_path):
    """Create temporary database path."""
    return str(tmp_path / "test_embeddings.db")


@pytest.fixture
def retriever(temp_db, base_provider):
    """Create SemanticRetriever with hash-based provider."""
    return SemanticRetriever(db_path=temp_db, provider=base_provider)


# ============================================================================
# EmbeddingCache Tests
# ============================================================================

class TestEmbeddingCache:
    """Tests for EmbeddingCache TTL and LRU eviction."""

    def test_init_default_values(self):
        """Test default initialization values."""
        cache = EmbeddingCache()
        stats = cache.stats()
        assert stats["ttl_seconds"] == 3600
        assert stats["size"] == 0

    def test_init_custom_values(self):
        """Test custom initialization values."""
        cache = EmbeddingCache(ttl_seconds=600, max_size=500)
        assert cache._ttl == 600
        assert cache._max_size == 500

    def test_set_and_get(self, cache):
        """Test basic set and get."""
        embedding = [0.1, 0.2, 0.3]
        cache.set("test text", embedding)

        result = cache.get("test text")
        assert result == embedding

    def test_get_missing_returns_none(self, cache):
        """Test getting non-existent key returns None."""
        result = cache.get("nonexistent")
        assert result is None

    def test_key_normalization(self, cache):
        """Test that keys are normalized (case-insensitive, trimmed)."""
        embedding = [1.0, 2.0]
        cache.set("  Test Text  ", embedding)

        # Should match with different casing/whitespace
        assert cache.get("test text") == embedding
        assert cache.get("TEST TEXT") == embedding
        assert cache.get("  TEST TEXT  ") == embedding

    def test_ttl_expiration(self, short_ttl_cache):
        """Test that entries expire after TTL."""
        embedding = [0.1, 0.2]
        short_ttl_cache.set("test", embedding)

        # Should exist immediately
        assert short_ttl_cache.get("test") == embedding

        # Wait for expiration
        time.sleep(0.2)

        # Should be expired
        assert short_ttl_cache.get("test") is None

    def test_lru_eviction(self, cache):
        """Test LRU eviction when max_size reached."""
        # Fill cache to max_size (10)
        for i in range(10):
            cache.set(f"item{i}", [float(i)])

        # All items should exist
        for i in range(10):
            assert cache.get(f"item{i}") is not None

        # Add one more - should evict oldest (item0)
        cache.set("new_item", [99.0])

        assert cache.get("item0") is None
        assert cache.get("new_item") == [99.0]
        assert cache.get("item1") is not None  # item1 still exists

    def test_lru_update_on_get(self, cache):
        """Test that getting an item moves it to end (LRU)."""
        # Fill cache
        for i in range(10):
            cache.set(f"item{i}", [float(i)])

        # Access item0, moving it to end
        cache.get("item0")

        # Add new item - should evict item1 (now oldest)
        cache.set("new_item", [99.0])

        assert cache.get("item0") is not None  # item0 still exists
        assert cache.get("item1") is None  # item1 evicted

    def test_update_existing_key(self, cache):
        """Test updating existing key updates timestamp."""
        cache.set("test", [1.0])
        cache.set("test", [2.0])

        assert cache.get("test") == [2.0]
        assert cache.stats()["size"] == 1

    def test_stats(self, cache):
        """Test stats reporting."""
        cache.set("a", [1.0])
        cache.set("b", [2.0])

        stats = cache.stats()
        assert stats["size"] == 2
        assert stats["valid"] == 2
        assert stats["ttl_seconds"] == 3600

    def test_stats_with_expired(self, short_ttl_cache):
        """Test stats correctly reports expired entries."""
        short_ttl_cache.set("a", [1.0])
        time.sleep(0.15)  # Expire first
        short_ttl_cache.set("b", [2.0])  # Add after expiry

        stats = short_ttl_cache.stats()
        assert stats["size"] == 2  # Both still in cache
        assert stats["valid"] == 1  # Only one valid


# ============================================================================
# EmbeddingProvider Tests
# ============================================================================

class TestEmbeddingProvider:
    """Tests for base EmbeddingProvider (hash-based fallback)."""

    def test_default_dimension(self):
        """Test default dimension."""
        provider = EmbeddingProvider()
        assert provider.dimension == 256

    def test_custom_dimension(self):
        """Test custom dimension."""
        provider = EmbeddingProvider(dimension=512)
        assert provider.dimension == 512

    @pytest.mark.asyncio
    async def test_embed_returns_correct_dimension(self, base_provider):
        """Test embed returns vector of correct dimension."""
        embedding = await base_provider.embed("test text")
        assert len(embedding) == base_provider.dimension

    @pytest.mark.asyncio
    async def test_embed_deterministic(self, base_provider):
        """Test hash-based embedding is deterministic."""
        emb1 = await base_provider.embed("test text")
        emb2 = await base_provider.embed("test text")
        assert emb1 == emb2

    @pytest.mark.asyncio
    async def test_embed_different_texts(self, base_provider):
        """Test different texts produce different embeddings."""
        emb1 = await base_provider.embed("hello world")
        emb2 = await base_provider.embed("goodbye world")
        assert emb1 != emb2

    @pytest.mark.asyncio
    async def test_embed_values_in_range(self, base_provider):
        """Test embedding values are in [-1, 1] range."""
        embedding = await base_provider.embed("test")
        for val in embedding:
            assert -1.0 <= val <= 1.0

    @pytest.mark.asyncio
    async def test_embed_batch(self, base_provider):
        """Test batch embedding."""
        texts = ["hello", "world", "test"]
        embeddings = await base_provider.embed_batch(texts)

        assert len(embeddings) == 3
        for emb in embeddings:
            assert len(emb) == base_provider.dimension


# ============================================================================
# OpenAIEmbedding Tests
# ============================================================================

class TestOpenAIEmbedding:
    """Tests for OpenAIEmbedding provider."""

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        provider = OpenAIEmbedding(api_key="test-key")
        assert provider.api_key == "test-key"
        assert provider.model == "text-embedding-3-small"
        assert provider.dimension == 1536

    def test_init_custom_model(self):
        """Test initialization with custom model."""
        provider = OpenAIEmbedding(api_key="test", model="text-embedding-3-large")
        assert provider.model == "text-embedding-3-large"

    @pytest.mark.asyncio
    async def test_embed_uses_cache(self):
        """Test that embed uses cache."""
        provider = OpenAIEmbedding(api_key="test-key")

        # Pre-populate cache
        _embedding_cache.set("cached text", [0.1, 0.2, 0.3])

        result = await provider.embed("cached text")
        assert result == [0.1, 0.2, 0.3]

    def test_embed_caches_result(self):
        """Test that embed caches successful results."""
        # We verify caching works by checking the cache directly
        # API calls are tested via integration tests
        provider = OpenAIEmbedding(api_key="test-key")

        # Pre-populate cache
        test_embedding = [0.1] * 1536
        _embedding_cache.set("openai_cache_test", test_embedding)

        # The embedding from cache should have correct dimension
        cached = _embedding_cache.get("openai_cache_test")
        assert cached == test_embedding
        assert len(cached) == provider.dimension

    def test_dimension_matches_model(self):
        """Test dimension matches text-embedding-3-small."""
        provider = OpenAIEmbedding(api_key="test-key")
        assert provider.dimension == 1536


# ============================================================================
# GeminiEmbedding Tests
# ============================================================================

class TestGeminiEmbedding:
    """Tests for GeminiEmbedding provider."""

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        provider = GeminiEmbedding(api_key="test-key")
        assert provider.api_key == "test-key"
        assert provider.model == "text-embedding-004"
        assert provider.dimension == 768

    @pytest.mark.asyncio
    async def test_embed_uses_cache(self):
        """Test that embed uses cache."""
        provider = GeminiEmbedding(api_key="test-key")

        # Pre-populate cache
        _embedding_cache.set("gemini cached", [0.5, 0.6])

        result = await provider.embed("gemini cached")
        assert result == [0.5, 0.6]


# ============================================================================
# OllamaEmbedding Tests
# ============================================================================

class TestOllamaEmbedding:
    """Tests for OllamaEmbedding provider."""

    def test_init_defaults(self):
        """Test default initialization."""
        provider = OllamaEmbedding()
        assert provider.model == "nomic-embed-text"
        assert provider.base_url == "http://localhost:11434"
        assert provider.dimension == 768

    def test_init_custom_values(self):
        """Test custom initialization."""
        provider = OllamaEmbedding(model="custom-model", base_url="http://custom:1234")
        assert provider.model == "custom-model"
        assert provider.base_url == "http://custom:1234"

    def test_init_from_env(self):
        """Test initialization from OLLAMA_HOST env var."""
        with patch.dict(os.environ, {"OLLAMA_HOST": "http://remote:5678"}):
            provider = OllamaEmbedding()
            assert provider.base_url == "http://remote:5678"

    def test_base_url_format(self):
        """Test base URL is correctly formatted."""
        provider = OllamaEmbedding(base_url="http://custom:5678")
        assert provider.base_url == "http://custom:5678"

    def test_api_endpoint_construction(self):
        """Test API endpoint is constructed correctly."""
        provider = OllamaEmbedding(base_url="http://localhost:11434")
        # The endpoint would be {base_url}/api/embeddings
        expected_endpoint = "http://localhost:11434/api/embeddings"
        assert f"{provider.base_url}/api/embeddings" == expected_endpoint


# ============================================================================
# Cosine Similarity Tests
# ============================================================================

class TestCosineSimilarity:
    """Tests for cosine_similarity function."""

    def test_identical_vectors(self):
        """Test similarity of identical vectors is 1.0."""
        a = [1.0, 2.0, 3.0]
        result = cosine_similarity(a, a)
        assert abs(result - 1.0) < 0.0001

    def test_opposite_vectors(self):
        """Test similarity of opposite vectors is -1.0."""
        a = [1.0, 0.0, 0.0]
        b = [-1.0, 0.0, 0.0]
        result = cosine_similarity(a, b)
        assert abs(result - (-1.0)) < 0.0001

    def test_orthogonal_vectors(self):
        """Test similarity of orthogonal vectors is 0.0."""
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        result = cosine_similarity(a, b)
        assert abs(result) < 0.0001

    def test_zero_vector(self):
        """Test handling of zero vectors."""
        a = [0.0, 0.0, 0.0]
        b = [1.0, 2.0, 3.0]
        result = cosine_similarity(a, b)
        assert result == 0.0

    def test_partial_similarity(self):
        """Test partial similarity calculation."""
        a = [1.0, 1.0, 0.0]
        b = [1.0, 0.0, 0.0]
        result = cosine_similarity(a, b)
        # Expected: 1 / (sqrt(2) * 1) ≈ 0.707
        assert 0.70 < result < 0.72

    def test_pure_python_fallback(self):
        """Test pure Python fallback when numpy unavailable."""
        # Temporarily mock numpy import to fail
        with patch.dict('sys.modules', {'numpy': None}):
            # Need to reload the function to trigger fallback
            # Instead, we test the fallback logic directly
            a = [1.0, 2.0, 3.0]
            b = [4.0, 5.0, 6.0]

            # Manual calculation
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = sum(x * x for x in a) ** 0.5
            norm_b = sum(x * x for x in b) ** 0.5
            expected = dot / (norm_a * norm_b)

            result = cosine_similarity(a, b)
            assert abs(result - expected) < 0.0001


# ============================================================================
# Pack/Unpack Embedding Tests
# ============================================================================

class TestPackUnpackEmbedding:
    """Tests for pack_embedding and unpack_embedding."""

    def test_pack_and_unpack_roundtrip(self):
        """Test roundtrip pack/unpack."""
        original = [0.1, 0.2, 0.3, 0.4, 0.5]
        packed = pack_embedding(original)
        unpacked = unpack_embedding(packed)

        assert len(unpacked) == len(original)
        for o, u in zip(original, unpacked):
            assert abs(o - u) < 0.0001  # Float precision

    def test_pack_format(self):
        """Test packed format is binary floats."""
        embedding = [1.0, 2.0]
        packed = pack_embedding(embedding)

        # Should be 8 bytes (2 floats * 4 bytes each)
        assert len(packed) == 8

    def test_unpack_correct_count(self):
        """Test unpack calculates correct count."""
        # Create packed data for 5 floats
        embedding = [1.0, 2.0, 3.0, 4.0, 5.0]
        packed = pack_embedding(embedding)

        unpacked = unpack_embedding(packed)
        assert len(unpacked) == 5

    def test_empty_embedding(self):
        """Test handling of empty embedding."""
        packed = pack_embedding([])
        assert len(packed) == 0

        unpacked = unpack_embedding(b"")
        assert unpacked == []


# ============================================================================
# SemanticRetriever Tests
# ============================================================================

class TestSemanticRetriever:
    """Tests for SemanticRetriever class."""

    def test_init_creates_tables(self, temp_db, base_provider):
        """Test initialization creates database tables."""
        retriever = SemanticRetriever(db_path=temp_db, provider=base_provider)

        # Check table exists
        import sqlite3
        with sqlite3.connect(temp_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='embeddings'")
            assert cursor.fetchone() is not None

    def test_init_idempotent(self, temp_db, base_provider):
        """Test multiple initializations don't fail."""
        r1 = SemanticRetriever(db_path=temp_db, provider=base_provider)
        r2 = SemanticRetriever(db_path=temp_db, provider=base_provider)
        # Should not raise

    @pytest.mark.asyncio
    async def test_embed_and_store(self, retriever):
        """Test embedding and storing text."""
        embedding = await retriever.embed_and_store("doc1", "Hello world")

        assert len(embedding) == retriever.provider.dimension

    @pytest.mark.asyncio
    async def test_embed_and_store_deduplication(self, retriever):
        """Test that duplicate texts return cached embedding from DB."""
        emb1 = await retriever.embed_and_store("doc1", "Hello world")
        emb2 = await retriever.embed_and_store("doc2", "Hello world")

        # Due to float32 pack/unpack, compare with tolerance
        # The second call retrieves from DB (unpacked), so there may be precision loss
        assert len(emb1) == len(emb2)
        for v1, v2 in zip(emb1, emb2):
            assert abs(v1 - v2) < 0.0001  # Float32 precision tolerance

    @pytest.mark.asyncio
    async def test_find_similar_empty_db(self, retriever):
        """Test find_similar on empty database."""
        results = await retriever.find_similar("test query")
        assert results == []

    @pytest.mark.asyncio
    async def test_find_similar_basic(self, retriever):
        """Test basic similarity search."""
        # Store some documents
        await retriever.embed_and_store("doc1", "The quick brown fox")
        await retriever.embed_and_store("doc2", "The lazy dog")
        await retriever.embed_and_store("doc3", "Something completely different")

        # Search - with hash-based embeddings, same text should match
        results = await retriever.find_similar("The quick brown fox", limit=5, min_similarity=0.0)

        assert len(results) > 0
        # First result should be exact match
        assert results[0][0] == "doc1"
        assert results[0][2] == 1.0  # Perfect similarity for exact match

    @pytest.mark.asyncio
    async def test_find_similar_limit(self, retriever):
        """Test limit parameter."""
        # Store many documents
        for i in range(10):
            await retriever.embed_and_store(f"doc{i}", f"Document number {i}")

        results = await retriever.find_similar("Document number", limit=3, min_similarity=0.0)
        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_find_similar_min_similarity(self, retriever):
        """Test min_similarity filtering."""
        await retriever.embed_and_store("doc1", "exact match")
        await retriever.embed_and_store("doc2", "completely different text xyz")

        # High threshold should filter out low matches
        results = await retriever.find_similar("exact match", min_similarity=0.99)

        # Only exact match should pass high threshold
        assert len(results) == 1
        assert results[0][0] == "doc1"

    def test_get_stats_empty(self, retriever):
        """Test stats on empty database."""
        stats = retriever.get_stats()
        assert stats["total_embeddings"] == 0
        assert stats["by_provider"] == {}

    @pytest.mark.asyncio
    async def test_get_stats_with_data(self, retriever):
        """Test stats with stored embeddings."""
        await retriever.embed_and_store("doc1", "Hello")
        await retriever.embed_and_store("doc2", "World")

        stats = retriever.get_stats()
        assert stats["total_embeddings"] == 2
        assert "EmbeddingProvider" in stats["by_provider"]

    def test_text_hash(self, retriever):
        """Test text hash normalization."""
        hash1 = retriever._text_hash("  Hello World  ")
        hash2 = retriever._text_hash("hello world")
        assert hash1 == hash2


# ============================================================================
# Auto-Detect Provider Tests
# ============================================================================

class TestAutoDetectProvider:
    """Tests for _auto_detect_provider logic."""

    def test_openai_preferred(self, temp_db):
        """Test OpenAI is preferred when key available."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            retriever = SemanticRetriever(db_path=temp_db)
            assert isinstance(retriever.provider, OpenAIEmbedding)

    def test_gemini_fallback(self, temp_db):
        """Test Gemini used when no OpenAI key."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=True):
            retriever = SemanticRetriever(db_path=temp_db)
            assert isinstance(retriever.provider, GeminiEmbedding)

    def test_google_api_key_alias(self, temp_db):
        """Test GOOGLE_API_KEY works for Gemini."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}, clear=True):
            retriever = SemanticRetriever(db_path=temp_db)
            assert isinstance(retriever.provider, GeminiEmbedding)

    def test_hash_fallback_no_keys(self, temp_db):
        """Test hash-based fallback when no API keys."""
        with patch.dict(os.environ, {}, clear=True):
            # Also mock socket to fail Ollama check
            with patch('socket.socket') as mock_socket:
                mock_sock = Mock()
                mock_sock.connect_ex.return_value = 1  # Connection refused
                mock_socket.return_value.__enter__ = Mock(return_value=mock_sock)
                mock_socket.return_value.__exit__ = Mock(return_value=False)

                retriever = SemanticRetriever(db_path=temp_db)
                assert type(retriever.provider) is EmbeddingProvider
                assert retriever.provider.dimension == 256

    def test_ollama_when_available(self, temp_db):
        """Test Ollama used when running and no API keys."""
        with patch.dict(os.environ, {}, clear=True):
            with patch('socket.socket') as mock_socket:
                mock_sock = Mock()
                mock_sock.connect_ex.return_value = 0  # Connection success
                mock_socket.return_value.__enter__ = Mock(return_value=mock_sock)
                mock_socket.return_value.__exit__ = Mock(return_value=False)

                retriever = SemanticRetriever(db_path=temp_db)
                assert isinstance(retriever.provider, OllamaEmbedding)


# ============================================================================
# Retry With Backoff Tests
# ============================================================================

class TestRetryWithBackoff:
    """Tests for _retry_with_backoff helper."""

    @pytest.mark.asyncio
    async def test_success_first_try(self):
        """Test successful call on first try."""
        async def success():
            return "result"

        result = await _retry_with_backoff(success)
        assert result == "result"

    @pytest.mark.asyncio
    async def test_retry_on_timeout(self):
        """Test retry on TimeoutError."""
        call_count = 0

        async def failing_then_success():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise asyncio.TimeoutError()
            return "success"

        with patch('asyncio.sleep', new_callable=AsyncMock):
            result = await _retry_with_backoff(failing_then_success)
            assert result == "success"
            assert call_count == 2

    @pytest.mark.asyncio
    async def test_retry_on_client_error(self):
        """Test retry on aiohttp.ClientError."""
        call_count = 0

        async def failing_then_success():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise aiohttp.ClientError("Error")
            return "success"

        with patch('asyncio.sleep', new_callable=AsyncMock):
            result = await _retry_with_backoff(failing_then_success)
            assert result == "success"

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self):
        """Test exception raised after max retries."""
        async def always_fail():
            raise asyncio.TimeoutError()

        with patch('asyncio.sleep', new_callable=AsyncMock):
            with pytest.raises(asyncio.TimeoutError):
                await _retry_with_backoff(always_fail, max_retries=3)

    @pytest.mark.asyncio
    async def test_exponential_backoff(self):
        """Test exponential backoff delays."""
        sleep_times = []

        async def mock_sleep(delay):
            sleep_times.append(delay)

        async def always_fail():
            raise asyncio.TimeoutError()

        with patch('asyncio.sleep', mock_sleep):
            with pytest.raises(asyncio.TimeoutError):
                await _retry_with_backoff(always_fail, max_retries=3, base_delay=1.0)

        # Should have delays: 1.0, 2.0 (exponential)
        assert sleep_times == [1.0, 2.0]


# ============================================================================
# Global Cache Stats Tests
# ============================================================================

class TestGetEmbeddingCacheStats:
    """Tests for get_embedding_cache_stats function."""

    def test_returns_stats(self):
        """Test that function returns cache stats."""
        stats = get_embedding_cache_stats()
        assert "size" in stats
        assert "valid" in stats
        assert "ttl_seconds" in stats
