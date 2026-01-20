"""Tests for the unified embedding service.

Tests cover:
- Service initialization and provider detection
- Embedding generation (single and batch)
- Caching behavior
- Similarity computation
- Backwards compatibility with memory.embeddings
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.core.embeddings import (
    EmbeddingConfig,
    EmbeddingResult,
    UnifiedEmbeddingService,
    get_embedding_service,
)
from aragora.core.embeddings.service import (
    cosine_similarity,
    pack_embedding,
    unpack_embedding,
    reset_embedding_service,
)
from aragora.core.embeddings.cache import (
    EmbeddingCache,
    ScopedCacheManager,
    get_global_cache,
    get_scoped_cache,
    reset_caches,
)
from aragora.core.embeddings.backends import (
    EmbeddingBackend,
    HashBackend,
    OpenAIBackend,
    GeminiBackend,
    OllamaBackend,
)


class TestEmbeddingConfig:
    """Test EmbeddingConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EmbeddingConfig()
        assert config.provider is None
        assert config.model is None
        assert config.dimension == 1536
        assert config.cache_enabled is True
        assert config.cache_ttl == 3600.0
        assert config.cache_max_size == 1000

    def test_dimension_auto_set_from_model(self):
        """Test dimension is auto-set based on model."""
        config = EmbeddingConfig(model="text-embedding-3-large")
        assert config.dimension == 3072

        config = EmbeddingConfig(model="text-embedding-004")
        assert config.dimension == 768

    def test_explicit_dimension_not_overwritten(self):
        """Test explicit dimension is preserved."""
        config = EmbeddingConfig(dimension=512)
        assert config.dimension == 512


class TestEmbeddingCache:
    """Test EmbeddingCache class."""

    def test_cache_set_and_get(self):
        """Test basic cache operations."""
        cache = EmbeddingCache(ttl_seconds=3600, max_size=100)
        embedding = [0.1, 0.2, 0.3]

        cache.set("test text", embedding)
        cached = cache.get("test text")

        assert cached == embedding

    def test_cache_miss_returns_none(self):
        """Test cache miss returns None."""
        cache = EmbeddingCache()
        assert cache.get("nonexistent") is None

    def test_cache_ttl_expiration(self):
        """Test cache entries expire after TTL."""
        cache = EmbeddingCache(ttl_seconds=0.001, max_size=100)
        cache.set("test", [0.1])

        import time
        time.sleep(0.01)

        assert cache.get("test") is None

    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = EmbeddingCache(ttl_seconds=3600, max_size=2)

        cache.set("first", [0.1])
        cache.set("second", [0.2])
        cache.set("third", [0.3])  # Should evict "first"

        assert cache.get("first") is None
        assert cache.get("second") == [0.2]
        assert cache.get("third") == [0.3]

    def test_cache_stats(self):
        """Test cache statistics."""
        cache = EmbeddingCache()
        cache.set("test", [0.1])
        cache.get("test")  # Hit
        cache.get("miss")  # Miss

        stats = cache.get_stats()
        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.size == 1

    def test_cache_clear(self):
        """Test cache clearing."""
        cache = EmbeddingCache()
        cache.set("test", [0.1])
        cache.clear()

        assert len(cache) == 0
        assert cache.get("test") is None


class TestScopedCacheManager:
    """Test ScopedCacheManager class."""

    def test_get_creates_cache(self):
        """Test get_cache creates new cache for scope."""
        manager = ScopedCacheManager()
        cache = manager.get_cache("debate_1")

        assert cache is not None
        assert isinstance(cache, EmbeddingCache)

    def test_get_returns_same_cache(self):
        """Test get_cache returns same cache for same scope."""
        manager = ScopedCacheManager()
        cache1 = manager.get_cache("debate_1")
        cache2 = manager.get_cache("debate_1")

        assert cache1 is cache2

    def test_different_scopes_have_different_caches(self):
        """Test different scopes have isolated caches."""
        manager = ScopedCacheManager()
        cache1 = manager.get_cache("debate_1")
        cache2 = manager.get_cache("debate_2")

        cache1.set("test", [0.1])
        assert cache2.get("test") is None

    def test_cleanup_removes_cache(self):
        """Test cleanup removes and clears cache."""
        manager = ScopedCacheManager()
        cache = manager.get_cache("debate_1")
        cache.set("test", [0.1])

        manager.cleanup("debate_1")

        new_cache = manager.get_cache("debate_1")
        assert new_cache.get("test") is None


class TestHashBackend:
    """Test HashBackend (fallback provider)."""

    @pytest.mark.asyncio
    async def test_embed_returns_vector(self):
        """Test embed returns vector of correct dimension."""
        config = EmbeddingConfig(dimension=256)
        backend = HashBackend(config)

        embedding = await backend.embed("test text")

        assert len(embedding) == 256
        assert all(isinstance(x, float) for x in embedding)

    @pytest.mark.asyncio
    async def test_embed_is_deterministic(self):
        """Test same text produces same embedding."""
        config = EmbeddingConfig(dimension=256)
        backend = HashBackend(config)

        emb1 = await backend.embed("test text")
        emb2 = await backend.embed("test text")

        assert emb1 == emb2

    @pytest.mark.asyncio
    async def test_embed_batch(self):
        """Test batch embedding."""
        config = EmbeddingConfig(dimension=256)
        backend = HashBackend(config)

        embeddings = await backend.embed_batch(["text 1", "text 2"])

        assert len(embeddings) == 2
        assert all(len(e) == 256 for e in embeddings)

    def test_is_available(self):
        """Test hash backend is always available."""
        config = EmbeddingConfig()
        backend = HashBackend(config)
        assert backend.is_available() is True


class TestUnifiedEmbeddingService:
    """Test UnifiedEmbeddingService class."""

    def setup_method(self):
        """Reset global state before each test."""
        reset_embedding_service()
        reset_caches()

    @pytest.mark.asyncio
    async def test_embed_returns_result(self):
        """Test embed returns EmbeddingResult."""
        config = EmbeddingConfig(provider="hash", dimension=256)
        service = UnifiedEmbeddingService(config=config)

        result = await service.embed("test text")

        assert isinstance(result, EmbeddingResult)
        assert len(result.embedding) == 256
        assert result.provider == "hash"

    @pytest.mark.asyncio
    async def test_embed_caches_result(self):
        """Test embed caches results."""
        from aragora.core.embeddings.cache import EmbeddingCache

        # Use isolated cache to avoid interference from global resets
        cache = EmbeddingCache(ttl_seconds=3600, max_size=100)
        config = EmbeddingConfig(provider="hash", dimension=256, cache_enabled=True)
        service = UnifiedEmbeddingService(config=config, cache=cache)

        result1 = await service.embed("test text")
        result2 = await service.embed("test text")

        assert result1.cached is False
        assert result2.cached is True
        assert result1.embedding == result2.embedding

    @pytest.mark.asyncio
    async def test_embed_batch(self):
        """Test batch embedding."""
        config = EmbeddingConfig(provider="hash", dimension=256)
        service = UnifiedEmbeddingService(config=config)

        results = await service.embed_batch(["text 1", "text 2", "text 3"])

        assert len(results) == 3
        assert all(isinstance(r, EmbeddingResult) for r in results)

    @pytest.mark.asyncio
    async def test_embed_raw(self):
        """Test raw embedding (no metadata)."""
        config = EmbeddingConfig(provider="hash", dimension=256)
        service = UnifiedEmbeddingService(config=config)

        embedding = await service.embed_raw("test text")

        assert isinstance(embedding, list)
        assert len(embedding) == 256

    @pytest.mark.asyncio
    async def test_similarity(self):
        """Test similarity computation."""
        config = EmbeddingConfig(provider="hash", dimension=256)
        service = UnifiedEmbeddingService(config=config)

        # Same text should have similarity ~1.0
        sim = await service.similarity("hello", "hello")
        assert sim == pytest.approx(1.0, abs=0.01)

    @pytest.mark.asyncio
    async def test_find_similar(self):
        """Test find_similar returns ranked results."""
        config = EmbeddingConfig(provider="hash", dimension=256)
        service = UnifiedEmbeddingService(config=config)

        results = await service.find_similar(
            "query",
            ["candidate 1", "candidate 2", "candidate 3"],
            top_k=2,
        )

        assert len(results) <= 2
        assert all(len(r) == 3 for r in results)  # (index, text, similarity)


class TestUtilityFunctions:
    """Test utility functions."""

    def test_cosine_similarity_identical_vectors(self):
        """Test cosine similarity of identical vectors is 1."""
        v = [0.1, 0.2, 0.3]
        assert cosine_similarity(v, v) == pytest.approx(1.0, abs=0.001)

    def test_cosine_similarity_orthogonal_vectors(self):
        """Test cosine similarity of orthogonal vectors is 0."""
        v1 = [1.0, 0.0]
        v2 = [0.0, 1.0]
        assert cosine_similarity(v1, v2) == pytest.approx(0.0, abs=0.001)

    def test_cosine_similarity_opposite_vectors(self):
        """Test cosine similarity of opposite vectors is -1."""
        v1 = [1.0, 0.0]
        v2 = [-1.0, 0.0]
        assert cosine_similarity(v1, v2) == pytest.approx(-1.0, abs=0.001)

    def test_pack_unpack_roundtrip(self):
        """Test pack/unpack preserves values."""
        import math
        original = [0.1, 0.2, 0.3, -0.5, 1.0]
        packed = pack_embedding(original)
        unpacked = unpack_embedding(packed)

        assert len(unpacked) == len(original)
        for o, u in zip(original, unpacked):
            assert math.isclose(o, u, rel_tol=1e-6)

    def test_pack_produces_bytes(self):
        """Test pack produces bytes."""
        embedding = [0.1, 0.2, 0.3]
        packed = pack_embedding(embedding)
        assert isinstance(packed, bytes)
        assert len(packed) == 12  # 3 floats * 4 bytes


class TestGetEmbeddingService:
    """Test get_embedding_service factory function."""

    def setup_method(self):
        """Reset global state."""
        reset_embedding_service()

    def test_returns_singleton(self):
        """Test returns same instance without config."""
        service1 = get_embedding_service()
        service2 = get_embedding_service()
        assert service1 is service2

    def test_custom_config_creates_new(self):
        """Test custom config creates new instance."""
        config = EmbeddingConfig(provider="hash")
        service1 = get_embedding_service()
        service2 = get_embedding_service(config=config)
        assert service1 is not service2

    def test_scoped_creates_isolated(self):
        """Test scope_id creates isolated instance."""
        service1 = get_embedding_service()
        service2 = get_embedding_service(scope_id="debate_1")
        assert service1 is not service2


class TestBackwardsCompatibility:
    """Test backwards compatibility with aragora.memory.embeddings."""

    def test_memory_embeddings_imports(self):
        """Test memory.embeddings still exports utilities."""
        from aragora.memory.embeddings import (
            cosine_similarity,
            pack_embedding,
            unpack_embedding,
        )
        # Just verify they're callable
        assert callable(cosine_similarity)
        assert callable(pack_embedding)
        assert callable(unpack_embedding)

    def test_core_types_import(self):
        """Test aragora.core still exports types."""
        from aragora.core import (
            Critique,
            DebateResult,
            Message,
            Agent,
            Vote,
        )
        # Verify they're classes
        assert Critique is not None
        assert DebateResult is not None
