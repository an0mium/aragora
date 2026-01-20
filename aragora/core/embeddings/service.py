"""Unified embedding service for all Aragora subsystems.

This module provides the main entry point for generating text embeddings.
It consolidates previously fragmented implementations and provides a single,
consistent interface.
"""

import logging
import struct
from typing import Optional

from aragora.core.embeddings.types import CacheStats, EmbeddingConfig, EmbeddingResult
from aragora.core.embeddings.cache import EmbeddingCache, get_global_cache, get_scoped_cache
from aragora.core.embeddings.backends import (
    EmbeddingBackend,
    OpenAIBackend,
    GeminiBackend,
    OllamaBackend,
    HashBackend,
)

logger = logging.getLogger(__name__)


class UnifiedEmbeddingService:
    """Unified embedding service for all Aragora subsystems.

    This service provides:
    - Automatic provider detection (OpenAI, Gemini, Ollama, hash fallback)
    - Unified caching across all subsystems
    - Consistent interface for all embedding operations
    - Cosine similarity computation
    - Batch embedding support

    Example:
        service = UnifiedEmbeddingService()

        # Generate single embedding
        result = await service.embed("Hello world")
        print(result.embedding)

        # Batch embedding
        results = await service.embed_batch(["Text 1", "Text 2"])

        # Compute similarity
        sim = await service.similarity("Hello", "Hi there")
    """

    def __init__(
        self,
        config: Optional[EmbeddingConfig] = None,
        cache: Optional[EmbeddingCache] = None,
    ):
        """Initialize the unified embedding service.

        Args:
            config: Embedding configuration (auto-detected if not specified)
            cache: Cache instance (global cache used if not specified)
        """
        self.config = config or EmbeddingConfig()
        if cache is not None:
            self._cache = cache
        elif self.config.cache_enabled:
            self._cache = get_global_cache()
        else:
            self._cache = None
        self._backend = self._create_backend()

    def _create_backend(self) -> EmbeddingBackend:
        """Create the appropriate backend based on config and availability."""
        provider = self.config.provider

        if provider:
            # Explicit provider specified
            backends = {
                "openai": OpenAIBackend,
                "gemini": GeminiBackend,
                "ollama": OllamaBackend,
                "hash": HashBackend,
            }
            backend_class = backends.get(provider.lower())
            if not backend_class:
                raise ValueError(f"Unknown provider: {provider}")
            return backend_class(self.config)

        # Auto-detect best available provider
        for backend_class in [OpenAIBackend, GeminiBackend, OllamaBackend, HashBackend]:
            backend = backend_class(self.config)
            if backend.is_available():
                logger.info(f"Using {backend.provider_name} embedding provider")
                return backend

        # Fallback to hash (always available)
        logger.warning("No embedding provider available, using hash fallback")
        return HashBackend(self.config)

    @property
    def provider(self) -> str:
        """Get the active provider name."""
        return self._backend.provider_name

    @property
    def model(self) -> str:
        """Get the active model name."""
        return self._backend.model_name

    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        return self._backend.dimension

    async def embed(self, text: str) -> EmbeddingResult:
        """Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            EmbeddingResult with vector and metadata
        """
        # Check cache first
        if self._cache is not None:
            cached = self._cache.get(text)
            if cached is not None:
                logger.debug(f"Cache hit for {self._backend.provider_name}")
                return EmbeddingResult(
                    embedding=cached,
                    text=text[:100],
                    provider=self._backend.provider_name,
                    model=self._backend.model_name,
                    cached=True,
                    dimension=len(cached),
                )

        # Generate embedding
        embedding = await self._backend.embed(text)

        # Cache result
        if self._cache is not None:
            self._cache.set(text, embedding)

        return EmbeddingResult(
            embedding=embedding,
            text=text[:100],
            provider=self._backend.provider_name,
            model=self._backend.model_name,
            cached=False,
            dimension=len(embedding),
        )

    async def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        """Generate embeddings for multiple texts.

        Uses caching and batching for efficiency.

        Args:
            texts: List of texts to embed

        Returns:
            List of EmbeddingResult objects
        """
        if not texts:
            return []

        results: list[EmbeddingResult] = []
        texts_to_embed: list[tuple[int, str]] = []  # (index, text)

        # Check cache for each text
        for i, text in enumerate(texts):
            if self._cache is not None:
                cached = self._cache.get(text)
                if cached is not None:
                    results.append(
                        EmbeddingResult(
                            embedding=cached,
                            text=text[:100],
                            provider=self._backend.provider_name,
                            model=self._backend.model_name,
                            cached=True,
                            dimension=len(cached),
                        )
                    )
                    continue

            texts_to_embed.append((i, text))
            results.append(None)  # Placeholder

        # Batch embed uncached texts
        if texts_to_embed:
            uncached_texts = [t for _, t in texts_to_embed]
            embeddings = await self._backend.embed_batch(uncached_texts)

            for (original_idx, text), embedding in zip(texts_to_embed, embeddings):
                if self._cache is not None:
                    self._cache.set(text, embedding)

                results[original_idx] = EmbeddingResult(
                    embedding=embedding,
                    text=text[:100],
                    provider=self._backend.provider_name,
                    model=self._backend.model_name,
                    cached=False,
                    dimension=len(embedding),
                )

        return results

    async def embed_raw(self, text: str) -> list[float]:
        """Generate raw embedding vector (no metadata).

        Convenience method for when you only need the vector.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        result = await self.embed(text)
        return result.embedding

    async def embed_batch_raw(self, texts: list[str]) -> list[list[float]]:
        """Generate raw embedding vectors for batch (no metadata).

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        results = await self.embed_batch(texts)
        return [r.embedding for r in results]

    async def similarity(self, text_a: str, text_b: str) -> float:
        """Compute cosine similarity between two texts.

        Args:
            text_a: First text
            text_b: Second text

        Returns:
            Cosine similarity score (-1 to 1)
        """
        results = await self.embed_batch([text_a, text_b])
        return cosine_similarity(results[0].embedding, results[1].embedding)

    async def find_similar(
        self,
        query: str,
        candidates: list[str],
        top_k: int = 5,
        min_similarity: float = 0.0,
    ) -> list[tuple[int, str, float]]:
        """Find the most similar texts to a query.

        Args:
            query: Query text
            candidates: List of candidate texts
            top_k: Maximum number of results
            min_similarity: Minimum similarity threshold

        Returns:
            List of (index, text, similarity) tuples, sorted by similarity
        """
        if not candidates:
            return []

        # Embed query and all candidates
        all_texts = [query] + candidates
        results = await self.embed_batch(all_texts)

        query_embedding = results[0].embedding
        candidate_embeddings = [r.embedding for r in results[1:]]

        # Compute similarities
        scored = []
        for i, (text, emb) in enumerate(zip(candidates, candidate_embeddings)):
            sim = cosine_similarity(query_embedding, emb)
            if sim >= min_similarity:
                scored.append((i, text, sim))

        # Sort by similarity descending
        scored.sort(key=lambda x: x[2], reverse=True)

        return scored[:top_k]

    def get_cache_stats(self) -> Optional[CacheStats]:
        """Get cache statistics.

        Returns:
            CacheStats if caching is enabled, None otherwise
        """
        if self._cache is not None:
            return self._cache.get_stats()
        return None

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        if self._cache is not None:
            self._cache.clear()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Optimized with NumPy when available, falls back to pure Python.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Cosine similarity (-1 to 1)
    """
    try:
        import numpy as np

        a_arr = np.asarray(a, dtype=np.float32)
        b_arr = np.asarray(b, dtype=np.float32)
        dot = np.dot(a_arr, b_arr)
        norm_a = np.linalg.norm(a_arr)
        norm_b = np.linalg.norm(b_arr)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))
    except ImportError:
        # Fallback to pure Python
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


def pack_embedding(embedding: list[float]) -> bytes:
    """Pack embedding as binary for SQLite storage.

    Args:
        embedding: Embedding vector

    Returns:
        Binary representation
    """
    return struct.pack(f"{len(embedding)}f", *embedding)


def unpack_embedding(data: bytes) -> list[float]:
    """Unpack embedding from binary.

    Args:
        data: Binary representation

    Returns:
        Embedding vector
    """
    count = len(data) // 4  # 4 bytes per float
    return list(struct.unpack(f"{count}f", data))


# Global service instance
_global_service: Optional[UnifiedEmbeddingService] = None


def get_embedding_service(
    config: Optional[EmbeddingConfig] = None,
    scope_id: Optional[str] = None,
) -> UnifiedEmbeddingService:
    """Get or create the embedding service.

    Args:
        config: Optional configuration override
        scope_id: Optional scope ID for isolated caching

    Returns:
        UnifiedEmbeddingService instance
    """
    global _global_service

    if scope_id:
        # Scoped service with isolated cache
        cache = get_scoped_cache(scope_id)
        return UnifiedEmbeddingService(config=config, cache=cache)

    if config is not None:
        # Custom config - create new instance
        return UnifiedEmbeddingService(config=config)

    # Use global singleton
    if _global_service is None:
        _global_service = UnifiedEmbeddingService()

    return _global_service


def reset_embedding_service() -> None:
    """Reset the global embedding service (for testing)."""
    global _global_service
    if _global_service:
        _global_service.clear_cache()
    _global_service = None


__all__ = [
    "UnifiedEmbeddingService",
    "get_embedding_service",
    "reset_embedding_service",
    "cosine_similarity",
    "pack_embedding",
    "unpack_embedding",
]
