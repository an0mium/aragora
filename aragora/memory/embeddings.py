"""
Semantic retrieval using embeddings.

Provides similarity-based pattern retrieval for the CritiqueStore.
Uses the unified embedding service from aragora.core.embeddings.

Note: This module now re-exports from the unified embedding service.
New code should import directly from aragora.core.embeddings.
"""

__all__ = [
    "EmbeddingCache",
    "EmbeddingProvider",
    "OpenAIEmbedding",
    "GeminiEmbedding",
    "OllamaEmbedding",
    "SemanticRetriever",
    "get_embedding_cache",
    "get_embedding_cache_stats",
    "cosine_similarity",
    "pack_embedding",
    "unpack_embedding",
]

import asyncio
import hashlib
import json
import logging
import os
import struct
import time
from pathlib import Path
from typing import Optional

import aiohttp

from aragora.config import CACHE_TTL_EMBEDDINGS, get_api_key
from aragora.exceptions import ExternalServiceError
from aragora.memory.database import MemoryDatabase

# Re-export utilities from unified service
from aragora.core.embeddings.service import (
    cosine_similarity,
    pack_embedding,
    unpack_embedding,
)

# Use unified embedding cache from core
from aragora.core.embeddings.cache import EmbeddingCache, get_global_cache

logger = logging.getLogger(__name__)


# Global embedding cache (now uses unified cache from core)
_embedding_cache: Optional[EmbeddingCache] = None


def _get_embedding_cache() -> EmbeddingCache:
    """Get or create the global embedding cache."""
    global _embedding_cache
    if _embedding_cache is None:
        _embedding_cache = EmbeddingCache(
            ttl_seconds=CACHE_TTL_EMBEDDINGS,
            max_size=1000
        )
    return _embedding_cache

# Default API timeout
_API_TIMEOUT = aiohttp.ClientTimeout(total=30)

# Track registration status
_embedding_cache_registered = False


def _register_embedding_cache() -> None:
    """Register embedding cache with ServiceRegistry for observability."""
    global _embedding_cache_registered
    if _embedding_cache_registered:
        return

    try:
        from aragora.services import EmbeddingCacheService, ServiceRegistry

        cache = _get_embedding_cache()
        registry = ServiceRegistry.get()
        if not registry.has(EmbeddingCacheService):
            registry.register(EmbeddingCacheService, cache)
        _embedding_cache_registered = True
        logger.debug("Embedding cache registered with ServiceRegistry")
    except ImportError:
        pass  # Services module not available


def get_embedding_cache() -> EmbeddingCache:
    """Get the global embedding cache, registering with ServiceRegistry if available."""
    _register_embedding_cache()
    return _get_embedding_cache()


async def _retry_with_backoff(coro_fn, max_retries=3, base_delay=1.0):
    """Retry async function with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return await coro_fn()
        except (asyncio.TimeoutError, aiohttp.ClientError) as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2**attempt)
            logger.warning(f"API call failed (attempt {attempt + 1}), retrying in {delay}s: {e}")
            await asyncio.sleep(delay)


class EmbeddingProvider:
    """Base class for embedding providers."""

    def __init__(self, dimension: int = 256):
        self.dimension = dimension

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for text.

        Default implementation uses hash-based pseudo-embedding for graceful
        degradation when no API keys are available. Subclasses should override
        for proper semantic embeddings.
        """
        # Hash-based fallback embedding (deterministic, no API required)
        # Uses multiple hash seeds to create a fixed-dimension vector
        embedding = []
        for seed in range(self.dimension):
            h = hashlib.sha256(f"{seed}:{text}".encode()).digest()
            # Convert first 4 bytes to float in [-1, 1]
            val = struct.unpack("<i", h[:4])[0] / (2**31)
            embedding.append(val)
        return embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Uses asyncio.gather for parallel execution when subclass embed() is async.
        Subclasses with native batch APIs should override for better performance.

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings. Failed embeddings are replaced with zero vectors
            to maintain list alignment with input texts.
        """
        import asyncio
        import logging

        logger = logging.getLogger(__name__)

        # Use return_exceptions to prevent first failure from canceling others
        results = await asyncio.gather(*[self.embed(t) for t in texts], return_exceptions=True)

        # Process results, replacing exceptions with zero vectors
        embeddings = []
        for i, result in enumerate(results):
            if isinstance(result, BaseException):
                logger.warning(f"embed_batch: failed to embed text {i}: {result}")
                # Return zero vector to maintain alignment
                embeddings.append([0.0] * self.dimension)
            else:
                embeddings.append(result)

        return embeddings


class OpenAIEmbedding(EmbeddingProvider):
    """OpenAI text-embedding-3-small embeddings."""

    def __init__(self, api_key: str | None = None, model: str = "text-embedding-3-small"):
        self.api_key = api_key or get_api_key("OPENAI_API_KEY")
        self.model = model
        self.dimension = 1536  # text-embedding-3-small

    async def embed(self, text: str) -> list[float]:
        # Check cache first
        cached = _get_embedding_cache().get(text)
        if cached is not None:
            logger.debug("Embedding cache hit for OpenAI")
            return cached

        async def _call():
            async with aiohttp.ClientSession(timeout=_API_TIMEOUT) as session:
                async with session.post(
                    "https://api.openai.com/v1/embeddings",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={"model": self.model, "input": text},
                ) as response:
                    if response.status == 429:
                        raise aiohttp.ClientError("Rate limited")
                    if response.status != 200:
                        raise ExternalServiceError(
                            service="OpenAI Embedding",
                            reason=await response.text(),
                            status_code=response.status,
                        )
                    data = await response.json()
                    return data["data"][0]["embedding"]

        embedding = await _retry_with_backoff(_call)
        _get_embedding_cache().set(text, embedding)
        return embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        async def _call():
            async with aiohttp.ClientSession(timeout=_API_TIMEOUT) as session:
                async with session.post(
                    "https://api.openai.com/v1/embeddings",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={"model": self.model, "input": texts},
                ) as response:
                    if response.status == 429:
                        raise aiohttp.ClientError("Rate limited")
                    if response.status != 200:
                        raise ExternalServiceError(
                            service="OpenAI Embedding",
                            reason=await response.text(),
                            status_code=response.status,
                        )
                    data = await response.json()
                    return [d["embedding"] for d in sorted(data["data"], key=lambda x: x["index"])]

        return await _retry_with_backoff(_call)


class GeminiEmbedding(EmbeddingProvider):
    """Google Gemini embeddings."""

    def __init__(self, api_key: str | None = None, model: str = "text-embedding-004"):
        self.api_key = api_key or get_api_key("GEMINI_API_KEY", "GOOGLE_API_KEY")
        self.model = model
        self.dimension = 768

    async def embed(self, text: str) -> list[float]:
        # Check cache first
        cached = _get_embedding_cache().get(text)
        if cached is not None:
            logger.debug("Embedding cache hit for Gemini")
            return cached

        # Use header-based auth instead of URL parameter (security best practice)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:embedContent"

        async def _call():
            async with aiohttp.ClientSession(timeout=_API_TIMEOUT) as session:
                async with session.post(
                    url,
                    headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
                    json={"content": {"parts": [{"text": text}]}},
                ) as response:
                    if response.status == 429:
                        raise aiohttp.ClientError("Rate limited")
                    if response.status != 200:
                        raise ExternalServiceError(
                            service="Gemini Embedding",
                            reason=await response.text(),
                            status_code=response.status,
                        )
                    data = await response.json()
                    return data["embedding"]["values"]

        embedding = await _retry_with_backoff(_call)
        _get_embedding_cache().set(text, embedding)
        return embedding


class OllamaEmbedding(EmbeddingProvider):
    """Local Ollama embeddings."""

    def __init__(self, model: str = "nomic-embed-text", base_url: str | None = None):
        self.model = model
        self.base_url = base_url or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.dimension = 768  # nomic-embed-text

    async def embed(self, text: str) -> list[float]:
        async with aiohttp.ClientSession(timeout=_API_TIMEOUT) as session:
            try:
                async with session.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model, "prompt": text},
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise ExternalServiceError(
                            service="Ollama Embedding",
                            reason=error_text,
                            status_code=response.status,
                        )
                    try:
                        data = await response.json()
                        return data["embedding"]
                    except (json.JSONDecodeError, KeyError) as e:
                        raise ExternalServiceError(
                            service="Ollama Embedding", reason=f"Invalid response format: {e}"
                        ) from e
            except aiohttp.ClientConnectorError as e:
                raise ExternalServiceError(
                    service="Ollama Embedding",
                    reason=f"Cannot connect to Ollama at {self.base_url}. Is Ollama running? Start with: ollama serve",
                ) from e


# Note: cosine_similarity, pack_embedding, and unpack_embedding are now
# imported from aragora.core.embeddings.service and re-exported above.


class SemanticRetriever:
    """
    Semantic retrieval for the CritiqueStore.

    Enables finding similar patterns based on meaning, not just keywords.
    """

    def __init__(
        self,
        db_path: str,
        provider: EmbeddingProvider = None,
    ):
        self.db_path = Path(db_path)
        self.db = MemoryDatabase(db_path)
        self.provider = provider or self._auto_detect_provider()
        self._init_tables()

    def _auto_detect_provider(self) -> EmbeddingProvider:
        """Auto-detect best available embedding provider.

        Falls back gracefully to hash-based embeddings if no API keys
        are available and Ollama is not running.
        """
        if os.environ.get("OPENAI_API_KEY"):
            return OpenAIEmbedding()
        elif os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
            return GeminiEmbedding()
        else:
            # Try Ollama, but fall back to hash-based if not available
            try:
                import socket

                ollama = OllamaEmbedding()
                # Quick connectivity check (non-blocking)
                host = ollama.base_url.replace("http://", "").replace("https://", "")
                port = 11434  # Default Ollama port
                if ":" in host:
                    # Handle host:port format (use rsplit to handle IPv6 or malformed URLs)
                    parts = host.rsplit(":", 1)
                    if len(parts) == 2:
                        host = parts[0]
                        try:
                            port = int(parts[1])
                        except ValueError:
                            logger.debug(f"Invalid port in Ollama URL: {parts[1]}, using default")
                            port = 11434
                # Use context manager to guarantee socket cleanup in all code paths
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.settimeout(0.5)
                    result = sock.connect_ex((host, port))
                    if result == 0:
                        return ollama
            except (OSError, socket.error, socket.timeout, ConnectionError, TimeoutError) as e:
                logger.debug(f"Failed to connect to Ollama: {e}")
            # Fall back to hash-based embeddings (always works, no API needed)
            return EmbeddingProvider(dimension=256)

    def _init_tables(self):
        """Initialize embedding tables."""
        with self.db.connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    id TEXT PRIMARY KEY,
                    text_hash TEXT UNIQUE,
                    text TEXT,
                    embedding BLOB,
                    provider TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_embeddings_hash ON embeddings(text_hash)"
            )

            conn.commit()

    def _text_hash(self, text: str) -> str:
        """Generate hash for text deduplication."""
        return hashlib.sha256(text.lower().strip().encode()).hexdigest()

    def _sync_get_existing_embedding(self, text_hash: str) -> Optional[bytes]:
        """Sync helper: Check if embedding already exists."""
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT embedding FROM embeddings WHERE text_hash = ?", (text_hash,))
            row = cursor.fetchone()
            return row[0] if row else None

    def _sync_store_embedding(
        self, id: str, text_hash: str, text: str, embedding: list[float]
    ) -> None:
        """Sync helper: Store embedding in database."""
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO embeddings (id, text_hash, text, embedding, provider)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    id,
                    text_hash,
                    text[:1000],
                    pack_embedding(embedding),
                    type(self.provider).__name__,
                ),
            )
            conn.commit()

    async def embed_and_store(self, id: str, text: str) -> list[float]:
        """Embed text and store in database."""
        text_hash = self._text_hash(text)

        # Check if already embedded (non-blocking)
        existing = await asyncio.to_thread(self._sync_get_existing_embedding, text_hash)
        if existing:
            return unpack_embedding(existing)

        # Generate embedding (async API call)
        embedding = await self.provider.embed(text)

        # Store (non-blocking)
        await asyncio.to_thread(self._sync_store_embedding, id, text_hash, text, embedding)

        return embedding

    def _sync_get_all_embeddings(self) -> list[tuple]:
        """Sync helper: Retrieve all embeddings from database."""
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, text, embedding FROM embeddings")
            return cursor.fetchall()

    async def find_similar(
        self,
        query: str,
        limit: int = 5,
        min_similarity: float = 0.5,
    ) -> list[tuple[str, str, float]]:
        """
        Find similar stored texts.

        Returns list of (id, text, similarity) tuples.
        """
        query_embedding = await self.provider.embed(query)

        # Fetch all embeddings (non-blocking)
        rows = await asyncio.to_thread(self._sync_get_all_embeddings)

        if not rows:
            return []

        # Calculate similarities
        results = []
        for id, text, emb_bytes in rows:
            stored_embedding = unpack_embedding(emb_bytes)
            similarity = cosine_similarity(query_embedding, stored_embedding)
            if similarity >= min_similarity:
                results.append((id, text, similarity))

        # Sort by similarity descending
        results.sort(key=lambda x: x[2], reverse=True)

        return results[:limit]

    def get_stats(self) -> dict:
        """Get embedding statistics."""
        with self.db.connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM embeddings")
            row = cursor.fetchone()
            total = row[0] if row else 0

            cursor.execute("SELECT provider, COUNT(*) FROM embeddings GROUP BY provider")
            by_provider = dict(cursor.fetchall())

        return {
            "total_embeddings": total,
            "by_provider": by_provider,
        }


def get_embedding_cache_stats() -> dict:
    """Get global embedding cache statistics."""
    cache = _get_embedding_cache()
    stats = cache.get_stats()
    # Convert CacheStats dataclass to dict for backwards compatibility
    return {
        "size": stats.size,
        "valid": stats.valid,
        "ttl_seconds": stats.ttl_seconds,
        "hits": stats.hits,
        "misses": stats.misses,
        "hit_rate": stats.hit_rate,
    }
