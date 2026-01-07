"""
Semantic retrieval using embeddings.

Provides similarity-based pattern retrieval for the CritiqueStore.
Uses OpenAI, Gemini, or local embeddings depending on availability.
"""

import asyncio
import aiohttp
from collections import OrderedDict
import hashlib
import json
import logging
import os
import struct
import time
from pathlib import Path
from typing import Optional
import sqlite3

from aragora.config import DB_TIMEOUT_SECONDS, get_api_key

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """Simple async-compatible TTL cache for embeddings.

    Uses OrderedDict for O(1) LRU eviction instead of O(n) min() scan.
    """

    def __init__(self, ttl_seconds: float = 3600, max_size: int = 1000):
        # OrderedDict maintains insertion order - oldest first for O(1) eviction
        self._cache: OrderedDict[str, tuple[float, list[float]]] = OrderedDict()
        self._ttl = ttl_seconds
        self._max_size = max_size

    def _make_key(self, text: str) -> str:
        """Generate cache key from text."""
        return hashlib.md5(text.lower().strip().encode()).hexdigest()

    def get(self, text: str) -> Optional[list[float]]:
        """Get cached embedding if valid."""
        key = self._make_key(text)
        if key in self._cache:
            timestamp, embedding = self._cache[key]
            if time.time() - timestamp < self._ttl:
                # Move to end to mark as recently used (LRU)
                self._cache.move_to_end(key)
                return embedding
            # Expired - remove
            del self._cache[key]
        return None

    def set(self, text: str, embedding: list[float]) -> None:
        """Cache an embedding."""
        key = self._make_key(text)
        # If key exists, update timestamp and move to end
        if key in self._cache:
            self._cache[key] = (time.time(), embedding)
            self._cache.move_to_end(key)
            return
        # Evict oldest entry if at capacity - O(1) with popitem(last=False)
        if len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)
        self._cache[key] = (time.time(), embedding)

    def stats(self) -> dict:
        """Get cache statistics."""
        now = time.time()
        valid = sum(1 for ts, _ in self._cache.values() if now - ts < self._ttl)
        return {"size": len(self._cache), "valid": valid, "ttl_seconds": self._ttl}


# Global embedding cache (shared across providers)
_embedding_cache = EmbeddingCache(ttl_seconds=3600, max_size=1000)

# Default API timeout
_API_TIMEOUT = aiohttp.ClientTimeout(total=30)


async def _retry_with_backoff(coro_fn, max_retries=3, base_delay=1.0):
    """Retry async function with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return await coro_fn()
        except (asyncio.TimeoutError, aiohttp.ClientError) as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
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
            h = hashlib.md5(f"{seed}:{text}".encode()).digest()
            # Convert first 4 bytes to float in [-1, 1]
            val = struct.unpack('<i', h[:4])[0] / (2**31)
            embedding.append(val)
        return embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Uses asyncio.gather for parallel execution when subclass embed() is async.
        Subclasses with native batch APIs should override for better performance.
        """
        import asyncio
        return await asyncio.gather(*[self.embed(t) for t in texts])


class OpenAIEmbedding(EmbeddingProvider):
    """OpenAI text-embedding-3-small embeddings."""

    def __init__(self, api_key: str | None = None, model: str = "text-embedding-3-small"):
        self.api_key = api_key or get_api_key("OPENAI_API_KEY")
        self.model = model
        self.dimension = 1536  # text-embedding-3-small

    async def embed(self, text: str) -> list[float]:
        # Check cache first
        cached = _embedding_cache.get(text)
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
                        raise RuntimeError(f"OpenAI embedding error: {await response.text()}")
                    data = await response.json()
                    return data["data"][0]["embedding"]

        embedding = await _retry_with_backoff(_call)
        _embedding_cache.set(text, embedding)
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
                        raise RuntimeError(f"OpenAI embedding error: {await response.text()}")
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
        cached = _embedding_cache.get(text)
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
                        raise RuntimeError(f"Gemini embedding error: {await response.text()}")
                    data = await response.json()
                    return data["embedding"]["values"]

        embedding = await _retry_with_backoff(_call)
        _embedding_cache.set(text, embedding)
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
                        raise RuntimeError(f"Ollama embedding error: {error_text}")
                    try:
                        data = await response.json()
                        return data["embedding"]
                    except (json.JSONDecodeError, KeyError) as e:
                        raise RuntimeError(f"Invalid Ollama response format: {e}")
            except aiohttp.ClientConnectorError:
                raise RuntimeError(
                    f"Cannot connect to Ollama at {self.base_url}. "
                    "Is Ollama running? Start with: ollama serve"
                )


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Optimized with NumPy when available, falls back to pure Python.
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
    """Pack embedding as binary for SQLite storage."""
    return struct.pack(f"{len(embedding)}f", *embedding)


def unpack_embedding(data: bytes) -> list[float]:
    """Unpack embedding from binary."""
    count = len(data) // 4  # 4 bytes per float
    return list(struct.unpack(f"{count}f", data))


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
            except Exception as e:
                logger.debug(f"Failed to connect to Ollama: {e}")
            # Fall back to hash-based embeddings (always works, no API needed)
            return EmbeddingProvider(dimension=256)

    def _init_tables(self):
        """Initialize embedding tables."""
        with sqlite3.connect(self.db_path, timeout=DB_TIMEOUT_SECONDS) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id TEXT PRIMARY KEY,
                    text_hash TEXT UNIQUE,
                    text TEXT,
                    embedding BLOB,
                    provider TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_hash ON embeddings(text_hash)")

            conn.commit()

    def _text_hash(self, text: str) -> str:
        """Generate hash for text deduplication."""
        return hashlib.md5(text.lower().strip().encode()).hexdigest()

    def _sync_get_existing_embedding(self, text_hash: str) -> Optional[bytes]:
        """Sync helper: Check if embedding already exists."""
        with sqlite3.connect(self.db_path, timeout=DB_TIMEOUT_SECONDS) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT embedding FROM embeddings WHERE text_hash = ?", (text_hash,))
            row = cursor.fetchone()
            return row[0] if row else None

    def _sync_store_embedding(
        self, id: str, text_hash: str, text: str, embedding: list[float]
    ) -> None:
        """Sync helper: Store embedding in database."""
        with sqlite3.connect(self.db_path, timeout=DB_TIMEOUT_SECONDS) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO embeddings (id, text_hash, text, embedding, provider)
                VALUES (?, ?, ?, ?, ?)
            """,
                (id, text_hash, text[:1000], pack_embedding(embedding), type(self.provider).__name__),
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
        await asyncio.to_thread(
            self._sync_store_embedding, id, text_hash, text, embedding
        )

        return embedding

    def _sync_get_all_embeddings(self) -> list[tuple]:
        """Sync helper: Retrieve all embeddings from database."""
        with sqlite3.connect(self.db_path, timeout=DB_TIMEOUT_SECONDS) as conn:
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
        with sqlite3.connect(self.db_path, timeout=DB_TIMEOUT_SECONDS) as conn:
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
    return _embedding_cache.stats()
