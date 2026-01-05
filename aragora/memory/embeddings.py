"""
Semantic retrieval using embeddings.

Provides similarity-based pattern retrieval for the CritiqueStore.
Uses OpenAI, Gemini, or local embeddings depending on availability.
"""

import asyncio
import aiohttp
import hashlib
import json
import logging
import os
import struct
from pathlib import Path
from typing import Optional
import sqlite3

logger = logging.getLogger(__name__)


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
        """Generate embeddings for multiple texts."""
        return [await self.embed(t) for t in texts]


class OpenAIEmbedding(EmbeddingProvider):
    """OpenAI text-embedding-3-small embeddings."""

    def __init__(self, api_key: str = None, model: str = "text-embedding-3-small"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.dimension = 1536  # text-embedding-3-small

    async def embed(self, text: str) -> list[float]:
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY required for OpenAI embeddings")

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.openai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": self.model, "input": text},
            ) as response:
                if response.status != 200:
                    raise RuntimeError(f"OpenAI embedding error: {await response.text()}")
                data = await response.json()
                return data["data"][0]["embedding"]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY required for OpenAI embeddings")

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.openai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": self.model, "input": texts},
            ) as response:
                if response.status != 200:
                    raise RuntimeError(f"OpenAI embedding error: {await response.text()}")
                data = await response.json()
                return [d["embedding"] for d in sorted(data["data"], key=lambda x: x["index"])]


class GeminiEmbedding(EmbeddingProvider):
    """Google Gemini embeddings."""

    def __init__(self, api_key: str = None, model: str = "text-embedding-004"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.model = model
        self.dimension = 768

    async def embed(self, text: str) -> list[float]:
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY required for Gemini embeddings")

        # Use header-based auth instead of URL parameter (security best practice)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:embedContent"

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
                json={"content": {"parts": [{"text": text}]}},
            ) as response:
                if response.status != 200:
                    raise RuntimeError(f"Gemini embedding error: {await response.text()}")
                data = await response.json()
                return data["embedding"]["values"]


class OllamaEmbedding(EmbeddingProvider):
    """Local Ollama embeddings."""

    def __init__(self, model: str = "nomic-embed-text", base_url: str = None):
        self.model = model
        self.base_url = base_url or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.dimension = 768  # nomic-embed-text

    async def embed(self, text: str) -> list[float]:
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model, "prompt": text},
                ) as response:
                    if response.status != 200:
                        raise RuntimeError(f"Ollama embedding error: {await response.text()}")
                    data = await response.json()
                    return data["embedding"]
            except aiohttp.ClientConnectorError:
                raise RuntimeError(
                    f"Cannot connect to Ollama at {self.base_url}. "
                    "Is Ollama running? Start with: ollama serve"
                )


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
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
                if ":" in host:
                    host, port = host.split(":")
                    port = int(port)
                else:
                    port = 11434
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(0.5)
                result = sock.connect_ex((host, port))
                sock.close()
                if result == 0:
                    return ollama
            except Exception as e:
                logger.debug(f"Failed to connect to Ollama: {e}")
            # Fall back to hash-based embeddings (always works, no API needed)
            return EmbeddingProvider(dimension=256)

    def _init_tables(self):
        """Initialize embedding tables."""
        with sqlite3.connect(self.db_path, timeout=30.0) as conn:
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

    async def embed_and_store(self, id: str, text: str) -> list[float]:
        """Embed text and store in database."""
        text_hash = self._text_hash(text)

        with sqlite3.connect(self.db_path, timeout=30.0) as conn:
            cursor = conn.cursor()

            # Check if already embedded
            cursor.execute("SELECT embedding FROM embeddings WHERE text_hash = ?", (text_hash,))
            row = cursor.fetchone()
            if row:
                return unpack_embedding(row[0])

            # Generate embedding
            embedding = await self.provider.embed(text)

            # Store
            cursor.execute(
                """
                INSERT OR REPLACE INTO embeddings (id, text_hash, text, embedding, provider)
                VALUES (?, ?, ?, ?, ?)
            """,
                (id, text_hash, text[:1000], pack_embedding(embedding), type(self.provider).__name__),
            )

            conn.commit()

        return embedding

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

        with sqlite3.connect(self.db_path, timeout=30.0) as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT id, text, embedding FROM embeddings")
            rows = cursor.fetchall()

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
        with sqlite3.connect(self.db_path, timeout=30.0) as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM embeddings")
            total = cursor.fetchone()[0]

            cursor.execute("SELECT provider, COUNT(*) FROM embeddings GROUP BY provider")
            by_provider = dict(cursor.fetchall())

        return {
            "total_embeddings": total,
            "by_provider": by_provider,
        }
