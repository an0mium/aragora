"""LRU cache for text embeddings with optional persistence.

This module provides EmbeddingCache, which caches embeddings at the text
level to avoid redundant model.encode() calls.

Performance impact:
    - Without cache: O(n²) encode calls for n texts
    - With cache: O(n) encode calls (amortized)
    - Expected speedup: 10-100x for repeated text comparisons
"""

import hashlib
import logging
import threading
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """
    LRU cache for text embeddings with optional persistence.

    Caches embeddings at the text level, so the same text encoded in
    different pairs only requires one model.encode() call.

    Performance impact:
        - Without cache: O(n²) encode calls for n texts
        - With cache: O(n) encode calls (amortized)
        - Expected speedup: 10-100x for repeated text comparisons
    """

    def __init__(
        self,
        max_size: int = 1024,
        persist: bool = False,
        db_path: Optional[str] = None,
    ):
        """
        Initialize embedding cache.

        Args:
            max_size: Maximum entries in memory cache (default 1024)
            persist: Whether to persist to database (default False)
            db_path: Path to embeddings database (uses core.db if None)
        """
        self.max_size = max_size
        self.persist = persist
        self.db_path = db_path
        self._cache: dict[str, np.ndarray] = {}
        self._access_order: list[str] = []
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def _hash_text(self, text: str) -> str:
        """Generate hash key for text."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]

    def get(self, text: str) -> Optional[np.ndarray]:
        """Get embedding from cache."""
        key = self._hash_text(text)

        with self._lock:
            if key in self._cache:
                self._hits += 1
                # Move to end of access order (LRU)
                if key in self._access_order:
                    self._access_order.remove(key)
                self._access_order.append(key)
                return self._cache[key]

        # Try persistent cache
        if self.persist:
            embedding = self._load_from_db(key)
            if embedding is not None:
                self._hits += 1
                with self._lock:
                    self._store_in_memory(key, embedding)
                return embedding

        self._misses += 1
        return None

    def put(self, text: str, embedding: np.ndarray) -> None:
        """Store embedding in cache."""
        key = self._hash_text(text)

        with self._lock:
            self._store_in_memory(key, embedding)

        if self.persist:
            self._save_to_db(key, text[:1000], embedding)  # Truncate text

    def _store_in_memory(self, key: str, embedding: np.ndarray) -> None:
        """Store in memory cache with LRU eviction."""
        # Evict oldest entries if at capacity
        while len(self._cache) >= self.max_size and self._access_order:
            oldest = self._access_order.pop(0)
            self._cache.pop(oldest, None)

        self._cache[key] = embedding
        if key not in self._access_order:
            self._access_order.append(key)

    def _load_from_db(self, key: str) -> Optional[np.ndarray]:
        """Load embedding from database."""
        if not self.db_path:
            return None

        try:
            import sqlite3

            conn = sqlite3.connect(self.db_path, timeout=10.0)
            cursor = conn.execute(
                "SELECT embedding FROM embeddings WHERE text_hash = ?", (key,)
            )
            row = cursor.fetchone()
            conn.close()

            if row and row[0]:
                return np.frombuffer(row[0], dtype=np.float32)
        except (OSError, IOError) as e:
            # Expected: DB file issues, permissions
            logger.debug(f"Failed to load embedding from DB: {e}")
        except Exception as e:
            # Unexpected: log at warning for visibility
            logger.warning(f"Unexpected error loading embedding: {type(e).__name__}: {e}")

        return None

    def _save_to_db(self, key: str, text: str, embedding: np.ndarray) -> None:
        """Save embedding to database."""
        if not self.db_path:
            return

        try:
            import sqlite3
            import uuid

            conn = sqlite3.connect(self.db_path, timeout=10.0)
            conn.execute(
                """
                INSERT OR REPLACE INTO embeddings (id, text_hash, text, embedding, provider, created_at)
                VALUES (?, ?, ?, ?, ?, datetime('now'))
                """,
                (
                    str(uuid.uuid4()),
                    key,
                    text,
                    embedding.astype(np.float32).tobytes(),
                    "sentence-transformer",
                ),
            )
            conn.commit()
            conn.close()
        except (OSError, IOError) as e:
            # Expected: DB file issues, disk full, permissions
            logger.debug(f"Failed to save embedding to DB: {e}")
        except Exception as e:
            # Unexpected: log at warning for visibility
            logger.warning(f"Unexpected error saving embedding: {type(e).__name__}: {e}")

    def get_stats(self) -> dict:
        """Get cache statistics."""
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
            "size": len(self._cache),
            "max_size": self.max_size,
        }

    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._hits = 0
            self._misses = 0


# Global embedding cache instance
_embedding_cache: Optional[EmbeddingCache] = None
_embedding_cache_lock = threading.Lock()


def get_embedding_cache(
    max_size: int = 1024,
    persist: bool = False,
    db_path: Optional[str] = None,
) -> EmbeddingCache:
    """
    Get or create global embedding cache.

    Args:
        max_size: Maximum cache entries
        persist: Enable database persistence
        db_path: Database path for persistence

    Returns:
        EmbeddingCache instance
    """
    global _embedding_cache

    with _embedding_cache_lock:
        if _embedding_cache is None:
            # Default to core.db for persistence
            if persist and db_path is None:
                from aragora.persistence.db_config import (
                    get_db_path_str,
                    DatabaseType,
                )

                db_path = get_db_path_str(DatabaseType.EMBEDDINGS)

            _embedding_cache = EmbeddingCache(
                max_size=max_size,
                persist=persist,
                db_path=db_path,
            )

        return _embedding_cache


def reset_embedding_cache() -> None:
    """Reset the global embedding cache (for testing)."""
    global _embedding_cache
    with _embedding_cache_lock:
        if _embedding_cache is not None:
            _embedding_cache.clear()
        _embedding_cache = None


__all__ = [
    "EmbeddingCache",
    "get_embedding_cache",
    "reset_embedding_cache",
]
