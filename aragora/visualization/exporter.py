"""Export utilities for argument graphs.

Supports both in-memory and Redis caching backends for export content.
Redis backend is used when ARAGORA_REDIS_URL is configured, enabling
distributed caching across multiple server instances.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from aragora.visualization.mapper import ArgumentCartographer

logger = logging.getLogger(__name__)

# Cache configuration
_EXPORT_CACHE_TTL = 300.0  # 5 minutes
_MAX_CACHE_ENTRIES = 100
_CLEANUP_INTERVAL = 60.0  # Minimum seconds between cleanups
_REDIS_KEY_PREFIX = "aragora:export:"


class ExportCacheBackend(ABC):
    """Abstract base class for export cache backends."""

    @abstractmethod
    def get(self, debate_id: str, format_name: str, graph_hash: str) -> str | None:
        """Get cached export content."""
        ...

    @abstractmethod
    def set(self, debate_id: str, format_name: str, graph_hash: str, content: str) -> None:
        """Cache export content."""
        ...

    @abstractmethod
    def clear(self) -> int:
        """Clear all cached exports. Returns count cleared."""
        ...

    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        ...

    @abstractmethod
    def cleanup_expired(self) -> int:
        """Remove expired entries. Returns count removed."""
        ...


class InMemoryCacheBackend(ExportCacheBackend):
    """In-memory export cache with TTL and LRU eviction."""

    def __init__(self, ttl: float = _EXPORT_CACHE_TTL, max_entries: int = _MAX_CACHE_ENTRIES):
        self._cache: dict[tuple[str, str, str], tuple[str, float]] = {}
        self._lock = threading.Lock()
        self._ttl = ttl
        self._max_entries = max_entries
        self._last_cleanup = 0.0
        self._hits = 0
        self._misses = 0

    def _make_key(self, debate_id: str, format_name: str, graph_hash: str) -> tuple[str, str, str]:
        return (debate_id, format_name, graph_hash)

    def get(self, debate_id: str, format_name: str, graph_hash: str) -> str | None:
        key = self._make_key(debate_id, format_name, graph_hash)
        with self._lock:
            if key in self._cache:
                content, timestamp = self._cache[key]
                if time.time() - timestamp < self._ttl:
                    self._hits += 1
                    return content
                else:
                    del self._cache[key]
            self._misses += 1
        return None

    def set(self, debate_id: str, format_name: str, graph_hash: str, content: str) -> None:
        self._maybe_cleanup()
        key = self._make_key(debate_id, format_name, graph_hash)
        with self._lock:
            if len(self._cache) >= self._max_entries:
                self._evict_one()
            self._cache[key] = (content, time.time())

    def _evict_one(self) -> None:
        """Evict oldest entry. Must be called with lock held."""
        now = time.time()
        # First try to remove expired
        expired = [k for k, (_, ts) in self._cache.items() if now - ts > self._ttl]
        if expired:
            del self._cache[expired[0]]
        elif self._cache:
            oldest = min(self._cache.keys(), key=lambda k: self._cache[k][1])
            del self._cache[oldest]

    def _maybe_cleanup(self) -> None:
        now = time.time()
        if now - self._last_cleanup >= _CLEANUP_INTERVAL:
            self.cleanup_expired()

    def clear(self) -> int:
        with self._lock:
            count = len(self._cache)
            self._cache = {}
            self._hits = 0
            self._misses = 0
            return count

    def cleanup_expired(self) -> int:
        with self._lock:
            now = time.time()
            expired = [k for k, (_, ts) in self._cache.items() if now - ts > self._ttl]
            for k in expired:
                del self._cache[k]
            self._last_cleanup = now
            return len(expired)

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            now = time.time()
            total = len(self._cache)
            expired = sum(1 for _, (_, ts) in self._cache.items() if now - ts > self._ttl)
            total_size = sum(len(c) for c, _ in self._cache.values())
            hit_rate = (
                self._hits / (self._hits + self._misses) if (self._hits + self._misses) > 0 else 0.0
            )
            return {
                "backend": "in_memory",
                "total_entries": total,
                "expired_entries": expired,
                "active_entries": total - expired,
                "max_entries": self._max_entries,
                "ttl_seconds": self._ttl,
                "estimated_memory_bytes": total_size,
                "cleanup_interval_seconds": _CLEANUP_INTERVAL,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(hit_rate, 3),
            }


class RedisCacheBackend(ExportCacheBackend):
    """Redis-backed export cache for distributed deployments."""

    def __init__(self, redis_url: str, ttl: float = _EXPORT_CACHE_TTL):
        self._redis_url = redis_url
        self._ttl = int(ttl)
        self._client: Any = None
        self._hits = 0
        self._misses = 0

    def _get_client(self) -> Any:
        """Lazy initialize Redis client."""
        if self._client is None:
            try:
                import redis

                self._client = redis.from_url(self._redis_url, decode_responses=True)
            except ImportError:
                logger.warning("redis package not installed, falling back to in-memory cache")
                raise
        return self._client

    def _make_key(self, debate_id: str, format_name: str, graph_hash: str) -> str:
        return f"{_REDIS_KEY_PREFIX}{debate_id}:{format_name}:{graph_hash}"

    def get(self, debate_id: str, format_name: str, graph_hash: str) -> str | None:
        try:
            client = self._get_client()
            key = self._make_key(debate_id, format_name, graph_hash)
            content = client.get(key)
            if content:
                self._hits += 1
                return content
            self._misses += 1
            return None
        except (ConnectionError, TimeoutError, OSError, ValueError) as e:
            logger.debug("Redis cache get failed: %s", e)
            self._misses += 1
            return None

    def set(self, debate_id: str, format_name: str, graph_hash: str, content: str) -> None:
        try:
            client = self._get_client()
            key = self._make_key(debate_id, format_name, graph_hash)
            client.setex(key, self._ttl, content)
        except (ConnectionError, TimeoutError, OSError, ValueError) as e:
            logger.debug("Redis cache set failed: %s", e)

    def clear(self) -> int:
        try:
            client = self._get_client()
            pattern = f"{_REDIS_KEY_PREFIX}*"
            keys = list(client.scan_iter(match=pattern, count=100))
            if keys:
                client.delete(*keys)
            return len(keys)
        except (ConnectionError, TimeoutError, OSError, ValueError) as e:
            logger.debug("Redis cache clear failed: %s", e)
            return 0

    def cleanup_expired(self) -> int:
        # Redis handles TTL expiration automatically
        return 0

    def get_stats(self) -> dict[str, Any]:
        try:
            client = self._get_client()
            pattern = f"{_REDIS_KEY_PREFIX}*"
            keys = list(client.scan_iter(match=pattern, count=1000))
            total_size = 0
            for key in keys[:100]:  # Sample first 100 for size estimate
                try:
                    val = client.get(key)
                    if val:
                        total_size += len(val)
                except (ConnectionError, TimeoutError, OSError):
                    # Skip keys that fail to retrieve (connection issues)
                    pass
            avg_size = total_size / len(keys[:100]) if keys else 0
            estimated_total = int(avg_size * len(keys))
            hit_rate = (
                self._hits / (self._hits + self._misses) if (self._hits + self._misses) > 0 else 0.0
            )
            return {
                "backend": "redis",
                "total_entries": len(keys),
                "ttl_seconds": self._ttl,
                "estimated_memory_bytes": estimated_total,
                "redis_url": (
                    self._redis_url.split("@")[-1] if "@" in self._redis_url else "localhost"
                ),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(hit_rate, 3),
            }
        except (ConnectionError, TimeoutError, OSError, ValueError) as e:
            logger.debug("Redis stats failed: %s", e)
            return {"backend": "redis", "error": "Failed to fetch Redis stats"}


# Singleton cache backend
_cache_backend: ExportCacheBackend | None = None
_cache_backend_lock = threading.Lock()


def _get_cache_backend() -> ExportCacheBackend:
    """Get or create the cache backend singleton."""
    global _cache_backend
    if _cache_backend is None:
        with _cache_backend_lock:
            if _cache_backend is None:
                redis_url = os.environ.get("ARAGORA_REDIS_URL") or os.environ.get("REDIS_URL")
                if redis_url:
                    try:
                        _cache_backend = RedisCacheBackend(redis_url)
                        # Test connection
                        _cache_backend.get_stats()
                        logger.info("Using Redis backend for export cache")
                    except (ImportError, ConnectionError, TimeoutError, OSError, ValueError) as e:
                        logger.warning("Redis unavailable (%s), using in-memory cache", e)
                        _cache_backend = InMemoryCacheBackend()
                else:
                    _cache_backend = InMemoryCacheBackend()
    return _cache_backend


def _get_graph_hash(cartographer: ArgumentCartographer) -> str:
    """Get a hash of the current graph state for caching."""
    stats = cartographer.get_statistics()
    # Include key metrics that affect output
    hash_input = f"{stats['node_count']}:{stats['edge_count']}:{stats['rounds']}:{','.join(sorted(stats['agents']))}"
    return hashlib.sha256(hash_input.encode()).hexdigest()[:16]


def _get_cached_export(debate_id: str, format_name: str, graph_hash: str) -> str | None:
    """Get cached export if valid. Uses configured backend (Redis or in-memory)."""
    return _get_cache_backend().get(debate_id, format_name, graph_hash)


def _cache_export(debate_id: str, format_name: str, graph_hash: str, content: str) -> None:
    """Cache an export. Uses configured backend (Redis or in-memory)."""
    _get_cache_backend().set(debate_id, format_name, graph_hash, content)


def cleanup_expired_exports() -> int:
    """
    Remove expired entries from the export cache.

    This function proactively removes all expired entries, not just when
    they're accessed. Call periodically to prevent memory buildup.

    Note: For Redis backend, TTL expiration is automatic.

    Returns:
        Number of entries removed.
    """
    return _get_cache_backend().cleanup_expired()


def clear_export_cache() -> int:
    """Clear the export cache. Returns number of entries cleared."""
    return _get_cache_backend().clear()


def get_export_cache_stats() -> dict[str, Any]:
    """
    Get statistics about the export cache.

    Returns:
        Dictionary with cache statistics including count, expired count,
        memory estimate, TTL configuration, and backend type.
    """
    return _get_cache_backend().get_stats()


def save_debate_visualization(
    cartographer: ArgumentCartographer,
    output_dir: Path,
    debate_id: str,
    formats: list | None = None,
    use_cache: bool = True,
) -> dict:
    """
    Save debate visualization to multiple formats.

    Args:
        cartographer: The cartographer with the debate graph
        output_dir: Directory to save files
        debate_id: ID for naming files
        formats: List of formats to export ("mermaid", "json", "html")
        use_cache: Whether to use caching for export content

    Returns:
        Dictionary mapping format to output file path
    """
    formats = formats or ["mermaid", "json"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get graph hash for caching
    graph_hash = _get_graph_hash(cartographer) if use_cache else ""

    results = {}

    if "mermaid" in formats:
        mermaid_path = output_dir / f"{debate_id}_graph.mermaid"
        content = None
        if use_cache:
            content = _get_cached_export(debate_id, "mermaid", graph_hash)
        if content is None:
            content = cartographer.export_mermaid()
            if use_cache:
                _cache_export(debate_id, "mermaid", graph_hash, content)
        mermaid_path.write_text(content)
        results["mermaid"] = str(mermaid_path)

    if "json" in formats:
        json_path = output_dir / f"{debate_id}_graph.json"
        content = None
        if use_cache:
            content = _get_cached_export(debate_id, "json", graph_hash)
        if content is None:
            content = cartographer.export_json(include_full_content=True)
            if use_cache:
                _cache_export(debate_id, "json", graph_hash, content)
        json_path.write_text(content)
        results["json"] = str(json_path)

    if "html" in formats:
        html_path = output_dir / f"{debate_id}_graph.html"
        content = None
        if use_cache:
            content = _get_cached_export(debate_id, "html", graph_hash)
        if content is None:
            content = generate_standalone_html(cartographer)
            if use_cache:
                _cache_export(debate_id, "html", graph_hash, content)
        html_path.write_text(content)
        results["html"] = str(html_path)

    return results


def generate_standalone_html(
    cartographer: ArgumentCartographer,
    title: str = "Debate Argument Map",
) -> str:
    """Generate a self-contained HTML file with an interactive force-directed graph.

    This is a convenience wrapper around
    :meth:`ArgumentCartographer.export_html`.  The returned HTML uses inline
    vanilla JavaScript (no CDN dependencies) and works fully offline.

    Args:
        cartographer: The cartographer holding the debate graph.
        title: Page title shown in the browser tab and header.

    Returns:
        A complete HTML string.
    """
    return cartographer.export_html(title=title)
