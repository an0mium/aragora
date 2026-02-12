"""
Message idempotency and deduplication for streaming connectors.

Prevents duplicate message processing by tracking message fingerprints
in Redis with configurable TTL.

Usage:
    from aragora.connectors.enterprise.streaming.idempotency import (
        IdempotencyTracker,
        MessageFingerprint,
    )

    # Create tracker with Redis
    tracker = IdempotencyTracker(redis_client, ttl_seconds=3600)

    # Check and mark message as processed
    fingerprint = tracker.compute_fingerprint(key, body)
    if await tracker.is_duplicate(fingerprint):
        logger.info("Skipping duplicate message")
        return

    # Process message...

    # Mark as processed
    await tracker.mark_processed(fingerprint)
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class MessageFingerprint:
    """Unique identifier for a message based on its content.

    Attributes:
        hash: SHA256 hash of the message content
        key: Original message key (if any)
        topic: Topic/queue the message came from
        created_at: Timestamp when fingerprint was created
    """

    hash: str
    key: str | None = None
    topic: str | None = None
    created_at: float = field(default_factory=time.time)

    @property
    def redis_key(self) -> str:
        """Redis key for storing this fingerprint."""
        return f"idempotency:{self.hash}"


class IdempotencyTracker:
    """
    Tracks processed messages to prevent duplicate processing.

    Uses Redis to store message fingerprints with configurable TTL.
    Fingerprints are SHA256 hashes of the message key + body.

    Features:
    - Content-based deduplication (same content = same fingerprint)
    - Configurable deduplication window (TTL)
    - Metrics for duplicate detection
    - Optional fallback to in-memory cache when Redis unavailable
    """

    def __init__(
        self,
        redis_client: Any | None = None,
        ttl_seconds: int = 3600,
        key_prefix: str = "idempotency",
        use_memory_fallback: bool = True,
    ):
        """Initialize the idempotency tracker.

        Args:
            redis_client: Redis client instance (optional)
            ttl_seconds: How long to remember processed messages (default: 1 hour)
            key_prefix: Redis key prefix (default: "idempotency")
            use_memory_fallback: Use in-memory cache when Redis unavailable
        """
        self._redis = redis_client
        self._ttl = ttl_seconds
        self._key_prefix = key_prefix
        self._use_memory_fallback = use_memory_fallback

        # In-memory fallback cache (LRU-style with TTL)
        self._memory_cache: dict[str, float] = {}  # fingerprint -> expiry timestamp
        self._memory_cache_max_size = 10000

        # Metrics
        self._total_checked = 0
        self._total_duplicates = 0
        self._cache_hits = 0
        self._redis_hits = 0

    def compute_fingerprint(
        self,
        key: str | bytes | None,
        body: str | bytes | dict | None,
        topic: str | None = None,
        include_headers: dict[str, Any] | None = None,
    ) -> MessageFingerprint:
        """Compute a fingerprint for a message.

        The fingerprint is a SHA256 hash of:
        - Message key (if present)
        - Message body (serialized to JSON if dict)
        - Selected headers (if specified)

        Args:
            key: Message key
            body: Message body (str, bytes, or dict)
            topic: Topic/queue name (for context)
            include_headers: Optional headers to include in fingerprint

        Returns:
            MessageFingerprint with computed hash
        """
        hasher = hashlib.sha256()

        # Include key
        if key is not None:
            if isinstance(key, str):
                hasher.update(key.encode("utf-8"))
            else:
                hasher.update(key)

        # Include body
        if body is not None:
            if isinstance(body, dict):
                # Sort keys for consistent serialization
                body_bytes = json.dumps(body, sort_keys=True).encode("utf-8")
            elif isinstance(body, str):
                body_bytes = body.encode("utf-8")
            else:
                body_bytes = body
            hasher.update(body_bytes)

        # Include headers if specified
        if include_headers:
            headers_str = json.dumps(include_headers, sort_keys=True)
            hasher.update(headers_str.encode("utf-8"))

        hash_value = hasher.hexdigest()
        key_str = key.decode("utf-8") if isinstance(key, bytes) else key

        return MessageFingerprint(
            hash=hash_value,
            key=key_str,
            topic=topic,
        )

    async def is_duplicate(self, fingerprint: MessageFingerprint) -> bool:
        """Check if a message with this fingerprint was already processed.

        Args:
            fingerprint: Message fingerprint to check

        Returns:
            True if message was already processed, False otherwise
        """
        self._total_checked += 1
        redis_key = f"{self._key_prefix}:{fingerprint.hash}"

        # Try Redis first
        if self._redis is not None:
            try:
                exists = await self._redis_exists(redis_key)
                if exists:
                    self._total_duplicates += 1
                    self._redis_hits += 1
                    logger.debug(
                        "Duplicate message detected (Redis): %s",
                        fingerprint.hash[:16],
                    )
                    return True
            except (ConnectionError, TimeoutError, OSError, AttributeError) as e:
                logger.warning("Redis check failed, using memory fallback: %s", e)

        # Fallback to memory cache
        if self._use_memory_fallback:
            self._cleanup_memory_cache()
            if fingerprint.hash in self._memory_cache:
                if self._memory_cache[fingerprint.hash] > time.time():
                    self._total_duplicates += 1
                    self._cache_hits += 1
                    logger.debug(
                        "Duplicate message detected (memory): %s",
                        fingerprint.hash[:16],
                    )
                    return True
                else:
                    # Expired entry
                    del self._memory_cache[fingerprint.hash]

        return False

    async def mark_processed(
        self,
        fingerprint: MessageFingerprint,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Mark a message as processed.

        Args:
            fingerprint: Message fingerprint
            metadata: Optional metadata to store with the fingerprint

        Returns:
            True if successfully marked, False otherwise
        """
        redis_key = f"{self._key_prefix}:{fingerprint.hash}"

        # Store in Redis
        if self._redis is not None:
            try:
                value = json.dumps(
                    {
                        "hash": fingerprint.hash,
                        "key": fingerprint.key,
                        "topic": fingerprint.topic,
                        "processed_at": time.time(),
                        "metadata": metadata or {},
                    }
                )
                await self._redis_setex(redis_key, self._ttl, value)
                return True
            except (TypeError, ConnectionError, TimeoutError, OSError, AttributeError) as e:
                logger.warning("Redis mark failed, using memory fallback: %s", e)

        # Fallback to memory cache
        if self._use_memory_fallback:
            self._cleanup_memory_cache()
            self._memory_cache[fingerprint.hash] = time.time() + self._ttl
            return True

        return False

    async def check_and_mark(
        self,
        fingerprint: MessageFingerprint,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Check if duplicate and mark as processed atomically.

        Uses Redis SETNX for atomic check-and-set.

        Args:
            fingerprint: Message fingerprint
            metadata: Optional metadata to store

        Returns:
            True if this is a new message (processed), False if duplicate
        """
        redis_key = f"{self._key_prefix}:{fingerprint.hash}"
        self._total_checked += 1

        # Try Redis atomic operation
        if self._redis is not None:
            try:
                value = json.dumps(
                    {
                        "hash": fingerprint.hash,
                        "key": fingerprint.key,
                        "topic": fingerprint.topic,
                        "processed_at": time.time(),
                        "metadata": metadata or {},
                    }
                )
                # SETNX returns True if key was set (new message)
                was_set = await self._redis_setnx(redis_key, value, self._ttl)
                if not was_set:
                    self._total_duplicates += 1
                    self._redis_hits += 1
                    logger.debug(
                        "Duplicate message (atomic check): %s",
                        fingerprint.hash[:16],
                    )
                    return False
                return True
            except (TypeError, ConnectionError, TimeoutError, OSError, AttributeError) as e:
                logger.warning("Redis atomic check failed: %s", e)

        # Fallback to non-atomic check (memory)
        if self._use_memory_fallback:
            if await self.is_duplicate(fingerprint):
                return False
            await self.mark_processed(fingerprint, metadata)
            return True

        return True  # Process if we can't track

    def get_stats(self) -> dict[str, Any]:
        """Get deduplication statistics.

        Returns:
            Dict with metrics about deduplication
        """
        return {
            "total_checked": self._total_checked,
            "total_duplicates": self._total_duplicates,
            "duplicate_rate": (
                (self._total_duplicates / self._total_checked * 100)
                if self._total_checked > 0
                else 0.0
            ),
            "cache_hits": self._cache_hits,
            "redis_hits": self._redis_hits,
            "memory_cache_size": len(self._memory_cache),
            "ttl_seconds": self._ttl,
            "redis_available": self._redis is not None,
        }

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self._total_checked = 0
        self._total_duplicates = 0
        self._cache_hits = 0
        self._redis_hits = 0

    def _cleanup_memory_cache(self) -> None:
        """Remove expired entries and enforce size limit."""
        current_time = time.time()

        # Remove expired entries
        expired_keys = [k for k, expiry in self._memory_cache.items() if expiry <= current_time]
        for k in expired_keys:
            del self._memory_cache[k]

        # Enforce size limit (remove oldest entries)
        if len(self._memory_cache) > self._memory_cache_max_size:
            # Sort by expiry and remove oldest half
            sorted_items = sorted(self._memory_cache.items(), key=lambda x: x[1])
            items_to_remove = len(self._memory_cache) - (self._memory_cache_max_size // 2)
            for k, _ in sorted_items[:items_to_remove]:
                del self._memory_cache[k]

    async def _redis_exists(self, key: str) -> bool:
        """Check if key exists in Redis."""
        if hasattr(self._redis, "exists"):
            # async redis
            result = await self._redis.exists(key)
            return bool(result)
        return False

    async def _redis_setex(self, key: str, ttl: int, value: str) -> bool:
        """Set key with expiry in Redis."""
        if hasattr(self._redis, "setex"):
            await self._redis.setex(key, ttl, value)
            return True
        if hasattr(self._redis, "set"):
            await self._redis.set(key, value, ex=ttl)
            return True
        return False

    async def _redis_setnx(self, key: str, value: str, ttl: int) -> bool:
        """Set key if not exists (atomic) with expiry."""
        if hasattr(self._redis, "set"):
            # Use SET with NX and EX options
            result = await self._redis.set(key, value, nx=True, ex=ttl)
            return result is not None
        return False


# Convenience function for creating tracker with global Redis
def create_idempotency_tracker(
    ttl_seconds: int = 3600,
    key_prefix: str = "aragora:idempotency",
) -> IdempotencyTracker:
    """Create an idempotency tracker with the global Redis client.

    Args:
        ttl_seconds: Deduplication window in seconds
        key_prefix: Redis key prefix

    Returns:
        Configured IdempotencyTracker
    """
    redis_client = None
    try:
        from aragora.server.redis_cluster import get_redis_client

        redis_client = get_redis_client()
    except ImportError:
        logger.warning("Redis client not available, using memory-only idempotency")

    return IdempotencyTracker(
        redis_client=redis_client,
        ttl_seconds=ttl_seconds,
        key_prefix=key_prefix,
        use_memory_fallback=True,
    )


__all__ = [
    "IdempotencyTracker",
    "MessageFingerprint",
    "create_idempotency_tracker",
]
