"""
Redis-backed RBAC Cache for Distributed Deployments.

Provides a distributed cache for RBAC permission decisions that works across
multiple server instances using Redis as the backend with in-memory fallback.

Features:
- Distributed cache coherence via Redis pub/sub
- Automatic cache invalidation propagation
- In-memory L1 cache for performance
- Metrics and statistics
- Graceful degradation when Redis unavailable

Usage:
    from aragora.rbac.cache import get_rbac_cache, RBACCacheConfig

    # Configure cache (usually done at startup)
    config = RBACCacheConfig.from_env()
    cache = get_rbac_cache(config)

    # Use with PermissionChecker
    from aragora.rbac.checker import PermissionChecker
    checker = PermissionChecker(cache_backend=cache)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class RBACCacheConfig:
    """Configuration for RBAC distributed cache."""

    # Redis settings
    redis_url: Optional[str] = None
    redis_prefix: str = "aragora:rbac"

    # TTL settings
    decision_ttl_seconds: int = 300  # 5 minutes for permission decisions
    role_ttl_seconds: int = 600  # 10 minutes for role assignments
    permission_ttl_seconds: int = 900  # 15 minutes for permission sets

    # L1 (in-memory) cache settings
    l1_enabled: bool = True
    l1_max_size: int = 10000
    l1_ttl_seconds: int = 60  # Short TTL for L1 (1 minute)

    # Invalidation settings
    enable_pubsub: bool = True
    invalidation_channel: str = "aragora:rbac:invalidate"

    # Stats settings
    enable_metrics: bool = True

    @classmethod
    def from_env(cls) -> "RBACCacheConfig":
        """Create config from environment variables."""
        return cls(
            redis_url=os.environ.get("REDIS_URL") or os.environ.get("ARAGORA_REDIS_URL"),
            redis_prefix=os.environ.get("RBAC_CACHE_PREFIX", "aragora:rbac"),
            decision_ttl_seconds=int(os.environ.get("RBAC_CACHE_DECISION_TTL", "300")),
            role_ttl_seconds=int(os.environ.get("RBAC_CACHE_ROLE_TTL", "600")),
            permission_ttl_seconds=int(os.environ.get("RBAC_CACHE_PERMISSION_TTL", "900")),
            l1_enabled=os.environ.get("RBAC_CACHE_L1_ENABLED", "true").lower() == "true",
            l1_max_size=int(os.environ.get("RBAC_CACHE_L1_MAX_SIZE", "10000")),
            l1_ttl_seconds=int(os.environ.get("RBAC_CACHE_L1_TTL", "60")),
            enable_pubsub=os.environ.get("RBAC_CACHE_PUBSUB", "true").lower() == "true",
            enable_metrics=os.environ.get("RBAC_CACHE_METRICS", "true").lower() == "true",
        )


@dataclass
class CacheStats:
    """Statistics for RBAC cache."""

    l1_hits: int = 0
    l1_misses: int = 0
    l2_hits: int = 0
    l2_misses: int = 0
    invalidations: int = 0
    pubsub_messages: int = 0
    errors: int = 0
    evictions: int = 0

    @property
    def total_hits(self) -> int:
        return self.l1_hits + self.l2_hits

    @property
    def total_misses(self) -> int:
        # L2 miss only counts if L1 also missed
        return self.l2_misses

    @property
    def hit_rate(self) -> float:
        total = self.total_hits + self.total_misses
        return self.total_hits / total if total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "l1_hits": self.l1_hits,
            "l1_misses": self.l1_misses,
            "l2_hits": self.l2_hits,
            "l2_misses": self.l2_misses,
            "total_hits": self.total_hits,
            "total_misses": self.total_misses,
            "hit_rate": self.hit_rate,
            "invalidations": self.invalidations,
            "pubsub_messages": self.pubsub_messages,
            "errors": self.errors,
            "evictions": self.evictions,
        }


class RBACDistributedCache:
    """
    Distributed RBAC cache with Redis backend and in-memory L1.

    Architecture:
    - L1 Cache: Local in-memory OrderedDict (fast, per-instance)
    - L2 Cache: Redis (shared, distributed)
    - Invalidation: Redis pub/sub for cross-instance invalidation

    Cache Keys:
    - decision:{user}:{org}:{roles_hash}:{permission}:{resource}
    - roles:{user}:{org}
    - permissions:{role}
    """

    def __init__(self, config: Optional[RBACCacheConfig] = None):
        """Initialize the distributed cache."""
        self.config = config or RBACCacheConfig.from_env()
        self._stats = CacheStats()

        # L1 (in-memory) cache
        self._l1_cache: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._l1_lock = threading.RLock()

        # Redis client (lazy initialized)
        self._redis: Optional[Any] = None
        self._redis_checked = False
        self._pubsub: Optional[Any] = None
        self._pubsub_thread: Optional[threading.Thread] = None
        self._running = False

        # Invalidation callbacks
        self._invalidation_callbacks: list[Callable[[str], None]] = []

    @property
    def stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats

    @property
    def is_distributed(self) -> bool:
        """Check if Redis is available for distributed caching."""
        return self._get_redis() is not None

    def _get_redis(self) -> Optional[Any]:
        """Get Redis client (lazy initialization)."""
        if self._redis_checked:
            return self._redis

        if not self.config.redis_url:
            self._redis_checked = True
            return None

        try:
            import redis

            self._redis = redis.from_url(
                self.config.redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
            )
            # Test connection
            self._redis.ping()
            self._redis_checked = True
            logger.info("RBAC cache connected to Redis (distributed mode)")
            return self._redis
        except Exception as e:
            logger.warning(f"RBAC cache Redis unavailable, using local-only: {e}")
            self._redis_checked = True
            self._redis = None
            return None

    def start(self) -> None:
        """Start the cache (initializes pub/sub listener)."""
        if self._running:
            return

        self._running = True
        redis = self._get_redis()

        if redis and self.config.enable_pubsub:
            try:
                self._pubsub = redis.pubsub()
                self._pubsub.subscribe(self.config.invalidation_channel)
                self._pubsub_thread = threading.Thread(
                    target=self._pubsub_listener,
                    daemon=True,
                    name="rbac-cache-pubsub",
                )
                self._pubsub_thread.start()
                logger.debug("RBAC cache pub/sub listener started")
            except Exception as e:
                logger.warning(f"Failed to start RBAC cache pub/sub: {e}")

    def stop(self) -> None:
        """Stop the cache (cleanup pub/sub)."""
        self._running = False
        if self._pubsub:
            try:
                self._pubsub.unsubscribe()
                self._pubsub.close()
            except Exception as e:  # noqa: BLE001 - Cleanup must not raise
                logger.debug(f"RBAC pub/sub cleanup error: {e}")
            self._pubsub = None

    def _pubsub_listener(self) -> None:
        """Listen for invalidation messages from other instances."""
        while self._running and self._pubsub:
            try:
                message = self._pubsub.get_message(timeout=1.0)
                if message and message.get("type") == "message":
                    self._stats.pubsub_messages += 1
                    key = message.get("data", "")
                    if key:
                        self._invalidate_local(key)
            except Exception as e:
                if self._running:
                    logger.debug(f"RBAC pub/sub error: {e}")
                    time.sleep(1)

    def _redis_key(self, key_type: str, *parts: str) -> str:
        """Build Redis key with prefix."""
        key_parts = [self.config.redis_prefix, key_type] + list(parts)
        return ":".join(key_parts)

    def _l1_key(self, key: str) -> str:
        """Create L1 cache key (hash long keys)."""
        if len(key) > 100:
            return hashlib.sha256(key.encode()).hexdigest()[:32]
        return key

    # --------------------------------------------------------------------------
    # Permission Decision Cache
    # --------------------------------------------------------------------------

    def get_decision(
        self,
        user_id: str,
        org_id: Optional[str],
        roles_hash: str,
        permission_key: str,
        resource_id: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """Get cached permission decision."""
        cache_key = self._decision_key(user_id, org_id, roles_hash, permission_key, resource_id)

        # Check L1 first
        if self.config.l1_enabled:
            result = self._l1_get(cache_key)
            if result is not None:
                self._stats.l1_hits += 1
                self._record_cache_hit("decision", True)
                return result
            self._stats.l1_misses += 1

        # Check L2 (Redis)
        redis = self._get_redis()
        if redis:
            try:
                redis_key = self._redis_key("decision", cache_key)
                data = redis.get(redis_key)
                if data:
                    self._stats.l2_hits += 1
                    self._record_cache_hit("decision", False)
                    result = json.loads(data)
                    # Populate L1
                    if self.config.l1_enabled:
                        self._l1_set(cache_key, result)
                    return result
                self._stats.l2_misses += 1
            except Exception as e:
                self._stats.errors += 1
                logger.debug(f"Redis get_decision error: {e}")

        self._record_cache_miss("decision")
        return None

    def set_decision(
        self,
        user_id: str,
        org_id: Optional[str],
        roles_hash: str,
        permission_key: str,
        resource_id: Optional[str],
        decision: dict[str, Any],
    ) -> None:
        """Cache a permission decision."""
        cache_key = self._decision_key(user_id, org_id, roles_hash, permission_key, resource_id)

        # Set in L1
        if self.config.l1_enabled:
            self._l1_set(cache_key, decision)

        # Set in L2 (Redis)
        redis = self._get_redis()
        if redis:
            try:
                redis_key = self._redis_key("decision", cache_key)
                redis.setex(
                    redis_key,
                    self.config.decision_ttl_seconds,
                    json.dumps(decision, default=str),
                )
            except Exception as e:
                self._stats.errors += 1
                logger.debug(f"Redis set_decision error: {e}")

    def _decision_key(
        self,
        user_id: str,
        org_id: Optional[str],
        roles_hash: str,
        permission_key: str,
        resource_id: Optional[str],
    ) -> str:
        """Build cache key for permission decision."""
        return f"{user_id}:{org_id or ''}:{roles_hash}:{permission_key}:{resource_id or ''}"

    # --------------------------------------------------------------------------
    # Role Assignment Cache
    # --------------------------------------------------------------------------

    def get_user_roles(self, user_id: str, org_id: Optional[str] = None) -> Optional[set[str]]:
        """Get cached roles for a user."""
        cache_key = f"{user_id}:{org_id or ''}"

        # Check L1
        if self.config.l1_enabled:
            result = self._l1_get(f"roles:{cache_key}")
            if result is not None:
                self._stats.l1_hits += 1
                return set(result)
            self._stats.l1_misses += 1

        # Check L2
        redis = self._get_redis()
        if redis:
            try:
                redis_key = self._redis_key("roles", cache_key)
                data = redis.get(redis_key)
                if data:
                    self._stats.l2_hits += 1
                    roles = set(json.loads(data))
                    if self.config.l1_enabled:
                        self._l1_set(f"roles:{cache_key}", list(roles))
                    return roles
                self._stats.l2_misses += 1
            except Exception as e:
                self._stats.errors += 1
                logger.debug(f"Redis get_user_roles error: {e}")

        return None

    def set_user_roles(self, user_id: str, org_id: Optional[str], roles: set[str]) -> None:
        """Cache roles for a user."""
        cache_key = f"{user_id}:{org_id or ''}"

        if self.config.l1_enabled:
            self._l1_set(f"roles:{cache_key}", list(roles))

        redis = self._get_redis()
        if redis:
            try:
                redis_key = self._redis_key("roles", cache_key)
                redis.setex(
                    redis_key,
                    self.config.role_ttl_seconds,
                    json.dumps(list(roles)),
                )
            except Exception as e:
                self._stats.errors += 1
                logger.debug(f"Redis set_user_roles error: {e}")

    # --------------------------------------------------------------------------
    # Permission Set Cache
    # --------------------------------------------------------------------------

    def get_role_permissions(self, role_name: str) -> Optional[set[str]]:
        """Get cached permissions for a role."""
        cache_key = f"perms:{role_name}"

        if self.config.l1_enabled:
            result = self._l1_get(cache_key)
            if result is not None:
                self._stats.l1_hits += 1
                return set(result)
            self._stats.l1_misses += 1

        redis = self._get_redis()
        if redis:
            try:
                redis_key = self._redis_key("permissions", role_name)
                data = redis.get(redis_key)
                if data:
                    self._stats.l2_hits += 1
                    perms = set(json.loads(data))
                    if self.config.l1_enabled:
                        self._l1_set(cache_key, list(perms))
                    return perms
                self._stats.l2_misses += 1
            except Exception as e:
                self._stats.errors += 1
                logger.debug(f"Redis get_role_permissions error: {e}")

        return None

    def set_role_permissions(self, role_name: str, permissions: set[str]) -> None:
        """Cache permissions for a role."""
        cache_key = f"perms:{role_name}"

        if self.config.l1_enabled:
            self._l1_set(cache_key, list(permissions))

        redis = self._get_redis()
        if redis:
            try:
                redis_key = self._redis_key("permissions", role_name)
                redis.setex(
                    redis_key,
                    self.config.permission_ttl_seconds,
                    json.dumps(list(permissions)),
                )
            except Exception as e:
                self._stats.errors += 1
                logger.debug(f"Redis set_role_permissions error: {e}")

    # --------------------------------------------------------------------------
    # Invalidation
    # --------------------------------------------------------------------------

    def invalidate_user(self, user_id: str) -> None:
        """Invalidate all cache entries for a user (decisions and roles)."""
        self._stats.invalidations += 1

        # Build pattern for this user
        pattern = f"{user_id}:*"

        # Invalidate L1
        self._invalidate_l1_pattern(pattern)
        self._invalidate_l1_pattern(f"roles:{user_id}:*")

        # Invalidate L2 and broadcast
        redis = self._get_redis()
        if redis:
            try:
                # Delete matching keys
                for key_type in ["decision", "roles"]:
                    full_pattern = self._redis_key(key_type, pattern)
                    keys = redis.keys(full_pattern)
                    if keys:
                        redis.delete(*keys)

                # Broadcast invalidation
                if self.config.enable_pubsub:
                    redis.publish(self.config.invalidation_channel, f"user:{user_id}")
            except Exception as e:
                self._stats.errors += 1
                logger.debug(f"Redis invalidate_user error: {e}")

        # Call callbacks
        for callback in self._invalidation_callbacks:
            try:
                callback(f"user:{user_id}")
            except Exception as e:  # noqa: BLE001 - Callbacks must not break invalidation
                logger.debug(f"RBAC invalidation callback error: {e}")

    def invalidate_role(self, role_name: str) -> None:
        """Invalidate cached permissions for a role (affects all users)."""
        self._stats.invalidations += 1

        cache_key = f"perms:{role_name}"
        self._l1_delete(cache_key)

        redis = self._get_redis()
        if redis:
            try:
                redis_key = self._redis_key("permissions", role_name)
                redis.delete(redis_key)

                if self.config.enable_pubsub:
                    redis.publish(self.config.invalidation_channel, f"role:{role_name}")
            except Exception as e:
                self._stats.errors += 1
                logger.debug(f"Redis invalidate_role error: {e}")

    def invalidate_all(self) -> int:
        """Clear entire cache. Returns count of local entries cleared."""
        self._stats.invalidations += 1

        # Clear L1
        with self._l1_lock:
            count = len(self._l1_cache)
            self._l1_cache.clear()

        # Clear L2
        redis = self._get_redis()
        if redis:
            try:
                for key_type in ["decision", "roles", "permissions"]:
                    pattern = self._redis_key(key_type, "*")
                    keys = redis.keys(pattern)
                    if keys:
                        redis.delete(*keys)

                if self.config.enable_pubsub:
                    redis.publish(self.config.invalidation_channel, "all")
            except Exception as e:
                self._stats.errors += 1
                logger.debug(f"Redis invalidate_all error: {e}")

        return count

    def _invalidate_local(self, key: str) -> None:
        """Handle invalidation from pub/sub (local only)."""
        if key == "all":
            with self._l1_lock:
                self._l1_cache.clear()
        elif key.startswith("user:"):
            user_id = key[5:]
            self._invalidate_l1_pattern(f"{user_id}:*")
            self._invalidate_l1_pattern(f"roles:{user_id}:*")
        elif key.startswith("role:"):
            role_name = key[5:]
            self._l1_delete(f"perms:{role_name}")

    def add_invalidation_callback(self, callback: Callable[[str], None]) -> None:
        """Add callback to be called on invalidation."""
        self._invalidation_callbacks.append(callback)

    # --------------------------------------------------------------------------
    # L1 Cache Operations
    # --------------------------------------------------------------------------

    def _l1_get(self, key: str) -> Optional[Any]:
        """Get from L1 cache."""
        l1_key = self._l1_key(key)
        with self._l1_lock:
            if l1_key in self._l1_cache:
                cached_time, value = self._l1_cache[l1_key]
                if time.time() - cached_time < self.config.l1_ttl_seconds:
                    self._l1_cache.move_to_end(l1_key)
                    return value
                else:
                    del self._l1_cache[l1_key]
        return None

    def _l1_set(self, key: str, value: Any) -> None:
        """Set in L1 cache."""
        l1_key = self._l1_key(key)
        with self._l1_lock:
            while len(self._l1_cache) >= self.config.l1_max_size:
                self._l1_cache.popitem(last=False)
                self._stats.evictions += 1
            self._l1_cache[l1_key] = (time.time(), value)

    def _l1_delete(self, key: str) -> bool:
        """Delete from L1 cache."""
        l1_key = self._l1_key(key)
        with self._l1_lock:
            if l1_key in self._l1_cache:
                del self._l1_cache[l1_key]
                return True
        return False

    def _invalidate_l1_pattern(self, pattern: str) -> int:
        """Invalidate L1 entries matching pattern (simple glob)."""
        count = 0
        prefix = pattern.rstrip("*")
        with self._l1_lock:
            keys_to_delete = [k for k in self._l1_cache if k.startswith(prefix)]
            for key in keys_to_delete:
                del self._l1_cache[key]
                count += 1
        return count

    # --------------------------------------------------------------------------
    # Metrics
    # --------------------------------------------------------------------------

    def _record_cache_hit(self, cache_type: str, l1: bool) -> None:
        """Record cache hit metric."""
        if not self.config.enable_metrics:
            return
        try:
            from aragora.observability.metrics.security import record_rbac_cache_hit

            record_rbac_cache_hit(cache_type, "l1" if l1 else "l2")
        except ImportError:
            pass

    def _record_cache_miss(self, cache_type: str) -> None:
        """Record cache miss metric."""
        if not self.config.enable_metrics:
            return
        try:
            from aragora.observability.metrics.security import record_rbac_cache_miss

            record_rbac_cache_miss(cache_type)
        except ImportError:
            pass

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive cache statistics."""
        redis = self._get_redis()
        with self._l1_lock:
            l1_size = len(self._l1_cache)

        return {
            **self._stats.to_dict(),
            "l1_size": l1_size,
            "l1_max_size": self.config.l1_max_size,
            "distributed": redis is not None,
            "pubsub_enabled": self.config.enable_pubsub and redis is not None,
        }


# Global cache instance
_rbac_cache: Optional[RBACDistributedCache] = None


def get_rbac_cache(config: Optional[RBACCacheConfig] = None) -> RBACDistributedCache:
    """Get or create the global RBAC cache instance."""
    global _rbac_cache

    if _rbac_cache is None:
        _rbac_cache = RBACDistributedCache(config)

    return _rbac_cache


def set_rbac_cache(cache: RBACDistributedCache) -> None:
    """Set the global RBAC cache instance."""
    global _rbac_cache
    _rbac_cache = cache


def reset_rbac_cache() -> None:
    """Reset the global RBAC cache (for testing)."""
    global _rbac_cache
    if _rbac_cache:
        _rbac_cache.stop()
    _rbac_cache = None


__all__ = [
    "RBACCacheConfig",
    "RBACDistributedCache",
    "CacheStats",
    "get_rbac_cache",
    "set_rbac_cache",
    "reset_rbac_cache",
]
