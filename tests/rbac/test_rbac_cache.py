"""
Tests for RBAC Distributed Cache.

Covers:
- RBACCacheConfig configuration
- CacheStats statistics tracking
- RBACDistributedCache L1 cache operations
- Permission decision caching
- Role assignment caching
- Permission set caching
- Cache invalidation (user, role, all)
- Metrics recording
- Global cache management
"""

from __future__ import annotations

import pytest
import time
import threading
from unittest.mock import MagicMock, patch, PropertyMock

from aragora.rbac.cache import (
    RBACCacheConfig,
    CacheStats,
    RBACDistributedCache,
    get_rbac_cache,
    set_rbac_cache,
    reset_rbac_cache,
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_global_cache():
    """Reset global cache before and after each test."""
    reset_rbac_cache()
    yield
    reset_rbac_cache()


@pytest.fixture
def cache_config():
    """Create a test cache configuration."""
    return RBACCacheConfig(
        redis_url=None,  # No Redis for tests
        decision_ttl_seconds=300,
        role_ttl_seconds=600,
        permission_ttl_seconds=900,
        l1_enabled=True,
        l1_max_size=100,
        l1_ttl_seconds=60,
        enable_pubsub=False,
        enable_metrics=False,
    )


@pytest.fixture
def cache(cache_config):
    """Create a cache instance for testing."""
    return RBACDistributedCache(cache_config)


# -----------------------------------------------------------------------------
# RBACCacheConfig Tests
# -----------------------------------------------------------------------------


class TestRBACCacheConfig:
    """Tests for RBACCacheConfig."""

    def test_default_config(self):
        """Default config has sensible defaults."""
        config = RBACCacheConfig()
        assert config.redis_url is None
        assert config.redis_prefix == "aragora:rbac"
        assert config.decision_ttl_seconds == 300
        assert config.role_ttl_seconds == 600
        assert config.permission_ttl_seconds == 900
        assert config.l1_enabled is True
        assert config.l1_max_size == 10000
        assert config.l1_ttl_seconds == 60
        assert config.enable_pubsub is True
        assert config.enable_metrics is True

    def test_from_env(self):
        """from_env reads environment variables."""
        with patch.dict(
            "os.environ",
            {
                "REDIS_URL": "redis://test:6379",
                "RBAC_CACHE_PREFIX": "test:rbac",
                "RBAC_CACHE_DECISION_TTL": "120",
                "RBAC_CACHE_ROLE_TTL": "240",
                "RBAC_CACHE_PERMISSION_TTL": "480",
                "RBAC_CACHE_L1_ENABLED": "false",
                "RBAC_CACHE_L1_MAX_SIZE": "5000",
                "RBAC_CACHE_L1_TTL": "30",
                "RBAC_CACHE_PUBSUB": "false",
                "RBAC_CACHE_METRICS": "false",
            },
        ):
            config = RBACCacheConfig.from_env()
            assert config.redis_url == "redis://test:6379"
            assert config.redis_prefix == "test:rbac"
            assert config.decision_ttl_seconds == 120
            assert config.role_ttl_seconds == 240
            assert config.permission_ttl_seconds == 480
            assert config.l1_enabled is False
            assert config.l1_max_size == 5000
            assert config.l1_ttl_seconds == 30
            assert config.enable_pubsub is False
            assert config.enable_metrics is False

    def test_from_env_with_aragora_redis_url(self):
        """from_env reads ARAGORA_REDIS_URL as fallback."""
        with patch.dict("os.environ", {"ARAGORA_REDIS_URL": "redis://aragora:6379"}, clear=False):
            with patch.dict("os.environ", {"REDIS_URL": ""}, clear=False):
                # Need to remove REDIS_URL to test fallback
                import os
                orig = os.environ.pop("REDIS_URL", None)
                try:
                    config = RBACCacheConfig.from_env()
                    assert config.redis_url == "redis://aragora:6379"
                finally:
                    if orig:
                        os.environ["REDIS_URL"] = orig


# -----------------------------------------------------------------------------
# CacheStats Tests
# -----------------------------------------------------------------------------


class TestCacheStats:
    """Tests for CacheStats."""

    def test_initial_values(self):
        """CacheStats starts with zero values."""
        stats = CacheStats()
        assert stats.l1_hits == 0
        assert stats.l1_misses == 0
        assert stats.l2_hits == 0
        assert stats.l2_misses == 0
        assert stats.invalidations == 0
        assert stats.pubsub_messages == 0
        assert stats.errors == 0
        assert stats.evictions == 0

    def test_total_hits(self):
        """total_hits sums L1 and L2 hits."""
        stats = CacheStats(l1_hits=10, l2_hits=5)
        assert stats.total_hits == 15

    def test_total_misses(self):
        """total_misses returns L2 misses (final miss count)."""
        stats = CacheStats(l1_misses=10, l2_misses=5)
        assert stats.total_misses == 5

    def test_hit_rate_with_hits(self):
        """hit_rate calculates correct percentage."""
        stats = CacheStats(l1_hits=80, l2_hits=10, l2_misses=10)
        assert stats.hit_rate == 0.9  # 90 hits / 100 total

    def test_hit_rate_no_requests(self):
        """hit_rate returns 0 when no requests."""
        stats = CacheStats()
        assert stats.hit_rate == 0.0

    def test_to_dict(self):
        """to_dict returns all statistics."""
        stats = CacheStats(
            l1_hits=10,
            l1_misses=5,
            l2_hits=3,
            l2_misses=2,
            invalidations=1,
            pubsub_messages=7,
            errors=0,
            evictions=4,
        )
        result = stats.to_dict()
        assert result["l1_hits"] == 10
        assert result["l1_misses"] == 5
        assert result["l2_hits"] == 3
        assert result["l2_misses"] == 2
        assert result["total_hits"] == 13
        assert result["total_misses"] == 2
        assert result["invalidations"] == 1
        assert result["pubsub_messages"] == 7
        assert result["errors"] == 0
        assert result["evictions"] == 4
        assert "hit_rate" in result


# -----------------------------------------------------------------------------
# RBACDistributedCache L1 Operations Tests
# -----------------------------------------------------------------------------


class TestRBACDistributedCacheL1:
    """Tests for L1 (in-memory) cache operations."""

    def test_l1_set_and_get(self, cache):
        """L1 cache stores and retrieves values."""
        cache._l1_set("test_key", {"value": 123})
        result = cache._l1_get("test_key")
        assert result == {"value": 123}

    def test_l1_get_missing_key(self, cache):
        """L1 get returns None for missing keys."""
        result = cache._l1_get("nonexistent")
        assert result is None

    def test_l1_ttl_expiration(self, cache_config):
        """L1 cache expires entries after TTL."""
        cache_config.l1_ttl_seconds = 0.1  # Very short TTL
        cache = RBACDistributedCache(cache_config)

        cache._l1_set("expire_key", "value")
        assert cache._l1_get("expire_key") == "value"

        time.sleep(0.15)  # Wait for expiration
        assert cache._l1_get("expire_key") is None

    def test_l1_eviction_on_max_size(self, cache_config):
        """L1 cache evicts oldest entries when full."""
        cache_config.l1_max_size = 3
        cache = RBACDistributedCache(cache_config)

        cache._l1_set("key1", "value1")
        cache._l1_set("key2", "value2")
        cache._l1_set("key3", "value3")
        cache._l1_set("key4", "value4")  # Should evict key1

        assert cache._l1_get("key1") is None
        assert cache._l1_get("key2") == "value2"
        assert cache._l1_get("key3") == "value3"
        assert cache._l1_get("key4") == "value4"
        assert cache._stats.evictions >= 1

    def test_l1_delete(self, cache):
        """L1 delete removes entries."""
        cache._l1_set("delete_key", "value")
        assert cache._l1_get("delete_key") == "value"

        result = cache._l1_delete("delete_key")
        assert result is True
        assert cache._l1_get("delete_key") is None

    def test_l1_delete_missing_key(self, cache):
        """L1 delete returns False for missing keys."""
        result = cache._l1_delete("nonexistent")
        assert result is False

    def test_l1_key_hashing(self, cache):
        """L1 hashes long keys."""
        long_key = "x" * 200  # Very long key
        short_result = cache._l1_key(long_key)
        assert len(short_result) == 32  # SHA256 hash prefix

        normal_key = "normal_key"
        normal_result = cache._l1_key(normal_key)
        assert normal_result == "normal_key"  # Unchanged

    def test_l1_pattern_invalidation(self, cache):
        """Pattern invalidation clears matching L1 entries."""
        cache._l1_set("user123:org1:hash:read", "value1")
        cache._l1_set("user123:org1:hash:write", "value2")
        cache._l1_set("user456:org1:hash:read", "value3")

        count = cache._invalidate_l1_pattern("user123:*")
        assert count == 2
        assert cache._l1_get("user123:org1:hash:read") is None
        assert cache._l1_get("user456:org1:hash:read") == "value3"


# -----------------------------------------------------------------------------
# Permission Decision Cache Tests
# -----------------------------------------------------------------------------


class TestPermissionDecisionCache:
    """Tests for permission decision caching."""

    def test_set_and_get_decision(self, cache):
        """Decision can be cached and retrieved."""
        decision = {"allowed": True, "reason": "Permission granted"}
        cache.set_decision(
            user_id="user1",
            org_id="org1",
            roles_hash="hash123",
            permission_key="debates.read",
            resource_id="debate-1",
            decision=decision,
        )

        result = cache.get_decision(
            user_id="user1",
            org_id="org1",
            roles_hash="hash123",
            permission_key="debates.read",
            resource_id="debate-1",
        )
        assert result == decision
        assert cache._stats.l1_hits == 1

    def test_get_decision_miss(self, cache):
        """Get decision returns None when not cached."""
        result = cache.get_decision(
            user_id="user1",
            org_id="org1",
            roles_hash="hash123",
            permission_key="debates.read",
            resource_id=None,
        )
        assert result is None
        assert cache._stats.l1_misses == 1

    def test_decision_cache_with_none_resource_id(self, cache):
        """Decision cache handles None resource_id."""
        decision = {"allowed": True, "reason": "Granted"}
        cache.set_decision(
            user_id="user1",
            org_id="org1",
            roles_hash="hash123",
            permission_key="debates.read",
            resource_id=None,
            decision=decision,
        )

        result = cache.get_decision(
            user_id="user1",
            org_id="org1",
            roles_hash="hash123",
            permission_key="debates.read",
            resource_id=None,
        )
        assert result == decision

    def test_decision_cache_with_none_org_id(self, cache):
        """Decision cache handles None org_id."""
        decision = {"allowed": False, "reason": "Denied"}
        cache.set_decision(
            user_id="user1",
            org_id=None,
            roles_hash="hash123",
            permission_key="debates.read",
            resource_id="d1",
            decision=decision,
        )

        result = cache.get_decision(
            user_id="user1",
            org_id=None,
            roles_hash="hash123",
            permission_key="debates.read",
            resource_id="d1",
        )
        assert result == decision


# -----------------------------------------------------------------------------
# Role Assignment Cache Tests
# -----------------------------------------------------------------------------


class TestRoleAssignmentCache:
    """Tests for user role caching."""

    def test_set_and_get_user_roles(self, cache):
        """User roles can be cached and retrieved."""
        roles = {"admin", "analyst", "member"}
        cache.set_user_roles(user_id="user1", org_id="org1", roles=roles)

        result = cache.get_user_roles(user_id="user1", org_id="org1")
        assert result == roles
        assert cache._stats.l1_hits == 1

    def test_get_user_roles_miss(self, cache):
        """Get user roles returns None when not cached."""
        result = cache.get_user_roles(user_id="user1", org_id="org1")
        assert result is None

    def test_user_roles_with_none_org(self, cache):
        """User roles cache handles None org_id."""
        roles = {"viewer"}
        cache.set_user_roles(user_id="user1", org_id=None, roles=roles)

        result = cache.get_user_roles(user_id="user1", org_id=None)
        assert result == roles


# -----------------------------------------------------------------------------
# Permission Set Cache Tests
# -----------------------------------------------------------------------------


class TestPermissionSetCache:
    """Tests for role permission caching."""

    def test_set_and_get_role_permissions(self, cache):
        """Role permissions can be cached and retrieved."""
        permissions = {"debates.read", "debates.create", "analytics.view"}
        cache.set_role_permissions(role_name="analyst", permissions=permissions)

        result = cache.get_role_permissions(role_name="analyst")
        assert result == permissions
        assert cache._stats.l1_hits == 1

    def test_get_role_permissions_miss(self, cache):
        """Get role permissions returns None when not cached."""
        result = cache.get_role_permissions(role_name="unknown_role")
        assert result is None


# -----------------------------------------------------------------------------
# Cache Invalidation Tests
# -----------------------------------------------------------------------------


class TestCacheInvalidation:
    """Tests for cache invalidation."""

    def test_invalidate_user(self, cache):
        """Invalidate user clears all user's cache entries."""
        # Set up various cache entries for user
        cache.set_decision(
            "user1", "org1", "hash1", "debates.read", "d1", {"allowed": True}
        )
        cache.set_decision(
            "user1", "org1", "hash1", "debates.write", "d2", {"allowed": False}
        )
        cache.set_user_roles("user1", "org1", {"admin"})

        # Also set entries for another user
        cache.set_decision(
            "user2", "org1", "hash2", "debates.read", "d1", {"allowed": True}
        )

        # Invalidate user1
        cache.invalidate_user("user1")

        # User1's entries should be gone
        assert cache.get_decision("user1", "org1", "hash1", "debates.read", "d1") is None
        assert cache.get_decision("user1", "org1", "hash1", "debates.write", "d2") is None

        # User2's entries should remain
        assert cache.get_decision("user2", "org1", "hash2", "debates.read", "d1") is not None

        assert cache._stats.invalidations == 1

    def test_invalidate_role(self, cache):
        """Invalidate role clears role's permission cache."""
        cache.set_role_permissions("admin", {"*"})
        cache.set_role_permissions("analyst", {"debates.read"})

        cache.invalidate_role("admin")

        assert cache.get_role_permissions("admin") is None
        assert cache.get_role_permissions("analyst") is not None
        assert cache._stats.invalidations == 1

    def test_invalidate_all(self, cache):
        """Invalidate all clears entire cache."""
        # Populate cache
        cache.set_decision("user1", "org1", "hash1", "p1", "r1", {"allowed": True})
        cache.set_user_roles("user1", "org1", {"admin"})
        cache.set_role_permissions("admin", {"*"})

        count = cache.invalidate_all()
        assert count >= 3  # At least 3 entries cleared

        # All should be gone
        assert cache.get_decision("user1", "org1", "hash1", "p1", "r1") is None
        assert cache.get_role_permissions("admin") is None

        assert cache._stats.invalidations == 1

    def test_invalidation_callback(self, cache):
        """Invalidation callbacks are called."""
        callback_calls = []

        def my_callback(key: str):
            callback_calls.append(key)

        cache.add_invalidation_callback(my_callback)
        cache.invalidate_user("user1")

        assert len(callback_calls) == 1
        assert callback_calls[0] == "user:user1"


# -----------------------------------------------------------------------------
# Cache Statistics Tests
# -----------------------------------------------------------------------------


class TestCacheStatistics:
    """Tests for cache statistics."""

    def test_get_stats(self, cache):
        """get_stats returns comprehensive statistics."""
        # Generate some activity
        cache.set_decision("user1", "org1", "h1", "p1", None, {"allowed": True})
        cache.get_decision("user1", "org1", "h1", "p1", None)  # Hit
        cache.get_decision("user1", "org1", "h1", "p2", None)  # Miss

        stats = cache.get_stats()
        assert "l1_hits" in stats
        assert "l1_misses" in stats
        assert "l1_size" in stats
        assert "l1_max_size" in stats
        assert "distributed" in stats
        assert stats["distributed"] is False  # No Redis

    def test_stats_property(self, cache):
        """stats property returns CacheStats object."""
        assert isinstance(cache.stats, CacheStats)


# -----------------------------------------------------------------------------
# Cache Lifecycle Tests
# -----------------------------------------------------------------------------


class TestCacheLifecycle:
    """Tests for cache start/stop lifecycle."""

    def test_start_without_redis(self, cache):
        """start() is safe without Redis."""
        cache.start()
        assert cache._running is True
        cache.stop()

    def test_stop_cleans_up(self, cache):
        """stop() cleans up resources."""
        cache.start()
        cache.stop()
        assert cache._running is False

    def test_is_distributed_without_redis(self, cache):
        """is_distributed returns False without Redis."""
        assert cache.is_distributed is False


# -----------------------------------------------------------------------------
# Global Cache Management Tests
# -----------------------------------------------------------------------------


class TestGlobalCacheManagement:
    """Tests for global cache singleton management."""

    def test_get_rbac_cache_creates_singleton(self):
        """get_rbac_cache creates a singleton instance."""
        cache1 = get_rbac_cache()
        cache2 = get_rbac_cache()
        assert cache1 is cache2

    def test_set_rbac_cache_replaces_singleton(self):
        """set_rbac_cache replaces the global instance."""
        original = get_rbac_cache()
        new_cache = RBACDistributedCache()
        set_rbac_cache(new_cache)

        assert get_rbac_cache() is new_cache
        assert get_rbac_cache() is not original

    def test_reset_rbac_cache_clears_singleton(self):
        """reset_rbac_cache clears the global instance."""
        cache = get_rbac_cache()
        cache.set_decision("user1", "org1", "h1", "p1", None, {"allowed": True})

        reset_rbac_cache()
        new_cache = get_rbac_cache()

        # New cache should be empty
        assert new_cache.get_decision("user1", "org1", "h1", "p1", None) is None

    def test_get_rbac_cache_with_config(self):
        """get_rbac_cache accepts custom config."""
        config = RBACCacheConfig(l1_max_size=50)
        cache = get_rbac_cache(config)
        assert cache.config.l1_max_size == 50


# -----------------------------------------------------------------------------
# Thread Safety Tests
# -----------------------------------------------------------------------------


class TestThreadSafety:
    """Tests for thread safety of cache operations."""

    def test_concurrent_l1_access(self, cache):
        """L1 cache is thread-safe for concurrent access."""
        errors = []
        iterations = 100

        def writer():
            for i in range(iterations):
                try:
                    cache._l1_set(f"key_{i}", {"value": i})
                except Exception as e:
                    errors.append(e)

        def reader():
            for i in range(iterations):
                try:
                    cache._l1_get(f"key_{i}")
                except Exception as e:
                    errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=writer),
            threading.Thread(target=reader),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"

    def test_concurrent_invalidation(self, cache):
        """Invalidation is thread-safe."""
        errors = []
        iterations = 50

        def populate():
            for i in range(iterations):
                try:
                    cache.set_decision(f"user{i}", "org1", "h1", "p1", None, {"allowed": True})
                except Exception as e:
                    errors.append(e)

        def invalidate():
            for i in range(iterations):
                try:
                    cache.invalidate_user(f"user{i}")
                except Exception as e:
                    errors.append(e)

        threads = [
            threading.Thread(target=populate),
            threading.Thread(target=invalidate),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# -----------------------------------------------------------------------------
# L1 Cache Disabled Tests
# -----------------------------------------------------------------------------


class TestL1Disabled:
    """Tests for cache with L1 disabled."""

    def test_operations_without_l1(self):
        """Cache operations work with L1 disabled."""
        config = RBACCacheConfig(l1_enabled=False, enable_metrics=False)
        cache = RBACDistributedCache(config)

        # Without L1 and Redis, get should return None
        result = cache.get_decision("user1", "org1", "h1", "p1", None)
        assert result is None

        # Set should not error
        cache.set_decision("user1", "org1", "h1", "p1", None, {"allowed": True})

        # Still None without Redis
        result = cache.get_decision("user1", "org1", "h1", "p1", None)
        assert result is None
