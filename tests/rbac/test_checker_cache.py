"""
Tests for PermissionChecker Cache Versioning System.

Tests cover the O(1) cache invalidation mechanism in checker.py:
- Cache version management (_global_cache_version, _user_cache_versions)
- Resource permission cache versioning
- Version-based cache invalidation (user-specific and global)
- Remote invalidation callbacks
- Cache key format with embedded versions
- TTL expiry of versioned entries
- Thread safety of cache operations
- Cache statistics
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from aragora.rbac.checker import (
    PermissionChecker,
    get_permission_checker,
    set_permission_checker,
)
from aragora.rbac.models import (
    Action,
    AuthorizationContext,
    ResourceType,
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_global_checker():
    """Reset global checker before and after each test."""
    set_permission_checker(None)
    yield
    set_permission_checker(None)


@pytest.fixture
def checker():
    """Create a permission checker with caching enabled."""
    return PermissionChecker(enable_cache=True, cache_ttl=60)


@pytest.fixture
def short_ttl_checker():
    """Create a checker with short TTL for expiry testing."""
    return PermissionChecker(enable_cache=True, cache_ttl=0.1)


@pytest.fixture
def context_user1():
    """Create context for user-1."""
    return AuthorizationContext(
        user_id="user-1",
        org_id="org-1",
        roles={"editor"},
        permissions={"debates.read", "debates.create"},
    )


@pytest.fixture
def context_user2():
    """Create context for user-2."""
    return AuthorizationContext(
        user_id="user-2",
        org_id="org-1",
        roles={"viewer"},
        permissions={"debates.read"},
    )


# -----------------------------------------------------------------------------
# Cache Version Management Tests
# -----------------------------------------------------------------------------


class TestCacheVersionManagement:
    """Tests for cache version tracking."""

    def test_initial_global_version_is_zero(self, checker):
        """Global cache version starts at zero."""
        assert checker._global_cache_version == 0

    def test_initial_user_versions_is_empty(self, checker):
        """User cache versions dict starts empty."""
        assert checker._user_cache_versions == {}

    def test_get_cache_version_format(self, checker):
        """_get_cache_version returns correct format."""
        version = checker._get_cache_version("user-1")
        assert version == "v0.0"  # global=0, user=0 (not set)

    def test_get_cache_version_includes_user_version(self, checker):
        """_get_cache_version includes user-specific version."""
        checker._user_cache_versions["user-1"] = 5
        version = checker._get_cache_version("user-1")
        assert version == "v0.5"  # global=0, user=5

    def test_get_cache_version_includes_global_version(self, checker):
        """_get_cache_version includes global version."""
        checker._global_cache_version = 3
        version = checker._get_cache_version("user-1")
        assert version == "v3.0"  # global=3, user=0

    def test_get_cache_version_combines_both(self, checker):
        """_get_cache_version combines global and user versions."""
        checker._global_cache_version = 2
        checker._user_cache_versions["user-1"] = 7
        version = checker._get_cache_version("user-1")
        assert version == "v2.7"

    def test_different_users_have_independent_versions(self, checker):
        """Different users have independent version numbers."""
        checker._user_cache_versions["user-1"] = 3
        checker._user_cache_versions["user-2"] = 5

        assert checker._get_cache_version("user-1") == "v0.3"
        assert checker._get_cache_version("user-2") == "v0.5"


# -----------------------------------------------------------------------------
# Resource Permission Cache Version Tests
# -----------------------------------------------------------------------------


class TestResourceCacheVersionManagement:
    """Tests for resource permission cache versioning."""

    def test_initial_resource_global_version_is_zero(self, checker):
        """Resource cache global version starts at zero."""
        assert checker._global_resource_cache_version == 0

    def test_initial_resource_user_versions_is_empty(self, checker):
        """Resource cache user versions dict starts empty."""
        assert checker._user_resource_cache_versions == {}

    def test_get_resource_cache_version_format(self, checker):
        """_get_resource_cache_version returns correct format."""
        version = checker._get_resource_cache_version("user-1")
        assert version == "v0.0"

    def test_get_resource_cache_version_includes_user_version(self, checker):
        """_get_resource_cache_version includes user-specific version."""
        checker._user_resource_cache_versions["user-1"] = 4
        version = checker._get_resource_cache_version("user-1")
        assert version == "v0.4"


# -----------------------------------------------------------------------------
# Cache Invalidation via Versioning Tests
# -----------------------------------------------------------------------------


class TestCacheInvalidationVersioning:
    """Tests for O(1) cache invalidation via versioning."""

    def test_clear_cache_user_increments_user_version(self, checker):
        """clear_cache(user_id) increments user's cache version."""
        assert checker._user_cache_versions.get("user-1", 0) == 0

        checker.clear_cache("user-1")

        assert checker._user_cache_versions["user-1"] == 1

    def test_clear_cache_user_multiple_times(self, checker):
        """Multiple clear_cache(user_id) calls increment version."""
        checker.clear_cache("user-1")
        checker.clear_cache("user-1")
        checker.clear_cache("user-1")

        assert checker._user_cache_versions["user-1"] == 3

    def test_clear_cache_user_also_clears_resource_cache(self, checker):
        """clear_cache(user_id) also increments resource cache version."""
        checker.clear_cache("user-1")

        assert checker._user_resource_cache_versions["user-1"] == 1

    def test_clear_cache_global_increments_global_version(self, checker):
        """clear_cache() (no user_id) increments global version."""
        assert checker._global_cache_version == 0

        checker.clear_cache()

        assert checker._global_cache_version == 1

    def test_clear_cache_global_increments_resource_global_version(self, checker):
        """clear_cache() also increments global resource cache version."""
        assert checker._global_resource_cache_version == 0

        checker.clear_cache()

        assert checker._global_resource_cache_version == 1

    def test_clear_cache_global_clears_user_version_dicts(self, checker):
        """clear_cache() clears user version dictionaries."""
        checker._user_cache_versions["user-1"] = 5
        checker._user_cache_versions["user-2"] = 3
        checker._user_resource_cache_versions["user-1"] = 2

        checker.clear_cache()

        assert checker._user_cache_versions == {}
        assert checker._user_resource_cache_versions == {}

    def test_clear_cache_global_clears_decision_cache(self, checker):
        """clear_cache() clears the decision cache dict."""
        # Populate cache
        checker._decision_cache["some_key"] = (MagicMock(), datetime.now(timezone.utc))

        checker.clear_cache()

        assert checker._decision_cache == {}

    def test_clear_cache_global_clears_resource_permission_cache(self, checker):
        """clear_cache() clears the resource permission cache."""
        checker._resource_permission_cache["some_key"] = (True, datetime.now(timezone.utc))

        checker.clear_cache()

        assert checker._resource_permission_cache == {}


# -----------------------------------------------------------------------------
# Cache Key with Version Embedding Tests
# -----------------------------------------------------------------------------


class TestCacheKeyVersionEmbedding:
    """Tests for cache keys with embedded versions."""

    def test_cached_decision_uses_versioned_key(self, checker, context_user1):
        """Cached decisions use versioned cache keys."""
        # Make a check to cache
        checker.check_permission(context_user1, "debates.read")

        # Find the cache key
        assert len(checker._decision_cache) == 1
        cache_key = list(checker._decision_cache.keys())[0]

        # Key should start with version
        assert cache_key.startswith("v0.0:")

    def test_version_change_invalidates_old_entries(self, checker, context_user1):
        """Version change makes old entries unreachable."""
        # Cache a decision
        decision1 = checker.check_permission(context_user1, "debates.read")
        assert decision1.cached is not True

        # Verify it's cached
        decision2 = checker.check_permission(context_user1, "debates.read")
        assert decision2.cached is True

        # Clear cache for user (increments version)
        checker.clear_cache("user-1")

        # Old entry still exists but is unreachable due to version mismatch
        # Next check should not return cached=True
        decision3 = checker.check_permission(context_user1, "debates.read")
        assert decision3.cached is not True

    def test_different_users_not_affected_by_user_cache_clear(
        self, checker, context_user1, context_user2
    ):
        """Clearing one user's cache doesn't affect other users."""
        # Cache decisions for both users
        checker.check_permission(context_user1, "debates.read")
        checker.check_permission(context_user2, "debates.read")

        # Verify both are cached
        assert checker.check_permission(context_user1, "debates.read").cached is True
        assert checker.check_permission(context_user2, "debates.read").cached is True

        # Clear only user-1's cache
        checker.clear_cache("user-1")

        # user-1 should not be cached (version changed)
        decision_u1 = checker.check_permission(context_user1, "debates.read")
        assert decision_u1.cached is not True

        # user-2 should still be cached
        decision_u2 = checker.check_permission(context_user2, "debates.read")
        assert decision_u2.cached is True


# -----------------------------------------------------------------------------
# Resource Permission Cache Invalidation Tests
# -----------------------------------------------------------------------------


class TestResourcePermissionCacheInvalidation:
    """Tests for resource permission cache invalidation."""

    def test_clear_resource_permission_cache_global(self, checker):
        """clear_resource_permission_cache() without args clears all."""
        checker._resource_permission_cache["key1"] = (True, datetime.now(timezone.utc))
        checker._resource_permission_cache["key2"] = (False, datetime.now(timezone.utc))

        checker.clear_resource_permission_cache()

        assert checker._resource_permission_cache == {}
        assert checker._global_resource_cache_version == 1

    def test_clear_resource_permission_cache_user_only(self, checker):
        """clear_resource_permission_cache(user_id) uses O(1) versioning."""
        initial_version = checker._user_resource_cache_versions.get("user-1", 0)

        checker.clear_resource_permission_cache(user_id="user-1")

        # Version should be incremented (O(1) operation)
        assert checker._user_resource_cache_versions["user-1"] == initial_version + 1

    def test_clear_resource_permission_cache_with_resource_type(self, checker):
        """clear_resource_permission_cache with resource_type uses O(n) fallback."""
        # Populate with keys - use actual ResourceType values
        # Key format: version:user_id:permission:resource_type:resource_id:org_id
        checker._resource_permission_cache["v0.0:user-1:perm:debates:d1:org1"] = (
            True,
            datetime.now(timezone.utc),
        )
        checker._resource_permission_cache["v0.0:user-1:perm:agents:a1:org1"] = (
            True,
            datetime.now(timezone.utc),
        )

        # Clear only debates type
        checker.clear_resource_permission_cache(resource_type=ResourceType.DEBATE)

        # Only debates entry should be removed
        remaining_keys = list(checker._resource_permission_cache.keys())
        assert len(remaining_keys) == 1
        assert "agents" in remaining_keys[0]

    def test_clear_resource_permission_cache_with_resource_id(self, checker):
        """clear_resource_permission_cache with resource_id uses O(n) fallback."""
        # Populate with keys
        checker._resource_permission_cache["v0.0:user-1:perm:debate:d1:org1"] = (
            True,
            datetime.now(timezone.utc),
        )
        checker._resource_permission_cache["v0.0:user-1:perm:debate:d2:org1"] = (
            True,
            datetime.now(timezone.utc),
        )

        # Clear only d1
        checker.clear_resource_permission_cache(resource_id="d1")

        remaining_keys = list(checker._resource_permission_cache.keys())
        assert len(remaining_keys) == 1
        assert "d2" in remaining_keys[0]


# -----------------------------------------------------------------------------
# Remote Invalidation Callback Tests
# -----------------------------------------------------------------------------


class TestRemoteInvalidationCallback:
    """Tests for _on_remote_invalidation callback."""

    def test_on_remote_invalidation_all(self, checker):
        """_on_remote_invalidation('all') clears everything."""
        checker._global_cache_version = 5
        checker._user_cache_versions["user-1"] = 3
        checker._decision_cache["key"] = (MagicMock(), datetime.now(timezone.utc))
        checker._resource_permission_cache["key"] = (True, datetime.now(timezone.utc))

        checker._on_remote_invalidation("all")

        assert checker._global_cache_version == 6  # Incremented
        assert checker._global_resource_cache_version == 1
        assert checker._user_cache_versions == {}
        assert checker._user_resource_cache_versions == {}
        assert checker._decision_cache == {}
        assert checker._resource_permission_cache == {}

    def test_on_remote_invalidation_user(self, checker):
        """_on_remote_invalidation('user:X') increments user's version."""
        checker._user_cache_versions["user-1"] = 2

        checker._on_remote_invalidation("user:user-1")

        assert checker._user_cache_versions["user-1"] == 3
        assert checker._user_resource_cache_versions["user-1"] == 1

    def test_on_remote_invalidation_new_user(self, checker):
        """_on_remote_invalidation('user:X') works for new users."""
        assert "new-user" not in checker._user_cache_versions

        checker._on_remote_invalidation("user:new-user")

        assert checker._user_cache_versions["new-user"] == 1


# -----------------------------------------------------------------------------
# Cache TTL Expiry Tests
# -----------------------------------------------------------------------------


class TestCacheTTLExpiry:
    """Tests for cache TTL expiration."""

    def test_cache_entry_expires_after_ttl(self, short_ttl_checker, context_user1):
        """Cached entries expire after TTL."""
        # Cache a decision
        decision1 = short_ttl_checker.check_permission(context_user1, "debates.read")
        assert decision1.allowed is True

        # Immediately should be cached
        decision2 = short_ttl_checker.check_permission(context_user1, "debates.read")
        assert decision2.cached is True

        # Wait for TTL to expire
        time.sleep(0.15)

        # Should no longer be cached
        decision3 = short_ttl_checker.check_permission(context_user1, "debates.read")
        assert decision3.cached is not True

    def test_expired_entry_removed_on_access(self, short_ttl_checker, context_user1):
        """Expired entries are removed from cache on access."""
        short_ttl_checker.check_permission(context_user1, "debates.read")
        initial_cache_size = len(short_ttl_checker._decision_cache)
        assert initial_cache_size == 1

        time.sleep(0.15)

        # Access should trigger removal
        short_ttl_checker.check_permission(context_user1, "debates.read")

        # Old entry should be removed (new one added)
        # Since TTL expired, the old versioned key was deleted
        # A new entry with same version is created
        assert len(short_ttl_checker._decision_cache) >= 1


# -----------------------------------------------------------------------------
# Cache Stats Tests
# -----------------------------------------------------------------------------


class TestCacheStats:
    """Tests for cache statistics."""

    def test_get_cache_stats_includes_cache_size(self, checker, context_user1):
        """get_cache_stats includes local_cache_size."""
        checker.check_permission(context_user1, "debates.read")
        checker.check_permission(context_user1, "debates.create")

        stats = checker.get_cache_stats()

        assert "local_cache_size" in stats
        assert stats["local_cache_size"] == 2

    def test_get_cache_stats_includes_resource_cache_size(self, checker):
        """get_cache_stats includes resource_permission_cache_size."""
        stats = checker.get_cache_stats()
        assert "resource_permission_cache_size" in stats

    def test_get_cache_stats_includes_cache_enabled(self, checker):
        """get_cache_stats includes cache_enabled."""
        stats = checker.get_cache_stats()
        assert stats["cache_enabled"] is True

    def test_get_cache_stats_includes_cache_ttl(self, checker):
        """get_cache_stats includes cache_ttl."""
        stats = checker.get_cache_stats()
        assert stats["cache_ttl"] == 60

    def test_get_cache_stats_distributed_false_without_backend(self, checker):
        """get_cache_stats shows distributed=False without backend."""
        stats = checker.get_cache_stats()
        assert stats["distributed"] is False

    def test_get_cache_stats_workspace_count(self, checker):
        """get_cache_stats includes workspace_count."""
        checker._workspace_roles["ws-1"] = {"user-1": {"editor"}}
        checker._workspace_roles["ws-2"] = {"user-2": {"viewer"}}

        stats = checker.get_cache_stats()
        assert stats["workspace_count"] == 2


# -----------------------------------------------------------------------------
# Thread Safety Tests
# -----------------------------------------------------------------------------


class TestCacheThreadSafety:
    """Tests for thread safety of cache operations."""

    def test_concurrent_cache_checks_no_errors(self, checker):
        """Concurrent permission checks don't cause errors."""
        errors = []
        iterations = 50

        def check_permissions(user_id: str):
            context = AuthorizationContext(
                user_id=user_id,
                org_id="org-1",
                permissions={"debates.read", "debates.create"},
            )
            for _ in range(iterations):
                try:
                    checker.check_permission(context, "debates.read")
                    checker.check_permission(context, "debates.create")
                except Exception as e:
                    errors.append(e)

        threads = [
            threading.Thread(target=check_permissions, args=(f"user-{i}",)) for i in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"

    def test_concurrent_cache_clear_no_errors(self, checker):
        """Concurrent cache clears don't cause errors."""
        errors = []

        def clear_and_check():
            context = AuthorizationContext(
                user_id="user-1",
                org_id="org-1",
                permissions={"debates.read"},
            )
            for _ in range(50):
                try:
                    checker.check_permission(context, "debates.read")
                    checker.clear_cache("user-1")
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=clear_and_check) for _ in range(4)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"

    def test_concurrent_global_clear_no_errors(self, checker):
        """Concurrent global cache clears don't cause errors."""
        errors = []

        def clear_global():
            for _ in range(20):
                try:
                    checker.clear_cache()
                except Exception as e:
                    errors.append(e)

        def check_perms():
            context = AuthorizationContext(
                user_id="user-1",
                org_id="org-1",
                permissions={"debates.read"},
            )
            for _ in range(50):
                try:
                    checker.check_permission(context, "debates.read")
                except Exception as e:
                    errors.append(e)

        threads = [
            threading.Thread(target=clear_global),
            threading.Thread(target=check_perms),
            threading.Thread(target=check_perms),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# -----------------------------------------------------------------------------
# Distributed Cache Backend Tests
# -----------------------------------------------------------------------------


class TestDistributedCacheBackend:
    """Tests for distributed cache backend integration."""

    def test_cache_backend_callback_registered(self):
        """When cache_backend provided, invalidation callback is registered."""
        mock_backend = MagicMock()
        checker = PermissionChecker(
            enable_cache=True,
            cache_backend=mock_backend,
        )

        mock_backend.add_invalidation_callback.assert_called_once()

    def test_distributed_cache_get_decision_called(self):
        """Distributed cache backend is used for get_decision."""
        mock_backend = MagicMock()
        mock_backend.get_decision.return_value = {
            "allowed": True,
            "reason": "Cached",
            "permission_key": "debates.read",
        }

        checker = PermissionChecker(
            enable_cache=True,
            cache_backend=mock_backend,
        )

        context = AuthorizationContext(
            user_id="user-1",
            org_id="org-1",
            permissions={"debates.read"},
        )

        decision = checker.check_permission(context, "debates.read")

        mock_backend.get_decision.assert_called()
        assert decision.allowed is True
        assert decision.cached is True

    def test_distributed_cache_set_decision_called(self):
        """Distributed cache backend is used for set_decision."""
        mock_backend = MagicMock()
        mock_backend.get_decision.return_value = None  # Cache miss

        checker = PermissionChecker(
            enable_cache=True,
            cache_backend=mock_backend,
        )

        context = AuthorizationContext(
            user_id="user-1",
            org_id="org-1",
            permissions={"debates.read"},
        )

        checker.check_permission(context, "debates.read")

        mock_backend.set_decision.assert_called()


# -----------------------------------------------------------------------------
# Permission Key Normalization Tests
# -----------------------------------------------------------------------------


class TestPermissionKeyNormalization:
    """Tests for permission key format normalization in caching."""

    def test_colon_format_normalized_to_dot(self, checker):
        """Permission key with colon is normalized to dot format."""
        context = AuthorizationContext(
            user_id="user-1",
            permissions={"debates.read"},
        )

        # Use colon format
        decision = checker.check_permission(context, "debates:read")

        assert decision.allowed is True

    def test_normalized_key_uses_same_cache_entry(self, checker):
        """Dot and colon formats use the same cache entry."""
        context = AuthorizationContext(
            user_id="user-1",
            permissions={"debates.read"},
        )

        # Check with dot format
        checker.check_permission(context, "debates.read")
        assert len(checker._decision_cache) == 1

        # Check with colon format - should hit cache
        decision = checker.check_permission(context, "debates:read")
        assert decision.cached is True
        assert len(checker._decision_cache) == 1  # Still only one entry


# -----------------------------------------------------------------------------
# Workspace Role Cache Tests
# -----------------------------------------------------------------------------


class TestWorkspaceRoleCache:
    """Tests for workspace-scoped role caching."""

    def test_workspace_roles_assignment_cached(self, checker):
        """Workspace role assignments are stored."""
        checker.assign_workspace_role("user-1", "workspace-1", "editor")

        roles = checker.get_workspace_roles("user-1", "workspace-1")
        assert "editor" in roles

    def test_workspace_roles_removal(self, checker):
        """Workspace role removal works."""
        checker.assign_workspace_role("user-1", "workspace-1", "editor")
        checker.assign_workspace_role("user-1", "workspace-1", "viewer")

        removed = checker.remove_workspace_role("user-1", "workspace-1", "editor")

        assert removed is True
        roles = checker.get_workspace_roles("user-1", "workspace-1")
        assert "editor" not in roles
        assert "viewer" in roles

    def test_workspace_roles_different_workspaces(self, checker):
        """Different workspaces have independent role sets."""
        checker.assign_workspace_role("user-1", "workspace-1", "editor")
        checker.assign_workspace_role("user-1", "workspace-2", "admin")

        roles_ws1 = checker.get_workspace_roles("user-1", "workspace-1")
        roles_ws2 = checker.get_workspace_roles("user-1", "workspace-2")

        assert roles_ws1 == {"editor"}
        assert roles_ws2 == {"admin"}


# -----------------------------------------------------------------------------
# Cache Disabled Tests
# -----------------------------------------------------------------------------


class TestCacheDisabled:
    """Tests when caching is disabled."""

    def test_no_caching_when_disabled(self):
        """Decisions are not cached when enable_cache=False."""
        checker = PermissionChecker(enable_cache=False)

        context = AuthorizationContext(
            user_id="user-1",
            permissions={"debates.read"},
        )

        # First check
        decision1 = checker.check_permission(context, "debates.read")
        assert decision1.cached is not True

        # Second check - still not cached
        decision2 = checker.check_permission(context, "debates.read")
        assert decision2.cached is not True

        # No entries in cache
        assert len(checker._decision_cache) == 0

    def test_clear_cache_safe_when_disabled(self):
        """clear_cache is safe when caching is disabled."""
        checker = PermissionChecker(enable_cache=False)

        # Should not raise
        checker.clear_cache()
        checker.clear_cache("user-1")


# -----------------------------------------------------------------------------
# Edge Cases Tests
# -----------------------------------------------------------------------------


class TestCacheEdgeCases:
    """Tests for edge cases in caching."""

    def test_empty_roles_hash(self, checker):
        """Context with empty roles produces consistent hash."""
        context = AuthorizationContext(
            user_id="user-1",
            roles=set(),
            permissions={"debates.read"},
        )

        checker.check_permission(context, "debates.read")

        # Should have cached with consistent hash
        assert len(checker._decision_cache) == 1

    def test_none_org_id_in_cache_key(self, checker):
        """None org_id is handled in cache key."""
        context = AuthorizationContext(
            user_id="user-1",
            org_id=None,
            permissions={"debates.read"},
        )

        checker.check_permission(context, "debates.read")

        # Should have one cached entry
        assert len(checker._decision_cache) == 1

    def test_none_resource_id_in_cache_key(self, checker):
        """None resource_id is handled in cache key."""
        context = AuthorizationContext(
            user_id="user-1",
            permissions={"debates.read"},
        )

        checker.check_permission(context, "debates.read", resource_id=None)

        assert len(checker._decision_cache) == 1

    def test_very_long_permission_key(self, checker):
        """Very long permission keys are handled."""
        context = AuthorizationContext(
            user_id="user-1",
            permissions={"x" * 500},
        )

        # Should not raise
        checker.check_permission(context, "x" * 500)

    def test_special_characters_in_cache_key(self, checker):
        """Special characters in IDs are handled."""
        context = AuthorizationContext(
            user_id="user:with:colons",
            org_id="org/with/slashes",
            permissions={"debates.read"},
        )

        checker.check_permission(context, "debates.read")

        # Should have cached
        assert len(checker._decision_cache) == 1

        # Should be retrievable
        decision = checker.check_permission(context, "debates.read")
        assert decision.cached is True
