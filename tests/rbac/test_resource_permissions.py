"""
Tests for RBAC Resource-Level Permissions.

Comprehensive test suite for fine-grained resource-level access control,
covering:
- ResourcePermission dataclass functionality
- ResourcePermissionStore CRUD operations
- Permission granting and revoking
- Permission checking with resource IDs
- Cache behavior and invalidation
- Integration with role-based permissions
- Expiration and deactivation
- Edge cases and error handling

Target: 40+ tests for enterprise-grade coverage.
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch
from uuid import uuid4

from aragora.rbac.resource_permissions import (
    ResourcePermission,
    ResourcePermissionStore,
    get_resource_permission_store,
    set_resource_permission_store,
    grant_resource_permission,
    revoke_resource_permission,
    check_resource_permission,
)
from aragora.rbac.models import (
    Action,
    AuthorizationContext,
    ResourceType,
)
from aragora.rbac.checker import PermissionChecker


# =============================================================================
# ResourcePermission Dataclass Tests
# =============================================================================


class TestResourcePermission:
    """Tests for ResourcePermission dataclass."""

    def test_create_basic_permission(self):
        """Create a basic resource permission."""
        perm = ResourcePermission(
            id="perm-1",
            user_id="user-1",
            permission_id="debates.read",
            resource_type=ResourceType.DEBATE,
            resource_id="debate-123",
        )
        assert perm.id == "perm-1"
        assert perm.user_id == "user-1"
        assert perm.permission_id == "debates.read"
        assert perm.resource_type == ResourceType.DEBATE
        assert perm.resource_id == "debate-123"
        assert perm.is_active is True
        assert perm.is_valid is True

    def test_create_factory_method(self):
        """Test ResourcePermission.create factory method."""
        perm = ResourcePermission.create(
            user_id="user-1",
            permission_id="debates.update",
            resource_type=ResourceType.DEBATE,
            resource_id="debate-456",
            granted_by="admin-1",
            org_id="org-1",
        )
        assert perm.id is not None
        assert perm.user_id == "user-1"
        assert perm.granted_by == "admin-1"
        assert perm.org_id == "org-1"
        assert perm.granted_at is not None

    def test_permission_with_expiration(self):
        """Test permission with expiration date."""
        future = datetime.now(timezone.utc) + timedelta(hours=1)
        perm = ResourcePermission.create(
            user_id="user-1",
            permission_id="debates.read",
            resource_type=ResourceType.DEBATE,
            resource_id="debate-123",
            expires_at=future,
        )
        assert perm.is_expired is False
        assert perm.is_valid is True

    def test_expired_permission(self):
        """Test that expired permission is invalid."""
        past = datetime.now(timezone.utc) - timedelta(hours=1)
        perm = ResourcePermission.create(
            user_id="user-1",
            permission_id="debates.read",
            resource_type=ResourceType.DEBATE,
            resource_id="debate-123",
            expires_at=past,
        )
        assert perm.is_expired is True
        assert perm.is_valid is False

    def test_inactive_permission(self):
        """Test that inactive permission is invalid."""
        perm = ResourcePermission(
            id="perm-1",
            user_id="user-1",
            permission_id="debates.read",
            resource_type=ResourceType.DEBATE,
            resource_id="debate-123",
            is_active=False,
        )
        assert perm.is_active is False
        assert perm.is_valid is False

    def test_permission_action_extraction(self):
        """Test action extraction from permission_id."""
        perm = ResourcePermission.create(
            user_id="user-1",
            permission_id="debates.update",
            resource_type=ResourceType.DEBATE,
            resource_id="debate-123",
        )
        assert perm.action == "update"

    def test_permission_matches_exact(self):
        """Test exact permission matching."""
        perm = ResourcePermission.create(
            user_id="user-1",
            permission_id="debates.read",
            resource_type=ResourceType.DEBATE,
            resource_id="debate-123",
        )
        assert perm.matches("user-1", "debates.read", ResourceType.DEBATE, "debate-123")
        assert not perm.matches("user-2", "debates.read", ResourceType.DEBATE, "debate-123")
        assert not perm.matches("user-1", "debates.update", ResourceType.DEBATE, "debate-123")
        assert not perm.matches("user-1", "debates.read", ResourceType.DEBATE, "debate-999")

    def test_permission_matches_wildcard(self):
        """Test wildcard permission matching."""
        perm = ResourcePermission.create(
            user_id="user-1",
            permission_id="debates.*",
            resource_type=ResourceType.DEBATE,
            resource_id="debate-123",
        )
        assert perm.matches("user-1", "debates.read", ResourceType.DEBATE, "debate-123")
        assert perm.matches("user-1", "debates.update", ResourceType.DEBATE, "debate-123")
        assert perm.matches("user-1", "debates.delete", ResourceType.DEBATE, "debate-123")

    def test_permission_matches_org_scope(self):
        """Test permission matching respects org scope."""
        perm = ResourcePermission.create(
            user_id="user-1",
            permission_id="debates.read",
            resource_type=ResourceType.DEBATE,
            resource_id="debate-123",
            org_id="org-1",
        )
        assert perm.matches("user-1", "debates.read", ResourceType.DEBATE, "debate-123", "org-1")
        assert not perm.matches(
            "user-1", "debates.read", ResourceType.DEBATE, "debate-123", "org-2"
        )

    def test_permission_to_dict(self):
        """Test serialization to dictionary."""
        perm = ResourcePermission.create(
            user_id="user-1",
            permission_id="debates.read",
            resource_type=ResourceType.DEBATE,
            resource_id="debate-123",
            granted_by="admin-1",
        )
        data = perm.to_dict()
        assert data["user_id"] == "user-1"
        assert data["permission_id"] == "debates.read"
        assert data["resource_type"] == "debates"
        assert data["resource_id"] == "debate-123"
        assert "granted_at" in data

    def test_permission_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "id": "perm-123",
            "user_id": "user-1",
            "permission_id": "debates.read",
            "resource_type": "debates",
            "resource_id": "debate-123",
            "granted_at": "2024-01-01T00:00:00+00:00",
            "is_active": True,
        }
        perm = ResourcePermission.from_dict(data)
        assert perm.id == "perm-123"
        assert perm.user_id == "user-1"
        assert perm.resource_type == ResourceType.DEBATE

    def test_permission_with_conditions(self):
        """Test permission with ABAC conditions."""
        perm = ResourcePermission.create(
            user_id="user-1",
            permission_id="debates.read",
            resource_type=ResourceType.DEBATE,
            resource_id="debate-123",
            conditions={"ip_range": "10.0.0.0/8", "time_window": "business_hours"},
        )
        assert perm.conditions["ip_range"] == "10.0.0.0/8"
        assert perm.conditions["time_window"] == "business_hours"

    def test_permission_with_metadata(self):
        """Test permission with metadata."""
        perm = ResourcePermission.create(
            user_id="user-1",
            permission_id="debates.read",
            resource_type=ResourceType.DEBATE,
            resource_id="debate-123",
            metadata={"reason": "Project collaboration", "ticket": "JIRA-123"},
        )
        assert perm.metadata["reason"] == "Project collaboration"
        assert perm.metadata["ticket"] == "JIRA-123"


# =============================================================================
# ResourcePermissionStore Tests
# =============================================================================


class TestResourcePermissionStore:
    """Tests for ResourcePermissionStore class."""

    @pytest.fixture
    def store(self):
        """Create a fresh permission store."""
        return ResourcePermissionStore(enable_cache=False)

    @pytest.fixture
    def cached_store(self):
        """Create a permission store with caching enabled."""
        return ResourcePermissionStore(enable_cache=True, cache_ttl=60)

    def test_grant_permission(self, store):
        """Grant a resource permission."""
        perm = store.grant_permission(
            user_id="user-1",
            permission_id="debates.read",
            resource_type=ResourceType.DEBATE,
            resource_id="debate-123",
            granted_by="admin-1",
        )
        assert perm.id is not None
        assert perm.user_id == "user-1"
        assert perm.permission_id == "debates.read"
        assert perm.granted_by == "admin-1"

    def test_grant_duplicate_permission_updates(self, store):
        """Granting duplicate permission updates existing."""
        perm1 = store.grant_permission(
            user_id="user-1",
            permission_id="debates.read",
            resource_type=ResourceType.DEBATE,
            resource_id="debate-123",
        )
        # Grant same permission again
        perm2 = store.grant_permission(
            user_id="user-1",
            permission_id="debates.read",
            resource_type=ResourceType.DEBATE,
            resource_id="debate-123",
            metadata={"updated": True},
        )
        # Should be same permission, updated
        assert perm1.id == perm2.id
        assert perm2.metadata.get("updated") is True

    def test_revoke_permission(self, store):
        """Revoke a resource permission."""
        perm = store.grant_permission(
            user_id="user-1",
            permission_id="debates.read",
            resource_type=ResourceType.DEBATE,
            resource_id="debate-123",
        )
        result = store.revoke_permission(perm.id)
        assert result is True
        # Permission should no longer be valid
        assert (
            store.check_resource_permission(
                "user-1", "debates.read", ResourceType.DEBATE, "debate-123"
            )
            is False
        )

    def test_revoke_nonexistent_permission(self, store):
        """Revoking nonexistent permission returns False."""
        result = store.revoke_permission("nonexistent-id")
        assert result is False

    def test_revoke_all_for_user(self, store):
        """Revoke all permissions for a user."""
        store.grant_permission(
            user_id="user-1",
            permission_id="debates.read",
            resource_type=ResourceType.DEBATE,
            resource_id="debate-1",
        )
        store.grant_permission(
            user_id="user-1",
            permission_id="debates.read",
            resource_type=ResourceType.DEBATE,
            resource_id="debate-2",
        )
        store.grant_permission(
            user_id="user-2",
            permission_id="debates.read",
            resource_type=ResourceType.DEBATE,
            resource_id="debate-1",
        )

        count = store.revoke_all_for_user("user-1")
        assert count == 2
        # user-2 permission should still exist
        assert (
            store.check_resource_permission(
                "user-2", "debates.read", ResourceType.DEBATE, "debate-1"
            )
            is True
        )

    def test_revoke_all_for_resource(self, store):
        """Revoke all permissions for a resource."""
        store.grant_permission(
            user_id="user-1",
            permission_id="debates.read",
            resource_type=ResourceType.DEBATE,
            resource_id="debate-123",
        )
        store.grant_permission(
            user_id="user-2",
            permission_id="debates.update",
            resource_type=ResourceType.DEBATE,
            resource_id="debate-123",
        )
        store.grant_permission(
            user_id="user-1",
            permission_id="debates.read",
            resource_type=ResourceType.DEBATE,
            resource_id="debate-456",
        )

        count = store.revoke_all_for_resource(ResourceType.DEBATE, "debate-123")
        assert count == 2
        # Other resource permission should still exist
        assert (
            store.check_resource_permission(
                "user-1", "debates.read", ResourceType.DEBATE, "debate-456"
            )
            is True
        )

    def test_check_resource_permission_granted(self, store):
        """Check permission returns True when granted."""
        store.grant_permission(
            user_id="user-1",
            permission_id="debates.read",
            resource_type=ResourceType.DEBATE,
            resource_id="debate-123",
        )
        result = store.check_resource_permission(
            "user-1", "debates.read", ResourceType.DEBATE, "debate-123"
        )
        assert result is True

    def test_check_resource_permission_not_granted(self, store):
        """Check permission returns False when not granted."""
        result = store.check_resource_permission(
            "user-1", "debates.read", ResourceType.DEBATE, "debate-123"
        )
        assert result is False

    def test_check_respects_expiration(self, store):
        """Check permission returns False for expired permissions."""
        past = datetime.now(timezone.utc) - timedelta(hours=1)
        store.grant_permission(
            user_id="user-1",
            permission_id="debates.read",
            resource_type=ResourceType.DEBATE,
            resource_id="debate-123",
            expires_at=past,
        )
        result = store.check_resource_permission(
            "user-1", "debates.read", ResourceType.DEBATE, "debate-123"
        )
        assert result is False

    def test_check_respects_org_scope(self, store):
        """Check permission respects organization scope."""
        store.grant_permission(
            user_id="user-1",
            permission_id="debates.read",
            resource_type=ResourceType.DEBATE,
            resource_id="debate-123",
            org_id="org-1",
        )
        # Same org - should pass
        assert (
            store.check_resource_permission(
                "user-1", "debates.read", ResourceType.DEBATE, "debate-123", "org-1"
            )
            is True
        )
        # Different org - should fail
        assert (
            store.check_resource_permission(
                "user-1", "debates.read", ResourceType.DEBATE, "debate-123", "org-2"
            )
            is False
        )

    def test_list_permissions_for_resource(self, store):
        """List all permissions for a resource."""
        store.grant_permission(
            user_id="user-1",
            permission_id="debates.read",
            resource_type=ResourceType.DEBATE,
            resource_id="debate-123",
        )
        store.grant_permission(
            user_id="user-2",
            permission_id="debates.update",
            resource_type=ResourceType.DEBATE,
            resource_id="debate-123",
        )
        store.grant_permission(
            user_id="user-1",
            permission_id="debates.read",
            resource_type=ResourceType.DEBATE,
            resource_id="debate-456",
        )

        perms = store.list_permissions_for_resource(ResourceType.DEBATE, "debate-123")
        assert len(perms) == 2
        user_ids = {p.user_id for p in perms}
        assert user_ids == {"user-1", "user-2"}

    def test_list_permissions_for_user(self, store):
        """List all permissions for a user."""
        store.grant_permission(
            user_id="user-1",
            permission_id="debates.read",
            resource_type=ResourceType.DEBATE,
            resource_id="debate-1",
        )
        store.grant_permission(
            user_id="user-1",
            permission_id="agents.read",
            resource_type=ResourceType.AGENT,
            resource_id="agent-1",
        )
        store.grant_permission(
            user_id="user-2",
            permission_id="debates.read",
            resource_type=ResourceType.DEBATE,
            resource_id="debate-1",
        )

        perms = store.list_permissions_for_user("user-1")
        assert len(perms) == 2

        # Filter by resource type
        debate_perms = store.list_permissions_for_user("user-1", ResourceType.DEBATE)
        assert len(debate_perms) == 1

    def test_get_permission(self, store):
        """Get a specific permission by ID."""
        perm = store.grant_permission(
            user_id="user-1",
            permission_id="debates.read",
            resource_type=ResourceType.DEBATE,
            resource_id="debate-123",
        )
        retrieved = store.get_permission(perm.id)
        assert retrieved is not None
        assert retrieved.id == perm.id

    def test_get_nonexistent_permission(self, store):
        """Get nonexistent permission returns None."""
        result = store.get_permission("nonexistent-id")
        assert result is None

    def test_count_permissions(self, store):
        """Count permissions with filters."""
        store.grant_permission(
            user_id="user-1",
            permission_id="debates.read",
            resource_type=ResourceType.DEBATE,
            resource_id="debate-1",
        )
        store.grant_permission(
            user_id="user-1",
            permission_id="debates.read",
            resource_type=ResourceType.DEBATE,
            resource_id="debate-2",
        )
        store.grant_permission(
            user_id="user-2",
            permission_id="debates.read",
            resource_type=ResourceType.DEBATE,
            resource_id="debate-1",
        )

        assert store.count_permissions() == 3
        assert store.count_permissions(user_id="user-1") == 2
        assert store.count_permissions(resource_type=ResourceType.DEBATE) == 3
        assert store.count_permissions(resource_id="debate-1") == 2

    def test_cleanup_expired(self, store):
        """Cleanup removes expired permissions."""
        past = datetime.now(timezone.utc) - timedelta(hours=1)
        future = datetime.now(timezone.utc) + timedelta(hours=1)

        store.grant_permission(
            user_id="user-1",
            permission_id="debates.read",
            resource_type=ResourceType.DEBATE,
            resource_id="debate-1",
            expires_at=past,
        )
        store.grant_permission(
            user_id="user-2",
            permission_id="debates.read",
            resource_type=ResourceType.DEBATE,
            resource_id="debate-2",
            expires_at=future,
        )

        count = store.cleanup_expired()
        assert count == 1
        assert store.count_permissions(active_only=True) == 1

    def test_get_stats(self, store):
        """Get store statistics."""
        store.grant_permission(
            user_id="user-1",
            permission_id="debates.read",
            resource_type=ResourceType.DEBATE,
            resource_id="debate-1",
        )

        stats = store.get_stats()
        assert stats["total_permissions"] >= 1
        assert stats["active_permissions"] >= 1
        assert "unique_users" in stats
        assert "unique_resources" in stats


# =============================================================================
# Cache Behavior Tests
# =============================================================================


class TestResourcePermissionCaching:
    """Tests for permission caching behavior."""

    @pytest.fixture
    def cached_store(self):
        """Create a permission store with caching enabled."""
        return ResourcePermissionStore(enable_cache=True, cache_ttl=60)

    def test_cache_hit_on_repeated_check(self, cached_store):
        """Repeated permission checks use cache."""
        cached_store.grant_permission(
            user_id="user-1",
            permission_id="debates.read",
            resource_type=ResourceType.DEBATE,
            resource_id="debate-123",
        )

        # First check - cache miss
        result1 = cached_store.check_resource_permission(
            "user-1", "debates.read", ResourceType.DEBATE, "debate-123"
        )
        # Second check - should use cache
        result2 = cached_store.check_resource_permission(
            "user-1", "debates.read", ResourceType.DEBATE, "debate-123"
        )

        assert result1 is True
        assert result2 is True
        # Cache should have entry
        assert cached_store.get_stats()["cache_size"] >= 1

    def test_cache_invalidation_on_grant(self, cached_store):
        """Cache is invalidated when permission is granted."""
        # Check initially (not granted)
        result1 = cached_store.check_resource_permission(
            "user-1", "debates.read", ResourceType.DEBATE, "debate-123"
        )
        assert result1 is False

        # Grant permission
        cached_store.grant_permission(
            user_id="user-1",
            permission_id="debates.read",
            resource_type=ResourceType.DEBATE,
            resource_id="debate-123",
        )

        # Check again - should reflect new permission
        result2 = cached_store.check_resource_permission(
            "user-1", "debates.read", ResourceType.DEBATE, "debate-123"
        )
        assert result2 is True

    def test_cache_invalidation_on_revoke(self, cached_store):
        """Cache is invalidated when permission is revoked."""
        perm = cached_store.grant_permission(
            user_id="user-1",
            permission_id="debates.read",
            resource_type=ResourceType.DEBATE,
            resource_id="debate-123",
        )

        # Check (should be True)
        result1 = cached_store.check_resource_permission(
            "user-1", "debates.read", ResourceType.DEBATE, "debate-123"
        )
        assert result1 is True

        # Revoke permission
        cached_store.revoke_permission(perm.id)

        # Check again - should reflect revocation
        result2 = cached_store.check_resource_permission(
            "user-1", "debates.read", ResourceType.DEBATE, "debate-123"
        )
        assert result2 is False

    def test_clear_cache(self, cached_store):
        """Clear cache removes all cached entries."""
        cached_store.grant_permission(
            user_id="user-1",
            permission_id="debates.read",
            resource_type=ResourceType.DEBATE,
            resource_id="debate-123",
        )
        cached_store.check_resource_permission(
            "user-1", "debates.read", ResourceType.DEBATE, "debate-123"
        )

        cached_store.clear_cache()
        assert cached_store.get_stats()["cache_size"] == 0

    def test_clear_cache_for_user(self, cached_store):
        """Clear cache for specific user."""
        cached_store.grant_permission(
            user_id="user-1",
            permission_id="debates.read",
            resource_type=ResourceType.DEBATE,
            resource_id="debate-1",
        )
        cached_store.grant_permission(
            user_id="user-2",
            permission_id="debates.read",
            resource_type=ResourceType.DEBATE,
            resource_id="debate-2",
        )

        # Populate cache
        cached_store.check_resource_permission(
            "user-1", "debates.read", ResourceType.DEBATE, "debate-1"
        )
        cached_store.check_resource_permission(
            "user-2", "debates.read", ResourceType.DEBATE, "debate-2"
        )

        # Clear only user-1
        cached_store.clear_cache("user-1")

        # user-2 cache should remain
        assert cached_store.get_stats()["cache_size"] >= 1


# =============================================================================
# Integration with PermissionChecker Tests
# =============================================================================


class TestCheckerIntegration:
    """Tests for integration between PermissionChecker and ResourcePermissionStore."""

    @pytest.fixture
    def store(self):
        """Create a resource permission store."""
        return ResourcePermissionStore(enable_cache=False)

    @pytest.fixture
    def checker_with_store(self, store):
        """Create a permission checker with resource permission store."""
        return PermissionChecker(
            enable_cache=False,
            resource_permission_store=store,
        )

    def test_check_resource_permission_with_store(self, checker_with_store, store):
        """Checker uses resource permission store."""
        store.grant_permission(
            user_id="user-1",
            permission_id="debates.read",
            resource_type=ResourceType.DEBATE,
            resource_id="debate-123",
        )

        decision = checker_with_store.check_resource_permission(
            user_id="user-1",
            action=Action.READ,
            resource_type=ResourceType.DEBATE,
            resource_id="debate-123",
        )

        assert decision.allowed is True
        assert "Resource-level permission" in decision.reason

    def test_fallback_to_role_permissions(self, checker_with_store):
        """Falls back to role-based permissions when no resource permission."""
        context = AuthorizationContext(
            user_id="user-1",
            org_id="org-1",
            roles={"admin"},
            permissions={"debates.read"},
        )

        decision = checker_with_store.check_resource_permission(
            user_id="user-1",
            action=Action.READ,
            resource_type=ResourceType.DEBATE,
            resource_id="debate-123",
            context=context,
        )

        assert decision.allowed is True

    def test_denied_without_context_or_resource_permission(self, checker_with_store):
        """Denied when no resource permission and no context."""
        decision = checker_with_store.check_resource_permission(
            user_id="user-1",
            action=Action.READ,
            resource_type=ResourceType.DEBATE,
            resource_id="debate-123",
        )

        assert decision.allowed is False
        assert "No resource-level permission" in decision.reason

    def test_set_resource_permission_store(self):
        """Set resource permission store on existing checker."""
        checker = PermissionChecker(enable_cache=False)
        store = ResourcePermissionStore(enable_cache=False)

        store.grant_permission(
            user_id="user-1",
            permission_id="debates.read",
            resource_type=ResourceType.DEBATE,
            resource_id="debate-123",
        )

        # Initially no store
        decision1 = checker.check_resource_permission(
            user_id="user-1",
            action=Action.READ,
            resource_type=ResourceType.DEBATE,
            resource_id="debate-123",
        )
        assert decision1.allowed is False

        # Set store
        checker.set_resource_permission_store(store)

        # Now should work
        decision2 = checker.check_resource_permission(
            user_id="user-1",
            action=Action.READ,
            resource_type=ResourceType.DEBATE,
            resource_id="debate-123",
        )
        assert decision2.allowed is True

    def test_checker_cache_stats_include_resource_store(self, checker_with_store, store):
        """Cache stats include resource permission store stats."""
        store.grant_permission(
            user_id="user-1",
            permission_id="debates.read",
            resource_type=ResourceType.DEBATE,
            resource_id="debate-123",
        )

        stats = checker_with_store.get_cache_stats()
        assert stats["resource_permission_store_enabled"] is True
        assert "resource_permission_store_stats" in stats

    def test_clear_resource_permission_cache(self, checker_with_store, store):
        """Clear resource permission cache."""
        checker = PermissionChecker(enable_cache=True)
        checker.set_resource_permission_store(store)

        store.grant_permission(
            user_id="user-1",
            permission_id="debates.read",
            resource_type=ResourceType.DEBATE,
            resource_id="debate-123",
        )

        # Make a check to populate cache
        checker.check_resource_permission(
            user_id="user-1",
            action=Action.READ,
            resource_type=ResourceType.DEBATE,
            resource_id="debate-123",
        )

        # Clear cache
        checker.clear_resource_permission_cache()
        assert checker.get_cache_stats()["resource_permission_cache_size"] == 0


# =============================================================================
# Global Functions Tests
# =============================================================================


class TestGlobalFunctions:
    """Tests for module-level convenience functions."""

    def teardown_method(self):
        """Reset global store after each test."""
        set_resource_permission_store(None)

    def test_get_resource_permission_store_creates_default(self):
        """get_resource_permission_store creates default instance."""
        store = get_resource_permission_store()
        assert isinstance(store, ResourcePermissionStore)

    def test_set_resource_permission_store(self):
        """set_resource_permission_store replaces global instance."""
        custom = ResourcePermissionStore(cache_ttl=999)
        set_resource_permission_store(custom)

        retrieved = get_resource_permission_store()
        assert retrieved._cache_ttl == 999

    def test_grant_resource_permission_global(self):
        """grant_resource_permission uses global store."""
        perm = grant_resource_permission(
            user_id="user-1",
            permission_id="debates.read",
            resource_type=ResourceType.DEBATE,
            resource_id="debate-123",
        )
        assert perm.id is not None
        assert perm.user_id == "user-1"

    def test_revoke_resource_permission_global(self):
        """revoke_resource_permission uses global store."""
        perm = grant_resource_permission(
            user_id="user-1",
            permission_id="debates.read",
            resource_type=ResourceType.DEBATE,
            resource_id="debate-123",
        )
        result = revoke_resource_permission(perm.id)
        assert result is True

    def test_check_resource_permission_global(self):
        """check_resource_permission uses global store."""
        grant_resource_permission(
            user_id="user-1",
            permission_id="debates.read",
            resource_type=ResourceType.DEBATE,
            resource_id="debate-123",
        )
        result = check_resource_permission(
            user_id="user-1",
            permission_id="debates.read",
            resource_type=ResourceType.DEBATE,
            resource_id="debate-123",
        )
        assert result is True


# =============================================================================
# Edge Cases and Error Handling Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def store(self):
        """Create a fresh permission store."""
        return ResourcePermissionStore(enable_cache=False)

    def test_empty_resource_id(self, store):
        """Handle empty resource ID."""
        perm = store.grant_permission(
            user_id="user-1",
            permission_id="debates.read",
            resource_type=ResourceType.DEBATE,
            resource_id="",
        )
        assert perm.resource_id == ""

    def test_special_characters_in_ids(self, store):
        """Handle special characters in IDs."""
        perm = store.grant_permission(
            user_id="user:with:colons",
            permission_id="debates.read",
            resource_type=ResourceType.DEBATE,
            resource_id="debate/with/slashes",
        )
        result = store.check_resource_permission(
            "user:with:colons",
            "debates.read",
            ResourceType.DEBATE,
            "debate/with/slashes",
        )
        assert result is True

    def test_unicode_in_metadata(self, store):
        """Handle unicode in metadata."""
        perm = store.grant_permission(
            user_id="user-1",
            permission_id="debates.read",
            resource_type=ResourceType.DEBATE,
            resource_id="debate-123",
            metadata={"description": "Debate about AI safety in Japanese: AI\u5b89\u5168\u6027"},
        )
        assert "\u5b89\u5168\u6027" in perm.metadata["description"]

    def test_very_long_resource_id(self, store):
        """Handle very long resource IDs."""
        long_id = "x" * 10000
        perm = store.grant_permission(
            user_id="user-1",
            permission_id="debates.read",
            resource_type=ResourceType.DEBATE,
            resource_id=long_id,
        )
        result = store.check_resource_permission(
            "user-1", "debates.read", ResourceType.DEBATE, long_id
        )
        assert result is True

    def test_concurrent_grant_same_permission(self, store):
        """Handle concurrent grants of same permission."""
        # Simulate concurrent grants
        perm1 = store.grant_permission(
            user_id="user-1",
            permission_id="debates.read",
            resource_type=ResourceType.DEBATE,
            resource_id="debate-123",
        )
        perm2 = store.grant_permission(
            user_id="user-1",
            permission_id="debates.read",
            resource_type=ResourceType.DEBATE,
            resource_id="debate-123",
        )
        # Should be same permission (updated)
        assert perm1.id == perm2.id
        # Should only have one permission
        assert store.count_permissions() == 1

    def test_list_with_no_permissions(self, store):
        """List operations with no permissions."""
        perms = store.list_permissions_for_user("nonexistent-user")
        assert perms == []

        perms = store.list_permissions_for_resource(ResourceType.DEBATE, "nonexistent")
        assert perms == []

    def test_revoke_all_with_filters(self, store):
        """Revoke all with resource type and ID filters."""
        store.grant_permission(
            user_id="user-1",
            permission_id="debates.read",
            resource_type=ResourceType.DEBATE,
            resource_id="debate-1",
        )
        store.grant_permission(
            user_id="user-1",
            permission_id="agents.read",
            resource_type=ResourceType.AGENT,
            resource_id="agent-1",
        )

        # Revoke only debates
        count = store.revoke_all_for_user("user-1", resource_type=ResourceType.DEBATE)
        assert count == 1
        # Agent permission should remain
        assert (
            store.check_resource_permission("user-1", "agents.read", ResourceType.AGENT, "agent-1")
            is True
        )

    def test_expired_permission_in_list(self, store):
        """Expired permissions excluded from list by default."""
        past = datetime.now(timezone.utc) - timedelta(hours=1)
        store.grant_permission(
            user_id="user-1",
            permission_id="debates.read",
            resource_type=ResourceType.DEBATE,
            resource_id="debate-1",
            expires_at=past,
        )
        store.grant_permission(
            user_id="user-1",
            permission_id="debates.read",
            resource_type=ResourceType.DEBATE,
            resource_id="debate-2",
        )

        # Without expired
        perms = store.list_permissions_for_user("user-1")
        assert len(perms) == 1

        # With expired
        perms_with_expired = store.list_permissions_for_user("user-1", include_expired=True)
        assert len(perms_with_expired) == 2
