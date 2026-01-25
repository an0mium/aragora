"""
Tests for RBAC Resource Ownership module.

Tests cover:
- Owner tracking (set, get, is_owner)
- Implicit owner permissions
- Ownership transfer with audit trail
- Cache behavior
- Integration with AuthorizationContext
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock

from aragora.rbac.ownership import (
    OwnershipRecord,
    OwnershipManager,
    DEFAULT_OWNER_PERMISSIONS,
    get_ownership_manager,
    set_ownership_manager,
    set_resource_owner,
    get_resource_owner,
    is_resource_owner,
    check_owner_permission,
)
from aragora.rbac.models import (
    ResourceType,
    AuthorizationContext,
    AuthorizationDecision,
)


class TestOwnershipRecord:
    """Tests for OwnershipRecord dataclass."""

    def test_create_ownership_record(self):
        """Test creating an ownership record."""
        record = OwnershipRecord.create(
            resource_type=ResourceType.DEBATE,
            resource_id="debate-123",
            owner_id="user-456",
            org_id="org-789",
        )

        assert record.id is not None
        assert record.resource_type == ResourceType.DEBATE
        assert record.resource_id == "debate-123"
        assert record.owner_id == "user-456"
        assert record.org_id == "org-789"
        assert record.created_at is not None
        assert record.transferred_from is None
        assert record.transferred_at is None
        assert record.transferred_by is None

    def test_create_with_metadata(self):
        """Test creating record with metadata."""
        metadata = {"created_by_api": True, "source": "web"}
        record = OwnershipRecord.create(
            resource_type=ResourceType.WORKFLOW,
            resource_id="wf-1",
            owner_id="user-1",
            org_id="org-1",
            metadata=metadata,
        )

        assert record.metadata == metadata

    def test_to_dict(self):
        """Test serialization to dictionary."""
        record = OwnershipRecord.create(
            resource_type=ResourceType.DEBATE,
            resource_id="debate-123",
            owner_id="user-456",
            org_id="org-789",
        )

        data = record.to_dict()

        assert data["id"] == record.id
        assert data["resource_type"] == "debates"
        assert data["resource_id"] == "debate-123"
        assert data["owner_id"] == "user-456"
        assert data["org_id"] == "org-789"
        assert "created_at" in data

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "id": "record-1",
            "resource_type": "debates",
            "resource_id": "debate-123",
            "owner_id": "user-456",
            "org_id": "org-789",
            "created_at": "2024-01-01T00:00:00+00:00",
            "transferred_from": None,
            "transferred_at": None,
            "transferred_by": None,
            "metadata": {},
        }

        record = OwnershipRecord.from_dict(data)

        assert record.id == "record-1"
        assert record.resource_type == ResourceType.DEBATE
        assert record.resource_id == "debate-123"
        assert record.owner_id == "user-456"

    def test_from_dict_with_transfer(self):
        """Test deserialization with transfer data."""
        data = {
            "id": "record-1",
            "resource_type": "debates",
            "resource_id": "debate-123",
            "owner_id": "user-new",
            "org_id": "org-789",
            "created_at": "2024-01-01T00:00:00+00:00",
            "transferred_from": "user-old",
            "transferred_at": "2024-01-02T00:00:00+00:00",
            "transferred_by": "admin-1",
            "metadata": {},
        }

        record = OwnershipRecord.from_dict(data)

        assert record.transferred_from == "user-old"
        assert record.transferred_at is not None
        assert record.transferred_by == "admin-1"


class TestOwnershipManager:
    """Tests for OwnershipManager class."""

    @pytest.fixture
    def manager(self):
        """Create a fresh ownership manager for each test."""
        return OwnershipManager()

    def test_set_owner(self, manager):
        """Test setting resource owner."""
        record = manager.set_owner(
            resource_type=ResourceType.DEBATE,
            resource_id="debate-123",
            owner_id="user-456",
            org_id="org-789",
        )

        assert record.owner_id == "user-456"
        assert record.resource_type == ResourceType.DEBATE
        assert record.resource_id == "debate-123"

    def test_set_owner_replaces_existing(self, manager):
        """Test that set_owner replaces existing ownership."""
        # Set initial owner
        manager.set_owner(
            resource_type=ResourceType.DEBATE,
            resource_id="debate-123",
            owner_id="user-1",
            org_id="org-1",
        )

        # Replace with new owner
        record = manager.set_owner(
            resource_type=ResourceType.DEBATE,
            resource_id="debate-123",
            owner_id="user-2",
            org_id="org-1",
        )

        assert record.owner_id == "user-2"
        assert manager.get_owner(ResourceType.DEBATE, "debate-123") == "user-2"

    def test_set_owner_same_owner_no_change(self, manager):
        """Test that setting same owner returns existing record."""
        record1 = manager.set_owner(
            resource_type=ResourceType.DEBATE,
            resource_id="debate-123",
            owner_id="user-1",
            org_id="org-1",
        )

        record2 = manager.set_owner(
            resource_type=ResourceType.DEBATE,
            resource_id="debate-123",
            owner_id="user-1",
            org_id="org-1",
        )

        assert record1.id == record2.id

    def test_get_owner(self, manager):
        """Test getting resource owner."""
        manager.set_owner(
            resource_type=ResourceType.DEBATE,
            resource_id="debate-123",
            owner_id="user-456",
            org_id="org-789",
        )

        owner = manager.get_owner(ResourceType.DEBATE, "debate-123")

        assert owner == "user-456"

    def test_get_owner_not_found(self, manager):
        """Test getting owner for unowned resource."""
        owner = manager.get_owner(ResourceType.DEBATE, "nonexistent")

        assert owner is None

    def test_is_owner(self, manager):
        """Test checking ownership."""
        manager.set_owner(
            resource_type=ResourceType.DEBATE,
            resource_id="debate-123",
            owner_id="user-456",
            org_id="org-789",
        )

        assert manager.is_owner("user-456", ResourceType.DEBATE, "debate-123") is True
        assert manager.is_owner("user-other", ResourceType.DEBATE, "debate-123") is False

    def test_get_ownership_record(self, manager):
        """Test getting full ownership record."""
        manager.set_owner(
            resource_type=ResourceType.DEBATE,
            resource_id="debate-123",
            owner_id="user-456",
            org_id="org-789",
        )

        record = manager.get_ownership_record(ResourceType.DEBATE, "debate-123")

        assert record is not None
        assert record.owner_id == "user-456"
        assert record.org_id == "org-789"

    def test_get_ownership_record_not_found(self, manager):
        """Test getting record for unowned resource."""
        record = manager.get_ownership_record(ResourceType.DEBATE, "nonexistent")

        assert record is None


class TestOwnershipTransfer:
    """Tests for ownership transfer functionality."""

    @pytest.fixture
    def manager(self):
        """Create a fresh ownership manager for each test."""
        return OwnershipManager()

    def test_transfer_ownership(self, manager):
        """Test transferring ownership."""
        manager.set_owner(
            resource_type=ResourceType.DEBATE,
            resource_id="debate-123",
            owner_id="user-old",
            org_id="org-1",
        )

        record = manager.transfer_ownership(
            resource_type=ResourceType.DEBATE,
            resource_id="debate-123",
            new_owner_id="user-new",
            transferred_by="admin-1",
            reason="User requested transfer",
        )

        assert record is not None
        assert record.owner_id == "user-new"
        assert record.transferred_from == "user-old"
        assert record.transferred_by == "admin-1"
        assert record.transferred_at is not None

    def test_transfer_ownership_updates_get_owner(self, manager):
        """Test that transfer updates ownership lookup."""
        manager.set_owner(
            resource_type=ResourceType.DEBATE,
            resource_id="debate-123",
            owner_id="user-old",
            org_id="org-1",
        )

        manager.transfer_ownership(
            resource_type=ResourceType.DEBATE,
            resource_id="debate-123",
            new_owner_id="user-new",
        )

        assert manager.get_owner(ResourceType.DEBATE, "debate-123") == "user-new"
        assert manager.is_owner("user-new", ResourceType.DEBATE, "debate-123") is True
        assert manager.is_owner("user-old", ResourceType.DEBATE, "debate-123") is False

    def test_transfer_ownership_not_found(self, manager):
        """Test transferring non-existent resource."""
        record = manager.transfer_ownership(
            resource_type=ResourceType.DEBATE,
            resource_id="nonexistent",
            new_owner_id="user-new",
        )

        assert record is None

    def test_transfer_ownership_same_owner(self, manager):
        """Test transferring to same owner is no-op."""
        manager.set_owner(
            resource_type=ResourceType.DEBATE,
            resource_id="debate-123",
            owner_id="user-1",
            org_id="org-1",
        )

        record = manager.transfer_ownership(
            resource_type=ResourceType.DEBATE,
            resource_id="debate-123",
            new_owner_id="user-1",
        )

        assert record is not None
        assert record.transferred_from is None  # No transfer occurred

    def test_transfer_history(self, manager):
        """Test transfer history tracking."""
        manager.set_owner(
            resource_type=ResourceType.DEBATE,
            resource_id="debate-123",
            owner_id="user-1",
            org_id="org-1",
        )

        manager.transfer_ownership(
            resource_type=ResourceType.DEBATE,
            resource_id="debate-123",
            new_owner_id="user-2",
            transferred_by="admin-1",
            reason="First transfer",
        )

        manager.transfer_ownership(
            resource_type=ResourceType.DEBATE,
            resource_id="debate-123",
            new_owner_id="user-3",
            transferred_by="admin-2",
            reason="Second transfer",
        )

        history = manager.get_transfer_history()

        assert len(history) == 2
        assert history[0]["from_owner"] == "user-1"
        assert history[0]["to_owner"] == "user-2"
        assert history[1]["from_owner"] == "user-2"
        assert history[1]["to_owner"] == "user-3"

    def test_transfer_history_filter(self, manager):
        """Test filtering transfer history."""
        manager.set_owner(
            resource_type=ResourceType.DEBATE,
            resource_id="debate-1",
            owner_id="user-1",
            org_id="org-1",
        )
        manager.set_owner(
            resource_type=ResourceType.WORKFLOW,
            resource_id="wf-1",
            owner_id="user-1",
            org_id="org-1",
        )

        manager.transfer_ownership(ResourceType.DEBATE, "debate-1", "user-2")
        manager.transfer_ownership(ResourceType.WORKFLOW, "wf-1", "user-2")

        debate_history = manager.get_transfer_history(resource_type=ResourceType.DEBATE)
        wf_history = manager.get_transfer_history(resource_type=ResourceType.WORKFLOW)

        assert len(debate_history) == 1
        assert len(wf_history) == 1


class TestImplicitOwnerPermissions:
    """Tests for implicit owner permission checking."""

    @pytest.fixture
    def manager(self):
        """Create a fresh ownership manager for each test."""
        return OwnershipManager()

    @pytest.fixture
    def context(self):
        """Create a test authorization context."""
        return AuthorizationContext(
            user_id="user-456",
            org_id="org-789",
        )

    def test_check_owner_permission_allowed(self, manager, context):
        """Test that owner gets implicit permissions."""
        manager.set_owner(
            resource_type=ResourceType.DEBATE,
            resource_id="debate-123",
            owner_id="user-456",
            org_id="org-789",
        )

        decision = manager.check_owner_permission(
            context=context,
            resource_type=ResourceType.DEBATE,
            permission_key="debates.read",
            resource_id="debate-123",
        )

        assert decision is not None
        assert decision.allowed is True
        assert "Owner implicit access" in decision.reason

    def test_check_owner_permission_not_owner(self, manager):
        """Test that non-owner does not get implicit permission."""
        context = AuthorizationContext(user_id="user-other", org_id="org-789")

        manager.set_owner(
            resource_type=ResourceType.DEBATE,
            resource_id="debate-123",
            owner_id="user-456",
            org_id="org-789",
        )

        decision = manager.check_owner_permission(
            context=context,
            resource_type=ResourceType.DEBATE,
            permission_key="debates.read",
            resource_id="debate-123",
        )

        assert decision is None  # Not owner, continue to other checks

    def test_check_owner_permission_not_implicit(self, manager, context):
        """Test that owner doesn't get non-implicit permissions."""
        manager.set_owner(
            resource_type=ResourceType.DEBATE,
            resource_id="debate-123",
            owner_id="user-456",
            org_id="org-789",
        )

        # "debates.admin" is not in the implicit permissions
        decision = manager.check_owner_permission(
            context=context,
            resource_type=ResourceType.DEBATE,
            permission_key="debates.some_random_action",
            resource_id="debate-123",
        )

        # Returns None because the permission is not in implicit list
        assert decision is None

    def test_get_implicit_permissions(self, manager):
        """Test getting implicit permissions for resource type."""
        perms = manager.get_implicit_permissions(ResourceType.DEBATE)

        assert "debates.read" in perms
        assert "debates.update" in perms
        assert "debates.delete" in perms
        assert "debates.run" in perms

    def test_custom_owner_permissions(self):
        """Test manager with custom owner permissions."""
        custom_perms = {
            ResourceType.DEBATE: ["debates.read", "debates.custom_action"],
        }
        manager = OwnershipManager(owner_permissions=custom_perms)

        perms = manager.get_implicit_permissions(ResourceType.DEBATE)

        assert perms == ["debates.read", "debates.custom_action"]


class TestOwnershipByOwnerQueries:
    """Tests for querying resources by owner."""

    @pytest.fixture
    def manager(self):
        """Create a fresh ownership manager for each test."""
        return OwnershipManager()

    def test_get_owned_resources(self, manager):
        """Test getting all resources owned by a user."""
        manager.set_owner(ResourceType.DEBATE, "debate-1", "user-1", "org-1")
        manager.set_owner(ResourceType.DEBATE, "debate-2", "user-1", "org-1")
        manager.set_owner(ResourceType.WORKFLOW, "wf-1", "user-1", "org-1")
        manager.set_owner(ResourceType.DEBATE, "debate-3", "user-2", "org-1")

        owned = manager.get_owned_resources("user-1")

        assert len(owned) == 3
        resource_ids = [r.resource_id for r in owned]
        assert "debate-1" in resource_ids
        assert "debate-2" in resource_ids
        assert "wf-1" in resource_ids
        assert "debate-3" not in resource_ids

    def test_get_owned_resources_by_type(self, manager):
        """Test filtering owned resources by type."""
        manager.set_owner(ResourceType.DEBATE, "debate-1", "user-1", "org-1")
        manager.set_owner(ResourceType.WORKFLOW, "wf-1", "user-1", "org-1")

        debates = manager.get_owned_resources("user-1", resource_type=ResourceType.DEBATE)
        workflows = manager.get_owned_resources("user-1", resource_type=ResourceType.WORKFLOW)

        assert len(debates) == 1
        assert len(workflows) == 1
        assert debates[0].resource_id == "debate-1"
        assert workflows[0].resource_id == "wf-1"

    def test_get_owned_resources_by_org(self, manager):
        """Test filtering owned resources by organization."""
        manager.set_owner(ResourceType.DEBATE, "debate-1", "user-1", "org-1")
        manager.set_owner(ResourceType.DEBATE, "debate-2", "user-1", "org-2")

        org1_resources = manager.get_owned_resources("user-1", org_id="org-1")
        org2_resources = manager.get_owned_resources("user-1", org_id="org-2")

        assert len(org1_resources) == 1
        assert len(org2_resources) == 1

    def test_get_owned_resources_empty(self, manager):
        """Test getting resources for user with no ownership."""
        owned = manager.get_owned_resources("user-nonexistent")

        assert owned == []


class TestOwnershipRemoval:
    """Tests for ownership removal."""

    @pytest.fixture
    def manager(self):
        """Create a fresh ownership manager for each test."""
        return OwnershipManager()

    def test_remove_ownership(self, manager):
        """Test removing ownership record."""
        manager.set_owner(ResourceType.DEBATE, "debate-123", "user-1", "org-1")

        removed = manager.remove_ownership(ResourceType.DEBATE, "debate-123")

        assert removed is True
        assert manager.get_owner(ResourceType.DEBATE, "debate-123") is None

    def test_remove_ownership_not_found(self, manager):
        """Test removing non-existent ownership."""
        removed = manager.remove_ownership(ResourceType.DEBATE, "nonexistent")

        assert removed is False

    def test_remove_ownership_clears_indexes(self, manager):
        """Test that removal clears all indexes."""
        manager.set_owner(ResourceType.DEBATE, "debate-123", "user-1", "org-1")

        manager.remove_ownership(ResourceType.DEBATE, "debate-123")

        # Owner should no longer have this in their list
        owned = manager.get_owned_resources("user-1")
        assert len(owned) == 0


class TestOwnershipCache:
    """Tests for ownership caching behavior."""

    def test_cache_enabled(self):
        """Test that caching works when enabled."""
        manager = OwnershipManager(enable_cache=True, cache_ttl=300)

        manager.set_owner(ResourceType.DEBATE, "debate-123", "user-1", "org-1")

        # First call populates cache
        owner1 = manager.get_owner(ResourceType.DEBATE, "debate-123")

        # Second call should hit cache
        owner2 = manager.get_owner(ResourceType.DEBATE, "debate-123")

        assert owner1 == owner2 == "user-1"
        assert len(manager._ownership_cache) > 0

    def test_cache_disabled(self):
        """Test that cache is not used when disabled."""
        manager = OwnershipManager(enable_cache=False)

        manager.set_owner(ResourceType.DEBATE, "debate-123", "user-1", "org-1")
        manager.get_owner(ResourceType.DEBATE, "debate-123")

        assert len(manager._ownership_cache) == 0

    def test_cache_invalidation_on_set(self):
        """Test that cache is invalidated when ownership changes."""
        manager = OwnershipManager(enable_cache=True)

        manager.set_owner(ResourceType.DEBATE, "debate-123", "user-1", "org-1")
        manager.get_owner(ResourceType.DEBATE, "debate-123")  # Populate cache

        # Change owner
        manager.set_owner(ResourceType.DEBATE, "debate-123", "user-2", "org-1")

        # Cache should be updated
        owner = manager.get_owner(ResourceType.DEBATE, "debate-123")
        assert owner == "user-2"

    def test_cache_invalidation_on_transfer(self):
        """Test that cache is invalidated on transfer."""
        manager = OwnershipManager(enable_cache=True)

        manager.set_owner(ResourceType.DEBATE, "debate-123", "user-1", "org-1")
        manager.get_owner(ResourceType.DEBATE, "debate-123")  # Populate cache

        # Transfer ownership
        manager.transfer_ownership(ResourceType.DEBATE, "debate-123", "user-2")

        # Cache should be updated
        owner = manager.get_owner(ResourceType.DEBATE, "debate-123")
        assert owner == "user-2"

    def test_clear_cache(self):
        """Test clearing the cache."""
        manager = OwnershipManager(enable_cache=True)

        manager.set_owner(ResourceType.DEBATE, "debate-123", "user-1", "org-1")
        manager.get_owner(ResourceType.DEBATE, "debate-123")

        assert len(manager._ownership_cache) > 0

        manager.clear_cache()

        assert len(manager._ownership_cache) == 0


class TestOwnershipStats:
    """Tests for ownership statistics."""

    @pytest.fixture
    def manager(self):
        """Create a fresh ownership manager for each test."""
        return OwnershipManager()

    def test_get_stats(self, manager):
        """Test getting ownership statistics."""
        manager.set_owner(ResourceType.DEBATE, "debate-1", "user-1", "org-1")
        manager.set_owner(ResourceType.DEBATE, "debate-2", "user-1", "org-1")
        manager.set_owner(ResourceType.WORKFLOW, "wf-1", "user-2", "org-1")

        stats = manager.get_stats()

        assert stats["total_records"] == 3
        assert stats["unique_owners"] == 2
        assert stats["unique_orgs"] == 1
        assert stats["by_resource_type"]["debates"] == 2
        assert stats["by_resource_type"]["workflows"] == 1

    def test_stats_with_transfers(self, manager):
        """Test stats include transfer count."""
        manager.set_owner(ResourceType.DEBATE, "debate-1", "user-1", "org-1")
        manager.transfer_ownership(ResourceType.DEBATE, "debate-1", "user-2")

        stats = manager.get_stats()

        assert stats["transfer_count"] == 1


class TestGlobalOwnershipManager:
    """Tests for global ownership manager functions."""

    def setup_method(self):
        """Reset global manager before each test."""
        set_ownership_manager(None)

    def teardown_method(self):
        """Reset global manager after each test."""
        set_ownership_manager(None)

    def test_get_ownership_manager_creates_instance(self):
        """Test that get_ownership_manager creates instance if none exists."""
        manager = get_ownership_manager()

        assert manager is not None
        assert isinstance(manager, OwnershipManager)

    def test_get_ownership_manager_returns_same_instance(self):
        """Test that get_ownership_manager returns same instance."""
        manager1 = get_ownership_manager()
        manager2 = get_ownership_manager()

        assert manager1 is manager2

    def test_set_ownership_manager(self):
        """Test setting custom ownership manager."""
        custom_manager = OwnershipManager(cache_ttl=600)

        set_ownership_manager(custom_manager)

        assert get_ownership_manager() is custom_manager

    def test_set_resource_owner_convenience(self):
        """Test set_resource_owner convenience function."""
        record = set_resource_owner(
            resource_type=ResourceType.DEBATE,
            resource_id="debate-123",
            owner_id="user-456",
            org_id="org-789",
        )

        assert record.owner_id == "user-456"
        assert get_resource_owner(ResourceType.DEBATE, "debate-123") == "user-456"

    def test_is_resource_owner_convenience(self):
        """Test is_resource_owner convenience function."""
        set_resource_owner(ResourceType.DEBATE, "debate-123", "user-456", "org-789")

        assert is_resource_owner("user-456", ResourceType.DEBATE, "debate-123") is True
        assert is_resource_owner("user-other", ResourceType.DEBATE, "debate-123") is False

    def test_check_owner_permission_convenience(self):
        """Test check_owner_permission convenience function."""
        set_resource_owner(ResourceType.DEBATE, "debate-123", "user-456", "org-789")

        context = AuthorizationContext(user_id="user-456", org_id="org-789")

        decision = check_owner_permission(
            context=context,
            resource_type=ResourceType.DEBATE,
            permission_key="debates.read",
            resource_id="debate-123",
        )

        assert decision is not None
        assert decision.allowed is True


class TestDefaultOwnerPermissions:
    """Tests for default owner permission configuration."""

    def test_debate_default_permissions(self):
        """Test default permissions for debates."""
        assert "debates.read" in DEFAULT_OWNER_PERMISSIONS[ResourceType.DEBATE]
        assert "debates.update" in DEFAULT_OWNER_PERMISSIONS[ResourceType.DEBATE]
        assert "debates.delete" in DEFAULT_OWNER_PERMISSIONS[ResourceType.DEBATE]
        assert "debates.run" in DEFAULT_OWNER_PERMISSIONS[ResourceType.DEBATE]
        assert "debates.stop" in DEFAULT_OWNER_PERMISSIONS[ResourceType.DEBATE]
        assert "debates.fork" in DEFAULT_OWNER_PERMISSIONS[ResourceType.DEBATE]

    def test_workflow_default_permissions(self):
        """Test default permissions for workflows."""
        assert "workflows.read" in DEFAULT_OWNER_PERMISSIONS[ResourceType.WORKFLOW]
        assert "workflows.update" in DEFAULT_OWNER_PERMISSIONS[ResourceType.WORKFLOW]
        assert "workflows.delete" in DEFAULT_OWNER_PERMISSIONS[ResourceType.WORKFLOW]
        assert "workflows.run" in DEFAULT_OWNER_PERMISSIONS[ResourceType.WORKFLOW]

    def test_agent_default_permissions(self):
        """Test default permissions for agents."""
        assert "agents.read" in DEFAULT_OWNER_PERMISSIONS[ResourceType.AGENT]
        assert "agents.update" in DEFAULT_OWNER_PERMISSIONS[ResourceType.AGENT]
        assert "agents.delete" in DEFAULT_OWNER_PERMISSIONS[ResourceType.AGENT]
        assert "agents.deploy" in DEFAULT_OWNER_PERMISSIONS[ResourceType.AGENT]
