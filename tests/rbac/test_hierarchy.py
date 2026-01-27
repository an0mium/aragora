"""
Tests for RBAC Resource Hierarchy - Permission Inheritance.

Tests hierarchical permission inheritance where granting access to a parent
resource (e.g., Organization) cascades to child resources (Workspace -> Debate).
"""

import pytest

from aragora.rbac.hierarchy import (
    HierarchyRegistry,
    ResourceNode,
    get_hierarchy_registry,
    set_hierarchy_registry,
)
from aragora.rbac.models import ResourceType
from aragora.rbac.resource_permissions import (
    ResourcePermission,
    ResourcePermissionStore,
)


class TestResourceNode:
    """Tests for ResourceNode dataclass."""

    def test_node_creation(self):
        """Create a basic resource node."""
        node = ResourceNode(
            resource_type=ResourceType.DEBATE,
            resource_id="debate-123",
        )
        assert node.resource_type == ResourceType.DEBATE
        assert node.resource_id == "debate-123"
        assert node.parent_type is None
        assert node.parent_id is None

    def test_node_with_parent(self):
        """Create a resource node with parent."""
        node = ResourceNode(
            resource_type=ResourceType.DEBATE,
            resource_id="debate-123",
            parent_type=ResourceType.WORKSPACE,
            parent_id="ws-456",
        )
        assert node.has_parent
        assert node.parent_type == ResourceType.WORKSPACE
        assert node.parent_id == "ws-456"

    def test_node_key(self):
        """Node key is resource_type:resource_id."""
        node = ResourceNode(
            resource_type=ResourceType.DEBATE,
            resource_id="debate-123",
        )
        assert node.key == "debates:debate-123"

    def test_node_parent_key(self):
        """Parent key is parent_type:parent_id."""
        node = ResourceNode(
            resource_type=ResourceType.DEBATE,
            resource_id="debate-123",
            parent_type=ResourceType.WORKSPACE,
            parent_id="ws-456",
        )
        assert node.parent_key == "workspace:ws-456"

    def test_node_parent_key_none(self):
        """Parent key is None when no parent."""
        node = ResourceNode(
            resource_type=ResourceType.DEBATE,
            resource_id="debate-123",
        )
        assert node.parent_key is None

    def test_node_to_dict(self):
        """Convert node to dictionary."""
        node = ResourceNode(
            resource_type=ResourceType.DEBATE,
            resource_id="debate-123",
            parent_type=ResourceType.WORKSPACE,
            parent_id="ws-456",
            metadata={"name": "Test Debate"},
        )
        data = node.to_dict()
        assert data["resource_type"] == "debates"
        assert data["resource_id"] == "debate-123"
        assert data["parent_type"] == "workspace"
        assert data["parent_id"] == "ws-456"
        assert data["metadata"] == {"name": "Test Debate"}


class TestHierarchyRegistry:
    """Tests for HierarchyRegistry."""

    @pytest.fixture
    def registry(self):
        """Create a fresh registry for each test."""
        return HierarchyRegistry()

    def test_register_node(self, registry):
        """Register a resource node."""
        node = ResourceNode(
            resource_type=ResourceType.WORKSPACE,
            resource_id="ws-123",
            parent_type=ResourceType.ORGANIZATION,
            parent_id="org-456",
        )
        registry.register(node)
        retrieved = registry.get_node(ResourceType.WORKSPACE, "ws-123")
        assert retrieved is not None
        assert retrieved.resource_id == "ws-123"

    def test_register_invalid_parent_type(self, registry):
        """Raise error when parent type is not valid for resource type."""
        # Debates can't have debates as parents
        node = ResourceNode(
            resource_type=ResourceType.DEBATE,
            resource_id="debate-123",
            parent_type=ResourceType.DEBATE,  # Invalid!
            parent_id="debate-456",
        )
        with pytest.raises(ValueError, match="Invalid parent type"):
            registry.register(node)

    def test_get_parent(self, registry):
        """Get the parent of a resource."""
        # Register org
        org_node = ResourceNode(
            resource_type=ResourceType.ORGANIZATION,
            resource_id="org-123",
        )
        registry.register(org_node)

        # Register workspace under org
        ws_node = ResourceNode(
            resource_type=ResourceType.WORKSPACE,
            resource_id="ws-456",
            parent_type=ResourceType.ORGANIZATION,
            parent_id="org-123",
        )
        registry.register(ws_node)

        parent = registry.get_parent(ResourceType.WORKSPACE, "ws-456")
        assert parent is not None
        assert parent.resource_type == ResourceType.ORGANIZATION
        assert parent.resource_id == "org-123"

    def test_get_parent_no_parent(self, registry):
        """Get parent returns None for root resources."""
        node = ResourceNode(
            resource_type=ResourceType.ORGANIZATION,
            resource_id="org-123",
        )
        registry.register(node)
        parent = registry.get_parent(ResourceType.ORGANIZATION, "org-123")
        assert parent is None

    def test_get_ancestors(self, registry):
        """Get ancestors returns full ancestry chain."""
        # Create hierarchy: org -> workspace -> debate
        org_node = ResourceNode(
            resource_type=ResourceType.ORGANIZATION,
            resource_id="org-1",
        )
        ws_node = ResourceNode(
            resource_type=ResourceType.WORKSPACE,
            resource_id="ws-1",
            parent_type=ResourceType.ORGANIZATION,
            parent_id="org-1",
        )
        debate_node = ResourceNode(
            resource_type=ResourceType.DEBATE,
            resource_id="debate-1",
            parent_type=ResourceType.WORKSPACE,
            parent_id="ws-1",
        )

        registry.register(org_node)
        registry.register(ws_node)
        registry.register(debate_node)

        ancestors = registry.get_ancestors(ResourceType.DEBATE, "debate-1")
        assert len(ancestors) == 2
        # First ancestor is immediate parent (workspace)
        assert ancestors[0].resource_type == ResourceType.WORKSPACE
        assert ancestors[0].resource_id == "ws-1"
        # Second ancestor is grandparent (org)
        assert ancestors[1].resource_type == ResourceType.ORGANIZATION
        assert ancestors[1].resource_id == "org-1"

    def test_get_ancestors_max_depth(self, registry):
        """Get ancestors respects max_depth."""
        # Create hierarchy: org -> workspace -> debate
        org_node = ResourceNode(
            resource_type=ResourceType.ORGANIZATION,
            resource_id="org-1",
        )
        ws_node = ResourceNode(
            resource_type=ResourceType.WORKSPACE,
            resource_id="ws-1",
            parent_type=ResourceType.ORGANIZATION,
            parent_id="org-1",
        )
        debate_node = ResourceNode(
            resource_type=ResourceType.DEBATE,
            resource_id="debate-1",
            parent_type=ResourceType.WORKSPACE,
            parent_id="ws-1",
        )

        registry.register(org_node)
        registry.register(ws_node)
        registry.register(debate_node)

        # Only get immediate parent
        ancestors = registry.get_ancestors(ResourceType.DEBATE, "debate-1", max_depth=1)
        assert len(ancestors) == 1
        assert ancestors[0].resource_type == ResourceType.WORKSPACE

    def test_get_children(self, registry):
        """Get children of a resource."""
        org_node = ResourceNode(
            resource_type=ResourceType.ORGANIZATION,
            resource_id="org-1",
        )
        ws_node1 = ResourceNode(
            resource_type=ResourceType.WORKSPACE,
            resource_id="ws-1",
            parent_type=ResourceType.ORGANIZATION,
            parent_id="org-1",
        )
        ws_node2 = ResourceNode(
            resource_type=ResourceType.WORKSPACE,
            resource_id="ws-2",
            parent_type=ResourceType.ORGANIZATION,
            parent_id="org-1",
        )

        registry.register(org_node)
        registry.register(ws_node1)
        registry.register(ws_node2)

        children = registry.get_children(ResourceType.ORGANIZATION, "org-1")
        assert len(children) == 2
        child_ids = {c.resource_id for c in children}
        assert child_ids == {"ws-1", "ws-2"}

    def test_get_children_recursive(self, registry):
        """Get all descendants recursively."""
        # Create hierarchy: org -> workspace -> debate
        org_node = ResourceNode(
            resource_type=ResourceType.ORGANIZATION,
            resource_id="org-1",
        )
        ws_node = ResourceNode(
            resource_type=ResourceType.WORKSPACE,
            resource_id="ws-1",
            parent_type=ResourceType.ORGANIZATION,
            parent_id="org-1",
        )
        debate_node = ResourceNode(
            resource_type=ResourceType.DEBATE,
            resource_id="debate-1",
            parent_type=ResourceType.WORKSPACE,
            parent_id="ws-1",
        )

        registry.register(org_node)
        registry.register(ws_node)
        registry.register(debate_node)

        descendants = registry.get_children(ResourceType.ORGANIZATION, "org-1", recursive=True)
        assert len(descendants) == 2
        types = {d.resource_type for d in descendants}
        assert types == {ResourceType.WORKSPACE, ResourceType.DEBATE}

    def test_unregister(self, registry):
        """Unregister removes node from registry."""
        node = ResourceNode(
            resource_type=ResourceType.WORKSPACE,
            resource_id="ws-123",
        )
        registry.register(node)
        assert registry.get_node(ResourceType.WORKSPACE, "ws-123") is not None

        result = registry.unregister(ResourceType.WORKSPACE, "ws-123")
        assert result is True
        assert registry.get_node(ResourceType.WORKSPACE, "ws-123") is None

    def test_unregister_not_found(self, registry):
        """Unregister returns False when node not found."""
        result = registry.unregister(ResourceType.WORKSPACE, "nonexistent")
        assert result is False

    def test_is_ancestor_of(self, registry):
        """Check if one resource is an ancestor of another."""
        org_node = ResourceNode(
            resource_type=ResourceType.ORGANIZATION,
            resource_id="org-1",
        )
        ws_node = ResourceNode(
            resource_type=ResourceType.WORKSPACE,
            resource_id="ws-1",
            parent_type=ResourceType.ORGANIZATION,
            parent_id="org-1",
        )
        debate_node = ResourceNode(
            resource_type=ResourceType.DEBATE,
            resource_id="debate-1",
            parent_type=ResourceType.WORKSPACE,
            parent_id="ws-1",
        )

        registry.register(org_node)
        registry.register(ws_node)
        registry.register(debate_node)

        # Org is ancestor of debate
        assert registry.is_ancestor_of(
            ResourceType.ORGANIZATION, "org-1", ResourceType.DEBATE, "debate-1"
        )
        # Workspace is ancestor of debate
        assert registry.is_ancestor_of(
            ResourceType.WORKSPACE, "ws-1", ResourceType.DEBATE, "debate-1"
        )
        # Debate is not ancestor of org
        assert not registry.is_ancestor_of(
            ResourceType.DEBATE, "debate-1", ResourceType.ORGANIZATION, "org-1"
        )

    def test_get_valid_parent_types(self, registry):
        """Get valid parent types for a resource type."""
        valid = registry.get_valid_parent_types(ResourceType.DEBATE)
        assert ResourceType.WORKSPACE in valid
        assert ResourceType.ORGANIZATION in valid

    def test_clear(self, registry):
        """Clear all registered hierarchies."""
        node = ResourceNode(
            resource_type=ResourceType.WORKSPACE,
            resource_id="ws-123",
        )
        registry.register(node)
        registry.clear()
        assert registry.get_node(ResourceType.WORKSPACE, "ws-123") is None

    def test_get_stats(self, registry):
        """Get registry statistics."""
        org_node = ResourceNode(
            resource_type=ResourceType.ORGANIZATION,
            resource_id="org-1",
        )
        ws_node = ResourceNode(
            resource_type=ResourceType.WORKSPACE,
            resource_id="ws-1",
            parent_type=ResourceType.ORGANIZATION,
            parent_id="org-1",
        )
        registry.register(org_node)
        registry.register(ws_node)

        stats = registry.get_stats()
        assert stats["total_nodes"] == 2
        assert stats["total_relationships"] == 1
        assert stats["registrations"] == 2


class TestHierarchySingleton:
    """Tests for hierarchy singleton functions."""

    def test_get_and_set_hierarchy_registry(self):
        """Get and set the global registry."""
        original = get_hierarchy_registry()

        new_registry = HierarchyRegistry()
        set_hierarchy_registry(new_registry)

        assert get_hierarchy_registry() is new_registry

        # Restore original
        set_hierarchy_registry(original)


class TestHierarchicalPermissions:
    """Tests for hierarchical permission inheritance."""

    @pytest.fixture
    def registry(self):
        """Create a registry with a hierarchy."""
        registry = HierarchyRegistry()

        # Create hierarchy: org -> workspace -> debate
        org_node = ResourceNode(
            resource_type=ResourceType.ORGANIZATION,
            resource_id="org-1",
        )
        ws_node = ResourceNode(
            resource_type=ResourceType.WORKSPACE,
            resource_id="ws-1",
            parent_type=ResourceType.ORGANIZATION,
            parent_id="org-1",
        )
        debate_node = ResourceNode(
            resource_type=ResourceType.DEBATE,
            resource_id="debate-1",
            parent_type=ResourceType.WORKSPACE,
            parent_id="ws-1",
        )

        registry.register(org_node)
        registry.register(ws_node)
        registry.register(debate_node)

        return registry

    @pytest.fixture
    def store(self, registry):
        """Create a permission store with hierarchy."""
        return ResourcePermissionStore(
            enable_cache=False,
            hierarchy_registry=registry,
        )

    def test_direct_permission_found(self, store):
        """Direct permission is found without hierarchy lookup."""
        store.grant_permission(
            user_id="user-1",
            permission_id="debates.read",
            resource_type=ResourceType.DEBATE,
            resource_id="debate-1",
        )

        perm, inherited = store.find_permission_hierarchical(
            user_id="user-1",
            permission_id="debates.read",
            resource_type=ResourceType.DEBATE,
            resource_id="debate-1",
        )

        assert perm is not None
        assert inherited is False

    def test_inherited_from_parent(self, store):
        """Permission inherited from parent workspace."""
        # Grant on workspace
        store.grant_permission(
            user_id="user-1",
            permission_id="debates.read",
            resource_type=ResourceType.WORKSPACE,
            resource_id="ws-1",
        )

        # Check on debate (should inherit)
        perm, inherited = store.find_permission_hierarchical(
            user_id="user-1",
            permission_id="debates.read",
            resource_type=ResourceType.DEBATE,
            resource_id="debate-1",
        )

        assert perm is not None
        assert inherited is True
        assert perm.resource_type == ResourceType.WORKSPACE
        assert perm.resource_id == "ws-1"

    def test_inherited_from_grandparent(self, store):
        """Permission inherited from grandparent organization."""
        # Grant on org
        store.grant_permission(
            user_id="user-1",
            permission_id="debates.read",
            resource_type=ResourceType.ORGANIZATION,
            resource_id="org-1",
        )

        # Check on debate (should inherit from org)
        perm, inherited = store.find_permission_hierarchical(
            user_id="user-1",
            permission_id="debates.read",
            resource_type=ResourceType.DEBATE,
            resource_id="debate-1",
        )

        assert perm is not None
        assert inherited is True
        assert perm.resource_type == ResourceType.ORGANIZATION
        assert perm.resource_id == "org-1"

    def test_inherit_to_children_false(self, store):
        """Permission with inherit_to_children=False doesn't cascade."""
        # Grant on workspace with no inheritance
        store.grant_permission(
            user_id="user-1",
            permission_id="debates.read",
            resource_type=ResourceType.WORKSPACE,
            resource_id="ws-1",
            inherit_to_children=False,
        )

        # Check on debate (should NOT inherit)
        perm, inherited = store.find_permission_hierarchical(
            user_id="user-1",
            permission_id="debates.read",
            resource_type=ResourceType.DEBATE,
            resource_id="debate-1",
        )

        assert perm is None
        assert inherited is False

    def test_max_inheritance_depth(self, store):
        """Permission with max_inheritance_depth limits inheritance."""
        # Grant on org with depth limit of 1
        store.grant_permission(
            user_id="user-1",
            permission_id="debates.read",
            resource_type=ResourceType.ORGANIZATION,
            resource_id="org-1",
            max_inheritance_depth=1,
        )

        # Check on workspace (depth 1, should inherit)
        perm, inherited = store.find_permission_hierarchical(
            user_id="user-1",
            permission_id="debates.read",
            resource_type=ResourceType.WORKSPACE,
            resource_id="ws-1",
        )
        assert perm is not None
        assert inherited is True

        # Check on debate (depth 2, should NOT inherit due to depth limit)
        perm, inherited = store.find_permission_hierarchical(
            user_id="user-1",
            permission_id="debates.read",
            resource_type=ResourceType.DEBATE,
            resource_id="debate-1",
        )
        assert perm is None
        assert inherited is False

    def test_check_resource_permission_with_hierarchy(self, store):
        """check_resource_permission uses hierarchy when enabled."""
        # Grant on org
        store.grant_permission(
            user_id="user-1",
            permission_id="debates.read",
            resource_type=ResourceType.ORGANIZATION,
            resource_id="org-1",
        )

        # Check on debate with hierarchy
        result = store.check_resource_permission(
            user_id="user-1",
            permission_id="debates.read",
            resource_type=ResourceType.DEBATE,
            resource_id="debate-1",
            check_hierarchy=True,
        )
        assert result is True

        # Check on debate without hierarchy
        result = store.check_resource_permission(
            user_id="user-1",
            permission_id="debates.read",
            resource_type=ResourceType.DEBATE,
            resource_id="debate-1",
            check_hierarchy=False,
        )
        assert result is False

    def test_no_permission_found(self, store):
        """No permission found returns (None, False)."""
        perm, inherited = store.find_permission_hierarchical(
            user_id="user-1",
            permission_id="debates.read",
            resource_type=ResourceType.DEBATE,
            resource_id="debate-1",
        )

        assert perm is None
        assert inherited is False

    def test_no_hierarchy_registry(self):
        """Store without hierarchy registry only checks direct permissions."""
        store = ResourcePermissionStore(enable_cache=False)

        # Grant on org (no hierarchy registered)
        store.grant_permission(
            user_id="user-1",
            permission_id="debates.read",
            resource_type=ResourceType.ORGANIZATION,
            resource_id="org-1",
        )

        # Check on debate (no hierarchy, so no inheritance)
        perm, inherited = store.find_permission_hierarchical(
            user_id="user-1",
            permission_id="debates.read",
            resource_type=ResourceType.DEBATE,
            resource_id="debate-1",
        )

        assert perm is None
        assert inherited is False


class TestResourcePermissionInheritanceFields:
    """Tests for new inheritance fields on ResourcePermission."""

    def test_default_inherit_to_children(self):
        """inherit_to_children defaults to True."""
        perm = ResourcePermission.create(
            user_id="user-1",
            permission_id="debates.read",
            resource_type=ResourceType.DEBATE,
            resource_id="debate-1",
        )
        assert perm.inherit_to_children is True

    def test_default_max_inheritance_depth(self):
        """max_inheritance_depth defaults to None (unlimited)."""
        perm = ResourcePermission.create(
            user_id="user-1",
            permission_id="debates.read",
            resource_type=ResourceType.DEBATE,
            resource_id="debate-1",
        )
        assert perm.max_inheritance_depth is None

    def test_custom_inheritance_settings(self):
        """Custom inheritance settings are preserved."""
        perm = ResourcePermission.create(
            user_id="user-1",
            permission_id="debates.read",
            resource_type=ResourceType.DEBATE,
            resource_id="debate-1",
            inherit_to_children=False,
            max_inheritance_depth=2,
        )
        assert perm.inherit_to_children is False
        assert perm.max_inheritance_depth == 2

    def test_to_dict_includes_inheritance_fields(self):
        """to_dict includes inheritance fields."""
        perm = ResourcePermission.create(
            user_id="user-1",
            permission_id="debates.read",
            resource_type=ResourceType.DEBATE,
            resource_id="debate-1",
            inherit_to_children=False,
            max_inheritance_depth=3,
        )
        data = perm.to_dict()
        assert data["inherit_to_children"] is False
        assert data["max_inheritance_depth"] == 3

    def test_from_dict_loads_inheritance_fields(self):
        """from_dict loads inheritance fields."""
        data = {
            "id": "perm-1",
            "user_id": "user-1",
            "permission_id": "debates.read",
            "resource_type": "debates",
            "resource_id": "debate-1",
            "granted_at": "2024-01-01T00:00:00",
            "inherit_to_children": False,
            "max_inheritance_depth": 5,
        }
        perm = ResourcePermission.from_dict(data)
        assert perm.inherit_to_children is False
        assert perm.max_inheritance_depth == 5

    def test_from_dict_defaults_inheritance_fields(self):
        """from_dict uses defaults when fields are missing."""
        data = {
            "id": "perm-1",
            "user_id": "user-1",
            "permission_id": "debates.read",
            "resource_type": "debates",
            "resource_id": "debate-1",
            "granted_at": "2024-01-01T00:00:00",
        }
        perm = ResourcePermission.from_dict(data)
        assert perm.inherit_to_children is True
        assert perm.max_inheritance_depth is None
