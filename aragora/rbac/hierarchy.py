"""
RBAC Resource Hierarchy - Parent-child resource relationships.

Enables hierarchical permission inheritance where granting access to a parent
resource (e.g., Organization) cascades to child resources (Workspace -> Project -> Debate).

Usage:
    from aragora.rbac.hierarchy import HierarchyRegistry, ResourceNode

    registry = HierarchyRegistry()

    # Register a workspace under an organization
    registry.register(ResourceNode(
        resource_type=ResourceType.WORKSPACE,
        resource_id="ws-123",
        parent_type=ResourceType.ORGANIZATION,
        parent_id="org-456",
    ))

    # Get ancestors for hierarchical permission check
    ancestors = registry.get_ancestors(ResourceType.WORKSPACE, "ws-123")
    # Returns: [ResourceNode(ORGANIZATION, "org-456")]

Use Cases:
    1. Organization Admin: Grant on org -> inherits to all workspaces/projects/debates
    2. Workspace Manager: Grant on workspace -> inherits to projects in that workspace
    3. Project Lead: Grant on project -> inherits to all debates in project
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .models import ResourceType

logger = logging.getLogger(__name__)


@dataclass
class ResourceNode:
    """
    Node in the resource hierarchy tree.

    Represents a resource and its parent relationship.

    Attributes:
        resource_type: Type of this resource
        resource_id: Unique identifier for this resource
        parent_type: Type of the parent resource (None if root)
        parent_id: ID of the parent resource (None if root)
        created_at: When this hierarchy relationship was established
        metadata: Additional metadata about the relationship
    """

    resource_type: ResourceType
    resource_id: str
    parent_type: ResourceType | None = None
    parent_id: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        """Generate unique key for this node."""
        return f"{self.resource_type.value}:{self.resource_id}"

    @property
    def parent_key(self) -> str | None:
        """Generate key for parent node."""
        if self.parent_type and self.parent_id:
            return f"{self.parent_type.value}:{self.parent_id}"
        return None

    @property
    def has_parent(self) -> bool:
        """Check if this node has a parent."""
        return self.parent_type is not None and self.parent_id is not None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "resource_type": self.resource_type.value,
            "resource_id": self.resource_id,
            "parent_type": self.parent_type.value if self.parent_type else None,
            "parent_id": self.parent_id,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


class HierarchyRegistry:
    """
    Registry for resource parent-child relationships.

    Maintains a graph of resource hierarchy relationships, enabling:
    - Hierarchical permission lookups (check parent if child has no direct grant)
    - Cascade invalidation (invalidate children when parent permission changes)
    - Ancestry queries (get full path from child to root)

    Thread-safe for read operations. Write operations should be synchronized
    externally if used in concurrent contexts.
    """

    # Predefined type hierarchy: which resource types can be parents of which
    # Key = child type, Value = list of valid parent types (in order of preference)
    TYPE_HIERARCHY: dict[ResourceType, list[ResourceType]] = {
        # Debates can belong to workspaces or directly to orgs
        ResourceType.DEBATE: [ResourceType.WORKSPACE, ResourceType.ORGANIZATION],
        # Workflows can belong to workspaces or directly to orgs
        ResourceType.WORKFLOW: [ResourceType.WORKSPACE, ResourceType.ORGANIZATION],
        # Agents can belong to workspaces or directly to orgs
        ResourceType.AGENT: [ResourceType.WORKSPACE, ResourceType.ORGANIZATION],
        # Knowledge bases can belong to workspaces or directly to orgs
        ResourceType.KNOWLEDGE: [ResourceType.WORKSPACE, ResourceType.ORGANIZATION],
        # Evidence belongs to debates or workspaces
        ResourceType.EVIDENCE: [ResourceType.DEBATE, ResourceType.WORKSPACE],
        # Workspaces belong to organizations
        ResourceType.WORKSPACE: [ResourceType.ORGANIZATION],
        # Teams belong to organizations
        ResourceType.TEAM: [ResourceType.ORGANIZATION],
        # Connectors belong to workspaces or orgs
        ResourceType.CONNECTOR: [ResourceType.WORKSPACE, ResourceType.ORGANIZATION],
        # Bots belong to workspaces or orgs
        ResourceType.BOT: [ResourceType.WORKSPACE, ResourceType.ORGANIZATION],
        # Templates belong to workspaces or orgs
        ResourceType.TEMPLATE: [ResourceType.WORKSPACE, ResourceType.ORGANIZATION],
        # API keys belong to users or orgs
        ResourceType.API_KEY: [ResourceType.USER, ResourceType.ORGANIZATION],
    }

    def __init__(self) -> None:
        """Initialize the hierarchy registry."""
        # Primary storage: key -> ResourceNode
        self._nodes: dict[str, ResourceNode] = {}

        # Index: parent_key -> set of child keys (for efficient children lookup)
        self._children_index: dict[str, set[str]] = {}

        # Statistics
        self._stats = {
            "registrations": 0,
            "unregistrations": 0,
            "ancestor_lookups": 0,
            "children_lookups": 0,
        }

    def register(self, node: ResourceNode) -> None:
        """
        Register a resource in the hierarchy.

        Args:
            node: ResourceNode to register

        Raises:
            ValueError: If parent type is not valid for the resource type
        """
        # Validate parent type if specified
        if node.has_parent:
            valid_parents = self.TYPE_HIERARCHY.get(node.resource_type, [])
            if node.parent_type not in valid_parents:
                raise ValueError(
                    f"Invalid parent type {node.parent_type} for {node.resource_type}. "
                    f"Valid parents: {valid_parents}"
                )

        key = node.key
        parent_key = node.parent_key

        # Remove from old parent's children if re-registering
        old_node = self._nodes.get(key)
        if old_node and old_node.parent_key:
            old_children = self._children_index.get(old_node.parent_key)
            if old_children:
                old_children.discard(key)

        # Register node
        self._nodes[key] = node

        # Update children index
        if parent_key:
            if parent_key not in self._children_index:
                self._children_index[parent_key] = set()
            self._children_index[parent_key].add(key)

        self._stats["registrations"] += 1
        logger.debug(f"Registered hierarchy: {key} -> parent: {parent_key}")

    def unregister(self, resource_type: ResourceType, resource_id: str) -> bool:
        """
        Remove a resource from the hierarchy.

        Note: Does not automatically remove children. Children will become
        orphaned (have invalid parent references).

        Args:
            resource_type: Type of resource to remove
            resource_id: ID of resource to remove

        Returns:
            True if resource was found and removed, False otherwise
        """
        key = f"{resource_type.value}:{resource_id}"
        node = self._nodes.get(key)

        if not node:
            return False

        # Remove from parent's children
        if node.parent_key:
            children = self._children_index.get(node.parent_key)
            if children:
                children.discard(key)

        # Remove this node's children index entry
        self._children_index.pop(key, None)

        # Remove the node
        del self._nodes[key]

        self._stats["unregistrations"] += 1
        logger.debug(f"Unregistered hierarchy: {key}")
        return True

    def get_node(self, resource_type: ResourceType, resource_id: str) -> ResourceNode | None:
        """
        Get a registered resource node.

        Args:
            resource_type: Type of resource
            resource_id: ID of resource

        Returns:
            ResourceNode if found, None otherwise
        """
        key = f"{resource_type.value}:{resource_id}"
        return self._nodes.get(key)

    def get_parent(self, resource_type: ResourceType, resource_id: str) -> ResourceNode | None:
        """
        Get the parent of a resource.

        Args:
            resource_type: Type of resource
            resource_id: ID of resource

        Returns:
            Parent ResourceNode if exists, None otherwise
        """
        node = self.get_node(resource_type, resource_id)
        if not node or not node.parent_key:
            return None

        return self._nodes.get(node.parent_key)

    def get_ancestors(
        self,
        resource_type: ResourceType,
        resource_id: str,
        max_depth: int | None = None,
    ) -> list[ResourceNode]:
        """
        Get all ancestors of a resource (parent, grandparent, etc.).

        Args:
            resource_type: Type of resource
            resource_id: ID of resource
            max_depth: Maximum depth to traverse (None = unlimited)

        Returns:
            List of ancestor ResourceNodes, from nearest (parent) to farthest (root)
        """
        self._stats["ancestor_lookups"] += 1

        ancestors: list[ResourceNode] = []
        current = self.get_node(resource_type, resource_id)
        depth = 0

        while current and current.has_parent:
            if max_depth is not None and depth >= max_depth:
                break

            parent = self._nodes.get(current.parent_key)  # type: ignore
            if parent:
                ancestors.append(parent)
                current = parent
                depth += 1
            else:
                # Parent not registered, stop traversal
                break

        return ancestors

    def get_children(
        self,
        resource_type: ResourceType,
        resource_id: str,
        recursive: bool = False,
        max_depth: int | None = None,
    ) -> list[ResourceNode]:
        """
        Get children of a resource.

        Args:
            resource_type: Type of resource
            resource_id: ID of resource
            recursive: If True, get all descendants (children, grandchildren, etc.)
            max_depth: Maximum depth for recursive lookup (None = unlimited)

        Returns:
            List of child ResourceNodes
        """
        self._stats["children_lookups"] += 1

        key = f"{resource_type.value}:{resource_id}"
        child_keys = self._children_index.get(key, set())

        children: list[ResourceNode] = []
        for child_key in child_keys:
            child = self._nodes.get(child_key)
            if child:
                children.append(child)

                # Recursive descent
                if recursive and (max_depth is None or max_depth > 1):
                    descendants = self.get_children(
                        child.resource_type,
                        child.resource_id,
                        recursive=True,
                        max_depth=max_depth - 1 if max_depth else None,
                    )
                    children.extend(descendants)

        return children

    def get_all_descendant_keys(
        self,
        resource_type: ResourceType,
        resource_id: str,
    ) -> set[str]:
        """
        Get all descendant keys (for cache invalidation).

        More efficient than get_children(recursive=True) when you only need keys.

        Args:
            resource_type: Type of resource
            resource_id: ID of resource

        Returns:
            Set of descendant keys (resource_type:resource_id format)
        """
        key = f"{resource_type.value}:{resource_id}"
        result: set[str] = set()

        def collect_descendants(parent_key: str) -> None:
            child_keys = self._children_index.get(parent_key, set())
            for child_key in child_keys:
                if child_key not in result:
                    result.add(child_key)
                    collect_descendants(child_key)

        collect_descendants(key)
        return result

    def is_ancestor_of(
        self,
        ancestor_type: ResourceType,
        ancestor_id: str,
        descendant_type: ResourceType,
        descendant_id: str,
    ) -> bool:
        """
        Check if one resource is an ancestor of another.

        Args:
            ancestor_type: Type of potential ancestor
            ancestor_id: ID of potential ancestor
            descendant_type: Type of potential descendant
            descendant_id: ID of potential descendant

        Returns:
            True if ancestor_id is an ancestor of descendant_id
        """
        ancestor_key = f"{ancestor_type.value}:{ancestor_id}"
        ancestors = self.get_ancestors(descendant_type, descendant_id)
        return any(a.key == ancestor_key for a in ancestors)

    def get_valid_parent_types(self, resource_type: ResourceType) -> list[ResourceType]:
        """
        Get valid parent types for a resource type.

        Args:
            resource_type: The resource type to query

        Returns:
            List of valid parent ResourceTypes
        """
        return self.TYPE_HIERARCHY.get(resource_type, [])

    def get_stats(self) -> dict[str, int]:
        """Get registry statistics."""
        return {
            **self._stats,
            "total_nodes": len(self._nodes),
            "total_relationships": sum(len(children) for children in self._children_index.values()),
        }

    def clear(self) -> None:
        """Clear all registered hierarchies."""
        self._nodes.clear()
        self._children_index.clear()
        logger.info("Cleared hierarchy registry")


# Singleton instance
_hierarchy_registry: HierarchyRegistry | None = None


def get_hierarchy_registry() -> HierarchyRegistry:
    """Get the global HierarchyRegistry instance."""
    global _hierarchy_registry
    if _hierarchy_registry is None:
        _hierarchy_registry = HierarchyRegistry()
    return _hierarchy_registry


def set_hierarchy_registry(registry: HierarchyRegistry | None) -> None:
    """Set the global HierarchyRegistry instance (for testing)."""
    global _hierarchy_registry
    _hierarchy_registry = registry


__all__ = [
    "ResourceNode",
    "HierarchyRegistry",
    "get_hierarchy_registry",
    "set_hierarchy_registry",
]
