"""
RBAC Resource-Level Permissions - Fine-grained access control for specific resources.

Provides the ability to check permissions against specific resource IDs,
enabling enterprise-grade fine-grained RBAC. For example:
- "Can user X read debate Y?" rather than just "Can user X read debates?"
- Grant specific users access to specific resources without role changes
- Time-limited resource access with automatic expiration

This module extends the role-based permission system with resource-level grants,
allowing for precise access control while maintaining performance through caching.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Protocol
from uuid import uuid4

from .models import ResourceType

if TYPE_CHECKING:
    from .cache import RBACDistributedCache

logger = logging.getLogger(__name__)


@dataclass
class ResourcePermission:
    """
    A permission grant for a specific resource.

    Represents the ability for a user to perform a specific action
    on a specific resource instance.

    Attributes:
        id: Unique identifier for this permission grant
        user_id: User receiving the permission
        permission_id: Permission key (e.g., "debates.read")
        resource_type: Type of resource (e.g., ResourceType.DEBATE)
        resource_id: Specific resource ID this permission applies to
        granted_at: When the permission was granted
        granted_by: User who granted the permission
        expires_at: When the permission expires (None = never)
        is_active: Whether the permission is currently active
        org_id: Organization scope for this permission
        conditions: Optional conditions for conditional access (ABAC)
        metadata: Additional metadata for auditing/tracking
    """

    id: str
    user_id: str
    permission_id: str
    resource_type: ResourceType
    resource_id: str
    granted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    granted_by: str | None = None
    expires_at: datetime | None = None
    is_active: bool = True
    org_id: str | None = None
    conditions: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        user_id: str,
        permission_id: str,
        resource_type: ResourceType,
        resource_id: str,
        granted_by: str | None = None,
        expires_at: datetime | None = None,
        org_id: str | None = None,
        conditions: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ResourcePermission:
        """
        Factory method to create a new ResourcePermission.

        Args:
            user_id: User receiving the permission
            permission_id: Permission key (e.g., "debates.read")
            resource_type: Type of resource
            resource_id: Specific resource ID
            granted_by: User granting the permission
            expires_at: Optional expiration time
            org_id: Organization scope
            conditions: Optional ABAC conditions
            metadata: Optional metadata

        Returns:
            New ResourcePermission instance
        """
        return cls(
            id=str(uuid4()),
            user_id=user_id,
            permission_id=permission_id,
            resource_type=resource_type,
            resource_id=resource_id,
            granted_at=datetime.now(timezone.utc),
            granted_by=granted_by,
            expires_at=expires_at,
            org_id=org_id,
            conditions=conditions or {},
            metadata=metadata or {},
        )

    @property
    def is_expired(self) -> bool:
        """Check if the permission has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    @property
    def is_valid(self) -> bool:
        """Check if the permission is currently valid (active and not expired)."""
        return self.is_active and not self.is_expired

    @property
    def action(self) -> str:
        """Extract the action from the permission_id."""
        if "." in self.permission_id:
            return self.permission_id.split(".", 1)[1]
        return self.permission_id

    def matches(
        self,
        user_id: str,
        permission_id: str,
        resource_type: ResourceType,
        resource_id: str,
        org_id: str | None = None,
    ) -> bool:
        """
        Check if this permission matches the requested access.

        Args:
            user_id: User requesting access
            permission_id: Permission being checked
            resource_type: Type of resource
            resource_id: Specific resource ID
            org_id: Organization context

        Returns:
            True if this permission grants the requested access
        """
        if not self.is_valid:
            return False

        if self.user_id != user_id:
            return False

        if self.resource_type != resource_type:
            return False

        if self.resource_id != resource_id:
            return False

        # Check org scope if specified
        if self.org_id is not None and org_id is not None and self.org_id != org_id:
            return False

        # Check permission match (exact or wildcard)
        if self.permission_id == permission_id:
            return True

        # Check wildcard (e.g., "debates.*" matches "debates.read")
        if self.permission_id.endswith(".*"):
            resource_prefix = self.permission_id[:-2]
            if permission_id.startswith(resource_prefix + "."):
                return True

        return False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "permission_id": self.permission_id,
            "resource_type": self.resource_type.value,
            "resource_id": self.resource_id,
            "granted_at": self.granted_at.isoformat(),
            "granted_by": self.granted_by,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "is_active": self.is_active,
            "org_id": self.org_id,
            "conditions": self.conditions,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResourcePermission:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            user_id=data["user_id"],
            permission_id=data["permission_id"],
            resource_type=ResourceType(data["resource_type"]),
            resource_id=data["resource_id"],
            granted_at=(
                datetime.fromisoformat(data["granted_at"])
                if isinstance(data["granted_at"], str)
                else data["granted_at"]
            ),
            granted_by=data.get("granted_by"),
            expires_at=(
                datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None
            ),
            is_active=data.get("is_active", True),
            org_id=data.get("org_id"),
            conditions=data.get("conditions", {}),
            metadata=data.get("metadata", {}),
        )


class ResourcePermissionBackend(Protocol):
    """
    Protocol for resource permission storage backends.

    Implement this protocol to provide custom storage (e.g., PostgreSQL, Redis).
    """

    def save(self, permission: ResourcePermission) -> None:
        """Save a resource permission."""
        ...

    def delete(self, permission_id: str) -> bool:
        """Delete a resource permission by ID."""
        ...

    def get(self, permission_id: str) -> ResourcePermission | None:
        """Get a resource permission by ID."""
        ...

    def find_by_user(self, user_id: str, org_id: str | None = None) -> list[ResourcePermission]:
        """Find all permissions for a user."""
        ...

    def find_by_resource(
        self,
        resource_type: ResourceType,
        resource_id: str,
    ) -> list[ResourcePermission]:
        """Find all permissions for a resource."""
        ...

    def find_matching(
        self,
        user_id: str,
        permission_id: str,
        resource_type: ResourceType,
        resource_id: str,
        org_id: str | None = None,
    ) -> ResourcePermission | None:
        """Find a matching permission."""
        ...


class ResourcePermissionStore:
    """
    Store for managing resource-level permissions.

    Provides CRUD operations for resource permissions with caching
    for performance. Supports both in-memory and distributed storage.

    Example:
        store = ResourcePermissionStore()

        # Grant permission
        perm = store.grant_permission(
            user_id="user-123",
            permission_id="debates.read",
            resource_type=ResourceType.DEBATE,
            resource_id="debate-456",
            granted_by="admin-1",
        )

        # Check permission
        has_access = store.check_resource_permission(
            user_id="user-123",
            permission_id="debates.read",
            resource_type=ResourceType.DEBATE,
            resource_id="debate-456",
        )

        # Revoke permission
        store.revoke_permission(perm.id)
    """

    def __init__(
        self,
        cache_ttl: int = 300,
        enable_cache: bool = True,
        cache_backend: "RBACDistributedCache | None" = None,
    ) -> None:
        """
        Initialize the resource permission store.

        Args:
            cache_ttl: Cache time-to-live in seconds (default 5 minutes)
            enable_cache: Whether to enable permission caching
            cache_backend: Optional distributed cache backend
        """
        self._cache_ttl = cache_ttl
        self._enable_cache = enable_cache
        self._cache_backend = cache_backend

        # In-memory storage (primary or fallback)
        self._permissions: dict[str, ResourcePermission] = {}

        # Indexes for fast lookup
        self._by_user: dict[str, set[str]] = {}  # user_id -> permission_ids
        self._by_resource: dict[str, set[str]] = {}  # resource_key -> permission_ids
        self._by_user_resource: dict[str, set[str]] = {}  # user_resource_key -> permission_ids

        # Cache for check results
        self._check_cache: dict[str, tuple[bool, datetime]] = {}

    def grant_permission(
        self,
        user_id: str,
        permission_id: str,
        resource_type: ResourceType,
        resource_id: str,
        granted_by: str | None = None,
        expires_at: datetime | None = None,
        org_id: str | None = None,
        conditions: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ResourcePermission:
        """
        Grant a resource-level permission to a user.

        Args:
            user_id: User receiving the permission
            permission_id: Permission key (e.g., "debates.read")
            resource_type: Type of resource
            resource_id: Specific resource ID
            granted_by: User granting the permission
            expires_at: Optional expiration time
            org_id: Organization scope
            conditions: Optional ABAC conditions
            metadata: Optional metadata

        Returns:
            The created ResourcePermission

        Raises:
            ValueError: If permission already exists
        """
        # Check for existing permission
        existing = self.find_permission(
            user_id=user_id,
            permission_id=permission_id,
            resource_type=resource_type,
            resource_id=resource_id,
            org_id=org_id,
        )
        if existing and existing.is_valid:
            logger.warning(
                f"Permission already exists: {existing.id} for user {user_id} "
                f"on {resource_type.value}/{resource_id}"
            )
            # Update existing permission instead of creating duplicate
            existing.expires_at = expires_at
            existing.is_active = True
            if conditions:
                existing.conditions = conditions
            if metadata:
                existing.metadata.update(metadata)
            self._invalidate_cache_for_user_resource(user_id, resource_type, resource_id)
            return existing

        # Create new permission
        permission = ResourcePermission.create(
            user_id=user_id,
            permission_id=permission_id,
            resource_type=resource_type,
            resource_id=resource_id,
            granted_by=granted_by,
            expires_at=expires_at,
            org_id=org_id,
            conditions=conditions,
            metadata=metadata,
        )

        self._store_permission(permission)
        self._invalidate_cache_for_user_resource(user_id, resource_type, resource_id)

        logger.info(
            f"Granted permission {permission_id} to user {user_id} "
            f"for {resource_type.value}/{resource_id}"
        )

        return permission

    def revoke_permission(self, permission_id: str) -> bool:
        """
        Revoke a resource permission by its ID.

        Args:
            permission_id: ID of the permission to revoke

        Returns:
            True if the permission was revoked, False if not found
        """
        permission = self._permissions.get(permission_id)
        if not permission:
            logger.warning(f"Permission not found for revocation: {permission_id}")
            return False

        # Mark as inactive (soft delete for audit trail)
        permission.is_active = False

        # Remove from indexes
        self._remove_from_indexes(permission)

        # Invalidate cache
        self._invalidate_cache_for_user_resource(
            permission.user_id,
            permission.resource_type,
            permission.resource_id,
        )

        logger.info(
            f"Revoked permission {permission.permission_id} from user {permission.user_id} "
            f"for {permission.resource_type.value}/{permission.resource_id}"
        )

        return True

    def revoke_all_for_user(
        self,
        user_id: str,
        resource_type: ResourceType | None = None,
        resource_id: str | None = None,
    ) -> int:
        """
        Revoke all resource permissions for a user.

        Args:
            user_id: User whose permissions to revoke
            resource_type: Optional filter by resource type
            resource_id: Optional filter by resource ID

        Returns:
            Number of permissions revoked
        """
        permission_ids = self._by_user.get(user_id, set()).copy()
        revoked_count = 0

        for perm_id in permission_ids:
            perm = self._permissions.get(perm_id)
            if not perm or not perm.is_active:
                continue

            if resource_type and perm.resource_type != resource_type:
                continue
            if resource_id and perm.resource_id != resource_id:
                continue

            if self.revoke_permission(perm_id):
                revoked_count += 1

        return revoked_count

    def revoke_all_for_resource(
        self,
        resource_type: ResourceType,
        resource_id: str,
    ) -> int:
        """
        Revoke all permissions for a specific resource.

        Useful when deleting a resource to clean up permissions.

        Args:
            resource_type: Type of resource
            resource_id: Specific resource ID

        Returns:
            Number of permissions revoked
        """
        resource_key = self._resource_key(resource_type, resource_id)
        permission_ids = self._by_resource.get(resource_key, set()).copy()
        revoked_count = 0

        for perm_id in permission_ids:
            if self.revoke_permission(perm_id):
                revoked_count += 1

        return revoked_count

    def check_resource_permission(
        self,
        user_id: str,
        permission_id: str,
        resource_type: ResourceType,
        resource_id: str,
        org_id: str | None = None,
    ) -> bool:
        """
        Check if a user has a specific permission on a resource.

        This checks resource-level permissions only, not role-based permissions.

        Args:
            user_id: User to check
            permission_id: Permission key (e.g., "debates.read")
            resource_type: Type of resource
            resource_id: Specific resource ID
            org_id: Organization context

        Returns:
            True if the user has the permission on the resource
        """
        # Check cache first
        if self._enable_cache:
            cache_key = self._check_cache_key(
                user_id, permission_id, resource_type, resource_id, org_id
            )
            cached = self._get_cached_check(cache_key)
            if cached is not None:
                return cached

        # Find matching permission
        permission = self.find_permission(
            user_id=user_id,
            permission_id=permission_id,
            resource_type=resource_type,
            resource_id=resource_id,
            org_id=org_id,
        )

        result = permission is not None and permission.is_valid

        # Cache result
        if self._enable_cache:
            self._cache_check(cache_key, result)

        return result

    def find_permission(
        self,
        user_id: str,
        permission_id: str,
        resource_type: ResourceType,
        resource_id: str,
        org_id: str | None = None,
    ) -> ResourcePermission | None:
        """
        Find a specific resource permission.

        Args:
            user_id: User to check
            permission_id: Permission key
            resource_type: Type of resource
            resource_id: Specific resource ID
            org_id: Organization context

        Returns:
            The matching ResourcePermission or None
        """
        # Use user+resource index for faster lookup
        user_resource_key = self._user_resource_key(user_id, resource_type, resource_id)
        permission_ids = self._by_user_resource.get(user_resource_key, set())

        for perm_id in permission_ids:
            perm = self._permissions.get(perm_id)
            if perm and perm.matches(user_id, permission_id, resource_type, resource_id, org_id):
                return perm

        # Also check for wildcard permissions
        for perm_id in permission_ids:
            perm = self._permissions.get(perm_id)
            if perm and perm.is_valid and perm.permission_id.endswith(".*"):
                if perm.matches(user_id, permission_id, resource_type, resource_id, org_id):
                    return perm

        return None

    def list_permissions_for_resource(
        self,
        resource_type: ResourceType,
        resource_id: str,
        include_expired: bool = False,
    ) -> list[ResourcePermission]:
        """
        List all permissions for a specific resource.

        Args:
            resource_type: Type of resource
            resource_id: Specific resource ID
            include_expired: Whether to include expired permissions

        Returns:
            List of ResourcePermission objects
        """
        resource_key = self._resource_key(resource_type, resource_id)
        permission_ids = self._by_resource.get(resource_key, set())

        permissions = []
        for perm_id in permission_ids:
            perm = self._permissions.get(perm_id)
            if perm:
                if include_expired or perm.is_valid:
                    permissions.append(perm)

        return sorted(permissions, key=lambda p: p.granted_at, reverse=True)

    def list_permissions_for_user(
        self,
        user_id: str,
        resource_type: ResourceType | None = None,
        include_expired: bool = False,
    ) -> list[ResourcePermission]:
        """
        List all resource permissions for a user.

        Args:
            user_id: User to query
            resource_type: Optional filter by resource type
            include_expired: Whether to include expired permissions

        Returns:
            List of ResourcePermission objects
        """
        permission_ids = self._by_user.get(user_id, set())

        permissions = []
        for perm_id in permission_ids:
            perm = self._permissions.get(perm_id)
            if perm:
                if resource_type and perm.resource_type != resource_type:
                    continue
                if include_expired or perm.is_valid:
                    permissions.append(perm)

        return sorted(permissions, key=lambda p: p.granted_at, reverse=True)

    def get_permission(self, permission_id: str) -> ResourcePermission | None:
        """
        Get a specific resource permission by ID.

        Args:
            permission_id: ID of the permission

        Returns:
            The ResourcePermission or None if not found
        """
        return self._permissions.get(permission_id)

    def count_permissions(
        self,
        user_id: str | None = None,
        resource_type: ResourceType | None = None,
        resource_id: str | None = None,
        active_only: bool = True,
    ) -> int:
        """
        Count resource permissions matching criteria.

        Args:
            user_id: Optional filter by user
            resource_type: Optional filter by resource type
            resource_id: Optional filter by resource ID
            active_only: Only count active (non-expired, non-revoked) permissions

        Returns:
            Count of matching permissions
        """
        count = 0
        for perm in self._permissions.values():
            if user_id and perm.user_id != user_id:
                continue
            if resource_type and perm.resource_type != resource_type:
                continue
            if resource_id and perm.resource_id != resource_id:
                continue
            if active_only and not perm.is_valid:
                continue
            count += 1
        return count

    def cleanup_expired(self) -> int:
        """
        Remove expired permissions from the store.

        Returns:
            Number of permissions removed
        """
        expired_ids = [
            perm_id
            for perm_id, perm in self._permissions.items()
            if perm.is_expired or not perm.is_active
        ]

        for perm_id in expired_ids:
            perm = self._permissions.pop(perm_id, None)
            if perm:
                self._remove_from_indexes(perm)

        # Clear check cache on cleanup
        self._check_cache.clear()

        logger.info(f"Cleaned up {len(expired_ids)} expired/inactive permissions")
        return len(expired_ids)

    def clear_cache(self, user_id: str | None = None) -> None:
        """
        Clear the permission check cache.

        Args:
            user_id: If provided, only clear cache for this user
        """
        if user_id:
            keys_to_remove = [k for k in self._check_cache if k.startswith(f"{user_id}:")]
            for key in keys_to_remove:
                del self._check_cache[key]
        else:
            self._check_cache.clear()

    def get_stats(self) -> dict[str, Any]:
        """
        Get statistics about the permission store.

        Returns:
            Dictionary with store statistics
        """
        active_count = sum(1 for p in self._permissions.values() if p.is_valid)
        expired_count = sum(1 for p in self._permissions.values() if p.is_expired)
        inactive_count = sum(1 for p in self._permissions.values() if not p.is_active)

        return {
            "total_permissions": len(self._permissions),
            "active_permissions": active_count,
            "expired_permissions": expired_count,
            "inactive_permissions": inactive_count,
            "unique_users": len(self._by_user),
            "unique_resources": len(self._by_resource),
            "cache_enabled": self._enable_cache,
            "cache_size": len(self._check_cache),
            "cache_ttl": self._cache_ttl,
        }

    # Private helper methods

    def _store_permission(self, permission: ResourcePermission) -> None:
        """Store a permission and update indexes."""
        self._permissions[permission.id] = permission

        # Update user index
        if permission.user_id not in self._by_user:
            self._by_user[permission.user_id] = set()
        self._by_user[permission.user_id].add(permission.id)

        # Update resource index
        resource_key = self._resource_key(permission.resource_type, permission.resource_id)
        if resource_key not in self._by_resource:
            self._by_resource[resource_key] = set()
        self._by_resource[resource_key].add(permission.id)

        # Update user+resource index
        user_resource_key = self._user_resource_key(
            permission.user_id, permission.resource_type, permission.resource_id
        )
        if user_resource_key not in self._by_user_resource:
            self._by_user_resource[user_resource_key] = set()
        self._by_user_resource[user_resource_key].add(permission.id)

    def _remove_from_indexes(self, permission: ResourcePermission) -> None:
        """Remove a permission from indexes."""
        # Remove from user index
        if permission.user_id in self._by_user:
            self._by_user[permission.user_id].discard(permission.id)

        # Remove from resource index
        resource_key = self._resource_key(permission.resource_type, permission.resource_id)
        if resource_key in self._by_resource:
            self._by_resource[resource_key].discard(permission.id)

        # Remove from user+resource index
        user_resource_key = self._user_resource_key(
            permission.user_id, permission.resource_type, permission.resource_id
        )
        if user_resource_key in self._by_user_resource:
            self._by_user_resource[user_resource_key].discard(permission.id)

    def _resource_key(self, resource_type: ResourceType, resource_id: str) -> str:
        """Generate a key for resource indexing."""
        return f"{resource_type.value}:{resource_id}"

    def _user_resource_key(
        self, user_id: str, resource_type: ResourceType, resource_id: str
    ) -> str:
        """Generate a key for user+resource indexing."""
        return f"{user_id}:{resource_type.value}:{resource_id}"

    def _check_cache_key(
        self,
        user_id: str,
        permission_id: str,
        resource_type: ResourceType,
        resource_id: str,
        org_id: str | None,
    ) -> str:
        """Generate a cache key for permission checks."""
        return f"{user_id}:{permission_id}:{resource_type.value}:{resource_id}:{org_id or ''}"

    def _get_cached_check(self, cache_key: str) -> bool | None:
        """Get a cached permission check result."""
        if cache_key not in self._check_cache:
            return None

        result, cached_at = self._check_cache[cache_key]
        age = (datetime.now(timezone.utc) - cached_at).total_seconds()

        if age > self._cache_ttl:
            del self._check_cache[cache_key]
            return None

        return result

    def _cache_check(self, cache_key: str, result: bool) -> None:
        """Cache a permission check result."""
        self._check_cache[cache_key] = (result, datetime.now(timezone.utc))

    def _invalidate_cache_for_user_resource(
        self,
        user_id: str,
        resource_type: ResourceType,
        resource_id: str,
    ) -> None:
        """Invalidate cache entries for a user+resource combination."""
        prefix = f"{user_id}:"
        resource_suffix = f":{resource_type.value}:{resource_id}:"
        keys_to_remove = [
            k for k in self._check_cache if k.startswith(prefix) and resource_suffix in k
        ]
        for key in keys_to_remove:
            del self._check_cache[key]


# Global resource permission store instance
_resource_permission_store: ResourcePermissionStore | None = None


def get_resource_permission_store() -> ResourcePermissionStore:
    """Get or create the global resource permission store."""
    global _resource_permission_store
    if _resource_permission_store is None:
        _resource_permission_store = ResourcePermissionStore()
    return _resource_permission_store


def set_resource_permission_store(store: ResourcePermissionStore | None) -> None:
    """Set the global resource permission store."""
    global _resource_permission_store
    _resource_permission_store = store


def grant_resource_permission(
    user_id: str,
    permission_id: str,
    resource_type: ResourceType,
    resource_id: str,
    granted_by: str | None = None,
    expires_at: datetime | None = None,
    org_id: str | None = None,
) -> ResourcePermission:
    """Convenience function to grant a resource permission using the global store."""
    return get_resource_permission_store().grant_permission(
        user_id=user_id,
        permission_id=permission_id,
        resource_type=resource_type,
        resource_id=resource_id,
        granted_by=granted_by,
        expires_at=expires_at,
        org_id=org_id,
    )


def revoke_resource_permission(permission_id: str) -> bool:
    """Convenience function to revoke a resource permission using the global store."""
    return get_resource_permission_store().revoke_permission(permission_id)


def check_resource_permission(
    user_id: str,
    permission_id: str,
    resource_type: ResourceType,
    resource_id: str,
    org_id: str | None = None,
) -> bool:
    """Convenience function to check a resource permission using the global store."""
    return get_resource_permission_store().check_resource_permission(
        user_id=user_id,
        permission_id=permission_id,
        resource_type=resource_type,
        resource_id=resource_id,
        org_id=org_id,
    )
