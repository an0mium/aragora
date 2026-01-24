"""
RBAC Permission Delegation - Allows users to delegate permissions to others.

Enables enterprise delegation workflows:
- Users can delegate their permissions to other users
- Time-limited delegations with automatic expiration
- Scope constraints (org, workspace, resource)
- Delegation chains with depth limits
- Audit trail for all delegation actions

Example:
    # Manager delegates debate permissions to assistant
    delegation = delegate_permission(
        delegator_id="manager-123",
        delegatee_id="assistant-456",
        permission_id="debates.create",
        org_id="org-789",
        expires_at=datetime.now() + timedelta(days=7),
        reason="Covering vacation",
    )

    # Check if delegatee has permission via delegation
    has_access = check_delegated_permission(
        user_id="assistant-456",
        permission_id="debates.create",
        org_id="org-789",
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from .models import ResourceType

logger = logging.getLogger(__name__)


class DelegationStatus(str, Enum):
    """Status of a permission delegation."""

    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    PENDING_APPROVAL = "pending_approval"


class DelegationConstraint(str, Enum):
    """Types of constraints that can be applied to delegations."""

    TIME_LIMIT = "time_limit"  # Maximum duration
    RESOURCE_SCOPE = "resource_scope"  # Specific resources only
    ACTION_SCOPE = "action_scope"  # Specific actions only
    NO_REDELEGATION = "no_redelegation"  # Cannot be further delegated
    APPROVAL_REQUIRED = "approval_required"  # Requires admin approval


@dataclass
class PermissionDelegation:
    """
    A delegation of permission from one user to another.

    Attributes:
        id: Unique identifier
        delegator_id: User granting the delegation
        delegatee_id: User receiving the delegation
        permission_id: Permission being delegated (e.g., "debates.create")
        org_id: Organization scope
        workspace_id: Optional workspace scope (more specific than org)
        resource_type: Optional resource type restriction
        resource_ids: Optional specific resource IDs
        created_at: When the delegation was created
        expires_at: When the delegation expires
        revoked_at: When the delegation was revoked (if applicable)
        revoked_by: User who revoked the delegation
        status: Current status of the delegation
        reason: Reason for the delegation
        can_redelegate: Whether delegatee can further delegate
        max_chain_depth: Maximum delegation chain depth allowed
        chain_depth: Current depth in delegation chain
        parent_delegation_id: If this is a redelegation, the parent delegation
        constraints: Additional constraints on the delegation
        metadata: Additional metadata for auditing
    """

    id: str
    delegator_id: str
    delegatee_id: str
    permission_id: str
    org_id: str | None = None
    workspace_id: str | None = None
    resource_type: ResourceType | None = None
    resource_ids: set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime | None = None
    revoked_at: datetime | None = None
    revoked_by: str | None = None
    status: DelegationStatus = DelegationStatus.ACTIVE
    reason: str = ""
    can_redelegate: bool = False
    max_chain_depth: int = 2
    chain_depth: int = 0
    parent_delegation_id: str | None = None
    constraints: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        delegator_id: str,
        delegatee_id: str,
        permission_id: str,
        org_id: str | None = None,
        workspace_id: str | None = None,
        resource_type: ResourceType | None = None,
        resource_ids: set[str] | None = None,
        expires_at: datetime | None = None,
        reason: str = "",
        can_redelegate: bool = False,
        max_chain_depth: int = 2,
        parent_delegation_id: str | None = None,
        chain_depth: int = 0,
    ) -> PermissionDelegation:
        """Factory method to create a new delegation."""
        return cls(
            id=str(uuid4()),
            delegator_id=delegator_id,
            delegatee_id=delegatee_id,
            permission_id=permission_id,
            org_id=org_id,
            workspace_id=workspace_id,
            resource_type=resource_type,
            resource_ids=resource_ids or set(),
            created_at=datetime.now(timezone.utc),
            expires_at=expires_at,
            reason=reason,
            can_redelegate=can_redelegate,
            max_chain_depth=max_chain_depth,
            chain_depth=chain_depth,
            parent_delegation_id=parent_delegation_id,
        )

    @property
    def is_expired(self) -> bool:
        """Check if the delegation has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    @property
    def is_valid(self) -> bool:
        """Check if the delegation is currently valid."""
        return self.status == DelegationStatus.ACTIVE and not self.is_expired

    def matches(
        self,
        user_id: str,
        permission_id: str,
        org_id: str | None = None,
        workspace_id: str | None = None,
        resource_type: ResourceType | None = None,
        resource_id: str | None = None,
    ) -> bool:
        """Check if this delegation matches the requested access."""
        if not self.is_valid:
            return False

        if self.delegatee_id != user_id:
            return False

        # Check permission match (exact or wildcard)
        if self.permission_id != permission_id:
            if not self.permission_id.endswith(".*"):
                return False
            prefix = self.permission_id[:-2]
            if not permission_id.startswith(prefix + "."):
                return False

        # Check org scope
        if self.org_id is not None and org_id is not None and self.org_id != org_id:
            return False

        # Check workspace scope
        if (
            self.workspace_id is not None
            and workspace_id is not None
            and self.workspace_id != workspace_id
        ):
            return False

        # Check resource type
        if (
            self.resource_type is not None
            and resource_type is not None
            and self.resource_type != resource_type
        ):
            return False

        # Check resource IDs
        if self.resource_ids and resource_id and resource_id not in self.resource_ids:
            return False

        return True

    def revoke(self, revoked_by: str) -> None:
        """Revoke this delegation."""
        self.status = DelegationStatus.REVOKED
        self.revoked_at = datetime.now(timezone.utc)
        self.revoked_by = revoked_by

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "delegator_id": self.delegator_id,
            "delegatee_id": self.delegatee_id,
            "permission_id": self.permission_id,
            "org_id": self.org_id,
            "workspace_id": self.workspace_id,
            "resource_type": self.resource_type.value if self.resource_type else None,
            "resource_ids": list(self.resource_ids),
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "revoked_at": self.revoked_at.isoformat() if self.revoked_at else None,
            "revoked_by": self.revoked_by,
            "status": self.status.value,
            "reason": self.reason,
            "can_redelegate": self.can_redelegate,
            "max_chain_depth": self.max_chain_depth,
            "chain_depth": self.chain_depth,
            "parent_delegation_id": self.parent_delegation_id,
            "constraints": self.constraints,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PermissionDelegation:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            delegator_id=data["delegator_id"],
            delegatee_id=data["delegatee_id"],
            permission_id=data["permission_id"],
            org_id=data.get("org_id"),
            workspace_id=data.get("workspace_id"),
            resource_type=ResourceType(data["resource_type"])
            if data.get("resource_type")
            else None,
            resource_ids=set(data.get("resource_ids", [])),
            created_at=datetime.fromisoformat(data["created_at"])
            if isinstance(data["created_at"], str)
            else data["created_at"],
            expires_at=datetime.fromisoformat(data["expires_at"])
            if data.get("expires_at")
            else None,
            revoked_at=datetime.fromisoformat(data["revoked_at"])
            if data.get("revoked_at")
            else None,
            revoked_by=data.get("revoked_by"),
            status=DelegationStatus(data.get("status", "active")),
            reason=data.get("reason", ""),
            can_redelegate=data.get("can_redelegate", False),
            max_chain_depth=data.get("max_chain_depth", 2),
            chain_depth=data.get("chain_depth", 0),
            parent_delegation_id=data.get("parent_delegation_id"),
            constraints=data.get("constraints", {}),
            metadata=data.get("metadata", {}),
        )


class DelegationManager:
    """
    Manages permission delegations between users.

    Provides CRUD operations for delegations with validation,
    chain depth enforcement, and expiration handling.
    """

    # Default max delegation duration
    DEFAULT_MAX_DURATION = timedelta(days=30)
    # Max allowed chain depth
    MAX_CHAIN_DEPTH = 3

    def __init__(
        self,
        max_duration: timedelta | None = None,
        max_chain_depth: int = MAX_CHAIN_DEPTH,
        require_approval_for_admin: bool = True,
    ) -> None:
        """
        Initialize the delegation manager.

        Args:
            max_duration: Maximum allowed delegation duration
            max_chain_depth: Maximum delegation chain depth
            require_approval_for_admin: Whether admin delegations need approval
        """
        self._max_duration = max_duration or self.DEFAULT_MAX_DURATION
        self._max_chain_depth = max_chain_depth
        self._require_approval_for_admin = require_approval_for_admin

        # In-memory storage (should be backed by database in production)
        self._delegations: dict[str, PermissionDelegation] = {}
        self._by_delegator: dict[str, set[str]] = {}
        self._by_delegatee: dict[str, set[str]] = {}

    def delegate(
        self,
        delegator_id: str,
        delegatee_id: str,
        permission_id: str,
        org_id: str | None = None,
        workspace_id: str | None = None,
        resource_type: ResourceType | None = None,
        resource_ids: set[str] | None = None,
        expires_at: datetime | None = None,
        reason: str = "",
        can_redelegate: bool = False,
        delegator_permissions: set[str] | None = None,
        parent_delegation: PermissionDelegation | None = None,
    ) -> PermissionDelegation:
        """
        Create a new permission delegation.

        Args:
            delegator_id: User granting the delegation
            delegatee_id: User receiving the delegation
            permission_id: Permission to delegate
            org_id: Organization scope
            workspace_id: Workspace scope
            resource_type: Resource type restriction
            resource_ids: Specific resource IDs
            expires_at: Expiration time
            reason: Reason for delegation
            can_redelegate: Whether delegatee can further delegate
            delegator_permissions: Delegator's current permissions (for validation)
            parent_delegation: Parent delegation if redelegating

        Returns:
            The created PermissionDelegation

        Raises:
            ValueError: If validation fails
        """
        # Validate delegator has the permission
        if delegator_permissions is not None:
            if permission_id not in delegator_permissions:
                # Check for wildcard
                resource = permission_id.split(".")[0]
                if (
                    f"{resource}.*" not in delegator_permissions
                    and "*" not in delegator_permissions
                ):
                    raise ValueError(
                        f"Delegator {delegator_id} does not have permission {permission_id}"
                    )

        # Validate chain depth
        chain_depth = 0
        if parent_delegation:
            chain_depth = parent_delegation.chain_depth + 1
            if chain_depth >= self._max_chain_depth:
                raise ValueError(
                    f"Maximum delegation chain depth ({self._max_chain_depth}) exceeded"
                )
            if not parent_delegation.can_redelegate:
                raise ValueError("Parent delegation does not allow redelegation")

        # Validate duration
        if expires_at:
            duration = expires_at - datetime.now(timezone.utc)
            if duration > self._max_duration:
                logger.warning(
                    f"Delegation duration {duration} exceeds max {self._max_duration}, capping"
                )
                expires_at = datetime.now(timezone.utc) + self._max_duration
        else:
            # Default to max duration
            expires_at = datetime.now(timezone.utc) + self._max_duration

        # Create delegation
        delegation = PermissionDelegation.create(
            delegator_id=delegator_id,
            delegatee_id=delegatee_id,
            permission_id=permission_id,
            org_id=org_id,
            workspace_id=workspace_id,
            resource_type=resource_type,
            resource_ids=resource_ids,
            expires_at=expires_at,
            reason=reason,
            can_redelegate=can_redelegate,
            max_chain_depth=self._max_chain_depth,
            chain_depth=chain_depth,
            parent_delegation_id=parent_delegation.id if parent_delegation else None,
        )

        # Store delegation
        self._store_delegation(delegation)

        logger.info(
            f"Created delegation {delegation.id}: {delegator_id} -> {delegatee_id} "
            f"for {permission_id} (expires: {expires_at})"
        )

        return delegation

    def revoke(self, delegation_id: str, revoked_by: str) -> bool:
        """
        Revoke a delegation.

        Also revokes any child delegations (redelegations from this delegation).

        Args:
            delegation_id: ID of the delegation to revoke
            revoked_by: User revoking the delegation

        Returns:
            True if revoked, False if not found
        """
        delegation = self._delegations.get(delegation_id)
        if not delegation:
            return False

        delegation.revoke(revoked_by)

        # Revoke child delegations
        for d in list(self._delegations.values()):
            if d.parent_delegation_id == delegation_id and d.is_valid:
                d.revoke(revoked_by)
                logger.info(f"Revoked child delegation {d.id}")

        logger.info(f"Revoked delegation {delegation_id} by {revoked_by}")
        return True

    def revoke_all_by_delegator(self, delegator_id: str, revoked_by: str) -> int:
        """Revoke all delegations made by a user."""
        delegation_ids = self._by_delegator.get(delegator_id, set()).copy()
        count = 0
        for d_id in delegation_ids:
            if self.revoke(d_id, revoked_by):
                count += 1
        return count

    def revoke_all_for_delegatee(self, delegatee_id: str, revoked_by: str) -> int:
        """Revoke all delegations granted to a user."""
        delegation_ids = self._by_delegatee.get(delegatee_id, set()).copy()
        count = 0
        for d_id in delegation_ids:
            if self.revoke(d_id, revoked_by):
                count += 1
        return count

    def check_delegation(
        self,
        user_id: str,
        permission_id: str,
        org_id: str | None = None,
        workspace_id: str | None = None,
        resource_type: ResourceType | None = None,
        resource_id: str | None = None,
    ) -> PermissionDelegation | None:
        """
        Check if a user has a delegated permission.

        Args:
            user_id: User to check
            permission_id: Permission to check
            org_id: Organization context
            workspace_id: Workspace context
            resource_type: Resource type
            resource_id: Resource ID

        Returns:
            The matching delegation or None
        """
        delegation_ids = self._by_delegatee.get(user_id, set())

        for d_id in delegation_ids:
            delegation = self._delegations.get(d_id)
            if delegation and delegation.matches(
                user_id=user_id,
                permission_id=permission_id,
                org_id=org_id,
                workspace_id=workspace_id,
                resource_type=resource_type,
                resource_id=resource_id,
            ):
                return delegation

        return None

    def has_delegated_permission(
        self,
        user_id: str,
        permission_id: str,
        org_id: str | None = None,
        workspace_id: str | None = None,
        resource_type: ResourceType | None = None,
        resource_id: str | None = None,
    ) -> bool:
        """Check if a user has a delegated permission (convenience method)."""
        return (
            self.check_delegation(
                user_id=user_id,
                permission_id=permission_id,
                org_id=org_id,
                workspace_id=workspace_id,
                resource_type=resource_type,
                resource_id=resource_id,
            )
            is not None
        )

    def list_delegations_by_delegator(
        self,
        delegator_id: str,
        include_expired: bool = False,
    ) -> list[PermissionDelegation]:
        """List all delegations made by a user."""
        delegation_ids = self._by_delegator.get(delegator_id, set())
        delegations = []

        for d_id in delegation_ids:
            d = self._delegations.get(d_id)
            if d and (include_expired or d.is_valid):
                delegations.append(d)

        return sorted(delegations, key=lambda d: d.created_at, reverse=True)

    def list_delegations_for_delegatee(
        self,
        delegatee_id: str,
        include_expired: bool = False,
    ) -> list[PermissionDelegation]:
        """List all delegations granted to a user."""
        delegation_ids = self._by_delegatee.get(delegatee_id, set())
        delegations = []

        for d_id in delegation_ids:
            d = self._delegations.get(d_id)
            if d and (include_expired or d.is_valid):
                delegations.append(d)

        return sorted(delegations, key=lambda d: d.created_at, reverse=True)

    def get_delegation(self, delegation_id: str) -> PermissionDelegation | None:
        """Get a delegation by ID."""
        return self._delegations.get(delegation_id)

    def cleanup_expired(self) -> int:
        """Remove expired delegations from storage."""
        expired_ids = [d_id for d_id, d in self._delegations.items() if d.is_expired]

        for d_id in expired_ids:
            d = self._delegations.pop(d_id, None)
            if d:
                self._by_delegator.get(d.delegator_id, set()).discard(d_id)
                self._by_delegatee.get(d.delegatee_id, set()).discard(d_id)

        logger.info(f"Cleaned up {len(expired_ids)} expired delegations")
        return len(expired_ids)

    def get_stats(self) -> dict[str, Any]:
        """Get delegation statistics."""
        active = sum(1 for d in self._delegations.values() if d.is_valid)
        expired = sum(1 for d in self._delegations.values() if d.is_expired)
        revoked = sum(1 for d in self._delegations.values() if d.status == DelegationStatus.REVOKED)

        return {
            "total_delegations": len(self._delegations),
            "active_delegations": active,
            "expired_delegations": expired,
            "revoked_delegations": revoked,
            "unique_delegators": len(self._by_delegator),
            "unique_delegatees": len(self._by_delegatee),
            "max_chain_depth": self._max_chain_depth,
            "max_duration_hours": self._max_duration.total_seconds() / 3600,
        }

    def _store_delegation(self, delegation: PermissionDelegation) -> None:
        """Store a delegation and update indexes."""
        self._delegations[delegation.id] = delegation

        if delegation.delegator_id not in self._by_delegator:
            self._by_delegator[delegation.delegator_id] = set()
        self._by_delegator[delegation.delegator_id].add(delegation.id)

        if delegation.delegatee_id not in self._by_delegatee:
            self._by_delegatee[delegation.delegatee_id] = set()
        self._by_delegatee[delegation.delegatee_id].add(delegation.id)


# Global delegation manager instance
_delegation_manager: DelegationManager | None = None


def get_delegation_manager() -> DelegationManager:
    """Get or create the global delegation manager."""
    global _delegation_manager
    if _delegation_manager is None:
        _delegation_manager = DelegationManager()
    return _delegation_manager


def set_delegation_manager(manager: DelegationManager | None) -> None:
    """Set the global delegation manager."""
    global _delegation_manager
    _delegation_manager = manager


def delegate_permission(
    delegator_id: str,
    delegatee_id: str,
    permission_id: str,
    org_id: str | None = None,
    workspace_id: str | None = None,
    expires_at: datetime | None = None,
    reason: str = "",
    can_redelegate: bool = False,
) -> PermissionDelegation:
    """Convenience function to delegate a permission using the global manager."""
    return get_delegation_manager().delegate(
        delegator_id=delegator_id,
        delegatee_id=delegatee_id,
        permission_id=permission_id,
        org_id=org_id,
        workspace_id=workspace_id,
        expires_at=expires_at,
        reason=reason,
        can_redelegate=can_redelegate,
    )


def check_delegated_permission(
    user_id: str,
    permission_id: str,
    org_id: str | None = None,
    workspace_id: str | None = None,
) -> bool:
    """Convenience function to check a delegated permission."""
    return get_delegation_manager().has_delegated_permission(
        user_id=user_id,
        permission_id=permission_id,
        org_id=org_id,
        workspace_id=workspace_id,
    )


def revoke_delegation(delegation_id: str, revoked_by: str) -> bool:
    """Convenience function to revoke a delegation."""
    return get_delegation_manager().revoke(delegation_id, revoked_by)
