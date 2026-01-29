"""
RBAC Helper Functions.

Provides utility functions for permission and role management.
"""

from __future__ import annotations

from aragora.rbac.models import Permission, Role

from .registry import SYSTEM_PERMISSIONS
from .roles import ROLE_HIERARCHY, SYSTEM_ROLES
from .permissions import (
    PERM_AGENT_CREATE,
    PERM_AGENT_UPDATE,
    PERM_CONNECTOR_CREATE,
    PERM_TRAINING_CREATE,
    PERM_DEBATE_CREATE,
    PERM_DEBATE_RUN,
    PERM_USER_READ,
    PERM_ORG_AUDIT,
)


def get_permission(key: str) -> Permission | None:
    """Get a permission by its key."""
    return SYSTEM_PERMISSIONS.get(key)


def get_role(name: str) -> Role | None:
    """Get a role by its name."""
    return SYSTEM_ROLES.get(name)


def get_role_permissions(role_name: str, include_inherited: bool = True) -> set[str]:
    """
    Get all permissions for a role, optionally including inherited permissions.

    Args:
        role_name: Name of the role
        include_inherited: Whether to include permissions from parent roles

    Returns:
        Set of permission keys
    """
    role = get_role(role_name)
    if not role:
        return set()

    permissions = set(role.permissions)

    if include_inherited:
        for parent_name in ROLE_HIERARCHY.get(role_name, []):
            permissions |= get_role_permissions(parent_name, include_inherited=True)

    return permissions


def create_custom_role(
    name: str,
    display_name: str,
    description: str,
    permission_keys: set[str],
    org_id: str,
    base_role: str | None = None,
) -> Role:
    """
    Create a custom role for an organization.

    Args:
        name: Role name (must be unique within org)
        display_name: Human-readable name
        description: Role description
        permission_keys: Set of permission keys to grant
        org_id: Organization ID
        base_role: Optional base role to inherit from

    Returns:
        New custom Role instance
    """
    # Start with base role permissions if specified
    permissions = set()
    parent_roles = []

    if base_role:
        base = get_role(base_role)
        if base:
            permissions = get_role_permissions(base_role)
            parent_roles = [base_role]

    # Add specified permissions
    permissions |= permission_keys

    # Validate all permissions exist
    for key in permissions:
        if key not in SYSTEM_PERMISSIONS and not key.endswith(".*"):
            raise ValueError(f"Unknown permission: {key}")

    return Role(
        id=f"{org_id}:{name}",
        name=name,
        display_name=display_name,
        description=description,
        permissions=permissions,
        parent_roles=parent_roles,
        is_system=False,
        is_custom=True,
        org_id=org_id,
        priority=45,  # Between member and debate_creator
    )


# Predefined role templates for quick setup
ROLE_TEMPLATES = {
    "engineering": {
        "base": "debate_creator",
        "add": {PERM_AGENT_CREATE.key, PERM_AGENT_UPDATE.key, PERM_CONNECTOR_CREATE.key},
        "description": "Engineering team with agent management",
    },
    "research": {
        "base": "analyst",
        "add": {PERM_TRAINING_CREATE.key, PERM_DEBATE_CREATE.key, PERM_DEBATE_RUN.key},
        "description": "Research team with training data access",
    },
    "support": {
        "base": "viewer",
        "add": {PERM_USER_READ.key, PERM_ORG_AUDIT.key},
        "description": "Support team with user visibility",
    },
    "external": {
        "base": "viewer",
        "add": set(),
        "description": "External collaborators with minimal access",
    },
}
