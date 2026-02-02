"""
Helper functions for permission creation.

Internal module - not exported from the package.
"""

from __future__ import annotations

from aragora.rbac.models import Action, Permission, ResourceType


def _permission(
    resource: ResourceType,
    action: Action,
    name: str = "",
    description: str = "",
) -> Permission:
    """Helper to create a permission with auto-generated ID."""
    key = f"{resource.value}.{action.value}"
    return Permission(
        id=key,  # Use key as ID for simplicity
        name=name or key.replace(".", " ").replace("_", " ").title(),
        resource=resource,
        action=action,
        description=description,
    )
