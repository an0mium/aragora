"""
RBAC Module.

Role-based access control for the Aragora enterprise control plane.
Supports hierarchical permissions: global -> organization -> workspace -> resource.
"""

from .enforcer import (
    RBACEnforcer,
    RBACConfig,
    PermissionCheckResult,
    PermissionDeniedException,
    get_rbac_enforcer,
)
from .types import (
    ResourceType,
    Action,
    Scope,
    Permission,
    Role,
    RoleAssignment,
    IsolationContext,
    SYSTEM_ROLES,
)

__all__ = [
    # Enforcer
    "RBACEnforcer",
    "RBACConfig",
    "PermissionCheckResult",
    "PermissionDeniedException",
    "get_rbac_enforcer",
    # Types
    "ResourceType",
    "Action",
    "Scope",
    "Permission",
    "Role",
    "RoleAssignment",
    "IsolationContext",
    "SYSTEM_ROLES",
]
