"""
Aragora RBAC (Role-Based Access Control) Module.

Provides enterprise-grade access control for the Aragora platform:
- Fine-grained permissions (50+ defined permissions)
- Role hierarchy (owner > admin > debate_creator > analyst > viewer)
- Permission decorators for handlers
- HTTP middleware for route protection
- Comprehensive audit logging

Quick Start:
    from aragora.rbac import (
        require_permission,
        require_role,
        AuthorizationContext,
        get_permission_checker,
    )

    # Use decorators on handlers
    @require_permission("debates.create")
    async def create_debate(context: AuthorizationContext, ...):
        ...

    @require_role("admin")
    async def admin_action(context: AuthorizationContext, ...):
        ...
"""

# Legacy types (for backward compatibility)
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

# Legacy enforcer (for backward compatibility)
from .enforcer import (
    RBACEnforcer,
    RBACConfig,
    PermissionCheckResult,
    PermissionDeniedException,
    get_rbac_enforcer,
)

# New models (enhanced types)
from .models import (
    ResourceType as ResourceTypeV2,
    Action as ActionV2,
    Permission as PermissionV2,
    Role as RoleV2,
    RoleAssignment as RoleAssignmentV2,
    APIKeyScope,
    AuthorizationContext,
    AuthorizationDecision,
)

# New defaults (comprehensive permission matrix)
from .defaults import (
    SYSTEM_PERMISSIONS,
    SYSTEM_ROLES as SYSTEM_ROLES_V2,
    ROLE_HIERARCHY,
    ROLE_TEMPLATES,
    get_permission,
    get_role,
    get_role_permissions,
    create_custom_role,
    PERM_DEBATE_CREATE,
    PERM_DEBATE_READ,
    PERM_DEBATE_RUN,
    PERM_ADMIN_ALL,
)

# New checker
from .checker import (
    PermissionChecker,
    get_permission_checker,
    set_permission_checker,
    check_permission,
    has_permission,
)

# Decorators for handlers
from .decorators import (
    PermissionDeniedError,
    RoleRequiredError,
    require_permission,
    require_role,
    require_owner,
    require_admin,
    require_org_access,
    require_self_or_admin,
    with_permission_context,
)

# Middleware for HTTP routes
from .middleware import (
    RBACMiddleware,
    RBACMiddlewareConfig,
    RoutePermission,
    DEFAULT_ROUTE_PERMISSIONS,
    get_middleware,
    set_middleware,
    check_route_access,
    create_permission_handler,
)

# Audit logging
from .audit import (
    AuditEventType,
    AuditEvent,
    AuthorizationAuditor,
    get_auditor,
    set_auditor,
    log_permission_check,
)

# Distributed cache (for horizontal scaling)
from .cache import (
    RBACCacheConfig,
    RBACDistributedCache,
    CacheStats,
    get_rbac_cache,
    set_rbac_cache,
    reset_rbac_cache,
)

# Resource-level permissions (fine-grained RBAC)
from .resource_permissions import (
    ResourcePermission,
    ResourcePermissionStore,
    ResourcePermissionBackend,
    get_resource_permission_store,
    set_resource_permission_store,
    grant_resource_permission,
    revoke_resource_permission,
    check_resource_permission as check_resource_permission_func,
)

# Permission delegation (enterprise RBAC v2)
from .delegation import (
    DelegationStatus,
    DelegationConstraint,
    PermissionDelegation,
    DelegationManager,
    get_delegation_manager,
    set_delegation_manager,
    delegate_permission,
    check_delegated_permission,
    revoke_delegation,
)

# Profile system (lite/standard/enterprise)
from .profiles import (
    RBACProfile,
    ProfileConfig,
    PROFILE_CONFIGS,
    get_profile_config,
    get_profile_roles,
    get_default_role,
    get_available_roles_for_assignment,
    can_upgrade_profile,
    get_migration_plan,
    get_lite_role_summary,
)


__all__ = [
    # Legacy Types (backward compatible)
    "ResourceType",
    "Action",
    "Scope",
    "Permission",
    "Role",
    "RoleAssignment",
    "IsolationContext",
    "SYSTEM_ROLES",
    # Legacy Enforcer
    "RBACEnforcer",
    "RBACConfig",
    "PermissionCheckResult",
    "PermissionDeniedException",
    "get_rbac_enforcer",
    # V2 Models
    "ResourceTypeV2",
    "ActionV2",
    "PermissionV2",
    "RoleV2",
    "RoleAssignmentV2",
    "APIKeyScope",
    "AuthorizationContext",
    "AuthorizationDecision",
    # Defaults & Permission Matrix
    "SYSTEM_PERMISSIONS",
    "SYSTEM_ROLES_V2",
    "ROLE_HIERARCHY",
    "ROLE_TEMPLATES",
    "get_permission",
    "get_role",
    "get_role_permissions",
    "create_custom_role",
    "PERM_DEBATE_CREATE",
    "PERM_DEBATE_READ",
    "PERM_DEBATE_RUN",
    "PERM_ADMIN_ALL",
    # Checker
    "PermissionChecker",
    "get_permission_checker",
    "set_permission_checker",
    "check_permission",
    "has_permission",
    # Decorators
    "PermissionDeniedError",
    "RoleRequiredError",
    "require_permission",
    "require_role",
    "require_owner",
    "require_admin",
    "require_org_access",
    "require_self_or_admin",
    "with_permission_context",
    # Middleware
    "RBACMiddleware",
    "RBACMiddlewareConfig",
    "RoutePermission",
    "DEFAULT_ROUTE_PERMISSIONS",
    "get_middleware",
    "set_middleware",
    "check_route_access",
    "create_permission_handler",
    # Audit
    "AuditEventType",
    "AuditEvent",
    "AuthorizationAuditor",
    "get_auditor",
    "set_auditor",
    "log_permission_check",
    # Distributed Cache
    "RBACCacheConfig",
    "RBACDistributedCache",
    "CacheStats",
    "get_rbac_cache",
    "set_rbac_cache",
    "reset_rbac_cache",
    # Resource-Level Permissions (Fine-Grained RBAC)
    "ResourcePermission",
    "ResourcePermissionStore",
    "ResourcePermissionBackend",
    "get_resource_permission_store",
    "set_resource_permission_store",
    "grant_resource_permission",
    "revoke_resource_permission",
    "check_resource_permission_func",
    # Permission Delegation (Enterprise RBAC v2)
    "DelegationStatus",
    "DelegationConstraint",
    "PermissionDelegation",
    "DelegationManager",
    "get_delegation_manager",
    "set_delegation_manager",
    "delegate_permission",
    "check_delegated_permission",
    "revoke_delegation",
    # Profile System (Lite/Standard/Enterprise)
    "RBACProfile",
    "ProfileConfig",
    "PROFILE_CONFIGS",
    "get_profile_config",
    "get_profile_roles",
    "get_default_role",
    "get_available_roles_for_assignment",
    "can_upgrade_profile",
    "get_migration_plan",
    "get_lite_role_summary",
]
