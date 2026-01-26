"""
Auth Requirements Manifest.

Declarative specification of authentication requirements for API endpoints.
Used by contract tests to ensure OpenAPI spec matches actual handler protection.

This manifest serves as the source of truth for:
1. Which endpoints require authentication
2. What level of access is needed (public, authenticated, admin)
3. Specific permission requirements for RBAC-protected endpoints
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class AuthLevel(Enum):
    """Authentication level for endpoints."""

    PUBLIC = "public"  # No authentication required
    AUTHENTICATED = "authenticated"  # Any valid auth (JWT, API key, bearer token)
    USER = "user"  # User-level JWT/API key authentication
    PERMISSION = "permission"  # Specific RBAC permission required
    ADMIN = "admin"  # Admin role required
    OWNER = "owner"  # Owner role required


@dataclass
class EndpointAuth:
    """Authentication requirement for a single endpoint."""

    path: str
    method: str
    level: AuthLevel
    permission: Optional[str] = None  # e.g., "debates:create"
    description: str = ""
    # OpenAPI security requirement that should be present
    openapi_security: list[str] = field(default_factory=lambda: ["bearerAuth"])


# =============================================================================
# Public Endpoints - No authentication required
# =============================================================================

PUBLIC_ENDPOINTS = [
    # Health checks - must be accessible for load balancers
    EndpointAuth("/api/health", "get", AuthLevel.PUBLIC, description="Health check"),
    EndpointAuth("/api/healthz", "get", AuthLevel.PUBLIC, description="Kubernetes liveness"),
    EndpointAuth("/api/readyz", "get", AuthLevel.PUBLIC, description="Kubernetes readiness"),
    EndpointAuth("/api/metrics", "get", AuthLevel.PUBLIC, description="Prometheus metrics"),
    # OpenAPI spec - publicly accessible for SDK generation
    EndpointAuth("/api/openapi", "get", AuthLevel.PUBLIC, description="OpenAPI spec"),
    EndpointAuth("/api/openapi.yaml", "get", AuthLevel.PUBLIC, description="OpenAPI spec YAML"),
    # Authentication endpoints - must be accessible to authenticate
    EndpointAuth("/api/auth/login", "post", AuthLevel.PUBLIC, description="User login"),
    EndpointAuth("/api/auth/register", "post", AuthLevel.PUBLIC, description="User registration"),
    EndpointAuth("/api/auth/refresh", "post", AuthLevel.PUBLIC, description="Token refresh"),
    EndpointAuth(
        "/api/auth/forgot-password", "post", AuthLevel.PUBLIC, description="Password reset"
    ),
    # OAuth flows
    EndpointAuth("/api/auth/oauth/google", "get", AuthLevel.PUBLIC, description="Google OAuth"),
    EndpointAuth("/api/auth/oauth/github", "get", AuthLevel.PUBLIC, description="GitHub OAuth"),
    EndpointAuth("/api/auth/oauth/callback", "get", AuthLevel.PUBLIC, description="OAuth callback"),
    # Public API info
    EndpointAuth("/api/modes", "get", AuthLevel.PUBLIC, description="List operational modes"),
    # Nomic health - public for monitoring
    EndpointAuth("/api/nomic/health", "get", AuthLevel.PUBLIC, description="Nomic loop health"),
]

# =============================================================================
# Authenticated Endpoints - Any valid authentication
# =============================================================================

AUTHENTICATED_ENDPOINTS = [
    # Debate operations
    EndpointAuth("/api/debates", "get", AuthLevel.AUTHENTICATED, description="List debates"),
    EndpointAuth("/api/debates", "post", AuthLevel.AUTHENTICATED, description="Create debate"),
    EndpointAuth("/api/debates/{id}", "get", AuthLevel.AUTHENTICATED, description="Get debate"),
    EndpointAuth(
        "/api/debates/{id}", "delete", AuthLevel.AUTHENTICATED, description="Delete debate"
    ),
    EndpointAuth(
        "/api/debates/{id}/result", "get", AuthLevel.AUTHENTICATED, description="Get result"
    ),
    EndpointAuth(
        "/api/debates/{id}/resume", "post", AuthLevel.AUTHENTICATED, description="Resume debate"
    ),
    EndpointAuth(
        "/api/debates/{id}/pause", "post", AuthLevel.AUTHENTICATED, description="Pause debate"
    ),
    EndpointAuth(
        "/api/debates/{id}/cancel", "post", AuthLevel.AUTHENTICATED, description="Cancel debate"
    ),
    EndpointAuth(
        "/api/debates/{id}/export", "get", AuthLevel.AUTHENTICATED, description="Export debate"
    ),
    # Agent operations
    EndpointAuth("/api/agents", "get", AuthLevel.AUTHENTICATED, description="List agents"),
    EndpointAuth("/api/agent/{name}", "get", AuthLevel.AUTHENTICATED, description="Get agent"),
    EndpointAuth(
        "/api/agent/{name}/stats", "get", AuthLevel.AUTHENTICATED, description="Agent stats"
    ),
    # Memory operations
    EndpointAuth("/api/memory/query", "post", AuthLevel.AUTHENTICATED, description="Query memory"),
    EndpointAuth("/api/memory/store", "post", AuthLevel.AUTHENTICATED, description="Store memory"),
    # Consensus operations
    EndpointAuth(
        "/api/consensus/similar", "get", AuthLevel.AUTHENTICATED, description="Find similar"
    ),
    EndpointAuth(
        "/api/consensus/settled", "get", AuthLevel.AUTHENTICATED, description="Settled questions"
    ),
    EndpointAuth(
        "/api/consensus/stats", "get", AuthLevel.AUTHENTICATED, description="Consensus stats"
    ),
    # Knowledge operations
    EndpointAuth(
        "/api/knowledge/query", "post", AuthLevel.AUTHENTICATED, description="Query knowledge"
    ),
    EndpointAuth(
        "/api/knowledge/upload", "post", AuthLevel.AUTHENTICATED, description="Upload document"
    ),
    # User profile
    EndpointAuth("/api/user/profile", "get", AuthLevel.AUTHENTICATED, description="Get profile"),
    EndpointAuth("/api/user/profile", "put", AuthLevel.AUTHENTICATED, description="Update profile"),
    EndpointAuth("/api/user/settings", "get", AuthLevel.AUTHENTICATED, description="Get settings"),
    EndpointAuth(
        "/api/user/settings", "put", AuthLevel.AUTHENTICATED, description="Update settings"
    ),
]

# =============================================================================
# Permission-Protected Endpoints - Specific RBAC permissions required
# =============================================================================

PERMISSION_ENDPOINTS = [
    # Plugin management
    EndpointAuth(
        "/api/plugins",
        "get",
        AuthLevel.PERMISSION,
        permission="plugins:read",
        description="List plugins",
    ),
    EndpointAuth(
        "/api/plugins",
        "post",
        AuthLevel.PERMISSION,
        permission="plugins:install",
        description="Install plugin",
    ),
    EndpointAuth(
        "/api/plugins/{id}",
        "delete",
        AuthLevel.PERMISSION,
        permission="plugins:uninstall",
        description="Uninstall plugin",
    ),
    EndpointAuth(
        "/api/plugins/{id}/execute",
        "post",
        AuthLevel.PERMISSION,
        permission="plugins:execute",
        description="Execute plugin",
    ),
    # Training data
    EndpointAuth(
        "/api/training/data",
        "get",
        AuthLevel.PERMISSION,
        permission="training:read",
        description="Get training data",
    ),
    EndpointAuth(
        "/api/training/data",
        "post",
        AuthLevel.PERMISSION,
        permission="training:create",
        description="Create training data",
    ),
    EndpointAuth(
        "/api/training/export",
        "get",
        AuthLevel.PERMISSION,
        permission="training:export",
        description="Export training data",
    ),
    # Connector management
    EndpointAuth(
        "/api/connectors",
        "get",
        AuthLevel.PERMISSION,
        permission="connectors:read",
        description="List connectors",
    ),
    EndpointAuth(
        "/api/connectors",
        "post",
        AuthLevel.PERMISSION,
        permission="connectors:create",
        description="Create connector",
    ),
    EndpointAuth(
        "/api/connectors/{id}",
        "delete",
        AuthLevel.PERMISSION,
        permission="connectors:delete",
        description="Delete connector",
    ),
    # ML operations
    EndpointAuth(
        "/api/ml/models",
        "get",
        AuthLevel.PERMISSION,
        permission="ml:read",
        description="List ML models",
    ),
    EndpointAuth(
        "/api/ml/train",
        "post",
        AuthLevel.PERMISSION,
        permission="ml:train",
        description="Train model",
    ),
    EndpointAuth(
        "/api/ml/deploy",
        "post",
        AuthLevel.PERMISSION,
        permission="ml:deploy",
        description="Deploy model",
    ),
    # API key management
    EndpointAuth(
        "/api/apikeys",
        "get",
        AuthLevel.PERMISSION,
        permission="apikeys:read",
        description="List API keys",
    ),
    EndpointAuth(
        "/api/apikeys",
        "post",
        AuthLevel.PERMISSION,
        permission="apikeys:create",
        description="Create API key",
    ),
    EndpointAuth(
        "/api/apikeys/{id}",
        "delete",
        AuthLevel.PERMISSION,
        permission="apikeys:delete",
        description="Revoke API key",
    ),
]

# =============================================================================
# Admin Endpoints - Admin role required
# =============================================================================

ADMIN_ENDPOINTS = [
    # System administration
    EndpointAuth("/api/health/detailed", "get", AuthLevel.ADMIN, description="Detailed health"),
    EndpointAuth("/api/system/maintenance", "get", AuthLevel.ADMIN, description="Run maintenance"),
    EndpointAuth("/api/system/config", "get", AuthLevel.ADMIN, description="Get system config"),
    EndpointAuth("/api/system/config", "put", AuthLevel.ADMIN, description="Update system config"),
    # User administration
    EndpointAuth("/api/admin/users", "get", AuthLevel.ADMIN, description="List all users"),
    EndpointAuth("/api/admin/users/{id}", "get", AuthLevel.ADMIN, description="Get user"),
    EndpointAuth("/api/admin/users/{id}", "put", AuthLevel.ADMIN, description="Update user"),
    EndpointAuth("/api/admin/users/{id}", "delete", AuthLevel.ADMIN, description="Delete user"),
    # Audit logs
    EndpointAuth("/api/admin/audit", "get", AuthLevel.ADMIN, description="Get audit logs"),
    EndpointAuth(
        "/api/admin/audit/export", "get", AuthLevel.ADMIN, description="Export audit logs"
    ),
    # Metrics and monitoring
    EndpointAuth("/api/admin/metrics", "get", AuthLevel.ADMIN, description="Admin metrics"),
    EndpointAuth(
        "/api/admin/metrics/detailed", "get", AuthLevel.ADMIN, description="Detailed metrics"
    ),
    # Nomic loop management
    EndpointAuth("/api/nomic/state", "get", AuthLevel.ADMIN, description="Nomic state"),
    EndpointAuth("/api/nomic/log", "get", AuthLevel.ADMIN, description="Nomic logs"),
    EndpointAuth("/api/nomic/risk-register", "get", AuthLevel.ADMIN, description="Risk register"),
    # Control plane
    EndpointAuth(
        "/api/control-plane/agents",
        "get",
        AuthLevel.ADMIN,
        description="Control plane agents",
    ),
    EndpointAuth(
        "/api/control-plane/tasks",
        "get",
        AuthLevel.ADMIN,
        description="Control plane tasks",
    ),
    EndpointAuth(
        "/api/control-plane/policies",
        "get",
        AuthLevel.ADMIN,
        description="Control plane policies",
    ),
    # Billing (admin view)
    EndpointAuth(
        "/api/admin/billing/overview", "get", AuthLevel.ADMIN, description="Billing overview"
    ),
    EndpointAuth("/api/admin/billing/usage", "get", AuthLevel.ADMIN, description="Usage report"),
]

# =============================================================================
# Owner Endpoints - Owner role required
# =============================================================================

OWNER_ENDPOINTS = [
    # Organization management
    EndpointAuth("/api/org/settings", "put", AuthLevel.OWNER, description="Update org settings"),
    EndpointAuth("/api/org/delete", "delete", AuthLevel.OWNER, description="Delete organization"),
    EndpointAuth("/api/org/billing", "get", AuthLevel.OWNER, description="Get billing"),
    EndpointAuth("/api/org/billing", "put", AuthLevel.OWNER, description="Update billing"),
    # Dangerous admin operations
    EndpointAuth(
        "/api/admin/system/reset",
        "post",
        AuthLevel.OWNER,
        description="Reset system (dangerous)",
    ),
    EndpointAuth(
        "/api/admin/system/backup",
        "post",
        AuthLevel.OWNER,
        description="Trigger backup",
    ),
    EndpointAuth(
        "/api/admin/system/restore",
        "post",
        AuthLevel.OWNER,
        description="Restore from backup",
    ),
]


# =============================================================================
# Aggregated Requirements
# =============================================================================


def get_all_requirements() -> list[EndpointAuth]:
    """Get all endpoint authentication requirements."""
    return (
        PUBLIC_ENDPOINTS
        + AUTHENTICATED_ENDPOINTS
        + PERMISSION_ENDPOINTS
        + ADMIN_ENDPOINTS
        + OWNER_ENDPOINTS
    )


def get_requirements_by_path(path: str) -> list[EndpointAuth]:
    """Get all requirements for a specific path (all methods)."""
    return [req for req in get_all_requirements() if req.path == path]


def get_requirement(path: str, method: str) -> Optional[EndpointAuth]:
    """Get the requirement for a specific path and method."""
    method = method.lower()
    for req in get_all_requirements():
        if req.path == path and req.method.lower() == method:
            return req
    return None


def get_protected_prefixes() -> list[str]:
    """Get path prefixes that should be protected by default."""
    return [
        "/api/debates",
        "/api/agents",
        "/api/agent/",
        "/api/memory",
        "/api/consensus",
        "/api/knowledge",
        "/api/documents",
        "/api/plugins",
        "/api/training",
        "/api/connectors",
        "/api/ml",
        "/api/user",
        "/api/org",
        "/api/admin",
        "/api/control-plane",
        "/api/apikeys",
    ]


def get_public_paths() -> set[str]:
    """Get paths that are explicitly public."""
    return {req.path for req in PUBLIC_ENDPOINTS}


def requires_auth(path: str, method: str = "get") -> bool:
    """Check if a path/method combination requires authentication."""
    req = get_requirement(path, method)
    if req:
        return req.level != AuthLevel.PUBLIC

    # Default: protected prefixes require auth
    for prefix in get_protected_prefixes():
        if path.startswith(prefix):
            return True

    return False


def get_required_permission(path: str, method: str = "get") -> Optional[str]:
    """Get the required permission for a path/method combination."""
    req = get_requirement(path, method)
    if req and req.permission:
        return req.permission
    return None


__all__ = [
    "AuthLevel",
    "EndpointAuth",
    "PUBLIC_ENDPOINTS",
    "AUTHENTICATED_ENDPOINTS",
    "PERMISSION_ENDPOINTS",
    "ADMIN_ENDPOINTS",
    "OWNER_ENDPOINTS",
    "get_all_requirements",
    "get_requirements_by_path",
    "get_requirement",
    "get_protected_prefixes",
    "get_public_paths",
    "requires_auth",
    "get_required_permission",
]
