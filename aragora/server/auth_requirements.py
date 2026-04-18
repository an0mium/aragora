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
import re


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
    permission: str | None = None  # e.g., "debates:create"
    description: str = ""
    # OpenAPI security requirement that should be present
    openapi_security: list[str] = field(default_factory=lambda: ["bearerAuth"])


INBOX_PERMISSION_ENDPOINTS = [
    EndpointAuth(
        "/api/v1/inbox/oauth/gmail",
        "get",
        AuthLevel.PERMISSION,
        permission="inbox:read",
        description="Get Gmail inbox OAuth URL",
    ),
    EndpointAuth(
        "/api/v1/inbox/oauth/outlook",
        "get",
        AuthLevel.PERMISSION,
        permission="inbox:read",
        description="Get Outlook inbox OAuth URL",
    ),
    EndpointAuth(
        "/api/v1/inbox/connect",
        "post",
        AuthLevel.PERMISSION,
        permission="inbox:read",
        description="Connect a unified inbox account",
    ),
    EndpointAuth(
        "/api/v1/inbox/accounts",
        "get",
        AuthLevel.PERMISSION,
        permission="inbox:read",
        description="List unified inbox accounts",
    ),
    EndpointAuth(
        "/api/v1/inbox/accounts/{account_id}",
        "delete",
        AuthLevel.PERMISSION,
        permission="inbox:read",
        description="Disconnect a unified inbox account",
    ),
    EndpointAuth(
        "/api/v1/inbox/messages",
        "get",
        AuthLevel.PERMISSION,
        permission="inbox:read",
        description="List unified inbox messages",
    ),
    EndpointAuth(
        "/api/v1/inbox/messages/{message_id}",
        "get",
        AuthLevel.PERMISSION,
        permission="inbox:read",
        description="Get unified inbox message details",
    ),
    EndpointAuth(
        "/api/v1/inbox/messages/{message_id}/debate",
        "post",
        AuthLevel.PERMISSION,
        permission="inbox:read",
        description="Create a trust-wedge debate for an inbox message",
    ),
    EndpointAuth(
        "/api/v1/inbox/triage",
        "post",
        AuthLevel.PERMISSION,
        permission="inbox:read",
        description="Run unified inbox triage",
    ),
    EndpointAuth(
        "/api/v1/inbox/bulk-action",
        "post",
        AuthLevel.PERMISSION,
        permission="inbox:read",
        description="Execute a unified inbox bulk action",
    ),
    EndpointAuth(
        "/api/v1/inbox/stats",
        "get",
        AuthLevel.PERMISSION,
        permission="inbox:read",
        description="Get unified inbox statistics",
    ),
    EndpointAuth(
        "/api/v1/inbox/trends",
        "get",
        AuthLevel.PERMISSION,
        permission="inbox:read",
        description="Get unified inbox trends",
    ),
    EndpointAuth(
        "/api/v1/inbox/command",
        "get",
        AuthLevel.PERMISSION,
        permission="inbox:read",
        description="Fetch the inbox command center view",
    ),
    EndpointAuth(
        "/api/v1/inbox/actions",
        "post",
        AuthLevel.PERMISSION,
        permission="inbox:write",
        description="Execute an inbox quick action",
    ),
    EndpointAuth(
        "/api/v1/inbox/bulk-actions",
        "post",
        AuthLevel.PERMISSION,
        permission="inbox:write",
        description="Execute an inbox bulk action by filter",
    ),
    EndpointAuth(
        "/api/v1/inbox/sender-profile",
        "get",
        AuthLevel.PERMISSION,
        permission="inbox:read",
        description="Get an inbox sender profile",
    ),
    EndpointAuth(
        "/api/v1/inbox/daily-digest",
        "get",
        AuthLevel.PERMISSION,
        permission="inbox:read",
        description="Get the inbox daily digest",
    ),
    EndpointAuth(
        "/api/v1/inbox/reprioritize",
        "post",
        AuthLevel.PERMISSION,
        permission="inbox:write",
        description="Trigger inbox reprioritization",
    ),
]

COORDINATION_PUBLIC_ENDPOINTS = [
    EndpointAuth(
        "/api/v1/coordination/health",
        "get",
        AuthLevel.PUBLIC,
        description="Cross-workspace coordination health",
    ),
]

COORDINATION_PERMISSION_ENDPOINTS = [
    EndpointAuth(
        "/api/v1/coordination/workspaces",
        "get",
        AuthLevel.PERMISSION,
        permission="coordination:read",
        description="List registered coordination workspaces",
    ),
    EndpointAuth(
        "/api/v1/coordination/workspaces",
        "post",
        AuthLevel.PERMISSION,
        permission="coordination:write",
        description="Register a coordination workspace",
    ),
    EndpointAuth(
        "/api/v1/coordination/workspaces/{workspace_id}",
        "delete",
        AuthLevel.PERMISSION,
        permission="coordination:delete",
        description="Unregister a coordination workspace",
    ),
    EndpointAuth(
        "/api/v1/coordination/federation",
        "get",
        AuthLevel.PERMISSION,
        permission="coordination:read",
        description="List coordination federation policies",
    ),
    EndpointAuth(
        "/api/v1/coordination/federation",
        "post",
        AuthLevel.PERMISSION,
        permission="coordination:write",
        description="Create a coordination federation policy",
    ),
    EndpointAuth(
        "/api/v1/coordination/execute",
        "post",
        AuthLevel.PERMISSION,
        permission="coordination:execute",
        description="Execute a cross-workspace coordination request",
    ),
    EndpointAuth(
        "/api/v1/coordination/executions",
        "get",
        AuthLevel.PERMISSION,
        permission="coordination:read",
        description="List pending coordination executions",
    ),
    EndpointAuth(
        "/api/v1/coordination/consent",
        "get",
        AuthLevel.PERMISSION,
        permission="coordination:read",
        description="List coordination consents",
    ),
    EndpointAuth(
        "/api/v1/coordination/consent",
        "post",
        AuthLevel.PERMISSION,
        permission="coordination:write",
        description="Grant coordination consent",
    ),
    EndpointAuth(
        "/api/v1/coordination/consent/{consent_id}",
        "delete",
        AuthLevel.PERMISSION,
        permission="coordination:delete",
        description="Revoke coordination consent",
    ),
    EndpointAuth(
        "/api/v1/coordination/approve/{request_id}",
        "post",
        AuthLevel.PERMISSION,
        permission="coordination:admin",
        description="Approve a pending coordination execution",
    ),
    EndpointAuth(
        "/api/v1/coordination/stats",
        "get",
        AuthLevel.PERMISSION,
        permission="coordination:read",
        description="Get coordination statistics",
    ),
]


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
    # Nomic - public for dashboard
    EndpointAuth("/api/nomic/health", "get", AuthLevel.PUBLIC, description="Nomic loop health"),
    EndpointAuth("/api/nomic/state", "get", AuthLevel.PUBLIC, description="Nomic state"),
    # Leaderboard - public dashboard data
    EndpointAuth("/api/leaderboard-view", "get", AuthLevel.PUBLIC, description="Leaderboard view"),
    # Breakpoints - public read-only
    EndpointAuth(
        "/api/breakpoints/pending",
        "get",
        AuthLevel.PUBLIC,
        description="Pending breakpoints",
    ),
    # Metrics - public dashboard monitoring (matches AUTH_EXEMPT_GET_PREFIXES)
    EndpointAuth("/api/metrics/health", "get", AuthLevel.PUBLIC, description="Metrics health"),
    EndpointAuth("/api/metrics/cache", "get", AuthLevel.PUBLIC, description="Cache metrics"),
    EndpointAuth("/api/metrics/system", "get", AuthLevel.PUBLIC, description="System metrics"),
    # Consensus - public read-only dashboard data (matches AUTH_EXEMPT_GET_PREFIXES)
    EndpointAuth(
        "/api/consensus/contrarian-views",
        "get",
        AuthLevel.PUBLIC,
        description="Contrarian views",
    ),
    EndpointAuth(
        "/api/consensus/risk-warnings",
        "get",
        AuthLevel.PUBLIC,
        description="Risk warnings",
    ),
    EndpointAuth("/api/consensus/dissents", "get", AuthLevel.PUBLIC, description="Recent dissents"),
    EndpointAuth(
        "/api/consensus/similar", "get", AuthLevel.PUBLIC, description="Find similar debates"
    ),
    EndpointAuth("/api/consensus/settled", "get", AuthLevel.PUBLIC, description="Settled topics"),
    EndpointAuth("/api/consensus/stats", "get", AuthLevel.PUBLIC, description="Consensus stats"),
    # Playground - public demo endpoints (rate-limited, mock agents only)
    EndpointAuth(
        "/api/v1/playground/debate",
        "post",
        AuthLevel.PUBLIC,
        description="Public playground debate",
    ),
    EndpointAuth(
        "/api/v1/playground/debate/",
        "post",
        AuthLevel.PUBLIC,
        description="Public playground debate",
    ),
    EndpointAuth(
        "/api/v1/playground/assess",
        "post",
        AuthLevel.PUBLIC,
        description="Landing question ambiguity assessment",
    ),
    EndpointAuth(
        "/api/v1/playground/landing/events",
        "post",
        AuthLevel.PUBLIC,
        description="Public landing telemetry",
    ),
    EndpointAuth(
        "/api/v1/playground/landing/feedback",
        "post",
        AuthLevel.PUBLIC,
        description="Public landing feedback capture",
    ),
    EndpointAuth(
        "/api/v1/playground/status",
        "get",
        AuthLevel.PUBLIC,
        description="Playground health check",
    ),
    # Spectate - public read-only live observation (debate IDs redacted for unauthenticated)
    EndpointAuth(
        "/api/v1/spectate/recent",
        "get",
        AuthLevel.PUBLIC,
        description="Recent spectate events",
    ),
    EndpointAuth(
        "/api/v1/spectate/status",
        "get",
        AuthLevel.PUBLIC,
        description="Spectate bridge status (redacted for unauthenticated)",
    ),
    EndpointAuth(
        "/api/v1/spectate/stream",
        "get",
        AuthLevel.PUBLIC,
        description="Spectate SSE snapshot / JSON preview",
    ),
    # Onboarding - public for first-time users
    EndpointAuth(
        "/api/v1/onboarding/templates",
        "get",
        AuthLevel.PUBLIC,
        description="Onboarding starter templates",
    ),
    # Public surface discovery
    EndpointAuth(
        "/api/v1/public/surfaces",
        "get",
        AuthLevel.PUBLIC,
        description="List available public API surfaces",
    ),
    EndpointAuth(
        "/api/v2/receipts/share/{token}",
        "get",
        AuthLevel.PUBLIC,
        description="Access a shared receipt via public token",
    ),
    *COORDINATION_PUBLIC_ENDPOINTS,
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
    # Note: Consensus read endpoints moved to PUBLIC_ENDPOINTS
    # (they are exempt via AUTH_EXEMPT_GET_PREFIXES in auth_checks.py)
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
    *INBOX_PERMISSION_ENDPOINTS,
    *COORDINATION_PERMISSION_ENDPOINTS,
    # Plugin management
    EndpointAuth(
        "/api/plugins",
        "get",
        AuthLevel.PERMISSION,
        permission="plugins:read",
        description="List plugins",
    ),
    EndpointAuth(
        "/api/v2/receipts/{receipt_id}/share",
        "post",
        AuthLevel.PERMISSION,
        permission="receipts:share",
        description="Create receipt share link",
    ),
    EndpointAuth(
        "/api/v2/receipts/{receipt_id}/send-to-channel",
        "post",
        AuthLevel.PERMISSION,
        permission="receipts:share",
        description="Send receipt to an external channel",
    ),
    EndpointAuth(
        "/api/settlements",
        "get",
        AuthLevel.PERMISSION,
        permission="settlements:read",
        description="List pending settlements",
    ),
    EndpointAuth(
        "/api/settlements/history",
        "get",
        AuthLevel.PERMISSION,
        permission="settlements:read",
        description="Get settlement history",
    ),
    EndpointAuth(
        "/api/settlements/summary",
        "get",
        AuthLevel.PERMISSION,
        permission="settlements:read",
        description="Get settlement summary",
    ),
    EndpointAuth(
        "/api/settlements/{id}",
        "get",
        AuthLevel.PERMISSION,
        permission="settlements:read",
        description="Get settlement details",
    ),
    EndpointAuth(
        "/api/settlements/agent/{agent}/accuracy",
        "get",
        AuthLevel.PERMISSION,
        permission="settlements:read",
        description="Get settlement accuracy for an agent",
    ),
    EndpointAuth(
        "/api/settlements/{id}/settle",
        "post",
        AuthLevel.PERMISSION,
        permission="settlements:write",
        description="Settle a claim",
    ),
    EndpointAuth(
        "/api/settlements/batch",
        "post",
        AuthLevel.PERMISSION,
        permission="settlements:write",
        description="Settle claims in batch",
    ),
    EndpointAuth(
        "/api/v1/settlements",
        "get",
        AuthLevel.PERMISSION,
        permission="settlements:read",
        description="List pending settlements",
    ),
    EndpointAuth(
        "/api/v1/settlements/history",
        "get",
        AuthLevel.PERMISSION,
        permission="settlements:read",
        description="Get settlement history",
    ),
    EndpointAuth(
        "/api/v1/settlements/summary",
        "get",
        AuthLevel.PERMISSION,
        permission="settlements:read",
        description="Get settlement summary",
    ),
    EndpointAuth(
        "/api/v1/settlements/{id}",
        "get",
        AuthLevel.PERMISSION,
        permission="settlements:read",
        description="Get settlement details",
    ),
    EndpointAuth(
        "/api/v1/settlements/agent/{agent}/accuracy",
        "get",
        AuthLevel.PERMISSION,
        permission="settlements:read",
        description="Get settlement accuracy for an agent",
    ),
    EndpointAuth(
        "/api/v1/settlements/{id}/settle",
        "post",
        AuthLevel.PERMISSION,
        permission="settlements:write",
        description="Settle a claim",
    ),
    EndpointAuth(
        "/api/v1/settlements/batch",
        "post",
        AuthLevel.PERMISSION,
        permission="settlements:write",
        description="Settle claims in batch",
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
    # /api/nomic/state moved to PUBLIC_ENDPOINTS for dashboard access
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
    return [req for req in get_all_requirements() if _path_matches(req.path, path)]


def _normalize_path(path: str) -> str:
    """Normalize paths before manifest matching."""
    if not path:
        return "/"
    normalized = path.rstrip("/")
    return normalized or "/"


def _path_matches(template: str, path: str) -> bool:
    """Match concrete API paths against manifest templates with path params."""
    normalized_template = _normalize_path(template)
    normalized_path = _normalize_path(path)
    if normalized_template == normalized_path:
        return True

    pattern = re.sub(r"\{[^/]+\}", r"[^/]+", normalized_template)
    return re.fullmatch(pattern, normalized_path) is not None


def get_requirement(path: str, method: str) -> EndpointAuth | None:
    """Get the requirement for a specific path and method."""
    method = method.lower()
    for req in get_all_requirements():
        if req.method.lower() == method and _path_matches(req.path, path):
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
        "/api/settlements",
        "/api/v1/settlements",
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


def get_required_permission(path: str, method: str = "get") -> str | None:
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
