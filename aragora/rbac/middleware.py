"""
RBAC Middleware - HTTP middleware for enforcing access control.

Provides middleware that integrates with the server to enforce
permission checks on all requests automatically.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Pattern

from .checker import PermissionChecker, get_permission_checker
from .models import AuthorizationContext

logger = logging.getLogger(__name__)


@dataclass
class RoutePermission:
    """
    Permission requirement for a route.

    Attributes:
        pattern: Regex pattern to match route
        method: HTTP method (GET, POST, etc.) or * for all
        permission_key: Required permission
        resource_id_group: Regex group index for resource ID extraction
        allow_unauthenticated: Allow requests without authentication
    """

    pattern: str | Pattern[str]
    method: str
    permission_key: str
    resource_id_group: int | None = None
    allow_unauthenticated: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.pattern, str):
            self.pattern = re.compile(self.pattern)

    def matches(self, path: str, method: str) -> tuple[bool, str | None]:
        """
        Check if this rule matches the request.

        Returns:
            Tuple of (matches, resource_id or None)
        """
        if self.method != "*" and self.method.upper() != method.upper():
            return False, None

        # Pattern is always compiled in __post_init__
        assert isinstance(self.pattern, Pattern), "pattern must be compiled"
        match = self.pattern.match(path)
        if not match:
            return False, None

        resource_id = None
        if self.resource_id_group is not None:
            try:
                resource_id = match.group(self.resource_id_group)
            except IndexError:
                pass

        return True, resource_id


@dataclass
class RBACMiddlewareConfig:
    """
    Configuration for RBAC middleware.

    Attributes:
        route_permissions: List of route permission rules
        default_authenticated: Require authentication by default
        bypass_paths: Paths that bypass all permission checks
        bypass_methods: Methods that bypass checks (e.g., OPTIONS for CORS)
        permission_checker: Custom permission checker instance
    """

    route_permissions: list[RoutePermission] = field(default_factory=list)
    default_authenticated: bool = True
    bypass_paths: set[str] = field(
        default_factory=lambda: {
            "/health",
            "/healthz",
            "/ready",
            "/metrics",
            "/api/docs",
            "/openapi.json",
        }
    )
    bypass_methods: set[str] = field(default_factory=lambda: {"OPTIONS"})
    permission_checker: PermissionChecker | None = None


# Default route permission rules
DEFAULT_ROUTE_PERMISSIONS = [
    # Debates
    RoutePermission(r"^/api/debates?$", "POST", "debates.create"),
    RoutePermission(r"^/api/debates?$", "GET", "debates.read"),
    RoutePermission(r"^/api/debates?/([^/]+)$", "GET", "debates.read", 1),
    RoutePermission(r"^/api/debates?/([^/]+)$", "PUT", "debates.update", 1),
    RoutePermission(r"^/api/debates?/([^/]+)$", "PATCH", "debates.update", 1),
    RoutePermission(r"^/api/debates?/([^/]+)$", "DELETE", "debates.delete", 1),
    RoutePermission(r"^/api/debates?/([^/]+)/run$", "POST", "debates.run", 1),
    RoutePermission(r"^/api/debates?/([^/]+)/stop$", "POST", "debates.stop", 1),
    RoutePermission(r"^/api/debates?/([^/]+)/fork$", "POST", "debates.fork", 1),
    # Agents
    RoutePermission(r"^/api/agents?$", "GET", "agents.read"),
    RoutePermission(r"^/api/agents?$", "POST", "agents.create"),
    RoutePermission(r"^/api/agents?/([^/]+)$", "GET", "agents.read", 1),
    RoutePermission(r"^/api/agents?/([^/]+)$", "PUT", "agents.update", 1),
    RoutePermission(r"^/api/agents?/([^/]+)$", "DELETE", "agents.delete", 1),
    # Workflows
    RoutePermission(r"^/api/workflows?$", "GET", "workflows.read"),
    RoutePermission(r"^/api/workflows?$", "POST", "workflows.create"),
    RoutePermission(r"^/api/workflows?/([^/]+)$", "GET", "workflows.read", 1),
    RoutePermission(r"^/api/workflows?/([^/]+)$", "DELETE", "workflows.delete", 1),
    RoutePermission(r"^/api/workflows?/([^/]+)/execute$", "POST", "workflows.run", 1),
    # Memory
    RoutePermission(r"^/api/memory", "GET", "memory.read"),
    RoutePermission(r"^/api/memory", "POST", "memory.update"),
    RoutePermission(r"^/api/memory", "DELETE", "memory.delete"),
    # Analytics
    RoutePermission(r"^/api/analytics", "GET", "analytics.read"),
    RoutePermission(r"^/api/analytics/export", "POST", "analytics.export_data"),
    # Training
    RoutePermission(r"^/api/training", "GET", "training.read"),
    RoutePermission(r"^/api/training/export", "POST", "training.create"),
    # Evidence
    RoutePermission(r"^/api/evidence", "GET", "evidence.read"),
    RoutePermission(r"^/api/evidence", "POST", "evidence.create"),
    # Connectors
    RoutePermission(r"^/api/connectors?$", "GET", "connectors.read"),
    RoutePermission(r"^/api/connectors?$", "POST", "connectors.create"),
    RoutePermission(r"^/api/connectors?/([^/]+)$", "DELETE", "connectors.delete", 1),
    # Webhooks
    RoutePermission(r"^/api/webhooks?$", "GET", "webhooks.read"),
    RoutePermission(r"^/api/webhooks?$", "POST", "webhooks.create"),
    RoutePermission(r"^/api/webhooks?/([^/]+)$", "DELETE", "webhooks.delete", 1),
    # Checkpoints
    RoutePermission(r"^/api/checkpoints?$", "GET", "checkpoints.read"),
    RoutePermission(r"^/api/checkpoints?$", "POST", "checkpoints.create"),
    RoutePermission(r"^/api/checkpoints?/([^/]+)$", "DELETE", "checkpoints.delete", 1),
    # Admin routes - require admin permission
    RoutePermission(r"^/api/admin", "*", "admin.*"),
    # User management
    RoutePermission(r"^/api/users?$", "GET", "users.read"),
    RoutePermission(r"^/api/users?/invite$", "POST", "users.invite"),
    RoutePermission(r"^/api/users?/([^/]+)$", "DELETE", "users.remove", 1),
    RoutePermission(r"^/api/users?/([^/]+)/role$", "PUT", "users.change_role", 1),
    # Organization
    RoutePermission(r"^/api/org", "GET", "organization.read"),
    RoutePermission(r"^/api/org", "PUT", "organization.update"),
    RoutePermission(r"^/api/org", "PATCH", "organization.update"),
    RoutePermission(r"^/api/org/billing", "*", "organization.manage_billing"),
    RoutePermission(r"^/api/org/audit", "GET", "organization.view_audit"),
    RoutePermission(r"^/api/org/export", "POST", "organization.export_data"),
    # API keys
    RoutePermission(r"^/api/keys?$", "GET", "api.generate_key"),
    RoutePermission(r"^/api/keys?$", "POST", "api.generate_key"),
    RoutePermission(r"^/api/keys?/([^/]+)$", "DELETE", "api.revoke_key", 1),
    # Auth routes - allow unauthenticated
    RoutePermission(r"^/api/auth/login", "POST", "", allow_unauthenticated=True),
    RoutePermission(r"^/api/auth/register", "POST", "", allow_unauthenticated=True),
    RoutePermission(r"^/api/auth/callback", "*", "", allow_unauthenticated=True),
    RoutePermission(r"^/api/auth/refresh", "POST", "", allow_unauthenticated=True),
    # Decisions (unified decision routing)
    RoutePermission(r"^/api/decisions?$", "POST", "decisions.create"),
    RoutePermission(r"^/api/decisions?$", "GET", "decisions.read"),
    RoutePermission(r"^/api/decisions?/([^/]+)$", "GET", "decisions.read", 1),
    RoutePermission(r"^/api/decisions?/([^/]+)/status$", "GET", "decisions.read", 1),
    RoutePermission(r"^/api/decisions?/([^/]+)/explain$", "GET", "decisions.read", 1),
    # Versioned decision endpoints (v1)
    RoutePermission(r"^/api/v1/decisions?$", "POST", "decisions.create"),
    RoutePermission(r"^/api/v1/decisions?$", "GET", "decisions.read"),
    RoutePermission(r"^/api/v1/decisions?/([^/]+)$", "GET", "decisions.read", 1),
    RoutePermission(r"^/api/v1/decisions?/([^/]+)/status$", "GET", "decisions.read", 1),
    RoutePermission(r"^/api/v1/decisions?/([^/]+)/explain$", "GET", "decisions.read", 1),
    # Audit findings workflow
    RoutePermission(r"^/api/audit/findings/bulk-action$", "POST", "findings.bulk"),
    RoutePermission(r"^/api/audit/findings/my-assignments$", "GET", "findings.read"),
    RoutePermission(r"^/api/audit/findings/overdue$", "GET", "findings.read"),
    RoutePermission(r"^/api/audit/findings/([^/]+)/status$", "PATCH", "findings.update", 1),
    RoutePermission(r"^/api/audit/findings/([^/]+)/assign$", "PATCH", "findings.assign", 1),
    RoutePermission(r"^/api/audit/findings/([^/]+)/unassign$", "POST", "findings.assign", 1),
    RoutePermission(r"^/api/audit/findings/([^/]+)/comments$", "*", "findings.read", 1),
    RoutePermission(r"^/api/audit/findings/([^/]+)/history$", "GET", "findings.read", 1),
    RoutePermission(r"^/api/audit/findings/([^/]+)/priority$", "PATCH", "findings.update", 1),
    RoutePermission(r"^/api/audit/findings/([^/]+)/due-date$", "PATCH", "findings.update", 1),
    RoutePermission(r"^/api/audit/findings/([^/]+)/link$", "POST", "findings.update", 1),
    RoutePermission(r"^/api/audit/findings/([^/]+)/duplicate$", "POST", "findings.update", 1),
    RoutePermission(r"^/api/audit/workflow/states$", "GET", "findings.read"),
    RoutePermission(r"^/api/audit/presets$", "GET", "findings.read"),
    RoutePermission(r"^/api/audit/types$", "GET", "findings.read"),
]


class RBACMiddleware:
    """
    HTTP middleware for enforcing role-based access control.

    Integrates with the server request/response cycle to check
    permissions before handlers are invoked.
    """

    def __init__(self, config: RBACMiddlewareConfig | None = None) -> None:
        """
        Initialize the middleware.

        Args:
            config: Middleware configuration
        """
        self.config = config or RBACMiddlewareConfig()
        self._checker = self.config.permission_checker or get_permission_checker()

        # Add default route permissions if none specified
        if not self.config.route_permissions:
            self.config.route_permissions = DEFAULT_ROUTE_PERMISSIONS.copy()

    def check_request(
        self,
        path: str,
        method: str,
        context: AuthorizationContext | None,
    ) -> tuple[bool, str, str | None]:
        """
        Check if a request is allowed.

        Args:
            path: Request path
            method: HTTP method
            context: Authorization context (None if unauthenticated)

        Returns:
            Tuple of (allowed, reason, permission_key)
        """
        # Bypass paths
        if path in self.config.bypass_paths:
            return True, "Bypass path", None

        # Bypass methods
        if method.upper() in self.config.bypass_methods:
            return True, "Bypass method", None

        # Find matching route permission
        for rule in self.config.route_permissions:
            matches, resource_id = rule.matches(path, method)
            if not matches:
                continue

            # Allow unauthenticated if rule permits
            if rule.allow_unauthenticated:
                return True, "Unauthenticated access allowed", None

            # Require authentication
            if context is None:
                return False, "Authentication required", None

            # Empty permission key means authenticated access is sufficient
            if not rule.permission_key:
                return True, "Authenticated access", None

            # Check permission
            decision = self._checker.check_permission(
                context,
                rule.permission_key,
                resource_id,
            )

            return decision.allowed, decision.reason, rule.permission_key

        # No matching rule - apply default policy
        if self.config.default_authenticated and context is None:
            return False, "Authentication required", None

        return True, "No permission rule matched", None

    def add_route_permission(self, rule: RoutePermission) -> None:
        """Add a route permission rule."""
        self.config.route_permissions.append(rule)

    def remove_route_permission(self, pattern: str, method: str) -> None:
        """Remove a route permission rule by pattern and method."""
        self.config.route_permissions = [
            r
            for r in self.config.route_permissions
            if not (
                isinstance(r.pattern, Pattern)
                and r.pattern.pattern == pattern
                and r.method == method
            )
        ]

    def get_required_permission(self, path: str, method: str) -> str | None:
        """Get the required permission for a route."""
        for rule in self.config.route_permissions:
            matches, _ = rule.matches(path, method)
            if matches:
                return rule.permission_key
        return None


def create_permission_handler(
    permission_key: str,
    resource_id_extractor: Callable[[Any], str | None] | None = None,
) -> Callable[[Any, AuthorizationContext], tuple[bool, str]]:
    """
    Create a permission check handler for custom integrations.

    Args:
        permission_key: Permission to check
        resource_id_extractor: Optional function to extract resource ID from request

    Returns:
        Handler function that takes (request, context) and returns (allowed, reason)
    """

    def handler(request: Any, context: AuthorizationContext) -> tuple[bool, str]:
        resource_id = None
        if resource_id_extractor:
            resource_id = resource_id_extractor(request)

        checker = get_permission_checker()
        decision = checker.check_permission(context, permission_key, resource_id)

        return decision.allowed, decision.reason

    return handler


# Global middleware instance
_middleware: RBACMiddleware | None = None


def get_middleware() -> RBACMiddleware:
    """Get or create the global middleware instance."""
    global _middleware
    if _middleware is None:
        _middleware = RBACMiddleware()
    return _middleware


def set_middleware(middleware: RBACMiddleware) -> None:
    """Set the global middleware instance."""
    global _middleware
    _middleware = middleware


def check_route_access(
    path: str,
    method: str,
    context: AuthorizationContext | None,
) -> tuple[bool, str]:
    """Convenience function to check route access using global middleware."""
    allowed, reason, _ = get_middleware().check_request(path, method, context)
    return allowed, reason
