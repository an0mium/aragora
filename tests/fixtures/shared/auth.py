"""
Shared RBAC and authentication mocking utilities.

This module provides centralized auth mocking for handler tests,
eliminating ~150 lines of duplicate auth bypass code.

Usage:
    from tests.fixtures.shared.auth import (
        create_mock_auth_context,
        patch_get_auth_context,
    )

    def test_handler_endpoint(monkeypatch):
        ctx = create_mock_auth_context(roles=["admin"])
        with patch_get_auth_context(monkeypatch, ctx):
            result = handler.handle("/api/v1/resource", {}, mock_http, "GET")
            assert result.status_code == 200
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, List, Optional, Set
from unittest.mock import MagicMock, patch


# ============================================================================
# Mock Authorization Context
# ============================================================================


@dataclass
class MockAuthorizationContext:
    """Mock authorization context for testing RBAC-protected handlers.

    This mirrors the real AuthorizationContext from aragora.rbac.models
    but is test-friendly with sensible defaults.

    Attributes:
        user_id: Unique user identifier
        user_email: User's email address
        org_id: Organization identifier
        workspace_id: Workspace identifier
        roles: Set of role names (e.g., {"admin", "owner"})
        permissions: Set of permission strings (e.g., {"debates:read"})
        api_key_scope: Optional API key scope
        ip_address: Client IP address
        user_agent: Client user agent
        request_id: Unique request identifier
        timestamp: Request timestamp
    """

    user_id: str = "test-user-001"
    user_email: str = "test@example.com"
    org_id: str = "test-org-001"
    workspace_id: str = "test-ws-001"
    roles: Set[str] = field(default_factory=lambda: {"admin"})
    permissions: Set[str] = field(default_factory=set)
    api_key_scope: Optional[str] = None
    ip_address: str = "127.0.0.1"
    user_agent: str = "test-agent"
    request_id: str = "req-test-001"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def __post_init__(self):
        # Ensure roles and permissions are sets
        if isinstance(self.roles, list):
            self.roles = set(self.roles)
        if isinstance(self.permissions, list):
            self.permissions = set(self.permissions)

        # Set default comprehensive permissions if none provided
        if not self.permissions:
            self.permissions = DEFAULT_TEST_PERMISSIONS.copy()

    def has_permission(self, permission: str) -> bool:
        """Check if context has a specific permission."""
        # Wildcard grants all permissions
        if "*" in self.permissions:
            return True
        return permission in self.permissions

    def has_role(self, role: str) -> bool:
        """Check if context has a specific role."""
        return role in self.roles

    def has_any_permission(self, permissions: List[str]) -> bool:
        """Check if context has any of the specified permissions."""
        return any(self.has_permission(p) for p in permissions)

    def has_all_permissions(self, permissions: List[str]) -> bool:
        """Check if context has all of the specified permissions."""
        return all(self.has_permission(p) for p in permissions)


# Default permissions for test contexts
DEFAULT_TEST_PERMISSIONS: Set[str] = {
    "debates:read",
    "debates:write",
    "debates:create",
    "debates:delete",
    "agents:read",
    "agents:write",
    "memory.read",
    "memory.write",
    "knowledge:read",
    "knowledge:write",
    "knowledge.read",
    "knowledge.write",
    "knowledge.delete",
    "workflows:read",
    "workflows:write",
    "admin:read",
    "admin:write",
    "billing:read",
    "billing:write",
    "costs.read",
    "costs.write",
    "audit:read",
    "backups:read",
    "backups:write",
    "tournaments:read",
    "tournaments:create",
}


# ============================================================================
# Context Factory Functions
# ============================================================================


def create_mock_auth_context(
    user_id: str = "test-user-001",
    user_email: str = "test@example.com",
    org_id: str = "test-org-001",
    workspace_id: str = "test-ws-001",
    roles: Optional[Set[str]] = None,
    permissions: Optional[Set[str]] = None,
    **kwargs: Any,
) -> MockAuthorizationContext:
    """Create a mock authorization context with custom settings.

    Args:
        user_id: User identifier
        user_email: User email
        org_id: Organization ID
        workspace_id: Workspace ID
        roles: Set of roles (defaults to {"admin"})
        permissions: Set of permissions (defaults to comprehensive set)
        **kwargs: Additional context attributes

    Returns:
        MockAuthorizationContext instance
    """
    return MockAuthorizationContext(
        user_id=user_id,
        user_email=user_email,
        org_id=org_id,
        workspace_id=workspace_id,
        roles=roles or {"admin"},
        permissions=permissions or set(),
        **kwargs,
    )


def create_admin_context(
    user_id: str = "test-admin-001",
    org_id: str = "test-org-001",
) -> MockAuthorizationContext:
    """Create an admin authorization context with full permissions."""
    return MockAuthorizationContext(
        user_id=user_id,
        user_email="admin@example.com",
        org_id=org_id,
        roles={"admin", "owner"},
        permissions={"*"},  # Wildcard grants all
    )


def create_viewer_context(
    user_id: str = "test-viewer-001",
    org_id: str = "test-org-001",
) -> MockAuthorizationContext:
    """Create a viewer authorization context (read-only permissions)."""
    return MockAuthorizationContext(
        user_id=user_id,
        user_email="viewer@example.com",
        org_id=org_id,
        roles={"viewer"},
        permissions={
            "debates:read",
            "agents:read",
            "memory.read",
            "knowledge:read",
            "knowledge.read",
            "workflows:read",
        },
    )


def create_editor_context(
    user_id: str = "test-editor-001",
    org_id: str = "test-org-001",
) -> MockAuthorizationContext:
    """Create an editor authorization context (read/write, no admin)."""
    return MockAuthorizationContext(
        user_id=user_id,
        user_email="editor@example.com",
        org_id=org_id,
        roles={"editor"},
        permissions={
            "debates:read",
            "debates:write",
            "debates:create",
            "agents:read",
            "agents:write",
            "memory.read",
            "memory.write",
            "knowledge:read",
            "knowledge:write",
            "knowledge.read",
            "knowledge.write",
            "workflows:read",
            "workflows:write",
        },
    )


# ============================================================================
# Patching Utilities
# ============================================================================


def patch_get_auth_context(
    monkeypatch: Any,
    auth_context: Optional[MockAuthorizationContext] = None,
) -> None:
    """Patch get_auth_context across all handler modules.

    This patches get_auth_context at various locations to return the
    provided auth context, enabling RBAC bypass for tests.

    Args:
        monkeypatch: pytest monkeypatch fixture
        auth_context: Context to return (defaults to admin context)
    """
    if auth_context is None:
        auth_context = create_admin_context()

    async def mock_get_auth_context(request, require_auth=False):
        """Mock get_auth_context that returns the configured context."""
        return auth_context

    # Patch at various locations
    _patch_locations = [
        "aragora.server.handlers.utils.auth.get_auth_context",
        "aragora.server.handlers.secure.get_auth_context",
    ]

    for location in _patch_locations:
        try:
            monkeypatch.setattr(location, mock_get_auth_context)
        except (ImportError, AttributeError):
            pass

    # Patch autonomous handler modules
    autonomous_modules = ["triggers", "alerts", "approvals", "learning", "monitoring"]
    for mod_name in autonomous_modules:
        try:
            from aragora.server.handlers import autonomous

            mod = getattr(autonomous, mod_name, None)
            if mod and hasattr(mod, "get_auth_context"):
                monkeypatch.setattr(mod, "get_auth_context", mock_get_auth_context)
        except (ImportError, AttributeError):
            pass


def patch_rbac_decorators(monkeypatch: Any) -> None:
    """Patch RBAC decorators to be pass-through for tests.

    This makes @require_permission and similar decorators no-ops,
    allowing handler logic to be tested without auth checks.

    Args:
        monkeypatch: pytest monkeypatch fixture
    """

    def passthrough_decorator(*args, **kwargs):
        """Decorator that does nothing."""

        def decorator(func):
            return func

        return decorator

    try:
        monkeypatch.setattr(
            "aragora.rbac.decorators.require_permission",
            passthrough_decorator,
        )
        monkeypatch.setattr(
            "aragora.rbac.decorators.require_role",
            passthrough_decorator,
        )
    except (ImportError, AttributeError):
        pass


def patch_context_from_args(
    monkeypatch: Any,
    auth_context: Optional[MockAuthorizationContext] = None,
) -> None:
    """Patch _get_context_from_args to return mock context.

    This is critical for functions that are already decorated at import time.
    When the decorator can't find a context in args/kwargs, this patch
    provides the mock context.

    Args:
        monkeypatch: pytest monkeypatch fixture
        auth_context: Context to return when none found
    """
    if auth_context is None:
        auth_context = create_admin_context()

    try:
        from aragora.rbac import decorators

        original_get_context = decorators._get_context_from_args

        def patched_get_context_from_args(args, kwargs, context_param):
            """Return mock context if no real context found."""
            result = original_get_context(args, kwargs, context_param)
            if result is None:
                return auth_context
            return result

        monkeypatch.setattr(
            decorators,
            "_get_context_from_args",
            patched_get_context_from_args,
        )
    except (ImportError, AttributeError):
        pass


def patch_user_auth_context(
    monkeypatch: Any,
    auth_context: Optional[MockAuthorizationContext] = None,
) -> None:
    """Patch extract_user_from_request for JWT-based user auth.

    This fixes tests that use require_auth_or_error() / get_current_user().

    Args:
        monkeypatch: pytest monkeypatch fixture
        auth_context: Context to base user auth on
    """
    if auth_context is None:
        auth_context = create_admin_context()

    try:
        from aragora.billing.auth.context import UserAuthContext

        mock_user_ctx = UserAuthContext(
            authenticated=True,
            user_id=auth_context.user_id,
            email=auth_context.user_email,
            org_id=auth_context.org_id,
            role="admin",
            token_type="access",
        )
        # Add permissions and roles for _check_permission in handlers
        mock_user_ctx.permissions = auth_context.permissions  # type: ignore[attr-defined]
        mock_user_ctx.roles = auth_context.roles  # type: ignore[attr-defined]

        def mock_extract_user(handler, user_store=None):
            """Mock extract_user_from_request returning authenticated context."""
            return mock_user_ctx

        monkeypatch.setattr(
            "aragora.billing.jwt_auth.extract_user_from_request",
            mock_extract_user,
        )
    except (ImportError, AttributeError):
        pass


def setup_full_auth_bypass(
    monkeypatch: Any,
    auth_context: Optional[MockAuthorizationContext] = None,
) -> MockAuthorizationContext:
    """Set up complete auth bypass for handler tests.

    This is a convenience function that calls all auth patching functions.
    Use this in autouse fixtures for comprehensive auth bypass.

    Args:
        monkeypatch: pytest monkeypatch fixture
        auth_context: Context to use (defaults to admin context)

    Returns:
        The mock auth context being used
    """
    if auth_context is None:
        auth_context = create_admin_context()

    patch_get_auth_context(monkeypatch, auth_context)
    patch_context_from_args(monkeypatch, auth_context)
    patch_user_auth_context(monkeypatch, auth_context)

    return auth_context


# ============================================================================
# Context Manager for Handler Authentication
# ============================================================================


@contextmanager
def authenticated_handler_context(
    handler_instance: Any,
    user_id: str = "test-user-001",
    org_id: str = "test-org-001",
    roles: Optional[Set[str]] = None,
    permissions: Optional[Set[str]] = None,
):
    """Context manager that patches a handler instance to bypass RBAC.

    This patches the get_auth_context and check_permission methods
    on handler instances so they don't require actual authentication.

    Args:
        handler_instance: The handler to patch
        user_id: User ID for the mock context
        org_id: Org ID for the mock context
        roles: Roles for the mock context
        permissions: Permissions for the mock context

    Yields:
        Tuple of (context, mock_get_auth, mock_check)

    Example:
        with authenticated_handler_context(handler) as (ctx, _, _):
            result = handler.handle("/api/v1/protected", {}, mock_http, "GET")
            assert result.status_code == 200
    """
    ctx = create_mock_auth_context(
        user_id=user_id,
        org_id=org_id,
        roles=roles,
        permissions=permissions,
    )

    with patch.object(handler_instance, "get_auth_context", return_value=ctx) as mock_get_auth:
        with patch.object(handler_instance, "check_permission", return_value=True) as mock_check:
            yield ctx, mock_get_auth, mock_check
