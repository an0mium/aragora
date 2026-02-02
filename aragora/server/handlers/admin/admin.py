"""
Admin API Handlers - Backward Compatibility Shim.

This module provides backward compatibility for imports from the old location.
The actual implementation has been decomposed into focused submodules:

- handler.py: Main AdminHandler class (facade composing all mixins)
- metrics_dashboard.py: Platform metrics and statistics endpoints
- users.py: User and organization management endpoints
- nomic_admin.py: Nomic loop control endpoints

New code should import from aragora.server.handlers.admin directly:
    from aragora.server.handlers.admin import AdminHandler, ADMIN_ROLES

Or from the specific submodules:
    from aragora.server.handlers.admin.handler import AdminHandler
    from aragora.server.handlers.admin.users import UserManagementMixin

Endpoints:
- GET /api/v1/admin/organizations - List all organizations
- GET /api/v1/admin/users - List all users
- GET /api/v1/admin/stats - Get system-wide statistics
- GET /api/v1/admin/system/metrics - Get aggregated system metrics
- POST /api/v1/admin/impersonate/:user_id - Create impersonation token
- POST /api/v1/admin/users/:user_id/deactivate - Deactivate a user
- POST /api/v1/admin/users/:user_id/activate - Activate a user
- POST /api/v1/admin/users/:user_id/unlock - Unlock a locked user account
- GET /api/v1/admin/nomic/status - Get detailed nomic status
- GET /api/v1/admin/nomic/circuit-breakers - Get circuit breaker status
- POST /api/v1/admin/nomic/reset - Reset nomic to a specific phase
- POST /api/v1/admin/nomic/pause - Pause the nomic loop
- POST /api/v1/admin/nomic/resume - Resume the nomic loop
- POST /api/v1/admin/nomic/circuit-breakers/reset - Reset all circuit breakers
"""

from __future__ import annotations

# Re-export everything from the new location for backward compatibility
from .handler import (
    AdminHandler,
    ADMIN_ROLES,
    admin_secure_endpoint,
    PERM_ADMIN_USERS_WRITE,
    PERM_ADMIN_IMPERSONATE,
    PERM_ADMIN_NOMIC_WRITE,
    PERM_ADMIN_SYSTEM_WRITE,
)

# Re-export mixins for those who want to extend
from .metrics_dashboard import MetricsDashboardMixin
from .users import UserManagementMixin
from .nomic_admin import NomicAdminMixin

# Re-export commonly used utilities that tests may patch
from aragora.billing.jwt_auth import extract_user_from_request

__all__ = [
    # Main handler and roles
    "AdminHandler",
    "ADMIN_ROLES",
    "admin_secure_endpoint",
    # Permission constants
    "PERM_ADMIN_USERS_WRITE",
    "PERM_ADMIN_IMPERSONATE",
    "PERM_ADMIN_NOMIC_WRITE",
    "PERM_ADMIN_SYSTEM_WRITE",
    # Mixins
    "MetricsDashboardMixin",
    "UserManagementMixin",
    "NomicAdminMixin",
    # Utilities (for test patching compatibility)
    "extract_user_from_request",
]
