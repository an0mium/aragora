"""Admin handlers - administration, dashboard, billing, health, security, and system.

This package provides administrative endpoints organized into focused modules:

- handler.py: Main AdminHandler class (facade composing all mixins)
- metrics_dashboard.py: Platform metrics and statistics endpoints
- users.py: User and organization management endpoints
- nomic_admin.py: Nomic loop control endpoints
- dashboard.py: DashboardHandler for dashboard-specific views
- health/: HealthHandler for system health checks
- security.py: SecurityHandler for security-related endpoints
- system.py: SystemHandler for system management

Note: BillingHandler has been migrated to the billing/ subpackage but is
re-exported here for backward compatibility.

Note: The original admin.py has been decomposed into handler.py and the
mixin modules. AdminHandler is now re-exported from handler.py.
"""

from __future__ import annotations

from typing import Any

from aragora.billing.jwt_auth import extract_user_from_request

__all__ = [
    # Main handler and roles
    "ADMIN_ROLES",
    "AdminHandler",
    "admin_secure_endpoint",
    # Permission constants
    "PERM_ADMIN_USERS_WRITE",
    "PERM_ADMIN_IMPERSONATE",
    "PERM_ADMIN_NOMIC_WRITE",
    "PERM_ADMIN_SYSTEM_WRITE",
    # Mixins (for extension)
    "MetricsDashboardMixin",
    "UserManagementMixin",
    "NomicAdminMixin",
    # Other handlers
    "BillingHandler",
    "DashboardHandler",
    "HealthHandler",
    "SecurityHandler",
    "SystemHandler",
    # Utilities
    "extract_user_from_request",
]


def __getattr__(name: str) -> Any:
    if name == "ADMIN_ROLES":
        from .handler import ADMIN_ROLES

        return ADMIN_ROLES
    if name == "AdminHandler":
        from .handler import AdminHandler

        return AdminHandler
    if name == "admin_secure_endpoint":
        from .handler import admin_secure_endpoint

        return admin_secure_endpoint
    if name == "PERM_ADMIN_USERS_WRITE":
        from .handler import PERM_ADMIN_USERS_WRITE

        return PERM_ADMIN_USERS_WRITE
    if name == "PERM_ADMIN_IMPERSONATE":
        from .handler import PERM_ADMIN_IMPERSONATE

        return PERM_ADMIN_IMPERSONATE
    if name == "PERM_ADMIN_NOMIC_WRITE":
        from .handler import PERM_ADMIN_NOMIC_WRITE

        return PERM_ADMIN_NOMIC_WRITE
    if name == "PERM_ADMIN_SYSTEM_WRITE":
        from .handler import PERM_ADMIN_SYSTEM_WRITE

        return PERM_ADMIN_SYSTEM_WRITE
    if name == "MetricsDashboardMixin":
        from .metrics_dashboard import MetricsDashboardMixin

        return MetricsDashboardMixin
    if name == "UserManagementMixin":
        from .users import UserManagementMixin

        return UserManagementMixin
    if name == "NomicAdminMixin":
        from .nomic_admin import NomicAdminMixin

        return NomicAdminMixin
    if name == "DashboardHandler":
        from .dashboard import DashboardHandler

        return DashboardHandler
    if name == "HealthHandler":
        from .health import HealthHandler

        return HealthHandler
    if name == "SecurityHandler":
        from .security import SecurityHandler

        return SecurityHandler
    if name == "SystemHandler":
        from .system import SystemHandler

        return SystemHandler
    if name == "BillingHandler":
        from ..billing import BillingHandler

        return BillingHandler
    if name == "extract_user_from_request":
        return extract_user_from_request

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
