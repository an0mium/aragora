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

from aragora.billing.jwt_auth import extract_user_from_request

# Import from decomposed modules
from .handler import (
    ADMIN_ROLES,
    AdminHandler,
    admin_secure_endpoint,
    PERM_ADMIN_USERS_WRITE,
    PERM_ADMIN_IMPERSONATE,
    PERM_ADMIN_NOMIC_WRITE,
    PERM_ADMIN_SYSTEM_WRITE,
)
from .metrics_dashboard import MetricsDashboardMixin
from .users import UserManagementMixin
from .nomic_admin import NomicAdminMixin

# Import other handlers
from .dashboard import DashboardHandler
from .health import HealthHandler
from .security import SecurityHandler
from .system import SystemHandler

# Import from new location for backward compatibility
from ..billing import BillingHandler

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
