"""Analytics Dashboard handler package.

Re-exports all public symbols for backward compatibility so that
``from aragora.server.handlers.analytics_dashboard import ...`` continues
to work exactly as before.
"""

from ._shared import (
    ANALYTICS_STUB_RESPONSES,
    PERM_ANALYTICS_ADMIN,
    PERM_ANALYTICS_COMPLIANCE,
    PERM_ANALYTICS_COST,
    PERM_ANALYTICS_DELIBERATIONS,
    PERM_ANALYTICS_EXPORT,
    PERM_ANALYTICS_FLIPS,
    PERM_ANALYTICS_READ,
    PERM_ANALYTICS_TOKENS,
    PERM_ANALYTICS_WRITE,
    RBAC_AVAILABLE,
    METRICS_AVAILABLE,
    get_analytics_response,
    _run_async,  # noqa: F401
)
from .handler import AnalyticsDashboardHandler

__all__ = [
    "ANALYTICS_STUB_RESPONSES",
    "AnalyticsDashboardHandler",
    "PERM_ANALYTICS_ADMIN",
    "PERM_ANALYTICS_COMPLIANCE",
    "PERM_ANALYTICS_COST",
    "PERM_ANALYTICS_DELIBERATIONS",
    "PERM_ANALYTICS_EXPORT",
    "PERM_ANALYTICS_FLIPS",
    "PERM_ANALYTICS_READ",
    "PERM_ANALYTICS_TOKENS",
    "PERM_ANALYTICS_WRITE",
    "RBAC_AVAILABLE",
    "METRICS_AVAILABLE",
    "get_analytics_response",
]
