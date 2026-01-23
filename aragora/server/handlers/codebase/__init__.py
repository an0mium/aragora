"""
Codebase Analysis Handlers.

Provides HTTP API handlers for:
- Security vulnerability scanning
- Code quality metrics analysis
- Dependency auditing
"""

from .security import (
    SecurityHandler,
    handle_scan_repository,
    handle_get_vulnerabilities,
    handle_get_scan_status,
)
from .metrics import (
    MetricsHandler,
    handle_analyze_metrics,
    handle_get_hotspots,
)

__all__ = [
    "SecurityHandler",
    "handle_scan_repository",
    "handle_get_vulnerabilities",
    "handle_get_scan_status",
    "MetricsHandler",
    "handle_analyze_metrics",
    "handle_get_hotspots",
]
