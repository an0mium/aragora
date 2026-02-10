"""Auto-generated missing SDK endpoints.

This module aggregates SDK endpoint definitions from category-specific modules.
The endpoints are split into logical categories for maintainability:

- sdk_missing_core.py: Helper functions and shared utilities
- sdk_missing_costs.py: Costs, Payments, and Accounting endpoints
- sdk_missing_compliance.py: Compliance, Policies, Audit, and Privacy endpoints
- sdk_missing_analytics.py: Analytics endpoints
- sdk_missing_integration.py: Integrations, Webhooks, and Connectors endpoints
- sdk_missing_debates.py: Debates, Replays, Leaderboard, and Routing endpoints
- sdk_missing_media.py: Notifications and Pulse (trending topics) endpoints
- sdk_missing_platform.py: Platform services (keys, knowledge, personas, skills, users, SCIM)

For backward compatibility, all endpoints are re-exported from this module.
"""

# Re-export shared helper
from aragora.server.openapi.endpoints.sdk_missing_core import _method_stub

# Re-export categorized endpoint dicts
from aragora.server.openapi.endpoints.sdk_missing_costs import SDK_MISSING_COSTS_ENDPOINTS
from aragora.server.openapi.endpoints.sdk_missing_compliance import SDK_MISSING_COMPLIANCE_ENDPOINTS
from aragora.server.openapi.endpoints.sdk_missing_analytics import SDK_MISSING_ANALYTICS_ENDPOINTS
from aragora.server.openapi.endpoints.sdk_missing_integration import (
    SDK_MISSING_INTEGRATION_ENDPOINTS,
)

# Re-export from new submodules (debates, media, platform)
from aragora.server.openapi.endpoints.sdk_missing_debates import (
    SDK_MISSING_DEBATES_ENDPOINTS,
    SDK_MISSING_DEBATES_ADDITIONAL,
)
from aragora.server.openapi.endpoints.sdk_missing_media import (
    SDK_MISSING_MEDIA_ENDPOINTS,
    SDK_MISSING_MEDIA_ADDITIONAL,
)
from aragora.server.openapi.endpoints.sdk_missing_platform import (
    SDK_MISSING_PLATFORM_ENDPOINTS,
    SDK_MISSING_PLATFORM_ADDITIONAL,
)


# Build the main endpoint dictionary by combining all categorized endpoints
SDK_MISSING_ENDPOINTS: dict = {}

# Merge categorized endpoints (original submodules)
SDK_MISSING_ENDPOINTS.update(SDK_MISSING_COSTS_ENDPOINTS)
SDK_MISSING_ENDPOINTS.update(SDK_MISSING_COMPLIANCE_ENDPOINTS)
SDK_MISSING_ENDPOINTS.update(SDK_MISSING_ANALYTICS_ENDPOINTS)
SDK_MISSING_ENDPOINTS.update(SDK_MISSING_INTEGRATION_ENDPOINTS)

# Merge new submodule endpoints (debates, media, platform)
SDK_MISSING_ENDPOINTS.update(SDK_MISSING_DEBATES_ENDPOINTS)
SDK_MISSING_ENDPOINTS.update(SDK_MISSING_MEDIA_ENDPOINTS)
SDK_MISSING_ENDPOINTS.update(SDK_MISSING_PLATFORM_ENDPOINTS)

# Merge additional method stubs into SDK_MISSING_ENDPOINTS
for _additional_dict in (
    SDK_MISSING_DEBATES_ADDITIONAL,
    SDK_MISSING_MEDIA_ADDITIONAL,
    SDK_MISSING_PLATFORM_ADDITIONAL,
):
    for path, methods in _additional_dict.items():
        if path in SDK_MISSING_ENDPOINTS:
            SDK_MISSING_ENDPOINTS[path].update(methods)  # type: ignore[attr-defined]
        else:
            SDK_MISSING_ENDPOINTS[path] = methods


__all__ = [
    "SDK_MISSING_ENDPOINTS",
    "SDK_MISSING_COSTS_ENDPOINTS",
    "SDK_MISSING_COMPLIANCE_ENDPOINTS",
    "SDK_MISSING_ANALYTICS_ENDPOINTS",
    "SDK_MISSING_INTEGRATION_ENDPOINTS",
    "SDK_MISSING_DEBATES_ENDPOINTS",
    "SDK_MISSING_DEBATES_ADDITIONAL",
    "SDK_MISSING_MEDIA_ENDPOINTS",
    "SDK_MISSING_MEDIA_ADDITIONAL",
    "SDK_MISSING_PLATFORM_ENDPOINTS",
    "SDK_MISSING_PLATFORM_ADDITIONAL",
    "_method_stub",
]
