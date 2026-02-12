"""Auto-generated missing SDK endpoints.

This module aggregates SDK endpoint definitions from category-specific modules.
The endpoints are split into logical categories for maintainability:

- sdk_missing_core.py: Helper functions and shared utilities
- sdk_missing_costs.py: Planned AP/AR/invoice/expense features (not yet implemented)
- sdk_missing_analytics.py: Planned analytics query/reporting endpoints (partially implemented)
- sdk_missing_platform.py: Planned platform services: knowledge mound management, personas, RLM, verticals, ML, probes

Fully redundant modules have been removed (their routes are now covered by
proper handler implementations and richer OpenAPI endpoint definitions):
- sdk_missing_compliance.py → handlers/compliance/ (13 files)
- sdk_missing_integration.py → integrations.py, webhooks.py endpoints + handlers
- sdk_missing_debates.py → debates.py, agents.py, replays.py endpoints + handlers/debates/
- sdk_missing_media.py → pulse.py endpoints + handlers/features/pulse.py, handlers/social/

For backward compatibility, all endpoints are re-exported from this module.
"""

# Re-export shared helper
from aragora.server.openapi.endpoints.sdk_missing_core import _method_stub

# Re-export categorized endpoint dicts
from aragora.server.openapi.endpoints.sdk_missing_costs import SDK_MISSING_COSTS_ENDPOINTS
from aragora.server.openapi.endpoints.sdk_missing_analytics import SDK_MISSING_ANALYTICS_ENDPOINTS
from aragora.server.openapi.endpoints.sdk_missing_platform import (
    SDK_MISSING_PLATFORM_ENDPOINTS,
    SDK_MISSING_PLATFORM_ADDITIONAL,
)


# Build the main endpoint dictionary by combining all categorized endpoints
SDK_MISSING_ENDPOINTS: dict = {}

# Merge categorized endpoints
SDK_MISSING_ENDPOINTS.update(SDK_MISSING_COSTS_ENDPOINTS)
SDK_MISSING_ENDPOINTS.update(SDK_MISSING_ANALYTICS_ENDPOINTS)
SDK_MISSING_ENDPOINTS.update(SDK_MISSING_PLATFORM_ENDPOINTS)

# Merge additional method stubs into SDK_MISSING_ENDPOINTS
for path, methods in SDK_MISSING_PLATFORM_ADDITIONAL.items():
    if path in SDK_MISSING_ENDPOINTS:
        SDK_MISSING_ENDPOINTS[path].update(methods)  # type: ignore[attr-defined]
    else:
        SDK_MISSING_ENDPOINTS[path] = methods


__all__ = [
    "SDK_MISSING_ENDPOINTS",
    "SDK_MISSING_COSTS_ENDPOINTS",
    "SDK_MISSING_ANALYTICS_ENDPOINTS",
    "SDK_MISSING_PLATFORM_ENDPOINTS",
    "SDK_MISSING_PLATFORM_ADDITIONAL",
    "_method_stub",
]
