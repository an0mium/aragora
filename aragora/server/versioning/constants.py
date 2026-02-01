"""
Central constants for API versioning and sunset management.

All sunset dates, deprecation announcements, and migration URLs
should be defined here to ensure consistency across the codebase.

Usage:
    from aragora.server.versioning.constants import (
        V1_SUNSET_DATE,
        V1_SUNSET_ISO,
        V1_DEPRECATION_ANNOUNCED,
        MIGRATION_DOCS_URL,
    )
"""

from __future__ import annotations

from datetime import date, datetime, timezone

# =============================================================================
# V1 API Sunset Configuration
# =============================================================================

# The date when v1 API was deprecated (deprecation announcement)
V1_DEPRECATION_ANNOUNCED: date = date(2025, 1, 15)

# The date when v1 API endpoints will be removed
V1_SUNSET_DATE: date = date(2026, 6, 1)

# ISO 8601 string representation for headers and configs
V1_SUNSET_ISO: str = "2026-06-01"

# HTTP-date format (RFC 7231) for Sunset header per RFC 8594
V1_SUNSET_HTTP_DATE: str = "Mon, 01 Jun 2026 00:00:00 GMT"

# Unix timestamp for RFC 8594 Deprecation header (@<timestamp> format)
V1_DEPRECATION_TIMESTAMP: int = int(
    datetime.combine(V1_SUNSET_DATE, datetime.min.time()).replace(tzinfo=timezone.utc).timestamp()
)

# =============================================================================
# Current API Version
# =============================================================================

# The current stable API version that clients should migrate to
CURRENT_API_VERSION: str = "v2"

# =============================================================================
# Documentation URLs
# =============================================================================

# URL to the migration guide documentation
MIGRATION_DOCS_URL: str = "https://docs.aragora.ai/migration/v1-to-v2"

# URL to the API versioning documentation
API_VERSIONING_DOCS_URL: str = "https://docs.aragora.ai/api/versioning"

# Relative path within docs/ for the migration guide
MIGRATION_GUIDE_PATH: str = "/docs/MIGRATION_V1_TO_V2.md"

# =============================================================================
# Environment Variable Names
# =============================================================================

# Set to "false" to disable the v1 deprecation middleware
ENV_DISABLE_DEPRECATION_HEADERS: str = "ARAGORA_DISABLE_V1_DEPRECATION"

# Set to "true" to block v1 requests entirely (post-sunset enforcement)
ENV_BLOCK_SUNSET_ENDPOINTS: str = "ARAGORA_BLOCK_SUNSET_ENDPOINTS"

# Set to "true" to enable verbose logging of v1 usage
ENV_LOG_DEPRECATED_USAGE: str = "ARAGORA_LOG_DEPRECATED_USAGE"


# =============================================================================
# Helpers
# =============================================================================


def days_until_v1_sunset() -> int:
    """Return the number of days until the v1 API sunset date.

    Returns 0 if the sunset date has already passed.
    """
    delta = V1_SUNSET_DATE - date.today()
    return max(0, delta.days)


def is_v1_sunset() -> bool:
    """Check whether the v1 API sunset date has passed."""
    return date.today() > V1_SUNSET_DATE


def deprecation_level() -> str:
    """Return the current deprecation severity level for v1.

    Returns:
        'sunset' if past the sunset date,
        'critical' if within 30 days of sunset,
        'warning' otherwise.
    """
    days = days_until_v1_sunset()
    if is_v1_sunset():
        return "sunset"
    if days < 30:
        return "critical"
    return "warning"


__all__ = [
    "API_VERSIONING_DOCS_URL",
    "CURRENT_API_VERSION",
    "ENV_BLOCK_SUNSET_ENDPOINTS",
    "ENV_DISABLE_DEPRECATION_HEADERS",
    "ENV_LOG_DEPRECATED_USAGE",
    "MIGRATION_DOCS_URL",
    "MIGRATION_GUIDE_PATH",
    "V1_DEPRECATION_ANNOUNCED",
    "V1_DEPRECATION_TIMESTAMP",
    "V1_SUNSET_DATE",
    "V1_SUNSET_HTTP_DATE",
    "V1_SUNSET_ISO",
    "days_until_v1_sunset",
    "deprecation_level",
    "is_v1_sunset",
]
