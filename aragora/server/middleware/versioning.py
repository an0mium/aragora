"""
API Versioning Middleware.

DEPRECATED: This module is not used in the codebase.
The canonical versioning system is in aragora/server/versioning/ directory.
Use `from aragora.server.versioning import ...` instead.
This file is kept for reference only and will be removed in a future version.

See: aragora/server/versioning/__init__.py for the canonical API.

---

Provides infrastructure for API versioning with these features:
1. Version prefix routing (/api/v1/*, /api/v2/*)
2. Version header injection (X-API-Version)
3. Version deprecation warnings
4. Backwards compatibility layer

Usage:
    # In unified_server.py or handler dispatch
    from aragora.server.middleware.versioning import (
        normalize_path,
        inject_version_headers,
        get_api_version,
        API_VERSIONS,
    )

    # Normalize incoming path
    normalized_path, version = normalize_path("/api/v1/debates")
    # normalized_path = "/api/debates", version = "v1"

    # Inject headers in response
    headers = inject_version_headers(version)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# =============================================================================
# Version Configuration
# =============================================================================


@dataclass
class APIVersion:
    """Configuration for an API version."""

    version: str  # "v1", "v2", etc.
    status: str  # "current", "deprecated", "sunset"
    release_date: str  # ISO date
    sunset_date: Optional[str] = None  # When version will be removed
    description: str = ""


# Supported API versions
API_VERSIONS: dict[str, APIVersion] = {
    "v0": APIVersion(
        version="v0",
        status="current",
        release_date="2024-01-01",
        description="Original API (no version prefix). Maintained for backwards compatibility.",
    ),
    "v1": APIVersion(
        version="v1",
        status="current",
        release_date="2026-01-13",
        description="First versioned API. All new endpoints should use v1 prefix.",
    ),
}

# Current default version for new clients
CURRENT_VERSION = "v1"

# Version prefix pattern
VERSION_PATTERN = re.compile(r"^/api/(v\d+)/(.+)$")


# =============================================================================
# Path Normalization
# =============================================================================


def normalize_path(path: str) -> tuple[str, str]:
    """Normalize a path by extracting version prefix.

    Converts versioned paths to base paths while tracking the version.

    Args:
        path: The incoming request path (e.g., "/api/v1/debates")

    Returns:
        Tuple of (normalized_path, version)
        - normalized_path: Path without version prefix (e.g., "/api/debates")
        - version: Extracted version or "v0" for unversioned paths

    Examples:
        >>> normalize_path("/api/v1/debates")
        ("/api/debates", "v1")

        >>> normalize_path("/api/debates")
        ("/api/debates", "v0")

        >>> normalize_path("/api/v2/agents/leaderboard")
        ("/api/agents/leaderboard", "v2")
    """
    match = VERSION_PATTERN.match(path)
    if match:
        version = match.group(1)
        remainder = match.group(2)
        normalized = f"/api/{remainder}"
        return normalized, version

    # No version prefix - use v0 (legacy)
    return path, "v0"


def add_version_prefix(path: str, version: str = CURRENT_VERSION) -> str:
    """Add version prefix to a path.

    Args:
        path: Base API path (e.g., "/api/debates")
        version: Version to add (default: current version)

    Returns:
        Versioned path (e.g., "/api/v1/debates")

    Examples:
        >>> add_version_prefix("/api/debates")
        "/api/v1/debates"

        >>> add_version_prefix("/api/debates", "v2")
        "/api/v2/debates"
    """
    if version == "v0":
        return path  # v0 has no prefix

    if path.startswith("/api/"):
        return f"/api/{version}/{path[5:]}"
    return path


# =============================================================================
# Version Headers
# =============================================================================


def inject_version_headers(version: str) -> dict[str, str]:
    """Generate headers for API versioning.

    Args:
        version: The API version being used

    Returns:
        Dict of headers to add to the response
    """
    headers = {
        "X-API-Version": version,
    }

    version_info = API_VERSIONS.get(version)
    if version_info:
        if version_info.status == "deprecated":
            headers["X-API-Deprecated"] = "true"
            if version_info.sunset_date:
                headers["X-API-Sunset"] = version_info.sunset_date
                headers["Warning"] = (
                    f'299 - "API version {version} is deprecated. '
                    f"Sunset date: {version_info.sunset_date}. "
                    f'Please migrate to {CURRENT_VERSION}."'
                )

    return headers


def get_api_version(headers: dict) -> str:
    """Extract requested API version from headers.

    Checks for explicit version request via X-API-Version header.

    Args:
        headers: Request headers dict

    Returns:
        Requested version or current default
    """
    requested = headers.get("X-API-Version", "").strip()
    if requested and requested in API_VERSIONS:
        return requested
    return CURRENT_VERSION


# =============================================================================
# Version Validation
# =============================================================================


def is_version_supported(version: str) -> bool:
    """Check if a version is still supported.

    Args:
        version: Version string (e.g., "v1")

    Returns:
        True if version is supported, False if sunset
    """
    version_info = API_VERSIONS.get(version)
    if not version_info:
        return False

    if version_info.status == "sunset":
        return False

    if version_info.sunset_date:
        sunset = datetime.fromisoformat(version_info.sunset_date)
        if datetime.now() > sunset:
            return False

    return True


def get_version_info(version: str) -> Optional[dict]:
    """Get detailed information about an API version.

    Args:
        version: Version string

    Returns:
        Dict with version details or None if unknown
    """
    version_info = API_VERSIONS.get(version)
    if not version_info:
        return None

    return {
        "version": version_info.version,
        "status": version_info.status,
        "release_date": version_info.release_date,
        "sunset_date": version_info.sunset_date,
        "description": version_info.description,
        "is_current": version == CURRENT_VERSION,
        "is_supported": is_version_supported(version),
    }


def get_all_versions() -> list[dict]:
    """Get information about all API versions.

    Returns:
        List of version info dicts
    """
    return [info for v in sorted(API_VERSIONS.keys()) if (info := get_version_info(v)) is not None]


# =============================================================================
# Deprecation Utilities
# =============================================================================


def deprecate_version(version: str, sunset_date: str) -> None:
    """Mark a version as deprecated.

    Args:
        version: Version to deprecate
        sunset_date: ISO date when version will be removed
    """
    if version in API_VERSIONS:
        API_VERSIONS[version].status = "deprecated"
        API_VERSIONS[version].sunset_date = sunset_date
        logger.warning(f"API version {version} deprecated. Sunset: {sunset_date}")


def log_version_usage(version: str, path: str) -> None:
    """Log API version usage for analytics.

    Args:
        version: Version being used
        path: Request path
    """
    version_info = API_VERSIONS.get(version)
    if version_info and version_info.status == "deprecated":
        logger.warning(
            f"Deprecated API version used: {version} for {path}. "
            f"Sunset date: {version_info.sunset_date}"
        )


# =============================================================================
# Middleware Class
# =============================================================================


class APIVersionMiddleware:
    """Middleware for handling API versioning.

    Can be used as ASGI/WSGI middleware or called directly.

    Usage:
        middleware = APIVersionMiddleware()

        # Process request
        normalized_path, version = middleware.process_request(path, headers)

        # Add response headers
        response_headers = middleware.process_response(version)
    """

    def __init__(
        self,
        default_version: str = CURRENT_VERSION,
        log_deprecated: bool = True,
    ):
        """Initialize middleware.

        Args:
            default_version: Default version for unversioned requests
            log_deprecated: Whether to log deprecated version usage
        """
        self.default_version = default_version
        self.log_deprecated = log_deprecated

    def process_request(
        self,
        path: str,
        headers: Optional[dict] = None,
    ) -> tuple[str, str]:
        """Process incoming request for versioning.

        Args:
            path: Request path
            headers: Request headers

        Returns:
            Tuple of (normalized_path, version)
        """
        normalized_path, path_version = normalize_path(path)

        # Path version takes precedence over header
        if path_version != "v0":
            version = path_version
        elif headers:
            version = get_api_version(headers)
        else:
            version = self.default_version

        # Log deprecated version usage
        if self.log_deprecated:
            log_version_usage(version, path)

        return normalized_path, version

    def process_response(self, version: str) -> dict[str, str]:
        """Generate response headers for versioning.

        Args:
            version: Version used for the request

        Returns:
            Headers dict to merge into response
        """
        return inject_version_headers(version)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "API_VERSIONS",
    "APIVersion",
    "APIVersionMiddleware",
    "CURRENT_VERSION",
    "add_version_prefix",
    "deprecate_version",
    "get_all_versions",
    "get_api_version",
    "get_version_info",
    "inject_version_headers",
    "is_version_supported",
    "log_version_usage",
    "normalize_path",
]
