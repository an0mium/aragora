"""
API Versioning middleware for Aragora.

Supports:
- Path-based versioning: /api/v1/debates
- Header-based version: Accept: application/vnd.aragora.v1+json
- Default version fallback for unversioned paths
- Deprecation headers for legacy endpoints

Usage:
    from aragora.server.versioning import (
        APIVersion,
        get_version_config,
        extract_version,
        version_response_headers,
        normalize_path_version,
    )

    # Extract version from request
    version = extract_version(path, headers)

    # Add version headers to response
    headers = version_response_headers(version)

    # Normalize legacy path to versioned path
    versioned_path = normalize_path_version("/api/debates")  # -> "/api/v1/debates"
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class APIVersion(Enum):
    """Supported API versions."""

    V1 = "v1"
    V2 = "v2"  # Current version (V1 deprecated, sunset 2026-06-01)


# Calendar-based API release version (for breaking change tracking)
# Format: YYYY-MM (e.g., "2026-01" for January 2026 release)
# This is used alongside v1/v2 versioning to track breaking changes within a major version
API_RELEASE_VERSION = "2026-01"


# Version path prefix pattern
VERSION_PATTERN = re.compile(r"^/api/(v\d+)/")

# Legacy path pattern (unversioned)
LEGACY_PATTERN = re.compile(r"^/api/(?!v\d+)")


@dataclass
class VersionConfig:
    """Configuration for API versioning.

    Attributes:
        current: The current (default) API version
        supported: List of all supported versions
        deprecated: List of deprecated versions (still work but emit warnings)
        sunset_dates: Map of version to sunset date (ISO format)
        default_for_legacy: Version to use for unversioned /api/* paths
    """

    current: APIVersion = APIVersion.V2
    supported: list[APIVersion] = field(default_factory=lambda: [APIVersion.V1, APIVersion.V2])
    deprecated: list[APIVersion] = field(default_factory=lambda: [APIVersion.V1])
    sunset_dates: Dict[APIVersion, str] = field(
        default_factory=lambda: {APIVersion.V1: "2026-06-01"}
    )
    default_for_legacy: APIVersion = APIVersion.V1  # Legacy paths still route to v1

    def is_supported(self, version: APIVersion) -> bool:
        """Check if a version is supported."""
        return version in self.supported

    def is_deprecated(self, version: APIVersion) -> bool:
        """Check if a version is deprecated."""
        return version in self.deprecated

    def get_sunset_date(self, version: APIVersion) -> Optional[str]:
        """Get sunset date for a version."""
        return self.sunset_dates.get(version)


# Global version configuration (singleton)
_version_config: Optional[VersionConfig] = None


def get_version_config() -> VersionConfig:
    """Get the global version configuration."""
    global _version_config
    if _version_config is None:
        _version_config = VersionConfig()
    return _version_config


def set_version_config(config: VersionConfig) -> None:
    """Set the global version configuration."""
    global _version_config
    _version_config = config


def extract_version(path: str, headers: Optional[Dict[str, str]] = None) -> Tuple[APIVersion, bool]:
    """
    Extract API version from request path or headers.

    Version is determined in the following priority:
    1. Path-based: /api/v1/debates -> V1
    2. Header-based: Accept: application/vnd.aragora.v1+json -> V1
    3. Default: Current version from config

    Args:
        path: Request path (e.g., "/api/v1/debates")
        headers: Request headers dict

    Returns:
        Tuple of (APIVersion, is_legacy) where is_legacy=True if path was unversioned
    """
    config = get_version_config()
    headers = headers or {}

    # Check path-based versioning first
    match = VERSION_PATTERN.match(path)
    if match:
        version_str = match.group(1)
        try:
            version = APIVersion(version_str)
            if config.is_supported(version):
                return version, False
            else:
                logger.warning(f"unsupported_api_version: {version_str}")
                return config.current, False
        except ValueError:
            logger.warning(f"invalid_api_version: {version_str}")
            return config.current, False

    # Check header-based versioning
    accept = headers.get("Accept", "")
    if "vnd.aragora." in accept:
        # Parse Accept: application/vnd.aragora.v1+json
        for version in APIVersion:
            if f"vnd.aragora.{version.value}" in accept:
                if config.is_supported(version):
                    return version, False
                else:
                    logger.warning(f"unsupported_api_version_header: {version.value}")

    # Check if this is a legacy (unversioned) API path
    if LEGACY_PATTERN.match(path):
        return config.default_for_legacy, True

    # Default to current version (non-API paths)
    return config.current, False


def version_response_headers(
    version: APIVersion,
    is_legacy: bool = False,
    config: Optional[VersionConfig] = None,
) -> Dict[str, str]:
    """
    Generate version-related response headers.

    Args:
        version: The API version being used
        is_legacy: Whether the request used a legacy (unversioned) path
        config: Optional version config (uses global if not provided)

    Returns:
        Dict of headers to add to response
    """
    config = config or get_version_config()

    headers = {
        "X-API-Version": version.value,
        "X-API-Release": API_RELEASE_VERSION,  # Calendar-based release version
        "X-API-Supported-Versions": ",".join(v.value for v in config.supported),
    }

    # Add deprecation warning for legacy paths
    if is_legacy:
        headers["X-API-Legacy"] = "true"
        headers["X-API-Migration"] = f"Use /api/{version.value}/ prefix for versioned endpoints"

    # Add deprecation headers for deprecated versions
    if config.is_deprecated(version):
        headers["X-API-Deprecated"] = "true"
        sunset_date = config.get_sunset_date(version)
        if sunset_date:
            headers["X-API-Sunset"] = sunset_date

    return headers


def normalize_path_version(
    path: str,
    target_version: Optional[APIVersion] = None,
) -> str:
    """
    Normalize a path to include version prefix.

    Converts legacy paths like /api/debates to /api/v1/debates.
    Already versioned paths are returned unchanged.

    Args:
        path: Request path
        target_version: Version to use (defaults to current)

    Returns:
        Path with version prefix
    """
    config = get_version_config()
    version = target_version or config.current

    # Already versioned
    if VERSION_PATTERN.match(path):
        return path

    # Legacy API path - add version
    if LEGACY_PATTERN.match(path):
        # /api/debates -> /api/v1/debates
        return path.replace("/api/", f"/api/{version.value}/", 1)

    # Non-API path - return unchanged
    return path


def strip_version_prefix(path: str) -> str:
    """
    Strip version prefix from path for handler matching.

    This allows handlers to match against /api/debates even when
    the actual request path is /api/v1/debates.

    Args:
        path: Request path (e.g., "/api/v1/debates")

    Returns:
        Path without version prefix (e.g., "/api/debates")
    """
    match = VERSION_PATTERN.match(path)
    if match:
        # /api/v1/debates -> /api/debates
        version_prefix = match.group(0)  # "/api/v1/"
        remainder = path[len(version_prefix) - 1 :]  # "/debates"
        return f"/api{remainder}"
    return path


def is_versioned_path(path: str) -> bool:
    """Check if a path includes a version prefix."""
    return VERSION_PATTERN.match(path) is not None


def is_legacy_path(path: str) -> bool:
    """Check if a path is a legacy (unversioned) API path."""
    return LEGACY_PATTERN.match(path) is not None


def get_path_version(path: str) -> Optional[APIVersion]:
    """
    Extract version from path if present.

    Args:
        path: Request path

    Returns:
        APIVersion if path is versioned, None otherwise
    """
    match = VERSION_PATTERN.match(path)
    if match:
        try:
            return APIVersion(match.group(1))
        except ValueError:
            return None
    return None


__all__ = [
    "APIVersion",
    "API_RELEASE_VERSION",
    "VersionConfig",
    "get_version_config",
    "set_version_config",
    "extract_version",
    "version_response_headers",
    "normalize_path_version",
    "strip_version_prefix",
    "is_versioned_path",
    "is_legacy_path",
    "get_path_version",
]
