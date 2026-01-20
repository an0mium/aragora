"""
Compatibility layer for versioning module.

Provides backward-compatible functions for existing code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Optional, Set, Tuple

from aragora.server.versioning.router import APIVersion


# Current release version
API_RELEASE_VERSION = "2.0.3"


@dataclass
class VersionConfig:
    """Configuration for API versioning."""

    current: APIVersion = APIVersion.V2
    supported: Set[APIVersion] = field(default_factory=lambda: {APIVersion.V1, APIVersion.V2})
    deprecated: Set[APIVersion] = field(default_factory=lambda: {APIVersion.V1})
    sunset_dates: Dict[APIVersion, str] = field(
        default_factory=lambda: {
            APIVersion.V1: "2026-12-31",
        }
    )
    default_for_legacy: APIVersion = APIVersion.V1

    def is_supported(self, version: APIVersion) -> bool:
        return version in self.supported

    def is_deprecated(self, version: APIVersion) -> bool:
        return version in self.deprecated

    def get_sunset_date(self, version: APIVersion) -> Optional[str]:
        return self.sunset_dates.get(version)


# Global config
_config = VersionConfig()


def get_version_config() -> VersionConfig:
    """Get current version configuration."""
    return _config


def set_version_config(config: VersionConfig) -> None:
    """Set version configuration."""
    global _config
    _config = config


def extract_version(path: str, headers: Optional[Dict[str, str]] = None) -> Tuple[APIVersion, bool]:
    """
    Extract API version from request path or headers.

    Returns:
        Tuple of (version, is_legacy)
    """
    config = get_version_config()

    # Check path prefix
    match = re.match(r"^/api/(v\d+)/", path)
    if match:
        version_str = match.group(1)
        version = APIVersion.from_string(version_str)
        if version and config.is_supported(version):
            return version, False
        # Invalid or unsupported version - return current
        return config.current, False

    # Check headers
    if headers:
        # Check X-API-Version header
        header_version = headers.get("X-API-Version") or headers.get("x-api-version")
        if header_version:
            version = APIVersion.from_string(header_version)
            if version and config.is_supported(version):
                return version, False

        # Check Accept header for version
        accept = headers.get("Accept") or headers.get("accept") or ""
        accept_match = re.search(r"vnd\.aragora\.v(\d+)", accept)
        if accept_match:
            version = APIVersion.from_string(accept_match.group(1))
            if version and config.is_supported(version):
                return version, False

    # Non-API path - return current
    if not path.startswith("/api/"):
        return config.current, False

    # Legacy path (no version prefix)
    return config.default_for_legacy, True


def version_response_headers(version: APIVersion, is_legacy: bool = False) -> Dict[str, str]:
    """Generate response headers for a version."""
    config = get_version_config()
    supported = ",".join(v.value for v in config.supported)

    headers = {
        "X-API-Version": version.value,
        "X-API-Release": API_RELEASE_VERSION,
        "X-API-Supported-Versions": supported,
    }

    if is_legacy:
        headers["X-API-Legacy"] = "true"
        headers["X-API-Migration"] = (
            f"Use /api/{config.current.value}/ prefix for versioned endpoints"
        )

    if config.is_deprecated(version):
        headers["X-API-Deprecated"] = "true"
        sunset = config.get_sunset_date(version)
        if sunset:
            headers["X-API-Sunset"] = sunset
            headers["Deprecation"] = "true"
            headers["Sunset"] = sunset

    return headers


def normalize_path_version(path: str, target_version: Optional[APIVersion] = None) -> str:
    """Normalize path to use specific version prefix."""
    config = get_version_config()
    target = target_version or config.current

    # Non-API path - return as-is
    if not path.startswith("/api/"):
        return path

    # Already versioned - preserve
    if is_versioned_path(path):
        return path

    # Add version prefix to legacy path
    rest = path[4:]  # Remove /api
    return f"/api/{target.value}{rest}"


def strip_version_prefix(path: str) -> str:
    """Remove version prefix from path, keeping /api/."""
    match = re.match(r"^/api/v\d+(/.*)?$", path)
    if match:
        rest = match.group(1) or ""
        return f"/api{rest}"
    # Already no version prefix, return as-is
    return path


def is_versioned_path(path: str) -> bool:
    """Check if path has version prefix."""
    return bool(re.match(r"^/api/v\d+/", path))


def is_legacy_path(path: str) -> bool:
    """Check if path is legacy (no version)."""
    return path.startswith("/api/") and not is_versioned_path(path)


def get_path_version(path: str) -> Optional[APIVersion]:
    """Extract version from path, or None if not versioned."""
    match = re.match(r"^/api/(v\d+)/", path)
    if match:
        return APIVersion.from_string(match.group(1))
    return None
