"""API versioning middleware.

Provides version management for the Aragora API including:
- Version routing and negotiation
- Deprecation warnings
- Version metadata injection
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class APIVersion(Enum):
    """Supported API versions."""

    V1 = "v1"
    V2 = "v2"


# Current default API version
CURRENT_VERSION = APIVersion.V1

# All supported versions
API_VERSIONS: Dict[str, APIVersion] = {
    "v1": APIVersion.V1,
    "v2": APIVersion.V2,
}


@dataclass
class VersionInfo:
    """Information about an API version."""

    version: APIVersion
    deprecated: bool = False
    sunset_date: Optional[str] = None
    description: str = ""


# Version registry
_version_info: Dict[APIVersion, VersionInfo] = {
    APIVersion.V1: VersionInfo(version=APIVersion.V1, description="Initial API version"),
    APIVersion.V2: VersionInfo(version=APIVersion.V2, description="Enhanced API version"),
}


def get_api_version(path: str) -> APIVersion:
    """Extract API version from request path.

    Args:
        path: Request path (e.g., "/api/v1/debates")

    Returns:
        APIVersion enum value, defaults to CURRENT_VERSION
    """
    parts = path.strip("/").split("/")
    if len(parts) >= 2 and parts[0] == "api":
        version_str = parts[1].lower()
        if version_str in API_VERSIONS:
            return API_VERSIONS[version_str]
    return CURRENT_VERSION


def is_version_supported(version: APIVersion) -> bool:
    """Check if a version is currently supported."""
    return version in _version_info


def get_version_info(version: APIVersion) -> Optional[VersionInfo]:
    """Get metadata for a specific version."""
    return _version_info.get(version)


def get_all_versions() -> List[VersionInfo]:
    """Get info for all supported versions."""
    return list(_version_info.values())


def deprecate_version(version: APIVersion, sunset_date: Optional[str] = None) -> None:
    """Mark a version as deprecated.

    Args:
        version: Version to deprecate
        sunset_date: Optional date when version will be removed (ISO format)
    """
    if version in _version_info:
        _version_info[version].deprecated = True
        _version_info[version].sunset_date = sunset_date


def add_version_prefix(path: str, version: Optional[APIVersion] = None) -> str:
    """Add version prefix to a path.

    Args:
        path: Path without version (e.g., "/debates")
        version: Version to add, defaults to CURRENT_VERSION

    Returns:
        Versioned path (e.g., "/api/v1/debates")
    """
    version = version or CURRENT_VERSION
    if path.startswith("/api/"):
        return path
    if path.startswith("/"):
        path = path[1:]
    return f"/api/{version.value}/{path}"


def normalize_path(path: str) -> str:
    """Remove version prefix from path.

    Args:
        path: Potentially versioned path

    Returns:
        Path without version prefix
    """
    parts = path.strip("/").split("/")
    if len(parts) >= 2 and parts[0] == "api" and parts[1] in API_VERSIONS:
        return "/" + "/".join(parts[2:])
    return path


def inject_version_headers(handler: Any, version: APIVersion) -> None:
    """Inject version headers into response.

    Args:
        handler: HTTP handler with send_header method
        version: API version being used
    """
    if hasattr(handler, "send_header"):
        handler.send_header("X-API-Version", version.value)
        info = get_version_info(version)
        if info and info.deprecated:
            handler.send_header("X-API-Deprecated", "true")
            if info.sunset_date:
                handler.send_header("X-API-Sunset", info.sunset_date)


def log_version_usage(version: APIVersion, path: str, client_ip: str = "") -> None:
    """Log API version usage for analytics.

    Args:
        version: Version being used
        path: Request path
        client_ip: Client IP address
    """
    import logging

    logger = logging.getLogger(__name__)
    logger.debug(f"API {version.value} request: {path} from {client_ip}")


class APIVersionMiddleware:
    """Middleware for handling API version negotiation.

    Extracts version from request path and injects version headers.
    """

    def __init__(self, handler_class: type):
        """Initialize middleware with handler class.

        Args:
            handler_class: Base handler class to wrap
        """
        self.handler_class = handler_class

    def __call__(self, *args, **kwargs) -> Any:
        """Create wrapped handler instance."""
        handler = self.handler_class(*args, **kwargs)
        return handler

    @staticmethod
    def extract_version(path: str) -> APIVersion:
        """Extract version from path."""
        return get_api_version(path)


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
