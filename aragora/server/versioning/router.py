"""
Versioned API Router.

Provides version-aware routing with support for:
- URL prefix versioning (/api/v1/...)
- Header-based version selection (X-API-Version)
- Accept header version negotiation
- Automatic version fallback

Usage:
    router = VersionedRouter()

    @router.route("/users", version=APIVersion.V1)
    def get_users_v1(request):
        return {"users": [...]}

    @router.route("/users", version=APIVersion.V2)
    def get_users_v2(request):
        return {"data": {"users": [...]}, "meta": {...}}
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Union

logger = logging.getLogger(__name__)


# ============================================================================
# Version Types
# ============================================================================


class APIVersion(Enum):
    """Supported API versions."""

    V1 = "v1"
    V2 = "v2"
    V3 = "v3"

    @classmethod
    def from_string(cls, version_str: str) -> Optional["APIVersion"]:
        """Parse version from string."""
        if not version_str:
            return None

        # Handle 'v1', 'V1', '1', '1.0' formats
        version_str = version_str.lower().strip()
        if version_str.startswith("v"):
            version_str = version_str[1:]

        # Extract major version
        if "." in version_str:
            version_str = version_str.split(".")[0]

        try:
            return cls(f"v{version_str}")
        except ValueError:
            return None

    @classmethod
    def latest(cls) -> "APIVersion":
        """Get latest stable version."""
        return cls.V1  # V1 is current stable

    @classmethod
    def all_versions(cls) -> List["APIVersion"]:
        """Get all versions in order."""
        return [cls.V1, cls.V2, cls.V3]

    def __lt__(self, other: "APIVersion") -> bool:
        versions = self.all_versions()
        return versions.index(self) < versions.index(other)

    def __le__(self, other: "APIVersion") -> bool:
        return self == other or self < other


@dataclass
class VersionInfo:
    """API version information."""

    version: APIVersion
    released: date
    deprecated: bool = False
    sunset_date: Optional[date] = None
    description: str = ""
    changelog_url: Optional[str] = None

    @property
    def is_sunset(self) -> bool:
        """Check if version has passed sunset date."""
        if not self.sunset_date:
            return False
        return date.today() > self.sunset_date

    @property
    def days_until_sunset(self) -> Optional[int]:
        """Days until sunset, if applicable."""
        if not self.sunset_date:
            return None
        delta = self.sunset_date - date.today()
        return max(0, delta.days)


# Version registry
VERSION_INFO: Dict[APIVersion, VersionInfo] = {
    APIVersion.V1: VersionInfo(
        version=APIVersion.V1,
        released=date(2024, 1, 1),
        description="Initial stable API release",
    ),
    APIVersion.V2: VersionInfo(
        version=APIVersion.V2,
        released=date(2025, 1, 1),
        description="Enhanced response formats with metadata",
    ),
    APIVersion.V3: VersionInfo(
        version=APIVersion.V3,
        released=date(2026, 1, 1),
        description="GraphQL-style flexible queries",
    ),
}


# ============================================================================
# Version Extraction
# ============================================================================


def get_version_from_request(
    path: str,
    headers: Optional[Dict[str, str]] = None,
    default: APIVersion = APIVersion.V1,
) -> Tuple[APIVersion, str]:
    """
    Extract API version from request.

    Priority:
    1. URL path prefix (/api/v1/...)
    2. X-API-Version header
    3. Accept header version parameter
    4. Default version

    Returns:
        Tuple of (version, cleaned_path)
    """
    headers = headers or {}

    # 1. Check URL path prefix
    path_match = re.match(r"^/api/(v\d+)/(.*)$", path)
    if path_match:
        version_str, rest_path = path_match.groups()
        version = APIVersion.from_string(version_str)
        if version:
            return version, f"/api/{rest_path}"

    # 2. Check X-API-Version header
    header_version = headers.get("X-API-Version") or headers.get("x-api-version")
    if header_version:
        version = APIVersion.from_string(header_version)
        if version:
            return version, path

    # 3. Check Accept header
    accept = headers.get("Accept") or headers.get("accept") or ""
    version_match = re.search(r"version=(\d+)", accept)
    if version_match:
        version = APIVersion.from_string(version_match.group(1))
        if version:
            return version, path

    # 4. Default
    return default, path


# ============================================================================
# Versioned Router
# ============================================================================


@dataclass
class RouteEntry:
    """Single route entry."""

    path: str
    method: str
    handler: Callable
    version: APIVersion
    deprecated: bool = False
    sunset_date: Optional[date] = None
    replacement: Optional[str] = None


@dataclass
class VersionedRouter:
    """
    Router with API versioning support.

    Allows registering handlers for specific API versions and
    automatically routes requests to the appropriate handler.
    """

    routes: Dict[str, Dict[APIVersion, RouteEntry]] = field(default_factory=dict)
    default_version: APIVersion = APIVersion.V1
    strict_versioning: bool = False  # Require explicit version

    def _route_key(self, path: str, method: str) -> str:
        """Generate route key."""
        return f"{method.upper()}:{path}"

    def route(
        self,
        path: str,
        method: str = "GET",
        version: APIVersion = APIVersion.V1,
        deprecated: bool = False,
        sunset_date: Optional[date] = None,
        replacement: Optional[str] = None,
    ) -> Callable:
        """
        Decorator to register a versioned route.

        Args:
            path: URL path pattern
            method: HTTP method
            version: API version for this handler
            deprecated: Whether this endpoint is deprecated
            sunset_date: When this endpoint will be removed
            replacement: URL of replacement endpoint
        """
        def decorator(handler: Callable) -> Callable:
            key = self._route_key(path, method)

            if key not in self.routes:
                self.routes[key] = {}

            self.routes[key][version] = RouteEntry(
                path=path,
                method=method,
                handler=handler,
                version=version,
                deprecated=deprecated,
                sunset_date=sunset_date,
                replacement=replacement,
            )

            @wraps(handler)
            def wrapper(*args, **kwargs):
                return handler(*args, **kwargs)

            return wrapper

        return decorator

    def get(self, path: str, **kwargs) -> Callable:
        """Register GET route."""
        return self.route(path, method="GET", **kwargs)

    def post(self, path: str, **kwargs) -> Callable:
        """Register POST route."""
        return self.route(path, method="POST", **kwargs)

    def put(self, path: str, **kwargs) -> Callable:
        """Register PUT route."""
        return self.route(path, method="PUT", **kwargs)

    def delete(self, path: str, **kwargs) -> Callable:
        """Register DELETE route."""
        return self.route(path, method="DELETE", **kwargs)

    def patch(self, path: str, **kwargs) -> Callable:
        """Register PATCH route."""
        return self.route(path, method="PATCH", **kwargs)

    def resolve(
        self,
        path: str,
        method: str,
        version: APIVersion,
    ) -> Optional[RouteEntry]:
        """
        Resolve handler for request.

        Falls back to lower versions if exact version not found.
        """
        key = self._route_key(path, method)
        version_routes = self.routes.get(key)

        if not version_routes:
            return None

        # Try exact version first
        if version in version_routes:
            return version_routes[version]

        # If strict versioning, don't fallback
        if self.strict_versioning:
            return None

        # Fallback to lower versions
        for v in reversed(APIVersion.all_versions()):
            if v < version and v in version_routes:
                return version_routes[v]

        return None

    def get_all_routes(self) -> List[RouteEntry]:
        """Get all registered routes."""
        all_routes = []
        for versions in self.routes.values():
            all_routes.extend(versions.values())
        return all_routes

    def get_routes_for_version(self, version: APIVersion) -> List[RouteEntry]:
        """Get routes available for a specific version."""
        routes = []
        for key in self.routes:
            entry = self.resolve(
                self.routes[key][list(self.routes[key].keys())[0]].path,
                self.routes[key][list(self.routes[key].keys())[0]].method,
                version,
            )
            if entry:
                routes.append(entry)
        return routes


# ============================================================================
# Decorator Helpers
# ============================================================================


def version_route(
    version: APIVersion = APIVersion.V1,
    deprecated: bool = False,
    sunset_date: Optional[date] = None,
) -> Callable:
    """
    Mark a function as handling a specific API version.

    Can be used with any routing framework.

    Example:
        @version_route(APIVersion.V1)
        def my_handler(request):
            ...
    """
    def decorator(func: Callable) -> Callable:
        func._api_version = version
        func._deprecated = deprecated
        func._sunset_date = sunset_date

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator


# ============================================================================
# Response Helpers
# ============================================================================


def versioned_response(
    data: Any,
    version: APIVersion,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create version-appropriate response format.

    V1: Direct data
    V2+: Wrapped with meta
    """
    if version == APIVersion.V1:
        return data if isinstance(data, dict) else {"data": data}

    return {
        "data": data,
        "meta": {
            "version": version.value,
            "timestamp": datetime.utcnow().isoformat(),
            **(meta or {}),
        },
    }


def add_version_headers(
    headers: Dict[str, str],
    version: APIVersion,
    route: Optional[RouteEntry] = None,
) -> Dict[str, str]:
    """Add version-related headers to response."""
    headers["X-API-Version"] = version.value

    if route:
        if route.deprecated:
            headers["Deprecation"] = "true"
            headers["X-Deprecated"] = "true"

        if route.sunset_date:
            headers["Sunset"] = route.sunset_date.isoformat()

        if route.replacement:
            headers["Link"] = f'<{route.replacement}>; rel="successor-version"'

    return headers
