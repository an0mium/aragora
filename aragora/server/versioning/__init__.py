"""
API Versioning for Aragora Server.

Provides:
- URL prefix versioning (/api/v1/, /api/v2/)
- Header-based version negotiation
- Deprecation warnings and sunset headers
- Version-specific route registration

Usage:
    from aragora.server.versioning import VersionedRouter, APIVersion

    router = VersionedRouter()

    @router.route("/debates", version=APIVersion.V1)
    async def list_debates_v1(request):
        ...

    @router.route("/debates", version=APIVersion.V2)
    async def list_debates_v2(request):
        # New response format
        ...
"""

from aragora.server.versioning.router import (
    APIVersion,
    VersionedRouter,
    VersionInfo,
    get_version_from_request,
    version_route,
)
from aragora.server.versioning.deprecation import (
    DeprecationLevel,
    DeprecationMiddleware,
    DeprecationWarning,
    deprecated,
    sunset_date,
)

__all__ = [
    # Version types
    "APIVersion",
    "VersionInfo",
    # Router
    "VersionedRouter",
    "version_route",
    "get_version_from_request",
    # Deprecation
    "DeprecationLevel",
    "DeprecationMiddleware",
    "DeprecationWarning",
    "deprecated",
    "sunset_date",
]
