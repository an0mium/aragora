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
from aragora.server.versioning.compat import (
    API_RELEASE_VERSION,
    VersionConfig,
    get_version_config,
    set_version_config,
    extract_version,
    version_response_headers,
    normalize_path_version,
    strip_version_prefix,
    is_versioned_path,
    is_legacy_path,
    get_path_version,
)
from aragora.server.versioning.constants import (
    CURRENT_API_VERSION,
    MIGRATION_DOCS_URL,
    V1_DEPRECATION_ANNOUNCED,
    V1_DEPRECATION_TIMESTAMP,
    V1_SUNSET_DATE,
    V1_SUNSET_HTTP_DATE,
    V1_SUNSET_ISO,
    days_until_v1_sunset,
    deprecation_level,
    is_v1_sunset,
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
    # Compat layer
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
    # V1 Sunset constants
    "CURRENT_API_VERSION",
    "MIGRATION_DOCS_URL",
    "V1_DEPRECATION_ANNOUNCED",
    "V1_DEPRECATION_TIMESTAMP",
    "V1_SUNSET_DATE",
    "V1_SUNSET_HTTP_DATE",
    "V1_SUNSET_ISO",
    "days_until_v1_sunset",
    "deprecation_level",
    "is_v1_sunset",
]
