"""Versioning compatibility module for handlers.

Re-exports versioning utilities from the main server versioning module.
"""

from aragora.server.versioning.compat import (
    strip_version_prefix,
    normalize_path_version,
    is_versioned_path,
    is_legacy_path,
    get_path_version,
    extract_version,
    version_response_headers,
    VersionConfig,
    get_version_config,
    set_version_config,
)

__all__ = [
    "strip_version_prefix",
    "normalize_path_version",
    "is_versioned_path",
    "is_legacy_path",
    "get_path_version",
    "extract_version",
    "version_response_headers",
    "VersionConfig",
    "get_version_config",
    "set_version_config",
]
