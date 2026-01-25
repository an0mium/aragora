"""
Version information for aragora package.

This module provides semantic versioning information following PEP 440.
Import VERSION_INFO for programmatic access to version components.
"""

from __future__ import annotations

# Version components
VERSION_MAJOR = 2
VERSION_MINOR = 4
VERSION_PATCH = 0
VERSION_SUFFIX = ""  # e.g., "a1", "b2", "rc1", or "" for final

# Construct version string
VERSION_INFO = (VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH)
__version__ = f"{VERSION_MAJOR}.{VERSION_MINOR}.{VERSION_PATCH}"
if VERSION_SUFFIX:
    __version__ += VERSION_SUFFIX

# Release date (ISO 8601 format)
RELEASE_DATE = "2026-01-24"

# Package metadata
PACKAGE_NAME = "aragora"
AUTHOR = "Agora Contributors"
LICENSE = "MIT"
REPOSITORY = "https://github.com/an0mium/aragora"


def get_version() -> str:
    """Return the current version string."""
    return __version__


def get_version_tuple() -> tuple[int, int, int]:
    """Return version as a tuple of (major, minor, patch)."""
    return VERSION_INFO


__all__ = [
    "__version__",
    "VERSION_INFO",
    "VERSION_MAJOR",
    "VERSION_MINOR",
    "VERSION_PATCH",
    "VERSION_SUFFIX",
    "RELEASE_DATE",
    "PACKAGE_NAME",
    "get_version",
    "get_version_tuple",
]
