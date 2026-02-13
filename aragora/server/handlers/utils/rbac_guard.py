"""
RBAC Import Guard - Fail-closed RBAC availability check.

SECURITY: When the RBAC module fails to import, handlers must NOT silently
skip permission checks. In production (ARAGORA_ENV=production), a missing
RBAC module means access is DENIED. In dev/test, the permissive fallback
is acceptable to avoid blocking local development.

Usage (replaces the common try/except ImportError pattern in handlers):

    from aragora.server.handlers.utils.rbac_guard import (
        rbac_available,
        rbac_fail_closed,
    )

    # Check if RBAC is available
    if not rbac_available():
        # In production: deny access
        # In dev/test: skip RBAC check (permissive)
        if rbac_fail_closed():
            return error_response("Service unavailable: access control module not loaded", 503)
        # else: dev/test mode, continue without RBAC
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# Cache the result of the RBAC import attempt
_rbac_import_attempted = False
_rbac_import_success = False


def _attempt_rbac_import() -> bool:
    """Attempt to import the RBAC module. Cached after first call."""
    global _rbac_import_attempted, _rbac_import_success
    if _rbac_import_attempted:
        return _rbac_import_success
    _rbac_import_attempted = True
    try:
        from aragora.rbac.checker import check_permission  # noqa: F401

        _rbac_import_success = True
    except ImportError:
        _rbac_import_success = False
        logger.warning("RBAC module failed to import")
    return _rbac_import_success


def rbac_available() -> bool:
    """Check whether the RBAC module is importable."""
    return _attempt_rbac_import()


def is_production_env() -> bool:
    """Check if we are running in a production-like environment."""
    env = os.getenv("ARAGORA_ENV", "").lower()
    return env in ("production", "prod", "staging", "live")


def rbac_fail_closed() -> bool:
    """Return True if RBAC failures should deny access (fail closed).

    In production environments, RBAC must be available. If it is not,
    access is denied. In dev/test environments, the permissive fallback
    (skip RBAC) is acceptable.

    Returns:
        True if the handler should deny access when RBAC is unavailable.
    """
    if rbac_available():
        return False  # RBAC is available, no need to fail closed
    # RBAC is NOT available
    if is_production_env():
        logger.error(
            "SECURITY: RBAC module unavailable in production environment. "
            "Access will be DENIED (fail-closed policy). "
            "Ensure aragora.rbac is properly installed."
        )
        return True
    # Dev/test: permissive fallback
    return False


__all__ = [
    "rbac_available",
    "rbac_fail_closed",
    "is_production_env",
]
