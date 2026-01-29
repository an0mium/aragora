"""
OAuth utility functions.

Module-level helpers for redirect URL validation, state management,
and rate limiting used across all OAuth provider mixins.
"""

from __future__ import annotations

import logging
from typing import Any

from aragora.server.handlers.oauth.config import _get_allowed_redirect_hosts
from aragora.server.oauth_state_store import (
    validate_oauth_state as _validate_state_internal,
)
from aragora.server.handlers.utils.rate_limit import RateLimiter

logger = logging.getLogger(__name__)

# Rate limiter for OAuth endpoints (20 requests per minute - auth attempts should be limited)
_oauth_limiter = RateLimiter(requests_per_minute=20)


def _validate_state(state: str) -> dict[str, Any] | None:
    """Validate and consume OAuth state token."""
    return _validate_state_internal(state)


def _validate_redirect_url(redirect_url: str) -> bool:
    """Validate redirect URL against allowed hosts (uses module-level _get_allowed_redirect_hosts)."""
    from urllib.parse import urlparse

    try:
        parsed = urlparse(redirect_url)
        if parsed.scheme not in ("http", "https"):
            logger.warning(f"oauth_redirect_blocked: scheme={parsed.scheme} not allowed")
            return False
        host = parsed.hostname
        if not host:
            return False
        host = host.lower()
        allowed_hosts = _get_allowed_redirect_hosts()
        if host in allowed_hosts:
            return True
        for allowed in allowed_hosts:
            if host.endswith(f".{allowed}"):
                return True
        logger.warning(f"oauth_redirect_blocked: host={host} not in allowlist")
        return False
    except Exception as e:
        logger.warning(f"oauth_redirect_validation_error: {e}")
        return False
