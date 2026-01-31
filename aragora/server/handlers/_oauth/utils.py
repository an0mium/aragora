"""
OAuth utility functions.

Provides the ``_impl()`` helper that all mixin modules use to resolve config
functions and utility names from the ``_oauth_impl`` backward-compatibility
shim at runtime.  This ensures that ``unittest.mock.patch`` calls targeting
``_oauth_impl.<name>`` are visible to the actual mixin code.

Also holds the shared ``_oauth_limiter`` instance with security audit logging.
"""

from __future__ import annotations

import logging
import sys
from types import ModuleType

from aragora.server.handlers.utils.rate_limit import RateLimiter

logger = logging.getLogger(__name__)


class OAuthRateLimiter(RateLimiter):
    """Rate limiter with security audit logging for OAuth endpoints."""

    def is_allowed(self, key: str) -> bool:
        """Check if request is allowed, log security event if rate limited."""
        allowed = super().is_allowed(key)
        if not allowed:
            # Log warning with security context
            logger.warning(
                "AUTH RATE LIMIT: OAuth endpoint exceeded (ip=%s, limit=%d/min)",
                key,
                self.rpm,
            )
            # Log security audit event
            self._log_security_event(key)
        return allowed

    def _log_security_event(self, client_ip: str) -> None:
        """Log security audit event for rate limit hit on OAuth endpoint."""
        try:
            from aragora.audit.unified import audit_security

            audit_security(
                event_type="anomaly",
                actor_id=client_ip,
                ip_address=client_ip,
                reason="auth_rate_limit_exceeded:OAuth",
                details={
                    "endpoint": "OAuth",
                    "limiter": "oauth_limiter",
                    "limit_rpm": self.rpm,
                },
            )
        except ImportError:
            # Audit module not available - just log
            pass
        except Exception as e:
            logger.debug(f"Failed to log security audit event: {e}")


# Rate limiter for OAuth endpoints (20 requests per minute - auth attempts should be limited)
_oauth_limiter = OAuthRateLimiter(requests_per_minute=20)

# Module path constant for the backward-compat shim that tests patch against.
_IMPL_MODULE = "aragora.server.handlers._oauth_impl"


def _impl() -> ModuleType:
    """Return the ``_oauth_impl`` module lazily to avoid circular imports.

    All mixin modules call ``_impl().<name>`` instead of importing config
    functions directly so that ``unittest.mock.patch`` applied to
    ``_oauth_impl.<name>`` is visible to the running code.
    """
    return sys.modules[_IMPL_MODULE]
