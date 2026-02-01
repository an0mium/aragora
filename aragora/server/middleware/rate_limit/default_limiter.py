"""
Default rate limiter helpers.

Provides a conservative, always-on rate limiting layer for handlers that
do not explicitly use the rate_limit decorators.
"""

from __future__ import annotations

import logging
from typing import Any

from .base import DEFAULT_RATE_LIMIT, _extract_client_ip, sanitize_rate_limit_key_component
from .limiter import RateLimitResult
from .registry import get_rate_limiter

logger = logging.getLogger(__name__)

# Conservative defaults by auth tier
DEFAULT_RATE_LIMITS: dict[str, int] = {
    "anonymous": DEFAULT_RATE_LIMIT,
    "authenticated": max(DEFAULT_RATE_LIMIT, int(DEFAULT_RATE_LIMIT * 2)),
    "admin": max(DEFAULT_RATE_LIMIT, int(DEFAULT_RATE_LIMIT * 5)),
}


def determine_auth_tier(handler: Any) -> str:
    """Determine auth tier for a request handler."""
    auth_ctx = getattr(handler, "_auth_context", None)
    if auth_ctx is None:
        return "anonymous"

    roles = {str(role).lower() for role in getattr(auth_ctx, "roles", [])}
    if {"admin", "owner", "superadmin", "sysadmin"} & roles:
        return "admin"
    return "authenticated"


def _extract_request_path(handler: Any) -> str | None:
    path = getattr(handler, "path", None)
    if not path:
        return None
    return path.split("?", 1)[0]


def _extract_token(handler: Any) -> str | None:
    headers = getattr(handler, "headers", None)
    if not headers:
        return None
    token = headers.get("Authorization") or headers.get("authorization")
    if not token:
        return None
    return sanitize_rate_limit_key_component(str(token))


def _extract_tenant_id(handler: Any) -> str | None:
    auth_ctx = getattr(handler, "_auth_context", None)
    if auth_ctx is not None:
        tenant = getattr(auth_ctx, "org_id", None) or getattr(auth_ctx, "workspace_id", None)
        if tenant:
            return sanitize_rate_limit_key_component(str(tenant))

    headers = getattr(handler, "headers", None)
    if headers:
        tenant = headers.get("X-Tenant-ID") or headers.get("X-Workspace-ID")
        if tenant:
            return sanitize_rate_limit_key_component(str(tenant))
    return None


def get_handler_default_rpm(handler: Any) -> int:
    """Get default requests-per-minute for the handler's auth tier."""
    tier = determine_auth_tier(handler)
    return DEFAULT_RATE_LIMITS.get(tier, DEFAULT_RATE_LIMIT)


def handler_has_rate_limit_decorator(handler: Any, handler_method_name: str) -> bool:
    """Check if the handler method has an explicit rate limit decorator."""
    if not handler or not handler_method_name:
        return False
    method = getattr(handler, handler_method_name, None)
    if method is None:
        return False
    return bool(getattr(method, "_rate_limited", False))


def should_apply_default_rate_limit(handler: Any, handler_method_name: str) -> bool:
    """Apply default limiter only when no explicit decorator is present."""
    if getattr(handler, "_skip_default_rate_limit", False):
        return False
    return not handler_has_rate_limit_decorator(handler, handler_method_name)


class DefaultRateLimiter:
    """Wrapper around the shared default rate limiter."""

    def __init__(self, rate_limits: dict[str, int] | None = None) -> None:
        self._limits = rate_limits or DEFAULT_RATE_LIMITS

    def allow(self, handler: Any) -> RateLimitResult:
        tier = determine_auth_tier(handler)
        _ = self._limits.get(tier, DEFAULT_RATE_LIMIT)

        limiter = get_rate_limiter()

        remote_ip = "anonymous"
        if hasattr(handler, "client_address"):
            addr = handler.client_address
            if isinstance(addr, tuple) and len(addr) >= 1:
                remote_ip = str(addr[0])

        headers = {}
        if hasattr(handler, "headers"):
            headers = {
                "X-Forwarded-For": handler.headers.get("X-Forwarded-For", ""),
                "X-Real-IP": handler.headers.get("X-Real-IP", ""),
            }

        client_ip = _extract_client_ip(headers, remote_ip)
        endpoint = _extract_request_path(handler)
        token = _extract_token(handler)
        tenant_id = _extract_tenant_id(handler)

        return limiter.allow(
            client_ip,
            endpoint=endpoint,
            token=token,
            tenant_id=tenant_id,
        )


_default_limiter: DefaultRateLimiter | None = None


def get_default_rate_limiter() -> DefaultRateLimiter:
    """Get global DefaultRateLimiter instance."""
    global _default_limiter
    if _default_limiter is None:
        _default_limiter = DefaultRateLimiter()
    return _default_limiter


def reset_default_rate_limiter() -> None:
    """Reset global DefaultRateLimiter instance."""
    global _default_limiter
    _default_limiter = None


def check_default_rate_limit(handler: Any) -> RateLimitResult:
    """Check default rate limit for a handler."""
    return get_default_rate_limiter().allow(handler)


__all__ = [
    "DEFAULT_RATE_LIMITS",
    "DefaultRateLimiter",
    "get_default_rate_limiter",
    "reset_default_rate_limiter",
    "determine_auth_tier",
    "check_default_rate_limit",
    "handler_has_rate_limit_decorator",
    "should_apply_default_rate_limit",
    "get_handler_default_rpm",
]
