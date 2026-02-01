"""
Tenant-based rate limiter implementation.

Provides per-tenant rate limiting to enforce fair usage across organizations
or workspaces.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional

from .base import (
    BURST_MULTIPLIER,
    DEFAULT_RATE_LIMIT,
    _extract_client_ip,
    sanitize_rate_limit_key_component,
)
from .bucket import TokenBucket
from .limiter import RateLimitResult

logger = logging.getLogger(__name__)


# Per-tenant rate limits (requests per minute)
DEFAULT_TENANT_RATE_LIMITS: dict[str, int] = {
    "default": DEFAULT_RATE_LIMIT,
}


@dataclass
class TenantRateLimitConfig:
    """Configuration for tenant rate limiting."""

    requests_per_minute: int = DEFAULT_RATE_LIMIT
    burst_size: int | None = None
    max_tenants: int = 10000


class TenantRateLimiter:
    """
    Per-tenant rate limiter.

    Uses a token bucket per (action, tenant_id). This allows different
    limits for different operations while isolating tenants.
    """

    def __init__(
        self,
        action_limits: Optional[dict[str, int]] = None,
        default_limit: int = DEFAULT_RATE_LIMIT,
        max_tenants: int = 10000,
    ) -> None:
        self.action_limits = action_limits or DEFAULT_TENANT_RATE_LIMITS
        self.default_limit = default_limit
        self.max_tenants = max_tenants

        self._tenant_buckets: dict[str, OrderedDict[str, TokenBucket]] = {}
        self._lock = threading.Lock()
        self._last_cleanup = time.monotonic()

    def get_action_limit(self, action: str) -> int:
        """Get rate limit for an action."""
        return self.action_limits.get(action, self.default_limit)

    def allow(self, tenant_id: str, action: str = "default") -> RateLimitResult:
        """Check if tenant can perform action."""
        limit = self.get_action_limit(action)
        burst = int(limit * BURST_MULTIPLIER)

        with self._lock:
            if action not in self._tenant_buckets:
                self._tenant_buckets[action] = OrderedDict()

            buckets = self._tenant_buckets[action]

            if tenant_id in buckets:
                buckets.move_to_end(tenant_id)
                bucket = buckets[tenant_id]
            else:
                max_per_action = self.max_tenants // max(1, len(self._tenant_buckets))
                while len(buckets) >= max_per_action:
                    buckets.popitem(last=False)

                bucket = TokenBucket(limit, burst_size=burst)
                buckets[tenant_id] = bucket

        allowed = bucket.consume(1)

        return RateLimitResult(
            allowed=allowed,
            remaining=bucket.remaining,
            limit=limit,
            retry_after=bucket.get_retry_after() if not allowed else 0,
            key=f"tenant:{tenant_id}:{action}",
        )

    def cleanup(self, max_age_seconds: int = 600) -> int:
        """Remove stale tenant entries."""
        with self._lock:
            now = time.monotonic()
            removed = 0

            for action, buckets in list(self._tenant_buckets.items()):
                stale = [
                    tenant
                    for tenant, bucket in buckets.items()
                    if now - bucket.last_refill > max_age_seconds
                ]
                for tenant in stale:
                    del buckets[tenant]
                    removed += 1

                if not buckets:
                    del self._tenant_buckets[action]

            return removed

    def reset(self) -> None:
        """Reset all tenant rate limiter state."""
        with self._lock:
            self._tenant_buckets.clear()


_tenant_limiter: TenantRateLimiter | None = None


def get_tenant_rate_limiter() -> TenantRateLimiter:
    """Get the global tenant rate limiter instance."""
    global _tenant_limiter
    if _tenant_limiter is None:
        _tenant_limiter = TenantRateLimiter()
    return _tenant_limiter


def reset_tenant_rate_limiter() -> None:
    """Reset the global tenant rate limiter."""
    global _tenant_limiter
    _tenant_limiter = None


def _extract_tenant_id(handler: Any) -> str:
    auth_ctx = getattr(handler, "_auth_context", None)
    tenant = None
    if auth_ctx is not None:
        tenant = getattr(auth_ctx, "org_id", None) or getattr(auth_ctx, "workspace_id", None)

    headers = getattr(handler, "headers", None)
    if headers and tenant is None:
        tenant = headers.get("X-Tenant-ID") or headers.get("X-Workspace-ID")

    if tenant:
        return sanitize_rate_limit_key_component(str(tenant))

    # Fall back to IP-based key when tenant context is absent
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

    return sanitize_rate_limit_key_component(_extract_client_ip(headers, remote_ip))


def check_tenant_rate_limit(handler: Any, action: str = "default") -> RateLimitResult:
    """Check tenant rate limit for handler."""
    limiter = get_tenant_rate_limiter()
    tenant_id = _extract_tenant_id(handler)
    return limiter.allow(tenant_id, action)


__all__ = [
    "DEFAULT_TENANT_RATE_LIMITS",
    "TenantRateLimiter",
    "TenantRateLimitConfig",
    "get_tenant_rate_limiter",
    "reset_tenant_rate_limiter",
    "check_tenant_rate_limit",
]
