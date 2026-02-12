"""
Proxy request/response models and hook type definitions.

Defines ProxyRequest, ProxyResponse dataclasses for representing proxied
HTTP requests and responses, along with hook type signatures for
pre-request, post-request, and error handling.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any
from collections.abc import Awaitable, Callable

from .config import HealthStatus


@dataclass
class ProxyRequest:
    """Represents a proxy request for hook processing.

    Attributes:
        framework: Target framework name.
        method: HTTP method (GET, POST, etc.).
        url: Full request URL.
        headers: Request headers.
        body: Request body (if any).
        tenant_id: Tenant identifier.
        correlation_id: Request correlation ID.
        auth_context: Authentication context.
        metadata: Additional request metadata.
        timestamp: Request timestamp.
    """

    framework: str
    method: str
    url: str
    headers: dict[str, str]
    body: bytes | None = None
    tenant_id: str | None = None
    correlation_id: str | None = None
    auth_context: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def body_hash(self) -> str | None:
        """Get SHA-256 hash of request body for audit."""
        if self.body is None:
            return None
        return hashlib.sha256(self.body).hexdigest()


@dataclass
class ProxyResponse:
    """Represents a proxy response for hook processing.

    Attributes:
        status_code: HTTP status code.
        headers: Response headers.
        body: Response body.
        elapsed_ms: Request duration in milliseconds.
        framework: Source framework name.
        correlation_id: Request correlation ID.
        from_cache: Whether response was from cache.
        metadata: Additional response metadata.
        timestamp: Response timestamp.
    """

    status_code: int
    headers: dict[str, str]
    body: bytes
    elapsed_ms: float
    framework: str
    correlation_id: str | None = None
    from_cache: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    @property
    def is_success(self) -> bool:
        """Check if response indicates success (2xx status)."""
        return 200 <= self.status_code < 300

    @property
    def is_client_error(self) -> bool:
        """Check if response indicates client error (4xx status)."""
        return 400 <= self.status_code < 500

    @property
    def is_server_error(self) -> bool:
        """Check if response indicates server error (5xx status)."""
        return 500 <= self.status_code < 600

    def body_hash(self) -> str:
        """Get SHA-256 hash of response body for audit."""
        return hashlib.sha256(self.body).hexdigest()


@dataclass
class HealthCheckResult:
    """Result of a health check probe.

    Attributes:
        framework: Framework name.
        status: Health status.
        latency_ms: Probe latency in milliseconds.
        last_check: Timestamp of last check.
        error: Error message if unhealthy.
        consecutive_failures: Number of consecutive failures.
    """

    framework: str
    status: HealthStatus
    latency_ms: float | None = None
    last_check: float = field(default_factory=time.time)
    error: str | None = None
    consecutive_failures: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize for API response."""
        return {
            "framework": self.framework,
            "status": self.status.value,
            "latency_ms": self.latency_ms,
            "last_check": self.last_check,
            "error": self.error,
            "consecutive_failures": self.consecutive_failures,
        }


# Hook type definitions
PreRequestHook = Callable[[ProxyRequest], Awaitable[ProxyRequest | None]]
PostRequestHook = Callable[[ProxyRequest, ProxyResponse], Awaitable[None]]
ErrorHook = Callable[[ProxyRequest, Exception], Awaitable[None]]


__all__ = [
    "ProxyRequest",
    "ProxyResponse",
    "HealthCheckResult",
    "PreRequestHook",
    "PostRequestHook",
    "ErrorHook",
]
