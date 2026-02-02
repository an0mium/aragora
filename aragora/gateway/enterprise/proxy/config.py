"""
Proxy configuration dataclasses and enums.

Defines configuration structures for circuit breakers, retry policies,
bulkhead isolation, request sanitization, framework settings, and
global proxy configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class HealthStatus(str, Enum):
    """Health status for external frameworks."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class RetryStrategy(str, Enum):
    """Retry backoff strategies."""

    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    CONSTANT = "constant"


@dataclass
class CircuitBreakerSettings:
    """Circuit breaker settings for a framework.

    Attributes:
        failure_threshold: Consecutive failures before opening circuit.
        success_threshold: Successes in half-open state before closing.
        cooldown_seconds: Time circuit stays open before half-open.
        half_open_max_calls: Max concurrent calls in half-open state.
    """

    failure_threshold: int = 5
    success_threshold: int = 2
    cooldown_seconds: float = 60.0
    half_open_max_calls: int = 3

    def __post_init__(self) -> None:
        """Validate settings."""
        if self.failure_threshold < 1:
            raise ValueError("failure_threshold must be at least 1")
        if self.success_threshold < 1:
            raise ValueError("success_threshold must be at least 1")
        if self.cooldown_seconds <= 0:
            raise ValueError("cooldown_seconds must be positive")
        if self.half_open_max_calls < 1:
            raise ValueError("half_open_max_calls must be at least 1")


@dataclass
class RetrySettings:
    """Retry settings for a framework.

    Attributes:
        max_retries: Maximum retry attempts (not counting initial).
        base_delay: Base delay in seconds between retries.
        max_delay: Maximum delay cap in seconds.
        strategy: Backoff strategy to use.
        jitter: Whether to apply jitter to delays.
        retryable_status_codes: HTTP status codes to retry on.
    """

    max_retries: int = 3
    base_delay: float = 0.5
    max_delay: float = 30.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    jitter: bool = True
    retryable_status_codes: frozenset[int] = field(
        default_factory=lambda: frozenset({429, 500, 502, 503, 504})
    )

    def __post_init__(self) -> None:
        """Validate settings."""
        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        if self.base_delay <= 0:
            raise ValueError("base_delay must be positive")
        if self.max_delay < self.base_delay:
            raise ValueError("max_delay must be >= base_delay")


@dataclass
class BulkheadSettings:
    """Bulkhead isolation settings for a framework.

    Attributes:
        max_concurrent: Maximum concurrent requests allowed.
        wait_timeout: Seconds to wait for semaphore before failing.
    """

    max_concurrent: int = 50
    wait_timeout: float = 10.0

    def __post_init__(self) -> None:
        """Validate settings."""
        if self.max_concurrent < 1:
            raise ValueError("max_concurrent must be at least 1")
        if self.wait_timeout <= 0:
            raise ValueError("wait_timeout must be positive")


@dataclass
class SanitizationSettings:
    """Request/response sanitization settings.

    Attributes:
        redact_headers: Header names to redact from logs.
        redact_body_patterns: Regex patterns to redact from body.
        max_body_log_size: Maximum body size to log (bytes).
        strip_sensitive_headers: Headers to strip from outgoing requests.
    """

    redact_headers: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                "authorization",
                "x-api-key",
                "api-key",
                "cookie",
                "set-cookie",
                "x-auth-token",
            }
        )
    )
    redact_body_patterns: list[str] = field(
        default_factory=lambda: [
            r'"api_key"\s*:\s*"[^"]*"',
            r'"password"\s*:\s*"[^"]*"',
            r'"secret"\s*:\s*"[^"]*"',
            r'"token"\s*:\s*"[^"]*"',
        ]
    )
    max_body_log_size: int = 4096
    strip_sensitive_headers: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                "x-forwarded-for",
                "x-real-ip",
            }
        )
    )


@dataclass
class ExternalFrameworkConfig:
    """Configuration for an external framework.

    Attributes:
        base_url: Base URL for the framework API.
        timeout: Request timeout in seconds.
        connect_timeout: Connection timeout in seconds.
        circuit_breaker: Circuit breaker settings.
        retry: Retry settings.
        bulkhead: Bulkhead isolation settings.
        sanitization: Sanitization settings.
        default_headers: Default headers to include in requests.
        health_check_path: Path for health check probes.
        health_check_interval: Seconds between health checks.
        enabled: Whether this framework is enabled.
        metadata: Additional framework metadata.
    """

    base_url: str
    timeout: float = 30.0
    connect_timeout: float = 10.0
    circuit_breaker: CircuitBreakerSettings = field(default_factory=CircuitBreakerSettings)
    retry: RetrySettings = field(default_factory=RetrySettings)
    bulkhead: BulkheadSettings = field(default_factory=BulkheadSettings)
    sanitization: SanitizationSettings = field(default_factory=SanitizationSettings)
    default_headers: dict[str, str] = field(default_factory=dict)
    health_check_path: str | None = None
    health_check_interval: float = 30.0
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not self.base_url:
            raise ValueError("base_url is required")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        if self.connect_timeout <= 0:
            raise ValueError("connect_timeout must be positive")
        if self.health_check_interval <= 0:
            raise ValueError("health_check_interval must be positive")

        # Normalize base URL (remove trailing slash)
        self.base_url = self.base_url.rstrip("/")


@dataclass
class ProxyConfig:
    """Global proxy configuration.

    Attributes:
        default_timeout: Default request timeout in seconds.
        default_connect_timeout: Default connection timeout in seconds.
        max_connections: Maximum connections in the pool.
        max_connections_per_host: Maximum connections per host.
        keepalive_timeout: Connection keepalive timeout in seconds.
        enable_connection_pooling: Whether to use connection pooling.
        enable_audit_logging: Whether to log requests for audit.
        enable_metrics: Whether to emit metrics.
        tenant_header_name: Header name for tenant context.
        correlation_header_name: Header name for request correlation.
        user_agent: User-Agent header value.
    """

    default_timeout: float = 30.0
    default_connect_timeout: float = 10.0
    max_connections: int = 100
    max_connections_per_host: int = 10
    keepalive_timeout: float = 30.0
    enable_connection_pooling: bool = True
    enable_audit_logging: bool = True
    enable_metrics: bool = True
    tenant_header_name: str = "X-Tenant-ID"
    correlation_header_name: str = "X-Correlation-ID"
    user_agent: str = "Aragora-EnterpriseProxy/1.0"

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.max_connections < 1:
            raise ValueError("max_connections must be at least 1")
        if self.max_connections_per_host < 1:
            raise ValueError("max_connections_per_host must be at least 1")


__all__ = [
    # Enums
    "HealthStatus",
    "RetryStrategy",
    # Settings
    "CircuitBreakerSettings",
    "RetrySettings",
    "BulkheadSettings",
    "SanitizationSettings",
    # Configuration
    "ExternalFrameworkConfig",
    "ProxyConfig",
]
