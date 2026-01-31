"""
Enterprise Security Proxy for External Framework Integration.

Provides a secure, resilient proxy layer for all external framework calls with:
- Connection pooling and timeout management
- Circuit breaker per external framework
- Retry with exponential backoff
- Bulkhead isolation (max concurrent requests)
- Pre/post request hooks for auth and audit
- Request/response sanitization
- Tenant context header injection

Usage:
    from aragora.gateway.enterprise.proxy import (
        EnterpriseProxy,
        ExternalFrameworkConfig,
        ProxyConfig,
    )

    # Configure external frameworks
    proxy = EnterpriseProxy(
        config=ProxyConfig(
            default_timeout=30.0,
            max_connections=100,
        ),
        frameworks={
            "openai": ExternalFrameworkConfig(
                base_url="https://api.openai.com",
                timeout=60.0,
                max_retries=3,
            ),
            "anthropic": ExternalFrameworkConfig(
                base_url="https://api.anthropic.com",
                timeout=120.0,
                circuit_breaker_threshold=3,
            ),
        },
    )

    # Make proxied requests
    async with proxy:
        response = await proxy.request(
            framework="openai",
            method="POST",
            path="/v1/chat/completions",
            json={"model": "gpt-4", "messages": [...]},
            auth_context=auth_ctx,
            tenant_id="tenant-123",
        )

    # Register hooks for custom auth/audit
    proxy.add_pre_request_hook(verify_auth_hook)
    proxy.add_post_request_hook(audit_log_hook)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Mapping,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================


class ProxyError(Exception):
    """Base exception for proxy errors."""

    def __init__(
        self,
        message: str,
        code: str = "PROXY_ERROR",
        framework: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.framework = framework
        self.details = details or {}


class CircuitOpenError(ProxyError):
    """Raised when circuit breaker is open for a framework."""

    def __init__(
        self,
        framework: str,
        cooldown_remaining: float,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            f"Circuit breaker open for '{framework}'. Retry in {cooldown_remaining:.1f}s",
            code="CIRCUIT_OPEN",
            framework=framework,
            details=details or {"cooldown_remaining": cooldown_remaining},
        )
        self.cooldown_remaining = cooldown_remaining


class BulkheadFullError(ProxyError):
    """Raised when bulkhead semaphore is full."""

    def __init__(
        self,
        framework: str,
        max_concurrent: int,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            f"Bulkhead full for '{framework}'. Max concurrent requests: {max_concurrent}",
            code="BULKHEAD_FULL",
            framework=framework,
            details=details or {"max_concurrent": max_concurrent},
        )
        self.max_concurrent = max_concurrent


class RequestTimeoutError(ProxyError):
    """Raised when request times out."""

    def __init__(
        self,
        framework: str,
        timeout: float,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            f"Request to '{framework}' timed out after {timeout:.1f}s",
            code="REQUEST_TIMEOUT",
            framework=framework,
            details=details or {"timeout": timeout},
        )
        self.timeout = timeout


class FrameworkNotConfiguredError(ProxyError):
    """Raised when framework is not configured."""

    def __init__(self, framework: str) -> None:
        super().__init__(
            f"Framework '{framework}' is not configured",
            code="FRAMEWORK_NOT_CONFIGURED",
            framework=framework,
        )


class SanitizationError(ProxyError):
    """Raised when request/response sanitization fails."""

    def __init__(
        self,
        message: str,
        framework: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            code="SANITIZATION_ERROR",
            framework=framework,
            details=details,
        )


# =============================================================================
# Enums
# =============================================================================


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


# =============================================================================
# Configuration Dataclasses
# =============================================================================


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


# =============================================================================
# Hook Types
# =============================================================================


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


# Hook type definitions
PreRequestHook = Callable[[ProxyRequest], Awaitable[ProxyRequest | None]]
PostRequestHook = Callable[[ProxyRequest, ProxyResponse], Awaitable[None]]
ErrorHook = Callable[[ProxyRequest, Exception], Awaitable[None]]


# =============================================================================
# Circuit Breaker Implementation
# =============================================================================


class FrameworkCircuitBreaker:
    """Circuit breaker for a single external framework.

    Implements the circuit breaker pattern with three states:
    - CLOSED: Normal operation, requests allowed
    - OPEN: After failure threshold, requests blocked
    - HALF-OPEN: After cooldown, trial requests allowed
    """

    def __init__(self, framework: str, settings: CircuitBreakerSettings) -> None:
        """Initialize circuit breaker.

        Args:
            framework: Framework name for logging.
            settings: Circuit breaker settings.
        """
        self.framework = framework
        self.settings = settings

        self._failures = 0
        self._successes = 0
        self._open_at: float | None = None
        self._half_open_calls = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> str:
        """Get current circuit state."""
        if self._open_at is None:
            return "closed"
        elapsed = time.time() - self._open_at
        if elapsed >= self.settings.cooldown_seconds:
            return "half-open"
        return "open"

    @property
    def is_open(self) -> bool:
        """Check if circuit is open."""
        return self.state == "open"

    @property
    def cooldown_remaining(self) -> float:
        """Get remaining cooldown time in seconds."""
        if self._open_at is None:
            return 0.0
        elapsed = time.time() - self._open_at
        remaining = self.settings.cooldown_seconds - elapsed
        return max(0.0, remaining)

    async def can_proceed(self) -> bool:
        """Check if request can proceed."""
        async with self._lock:
            state = self.state

            if state == "closed":
                return True

            if state == "half-open":
                if self._half_open_calls < self.settings.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False

            return False  # open

    async def record_success(self) -> None:
        """Record a successful request."""
        async with self._lock:
            self._failures = 0

            if self._open_at is not None:
                self._successes += 1
                if self._successes >= self.settings.success_threshold:
                    logger.info(f"Circuit breaker CLOSED for {self.framework}")
                    self._open_at = None
                    self._successes = 0
                    self._half_open_calls = 0

    async def record_failure(self) -> bool:
        """Record a failed request. Returns True if circuit just opened."""
        async with self._lock:
            self._failures += 1
            self._successes = 0

            if self._failures >= self.settings.failure_threshold:
                if self._open_at is None:
                    self._open_at = time.time()
                    self._half_open_calls = 0
                    logger.warning(
                        f"Circuit breaker OPEN for {self.framework} after {self._failures} failures"
                    )
                    return True

            return False

    async def reset(self) -> None:
        """Reset circuit breaker state."""
        async with self._lock:
            self._failures = 0
            self._successes = 0
            self._open_at = None
            self._half_open_calls = 0
            logger.info(f"Circuit breaker reset for {self.framework}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize state for monitoring."""
        return {
            "framework": self.framework,
            "state": self.state,
            "failures": self._failures,
            "successes": self._successes,
            "cooldown_remaining": self.cooldown_remaining,
            "half_open_calls": self._half_open_calls,
        }


# =============================================================================
# Bulkhead Implementation
# =============================================================================


class FrameworkBulkhead:
    """Bulkhead isolation for a single external framework.

    Limits concurrent requests to prevent resource exhaustion.
    """

    def __init__(self, framework: str, settings: BulkheadSettings) -> None:
        """Initialize bulkhead.

        Args:
            framework: Framework name for logging.
            settings: Bulkhead settings.
        """
        self.framework = framework
        self.settings = settings
        self._semaphore = asyncio.Semaphore(settings.max_concurrent)
        self._active = 0
        self._lock = asyncio.Lock()

    @property
    def active_count(self) -> int:
        """Get number of active requests."""
        return self._active

    @property
    def available_slots(self) -> int:
        """Get number of available request slots."""
        return self.settings.max_concurrent - self._active

    @asynccontextmanager
    async def acquire(self) -> AsyncGenerator[None, None]:
        """Acquire a slot in the bulkhead.

        Raises:
            BulkheadFullError: If no slots available within timeout.
        """
        try:
            acquired = await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=self.settings.wait_timeout,
            )
            if not acquired:
                raise BulkheadFullError(
                    self.framework,
                    self.settings.max_concurrent,
                )
        except asyncio.TimeoutError:
            raise BulkheadFullError(
                self.framework,
                self.settings.max_concurrent,
                {"wait_timeout": self.settings.wait_timeout},
            )

        async with self._lock:
            self._active += 1

        try:
            yield
        finally:
            self._semaphore.release()
            async with self._lock:
                self._active -= 1

    def to_dict(self) -> dict[str, Any]:
        """Serialize state for monitoring."""
        return {
            "framework": self.framework,
            "active": self._active,
            "max_concurrent": self.settings.max_concurrent,
            "available_slots": self.available_slots,
        }


# =============================================================================
# Health Check Implementation
# =============================================================================


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


# =============================================================================
# Request Sanitizer
# =============================================================================


class RequestSanitizer:
    """Sanitizes requests and responses for security and logging."""

    def __init__(self, settings: SanitizationSettings) -> None:
        """Initialize sanitizer.

        Args:
            settings: Sanitization settings.
        """
        self.settings = settings
        self._compiled_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in settings.redact_body_patterns
        ]

    def sanitize_headers(
        self,
        headers: Mapping[str, str],
        for_logging: bool = False,
    ) -> dict[str, str]:
        """Sanitize headers.

        Args:
            headers: Headers to sanitize.
            for_logging: If True, redact sensitive values for logging.

        Returns:
            Sanitized headers dictionary.
        """
        result = {}
        redact_headers = {h.lower() for h in self.settings.redact_headers}
        strip_headers = {h.lower() for h in self.settings.strip_sensitive_headers}

        for key, value in headers.items():
            key_lower = key.lower()

            # Strip sensitive headers from outgoing requests
            if key_lower in strip_headers:
                continue

            # Redact sensitive values for logging
            if for_logging and key_lower in redact_headers:
                result[key] = "[REDACTED]"
            else:
                result[key] = value

        return result

    def sanitize_body_for_logging(self, body: bytes | None) -> str:
        """Sanitize body content for logging.

        Args:
            body: Request/response body.

        Returns:
            Sanitized body string for logging.
        """
        if body is None:
            return ""

        # Truncate if too large
        if len(body) > self.settings.max_body_log_size:
            try:
                text = body[: self.settings.max_body_log_size].decode("utf-8", errors="replace")
            except Exception as e:
                logger.debug(f"Failed to decode truncated body as UTF-8: {type(e).__name__}: {e}")
                text = f"[Binary data, {len(body)} bytes]"
            return f"{text}... [truncated, {len(body)} bytes total]"

        try:
            text = body.decode("utf-8", errors="replace")
        except Exception as e:
            logger.debug(f"Failed to decode body as UTF-8: {type(e).__name__}: {e}")
            return f"[Binary data, {len(body)} bytes]"

        # Apply redaction patterns
        for pattern in self._compiled_patterns:
            text = pattern.sub('"[REDACTED]"', text)

        return text

    def validate_request(self, request: ProxyRequest) -> None:
        """Validate request for security issues.

        Args:
            request: Request to validate.

        Raises:
            SanitizationError: If validation fails.
        """
        # Check for header injection attempts
        for key, value in request.headers.items():
            if "\n" in key or "\r" in key or "\n" in value or "\r" in value:
                raise SanitizationError(
                    "Header injection attempt detected",
                    framework=request.framework,
                    details={"header": key},
                )

        # Check URL for common injection patterns
        if request.url:
            suspicious_patterns = ["<script", "javascript:", "data:", "file://"]
            url_lower = request.url.lower()
            for pattern in suspicious_patterns:
                if pattern in url_lower:
                    raise SanitizationError(
                        "Suspicious URL pattern detected",
                        framework=request.framework,
                        details={"pattern": pattern},
                    )


# =============================================================================
# Enterprise Proxy Implementation
# =============================================================================


class EnterpriseProxy:
    """
    Enterprise security proxy for external framework integration.

    Provides a secure, resilient proxy layer for all external framework
    calls with connection pooling, circuit breakers, retries, bulkhead
    isolation, and comprehensive security hooks.

    Example:
        >>> proxy = EnterpriseProxy(
        ...     config=ProxyConfig(max_connections=100),
        ...     frameworks={
        ...         "openai": ExternalFrameworkConfig(
        ...             base_url="https://api.openai.com",
        ...             timeout=60.0,
        ...         ),
        ...     },
        ... )
        >>> async with proxy:
        ...     response = await proxy.request(
        ...         framework="openai",
        ...         method="POST",
        ...         path="/v1/chat/completions",
        ...         json={"model": "gpt-4"},
        ...     )
    """

    def __init__(
        self,
        config: ProxyConfig | None = None,
        frameworks: dict[str, ExternalFrameworkConfig] | None = None,
    ) -> None:
        """Initialize enterprise proxy.

        Args:
            config: Global proxy configuration.
            frameworks: Per-framework configurations.
        """
        self.config = config or ProxyConfig()
        self._frameworks: dict[str, ExternalFrameworkConfig] = frameworks or {}

        # Circuit breakers per framework
        self._circuit_breakers: dict[str, FrameworkCircuitBreaker] = {}

        # Bulkheads per framework
        self._bulkheads: dict[str, FrameworkBulkhead] = {}

        # Sanitizers per framework
        self._sanitizers: dict[str, RequestSanitizer] = {}

        # Health check results
        self._health_results: dict[str, HealthCheckResult] = {}

        # Hooks
        self._pre_request_hooks: list[PreRequestHook] = []
        self._post_request_hooks: list[PostRequestHook] = []
        self._error_hooks: list[ErrorHook] = []

        # HTTP session (lazy initialized)
        self._session: Any = None  # aiohttp.ClientSession
        self._session_lock = asyncio.Lock()

        # Health check task
        self._health_check_task: asyncio.Task | None = None
        self._shutdown_event = asyncio.Event()

        # Initialize framework components
        for name, fw_config in self._frameworks.items():
            self._init_framework(name, fw_config)

        logger.info(f"EnterpriseProxy initialized with {len(self._frameworks)} frameworks")

    def _init_framework(self, name: str, config: ExternalFrameworkConfig) -> None:
        """Initialize components for a framework.

        Args:
            name: Framework name.
            config: Framework configuration.
        """
        self._circuit_breakers[name] = FrameworkCircuitBreaker(name, config.circuit_breaker)
        self._bulkheads[name] = FrameworkBulkhead(name, config.bulkhead)
        self._sanitizers[name] = RequestSanitizer(config.sanitization)
        self._health_results[name] = HealthCheckResult(
            framework=name,
            status=HealthStatus.UNKNOWN,
        )

    # =========================================================================
    # Lifecycle Management
    # =========================================================================

    async def __aenter__(self) -> "EnterpriseProxy":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.shutdown()

    async def start(self) -> None:
        """Start the proxy and initialize resources."""
        await self._ensure_session()

        # Start health check background task
        if any(fw.health_check_path for fw in self._frameworks.values()):
            self._health_check_task = asyncio.create_task(self._health_check_loop())

        logger.info("EnterpriseProxy started")

    async def shutdown(self) -> None:
        """Shutdown the proxy and cleanup resources."""
        self._shutdown_event.set()

        # Cancel health check task
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        # Close HTTP session
        if self._session:
            await self._session.close()
            self._session = None

        logger.info("EnterpriseProxy shutdown complete")

    async def _ensure_session(self) -> Any:
        """Ensure HTTP session is initialized.

        Returns:
            aiohttp.ClientSession instance.
        """
        if self._session is not None:
            return self._session

        async with self._session_lock:
            if self._session is not None:
                return self._session

            try:
                import aiohttp

                # Configure connection pooling
                connector = aiohttp.TCPConnector(
                    limit=self.config.max_connections,
                    limit_per_host=self.config.max_connections_per_host,
                    keepalive_timeout=self.config.keepalive_timeout,
                    enable_cleanup_closed=True,
                )

                timeout = aiohttp.ClientTimeout(
                    total=self.config.default_timeout,
                    connect=self.config.default_connect_timeout,
                )

                self._session = aiohttp.ClientSession(
                    connector=connector,
                    timeout=timeout,
                    headers={"User-Agent": self.config.user_agent},
                )

                return self._session

            except ImportError:
                raise RuntimeError(
                    "aiohttp is required for EnterpriseProxy. Install with: pip install aiohttp"
                )

    # =========================================================================
    # Framework Management
    # =========================================================================

    def register_framework(
        self,
        name: str,
        config: ExternalFrameworkConfig,
    ) -> None:
        """Register a new external framework.

        Args:
            name: Unique framework identifier.
            config: Framework configuration.
        """
        if name in self._frameworks:
            logger.warning(f"Overwriting existing framework config for '{name}'")

        self._frameworks[name] = config
        self._init_framework(name, config)
        logger.info(f"Registered framework: {name}")

    def unregister_framework(self, name: str) -> bool:
        """Unregister an external framework.

        Args:
            name: Framework identifier to remove.

        Returns:
            True if framework was removed, False if not found.
        """
        if name not in self._frameworks:
            return False

        del self._frameworks[name]
        del self._circuit_breakers[name]
        del self._bulkheads[name]
        del self._sanitizers[name]
        del self._health_results[name]

        logger.info(f"Unregistered framework: {name}")
        return True

    def get_framework_config(self, name: str) -> ExternalFrameworkConfig | None:
        """Get configuration for a framework.

        Args:
            name: Framework identifier.

        Returns:
            Framework configuration or None if not found.
        """
        return self._frameworks.get(name)

    def list_frameworks(self) -> list[str]:
        """List all registered framework names.

        Returns:
            List of framework names.
        """
        return list(self._frameworks.keys())

    # =========================================================================
    # Hook Management
    # =========================================================================

    def add_pre_request_hook(self, hook: PreRequestHook) -> None:
        """Add a pre-request hook.

        Pre-request hooks are called before each request is sent.
        They can modify the request or return None to abort.

        Args:
            hook: Async function taking ProxyRequest, returning modified
                  request or None to abort.
        """
        self._pre_request_hooks.append(hook)

    def add_post_request_hook(self, hook: PostRequestHook) -> None:
        """Add a post-request hook.

        Post-request hooks are called after each successful response.

        Args:
            hook: Async function taking ProxyRequest and ProxyResponse.
        """
        self._post_request_hooks.append(hook)

    def add_error_hook(self, hook: ErrorHook) -> None:
        """Add an error hook.

        Error hooks are called when a request fails.

        Args:
            hook: Async function taking ProxyRequest and Exception.
        """
        self._error_hooks.append(hook)

    def remove_pre_request_hook(self, hook: PreRequestHook) -> bool:
        """Remove a pre-request hook.

        Args:
            hook: Hook to remove.

        Returns:
            True if hook was removed, False if not found.
        """
        try:
            self._pre_request_hooks.remove(hook)
            return True
        except ValueError:
            return False

    def remove_post_request_hook(self, hook: PostRequestHook) -> bool:
        """Remove a post-request hook.

        Args:
            hook: Hook to remove.

        Returns:
            True if hook was removed, False if not found.
        """
        try:
            self._post_request_hooks.remove(hook)
            return True
        except ValueError:
            return False

    def remove_error_hook(self, hook: ErrorHook) -> bool:
        """Remove an error hook.

        Args:
            hook: Hook to remove.

        Returns:
            True if hook was removed, False if not found.
        """
        try:
            self._error_hooks.remove(hook)
            return True
        except ValueError:
            return False

    # =========================================================================
    # Request Handling
    # =========================================================================

    async def request(
        self,
        framework: str,
        method: str,
        path: str,
        *,
        headers: dict[str, str] | None = None,
        json: dict[str, Any] | None = None,
        data: bytes | None = None,
        params: dict[str, str] | None = None,
        timeout: float | None = None,
        tenant_id: str | None = None,
        correlation_id: str | None = None,
        auth_context: Any | None = None,
        skip_circuit_breaker: bool = False,
        skip_retry: bool = False,
    ) -> ProxyResponse:
        """Make a proxied request to an external framework.

        Args:
            framework: Target framework name.
            method: HTTP method (GET, POST, etc.).
            path: Request path (appended to base_url).
            headers: Additional request headers.
            json: JSON body (will be serialized).
            data: Raw body data.
            params: URL query parameters.
            timeout: Request timeout override.
            tenant_id: Tenant identifier for context.
            correlation_id: Request correlation ID.
            auth_context: Authentication context for hooks.
            skip_circuit_breaker: Skip circuit breaker check.
            skip_retry: Skip retry logic.

        Returns:
            ProxyResponse with status, headers, and body.

        Raises:
            FrameworkNotConfiguredError: If framework is not registered.
            CircuitOpenError: If circuit breaker is open.
            BulkheadFullError: If no bulkhead slots available.
            RequestTimeoutError: If request times out.
            ProxyError: For other proxy errors.
        """
        # Validate framework
        fw_config = self._frameworks.get(framework)
        if fw_config is None:
            raise FrameworkNotConfiguredError(framework)

        if not fw_config.enabled:
            raise ProxyError(
                f"Framework '{framework}' is disabled",
                code="FRAMEWORK_DISABLED",
                framework=framework,
            )

        # Build full URL
        url = f"{fw_config.base_url}{path}"

        # Merge headers
        all_headers = dict(fw_config.default_headers)
        if headers:
            all_headers.update(headers)

        # Add tenant context header
        if tenant_id:
            all_headers[self.config.tenant_header_name] = tenant_id

        # Add correlation ID header
        if correlation_id:
            all_headers[self.config.correlation_header_name] = correlation_id

        # Serialize JSON body
        body: bytes | None = None
        if json is not None:
            import json as json_module

            body = json_module.dumps(json).encode("utf-8")
            all_headers.setdefault("Content-Type", "application/json")
        elif data is not None:
            body = data

        # Create proxy request
        proxy_request = ProxyRequest(
            framework=framework,
            method=method.upper(),
            url=url,
            headers=all_headers,
            body=body,
            tenant_id=tenant_id,
            correlation_id=correlation_id,
            auth_context=auth_context,
        )

        # Validate request
        sanitizer = self._sanitizers[framework]
        sanitizer.validate_request(proxy_request)

        # Run pre-request hooks
        for hook in self._pre_request_hooks:
            try:
                result = await hook(proxy_request)
                if result is None:
                    raise ProxyError(
                        "Request aborted by pre-request hook",
                        code="REQUEST_ABORTED",
                        framework=framework,
                    )
                proxy_request = result
            except ProxyError:
                raise
            except Exception as e:
                logger.error(f"Pre-request hook failed: {e}")
                raise ProxyError(
                    f"Pre-request hook failed: {e}",
                    code="HOOK_ERROR",
                    framework=framework,
                )

        # Execute with resilience patterns
        try:
            return await self._execute_with_resilience(
                proxy_request,
                fw_config,
                timeout=timeout,
                skip_circuit_breaker=skip_circuit_breaker,
                skip_retry=skip_retry,
            )
        except Exception as e:
            # Run error hooks
            for hook in self._error_hooks:
                try:
                    await hook(proxy_request, e)
                except Exception as hook_error:
                    logger.error(f"Error hook failed: {hook_error}")
            raise

    async def _execute_with_resilience(
        self,
        request: ProxyRequest,
        config: ExternalFrameworkConfig,
        *,
        timeout: float | None = None,
        skip_circuit_breaker: bool = False,
        skip_retry: bool = False,
    ) -> ProxyResponse:
        """Execute request with resilience patterns.

        Args:
            request: Proxy request to execute.
            config: Framework configuration.
            timeout: Request timeout override.
            skip_circuit_breaker: Skip circuit breaker check.
            skip_retry: Skip retry logic.

        Returns:
            ProxyResponse from the framework.
        """
        framework = request.framework
        circuit_breaker = self._circuit_breakers[framework]
        bulkhead = self._bulkheads[framework]

        # Check circuit breaker
        if not skip_circuit_breaker:
            can_proceed = await circuit_breaker.can_proceed()
            if not can_proceed:
                raise CircuitOpenError(
                    framework,
                    circuit_breaker.cooldown_remaining,
                )

        # Acquire bulkhead slot
        async with bulkhead.acquire():
            # Execute with retry
            if skip_retry:
                return await self._execute_request(request, config, timeout=timeout)

            return await self._execute_with_retry(request, config, timeout=timeout)

    async def _execute_with_retry(
        self,
        request: ProxyRequest,
        config: ExternalFrameworkConfig,
        *,
        timeout: float | None = None,
    ) -> ProxyResponse:
        """Execute request with retry logic.

        Args:
            request: Proxy request to execute.
            config: Framework configuration.
            timeout: Request timeout override.

        Returns:
            ProxyResponse from the framework.
        """
        retry_settings = config.retry
        circuit_breaker = self._circuit_breakers[request.framework]
        last_exception: Exception | None = None

        for attempt in range(retry_settings.max_retries + 1):
            try:
                response = await self._execute_request(request, config, timeout=timeout)

                # Check if response status indicates retry
                if response.status_code in retry_settings.retryable_status_codes:
                    if attempt < retry_settings.max_retries:
                        delay = self._calculate_retry_delay(attempt, retry_settings)
                        logger.debug(
                            f"Retrying {request.framework} request "
                            f"(attempt {attempt + 1}/{retry_settings.max_retries}) "
                            f"after {delay:.2f}s due to status {response.status_code}"
                        )
                        await asyncio.sleep(delay)
                        continue

                # Record success for non-retryable responses
                if response.is_success:
                    await circuit_breaker.record_success()

                return response

            except (asyncio.TimeoutError, ConnectionError, OSError) as e:
                last_exception = e
                await circuit_breaker.record_failure()

                if attempt < retry_settings.max_retries:
                    delay = self._calculate_retry_delay(attempt, retry_settings)
                    logger.debug(
                        f"Retrying {request.framework} request "
                        f"(attempt {attempt + 1}/{retry_settings.max_retries}) "
                        f"after {delay:.2f}s: {e}"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.warning(
                        f"Request to {request.framework} failed after "
                        f"{retry_settings.max_retries + 1} attempts: {e}"
                    )
                    raise

        if last_exception:
            raise last_exception
        raise RuntimeError("Unexpected retry state")

    def _calculate_retry_delay(self, attempt: int, settings: RetrySettings) -> float:
        """Calculate retry delay for an attempt.

        Args:
            attempt: Attempt number (0-indexed).
            settings: Retry settings.

        Returns:
            Delay in seconds.
        """
        import random

        if settings.strategy == RetryStrategy.EXPONENTIAL:
            delay = settings.base_delay * (2**attempt)
        elif settings.strategy == RetryStrategy.LINEAR:
            delay = settings.base_delay * (attempt + 1)
        else:  # CONSTANT
            delay = settings.base_delay

        # Cap at max delay
        delay = min(delay, settings.max_delay)

        # Apply jitter
        if settings.jitter:
            jitter_factor = 0.25
            factor = 1.0 + (random.random() * 2 - 1) * jitter_factor
            delay = delay * factor

        return max(0, delay)

    async def _execute_request(
        self,
        request: ProxyRequest,
        config: ExternalFrameworkConfig,
        *,
        timeout: float | None = None,
    ) -> ProxyResponse:
        """Execute a single HTTP request.

        Args:
            request: Proxy request to execute.
            config: Framework configuration.
            timeout: Request timeout override.

        Returns:
            ProxyResponse from the framework.
        """
        import aiohttp

        session = await self._ensure_session()
        request_timeout = timeout or config.timeout

        start_time = time.time()

        try:
            client_timeout = aiohttp.ClientTimeout(
                total=request_timeout,
                connect=config.connect_timeout,
            )

            # Sanitize headers for outgoing request
            sanitizer = self._sanitizers[request.framework]
            sanitized_headers = sanitizer.sanitize_headers(request.headers)

            async with session.request(
                method=request.method,
                url=request.url,
                headers=sanitized_headers,
                data=request.body,
                timeout=client_timeout,
            ) as response:
                body = await response.read()
                elapsed_ms = (time.time() - start_time) * 1000

                proxy_response = ProxyResponse(
                    status_code=response.status,
                    headers=dict(response.headers),
                    body=body,
                    elapsed_ms=elapsed_ms,
                    framework=request.framework,
                    correlation_id=request.correlation_id,
                )

                # Log request if audit enabled
                if self.config.enable_audit_logging:
                    self._log_request(request, proxy_response, sanitizer)

                # Run post-request hooks
                for hook in self._post_request_hooks:
                    try:
                        await hook(request, proxy_response)
                    except Exception as e:
                        logger.error(f"Post-request hook failed: {e}")

                return proxy_response

        except asyncio.TimeoutError:
            raise RequestTimeoutError(request.framework, request_timeout)

    def _log_request(
        self,
        request: ProxyRequest,
        response: ProxyResponse,
        sanitizer: RequestSanitizer,
    ) -> None:
        """Log request/response for audit.

        Args:
            request: The proxy request.
            response: The proxy response.
            sanitizer: Sanitizer for redaction.
        """
        sanitized_headers = sanitizer.sanitize_headers(request.headers, for_logging=True)
        sanitized_body = sanitizer.sanitize_body_for_logging(request.body)

        logger.info(
            f"Proxy request: {request.method} {request.url} "
            f"-> {response.status_code} ({response.elapsed_ms:.1f}ms)",
            extra={
                "framework": request.framework,
                "method": request.method,
                "url": request.url,
                "status_code": response.status_code,
                "elapsed_ms": response.elapsed_ms,
                "tenant_id": request.tenant_id,
                "correlation_id": request.correlation_id,
                "request_headers": sanitized_headers,
                "request_body_preview": sanitized_body[:200] if sanitized_body else None,
            },
        )

    # =========================================================================
    # Health Checks
    # =========================================================================

    async def check_health(self, framework: str) -> HealthCheckResult:
        """Perform a health check for a framework.

        Args:
            framework: Framework to check.

        Returns:
            HealthCheckResult with status and latency.
        """
        fw_config = self._frameworks.get(framework)
        if fw_config is None:
            return HealthCheckResult(
                framework=framework,
                status=HealthStatus.UNKNOWN,
                error="Framework not configured",
            )

        if not fw_config.health_check_path:
            return HealthCheckResult(
                framework=framework,
                status=HealthStatus.UNKNOWN,
                error="No health check path configured",
            )

        start_time = time.time()

        try:
            response = await self.request(
                framework=framework,
                method="GET",
                path=fw_config.health_check_path,
                timeout=10.0,
                skip_circuit_breaker=True,
                skip_retry=True,
            )

            latency_ms = (time.time() - start_time) * 1000

            if response.is_success:
                status = HealthStatus.HEALTHY
                error = None
            else:
                status = HealthStatus.DEGRADED
                error = f"Non-2xx status: {response.status_code}"

            result = HealthCheckResult(
                framework=framework,
                status=status,
                latency_ms=latency_ms,
                error=error,
                consecutive_failures=0,
            )

        except Exception as e:
            prev_result = self._health_results.get(framework)
            consecutive_failures = prev_result.consecutive_failures + 1 if prev_result else 1

            result = HealthCheckResult(
                framework=framework,
                status=HealthStatus.UNHEALTHY,
                error=str(e),
                consecutive_failures=consecutive_failures,
            )

        self._health_results[framework] = result
        return result

    async def check_all_health(self) -> dict[str, HealthCheckResult]:
        """Perform health checks for all frameworks.

        Returns:
            Dictionary of framework name to health check result.
        """
        results = {}
        for framework in self._frameworks:
            results[framework] = await self.check_health(framework)
        return results

    async def _health_check_loop(self) -> None:
        """Background task for periodic health checks."""
        while not self._shutdown_event.is_set():
            for name, config in self._frameworks.items():
                if config.health_check_path:
                    try:
                        await self.check_health(name)
                    except Exception as e:
                        logger.error(f"Health check failed for {name}: {e}")

                    await asyncio.sleep(config.health_check_interval)

            # Sleep before next round
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=10.0,
                )
            except asyncio.TimeoutError:
                pass

    # =========================================================================
    # Monitoring and Statistics
    # =========================================================================

    def get_circuit_breaker_status(
        self,
        framework: str | None = None,
    ) -> dict[str, Any]:
        """Get circuit breaker status.

        Args:
            framework: Specific framework, or None for all.

        Returns:
            Circuit breaker status dictionary.
        """
        if framework:
            cb = self._circuit_breakers.get(framework)
            return cb.to_dict() if cb else {}

        return {name: cb.to_dict() for name, cb in self._circuit_breakers.items()}

    def get_bulkhead_status(
        self,
        framework: str | None = None,
    ) -> dict[str, Any]:
        """Get bulkhead status.

        Args:
            framework: Specific framework, or None for all.

        Returns:
            Bulkhead status dictionary.
        """
        if framework:
            bh = self._bulkheads.get(framework)
            return bh.to_dict() if bh else {}

        return {name: bh.to_dict() for name, bh in self._bulkheads.items()}

    def get_health_status(
        self,
        framework: str | None = None,
    ) -> dict[str, Any]:
        """Get health check status.

        Args:
            framework: Specific framework, or None for all.

        Returns:
            Health status dictionary.
        """
        if framework:
            result = self._health_results.get(framework)
            return result.to_dict() if result else {}

        return {name: result.to_dict() for name, result in self._health_results.items()}

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive proxy statistics.

        Returns:
            Dictionary of proxy statistics.
        """
        return {
            "config": {
                "max_connections": self.config.max_connections,
                "max_connections_per_host": self.config.max_connections_per_host,
                "default_timeout": self.config.default_timeout,
                "audit_logging_enabled": self.config.enable_audit_logging,
                "metrics_enabled": self.config.enable_metrics,
            },
            "frameworks": {
                name: {
                    "enabled": fw.enabled,
                    "base_url": fw.base_url,
                    "timeout": fw.timeout,
                }
                for name, fw in self._frameworks.items()
            },
            "circuit_breakers": self.get_circuit_breaker_status(),
            "bulkheads": self.get_bulkhead_status(),
            "health": self.get_health_status(),
            "hooks": {
                "pre_request": len(self._pre_request_hooks),
                "post_request": len(self._post_request_hooks),
                "error": len(self._error_hooks),
            },
        }

    async def reset_circuit_breaker(self, framework: str) -> bool:
        """Reset circuit breaker for a framework.

        Args:
            framework: Framework to reset.

        Returns:
            True if reset successful, False if framework not found.
        """
        cb = self._circuit_breakers.get(framework)
        if cb is None:
            return False
        await cb.reset()
        return True


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Exceptions
    "ProxyError",
    "CircuitOpenError",
    "BulkheadFullError",
    "RequestTimeoutError",
    "FrameworkNotConfiguredError",
    "SanitizationError",
    # Enums
    "HealthStatus",
    "RetryStrategy",
    # Configuration
    "CircuitBreakerSettings",
    "RetrySettings",
    "BulkheadSettings",
    "SanitizationSettings",
    "ExternalFrameworkConfig",
    "ProxyConfig",
    # Request/Response
    "ProxyRequest",
    "ProxyResponse",
    # Health
    "HealthCheckResult",
    # Components
    "FrameworkCircuitBreaker",
    "FrameworkBulkhead",
    "RequestSanitizer",
    # Main class
    "EnterpriseProxy",
]
