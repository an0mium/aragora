"""
Proxy exception classes.

Defines all exception types raised by the enterprise proxy layer including
circuit breaker errors, bulkhead capacity errors, timeouts, and sanitization failures.
"""

from __future__ import annotations

from typing import Any


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


__all__ = [
    "ProxyError",
    "CircuitOpenError",
    "BulkheadFullError",
    "RequestTimeoutError",
    "FrameworkNotConfiguredError",
    "SanitizationError",
]
