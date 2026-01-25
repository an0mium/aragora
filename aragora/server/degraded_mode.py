"""
Degraded Mode Handler for Aragora Server.

Provides graceful degradation when the server cannot start in full operational mode
due to misconfiguration. Instead of crash-looping, the server starts in "degraded mode"
where it:

1. Returns 503 Service Unavailable for all non-health endpoints
2. Reports the degraded status and reason via health endpoints
3. Allows load balancers to properly route traffic away
4. Enables operators to diagnose the issue via logs and health checks

This prevents crash-looping scenarios that make debugging difficult and cause
unnecessary resource consumption from repeated restart attempts.

Usage:
    from aragora.server.degraded_mode import (
        is_degraded,
        get_degraded_reason,
        set_degraded,
        clear_degraded,
        DegradedModeMiddleware,
    )

    # In startup sequence
    try:
        await run_startup_sequence()
    except ConfigurationError as e:
        set_degraded(str(e), error_code="CONFIG_ERROR")

    # In health endpoint
    if is_degraded():
        return {"status": "degraded", "reason": get_degraded_reason()}
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class DegradedErrorCode(str, Enum):
    """Error codes for degraded mode."""

    CONFIG_ERROR = "CONFIG_ERROR"
    DATABASE_UNAVAILABLE = "DATABASE_UNAVAILABLE"
    REDIS_UNAVAILABLE = "REDIS_UNAVAILABLE"
    DISTRIBUTED_REQUIRED = "DISTRIBUTED_REQUIRED"
    ENCRYPTION_KEY_MISSING = "ENCRYPTION_KEY_MISSING"
    BACKEND_CONNECTIVITY = "BACKEND_CONNECTIVITY"
    STARTUP_FAILED = "STARTUP_FAILED"
    UNKNOWN = "UNKNOWN"


@dataclass
class DegradedState:
    """Represents the current degraded state of the server."""

    is_degraded: bool = False
    reason: str = ""
    error_code: DegradedErrorCode = DegradedErrorCode.UNKNOWN
    timestamp: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)
    recovery_hint: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "is_degraded": self.is_degraded,
            "reason": self.reason,
            "error_code": self.error_code.value,
            "timestamp": self.timestamp,
            "details": self.details,
            "recovery_hint": self.recovery_hint,
            "uptime_degraded_seconds": time.time() - self.timestamp if self.is_degraded else 0,
        }


# Global degraded state with thread-safe access
_degraded_state = DegradedState()
_degraded_lock = threading.Lock()

# Paths that are always allowed even when degraded
ALWAYS_ALLOWED_PATHS = frozenset(
    {
        "/health",
        "/healthz",
        "/ready",
        "/api/health",
        "/api/healthz",
        "/api/ready",
        "/api/admin/health",
        "/metrics",
        "/api/metrics",
    }
)


def is_degraded() -> bool:
    """Check if the server is in degraded mode.

    Returns:
        True if server is degraded, False if operating normally
    """
    with _degraded_lock:
        return _degraded_state.is_degraded


def get_degraded_state() -> DegradedState:
    """Get the current degraded state.

    Returns:
        Copy of the current DegradedState
    """
    with _degraded_lock:
        return DegradedState(
            is_degraded=_degraded_state.is_degraded,
            reason=_degraded_state.reason,
            error_code=_degraded_state.error_code,
            timestamp=_degraded_state.timestamp,
            details=dict(_degraded_state.details),
            recovery_hint=_degraded_state.recovery_hint,
        )


def get_degraded_reason() -> str:
    """Get the reason for degraded mode.

    Returns:
        Reason string, or empty string if not degraded
    """
    with _degraded_lock:
        return _degraded_state.reason


def set_degraded(
    reason: str,
    error_code: str | DegradedErrorCode = DegradedErrorCode.UNKNOWN,
    details: Optional[dict[str, Any]] = None,
    recovery_hint: str = "",
) -> None:
    """Set the server to degraded mode.

    Args:
        reason: Human-readable explanation of why the server is degraded
        error_code: Machine-readable error code for categorization
        details: Additional context for debugging
        recovery_hint: Suggestion for how to resolve the issue
    """
    global _degraded_state

    if isinstance(error_code, str):
        try:
            error_code = DegradedErrorCode(error_code)
        except ValueError:
            error_code = DegradedErrorCode.UNKNOWN

    with _degraded_lock:
        was_degraded = _degraded_state.is_degraded
        _degraded_state = DegradedState(
            is_degraded=True,
            reason=reason,
            error_code=error_code,
            timestamp=time.time(),
            details=details or {},
            recovery_hint=recovery_hint or _get_default_recovery_hint(error_code),
        )

    if not was_degraded:
        logger.error(
            f"[DEGRADED MODE] Server entering degraded mode: {reason} (code: {error_code.value})"
        )
        if recovery_hint or _degraded_state.recovery_hint:
            logger.info(f"[DEGRADED MODE] Recovery hint: {_degraded_state.recovery_hint}")


def clear_degraded() -> None:
    """Clear degraded mode and return to normal operation.

    This should be called when the configuration issue is resolved
    and the server can operate normally.
    """
    global _degraded_state

    with _degraded_lock:
        was_degraded = _degraded_state.is_degraded
        _degraded_state = DegradedState()

    if was_degraded:
        logger.info("[DEGRADED MODE] Server recovered, returning to normal operation")


def _get_default_recovery_hint(error_code: DegradedErrorCode) -> str:
    """Get default recovery hint for an error code."""
    hints = {
        DegradedErrorCode.CONFIG_ERROR: (
            "Check environment variables and configuration files. "
            "Run 'aragora config validate' to identify issues."
        ),
        DegradedErrorCode.DATABASE_UNAVAILABLE: (
            "Verify DATABASE_URL is correct and the database is reachable. "
            "Check firewall rules and database server status."
        ),
        DegradedErrorCode.REDIS_UNAVAILABLE: (
            "Verify REDIS_URL is correct and Redis is reachable. "
            "For single-instance deployments, set ARAGORA_SINGLE_INSTANCE=true."
        ),
        DegradedErrorCode.DISTRIBUTED_REQUIRED: (
            "Multi-instance deployment requires distributed storage. "
            "Configure DATABASE_URL and/or REDIS_URL, or set ARAGORA_SINGLE_INSTANCE=true."
        ),
        DegradedErrorCode.ENCRYPTION_KEY_MISSING: (
            "Set ARAGORA_ENCRYPTION_KEY with a 32-byte hex string. "
            "Generate one with: openssl rand -hex 32"
        ),
        DegradedErrorCode.BACKEND_CONNECTIVITY: (
            "One or more backend services are unreachable. "
            "Check network connectivity, firewall rules, and service health."
        ),
        DegradedErrorCode.STARTUP_FAILED: (
            "Server startup failed. Check logs for detailed error messages."
        ),
    }
    return hints.get(error_code, "Check server logs for detailed error information.")


def check_path_allowed(path: str) -> bool:
    """Check if a path is allowed in degraded mode.

    Args:
        path: Request path to check

    Returns:
        True if the path is allowed even when degraded
    """
    # Normalize path
    normalized = path.rstrip("/").lower()

    # Check exact matches
    if normalized in ALWAYS_ALLOWED_PATHS:
        return True

    # Check if it's a health-related path
    if normalized.endswith(("/health", "/healthz", "/ready")):
        return True

    return False


class DegradedModeMiddleware:
    """HTTP middleware that returns 503 when the server is degraded.

    This middleware should be installed early in the request chain to
    short-circuit requests when the server cannot process them properly.

    Example response when degraded:
        HTTP/1.1 503 Service Unavailable
        Content-Type: application/json
        Retry-After: 30

        {
            "error": "Service temporarily unavailable",
            "code": "DISTRIBUTED_REQUIRED",
            "reason": "Distributed storage required but DATABASE_URL not configured",
            "recovery_hint": "Configure DATABASE_URL or set ARAGORA_SINGLE_INSTANCE=true",
            "health_endpoint": "/api/health"
        }
    """

    def __init__(
        self,
        retry_after_seconds: int = 30,
        custom_allowed_paths: Optional[set[str]] = None,
    ):
        """Initialize the middleware.

        Args:
            retry_after_seconds: Value for Retry-After header
            custom_allowed_paths: Additional paths to allow when degraded
        """
        self.retry_after_seconds = retry_after_seconds
        self.custom_allowed_paths = custom_allowed_paths or set()

    def should_block(self, path: str) -> bool:
        """Check if a request should be blocked due to degraded mode.

        Args:
            path: Request path

        Returns:
            True if request should return 503, False to allow through
        """
        if not is_degraded():
            return False

        if check_path_allowed(path):
            return False

        if path in self.custom_allowed_paths:
            return False

        return True

    def get_error_response(self) -> dict[str, Any]:
        """Get the error response body for degraded mode.

        Returns:
            Dictionary suitable for JSON serialization
        """
        state = get_degraded_state()
        return {
            "error": "Service temporarily unavailable",
            "code": state.error_code.value,
            "reason": state.reason,
            "recovery_hint": state.recovery_hint,
            "health_endpoint": "/api/health",
            "details": state.details if state.details else None,
        }

    def get_headers(self) -> dict[str, str]:
        """Get response headers for degraded mode response.

        Returns:
            Dictionary of headers
        """
        return {
            "Content-Type": "application/json",
            "Retry-After": str(self.retry_after_seconds),
            "X-Aragora-Degraded": "true",
        }


def get_health_status() -> dict[str, Any]:
    """Get health status including degraded mode information.

    This should be called by health check endpoints to provide
    comprehensive status information.

    Returns:
        Health status dictionary
    """
    state = get_degraded_state()

    if state.is_degraded:
        return {
            "status": "degraded",
            "healthy": False,
            "degraded": state.to_dict(),
            "message": f"Server in degraded mode: {state.reason}",
        }

    return {
        "status": "healthy",
        "healthy": True,
        "degraded": None,
        "message": "Server operating normally",
    }


# Callback for recovery monitoring
_recovery_callbacks: list[Callable[[], bool]] = []


def register_recovery_callback(callback: Callable[[], bool]) -> None:
    """Register a callback to check if the server can recover.

    The callback should return True if the issue is resolved and
    the server can exit degraded mode.

    Args:
        callback: Function that returns True if recovery is possible
    """
    _recovery_callbacks.append(callback)


async def attempt_recovery() -> bool:
    """Attempt to recover from degraded mode.

    Runs all registered recovery callbacks. If any returns True,
    clears degraded mode.

    Returns:
        True if recovery was successful
    """
    if not is_degraded():
        return True

    for callback in _recovery_callbacks:
        try:
            if callback():
                clear_degraded()
                return True
        except Exception as e:
            logger.debug(f"Recovery callback failed: {e}")

    return False


__all__ = [
    "DegradedErrorCode",
    "DegradedState",
    "DegradedModeMiddleware",
    "is_degraded",
    "get_degraded_state",
    "get_degraded_reason",
    "set_degraded",
    "clear_degraded",
    "check_path_allowed",
    "get_health_status",
    "register_recovery_callback",
    "attempt_recovery",
    "ALWAYS_ALLOWED_PATHS",
]
