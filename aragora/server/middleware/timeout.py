"""
Request Timeout Middleware.

Provides request-level timeout enforcement for HTTP handlers:
- Configurable per-endpoint timeouts
- Graceful timeout handling with proper HTTP 504 responses
- Integration with both sync and async handlers

Usage:
    from aragora.server.middleware.timeout import (
        with_timeout,
        async_with_timeout,
        RequestTimeoutConfig,
    )

    # Decorator style (sync)
    @with_timeout(30)  # 30 second timeout
    def slow_handler(self, handler):
        ...

    # Decorator style (async)
    @async_with_timeout(60)  # 60 second timeout
    async def async_slow_handler(self, handler):
        ...

Configuration via environment:
    ARAGORA_REQUEST_TIMEOUT=30        # Default timeout in seconds
    ARAGORA_SLOW_REQUEST_TIMEOUT=120  # Timeout for known slow endpoints
    ARAGORA_MAX_REQUEST_TIMEOUT=600   # Maximum allowed timeout
"""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import signal
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, TypeVar, cast

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class RequestTimeoutConfig:
    """Configuration for request timeouts."""

    # Default timeout in seconds
    default_timeout: float = float(os.environ.get("ARAGORA_REQUEST_TIMEOUT", "30"))

    # Timeout for known slow endpoints (debates, batch operations)
    slow_timeout: float = float(os.environ.get("ARAGORA_SLOW_REQUEST_TIMEOUT", "120"))

    # Maximum allowed timeout (hard cap)
    max_timeout: float = float(os.environ.get("ARAGORA_MAX_REQUEST_TIMEOUT", "600"))

    # Per-endpoint timeout overrides
    endpoint_timeouts: Dict[str, float] = field(default_factory=dict)

    def get_timeout(self, path: str) -> float:
        """Get timeout for a specific endpoint path.

        Args:
            path: Request path

        Returns:
            Timeout in seconds
        """
        # Check explicit overrides first
        for pattern, timeout in self.endpoint_timeouts.items():
            if pattern in path:
                return min(timeout, self.max_timeout)

        # Use slow timeout for known slow operations
        slow_patterns = [
            "/api/debates/create",
            "/api/debates/batch",
            "/api/gauntlet",
            "/api/evolution",
            "/api/verify",
            "/api/evidence/collect",
            "/api/broadcast",
        ]

        for pattern in slow_patterns:
            if pattern in path:
                return min(self.slow_timeout, self.max_timeout)

        return min(self.default_timeout, self.max_timeout)


# Global config instance
_timeout_config: Optional[RequestTimeoutConfig] = None


def get_timeout_config() -> RequestTimeoutConfig:
    """Get or create the global timeout configuration."""
    global _timeout_config
    if _timeout_config is None:
        _timeout_config = RequestTimeoutConfig()
    return _timeout_config


def configure_timeout(
    default_timeout: Optional[float] = None,
    slow_timeout: Optional[float] = None,
    max_timeout: Optional[float] = None,
    endpoint_overrides: Optional[Dict[str, float]] = None,
) -> RequestTimeoutConfig:
    """Configure request timeout settings.

    Args:
        default_timeout: Default timeout in seconds
        slow_timeout: Timeout for slow endpoints
        max_timeout: Maximum allowed timeout
        endpoint_overrides: Per-endpoint timeout overrides

    Returns:
        Updated configuration
    """
    global _timeout_config
    config = get_timeout_config()

    if default_timeout is not None:
        config.default_timeout = default_timeout
    if slow_timeout is not None:
        config.slow_timeout = slow_timeout
    if max_timeout is not None:
        config.max_timeout = max_timeout
    if endpoint_overrides is not None:
        config.endpoint_timeouts.update(endpoint_overrides)

    _timeout_config = config
    return config


# =============================================================================
# Timeout Error
# =============================================================================


class RequestTimeoutError(Exception):
    """Exception raised when a request times out."""

    def __init__(
        self,
        message: str = "Request timed out",
        timeout: float = 0,
        path: str = "",
    ):
        self.timeout = timeout
        self.path = path
        super().__init__(f"{message} (timeout={timeout}s, path={path})")


# =============================================================================
# Sync Timeout Implementation
# =============================================================================

# Thread pool for running sync functions with timeout
_executor: Optional[ThreadPoolExecutor] = None


def get_executor() -> ThreadPoolExecutor:
    """Get or create thread pool executor for timeout handling."""
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(
            max_workers=int(os.environ.get("ARAGORA_TIMEOUT_WORKERS", "10")),
            thread_name_prefix="timeout-",
        )
    return _executor


def shutdown_executor() -> None:
    """Shutdown the timeout executor gracefully."""
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=False)
        _executor = None


F = TypeVar("F", bound=Callable[..., Any])


def with_timeout(
    timeout: Optional[float] = None,
    error_response: Optional[Callable[[float, str], Any]] = None,
) -> Callable[[F], F]:
    """
    Decorator to add timeout to sync handler functions.

    If the handler doesn't complete within the timeout, returns a 504
    Gateway Timeout response.

    Args:
        timeout: Timeout in seconds (default: from config)
        error_response: Custom error response generator

    Returns:
        Decorated function with timeout enforcement

    Usage:
        @with_timeout(30)
        def my_handler(self, path, query_params, handler):
            # Will timeout after 30 seconds
            ...
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Determine timeout
            effective_timeout = timeout
            if effective_timeout is None:
                # Try to get path from args for path-based timeout
                config = get_timeout_config()
                path = ""
                if len(args) >= 2 and isinstance(args[1], str):
                    path = args[1]
                effective_timeout = config.get_timeout(path)

            executor = get_executor()

            try:
                future = executor.submit(func, *args, **kwargs)
                result = future.result(timeout=effective_timeout)
                return result

            except FuturesTimeoutError:
                path = args[1] if len(args) >= 2 else "unknown"
                logger.warning(f"Request timeout after {effective_timeout}s: {path}")

                # Cancel the running task if possible
                future.cancel()

                # Return custom error response or default 504
                if error_response:
                    return error_response(effective_timeout, path)

                # Default 504 response
                return (
                    {
                        "error": "Request timed out",
                        "code": "request_timeout",
                        "timeout_seconds": effective_timeout,
                        "path": path,
                    },
                    504,
                    {"X-Timeout": str(effective_timeout)},
                )

        return cast(F, wrapper)

    return decorator


# =============================================================================
# Async Timeout Implementation
# =============================================================================


def async_with_timeout(
    timeout: Optional[float] = None,
    error_response: Optional[Callable[[float, str], Any]] = None,
) -> Callable[[F], F]:
    """
    Decorator to add timeout to async handler functions.

    If the handler doesn't complete within the timeout, returns a 504
    Gateway Timeout response.

    Args:
        timeout: Timeout in seconds (default: from config)
        error_response: Custom error response generator

    Returns:
        Decorated function with timeout enforcement

    Usage:
        @async_with_timeout(60)
        async def my_async_handler(self, path, query_params, handler):
            # Will timeout after 60 seconds
            ...
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Determine timeout
            effective_timeout = timeout
            if effective_timeout is None:
                config = get_timeout_config()
                path = ""
                if len(args) >= 2 and isinstance(args[1], str):
                    path = args[1]
                effective_timeout = config.get_timeout(path)

            try:
                result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=effective_timeout,
                )
                return result

            except asyncio.TimeoutError:
                path = args[1] if len(args) >= 2 else "unknown"
                logger.warning(f"Async request timeout after {effective_timeout}s: {path}")

                # Return custom error response or default 504
                if error_response:
                    return error_response(effective_timeout, path)

                # Default 504 response
                return (
                    {
                        "error": "Request timed out",
                        "code": "request_timeout",
                        "timeout_seconds": effective_timeout,
                        "path": path,
                    },
                    504,
                    {"X-Timeout": str(effective_timeout)},
                )

        return cast(F, wrapper)

    return decorator


# =============================================================================
# Context Manager Style
# =============================================================================


@contextmanager
def timeout_context(
    timeout: float,
    path: str = "",
):
    """
    Context manager for timeout enforcement.

    Usage:
        with timeout_context(30, "/api/debates"):
            result = slow_operation()

    Note: This only works properly on Unix systems due to signal.alarm.
    On Windows, use the decorator-based approach instead.
    """
    # signal.alarm doesn't work in threads, so we use a simpler approach
    import platform

    if platform.system() != "Windows":

        def timeout_handler(signum, frame):
            raise RequestTimeoutError(
                "Request timed out",
                timeout=timeout,
                path=path,
            )

        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout))

        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    else:
        # Windows fallback - no timeout enforcement in context manager
        logger.debug("timeout_context: signal.alarm not available on Windows")
        yield


# =============================================================================
# Health Check
# =============================================================================


def get_timeout_stats() -> Dict[str, Any]:
    """Get statistics about timeout configuration and state."""
    config = get_timeout_config()

    executor = _executor
    executor_stats = {}
    if executor is not None:
        executor_stats = {
            "active_threads": len(executor._threads) if hasattr(executor, "_threads") else 0,
        }

    return {
        "config": {
            "default_timeout": config.default_timeout,
            "slow_timeout": config.slow_timeout,
            "max_timeout": config.max_timeout,
            "endpoint_overrides": len(config.endpoint_timeouts),
        },
        "executor": executor_stats,
    }


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Configuration
    "RequestTimeoutConfig",
    "get_timeout_config",
    "configure_timeout",
    # Errors
    "RequestTimeoutError",
    # Decorators
    "with_timeout",
    "async_with_timeout",
    # Context manager
    "timeout_context",
    # Utilities
    "get_timeout_stats",
    "shutdown_executor",
]
