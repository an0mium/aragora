"""
Error monitoring integration for Aragora.

Provides centralized error tracking with optional Sentry integration.
Captures exceptions, performance data, and custom events.

Configuration via environment variables:
    SENTRY_DSN: Sentry DSN for error reporting (optional)
    SENTRY_ENVIRONMENT: Environment name (default: "development")
    SENTRY_TRACES_SAMPLE_RATE: Percentage of requests to trace (default: 0.1)

Usage:
    from aragora.server.error_monitoring import init_monitoring, capture_exception

    # Initialize at startup
    init_monitoring()

    # Capture exceptions
    try:
        risky_operation()
    except Exception as e:
        capture_exception(e, context={"debate_id": "123"})
"""

import logging
import os
from functools import wraps
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

# Type variable for generic function signatures
F = TypeVar('F', bound=Callable[..., Any])

# Global state
_initialized = False
_sentry_available = False


def init_monitoring() -> bool:
    """Initialize error monitoring.

    Returns:
        True if Sentry was initialized, False otherwise.
    """
    global _initialized, _sentry_available

    if _initialized:
        return _sentry_available

    dsn = os.environ.get("SENTRY_DSN")
    if not dsn:
        logger.info("SENTRY_DSN not set, error monitoring disabled")
        _initialized = True
        return False

    try:
        import sentry_sdk
        from sentry_sdk.integrations.aiohttp import AioHttpIntegration
        from sentry_sdk.integrations.logging import LoggingIntegration

        environment = os.environ.get("SENTRY_ENVIRONMENT", "development")
        traces_sample_rate = float(os.environ.get("SENTRY_TRACES_SAMPLE_RATE", "0.1"))

        sentry_sdk.init(
            dsn=dsn,
            environment=environment,
            traces_sample_rate=traces_sample_rate,
            integrations=[
                AioHttpIntegration(),
                LoggingIntegration(
                    level=logging.INFO,
                    event_level=logging.ERROR,
                ),
            ],
            # Don't send PII by default
            send_default_pii=False,
            # Attach stack trace to log messages
            attach_stacktrace=True,
            # Filter out health check transactions
            before_send_transaction=_filter_health_checks,
        )

        _sentry_available = True
        logger.info(f"Sentry initialized for environment: {environment}")

    except ImportError:
        logger.warning("sentry-sdk not installed, error monitoring disabled")
        _sentry_available = False
    except Exception as e:
        logger.error(f"Failed to initialize Sentry: {e}")
        _sentry_available = False

    _initialized = True
    return _sentry_available


def _filter_health_checks(event: dict, hint: dict) -> dict | None:
    """Filter out health check endpoints from transaction sampling."""
    if event.get("transaction", "").startswith("/api/health"):
        return None
    return event


def capture_exception(
    exception: Exception,
    context: dict[str, Any] | None = None,
    level: str = "error",
) -> str | None:
    """Capture an exception for error monitoring.

    Args:
        exception: The exception to capture.
        context: Additional context to attach to the event.
        level: Severity level (error, warning, info).

    Returns:
        Event ID if captured, None otherwise.
    """
    if not _sentry_available:
        logger.exception(f"Uncaptured exception: {exception}")
        return None

    try:
        import sentry_sdk

        with sentry_sdk.push_scope() as scope:
            if context:
                for key, value in context.items():
                    scope.set_extra(key, value)
            scope.level = level
            return sentry_sdk.capture_exception(exception)
    except Exception as e:
        logger.error(f"Failed to capture exception: {e}")
        return None


def capture_message(
    message: str,
    level: str = "info",
    context: dict[str, Any] | None = None,
) -> str | None:
    """Capture a message for error monitoring.

    Args:
        message: The message to capture.
        level: Severity level (error, warning, info).
        context: Additional context to attach to the event.

    Returns:
        Event ID if captured, None otherwise.
    """
    if not _sentry_available:
        logger.log(
            logging.ERROR if level == "error" else logging.INFO,
            f"Uncaptured message: {message}"
        )
        return None

    try:
        import sentry_sdk

        with sentry_sdk.push_scope() as scope:
            if context:
                for key, value in context.items():
                    scope.set_extra(key, value)
            return sentry_sdk.capture_message(message, level=level)
    except Exception as e:
        logger.error(f"Failed to capture message: {e}")
        return None


def set_user(user_id: str, ip_address: str | None = None):
    """Set user context for error tracking.

    Args:
        user_id: User identifier (can be hashed/anonymized).
        ip_address: Optional IP address.
    """
    if not _sentry_available:
        return

    try:
        import sentry_sdk

        sentry_sdk.set_user({
            "id": user_id,
            "ip_address": ip_address,
        })
    except Exception as e:
        logger.error(f"Failed to set user: {e}")


def set_tag(key: str, value: str):
    """Set a tag on the current scope.

    Args:
        key: Tag key.
        value: Tag value.
    """
    if not _sentry_available:
        return

    try:
        import sentry_sdk
        sentry_sdk.set_tag(key, value)
    except Exception as e:
        logger.error(f"Failed to set tag: {e}")


def monitor_errors(func: F) -> F:
    """Decorator to automatically capture exceptions from a function.

    Usage:
        @monitor_errors
        async def risky_operation():
            ...
    """
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            capture_exception(e, context={"function": func.__name__})
            raise

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            capture_exception(e, context={"function": func.__name__})
            raise

    import asyncio
    if asyncio.iscoroutinefunction(func):
        return async_wrapper  # type: ignore
    return sync_wrapper  # type: ignore


def get_status() -> dict[str, Any]:
    """Get error monitoring status.

    Returns:
        Status dictionary with monitoring state.
    """
    return {
        "initialized": _initialized,
        "sentry_available": _sentry_available,
        "dsn_configured": bool(os.environ.get("SENTRY_DSN")),
        "environment": os.environ.get("SENTRY_ENVIRONMENT", "development"),
    }
