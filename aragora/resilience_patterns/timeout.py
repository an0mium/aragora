"""
Unified Timeout Management for Aragora.

Provides consistent timeout handling across async and sync operations.

Usage:
    from aragora.resilience_patterns import with_timeout, TimeoutConfig

    @with_timeout(5.0)
    async def bounded_operation():
        ...

    @with_timeout(TimeoutConfig(seconds=10.0, on_timeout=my_callback))
    async def operation_with_callback():
        ...
"""

from __future__ import annotations

import asyncio
import functools
import logging
import signal
import sys
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Awaitable, Callable, Iterator, Optional, ParamSpec, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
P = ParamSpec("P")

# Import asyncio.timeout for Python 3.11+, fallback for earlier versions
if sys.version_info >= (3, 11):
    asyncio_timeout = asyncio.timeout
else:
    try:
        from async_timeout import timeout as asyncio_timeout
    except ImportError:

        @asynccontextmanager
        async def asyncio_timeout(delay: Optional[float]) -> AsyncIterator[None]:  # type: ignore[misc]
            """Fallback timeout context manager (no actual timeout)."""
            if delay is not None:
                logger.warning(
                    "async-timeout not installed, timeout not enforced. "
                    "Install with: pip install async-timeout"
                )
            yield


class TimeoutError(asyncio.TimeoutError):
    """Timeout error with additional context."""

    def __init__(
        self,
        message: str = "Operation timed out",
        timeout_seconds: Optional[float] = None,
        operation: Optional[str] = None,
    ):
        super().__init__(message)
        self.timeout_seconds = timeout_seconds
        self.operation = operation


@dataclass
class TimeoutConfig:
    """Configuration for timeout behavior.

    Attributes:
        seconds: Timeout in seconds
        on_timeout: Optional callback when timeout occurs
        error_class: Exception class to raise on timeout
        message: Custom error message
    """

    seconds: float
    on_timeout: Optional[Callable[[str], None]] = None
    error_class: type = TimeoutError
    message: Optional[str] = None

    def get_message(self, operation: str) -> str:
        """Get the error message for a timeout."""
        if self.message:
            return self.message
        return f"Operation '{operation}' timed out after {self.seconds}s"


def with_timeout(
    timeout: float | TimeoutConfig,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """Decorator for async functions with timeout.

    Args:
        timeout: Timeout in seconds or TimeoutConfig instance

    Returns:
        Decorator function

    Example:
        @with_timeout(5.0)
        async def bounded_operation():
            ...

        @with_timeout(TimeoutConfig(seconds=10.0))
        async def another_operation():
            ...
    """
    config = timeout if isinstance(timeout, TimeoutConfig) else TimeoutConfig(seconds=timeout)

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            try:
                async with asyncio_timeout(config.seconds):
                    return await func(*args, **kwargs)
            except asyncio.TimeoutError:
                operation = func.__name__
                message = config.get_message(operation)

                if config.on_timeout:
                    try:
                        config.on_timeout(operation)
                    except Exception as e:
                        logger.warning(f"Timeout callback error for {operation}: {e}")

                logger.warning(message)
                raise config.error_class(
                    message,
                    timeout_seconds=config.seconds,
                    operation=operation,
                ) from None

        return wrapper

    return decorator


def with_timeout_sync(
    timeout: float | TimeoutConfig,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator for sync functions with timeout (Unix-only, uses SIGALRM).

    Note: This only works on Unix-like systems and only in the main thread.
    For cross-platform sync timeout, consider using threading or multiprocessing.

    Args:
        timeout: Timeout in seconds or TimeoutConfig instance

    Returns:
        Decorator function

    Example:
        @with_timeout_sync(5.0)
        def bounded_sync_operation():
            ...
    """
    config = timeout if isinstance(timeout, TimeoutConfig) else TimeoutConfig(seconds=timeout)

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Check if we can use signals (Unix only, main thread only)
            if not hasattr(signal, "SIGALRM"):
                logger.warning(f"SIGALRM not available, timeout not enforced for {func.__name__}")
                return func(*args, **kwargs)

            def timeout_handler(signum: int, frame: Any) -> None:
                operation = func.__name__
                message = config.get_message(operation)
                if config.on_timeout:
                    try:
                        config.on_timeout(operation)
                    except Exception as e:
                        logger.warning(f"Timeout callback error for {operation}: {e}")
                raise config.error_class(
                    message,
                    timeout_seconds=config.seconds,
                    operation=operation,
                )

            # Set up signal handler
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.setitimer(signal.ITIMER_REAL, config.seconds)

            try:
                return func(*args, **kwargs)
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, old_handler)

        return wrapper

    return decorator


@asynccontextmanager
async def timeout_context(
    seconds: float,
    on_timeout: Optional[Callable[[str], None]] = None,
    context_name: str = "operation",
) -> AsyncIterator[None]:
    """Async context manager for timeout with optional callback.

    Args:
        seconds: Timeout in seconds
        on_timeout: Optional callback when timeout occurs
        context_name: Name for logging/callback

    Yields:
        None

    Example:
        async with timeout_context(5.0, context_name="fetch"):
            result = await fetch_data()
    """
    try:
        async with asyncio_timeout(seconds):
            yield
    except asyncio.TimeoutError:
        message = f"Context '{context_name}' timed out after {seconds}s"
        if on_timeout:
            try:
                on_timeout(context_name)
            except Exception as e:
                logger.warning(f"Timeout callback error for {context_name}: {e}")
        logger.warning(message)
        raise TimeoutError(message, timeout_seconds=seconds, operation=context_name) from None


@contextmanager
def timeout_context_sync(
    seconds: float,
    on_timeout: Optional[Callable[[str], None]] = None,
    context_name: str = "operation",
) -> Iterator[None]:
    """Sync context manager for timeout (Unix-only).

    Args:
        seconds: Timeout in seconds
        on_timeout: Optional callback when timeout occurs
        context_name: Name for logging/callback

    Yields:
        None
    """
    if not hasattr(signal, "SIGALRM"):
        logger.warning(f"SIGALRM not available, timeout not enforced for {context_name}")
        yield
        return

    def timeout_handler(signum: int, frame: Any) -> None:
        message = f"Context '{context_name}' timed out after {seconds}s"
        if on_timeout:
            try:
                on_timeout(context_name)
            except Exception as e:
                logger.warning(f"Timeout callback error for {context_name}: {e}")
        raise TimeoutError(message, timeout_seconds=seconds, operation=context_name)

    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)

    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
