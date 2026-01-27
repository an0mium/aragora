"""
Safe Fetch Utilities for consistent error handling in data fetching.

This module provides utilities for safely fetching data with consistent
error handling, logging, and fallback values across handlers.

Usage:
    # Synchronous fetch with fallback
    result = safe_fetch(
        lambda: self._get_memory_context(debate_id),
        fallback={"available": False},
        context="memory for debate-123",
        logger=logger,
    )

    # Async fetch with fallback
    result = await safe_fetch_async(
        lambda: self._get_async_data(id),
        fallback={"available": False},
        context="async data",
        logger=logger,
    )
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, TypeVar

T = TypeVar("T")

# Default exceptions considered "data errors" (expected, non-critical)
DATA_EXCEPTIONS = (KeyError, ValueError, TypeError, AttributeError)

# Default exceptions considered "system errors" (unexpected, critical)
SYSTEM_EXCEPTIONS = (OSError, IOError, RuntimeError, ConnectionError)


def safe_fetch(
    getter: Callable[[], T],
    fallback: T,
    context: str,
    logger: logging.Logger,
    *,
    data_exceptions: tuple = DATA_EXCEPTIONS,
    log_data_errors: bool = True,
    log_level_data: int = logging.WARNING,
    log_level_unexpected: int = logging.ERROR,
    include_error_in_fallback: bool = True,
) -> T:
    """
    Safely fetch data with consistent error handling.

    Args:
        getter: Callable that returns the data
        fallback: Value to return if fetch fails
        context: Description for logging (e.g., "memory for debate-123")
        logger: Logger instance to use
        data_exceptions: Tuple of exception types considered data errors
        log_data_errors: Whether to log data errors (default True)
        log_level_data: Log level for data errors (default WARNING)
        log_level_unexpected: Log level for unexpected errors (default EXCEPTION)
        include_error_in_fallback: Whether to add error info to fallback dict

    Returns:
        The fetched data, or the fallback value with error info
    """
    try:
        return getter()
    except data_exceptions as e:
        if log_data_errors:
            logger.log(log_level_data, f"Data error fetching {context}: {e}")
        return _make_fallback(fallback, str(e), include_error_in_fallback)
    except Exception as e:
        if log_level_unexpected == logging.ERROR:
            logger.exception(f"Unexpected error fetching {context}: {e}")
        else:
            logger.log(log_level_unexpected, f"Unexpected error fetching {context}: {e}")
        return _make_fallback(fallback, "Internal error", include_error_in_fallback)


async def safe_fetch_async(
    getter: Callable[[], Awaitable[T]],
    fallback: T,
    context: str,
    logger: logging.Logger,
    *,
    data_exceptions: tuple = DATA_EXCEPTIONS,
    log_data_errors: bool = True,
    log_level_data: int = logging.WARNING,
    log_level_unexpected: int = logging.ERROR,
    include_error_in_fallback: bool = True,
) -> T:
    """
    Safely fetch data asynchronously with consistent error handling.

    Args:
        getter: Async callable that returns the data
        fallback: Value to return if fetch fails
        context: Description for logging (e.g., "memory for debate-123")
        logger: Logger instance to use
        data_exceptions: Tuple of exception types considered data errors
        log_data_errors: Whether to log data errors (default True)
        log_level_data: Log level for data errors (default WARNING)
        log_level_unexpected: Log level for unexpected errors (default EXCEPTION)
        include_error_in_fallback: Whether to add error info to fallback dict

    Returns:
        The fetched data, or the fallback value with error info
    """
    try:
        return await getter()
    except data_exceptions as e:
        if log_data_errors:
            logger.log(log_level_data, f"Data error fetching {context}: {e}")
        return _make_fallback(fallback, str(e), include_error_in_fallback)
    except Exception as e:
        if log_level_unexpected == logging.ERROR:
            logger.exception(f"Unexpected error fetching {context}: {e}")
        else:
            logger.log(log_level_unexpected, f"Unexpected error fetching {context}: {e}")
        return _make_fallback(fallback, "Internal error", include_error_in_fallback)


def _make_fallback(fallback: T, error: str, include_error: bool) -> T:
    """
    Create fallback value, optionally adding error info.

    If fallback is a dict and include_error is True, adds error info.
    Otherwise returns fallback as-is.
    """
    if not include_error:
        return fallback

    if isinstance(fallback, dict):
        # Create a copy to avoid mutating the original
        result = fallback.copy()
        result["error"] = error
        result["available"] = False
        return result  # type: ignore

    return fallback


class SafeFetchContext:
    """
    Context manager for batching multiple safe fetches with shared configuration.

    Usage:
        with SafeFetchContext(logger=logger, context_prefix="debate-123") as ctx:
            context["memory"] = ctx.fetch(
                lambda: self._get_memory_context(debate_id),
                fallback={"available": False},
                name="memory",
            )
            context["knowledge"] = ctx.fetch(
                lambda: self._get_knowledge_context(debate_id),
                fallback={"available": False},
                name="knowledge",
            )
    """

    def __init__(
        self,
        logger: logging.Logger,
        context_prefix: str = "",
        data_exceptions: tuple = DATA_EXCEPTIONS,
        log_data_errors: bool = True,
        log_level_data: int = logging.WARNING,
        log_level_unexpected: int = logging.ERROR,
        include_error_in_fallback: bool = True,
    ):
        self.logger = logger
        self.context_prefix = context_prefix
        self.data_exceptions = data_exceptions
        self.log_data_errors = log_data_errors
        self.log_level_data = log_level_data
        self.log_level_unexpected = log_level_unexpected
        self.include_error_in_fallback = include_error_in_fallback
        self._errors: list[tuple[str, str]] = []

    def __enter__(self) -> "SafeFetchContext":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass

    def fetch(
        self,
        getter: Callable[[], T],
        fallback: T,
        name: str,
    ) -> T:
        """
        Fetch data with this context's configuration.

        Args:
            getter: Callable that returns the data
            fallback: Value to return if fetch fails
            name: Name for this fetch (combined with context_prefix for logging)

        Returns:
            The fetched data, or the fallback value
        """
        context = f"{name} for {self.context_prefix}" if self.context_prefix else name
        result = safe_fetch(
            getter,
            fallback,
            context,
            self.logger,
            data_exceptions=self.data_exceptions,
            log_data_errors=self.log_data_errors,
            log_level_data=self.log_level_data,
            log_level_unexpected=self.log_level_unexpected,
            include_error_in_fallback=self.include_error_in_fallback,
        )

        # Track errors for reporting
        if isinstance(result, dict) and result.get("error"):
            self._errors.append((name, result["error"]))

        return result

    async def fetch_async(
        self,
        getter: Callable[[], Awaitable[T]],
        fallback: T,
        name: str,
    ) -> T:
        """
        Fetch data asynchronously with this context's configuration.

        Args:
            getter: Async callable that returns the data
            fallback: Value to return if fetch fails
            name: Name for this fetch (combined with context_prefix for logging)

        Returns:
            The fetched data, or the fallback value
        """
        context = f"{name} for {self.context_prefix}" if self.context_prefix else name
        result = await safe_fetch_async(
            getter,
            fallback,
            context,
            self.logger,
            data_exceptions=self.data_exceptions,
            log_data_errors=self.log_data_errors,
            log_level_data=self.log_level_data,
            log_level_unexpected=self.log_level_unexpected,
            include_error_in_fallback=self.include_error_in_fallback,
        )

        # Track errors for reporting
        if isinstance(result, dict) and result.get("error"):
            self._errors.append((name, result["error"]))

        return result

    @property
    def errors(self) -> list[tuple[str, str]]:
        """Get list of (name, error) tuples for any failed fetches."""
        return self._errors.copy()

    @property
    def has_errors(self) -> bool:
        """Check if any fetches failed."""
        return len(self._errors) > 0

    @property
    def success_count(self) -> int:
        """Get count of successful fetches (Note: only tracked for dict fallbacks)."""
        # This is a rough estimate - would need more tracking for accuracy
        return 0  # Not tracked in current implementation


def fetch_multiple(
    fetchers: dict[str, Callable[[], T]],
    fallback: T,
    context_prefix: str,
    logger: logging.Logger,
    **kwargs: Any,
) -> dict[str, T]:
    """
    Fetch multiple items with consistent error handling.

    Args:
        fetchers: Dict mapping names to getter callables
        fallback: Default fallback for all fetches
        context_prefix: Prefix for logging context
        logger: Logger instance
        **kwargs: Additional arguments passed to safe_fetch

    Returns:
        Dict mapping names to fetched values
    """
    results: dict[str, T] = {}
    with SafeFetchContext(logger, context_prefix, **kwargs) as ctx:
        for name, getter in fetchers.items():
            results[name] = ctx.fetch(getter, fallback, name)
    return results


async def fetch_multiple_async(
    fetchers: dict[str, Callable[[], Awaitable[T]]],
    fallback: T,
    context_prefix: str,
    logger: logging.Logger,
    **kwargs: Any,
) -> dict[str, T]:
    """
    Fetch multiple items asynchronously with consistent error handling.

    Args:
        fetchers: Dict mapping names to async getter callables
        fallback: Default fallback for all fetches
        context_prefix: Prefix for logging context
        logger: Logger instance
        **kwargs: Additional arguments passed to safe_fetch_async

    Returns:
        Dict mapping names to fetched values
    """
    results: dict[str, T] = {}
    with SafeFetchContext(logger, context_prefix, **kwargs) as ctx:
        for name, getter in fetchers.items():
            results[name] = await ctx.fetch_async(getter, fallback, name)
    return results
