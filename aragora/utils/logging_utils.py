"""
Logging utilities for performance-conscious logging.

This module provides utilities to avoid the f-string logging performance trap.
When using f-strings with logging, the string is always formatted even if the
log level is disabled:

    # BAD: f-string evaluated even if DEBUG is disabled
    logger.debug(f"Expensive: {expensive_call()}")

    # GOOD: % formatting is lazy - only formats if log level enabled
    logger.debug("Expensive: %s", expensive_call())

    # BEST: For truly expensive calls, check level first
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Expensive: %s", expensive_call())

Usage:
    from aragora.utils.logging_utils import lazy_debug, lazy_format

    # Lazy debug with deferred evaluation
    lazy_debug(logger, "Value: %s, Data: %s", lambda: expensive_value, lambda: expensive_data)

    # Or use LazyStr for individual expensive values
    logger.debug("Result: %s", LazyStr(lambda: expensive_call()))
"""

from __future__ import annotations

import logging
from typing import Any, Callable


class LazyStr:
    """A lazy string that only evaluates when converted to str.

    Use this to wrap expensive computations in log statements:

        logger.debug("Result: %s", LazyStr(lambda: expensive_computation()))

    The lambda is only called if the log message is actually formatted.
    """

    __slots__ = ("_func",)

    def __init__(self, func: Callable[[], Any]):
        self._func = func

    def __str__(self) -> str:
        return str(self._func())

    def __repr__(self) -> str:
        return f"LazyStr({self._func})"


def lazy_debug(
    logger: logging.Logger,
    msg: str,
    *args: Callable[[], Any] | Any,
    **kwargs: Any,
) -> None:
    """Log debug message with lazy argument evaluation.

    Arguments can be:
    - Regular values (used directly)
    - Callables (called only if debug is enabled)

    Example:
        lazy_debug(logger, "User %s has %d items", user_id, lambda: len(fetch_items()))
    """
    if logger.isEnabledFor(logging.DEBUG):
        evaluated_args = tuple(arg() if callable(arg) else arg for arg in args)
        logger.debug(msg, *evaluated_args, **kwargs)


def lazy_info(
    logger: logging.Logger,
    msg: str,
    *args: Callable[[], Any] | Any,
    **kwargs: Any,
) -> None:
    """Log info message with lazy argument evaluation."""
    if logger.isEnabledFor(logging.INFO):
        evaluated_args = tuple(arg() if callable(arg) else arg for arg in args)
        logger.info(msg, *evaluated_args, **kwargs)


def lazy_warning(
    logger: logging.Logger,
    msg: str,
    *args: Callable[[], Any] | Any,
    **kwargs: Any,
) -> None:
    """Log warning message with lazy argument evaluation."""
    if logger.isEnabledFor(logging.WARNING):
        evaluated_args = tuple(arg() if callable(arg) else arg for arg in args)
        logger.warning(msg, *evaluated_args, **kwargs)


def lazy_format(msg: str, *args: Callable[[], Any]) -> str:
    """Format a message lazily, calling arg functions only when needed.

    Useful when you need to build a string conditionally:

        if some_condition:
            result = lazy_format("Data: %s", lambda: expensive_fetch())
    """
    evaluated_args = tuple(arg() if callable(arg) else arg for arg in args)
    return msg % evaluated_args


__all__ = [
    "LazyStr",
    "lazy_debug",
    "lazy_info",
    "lazy_warning",
    "lazy_format",
]
