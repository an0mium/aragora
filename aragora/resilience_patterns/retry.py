"""
Unified Retry Logic for Aragora.

Provides consistent retry behavior across the codebase with configurable
backoff strategies and jitter.

This module consolidates retry patterns from:
- aragora/resilience.py
- aragora/knowledge/mound/resilience.py
- aragora/agents/errors/decorators.py
- aragora/agents/api_agents/rate_limiter.py

Usage:
    from aragora.resilience_patterns import RetryConfig, with_retry

    @with_retry(RetryConfig(max_retries=3))
    async def flaky_api_call():
        ...

    # With custom strategy
    config = RetryConfig(
        max_retries=5,
        strategy=RetryStrategy.EXPONENTIAL,
        base_delay=0.1,
        max_delay=30.0,
        jitter_mode="additive",
    )

    @with_retry(config)
    async def resilient_operation():
        ...
"""

from __future__ import annotations

import asyncio
import functools
import logging
import random
import time
from dataclasses import dataclass
from enum import Enum
from typing import (
    Awaitable,
    Callable,
    Optional,
    ParamSpec,
    Tuple,
    Type,
    TypeVar,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")
P = ParamSpec("P")


class RetryStrategy(str, Enum):
    """Retry backoff strategies."""

    EXPONENTIAL = "exponential"  # 2^n * base_delay
    LINEAR = "linear"  # n * base_delay
    CONSTANT = "constant"  # base_delay always
    FIBONACCI = "fibonacci"  # Fibonacci sequence * base_delay


class JitterMode(str, Enum):
    """Jitter application modes to prevent thundering herd."""

    NONE = "none"  # No jitter
    ADDITIVE = "additive"  # delay + random(0, jitter_max)
    MULTIPLICATIVE = "multiplicative"  # delay * random(1-jitter_factor, 1+jitter_factor)
    FULL = "full"  # random(0, delay) - decorrelated jitter


# Default exceptions that are considered retryable
DEFAULT_RETRYABLE_EXCEPTIONS: Tuple[Type[Exception], ...] = (
    ConnectionError,
    TimeoutError,
    OSError,
    IOError,
)


@dataclass
class RetryConfig:
    """Configuration for retry behavior.

    Attributes:
        max_retries: Maximum number of retry attempts (not counting initial try)
        base_delay: Base delay in seconds between retries
        max_delay: Maximum delay cap in seconds
        strategy: Backoff strategy to use
        jitter_mode: How to apply jitter to delays
        jitter_factor: Jitter factor (for multiplicative mode, 0.25 = ±25%)
        retryable_exceptions: Tuple of exception types to retry on
        on_retry: Optional callback called on each retry (attempt, exception, delay)
        should_retry: Optional function to determine if specific exception should be retried
    """

    max_retries: int = 3
    base_delay: float = 0.1
    max_delay: float = 30.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    jitter_mode: JitterMode = JitterMode.MULTIPLICATIVE
    jitter_factor: float = 0.25  # ±25% jitter
    retryable_exceptions: Tuple[Type[Exception], ...] = DEFAULT_RETRYABLE_EXCEPTIONS
    on_retry: Optional[Callable[[int, Exception, float], None]] = None
    should_retry: Optional[Callable[[Exception], bool]] = None

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for a given attempt number (0-indexed).

        Args:
            attempt: The attempt number (0 = first retry, 1 = second retry, etc.)

        Returns:
            Delay in seconds with jitter applied
        """
        return calculate_backoff_delay(
            attempt=attempt,
            base_delay=self.base_delay,
            max_delay=self.max_delay,
            strategy=self.strategy,
            jitter_mode=self.jitter_mode,
            jitter_factor=self.jitter_factor,
        )

    def is_retryable(self, exception: Exception) -> bool:
        """Check if an exception should trigger a retry.

        Args:
            exception: The exception to check

        Returns:
            True if the exception should be retried
        """
        if self.should_retry is not None:
            return self.should_retry(exception)
        return isinstance(exception, self.retryable_exceptions)


def calculate_backoff_delay(
    attempt: int,
    base_delay: float = 0.1,
    max_delay: float = 30.0,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
    jitter_mode: JitterMode = JitterMode.MULTIPLICATIVE,
    jitter_factor: float = 0.25,
) -> float:
    """Calculate backoff delay with configurable strategy and jitter.

    This is the core delay calculation function used by RetryConfig
    and can be used standalone.

    Args:
        attempt: The attempt number (0-indexed)
        base_delay: Base delay in seconds
        max_delay: Maximum delay cap
        strategy: Backoff strategy
        jitter_mode: Jitter mode
        jitter_factor: Jitter factor for multiplicative mode

    Returns:
        Delay in seconds
    """
    # Calculate base delay based on strategy
    if strategy == RetryStrategy.EXPONENTIAL:
        delay = base_delay * (2**attempt)
    elif strategy == RetryStrategy.LINEAR:
        delay = base_delay * (attempt + 1)
    elif strategy == RetryStrategy.FIBONACCI:
        fib = _fibonacci(attempt + 2)  # Start with fib(2) = 1
        delay = base_delay * fib
    else:  # CONSTANT
        delay = base_delay

    # Cap at max delay
    delay = min(delay, max_delay)

    # Apply jitter
    if jitter_mode == JitterMode.ADDITIVE:
        jitter = random.random() * base_delay * jitter_factor
        delay = delay + jitter
    elif jitter_mode == JitterMode.MULTIPLICATIVE:
        factor = 1.0 + (random.random() * 2 - 1) * jitter_factor
        delay = delay * factor
    elif jitter_mode == JitterMode.FULL:
        delay = random.random() * delay
    # NONE: no jitter applied

    return max(0, delay)


def _fibonacci(n: int) -> int:
    """Calculate nth Fibonacci number."""
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


class ExponentialBackoff:
    """Iterator-based exponential backoff for manual retry loops.

    Usage:
        backoff = ExponentialBackoff(max_retries=3)
        for delay in backoff:
            try:
                result = risky_operation()
                break
            except ConnectionError:
                time.sleep(delay)
        else:
            raise Exception("All retries exhausted")
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 0.1,
        max_delay: float = 30.0,
        jitter: bool = True,
    ):
        """Initialize backoff iterator.

        Args:
            max_retries: Maximum number of retries
            base_delay: Initial delay in seconds
            max_delay: Maximum delay cap
            jitter: Whether to apply jitter
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter
        self._attempt = 0

    def __iter__(self) -> "ExponentialBackoff":
        self._attempt = 0
        return self

    def __next__(self) -> float:
        if self._attempt >= self.max_retries:
            raise StopIteration

        delay = calculate_backoff_delay(
            attempt=self._attempt,
            base_delay=self.base_delay,
            max_delay=self.max_delay,
            strategy=RetryStrategy.EXPONENTIAL,
            jitter_mode=JitterMode.MULTIPLICATIVE if self.jitter else JitterMode.NONE,
        )
        self._attempt += 1
        return delay

    def reset(self) -> None:
        """Reset the backoff iterator."""
        self._attempt = 0


def with_retry(
    config: Optional[RetryConfig] = None,
    *,
    max_retries: int = 3,
    base_delay: float = 0.1,
    retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """Decorator for async functions with retry logic.

    Can be used with a RetryConfig or with keyword arguments for convenience.

    Args:
        config: RetryConfig instance (if provided, other args are ignored)
        max_retries: Maximum retry attempts
        base_delay: Base delay between retries
        retryable_exceptions: Exceptions to retry on

    Returns:
        Decorator function

    Example:
        @with_retry(max_retries=3)
        async def flaky_call():
            ...

        @with_retry(RetryConfig(strategy=RetryStrategy.LINEAR))
        async def another_call():
            ...
    """
    if config is None:
        config = RetryConfig(
            max_retries=max_retries,
            base_delay=base_delay,
            retryable_exceptions=retryable_exceptions or DEFAULT_RETRYABLE_EXCEPTIONS,
        )

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exception: Optional[Exception] = None

            for attempt in range(config.max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except config.retryable_exceptions as e:
                    last_exception = e

                    if not config.is_retryable(e):
                        raise

                    if attempt >= config.max_retries:
                        logger.warning(
                            f"Retry exhausted for {func.__name__} after {attempt + 1} attempts: {e}"
                        )
                        raise

                    delay = config.calculate_delay(attempt)
                    logger.debug(
                        f"Retry {attempt + 1}/{config.max_retries} for {func.__name__} "
                        f"after {delay:.2f}s: {e}"
                    )

                    if config.on_retry:
                        config.on_retry(attempt, e, delay)

                    await asyncio.sleep(delay)

            # Should not reach here, but for type safety
            if last_exception:
                raise last_exception
            raise RuntimeError("Unexpected retry state")

        return wrapper

    return decorator


def with_retry_sync(
    config: Optional[RetryConfig] = None,
    *,
    max_retries: int = 3,
    base_delay: float = 0.1,
    retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator for sync functions with retry logic.

    Same as with_retry but for synchronous functions.

    Args:
        config: RetryConfig instance
        max_retries: Maximum retry attempts
        base_delay: Base delay between retries
        retryable_exceptions: Exceptions to retry on

    Returns:
        Decorator function
    """
    if config is None:
        config = RetryConfig(
            max_retries=max_retries,
            base_delay=base_delay,
            retryable_exceptions=retryable_exceptions or DEFAULT_RETRYABLE_EXCEPTIONS,
        )

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exception: Optional[Exception] = None

            for attempt in range(config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except config.retryable_exceptions as e:
                    last_exception = e

                    if not config.is_retryable(e):
                        raise

                    if attempt >= config.max_retries:
                        logger.warning(
                            f"Retry exhausted for {func.__name__} after {attempt + 1} attempts: {e}"
                        )
                        raise

                    delay = config.calculate_delay(attempt)
                    logger.debug(
                        f"Retry {attempt + 1}/{config.max_retries} for {func.__name__} "
                        f"after {delay:.2f}s: {e}"
                    )

                    if config.on_retry:
                        config.on_retry(attempt, e, delay)

                    time.sleep(delay)

            if last_exception:
                raise last_exception
            raise RuntimeError("Unexpected retry state")

        return wrapper

    return decorator
