"""Timeout utilities for preventing deadlocks and hanging operations.

This module provides context managers and utilities for operations that
need timeout protection, particularly for thread synchronization primitives.
"""

import asyncio
import threading
from contextlib import contextmanager
from typing import Generator, Optional

# Default timeout for lock acquisition (seconds)
DEFAULT_LOCK_TIMEOUT = 30.0


@contextmanager
def timed_lock(
    lock: threading.Lock,
    timeout: float = DEFAULT_LOCK_TIMEOUT,
    *,
    name: Optional[str] = None,
) -> Generator[None, None, None]:
    """Acquire a threading lock with a timeout.

    This context manager prevents deadlocks by raising TimeoutError
    if the lock cannot be acquired within the specified timeout.

    Args:
        lock: The threading.Lock (or RLock) to acquire
        timeout: Maximum time to wait for lock acquisition (seconds)
        name: Optional name for the lock (used in error messages)

    Yields:
        None (the lock is held during the context)

    Raises:
        TimeoutError: If the lock cannot be acquired within timeout

    Usage:
        lock = threading.Lock()
        with timed_lock(lock, timeout=10.0, name="database"):
            # Critical section
            do_database_operation()

    Note:
        This is essential for preventing test hangs when database
        singletons or other shared resources use threading locks.
    """
    acquired = lock.acquire(timeout=timeout)
    if not acquired:
        lock_name = f" '{name}'" if name else ""
        raise TimeoutError(
            f"Failed to acquire lock{lock_name} within {timeout}s. " "This may indicate a deadlock."
        )
    try:
        yield
    finally:
        lock.release()


@contextmanager
def timed_rlock(
    lock: threading.RLock,
    timeout: float = DEFAULT_LOCK_TIMEOUT,
    *,
    name: Optional[str] = None,
) -> Generator[None, None, None]:
    """Acquire a threading RLock with a timeout.

    Same as timed_lock but explicitly for reentrant locks.

    Args:
        lock: The threading.RLock to acquire
        timeout: Maximum time to wait for lock acquisition (seconds)
        name: Optional name for the lock (used in error messages)

    Yields:
        None (the lock is held during the context)

    Raises:
        TimeoutError: If the lock cannot be acquired within timeout
    """
    acquired = lock.acquire(timeout=timeout)
    if not acquired:
        lock_name = f" '{name}'" if name else ""
        raise TimeoutError(
            f"Failed to acquire RLock{lock_name} within {timeout}s. "
            "This may indicate a deadlock or excessive recursive locking."
        )
    try:
        yield
    finally:
        lock.release()


async def async_timeout(
    coro,
    timeout: float = DEFAULT_LOCK_TIMEOUT,
    *,
    operation_name: Optional[str] = None,
):
    """Run an async operation with a timeout.

    Wrapper around asyncio.wait_for with better error messages.

    Args:
        coro: The coroutine to run
        timeout: Maximum time to wait (seconds)
        operation_name: Optional name for the operation (used in error messages)

    Returns:
        The result of the coroutine

    Raises:
        TimeoutError: If the operation does not complete within timeout
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        op_name = f" ({operation_name})" if operation_name else ""
        raise TimeoutError(f"Async operation{op_name} timed out after {timeout}s") from None


__all__ = [
    "DEFAULT_LOCK_TIMEOUT",
    "timed_lock",
    "timed_rlock",
    "async_timeout",
]
