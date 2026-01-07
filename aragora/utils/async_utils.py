"""
Async utility functions for safe sync/async bridging.

Provides utilities for running async code from sync contexts, handling
the common case where code may be called from either sync or async contexts.
"""

import asyncio
import concurrent.futures
import logging
from typing import Any, Coroutine, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def run_async(coro: Coroutine[Any, Any, T], timeout: float = 30.0) -> T:
    """Run async coroutine from sync context, handling nested event loops.

    This handles the case where we're in a sync context but need to call async code.
    It properly handles:
    1. No running event loop - uses asyncio.run() directly
    2. Running event loop - uses ThreadPoolExecutor to avoid nested loop issues

    Args:
        coro: Coroutine to execute
        timeout: Maximum time to wait (seconds), default 30

    Returns:
        Result from the coroutine

    Raises:
        Exception: Any exception from the coroutine
        TimeoutError: If execution exceeds timeout
    """
    try:
        # Check if there's a running loop (avoids deprecation warning)
        try:
            loop = asyncio.get_running_loop()
            # If we get here, there's a running loop - can't use run_until_complete
            # Use ThreadPoolExecutor to run in a new thread with its own loop
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result(timeout=timeout)
        except RuntimeError:
            # No running loop, safe to use asyncio.run()
            return asyncio.run(coro)
    except (RuntimeError, asyncio.InvalidStateError) as e:
        # Fallback: create new event loop for edge cases
        logger.debug(f"Creating new event loop after: {type(e).__name__}: {e}")
        return asyncio.run(coro)


__all__ = ["run_async"]
