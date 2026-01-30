"""
Async utility functions for safe sync/async bridging.

Provides utilities for running async code from sync contexts, handling
the common case where code may be called from either sync or async contexts.

Also includes async subprocess utilities for non-blocking command execution.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Coroutine, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def run_async(coro: Coroutine[Any, Any, T], timeout: float = 30.0) -> T:
    """Run async coroutine from sync context ONLY.

    IMPORTANT: This function should ONLY be called from synchronous code.
    If called from an async context, it will raise RuntimeError to prevent
    event loop cross-contamination that breaks asyncpg connection pools.

    For async code, use `await coro` directly instead of `run_async(coro)`.

    Args:
        coro: Coroutine to execute
        timeout: Maximum time to wait (seconds), default 30

    Returns:
        Result from the coroutine

    Raises:
        RuntimeError: If called from within an async context
        Exception: Any exception from the coroutine
        TimeoutError: If execution exceeds timeout
    """
    # Check if there's a running loop - if so, FAIL FAST
    # Using ThreadPoolExecutor with asyncio.run() creates a new event loop,
    # which breaks asyncpg pools (they're bound to specific event loops).
    try:
        loop = asyncio.get_running_loop()
        # We're in an async context - this is a caller bug
        # Close the coroutine to prevent "coroutine was never awaited" warning
        coro.close()
        raise RuntimeError(
            "run_async() cannot be called from an async context. "
            "asyncpg connection pools are bound to specific event loops. "
            "Use 'await coro' directly instead of 'run_async(coro)'. "
            f"Current event loop: {loop}"
        )
    except RuntimeError as e:
        if "no running event loop" in str(e).lower():
            # No running loop - safe to use asyncio.run()
            return asyncio.run(coro)
        # Re-raise other RuntimeErrors (including our own from above)
        # Close the coroutine to prevent warnings
        coro.close()
        raise


# Semaphore to limit concurrent subprocess calls (prevent resource exhaustion)
_subprocess_semaphore = asyncio.Semaphore(10)


async def run_command(
    cmd: list[str],
    cwd: Path | None = None,
    timeout: float = 60.0,
    input_data: bytes | None = None,
) -> tuple[int, bytes, bytes]:
    """Run command asynchronously without blocking event loop.

    Uses asyncio.create_subprocess_exec for non-blocking execution.
    Limits concurrent subprocess calls to prevent resource exhaustion.

    Args:
        cmd: Command and arguments as list
        cwd: Optional working directory
        timeout: Timeout in seconds (default 60)
        input_data: Optional stdin data

    Returns:
        Tuple of (return_code, stdout, stderr)

    Raises:
        asyncio.TimeoutError: If command exceeds timeout
        FileNotFoundError: If command not found
    """
    async with _subprocess_semaphore:
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE if input_data else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(input_data), timeout=timeout)
            return proc.returncode or 0, stdout, stderr
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise


async def run_git_command(args: list[str], cwd: Path, timeout: float = 30.0) -> tuple[bool, str]:
    """Run git command asynchronously.

    Convenience wrapper for common git operations.

    Args:
        args: Git subcommand and arguments (e.g., ["status", "-s"])
        cwd: Repository directory
        timeout: Timeout in seconds (default 30)

    Returns:
        Tuple of (success: bool, output_or_error: str)
    """
    try:
        returncode, stdout, stderr = await run_command(["git"] + args, cwd=cwd, timeout=timeout)
        if returncode == 0:
            return True, stdout.decode(errors="replace")
        return False, stderr.decode(errors="replace")
    except asyncio.TimeoutError:
        return False, "Git command timed out"
    except FileNotFoundError:
        return False, "Git not found"
    except Exception as e:
        return False, str(e)


def get_event_loop_safe() -> asyncio.AbstractEventLoop:
    """Get event loop safely, handling the deprecation of asyncio.get_event_loop().

    This function handles the Python 3.10+ deprecation of get_event_loop() which
    emits a DeprecationWarning when called outside an async context.

    Returns:
        The running event loop if available, otherwise creates a new one.

    Note:
        In async code, prefer asyncio.get_running_loop() directly.
        This function is mainly for sync code that needs to schedule work
        on an event loop.
    """
    try:
        # First try to get a running loop (we're in async context)
        return asyncio.get_running_loop()
    except RuntimeError:
        # No running loop - we're in sync context
        # In Python 3.10+, get_event_loop() is deprecated when no loop is running
        # Use get_event_loop() anyway as it still works and creates/returns a loop
        # for the main thread, or use new_event_loop for other threads
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            return loop
        except RuntimeError:
            # No current event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop


def run_sync_in_async_context(func: Any, *args: Any, **kwargs: Any) -> asyncio.Future:
    """Run a synchronous function in the executor from an async context.

    This is the proper replacement for:
        asyncio.get_event_loop().run_in_executor(None, func, *args)

    Args:
        func: Synchronous callable to run
        *args: Positional arguments for func
        **kwargs: Keyword arguments for func (will use functools.partial)

    Returns:
        Future that resolves to the function's return value

    Raises:
        RuntimeError: If not called from an async context
    """
    import functools

    loop = asyncio.get_running_loop()  # Raises if not in async context
    if kwargs:
        func = functools.partial(func, **kwargs)
    return loop.run_in_executor(None, func, *args)


def schedule_background_task(coro: Coroutine[Any, Any, T]) -> None:
    """Schedule a coroutine to run in the background (fire-and-forget).

    Works from both sync and async contexts. In async context, uses create_task.
    In sync context, schedules on the event loop.

    Args:
        coro: Coroutine to execute in background

    Note:
        The coroutine's result is ignored. Exceptions are logged but not raised.
    """
    try:
        loop = asyncio.get_running_loop()
        task = loop.create_task(coro)
        task.add_done_callback(_log_task_exception)
    except RuntimeError:
        # No running loop - need to handle differently
        loop = get_event_loop_safe()
        if loop.is_running():
            loop.call_soon_threadsafe(lambda: loop.create_task(coro))
        else:
            # Run the coroutine blocking if no loop is available
            try:
                asyncio.run(coro)
            except (OSError, RuntimeError, ValueError) as e:
                logger.warning(f"Background task failed: {e}")


def _log_task_exception(task: asyncio.Task) -> None:
    """Callback to log exceptions from background tasks."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error(f"Background task failed: {exc}", exc_info=exc)


__all__ = [
    "get_event_loop_safe",
    "run_async",
    "run_command",
    "run_git_command",
    "run_sync_in_async_context",
    "schedule_background_task",
]
