"""
Async utility functions for safe sync/async bridging.

Provides utilities for running async code from sync contexts, handling
the common case where code may be called from either sync or async contexts.

Also includes async subprocess utilities for non-blocking command execution.
"""

import asyncio
import concurrent.futures
import logging
from pathlib import Path
from typing import Any, Coroutine, List, Optional, Tuple, TypeVar

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


# Semaphore to limit concurrent subprocess calls (prevent resource exhaustion)
_subprocess_semaphore = asyncio.Semaphore(10)


async def run_command(
    cmd: List[str],
    cwd: Optional[Path] = None,
    timeout: float = 60.0,
    input_data: Optional[bytes] = None,
) -> Tuple[int, bytes, bytes]:
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


async def run_git_command(args: List[str], cwd: Path, timeout: float = 30.0) -> Tuple[bool, str]:
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


__all__ = ["run_async", "run_command", "run_git_command"]
