"""Managed resource utilities for test fixtures.

Provides a universal pattern for fixture cleanup, ensuring resources
are properly closed even when tests fail or hang.
"""

import logging
from contextlib import contextmanager
from typing import Any, Callable, Generator, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@contextmanager
def managed_fixture(
    resource: T,
    cleanup_fn: Optional[Callable[[], None]] = None,
    *,
    name: Optional[str] = None,
) -> Generator[T, None, None]:
    """Universal fixture wrapper guaranteeing resource cleanup.

    This context manager ensures cleanup runs even if:
    - The test fails with an exception
    - The test times out
    - Assertions fail

    Args:
        resource: The resource to manage (any object)
        cleanup_fn: Optional explicit cleanup function. If not provided,
                   will call resource.close() if available.
        name: Optional name for logging purposes

    Yields:
        The resource, unchanged

    Usage:
        @pytest.fixture
        def elo_system(temp_db):
            system = EloSystem(db_path=temp_db)
            with managed_fixture(system, system.close, name="EloSystem"):
                yield system

        @pytest.fixture
        def db_connection(temp_db):
            conn = sqlite3.connect(temp_db)
            with managed_fixture(conn):  # Will auto-call conn.close()
                yield conn
    """
    resource_name = name or type(resource).__name__
    try:
        yield resource
    finally:
        _cleanup_resource(resource, cleanup_fn, resource_name)


def _cleanup_resource(
    resource: Any,
    cleanup_fn: Optional[Callable[[], None]],
    name: str,
) -> None:
    """Execute cleanup for a resource, handling errors gracefully."""
    try:
        if cleanup_fn is not None:
            cleanup_fn()
        elif hasattr(resource, "close"):
            close_result = resource.close()
            # Handle async close methods that return coroutines
            if hasattr(close_result, "__await__"):
                import asyncio

                try:
                    loop = asyncio.get_running_loop()
                    loop.run_until_complete(close_result)
                except RuntimeError:
                    # No running loop - create one
                    asyncio.run(close_result)
    except Exception as e:
        # Log but don't fail teardown - we want other cleanup to continue
        logger.debug(f"Cleanup error for {name}: {e}")


@contextmanager
def managed_async_fixture(
    resource: T,
    cleanup_coro_fn: Optional[Callable[[], Any]] = None,
    *,
    name: Optional[str] = None,
) -> Generator[T, None, None]:
    """Fixture wrapper for async resources needing async cleanup.

    Similar to managed_fixture but handles async cleanup functions.

    Args:
        resource: The resource to manage
        cleanup_coro_fn: Async cleanup function (returns coroutine)
        name: Optional name for logging

    Usage:
        @pytest.fixture
        async def async_store(temp_db):
            store = AsyncStore(db_path=temp_db)
            await store.connect()
            with managed_async_fixture(store, store.disconnect):
                yield store
    """
    import asyncio

    resource_name = name or type(resource).__name__
    try:
        yield resource
    finally:
        try:
            if cleanup_coro_fn is not None:
                coro = cleanup_coro_fn()
                if hasattr(coro, "__await__"):
                    try:
                        loop = asyncio.get_running_loop()
                        loop.run_until_complete(coro)
                    except RuntimeError:
                        asyncio.run(coro)
            elif hasattr(resource, "close"):
                close_result = resource.close()
                if hasattr(close_result, "__await__"):
                    try:
                        loop = asyncio.get_running_loop()
                        loop.run_until_complete(close_result)
                    except RuntimeError:
                        asyncio.run(close_result)
        except Exception as e:
            logger.debug(f"Async cleanup error for {resource_name}: {e}")


__all__ = ["managed_fixture", "managed_async_fixture"]
