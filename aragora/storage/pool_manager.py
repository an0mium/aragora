"""
PostgreSQL Pool Manager - Event-loop aware pool lifecycle management.

Provides centralized pool creation and access that respects asyncpg's
event loop binding requirements. The shared pool MUST be initialized
early in the server startup sequence (after asyncio.run() starts).

Usage:
    # In server startup (inside async context):
    await initialize_shared_pool()

    # In store factories:
    if is_pool_initialized():
        pool = get_shared_pool()
        store = PostgresMyStore(pool)

    # In shutdown:
    await close_shared_pool()
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from asyncpg import Pool


logger = logging.getLogger(__name__)

# Global pool state
_shared_pool: Optional["Pool"] = None
_pool_event_loop: asyncio.AbstractEventLoop | None = None
_pool_config: dict[str, Any] = {}


def _is_shared_pool_enabled() -> bool:
    """Check if shared pool feature is enabled via environment variable."""
    return os.environ.get("ARAGORA_USE_SHARED_POOL", "true").lower() in ("true", "1", "yes")


async def initialize_shared_pool(
    dsn: str | None = None,
    min_size: int = 5,
    max_size: int = 20,
    command_timeout: float = 60.0,
    statement_timeout: int = 60,
    force: bool = False,
) -> Optional["Pool"]:
    """
    Initialize the shared PostgreSQL pool in the current event loop.

    MUST be called from within an async context (after asyncio.run() starts).
    The pool will be bound to the current event loop and cannot be used
    from other event loops.

    If a stale pool exists (e.g., created on a temporary event loop during
    module imports), it will be closed and replaced with a fresh pool
    bound to the current (correct) event loop.

    Args:
        dsn: PostgreSQL connection string. If not provided, uses environment.
        min_size: Minimum pool connections (default 5)
        max_size: Maximum pool connections (default 20)
        command_timeout: Command timeout in seconds (default 60)
        statement_timeout: PostgreSQL statement_timeout in seconds (default 60)

    Returns:
        Pool instance if PostgreSQL is configured and initialization succeeds,
        None if PostgreSQL is not configured or shared pool is disabled.

    Raises:
        RuntimeError: If pool creation fails (after logging error)
    """
    global _shared_pool, _pool_event_loop, _pool_config

    # Check if feature is enabled
    if not _is_shared_pool_enabled():
        logger.info("[pool_manager] Shared pool disabled via ARAGORA_USE_SHARED_POOL=false")
        return None

    # Check if already initialized on THIS event loop
    if _shared_pool is not None and not force:
        current_loop = asyncio.get_running_loop()
        if _pool_event_loop is current_loop:
            logger.debug("[pool_manager] Shared pool already initialized on this loop")
            return _shared_pool

    # Force-close the existing pool when refreshing due to stale connections
    if force and _shared_pool is not None:
        logger.warning("[pool_manager] Force-reinitializing shared pool (stale connections)")
        try:
            await _shared_pool.close()
        except Exception as close_err:
            logger.warning("[pool_manager] Error closing stale pool: %s", close_err)
            try:
                _shared_pool.terminate()
            except Exception:
                pass
        _shared_pool = None
        _pool_event_loop = None

    # Check if PostgreSQL is configured
    from aragora.storage.connection_factory import (
        StorageBackendType,
        resolve_database_config,
    )

    config = resolve_database_config("shared", allow_sqlite=True)
    if config.backend_type == StorageBackendType.SQLITE:
        logger.info("[pool_manager] No PostgreSQL configured, skipping pool creation")
        return None

    # Create pool in current event loop
    try:
        from aragora.storage import postgres_store as _ps_mod

        effective_dsn = dsn or config.dsn
        current_loop = asyncio.get_running_loop()

        # Close any stale pool created on a different event loop.
        # This happens when module-level code (e.g., workflows.py template
        # registration) triggers get_postgres_pool() via run_async() /
        # asyncio.run(), which creates a temporary event loop. That pool
        # is then bound to the now-closed temp loop and is unusable.
        if _ps_mod._pool is not None:
            logger.warning(
                "[pool_manager] Clearing stale postgres_store._pool "
                "(created on different event loop) before shared pool init"
            )
            # Don't terminate() the stale pool - it was created on a now-dead
            # temporary event loop (from asyncio.run() during module import).
            # Calling terminate() sends disconnect commands to the database
            # server which can cause Supabase to temporarily reject new
            # connections ("Tenant or user not found"). Just null the reference
            # and let GC collect it - the dead loop means its connections are
            # already effectively orphaned.
            _ps_mod._pool = None

        # Also reset _shared_pool if it references the same stale pool
        if _shared_pool is not None:
            _shared_pool = None
            _pool_event_loop = None

        # Now create a fresh pool on the current event loop (with retry)
        from aragora.storage.postgres_store import get_postgres_pool

        last_err: Exception | None = None
        for attempt in range(3):
            try:
                _ps_mod._pool = None  # Ensure clean state for each attempt
                _shared_pool = await get_postgres_pool(
                    dsn=effective_dsn,
                    min_size=min_size,
                    max_size=max_size,
                    command_timeout=command_timeout,
                    statement_timeout=statement_timeout,
                )
                break
            except Exception as e:
                last_err = e
                logger.warning(f"[pool_manager] Pool creation attempt {attempt + 1}/3 failed: {e}")
                _ps_mod._pool = None
                if attempt < 2:
                    await asyncio.sleep(2.0 * (attempt + 1))
        else:
            # last_err is guaranteed to be set here since we only reach the else
            # branch if we never broke out of the loop (i.e., all attempts failed)
            if last_err is None:
                raise RuntimeError("Connection retry loop completed without error - logic error")
            raise last_err
        _pool_event_loop = current_loop
        _pool_config = {
            "dsn_hash": hash(effective_dsn) if effective_dsn else None,
            "min_size": min_size,
            "max_size": max_size,
            "is_supabase": config.is_supabase,
        }

        # Apply nest_asyncio to allow nested run_until_complete() calls.
        # This is required because PostgreSQL store sync wrappers use
        # run_async() which needs to work
        # from within handler coroutines running on this event loop.
        try:
            import nest_asyncio

            nest_asyncio.apply(_pool_event_loop)
            logger.warning("[pool_manager] nest_asyncio applied to main event loop")
        except ImportError:
            logger.warning(
                "[pool_manager] nest_asyncio not installed. "
                "Sync store wrappers may deadlock in async contexts. "
                "Install with: pip install nest_asyncio"
            )

        # Verify pool health
        try:
            async with _shared_pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
        except Exception as e:
            logger.error(f"[pool_manager] Pool health check failed after creation: {e}")
            # Pool is broken, clean up and fall back
            try:
                _shared_pool.terminate()
            except Exception as e:
                logger.warning("Failed to terminate broken database pool: %s", e)
            _shared_pool = None
            _pool_event_loop = None
            _pool_config = {}
            _ps_mod._pool = None
            return None

        pool_size = _shared_pool.get_size() if hasattr(_shared_pool, "get_size") else "unknown"
        backend_name = "Supabase" if config.is_supabase else "PostgreSQL"
        logger.warning(
            f"[pool_manager] Shared {backend_name} pool initialized "
            f"(size: {pool_size}, event_loop: {id(_pool_event_loop)})"
        )
        return _shared_pool

    except Exception as e:
        logger.error(f"[pool_manager] Failed to initialize shared pool: {e}")
        # Don't raise - let caller handle fallback to SQLite
        return None


def get_shared_pool() -> Optional["Pool"]:
    """
    Get the shared pool, validating it's being accessed from the correct event loop.

    Returns:
        Pool if available and event loop matches, None if not initialized.

    Raises:
        RuntimeError: If accessed from a different event loop than it was created in.
    """
    global _shared_pool, _pool_event_loop

    if _shared_pool is None:
        return None

    # Validate event loop when in async context
    try:
        current_loop = asyncio.get_running_loop()
        if _pool_event_loop and current_loop != _pool_event_loop:
            raise RuntimeError(
                f"Shared pool was created in event loop {id(_pool_event_loop)} "
                f"but accessed from {id(current_loop)}. "
                "asyncpg pools are event-loop bound and cannot be shared across loops."
            )
    except RuntimeError as e:
        if "no running event loop" in str(e).lower():
            # Sync context - pool access is valid but operations must be awaited
            pass
        else:
            raise

    return _shared_pool


def is_pool_initialized() -> bool:
    """Check if the shared pool has been initialized."""
    return _shared_pool is not None and _is_shared_pool_enabled()


def get_pool_event_loop() -> asyncio.AbstractEventLoop | None:
    """Get the event loop the shared pool was created in (the main event loop).

    This is used by handler_registry to schedule async handler coroutines
    on the correct event loop via asyncio.run_coroutine_threadsafe().

    Returns:
        The main event loop if pool is initialized, None otherwise.
    """
    return _pool_event_loop


def get_pool_info() -> dict[str, Any]:
    """Get information about the shared pool for diagnostics."""
    global _shared_pool, _pool_event_loop, _pool_config

    if _shared_pool is None:
        return {
            "initialized": False,
            "enabled": _is_shared_pool_enabled(),
        }

    pool_size = _shared_pool.get_size() if hasattr(_shared_pool, "get_size") else None
    free_size = _shared_pool.get_idle_size() if hasattr(_shared_pool, "get_idle_size") else None

    return {
        "initialized": True,
        "enabled": True,
        "pool_size": pool_size,
        "free_connections": free_size,
        "event_loop_id": id(_pool_event_loop) if _pool_event_loop else None,
        "is_supabase": _pool_config.get("is_supabase", False),
        "min_size": _pool_config.get("min_size"),
        "max_size": _pool_config.get("max_size"),
    }


async def close_shared_pool() -> None:
    """
    Close the shared pool during shutdown.

    Safe to call multiple times or when pool is not initialized.
    """
    global _shared_pool, _pool_event_loop, _pool_config

    if _shared_pool is None:
        logger.debug("[pool_manager] No shared pool to close")
        return

    try:
        await _shared_pool.close()
        logger.info("[pool_manager] Shared pool closed")
    except Exception as e:
        logger.warning(f"[pool_manager] Error closing shared pool: {e}")
    finally:
        _shared_pool = None
        _pool_event_loop = None
        _pool_config = {}


def reset_shared_pool() -> None:
    """
    Reset pool state without closing (for testing only).

    WARNING: This does NOT close the pool properly. Only use in tests.
    """
    global _shared_pool, _pool_event_loop, _pool_config
    _shared_pool = None
    _pool_event_loop = None
    _pool_config = {}


__all__ = [
    "initialize_shared_pool",
    "get_shared_pool",
    "get_pool_event_loop",
    "is_pool_initialized",
    "get_pool_info",
    "close_shared_pool",
    "reset_shared_pool",
]
