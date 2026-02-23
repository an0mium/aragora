"""
Database initialization for server startup.

Handles early PostgreSQL pool creation to ensure pools are bound
to the correct event loop before subsystems are initialized.

This module MUST be called early in the startup sequence, BEFORE
any subsystems that need database access are initialized.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


async def init_postgres_pool() -> dict[str, Any]:
    """
    Initialize PostgreSQL connection pool during server startup.

    This MUST be called early in the startup sequence, BEFORE any
    subsystems that need database access are initialized. The pool
    is bound to the current event loop.

    Environment Variables:
        ARAGORA_USE_SHARED_POOL: Set to "false" to disable (default: "true")
        ARAGORA_DB_BACKEND: Backend selection ("supabase", "postgres", "sqlite")
        DATABASE_URL: PostgreSQL connection string
        SUPABASE_URL + SUPABASE_DB_PASSWORD: Supabase credentials

    Returns:
        Status dict with pool initialization info:
        - enabled: bool - Whether PostgreSQL is enabled
        - backend: str - Backend type ("supabase", "postgres", "sqlite")
        - pool_size: int - Current pool size (if enabled)
        - error: str - Error message (if initialization failed)
    """
    # Check if shared pool feature is enabled
    use_shared_pool = os.environ.get("ARAGORA_USE_SHARED_POOL", "true").lower() in (
        "true",
        "1",
        "yes",
    )
    if not use_shared_pool:
        logger.info("[startup/db] Shared pool disabled via ARAGORA_USE_SHARED_POOL=false")
        return {"enabled": False, "backend": "sqlite", "reason": "disabled_by_env"}

    # Check configured backend
    try:
        from aragora.storage.factory import StorageBackend, get_storage_backend

        backend = get_storage_backend()
    except ImportError:
        backend = None

    if backend is None or backend == StorageBackend.SQLITE:
        logger.info("[startup/db] PostgreSQL not configured, using SQLite")
        return {"enabled": False, "backend": "sqlite"}

    # Initialize the shared pool
    try:
        from aragora.storage.pool_manager import get_pool_info, initialize_shared_pool

        # Get pool configuration from environment
        min_size = int(os.environ.get("ARAGORA_POOL_MIN_SIZE", "5"))
        max_size = int(os.environ.get("ARAGORA_POOL_MAX_SIZE", "20"))
        command_timeout = float(os.environ.get("ARAGORA_POOL_COMMAND_TIMEOUT", "60.0"))
        statement_timeout = int(os.environ.get("ARAGORA_POOL_STATEMENT_TIMEOUT", "60"))

        pool = await initialize_shared_pool(
            min_size=min_size,
            max_size=max_size,
            command_timeout=command_timeout,
            statement_timeout=statement_timeout,
        )

        if pool:
            info = get_pool_info()
            logger.warning(
                "[startup/db] PostgreSQL pool initialized (backend: %s, size: %s)",
                backend.value,
                info.get("pool_size", "unknown"),
            )
            return {
                "enabled": True,
                "backend": backend.value,
                "pool_size": info.get("pool_size"),
                "is_supabase": info.get("is_supabase", False),
            }
        else:
            logger.warning("[startup/db] Pool initialization returned None, falling back to SQLite")
            return {"enabled": False, "backend": "sqlite", "reason": "pool_init_returned_none"}

    except (OSError, RuntimeError, ValueError, ConnectionError, TimeoutError) as e:
        logger.error("[startup/db] PostgreSQL pool initialization failed: %s", e)
        return {
            "enabled": False,
            "backend": "sqlite",
            "error": "PostgreSQL pool initialization failed",
            "reason": "initialization_failed",
        }


async def close_postgres_pool() -> None:
    """
    Close PostgreSQL connection pool during shutdown.

    Safe to call even if pool was never initialized.
    """
    try:
        from aragora.storage.pool_manager import close_shared_pool

        await close_shared_pool()
        logger.info("[startup/db] PostgreSQL pool closed")
    except ImportError:
        pass  # pool_manager not available
    except (OSError, RuntimeError, ConnectionError) as e:
        logger.warning("[startup/db] Error closing PostgreSQL pool: %s", e)


__all__ = ["init_postgres_pool", "close_postgres_pool"]
