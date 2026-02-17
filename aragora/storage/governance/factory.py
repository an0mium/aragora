"""
Factory functions for creating and managing governance store instances.

Provides singleton access to GovernanceStore (SQLite) and
PostgresGovernanceStore (async PostgreSQL) with automatic backend
selection based on environment configuration.
"""

from __future__ import annotations

import logging
import os

from .postgres_store import PostgresGovernanceStore
from .store import GovernanceStore

logger = logging.getLogger(__name__)


# Module-level singletons
_default_store: GovernanceStore | None = None
_postgres_store: PostgresGovernanceStore | None = None


def get_governance_store(
    db_path: str = "aragora_governance.db",
    backend: str | None = None,
    database_url: str | None = None,
) -> GovernanceStore | PostgresGovernanceStore:
    """
    Get or create the default GovernanceStore instance.

    Uses environment variables to configure (preference order):
    1. SUPABASE_URL + SUPABASE_DB_PASSWORD: Supabase PostgreSQL (preferred)
    2. ARAGORA_POSTGRES_DSN or DATABASE_URL: Self-hosted PostgreSQL
    3. SQLite: Last resort (with production guard)

    Backend overrides:
    - ARAGORA_GOVERNANCE_STORE_BACKEND: Store-specific override
    - ARAGORA_DB_BACKEND: Global database backend override

    Returns:
        Configured GovernanceStore or PostgresGovernanceStore instance
    """
    global _default_store, _postgres_store

    from aragora.storage.connection_factory import (
        resolve_database_config,
        StorageBackendType,
    )
    from aragora.utils.async_utils import run_async

    # Check store-specific backend first, then global database backend
    backend_type = os.environ.get("ARAGORA_GOVERNANCE_STORE_BACKEND")
    if not backend_type and backend is None:
        # Use connection factory to determine backend with preference order
        config = resolve_database_config("governance", allow_sqlite=True)
        if config.backend_type in (StorageBackendType.SUPABASE, StorageBackendType.POSTGRES):
            backend_type = "postgres"
        else:
            backend_type = "sqlite"
    elif backend:
        backend_type = backend.lower()
    else:
        backend_type = backend_type.lower() if backend_type else "sqlite"

    if backend_type in ("postgres", "postgresql", "supabase"):
        if _postgres_store is not None:
            return _postgres_store

        logger.info("Using PostgreSQL governance store")
        try:
            from aragora.storage.postgres_store import get_postgres_pool

            # Get DSN from connection factory (handles Supabase preference)
            config = resolve_database_config("governance", allow_sqlite=True)

            # Initialize PostgreSQL store with connection pool using run_async
            async def init_postgres_store():
                pool = await get_postgres_pool(dsn=config.dsn)
                store = PostgresGovernanceStore(pool)
                await store.initialize()
                return store

            _postgres_store = run_async(init_postgres_store())
            return _postgres_store
        except (OSError, RuntimeError, ValueError) as e:
            logger.warning(f"PostgreSQL not available, falling back to SQLite: {e}")

            # Enforce distributed storage in production for RBAC policies
            from aragora.storage.production_guards import (
                require_distributed_store,
                StorageMode,
            )

            require_distributed_store(
                "governance_store",
                StorageMode.SQLITE,
                f"RBAC/governance policies must use distributed storage in production. "
                f"PostgreSQL unavailable: {e}",
            )
            # Fall through to SQLite

    if _default_store is None:
        # Enforce distributed storage in production for RBAC policies
        from aragora.storage.production_guards import (
            require_distributed_store,
            StorageMode,
        )

        require_distributed_store(
            "governance_store",
            StorageMode.SQLITE,
            "RBAC/governance policies must use distributed storage in production. "
            "Configure Supabase or PostgreSQL.",
        )
        _default_store = GovernanceStore(
            db_path=db_path,
            backend=backend,
            database_url=database_url,
        )
    return _default_store


def reset_governance_store() -> None:
    """Reset the default store instance (for testing)."""
    global _default_store, _postgres_store
    if _default_store is not None:
        _default_store.close()
        _default_store = None
    if _postgres_store is not None:
        _postgres_store.close()
        _postgres_store = None


__all__ = [
    "get_governance_store",
    "reset_governance_store",
]
