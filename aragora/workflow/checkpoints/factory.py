"""
Factory functions for checkpoint store selection and configuration.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Optional

from aragora.workflow.checkpoints._compat import (
    MAX_CHECKPOINT_CACHE_SIZE,
    _PoolType,
)
from aragora.workflow.checkpoints.cache import CachingCheckpointStore
from aragora.workflow.checkpoints.file import FileCheckpointStore
from aragora.workflow.checkpoints.mound import KnowledgeMoundCheckpointStore
from aragora.workflow.checkpoints.postgres import PostgresCheckpointStore
from aragora.workflow.checkpoints.protocol import CheckpointStore
from aragora.workflow.checkpoints.redis import RedisCheckpointStore

if TYPE_CHECKING:
    from aragora.knowledge.mound import KnowledgeMound

logger = logging.getLogger(__name__)


# Module-level default KnowledgeMound for checkpoint storage
_default_mound: KnowledgeMound | None = None


def set_default_knowledge_mound(mound: KnowledgeMound) -> None:
    """
    Set the default KnowledgeMound for checkpoint storage.

    When set, get_checkpoint_store() will use KnowledgeMoundCheckpointStore
    instead of FileCheckpointStore, enabling durable checkpoint persistence
    with the Knowledge Mound backend.

    Usage:
        from aragora.knowledge.mound import KnowledgeMound
        from aragora.workflow.checkpoint_store import set_default_knowledge_mound

        mound = KnowledgeMound(workspace_id="production")
        await mound.initialize()
        set_default_knowledge_mound(mound)

    Args:
        mound: KnowledgeMound instance to use for checkpoints
    """
    global _default_mound
    _default_mound = mound
    logger.info("Set default KnowledgeMound for workflow checkpoints")


def get_default_knowledge_mound() -> KnowledgeMound | None:
    """Get the default KnowledgeMound for checkpoint storage."""
    return _default_mound


def get_checkpoint_store(
    mound: KnowledgeMound | None = None,
    fallback_dir: str = ".checkpoints",
    use_default_mound: bool = True,
    prefer_redis: bool = True,
    prefer_postgres: bool = False,
    enable_caching: bool = False,
    cache_size: int = MAX_CHECKPOINT_CACHE_SIZE,
) -> CheckpointStore:
    """
    Get the appropriate checkpoint store based on availability.

    Environment Variables:
    - ARAGORA_DB_BACKEND: If set to "postgres" or "postgresql", enables PostgreSQL preference
    - ARAGORA_CHECKPOINT_STORE_BACKEND: Store-specific override ("postgres", "redis", "file")
    - ARAGORA_CHECKPOINT_CACHE: If "true", enables caching layer

    Priority order:
    1. Explicitly provided KnowledgeMound
    2. Default KnowledgeMound (if set via set_default_knowledge_mound)
    3. RedisCheckpointStore (if prefer_redis=True and Redis is available)
    4. PostgresCheckpointStore (if prefer_postgres=True and Postgres is available)
    5. FileCheckpointStore (persistent file-based fallback)

    Args:
        mound: Optional KnowledgeMound instance (highest priority)
        fallback_dir: Fallback directory for file-based storage
        use_default_mound: Whether to use the default mound if no mound provided
        prefer_redis: Try Redis before file fallback (default True)
        prefer_postgres: Try Postgres before file fallback (default False)
        enable_caching: Wrap store with LRU cache (default False)
        cache_size: Maximum cache entries when caching enabled (default 100)

    Returns:
        CheckpointStore implementation (optionally wrapped with cache)
    """
    # Check environment variables for backend preference
    store_backend = os.environ.get("ARAGORA_CHECKPOINT_STORE_BACKEND", "").lower()
    global_backend = os.environ.get("ARAGORA_DB_BACKEND", "").lower()
    cache_env = os.environ.get("ARAGORA_CHECKPOINT_CACHE", "").lower()

    # Check if caching enabled via environment
    if cache_env in ("true", "1", "yes", "on"):
        enable_caching = True

    def _maybe_wrap_with_cache(store: CheckpointStore) -> CheckpointStore:
        """Wrap store with caching if enabled."""
        if enable_caching:
            logger.debug(f"Wrapping checkpoint store with cache (size={cache_size})")
            return CachingCheckpointStore(store, max_cache_size=cache_size)
        return store

    # Store-specific override takes precedence
    if store_backend == "postgres" or store_backend == "postgresql":
        prefer_postgres = True
    elif store_backend == "redis":
        prefer_redis = True
        prefer_postgres = False
    elif store_backend == "file":
        prefer_redis = False
        prefer_postgres = False
    # Fall back to global backend if no store-specific setting
    elif global_backend in ("postgres", "postgresql"):
        prefer_postgres = True
    # Use explicitly provided mound
    if mound is not None:
        logger.debug("Using provided KnowledgeMound for checkpoints")
        return _maybe_wrap_with_cache(KnowledgeMoundCheckpointStore(mound))

    # Try default mound
    if use_default_mound and _default_mound is not None:
        logger.debug("Using default KnowledgeMound for checkpoints")
        return _maybe_wrap_with_cache(KnowledgeMoundCheckpointStore(_default_mound))

    # Import from stub for test patching compatibility
    import aragora.workflow.checkpoint_store as _compat_stub

    # Try Redis if preferred
    if prefer_redis and _compat_stub.REDIS_AVAILABLE:
        try:
            store = RedisCheckpointStore()
            # Test Redis availability
            redis = store._get_redis()
            if redis is not None:
                logger.info("Using RedisCheckpointStore for checkpoints")
                return _maybe_wrap_with_cache(store)
        except Exception as e:
            logger.debug(f"Redis checkpoint store not available: {e}")

    # Try Postgres if preferred
    if prefer_postgres and _compat_stub.ASYNCPG_AVAILABLE:
        try:
            # Import here to avoid circular imports
            import asyncio
            from aragora.storage.postgres_store import get_postgres_pool

            # Get pool synchronously if possible
            try:
                asyncio.get_running_loop()
                # Can't await in sync context, skip Postgres
                logger.debug("Postgres pool requires async context, skipping")
            except RuntimeError:
                # No running event loop - create one and run synchronously
                try:
                    loop = asyncio.new_event_loop()
                    pool: _PoolType = loop.run_until_complete(get_postgres_pool())
                    pg_store = PostgresCheckpointStore(pool)
                    loop.run_until_complete(pg_store.initialize())
                    logger.info("Using PostgresCheckpointStore for checkpoints")
                    loop.close()
                    return _maybe_wrap_with_cache(pg_store)
                except RuntimeError:
                    # No event loop
                    logger.debug("No event loop for Postgres initialization")
        except Exception as e:
            logger.debug(f"Postgres checkpoint store not available: {e}")

    # Fall back to file-based storage
    # SECURITY: Check production guards before allowing file fallback
    try:
        from aragora.storage.production_guards import (
            require_distributed_store,
            StorageMode,
        )

        require_distributed_store(
            "checkpoint_store",
            StorageMode.FILE,
            "No Redis or PostgreSQL available for checkpoint storage",
        )
    except ImportError:
        pass  # Guards not available, allow fallback

    logger.debug(f"Using FileCheckpointStore in {fallback_dir}")
    return _maybe_wrap_with_cache(FileCheckpointStore(fallback_dir))


async def get_checkpoint_store_async(
    mound: KnowledgeMound | None = None,
    fallback_dir: str = ".checkpoints",
    use_default_mound: bool = True,
    prefer_redis: bool = True,
    prefer_postgres: bool = True,
    enable_caching: bool = False,
    cache_size: int = MAX_CHECKPOINT_CACHE_SIZE,
) -> CheckpointStore:
    """
    Get the appropriate checkpoint store (async version).

    This is the recommended function for async contexts as it properly
    initializes Postgres connection pools.

    Environment Variables:
    - ARAGORA_DB_BACKEND: If set to "postgres" or "postgresql", enables PostgreSQL preference
    - ARAGORA_CHECKPOINT_STORE_BACKEND: Store-specific override ("postgres", "redis", "file")
    - ARAGORA_CHECKPOINT_CACHE: If "true", enables caching layer

    Priority order:
    1. Explicitly provided KnowledgeMound
    2. Default KnowledgeMound (if set via set_default_knowledge_mound)
    3. RedisCheckpointStore (if prefer_redis=True and Redis is available)
    4. PostgresCheckpointStore (if prefer_postgres=True and Postgres is available)
    5. FileCheckpointStore (persistent file-based fallback)

    Args:
        mound: Optional KnowledgeMound instance (highest priority)
        fallback_dir: Fallback directory for file-based storage
        use_default_mound: Whether to use the default mound if no mound provided
        prefer_redis: Try Redis before file fallback (default True)
        prefer_postgres: Try Postgres before file fallback (default True)
        enable_caching: Wrap store with LRU cache (default False)
        cache_size: Maximum cache entries when caching enabled (default 100)

    Returns:
        CheckpointStore implementation (optionally wrapped with cache)
    """
    # Check environment variables for backend preference
    store_backend = os.environ.get("ARAGORA_CHECKPOINT_STORE_BACKEND", "").lower()
    global_backend = os.environ.get("ARAGORA_DB_BACKEND", "").lower()
    cache_env = os.environ.get("ARAGORA_CHECKPOINT_CACHE", "").lower()

    # Check if caching enabled via environment
    if cache_env in ("true", "1", "yes", "on"):
        enable_caching = True

    def _maybe_wrap_with_cache(store: CheckpointStore) -> CheckpointStore:
        """Wrap store with caching if enabled."""
        if enable_caching:
            logger.debug(f"Wrapping checkpoint store with cache (size={cache_size})")
            return CachingCheckpointStore(store, max_cache_size=cache_size)
        return store

    # Store-specific override takes precedence
    if store_backend == "postgres" or store_backend == "postgresql":
        prefer_postgres = True
    elif store_backend == "redis":
        prefer_redis = True
        prefer_postgres = False
    elif store_backend == "file":
        prefer_redis = False
        prefer_postgres = False
    # Fall back to global backend if no store-specific setting
    elif global_backend in ("postgres", "postgresql"):
        prefer_postgres = True

    # Use explicitly provided mound
    if mound is not None:
        logger.debug("Using provided KnowledgeMound for checkpoints")
        return _maybe_wrap_with_cache(KnowledgeMoundCheckpointStore(mound))

    # Try default mound
    if use_default_mound and _default_mound is not None:
        logger.debug("Using default KnowledgeMound for checkpoints")
        return _maybe_wrap_with_cache(KnowledgeMoundCheckpointStore(_default_mound))

    # Import from stub for test patching compatibility
    import aragora.workflow.checkpoint_store as _compat_stub

    # Try Redis if preferred
    if prefer_redis and _compat_stub.REDIS_AVAILABLE:
        try:
            store = RedisCheckpointStore()
            redis = store._get_redis()
            if redis is not None:
                logger.info("Using RedisCheckpointStore for checkpoints")
                return _maybe_wrap_with_cache(store)
        except Exception as e:
            logger.debug(f"Redis checkpoint store not available: {e}")

    # Try Postgres if preferred
    if prefer_postgres and _compat_stub.ASYNCPG_AVAILABLE:
        try:
            from aragora.storage.postgres_store import get_postgres_pool

            pool: _PoolType = await get_postgres_pool()
            pg_store = PostgresCheckpointStore(pool)
            await pg_store.initialize()
            logger.info("Using PostgresCheckpointStore for checkpoints")
            return _maybe_wrap_with_cache(pg_store)
        except Exception as e:
            logger.debug(f"Postgres checkpoint store not available: {e}")

    # Fall back to file-based storage
    # SECURITY: Check production guards before allowing file fallback
    try:
        from aragora.storage.production_guards import (
            require_distributed_store,
            StorageMode,
        )

        require_distributed_store(
            "checkpoint_store",
            StorageMode.FILE,
            "No Redis or PostgreSQL available for checkpoint storage (async)",
        )
    except ImportError:
        pass  # Guards not available, allow fallback

    logger.debug(f"Using FileCheckpointStore in {fallback_dir}")
    return _maybe_wrap_with_cache(FileCheckpointStore(fallback_dir))
