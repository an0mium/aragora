"""
Unified Storage Interface Protocols.

This module provides the canonical storage interface hierarchy for Aragora.
All persistence layers (server, CLI, memory) should converge on these protocols.

Architecture Overview
---------------------

Aragora uses a three-tier storage interface pattern:

1. **Protocol Layer** (this file)
   - `StorageInterface` / `AsyncStorageInterface` - Minimal key-value protocols
   - `StoreBackend` / `AsyncStoreBackend` - Enhanced domain store protocols

2. **Generic Implementation Layer** (generic_store.py)
   - `GenericStoreBackend` - Abstract base for domain stores
   - `GenericInMemoryStore` - Thread-safe testing backend
   - `GenericSQLiteStore` - SQLite with data_json blob pattern
   - `GenericPostgresStore` - Async PostgreSQL backend

3. **Domain Store Layer** (individual *_store.py files)
   - Extend GenericStoreBackend with domain-specific methods
   - Define table schema, indexes, and queries

Recommended Patterns
--------------------

**For new stores:** Extend `GenericStoreBackend` (most successful pattern)

    from aragora.storage.generic_store import (
        GenericStoreBackend,
        GenericSQLiteStore,
        GenericPostgresStore,
    )

    class MyStoreBackend(GenericStoreBackend):
        @abstractmethod
        async def list_by_status(self, status: str) -> list[dict]: ...

    class SQLiteMyStore(GenericSQLiteStore, MyStoreBackend):
        TABLE_NAME = "my_items"
        PRIMARY_KEY = "item_id"
        ...

**For protocol-based type hints:** Use `StoreBackend` or `AsyncStoreBackend`

    def process_items(store: StoreBackend) -> None:
        items = store.list_all()
        ...

**For memory backends:** Use protocols from `aragora.memory.protocols`

    from aragora.memory.protocols import MemoryBackend

See Also
--------
- `aragora.storage.generic_store` - Implementation base classes
- `aragora.storage.backends` - Database connection abstraction
- `aragora.memory.protocols` - Memory-specific protocols
"""

from __future__ import annotations

import logging

import asyncio
from typing import Any, Protocol, TypeVar, runtime_checkable

logger = logging.getLogger(__name__)

T = TypeVar("T")


# =============================================================================
# Minimal Key-Value Protocols (Legacy, use StoreBackend for new code)
# =============================================================================


@runtime_checkable
class StorageInterface(Protocol):
    """Minimal synchronous key-value storage interface.

    Use `StoreBackend` for new code - this protocol is kept for backwards
    compatibility with existing implementations.
    """

    def save(self, key: str, data: dict[str, Any]) -> None:
        """Save data with the given key."""
        ...

    def get(self, key: str) -> dict[str, Any] | None:
        """Get data by key, returns None if not found."""
        ...

    def delete(self, key: str) -> bool:
        """Delete by key, returns True if existed."""
        ...

    def query(self, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Query with optional filters."""
        ...


@runtime_checkable
class AsyncStorageInterface(Protocol):
    """Minimal async key-value storage interface.

    Use `AsyncStoreBackend` for new code - this protocol is kept for backwards
    compatibility with existing implementations.
    """

    async def save(self, key: str, data: dict[str, Any]) -> None:
        """Save data with the given key."""
        ...

    async def get(self, key: str) -> dict[str, Any] | None:
        """Get data by key, returns None if not found."""
        ...

    async def delete(self, key: str) -> bool:
        """Delete by key, returns True if existed."""
        ...

    async def query(self, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Query with optional filters."""
        ...


# =============================================================================
# Enhanced Store Backend Protocols (Recommended for new code)
# =============================================================================


@runtime_checkable
class StoreBackend(Protocol):
    """Synchronous store backend protocol.

    Provides the complete interface for domain stores including CRUD operations,
    existence checks, counting, and resource cleanup.

    Compatible with `GenericStoreBackend` implementations when wrapped with
    `sync_backend()`.
    """

    def get(self, item_id: str) -> dict[str, Any] | None:
        """Get item by primary key."""
        ...

    def save(self, data: dict[str, Any]) -> None:
        """Save (upsert) item data. Primary key must be in data."""
        ...

    def delete(self, item_id: str) -> bool:
        """Delete item by primary key. Returns True if existed."""
        ...

    def list_all(self) -> list[dict[str, Any]]:
        """List all items in the store."""
        ...

    def exists(self, item_id: str) -> bool:
        """Check if item exists."""
        ...

    def count(self) -> int:
        """Count total items."""
        ...

    def close(self) -> None:
        """Close/cleanup resources."""
        ...


@runtime_checkable
class AsyncStoreBackend(Protocol):
    """Async store backend protocol.

    This is the canonical protocol for domain stores. All `GenericStoreBackend`
    implementations conform to this protocol.

    Example:
        async def process_items(store: AsyncStoreBackend) -> None:
            items = await store.list_all()
            for item in items:
                await store.save({**item, "processed": True})
    """

    async def get(self, item_id: str) -> dict[str, Any] | None:
        """Get item by primary key."""
        ...

    async def save(self, data: dict[str, Any]) -> None:
        """Save (upsert) item data. Primary key must be in data."""
        ...

    async def delete(self, item_id: str) -> bool:
        """Delete item by primary key. Returns True if existed."""
        ...

    async def list_all(self) -> list[dict[str, Any]]:
        """List all items in the store."""
        ...

    async def exists(self, item_id: str) -> bool:
        """Check if item exists."""
        ...

    async def count(self) -> int:
        """Count total items."""
        ...

    async def close(self) -> None:
        """Close/cleanup resources."""
        ...


# =============================================================================
# Batch Operations Protocol (Optional extension)
# =============================================================================


@runtime_checkable
class BatchStoreBackend(Protocol):
    """Extension protocol for stores supporting batch operations."""

    async def save_batch(self, items: list[dict[str, Any]]) -> int:
        """Save multiple items. Returns count saved."""
        ...

    async def delete_batch(self, item_ids: list[str]) -> int:
        """Delete multiple items. Returns count deleted."""
        ...

    async def get_batch(self, item_ids: list[str]) -> list[dict[str, Any]]:
        """Get multiple items by IDs."""
        ...


# =============================================================================
# Health Check Protocol (Optional extension)
# =============================================================================


@runtime_checkable
class HealthCheckBackend(Protocol):
    """Extension protocol for stores with health monitoring."""

    async def health_check(self) -> dict[str, Any]:
        """Check backend health. Returns status dict with 'healthy' bool."""
        ...

    async def get_stats(self) -> dict[str, Any]:
        """Get backend statistics (item counts, latencies, etc.)."""
        ...


# =============================================================================
# Sync/Async Bridge Utilities
# =============================================================================


class SyncBackendWrapper:
    """Wraps an async backend for synchronous use.

    Example:
        async_store = SQLiteMyStore(db_path)
        sync_store = SyncBackendWrapper(async_store)
        item = sync_store.get("id-123")  # Blocking call
    """

    def __init__(self, async_backend: AsyncStoreBackend) -> None:
        self._backend = async_backend

    def _run(self, coro: Any) -> Any:
        """Run coroutine in event loop."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            # Already in async context - use nested event loop
            import nest_asyncio

            nest_asyncio.apply()
            return asyncio.get_event_loop().run_until_complete(coro)
        else:
            return asyncio.run(coro)

    def get(self, item_id: str) -> dict[str, Any] | None:
        return self._run(self._backend.get(item_id))

    def save(self, data: dict[str, Any]) -> None:
        return self._run(self._backend.save(data))

    def delete(self, item_id: str) -> bool:
        return self._run(self._backend.delete(item_id))

    def list_all(self) -> list[dict[str, Any]]:
        return self._run(self._backend.list_all())

    def exists(self, item_id: str) -> bool:
        return self._run(self._backend.exists(item_id))

    def count(self) -> int:
        return self._run(self._backend.count())

    def close(self) -> None:
        return self._run(self._backend.close())


def sync_backend(async_backend: AsyncStoreBackend) -> StoreBackend:
    """Create a synchronous wrapper for an async backend.

    Useful for CLI commands and legacy code that needs sync access.

    Example:
        async_store = get_my_store()  # Returns AsyncStoreBackend
        sync_store = sync_backend(async_store)
        items = sync_store.list_all()  # Blocking
    """
    return SyncBackendWrapper(async_backend)  # type: ignore


# =============================================================================
# Redis Fallback Mixin (Reduces duplication across Redis-backed stores)
# =============================================================================


class RedisStoreMixin:
    """
    Mixin providing Redis connection management with SQLite fallback.

    This mixin eliminates ~200 LOC of duplicated Redis fallback logic
    per store. Subclasses only need to define their domain-specific
    methods and table schema.

    Usage:
        class MyRedisStore(RedisStoreMixin, MyStoreBackend):
            REDIS_PREFIX = "aragora:my_store:"

            def __init__(self, fallback_db_path=None, redis_url=None):
                # Initialize fallback first (required by mixin)
                self._fallback = SQLiteMyStore(fallback_db_path)
                self._init_redis(redis_url)

            async def my_domain_method(self) -> list[dict]:
                if self._using_fallback:
                    return await self._fallback.my_domain_method()
                try:
                    # Redis-specific implementation
                    ...
                except Exception as e:
                    self._log_redis_fallback("my_domain_method", e)
                    return await self._fallback.my_domain_method()

    Attributes:
        REDIS_PREFIX: Key prefix for this store (must be set by subclass)
        _fallback: SQLite fallback store instance
        _redis_client: Redis client (or None if unavailable)
        _using_fallback: True if Redis unavailable
    """

    import json
    import logging
    import os
    import threading

    REDIS_PREFIX: str = "aragora:store:"  # Override in subclass

    _redis_client: Any = None
    _fallback: AsyncStoreBackend
    _using_fallback: bool = False
    _redis_lock: Any = None

    def _init_redis(self, redis_url: str | None = None) -> None:
        """
        Initialize Redis connection with fallback.

        Call this in __init__ after setting self._fallback.

        Args:
            redis_url: Redis URL (defaults to ARAGORA_REDIS_URL env var)
        """
        import logging
        import os
        import threading

        self._redis_lock = threading.RLock()
        self._redis_url = redis_url or os.getenv("ARAGORA_REDIS_URL", "")
        self._logger = logging.getLogger(self.__class__.__name__)

        if not self._redis_url:
            self._logger.info("No Redis URL configured, using SQLite fallback")
            self._using_fallback = True
            return

        try:
            import redis

            self._redis_client = redis.from_url(self._redis_url)
            self._redis_client.ping()
            self._logger.info(f"Connected to Redis for {self.__class__.__name__}")
            self._using_fallback = False
        except Exception as e:
            self._logger.warning(f"Redis connection failed, using SQLite fallback: {e}")
            self._using_fallback = True
            self._redis_client = None

    def _redis_key(self, item_id: str) -> str:
        """Build Redis key for an item."""
        return f"{self.REDIS_PREFIX}{item_id}"

    def _index_key(self, index_name: str, value: str) -> str:
        """Build Redis key for an index set."""
        return f"{self.REDIS_PREFIX}idx:{index_name}:{value}"

    def _log_redis_fallback(self, operation: str, error: Exception) -> None:
        """Log Redis failure and fallback."""
        self._logger.warning(f"Redis {operation} failed, using fallback: {error}")

    async def _redis_get(self, item_id: str) -> dict[str, Any] | None:
        """Get item from Redis with fallback."""
        import json

        if self._using_fallback:
            return await self._fallback.get(item_id)
        try:
            data = self._redis_client.get(self._redis_key(item_id))
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            self._log_redis_fallback("get", e)
            return await self._fallback.get(item_id)

    async def _redis_save(
        self,
        data: dict[str, Any],
        primary_key: str,
        indexes: dict[str, str] | None = None,
    ) -> None:
        """
        Save item to Redis with fallback.

        Always saves to SQLite fallback for durability.

        Args:
            data: Item data to save
            primary_key: Key name for the primary ID field
            indexes: Dict mapping index names to data field names
        """
        import json

        item_id = data.get(primary_key)
        if not item_id:
            raise ValueError(f"{primary_key} is required")

        # Always save to SQLite fallback for durability
        await self._fallback.save(data)

        if self._using_fallback:
            return

        try:
            data_json = json.dumps(data)
            pipe = self._redis_client.pipeline()
            pipe.set(self._redis_key(item_id), data_json)

            # Update indexes
            if indexes:
                for index_name, field_name in indexes.items():
                    field_value = data.get(field_name)
                    if field_value:
                        pipe.sadd(self._index_key(index_name, field_value), item_id)

            pipe.execute()
        except Exception as e:
            self._log_redis_fallback("save", e)

    async def _redis_delete(
        self,
        item_id: str,
        primary_key: str,
        indexes: dict[str, str] | None = None,
    ) -> bool:
        """
        Delete item from Redis with fallback.

        Args:
            item_id: ID of item to delete
            primary_key: Key name for the primary ID field
            indexes: Dict mapping index names to data field names
        """
        import json

        result = await self._fallback.delete(item_id)

        if self._using_fallback:
            return result

        try:
            data_bytes = self._redis_client.get(self._redis_key(item_id))
            if data_bytes:
                data = json.loads(data_bytes)
                pipe = self._redis_client.pipeline()

                # Remove from indexes
                if indexes:
                    for index_name, field_name in indexes.items():
                        field_value = data.get(field_name)
                        if field_value:
                            pipe.srem(self._index_key(index_name, field_value), item_id)

                pipe.delete(self._redis_key(item_id))
                pipe.execute()
                return True
            return result
        except Exception as e:
            self._log_redis_fallback("delete", e)
            return result

    async def _redis_list_all(self) -> list[dict[str, Any]]:
        """List all items from Redis with fallback."""
        import json

        if self._using_fallback:
            return await self._fallback.list_all()

        try:
            results = []
            cursor: str | int = 0
            while True:
                cursor, keys = self._redis_client.scan(
                    cursor=cursor,
                    match=f"{self.REDIS_PREFIX}*",
                    count=100,
                )
                if keys:
                    # Filter out index keys
                    data_keys = [k for k in keys if b":idx:" not in k and b"idx:" not in k]
                    if data_keys:
                        values = self._redis_client.mget(data_keys)
                        for v in values:
                            if v:
                                results.append(json.loads(v))
                if cursor in (0, "0"):
                    break
            return results
        except Exception as e:
            self._log_redis_fallback("list_all", e)
            return await self._fallback.list_all()

    async def _redis_list_by_index(
        self,
        index_name: str,
        value: str,
    ) -> list[dict[str, Any]]:
        """
        List items by index value from Redis with fallback.

        This assumes a corresponding fallback method exists.
        """
        import json

        if self._using_fallback:
            # Delegate to fallback - subclass should implement domain method
            raise NotImplementedError("Subclass must handle fallback for indexed queries")

        try:
            item_ids = self._redis_client.smembers(self._index_key(index_name, value))
            if not item_ids:
                return []
            keys = [self._redis_key(rid.decode()) for rid in item_ids]
            values = self._redis_client.mget(keys)
            return [json.loads(v) for v in values if v]
        except Exception as e:
            self._log_redis_fallback(f"list_by_{index_name}", e)
            raise

    async def _redis_close(self) -> None:
        """Close Redis connection and fallback."""
        await self._fallback.close()
        if self._redis_client:
            try:
                self._redis_client.close()
            except Exception:
                logger.debug("Failed to close Redis client", exc_info=True)


__all__ = [
    # Legacy protocols
    "StorageInterface",
    "AsyncStorageInterface",
    # Recommended protocols
    "StoreBackend",
    "AsyncStoreBackend",
    # Extension protocols
    "BatchStoreBackend",
    "HealthCheckBackend",
    # Utilities
    "SyncBackendWrapper",
    "sync_backend",
    # Redis mixin
    "RedisStoreMixin",
]
