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

import asyncio
from typing import Any, Protocol, TypeVar, runtime_checkable

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
]
