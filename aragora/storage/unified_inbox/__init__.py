"""
Unified Inbox Store package.

Provides persistent storage for unified inbox accounts, messages, and triage results.
Backends:
- InMemoryUnifiedInboxStore: Testing
- SQLiteUnifiedInboxStore: Default single-instance persistence
- PostgresUnifiedInboxStore: Multi-instance production persistence
"""

from __future__ import annotations

import threading

from aragora.storage.unified_inbox.base import UnifiedInboxStoreBackend
from aragora.storage.unified_inbox.memory import InMemoryUnifiedInboxStore
from aragora.storage.unified_inbox.postgres import PostgresUnifiedInboxStore
from aragora.storage.unified_inbox.sqlite import DEFAULT_DB_NAME, SQLiteUnifiedInboxStore

_unified_inbox_store: UnifiedInboxStoreBackend | None = None
_store_lock = threading.Lock()


def get_unified_inbox_store() -> UnifiedInboxStoreBackend:
    """
    Get the unified inbox store.

    Backend selection (in preference order):
    1. Supabase PostgreSQL (if SUPABASE_URL + SUPABASE_DB_PASSWORD configured)
    2. Self-hosted PostgreSQL (if DATABASE_URL or ARAGORA_POSTGRES_DSN configured)
    3. SQLite (fallback, with production warning)

    Override via:
    - ARAGORA_INBOX_STORE_BACKEND: "memory", "sqlite", "postgres", or "supabase"
    - ARAGORA_DB_BACKEND: Global override
    """
    global _unified_inbox_store

    if _unified_inbox_store is not None:
        return _unified_inbox_store

    with _store_lock:
        if _unified_inbox_store is not None:
            return _unified_inbox_store

        from aragora.storage.connection_factory import create_persistent_store

        _unified_inbox_store = create_persistent_store(
            store_name="inbox",
            sqlite_class=SQLiteUnifiedInboxStore,
            postgres_class=PostgresUnifiedInboxStore,
            db_filename=DEFAULT_DB_NAME,
            memory_class=InMemoryUnifiedInboxStore,
        )

        return _unified_inbox_store


def set_unified_inbox_store(store: UnifiedInboxStoreBackend) -> None:
    """Set a custom unified inbox store (testing or customization)."""
    global _unified_inbox_store
    _unified_inbox_store = store


def reset_unified_inbox_store() -> None:
    """Reset the unified inbox store singleton (testing)."""
    global _unified_inbox_store
    _unified_inbox_store = None


__all__ = [
    "UnifiedInboxStoreBackend",
    "InMemoryUnifiedInboxStore",
    "SQLiteUnifiedInboxStore",
    "PostgresUnifiedInboxStore",
    "get_unified_inbox_store",
    "set_unified_inbox_store",
    "reset_unified_inbox_store",
]
