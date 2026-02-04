"""
Singleton management for UserStore and PostgresUserStore instances.

Provides global access to the user store with factory functions
that configure the appropriate backend based on environment variables.
"""

from __future__ import annotations

from .sqlite_store import UserStore
from .postgres_store import PostgresUserStore

# Singleton instance for global access
_user_store_instance: UserStore | None = None
_postgres_user_store_instance: PostgresUserStore | None = None


def get_user_store() -> UserStore | PostgresUserStore | None:
    """
    Get or create the user store.

    Uses environment variables to configure:
    - ARAGORA_DB_BACKEND: "sqlite", "postgres", or "supabase"
    - ARAGORA_USER_STORE_BACKEND: Per-store override ("sqlite", "postgres", "supabase")
    - ARAGORA_DATA_DIR: Directory for SQLite database
    - SUPABASE_URL + SUPABASE_DB_PASSWORD or SUPABASE_POSTGRES_DSN
    - ARAGORA_POSTGRES_DSN or DATABASE_URL

    Returns:
        Configured UserStore or PostgresUserStore instance
    """
    global _user_store_instance, _postgres_user_store_instance
    if _postgres_user_store_instance is not None:
        return _postgres_user_store_instance
    if _user_store_instance is not None:
        return _user_store_instance

    # Preserve legacy data directory if configured
    data_dir = None
    try:
        from aragora.persistence.db_config import get_default_data_dir

        data_dir = get_default_data_dir()
    except ImportError:
        data_dir = None

    from aragora.storage.connection_factory import create_persistent_store

    store = create_persistent_store(
        store_name="user",
        sqlite_class=UserStore,
        postgres_class=PostgresUserStore,
        db_filename="users.db",
        data_dir=str(data_dir) if data_dir else None,
    )

    if isinstance(store, PostgresUserStore):
        _postgres_user_store_instance = store
    else:
        _user_store_instance = store

    return store


def set_user_store(store: UserStore | PostgresUserStore) -> None:
    """Set the global UserStore singleton instance."""
    global _user_store_instance, _postgres_user_store_instance
    if isinstance(store, PostgresUserStore):
        _postgres_user_store_instance = store
    else:
        _user_store_instance = store


def reset_user_store() -> None:
    """Reset the global user store (for testing)."""
    global _user_store_instance, _postgres_user_store_instance
    _user_store_instance = None
    _postgres_user_store_instance = None
