"""
Aragora Storage Module.

Provides persistent storage backends for users, organizations, and usage tracking.
Supports both SQLite (default) and PostgreSQL (for production scale).
"""

from .base_database import BaseDatabase
from .base_store import SQLiteStore
from .user_store import UserStore
from .organization_store import OrganizationStore
from .webhook_store import (
    WebhookStoreBackend,
    InMemoryWebhookStore,
    SQLiteWebhookStore,
    get_webhook_store,
    set_webhook_store,
    reset_webhook_store,
)
from .backends import (
    DatabaseBackend,
    SQLiteBackend,
    PostgreSQLBackend,
    get_database_backend,
    reset_database_backend,
    POSTGRESQL_AVAILABLE,
)
from .share_store import ShareLinkStore

__all__ = [
    # Legacy base classes
    "BaseDatabase",
    "SQLiteStore",
    "UserStore",
    "OrganizationStore",
    # Webhook idempotency
    "WebhookStoreBackend",
    "InMemoryWebhookStore",
    "SQLiteWebhookStore",
    "get_webhook_store",
    "set_webhook_store",
    "reset_webhook_store",
    # Database backends
    "DatabaseBackend",
    "SQLiteBackend",
    "PostgreSQLBackend",
    "get_database_backend",
    "reset_database_backend",
    "POSTGRESQL_AVAILABLE",
    # Share links
    "ShareLinkStore",
]
