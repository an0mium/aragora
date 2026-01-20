"""
Aragora Storage Module.

Provides persistent storage backends for users, organizations, and usage tracking.
Supports both SQLite (default) and PostgreSQL (for production scale).
"""

from .audit_store import AuditStore
from .backends import (
    POSTGRESQL_AVAILABLE,
    DatabaseBackend,
    PostgreSQLBackend,
    SQLiteBackend,
    get_database_backend,
    reset_database_backend,
)
from .base_database import BaseDatabase
from .base_store import SQLiteStore
from .organization_store import OrganizationStore
from .share_store import ShareLinkStore
from .user_store import UserStore
from .webhook_store import (
    InMemoryWebhookStore,
    SQLiteWebhookStore,
    WebhookStoreBackend,
    get_webhook_store,
    reset_webhook_store,
    set_webhook_store,
)
from .integration_store import (
    IntegrationConfig,
    IntegrationStoreBackend,
    InMemoryIntegrationStore,
    SQLiteIntegrationStore,
    RedisIntegrationStore,
    get_integration_store,
    set_integration_store,
    reset_integration_store,
    VALID_INTEGRATION_TYPES,
)
from .gmail_token_store import (
    GmailUserState,
    SyncJobState,
    GmailTokenStoreBackend,
    InMemoryGmailTokenStore,
    SQLiteGmailTokenStore,
    RedisGmailTokenStore,
    get_gmail_token_store,
    set_gmail_token_store,
    reset_gmail_token_store,
)

__all__ = [
    # Legacy base classes
    "BaseDatabase",
    "SQLiteStore",
    "UserStore",
    "OrganizationStore",
    "AuditStore",
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
    # Integration config storage
    "IntegrationConfig",
    "IntegrationStoreBackend",
    "InMemoryIntegrationStore",
    "SQLiteIntegrationStore",
    "RedisIntegrationStore",
    "get_integration_store",
    "set_integration_store",
    "reset_integration_store",
    "VALID_INTEGRATION_TYPES",
    # Gmail token storage
    "GmailUserState",
    "SyncJobState",
    "GmailTokenStoreBackend",
    "InMemoryGmailTokenStore",
    "SQLiteGmailTokenStore",
    "RedisGmailTokenStore",
    "get_gmail_token_store",
    "set_gmail_token_store",
    "reset_gmail_token_store",
]
