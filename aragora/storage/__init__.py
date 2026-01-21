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
from .finding_workflow_store import (
    WorkflowDataItem,
    FindingWorkflowStoreBackend,
    InMemoryFindingWorkflowStore,
    SQLiteFindingWorkflowStore,
    RedisFindingWorkflowStore,
    get_finding_workflow_store,
    set_finding_workflow_store,
    reset_finding_workflow_store,
)
from .federation_registry_store import (
    FederatedRegionConfig,
    FederationRegistryStoreBackend,
    InMemoryFederationRegistryStore,
    SQLiteFederationRegistryStore,
    RedisFederationRegistryStore,
    get_federation_registry_store,
    set_federation_registry_store,
    reset_federation_registry_store,
)
from .redis_utils import (
    get_redis_client,
    reset_redis_client,
    is_cluster_mode,
)
from .gauntlet_run_store import (
    GauntletRunItem,
    GauntletRunStoreBackend,
    InMemoryGauntletRunStore,
    SQLiteGauntletRunStore,
    RedisGauntletRunStore,
    get_gauntlet_run_store,
    set_gauntlet_run_store,
    reset_gauntlet_run_store,
)
from .approval_request_store import (
    ApprovalRequestItem,
    ApprovalRequestStoreBackend,
    InMemoryApprovalRequestStore,
    SQLiteApprovalRequestStore,
    RedisApprovalRequestStore,
    get_approval_request_store,
    set_approval_request_store,
    reset_approval_request_store,
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
    # Finding workflow storage
    "WorkflowDataItem",
    "FindingWorkflowStoreBackend",
    "InMemoryFindingWorkflowStore",
    "SQLiteFindingWorkflowStore",
    "RedisFindingWorkflowStore",
    "get_finding_workflow_store",
    "set_finding_workflow_store",
    "reset_finding_workflow_store",
    # Federation registry storage
    "FederatedRegionConfig",
    "FederationRegistryStoreBackend",
    "InMemoryFederationRegistryStore",
    "SQLiteFederationRegistryStore",
    "RedisFederationRegistryStore",
    "get_federation_registry_store",
    "set_federation_registry_store",
    "reset_federation_registry_store",
    # Redis client utilities
    "get_redis_client",
    "reset_redis_client",
    "is_cluster_mode",
    # Gauntlet run storage
    "GauntletRunItem",
    "GauntletRunStoreBackend",
    "InMemoryGauntletRunStore",
    "SQLiteGauntletRunStore",
    "RedisGauntletRunStore",
    "get_gauntlet_run_store",
    "set_gauntlet_run_store",
    "reset_gauntlet_run_store",
    # Approval request storage
    "ApprovalRequestItem",
    "ApprovalRequestStoreBackend",
    "InMemoryApprovalRequestStore",
    "SQLiteApprovalRequestStore",
    "RedisApprovalRequestStore",
    "get_approval_request_store",
    "set_approval_request_store",
    "reset_approval_request_store",
]
