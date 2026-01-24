"""
Email Connectors Package.

Provides background sync services for email providers:
- Gmail Sync: Real-time sync with Pub/Sub webhooks
- Outlook Sync: Real-time sync with Graph change notifications

These sync services complement the enterprise connectors by providing:
- Background sync workers
- Webhook handlers for real-time notifications
- Integration with EmailPrioritizer for scoring
- Tenant-isolated state management

For direct API access, use the enterprise connectors instead:
- GmailConnector: Full Gmail API with Pub/Sub, prioritization, and state persistence
- OutlookConnector: Full Microsoft Graph Mail API

Example::

    # Enterprise connector (recommended for new code)
    from aragora.connectors.enterprise.communication.gmail import GmailConnector
    connector = GmailConnector()
    await connector.authenticate(refresh_token=token)
    await connector.setup_watch(topic_name="gmail-notifications")

    # Sync service (for background workers with callbacks)
    from aragora.connectors.email import GmailSyncService, GmailSyncConfig
    service = GmailSyncService(tenant_id="t1", user_id="u1", config=config)
    await service.start(refresh_token=token)
"""

import warnings

# Import with deprecation warning suppression for package-level re-exports
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from aragora.connectors.email.gmail_sync import (
        GmailSyncService,
        GmailSyncConfig,
        SyncStatus,
        SyncedMessage,
        start_gmail_sync,
    )

# Re-export unified types from enterprise connector
from aragora.connectors.enterprise.communication.models import (
    GmailSyncState,
    GmailWebhookPayload,
)

# Also expose the unified connector for easy access
from aragora.connectors.enterprise.communication.gmail import (
    GmailConnector,
)
from aragora.connectors.email.outlook_sync import (
    OutlookSyncService,
    OutlookSyncConfig,
    OutlookSyncState,
    OutlookWebhookPayload,
    OutlookSyncStatus,
    OutlookSyncedMessage,
    start_outlook_sync,
)
from aragora.connectors.email.resilience import (
    ResilientEmailClient,
    OAuthTokenStore,
    OAuthToken,
    EmailCircuitBreaker,
    RetryExecutor,
    RetryConfig,
    CircuitBreakerConfig,
    RateLimitConfig,
    CircuitBreakerOpenError,
)

__all__ = [
    # Unified Gmail Connector (recommended)
    "GmailConnector",
    "GmailSyncState",
    "GmailWebhookPayload",
    # Gmail Sync Service (for background workers with callbacks)
    "GmailSyncService",
    "GmailSyncConfig",
    "SyncStatus",
    "SyncedMessage",
    "start_gmail_sync",
    # Outlook Sync
    "OutlookSyncService",
    "OutlookSyncConfig",
    "OutlookSyncState",
    "OutlookWebhookPayload",
    "OutlookSyncStatus",
    "OutlookSyncedMessage",
    "start_outlook_sync",
    # Resilience
    "ResilientEmailClient",
    "OAuthTokenStore",
    "OAuthToken",
    "EmailCircuitBreaker",
    "RetryExecutor",
    "RetryConfig",
    "CircuitBreakerConfig",
    "RateLimitConfig",
    "CircuitBreakerOpenError",
]
