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
"""

from aragora.connectors.email.gmail_sync import (
    GmailSyncService,
    GmailSyncConfig,
    GmailSyncState,
    GmailWebhookPayload,
    SyncStatus,
    SyncedMessage,
    start_gmail_sync,
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
    # Gmail Sync
    "GmailSyncService",
    "GmailSyncConfig",
    "GmailSyncState",
    "GmailWebhookPayload",
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
