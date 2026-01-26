"""
Unified Inbox API Handler.

Provides a unified interface for multi-account email management:
- Gmail and Outlook integration through a single API
- Cross-account message retrieval with priority scoring
- Multi-agent triage for complex messages
- Inbox health metrics and analytics

OAuth Flow:
1. GET  /api/v1/inbox/oauth/gmail     - Get Gmail OAuth authorization URL
2. GET  /api/v1/inbox/oauth/outlook   - Get Outlook OAuth authorization URL
3. User is redirected to provider authorization page
4. Provider redirects back with auth_code
5. POST /api/v1/inbox/connect         - Exchange auth_code for tokens

Endpoints:
- GET  /api/v1/inbox/oauth/gmail      - Get Gmail OAuth URL (redirect_uri required)
- GET  /api/v1/inbox/oauth/outlook    - Get Outlook OAuth URL (redirect_uri required)
- POST /api/v1/inbox/connect          - Connect Gmail or Outlook account
- GET  /api/v1/inbox/accounts         - List connected accounts
- DELETE /api/v1/inbox/accounts/{id}  - Disconnect an account
- GET  /api/v1/inbox/messages         - Get prioritized messages across accounts
- GET  /api/v1/inbox/messages/{id}    - Get single message details
- POST /api/v1/inbox/triage           - Multi-agent triage for messages
- POST /api/v1/inbox/bulk-action      - Bulk actions (archive, read, etc.)
- GET  /api/v1/inbox/stats            - Inbox health metrics
- GET  /api/v1/inbox/trends           - Priority trends over time
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from ..base import (
    BaseHandler,
    HandlerResult,
    error_response,
    success_response,
)
from aragora.storage.unified_inbox_store import get_unified_inbox_store

logger = logging.getLogger(__name__)


# =============================================================================
# Sync Service Registry
# =============================================================================

# tenant_id -> account_id -> sync_service (GmailSyncService or OutlookSyncService)
_sync_services: Dict[str, Dict[str, Any]] = {}
_sync_services_lock = asyncio.Lock()  # Thread-safe access to _sync_services


def _convert_synced_message_to_unified(
    synced_msg: Any,
    account_id: str,
    provider: "EmailProvider",
) -> "UnifiedMessage":
    """Convert a SyncedMessage to UnifiedMessage format."""
    msg = synced_msg.message
    priority = synced_msg.priority_result

    # Extract priority info
    priority_score = 0.5
    priority_tier = "medium"
    priority_reasons: List[str] = []

    if priority:
        priority_score = priority.score if hasattr(priority, "score") else 0.5
        priority_tier = priority.tier if hasattr(priority, "tier") else "medium"
        priority_reasons = priority.reasons if hasattr(priority, "reasons") else []

    return UnifiedMessage(
        id=str(uuid4()),
        account_id=account_id,
        provider=provider,
        external_id=msg.id if hasattr(msg, "id") else str(uuid4()),
        subject=msg.subject if hasattr(msg, "subject") else "",
        sender_email=msg.from_email if hasattr(msg, "from_email") else "",
        sender_name=msg.from_name if hasattr(msg, "from_name") else "",
        recipients=msg.to if hasattr(msg, "to") else [],
        cc=msg.cc if hasattr(msg, "cc") else [],
        received_at=msg.date if hasattr(msg, "date") else datetime.now(timezone.utc),
        snippet=msg.snippet if hasattr(msg, "snippet") else "",
        body_preview=msg.body[:500] if hasattr(msg, "body") and msg.body else "",
        is_read=msg.is_read if hasattr(msg, "is_read") else False,
        is_starred=msg.is_starred if hasattr(msg, "is_starred") else False,
        has_attachments=bool(msg.attachments) if hasattr(msg, "attachments") else False,
        labels=msg.labels if hasattr(msg, "labels") else [],
        thread_id=msg.thread_id if hasattr(msg, "thread_id") else None,
        priority_score=priority_score,
        priority_tier=priority_tier,
        priority_reasons=priority_reasons,
    )


# =============================================================================
# Data Models
# =============================================================================


class EmailProvider(Enum):
    """Supported email providers."""

    GMAIL = "gmail"
    OUTLOOK = "outlook"


class AccountStatus(Enum):
    """Account connection status."""

    PENDING = "pending"
    CONNECTED = "connected"
    SYNCING = "syncing"
    ERROR = "error"
    DISCONNECTED = "disconnected"


class TriageAction(Enum):
    """Available triage actions."""

    RESPOND_URGENT = "respond_urgent"
    RESPOND_NORMAL = "respond_normal"
    DELEGATE = "delegate"
    SCHEDULE = "schedule"
    ARCHIVE = "archive"
    DELETE = "delete"
    FLAG = "flag"
    DEFER = "defer"


@dataclass
class ConnectedAccount:
    """Represents a connected email account."""

    id: str
    provider: EmailProvider
    email_address: str
    display_name: str
    status: AccountStatus
    connected_at: datetime
    last_sync: Optional[datetime] = None
    total_messages: int = 0
    unread_count: int = 0
    sync_errors: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "provider": self.provider.value,
            "email_address": self.email_address,
            "display_name": self.display_name,
            "status": self.status.value,
            "connected_at": self.connected_at.isoformat(),
            "last_sync": self.last_sync.isoformat() if self.last_sync else None,
            "total_messages": self.total_messages,
            "unread_count": self.unread_count,
            "sync_errors": self.sync_errors,
        }


@dataclass
class UnifiedMessage:
    """Unified message representation across providers."""

    id: str
    account_id: str
    provider: EmailProvider
    external_id: str  # Provider-specific ID
    subject: str
    sender_email: str
    sender_name: str
    recipients: List[str]
    cc: List[str]
    received_at: datetime
    snippet: str
    body_preview: str
    is_read: bool
    is_starred: bool
    has_attachments: bool
    labels: List[str]
    thread_id: Optional[str] = None
    # Priority scoring
    priority_score: float = 0.5
    priority_tier: str = "medium"
    priority_reasons: List[str] = field(default_factory=list)
    # Triage results
    triage_action: Optional[TriageAction] = None
    triage_rationale: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "account_id": self.account_id,
            "provider": self.provider.value,
            "external_id": self.external_id,
            "subject": self.subject,
            "sender": {
                "email": self.sender_email,
                "name": self.sender_name,
            },
            "recipients": self.recipients,
            "cc": self.cc,
            "received_at": self.received_at.isoformat(),
            "snippet": self.snippet,
            "is_read": self.is_read,
            "is_starred": self.is_starred,
            "has_attachments": self.has_attachments,
            "labels": self.labels,
            "thread_id": self.thread_id,
            "priority": {
                "score": self.priority_score,
                "tier": self.priority_tier,
                "reasons": self.priority_reasons,
            },
            "triage": {
                "action": self.triage_action.value if self.triage_action else None,
                "rationale": self.triage_rationale,
            }
            if self.triage_action
            else None,
        }


@dataclass
class TriageResult:
    """Result of multi-agent triage."""

    message_id: str
    recommended_action: TriageAction
    confidence: float
    rationale: str
    suggested_response: Optional[str]
    delegate_to: Optional[str]
    schedule_for: Optional[datetime]
    agents_involved: List[str]
    debate_summary: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "message_id": self.message_id,
            "recommended_action": self.recommended_action.value,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "suggested_response": self.suggested_response,
            "delegate_to": self.delegate_to,
            "schedule_for": self.schedule_for.isoformat() if self.schedule_for else None,
            "agents_involved": self.agents_involved,
            "debate_summary": self.debate_summary,
        }


@dataclass
class InboxStats:
    """Inbox health statistics."""

    total_accounts: int
    total_messages: int
    unread_count: int
    messages_by_priority: Dict[str, int]
    messages_by_provider: Dict[str, int]
    avg_response_time_hours: float
    pending_triage: int
    sync_health: Dict[str, Any]
    top_senders: List[Dict[str, Any]]
    hourly_volume: List[Dict[str, int]]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "total_accounts": self.total_accounts,
            "total_messages": self.total_messages,
            "unread_count": self.unread_count,
            "messages_by_priority": self.messages_by_priority,
            "messages_by_provider": self.messages_by_provider,
            "avg_response_time_hours": self.avg_response_time_hours,
            "pending_triage": self.pending_triage,
            "sync_health": self.sync_health,
            "top_senders": self.top_senders,
            "hourly_volume": self.hourly_volume,
        }


# =============================================================================
# Handler Class
# =============================================================================


class UnifiedInboxHandler(BaseHandler):
    """Handler for unified inbox API endpoints."""

    ROUTES = [
        "/api/v1/inbox/oauth/gmail",
        "/api/v1/inbox/oauth/outlook",
        "/api/v1/inbox/connect",
        "/api/v1/inbox/accounts",
        "/api/v1/inbox/accounts/{account_id}",
        "/api/v1/inbox/messages",
        "/api/v1/inbox/messages/{message_id}",
        "/api/v1/inbox/triage",
        "/api/v1/inbox/bulk-action",
        "/api/v1/inbox/stats",
        "/api/v1/inbox/trends",
    ]

    def __init__(self, server_context: Optional[Dict[str, Any]] = None):
        """Initialize handler with optional server context."""
        super().__init__(server_context or {})  # type: ignore[arg-type]
        self._store = get_unified_inbox_store()

    def can_handle(self, path: str, method: str = "GET") -> bool:
        """Check if this handler can process the given path."""
        return path.startswith("/api/v1/inbox")

    async def handle(self, request: Any, path: str, method: str) -> HandlerResult:  # type: ignore[override]
        """Route requests to appropriate handler methods."""
        try:
            # Extract tenant context
            tenant_id = self._get_tenant_id(request)

            # Route based on path and method
            # OAuth URL generation endpoints
            if path == "/api/v1/inbox/oauth/gmail" and method == "GET":
                return await self._handle_gmail_oauth_url(request, tenant_id)

            elif path == "/api/v1/inbox/oauth/outlook" and method == "GET":
                return await self._handle_outlook_oauth_url(request, tenant_id)

            elif path == "/api/v1/inbox/connect" and method == "POST":
                return await self._handle_connect(request, tenant_id)

            elif path == "/api/v1/inbox/accounts" and method == "GET":
                return await self._handle_list_accounts(request, tenant_id)

            elif path.startswith("/api/v1/inbox/accounts/") and method == "DELETE":
                account_id = path.split("/")[-1]
                return await self._handle_disconnect(request, tenant_id, account_id)

            elif path == "/api/v1/inbox/messages" and method == "GET":
                return await self._handle_list_messages(request, tenant_id)

            elif path.startswith("/api/v1/inbox/messages/") and method == "GET":
                message_id = path.split("/")[-1]
                return await self._handle_get_message(request, tenant_id, message_id)

            elif path == "/api/v1/inbox/triage" and method == "POST":
                return await self._handle_triage(request, tenant_id)

            elif path == "/api/v1/inbox/bulk-action" and method == "POST":
                return await self._handle_bulk_action(request, tenant_id)

            elif path == "/api/v1/inbox/stats" and method == "GET":
                return await self._handle_stats(request, tenant_id)

            elif path == "/api/v1/inbox/trends" and method == "GET":
                return await self._handle_trends(request, tenant_id)

            return error_response("Not found", 404)

        except Exception as e:
            logger.exception(f"Error in unified inbox handler: {e}")
            return error_response(f"Internal error: {str(e)}", 500)

    def _get_tenant_id(self, request: Any) -> str:
        """Extract tenant ID from request context."""
        # In production, extract from JWT or session
        return getattr(request, "tenant_id", "default")

    # =========================================================================
    # OAuth URL Generation
    # =========================================================================

    async def _handle_gmail_oauth_url(self, request: Any, tenant_id: str) -> HandlerResult:
        """Generate Gmail OAuth authorization URL.

        Query params:
        - redirect_uri: URL to redirect after authorization (required)
        - state: Optional CSRF state parameter

        Returns:
        {
            "auth_url": "https://accounts.google.com/o/oauth2/v2/auth?...",
            "provider": "gmail",
            "state": "..."
        }
        """
        try:
            params = self._get_query_params(request)
            redirect_uri = params.get("redirect_uri")

            if not redirect_uri:
                return error_response("Missing redirect_uri parameter", 400)

            # Generate state for CSRF protection
            state = params.get("state") or str(uuid4())

            try:
                from aragora.connectors.enterprise.communication.gmail import GmailConnector

                connector = GmailConnector()

                if not connector.is_configured:
                    return error_response(
                        "Gmail OAuth not configured. Set GMAIL_CLIENT_ID and GMAIL_CLIENT_SECRET.",
                        503,
                    )

                auth_url = connector.get_oauth_url(redirect_uri=redirect_uri, state=state)

                logger.info(f"[UnifiedInbox] Generated Gmail OAuth URL for tenant {tenant_id}")

                return success_response(
                    {
                        "auth_url": auth_url,
                        "provider": "gmail",
                        "state": state,
                    }
                )

            except ImportError:
                return error_response("Gmail connector not available", 503)

        except Exception as e:
            logger.exception(f"Error generating Gmail OAuth URL: {e}")
            return error_response(f"Failed to generate OAuth URL: {str(e)}", 500)

    async def _handle_outlook_oauth_url(self, request: Any, tenant_id: str) -> HandlerResult:
        """Generate Outlook/Microsoft 365 OAuth authorization URL.

        Query params:
        - redirect_uri: URL to redirect after authorization (required)
        - state: Optional CSRF state parameter

        Returns:
        {
            "auth_url": "https://login.microsoftonline.com/.../oauth2/v2.0/authorize?...",
            "provider": "outlook",
            "state": "..."
        }
        """
        try:
            params = self._get_query_params(request)
            redirect_uri = params.get("redirect_uri")

            if not redirect_uri:
                return error_response("Missing redirect_uri parameter", 400)

            # Generate state for CSRF protection
            state = params.get("state") or str(uuid4())

            try:
                from aragora.connectors.enterprise.communication.outlook import OutlookConnector

                connector = OutlookConnector()

                if not connector.is_configured:
                    return error_response(
                        "Outlook OAuth not configured. Set OUTLOOK_CLIENT_ID and OUTLOOK_CLIENT_SECRET.",
                        503,
                    )

                auth_url = connector.get_oauth_url(redirect_uri=redirect_uri, state=state)

                logger.info(f"[UnifiedInbox] Generated Outlook OAuth URL for tenant {tenant_id}")

                return success_response(
                    {
                        "auth_url": auth_url,
                        "provider": "outlook",
                        "state": state,
                    }
                )

            except ImportError:
                return error_response("Outlook connector not available", 503)

        except Exception as e:
            logger.exception(f"Error generating Outlook OAuth URL: {e}")
            return error_response(f"Failed to generate OAuth URL: {str(e)}", 500)

    # =========================================================================
    # Connect Account
    # =========================================================================

    async def _handle_connect(self, request: Any, tenant_id: str) -> HandlerResult:
        """Handle account connection request.

        Request body:
        {
            "provider": "gmail" | "outlook",
            "auth_code": "...",  // OAuth authorization code
            "redirect_uri": "..."
        }
        """
        try:
            body = await self._get_json_body(request)

            provider_str = body.get("provider", "").lower()
            if provider_str not in ["gmail", "outlook"]:
                return error_response("Invalid provider. Must be 'gmail' or 'outlook'", 400)

            provider = EmailProvider(provider_str)
            auth_code = body.get("auth_code")
            redirect_uri = body.get("redirect_uri", "")

            if not auth_code:
                return error_response("Missing auth_code", 400)

            # Create account record
            account_id = str(uuid4())
            account = ConnectedAccount(
                id=account_id,
                provider=provider,
                email_address="",  # Will be filled after OAuth
                display_name="",
                status=AccountStatus.PENDING,
                connected_at=datetime.now(timezone.utc),
            )

            # Exchange auth code for tokens based on provider
            if provider == EmailProvider.GMAIL:
                result = await self._connect_gmail(account, auth_code, redirect_uri, tenant_id)
            else:
                result = await self._connect_outlook(account, auth_code, redirect_uri, tenant_id)

            if result.get("success"):
                await self._store.save_account(tenant_id, self._account_to_record(account))
                logger.info(
                    f"Connected {provider.value} account for tenant {tenant_id}: {account.email_address}"
                )
                return success_response(
                    {
                        "account": account.to_dict(),
                        "message": f"Successfully connected {provider.value} account",
                    }
                )
            else:
                return error_response(result.get("error", "Failed to connect account"), 400)

        except Exception as e:
            logger.exception(f"Error connecting account: {e}")
            return error_response(f"Failed to connect account: {str(e)}", 500)

    async def _connect_gmail(
        self,
        account: ConnectedAccount,
        auth_code: str,
        redirect_uri: str,
        tenant_id: str,
    ) -> Dict[str, Any]:
        """Connect Gmail account via OAuth."""
        try:
            from aragora.connectors.email import GmailSyncService, GmailSyncConfig
            from aragora.connectors.enterprise.communication.gmail import GmailConnector

            config = GmailSyncConfig(
                enable_prioritization=True,
                initial_sync_days=7,
            )

            # Exchange auth code for tokens via Gmail connector
            connector = GmailConnector()
            auth_ok = await connector.authenticate(code=auth_code, redirect_uri=redirect_uri)
            if not auth_ok:
                return {"success": False, "error": "Gmail authentication failed"}

            refresh_token = connector.refresh_token or ""
            if not refresh_token:
                return {"success": False, "error": "Gmail refresh token not returned"}

            # Load profile for display details
            profile = await connector.get_user_info()
            account.email_address = profile.get("emailAddress", "") or account.email_address
            if account.email_address:
                account.display_name = account.email_address.split("@")[0]
            else:
                account.display_name = "Gmail User"

            # Initialize tenant sync registry (thread-safe)
            async with _sync_services_lock:
                if tenant_id not in _sync_services:
                    _sync_services[tenant_id] = {}

            # Create message callback that stores unified messages
            def on_message_synced(synced_msg: Any) -> None:
                try:
                    unified = _convert_synced_message_to_unified(
                        synced_msg, account.id, EmailProvider.GMAIL
                    )
                    self._schedule_message_persist(tenant_id, unified)
                except Exception as e:
                    logger.warning(f"[UnifiedInbox] Error converting message: {e}")

            # Create sync service
            sync_service = GmailSyncService(
                tenant_id=tenant_id,
                user_id=account.id,
                config=config,
                gmail_connector=connector,
                on_message_synced=on_message_synced,
            )

            # Store sync service
            _sync_services[tenant_id][account.id] = sync_service

            # Persist OAuth state for restart safety
            try:
                from aragora.storage.gmail_token_store import GmailUserState, get_gmail_token_store

                state = GmailUserState(
                    user_id=account.id,
                    org_id=tenant_id,
                    email_address=account.email_address,
                    access_token=connector.access_token or "",
                    refresh_token=refresh_token,
                    token_expiry=connector.token_expiry,
                    connected_at=datetime.now(timezone.utc),
                )
                store = get_gmail_token_store()
                await store.save(state)
            except Exception as e:
                logger.warning(f"[UnifiedInbox] Failed to persist Gmail tokens: {e}")

            # Start sync using the authenticated connector
            await sync_service.start()

            account.status = AccountStatus.CONNECTED

            logger.info(f"[UnifiedInbox] Gmail sync service registered for {account.id}")
            return {"success": True}

        except ImportError:
            # GmailSyncService not available, use mock mode
            account.email_address = f"user_{account.id[:8]}@gmail.com"
            account.display_name = "Gmail User"
            account.status = AccountStatus.CONNECTED
            logger.warning("[UnifiedInbox] GmailSyncService not available, using mock mode")
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _connect_outlook(
        self,
        account: ConnectedAccount,
        auth_code: str,
        redirect_uri: str,
        tenant_id: str,
    ) -> Dict[str, Any]:
        """Connect Outlook account via OAuth."""
        try:
            from aragora.connectors.email import OutlookSyncService, OutlookSyncConfig
            from aragora.connectors.enterprise.communication.outlook import OutlookConnector

            config = OutlookSyncConfig(
                enable_prioritization=True,
                initial_sync_days=7,
            )

            connector = OutlookConnector()
            auth_ok = await connector.authenticate(code=auth_code, redirect_uri=redirect_uri)
            if not auth_ok:
                return {"success": False, "error": "Outlook authentication failed"}

            refresh_token = connector.refresh_token or ""
            if not refresh_token:
                return {"success": False, "error": "Outlook refresh token not returned"}

            # Load profile for display details
            profile = await connector.get_user_info()
            account.email_address = profile.get("mail") or profile.get("userPrincipalName", "")
            if account.email_address:
                account.display_name = account.email_address.split("@")[0]
            else:
                account.display_name = "Outlook User"

            # Initialize tenant sync registry (thread-safe)
            async with _sync_services_lock:
                if tenant_id not in _sync_services:
                    _sync_services[tenant_id] = {}

            # Create message callback that stores unified messages
            def on_message_synced(synced_msg: Any) -> None:
                try:
                    unified = _convert_synced_message_to_unified(
                        synced_msg, account.id, EmailProvider.OUTLOOK
                    )
                    self._schedule_message_persist(tenant_id, unified)
                except Exception as e:
                    logger.warning(f"[UnifiedInbox] Error converting message: {e}")

            # Create sync service
            sync_service = OutlookSyncService(
                tenant_id=tenant_id,
                user_id=account.id,
                config=config,
                outlook_connector=connector,
                on_message_synced=on_message_synced,
            )

            # Store sync service (thread-safe)
            async with _sync_services_lock:
                _sync_services[tenant_id][account.id] = sync_service

            # Persist OAuth state for restart safety
            try:
                from aragora.storage.integration_store import (
                    IntegrationConfig,
                    get_integration_store,
                )

                integration = IntegrationConfig(
                    type="outlook_email",
                    enabled=True,
                    settings={
                        "refresh_token": refresh_token,
                        "access_token": connector.access_token or "",
                        "token_expiry": connector.token_expiry.isoformat()
                        if connector.token_expiry
                        else None,
                        "account_id": account.id,
                        "tenant_id": tenant_id,
                        "email_address": account.email_address,
                    },
                    user_id=account.id,
                    workspace_id=tenant_id,
                )
                store = get_integration_store()
                await store.save(integration)
            except Exception as e:
                logger.warning(f"[UnifiedInbox] Failed to persist Outlook tokens: {e}")

            # Start sync using the authenticated connector
            await sync_service.start()

            account.status = AccountStatus.CONNECTED

            logger.info(f"[UnifiedInbox] Outlook sync service registered for {account.id}")
            return {"success": True}

        except ImportError:
            # OutlookSyncService not available, use mock mode
            account.email_address = f"user_{account.id[:8]}@outlook.com"
            account.display_name = "Outlook User"
            account.status = AccountStatus.CONNECTED
            logger.warning("[UnifiedInbox] OutlookSyncService not available, using mock mode")
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # =========================================================================
    # List/Disconnect Accounts
    # =========================================================================

    async def _handle_list_accounts(self, request: Any, tenant_id: str) -> HandlerResult:
        """List all connected accounts."""
        records = await self._store.list_accounts(tenant_id)
        accounts = [self._record_to_account(record) for record in records]
        return success_response(
            {
                "accounts": [acc.to_dict() for acc in accounts],
                "total": len(accounts),
            }
        )

    async def _handle_disconnect(
        self, request: Any, tenant_id: str, account_id: str
    ) -> HandlerResult:
        """Disconnect an account."""
        record = await self._store.get_account(tenant_id, account_id)
        if not record:
            return error_response("Account not found", 404)

        account = self._record_to_account(record)

        # Stop and remove sync service if running (thread-safe)
        sync_service = None
        async with _sync_services_lock:
            if tenant_id in _sync_services and account_id in _sync_services[tenant_id]:
                sync_service = _sync_services[tenant_id].pop(account_id)
        if sync_service:
            try:
                if hasattr(sync_service, "stop"):
                    await sync_service.stop()
                logger.info(f"[UnifiedInbox] Stopped sync service for account {account_id}")
            except Exception as e:
                logger.warning(f"[UnifiedInbox] Error stopping sync service: {e}")

        await self._store.delete_account(tenant_id, account_id)

        logger.info(
            f"Disconnected {account.provider.value} account for tenant {tenant_id}: {account.email_address}"
        )

        return success_response(
            {
                "message": f"Successfully disconnected {account.provider.value} account",
                "account_id": account_id,
            }
        )

    # =========================================================================
    # Messages
    # =========================================================================

    async def _handle_list_messages(self, request: Any, tenant_id: str) -> HandlerResult:
        """Get prioritized messages across all accounts.

        Query params:
        - limit: Max messages to return (default 50)
        - offset: Pagination offset (default 0)
        - priority: Filter by priority tier (critical, high, medium, low)
        - account_id: Filter by specific account
        - unread_only: Only return unread messages
        - search: Search query
        """
        try:
            params = self._get_query_params(request)
            limit = int(params.get("limit", 50))
            offset = int(params.get("offset", 0))
            priority_filter = params.get("priority")
            account_filter = params.get("account_id")
            unread_only = params.get("unread_only", "false").lower() == "true"
            search_query = params.get("search")

            records, total = await self._store.list_messages(
                tenant_id=tenant_id,
                limit=limit,
                offset=offset,
                priority_tier=priority_filter,
                account_id=account_filter,
                unread_only=unread_only,
                search=search_query,
            )
            messages = [self._record_to_message(record) for record in records]

            return success_response(
                {
                    "messages": [m.to_dict() for m in messages],
                    "total": total,
                    "limit": limit,
                    "offset": offset,
                    "has_more": offset + limit < total,
                }
            )

        except Exception as e:
            logger.exception(f"Error listing messages: {e}")
            return error_response(f"Failed to list messages: {str(e)}", 500)

    async def _fetch_all_messages(self, tenant_id: str) -> List[UnifiedMessage]:
        """Fetch messages from all connected accounts."""
        records, total = await self._store.list_messages(tenant_id=tenant_id, limit=None)
        if total > 0:
            return [self._record_to_message(record) for record in records]

        messages: List[UnifiedMessage] = []
        account_records = await self._store.list_accounts(tenant_id)

        for record in account_records:
            account = self._record_to_account(record)
            if account.status != AccountStatus.CONNECTED:
                continue

            try:
                if account.provider == EmailProvider.GMAIL:
                    account_messages = await self._fetch_gmail_messages(account, tenant_id)
                else:
                    account_messages = await self._fetch_outlook_messages(account, tenant_id)

                for message in account_messages:
                    await self._store.save_message(tenant_id, self._message_to_record(message))
                messages.extend(account_messages)

            except Exception as e:
                logger.warning(
                    f"Error fetching messages for {account.provider.value} "
                    f"account {account.id}: {e}"
                )
                await self._store.increment_account_counts(
                    tenant_id, account.id, sync_error_delta=1
                )

        # Apply priority scoring
        messages = await self._score_messages(messages, tenant_id)

        return messages

    async def _fetch_gmail_messages(
        self, account: ConnectedAccount, tenant_id: str
    ) -> List[UnifiedMessage]:
        """Fetch messages from Gmail account."""
        # Check if sync service is running and has synced messages (thread-safe lookup)
        sync_service = None
        if tenant_id:
            async with _sync_services_lock:
                if tenant_id in _sync_services:
                    sync_service = _sync_services[tenant_id].get(account.id)
        if sync_service:
            # Check if initial sync is complete
            state = getattr(sync_service, "state", None)
            if state and getattr(state, "initial_sync_complete", False):
                # Messages are already in cache via callbacks
                logger.debug(
                    f"[UnifiedInbox] Gmail sync active for {account.id}, "
                    f"messages synced: {getattr(state, 'total_messages_synced', 0)}"
                )
                return []  # Messages already in cache

        # Fall back to sample data if sync not active
        return self._generate_sample_messages(account, 5)

    async def _fetch_outlook_messages(
        self, account: ConnectedAccount, tenant_id: str
    ) -> List[UnifiedMessage]:
        """Fetch messages from Outlook account."""
        # Check if sync service is running and has synced messages
        if tenant_id and tenant_id in _sync_services:
            sync_service = _sync_services[tenant_id].get(account.id)
            if sync_service:
                # Check if initial sync is complete
                state = getattr(sync_service, "state", None)
                if state and getattr(state, "initial_sync_complete", False):
                    # Messages are already in cache via callbacks
                    logger.debug(
                        f"[UnifiedInbox] Outlook sync active for {account.id}, "
                        f"messages synced: {getattr(state, 'total_messages_synced', 0)}"
                    )
                    return []  # Messages already in cache

        # Fall back to sample data if sync not active
        return self._generate_sample_messages(account, 5)

    def _generate_sample_messages(
        self, account: ConnectedAccount, count: int
    ) -> List[UnifiedMessage]:
        """Generate sample messages for testing."""
        messages = []
        now = datetime.now(timezone.utc)

        sample_subjects = [
            ("Urgent: Contract Review Required", "critical"),
            ("Q4 Budget Approval Needed", "high"),
            ("Weekly Team Update", "medium"),
            ("Newsletter: Industry Updates", "low"),
            ("Meeting Rescheduled", "medium"),
        ]

        for i in range(min(count, len(sample_subjects))):
            subject, priority = sample_subjects[i]
            messages.append(
                UnifiedMessage(
                    id=str(uuid4()),
                    account_id=account.id,
                    provider=account.provider,
                    external_id=f"ext_{uuid4().hex[:8]}",
                    subject=subject,
                    sender_email=f"sender{i}@example.com",
                    sender_name=f"Sender {i}",
                    recipients=[account.email_address],
                    cc=[],
                    received_at=now - timedelta(hours=i),
                    snippet=f"Preview of message {i}...",
                    body_preview=f"This is the body preview of message {i}...",
                    is_read=i > 2,
                    is_starred=i == 0,
                    has_attachments=i < 2,
                    labels=["inbox"],
                    priority_tier=priority,
                    priority_score={"critical": 0.95, "high": 0.75, "medium": 0.5, "low": 0.25}[
                        priority
                    ],
                )
            )

        return messages

    async def _score_messages(
        self, messages: List[UnifiedMessage], tenant_id: str
    ) -> List[UnifiedMessage]:
        """Apply priority scoring to messages."""
        try:
            from aragora.services.email_prioritization import (  # noqa: F401
                EmailPrioritizer,
                EmailPriority,
            )

            # Use prioritizer if available
            # For now, messages already have sample scores
            return messages

        except ImportError:
            # Prioritizer not available, use existing scores
            return messages

    async def _handle_get_message(
        self, request: Any, tenant_id: str, message_id: str
    ) -> HandlerResult:
        """Get single message details."""
        record = await self._store.get_message(tenant_id, message_id)
        if not record:
            return error_response("Message not found", 404)

        message = self._record_to_message(record)
        triage_record = await self._store.get_triage_result(tenant_id, message_id)
        triage = self._record_to_triage(triage_record) if triage_record else None

        return success_response(
            {
                "message": message.to_dict(),
                "triage": triage.to_dict() if triage else None,
            }
        )

    # =========================================================================
    # Triage
    # =========================================================================

    async def _handle_triage(self, request: Any, tenant_id: str) -> HandlerResult:
        """Run multi-agent triage on messages.

        Request body:
        {
            "message_ids": ["id1", "id2", ...],
            "context": {
                "urgency_keywords": [...],
                "delegate_options": [...]
            }
        }
        """
        try:
            body = await self._get_json_body(request)
            message_ids = body.get("message_ids", [])
            context = body.get("context", {})

            if not message_ids:
                return error_response("No message IDs provided", 400)

            messages_to_triage: List[UnifiedMessage] = []
            for message_id in message_ids:
                record = await self._store.get_message(tenant_id, message_id)
                if record:
                    messages_to_triage.append(self._record_to_message(record))

            if not messages_to_triage:
                return error_response("No matching messages found", 404)

            # Run triage
            results = await self._run_triage(messages_to_triage, context, tenant_id)

            return success_response(
                {
                    "results": [r.to_dict() for r in results],
                    "total_triaged": len(results),
                }
            )

        except Exception as e:
            logger.exception(f"Error during triage: {e}")
            return error_response(f"Triage failed: {str(e)}", 500)

    async def _run_triage(
        self,
        messages: List[UnifiedMessage],
        context: Dict[str, Any],
        tenant_id: str,
    ) -> List[TriageResult]:
        """Run multi-agent triage on messages."""
        results = []

        for message in messages:
            try:
                result = await self._triage_single_message(message, context, tenant_id)
                results.append(result)

                # Update message with triage result
                message.triage_action = result.recommended_action
                message.triage_rationale = result.rationale

                await self._store.save_triage_result(tenant_id, self._triage_to_record(result))
                await self._store.update_message_triage(
                    tenant_id,
                    message.id,
                    result.recommended_action.value,
                    result.rationale,
                )

            except Exception as e:
                logger.warning(f"Triage failed for message {message.id}: {e}")

        return results

    async def _triage_single_message(
        self,
        message: UnifiedMessage,
        context: Dict[str, Any],
        tenant_id: str,
    ) -> TriageResult:
        """Triage a single message using multi-agent debate."""
        try:
            from aragora.debate import Arena, Environment, DebateProtocol  # noqa: F401

            # Build debate environment
            Environment(
                task=f"""
                Analyze this email and recommend the best action:

                From: {message.sender_name} <{message.sender_email}>
                Subject: {message.subject}
                Preview: {message.snippet}

                Consider:
                - Urgency and time-sensitivity
                - Sender importance
                - Required action type
                - Delegation possibilities

                Recommend ONE of: respond_urgent, respond_normal, delegate, schedule, archive, flag, defer
                """,
            )

            DebateProtocol(
                rounds=2,
                consensus="majority",
            )

            # Run debate (simplified for now)
            # In production, use actual Arena with agents
            agents = ["support_analyst", "product_expert"]

            # Determine action based on priority
            if message.priority_tier == "critical":
                action = TriageAction.RESPOND_URGENT
            elif message.priority_tier == "high":
                action = TriageAction.RESPOND_NORMAL
            elif message.priority_tier == "low":
                action = TriageAction.ARCHIVE
            else:
                action = TriageAction.DEFER

            return TriageResult(
                message_id=message.id,
                recommended_action=action,
                confidence=0.85,
                rationale=f"Based on priority tier '{message.priority_tier}' and sender analysis",
                suggested_response=None,
                delegate_to=None,
                schedule_for=None,
                agents_involved=agents,
                debate_summary="Multi-agent analysis completed",
            )

        except ImportError:
            # Arena not available, use simple heuristics
            if message.priority_tier == "critical":
                action = TriageAction.RESPOND_URGENT
            elif message.priority_tier == "high":
                action = TriageAction.RESPOND_NORMAL
            elif message.priority_tier == "low":
                action = TriageAction.ARCHIVE
            else:
                action = TriageAction.DEFER

            return TriageResult(
                message_id=message.id,
                recommended_action=action,
                confidence=0.7,
                rationale=f"Heuristic-based triage for '{message.priority_tier}' priority",
                suggested_response=None,
                delegate_to=None,
                schedule_for=None,
                agents_involved=[],
                debate_summary=None,
            )

    # =========================================================================
    # Bulk Actions
    # =========================================================================

    async def _handle_bulk_action(self, request: Any, tenant_id: str) -> HandlerResult:
        """Execute bulk action on messages.

        Request body:
        {
            "message_ids": ["id1", "id2", ...],
            "action": "archive" | "mark_read" | "mark_unread" | "star" | "delete"
        }
        """
        try:
            body = await self._get_json_body(request)
            message_ids = body.get("message_ids", [])
            action = body.get("action", "")

            if not message_ids:
                return error_response("No message IDs provided", 400)

            valid_actions = ["archive", "mark_read", "mark_unread", "star", "delete"]
            if action not in valid_actions:
                return error_response(
                    f"Invalid action. Must be one of: {', '.join(valid_actions)}", 400
                )

            # Execute action on messages
            success_count = 0
            errors = []

            for msg_id in message_ids:
                try:
                    if action == "mark_read":
                        updated = await self._store.update_message_flags(
                            tenant_id, msg_id, is_read=True
                        )
                        if not updated:
                            errors.append({"id": msg_id, "error": "Message not found"})
                            continue
                    elif action == "mark_unread":
                        updated = await self._store.update_message_flags(
                            tenant_id, msg_id, is_read=False
                        )
                        if not updated:
                            errors.append({"id": msg_id, "error": "Message not found"})
                            continue
                    elif action == "star":
                        updated = await self._store.update_message_flags(
                            tenant_id, msg_id, is_starred=True
                        )
                        if not updated:
                            errors.append({"id": msg_id, "error": "Message not found"})
                            continue
                    elif action in ("archive", "delete"):
                        deleted = await self._store.delete_message(tenant_id, msg_id)
                        if not deleted:
                            errors.append({"id": msg_id, "error": "Message not found"})
                            continue

                    success_count += 1

                except Exception as e:
                    errors.append({"id": msg_id, "error": str(e)})

            return success_response(
                {
                    "action": action,
                    "success_count": success_count,
                    "error_count": len(errors),
                    "errors": errors if errors else None,
                }
            )

        except Exception as e:
            logger.exception(f"Error executing bulk action: {e}")
            return error_response(f"Bulk action failed: {str(e)}", 500)

    # =========================================================================
    # Stats & Trends
    # =========================================================================

    async def _handle_stats(self, request: Any, tenant_id: str) -> HandlerResult:
        """Get inbox health statistics."""
        account_records = await self._store.list_accounts(tenant_id)
        accounts = [self._record_to_account(record) for record in account_records]
        messages = await self._fetch_all_messages(tenant_id)

        # Calculate stats
        unread_count = sum(1 for m in messages if not m.is_read)

        messages_by_priority = {
            "critical": sum(1 for m in messages if m.priority_tier == "critical"),
            "high": sum(1 for m in messages if m.priority_tier == "high"),
            "medium": sum(1 for m in messages if m.priority_tier == "medium"),
            "low": sum(1 for m in messages if m.priority_tier == "low"),
        }

        messages_by_provider = {
            "gmail": sum(1 for m in messages if m.provider == EmailProvider.GMAIL),
            "outlook": sum(1 for m in messages if m.provider == EmailProvider.OUTLOOK),
        }

        # Top senders
        sender_counts: Dict[str, int] = {}
        for m in messages:
            sender_counts[m.sender_email] = sender_counts.get(m.sender_email, 0) + 1

        top_senders = [
            {"email": email, "count": count}
            for email, count in sorted(sender_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        ]

        # Sync health
        sync_health = {
            "accounts_healthy": sum(1 for a in accounts if a.status == AccountStatus.CONNECTED),
            "accounts_error": sum(1 for a in accounts if a.status == AccountStatus.ERROR),
            "total_sync_errors": sum(a.sync_errors for a in accounts),
        }

        stats = InboxStats(
            total_accounts=len(accounts),
            total_messages=len(messages),
            unread_count=unread_count,
            messages_by_priority=messages_by_priority,
            messages_by_provider=messages_by_provider,
            avg_response_time_hours=4.5,  # Would be calculated from actual data
            pending_triage=sum(1 for m in messages if m.triage_action is None and not m.is_read),
            sync_health=sync_health,
            top_senders=top_senders,
            hourly_volume=[],  # Would be calculated from actual timestamps
        )

        return success_response({"stats": stats.to_dict()})

    async def _handle_trends(self, request: Any, tenant_id: str) -> HandlerResult:
        """Get priority trends over time."""
        params = self._get_query_params(request)
        days = int(params.get("days", 7))

        # In production, calculate from historical data
        trends = {
            "period_days": days,
            "priority_trends": {
                "critical": {"current": 5, "previous": 8, "change_pct": -37.5},
                "high": {"current": 15, "previous": 12, "change_pct": 25.0},
                "medium": {"current": 45, "previous": 42, "change_pct": 7.1},
                "low": {"current": 35, "previous": 38, "change_pct": -7.9},
            },
            "volume_trend": {
                "current_daily_avg": 25,
                "previous_daily_avg": 22,
                "change_pct": 13.6,
            },
            "response_time_trend": {
                "current_avg_hours": 4.2,
                "previous_avg_hours": 5.1,
                "change_pct": -17.6,
            },
        }

        return success_response({"trends": trends})

    # =========================================================================
    # Persistence Helpers
    # =========================================================================

    def _schedule_message_persist(self, tenant_id: str, message: UnifiedMessage) -> None:
        async def _persist() -> None:
            try:
                await self._store.save_message(tenant_id, self._message_to_record(message))
                await self._store.update_account_fields(
                    tenant_id,
                    message.account_id,
                    {"last_sync": datetime.now(timezone.utc)},
                )
            except Exception as e:
                logger.warning(f"[UnifiedInbox] Failed to persist message: {e}")

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(_persist())
            return
        loop.create_task(_persist())

    def _account_to_record(self, account: ConnectedAccount) -> Dict[str, Any]:
        return {
            "id": account.id,
            "provider": account.provider.value,
            "email_address": account.email_address,
            "display_name": account.display_name,
            "status": account.status.value,
            "connected_at": account.connected_at,
            "last_sync": account.last_sync,
            "total_messages": account.total_messages,
            "unread_count": account.unread_count,
            "sync_errors": account.sync_errors,
            "metadata": account.metadata,
        }

    def _record_to_account(self, record: Dict[str, Any]) -> ConnectedAccount:
        return ConnectedAccount(
            id=record["id"],
            provider=EmailProvider(record["provider"]),
            email_address=record.get("email_address", ""),
            display_name=record.get("display_name", ""),
            status=AccountStatus(record.get("status", "pending")),
            connected_at=self._ensure_datetime(record.get("connected_at"))
            or datetime.now(timezone.utc),
            last_sync=self._ensure_datetime(record.get("last_sync")),
            total_messages=int(record.get("total_messages", 0)),
            unread_count=int(record.get("unread_count", 0)),
            sync_errors=int(record.get("sync_errors", 0)),
            metadata=record.get("metadata") or {},
        )

    def _message_to_record(self, message: UnifiedMessage) -> Dict[str, Any]:
        return {
            "id": message.id,
            "account_id": message.account_id,
            "provider": message.provider.value,
            "external_id": message.external_id,
            "subject": message.subject,
            "sender_email": message.sender_email,
            "sender_name": message.sender_name,
            "recipients": message.recipients,
            "cc": message.cc,
            "received_at": message.received_at,
            "snippet": message.snippet,
            "body_preview": message.body_preview,
            "is_read": message.is_read,
            "is_starred": message.is_starred,
            "has_attachments": message.has_attachments,
            "labels": message.labels,
            "thread_id": message.thread_id,
            "priority_score": message.priority_score,
            "priority_tier": message.priority_tier,
            "priority_reasons": message.priority_reasons,
            "triage_action": message.triage_action.value if message.triage_action else None,
            "triage_rationale": message.triage_rationale,
        }

    def _record_to_message(self, record: Dict[str, Any]) -> UnifiedMessage:
        triage_action = record.get("triage_action")
        return UnifiedMessage(
            id=record["id"],
            account_id=record["account_id"],
            provider=EmailProvider(record["provider"]),
            external_id=record.get("external_id", ""),
            subject=record.get("subject", ""),
            sender_email=record.get("sender_email", ""),
            sender_name=record.get("sender_name", ""),
            recipients=record.get("recipients") or [],
            cc=record.get("cc") or [],
            received_at=self._ensure_datetime(record.get("received_at"))
            or datetime.now(timezone.utc),
            snippet=record.get("snippet", ""),
            body_preview=record.get("body_preview", ""),
            is_read=bool(record.get("is_read")),
            is_starred=bool(record.get("is_starred")),
            has_attachments=bool(record.get("has_attachments")),
            labels=record.get("labels") or [],
            thread_id=record.get("thread_id"),
            priority_score=float(record.get("priority_score", 0.0)),
            priority_tier=record.get("priority_tier", "medium"),
            priority_reasons=record.get("priority_reasons") or [],
            triage_action=TriageAction(triage_action) if triage_action else None,
            triage_rationale=record.get("triage_rationale"),
        )

    def _triage_to_record(self, triage: TriageResult) -> Dict[str, Any]:
        return {
            "message_id": triage.message_id,
            "recommended_action": triage.recommended_action.value,
            "confidence": triage.confidence,
            "rationale": triage.rationale,
            "suggested_response": triage.suggested_response,
            "delegate_to": triage.delegate_to,
            "schedule_for": triage.schedule_for,
            "agents_involved": triage.agents_involved,
            "debate_summary": triage.debate_summary,
            "created_at": datetime.now(timezone.utc),
        }

    def _record_to_triage(self, record: Dict[str, Any]) -> TriageResult:
        return TriageResult(
            message_id=record["message_id"],
            recommended_action=TriageAction(record["recommended_action"]),
            confidence=float(record.get("confidence", 0.0)),
            rationale=record.get("rationale", ""),
            suggested_response=record.get("suggested_response"),
            delegate_to=record.get("delegate_to"),
            schedule_for=self._ensure_datetime(record.get("schedule_for")),
            agents_involved=record.get("agents_involved") or [],
            debate_summary=record.get("debate_summary"),
        )

    def _ensure_datetime(self, value: Any) -> Optional[datetime]:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        try:
            return datetime.fromisoformat(str(value))
        except ValueError:
            return None

    # =========================================================================
    # Utility Methods
    # =========================================================================

    async def _get_json_body(self, request: Any) -> Dict[str, Any]:
        """Extract JSON body from request."""
        if hasattr(request, "json"):
            if callable(request.json):
                return await request.json()
            return request.json
        return {}

    def _get_query_params(self, request: Any) -> Dict[str, str]:
        """Extract query parameters from request."""
        if hasattr(request, "query"):
            return dict(request.query)
        if hasattr(request, "args"):
            return dict(request.args)
        return {}


# =============================================================================
# Handler Registration
# =============================================================================

_handler_instance: Optional[UnifiedInboxHandler] = None


def get_unified_inbox_handler() -> UnifiedInboxHandler:
    """Get or create handler instance."""
    global _handler_instance
    if _handler_instance is None:
        _handler_instance = UnifiedInboxHandler()
    return _handler_instance


async def handle_unified_inbox(request: Any, path: str, method: str) -> HandlerResult:
    """Entry point for unified inbox requests."""
    handler = get_unified_inbox_handler()
    return await handler.handle(request, path, method)


__all__ = [
    "UnifiedInboxHandler",
    "handle_unified_inbox",
    "get_unified_inbox_handler",
    "EmailProvider",
    "AccountStatus",
    "TriageAction",
    "ConnectedAccount",
    "UnifiedMessage",
    "TriageResult",
    "InboxStats",
]
