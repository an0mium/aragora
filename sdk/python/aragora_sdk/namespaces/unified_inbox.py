"""
Unified Inbox Namespace API.

Provides a namespaced interface for multi-account email management:
- Gmail and Outlook with priority scoring
- Multi-agent triage for complex messages
- Inbox health metrics and analytics
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypedDict

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

EmailProvider = Literal["gmail", "outlook"]
AccountStatus = Literal["pending", "connected", "syncing", "error", "disconnected"]
TriageAction = Literal[
    "respond_urgent",
    "respond_normal",
    "delegate",
    "schedule",
    "archive",
    "delete",
    "flag",
    "defer",
]
PriorityTier = Literal["critical", "high", "medium", "low"]
BulkAction = Literal["archive", "mark_read", "mark_unread", "star", "delete"]

class ConnectedAccount(TypedDict, total=False):
    """Connected email account."""

    id: str
    provider: str
    email_address: str
    display_name: str
    status: str
    connected_at: str
    last_sync: str | None
    total_messages: int
    unread_count: int
    sync_errors: int

class UnifiedMessage(TypedDict, total=False):
    """Unified message across providers."""

    id: str
    account_id: str
    provider: str
    external_id: str
    subject: str
    sender: dict[str, str]
    recipients: list[str]
    cc: list[str]
    received_at: str
    snippet: str
    is_read: bool
    is_starred: bool
    has_attachments: bool
    labels: list[str]
    thread_id: str | None
    priority: dict[str, Any]
    triage: dict[str, Any] | None

class TriageResult(TypedDict, total=False):
    """Triage result from multi-agent analysis."""

    message_id: str
    recommended_action: str
    confidence: float
    rationale: str
    suggested_response: str | None
    delegate_to: str | None
    schedule_for: str | None
    agents_involved: list[str]
    debate_summary: str | None

class InboxStats(TypedDict, total=False):
    """Inbox health statistics."""

    total_accounts: int
    total_messages: int
    unread_count: int
    messages_by_priority: dict[str, int]
    messages_by_provider: dict[str, int]
    avg_response_time_hours: float
    pending_triage: int
    sync_health: dict[str, Any]
    top_senders: list[dict[str, Any]]
    hourly_volume: list[dict[str, Any]]

class UnifiedInboxAPI:
    """
    Synchronous Unified Inbox API.

    Provides a single interface for Gmail and Outlook accounts with:
    - Cross-account message retrieval with priority scoring
    - Multi-agent triage for complex messages
    - Inbox health metrics and analytics

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> # List all connected accounts
        >>> accounts = client.unified_inbox.list_accounts()
        >>> # Get prioritized messages
        >>> messages = client.unified_inbox.list_messages(priority="critical")
        >>> # Run triage on messages
        >>> triage = client.unified_inbox.triage(message_ids=[msg["id"] for msg in messages])
    """

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    # =========================================================================
    # OAuth Flow
    # =========================================================================

    def list(self, **kwargs: Any) -> dict[str, Any]:
        """List messages (alias for list_messages)."""
        return self.list_messages(**kwargs)

    def get(self, message_id: str) -> dict[str, Any]:
        """Get message by ID."""
        result = self.get_message(message_id)
        msg: dict[str, Any] = result.get("message", result)
        return msg

class AsyncUnifiedInboxAPI:
    """Asynchronous Unified Inbox API."""

    def __init__(self, client: AragoraAsyncClient) -> None:
        self._client = client

    async def list(self, **kwargs: Any) -> dict[str, Any]:
        """List messages (alias for list_messages)."""
        return await self.list_messages(**kwargs)

    async def get(self, message_id: str) -> dict[str, Any]:
        """Get message by ID."""
        result = await self.get_message(message_id)
        msg: dict[str, Any] = result.get("message", result)
        return msg

