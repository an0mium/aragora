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

    def get_gmail_oauth_url(self, redirect_uri: str, state: str | None = None) -> dict[str, Any]:
        """
        Get Gmail OAuth authorization URL.

        Args:
            redirect_uri: URL to redirect after authorization.
            state: Optional CSRF state parameter.

        Returns:
            OAuth URL and state.
        """
        params: dict[str, Any] = {"redirect_uri": redirect_uri}
        if state:
            params["state"] = state
        return self._client.request("GET", "/inbox/oauth/gmail", params=params)

    def get_outlook_oauth_url(self, redirect_uri: str, state: str | None = None) -> dict[str, Any]:
        """
        Get Outlook OAuth authorization URL.

        Args:
            redirect_uri: URL to redirect after authorization.
            state: Optional CSRF state parameter.

        Returns:
            OAuth URL and state.
        """
        params: dict[str, Any] = {"redirect_uri": redirect_uri}
        if state:
            params["state"] = state
        return self._client.request("GET", "/inbox/oauth/outlook", params=params)

    # =========================================================================
    # Account Management
    # =========================================================================

    def connect(
        self,
        provider: EmailProvider,
        auth_code: str,
        redirect_uri: str,
    ) -> dict[str, Any]:
        """
        Connect an email account using OAuth authorization code.

        Args:
            provider: Email provider (gmail or outlook).
            auth_code: OAuth authorization code.
            redirect_uri: Redirect URI used in OAuth flow.

        Returns:
            Connected account details.
        """
        return self._client.request(
            "POST",
            "/inbox/connect",
            json={
                "provider": provider,
                "auth_code": auth_code,
                "redirect_uri": redirect_uri,
            },
        )

    def list_accounts(self) -> dict[str, Any]:
        """
        List all connected email accounts.

        Returns:
            Dict with 'accounts' list and 'total' count.
        """
        return self._client.request("GET", "/inbox/accounts")

    def disconnect(self, account_id: str) -> dict[str, Any]:
        """
        Disconnect an email account.

        Args:
            account_id: Account ID to disconnect.

        Returns:
            Confirmation message.
        """
        return self._client.request("DELETE", f"/inbox/accounts/{account_id}")

    # =========================================================================
    # Messages
    # =========================================================================

    def list_messages(
        self,
        priority: PriorityTier | None = None,
        account_id: str | None = None,
        unread_only: bool = False,
        search: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict[str, Any]:
        """
        Get prioritized messages across all accounts.

        Args:
            priority: Filter by priority tier.
            account_id: Filter by account.
            unread_only: Only return unread messages.
            search: Search query.
            limit: Maximum number of results.
            offset: Pagination offset.

        Returns:
            Paginated message list sorted by priority.
        """
        params: dict[str, Any] = {}
        if priority:
            params["priority"] = priority
        if account_id:
            params["account_id"] = account_id
        if unread_only:
            params["unread_only"] = True
        if search:
            params["search"] = search
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset
        return self._client.request("GET", "/inbox/messages", params=params if params else None)

    def get_message(self, message_id: str) -> dict[str, Any]:
        """
        Get details of a specific message.

        Args:
            message_id: Message ID.

        Returns:
            Message details with triage result if available.
        """
        return self._client.request("GET", f"/inbox/messages/{message_id}")

    # =========================================================================
    # Triage
    # =========================================================================

    def triage(
        self,
        message_ids: list[str],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Run multi-agent triage on messages.

        Uses AI agents to analyze messages and recommend actions
        based on priority, sender importance, and content analysis.

        Args:
            message_ids: List of message IDs to triage.
            context: Optional context with urgency keywords or delegate options.

        Returns:
            Triage results with recommendations.
        """
        data: dict[str, Any] = {"message_ids": message_ids}
        if context:
            data["context"] = context
        return self._client.request("POST", "/inbox/triage", json=data)

    def bulk_action(
        self,
        message_ids: list[str],
        action: BulkAction,
    ) -> dict[str, Any]:
        """
        Execute bulk action on messages.

        Args:
            message_ids: List of message IDs.
            action: Action to perform (archive, mark_read, etc.).

        Returns:
            Action results with success/error counts.
        """
        return self._client.request(
            "POST",
            "/inbox/bulk-action",
            json={"message_ids": message_ids, "action": action},
        )

    # =========================================================================
    # Analytics
    # =========================================================================

    def get_stats(self) -> dict[str, Any]:
        """
        Get inbox health statistics.

        Returns:
            Comprehensive inbox metrics.
        """
        return self._client.request("GET", "/inbox/stats")

    def get_trends(self, days: int = 7) -> dict[str, Any]:
        """
        Get priority trends over time.

        Args:
            days: Number of days to analyze (default: 7).

        Returns:
            Trend analysis.
        """
        return self._client.request("GET", "/inbox/trends", params={"days": days})

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def list(self, **kwargs: Any) -> dict[str, Any]:
        """List messages (alias for list_messages)."""
        return self.list_messages(**kwargs)

    def get(self, message_id: str) -> UnifiedMessage:
        """Get message by ID."""
        result = self.get_message(message_id)
        return result.get("message", result)

    def send(
        self,
        channel: str,
        to: str,
        content: str,
        subject: str | None = None,
    ) -> dict[str, Any]:
        """
        Send a new message.

        Args:
            channel: Channel to send through.
            to: Recipient address.
            content: Message content.
            subject: Optional subject line.

        Returns:
            Send result with message ID.
        """
        data: dict[str, Any] = {"channel": channel, "to": to, "content": content}
        if subject:
            data["subject"] = subject
        return self._client.request("POST", "/inbox/messages/send", json=data)

    def reply(self, message_id: str, content: str) -> dict[str, Any]:
        """
        Reply to a message.

        Args:
            message_id: ID of message to reply to.
            content: Reply content.

        Returns:
            Reply result with new message ID.
        """
        return self._client.request(
            "POST",
            f"/inbox/messages/{message_id}/reply",
            json={"content": content},
        )


class AsyncUnifiedInboxAPI:
    """Asynchronous Unified Inbox API."""

    def __init__(self, client: AragoraAsyncClient) -> None:
        self._client = client

    async def get_gmail_oauth_url(
        self, redirect_uri: str, state: str | None = None
    ) -> dict[str, Any]:
        """Get Gmail OAuth authorization URL."""
        params: dict[str, Any] = {"redirect_uri": redirect_uri}
        if state:
            params["state"] = state
        return await self._client.request("GET", "/inbox/oauth/gmail", params=params)

    async def get_outlook_oauth_url(
        self, redirect_uri: str, state: str | None = None
    ) -> dict[str, Any]:
        """Get Outlook OAuth authorization URL."""
        params: dict[str, Any] = {"redirect_uri": redirect_uri}
        if state:
            params["state"] = state
        return await self._client.request("GET", "/inbox/oauth/outlook", params=params)

    async def connect(
        self,
        provider: EmailProvider,
        auth_code: str,
        redirect_uri: str,
    ) -> dict[str, Any]:
        """Connect an email account using OAuth authorization code."""
        return await self._client.request(
            "POST",
            "/inbox/connect",
            json={
                "provider": provider,
                "auth_code": auth_code,
                "redirect_uri": redirect_uri,
            },
        )

    async def list_accounts(self) -> dict[str, Any]:
        """List all connected email accounts."""
        return await self._client.request("GET", "/inbox/accounts")

    async def disconnect(self, account_id: str) -> dict[str, Any]:
        """Disconnect an email account."""
        return await self._client.request("DELETE", f"/inbox/accounts/{account_id}")

    async def list_messages(
        self,
        priority: PriorityTier | None = None,
        account_id: str | None = None,
        unread_only: bool = False,
        search: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict[str, Any]:
        """Get prioritized messages across all accounts."""
        params: dict[str, Any] = {}
        if priority:
            params["priority"] = priority
        if account_id:
            params["account_id"] = account_id
        if unread_only:
            params["unread_only"] = True
        if search:
            params["search"] = search
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset
        return await self._client.request(
            "GET", "/inbox/messages", params=params if params else None
        )

    async def get_message(self, message_id: str) -> dict[str, Any]:
        """Get details of a specific message."""
        return await self._client.request("GET", f"/inbox/messages/{message_id}")

    async def triage(
        self,
        message_ids: list[str],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run multi-agent triage on messages."""
        data: dict[str, Any] = {"message_ids": message_ids}
        if context:
            data["context"] = context
        return await self._client.request("POST", "/inbox/triage", json=data)

    async def bulk_action(
        self,
        message_ids: list[str],
        action: BulkAction,
    ) -> dict[str, Any]:
        """Execute bulk action on messages."""
        return await self._client.request(
            "POST",
            "/inbox/bulk-action",
            json={"message_ids": message_ids, "action": action},
        )

    async def get_stats(self) -> dict[str, Any]:
        """Get inbox health statistics."""
        return await self._client.request("GET", "/inbox/stats")

    async def get_trends(self, days: int = 7) -> dict[str, Any]:
        """Get priority trends over time."""
        return await self._client.request("GET", "/inbox/trends", params={"days": days})

    async def list(self, **kwargs: Any) -> dict[str, Any]:
        """List messages (alias for list_messages)."""
        return await self.list_messages(**kwargs)

    async def get(self, message_id: str) -> UnifiedMessage:
        """Get message by ID."""
        result = await self.get_message(message_id)
        return result.get("message", result)

    async def send(
        self,
        channel: str,
        to: str,
        content: str,
        subject: str | None = None,
    ) -> dict[str, Any]:
        """Send a new message."""
        data: dict[str, Any] = {"channel": channel, "to": to, "content": content}
        if subject:
            data["subject"] = subject
        return await self._client.request("POST", "/inbox/messages/send", json=data)

    async def reply(self, message_id: str, content: str) -> dict[str, Any]:
        """Reply to a message."""
        return await self._client.request(
            "POST",
            f"/inbox/messages/{message_id}/reply",
            json={"content": content},
        )
