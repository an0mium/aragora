"""
Shared Inbox Namespace API

Provides methods for shared inbox management:
- Email account connections and OAuth
- Cross-account message listing and details
- Message routing rules
- Inbox statistics and trends
- AI-powered triage
- Bulk actions
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class SharedInboxAPI:
    """
    Synchronous Shared Inbox API.

    Provides a unified view across connected email accounts with
    routing rules, triage, and analytics.

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> accounts = client.shared_inbox.list_accounts()
        >>> messages = client.shared_inbox.list_messages(limit=50)
        >>> stats = client.shared_inbox.get_stats()
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    # =========================================================================
    # Account Management
    # =========================================================================

    def list_accounts(self) -> dict[str, Any]:
        """
        List connected email accounts.

        Returns:
            Dict with connected email accounts and their configurations.
        """
        return self._client.request("GET", "/api/v1/inbox/accounts")

    def get_account(self, account_id: str) -> dict[str, Any]:
        """
        Get details for a specific email account.

        Args:
            account_id: Account identifier.

        Returns:
            Dict with account details and sync status.
        """
        return self._client.request("GET", f"/api/v1/inbox/accounts/{account_id}")

    def connect(self, **kwargs: Any) -> dict[str, Any]:
        """
        Connect a new email account.

        Args:
            **kwargs: Connection parameters including:
                - provider: Email provider (gmail, outlook)
                - credentials: Authentication credentials

        Returns:
            Dict with connection status and account details.
        """
        return self._client.request("POST", "/api/v1/inbox/connect", json=kwargs)

    def oauth_gmail(self, **kwargs: Any) -> dict[str, Any]:
        """
        Initiate Gmail OAuth flow.

        Returns:
            Dict with OAuth authorization URL and state token.
        """
        return self._client.request("GET", "/api/v1/inbox/oauth/gmail", params=kwargs or None)

    def oauth_outlook(self, **kwargs: Any) -> dict[str, Any]:
        """
        Initiate Outlook OAuth flow.

        Returns:
            Dict with OAuth authorization URL and state token.
        """
        return self._client.request("GET", "/api/v1/inbox/oauth/outlook", params=kwargs or None)

    # =========================================================================
    # Messages
    # =========================================================================

    def list_messages(
        self,
        limit: int = 50,
        offset: int = 0,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        List messages across all connected accounts.

        Args:
            limit: Maximum messages to return.
            offset: Pagination offset.
            **kwargs: Additional filters (status, labels, etc.).

        Returns:
            Dict with messages list sorted by priority.
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset, **kwargs}
        return self._client.request("GET", "/api/v1/inbox/messages", params=params)

    def get_message(self, message_id: str) -> dict[str, Any]:
        """
        Get a specific message by ID.

        Args:
            message_id: Message identifier.

        Returns:
            Dict with full message details including body and metadata.
        """
        return self._client.request("GET", f"/api/v1/inbox/messages/{message_id}")

    # =========================================================================
    # Shared Inbox & Routing
    # =========================================================================

    def list_shared(self, **kwargs: Any) -> dict[str, Any]:
        """
        List shared inbox items.

        Returns:
            Dict with shared inbox items visible to the team.
        """
        return self._client.request("GET", "/api/v1/inbox/shared", params=kwargs or None)

    def list_routing_rules(self) -> dict[str, Any]:
        """
        List inbox routing rules.

        Returns:
            Dict with configured routing rules.
        """
        return self._client.request("GET", "/api/v1/inbox/routing/rules")

    # =========================================================================
    # Analytics & Triage
    # =========================================================================

    def get_stats(self) -> dict[str, Any]:
        """
        Get inbox statistics.

        Returns:
            Dict with inbox metrics including volume, response times,
            and team performance.
        """
        return self._client.request("GET", "/api/v1/inbox/stats")

    def get_trends(self, **kwargs: Any) -> dict[str, Any]:
        """
        Get inbox trends over time.

        Returns:
            Dict with trend data for message volume, response times,
            and other metrics.
        """
        return self._client.request("GET", "/api/v1/inbox/trends", params=kwargs or None)

    def triage(self, **kwargs: Any) -> dict[str, Any]:
        """
        AI-powered inbox triage.

        Analyzes messages and suggests priority, category, and
        assignment based on content and context.

        Args:
            **kwargs: Triage parameters including:
                - message_ids: List of messages to triage
                - auto_assign: Whether to auto-assign based on rules

        Returns:
            Dict with triage results including priority and routing
            suggestions for each message.
        """
        return self._client.request("POST", "/api/v1/inbox/triage", json=kwargs)

    def bulk_action(self, **kwargs: Any) -> dict[str, Any]:
        """
        Perform bulk actions on multiple messages.

        Args:
            **kwargs: Bulk action parameters including:
                - message_ids: List of message IDs
                - action: Action to perform (archive, mark_read, assign, tag)
                - value: Action value (e.g., assignee for 'assign')

        Returns:
            Dict with bulk action results and any failures.
        """
        return self._client.request("POST", "/api/v1/inbox/bulk-action", json=kwargs)

    def list_mentions(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        List inbox mentions for the current user.

        Args:
            limit: Maximum mentions to return.
            offset: Pagination offset.

        Returns:
            Dict with mentions array and count.
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        return self._client.request("GET", "/api/v1/inbox/mentions", params=params)


class AsyncSharedInboxAPI:
    """
    Asynchronous Shared Inbox API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     accounts = await client.shared_inbox.list_accounts()
        ...     messages = await client.shared_inbox.list_messages()
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    # =========================================================================
    # Account Management
    # =========================================================================

    async def list_accounts(self) -> dict[str, Any]:
        """List connected email accounts."""
        return await self._client.request("GET", "/api/v1/inbox/accounts")

    async def get_account(self, account_id: str) -> dict[str, Any]:
        """Get details for a specific email account."""
        return await self._client.request("GET", f"/api/v1/inbox/accounts/{account_id}")

    async def connect(self, **kwargs: Any) -> dict[str, Any]:
        """Connect a new email account."""
        return await self._client.request("POST", "/api/v1/inbox/connect", json=kwargs)

    async def oauth_gmail(self, **kwargs: Any) -> dict[str, Any]:
        """Initiate Gmail OAuth flow."""
        return await self._client.request("GET", "/api/v1/inbox/oauth/gmail", params=kwargs or None)

    async def oauth_outlook(self, **kwargs: Any) -> dict[str, Any]:
        """Initiate Outlook OAuth flow."""
        return await self._client.request(
            "GET", "/api/v1/inbox/oauth/outlook", params=kwargs or None
        )

    # =========================================================================
    # Messages
    # =========================================================================

    async def list_messages(
        self,
        limit: int = 50,
        offset: int = 0,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """List messages across all connected accounts."""
        params: dict[str, Any] = {"limit": limit, "offset": offset, **kwargs}
        return await self._client.request("GET", "/api/v1/inbox/messages", params=params)

    async def get_message(self, message_id: str) -> dict[str, Any]:
        """Get a specific message by ID."""
        return await self._client.request("GET", f"/api/v1/inbox/messages/{message_id}")

    # =========================================================================
    # Shared Inbox & Routing
    # =========================================================================

    async def list_shared(self, **kwargs: Any) -> dict[str, Any]:
        """List shared inbox items."""
        return await self._client.request("GET", "/api/v1/inbox/shared", params=kwargs or None)

    async def list_routing_rules(self) -> dict[str, Any]:
        """List inbox routing rules."""
        return await self._client.request("GET", "/api/v1/inbox/routing/rules")

    # =========================================================================
    # Analytics & Triage
    # =========================================================================

    async def get_stats(self) -> dict[str, Any]:
        """Get inbox statistics."""
        return await self._client.request("GET", "/api/v1/inbox/stats")

    async def get_trends(self, **kwargs: Any) -> dict[str, Any]:
        """Get inbox trends over time."""
        return await self._client.request("GET", "/api/v1/inbox/trends", params=kwargs or None)

    async def triage(self, **kwargs: Any) -> dict[str, Any]:
        """AI-powered inbox triage."""
        return await self._client.request("POST", "/api/v1/inbox/triage", json=kwargs)

    async def bulk_action(self, **kwargs: Any) -> dict[str, Any]:
        """Perform bulk actions on multiple messages."""
        return await self._client.request("POST", "/api/v1/inbox/bulk-action", json=kwargs)

    async def list_mentions(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List inbox mentions for the current user."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        return await self._client.request("GET", "/api/v1/inbox/mentions", params=params)
