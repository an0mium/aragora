"""
Inbox Command Center Namespace API.

Provides methods for inbox management:
- Prioritized email fetching
- Quick actions (archive, snooze, reply, forward)
- Bulk operations
- Sender profiles
- Daily digest statistics
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

Priority = Literal["critical", "high", "medium", "low", "defer"]
Action = Literal[
    "archive", "snooze", "reply", "forward", "spam", "mark_important", "mark_vip", "block", "delete"
]
BulkFilter = Literal["low", "deferred", "spam", "read", "all"]
ForceTier = Literal["tier_1_rules", "tier_2_lightweight", "tier_3_debate"]


class InboxCommandAPI:
    """
    Synchronous Inbox Command API.

    Provides methods for managing and prioritizing inbox emails.

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> inbox = client.inbox_command.get_inbox(limit=50)
        >>> for email in inbox["emails"]:
        ...     print(f"{email['priority']}: {email['subject']}")
    """

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    # =========================================================================
    # Inbox
    # =========================================================================

    def get_inbox(self) -> dict[str, Any]:
        """Fetch prioritized inbox."""
        return self._client.request("GET", "/api/v1/inbox/command")

    def execute_action(self, **kwargs: Any) -> dict[str, Any]:
        """Execute a quick action on a message."""
        return self._client.request("POST", "/api/v1/inbox/actions", json=kwargs)

    def bulk_actions(self, **kwargs: Any) -> dict[str, Any]:
        """Execute bulk actions on messages."""
        return self._client.request("POST", "/api/v1/inbox/bulk-actions", json=kwargs)

    def get_sender_profile(self) -> dict[str, Any]:
        """Get sender profile."""
        return self._client.request("GET", "/api/v1/inbox/sender-profile")

    def get_daily_digest(self) -> dict[str, Any]:
        """Get daily inbox digest."""
        return self._client.request("GET", "/api/v1/inbox/daily-digest")

    def reprioritize(self, **kwargs: Any) -> dict[str, Any]:
        """Trigger AI re-prioritization."""
        return self._client.request("POST", "/api/v1/inbox/reprioritize", json=kwargs)

    def get_stats(self) -> dict[str, Any]:
        """Get inbox statistics."""
        return self._client.request("GET", "/api/v1/inbox/stats")

    def get_trends(self) -> dict[str, Any]:
        """Get inbox trends."""
        return self._client.request("GET", "/api/v1/inbox/trends")

    def get_triage(self) -> dict[str, Any]:
        """Get inbox triage suggestions."""
        return self._client.request("GET", "/api/v1/inbox/triage")

    # =========================================================================
    # Messages
    # =========================================================================

    def list_messages(self) -> dict[str, Any]:
        """List inbox messages."""
        return self._client.request("GET", "/api/v1/inbox/messages")

    def get_message(self, message_id: str) -> dict[str, Any]:
        """Get a specific inbox message."""
        return self._client.request("GET", f"/api/v1/inbox/messages/{message_id}")

    # =========================================================================
    # Accounts & OAuth
    # =========================================================================

    def list_accounts(self) -> dict[str, Any]:
        """List connected inbox accounts."""
        return self._client.request("GET", "/api/v1/inbox/accounts")

    def get_account(self, account_id: str) -> dict[str, Any]:
        """Get a connected inbox account."""
        return self._client.request("GET", f"/api/v1/inbox/accounts/{account_id}")

    def connect(self) -> dict[str, Any]:
        """Connect a new inbox account."""
        return self._client.request("GET", "/api/v1/inbox/connect")

    def oauth_gmail(self) -> dict[str, Any]:
        """Get Gmail OAuth URL."""
        return self._client.request("GET", "/api/v1/inbox/oauth/gmail")

    def oauth_outlook(self) -> dict[str, Any]:
        """Get Outlook OAuth URL."""
        return self._client.request("GET", "/api/v1/inbox/oauth/outlook")

    # =========================================================================
    # Mentions
    # =========================================================================

    def list_mentions(self) -> dict[str, Any]:
        """List mentions."""
        return self._client.request("GET", "/api/v1/inbox/mentions")

    def acknowledge_mention(self, mention_id: str) -> dict[str, Any]:
        """Acknowledge a mention."""
        return self._client.request("POST", f"/api/v1/inbox/mentions/{mention_id}/acknowledge")

    # =========================================================================
    # Routing Rules
    # =========================================================================

    def list_routing_rules(self) -> dict[str, Any]:
        """List routing rules."""
        return self._client.request("GET", "/api/v1/inbox/routing/rules")

    def create_routing_rule(self, **kwargs: Any) -> dict[str, Any]:
        """Create a routing rule."""
        return self._client.request("POST", "/api/v1/inbox/routing/rules", json=kwargs)

    # =========================================================================
    # Shared Inboxes
    # =========================================================================

    def list_shared_inboxes(self) -> dict[str, Any]:
        """List shared inboxes."""
        return self._client.request("GET", "/api/v1/inbox/shared")

    def create_shared_inbox(self, **kwargs: Any) -> dict[str, Any]:
        """Create a shared inbox."""
        return self._client.request("POST", "/api/v1/inbox/shared", json=kwargs)


class AsyncInboxCommandAPI:
    """Asynchronous Inbox Command API."""

    def __init__(self, client: AragoraAsyncClient) -> None:
        self._client = client

    async def get_inbox(self) -> dict[str, Any]:
        """Fetch prioritized inbox."""
        return await self._client.request("GET", "/api/v1/inbox/command")

    async def execute_action(self, **kwargs: Any) -> dict[str, Any]:
        """Execute a quick action on a message."""
        return await self._client.request("POST", "/api/v1/inbox/actions", json=kwargs)

    async def bulk_actions(self, **kwargs: Any) -> dict[str, Any]:
        """Execute bulk actions on messages."""
        return await self._client.request("POST", "/api/v1/inbox/bulk-actions", json=kwargs)

    async def get_sender_profile(self) -> dict[str, Any]:
        """Get sender profile."""
        return await self._client.request("GET", "/api/v1/inbox/sender-profile")

    async def get_daily_digest(self) -> dict[str, Any]:
        """Get daily inbox digest."""
        return await self._client.request("GET", "/api/v1/inbox/daily-digest")

    async def reprioritize(self, **kwargs: Any) -> dict[str, Any]:
        """Trigger AI re-prioritization."""
        return await self._client.request("POST", "/api/v1/inbox/reprioritize", json=kwargs)

    async def get_stats(self) -> dict[str, Any]:
        """Get inbox statistics."""
        return await self._client.request("GET", "/api/v1/inbox/stats")

    async def get_trends(self) -> dict[str, Any]:
        """Get inbox trends."""
        return await self._client.request("GET", "/api/v1/inbox/trends")

    async def get_triage(self) -> dict[str, Any]:
        """Get inbox triage suggestions."""
        return await self._client.request("GET", "/api/v1/inbox/triage")

    async def list_messages(self) -> dict[str, Any]:
        """List inbox messages."""
        return await self._client.request("GET", "/api/v1/inbox/messages")

    async def get_message(self, message_id: str) -> dict[str, Any]:
        """Get a specific inbox message."""
        return await self._client.request("GET", f"/api/v1/inbox/messages/{message_id}")

    async def list_accounts(self) -> dict[str, Any]:
        """List connected inbox accounts."""
        return await self._client.request("GET", "/api/v1/inbox/accounts")

    async def get_account(self, account_id: str) -> dict[str, Any]:
        """Get a connected inbox account."""
        return await self._client.request("GET", f"/api/v1/inbox/accounts/{account_id}")

    async def connect(self) -> dict[str, Any]:
        """Connect a new inbox account."""
        return await self._client.request("GET", "/api/v1/inbox/connect")

    async def oauth_gmail(self) -> dict[str, Any]:
        """Get Gmail OAuth URL."""
        return await self._client.request("GET", "/api/v1/inbox/oauth/gmail")

    async def oauth_outlook(self) -> dict[str, Any]:
        """Get Outlook OAuth URL."""
        return await self._client.request("GET", "/api/v1/inbox/oauth/outlook")

    async def list_mentions(self) -> dict[str, Any]:
        """List mentions."""
        return await self._client.request("GET", "/api/v1/inbox/mentions")

    async def acknowledge_mention(self, mention_id: str) -> dict[str, Any]:
        """Acknowledge a mention."""
        return await self._client.request("POST", f"/api/v1/inbox/mentions/{mention_id}/acknowledge")

    async def list_routing_rules(self) -> dict[str, Any]:
        """List routing rules."""
        return await self._client.request("GET", "/api/v1/inbox/routing/rules")

    async def create_routing_rule(self, **kwargs: Any) -> dict[str, Any]:
        """Create a routing rule."""
        return await self._client.request("POST", "/api/v1/inbox/routing/rules", json=kwargs)

    async def list_shared_inboxes(self) -> dict[str, Any]:
        """List shared inboxes."""
        return await self._client.request("GET", "/api/v1/inbox/shared")

    async def create_shared_inbox(self, **kwargs: Any) -> dict[str, Any]:
        """Create a shared inbox."""
        return await self._client.request("POST", "/api/v1/inbox/shared", json=kwargs)
