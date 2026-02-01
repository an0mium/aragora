"""
Inbox Command Center Namespace API.

Provides methods for inbox management:
- Prioritized email fetching
- Quick actions (archive, snooze, reply, forward)
- Bulk operations
- Sender profiles
- Daily digest statistics

Endpoints:
    GET  /api/v1/inbox/command        - Fetch prioritized inbox
    POST /api/v1/inbox/actions        - Execute quick action
    POST /api/v1/inbox/bulk-actions   - Execute bulk action
    GET  /api/v1/inbox/sender-profile - Get sender profile
    GET  /api/v1/inbox/daily-digest   - Get daily digest
    POST /api/v1/inbox/reprioritize   - Trigger AI re-prioritization
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
        >>> # Get prioritized inbox
        >>> inbox = client.inbox_command.get_inbox(limit=50)
        >>> for email in inbox["emails"]:
        ...     print(f"{email['priority']}: {email['subject']}")
        >>> # Archive low-priority emails
        >>> client.inbox_command.bulk_action("archive", "low")
    """

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    # =========================================================================
    # Inbox
    # =========================================================================

    def get_inbox(
        self,
        limit: int = 50,
        offset: int = 0,
        priority: Priority | None = None,
        unread_only: bool = False,
    ) -> dict[str, Any]:
        """
        Fetch prioritized inbox with stats.

        Args:
            limit: Max emails to return (default 50, max 1000).
            offset: Pagination offset.
            priority: Filter by priority level.
            unread_only: Only return unread emails.

        Returns:
            Dict with emails array, total count, and stats.
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if priority:
            params["priority"] = priority
        if unread_only:
            params["unread_only"] = "true"

        return self._client.request("GET", "/api/v1/inbox/command", params=params)

    # =========================================================================
    # Actions
    # =========================================================================

    def quick_action(
        self,
        action: Action,
        email_ids: list[str],
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Execute quick action on email(s).

        Args:
            action: Action to perform.
            email_ids: List of email IDs to act on.
            params: Optional action-specific parameters.

        Returns:
            Dict with action, processed count, and results.
        """
        data: dict[str, Any] = {"action": action, "emailIds": email_ids}
        if params:
            data["params"] = params

        return self._client.request("POST", "/api/v1/inbox/actions", json=data)

    def bulk_action(
        self,
        action: Action,
        filter: BulkFilter,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Execute bulk action based on filter.

        Args:
            action: Action to perform.
            filter: Filter to apply (low, deferred, spam, read, all).
            params: Optional action-specific parameters.

        Returns:
            Dict with action, filter, processed count, and results.
        """
        data: dict[str, Any] = {"action": action, "filter": filter}
        if params:
            data["params"] = params

        return self._client.request("POST", "/api/v1/inbox/bulk-actions", json=data)

    # =========================================================================
    # Sender Profile
    # =========================================================================

    def get_sender_profile(self, email: str) -> dict[str, Any]:
        """
        Get profile information for a sender.

        Args:
            email: Sender email address.

        Returns:
            Dict with sender profile (name, isVip, responseRate, etc.).
        """
        return self._client.request("GET", "/api/v1/inbox/sender-profile", params={"email": email})

    # =========================================================================
    # Daily Digest
    # =========================================================================

    def get_daily_digest(self) -> dict[str, Any]:
        """
        Get daily digest statistics.

        Returns:
            Dict with emailsReceived, processed, criticalHandled, timeSaved, etc.
        """
        return self._client.request("GET", "/api/v1/inbox/daily-digest")

    # =========================================================================
    # Reprioritization
    # =========================================================================

    def reprioritize(
        self,
        email_ids: list[str] | None = None,
        force_tier: ForceTier | None = None,
    ) -> dict[str, Any]:
        """
        Trigger AI re-prioritization of inbox.

        Args:
            email_ids: Optional list of specific email IDs to reprioritize.
            force_tier: Optional tier to force (tier_1_rules, tier_2_lightweight, tier_3_debate).

        Returns:
            Dict with reprioritized count, changes, and tier_used.
        """
        data: dict[str, Any] = {}
        if email_ids:
            data["emailIds"] = email_ids
        if force_tier:
            data["force_tier"] = force_tier

        return self._client.request("POST", "/api/v1/inbox/reprioritize", json=data)


class AsyncInboxCommandAPI:
    """Asynchronous Inbox Command API."""

    def __init__(self, client: AragoraAsyncClient) -> None:
        self._client = client

    # =========================================================================
    # Inbox
    # =========================================================================

    async def get_inbox(
        self,
        limit: int = 50,
        offset: int = 0,
        priority: Priority | None = None,
        unread_only: bool = False,
    ) -> dict[str, Any]:
        """Fetch prioritized inbox with stats."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if priority:
            params["priority"] = priority
        if unread_only:
            params["unread_only"] = "true"

        return await self._client.request("GET", "/api/v1/inbox/command", params=params)

    # =========================================================================
    # Actions
    # =========================================================================

    async def quick_action(
        self,
        action: Action,
        email_ids: list[str],
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute quick action on email(s)."""
        data: dict[str, Any] = {"action": action, "emailIds": email_ids}
        if params:
            data["params"] = params

        return await self._client.request("POST", "/api/v1/inbox/actions", json=data)

    async def bulk_action(
        self,
        action: Action,
        filter: BulkFilter,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute bulk action based on filter."""
        data: dict[str, Any] = {"action": action, "filter": filter}
        if params:
            data["params"] = params

        return await self._client.request("POST", "/api/v1/inbox/bulk-actions", json=data)

    # =========================================================================
    # Sender Profile
    # =========================================================================

    async def get_sender_profile(self, email: str) -> dict[str, Any]:
        """Get profile information for a sender."""
        return await self._client.request(
            "GET", "/api/v1/inbox/sender-profile", params={"email": email}
        )

    # =========================================================================
    # Daily Digest
    # =========================================================================

    async def get_daily_digest(self) -> dict[str, Any]:
        """Get daily digest statistics."""
        return await self._client.request("GET", "/api/v1/inbox/daily-digest")

    # =========================================================================
    # Reprioritization
    # =========================================================================

    async def reprioritize(
        self,
        email_ids: list[str] | None = None,
        force_tier: ForceTier | None = None,
    ) -> dict[str, Any]:
        """Trigger AI re-prioritization of inbox."""
        data: dict[str, Any] = {}
        if email_ids:
            data["emailIds"] = email_ids
        if force_tier:
            data["force_tier"] = force_tier

        return await self._client.request("POST", "/api/v1/inbox/reprioritize", json=data)
