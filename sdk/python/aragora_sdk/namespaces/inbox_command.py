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

from typing import TYPE_CHECKING, Literal

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

class AsyncInboxCommandAPI:
    """Asynchronous Inbox Command API."""

    def __init__(self, client: AragoraAsyncClient) -> None:
        self._client = client

    # =========================================================================
    # Inbox

