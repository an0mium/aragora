"""
Gmail Namespace API

Provides methods for Gmail integration:
- Message operations (labels, read status, archiving)
- Attachment handling
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

class GmailAPI:
    """
    Synchronous Gmail API.

    Provides methods for managing Gmail messages through the Aragora platform.

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> client.gmail.mark_read("message_id")
        >>> client.gmail.add_labels("message_id", ["Important", "Work"])
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    # ===========================================================================
    # Message Operations

class AsyncGmailAPI:
    """
    Asynchronous Gmail API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     await client.gmail.mark_read("message_id")
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    # Message Operations
