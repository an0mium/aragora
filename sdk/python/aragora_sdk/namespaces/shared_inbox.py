"""
Shared Inbox Namespace API

Provides methods for shared inbox management:
- Message handling
- Assignment and routing
- SLA tracking
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

class SharedInboxAPI:
    """Synchronous Shared Inbox API."""

    def __init__(self, client: AragoraClient):
        self._client = client

class AsyncSharedInboxAPI:
    """Asynchronous Shared Inbox API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

