"""
Notifications Namespace API

Provides endpoints for managing notification preferences and sending notifications.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

class NotificationsAPI:
    """Synchronous Notifications API."""

    def __init__(self, client: AragoraClient):
        self._client = client

class AsyncNotificationsAPI:
    """Asynchronous Notifications API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

