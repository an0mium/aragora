"""
Status Namespace API

Provides methods for system status and health:
- Service health checks
- System metrics
- Operational status
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

class StatusAPI:
    """Synchronous Status API."""

    def __init__(self, client: AragoraClient):
        self._client = client

class AsyncStatusAPI:
    """Asynchronous Status API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

