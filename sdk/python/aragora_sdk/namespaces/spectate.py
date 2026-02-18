"""
Spectate Namespace API

Provides methods for real-time debate observation:
- Connect to Server-Sent Events (SSE) stream for a debate
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class SpectateAPI:
    """
    Synchronous Spectate API.

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> stream = client.spectate.connect_sse("debate-123")
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    def connect_sse(self, debate_id: str) -> dict[str, Any]:
        """
        Connect to SSE stream for a debate.

        Returns connection details including the stream URL.
        Use the stream URL with an SSE client for real-time events.
        """
        return self._client.request("GET", f"/api/v1/spectate/{debate_id}/stream")


class AsyncSpectateAPI:
    """
    Asynchronous Spectate API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     stream = await client.spectate.connect_sse("debate-123")
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def connect_sse(self, debate_id: str) -> dict[str, Any]:
        """
        Connect to SSE stream for a debate.

        Returns connection details including the stream URL.
        """
        return await self._client.request("GET", f"/api/v1/spectate/{debate_id}/stream")
