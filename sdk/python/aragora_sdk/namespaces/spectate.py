"""
Spectate Namespace API

Provides methods for real-time debate observation:
- Connect to Server-Sent Events (SSE) stream for a debate
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from urllib.parse import quote

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


def _build_sse_stream_info(debate_id: str) -> dict[str, Any]:
    """Return the canonical live SSE endpoint without issuing a JSON request."""
    encoded_debate_id = quote(debate_id, safe="")
    return {
        "debate_id": debate_id,
        "stream_url": f"/api/v1/debates/{encoded_debate_id}/spectate",
    }


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
        Return connection details for a debate's live SSE stream.

        The generic SDK client only supports JSON APIs, so this helper returns
        the canonical stream URL instead of attempting to open the SSE request.
        """
        return _build_sse_stream_info(debate_id)

    def get_recent(self, *, count: int = 50, debate_id: str | None = None) -> dict[str, Any]:
        """Get recent buffered spectate events."""
        params: dict[str, Any] = {"count": count}
        if debate_id:
            params["debate_id"] = debate_id
        return self._client.request("GET", "/api/v1/spectate/recent", params=params)

    def get_status(self) -> dict[str, Any]:
        """Get spectate bridge status (active sessions, subscribers, buffer size)."""
        return self._client.request("GET", "/api/v1/spectate/status")

    def get_stream(self, *, count: int = 50, debate_id: str | None = None) -> dict[str, Any]:
        """Get spectate event stream snapshot."""
        params: dict[str, Any] = {"count": count}
        if debate_id:
            params["debate_id"] = debate_id
        return self._client.request("GET", "/api/v1/spectate/stream", params=params)


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
        Return connection details for a debate's live SSE stream.

        The generic SDK client only supports JSON APIs, so this helper returns
        the canonical stream URL instead of attempting to open the SSE request.
        """
        return _build_sse_stream_info(debate_id)

    async def get_recent(self, *, count: int = 50, debate_id: str | None = None) -> dict[str, Any]:
        """Get recent buffered spectate events."""
        params: dict[str, Any] = {"count": count}
        if debate_id:
            params["debate_id"] = debate_id
        return await self._client.request("GET", "/api/v1/spectate/recent", params=params)

    async def get_status(self) -> dict[str, Any]:
        """Get spectate bridge status (active sessions, subscribers, buffer size)."""
        return await self._client.request("GET", "/api/v1/spectate/status")

    async def get_stream(self, *, count: int = 50, debate_id: str | None = None) -> dict[str, Any]:
        """Get spectate event stream snapshot."""
        params: dict[str, Any] = {"count": count}
        if debate_id:
            params["debate_id"] = debate_id
        return await self._client.request("GET", "/api/v1/spectate/stream", params=params)
