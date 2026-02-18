"""
Audience Namespace API

Provides methods for audience suggestions:
- Get suggestions for a debate
- Submit audience suggestions
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class AudienceAPI:
    """
    Synchronous Audience API.

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> suggestions = client.audience.get_suggestions("debate-123")
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    def get_suggestions(self, debate_id: str) -> dict[str, Any]:
        """Get audience suggestions for a debate."""
        return self._client.request("GET", f"/api/v1/debates/{debate_id}/audience/suggestions")

    def submit_suggestion(self, debate_id: str, suggestion: dict[str, Any]) -> dict[str, Any]:
        """Submit an audience suggestion for a debate."""
        return self._client.request(
            "POST", f"/api/v1/debates/{debate_id}/audience/suggestions", json=suggestion
        )


class AsyncAudienceAPI:
    """
    Asynchronous Audience API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     suggestions = await client.audience.get_suggestions("debate-123")
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def get_suggestions(self, debate_id: str) -> dict[str, Any]:
        """Get audience suggestions for a debate."""
        return await self._client.request(
            "GET", f"/api/v1/debates/{debate_id}/audience/suggestions"
        )

    async def submit_suggestion(self, debate_id: str, suggestion: dict[str, Any]) -> dict[str, Any]:
        """Submit an audience suggestion for a debate."""
        return await self._client.request(
            "POST", f"/api/v1/debates/{debate_id}/audience/suggestions", json=suggestion
        )
