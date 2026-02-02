"""
Insights Namespace API

Provides access to debate insights and extracted knowledge.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class InsightsAPI:
    """Synchronous Insights API for debate insights."""

    def __init__(self, client: AragoraClient):
        self._client = client

    def get_recent(
        self,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Get recent insights.

        Args:
            limit: Maximum results.
            offset: Pagination offset.

        Returns:
            Recent insights with metadata.
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        return self._client.request("GET", "/api/v1/insights/recent", params=params)

    def extract_detailed(
        self,
        debate_id: str,
        include_evidence: bool = True,
    ) -> dict[str, Any]:
        """Extract detailed insights from a debate.

        Args:
            debate_id: Debate to extract from.
            include_evidence: Include supporting evidence.

        Returns:
            Extracted insights with details.
        """
        body: dict[str, Any] = {
            "debate_id": debate_id,
            "include_evidence": include_evidence,
        }
        return self._client.request("POST", "/api/v1/insights/extract-detailed", json=body)


class AsyncInsightsAPI:
    """Asynchronous Insights API for debate insights."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def get_recent(
        self,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Get recent insights.

        Args:
            limit: Maximum results.
            offset: Pagination offset.

        Returns:
            Recent insights with metadata.
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        return await self._client.request("GET", "/api/v1/insights/recent", params=params)

    async def extract_detailed(
        self,
        debate_id: str,
        include_evidence: bool = True,
    ) -> dict[str, Any]:
        """Extract detailed insights from a debate.

        Args:
            debate_id: Debate to extract from.
            include_evidence: Include supporting evidence.

        Returns:
            Extracted insights with details.
        """
        body: dict[str, Any] = {
            "debate_id": debate_id,
            "include_evidence": include_evidence,
        }
        return await self._client.request("POST", "/api/v1/insights/extract-detailed", json=body)
