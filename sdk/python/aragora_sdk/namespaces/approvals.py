"""
Approvals Namespace API

Provides methods for the unified approvals inbox:
- List pending approvals across subsystems
- Filter by source (workflow, decision_plan, computer_use, gateway)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

_List = list


class ApprovalsAPI:
    """
    Synchronous Approvals API.

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> pending = client.approvals.list_pending()
        >>> for approval in pending["approvals"]:
        ...     print(approval["id"], approval["source"])
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    def list_pending(
        self,
        sources: _List[str] | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """
        List pending approval requests.

        Args:
            sources: Filter by source (workflow, decision_plan, computer_use, gateway)
            limit: Maximum results (default 100, max 500)

        Returns:
            Pending approvals with count and source info
        """
        params: dict[str, Any] = {"status": "pending", "limit": limit}
        if sources:
            params["sources"] = ",".join(sources)

        return self._client.request("GET", "/api/v1/approvals/pending", params=params)

    def list(
        self,
        sources: _List[str] | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """
        List all approvals.

        Args:
            sources: Filter by source
            limit: Maximum results

        Returns:
            Approvals list
        """
        params: dict[str, Any] = {"limit": limit}
        if sources:
            params["sources"] = ",".join(sources)

        return self._client.request("GET", "/api/v1/approvals", params=params)


class AsyncApprovalsAPI:
    """
    Asynchronous Approvals API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     pending = await client.approvals.list_pending()
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def list_pending(
        self,
        sources: _List[str] | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """List pending approval requests."""
        params: dict[str, Any] = {"status": "pending", "limit": limit}
        if sources:
            params["sources"] = ",".join(sources)

        return await self._client.request("GET", "/api/v1/approvals/pending", params=params)

    async def list(
        self,
        sources: _List[str] | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """List all approvals."""
        params: dict[str, Any] = {"limit": limit}
        if sources:
            params["sources"] = ",".join(sources)

        return await self._client.request("GET", "/api/v1/approvals", params=params)
