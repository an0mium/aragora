"""Agent Bridge namespace API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class AgentBridgeAPI:
    """Synchronous Agent Bridge API."""

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    def list_runs(
        self,
        *,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        """List recorded agent-bridge runs."""
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return self._client.request("GET", "/api/v1/agent-bridge/runs", params=params)


class AsyncAgentBridgeAPI:
    """Asynchronous Agent Bridge API."""

    def __init__(self, client: AragoraAsyncClient) -> None:
        self._client = client

    async def list_runs(
        self,
        *,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        """List recorded agent-bridge runs."""
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return await self._client.request("GET", "/api/v1/agent-bridge/runs", params=params)
