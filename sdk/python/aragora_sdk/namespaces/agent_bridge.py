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
        status: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict[str, Any]:
        """List recorded agent-bridge runs."""
        params: dict[str, Any] = {}
        if status is not None:
            params["status"] = status
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset
        return self._client.request("GET", "/api/v1/agent-bridge/runs", params=params)


class AsyncAgentBridgeAPI:
    """Asynchronous Agent Bridge API."""

    def __init__(self, client: AragoraAsyncClient) -> None:
        self._client = client

    async def list_runs(
        self,
        *,
        status: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict[str, Any]:
        """List recorded agent-bridge runs."""
        params: dict[str, Any] = {}
        if status is not None:
            params["status"] = status
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset
        return await self._client.request("GET", "/api/v1/agent-bridge/runs", params=params)
