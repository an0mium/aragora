"""Blockchain namespace API (ERC-8004 endpoints)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

class BlockchainAPI:
    """Synchronous Blockchain API."""

    def __init__(self, client: AragoraClient):
        self._client = client

    def list_agents(self, skip: int = 0, limit: int = 100) -> dict[str, Any]:
        """List registered on-chain agents with pagination."""
        return self._client.request(
            "GET", "/api/v1/blockchain/agents", params={"skip": skip, "limit": limit}
        )

    def register_agent(
        self, agent_uri: str, metadata: dict[str, str] | None = None
    ) -> dict[str, Any]:
        """Register a new agent on the Identity Registry."""
        payload: dict[str, Any] = {"agent_uri": agent_uri}
        if metadata:
            payload["metadata"] = metadata
        return self._client.request("POST", "/api/v1/blockchain/agents", json=payload)

    def get_config(self) -> dict[str, Any]:
        """Get blockchain connector configuration."""
        return self._client.request("GET", "/api/v1/blockchain/config")

    def get_health(self) -> dict[str, Any]:
        """Get blockchain connector health."""
        return self._client.request("GET", "/api/v1/blockchain/health")

    def sync(
        self,
        sync_identities: bool = True,
        sync_reputation: bool = True,
        sync_validations: bool = True,
        agent_ids: list[int] | None = None,
    ) -> dict[str, Any]:
        """Trigger blockchain -> Knowledge Mound sync."""
        payload: dict[str, Any] = {
            "sync_identities": sync_identities,
            "sync_reputation": sync_reputation,
            "sync_validations": sync_validations,
        }
        if agent_ids is not None:
            payload["agent_ids"] = agent_ids
        return self._client.request("POST", "/api/v1/blockchain/sync", json=payload)

    def get_agent(self, token_id: int) -> dict[str, Any]:
        """Get on-chain agent identity by token ID."""
        return self._client.request("GET", f"/api/v1/blockchain/agents/{token_id}")

class AsyncBlockchainAPI:
    """Asynchronous Blockchain API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def list_agents(self, skip: int = 0, limit: int = 100) -> dict[str, Any]:
        """List registered on-chain agents with pagination."""
        return await self._client.request(
            "GET", "/api/v1/blockchain/agents", params={"skip": skip, "limit": limit}
        )

    async def register_agent(
        self, agent_uri: str, metadata: dict[str, str] | None = None
    ) -> dict[str, Any]:
        """Register a new agent on the Identity Registry."""
        payload: dict[str, Any] = {"agent_uri": agent_uri}
        if metadata:
            payload["metadata"] = metadata
        return await self._client.request("POST", "/api/v1/blockchain/agents", json=payload)

    async def get_config(self) -> dict[str, Any]:
        """Get blockchain connector configuration."""
        return await self._client.request("GET", "/api/v1/blockchain/config")

    async def get_health(self) -> dict[str, Any]:
        """Get blockchain connector health."""
        return await self._client.request("GET", "/api/v1/blockchain/health")

    async def sync(
        self,
        sync_identities: bool = True,
        sync_reputation: bool = True,
        sync_validations: bool = True,
        agent_ids: list[int] | None = None,
    ) -> dict[str, Any]:
        """Trigger blockchain -> Knowledge Mound sync."""
        payload: dict[str, Any] = {
            "sync_identities": sync_identities,
            "sync_reputation": sync_reputation,
            "sync_validations": sync_validations,
        }
        if agent_ids is not None:
            payload["agent_ids"] = agent_ids
        return await self._client.request("POST", "/api/v1/blockchain/sync", json=payload)

    async def get_agent(self, token_id: int) -> dict[str, Any]:
        """Get on-chain agent identity by token ID."""
        return await self._client.request("GET", f"/api/v1/blockchain/agents/{token_id}")

