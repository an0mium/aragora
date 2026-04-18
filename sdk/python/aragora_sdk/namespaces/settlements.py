"""
Settlements namespace API.

Provides SDK access to debate settlement and calibration endpoints.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class SettlementAPI:
    """Synchronous settlement API."""

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    def list(
        self,
        *,
        debate_id: str | None = None,
        domain: str | None = None,
        limit: int | None = 100,
    ) -> dict[str, Any]:
        """List pending settlements."""
        params: dict[str, Any] = {}
        if debate_id:
            params["debate_id"] = debate_id
        if domain:
            params["domain"] = domain
        if limit is not None:
            params["limit"] = limit
        return self._client.request("GET", "/api/v1/settlements", params=params or None)

    def get_history(
        self,
        *,
        debate_id: str | None = None,
        domain: str | None = None,
        limit: int | None = 100,
    ) -> dict[str, Any]:
        """Get settlement history."""
        params: dict[str, Any] = {}
        if debate_id:
            params["debate_id"] = debate_id
        if domain:
            params["domain"] = domain
        if limit is not None:
            params["limit"] = limit
        return self._client.request("GET", "/api/v1/settlements/history", params=params or None)

    def get_summary(self) -> dict[str, Any]:
        """Get settlement summary statistics."""
        return self._client.request("GET", "/api/v1/settlements/summary")

    def get(self, settlement_id: str) -> dict[str, Any]:
        """Get a settlement by ID."""
        return self._client.request("GET", f"/api/v1/settlements/{settlement_id}")

    def settle(
        self,
        settlement_id: str,
        *,
        outcome: str,
        evidence: str = "",
        settled_by: str = "api",
    ) -> dict[str, Any]:
        """Settle one claim outcome."""
        return self._client.request(
            "POST",
            f"/api/v1/settlements/{settlement_id}/settle",
            json={
                "outcome": outcome,
                "evidence": evidence,
                "settled_by": settled_by,
            },
        )

    def settle_batch(
        self,
        settlements: list[dict[str, Any]],
        *,
        settled_by: str = "api",
    ) -> dict[str, Any]:
        """Settle multiple claims at once."""
        return self._client.request(
            "POST",
            "/api/v1/settlements/batch",
            json={
                "settlements": settlements,
                "settled_by": settled_by,
            },
        )

    def get_agent_accuracy(self, agent_name: str) -> dict[str, Any]:
        """Get settlement accuracy stats for one agent."""
        return self._client.request(
            "GET",
            f"/api/v1/settlements/agent/{agent_name}/accuracy",
        )


class AsyncSettlementAPI:
    """Asynchronous settlement API."""

    def __init__(self, client: AragoraAsyncClient) -> None:
        self._client = client

    async def list(
        self,
        *,
        debate_id: str | None = None,
        domain: str | None = None,
        limit: int | None = 100,
    ) -> dict[str, Any]:
        """List pending settlements."""
        params: dict[str, Any] = {}
        if debate_id:
            params["debate_id"] = debate_id
        if domain:
            params["domain"] = domain
        if limit is not None:
            params["limit"] = limit
        return await self._client.request("GET", "/api/v1/settlements", params=params or None)

    async def get_history(
        self,
        *,
        debate_id: str | None = None,
        domain: str | None = None,
        limit: int | None = 100,
    ) -> dict[str, Any]:
        """Get settlement history."""
        params: dict[str, Any] = {}
        if debate_id:
            params["debate_id"] = debate_id
        if domain:
            params["domain"] = domain
        if limit is not None:
            params["limit"] = limit
        return await self._client.request(
            "GET",
            "/api/v1/settlements/history",
            params=params or None,
        )

    async def get_summary(self) -> dict[str, Any]:
        """Get settlement summary statistics."""
        return await self._client.request("GET", "/api/v1/settlements/summary")

    async def get(self, settlement_id: str) -> dict[str, Any]:
        """Get a settlement by ID."""
        return await self._client.request("GET", f"/api/v1/settlements/{settlement_id}")

    async def settle(
        self,
        settlement_id: str,
        *,
        outcome: str,
        evidence: str = "",
        settled_by: str = "api",
    ) -> dict[str, Any]:
        """Settle one claim outcome."""
        return await self._client.request(
            "POST",
            f"/api/v1/settlements/{settlement_id}/settle",
            json={
                "outcome": outcome,
                "evidence": evidence,
                "settled_by": settled_by,
            },
        )

    async def settle_batch(
        self,
        settlements: list[dict[str, Any]],
        *,
        settled_by: str = "api",
    ) -> dict[str, Any]:
        """Settle multiple claims at once."""
        return await self._client.request(
            "POST",
            "/api/v1/settlements/batch",
            json={
                "settlements": settlements,
                "settled_by": settled_by,
            },
        )

    async def get_agent_accuracy(self, agent_name: str) -> dict[str, Any]:
        """Get settlement accuracy stats for one agent."""
        return await self._client.request(
            "GET",
            f"/api/v1/settlements/agent/{agent_name}/accuracy",
        )
