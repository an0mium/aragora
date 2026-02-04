"""
Selection Namespace API

Provides methods for agent selection:
- Selection criteria management
- Agent matching
- Performance-based selection
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class SelectionAPI:
    """Synchronous Selection API."""

    def __init__(self, client: AragoraClient):
        self._client = client

    def select(self, task: str, count: int = 3, strategy: str = "balanced") -> dict[str, Any]:
        """Select agents for a task."""
        return self._client.request(
            "POST",
            "/api/v1/selection/select",
            json={
                "task": task,
                "count": count,
                "strategy": strategy,
            },
        )

    def get_strategies(self) -> dict[str, Any]:
        """List available selection strategies."""
        return self._client.request("GET", "/api/v1/selection/strategies")

    def get_strategy(self, strategy_name: str) -> dict[str, Any]:
        """Get strategy details."""
        return self._client.request("GET", f"/api/v1/selection/strategies/{strategy_name}")

    def create_criteria(self, name: str, rules: list[dict[str, Any]]) -> dict[str, Any]:
        """Create selection criteria."""
        return self._client.request(
            "POST",
            "/api/v1/selection/criteria",
            json={
                "name": name,
                "rules": rules,
            },
        )

    def list_criteria(self, limit: int = 20) -> dict[str, Any]:
        """List selection criteria."""
        return self._client.request("GET", "/api/v1/selection/criteria", params={"limit": limit})

    def get_performance_rankings(self, domain: str | None = None) -> dict[str, Any]:
        """Get agent performance rankings."""
        params: dict[str, Any] = {}
        if domain:
            params["domain"] = domain
        return self._client.request("GET", "/api/v1/selection/rankings", params=params)

    def record_feedback(
        self, selection_id: str, agent_id: str, score: float, notes: str | None = None
    ) -> dict[str, Any]:
        """Record selection feedback."""
        data: dict[str, Any] = {"agent_id": agent_id, "score": score}
        if notes:
            data["notes"] = notes
        return self._client.request("POST", f"/api/v1/selection/{selection_id}/feedback", json=data)


class AsyncSelectionAPI:
    """Asynchronous Selection API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def select(self, task: str, count: int = 3, strategy: str = "balanced") -> dict[str, Any]:
        """Select agents for a task."""
        return await self._client.request(
            "POST",
            "/api/v1/selection/select",
            json={
                "task": task,
                "count": count,
                "strategy": strategy,
            },
        )

    async def get_strategies(self) -> dict[str, Any]:
        """List available selection strategies."""
        return await self._client.request("GET", "/api/v1/selection/strategies")

    async def get_strategy(self, strategy_name: str) -> dict[str, Any]:
        """Get strategy details."""
        return await self._client.request("GET", f"/api/v1/selection/strategies/{strategy_name}")

    async def create_criteria(self, name: str, rules: list[dict[str, Any]]) -> dict[str, Any]:
        """Create selection criteria."""
        return await self._client.request(
            "POST",
            "/api/v1/selection/criteria",
            json={
                "name": name,
                "rules": rules,
            },
        )

    async def list_criteria(self, limit: int = 20) -> dict[str, Any]:
        """List selection criteria."""
        return await self._client.request(
            "GET", "/api/v1/selection/criteria", params={"limit": limit}
        )

    async def get_performance_rankings(self, domain: str | None = None) -> dict[str, Any]:
        """Get agent performance rankings."""
        params: dict[str, Any] = {}
        if domain:
            params["domain"] = domain
        return await self._client.request("GET", "/api/v1/selection/rankings", params=params)

    async def record_feedback(
        self, selection_id: str, agent_id: str, score: float, notes: str | None = None
    ) -> dict[str, Any]:
        """Record selection feedback."""
        data: dict[str, Any] = {"agent_id": agent_id, "score": score}
        if notes:
            data["notes"] = notes
        return await self._client.request(
            "POST", f"/api/v1/selection/{selection_id}/feedback", json=data
        )
