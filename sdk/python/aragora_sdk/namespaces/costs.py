"""
Costs Namespace API

Provides methods for cost tracking and management:
- View usage costs
- Manage budgets
- Generate cost reports
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class CostsAPI:
    """Synchronous Costs API."""

    def __init__(self, client: AragoraClient):
        self._client = client

    def get_summary(
        self, start_date: str | None = None, end_date: str | None = None
    ) -> dict[str, Any]:
        """Get cost summary."""
        params: dict[str, Any] = {}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._client.request("GET", "/api/v1/costs/summary", params=params)

    def get_breakdown(self, group_by: str = "model", limit: int = 20) -> dict[str, Any]:
        """Get cost breakdown."""
        return self._client.request(
            "GET", "/api/v1/costs/breakdown", params={"group_by": group_by, "limit": limit}
        )

    def get_daily(self, days: int = 30) -> dict[str, Any]:
        """Get daily cost data."""
        return self._client.request("GET", "/api/v1/costs/daily", params={"days": days})

    def get_by_model(self, model: str | None = None) -> dict[str, Any]:
        """Get costs by model."""
        params: dict[str, Any] = {}
        if model:
            params["model"] = model
        return self._client.request("GET", "/api/v1/costs/by-model", params=params)

    def get_by_debate(self, debate_id: str) -> dict[str, Any]:
        """Get costs for a specific debate."""
        return self._client.request("GET", f"/api/v1/costs/debates/{debate_id}")

    def export(
        self, format: str = "csv", start_date: str | None = None, end_date: str | None = None
    ) -> dict[str, Any]:
        """Export cost data."""
        params: dict[str, Any] = {"format": format}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._client.request("GET", "/api/v1/costs/export", params=params)


class AsyncCostsAPI:
    """Asynchronous Costs API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def get_summary(
        self, start_date: str | None = None, end_date: str | None = None
    ) -> dict[str, Any]:
        """Get cost summary."""
        params: dict[str, Any] = {}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return await self._client.request("GET", "/api/v1/costs/summary", params=params)

    async def get_breakdown(self, group_by: str = "model", limit: int = 20) -> dict[str, Any]:
        """Get cost breakdown."""
        return await self._client.request(
            "GET", "/api/v1/costs/breakdown", params={"group_by": group_by, "limit": limit}
        )

    async def get_daily(self, days: int = 30) -> dict[str, Any]:
        """Get daily cost data."""
        return await self._client.request("GET", "/api/v1/costs/daily", params={"days": days})

    async def get_by_model(self, model: str | None = None) -> dict[str, Any]:
        """Get costs by model."""
        params: dict[str, Any] = {}
        if model:
            params["model"] = model
        return await self._client.request("GET", "/api/v1/costs/by-model", params=params)

    async def get_by_debate(self, debate_id: str) -> dict[str, Any]:
        """Get costs for a specific debate."""
        return await self._client.request("GET", f"/api/v1/costs/debates/{debate_id}")

    async def export(
        self, format: str = "csv", start_date: str | None = None, end_date: str | None = None
    ) -> dict[str, Any]:
        """Export cost data."""
        params: dict[str, Any] = {"format": format}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return await self._client.request("GET", "/api/v1/costs/export", params=params)
