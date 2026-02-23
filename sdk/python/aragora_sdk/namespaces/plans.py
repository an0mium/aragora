"""
Plans Namespace API

Provides methods for plan management.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class PlansAPI:
    """Synchronous Plans API."""

    def __init__(self, client: AragoraClient):
        self._client = client

    def list(self, **params: Any) -> dict[str, Any]:
        """List all decision plans.

        Args:
            **params: Optional query parameters (status, debate_id, etc.).
        """
        return self._client.request("GET", "/api/v1/plans", params=params)

    def get(self, plan_id: str) -> dict[str, Any]:
        """Get a specific plan by ID."""
        return self._client.request("GET", f"/api/v1/plans/{plan_id}")

    def create(self, **kwargs: Any) -> dict[str, Any]:
        """Create a new decision plan.

        Args:
            **kwargs: Plan fields (debate_id, title, tasks, etc.).
        """
        return self._client.request("POST", "/api/v1/plans", json=kwargs)

    def update(self, plan_id: str, **kwargs: Any) -> dict[str, Any]:
        """Update a plan.

        Args:
            plan_id: ID of the plan to update.
            **kwargs: Plan update fields (status, tasks, etc.).
        """
        return self._client.request("PUT", f"/api/v1/plans/{plan_id}", json=kwargs)

    def approve(self, plan_id: str) -> dict[str, Any]:
        """Approve a decision plan for execution."""
        return self._client.request("POST", f"/api/v1/plans/{plan_id}/approve")

    def reject(self, plan_id: str, reason: str = "") -> dict[str, Any]:
        """Reject a decision plan.

        Args:
            plan_id: ID of the plan to reject.
            reason: Optional rejection reason.
        """
        return self._client.request(
            "POST", f"/api/v1/plans/{plan_id}/reject", json={"reason": reason}
        )

    def execute(self, plan_id: str) -> dict[str, Any]:
        """Execute an approved decision plan."""
        return self._client.request("POST", f"/api/v1/plans/{plan_id}/execute")


class AsyncPlansAPI:
    """Asynchronous Plans API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def list(self, **params: Any) -> dict[str, Any]:
        """List all decision plans."""
        return await self._client.request("GET", "/api/v1/plans", params=params)

    async def get(self, plan_id: str) -> dict[str, Any]:
        """Get a specific plan by ID."""
        return await self._client.request("GET", f"/api/v1/plans/{plan_id}")

    async def create(self, **kwargs: Any) -> dict[str, Any]:
        """Create a new decision plan."""
        return await self._client.request("POST", "/api/v1/plans", json=kwargs)

    async def update(self, plan_id: str, **kwargs: Any) -> dict[str, Any]:
        """Update a plan."""
        return await self._client.request(
            "PUT", f"/api/v1/plans/{plan_id}", json=kwargs
        )

    async def approve(self, plan_id: str) -> dict[str, Any]:
        """Approve a decision plan for execution."""
        return await self._client.request(
            "POST", f"/api/v1/plans/{plan_id}/approve"
        )

    async def reject(self, plan_id: str, reason: str = "") -> dict[str, Any]:
        """Reject a decision plan."""
        return await self._client.request(
            "POST", f"/api/v1/plans/{plan_id}/reject", json={"reason": reason}
        )

    async def execute(self, plan_id: str) -> dict[str, Any]:
        """Execute an approved decision plan."""
        return await self._client.request(
            "POST", f"/api/v1/plans/{plan_id}/execute"
        )
