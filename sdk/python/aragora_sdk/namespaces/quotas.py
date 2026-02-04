"""
Quotas Namespace API

Provides methods for quota management:
- View usage limits
- Request quota increases
- Monitor quota consumption
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class QuotasAPI:
    """Synchronous Quotas API."""

    def __init__(self, client: AragoraClient):
        self._client = client

    def list(self) -> dict[str, Any]:
        """List all quotas."""
        return self._client.request("GET", "/api/v1/quotas")

    def get(self, quota_name: str) -> dict[str, Any]:
        """Get quota by name."""
        return self._client.request("GET", f"/api/v1/quotas/{quota_name}")

    def get_usage(self, quota_name: str) -> dict[str, Any]:
        """Get quota usage."""
        return self._client.request("GET", f"/api/v1/quotas/{quota_name}/usage")

    def request_increase(
        self, quota_name: str, requested_limit: int, reason: str
    ) -> dict[str, Any]:
        """Request quota increase."""
        return self._client.request(
            "POST",
            f"/api/v1/quotas/{quota_name}/increase",
            json={
                "requested_limit": requested_limit,
                "reason": reason,
            },
        )

    def get_history(self, quota_name: str, days: int = 30) -> dict[str, Any]:
        """Get quota usage history."""
        return self._client.request(
            "GET", f"/api/v1/quotas/{quota_name}/history", params={"days": days}
        )

    def get_alerts(self) -> dict[str, Any]:
        """Get quota alerts."""
        return self._client.request("GET", "/api/v1/quotas/alerts")

    def set_alert(self, quota_name: str, threshold: float) -> dict[str, Any]:
        """Set quota alert threshold."""
        return self._client.request(
            "POST", f"/api/v1/quotas/{quota_name}/alert", json={"threshold": threshold}
        )


class AsyncQuotasAPI:
    """Asynchronous Quotas API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def list(self) -> dict[str, Any]:
        """List all quotas."""
        return await self._client.request("GET", "/api/v1/quotas")

    async def get(self, quota_name: str) -> dict[str, Any]:
        """Get quota by name."""
        return await self._client.request("GET", f"/api/v1/quotas/{quota_name}")

    async def get_usage(self, quota_name: str) -> dict[str, Any]:
        """Get quota usage."""
        return await self._client.request("GET", f"/api/v1/quotas/{quota_name}/usage")

    async def request_increase(
        self, quota_name: str, requested_limit: int, reason: str
    ) -> dict[str, Any]:
        """Request quota increase."""
        return await self._client.request(
            "POST",
            f"/api/v1/quotas/{quota_name}/increase",
            json={
                "requested_limit": requested_limit,
                "reason": reason,
            },
        )

    async def get_history(self, quota_name: str, days: int = 30) -> dict[str, Any]:
        """Get quota usage history."""
        return await self._client.request(
            "GET", f"/api/v1/quotas/{quota_name}/history", params={"days": days}
        )

    async def get_alerts(self) -> dict[str, Any]:
        """Get quota alerts."""
        return await self._client.request("GET", "/api/v1/quotas/alerts")

    async def set_alert(self, quota_name: str, threshold: float) -> dict[str, Any]:
        """Set quota alert threshold."""
        return await self._client.request(
            "POST", f"/api/v1/quotas/{quota_name}/alert", json={"threshold": threshold}
        )
