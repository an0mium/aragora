"""
Quotas Namespace API

Provides methods for quota management:
- View current usage and limits
- Get quota details by resource type
- Request quota increases
- Monitor quota consumption history
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class QuotasAPI:
    """
    Synchronous Quotas API.

    Provides methods for viewing and managing resource quotas including
    API rate limits, storage quotas, and usage limits.

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> quotas = client.quotas.list()
        >>> print(f"Debates used: {quotas['debates']['used']}/{quotas['debates']['limit']}")
        >>> details = client.quotas.get("debates")
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    def list(self) -> dict[str, Any]:
        """
        List all quotas and current usage.

        Returns:
            Dict with quota information for each resource type including
            current usage, limits, and percentage consumed.
        """
        return self._client.request("GET", "/api/v1/quotas")

    def get(self, resource: str) -> dict[str, Any]:
        """
        Get quota details for a specific resource type.

        Args:
            resource: Resource type (e.g., 'debates', 'agents', 'storage',
                'api_calls').

        Returns:
            Dict with detailed quota information including usage history,
            limit, and reset schedule.
        """
        return self._client.request("GET", f"/api/v1/quotas/{resource}")

    def get_usage(self, period: str | None = None) -> dict[str, Any]:
        """
        Get quota usage summary.

        Args:
            period: Time period for usage data (e.g., '1h', '24h', '7d', '30d').

        Returns:
            Dict with usage data across all resource types for the
            specified period.
        """
        params: dict[str, Any] = {}
        if period:
            params["period"] = period
        return self._client.request("GET", "/api/v1/quotas/usage", params=params or None)

    def request_increase(self, resource: str, **kwargs: Any) -> dict[str, Any]:
        """
        Request a quota increase for a resource type.

        Args:
            resource: Resource type to increase quota for.
            **kwargs: Request details including:
                - requested_limit: Desired new limit
                - justification: Reason for the increase

        Returns:
            Dict with request status and ticket ID.
        """
        data: dict[str, Any] = {"resource": resource, **kwargs}
        return self._client.request("POST", "/api/v1/quotas/request-increase", json=data)


class AsyncQuotasAPI:
    """
    Asynchronous Quotas API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     quotas = await client.quotas.list()
        ...     details = await client.quotas.get("debates")
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def list(self) -> dict[str, Any]:
        """List all quotas and current usage."""
        return await self._client.request("GET", "/api/v1/quotas")

    async def get(self, resource: str) -> dict[str, Any]:
        """Get quota details for a specific resource type."""
        return await self._client.request("GET", f"/api/v1/quotas/{resource}")

    async def get_usage(self, period: str | None = None) -> dict[str, Any]:
        """Get quota usage summary."""
        params: dict[str, Any] = {}
        if period:
            params["period"] = period
        return await self._client.request("GET", "/api/v1/quotas/usage", params=params or None)

    async def request_increase(self, resource: str, **kwargs: Any) -> dict[str, Any]:
        """Request a quota increase for a resource type."""
        data: dict[str, Any] = {"resource": resource, **kwargs}
        return await self._client.request("POST", "/api/v1/quotas/request-increase", json=data)
