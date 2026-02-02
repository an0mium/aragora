"""
Usage Metering Namespace API

Provides usage tracking and billing information:
- Usage summaries and breakdowns
- Usage limits and quotas
- Export usage data

Features:
- Track token and API usage
- Monitor billing limits
- Export reports
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


UsagePeriod = Literal["hour", "day", "week", "month", "quarter", "year"]
BillingTier = Literal["free", "starter", "professional", "enterprise", "enterprise_plus"]
UsageExportFormat = Literal["csv", "json"]


class UsageMeteringAPI:
    """
    Synchronous Usage Metering API.

    Provides methods for tracking and managing usage:
    - Get usage summaries and breakdowns
    - Check usage limits and quotas
    - Export usage data

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai", api_key="...")
        >>> usage = client.usage_metering.get_usage("month")
        >>> quotas = client.usage_metering.get_quotas()
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    def get_usage(self, period: UsagePeriod = "month") -> dict[str, Any]:
        """
        Get usage summary for a billing period.

        Args:
            period: Time period (hour, day, week, month, quarter, year)

        Returns:
            Dict with usage summary including:
            - period_start/period_end
            - tokens (input/output/total/cost)
            - counts (debates/api_calls)
            - by_model/by_provider breakdowns
            - limits and usage_percent
        """
        return self._client.request("GET", "/api/v1/billing/usage", params={"period": period})

    def get_breakdown(
        self,
        start: str | None = None,
        end: str | None = None,
    ) -> dict[str, Any]:
        """
        Get detailed usage breakdown.

        Args:
            start: Start date (ISO 8601)
            end: End date (ISO 8601)

        Returns:
            Dict with breakdown including:
            - totals (cost/tokens/debates/api_calls)
            - by_model: Model-level breakdown
            - by_provider: Provider-level breakdown
            - by_day: Daily breakdown
            - by_user: User-level breakdown
        """
        params: dict[str, Any] = {}
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        return self._client.request(
            "GET", "/api/v1/billing/usage/breakdown", params=params if params else None
        )

    def get_limits(self) -> dict[str, Any]:
        """
        Get current usage limits.

        Returns:
            Dict with:
            - org_id: Organization ID
            - tier: Billing tier
            - limits: Token/debate/API call limits
            - used: Current usage
            - percent: Usage percentage
            - exceeded: Whether limits exceeded
        """
        return self._client.request("GET", "/api/v1/billing/limits")

    def get_quotas(self) -> dict[str, Any]:
        """
        Get quota status for all resources.

        Returns:
            Dict with quotas for:
            - debates
            - api_requests
            - tokens
            - storage_bytes
            - knowledge_bytes
        """
        return self._client.request("GET", "/api/v1/quotas")

    def export_usage(
        self,
        start: str | None = None,
        end: str | None = None,
        format: UsageExportFormat = "json",
    ) -> dict[str, Any] | str:
        """
        Export usage data as CSV or JSON.

        Args:
            start: Start date (ISO 8601)
            end: End date (ISO 8601)
            format: Export format (csv or json)

        Returns:
            Usage data in requested format
        """
        params: dict[str, Any] = {"format": format}
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        return self._client.request("GET", "/api/v1/billing/usage/export", params=params)


class AsyncUsageMeteringAPI:
    """
    Asynchronous Usage Metering API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     usage = await client.usage_metering.get_usage("month")
        ...     breakdown = await client.usage_metering.get_breakdown()
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def get_usage(self, period: UsagePeriod = "month") -> dict[str, Any]:
        """Get usage summary for a billing period."""
        return await self._client.request("GET", "/api/v1/billing/usage", params={"period": period})

    async def get_breakdown(
        self,
        start: str | None = None,
        end: str | None = None,
    ) -> dict[str, Any]:
        """Get detailed usage breakdown."""
        params: dict[str, Any] = {}
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        return await self._client.request(
            "GET", "/api/v1/billing/usage/breakdown", params=params if params else None
        )

    async def get_limits(self) -> dict[str, Any]:
        """Get current usage limits."""
        return await self._client.request("GET", "/api/v1/billing/limits")

    async def get_quotas(self) -> dict[str, Any]:
        """Get quota status for all resources."""
        return await self._client.request("GET", "/api/v1/quotas")

    async def export_usage(
        self,
        start: str | None = None,
        end: str | None = None,
        format: UsageExportFormat = "json",
    ) -> dict[str, Any] | str:
        """Export usage data as CSV or JSON."""
        params: dict[str, Any] = {"format": format}
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        return await self._client.request("GET", "/api/v1/billing/usage/export", params=params)
