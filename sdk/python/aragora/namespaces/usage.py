"""
Usage Namespace API

Provides methods for usage tracking and SME dashboard:
- Usage summary and breakdown
- ROI analysis
- Budget tracking
- Forecasting
- Benchmarks
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

BenchmarkType = Literal["sme", "enterprise", "tech_startup", "consulting"]
ExportFormat = Literal["csv", "json"]


class UsageAPI:
    """
    Synchronous Usage API for SME dashboard.

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai", api_key="your-key")
        >>> summary = client.usage.get_summary()
        >>> print(f"Total debates: {summary['total_debates']}")
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    def get_summary(
        self,
        period: str = "month",
        organization_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Get unified usage metrics summary.

        Args:
            period: Time period (day, week, month, quarter, year)
            organization_id: Optional organization filter

        Returns:
            Usage summary with debates, tokens, costs, etc.
        """
        params: dict[str, Any] = {"period": period}
        if organization_id:
            params["organization_id"] = organization_id
        return self._client.request("GET", "/api/v1/usage/summary", params=params)

    def get_breakdown(
        self,
        group_by: str = "agent",
        period: str = "month",
    ) -> dict[str, Any]:
        """
        Get usage breakdown by dimension.

        Args:
            group_by: Dimension to group by (agent, model, day, debate_type)
            period: Time period

        Returns:
            Breakdown data by the specified dimension
        """
        return self._client.request(
            "GET",
            "/api/v1/usage/breakdown",
            params={"group_by": group_by, "period": period},
        )

    def get_roi(
        self,
        benchmark_type: BenchmarkType = "sme",
    ) -> dict[str, Any]:
        """
        Get ROI analysis with industry benchmarks.

        Args:
            benchmark_type: Benchmark to compare against

        Returns:
            ROI metrics with time saved, cost analysis, etc.
        """
        return self._client.request(
            "GET",
            "/api/v1/usage/roi",
            params={"benchmark_type": benchmark_type},
        )

    def get_budget_status(self) -> dict[str, Any]:
        """
        Get current budget utilization status.

        Returns:
            Budget status with limits, usage, and remaining
        """
        return self._client.request("GET", "/api/v1/usage/budget-status")

    def get_forecast(
        self,
        days: int = 30,
    ) -> dict[str, Any]:
        """
        Get usage forecast.

        Args:
            days: Number of days to forecast

        Returns:
            Projected usage and costs
        """
        return self._client.request(
            "GET",
            "/api/v1/usage/forecast",
            params={"days": days},
        )

    def get_benchmarks(
        self,
        benchmark_type: BenchmarkType = "sme",
    ) -> dict[str, Any]:
        """
        Get industry benchmarks for comparison.

        Args:
            benchmark_type: Industry benchmark type

        Returns:
            Benchmark data for comparison
        """
        return self._client.request(
            "GET",
            "/api/v1/usage/benchmarks",
            params={"type": benchmark_type},
        )

    def export(
        self,
        format: ExportFormat = "csv",
        period: str = "month",
    ) -> dict[str, Any]:
        """
        Export usage data.

        Args:
            format: Export format (csv or json)
            period: Time period to export

        Returns:
            Export data or download URL
        """
        return self._client.request(
            "GET",
            "/api/v1/usage/export",
            params={"format": format, "period": period},
        )


class AsyncUsageAPI:
    """
    Asynchronous Usage API for SME dashboard.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     summary = await client.usage.get_summary()
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def get_summary(
        self,
        period: str = "month",
        organization_id: str | None = None,
    ) -> dict[str, Any]:
        """Get unified usage metrics summary."""
        params: dict[str, Any] = {"period": period}
        if organization_id:
            params["organization_id"] = organization_id
        return await self._client.request("GET", "/api/v1/usage/summary", params=params)

    async def get_breakdown(
        self,
        group_by: str = "agent",
        period: str = "month",
    ) -> dict[str, Any]:
        """Get usage breakdown by dimension."""
        return await self._client.request(
            "GET",
            "/api/v1/usage/breakdown",
            params={"group_by": group_by, "period": period},
        )

    async def get_roi(
        self,
        benchmark_type: BenchmarkType = "sme",
    ) -> dict[str, Any]:
        """Get ROI analysis with industry benchmarks."""
        return await self._client.request(
            "GET",
            "/api/v1/usage/roi",
            params={"benchmark_type": benchmark_type},
        )

    async def get_budget_status(self) -> dict[str, Any]:
        """Get current budget utilization status."""
        return await self._client.request("GET", "/api/v1/usage/budget-status")

    async def get_forecast(
        self,
        days: int = 30,
    ) -> dict[str, Any]:
        """Get usage forecast."""
        return await self._client.request(
            "GET",
            "/api/v1/usage/forecast",
            params={"days": days},
        )

    async def get_benchmarks(
        self,
        benchmark_type: BenchmarkType = "sme",
    ) -> dict[str, Any]:
        """Get industry benchmarks for comparison."""
        return await self._client.request(
            "GET",
            "/api/v1/usage/benchmarks",
            params={"type": benchmark_type},
        )

    async def export(
        self,
        format: ExportFormat = "csv",
        period: str = "month",
    ) -> dict[str, Any]:
        """Export usage data."""
        return await self._client.request(
            "GET",
            "/api/v1/usage/export",
            params={"format": format, "period": period},
        )
