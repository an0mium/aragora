"""
SLO (Service Level Objective) Namespace API

Provides methods for monitoring SLO compliance, error budgets, violations, and alerts:
- Get overall SLO status
- Get individual SLO details (availability, latency, debate-success)
- Monitor error budget consumption
- List and track violations
- Check compliance status
- View active alerts
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class SLOAPI:
    """
    Synchronous SLO (Service Level Objective) API.

    Provides methods for monitoring service level objectives including
    compliance status, error budgets, violations, and alerts.

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> status = client.slo.get_status()
        >>> print(f"Overall status: {status['status']}")
        >>> budget = client.slo.get_error_budget()
        >>> print(f"Error budget remaining: {budget['remaining_percent']}%")
        >>> violations = client.slo.get_violations()
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    # =========================================================================
    # Overall Status
    # =========================================================================

    def get_status(self) -> dict[str, Any]:
        """
        Get overall SLO compliance status.

        Returns status of all SLOs including compliance percentages,
        summary statistics, and active alerts.

        Returns:
            Dict with overall SLO status including:
            - status: Overall health ('healthy', 'degraded', 'critical')
            - timestamp: Status timestamp
            - slos: Individual SLO statuses
            - alerts: Active alerts
            - summary: Compliance summary statistics
        """
        return self._client.request("GET", "/api/v2/slo/status")

    # =========================================================================
    # Individual SLOs
    # =========================================================================

    def get_availability(self) -> dict[str, Any]:
        """
        Get availability SLO status.

        Returns:
            Dict with availability SLO metrics including current value,
            target, and compliance status.
        """
        return self._client.request("GET", "/api/slos/availability")

    def get_latency(self) -> dict[str, Any]:
        """
        Get latency SLO status.

        Returns:
            Dict with latency SLO metrics including P50, P95, P99 values.
        """
        return self._client.request("GET", "/api/slos/latency")

    def get_debate_success(self) -> dict[str, Any]:
        """
        Get debate success rate SLO status.

        Returns:
            Dict with debate success SLO metrics.
        """
        return self._client.request("GET", "/api/slos/debate-success")

    def get_slo(self, slo_name: str) -> dict[str, Any]:
        """
        Get details for a specific named SLO.

        Args:
            slo_name: SLO name (e.g., 'availability', 'latency', 'debate_success').

        Returns:
            Dict with individual SLO details including current value and target.
        """
        return self._client.request("GET", f"/api/slos/{slo_name}")

    # =========================================================================
    # Error Budget
    # =========================================================================

    def get_error_budget(self) -> dict[str, Any]:
        """
        Get error budget timeline across all SLOs.

        Returns:
            Dict with error budget data including:
            - remaining_percent: Percentage of error budget remaining
            - consumed_percent: Percentage of error budget consumed
            - timeline: Error budget consumption over time
        """
        return self._client.request("GET", "/api/slos/error-budget")

    # =========================================================================
    # Violations
    # =========================================================================

    def get_violations(self) -> dict[str, Any]:
        """
        Get recent SLO violations.

        Returns:
            Dict with recent violations including timestamps,
            affected SLOs, and severity.
        """
        return self._client.request("GET", "/api/slos/violations")

    # =========================================================================
    # Targets
    # =========================================================================

    def get_targets(self) -> dict[str, Any]:
        """
        Get configured SLO targets.

        Returns:
            Dict with configured target values for each SLO including
            thresholds, time windows, and alerting rules.
        """
        return self._client.request("GET", "/api/slos/targets")


class AsyncSLOAPI:
    """
    Asynchronous SLO (Service Level Objective) API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     status = await client.slo.get_status()
        ...     budget = await client.slo.get_error_budget()
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    # =========================================================================
    # Overall Status
    # =========================================================================

    async def get_status(self) -> dict[str, Any]:
        """Get overall SLO compliance status."""
        return await self._client.request("GET", "/api/v2/slo/status")

    # =========================================================================
    # Individual SLOs
    # =========================================================================

    async def get_availability(self) -> dict[str, Any]:
        """Get availability SLO status."""
        return await self._client.request("GET", "/api/slos/availability")

    async def get_latency(self) -> dict[str, Any]:
        """Get latency SLO status."""
        return await self._client.request("GET", "/api/slos/latency")

    async def get_debate_success(self) -> dict[str, Any]:
        """Get debate success rate SLO status."""
        return await self._client.request("GET", "/api/slos/debate-success")

    async def get_slo(self, slo_name: str) -> dict[str, Any]:
        """Get details for a specific named SLO."""
        return await self._client.request("GET", f"/api/slos/{slo_name}")

    # =========================================================================
    # Error Budget
    # =========================================================================

    async def get_error_budget(self) -> dict[str, Any]:
        """Get error budget timeline across all SLOs."""
        return await self._client.request("GET", "/api/slos/error-budget")

    # =========================================================================
    # Violations
    # =========================================================================

    async def get_violations(self) -> dict[str, Any]:
        """Get recent SLO violations."""
        return await self._client.request("GET", "/api/slos/violations")

    # =========================================================================
    # Targets
    # =========================================================================

    async def get_targets(self) -> dict[str, Any]:
        """Get configured SLO targets."""
        return await self._client.request("GET", "/api/slos/targets")
