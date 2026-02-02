"""
SLO (Service Level Objective) Namespace API

Provides methods for monitoring SLO compliance, error budgets, violations, and alerts:
- Get overall SLO status
- Get individual SLO details
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
        >>> budget = client.slo.get_error_budget("availability")
        >>> print(f"Error budget remaining: {budget['remaining_percent']}%")
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    # ===========================================================================
    # SLO Status
    # ===========================================================================

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
        return self._client.request("GET", "/api/v1/slo/status")

    def get_slo(self, slo_id: str) -> dict[str, Any]:
        """
        Get details for a specific SLO.

        Args:
            slo_id: SLO identifier (e.g., 'availability', 'latency', 'debate-success')

        Returns:
            Dict with SLO details including:
            - name: SLO name
            - current_percent: Current compliance percentage
            - target_percent: Target compliance percentage
            - is_meeting: Whether target is being met
            - window_start/window_end: Measurement window
            - total_requests/successful_requests/failed_requests: Request counts
        """
        return self._client.request("GET", f"/api/v1/slo/{slo_id}")

    # ===========================================================================
    # Error Budget
    # ===========================================================================

    def get_error_budget(self, slo_id: str) -> dict[str, Any]:
        """
        Get error budget information for a specific SLO.

        The error budget represents the acceptable amount of downtime or
        failures before the SLO is breached.

        Args:
            slo_id: SLO identifier

        Returns:
            Dict with error budget details including:
            - slo_name: SLO name
            - budget_percent: Total error budget
            - consumed_percent: Consumed error budget
            - remaining_percent: Remaining error budget
            - is_exhausted: Whether budget is exhausted
            - burn_rate: Current consumption rate
            - projected_exhaustion: Projected exhaustion timestamp (if applicable)
            - window_days: Budget window in days
        """
        return self._client.request("GET", f"/api/v1/slo/{slo_id}/error-budget")

    # ===========================================================================
    # Violations
    # ===========================================================================

    def list_violations(
        self,
        slo_id: str,
        limit: int | None = None,
        since: str | None = None,
    ) -> dict[str, Any]:
        """
        List violations for a specific SLO.

        Args:
            slo_id: SLO identifier
            limit: Maximum number of violations to return (optional)
            since: ISO 8601 timestamp to filter violations since (optional)

        Returns:
            Dict with violations list, each containing:
            - slo_name: SLO name
            - timestamp: Violation timestamp
            - actual_percent: Actual compliance percentage
            - target_percent: Target compliance percentage
            - duration_seconds: Violation duration
            - severity: Violation severity ('warning', 'critical')
            - resolved: Whether violation is resolved
            - resolved_at: Resolution timestamp (if resolved)
        """
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if since is not None:
            params["since"] = since
        return self._client.request("GET", f"/api/v1/slo/{slo_id}/violations", params=params)

    # ===========================================================================
    # Compliance
    # ===========================================================================

    def is_compliant(self, slo_id: str) -> dict[str, Any]:
        """
        Check if a specific SLO is currently compliant.

        Args:
            slo_id: SLO identifier

        Returns:
            Dict with compliance status including:
            - compliant: Boolean indicating if SLO is meeting target
            - slo_name: SLO name
            - current_percent: Current compliance percentage
            - target_percent: Target compliance percentage
        """
        return self._client.request("GET", f"/api/v1/slo/{slo_id}/compliant")

    # ===========================================================================
    # Alerts
    # ===========================================================================

    def get_alerts(
        self,
        slo_id: str,
        active_only: bool | None = None,
    ) -> dict[str, Any]:
        """
        Get alerts for a specific SLO.

        Args:
            slo_id: SLO identifier
            active_only: If True, only return active (unacknowledged) alerts

        Returns:
            Dict with alerts list, each containing:
            - slo_name: SLO name
            - severity: Alert severity ('warning', 'critical')
            - message: Alert message
            - triggered_at: Alert trigger timestamp
            - acknowledged: Whether alert has been acknowledged
        """
        params: dict[str, Any] = {}
        if active_only is not None:
            params["active_only"] = active_only
        return self._client.request("GET", f"/api/v1/slo/{slo_id}/alerts", params=params)


class AsyncSLOAPI:
    """
    Asynchronous SLO (Service Level Objective) API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     status = await client.slo.get_status()
        ...     print(f"Overall status: {status['status']}")
        ...     budget = await client.slo.get_error_budget("availability")
        ...     print(f"Error budget remaining: {budget['remaining_percent']}%")
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    # ===========================================================================
    # SLO Status
    # ===========================================================================

    async def get_status(self) -> dict[str, Any]:
        """
        Get overall SLO compliance status.

        Returns status of all SLOs including compliance percentages,
        summary statistics, and active alerts.
        """
        return await self._client.request("GET", "/api/v1/slo/status")

    async def get_slo(self, slo_id: str) -> dict[str, Any]:
        """
        Get details for a specific SLO.

        Args:
            slo_id: SLO identifier (e.g., 'availability', 'latency', 'debate-success')
        """
        return await self._client.request("GET", f"/api/v1/slo/{slo_id}")

    # ===========================================================================
    # Error Budget
    # ===========================================================================

    async def get_error_budget(self, slo_id: str) -> dict[str, Any]:
        """
        Get error budget information for a specific SLO.

        Args:
            slo_id: SLO identifier
        """
        return await self._client.request("GET", f"/api/v1/slo/{slo_id}/error-budget")

    # ===========================================================================
    # Violations
    # ===========================================================================

    async def list_violations(
        self,
        slo_id: str,
        limit: int | None = None,
        since: str | None = None,
    ) -> dict[str, Any]:
        """
        List violations for a specific SLO.

        Args:
            slo_id: SLO identifier
            limit: Maximum number of violations to return (optional)
            since: ISO 8601 timestamp to filter violations since (optional)
        """
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if since is not None:
            params["since"] = since
        return await self._client.request("GET", f"/api/v1/slo/{slo_id}/violations", params=params)

    # ===========================================================================
    # Compliance
    # ===========================================================================

    async def is_compliant(self, slo_id: str) -> dict[str, Any]:
        """
        Check if a specific SLO is currently compliant.

        Args:
            slo_id: SLO identifier
        """
        return await self._client.request("GET", f"/api/v1/slo/{slo_id}/compliant")

    # ===========================================================================
    # Alerts
    # ===========================================================================

    async def get_alerts(
        self,
        slo_id: str,
        active_only: bool | None = None,
    ) -> dict[str, Any]:
        """
        Get alerts for a specific SLO.

        Args:
            slo_id: SLO identifier
            active_only: If True, only return active (unacknowledged) alerts
        """
        params: dict[str, Any] = {}
        if active_only is not None:
            params["active_only"] = active_only
        return await self._client.request("GET", f"/api/v1/slo/{slo_id}/alerts", params=params)
