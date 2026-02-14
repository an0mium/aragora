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
        return self._client.request("GET", "/api/v2/slo/status")

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
        return await self._client.request("GET", "/api/v2/slo/status")

