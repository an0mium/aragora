"""
Calibration Namespace API

Provides access to agent calibration data:
- Calibration leaderboard with scores and rankings
- Calibration visualization data
- Calibration curve data for individual agents
- Historical calibration trends
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class CalibrationAPI:
    """
    Synchronous Calibration API for agent calibration data.

    Calibration measures how well agents' confidence levels match
    their actual accuracy. Well-calibrated agents say they are 80%
    confident only when they are correct 80% of the time.

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> leaderboard = client.calibration.get_leaderboard()
        >>> for agent in leaderboard["agents"]:
        ...     print(f"{agent['name']}: Brier={agent['brier_score']:.3f}")
        >>> curve = client.calibration.get_curve("claude")
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    def get_leaderboard(self) -> dict[str, Any]:
        """
        Get calibration leaderboard.

        Returns:
            Dict with agent calibration scores and rankings, including
            Brier scores, reliability diagrams, and calibration quality
            assessments.
        """
        return self._client.request("GET", "/api/v1/calibration/leaderboard")

    def get_visualization(self) -> dict[str, Any]:
        """
        Get calibration visualization data.

        Returns:
            Dict with visualization data for calibration metrics,
            suitable for rendering charts and graphs.
        """
        return self._client.request("GET", "/api/v1/calibration/visualization")

    def get_curve(self, agent: str | None = None) -> dict[str, Any]:
        """
        Get calibration curve data.

        A calibration curve plots predicted probability against actual
        frequency of correct predictions across confidence bins.

        Args:
            agent: Optional agent name to filter calibration curve for.
                   If not provided, returns aggregate curve.

        Returns:
            Dict with calibration curve data including confidence bins,
            predicted probabilities, and actual frequencies.
        """
        if agent:
            return self._client.request(
                "GET", f"/api/v1/agent/{agent}/calibration-curve"
            )
        return self.get_visualization()

    def get_history(
        self,
        agent: str | None = None,
        period: str | None = None,
    ) -> dict[str, Any]:
        """
        Get calibration history over time.

        Args:
            agent: Optional agent name to filter history for.
            period: Time period (e.g., '7d', '30d', '90d').

        Returns:
            Dict with historical calibration data including trends
            and improvement/regression tracking.
        """
        if agent:
            params: dict[str, Any] = {}
            if period:
                params["period"] = period
            return self._client.request(
                "GET", f"/api/v1/agent/{agent}/calibration-summary", params=params or None
            )
        return self.get_visualization()


class AsyncCalibrationAPI:
    """
    Asynchronous Calibration API for agent calibration data.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     leaderboard = await client.calibration.get_leaderboard()
        ...     curve = await client.calibration.get_curve("claude")
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def get_leaderboard(self) -> dict[str, Any]:
        """Get calibration leaderboard."""
        return await self._client.request("GET", "/api/v1/calibration/leaderboard")

    async def get_visualization(self) -> dict[str, Any]:
        """Get calibration visualization data."""
        return await self._client.request("GET", "/api/v1/calibration/visualization")

    async def get_curve(self, agent: str | None = None) -> dict[str, Any]:
        """Get calibration curve data."""
        if agent:
            return await self._client.request(
                "GET", f"/api/v1/agent/{agent}/calibration-curve"
            )
        return await self.get_visualization()

    async def get_history(
        self,
        agent: str | None = None,
        period: str | None = None,
    ) -> dict[str, Any]:
        """Get calibration history over time."""
        if agent:
            params: dict[str, Any] = {}
            if period:
                params["period"] = period
            return await self._client.request(
                "GET", f"/api/v1/agent/{agent}/calibration-summary", params=params or None
            )
        return await self.get_visualization()
