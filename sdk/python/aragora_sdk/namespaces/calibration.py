"""
Calibration Namespace API

Provides access to agent calibration and leaderboard data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class CalibrationAPI:
    """Synchronous Calibration API for agent calibration data."""

    def __init__(self, client: AragoraClient):
        self._client = client

    def get_leaderboard(self) -> dict[str, Any]:
        """Get calibration leaderboard.

        Returns:
            Leaderboard with agent calibration scores and rankings.
        """
        return self._client.request("GET", "/api/v1/calibration/leaderboard")

    def get_visualization(self) -> dict[str, Any]:
        """Get calibration visualization data.

        Returns:
            Visualization data for calibration metrics.
        """
        return self._client.request("GET", "/api/v1/calibration/visualization")


class AsyncCalibrationAPI:
    """Asynchronous Calibration API for agent calibration data."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def get_leaderboard(self) -> dict[str, Any]:
        """Get calibration leaderboard.

        Returns:
            Leaderboard with agent calibration scores and rankings.
        """
        return await self._client.request("GET", "/api/v1/calibration/leaderboard")

    async def get_visualization(self) -> dict[str, Any]:
        """Get calibration visualization data.

        Returns:
            Visualization data for calibration metrics.
        """
        return await self._client.request("GET", "/api/v1/calibration/visualization")
