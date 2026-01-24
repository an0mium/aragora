"""Analytics API resource for the Aragora client."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from aragora.client.client import AragoraClient

from aragora.client.models import (
    ConsensusQualityAnalytics,
    DisagreementAnalytics,
    EarlyStopAnalytics,
    MemoryStats,
    RankingStats,
    RoleRotationAnalytics,
)


class AnalyticsAPI:
    """API interface for analytics."""

    def __init__(self, client: "AragoraClient"):
        self._client = client

    def get_disagreements(self, period: Optional[str] = None) -> DisagreementAnalytics:
        """
        Get disagreement analytics.

        Args:
            period: Time period filter (e.g., "7d", "30d").

        Returns:
            DisagreementAnalytics with disagreement statistics.
        """
        params: dict[str, Any] = {}
        if period:
            params["period"] = period
        response = self._client._get("/api/v1/analytics/disagreements", params=params)
        return DisagreementAnalytics(**response)

    async def get_disagreements_async(self, period: Optional[str] = None) -> DisagreementAnalytics:
        """Async version of get_disagreements()."""
        params: dict[str, Any] = {}
        if period:
            params["period"] = period
        response = await self._client._get_async("/api/v1/analytics/disagreements", params=params)
        return DisagreementAnalytics(**response)

    def get_role_rotation(self, period: Optional[str] = None) -> RoleRotationAnalytics:
        """
        Get role rotation analytics.

        Args:
            period: Time period filter (e.g., "7d", "30d").

        Returns:
            RoleRotationAnalytics with role assignment statistics.
        """
        params: dict[str, Any] = {}
        if period:
            params["period"] = period
        response = self._client._get("/api/v1/analytics/role-rotation", params=params)
        return RoleRotationAnalytics(**response)

    async def get_role_rotation_async(self, period: Optional[str] = None) -> RoleRotationAnalytics:
        """Async version of get_role_rotation()."""
        params: dict[str, Any] = {}
        if period:
            params["period"] = period
        response = await self._client._get_async("/api/v1/analytics/role-rotation", params=params)
        return RoleRotationAnalytics(**response)

    def get_early_stops(self, period: Optional[str] = None) -> EarlyStopAnalytics:
        """
        Get early stop analytics.

        Args:
            period: Time period filter (e.g., "7d", "30d").

        Returns:
            EarlyStopAnalytics with early termination statistics.
        """
        params: dict[str, Any] = {}
        if period:
            params["period"] = period
        response = self._client._get("/api/v1/analytics/early-stops", params=params)
        return EarlyStopAnalytics(**response)

    async def get_early_stops_async(self, period: Optional[str] = None) -> EarlyStopAnalytics:
        """Async version of get_early_stops()."""
        params: dict[str, Any] = {}
        if period:
            params["period"] = period
        response = await self._client._get_async("/api/v1/analytics/early-stops", params=params)
        return EarlyStopAnalytics(**response)

    def get_consensus_quality(self, period: Optional[str] = None) -> ConsensusQualityAnalytics:
        """
        Get consensus quality analytics.

        Args:
            period: Time period filter (e.g., "7d", "30d").

        Returns:
            ConsensusQualityAnalytics with quality metrics.
        """
        params: dict[str, Any] = {}
        if period:
            params["period"] = period
        response = self._client._get("/api/v1/analytics/consensus-quality", params=params)
        return ConsensusQualityAnalytics(**response)

    async def get_consensus_quality_async(
        self, period: Optional[str] = None
    ) -> ConsensusQualityAnalytics:
        """Async version of get_consensus_quality()."""
        params: dict[str, Any] = {}
        if period:
            params["period"] = period
        response = await self._client._get_async(
            "/api/v1/analytics/consensus-quality", params=params
        )
        return ConsensusQualityAnalytics(**response)

    def get_ranking_stats(self) -> RankingStats:
        """
        Get ranking statistics.

        Returns:
            RankingStats with ELO distribution and top performers.
        """
        response = self._client._get("/api/v1/ranking/stats")
        return RankingStats(**response)

    async def get_ranking_stats_async(self) -> RankingStats:
        """Async version of get_ranking_stats()."""
        response = await self._client._get_async("/api/v1/ranking/stats")
        return RankingStats(**response)

    def get_memory_stats(self) -> MemoryStats:
        """
        Get memory system statistics.

        Returns:
            MemoryStats with storage and tier information.
        """
        response = self._client._get("/api/v1/memory/stats")
        return MemoryStats(**response)

    async def get_memory_stats_async(self) -> MemoryStats:
        """Async version of get_memory_stats()."""
        response = await self._client._get_async("/api/v1/memory/stats")
        return MemoryStats(**response)


__all__ = ["AnalyticsAPI"]
