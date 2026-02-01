"""Analytics API resource for the Aragora client."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

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

    # ===========================================================================
    # Core Metrics (SDK namespace-style methods)
    # ===========================================================================

    def disagreements(self, period: str | None = None) -> Any:
        """Get disagreement analytics."""
        params: dict[str, Any] = {}
        if period:
            params["period"] = period
        return self._client.request("GET", "/api/v1/analytics/disagreements", params=params)

    def role_rotation(self, period: str | None = None) -> Any:
        """Get role rotation analytics."""
        params: dict[str, Any] = {}
        if period:
            params["period"] = period
        return self._client.request("GET", "/api/v1/analytics/role-rotation", params=params)

    def early_stops(self, period: str | None = None) -> Any:
        """Get early stop analytics."""
        params: dict[str, Any] = {}
        if period:
            params["period"] = period
        return self._client.request("GET", "/api/v1/analytics/early-stops", params=params)

    def consensus_quality(self, period: str | None = None) -> Any:
        """Get consensus quality analytics."""
        params: dict[str, Any] = {}
        if period:
            params["period"] = period
        return self._client.request("GET", "/api/v1/analytics/consensus-quality", params=params)

    def ranking_stats(self) -> Any:
        """Get ranking statistics."""
        return self._client.request("GET", "/api/v1/analytics/ranking")

    def memory_stats(self) -> Any:
        """Get memory system statistics."""
        return self._client.request("GET", "/api/v1/analytics/memory")

    # ===========================================================================
    # Dashboard Overview
    # ===========================================================================

    def get_summary(
        self,
        workspace_id: str | None = None,
        time_range: str | None = None,
    ) -> Any:
        """Get dashboard summary."""
        params: dict[str, Any] = {}
        if workspace_id:
            params["workspace_id"] = workspace_id
        if time_range:
            params["time_range"] = time_range
        return self._client.request("GET", "/api/analytics/summary", params=params)

    def get_risk_heatmap(
        self,
        workspace_id: str | None = None,
        time_range: str | None = None,
    ) -> Any:
        """Get risk heatmap data."""
        params: dict[str, Any] = {}
        if workspace_id:
            params["workspace_id"] = workspace_id
        if time_range:
            params["time_range"] = time_range
        return self._client.request("GET", "/api/analytics/heatmap", params=params)

    # ===========================================================================
    # Debate Analytics
    # ===========================================================================

    def debates_overview(self) -> Any:
        """Get debates overview metrics."""
        return self._client.request("GET", "/api/analytics/debates/overview")

    def debate_trends(
        self,
        time_range: str | None = None,
        granularity: str | None = None,
    ) -> Any:
        """Get debate trends over time."""
        params: dict[str, Any] = {}
        if time_range:
            params["time_range"] = time_range
        if granularity:
            params["granularity"] = granularity
        return self._client.request("GET", "/api/analytics/debates/trends", params=params)

    def debate_topics(self, limit: int | None = None) -> Any:
        """Get topic distribution."""
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        return self._client.request("GET", "/api/analytics/debates/topics", params=params)

    def debate_outcomes(self) -> Any:
        """Get debate outcome distribution."""
        return self._client.request("GET", "/api/analytics/debates/outcomes", params={})

    # ===========================================================================
    # Agent Analytics
    # ===========================================================================

    def agent_leaderboard(self, limit: int | None = None, domain: str | None = None) -> Any:
        """Get agent leaderboard."""
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if domain:
            params["domain"] = domain
        return self._client.request("GET", "/api/analytics/agents/leaderboard", params=params)

    def agent_performance(self, agent_id: str) -> Any:
        """Get individual agent performance."""
        return self._client.request(
            "GET", f"/api/analytics/agents/{agent_id}/performance", params={}
        )

    def compare_agents(self, agents: list[str]) -> Any:
        """Compare multiple agents."""
        params = {"agents": ",".join(agents)}
        return self._client.request("GET", "/api/analytics/agents/comparison", params=params)

    def calibration_stats(self, agent: str | None = None) -> Any:
        """Get calibration statistics."""
        params: dict[str, Any] = {}
        if agent:
            params["agent"] = agent
        return self._client.request("GET", "/api/analytics/calibration", params=params)

    # ===========================================================================
    # Usage & Cost Analytics
    # ===========================================================================

    def token_usage(self, time_range: str | None = None) -> Any:
        """Get token usage data."""
        params: dict[str, Any] = {}
        if time_range:
            params["time_range"] = time_range
        return self._client.request("GET", "/api/analytics/usage/tokens", params=params)

    def cost_breakdown(self, org_id: str | None = None) -> Any:
        """Get cost breakdown by provider."""
        params: dict[str, Any] = {}
        if org_id:
            params["org_id"] = org_id
        return self._client.request("GET", "/api/analytics/usage/costs", params=params)

    def active_users(self, time_range: str | None = None) -> Any:
        """Get active user counts."""
        params: dict[str, Any] = {}
        if time_range:
            params["time_range"] = time_range
        return self._client.request("GET", "/api/analytics/usage/active_users", params=params)

    # ===========================================================================
    # Flip Detection Analytics
    # ===========================================================================

    def flip_summary(self) -> Any:
        """Get flip detection summary."""
        return self._client.request("GET", "/api/analytics/flips/summary")

    def recent_flips(
        self,
        limit: int | None = None,
        agent: str | None = None,
        flip_type: str | None = None,
    ) -> Any:
        """Get recent flips."""
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if agent:
            params["agent"] = agent
        if flip_type:
            params["type"] = flip_type
        return self._client.request("GET", "/api/analytics/flips/recent", params=params)

    def agent_consistency(self, agents: list[str]) -> Any:
        """Get agent consistency scores."""
        params = {"agents": ",".join(agents)}
        return self._client.request("GET", "/api/analytics/flips/consistency", params=params)

    # ===========================================================================
    # Deliberation Analytics
    # ===========================================================================

    def deliberation_summary(self, days: int | None = None) -> Any:
        """Get deliberation summary."""
        params: dict[str, Any] = {}
        if days is not None:
            params["days"] = days
        return self._client.request("GET", "/api/analytics/deliberations", params=params)

    def consensus_rates(self, org_id: str | None = None, days: int | None = None) -> Any:
        """Get consensus rates by agent team."""
        params: dict[str, Any] = {}
        if org_id:
            params["org_id"] = org_id
        if days is not None:
            params["days"] = days
        return self._client.request("GET", "/api/analytics/deliberations/consensus", params=params)

    def get_disagreements(self, period: str | None = None) -> DisagreementAnalytics:
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

    async def get_disagreements_async(self, period: str | None = None) -> DisagreementAnalytics:
        """Async version of get_disagreements()."""
        params: dict[str, Any] = {}
        if period:
            params["period"] = period
        response = await self._client._get_async("/api/v1/analytics/disagreements", params=params)
        return DisagreementAnalytics(**response)

    def get_role_rotation(self, period: str | None = None) -> RoleRotationAnalytics:
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

    async def get_role_rotation_async(self, period: str | None = None) -> RoleRotationAnalytics:
        """Async version of get_role_rotation()."""
        params: dict[str, Any] = {}
        if period:
            params["period"] = period
        response = await self._client._get_async("/api/v1/analytics/role-rotation", params=params)
        return RoleRotationAnalytics(**response)

    def get_early_stops(self, period: str | None = None) -> EarlyStopAnalytics:
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

    async def get_early_stops_async(self, period: str | None = None) -> EarlyStopAnalytics:
        """Async version of get_early_stops()."""
        params: dict[str, Any] = {}
        if period:
            params["period"] = period
        response = await self._client._get_async("/api/v1/analytics/early-stops", params=params)
        return EarlyStopAnalytics(**response)

    def get_consensus_quality(self, period: str | None = None) -> ConsensusQualityAnalytics:
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
        self, period: str | None = None
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
