"""Analytics API for platform metrics and insights."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aragora_client.client import AragoraClient


class AnalyticsAPI:
    """API for analytics operations."""

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    # =========================================================================
    # Core Analytics
    # =========================================================================

    async def get_disagreement_stats(
        self,
        *,
        time_range: str = "30d",
        org_id: str | None = None,
    ) -> dict[str, Any]:
        """Get disagreement statistics.

        Args:
            time_range: Time range (7d, 30d, 90d, all)
            org_id: Filter by organization

        Returns:
            Disagreement patterns and frequencies
        """
        params: dict[str, Any] = {"time_range": time_range}
        if org_id:
            params["org_id"] = org_id
        return await self._client._get("/api/v1/analytics/disagreements", params=params)

    async def get_role_rotation_stats(
        self,
        *,
        time_range: str = "30d",
        org_id: str | None = None,
    ) -> dict[str, Any]:
        """Get role rotation statistics.

        Args:
            time_range: Time range (7d, 30d, 90d, all)
            org_id: Filter by organization

        Returns:
            Role rotation patterns and effectiveness
        """
        params: dict[str, Any] = {"time_range": time_range}
        if org_id:
            params["org_id"] = org_id
        return await self._client._get("/api/v1/analytics/role-rotation", params=params)

    async def get_early_stop_stats(
        self,
        *,
        time_range: str = "30d",
        org_id: str | None = None,
    ) -> dict[str, Any]:
        """Get early stopping statistics.

        Args:
            time_range: Time range (7d, 30d, 90d, all)
            org_id: Filter by organization

        Returns:
            Early stop rates and triggers
        """
        params: dict[str, Any] = {"time_range": time_range}
        if org_id:
            params["org_id"] = org_id
        return await self._client._get("/api/v1/analytics/early-stops", params=params)

    async def get_consensus_quality_stats(
        self,
        *,
        time_range: str = "30d",
        org_id: str | None = None,
    ) -> dict[str, Any]:
        """Get consensus quality statistics.

        Args:
            time_range: Time range (7d, 30d, 90d, all)
            org_id: Filter by organization

        Returns:
            Consensus strength and stability metrics
        """
        params: dict[str, Any] = {"time_range": time_range}
        if org_id:
            params["org_id"] = org_id
        return await self._client._get(
            "/api/v1/analytics/consensus-quality", params=params
        )

    async def get_memory_analytics(
        self,
        *,
        time_range: str = "30d",
        org_id: str | None = None,
    ) -> dict[str, Any]:
        """Get memory subsystem analytics.

        Args:
            time_range: Time range (7d, 30d, 90d, all)
            org_id: Filter by organization

        Returns:
            Memory utilization and hit rates
        """
        params: dict[str, Any] = {"time_range": time_range}
        if org_id:
            params["org_id"] = org_id
        return await self._client._get("/api/v1/analytics/memory", params=params)

    async def get_cross_pollination_analytics(
        self,
        *,
        time_range: str = "30d",
        org_id: str | None = None,
    ) -> dict[str, Any]:
        """Get cross-pollination analytics.

        Args:
            time_range: Time range (7d, 30d, 90d, all)
            org_id: Filter by organization

        Returns:
            Cross-pollination rates and impact
        """
        params: dict[str, Any] = {"time_range": time_range}
        if org_id:
            params["org_id"] = org_id
        return await self._client._get(
            "/api/v1/analytics/cross-pollination", params=params
        )

    async def get_learning_efficiency(
        self,
        *,
        agent: str | None = None,
        domain: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Get learning efficiency metrics.

        Args:
            agent: Filter by agent
            domain: Filter by domain
            limit: Maximum results

        Returns:
            Learning curves and efficiency scores
        """
        params: dict[str, Any] = {"limit": limit}
        if agent:
            params["agent"] = agent
        if domain:
            params["domain"] = domain
        return await self._client._get(
            "/api/v1/analytics/learning-efficiency", params=params
        )

    async def get_voting_accuracy(
        self,
        *,
        agent: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Get voting accuracy metrics.

        Args:
            agent: Filter by agent
            limit: Maximum results

        Returns:
            Voting accuracy and calibration
        """
        params: dict[str, Any] = {"limit": limit}
        if agent:
            params["agent"] = agent
        return await self._client._get(
            "/api/v1/analytics/voting-accuracy", params=params
        )

    async def get_calibration_analytics(
        self,
        *,
        agent: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Get calibration analytics.

        Args:
            agent: Filter by agent
            limit: Maximum results

        Returns:
            Calibration scores and trends
        """
        params: dict[str, Any] = {"limit": limit}
        if agent:
            params["agent"] = agent
        return await self._client._get("/api/v1/analytics/calibration", params=params)

    # =========================================================================
    # Debate Metrics
    # =========================================================================

    async def get_debates_overview(
        self,
        *,
        time_range: str = "30d",
        org_id: str | None = None,
    ) -> dict[str, Any]:
        """Get debates overview metrics.

        Args:
            time_range: Time range (7d, 30d, 90d, all)
            org_id: Filter by organization

        Returns:
            Debate counts, completion rates, duration stats
        """
        params: dict[str, Any] = {"time_range": time_range}
        if org_id:
            params["org_id"] = org_id
        return await self._client._get(
            "/api/v1/analytics/debates/overview", params=params
        )

    async def get_debates_trends(
        self,
        *,
        time_range: str = "30d",
        granularity: str = "daily",
        org_id: str | None = None,
    ) -> dict[str, Any]:
        """Get debate trends over time.

        Args:
            time_range: Time range (7d, 30d, 90d, all)
            granularity: Data granularity (hourly, daily, weekly)
            org_id: Filter by organization

        Returns:
            Time-series debate metrics
        """
        params: dict[str, Any] = {
            "time_range": time_range,
            "granularity": granularity,
        }
        if org_id:
            params["org_id"] = org_id
        return await self._client._get(
            "/api/v1/analytics/debates/trends", params=params
        )

    async def get_debates_topics(
        self,
        *,
        time_range: str = "30d",
        limit: int = 20,
        org_id: str | None = None,
    ) -> dict[str, Any]:
        """Get top debate topics.

        Args:
            time_range: Time range (7d, 30d, 90d, all)
            limit: Maximum topics to return
            org_id: Filter by organization

        Returns:
            Topic distribution and frequencies
        """
        params: dict[str, Any] = {"time_range": time_range, "limit": limit}
        if org_id:
            params["org_id"] = org_id
        return await self._client._get(
            "/api/v1/analytics/debates/topics", params=params
        )

    async def get_debates_outcomes(
        self,
        *,
        time_range: str = "30d",
        org_id: str | None = None,
    ) -> dict[str, Any]:
        """Get debate outcome statistics.

        Args:
            time_range: Time range (7d, 30d, 90d, all)
            org_id: Filter by organization

        Returns:
            Outcome distribution and quality metrics
        """
        params: dict[str, Any] = {"time_range": time_range}
        if org_id:
            params["org_id"] = org_id
        return await self._client._get(
            "/api/v1/analytics/debates/outcomes", params=params
        )

    # =========================================================================
    # Agent Analytics
    # =========================================================================

    async def get_agents_leaderboard(
        self,
        *,
        limit: int = 20,
        domain: str | None = None,
    ) -> dict[str, Any]:
        """Get agent leaderboard.

        Args:
            limit: Maximum agents to return
            domain: Filter by domain

        Returns:
            Agent rankings and scores
        """
        params: dict[str, Any] = {"limit": limit}
        if domain:
            params["domain"] = domain
        return await self._client._get(
            "/api/v1/analytics/agents/leaderboard", params=params
        )

    async def get_agent_analytics(
        self,
        agent_id: str,
        *,
        time_range: str = "30d",
    ) -> dict[str, Any]:
        """Get analytics for a specific agent.

        Args:
            agent_id: Agent ID
            time_range: Time range (7d, 30d, 90d, all)

        Returns:
            Agent performance metrics
        """
        return await self._client._get(
            f"/api/v1/analytics/agents/{agent_id}",
            params={"time_range": time_range},
        )

    async def compare_agents(
        self,
        agents: list[str],
    ) -> dict[str, Any]:
        """Compare multiple agents.

        Args:
            agents: List of agent IDs to compare

        Returns:
            Comparative metrics for agents
        """
        return await self._client._post(
            "/api/v1/analytics/agents/compare",
            {"agents": agents},
        )

    async def get_agents_trends(
        self,
        agents: list[str],
        *,
        time_range: str = "30d",
        granularity: str = "daily",
    ) -> dict[str, Any]:
        """Get agent performance trends.

        Args:
            agents: List of agent IDs
            time_range: Time range (7d, 30d, 90d, all)
            granularity: Data granularity (hourly, daily, weekly)

        Returns:
            Time-series agent metrics
        """
        return await self._client._post(
            "/api/v1/analytics/agents/trends",
            {
                "agents": agents,
                "time_range": time_range,
                "granularity": granularity,
            },
        )

    # =========================================================================
    # Usage Analytics
    # =========================================================================

    async def get_token_usage(
        self,
        org_id: str,
        *,
        time_range: str = "30d",
        granularity: str = "daily",
    ) -> dict[str, Any]:
        """Get token usage analytics.

        Args:
            org_id: Organization ID
            time_range: Time range (7d, 30d, 90d, all)
            granularity: Data granularity (hourly, daily, weekly)

        Returns:
            Token usage by model and time
        """
        return await self._client._get(
            f"/api/v1/analytics/usage/{org_id}/tokens",
            params={"time_range": time_range, "granularity": granularity},
        )

    async def get_cost_usage(
        self,
        org_id: str,
        *,
        time_range: str = "30d",
    ) -> dict[str, Any]:
        """Get cost usage analytics.

        Args:
            org_id: Organization ID
            time_range: Time range (7d, 30d, 90d, all)

        Returns:
            Cost breakdown by model and category
        """
        return await self._client._get(
            f"/api/v1/analytics/usage/{org_id}/costs",
            params={"time_range": time_range},
        )

    async def get_active_users_analytics(
        self,
        *,
        org_id: str | None = None,
        time_range: str = "7d",
    ) -> dict[str, Any]:
        """Get active users analytics.

        Args:
            org_id: Filter by organization
            time_range: Time range (7d, 30d, 90d, all)

        Returns:
            Active user counts and trends
        """
        params: dict[str, Any] = {"time_range": time_range}
        if org_id:
            params["org_id"] = org_id
        return await self._client._get("/api/v1/analytics/active-users", params=params)

    # =========================================================================
    # Flip Detection
    # =========================================================================

    async def get_flip_summary(self) -> dict[str, Any]:
        """Get position flip summary.

        Returns:
            Flip rates and common flip patterns
        """
        return await self._client._get("/api/v1/analytics/flips/summary")

    async def get_recent_flips(
        self,
        *,
        limit: int = 20,
        agent: str | None = None,
        flip_type: str | None = None,
    ) -> dict[str, Any]:
        """Get recent position flips.

        Args:
            limit: Maximum flips to return
            agent: Filter by agent
            flip_type: Filter by flip type

        Returns:
            Recent flip events with context
        """
        params: dict[str, Any] = {"limit": limit}
        if agent:
            params["agent"] = agent
        if flip_type:
            params["flip_type"] = flip_type
        return await self._client._get("/api/v1/analytics/flips/recent", params=params)

    async def get_flip_consistency(
        self,
        *,
        agents: list[str] | None = None,
    ) -> dict[str, Any]:
        """Get flip consistency metrics.

        Args:
            agents: Filter by specific agents

        Returns:
            Consistency scores for agents
        """
        if agents:
            return await self._client._post(
                "/api/v1/analytics/flips/consistency",
                {"agents": agents},
            )
        return await self._client._get("/api/v1/analytics/flips/consistency")

    async def get_flip_trends(
        self,
        *,
        days: int = 30,
        granularity: str = "day",
    ) -> dict[str, Any]:
        """Get flip trends over time.

        Args:
            days: Number of days to analyze
            granularity: Data granularity (hour, day, week)

        Returns:
            Time-series flip data
        """
        return await self._client._get(
            "/api/v1/analytics/flips/trends",
            params={"days": days, "granularity": granularity},
        )

    # =========================================================================
    # Deliberation Analytics
    # =========================================================================

    async def get_deliberation_analytics(
        self,
        org_id: str,
        *,
        days: int = 30,
    ) -> dict[str, Any]:
        """Get deliberation analytics.

        Args:
            org_id: Organization ID
            days: Number of days to analyze

        Returns:
            Deliberation patterns and metrics
        """
        return await self._client._get(
            f"/api/v1/analytics/deliberations/{org_id}",
            params={"days": days},
        )

    async def get_deliberation_channels(
        self,
        org_id: str,
        *,
        days: int = 30,
    ) -> dict[str, Any]:
        """Get deliberation by channel.

        Args:
            org_id: Organization ID
            days: Number of days to analyze

        Returns:
            Channel-wise deliberation metrics
        """
        return await self._client._get(
            f"/api/v1/analytics/deliberations/{org_id}/channels",
            params={"days": days},
        )

    async def get_deliberation_consensus(
        self,
        org_id: str,
        *,
        days: int = 30,
    ) -> dict[str, Any]:
        """Get deliberation consensus analytics.

        Args:
            org_id: Organization ID
            days: Number of days to analyze

        Returns:
            Consensus achievement rates and quality
        """
        return await self._client._get(
            f"/api/v1/analytics/deliberations/{org_id}/consensus",
            params={"days": days},
        )

    async def get_deliberation_performance(
        self,
        org_id: str,
        *,
        days: int = 30,
        granularity: str = "daily",
    ) -> dict[str, Any]:
        """Get deliberation performance over time.

        Args:
            org_id: Organization ID
            days: Number of days to analyze
            granularity: Data granularity (hourly, daily, weekly)

        Returns:
            Time-series deliberation metrics
        """
        return await self._client._get(
            f"/api/v1/analytics/deliberations/{org_id}/performance",
            params={"days": days, "granularity": granularity},
        )
