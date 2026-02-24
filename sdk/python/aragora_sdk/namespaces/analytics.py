"""
Analytics Namespace API

Provides methods for analytics and metrics operations.

Features:
- Disagreement patterns between agents
- Role rotation statistics
- Early stop analysis
- Consensus quality metrics
- Ranking and memory statistics
- Debate analytics
- Agent performance
- Usage and cost tracking
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class AnalyticsAPI:
    """
    Synchronous Analytics API.

    Provides methods for retrieving analytics and metrics:
    - Disagreement patterns between agents
    - Role rotation statistics
    - Early stop analysis
    - Consensus quality metrics
    - Ranking and memory statistics

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> disagreements = client.analytics.disagreements(period="7d")
        >>> quality = client.analytics.consensus_quality()
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    # ===========================================================================
    # Core Analytics
    # ===========================================================================

    def disagreements(self, period: str | None = None) -> dict[str, Any]:
        """
        Get disagreement analytics showing patterns of agent disagreements.

        Args:
            period: Time period (e.g., '7d', '30d', '90d')

        Returns:
            Disagreement analytics data
        """
        params = {}
        if period:
            params["period"] = period
        return self._client.request("GET", "/api/v1/analytics/disagreements", params=params)

    def role_rotation(self, period: str | None = None) -> dict[str, Any]:
        """
        Get role rotation analytics showing how agents switch roles.

        Args:
            period: Time period (e.g., '7d', '30d', '90d')

        Returns:
            Role rotation analytics data
        """
        params = {}
        if period:
            params["period"] = period
        return self._client.request("GET", "/api/v1/analytics/role-rotation", params=params)

    def early_stops(self, period: str | None = None) -> dict[str, Any]:
        """
        Get early stop analytics showing debates that ended early.

        Args:
            period: Time period (e.g., '7d', '30d', '90d')

        Returns:
            Early stop analytics data
        """
        params = {}
        if period:
            params["period"] = period
        return self._client.request("GET", "/api/v1/analytics/early-stops", params=params)

    def consensus_quality(self, period: str | None = None) -> dict[str, Any]:
        """
        Get consensus quality analytics.

        Args:
            period: Time period (e.g., '7d', '30d', '90d')

        Returns:
            Consensus quality metrics
        """
        params = {}
        if period:
            params["period"] = period
        return self._client.request("GET", "/api/v1/analytics/consensus-quality", params=params)

    # ===========================================================================
    # Dashboard Overview
    # ===========================================================================

    def get_summary(
        self,
        workspace_id: str | None = None,
        time_range: str | None = None,
    ) -> dict[str, Any]:
        """
        Get dashboard summary with key metrics.

        Args:
            workspace_id: Optional workspace filter
            time_range: Time range (e.g., '24h', '7d', '30d')

        Returns:
            Summary metrics
        """
        params = {}
        if workspace_id:
            params["workspace_id"] = workspace_id
        if time_range:
            params["time_range"] = time_range
        return self._client.request("GET", "/api/analytics/summary", params=params)

    def get_finding_trends(
        self,
        workspace_id: str | None = None,
        time_range: str | None = None,
        granularity: str | None = None,
    ) -> dict[str, Any]:
        """
        Get finding trends over time.

        Args:
            workspace_id: Optional workspace filter
            time_range: Time range (e.g., '24h', '7d', '30d')
            granularity: Data granularity (e.g., 'hour', 'day', 'week')

        Returns:
            Finding trends data
        """
        params = {}
        if workspace_id:
            params["workspace_id"] = workspace_id
        if time_range:
            params["time_range"] = time_range
        if granularity:
            params["granularity"] = granularity
        return self._client.request("GET", "/api/analytics/trends/findings", params=params)

    def get_risk_heatmap(
        self,
        workspace_id: str | None = None,
        time_range: str | None = None,
    ) -> dict[str, Any]:
        """
        Get risk heatmap data.

        Args:
            workspace_id: Optional workspace filter
            time_range: Time range

        Returns:
            Risk heatmap data
        """
        params = {}
        if workspace_id:
            params["workspace_id"] = workspace_id
        if time_range:
            params["time_range"] = time_range
        return self._client.request("GET", "/api/analytics/heatmap", params=params)

    # ===========================================================================
    # Debate Analytics
    # ===========================================================================

    def debates_overview(self) -> dict[str, Any]:
        """
        Get debates overview metrics.

        Returns:
            Dict with total, consensus_rate, average_rounds
        """
        return self._client.request("GET", "/api/analytics/debates/overview")

    def debate_trends(
        self,
        time_range: str | None = None,
        granularity: str | None = None,
    ) -> dict[str, Any]:
        """
        Get debate trends over time.

        Args:
            time_range: Time range
            granularity: Data granularity

        Returns:
            Debate trends data
        """
        params = {}
        if time_range:
            params["time_range"] = time_range
        if granularity:
            params["granularity"] = granularity
        return self._client.request("GET", "/api/analytics/debates/trends", params=params)

    def debate_topics(
        self,
        time_range: str | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        """
        Get topic distribution and consensus by topic.

        Args:
            time_range: Time range
            limit: Maximum number of topics

        Returns:
            Topic distribution data
        """
        params: dict[str, Any] = {"limit": limit}
        if time_range:
            params["time_range"] = time_range
        return self._client.request("GET", "/api/analytics/debates/topics", params=params)

    def debate_outcomes(self, time_range: str | None = None) -> dict[str, Any]:
        """
        Get debate outcome distribution.

        Args:
            time_range: Time range

        Returns:
            Outcome distribution data
        """
        params = {}
        if time_range:
            params["time_range"] = time_range
        return self._client.request("GET", "/api/analytics/debates/outcomes", params=params)

    def outcomes_summary(self, period: str = "30d") -> dict[str, Any]:
        """Get outcome analytics summary."""
        return self._client.request("GET", "/api/analytics/outcomes", params={"period": period})

    def outcomes_average_rounds(self, period: str = "30d") -> dict[str, Any]:
        """Get average rounds from outcome analytics."""
        return self._client.request(
            "GET",
            "/api/analytics/outcomes/average-rounds",
            params={"period": period},
        )

    def outcomes_consensus_rate(self, period: str = "30d") -> dict[str, Any]:
        """Get consensus rate from outcome analytics."""
        return self._client.request(
            "GET",
            "/api/analytics/outcomes/consensus-rate",
            params={"period": period},
        )

    def outcomes_contributions(self, period: str = "30d") -> dict[str, Any]:
        """Get contributions from outcome analytics."""
        return self._client.request(
            "GET",
            "/api/analytics/outcomes/contributions",
            params={"period": period},
        )

    def outcomes_quality_trend(self, period: str = "30d") -> dict[str, Any]:
        """Get quality trend from outcome analytics."""
        return self._client.request(
            "GET",
            "/api/analytics/outcomes/quality-trend",
            params={"period": period},
        )

    def outcomes_topics(self, period: str = "30d") -> dict[str, Any]:
        """Get topics from outcome analytics."""
        return self._client.request(
            "GET",
            "/api/analytics/outcomes/topics",
            params={"period": period},
        )

    def differentiation_summary(self) -> dict[str, Any]:
        """Get differentiation summary metrics."""
        return self._client.request("GET", "/api/differentiation/summary")

    def differentiation_vetting(self) -> dict[str, Any]:
        """Get differentiation vetting metrics."""
        return self._client.request("GET", "/api/differentiation/vetting")

    def differentiation_calibration(self) -> dict[str, Any]:
        """Get differentiation calibration metrics."""
        return self._client.request("GET", "/api/differentiation/calibration")

    def differentiation_memory(self) -> dict[str, Any]:
        """Get differentiation memory metrics."""
        return self._client.request("GET", "/api/differentiation/memory")

    def differentiation_benchmarks(self) -> dict[str, Any]:
        """Get differentiation benchmark metrics."""
        return self._client.request("GET", "/api/differentiation/benchmarks")

    # ===========================================================================
    # Agent Analytics
    # ===========================================================================

    def agent_leaderboard(
        self,
        limit: int = 20,
        domain: str | None = None,
    ) -> dict[str, Any]:
        """
        Get agent leaderboard with ELO rankings.

        Args:
            limit: Maximum number of agents
            domain: Filter by domain

        Returns:
            Agent leaderboard data
        """
        params: dict[str, Any] = {"limit": limit}
        if domain:
            params["domain"] = domain
        return self._client.request("GET", "/api/analytics/agents/leaderboard", params=params)

    def compare_agents(self, agents: list[str]) -> dict[str, Any]:
        """
        Get multi-agent comparison.

        Args:
            agents: List of agent IDs to compare

        Returns:
            Agent comparison data
        """
        return self._client.request(
            "GET",
            "/api/analytics/agents/comparison",
            params={"agents": ",".join(agents)},
        )

    def agents_performance_summary(
        self,
        time_range: str = "30d",
        org_id: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        """
        Get aggregate performance metrics for all agents.

        Args:
            time_range: Time range (e.g., '7d', '30d', '90d')
            org_id: Organization ID filter
            limit: Maximum number of agents to include

        Returns:
            Aggregate agent performance data including:
            - Total debates participated
            - Average consensus contribution
            - Win rates by agent
            - Performance trends
        """
        params: dict[str, Any] = {"time_range": time_range, "limit": limit}
        if org_id:
            params["org_id"] = org_id
        return self._client.request("GET", "/api/v1/analytics/agents/performance", params=params)

    def debates_summary(
        self,
        time_range: str = "30d",
        org_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Get summary statistics for debates.

        Args:
            time_range: Time range (e.g., '7d', '30d', '90d')
            org_id: Organization ID filter

        Returns:
            Debate summary including:
            - Total debates
            - Consensus rate
            - Average rounds
            - Outcomes distribution
        """
        params: dict[str, Any] = {"time_range": time_range}
        if org_id:
            params["org_id"] = org_id
        return self._client.request("GET", "/api/v1/analytics/debates/summary", params=params)

    def calibration_stats(self, agent: str | None = None) -> dict[str, Any]:
        """
        Get calibration statistics.

        Args:
            agent: Optional agent filter

        Returns:
            Calibration statistics
        """
        params = {}
        if agent:
            params["agent"] = agent
        return self._client.request("GET", "/api/analytics/calibration", params=params)

    def get_trends(
        self,
        period: str | None = None,
        granularity: str | None = None,
    ) -> dict[str, Any]:
        """Get general trend analysis across debates, consensus, and agents."""
        params: dict[str, Any] = {}
        if period:
            params["time_range"] = period
        if granularity:
            params["granularity"] = granularity
        return self._client.request("GET", "/api/v1/analytics/trends", params=params)

    # ===========================================================================
    # Usage & Costs
    # ===========================================================================

    def token_usage(
        self,
        org_id: str | None = None,
        time_range: str | None = None,
        granularity: str | None = None,
    ) -> dict[str, Any]:
        """
        Get token consumption trends.

        Args:
            org_id: Organization ID filter
            time_range: Time range
            granularity: Data granularity

        Returns:
            Token usage data
        """
        params = {}
        if org_id:
            params["org_id"] = org_id
        if time_range:
            params["time_range"] = time_range
        if granularity:
            params["granularity"] = granularity
        return self._client.request("GET", "/api/analytics/usage/tokens", params=params)

    def cost_breakdown(
        self,
        org_id: str | None = None,
        time_range: str | None = None,
    ) -> dict[str, Any]:
        """
        Get cost breakdown by provider and model.

        Args:
            org_id: Organization ID filter
            time_range: Time range

        Returns:
            Cost breakdown data
        """
        params = {}
        if org_id:
            params["org_id"] = org_id
        if time_range:
            params["time_range"] = time_range
        return self._client.request("GET", "/api/analytics/usage/costs", params=params)

    def cost_breakdown_dashboard(
        self,
        workspace_id: str | None = None,
        time_range: str | None = None,
    ) -> dict[str, Any]:
        """Get per-agent cost breakdown with budget utilization.

        @route GET /api/analytics/cost/breakdown
        """
        params: dict[str, str] = {}
        if workspace_id:
            params["workspace_id"] = workspace_id
        if time_range:
            params["time_range"] = time_range
        return self._client.request("GET", "/api/analytics/cost/breakdown", params=params or None)

    def active_users(
        self,
        org_id: str | None = None,
        time_range: str | None = None,
    ) -> dict[str, Any]:
        """
        Get active user counts and growth.

        Args:
            org_id: Organization ID filter
            time_range: Time range

        Returns:
            Active user data
        """
        params = {}
        if org_id:
            params["org_id"] = org_id
        if time_range:
            params["time_range"] = time_range
        return self._client.request("GET", "/api/analytics/usage/active_users", params=params)

    # ===========================================================================
    # Flip Detection
    # ===========================================================================

    def flip_summary(self) -> dict[str, Any]:
        """
        Get flip detection summary.

        Returns:
            Flip summary data
        """
        return self._client.request("GET", "/api/analytics/flips/summary")

    def recent_flips(
        self,
        limit: int = 20,
        agent: str | None = None,
        flip_type: str | None = None,
    ) -> dict[str, Any]:
        """
        Get recent flip events.

        Args:
            limit: Maximum number of flips
            agent: Filter by agent
            flip_type: Filter by flip type

        Returns:
            Recent flips data
        """
        params: dict[str, Any] = {"limit": limit}
        if agent:
            params["agent"] = agent
        if flip_type:
            params["flip_type"] = flip_type
        return self._client.request("GET", "/api/analytics/flips/recent", params=params)

    def agent_consistency(self, agents: list[str] | None = None) -> dict[str, Any]:
        """
        Get agent consistency scores.

        Args:
            agents: List of agent IDs to check

        Returns:
            Agent consistency data
        """
        params = {}
        if agents:
            params["agents"] = ",".join(agents)
        return self._client.request("GET", "/api/analytics/flips/consistency", params=params)

    # ===========================================================================
    # Deliberation Analytics
    # ===========================================================================

    def deliberation_summary(
        self,
        org_id: str | None = None,
        days: int = 30,
    ) -> dict[str, Any]:
        """
        Get deliberation summary.

        Args:
            org_id: Organization ID filter
            days: Number of days to include

        Returns:
            Deliberation summary data
        """
        params: dict[str, Any] = {"days": days}
        if org_id:
            params["org_id"] = org_id
        return self._client.request("GET", "/api/analytics/deliberations", params=params)

    def deliberations_by_channel(
        self,
        org_id: str | None = None,
        days: int = 30,
    ) -> dict[str, Any]:
        """
        Get deliberations by channel.

        Args:
            org_id: Organization ID filter
            days: Number of days to include

        Returns:
            Channel breakdown data
        """
        params: dict[str, Any] = {"days": days}
        if org_id:
            params["org_id"] = org_id
        return self._client.request("GET", "/api/analytics/deliberations/channels", params=params)

    def consensus_rates(
        self,
        org_id: str | None = None,
        days: int = 30,
    ) -> dict[str, Any]:
        """
        Get consensus rates by agent team.

        Args:
            org_id: Organization ID filter
            days: Number of days to include

        Returns:
            Consensus rate data
        """
        params: dict[str, Any] = {"days": days}
        if org_id:
            params["org_id"] = org_id
        return self._client.request("GET", "/api/analytics/deliberations/consensus", params=params)

    def get_agent_performance(self, agent_id: str) -> dict[str, Any]:
        """Get performance metrics for a specific agent."""
        return self._client.request("GET", f"/api/v1/analytics/agents/{agent_id}/performance")


class AsyncAnalyticsAPI:
    """
    Asynchronous Analytics API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     disagreements = await client.analytics.disagreements(period="7d")
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    # ===========================================================================
    # Core Analytics
    # ===========================================================================

    async def disagreements(self, period: str | None = None) -> dict[str, Any]:
        """Get disagreement analytics."""
        params = {}
        if period:
            params["period"] = period
        return await self._client.request("GET", "/api/v1/analytics/disagreements", params=params)

    async def role_rotation(self, period: str | None = None) -> dict[str, Any]:
        """Get role rotation analytics."""
        params = {}
        if period:
            params["period"] = period
        return await self._client.request("GET", "/api/v1/analytics/role-rotation", params=params)

    async def early_stops(self, period: str | None = None) -> dict[str, Any]:
        """Get early stop analytics."""
        params = {}
        if period:
            params["period"] = period
        return await self._client.request("GET", "/api/v1/analytics/early-stops", params=params)

    async def consensus_quality(self, period: str | None = None) -> dict[str, Any]:
        """Get consensus quality analytics."""
        params = {}
        if period:
            params["period"] = period
        return await self._client.request(
            "GET", "/api/v1/analytics/consensus-quality", params=params
        )

    # ===========================================================================
    # Dashboard Overview
    # ===========================================================================

    async def get_summary(
        self,
        workspace_id: str | None = None,
        time_range: str | None = None,
    ) -> dict[str, Any]:
        """Get dashboard summary with key metrics."""
        params = {}
        if workspace_id:
            params["workspace_id"] = workspace_id
        if time_range:
            params["time_range"] = time_range
        return await self._client.request("GET", "/api/analytics/summary", params=params)

    async def get_risk_heatmap(
        self,
        workspace_id: str | None = None,
        time_range: str | None = None,
    ) -> dict[str, Any]:
        """Get risk heatmap data."""
        params = {}
        if workspace_id:
            params["workspace_id"] = workspace_id
        if time_range:
            params["time_range"] = time_range
        return await self._client.request("GET", "/api/analytics/heatmap", params=params)

    # ===========================================================================
    # Debate Analytics
    # ===========================================================================

    async def debates_overview(self) -> dict[str, Any]:
        """Get debates overview metrics."""
        return await self._client.request("GET", "/api/analytics/debates/overview")

    async def debate_trends(
        self,
        time_range: str | None = None,
        granularity: str | None = None,
    ) -> dict[str, Any]:
        """Get debate trends over time."""
        params = {}
        if time_range:
            params["time_range"] = time_range
        if granularity:
            params["granularity"] = granularity
        return await self._client.request("GET", "/api/analytics/debates/trends", params=params)

    async def debate_topics(
        self,
        time_range: str | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Get topic distribution and consensus by topic."""
        params: dict[str, Any] = {"limit": limit}
        if time_range:
            params["time_range"] = time_range
        return await self._client.request("GET", "/api/analytics/debates/topics", params=params)

    async def outcomes_summary(self, period: str = "30d") -> dict[str, Any]:
        """Get outcome analytics summary."""
        return await self._client.request(
            "GET", "/api/analytics/outcomes", params={"period": period}
        )

    async def outcomes_average_rounds(self, period: str = "30d") -> dict[str, Any]:
        """Get average rounds from outcome analytics."""
        return await self._client.request(
            "GET",
            "/api/analytics/outcomes/average-rounds",
            params={"period": period},
        )

    async def outcomes_consensus_rate(self, period: str = "30d") -> dict[str, Any]:
        """Get consensus rate from outcome analytics."""
        return await self._client.request(
            "GET",
            "/api/analytics/outcomes/consensus-rate",
            params={"period": period},
        )

    async def outcomes_contributions(self, period: str = "30d") -> dict[str, Any]:
        """Get contributions from outcome analytics."""
        return await self._client.request(
            "GET",
            "/api/analytics/outcomes/contributions",
            params={"period": period},
        )

    async def outcomes_quality_trend(self, period: str = "30d") -> dict[str, Any]:
        """Get quality trend from outcome analytics."""
        return await self._client.request(
            "GET",
            "/api/analytics/outcomes/quality-trend",
            params={"period": period},
        )

    async def outcomes_topics(self, period: str = "30d") -> dict[str, Any]:
        """Get topics from outcome analytics."""
        return await self._client.request(
            "GET",
            "/api/analytics/outcomes/topics",
            params={"period": period},
        )

    async def differentiation_summary(self) -> dict[str, Any]:
        """Get differentiation summary metrics."""
        return await self._client.request("GET", "/api/differentiation/summary")

    async def differentiation_vetting(self) -> dict[str, Any]:
        """Get differentiation vetting metrics."""
        return await self._client.request("GET", "/api/differentiation/vetting")

    async def differentiation_calibration(self) -> dict[str, Any]:
        """Get differentiation calibration metrics."""
        return await self._client.request("GET", "/api/differentiation/calibration")

    async def differentiation_memory(self) -> dict[str, Any]:
        """Get differentiation memory metrics."""
        return await self._client.request("GET", "/api/differentiation/memory")

    async def differentiation_benchmarks(self) -> dict[str, Any]:
        """Get differentiation benchmark metrics."""
        return await self._client.request("GET", "/api/differentiation/benchmarks")

    # ===========================================================================
    # Agent Analytics
    # ===========================================================================

    async def agent_leaderboard(
        self,
        limit: int = 20,
        domain: str | None = None,
    ) -> dict[str, Any]:
        """Get agent leaderboard with ELO rankings."""
        params: dict[str, Any] = {"limit": limit}
        if domain:
            params["domain"] = domain
        return await self._client.request("GET", "/api/analytics/agents/leaderboard", params=params)

    async def compare_agents(self, agents: list[str]) -> dict[str, Any]:
        """Get multi-agent comparison."""
        return await self._client.request(
            "GET",
            "/api/analytics/agents/comparison",
            params={"agents": ",".join(agents)},
        )

    async def agents_performance_summary(
        self,
        time_range: str = "30d",
        org_id: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Get aggregate performance metrics for all agents."""
        params: dict[str, Any] = {"time_range": time_range, "limit": limit}
        if org_id:
            params["org_id"] = org_id
        return await self._client.request(
            "GET", "/api/v1/analytics/agents/performance", params=params
        )

    async def debates_summary(
        self,
        time_range: str = "30d",
        org_id: str | None = None,
    ) -> dict[str, Any]:
        """Get summary statistics for debates."""
        params: dict[str, Any] = {"time_range": time_range}
        if org_id:
            params["org_id"] = org_id
        return await self._client.request("GET", "/api/v1/analytics/debates/summary", params=params)

    # ===========================================================================
    # Usage & Costs
    # ===========================================================================

    async def token_usage(
        self,
        org_id: str | None = None,
        time_range: str | None = None,
        granularity: str | None = None,
    ) -> dict[str, Any]:
        """Get token consumption trends."""
        params = {}
        if org_id:
            params["org_id"] = org_id
        if time_range:
            params["time_range"] = time_range
        if granularity:
            params["granularity"] = granularity
        return await self._client.request("GET", "/api/analytics/usage/tokens", params=params)

    async def cost_breakdown(
        self,
        org_id: str | None = None,
        time_range: str | None = None,
    ) -> dict[str, Any]:
        """Get cost breakdown by provider and model."""
        params = {}
        if org_id:
            params["org_id"] = org_id
        if time_range:
            params["time_range"] = time_range
        return await self._client.request("GET", "/api/analytics/usage/costs", params=params)

    async def cost_breakdown_dashboard(
        self,
        workspace_id: str | None = None,
        time_range: str | None = None,
    ) -> dict[str, Any]:
        """Get per-agent cost breakdown with budget utilization.

        @route GET /api/analytics/cost/breakdown
        """
        params: dict[str, str] = {}
        if workspace_id:
            params["workspace_id"] = workspace_id
        if time_range:
            params["time_range"] = time_range
        return await self._client.request(
            "GET", "/api/analytics/cost/breakdown", params=params or None
        )

    # ===========================================================================
    # Flip Detection
    # ===========================================================================

    async def flip_summary(self) -> dict[str, Any]:
        """Get flip detection summary."""
        return await self._client.request("GET", "/api/analytics/flips/summary")

    async def recent_flips(
        self,
        limit: int = 20,
        agent: str | None = None,
        flip_type: str | None = None,
    ) -> dict[str, Any]:
        """Get recent flip events."""
        params: dict[str, Any] = {"limit": limit}
        if agent:
            params["agent"] = agent
        if flip_type:
            params["flip_type"] = flip_type
        return await self._client.request("GET", "/api/analytics/flips/recent", params=params)

    # ===========================================================================
    # Deliberation Analytics
    # ===========================================================================

    async def deliberation_summary(
        self,
        org_id: str | None = None,
        days: int = 30,
    ) -> dict[str, Any]:
        """Get deliberation summary."""
        params: dict[str, Any] = {"days": days}
        if org_id:
            params["org_id"] = org_id
        return await self._client.request("GET", "/api/analytics/deliberations", params=params)

    async def consensus_rates(
        self,
        org_id: str | None = None,
        days: int = 30,
    ) -> dict[str, Any]:
        """Get consensus rates by agent team."""
        params: dict[str, Any] = {"days": days}
        if org_id:
            params["org_id"] = org_id
        return await self._client.request(
            "GET", "/api/analytics/deliberations/consensus", params=params
        )

    # ===========================================================================
    # Trends
    # ===========================================================================

    async def get_trends(
        self,
        period: str | None = None,
        granularity: str | None = None,
    ) -> dict[str, Any]:
        """Get general trend analysis across debates, consensus, and agents."""
        params: dict[str, Any] = {}
        if period:
            params["time_range"] = period
        if granularity:
            params["granularity"] = granularity
        return await self._client.request("GET", "/api/v1/analytics/trends", params=params)

    async def get_agent_performance(self, agent_id: str) -> dict[str, Any]:
        """Get performance metrics for a specific agent."""
        return await self._client.request("GET", f"/api/v1/analytics/agents/{agent_id}/performance")
