"""
Analytics Dashboard to Team Selector Bridge.

Bridges agent metrics from AnalyticsDashboard (precision, agreement_rate, response_time)
into TeamSelector scoring, enabling data-driven team composition.

This closes the loop between:
1. AnalyticsDashboard: Computes AgentMetrics (precision, agreement_rate, finding_distribution)
2. TeamSelector: Selects teams based on ELO/calibration but ignores actual precision data

By connecting them, we enable:
- Precision-weighted team selection
- Agreement rate as calibration signal
- Response time factoring for time-sensitive tasks
- Finding distribution for domain expertise matching

Usage:
    from aragora.debate.analytics_selection_bridge import AnalyticsSelectionBridge

    bridge = AnalyticsSelectionBridge(
        analytics_dashboard=dashboard,
        precision_weight=0.3,
    )

    # Get selection boost for an agent
    boost = bridge.compute_selection_boost("claude")

    # Or integrate with TeamSelector
    team_selector.set_analytics_bridge(bridge)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from aragora.analytics.dashboard import AnalyticsDashboard, AgentMetrics

logger = logging.getLogger(__name__)


@dataclass
class SelectionBoost:
    """Computed selection boost for an agent."""

    agent_name: str
    total_boost: float
    precision_component: float = 0.0
    agreement_component: float = 0.0
    response_time_component: float = 0.0
    domain_expertise_component: float = 0.0
    confidence: float = 0.0  # Based on data quantity


@dataclass
class DomainExpertise:
    """Inferred domain expertise from finding distribution."""

    agent_name: str
    primary_domain: str
    domain_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class AnalyticsSelectionBridgeConfig:
    """Configuration for the analytics-selection bridge."""

    # Minimum findings before using an agent's metrics
    min_findings_for_boost: int = 10

    # Weight for precision in selection boost
    precision_weight: float = 0.35

    # Weight for agreement rate (proxy for calibration)
    agreement_weight: float = 0.25

    # Weight for response time (inverted - faster is better)
    response_time_weight: float = 0.15

    # Weight for domain expertise match
    domain_expertise_weight: float = 0.25

    # Response time baseline (ms) - agents faster than this get bonus
    response_time_baseline_ms: float = 5000.0

    # Maximum response time considered (ms)
    max_response_time_ms: float = 30000.0

    # Cache TTL for metrics (seconds)
    cache_ttl_seconds: int = 300


@dataclass
class AnalyticsSelectionBridge:
    """Bridges AnalyticsDashboard metrics into TeamSelector decisions.

    Key integration points:
    1. Converts precision scores into selection boosts
    2. Uses agreement rate as calibration signal
    3. Factors response time for time-sensitive tasks
    4. Infers domain expertise from finding distribution
    """

    analytics_dashboard: Optional["AnalyticsDashboard"] = None
    config: AnalyticsSelectionBridgeConfig = field(default_factory=AnalyticsSelectionBridgeConfig)

    # Cached metrics
    _metrics_cache: Dict[str, "AgentMetrics"] = field(default_factory=dict, repr=False)
    _cache_timestamp: Optional[datetime] = field(default=None, repr=False)
    _domain_expertise_cache: Dict[str, DomainExpertise] = field(default_factory=dict, repr=False)

    async def refresh_metrics(self, workspace_id: Optional[str] = None) -> int:
        """Refresh agent metrics from dashboard.

        Args:
            workspace_id: Optional workspace filter

        Returns:
            Number of agents with metrics loaded
        """
        if self.analytics_dashboard is None:
            logger.debug("No analytics dashboard configured")
            return 0

        try:
            # Get agent metrics from dashboard
            metrics_list = await self.analytics_dashboard.get_agent_metrics(
                workspace_id=workspace_id
            )

            self._metrics_cache.clear()
            for metrics in metrics_list:
                self._metrics_cache[metrics.agent_name] = metrics
                self._compute_domain_expertise(metrics)

            self._cache_timestamp = datetime.now()

            logger.info(f"analytics_metrics_refreshed agents={len(self._metrics_cache)}")
            return len(self._metrics_cache)

        except Exception as e:
            logger.warning(f"Failed to refresh analytics metrics: {e}")
            return 0

    def _compute_domain_expertise(self, metrics: "AgentMetrics") -> None:
        """Compute domain expertise from finding distribution.

        Args:
            metrics: Agent metrics with finding distribution
        """
        if not metrics.finding_distribution:
            return

        total_findings = sum(metrics.finding_distribution.values())
        if total_findings == 0:
            return

        # Normalize to get domain scores
        domain_scores = {
            domain: count / total_findings for domain, count in metrics.finding_distribution.items()
        }

        # Find primary domain
        primary_domain = max(domain_scores.keys(), key=lambda d: domain_scores[d])

        self._domain_expertise_cache[metrics.agent_name] = DomainExpertise(
            agent_name=metrics.agent_name,
            primary_domain=primary_domain,
            domain_scores=domain_scores,
        )

    def _is_cache_valid(self) -> bool:
        """Check if cached metrics are still valid."""
        if self._cache_timestamp is None:
            return False

        elapsed = (datetime.now() - self._cache_timestamp).total_seconds()
        return elapsed < self.config.cache_ttl_seconds

    def compute_selection_boost(
        self,
        agent_name: str,
        target_domain: Optional[str] = None,
        time_sensitive: bool = False,
    ) -> SelectionBoost:
        """Compute selection boost for an agent based on analytics.

        Args:
            agent_name: Name of the agent
            target_domain: Optional target domain for expertise matching
            time_sensitive: If True, weight response time more heavily

        Returns:
            SelectionBoost with component breakdown
        """
        metrics = self._metrics_cache.get(agent_name)

        if metrics is None or metrics.total_findings < self.config.min_findings_for_boost:
            # Return neutral boost for insufficient data
            return SelectionBoost(
                agent_name=agent_name,
                total_boost=0.0,
                confidence=0.0,
            )

        # Compute precision component (0-1, higher is better)
        precision_component = metrics.precision * self.config.precision_weight

        # Compute agreement component (0-1, higher is better calibration)
        agreement_component = metrics.agreement_rate * self.config.agreement_weight

        # Compute response time component (inverted, faster is better)
        if metrics.avg_response_time_ms > 0:
            # Normalize: baseline or faster = 1.0, max_response = 0.0
            time_score = max(
                0.0,
                1.0
                - (metrics.avg_response_time_ms - self.config.response_time_baseline_ms)
                / (self.config.max_response_time_ms - self.config.response_time_baseline_ms),
            )
            time_score = min(1.0, time_score)  # Cap at 1.0
        else:
            time_score = 0.5  # Neutral for no data

        # Adjust weight if time-sensitive
        time_weight = self.config.response_time_weight
        if time_sensitive:
            time_weight *= 1.5  # Boost importance of response time

        response_time_component = time_score * time_weight

        # Compute domain expertise component
        domain_component = 0.0
        if target_domain and agent_name in self._domain_expertise_cache:
            expertise = self._domain_expertise_cache[agent_name]
            # Check for exact or partial domain match
            domain_score = expertise.domain_scores.get(target_domain, 0.0)

            # Also check for related domains
            for domain, score in expertise.domain_scores.items():
                if (
                    target_domain.lower() in domain.lower()
                    or domain.lower() in target_domain.lower()
                ):
                    domain_score = max(domain_score, score * 0.8)  # 80% credit for partial match

            domain_component = domain_score * self.config.domain_expertise_weight

        # Compute total boost
        total_boost = (
            precision_component + agreement_component + response_time_component + domain_component
        )

        # Compute confidence based on data quantity
        confidence = min(1.0, metrics.total_findings / 100)  # Max confidence at 100 findings

        return SelectionBoost(
            agent_name=agent_name,
            total_boost=total_boost,
            precision_component=precision_component,
            agreement_component=agreement_component,
            response_time_component=response_time_component,
            domain_expertise_component=domain_component,
            confidence=confidence,
        )

    def get_all_selection_boosts(
        self,
        target_domain: Optional[str] = None,
        time_sensitive: bool = False,
    ) -> Dict[str, SelectionBoost]:
        """Get selection boosts for all tracked agents.

        Args:
            target_domain: Optional target domain for expertise matching
            time_sensitive: If True, weight response time more heavily

        Returns:
            Dict mapping agent names to SelectionBoost
        """
        return {
            agent_name: self.compute_selection_boost(agent_name, target_domain, time_sensitive)
            for agent_name in self._metrics_cache.keys()
        }

    def rank_agents_by_domain(self, domain: str) -> List[str]:
        """Rank agents by expertise in a specific domain.

        Args:
            domain: Target domain

        Returns:
            List of agent names sorted by domain expertise (descending)
        """
        agents_with_scores = []

        for agent_name, expertise in self._domain_expertise_cache.items():
            score = expertise.domain_scores.get(domain, 0.0)

            # Check for partial matches
            for d, s in expertise.domain_scores.items():
                if domain.lower() in d.lower() or d.lower() in domain.lower():
                    score = max(score, s * 0.8)

            agents_with_scores.append((agent_name, score))

        # Sort by score descending
        agents_with_scores.sort(key=lambda x: x[1], reverse=True)

        return [agent for agent, _ in agents_with_scores]

    def get_precision_leaders(self, top_n: int = 5) -> List[str]:
        """Get agents with highest precision.

        Args:
            top_n: Number of agents to return

        Returns:
            List of agent names sorted by precision
        """
        sorted_agents = sorted(
            self._metrics_cache.values(),
            key=lambda m: m.precision,
            reverse=True,
        )

        return [m.agent_name for m in sorted_agents[:top_n]]

    def get_fastest_agents(self, top_n: int = 5) -> List[str]:
        """Get agents with lowest response times.

        Args:
            top_n: Number of agents to return

        Returns:
            List of agent names sorted by response time (ascending)
        """
        # Filter out agents with no response time data
        agents_with_times = [m for m in self._metrics_cache.values() if m.avg_response_time_ms > 0]

        sorted_agents = sorted(agents_with_times, key=lambda m: m.avg_response_time_ms)

        return [m.agent_name for m in sorted_agents[:top_n]]

    def get_metrics(self, agent_name: str) -> Optional["AgentMetrics"]:
        """Get cached metrics for an agent.

        Args:
            agent_name: Name of the agent

        Returns:
            AgentMetrics if available
        """
        return self._metrics_cache.get(agent_name)

    def get_domain_expertise(self, agent_name: str) -> Optional[DomainExpertise]:
        """Get inferred domain expertise for an agent.

        Args:
            agent_name: Name of the agent

        Returns:
            DomainExpertise if available
        """
        return self._domain_expertise_cache.get(agent_name)

    def get_stats(self) -> Dict[str, Any]:
        """Get bridge statistics.

        Returns:
            Dict with bridge metrics
        """
        return {
            "agents_tracked": len(self._metrics_cache),
            "cache_valid": self._is_cache_valid(),
            "cache_age_seconds": (
                (datetime.now() - self._cache_timestamp).total_seconds()
                if self._cache_timestamp
                else None
            ),
            "agents_with_expertise": len(self._domain_expertise_cache),
        }


def create_analytics_selection_bridge(
    analytics_dashboard: Optional["AnalyticsDashboard"] = None,
    **config_kwargs: Any,
) -> AnalyticsSelectionBridge:
    """Create and configure an AnalyticsSelectionBridge.

    Args:
        analytics_dashboard: AnalyticsDashboard instance
        **config_kwargs: Additional configuration options

    Returns:
        Configured AnalyticsSelectionBridge instance
    """
    config = AnalyticsSelectionBridgeConfig(**config_kwargs)
    return AnalyticsSelectionBridge(
        analytics_dashboard=analytics_dashboard,
        config=config,
    )


__all__ = [
    "AnalyticsSelectionBridge",
    "AnalyticsSelectionBridgeConfig",
    "SelectionBoost",
    "DomainExpertise",
    "create_analytics_selection_bridge",
]
