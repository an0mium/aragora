"""Tests for AnalyticsSelectionBridge."""

from __future__ import annotations

import pytest
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from unittest.mock import MagicMock, AsyncMock

from aragora.debate.analytics_selection_bridge import (
    AnalyticsSelectionBridge,
    AnalyticsSelectionBridgeConfig,
    SelectionBoost,
    DomainExpertise,
    create_analytics_selection_bridge,
)


@dataclass
class MockAgentMetrics:
    """Mock agent metrics for testing."""

    agent_name: str = "test-agent"
    precision: float = 0.85
    agreement_rate: float = 0.75
    avg_response_time_ms: float = 3000.0
    total_findings: int = 50
    finding_distribution: Dict[str, int] = field(default_factory=dict)


class MockAnalyticsDashboard:
    """Mock analytics dashboard."""

    def __init__(self):
        self._metrics: List[MockAgentMetrics] = []

    async def get_agent_metrics(self, workspace_id: Optional[str] = None) -> List[MockAgentMetrics]:
        """Get agent metrics."""
        return self._metrics

    def add_agent_metrics(self, metrics: MockAgentMetrics) -> None:
        """Add metrics for an agent."""
        self._metrics.append(metrics)


class TestSelectionBoost:
    """Tests for SelectionBoost dataclass."""

    def test_defaults(self):
        """Test default values."""
        boost = SelectionBoost(agent_name="test", total_boost=0.5)
        assert boost.agent_name == "test"
        assert boost.total_boost == 0.5
        assert boost.precision_component == 0.0
        assert boost.confidence == 0.0

    def test_with_components(self):
        """Test with all components."""
        boost = SelectionBoost(
            agent_name="claude",
            total_boost=0.8,
            precision_component=0.3,
            agreement_component=0.2,
            response_time_component=0.15,
            domain_expertise_component=0.15,
            confidence=0.9,
        )
        assert boost.precision_component == 0.3
        assert boost.confidence == 0.9


class TestDomainExpertise:
    """Tests for DomainExpertise dataclass."""

    def test_creation(self):
        """Test DomainExpertise creation."""
        expertise = DomainExpertise(
            agent_name="claude",
            primary_domain="security",
            domain_scores={"security": 0.6, "performance": 0.3},
        )
        assert expertise.primary_domain == "security"
        assert expertise.domain_scores["security"] == 0.6


class TestAnalyticsSelectionBridge:
    """Tests for AnalyticsSelectionBridge."""

    def test_create_bridge(self):
        """Test bridge creation."""
        bridge = AnalyticsSelectionBridge()
        assert bridge.analytics_dashboard is None
        assert bridge.config is not None

    def test_create_with_config(self):
        """Test bridge creation with custom config."""
        config = AnalyticsSelectionBridgeConfig(
            min_findings_for_boost=5,
            precision_weight=0.5,
        )
        bridge = AnalyticsSelectionBridge(config=config)
        assert bridge.config.min_findings_for_boost == 5
        assert bridge.config.precision_weight == 0.5

    @pytest.mark.asyncio
    async def test_refresh_metrics_no_dashboard(self):
        """Test refresh with no dashboard."""
        bridge = AnalyticsSelectionBridge()
        count = await bridge.refresh_metrics()
        assert count == 0

    @pytest.mark.asyncio
    async def test_refresh_metrics(self):
        """Test refreshing metrics from dashboard."""
        dashboard = MockAnalyticsDashboard()
        dashboard.add_agent_metrics(
            MockAgentMetrics(
                agent_name="claude",
                precision=0.9,
                total_findings=50,
                finding_distribution={"security": 30, "performance": 20},
            )
        )

        bridge = AnalyticsSelectionBridge(analytics_dashboard=dashboard)
        count = await bridge.refresh_metrics()

        assert count == 1
        assert "claude" in bridge._metrics_cache

    def test_compute_selection_boost_no_data(self):
        """Test selection boost with no data."""
        bridge = AnalyticsSelectionBridge()
        boost = bridge.compute_selection_boost("unknown")

        assert boost.total_boost == 0.0
        assert boost.confidence == 0.0

    def test_compute_selection_boost_insufficient_findings(self):
        """Test selection boost with insufficient findings."""
        bridge = AnalyticsSelectionBridge(
            config=AnalyticsSelectionBridgeConfig(min_findings_for_boost=100)
        )
        bridge._metrics_cache["claude"] = MockAgentMetrics(agent_name="claude", total_findings=50)

        boost = bridge.compute_selection_boost("claude")
        assert boost.total_boost == 0.0

    def test_compute_selection_boost(self):
        """Test computing selection boost."""
        bridge = AnalyticsSelectionBridge(
            config=AnalyticsSelectionBridgeConfig(min_findings_for_boost=10)
        )
        bridge._metrics_cache["claude"] = MockAgentMetrics(
            agent_name="claude",
            precision=0.9,
            agreement_rate=0.8,
            avg_response_time_ms=2000.0,
            total_findings=50,
        )

        boost = bridge.compute_selection_boost("claude")

        assert boost.total_boost > 0
        assert boost.precision_component > 0
        assert boost.agreement_component > 0
        assert boost.confidence > 0

    def test_compute_selection_boost_time_sensitive(self):
        """Test selection boost with time sensitivity."""
        bridge = AnalyticsSelectionBridge(
            config=AnalyticsSelectionBridgeConfig(min_findings_for_boost=10)
        )

        # Fast agent
        bridge._metrics_cache["fast"] = MockAgentMetrics(
            agent_name="fast",
            precision=0.8,
            avg_response_time_ms=1000.0,
            total_findings=50,
        )

        # Slow agent
        bridge._metrics_cache["slow"] = MockAgentMetrics(
            agent_name="slow",
            precision=0.85,
            avg_response_time_ms=20000.0,
            total_findings=50,
        )

        fast_boost = bridge.compute_selection_boost("fast", time_sensitive=True)
        slow_boost = bridge.compute_selection_boost("slow", time_sensitive=True)

        assert fast_boost.response_time_component > slow_boost.response_time_component

    def test_compute_selection_boost_with_domain(self):
        """Test selection boost with domain expertise."""
        bridge = AnalyticsSelectionBridge(
            config=AnalyticsSelectionBridgeConfig(min_findings_for_boost=10)
        )
        bridge._metrics_cache["claude"] = MockAgentMetrics(
            agent_name="claude",
            precision=0.9,
            total_findings=50,
        )
        bridge._domain_expertise_cache["claude"] = DomainExpertise(
            agent_name="claude",
            primary_domain="security",
            domain_scores={"security": 0.6, "performance": 0.3},
        )

        boost = bridge.compute_selection_boost("claude", target_domain="security")
        assert boost.domain_expertise_component > 0

    def test_get_all_selection_boosts(self):
        """Test getting boosts for all agents."""
        bridge = AnalyticsSelectionBridge(
            config=AnalyticsSelectionBridgeConfig(min_findings_for_boost=10)
        )
        bridge._metrics_cache["claude"] = MockAgentMetrics(agent_name="claude", total_findings=50)
        bridge._metrics_cache["gpt-4"] = MockAgentMetrics(agent_name="gpt-4", total_findings=30)

        boosts = bridge.get_all_selection_boosts()
        assert "claude" in boosts
        assert "gpt-4" in boosts

    def test_rank_agents_by_domain(self):
        """Test ranking agents by domain expertise."""
        bridge = AnalyticsSelectionBridge()
        bridge._domain_expertise_cache["claude"] = DomainExpertise(
            agent_name="claude",
            primary_domain="security",
            domain_scores={"security": 0.7},
        )
        bridge._domain_expertise_cache["gpt-4"] = DomainExpertise(
            agent_name="gpt-4",
            primary_domain="performance",
            domain_scores={"security": 0.3, "performance": 0.6},
        )

        ranking = bridge.rank_agents_by_domain("security")
        assert ranking[0] == "claude"  # Higher security score

    def test_get_precision_leaders(self):
        """Test getting precision leaders."""
        bridge = AnalyticsSelectionBridge()
        bridge._metrics_cache["claude"] = MockAgentMetrics(agent_name="claude", precision=0.95)
        bridge._metrics_cache["gpt-4"] = MockAgentMetrics(agent_name="gpt-4", precision=0.85)
        bridge._metrics_cache["gemini"] = MockAgentMetrics(agent_name="gemini", precision=0.90)

        leaders = bridge.get_precision_leaders(top_n=2)
        assert leaders[0] == "claude"
        assert len(leaders) == 2

    def test_get_fastest_agents(self):
        """Test getting fastest agents."""
        bridge = AnalyticsSelectionBridge()
        bridge._metrics_cache["fast"] = MockAgentMetrics(
            agent_name="fast", avg_response_time_ms=1000.0
        )
        bridge._metrics_cache["slow"] = MockAgentMetrics(
            agent_name="slow", avg_response_time_ms=5000.0
        )

        fastest = bridge.get_fastest_agents(top_n=1)
        assert fastest[0] == "fast"

    def test_get_metrics(self):
        """Test getting metrics for an agent."""
        bridge = AnalyticsSelectionBridge()
        bridge._metrics_cache["claude"] = MockAgentMetrics(agent_name="claude")

        metrics = bridge.get_metrics("claude")
        assert metrics is not None
        assert metrics.agent_name == "claude"

        assert bridge.get_metrics("unknown") is None

    def test_get_domain_expertise(self):
        """Test getting domain expertise."""
        bridge = AnalyticsSelectionBridge()
        bridge._domain_expertise_cache["claude"] = DomainExpertise(
            agent_name="claude", primary_domain="security"
        )

        expertise = bridge.get_domain_expertise("claude")
        assert expertise is not None
        assert expertise.primary_domain == "security"

    def test_cache_validity(self):
        """Test cache validity checking."""
        bridge = AnalyticsSelectionBridge(
            config=AnalyticsSelectionBridgeConfig(cache_ttl_seconds=300)
        )

        # No cache timestamp
        assert not bridge._is_cache_valid()

        # Set cache timestamp
        bridge._cache_timestamp = datetime.now()
        assert bridge._is_cache_valid()

    def test_get_stats(self):
        """Test getting bridge stats."""
        bridge = AnalyticsSelectionBridge()
        bridge._metrics_cache["claude"] = MockAgentMetrics(agent_name="claude")
        bridge._domain_expertise_cache["claude"] = DomainExpertise(
            agent_name="claude", primary_domain="security"
        )

        stats = bridge.get_stats()
        assert stats["agents_tracked"] == 1
        assert stats["agents_with_expertise"] == 1

    def test_factory_function(self):
        """Test factory function."""
        dashboard = MockAnalyticsDashboard()
        bridge = create_analytics_selection_bridge(
            analytics_dashboard=dashboard,
            min_findings_for_boost=5,
        )
        assert bridge.analytics_dashboard is dashboard
        assert bridge.config.min_findings_for_boost == 5
