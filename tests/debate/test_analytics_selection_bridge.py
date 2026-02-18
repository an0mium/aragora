"""Tests for aragora.debate.analytics_selection_bridge — AnalyticsSelectionBridge."""

from __future__ import annotations

import pytest
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from aragora.debate.analytics_selection_bridge import (
    AnalyticsSelectionBridge,
    AnalyticsSelectionBridgeConfig,
    DomainExpertise,
    SelectionBoost,
    create_analytics_selection_bridge,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeAgentMetrics:
    """Minimal AgentMetrics stand-in."""

    agent_name: str = "claude"
    precision: float = 0.8
    agreement_rate: float = 0.7
    avg_response_time_ms: float = 3000.0
    total_findings: int = 50
    finding_distribution: dict[str, int] = field(default_factory=dict)


def _make_bridge_with_metrics(*metrics_list: FakeAgentMetrics) -> AnalyticsSelectionBridge:
    """Build a bridge and inject metrics directly into cache."""
    bridge = AnalyticsSelectionBridge()
    for m in metrics_list:
        bridge._metrics_cache[m.agent_name] = m
        bridge._compute_domain_expertise(m)
    bridge._cache_timestamp = datetime.now()
    return bridge


# ---------------------------------------------------------------------------
# SelectionBoost / DomainExpertise dataclasses
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_selection_boost_defaults(self):
        b = SelectionBoost(agent_name="claude", total_boost=0.5)
        assert b.precision_component == 0.0
        assert b.confidence == 0.0

    def test_domain_expertise_defaults(self):
        d = DomainExpertise(agent_name="claude", primary_domain="security")
        assert d.domain_scores == {}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_defaults(self):
        c = AnalyticsSelectionBridgeConfig()
        assert c.min_findings_for_boost == 10
        assert c.precision_weight == pytest.approx(0.35)
        assert c.agreement_weight == pytest.approx(0.25)
        assert c.response_time_weight == pytest.approx(0.15)
        assert c.domain_expertise_weight == pytest.approx(0.25)

    def test_custom(self):
        c = AnalyticsSelectionBridgeConfig(precision_weight=0.5, min_findings_for_boost=5)
        assert c.precision_weight == pytest.approx(0.5)
        assert c.min_findings_for_boost == 5


# ---------------------------------------------------------------------------
# _compute_domain_expertise
# ---------------------------------------------------------------------------


class TestComputeDomainExpertise:
    def test_empty_distribution(self):
        bridge = AnalyticsSelectionBridge()
        m = FakeAgentMetrics(finding_distribution={})
        bridge._compute_domain_expertise(m)
        assert "claude" not in bridge._domain_expertise_cache

    def test_single_domain(self):
        bridge = AnalyticsSelectionBridge()
        m = FakeAgentMetrics(finding_distribution={"security": 10})
        bridge._compute_domain_expertise(m)
        exp = bridge._domain_expertise_cache["claude"]
        assert exp.primary_domain == "security"
        assert exp.domain_scores["security"] == pytest.approx(1.0)

    def test_multiple_domains(self):
        bridge = AnalyticsSelectionBridge()
        m = FakeAgentMetrics(
            finding_distribution={"security": 6, "performance": 4}
        )
        bridge._compute_domain_expertise(m)
        exp = bridge._domain_expertise_cache["claude"]
        assert exp.primary_domain == "security"
        assert exp.domain_scores["security"] == pytest.approx(0.6)
        assert exp.domain_scores["performance"] == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# _is_cache_valid
# ---------------------------------------------------------------------------


class TestCacheValidity:
    def test_no_timestamp(self):
        bridge = AnalyticsSelectionBridge()
        assert bridge._is_cache_valid() is False

    def test_fresh_cache(self):
        bridge = AnalyticsSelectionBridge()
        bridge._cache_timestamp = datetime.now()
        assert bridge._is_cache_valid() is True

    def test_expired_cache(self):
        bridge = AnalyticsSelectionBridge(
            config=AnalyticsSelectionBridgeConfig(cache_ttl_seconds=10)
        )
        bridge._cache_timestamp = datetime.now() - timedelta(seconds=20)
        assert bridge._is_cache_valid() is False


# ---------------------------------------------------------------------------
# compute_selection_boost
# ---------------------------------------------------------------------------


class TestComputeSelectionBoost:
    def test_no_metrics_returns_neutral(self):
        bridge = AnalyticsSelectionBridge()
        boost = bridge.compute_selection_boost("unknown")
        assert boost.total_boost == 0.0
        assert boost.confidence == 0.0

    def test_insufficient_findings_returns_neutral(self):
        bridge = _make_bridge_with_metrics(
            FakeAgentMetrics(total_findings=3)  # below default 10
        )
        boost = bridge.compute_selection_boost("claude")
        assert boost.total_boost == 0.0

    def test_basic_boost(self):
        bridge = _make_bridge_with_metrics(
            FakeAgentMetrics(
                precision=0.9,
                agreement_rate=0.8,
                avg_response_time_ms=3000.0,
                total_findings=50,
            )
        )
        boost = bridge.compute_selection_boost("claude")
        assert boost.total_boost > 0
        assert boost.precision_component > 0
        assert boost.agreement_component > 0
        assert boost.response_time_component > 0

    def test_precision_component(self):
        bridge = _make_bridge_with_metrics(
            FakeAgentMetrics(precision=1.0, total_findings=50)
        )
        boost = bridge.compute_selection_boost("claude")
        assert boost.precision_component == pytest.approx(1.0 * 0.35)

    def test_response_time_fast_agent(self):
        bridge = _make_bridge_with_metrics(
            FakeAgentMetrics(avg_response_time_ms=1000.0, total_findings=50)
        )
        boost = bridge.compute_selection_boost("claude")
        # Faster than baseline → time_score capped at 1.0
        assert boost.response_time_component > 0

    def test_response_time_slow_agent(self):
        bridge = _make_bridge_with_metrics(
            FakeAgentMetrics(avg_response_time_ms=30000.0, total_findings=50)
        )
        boost = bridge.compute_selection_boost("claude")
        assert boost.response_time_component == pytest.approx(0.0, abs=0.01)

    def test_response_time_zero(self):
        bridge = _make_bridge_with_metrics(
            FakeAgentMetrics(avg_response_time_ms=0.0, total_findings=50)
        )
        boost = bridge.compute_selection_boost("claude")
        # No data → 0.5 neutral score
        assert boost.response_time_component > 0

    def test_time_sensitive_boosts_response_weight(self):
        m = FakeAgentMetrics(avg_response_time_ms=2000.0, total_findings=50)
        bridge = _make_bridge_with_metrics(m)
        normal = bridge.compute_selection_boost("claude", time_sensitive=False)
        sensitive = bridge.compute_selection_boost("claude", time_sensitive=True)
        assert sensitive.response_time_component > normal.response_time_component

    def test_domain_expertise_match(self):
        bridge = _make_bridge_with_metrics(
            FakeAgentMetrics(
                finding_distribution={"security": 8, "performance": 2},
                total_findings=50,
            )
        )
        boost = bridge.compute_selection_boost("claude", target_domain="security")
        assert boost.domain_expertise_component > 0

    def test_domain_expertise_no_match(self):
        bridge = _make_bridge_with_metrics(
            FakeAgentMetrics(
                finding_distribution={"security": 10},
                total_findings=50,
            )
        )
        boost = bridge.compute_selection_boost("claude", target_domain="healthcare")
        assert boost.domain_expertise_component == 0.0

    def test_domain_partial_match(self):
        bridge = _make_bridge_with_metrics(
            FakeAgentMetrics(
                finding_distribution={"security_audit": 10},
                total_findings=50,
            )
        )
        boost = bridge.compute_selection_boost("claude", target_domain="security")
        # "security" is in "security_audit" → partial match
        assert boost.domain_expertise_component > 0

    def test_confidence_scales_with_findings(self):
        bridge_low = _make_bridge_with_metrics(
            FakeAgentMetrics(agent_name="low", total_findings=20)
        )
        bridge_high = _make_bridge_with_metrics(
            FakeAgentMetrics(agent_name="high", total_findings=100)
        )
        assert bridge_low.compute_selection_boost("low").confidence == pytest.approx(0.2)
        assert bridge_high.compute_selection_boost("high").confidence == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# get_all_selection_boosts
# ---------------------------------------------------------------------------


class TestGetAllBoosts:
    def test_returns_all_agents(self):
        bridge = _make_bridge_with_metrics(
            FakeAgentMetrics(agent_name="a", total_findings=50),
            FakeAgentMetrics(agent_name="b", total_findings=50),
        )
        all_boosts = bridge.get_all_selection_boosts()
        assert "a" in all_boosts
        assert "b" in all_boosts

    def test_with_domain(self):
        bridge = _make_bridge_with_metrics(
            FakeAgentMetrics(
                agent_name="a",
                total_findings=50,
                finding_distribution={"security": 10},
            )
        )
        all_boosts = bridge.get_all_selection_boosts(target_domain="security")
        assert all_boosts["a"].domain_expertise_component > 0


# ---------------------------------------------------------------------------
# rank_agents_by_domain
# ---------------------------------------------------------------------------


class TestRankByDomain:
    def test_ranks_by_expertise(self):
        bridge = _make_bridge_with_metrics(
            FakeAgentMetrics(
                agent_name="expert",
                total_findings=50,
                finding_distribution={"security": 10},
            ),
            FakeAgentMetrics(
                agent_name="novice",
                total_findings=50,
                finding_distribution={"performance": 10},
            ),
        )
        ranked = bridge.rank_agents_by_domain("security")
        assert ranked[0] == "expert"

    def test_empty_no_expertise(self):
        bridge = AnalyticsSelectionBridge()
        assert bridge.rank_agents_by_domain("security") == []


# ---------------------------------------------------------------------------
# get_precision_leaders / get_fastest_agents
# ---------------------------------------------------------------------------


class TestLeadersAndFastest:
    def test_precision_leaders(self):
        bridge = _make_bridge_with_metrics(
            FakeAgentMetrics(agent_name="a", precision=0.9, total_findings=50),
            FakeAgentMetrics(agent_name="b", precision=0.7, total_findings=50),
            FakeAgentMetrics(agent_name="c", precision=0.95, total_findings=50),
        )
        leaders = bridge.get_precision_leaders(top_n=2)
        assert leaders[0] == "c"
        assert leaders[1] == "a"
        assert len(leaders) == 2

    def test_fastest_agents(self):
        bridge = _make_bridge_with_metrics(
            FakeAgentMetrics(agent_name="fast", avg_response_time_ms=1000.0, total_findings=50),
            FakeAgentMetrics(agent_name="slow", avg_response_time_ms=10000.0, total_findings=50),
            FakeAgentMetrics(agent_name="zero", avg_response_time_ms=0.0, total_findings=50),
        )
        fastest = bridge.get_fastest_agents(top_n=2)
        assert fastest[0] == "fast"
        assert len(fastest) == 2
        assert "zero" not in fastest  # filtered out (0 ms)


# ---------------------------------------------------------------------------
# get_metrics / get_domain_expertise / get_stats
# ---------------------------------------------------------------------------


class TestAccessors:
    def test_get_metrics(self):
        m = FakeAgentMetrics()
        bridge = _make_bridge_with_metrics(m)
        assert bridge.get_metrics("claude") is m
        assert bridge.get_metrics("unknown") is None

    def test_get_domain_expertise(self):
        bridge = _make_bridge_with_metrics(
            FakeAgentMetrics(finding_distribution={"security": 10})
        )
        exp = bridge.get_domain_expertise("claude")
        assert exp is not None
        assert exp.primary_domain == "security"
        assert bridge.get_domain_expertise("unknown") is None

    def test_get_stats(self):
        bridge = _make_bridge_with_metrics(
            FakeAgentMetrics(finding_distribution={"security": 10})
        )
        stats = bridge.get_stats()
        assert stats["agents_tracked"] == 1
        assert stats["cache_valid"] is True
        assert stats["agents_with_expertise"] == 1


# ---------------------------------------------------------------------------
# refresh_metrics (async)
# ---------------------------------------------------------------------------


class TestRefreshMetrics:
    def test_no_dashboard(self):
        import asyncio

        bridge = AnalyticsSelectionBridge(analytics_dashboard=None)
        count = asyncio.run(bridge.refresh_metrics())
        assert count == 0

    def test_with_dashboard(self):
        import asyncio
        from unittest.mock import AsyncMock

        m = FakeAgentMetrics(
            agent_name="claude",
            finding_distribution={"security": 5, "perf": 3},
        )
        dashboard = AsyncMock()
        dashboard.get_agent_metrics = AsyncMock(return_value=[m])

        bridge = AnalyticsSelectionBridge(analytics_dashboard=dashboard)
        count = asyncio.run(bridge.refresh_metrics())
        assert count == 1
        assert bridge.get_metrics("claude") is m
        assert bridge._cache_timestamp is not None

    def test_dashboard_error(self):
        import asyncio
        from unittest.mock import AsyncMock

        dashboard = AsyncMock()
        dashboard.get_agent_metrics = AsyncMock(side_effect=RuntimeError("fail"))

        bridge = AnalyticsSelectionBridge(analytics_dashboard=dashboard)
        count = asyncio.run(bridge.refresh_metrics())
        assert count == 0


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestFactory:
    def test_create_default(self):
        bridge = create_analytics_selection_bridge()
        assert isinstance(bridge, AnalyticsSelectionBridge)
        assert bridge.analytics_dashboard is None

    def test_create_with_kwargs(self):
        bridge = create_analytics_selection_bridge(precision_weight=0.5)
        assert bridge.config.precision_weight == pytest.approx(0.5)
