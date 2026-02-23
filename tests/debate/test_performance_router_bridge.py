"""Tests for performance router bridge.

Covers AgentRoutingScore, SyncResult, PerformanceRouterBridgeConfig,
PerformanceRouterBridge (compute_routing_score, sync_to_router,
get_best_agent_for_task, rank_agents_for_task, get_speed_tier),
and create_performance_router_bridge factory.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from aragora.debate.performance_router_bridge import (
    AgentRoutingScore,
    PerformanceRouterBridge,
    PerformanceRouterBridgeConfig,
    SyncResult,
    create_performance_router_bridge,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_metrics(
    avg_response_time=2.0,
    quality_score=0.7,
    consistency_score=0.8,
    total_calls=10,
    response_time_variance=None,
):
    m = MagicMock()
    m.avg_response_time = avg_response_time
    m.quality_score = quality_score
    m.consistency_score = consistency_score
    m.total_calls = total_calls
    if response_time_variance is not None:
        m.response_time_variance = response_time_variance
    return m


def make_monitor(agent_metrics: dict):
    """Create a mock performance monitor."""
    monitor = MagicMock()

    def get_agent_metrics(name):
        return agent_metrics.get(name)

    monitor.get_agent_metrics = get_agent_metrics
    return monitor


# ---------------------------------------------------------------------------
# AgentRoutingScore
# ---------------------------------------------------------------------------


class TestAgentRoutingScore:
    def test_defaults(self):
        s = AgentRoutingScore(agent_name="claude")
        assert s.overall_score == 0.0
        assert s.latency_score == 0.0
        assert s.quality_score == 0.0
        assert s.consistency_score == 0.0
        assert s.data_points == 0
        assert s.last_updated is None

    def test_is_reliable_below_threshold(self):
        s = AgentRoutingScore(agent_name="claude", data_points=3)
        assert s.is_reliable is False

    def test_is_reliable_at_threshold(self):
        s = AgentRoutingScore(agent_name="claude", data_points=5)
        assert s.is_reliable is True


# ---------------------------------------------------------------------------
# SyncResult
# ---------------------------------------------------------------------------


class TestSyncResult:
    def test_defaults(self):
        r = SyncResult()
        assert r.agents_synced == 0
        assert r.agents_skipped == 0
        assert r.success is True
        assert r.error is None
        assert isinstance(r.timestamp, datetime)


# ---------------------------------------------------------------------------
# PerformanceRouterBridgeConfig
# ---------------------------------------------------------------------------


class TestPerformanceRouterBridgeConfig:
    def test_defaults(self):
        c = PerformanceRouterBridgeConfig()
        assert c.min_data_points == 5
        assert c.latency_weight == 0.3
        assert c.quality_weight == 0.4
        assert c.consistency_weight == 0.3
        assert c.auto_sync is True
        assert c.sync_interval_seconds == 60
        assert c.fast_latency_threshold == 1.0
        assert c.slow_latency_threshold == 10.0


# ---------------------------------------------------------------------------
# PerformanceRouterBridge — compute_routing_score
# ---------------------------------------------------------------------------


class TestComputeRoutingScore:
    def test_no_monitor_returns_empty_score(self):
        bridge = PerformanceRouterBridge()
        score = bridge.compute_routing_score("claude")
        assert score.overall_score == 0.0
        assert score.data_points == 0

    def test_no_metrics_for_agent(self):
        monitor = make_monitor({})
        bridge = PerformanceRouterBridge(performance_monitor=monitor)
        score = bridge.compute_routing_score("claude")
        assert score.overall_score == 0.0

    def test_fast_agent_high_latency_score(self):
        metrics = make_metrics(avg_response_time=0.5, total_calls=10)
        monitor = make_monitor({"claude": metrics})
        bridge = PerformanceRouterBridge(performance_monitor=monitor)
        score = bridge.compute_routing_score("claude")
        assert score.latency_score == 1.0

    def test_slow_agent_low_latency_score(self):
        metrics = make_metrics(avg_response_time=15.0, total_calls=10)
        monitor = make_monitor({"claude": metrics})
        bridge = PerformanceRouterBridge(performance_monitor=monitor)
        score = bridge.compute_routing_score("claude")
        assert score.latency_score == 0.0

    def test_medium_latency_interpolated(self):
        metrics = make_metrics(avg_response_time=5.5, total_calls=10)
        monitor = make_monitor({"claude": metrics})
        bridge = PerformanceRouterBridge(performance_monitor=monitor)
        score = bridge.compute_routing_score("claude")
        assert 0.0 < score.latency_score < 1.0

    def test_quality_score_clamped(self):
        metrics = make_metrics(quality_score=1.5, total_calls=10)
        monitor = make_monitor({"claude": metrics})
        bridge = PerformanceRouterBridge(performance_monitor=monitor)
        score = bridge.compute_routing_score("claude")
        assert score.quality_score == 1.0

    def test_overall_score_weighted(self):
        metrics = make_metrics(
            avg_response_time=0.5, quality_score=1.0, consistency_score=1.0, total_calls=10
        )
        monitor = make_monitor({"claude": metrics})
        bridge = PerformanceRouterBridge(performance_monitor=monitor)
        score = bridge.compute_routing_score("claude")
        # All perfect scores → overall should be 1.0
        assert abs(score.overall_score - 1.0) < 0.01

    def test_speed_task_type_weights(self):
        metrics = make_metrics(
            avg_response_time=0.5, quality_score=0.5, consistency_score=0.5, total_calls=10
        )
        monitor = make_monitor({"claude": metrics})
        bridge = PerformanceRouterBridge(performance_monitor=monitor)
        speed_score = bridge.compute_routing_score("claude", task_type="speed")
        precision_score = bridge.compute_routing_score("claude", task_type="precision")
        # Fast agent should score higher for speed tasks
        assert speed_score.overall_score > precision_score.overall_score

    def test_score_is_cached(self):
        metrics = make_metrics(total_calls=10)
        monitor = make_monitor({"claude": metrics})
        bridge = PerformanceRouterBridge(performance_monitor=monitor)
        bridge.compute_routing_score("claude")
        assert "claude" in bridge._routing_scores

    def test_consistency_from_variance(self):
        metrics = MagicMock()
        metrics.avg_response_time = 2.0
        metrics.quality_score = 0.7
        del metrics.consistency_score
        metrics.response_time_variance = 1.0
        metrics.total_calls = 10
        monitor = make_monitor({"claude": metrics})
        bridge = PerformanceRouterBridge(performance_monitor=monitor)
        score = bridge.compute_routing_score("claude")
        assert score.consistency_score == pytest.approx(0.8, abs=0.01)


# ---------------------------------------------------------------------------
# sync_to_router
# ---------------------------------------------------------------------------


class TestSyncToRouter:
    def test_no_router_returns_error(self):
        bridge = PerformanceRouterBridge()
        result = bridge.sync_to_router()
        assert result.success is False
        assert "No agent router" in result.error

    def test_syncs_reliable_agents(self):
        router = MagicMock()
        router.set_agent_weight = MagicMock()
        bridge = PerformanceRouterBridge(agent_router=router)
        bridge._routing_scores["claude"] = AgentRoutingScore(
            agent_name="claude", overall_score=0.8, data_points=10
        )
        result = bridge.sync_to_router(force=True)
        assert result.agents_synced == 1
        router.set_agent_weight.assert_called_once_with("claude", 0.8)

    def test_skips_unreliable_agents(self):
        router = MagicMock()
        router.set_agent_weight = MagicMock()
        bridge = PerformanceRouterBridge(agent_router=router)
        bridge._routing_scores["claude"] = AgentRoutingScore(
            agent_name="claude", overall_score=0.8, data_points=2
        )
        result = bridge.sync_to_router(force=True)
        assert result.agents_skipped == 1
        assert result.agents_synced == 0

    def test_respects_sync_interval(self):
        router = MagicMock()
        router.set_agent_weight = MagicMock()
        bridge = PerformanceRouterBridge(
            agent_router=router,
            config=PerformanceRouterBridgeConfig(sync_interval_seconds=3600),
        )
        bridge._last_sync = datetime.now()
        bridge._routing_scores["claude"] = AgentRoutingScore(
            agent_name="claude", overall_score=0.8, data_points=10
        )
        result = bridge.sync_to_router()
        # Should skip due to interval
        assert result.agents_synced == 0

    def test_force_overrides_interval(self):
        router = MagicMock()
        router.set_agent_weight = MagicMock()
        bridge = PerformanceRouterBridge(agent_router=router)
        bridge._last_sync = datetime.now()
        bridge._routing_scores["claude"] = AgentRoutingScore(
            agent_name="claude", overall_score=0.8, data_points=10
        )
        result = bridge.sync_to_router(force=True)
        assert result.agents_synced == 1

    def test_sync_history_tracked(self):
        router = MagicMock()
        router.set_agent_weight = MagicMock()
        bridge = PerformanceRouterBridge(agent_router=router)
        bridge.sync_to_router(force=True)
        assert len(bridge.get_sync_history()) == 1

    def test_uses_update_weight_fallback(self):
        router = MagicMock(spec=["update_weight"])
        bridge = PerformanceRouterBridge(agent_router=router)
        bridge._routing_scores["claude"] = AgentRoutingScore(
            agent_name="claude", overall_score=0.8, data_points=10
        )
        result = bridge.sync_to_router(force=True)
        assert result.agents_synced == 1
        router.update_weight.assert_called_once()


# ---------------------------------------------------------------------------
# get_best_agent_for_task / rank_agents_for_task
# ---------------------------------------------------------------------------


class TestRoutingQueries:
    def test_best_agent_empty(self):
        bridge = PerformanceRouterBridge()
        assert bridge.get_best_agent_for_task([]) is None

    def test_best_agent_no_reliable_data(self):
        monitor = make_monitor({})
        bridge = PerformanceRouterBridge(performance_monitor=monitor)
        result = bridge.get_best_agent_for_task(["claude", "gpt"])
        assert result is None

    def test_best_agent_with_data(self):
        monitor = make_monitor(
            {
                "claude": make_metrics(avg_response_time=0.5, quality_score=0.9, total_calls=10),
                "gpt": make_metrics(avg_response_time=5.0, quality_score=0.5, total_calls=10),
            }
        )
        bridge = PerformanceRouterBridge(performance_monitor=monitor)
        best = bridge.get_best_agent_for_task(["claude", "gpt"])
        assert best == "claude"

    def test_rank_agents(self):
        monitor = make_monitor(
            {
                "claude": make_metrics(avg_response_time=0.5, quality_score=0.9, total_calls=10),
                "gpt": make_metrics(avg_response_time=5.0, quality_score=0.5, total_calls=10),
            }
        )
        bridge = PerformanceRouterBridge(performance_monitor=monitor)
        rankings = bridge.rank_agents_for_task(["claude", "gpt"])
        assert rankings[0][0] == "claude"
        assert rankings[0][1] > rankings[1][1]


# ---------------------------------------------------------------------------
# get_speed_tier
# ---------------------------------------------------------------------------


class TestSpeedTier:
    def test_fast_tier(self):
        monitor = make_monitor(
            {
                "claude": make_metrics(avg_response_time=0.5, total_calls=10),
            }
        )
        bridge = PerformanceRouterBridge(performance_monitor=monitor)
        bridge.compute_routing_score("claude")
        assert bridge.get_speed_tier("claude") == "fast"

    def test_slow_tier(self):
        monitor = make_monitor(
            {
                "claude": make_metrics(avg_response_time=15.0, total_calls=10),
            }
        )
        bridge = PerformanceRouterBridge(performance_monitor=monitor)
        bridge.compute_routing_score("claude")
        assert bridge.get_speed_tier("claude") == "slow"

    def test_medium_tier(self):
        monitor = make_monitor(
            {
                "claude": make_metrics(avg_response_time=5.0, total_calls=10),
            }
        )
        bridge = PerformanceRouterBridge(performance_monitor=monitor)
        bridge.compute_routing_score("claude")
        assert bridge.get_speed_tier("claude") == "medium"


# ---------------------------------------------------------------------------
# Utility methods
# ---------------------------------------------------------------------------


class TestUtilities:
    def test_get_routing_score_cached(self):
        bridge = PerformanceRouterBridge()
        assert bridge.get_routing_score("claude") is None
        bridge._routing_scores["claude"] = AgentRoutingScore(agent_name="claude")
        assert bridge.get_routing_score("claude") is not None

    def test_get_all_scores(self):
        bridge = PerformanceRouterBridge()
        bridge._routing_scores["a"] = AgentRoutingScore(agent_name="a")
        bridge._routing_scores["b"] = AgentRoutingScore(agent_name="b")
        assert len(bridge.get_all_scores()) == 2

    def test_clear_cache(self):
        bridge = PerformanceRouterBridge()
        bridge._routing_scores["a"] = AgentRoutingScore(agent_name="a")
        bridge.clear_cache()
        assert len(bridge._routing_scores) == 0

    def test_get_stats(self):
        bridge = PerformanceRouterBridge()
        stats = bridge.get_stats()
        assert stats["agents_tracked"] == 0
        assert stats["reliable_agents"] == 0
        assert stats["performance_monitor_attached"] is False
        assert stats["agent_router_attached"] is False


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestFactory:
    def test_create_default(self):
        bridge = create_performance_router_bridge()
        assert isinstance(bridge, PerformanceRouterBridge)
        assert bridge.performance_monitor is None

    def test_create_with_config(self):
        monitor = MagicMock()
        bridge = create_performance_router_bridge(
            performance_monitor=monitor,
            latency_weight=0.5,
            quality_weight=0.3,
            consistency_weight=0.2,
        )
        assert bridge.performance_monitor is monitor
        assert bridge.config.latency_weight == 0.5
